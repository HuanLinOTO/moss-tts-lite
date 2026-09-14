"""MOSS-Audio-Tokenizer (CAT) decoder — minimal reimplementation.

Decode-only port of `modeling_moss_audio_tokenizer.py` (MossAudioTokenizerModel):
    codes[Tq=32, B, T] --RLFQ dequant--> (B, 768, T) --decoder stages--> wav (B, 1, T*1920)

Pipeline (fp32, 24 kHz mono, downsample rate 1920 -> 12.5 code frames / second):

    RLFQ dequant   sum_i( WNConv1d_8->512(codebook_i[codes_i]) ), WNConv1d_512->768
    decoder.0      ProjectedTransformer  768->1280, d=1280, 32L, 20H, ff=5120, ctx=125
    decoder.1      PatchedPretransform(2)   (B,1280,T)   -> (B,640,2T)
    decoder.2      ProjectedTransformer  640->768,  d=768, 12L, 12H, ff=3072, ctx=250
    decoder.3      PatchedPretransform(2)   (B,768,2T)   -> (B,384,4T)
    decoder.4      ProjectedTransformer  384->768,  d=768, 12L, 12H, ff=3072, ctx=500
    decoder.5      PatchedPretransform(2)   (B,768,4T)   -> (B,384,8T)
    decoder.6      ProjectedTransformer  384->240,  d=768, 12L, 12H, ff=3072, ctx=1000
    decoder.7      PatchedPretransform(240) (B,240,8T)   -> (B,1,1920T)

Attention is causal with a sliding window of `context` frames (10 s at each
stage's own frame rate 12.5/25/50/100 Hz -> 125/250/500/1000), replicated from
`MossAudioTokenizerMultiheadAttention.forward`:
    attn_bias[i, j] = (pos_j >= 0) & (0 <= pos_i - pos_j < context)

Two decode paths, both faithful to the reference:
  * full-sequence:  single pass, [T, T] position mask (states: none)
  * chunked (`chunk_duration` s, default 8 -> 100 code frames / chunk):
    streaming ring KV-cache with capacity == context per attention, rope offset
    accumulating across chunks — replicates `model.decode(..., chunk_duration=8)`.
    NOTE: the ring cache evicts keys that the context window would still allow,
    so chunked != full decode for sequences longer than one chunk; parity is
    defined against the reference chunked path.

Only torch / numpy / stdlib (moss_tts_lite dependency rule).
"""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["MossCodecDecoder", "MossCodecStreamer", "delayed_rows_to_segments"]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _read_safetensors_file(path: str) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader used only until moss_tts_lite.st_loader lands."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        blob = f.read()  # single read; mmap through bytes is fine for one-shot use
    base = 8 + header_len
    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        t = torch.frombuffer(blob, dtype=_DTYPES[info["dtype"]], offset=base + s,
                             count=int(np.prod(info["shape"])) if info["shape"] else 1)
        tensors[name] = t.reshape(tuple(info["shape"]))
    return tensors


def _load_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """Load single-file or sharded safetensors checkpoint, original dtype."""
    try:
        from moss_tts_lite.st_loader import read_safetensors  # type: ignore
        return read_safetensors(model_dir, dtype=None)
    except ImportError:
        pass
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)
        tensors: dict[str, torch.Tensor] = {}
        for shard, names in shards.items():
            data = _read_safetensors_file(os.path.join(model_dir, shard))
            tensors.update({name: data[name] for name in names})
        return tensors
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return _read_safetensors_file(single)
    raise FileNotFoundError(f"no safetensors found in {model_dir}")


def _denorm_wn_conv(orig0: torch.Tensor, orig1: torch.Tensor) -> torch.Tensor:
    """Effective weight of nn.utils.parametrizations.weight_norm (dim=0).

    w[o, i] = g[o] * v[o, i] / ||v[o, :]||_2   (g = original0, v = original1)
    """
    g = orig0.float().squeeze(-1).squeeze(-1)          # [out]
    v = orig1.float().squeeze(-1)                      # [out, in]
    norm = v.norm(dim=1, keepdim=True)                 # [out, 1]
    return (g.unsqueeze(1) * v / norm).unsqueeze(-1)   # [out, in, 1]


# ---------------------------------------------------------------------------
# RLFQ dequantization
# ---------------------------------------------------------------------------

class _RLFQDequant:
    """Residual-LFQ dequantization (the quantizer.decode_codes path).

    per quantizer i:  z_i = WNConv1d_{8->512}( codebook_i[codes_i] )   (bias)
    zq = WNConv1d_{512->768}( sum_i z_i )                              (bias)
    Encode-side projections (quantizers' in_proj, quantizer.input_proj) unused.
    """

    def __init__(self, weights: dict[str, torch.Tensor], device: torch.device):
        nq = 0
        while f"quantizer.quantizers.{nq}.codebook.weight" in weights:
            nq += 1
        if nq == 0:
            raise KeyError("no quantizer.quantizers.*.codebook.weight in checkpoint")
        self.nq = nq
        self.codebooks = torch.stack(
            [weights[f"quantizer.quantizers.{i}.codebook.weight"].float() for i in range(nq)]
        ).to(device)                                            # [NQ, 1024, 8]
        self.out_w = torch.stack(
            [_denorm_wn_conv(
                weights[f"quantizer.quantizers.{i}.out_proj.parametrizations.weight.original0"],
                weights[f"quantizer.quantizers.{i}.out_proj.parametrizations.weight.original1"],
            ) for i in range(nq)]
        ).to(device)                                            # [NQ, 512, 8, 1]
        self.out_b = torch.stack(
            [weights[f"quantizer.quantizers.{i}.out_proj.bias"].float() for i in range(nq)]
        ).to(device)                                            # [NQ, 512]
        self.proj_w = _denorm_wn_conv(
            weights["quantizer.output_proj.parametrizations.weight.original0"],
            weights["quantizer.output_proj.parametrizations.weight.original1"],
        ).to(device)                                            # [768, 512, 1]
        self.proj_b = weights["quantizer.output_proj.bias"].float().to(device)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [NQ, B, T] long -> (B, 768, T) float32."""
        nq, B, T = codes.shape
        if nq != self.nq:
            raise ValueError(f"expected {self.nq} quantizers, got {nq}")
        emb = torch.zeros(B, self.out_b.shape[1], T, device=codes.device, dtype=torch.float32)
        for i in range(nq):
            z = F.embedding(codes[i], self.codebooks[i])        # [B, T, 8]
            z = F.conv1d(z.transpose(1, 2), self.out_w[i], self.out_b[i])  # [B, 512, T]
            emb += z
        return F.conv1d(emb, self.proj_w, self.proj_b)          # [B, 768, T]


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _rope_cache(T: int, head_dim: int, offset: int, device: torch.device):
    """cos/sin tables [T, head_dim//2] for absolute positions offset..offset+T-1.

    Matches apply_rope(max_period=10000): freqs = exp(-ln(max_period)*2*i/Dh),
    computed in float32.
    """
    ds = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(10000.0) * 2.0 / head_dim))
    ts = offset + torch.arange(T, device=device, dtype=torch.float32)
    ang = freqs.view(1, -1) * ts.view(-1, 1)
    return ang.cos(), ang.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, Dh], interleaved complex pairs -> roped x (float32)."""
    B, H, T, Dh = x.shape
    xr = x.float().view(B, H, T, Dh // 2, 2)
    xr_r, xr_i = xr[..., 0], xr[..., 1]
    c = cos.view(1, 1, T, -1)
    s = sin.view(1, 1, T, -1)
    out_r = xr_r * c - xr_i * s
    out_i = xr_r * s + xr_i * c
    return torch.stack([out_r, out_i], dim=-1).view(B, H, T, Dh)


# ---------------------------------------------------------------------------
# PatchedPretransform
# ---------------------------------------------------------------------------

def _patch_decode(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, d*h, L) -> (B, d, L*h): fold channel patches into time."""
    b, dh, l = x.shape
    h = patch_size
    return x.reshape(b, dh // h, h, l).permute(0, 1, 3, 2).reshape(b, dh // h, l * h)


# ---------------------------------------------------------------------------
# Transformer stage (full + streaming)
# ---------------------------------------------------------------------------

class _StreamCache:
    """Ring KV-cache + rope offset for one attention module (batch=1).

    Mirrors RingKVCache (capacity == context, respect_exec_mask=True): keys of
    the last `capacity` frames, slot positions, -1 = never written.
    """

    __slots__ = ("kbuf", "vbuf", "pos", "end", "offset")

    def __init__(self, num_heads: int, head_dim: int, capacity: int, device: torch.device):
        self.kbuf = torch.zeros(1, num_heads, capacity, head_dim, device=device, dtype=torch.float32)
        self.vbuf = torch.zeros(1, num_heads, capacity, head_dim, device=device, dtype=torch.float32)
        self.pos = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.end = 0     # total frames written (ring write pointer = end % capacity)
        self.offset = 0  # rope offset for the next chunk

    def reset(self) -> None:
        # reference only resets end_offset/offset; unwritten slots are masked
        # via positions == -1, so leaving stale kbuf contents is safe.
        self.pos.fill_(-1)
        self.end = 0
        self.offset = 0


class _TransformerStage:
    """One ProjectedTransformer with per-attention streaming state.

    Layer (gating == "none"):
        h = h + ls1 * attn(LN1(h));  h = h + ls2 * linear2(gelu(linear1(LN2(h))))
    """

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        prefix: str,
        input_dim: int,
        output_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        context: int,
        device: torch.device,
    ):
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.context = context  # == ring capacity in streaming mode

        self.in_w = (weights[f"{prefix}.input_proj.weight"].float().to(device)
                     if d_model != input_dim else None)
        self.out_w = (weights[f"{prefix}.output_proj.weight"].float().to(device)
                      if d_model != output_dim else None)

        self.qkv_w, self.o_w = [], []
        self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b = [], [], [], []
        self.ff1_w, self.ff2_w, self.ls1, self.ls2 = [], [], [], []
        for i in range(num_layers):
            lp = f"{prefix}.transformer.layers.{i}"
            self.qkv_w.append(weights[f"{lp}.self_attn.in_projs.0.weight"].float().to(device))
            self.o_w.append(weights[f"{lp}.self_attn.out_projs.0.weight"].float().to(device))
            self.ln1_w.append(weights[f"{lp}.norm1.weight"].float().to(device))
            self.ln1_b.append(weights[f"{lp}.norm1.bias"].float().to(device))
            self.ln2_w.append(weights[f"{lp}.norm2.weight"].float().to(device))
            self.ln2_b.append(weights[f"{lp}.norm2.bias"].float().to(device))
            self.ff1_w.append(weights[f"{lp}.linear1.weight"].float().to(device))
            self.ff2_w.append(weights[f"{lp}.linear2.weight"].float().to(device))
            self.ls1.append(weights[f"{lp}.layer_scale_1.scale"].float().to(device))
            self.ls2.append(weights[f"{lp}.layer_scale_2.scale"].float().to(device))

        self.caches = [
            _StreamCache(num_heads, self.head_dim, context, device) for _ in range(num_layers)
        ]

    def reset_stream(self) -> None:
        for c in self.caches:
            c.reset()

    # -- full-sequence -------------------------------------------------

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D_in, T] -> [B, D_out, T]."""
        h = x.transpose(1, 2)
        if self.in_w is not None:
            h = h @ self.in_w.t()
        B, T, C = h.shape
        H, Dh = self.num_heads, self.head_dim

        pos = torch.arange(T, device=h.device)
        delta = pos.view(-1, 1) - pos.view(1, -1)                 # [T, T] = i - j
        mask = (delta >= 0) & (delta < self.context)
        mask = mask.view(1, 1, T, T)
        cos, sin = _rope_cache(T, Dh, 0, h.device)

        for i in range(len(self.qkv_w)):
            n = F.layer_norm(h, (C,), self.ln1_w[i], self.ln1_b[i], 1e-5)
            qkv = (n @ self.qkv_w[i].t()).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
            q = _apply_rope(qkv[0], cos, sin)
            k = _apply_rope(qkv[1], cos, sin)
            o = F.scaled_dot_product_attention(q, k, qkv[2], mask, dropout_p=0.0)
            o = o.transpose(1, 2).reshape(B, T, C) @ self.o_w[i].t()
            h = h + self.ls1[i] * o
            n = F.layer_norm(h, (C,), self.ln2_w[i], self.ln2_b[i], 1e-5)
            o = F.linear(F.gelu(n @ self.ff1_w[i].t()), self.ff2_w[i])
            h = h + self.ls2[i] * o
        if self.out_w is not None:
            h = h @ self.out_w.t()
        return h.transpose(1, 2)

    # -- streaming chunk -------------------------------------------------

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """x: [1, D_in, Tc] -> [1, D_out, Tc]; updates per-attention ring caches."""
        h = x.transpose(1, 2)
        if self.in_w is not None:
            h = h @ self.in_w.t()
        B, T, C = h.shape
        if B != 1:
            raise ValueError("streaming decode supports batch_size=1 only")
        H, Dh = self.num_heads, self.head_dim

        for i in range(len(self.qkv_w)):
            n = F.layer_norm(h, (C,), self.ln1_w[i], self.ln1_b[i], 1e-5)
            qkv = (n @ self.qkv_w[i].t()).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
            cache = self.caches[i]
            cos, sin = _rope_cache(T, Dh, cache.offset, h.device)
            q = _apply_rope(qkv[0], cos, sin)
            k = _apply_rope(qkv[1], cos, sin)

            # write roped k/v into the ring (scatter at (end + t) % capacity)
            idx = (cache.end + torch.arange(T, device=h.device)) % self.context
            cache.kbuf[0][:, idx] = k[0]
            cache.vbuf[0][:, idx] = qkv[2][0]
            cache.pos[idx] = cache.end + torch.arange(T, device=h.device)
            cache.end += T

            # attention over the full ring with position-based causal+window mask
            q_pos = cache.offset + torch.arange(T, device=h.device)          # [T]
            delta = q_pos.view(-1, 1) - cache.pos.view(1, -1)                # [T, cap]
            mask = (cache.pos.view(1, -1) >= 0) & (delta >= 0) & (delta < self.context)
            mask = mask.view(1, 1, T, self.context)
            o = F.scaled_dot_product_attention(q, cache.kbuf, cache.vbuf, mask, dropout_p=0.0)
            o = o.transpose(1, 2).reshape(B, T, C) @ self.o_w[i].t()
            h = h + self.ls1[i] * o

            n = F.layer_norm(h, (C,), self.ln2_w[i], self.ln2_b[i], 1e-5)
            o = F.linear(F.gelu(n @ self.ff1_w[i].t()), self.ff2_w[i])
            h = h + self.ls2[i] * o

            cache.offset += T
        if self.out_w is not None:
            h = h @ self.out_w.t()
        return h.transpose(1, 2)


# ---------------------------------------------------------------------------
# Public decoder
# ---------------------------------------------------------------------------

# decoder_kwargs execution order (config.json): 4 transformers interleaved with
# patches; context computed per stage from its own frame rate (12.5/25/50/100 Hz).
_DECODER_TRANSFORMERS = [
    # (input_dim, output_dim, d_model, num_layers, num_heads)
    (768, 1280, 1280, 32, 20),
    (640, 768, 768, 12, 12),
    (384, 768, 768, 12, 12),
    (384, 240, 768, 12, 12),
]
_DECODER_PATCH_AFTER = [2, 2, 2, 240]  # patch after transformer i (last folds 240ch -> wav)


class MossCodecDecoder:
    """Minimal fp32 decoder for MOSS-Audio-Tokenizer (24 kHz mono).

    Contract (COORD.md):
        MossCodecDecoder(model_dir, device='cuda', dtype=torch.float32)
        .decode(codes: LongTensor[T, 32]) -> np.ndarray float32 mono 24 kHz

    Default decode path mirrors the reference pipeline call
    `model.decode(audio_codes, padding_mask, chunk_duration=8)` exactly:
    100-code-frame chunks, streaming ring KV-cache (capacity == context).
    """

    def __init__(self, model_dir: str, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float32):
        if dtype != torch.float32:
            raise ValueError("MossCodecDecoder supports float32 only (exact-parity decode)")
        self.device = torch.device(device)
        self.sampling_rate = 24000
        self.downsample_rate = 1920

        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        if config["sampling_rate"] != self.sampling_rate or config["downsample_rate"] != self.downsample_rate:
            raise ValueError("unsupported codec config (expect 24 kHz / 1920 downsample)")
        qk = config["quantizer_kwargs"]
        if qk.get("quantizer_type", config.get("quantizer_type")) != "rlfq":
            raise ValueError("only quantizer_type='rlfq' is supported")
        context_duration = float(config["causal_transformer_context_duration"])

        weights = _load_weights(model_dir)
        self.quantizer = _RLFQDequant(weights, self.device)

        frame_rate = self.sampling_rate / self.downsample_rate  # 12.5 Hz after quantizer
        self.stages: list[_TransformerStage] = []
        for si, (in_dim, out_dim, d_model, num_layers, num_heads) in enumerate(_DECODER_TRANSFORMERS):
            self.stages.append(_TransformerStage(
                weights,
                prefix=f"decoder.{si * 2}",
                input_dim=in_dim,
                output_dim=out_dim,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                context=int(frame_rate * context_duration),
                device=self.device,
            ))
            frame_rate *= 2  # PatchedPretransform(2) between stages

        self.patch_sizes = _DECODER_PATCH_AFTER

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_stages(self, d: torch.Tensor, chunked: bool) -> torch.Tensor:
        for si, stage in enumerate(self.stages):
            d = stage.forward_chunk(d) if chunked else stage.forward_full(d)
            d = _patch_decode(d, self.patch_sizes[si])
        return d

    @torch.no_grad()
    def decode_full(self, codes: torch.Tensor) -> np.ndarray:
        """Full-sequence decode (reference: _decode_frame, no streaming state)."""
        codes = self._prepare_codes(codes)                       # [NQ, T]
        d = self.quantizer.decode_codes(codes.unsqueeze(1))      # [1, 768, T]
        wav = self._run_stages(d, chunked=False)                 # [1, 1, T*1920]
        return wav[0, 0].cpu().numpy()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, chunk_duration: float | None = 8.0) -> np.ndarray:
        """Decode codes [T, 32] into float32 mono 24 kHz numpy waveform.

        chunk_duration=None  -> full-sequence decode.
        chunk_duration=8.0   -> reference pipeline semantics: 100-frame chunks
        with streaming ring KV-caches (== model.decode(..., chunk_duration=8)).
        """
        codes = self._prepare_codes(codes)                       # [NQ, T]
        T = codes.shape[1]

        if chunk_duration is None:
            return self.decode_full(codes.transpose(0, 1))

        chunk_length = int(round(chunk_duration * self.sampling_rate))
        if chunk_length <= 0:
            raise ValueError("chunk_duration too small")
        if chunk_length % self.downsample_rate != 0:
            raise ValueError("chunk_duration * sampling_rate must be divisible by downsample_rate")
        chunk_frames = chunk_length // self.downsample_rate      # 100 for 8 s

        if T <= chunk_frames:
            # reference: single _decode_frame, no streaming state
            return self.decode_full(codes.transpose(0, 1))

        for stage in self.stages:
            stage.reset_stream()
        wavs: list[torch.Tensor] = []
        for start in range(0, T, chunk_frames):
            n = min(chunk_frames, T - start)
            d = self.quantizer.decode_codes(codes[:, start:start + n].unsqueeze(1))
            d = self._run_stages(d, chunked=True)                # [1, 1, n*1920]
            wavs.append(d[0, 0])
        return torch.cat(wavs).cpu().numpy()

    # ------------------------------------------------------------------
    def _prepare_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Accept [T, 32] (contract), [32, T] or [32, 1, T] -> [NQ, T] long."""
        if codes.dim() == 3:
            if codes.shape[1] != 1:
                raise ValueError(f"batch decode not supported, got {tuple(codes.shape)}")
            codes = codes[:, 0]                                  # [NQ, 1, T] -> [NQ, T]
        if codes.dim() == 2:
            if codes.shape[1] == self.quantizer.nq:
                codes = codes.transpose(0, 1)                    # [T, 32] -> [32, T]
            elif codes.shape[0] != self.quantizer.nq:
                raise ValueError(f"expected codes [T, 32], got {tuple(codes.shape)}")
        if codes.dim() != 2 or codes.shape[0] != self.quantizer.nq:
            raise ValueError(f"expected codes [T, 32], got {tuple(codes.shape)}")
        return codes.to(device=self.device, dtype=torch.long)


class MossCodecStreamer:
    """Stateful incremental decoder aligned with reference streaming semantics.

    Design: ring KV-caches make the audio
    for a code frame depend on the write schedule (write-count eviction), so a
    faithful incremental stream must feed the engine in fixed `chunk_frames`
    blocks. `push()` buffers incoming frames and emits wav exactly when a full
    block is available, reproducing `decoder.decode(codes, chunk_duration=8)`
    bitwise (verified in tests/test_audio.py::test_streamer_matches_decode).

    Usage with generation interleave:
        s = MossCodecStreamer(decoder)
        for frame_batch in gen_frames:            # LongTensor [n, 32]
            wav = s.push(frame_batch)             # np.ndarray or None
            if wav is not None: play/enqueue(wav)
        wav = s.flush()                           # tail (padded to a full block)
    """

    def __init__(self, decoder: MossCodecDecoder, chunk_duration: float = 8.0):
        chunk_length = int(round(chunk_duration * decoder.sampling_rate))
        if chunk_length <= 0 or chunk_length % decoder.downsample_rate != 0:
            raise ValueError("chunk_duration * sampling_rate must be a positive multiple of downsample_rate")
        self.decoder = decoder
        self.chunk_frames = chunk_length // decoder.downsample_rate
        self._buf: list[torch.Tensor] = []
        self._buf_frames = 0
        self._reset()

    def _reset(self) -> None:
        for stage in self.decoder.stages:
            stage.reset_stream()
        self._buf = []
        self._buf_frames = 0

    @torch.no_grad()
    def push(self, codes: torch.Tensor) -> np.ndarray | None:
        """Feed [n, 32] code frames; return wav samples when a block completes."""
        codes = self.decoder._prepare_codes(codes)           # [NQ, n]
        self._buf.append(codes)
        self._buf_frames += codes.shape[1]
        if self._buf_frames < self.chunk_frames:
            return None
        block = torch.cat(self._buf, dim=1)[:, : self.chunk_frames]
        # keep the remainder for the next block
        rest = self._buf_frames - self.chunk_frames
        tail = torch.cat(self._buf, dim=1)[:, self.chunk_frames:]
        self._buf = [tail] if rest else []
        self._buf_frames = rest
        d = self.decoder.quantizer.decode_codes(block.unsqueeze(1))
        return self.decoder._run_stages(d, chunked=True)[0, 0].cpu().numpy()

    @torch.no_grad()
    def flush(self) -> np.ndarray:
        """Finish the stream: decode the buffered tail (< chunk_frames) as-is.

        No zero-padding: padding would write extra frames into the ring
        KV-caches and evict keys that real queries can still legally attend
        to (write-count eviction), changing the output vs decode().
        """
        if self._buf_frames == 0:
            return np.zeros(0, dtype=np.float32)
        n = self._buf_frames
        block = torch.cat(self._buf, dim=1)                      # [NQ, n]
        self._buf, self._buf_frames = [], 0
        d = self.decoder.quantizer.decode_codes(block.unsqueeze(1))
        wav = self.decoder._run_stages(d, chunked=True)[0, 0]    # n*1920 samples
        return wav.cpu().numpy()

    def close(self) -> None:
        self._reset()


# ---------------------------------------------------------------------------
# TTS generation rows -> decode-ready code segments
# ---------------------------------------------------------------------------

def delayed_rows_to_segments(
    delayed: torch.Tensor, audio_pad_code: int = 1024
) -> list[torch.Tensor]:
    """Split raw TTS generation audio rows into decode-ready code segments.

    Replicates the codes-level part of `MossTTSProcessor._parse_audio_codes`
    (processing_moss_tts.py), which the reference pipeline runs on
    `generation_ids[:, 1:]` before handing codes to the codec:
      1. `apply_de_delay_pattern`: [T, n_vq] -> [T - n_vq + 1, n_vq]
         (audio channel i starts at delayed row i; ramp-up/down corners excluded)
      2. drop rows that are entirely `audio_pad_code` (1024) — real generation
         emits those between audio segments (separators)
      3. split the remaining rows into contiguous runs, one tensor per segment

    Each returned segment [T_i, 32] feeds `MossCodecDecoder.decode` directly
    (or `.t()` for the [32, T] tensor layout). `audio_pad_code` defaults to
    the fixed MOSS-TTS-v1.5 value (coord: audio_pad_code=1024).
    """
    if delayed.dim() != 2:
        raise ValueError(f"expected delayed rows [T, 32], got {tuple(delayed.shape)}")
    T_rows, n_vq = delayed.shape
    if T_rows < n_vq:
        raise ValueError(
            f"delayed rows too short to de-delay: T={T_rows} < n_vq={n_vq}"
        )

    # 1) de-delay (apply_de_delay_pattern): channel i takes delayed[i : i + out, i]
    out_len = T_rows - n_vq + 1
    codes = torch.empty(out_len, n_vq, device=delayed.device, dtype=delayed.dtype)
    for i in range(n_vq):
        codes[:, i] = delayed[i : i + out_len, i]

    # 2) rows that are all pad are separators between real audio segments
    non_pad = ~(codes == audio_pad_code).all(dim=1)
    if not bool(non_pad.any()):
        return []

    # 3) contiguous runs of non-pad rows
    idx = torch.nonzero(non_pad).squeeze(1)
    breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    if breaks.numel() == 0:
        segments_idx = [idx]
    else:
        # split sizes at the gaps. NOTE: the reference passes the break
        # positions themselves to torch.split, which raises for >= 2 segments
        # (split sizes must sum to len(idx)); we implement the intended
        # contiguous-run split. For a single segment (the golden case) the
        # outputs are identical.
        b = breaks.tolist()
        sizes = [b[0]] + [b[i] - b[i - 1] for i in range(1, len(b))] + [idx.numel() - b[-1]]
        segments_idx = torch.split(idx, sizes)
    return [codes[s] for s in segments_idx]
