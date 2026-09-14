"""MOSS-Audio-Tokenizer (CAT) decoder — minimal reimplementation."""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["MossCodecDecoder", "MossCodecStreamer", "delayed_rows_to_segments"]

_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}

def _read_safetensors_file(path: str) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader used only until moss_tts_lite."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        blob = f.read()
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
        from moss_tts_lite.st_loader import read_safetensors    # type: ignore
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
    """Effective weight of nn."""
    g = orig0.float().squeeze(-1).squeeze(-1)
    v = orig1.float().squeeze(-1)
    norm = v.norm(dim=1, keepdim=True)
    return (g.unsqueeze(1) * v / norm).unsqueeze(-1)

class _RLFQDequant:
    """Residual-LFQ dequantization (the quantizer."""

    def __init__(self, weights: dict[str, torch.Tensor], device: torch.device):
        nq = 0
        while f"quantizer.quantizers.{nq}.codebook.weight" in weights:
            nq += 1
        if nq == 0:
            raise KeyError("no quantizer.quantizers.*.codebook.weight in checkpoint")
        self.nq = nq
        self.codebooks = torch.stack(
            [weights[f"quantizer.quantizers.{i}.codebook.weight"].float() for i in range(nq)]
        ).to(device)
        self.out_w = torch.stack(
            [_denorm_wn_conv(
                weights[f"quantizer.quantizers.{i}.out_proj.parametrizations.weight.original0"],
                weights[f"quantizer.quantizers.{i}.out_proj.parametrizations.weight.original1"],
            ) for i in range(nq)]
        ).to(device)
        self.out_b = torch.stack(
            [weights[f"quantizer.quantizers.{i}.out_proj.bias"].float() for i in range(nq)]
        ).to(device)
        self.proj_w = _denorm_wn_conv(
            weights["quantizer.output_proj.parametrizations.weight.original0"],
            weights["quantizer.output_proj.parametrizations.weight.original1"],
        ).to(device)
        self.proj_b = weights["quantizer.output_proj.bias"].float().to(device)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes:"""
        nq, B, T = codes.shape
        if nq != self.nq:
            raise ValueError(f"expected {self.nq} quantizers, got {nq}")
        emb = torch.zeros(B, self.out_b.shape[1], T, device=codes.device, dtype=torch.float32)
        for i in range(nq):
            z = F.embedding(codes[i], self.codebooks[i])
            z = F.conv1d(z.transpose(1, 2), self.out_w[i], self.out_b[i])
            emb += z
        return F.conv1d(emb, self.proj_w, self.proj_b)

def _rope_cache(T: int, head_dim: int, offset: int, device: torch.device):
    """cos/sin tables [T, head_dim//2] for absolute positions offset."""
    ds = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(10000.0) * 2.0 / head_dim))
    ts = offset + torch.arange(T, device=device, dtype=torch.float32)
    ang = freqs.view(1, -1) * ts.view(-1, 1)
    return ang.cos(), ang.sin()

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x:"""
    B, H, T, Dh = x.shape
    xr = x.float().view(B, H, T, Dh // 2, 2)
    xr_r, xr_i = xr[..., 0], xr[..., 1]
    c = cos.view(1, 1, T, -1)
    s = sin.view(1, 1, T, -1)
    out_r = xr_r * c - xr_i * s
    out_i = xr_r * s + xr_i * c
    return torch.stack([out_r, out_i], dim=-1).view(B, H, T, Dh)

def _patch_decode(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, d*h, L) -> (B, d, L*h):"""
    b, dh, l = x.shape
    h = patch_size
    return x.reshape(b, dh // h, h, l).permute(0, 1, 3, 2).reshape(b, dh // h, l * h)

class _StreamCache:
    """Ring KV-cache + rope offset for one attention module (batch=1)."""

    __slots__ = ("kbuf", "vbuf", "pos", "end", "offset")

    def __init__(self, num_heads: int, head_dim: int, capacity: int, device: torch.device):
        self.kbuf = torch.zeros(1, num_heads, capacity, head_dim, device=device, dtype=torch.float32)
        self.vbuf = torch.zeros(1, num_heads, capacity, head_dim, device=device, dtype=torch.float32)
        self.pos = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.end = 0
        self.offset = 0

    def reset(self) -> None:

        self.pos.fill_(-1)
        self.end = 0
        self.offset = 0

class _TransformerStage:
    """One ProjectedTransformer with per-attention streaming state."""

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
        self.context = context

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

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """x:"""
        h = x.transpose(1, 2)
        if self.in_w is not None:
            h = h @ self.in_w.t()
        B, T, C = h.shape
        H, Dh = self.num_heads, self.head_dim

        pos = torch.arange(T, device=h.device)
        delta = pos.view(-1, 1) - pos.view(1, -1)
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

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """x:"""
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

            idx = (cache.end + torch.arange(T, device=h.device)) % self.context
            cache.kbuf[0][:, idx] = k[0]
            cache.vbuf[0][:, idx] = qkv[2][0]
            cache.pos[idx] = cache.end + torch.arange(T, device=h.device)
            cache.end += T

            q_pos = cache.offset + torch.arange(T, device=h.device)
            delta = q_pos.view(-1, 1) - cache.pos.view(1, -1)
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

_DECODER_TRANSFORMERS = [

    (768, 1280, 1280, 32, 20),
    (640, 768, 768, 12, 12),
    (384, 768, 768, 12, 12),
    (384, 240, 768, 12, 12),
]
_DECODER_PATCH_AFTER = [2, 2, 2, 240]

class MossCodecDecoder:
    """Minimal fp32 decoder for MOSS-Audio-Tokenizer (24 kHz mono)."""

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

        frame_rate = self.sampling_rate / self.downsample_rate
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
            frame_rate *= 2

        self.patch_sizes = _DECODER_PATCH_AFTER

    @torch.no_grad()
    def _run_stages(self, d: torch.Tensor, chunked: bool) -> torch.Tensor:
        for si, stage in enumerate(self.stages):
            d = stage.forward_chunk(d) if chunked else stage.forward_full(d)
            d = _patch_decode(d, self.patch_sizes[si])
        return d

    @torch.no_grad()
    def decode_full(self, codes: torch.Tensor) -> np.ndarray:
        """Full-sequence decode (reference:"""
        codes = self._prepare_codes(codes)
        d = self.quantizer.decode_codes(codes.unsqueeze(1))
        wav = self._run_stages(d, chunked=False)
        return wav[0, 0].cpu().numpy()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, chunk_duration: float | None = 8.0) -> np.ndarray:
        """Decode codes [T, 32] into float32 mono 24 kHz numpy waveform."""
        codes = self._prepare_codes(codes)
        T = codes.shape[1]

        if chunk_duration is None:
            return self.decode_full(codes.transpose(0, 1))

        chunk_length = int(round(chunk_duration * self.sampling_rate))
        if chunk_length <= 0:
            raise ValueError("chunk_duration too small")
        if chunk_length % self.downsample_rate != 0:
            raise ValueError("chunk_duration * sampling_rate must be divisible by downsample_rate")
        chunk_frames = chunk_length // self.downsample_rate

        if T <= chunk_frames:

            return self.decode_full(codes.transpose(0, 1))

        for stage in self.stages:
            stage.reset_stream()
        wavs: list[torch.Tensor] = []
        for start in range(0, T, chunk_frames):
            n = min(chunk_frames, T - start)
            d = self.quantizer.decode_codes(codes[:, start:start + n].unsqueeze(1))
            d = self._run_stages(d, chunked=True)
            wavs.append(d[0, 0])
        return torch.cat(wavs).cpu().numpy()

    def _prepare_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Accept [T, 32] (contract), [32, T] or [32, 1, T] -> [NQ, T] long."""
        if codes.dim() == 3:
            if codes.shape[1] != 1:
                raise ValueError(f"batch decode not supported, got {tuple(codes.shape)}")
            codes = codes[:, 0]
        if codes.dim() == 2:
            if codes.shape[1] == self.quantizer.nq:
                codes = codes.transpose(0, 1)
            elif codes.shape[0] != self.quantizer.nq:
                raise ValueError(f"expected codes [T, 32], got {tuple(codes.shape)}")
        if codes.dim() != 2 or codes.shape[0] != self.quantizer.nq:
            raise ValueError(f"expected codes [T, 32], got {tuple(codes.shape)}")
        return codes.to(device=self.device, dtype=torch.long)

class MossCodecStreamer:
    """Stateful incremental decoder aligned with reference streaming semantics."""

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
        """Feed [n, 32] code frames;"""
        codes = self.decoder._prepare_codes(codes)
        self._buf.append(codes)
        self._buf_frames += codes.shape[1]
        if self._buf_frames < self.chunk_frames:
            return None
        block = torch.cat(self._buf, dim=1)[:, : self.chunk_frames]

        rest = self._buf_frames - self.chunk_frames
        tail = torch.cat(self._buf, dim=1)[:, self.chunk_frames:]
        self._buf = [tail] if rest else []
        self._buf_frames = rest
        d = self.decoder.quantizer.decode_codes(block.unsqueeze(1))
        return self.decoder._run_stages(d, chunked=True)[0, 0].cpu().numpy()

    @torch.no_grad()
    def flush(self) -> np.ndarray:
        """Finish the stream:"""
        if self._buf_frames == 0:
            return np.zeros(0, dtype=np.float32)
        n = self._buf_frames
        block = torch.cat(self._buf, dim=1)
        self._buf, self._buf_frames = [], 0
        d = self.decoder.quantizer.decode_codes(block.unsqueeze(1))
        wav = self.decoder._run_stages(d, chunked=True)[0, 0]
        return wav.cpu().numpy()

    def close(self) -> None:
        self._reset()

def delayed_rows_to_segments(
    delayed: torch.Tensor, audio_pad_code: int = 1024
) -> list[torch.Tensor]:
    """Split raw TTS generation audio rows into decode-ready code segments."""
    if delayed.dim() != 2:
        raise ValueError(f"expected delayed rows [T, 32], got {tuple(delayed.shape)}")
    T_rows, n_vq = delayed.shape
    if T_rows < n_vq:
        raise ValueError(
            f"delayed rows too short to de-delay: T={T_rows} < n_vq={n_vq}"
        )

    out_len = T_rows - n_vq + 1
    codes = torch.empty(out_len, n_vq, device=delayed.device, dtype=delayed.dtype)
    for i in range(n_vq):
        codes[:, i] = delayed[i : i + out_len, i]

    non_pad = ~(codes == audio_pad_code).all(dim=1)
    if not bool(non_pad.any()):
        return []

    idx = torch.nonzero(non_pad).squeeze(1)
    breaks = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    if breaks.numel() == 0:
        segments_idx = [idx]
    else:

        b = breaks.tolist()
        sizes = [b[0]] + [b[i] - b[i - 1] for i in range(1, len(b))] + [idx.numel() - b[-1]]
        segments_idx = torch.split(idx, sizes)
    return [codes[s] for s in segments_idx]
