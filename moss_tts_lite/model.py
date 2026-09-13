"""MOSS-TTS-v1.5 minimal forward: Qwen3-8B backbone + 32 audio codebook
embeddings + 33 output heads, static preallocated KV cache.

Checkpoint keys keep their original names (language_model.*, emb_ext.N.weight,
lm_heads.N.weight).  Only torch is used.

Numerics follow transformers' Qwen3 modeling code (verified against
transformers 5.0 source; see .tmp/reports/):
- RMSNorm: fp32 stats, cast back to input dtype, then scale by weight;
- QK-norm: per-head RMSNorm over head_dim applied to q/k projections before rope;
- rope (theta 1e6) applied to q/k only; cos/sin computed in fp32, cast to model dtype;
- attention via torch SDPA, GQA 32q:8kv expanded with repeat_kv semantics,
  scale = head_dim ** -0.5, is_causal for the prefill block (no padding, batch=1);
- SwiGLU MLP; no biases anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---- MOSS-TTS-v1.5 constants (models/MOSS-TTS-v1.5/config.json) ----
N_VQ = 32
AUDIO_VOCAB = 1025          # embedding/logit table size; real codes 0..1023,
                            # index 1024 == audio_pad_code (masked -inf at output)
AUDIO_PAD_CODE = 1024
TEXT_VOCAB = 155648

PAD_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
AUDIO_START_TOKEN_ID = 151652
AUDIO_END_TOKEN_ID = 151653
AUDIO_USER_SLOT_TOKEN_ID = 151654
AUDIO_GEN_SLOT_TOKEN_ID = 151656   # <|video_pad|>
AUDIO_DELAY_SLOT_TOKEN_ID = 151662  # <|fim_pad|>

ROPE_THETA = 1_000_000.0
RMS_EPS = 1e-6


@dataclass
class HiddenState:
    """Backbone output: post-final-norm hidden states [1, L, hidden]."""

    last_hidden: torch.Tensor


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = RMS_EPS) -> torch.Tensor:
    """transformers Qwen3RMSNorm, op-for-op (fp32 stats, cast back, then scale)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MossTTSModel:
    """MossTTSDelayModel minus the HF scaffolding.

    input_ids are [1, L, 1+n_vq]: channel 0 is the text vocab, channels 1..32
    are audio codebook indices in [0, 1024].
    """

    def __init__(self, weights: dict, device="cuda", dtype=torch.bfloat16, max_seq_len=8192):
        dev = torch.device(device)
        self.device = dev
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        layer_ids = sorted({int(k.split(".")[2]) for k in weights
                            if k.startswith("language_model.layers.")})
        self.n_layers = len(layer_ids)
        if layer_ids != list(range(self.n_layers)):
            raise ValueError(f"unexpected layer ids {layer_ids}")

        emb = weights["language_model.embed_tokens.weight"]
        self.hidden_size = int(emb.shape[1])
        self.text_vocab = int(emb.shape[0])
        self.head_dim = int(weights["language_model.layers.0.self_attn.q_norm.weight"].shape[0])
        self.n_heads = int(weights["language_model.layers.0.self_attn.q_proj.weight"].shape[0]) // self.head_dim
        self.n_kv_heads = int(weights["language_model.layers.0.self_attn.k_proj.weight"].shape[0]) // self.head_dim
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.n_rep = self.n_heads // self.n_kv_heads

        def T(name: str) -> torch.Tensor:
            t = weights[name]
            if t.device != dev or t.dtype != dtype:
                t = t.to(device=dev, dtype=dtype)
            return t.contiguous()

        self.embed_tokens = T("language_model.embed_tokens.weight")
        self.final_norm = T("language_model.norm.weight")
        self.layers = []
        for i in layer_ids:
            p = f"language_model.layers.{i}"
            self.layers.append({
                "input_layernorm": T(f"{p}.input_layernorm.weight"),
                "q": T(f"{p}.self_attn.q_proj.weight"),
                "k": T(f"{p}.self_attn.k_proj.weight"),
                "v": T(f"{p}.self_attn.v_proj.weight"),
                "o": T(f"{p}.self_attn.o_proj.weight"),
                "q_norm": T(f"{p}.self_attn.q_norm.weight"),
                "k_norm": T(f"{p}.self_attn.k_norm.weight"),
                "post_attention_layernorm": T(f"{p}.post_attention_layernorm.weight"),
                "gate": T(f"{p}.mlp.gate_proj.weight"),
                "up": T(f"{p}.mlp.up_proj.weight"),
                "down": T(f"{p}.mlp.down_proj.weight"),
            })

        self.emb_ext = [T(f"emb_ext.{i}.weight") for i in range(N_VQ)]
        self.head_text = T("lm_heads.0.weight")  # [text_vocab, hidden]
        # audio heads stored channel-major [32, 1025, hidden]; each slice is a
        # contiguous per-head weight -> per-head GEMMs are bitwise-identical to
        # the reference. (A single fused [32*1025, hidden] GEMM selects a
        # different cuBLAS kernel whose bf16 rounding differs by 1-2 ulp and
        # flips near-tied audio argmaxes, breaking golden parity.)
        self.head_audio = torch.stack(
            [weights[f"lm_heads.{i + 1}.weight"] for i in range(N_VQ)], dim=0
        ).to(device=dev, dtype=dtype).contiguous()  # [32, 1025, hidden]
        # 2-way text head rows: [gen_slot, delay_slot]
        self.head_2way = self.head_text[
            [AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_DELAY_SLOT_TOKEN_ID]
        ].contiguous()  # [2, hidden]

        # rope tables (computed fp32 like HF, cast to model dtype)
        self.rope_cos, self.rope_sin = self._rope_tables(max_seq_len, dev, dtype,
                                                         self.head_dim)

        # static KV cache: [n_layers, n_kv, max_seq, head_dim] for K and V
        self.k_cache = torch.zeros(self.n_layers, self.n_kv_heads, max_seq_len,
                                   self.head_dim, dtype=dtype, device=dev)
        self.v_cache = torch.zeros_like(self.k_cache)
        self._seq = 0

    @staticmethod
    def _rope_tables(n: int, dev: torch.device, dtype: torch.dtype,
                     head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """rope cos/sin for `n` positions (the original fp32 chain, verbatim).

        Every entry depends on its own position only, so `table[:m]` for m <= n
        is bitwise-identical to a table built with n == m -- which is what lets
        `ensure_seq_len` resize without moving a single decoded value.
        """
        inv = 1.0 / (ROPE_THETA ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim))
        pos = torch.arange(n, device=dev, dtype=torch.float32)
        freqs = torch.outer(pos, inv)
        emb2 = torch.cat((freqs, freqs), dim=-1)
        return emb2.cos().to(dtype), emb2.sin().to(dtype)  # [n, head_dim] each

    def ensure_seq_len(self, n: int) -> int:
        """Grow the KV cache + rope tables so `n` positions fit; returns the size.

        Only growth is possible (tensors can never shrink without reallocating
        1.1 GiB at max_seq_len, and the caller owns the budget), and only the
        tail of the cache is zeroed: slots below the current sequence are the
        live KV state of an in-flight utterance and must not be touched.

        Value-preserving by construction: an unchanged `n` is a no-op, and a
        grown rope table reproduces the old table element-for-element.
        """
        n = int(n)
        if n <= self.max_seq_len:
            return self.max_seq_len
        if self._seq > self.max_seq_len:
            raise RuntimeError(f"live sequence {self._seq} exceeds cache "
                               f"{self.max_seq_len}; cannot grow")
        old = self.max_seq_len
        dev, dtype = self.device, self.dtype
        ck = torch.zeros(self.n_layers, self.n_kv_heads, n, self.head_dim,
                         dtype=dtype, device=dev)
        cv = torch.zeros_like(ck)
        ck[:, :, :old].copy_(self.k_cache)
        cv[:, :, :old].copy_(self.v_cache)
        cos, sin = self._rope_tables(n, dev, dtype, self.head_dim)
        cos[:old].copy_(self.rope_cos)
        sin[:old].copy_(self.rope_sin)
        self.k_cache, self.v_cache = ck, cv
        self.rope_cos, self.rope_sin = cos, sin
        self.max_seq_len = n
        return n

    # ------------------------------------------------------------------ utils
    @property
    def seq_len(self) -> int:
        return self._seq

    def reset(self) -> None:
        self._seq = 0

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Combined embedding: base text embed + 32 audio embeds, sequential bf16
        adds in channel order (matches MossTTSDelayModel.get_input_embeddings)."""
        h = F.embedding(ids[..., 0], self.embed_tokens)
        for i in range(N_VQ):
            h = h + F.embedding(ids[..., i + 1], self.emb_ext[i])
        return h

    def _layer(self, li: int, h: torch.Tensor, s0: int) -> torch.Tensor:
        """One decoder layer; h [1, T, hidden]; writes K/V into cache slots [s0, s0+T)."""
        lyr = self.layers[li]
        t = h.shape[1]
        s1 = s0 + t
        residual = h
        x = _rms_norm(h, lyr["input_layernorm"])

        q = F.linear(x, lyr["q"]).view(1, t, self.n_heads, self.head_dim)
        k = F.linear(x, lyr["k"]).view(1, t, self.n_kv_heads, self.head_dim)
        v = F.linear(x, lyr["v"]).view(1, t, self.n_kv_heads, self.head_dim)
        # per-head QK norm over head_dim, then transpose to [1, H, T, hd]
        q = _rms_norm(q, lyr["q_norm"]).transpose(1, 2)
        k = _rms_norm(k, lyr["k_norm"]).transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.rope_cos[s0:s1].unsqueeze(0)  # [1, t, hd]
        sin = self.rope_sin[s0:s1].unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        self.k_cache[li, :, s0:s1] = k[0]
        self.v_cache[li, :, s0:s1] = v[0]

        keys = self.k_cache[li, :, :s1].repeat_interleave(self.n_rep, dim=0).unsqueeze(0)
        vals = self.v_cache[li, :, :s1].repeat_interleave(self.n_rep, dim=0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, keys, vals, is_causal=(s1 == t and t > 1))
        out = out.transpose(1, 2).reshape(1, t, self.hidden_size)
        h = residual + F.linear(out, lyr["o"])

        residual = h
        x = _rms_norm(h, lyr["post_attention_layernorm"])
        h = residual + F.linear(
            F.silu(F.linear(x, lyr["gate"])) * F.linear(x, lyr["up"]), lyr["down"])
        return h

    def _check_ids(self, ids: torch.Tensor, t: int) -> torch.Tensor:
        if ids.shape[0] != 1 or ids.shape[-1] != N_VQ + 1 or ids.shape[1] != t:
            raise ValueError(f"expected input ids [1, {t}, {N_VQ + 1}], got {tuple(ids.shape)}")
        if self._seq + t > self.max_seq_len:
            raise ValueError(f"seq len {self._seq + t} exceeds KV cache {self.max_seq_len}")
        return ids.to(device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------ api
    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> HiddenState:
        """input_ids [1, L, 33]; resets the cache and fills slots [0, L)."""
        if input_ids.dim() != 3 or input_ids.shape[0] != 1 or input_ids.shape[-1] != N_VQ + 1:
            raise ValueError(f"expected input ids [1, L, {N_VQ + 1}], got {tuple(input_ids.shape)}")
        t = int(input_ids.shape[1])
        self.reset()
        ids = self._check_ids(input_ids, t)
        h = self._embed(ids)
        for li in range(self.n_layers):
            h = self._layer(li, h, 0)
        h = _rms_norm(h, self.final_norm)
        self._seq = t
        return HiddenState(h)

    @torch.inference_mode()
    def step(self, row: torch.Tensor) -> HiddenState:
        """Single decode step; row [1, 33] (or [1, 1, 33]); appends one cache slot."""
        if row.dim() == 2:            # [1, 33] -> [1, 1, 33]
            row = row.unsqueeze(0)
        ids = self._check_ids(row, 1)
        s0 = self._seq
        h = self._embed(ids)
        for li in range(self.n_layers):
            h = self._layer(li, h, s0)
        h = _rms_norm(h, self.final_norm)
        self._seq = s0 + 1
        return HiddenState(h)

    # ------------------------------------------------------------------ heads
    def _as_last_hidden(self, h) -> torch.Tensor:
        if isinstance(h, HiddenState):
            h = h.last_hidden
        return h

    def text_logits(self, h) -> torch.Tensor:
        """Full 155648-way text head. h [1,1,H]/[1,H]/[H] -> [V]; [1,T,H] -> [T,V]."""
        h = self._as_last_hidden(h)
        if h.dim() == 3:
            if h.shape[0] != 1:
                raise ValueError("batch must be 1")
            if h.shape[1] == 1:
                return F.linear(h[0, 0], self.head_text)
            return F.linear(h[0], self.head_text)
        if h.dim() == 2 and h.shape[0] == 1:
            return F.linear(h[0], self.head_text)
        if h.dim() == 1:
            return F.linear(h, self.head_text)
        raise ValueError(f"unsupported hidden shape {tuple(h.shape)}")

    def text_logits_2way(self, h) -> torch.Tensor:
        """Only the two audio-phase text logits: [gen_slot, delay_slot] order.
        h [1,1,H]/[1,H]/[H] -> [2]; [1,T,H] -> [T,2]."""
        h = self._as_last_hidden(h)
        if h.dim() == 3:
            if h.shape[0] != 1:
                raise ValueError("batch must be 1")
            if h.shape[1] == 1:
                return F.linear(h[0, 0], self.head_2way)
            return F.linear(h[0], self.head_2way)
        if h.dim() == 2 and h.shape[0] == 1:
            return F.linear(h[0], self.head_2way)
        if h.dim() == 1:
            return F.linear(h, self.head_2way)
        raise ValueError(f"unsupported hidden shape {tuple(h.shape)}")

    def audio_logits(self, h) -> torch.Tensor:
        """Audio heads (per-head GEMMs, bitwise-matched to the reference):
        [32, 1025] per position, pad column (1024) masked to -inf (matches
        forward's logits[..., -1] = -inf).
        h [1,1,H]/[1,H]/[H] -> [32,1025]; [1,T,H] -> [T,32,1025]."""
        h = self._as_last_hidden(h)
        squeeze = False
        if h.dim() == 3:
            if h.shape[0] != 1:
                raise ValueError("batch must be 1")
            if h.shape[1] == 1:
                h = h[0, 0]
                squeeze = True
            else:
                h = h[0]
        elif h.dim() == 2 and h.shape[0] == 1:
            h = h[0]
            squeeze = True
        elif h.dim() != 1:
            raise ValueError(f"unsupported hidden shape {tuple(h.shape)}")
        per_head = [F.linear(h, self.head_audio[i]) for i in range(N_VQ)]
        if per_head[0].dim() == 1:
            out = torch.stack(per_head, dim=0)   # [32, 1025]
        else:
            out = torch.stack(per_head, dim=1)   # [T, 32, 1025]
        out[..., AUDIO_PAD_CODE] = float("-inf")
        if squeeze:
            out = out.squeeze(0)
        return out
