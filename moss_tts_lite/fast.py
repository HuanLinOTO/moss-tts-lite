"""CUDA-graph accelerated decode for MOSS-TTS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .model import (
    AUDIO_END_TOKEN_ID,
    AUDIO_GEN_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    AUDIO_DELAY_SLOT_TOKEN_ID,
    IM_END_TOKEN_ID,
    N_VQ,
    PAD_TOKEN_ID,
    HiddenState,
    _rms_norm,
    _rotate_half,
)
from .sampling import sample_token
from .generate import GenResult

_INT64_MAX = 9223372036854775807

_LIN_NAMES = ("q", "k", "v", "o", "gate", "up", "down")

class FastMossTTS:
    """CUDA-graph decoder around a MossTTSModel (weights/cache are shared)."""

    def __init__(self, model, enable_gqa: bool = True, gqa_max_len: int = 640,
                 quant: str | None = None, inner_k_tiles: int = 8,
                 w4_group_size: int = 128):
        self.m = model
        dev = model.device
        self.enable_gqa = enable_gqa
        self.quant = quant

        self.gqa_max_len = gqa_max_len
        if quant not in (None, "w8", "w4"):
            raise ValueError(f"unsupported quant mode {quant!r} (None, 'w8' or 'w4')")
        self.inner_k_tiles = inner_k_tiles

        if w4_group_size not in (32, 64, 128, 256):
            raise ValueError("w4_group_size must be 32/64/128/256")
        self.w4_group_size = w4_group_size

        self.ids_row = torch.zeros(1, 1, N_VQ + 1, dtype=torch.long, device=dev)
        self.pos_dev = torch.zeros(1, dtype=torch.long, device=dev)
        self.q_static = torch.zeros(1, model.n_heads, 1, model.head_dim,
                                    dtype=model.dtype, device=dev)
        self.o_static = torch.zeros(1, model.hidden_size, dtype=model.dtype, device=dev)
        self.h_static = torch.zeros(1, 1, model.hidden_size, dtype=model.dtype, device=dev)
        self.h_final = torch.zeros(1, model.hidden_size, dtype=model.dtype, device=dev)
        self.lt_static = torch.zeros(model.text_vocab, dtype=model.dtype, device=dev)
        self.la_static = torch.zeros(N_VQ, AUDIO_PAD_CODE + 1, dtype=model.dtype, device=dev)

        self.two_static = torch.zeros(2, dtype=model.dtype, device=dev)
        self.lt_buf = torch.full((model.text_vocab,), float("-inf"),
                                 dtype=model.dtype, device=dev)
        self.qlayers: list | None = None
        if quant is not None:
            self._quantize()
        self.graphs: list | None = None
        self.capture_stream = torch.cuda.Stream() if dev.type == "cuda" else None

    def _quantize(self) -> None:
        """Quantize the 7 backbone linears/layer;"""
        m = self.m
        self.qlayers = []
        for li in range(m.n_layers):
            lyr = m.layers[li]
            qd = {}
            for name in _LIN_NAMES:
                w = lyr[name]
                if self.quant == "w8":
                    wf = w.float()
                    sc = (wf.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
                    w8 = torch.round(wf / sc[:, None]).clamp(-127, 127).to(torch.int8)
                    qd[name] = (w8.contiguous(), sc.contiguous())
                    del wf, w8
                else:
                    N, K = w.shape
                    g = self.w4_group_size
                    wg = w.float().reshape(N, K // g, g)
                    mx, mn = wg.amax(-1, keepdim=True), wg.amin(-1, keepdim=True)
                    s = ((mx - mn) / 15.0).clamp(min=1e-6)
                    q = ((wg - mn) / s).round().clamp(0, 15) \
                        .to(torch.uint8).reshape(N, K)

                    qsz = torch.stack(
                        [s.squeeze(-1), (mn + 8.0 * s).squeeze(-1)], -1) \
                        .bfloat16().transpose(0, 1).contiguous()
                    packed = torch.ops.aten._convert_weight_to_int4pack(
                        (q[:, 1::2] | (q[:, 0::2] << 4)).contiguous(),
                        self.inner_k_tiles)
                    qd[name] = (packed, qsz)
                    del wg, q

                lyr.pop(name)
            self.qlayers.append(qd)
        torch.cuda.empty_cache()

    def _linear(self, x: torch.Tensor, li: int, name: str) -> torch.Tensor:
        """Backbone linear in the active weight mode (bf16 / int8pack / int4pack)."""
        if self.quant is None:
            return F.linear(x, self.m.layers[li][name])
        wq = self.qlayers[li][name]
        x2 = x.reshape(-1, x.shape[-1])
        if self.quant == "w8":
            w8, sc = wq
            out = torch._weight_int8pack_mm(x2, w8, sc).to(self.m.dtype)
            return out.reshape(*x.shape[:-1], w8.shape[0])
        packed, qsz = wq
        out = torch._weight_int4pack_mm(x2, packed, self.w4_group_size, qsz)
        if out.dtype != self.m.dtype:
            out = out.to(self.m.dtype)
        return out.reshape(*x.shape[:-1], qsz.shape[1])

    def _p_embed(self) -> None:
        m = self.m
        h = F.embedding(self.ids_row[..., 0], m.embed_tokens)
        for i in range(N_VQ):
            h = h + F.embedding(self.ids_row[..., i + 1], m.emb_ext[i])
        self.h_static.copy_(h)

    def _p_pre(self, li: int) -> None:
        """input-norm + qkv + qk-norm + rope + cache write (graph-safe)."""
        m = self.m
        lyr = m.layers[li]
        x = _rms_norm(self.h_static, lyr["input_layernorm"])
        q = self._linear(x, li, "q").view(1, 1, m.n_heads, m.head_dim)
        k = self._linear(x, li, "k").view(1, 1, m.n_kv_heads, m.head_dim)
        v = self._linear(x, li, "v").view(1, 1, m.n_kv_heads, m.head_dim)
        q = _rms_norm(q, lyr["q_norm"]).transpose(1, 2)
        k = _rms_norm(k, lyr["k_norm"]).transpose(1, 2)
        v = v.transpose(1, 2)
        cos = m.rope_cos.index_select(0, self.pos_dev).unsqueeze(0)
        sin = m.rope_sin.index_select(0, self.pos_dev).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        m.k_cache[li].index_copy_(1, self.pos_dev, k[0])
        m.v_cache[li].index_copy_(1, self.pos_dev, v[0])
        self.q_static.copy_(q)

    def _p_attn(self, li: int, s1: int) -> None:
        """Eager attention at the EXACT current cache length (not captured)."""
        m = self.m
        if self.enable_gqa and s1 <= self.gqa_max_len:
            keys = m.k_cache[li, :, :s1].unsqueeze(0)
            vals = m.v_cache[li, :, :s1].unsqueeze(0)
            out = F.scaled_dot_product_attention(
                self.q_static, keys, vals, enable_gqa=True)
        else:
            keys = m.k_cache[li, :, :s1].repeat_interleave(m.n_rep, dim=0).unsqueeze(0)
            vals = m.v_cache[li, :, :s1].repeat_interleave(m.n_rep, dim=0).unsqueeze(0)
            out = F.scaled_dot_product_attention(self.q_static, keys, vals)
        self.o_static.view(1, m.n_heads, m.head_dim).copy_(
            out.view(1, m.n_heads, m.head_dim))

    def _p_post(self, li: int) -> None:
        """o_proj + residual + post-norm + SwiGLU MLP."""
        m = self.m
        lyr = m.layers[li]
        h = self.h_static + self._linear(self.o_static.view(1, 1, m.hidden_size), li, "o")
        x = _rms_norm(h, lyr["post_attention_layernorm"])
        h = h + self._linear(
            F.silu(self._linear(x, li, "gate")) * self._linear(x, li, "up"), li, "down")
        self.h_static.copy_(h)

    def _p_final(self) -> None:
        """final norm + full text head + 32 audio heads (bitwise = model heads)."""
        m = self.m
        hf = _rms_norm(self.h_static, m.final_norm)
        self.h_final.copy_(hf.view(1, m.hidden_size))
        hv = hf[0, 0]
        self.lt_static.copy_(F.linear(hv, m.head_text))
        la = torch.stack([F.linear(hv, m.head_audio[i]) for i in range(N_VQ)], dim=0)
        la[..., AUDIO_PAD_CODE] = float("-inf")
        self.la_static.copy_(la)

    def _p_final_audio2(self) -> None:
        """Audio-phase variant:"""
        m = self.m
        hf = _rms_norm(self.h_static, m.final_norm)
        self.h_final.copy_(hf.view(1, m.hidden_size))
        hv = hf[0, 0]
        self.two_static.copy_(F.linear(hv, m.head_2way))
        la = torch.stack([F.linear(hv, m.head_audio[i]) for i in range(N_VQ)], dim=0)
        la[..., AUDIO_PAD_CODE] = float("-inf")
        self.la_static.copy_(la)

    def _run_eager(self, s1: int) -> None:
        """One full decode step without graphs (capture warmup / debugging)."""
        n = self.m.n_layers
        self._p_embed()
        self._p_pre(0)
        for li in range(n):
            self._p_attn(li, s1)
            if li + 1 < n:
                self._p_post(li)
                self._p_pre(li + 1)
            else:
                self._p_post(li)
                self._p_final()

    def _embed_q(self, ids: torch.Tensor) -> torch.Tensor:
        """model."""
        m = self.m
        h = F.embedding(ids[..., 0], m.embed_tokens)
        for i in range(N_VQ):
            h = h + F.embedding(ids[..., i + 1], m.emb_ext[i])
        return h

    def _layer_q(self, li: int, h: torch.Tensor, s0: int) -> torch.Tensor:
        """model."""
        m = self.m
        lyr = m.layers[li]
        t = h.shape[1]
        s1 = s0 + t
        residual = h
        x = _rms_norm(h, lyr["input_layernorm"])
        q = self._linear(x, li, "q").view(1, t, m.n_heads, m.head_dim)
        k = self._linear(x, li, "k").view(1, t, m.n_kv_heads, m.head_dim)
        v = self._linear(x, li, "v").view(1, t, m.n_kv_heads, m.head_dim)
        q = _rms_norm(q, lyr["q_norm"]).transpose(1, 2)
        k = _rms_norm(k, lyr["k_norm"]).transpose(1, 2)
        v = v.transpose(1, 2)
        cos = m.rope_cos[s0:s1].unsqueeze(0)
        sin = m.rope_sin[s0:s1].unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        m.k_cache[li, :, s0:s1] = k[0]
        m.v_cache[li, :, s0:s1] = v[0]
        keys = m.k_cache[li, :, :s1].repeat_interleave(m.n_rep, dim=0).unsqueeze(0)
        vals = m.v_cache[li, :, :s1].repeat_interleave(m.n_rep, dim=0).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, keys, vals,
                                             is_causal=(s1 == t and t > 1))
        out = out.transpose(1, 2).reshape(1, t, m.hidden_size)
        h = residual + self._linear(out, li, "o")
        residual = h
        x = _rms_norm(h, lyr["post_attention_layernorm"])
        h = residual + self._linear(
            F.silu(self._linear(x, li, "gate")) * self._linear(x, li, "up"),
            li, "down")
        return h

    def prefill(self, input_ids: torch.Tensor) -> HiddenState:
        """Dispatch:"""
        if self.quant is None:
            return self.m.prefill(input_ids)
        m = self.m
        if (input_ids.dim() != 3 or input_ids.shape[0] != 1
                or input_ids.shape[-1] != N_VQ + 1):
            raise ValueError(f"expected input ids [1, L, {N_VQ + 1}], "
                             f"got {tuple(input_ids.shape)}")
        t = int(input_ids.shape[1])
        if t > m.max_seq_len:
            raise ValueError(f"seq len {t} exceeds KV cache {m.max_seq_len}")
        m.reset()
        ids = input_ids.to(device=m.device, dtype=torch.long)
        h = self._embed_q(ids)
        for li in range(m.n_layers):
            h = self._layer_q(li, h, 0)
        h = _rms_norm(h, m.final_norm)
        m._seq = t
        return HiddenState(h)

    def capture(self) -> None:
        """Capture the 37 step sub-graphs (1 + n_layers)."""
        if self.graphs is not None:
            return
        if self.m.device.type != "cuda":
            raise RuntimeError("FastMossTTS graphs require cuda")
        m = self.m
        n = m.n_layers
        scratch = m.max_seq_len - 1
        self.pos_dev.fill_(scratch)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._run_eager(m.seq_len)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        def cap_mid(li: int):
            return lambda: (self._p_post(li), self._p_pre(li + 1))

        pieces = [lambda: (self._p_embed(), self._p_pre(0))]
        pieces += [cap_mid(li) for li in range(n - 1)]
        pieces.append(lambda: (self._p_post(n - 1), self._p_final()))
        pieces.append(lambda: (self._p_post(n - 1), self._p_final_audio2()))

        pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        for fn in pieces:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=self.capture_stream):
                fn()
            self.graphs.append(g)
        torch.cuda.synchronize()

    def step(self, row: torch.Tensor | None, pos: int, audio2: bool = False):
        """Replay one decode step."""
        if self.graphs is None:
            raise RuntimeError("call capture() first")
        if row is not None:
            src = row if row.dim() == 2 else row[0]
            if src.data_ptr() != self.ids_row.data_ptr():
                self.ids_row.copy_(src)
        self.pos_dev.fill_(pos)
        g = self.graphs
        n = self.m.n_layers
        g[0].replay()
        for li in range(n - 1):
            self._p_attn(li, pos + 1)
            g[li + 1].replay()
        self._p_attn(n - 1, pos + 1)
        g[n + (1 if audio2 else 0)].replay()
        return self.lt_static, self.la_static, self.h_final, self.two_static

_WD_LOW_ENERGY_CH0 = frozenset(
    (9, 60, 109, 123, 163, 209, 216, 365, 399, 402, 421, 580, 715, 734,
     774, 795, 827, 888, 890, 936, 971, 996, 1001))

_WD_SEGMENT_FLOOR = 640
_WD_SEGMENT_BACKSTOP_RUN = 32

@dataclass
class GenResultWatchdog(GenResult):
    """GenResult + watchdog metadata (isinstance(res, GenResult) holds)."""

    watchdog_triggered: bool = False
    watchdog_reason: str = ""
    watchdog_stats: dict = field(default_factory=dict)

@torch.inference_mode()
def generate_fast(
    fast: FastMossTTS,
    prompt: dict,
    max_new_tokens=4096,
    seed=1234,
    greedy=False,
    text_temperature=1.5,
    text_top_p=1.0,
    text_top_k=50,
    audio_temperature=1.7,
    audio_top_p=0.8,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
    stats: dict | None = None,
    watchdog: bool = True,
    watchdog_silence_frames: int | None = None,
    watchdog_max_segment_frames: int | None = None,
    max_requested_pause_s: float | None = None,
) -> GenResult:
    """Delay-pattern generation on top of FastMossTTS."""
    model = fast.m
    input_ids = prompt["input_ids"]
    attention_mask = prompt.get("attention_mask")
    device = model.device
    input_ids = input_ids.to(device=device, dtype=torch.long)
    if input_ids.dim() != 3 or input_ids.shape[0] != 1 or input_ids.shape[-1] != N_VQ + 1:
        raise ValueError(f"expected prompt input_ids [1, L, {N_VQ + 1}], got {tuple(input_ids.shape)}")
    if attention_mask is None:
        attention_mask = torch.ones(1, int(input_ids.shape[1]),
                                    dtype=torch.bool, device=device)
    else:
        am = attention_mask.to(device)
        if am.dim() == 1:
            am = am.unsqueeze(0)
        if tuple(am.shape) != (1, input_ids.shape[1]):
            raise ValueError(f"bad attention_mask shape {tuple(am.shape)}")
        if not bool(am.all()):
            raise NotImplementedError("batch=1 generate_fast() requires an all-True prompt mask")
    if int(input_ids.shape[1]) + max_new_tokens > model.max_seq_len:
        raise ValueError(f"prompt {int(input_ids.shape[1])} + max_new_tokens {max_new_tokens} "
                         f"exceeds KV cache {model.max_seq_len}")

    seq_len = int(input_ids.shape[1])
    _stats = stats if (stats is not None and device.type == "cuda") else None

    def _t0():
        if _stats is None:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _t1(e0, slot):
        if _stats is None:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        if slot == "prefill":
            _stats["prefill"] = (e0, e)
        else:
            _stats.setdefault("step_ms", []).append((e0, e))

    if greedy:
        text_temperature = 0
        audio_temperature = 0
    text_do_sample = text_temperature > 0
    if not text_do_sample:
        text_temperature = 1
    audio_do_sample = audio_temperature > 0
    if not audio_do_sample:
        audio_temperature = 1

    torch.manual_seed(seed)

    col0 = input_ids[0, :, 0]
    last0 = int(col0[-1])
    is_continuation = last0 in (AUDIO_START_TOKEN_ID, AUDIO_GEN_SLOT_TOKEN_ID)
    hit = (col0 == AUDIO_START_TOKEN_ID).nonzero()
    start_idx = int(hit[-1]) if hit.numel() else -1
    al = seq_len - start_idx if (is_continuation and start_idx != -1) else 0
    is_audio = is_continuation and start_idx != -1
    dl = _INT64_MAX
    is_stopping = False

    pre_exclude_mask0 = torch.tensor(
        [PAD_TOKEN_ID, AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_DELAY_SLOT_TOKEN_ID,
         AUDIO_END_TOKEN_ID], device=device)

    generation_ids = input_ids.clone()
    text_steps: list = []
    audio_steps: list = []
    n_steps = 0
    pos = seq_len
    captured = False

    wd_stop = False
    wd_reason = ""
    wd_silent_run = 0
    wd_peak_segment = 0
    wd_stats: dict = {}
    wd_floor = (watchdog_max_segment_frames if watchdog_max_segment_frames
                is not None else _WD_SEGMENT_FLOOR)

    wd_sil = watchdog_silence_frames if watchdog_silence_frames is not None else 64
    if max_requested_pause_s is not None and max_requested_pause_s > 0:
        wd_sil = max(wd_sil, math.ceil((max_requested_pause_s + 2.0) * 12.5))

    for time_step in range(max_new_tokens):

        e0 = _t0()
        if time_step == 0:
            hs = fast.prefill(input_ids)
            h = hs.last_hidden[:, -1]
            lt_step = la_step = two_step = None
            if not captured:
                fast.capture()
                captured = True
        else:
            lt_step, la_step, h, two_step = fast.step(None, pos, audio2=is_audio)
            pos += 1
        _t1(e0, "prefill" if time_step == 0 else "step")

        next_text = PAD_TOKEN_ID
        if wd_stop:

            if dl == _INT64_MAX or dl < N_VQ:
                next_text = AUDIO_DELAY_SLOT_TOKEN_ID
            else:
                next_text = AUDIO_END_TOKEN_ID
                is_audio = False
        elif not is_stopping and dl < N_VQ:
            next_text = AUDIO_DELAY_SLOT_TOKEN_ID
        elif not is_stopping and dl == N_VQ:
            next_text = AUDIO_END_TOKEN_ID
            is_audio = False
        sampling_text = (not is_stopping) and (not wd_stop) and dl > N_VQ

        if sampling_text:
            if is_audio and not text_do_sample:

                two_raw = (two_step if two_step is not None
                           else F.linear(h[0], model.head_2way))
                two = two_raw / text_temperature
                if time_step == 0:
                    two[1] = float("-inf")
                pick = int(torch.argmax(two))
                next_text = (AUDIO_GEN_SLOT_TOKEN_ID if pick == 0
                             else AUDIO_DELAY_SLOT_TOKEN_ID)
            elif is_audio:

                two_raw = (two_step if two_step is not None
                           else F.linear(h[0], model.head_2way))
                lt_buf = fast.lt_buf
                lt_buf.fill_(float("-inf"))
                lt_buf[AUDIO_GEN_SLOT_TOKEN_ID] = two_raw[0] / text_temperature
                lt_buf[AUDIO_DELAY_SLOT_TOKEN_ID] = two_raw[1] / text_temperature
                if time_step == 0:
                    lt_buf[AUDIO_DELAY_SLOT_TOKEN_ID] = float("-inf")
                tok = sample_token(lt_buf.view(1, -1), top_p=text_top_p,
                                   top_k=text_top_k, do_sample=True)
                next_text = int(tok[0])
            else:
                lt = (lt_step if lt_step is not None else model.text_logits(h))
                lt = lt / text_temperature
                lt = lt.index_fill(0, pre_exclude_mask0, float("-inf"))
                if time_step == 0:
                    lt[AUDIO_DELAY_SLOT_TOKEN_ID] = float("-inf")
                if time_step <= N_VQ:
                    lt[IM_END_TOKEN_ID] = float("-inf")
                tok = sample_token(lt.view(1, -1), top_p=text_top_p, top_k=text_top_k,
                                   do_sample=text_do_sample)
                next_text = int(tok[0])
        if next_text == AUDIO_START_TOKEN_ID:
            is_audio = True
        if next_text == IM_END_TOKEN_ID:
            is_stopping = True

        smask = [(al > j) and (dl == _INT64_MAX or j > dl - 1) for j in range(N_VQ)]
        next_audio_tokens = torch.full((1, N_VQ), AUDIO_PAD_CODE, device=device,
                                       dtype=torch.long)
        if any(smask):
            la = (la_step if la_step is not None else model.audio_logits(h))
            audio_logit = la / audio_temperature
            if smask[0]:
                ch0 = audio_logit[0].view(1, -1)
                ch0[..., AUDIO_PAD_CODE] = float("-inf")
                tok0 = sample_token(
                    logits=ch0,
                    prev_tokens=generation_ids[:, :, 1],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p, top_k=audio_top_k, do_sample=audio_do_sample)
                next_audio_tokens[0, 0] = int(tok0[0])
            rest = [j for j in range(1, N_VQ) if smask[j]]
            if rest:
                rest_l = audio_logit[rest]
                rest_l[..., AUDIO_PAD_CODE] = float("-inf")
                tok = sample_token(
                    logits=rest_l,
                    prev_tokens=generation_ids[:, :, 2:],
                    repetition_penalty=audio_repetition_penalty,
                    top_p=audio_top_p, top_k=audio_top_k, do_sample=audio_do_sample)
                for k, j in enumerate(rest):
                    next_audio_tokens[0, j] = int(tok[k])

        ch0_val = int(next_audio_tokens[0, 0])
        if watchdog and not wd_stop and is_audio and dl == _INT64_MAX:
            if ch0_val in _WD_LOW_ENERGY_CH0:
                wd_silent_run += 1
            else:
                wd_silent_run = 0
            seg_bound = max(wd_floor, 2 * wd_peak_segment)
            if wd_silent_run >= wd_sil:
                wd_stop, wd_reason = True, "silence"
            elif al > seg_bound and wd_silent_run >= _WD_SEGMENT_BACKSTOP_RUN:
                wd_stop, wd_reason = True, "max_segment"
            if wd_stop:
                wd_stats = {"reason": wd_reason, "step": n_steps + 1,
                            "segment_frames": al, "silent_run": wd_silent_run,
                            "peak_completed_segment": wd_peak_segment,
                            "silence_threshold_frames": wd_sil,
                            "segment_bound_frames": seg_bound,
                            "max_requested_pause_s": max_requested_pause_s}

        if next_text in (AUDIO_START_TOKEN_ID, AUDIO_GEN_SLOT_TOKEN_ID,
                         AUDIO_DELAY_SLOT_TOKEN_ID):
            al += 1
        if next_text == AUDIO_END_TOKEN_ID:
            wd_peak_segment = max(wd_peak_segment, al)
            wd_silent_run = 0
            al = 0
        if dl == _INT64_MAX and next_text == AUDIO_DELAY_SLOT_TOKEN_ID:
            dl = 0
        if dl != _INT64_MAX:
            dl += 1
        if dl > N_VQ:
            dl = _INT64_MAX

        next_text_token = torch.full((1,), next_text, device=device, dtype=torch.long)
        current_input_ids = torch.cat(
            [next_text_token[:, None, None], next_audio_tokens[:, None, :]], dim=2)
        generation_ids = torch.cat([generation_ids, current_input_ids], dim=1)
        fast.ids_row.copy_(current_input_ids[0])

        text_steps.append(next_text)
        audio_steps.append(next_audio_tokens[0].clone())
        n_steps += 1

        if is_stopping:
            break
        if wd_stop and next_text == AUDIO_END_TOKEN_ID:
            break

    text_ids = (torch.tensor(text_steps, dtype=torch.long) if text_steps
                else torch.empty(0, dtype=torch.long))
    audio_frames = (torch.stack(audio_steps) if audio_steps
                    else torch.empty(0, N_VQ, dtype=torch.long))
    if _stats is not None:
        torch.cuda.synchronize()
        p = _stats.get("prefill")
        _stats["prefill_ms"] = float(p[0].elapsed_time(p[1])) if p else None
        _stats["step_ms"] = [float(a.elapsed_time(b))
                             for a, b in _stats.get("step_ms", [])]
        _stats["steps"] = len(_stats["step_ms"])
    return GenResultWatchdog(
        text_ids=text_ids.cpu(), audio_frames=audio_frames.cpu(),
        finished=bool(is_stopping) or bool(wd_stats), n_steps=n_steps,
        watchdog_triggered=bool(wd_stats), watchdog_reason=wd_reason,
        watchdog_stats=wd_stats)
