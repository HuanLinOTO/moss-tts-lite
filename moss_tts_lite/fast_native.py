"""Native-operator fast decode for MOSS-TTS (whole-step CUDA graph, W4).

Measured on an A10G-24G with the shipped w1p checkpoint, zh_plain, seed 1234,
audio-phase step at a fixed cache length (`.tmp/native_agent/same_len.py`):

    path                          ms/step   steps/s   text argmax  top-25 cover
    fast.py (38 sub-graphs)        12.23      81.7       100.00%      75.39%
    n1 whole-step graph            12.01      83.3       100.00%      98.91%
    n2 + fused GEMMs/norm/heads    10.21      97.9       100.00%      97.50%

`n1` is *bitwise-identical* to `fast.py`: verified over 40 teacher-forced rows in
both phases (`lt_buf`, `two_buf`, `la_buf` all max|d| = 0), across the
`gqa_max_len` attention-mode switch (0/36 mismatches at len 691-726), and its
generated utterance is row-for-row equal for all 164 steps.  It is only 2% faster, because
the step is GPU-bound at ~1080 kernels and merging graph launches does not
remove kernels.  `n2` buys the real 16% via kernel-count cuts; the cost is that
its forward is no longer bitwise (the `F.rms_norm` swap alone moves audio logits
by 0.5), so it decodes a *different* but equally valid utterance, and its
top-25 gate agreement (97.50%) still clears the >=95% hard gate that the
unmodified W4 path **fails** at 75.39%.

What this module does
---------------------
1. **Whole-step CUDA graph.**  `fast.py` keeps attention eager and splits each
   step into 38 sub-graphs, because a padded/masked SDPA dispatches to a
   different kernel than the exact-length call its bitwise gate demands.  Here
   embedding + all 36 layers incl. attention + all heads (and, in n3, the FSM
   and sampling) are one graph, so a step is a single replay.  Attention keeps
   `fast.py`'s exact-length call; since that length is fixed inside a graph, one
   graph is captured per cache length.  `bucket_for` documents why every
   padded/bucketed alternative was measured and rejected, and `max_graphs`
   bounds the resulting graph cache.

2. **Kernel-count reduction (where the milliseconds actually are).**  The 32
   audio heads become one `[32*1025, 4096]` GEMM, the audio-phase text rows one
   `[2, 4096]` GEMM, `(q,k)` and `(gate,up)` become single int4 GEMMs over
   concatenated payloads (bitwise-neutral: concatenating output rows leaves each
   row's K accumulation untouched), the 33 embedding gathers become one gather
   plus one reduction, the 6-op `_rms_norm` fp32 chain becomes `F.rms_norm`, and
   rope becomes one table gather (`_rms_norm` is only kept in `prefill`, where a
   per-step difference compounds into a different utterance).  Step kernels drop
   from ~2346 to ~1080.

3. **Tensorized delay FSM + in-graph sampling (arm n3) -- NOT recommended.**
   Kept for reference only: drawing all 32 channels every step (vs the
   reference's channel-subset draws) changes the RNG stream, and on this model
   that makes the trajectory run away (zh 3170 steps, en non-terminating), so it
   is both wrong and slow.  See the note at `ARMS["n3"]`.

Three bugs were found and fixed while building this; all three are pinned by
`tests/test_fast_native.py` phase C and verified to bite by mutation.  The
"audio rows diverged from step 1" and "en ran 1766 steps" symptoms both had the
*same* cause -- `replay()` publishing only `two_buf` in the audio phase, so
audio tokens were sampled from `model.audio_logits()` on the prefill hidden
state -- and both are fixed by taking `la_step` unconditionally.  The third
(`not is_stopping` guards in `text_decision`, below) is a real deviation from
the reference but is *not* observable as a step count on these prompts; it is
pinned as a differential table instead.

Not pursued (measured and left alone): the int4 GEMMs are 7.16 of the ~10 ms and
run at ~423 GB/s of the A10G's ~600 GB/s, so the remaining headroom is memory
bandwidth, not launch overhead -- the activations sum to under 1 MiB and the
cost of removing whole blocks (attention 0.46 ms, KV write 0.17 ms, q/k norms
0.12 ms) is only 0.70 ms in total.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .fast import (
    _WD_LOW_ENERGY_CH0,
    _WD_SEGMENT_BACKSTOP_RUN,
    _WD_SEGMENT_FLOOR,
    FastMossTTS,
    GenResultWatchdog,
)
from .generate import GenResult
from .model import (
    AUDIO_DELAY_SLOT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_GEN_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    IM_END_TOKEN_ID,
    N_VQ,
    PAD_TOKEN_ID,
    HiddenState,
    _rms_norm,
    _rotate_half,
)
from .sampling import sample_token

_INT64_MAX = 9223372036854775807

#: ablation arms (see `.reports/native-1.md`)
#:   n1 = whole-step graph only: one graph per step (attention still fast.py's
#:        exact-length call), everything else exactly as `fast.py` does it
#:   n2 = n1 + fused heads + fused (q,k)/(gate,up) int4 GEMMs + `F.rms_norm`
#:   n3 = n2 + tensorized delay FSM + batched in-graph sampling
ARMS: dict[str, dict] = {
    "n1": dict(legacy=True, fuse_heads=False, fuse_gemms=False,
               fused_norm=False, in_graph_sample=False),
    # single-feature arms for the ablation table
    "n2n": dict(legacy=False, fuse_heads=False, fuse_gemms=False,
                fused_norm=True, in_graph_sample=False),
    "n2h": dict(legacy=False, fuse_heads=True, fuse_gemms=True,
                fused_norm=False, in_graph_sample=False),
    "n2": dict(legacy=False, fuse_heads=True, fuse_gemms=True,
               fused_norm=True, in_graph_sample=False),
    # n3 is kept for reference but is NOT a win: moving sampling into the graph
    # draws all 32 channels every step (vs fast.py's channel-subset draws) so
    # the RNG stream diverges from the reference, which on this model makes the
    # trajectory run past any usable bound (measured: zh 3170 steps vs 127,
    # en never terminating within 4096).  It is also slower, because the
    # per-length graph cache thrashes once a run exceeds max_graphs.  n2 is the
    # default: it keeps fast.py's exact sampling calls (and so its RNG stream)
    # and only changes the graph structure and GEMM/norm kernels.
    "n3": dict(legacy=False, fuse_heads=True, fuse_gemms=True,
               fused_norm=True, in_graph_sample=True),
}

#: host-facing device readback: [next_text, audio0..31, al, dl, is_audio]
_HSYNC = N_VQ + 4


def text_decision(dl: int, n_vq: int, wd_stop: bool, is_stopping: bool,
                  is_audio: bool):
    """The delay-ramp text decision, isolated so it can be tested as a table.

    Mirrors `generate_fast`'s branch order exactly.  The `not is_stopping`
    guards on the two forced branches are the subtle part: the forced delay
    ramp is what *closes* the delay pattern, but once `IM_END` has been decided
    the reference stops forcing either branch, so a step that already reached
    `dl == n_vq` falls through to light sampling instead of being pinned to
    `audio_end` (and `sampling_text` turns the sampling block back on, which is
    what keeps the RNG stream aligned with the reference).  Note the honest
    scope: on the golden zh/en prompts both variants happen to reach the same
    step count, so this is pinned by `tests/test_fast_native.py` phase C as a
    differential table against the reference branch block, not by an observed
    step-count difference.

    Returns `(next_text, is_audio, sampling_text, forced)`; `forced` says no
    sampling call may consume RNG this step (the reference gates its whole
    sampling block on `sampling_text`).
    """
    if wd_stop:
        if dl == _INT64_MAX or dl < n_vq:
            return AUDIO_DELAY_SLOT_TOKEN_ID, is_audio, False, True
        return AUDIO_END_TOKEN_ID, False, False, True
    if not is_stopping and dl < n_vq:
        return AUDIO_DELAY_SLOT_TOKEN_ID, is_audio, False, True
    if not is_stopping and dl == n_vq:
        return AUDIO_END_TOKEN_ID, False, False, True
    sampling_text = (not is_stopping) and (not wd_stop) and dl > n_vq
    return PAD_TOKEN_ID, is_audio, sampling_text, False


def audio_sampling_mask(al: int, dl: int, n_vq: int) -> list[bool]:
    """Per-channel sampling mask: `al > j` and (dl == MAX or `j > dl - 1`).

    The host mirror of `generate.py`'s `pre_audio_mask & post_audio_mask` for
    batch=1, matching `generate_fast`.
    """
    return [(al > j) and (dl == _INT64_MAX or j > dl - 1) for j in range(n_vq)]


class FastNativeTTS:
    """Whole-step-graph decoder; weight container is a `FastMossTTS`.

    Usage::

        base = load_gptq_fast(model, state)      # or FastMossTTS(model, quant="w4")
        native = FastNativeTTS(base, arm="n3")
        res = generate_native(native, prompt, ...)

    With `fuse_gemms=True` the packed `(q,k)` and `(gate,up)` payloads are
    replaced by concatenated copies and the per-name originals dropped, so
    `base` is only a weight container afterwards (which is all this module
    uses it for).
    """

    def __init__(self, base: FastMossTTS, arm: str = "n2", max_graphs: int = 256):
        """Args:
            max_graphs: cap on captured graphs.  Each whole-step graph costs
                ~6-7.5 MiB of device memory in the shared graph pool (the ~1000
                kernel nodes' parameters and work descriptors; their activations
                sum to under 1 MiB), so the cap is a memory/latency tradeoff:
                one graph is captured per decoded length, 256 covers ~20 s of
                audio, and beyond the cap the least recently used graph is
                dropped and re-captured on demand (~40-80 ms each, which is why
                a trajectory much longer than the cap degrades badly -- the
                cache thrashes).  Measured reachable ceilings on a 24 GiB card:
                256 graphs is comfortable, 400 fits, 1200 raises a capture-time
                CUDA OOM on top of the 8.3 GiB of weights.
        """
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r} (expected one of {sorted(ARMS)})")
        cfg = ARMS[arm]
        self.arm = arm
        self.base = base
        self.m = base.m
        m = self.m
        dev = m.device
        self.max_graphs = int(max_graphs)
        self.legacy = bool(cfg["legacy"])
        self.fuse_heads = bool(cfg["fuse_heads"])
        self.fuse_gemms = bool(cfg["fuse_gemms"])
        self.fused_norm = bool(cfg["fused_norm"])
        self.in_graph_sample = bool(cfg["in_graph_sample"])
        self.scale = 1.0 / math.sqrt(m.head_dim)
        # sampling parameters (overwritten by generate_native)
        self.text_temperature = 1.5
        self.text_top_p = 1.0
        self.text_top_k = 50
        self.audio_temperature = 1.7
        self.audio_top_p = 0.8
        self.audio_top_k = 25

        #: the base's input/pos buffers are reused so arm n1 can call fast.py's
        #: own piece functions verbatim and still read the same row
        #: (`base.capture()` is never used on the native path)
        self.ids_row = base.ids_row
        self.pos_dev = base.pos_dev

        # ---- replay outputs -------------------------------------------------
        self.la_buf = torch.zeros(N_VQ, AUDIO_PAD_CODE + 1, dtype=m.dtype, device=dev)
        self.two_buf = torch.zeros(2, dtype=m.dtype, device=dev)
        self.lt_buf = torch.zeros(m.text_vocab, dtype=m.dtype, device=dev)
        self.hsync = torch.zeros(_HSYNC, dtype=torch.long, device=dev)
        #: device FSM [audio_lengths, delayed_lengths, is_audio]
        self.fsm = torch.zeros(3, dtype=torch.long, device=dev)
        self.seq_len = 0

        # ---- weights: fused heads, fused int4 GEMMs -------------------------
        self.W_audio = m.head_audio.reshape(N_VQ * (AUDIO_PAD_CODE + 1), m.hidden_size)
        self.W_two = m.head_2way
        self.qk_fused: list = [None] * m.n_layers
        self.gu_fused: list = [None] * m.n_layers
        if self.fuse_gemms:
            self._build_fused_gemms()

        # ---- packed rope: one table holds (cos, -sin reordered) -------------
        # rotate_half(x) == x[..., perm] * sgn with perm = [half..D) ++ [0..half)
        # and sgn = [-1]*half ++ [+1]*half, so cos and sin_signed can be
        # gathered together as a single [max_seq, 2, head_dim] table and the
        # whole rope collapses to 1 gather + 2 muls + 1 add per tensor instead
        # of gathering cos and sin separately and building rotate_half with a
        # chunk/cat/neg chain (bitwise-equal: negation is exact in bf16).
        half = m.head_dim // 2
        self.rope_perm = torch.cat((torch.arange(half, m.head_dim, device=dev),
                                    torch.arange(0, half, device=dev)))
        self.rope_signed_sin = None
        if not self.legacy:
            sgn = torch.cat((-torch.ones(half, dtype=m.dtype, device=dev),
                             torch.ones(half, dtype=m.dtype, device=dev)))
            self.rope_tab = torch.stack((m.rope_cos, m.rope_sin * sgn), 1)
            self.rope_signed_sin = sgn

        # ---- fused audio embedding table (33 gathers -> 2 + 1 sum) ----------
        self.emb_audio = None
        self.emb_offs = None
        if not self.legacy:
            self.emb_audio = torch.cat(m.emb_ext, 0).contiguous()  # [32*1025, H]
            self.emb_offs = (torch.arange(N_VQ, device=dev, dtype=torch.long)
                             * m.emb_ext[0].shape[0]).view(1, 1, -1)

        self.ar = torch.arange(N_VQ, dtype=torch.long, device=dev)
        self.graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        self._pool = torch.cuda.graph_pool_handle()
        self._capture_stream = torch.cuda.Stream()

    # ------------------------------------------------------------------ setup
    def _quant_rec(self, li: int, name: str):
        qd = self.base.qlayers[li] if self.base.qlayers else None
        if not qd or name not in qd:
            return None
        gmap = getattr(self.base, "group_size_map", None)
        g = gmap[li][name] if gmap else self.base.w4_group_size
        packed, qsz = qd[name]
        return packed, qsz, g

    def _build_fused_gemms(self) -> None:
        """Concat `(q,k)` / `(gate,up)` int4 payloads into single-GEMM weights.

        Requires both members to be int4 with the same group size (the w1p
        GPTQ state deliberately keeps `v_proj` bf16, and `v` is never fused).
        Output-row concatenation leaves every row's K-loop untouched, so the
        result is bitwise-identical to the two separate GEMMs.
        """
        m = self.m
        for li in range(m.n_layers):
            for key, grp in (("qk", ("q", "k")), ("gu", ("gate", "up"))):
                recs = [self._quant_rec(li, n) for n in grp]
                if any(r is None for r in recs) or len({r[2] for r in recs}) != 1:
                    continue
                packed = torch.cat([r[0] for r in recs], dim=0).contiguous()
                qsz = torch.cat([r[1] for r in recs], dim=1).contiguous()
                slot = self.qk_fused if key == "qk" else self.gu_fused
                slot[li] = (packed, qsz, recs[0][2], recs[0][1].shape[1])
                qd = self.base.qlayers[li]
                for n in grp:
                    qd.pop(n, None)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- primitives
    def _lin(self, x, li: int, name: str):
        return self.base._linear(x, li, name)

    def _lin_fused(self, x, li: int, fkey: str):
        packed, qsz, g, _ = (self.qk_fused if fkey == "qk" else self.gu_fused)[li]
        x2 = x.reshape(-1, x.shape[-1])
        out = torch._weight_int4pack_mm(x2, packed, g, qsz)
        if out.dtype != self.m.dtype:
            out = out.to(self.m.dtype)
        return out.reshape(*x.shape[:-1], qsz.shape[1])

    def _norm(self, x, w):
        if self.fused_norm:
            return F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        return _rms_norm(x, w)

    # ------------------------------------------------------------------ step
    def _step_fn(self, audio2: bool, length: int):
        """Build the whole-step closure captured for one (audio2, length).

        `length` is the KV length this graph's attention reads, i.e. the
        position being decoded is `length - 1` and every replay of this graph
        must use that same position (the attention shape is baked in).

        `legacy` (arm n1) reuses `fast.py`'s own piece functions -- static
        ping-pong buffers, the 6-op rms chain, per-head audio GEMMs -- so the
        ablation isolates the graph structure alone.  `legacy=False` uses this
        module's streamlined sequence (locals instead of static buffers,
        `F.rms_norm`, fused heads, fused int4 GEMMs), which is arms n2/n3.
        """
        m = self.m
        N = N_VQ
        base = self.base
        fuse_heads = self.fuse_heads
        fuse_gemms = self.fuse_gemms
        legacy = self.legacy
        sample = self.in_graph_sample and audio2
        ids = self.ids_row
        pos = self.pos_dev
        # above `gqa_max_len` the reference switches attention mode (flash
        # tiling changes at len 641), so the graph must switch with it
        use_gqa = base.enable_gqa and length <= base.gqa_max_len

        def legacy_step() -> None:
            base._p_embed()
            base._p_pre(0)
            # above `gqa_max_len` fast.py switches to repeat_interleave
            # (flash tiling changes at len 641); mirror it or n1 stops being
            # bitwise-equal to the reference past that length
            use_gqa = base.enable_gqa and length <= base.gqa_max_len
            for li in range(m.n_layers):
                keys = m.k_cache[li, :, :length].unsqueeze(0)
                vals = m.v_cache[li, :, :length].unsqueeze(0)
                if use_gqa:
                    o = F.scaled_dot_product_attention(base.q_static, keys, vals,
                                                       enable_gqa=True)
                else:
                    keys = keys[0].repeat_interleave(m.n_rep, dim=0).unsqueeze(0)
                    vals = vals[0].repeat_interleave(m.n_rep, dim=0).unsqueeze(0)
                    o = F.scaled_dot_product_attention(base.q_static, keys, vals)
                base.o_static.view(1, m.n_heads, m.head_dim).copy_(
                    o.view(1, m.n_heads, m.head_dim))
                base._p_post(li)
                if li + 1 < m.n_layers:
                    base._p_pre(li + 1)
            if audio2:
                base._p_final_audio2()          # 2-way text head + 32 audio heads
            else:
                base._p_final()
            # uniform readout: the legacy pieces write into fast.py's static
            # buffers, so mirror them into this module's outputs
            self.two_buf.copy_(base.two_static)
            self.lt_buf.copy_(base.lt_static)
            self.la_buf.copy_(base.la_static)

        def fast_step() -> None:
            if self.emb_audio is None:
                h = F.embedding(ids[..., 0], m.embed_tokens)
                for i in range(N):
                    h = h + F.embedding(ids[..., i + 1], m.emb_ext[i])
            else:
                # one gather into the stacked audio table + one reduction,
                # instead of 32 gathers and 32 (dependency-chained) adds
                h = F.embedding(ids[..., 0], m.embed_tokens)
                h = h + F.embedding(ids[..., 1:] + self.emb_offs,
                                    self.emb_audio).sum(-2, keepdim=True)

            for li in range(m.n_layers):
                lyr = m.layers[li]
                x = self._norm(h, lyr["input_layernorm"])
                if fuse_gemms and self.qk_fused[li] is not None:
                    qk = self._lin_fused(x, li, "qk")
                    q = qk[..., :m.n_heads * m.head_dim]
                    k = qk[..., m.n_heads * m.head_dim:]
                else:
                    q = self._lin(x, li, "q")
                    k = self._lin(x, li, "k")
                v = self._lin(x, li, "v")
                q = self._norm(q.view(1, 1, m.n_heads, m.head_dim),
                               lyr["q_norm"]).transpose(1, 2)
                k = self._norm(k.view(1, 1, m.n_kv_heads, m.head_dim),
                               lyr["k_norm"]).transpose(1, 2)
                v = v.view(1, 1, m.n_kv_heads, m.head_dim).transpose(1, 2)
                if self.rope_signed_sin is None:
                    cos = m.rope_cos.index_select(0, pos).unsqueeze(0)
                    sin = m.rope_sin.index_select(0, pos).unsqueeze(0)
                    q = q * cos + _rotate_half(q) * sin
                    k = k * cos + _rotate_half(k) * sin
                else:
                    cs = self.rope_tab.index_select(0, pos.view(-1))\
                        .view(2, 1, 1, 1, m.head_dim)
                    cos = cs[0].view(1, 1, 1, -1)
                    ssn = cs[1].view(1, 1, 1, -1)
                    q = q * cos + q[..., self.rope_perm] * ssn
                    k = k * cos + k[..., self.rope_perm] * ssn
                m.k_cache[li].index_copy_(1, pos, k[0])
                m.v_cache[li].index_copy_(1, pos, v[0])
                if use_gqa:
                    x = F.scaled_dot_product_attention(
                        q, m.k_cache[li, :, :length].unsqueeze(0),
                        m.v_cache[li, :, :length].unsqueeze(0), enable_gqa=True)
                else:
                    x = F.scaled_dot_product_attention(
                        q,
                        m.k_cache[li, :, :length].repeat_interleave(m.n_rep, dim=0).unsqueeze(0),
                        m.v_cache[li, :, :length].repeat_interleave(m.n_rep, dim=0).unsqueeze(0))
                h = h + self._lin(x.reshape(1, 1, m.hidden_size), li, "o")

                x = self._norm(h, lyr["post_attention_layernorm"])
                if fuse_gemms and self.gu_fused[li] is not None:
                    gu = self._lin_fused(x, li, "gu")
                    inter = self.gu_fused[li][3]
                    g, u = gu[..., :inter], gu[..., inter:]
                else:
                    g = self._lin(x, li, "gate")
                    u = self._lin(x, li, "up")
                h = h + self._lin(F.silu(g) * u, li, "down")

            hv = self._norm(h, m.final_norm).view(m.hidden_size)
            if audio2:
                self.two_buf.copy_(F.linear(hv, self.W_two))
            else:
                self.lt_buf.copy_(F.linear(hv, m.head_text))
            if fuse_heads:
                self.la_buf.copy_(F.linear(hv, self.W_audio).view(N, AUDIO_PAD_CODE + 1))
            else:
                self.la_buf.copy_(torch.stack(
                    [F.linear(hv, m.head_audio[i]) for i in range(N)], dim=0))
            self.la_buf[:, AUDIO_PAD_CODE] = float("-inf")

        def fn() -> None:
            if legacy:
                legacy_step()
            else:
                fast_step()
            if sample:
                self._fsm_and_sample()

        return fn

    # -------------------------------------------------- tensorized FSM (n3)
    def _fsm_and_sample(self) -> None:
        """Delay FSM + batched sampling, entirely inside the captured graph.

        Only used in the audio phase (the 2-way text head is the only head the
        reference consults there).  Mirrors `generate.py`:
          * `delayed_lengths < n_vq` forces the delay-slot ramp, `== n_vq`
            emits audio_end and clears `is_audio`;
          * channel j samples iff `audio_lengths > j` and the delay mask allows
            it; every other channel gets the pad code;
          * `audio_lengths` counts audio rows and resets on audio_end;
          * `delayed_lengths` MAX -> 0 -> +1 -> wraps back to MAX above n_vq.

        RNG note: all 32 channels are drawn every step, so the stream differs
        from the reference's channel-loop draws (approved deviation).
        """
        N = N_VQ
        fsm = self.fsm
        al = fsm[0]
        dl = fsm[1]
        is_audio = fsm[2].bool()
        maxv = torch.full_like(dl, _INT64_MAX)
        zero = torch.zeros_like(dl)

        # ---- text token -----------------------------------------------------
        two = self.two_buf.float() / self.text_temperature
        pick = torch.multinomial(torch.softmax(two, -1), 1)[0]
        sampled = torch.where(pick == 0,
                              torch.full_like(pick, AUDIO_GEN_SLOT_TOKEN_ID),
                              torch.full_like(pick, AUDIO_DELAY_SLOT_TOKEN_ID))
        forced = torch.where(dl < N,
                             torch.full_like(dl, AUDIO_DELAY_SLOT_TOKEN_ID),
                             torch.full_like(dl, AUDIO_END_TOKEN_ID))
        next_text = torch.where(dl > N, sampled, forced)

        # ---- audio tokens ---------------------------------------------------
        la = self.la_buf.float() / self.audio_temperature
        if 0 < self.audio_top_k < la.shape[-1]:
            # exactly `sampling.apply_top_k`: scatter the VALUES of the k
            # largest entries back, because ties at the k-th value are common
            # on this model (32/32 channels) and a `>= kth` rule would keep
            # 25-51 entries instead of 25.
            vals, idx = torch.topk(la, self.audio_top_k, dim=-1)
            keep = torch.full_like(la, float("-inf")).scatter(-1, idx, vals)
            la = keep
        if self.audio_top_p < 1.0:
            p = torch.softmax(la, -1)
            sp, si = torch.sort(p, descending=True, dim=-1)
            rm = torch.cumsum(sp, -1) > self.audio_top_p
            rm[..., 1:] = rm[..., :-1].clone()
            rm[..., 0] = False
            drop = torch.zeros_like(la, dtype=torch.bool).scatter_(-1, si, rm)
            la = la.masked_fill(drop, float("-inf"))
        tok = torch.multinomial(torch.softmax(la, -1), 1)[:, 0]
        ar = self.ar
        dl_safe = torch.where(dl == maxv, torch.full_like(dl, N), dl)
        smask = (al > ar) & ((ar > dl_safe - 1) | (dl == maxv))
        next_audio = torch.where(smask, tok, torch.full_like(tok, AUDIO_PAD_CODE))

        self.ids_row.copy_(torch.cat(
            (next_text.view(1, 1, 1), next_audio.view(1, 1, N)), dim=2))

        # ---- FSM update (reference order) -----------------------------------
        inc = ((next_text == AUDIO_START_TOKEN_ID)
               | (next_text == AUDIO_GEN_SLOT_TOKEN_ID)
               | (next_text == AUDIO_DELAY_SLOT_TOKEN_ID)).long()
        al2 = torch.where(next_text == AUDIO_END_TOKEN_ID, zero, al + inc)
        step1 = torch.where((dl == maxv)
                            & (next_text == AUDIO_DELAY_SLOT_TOKEN_ID), zero, dl)
        step2 = step1 + (step1 != maxv).long()
        dl2 = torch.where(step2 > N, maxv, step2)
        live = is_audio & (next_text != AUDIO_END_TOKEN_ID) \
            & (next_text != IM_END_TOKEN_ID)
        is_audio2 = live | (next_text == AUDIO_START_TOKEN_ID)
        fsm[0].copy_(al2)
        fsm[1].copy_(dl2)
        fsm[2].copy_(is_audio2.long())
        self.hsync[0].copy_(next_text)
        self.hsync[1:1 + N].copy_(next_audio)
        self.hsync[1 + N].copy_(al2)
        self.hsync[2 + N].copy_(dl2)
        self.hsync[3 + N].copy_(is_audio2.long())

    # ------------------------------------------------------------- bucketing
    #: Why one graph per cache length instead of a fixed bucket ladder:
    #: attention must read exactly the first `length` K/V rows to reproduce the
    #: reference numerics.  Every padded alternative was measured and rejected
    #: -- a zero pad row scores 0, which the softmax weights as `exp(-lse)`, and
    #: on this model that share reaches 0.85-1.0 (the attention logit spread is
    #: large), so pad rows can dominate the output; a `bucket`-sized `-inf`
    #: mask needs the dynamic length, `seqused_k` is silently ignored by the
    #: dense flash path, and the varlen path changes the kernel (4.9e-4 output
    #: drift) at 2x the cost.  Capturing per length keeps attention identical
    #: to `fast.py`'s call while still collapsing a step's ~2300 kernels into
    #: one replay.
    def bucket_for(self, length: int) -> int:
        """This design needs the graph's attention length to equal the cache
        length, so the "bucket" is the length itself."""
        return length

    def _capture(self, length: int, audio2: bool) -> torch.cuda.CUDAGraph:
        fn = self._step_fn(audio2, length)
        s = self._capture_stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool, stream=s):
            fn()
        torch.cuda.synchronize()
        return g

    # ------------------------------------------------------------------- api
    def prefill(self, input_ids: torch.Tensor):
        """Run the prefill with this module's own primitives (the fused weights
        make `base.prefill` unusable once `fuse_gemms` has run)."""
        m = self.m
        if input_ids.dim() != 3 or input_ids.shape[0] != 1 \
                or input_ids.shape[-1] != N_VQ + 1:
            raise ValueError(f"expected [1, L, {N_VQ + 1}] input ids, "
                             f"got {tuple(input_ids.shape)}")
        t = int(input_ids.shape[1])
        if t > m.max_seq_len:
            raise ValueError(f"seq len {t} exceeds KV cache {m.max_seq_len}")
        m.reset()
        m.k_cache.zero_()
        m.v_cache.zero_()
        ids = input_ids.to(device=m.device, dtype=torch.long)
        h = F.embedding(ids[..., 0], m.embed_tokens)
        for i in range(N_VQ):
            h = h + F.embedding(ids[..., i + 1], m.emb_ext[i])
        for li in range(m.n_layers):
            h = self._layer_prefill(li, h, t)
        h = _rms_norm(h, m.final_norm)
        m._seq = t
        self.seq_len = t
        # NOTE: captured graphs are *not* invalidated here.  They are keyed by
        # (cache length, phase) and read the cache/position through buffers, so
        # a new utterance at the same lengths reuses them -- which is what makes
        # per-length capture affordable across calls (the first call pays the
        # capture cost, later ones replay only).
        self.set_fsm(0, _INT64_MAX, False)
        return HiddenState(h)

    def _layer_prefill(self, li: int, h: torch.Tensor, t: int) -> torch.Tensor:
        """One layer over the whole prompt (mirrors `fast._layer_q`).

        Norms deliberately use the *reference* `_rms_norm` chain, not
        `F.rms_norm`: the prefill sets the KV cache and the prompt's last
        hidden state for the whole decode, so any per-step numerical difference
        here compounds into a different trajectory (measured: 5.0 hidden
        difference -> 164 vs 133 steps).  The decode step may use the faster
        kernel because its gate is per-step quality, but the prefill must match
        `fast.py` bit-for-bit or the two paths decode different utterances.
        """
        m = self.m
        lyr = m.layers[li]
        residual = h
        x = _rms_norm(h, lyr["input_layernorm"])
        if self.qk_fused[li] is not None:
            qk = self._lin_fused(x, li, "qk")
            q = qk[..., :m.n_heads * m.head_dim]
            k = qk[..., m.n_heads * m.head_dim:]
        else:
            q = self._lin(x, li, "q")
            k = self._lin(x, li, "k")
        v = self._lin(x, li, "v")
        q = _rms_norm(q.view(1, t, m.n_heads, m.head_dim),
                      lyr["q_norm"]).transpose(1, 2)
        k = _rms_norm(k.view(1, t, m.n_kv_heads, m.head_dim),
                      lyr["k_norm"]).transpose(1, 2)
        v = v.view(1, t, m.n_kv_heads, m.head_dim).transpose(1, 2)
        cos = m.rope_cos[:t].unsqueeze(0)
        sin = m.rope_sin[:t].unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        m.k_cache[li, :, :t] = k[0]
        m.v_cache[li, :, :t] = v[0]
        keys = k.repeat_interleave(m.n_rep, dim=1)
        vals = v.repeat_interleave(m.n_rep, dim=1)
        o = F.scaled_dot_product_attention(q, keys, vals,
                                           is_causal=(t > 1))
        h = residual + self._lin(
            o.transpose(1, 2).reshape(1, t, m.hidden_size), li, "o")
        residual = h
        x = _rms_norm(h, lyr["post_attention_layernorm"])
        if self.gu_fused[li] is not None:
            gu = self._lin_fused(x, li, "gu")
            inter = self.gu_fused[li][3]
            g, u = gu[..., :inter], gu[..., inter:]
        else:
            g = self._lin(x, li, "gate")
            u = self._lin(x, li, "up")
        return residual + self._lin(F.silu(g) * u, li, "down")

    def replay(self, pos: int, audio2: bool, row: torch.Tensor | None = None) -> None:
        """One whole-step graph replay; `pos` = cache slot the fed row occupies."""
        length = pos + 1
        if length < 1 or length > self.m.max_seq_len:
            raise ValueError(f"position {pos} outside the KV cache")
        if row is not None:
            self.ids_row.copy_(row if row.dim() == 3 else row.view(1, 1, -1))
        self.pos_dev.fill_(pos)
        key = (length, audio2)
        g = self.graphs.get(key)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                # graphs share one pool, so an evicted graph's memory returns
                # to the pool for the next capture
                self.graphs.pop(next(iter(self.graphs)))
            g = self._capture(length, audio2)
            self.graphs[key] = g
        g.replay()

    def graph_count(self) -> int:
        return len(self.graphs)

    def read_hsync(self):
        """One small D2H: (next_text, audio[N_VQ], al, dl, is_audio)."""
        v = self.hsync.cpu()
        return (int(v[0]), v[1:1 + N_VQ].clone(), int(v[1 + N_VQ]),
                int(v[2 + N_VQ]), int(v[3 + N_VQ]))

    def set_fsm(self, al: int, dl: int, is_audio: bool) -> None:
        self.fsm[0] = al
        self.fsm[1] = dl
        self.fsm[2] = 1 if is_audio else 0


# --------------------------------------------------------------- generation
@torch.inference_mode()
def generate_native(
    native: FastNativeTTS,
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
    """Delay-pattern generation on `FastNativeTTS`.

    Semantically mirrors `moss_tts_lite.fast.generate_fast` (same forced-token
    ramp, same masks, same watchdog); the host mirror of the delay FSM is
    authoritative and is pushed into the device FSM before every replay, so the
    two cannot drift.  In arm n3 the audio-phase step is resolved on device and
    one small D2H read supplies the host with the emitted row, the FSM counters
    and ch0 for the watchdog.
    """
    model = native.m
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
            raise NotImplementedError("batch=1 generate_native() needs an all-True mask")
    if int(input_ids.shape[1]) + max_new_tokens > model.max_seq_len:
        raise ValueError(f"prompt {int(input_ids.shape[1])} + max_new_tokens "
                         f"{max_new_tokens} exceeds KV cache {model.max_seq_len}")
    if greedy:
        raise NotImplementedError("native path samples; use fast.py for greedy")
    if audio_repetition_penalty != 1.0:
        raise NotImplementedError("native path assumes audio_repetition_penalty == 1.0")

    native.text_temperature = text_temperature
    native.text_top_p = text_top_p
    native.text_top_k = text_top_k
    native.audio_temperature = audio_temperature
    native.audio_top_p = audio_top_p
    native.audio_top_k = audio_top_k
    in_graph = native.in_graph_sample

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

    torch.manual_seed(seed)

    # ---- host mirror of the delay FSM --------------------------------------
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

    text_steps: list = []
    audio_steps: list = []
    n_steps = 0
    pos = seq_len
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
        la_step = None
        two_step = None
        if time_step == 0:
            hs = native.prefill(input_ids)
            h = hs.last_hidden[:, -1]
        else:
            native.set_fsm(al, dl, is_audio)
            native.replay(pos, audio2=bool(is_audio))
            pos += 1
            if in_graph and is_audio:
                # arm n3: the graph sampled the whole row and advanced the
                # device FSM, so both the row and the counters come from it.
                # The watchdog must still run here: it is what bounds a
                # degenerate trajectory, and skipping it is what made n3 run
                # away to thousands of steps.
                _t1(e0, "step")
                next_text, audio_row, al_dev, dl_dev, au_dev = native.read_hsync()
                al, dl, is_audio = al_dev, dl_dev, bool(au_dev)
                is_stopping = bool(next_text == IM_END_TOKEN_ID)
                ch0_val = int(audio_row[0])
                next_audio_tokens = audio_row.view(1, N_VQ).to(device)
                n_steps += 1
                if watchdog and not wd_stop and dl == _INT64_MAX:
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
                        wd_stats = {"reason": wd_reason, "step": n_steps,
                                    "segment_frames": al, "silent_run": wd_silent_run,
                                    "peak_completed_segment": wd_peak_segment,
                                    "silence_threshold_frames": wd_sil,
                                    "segment_bound_frames": seg_bound,
                                    "max_requested_pause_s": max_requested_pause_s}
                if next_text == AUDIO_END_TOKEN_ID:
                    wd_peak_segment = max(wd_peak_segment, al)
                    wd_silent_run = 0
                text_steps.append(int(next_text))
                audio_steps.append(next_audio_tokens[0].clone())
                if is_stopping:
                    break
                if wd_stop:
                    # drive the delay ramp to audio_end the same way the host
                    # path does, by forcing the row and letting the FSM run
                    native.ids_row[..., 0] = (
                        AUDIO_DELAY_SLOT_TOKEN_ID if (dl == _INT64_MAX or dl < N_VQ)
                        else AUDIO_END_TOKEN_ID)
                    native.ids_row[..., 1:] = AUDIO_PAD_CODE
                continue
            # The whole-step graph computes EVERY head, so both readouts are
            # always available from this replay; take both unconditionally.
            # (An earlier revision only took `two_step` in the audio phase,
            # which left `la_step = None` and made the audio tokens fall back
            # to `model.audio_logits(h)` on the *prefill* hidden state -- the
            # stale-head bug behind the audio rows diverging from step 1.)
            two_step = native.two_buf
            la_step = native.la_buf
        _t1(e0, "prefill" if time_step == 0 else "step")

        # ---- text token decision (verbatim mirror of generate_fast) --------
        next_text, is_audio, sampling_text, _forced = text_decision(
            dl, N_VQ, wd_stop, is_stopping, is_audio)

        if sampling_text:
            if is_audio:
                two_raw = two_step if two_step is not None \
                    else F.linear(h[0], model.head_2way)
                # bf16 division, as the reference does (fp32 here changes the
                # sampled token and hence the RNG stream)
                two = two_raw / text_temperature
                if time_step == 0:
                    two = two.clone()
                    two[1] = float("-inf")
                lt_buf = native.lt_buf
                lt_buf.fill_(float("-inf"))
                lt_buf[AUDIO_GEN_SLOT_TOKEN_ID] = two[0]
                lt_buf[AUDIO_DELAY_SLOT_TOKEN_ID] = two[1]
                tok = sample_token(lt_buf.view(1, -1), top_p=text_top_p,
                                   top_k=text_top_k, do_sample=True)
                next_text = int(tok[0])
            else:
                lt = model.text_logits(h) if time_step == 0 else native.lt_buf
                lt = lt / text_temperature
                lt = lt.index_fill(0, pre_exclude_mask0, float("-inf"))
                if time_step == 0:
                    lt = lt.clone()
                    lt[AUDIO_DELAY_SLOT_TOKEN_ID] = float("-inf")
                if time_step <= N_VQ:
                    lt = lt.clone()
                    lt[IM_END_TOKEN_ID] = float("-inf")
                tok = sample_token(lt.view(1, -1), top_p=text_top_p,
                                   top_k=text_top_k, do_sample=True)
                next_text = int(tok[0])

        if next_text == AUDIO_START_TOKEN_ID:
            is_audio = True
        if next_text == IM_END_TOKEN_ID:
            is_stopping = True

        # ---- audio tokens ---------------------------------------------------
        smask = audio_sampling_mask(al, dl, N_VQ)
        next_audio_tokens = torch.full((1, N_VQ), AUDIO_PAD_CODE, device=device,
                                       dtype=torch.long)
        if any(smask):
            la = la_step if la_step is not None else model.audio_logits(h)
            audio_logit = la / audio_temperature
            if smask[0]:
                ch0 = audio_logit[0].view(1, -1).clone()
                tok0 = sample_token(ch0, prev_tokens=None, repetition_penalty=1.0,
                                    top_p=audio_top_p, top_k=audio_top_k,
                                    do_sample=True)
                next_audio_tokens[0, 0] = int(tok0[0])
            rest = [j for j in range(1, N_VQ) if smask[j]]
            if rest:
                rest_l = audio_logit[rest].clone()
                tok = sample_token(rest_l, prev_tokens=None, repetition_penalty=1.0,
                                   top_p=audio_top_p, top_k=audio_top_k,
                                   do_sample=True)
                for k, j in enumerate(rest):
                    next_audio_tokens[0, j] = int(tok[k])
        ch0_val = int(next_audio_tokens[0, 0])

        # ---- watchdog -------------------------------------------------------
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

        # ---- counters (reference order) -------------------------------------
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

        cur = torch.cat([torch.tensor([[next_text]], device=device),
                         next_audio_tokens.view(1, -1)], dim=1)
        native.ids_row.copy_(cur.view(1, 1, N_VQ + 1))

        text_steps.append(int(next_text))
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
