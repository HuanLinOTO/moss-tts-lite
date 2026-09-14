"""Fast-path tests: CUDA-graph tier (fast.py) and the native whole-step tier.

Phase 0/1/2 M1 gates (bitwise step parity, golden-exact trajectory,
speed) plus the fast-native A/B/C gates: n1 bitwise vs fast.py, the
n2 quality floor, and regressions for the three bugs found while
building it.  GPU, under the lock.

Consolidated from:
  test_fast.py
  test_fast_native.py

Run:
  PYTHONPATH=. MOSS_TTS_ROOT=/root/MOSS-TTS python3 tests/test_fast.py
"""

from __future__ import annotations

import os
import sys

# `python3 tests/<this file>.py` straight from the repo root must import
# moss_tts_lite without an explicit PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from moss_tts_lite.fast import FastMossTTS, generate_fast
from moss_tts_lite.fast_native import (
    FastNativeTTS,
    audio_sampling_mask,
    generate_native,
    text_decision,
)
from moss_tts_lite.generate import (
    AUDIO_DELAY_SLOT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    PAD_TOKEN_ID,
)
from moss_tts_lite.gptq import load_gptq_fast
from moss_tts_lite.model import N_VQ, MossTTSModel

try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:  # pragma: no cover
    from tests._mini_loader import read_safetensors_min as read_safetensors

# --------------------------------------------------------------------------- #
# Unified root resolution.  MOSS_TTS_ROOT points at the MOSS-TTS asset checkout
# (weights, tokenizers, golden assets); it defaults to this repo's parent, so a
# checkout that keeps ``models/`` beside ``tests/`` works unchanged.
# --------------------------------------------------------------------------- #
ROOT = os.environ.get("MOSS_TTS_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
STATE = os.path.join(MODEL_DIR, "gptq", "w1p.pt")
SEED = 1234


# ------------------------------------------------------------------------- #
# ---- from test_fast.py ----
# ------------------------------------------------------------------------- #
try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:  # pragma: no cover
    from tests._mini_loader import read_safetensors_min as read_safetensors


TEXT_TEMPERATURE = 1.5
TEXT_TOP_P = 1.0
TEXT_TOP_K = 50
AUDIO_TEMPERATURE = 1.7
AUDIO_TOP_P = 0.8
AUDIO_TOP_K = 25
AUDIO_REPETITION_PENALTY = 1.0


def _steps_ps(step_ms, skip=0):
    ms = step_ms[skip:]
    return 1000.0 / (sum(ms) / len(ms)) if ms else float("nan")


def phase0(model, fast, ids):
    print("=== Phase 0: bitwise step parity (graph replay vs eager) ===")
    with torch.inference_mode():
        hs = model.prefill(ids)
        L0 = int(model.seq_len)
        fast.capture()
        # feed the prompt's own tail tokens as teacher-forced rows
        rows = ids[0, -6:].clone()   # [6, 33]
        lt_e, la_e, h_e = [], [], []
        for t in range(rows.shape[0]):
            hs = model.step(rows[t].view(1, 1, 33))
            h = hs.last_hidden
            lt_e.append(model.text_logits(h).clone())
            la_e.append(model.audio_logits(h).clone())
            h_e.append(h.clone())
        # rewind: prefill again and replay with the fast path
        model.prefill(ids)
        lt_g, la_g, h_g = [], [], []
        pos = L0
        for t in range(rows.shape[0]):
            lt, la, h, _two = fast.step(rows[t].view(1, 33), pos)
            pos += 1
            lt_g.append(lt.clone())
            la_g.append(la.clone())
            h_g.append(h.view(1, 1, -1).clone())
    ok = True
    for t in range(rows.shape[0]):
        bt = torch.equal(lt_e[t], lt_g[t])
        ba = torch.equal(la_e[t], la_g[t])
        bh = torch.equal(h_e[t], h_g[t])
        d = max((a.float() - b.float()).abs().max().item()
                for a, b in ((lt_e[t], lt_g[t]), (la_e[t], la_g[t]), (h_e[t], h_g[t])))
        print(f"  step {t} (seq={L0 + t + 1}): "
              f"h={bh} text={bt} audio={ba} max|d|={d:.3e}")
        ok = ok and bt and ba and bh
    print(f"  phase0: {'PASS (bitwise)' if ok else 'FAIL'}")
    return ok


def _run_case(model, fast, case, label, stats=None):
    from moss_tts_lite.generate import generate
    ids = case["input_ids"].cuda() if "input_ids" in case else case["ids"].cuda()
    # prompt_golden entries: {"input_ids", "attention_mask", "name"}
    mask = case["attention_mask"].cuda() if "attention_mask" in case else None
    gt = case["generation_ids"]
    res = generate_fast(
        fast, {"input_ids": ids, "attention_mask": mask},
        max_new_tokens=4096, seed=SEED,
        text_temperature=TEXT_TEMPERATURE, text_top_p=TEXT_TOP_P,
        text_top_k=TEXT_TOP_K, audio_temperature=AUDIO_TEMPERATURE,
        audio_top_p=AUDIO_TOP_P, audio_top_k=AUDIO_TOP_K,
        audio_repetition_penalty=AUDIO_REPETITION_PENALTY, stats=stats)
    t = res.text_ids
    a = res.audio_frames
    T = gt.shape[0]
    exact = (t.numel() == T and a.shape == gt[:, 1:].shape
             and torch.equal(t.reshape(-1), gt[:, 0])
             and torch.equal(a, gt[:, 1:]))
    t_match = (t.reshape(-1) == gt[:, 0]).float().mean().item() * 100 if t.numel() == T else float("nan")
    a_match = (a == gt[:, 1:]).float().mean().item() * 100 if a.shape == gt[:, 1:].shape else float("nan")
    print(f"  [{label}] T mine={t.numel()} golden={T} n_steps={res.n_steps} "
          f"finished={res.finished} (golden {case['n_steps']}/{case['finished']})")
    print(f"  [{label}] text={t_match:.2f}% audio={a_match:.2f}% EXACT={exact}")
    if not exact:
        L = min(t.numel(), T)
        fd = next((j for j in range(L)
                   if t.reshape(-1)[j] != gt[j, 0] or not torch.equal(a[j], gt[j, 1:])), L)
        print(f"  [{label}] first diverging row {fd}")
    return res, exact


def main_fast() -> int:
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    pg = torch.load(os.path.join(GOLDEN, "prompt_golden.pt"),
                    map_location="cpu", weights_only=False)
    gg = torch.load(os.path.join(GOLDEN, "gen_golden.pt"),
                    map_location="cpu", weights_only=False)
    names = [c["name"] for c in pg["cases"]]
    zh_i = names.index("zh_plain")
    en_i = names.index("en_language")

    print("loading weights ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()
    fast = FastMossTTS(model)
    print(f"weights+cache on device: "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    ok0 = phase0(model, fast, pg["input_ids"][zh_i].cuda())

    zh_case = dict(pg["cases"][zh_i])
    zh_case["input_ids"] = pg["input_ids"][zh_i]
    zh_case["attention_mask"] = pg["attention_mask"][zh_i]
    zh_case.update(next(c for c in gg["cases"] if c["case_name"] == "zh_plain"))
    en_case = dict(pg["cases"][en_i])
    en_case["input_ids"] = pg["input_ids"][en_i]
    en_case["attention_mask"] = pg["attention_mask"][en_i]
    en_case.update(next(c for c in gg["cases"] if c["case_name"] == "en_language"))

    print("=== Phase 1: full-trajectory gate (seed=1234, default sampling) ===")
    print("--- zh (graph capture happens inside the zh run) ---", flush=True)
    torch.cuda.reset_peak_memory_stats()
    res_zh, ok_zh = _run_case(model, fast, zh_case, "zh")
    print("--- en ---", flush=True)
    res_en, ok_en = _run_case(model, fast, en_case, "en")
    print(f"VRAM peak after generations: "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    # ---------- speed: rerun zh with per-step events ----------
    print("=== Phase 2: speed (zh, 137 steps) ===")
    stats: dict = {}
    res_s, ok_s = _run_case(model, fast, zh_case, "zh-speed", stats=stats)
    prefill_ms = stats.get("prefill_ms")
    step_ms = stats.get("step_ms") or []
    print(f"prefill: {prefill_ms:.1f} ms (L={int(zh_case['input_ids'].shape[1])})")
    print(f"decode avg: {sum(step_ms) / len(step_ms):.2f} ms/step over {len(step_ms)} "
          f"steps -> {_steps_ps(step_ms):.1f} steps/s")
    print(f"decode steady (10+): {_steps_ps(step_ms, 10):.1f} steps/s "
          f"({sum(step_ms[10:]) / len(step_ms[10:]):.2f} ms/step)")
    print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    ok = ok0 and ok_zh and ok_en and ok_s
    print(f"test_fast M1: phase0={'PASS' if ok0 else 'FAIL'} "
          f"zh={'EXACT' if ok_zh else 'DIVERGED'} en={'EXACT' if ok_en else 'DIVERGED'} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ------------------------------------------------------------------------- #
# ---- from test_fast_native.py ----
# ------------------------------------------------------------------------- #


_INT64_MAX = 9223372036854775807

#: n2 gate: the shipped W4 path scores 75.39% on the same metric
#: (tests/test_fast_m4 Phase A), so requiring 97% is a floor that both
#: protects quality and would catch any real regression in the fused kernels.
TOP25_GATE = 97.0
TEXT_ARGMAX_GATE = 99.0


def _fresh_base():
    """A model + GPTQ fast container.  `load_gptq_fast` consumes the packed
    payloads (`fuse_gemms` drops the per-name originals), so every arm needs
    its own base -- reusing one across arms raises KeyError('q')."""
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    base = load_gptq_fast(model, STATE)
    torch.cuda.empty_cache()
    return model, base


def _golden():
    pg = torch.load(os.path.join(GOLDEN, "prompt_golden.pt"),
                    map_location="cpu", weights_only=False)
    z = np.load(os.path.join(GOLDEN, "logits_golden.npz"))
    return pg, z


# --------------------------------------------------------------- Phase A
def phase_a(pg, n_positions=40, boundary=(690, 726)):
    """n1 == fast.py, bitwise, over positions and across the gqa boundary."""
    print("=== Phase A: n1 bitwise vs fast.py ===")
    ids = pg["input_ids"][[c["name"] for c in pg["cases"]].index("zh_plain")].cuda()
    row = ids[0, -1:].view(1, 1, 33)

    model, base = _fresh_base()
    nat = FastNativeTTS(base, arm="n1", max_graphs=512)
    L0 = int(ids.shape[1])
    ok = True
    worst = 0.0
    with torch.inference_mode():
        # --- both phases, a run of positions -------------------------------
        hs = nat.prefill(ids)
        base.prefill(ids)
        base.capture()
        for t in range(n_positions):
            pos = L0 + t
            for audio2 in (False, True):
                # reference: fast.py's own pieces at the same position/row
                base.ids_row.copy_(row)
                base.pos_dev.fill_(pos)
                base._p_embed()
                base._p_pre(0)
                for li in range(model.n_layers):
                    use_gqa = base.enable_gqa and (pos + 1) <= base.gqa_max_len
                    if use_gqa:
                        o = torch.nn.functional.scaled_dot_product_attention(
                            base.q_static, model.k_cache[li, :, :pos + 1].unsqueeze(0),
                            model.v_cache[li, :, :pos + 1].unsqueeze(0), enable_gqa=True)
                    else:
                        o = torch.nn.functional.scaled_dot_product_attention(
                            base.q_static,
                            model.k_cache[li, :, :pos + 1].repeat_interleave(
                                model.n_rep, dim=0).unsqueeze(0),
                            model.v_cache[li, :, :pos + 1].repeat_interleave(
                                model.n_rep, dim=0).unsqueeze(0))
                    base.o_static.view(1, model.n_heads, model.head_dim).copy_(
                        o.view(1, model.n_heads, model.head_dim))
                    base._p_post(li)
                    if li + 1 < model.n_layers:
                        base._p_pre(li + 1)
                (base._p_final_audio2 if audio2 else base._p_final)()
                ref = (base.lt_static.clone(), base.two_static.clone(),
                       base.la_static.clone())
                nat.replay(pos, audio2=audio2, row=row)
                got = (nat.lt_buf, nat.two_buf, nat.la_buf)
                # the pad column is -inf in both streams (-inf - -inf = nan),
                # so compare the real 1024 codes and the pad column separately
                for a, b in ((ref[0], got[0]), (ref[1], got[1]),
                             (ref[2][..., :1024], got[2][..., :1024])):
                    d = (a.float() - b.float()).abs().max().item()
                    worst = max(worst, d)
                    ok &= d == 0.0
                ok &= bool((ref[2][..., 1024] == float("-inf")).all()
                           and (got[2][..., 1024] == float("-inf")).all())
        print(f"  {n_positions} positions x 2 phases: text/two/audio all "
              f"max|d| = {worst:.1e}  ->  {'PASS' if ok else 'FAIL'}")

        # --- across the gqa_max_len switch (len 641 changes flash tiling) ---
        bad = 0
        for pos in range(boundary[0], boundary[1]):
            base.ids_row.copy_(row)
            base.pos_dev.fill_(pos)
            base._p_embed()
            base._p_pre(0)
            for li in range(model.n_layers):
                use_gqa = base.enable_gqa and (pos + 1) <= base.gqa_max_len
                if use_gqa:
                    o = torch.nn.functional.scaled_dot_product_attention(
                        base.q_static, model.k_cache[li, :, :pos + 1].unsqueeze(0),
                        model.v_cache[li, :, :pos + 1].unsqueeze(0), enable_gqa=True)
                else:
                    o = torch.nn.functional.scaled_dot_product_attention(
                        base.q_static,
                        model.k_cache[li, :, :pos + 1].repeat_interleave(
                            model.n_rep, dim=0).unsqueeze(0),
                        model.v_cache[li, :, :pos + 1].repeat_interleave(
                            model.n_rep, dim=0).unsqueeze(0))
                base.o_static.view(1, model.n_heads, model.head_dim).copy_(
                    o.view(1, model.n_heads, model.head_dim))
                base._p_post(li)
                if li + 1 < model.n_layers:
                    base._p_pre(li + 1)
            base._p_final()
            ref_lt = base.lt_static.clone()
            nat.replay(pos, audio2=False, row=row)
            if (ref_lt.float() - nat.lt_buf.float()).abs().max().item() != 0.0:
                bad += 1
        print(f"  gqa boundary len {boundary[0] + 1}-{boundary[1]}: "
              f"{bad}/{boundary[1] - boundary[0]} mismatches  ->  "
              f"{'PASS' if bad == 0 else 'FAIL'}")
        ok &= bad == 0
        del hs
    del nat, base, model
    torch.cuda.empty_cache()
    return ok


# --------------------------------------------------------------- Phase B
def phase_b(pg, z):
    """n2 gates: teacher-forced quality, no runaway, normal termination."""
    print("=== Phase B: n2 quality gate ===")
    gt_text = torch.from_numpy(z["logits_text"]).cuda()
    gt_audio = torch.from_numpy(z["logits_audio"]).cuda()[..., :1024]
    gt_rows = torch.from_numpy(z["selected_rows"]).reshape(-1, 33).cuda()
    names = [c["name"] for c in pg["cases"]]
    ids = pg["input_ids"][names.index("zh_plain")].cuda()

    model, base = _fresh_base()
    nat = FastNativeTTS(base, arm="n2", max_graphs=256)
    with torch.inference_mode():
        hs = nat.prefill(ids)
        L0 = int(model.seq_len)
        mt = [model.text_logits(hs.last_hidden[:, -1]).float()]
        ma = [model.audio_logits(hs.last_hidden[:, -1]).float()]
        for t in range(1, gt_rows.shape[0]):
            nat.replay(L0 + t - 1, audio2=False, row=gt_rows[t - 1].view(1, 1, 33))
            mt.append(nat.lt_buf.float())
            ma.append(nat.la_buf.float())
        mt = torch.stack(mt)
        ma = torch.stack(ma)[..., :1024]

    argmax = (mt.argmax(-1) == gt_text.argmax(-1)).float().mean().item() * 100
    gold_val = gt_audio.gather(-1, gt_audio.argmax(-1, keepdim=True))
    kth = ma.topk(25, dim=-1).values[..., -1:]
    top25 = (gold_val >= kth).float().mean().item() * 100
    print(f"  teacher-forced 40 steps: text argmax {argmax:.2f}% "
          f"(gate>={TEXT_ARGMAX_GATE:.0f})  top-25 {top25:.2f}% "
          f"(gate>={TOP25_GATE:.0f})")
    ok = argmax >= TEXT_ARGMAX_GATE and top25 >= TOP25_GATE

    # ---- generation: termination, watchdog, sane length -------------------
    # Both languages, both arms.  fast.py's own counts are the reference for
    # "normal termination"; n2 may differ (different kernels, different
    # utterance) but must stay in the same ballpark and must not loop.
    del nat, base, model
    torch.cuda.empty_cache()
    model, base = _fresh_base()
    base.prefill(pg["input_ids"][names.index("zh_plain")].cuda())
    base.capture()
    ref_steps = {}
    for case in ("zh_plain", "en_language"):
        r = generate_fast(base, {"input_ids": pg["input_ids"][names.index(case)].cuda()},
                          max_new_tokens=4096, seed=SEED)
        ref_steps[case] = r.n_steps
        assert r.finished and not r.watchdog_triggered, (case, r.watchdog_reason)
    del base, model
    torch.cuda.empty_cache()

    for arm, max_graphs in (("n1", 512), ("n2", 256)):
        model, base = _fresh_base()
        nat = FastNativeTTS(base, arm=arm, max_graphs=max_graphs)
        for case in ("zh_plain", "en_language"):
            r = generate_native(nat, {"input_ids": pg["input_ids"][names.index(case)].cuda()},
                                max_new_tokens=4096, seed=SEED)
            n_ref = ref_steps[case]
            # a runaway shows up as a step count far past the reference; the
            # 2x bound is loose on purpose (n2 decodes a different utterance)
            sane = r.finished and not r.watchdog_triggered \
                and r.n_steps <= 2 * n_ref and r.audio_frames.shape[0] > 0
            print(f"  [{arm}] {case:13s} steps={r.n_steps:4d} "
                  f"(fast.py {n_ref}) finished={r.finished} wd={r.watchdog_triggered} "
                  f"rows={r.audio_frames.shape[0]}  ->  {'OK' if sane else 'BAD'}")
            ok &= sane
        del nat, base, model
        torch.cuda.empty_cache()
    return ok


# --------------------------------------------------------------- Phase C
def phase_c(pg):
    """Behavioural regressions for the three bugs (see the module docstring)."""
    print("=== Phase C: bug regressions ===")
    names = [c["name"] for c in pg["cases"]]
    ids = pg["input_ids"][names.index("zh_plain")].cuda()
    row = ids[0, -1:].view(1, 1, 33)
    L0 = int(ids.shape[1])
    ok = True

    # ---- C1: the loop must consume the graph's fresh head readouts --------
    # The bug: in the audio phase only `two_buf` was taken, so `la_step` stayed
    # None and `generate_native` fell back to `model.audio_logits(h)` -- which
    # recomputes the audio heads from the hidden state, and after the prefill
    # step that hidden state is stale (it is not carried by the native loop at
    # all).  Result: every audio row was wrong from step 1 while the text
    # stream stayed right until step 105.
    #
    # This is asserted on the *consumption* path, not on the buffer: the whole
    # point of the whole-step graph is that it computes every head, so the loop
    # must never recompute an audio head.  `audio_logits` is therefore made to
    # raise, and generation must still succeed -- reintroducing the fallback
    # makes it raise.  (`text_logits` is left alone: the loop legitimately
    # calls it once at time_step 0.)
    model, base = _fresh_base()
    nat = FastNativeTTS(base, arm="n1", max_graphs=64)
    calls = {"audio_logits": 0}
    real_audio_logits = MossTTSModel.audio_logits

    def _forbid_audio_logits(self, h):                     # noqa: ANN001
        calls["audio_logits"] += 1
        raise AssertionError(
            "generate_native() recomputed audio heads instead of using the "
            "graph's la_buf (the C1 stale-head bug)")

    MossTTSModel.audio_logits = _forbid_audio_logits
    try:
        r = generate_native(nat, {"input_ids": ids}, max_new_tokens=24, seed=SEED)
        no_recompute = True
    except AssertionError as exc:
        print(f"  C1 FAIL: {exc}")
        no_recompute = False
        r = None
    finally:
        MossTTSModel.audio_logits = real_audio_logits
    # also assert the buffers the loop relies on are present and genuinely
    # different from the stale-prefill values (so the fallback is detectable)
    with torch.inference_mode():
        hs_pre = nat.prefill(ids)
        prefill_la = model.audio_logits(hs_pre.last_hidden[:, -1]).float()[..., :1024]
        nat.replay(L0, audio2=False, row=row)
        step_la = nat.la_buf[..., :1024].float().clone()
        step_two = nat.two_buf.float().clone()
    both_live = bool(torch.isfinite(step_la).all()) \
        and bool(torch.isfinite(step_two).all())
    distinguishable = (prefill_la - step_la).abs().max().item() > 0.0
    c1_ok = no_recompute and both_live and distinguishable
    print(f"  C1 graph readouts consumed (audio_logits never called by the "
          f"loop): {no_recompute}; both live: {both_live}; stale fallback "
          f"would differ: {distinguishable}  ->  {'PASS' if c1_ok else 'FAIL'}")
    ok &= c1_ok
    if r is not None:
        ok &= r.n_steps > 0
    del nat, base, model
    torch.cuda.empty_cache()

    # ---- C2: prefill uses the reference rms_norm chain --------------------
    # prefill must be bitwise-equal to fast.py's own quantized prefill, because
    # it fixes the KV cache for the whole decode.  A fast-norm prefill shows up
    # here as a nonzero hidden/KV difference (measured 5.0 / 2.0).
    #
    # Order matters: `FastNativeTTS(arm="n2")` fuses `(q,k)` and `(gate,up)`
    # and drops the per-name payloads, after which `base.prefill()` can no
    # longer run (KeyError 'q').  So take the reference prefill FIRST, from the
    # untouched base, and only then build the native object.
    model, base = _fresh_base()
    with torch.inference_mode():
        hs_ref = base.prefill(ids)
        ref_h = hs_ref.last_hidden.clone()
        ref_k = model.k_cache[:, :, :L0].clone()
        ref_v = model.v_cache[:, :, :L0].clone()
        nat = FastNativeTTS(base, arm="n2", max_graphs=64)   # n2 = risky arm
        hs = nat.prefill(ids)
        d_h = (ref_h.float() - hs.last_hidden.float()).abs().max().item()
        d_k = (ref_k.float() - model.k_cache[:, :, :L0].float()).abs().max().item()
        d_v = (ref_v.float() - model.v_cache[:, :, :L0].float()).abs().max().item()
    prefill_ok = d_h == 0.0 and d_k == 0.0 and d_v == 0.0
    print(f"  C2 prefill bitwise vs fast.py (n2 arm): hidden max|d|={d_h:.1e} "
          f"kv max|d|={d_k:.1e}/{d_v:.1e}  ->  {'PASS' if prefill_ok else 'FAIL'}")
    ok &= prefill_ok
    del nat, base, model
    torch.cuda.empty_cache()

    # ---- C3: the delay-ramp decision must match fast.py branch for branch -
    # The bug: the two forced branches (`dl < N_VQ` -> delay slot,
    # `dl == N_VQ` -> audio end) were taken unconditionally, dropping
    # `fast.py`'s `not is_stopping` guards.  Once `IM_END` has been decided the
    # reference stops forcing, and `sampling_text` also switches the whole
    # sampling block off -- so an unguarded implementation keeps forcing, and
    # (`sampling_text` disagreeing) keeps consuming RNG where the reference
    # does not.
    #
    # Pinned as a *differential* table against `fast.py`'s own branch text:
    # `_reference_text_decision` below is a transcription of
    # `generate_fast`'s `if wd_stop / elif not is_stopping and ...` block, and
    # every combination in the cross product must agree on all four outputs.
    # This is exact and cheap (no GPU), and it fails immediately if either side
    # changes -- which is the property a step-count assertion cannot give.
    bad = []
    for dl in (_INT64_MAX, N_VQ - 1, N_VQ, N_VQ + 1):
        for wd_stop in (False, True):
            for is_stopping in (False, True):
                for is_audio in (False, True):
                    got = text_decision(dl, N_VQ, wd_stop, is_stopping, is_audio)
                    ref = _reference_text_decision(dl, N_VQ, wd_stop, is_stopping,
                                                   is_audio)
                    if got != ref:
                        bad.append((dl, wd_stop, is_stopping, is_audio, got, ref))
    c3_ok = not bad
    print(f"  C3 text decision vs fast.py over 32 state combinations: "
          f"{len(bad)} mismatches  ->  {'PASS' if c3_ok else 'FAIL'}")
    for row in bad[:4]:
        print(f"      dl={row[0] if row[0] < 10**18 else 'MAX'} wd={row[1]} "
              f"stop={row[2]} audio={row[3]}: mine={row[4]} ref={row[5]}")
    ok &= c3_ok

    # the guards must actually be reachable: the unguarded variant differs
    unguarded_differs = any(
        (_unguarded_text_decision(dl, N_VQ, wd_stop, is_stopping, is_audio)
         != text_decision(dl, N_VQ, wd_stop, is_stopping, is_audio))
        for dl in (_INT64_MAX, N_VQ - 1, N_VQ, N_VQ + 1)
        for wd_stop in (False, True) for is_stopping in (False, True)
        for is_audio in (False, True))
    print(f"  C3 guards are observable (unguarded variant differs somewhere): "
          f"{unguarded_differs}  ->  {'PASS' if unguarded_differs else 'FAIL'}")
    ok &= unguarded_differs

    # and the audio mask helper must agree with the reference expression
    mask_bad = 0
    for al in (0, 1, N_VQ - 1, N_VQ, N_VQ + 3):
        for dl in (_INT64_MAX, 0, 1, N_VQ - 1, N_VQ, N_VQ + 1):
            want = [(al > j) and (dl == _INT64_MAX or j > dl - 1)
                    for j in range(N_VQ)]
            if audio_sampling_mask(al, dl, N_VQ) != want:
                mask_bad += 1
    print(f"  C3 audio sampling mask vs reference expression: {mask_bad} "
          f"mismatches  ->  {'PASS' if mask_bad == 0 else 'FAIL'}")
    ok &= mask_bad == 0
    return ok


def _reference_text_decision(dl, n_vq, wd_stop, is_stopping, is_audio):
    """`generate_fast`'s text-decision block, transcribed verbatim.

    Kept literally in the shape of `fast.py` (one branch per line, same order)
    so a reader can diff it against the source by eye; the test asserts the
    module's `text_decision` agrees with it over the whole state cross product.
    """
    next_text = PAD_TOKEN_ID
    if wd_stop:
        if dl == _INT64_MAX or dl < n_vq:
            next_text = AUDIO_DELAY_SLOT_TOKEN_ID
        else:
            next_text = AUDIO_END_TOKEN_ID
            is_audio = False
    elif not is_stopping and dl < n_vq:
        next_text = AUDIO_DELAY_SLOT_TOKEN_ID
    elif not is_stopping and dl == n_vq:
        next_text = AUDIO_END_TOKEN_ID
        is_audio = False
    sampling_text = (not is_stopping) and (not wd_stop) and dl > n_vq
    forced = next_text in (AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID)
    return next_text, is_audio, sampling_text, forced


def _unguarded_text_decision(dl, n_vq, wd_stop, is_stopping, is_audio):
    """The buggy variant: the same block without `not is_stopping`.

    Used only to prove the guards are observable; it must NOT match the
    reference for at least one state (the C3 pre-condition).
    """
    next_text = PAD_TOKEN_ID
    if wd_stop:
        if dl == _INT64_MAX or dl < n_vq:
            next_text = AUDIO_DELAY_SLOT_TOKEN_ID
        else:
            next_text = AUDIO_END_TOKEN_ID
            is_audio = False
    elif dl < n_vq:                       # guard dropped
        next_text = AUDIO_DELAY_SLOT_TOKEN_ID
    elif dl == n_vq:                      # guard dropped
        next_text = AUDIO_END_TOKEN_ID
        is_audio = False
    sampling_text = (not is_stopping) and (not wd_stop) and dl > n_vq
    forced = next_text in (AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID)
    return next_text, is_audio, sampling_text, forced


def main_fast_native() -> int:
    assert torch.cuda.is_available()
    if not os.path.exists(STATE):
        print(f"SKIP: GPTQ state {STATE} not present (4 GiB, not shipped)")
        return 0
    torch.cuda.reset_peak_memory_stats()
    pg, z = _golden()

    ok_a = phase_a(pg)
    ok_b = phase_b(pg, z)
    ok_c = phase_c(pg)

    vram = torch.cuda.max_memory_allocated() / 2**30
    ok = ok_a and ok_b and ok_c
    print(f"test_fast_native: A(bitwise n1)={'PASS' if ok_a else 'FAIL'} "
          f"B(n2 gates)={'PASS' if ok_b else 'FAIL'} "
          f"C(bug regressions)={'PASS' if ok_c else 'FAIL'} "
          f"peak={vram:.2f} GiB -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Unified entry point: each source file's own entry function, in order.
# --------------------------------------------------------------------------- #

def main() -> int:
    rc = 0
    rc |= main_fast() or 0
    rc |= main_fast_native() or 0

    return rc


if __name__ == "__main__":
    sys.exit(main())
