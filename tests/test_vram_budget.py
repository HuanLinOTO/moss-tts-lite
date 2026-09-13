"""VRAM budget tests — on-demand KV sizing + the 8 GB card target (kvfit).

`.tmp/reports/kvfit-1.md`.  Two independent halves:

  Phase 0  **Pure-arithmetic unit tests (no GPU, no weights).**  The KV size a
           run asks for is a pure function of three integers --
           ``(L0, max_new_tokens, user floor)`` -- so it is tested as one, over
           a matrix of the combinations that occur in practice (default CLI,
           explicit floor below/at/above the on-demand size, no floor, huge
           budget).  Also pins the resize identity that makes the on-demand
           cache value-preserving: an on-demand table must equal the tail of a
           table built at the old size, element for element.

  Phase 1  **GPU: measured peaks, as a regression gate.**  Loads the shipped
           w1p tier and asserts the *measured* VRAM ceiling of the fast path
           (weights + KV + graph pool + activations) for typical inputs, and
           asserts the 8 GB simulation (``set_per_process_memory_fraction``)
           passes end to end.  The numbers are the A10G-24G measurements from
           the report; they are deliberately loose ceilings (a regression gate,
           not a benchmark), because the allocator's reservation behaviour
           shifts between torch builds.

*** Why this file does not run nvidia-smi ***
`torch.cuda.max_memory_allocated()` cannot see the CUDA graph pool (`fast.py`
allocates its 38 sub-graphs in a private `graph_pool_handle`), so a gate built
on torch stats alone would miss the single largest non-weight item.  The
report's numbers come from `nvidia-smi` process memory; the gate here asserts
`max_memory_allocated` (a stable, comparable quantity that *includes* the KV
cache and weights) against ceilings anchored on those measurements, and asserts
the 8 GB simulation behaviourally (the run either fits in the cap or it does
not -- that is the real gate).

Run (GPU, under the lock):
  PYTORCH_ALLOC_CONF=expandable_segments:True flock /root/MOSS-TTS/.tmp/gpu.lock \
      python3 -m tests.test_vram_budget
"""

from __future__ import annotations

import gc
import os
import sys

import torch

from moss_tts_lite.cli import (
    DEFAULT_MAX_SEQ_LEN,
    KV_RAMP_MARGIN,
    _kv_mib,
    _resolve_max_seq_len,
)
from moss_tts_lite.fast import generate_fast
from moss_tts_lite.gptq import load_gptq_fast
from moss_tts_lite.model import MossTTSModel
from moss_tts_lite.prompt import build_tts_prompt

ROOT = os.environ.get("MOSS_TTS_ROOT", os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
STATE = os.path.join(MODEL_DIR, "gptq", "w1p.pt")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")

#: The 8 GB target is a *standalone* export: `base + w1p.pt` transiently
#: materializes the 16.6 GiB bf16 base to assemble the model (`_load_tts` ->
#: `MossTTSModel.__init__` copies every projection), which no 8 GB card can do
#: regardless of the KV cache.  The deployment path for a small card is the
#: self-contained directory (7.2 GiB on disk), where the int4 payload is
#: installed directly.  This test therefore measures BOTH, and the 8 GB
#: assertions apply to the standalone path.
STANDALONE = os.environ.get("MOSS_TTS_8GB_DIR", os.path.join(
    ROOT, ".tmp", "kvfit_agent", "standalone_w1"))

GIB = 2 ** 30
SEED = 1234
MAXNEW = 4096
OUT = os.path.join(ROOT, ".tmp", "kvfit_agent", "wav")

#: measured torch peaks on A10G-24G (kvfit-1 §2/§4).  Deliberately loose
#: ceilings: a regression gate, not a benchmark -- the allocator's reservation
#: behaviour moves between torch builds.
_PEAK_GIB = 7.90          # standalone w1p, fast path, on-demand KV, budget 4096
_PEAK_CODEC_GIB = 3.80    # MossCodecDecoder alone (measured 3.56)
_PEAK_BASE_GIB = 8.45     # base + w1p.pt (the heavier assembly path), measured 8.33

#: 8 GB simulation.  A `set_per_process_memory_fraction` budget is enforced by
#: torch's caching allocator, which counts *reserved* bytes; the CUDA context
#: (~0.29 GiB on A10G) lives outside it and is spent first.  So the faithful
#: simulation of an 8 GiB card is `fraction = (8.0 - context) / total`, and the
#: pass criterion is the *nvidia-smi* process peak (context + pool + tensors)
#: staying under 8.0 GiB.  Sizing the fraction as `7.5/24` instead would charge
#: the context inside the 7.5 and be stricter than any real 8 GiB card --
#: measured, w1p's weights alone (7.115) + context (0.285) = 7.40 > 7.36, so
#: that variant OOMs on the load itself, before any KV is allocated.
EIGHT_GB_CARD = 8.0

#: typical CLI inputs.  `budget` is the step budget each case is run with.
#: The primary budget is 4096 (the CLI default) but see `_PEAK_BUDGET`: at an
#: 8 GB *torch* cap the w1p tier only fits 2048 steps, so the 24 GiB peak gate
#: uses 4096 and the 8 GB gate uses the largest budget that tier actually fits.
CASES = [
    # name, text, language, budget
    ("zh12", "你好，欢迎收听这段试音。", None, MAXNEW),
    ("en", "Hello, this is a short test of the text to speech system.",
     "English", MAXNEW),
    ("pause", "我今天学习了一首中国的古诗，它的名字是[pause 8s]静夜思！", None, MAXNEW),
    ("long150",
     "人工智能正在改变我们的生活方式。从智能手机到自动驾驶，从医疗诊断到金融风控，"
     "机器学习算法已经渗透到各行各业。与此同时，人们也开始关注数据隐私、算法公平和"
     "就业结构等社会问题。算法推荐系统在提升信息获取效率的同时，也可能造成信息茧房"
     "效应。如何在个性化与公共性之间取得平衡，是这个时代需要认真思考的问题。",
     None, MAXNEW),
]

#: measured 8 GB boundary (kvfit-1 §4): on a simulated 8.0 GiB card the w1p
#: weights (7.12 GiB) + context (0.29) + pool (0.08) leave room for about 2048
#: steps of KV; 2560 already exceeds the card.  The 8 GB phase therefore runs
#: every case at this budget, which still covers ~164 s of audio.
EIGHT_GB_BUDGET = 2048


def _smi_total_mib() -> int:
    """Total nvidia-smi footprint of EVERY process on the device.

    `--query-compute-apps` reports host-namespace pids, which do not match
    `os.getpid()` inside this container, so a pid filter silently matches nothing
    (measured: nvidia-smi reported pid 768993 for a process whose `os.getpid()`
    was 217835, and such a filter returned 0 for a live 0.5 GiB context).
    Attribution is done by baseline subtraction instead: measure this before
    creating a CUDA context, subtract it from the peak, and what is left is this
    process's own context + tensors + graph pool.  That also correctly excludes a
    parent's still-resident context when phase 2 runs in child processes, which a
    raw sum gets wrong (it inflated the measured context from 0.285 to
    0.840 GiB and produced a false FAIL).
    """
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    total = 0
    for line in out.strip().splitlines():
        if line:
            total += int(line.split(",")[1].strip().split()[0])
    return total


#: nvidia-smi total at process start, before this process had a CUDA context
_SMI_BASELINE_MIB = 0


def _smi_mib() -> int:
    """This process's own nvidia-smi footprint (includes the graph pool)."""
    return _smi_total_mib() - _SMI_BASELINE_MIB


def _start_ctx_gib() -> float:
    """Create the CUDA context and return its cost in GiB.

    The baseline is taken *before* `torch.cuda.init()`, so the number is this
    process's own context -- not a neighbour's and not a parent's (see
    `_smi_total_mib`).
    """
    global _SMI_BASELINE_MIB
    _SMI_BASELINE_MIB = _smi_total_mib()
    torch.cuda.init()
    probe = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    del probe
    torch.cuda.empty_cache()
    return (_smi_total_mib() - _SMI_BASELINE_MIB) / 1024


# --------------------------------------------------------------- phase 0
def phase_0_sizing() -> bool:
    """The on-demand size arithmetic, over the combination matrix."""
    print("=== Phase 0: KV-size derivation (unit, no GPU) ===")
    ok = True
    rows = [
        # (requested floor, L0, max_new_tokens, expected, what it covers)
        (None, 66, 4096, 66 + 4096 + 64, "CLI default, short zh"),
        (None, 143, 4096, 143 + 4096 + 64, "CLI default, long zh"),
        (None, 71, 4096, 71 + 4096 + 64, "CLI default, en"),
        (None, 185, 4096, 185 + 4096 + 64, "continuation (120-frame prefix)"),
        (None, 66, 1024, 66 + 1024 + 64, "reduced budget"),
        (None, 66, 0, 66 + 64, "no generation at all"),
        (1024, 66, 4096, 66 + 4096 + 64, "floor BELOW the need: need wins"),
        (4226, 66, 4096, 4226, "floor EQUAL to the need: exact"),
        (8192, 66, 4096, 8192, "floor ABOVE the need: user wins (legacy size)"),
        (16384, 143, 4096, 16384, "floor far above: user wins"),
        (None, 1, 1, 1 + 1 + 64, "degenerate but sane"),
    ]
    for requested, l0, mnt, want, why in rows:
        got = _resolve_max_seq_len(requested, l0, mnt)
        good = got == want
        ok &= good
        print(f"  floor={str(requested):6s} L0={l0:4d} max_new={mnt:5d} -> "
              f"{got:6d} (want {want:6d}) {'ok' if good else 'FAIL'}  # {why}")
    # the ramp margin must actually cover the delay FSM's tail
    n_vq = 32
    print(f"  KV_RAMP_MARGIN={KV_RAMP_MARGIN} >= n_vq(={n_vq}) + audio_end rows: "
          f"{KV_RAMP_MARGIN >= n_vq + 2}")
    ok &= KV_RAMP_MARGIN >= n_vq + 2
    # default library size is the old hard-coded one, so library callers do not
    # silently change behaviour
    print(f"  library default (synthesize/MossTTSModel) stays {DEFAULT_MAX_SEQ_LEN}: "
          f"{DEFAULT_MAX_SEQ_LEN == 8192}")
    ok &= DEFAULT_MAX_SEQ_LEN == 8192
    # MiB arithmetic: 36 layers x 2 (K,V) x 8 kv heads x 128 head_dim x bf16
    for n, want in ((8192, 1152.0), (1024, 144.0)):
        got = _kv_mib(n)
        good = abs(got - want) < 1e-6
        ok &= good
        print(f"  _kv_mib({n}) = {got:.1f} MiB (want {want:.1f}) "
              f"{'ok' if good else 'FAIL'}")

    # ---- resize identity: an on-demand table is a prefix of a big one -------
    w = _tiny_weights()
    m_small = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=300)
    m_big = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=8192)
    rope_ok = bool(torch.equal(m_small.rope_cos, m_big.rope_cos[:300])
                   and torch.equal(m_small.rope_sin, m_big.rope_sin[:300]))
    m_grow = MossTTSModel(w, device="cpu", dtype=torch.bfloat16, max_seq_len=300)
    m_grow.ensure_seq_len(8192)
    grow_ok = bool(torch.equal(m_grow.rope_cos, m_big.rope_cos)
                   and torch.equal(m_grow.rope_sin, m_big.rope_sin)
                   and m_grow.k_cache.shape == m_big.k_cache.shape)
    # a no-op resize must not move anything
    noop_ok = m_grow.ensure_seq_len(1024) == 8192
    # growth must preserve live KV and zero only the tail
    m_grow.reset()
    m_grow._seq = 4
    m_grow.k_cache[:, :, :4] = 7.0
    m_grow.ensure_seq_len(16384)
    live_ok = bool((m_grow.k_cache[:, :, :4] == 7.0).all()) \
        and bool((m_grow.k_cache[:, :, 8192:] == 0).all())
    for label, good in (("rope[:n] == rope built at n", rope_ok),
                        ("ensure_seq_len reproduces a bigger model", grow_ok),
                        ("no-op resize changes nothing", noop_ok),
                        ("growth keeps live KV, zeroes the tail", live_ok)):
        ok &= good
        print(f"  resize identity: {label:44s} {'ok' if good else 'FAIL'}")
    del m_small, m_big, m_grow, w
    gc.collect()
    print(f"  phase 0 gate: {'PASS' if ok else 'FAIL'}")
    return ok


def _tiny_weights() -> dict:
    """Minimal weights dict: shapes are all MossTTSModel.__init__ inspects."""
    return {
        "language_model.embed_tokens.weight": torch.zeros(155648, 4096),
        "language_model.norm.weight": torch.zeros(4096),
        "language_model.layers.0.self_attn.q_norm.weight": torch.zeros(128),
        "language_model.layers.0.self_attn.k_norm.weight": torch.zeros(128),
        "language_model.layers.0.self_attn.q_proj.weight": torch.zeros(4096, 4096),
        "language_model.layers.0.self_attn.k_proj.weight": torch.zeros(1024, 4096),
        "language_model.layers.0.self_attn.v_proj.weight": torch.zeros(1024, 4096),
        "language_model.layers.0.self_attn.o_proj.weight": torch.zeros(4096, 4096),
        "language_model.layers.0.input_layernorm.weight": torch.zeros(4096),
        "language_model.layers.0.post_attention_layernorm.weight": torch.zeros(4096),
        "language_model.layers.0.mlp.gate_proj.weight": torch.zeros(12288, 4096),
        "language_model.layers.0.mlp.up_proj.weight": torch.zeros(12288, 4096),
        "language_model.layers.0.mlp.down_proj.weight": torch.zeros(4096, 12288),
        "lm_heads.0.weight": torch.zeros(155648, 4096),
        **{f"lm_heads.{i + 1}.weight": torch.zeros(1025, 4096) for i in range(32)},
        **{f"emb_ext.{i}.weight": torch.zeros(1025, 4096) for i in range(32)},
    }


# --------------------------------------------------------------- phase 1
def _fresh(max_seq_len: int):
    """Standalone w1p export -> (model, fast) at an explicit cache size.

    The standalone directory is preferred because it is the 8 GB deployment
    shape; if it is absent the base+state path is used (the same module, just
    assembled the slow way) so the peak gate still runs.
    """
    if os.path.isdir(STANDALONE):
        from moss_tts_lite.export import load_standalone_model
        return load_standalone_model(STANDALONE, device="cuda",
                                     max_seq_len=max_seq_len)
    from moss_tts_lite.st_loader import read_safetensors
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=max_seq_len)
    del weights
    fast = load_gptq_fast(model, STATE)
    torch.cuda.empty_cache()
    return model, fast


def phase_1_peaks() -> bool:
    """Measured peaks for typical inputs at the on-demand cache size."""
    print("=== Phase 1: measured peaks (on-demand KV, w1p fast path) ===")
    _start_ctx_gib()          # take the nvidia-smi baseline before anything else
    os.makedirs(OUT, exist_ok=True)
    ok = True
    worst = 0.0
    standalone = os.path.isdir(STANDALONE)
    ceiling = _PEAK_GIB if standalone else _PEAK_BASE_GIB
    print(f"  assembly path: {'standalone export' if standalone else 'base + w1p.pt'}; "
          f"ceiling {ceiling} GiB", flush=True)
    for name, text, language, budget in CASES:
        prompt = build_tts_prompt(text, language=language)
        l0 = int(prompt["input_ids"].shape[1])
        want = _resolve_max_seq_len(None, l0, budget)
        model, fast = _fresh(want)
        try:
            torch.cuda.reset_peak_memory_stats()
            res = generate_fast(fast, prompt, max_new_tokens=budget, seed=SEED)
            peak = torch.cuda.max_memory_allocated() / GIB
            kv = _kv_mib(int(model.max_seq_len)) / 1024
            good = peak <= ceiling and res.n_steps > 0
            ok &= good
            worst = max(worst, peak)
            print(f"  [{name:8s}] L0={l0:4d} max_seq_len={want} ({kv:.3f} GiB KV) "
                  f"steps={res.n_steps} peak={peak:.3f} GiB (ceiling {ceiling}) "
                  f"{'ok' if good else 'FAIL'}")
        finally:
            del fast, model
            gc.collect()
            torch.cuda.empty_cache()
    # the on-demand cache must be a real saving against the old fixed 8192
    model, fast = _fresh(DEFAULT_MAX_SEQ_LEN)
    prompt = build_tts_prompt(CASES[0][1])
    torch.cuda.reset_peak_memory_stats()
    generate_fast(fast, prompt, max_new_tokens=MAXNEW, seed=SEED)
    peak_8192 = torch.cuda.max_memory_allocated() / GIB
    del fast, model
    gc.collect()
    torch.cuda.empty_cache()
    saved = peak_8192 - worst
    # the KV saving itself is arithmetic and must be exactly the table's number
    kv_saved_mib = _kv_mib(8192) - _kv_mib(_resolve_max_seq_len(None, 66, MAXNEW))
    good = saved > 0.4 and abs(kv_saved_mib - 558) < 2
    ok &= good
    print(f"  fixed-8192 peak={peak_8192:.3f} GiB; on-demand saves {saved:.3f} GiB "
          f"({kv_saved_mib:.0f} MiB of KV) {'ok' if good else 'FAIL'}")
    print(f"  phase 1 gate: {'PASS' if ok else 'FAIL'} (worst {worst:.3f} GiB)")
    return ok


def _phase2_one(case: str) -> dict:
    """One 8 GB case, in its own process (see `_phase2_child` for why)."""
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    ctx = _start_ctx_gib()
    cap = EIGHT_GB_CARD - ctx
    torch.cuda.set_per_process_memory_fraction(cap / total)
    name, text, language, _ = next(c for c in CASES if c[0] == case)
    prompt = build_tts_prompt(text, language=language)
    l0 = int(prompt["input_ids"].shape[1])
    want = _resolve_max_seq_len(None, l0, EIGHT_GB_BUDGET)
    rec = dict(case=case, l0=l0, ml=want, ctx=ctx, cap=cap, ok=False)
    try:
        model, fast = _fresh(want)
        torch.cuda.reset_peak_memory_stats()
        res = generate_fast(fast, prompt, max_new_tokens=EIGHT_GB_BUDGET, seed=SEED)
        rec["steps"] = res.n_steps
        rec["peak_alloc"] = torch.cuda.max_memory_allocated() / GIB
        rec["ok"] = res.n_steps > 0
    except torch.cuda.OutOfMemoryError:
        rec["err"] = "OOM"
    return rec


def _phase2_child() -> bool:
    """Run phase 2 in fresh processes and return the verdict.

    This is not a convenience: the caching allocator cannot fully return what it
    reserved under a `set_per_process_memory_fraction` cap, so a phase 2 that ran
    after phase 1's several model loads -- or even several cases back to back --
    OOMs at a cap that passes standalone (measured: the second case onward fails
    at the correct 7.715 GiB cap, while each case in its own process passes).
    That is the same cross-iteration poisoning the report's `run_grid.sh` avoids
    with one process per arm.  A fresh CUDA context + allocator is the only way
    to measure a card-size budget honestly.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok = True
    ctx = None
    print(f"=== Phase 2: 8 GB card simulation ===")
    for case, _, _, _ in CASES:
        proc = subprocess.run([sys.executable, "-m", "tests.test_vram_budget"],
                              cwd=root, env=dict(os.environ, KVFIT_PHASE2=case),
                              capture_output=True, text=True)
        line = [l for l in proc.stdout.splitlines() if l.startswith("KVFITREC ")]
        if not line:
            print(f"  [{case:8s}] child failed:\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}")
            ok = False
            continue
        import json
        rec = json.loads(line[0][len("KVFITREC "):])
        ctx = rec["ctx"]
        smi = rec["smi_peak"]
        good = rec["ok"] and smi <= EIGHT_GB_CARD
        ok &= good
        print(f"  [{case:8s}] L0={rec['l0']:4d} max_seq_len={rec['ml']:5d} "
              f"steps={rec.get('steps', 0):4d} torch peak={rec.get('peak_alloc', 0):.3f} "
              f"smi peak={smi:.3f} (card {EIGHT_GB_CARD}, spare "
              f"{EIGHT_GB_CARD - smi:+.3f}) {'ok' if good else 'FAIL'}"
              + (f" [{rec.get('err', '')}]" if rec.get("err") else ""), flush=True)
    print(f"  CUDA context {ctx if ctx else float('nan'):.3f} GiB; torch capped at "
          f"{EIGHT_GB_CARD - (ctx or 0):.3f} GiB; pass criterion: "
          f"nvidia-smi peak <= {EIGHT_GB_CARD} GiB")

    # the codec phase (TTS freed) + the boundary control + the advice checks:
    # one more child each, for the same allocator reason
    for label, env in (("codec", "KVFIT_C1=c"), ("boundary", "KVFIT_C1=b"),
                       ("advice", "KVFIT_C1=a")):
        proc = subprocess.run([sys.executable, "-m", "tests.test_vram_budget"],
                              cwd=root, env=dict(os.environ, KVFIT_PHASE2="", **{env.split("=")[0]: env.split("=")[1]}),
                              capture_output=True, text=True)
        out = [l for l in proc.stdout.splitlines() if l.startswith("KVFITC1 ")]
        good = proc.returncode == 0 and bool(out) and out[0].endswith("PASS")
        ok &= good
        print("  " + (out[0][len("KVFITC1 "):] if out else f"{label} child failed"))
    print(f"  phase 2 gate: {'PASS' if ok else 'FAIL'}")
    return ok


def _phase2_rest(which: str) -> bool:
    """Codec / boundary-control / advice checks, each in its own capped process."""
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    ctx = _start_ctx_gib()
    cap = EIGHT_GB_CARD - ctx
    torch.cuda.set_per_process_memory_fraction(cap / total)
    ok = True
    if which == "c":
        from moss_tts_lite.codec import MossCodecDecoder
        codec_dir = os.environ.get("MOSS_AUDIO_MODEL_DIR") or os.path.join(
            ROOT, "models", "MOSS-Audio-Tokenizer")
        torch.cuda.reset_peak_memory_stats()
        codec = MossCodecDecoder(codec_dir, device=torch.device("cuda"))
        peak = torch.cuda.max_memory_allocated() / GIB
        del codec
        ok = peak <= _PEAK_CODEC_GIB and peak <= cap
        print(f"KVFITC1 codec alone: peak={peak:.3f} GiB (ceiling {_PEAK_CODEC_GIB}, "
              f"cap {cap:.3f}) -> {'PASS' if ok else 'FAIL'}")
    elif which == "b":
        # A budget well above the fit: 4 x 2048 = 8192 is robustly over an 8 GiB
        # card for this tier (the CLI default of 4096 is over too, and is covered
        # by the sweep in the report).  The control is deliberately NOT 2560: that
        # sits 14 MiB above the card in the report's sweep and flips with
        # allocator fragmentation, so asserting it would make this gate flaky.
        # The claim being pinned here is a yes/no one -- a budget far above what
        # the card can hold must OOM, and `--max-new-tokens 2048` must not.
        over = 4 * EIGHT_GB_BUDGET
        prompt = build_tts_prompt(CASES[0][1])
        l0 = int(prompt["input_ids"].shape[1])
        try:
            model, fast = _fresh(_resolve_max_seq_len(None, l0, over))
            try:
                generate_fast(fast, prompt, max_new_tokens=over, seed=SEED)
            finally:
                del fast, model
        except torch.cuda.OutOfMemoryError:
            ok = True
            print(f"KVFITC1 boundary control: --max-new-tokens {over} OOMs as "
                  f"measured -> PASS")
        else:
            ok = False
            print(f"KVFITC1 boundary control: --max-new-tokens {over} unexpectedly "
                  f"FIT (cap not binding; boundary numbers stale) -> FAIL")
    else:
        from moss_tts_lite.cli import _kv_advice, _oom_advice, synthesize
        msg_ok = False
        try:
            synthesize("你好。", os.path.join(OUT, "never.wav"), max_new_tokens=60000,
                       seed=SEED, fast=True, w4_group_size=32, gptq_state=STATE,
                       model_dir=MODEL_DIR)
        except (torch.cuda.OutOfMemoryError, ValueError) as exc:
            msg_ok = "max-new-tokens" in str(exc) and "segment" in str(exc)
        kv_msg = str(_kv_advice(ValueError("prompt 66 + max_new_tokens 4096 "
                                           "exceeds KV cache 512"), 66, 4096))
        oom_msg = str(_oom_advice(torch.cuda.OutOfMemoryError("CUDA out of memory."),
                                  "你好。", 60000))
        passthrough = str(_kv_advice(
            ValueError("w4_group_size must be 32/64/128/256"), 1, 1)) \
            == "w4_group_size must be 32/64/128/256"
        both = all("--max-new-tokens" in m and "segment" in m
                   for m in (kv_msg, oom_msg))
        ok = msg_ok and both and passthrough
        print(f"KVFITC1 advice: over-budget raised with advice={msg_ok}, "
              f"builders name both remedies={both}, unrelated ValueError "
              f"passes through={passthrough} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if os.environ.get("KVFIT_PHASE2"):
        import json
        rec = _phase2_one(os.environ["KVFIT_PHASE2"])
        rec["smi_peak"] = _smi_mib() / 1024
        print("KVFITREC " + json.dumps(rec))
        return 0 if rec["ok"] else 1
    if os.environ.get("KVFIT_C1"):
        return 0 if _phase2_rest(os.environ["KVFIT_C1"]) else 1
    ok0 = phase_0_sizing()
    if not torch.cuda.is_available():
        print("test_vram_budget: no CUDA -- phase 0 only")
        return 0 if ok0 else 1
    if not os.path.exists(STATE):
        print(f"SKIP phase 1/2: GPTQ state {STATE} not present")
        return 0 if ok0 else 1
    ok1 = phase_1_peaks()
    ok2 = _phase2_child()
    ok = ok0 and ok1 and ok2
    print(f"test_vram_budget: sizing={'PASS' if ok0 else 'FAIL'} "
          f"peaks={'PASS' if ok1 else 'FAIL'} "
          f"8gb={('PASS' if ok2 else 'FAIL')} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
