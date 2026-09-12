"""Fast-path (CUDA graph) tests — M1 gates (perf agent).

Phase 0  bitwise step parity: fast.step (graph replay) vs model.step eager,
         hidden + both heads compared with torch.equal over several steps and
         positions (before/after the gqa->repeat fallback boundary matters
         only for speed; both branches must stay bitwise).
Phase 1  trajectory gate: generate_fast on the golden prompts (zh + en,
         seed=1234, default sampling) must be EXACT vs .tmp/golden/gen_golden.
Phase 2  speed: steady-state steps/s (cuda events, steps 10+), prefill ms,
         VRAM.  M1 target >= 34 steps/s (bf16 eager baseline ~29.7).

Run (GPU, under the lock):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_fast
"""

import os
import sys

import torch

from ..model import MossTTSModel
from ..fast import FastMossTTS, generate_fast
try:
    from ..st_loader import read_safetensors
except ImportError:  # pragma: no cover
    from ._mini_loader import read_safetensors_min as read_safetensors

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")

SEED = 1234
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
    from ..generate import generate
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


def main() -> int:
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


if __name__ == "__main__":
    sys.exit(main())
