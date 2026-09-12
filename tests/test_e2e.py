"""E2E test (task 7): CLI-equivalent pipeline vs codec golden + speed baseline.

  1. zh golden text, seed=1234, default sampling -> full pipeline
     (build_tts_prompt -> MossTTSModel -> generate -> delayed_rows_to_segments
     -> MossCodecDecoder.decode) -> compare against .tmp/golden/codec_golden.wav
     (codes EXACT + codec SNR 118 dB -> expected max|delta| < 1e-4);
     assert duration 8.24 s +- 0.01.
  2. en text -> .tmp/tts_agent/e2e_en.wav for manual listening.
  3. speed baseline (torch.cuda.Event via generate(stats=...)): prefill ms,
     overall decode steps/s and steady-state (steps 10+) steps/s.

Run (GPU, under the lock):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_e2e
"""

import os
import sys

import numpy as np
import soundfile as sf
import torch

from moss_tts_lite.cli import synthesize

ROOT = os.environ.get("MOSS_TTS_ROOT", os.path.join(os.path.dirname(__file__), ".."))
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
OUT_DIR = os.path.join(ROOT, ".tmp", "tts_agent")
SR = 24000


def _steps_ps(step_ms, skip=0):
    ms = step_ms[skip:]
    if not ms:
        return float("nan")
    return 1000.0 / (sum(ms) / len(ms))


def main() -> int:
    gg = torch.load(os.path.join(GOLDEN, "gen_golden.pt"),
                    map_location="cpu", weights_only=False)
    zh = next(c for c in gg["cases"] if c["case_name"] == "zh_plain")
    en = next(c for c in gg["cases"] if c["case_name"] == "en_language")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 1) zh: golden text -> wav, compare with codec_golden.wav ----------
    print("=== E2E zh (golden text, seed=1234) ===", flush=True)
    stats: dict = {}
    wav, sr, res, strategy = synthesize(
        zh["text"], os.path.join(OUT_DIR, "e2e_zh.wav"),
        seed=zh["seed"], max_new_tokens=4096,
        sampling=dict(zh["sampling"]), stats=stats)
    assert sr == SR, sr
    assert torch.equal(res.audio_frames, zh["generation_ids"][:, 1:]), \
        "audio codes no longer EXACT vs gen_golden"
    assert res.finished and res.n_steps == zh["n_steps"]
    print(f"codes EXACT vs gen_golden (T={res.n_steps}), vram_strategy={strategy}")

    ref, sr_ref = sf.read(os.path.join(GOLDEN, "codec_golden.wav"), dtype="float32")
    assert sr_ref == SR
    L = min(len(wav), len(ref))
    d = np.abs(wav[:L].astype(np.float64) - ref[:L].astype(np.float64))
    dur = len(wav) / SR
    print(f"wav: {len(wav)} samples ({dur:.3f}s) vs golden {len(ref)} "
          f"({len(ref) / SR:.3f}s); max|d|={d.max():.3e} mean|d|={d.mean():.3e}")
    assert abs(dur - 8.24) <= 0.01, f"duration {dur:.4f}s outside 8.24+-0.01"
    assert len(wav) == len(ref), "sample count differs from golden"
    assert d.max() < 1e-4, f"wav max|d|={d.max():.3e} >= 1e-4"
    print("E2E zh vs codec_golden.wav: PASS (<1e-4, 8.24s+-0.01)")

    # ---------- 2) speed baseline (from the zh run above) ----------
    prefill_ms = stats.get("prefill_ms")
    step_ms = stats.get("step_ms") or []
    print("=== speed baseline (bf16, A10G, zh 137 steps) ===")
    print(f"prefill: {prefill_ms:.1f} ms (L={zh['prompt_L']})")
    print(f"decode:  avg {sum(step_ms) / len(step_ms):.2f} ms/step over "
          f"{len(step_ms)} steps -> {1000 / (sum(step_ms) / len(step_ms)):.1f} steps/s "
          f"(incl. prefill-equivalent first step amortized)")
    print(f"decode:  steady-state (steps 10+) {_steps_ps(step_ms, 10):.1f} steps/s "
          f"({sum(step_ms[10:]) / len(step_ms[10:]):.2f} ms/step)")

    # ---------- 3) en artifact for manual listening ----------
    print("=== E2E en (manual-listening artifact) ===", flush=True)
    stats_en: dict = {}
    wav_en, _, res_en, strat_en = synthesize(
        en["text"], os.path.join(OUT_DIR, "e2e_en.wav"),
        language=en["kwargs"].get("language"), seed=en["seed"],
        max_new_tokens=4096, sampling=dict(en["sampling"]), stats=stats_en)
    assert torch.equal(res_en.audio_frames, en["generation_ids"][:, 1:]), \
        "en audio codes no longer EXACT vs gen_golden"
    dur_en = len(wav_en) / SR
    ms_en = stats_en.get("step_ms") or []
    print(f"wrote {os.path.join(OUT_DIR, 'e2e_en.wav')}: {dur_en:.2f}s, "
          f"codes EXACT, strategy={strat_en}, steady {_steps_ps(ms_en, 10):.1f} steps/s")
    assert dur_en > 1.0

    print("test_e2e PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
