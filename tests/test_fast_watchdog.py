"""Watchdog tests v2 — W4 "non-terminating near-silence runaway" (perf-6/7).

v2 fixes two real side effects of the v1 watchdog (supervisor reproductions):
  1. explicit [pause Xs] markers were truncated by rule (a) (8 s -> 5.12 s):
     the effective silence threshold now scales with the longest requested
     pause parsed from the raw text: max(auto, ceil((pause + 2.0) * 12.5));
  2. healthy long-form single segments (280-char zh ≈ 800 frames) hit the
     640-frame max_segment floor: rule (b) is now compound — it fires only
     when the segment is over the floor AND the current tail is already
     silent (>= 32 consecutive low-energy ch0 frames, 2.56 s), so healthy
     long segments that keep producing sound never trigger.

Phases (GPU, under the lock):
  B   golden zh/en EXACT regression with the watchdog on (no trigger).
  L   275-char zh long-form text, bf16 fast: crosses the 640-frame floor,
      must NOT trigger, must finish naturally (steps/duration reported).
  P   [pause 8s] text, bf16 fast: must NOT trigger; the wav must keep a
      >= 7.5 s silent stretch (measured at the -45 dBFS verdict threshold).
  A   en4 W4 runaway seeds 3005/3009: still caught by rule (a); pre-trigger
      rows bitwise-identical to the verdict runs; forced end = the model's
      own delayed_lengths==n_vq closing ramp -> legal codec segments/wav.

Run (GPU, under the lock):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_fast_watchdog
"""

import os
import sys

import numpy as np
import soundfile as sf
import torch

from moss_tts_lite.model import MossTTSModel
from moss_tts_lite.prompt import build_tts_prompt, build_continuation_prompt
from moss_tts_lite.cli import _max_pause_s
from moss_tts_lite.fast import FastMossTTS, generate_fast
from moss_tts_lite.codec import MossCodecDecoder, delayed_rows_to_segments
try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:  # pragma: no cover
    from tests._mini_loader import read_safetensors_min as read_safetensors

ROOT = os.environ.get("MOSS_TTS_ROOT", os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
CODEC_DIR = os.path.join(ROOT, "models", "MOSS-Audio-Tokenizer")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
OUT = os.path.join(ROOT, ".tmp", "perf_agent")
LISTEN_AGENT = os.path.join(ROOT, ".tmp", "listen_agent")

SR = 24000
FPS = 12.5
HOP = 1920

LONG_TEXT = (
    "唐诗是中国文学史上最璀璨的明珠之一。从初唐四杰的意气风发，到盛唐李白的豪放飘逸、"
    "杜甫的沉郁顿挫，再到晚唐李商隐的深情绵邈，每一个时代都留下了不朽的篇章。诗人用最凝练"
    "的语言，描绘出山河的壮阔、田园的宁静、友情的真挚与思乡的愁绪。千百年过去，我们依然会"
    "在春天想起沾衣欲湿的杏花雨，在秋天抬头望明月，思念远方的亲人。这正是诗歌的力量，它穿"
    "越时空，把古人的情感与今人的心灵紧紧相连。宋词同样令人陶醉。苏轼的豪放，李清照的婉约，"
    "辛弃疾的慷慨悲歌，各自成家。读一首好词，就像品一杯清茶，初尝或许平淡，回味却悠长。愿这"
    "些古老的句子，继续陪伴我们走过每一个春夏秋冬。")
PAUSE_TEXT = "我今天学习了一首中国的古诗，它的名字是[pause 8s]静夜思！"


def _frame_db(wav: np.ndarray, n_frames: int) -> np.ndarray:
    out = np.full(n_frames, np.nan)
    for k in range(n_frames):
        a = np.asarray(wav[k * HOP:(k + 1) * HOP], dtype=np.float64)
        if len(a):
            out[k] = 20 * np.log10(max(float(np.abs(a).max()), 1e-10))
    return out


def _max_silent_run(wav: np.ndarray, tau_db: float = -45.0) -> int:
    n = int(len(wav) // HOP)
    db = _frame_db(wav, n)
    best = cur = 0
    for v in np.nan_to_num(db, nan=0.0):
        cur = cur + 1 if v <= tau_db else 0
        best = max(best, cur)
    return best


def _decode_to_wav(res, out_path):
    codec = MossCodecDecoder(CODEC_DIR, device=torch.device("cuda"))
    segments = delayed_rows_to_segments(res.audio_frames)
    wavs = [codec.decode(seg, chunk_duration=8.0) for seg in segments]
    wav = np.concatenate(wavs) if len(wavs) > 1 else (wavs[0] if wavs
                                                      else np.zeros(1))
    sf.write(out_path, wav, SR, subtype="FLOAT")
    del codec
    torch.cuda.empty_cache()
    return wav, len(segments)


def phase_b(fast):
    """Golden zh/en trajectories: watchdog on, EXACT preserved, no trigger."""
    print("=== Phase B: golden EXACT regression (watchdog on) ===")
    pg = torch.load(os.path.join(GOLDEN, "prompt_golden.pt"),
                    map_location="cpu", weights_only=False)
    gg = torch.load(os.path.join(GOLDEN, "gen_golden.pt"),
                    map_location="cpu", weights_only=False)
    names = [c["name"] for c in pg["cases"]]
    ok = True
    for case in ("zh_plain", "en_language"):
        i = names.index(case)
        res = generate_fast(fast, {"input_ids": pg["input_ids"][i].cuda(),
                                   "attention_mask": pg["attention_mask"][i].cuda()},
                            max_new_tokens=4096, seed=1234)
        gt = next(c for c in gg["cases"] if c["case_name"] == case)
        T = res.n_steps
        text_ok = bool((res.text_ids.long() ==
                        gt["generation_ids"][:T, 0].long()).all())
        audio_ok = bool((res.audio_frames.long() ==
                         gt["generation_ids"][:T, 1:].long()).all())
        print(f"  [{case}] steps={res.n_steps} (golden {gt['n_steps']}) "
              f"finished={res.finished} EXACT={text_ok and audio_ok} "
              f"triggered={res.watchdog_triggered}")
        ok &= text_ok and audio_ok and T == gt["n_steps"]
        ok &= res.finished == gt["finished"]
        ok &= not res.watchdog_triggered
    print(f"  phaseB gate: {'PASS' if ok else 'FAIL'}")
    return ok


def phase_long(fast):
    """275-char zh long-form: crosses the 640-frame floor, must NOT trigger."""
    print("=== Phase L: long-form zh (275 chars) must not trigger ===")
    assert _max_pause_s(LONG_TEXT) is None
    res = generate_fast(fast, build_tts_prompt(LONG_TEXT),
                        max_new_tokens=4096, seed=1234)
    dur = res.n_steps / FPS
    print(f"  [long] steps={res.n_steps} (~{dur:.1f}s) finished={res.finished} "
          f"triggered={res.watchdog_triggered} reason={res.watchdog_reason}")
    crossed = res.n_steps > 640          # exercised the v1 false-positive zone
    ok = (not res.watchdog_triggered) and res.finished and crossed
    print(f"  crossed 640-frame floor: {crossed}; gate: {'PASS' if ok else 'FAIL'}")
    return ok


def phase_pause(fast):
    """[pause 8s]: pause-aware threshold; the requested pause survives."""
    print("=== Phase P: explicit [pause 8s] preserved ===")
    assert _max_pause_s(PAUSE_TEXT) == 8.0
    assert _max_pause_s("no markers here") is None
    res = generate_fast(fast, build_tts_prompt(PAUSE_TEXT),
                        max_new_tokens=4096, seed=1234,
                        max_requested_pause_s=_max_pause_s(PAUSE_TEXT))
    print(f"  [pause] steps={res.n_steps} finished={res.finished} "
          f"triggered={res.watchdog_triggered} reason={res.watchdog_reason} "
          f"stats={res.watchdog_stats}")
    ok = not res.watchdog_triggered and res.finished
    wav, _ = _decode_to_wav(res, os.path.join(OUT, "watchdog_pause8s.wav"))
    run = _max_silent_run(wav)
    dur = len(wav) / SR
    print(f"  [pause] wav={dur:.2f}s max_silent_run={run} frames "
          f"({run / FPS:.2f}s @ -45 dBFS); required >= {int(7.5 * FPS)} frames")
    ok &= run >= int(7.5 * FPS)
    print(f"  phaseP gate: {'PASS' if ok else 'FAIL'}")
    return ok


# Continuation corpus for this gate (texts inlined: moss_tts_lite/ must not import
# from .tmp/, per the dependency-purity gate).  Source of the strings:
# .tmp/listen_agent/listen2_texts.py (pair 4, English) — the exact text used by
# the verdict-1 runaway reproduction, so the W4 trajectories stay comparable.
EN_REF_TRANSCRIPT = ("But I really can't complain about not having a normal "
                     "college experience to you.")
EN4_TEXT = ("Hello! This is a short continuation test. Please listen for a "
            "stable and natural voice.")


def phase_a(model):
    """Reproduce the en4 W4 runaway with the watchdog on (rule a)."""
    print("=== Phase A: en4 W4 runaway + production watchdog ===")
    fast = FastMossTTS(model, quant="w4", w4_group_size=128)
    ref_en = torch.load(os.path.join(LISTEN_AGENT, "ref_en_codes.pt"),
                        map_location="cpu", weights_only=False).to(torch.long)
    full = EN_REF_TRANSCRIPT + " " + EN4_TEXT
    prompt = build_continuation_prompt(text=full, language="English",
                                       prefix_codes=ref_en)
    ok = True
    for seed, expect_steps in ((3005, 400), (3009, 600)):
        res = generate_fast(fast, prompt, max_new_tokens=4096, seed=seed)
        st = res.watchdog_stats
        print(f"  [s{seed}] steps={res.n_steps} finished={res.finished} "
              f"triggered={res.watchdog_triggered} reason={res.watchdog_reason} "
              f"stats={st}")
        ok &= res.watchdog_triggered and res.finished
        ok &= res.n_steps < expect_steps
        verdict_codes = torch.load(
            os.path.join(ROOT, ".tmp", "verdict_agent", "codes",
                         f"extra_en4_w4_s{seed}.pt"),
            map_location="cpu", weights_only=True).long()
        S = int(st["step"])
        pre_mine = res.audio_frames[:S].long()
        pre_ref = verdict_codes[:S]
        same = bool(pre_mine.shape == pre_ref.shape
                    and (pre_mine == pre_ref).all())
        print(f"  [s{seed}] pre-trigger rows identical to verdict run: "
              f"{same} ({S} rows)")
        ok &= same
        tail = res.audio_frames[S:].long()
        ok &= tail.shape[0] == 33 and bool((tail[-1] == 1024).all())
        wav, nseg = _decode_to_wav(
            res, os.path.join(OUT, f"watchdog_en4_w4_s{seed}.wav"))
        dur = len(wav) / SR
        fin = bool(np.isfinite(wav).all())
        print(f"  [s{seed}] wav: {dur:.2f}s finite={fin} segments={nseg} "
              f"(expected ~{(res.n_steps - 31) / FPS:.1f}s)")
        ok &= fin and nseg >= 1 and dur > 2.0
    del fast
    torch.cuda.empty_cache()
    print(f"  phaseA gate: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    assert torch.cuda.is_available()
    print("loading weights ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()

    # bf16 phases first (pristine model); phase_a quantizes IN PLACE last.
    fast = FastMossTTS(model)
    ok_b = phase_b(fast)
    ok_long = phase_long(fast)
    ok_pause = phase_pause(fast)
    del fast
    torch.cuda.empty_cache()
    ok_a = phase_a(model)

    print(f"test_fast_watchdog: phaseA={'PASS' if ok_a else 'FAIL'} "
          f"phaseB={'PASS' if ok_b else 'FAIL'} "
          f"phaseLong={'PASS' if ok_long else 'FAIL'} "
          f"phasePause={'PASS' if ok_pause else 'FAIL'} "
          f"-> {'PASS' if (ok_a and ok_b and ok_long and ok_pause) else 'FAIL'}")
    return 0 if (ok_a and ok_b and ok_long and ok_pause) else 1


if __name__ == "__main__":
    sys.exit(main())
