"""Generation tests:"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import torch

from moss_tts_lite.cli import _max_pause_s
from moss_tts_lite.codec import MossCodecDecoder, delayed_rows_to_segments
from moss_tts_lite.fast import FastMossTTS, generate_fast
from moss_tts_lite.generate import generate
from moss_tts_lite.model import (AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID,
                                 AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_PAD_CODE,
                                 AUDIO_START_TOKEN_ID, IM_END_TOKEN_ID,
                                 MossTTSModel)
from moss_tts_lite.prompt import build_continuation_prompt, build_tts_prompt

try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors
from tests._mini_bpe import build_tts_prompt_dev

ROOT = os.environ.get("MOSS_TTS_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
CODEC_DIR = os.path.join(ROOT, "models", "MOSS-Audio-Tokenizer")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
OUT = os.path.join(ROOT, ".tmp", "perf_agent")
LISTEN_AGENT = os.path.join(ROOT, ".tmp", "listen_agent")

def main_generate():
    dev = torch.device("cuda")
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=dev, dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()

    prompt = build_tts_prompt_dev("Hello world, this is a smoke test.")
    print("prompt L =", prompt["input_ids"].shape[1])

    res = generate(model, prompt, max_new_tokens=200, greedy=True)

    assert res.audio_frames.shape == (res.n_steps, 32), res.audio_frames.shape
    assert res.text_ids.shape == (res.n_steps,)
    assert res.n_steps <= 200
    print(f"greedy: n_steps={res.n_steps}, finished={res.finished}, "
          f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    t = res.text_ids.tolist()
    fr = res.audio_frames

    if AUDIO_START_TOKEN_ID in t:
        i_start = t.index(AUDIO_START_TOKEN_ID)

        pre = [x for x in t[:i_start] if x != 151643]
        assert not pre, f"unexpected pre-audio text tokens: {pre[:8]}"
        i_first_gen = i_start + 1
        assert t[i_start + 1] == AUDIO_GEN_SLOT_TOKEN_ID, \
            f"first post-start row must be gen_slot, got {t[i_start + 1]}"

        i_delay = t.index(AUDIO_DELAY_SLOT_TOKEN_ID)
        assert i_delay >= i_first_gen, (i_delay, i_first_gen)

        delay_rows = [x for x in t[i_delay:] if x in (AUDIO_DELAY_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID)]
        assert delay_rows[0] == AUDIO_DELAY_SLOT_TOKEN_ID
        k = delay_rows.index(AUDIO_END_TOKEN_ID)
        assert k == 32, f"expected 32 delay rows before audio_end, got {k}"
        assert t[i_delay + 32] == AUDIO_END_TOKEN_ID

        assert (fr[i_delay + 32] == AUDIO_PAD_CODE).all(), \
            "audio_end row must be all-pad"
        print("delay-pattern trajectory: audio_start@%d, %d gen rows (~%d frames), "
              "32 delay rows, audio_end@%d" % (i_start, i_delay - i_first_gen,
                                               i_delay - i_first_gen + 1, i_delay + 32))
    if res.finished:
        assert t[-1] == IM_END_TOKEN_ID, f"last token {t[-1]} != im_end"
        assert IM_END_TOKEN_ID not in t[:-1], "im_end mid-stream"
    print("text stream:", [hex(x) for x in t[:8]], "...", [hex(x) for x in t[-6:]])

    assert fr.min() >= 0 and fr.max() <= 1024
    real = fr[fr != AUDIO_PAD_CODE]

    MAXD = 2**63 - 1
    al, delayed = 0, MAXD
    for r, tt in enumerate(t):
        active = {i for i in range(32) if al > i and (delayed == MAXD or i >= delayed)}
        row = fr[r].tolist()
        for i in range(32):
            if i in active:
                assert row[i] != AUDIO_PAD_CODE, f"row {r} ch {i} should be sampled"
            else:
                assert row[i] == AUDIO_PAD_CODE, f"row {r} ch {i} should be pad, got {row[i]}"
        if tt in (AUDIO_START_TOKEN_ID, AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_DELAY_SLOT_TOKEN_ID):
            al += 1
        elif tt == AUDIO_END_TOKEN_ID:
            al = 0
        if delayed == MAXD and tt == AUDIO_DELAY_SLOT_TOKEN_ID:
            delayed = 0
        if delayed != MAXD:
            delayed += 1
        if delayed > 32:
            delayed = MAXD
    print("audio mask state-walk: all 63 rows match reference semantics")
    print(f"audio frames: {tuple(fr.shape)}, pad-ratio={float((fr == AUDIO_PAD_CODE).float().mean()):.3f}, "
          + (f"non-pad codes in [{int(real.min())}, {int(real.max())}]" if real.numel() else "all pad"))

    res2 = generate(model, prompt, max_new_tokens=120, greedy=False, seed=1234)
    res3 = generate(model, prompt, max_new_tokens=120, greedy=False, seed=1234)
    assert res2.n_steps == res3.n_steps
    assert torch.equal(res2.text_ids, res3.text_ids), "same seed must reproduce text ids"
    assert torch.equal(res2.audio_frames, res3.audio_frames), "same seed must reproduce audio"
    print(f"seeded sampling: n_steps={res2.n_steps}, finished={res2.finished}, "
          f"deterministic rerun OK")

    print("test_generate PASS")

try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors

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
    """Golden zh/en trajectories:"""
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
    """275-char zh long-form:"""
    print("=== Phase L: long-form zh (275 chars) must not trigger ===")
    assert _max_pause_s(LONG_TEXT) is None
    res = generate_fast(fast, build_tts_prompt(LONG_TEXT),
                        max_new_tokens=4096, seed=1234)
    dur = res.n_steps / FPS
    print(f"  [long] steps={res.n_steps} (~{dur:.1f}s) finished={res.finished} "
          f"triggered={res.watchdog_triggered} reason={res.watchdog_reason}")
    crossed = res.n_steps > 640
    ok = (not res.watchdog_triggered) and res.finished and crossed
    print(f"  crossed 640-frame floor: {crossed}; gate: {'PASS' if ok else 'FAIL'}")
    return ok

def phase_pause(fast):
    """[pause 8s]:"""
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

        ref_path = os.path.join(ROOT, ".tmp", "verdict_agent", "codes",
                                f"extra_en4_w4_s{seed}.pt")
        if not os.path.exists(ref_path):
            print(f"  [s{seed}] pre-trigger vs verdict rows: SKIP "
                  f"({os.path.relpath(ref_path, ROOT)} not present)")
            continue
        verdict_codes = torch.load(ref_path, map_location="cpu",
                                   weights_only=True).long()
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

def main_fast_watchdog() -> int:
    assert torch.cuda.is_available()
    print("loading weights ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()

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

def main() -> int:
    rc = 0
    rc |= main_generate() or 0
    rc |= main_fast_watchdog() or 0

    return rc

if __name__ == "__main__":
    sys.exit(main())
