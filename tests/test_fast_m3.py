"""Fast-path W8 quantization tests — M3 gates (perf agent).

Phase A  teacher-forced 40 steps (golden rows) with the quantized forward:
         argmax agreement vs .tmp/golden/logits_golden.npz, text head and
         audio heads reported separately (gate >= 99% each; quantized
         numerics drift by design, so NOT a bitwise gate).
Phase B  E2E zh/en (seed=1234, default sampling) on the quantized path:
         structure valid (finished, non-empty audio segment), wav written to
         .tmp/perf_agent/m3_zh.wav / m3_en.wav for listening.
Phase C  speed + VRAM: steady steps/s (>= 50 target) and peak GiB (< 12).

Run (GPU, under the lock):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_fast_m3
"""

import os
import sys

import numpy as np
import soundfile as sf
import torch

from ..model import MossTTSModel
from ..fast import FastMossTTS, generate_fast
from ..codec import MossCodecDecoder, delayed_rows_to_segments
try:
    from ..st_loader import read_safetensors
except ImportError:  # pragma: no cover
    from ._mini_loader import read_safetensors_min as read_safetensors

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
CODEC_DIR = os.path.join(ROOT, "models", "MOSS-Audio-Tokenizer")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")
OUT = os.path.join(ROOT, ".tmp", "perf_agent")

SEED = 1234
SR = 24000


def _steps_ps(step_ms, skip=0):
    ms = step_ms[skip:]
    return 1000.0 / (sum(ms) / len(ms)) if ms else float("nan")


def phase_a(fast, zh_ids):
    """Teacher-forced 40 steps: quantized forward vs golden logits argmax."""
    print("=== Phase A: teacher-forced 40-step argmax (quantized forward) ===")
    z = np.load(os.path.join(GOLDEN, "logits_golden.npz"))
    gt_text = torch.from_numpy(z["logits_text"]).cuda()          # [40, V]
    gt_audio = torch.from_numpy(z["logits_audio"]).cuda()        # [40, 32, 1025]
    gt_rows = torch.from_numpy(z["selected_rows"]).reshape(-1, 33).cuda()
    model = fast.m
    mine_text, mine_audio = [], []
    with torch.inference_mode():
        hs = fast.prefill(zh_ids)
        L0 = int(model.seq_len)
        fast.capture()
        h = hs.last_hidden[:, -1]
        mine_text.append(model.text_logits(h).float())
        mine_audio.append(model.audio_logits(h).float())
        for t in range(1, gt_rows.shape[0]):
            lt, la, h, _two = fast.step(gt_rows[t - 1].view(1, 33), L0 + t - 1)
            mine_text.append(lt.float())
            mine_audio.append(la.float())
    mine_text = torch.stack(mine_text)
    mine_audio = torch.stack(mine_audio)
    ok_t = (mine_text.argmax(-1) == gt_text.argmax(-1)).float().mean().item() * 100
    ok_a = (mine_audio[..., :1024].argmax(-1)
            == gt_audio[..., :1024].argmax(-1)).float().mean().item() * 100
    d_t = (mine_text - gt_text).abs().max().item()
    d_a = (mine_audio[..., :1024] - gt_audio[..., :1024]).abs().max().item()
    # top-1 logit gap where argmax flipped (how near-tie are the misses)
    flip = mine_text.argmax(-1) != gt_text.argmax(-1)
    gap = "n/a"
    if flip.any():
        idx = flip.nonzero()[0]
        s = idx[0].item()
        g = gt_text[s]
        gap = f"step {s.item()}: golden top1={g.max().item():.3f} " \
              f"top2={g.topk(2).values[1].item():.3f}"
    print(f"  logits_text : max|d|={d_t:.4f}  argmax agree={ok_t:.2f}%")
    print(f"  logits_audio: max|d|={d_a:.4f}  argmax agree={ok_a:.2f}%")
    print(f"  text flips: {int(flip.sum())}/40  ({gap})")
    ok = ok_t >= 99.0 and ok_a >= 99.0
    print(f"  gate (>=99% both): {'PASS' if ok else 'FAIL'}")
    return ok, ok_t, ok_a


def _gen_wav(fast, ids, mask, out_path, label):
    model = fast.m
    res = generate_fast(fast, {"input_ids": ids.cuda(), "attention_mask": mask.cuda()},
                        max_new_tokens=4096, seed=SEED)
    codec = MossCodecDecoder(CODEC_DIR, device=torch.device("cuda"))
    segments = delayed_rows_to_segments(res.audio_frames)
    wavs = [codec.decode(seg, chunk_duration=8.0) for seg in segments]
    wav = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
    sf.write(out_path, wav, SR, subtype="FLOAT")
    fin = bool(np.isfinite(wav).all())
    dur = len(wav) / SR
    nonpad = (res.audio_frames != 1024).float().mean().item() * 100
    print(f"  [{label}] steps={res.n_steps} finished={res.finished} "
          f"segments={len(segments)} frames T={res.audio_frames.shape[0]} "
          f"non-pad codes={nonpad:.1f}%  wav={dur:.2f}s finite={fin}")
    del codec
    torch.cuda.empty_cache()
    ok = res.finished and res.audio_frames.shape[0] > 0 and fin and dur > 1.0
    return ok, dur, res


def main() -> int:
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    pg = torch.load(os.path.join(GOLDEN, "prompt_golden.pt"),
                    map_location="cpu", weights_only=False)
    gg = torch.load(os.path.join(GOLDEN, "gen_golden.pt"),
                    map_location="cpu", weights_only=False)
    names = [c["name"] for c in pg["cases"]]
    zh_i, en_i = names.index("zh_plain"), names.index("en_language")

    print("loading weights ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    fast = FastMossTTS(model, quant="w8")
    torch.cuda.empty_cache()
    print(f"after W8 quantize: {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
          f"(bf16 backbone linears freed, int8 + scales resident)")

    ok_a, ok_t, ok_aud = phase_a(fast, pg["input_ids"][zh_i].cuda())

    zh = dict(pg["cases"][zh_i])
    gt_zh = next(c for c in gg["cases"] if c["case_name"] == "zh_plain")
    en = dict(pg["cases"][en_i])
    gt_en = next(c for c in gg["cases"] if c["case_name"] == "en_language")

    print("=== Phase B: E2E zh/en (seed=1234, default sampling) ===")
    ok_zh, dur_zh, res_zh = _gen_wav(fast, pg["input_ids"][zh_i],
                                     pg["attention_mask"][zh_i],
                                     os.path.join(OUT, "m3_zh.wav"), "zh")
    print(f"  golden zh duration {gt_zh['n_steps']} steps / "
          f"{gt_zh.get('duration_s', 'n/a')}")
    ok_en, dur_en, res_en = _gen_wav(fast, pg["input_ids"][en_i],
                                     pg["attention_mask"][en_i],
                                     os.path.join(OUT, "m3_en.wav"), "en")

    print("=== Phase C: speed (zh) ===")
    stats: dict = {}
    torch.cuda.reset_peak_memory_stats()
    res_s = generate_fast(fast, {"input_ids": pg["input_ids"][zh_i].cuda(),
                                 "attention_mask": pg["attention_mask"][zh_i].cuda()},
                          max_new_tokens=4096, seed=SEED, stats=stats)
    step_ms = stats.get("step_ms") or []
    print(f"decode avg: {sum(step_ms) / len(step_ms):.2f} ms/step over "
          f"{len(step_ms)} steps -> {_steps_ps(step_ms):.1f} steps/s")
    print(f"decode steady (10+): {_steps_ps(step_ms, 10):.1f} steps/s "
          f"({sum(step_ms[10:]) / len(step_ms[10:]):.2f} ms/step)")
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"VRAM peak (decode only): {vram:.2f} GiB")
    ok_c = _steps_ps(step_ms, 10) >= 50.0 and vram < 12.0

    ok = ok_a and ok_zh and ok_en and ok_c
    print(f"test_fast_m3: phaseA={'PASS' if ok_a else 'FAIL'} "
          f"(text {ok_t:.2f}% audio {ok_aud:.2f}%) "
          f"zh={'OK' if ok_zh else 'BAD'} en={'OK' if ok_en else 'BAD'} "
          f"speed/vram={'PASS' if ok_c else 'FAIL'} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
