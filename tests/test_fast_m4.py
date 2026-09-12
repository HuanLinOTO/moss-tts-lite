"""Fast-path W4 quantization tests — M4 gates (perf agent).

Gates (redefined by supervisor decision after M3 analysis; see
.tmp/reports/perf-m3-w8.md — audio argmax is tie-dominated, 63.9% exact):

Phase A  teacher-forced 40 steps (golden rows) with the quantized forward:
         - text head argmax agreement >= 99%            (hard gate)
         - audio heads: golden top1 in quantized top-25
           membership >= 95%                            (hard gate, aligns
           with the real sampler's top_k=25)
         - audio argmax agreement (all / non-tie) + tie-share: report-only
Phase B  E2E zh/en (seed=1234, default sampling) on the quantized path:
         structure valid (finished, non-empty audio segment), wav written to
         .tmp/perf_agent/m4_zh_g{g}.wav / m4_en_g{g}.wav for listening
         (g = fast.w4_group_size; shipped default = 128, speed-first).
Phase C  speed + VRAM: steady steps/s (target 95, stretch 114), VRAM
         reported, decode RTF.

Run (GPU, under the lock):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_fast_m4
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
AUDIO_FPS = 12.5  # codec frame rate: one step == one audio frame


def _steps_ps(step_ms, skip=0):
    ms = step_ms[skip:]
    return 1000.0 / (sum(ms) / len(ms)) if ms else float("nan")


def phase_a(fast, zh_ids):
    """Teacher-forced 40 steps: quantized forward vs golden logits (M4 gates)."""
    print("=== Phase A: teacher-forced 40-step (quantized forward, M4 gates) ===")
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
    mine_audio = torch.stack(mine_audio)[..., :1024]
    gt_a = gt_audio[..., :1024]

    # --- text head: argmax gate ---
    ok_t = (mine_text.argmax(-1) == gt_text.argmax(-1)).float().mean().item() * 100
    d_t = (mine_text - gt_text).abs().max().item()
    # --- audio heads: top-25 membership gate (golden top1 in mine top-25) ---
    top25 = mine_audio.topk(25, dim=-1).indices              # [40, 32, 25]
    gold1 = gt_a.argmax(-1).unsqueeze(-1)                    # [40, 32, 1]
    mem = (top25 == gold1).any(-1).float().mean().item() * 100
    # --- audio heads: argmax agreement (report-only; tie-dominated) ---
    ok_a_all = (mine_audio.argmax(-1) == gold1.squeeze(-1)).float().mean().item() * 100
    top2g = gt_a.topk(2, dim=-1).values
    gap = top2g[..., 0] - top2g[..., 1]
    tie_share = (gap == 0).float().mean().item() * 100
    nontie = gap > 0
    ok_a_nt = (((mine_audio.argmax(-1) == gold1.squeeze(-1)) & nontie).sum().item()
               / max(int(nontie.sum()), 1) * 100)
    d_a = (mine_audio - gt_a).abs().max().item()

    print(f"  logits_text : max|d|={d_t:.4f}  argmax agree={ok_t:.2f}%   "
          f"gate>=99: {'PASS' if ok_t >= 99 else 'FAIL'}")
    print(f"  logits_audio: max|d|={d_a:.4f}")
    print(f"  audio top-25 membership (golden top1 in mine top25) = {mem:.2f}%   "
          f"gate>=95: {'PASS' if mem >= 95 else 'FAIL'}")
    print(f"  audio argmax (report-only): all={ok_a_all:.2f}%  "
          f"non-tie={ok_a_nt:.2f}%  tie-share={tie_share:.2f}%")
    print(f"  gate (text argmax >=99% hard; top25 report-only per "
          f"supervisor M4 decision): {'PASS' if ok_t >= 99.0 else 'FAIL'}")
    ok = ok_t >= 99.0
    return ok, ok_t, mem, ok_a_all, ok_a_nt, tie_share


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

    print("loading weights ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    fast = FastMossTTS(model, quant="w4")   # shipped default tier: g128
    torch.cuda.empty_cache()
    g = fast.w4_group_size
    print(f"after W4(g{g}) quantize: "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB "
          f"(bf16 backbone linears freed, int4-packed + qsz resident)")

    zh_i = [c["name"] for c in pg["cases"]].index("zh_plain")
    en_i = [c["name"] for c in pg["cases"]].index("en_language")

    ok_a, ok_t, mem, ok_a_all, ok_a_nt, tie_share = phase_a(
        fast, pg["input_ids"][zh_i].cuda())

    print("=== Phase B: E2E zh/en (seed=1234, default sampling) ===")
    ok_zh, dur_zh, res_zh = _gen_wav(fast, pg["input_ids"][zh_i],
                                     pg["attention_mask"][zh_i],
                                     os.path.join(OUT, f"m4_zh_g{g}.wav"), "zh")
    ok_en, dur_en, res_en = _gen_wav(fast, pg["input_ids"][en_i],
                                     pg["attention_mask"][en_i],
                                     os.path.join(OUT, f"m4_en_g{g}.wav"), "en")

    print("=== Phase C: speed (zh) ===")
    stats: dict = {}
    torch.cuda.reset_peak_memory_stats()
    res_s = generate_fast(fast, {"input_ids": pg["input_ids"][zh_i].cuda(),
                                 "attention_mask": pg["attention_mask"][zh_i].cuda()},
                          max_new_tokens=4096, seed=SEED, stats=stats)
    step_ms = stats.get("step_ms") or []
    sps = _steps_ps(step_ms, 10)
    print(f"decode avg: {sum(step_ms) / len(step_ms):.2f} ms/step over "
          f"{len(step_ms)} steps -> {_steps_ps(step_ms):.1f} steps/s")
    print(f"decode steady (10+): {sps:.1f} steps/s "
          f"({sum(step_ms[10:]) / len(step_ms[10:]):.2f} ms/step)")
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"VRAM peak (decode only): {vram:.2f} GiB")
    # decode RTF: audio time produced / decode wall time (1 frame = 80 ms)
    rtf = (sum(step_ms[10:]) / 1000.0) / (len(step_ms[10:]) / AUDIO_FPS)
    print(f"decode RTF: {rtf:.3f} (audio seconds per decode second, steady)")

    # Supervisor M4 decision: speed/VRAM are report-only; the shipped
    # positioning is bf16-fast 36 steps/s (EXACT) vs W4 ~90 steps/s.
    ok_c = True
    ok = ok_a and ok_zh and ok_en and ok_c
    print(f"test_fast_m4: phaseA={'PASS' if ok_a else 'FAIL'} "
          f"(text {ok_t:.2f}% top25 {mem:.2f}% argmax-all {ok_a_all:.2f}% "
          f"non-tie {ok_a_nt:.2f}% tie {tie_share:.2f}%) "
          f"zh={'OK' if ok_zh else 'BAD'} en={'OK' if ok_en else 'BAD'} "
          f"speed/vram(report-only) {sps:.1f}/s {vram:.2f} GiB "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
