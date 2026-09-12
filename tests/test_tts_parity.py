"""TTS golden parity (task 5): teacher-forced logits + full-generation
trajectories vs asset-produced golden files.

  Phase A  .tmp/golden/logits_golden.npz  (zh prompt, seed=1234, 40 steps)
      logits_text  [40, 155648] fp32  raw last-position logits (pre-temperature,
                   pre-masking); audio heads already carry -inf at index 1024.
      logits_audio [40, 32, 1025] fp32
      selected_rows[40, 33] int64   the row fed to the next step
      Gate: max|delta| <= 0.05, argmax agreement >= 99% (per head set).

  Phase B  .tmp/golden/gen_golden.pt  (zh 137 steps + en 174 steps, seed=1234,
      default sampling params, raw delay-pattern generation_ids [T,33])
      Gate: report exact token agreement; 100% match expected when the RNG
      stream and state machine are faithful.

Run (GPU, under the lock):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock .tmp/gpu.lock \
      python3 -m moss_tts_lite.tests.test_tts_parity
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ..model import (
    AUDIO_END_TOKEN_ID,
    AUDIO_GEN_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    AUDIO_DELAY_SLOT_TOKEN_ID,
    IM_END_TOKEN_ID,
    N_VQ,
    PAD_TOKEN_ID,
    MossTTSModel,
)
from ..sampling import find_last_equal_C, sample_token
try:  # prefer the real loader (tok-delivered); fall back to the temp mini one
    from ..st_loader import read_safetensors
except ImportError:
    from ._mini_loader import read_safetensors_min as read_safetensors

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")
GOLDEN = os.path.join(ROOT, ".tmp", "golden")

PAD = PAD_TOKEN_ID
IM_START = 151644
IM_END = IM_END_TOKEN_ID
AUDIO_START = AUDIO_START_TOKEN_ID
AUDIO_END = AUDIO_END_TOKEN_ID
GEN_SLOT = AUDIO_GEN_SLOT_TOKEN_ID
DELAY_SLOT = AUDIO_DELAY_SLOT_TOKEN_ID
INT64_MAX = 9223372036854775807

TEXT_TEMPERATURE = 1.5
TEXT_TOP_P = 1.0
TEXT_TOP_K = 50
AUDIO_TEMPERATURE = 1.7
AUDIO_TOP_P = 0.8
AUDIO_TOP_K = 25
AUDIO_REPETITION_PENALTY = 1.0
SEED = 1234


@torch.inference_mode()
def replicate_record(model, input_ids, max_new_tokens, seed=SEED,
                     record_steps=0, label=""):
    """Step-by-step replication of the reference generate() (mirrors
    .tmp/golden/make_logits_golden.py exactly) on top of moss_tts_lite pieces.
    Records raw logits (pre-temperature, pre-masking) for the first
    `record_steps` steps and the selected next rows."""
    device = input_ids.device
    torch.manual_seed(seed)

    batch_size, seq_len, _ = input_ids.shape
    n_vq = N_VQ

    generation_ids = input_ids.clone()
    is_stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
    audio_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
    delayed_lengths = torch.full((batch_size,), INT64_MAX,
                                 dtype=torch.int64, device=device)

    is_continuation = ((input_ids[:, -1, 0] == AUDIO_START)
                       | (input_ids[:, -1, 0] == GEN_SLOT))
    audio_start_indices = find_last_equal_C(input_ids[..., 0], AUDIO_START)
    audio_start_mask = is_continuation & (audio_start_indices != -1)
    audio_lengths[audio_start_mask] = seq_len - audio_start_indices[audio_start_mask]
    is_audio = audio_start_mask.clone()

    pre_exclude_mask0 = torch.tensor([PAD, GEN_SLOT, DELAY_SLOT, AUDIO_END],
                                     device=device)
    pre_exclude_mask1 = torch.ones(155648, device=device).bool()
    pre_exclude_mask1[[GEN_SLOT, DELAY_SLOT]] = False

    rec_lt, rec_la, rec_rows = [], [], []
    n_steps = 0
    finished = False
    current_input_ids = None

    for time_step in range(max_new_tokens):
        if time_step == 0:
            hs = model.prefill(input_ids)
        else:
            hs = model.step(current_input_ids)
        h = hs.last_hidden[:, -1]                      # [1, hidden]
        n_steps = time_step + 1

        lt_raw = model.text_logits(h).float().clone()      # [V] raw
        la_raw = model.audio_logits(h).float().clone()     # [32, 1025] raw
        if time_step < record_steps:
            rec_lt.append(lt_raw.cpu())
            rec_la.append(la_raw.cpu())

        lt = (lt_raw / TEXT_TEMPERATURE).unsqueeze(0)  # [1, V] (golden keeps batch dim)
        la = la_raw / AUDIO_TEMPERATURE
        next_text_token = torch.full((batch_size,), PAD, device=device)
        next_text_token[~is_stopping & (delayed_lengths < n_vq)] = DELAY_SLOT
        is_audio_eos = ~is_stopping & (delayed_lengths == n_vq)
        next_text_token[is_audio_eos] = AUDIO_END
        is_audio[is_audio_eos] = False
        sampling_text_mask = ~is_stopping & (delayed_lengths > n_vq)
        lt[~is_audio] = lt[~is_audio].index_fill(-1, pre_exclude_mask0, float("-inf"))
        lt[is_audio] = lt[is_audio].masked_fill(pre_exclude_mask1, float("-inf"))
        if time_step == 0:
            lt[..., DELAY_SLOT] = float("-inf")
        if time_step <= n_vq:
            lt[..., IM_END] = float("-inf")

        next_text_token[sampling_text_mask] = sample_token(
            logits=lt[sampling_text_mask],
            top_p=TEXT_TOP_P, top_k=TEXT_TOP_K, do_sample=True)
        is_audio[next_text_token == AUDIO_START] = True
        is_stopping[next_text_token == IM_END] = True

        next_audio_tokens = torch.full((batch_size, n_vq), AUDIO_PAD_CODE,
                                       dtype=torch.int64, device=device)
        pre_audio_mask = (audio_lengths.unsqueeze(1)
                          > torch.arange(n_vq, dtype=torch.int64,
                                         device=device).expand(batch_size, n_vq))
        post_audio_mask = (torch.arange(n_vq, dtype=torch.int64, device=device)
                           .expand(batch_size, n_vq)
                           > delayed_lengths.unsqueeze(1) - 1)
        post_audio_mask[delayed_lengths == INT64_MAX] = True
        sampling_audio_mask = pre_audio_mask & post_audio_mask
        next_audio_tokens[~sampling_audio_mask] = AUDIO_PAD_CODE

        if sampling_audio_mask.sum() > 0:
            ch0 = la[0:1][sampling_audio_mask[:, 0]]        # [n0, 1025]
            rest = la[1:][sampling_audio_mask[0, 1:]]        # [n, 1025]
            ch0[..., AUDIO_PAD_CODE] = float("-inf")
            rest[..., AUDIO_PAD_CODE] = float("-inf")
            next_audio_tokens[:, 0][sampling_audio_mask[:, 0]] = sample_token(
                logits=ch0, prev_tokens=generation_ids[:, :, 1],
                repetition_penalty=AUDIO_REPETITION_PENALTY,
                top_p=AUDIO_TOP_P, top_k=AUDIO_TOP_K, do_sample=True)
            next_audio_tokens[:, 1:][sampling_audio_mask[:, 1:]] = sample_token(
                logits=rest, prev_tokens=generation_ids[:, :, 2:],
                repetition_penalty=AUDIO_REPETITION_PENALTY,
                top_p=AUDIO_TOP_P, top_k=AUDIO_TOP_K, do_sample=True)

        audio_lengths[(next_text_token == AUDIO_START)
                      | (next_text_token == GEN_SLOT)
                      | (next_text_token == DELAY_SLOT)] += 1
        audio_lengths[next_text_token == AUDIO_END] = 0
        delayed_lengths[(delayed_lengths == INT64_MAX)
                        & (next_text_token == DELAY_SLOT)] = 0
        delayed_lengths[delayed_lengths != INT64_MAX] += 1
        delayed_lengths[delayed_lengths > n_vq] = INT64_MAX

        current_input_ids = torch.cat([next_text_token[:, None, None],
                                       next_audio_tokens[:, None, :]], dim=2)
        generation_ids = torch.cat([generation_ids, current_input_ids], dim=1)
        if time_step < record_steps:
            rec_rows.append(current_input_ids[0, 0].cpu())  # [33]

        if is_stopping.sum() == batch_size:
            finished = True
            break

    rec = {
        "logits_text": torch.stack(rec_lt) if rec_lt else None,   # [S, V]
        "logits_audio": torch.stack(rec_la) if rec_la else None,  # [S, 32, 1025]
        "rows": torch.stack(rec_rows) if rec_rows else None,      # [S, 33]
    }
    return generation_ids, n_steps, finished, rec


def _agree_report(name, mine, ref):
    d = (mine - ref).abs()
    am_mine = mine.argmax(dim=-1)
    am_ref = ref.argmax(dim=-1)
    agree = (am_mine == am_ref).float().mean().item() * 100.0
    print(f"  {name}: max|d|={d.max().item():.4f} mean|d|={d.mean().item():.5f} "
          f"argmax agree={agree:.2f}%")
    return d.max().item(), agree


def phase_a(model, zh_ids):
    print("=== Phase A: teacher-forced 40-step logits parity (zh, seed=1234) ===")
    z = np.load(os.path.join(GOLDEN, "logits_golden.npz"))
    gt_text = torch.from_numpy(z["logits_text"]).cuda()          # [40, V]
    gt_audio = torch.from_numpy(z["logits_audio"]).cuda()        # [40, 32, 1025]
    gt_rows = torch.from_numpy(z["selected_rows"]).reshape(-1, 33).cuda()  # [40, 33]

    mine_text, mine_audio = [], []
    with torch.inference_mode():
        hs = model.prefill(zh_ids)
        # step-0 golden logits come from the full [L]-row forward -> same M here
        lt_all = model.text_logits(hs.last_hidden)
        la_all = model.audio_logits(hs.last_hidden)
        print(f"  [shape] last_hidden={tuple(hs.last_hidden.shape)} "
              f"text={tuple(lt_all.shape)} audio={tuple(la_all.shape)}")
        mine_text.append((lt_all[-1] if lt_all.dim() >= 2 else lt_all).float())
        mine_audio.append((la_all[-1] if la_all.dim() >= 3 else la_all).float())
        for t in range(1, gt_rows.shape[0]):
            hs = model.step(gt_rows[t - 1].unsqueeze(0))          # [1,1,33]
            mine_text.append(model.text_logits(hs.last_hidden).float())          # [V]
            mine_audio.append(model.audio_logits(hs.last_hidden).float())        # [32,1025]
    mine_text = torch.stack(mine_text)      # [40, V]
    mine_audio = torch.stack(mine_audio)    # [40, 32, 1025]

    # per-step diff (audio excludes pad col: -inf - -inf = nan on both sides)
    d_t = (mine_text - gt_text).abs()
    d_a = (mine_audio[..., :1024] - gt_audio[..., :1024]).abs()
    per_step = torch.maximum(d_t.amax(dim=1), d_a.amax(dim=(1, 2)))
    print("  per-step max|d| (text/audio):",
          [f"{v:.3g}" for v in per_step.tolist()])

    dmax_t, agree_t = _agree_report("logits_text ", mine_text, gt_text)
    dmax_a, agree_a = _agree_report("logits_audio", mine_audio[..., :1024],
                                    gt_audio[..., :1024])
    gate_d = max(dmax_t, dmax_a)
    gate_a = min(agree_t, agree_a)
    ok = gate_d <= 0.05 and gate_a >= 99.0
    print(f"  gate: max|d|={gate_d:.4f} (<=0.05), argmax={gate_a:.2f}% (>=99) "
          f"-> {'PASS' if ok else 'CHECK ABOVE'}")
    return ok


def phase_b(model, pg, gg):
    print("=== Phase B: full-generation parity (seed=1234, max_new_tokens=4096) ===")
    names = [c["name"] for c in pg["cases"]]
    all_ok = True
    for key, case_name in [("zh", "zh_plain"), ("en", "en_language")]:
        i = names.index(case_name)
        ids = pg["input_ids"][i].cuda()
        mask = pg["attention_mask"][i].cuda()
        gc = next(c for c in gg["cases"] if c["case_name"] == case_name)
        gt = gc["generation_ids"]                   # [T, 33] cpu int64
        from ..generate import generate
        res = generate(model, {"input_ids": ids, "attention_mask": mask},
                       max_new_tokens=4096, seed=SEED,
                       text_temperature=TEXT_TEMPERATURE, text_top_p=TEXT_TOP_P,
                       text_top_k=TEXT_TOP_K,
                       audio_temperature=AUDIO_TEMPERATURE, audio_top_p=AUDIO_TOP_P,
                       audio_top_k=AUDIO_TOP_K,
                       audio_repetition_penalty=AUDIO_REPETITION_PENALTY)
        t = res.text_ids.cpu()
        a = res.audio_frames.cpu()
        T = gt.shape[0]
        t_match = (t.reshape(-1) == gt[:, 0]).float().mean().item() * 100 if t.numel() == T else float("nan")
        a_match = (a == gt[:, 1:]).float().mean().item() * 100 if a.shape == gt[:, 1:].shape else float("nan")
        steps_ok = res.n_steps == gc["n_steps"]
        fin_ok = res.finished == gc["finished"]
        exact = (t.numel() == T and a.shape == gt[:, 1:].shape
                 and torch.equal(t.reshape(-1), gt[:, 0])
                 and torch.equal(a, gt[:, 1:]))
        print(f"  [{key}] T: mine={t.numel()} golden={T}; n_steps mine={res.n_steps} "
              f"golden={gc['n_steps']}; finished mine={res.finished} golden={gc['finished']}")
        print(f"  [{key}] text match={t_match:.2f}%  audio match={a_match:.2f}%  "
              f"EXACT={exact}")
        if not exact:
            # first divergence
            L = min(t.numel(), T)
            fd = next((j for j in range(L)
                       if t.reshape(-1)[j] != gt[j, 0]
                       or not torch.equal(a[j], gt[j, 1:])), L)
            print(f"  [{key}] first diverging generated row: {fd} "
                  f"(mine text={int(t.reshape(-1)[fd]) if fd < t.numel() else '?'} "
                  f"golden text={int(gt[fd, 0])})")
            all_ok = False
        all_ok = all_ok and exact and steps_ok and fin_ok
    return all_ok


def main() -> int:
    assert torch.cuda.is_available()
    pg = torch.load(os.path.join(GOLDEN, "prompt_golden.pt"),
                    map_location="cpu", weights_only=False)
    gg = torch.load(os.path.join(GOLDEN, "gen_golden.pt"),
                    map_location="cpu", weights_only=False)

    print("loading weights via moss_tts_lite.st_loader ...", flush=True)
    weights = read_safetensors(MODEL_DIR)
    model = MossTTSModel(weights, device=torch.device("cuda"),
                         dtype=torch.bfloat16, max_seq_len=8192)
    del weights
    torch.cuda.empty_cache()

    names = [c["name"] for c in pg["cases"]]
    zh_ids = pg["input_ids"][names.index("zh_plain")].cuda()

    ok_a = phase_a(model, zh_ids)
    torch.cuda.empty_cache()
    ok_b = phase_b(model, pg, gg)
    print(f"tts_parity: phase_a={'PASS' if ok_a else 'FAIL'} "
          f"phase_b={'PASS' if ok_b else 'FAIL'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
