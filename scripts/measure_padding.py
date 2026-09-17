#!/usr/bin/env python
"""measure_padding.py -- padding 浪费率测量: real vs 批内pad vs 64桶."""
import sys, json
sys.path.insert(0, "/home/ubuntu")
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from moss_tts_lite.data import MossTTSTrainDataset
from moss_tts_lite.train import (load_jsonl, load_tokenizer, _detect_model_class,
                                 TokenBudgetBatchSampler)

MD = "/home/ubuntu/models/MOSS-TTS-Local-Transformer-v1.5"
records = load_jsonl("/home/ubuntu/dataset/train_codes_v2.jsonl")
records = [r for r in records if len(r["audio_codes"]) <= 3000]
tok = load_tokenizer(MD)
raw = json.loads((Path(MD) / "config.json").read_text())
slots = (int(raw["audio_user_slot_token_id"]), int(raw["audio_assistant_slot_token_id"]))
ds = MossTTSTrainDataset(records, tok, slot_ids=slots)
lengths = [int(ds.pack_record(r)["input_ids"].shape[0]) for r in records]
smp = TokenBudgetBatchSampler(lengths, 8, 900, seed=42)

tot_real = tot_bt = tot_btb64 = tot_btb32 = 0
n_micro = 0
for idxs in smp:
    batch = ds.collate_fn([ds[i] for i in idxs])
    mask = batch["attention_mask"]
    B, T = mask.shape
    real = int(mask.sum())
    tb64 = (T + 63) // 64 * 64
    tb32 = (T + 31) // 32 * 32
    tot_real += real; tot_bt += B * T
    tot_btb64 += B * tb64; tot_btb32 += B * tb32
    n_micro += 1

print(f"micro batches: {n_micro}")
print(f"in-batch padding waste : real/(B*T)     = {tot_real/tot_bt:.3f}  (waste {1-tot_real/tot_bt:.1%})")
print(f"bucket-64 efficiency   : real/(B*Tb64)  = {tot_real/tot_btb64:.3f}  (waste {1-tot_real/tot_btb64:.1%})")
print(f"bucket-32 efficiency   : real/(B*Tb32)  = {tot_real/tot_btb32:.3f}  (waste {1-tot_real/tot_btb32:.1%})")
print(f"perfect varlen (packed): efficiency     = 1.000 (M = sum(real) = {tot_real} tokens)")
