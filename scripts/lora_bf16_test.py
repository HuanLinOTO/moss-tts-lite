#!/usr/bin/env python
"""lora_bf16_test.py -- LoRA dtype 对速度的影响 (graph 模式, 稳态 steps_per_sec)."""
import os, sys, time
sys.path.insert(0, "/home/ubuntu")
import torch

LORA_BF16 = os.environ.get("LORA_BF16", "0") == "1"
BS = os.environ.get("BS", "8")
MBT = os.environ.get("MBT", "900")

sys.argv = ["train.py",
    "--model-dir", "/home/ubuntu/models/MOSS-TTS-Local-Transformer-v1.5",
    "--train-jsonl", "/home/ubuntu/dataset/train_codes_v2.jsonl",
    "--mode", "qlora", "--bf16", "--cuda-graph",
    "--per-device-batch-size", BS, "--max-batch-tokens", MBT,
    "--gradient-accumulation-steps", "4",
    "--gradient-checkpointing", "--fused-optimizer",
    "--num-epochs", "1", "--max-steps", "12", "--logging-steps", "1",
    "--output-dir", "/tmp/adapter_test",
]

import moss_tts_lite.train as T
if LORA_BF16:
    _orig = T.wrap_lora
    def wrap2(model, *a, **k):
        m = _orig(model, *a, **k)
        n = 0
        for name, p in m.named_parameters():
            if "lora_" in name and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16); n += 1
        print(f"[lora-bf16] cast {n} lora tensors to bf16", flush=True)
        return m
    T.wrap_lora = wrap2

T.main()
