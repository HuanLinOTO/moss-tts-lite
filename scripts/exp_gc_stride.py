#!/usr/bin/env python
"""exp_gc_stride.py -- 隔层 gradient checkpointing 收益验证 (graph+bf16lora)."""
import os, sys
sys.path.insert(0, "/home/ubuntu")
STRIDE = int(os.environ.get("GC_STRIDE", "2"))   # 2=每2层checkpoint1层, 1=全checkpoint
import torch
import torch.utils.checkpoint as cp
import moss_tts_lite.nn as nn_mod

def strided_fw(self, inputs_embeds, attn_bias, gradient_checkpointing=False):
    h = inputs_embeds
    for i, layer in enumerate(self.layers):
        if (gradient_checkpointing and self.training
                and (STRIDE == 1 or i % STRIDE == 0)):
            h = cp.checkpoint(layer, h, attn_bias["cos"], attn_bias["sin"],
                              attn_bias["mask"], use_reentrant=False,
                              preserve_rng_state=False)
        else:
            h = layer(h, attn_bias["cos"], attn_bias["sin"], attn_bias["mask"])
    return self.norm(h)
nn_mod.MossBackbone.forward = strided_fw
print(f"[gc-stride] STRIDE={STRIDE}", flush=True)

sys.argv = ["train.py",
    "--model-dir", "/home/ubuntu/models/MOSS-TTS-Local-Transformer-v1.5",
    "--train-jsonl", "/home/ubuntu/dataset/train_codes_v2.jsonl",
    "--mode", "qlora", "--bf16", "--cuda-graph", "--lora-dtype", "bf16",
    "--per-device-batch-size", "8", "--max-batch-tokens", "900",
    "--gradient-accumulation-steps", "4",
    "--gradient-checkpointing", "--fused-optimizer",
    "--num-epochs", "1", "--max-steps", "16", "--logging-steps", "1",
    "--output-dir", "/tmp/adp_gcs",
]
import moss_tts_lite.train as T
T.main()
