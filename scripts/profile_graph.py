#!/usr/bin/env python
"""profile_graph.py -- graph 模式瓶颈打点: 分段计时 + torch.profiler 稳态窗口.

计时锚点(全部外部 patch, 不改 train.py):
  loader_next  DataLoaderIter._next_data (dataset.getitem + collate + pin, 主线程)
  copy_        Tensor.copy_  (graph static input 拷贝 x3/micro + pad 内部)
  replay       CUDAGraph.replay (CPU 提交侧)
  clip         clip_grad_norm_
  opt_step     Optimizer.step (fused adamw)
  sched        LRScheduler.step
  zero_grad    Optimizer.zero_grad  (时间戳 -> opt 周期 wall)
profiler: 稳态后抓 N 个完整 opt 周期 (zero->zero), key_averages + trace.
"""
import os, sys, time, json
from collections import defaultdict

sys.path.insert(0, "/home/ubuntu")

import torch
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

PROF_OUT = "/home/ubuntu/prof_out"
os.makedirs(PROF_OUT, exist_ok=True)

sys.argv = ["train.py",
    "--model-dir", "/home/ubuntu/models/MOSS-TTS-Local-Transformer-v1.5",
    "--train-jsonl", "/home/ubuntu/dataset/train_codes_v2.jsonl",
    "--mode", "qlora", "--bf16", "--cuda-graph",
    "--per-device-batch-size", "8", "--max-batch-tokens", "900",
    "--gradient-accumulation-steps", "4",
    "--gradient-checkpointing", "--fused-optimizer",
    "--num-epochs", "1", "--max-steps", "30", "--logging-steps", "1",
    "--output-dir", "/home/ubuntu/prof_out/adapter",
]

stats = defaultdict(float); counts = defaultdict(int)
zero_ts = []

def wrap(cls, name, key):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        stats[key] += time.perf_counter() - t0; counts[key] += 1
        return r
    setattr(cls, name, w)
    return orig

# loader (collate 主线程)
_orig_next = _SingleProcessDataLoaderIter._next_data
def _next(self):
    t0 = time.perf_counter()
    r = _orig_next(self)
    stats["loader_next"] += time.perf_counter() - t0; counts["loader_next"] += 1
    return r
_SingleProcessDataLoaderIter._next_data = _next

wrap(torch.Tensor, "copy_", "copy_")
wrap(torch.cuda.CUDAGraph, "replay", "replay")
wrap(torch.nn.utils, "clip_grad_norm_", "clip")
_opt_orig = Optimizer.step
wrap(Optimizer, "step", "opt_step")
wrap(LRScheduler, "step", "sched")

# zero_grad: 计时 + 时间戳 + profiler 窗口控制
_zero_orig = Optimizer.zero_grad
PROF_START, PROF_STOP = 46, 48   # 跳过捕获期(每 graph 3 次 build zero)
from torch.profiler import profile, ProfilerActivity
prof_box = {"p": None, "dumped": False}

def _zero(self, *a, **k):
    n = counts["zero_grad"]
    if n == PROF_STOP and prof_box["p"] is not None and not prof_box["dumped"]:
        prof_box["p"].__exit__(None, None, None)
        _dump_prof(prof_box["p"]); prof_box["dumped"] = True
    t0 = time.perf_counter()
    r = _zero_orig(self, *a, **k)
    stats["zero_grad"] += time.perf_counter() - t0; counts["zero_grad"] += 1
    zero_ts.append(time.perf_counter())
    if n == PROF_START and prof_box["p"] is None:
        prof_box["p"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof_box["p"].__enter__()
    return r
Optimizer.zero_grad = _zero

def _dump_prof(p):
    ka = p.key_averages()
    with open(f"{PROF_OUT}/key_averages.txt", "w") as f:
        f.write(ka.table(sort_by="self_cpu_time_total", row_limit=60) + "\n\n")
        f.write(ka.table(sort_by="self_cuda_time_total", row_limit=60))
    p.export_chrome_trace(f"{PROF_OUT}/trace.json")
    print("[prof] key_averages + trace dumped", flush=True)

import moss_tts_lite.train as T
t_main0 = time.perf_counter()
T.main()
wall = time.perf_counter() - t_main0

# ---- 汇总 -------------------------------------------------------------
# 稳态 opt 周期: 取最后 8 个 zero_ts 间隔
periods = [b - a for a, b in zip(zero_ts[:-1], zero_ts[1:])]
steady = periods[-10:-2] if len(periods) >= 12 else periods
per_opt = sum(steady) / len(steady) if steady else float("nan")

acc = defaultdict(float); ncnt = defaultdict(int)
# 每周期 = 4 micro; 尾段(捕获期)剔除: 只统计 zero_ts 稳态窗口内? 简化: 全量平均 x4
def per_micro(key):
    return stats[key] / max(counts[key], 1)
def share(key):
    if key in ("loader_next", "copy_", "replay"):
        return per_micro(key) * 4 / per_opt * 100
    return per_micro(key) / per_opt * 100

rows = {}
for k in ("loader_next", "copy_", "replay", "clip", "opt_step", "sched", "zero_grad"):
    rows[k] = dict(total_s=round(stats[k], 3), n=counts[k],
                   avg_ms=round(per_micro(k) * 1e3, 3),
                   share_pct=round(share(k), 1))
known = sum(r["share_pct"] for r in rows.values())
summary = dict(
    wall_s=round(wall, 1), opt_steps=counts["opt_step"],
    steady_opt_period_s=round(per_opt, 4),
    steady_opt_per_s=round(1 / per_opt, 3) if per_opt else None,
    segments=rows, residual_pct=round(100 - known, 1),
    note="residual = H2D + pad + float() sync + python loop",
)
print("\n===== TIMING SUMMARY =====")
print(json.dumps(summary, indent=2), flush=True)
with open(f"{PROF_OUT}/timing.json", "w") as f:
    json.dump(summary, f, indent=2)
