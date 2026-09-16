"""CUDA Graphs v2: multi-bucket graphs (no compile), dedicated box.
Phases:
  1. shape combo census
  2. bucket-pad numeric check
  3. eager natural baseline x2
  4. eager bucketed control (isolates padding GPU cost)
  5. multi-bucket graphs, shared pool, lazy capture per (B, T_bucket)
"""
import json, os, sys, time, random, statistics
from collections import Counter
from pathlib import Path
import torch
from torch.optim import AdamW

QUANT = os.environ.get("QUANT", "0") == "1"
USE_LIGER = os.environ.get("LIGER", "0") == "1"
sys.path.insert(0, "/home/ubuntu")
import moss_tts_lite.nn as nn_mod
nn_mod.LIGER_ENABLED = USE_LIGER
from moss_tts_lite.train import (_detect_model_class, load_tokenizer,
                                 wrap_lora, DEFAULT_LORA_TARGETS,
                                 quantize_model_4bit, TokenBudgetBatchSampler)
from moss_tts_lite.data import MossTTSTrainDataset
from moss_tts_lite.model import AUDIO_PAD_CODE

MODEL_DIR = "/home/ubuntu/models/MOSS-TTS-Local-Transformer-v1.5"
JSONL = "/home/ubuntu/dataset/train_codes_v2.jsonl"
DEV = torch.device("cuda")
torch.manual_seed(42); random.seed(42)

tokenizer = load_tokenizer(MODEL_DIR)
cfg = json.loads((Path(MODEL_DIR) / "config.json").read_text(encoding="utf-8"))
slot_ids = (int(cfg["audio_user_slot_token_id"]),
            int(cfg["audio_assistant_slot_token_id"]))
records = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
dataset = MossTTSTrainDataset(records, tokenizer, slot_ids=slot_ids)
lengths = [int(dataset.pack_record(r)["input_ids"].shape[0]) for r in records]
sampler = TokenBudgetBatchSampler(lengths, 8, 900, seed=42)
batches = []
it = iter(sampler)
for _ in range(16):
    idx = next(it)
    b = dataset.collate_fn([dataset[i] for i in idx])
    batches.append({k: v.to(DEV) for k, v in b.items()})
Ts = [b["input_ids"].shape[1] for b in batches]
combos = Counter((b["input_ids"].shape[0], (b["input_ids"].shape[1] + 63) // 64 * 64) for b in batches)
print(f"batches=16 quant={QUANT} liger={USE_LIGER}", flush=True)
print("shape combos:", dict(sorted(combos.items())), flush=True)

_, model_cls = _detect_model_class(MODEL_DIR)
if QUANT:
    model = model_cls.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device=torch.device("cpu"))
    quantize_model_4bit(model, target_device=DEV)
    model.to(DEV)
else:
    model = model_cls.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device=DEV)
model = wrap_lora(model, 64, 128, DEFAULT_LORA_TARGETS.split(","), 0.0)
model.gradient_checkpointing_enable()
model.train()
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True)
CW = [1.0] + [32.0 / 12.0] * 12

def fwd(ids, mask, lab):
    return model(input_ids=ids, attention_mask=mask, labels=lab,
                 channelwise_loss_weight=CW)

def tb_of(T):
    return (T + 63) // 64 * 64

def pad_bucket(b):
    ids, mask, lab = b["input_ids"], b["attention_mask"], b["labels"]
    B, T, C = ids.shape
    TB = tb_of(T)
    o_ids = torch.full((B, TB, C), AUDIO_PAD_CODE, dtype=ids.dtype, device=DEV)
    o_ids[:, TB - T:] = ids
    o_mask = torch.zeros(B, TB, dtype=mask.dtype, device=DEV)
    o_mask[:, TB - T:] = mask
    o_lab = torch.full((B, TB, C), -100, dtype=lab.dtype, device=DEV)
    o_lab[:, TB - T:] = lab
    return o_ids, o_mask, o_lab

# ---- bucket numeric check ----
model.eval()
with torch.no_grad():
    l_nat = fwd(batches[0]["input_ids"], batches[0]["attention_mask"], batches[0]["labels"]).loss.item()
    ai, am, al = pad_bucket(batches[0])
    l_bkt = fwd(ai, am, al).loss.item()
model.train()
rel = abs(l_bkt - l_nat) / abs(l_nat)
print(f"BUCKET_CHECK natural={l_nat:.8f} bucket={l_bkt:.8f} rel={rel:.3e}", flush=True)
assert rel < 5e-3

import os as _os2
_pinned = _os2.environ.get("PINNED", "0") == "1"
cpu_batches = [{k: (v.cpu().pin_memory() if _pinned else v.cpu())
                for k, v in b.items()} for b in batches]
bi = 0
def eager_h2d_step():
    global bi
    b = cpu_batches[bi % len(batches)]; bi += 1
    g = {k: v.to(DEV, non_blocking=True) for k, v in b.items()}
    out = fwd(g["input_ids"], g["attention_mask"], g["labels"])
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    return out.loss.detach()

def eager_step(bucketed=False):
    global bi
    b = batches[bi % len(batches)]; bi += 1
    if bucketed:
        i, m, l = pad_bucket(b)
        out = fwd(i, m, l)
    else:
        out = fwd(b["input_ids"], b["attention_mask"], b["labels"])
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    return out.loss.detach()

def bench(fn, tag, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ls = []
    t0 = time.perf_counter()
    for _ in range(n):
        ls.append(fn())
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"{tag:14s} {dt*1000:.1f}ms/step ({1/dt:.2f}/s) loss {ls[0].item():.6f}->{ls[-1].item():.6f}", flush=True)
    return dt

d_e1 = bench(lambda: eager_step(False), "EAGER_NAT")
d_e2 = bench(lambda: eager_step(False), "EAGER_NAT")
d_eb = bench(lambda: eager_step(True), "EAGER_BUCKET")
import os as _os
if _os.environ.get("H2D", "1") == "1":
    d_h2d = bench(eager_h2d_step, "EAGER_H2D")

# ---- multi-bucket graphs ----
POOL = torch.cuda.graph_pool_handle()
GRAPHS = {}

def get_graph(B, TB):
    key = (B, TB)
    if key in GRAPHS:
        return GRAPHS[key]
    src = next((b for b in batches if b["input_ids"].shape[0] == B
                and tb_of(b["input_ids"].shape[1]) == TB), None)
    assert src is not None, f"no batch for {key}"
    C = src["input_ids"].shape[2]
    s_ids = torch.zeros(B, TB, C, dtype=torch.long, device=DEV)
    s_mask = torch.zeros(B, TB, dtype=torch.bool, device=DEV)
    s_lab = torch.full((B, TB, C), -100, dtype=torch.long, device=DEV)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            i, m, l = pad_bucket(src)
            fwd(i, m, l).loss.backward()
        opt.zero_grad(set_to_none=False)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=False)
    with torch.cuda.graph(g, pool=POOL):
        static_loss = fwd(s_ids, s_mask, s_lab).loss
        static_loss.backward()
    GRAPHS[key] = (g, s_ids, s_mask, s_lab, static_loss)
    return GRAPHS[key]

def graph_step():
    global bi
    b = batches[bi % len(batches)]; bi += 1
    B, T, _ = b["input_ids"].shape
    g, s_ids, s_mask, s_lab, static_loss = get_graph(B, tb_of(T))
    i, m, l = pad_bucket(b)
    s_ids.copy_(i); s_mask.copy_(m); s_lab.copy_(l)
    g.replay()
    opt.step()
    opt.zero_grad(set_to_none=False)
    return static_loss.detach().clone()

d_g1 = bench(graph_step, "GRAPH_MB", warm=18)   # full epoch warm: capture every shape first
d_g2 = bench(graph_step, "GRAPH_MB")
print(f"graphs captured: {len(GRAPHS)} peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
ea = statistics.mean([d_e1, d_e2])
print(f"SPEEDUP graph_vs_eager = {ea / statistics.mean([d_g1, d_g2]):.3f}x "
      f"(nat {ea*1000:.1f}ms, bucketed {d_eb*1000:.1f}ms, graph {statistics.mean([d_g1, d_g2])*1000:.1f}ms)", flush=True)
