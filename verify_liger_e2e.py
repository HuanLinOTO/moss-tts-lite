"""集成级验证: 同权重 LIGER off/on 单批 loss 对比 + 交替训练测速."""
import json, sys, time, random
from pathlib import Path
import torch
from torch.optim import AdamW

sys.path.insert(0, "/home/lxm/Projects/moss-workspace/moss-tts-lite")
import moss_tts_lite.nn as nn_mod
from moss_tts_lite.train import (_detect_model_class, load_tokenizer,
                                 wrap_lora, DEFAULT_LORA_TARGETS,
                                 quantize_model_4bit, TokenBudgetBatchSampler)
from moss_tts_lite.data import MossTTSTrainDataset

MODEL_DIR = "/home/lxm/Projects/models/MOSS-TTS-Local-Transformer-v1.5"
JSONL = "/home/lxm/Projects/dataset/纳西妲/train_codes_v2.jsonl"
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
print(f"batches={len(batches)} liger_installed", flush=True)

_, model_cls = _detect_model_class(MODEL_DIR)
model = model_cls.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device=torch.device("cpu"))
quantize_model_4bit(model, target_device=DEV)
model.to(DEV)
model = wrap_lora(model, 64, 128, DEFAULT_LORA_TARGETS.split(","), 0.0)
model.gradient_checkpointing_enable()
model.train()
opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True)
channelwise = [1.0] + [32.0 / 12.0] * 12

bi = 0
def step():
    global bi
    b = batches[bi % len(batches)]; bi += 1
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                labels=b["labels"], channelwise_loss_weight=channelwise)
    out.loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    return out.loss.detach()

# ---- A. 同权重 no_grad loss: off vs on, 前 3 个批 ----
losses = {}
for mode in [False, True]:
    nn_mod.LIGER_ENABLED = mode
    ls = []
    with torch.no_grad():
        model.eval()
        for i in range(3):
            b = batches[i]
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        labels=b["labels"], channelwise_loss_weight=channelwise)
            ls.append(out.loss.item())
        model.train()
    losses[mode] = ls
rel = abs(losses[True][0] - losses[False][0]) / max(abs(losses[False][0]), 1e-9)
print(f"LOSS off={[f'{x:.6f}' for x in losses[False]]}", flush=True)
print(f"LOSS on ={[f'{x:.6f}' for x in losses[True]]}", flush=True)
print(f"LOSS rel_diff(first batch) = {rel:.3e}", flush=True)

# ---- B. 交替测速 off/on x2 ----
def bench(mode, n=30, warm=5):
    nn_mod.LIGER_ENABLED = mode
    for _ in range(warm):
        step()
    torch.cuda.synchronize()
    ls = []
    t0 = time.perf_counter()
    for _ in range(n):
        ls.append(step())
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"{'LIGER_ON ' if mode else 'LIGER_OFF'} {dt*1000:.1f}ms/step ({1/dt:.2f}/s) "
          f"loss {ls[0].item():.6f}->{ls[-1].item():.6f} "
          f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    return dt

d_off1 = bench(False)
d_on1 = bench(True)
d_off2 = bench(False)
d_on2 = bench(True)
import statistics
off = statistics.mean([d_off1, d_off2]); on = statistics.mean([d_on1, d_on2])
print(f"SPEEDUP = {off/on:.3f}x  (off {off*1000:.1f}ms, on {on*1000:.1f}ms)", flush=True)
