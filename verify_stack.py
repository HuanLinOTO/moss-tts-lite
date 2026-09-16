"""叠加实验: verify 结构 -> CLI 结构逐层加特征, 定位 289ms/micro 缺口."""
import json, os, sys, time, random
from pathlib import Path
import torch
from torch.optim import AdamW

sys.path.insert(0, "/home/ubuntu")
from moss_tts_lite.train import (_detect_model_class, load_tokenizer,
                                 wrap_lora, DEFAULT_LORA_TARGETS,
                                 quantize_model_4bit, TokenBudgetBatchSampler)
from moss_tts_lite.data import MossTTSTrainDataset

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
gpu_batches = []
it = iter(sampler)
for _ in range(16):
    idx = next(it)
    b = dataset.collate_fn([dataset[i] for i in idx])
    gpu_batches.append({k: v.to(DEV) for k, v in b.items()})
cpu_batches = [{k: v.cpu() for k, v in b.items()} for b in gpu_batches]

_, model_cls = _detect_model_class(MODEL_DIR)
model = model_cls.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device=torch.device("cpu"))
quantize_model_4bit(model, target_device=DEV)
model.to(DEV)
model = wrap_lora(model, 64, 128, DEFAULT_LORA_TARGETS.split(","), 0.0)
model.gradient_checkpointing_enable()
model.train()
trainable = [p for p in model.parameters() if p.requires_grad]
opt = AdamW(trainable, lr=1e-4, fused=True)
CW = [1.0] + [32.0 / 12.0] * 12
ACC = 4

def fwd(ids, mask, lab):
    return model(input_ids=ids, attention_mask=mask, labels=lab,
                 channelwise_loss_weight=CW)

bi = 0
def make_step(stage):
    """stage bits: 1=accum4 2=clip 4=float_sync 8=cpu_to_dev"""
    def step():
        global bi
        b = cpu_batches[bi % 16] if (stage & 8) else gpu_batches[bi % 16]
        bi += 1
        if stage & 8:
            b = {k: v.to(DEV, non_blocking=True) for k, v in b.items()}
        out = fwd(b["input_ids"], b["attention_mask"], b["labels"])
        if stage & 1:
            (out.loss / ACC).backward()
            if bi % ACC == 0:
                if stage & 2:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(set_to_none=False)
                if stage & 4:
                    _ = float(out.loss.detach())
        else:
            out.loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
        return out.loss.detach()
    return step

def bench(fn, tag, n=40, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    print(f"{tag:28s} {dt*1000:7.1f}ms/step", flush=True)
    return dt

bench(make_step(0), "S0 verify baseline")
bench(make_step(1), "S1 +accum4")
bench(make_step(1 | 2), "S2 +clip")
bench(make_step(1 | 2 | 4), "S3 +float_sync")
bench(make_step(1 | 2 | 4 | 8), "S4 +cpu_to_dev")

# S5: real DataLoader (same construction as train.py)
from torch.utils.data import DataLoader
loader = DataLoader(dataset,
                    batch_sampler=TokenBudgetBatchSampler(lengths, 8, 900, seed=42),
                    collate_fn=dataset.collate_fn, num_workers=0, pin_memory=True)
loader_iter = iter(loader)
n_step = [0]
def loader_step():
    global loader_iter
    try:
        b = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        b = next(loader_iter)
    n_step[0] += 1
    b = {k: v.to(DEV, non_blocking=True) for k, v in b.items()}
    out = fwd(b["input_ids"], b["attention_mask"], b["labels"])
    (out.loss / ACC).backward()
    if n_step[0] % ACC == 0:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad(set_to_none=False)
        _ = float(out.loss.detach())
    return out.loss.detach()

bench(loader_step, "S5 real DataLoader")

# S6: invoke train.main() with exact CLI args IN THE SAME PROCESS
from moss_tts_lite import train as train_mod
argv6 = ["--model-dir", MODEL_DIR, "--train-jsonl", JSONL,
         "--output-dir", "/tmp/s6_out", "--mode", "qlora",
         "--per-device-batch-size", "8", "--max-batch-tokens", "900",
         "--gradient-accumulation-steps", "4", "--max-steps", "40",
         "--gradient-checkpointing", "--fused-optimizer",
         "--num-workers", "0", "--num-epochs", "1", "--logging-steps", "10"]
print("=== S6 train.main() same-process ===", flush=True)
train_mod.main(argv6)
