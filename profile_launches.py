import json, sys, time, random
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/home/lxm/Projects/moss-workspace/moss-tts-lite")
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
for _ in range(10):
    idx = next(it)
    b = dataset.collate_fn([dataset[i] for i in idx])
    batches.append({k: v.to(DEV) for k, v in b.items()})

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

for _ in range(3):
    step()
torch.cuda.synchronize()

N_ACT = int(__import__("os").environ.get("PROF_STEPS", "5"))
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(N_ACT):
        step()
    torch.cuda.synchronize()

ka = prof.key_averages(group_by_input_shape=True)
rows = []
for e in ka:
    cnt = e.count
    cpu = e.self_cpu_time_total
    cuda = getattr(e, "self_cuda_time_total", 0.0)
    key = e.key
    shapes = ""
    try:
        shapes = str(e.input_shapes[:1]) if e.input_shapes else ""
    except Exception:
        pass
    rows.append((cnt, cpu, cuda, key, shapes))

print("==== TOTAL cudaLaunchKernel / step ====")
for e in ka:
    if "cudaLaunchKernel" in e.key:
        print(f"launch/step = {e.count / N_ACT:.0f}")
        print(f"launch CPU time/step = {e.self_cpu_time_total / N_ACT:.1f}ms")
        break

print("\n==== TOP by CPU op count (count/step, op, shape hint) ====")
rows.sort(key=lambda r: -r[0])
for cnt, cpu, cuda, key, shapes in rows[:45]:
    if cnt < 5 * N_ACT:
        break
    print(f"{cnt / N_ACT:9.0f}/s cpu={cpu / N_ACT:7.1f}ms cuda={cuda / N_ACT:8.2f}ms  {key[:70]:70s} {shapes[:48]}")

print("\n==== TOP by self CUDA time ====")
rows.sort(key=lambda r: -r[2])
tot_cuda = sum(r[2] for r in rows)
print(f"total self cuda = {tot_cuda / N_ACT:.1f}ms/step")
for cnt, cpu, cuda, key, shapes in rows[:25]:
    print(f"{cuda / N_ACT:9.2f}ms ({100 * cuda / tot_cuda:4.1f}%) cnt={cnt / N_ACT:6.0f}/s  {key[:70]} {shapes[:40]}")

print("\n==== elementwise attribution: mul / copy_ / add families with shapes ====")
for fam in ["aten::mul", "aten::copy_", "aten::add", "aten::sub", "aten::div", "aten::erf", "aten::tanh", "aten::pow"]:
    sub = [r for r in rows if r[3] == fam]
    tc = sum(r[0] for r in sub); tcu = sum(r[2] for r in sub)
    if not sub:
        continue
    print(f"-- {fam}: {tc / N_ACT:.0f}/s cuda={tcu / N_ACT:.2f}ms")
    for cnt, cpu, cuda, key, shapes in sorted(sub, key=lambda r: -r[0])[:6]:
        print(f"     {cnt / N_ACT:7.0f}/s {shapes[:64]}")
