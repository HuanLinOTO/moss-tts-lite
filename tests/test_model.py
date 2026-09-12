"""MossTTSModel self-test on real weights: prefill + 40 decode steps.

Checks: no NaN, correct shapes, KV cache length bookkeeping, text-head vs
2-way-head agreement, causal sanity (hidden for token i independent of later
tokens).  bf16 ~17.5GB on GPU, run under the GPU lock:
  flock .tmp/gpu.lock python3 -m moss_tts_lite.tests.test_model
"""

import sys

import torch

from moss_tts_lite.model import AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_PAD_CODE, N_VQ, MossTTSModel
try:  # prefer the real loader (tok-delivered); fall back to the temp mini one
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors
from tests._mini_bpe import build_tts_prompt_dev

MODEL_DIR = "/root/MOSS-TTS/models/MOSS-TTS-v1.5"


def main():
    dev = torch.device("cuda")
    weights = read_safetensors(MODEL_DIR)
    assert len(weights) == 463, len(weights)

    model = MossTTSModel(weights, device=dev, dtype=torch.bfloat16, max_seq_len=8192)
    assert model.n_layers == 36 and model.hidden_size == 4096
    assert model.n_heads == 32 and model.n_kv_heads == 8 and model.head_dim == 128
    assert model.text_vocab == 155648
    del weights
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated() / 2**30
    print(f"weights on device: {mem:.2f} GiB")

    prompt = build_tts_prompt_dev("Hello world, this is a smoke test.")
    ids = prompt["input_ids"].to(dev)
    L = ids.shape[1]
    print("prompt L =", L)

    # ---- prefill ----
    hs = model.prefill(ids)
    h = hs.last_hidden
    assert h.shape == (1, L, 4096), h.shape
    assert model.seq_len == L, model.seq_len
    assert torch.isfinite(h.float()).all(), "NaN/Inf in prefill hidden"
    print(f"prefill hidden: shape {tuple(h.shape)}, finite OK, "
          f"mean|h|={h.float().abs().mean():.4f}, max|h|={h.float().abs().max():.4f}")

    # ---- heads on last position ----
    h_last = h[:, -1]  # [1, 4096]
    lt = model.text_logits(h_last)
    assert lt.shape == (155648,), lt.shape
    assert torch.isfinite(lt.float()).all(), "NaN/Inf in text logits"
    la = model.audio_logits(h_last)
    assert la.shape == (32, 1025), la.shape
    assert torch.isfinite(la[..., :1024].float()).all(), "NaN/Inf in audio logits"
    assert torch.isinf(la[:, AUDIO_PAD_CODE]).all(), "audio pad column must be -inf"
    top_text = int(lt.argmax())
    print(f"text logits: argmax={top_text}, max={lt.max().item():.3f}")
    assert top_text != AUDIO_GEN_SLOT_TOKEN_ID, "unexpected immediate gen_slot"
    la0 = la[0]
    top_audio = int(la0.argmax())
    assert top_audio != AUDIO_PAD_CODE
    print(f"audio ch0 logits: argmax={top_audio}, max={la0.max().item():.3f}")

    # 2-way head vs full head (greedy equivalence precondition)
    two = model.text_logits_2way(h_last)
    assert two.shape == (2,)
    assert int(two.argmax()) == (0 if int(lt[AUDIO_GEN_SLOT_TOKEN_ID]) > int(lt[151662]) else 1)

    # ---- causal sanity (same-shape test): changing tokens >= k must not change
    # hidden[:k] bitwise (same kernels -> no shape-induced rounding drift) ----
    k = L - 5
    ids2 = ids.clone()
    ids2[0, k:, 0] = (ids2[0, k:, 0] + 1) % 1000
    hs2 = model.prefill(ids2)
    d = (hs2.last_hidden[0, :k] - h[0, :k]).abs().max().item()
    assert d == 0.0, f"causality violated: max|d|={d}"
    print(f"causality check (suffix tokens mutated, same shape): max|d| = {d}")
    # informational: cross-shape prefill (L vs L-5) differs only by bf16 kernel-tiling noise
    hs3 = model.prefill(ids[:, :k])
    d2 = (hs3.last_hidden[0] - h[0, :k]).abs().max().item()
    print(f"cross-shape prefix diff (bf16 kernel noise, informational): max|d| = {d2:.4f}")

    # ---- 40 decode steps ----
    model.reset()
    hs = model.prefill(ids)
    for t in range(40):
        row = torch.full((1, 33), AUDIO_PAD_CODE, dtype=torch.long, device=dev)
        row[0, 0] = 151662 if t % 2 == 0 else 151656  # delay_slot / gen_slot
        row[0, 1:] = torch.randint(0, 1024, (32,), device=dev)
        hs = model.step(row)
        assert hs.last_hidden.shape == (1, 1, 4096)
        assert torch.isfinite(hs.last_hidden.float()).all(), f"NaN at step {t}"
        assert model.seq_len == L + t + 1, model.seq_len
        _lt = model.text_logits(hs.last_hidden)
        assert _lt.shape == (155648,) and torch.isfinite(_lt.float()).all()
        _la = model.audio_logits(hs.last_hidden)
        assert _la.shape == (32, 1025)
    print(f"40 steps OK; final seq_len={model.seq_len} (expected {L + 40}); "
          f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("test_model PASS")


if __name__ == "__main__":
    sys.exit(main())
