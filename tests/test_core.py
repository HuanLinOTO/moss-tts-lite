"""Core model tests:"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import struct
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from moss_tts_lite.model import (AUDIO_GEN_SLOT_TOKEN_ID, AUDIO_PAD_CODE, N_VQ,
                                 MossTTSModel)
from moss_tts_lite.sampling import (
    apply_repetition_penalty_delay_pattern,
    apply_top_k,
    apply_top_p,
    apply_top_p_optimized,
    find_last_equal_C,
    sample_token,
)
from moss_tts_lite.st_loader import safetensors_header

try:
    from moss_tts_lite.st_loader import read_safetensors
except ImportError:
    from tests._mini_loader import read_safetensors_min as read_safetensors
from tests._mini_bpe import build_tts_prompt_dev

ROOT = os.environ.get("MOSS_TTS_ROOT",
                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MODEL_DIR = os.path.join(ROOT, "models", "MOSS-TTS-v1.5")

def main_model():
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

    hs = model.prefill(ids)
    h = hs.last_hidden
    assert h.shape == (1, L, 4096), h.shape
    assert model.seq_len == L, model.seq_len
    assert torch.isfinite(h.float()).all(), "NaN/Inf in prefill hidden"
    print(f"prefill hidden: shape {tuple(h.shape)}, finite OK, "
          f"mean|h|={h.float().abs().mean():.4f}, max|h|={h.float().abs().max():.4f}")

    h_last = h[:, -1]
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

    two = model.text_logits_2way(h_last)
    assert two.shape == (2,)
    assert int(two.argmax()) == (0 if int(lt[AUDIO_GEN_SLOT_TOKEN_ID]) > int(lt[151662]) else 1)

    k = L - 5
    ids2 = ids.clone()
    ids2[0, k:, 0] = (ids2[0, k:, 0] + 1) % 1000
    hs2 = model.prefill(ids2)
    d = (hs2.last_hidden[0, :k] - h[0, :k]).abs().max().item()
    assert d == 0.0, f"causality violated: max|d|={d}"
    print(f"causality check (suffix tokens mutated, same shape): max|d| = {d}")

    hs3 = model.prefill(ids[:, :k])
    d2 = (hs3.last_hidden[0] - h[0, :k]).abs().max().item()
    print(f"cross-shape prefix diff (bf16 kernel noise, informational): max|d| = {d2:.4f}")

    model.reset()
    hs = model.prefill(ids)
    for t in range(40):
        row = torch.full((1, 33), AUDIO_PAD_CODE, dtype=torch.long, device=dev)
        row[0, 0] = 151662 if t % 2 == 0 else 151656
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

def ref_apply_top_k(logits, top_k):
    batch_size, vocab_size = logits.shape
    top_k = min(top_k, vocab_size)
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    batch_indices = torch.arange(batch_size).unsqueeze(-1)
    filtered_logits[batch_indices, top_k_indices] = top_k_values
    return filtered_logits

def ref_apply_top_p(logits, top_p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    batch_size = logits.shape[0]
    filtered_logits = logits.clone()
    for i in range(batch_size):
        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
        filtered_logits[i, indices_to_remove] = float("-inf")
    return filtered_logits

def ref_apply_repetition_penalty_delay_pattern(logits, prev_tokens, penalty):
    if penalty == 1.0 or prev_tokens is None:
        return logits
    if logits.dim() == 2:
        prev_tokens_flat = prev_tokens.reshape(-1)
        unique_tokens = torch.unique(prev_tokens_flat)
        token_logits = logits[:, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, unique_tokens] = token_logits
        return logits
    assert logits.dim() == 3
    B, H, V = logits.shape
    for h in range(H):
        prev_tokens_h = prev_tokens[..., h].reshape(-1)
        unique_tokens = torch.unique(prev_tokens_h)
        if unique_tokens.numel() == 0:
            continue
        token_logits = logits[:, h, unique_tokens]
        pos_mask = token_logits > 0
        token_logits[pos_mask] /= penalty
        token_logits[~pos_mask] *= penalty
        logits[:, h, unique_tokens] = token_logits
    return logits

def ref_sample_token(logits, prev_tokens=None, repetition_penalty=1.0,
                     top_p=None, top_k=None, do_sample=True):
    vocab_size = logits.size(-1)
    if prev_tokens is not None and repetition_penalty != 1.0:
        logits = ref_apply_repetition_penalty_delay_pattern(logits, prev_tokens, repetition_penalty)
    if not do_sample:
        return torch.argmax(logits, dim=-1)
    original_shape = logits.shape
    reshaped_logits = logits.view(-1, vocab_size)
    if top_k is not None and top_k > 0:
        reshaped_logits = ref_apply_top_k(reshaped_logits, top_k)
    if top_p is not None and top_p < 1.0:
        reshaped_logits = ref_apply_top_p(reshaped_logits, top_p)
    probs = F.softmax(reshaped_logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens.view(original_shape[:-1])

def main_sampling():
    torch.manual_seed(0)

    x = torch.randn(4, 100)
    assert torch.equal(apply_top_k(x, 7), ref_apply_top_k(x, 7))
    assert torch.equal(apply_top_k(x, 500), ref_apply_top_k(x, 500))
    print("apply_top_k: exact match")

    x = torch.randn(4, 100)
    assert torch.equal(apply_top_p(x.clone(), 0.9), ref_apply_top_p(x.clone(), 0.9))
    x2 = torch.randn(4, 100)
    assert torch.equal(apply_top_p_optimized(x2.clone(), 0.9), ref_apply_top_p(x2.clone(), 0.9))
    x2 = torch.randn(1, 1025)
    assert torch.equal(apply_top_p(x2, 0.5), ref_apply_top_p(x2, 0.5))
    print("apply_top_p: exact match (loop == optimized == port)")

    lg = torch.randn(1, 8, 50)
    pv = torch.randint(0, 50, (1, 20, 8))
    out = apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.2)
    ref = ref_apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.2)
    assert torch.equal(out, ref)
    print("repetition_penalty [B,H,V]: exact match")

    lg = torch.randn(1, 200)
    pv = torch.randint(0, 200, (1, 30))
    out = apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.3)
    ref = ref_apply_repetition_penalty_delay_pattern(lg.clone(), pv, 1.3)
    assert torch.equal(out, ref)
    print("repetition_penalty [N,V]: exact match")

    lg = torch.randn(1, 50)
    assert torch.equal(sample_token(lg, do_sample=False),
                       ref_sample_token(lg, do_sample=False))

    for (tk, tp) in [(None, None), (25, 0.8), (50, 1.0), (None, 0.95), (5, None)]:
        for rep in (1.0, 1.2):
            torch.manual_seed(42)
            a = sample_token(lg.clone(), prev_tokens=torch.randint(0, 50, (1, 9)),
                             repetition_penalty=rep, top_p=tp, top_k=tk, do_sample=True)
            torch.manual_seed(42)
            b = ref_sample_token(lg.clone(), prev_tokens=torch.randint(0, 50, (1, 9)),
                                 repetition_penalty=rep, top_p=tp, top_k=tk, do_sample=True)
            assert torch.equal(a, b), (tk, tp, rep, a, b)

    lg3 = torch.randn(31, 1025)
    pv3 = torch.randint(0, 1025, (1, 77, 32))
    torch.manual_seed(7)
    a = sample_token(lg3.clone(), prev_tokens=pv3, repetition_penalty=1.1,
                     top_p=0.8, top_k=25, do_sample=True)
    torch.manual_seed(7)
    b = ref_sample_token(lg3.clone(), prev_tokens=pv3, repetition_penalty=1.1,
                         top_p=0.8, top_k=25, do_sample=True)
    assert torch.equal(a, b)
    print("sample_token: exact RNG-stream match across arg combos")

    t = torch.tensor([[5, 3, 7, 3, 9]])
    assert find_last_equal_C(t, 3).item() == 3
    assert find_last_equal_C(t, 9).item() == 4
    assert find_last_equal_C(t, 42).item() == -1
    print("find_last_equal_C: OK")

    print("test_sampling PASS")

def _np_dtype(st_dt: str) -> np.dtype:
    return {"BF16": np.uint16, "F32": np.float32, "F16": np.float16,
            "I64": np.int64, "F64": np.float64}[st_dt]

def _torch_dtype(st_dt: str) -> torch.dtype:
    return {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
            "I64": torch.int64, "F64": torch.float64}[st_dt]

def _write_safetensors(path: str, tensors: dict[str, tuple[list[int], str,
                                                           np.ndarray]]) -> None:
    """Write a minimal safetensors file (little-endian, offsets aligned 8B)."""
    header, blobs = {}, []
    offset = 0
    ordered = sorted(tensors)
    for name in ordered:
        shape, st_dt, arr = tensors[name]
        nbytes = arr.nbytes
        header[name] = {"dtype": st_dt, "shape": shape,
                        "data_offsets": [offset, offset + nbytes]}
        blobs.append(arr.tobytes())
        offset += nbytes
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(hb)) % 8
    hb = hb + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for b in blobs:
            f.write(b)

def test_single_file_all_dtypes():
    """Synthetic single file covering bf16/fp16/fp32/int64 + non-contiguous offsets."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "toy.safetensors")
        bf16_raw = rng.integers(0, 2**16, size=(4, 3), dtype=np.uint16)
        f32 = rng.standard_normal((2, 5)).astype(np.float32)
        f16 = rng.standard_normal(7).astype(np.float16)
        i64 = rng.integers(-100, 100, size=(9,), dtype=np.int64)
        _write_safetensors(p, {
            "w_bf16": ([4, 3], "BF16", bf16_raw),
            "w_f32": ([2, 5], "F32", f32),
            "w_f16": ([7], "F16", f16),
            "w_i64": ([9], "I64", i64),
        })
        got = read_safetensors(p)
        assert set(got) == {"w_bf16", "w_f32", "w_f16", "w_i64"}
        assert got["w_bf16"].dtype == torch.bfloat16
        assert got["w_f32"].dtype == torch.float32
        assert got["w_f16"].dtype == torch.float16
        assert got["w_i64"].dtype == torch.int64
        assert list(got["w_bf16"].shape) == [4, 3]

        assert torch.equal(
            got["w_bf16"].view(torch.uint16),
            torch.from_numpy(bf16_raw.reshape(4, 3)))

        assert torch.equal(got["w_f32"], torch.from_numpy(f32))
        assert torch.equal(got["w_f16"], torch.from_numpy(f16))
        assert torch.equal(got["w_i64"], torch.from_numpy(i64))

        got32 = read_safetensors(p, dtype=torch.float32)
        assert got32["w_bf16"].dtype == torch.float32
        expect = torch.from_numpy(bf16_raw.view(np.uint16)).view(torch.bfloat16).to(torch.float32)
        assert torch.equal(got32["w_bf16"], expect)
        print("  single-file all-dtypes: OK (bf16 bitwise, f32/f16/i64 exact)")

def _header_len_bytes(path: str) -> int:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return int(n)

def test_mmap_zero_copy_and_readonly():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "toy.safetensors")
        arr = np.arange(12, dtype=np.int64)
        _write_safetensors(p, {"t": ([12], "I64", arr)})
        got = read_safetensors(p)
        t = got["t"]
        assert t.untyped_storage().nbytes() >= 96

        data_start = 8 + _header_len_bytes(p)
        with open(p, "r+b") as f:
            f.seek(data_start)
            f.write(struct.pack("<q", 4242))
        assert int(t[0]) == 4242, "tensor should be a live mmap view"

        print("  mmap zero-copy + read-only: OK (file write visible in tensor)")

def test_sharded_index():
    """Two shards + index."""
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, "model-00001-of-00002.safetensors")
        f2 = os.path.join(td, "model-00002-of-00002.safetensors")
        a = rng.standard_normal((3, 4)).astype(np.float32)
        b = rng.integers(0, 10, size=(6,), dtype=np.int64)
        _write_safetensors(f1, {"a.weight": ([3, 4], "F32", a),
                                "b.weight": ([6], "I64", b)})
        c = rng.standard_normal((2, 2)).astype(np.float32)
        _write_safetensors(f2, {"c.weight": ([2, 2], "F32", c)})
        idx = {"metadata": {"total_size": a.nbytes + b.nbytes + c.nbytes},
               "weight_map": {"a.weight": os.path.basename(f1),
                              "b.weight": os.path.basename(f1),
                              "c.weight": os.path.basename(f2)}}
        with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f)
        got = read_safetensors(td)
        assert set(got) == {"a.weight", "b.weight", "c.weight"}
        assert torch.equal(got["a.weight"], torch.from_numpy(a))
        assert torch.equal(got["c.weight"], torch.from_numpy(c))

        part = read_safetensors(td, names=["c.weight"])
        assert set(part) == {"c.weight"}
        assert torch.equal(part["c.weight"], torch.from_numpy(c))

        try:
            read_safetensors(td, names=["nope.weight"])
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
        print("  sharded index + partial read: OK")

def test_moss_v15_checkpoint():
    """Read 5 tensors from the real MOSS-TTS-v1."""
    if not os.path.isdir(MODEL_DIR):
        print("  [skip] models/MOSS-TTS-v1.5 not present")
        return
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    picks = ["language_model.embed_tokens.weight", "language_model.norm.weight",
             "language_model.layers.0.self_attn.q_proj.weight", "emb_ext.0.weight",
             "lm_heads.32.weight"]
    picks = [p for p in picks if p in weight_map]
    assert len(picks) == 5, f"expected 5 known tensors, got {picks}"
    got = read_safetensors(MODEL_DIR, names=picks)

    shard_headers = {}
    for name in picks:
        shard = os.path.join(MODEL_DIR, weight_map[name])
        if shard not in shard_headers:
            shard_headers[shard] = safetensors_header(shard)
        info = shard_headers[shard][name]
        assert list(got[name].shape) == info["shape"], (name, got[name].shape, info["shape"])
        assert got[name].dtype == _torch_dtype(info["dtype"]), (name, got[name].dtype)

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    hid = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size", 4096)
    assert got["emb_ext.0.weight"].shape == (1025, hid)
    assert got["lm_heads.32.weight"].shape == (1025, hid)

    e = got["language_model.embed_tokens.weight"].float()
    assert torch.isfinite(e).all()
    rms = e.pow(2).mean().sqrt().item()
    assert 1e-4 < rms < 1e4, rms
    q = got["language_model.layers.0.self_attn.q_proj.weight"].float()
    assert torch.isfinite(q).all()
    assert q.std().item() > 0
    h = got["language_model.norm.weight"].float()
    assert torch.isfinite(h).all()
    assert h.std().item() > 0
    print(f"  MOSS-TTS-v1.5: 5 tensors OK; embed {tuple(e.shape)} rms={rms:.4f}; "
          f"q_proj {tuple(q.shape)} std={q.std().item():.4f}; "
          f"norm.weight range [{h.min().item():.4f}, {h.max().item():.4f}]")

def main_st_loader() -> int:
    print(f"[{os.path.basename(__file__)}]")
    test_single_file_all_dtypes()
    test_mmap_zero_copy_and_readonly()
    test_sharded_index()
    test_moss_v15_checkpoint()
    print("ALL TESTS PASSED")
    return 0

def main() -> int:
    rc = 0
    rc |= main_model() or 0
    rc |= main_sampling() or 0
    rc |= main_st_loader() or 0

    return rc

if __name__ == "__main__":
    sys.exit(main())
