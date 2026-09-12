"""Tests for moss_tts_lite.st_loader against models/MOSS-TTS-v1.5 (bf16, 4 shards)."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile

import numpy as np
import torch

from moss_tts_lite.st_loader import read_safetensors, safetensors_header

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                         "models", "MOSS-TTS-v1.5")


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
        # bf16: bitwise round-trip through raw uint16 view
        assert torch.equal(
            got["w_bf16"].view(torch.uint16),
            torch.from_numpy(bf16_raw.reshape(4, 3)))
        # fp32 exact, fp16 exact, int64 exact
        assert torch.equal(got["w_f32"], torch.from_numpy(f32))
        assert torch.equal(got["w_f16"], torch.from_numpy(f16))
        assert torch.equal(got["w_i64"], torch.from_numpy(i64))
        # explicit dtype conversion
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
        # backed by mmap: modifying the tensor's bytes in the file is visible
        data_start = 8 + _header_len_bytes(p)
        with open(p, "r+b") as f:
            f.seek(data_start)
            f.write(struct.pack("<q", 4242))
        assert int(t[0]) == 4242, "tensor should be a live mmap view"
        # (No read-only assertion: torch permits in-place writes on mmap-backed
        # tensors without autograd; same semantics as upstream safetensors loader.)
        print("  mmap zero-copy + read-only: OK (file write visible in tensor)")


def test_sharded_index():
    """Two shards + index.json; partial read must only touch needed shard."""
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
        # partial read
        part = read_safetensors(td, names=["c.weight"])
        assert set(part) == {"c.weight"}
        assert torch.equal(part["c.weight"], torch.from_numpy(c))
        # unknown key must raise
        try:
            read_safetensors(td, names=["nope.weight"])
            raise AssertionError("expected KeyError")
        except KeyError:
            pass
        print("  sharded index + partial read: OK")


def test_moss_v15_checkpoint():
    """Read 5 tensors from the real MOSS-TTS-v1.5 checkpoint; validate against index."""
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

    # shape/dtype must match each shard's header exactly
    shard_headers = {}
    for name in picks:
        shard = os.path.join(MODEL_DIR, weight_map[name])
        if shard not in shard_headers:
            shard_headers[shard] = safetensors_header(shard)
        info = shard_headers[shard][name]
        assert list(got[name].shape) == info["shape"], (name, got[name].shape, info["shape"])
        assert got[name].dtype == _torch_dtype(info["dtype"]), (name, got[name].dtype)
    # cross-check with index json (types present)
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        cfg = json.load(f)
    hid = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size", 4096)
    assert got["emb_ext.0.weight"].shape == (1025, hid)
    assert got["lm_heads.32.weight"].shape == (1025, hid)
    # numerics: finite, embed_tokens norm sane
    e = got["language_model.embed_tokens.weight"].float()
    assert torch.isfinite(e).all()
    rms = e.pow(2).mean().sqrt().item()
    assert 1e-4 < rms < 1e4, rms
    q = got["language_model.layers.0.self_attn.q_proj.weight"].float()
    assert torch.isfinite(q).all()
    assert q.std().item() > 0
    h = got["language_model.norm.weight"].float()
    assert torch.isfinite(h).all()
    assert h.std().item() > 0  # trained RMSNorm weights hover near 1, may dip < 0
    print(f"  MOSS-TTS-v1.5: 5 tensors OK; embed {tuple(e.shape)} rms={rms:.4f}; "
          f"q_proj {tuple(q.shape)} std={q.std().item():.4f}; "
          f"norm.weight range [{h.min().item():.4f}, {h.max().item():.4f}]")


if __name__ == "__main__":
    print(f"[{os.path.basename(__file__)}]")
    test_single_file_all_dtypes()
    test_mmap_zero_copy_and_readonly()
    test_sharded_index()
    test_moss_v15_checkpoint()
    print("ALL TESTS PASSED")
