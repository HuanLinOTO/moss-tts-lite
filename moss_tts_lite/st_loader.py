"""Minimal safetensors reader with mmap-backed zero-copy tensors."""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import torch

__all__ = ["read_safetensors", "safetensors_header"]

_DT: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_NP_RAW: dict[str, np.dtype] = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

def safetensors_header(path: str) -> dict:
    """Parse the 8-byte length prefix + JSON header of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header

def _read_shard(path: str, want: set[str] | None, dtype: torch.dtype | None,
                out: dict[str, torch.Tensor]) -> None:
    header = safetensors_header(path)
    data_start = 8 + _header_len(path)

    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if want is not None and name not in want:
            continue
        dt_str = info["dtype"]
        if dt_str not in _DT:
            raise NotImplementedError(
                f"Unsupported safetensors dtype {dt_str!r} for tensor {name!r}")
        shape = info["shape"]
        lo, hi = info["data_offsets"]
        numel = 1
        for s in shape:
            numel *= s
        np_dt = np.dtype(_NP_RAW[dt_str])
        if np_dt.itemsize > 1 and (hi - lo) != numel * np_dt.itemsize:

            raise ValueError(
                f"Tensor {name!r}: data size {hi - lo} bytes does not match "
                f"{numel} elements of dtype {dt_str}")
        arr = np.frombuffer(mm, dtype=np_dt, count=numel, offset=data_start + lo)
        t = torch.from_numpy(arr)
        t.requires_grad = False
        if dt_str == "BF16":
            t = t.view(torch.bfloat16)
        t = t.reshape(tuple(shape))
        if dtype is not None:
            t = t.to(dtype)
        out[name] = t

def _header_len(path: str) -> int:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
    return int(n)

def read_safetensors(path: str, dtype: torch.dtype | None = None,
                     names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Read safetensors file or sharded directory into a dict of tensors."""
    want = set(names) if names is not None else None
    out: dict[str, torch.Tensor] = {}

    if os.path.isdir(path):
        index_path = os.path.join(path, "model.safetensors.index.json")
        single_path = os.path.join(path, "model.safetensors")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            if want is not None:
                missing = want - set(weight_map)
                if missing:
                    raise KeyError(f"Tensors not in index: {sorted(missing)[:5]}...")
                shards = sorted({weight_map[n] for n in want})
            else:
                shards = sorted(set(weight_map.values()))
            for shard in shards:
                _read_shard(os.path.join(path, shard), want, dtype, out)
        elif os.path.exists(single_path):
            _read_shard(single_path, want, dtype, out)
        else:
            raise FileNotFoundError(
                f"No model.safetensors.index.json or model.safetensors in {path}")
    else:
        _read_shard(path, want, dtype, out)
    return out
