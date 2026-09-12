"""TEMPORARY minimal safetensors reader for tts-agent self-tests.

Will be replaced by `moss_tts_lite.st_loader.read_safetensors` once delivered;
tests prefer the real loader and fall back to this one.  Not imported by any
moss_tts_lite runtime module.
"""

import json
import mmap
import struct
import warnings
from pathlib import Path

import torch

_DT = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}
_KEEPALIVE = []  # keep mmaps alive for the lifetime of the tensors


def read_safetensors_min(path, dtype=None):
    """Read one .safetensors file or a sharded dir (with index.json) into
    {name: torch.Tensor} on CPU. dtype=None keeps the on-disk dtype."""
    p = Path(path)
    if p.is_dir():
        idx = json.loads((p / "model.safetensors.index.json").read_text())
        shards = sorted(set(idx["weight_map"].values()))
        out = {}
        for s in shards:
            out.update(_read_file(p / s, dtype))
        return out
    return _read_file(p, dtype)


def _read_file(path, dtype):
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    _KEEPALIVE.append((fh, mm))
    (hdr_len,) = struct.unpack("<Q", mm[:8])
    header = json.loads(mm[8:8 + hdr_len])
    base = 8 + hdr_len
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dt = _DT[meta["dtype"]]
            start, _end = meta["data_offsets"]
            count = 1
            for s in meta["shape"]:
                count *= s
            t = torch.frombuffer(mm, dtype=dt, count=count,
                                 offset=base + start).reshape(tuple(meta["shape"]))
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            out[name] = t
    return out
