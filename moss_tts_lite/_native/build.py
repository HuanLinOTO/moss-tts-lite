#!/usr/bin/env python3
"""Build moss_tts_lite/_native/gemv_int4.cu into a .so next to it.

Usage:
    python3 -m moss_tts_lite._native.build            # auto-detect arch
    python3 -m moss_tts_lite._native.build --arch 86  # force sm_86

Only nvcc + CUDA runtime are required; no torch headers are used, so the
resulting .so is independent of the local PyTorch/Python version.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gemv_int4.cu")
SO_NAME = "libmoss_native.so"


def _detect_arch(env_sm: str | None) -> str:
    if env_sm:
        return env_sm
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return f"{cap[0]}{cap[1]}"
    except Exception:
        pass
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True,
                                 text=True, timeout=15).stdout
            archs = sorted(int(x.strip().split("_")[1]) for x in out.split() if "sm_" in x)
            if archs:
                return str(archs[-1])
        except Exception:
            pass
    return "86"


def build(arch: str | None = None, out_dir: str | None = None, verbose: bool = False,
          extra_flags: list[str] | None = None) -> str:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found on PATH")
    sm = _detect_arch(arch or os.environ.get("MOSS_NATIVE_SM"))
    out_dir = out_dir or HERE
    os.makedirs(out_dir, exist_ok=True)
    so = os.path.join(out_dir, SO_NAME)
    tmp = so + ".tmp"
    cmd = [nvcc, "-O3", "-std=c++17", f"-arch=sm_{sm}", "--use_fast_math",
           "-Xptxas", "-O3", "-Xptxas", "--allow-expensive-optimizations=true",
           "-lineinfo", "--shared", "-Xcompiler", "-fPIC",
           "-o", tmp, SRC]
    if extra_flags:
        cmd[1:1] = extra_flags
    if verbose:
        print(" ".join(cmd), file=sys.stderr)
    # nvcc's -Xcompiler only forwards the next flag, so add the rest here
    cmd = cmd[:-0] if False else cmd
    r = subprocess.run(cmd + ["-lcudart"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{r.stdout}\n{r.stderr}")
    if r.stderr.strip() and verbose:
        print(r.stderr, file=sys.stderr)
    os.replace(tmp, so)
    return so


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None, help="sm arch digits, e.g. 86")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    so = build(a.arch, a.out_dir, a.verbose)
    print(so)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
