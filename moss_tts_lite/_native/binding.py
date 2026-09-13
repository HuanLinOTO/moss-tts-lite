"""ctypes binding for the native W4 GEMV kernels (no libtorch linkage).

The .so depends only on `libcudart`; every tensor crosses the boundary as a
`data_ptr()` integer plus plain ints.  This makes the prebuilt artefact
independent of the local PyTorch and Python versions -- the only contract is
CUDA's ABI (a CUDA context already current on the calling thread) and the
layout described in `layout.py`.

Loading never raises: `available()` reports whether the kernels can be used,
and `load()` returns `None` when they cannot (missing .so, wrong SM, a CUDA
error at bind time), which is what `fast_native`'s silent fallback keys off.
"""
from __future__ import annotations

import ctypes
import glob
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SO_NAME = "libmoss_native.so"

#: arch -> (major, minor).  A .so built for an older arch runs unchanged on a
#: newer one of the same family via PTX JIT only if it was built with PTX; we
#: build SASS for one arch, so the check is exact for the SASS it carries.  The
#: loader accepts any device whose capability is >= the built target and still
#: in the same major, which is the case CUDA guarantees binary compatibility.
_LIB = None
_LIB_ERR: str | None = None
_LIB_PATH: str | None = None


def candidate_paths() -> list[str]:
    """Search order: explicit env override, the package dir, then build dirs."""
    out = []
    env = os.environ.get("MOSS_NATIVE_SO")
    if env:
        out.append(env)
    out.append(os.path.join(HERE, SO_NAME))
    out += sorted(glob.glob(os.path.join(HERE, "..", "..", "**", SO_NAME),
                            recursive=True))
    return out


def _optional(lib, name, restype, argtypes):
    """Bind an entry point that may be absent in an older .so."""
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _bind(path: str):
    lib = ctypes.CDLL(path)
    c_void_p = ctypes.c_void_p
    # NOTE: keep these lists in lockstep with the `MOSS_EXPORT` signatures at
    # the bottom of gemv_int4.cu.  A mismatch does not raise at bind time --
    # ctypes silently truncates trailing pointer arguments to 32 bits, which
    # surfaces much later as a device-side illegal address.  `selftest()` checks
    # the arity of every entry point before the kernels are used.
    lib.moss_gemv_int4.restype = ctypes.c_int
    lib.moss_gemv_int4.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, c_void_p]
    lib.moss_gemv_int4_bf16.restype = ctypes.c_int
    lib.moss_gemv_int4_bf16.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, c_void_p]
    lib.moss_gemv_bf16_launch.restype = ctypes.c_int
    lib.moss_gemv_bf16_launch.argtypes = [c_void_p, c_void_p, c_void_p,
                                          ctypes.c_int, ctypes.c_int, c_void_p]
    lib.moss_device_info.restype = ctypes.c_int
    lib.moss_device_info.argtypes = [ctypes.POINTER(ctypes.c_int)] * 5
    lib.moss_version.restype = ctypes.c_char_p
    lib.moss_version.argtypes = []
    _optional(lib, "moss_bf16_rows_to_f32", ctypes.c_int,
              [c_void_p, c_void_p, ctypes.c_int, ctypes.c_int, c_void_p])
    return lib


def load(force: bool = False):
    """Return the loaded library or None; never raises."""
    global _LIB, _LIB_ERR, _LIB_PATH
    if _LIB is not None and not force:
        return _LIB
    last = None
    for p in candidate_paths():
        if not p or not os.path.isfile(p):
            continue
        try:
            lib = _bind(p)
        except OSError as e:
            last = f"{p}: {e}"
            continue
        _LIB, _LIB_PATH, _LIB_ERR = lib, p, None
        return lib
    _LIB_ERR = last or f"{SO_NAME} not found (looked in {candidate_paths()})"
    return None


def loaded_path() -> str | None:
    load()
    return _LIB_PATH


def load_error() -> str | None:
    load()
    return _LIB_ERR


def device_info():
    """(sm_count, smem_per_block_optin, smem_per_sm, cc_major, cc_minor) or None."""
    lib = load()
    if lib is None:
        return None
    ints = [ctypes.c_int(0) for _ in range(5)]
    rc = lib.moss_device_info(*[ctypes.byref(i) for i in ints])
    if rc != 0:
        return None
    return tuple(i.value for i in ints)


def check_sm(min_major: int = 8) -> bool:
    """True when this device can run the kernels."""
    info = device_info()
    if info is None:
        return False
    major, minor = info[3], info[4]
    return (major, minor) >= (min_major, 0)


def torch_device_index(device=None) -> int:
    """CUDA device index for a torch device (or the current one)."""
    if device is None:
        try:
            import torch
            return torch.cuda.current_device()
        except Exception:
            return 0
    if isinstance(device, int):
        return device
    idx = getattr(device, "index", None)
    if idx is not None:
        return int(idx)
    return 0


def _current_stream_handle(device=None) -> int:
    """Raw cudaStream_t handle of the current torch stream (0 == legacy).

    Using torch's current stream is what keeps the kernels ordered with respect
    to everything else without a synchronize.
    """
    import torch
    s = torch.cuda.current_stream(device if device is not None else None)
    return int(s.cuda_stream)


def _ptr(t):
    if t is None:
        return 0
    return int(t.data_ptr())


def _check(rc: int, what: str) -> None:
    if rc == 0:
        return
    if rc > 0:
        raise RuntimeError(f"{what}: CUDA error {rc}")
    raise RuntimeError(f"{what}: bad arguments (rc={rc})")


def gemv_int4_bf16(payload, meta, x_bf16, out_bf16, N: int, K: int, g: int,
                   ksplit: int = 1, xglobal: bool = False, threads: int = 256,
                   unroll: int = 1, device=None) -> None:
    """As `gemv_int4` but converts bf16 x in the kernel's own prologue.

    Fusing the conversion matters: a separate `xbuf.copy_(x)` is a dependent
    kernel launch, and at 252 int4 linears per step it cost 0.485 ms/step
    (measured, `.tmp/reports/kern-1.md` §2).
    """
    lib = load()
    if lib is None:
        raise RuntimeError(f"native kernels unavailable: {load_error()}")
    dev = device if device is not None else torch_device_index()
    rc = lib.moss_gemv_int4_bf16(_ptr(payload), _ptr(meta), _ptr(x_bf16),
                                 _ptr(out_bf16), int(N), int(K), int(g),
                                 int(ksplit), 1 if xglobal else 0, int(threads),
                                 int(unroll),
                                 ctypes.c_void_p(_current_stream_handle(dev)))
    _check(rc, "moss_gemv_int4_bf16")


def gemv_int4(payload, meta, x_f32, out_bf16, N: int, K: int, g: int,
              ksplit: int = 1, xglobal: bool = False, threads: int = 256,
              unroll: int = 1, device=None) -> None:
    """int4 GEMV, one kernel: out[n] = sum_k dequant(w[n,k]) * x[k]  (M == 1).

    Args:
        payload: native payload bytes (`layout.repack_int4`).
        meta: native meta bytes (`layout.repack_meta`).
        x_f32: fp32 [K] activation vector.
        out_bf16: bf16 [N] output.
        ksplit: k-slices per block (1/2/4/8); higher = more warps in flight,
            which is what the narrow shapes need.
        xglobal: read x from global instead of staging it in shared memory.
            Cheaper for short kernels, where the staging load plus
            `__syncthreads` is on the critical path.
        threads: threads per block (256 or 512).
        unroll: groups issued per loop iteration; more keeps more independent
            tile loads in flight.
    """
    lib = load()
    if lib is None:
        raise RuntimeError(f"native kernels unavailable: {load_error()}")
    dev = device if device is not None else torch_device_index()
    rc = lib.moss_gemv_int4(_ptr(payload), _ptr(meta), _ptr(x_f32),
                            _ptr(out_bf16), int(N), int(K), int(g),
                            int(ksplit), 1 if xglobal else 0, int(threads),
                            int(unroll),
                            ctypes.c_void_p(_current_stream_handle(dev)))
    _check(rc, "moss_gemv_int4")


def gemv_bf16(w, x_f32, out_bf16, N: int, K: int, device=None) -> None:
    lib = load()
    if lib is None:
        raise RuntimeError(f"native kernels unavailable: {load_error()}")
    dev = device if device is not None else torch_device_index()
    rc = lib.moss_gemv_bf16_launch(_ptr(w), _ptr(x_f32), _ptr(out_bf16),
                                   int(N), int(K),
                                   ctypes.c_void_p(_current_stream_handle(dev)))
    _check(rc, "moss_gemv_bf16_launch")


def bf16_to_f32(src, dst_f32, n: int, device=None) -> None:
    """Single-row bf16 -> fp32 conversion (thin wrapper over the batch form)."""
    bf16_rows_to_f32(src, dst_f32, 1, n, device=device)


def bf16_rows_to_f32(src, dst_f32, rows: int, K: int, device=None) -> None:
    """Convert `rows` contiguous [K] bf16 rows into an fp32 scratch in one launch."""
    lib = load()
    if lib is None:
        raise RuntimeError(f"native kernels unavailable: {load_error()}")
    dev = device if device is not None else torch_device_index()
    rc = lib.moss_bf16_rows_to_f32(_ptr(src), _ptr(dst_f32), int(rows), int(K),
                                   ctypes.c_void_p(_current_stream_handle(dev)))
    _check(rc, "moss_bf16_rows_to_f32")
