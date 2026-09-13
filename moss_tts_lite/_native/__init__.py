"""Native (ctypes) W4 GEMV kernels: weight preparation and dispatch.

This package is the bridge between the shipped GPTQ state and the pure-CUDA
kernels in `gemv_int4.cu`:

    from moss_tts_lite._native import NativeW4
    w4 = NativeW4.from_qsz(packed, qsz, group_size)   # one linear
    w4.gemv(x_bf16)                                   # bf16 [K] -> bf16 [N]
    NativeW4.available()                              # False -> caller falls back

Nothing here imports torch at module scope beyond what the caller already has,
and a failure to load the .so (missing file, wrong SM, no CUDA) is reported
through `available()` rather than raised, which is what `fast_native`'s silent
fallback to the tinygemm path keys off.
"""
from __future__ import annotations

import torch

from . import binding as _b
from .layout import repack_int4, repack_meta

#: per-shape (ksplit, xglobal) chosen by `.tmp/kern_agent/chain.py` on the A10G;
#: keyed by (N, K, g) and consulted only for exact matches, with a safe default.
_TUNED: dict[tuple[int, int, int], tuple[int, bool, int, int]] = {
    # (ksplit, xglobal, threads, unroll), keyed by (N, K, group_size).
    # Measured on the A10G by `.tmp/kern_agent/tune.py` on the metric the
    # step actually sees -- 36 back-to-back launches of the whole layer --
    # because single-shot tuning is dominated by the drain ramp and picks
    # configs that lose at k=36.  Note the shipped n2 state is grouped at
    # **32**, not 64 (`common.py`'s W1P), so the g=32 rows are the ones the
    # live step uses.  Shapes not listed fall back to `_default_ksplit`
    # with the widely-good (256, 1, xglobal=False).
    (4096,4096,32): (16, True, 1024, 1),
    (4096,12288,32): (16, True, 1024, 1),
    (5120,4096,32): (16, True, 512, 1),
    (24576,4096,32): (16, True, 512, 1),
}

#: bytes of zero padding appended to every payload so the kernel's 512-byte
#: tile reads stay in bounds when N is not a multiple of 32.
_PAD = 512


def release_source_tensors(*tensors) -> int:
    """Free the storage of the shipped `packed`/`qsz` tensors in place.

    The repack into the native layout is a pure permutation, so after it
    succeeds the aten-format tensors are redundant: they hold the same nibbles
    and the same bf16 scale/zero pairs, just ordered differently.  Keeping both
    costs the *whole* int4 payload plus metadata -- measured at 1.58 GiB for the
    n2 state (the model weights themselves are 8.24 GiB).

    Deliberately not a `torch.cuda.empty_cache()`: the point is to drop the
    tensors' own storage, not the caching allocator's arena, and the freed
    blocks go straight back to the allocator for reuse either way.

    In-place rather than `del`, because the caller's `qlayers` dict still holds
    references -- the same reason `fast_native`'s fused arm *pops* the per-name
    q/k entries out of the base instead of simply going out of scope.

    Returns the number of tensors actually released.
    """
    freed = 0
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        # Assign a *new* empty tensor rather than `resize_(0)`: resize keeps the
        # original storage alive (measured: a 64 MiB tensor still held 64 MiB
        # after `resize_(0)`, and 0 after this), so it would not actually free
        # anything.  The tensor object stays valid and empty, so any stale alias
        # in the caller fails loudly on the next use instead of reading freed
        # memory.
        try:
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
            freed += 1
        except Exception:                                    # noqa: BLE001
            pass
    return freed


def available() -> bool:
    """True when the native kernels can run on the current device."""
    if not torch.cuda.is_available():
        return False
    if _b.load() is None:
        return False
    return _b.check_sm(8)


def unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "cuda unavailable"
    if _b.load() is None:
        return _b.load_error()
    if not _b.check_sm(8):
        info = _b.device_info()
        return f"device compute capability {info[3]}.{info[4]} < 8.0"
    return None


class NativeW4:
    """One int4 linear in the native layout, ready for `gemv`.

    Attributes:
        N, K, group_size: the logical GEMM shape (out_features, in_features).
        payload: uint8 [N*K/2 (+pad)] in the 512-byte-tile native layout.
        meta: bf16 [K/g * Np * 2] in the 32-row-block native layout.
    """

    __slots__ = ("N", "K", "group_size", "payload", "meta", "ksplit", "xglobal",
                 "threads", "unroll", "_xbuf", "_out", "_device")

    def __init__(self, payload, meta, N: int, K: int, group_size: int,
                 ksplit: int | None = None, xglobal: bool | None = None,
                 threads: int | None = None, unroll: int | None = None,
                 device=None):
        self.N, self.K, self.group_size = int(N), int(K), int(group_size)
        self.payload = payload
        self.meta = meta
        tuned = _TUNED.get((self.N, self.K, self.group_size))
        self.ksplit = int(ksplit if ksplit is not None
                          else (tuned[0] if tuned else self._default_ksplit()))
        self.xglobal = bool(xglobal if xglobal is not None
                            else (tuned[1] if tuned else False))
        self.threads = int(threads if threads is not None
                           else (tuned[2] if tuned else 256))
        self.unroll = int(unroll if unroll is not None
                          else (tuned[3] if tuned else 1))
        self._device = device
        self._xbuf = None
        self._out = None

    # ------------------------------------------------------------- tuning
    def _default_ksplit(self) -> int:
        """Largest ksplit whose slice still holds whole groups.

        A group is `g/32` consecutive 32-k tiles and may not straddle a slice
        (the epilogue applies the group's (scale, zero) to the group's own sum
        of x), and slicing must divide the tile count evenly.
        """
        nj = self.K // 32
        cpg = self.group_size // 32
        for ks in (8, 4, 2, 1):
            if nj % ks == 0 and (nj // ks) % cpg == 0:
                return ks
        return 1

    # ------------------------------------------------------------ builders
    @classmethod
    def from_qsz(cls, packed: torch.Tensor, qsz: torch.Tensor, group_size: int,
                 device=None, release_source: bool = False,
                 **kw) -> "NativeW4":
        """Build from the shipped `(packed, qsz)` pair (a pure permutation).

        Args:
            packed: int32 payload as produced by
                `aten._convert_weight_to_int4pack`.
            qsz: bf16 [K/g, N, 2] scale/zero pairs, exactly as shipped.
            group_size: the group size the state was quantized with.
            release_source: after a successful repack, free the caller's aten
                `packed`/`qsz` in place (see `release_source_tensors`).  The
                repack is a pure permutation, so the two layouts hold the same
                information and keeping both costs the full 1.58 GiB of int4
                payload plus metadata for the n2 state.  Off by default: the
                caller usually still needs the aten layout (the tinygemm
                fallback, `_weight_int4pack_mm` tests) and freeing a tensor it
                still holds would be a use-after-free, not a leak fix.
        """
        N = int(qsz.shape[1])
        K = int(qsz.shape[0]) * int(group_size)
        # int32 payloads may be non-contiguous views after a `cat`; flattening
        # through `reshape(-1)` on the byte view is the only safe order.
        pay = repack_int4(packed.contiguous(), N, K, out_bytes_pad=_PAD)
        meta = repack_meta(qsz, N, group_size)
        if release_source:
            release_source_tensors(packed, qsz)
        return cls(pay, meta, N, K, group_size, device=device, **kw)

    @classmethod
    def from_state_record(cls, rec: dict, group_size: int, **kw) -> "NativeW4":
        """Build from a `load_gptq_state` entry (`{"packed": ..., "qsz": ...}`)."""
        return cls.from_qsz(rec["packed"], rec["qsz"], group_size, **kw)

    # --------------------------------------------------------------- run
    def _buffers(self, device):
        if self._xbuf is None or self._xbuf.device != device:
            self._xbuf = torch.empty(self.K, dtype=torch.float32, device=device)
            self._out = torch.empty(self.N, dtype=torch.bfloat16, device=device)
        return self._xbuf, self._out

    def gemv(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """`x` bf16 [K] -> bf16 [N]; M == 1 only (the kernels are one row per lane).

        The returned tensor is an internal buffer, valid until the next call --
        the same lifetime contract as `fast.py`'s static graph outputs.
        """
        n = x_bf16.numel()
        if n != self.K:
            raise ValueError(
                f"native int4 GEMV is M=1 only: {self.N}x{self.K} given "
                f"{n} input elements ({tuple(x_bf16.shape)}).  Prefill and any "
                f"batched call must stay on torch._weight_int4pack_mm.")
        device = x_bf16.device
        xbuf, out = self._buffers(device)
        # The kernel wants fp32 x (it accumulates in fp32 FMA, matching
        # `_weight_int4pack_mm`), and the conversion is folded into the GEMV's
        # own prologue.  Doing it as `xbuf.copy_(x)` first would be a *separate
        # dependent kernel launch* per linear -- 252 of them per step, measured
        # at 0.485 ms/step, i.e. the whole live-step deficit.
        _b.gemv_int4_bf16(self.payload, self.meta, x_bf16.reshape(-1), out,
                          self.N, self.K, self.group_size, self.ksplit,
                          self.xglobal, threads=self.threads,
                          unroll=self.unroll, device=device)
        return out

    def gemv_f32(self, x_f32: torch.Tensor) -> torch.Tensor:
        """As `gemv` for an already-widened fp32 input.

        Same kernel, same result as `gemv` on the equivalent bf16 input -- kept
        separate so the two conversion paths can be tested against each other
        rather than one silently shadowing the other.
        """
        n = x_f32.numel()
        if n != self.K:
            raise ValueError(
                f"native int4 GEMV is M=1 only: {self.N}x{self.K} given {n} "
                f"input elements ({tuple(x_f32.shape)}).")
        device = x_f32.device
        _xbuf, out = self._buffers(device)
        _b.gemv_int4(self.payload, self.meta,
                     x_f32.reshape(-1).float().contiguous(), out, self.N,
                     self.K, self.group_size, self.ksplit, self.xglobal,
                     threads=self.threads, unroll=self.unroll, device=device)
        return out

    def gemv_bf16(self, x_bf16: torch.Tensor) -> torch.Tensor:
        """Alias for `gemv` (the input is bf16, the kernel is int4)."""
        return self.gemv(x_bf16)


class NativeLinear(torch.nn.Module):
    """`torch.nn.Linear`-shaped wrapper so `fast_native` can swap it in.

    Deliberately *not* a `Linear` subclass: it holds no parameters (the payload
    is opaque bytes) and its forward is M=1-only, so anything that assumes
    `weight` or batched input should fail loudly rather than silently take a
    slow path.
    """

    def __init__(self, w4: NativeW4):
        super().__init__()
        self.w4 = w4
        self.in_features = w4.K
        self.out_features = w4.N

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        shp = x.shape
        if x.numel() != self.in_features:
            raise ValueError(
                f"native int4 GEMV is M=1 only: got input of shape {tuple(shp)}")
        return self.w4.gemv(x.reshape(-1)).reshape(*shp[:-1], self.out_features)
