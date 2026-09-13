#!/usr/bin/env python3
"""Layout code for the native W4 GEMV kernels.

Two jobs:

1. `byte_nib_index` -- the closed form of `aten._convert_weight_to_int4pack`'s
   layout.  Derived by bit-probing the operator on this stack and verified by
   exact round trip on every shape the model uses (4096x4096, 5120x4096,
   24576x4096, 4096x12288) as well as small tiles.  Kept here because the
   *repack* needs to read the shipped payload correctly; nothing depends on the
   layout beyond this file.

2. `repack_int4` / `repack_meta` -- permutation into the native layout the CUDA
   kernel wants (see `gemv_int4.cu` for the byte-level description).  This is a
   pure permutation of the shipped bytes: no value is re-quantized, so the
   kernel's arithmetic is *identical* to `_weight_int4pack_mm`'s dequant.

Both are torch ops, used at load time only; the kernels themselves never see
torch.
"""
from __future__ import annotations

import torch

#: 512-byte tile = 32 rows x 32 k, the unit the kernel reads with one uint4 per
#: lane.  The shipped layout's own tile is 8 rows x 128 k, which is why a
#: repack is needed for a coalesced M=1 read.
NTILE_ROWS = 32
NTILE_K = 32


# ---------------------------------------------------------------------------
# shipped layout (aten._convert_weight_to_int4pack)
# ---------------------------------------------------------------------------
def _intra_b(kl: torch.Tensor) -> torch.Tensor:
    """Byte offset inside a 128-k half-tile for k-within-128 == `kl`.

    Recovered from the bit probe: bit i of `kl` goes to byte bit p(i) with
    p = (4 -> 0, 0 -> 1, 5 -> 2, 6 -> 3, 1 -> 4, 2 -> 5); bit 3 selects the
    nibble instead of contributing to the byte index.
    """
    return (((kl >> 4) & 1) | (((kl >> 0) & 1) << 1) | (((kl >> 5) & 1) << 2)
            | (((kl >> 6) & 1) << 3) | (((kl >> 1) & 1) << 4) | (((kl >> 2) & 1) << 5))


def byte_nib_index(N: int, K: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] index of the byte holding, and nibble slot of, each logical (n,k).

    `byte = (n//8)*(512*(K//128)) + (k//128)*512 + (n%8)*64 + intra_b(k%128)`,
    `nib = ((k % 128) // 8) % 2`, low nibble first.
    """
    n = torch.arange(N, device=device, dtype=torch.long)[:, None]
    k = torch.arange(K, device=device, dtype=torch.long)[None, :]
    kl = k & 127
    byte = ((n >> 3) * (512 * (K >> 7)) + (k >> 7) * 512 + (n & 7) * 64
            + _intra_b(kl))
    nib = (kl >> 3) & 1
    return byte, nib


def unpack_int4(packed: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """uint8 [N, K] natural-order nibbles out of a shipped `packed` payload."""
    src = packed.reshape(-1).view(torch.uint8)
    byte, nib = byte_nib_index(N, K, device=packed.device)
    return ((src[byte] >> (nib * 4).to(torch.uint8)) & 0xF).to(torch.uint8)


# ---------------------------------------------------------------------------
# native layout (what gemv_int4.cu reads)
# ---------------------------------------------------------------------------
def _native_offsets(N: int, K: int, device=None):
    """Per-(n,k) byte offset in the native payload and its nibble slot."""
    n = torch.arange(N, device=device, dtype=torch.long)[:, None]
    k = torch.arange(K, device=device, dtype=torch.long)[None, :]
    # k -> (j = k//32, m = (k%32)//2, slot = k%2)
    j = k >> 5
    m = (k >> 1) & 15
    slot = k & 1
    byte = (((n >> 5) * (K >> 5) + j) * 512) + (n & 31) * 16 + m
    return byte, slot


def repack_int4(packed: torch.Tensor, N: int, K: int,
                out_bytes_pad: int = 0) -> torch.Tensor:
    """Shipped `packed` -> native payload (uint8, 512-byte tiles, 32-row blocks).

    `out_bytes_pad` appends zero bytes (the kernel may read a padded tail for
    the last partial 32-row block; the pad keeps those reads in bounds).
    """
    dev = packed.device
    q = unpack_int4(packed, N, K)
    npad = (-N) % NTILE_ROWS
    if npad:
        q = torch.cat([q, torch.zeros(npad, K, dtype=q.dtype, device=dev)], 0)
    Np = N + npad
    byte, slot = _native_offsets(Np, K, device=dev)
    flat = torch.zeros(Np * K // 2, dtype=torch.uint8, device=dev)
    # low nibble (even k) and high nibble (odd k) travel together in one byte
    even = q[:, 0::2].to(torch.int32)
    odd = q[:, 1::2].to(torch.int32)
    be, _ = _native_offsets(Np, K, device=dev)
    be = be[:, 0::2]
    flat[be.reshape(-1)] |= (even | (odd << 4)).to(torch.uint8).reshape(-1)
    if out_bytes_pad:
        flat = torch.cat([flat, torch.zeros(out_bytes_pad, dtype=torch.uint8,
                                            device=dev)])
    return flat.contiguous()


def repack_meta(qsz: torch.Tensor, N: int, g: int) -> torch.Tensor:
    """qsz bf16 [K/g, N, 2] -> native meta (bf16 pairs by 32-row block).

    Layout: `((rb * GC + gi) * 128 + (n%32)*4 + 2*t)` bytes, t = 0 scale,
    t = 1 zero.  A pure reordering of the shipped qsz: bf16 values unchanged.
    """
    K = qsz.shape[0] * g
    GC = K // g
    dev = qsz.device
    npad = (-N) % NTILE_ROWS
    m = qsz.permute(1, 0, 2).contiguous()            # [N, GC, 2]
    if npad:
        m = torch.cat([m, torch.zeros(npad, GC, 2, dtype=m.dtype, device=dev)], 0)
    Np = N + npad
    # [rb, N%32, gi, 2] -> [rb, gi, N%32, 2]
    m = m.view(Np // NTILE_ROWS, NTILE_ROWS, GC, 2) \
         .permute(0, 2, 1, 3).contiguous()
    return m.view(-1).contiguous()
