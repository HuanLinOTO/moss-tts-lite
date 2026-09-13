# `moss_tts_lite/_native/` — **experimental, not enabled**

Hand-written CUDA int4 GEMV kernels for the decode step, plus the packing layout
they need. **Nothing in the package imports this module.** The shipped decode
path is `tinygemm` (`torch._weight_int4pack_mm`) and stays that way; there is no
automatic detection, no CLI switch, and no change to any default behaviour.

This directory is kept in the repository as a knowledge asset, not as a
supported feature. See "Why it is archived" below before enabling it.

---

## What is here

| file | role |
| --- | --- |
| `gemv_int4.cu` | the kernels: int4 GEMV (bf16 or fp32 x), bf16 GEMV, bf16→fp32 |
| `layout.py` | `repack_int4` / `repack_meta` — a pure **permutation** of the shipped bytes |
| `binding.py` | ctypes ABI; `available()` reports load failure instead of raising |
| `build.py` | `nvcc -arch=sm_XX` → `libmoss_native.so` |
| `__init__.py` | `NativeW4`, the tuned per-shape config table, the M=1 contract |
| `libmoss_native.so` | prebuilt sm_86 (A10G) binary, committed for reference |

Tests live in `tests/test_native_kernel.py` (60 tests) and run **without** this
module being enabled — they are the reason the directory is worth keeping.

## Measured result

On the n2/g32 tier, interleaved A/B, six paired rounds, alternating order, each
arm in its own process, 150 steps/round, CUDA-event timed:

| | ms/step | steps/s |
| --- | --- | --- |
| tinygemm | 10.214 | 97.5 |
| native | 9.880 | 100.8 |
| | | **+3.37% ± 0.02%, 6/6 rounds** |

Fidelity on the shipped checkpoint (18 linears across layers 0/17/35, real
activations from the model's own prompt embedding): 100% exact argmax agreement,
all within the reference's top-5, worst relative error 7.8e-3, correlation
0.999996. Against a float64 reference on real weights the kernel's mean absolute
error is **0.82x** tinygemm's.

## How to enable it (if you decide to)

The module is deliberately not wired in. To use it you must do all of:

1. **Build for your SM.** The committed `.so` is sm_86 only.
   ```bash
   python3 -m moss_tts_lite._native.build --arch 86    # or 80, 89, 90
   ```
2. **Check availability** — `available()` is False for a missing `.so` or a
   pre-Ampere device, and `unavailable_reason()` says which.
   ```python
   from moss_tts_lite._native import available, unavailable_reason
   ```
3. **Swap the linears yourself.** `NativeW4.from_qsz(packed, qsz, group_size)`
   builds one linear; `NativeLinear(w4)` is a `nn.Linear`-shaped wrapper. The
   integration point is `GptqMossTTS._linear` / `FastNativeTTS._lin_fused`, and
   the reference wiring is in the harness scripts under `.tmp/kern_agent/live.py`
   (not committed — see "Reproducing" below).
4. **Decide about VRAM.** The repack duplicates the weights. Pass
   `release_source=True` to `from_qsz` to free the aten tensors in place once
   the native layout exists — that brings the net cost to zero (measured: 3.955
   GiB reclaimed across the n2 state's 216 int4 linears, all kernels still
   producing correct output afterwards). **Off by default**, because the caller
   usually still needs the aten layout for the tinygemm fallback and for the
   `_weight_int4pack_mm` comparison tests.

`NativeW4` is **M=1 only** (`lane == output row`). Prefill and any batched call
must stay on `_weight_int4pack_mm`; `gemv` raises rather than silently
computing a wrong answer.

## Why it is archived (the ROI decision)

**+3.4% does not justify a precompiled binary distribution matrix.** Enabling
this in the shipped path means committing to building, testing and publishing a
CUDA binary for every supported SM × CUDA-toolkit combination, forever, in
exchange for a low-single-digit speedup on one tier. The user's call, recorded
verbatim in `docs/kern-1.md` §8.

The knowledge, however, has lasting value and is what this directory preserves:

* **The bandwidth-ramp correction.** The "648 GB/s wall" the earlier
  optimisation round targeted is a *512 MB* measurement. A single W4 tensor in
  this model is 9–63 MB, and at those sizes the achievable read rate is
  119–592 GB/s — a ramp, not a cliff. Every per-shape target derived from 648
  GB/s was therefore unreachable by *any* kernel. This is the single most
  reusable finding here and it generalises to any small-burst-bandwidth work.
* **Four silent-wrong-answer bug classes** with regression tests, all of which a
  hand-written replacement can exhibit and `tinygemm` cannot (it has no knobs):
  a dispatch key that omits a template argument, a group straddling a k-slice,
  an opt-in shared-memory attribute skipped at exactly the 48 KB boundary, and a
  shared-memory index that is not the one used to derive the row.
* **A determinism-versus-race discrimination method.** A ~4% flaky test was
  proven to be deterministic arithmetic, not state leakage, by replaying a fixed
  failing input across four probes (repeat, interleaved work, explicit sync,
  fresh object) and showing the result set had exactly one element in each.
  See `docs/kern-1.md` §7.

## Reproducing the numbers

The measurement and tuning scripts live in `.tmp/kern_agent/` (gitignored —
regenerate from `docs/kern-1.md` §7 and the script list below):

| script | what it measures |
| --- | --- |
| `tune.py` | per-shape config search on the 36×-chain metric |
| `chain.py` | 36-layer chain bandwidth, native vs tinygemm |
| `live_ab.py` + `live.py --one` | the interleaved A/B live-step result |
| `fidelity.py` | logit agreement on the real checkpoint |
| `release.py` | the VRAM reclaimed by `release_source=True` |
| `flaky.py`…`flaky8.py` | the flaky-test root cause |

Run the tests (no checkpoint needed):
```bash
python3 -m pytest tests/test_native_kernel.py -q
MOSS_NATIVE_W4STATE=1 python3 -m pytest tests/test_native_kernel.py -q  # + checkpoint fidelity
MOSS_NATIVE_BW=1 MOSS_NATIVE_LIVE=1 python3 -m pytest tests/test_native_kernel.py -q  # all gates
```
