# kern-1: hand-written int4 GEMV for the n2 decode step

**Status: archived, not enabled. Decision record in §8.**

The kernel works and is measurably faster than `tinygemm` on the live n2 step
(**+3.36% +- 0.02%, 6/6 and 4/4 paired rounds across two independent A/B runs**),
with fidelity verified on the shipped weights (100% exact argmax agreement over
18 real linears, worst relative error 7.8e-3). It is nevertheless **not** wired
into `fast_native` or the CLI: the user's assessment is that a sub-5% gain does
not justify committing to a precompiled CUDA binary distribution matrix. The
directory stays in the repository as a knowledge asset -- see
`moss_tts_lite/_native/README.md` for how to enable it, and §8 for the decision.

> **On the `.tmp/kern_agent/...` references below.** The measurement and tuning
> scripts cited throughout this report live under `.tmp/`, which is gitignored,
> so those paths are not in the repository. Every *number* here was measured in
> the process that printed it and the methodology is described inline; the
> script names are recorded so each measurement can be re-derived. The two
> artifacts worth keeping are committed: `moss_tts_lite/_native/` (kernels +
> `README.md`) and `tests/test_native_kernel.py` (60 tests), which together pin
> every claim in §4 and §7.1.

**The project's framing was wrong in a way worth recording**: the 648 GB/s wall
is a 512 MB measurement, and a single W4 tensor is 9-63 MB. At those sizes the
ramp, not the asymptote, is the limit. That correction (§5) is the most reusable
finding here and it invalidated the per-shape targets every earlier round was
chasing. The ramp is real, both kernels fight it, and a kernel that keeps more
requests in flight beats it by a few percent.

---

## 1. What was built

`moss_tts_lite/_native/` -- pure-CUDA (no libtorch) int4 GEMV plus the packing
layout it needs:

| file | role |
| --- | --- |
| `gemv_int4.cu` | the kernels: int4 GEMV (bf16 or fp32 x), bf16 GEMV, bf16->fp32 |
| `layout.py` | `repack_int4` / `repack_meta`: a pure **permutation** of the shipped bytes |
| `binding.py` | ctypes ABI; `available()` reports load failure instead of raising |
| `build.py` | `nvcc -arch=sm_XX` -> `libmoss_native.so` |
| `__init__.py` | `NativeW4`, the tuned per-shape config table, the M=1 contract |

`tests/test_native_kernel.py`: 55 tests in five layers -- layout algebra, kernel
numerics over every real shape (synthetic and, with `MOSS_NATIVE_W4STATE=1`, the
real checkpoint), fallback, a bandwidth gate (`MOSS_NATIVE_BW=1`), and an opt-in
live-step gate (`MOSS_NATIVE_LIVE=1`).

### Why a repack is needed

The shipped `_convert_weight_to_int4pack` layout stores an **8-row x 128-k**
tile in 512 contiguous bytes. An M=1 GEMV has `lane == output row`, so a warp
reading that layout for 32 rows touches 4 non-adjacent 128-byte segments and
cannot coalesce. The native layout is a bit-scatter of the same bytes into a
**32-row x 32-k** tile:

```
offset(rb, j, n, m) = ((rb * KC + j) * 512) + (n & 31) * 16 + m
byte m holds k = 32j + 2m     in its low nibble
             k = 32j + 2m + 1 in its high nibble
```

so one lane's `uint4` covers a whole 32-k tile and a warp covers the whole
32x32 tile in one fully-coalesced 512-byte transaction. Nothing is
re-quantized: `test_repack_is_a_pure_permutation` decodes the native payload with
the kernel's own index rule and compares against `unpack_int4`, and
`test_repack_meta_is_a_pure_reorder` does the same for the `qsz` pairs.

---

## 2. The result

### Live step (the number that matters)

Interleaved A/B, six paired rounds, alternating order, each arm in its own
process (two 8 GiB instances do not fit on the 24 GiB card), 150 steps/round,
CUDA-event timed (`.tmp/kern_agent/live_ab.py`):

| round | tinygemm ms | native ms | native |
| --- | --- | --- | --- |
| 0 | 10.2156 | 9.8861 | +3.33% |
| 1 | 10.2148 | 9.8811 | +3.38% |
| 2 | 10.2142 | 9.8810 | +3.37% |
| 3 | 10.2149 | 9.8792 | +3.40% |
| 4 | 10.2139 | 9.8812 | +3.37% |
| 5 | 10.2138 | 9.8816 | +3.36% |

**median +3.37%, sd 0.0002, 6/6 wins.** 97.5 -> 100.8 steps/s on the n2/g32
tier. The spread is 2e-4 on the ratio, so this is not a lucky rep.

### Fidelity on the real checkpoint

`.tmp/kern_agent/fidelity.py`, `MOSS_NATIVE_W4STATE=1` pytest gate: every int4
linear of layers 0/17/35, real weights, real activation rows from the model's
own prompt embedding:

```
18/18 exact argmax agreement (100.0%)
18/18 within the reference's top-5
worst relative max error 7.75e-03
worst correlation 0.999996
```

The kernels are *not* bitwise identical -- the dot accumulates in a different
order and the affine `zero` term is algebraically refactored -- so bitwise
equality is the wrong gate; argmax/top-k agreement under the tier's own
tie-robust criteria is the right one. A `randn`-weight test is easier than the
real weights (uniform dynamic range, unit-scaled activations), which is why the
checkpoint gate exists separately.

### Per-shape, 36x chain

Config from `.tmp/kern_agent/tune.py` (tuned on the 36x-chain metric, since
single-shot tuning is dominated by the drain ramp and picks configs that lose at
k=36):

| shape | MB | native GB/s | tinygemm GB/s | ratio |
| --- | --- | --- | --- | --- |
| o (4096x4096) | 9.44 | 560.6 (ks=8,th=512,un=2) | 555 | 1.01 |
| qk_fused (5120x4096) | 11.80 | 613.5 (ks=8,th=512) | 594 | 1.03 |
| down (4096x12288) | 28.31 | 656.3 (ks=16,xg=1,th=1024) | 516 | 1.27 |
| gu_fused (24576x4096) | 56.62 | 640.2 (ks=16,th=512) | 636 | 1.01 |

---

## 3. What actually produced the win

Four things, in order of size. Three of them were *not* kernel optimizations.

1. **6 of 7 int4 linears were silently on tinygemm** (+~1 pp). The wrapper's
   `_lin_fused` looked up `self.native[(li, "__fused_qk")]` but the table was
   built under `self.native[(li, "qk")]`, so it never found the entry, fell back
   to `super()`, and the two biggest linears of each layer were never measured.
   The `native` arm was really "native for o and down, tinygemm for qk and gu".
2. **A separate bf16->fp32 conversion kernel per call** (+2.2 pp). `gemv` did
   `xbuf.copy_(x)` before each of 252 int4 linears -- a dependent launch, 1.9 us
   each, 0.485 ms/step. Widening x inside the GEMV prologue removed it
   (`gemv_int4_bf16`, and the kernel is templated on the x element type).
3. **CPG == 1 group sums were being shuffle-reduced** (+~1 pp). The shipped
   grouping is **g=32**, so a group is exactly one 32-k tile and every lane
   already holds its own element of the group sum. The generic
   load-`G/32`-then-shuffle-reduce path ran per tile; specialising it was worth
   ~5-15% on the g=32 shapes (qk 697 -> 713 GB/s, o 627 -> 666 GB/s).
4. **The kernel work itself** (+~1 pp): k-split (`gridDim.y`) to multiply warps
   for free, streaming payload loads (`__ldcs` -- the payload is read once, so
   evict-first leaves L2 to the meta and x, which *are* reused), and
   `UNROLL`/threads tuning measured on the chain rather than in isolation.

Measured trajectory of the live delta: **-4.70% -> -2.40% -> -1.55% -> +3.34%**.

---

## 4. Bugs found (all have regression tests)

1. **Silent wrong answer from the host dispatch table.** The table was keyed on
   `(ksplit, unroll)` with `threads` selected *inside* each case, so `ksplit=3`
   matched the `ksplit=1` instantiation and returned `rc=0` with a
   wrong-but-plausible vector instead of the -6 `launch_one` raises for a split
   that cuts a group in half. Now keyed on every template argument;
   `test_illegal_knobs_are_rejected` covers five combinations.
2. **Group straddling a k-slice.** With `gridDim.y > 1` the per-group `(scale,
   zero)` was applied to a partial group sum -- silently wrong, not an error,
   whenever `nj/ksplit % (g/32) != 0` (K=4096, g=64 hits it at `ksplit=3`).
   `test_every_legal_ksplit_agrees` pins it.
3. **Opt-in shared memory at exactly the 48 KB default.** `cudaFuncSetAttribute`
   was skipped when the request was `<= 48 KB`, but the total per-block budget
   also covers the static reduce array, so K=12288 launched at `ksplit=1` and
   failed with `rc=1` at `ksplit>1`. The attribute is now always set.
4. **A shared-memory index that was not the one used to derive the row**
   (`rblk = warp` instead of `warp / KSPLIT`) made 4-slice blocks write rows 0..3
   of rows 4..7 -- an out-of-range write at `ksplit>1`.
5. **ctypes argtypes drift.** A stale `argtypes` list silently truncates the
   trailing pointer arguments to 32 bits; the symptom is a device-side illegal
   address far from the cause. Commented as needing to stay in lockstep with the
   `MOSS_EXPORT` signatures.
6. **A nested `group_size_map`.** It is `{layer -> {name -> g}}` and the g=128
   entries are the *bf16* v_proj, which is not in `qlayers` at all. Treating it
   as flat made a fidelity check build int4 kernels with K four times too large.

---

## 5. The bandwidth analysis (what the numbers actually say)

The "648 GB/s wall" is a 512 MB copy. Measured properly, transfer size
dominates:

| transfer | read GB/s |
| --- | --- |
| 2 MB | 119 |
| 9.44 MB (o) | 398 |
| 11.8 MB (qk) | 452 |
| 28.3 MB (down) | 548 |
| 56.6 MB (gu) | 592 |
| 256 MB | 627 |
| 512 MB | 639 |

It is a ramp, not a cliff. Back-to-back repeats confirm it: the same `o` GEMV
run 16 times in one graph reaches 467 GB/s/iteration versus 402 for a single
launch, and the full 36-layer chain (3822 MB) reaches 603 GB/s. The native
kernel's per-shape numbers in section 2 are all *above* the copy wall at their
own size, which is the correct way to see whether a shape is done.

Three structural findings about the M=1 shape, each measured:

* **Latency, not bandwidth, limits the narrow shapes.** N=4096 is 128 row-blocks
  of 32 rows = 128 warps at `ksplit=1`, i.e. 1.6 warps per SM. `ksplit`
  multiplies that for free: at `ksplit=1` the kernel gets 152 GB/s, at
  `ksplit=8` 420+. It is the single biggest lever for small N.
* **x must reach the FMA units as scalars.** Passing `xv + offset` (an array
  pointer) to the dot helper made the compiler materialise x in local memory and
  cost **2x** (26.2 us -> 16.8 us at N=4096 K=4096 for identical math).
* **A 16-tile prefetch ring is a trap.** `RBLK*PF` live `uint4`s pushed the
  kernel to 250+ registers, ptxas spilled, and the same math dropped from 560 to
  110 GB/s. The structure that survives is one-to-two groups per iteration with
  compile-time indices only.
* **A serial group-sum reduction is on the critical path.** A 64-add chain per
  group cost 5.3 us of a 23 us kernel (`probe4.cu`); lane-parallel shuffle
  reduction is ~2% of the group's FMA count and leaves the critical path free --
  and at CPG==1 even that is unnecessary (section 3.3).

---

## 6. Remaining headroom, honestly

### Cost: the repack duplicates the weights, and how to avoid that

The repack is a pure permutation, so the aten layout becomes redundant once the
native one exists and the net cost can be **zero**. As originally written
`NativeW4.from_qsz` kept both, which for the full n2 state costs **3.955 GiB**
(216 int4 linears; the model weights themselves are 8.242 GiB).

Measured, before and after adding the release
(`.tmp/kern_agent/release.py`):

```
after model load                          : 8.242 GiB
+ 216 native layouts (both kept)          : 12.197 GiB   (+3.955)
+ release_source=True                      :  8.242 GiB   (-3.955)
net extra VRAM for the native path         :  0.000 GiB
```

A first attempt used `resize_(0)`, which looks like it frees the storage but
does not: a 64 MiB tensor still occupied 64 MiB afterwards (measured), because
`resize_` keeps the original allocation alive. Assigning a fresh empty tensor
does free it, and it leaves the released tensor object valid-and-empty so a
stale alias raises `IndexError` instead of silently reading freed memory. Both
properties are pinned by `TestReleaseSource`.

**`release_source` is off by default.** The caller normally still needs the aten
layout -- for the tinygemm fallback and for the `_weight_int4pack_mm` comparison
tests -- so freeing it by default would turn a leak into a use-after-free. This
is the same reasoning as `fast_native`'s fused arm *popping* the per-name q/k
entries out of the base rather than letting them fall out of scope.

### Where the remaining time is

The int4 GEMMs are now the *fast* part. Against the measured chain rate of
603 GB/s the four shapes sit at 560-656 GB/s, and `o` (the smallest, 9.4 MB) is
the weakest at 1.01x. The lever that remains is `o`'s ramp, i.e. its 22 us
burst, not its kernel: there is no instruction-level slack left that the probe
could find (`probe5.cu` swept 4 thread counts x 32 ksplit x 2 xglobal x 2 unroll
without a better point).

For the project's 103.9 -> 125 steps/s goal, this work buys +3.4% and the rest
has to come from elsewhere. Per `bwopt-1` section 3 the step is 70% int4 GEMM by
time, and that 70% is now at its ramp ceiling; the remaining 30% (attention
4.5%, "other" 20%, KV write, and the bf16 v_proj / audio head, which measured
426 GB/s at 8.4 MB and 668 GB/s at 269 MB respectively) is where an
unchanged-kernel win would have to come from. The single biggest structural
lever visible from here is reducing the *number* of small bursts per step --
fusing linears, or batching the 36 layers' worth of small GEMMs -- because at
9-30 MB each one is paying the ramp 252 times per step.

---

## 7. Wrap-up record

Closing pass on the kern line. Three items: fixed the flaky test's root cause,
added the (default-off) layout release, and archived the module instead of
integrating it.

### 7.1 The flaky test: deterministic arithmetic, not a race

**`test_extreme_weights_are_exact` was not flaky in the sense of a race.** It
failed ~4% of runs because it asserted `abs(y).max() == 0.0` for a weight whose
exact answer is 0 (`q == 8`, offset-binary zero), and the kernel's arithmetic
cannot deliver a bitwise zero for that input.

Four probes on a fixed failing input, each expected to vary if anything leaked
between calls (`.tmp/kern_agent/flaky4.py`):

| probe | iter 67 | iter 113 |
| --- | --- | --- |
| 20 consecutive replays | `{1.9073486328125e-06}` | `{3.814697265625e-06}` |
| interleaved with other GEMVs + allocations | `{1.9073486328125e-06}` | `{3.814697265625e-06}` |
| explicit `cuda.synchronize()` per call | `{1.9073486328125e-06}` | `{3.814697265625e-06}` |
| a freshly constructed `NativeW4` | `{1.9073486328125e-06}` | `{3.814697265625e-06}` |

Every result set has exactly one element. No leaked fp-exception flag, stream
state or L2 residue can produce that; determinism rules out the whole class of
hypotheses that were on the table.

The residue is exactly proportional to the group size, which a race could not
know (`.tmp/kern_agent/flaky8.py`):

| group size | nonzero runs (of 200) | worst residue | power of two |
| --- | --- | --- | --- |
| 32 | 8 | 1.9073e-06 | 2^-19 |
| 64 | 16 | 3.8147e-06 | 2^-18 |
| 128 | 13 | 7.6294e-06 | 2^-17 |

**Mechanism.** The hot loop avoids a convert instruction with the
set-the-exponent trick:

```c
nib(n) = __uint_as_float(0x3F800000u | n) - 1.0f   // == n * 2^-23 exactly
```

so it accumulates `acc = 2^-23 * sum_k q_k x_k` -- a magnitude 2^23 too small --
and the epilogue scales it back:

```c
part = fmaf(sc, fmaf(kNibScale /* 2^23 */, acc, -8.0f * S), part);
```

With `q == 8`, `2^23 * acc` is exactly `8 * S`, so the terms cancel and the true
answer is 0 -- but every rounding event picked up while accumulating `acc` is
amplified by 2^23 on the way out. The floor is
`eps * 2^23 * |acc| = eps * 8 * max|S|`, which is why it scales with `g`.
`_weight_int4pack_mm` returns exactly 0 (0/500 runs nonzero, versus 11/500 for
native, same inputs -- `.tmp/kern_agent/flaky2.py`) only because its own offset
cancels in the integer domain before any multiply; it never forms the
difference.

**The kernel is not less accurate.** Against a float64 reference built from the
dequantized weights on real min/max-affine weights, 25 random `x` per shape:

| g | mean abs error, native | mean abs error, tinygemm | ratio |
| --- | --- | --- | --- |
| 32 | 1.5946e-02 | 1.9409e-02 | **0.82** |
| 64 | 1.6061e-02 | 1.9591e-02 | **0.82** |

Native is 18% *closer* to the true value. The degenerate case was the one input
that could expose the amplified floor against an exactly-zero answer.

**Why it looked intermittent.** Two reasons, and the first is a genuine gap in
the test file: this test drew from the **unseeded global RNG** while every other
test built its inputs from a seeded `Generator` (`_make_case`), so neither the
failure nor the input was reproducible. Second, when it tripped the residue was
1.9e-06..7.6e-06, surfacing as a bare `assert 1.9e-06 == 0.0`.

**Fix.** The test keeps its intent and asserts the accuracy the arithmetic
actually has: bitwise zero for inputs where exactness *is* guaranteed (zeros,
ones, powers of two all measured 0.0), and `<= 1e-04` for random `x`. The
tolerance is derived, not fitted (`eps * 8 * max|S|` is ~1.5e-05 for `g=64`;
observed worst over 600 draws is 7.6e-06, so 1e-04 is ~13x the worst and still 6
orders of magnitude below real outputs of ~1e2).

Two regression tests were added: `test_extreme_weights_residue_is_deterministic`
(asserts the four probes above agree exactly -- it fails the moment anything
really does leak between calls) and `test_is_at_least_as_accurate_as_tinygemm`
(the counter-evidence to the degenerate case, pinning the 0.82x ratio). An
autouse fixture now gives every test a clean `cuda.synchronize()` boundary and a
seeded RNG; it is explicitly *not* claimed as the fix, only as insurance against
a future test that does depend on device state.

**Stability: 5 consecutive full runs, 60 passed / 1 skipped each time.**

### 7.2 Releasing the duplicated layout

Implemented as `NativeW4.from_qsz(..., release_source=True)` plus the module-level
`release_source_tensors()`. Measured on the real n2 state
(`.tmp/kern_agent/release.py`):

```
after model load                          : 8.242 GiB, reserved 8.463 GiB
216 int4 linears in the state
their aten payload+meta                    : 3.955 GiB
+ 216 native layouts (both kept)           : 12.197 GiB   (+3.955)
+ release_source=True                      :  8.242 GiB   (-3.955)
net extra VRAM for the native path          :  0.000 GiB
reusing a released tensor                   : IndexError (loud, not silent)
post-release gemv on L0.q                   : (4096,) finite
```

That corrects §6's earlier 1.58 GiB figure, which was measured only over the
linears that happened to fit before an OOM; the true total for all 216 int4
linears is **3.955 GiB**.

One bug was found and fixed while implementing it: `resize_(0)` *looks* like it
frees a tensor but keeps the storage alive (measured: a 64 MiB tensor still
occupied 64 MiB), so the first version released nothing. Assigning a fresh empty
tensor frees it and leaves the object valid-and-empty, so a stale alias raises
instead of reading freed memory. Both properties are in `TestReleaseSource`.

**Off by default.** The caller normally still needs the aten layout (the
tinygemm fallback, the `_weight_int4pack_mm` tests), so freeing it by default
would convert a leak into a use-after-free.

### 7.3 Speed did not regress

Re-ran the interleaved A/B after the release change: **+3.36% (median), 4/4
rounds**, sd 0.0002, versus +3.37% before -- identical within the measurement's
own noise, so well inside the "must not regress >0.5%" acceptance bar.

### 7.4 Non-integration verified

`git status` before the archive commit showed only two untracked paths,
`moss_tts_lite/_native/` and `tests/test_native_kernel.py`; no tracked file was
modified. `fast_native.py` and `cli.py` contain no reference to `_native`, so
the shipped default (tinygemm) is untouched and no CLI switch was added. The
only wiring that ever existed lived in gitignored harness scripts under
`.tmp/kern_agent/live.py`.

---

## 8. Decision record: archived, not integrated

**Decision (user, verbatim points): the hand-written kernel line is abandoned --
a sub-5% gain does not justify the ongoing maintenance cost of a precompiled
binary distribution matrix. Do not integrate.**

Rationale, as recorded:

* **The gain is below the threshold that matters.** +3.36% on one tier. The
  user's framing: a gain under 5% is not meaningful here.
* **The distribution cost is permanent.** Enabling this in the shipped path
  commits to building, testing and publishing a CUDA binary for every supported
  SM x toolkit combination, indefinitely, plus an `nvcc` dependency in the
  release process. That is a recurring cost paid against a one-time
  low-single-digit speedup -- the wrong trade regardless of how good the kernel
  is.
* **The correctness surface grows in a way tinygemm's does not.** The kernels
  expose tuning knobs the reference cannot have, and four of the six bugs found
  in this work were silent-wrong-answer bugs reachable only through those knobs
  (§4).

What is kept, and why it is worth the repository space:

1. **`moss_tts_lite/_native/`** -- kernels, binding, layout code and the
   prebuilt sm_86 `.so`, with `README.md` stating plainly that it is
   experimental, not enabled, and documenting exactly how to enable it
   (build for your SM, check `available()`, swap the linears, decide about
   `release_source`) should anyone revisit the decision.
2. **`tests/test_native_kernel.py`** -- 60 tests, runnable without enabling
   anything. This is the part with the clearest long-term value: the four
   silent-wrong-answer bug classes have pins, the `release_source` semantics are
   pinned, and the degenerate-weight numerics are pinned from both sides
   (determinism, and the 0.82x accuracy ratio against the reference).
3. **This report** -- the bandwidth-ramp correction (§5), which invalidated the
   per-shape targets every earlier optimisation round was chasing, and the
   determinism-versus-race discrimination method (§7.1), which is reusable for
   any future flaky-GPU-test diagnosis.

**No integration was reverted because none was ever committed** (§7.4). The
default path is unchanged: `tinygemm`, no autodetection, no CLI switch.
