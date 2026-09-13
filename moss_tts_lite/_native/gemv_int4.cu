// MOSS-TTS native W4 GEMV kernels (pure CUDA, no libtorch).
//
// The .so built from this file depends only on the CUDA runtime; the Python
// side passes raw device pointers through ctypes, so the artefact is
// independent of the local PyTorch/Python versions.
//
// ---------------------------------------------------------------------------
// Native payload layout: a pure permutation of the shipped bytes
// ---------------------------------------------------------------------------
// For an int4 tensor [N, K], K % 32 == 0, rows zero-padded up to a multiple of
// 32 (`repack_int4` in layout.py):
//
//   offset(rb, j, n, m) = (((size_t)rb * KC + j) * 512) + (n & 31) * 16 + m
//   rb = n >> 5,  j = k >> 5,  KC = K / 32,  m = (k >> 1) & 15
//   byte m holds k = 32j + 2m     in its low nibble
//                k = 32j + 2m + 1 in its high nibble
//
// so a 32-row x 32-k tile is one contiguous 512-byte block: a warp reads one
// uint4 per lane and covers a whole tile in one coalesced 128-byte-per-sector
// transaction.  (In the shipped `_convert_weight_to_int4pack` layout those 512
// bytes hold an 8-row x 128-k tile, which an M=1 GEMV cannot read coalesced.)
//
//   meta (the shipped `qsz` pairs, bf16, values unmodified -- only reordered):
//   offset(rb, gi, lane, t) = (((size_t)rb * GC + gi) * 128) + lane * 4 + 2*t
//   t = 0 -> scale, t = 1 -> zero,  gi = k / g,  GC = K / g
//
// ---------------------------------------------------------------------------
// Kernel shape (why it looks like this)
// ---------------------------------------------------------------------------
// lane == row; 32 rows per warp; 8 warps per block.  Each block owns a
// contiguous k-slice of `nj` 32-k tiles (gridDim.y = ksplit), because the M=1
// GEMV is latency bound long before it is throughput bound: the narrow shapes
// have only N/32 = 128 warps at ksplit=1, i.e. 1.6 warps per SM, nowhere near
// the in-flight bytes needed to cover DRAM latency.  Splitting K multiplies the
// warp count without adding any payload traffic.
//
// The k-loop is *not* fully unrolled into one giant straight line.  That was
// the first design and it is a trap: unrolling 16-32 tiles of RBLK*PF uint4
// prefetch registers pushes the kernel to 250+ registers, at which point ptxas
// spills into local memory and the achieved bandwidth collapses (measured:
// N=24576 K=4096, ks=1 rblk=4 pf=16 -> 110 GB/s vs 560 GB/s for the same math
// with pf=4).  The structure here is one tile per iteration with a PF-deep
// register ring indexed by a compile-time-unrolled inner loop, so register use
// is bounded by PF * (RBLK + x) rather than by nj.
//
// Dequantisation uses the "0x3F800000 | n" bit trick:
//     (float)(0x3F800000 | n) - 1.0f  ==  n * 2^-23     (exact, n in [0,15])
// so a nibble becomes a float with one LOP3 + one FADD and the dot is plain
// fp32 FMA, with no int->float conversion pipe traffic.  The 2^-23 is folded
// back into the scale in the group epilogue, which applies the affine pair
// exactly as `_weight_int4pack_mm` defines it:
//
//     out = sum_k ( (q_k - 8) * scale_g + zero_g ) * x_k
//         = scale_g * sum_g (q_k - 8) * x_k  +  zero_g * sum_g x_k
//
// and `sum_g (q-8)x = 2^23 * acc - 8 * ssum[g]`, where `ssum[g]` is the group
// sum of x, computed once per block in the prologue (so the zero term is free).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

#define MOSS_EXPORT extern "C" __attribute__((visibility("default")))

namespace moss {

constexpr int kThreads = 256;             // 8 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kRowBlock = 32;             // rows per warp slice
constexpr int kTileLoads = 32;            // uint4 per lane per 32-k tile

// (float)(0x3F800000 | n) == 1.0f + n*2^-23 exactly, so the subtraction is
// exact and the result is a scaled nibble -- no cancellation anywhere.
__device__ __forceinline__ float nib(uint32_t n) {
    return __uint_as_float(0x3F800000u | n) - 1.0f;
}
constexpr float kNibScale = 8388608.0f;   // 2^23

// One uint32 = 8 nibbles against x0..x7 (x0 == the low nibble == even k).
//
// The x operands are *scalars*, never a pointer: passing `xv + offset` makes
// the compiler materialise the x array in local memory, which costs 2x the
// bandwidth of this kernel (measured: N=4096 K=4096, 26.2 us vs 16.8 us for the
// identical math with scalar operands).
#define MOSS_DOT8S(w, x0, x1, x2, x3, x4, x5, x6, x7, a, b)   \
    a = fmaf(nib((w) & 0xFu),          x0, a);                 \
    a = fmaf(nib(((w) >> 4) & 0xFu),   x1, a);                 \
    a = fmaf(nib(((w) >> 8) & 0xFu),   x2, a);                 \
    a = fmaf(nib(((w) >> 12) & 0xFu),  x3, a);                 \
    b = fmaf(nib(((w) >> 16) & 0xFu),  x4, b);                 \
    b = fmaf(nib(((w) >> 20) & 0xFu),  x5, b);                 \
    b = fmaf(nib(((w) >> 24) & 0xFu),  x6, b);                 \
    b = fmaf(nib((w) >> 28),           x7, b);

// one uint4 = 32 nibbles against x[0..32), a 32-entry register array indexed
// only by compile-time constants
#define MOSS_DOT32(vec, xs, a0, a1, a2, a3)                              \
    MOSS_DOT8S((vec).x, xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6],  \
               xs[7], a0, a1)                                            \
    MOSS_DOT8S((vec).y, xs[8], xs[9], xs[10], xs[11], xs[12], xs[13],     \
               xs[14], xs[15], a0, a1)                                   \
    MOSS_DOT8S((vec).z, xs[16], xs[17], xs[18], xs[19], xs[20], xs[21],   \
               xs[22], xs[23], a2, a3)                                   \
    MOSS_DOT8S((vec).w, xs[24], xs[25], xs[26], xs[27], xs[28], xs[29],   \
               xs[30], xs[31], a2, a3)

__device__ __forceinline__ float bf16_to_f32(uint32_t hi16) {
    return __uint_as_float(hi16 << 16);
}

// The payload is streamed: every byte is read exactly once and never revisited,
// so it takes the evict-first hint and leaves L2 to the metadata and x, which
// ARE reused across row blocks.  `__ldcs` vs `__ldg` is worth a few percent on
// the short shapes (`.tmp/reports/kern-1.md` section 4).
__device__ __forceinline__ uint4 ld_payload(const uint4* p) {
    return __ldcs(p);
}

// x may arrive as bf16 (the model's activations) or fp32.  Widening in the
// kernel rather than in a preceding pass is worth real time: a separate
// `xbuf.copy_(x)` is a *dependent* launch of its own, and at 252 int4 linears
// per step that measured 0.485 ms/step (`.tmp/reports/kern-1.md` section 2).
__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// ---------------------------------------------------------------------------
// int4 GEMV
// ---------------------------------------------------------------------------
// CPG      : 32-k tiles per quantization group == g / 32   (1, 2, 4)
// KSPLIT   : k-slices per block (1, 2, 4, 8)
// XGLOBAL  : read x from global memory instead of staging it in shared memory
//
// One warp owns 32 rows (lane == row) and one k-slice; the block (8 warps)
// covers 8/KSPLIT row blocks, and the KSPLIT partial sums of each row are
// reduced in shared memory before the bf16 store.  So a *single* kernel does
// what first took three (gemv + group-sum + reduce), and KSPLIT is the lever
// that gives the narrow shapes their warps: N=4096 is only 128 row blocks of
// 32 rows, i.e. 1.6 warps per SM, nowhere near the ~400 KB of in-flight data
// needed to cover DRAM latency at 650 GB/s.  KSPLIT=8 turns those into 1024
// warps with no extra payload traffic and no launch cost.
//
// Two costs are deliberately avoided:
//
//  * The affine `zero` term needs the per-group sum of x.  Computing it as a
//    serial chain over the group, once per group per warp, costs as much shared
//    traffic as the dot product itself (64 LDS + 64 FADD per group against 128
//    FMA per group).  Instead each warp reduces *its own slice's* group sums
//    once, lane-parallel, into registers before the main loop; the epilogue
//    then reads a register.
//  * Staging x in shared memory (`XGLOBAL=false`) costs one 16-48 KB load plus
//    a `__syncthreads` before any useful work, and it is redundant across the
//    N/32 blocks of the same k-slice.  For the narrow shapes the sync is on the
//    critical path of a 24 us kernel, so `XGLOBAL=true` (each warp reads x
//    straight from global; L1 broadcast serves the 32 lanes) is available and
//    selected per shape by the caller.
//
// Dequantisation uses the "0x3F800000 | n" bit trick:
//     (float)(0x3F800000 | n) - 1.0f  ==  n * 2^-23     (exact, n in [0,15])
// so a nibble becomes a float with one LOP3 + one FADD and the dot is plain
// fp32 FMA, with no int->float conversion pipe traffic.  The 2^-23 is folded
// back into the scale in the epilogue, which applies
//
//     out = sum_k ( (q_k - 8) * scale_g + zero_g ) * x_k
//         = scale_g * 2^23 * a  +  (zero_g - 8 * scale_g) * sum_g x_k
//
// exactly as `_weight_int4pack_mm` defines it.
template <typename XT, int THREADS, int CPG, int KSPLIT, int UNROLL,
          bool XGLOBAL>
__global__ void __launch_bounds__(THREADS)
gemv_int4_kernel(const uint4* __restrict__ payload,
                 const uint32_t* __restrict__ meta,
                 const XT* __restrict__ x,
                 __nv_bfloat16* __restrict__ out_bf16,
                 const int N, const int KC, const int GC, const int nj)
{
    constexpr int W = THREADS / 32;      // warps per block
    // [row block within this grid block][slice][lane].  KSPLIT may not exceed W
    // (a slice must own at least one warp), which the host checks; the clamp
    // here keeps the array well-formed for the KSPLIT==1 instantiation where
    // it is never touched.
    constexpr int kRowsInBlock = (W / (KSPLIT < W ? KSPLIT : W)) > 0
                               ? (W / (KSPLIT < W ? KSPLIT : W)) : 1;
    __shared__ float sred[kRowsInBlock][KSPLIT][kRowBlock];
    extern __shared__ float sx[];        // XGLOBAL ? 0 : K floats

    const int K = KC * 32;
    const int G = K / GC;                // group size
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (!XGLOBAL) {
        // bf16 x is widened to fp32 *here*, in the same kernel, rather than by
        // a separate pass: `xbuf.copy_(x)` is a dependent launch of its own,
        // and at 252 int4 linears per step that measured 0.485 ms/step
        // (`.tmp/reports/kern-1.md` section 2) -- the entire live-step deficit.
        for (int i = tid; i < K; i += THREADS) sx[i] = widen(x[i]);
        __syncthreads();
    }

    // warp -> (row block, k-slice): slice varies fastest, so the warps of a
    // block read neighbouring k-tiles of the same 512-byte rows
    const int slice = warp % KSPLIT;
    const int rblk = warp / KSPLIT;
    const int rb = blockIdx.x * (W / KSPLIT) + rblk;
    const int n = rb * kRowBlock + lane;
    const int nj_s = nj / KSPLIT;        // 32-k tiles in this warp's slice
    const int j0 = slice * nj_s;
    const int rb_limit = (N + kRowBlock - 1) / kRowBlock;
    const bool live = (rb < rb_limit);

    float part = 0.f;
    if (live) {
        const uint4* p = payload + ((size_t)rb * KC + j0) * kTileLoads + lane;
        const uint32_t* mp = meta + (size_t)rb * GC * 32 + lane;
        const int gi0 = j0 / CPG;
        // UNROLL groups per iteration: issue all UNROLL*CPG tile loads *before*
        // any of their uses, so the warp keeps that many 512-byte transactions
        // in flight instead of one.  Little's law at 650 GB/s with ~700 ns of
        // DRAM latency wants ~400 KB in flight; a warp with a single dependent
        // load per iteration cannot get there no matter how many warps are
        // resident, which is part of why the narrow shapes (9-12 MB, 17-23 us)
        // sit near 500 GB/s.  Every address is known up front, so this is
        // ordinary unrolling: no dynamic register indexing, no spills.
        for (int j = 0; j < nj_s; j += CPG * UNROLL) {
            uint4 w[UNROLL][CPG];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
#pragma unroll
                for (int c = 0; c < CPG; ++c)
                    w[u][c] = ld_payload(
                        &p[(size_t)(j + u * CPG + c) * kTileLoads]);
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                for (int c = 0; c < CPG; ++c) {
                    const int jc = j + u * CPG + c;
                    float xv[32];
                    if (XGLOBAL) {
                        // One coalesced 128-byte load per tile, then a warp
                        // transpose: lane t holds x[32jc + t], and the 32 dot
                        // operands are the 32 shuffles.  Reading x with 32
                        // lane-uniform `__ldg`s instead would issue 32 L1
                        // broadcasts per tile, which is the same traffic but
                        // 32x the instructions.
                        const XT* xp = x + (size_t)(j0 + jc) * 32;
                        const float mine = widen(__ldg(&xp[lane]));
#pragma unroll
                        for (int t = 0; t < 32; ++t)
                            xv[t] = __shfl_sync(0xFFFFFFFFu, mine, t);
                    } else {
                        const float* xp = sx + (size_t)(j0 + jc) * 32;
#pragma unroll
                        for (int t = 0; t < 32; ++t) xv[t] = xp[t];
                    }
                    MOSS_DOT32(w[u][c], xv, a0, a1, a2, a3);
                }
                const int gi = gi0 + (j / CPG) + u;
                const uint32_t mv = __ldg(&mp[(size_t)gi * 32]);
                const float sc = bf16_to_f32(mv & 0xFFFFu);
                const float z  = bf16_to_f32(mv >> 16);
                // Group sum of x, lane-parallel: G/32 reads plus a 5-step
                // shuffle reduce, i.e. ~2% of the group's 128*CPG FMA.  The
                // serial form (`for t < G: S += gx[t]`) costs a 64-add chain
                // per group on the critical path and measured 5.3 us of a 23 us
                // kernel at K=4096 (`.tmp/kern_agent/probe4.cu`).  The group
                // always lies wholly inside this warp's slice (a group is CPG
                // tiles; nj/KSPLIT is a multiple of CPG), so no cross-warp
                // reduction is needed.
                // The group sum of x.  At CPG == 1 a group is exactly the
                // 32 values of one tile, i.e. *every* lane already has its own
                // element of the sum in `xv[0]` -- so the whole reduction is a
                // single shuffle tree, or, better, nothing at all: the sum is
                // just the sum of the tile, which the dot's own accumulator
                // does not give us, but a 5-step tree over the value each lane
                // read is still 5 instructions instead of G/32 loads.  At
                // CPG > 1 the group spans several tiles, so the loads are real.
                float S = 0.f;
                if (CPG == 1) {
                    S = XGLOBAL ? widen(__ldg(&x[(size_t)gi * G + lane]))
                                : sx[(size_t)gi * G + lane];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        S += __shfl_xor_sync(0xFFFFFFFFu, S, o);
                } else if (XGLOBAL) {
                    const XT* gx = x + (size_t)gi * G;
                    for (int t = lane; t < G; t += 32) S += widen(__ldg(&gx[t]));
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        S += __shfl_xor_sync(0xFFFFFFFFu, S, o);
                } else {
                    const float* gx = sx + (size_t)gi * G;
                    for (int t = lane; t < G; t += 32) S += gx[t];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        S += __shfl_xor_sync(0xFFFFFFFFu, S, o);
                }
                const float acc = (a0 + a1) + (a2 + a3);
                // sum_g (q-8)*x == 2^23 * acc - 8 * S
                part = fmaf(sc, fmaf(kNibScale, acc, -8.0f * S), part);
                part = fmaf(z, S, part);
            }
        }
    }

    if (KSPLIT == 1) {
        if (live && n < N) out_bf16[n] = __float2bfloat16(part);
        return;
    }

    // every warp contributes (0 if it had no live rows) so the reduction below
    // is uniform and needs no extra synchronisation
    sred[rblk][slice][lane] = live ? part : 0.f;
    __syncthreads();
    if (slice == 0) {
        const int row = blockIdx.x * (W / KSPLIT) + rblk;
        const int nn = row * kRowBlock + lane;
        if (row < rb_limit && nn < N) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < KSPLIT; ++q) s += sred[rblk][q][lane];
            out_bf16[nn] = __float2bfloat16(s);
        }
    }
}

// ---------------------------------------------------------------------------
// bf16 GEMV (v_proj): warp per row, 128-bit loads, fp32 accumulation
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads)
gemv_bf16_kernel(const uint4* __restrict__ w,
                 const float* __restrict__ xf,
                 __nv_bfloat16* __restrict__ out,
                 const int N, const int K)
{
    const int warp = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (warp >= N) return;
    const int lane = threadIdx.x & 31;
    const uint4* rp = w + (size_t)warp * (K / 8);
    float s0 = 0.f, s1 = 0.f;
    const int n4 = K / 8;
    for (int i = lane; i < n4; i += 32) {
        uint4 v = __ldg(&rp[i]);
        const float* xp = xf + 8 * i;
        s0 = fmaf(bf16_to_f32(v.x & 0xFFFFu), xp[0], s0);
        s0 = fmaf(bf16_to_f32(v.x >> 16),    xp[1], s0);
        s0 = fmaf(bf16_to_f32(v.y & 0xFFFFu), xp[2], s0);
        s0 = fmaf(bf16_to_f32(v.y >> 16),    xp[3], s0);
        s1 = fmaf(bf16_to_f32(v.z & 0xFFFFu), xp[4], s1);
        s1 = fmaf(bf16_to_f32(v.z >> 16),    xp[5], s1);
        s1 = fmaf(bf16_to_f32(v.w & 0xFFFFu), xp[6], s1);
        s1 = fmaf(bf16_to_f32(v.w >> 16),    xp[7], s1);
    }
    float s = s0 + s1;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) out[warp] = __float2bfloat16(s);
}

// batch bf16 -> fp32, one row per gridDim.y slice
__global__ void bf16_rows_to_f32_kernel(const __nv_bfloat16* __restrict__ in,
                                        float* __restrict__ out,
                                        const int rows, const int K)
{
    const int r = blockIdx.y;
    if (r >= rows) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K;
         i += gridDim.x * blockDim.x)
        out[(size_t)r * K + i] = __bfloat162float(in[(size_t)r * K + i]);
}

}  // namespace moss

// ---------------------------------------------------------------------------
// host-side entry points (plain-C ABI)
// ---------------------------------------------------------------------------
// Request the opt-in shared-memory size for `fn`.  The threshold is the
// 48 KB default but the *total* per-block budget also has to cover the static
// `sred` array, so the call is made unconditionally: asking for a value below
// the default is a no-op, and asking for a value above it fails loudly instead
// of at launch time (which is how a "works at ksplit=1, rc=1 at ksplit>1"
// difference appears when the dynamic size is right at 48 KB -- K=12288).
template <typename Fn>
static int optin_smem(Fn fn, int bytes) {
    cudaError_t e = cudaFuncSetAttribute(
        fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (e == cudaSuccess) return 1;
    // asking for no more than the 48 KB default is a no-op, so tolerate the
    // resulting invalid-value; anything else is a real failure.  (Without the
    // unconditional call, a kernel whose dynamic request sits exactly at 48 KB
    // launches at ksplit=1 and fails with rc=1 at ksplit>1 -- K=12288.)
    cudaError_t last = cudaGetLastError();
    if (last == cudaErrorInvalidValue && bytes <= 48 * 1024) return 1;
    return 0;
}

// One block covers (THREADS/32)/KSPLIT row blocks x KSPLIT k-slices.
template <typename XT, int THREADS, int CPG, int KSPLIT, int UNROLL,
          bool XGLOBAL>
static int launch_one(const void* payload, const void* meta, const void* x,
                      void* out_bf16, int N, int K, int g, cudaStream_t stream) {
    const int GC = K / g;
    const int nj = K / 32;
    // A group is CPG consecutive 32-k tiles and must live entirely inside one
    // slice, otherwise the epilogue would apply a partial group's (scale, zero)
    // to a full group's sum.  This is exactly nj/KSPLIT % CPG == 0.
    if (nj % KSPLIT || (nj / KSPLIT) % CPG || (CPG * 32) != g) return -6;
    if (KSPLIT > THREADS / 32) return -9;   // a slice needs at least one warp
    if (UNROLL < 1 || UNROLL > 2) return -10;
    if ((nj / KSPLIT) % (CPG * UNROLL)) return -10;
    const size_t smem = XGLOBAL ? 0 : (size_t)K * sizeof(float);
    if (smem > 96 * 1024) return -7;
    if (!optin_smem(
            moss::gemv_int4_kernel<XT, THREADS, CPG, KSPLIT, UNROLL, XGLOBAL>,
            (int)smem))
        return -2;
    const int rows_per_block = (THREADS / 32) / KSPLIT * moss::kRowBlock;
    const int blocks = (N + rows_per_block - 1) / rows_per_block;
    moss::gemv_int4_kernel<XT, THREADS, CPG, KSPLIT, UNROLL, XGLOBAL>
        <<<blocks, THREADS, smem, stream>>>(
            (const uint4*)payload, (const uint32_t*)meta, (const XT*)x,
            (__nv_bfloat16*)out_bf16, N, K / 32, GC, nj);
    return (int)cudaGetLastError();
}

// Dispatch table.  The key must encode *every* template argument that the
// callee depends on: keying only on (ksplit, unroll) and then adding `threads`
// separately inside each case had two names collide -- ksplit=3 fell through to
// the ksplit=1 instantiation and silently returned a wrong answer instead of
// the -6 that `launch_one` would have raised.  Unsupported (or illegal)
// combinations must reach the `default` and fail loudly.
#define MOSS_LAUNCH(TH, KS, UN)                                                \
    (xbf16 ? (int)launch_one<__nv_bfloat16, TH, CPG, KS, UN, XGLOBAL>(         \
                 payload, meta, x, out_bf16, N, K, g, stream)                  \
           : (int)launch_one<float, TH, CPG, KS, UN, XGLOBAL>(                 \
                 payload, meta, x, out_bf16, N, K, g, stream))

// Dispatch table.  The key encodes every template argument the callee depends
// on.  Keying only on (ksplit, unroll) and then adding `threads` inside each
// case had two names collide -- ksplit=3 fell through to the ksplit=1
// instantiation and silently returned a wrong answer with rc=0 instead of the
// -6 that `launch_one` raises for a split that cuts a group in half.
// Unsupported combinations must reach the `default` and fail loudly.
#define MOSS_CASE(TH, KS, UN) case (TH) / 128 * 10000 + (KS) * 100 + (UN)

template <int CPG, bool XGLOBAL>
static int launch_ks(int ks, int threads, int unroll, bool xbf16,
                     const void* payload, const void* meta, const void* x,
                     void* out_bf16, int N, int K, int g,
                     cudaStream_t stream) {
    switch (threads / 128 * 10000 + ks * 100 + unroll) {
        MOSS_CASE(256, 1, 1): return MOSS_LAUNCH(256, 1, 1);
        MOSS_CASE(256, 1, 2): return MOSS_LAUNCH(256, 1, 2);
        MOSS_CASE(256, 2, 1): return MOSS_LAUNCH(256, 2, 1);
        MOSS_CASE(256, 2, 2): return MOSS_LAUNCH(256, 2, 2);
        MOSS_CASE(256, 4, 1): return MOSS_LAUNCH(256, 4, 1);
        MOSS_CASE(256, 4, 2): return MOSS_LAUNCH(256, 4, 2);
        MOSS_CASE(256, 8, 1): return MOSS_LAUNCH(256, 8, 1);
        MOSS_CASE(256, 8, 2): return MOSS_LAUNCH(256, 8, 2);
        MOSS_CASE(512, 1, 1): return MOSS_LAUNCH(512, 1, 1);
        MOSS_CASE(512, 1, 2): return MOSS_LAUNCH(512, 1, 2);
        MOSS_CASE(512, 2, 1): return MOSS_LAUNCH(512, 2, 1);
        MOSS_CASE(512, 2, 2): return MOSS_LAUNCH(512, 2, 2);
        MOSS_CASE(512, 4, 1): return MOSS_LAUNCH(512, 4, 1);
        MOSS_CASE(512, 4, 2): return MOSS_LAUNCH(512, 4, 2);
        MOSS_CASE(512, 8, 1): return MOSS_LAUNCH(512, 8, 1);
        MOSS_CASE(512, 8, 2): return MOSS_LAUNCH(512, 8, 2);
        MOSS_CASE(512, 16, 1): return MOSS_LAUNCH(512, 16, 1);
        MOSS_CASE(512, 16, 2): return MOSS_LAUNCH(512, 16, 2);
        MOSS_CASE(1024, 1, 1): return MOSS_LAUNCH(1024, 1, 1);
        MOSS_CASE(1024, 2, 1): return MOSS_LAUNCH(1024, 2, 1);
        MOSS_CASE(1024, 4, 1): return MOSS_LAUNCH(1024, 4, 1);
        MOSS_CASE(1024, 8, 1): return MOSS_LAUNCH(1024, 8, 1);
        MOSS_CASE(1024, 16, 1): return MOSS_LAUNCH(1024, 16, 1);
        MOSS_CASE(1024, 32, 1): return MOSS_LAUNCH(1024, 32, 1);
        default: return -5;
    }
}
#undef MOSS_CASE
#undef MOSS_LAUNCH

static int gemv_common(const void* payload, const void* meta, const void* x,
                       void* out_bf16, int N, int K, int g, int ksplit,
                       int xglobal, int threads, int unroll, bool xbf16,
                       void* stream)
{
    if (g % 32 || K % 32 || K % g) return -3;
    cudaStream_t s = (cudaStream_t)stream;
#define MOSS_CPG_CASE(C)                                                        \
    case C:                                                                     \
        return xglobal                                                          \
            ? launch_ks<C, true>(ksplit, threads, unroll, xbf16, payload, meta, \
                                 x, out_bf16, N, K, g, s)                       \
            : launch_ks<C, false>(ksplit, threads, unroll, xbf16, payload,      \
                                  meta, x, out_bf16, N, K, g, s);
    switch (g / 32) {
        MOSS_CPG_CASE(1)
        MOSS_CPG_CASE(2)
        MOSS_CPG_CASE(4)
    }
#undef MOSS_CPG_CASE
    return -5;
}

MOSS_EXPORT int moss_gemv_int4(const void* payload, const void* meta,
                               const void* x, void* out_bf16,
                               int N, int K, int g, int ksplit, int xglobal,
                               int threads, int unroll, void* stream)
{
    return gemv_common(payload, meta, x, out_bf16, N, K, g, ksplit, xglobal,
                       threads, unroll, false, stream);
}

MOSS_EXPORT int moss_gemv_int4_bf16(const void* payload, const void* meta,
                                    const void* x, void* out_bf16,
                                    int N, int K, int g, int ksplit, int xglobal,
                                    int threads, int unroll, void* stream)
{
    return gemv_common(payload, meta, x, out_bf16, N, K, g, ksplit, xglobal,
                       threads, unroll, true, stream);
}

MOSS_EXPORT int moss_gemv_bf16_launch(const void* w, const void* xf,
                                      void* out_bf16, int N, int K, void* stream)
{
    if (K % 8 || N <= 0) return -3;
    cudaStream_t s = (cudaStream_t)stream;
    moss::gemv_bf16_kernel<<<(N + moss::kWarps - 1) / moss::kWarps,
                             moss::kThreads, 0, s>>>(
        (const uint4*)w, (const float*)xf, (__nv_bfloat16*)out_bf16, N, K);
    return (int)cudaGetLastError();
}

MOSS_EXPORT int moss_bf16_rows_to_f32(const void* in, void* out, int rows,
                                      int K, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(32, rows);
    moss::bf16_rows_to_f32_kernel<<<grid, moss::kThreads, 0, s>>>(
        (const __nv_bfloat16*)in, (float*)out, rows, K);
    return (int)cudaGetLastError();
}

MOSS_EXPORT int moss_device_info(int* sm_count, int* smem_per_block,
                                 int* smem_per_sm, int* cc_major, int* cc_minor)
{
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, dev);
    if (e != cudaSuccess) { cudaGetLastError(); return -1; }
    if (sm_count) *sm_count = p.multiProcessorCount;
    if (smem_per_block) *smem_per_block = (int)p.sharedMemPerBlockOptin;
    if (smem_per_sm) *smem_per_sm = (int)p.sharedMemPerMultiprocessor;
    if (cc_major) *cc_major = p.major;
    if (cc_minor) *cc_minor = p.minor;
    return 0;
}

MOSS_EXPORT const char* moss_version(void) { return "moss-native-1"; }
