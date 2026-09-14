# fast-native design notes (`--fast-native`, arm n2)

Measured facts that shaped this module. Numbers: A10G-24G, shipped w1p
checkpoint, zh_plain, seed 1234, audio-phase step at a fixed cache length.

## Why whole-step graphs (n1) only bought 2%

| path | ms/step | steps/s | text argmax | top-25 cover |
|---|---|---|---|---|
| fast.py (38 sub-graphs) | 12.23 | 81.7 | 100.00% | 75.39% |
| n1 whole-step graph | 12.01 | 83.3 | 100.00% | 98.91% |
| n2 + fused GEMMs/norm/heads | 10.21 | 97.9 | 100.00% | 97.50% |

The step is GPU-bound at ~1080 kernels; merging graph launches removes no
kernels. n1 is bitwise-identical to fast.py (verified over 40 teacher-forced
rows in both phases: `lt_buf`/`two_buf`/`la_buf` max|d| = 0, across the
`gqa_max_len` attention-mode switch at len 691-726; generated utterance
row-for-row equal for all 164 steps).

## Where n2's 16% comes from (and what it costs)

Kernel-count cuts; step kernels drop from ~2346 to ~1080:

- 32 audio heads → one `[32*1025, 4096]` GEMM
- audio-phase text rows → one `[2, 4096]` GEMM
- `(q,k)` and `(gate,up)` → single int4 GEMMs over concatenated payloads
  (bitwise-neutral: concatenating output rows leaves each row's K
  accumulation untouched)
- 33 embedding gathers → one gather + one reduction
- 6-op `_rms_norm` fp32 chain → `F.rms_norm` (the swap alone moves audio
  logits by 0.5; this is why n2 is NOT bitwise — it decodes a different but
  equally valid utterance, top-25 gate agreement 97.50% clears the >=95%
  hard gate that the unmodified W4 path fails at 75.39%)
- rope → one table gather (`_rms_norm` kept only in `prefill`, where a
  per-step difference compounds into a different utterance)

Attention keeps fast.py's exact-length call; since that length is fixed inside
a graph, one graph is captured per cache length (`bucket_for` documents why
every padded/bucketed alternative was measured and rejected; `max_graphs`
bounds the graph cache).

## n3 (in-graph FSM + sampling) — measured and rejected

Drawing all 32 channels every step (vs the reference's channel-subset draws)
changes the RNG stream, and on this model that makes the trajectory run away
(zh 3170 steps vs 127; en non-terminating within 4096), so it is both wrong
and slow (the per-length graph cache thrashes once a run exceeds
`max_graphs`). Kept for reference only; see `ARMS["n3"]`.

## Bugs found while building this (pinned by tests/test_fast.py phase C)

1. "audio rows diverged from step 1" and "en ran 1766 steps" had the same
   cause — `replay()` publishing only `two_buf` in the audio phase, so audio
   tokens were sampled from `model.audio_logits()` on the prefill hidden
   state. Fixed by taking `la_step` unconditionally.
2. `not is_stopping` guards in `text_decision` — a real deviation from the
   reference, not observable as a step count on these prompts; pinned as a
   differential table.

Both fixes are verified to bite by mutation.

## Not pursued (measured and left alone)

The int4 GEMMs are 7.16 of the ~10 ms and run at ~423 GB/s of the A10G's
~600 GB/s, so the remaining headroom is memory bandwidth, not launch overhead
— the activations sum to under 1 MiB and the cost of removing whole blocks
(attention 0.46 ms, KV write 0.17 ms, q/k norms 0.12 ms) is only 0.70 ms in
total.
