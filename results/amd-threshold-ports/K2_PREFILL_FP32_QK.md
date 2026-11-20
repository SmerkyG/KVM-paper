# K2 Horizon 64K threshold prefill on MI325X

K2 Horizon 32B FP8, BF16 KV, batch one, and a fixed 65,536-token ProLong prompt. The threshold runs use normalized region mass with c=16 throughout remote prefill and the original 16,384-token exact prefix. They retain the region schedule, exact-leaf key order, and serving model projection kernels. Timings are medians of three warmed requests and include cache construction.

## Full-model timing

| Path | Median prefill | GPU node |
|---|---:|---|
| [Native top-eight, earlier baseline](k2-prefill-fast-centroid-v1/amd-k2-fast-centroid-top8.json) | 13.66 s | 2 |
| [Native top-eight, same-round repeat](k2-prefill-combined-path-v1/amd-k2-same-round-top8.json) | 13.50 s | 2 |
| [Previous fastest c=16](k2-prefill-storage-reuse-v1/amd-k2-bitwise-speed-fused-pad.json) | 15.60 s | 2 |
| [c=16 with a shorter value MMA for short frontiers](k2-prefill-value-half-v1/amd-k2-value-half-speed.json) | 15.61 s | 2 |
| [Plus tiled FP32 centroid QK](k2-prefill-tuned-qk-v1/amd-k2-qktile-speed.json) | 15.44 s | 2 |
| [Plus aligned selector loads](k2-prefill-align-selector-v1/amd-k2-padded-score-speed.json) | 15.33 s | 2 |
| [Plus overlapping the coarse value GEMM](k2-prefill-buffer-pv-v1/amd-k2-pv-overlap-speed.json) | 15.26 s | 4 |
| [Tiled QK, aligned selector, short MMA and coarse overlap on node 2](k2-prefill-combined-path-v1/amd-k2-overlap-final-control.json) | 15.38 s | 2 |
| [Also gathering exact KV into contiguous regions](k2-prefill-combined-path-v1/amd-k2-combined-speed.json) | **15.21 s** | 2 |
| [Retained implementation, repeat](k2-prefill-qk-final-v1/amd-k2-final-fp32-64k.json) | **15.24 s** | 2 |

The combined runs used three warmed measurements from the same prompt and produced matching greedy output tokens. The retained-implementation repeat also matched the greedy tokens. The 15.21 s run took 15.204, 15.207 and 15.226 s. It is **1.55 s above 13.66 s** and **1.71 s above the same-round top-eight run**. [Retained launch settings](k2-prefill-qk-final-v1/amd-k2-final-fp32-64k-submission.json) · [Coarse-overlap control on node 4](k2-prefill-buffer-pv-v1/amd-k2-pv-overlap-control.json)

## What changed

- The centroid GEMM groups query tiles in AITER-style order and permutes the workgroups across eight chiplets. It uses BF16 inputs, FP32 accumulation and output, and shared KV storage across query heads. At the 3,547-region shape in a 16K-query replay, the tiled kernel measured **4.24 ms** and the stock GEMM **5.55 ms**. The 2,048-region replay measured **2.46 ms** for the tiled kernel. [Tiled 3,547-region sweep](k2-prefill-cache-qk-v1/amd-k2-matmul-3547.json) · [Stock QK control](k2-prefill-opus-qk-v1/amd-k2-opus-qk-3547.json) · [2,048 regions](k2-prefill-cache-qk-v1/amd-k2-matmul-2048.json)
- Pad the log-count metadata to the score stride so the selector can read aligned vectors. Empty and padded regions retain their existing masks. For short exact frontiers, the merge skips the zero upper half of the value MMA while retaining the full-width FP32 mass reduction.
- The closed-summary value GEMM starts on a separate stream after selection. Exact-route packing and exact-leaf attention can proceed while it runs. The final merge waits on both branches.
- Gather BF16 KV once per used region into aligned contiguous storage. Grouped query tiles reuse that storage and apply the same key-validity masks. In the instrumented combined run, grouped exact attention took **1.12 s** and the gather **0.016 s** across 192 remote attention calls. The preceding instrumented configuration spent **1.32 s** in grouped exact attention. [Combined device profile](k2-prefill-combined-validation-v1/amd-k2-combined-profile.json) · [Previous device profile](k2-prefill-align-selector-v1/amd-k2-current-profile.json)

The combined profile sums **1.72 s** in threshold selection, **1.12 s** in grouped exact leaves, **1.00 s** in merge and **0.97 s** in centroid QK. GPU activity durations can overlap. A faster isolated QK therefore translates only partly into lower end-to-end prefill time.

## Other backend and scheduling trials

| Trial | Measurement |
|---|---|
| [hipBLAS QK with short MMA](k2-prefill-tuned-qk-v1/amd-k2-hipblas-speed.json) | 15.42 s end to end. The earlier hipBLAS audit found 15 changed route rows across 3.15 million. |
| [AITER Opus FP32 QK](k2-prefill-buffer-pv-v1/amd-k2-opus-graph-workspace.json) | 11.39 ms at 3,547 regions, versus 5.71 ms for stock BLAS in the same replay. |
| [GEMM/selector stripes](k2-prefill-cache-qk-v1/amd-k2-cache-3547.json) | Best tested striped schedule took 17.06 ms at 3,547 regions, versus 16.34 ms for the unstriped stock replay. |
| [Fused QK and selection with interleaved MFMA keys](k2-prefill-packed-fusion-v2/amd-k2-fusion-vector-mfma.json) | 5.90 ms at 4,096 queries, versus 4.18 ms for the separate kernels. The best fused kernel still spilled registers and produced different coarse values. |
| [8K](k2-prefill-cu-schedule-v1/amd-k2-8k-routing-tile.json) and [4K](k2-prefill-cu-schedule-v1/amd-k2-4k-routing-tile.json) internal query tiles | 15.52 s and 15.93 s, respectively. Both used the 16K exact prefix. |
| [Priority local stream](k2-prefill-meta-overlap-v1/amd-k2-local-priority-speed.json) and [restricted CU affinity](k2-prefill-cu-schedule-v1/amd-k2-local-cu152-interleave.json) | 17.10 s and 16.59 s, respectively. |
| [Physical-page copy](k2-prefill-materialize-kv-v1/amd-k2-materialize-pages-speed.json) and [two-stage region loads](k2-prefill-region-kv-v1/amd-k2-region-kv-stages2-speed.json) | 15.65 s and 15.70 s, respectively. |

## Correctness

The tiled QK, aligned selector, short MMA, coarse overlap and contiguous region storage together preserved every route ID, coarse BF16 value, coarse log partition and final remote BF16 output bit across **3,145,728 query-head rows** on saved 16K query blocks from layers 0, 32 and 63. The independent frontier check passed 14 cases. [Layer 0](k2-prefill-qk-final-v1/amd-k2-final-bitwise-00.json) · [Layer 32](k2-prefill-qk-final-v1/amd-k2-final-bitwise-32.json) · [Layer 63](k2-prefill-qk-final-v1/amd-k2-final-bitwise-63.json) · [Independent frontier checks](k2-prefill-qk-final-v1/amd-k2-final-fp32-frontier.json)

The combined model path scored **16/16** over eight saved 64K prompts each for RULER multikey-2 and multikey-3, with c=16 decode. All tokenized prompts and prediction strings matched the previous c=16 quality run. [Multikey-2](k2-prefill-combined-validation-v1/amd-k2-combined-ruler-mk2.json) · [Multikey-3](k2-prefill-combined-validation-v1/amd-k2-combined-ruler-mk3.json)

The [AITER-stage follow-up](K2_PREFILL_AITER_STAGES.md) compares direct variable-length exact leaves, wave-level selector and merge trials, and interleaved full-model timing.

The [separate-merge and fused-consumer trials](K2_PREFILL_MERGE_AND_CONSUMER.md) also test an indexed AITER Triton adapter on saved 64K states.
