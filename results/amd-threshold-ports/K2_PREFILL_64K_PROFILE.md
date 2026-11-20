# K2 Horizon 64K attention-only prefill profile

The comparison uses K2 Horizon 32B FP8 weights, BF16 KV, two-tier LoD, one MI325X, and one batch-one 65,536-token ProLong prompt. The first 16,384 tokens stay exact. The three remote chunks retain normalized equal-per-region c=16, the 1,024-entry region eligibility limit, and every eligible selection above the threshold. The FP8 model projection kernels are the original vLLM kernels in all timing and quality rows below.

## Why QK reuse only saved about 0.16 s

The streaming backend fuses its second QK with threshold selection and the closed-summary value accumulation. The single-QK backend replaces that fused work with a separately timed selector and value GEMM. The table uses GPU events over all 64 layers and all three remote chunks. Each row is a separate warmed full-model run with kernel-level instrumentation. This instrumentation adds a small amount of runtime to the outer event intervals.

| Routing operation | Streaming | Saved FP32 QK |
|---|---:|---:|
| First QK and proxy partials, or saved FP32 QK GEMM | 0.958 s | 1.090 s |
| Cutoff reduction | 0.003 s | Fused into selector |
| Second QK, selection and closed accumulation | 2.125 s | |
| Select regions and write closed probabilities | | 1.716 s |
| Closed-summary value GEMM | | 0.316 s |
| Merge coarse partitions | 0.164 s | |
| Compact selected regions | 0.143 s | 0.168 s |
| Total route and coarse interval | **3.485 s** | **3.384 s** |

The first saved-QK GEMM takes 0.13 s longer than the streaming proxy pass. Selection and the value GEMM take 2.03 s of the 2.12 s consumed by the original second pass. The replacement removes the coarse-partition merge. The previous lower-overhead comparison observed a 3.48 to 3.31 s change. Fresh stage-only timings without individual route kernel events observed 3.48 to 3.29 s. [Streaming kernel events](k2-prefill-64k-route-events-v1/amd-k2-64k-events-stream.json) · [Saved-QK kernel events](k2-prefill-64k-route-events-v1/amd-k2-64k-events-reuse.json) · [Streaming stage profile](k2-prefill-64k-kernels-v1/amd-k2-64k-device-stream.json) · [Saved-QK stage profile](k2-prefill-64k-kernels-v1/amd-k2-64k-device-reuse.json)

| Remote region slots | Streaming route and coarse | Saved FP32 QK |
|---:|---:|---:|
| 2,048 | 0.796 s | 0.851 s |
| 2,896 | 1.134 s | 1.155 s |
| 3,547 | 1.555 s | 1.379 s |

The gain comes from the last remote chunk. The materialized path writes and reads roughly 6.9 TB of logical FP32 score and BF16 probability data across the request. That estimate uses the query-head count, three region strides, both matrix element widths, and the read/write count. Device counters would be needed to measure physical HBM traffic.

## Largest kernels on the original optimized path

| GPU activity summed across the request | Time |
|---|---:|
| FP8 model projections, common to both attention settings | 7.35 s |
| BF16 exact-local attention | 2.19 s |
| Threshold selector | 1.71 s |
| FP32 QK GEMMs | 1.62 s |
| Grouped exact-leaf attention | 1.32 s |
| Fused exact-leaf reduction and merge | 1.04 s |
| BF16 closed-summary value projection | 0.31 s |

Exact-local attention overlaps some remote work, so summed kernel durations and event intervals can overlap. The complete device trace above comes from the optimized configuration. The route event tables use separate runs without local overlap for cleaner attribution. [Complete kernel trace](k2-prefill-64k-kernels-v1/amd-k2-64k-device-optimized.json)

In the final wave-selector trace, selection falls to **1.40 s**. The remaining largest threshold-side GPU durations are **1.58 s** for FP32 QK, **1.30 s** for grouped exact leaves, and **1.03 s** for the fused merge. Exact-local overlap contributes to the longer measured QK duration compared with the serial routing table. [Final device and stage profile](k2-prefill-attention-only-v2/amd-k2-wave-final-profile.json)

## Attention-only changes

The explicit AMD wave layout vectorizes the score and weight traffic while keeping the FP32 cutoff and the same normalized coarse weights. It reduced selector time from 3.29 to 2.77 ms on the middle-layer replay at 4,096 queries. The independent frontier check passed all 14 cases. [Original replay](k2-select-bitset-v1/amd-k2-select-bitset-2-8.json) · [Wave-layout replay](k2-prefill-gluon-v1/amd-k2-gluon-2-8-2-4.json) · [Frontier validation](k2-prefill-gluon-v2/amd-k2-gluon-frontier.json)

| Attention setting | 64K prefill median |
|---|---:|
| [Native top-eight](k2-prefill-final-v2/amd-k2-final-top8-speed.json) | 13.42 s |
| [Previous optimized c=16](k2-prefill-final-v2/amd-k2-final-16k-speed.json) | 16.17 s |
| [c=16 with the AMD wave selector](k2-prefill-gluon-v2/amd-k2-gluon-64k-speed.json) | **15.88 s** |
| [Start exact-local after QK](k2-attention-overlap-v2/amd-k2-overlap-after-qk.json) | 15.97 s |
| [Start exact-local after all routing and coarse work](k2-attention-overlap-v2/amd-k2-overlap-after-route.json) | 16.30 s |

Seconds are the median of three warmed timings on the same tokenized prompt. The selector shrinks the gap to native top-eight by 0.29 s. The AMD wave selector is opt-in.

The matched node-2 rerun used the final source snapshot for all three arms and the same ProLong token IDs. Native top-eight measured **13.44 s**, the previous c=16 path **16.18 s**, and the AMD wave selector **15.87 s**. That narrows the gap to top-eight from **2.74 to 2.43 s** and leaves **1.17 s** to reach the 14.70 s target. The final selector also measured **16.07 s** on node 4. [Matched top-eight](k2-prefill-attention-only-v2/amd-k2-wave-matched-top8.json) · [Matched c=16 control](k2-prefill-attention-only-v2/amd-k2-wave-matched-prior.json) · [Matched wave selector](k2-prefill-attention-only-v2/amd-k2-wave-matched-wave.json) · [Node 4 repeat](k2-prefill-attention-only-v2/amd-k2-wave-final-speed.json)

The wave selector matched prediction text and scores on all eight RULER 64K multikey prompts tested with the 16K exact field and 16K scratch tile. Prompt and answer hashes match the archived c=16 run. [Multikey-2](k2-prefill-attention-only-v1/amd-k2-attention-only-ruler-mk2.json) · [Multikey-3](k2-prefill-attention-only-v1/amd-k2-attention-only-ruler-mk3.json)

Fusing expert counts into bit compaction sped up leaf dispatch by about 0.2 ms on the middle-layer replay but added about 2 ms to routing. The best explicit vector merge tested took 2.02 ms against the existing tensor-core merge's roughly 1.39 ms. Larger exact-leaf tiles also took more time than the current 128-query, 32-key tile on the three saved layers. [Expert counts](k2-prefill-compact-experts-v1/amd-k2-expert-counts-replay.json) · [Vector merge](k2-prefill-gluon-merge-v1/amd-k2-merge-wave-4-32-4-2-8.json) · [Leaf tile sweep](k2-prefill-leaf-tiles-v1/amd-k2-leaf-tiles-layer32.json)

The [centroid and dispatch follow-up](K2_PREFILL_CENTROID_AND_DISPATCH.md) gives the newer attention-only timings, the 64K retrieval checks, and the direct-fusion results.
