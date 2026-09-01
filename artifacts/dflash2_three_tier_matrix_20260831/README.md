# Optimized recursive DFlash2 matrix

This is the matched post-optimization speed panel for recursive three-tier
BF16 LOD target verification on `Qwen/Qwen3.8-27B-FP8`, with
`z-lab/Qwen3.8-27B-DFlash2` drafting seven tokens.  It covers TP1 and TP4 at
batch one and batch eight on MI325X.  Prompts are distinct, non-repeated real
ProLong documents, prefill uses a 16,384-token aggregate scheduler budget,
and sampling is greedy.  Every row has one warmup followed by three measured
repetitions and emits 256 tokens.

The device audit in every arm records flattened speculative execution,
paired/shared route execution, and fused shared-local execution as configured
and observed.  All arms use three LOD tiers and BF16 leaf storage.  The TP1/B8
arm used vLLM's ordinary local weight loader after its first placement landed
on a GPU outside that node's weight-cache daemon set; model loading occurs
before measurement and does not affect the timings.

## Target-verifier cycles

Milliseconds are per complete target-verifier cycle and do not depend on how
many proposed tokens were accepted.  This is the primary kernel comparison.

| context | TP1/B1 | TP1/B8 | TP4/B1 | TP4/B8 |
|---:|---:|---:|---:|---:|
| 8K | 39.839 | 53.613 | 25.313 | 32.154 |
| 16K | 39.332 | 56.206 | 25.073 | 31.583 |
| 32K | 39.767 | 57.150 | 25.483 | 32.248 |
| 64K | 39.707 | 52.968 | 25.283 | 31.389 |
| 128K | 40.555 | 50.219 | 25.452 | 30.334 |

The optimized cycle remains essentially context-independent.  TP4 reduces
the absolute target cycle to about 25 ms at B1 and 30--32 ms at B8.  Relative
to the previously published recursive DFlash2 implementation, the long rows
improve as follows:

| geometry | old 64K | new 64K | improvement | old 128K | new 128K | improvement |
|---|---:|---:|---:|---:|---:|---:|
| TP1/B1 | 41.195 | 39.707 | 3.6% | 42.085 | 40.555 | 3.6% |
| TP1/B8 | 62.444 | 52.968 | **15.2%** | 58.765 | 50.219 | **14.5%** |
| TP4/B1 | 26.284 | 25.283 | 3.8% | 26.255 | 25.452 | 3.1% |
| TP4/B8 | 33.107 | 31.389 | 5.2% | 31.977 | 30.334 | 5.1% |

The largest gain is TP1/B8, where pairing speculative positions removes the
most duplicated state/local work.  TP4 already distributes the 24 query heads
as six per rank, so route/local work is a smaller fraction of its cycle and
the remaining gain is correspondingly smaller.

## End-to-end decode

Milliseconds are per emitted token at B1 and per emitted batch step at B8.
They include DFlash acceptance behavior, so non-monotonic context trends are
trajectory effects rather than target-kernel scaling.

| context | TP1/B1 | TP1/B8 | TP4/B1 | TP4/B8 |
|---:|---:|---:|---:|---:|
| 8K | 16.277 | 23.757 | 10.618 | 13.600 |
| 16K | 9.562 | 22.919 | 5.604 | 15.106 |
| 32K | 14.194 | 22.415 | 8.994 | 12.523 |
| 64K | 12.769 | 23.274 | 8.908 | 12.316 |
| 128K | 10.612 | 21.434 | 9.176 | 12.832 |

The current prompts have identical SHA-256 hashes to the historical full
attention controls.  At the shared 64K and 128K points:

| geometry | 64K full | 64K LOD | speedup | 128K full | 128K LOD | speedup |
|---|---:|---:|---:|---:|---:|---:|
| TP1/B1 | 19.316 | 12.769 | **1.51x** | 22.054 | 10.612 | **2.08x** |
| TP1/B8 | 42.496 | 23.274 | **1.83x** | 54.938 | 21.434 | **2.56x** |
| TP4/B1 | 9.777 | 8.908 | **1.10x** | 11.702 | 9.176 | **1.28x** |
| TP4/B8 | 17.255 | 12.316 | **1.40x** | 20.888 | 12.832 | **1.63x** |

These are valid serving comparisons, but verifier-cycle latency should be
used to attribute the kernel optimization because floating-point regrouping
can change the greedy trajectory and DFlash acceptance.

## Prefill

Seconds are complete batch prefill medians.  The current automatic TP1 policy
uses complete selected centroid experts through 64K and residual-page
selection for a 128K request.  The current TP4 per-rank geometry uses the
recursive residual-page path throughout.

| context | TP1/B1 | TP1/B8 | TP4/B1 | TP4/B8 |
|---:|---:|---:|---:|---:|
| 8K | 1.004 | 8.495 | 0.476 | 3.971 |
| 16K | 2.068 | 17.135 | 1.025 | 8.293 |
| 32K | 4.186 | 34.872 | 2.108 | 17.351 |
| 64K | 8.603 | 72.087 | 4.425 | 36.758 |
| 128K | 18.485 | 155.766 | 9.495 | 79.532 |

Against the hash-matched historical full controls, prefill speedup is
1.64x/2.31x for TP1/B1, 1.57x/2.21x for TP1/B8, 1.21x/1.54x for TP4/B1, and
1.19x/1.51x for TP4/B8 at 64K/128K respectively.

## Raw records

- `lod3_opt_tp1_b1_8k128k_r3_d256.json` (cluster 12303)
- `lod3_opt_tp1_b8_8k128k_r3_d256.json` (cluster 12307)
- `lod3_opt_tp4_b1_8k128k_r3_d256.json` (cluster 12305)
- `lod3_opt_tp4_b8_8k128k_r3_d256.json` (cluster 12306)
