# K2 Horizon matrix-free threshold prefill on MI325X

The experimental two-tier region-threshold path avoids allocating a dense query-by-region score matrix or probability matrix. The first pass streams the normalized routing logits to obtain per-query proxy mass. The second pass recomputes each tile of QK, applies the same strict c=16 threshold, and accumulates the closed-region numerator and denominator. A bitmap and prefix scan produce the same ascending-region format consumed by the existing exact leaf kernel. The 16,384-token initial exact field, oversized-region eligibility limit, exact leaf attention and cache construction use the existing settings.

## Fixed-query replay

Replay uses the same 4,096 queries, state with 3,547 region slots and exact KV for both paths at layers 0, 32 and 63 of a saved 64K K2 ProLong prompt. The BLAS control uses aligned GEMMs, compact dispatch, and the two-query selector. Routing workspace measures buffers allocated by the routing and coarse kernels. It excludes the KV cache, model buffers and exact leaf workspace.

| Route | Routing workspace | Routing and coarse, median | Remote attention, median |
|---|---:|---:|---:|
| Aligned materialized path | 5.76 GiB | 5.95 ms | 12.20 ms |
| Matrix-free, 128 queries × 64 regions, 512-region partitions | 1.51 GiB | 6.59 ms | 12.27 ms |
| Aligned materialized path, fresh 1,024-region control | 5.76 GiB | 5.99 ms | 12.29 ms |
| Matrix-free, 128 queries × 64 regions, 1,024-region partitions | 1.13 GiB | 6.14 ms | 11.86 ms |

Both matrix-free settings open the same regions on layers 0 and 32 and change one query-head row on layer 63 out of 786,432 rows across the three layers. The sampled independent PyTorch routing checks found no selection errors beyond the threshold tolerance. The matrix-free coarse projection rounds the running softmax weights to BF16 before the value dot product. It accumulates the numerator, denominator and partition merge in FP32. The change in where the weights round can change output values.

[Three-layer replay, 512-region partitions](k2-prefill-stream-final/amd-k2-stream-final-replay.json) · [Three-layer replay, 1,024-region partitions](k2-prefill-stream-final/amd-k2-stream-final-p1024-replay.json)

At layer 32, 1,024-region partitions cut routing workspace further to 1.14 GiB and took 11.88 ms for remote attention, versus 12.14 ms for the corresponding BLAS control. The earlier 64-query × 128-region tile took 20.00 ms against 12.04 ms with BLAS. Its fused coarse kernel alone took 11.61 ms. Widening the query tile and narrowing the region tile reduced that kernel to about 4.24 ms while the first proxy pass took about 1.62 ms. Retaining a separate BLAS value projection after streaming scores reached 12.49 ms and used 4.19 GiB on its best tested tile.

[1,024-region partition replay](k2-prefill-stream-v2/amd-k2-stream-tune-128-64-1024-4-1.json) · [64-query tile replay](k2-prefill-stream-v1/amd-k2-stream-replay-64-128-512-4-1.json) · [Wider tile trace](k2-prefill-stream-v2/amd-k2-stream-replay-128-64-512-4-1.json) · [Separate BLAS projection](k2-prefill-stream-v2/amd-k2-stream-replay-64-256-512-8-1.json)

## Full-model prefill

First-token latency from scheduling, batch one, K2-Horizon-32B-FP8 with BF16 KV on MI325X. Medians use three repetitions after warmup. Prompt token IDs, scheduler chunk size and the initial exact field agree between runs. The timings include cache updates.

| Prefill path | Internal query tile | 32K | 64K |
|---|---:|---:|---:|
| [Dense](k2-prefill-grouped-v1/amd-k2-prefill-speed-dense.json) | | 6.352 s | 17.292 s |
| [Native top-eight](k2-prefill-aligned-speed-v1/amd-k2-prefill-speed-top8.json) | | 5.919 s | 13.378 s |
| [Aligned materialized threshold](k2-prefill-stream-v2/amd-k2-64k-blas.json) | 8K | 6.867 s | 17.505 s |
| [Aligned materialized threshold](k2-prefill-stream-final/amd-k2-blas-speed-tile16384.json) | 16K | 6.888 s | 17.500 s |
| [Matrix-free, 64-query × 128-region tile](k2-prefill-stream-v2/amd-k2-64k-stream.json) | 8K | 7.274 s | 20.554 s |
| [Matrix-free, 128-query × 64-region tile](k2-prefill-stream-v2/amd-k2-stream-speed-tile8192.json) | 8K | 6.784 s | 17.359 s |
| [Matrix-free, 128-query × 64-region tile](k2-prefill-stream-v2/amd-k2-stream-speed-tile16384.json) | 16K | 6.745 s | 17.156 s |
| [Matrix-free, 1,024-region partitions, initial timing](k2-prefill-stream-v2/amd-k2-stream-speed-p1024.json) | 16K | 6.692 s | 16.996 s |
| [Matrix-free, 1,024-region partitions, final source](k2-prefill-stream-final/amd-k2-stream-final-speed-p1024.json) | 16K | 6.724 s | 17.012 s |

On final source the 16K matrix-free arm used 2.8% less time than the 16K materialized control at 64K. Its three 64K timings were 17.703, 17.012 and 16.993 s. The materialized control measured 17.430, 17.504 and 17.500 s.

The experimental K2 setting uses `LOD_AMD_PREFILL_MATMUL=stream LOD_AMD_PREFILL_GROUP_PACK=0 LOD_AMD_PREFILL_QUERY_TILE=16384 LOD_AMD_PREFILL_STREAM_GEOMETRY=128,64,1024,4,1` over the tuned c=16 configuration with the MMA reducer and packed state max. The geometry specifies query-tile size, region-tile size, partition width, warps and pipeline stages. The 512-region tests use the D128 streaming default geometry. Longer contexts use a region-dependent number of small coarse partitions, and this implementation keeps the uncapped region-threshold rule.

## Verification

The final source passed 14 independent PyTorch tests for default geometry and 14 more with split routing forced by 128-region partitions. The tests cover odd query counts, empty regions, large frontiers and graph replay. The three-layer replay above checked 786,432 query-head rows against the materialized kernel and spot-checked each layer against independent PyTorch. [Default geometry](k2-prefill-stream-final/amd-k2-stream-final-default-check.json) · [Split geometry](k2-prefill-stream-final/amd-k2-stream-final-split-check.json)

The matrix-free 8K arm matched the prediction text and scored 100% on the first four examples each of 64K RULER multikey-2 and multikey-3. Decode kept c=16 and prompt and target hashes agree with the frozen materialized reference. [Reference](k2-prefill-grouped-v1/amd-k2-prefill-ruler-reference.json) · [Multikey-2](k2-prefill-stream-v2/amd-k2-stream-ruler-niah_multikey_2.json) · [Multikey-3](k2-prefill-stream-v2/amd-k2-stream-ruler-niah_multikey_3.json)

The 16K setting matched those same predictions and scores with [512-region multikey-2](k2-prefill-stream-final/amd-k2-stream-final-ruler-niah_multikey_2.json), [512-region multikey-3](k2-prefill-stream-final/amd-k2-stream-final-ruler-niah_multikey_3.json), [1,024-region multikey-2](k2-prefill-stream-final/amd-k2-stream-p1024-ruler-niah_multikey_2.json) and [1,024-region multikey-3](k2-prefill-stream-final/amd-k2-stream-p1024-ruler-niah_multikey_3.json).

The jobs use immutable source snapshots and `uv` through `cluster-run`. Each run's sibling `-submission.json` records the full invocation and cluster log path.
