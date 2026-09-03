# Prefill expert-dispatch transfer

This experiment tests whether the low-row/direct-routing idea from decode can
remove the global route sort before expert-major exact-leaf prefill. The
candidate counts routes into exact expert buckets, prefix-sums the counts, and
scatters route-row IDs into the existing expert-major attention layout. It does
not change routing, selected leaves, attention math, or the route/LSE merge.

## Correctness

Direct and sorted dispatch produced bit-identical BF16 outputs and LSEs in all
synthetic geometry tests. This includes a test with every second query's final
route set to `-1`. Qwen3.5-0.8B also scored 8/8 on chat-formatted 64K NIAH-S3
with the direct path active.

## Uniform-route microbenchmarks

The first synthetic benchmark spread routes evenly across all 4,352 state
slots. Production prefill opens three centroids per query, so the relevant
results use top-three rather than top-eight.

| geometry | batch | sorted leaf stage | direct leaf stage | change |
|---|---:|---:|---:|---:|
| Qwen D256/KV2/GQA4 | 1 | 0.852 ms | 0.748 ms | 12.2% faster |
| Qwen D256/KV2/GQA4 | 8 | 3.950 ms | 3.627 ms | 8.2% faster |
| Muse D128/KV2/GQA16 | 8 | 5.668 ms | 5.238 ms | 7.6% faster |
| OLMo D128/KV8/GQA5 | 8 | 8.064 ms | 7.768 ms | 3.7% faster |
| Gemma D512/KV2/GQA8 | 8 | 13.727 ms | 13.717 ms | neutral |

For Qwen B8, dispatch itself fell from 0.581 to 0.280 ms. These measurements
were useful kernel checks, but the uniform route population was not a valid
surrogate for Muse.

## Production vLLM A/B

The matched serving runs use eight distinct real ProLong documents, 64K cold
prefill, 16K chunking for Qwen, 128K maximum batched tokens for Muse, and two
decode tokens so prefill dominates the result.

| model | sorted | direct | result |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 4.168 s | 4.128 s | direct 0.96% faster |
| Muse-Glimmer-30B | 47.849 s | 48.594 s | direct 1.56% slower |

The earlier 49.407-second Muse record is not used for this conclusion. A fresh
same-node control produced the stable 47.849-second median above.

## Why Muse reverses

GPU-event profiles over the actual vLLM route tensors resolve the apparent
contradiction. Without branch overlap, the per-layer/per-4K-chunk times are:

| phase | sorted | direct | delta |
|---|---:|---:|---:|
| expert dispatch | 0.499 ms | 2.039 ms | +1.541 ms |
| exact-leaf MMA | 1.229 ms | 1.202 ms | -0.027 ms |
| complete exact-leaf branch | 1.980 ms | 3.497 ms | +1.517 ms |
| complete two-level attention | 11.214 ms | 12.905 ms | +1.691 ms |

The direct count and scatter contribute 0.782 and 1.086 ms per call. Muse's
semantic routes repeatedly target a small set of centroids, so many lanes
atomically update the same counters and cursors. The radix sort is insensitive
to that concentration and takes only 0.182 ms. The direct scatter also loses
the query locality retained by the radix-sort output.

A controlled route-working-set sweep reproduces the crossover while retaining
the full 4,352-slot allocation:

| routed slots | sorted dispatch | direct dispatch | leaf-stage result |
|---:|---:|---:|---:|
| 64 | 0.502 ms | 0.949 ms | direct 7.9% slower |
| 256 | 0.573 ms | 0.432 ms | direct 1.3% faster |
| 1,024 | 0.505 ms | 0.282 ms | direct 2.9% faster |
| 4,352 | 0.576 ms | 0.256 ms | direct 5.0% faster |

Holding routes fixed for adjacent queries makes the atomic path still worse,
matching production's temporal concentration. A block-local and GQA-local
duplicate-coalescing variant reduced atomic contention but remained slower
than radix sorting, so it was removed.

## Overlap scheduling check

Muse overlaps coarse attention with the exact-leaf branch. Moving the coarse
launch later does make sorting cheaper, but increases direct overlap between
coarse attention and leaf MMA. Per-layer/per-chunk two-level totals were:

| coarse launch | two-level total |
|---|---:|
| before expert dispatch (current) | 10.792 ms |
| after radix sort | 10.829 ms |
| after all dispatch metadata | 11.187 ms |
| no coarse/leaf overlap | 11.214 ms |

The existing early launch is therefore retained.

## Resulting policy

Direct expert buckets are automatic only for measured two-level BF16
`D=256, GQA=4, KV-heads=2` expert prefill (Qwen3.5-0.8B geometry). Other
geometries retain radix sorting. `VLLM_LOD_PREFILL_DIRECT_EXPERT_BUCKETS=0/1`
can override the policy for diagnostics. INT8 and compound tiny/split expert
routing remain on their existing dispatch implementations.
