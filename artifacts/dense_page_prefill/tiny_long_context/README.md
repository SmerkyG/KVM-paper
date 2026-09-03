# Small-expert two-level prefill, batch 8

Qwen3.5-0.8B, BF16 virtual leaf storage, top-8 two-level routing,
`16*sqrt(T)` state schedule, gfx942. Times include all six full-attention
layers. End-to-end results are warm prefill measurements.

## Exact-leaf phase

| Context | Original expert | Tuned generic expert | Adaptive small expert | Change vs original |
|---:|---:|---:|---:|---:|
| 8K | 39.50 ms | 23.34 ms | 17.94 ms | -54.6% |
| 16K | 90.92 ms | 53.63 ms | 51.52 ms | -43.3% |
| 32K | 207.77 ms | 129.25 ms | 119.92 ms | -42.3% |
| 64K | 476.41 ms | 314.29 ms | 295.92 ms | -37.9% |
| 128K | 2223.23 ms | 1730.14 ms | 1727.98 ms | -22.3% |

The original path uses `M16/N32`, four attention warps, and four route-reduce
warps. Tuned generic uses two attention warps and one route-reduce warp.
Adaptive small expert additionally specializes exact leaf counts 1--8 until
the accumulated leaf context reaches 64K, then uses tuned generic attention.

At 128K, N<=8 accounts for only 16.7% of ragged programs, down from 30.8% at
64K. Always specializing through 128K was therefore neutral/slightly slower:
1734.01 ms versus 1730.14 ms for tuned generic. Cutoffs of 64K and 96K measured
1727.98 ms and 1736.14 ms, respectively.

## End-to-end optimized prefill

| Context | Mean | Throughput | Peak memory | Persistent LOD cache |
|---:|---:|---:|---:|---:|
| 8K | 426.04 ms | 153,827 tok/s | 7.10 GiB | 2.48 GiB |
| 16K | 993.70 ms | 131,903 tok/s | 12.36 GiB | 4.45 GiB |
| 32K | 2374.12 ms | 110,418 tok/s | 22.58 GiB | 8.06 GiB |
| 64K | 5705.03 ms | 91,899 tok/s | 42.95 GiB | 15.23 GiB |
| 128K | 15039.07 ms | 69,723 tok/s | 83.36 GiB | 29.22 GiB |

The final 128K result uses five repeats: 15110.61, 15023.09, 15021.74,
15014.75, and 15025.19 ms. Its median is 15023.09 ms. The original expert
path's three-repeat median was 15463.43 ms, so the robust end-to-end gain at
128K is 2.85%.

## Correctness

- NIAH-S3 8K: 8/8 exact.
- Integrated batch-8 prefill: finite logits and identical raw/specialized top-1
  predictions for all eight rows.
- Raw/specialized logit MAE: 0.02457; RMSE: 0.03247. This is similar to the
  prior N<=4 specialization and comes from a different BF16 reduction order.

## Findings and next target

- One warp is the correct shape for merging eight route outputs. At 64K it
  reduced route-merge time from 152.18 ms to 26.75 ms.
- Two warps are best for the general `M16/N32/D128` leaf kernel. Larger M/N
  tiles regressed, one warp underfilled the matrix work, and `M8` was
  catastrophically slow.
- Extending the scalar small-N kernel through N=16 regressed. More small-expert
  specialization is not the long-context target.
- At 128K, the general N>8 kernel is about 1.58 s of the 1.73 s exact-leaf
  phase. Dispatch is about 92 ms and route reduction about 52 ms. The next
  worthwhile optimization is split-N execution for the longest posting lists:
  process independent leaf ranges in parallel, then merge their output/LSE.
  This targets tail-dominant large experts without increasing persistent state.

The detailed phase profiler now constructs contexts longer than 64K by
concatenating deterministic 64K ProLong pieces, matching the end-to-end
profiler. Previously it searched the 64K dataset for a nonexistent 128K
document.
