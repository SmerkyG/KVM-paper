# BF16 matrix-native flat two-tier prefill

This experiment makes flat two-tier exact-leaf attention matrix-native before
attempting INT8. Queries are grouped by `(KV row, routed centroid)` and each
group executes BF16 `tl.dot` operations for both QK and PV. The hot operations
are therefore directly replaceable by an INT8 MMA implementation later.

The previous expert-major kernel used a rectangular launch grid based on the
largest query group. Every smaller expert launched the same number of query
tiles, and even empty tiles traversed that expert's leaves. The new launch is
ragged: it materializes only `ceil(M_expert / BLOCK_M)` program descriptors and
derives the local tile index from prefix offsets.

All timings use `Qwen/Qwen3.5-0.8B`, BF16, batch 8, prefill top-k 3, and the
shared cluster GPU. Unless noted, the state schedule is `8 sqrt(T)`.

## Exact-leaf branch

| Context | Query-major | Grouped BF16 matmul | Speedup |
|---:|---:|---:|---:|
| 8K | 22.42 ms | 14.73 ms | 1.52x |
| 32K | 288.69 ms | 77.59 ms | 3.72x |
| 64K | 966.13 ms | 188.80 ms | 5.12x |

At 32K with the standard `16 sqrt(T)` schedule, query-major takes 203.78 ms
and grouped BF16 takes 76.32 ms, a 2.67x speedup.

The selected expert configuration is `BLOCK_M=16`, `BLOCK_N=16` at 8K and
`BLOCK_M=16`, `BLOCK_N=32` at 32K/64K, with two warps.

## Uninstrumented end-to-end prefill versus full attention

These are three-run means from `profile_qwen35_prefill_total.py`, including the
same accelerated Qwen GDN layers in both modes.

| Context | Native full attention | Flat grouped-BF16 LOD | LOD speedup |
|---:|---:|---:|---:|
| 32K | 2026.84 ms | 1887.16 ms | 1.07x |
| 64K | 5973.87 ms | 4215.32 ms | 1.42x |

The 64K full-attention control was rerun alone and was stable at 5971.6-5977.8
ms. Flat grouped LOD was stable at 4201.3-4241.6 ms, a 29.4% latency reduction.
Peak allocated memory was 35.80 GiB for full attention and 44.34 GiB for this
uncompressed BF16 flat-LOD prototype; this experiment targets speed rather than
cache compression.

## Instrumented whole-prefill phase control

| Context / schedule | Query-major flat | Grouped BF16 flat | Recursive three-tier |
|---:|---:|---:|---:|
| 32K / 8 sqrt(T) | 2303.48 ms | 2120.20 ms | - |
| 64K / 8 sqrt(T) | 5679.05 ms | 4874.70 ms | 5042.36 ms |
| 32K / 16 sqrt(T) | 2830.00 ms | 2709.50 ms | - |

Thus the simplified flat method is 14.2% lower latency than query-major at
64K and is 3.3% faster than the more complicated recursive three-tier method.

## Why the fix works

At 32K and `8 sqrt(T)`:

- Old rectangular programs: 139,623,595
- Ragged programs: 2,576,809 (1.85% as many)
- Old useful QK tile fraction: 2.30%
- Ragged useful QK tile fraction: 70.49%
- Mean routed query rows per active expert: 72.84

At `16 sqrt(T)`, ragged QK tile utilization is still 54.95%, versus 1.14% for
the old rectangular grid.

## Correctness and INT8 ceiling

The complete paged-attention verification passes. Against the packed reference,
the grouped kernel's maximum output error is 0.0009766 and maximum LSE error is
zero in the targeted test.

At 64K, the BF16 QK/PV kernel itself is 103.03 ms of the 188.80 ms exact-leaf
branch. An ideal 2x INT8 speedup of only this kernel would make the branch about
137.28 ms (1.38x faster), not 2x, because sorting/dispatch and route reduction
remain. It would save only about 1% of the complete 0.8B-model prefill. The
route/coarse LOD kernels already use BF16 `tl.dot` as well, so a later INT8 pass
must cover those matmuls—not only exact leaves—to materially change end-to-end
prefill latency.

Rejected alternatives in this experiment:

- Counting-sort dispatch was slower than GPU radix sort.
- A one-program-per-row fused route reduction was slower than the PyTorch
  softmax/weighted-sum/logsumexp sequence.
- The natural `M x N` by `N x D` PV expression produced incorrect results with
  the current Triton/ROCm layout. The existing transposed `D x N` by `N x M`
  operation is still a real tensor-core matmul and verified correctly.

## Unmasked query-tile centroid unions

An additional diagnostic tested replacing expert grouping with a fixed query
tile. Every query in the tile would attend the unmasked union of all centroids
selected by the tile's queries. The table reports the mean number of unique
centroids in that union and the resulting leaf-weighted attention work relative
to the original per-query top-3 sets.

| Context | Tile | Mean union centroids | Leaf-work inflation |
|---:|---:|---:|---:|
| 8K | 2 | 5.17 | 1.65x |
| 8K | 4 | 8.94 | 2.73x |
| 8K | 8 | 15.25 | 4.46x |
| 8K | 16 | 25.38 | 7.02x |
| 32K | 2 | 5.11 | 1.59x |
| 32K | 4 | 8.75 | 2.53x |
| 32K | 8 | 14.89 | 3.96x |
| 32K | 16 | 24.88 | 6.02x |
| 64K | 2 | 4.99 | 1.57x |
| 64K | 4 | 8.41 | 2.50x |
| 64K | 8 | 14.18 | 3.96x |
| 64K | 16 | 23.77 | 6.22x |

Only tile 2 remains below the approximately 1.8-2x BF16 compute headroom, but
an `M=2` attention does not provide the matrix occupancy this design was meant
to recover. Tile 4 and larger perform too much extra leaf work before accounting
for union construction or their lower M utilization. The `16 sqrt(T)` schedule
does not improve the result: at 32K its tile-4 union has 9.16 centroids and
2.54x leaf-work inflation.
