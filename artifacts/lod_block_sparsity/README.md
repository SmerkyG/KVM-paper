# Qwen3.5-0.8B LOD block-list sparsity

Measured on 2026-08-21 with `Qwen/Qwen3.5-0.8B`, two deterministic ProLong
documents per length, and the six full-attention layers (3, 7, 11, 15, 19,
23). The model has 8 query heads, 2 KV heads, and GQA ratio 4. The current
`qk_norm_aware` state-clustering policy resolves to coherence routing for this
Q/K-normalized model.

The model continued to use its normal top-3 prefill detail path. A separate
observational route selected the top 8 state centroids for each query. For a
chronological KV block width `B`, every centroid's block list contains all
blocks holding at least one of its owned leaves. Reported density is the union
of those lists divided by all remote blocks at the final (largest) prefill
route. Values average both samples and all six attention layers.

## 128-token KV blocks

| Context | Remote tokens | 128-blocks | 1 row / Q head | 1 row / GQA union | 16 rows / Q head | 16 rows / GQA union |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 4,352 | 34 | 67.8% | 86.7% | 97.2% | 99.4% |
| 16K | 12,544 | 98 | 53.2% | 75.1% | 91.2% | 97.0% |
| 32K | 28,928 | 226 | 43.1% | 62.7% | 79.9% | 90.0% |
| 64K | 61,696 | 482 | 40.9% | 61.2% | 79.0% | 87.9% |

Ordinary 128-wide block sparsity is therefore weak for a 16-row query tile.
Keeping query heads separate helps, but the average tile still opens about 381
of 482 blocks at 64K.

## Fast-fail width at 64K, 16 query rows

| KV mask width | Separate Q heads | OR over 4 GQA heads |
| ---: | ---: | ---: |
| 16 | 45.9% | 62.2% |
| 32 | 59.4% | 73.6% |
| 64 | 70.5% | 81.9% |
| 128 | 79.0% | 87.9% |

This supports a large outer scheduling group with an inexpensive 16-token
inner mask more than a conventional 128-wide block-sparse mask. It also argues
for keeping query heads separate: GQA union loses 16.3 percentage points of
16-wide sparsity at 64K.

## Layer dependence at 64K, 16 query rows

| Layer | 128 / Q head | 128 / GQA | 16 / Q head | 16 / GQA |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 97.0% | 99.7% | 70.1% | 85.2% |
| 7 | 92.5% | 97.5% | 49.2% | 67.0% |
| 11 | 54.2% | 69.7% | 20.9% | 33.2% |
| 15 | 48.1% | 66.2% | 18.3% | 30.0% |
| 19 | 86.1% | 94.6% | 53.0% | 70.5% |
| 23 | 96.0% | 99.4% | 63.9% | 87.5% |

The middle layers have useful sparsity; the earliest and latest attention
layers are close to dense. A kernel should permit a dense fallback rather than
pay block-sparse scheduling overhead on every layer.

Across all 64K prefill route calls, only about 1.7--2.1% of the score cells in
scheduled 128-wide query-head tiles belong to the query row's selected owners.
Thus selecting a block cannot silently make all of its leaves exact without
also correcting the low-LOD remainder. An owner predicate (or equivalent
coarse-mass subtraction) remains necessary.

Raw JSON includes block widths 16, 32, 64, and 128; query tile widths 1, 4, 8,
and 16; per-layer results; distribution summaries; and useful-cell density.
