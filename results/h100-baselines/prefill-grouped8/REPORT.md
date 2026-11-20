# H100 threshold routing through the original grouped leaf kernel

BF16 Llama-3.1-8B-Instruct, batch one, two-tier entry-16 prefill, unchanged 16,384-token exact first chunk. A threshold candidate is opened only if it also fits the original strict entry-work bound; we keep at most the first eight surviving candidates in region order. This is not top-8 by score. All other regions remain coarse. The cap is a speed experiment with a different frontier from uncapped threshold.

The grouped variant calls the same `paged_leaf_attention` region-grouped kernel as top-8; an additional variant produces its per-head route counts and offsets during selection, as top-8 does. No per-query GPU-to-CPU count readback is used. Prefill times are medians of three runs after one warmup.

| Configuration | 32K full prefill | 64K full prefill |
|---|---:|---:|
| Original top-8 routing and grouped kernel | 1.656 s | 4.193 s |
| Capped threshold, per-query indexed kernel | 2.133 s | 5.829 s |
| Capped threshold, grouped kernel | 2.137 s | 5.935 s |
| Capped threshold, grouped kernel with fused counts | 2.326 s | 6.732 s |

The three capped-backend rows use the same source snapshot and prompt; top-8 is a different routing rule and is shown as context. Results do not establish equivalent RULER quality.

On one 32K prompt, auditing the first 64 remote queries per head and layer (65,536 query heads), 97.49% had more than eight qualifying regions. Average opened regions fell from 77.47 to 7.91, and average opened entries from 254.59 to 30.82. 0 capped selections differed from the reference.

Grouped versus indexed, using exactly the same query, selected regions, and KV state: 16,384 sampled query heads, maximum merged-output absolute difference 0.00781250, mean 0.00000311.

Grouped with fused counts versus indexed, using exactly the same query, selected regions, and KV state: 16,384 sampled query heads, maximum merged-output absolute difference 0.00781250, mean 0.00000311.

The capped backends matched the reference selection on 49,152 sampled query heads across three RULER examples per backend. Their first-response token sequences matched on 3 of 3 examples; their full fixed-length sequences matched on 0 of 3 (the runs continue generating after end-of-turn).

32K synchronized diagnostic timings, summed across layers:

| Backend | Score generation | Select + coarse | Index expansion | Exact leaves |
|---|---:|---:|---:|---:|
| Per-query | 0.173 s | 0.517 s | 0.082 s | 0.268 s |
| Grouped | 0.178 s | 0.495 s | 0.000 s | 0.632 s |
| Grouped + fused counts | 0.180 s | 0.706 s | 0.000 s | 0.599 s |

These stage measurements add synchronization and instrumentation overhead; use the uninstrumented full-prefill table for speed comparisons. Grouped exact leaves include route grouping; the per-query path has a separate index-expansion stage.

64K synchronized diagnostic timings, summed across layers:

| Backend | Score generation | Select + coarse | Index expansion | Exact leaves |
|---|---:|---:|---:|---:|
| Per-query | 0.635 s | 2.041 s | 0.255 s | 0.822 s |
| Grouped | 0.647 s | 1.939 s | 0.000 s | 2.068 s |
| Grouped + fused counts | 0.649 s | 2.806 s | 0.000 s | 1.967 s |

These stage measurements add synchronization and instrumentation overhead; use the uninstrumented full-prefill table for speed comparisons. Grouped exact leaves include route grouping; the per-query path has a separate index-expansion stage.

Complete source hashes, commands, logs, and raw outputs are retained in the `provenance` directory. The earlier uncapped threshold result is in the [one-pass report](../prefill-onepass/REPORT.md).
