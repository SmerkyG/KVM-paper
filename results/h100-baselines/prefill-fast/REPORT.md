# H100 threshold prefill with a fused full-coarse producer

BF16 Llama-3.1-8B-Instruct, batch 1, same replayed prompts and unchanged 16,384-token exact first chunk. Full-model prefill includes LoD cache updates. Cache initialization and decode graph capture are excluded. Each timing is the median of three runs after one warmup. All GPU experiments run through the logged H100 cluster-run adapter with uv.

| Configuration | 32K prefill | 64K prefill |
|---|---:|---:|
| Original top-8 grouped | [1.659 s](threshold-fast-v6-top8-32k.json) | [4.195 s](threshold-fast-v6-top8-64k.json) |
| Prior entry-16 pipeline, uncapped, indexed leaves | [2.508 s](threshold-fast-v6-legacy-32k.json) | [7.543 s](threshold-fast-v6-legacy-64k.json) |
| Fast threshold, first eight regions, grouped leaves | [1.721 s](threshold-fast-v6-cap8-32k.json) | [4.417 s](threshold-fast-v6-cap8-64k.json) |
| Fast entry-16, uncapped, indexed leaves | [2.473 s](threshold-fast-v6-direct-32k.json) | [7.277 s](threshold-fast-v6-direct-64k.json) |
| Fast entry-16, uncapped, grouped + exact overflow fallback | [2.850 s](threshold-fast-v6-grouped-32k.json) | [8.074 s](threshold-fast-v6-grouped-64k.json) |

The new uncapped indexed path reduces prefill by 1.4% at 32K and 3.5% at 64K versus the prior entry-16 pipeline in these short runs. It remains substantially slower than top-8. The capped diagnostic is 3.7% slower than top-8 at 32K and 5.3% slower at 64K. The grouped uncapped consumer is slower than the indexed consumer.

The capped frontier retains the first eight qualified regions in region order, not the eight highest scores. It is a diagnostic and is not equivalent to uncapped entry-16. Both new uncapped consumers preserve every region selected by the threshold and strict entry-work bound.

The new producer computes native query–mean-key products once, scales them by inverse query RMS for routing, and accumulates full coarse attention and the routing normalizer together. It stores native scores for a single selection pass. A linear prefix scan compacts selected regions. The capped selector stops after eight winners. The uncapped selector handles sixteen queries together, and its merge reuses saved native scores; a one-bit-per-region mask supports the numerical fallback. The final merge subtracts their coarse contributions and adds exact attention, local attention, and sinks. This uses the top-8 score-reuse simplification and mean preparation; it does not reproduce the older BF16-normalized-query arithmetic exactly.

The fast pipeline uses an 8,192-query internal tile, two per remote prefill chunk. It does not change the exact first chunk or cache update schedule. The grouped consumer uses the existing paged leaf kernel with one extra route-to-query lookup for variable-length route lists. The output workspace reserves 64 routes per query on average. This memory budget is not a routing cap: whole queries beyond it use exact per-query attention on the GPU. The final merge recomputes the closed coarse field on the GPU when subtraction would leave less than 0.0001 of full coarse mass. Both fallbacks count toward the latency.

Validation compares against a reference using the new score arithmetic and the actual native dot products produced by the kernel. It independently checks selection, the strict opened-entry bound, and a PyTorch residual-coarse computation with independently dispatched exact leaves. This validates execution, not RULER quality equivalence to the old routing arithmetic.

| 64K validation | Sampled query heads | Changed region decisions | Outside 2e-5 threshold tolerance | Maximum attention output error |
|---|---:|---:|---:|---:|
| Capped grouped | 49,152 | 0 | 0 | 0.007812 |
| Uncapped indexed | 49,152 | 1 | 0 | 0.015625 |
| Uncapped grouped | 49,152 | 0 | 0 | 0.015625 |

The following counts cover the first 8,192-query internal tile of each remote chunk and all layers. They do not cover the second internal tile.

| 64K sampled remote queries | Opened regions/head | Opened entries/head | Numerical fallbacks | Route-workspace fallbacks |
|---|---:|---:|---:|---:|
| Capped grouped | 7.91 | 36.63 | 12/25,165,824 | 0/25,165,824 |
| Uncapped indexed | 82.06 | 304.20 | 21,239/25,165,824 | 0/25,165,824 |
| Uncapped grouped | 82.85 | 308.54 | 21,239/25,165,824 | 5,424,398/25,165,824 |

Synthetic checks cover deliberately uneven region sizes, the eight-region cap, a tight work bound, and concentrated native mass that triggers coarse recomputation. A separate run forces a route workspace of only one route per query on average, exercising exact overflow fallback.

GPU event diagnostics below record stage boundaries without synchronizing the CPU after each stage. They remain instrumented measurements; the table above uses separate uninstrumented runs. Exact grouped timings include dispatch and reduction. The preparation and event stage scopes partition the fast prefill attention path, not the entire model.

| Diagnostic | Preparation | Producer | Selection | Exact consumer | Merge/correction |
|---|---:|---:|---:|---:|---:|
| cap8 32K | 0.001 s | 0.181 s | 0.123 s | 0.072 s | 0.061 s |
| grouped 32K | 0.001 s | 0.176 s | 0.098 s | 1.020 s | 0.297 s |
| direct 32K | 0.001 s | 0.178 s | 0.099 s | 0.631 s | 0.301 s |
| direct 64K | 0.002 s | 0.760 s | 0.466 s | 2.503 s | 0.921 s |

Failed preliminary trials are preserved in provenance: these include a numerical-boundary check, compiler errors during merge and early-exit development, and out-of-memory failures during repeated uncapped 64K runs with 16K internal tiles. Their dependent runs were skipped, and superseded queued trials were canceled. The final comparison uses 8K internal tiles and a smaller route workspace. The validation now records threshold-boundary discrepancies explicitly and rejects any outside the stated tolerance. Only successful later runs appear in the tables.

The experiment is opt-in: `LOD_H100_PREFILL_RULE=entry LOD_H100_PREFILL_IMPLEMENTATION=fast LOD_H100_FAST_TILE=8192 LOD_H100_FAST_SELECT_M=16 LOD_H100_FAST_SAVED_CORRECTION=1 LOD_H100_FAST_ROUTE_CAPACITY=64`, with `LOD_H100_FAST_LEAF=direct` or `grouped`; `LOD_H100_PREFILL_MAX_REGIONS=8` enables the separate capped diagnostic. Serving defaults remain unchanged. These experiments do not establish a subquadratic bound for complete prefill or amortized cache updates.

Commands, hardware identity, source hashes, logs, per-layer validation, and raw timings are retained in the JSON files and `provenance` directory.

Follow-up: [entry-16 capped at sixteen regions, with rerun cap-eight and top-eight controls](CAP16.md).
