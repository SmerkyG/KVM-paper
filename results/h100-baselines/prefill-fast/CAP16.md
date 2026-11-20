# Entry-16 prefill capped at sixteen regions

BF16 Llama-3.1-8B-Instruct, batch 1, H100 80GB. Full-model prefill includes cache updates; initialization and decode graph capture are excluded. Median of three measured repetitions after one warmup, with identical replayed prompts. Every GPU experiment ran with uv through the logged cluster-run adapter.

The two capped variants apply the same entry-16 threshold and strict opened-entry work guard, then retain the first eight or sixteen qualifying regions in region order. They do not choose the highest-scoring eight or sixteen. Remaining regions stay coarse. The exact initial prefill chunk remains 16,384 tokens, internal tiles remain 8,192 queries, and the cache update schedule is unchanged. Both use the same score-reuse producer and grouped exact-attention kernel.

| Configuration | 32K prefill | 64K prefill |
|---|---:|---:|
| Original top-8 routing | [1.660 s](threshold-fast-cap16-top8-32k.json) | [4.195 s](threshold-fast-cap16-top8-64k.json) |
| Entry-16 capped at 8 regions | [1.718 s](threshold-fast-cap16-cap8-32k.json) | [4.412 s](threshold-fast-cap16-cap8-64k.json) |
| Entry-16 capped at 16 regions | [1.852 s](threshold-fast-cap16-cap16-32k.json) | [4.839 s](threshold-fast-cap16-cap16-64k.json) |

Raising the cap from eight to sixteen changes latency by +7.8% at 32K and +9.7% at 64K. The sixteen-region version is 11.5% slower than original top-8 at 32K and 15.4% slower at 64K.

| 64K audited configuration | Mean opened regions/head | Mean opened entries/head | Query heads reaching the cap | Maximum opened regions |
|---|---:|---:|---:|---:|
| Cap 8 | 7.91 | 36.63 | 97.85% | 8 |
| Cap 16 | 15.64 | 72.56 | 95.63% | 16 |

Counts cover the first 8,192-query internal tile of each remote chunk across all layers, in each configuration’s own evolving model state. They exclude the second internal tile. No query-wide CPU synchronization is introduced into timed attention.

| 64K correctness check | Sampled query heads | Changed region decisions | Outside 2e-5 boundary tolerance | Maximum attention output difference |
|---|---:|---:|---:|---:|
| Cap 8 | 49,152 | 0 | 0 | 0.007812 |
| Cap 16 | 49,152 | 0 | 0 | 0.015625 |

The independent reference checks selection, the strict entry-work bound, and attention output using the new score-reuse arithmetic. Synthetic checks also cover uneven sizes, both caps, a tighter work guard, and concentrated mass that exercises numerical fallback. These are implementation checks; no RULER quality panel was run for cap sixteen. A cap changes the retrieval frontier and does not inherit the uncapped method’s quality.

The prior uncapped entry-16 timings and detailed pipeline scope remain in [the producer comparison](REPORT.md). This follow-up reruns cap eight and original top-eight controls with the same source snapshot as cap sixteen.

Reproduce with the earlier fast settings and `LOD_H100_PREFILL_MAX_REGIONS=16`. These remain opt-in experiments; serving defaults are unchanged. Commands, GPU identity, source hashes, validation, and raw timings are retained alongside this report.
