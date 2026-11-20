# Top-eight routing with per-query leaf attention on H100

BF16 Llama-3.1-8B-Instruct, batch 1, two-tier prefill, with an unchanged 16,384-token exact first chunk. Both configurations use the ordinary top-eight selector and the same coarse, local, and sink attention. Prefill times are medians of three runs after one warmup, on the same replayed prompt and source snapshot.

| Leaf backend | 32K full prefill | 64K full prefill |
|---|---:|---:|
| Original region-grouped | 1.661 s | 4.188 s |
| Per-query indexed | 2.186 s | 6.265 s |
| Per-query slowdown | 31.6% | 49.6% |

The per-query adapter compacts the already selected routes, counts and prefix-packs their KV entries, expands their indices, runs the per-query attention kernel, and formats the combined output for the existing per-region merge. It does not read route counts back to the CPU. The unchanged top-eight producer still creates the grouped-dispatch metadata, although the per-query backend does not consume it. Therefore these are timings for the complete backend swap, including both the adaptation overhead and retained producer work, rather than isolated kernel timings.

To compare attention on identical inputs, the audit invoked both leaf backends using the same queries, selected regions, and cache. The original grouped outputs were reduced to one exact branch for comparison. The 32K audit covered 16,384 sampled query heads across 32 layers. The 64K audit sampled the first 16 queries per head in each remote chunk and layer, covering 49,152 query heads. The 64K audit also included 146 padded or otherwise closed route slots, which the per-query adapter ignores.

| Context | Maximum exact-output absolute difference | Mean absolute difference | Maximum exact LSE difference | Average opened entries per sampled head |
|---|---:|---:|---:|---:|
| 32K | 0.01131 | 0.000203 | 0.0000038 | 268.7 |
| 64K | 0.01220 | 0.000181 | 0.0000058 | 323.8 |

The two timed backends generated the same two token IDs on each prompt. This is a numerical and execution check, not a RULER quality evaluation. The all-chunk 64K audit used a later source snapshot that changes only the audit accumulation; the uninstrumented timing runs all used identical source hashes.

Runs, commands, complete source hashes, raw results, and logs are retained in `provenance` and the JSON files. The experimental backend is enabled only by `LOD_H100_TOP8_QUERY_BACKEND=1`; it does not change serving defaults.
