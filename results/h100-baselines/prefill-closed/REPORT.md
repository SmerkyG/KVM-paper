# H100 two-tier threshold prefill

Llama-3.1-8B-Instruct, BF16 model and KV, batch one, H100 80GB HBM3, RetroInfer runtime. All experiments use uv and cluster-run with the same fixed ProLong prompt at each length. These are full-model prefill times, median of three warmed repetitions, with model loading, cache allocation and graph capture excluded. The initial exact 16,384-token chunk, subsequent prefill chunks, local/sink attention and cache-update schedule remain unchanged. Decode stays at per-region c=16 in every LoD arm; the dense control uses native dense attention and its own cache. Prefill uses eager GPU launches in every arm; CUDA graphs apply only to the separately timed decode transition.

| Prefill routing | 32K (s) | 64K (s) |
|---|---:|---:|
| Dense | [1.553](prefill-closed-final-dense-32k.json) | [4.760](prefill-closed-final-dense-64k.json) |
| Fixed top-eight | [1.648](prefill-closed-final2-top8-32k.json) | [4.181](prefill-closed-final2-top8-64k.json) |
| Per-region c=16, previous fused implementation | [2.016](prefill-closed-final2-region-legacy-32k.json) | [5.673](prefill-closed-final2-region-legacy-64k.json) |
| Per-region c=16, optimized | [1.880](prefill-closed-final2-region-closed-32k.json) | [5.467](prefill-closed-final3-region-fast-native-64k.json) |
| Entry-16, previous optimized implementation | [2.498](prefill-closed-final2-entry-legacy-32k.json) | [7.525](prefill-closed-final2-entry-legacy-64k.json) |

The optimized per-region path remains 21.1% / 14.9% slower than dense at 32K / 64K. The dense controls were rerun on the same GPU instance, checkpoint, precision, prompt and prefill timing scope; these are not the earlier dense numbers from a different H100 instance.

The per-region reduction is 6.8% / 3.6% at 32K / 64K. Entry-16 retains its previous implementation: the new grouped, packed-output and per-query-indexed pilots did not establish a speed improvement. The two routing rules select different frontiers; the per-region improvement does not apply to entry-16.

The previous fused control uses a 4,096-query internal tile. Both one-warp and four-warp route reduction were measured; the table uses the faster result at each length.

## Cache accounting

Before stopping the LoD prefill timer, each layer explicitly completes any deferred catch-up that the harness would otherwise perform at the start of graph setup. This cost is included in the table and separately recorded as `prefill_cache_completion_s`. The graph setup timer contains graph preparation and capture. Two output tokens are used to exercise the transition to decode; that one decode step is not a steady-state decode benchmark.

| Context | Top-eight final catch-up | Previous c=16 final catch-up | New c=16 final catch-up |
|---|---:|---:|---:|
| 32K | 0.160 ms | 0.138 ms | 0.157 ms |
| 64K | 0.169 ms | 0.145 ms | 0.157 ms |

## Implementation and checks

The 32K path prepares BF16 means once per remote chunk, computes shared-GQA QK into FP32 scores, reads those scores once to normalize routing and pack all selected regions, and computes the closed-summary field directly. Query/head/expert counters are fused into selection. Exact attention uses persistent region-grouped workers; dispatch visits actual route counts, and the reducer has a 128-route fast case plus an uncapped fallback. Selections remain uncapped, including the existing 1,024-entry eligibility limit for per-region c=16. There is no selected-count CPU readback or GQA KV expansion.

The 32K configuration forms closed summaries directly. At 64K the faster configuration uses the fused coarse producer, retains its native scores and routing normalizer, and uses a full-row parallel prefix scan instead of a serial block loop for selection. It subtracts opened summaries in the final merge. A numerical fallback recomputes the closed field when subtraction would lose precision. It uses the same persistent exact workers and count-aware dispatch. Routing normalization uses FP32 saved dot products and query RMS; the older fused control rounds the normalized query to BF16 before its routing matmul, so this is not a claim of bitwise-equivalent old/new selections.

The seven independent closed-summary/native-leaf cases pass on PyTorch 2.5.1+cu124 and 2.13.0+cu130, including uneven and empty regions, entry-work guards, more than 128 selected regions, internal query-tile remainders and CUDA-graph replay. Real model checks at 64K cover 12,288 sampled query/head frontiers: 0 selection differences; maximum exact-output difference 0.015625 against the independent scalar kernel and coarse-output difference 0.015859 against PyTorch. These are numerical checks; the full RULER panel has not been rerun. Serving defaults are unchanged.

The 64K score-reuse path additionally checks 49,152 sampled query/head frontiers, with 0 selection differences and maximum final-output difference 0.015625 against its reference. The full-row selector also passes three uncapped synthetic cases, including concentrated coarse mass and precision fallback. The native CUDA regression suite passes 21 tests.

Full-model measurements use PyTorch 2.5.1 and Triton 3.7.1. Its compiled RetroInfer dependencies remain in their existing environment. The separate PyTorch 2.13 check uses the native FP32-output bmm interface; the older environment uses its existing cuBLAS handle for the same BF16-input, FP32-output operation. No full-model PyTorch-version speedup is claimed.

## Reproduction and limits

Set `LOD_H100_PREFILL_RULE=region`, `LOD_H100_PREFILL_MULTIPLIER=16`, `LOD_H100_PREFILL_IMPLEMENTATION=closed`, `LOD_H100_CLOSED_MATMUL=blas`, `LOD_H100_CLOSED_METADATA=atomic`, `LOD_H100_CLOSED_SELECT_QM=1`, `LOD_H100_CLOSED_REDUCE=wide`, `LOD_H100_CLOSED_LEAF_M=64`, `LOD_H100_CLOSED_LEAF_N=32`, and `LOD_H100_CLOSED_LEAF_WARPS=4`. The measured internal query tile is 4,096 and the persistent worker count is 264. The internal query tile bounds scratch; it does not change the exact-prefill chunk size.

For the 64K score-reuse configuration, use the same leaf/worker settings with `LOD_H100_PREFILL_IMPLEMENTATION=fast`, `LOD_H100_FAST_LEAF=native`, `LOD_H100_FAST_SAVED_CORRECTION=1`, `LOD_H100_FAST_FULL_SELECT=1`, and `LOD_H100_FAST_TILE=4096`. The exact commands in the selected run manifests are authoritative.

Routing selection and packing use linear prefix scans in region count. Shared score production still visits every query/region pair, and the cache path still performs ownership assignment, clustering and periodic maintenance. This report makes no new subquadratic claim for complete prefill or amortized cache updates. The deterministic entry-work bound belongs to the entry rule and is not a coverage or retrieval guarantee.

Raw pilots and failures remain alongside the selected runs. Exclusions include the first unsupported old-PyTorch bmm attempt, missing pytest in an early test environment, a cancelled broad three-tier compile, invalid inherited entry-only options in the first region controls, and a grouped-reducer address-overflow pilot fixed by 64-bit addressing. Smaller exact-attention tiles, grouped-query selection/reduction and packed outputs did not beat the selected configurations. Commands, GPU identity, logs and source hashes are in `provenance/`; `source/` contains the latest selected snapshot. Per-run copies preserve the three evolving benchmark modules. All selected runs share identical cache and leaf-kernel sources; the harness change between the final rounds only records the new selection option in result metadata.
