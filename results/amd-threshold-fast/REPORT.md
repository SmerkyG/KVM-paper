# AMD two-tier compact threshold decode

Follow-up: [AMD threshold kernel optimization and matched top-8 controls](../amd-threshold-fast-v2/REPORT.md). This newer report includes the compact-table implementation, complete speed runs and explicit quality and timing limitations.

Batch one, 65,536-token prompts, BF16 KV and the FP8-weight Qwen3.8-27B / K2-Horizon-32B models on MI325X. The primary threshold opens regions above 16 times mean normalized-query proxy mass, retaining the existing 1,024-entry eligibility limit. This is an opt-in decode change. Exact first-chunk prefill remains 16,384 tokens; later prefill uses the existing top-8 path.

The port fuses threshold selection and linear prefix packing, then traverses selected entries directly. It removes page-descriptor expansion and keeps AMD FP32 count division, coarse summaries, local/sink attention and KV append. The selected setting for both models is `indexed`, 64 splits, 32 entries per block, two warps and routing tile 64. It uses six attention kernels instead of nine. The five-kernel combined-local variant was also checked and timed.

## Full-model autoregressive decode

One warmup generation followed by two measured generations, each with 1,024 timed decode steps. Prefill is excluded from these milliseconds per token; serving overhead and normal amortized cache updates are included. Controls and variants use the same physical GPU within each model and run serially. Methods can generate different continuations, so this is not a quality-matched or identical-query speed comparison. Native AMD top-8 uses native-query ranking; thresholds use normalized-query ranking and a variable exact-work budget.

| Model | Top-8 ms/token | Old c=16 | Compact c=16 | Less time than old threshold |
|---|---:|---:|---:|---:|
| QWEN | [27.462](speed-controls-v1/amd-fast-speed-controls-v1-qwen-top8.json) | [29.242](speed-final-v1/amd-fast-speed-final-v1-qwen-legacy.json) | [28.888](speed-final-v1/amd-fast-speed-final-v1-qwen-fast.json) | 1.2% |
| K2 | [38.652](speed-controls-v1/amd-fast-speed-controls-v1-k2-top8.json) | [43.832](speed-final-v1/amd-fast-speed-final-v1-k2-legacy.json) | [41.335](speed-final-v1/amd-fast-speed-final-v1-k2-fast.json) | 5.7% |

The compact worker improves the existing threshold implementation, but remains slower than native AMD top-8 in these full-model runs. The Qwen difference is small and comes from one prompt and two timed repetitions. Native top-8 already shares KV work across GQA heads, whereas the compact exact workers remain per query head. The Qwen model has 16 LoD attention layers and K2 has 64, so the per-attention-call improvements contribute differently to total latency.

## Fixed real-query attention diagnostics

Identical query, KV state and selected regions, sampled from first/middle/last attention layers after a three-token generation. Times include the complete attention pipeline and the same local-length rollback. Each timing is a median of three measurements of a graph with eight calls, replayed 32 times. These are averages over three sampled layers, excluding projections, FFNs, model scheduling and cache catch-up. The worker retains the diagnostic request cache after completion; normal speed and quality runs do not install that hook.

| Model | Rule | Old threshold µs | Compact µs | Less time | Mean open regions/head | Mean exact remote entries/head |
|---|---|---:|---:|---:|---:|---:|
| QWEN | region | 87.57 | 71.18 | 18.7% | 37.89 | 2584.7 |
| QWEN | entry | 61.18 | 57.37 | 6.2% | 21.11 | 456.4 |
| K2 | region | 128.82 | 82.93 | 35.6% | 29.22 | 3067.9 |
| K2 | entry | 88.10 | 59.74 | 32.2% | 78.44 | 519.9 |

Opened-work counts average over sampled layers and query heads at this single decode step, not across generated tokens.

Both per-region c=16 and per-entry bound-16 routing were checked. The per-entry rule has kernel-only timings here; no full-model or RULER claim is made for that rule. All 216 real-query configurations matched the old ordered region selections exactly. Floating reduction orders differ: maximum attention-component difference across all tuned configurations was 0.001953125 for Qwen and 0.00390625 for K2.

## Retrieval implementation smoke check

Eight archived 64K examples per task: four historical failures plus four successes, separately selected for each model. This small, deliberately enriched subset checks implementation regression; it is not a full RULER panel or an accuracy estimate. Old and compact c=16 use identical tokenized requests and references.

| Model | Task | Old c=16 | Compact c=16 |
|---|---|---:|---:|
| QWEN | niah_multikey_2 | 75.00% | 75.00% |
| QWEN | niah_multikey_3 | 100.00% | 100.00% |
| K2 | niah_multikey_2 | 62.50% | 62.50% |
| K2 | niah_multikey_3 | 87.50% | 87.50% |

- QWEN: old/compact retrieval examples improved 0, declined 0; changed prediction texts 3/16.
- QWEN speed repeat checks against warmup: top8: identical, legacy: differed at [[57], [None]], fast: identical.
- K2: old/compact retrieval examples improved 0, declined 0; changed prediction texts 1/16.
- K2 speed repeat checks against warmup: top8: identical, legacy: differed at [[52], [52]], fast: identical.

## Validation and provenance

The independent PyTorch reference suite passed two sets of 54 cases across the packed, indexed and combined variants, including the chosen two-warp exact worker, covering both model geometries, empty selections, oversized regions, region imbalance, and a strict per-entry work guard. The largest reference attention-component error was 0.000553534, with zero selection mismatches. Passing these checks does not establish bitwise equivalence or full-panel quality.

The initial `probe-v1` jobs failed because completed-request cleanup had cleared the cache; they contribute no timings. The initial legacy speed submissions in `speed-controls-v1` and `speed-legacy-v1` were cancelled before measurement because explicit GPU filters allow overlapping one’s own jobs. Those cancelled runs contribute no timings. Final legacy/compact measurements execute sequentially inside one allocation per model. Source snapshots are immutable and hash-validated; every GPU run uses `uv` through `cluster-run`.

[Raw summary](REPORT.json), [chosen settings](settings.json), [jobs and placement](provenance.json), [Qwen fixed-query inputs/timings](probe-v2/amd-fast-probe-v2-qwen.json), [K2 fixed-query inputs/timings](probe-v2/amd-fast-probe-v2-k2.json), [retrieval cohort manifest](quality-v1/cohort-manifest.json).

Serving defaults are unchanged. Enable the experimental worker with `LOD_AMD_THRESHOLD_MULTIPLIER=16`, `LOD_AMD_THRESHOLD_IMPLEMENTATION=indexed`, `LOD_AMD_THRESHOLD_SPLITS=64`, `LOD_AMD_THRESHOLD_ENTRY_BLOCK=32`, and `LOD_AMD_THRESHOLD_ENTRY_WARPS=2`. Prefill and complete-execution asymptotic work were not changed or re-audited by this decode port.
