# Dynamic top-k within the enlarged static cohort

## Result

Restricting dynamic routing to the new
`max(32, ceil(sqrt(T) / 8))` small-centroid cohort does **not** preserve the
quality benefit of either parent policy. At 128K on Qwen3.5-0.8B, literal
prefill top-8 improves slightly over production prefill top-3, but both are
worse than unrestricted routing and opening every eligible static-cohort
leaf.

All ProLong rows use the same eight deterministic concatenated documents as
`artifacts/static_cohort_schedule_20260826`, covering 1,048,568 next-token
predictions with BF16, B8, 16K scheduler chunks, and 4K state updates.

| 128K ProLong policy | Prefill routes | Cross entropy | Perplexity | CE delta vs full |
|---|---:|---:|---:|---:|
| Full attention | - | **2.511656** | **12.325328** | - |
| Unrestricted production top-k | 3 | 2.538752 | 12.663861 | +0.027096 |
| New static cohort, open all | all eligible | 2.532484 | 12.584730 | +0.020828 |
| New cohort-constrained top-k | 3 | 2.557946 | 12.909280 | +0.046290 |
| New cohort-constrained top-k | 8 | 2.554047 | 12.859045 | +0.042391 |

Literal prefill top-8 improves cohort-constrained CE by 0.003899 over top-3
(paired standard error 0.001303), but remains 0.015295 CE worse than
unrestricted routing (paired SE 0.003382) and 0.021563 worse than opening the
whole new cohort (paired SE 0.004441). Thus the main damage comes from making
large centroids ineligible, not from using too few routes inside the cohort.

The matched, chat-formatted 64-example NIAH-S3 panel is less discriminating:

| 128K NIAH-S3 policy | Score |
|---|---:|
| Cohort-constrained production top-k (prefill 3, decode 8) | **64/64** |
| Cohort-constrained literal top-8 (prefill 8, decode 8) | 63/64 |

The literal-top-8 failure is a one-character GUID error on sample 56. Given
the perfect top-3 result and improved top-8 CE, this is best treated as a
boundary-sensitive retrieval error rather than evidence that opening fewer
routes is generally better.

## Interpretation

The static policy succeeds by opening **every** eligible small posting list.
Adding top-k changes that into a much smaller, query-dependent subset while
also forbidding top-scoring large centroids from opening. The excluded large
centroids are not merely expensive garbage: some carry useful fine-grained
information that unrestricted top-k selects. This hybrid therefore gives up
both the static policy's broad exact coverage and unrestricted top-k's ability
to follow mass into a large centroid.

The implementation uses `VLLM_LOD_PREFILL_ROUTE_COHORT=1` and
`VLLM_LOD_DECODE_ROUTE_COHORT=1`. `VLLM_LOD_PREFILL_OPEN_COUNT` selects a
literal prefill route count; when unset, vLLM keeps the production
`min(3, VLLM_LOD_OPEN_COUNT)` prefill setting. The decode route count remains
`VLLM_LOD_OPEN_COUNT=8`. Artifact audits record the requested cohort and route
counts, and the final decode eligibility limit is 47 (the exclusive-kernel
form of the inclusive 46-leaf cap).

## Artifacts

* `qwen08_prolong_det_cohort_top3_sqrt8_min32_128k_b8_s8.json`
* `qwen08_prolong_det_cohort_top8_sqrt8_min32_128k_b8_s8.json`
* `qwen08_niah_cohort_top3_sqrt8_min32_128k_b8_s64.json`
* `qwen08_niah_cohort_top8_sqrt8_min32_128k_b8_s64.json`

`smoke_true8_8k_s1.json` is only the dispatch smoke test and is not part of
the quality comparison.
