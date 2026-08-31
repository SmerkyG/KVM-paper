# Short-context recursive DFlash2 analysis

This panel diagnoses and removes the short-context penalty of recursive
three-tier LOD target verification.  The target is
`Qwen/Qwen3.8-27B-FP8` at TP1/B1 on MI325X and the drafter is
`z-lab/Qwen3.8-27B-DFlash2`, with seven draft tokens.  Speed prompts are real,
non-repeated ProLong documents, sampling is greedy, and prefill uses a 16K
aggregate scheduler budget.  The primary verifier-cycle rows emit 256 tokens
after one warmup and are medians of three repetitions.

## Diagnosis

The third tier has a genuine but small fixed cost: it maintains page summaries
and, after selecting a centroid, chooses and evaluates a residual page.  At
short context the selected two-tier posting lists are still small, so this
extra indirection has little leaf traffic to save.  That was not the main
measured DFlash2 problem, however.

The crossover has a direct geometric interpretation.  With `16*sqrt(T)` state
entries, average centroid posting length is `sqrt(T)/16`.  It reaches the
16-token residual-page size at `T = (16*16)^2 = 65,536`, corresponding to 4096
state entries.  Below that point a third tier cannot reduce the average exact
leaf count at all; it can only help unusually large centroids, while paying its
page-routing overhead everywhere.

DFlash2 presents eight target positions in one step-major flattened verifier.
The native grouped route/local kernel was restricted to exactly two positions,
so recursive DFlash2 fell back to eight independent GQA6 route scans.  Each
scan had only six useful rows in an M16 tile and independently loaded the same
centroid K/V.  Local QK and PV were also a later, separate kernel chain.  The
two-tier fixed-mask verifier did not pay this particular fallback cost.

The route kernel now handles any even speculative depth as adjacent independent
pairs.  The eight-position verifier therefore issues four M12 groups instead
of eight M6 groups, halves centroid K/V loading, and fills the native M16 shape
much better.  Every position still uses its own current query, top-eight route,
and causal local length; no route is shared, predicted, or lagged.  The same
programs can fold each pair's causal 512-token local branch into their partial
attention, and the recursive reducer then omits the redundant separate-local
branch.

## Verifier-cycle results

Milliseconds are complete-model target-verifier cycles and therefore do not
depend on how many draft tokens happened to be accepted.  The two-tier control
is the matched three-repeat fixed-mask panel in
`artifacts/dflash2_qwen38_20260831`.

| context | two-tier | old three-tier | paired route only | paired route + local | old / new | two-tier / new |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 40.123 | 40.693 | 39.758 | **39.754** | 1.024x | 1.009x |
| 16K | 39.715 | 40.885 | 39.717 | **39.242** | 1.042x | 1.012x |
| 32K | 39.879 | 40.886 | 40.032 | **39.757** | 1.028x | 1.003x |
| 64K | 40.731 | 41.195 | 40.339 | **39.676** | 1.038x | 1.027x |

Thus the optimized recursive verifier is no longer slower at short context:
it is 0.3--2.7% faster than the current two-tier verifier across 8--64K.  The
controlled no-local run attributes 0.85--1.17 ms of the old penalty to paired
state routing.  Fusing local is neutral at 8K and saves another
0.28/0.48/0.66 ms at 32/16/64K.  Page selection remains part of the recursive
algorithm, but it was not the dominant short-context regression.

End-to-end milliseconds per emitted token for the new recursive path are
16.529, 8.463, 15.162, and 13.654 ms at 8/16/32/64K.  These should not be used
to isolate attention: harmless floating-point regrouping changes the greedy
trajectory and DFlash acceptance.  The verifier-cycle table is the stable
kernel comparison.

## Prefill

Pairwise routing is decode-only.  In the original three-tier prefill profile,
page-routed exact attention cost 435.7 ms over the complete 64K prompt versus
310.7 ms for two-tier exact-leaf attention, and recursive page maintenance cost
another 12.9 ms.  Recursive top-level routing was simultaneously 107.9 ms
faster, leaving only a small wall-time gap.

For this D256/GQA6/KVH4 TP1 geometry, the existing regular expert/MFMA kernel
can instead attend to every leaf of each selected centroid during short
prefill, while updates continue constructing the recursive page archive for
decode.  This removes page choice from the prefill consumer and evaluates more
tokens with a substantially more GPU-friendly shape.

| context | two-tier | page-selecting three-tier | complete-expert three-tier | page-selecting / complete |
|---:|---:|---:|---:|---:|
| 8K | **0.984 s** | 1.001 s | 0.994 s | 1.007x |
| 16K | **2.053 s** | 2.061 s | 2.055 s | 1.003x |
| 32K | 4.171 s | 4.212 s | **4.154 s** | 1.014x |
| 64K | 8.630 s | 8.701 s | **8.531 s** | 1.020x |

The automatic policy uses complete experts only for recursive BF16
`(D, GQA, KVH) = (256, 6, 4)` when the request's total prompt length is at most
64K, the point where `sqrt(T)/16` reaches the 16-token residual-page size and
the scheduled state reaches 4096 entries.  Longer requests use page selection
for their complete prefill, including their early chunks; this avoids making a
128K request pay a short-context policy during its first half.  Explicit
`VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES=0/1` still means never/always.  Phi's
previously validated D128/GQA4/KVH2 complete-expert policy remains unbounded.

The two consumers also retain independent launch geometry.  Complete-expert
MFMA uses two waves, whereas the D256 query-major residual-page kernel keeps
its measured one-wave launch.  Sharing the expert setting with the page path
made automatic 128K prefill 19.230 seconds despite choosing pages correctly.
After separating them, automatic 128K is 18.316 seconds, matching the explicit
page-only control's 18.338 seconds within 0.12%.  A chunk-local policy that used
complete experts for the first 64K of the same request was slower at 18.612
seconds.

## Quality and execution audit

The final automatic path scores 8/8 on chat-formatted NIAH-S3 at both 8K and
64K.  Its device-written audit records flattened speculative execution,
pairwise route execution, and fused local execution as both configured and
observed.  The dispatch record reports the 65536-token adaptive prefill bound.

The normal, non-speculative prefill path was also compared with teacher-forced
next-token loss on eight deterministic 8K ProLong documents.  All arms use the
same 65,528 scored tokens, batch eight, TP1, no chat template, the language
model only, BF16 LOD storage, and a 16K aggregate scheduler budget.  Token
SHA-256 hashes match sample by sample, and the execution audit records native
prefill for the full control and 80 direct-LOD prefill calls for each LOD arm.

| prefill | token CE | CE delta vs full | perplexity | PPL delta vs full |
|---|---:|---:|---:|---:|
| full attention | **0.890538** | -- | **2.436439** | -- |
| two-tier LOD | 0.891943 | +0.001405 | 2.439865 | +0.1406% |
| three-tier LOD, current auto policy | 0.892186 | +0.001648 | 2.440458 | +0.1650% |

Thus both LOD forms are within 0.17% of full-attention perplexity.  Three-tier
is only +0.000243 CE / +0.0243% PPL behind two-tier.  At 8K, the current
three-tier auto policy uses the regular complete-expert prefill consumer while
continuing to build the recursive page archive for decode; this row measures
the actual default short-context policy, not a forced residual-page-selection
ablation.

## Raw records

- `lod3_bf16_b1_8k64k_r3_d256.json`: old recursive baseline (cluster 12277)
- `lod3_pairroute_v2_bf16_b1_8k64k_r3_d256.json`: paired route/local result
  (cluster 12281)
- `lod3_pairroute_no_local_bf16_b1_8k64k_r3_d256.json`: controlled local
  ablation (cluster 12288)
- `lod3_pairroute_niah_bf16_b1_8k64k_n8.json`: pairwise-kernel quality
  control (cluster 12283)
- `lod3_prefill_allleaves_bf16_b1_8k64k_r3_d64.json`: forced complete-expert
  prefill panel (cluster 12291)
- `lod3_prompt_auto_bf16_b1_8k64k_n8_r3_d64.json`: final automatic speed,
  quality, and execution audit (cluster 12296)
- `lod3_pageonly_bf16_b1_128k_r3_d64.json` and
  `lod3_prompt_auto_bf16_b1_128k_r3_d64.json`: explicit-page and final
  automatic 128K controls (clusters 12294 and 12298)
- `lod2_bf16_b1_8k64k_profile_r1_d64.json` and
  `lod3_bf16_b1_8k64k_profile_r1_d64.json`: matched prefill phase profiles
  (clusters 12276 and 12275)
- `qwen38_prolong8k_full_b8_s8.json`,
  `qwen38_prolong8k_two_b8_s8.json`, and
  `qwen38_prolong8k_three_b8_s8.json`: matched ProLong loss panel (clusters
  12300--12302)
