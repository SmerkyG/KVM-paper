# Fixed adjacent T/16, top-16 speed study (2026-08-27)

## Result

Fixed adjacent groups of 16 with top-16 opening are **not significantly faster
than the current learned two-tier implementation during prefill**.  On
Qwen3.5-0.8B BF16, batch 8, and real ProLong text, the fixed variant is 1.4%
faster at 16K, tied at 32K, and 4.7% slower at 64K.  It remains substantially
faster than full attention at long context, but does not improve materially on
the current LoD implementation.

The fixed layout is attractive for simplicity and predictable posting-list
lengths.  Its speed advantage from cheaper state construction and routing is
almost exactly spent on opening 256 exact remote tokens per query (16 groups x
16 leaves), rather than the current prefill top-3 routes.

## Matched prefill results

All LoD rows use this worktree's current kernels, grouped-64 coarse routing,
real ProLong documents, Qwen/Qwen3.5-0.8B, BF16, and batch 8.  The 32K rows are
phase-instrumented; the others are uninstrumented.

| Context | Fixed T/16 top-16 | Current learned two-tier top-3 | Fixed change |
|---:|---:|---:|---:|
| 16K | 747.52 ms | 757.94 ms | **1.37% faster** |
| 32K | 1,634.21 ms | 1,641.72 ms | **0.46% faster** |
| 64K | 3,747.23 ms | **3,580.17 ms** | **4.67% slower** |

For context, the historical matched Hugging Face full-attention measurements
were 2,026.84 ms at 32K and 5,973.87 ms at 64K.  Thus fixed T/16 top-16 is
1.24x and 1.59x faster than full attention at those lengths, respectively, but
the relevant current-LoD comparison above shows no new speed win.

Artifacts:

- `screen_16k_b8_fixed.json`
- `screen_16k_b8_ordinary_top3.json`
- `phase_32k_b8_expert_fixed_group64.json`
- `phase_32k_b8_ordinary_top3_group64.json`
- `screen_64k_b8_expert_fixed_group64.json`
- `screen_64k_b8_ordinary_top3_group64.json`

## Why it ties at 32K

The phase breakdown accounts for all six full-attention layers over the whole
batch-8 prefill.  `route_logits` is a diagnostic subset of `route` and is not
added separately.

| 32K phase | Fixed T/16 top-16 | Learned top-3 | Difference |
|---|---:|---:|---:|
| Routing | 128.60 ms | 141.81 ms | -13.21 ms |
| Coarse attention | 79.95 ms | 99.33 ms | -19.38 ms |
| Exact leaves | 168.99 ms | 68.78 ms | +100.21 ms |
| Local attention | 186.03 ms | 190.01 ms | -3.98 ms |
| State update | 33.14 ms | 103.50 ms | -70.35 ms |
| Page append | 14.44 ms | 14.81 ms | -0.37 ms |
| Model/other | 1,023.05 ms | 1,023.48 ms | -0.42 ms |
| **Total** | **1,634.21 ms** | **1,641.72 ms** | **-7.51 ms** |

Fixed grouping saves 103.3 ms in routing, coarse attention, and state update,
but exact attention costs an additional 100.2 ms.  The net attention-path gain
is therefore only about 7.5 ms over a 1.64 s model invocation.

The expert-major leaf organization is already the right layout for this case:
at 32K every active posting list has exactly 16 leaves and an expert receives
about 293 queries on average.  Switching to query-major execution regressed
32K prefill from 1,897 ms to 2,176 ms in the initial experiment.  Retuning the
exact kernel from N=32 to N=16 reduced its isolated time only from 176.24 ms to
173.29 ms (1.7%, roughly 0.2% end to end):

- `exact_32k_b8_m16n32w2.json`
- `exact_32k_b8_m16n16w2.json`

## Context-length crossover

At 32K, T/16 has fewer centroids than the normal `16*sqrt(T)` schedule.  At
64K the schedules both produce about 4,096 centroids.  The fixed method then
loses its centroid-count advantage while still opening 16 routes instead of
three during prefill.  Beyond 64K, T/16 would create *more* centroids than the
sublinear schedule unless another level or an explicit cap is added.  This is
why the method is not a promising general speed replacement despite its
regular geometry.

## Matched vLLM serving-prefill result

The production-path conclusion is the same.  This comparison uses 64K real
ProLong document prompts, batch 8, BF16, 16K scheduler chunks per request
(`max_num_batched_tokens=131072`), synchronous scheduling, one generated token
to isolate prefill, and three measured repeats.  The persistent weight daemon
was used only to reduce startup time.

| vLLM 64K B8 | Median prefill | Prompt throughput | Relative to learned |
|---|---:|---:|---:|
| Fixed T/16, prefill top-16 | 3.9820 s | 131,664 tok/s | **5.42% slower** |
| Learned two-tier, prefill top-3 | **3.7773 s** | **138,800 tok/s** | baseline |

Diagnostics confirm that the fixed run used `state_premerge_factor=16`,
`effective_prefill_open_count=16`, and the expert leaf layout; its execution
log records `_route_logits_topk_coarse_attention_kernel`, confirming the fused
top-k/coarse route.  The learned control used premerge 1 and prefill top-3.
Both artifacts report the fused route as eligible, all eight prompt hashes are
unique, and all three repeats produced stable output.  The result is therefore
not an accidental fallback or a comparison against serialized nominal batches.

Artifacts:

- `vllm_prolong_64k_b8_prefill_top16_mbt128k_d1.json`
- `vllm_prolong_64k_b8_ordinary_top3_mbt128k_d1.json`

The normal multi-token benchmark currently encounters an existing vLLM
lifecycle limitation when decode for an earlier-admitted request is mixed with
prefill for later requests after native remote-K/V fallback was removed.  The
one-token setup avoids that unrelated mixed-phase path and measures only the
prefill requested here.

## Decode-oriented upper-bound experiment

The Hugging Face decode path did benefit from exploiting the fixed 16-leaf
posting lists.  Extending the fused indexed final-attention scan to accept 16
preselected routes reduced the 64K batch-8 step from 19.169 ms to 14.545 ms.
That is 24.1% faster than the non-fused fixed implementation and 21.5% faster
than the matched Hugging Face learned-top-8 run (18.537 ms).

These are **not production vLLM decode results**.  The generic Hugging Face
learned baseline contains very large selected posting lists (up to roughly
2,700 leaves), whereas the production vLLM route path caps overfull selected
centroids and uses captured serving kernels.  The known vLLM two-tier decode
result is therefore much faster than either Hugging Face number.  A fair vLLM
top-16 decode implementation would require widening the fused route-candidate
and reduction path from eight to sixteen; the final indexed-attention scan
alone is no longer the blocker.

Artifacts:

- `decode_64k_b8_fixed_top16_nonfused.json`
- `decode_64k_b8_fixed_top16_fusedscan.json`
- `decode_64k_b8_ordinary_top8_screen.json`

## Correctness issue found during the study

The first paged runs incorrectly showed 64-token posting lists even though the
fixed centroid count was 16.  Batched premerge cached views into a reusable
owner workspace; later updates overwrote those views before page append.  The
owner indices are now snapshotted before reuse.  All headline results above
are reruns after the fix and report both `max_centroid_tokens=16` and
`max_slot_tokens=16`.  The earlier `screen_32k_b8_expert.json`,
`screen_32k_b8_query.json`, and `screen_64k_b8_expert_fused16.json` should not
be used as performance results.
