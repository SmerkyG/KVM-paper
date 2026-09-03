# Static-cohort prefill with current kernels (2026-08-26)

## Result

The native-GQA coarse-attention packing added for routed two-level prefill also
applies to static-cohort prefill because both modes share
`route_logits_coarse_attention`.  No static routing rule or cohort membership
changes.  The automatic production geometries are:

| model geometry | control | native-GQA |
|---|---|---|
| Qwen3.8 D256/GQA6/KV4 | M64/N32/W8 | M64/N16/W8 |
| OLMo D128/GQA5/KV8 | M64/N32/W8 | M128/N16/W8 |

Matched 64K, batch-8 runs use eight distinct real ProLong documents, 16K vLLM
scheduler chunks, 4K LOD catch-up blocks, BF16 LOD storage, an untimed warmup,
and the median of three repetitions:

| model | direct-GQA off | current static | change |
|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | 79.422 s | **69.648 s** | -12.31% |
| OLMo-3-1125-32B | 72.053 s | **70.901 s** | -1.60% |

Worker diagnostics observed the requested production kernels, not merely the
host-side configuration: Qwen executed `(M64, N16, W8)` and OLMo executed
`(M128, N16, W8)`.  The exact repetition vectors are:

- Qwen off: `[80.719, 79.377, 79.422]` seconds;
- Qwen current: `[70.980, 69.643, 69.648]` seconds;
- OLMo off: `[72.060, 72.049, 72.053]` seconds; and
- OLMo current: `[71.453, 70.901, 70.869]` seconds.

Relative to the original static-cohort panel, which preceded several shared
prefill/cache fixes, current static prefill is 20.54% faster on Qwen
(87.652 to 69.648 seconds) and 27.14% faster on OLMo (97.306 to 70.901
seconds).  The matched controls above isolate the portion due to direct-GQA;
they also show that most of OLMo's old 97-second anomaly had already been
removed by the intervening shared-path fixes.

For context, the latest routed prefill is 65.062 seconds on Qwen and 69.286
seconds on OLMo.  Static is now 7.05% and 2.33% slower, respectively.  Against
historical full attention, static Qwen is 1.59x faster (110.565 seconds full),
while static OLMo is 3.45% slower (68.537 seconds full).

## Other transferable changes

Muse static prefill with the current shared path measures 53.452 seconds,
down from 54.885 seconds in the original panel.  Its power-of-two GQA16
geometry already fills the old coarse tile, so native-GQA remapping is not a
win.  It remains 3.76% slower than the latest 51.514-second routed result and
2.84% slower than the historical 51.933-second full-attention result.

The routed Muse implementation also overlaps coarse, exact-leaf, and local
branches.  Applying that overlap mechanically to static prefill was rejected:
at the normal 0.8 vLLM memory setting, the full query-by-centroid logits field
remained live beside the much larger static exact-leaf AITER working set and
exhausted transient GPU memory.  Static mode is consequently excluded from
the automatic routed-overlap dispatch; the overlap setting remains a routed
prefill diagnostic rather than a static production option.

## Static phase profile

A 64K batch-8 Qwen3.5-0.8B profile on real ProLong text measured 4.702 seconds
end to end and 1.919 seconds inside two-level LOD prefill.  Its principal LOD
subphases were:

| phase | aggregate GPU time | share of two-level prefill |
|---|---:|---:|
| exact-leaf AITER | 1.246 s | 64.9% |
| local attention | 0.320 s | 16.7% |
| coarse attention kernel | 0.283 s | 14.8% |
| route QK GEMM | 0.123 s | 6.4% |
| static table construction + unpack | 0.068 s | 3.5% |
| centroid K/V mean preparation | 0.036 s | 1.9% |

Some phases overlap, so their percentages are diagnostic rather than additive.
The profile shows that eliminating the materialized route-logit field is useful
primarily for transient memory and has only a modest standalone latency ceiling.
The next large static-specific speed opportunity is the exact-leaf branch, not
top-k (which static routing already eliminates) or table construction.

## Correctness and artifacts

Direct-GQA only changes row packing.  Its isolated BF16 checks have maximum
output error `3.052e-5` and maximum LSE error `1.907e-6`; the corresponding
routed NIAH-S3 A/B retained Qwen at 8/8 and OLMo at 6/8.  Static cohort
membership and exact-leaf attention are unchanged.

- `qwen38_direct_{off,auto}_64k_b8_r3.json`
- `olmo_direct_{off,auto}_64k_b8_r3.json`
- `muse_static_off_64k_b8_r3.json`
- `qwen08_static_64k_profile.json`

The failed static-overlap run is intentionally not reported as a timing
result.  Python compilation and `git diff --check` pass for the retained code.
