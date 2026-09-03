# Static-cohort two-level LOD prefill panel

> **Superseded timing notice (2026-08-26):** the large-model static timings
> below predate subsequent shared prefill/cache fixes and native-GQA coarse
> packing.  See `../static_prefill_recent_20260826/README.md` for matched current
> Qwen, OLMo, and Muse results.  In particular, Qwen is now 69.648 seconds and
> OLMo is 70.901 seconds rather than 87.652 and 97.306 seconds.

## Setup

This panel compares query-independent static-cohort prefill against the current
query-dependent routed two-level prefill and full attention. Despite the
historical `top8` artifact names and `VLLM_LOD_OPEN_COUNT=8`, the vLLM pool
sets `prefill_two_level_topk = min(3, open_count)`, so these prefill controls
are effective top-3. Decode remains top-8. The static cap is

```text
max(16, ceil(sqrt(T) / 16))
```

All contexts in this panel are at or below 64K, so the low-end floor makes the
effective cap 16. The static table is rebuilt after each 4,096-token state
catch-up. Runs use batch 8, real distinct ProLong documents, 16K scheduler
chunks, BF16 LOD storage, and the median of three measured repetitions. Chat
templates and model-specific options match the previously validated panel.

## Qwen3.5-0.8B context sweep

Lower prefill time is better. `static delta` is `(static / routed top-3 - 1)`; the
last column is full-attention time divided by static time.

| context | static cohort | routed top-3 LOD | full attention | static delta | static speedup vs full |
|---:|---:|---:|---:|---:|---:|
| 8K | 0.403 s | 0.475 s | **0.387 s** | -15.2% | 0.96x |
| 16K | 1.006 s | **0.973 s** | 1.013 s | +3.4% | 1.01x |
| 32K | **2.317 s** | 2.364 s | 2.991 s | -2.0% | 1.29x |
| 64K (prior matched run) | **4.732 s** | 5.168 s | 9.930 s | -8.4% | 2.10x |

The 8K and 16K ordering is small and not monotonic. By 32K static is stable,
slightly faster than routed top-3, and 1.29x faster than full attention. The prior
64K matched run remains the clearer win: 8.4% faster than routed top-3 and 2.10x
faster than full attention, with 64/64 NIAH-S3.

## 64K large-model panel

The new `static cohort` column is a fresh post-cache-fix run. The routed top-3 and
full-attention columns are the post-cache-fix controls recorded in
`../static_vs_top8_30b_20260825/README.md`; full-attention reuse was permitted
for this panel.

| model | static cohort | routed top-3 LOD | full attention | static delta | static speedup vs full |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 87.652 s | **81.190 s** | 110.565 s | +8.0% | 1.26x |
| Gemma-4-26B-A4B | 33.495 s | **18.909 s** | 40.063 s | +77.1% | 1.20x |
| Phi-4 TP5 | 31.336 s | 32.930 s | **28.119 s** | -4.8% | 0.90x |
| Muse-Glimmer-30B | 54.885 s | 56.508 s | **51.933 s** | -2.9% | 0.95x |
| OLMo-3-1125-32B | 97.306 s | 74.457 s | **67.892 s** | +30.7% | 0.70x |

Static selection modestly improves on routed top-3 for Phi and Muse, but neither
beats native full attention. It loses to routed top-3 on Qwen3.8 and OLMo. OLMo's
97.306 s result independently reproduces the prior anomalous 97.710 s arm, so
that slowdown is real rather than a one-off measurement error.

Gemma needs a separate qualification. AITER batch prefill rejects D=512, so
this work adds `_static_cap_wide_indexed_prefill_kernel`, a regular M16/N16
indexed QK/PV kernel over the static cohort. It agrees with dense reference
attention (maximum BF16 output error 0.0078125 and LSE error below 1e-6), but
is not yet optimized to the old routed kernel's speed. Its 33.495 s result is
still 1.20x faster than full attention, but the 77% loss to routed top-3 is currently
a D=512 kernel implementation deficit, not evidence in favor of static
routing on Gemma.

Overall, static-cohort prefill is attractive on the small Qwen at 32K--64K,
but is not a uniform replacement for routed top-3 across the large-model set.

## Execution checks

Every new LOD artifact records:

- `prefill_static_leaf_aiter: true`;
- `static_prefill_layers_executed > 0`;
- observed cap minimum and maximum both equal to 16; and
- real, non-repeating ProLong prompt hashes and block-uniqueness diagnostics.

The standalone verifier covers inline pages, the two-level page directory,
buffer reuse across catch-up shapes, and the D=512 path against dense
attention.

## Artifacts

- `qwen08_static_8k_16k_32k_b8_r3.json`
- `qwen08_top8_8k_16k_32k_b8_r3.json`
- `qwen08_full_8k_16k_32k_b8_r3.json`
- `qwen38_static_prefill_64k_b8_r3.json`
- `gemma4_static_prefill_64k_b8_r3.json`
- `phi4_tp5_static_prefill_64k_b8_r3.json`
- `muse_static_prefill_64k_b8_r3.json`
- `olmo_static_prefill_64k_b8_r3.json`

The excluded `10817` Gemma attempt failed before inference because AITER CK
does not support batch-prefill head dimensions above 256. Run `10823` uses the
validated D=512 fallback and is the result reported above.
