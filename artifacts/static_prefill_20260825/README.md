# Static two-level LoD prefill

This experiment replaces query-dependent top-8 leaf routing during prefill with
a query-independent cohort:

```text
leaf cap(T) = max(16, ceil(sqrt(T) / 16))
```

Every leaf of a centroid at or below the cap is concatenated into one
page-size-one AITER varlen sequence per batch/KV-head row. Centroids above the
cap remain in the coarse branch. The cohort table is rebuilt after each
4,096-token state catch-up, so a long scheduler prefill never continues using a
table from stale centroid membership.

## Qwen3.5-0.8B, 64k, batch 8

All speed rows use the same current worktree, chat template, thinking disabled,
16k vLLM scheduler budget, 4k long-prefill threshold, real distinct ProLong
documents, BF16 LoD storage, and three measured repeats. Full attention uses
`ROCM_AITER_UNIFIED_ATTN`.

| Prefill path | Median prefill | Prompt tok/s | Relative to static |
|---|---:|---:|---:|
| Static cohort | 4.732 s | 110,789 | 1.00x |
| Top-8 two-level LoD | 5.168 s | 101,454 | 1.092x slower |
| Full attention | 9.930 s | 52,799 | 2.098x slower |

Static timings were `[4.7362, 4.7323, 4.7239]` seconds. At 64k the scheduled
cap is 16 and the observed exact cohort contained 19,109--24,223 leaves per KV
row across calls. The static path still scored **64/64 NIAH-S3 at 64k**.

Artifacts:

- `qwen08_static_prefill_64k_b8_r3.json`
- `qwen08_top8_prefill_64k_current_b8_r3.json`
- `qwen08_full_64k_current_b8_r3.json`
- `qwen08_static_prefill_64k_niah64.json`

`qwen08_static_prefill_64k_fixed_b8_r1.json` was the bounds-checked diagnostic
run. Its 6.157 s timing is intentionally excluded: index validation forces
multiple GPU/CPU synchronizations on every static-prefill call.

## Correctness regression

`scripts/verify_static_cap_aiter_prefill.py` checks both inline and two-level
page-directory metadata against dense PyTorch attention. It also reproduces the
vLLM transition from a pool-sized catch-up buffer to a smaller initial-prefill
batch. The latter caught a non-contiguous narrowed metadata view that caused
rows after the first to use the wrong physical stride; the static buffer
allocator now replaces such a view with compact storage.
