# Latest five-family timing panel (batch 8)

This is a speed-only refresh using the current three-tier BF16 LOD kernels.
Each point has one warmup followed by the median of three runs. Prefill is
aggregate prompt tokens/s; decode is milliseconds per batch-eight step.
Cells below are `Full / LOD (LOD speedup)`. Prefill speedup is `LOD / Full`;
decode speedup is `Full / LOD`.

Common settings are batch 8, 64 generated timing tokens, a 16,384-token vLLM
scheduler budget, 4,096-token LOD prefill chunks, 1,280-token state updates,
dense leaves, state factor 16, and BF16 LOD state. OLMo and Phi stop at 64K
because their prior 128K forced-extrapolation runs faulted reproducibly. Phi
uses TP5; every other model uses one MI300X GPU.

## Prefill throughput

| Model | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 35,741 / 31,341 (0.88x) | 28,669 / 28,124 (0.98x) | 19,889 / 25,417 (1.28x) | 13,087 / 21,768 (1.66x) | 7,675 / 18,009 (2.35x) |
| Qwen3.8-27B-FP8 | 8,586 / 8,212 (0.96x) | 7,717 / 7,851 (1.02x) | 6,382 / 7,282 (1.14x) | 4,742 / 6,647 (1.40x) | 3,128 / 5,910 (1.89x) |
| Muse-Glimmer-30B | 11,744 / 10,505 (0.89x) | 11,434 / 9,974 (0.87x) | 10,930 / 9,442 (0.86x) | 10,086 / 8,499 (0.84x) | 8,771 / 7,247 (0.83x) |
| OLMo-3-1125-32B | 9,304 / 8,120 (0.87x) | 8,956 / 7,659 (0.86x) | 8,482 / 7,103 (0.84x) | 7,715 / 6,522 (0.85x) | - |
| Phi-4 | 24,724 / 18,334 (0.74x) | 23,482 / 11,018 (0.47x) | 21,432 / 9,838 (0.46x) | 18,627 / 8,586 (0.46x) | - |

## Decode latency

| Model | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 8.51 / 9.85 (0.86x) | 8.87 / 9.97 (0.89x) | 10.50 / 9.85 (1.07x) | 11.61 / 10.04 (1.16x) | 14.30 / 9.02 (1.59x) |
| Qwen3.8-27B-FP8 | 37.34 / 36.28 (1.03x) | 40.24 / 36.38 (1.11x) | 44.68 / 36.50 (1.22x) | 52.03 / 36.78 (1.41x) | 63.92 / 37.25 (1.72x) |
| Muse-Glimmer-30B | 18.64 / 23.50 (0.79x) | 18.72 / 23.41 (0.80x) | 18.98 / 23.21 (0.82x) | 19.29 / 23.06 (0.84x) | 19.49 / 22.46 (0.87x) |
| OLMo-3-1125-32B | 26.20 / 34.44 (0.76x) | 26.93 / 34.58 (0.78x) | 28.21 / 34.69 (0.81x) | 30.48 / 34.57 (0.88x) | - |
| Phi-4 | 7.40 / 13.31 (0.56x) | 7.82 / 12.71 (0.62x) | 8.61 / 11.98 (0.72x) | 9.97 / 9.78 (1.02x) | - |

## LOD dispatch

| Model | State route | Page scores | Exact full backend |
|---|---|---|---|
| Gemma | re-split normalized MFMA PV | materialized | Triton |
| Qwen | re-split normalized MFMA PV | legacy selector | AITER |
| Muse | grouped/fused | materialized | AITER |
| OLMo | re-split normalized MFMA PV | materialized | AITER |
| Phi | grouped/fused | materialized | AITER |

The grouped route remains faster for the D128/KV2 Muse and Phi geometries.
The re-split route is used for wide heads or high KV-head count. Qwen retains
the legacy page selector because materializing all page scores was a measured
regression for its D256/GQA4 geometry.

Gemma's LOD local branch uses the existing D512 GQA-shared, AITER-style MFMA
specialization automatically. That kernel materializes only a fixed 512-token
local score field and therefore cannot directly replace the exact 128K global
attention backend. Gemma's 128K LOD decode result was repeated independently:
9.019 ms initially and 9.034 ms in the repeat, so the non-monotonic point is
not timing noise.

## Change from the previous two-tier panel

At 64K, the refreshed LOD decode latency versus the August 20 two-tier panel
is 246.77 -> 10.04 ms for Gemma, 45.88 -> 36.78 ms for Qwen, 248.44 -> 23.06
ms for Muse, 333.38 -> 34.57 ms for OLMo, and 16.89 -> 9.78 ms for Phi. These
large family-dependent changes are the cumulative effect of moving to the
current three-tier design and all subsequent local, page, routing, and PV
kernels; they must not be attributed to normalized PV fusion alone.

Source artifacts are the paired `*_full.json` and `*_lod.json` files in this
directory. `gemma4_lod_128k_repeat.json` is the independent Gemma repeat.
