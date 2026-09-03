# NIAH-S3 family panel (batch 8)

Each quality cell is `Full / LOD` correct out of 64. Speed uses one warmup and the median of three runs with a 16,384-token vLLM scheduler budget; prefill is aggregate prompt tokens/s and decode is milliseconds per batch-8 step (lower is better).

## Quality

| Model | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 64/64 / 64/64 | 64/64 / 64/64 | 64/64 / 64/64 | 64/64 / 62/64 | 64/64 / 62/64 |
| Qwen3.8-27B-FP8 | 64/64 / 64/64 | 64/64 / 64/64 | 64/64 / 64/64 | 64/64 / 64/64 | 64/64 / 64/64 |
| Muse-Glimmer-30B | 64/64 / 64/64 | 64/64 / 64/64 | 63/64 / 64/64 | 64/64 / 64/64 | 64/64 / 64/64 |
| OLMo-3-1125-32B | 64/64 / 64/64 | 64/64 / 64/64 | 64/64 / 60/64 | 64/64 / 54/64 | — / — |
| Phi-4 | 64/64 / 64/64 | 64/64 / 63/64 | 0/64 / 0/64 | 0/64 / 0/64 | — / — |

## Prefill throughput

Cells are `Full / LOD (LOD speedup)` aggregate prompt tok/s.

| Model | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 36,041 / 22,826 (0.63×) | 28,686 / 16,703 (0.58×) | 20,453 / 11,465 (0.56×) | 13,091 / 7,212 (0.55×) | 7,652 / 4,185 (0.55×) |
| Qwen3.8-27B-FP8 | 8,427 / 8,276 (0.98×) | 7,596 / 7,594 (1.00×) | 6,298 / 7,184 (1.14×) | 4,690 / 6,487 (1.38×) | 3,092 / 5,622 (1.82×) |
| Muse-Glimmer-30B | 11,277 / 8,976 (0.80×) | 10,955 / 6,533 (0.60×) | 10,824 / 4,992 (0.46×) | 10,013 / 3,462 (0.35×) | 8,807 / 2,175 (0.25×) |
| OLMo-3-1125-32B | 9,129 / 6,733 (0.74×) | 8,828 / 4,585 (0.52×) | 8,348 / 3,624 (0.43×) | 7,642 / 2,319 (0.30×) | — |
| Phi-4 | 24,555 / 17,642 (0.72×) | 23,336 / 11,185 (0.48×) | 21,331 / 10,154 (0.48×) | 18,574 / 9,062 (0.49×) | — |

## Decode latency

Cells are `Full / LOD (LOD speedup)` ms per batch-8 step.

| Model | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Gemma-4-26B-A4B-it | 8.6 / 38.1 (0.23×) | 8.7 / 68.0 (0.13×) | 10.3 / 127.5 (0.08×) | 11.7 / 246.8 (0.05×) | 14.1 / 484.8 (0.03×) |
| Qwen3.8-27B-FP8 | 37.1 / 38.2 (0.97×) | 40.2 / 39.4 (1.02×) | 44.5 / 41.2 (1.08×) | 52.0 / 45.9 (1.13×) | 64.0 / 52.5 (1.22×) |
| Muse-Glimmer-30B | 19.2 / 48.9 (0.39×) | 18.7 / 79.5 (0.24×) | 18.9 / 140.3 (0.14×) | 19.2 / 248.4 (0.08×) | 19.5 / 451.5 (0.04×) |
| OLMo-3-1125-32B | 25.9 / 64.4 (0.40×) | 26.7 / 104.1 (0.26×) | 27.9 / 181.4 (0.15×) | 30.4 / 333.4 (0.09×) | — |
| Phi-4 | 7.2 / 13.7 (0.53×) | 7.7 / 15.1 (0.51×) | 8.5 / 15.6 (0.55×) | 9.8 / 16.9 (0.58×) | — |

## Notes

- NIAH-S3 uses greedy 64-token generation, proper chat templates for instruction checkpoints, and raw prompting for base OLMo.
- LOD is the current two-tier BF16 design: state factor 16, top-8 routes, dense leaf storage, and automatic spherical/coherence-aware routing.
- Qwen weights use the requested FP8 checkpoint; activations and LOD state are BF16. Other checkpoints use BF16 weights/state.
- Phi-4 uses TP=5 for both modes because it has 10 KV heads. Other models use one GPU. Ratios are paired within a model; absolute Phi throughput is not cross-model comparable.
- At advertised context boundaries, the speed prompt reserves the 64 decode positions: 65,472 input tokens for OLMo 64K and 131,008 for Muse 128K.
- Gemma full attention uses Triton because AITER cannot support its 512-wide global heads. Other reported full rows use AITER.
- OLMo 128K and Phi 128K are unavailable for both modes after reproducible forced-extrapolation faults; no scores or timings are imputed.

Machine-readable consolidated data is in `panel.json`; source JSON files in this directory retain per-example responses and all timing repetitions.
