# K2 three-tier page-budget diagnostic at 64K

All runs use native top-eight prefill plus c=16 region routing in decode. The retrieval diagnostic uses 32 matched requests per task and BF16 KV. Opened-work counts average across query heads, layers, and the last decode step of each request. The two-tier row opens regions directly.

| Decode refinement | Multikey-2 | Multikey-3 | Exact pages / head | Exact remote entries / head |
|---|---:|---:|---:|---:|
| two-tier | 90.62% | 100.00% | — | 2945.9 |
| fixed-2 | 87.50% | 93.75% | 39.0 | 465.7 |
| fixed-4 | 90.62% | 100.00% | 62.9 | 826.6 |
| adaptive-0.5 | 90.62% | 100.00% | 164.7 | 2457.9 |

Same-GPU speed pairs use one matched 64K prompt, one warmup and three repeats. Decode includes 1,024 timed steps and cache catch-up. The timing runs use the unmodified serving work for each arm, with fixed-page and two-tier diagnostic counters disabled.

| Refinement | Decode, candidate / two-tier (ms/token) | Decode change |
|---|---:|---:|
| fixed-2 | 41.50 / 38.66 | +7.3% |
| fixed-4 | 46.26 / 38.83 | +19.1% |
| adaptive-0.5 | 43.20 / 38.96 | +10.9% |

[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)
