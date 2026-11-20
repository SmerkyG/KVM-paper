# K2 Horizon centroid count and decode window at 64K

All 13 RULER tasks use the same 32 requests per setting. K2 Horizon FP8, two-tier BF16 KV, c=16 in both phases and batch one. Runs keep the 16,384-token exact prefix and 256-token decode catch-up. Prefix caching is disabled.

| Centroid multiplier | Configured decode window | RULER mean | Paired delta vs (16,512), points | 95% interval |
|---:|---:|---:|---:|---:|
| 8 | 512 | 92.24% | -0.79 | [-1.73, -0.01] |
| 16 | 512 | 93.03% | — | — |
| 32 | 512 | 93.72% | +0.69 | [+0.09, +1.51] |
| 16 | 256 | 93.21% | +0.18 | [-0.18, +0.77] |
| 16 | 1024 | 92.84% | -0.19 | [-0.45, +0.02] |

| Task | m8-w512 | m16-w512 | m32-w512 | m16-w256 | m16-w1024 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 93.75% | 96.88% | 100.00% | 100.00% | 96.88% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 98.44% | 98.44% | 97.66% | 97.66% | 97.66% |
| ruler_cwe | 94.69% | 93.75% | 96.25% | 93.75% | 93.12% |
| ruler_fwe | 85.42% | 86.46% | 86.46% | 86.46% | 85.42% |
| ruler_qa_hotpot | 62.50% | 65.62% | 68.75% | 65.62% | 65.62% |
| ruler_qa_squad | 64.32% | 68.23% | 69.27% | 68.23% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

The serving checks used 1,024 decode steps on the same ProLong input. Their single measured repetitions on separate GPUs validate the configurations through multiple catch-up cycles and provide preliminary timings.

[Settings](settings.json) · [Serving checks](preflight/jobs.json) · [Commands](jobs.json) · [Scores](REPORT.json)
