# K2 Horizon region threshold ablation at 64K

K2-Horizon-32B-FP8 on MI325X, batch one, two-tier BF16 KV. Each configuration evaluates the same first 32 archived examples of every RULER task with greedy generation. All runs use a 16,384-token exact prefix, 16,384-token scheduler chunks and 4,096-query internal tiles. A c=8 prefill smoke test covered the CWE prompt from the previous scratch allocation failure and three of the longest prompts in the cohort.

| Prefill c | Decode c | 13-task mean | Delta vs (16,16), points | Paired 95% interval, points | Improved / declined |
|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 93.22% | — | — | — |
| 16 | 8 | 92.91% | -0.31 | [-1.15, +0.30] | 6 / 4 |
| 16 | 32 | 93.04% | -0.18 | [-0.85, +0.25] | 5 / 6 |
| 8 | 16 | 93.16% | -0.06 | [-0.67, +0.41] | 8 / 4 |
| 32 | 16 | 92.43% | -0.79 | [-1.72, -0.06] | 6 / 9 |

| Task | (16, 16) | (16, 8) | (16, 32) | (8, 16) | (32, 16) |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 96.88% | 100.00% | 93.75% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 98.44% | 96.88% | 98.44% | 97.66% |
| ruler_cwe | 94.69% | 95.31% | 94.38% | 93.12% | 94.06% |
| ruler_fwe | 86.46% | 86.46% | 87.50% | 87.50% | 85.42% |
| ruler_qa_hotpot | 65.62% | 62.50% | 65.62% | 65.62% | 62.50% |
| ruler_qa_squad | 68.23% | 68.23% | 68.23% | 66.41% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

The mean weights the 13 tasks equally. Intervals resample paired scores within each task and describe uncertainty on this cohort. The cohort was also used for the preceding threshold quality comparison.

[Settings](settings.json) · [Memory check](preflight/jobs.json) · [Commands](jobs.json) · [Exit statuses](provenance.json) · [Scores](REPORT.json)
