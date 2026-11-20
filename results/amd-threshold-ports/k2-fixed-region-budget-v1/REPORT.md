# K2 Horizon normalized fixed region budgets at 64K

Two-tier BF16 KV, batch one, same 32 examples in each of 13 RULER tasks. Top-4, top-8 and top-16 use the same RMS-normalized proxy score and 1,024-entry eligibility cap as the c=16 threshold. Each condition changes one phase. The other phase remains at c=16. Ties select the smaller region index first. Attention and cache updates share the threshold implementation.

| Setting | RULER mean | Paired delta vs c=16, points | Paired 95% interval |
|---|---:|---:|---:|
| Threshold prefill and decode | 93.06% | — | — |
| prefill-top4 | 90.31% | -2.76 | [-4.27, -1.41] |
| decode-top4 | 90.05% | -3.01 | [-4.58, -1.51] |
| prefill-top8 | 92.27% | -0.80 | [-1.97, +0.24] |
| decode-top8 | 92.64% | -0.42 | [-1.22, +0.20] |
| prefill-top16 | 92.05% | -1.02 | [-2.02, -0.20] |
| decode-top16 | 92.88% | -0.18 | [-0.73, +0.18] |

| Task | threshold | prefill-top4 | decode-top4 | prefill-top8 | decode-top8 | prefill-top16 | decode-top16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 96.88% | 100.00% | 96.88% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 96.88% | 84.38% | 87.50% | 93.75% | 96.88% | 93.75% | 96.88% |
| niah_multikey_3 | 100.00% | 90.62% | 81.25% | 100.00% | 93.75% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 97.66% | 96.09% | 96.09% | 96.88% | 98.44% | 99.22% | 98.44% |
| ruler_cwe | 95.00% | 94.06% | 92.81% | 93.75% | 95.00% | 93.75% | 95.00% |
| ruler_fwe | 86.46% | 85.42% | 85.42% | 87.50% | 86.46% | 83.33% | 86.46% |
| ruler_qa_hotpot | 65.62% | 59.38% | 62.50% | 59.38% | 65.62% | 59.38% | 62.50% |
| ruler_qa_squad | 68.23% | 67.19% | 65.10% | 71.35% | 68.23% | 67.19% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

[GPU checks](preflight/jobs.json) · [Settings](settings.json) · [Commands](jobs.json) · [Scores](REPORT.json)
