# K2 Horizon c=64 supplement at 64K

Same frozen 32 prompts in all 13 tasks, c=16 control, 4,096-query internal tile, and 16,384-token exact prefix as the [five-setting sweep](../k2-threshold-ablation-v1/REPORT.md).

| Prefill c | Decode c | 13-task mean | Paired delta vs (16,16), points | Paired 95% interval |
|---:|---:|---:|---:|---:|
| 16 | 16 | 93.22% | — | — |
| 16 | 64 | 91.44% | -1.78 | [-3.08, -0.68] |
| 64 | 16 | 91.30% | -1.92 | [-3.49, -0.48] |

| Task | (16, 16) | (16, 64) | (64, 16) |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 96.88% |
| niah_multikey_2 | 100.00% | 96.88% | 84.38% |
| niah_multikey_3 | 100.00% | 87.50% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 97.66% | 96.09% |
| ruler_cwe | 94.69% | 94.69% | 96.56% |
| ruler_fwe | 86.46% | 84.38% | 85.42% |
| ruler_qa_hotpot | 65.62% | 62.50% | 59.38% |
| ruler_qa_squad | 68.23% | 65.10% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% |

[Settings](settings.json) · [Jobs](jobs.json) · [Scores](REPORT.json)
