# K2 Horizon c=4 supplement at 64K

Same frozen 32 prompts in all 13 tasks, c=16 control, 4,096-query internal tile, and 16,384-token exact prefix as the [five-setting sweep](../k2-threshold-ablation-v1/REPORT.md).

| Prefill c | Decode c | 13-task mean | Paired delta vs (16,16), points | Paired 95% interval |
|---:|---:|---:|---:|---:|
| 16 | 16 | 93.22% | — | — |
| 16 | 4 | 93.12% | -0.10 | [-0.76, +0.31] |
| 4 | 16 | 93.12% | -0.10 | [-0.97, +0.57] |

| Task | (16, 16) | (16, 4) | (4, 16) |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 98.44% | 99.22% |
| ruler_cwe | 94.69% | 95.94% | 90.00% |
| ruler_fwe | 86.46% | 85.42% | 87.50% |
| ruler_qa_hotpot | 65.62% | 65.62% | 65.62% |
| ruler_qa_squad | 68.23% | 68.23% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% |

[Settings](settings.json) · [Jobs](jobs.json) · [Memory check](preflight/jobs.json) · [Scores](REPORT.json)
