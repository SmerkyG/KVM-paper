# K2 Horizon c=2 supplement at 64K

Same frozen 32 prompts in all 13 tasks, c=16 control, and 16,384-token exact prefix as the [five-setting sweep](../k2-threshold-ablation-v1/REPORT.md).

The c=2 prefill arm uses a 2,048-query internal tile. The c=16 prefill and its decode-only sweep use the original 4,096-query tile.

| Prefill c | Decode c | 13-task mean | Paired delta vs (16,16), points | Paired 95% interval |
|---:|---:|---:|---:|---:|
| 16 | 16 | 93.22% | — | — |
| 16 | 2 | 92.95% | -0.27 | [-0.92, +0.14] |
| 2 | 16 | 93.20% | -0.02 | [-0.54, +0.40] |

| Task | (16, 16) | (16, 2) | (2, 16) |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 96.88% | 97.66% | 98.44% |
| ruler_cwe | 94.69% | 95.31% | 95.94% |
| ruler_fwe | 86.46% | 85.42% | 84.38% |
| ruler_qa_hotpot | 65.62% | 65.62% | 65.62% |
| ruler_qa_squad | 68.23% | 67.45% | 67.19% |
| ruler_vt | 100.00% | 100.00% | 100.00% |

[Settings](settings.json) · [Jobs](jobs.json) · [Scores](REPORT.json) · [Memory check](preflight/jobs.json)
