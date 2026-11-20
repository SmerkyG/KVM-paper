# K2 centroid ablation versus dense on matched RULER requests

At 64K, the 32√N run scored 93.72% and dense scored 93.23% on the same 32 requests for each of 13 tasks. The difference was +0.49 points (paired 95% bootstrap interval -0.14 to +1.29).

The two full-request wins account for nearly the entire macro-average difference. A fresh dense run reproduced the empty CWE response and the HotpotQA answer “Sergio Pérez”. The 32√N configuration recovered the CWE list and answered “Pedro Rodríguez” on the latter.

| Task with score changes | Mean change (points) | Improved prompts | Regressed prompts |
|---|---:|---:|---:|
| niah_multivalue | -0.78 | 1 | 2 |
| ruler_cwe | +4.06 | 7 | 3 |
| ruler_qa_hotpot | +3.12 | 1 | 0 |
| ruler_fwe | +0.00 | 1 | 1 |

[Matched samples](../quality32-final-v1/REPORT.md) · [Audit](DENSE_COMPARISON.json) · [Dense rerun command](../k2-prefill-512-audit-v2/jobs.json)
