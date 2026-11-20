# AMD threshold prefill and decode quality at 64K

All 13 RULER tasks, 32 matched examples per task, two-tier BF16 KV, batch one. Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8 use c=16 for threshold routing. The initial exact prefix and scheduler chunks remain 16,384 tokens. Qwen threshold decode includes its probability-residual correction. The three fresh arms share one source freeze and the packed-max cache update.

Dense is an archived control with identical requests and targets, from a different runtime. The frozen first-32 cohort has been used in earlier validation; these results are not an independent holdout. All-task means weight tasks equally and retain fractional recall. Quality-run elapsed times include setup effects and are not speed-benchmark results.

| Model | Archived dense | Top-8 throughout | Threshold decode only | Threshold prefill + decode |
|---|---:|---:|---:|---:|
| QWEN | 96.15 | 95.41 | 94.47 | 95.39 |
| K2 | 93.23 | 90.08 | 92.24 | 93.37 |

## QWEN per-task results

| Task | Archived dense | Top-8 throughout | Threshold decode only | Threshold prefill + decode |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_1 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_2 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_3 | 100.00 | 100.00 | 96.88 | 100.00 |
| niah_multiquery | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multivalue | 100.00 | 99.22 | 98.44 | 99.22 |
| ruler_cwe | 99.69 | 100.00 | 100.00 | 100.00 |
| ruler_fwe | 100.00 | 100.00 | 100.00 | 100.00 |
| ruler_qa_hotpot | 75.00 | 68.75 | 59.38 | 68.75 |
| ruler_qa_squad | 75.26 | 72.40 | 73.44 | 72.14 |
| ruler_vt | 100.00 | 100.00 | 100.00 | 100.00 |

Top-8 prefill / c=16 decode minus Top-8 prefill / top-8 decode: -0.94 points; 2 improved / 7 declined examples. Paired bootstrap 95% interval [-2.14, +0.16] points.

c=16 prefill / c=16 decode minus Top-8 prefill / top-8 decode: -0.02 points; 3 improved / 3 declined examples. Paired bootstrap 95% interval [-0.98, +0.94] points.

c=16 prefill / c=16 decode minus Top-8 prefill / c=16 decode: +0.92 points; 7 improved / 2 declined examples. Paired bootstrap 95% interval [-0.16, +2.16] points.

## K2 per-task results

| Task | Archived dense | Top-8 throughout | Threshold decode only | Threshold prefill + decode |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 96.88 | 100.00 | 100.00 |
| niah_multikey_1 | 100.00 | 96.88 | 96.88 | 100.00 |
| niah_multikey_2 | 100.00 | 84.38 | 90.62 | 100.00 |
| niah_multikey_3 | 100.00 | 93.75 | 100.00 | 100.00 |
| niah_multiquery | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multivalue | 98.44 | 99.22 | 99.22 | 98.44 |
| ruler_cwe | 92.19 | 92.19 | 95.94 | 94.06 |
| ruler_fwe | 86.46 | 83.33 | 86.46 | 87.50 |
| ruler_qa_hotpot | 65.62 | 59.38 | 62.50 | 65.62 |
| ruler_qa_squad | 69.27 | 65.10 | 67.45 | 68.23 |
| ruler_vt | 100.00 | 100.00 | 100.00 | 100.00 |

Top-8 prefill / c=16 decode minus Top-8 prefill / top-8 decode: +2.15 points; 14 improved / 2 declined examples. Paired bootstrap 95% interval [+0.93, +3.49] points.

c=16 prefill / c=16 decode minus Top-8 prefill / top-8 decode: +3.29 points; 21 improved / 6 declined examples. Paired bootstrap 95% interval [+1.76, +5.04] points.

c=16 prefill / c=16 decode minus Top-8 prefill / c=16 decode: +1.14 points; 9 improved / 7 declined examples. Paired bootstrap 95% interval [+0.18, +2.33] points.

Thirty-two examples per task can expose large differences but cannot establish small improvements reliably. Bootstrap intervals resample within each task with 2,000 draws and seed zero.

K2 ruler_cwe was rerun in full after a scratch allocation failed. The successful run used an internal 8,192-query tile with the same threshold and 16,384-token exact prefix.

[Settings](settings.json) · [Source](source-manifest.json) · [Requests](cohort-manifest.json) · [Token audit](token-audit.json) · [Commands](jobs.json) · [Exit statuses](provenance.json) · [Changed predictions](changed-examples.json) · [Scores](REPORT.json)
