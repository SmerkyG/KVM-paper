# K2 Horizon 512-token BSWA decode blocks at 64K

All 13 RULER tasks use the same 32 requests per setting. K2 Horizon FP8, two-tier BF16 KV, c=16 in both phases and batch one. Runs keep 16,384-token prefill chunks and the 16,384-token exact prefix. Prefix caching is disabled.

| BSWA block and catch-up | Configured decode window | RULER mean | Paired delta vs b256-w512, points | 95% interval |
|---:|---:|---:|---:|---:|
| 256 | 512 | 93.29% | — | — |
| 512 | 512 | 92.86% | -0.42 | [-1.09, -0.01] |
| 512 | 1024 | 93.10% | -0.19 | [-0.86, +0.20] |
| 512 | 1536 | 93.08% | -0.21 | [-0.87, +0.22] |
| 512 | 2048 | 93.04% | -0.24 | [-0.93, +0.18] |

| Task | b256-w512 | b512-w512 | b512-w1024 | b512-w1536 | b512-w2048 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 96.88% | 96.88% | 96.88% | 96.88% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 97.66% | 97.66% | 98.44% | 98.44% | 98.44% |
| ruler_cwe | 93.75% | 93.44% | 94.69% | 94.38% | 95.00% |
| ruler_fwe | 87.50% | 85.42% | 86.46% | 86.46% | 85.42% |
| ruler_qa_hotpot | 65.62% | 65.62% | 65.62% | 65.62% | 65.62% |
| ruler_qa_squad | 68.23% | 68.23% | 68.23% | 68.23% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Each timing pair runs on one GPU in sequence, with one warmup and two measured requests per setting on the same 64K ProLong prompt. Decode timing covers 1,024 steps with normal cache catch-up. Greedy continuations can differ. Prefill is time to first token. Any prefill work deferred to the first decode step is included in decode.

| Candidate | Exact local tokens | Prefill, candidate / control (s) | Decode, candidate / control (ms/token) | Decode delta |
|---|---:|---:|---:|---:|
| b512-w512 | 1–512 | 15.73 / 15.67 | 38.76 / 38.83 | -0.2% |
| b512-w1024 | 513–1024 | 15.78 / 15.78 | 38.74 / 38.94 | -0.5% |
| b512-w1536 | 1025–1536 | 15.97 / 15.91 | 39.07 / 38.95 | +0.3% |
| b512-w2048 | 1537–2048 | 15.78 / 15.75 | 38.81 / 38.85 | -0.1% |

[Settings](settings.json) · [Serving checks](preflight/jobs.json) · [Commands](jobs.json) · [Scores](REPORT.json)
