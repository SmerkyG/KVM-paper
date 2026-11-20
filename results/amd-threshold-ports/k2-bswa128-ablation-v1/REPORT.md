# K2 Horizon 128-token BSWA decode blocks at 64K

All 13 RULER tasks use the same 32 requests per setting. K2 Horizon FP8, two-tier BF16 KV, c=16 in both phases and batch one. Runs keep 16,384-token prefill chunks and the 16,384-token exact prefix. Prefix caching is disabled.

| BSWA block and catch-up | Configured decode window | RULER mean | Paired delta vs b256-w512, points | 95% interval |
|---:|---:|---:|---:|---:|
| 256 | 512 | 92.97% | — | — |
| 128 | 128 | 92.38% | -0.58 | [-1.50, +0.13] |
| 128 | 256 | 93.05% | +0.08 | [+0.00, +0.23] |
| 128 | 384 | 93.13% | +0.16 | [-0.02, +0.42] |
| 128 | 512 | 93.01% | +0.04 | [-0.19, +0.26] |

| Task | b256-w512 | b128-w128 | b128-w256 | b128-w384 | b128-w512 |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 96.88% | 93.75% | 96.88% | 96.88% | 96.88% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 97.66% | 99.22% | 98.44% | 98.44% | 97.66% |
| ruler_cwe | 93.75% | 88.75% | 94.06% | 94.06% | 95.31% |
| ruler_fwe | 86.46% | 85.42% | 86.46% | 87.50% | 85.42% |
| ruler_qa_hotpot | 65.62% | 65.62% | 65.62% | 65.62% | 65.62% |
| ruler_qa_squad | 68.23% | 68.23% | 68.23% | 68.23% | 68.23% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Each timing pair runs on one GPU in sequence, with one warmup and two measured requests per setting on the same 64K ProLong prompt. Decode timing covers 1,024 steps with normal cache catch-up. Greedy continuations can differ. Prefill is time to first token. Any prefill work deferred to the first decode step is included in decode.

| Candidate | Exact local tokens | Prefill, candidate / control (s) | Decode, candidate / control (ms/token) | Decode delta |
|---|---:|---:|---:|---:|
| b128-w128 | 1–128 | 15.73 / 15.67 | 38.75 / 38.74 | +0.0% |
| b128-w256 | 129–256 | 15.80 / 15.76 | 38.73 / 38.84 | -0.3% |
| b128-w384 | 257–384 | 15.92 / 15.91 | 38.97 / 39.22 | -0.7% |
| b128-w512 | 385–512 | 15.78 / 15.76 | 38.85 / 38.89 | -0.1% |

[Settings](settings.json) · [Serving checks](preflight/jobs.json) · [Commands](jobs.json) · [Scores](REPORT.json)
