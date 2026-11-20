# K2 Horizon 512-token BSWA timings at 64K

Each timing pair runs on one GPU in sequence, with one warmup and two measured requests per setting on the same 64K ProLong prompt. Decode timing covers 1,024 steps with normal cache catch-up. Greedy continuations can differ. Prefill is time to first token. Any prefill work deferred to the first decode step is included in decode.

| Candidate | Exact local tokens | Prefill, candidate / control (s) | Decode, candidate / control (ms/token) | Decode delta |
|---|---:|---:|---:|---:|
| b512-w512 | 1–512 | 15.73 / 15.67 | 38.76 / 38.83 | -0.2% |
| b512-w1024 | 513–1024 | 15.78 / 15.78 | 38.74 / 38.94 | -0.5% |
| b512-w1536 | 1025–1536 | 15.97 / 15.91 | 39.07 / 38.95 | +0.3% |
| b512-w2048 | 1537–2048 | 15.78 / 15.75 | 38.81 / 38.85 | -0.1% |

[Timing commands](preflight/jobs.json)
