# K2 Horizon 128-token BSWA timings at 64K

Each timing pair runs on one GPU in sequence, with one warmup and two measured requests per setting on the same 64K ProLong prompt. Decode timing covers 1,024 steps with normal cache catch-up. Greedy continuations can differ. Prefill is time to first token. Any prefill work deferred to the first decode step is included in decode.

| Candidate | Exact local tokens | Prefill, candidate / control (s) | Decode, candidate / control (ms/token) | Decode delta |
|---|---:|---:|---:|---:|
| b128-w128 | 1–128 | 15.73 / 15.67 | 38.75 / 38.74 | +0.0% |
| b128-w256 | 129–256 | 15.80 / 15.76 | 38.73 / 38.84 | -0.3% |
| b128-w384 | 257–384 | 15.92 / 15.91 | 38.97 / 39.22 | -0.7% |
| b128-w512 | 385–512 | 15.78 / 15.76 | 38.85 / 38.89 | -0.1% |

[Timing commands](preflight/jobs.json)
