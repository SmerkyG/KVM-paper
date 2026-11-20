# K2 region mass threshold timings at 64K

K2 Horizon FP8, two-tier BF16 KV, batch one. The threshold multiplier changes in one phase and the other stays at c=16. Each pair uses the same MI325X and ProLong prompt, one warmup and three measured repeats per setting. Decode includes 1,024 steps and periodic cache updates. The continuation is generated greedily and can differ by method.

Prefill is time from scheduling to the first token. Work deferred into the first decode step counts toward decode.

The full c=16 quality control scored 93.22% on the matched 32-example-per-task RULER panel.

| Routing rule | RULER mean | Prefill, candidate / control (s) | Prefill change | Decode, candidate / control (ms/token) | Decode change |
|---|---:|---:|---:|---:|---:|
| Prefill c=4 | 93.12% | 25.32 / 16.03 | +58.0% | 39.11 / 39.25 | -0.4% |
| Decode c=4 | 93.12% | 15.65 / 15.65 | -0.0% | 40.18 / 38.94 | +3.2% |
| Prefill c=8 | 93.16% | 18.29 / 15.74 | +16.2% | 38.61 / 38.92 | -0.8% |
| Decode c=8 | 92.91% | 15.74 / 15.71 | +0.2% | 39.51 / 38.98 | +1.3% |
| Prefill c=32 | 92.43% | 15.06 / 15.97 | -5.7% | 38.89 / 38.91 | -0.1% |
| Decode c=32 | 93.04% | 15.79 / 15.78 | +0.1% | 38.67 / 38.97 | -0.8% |
| Prefill c=64 | 91.30% | 14.31 / 15.63 | -8.4% | 38.71 / 38.85 | -0.3% |
| Decode c=64 | 91.44% | 15.97 / 15.96 | +0.1% | 38.66 / 39.04 | -1.0% |

[Quality](../REPORT.md) · [c=4 quality](../../k2-threshold-c4-v1/REPORT.md) · [c=64 quality](../../k2-threshold-c64-v1/REPORT.md) · [Samples](REPORT.json) · [Timing commands](jobs.json)
