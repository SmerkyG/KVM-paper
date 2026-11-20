# K2 region mass threshold timings at 64K

K2 Horizon FP8, two-tier BF16 KV, batch one. The threshold multiplier changes in one phase and the other stays at c=16. The prefill pair uses 2,048-query internal tiles in both arms. The decode pair uses the original 4,096-query prefill tile. Each pair uses the same MI325X and ProLong prompt, one warmup and three measured repeats per setting. Decode includes 1,024 steps and periodic cache updates. The continuation is generated greedily and can differ by method.

Prefill is time from scheduling to the first token. Work deferred into the first decode step counts toward decode.

The full c=16 quality control scored 93.22% on the matched 32-example-per-task RULER panel.

| Routing rule | RULER mean | Prefill, candidate / control (s) | Prefill change | Decode, candidate / control (ms/token) | Decode change |
|---|---:|---:|---:|---:|---:|
| Prefill c=2 | 93.20% | 39.46 / 16.17 | +144.0% | 38.74 / 38.89 | -0.4% |
| Decode c=2 | 92.95% | 15.73 / 15.73 | +0.0% | 41.04 / 38.89 | +5.5% |

[Quality](../REPORT.md) · [Samples](REPORT.json) · [Timing commands](jobs.json)
