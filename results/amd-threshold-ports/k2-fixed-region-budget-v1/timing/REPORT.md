# K2 normalized fixed region budget timings at 64K

K2 Horizon FP8, two-tier BF16 KV, batch one. A fixed budget replaces c=16 in one phase. The other phase remains at c=16. Each pair uses the same MI325X and ProLong prompt, one warmup and three measured repeats per setting. Decode includes 1,024 steps and periodic cache updates. The continuation is generated greedily and can differ by method.

Prefill is time from scheduling to the first token. Work deferred into the first decode step counts toward decode. These timings use the fixed-budget selectors from the matched quality panel and characterize those benchmark implementations.

The full c=16 quality control scored 93.06% on the matched 32-example-per-task RULER panel.

| Fixed rule | RULER mean | Prefill, candidate / control (s) | Prefill change | Decode, candidate / control (ms/token) | Decode change |
|---|---:|---:|---:|---:|---:|
| Prefill top-4 | 90.31% | 15.30 / 15.67 | -2.4% | 38.78 / 38.75 | +0.1% |
| Prefill top-8 | 92.27% | 16.38 / 15.92 | +2.9% | 38.99 / 38.89 | +0.3% |
| Prefill top-16 | 92.05% | 17.91 / 15.76 | +13.7% | 38.67 / 38.78 | -0.3% |
| Decode top-4 | 90.05% | 15.77 / 15.80 | -0.2% | 39.54 / 39.11 | +1.1% |
| Decode top-8 | 92.64% | 15.76 / 15.78 | -0.1% | 40.00 / 38.99 | +2.6% |
| Decode top-16 | 92.88% | 15.74 / 15.78 | -0.2% | 40.35 / 39.07 | +3.3% |

[Quality](../REPORT.md) · [Samples](REPORT.json) · [Timing commands](jobs.json)
