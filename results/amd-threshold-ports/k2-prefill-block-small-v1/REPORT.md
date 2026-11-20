# K2 coupled prefill block ablation at 64K

K2 Horizon FP8, two-tier BF16 KV, batch one, region c=16 in both phases. The scheduler chunk, length of the first exact chunk and prefill cache catch-up move together. The subsequent exact lookback is 256 tokens and the internal query tile limit is 4,096. Each speed pair uses one GPU and the same prompt with three warmed repeats. Two-token latency includes time through the first decode token, including any deferred cache construction and one decode step.

| Block vs 16K | First token, candidate / control (s) | Two tokens, candidate / control (s) | Peak PyTorch reserved, candidate / control (GiB) |
|---|---:|---:|---:|
| 2k | 19.330 / 15.654 | 19.367 / 15.693 | 86.60 / 117.69 |
| 4k | 17.278 / 15.811 | 17.314 / 15.850 | 113.86 / 117.69 |

Memory is the worker’s PyTorch allocation high-water mark, including setup and warmup. It excludes memory managed outside PyTorch.

| Block | Prompt loss | Loss delta vs 16K | Perplexity |
|---|---:|---:|---:|
| 2k | 0.496246 | +0.000530 | 1.642544 |
| 4k | 0.495547 | -0.000169 | 1.641396 |
| 16k | 0.495716 | +0.000000 | 1.641674 |

[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)
