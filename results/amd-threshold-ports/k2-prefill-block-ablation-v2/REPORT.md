# K2 coupled prefill block ablation at 64K

K2 Horizon FP8, two-tier BF16 KV, batch one, region c=16 in both phases. The scheduler chunk, length of the first exact chunk and prefill cache catch-up move together. The subsequent exact lookback is 256 tokens and the internal query tile is 4,096. Both speed pairs use one GPU and the same prompt with three warmed repeats. Two-token latency includes time through the first decode token, including any deferred cache construction and one decode step.

| Block vs 16K | First token, candidate / control (s) | Two tokens, candidate / control (s) | Peak PyTorch reserved, candidate / control (GiB) |
|---|---:|---:|---:|
| 8k | 16.400 / 15.639 | 16.439 / 15.678 | 113.75 / 117.69 |
| 32k | 15.688 / 15.735 | 15.727 / 15.774 | 143.26 / 117.69 |

Memory is the worker’s PyTorch allocation high-water mark, including setup and warmup. It excludes memory managed outside PyTorch.

| Block | Prompt loss | Loss delta vs 16K | Perplexity |
|---|---:|---:|---:|
| 8k | 0.495659 | -0.000058 | 1.641579 |
| 16k | 0.495716 | +0.000000 | 1.641674 |
| 32k | 0.496268 | +0.000552 | 1.642580 |

The first launch initialized ROCm before vLLM forked its worker. The completed runs read the same block size directly from the environment. [First attempts](../k2-prefill-block-ablation-v1/jobs.json).

[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)
