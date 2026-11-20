# K2 coupled prefill block ablation at 64K

The original 512-token measurements were affected by a BF16 cache handoff bug. The [corrected 512-token rerun](../k2-prefill-block-512-fixed-v1/REPORT.md) supersedes those measurements. [Audit](../k2-prefill-block-512-fixed-v1/CACHE_HANDOFF_AUDIT.md).

K2 Horizon FP8, two-tier BF16 KV, batch one, region c=16 in both phases. The scheduler chunk, length of the first exact chunk and prefill cache catch-up move together. The subsequent exact lookback is 256 tokens and the internal query tile limit is 4,096. Each speed pair uses one GPU and the same prompt with three warmed repeats. Two-token latency includes time through the first decode token, including any deferred cache construction and one decode step.

| Block vs 16K | First token, candidate / control (s) | Two tokens, candidate / control (s) | Peak PyTorch reserved, candidate / control (GiB) |
|---|---:|---:|---:|
| 512 | 25.177 / 15.756 | 25.214 / 15.795 | 70.07 / 117.69 |
| 1k | 21.850 / 15.749 | 21.887 / 15.787 | 75.92 / 117.69 |

Memory is the worker’s PyTorch allocation high-water mark, including setup and warmup. It excludes memory managed outside PyTorch.

| Block | Prompt loss | Loss delta vs 16K | Perplexity |
|---|---:|---:|---:|
| 512 | 0.512376 | +0.016643 | 1.669252 |
| 1k | 0.495920 | +0.000187 | 1.642008 |
| 16k | 0.495732 | +0.000000 | 1.641700 |

[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)
