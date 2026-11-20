# K2 coupled prefill block ablation at 64K

These 512-token results use the repaired cache handoff. The initial exact tail is copied into the page archive before the cross-layer cached update consumes it. [Cache audit](CACHE_HANDOFF_AUDIT.md).

K2 Horizon FP8, two-tier BF16 KV, batch one, region c=16 in both phases. The scheduler chunk, length of the first exact chunk and prefill cache catch-up move together. The subsequent exact lookback is 256 tokens and the internal query tile limit is 4,096. Each speed pair uses one GPU and the same prompt with three warmed repeats. Two-token latency includes time through the first decode token, including any deferred cache construction and one decode step.

| Block vs 16K | First token, candidate / control (s) | Two tokens, candidate / control (s) | Peak PyTorch reserved, candidate / control (GiB) |
|---|---:|---:|---:|
| 512 | 26.104 / 15.666 | 26.142 / 15.705 | 69.95 / 117.69 |

Memory is the worker’s PyTorch allocation high-water mark, including setup and warmup. It excludes memory managed outside PyTorch.

| Block | Prompt loss | Loss delta vs 16K | Perplexity |
|---|---:|---:|---:|
| 512 | 0.495943 | +0.000222 | 1.642045 |
| 16k | 0.495721 | +0.000000 | 1.641681 |

[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)
