# K2 coupled prefill block ablation at 64K

This run uses the repaired BF16 cache handoff and a fresh 16K control. [Handoff audit](../prefill-handoff-paper-audit-v1/REPORT.md).

K2 Horizon FP8, two-tier BF16 KV, batch one, region c=16 in both phases. The scheduler chunk, length of the first exact chunk and prefill cache catch-up move together. The subsequent exact lookback is 256 tokens and the internal query tile limit is 4,096. Each speed pair uses one GPU and the same prompt with three warmed repeats. Two-token latency includes time through the first decode token, including any deferred cache construction and one decode step.

| Block vs 16K | First token, candidate / control (s) | Two tokens, candidate / control (s) | Peak PyTorch reserved, candidate / control (GiB) |
|---|---:|---:|---:|
| 256 | 33.775 / 15.788 | 33.812 / 15.827 | 66.63 / 117.69 |

Memory is the worker’s PyTorch allocation high-water mark, including setup and warmup. It excludes memory managed outside PyTorch.

| Block | Prompt loss | Loss delta vs 16K | Perplexity |
|---|---:|---:|---:|
| 256 | 0.496938 | +0.001205 | 1.643681 |
| 16k | 0.495733 | +0.000000 | 1.641701 |

[Settings](settings.json) · [Commands](jobs.json) · [Samples](REPORT.json)
