# LongBench v2 untruncated Qwen3.8 summary ablation

The original 32 and additional 64 examples form a matched 96-example subset of the 298 prompts that fit intact under the paper’s 131,072-token cap. Rendered inputs range from 10,927 to 131,026 tokens. Sampling used domain and length before decoding. Each shard ran its four arms on the same GPU with the same answer constraint and BF16 attention settings as the original comparison.

| Arm | Original correct / 32 | Added correct / 64 | Combined correct / 96 | Combined accuracy |
|---|---:|---:|---:|---:|
| Dense | 22 | 38 | 60 | 62.50% |
| Standard LoD | 22 | 35 | 57 | 59.38% |
| Pure sparse | 24 | 39 | 63 | 65.62% |
| Unweighted summaries | 22 | 40 | 62 | 64.58% |

| Paired contrast | Combined score difference (pp) | Paired 95% bootstrap interval |
|---|---:|---:|
| Standard LoD minus Pure sparse | -6.25 | [-11.46, -2.08] |
| Standard LoD minus Unweighted summaries | -5.21 | [-10.42, +0.00] |
| Dense minus Standard LoD | +3.12 | [-1.04, +8.33] |

Population correction is retained in routing for all three LoD arms. The pure-sparse prefill reuses the fused routing pass and drops its summary output. Unweighted prefill uses an additional centroid aggregation pass. LoD exact work is instrumented. Dense exact work is derived from lengths.

| Arm | Prefill exact KV / head | Prefill remote exact / head | Decode exact KV / head | Decode remote exact / head | Median TTFT (s) | Median request (s) | Median decode (ms/token) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 39786.19 | — | 56760.22 | — | 8.36 | 8.58 | 30.69 |
| Standard LoD | 8263.93 | 273.48 | 1338.11 | 969.01 | 5.51 | 5.71 | 28.47 |
| Pure sparse | 8265.04 | 274.59 | 1369.39 | 1000.29 | 5.55 | 5.74 | 28.46 |
| Unweighted summaries | 8266.08 | 275.63 | 1368.21 | 999.11 | 5.74 | 5.93 | 28.38 |

[All 96 frozen IDs](longbench-selection.json) · [Expansion runs](shards/) · [Original 32](../reviewer2-longbench-untruncated-v1/REPORT.md)
