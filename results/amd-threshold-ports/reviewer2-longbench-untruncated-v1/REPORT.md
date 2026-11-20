# LongBench v2 untruncated summary ablation on Qwen3.8

The 32 examples were selected across domain and length from the 298 of 503 prompts that fit intact under the paper’s 131,072-token input cap. After chat templating, inputs ranged from 11,480 to 131,026 tokens. The four arms used the same tokens, two-tier BF16 for LoD, and an A/B/C/D constrained answer with an output limit of 32 tokens.

| Arm | Correct / 32 | Accuracy | Prefill exact KV / head | Prefill remote exact / head | Decode exact KV / head | Decode remote exact / head | Median TTFT (s) | Median request (s) | Median decode (ms/token) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 22 | 68.75% | 40406.97 | — | 56966.91 | — | 7.85 | 8.06 | 30.59 |
| Standard LoD | 22 | 68.75% | 8278.10 | 284.90 | 1369.62 | 994.24 | 5.26 | 5.45 | 28.54 |
| Pure sparse | 24 | 75.00% | 8279.82 | 286.62 | 1395.33 | 1019.96 | 5.29 | 5.49 | 28.49 |
| Unweighted summaries | 22 | 68.75% | 8280.91 | 287.71 | 1391.46 | 1016.08 | 5.46 | 5.65 | 27.47 |

| Paired contrast | Score difference (pp) | Paired 95% bootstrap interval |
|---|---:|---:|
| Standard LoD minus Pure sparse | -6.25 | [-15.62, +0.00] |
| Standard LoD minus Unweighted summaries | +0.00 | [-9.38, +9.38] |
| Dense minus Standard LoD | +0.00 | [-9.38, +9.38] |

The three LoD arms retain population correction for routing. Unweighted prefill uses a second centroid aggregation pass. LoD work counters run during generation. The four arms ran on the same GPU after warmup.

[Frozen IDs and lengths](longbench-selection.json) · [Commands](jobs.json) · [Raw outputs](runs/longbench-v2/)
