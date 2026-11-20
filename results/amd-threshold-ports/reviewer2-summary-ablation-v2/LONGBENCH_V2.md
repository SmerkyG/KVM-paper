# LongBench v2 summary ablation on Qwen3.8

All four arms ran the same 32 examples from the 503-example split, selected before decoding across domain and length. Eleven base prompts were truncated by retaining the first and last halves under the 131,072-token cap before applying the chat template. The others kept their original lengths. The evaluator constrains the answer to A/B/C/D and permits up to 32 output tokens.

The LoD arms use two-tier BF16 KV and top-eight population-corrected routing for prefill and decode. Pure sparse omits the closed summaries from aggregation. Unweighted summaries removes the population bias from their attention logits. Dense is a fresh BF16 control.

| Arm | Correct / 32 | Accuracy | Exact KV per prefill query head | Remote exact KV per prefill query head | Exact KV per decode query head | Remote exact KV per decode query head | Median TTFT (s) | Median request (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard LoD | 15 | 46.88% | 8575.1 | 358.3 | 1541.1 | 1208.1 | 11.19 | 11.39 |
| Pure sparse | 16 | 50.00% | 8578.2 | 361.4 | 1587.6 | 1254.6 | 11.26 | 11.46 |
| Unweighted summaries | 17 | 53.12% | 8578.7 | 361.9 | 1583.0 | 1250.0 | 11.81 | 12.00 |
| Dense | 16 | 50.00% | 53878.2 | — | 85595.3 | — | 23.17 | 23.39 |

| Paired contrast | Difference (pp) | Paired 95% bootstrap interval |
|---|---:|---:|
| Standard LoD minus Pure sparse | -3.12 | [-15.62, +6.25] |
| Standard LoD minus Unweighted summaries | -6.25 | [-15.62, +0.00] |
| Dense minus Standard LoD | +3.12 | [+0.00, +9.38] |

Each arm ran on the same GPU after warmup. The timing includes cache construction and LoD work counters. The unweighted prefill implementation uses a second centroid aggregation pass. Dense exact work is calculated from the sequence lengths.

[Frozen IDs and lengths](longbench-selection.json) · [Commands](jobs.json) · [Raw outputs](runs/longbench-v2/)
