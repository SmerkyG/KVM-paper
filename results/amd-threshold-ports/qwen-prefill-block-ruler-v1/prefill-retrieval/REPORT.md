# Qwen3.8 retrieval from prefill logits at 64K

Dense, 1K top-eight LoD, and 16K top-eight LoD score identical prompts. The query asks for the codeword matching two identifiers. Each example shares its haystack, question, options, and randomized answer label across all three placements. The document table is local under both settings, local only under 16K, or remote under both. The table ends 512, 8,192, or 32,768 tokens before the last prompt position, respectively. The label with the highest next-token log probability is the prediction. Only one new token is requested, so there are no decoder attention steps.

| Evidence placement | Arm | Correct / total | Mean gold-vs-best-distractor log-odds |
|---|---|---:|---:|
| local-both | dense | 48/48 | +7.414 |
| local-both | 1024 | 48/48 | +8.034 |
| local-both | 16384 | 48/48 | +7.909 |
| local-16k | dense | 48/48 | +7.745 |
| local-16k | 1024 | 48/48 | +8.221 |
| local-16k | 16384 | 48/48 | +7.958 |
| remote-both | dense | 48/48 | +7.708 |
| remote-both | 1024 | 48/48 | +7.734 |
| remote-both | 16384 | 48/48 | +7.951 |

| Placement | Comparison | Accuracy change (pp) | Paired 95% interval | Mean margin change | Paired 95% margin interval |
|---|---|---:|---:|---:|---:|
| local-both | 1024 − 16384 | +0.00 | No discordances | +0.125 | [-0.102, +0.344] |
| local-both | 1024 − dense | +0.00 | No discordances | +0.620 | [+0.380, +0.846] |
| local-both | 16384 − dense | +0.00 | No discordances | +0.495 | [+0.258, +0.732] |
| local-16k | 1024 − 16384 | +0.00 | No discordances | +0.263 | [-0.068, +0.565] |
| local-16k | 1024 − dense | +0.00 | No discordances | +0.477 | [+0.089, +0.852] |
| local-16k | 16384 − dense | +0.00 | No discordances | +0.214 | [-0.029, +0.432] |
| remote-both | 1024 − 16384 | +0.00 | No discordances | -0.216 | [-0.648, +0.187] |
| remote-both | 1024 − dense | +0.00 | No discordances | +0.026 | [-0.430, +0.437] |
| remote-both | 16384 − dense | +0.00 | No discordances | +0.242 | [+0.003, +0.492] |

[Commands](jobs.json) · [Canonical RULER](../REPORT.md)
