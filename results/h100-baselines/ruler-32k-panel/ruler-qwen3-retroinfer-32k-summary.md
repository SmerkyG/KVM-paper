# Qwen/Qwen3-8B: paired 32K RULER comparison

Each selected task has 32 matched examples. Model and KV use BF16; generation is greedy and uses the same fixed task limits. RetroInfer uses its released GPU cache with dense prefill, 1.8% cluster retrieval and 23.2% estimated cluster summaries, four sink tokens and an initially 64-token exact local buffer. LoD uses sparse prefill. These are selected tasks, not a full RULER panel.

## First response

| Task | Dense | Two-tier | RetroInfer | Retro minus two-tier, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_multikey_2 | 93.75 | 87.50 | 93.75 | +6.25 [+0.00, +15.62] |
| niah_multikey_3 | 100.00 | 93.75 | 100.00 | +6.25 [+0.00, +15.62] |
| cwe | 79.38 | 73.75 | 78.44 | +4.69 [-0.31, +10.31] |
| qa_1 | 62.50 | 56.25 | 65.62 | +9.38 [+0.00, +21.88] |

## Full fixed-length output

| Task | Dense | Two-tier | RetroInfer | Retro minus two-tier, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_multikey_2 | 93.75 | 87.50 | 93.75 | +6.25 [+0.00, +15.62] |
| niah_multikey_3 | 100.00 | 93.75 | 100.00 | +6.25 [+0.00, +15.62] |
| cwe | 79.69 | 73.75 | 78.75 | +5.00 [-0.00, +10.62] |
| qa_1 | 62.50 | 56.25 | 65.62 | +9.38 [+0.00, +21.88] |

First-response scoring stops at the first end-of-turn or end-of-text token. The fixed-length ignore-EOS score can include later text. Intervals are paired bootstrap intervals over 32 examples. For Qwen3, RetroInfer uses a Qwen3 model adapter and the same geometry-matched cache settings used for Llama; this is not a released Qwen3 model integration.

RetroInfer selected 33–37 exact clusters per KV head; it estimated 425–475 additional cluster summaries. Index rebuilds during generation: 0.

Frozen cohort SHA-256: `175e833ec2c3309165c2ad05d226c37a939a8c5fdb16b7b705b31b43a8b86e8e`.
