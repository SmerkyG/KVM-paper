# meta-llama/Llama-3.1-8B-Instruct: paired 32K RULER comparison

Each selected task has 32 matched examples. Model and KV use BF16; generation is greedy and uses the same fixed task limits. RetroInfer uses its released GPU cache with dense prefill, 1.8% cluster retrieval and 23.2% estimated cluster summaries, four sink tokens and an initially 64-token exact local buffer. LoD uses sparse prefill. These are selected tasks, not a full RULER panel.

## First response

| Task | Dense | Three-tier | Two-tier | RetroInfer | Retro minus two-tier, pp (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| niah_multikey_1 | 100.00 | 96.88 | 96.88 | 100.00 | +3.12 [+0.00, +9.38] |
| niah_multikey_2 | 100.00 | 75.00 | 96.88 | 100.00 | +3.12 [+0.00, +9.38] |
| niah_multikey_3 | 100.00 | 50.00 | 78.12 | 100.00 | +21.88 [+9.38, +37.50] |
| vt | 100.00 | 95.63 | 97.50 | 100.00 | +2.50 [+0.00, +5.62] |
| fwe | 90.63 | 89.58 | 87.50 | 86.46 | -1.04 [-4.17, +2.08] |
| cwe | 37.50 | 27.19 | 23.12 | 36.25 | +13.12 [+1.56, +24.38] |
| qa_1 | 81.25 | 68.75 | 68.75 | 81.25 | +12.50 [+3.12, +25.00] |
| qa_2 | 50.00 | 46.88 | 46.88 | 50.00 | +3.12 [-6.25, +15.62] |

## Full fixed-length output

| Task | Dense | Three-tier | Two-tier | RetroInfer | Retro minus two-tier, pp (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| niah_multikey_1 | 100.00 | 96.88 | 96.88 | 100.00 | +3.12 [+0.00, +9.38] |
| niah_multikey_2 | 100.00 | 93.75 | 96.88 | 100.00 | +3.12 [+0.00, +9.38] |
| niah_multikey_3 | 100.00 | 53.12 | 78.12 | 100.00 | +21.88 [+9.38, +37.50] |
| vt | 100.00 | 95.63 | 97.50 | 100.00 | +2.50 [+0.00, +5.62] |
| fwe | 90.63 | 89.58 | 87.50 | 86.46 | -1.04 [-4.17, +2.08] |
| cwe | 37.50 | 27.19 | 23.12 | 36.25 | +13.12 [+1.56, +24.38] |
| qa_1 | 81.25 | 71.88 | 71.88 | 84.38 | +12.50 [+3.12, +25.00] |
| qa_2 | 53.12 | 50.00 | 53.12 | 56.25 | +3.12 [-6.25, +15.62] |

First-response scoring stops at the first end-of-turn or end-of-text token. The fixed-length ignore-EOS score can include later text. Intervals are paired bootstrap intervals over 32 examples. For Qwen3, RetroInfer uses a Qwen3 model adapter and the same geometry-matched cache settings used for Llama; this is not a released Qwen3 model integration.

RetroInfer selected 32–37 exact clusters per KV head; it estimated 412–475 additional cluster summaries. Index rebuilds during generation: 0.

Frozen cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
