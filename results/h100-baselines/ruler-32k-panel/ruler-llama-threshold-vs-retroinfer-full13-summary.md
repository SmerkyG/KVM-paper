# meta-llama/Llama-3.1-8B-Instruct: paired 32K RetroInfer and LoD threshold comparison

Each task has 32 matched examples and the average weights tasks equally. Model and KV use BF16; generation is greedy and uses the same fixed task limits. RetroInfer uses its released GPU cache with dense prefill, 1.8% cluster retrieval and 23.2% estimated cluster summaries, four sink tokens and an initially 64-token exact local buffer. LoD uses sparse prefill and two-tier c=16 threshold decode. All 13 RULER tasks are included.

## First response

| Task | Dense | Three-tier top-2 | Two-tier threshold | RetroInfer | Retro minus threshold, pp (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| niah_multikey_1 | 100.00 | 96.88 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_2 | 100.00 | 75.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_3 | 100.00 | 50.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| vt | 100.00 | 95.63 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| fwe | 90.63 | 89.58 | 88.54 | 86.46 | -2.08 [-5.21, +0.00] |
| cwe | 37.50 | 27.19 | 29.69 | 36.25 | +6.56 [-5.62, +18.75] |
| qa_1 | 81.25 | 68.75 | 68.75 | 81.25 | +12.50 [+3.12, +25.00] |
| qa_2 | 50.00 | 46.88 | 46.88 | 50.00 | +3.12 [-6.25, +15.62] |
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_3 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multivalue | 96.88 | 99.22 | 99.22 | 98.44 | -0.78 [-2.34, +0.00] |
| niah_multiquery | 99.22 | 98.44 | 98.44 | 98.44 | +0.00 [-2.34, +2.34] |
| **RULER average** | **88.88** | **80.58** | **87.04** | **88.53** | +1.49 |

## Full fixed-length output

| Task | Dense | Three-tier top-2 | Two-tier threshold | RetroInfer | Retro minus threshold, pp (paired 95% interval) |
|---|---:|---:|---:|---:|---:|
| niah_multikey_1 | 100.00 | 96.88 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_2 | 100.00 | 93.75 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_3 | 100.00 | 53.12 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| vt | 100.00 | 95.63 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| fwe | 90.63 | 89.58 | 88.54 | 86.46 | -2.08 [-5.21, +0.00] |
| cwe | 37.50 | 27.19 | 29.69 | 36.25 | +6.56 [-5.62, +18.75] |
| qa_1 | 81.25 | 71.88 | 71.88 | 84.38 | +12.50 [+3.12, +25.00] |
| qa_2 | 53.12 | 50.00 | 50.00 | 56.25 | +6.25 [-6.25, +18.75] |
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_3 | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multivalue | 99.22 | 99.22 | 99.22 | 100.00 | +0.78 [+0.00, +2.34] |
| niah_multiquery | 100.00 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| **RULER average** | **89.36** | **82.86** | **87.64** | **89.49** | +1.85 |

First-response scoring stops at the first end-of-turn or end-of-text token. The fixed-length ignore-EOS score can include later text. Intervals are paired bootstrap intervals over 32 examples. The small sample size limits the precision of each task score.

RetroInfer selected 32–37 exact clusters per KV head; it estimated 412–475 additional cluster summaries. Index rebuilds during generation: 0.

Frozen cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
RetroInfer outputs: [Earlier eight tasks](ruler-llama-retroinfer-32k-selected.json) · [Five additional tasks](ruler-llama-retroinfer-32k-missing5.json). [LoD threshold outputs](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json).
