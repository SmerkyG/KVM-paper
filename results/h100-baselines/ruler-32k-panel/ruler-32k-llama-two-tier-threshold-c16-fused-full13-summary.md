# Llama 3.1 32K RULER: two-tier threshold routing

Paired 32-example tasks; BF16 model and KV, same frozen tokenized inputs, greedy generation, LoD sparse prefill, and graph decode. Two-tier opens all eligible regions whose normalized-query proxy mass exceeds 16 times the mean. The 13-task average weights each task equally. Three-tier is the prior two-page result.

## First response

| Task | Dense | Three-tier top-2 | Two-tier threshold | Two minus three, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_2 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_3 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_1 | 100.00 | 96.88 | 100.00 | +3.12 [+0.00, +9.38] |
| niah_multikey_2 | 100.00 | 75.00 | 100.00 | +25.00 [+9.38, +40.62] |
| niah_multikey_3 | 100.00 | 50.00 | 100.00 | +50.00 [+34.38, +68.75] |
| niah_multivalue | 96.88 | 99.22 | 99.22 | +0.00 [+0.00, +0.00] |
| niah_multiquery | 99.22 | 98.44 | 98.44 | +0.00 [-2.34, +2.34] |
| vt | 100.00 | 95.63 | 100.00 | +4.37 [+1.25, +8.75] |
| fwe | 90.63 | 89.58 | 88.54 | -1.04 [-3.12, +0.00] |
| cwe | 37.50 | 27.19 | 29.69 | +2.50 [-5.00, +9.06] |
| qa_1 | 81.25 | 68.75 | 68.75 | +0.00 [+0.00, +0.00] |
| qa_2 | 50.00 | 46.88 | 46.88 | +0.00 [+0.00, +0.00] |
| **Average** | **88.88** | **80.58** | **87.04** | +6.46 |

## Full fixed-length output

| Task | Dense | Three-tier top-2 | Two-tier threshold | Two minus three, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_2 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_3 | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_1 | 100.00 | 96.88 | 100.00 | +3.12 [+0.00, +9.38] |
| niah_multikey_2 | 100.00 | 93.75 | 100.00 | +6.25 [+0.00, +15.62] |
| niah_multikey_3 | 100.00 | 53.12 | 100.00 | +46.88 [+31.25, +65.62] |
| niah_multivalue | 99.22 | 99.22 | 99.22 | +0.00 [+0.00, +0.00] |
| niah_multiquery | 100.00 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| vt | 100.00 | 95.63 | 100.00 | +4.37 [+1.25, +8.75] |
| fwe | 90.63 | 89.58 | 88.54 | -1.04 [-3.12, +0.00] |
| cwe | 37.50 | 27.19 | 29.69 | +2.50 [-5.00, +9.06] |
| qa_1 | 81.25 | 71.88 | 71.88 | +0.00 [+0.00, +0.00] |
| qa_2 | 53.12 | 50.00 | 50.00 | +0.00 [+0.00, +0.00] |
| **Average** | **89.36** | **82.86** | **87.64** | +4.78 |

First-response scoring stops at the first end-of-turn or end-of-text token. Full-output scoring follows RetroInfer’s fixed-length ignore-EOS convention and can count words generated after the first turn. Intervals are paired bootstrap intervals; 32 examples provide limited precision.

Frozen full-cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
