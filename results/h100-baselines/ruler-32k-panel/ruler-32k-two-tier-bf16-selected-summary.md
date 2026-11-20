# Llama 3.1 32K RULER: two-tier quality on selected tasks

Paired 32-example tasks; BF16 model and KV, same frozen tokenized inputs, greedy generation, LoD sparse prefill, and graph decode. The eight tasks were selected for relatively low three-tier scores or gaps to dense; they are not a full-panel average.

## First response

| Task | Dense | Three-tier | Two-tier | Two minus three, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_multikey_1 | 100.00 | 96.88 | 96.88 | +0.00 [+0.00, +0.00] |
| niah_multikey_2 | 100.00 | 75.00 | 96.88 | +21.88 [+9.38, +37.50] |
| niah_multikey_3 | 100.00 | 50.00 | 78.12 | +28.12 [+12.50, +43.75] |
| vt | 100.00 | 95.63 | 97.50 | +1.88 [-1.25, +5.62] |
| fwe | 90.63 | 89.58 | 87.50 | -2.08 [-5.21, +0.00] |
| cwe | 37.50 | 27.19 | 23.12 | -4.06 [-9.38, +0.00] |
| qa_1 | 81.25 | 68.75 | 68.75 | +0.00 [+0.00, +0.00] |
| qa_2 | 50.00 | 46.88 | 46.88 | +0.00 [+0.00, +0.00] |

## Full fixed-length output

| Task | Dense | Three-tier | Two-tier | Two minus three, pp (paired 95% interval) |
|---|---:|---:|---:|---:|
| niah_multikey_1 | 100.00 | 96.88 | 96.88 | +0.00 [+0.00, +0.00] |
| niah_multikey_2 | 100.00 | 93.75 | 96.88 | +3.12 [-6.25, +15.62] |
| niah_multikey_3 | 100.00 | 53.12 | 78.12 | +25.00 [+9.38, +40.62] |
| vt | 100.00 | 95.63 | 97.50 | +1.88 [-1.25, +5.62] |
| fwe | 90.63 | 89.58 | 87.50 | -2.08 [-5.21, +0.00] |
| cwe | 37.50 | 27.19 | 23.12 | -4.06 [-9.38, +0.00] |
| qa_1 | 81.25 | 71.88 | 71.88 | +0.00 [+0.00, +0.00] |
| qa_2 | 53.12 | 50.00 | 53.12 | +3.12 [+0.00, +9.38] |

First-response scoring stops at the first end-of-turn or end-of-text token. Full-output scoring follows RetroInfer’s fixed-length ignore-EOS convention and can count words generated after the first turn. Intervals are paired bootstrap intervals; 32 examples provide limited precision.

Frozen full-cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
