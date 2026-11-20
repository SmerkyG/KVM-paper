# Qwen3-8B 32K RULER: dense versus two-tier LoD

BF16 model and KV; 32 paired examples per selected task. These examples were generated using the official RULER task generators and Qwen3’s tokenizer and chat template with thinking disabled; they differ from the Llama cohort. LoD uses sparse prefill and two-tier decode. No full-panel mean is implied.

## First response

| Task | Dense | Two-tier | Difference, pp (paired 95% interval) |
|---|---:|---:|---:|
| niah_multikey_2 | 93.75 | 90.62 | -3.12 [-9.38, +0.00] |
| niah_multikey_3 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| cwe | 79.38 | 71.56 | -7.81 [-14.69, -2.19] |
| qa_1 | 62.50 | 59.38 | -3.12 [-9.38, +0.00] |

## Full fixed-length output

| Task | Dense | Two-tier | Difference, pp (paired 95% interval) |
|---|---:|---:|---:|
| niah_multikey_2 | 93.75 | 90.62 | -3.12 [-9.38, +0.00] |
| niah_multikey_3 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| cwe | 79.69 | 71.56 | -8.12 [-15.00, -2.50] |
| qa_1 | 62.50 | 59.38 | -3.12 [-9.38, +0.00] |

The first-response score stops at the first end-of-turn or end-of-text token; fixed-length scoring can count text generated afterward. Intervals are paired bootstrap intervals over 32 examples.

Frozen source cohort SHA-256: `175e833ec2c3309165c2ad05d226c37a939a8c5fdb16b7b705b31b43a8b86e8e`.
