# Llama 3.1 32K RULER: 13-task exploratory comparison

Each task contains 32 frozen examples, using the same tokenized inputs for dense and LoD. The model weights and KV are BF16; LoD uses three-tier top-two and sparse prefill. Generation is greedy with the reference task limits and ignores EOS, matching the RetroInfer RULER generation convention.

| Task | Dense | LoD BF16 | Difference, pp (paired 95% interval) |
|---|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_2 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_3 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_1 | 100.00 | 96.88 | -3.12 [-9.38, +0.00] |
| niah_multikey_2 | 100.00 | 93.75 | -6.25 [-15.62, +0.00] |
| niah_multikey_3 | 100.00 | 53.12 | -46.88 [-65.62, -31.25] |
| niah_multivalue | 99.22 | 99.22 | +0.00 [+0.00, +0.00] |
| niah_multiquery | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| vt | 100.00 | 95.63 | -4.37 [-8.75, -1.25] |
| fwe | 90.63 | 89.58 | -1.04 [-3.12, +0.00] |
| cwe | 37.50 | 27.19 | -10.31 [-23.13, +2.50] |
| qa_1 | 81.25 | 71.88 | -9.38 [-18.75, +0.00] |
| qa_2 | 53.12 | 50.00 | -3.12 [-15.62, +6.25] |
| **13-task mean** | **89.36** | **82.86** | **-6.50 [-8.67, -4.31]** |

The mean weights tasks equally. Intervals use paired bootstrap resampling within each task; the mean interval resamples independently within all 13 tasks. Each task has only 32 examples, so small differences are uncertain. Zero-width intervals do not rule out rare failures.

Double-P Table 2 reports a 32K 13-task mean of 90.00% dense and 89.91% Double-P. These are separate literature results, not matched runs on our frozen examples. The paper does not establish identical model variant, prompts or sample selection. Its `Avg.` column additionally averages over 16K, 32K and 64K, which we did not run.

Frozen cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
