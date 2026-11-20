# Qwen3.8 top-8 two-tier RULER at 64K

Both arms use all 500 matched archived prompts on each of the 13 tasks with the frozen source and serving settings from the first-32 pilot. Both use FP8 weights, BF16 KV, greedy generation at batch one, native top-eight prefill and decode, and a 256-token prefill lookback. The initial exact block, scheduler chunk, and cache catch-up interval change together. Dense scores come from the same archived requests in the original serving runtime.

The 1K minus 16K macro-average is +0.55 points (paired 95% bootstrap interval +0.24 to +0.88).

| Task | Archived dense | 16K | 1K | 1K minus 16K (pp) | Paired 95% interval (pp) | Improved / declined |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | +0.00 | [+0.00, +0.00] | 0 / 0 |
| niah_single_2 | 100.00% | 99.80% | 100.00% | +0.20 | [+0.00, +0.60] | 1 / 0 |
| niah_single_3 | 100.00% | 99.80% | 100.00% | +0.20 | [+0.00, +0.60] | 1 / 0 |
| niah_multikey_1 | 100.00% | 99.80% | 100.00% | +0.20 | [+0.00, +0.60] | 1 / 0 |
| niah_multikey_2 | 100.00% | 98.00% | 97.40% | -0.60 | [-2.20, +1.00] | 8 / 11 |
| niah_multikey_3 | 100.00% | 97.80% | 99.60% | +1.80 | [+0.40, +3.40] | 11 / 2 |
| niah_multiquery | 100.00% | 99.95% | 100.00% | +0.05 | [+0.00, +0.15] | 1 / 0 |
| niah_multivalue | 100.00% | 99.35% | 99.65% | +0.30 | [-0.20, +0.80] | 12 / 5 |
| ruler_cwe | 99.98% | 100.00% | 100.00% | +0.00 | [+0.00, +0.00] | 0 / 0 |
| ruler_fwe | 99.40% | 98.73% | 99.07% | +0.33 | [-0.07, +0.80] | 8 / 3 |
| ruler_qa_hotpot | 72.00% | 65.60% | 67.80% | +2.20 | [-0.40, +4.80] | 32 / 21 |
| ruler_qa_squad | 78.85% | 74.42% | 76.78% | +2.37 | [+0.47, +4.20] | 22 / 10 |
| ruler_vt | 100.00% | 99.96% | 100.00% | +0.04 | [+0.00, +0.12] | 1 / 0 |
| **Macro-average** | **96.17%** | **94.86%** | **95.41%** | **+0.55** | **[+0.24, +0.88]** | — |

Scores on the retested first 32 prompts per task differed from the pilot on 5 prompts at 1K and 8 at 16K.

[Frozen inputs](cohort-manifest.json) · [Commands](jobs.json) · [Pilot](../qwen-prefill-block-ruler-v1/REPORT.md)
