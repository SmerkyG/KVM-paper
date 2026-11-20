# K2 BF16 three-tier page budget on 64K RULER

All five arms use the same 32 prompts for each of the 13 tasks, native top-eight prefill with the 16K exact prefix, and region c=16 in decode. The Multikey-2 and Multikey-3 results come from the original diagnostic on the same frozen source. The other 11 tasks are fresh runs. Exact-work columns use the Multikey-2 and Multikey-3 diagnostic, averaged over requests, query heads, and layers at the last decode step. Dense quality uses the matched archived cohort.

| Refinement | RULER mean | Delta vs two-tier (pp) | Paired 95% interval | Exact pages per head | Exact remote entries per head |
|---|---:|---:|---:|---:|---:|
| two-tier | 92.04% | +0.00 | — | — | 2945.9 |
| fixed-1 | 87.22% | -4.82 | [-6.61, -3.07] | 24.6 | 254.7 |
| fixed-2 | 90.72% | -1.31 | [-2.52, -0.27] | 39.0 | 465.7 |
| fixed-4 | 92.27% | +0.23 | [-0.05, +0.58] | 62.9 | 826.6 |
| adaptive-0.5 | 91.92% | -0.12 | [-0.74, +0.34] | 164.7 | 2457.9 |

| Task | Dense | two-tier | fixed-1 | fixed-2 | fixed-4 | adaptive-0.5 |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 96.88% | 96.88% | 96.88% | 96.88% | 96.88% |
| niah_multikey_2 | 100.00% | 90.62% | 65.62% | 87.50% | 90.62% | 90.62% |
| niah_multikey_3 | 100.00% | 100.00% | 65.62% | 93.75% | 100.00% | 100.00% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 98.44% | 97.66% | 99.22% | 98.44% | 98.44% | 99.22% |
| ruler_cwe | 92.19% | 95.94% | 95.31% | 95.00% | 95.31% | 95.94% |
| ruler_fwe | 86.46% | 85.42% | 84.38% | 83.33% | 87.50% | 85.42% |
| ruler_qa_hotpot | 65.62% | 62.50% | 62.50% | 59.38% | 62.50% | 59.38% |
| ruler_qa_squad | 69.27% | 67.45% | 64.32% | 65.10% | 68.23% | 67.45% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| **Macro-average** | **93.23%** | **92.04%** | **87.22%** | **90.72%** | **92.27%** | **91.92%** |

[New and reused commands](jobs.json) · [Matched decode timings](../k2-page-budget-fixed1-v1/REPORT.md)
