# Llama 3.1 32K: fixed-eight versus threshold on paired tasks

The prior fixed-eight run covered these eight selected tasks, not the full panel. Each task uses 32 identical tokenized inputs and first-response scoring. Both runs use two-tier BF16 LoD with sparse prefill; only decode routing changes. Confidence intervals are paired bootstrap intervals over examples.

| Task | Fixed eight | c=16 threshold | Difference, points (paired 95% interval) |
|---|---:|---:|---:|
| niah_multikey_1 | 96.88% | 100.00% | +3.12 [+0.00, +9.38] |
| niah_multikey_2 | 96.88% | 100.00% | +3.12 [+0.00, +9.38] |
| niah_multikey_3 | 78.12% | 100.00% | +21.88 [+9.38, +37.50] |
| vt | 97.50% | 100.00% | +2.50 [+0.00, +5.62] |
| fwe | 87.50% | 88.54% | +1.04 [+0.00, +3.12] |
| cwe | 23.12% | 29.69% | +6.56 [+1.87, +11.87] |
| qa_1 | 68.75% | 68.75% | +0.00 [+0.00, +0.00] |
| qa_2 | 46.88% | 46.88% | +0.00 [+0.00, +0.00] |

The tasks were selected for analysis; no full-panel fixed-eight average can be inferred.
