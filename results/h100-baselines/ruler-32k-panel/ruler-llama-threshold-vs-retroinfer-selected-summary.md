# Llama 3.1 32K RULER: two-tier threshold versus RetroInfer

Eight selected tasks with 32 matched examples each, BF16 weights and KV. Scores use only the first response; the mean weights tasks equally. This selected-task mean is not a 13-task RULER average. LoD uses sparse prefill and two-tier c=16 threshold decode. RetroInfer uses dense prefill and its released GPU cache with 1.8% retrieval and 23.2% estimated summaries.

| Task | Dense | LoD threshold | RetroInfer | RetroInfer minus LoD, points |
|---|---:|---:|---:|---:|
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | +0.00 |
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | +0.00 |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | +0.00 |
| vt | 100.00% | 100.00% | 100.00% | +0.00 |
| fwe | 90.63% | 88.54% | 86.46% | -2.08 |
| cwe | 37.50% | 29.69% | 36.25% | +6.56 |
| qa_1 | 81.25% | 68.75% | 81.25% | +12.50 |
| qa_2 | 50.00% | 46.88% | 50.00% | +3.12 |
| **Selected-task average** | **82.42%** | **79.23%** | **81.75%** | +2.51 |

The displayed averages are computed from the source reports’ two-decimal task scores, so they can differ by at most a few thousandths from averages computed directly from unrounded scores. Small differences are uncertain with 32 examples per task. The remaining five tasks have not been run with RetroInfer.

[Full LoD 13-task report](ruler-32k-llama-two-tier-threshold-c16-fused-full13-summary.md) · [Original RetroInfer report](ruler-llama-retroinfer-32k-summary.md) · [Threshold raw results](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json) · [RetroInfer raw results](ruler-llama-retroinfer-32k-selected.json)
