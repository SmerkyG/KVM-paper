# Qwen3.8 top-8 two-tier RULER at 64K

Both arms use the same 32 matched requests per task, source, FP8 checkpoint, BF16 KV, top-eight prefill and decode, and 256-token prefill lookback. The scheduler chunk, initial exact block, and prefill catch-up interval move together.

The 1K minus 16K macro-average is +0.34 points (paired 95% bootstrap interval -0.93 to +1.50).

| Task | 16K | 1K | 1K minus 16K (pp) |
|---|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | +0.00 |
| niah_single_2 | 100.00% | 100.00% | +0.00 |
| niah_single_3 | 100.00% | 100.00% | +0.00 |
| niah_multikey_1 | 100.00% | 100.00% | +0.00 |
| niah_multikey_2 | 100.00% | 100.00% | +0.00 |
| niah_multikey_3 | 100.00% | 96.88% | -3.12 |
| niah_multiquery | 100.00% | 100.00% | +0.00 |
| niah_multivalue | 99.22% | 100.00% | +0.78 |
| ruler_cwe | 100.00% | 99.69% | -0.31 |
| ruler_fwe | 100.00% | 100.00% | +0.00 |
| ruler_qa_hotpot | 65.62% | 71.88% | +6.25 |
| ruler_qa_squad | 74.48% | 75.26% | +0.78 |
| ruler_vt | 100.00% | 100.00% | +0.00 |
| **Macro-average** | **95.33%** | **95.67%** | **+0.34** |

[Commands](jobs.json) · [First-token retrieval](prefill-retrieval/REPORT.md)
