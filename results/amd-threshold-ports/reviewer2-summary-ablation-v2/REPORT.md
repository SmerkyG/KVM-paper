# Qwen3.8 BF16 summary ablation

The 128K RULER setting uses 32 matched prompts per task from lm-eval 0.4.13. LongBench v2 uses 96 frozen examples sampled across domain and length from the 503-example split. Prompts over 131,072 input tokens retain the first and last halves before chat templating. LongBench retains original shorter prompts, constrains the answer to A/B/C/D and permits 32 output tokens.

All 96 LongBench prompts fit intact under the cap. The sample was drawn from 298 eligible examples.

All LoD arms keep the BF16 two-tier cache, 16K exact first prefill block, local fields, and top-eight population-corrected region router. The sparse arm omits closed summaries. The unweighted arm retains each closed summary with zero population bias in aggregation. Dense is a fresh BF16 control. All runs use batch one and a shared prompt and output policy.

| Task | Dense | Standard LoD | Pure sparse | Unweighted summaries |
|---|---:|---:|---:|---:|
| niah_single_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_1 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_2 | 100.00% | 93.75% | 87.50% | 87.50% |
| niah_multikey_3 | 100.00% | 93.75% | 71.88% | 68.75% |
| niah_multiquery | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multivalue | 100.00% | 100.00% | 100.00% | 100.00% |
| ruler_cwe | 100.00% | 100.00% | 100.00% | 100.00% |
| ruler_fwe | 100.00% | 100.00% | 100.00% | 100.00% |
| ruler_qa_hotpot | 65.62% | 53.12% | 53.12% | 53.12% |
| ruler_qa_squad | 74.22% | 74.22% | 69.01% | 71.09% |
| ruler_vt | 100.00% | 100.00% | 100.00% | 100.00% |
| **128K RULER macro-average** | **95.37%** | **93.45%** | **90.89%** | **90.81%** |
| **LongBench v2 untruncated subset** | **62.50%** | **59.38%** | **65.62%** | **64.58%** |

| Benchmark | Paired score contrast | Difference (pp) | Paired 95% bootstrap interval |
|---|---|---:|---:|
| 128K RULER | Standard LoD minus Pure sparse | +2.56 | [+0.48, +4.65] |
| 128K RULER | Standard LoD minus Unweighted summaries | +2.64 | [+0.88, +4.49] |
| 128K RULER | Dense minus Standard LoD | +1.92 | [+0.48, +3.37] |
| LongBench v2 untruncated subset | Standard LoD minus Pure sparse | -6.25 | [-11.46, -2.08] |
| LongBench v2 untruncated subset | Standard LoD minus Unweighted summaries | -5.21 | [-10.42, +0.00] |
| LongBench v2 untruncated subset | Dense minus Standard LoD | +3.12 | [-1.04, +8.33] |

Exact work averages individual exact KV entries per query head, including the causal first prefill block. LoD decode counters record the actual executed attention queries, including any additional terminal step. Dense exact work is calculated from sequence lengths. The median request latencies include LoD cache construction, exclude model load and warmup, and use the same GPU within each task. LoD work counters run during generation. Unweighted prefill currently uses a second centroid aggregation pass.

| Benchmark | Arm | Prefill exact KV / head | Prefill remote exact / head | Decode exact KV / head | Decode remote exact / head | Median TTFT (s) | Median request (s) | Median decode (ms/token) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128K RULER | Standard LoD | 8877.73 | 528.39 | 2313.57 | 1909.02 | 16.52 | 17.06 | 27.68 |
| 128K RULER | Pure sparse | 8882.56 | 533.22 | 2437.95 | 2032.99 | 16.52 | 17.09 | 28.44 |
| 128K RULER | Unweighted summaries | 8884.71 | 535.37 | 2433.32 | 2029.58 | 17.61 | 18.23 | 27.76 |
| 128K RULER | Dense | 64740.01 | — | 129787.67 | — | 42.08 | 42.87 | 34.23 |
| LongBench v2 untruncated subset | Standard LoD | 8263.93 | 273.48 | 1338.11 | 969.01 | 5.51 | 5.71 | 28.47 |
| LongBench v2 untruncated subset | Pure sparse | 8265.04 | 274.59 | 1369.39 | 1000.29 | 5.55 | 5.74 | 28.46 |
| LongBench v2 untruncated subset | Unweighted summaries | 8266.08 | 275.63 | 1368.21 | 999.11 | 5.74 | 5.93 | 28.38 |
| LongBench v2 untruncated subset | Dense | 39786.19 | — | 56760.22 | — | 8.36 | 8.58 | 30.69 |

| Benchmark | Paired timing contrast | Mean TTFT difference (s) | Mean request difference (s) |
|---|---|---:|---:|
| 128K RULER | Standard LoD minus Pure sparse | +0.02 [+0.01, +0.03] | +0.00 [-0.02, +0.03] |
| 128K RULER | Standard LoD minus Unweighted summaries | -1.10 [-1.11, -1.09] | -1.12 [-1.14, -1.10] |
| 128K RULER | Dense minus Standard LoD | +25.25 [+25.19, +25.31] | +25.42 [+25.36, +25.48] |
| LongBench v2 untruncated subset | Standard LoD minus Pure sparse | -0.01 [-0.03, +0.01] | -0.01 [-0.03, +0.01] |
| LongBench v2 untruncated subset | Standard LoD minus Unweighted summaries | -0.33 [-0.39, -0.26] | -0.33 [-0.39, -0.26] |
| LongBench v2 untruncated subset | Dense minus Standard LoD | +6.59 [+5.19, +8.00] | +6.61 [+5.21, +8.03] |

[RULER commands and source](jobs.json) · [Selected LongBench IDs and truncation metadata](../reviewer2-longbench-add64-v1/longbench-selection.json)
[LongBench rerun commands](../reviewer2-longbench-untruncated-v1/jobs.json)
[Additional 64 LongBench examples](../reviewer2-longbench-add64-v1/REPORT.md)
