# H100 coarse-arithmetic regression and combinations

Llama-3.1-8B-Instruct BF16, batch one. All GPU experiments use uv and immutable cluster-run snapshots on the same H100. The c=16 region threshold is uncapped; the 16,384-token initial exact field, local/sink fields, 1,024-entry region eligibility and cache-completion scope remain fixed.

**Subsequent result:** the [normalization and offset follow-up](../closed-offset/REPORT.md) keeps one direct closed-summary projection, using FP16 operands and FP32 normalization afterward: **5.019 s at 64K**, **32.81% CWE recall on the 32 frozen examples at 32K**. Model weights and KV storage remain BF16. This supersedes the two-projection combination's speed/quality tradeoff below.

## Controlled ablations

The aggressive candidate already includes the conservative prefix selection improvement. These ablations isolate its larger producer tile, closed-only coarse attention, and BF16 probability rounding. These first checks contain only two CWE examples; they are not RULER averages.

| Change from conservative prefix selection | CWE row 9 | CWE row 14 |
|---|---:|---:|
| Conservative prefix selection | 0% | 90% |
| Larger producer tile only | 0% | 90% |
| Direct closed projection, old producer tile | 0% | 70% |
| Direct closed projection, split BF16 probabilities | 0% | 70% |
| Online coarse/correction coefficients through cuBLAS | 0% | 90% |
| Online coefficients, single BF16 projection | 0% | 70% |

A higher-precision projection encodes each FP32 probability as two BF16 values and sums two FP32 GEMM outputs. It isolates probability quantization from the change in coarse-attention arithmetic. The online combination instead reconstructs the original 64-region online weights, including their BF16 rounding, before subtracting unrounded opened-summary weights. It projects these effective coefficients with shared-KV cuBLAS. This approximates the original arithmetic, rather than giving bitwise equivalence: the full normalizer and GEMM accumulation order also differ.

## Full-model 64K timing

Median of three measured prefills after one warmup. All include deferred cache completion. The original matched controls are retained as dated controls; an implementation combination receives its own fresh run.

| Implementation | Prefill seconds |
|---|---:|
| Original threshold | 5.366 |
| Conservative prefix selection | 5.237 |
| Aggressive direct closed projection | 4.989 |
| Conservative with larger producer tile | 5.515 |
| Conservative with shared log-counts | 5.231 |
| Online-compatible cuBLAS combination | 5.535 |
| Online coefficients, single BF16 projection | 5.210 |
| Dense | 4.750 |
| Top eight | 4.159 |

## All 32 frozen CWE examples at 32K

First-response CWE recall: conservative **30.0000%**, aggressive **27.1875%**. 9/32 examples change score. This is the complete frozen CWE task, not a new 13-task RULER average.

| Source row | Conservative | Aggressive |
|---:|---:|---:|
| 14 | 90% | 70% |
| 66 | 60% | 30% |
| 76 | 100% | 80% |
| 218 | 70% | 80% |
| 229 | 70% | 80% |
| 257 | 80% | 70% |
| 355 | 50% | 60% |
| 425 | 60% | 0% |
| 450 | 60% | 80% |

The online-compatible combination preserves **8/8** first-response scores on the separate mk2/mk3/CWE/QA correctness panel.

On all 32 CWE examples, the two-projection online-compatible combination scores **30.3125%**, with 10/32 scores changed relative to conservative prefix selection.

## Identical-input numerical audit

Four live prefill calls, 32 queries per head per call. These compare the same saved FP32 QK scores, region sizes, selected bits and BF16 values. PyTorch computes an FP32 closed-frontier reference. This diagnostic is not model timing or a quality estimate.

| Call | Regions | Original vs FP32 RMSE | Direct BF16 vs FP32 RMSE | Split direct vs FP32 RMSE | Online combination vs original max error |
|---:|---:|---:|---:|---:|---:|
| 0 | 2048 | 1.42e-06 | 1.35e-06 | 2.16e-08 | 1.42e-06 |
| 16 | 2896 | 1.84e-05 | 5.55e-06 | 6.98e-08 | 0.00178 |
| 48 | 2048 | 1.95e-05 | 1.61e-05 | 2.72e-07 | 7.23e-05 |
| 80 | 3547 | 6.19e-05 | 1.73e-05 | 1.66e-07 | 9.66e-06 |

## Frontier and graph checks

- `combine-tile-v1-prefix-tuned-65536-verify`: 96 sampled frontiers across 32 layers; zero selection mismatches outside tolerance; reference attention maximum absolute error 0.007812; same-cache eager/graph logits match exactly.
- `combine-log-v1-prefix-65536-verify`: 96 sampled frontiers across 32 layers; zero selection mismatches outside tolerance; reference attention maximum absolute error 0.015625; same-cache eager/graph logits match exactly.
- `combine-online-v1-prefix-closed-tuned-65536-verify`: 96 sampled frontiers across 32 layers; zero selection mismatches outside tolerance; reference attention maximum absolute error 0.015625; same-cache eager/graph logits match exactly.
- `combine-online-single-v1-prefix-closed-tuned-65536-verify`: 96 sampled frontiers across 32 layers; zero selection mismatches outside tolerance; reference attention maximum absolute error 0.007812; same-cache eager/graph logits match exactly.

No complete-execution subquadratic claim is made. Selection has no cap, CPU count readback or change in exact prefill size. Failed and partial runs are excluded. 0 pending, 1 failed and 1 cancelled combination jobs. Commands, logs and content-verified executed sources are in the adjacent provenance directory and analysis command ledger.

The first numerical probe was rejected before measurement because its prompt-replay option required token capture. The second, still-queued snapshot had the same issue and was cancelled. The corrected third probe adds token capture; neither earlier job contributes a measurement.
