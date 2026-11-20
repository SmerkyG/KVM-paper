# Direct closed-summary attention: scaling and offsets

Llama-3.1-8B-Instruct, BF16 model and KV cache, batch one, H100 80GB. The c=16 per-region threshold opens every qualifying eligible region. The 16,384-token initial exact field, local/sink fields, 1,024-entry region eligibility and full cache completion are unchanged. All GPU experiments use uv and immutable cluster-run snapshots with one serial GPU lock.

The single-projection FP16 candidate takes **5.019 s at 64K**, 0.47% above the previous direct BF16 path, and scores **32.81% on 32K CWE** versus 30.00% conservative and 27.19% previous direct BF16. It remains 5.6% slower than dense at 64K. It is an experimental direct-summary candidate; the full 13-task RULER panel has not been rerun.

## Formulations

All versions compute only the closed-summary frontier, without subtracting an opened-summary attention field. For closed native summary logits z, the scaled versions compute m=max(z), w=exp(z-m), d=sum(w) in FP32, project w in one BF16 or FP16 tensor-core GEMM with FP32 output, and divide that output by d in FP32. The closed log partition remains m+log(d). This defers normalization until after value accumulation.

FP16 variants convert only coarse projection operands; model weights, KV storage, routing and exact attention stay BF16. Values are shared across GQA heads. No selected-count CPU readback is introduced.

The value-offset trial uses v0+sum(alpha*(v-v0)). The probability-offset trial uses mean(v)+sum((alpha-1/R)*v) across all R nonempty summaries, with alpha=0 on opened summaries. Its common mean is shared across queries, so it also requires only one GEMM. These offsets are exact identities before rounding; they need not improve finite-precision error.

## Matched 64K speed and 32K CWE quality

Prefill timings are fresh medians of three measured runs after one warmup, including deferred cache completion. They use identical frozen 65,536-token prompts, model/MLP chunks and scope. CWE quality uses all 32 frozen examples at 32K and is scored only before the first EOS/EOT. Baseline quality comes from the earlier identical frozen cohort; baseline timings are rerun here. These CWE scores are not RULER macro averages.

| Method | 64K prefill | CWE, 32 examples at 32K |
|---|---:|---:|
| Conservative prefix selection | 5.225 s | 30.00% |
| Previous direct closed BF16 | 4.996 s | 27.19% |
| Unnormalized BF16 weights, FP32 division | 4.996 s | 27.50% |
| Unnormalized FP16 weights, FP32 division | 5.019 s | 32.81% |
| BF16 with uniform probability offset | 5.048 s | Pilot only |
| Dense | 4.752 s | 37.50% |

Unnormalized BF16 weights, FP32 division: 6/6 first-response scores unchanged on two examples each of mk2, mk3 and QA1.

Unnormalized FP16 weights, FP32 division: 6/6 first-response scores unchanged on two examples each of mk2, mk3 and QA1.

## Context-length scaling

Fresh 32,768- and 128,000-token matched runs, one warmup and three measured repetitions. The 65,536-token row reuses the matched runs above. Prefill includes full cache completion, local/sink attention and the unchanged 16,384-token initial exact field. The new 32K/128K conservative controls use automatic packing dispatch (prefix filter at 2,048–4,095 slots; prefix scan from 4,096). Both direct variants use prefix filtering with the same producer tile so their projection costs can be compared.

| Context tokens | Conservative | Previous direct BF16 | New direct FP16 | Dense | Top eight |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.899 s | 1.820 s | 1.824 s | 1.560 s | 1.652 s |
| 65,536 | 5.225 s | 4.996 s | 5.019 s | 4.752 s | — |
| 128,000 | 13.226 s | 12.219 s | 12.258 s | 15.703 s | 9.540 s |

At 32,768 tokens, new direct FP16 uses 16.9% more prefill time than dense, 4.0% less than conservative and +0.22% relative to previous direct BF16. Peak PyTorch allocation is 33.43 GiB. 32 sampled frontiers across 32 layers pass; selection has zero differences outside tolerance and same-cache eager/graph logits match exactly.

At 128,000 tokens, new direct FP16 uses 21.9% less prefill time than dense, 7.3% less than conservative and +0.32% relative to previous direct BF16. Peak PyTorch allocation is 66.44 GiB. 224 sampled frontiers across 32 layers pass; selection has zero differences outside tolerance and same-cache eager/graph logits match exactly.

Top-eight was not rerun at 64K in this series. These are prefill scaling results, not additional quality panels or amortized decode measurements. No complete-execution subquadratic bound is inferred.

## Two-example CWE ablations

These pilots locate the previously changed example; they are separate from full-task quality and must not be treated as aggregate estimates.

| Variant | CWE row 9 | CWE row 14 |
|---|---:|---:|
| Unnormalized BF16 weights, FP32 division | 0% | 90% |
| Unnormalized FP16 weights, FP32 division | 0% | 90% |
| FP16 with centered values | 0% | 70% |
| BF16 with uniform probability offset | 0% | 90% |
| FP16 with uniform probability offset | 0% | 70% |

## Identical-input numerical audit

Four live 64K prefill calls, 32 queries per head per call, checked against FP32 PyTorch on the same scores, selected bits, sizes and values. This diagnostic is not a model timing or quality estimate.

| Call | Direct BF16 RMSE | Scaled BF16 RMSE | Scaled FP16 RMSE | FP16 value offset RMSE | BF16 probability offset RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.35e-06 | 1.23e-06 | 1.51e-07 | 2.25e-06 | 1.48e-06 |
| 16 | 5.55e-06 | 4.48e-06 | 5.8e-07 | 2.23e-06 | 5.49e-06 |
| 48 | 1.61e-05 | 1.41e-05 | 1.83e-06 | 7.58e-06 | 1.51e-05 |
| 80 | 1.73e-05 | 1.58e-05 | 1.99e-06 | 8.28e-06 | 2.36e-05 |

The maximum discarded FP16 exponential tail mass on these fixtures is 2.21e-07 of the closed mass. Maximum absolute coarse value is 3.25. The split-BF16 direct reference remains more arithmetically accurate but did not recover the original CWE example in the previous investigation. Numerical accuracy and task recall must be checked separately.

## Verification and provenance

- scaled-bf16: 96 sampled frontiers across 32 layers; zero selection differences outside tolerance; same-cache eager/graph logits match exactly; maximum reference attention error 0.007812.
- scaled-fp16: 96 sampled frontiers across 32 layers; zero selection differences outside tolerance; same-cache eager/graph logits match exactly; maximum reference attention error 0.007812.
- offset-bf16: 96 sampled frontiers across 32 layers; zero selection differences outside tolerance; same-cache eager/graph logits match exactly; maximum reference attention error 0.015625.

0 pending; 0 failed; 2 cancelled jobs. Only completed successful runs enter the tables. Raw commands, logs and executed-source hashes are retained in provenance/ and analysis-commands.json. SUMMARY.json hashes analyzed inputs. Full-model timings include GPU preparation, conversion and output restoration; standalone projection timers exclude preparation and are not used for the headline.

The optional full-CWE and retrieval expansions for the probability-offset BF16 prototype were cancelled while still queued after the scaled FP16 candidate completed its quality checks. Its quality results remain limited to the two-example pilot; no full-task score is inferred.

These remain experimental formulations. No serving default or exact prefill size changes, and no complete-execution subquadratic claim is made.
