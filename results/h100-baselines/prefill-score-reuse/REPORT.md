# H100 threshold-prefill score reuse

BF16 Llama-3.1-8B-Instruct, batch 1, entry-16 two-tier prefill and decode. Score reuse computes native BF16 query–mean-key products once and scales them by inverse query RMS for routing. The reference separately computes products from a BF16-normalized query. This experiment is opt-in; the exact initial prefill chunk stays 16,384 tokens.

| Context | Baseline prefill | Score reuse prefill | Reduction |
|---|---:|---:|---:|
| 32K | 2.512 s | 2.410 s | 4.07% |
| 64K | 7.985 s | 7.678 s | 3.84% |

Full-model, uninstrumented prefill latency; each is the median of three runs after one warmup on the same prompt and H100. Diagnostic instrumentation is disabled for these timings.

Synthetic paired check (eight queries, 32 heads, 48 deliberately imbalanced regions):

- Entry bound 16, strict work bound 16: 8 / 12,288 region decisions changed; maximum output difference 0.066055.
- Entry bound 32, strict work bound 1: 0 / 12,288 region decisions changed; maximum output difference 0.000000.

The initial reuse-only frontier check failed its 0.025 maximum-error tolerance; the paired check confirmed that the original fused path passed on those inputs.

Real-state selection audit on one 32K prompt: the first 64 later queries per head and layer, all 32 layers. It compares against the original normalized-query matmul using the same tile shape.

- 59,901 / 134,217,728 query–region decisions changed (0.0446%).
- 31,485 / 65,536 query heads had at least one changed region (48.04%).
- Mean opened entries per query head: 254.156 before and 254.193 after. Maximum absolute routing-logit difference: 0.123528.

| RULER task | Examples | Baseline first-response score | Reuse first-response score | Improved | Declined |
|---|---:|---:|---:|---:|---:|
| qa_1 | 32 | 68.75% | 75.00% | 2 | 0 |
| niah_multikey_2 | 32 | 75.00% | 71.88% | 1 | 2 |

Quality is measured on paired frozen RULER examples, scoring only the first response through the first end-of-turn or end-of-text token. This limited panel does not establish equivalence on the full RULER suite. One net multikey regression in 32 examples is an observed sample result, not a precise estimate of the full-suite effect. Run metadata and individual outputs are preserved alongside this report. Baseline quality outputs are in the [matched prefill matrix](../ruler-32k-prefill-matrix/REPORT.md).
