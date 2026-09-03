# Recursive three-tier LOD with DFlash2

This experiment makes recursive three-tier LOD a graph-capturable DFlash2
target verifier and tests whether bounded page opening makes INT4 practical.
The target is `Qwen/Qwen3.8-27B-FP8` at TP1 and the drafter is
`z-lab/Qwen3.8-27B-DFlash2`, with seven draft tokens, greedy sampling, 16K
aggregate prefill chunks, and real non-repeated ProLong documents. Speed runs
emit 256 tokens after one warmup and use one measured repetition. Full and
fixed-mask two-tier controls below are the matched three-repeat results from
`artifacts/dflash2_qwen38_20260831/README.md`.

## Implementation

All eight target positions are staged before attention and flattened into one
recursive LOD call. Each position computes its own current top-eight centroid
routes and page choices against the shared immutable remote archive; routes
are neither shared nor lagged. Proposal K/V is staged in the exact recent
cache first, and a per-position logical recent length makes target position
`i` see precisely positions `0..i`. The physical cache row remains shared.

This exposed and fixed a wide-GQA local-attention bug: the QK and PV kernels
used the repeated physical cache row to load the recent length, which hid prior
proposal positions from every row after position zero. They now use the
logical flattened row when speculative metadata is supplied.

The ordinary recursive Qwen decode policy still selects the faster re-split
state router at long context. That materialized score-table route currently
causes an HSA memory fault under speculative target verification at 128K,
including eager, serialized-launch, and serial-verifier controls. Ordinary
non-speculative re-split decode and full-attention DFlash2 both pass the same
128K control. Speculative recursive verification therefore defaults to the
stable grouped (`fused`) state router while leaving the ordinary policy
unchanged. `VLLM_LOD_SPECULATIVE_RECURSIVE_STATE_ROUTE_BACKEND=resplit` is
retained only for diagnosis.

## Decode speed

End-to-end milliseconds are per emitted output token at B1 and per batch step
at B8. They include draft acceptance, so the verifier-cycle table is the
cleaner attention-side comparison.

| batch | context | full DFlash2 | two-tier BF16 | three-tier BF16 | three-tier INT4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64K | 19.316 ms | 15.091 ms | 13.447 ms | **11.791 ms** |
| 1 | 128K | 22.054 ms | 14.621 ms | 17.164 ms | **11.217 ms** |
| 8 | 64K | 42.496 ms | **23.766 ms** | 25.957 ms | 25.173 ms |
| 8 | 128K | 54.938 ms | 19.746 ms | **17.745 ms** | 22.230 ms |

The unusually favorable B1 INT4 output times and unfavorable B8/128K INT4
time are acceptance-trajectory effects, not corresponding changes in target
attention cost. One-repetition output timing should not be used to rank BF16
and INT4 by itself.

| batch | context | two-tier BF16 cycle | three-tier BF16 cycle | three-tier INT4 cycle | INT4 / BF16 cost |
|---:|---:|---:|---:|---:|---:|
| 1 | 64K | 40.731 ms | 41.313 ms | 41.760 ms | 1.011x |
| 1 | 128K | 41.495 ms | 42.085 ms | **42.064 ms** | 1.000x |
| 8 | 64K | n/a | 62.444 ms | **61.133 ms** | 0.979x |
| 8 | 128K | n/a | 58.765 ms | **55.036 ms** | 0.937x |

At B1, recursive BF16 is within 1.4% of the prior two-tier verifier cycle.
INT4 is effectively free at 128K/B1 and is 2.1--6.3% faster than recursive
BF16 in the measured B8 runs. This validates the reason to combine DFlash2
with three tiers: decode opens one 16-token residual page for each selected
centroid instead of dequantizing a selected centroid's complete posting list.

For BF16, three-tier is not a universal replacement for the current two-tier
verifier. At 64K/B8, two-tier remains faster end to end (23.766 versus 25.957
ms); at 128K/B8, three-tier wins (17.745 versus 19.746 ms). B1 target-cycle
cost is about 1.4% higher for three-tier at both lengths. The current TP1
decision is therefore two-tier BF16 at 64K and three-tier BF16 at 128K/B8;
three-tier is preferred whenever recursive INT4 storage is required. Because
the three-tier panel has one measured speed repetition, its acceptance-driven
output crossover should be confirmed before becoming a broader automatic
dispatch rule.

## Prefill

| batch | context | full attention | two-tier BF16 | three-tier BF16 | three-tier INT4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64K | 14.081 s | **8.630 s** | 8.664 s | 9.486 s |
| 1 | 128K | 42.711 s | **18.083 s** | 18.255 s | 20.307 s |
| 8 | 64K | 113.520 s | 72.722 s | **72.348 s** | 77.990 s |
| 8 | 128K | 343.913 s | 153.551 s | **153.428 s** | 167.695 s |

Recursive BF16 matches the fastest two-tier prefill within 1%. INT4 costs
7.8--11.2% here because construction must quantize the complete recursive
archive; unlike decode, that work cannot be avoided by page selection.

## Attention memory

The pool is allocated for 128K capacity, so the same allocation serves both
length rows.

| batch | BF16 LOD cache | INT4 LOD cache | cache reduction | BF16 cache + scratch | INT4 cache + scratch |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.748 GiB | 3.716 GiB | 61.9% | 9.827 GiB | 3.795 GiB |
| 8 | 77.984 GiB | 29.731 GiB | 61.9% | 78.613 GiB | 30.360 GiB |

Scratch is precision-independent. Including it, the measured attention-state
reduction is 61.4%.

## Quality

| precision | 64K NIAH-S3 | 128K NIAH-S3 |
|---|---:|---:|
| BF16 | 8/8 | 8/8 |
| INT4 | 8/8 | 8/8 |

These are post-fix, chat-formatted batch-eight runs with 64 generated tokens.
The audit records the ordinary recursive policy as `resplit`, the
speculative-only policy as `fused`, and both configured and executed
flattened verification as true. The device marker counted 36 BF16 and 38 INT4
verifier cycles in the final inspected layer, ruling out silent fallback to
piecewise target attention. The earlier pre-fix INT4 7/8 at 128K is obsolete:
that run used the incorrect physical-row local length for later proposal
positions.

## Raw records

- `bf16_fused_b1_64k128k_r1_d256.json` (cluster run 12253)
- `int4_fused_b1_64k128k_r1_d256.json` (cluster run 12254)
- `bf16_fused_b8_64k128k_r1_d256.json` (cluster run 12255)
- `int4_fused_b8_64k128k_r1_d256.json` (cluster run 12256)
- `niah_fused_bf16_b8_64k128k_n8.json` (cluster run 12257)
- `niah_fused_int4_b8_64k128k_n8.json` (cluster run 12258)
- `control_full_b1_128k_r1_d128.json`: full-attention DFlash2 128K control
- `control_normal_bf16_b1_128k_r1_d128.json`: ordinary non-speculative
  recursive re-split 128K control
- `control_fused_bf16_b1_128k_r1_d64.json`: first stable grouped speculative
  128K control
