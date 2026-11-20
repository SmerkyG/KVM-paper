# Optimizing threshold prefill on H100

BF16 Llama 3.1, batch one, 32,768 prompt tokens. Threshold variants keep equal-region c=16 decode fixed. The optimized path fuses threshold selection with unopened-region attention, compacts selected regions in groups, and shares prefill scratch across sequential layers. Entry 24 retains a strict 16 sqrt(n) exact-remote-entry cap with deterministic prefix clipping. This bound counts logical remote entries per query head; it does not bound local attention, padded work, cache construction or updates, or the complete model execution.

To enable the experimental path in a logged H100 cluster run, set `LOD_H100_PREFILL_RULE=entry` and `LOD_H100_PREFILL_IMPLEMENTATION=fused`. The default entry and work bounds are both 16; set `LOD_H100_PREFILL_ENTRY_BOUND=24` and `LOD_H100_PREFILL_WORK_BOUND=16` for the capped 24 case. `LOD_H100_PREFILL_SELECT_N=128` enables the wider selection experiment; it has been checked for entry 16. `LOD_H100_PREFILL_IMPLEMENTATION=reference` selects the earlier implementation. These settings do not change ordinary serving defaults.

## Whole-model prefill

Matched teacher-forced inputs, one warmup and two measured repeats. Figures are medians and include cache construction. Reference means the earlier tiled PyTorch implementation.

| Prefill rule and implementation | Query tile | Seconds | Relative to fixed 8 |
|---|---:|---:|---:|
| Fixed 8 | — | 1.663 | 1.00x |
| Region c=16, reference | — | 5.707 | 3.43x |
| Entry 16, reference | — | 14.504 | 8.72x |
| Entry 24 cap 16, reference | — | 18.130 | 10.90x |
| [Region c=16, fused, compact launch](threshold-prefill-fused-compact-region-16-tile1024-32768-timing.json) | 1024 | 2.247 | 1.35x |
| [Entry 16, fused, indexed default with wider selection](threshold-prefill-select128-entry-16-tile512-32768-timing.json) | 512 | 2.537 | 1.53x |
| [Entry 24 cap 16, fused, precomputed KV addresses (block 128, 4 warps)](threshold-prefill-indexed-nofill-b128-w4-entry-24-tile512-32768-timing.json) | 512 | 2.959 | 1.78x |

## Correctness checks

The independent synthetic frontier check covers region and entry thresholds, including a strict cap configuration. The real-state check compares selected regions using the same routing logits and compares output with the reference for 16 queries per layer; differences in decisions within 2e-5 of a log threshold are reported as numerical ties. Three matched RULER examples are only a smoke test, not a quality estimate.

| Task | Region c=16 | Entry 16 | Entry 24 cap 16 |
|---|---:|---:|---:|
| niah_multikey_2 | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% |

Real-state validation: 147,456 query heads across layers; 1 threshold-tie decision differences, 0 outside tolerance; maximum attention-output difference 0.01562.

[Synthetic frontier check](threshold-prefill-default-frontier-stress.json) · [Region quality](threshold-prefill-fused-compact-region-16-ruler32k-verify.json) · [Entry 16 quality](threshold-prefill-default-entry-16-ruler32k-verify.json) · [Entry 24 quality](threshold-prefill-default-entry-24-ruler32k-verify.json).

The timing table shows the fastest checked implementation of each rule among the measured settings. Other timing outputs are retained in this directory.

## Bottleneck diagnostic

One separately instrumented run of entry 16 with the compact launch and original one-warp reducer. Instrumentation synchronizes the device and is not a matched speed measurement.

Summed across layers: dispatch 0.168 s, pack 0.002 s, kernel 0.667 s, reduce 7.015 s.

[Diagnostic measurements](threshold-prefill-fused-compact-entry16-leaf-profile.json).

The optimized default was also profiled after a warmup. Synchronized timings are diagnostic and cannot be substituted for the matched measurements above: route_logits 0.306 s, select_and_coarse 0.569 s, compact 0.098 s, expand_indices 0.172 s, exact_leaves 0.508 s, merge 0.112 s.

[Warm diagnostic measurements](threshold-prefill-indexed-nofill-b128-w4-warm-profile.json).

## Matched retrieval sample

Eight examples per task, with the same inputs for both implementations. The optimized columns use the fused default with precomputed KV addresses. This is a limited quality check, not a full RULER result.

| Task | Entry 16 reference | Entry 16 optimized | Entry 24 cap 16 reference | Entry 24 cap 16 optimized |
|---|---:|---:|---:|---:|
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% |

Audit outputs: [threshold-prefill-entry-16-cap16-ruler32k-audit8](threshold-prefill-entry-16-cap16-ruler32k-audit8.json) · [threshold-prefill-default-entry-16-cap16-ruler32k-audit8](threshold-prefill-default-entry-16-cap16-ruler32k-audit8.json) · [threshold-prefill-entry-24-cap16-ruler32k-audit8](threshold-prefill-entry-24-cap16-ruler32k-audit8.json) · [threshold-prefill-default-entry-24-cap16-ruler32k-audit8](threshold-prefill-default-entry-24-cap16-ruler32k-audit8.json).
