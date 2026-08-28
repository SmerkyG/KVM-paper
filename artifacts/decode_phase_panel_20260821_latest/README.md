# Current three-tier decode kernel attribution (batch 8)

> **Superseded for 64K diagnosis.** The standalone stage table below does not
> reproduce the complete production vLLM dispatch and must not be compared to
> its full-attention table. Use `DISPATCH_AUDIT.md`, `AUDITED_64K_B8.json`, and
> `MUSE_PHI_64K_DIAGNOSIS.md` for the audited batch-eight 64K comparison.

This refresh separates periodic state maintenance from the steady decode
attention path.  All standalone timings use the current BF16 kernels on one
MI325X (`gfx942`) and batch eight.  Phi's listed geometry is its per-TP-rank
geometry.

The five full-model phase-profile runs use the exact dispatch from the latest
speed panel at 8K, 16K, 32K, 64K, and (where supported) 128K.  Every one of
those runs records zero `state_update_decode` calls and zero separate
`page_append_decode` calls.  This is expected: the speed panel generates 64
tokens, while explicit decode state rebuilds occur every 256 tokens.  The
current-token K/V insertion is part of the bounded local path.

The full-model event profiler is not suitable for summing the primary
batch-eight path: vLLM captures that CUDA graph during the warmup before the
phase timers are installed, so the event records cover only later catch-up
batch shapes.  The files remain useful for dispatch and update-call
verification.  Kernel attribution below instead uses direct batch-eight
launches of the current production kernels.

## Matched full-attention versus LOD layers

The number of eligible global-attention layers is not an architectural cause
of a speed win or loss.  Dividing the matched whole-model LOD-minus-full delta
by that count merely expresses the same delta for one replaced global layer;
it cannot change its sign.  The resulting microseconds per eligible layer are:

| Model | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Gemma | +268.2 | +220.0 | -130.4 | -315.2 | -1057.0 |
| Qwen | -65.8 | -241.6 | -510.9 | -953.4 | -1667.1 |
| Muse | +373.6 | +360.3 | +325.6 | +290.0 | +228.6 |
| OLMo | +515.0 | +478.2 | +405.5 | +255.3 | -- |
| Phi | +147.8 | +122.3 | +84.4 | -4.7 | -- |

Positive means the LOD replacement is slower than full attention for the
same eligible layer.  Thus Muse and OLMo have a genuine per-layer kernel
deficit; Gemma and Phi cross over as full attention grows; Qwen wins at every
measured length.  Model depth explains only how a per-layer delta appears in a
whole-model number, not why that delta exists.

For reference, a direct batch-eight launch of the full AITER decode attention
kernel gives the following per-layer times.  D=512 Gemma is omitted because
the installed unified-attention kernel does not support that head dimension.

| Geometry | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Muse D128/KV2/G16 | 65.8 us | 80.3 us | 109.7 us | 172.7 us | 286.0 us |
| OLMo D128/KV8/G5 | 115.6 us | 173.7 us | 284.2 us | 495.5 us | 970.4 us |
| Phi D128/KV2/G4 | 54.3 us | 68.7 us | 95.0 us | 152.9 us | 264.1 us |
| Qwen D256/KV4/G6 | 131.7 us | 293.9 us | 498.7 us | 906.8 us | 1799.8 us |

## Current route pipeline

Microseconds per LOD layer.  Muse and Phi use the grouped/fused route; OLMo,
Qwen, and Gemma use fused-top-k/LSE plus normalized-PV re-split routing.

| Geometry | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Muse D128/KV2/G16 | 36.1 | 36.1 | 37.9 | 50.4 | 73.3 |
| OLMo D128/KV8/G5 | 66.0 | 65.9 | 66.0 | 74.2 | 105.3 |
| Phi D128/KV2/G4 | 34.7 | 34.7 | 35.7 | 39.7 | 67.5 |
| Qwen D256/KV4/G6 | 66.2 | 66.4 | 65.6 | 63.1 | 87.1 |
| Gemma D512/KV2/G8 | 66.4 | 65.8 | 67.0 | 63.9 | 84.7 |

The route is approximately launch-floor-bound through 32K.  It begins to
grow at 64K for Muse/OLMo and at 128K for every geometry, but it is not the
only fixed decode cost.

## Materialized page-summary QK

Microseconds per LOD layer.  Qwen deliberately uses the legacy page selector,
so the materialized page-score operation is not part of its current path.

| Geometry | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|
| Muse D128/KV2/G16 | 14.1 | 14.0 | 13.9 | 13.8 | 16.8 |
| OLMo D128/KV8/G5 | 14.0 | 15.3 | 14.4 | 26.7 | 49.1 |
| Phi D128/KV2/G4 | 14.2 | 14.1 | 14.1 | 13.6 | 14.5 |
| Gemma D512/KV2/G8 | 14.4 | 14.4 | 16.4 | 33.7 | 60.3 |

## Other common fixed kernels

The selected-page residual/exact-attention kernel remains 47.5--51.2 us per
LOD layer over the entire 8K--128K sweep and all five geometries.  Opening the
appropriate number of page summaries adds at most 2.6 us; the cost is exact
page attention and residual/output construction, not page choice.

The fixed 512-token GQA local branch is also almost geometry-independent after
the current specialization:

| Geometry | Local branch |
|---|---:|
| Muse D128/KV2/G16 | 31.6 us |
| OLMo D128/KV8/G5 | 32.1 us |
| Phi D128/KV2/G4 | 31.3 us |
| Qwen D256/KV4/G6 | 32.3 us |
| Gemma D512/KV2/G8 | 32.5 us |

At 64K, summing only route, materialized page scoring, selected-page
attention, and local attention gives 144 us/layer for Muse, 181 us/layer for
OLMo, 134 us/layer for Phi, and 179 us/layer for Gemma.  These subtotals
exclude final output/LSE merge, dispatch gaps, and Qwen's legacy page selector.
They identify a serial fixed-cost pipeline *within each LOD layer*, not a cost
caused by how many such layers the model contains.  The exact-page kernel is
the largest common single floor; routing is the largest family-dependent
component.

## Periodic state rebuild

`profile_lod_state_update.py` times one 256-token centroid-state rebuild for
the same batch-eight geometries.  Its relevant comparison is the amortized
cost per eligible layer, obtained by dividing one layer's rebuild by 256.

| Model | 64K rebuild/layer | 64K amortized/layer | 128K rebuild/layer | 128K amortized/layer |
|---|---:|---:|---:|---:|
| Gemma | 0.511 ms | 2.00 us | 0.593 ms | 2.32 us |
| Qwen | 0.540 ms | 2.11 us | 0.635 ms | 2.48 us |
| Muse | 0.475 ms | 1.86 us | 0.461 ms | 1.80 us |
| OLMo | 0.694 ms | 2.71 us | 0.824 ms | 3.22 us |
| Phi | 0.445 ms | 1.74 us | 0.453 ms | 1.77 us |

This standalone rebuild covers centroid state maintenance, not every possible
page-directory operation at a sealing boundary.  It nevertheless establishes
that centroid updates cannot explain the several-millisecond steady decode
gap, and the matched 64-token model profiles execute no such boundary work at
all.

Authoritative standalone artifacts are `kernel_current_*.json`,
`leaf_current_*.json`, and `update_*.json`.  The `kernel_sweep_*.json` files
are deliberately unfused controls and are not the current re-split route.
