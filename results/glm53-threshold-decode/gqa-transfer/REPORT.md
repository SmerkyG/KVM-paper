# Qwen3.8 and K2 Horizon optimizations in GLM-5.3-Flash threshold decode

## 64K attention-layer results on MI325X

The control is the previously optimized GLM per-head c=16, BF16, two-tier decoder, including directory runs and final-merge normalization. Both arms in this table select exactly the same physical leaves and per-head membership, and includes local, sink, current-token append and length advancement. Times are µs per layer per batch step. Each capture contributes equally to the mean.

| Batch | Previous | New | Latency reduction | Layer 3, previous → new | Layer 19, previous → new |
|---|---:|---:|---:|---:|---:|
| 1 | 40.85 | 36.54 | 10.6% | 43.51 → 39.79 | 38.20 → 33.28 |
| 8 | 104.48 | 94.44 | 9.6% | 119.96 → 107.28 | 89.00 → 81.60 |

## Full-model serving

| Batch | Previous (ms/step) | New (ms/step) | Latency reduction | Previous repeats | New repeats |
|---|---:|---:|---:|---:|---:|
| 1 | 8.161 | 8.120 | +0.50% | 8.160 / 8.162 | 8.119 / 8.120 |
| 8 | 13.527 | 13.460 | +0.49% | 13.543 / 13.511 | 13.455 / 13.465 |

One TP4 group of MI325Xs, native FP8 weights, BF16 KV, top-eight prefill, 16,384 exact prefix tokens, 65,536 prompt tokens and 1,025 timed decode steps per request. Each arm ran a full warmup and two timed generations. The displayed pairs ran previous then new. The initial B=8 pair ran in the opposite order and is listed separately below. Each measured step includes every language layer and periodic cache maintenance, with four 256-token updates per request, per MLA layer and rank. Initial cache construction belongs to prefill. The worker audit verifies dispatch on all 11 MLA layers across all four ranks, and all prompts and forced output traces match. The 45-layer model also contains 34 non-MLA layers. Forced traces measure speed at matching workloads. Downstream task quality was outside this experiment. [Serving harness](../serving.py) · [Audit](collect.py)

The earlier B=8 pair measured 13.515 → 13.430 ms/step (0.63% lower). Its audit kept the last Python dispatch, which was the B=1 graph. The displayed B=8 repeat records the schedule for each graph shape during GPU graph capture and verifies the B=8 schedule on every MLA layer and rank. Both pairs used all eight requests for the entire decode interval. [Earlier pair](serving-b8-transfer.json) · [Shape-audited repeat](b8-audit/serving-b8-transfer.json)

## What transfers

The donor code is on branch `codex/amd-threshold-ports` in `benchmarks/amd_threshold_combined.py`, `lod_attention/kernels/decode_stream_frontier.py`, and `lod_attention/kernels/decode_partial_merge.py`. Qwen3.8 uses separate 256-channel K and V with six query heads per KV head. K2 Horizon uses separate 128-channel K and V with eight. On GLM, 16 query heads on each TP rank share one 512-channel latent used for both K and V. The GLM worker loads each half once, forms the query-key score in FP32 across both halves, and reuses those BF16 latent halves in the transposed probability–value multiplication. This permits a 64-entry attention tile at both batch sizes. A direct 128-entry × 512-channel variant requested 128 KiB of on-chip shared memory on a device with a 64 KiB workgroup limit. We also tested the 128-entry tile with the dimension split. [Attention worker](../../../lod_attention/kernels/glm_threshold_decode.py)

Both selected configurations use 64-channel strips for the FP32 partial outputs. At B=1 the kernel uses 256 workers, 32-entry directory runs, and 16 channels per final reduction workgroup. It processes closed summaries first, with a compile-time loop bound. At B=8 it uses 64 workers, 8-entry directory runs, and 32 channels per final reduction workgroup. It scores and selects 128 regions per workgroup, while keeping the closed-summary softmax in 64-region tiles. Both skip atomic reservation for empty selection tiles. The opt-in serving switch is `LOD_GLM_THRESHOLD_GQA_PORTS=1` with `LOD_GLM_DECODE_THRESHOLD=16`. Other batch sizes use the earlier deferred-merge schedule.

## Controlled alternatives

Entries are changes in whole-layer latency relative to the previous GLM configuration timed in the same sweep. A negative number is faster. All rows use per-head region membership except the explicitly labelled shared-region diagnostic. The Qwen and K2 schedule rows port their execution geometry and arithmetic onto GLM’s latent and directory reader.

| Change | B=1 | B=8 |
|---|---:|---:|
| Qwen direct worker and selector schedule | +21.9% | +37.8% |
| K2 worker and selector schedule | +27.8% | exceeds shared memory |
| K2 schedule, with split latent channels | +11.1% | +99.8% |
| Transposed value multiplication alone | +3.4% | +3.3% |
| Split channels, transposed value multiply, 64-entry tile | -4.2% | -5.0% |
| Base-2 exact softmax | +1.4% | +0.6% |
| L2 cache hint on latent loads | +4.9% | +1.4% |
| Two waves per execution unit, 16-wide MFMA hint | +2.5% | +4.4% |
| BF16 residual probability multiplication for precision | +11.8% | +50.9% |
| Compute each head’s cutoff once in an extra kernel | +2.1% | +1.1% |
| 128-region score and selection tiles | +4.0% | -2.7% |
| Skip reservation for empty tiles | -0.8% | -0.9% |
| 16-bit per-run head masks | -0.3% | -0.0% |
| Deterministic tile-ordered runs, no packing atomic | +6.9% | +15.0% |
| Packed maximum and mass statistics | -0.3% | -0.1% |
| One shared region union across the 16 heads, changed routing | +0.6% | +0.3% |

The tuning sweeps also cover 16–128-entry attention tiles, 32–256 exact workers, four/eight warps, separate MFMA and wave hints, direct accumulator fusion, 32/64/128-region score tiles, static/dynamic closed loops, closed-first/last scheduling, strip-major partials, 16–128-channel reduction widths, cached reduction loads and interactions among the best candidates. At B=1 there were 116 distinct timed configurations and 4 distinct resource-limited configurations. At B=8 there were 112 and 6, respectively. The earlier study already covered flat ID writers, directory run width, exact/closed fusion and contiguous worker assignment. Fixed-size centroids, three-tier page refinement and quantized KV change the method or data format. [Per-variant timings and resource limits](summary.json) · [Prior flat packing study](../flat-pack/REPORT.md) · [Earlier fusion and scheduling study](../gqa-ports/REPORT.md)

The shared-region control opened the same union of physical rows, but assigned the union to every head. Mean exact leaves per head rose from 7,162 to 19,086 at B=1 and from 6,682 to 17,266 at B=8. This change gave no consistent speed benefit and was excluded from the selected configuration.

## Removing one change from the selected configuration

Each entry is the change in layer latency relative to the complete tuned configuration in the same run. Positive values indicate that removing the change costs time.

| Change removed | B=1 | B=8 |
|---|---:|---:|
| 64-entry tile (back to 32 entries) | +12.8% | +10.5% |
| Split latent (back to a single 512-channel dot) | +1.9% | +25.6% |
| Transposed value multiply | -0.6% | +25.5% |
| Strip-major partials and tuned reduction width | +4.6% | +1.1% |
| Empty-tile atomic guard | +0.5% | +0.9% |

## GPU stage medians

Stage times in µs are device activities recorded in an additional full-graph replay on each capture. The main speed result uses unprofiled graph timings. The activity recorder retained 44–50 samples per stage out of 50 graph launches. The stage medians exclude graph and launch overhead and do not add to those timings.

| Stage | B=1 previous | B=1 new | B=8 previous | B=8 new |
|---|---:|---:|---:|---:|
| Score and append | 5.20 | 5.35 | 14.81 | 13.06 |
| Select and pack | 5.12 | 4.90 | 20.70 | 18.78 |
| Attention worker | 20.66 | 16.48 | 54.05 | 49.82 |
| Final reduction | 3.76 | 3.86 | 8.96 | 9.17 |

## Validation

Each replay uses a real TP4 rank’s absorbed queries and prefix KV from GLM layers 3 and 19, at 65,536 and 64,000 tokens respectively. B=8 replays eight neighboring query positions with eight separately built prefix caches. Each warmed graph ran in 11 interleaved, randomly ordered timing rounds of 20 replays. The final two-arm confirmation used a separate order. The tuned arm was faster in all 44 paired timing rounds across both captures and batch sizes. These two captures were also used for kernel tuning. The four-GPU serving run exercises the complete model on the archived ProLong traces.

Every timed candidate passed an independent FP32 check of physical leaf IDs, region and per-leaf membership, output, log normalizer, current-token cache write and length advancement. The selected configuration’s relative L2 error stayed below 0.17% on these captures. Numerical differences from BF16 matrix multiplication and reduction order are possible. All 76 GPU regression tests passed. The original separate and fused flat-ID paths also passed their FP32 replay checks at both batches after the port. They cover graph replay with changed queries and cache rows, empty fields, over-cap regions and 32 heads. [Reference and timing harness](../../../benchmarks/glm53_threshold_decode.py) · [GPU regression tests](../../../tests/test_glm_threshold_decode.py) · [Placement and commands](jobs.json)
