# Qwen3.5-0.8B batch-8 coherence overhead (2026-08-13)

This is a matched raw-versus-coherence sweep for `Qwen/Qwen3.5-0.8B` on
MI325X at batch size 8. It uses the same recursive BF16 virtual-page LOD
configuration as `speed_sweep_20260813_current_b8`: 16-sqrt-N state growth,
top-3 prefill regions, top-8 decode regions, recursive page block 4, 4,096-token
prefill regions, and 1,280-token state catch-up updates.

Prefill is the arithmetic mean of three measurements after one warmup.
Decode is the mean latency of 1,024 measured tokens after 272 warmup tokens and
includes four 256-token state updates per LOD attention layer. The end-to-end
column is one measured prefill plus the 1,024 measured decode tokens; it does
not include the decode warmup. All ten result records have finite logits.

Qwen3.5-0.8B has six routed full-attention layers with two KV heads and head
dimension 256. Its other eighteen layers use GDN and do not pay the coherence
state-routing cost.

| Context | Raw prefill | Coherence prefill | Delta | Raw decode | Coherence decode | Delta | Raw end-to-end | Coherence end-to-end | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 0.374 s | 0.384 s | +2.66% | 13.53 ms/tok | 13.95 ms/tok | +3.16% | 14.224 s | 14.671 s | +3.14% |
| 16K | 0.783 s | 0.803 s | +2.65% | 13.90 ms/tok | 14.03 ms/tok | +0.90% | 15.016 s | 15.165 s | +1.00% |
| 32K | 1.735 s | 1.785 s | +2.87% | 13.97 ms/tok | 13.88 ms/tok | -0.67% | 16.040 s | 15.993 s | -0.29% |
| 64K | 3.945 s | 4.023 s | +1.97% | 13.49 ms/tok | 13.66 ms/tok | +1.27% | 17.761 s | 18.015 s | +1.43% |
| 128K | 11.163 s | 11.537 s | +3.35% | 15.02 ms/tok | 15.07 ms/tok | +0.32% | 26.541 s | 26.964 s | +1.59% |
| Geometric mean |  |  | **+2.70%** |  |  | **+0.99%** |  |  | **+1.37%** |

The older `speed_sweep_20260813_current_b8` records used raw clustering. A
direct comparison of this coherence run against those older raw records gives
a +2.02% geometric-mean end-to-end difference, but the individual differences
range from -3.35% to +6.27% because the branches and run placements differ.
The matched current-code raw/coherence delta above is therefore the useful
estimate of coherence overhead.

Using the older sweep's full-attention controls, coherence retains the
long-context speedups: at 64K it is 1.47x faster for prefill and 3.00x faster
for decode; at 128K it is 1.67x faster for prefill and 4.95x faster for decode.

Coherence does not add state slots or stored attention K/V channels. Peak
allocated memory is 0.15-0.55 GiB higher across this sweep because the fast
implementation caches transient prepared route keys; the persistent addition
is the FP32 constituent-norm sum per centroid.

## Single-matmul coherence optimization

The two coherence centroid views are collinear. If `mean_k` is a centroid's
mean key, then its spherical append key and coherence assignment key are

```
append_k = mean_k / rms(mean_k)
route_k  = mean_k / mean(rms(constituent_k))
route_k  = append_k * rms(mean_k) / mean(rms(constituent_k))
```

The optimized path therefore computes the append-key score matrix once and a
Triton reduction applies the per-centroid ratio while finding the assignment
maximum. This removes the second GEMM and the second D-wide prepared-key cache.
A fused constituent-RMS kernel also avoids the former FP32 key-sized temporary.
Neither change adds state entries.

The optimized batch-8 prefill sweep is:

| Context | Raw | Prior two-GEMM coherence | Optimized coherence | Optimized vs raw |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 0.374 s | 0.384 s | 0.377 s | +0.97% |
| 16K | 0.783 s | 0.803 s | 0.794 s | +1.46% |
| 32K | 1.735 s | 1.785 s | 1.757 s | +1.27% |
| 64K | 3.945 s | 4.023 s | 3.927 s | -0.45% |
| 128K | 11.163 s | 11.537 s | 11.180 s | +0.15% |
| Geometric mean delta |  | **+2.70%** |  | **+0.68%** |

At the representative batch-8 `overflow=1280`, `state=4096`, `head_dim=256`
geometry, the fused constituent RMS is 0.0070 ms versus 0.0375 ms for the
PyTorch expression, and maintaining key-norm sums adds about 1% to the fused
state update itself. The optimized path scored 64/64 on 8K NIAH-S3; the prior
two-GEMM reference run scored 63/64 under the same evaluation configuration.

Decode was remeasured as two same-GPU 64K raw/coherence pairs in opposite
orders, using 2,048 measured tokens per run. The geometric paired deltas were
-0.37% for prefill, +0.62% per decode token, and +0.51% for one prefill plus
2,048 decode tokens. Individual decode-pair deltas were -0.55% and +1.81%, so
the remaining sub-percent difference is close to run-to-run variance.
