# Sealed exact-leaf archives

## Semantics

`leaf_seal_capacity` caps only the exact leaves retained for a state slot. Once
a slot reaches the cap:

- new assignments continue updating its key sum, value sum, and count;
- its existing exact leaves remain available;
- no new pages or exact leaves are appended; and
- the slot stays in the coarse residual, but is excluded from exact-leaf
  opening because its archive no longer represents the entire centroid.

This does not add state entries, reroute assignments, or freeze a centroid.
The implementation currently supports the flat paged leaf cache; combining
sealing with virtual/recursive page storage is rejected.

## Configuration

Unless noted otherwise, measurements use `Qwen/Qwen3.5-0.8B`, batch 8, state
growth `8*sqrt(T)`, prefill top-k 3, decode top-k 8, 4,864-token local
attention, a 1,280-token state-update interval, 16-token pages, and the flat
expert-layout page kernel. CE is a deterministic eight-document ProLong panel
at 32K. NIAH is the same 64 fixed NIAH-S3 examples at 128K.

## Speed

| Method | Prefill (ms) | Decode (ms/token) |
|---|---:|---:|
| Flat, uncapped, 128-entry table | 14,128.27 | - |
| Recursive page LOD | 11,594.82 | 14.6190 |
| Flat, uncapped, 2,048-entry table | 10,175.58 | 19.7443 |
| Sealed flat, cap 2,048 | 9,946.95 | 14.2828 |
| Sealed flat, cap 1,024 | 9,837.65 | - |
| Sealed flat BF16, cap 256 | 9,696.68 | - |
| Sealed flat BF16, cap 256, `16*sqrt(T)` | 13,277.92 | 15.5187 |
| Sealed flat BF16, cap 128 | 9,574.59 | - |
| Sealed flat INT8, cap 128 | 7,878.14 | - |

The matched uncapped BF16 wide-table schedule sweep at 128K is:

| State growth | NIAH-S3 | Prefill (ms) | Prefill tokens/s | Decode (ms/token) |
|---:|---:|---:|---:|---:|
| `8*sqrt(T)` | 57/64 | 10,175.58 | 103,048 | 19.7443 |
| `16*sqrt(T)` | 62/64 | 13,502.54 | 77,658 | 18.2804 |
| `32*sqrt(T)` | 63/64 | 20,607.56 | 50,883 | 17.5257 |

Doubling growth from 8 to 16 costs 32.7% in prefill latency; quadrupling it
to 32 costs 102.5%. Decode moves in the opposite direction, improving by 7.4%
and 11.2%, respectively, because the selected centroids contain fewer exact
leaves even though coarse routing covers more state entries.

The 2,048-token seal reduces prefill latency by 29.6% versus the original flat
128-entry-table run and by 14.2% versus recursive page LOD. Its fused decode is
27.7% faster than the wide-table flat control and 2.3% faster than recursive
decode. Reducing the BF16 cap from 2,048 to 128 saves another 3.7% at 128K.

The cap affects the archive rather than the centroid statistics. At cap 128,
the largest exact archive was 128 tokens while the largest centroid count was
22,549. There were no page-table overflows.

The flat page arena is still conservatively allocated from sequence capacity,
so lowering the seal cap reduces pages actually written and scanned but does
not yet shrink the reserved BF16 page tensors. The memory reduction below is
from native INT8 storage, not from relying on unallocated sealed pages.

### Native INT8 prefill

The native path is enabled with both `--prefill-int8-leaf-mma` and
`--prefill-int8-coarse-mma`. Exact leaf K/V are stored as signed INT8 with one
BF16 scale per token. Queries are quantized once per leaf-attention call; QK
and quantized-probability PV both execute as INT8 MMA. Coarse state means are
quantized just in time with one scale per value channel per 64-centroid tile,
so the probability operand retains a factorable row scale. Routing logits and
local attention remain BF16.

| Context | BF16 cap 128 | INT8 cap 128 | Speedup | Peak GiB BF16 / INT8 | Persistent LOD GiB BF16 / INT8 |
|---:|---:|---:|---:|---:|---:|
| 32K | 1,846.15 ms | 1,692.33 ms | 1.09x | 23.53 / 21.36 | - |
| 64K | 4,125.94 ms | 3,594.34 ms | 1.15x | 44.34 / 40.45 | 9.73 / 5.07 |
| 128K | 9,574.59 ms | 7,878.14 ms | 1.22x | 84.79 / 77.96 | 16.95 / 8.75 |

At 32K, the routed LOD kernels (coarse residual, route selection, and exact
leaves) take 437 ms in BF16 and 238 ms in INT8, a 1.84x attention-kernel
speedup. End-to-end speedup is lower because local FlashAttention, projections,
GDN, and the remainder of the model are unchanged. INT8 reduces the persistent
LOD cache by 47.9% at 64K and 48.4% at 128K; weights and non-LOD activations
limit peak-memory reduction to about 8%.

Raw results:

- `speed_128k_cap2048_b8_r3.json`
- `speed_128k_cap1024_b8_r3.json`
- `speed_128k_cap256_bf16_b8_r3.json`
- `speed_128k_cap128_bf16_b8_r3.json`
- `speed_128k_cap128_int8_channel_b8_r3.json`
- `speed_64k_cap128_bf16_b8_r3.json`
- `speed_64k_cap128_int8_channel_b8_r3.json`
- `profile_32k_cap128_bf16_b8.json`
- `profile_32k_cap128_int8_channel_b8.json`
- `decode_128k_cap2048_fused_b8.json`
- controls in the parent directory: `longctx_flat_128k_b8_r3.json`,
  `longctx_recursive_128k_b8_r3.json`,
  `control_flat_inline2048_128k_b8_r3.json`,
  `decode_flat_inline2048_128k_b8.json`, and
  `decode_recursive_128k_b8.json`

## Quality

### CE sweep

| Exact-leaf cap | Mean CE | Delta vs uncapped LOD |
|---:|---:|---:|
| Full attention | 3.227077 | - |
| Uncapped LOD | 3.246882 | 0 |
| 2,048 | 3.246882 | 0.000000 |
| 1,024 | 3.246862 | -0.000020 |
| 512 | 3.246978 | +0.000096 |
| 256 | 3.248100 | +0.001218 |
| 128 | 3.250273 | +0.003390 |
| 64 | 3.255657 | +0.008775 |

The CE knee is between 128 and 64. Cap 256 is the conservative low-cap point;
cap 128 is an aggressive but still small perturbation (about 0.10% in CE).
The matched expert-layout cap-128 BF16 result is 3.250318. The final INT8
cap-128 result is 3.248815, a -0.001503 delta versus that BF16 control; this is
best treated as no measurable INT8 regression rather than a claimed gain.

### NIAH-S3

| Method | Exact matches |
|---|---:|
| Full attention | **64/64** |
| Uncapped flat baseline, `8*sqrt(T)` | 57/64 |
| Uncapped flat, `16*sqrt(T)` | 62/64 |
| Uncapped flat, `32*sqrt(T)` | **63/64** |
| Sealed flat, cap 1,024, fused decode route | 57/64 |
| Sealed flat, cap 2,048, fused decode route | 58/64 |
| Sealed flat BF16, cap 256 | 58/64 |
| Sealed flat BF16, cap 256, `16*sqrt(T)` | 58/64 |
| Sealed flat BF16, cap 128 | 57/64 |
| Sealed flat INT8, cap 128 | **59/64** |

Relative to BF16 cap 128, the final INT8 run has four paired improvements, two
paired regressions, and three shared failures. The aggregate result is the best
of this sweep, though the paired churn means the one-point differences should
not be overinterpreted.

The recommended BF16 default is cap 256: it retains 58/64 NIAH with only
+0.00122 CE. Cap 128 is a viable aggressive setting, and is the preferred
setting for the tested native INT8 path because it reached 59/64 with no CE
regression. Cap 64 crosses the observed CE knee and is not recommended.

## Verification

`scripts/verify_triton_paged_leaf_attention.py` includes a two-chunk append
test that checks the archive stops exactly at the cap, allocates no extra page,
and retains the earliest archived leaves. Both 8K model prefill and fused-decode
smoke tests were finite and showed centroid counts continuing beyond the cap.
It also checks the INT8 page-attention kernel against BF16. The state-kernel
verification covers the INT8 coarse PV path; the full script passes. The
INT8 implementation currently requires flat expert-layout pages and K/V
dimensions divisible by 32. Coarse INT8 MMA is prefill-only. The INT8 leaf
backing is still readable during decode through the non-fused page kernel;
decode speed was not an optimization target in this experiment.
