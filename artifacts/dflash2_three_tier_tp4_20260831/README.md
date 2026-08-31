# TP4 DFlash2: full attention versus recursive three-tier LOD

This panel compares full target attention with the current recursive
three-tier BF16 LOD verifier on `Qwen/Qwen3.8-27B-FP8` at TP4.  The DFlash2
drafter is `z-lab/Qwen3.8-27B-DFlash2` with seven draft tokens.  Prompts are
distinct, non-repeated ProLong documents; prefill uses a 16K aggregate token
budget.  Speed rows emit 256 tokens after one warmup and report the median of
three measured repetitions.  Quality is chat-formatted NIAH-S3 at batch eight
with eight examples and 64 generated tokens per length.

Each rank has six query heads, one KV head, and head dimension 256.  Full
attention uses `ROCM_AITER_UNIFIED_ATTN`.  Recursive LOD uses the grouped
MFMA state router, independently routes all eight target positions, scans one
selected 16-token residual page per opened centroid, and performs one
flattened target-verifier launch.  The audit records all 16 global-attention
layers executing the parallel speculative path; no route is shared or lagged.

## Results

Milliseconds are per emitted batch step, not per individual request.

| batch | context | full DFlash2 | three-tier BF16 | speedup |
|---:|---:|---:|---:|---:|
| 1 | 64K | 9.777 ms | **9.219 ms** | **1.06x** |
| 1 | 128K | 11.702 ms | **9.144 ms** | **1.28x** |
| 8 | 64K | 17.255 ms | **13.399 ms** | **1.29x** |
| 8 | 128K | 20.888 ms | **13.099 ms** | **1.59x** |

The measured recursive target-verifier cycle is 26.284 ms at 64K/B1 and
26.255 ms at 128K/B1.  End-to-end DFlash2 time also depends on the generated
acceptance trajectory; the cycle metric isolates the LOD target cost, while
the matched three-repeat end-to-end rows are the serving comparison.

| batch | context | full prefill | three-tier BF16 prefill | speedup |
|---:|---:|---:|---:|---:|
| 1 | 64K | 5.367 s | **4.401 s** | **1.22x** |
| 1 | 128K | 14.583 s | **9.447 s** | **1.54x** |
| 8 | 64K | 43.562 s | **36.954 s** | **1.18x** |
| 8 | 128K | 120.436 s | **79.846 s** | **1.51x** |

| mode | 64K NIAH-S3 | 128K NIAH-S3 |
|---|---:|---:|
| full attention | 8/8 | 8/8 |
| three-tier BF16 | 8/8 | 8/8 |

## TP4 graph-capture and catch-up fixes

PyTorch's ROCm ProcessGroupNCCL watchdog polls HIP events from a background
thread.  Those event queries are illegal while another thread captures a
DFlash graph.  At TP4 this intermittently terminated both full and LOD engine
startup even though vLLM selected its graph-safe custom all-reduce for model
collectives.  Multi-GPU DFlash runs in the panel wrapper now default to
`TORCH_NCCL_BLOCKING_WAIT=1`, with asynchronous error handling and its monitor
disabled; caller overrides remain honored.  Captured TP4 DFlash graphs pass
for both full and LOD after this change.

Repeated speculative runs also exposed an overlap in recursive recent-cache
catch-up when an update boundary left a non-empty exact suffix.  Catch-up now
materializes that short source suffix before shifting it to the start of the
fixed cache row.  This occurs only at the amortized update boundary and does
not change graph shape or the hot decode kernels.  The three-repeat B1 and B8
panels and both batch-eight quality lengths pass; the B8 speed audit alone
records 1,792 catch-up batches.

## Raw records

- `full_b1_64k128k_r3_d256.json` (cluster run 12259)
- `lod3_bf16_b1_64k128k_r3_d256.json` (cluster run 12270)
- `full_b8_64k128k_r3_d256.json` (cluster run 12271)
- `lod3_bf16_b8_64k128k_r3_d256.json` (cluster run 12272)
- `niah_full_b8_64k128k_n8.json` (cluster run 12274)
- `niah_lod3_bf16_b8_64k128k_n8.json` (cluster run 12273)
