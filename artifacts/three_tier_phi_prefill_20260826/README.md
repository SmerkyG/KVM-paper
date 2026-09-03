# Phi three-tier prefill diagnosis and correction

## Result

Phi's recursive prefill was slow for two independent implementation reasons,
not because D=128/GQA4 is intrinsically unfavorable to LOD:

1. A 4,096-token archive step was split into state updates of 1,280, 1,280,
   1,280, and 256 tokens. This multiplied state-assignment/update launches.
2. Recursive exact attention used one query-major scalar/vector program per
   query row. It selected one 16-token page inside each routed centroid, then
   evaluated that page and the centroid residual. It did not use the D=128
   expert/MFMA path that made two-tier Phi much faster.

The corrected default uses one 4,096-token state-update batch. On the measured
recursive D128/GQA4/KVH2 geometry it also keeps constructing the three-tier
page archive, but opens complete selected centroids during prefill with the
existing expert/MFMA kernel. Decode remains ordinary recursive centroid-to-page
routing. Thus this is a hybrid prefill consumer, not a change to the stored
three-tier state or to decode.

## Matched 64K speed

These are vLLM batch-eight Phi-4 TP5 runs over eight distinct real ProLong
documents, with 16K scheduler chunks, BF16 LOD storage, one untimed warmup,
and three measured repetitions. Lower is better.

| configuration | prefill | change vs old recursive | change vs historical full |
|---|---:|---:|---:|
| historical full attention | 28.119 s | - | - |
| historical fast two-tier, raw routing | 32.930 s | - | +17.1% |
| old recursive, auto spherical, 1,280 updates | 42.124 s | - | +49.8% |
| recursive, auto spherical, 4,096 updates | 37.550 s | **-10.86%** | +33.5% |
| hybrid expert/MFMA, auto spherical, 4,096 updates | **34.333 s** | **-18.49%** | +22.1% |
| final automatic-dispatch confirmation | 34.788 s | -17.42% | +23.7% |
| hybrid expert/MFMA, raw routing, 4,096 updates | 31.481 s | -25.27% | +12.0% |

The raw-routing hybrid is also 4.40% faster than the historical two-tier
control, but it is not the default because its matched CE result is worse.
The architecture-dispatched default retains the quality-motivated automatic
spherical geometry.

The exact two-stage hierarchical top-three selector remains enabled. With
raw routing and 4,096-token updates, disabling it regressed 35.982 to
37.272 seconds, so replacing it with the nominal fused selector was rejected.

## Quality check

NIAH-S3 is not a useful Phi discriminator: the historical 64K full-attention
run also scored 0/64. Instead, the promotion check uses 32 identical 16K
ProLong documents and next-token CE loss. Token SHA-256 hashes match across
both rows.

| configuration | CE loss | perplexity | delta vs old recursive |
|---|---:|---:|---:|
| old recursive, auto spherical, 1,280 updates | 1.057098 | 2.878007 | - |
| final automatic expert/MFMA path | **1.057240** | **2.878415** | **+0.013%** |

The earlier identical eight-sample ablation separated the two changes:

| configuration | CE loss | perplexity |
|---|---:|---:|
| old recursive, auto spherical, 1,280 updates | 0.879073 | 2.408667 |
| literal-page recursive, auto spherical, 4,096 updates | 0.880822 | 2.412882 |
| hybrid expert/MFMA, auto spherical, 4,096 updates | **0.879667** | **2.410096** |
| hybrid expert/MFMA, raw routing, 4,096 updates | 0.884879 | 2.422692 |

The expert/MFMA hybrid is therefore effectively neutral and slightly improves
on the literal-page 4,096-update variant in the ablation. Raw routing buys
another 8.31% relative to the spherical hybrid at 64K, but its larger loss
shift is why it remains an explicit speed diagnostic.

## Profile evidence

The old recursive phase profile is intentionally used only to rank hotspots;
GPU-event instrumentation changes scheduler fragmentation, so its component
totals are not wall-clock timings. Across the instrumented pass the largest
components were:

| component | GPU event time | calls |
|---|---:|---:|
| state update | 7,351.9 ms | 14,640 |
| recursive exact page/residual attention | 6,153.4 ms | 8,800 |
| coarse attention | 4,695.3 ms prefill | 7,440 |
| route selection | 2,695.3 ms prefill | 7,440 |
| AITER exact local attention | 2,257.6 ms | 7,520 |
| page append | 812.1 ms | 4,080 |

This rules out the local window as the primary Phi problem: it was already
using AITER CK FMHA v3. Phi's power-of-two GQA4 geometry also already filled
all 64 rows in the established coarse MFMA tile, so the irregular-GQA packing
fix used by OLMo and Qwen3.8 was correctly rejected for Phi. The missing reuse
was specifically in recursive exact-leaf attention, while state-update
fragmentation added a separate launch-bound cost.

## Dispatch and controls

Automatic hybrid dispatch is based on `(levels, D, GQA, KV heads)`, not a model
name. It currently selects only recursive `(3, 128, 4, 2)`. Set
`VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES=0` for literal page-selecting recursive
prefill or `1` to force the hybrid on another BF16 geometry. The setting does
not alter decode.

The relevant artifacts are:

- `phi_current_64k_b8_r3.json` and `phi_current_profile_64k_b8_r1.json`
- `phi_update4k_auto_64k_b8_r3.json`
- `phi_update4k_raw_64k_b8_r3.json`
- `phi_update4k_raw_fusedsel_64k_b8_r3.json`
- `phi_hybrid_auto_64k_b8_r3.json`
- `phi_hybrid_raw_64k_b8_r3.json`
- `phi_default_auto_64k_b8_r3.json`
- `phi_auto_update1280_prolong16k_b8_s8.json`
- `phi_auto_update4096_prolong16k_b8_s8.json`
- `phi_hybrid_auto_prolong16k_b8_s8.json`
- `phi_hybrid_raw_prolong16k_b8_s8.json`
- `phi_old_auto_prolong16k_b8_s32.json`
- `phi_default_auto_prolong16k_b8_s32.json`
