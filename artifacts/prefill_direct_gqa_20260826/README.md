# Direct-GQA coarse prefill geometry (2026-08-26)

## Result

The production two-level LOD prefill kernel now folds the native GQA ratio
directly into the MFMA `M` dimension on the two irregular-ratio geometries
where this is an end-to-end win:

| geometry | automatic coarse tile | former live rows | new live rows |
|---|---|---:|---:|
| OLMo D128/GQA5/KV8 | M128/N16/W8 | 40/64 | 125/128 |
| Qwen3.8 D256/GQA6/KV4 | M64/N16/W8 | 48/64 | 60/64 |

The row mapping remains head-major in memory, but the kernel advances by the
largest whole number of query positions that fits in the selected matrix-row
tile. Tail rows are masked rather than padding GQA5 to GQA8 or GQA6 to GQA8.
Production dispatch is selected from `(head_dim, GQA, KV heads)`, not a model
name. `VLLM_LOD_PREFILL_COARSE_DIRECT_GQA=0` disables it; setting the option to
`1` together with `VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS`,
`VLLM_LOD_PREFILL_COARSE_BLOCK_N`, and
`VLLM_LOD_PREFILL_COARSE_NUM_WARPS` provides an explicit diagnostic geometry.

## Isolated coarse-attention A/B

These BF16 measurements use batch 8, Q=512, and 4,352 coarse entries. The
control is the actual former production M64/N32/W8 kernel, not the wrapper's
older default geometry.

| geometry | former production | best direct-GQA | change |
|---|---:|---:|---:|
| OLMo D128/GQA5 | 3.253 ms | **2.478 ms** (M128/N16/W8) | -23.8% |
| Qwen3.8 D256/GQA6 | 2.733 ms | **2.030 ms** (M64/N16/W8) | -25.7% |
| Phi D128/GQA4 | **0.570 ms** | 0.623 ms (M128/N16/W8) | +9.3% |
| Muse D128/GQA16 | **1.788 ms** | 1.836 ms (M128/N16/W8) | +2.6% |

The maximum direct-versus-control output difference is 3.052e-5 and the
maximum LSE difference is 1.907e-6. Phi and Muse already use all 64 rows with
their power-of-two GQA ratios; increasing their tile or changing the mapping
only adds accumulator pressure, so their production geometry is unchanged.

`coarse_direct_gqa_corrected_controls.json` and
`coarse_direct_qwen_g6_d256_sweep.json` are authoritative. Earlier exploratory
numbers that treated the wrapper's M64/N32/W4 default as the production
control are quarantined; they overstated the gain.

## 64K end-to-end prefill

The production A/B uses batch 8, eight distinct real ProLong documents,
16,384 scheduler-token chunks, 4,096-token LOD updates, an untimed warmup, and
three measured repetitions. Qwen uses its chat template and language-only
path; OLMo uses the base model's raw prompt. Both runs used the VRAM weight
cache, and their worker audits observed the intended
`_route_logits_coarse_attention_kernel` geometry.

| model | former LOD | direct-GQA LOD | change | historical full attention |
|---|---:|---:|---:|---:|
| OLMo-3-1125-32B | 70.305 s | **69.286 s** | -1.45% | 68.537 s |
| Qwen3.8-27B-FP8 | 74.384 s | **65.062 s** | -12.53% | not rerun in this A/B |

OLMo's remaining gap to its historical full-attention result is 1.09%, down
from 2.58% for the fresh former-geometry control. The Qwen result is a 9.323-s
absolute reduction on the same code and prompt corpus.

An additional default-dispatch smoke test (with no geometry override) recorded
the requested and actually executed configurations in the vLLM worker:

| model | configured | executed |
|---|---|---|
| OLMo-3-1125-32B | direct, M128/N16/W8 | M128/N16/W8 |
| Qwen3.8-27B-FP8 | direct, M64/N16/W8 | M64/N16/W8 |

The corresponding artifacts are `olmo_8k_auto_smoke.json` and
`qwen38_8k_auto_smoke.json`. They use batch 8 and real ProLong text; their
purpose is dispatch validation rather than a new timing comparison.

## Quality and rejected variants

Matched first-eight 64K NIAH-S3 samples preserve the control score:

| model | former LOD | direct-GQA LOD |
|---|---:|---:|
| OLMo-3-1125-32B | 6/8 | 6/8 |
| Qwen3.8-27B-FP8 | 8/8 | 8/8 |

Gemma D512/GQA8 bypasses this Triton coarse kernel. Its analogous experiment
packed the broadcast GQA/query dimensions into one larger PyTorch PV GEMM.
Although that isolated normalized-probability GEMM improved from 1.048 to
0.427 ms, it did not improve production prefill. The first ordered A/B was
18.341 s current versus 18.367 s packed; the reversed five-repeat A/B was
18.110 s current versus 18.497 s packed. The production option was therefore
removed. Gemma, Phi, and Muse keep their established paths.

The Gemma A/B used ordinary checkpoint loading because the current `ipc_cache`
client fails before attention on this unquantized MoE model: the daemon owns
the finalized tensors, but the meta-constructed client does not reconstruct
the non-tensor `moe_kernel` runtime object. This is a weight-cache integration
issue and does not affect the matched attention comparison.

## Artifacts

- `olmo64_{current_m64n32w8,direct_m128n16w8}_b8_r3.json`
- `olmo64_{current,direct_m128n16w8}_niah_s3_s8.json`
- `qwen38_64k_{current,direct_m64n16w8}_b8_r3.json`
- `qwen38_64k_{current,direct_m64n16w8}_niah_s3_s8.json`
- `{olmo,qwen38}_8k_auto_smoke.json`
- `gemma4_64k_{current,packed_gemm}_b8_r3.json`
- `gemma4_64k_{packed_gemm_reverse,current_reverse}_b8_r5.json`
- `gemma_packed_coarse_normalized_q512.json`

Python compilation and `git diff --check` pass for the retained implementation.
