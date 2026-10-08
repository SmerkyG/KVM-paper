# Kimi K3 / GLM5.3-Flash three-tier, top-two-page port

October 8, 2026, `lod-k3` development branch. This ports the release
`31ee46a723b5db869702b2e4110a81aab51d5462` top-two-page policy, not a new
selection heuristic. No model weights change.

## Calculation and implementation

- Both phases rank the same top eight centroids and retain the existing
  post-ranking closure above 1,024 leaves. Sinks remain separate.
- Prefill retains the release's **all-leaves** refinement in selected
  centroids. “Top-two pages” applies to decode, not prefill.
- Decode scans each selected centroid's page summaries once, selects two
  distinct pages (or the sole page), and opens their tokens. The two page
  sums/counts are subtracted together from the parent to form **one disjoint
  remainder**. The parent is replaced, never double counted.
- Persistent caches remain one latent record: GLM K=V=512 channels; Kimi
  K=512+64 and V aliases its first 512 channels. Explicit value strides
  preserve this alias for BF16, packed INT4, scales and summary tensors.
- The coarse/local/sink baseline uses the existing 16-head Gluon MLA
  decoder. Its fixed index prefix contains no historical leaf list. Selected
  parents are masked per head before softmax; the final reducer merges this
  disjoint baseline with page refinements, without subtracting BF16 outputs.
  Page refinement reuses the release kernel. Final natural-log LSE is emitted
  for DCP's distributed attention merge.
- INT4 prefill decodes **selected semantic leaves directly inside their
  transient UK/UV projection**. It does not restore an all-history BF16
  cache. Projection scratch is still a reusable worst-case allocation;
  selected-only computation does not imply selected-only allocated VRAM.
- Global-sequence update cadence is unchanged: 16K during prefill, 256
  during decode, independent of batch and DCP size. DCP scratch is reserved
  for the gathered head count, not the narrower TP query shard.

The GLM port remains vLLM-only, DCP1. Kimi ordinary DCP is supported.
The existing experimental request-per-GPU Kimi layout is still two-tier
only; this report does not claim a three-tier owner-layout/full-K3 speedup.

## Trained GLM 64K, TP4/B8

All 45 language layers / 11 MLA layers, original native FP8 weights, four
MI325X GPUs. Real frozen ProLong prompts and natural teacher-forced
continuations; untimed warmup then one measurement of 1,026 outputs / 1,025
decode steps. All eight requests remain live throughout measured decode;
four updates per row/layer are verified. Startup/JIT are excluded.
Same 16K scheduler chunk and 16 GiB native KV reservation. Native uses
its learned 2,048-token sparse indexer, **not dense all-history attention**.

| Attention mode | Prefill, cohort seconds | Decode, ms/batch step |
|---|---:|---:|
| Native sparse, earlier matched control | 24.457 | 17.837 |
| Two-tier BF16, earlier matched control | 23.632 | 13.745 |
| Three-tier BF16, top-two pages | 23.743 | 14.408 |
| Three-tier INT4, top-two pages | 24.272 | 14.137 |

BF16 three-tier is 0.47% / 4.82% slower than two-tier for prefill/decode.
INT4 is 2.71% / 2.86% slower than two-tier. Versus native sparse, BF16
prefill is 1.03x and decode is 1.24x faster; INT4 prefill is essentially tied
(0.76% shorter) and decode is 1.26x faster;
this is not a claim that three-tier improves all workloads.

Raw data:
[BF16](../glm53-flash-full-model/top2-fixed-bf16-tp4-b8-65536.json),
[INT4](../glm53-flash-full-model/top2-shared-int4-tp4-b8-65536.json).
Native/two-tier provenance and quality controls are in the
[GLM report](../glm53-flash-full-model/README.md#updated-longer-context-speed-sweep).
Earlier `top2-bf16` / `top2-int4` development speed files preceded the fixes
below and must not be treated as validated serving timings.

## Kimi full-width fixture

Four MLA-only layers, 96 query heads, latent512+direct64/V512, TP1/B1,
128K prompt, 258 outputs / 257 measured steps, warmed decode graphs. These
are deterministic **dummy weights/tokens**, not full-model timing or quality.

| Attention mode | Prefill seconds | Decode ms/step |
|---|---:|---:|
| Dense Gluon | 4.055 | 1.804 |
| Three-tier BF16, top-two pages | 2.478 | 1.442 |
| Three-tier INT4, top-two pages | 2.538 | 1.342 |

BF16 speedups are 1.64x / 1.25x; INT4 speedups are 1.60x / 1.34x.
These are still dummy-fixture results, not trained full-K3 measurements.
Raw: [dense](../kimi-k3-mla-stack/top2-dense-128k-b1-control.json),
[BF16](../kimi-k3-mla-stack/top2-shared-bf16-128k-b1.json),
[INT4](../kimi-k3-mla-stack/top2-shared-int4-128k-b1.json).
The updated BF16 TP2/DCP2 fixture also completes a global 256-token update:
[result](../kimi-k3-mla-stack/top2-masked-bf16-32k-dcp2.json).
The corrected shared-buffer INT4 TP2/DCP2 fixture also completes:
[result](../kimi-k3-mla-stack/top2-shared-int4-32k-dcp2.json).

## Quality and resolved output-layout bug

The corrected trained GLM three-tier BF16 path scores **8/8** on the matched
64K NIAH-S3 smoke, the same as native and two-tier. This is a small smoke
panel, not a full RULER result.
[Raw corrected run](../glm53-flash-full-model/top2-output-fix-bf16-s3-tp4-b8-65536.json).

The initial three-tier port scored 0/8 because its final reducer assumed a
contiguous `[batch, heads, channels]` output. GLM's RoPE-free query absorption
returns a transposed, head-major `bmm` view; `empty_like` preserved that
layout. The reducer then mixed requests and heads at B8. Kimi's concatenated
512+64 query is contiguous, so the same failure was not exposed there.
The reducer now honors explicit output batch/head strides, and the GLM
adapter explicitly allocates contiguous output scratch. Regression tests
use GLM's actual transposed output layout at B8, independent top-eight
ranking and full output/LSE references, including sharp retrieval queries.

The pre-fix ProLong/LongBench panel and page-budget/exact-prefill ablations
are **not valid three-tier decode quality evidence**. Their archives remain
for debugging provenance, but the temporary page-budget and head-union
diagnostic implementations were removed after the layout bug was isolated.
No larger page budget or head-union policy is needed to restore 8/8.
Selected-parent pre-softmax masking remains a numerical robustness measure;
post-softmax cancellation was not the cause of this particular failure.

Corrected matched panels:
[BF16 ProLong/LongBench/NIAH](../glm53-flash-full-model/top2-fixed-bf16-quality-tp4-b8-65536.json),
[INT4 ProLong/LongBench/NIAH](../glm53-flash-full-model/top2-shared-int4-quality-tp4-b8-65536.json).

| Attention mode | ProLong perplexity | LongBench smoke | NIAH-S3 64K smoke |
|---|---:|---:|---:|
| Two-tier BF16, matched control | 1.487881 | 11/16 | 8/8 |
| Three-tier BF16, corrected top-two pages | 1.488252 (+0.025%) | 11/16 | 8/8 |
| Three-tier INT4, corrected top-two pages | 1.488075 (+0.013%) | 9/16 | 8/8 |

ProLong uses the same eight 65,536-token documents (offset eight); LongBench
uses the same 16 chat-templated prompts. An equal aggregate smoke score does
not establish full-dataset equivalence or identical answers.

The INT4 check also exposed a shared-storage maintenance bug: the cache
writer processed K then V independently, even though MLA stores one record.
For Kimi's 576-channel record, the 512-channel V writer used the wrong
physical destination width. Requantization could also read codes that the
K pass had already changed. Shared K/V records are now quantized,
requantized and summarized once; V remains a strided prefix view. Initial
quantization and incremental appends match separate-storage references
bit-for-bit after matching pages by membership (physical page IDs can differ
because allocation is atomic). The pre-fix INT4 quality panel is invalid.

## Validation and reproduction

Targeted GPU checks cover release Qwen/K2 geometry, GLM/Kimi shared-latent
strides, BF16/INT4, BF16/INT8 summaries, two-page disjoint references,
selected-only dequantization/projection, remapped request rows, full combined
output/LSE with noncontiguous destinations, and CUDA graph replay. The
targeted CPU compatibility suite passes (217 passed; 153 GPU cases skipped
locally). The combined GPU suite passes 88 cases, including twelve
combined-field MLA cases covering BF16/INT4,
512/576 channels, B2/B8, sharp/diffuse queries and remapped request rows.
Six additional GLM projected-prefill cases pass against latent attention
references, including quantized all-leaves refinement.

In the AMD Kimi v10 environment, with the same local FP8 weight daemon as
the existing GLM report:

```bash
LOD_GLM_PROJECTED_LEAVES=1 LOD_GLM_KDA_PREFILL=1 \
python -m benchmarks.glm53_flash_full \
  --checkpoint /local/models/GLM-5.3-Flash \
  --mode three-tier-int4 --tp 4 --batch-size 8 --length 65536 \
  --measure speed --decode-tokens 1026 --kv-cache-gib 16 \
  --weight-cache-id glm53-flash-tp4 --output results/glm-top2-speed.json

LOD_KIMI_COMPACT_SELECTED_PROJECTION=1 LOD_KIMI_SORT_LEAF_ROUTES=1 \
python -m benchmarks.kimi_k3_smoke \
  --checkpoint tests/fixtures/kimi-k3-mla-stack --fixture-layers 4 \
  --mode three-tier-int4 --tensor-parallel-size 1 --batch-size 1 \
  --prompt-tokens 131072 --decode-tokens 258 --warmup-runs 1 \
  --load-format dummy --skip-tokenizer-init \
  --kv-cache-memory-bytes 1073741824 --output results/kimi-top2-smoke.json

LOD_RUN_GPU_TESTS=1 python -m pytest -q \
  tests/test_mla_recursive_decode.py tests/test_mla_int4_projection.py
```

Use `three-tier-bf16` for the matching BF16 arm. Seeds are 1234 for
generation; the ProLong document shuffle is 20260824. Compilation and model
staging must remain on local disk. Tests/fixtures are not trained checkpoints.
The separately downloaded Kimi-K3-for-All checkpoint has different geometry
and a checkpoint naming layout that this vLLM image does not load; its failed
startup is not counted as a successful trained-model test.
