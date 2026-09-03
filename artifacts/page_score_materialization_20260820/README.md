# Decode page-score materialization

This experiment changes only recursive (three-tier) decode page ranking.  It
does not change centroid routing, the selected posting lists, exact leaf
attention, or prefill.

For each batch/KV-head pair, one GQA-cooperative BF16 MMA scores the decode
queries against every live page-summary key and writes an FP32 score table.
The existing recursive page kernel then searches only the entries belonging to
the query's top-eight centroid posting lists.  Thus each page-summary key is
loaded once for all GQA heads rather than once per query head that routes to a
posting list containing that page.

## Gemma 4 26B-A4B, batch 8

Geometry: 16 query heads, 2 KV heads (GQA 8), head dimension 512, five global
LOD layers.  Prefill used 16K chunks.  Times below are the median marginal
decode times from clean runs without phase instrumentation.  `Two-tier` is the
previously reported primary-design baseline; `three-tier legacy` means the
existing recursive page selector, not the legacy LOD design.

| Context | Full | Two-tier | Three-tier legacy | Three-tier materialized |
|---:|---:|---:|---:|---:|
| 32K | 10.265 ms | 11.553 ms | 10.615 ms | 10.306 ms |
| 64K | 11.694 ms | 13.414 ms | 10.832 ms | 10.358 ms |

At 64K, recursive filtering accounts for most of the recovery from the slow
two-tier result. Materialization is a further 4.37% reduction from matched
three-tier and the final result is 11.42% below full-attention latency. At 32K,
the final result is effectively tied with full attention (+0.40%).

Matched instrumented runs attribute the page-ranking branch as follows:

| Context | Legacy selector | Materialize + select | Change |
|---:|---:|---:|---:|
| 32K | 0.0718 ms/layer | 0.0382 ms/layer | -46.9% |
| 64K | 0.1437 ms/layer | 0.0416 ms/layer | -71.0% |

### Complete 64K LOD decode attribution

A separate matched run times the inclusive LOD attention call as well as every
disjoint fused-decode phase.  Gemma has five global LOD layers.  Percentages
below use the inclusive 0.3098 ms/layer LOD call, rather than the complete
model-step latency.

| LOD work | ms / layer | Share of LOD call |
|---|---:|---:|
| Local-window attention | 0.0925 | 29.9% |
| Centroid route reduction | 0.0580 | 18.7% |
| Centroid route score groups | 0.0494 | 16.0% |
| Setup / launch gaps outside the phase timers | 0.0579 | 18.7% |
| Residual + selected exact-page attention | 0.0206 | 6.6% |
| Dense page-summary scoring and score-table write | 0.0196 | 6.3% |
| Final output/LSE merge | 0.0118 | 3.8% |
| **Inclusive LOD call** | **0.3098** | **100.0%** |

The main measured costs are therefore local attention (29.9%) and centroid
routing (34.7% combined), not reading the materialized scores and selecting a
page.  The latter operation is only one part of the 6.6% residual/exact-page
kernel, so 6.6% is a strict upper bound on its share.  Even the entire page
section (score/write plus residual/select/exact attention) is only 13.0% of the
LOD call, or about 0.201 ms across all five LOD layers per model step.

Artifact: `vllm_gemma4_26b_b8_64k_matched_auto_materialized_breakdown_r3.json`.

### D=512 GQA-local specialization

The original local decoder launched one scalar program per query head, so
Gemma's eight query heads sharing a KV head independently reloaded the same
recent K/V.  The replacement uses a GQA-shared BF16 MFMA QK pass followed by a
single tiled value pass that reconstructs the row maximum/denominator, applies
softmax, emits LSE, performs MFMA PV, and appends the current K/V. Gemma's
normalized FP32 query is rounded to BF16 only at the MFMA boundary, matching
the native AITER compute path.

This is a two-launch design. QK remains tiled independently over the token
axis so it has enough GPU workgroups; folding it into the value launch would
either serialize the entire 512-token field into only 16 Gemma workgroups or
repeat the D=512 QK calculation for every output-dimension tile. The former
loses occupancy and the latter multiplies QK work. Fusing softmax into PV
removes the useful boundary without either penalty. When page scores are also
materialized, local QK reuses that non-overlapping FP32 workspace, so the
specialization reserves no second score buffer and adds no persistent state.

At batch 8 with the actual 512-token local field:

| Test | Time |
|---|---:|
| Previous scalar LOD local kernel (isolated) | 0.1094 ms |
| Three-launch GQA MFMA draft (isolated) | 0.0437 ms |
| Two-launch fused-softmax GQA MFMA (isolated) | 0.0356 ms |
| Previous local phase in matched 64K model run | 0.0925 ms/layer |
| Three-launch local phase in matched 64K model run | 0.0515 ms/layer |
| Two-launch local phase in matched 64K model run | 0.0431 ms/layer |
| Previous inclusive LOD call | 0.3098 ms/layer |
| Three-launch inclusive LOD call | 0.2728 ms/layer |
| Previous clean 64K model decode | 10.358 ms/step |
| Two-launch clean 64K model decode | 9.942 ms/step |

The two-launch specialization is 3.00x faster than the isolated scalar kernel.
In the instrumented model it lowers the local phase another 16.4% from the
three-launch draft and 53.4% from the original scalar phase. The clean complete
model result improves 4.0% over the pre-specialization materialized run and is
15.0% faster than the 11.694 ms full-attention result. The fused work intervals
are 0.0087 ms QK and 0.0165 ms softmax/PV per layer; the inclusive local phase
is 0.0431 ms.

For comparison, Gemma's native AITER BSWA decode kernel measured 0.1584 ms at
batch 8 for D=256, GQA-2, and a 1024-token window.  The BSWA and LOD geometries
perform the same total QK/PV arithmetic, but the D=512/GQA-8 LOD geometry has
four times less unique KV traffic.  The new specialization is therefore faster
than the native BSWA call rather than merely faster than the old scalar LOD
fallback.

Numerical checks against the scalar implementation, including the current
decode token and FP32 Gemma queries, have maximum output error 0.00108 and LSE
error 0.00055.

Artifacts and reproducers:

- `vllm_gemma4_26b_b8_64k_wide_local_final2_r1.json`
- `vllm_gemma4_26b_b8_64k_local_fused2_phase_r1.json`
- `vllm_gemma4_26b_b8_64k_local_fused2_clean_r3.json`
- `scripts/benchmark_gemma_wide_local_decode.py`
- `scripts/benchmark_gemma_bswa_aiter_decode.py`

An earlier draft of this report incorrectly compared the published two-tier
numbers with a raw-routing three-tier control, and also used instrumented
end-to-end timings. The table above uses `routing_geometry=auto` consistently
and clean timings for all new three-tier entries.

Artifacts:

- `vllm_gemma4_26b_b8_32k64k_matched_auto_legacy_clean_r3.json`
- `vllm_gemma4_26b_b8_32k64k_matched_auto_materialized_clean_r3.json`
- `vllm_gemma4_26b_b8_32k64k_matched_auto_legacy_r3.json`
- `vllm_gemma4_26b_b8_32k64k_matched_auto_materialized_r3.json`

## D=128 family results

All entries are matched batch-8 clean decode runs. Phi uses TP5. The previous
two-tier controls disabled the cooperative exact-leaf decode path, so the new
three-tier runs retain that setting as well.

| Model / context | Full | Two-tier | Three-tier legacy | Three-tier materialized |
|---|---:|---:|---:|---:|
| Muse 32K | 18.943 ms | 26.086 ms | 22.794 ms | 23.056 ms |
| Muse 64K | 19.180 ms | 28.368 ms | 22.998 ms | 22.648 ms |
| OLMo 32K | 27.911 ms | 44.514 ms | 36.100 ms | 36.653 ms |
| OLMo 64K | 30.359 ms | 52.474 ms | 36.596 ms | 36.359 ms |
| Phi 32K | 8.512 ms | 14.773 ms | 12.181 ms | 12.058 ms |
| Phi 64K | 9.758 ms | 16.305 ms | 10.874 ms | 9.901 ms |

Three-tier recursive filtering removes 19-33% of the previous 64K two-tier
latency, but Muse and OLMo remain about 18-20% slower than full attention. Phi
with materialized scores is within 1.5% of full attention at 64K. The dense
score table is not consistently worthwhile at D=128: it is neutral for Muse
and OLMo and becomes useful for Phi only at 64K.

### Two-launch local re-timing

The GQA-local specialization was generalized to D=128 and D=256 and the clean
panel was rerun after softmax/PV fusion. These are matched batch-8 graph-mode
runs; Muse, OLMo, and Phi reserve the same 64 positions as their full-attention
controls. Qwen uses the panel's original zero-reserve prompts.

| Model / context | Full | Fused-local LOD | LOD vs full |
|---|---:|---:|---:|
| Muse 32K | 18.943 ms | 22.990 ms | +21.36% |
| Muse 64K | 19.180 ms | 22.706 ms | +18.39% |
| OLMo 32K | 27.911 ms | 35.286 ms | +26.42% |
| OLMo 64K | 30.359 ms | 36.315 ms | +19.62% |
| Phi 32K | 8.512 ms | 12.065 ms | +41.74% |
| Phi 64K | 9.758 ms | 9.713 ms | -0.46% |
| Qwen3.8 27B FP8 32K | 44.541 ms | 37.984 ms | -14.72% |
| Qwen3.8 27B FP8 64K | 52.017 ms | 37.905 ms | -27.13% |

The D=128 local kernel itself is not failing: isolated at batch eight and a
512-token local field it is 1.30x faster for Muse's GQA-16 geometry, 1.72x for
OLMo's GQA-5 geometry, and 1.09x for Phi's GQA-4 geometry. The complete model
movement is smaller because local attention is a minority of LOD time.

To avoid mixing primary batch-8 decode with smaller catch-up calls, the phase
profiler now records batch-size-qualified events. The table below reports only
the `_b8` recursive calls. `Page work` combines dense page-summary scoring and
selected exact-page attention; `gaps` is the inclusive recursive interval not
covered by a named GPU phase.

| Model / context | LOD ms/layer | Routing | Page work | Local | Final merge | Gaps |
|---|---:|---:|---:|---:|---:|---:|
| Muse 32K | 0.2700 | 33.77% | 28.77% | 17.69% | 6.74% | 13.04% |
| Muse 64K | 0.2986 | 31.80% | 30.53% | 17.73% | 7.81% | 12.13% |
| OLMo 32K | 0.4526 | 47.55% | 29.63% | 11.85% | 2.23% | 8.74% |
| OLMo 64K | 0.4639 | 46.50% | 31.66% | 11.44% | 2.16% | 8.25% |
| Phi 32K | 0.2264 | 30.36% | 24.30% | 21.96% | 7.73% | 15.65% |

Thus local fusion was the right fix for Gemma's D=512/GQA-8 pathology, but it
is not the remaining cross-family bottleneck. Muse and Phi divide most time
between routing and page work; OLMo is dominated by routing alone. Further
D=128 work should target those sections rather than the local window.

Panel artifacts:

- `vllm_muse30b_b8_32k64k_local_fused_reserve64_clean_r3.json`
- `vllm_olmo3_32b_b8_32k_local_fused_clean_reserve64_r3.json`
- `vllm_olmo3_32b_b8_64k_local_fused_clean_reserve64_r3.json`
- `vllm_phi4_tp5_b8_32k64k_local_fused_reserve64_clean_r3.json`
- `vllm_qwen38_27b_b8_32k64k_local_fused_clean_r3.json`
- `vllm_muse30b_b8_32k64k_local_fused_b8_phase_r1.json`
- `vllm_olmo3_32b_b8_32k64k_local_fused_b8_phase_r1.json`
- `vllm_phi4_tp5_b8_32k_local_fused_b8_phase_r1.json`

## Qwen3.5 0.8B counterexample

Qwen has weaker geometry for this method (GQA 4, head dimension 256).  In the
HF recursive decode benchmark, materialization made the complete page-ranking
branch 31% slower at 32K (0.298 vs 0.228 ms/step) and 32% slower at 64K (0.244
vs 0.185 ms/step).  This path should therefore not replace the legacy selector
unconditionally.

## Dispatch rationale and limitations

Ignoring metadata and compute, with BF16 summaries the materialized pass wins
on summary traffic when the mean routed posting-list union fraction `f` is
roughly

```
f > 1 / GQA_group_size + 2 / head_dim
```

The first term is the shared key load; the second is the FP32 score-table write
relative to a BF16 key vector.  MMA utilization and duplicate pages across the
eight posting lists also matter in practice.  This explains why GQA 8 / D=512
Gemma benefits while GQA 4 / D=256 Qwen does not.  A safe policy is to keep the
feature opt-in until dispatch can also use observed posting-list coverage.

The current implementation:

- is decode-only and requires `VLLM_LOD_LEVELS=3`;
- uses temporary FP32 score scratch, not persistent LOD state;
- currently requires BF16 page summaries (the legacy recursive path can use
  quantized summaries);
- is enabled with `VLLM_LOD_MATERIALIZE_PAGE_SCORES=1` and is off by default.

The direct residual-page numerical verifier passes with the materialized path.
