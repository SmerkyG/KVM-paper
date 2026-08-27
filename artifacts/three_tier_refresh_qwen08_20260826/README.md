# Recursive three-tier prefill refresh: Qwen3.5-0.8B

Date: 2026-08-26. Hardware: one gfx942 GPU, shared through the resident vLLM
IPC weight daemon. Model: `Qwen/Qwen3.5-0.8B`, BF16, TP1. All speed rows use
batch 8, context 65,536, 16,384-token chunked prefill, chat formatting, greedy
64-token decode, and eight distinct real documents from
`Seerkfang/prolong-64k-512-new`. Times are medians after warmup; the primary
comparison uses seven repetitions.

| configuration | prefill (s) | change vs recursive control | decode (ms/batch step) |
|---|---:|---:|---:|
| AITER full attention | 9.909 | — | 5.822 |
| recursive control: update 1,280, ordinary selector, no overlap | 4.646 | — | 2.988 |
| update 1,280, exact hierarchical selector | 4.396 | -5.38% | 3.028 |
| hierarchical selector + local/LOD overlap, grouped route | 4.348 | -6.43% | 2.657 |
| hierarchical selector + local/LOD overlap, re-split route | **4.323** | **-6.96%** | **2.369** |

The final recursive LOD run is 2.29x faster than current AITER full-attention
prefill and 2.46x faster in the measured decode step. Exact hierarchical
prefill selection and local/LOD overlap do not alter decode. The final decode
gain comes from replacing grouped recursive state routing with the re-split
implementation: 2.369 versus 2.657 ms, a 10.8% reduction.

The initial three-repeat state-update screen rejected changing the recursive
cadence to 4,096 tokens: the control median was 4.643 seconds and the 4,096
variant was 4.710 seconds (+1.44%). Recursive LOD therefore retains the engine's
historical 1,280-token cadence rather than inheriting the flat two-tier 4,096
setting.

Quality was tested with the prefill options selected automatically, then again
with re-split state routing forced before making it the measured automatic
choice. Chat-formatted NIAH-S3 at 64K scored 8/8 for the first screen and 64/64
for the final re-split path. Its dispatch audit records:

- `levels=3`, `prefill_state_update_len=1280`;
- `_route_logits_tile_topk_kernel` plus
  `_reduce_route_logits_tile_topk_kernel` for exact hierarchical selection;
- `_query_major_residual_page_attention_kernel` with recursive indexed pages;
- AITER CK FMHA v3 local attention and `prefill_overlap_local_lod=true`;
- the allocation-free `resplit` recursive state-route backend;
- no direct-GQA packing for this power-of-two GQA4 geometry.

Residual-page geometry screens did not improve the selected `N=4`, one-warp
kernel. `N=2` regressed to 5.135 seconds prefill and 2.687 ms decode. Two warps
at `N=4` left decode neutral at 2.376 ms and did not improve prefill. `N=8`
exceeded a ROCm Triton layout/resource limit at D=256 and failed compilation,
so none of those diagnostic controls were retained in the integration.

The exact Qwen-0.8B re-split microbenchmark also rejected further tuning.
Four state-PV partitions were only about 0.2 microseconds faster than eight in
a roughly 76-microsecond route. The best score-QK launch differed from the
current launch by only about 0.2 microseconds. The current conservative tile
top-k/LSE plus reducer fusions were decisively useful, reducing the isolated
route from 86.1 to 74.8 microseconds while preserving the exact top-eight set,
and were already enabled. A final no-override vLLM run recorded `resplit` in
the dispatch audit, 4.311 seconds prefill, and 2.363 ms decode.

The dense recursive page-attention early-return branch now waits for the local
stream before consuming its output. Direct-GQA coarse packing and automatic
hierarchical routing are no longer artificially restricted to two-level LOD;
their existing measured geometry allowlists still control automatic dispatch.
The proposed query-grouped ragged expert/MMA prefill change was deliberately
not implemented or tested in this work.

Files:

- `control_u1280_64k_b8_r7.json`: matched recursive control.
- `u1280_hier_64k_b8_r7.json`: hierarchical selector only.
- `u1280_hier_overlap_64k_b8_r7.json`: final speed candidate.
- `auto_hier_overlap_niah64k_s8.json`: automatic-policy dispatch and 8/8
  NIAH-S3 validation.
- `auto_hier_overlap_resplit_64k_b8_r7.json`: final re-split speed result.
- `auto_hier_overlap_resplit_niah64_64k.json`: final 64/64 NIAH-S3 result.
- `auto_resplit_dispatch_64k_b8_r3.json`: no-override automatic-dispatch
  validation.
- `resplit_pv_splits_qwen08_64k_b8.json`,
  `resplit_score_sweep_qwen08_64k_b8.json`, and
  `resplit_fusion_sweep_qwen08_64k_b8.json`: isolated re-split tuning screens.
- `full_aiter_64k_b8_r7.json`: current AITER full-attention control.
- `u4096_64k_b8_r3.json`: rejected 4,096-token update screen.
