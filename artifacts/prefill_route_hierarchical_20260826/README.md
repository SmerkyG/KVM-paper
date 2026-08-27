# Hierarchical route selection

## Change

Muse's former materialized-logit selector assigned one program 16 query
positions across all 16 GQA heads (256 logical rows) and made that program scan
the complete centroid field.  The replacement has two exact stages:

1. independent wide centroid tiles emit their local top-three candidates;
2. a small reduction chooses the global top three and performs the established
   boundary-last route reorder inside the same kernel.

The selector now supports every validated grouped-GQA geometry.  Automatic
dispatch remains geometry-specific because the extra reduction launch is not
profitable on every model size.  The coarse and exact-leaf attention
algorithms are unchanged.

## Isolated result

Muse production geometry: batch 8, QH=32, KVH=2, GQA=16, Q=512, S=4352,
D=128, BF16 logits.

| selector | median |
|---|---:|
| former grouped M16/N32/W8 | 3.762 ms |
| hierarchical M8/N1024/W2 + reduction W2 | 0.876 ms |

This is a 76.7% reduction.  The selected centroid set **and route order** were
exact for every tested row.  Additional sweeps covered Q lengths 8--512 and
state lengths 256--4352; the hierarchical schedule won at every size.

## Other model geometries

The same exact two-stage selector was compared with each production grouped
selector at batch 8, Q=512, S=4352.  These measurements isolate route
selection from model execution:

| geometry | former selector | hierarchical | reduction |
|---|---:|---:|---:|
| Muse, D128/GQA16 | 3.762 ms | 0.876 ms | 76.7% |
| OLMo, D128/GQA5 | 3.180 ms | 1.985 ms | 37.6% |
| Phi, D128/GQA4 | 1.406 ms | 0.406 ms | 71.2% |
| Qwen3.8, D256/GQA6 | 1.980 ms | 1.200 ms | 39.4% |
| Gemma-4, D512/GQA8 | 2.800 ms | 0.800 ms | 71.4% |

Every comparison preserved both the selected centroid set and its established
boundary-last route order exactly.

The real 64K/B8 ProLong-prefill A/B is more selective because routing is only
one part of prefill:

| model | former prefill | hierarchical | change |
|---|---:|---:|---:|
| Muse-Glimmer-30B | 56.508 s | 52.854 s | -6.47% |
| Phi-4 TP5 | 46.414 s | 43.216 s | -6.89% |
| Gemma-4-26B-A4B | 18.971 s | 18.429 s | -2.86% |
| OLMo-3-32B | 73.059 s | 72.355 s | -0.96% |
| Qwen3.8-27B-FP8 | 79.098 s | 78.994 s | -0.13% |
| Qwen3.5-0.8B | 4.545 s | 4.965 s | **+9.25%** |

These selector A/Bs use the current automatic routing geometry. In
particular, Phi uses spherical routing (`cosine/none/query`), which is not
eligible for the fused route/coarse producer, while OLMo uses coherence-aware
routing (`none/coherence/none`) and remains fused-route eligible. Therefore
the 43.216-second Phi result must not be compared as a pure hierarchy delta
against the older 32.930-second raw-routing result. The hierarchy-only Phi A/B
is 46.414 to 43.216 seconds within the same spherical path. OLMo's current
coherence-aware 72.355-second result is effectively identical to its
72.364-second raw-routing top-three/cap-16 diagnostic, so routing geometry is
not the source of OLMo's remaining deficit.

The D128 Phi/OLMo prefill attention kernels are not generally M=16. Route QK
is a PyTorch matmul over the full query chunk. The hierarchical top-k
consumer uses eight query positions per program, but it only scans already
materialized logits and performs no QK/PV MFMA. Coarse attention is configured
with `BLOCK_M=16` before GQA folding and a 64-row limit: Phi executes 16 query
positions times four heads, for 64 valid rows; OLMo shrinks to eight positions
times a GQA group padded from five to eight, for a 64-row tile with 40 valid
and 24 masked rows. The D128 exact-leaf geometry tuner replaces its nominal
M16/N32 setting with M64/N64/W4. Exact local attention uses AITER CK FMHA.

The full-attention OLMo reference takes a materially different path. vLLM
dispatches `ROCM_AITER_UNIFIED_ATTN`, whose gfx942 long-prefill configuration
uses the AITER Triton 2-D unified-attention kernel with M128/N64, W4, and one
program per KV head and query block. OLMo's five query heads per KV head are
folded directly into M: row `m` addresses query position `m // 5` and query
head `m % 5`. Since `128 // 5 = 25`, each program advances 25 positions; 125
of its 128 rows are new query/head pairs and the remaining three repeat the
first three heads of the boundary position in the adjacent tile. Thus the
steady-state matrix-row efficiency is 125/128 = 97.7%, with no padding of
GQA5 to GQA8. Each D128 K/V tile is loaded once and reused by all five query
heads across those positions. This is why the unusual GQA ratio itself is
nearly free in full prefill, unlike the current OLMo LOD coarse kernel's
40-live/64-row geometry.

The follow-up direct-GQA implementation closes most of this packing gap. OLMo
now uses 125 live rows in an M128 tile and improves fresh 64K/B8 prefill from
70.305 to 69.286 seconds without changing its matched NIAH-S3 score. The same
mechanism improves Qwen3.8 D256/GQA6 from 74.384 to 65.062 seconds. Phi and
Muse already fill all rows with their power-of-two GQA ratios and do not use
the alternative. See `artifacts/prefill_direct_gqa_20260826/README.md` for the
corrected isolated controls, production dispatch audits, and artifacts.

The large-model geometries are enabled automatically.  Qwen3.5-0.8B retains
the former single-stage selector: its six custom-attention layers do too
little routing work to amortize another launch.  The explicit
`VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE={0,1}` setting force-disables/enables the
selector for diagnostics and unvalidated geometries.

## Decode analogue

Decode already computes route and coarse attention together, so the analogous
change is to let independent centroid segments emit local top-eight candidates
and online-softmax partials, then reduce that short field in parallel.  A
variable-count 64K/B8 control selected the exact same top-eight set for all
five large-model geometries:

| model geometry | former route+coarse | segmented route+coarse | reduction |
|---|---:|---:|---:|
| Muse D128/GQA16 | 54.185 us | 38.268 us | 29.4% |
| OLMo D128/GQA5 | 116.400 us | 114.736 us | 1.4% |
| Phi D128/GQA4 | 40.761 us | 36.421 us | 10.6% |
| Qwen D256/GQA6 | 129.478 us | 110.796 us | 14.4% |
| Gemma D512/GQA8 | 141.739 us | 109.502 us | 22.7% |

The production schedules retain the established BF16-rounded centroid means.
The coarse output changes by at most 1.42e-4 solely because the same online
softmax terms are associated in a different segment order.

### Corrected production decode comparison

The first production table in this directory was invalid.  Its nominal
route-only runs did toggle only `VLLM_LOD_DECODE_HIERARCHICAL_ROUTE`, but the
harness failed to enable `VLLM_LOD_DECODE_GQA_UNION`,
`VLLM_LOD_DECODE_GQA_UNION_HIP`, and
`VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER`.  The dispatch audit therefore recorded
`flat two-tier leaf dispatch` and
`_reduce_routed_split_decode_lod_attention_kernel`, rather than the historical
fast persistent fixed-list page-size-one final scan.  The resulting claimed
1.4--8.1% gains compared two route schedules inside the wrong decoder and must
not be used.

The corrected A/B uses B8, eight distinct real 64K ProLong prompts, a 64-token
prompt reserve, 64 decode tokens, and three repetitions.  Both arms execute
the fixed-mask HIP/AITER final path; only the route producer changes.  Gemma's
arms ran concurrently on the same node to remove the previously observed
engine-order bias.

| model | grouped fast top-8 | segmented fast top-8 | change |
|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | **36.334 ms** | 36.478 ms | +0.40% |
| Gemma-4-26B-A4B | **10.235 ms** | 10.300 ms | +0.64% |
| Phi-4 TP5 | 11.198 ms | **11.163 ms** | -0.32% |
| OLMo-3-32B | 28.769 ms | **28.436 ms** | -1.16%* |

Muse is not an additional new decode win: its historical 19.349-ms fast
top-eight result already used the segmented D128/GQA16 schedule.  The current
same-schedule rerun is 19.153 ms, consistent within run variation.

*OLMo's requested tuned arm retained
`_decode_route_coarse_gqa_groups_kernel` with one tile per producer.  It changed
the grouped tile/reducer geometry (32/W4 to 64/W2), not producer segmentation,
so the 1.16% difference is not a hierarchical-route result and the strict
route-pair validator rejects it.

The corrected grouped controls also reproduce the earlier authoritative panel:
36.247 ms Qwen, 10.421 ms Gemma, 10.913 ms Phi, and 28.878 ms OLMo.  Automatic
segmented decode routing is consequently limited to Muse.  Qwen, Gemma, and
OLMo retain grouped routing; Phi's 0.32% difference is too small to justify an
automatic change.  This decode conclusion does not alter the valid prefill
selector results above.

Authoritative corrected artifacts are:

- `qwen38_decode_fastpath_corrected_{grouped,hierarchical}_64k_b8_r3_d64.json`
- `gemma4_decode_fastpath_concurrent_{grouped,hierarchical}_64k_b8_r3_d64.json`
- `phi4_decode_fastpath_corrected_{grouped,hierarchical}_64k_b8_r3_d64.json`
- `olmo3_decode_fastpath_corrected_{grouped,hierarchical}_64k_b8_r3_d64.json`
- `muse_decode_fastpath_current_hierarchical_64k_b8_r3_d64.json`

All earlier `*_decode_routeonly_*`, `*_decode_warm_*`, and unqualified
`*_decode_{grouped,hierarchical}_*` files in this directory are quarantined
diagnostics, not production comparisons.  The corrected harness now fails a
run unless the dispatch audit confirms fixed-mask execution, HIP execution,
the page-size-one final path, and the requested grouped or segmented producer.

## Muse 64K end-to-end

All timings use batch 8, 16K vLLM scheduler chunks, 4K LOD update chunks, BF16
LOD storage, and eight distinct real ProLong prompts.  The final dispatch audit
records `_route_logits_tile_topk_kernel`,
`_reduce_route_logits_tile_topk_kernel`, and the unchanged
`_route_logits_coarse_attention_kernel`.

| configuration | prefill time |
|---|---:|
| historical full attention | 51.933 s |
| former ordinary top-three LOD | 56.508 s |
| hierarchical LOD run 1 | 51.960 s |
| hierarchical LOD run 2 | 52.862 s |
| hierarchical LOD final/order-exact | 52.854 s |
| hierarchical LOD median | **52.854 s** |

The new path is 6.47% faster (1.069x) than the former LOD path and is 1.77%
slower than the historical full-attention timing.  The remaining materialized
centroid QK and stable coarse-attention pass are unchanged and are now the
next meaningful targets.

## Post-selector profile and branch overlap

The real post-change Muse profile confirms that the selector is no longer the
dominant stage:

| component | mean per call |
|---|---:|
| dense centroid QK | 379 us |
| hierarchical top-three | 877 us |
| stable coarse attention | 1,195 us |
| exact-leaf dispatch | 405 us |
| exact-leaf attention | 388 us |
| exact local AITER attention | 420 us |

The stable coarse, exact-leaf, and exact-local branches depend on the selected
routes but not on each other's outputs.  For Muse's two-level GQA-16 path they
now launch on separate CUDA streams after routing and synchronize only at the
final LSE merge.  This preserves the established N=32 coarse arithmetic and
does not lag routes or change the attention algorithm.  Explicit environment
overrides remain available as
`VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF={0,1}` and
`VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD={0,1}`.

Two same-engine measurements plus an independent automatic-dispatch run with
the original coarse geometry were 51.352 s, 52.136 s, and 51.514 s, for a
51.514 s median:

| configuration | median Muse 64K B8 prefill |
|---|---:|
| historical full attention | 51.933 s |
| former ordinary top-three LOD | 56.508 s |
| hierarchical selector only | 52.854 s |
| hierarchical selector + branch overlap | **51.514 s** |

The complete result is 8.84% faster than the former LOD path and 2.54% faster
than the selector-only median.  It is within measurement noise of full
attention, with the measured median 0.81% faster.  The production dispatch
audit records N=32, 64 grouped rows, W8, and both overlap flags enabled.

An N=64/16-row coarse retile looked about 8% faster in an isolated endpoint
microbenchmark, but regressed the real overlapped prefill median to 52.196 s,
so it is not enabled.  Recomputing QK in the coarse pass instead of rereading
materialized BF16 logits also preserved outputs closely but was 2.1x slower;
that experimental code was removed.

## Validation

- Production vLLM JIT monitoring observed both new kernels during inference.
- The final vLLM dispatch audit names the new kernels and reports top-three,
  GQA=16, D=128.
- Randomized prefill-selector comparisons matched both route set and ordering
  on all five large-model geometries.
- Variable-count decode comparisons matched the complete top-eight route set
  on all five geometries; the maximum coarse-output reassociation difference
  was 1.42e-4.
- Corrected decode production audits record the requested grouped/segmented
  score-only producer, fixed-mask execution, HIP execution, and page-size-one
  final attention in every authoritative A/B arm.
- The final N=32 branch-overlap path scored **8/8** on canonical 64K NIAH-S3
  with Muse chat formatting and greedy 64-token generation.
- Python compilation and `git diff --check` pass for the touched code.

Relevant outputs:

- `muse64_hier_route_b8_r3.json`
- `muse64_hier_route_b8_r4.json`
- `muse64_hier_route_b8_final.json`
- `muse64_hier_route_b8_finephase.json`
- `muse64_overlap_both_original_coarse_b8_r2.json`
- `muse64_overlap_auto_default_b8_r3.json`
- `muse64_overlap_both_niah_s3_s8.json`
- `coarse_muse_final_sweep_parity.json`
