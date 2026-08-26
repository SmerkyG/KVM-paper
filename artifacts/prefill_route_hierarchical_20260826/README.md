# High-GQA hierarchical prefill route selection

## Change

Muse's former materialized-logit selector assigned one program 16 query
positions across all 16 GQA heads (256 logical rows) and made that program scan
the complete centroid field.  The replacement has two exact stages:

1. independent wide centroid tiles emit their local top-three candidates;
2. a small reduction chooses the global top three and performs the established
   boundary-last route reorder inside the same kernel.

The selector is automatically enabled only for the validated materialized,
route-only, top-three, GQA>=16 prefill path.  The coarse and exact-leaf
attention algorithms are unchanged.

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

## Validation

- Production vLLM JIT monitoring observed both new kernels during inference.
- The final vLLM dispatch audit names the new kernels and reports top-three,
  GQA=16, D=128.
- Randomized selector comparisons matched both route set and ordering at the
  production geometry.
- Python compilation and `git diff --check` pass for the touched code.

Relevant outputs:

- `muse64_hier_route_b8_r3.json`
- `muse64_hier_route_b8_r4.json`
- `muse64_hier_route_b8_final.json`
