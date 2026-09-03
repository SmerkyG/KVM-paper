# Decode split/reduction experiments, batch 8

Qwen3.5-0.8B, BF16 virtual leaf storage, top-8 two-level routing, gfx942,
256 cached decode steps after 32K/64K/128K prefill. The phase numbers below
are medians across three repeats and include exact leaf/local attention plus
the final branch reduction; they exclude route scoring/grouping.

## Combined route-split and final reduction

The cooperative HIP leaf kernel normally writes eight routes times `S` split
partials, a first Triton kernel reduces `S` within each route, and the final
kernel merges the eight routes with coarse/local/sink attention. The combined
path lets the final kernel consume the HIP partials directly, eliminating the
intermediate route reduction.

At 32K (`S=8`), two runs gave pooled-median leaf-plus-final time of 0.834
ms/step versus 0.902 ms/step for the ordinary tree, a 7.6% reduction in this
part of decode. One matched run improved end-to-end decode by 2.7%; whole-step
timing was substantially noisier on repetition. All batch rows retained the
same top-1 token in all three repeats. A direct kernel comparison measured
exactly zero maximum difference between the ordinary and combined outputs.

The direct fold becomes serially expensive at 16 or 32 route splits. It is
therefore enabled only when `S<=8`; longer contexts retain the two-stage
reduction. This bounded version is now the default.

## Fixed split-count sweep

| Context | Splits | Leaf + final (ms/step) |
|---:|---:|---:|
| 8K | scalar | 0.670 |
| 8K | 4, combined reduction | 0.799 |
| 8K | 8, combined reduction | 0.761 |
| 16K | scalar | 0.720 |
| 16K | 4, combined reduction | 0.802 |
| 16K | 8, combined reduction | 0.765 |
| 32K | 4 | 1.099 |
| 32K | 8, ordinary tree | 0.902 |
| 32K | 16 | 0.842 |
| 32K | 32 | 0.888 |
| 32K | 8, combined reduction | 0.834 pooled median |
| 64K | 8 | 0.982 |
| 64K | 16 | 0.927 |
| 64K | 32 | 1.016 |
| 128K | 16 | 4.491 |
| 128K | 32 | 3.734 |

This validates the existing context schedule: 8 splits at 32K (with the new
combined reduction), 16 at 64K, and 32 at 128K. Per-centroid adaptive split
counts were neutral: inactive HIP blocks did less leaf work, but the maximum
grid and fixed-width first-stage reduction remained, so launch/reduction cost
did not shrink.

The 8K/16K rows are current-code five-repeat medians over 512 decode steps.
Forced splitting regresses the targeted phase by 13.6% at 8K and at least
6.2% at 16K. Median end-to-end decode regresses by 6.7% at 8K; at 16K the
lower-overhead split-4 path is effectively neutral but still 0.5% slower,
while its targeted phase is 11.4% slower. Automatic dispatch therefore keeps
the scalar two-tier leaf kernel below 32K.
