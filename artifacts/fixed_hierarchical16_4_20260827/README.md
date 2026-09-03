# Fixed adjacent hierarchy diagnostic (Qwen3.5-0.8B)

This tests whether regular adjacent pooling can be made hierarchical: fixed
groups of 16 tokens form parents, fixed groups of four form child pages, and
the original tokens remain exact leaves. The comparison also includes fixed
groups of eight as parents and a simpler variant that opens every leaf of each
selected T/16 parent.

All runs use BF16, a 512-token exact local window, exact leaf storage, coarse
mean K/V with `log(count)`, chat-formatted NIAH-S3 with thinking disabled, and
greedy decoding. `state_growth_factor=128` prevents similarity-based merging,
so the fixed adjacent groups remain independent. ProLong uses the same eight
deterministic 8K documents (65,528 prediction tokens) as the adjacent-2 and
adjacent-4 diagnostics.

## Results

| Method | Exact remote leaves per query | 8K NIAH-S3 | ProLong CE | ProLong PPL | PPL vs full |
|---|---:|---:|---:|---:|---:|
| Full attention | all | 64/64 | 3.252045 | 25.8431 | baseline |
| Similarity-routed KVM LOD, top-8 | variable | 64/64 | 3.254731 | 25.9126 | +0.27% |
| Fixed T/2, top-8 | 16 | 64/64 | 3.258194 | 26.0025 | +0.62% |
| Fixed T/4, top-8 | 32 | 64/64 | 3.272414 | 26.3749 | +2.06% |
| T/8 -> T/4, top-8 parents, one child per parent | 32 | 22/64 | 3.287456 | 26.7747 | +3.60% |
| T/16 -> T/4, top-4 parents, one child per parent | 16 | 17/64 | 3.302533 | 27.1814 | +5.18% |
| T/16 -> T/4, top-8 parents, one child per parent | 32 | 18/64 | 3.292793 | 26.9179 | +4.16% |
| T/16 -> T/4, page-mass-reranked top-16 parents | 64 | 18/64 | 3.282034 | 26.6299 | +3.04% |
| **Fixed T/16, top-8 parents, all leaves** | **128** | **63/64** | **3.273635** | **26.4072** | **+2.18%** |
| **Fixed T/16, top-16 parents, all leaves** | **256** | **64/64** | **3.265883** | **26.2032** | **+1.39%** |
| Fixed T/16, mass 1/128 capped at 16 parents, all leaves | variable | 54/64 | 3.323529 | 27.7581 | +7.41% |

At 64K, fixed T/16 top-8 with all leaves scored 7/8 NIAH-S3 and top-16 scored
8/8. The matched fixed T/2 and T/4 top-k diagnostics also scored 8/8, though
those panels contain only eight examples and should not be read as a precise
difference.

## Interpretation

The T/16 parent summaries are adequate for retrieval routing. Opening every
leaf of the top-eight T/16 parents restores 63/64 NIAH, whereas choosing only
one T/4 child per selected parent never exceeds 18/64 even after:

- increasing the parent route count from eight to sixteen;
- reranking parent candidates using the summed mass of their child pages; or
- shrinking parents from 16 tokens to eight.

The failure is therefore the per-parent one-child bottleneck and its residual
approximation, not primarily blurry T/16 parent routing. The all-leaves
variant is also much simpler: score T/16 parents, select eight or sixteen, and
exact 16 leaves from each. Top-8 ProLong loss is almost identical to flat T/4
top-8, while its coarse field has one quarter as many entries. Top-16 reduces
the perplexity gap to full attention from 2.18% to 1.39% and restores perfect
retrieval on both tested panels. The tradeoff is exact attention to 128 or 256
remote tokens per query, versus 32 for fixed T/4 top-8.

The 1/128 mass cutoff is not a quality-equivalent replacement for top-8 here.
It loses nine additional NIAH examples and increases perplexity substantially.

A plausible selective follow-up is to rank all T/4 children globally rather
than forcing exactly one child from each chosen parent. That removes the bad
allocation constraint, but it has not been tested in this panel. It should be
compared against the simple all-leaves result, which is the current reference
for this hierarchy.
