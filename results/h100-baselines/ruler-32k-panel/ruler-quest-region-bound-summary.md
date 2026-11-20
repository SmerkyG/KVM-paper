# QUEST-style bounds on the LoD semantic regions

This diagnostic replays the original Llama-3.1-8B-Instruct BF16 two-tier LoD
trajectories for source rows 78, 126, 302 and 447 of RULER multi-key 3. Both
LoD modes failed those four questions and dense succeeded. At the two-tier run's
first divergence, we evaluate the 32 layers' actual query heads and their
actual cached regions. The original model outputs remain unchanged; this is a
fixed-query routing analysis, not a full quality or speed run.

For each region we recompute exact per-channel minimum and maximum from all
its stored BF16 keys, accumulating in FP32. For native query `q` and model
scale `s`, the QUEST-style upper bound is

`U_r = s * sum_d max(q_d * min_{i in r}(k_i[d]), q_d * max_{i in r}(k_i[d]))`.

This bounds the largest individual token logit in the region, not the region's
actual softmax mass. We tested ranking the top eight regions by `U_r` and also
by `U_r + log(n_r)`; the latter is a conservative upper bound on log region
mass. The current LoD router uses its normalized query and count-corrected
mean-key score. The candidates use the same eight-region limit and existing
1,024-entry eligibility rule. All selections are scored with true dense
attention on the same query and cache. Bounds were verified against exact
maximum token logits for every nonempty region.

| Row | Layer / head | Region entries | Answer token logit | Mean of all region token logits | QUEST bound | Current rank | Bound rank | Bound + log count rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 78 | 15 / 30 | 14 | 7.60 | -13.15 | 32.06 | 37 | 79 | 90 |
| 126 | 15 / 30 | 21 | 9.99 | -12.90 | 42.66 | 17 | 25 | 34 |
| 302 | 15 / 30 | 18 | 11.54 | -11.95 | 52.56 | 23 | 46 | 58 |
| 447 | 25 / 13 | 9 | 8.49 | -4.02 | 24.72 | 31 | 308 | 347 |

The example token in each row is the most-attended answer token in the head
that pays the greatest total attention to answer tokens. The mean logits above
are computed from the original per-token keys, rather than the stored BF16
sum. In these four regions, cosine similarity between the highlighted answer
key and its mean key was 0.847, 0.815, 0.899 and 0.865. Thus the outlier's
query-specific attention score is not explained merely by a very low global
key-to-centroid cosine similarity.

The bound can be very loose: different keys may supply the per-channel
extrema, creating a hypothetical key that need not exist. Many other regions
also acquire high upper bounds and outrank the answer-containing region.
Comparing raw bound magnitudes against a centroid score is not a comparison of
probabilities; rank and exact selected-region mass are the meaningful tests.

Across the four selected examples' 4,096 head/layer queries, the top eight
regions captured an average of 5.75% of total dense attention mass using the
current router, 1.42% using the QUEST-style bound, and 1.21% using the bound
plus log count. (Local and sink attention is shared and is not included in
those remote-region figures.) Among 202 heads where a single answer token had
at least 10% of true dense attention, its region ranked in the top eight in
139 cases under current routing, 34 under the bound, and 22 under bound plus
log count. The examples were deliberately selected failures; these figures
are not a RULER accuracy measurement.

The FP32 bounds are computed as an oracle analysis from the full archive,
with no production storage quantization, bound-update cost, graph capture,
latency measurement, or autoregressive quality run. They do not reproduce
QUEST's contiguous page partition or constitute a benchmark of QUEST itself.
A method that uses bounds in smaller subdivisions, mixed with another ranking,
or with a larger opening budget may behave differently. Bounding metadata
would be used for selection, while the current key/value means would still be
needed for LoD's approximate softmax contribution.

[Per-layer/head data](ruler-quest-region-bound-details.json),
[experiment script](../../../benchmarks/retro_ruler_quest_region_probe.py),
[run record](1794112327438168379-ruler-quest-region-bound-details-run.json),
and [earlier frontier and cache validation](ruler-frontier-audit-summary.md).
Run `1794112327438168379-ruler-quest-region-bound-details` used `uv` via
`cluster-run`.
