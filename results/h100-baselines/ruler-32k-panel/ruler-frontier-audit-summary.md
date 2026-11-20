# Llama 32K multi-key 3: attention frontier audit

Four selected questions (source rows 78, 126, 302 and 447) failed in both
published LoD runs while dense succeeded. For each mode, we reproduced the
original prefill token and decoded eagerly through the first token where that
run diverged from the original dense run, checking every reproduced token
against the recorded LoD output. At that step, on each layer's actual LoD
query and KV state, we computed full dense attention and a PyTorch LoD
frontier. These are local comparisons on the LoD trajectory, not a replay of
the dense model's hidden states.

## Answer-token outliers in the actual two-tier cache

For each example, the table picks the head with the largest *total dense
attention* to the answer UUID's prompt tokens. Its highest-attended answer
token is in a region that the normal eight-region router does not open.
The estimated region mass is the region's centroid mass divided by the
**true dense partition function**; it is not the probability after LoD
renormalizes its approximate frontier. The final columns compare that head's
output with dense using relative L2 error. The oracle replaces the last of the
eight selected regions with the region containing the highest-attended answer
token, using answer knowledge and exact dense attention. It is not a runnable
routing policy and does not test whether generation becomes correct.

| Row | Layer / query head | Entries in region | Dense attention to that region | Centroid mass / dense partition | Region rank: current / native query | Two-tier output relative L2 error | Oracle-open error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 78 | 15 / 30 | 14 | 99.06% | 1.37e-8 | 37 / 91 | 91.3% | 0.30% |
| 126 | 15 / 30 | 21 | 99.73% | 2.48e-9 | 17 / 32 | 95.6% | 0.26% |
| 302 | 15 / 30 | 18 | 95.43% | 1.07e-9 | 23 / 31 | 96.6% | 2.11% |
| 447 | 25 / 13 | 9 | 88.17% | 3.00e-5 | 31 / 42 | 95.1% | 6.73% |

This identifies a concrete outlier failure: the mean-key region summary is a
very poor proxy for a region containing a single dominating key. In these
particular heads, selectively opening the correct region largely repairs the
local attention output. It does not establish that this is the only error or
that fixing this one head fixes the generated answer. The native-query rank is
a same-query counterfactual, not the result of rerunning the full model under
that routing policy.

The actual two-tier caches had 2,857–2,862 centroids in these traces. At the
four audited decode steps, averaged equally across all 4,096 query heads
(4 examples × 32 layers × 32 heads), true dense attention assigned 23.65%
to the remote archive, of which 5.75% of total attention was in selected
regions. The remaining remote entries were represented by summaries, not
dropped. This is a diagnostic average over deliberately failed examples, not
an estimate over all RULER tasks.

On the corresponding three-tier trajectories, the same averaging gave 24.61%
remote mass, 5.82% in selected regions and 5.25% in the selected pages. On
those same three-tier queries and caches, reference attention's mean
per-head relative L2 error was 18.10% for top-two pages and 17.62% when the
same eight selected regions were opened in full. Thus, for this selected
subset and these steps, page truncation is a smaller change than the error
remaining after whole-region opening. The separately run two-tier models have
their own K/V states; this counterfactual uses three-tier states deliberately.

## Verification and limits

The exact answer UUID token positions were independently mapped back to the
original prefill K and V and checked against the stored cache at every layer
for all four inputs and both modes. Page ownership and region counts were
checked for complete coverage and chronological page ordering. The PyTorch
reference and optimized decode attention on identical inputs had maximum
absolute per-component differences up to 0.03125 (two-tier) and 0.03272
(three-tier); their average per-layer mean absolute differences were 0.000125
and 0.000138 respectively. These checks do not rule out cache construction or
other errors. The traces were chosen because both modes failed, so they must
not be generalized to the other tasks or successful examples without testing.

Data: [two-tier oracle and routing details](ruler-frontier-oracle-gold-two.json),
[three-tier attention traces](ruler-frontier-audit-three-tier.json),
[two-tier answer-KV validation](ruler-frontier-gold-check-two.json), and
[three-tier answer-KV validation](ruler-frontier-gold-check-three.json).
The [audit script](../../../benchmarks/retro_ruler_frontier_audit.py) and
[KV validation script](../../../benchmarks/retro_ruler_gold_cache_check.py)
were run using `uv` via `cluster-run`; recorded runs include
[the two-tier oracle](1794111993192824218-ruler-frontier-oracle-gold-two-run.json)
and [the three-tier trace](1794111808863881474-ruler-frontier-audit-three-tier-run.json).
