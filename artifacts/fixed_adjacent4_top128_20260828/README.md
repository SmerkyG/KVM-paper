# Fixed-adjacent prefill experiments (2026-08-28)

All reported timings use Qwen3.5-0.8B, batch one, 16K chunked prefill, and
non-repeating ProLong document text. Full attention uses
`ROCM_AITER_UNIFIED_ATTN`. NIAH-S3 uses the model chat template, thinking
disabled, greedy decoding, and eight examples. Diagnostic cutoff simulations
perform many extra score scans and their wall times are not speed results.

## Length transition

| Context | Method | Prefill | Versus full | NIAH-S3 |
|---:|---|---:|---:|---:|
| 64K | Full AITER | 1.356 s | 1.00x | not rerun here |
| 64K | T/4 top-128, direct M16 query-tile union | 1.327 s | 1.02x | 7/8 |
| 128K | Full AITER | 4.727 s | 1.00x | 8/8 |
| 128K | T/4 top-128, exact expert leaves | 4.011 s | 1.18x | 8/8 |
| 128K | T/8 top-16, exact expert leaves | 2.499 s | 1.89x | 6/8 |
| 128K | T/16 top-16, exact expert leaves | 1.871 s | 2.53x | 6/8 |
| 128K | T/16 top-32, broad duplicate-coarse | 1.923 s | 2.46x | 5/8 |
| 256K | Full AITER | 17.436 s | 1.00x | 8/8 |
| 256K | T/16 top-16, exact expert leaves | 5.096 s | 3.42x | 8/8 |

The corrected 128K full-attention comparator is
`qwen08_full_aiter_128k_b1_r3.json`. The similarly named
`qwen08_full_128k_b1_r3.json` used the Triton backend and is not a valid AITER
comparison. Eight NIAH examples are only a smoke test: the observed result
supports a provisional T/4-to-T/16 transition at 256K, but not a production
quality claim.

The 64K direct-union experiment opens the union of the selected centroids over
each M16 query tile, so it intentionally performs more exact attention than
per-query expert routing. Its 7/8 result means it is not the current
high-quality path. The broad duplicate-coarse experiments intentionally retain
selected centroid summaries in the coarse result; the exact-expert experiments
retain the established selected-centroid subtraction semantics.

## Routing-cutoff diagnostics

For learned top-8 routing at a 4K-centroid state, 68.9% of raw routing dot
products were positive. Only 3.1% of 64-centroid tiles were entirely negative,
and 10.4% contained fewer than eight positive scores. Consequently, filtering
negative dots cannot skip enough score tiles, and a global compaction would
still retain 68.9% of the population.

Reusing one query's exact top-n boundary was also unstable. At period four and
zero margin it recovered 74.4% of learned top-8 entries while selecting 21.3
centroids on average. A margin of 1.0 recovered 90.8% while selecting 61.3.
For T/4 top-128 the analogous points were 73.1% recall / 542 selected and 90.5%
recall / 1,630 selected. These measurements motivate deriving a conservative
boundary from all queries in the preceding state-update segment instead.

The preceding segment's minimum **raw** boundary is safe but far too broad: it
selects 776 learned centroids for 99.997% top-8 recall and 8,442 adjacent T/4
centroids for 99.990% top-128 recall. Normalizing the boundary by the current
coarse-attention LSE removes most of that query-dependent baseline drift:

| Routes | Previous-segment statistic | Margin | Mean selected | Route recall | Rows with every top-n route |
|---|---:|---:|---:|---:|---:|
| learned top-8 | minimum | 0 | 136.3 | 99.987% | 99.947% |
| learned top-8 | 1st percentile | 0.25 | 65.1 | 99.905% | 99.632% |
| learned top-8 | 5th percentile | 0 | 19.2 | 98.631% | 95.399% |
| adjacent T/4 top-128 | minimum | 0 | 499.5 | 99.971% | 99.904% |
| adjacent T/4 top-128 | 1st percentile | 0 | 313.3 | 99.771% | 98.709% |
| adjacent T/4 top-128 | 5th percentile | 0.25 | 401.4 | 99.915% | 99.721% |

These full-state-LSE results are only a diagnostic upper bound. They do **not**
define an acceptable low-batch implementation: obtaining the current query's
LSE requires the extra complete coarse scan that this cutoff is intended to
remove.

A second diagnostic estimated the query-dependent baseline and scale from 64
uniformly spaced pilot centroids. Pilot log-sum-exp was too inaccurate. Pilot
mean and standard deviation (`pilot64_z`) worked substantially better for
learned top-8:

| Previous-segment statistic | Margin | Mean selected | Route recall | Rows with every top-8 route | Mean exact leaves | Versus exact top-8 leaves |
|---:|---:|---:|---:|---:|---:|---:|
| minimum | 0 | 102.3 | 99.876% | 99.647% | 5,473 | 5.54x |
| minimum | 0.25 | 200.7 | 99.999% | 99.997% | 8,942 | 9.05x |
| 1st percentile | 0 | 59.1 | 98.831% | 97.093% | 3,634 | 3.68x |
| 1st percentile | 0.25 | 124.3 | 99.973% | 99.923% | 6,216 | 6.29x |
| 5th percentile | 0.25 | 89.9 | 99.826% | 99.500% | 4,881 | 4.94x |

Exact learned top-8 visits 988 leaf tokens per query on average in the same
sample. Thus a 64-centroid pilot avoids another full scan and could replace
online top-8 with threshold comparison, but the safe cutoffs open roughly
5--9 times as many leaves. At the minimum/zero-margin point, however, the
complete regular-LoD attention population is only about 10K entries (4K
coarse + 512 local + 5.5K leaves), versus about 5.5K for top-8 and 64K for full
attention. This is therefore a viable speed candidate rather than a rejection:
it needs a real fused threshold-routing/fixed-mask kernel timing and quality
test. If the normalized cutoff continues to track the top-8 order statistic as
the state grows, its selected-centroid count can remain approximately constant
and its leaf work remains O(sqrt(T)). A tighter calibration statistic available
without a current-query coarse scan could improve it further; centroid moments
maintained at update time are one possible source.

## Production pilot-threshold decode

The candidate is now implemented in the regular two-tier vLLM decode path.
Prefill samples 64 queries and retains, per layer and query head, the minimum
standardized eighth-best centroid score. Each decode query scores 64 evenly
spaced pilot centroids to estimate its current score mean and standard
deviation, converts the retained standardized boundary to an absolute score,
and stamps every centroid above that boundary into the existing fixed-list
page-size-one attention mask. This removes the global online top-eight
selection and does not lag routes from the preceding decoded token.

The authoritative Qwen3.5-0.8B batch-one speed comparison uses a 65,280-token
real ProLong prompt, 256 generated tokens, warm prefix reuse, and medians of
seven runs. The earlier eight-token screens (including the apparent 1.969-ms
pilot result) are excluded because their fixed overhead made the small
difference misleading.

| 64K decode route | Decode ms/step | Versus matched top-8 |
|---|---:|---:|
| full AITER | 2.406 | +21.3% |
| current top-8 | **1.984** | - |
| pilot threshold, margin 0, separate reset launch | 2.008 | +1.20% |
| pilot threshold, margin 0, reset fused into final reduction | 1.998 | +0.71% |

Thus removing online top-eight is essentially latency-neutral in two-tier LOD,
but it is not a speed win on this model. The final pilot path is still 17.0%
lower latency than full attention. Its last-step audit before reset fusion
showed 126 centroid-union entries versus 32 for matched top-eight, demonstrating
that the large increase in exact work is almost fully hidden by the more
parallel route construction. The fused reducer advances the route epoch and
clears the next queue without a separate launch.

The zero-margin implementation scored **64/64** on chat-formatted 64K NIAH-S3
at batch eight. After fusing queue reset, the unchanged selection rule scored
8/8 in a fresh smoke test. Both execution audits confirmed the intended HIP
GQA union, fixed-mask, and page-size-one final-attention paths.

The existing recursive three-tier Qwen decode remains faster at 64K batch one
(1.868 ms/step, 6.5% below the final pilot latency). The pilot boundary cannot
simply replace its fixed eight
routes: recursive decode currently allocates one page-selection and reduction
split per route. A useful three-tier transfer should instead stamp the
variable centroid union, score pages only within that union, and feed the
result to a fixed-list page/leaf scan. That may benefit more from the broader
cutoff because page selection can suppress its extra leaf work, but it is a
separate variable-route recursive kernel rather than a configuration-only
test.

## Pilot threshold on adjacent T/4

The same production calibration now accepts a configurable reference order
statistic.  Setting the state premerge factor to four and the reference route
count to 128 applies it to the adjacent-T/4 experiment: prefill calibrates the
standardized 128th score and decode uses the current query's 64-centroid pilot
tile.  This is a decode experiment; the N/4 prefill path itself still uses its
existing exact top-128 selection.

At 64K, the adjacent state contained 15,424 live four-token groups.  The
authoritative batch-one timing again uses real ProLong text, 256 decoded
tokens, warm prefix reuse, and five repeats:

| 64K decode route | Decode ms/step | Versus full |
|---|---:|---:|
| full AITER | 2.406 | 1.00x |
| ordinary two-tier top-8 | **1.984** | 1.21x faster |
| ordinary two-tier pilot threshold | 1.998 | 1.20x faster |
| adjacent T/4 pilot, fixed-list fast-fail | 2.253 | 1.07x faster |
| adjacent T/4 pilot, compact selected list | 3.537 | 1.47x slower |

This is negative relative to ordinary LOD.  The fixed-list path was 12.8%
slower than the ordinary pilot and scanned a persistent 98,560-entry arena
(leaves plus coarse/local storage), relying on the mask to fast-fail unopened
tiles.  Compacting selected entries was worse: the four-head GQA union averaged
11,062 centroids and peaked at 12,031, or roughly 72% of the live adjacent
state.  Its exact-list capacity was consequently saturated.  The pilot
threshold therefore did not preserve the intended top-128 selectivity in the
production N/4 decode path, despite the earlier offline previous-segment
simulation predicting only about 500 selected groups per head.

The fixed-list path nevertheless scored **8/8** on chat-formatted 64K NIAH-S3
at batch eight, after correcting an early-prefill stale calibration shape
guard.  Prefix-reuse probes diverged from the cold generated output after nine
tokens for the fixed path and three for the overflowing compact path, however.
Thus the small NIAH smoke test passed, but these N/4 timings are diagnostic
rather than a validated high-quality configuration.
The likely next correction is to make the calibration population exactly match
decode eligibility and to use a less extreme cross-query statistic than the
minimum standardized boundary; simply increasing compact-list capacity would
make this experiment slower without repairing its selection rule.
