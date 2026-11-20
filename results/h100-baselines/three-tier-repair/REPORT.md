# H100 three-tier page thresholds and retrieval diagnosis

Page thresholds recover the targeted multikey loss and run faster than fixed
four-page refinement. Optimized fixed two-page and two-tier decode remain faster.
Both page-threshold candidates now have complete 13-task panels. Serving
defaults are unchanged.

## Scope

Llama-3.1-8B-Instruct, BF16 model and BF16 KV, batch one, H100 80GB. Quality
uses the frozen 32K RULER cohort, 32 examples for each of 13 tasks. First-response
scores stop at the first EOS/EOT; later corrections receive no credit. Full
fixed-length ignore-EOS scores are retained separately in the audit.

All LoD arms retain top-eight sparse prefill, the exact 16,384-token initial
chunk, local/sink attention, and the existing 1,024-entry region eligibility
limit. Region routing uses RMS-normalized queries. Final attention uses native
queries. Region and page multipliers are separate settings. There is no new
cap on the number of selected eligible regions or pages. All GPU experiments
use uv and cluster-run, immutable source snapshots and serialized GPU access.

## Why two pages fail

Fresh fastest two-tier c=16 gets 100% on multikey-3 (32 examples). Fixed
two-page three-tier gets 90.625%, missing rows 59, 316 and 455. The earlier
93.75% result used 16 examples and is a different sample size.

Same-query diagnostics reconstruct the real cache at the first wrong output
token. In all three failures, the correct region is opened, but the answer page
ranks third or fourth by its average-key proxy despite ranking first by true
attention mass. Two-tier reads the whole region; two-page refinement excludes
the crucial page. Four pages correct all three examples.

| Source row | First divergent output position, zero-based | Layer / query head | Dense mass on answer tokens | Region entries | Page proxy rank | True page-mass rank |
|---|---:|---|---:|---:|---:|---:|
| 59 | 25 | 15 / 30 | 99.89% | 143 | 4 | 1 |
| 316 | 20 | 15 / 30 | 98.35% | 300 | 4 | 1 |
| 455 | 20 | 16 / 1 | 99.50% | 202 | 3 | 1 |

The kernel agrees with the independent PyTorch frontier within BF16 rounding
(maximum absolute attention-output differences below 0.008). Reconstructing
sums from original BF16 leaves barely changes these failures. Channel bounds,
representative outliers and diagonal variance corrections were insufficient
as two-page fixes. Exact page-mass/max policies are **oracles** that read all
leaves and are never used in efficient generation. See [the traces](three-tier-repair-diagnose-v3.json).

## Adaptive page rule and implementation

For an opened region r with m_r pages:

    a_p = s(q_route, mean_key_p) + log(n_p)
    u_p = exp(a_p) / sum_{j in pages(r)} exp(a_j)
    open p iff u_p > c_page / m_r.

The promising setting is q_route = q/RMS(q), c_page = 0.5: open pages above
half their region's average page proxy mass. A sole page is always exact.
Unopened pages share one disjoint residual summary. Native q determines exact,
residual and closed-region attention. There is no two-page minimum or hard
page-count cap.

The kernel loads page summaries once, computes scores and their normalizer,
reduces decisions into a 64-bit mask, then iterates its set bits in increasing
page order to read exact KV. The region eligibility limit bounds each list to
64 sixteen-entry pages. This avoids repeated page ranking, sorting, quadratic
packing and GPU-to-CPU count reads. No new persistent cache statistic is needed.
CUDA graphs capture the same complete attention call.

The count-weighted alternative u_p > c_page*n_p/n_r was also tested. Neither
page rule guarantees accuracy. The tested equal-per-region outer rule does
**not** inherit the earlier per-entry rule's deterministic sublinear bound on
exact KV work. A below-average page cutoff can open every page.

## Targeted quality ablations

First-response percentages, the same 32 examples per task in every row:

| Region multiplier | Page rule | Page query | Multikey-3 | Multi-query |
|---:|---|---|---:|---:|
| 16 | Two-tier, whole selected regions | — | 100.00 | 97.66 |
| 16 | Fixed two pages | Native | 90.63 | — |
| 16 | Fixed four pages | Native | 100.00 | 95.31 |
| 16 | Mean-page threshold 0.25 | Native | 96.88 | 95.31 |
| 16 | Mean-page threshold 0.5 | Native | 93.75 | 98.44 |
| 16 | Mean-page threshold 1.0 | Native | 71.88 | 98.44 |
| 16 | Per-entry page threshold 0.5 | Native | 87.50 | 95.31 |
| 16 | Mean-page threshold 0.5 | RMS-normalized | 100.00 | 95.31 |
| 16 | Per-entry page threshold 0.5 | RMS-normalized | 96.88 | 98.44 |
| 8 | Fixed two pages | Native | 93.75 | 97.66 |
| 4 | Fixed two pages | Native | 93.75 | 98.44 |
| 8 | Mean-page threshold 0.5 | RMS-normalized | 100.00 | 97.66 |

The remaining native-page 0.25 miss, row 447, illustrates the outer routing
limitation: selected pages cover 47.19% of gold-token dense mass at the first
wrong digit, versus 47.25% for two-tier on the **same query/cache**. Most missing
mass lies in closed regions even where the answer page would rank first.
This is a local diagnostic, not a claim that an independent two-tier generation
also fails. See [the remaining miss](three-tier-repair-page-miss-diagnose-v1.json).

## Full RULER panel

All six configurations have complete 13-task panels, with 32 matched examples per task. Values are first-response percentages; the average weights tasks equally. Both adaptive page columns use normalized page queries and page multiplier 0.5.

| Task | Dense | RetroInfer | Two-tier c16 | Four pages | Page threshold c16 | Page threshold c8 |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_single_3 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_2 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_3 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multivalue | 96.88 | 98.44 | 99.22 | 99.22 | 99.22 | 99.22 |
| niah_multiquery | 99.22 | 98.44 | 97.66 | 95.31 | 95.31 | 97.66 |
| vt | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| fwe | 90.62 | 86.46 | 88.54 | 89.58 | 88.54 | 88.54 |
| cwe | 37.50 | 36.25 | 29.38 | 27.50 | 28.75 | 31.25 |
| qa_1 | 81.25 | 81.25 | 68.75 | 68.75 | 68.75 | 68.75 |
| qa_2 | 50.00 | 50.00 | 46.88 | 46.88 | 46.88 | 46.88 |
| **RULER average** | **88.88** | **88.53** | **86.96** | **86.71** | **86.73** | **87.10** |

The two-tier column is a fresh rerun of the optimized materialized decoder. Its older 87.04% result is not substituted for this control. RetroInfer uses dense prefill; these compare complete method configurations, not decode alone with identical prefill. The 32-example task size limits precision. Page settings were selected using multikey-3 and multi-query, which are reused here; this is not an independent holdout evaluation.

Paired comparisons and bootstrap intervals: [four pages](three-tier-repair-four-page-full13-summary.md), [page threshold with region c16](three-tier-repair-normpage-full13-summary.md), [page threshold with region c8](three-tier-repair-joint8-full13-summary.md).

The complete first-response audit and SUMMARY.json retain raw file hashes and per-task values.

## Complete attention-call timings

Rows share real queries and caches within each context from layers 0, 15 and
31. Values average these layers after taking the median of three graph-timing
repetitions per layer. Scope includes routing, coarse/page/exact attention,
local/sink, and identical current-token append/rollback. It excludes model
layers and periodic cache maintenance. Counts are remote exact entries per
query head at these sampled states, **not** average work over full responses.

| Frontier, region c=16 | 32K attention | 32K exact entries | 64K attention | 64K exact entries |
|---|---:|---:|---:|---:|
| Fixed two pages | 36.35 µs | 348.31 | 44.59 µs | 639.54 |
| Fixed four pages | 67.49 µs | 512.38 | 100.72 µs | 940.70 |
| Normalized page threshold 0.5 | 49.35 µs | 587.52 | 86.23 µs | 1,245.14 |

Thresholds outperform this four-page implementation despite reading more
entries, by avoiding repeated page ranking. This does not establish a lower
bound on a differently optimized fixed-four implementation. Top-two retains
its specialized runner-up optimization and remains faster.

Matched raw probes: [32K](three-tier-repair-stable-norm-v1-page-probe-32k.json),
[64K](three-tier-repair-stable-norm-v1-page-probe-64k.json). Their optional
remaining-page residual rows are a separate ablation.

## Full-model decode with amortized updates

Complete-model CUDA graphs, 512 untimed steps followed by 1,024 measured steps,
one warmup repetition and two measured repetitions. Each measured repetition
contains 128 layer catch-up events. Appends, catch-up and cached-mean refreshes
are included. Prefill and graph capture are outside decode timing.

| Three-tier frontier | Region multiplier | 32K ms/token | 64K ms/token |
|---|---:|---:|---:|
| Fixed two pages | 16 | 7.714 | 7.943 |
| Fixed four pages | 16 | 10.001 | 10.579 |
| Normalized page threshold 0.5 | 16 | 8.498 | 9.172 |
| Fixed two pages | 8 | 8.207 | 8.481 |
| Normalized page threshold 0.5 | 8 | 8.969 | 9.986 |

At c=16 thresholds reduce full-model latency versus four pages by 15.0% at
32K and 13.3% at 64K, but cost 10.2% and 15.5% versus fixed two pages.
Prior-session optimized two-tier times are 7.236/7.504 ms/token in the
[two-tier transfer report](../amd-transfer/REPORT.md), on the same hardware
and timing boundaries; its quality has now been refreshed here. A fresh,
amortization-matched RetroInfer timing is not part of this investigation.
Older timings cannot establish a new speed win over RetroInfer.

## Residual precision ablation

BF16 region-minus-opened-page subtraction can corrupt a small residual. In a
controlled valid cache, 257 identical keys of value 3 sum to 771, stored as BF16
772. Removing 256 keys in full pages subtracts 768, inventing a residual key of
4 instead of 3. The old GPU output is 0.996094 versus dense 0.00389105; summing
unopened page summaries in FP32 gives 0.00389099. This is a numerical fixture,
**not** a model-quality example. [GPU regression](three-tier-repair-stable-residual-check-v1.json).

The opt-in LOD_H100_PAGE_RESIDUAL=pages path reuses loaded page keys and reads
remaining page-value summaries only for a nonempty residual, with no new cache
state. Independent checks pass. On the targeted 64-example panel, stable
residuals leave normalized-page scores unchanged (100.00/95.31); native-page
multikey falls from 93.75 to 87.50. The normalized stable variant costs about
4–5% in real-query probes. It remains an explicit numerical ablation, not a
demonstrated quality improvement. The three initial failures principally show
page undercoverage; this fixture does not establish their cause.

## Reproduction and validation limits

The BF16 threshold path is opt-in before cache allocation:

```bash
LOD_H100_THRESHOLD_IMPLEMENTATION=three-tier
LOD_H100_THRESHOLD_RULE=region
LOD_H100_THRESHOLD_MULTIPLIER=16
LOD_H100_PAGE_THRESHOLD=0.5
LOD_H100_PAGE_THRESHOLD_RULE=page
LOD_H100_PAGE_QUERY=normalized
LOD_H100_PAGE_RESIDUAL=region
LOD_H100_THREE_TIER_WORKERS=32
LOD_H100_PAGE_THRESHOLD_WARPS=2
```

The c=8 candidate uses 64 workers with the same page settings. Exact commands,
geometry, GPU identity, dependency gates and hashes are in the *-runs.json
manifests and provenance/. The run.json source_files map reconstructs every
frozen source from source-objects/<sha256>; executed-source/ provides convenient
copies of key files. The first-response audit hashes every scored input file.
The queued per-task analyses and local collector finalize this report after
both full panels pass. finalization-status.json records progress or failure;
failed dependencies prevent publication of a complete-panel average.
The raw quality field page_threshold_work samples all heads/layers at the final
fixed-length decode step, which can be after EOS. It is not a first-response
average; the work-count table above uses matched real-query probes instead.

Independent PyTorch frontier checks cover empty selections, oversized regions,
partial/overflow pages, ties, cached/uncached means, appends and graph replay.
Real-query probes check outputs and exact work counts. Same-cache model checks
compare cached and uncached region-mean paths at 17 steps per context, including
update boundaries. They do not replace quality evaluation.

status.json retains failed, skipped and canceled trials; none contributes to
tables. Early larger-budget kernels failed compilation; scalar page-rank history
fixes the validated four-page path while retaining optimized top-two selection.
Eight-page validation was stopped; there is no eight-page quality claim.
Early model runs missing RetroInfer's module path and an overly strict probe
reference tolerance were superseded by successful checked runs.

This page-threshold implementation supports BF16 batch-one Llama GQA4 only;
unsupported precision/routing raises an error. Quality is at 32K; 64K has timing
and numerical/model checks only. No serving defaults, AMD quality claims, or
complete-execution subquadratic guarantees follow. Prefill and amortized cache
work remain present and timed.
