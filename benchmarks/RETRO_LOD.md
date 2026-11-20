# LoD in RetroInfer's Llama runtime

**LoD result validity:** Historical H100 LoD measurements below used a
state-update kernel later found capable of dropping value-summary updates.
Treat those older LoD rows as timing-only, not validated model results, until
rerun with the synchronization fix. The current threshold sections, batch-four results below and the
paper-style results in [MULTIPOLE.md](MULTIPOLE.md) use the corrected kernel.
This LoD kernel issue does not affect the
Multipole and dense measurements; an earlier Multipole graph-replay issue is
described in the synthetic comparison.

**Measurement scope:** The Qwen-shaped graph tables in [MULTIPOLE.md](MULTIPOLE.md) are synthetic
single-layer measurements. The historical Llama tables report full-model
results and use different prompts, contexts and timing boundaries. A full
batch-eight 128K BF16 Llama model does not fit on the 80 GB H100.

Work in progress. This experiment runs LoD three-tier top-two attention inside
the released RetroInfer Llama-3.1 model execution. The benchmark-only adapter
reuses the author's QKV and output projections, RoPE, model weights, MLP,
sampling and decode loop. It installs the production LoD per-layer cache and
routing kernels at the attention boundary. LoD's prefill construction and
periodic decode cache updates are timed; initial cache allocation is outside
timed prefill, as it is for the author's comparison paths.

## Multipole paper-component comparison

The current graph-captured Qwen-shaped comparison, including local and sink
variants and isolated update accounting, is in [MULTIPOLE.md](MULTIPOLE.md).
The [Llama shared-centroid comparison](../results/h100-baselines/llama-shared-index/REPORT.md)
uses one optimized LoD prefill for dense, two-tier LoD, Multipole, and Squeezed
p90. Multipole and Squeezed share the initial 2,900-centroid index.

## ClusterKV on Llama-3.1-8B-Instruct

The [32K ClusterKV comparison](../results/h100-baselines/clusterkv/REPORT.md#whole-model-cuda-graph-comparison)
uses whole-model CUDA graphs at batch one for ClusterKV, dense, GPU-only
RetroInfer sparse decode, and two-tier LoD. Timed decode includes four ClusterKV
window updates and one RetroInfer index rebuild with graph recapture. The report
includes precision-matched dense controls for FP16 ClusterKV and BF16 LoD,
plus the earlier eager comparison and paired 13-task RULER results. The
[32K batch-eight retime](../results/h100-baselines/clusterkv/REPORT.md#batch-eight-cuda-graph-comparison)
uses the same 1,024-step decode window and reports both batch-step latency
and aggregate output throughput. The [optimized LoD timings at both batch sizes](../results/h100-baselines/clusterkv/REPORT.md#optimized-lod-at-batch-one-and-eight)
cover two-tier BF16, fixed-page three-tier BF16 and INT4, and adaptive-page
three-tier BF16. The report also shows retimes of the earlier batch-one settings.
Selecting `LOD_H100_PAGE_THRESHOLD=0.5` for BF16 three-tier threshold decode
on H100 uses the optimized kernel and RMS-normalized page routing by default.
Set `LOD_H100_ADAPTIVE_TUNED_DEFAULTS=0` to restore the earlier defaults.

## Uncapped two-tier threshold decode

The latest [AMD-derived H100 decode report](../results/h100-baselines/amd-transfer/REPORT.md)
compares the current indexed H100 implementation with cached means, persistent
leaf indices, fused threshold/coarse packing and shared-GQA tensor-core attention.
It keeps the per-region c=16 rule and top-eight prefill. Fresh controls bracket
the candidate at batch one, 32K and 64K; complete-model graph timings include
cache catch-up and the candidate's index rebuilds over 1,024 measured decode
steps. The report also records fixed-query checks, sampled model checks and
the separate prefill cost. This candidate supports per-region thresholds;
entry-16 retains its earlier implementation. Serving defaults are unchanged.

| Context | Current H100 c=16 decode | AMD-derived c=16 decode | Latency reduction |
|---|---:|---:|---:|
| 32K | 7.529 ms/token | 7.236 ms/token | 3.9% |
| 64K | 7.950 ms/token | 7.504 ms/token | 5.6% |

Both rows use Llama-3.1-8B-Instruct, BF16, and the corrected cache-update
kernel. These are complete-model timings with updates included, separate from
the report's 32–39% attention-call reductions. A subsequent
[full 13-task RULER rerun](../results/h100-baselines/three-tier-repair/REPORT.md)
of this implementation scores 86.96% on first responses, versus 88.53% for
RetroInfer and 88.88% for dense, using 32 matched examples per task at 32K.

The earlier [threshold decode optimization report](../results/h100-baselines/decode-fast/REPORT.md)
compares the prior and optimized per-region c=16 and entry-16 implementations
at batch one, 32K and 64K, alongside fresh dense and fixed top-eight controls.
These runs use BF16 and the corrected cache-update kernel. The LoD paths capture complete model steps and include cache catch-up
over 1,024 measured decode steps after 512 untimed steps. Selection remains
uncapped; the report separates fixed-query attention diagnostics from full-model
timings and records the reference checks. The optimization remains opt-in.

## H100 adaptive page refinement

The [page-threshold investigation](../results/h100-baselines/three-tier-repair/REPORT.md)
diagnoses the three-tier retrieval deficit: the answer page can rank third or
fourth by its mean-key proxy even when the correct region is opened. In a
32-example BF16 multikey-3 rerun, fixed two pages score 90.63%, while fixed
four pages and RMS-normalized page thresholds score 100%.

Opening pages above half the region's average page proxy mass avoids sorting
and repeated ranking. At region c=16 it takes 8.498/9.172 ms per token at
32K/64K, versus 10.001/10.579 ms for fixed four pages and 7.714/7.943 ms for
fixed two pages. These full-model graphs include amortized cache updates.
The faster page-threshold candidate still has a multi-query deficit; region
c=8 recovers the targeted two-tier scores with more exact work, taking
8.969/9.986 ms per token. The full 13-task first-response averages are 86.73% at c=16 and 87.10% at c=8, versus 86.96% for the fresh two-tier control. The BF16 page path is experimental;
serving defaults remain unchanged. The linked report separates quality,
matched attention probes, residual-precision ablations and implementation limits.

## H100 three-tier threshold decode

The [three-tier AMD-to-H100 report](../results/h100-baselines/three-tier-transfer/REPORT.md)
ports threshold selection, linear region packing, persistent top-two-page
workers, the dimension-blocked merge and cached region means to Llama GQA.
These are batch-one Llama-3.1-8B-Instruct full-model decode times in ms/token,
with complete-model graphs and amortized cache updates included. Fixed top-eight
controls bracket each candidate; dense controls use the same GPU and prompts.

| Context | Dense | Top-eight BF16 | Threshold BF16 | Top-eight INT4 | Threshold INT4 |
|---|---:|---:|---:|---:|---:|
| 32K | 8.725 | 7.828 | 7.715 | 7.780 | 7.562 |
| 64K | 10.087 | 8.268 | 7.943 | 8.122 | 7.876 |

Threshold routing uses per-region c=16 with uncapped selections. The full-model
reduction versus top-eight is 1.4–2.8% at 32K and 3.0–3.9% at 64K. Each timed
window has 1,024 decode steps after 512 untimed steps and includes 128 layer
cache-update events. Mean caching alone reduces latency by 0.5–1.7% with
identical routing and settings; the report separates that comparison.

On first-response scoring in the limited 32K retrieval panel, `niah_multikey_3`
improves from 43.75% to 93.75% in BF16 and from 31.25% to 87.50% in INT4.
`niah_multivalue` stays at
100% in both precisions. Each task has 16 examples, for 32 examples per
precision. One INT4 example regresses despite the aggregate gain; both remaining
misses were rerun without mean caching and produced identical tokens. This is
a two-task panel, separate from full RULER averages and 64K quality. The earlier
93.75% INT4 multikey score included text after EOS/EOT; the linked report retains
both scoring variants and the per-example scoring audit.

The 24 independent frontier cases and 68 sampled same-cache model checks pass.
Prefill keeps top-eight routing, all selected-region leaves and the exact
16,384-token initial chunk. Decode keeps two exact pages plus a disjoint
residual per opened region, along with local and sink attention. The path is
opt-in in the RetroInfer adapter; serving defaults are unchanged. Enable flags,
prefill costs, fixed-query diagnostics, source audits and all raw runs are in
the linked report.

## H100 two-tier threshold prefill

<!-- direct-opt-current -->
The [current optimized threshold results](../results/h100-baselines/threshold-next/REPORT.md#latest-optimized-threshold-prefill) retain the shared saved-score selection/closed-weight pass and globally shifted FP16 coarse projection. Small exact frontiers now use a tensor-core reduction with three BF16 parts of each FP32 weight, falling back to the original uncapped reducer above 32 routes. Cache-key normalization reproduces PyTorch 2.5 rounding, and a packed max/index reduction consumes the original cuBLAS state scores. Larger adaptive scratch tiles and producer pipelining also reduce overhead. Faster candidates that changed coarse arithmetic or cache assignment lost quality and were rejected.

BF16 model and KV cache, batch one, same H100 and frozen prompt per length. One warmup and three measured repetitions; medians include full cache completion. The equal-per-region c=16 rule remains uncapped, with the unchanged initial 16,384-token exact field, local/sink fields and update schedule. Coarse projection operands are FP16 with FP32 accumulation and division afterward.

| Context tokens | Dense | Top-eight | Previous optimized threshold | New threshold | New versus dense |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 1.553 s | 1.622 s | 1.756 s | 1.663 s | 7.1% more time |
| 65,536 | 4.749 s | 4.101 s | 4.737 s | 4.518 s | 4.9% less time |
| 128,000 | 15.697 s | 9.420 s | 11.816 s | 11.528 s | 26.6% less time |

The new threshold path beats dense at 64K; it remains 7.1% slower at 32K and 2.5% slower than the fresh top-eight control there. Prefill is eager for every arm, including all cache updates/completion; loading, initial cache allocation and decode graph capture are excluded consistently. Top-eight uses the same cache normalization/max improvements. Peak PyTorch allocation, including weights, is **64.20 GiB / 70.68 GiB** at 32K/64K, versus 33.43/41.06 GiB for the preceding threshold configuration; external driver/library allocations are excluded.

The new full 32-example CWE check scores **35.94%**, versus **34.38%** for the fresh previous-selector control. Two examples each of multikey-2, multikey-3 and QA-1 retain 100%. Individual CWE scores change in both directions; these limited checks are not a new full 13-task RULER average or evidence of an improved routing rule. Sampled frontiers have zero selection differences outside tolerance, and same-cache eager/graph decode logits match. The [optimization retime report](../results/h100-baselines/direct-opt-v3/REPORT.md) records the enable flags, numerical checks, raw samples and rejected trials. The new overrides are experimental; serving defaults are unchanged.

The 128,000-token retime uses **4,096-query scratch tiles throughout** for memory headroom, retaining the exact 16,384-token initial field. It is **2.4% faster** than the preceding threshold version and uses **26.6% less time than dense**, while remaining 22.4% slower than top-eight. Peak PyTorch allocation is **66.44 GiB**, including weights and excluding external driver/library allocations. All 224 sampled frontiers have zero selection differences outside tolerance, maximum reference attention error 0.015625, and exact same-cache eager/graph logits. Dense and top-eight are fresh controls; the previous-threshold column uses the earlier completed matched retime. The longest prompt is 128,000, not 131,072 tokens. No new 128K quality panel has been run.

<details>
<summary>Earlier threshold-prefill results and work audits</summary>

The [64K selection follow-up](../results/h100-baselines/selectpack-64k/REPORT.md)
reduces the separate selection/packing stage from **0.711 s to 0.472 s**
by marking conservative candidates during the producer's existing score pass,
then reading candidate scores from compacted bitmap words. It retains every
qualifying c=16 region and the previous coarse-attention arithmetic. Fresh
64K full-model medians are **5.366 s before → 5.237 s after (2.4%)**, versus
**4.750 s dense** and **4.159 s top-eight**. It still does not beat dense at
64K. All eight first-response RULER scores match on the small 32K check;
six first-response token sequences match exactly. This is not a full RULER rerun.

Within the opt-in fast threshold path, automatic dispatch now uses this
prefix filter at 2,048–4,095 active region slots, the previously optimized
scan at 4,096 or more, and the original pass below 2,048. Additional automatic
runs take **1.899 s at 32K** and **13.246 s at 128,000 tokens**. The exact
initial 16,384-token field, local/sink attention and cache updates are retained.
The linked report separates full-model medians from stage pilots and records
the enable flags, source hashes and full-model reference checks.

The [normalization and offset follow-up](../results/h100-baselines/closed-offset/REPORT.md)
provides a faster direct-closed-summary candidate: exponentials are centered
at the closed maximum, projected once using FP16 operands with FP32 output,
and normalized afterward in FP32. Model weights, KV storage, routing and
exact attention remain BF16. Fresh matched 64K medians are **5.019 s** for
this candidate, **4.996 s** for the previous direct BF16 path, **5.225 s**
for conservative prefix selection and **4.752 s dense**. The candidate keeps
the direct path's speed within 0.5%, but remains 5.6% slower than dense.
On all 32 frozen CWE examples at 32K, it scores **32.81%**, versus
**30.00%** conservative, **27.19%** previous direct BF16 and **37.50%** dense.
Six multikey/QA correctness checks retain their scores. These are CWE-task
results and small retrieval/QA checks, not a new full RULER average.
Serving defaults and automatic threshold dispatch remain unchanged pending
broader quality validation; enable flags and executed commands are in the report.

The [context-length scaling check](../results/h100-baselines/closed-offset/REPORT.md#context-length-scaling)
measures **1.824 s at 32,768 tokens** and **12.258 s at 128,000**, versus
**1.560 s / 15.703 s dense** and **1.899 s / 13.226 s conservative threshold**.
Full-prefill times differ by only +0.22%/+0.32% from previous direct-BF16 timings at
these lengths. At 128,000 tokens it uses **21.9% less prefill time than dense**
and **7.3% less than conservative threshold**, with 66.4 GiB peak PyTorch
allocation on the 80GB H100. These runs include full cache completion and
retain the 16K exact initial field. Reference checks at both lengths and
same-cache eager/graph checks pass; these are timing and numerical checks,
not additional RULER quality panels.

The earlier [controlled arithmetic follow-up](../results/h100-baselines/selectpack-64k/COMBINATION.md)
isolated the original CWE example's difference to the coarse-attention
rewrite. A normalized split-BF16 projection did not recover its missed words.
Reconstructing the old online rounding/correction coefficients through two
cuBLAS projections recovered aggregate CWE quality but took **5.535 s at 64K**.
The normalization follow-up above supersedes that speed/quality tradeoff;
the original ablations remain available for provenance.

The [selection/packing follow-up](../results/h100-baselines/selectpack/REPORT.md)
reduces 128,000-token fused threshold prefill from **14.269 s to 13.437 s
(5.8%)** in a fresh matched pair. Selection/packing itself falls from
**2.886 s to 2.081 s (27.9%)** over all layers. The selected implementation
uses 32-query × 256-region tiles, four warps, device prefix scans and an
empty-tile packing shortcut; the opt-in fused threshold path automatically
uses it from 4,096 active regions onward. That earlier comparison retained
the original pass at 32K and 64K; the prefix-filter results above supersede
that small-context setting. The c=16 rule remains
uncapped, with the same 1,024-entry eligible-region limit, exact initial
16,384-token prefill, local/sink fields and cache updates. These new timings
remain slower than the top-eight reference below. The linked report records
the correctness checks separately from the historical full RULER scores.

The [earlier full-panel follow-up](../results/h100-baselines/threshold-next/REPORT.md) tests
further optimizations, full-model prefill at 32K/64K/128,000 tokens, and fresh
combined threshold-prefill/decode RULER quality. It also records the separate
Qwen real-state graph and quality comparisons with Multipole/Squeezed. The fresh
combined threshold-prefill/decode run scores **87.52%** over all 13 tasks with
32 examples/task, versus 88.53% RetroInfer and 88.88% dense. The historical
88.02% result belongs to the earlier implementation. On the four-task Qwen
panel, threshold prefill raises CWE from 71.56% to 78.44%, versus 79.38% dense,
with the same optimized threshold decoder. The linked report records the
128K benchmark cache-cleanup investigation separately from timing conclusions.

With the benchmark cleanup fixed, three measured 128,000-token repetitions
complete at **14.555 s** for closed threshold prefill, versus **9.535 s** for
top-eight and **15.695 s** for dense. Peak PyTorch allocations are 63.47,
41.35 and 34.36 GiB respectively. The initial failure retained the previous
cache between repetitions; collection after the benchmark frame returns fixes
that overlap outside timing. The threshold scratch tile remains 4,096 queries,
and the exact initial prefill field remains 16,384 tokens. Threshold still
uses more scratch memory and is slower than top-eight, but fits and beats
dense by 7.3% at this length.

Fused threshold also completes all three measured repetitions after enabling
expandable allocation segments: **14.277 s**, with 65.38 GiB peak PyTorch
allocation, versus 9.535 s top-eight and 15.695 s dense. The
[full-prefill work audit](../results/h100-baselines/threshold-next/REPORT.md)
counts 22.62 opened regions and 1,466 exact entries per remote query/head at
128K, versus 7.98 and 524 for top-eight. At 32K the corresponding entry counts
are 313 and 262. Diagnostic GPU events attribute 2.897 s to threshold
selection/packing at 128K, 1.999 s to exact KV kernels, 0.681 s to separate
route reduction and 0.963 s to correction/merge. Top-eight exact KV kernels
take 0.718 s; its route reduction is part of its final merge. These are
instrumented stage totals, distinct from the full-model timing medians.

The [threshold-prefill report](../results/h100-baselines/prefill-closed/REPORT.md)
compares fresh Llama-3.1-8B-Instruct BF16 batch-one runs on the same H100.
These are full-model prefill times, median of three warmed repetitions, including
explicit final cache catch-up for LoD. The exact initial 16,384-token chunk is
unchanged; all LoD arms use the same per-region c=16 decode. The dense control
uses native dense attention and its own cache.

| Prefill routing | 32K | 64K |
|---|---:|---:|
| Dense | 1.553 s | 4.760 s |
| Fixed top-eight | 1.648 s | 4.181 s |
| Per-region c=16, previous fused implementation | 2.016 s | 5.673 s |
| Per-region c=16, optimized | 1.880 s | 5.467 s |
| Entry-16, previous optimized implementation | 2.498 s | 7.525 s |

The per-region reduction is 6.8% at 32K and 3.6% at 64K. At 32K the selected
path computes closed summaries directly; at 64K it reuses the fused coarse
producer, uses a parallel prefix scan for selection, and corrects opened
summaries in the final merge. Both use persistent exact workers and dispatch
over actual route counts, with uncapped selection and no selected-count CPU
readback. The entry-16 pilots did not establish an improvement.

The optimized per-region path is still 21.1% slower than dense at 32K and 14.9%
slower at 64K. These dense controls were rerun on the same GPU instance and
prompts as the LoD runs. Top-eight is 6.1% slower than dense at 32K and 12.2%
faster at 64K.

Closed-summary/native-leaf checks pass on PyTorch 2.5.1 and 2.13; full-model timings
retain the existing PyTorch 2.5.1 runtime. The report records separate real-state
checks for both selected paths. Full RULER has not been rerun, and serving
defaults are unchanged. Complete prefill/cache work has no new subquadratic
complexity claim.

</details>

## Qwen3-8B threshold prefill on H100

The [Qwen3 transfer report](../results/h100-baselines/qwen-opt/REPORT.md) covers
the shared saved-score selector, direct FP16 coarse projection, tensor-core
exact-output reduction and packed state maximum. Qwen keeps its native Q/K
norms and uses a fused version of its own cache coherence division. With
BF16 and batch one, complete threshold prefill takes **1.965 / 5.569 / 15.193 s**
at 32,768 / 65,536 / 128,000 tokens, versus **2.119 / 6.456 / 18.255 s** for
the previous threshold path. At 128,000 tokens it uses **13.2% less time**
than dense; at 32K and 64K it remains slower than dense. The initial exact
16,384-token field and uncapped c=16 rule are unchanged. The report keeps
these full-model times and quality checks separate from the read-only
[Qwen3 attention graphs](QWEN3_REAL_ATTENTION.md).

On the matched 32K panel, both multikey tasks and QA-1 retain their previous
threshold scores. CWE decreases from **78.44% to 76.88%**. Each task has 32
examples; this is a four-task check, not a full RULER average. The optimized
settings remain opt-in.

## Llama 3.1: 32K, batch-four amortized comparison

This full-model comparison uses Llama-3.1-8B-Instruct, BF16 model weights,
one H100 80GB HBM3, and four copies of the same 32,768-token ProLong prompt.
Each method runs in the RetroInfer runtime with whole-model CUDA graphs. One
repetition is warmup; the table gives the median of two timed repetitions.
After prefill and initial graph capture, 1,023 decode steps warm the state;
the next 2,048 steps are timed. The context grows over that window. Prefill
and initial graph capture are outside decode timing; prefill excludes initial
cache allocation. RetroInfer's periodic index rebuilds and graph recaptures,
and LoD's periodic updates, are included in the timed window. Each measured
RetroInfer repetition contained two rebuilds.

| Method | KV precision | Prefill | Decode per batch step | Amortized per output token | Output tokens/s |
|:---|:---|---:|---:|---:|---:|
| Dense | BF16 | 6.323 s | 13.058 ms | 3.265 ms | 306.3 |
| RetroInfer GPU-only | BF16 | 7.044 s | 10.848 ms | 2.712 ms | 368.7 |
| LoD three-tier, top-two pages | BF16 | 7.070 s | 9.407 ms | 2.352 ms | 425.2 |
| LoD three-tier, top-two pages | INT4 | 7.708 s | 9.297 ms | 2.324 ms | 430.2 |

Amortized per-output-token time is batch-step time divided by four; throughput
counts all four output tokens per step. Each request waits for the full batch
step, so the divided value is not the per-request step latency. Relative to our dense path, RetroInfer is
1.20× faster; LoD BF16 and INT4 are 1.39× and 1.40× faster. Relative to our
RetroInfer run, LoD BF16 and INT4 are 1.15× and 1.17× faster. All rows use
the same batch and prompt, but are not matched on quality. The LoD state
update includes the later synchronization fix.

Separate teacher-forced checks compared graph and eager logits on the same
cache at 26 sampled positions for each method. All sampled logits matched
exactly; the RetroInfer check included the steps around its index rebuild.
These checks do not substitute for a RULER quality evaluation. Raw
[RetroInfer](../results/h100-baselines/figure8-b4-32k-retro-timing.json),
[dense](../results/h100-baselines/figure8-b4-32k-retro-full-timing.json),
[LoD BF16](../results/h100-baselines/figure8-b4-32k-retro-lod-bf16-timing.json),
[LoD INT4](../results/h100-baselines/figure8-b4-32k-retro-lod-int4-timing.json),
[checks and run records](../results/h100-baselines/provenance/figure8-b4-32k/)
are retained for review.

**Comparison with Double-P Figure 8:** The paper's 32K, batch-four plot labels
dense as 50.4 ms and labels RetroInfer and Double-P as 1.98× and 2.13× faster
than dense. Those labels imply approximately 25.45 ms for RetroInfer and
23.66 ms for Double-P in the paper's plotted units. They are paper values,
not measurements on our H100. The paper describes end-to-end latency per
output token; its batch normalization and decode-window details are not
established by our reproduction. Its hardware is an H100 PCIe, whereas ours
reports H100 80GB HBM3; software and prompts differ as well. RetroInfer starts
with four sink and 64 recent tokens, as in the paper's stated setup, but its
exact recent buffer grows between rebuilds. LoD uses its own local/sink policy.
See [Double-P, Figure 8 and experimental setup](https://arxiv.org/pdf/2602.05191).

For the previously requested illustrative ratio estimate, applying the paper's
RetroInfer-to-Double-P ratio to our measured RetroInfer batch step gives
10.848 × 1.98 / 2.13 ≈ **10.08 ms per batch step**. This is an extrapolation,
not a Double-P run. In particular, our measured dense/RetroInfer ratio of
1.20× differs substantially from the paper's 1.98×, so the estimate should
not be treated as a validated prediction or a quality-matched result.

## Historical Llama full-model prefill and decode comparison

All rows use batch one, frozen prompts, the RetroInfer model runtime and
whole-model decode graphs. Times are medians of three 1,025-token generations
after two warmups. Prefill includes cache population; decode is reported
per token. Initial cache allocation and graph setup are excluded from both.
Squeezed uses a prebuilt prefix index; its index preparation and loading costs
are reported separately below. The QUEST row uses
the author's kernels with the shared-GQA addressing patch and a 2,048-token
budget. The FP16 dense row is its precision-matched reference. LoD and
RetroInfer use BF16 model weights, so comparisons with the FP16 rows also
include a precision difference.

**Limitation:** These RetroInfer decode timings do not include its periodic
index rebuild. LoD's periodic updates are included, so the rows do not give a
fair comparison of amortized maintenance costs.

| Method | Model / KV precision | 32K prefill | 32K decode | 64K prefill | 64K decode |
|---|---|---:|---:|---:|---:|
| Dense | BF16 / BF16 | 1.671 s | 8.730 ms/token | 5.314 s | 10.085 ms/token |
| RetroInfer GPU-only | BF16 / BF16 | 1.985 s | 9.766 ms/token | 5.862 s | 10.583 ms/token |
| LoD three-tier top-two, tuned | BF16 / BF16 | 1.835 s | 8.012 ms/token | 4.749 s | 8.443 ms/token |
| LoD three-tier top-two, tuned | BF16 / INT4 | 1.977 s | 7.980 ms/token | 5.103 s | 8.253 ms/token |
| Squeezed author kernels, adapted, prebuilt index | BF16 / BF16 | 1.624 s | 10.210 ms/token | 5.199 s | 11.607 ms/token |
| Multipole author kernels, adapted, online index | BF16 / BF16 | 2.299 s | 15.279 ms/token | 6.561 s | 19.035 ms/token |
| Multipole kernels, replacement disabled, online index | BF16 / BF16 | 2.312 s | 13.861 ms/token | 6.499 s | 17.547 ms/token |
| Dense | FP16 / FP16 | 1.719 s | 8.660 ms/token | 5.392 s | 10.025 ms/token |
| QUEST author kernels, shared KV, 2,048 tokens | FP16 / FP16 | 1.677 s | 8.463 ms/token | 5.336 s | 9.691 ms/token |

The QUEST implementation has two dense layers and uses modified kernel
addressing for shared GQA storage. Its correctness checks, the original
expanded-KV measurements, graph setup costs and quality limitations are
documented [below](#released-quest-kernels-in-the-retroinfer-runtime).
The Squeezed row selects keys independently per query head and uses modified
GQA addressing; its index costs and checks are [below](#squeezed-attention-in-the-retroinfer-runtime).

## Matched whole-step graph comparison before H100 route tuning

All three methods capture the model decode step, including projections,
attention, MLP and sampling. Dynamic position updates and LoD's periodic
cache maintenance run outside the captured step and are included in decode
time. RetroInfer's released GPU-only `capture_cuda_graph` is a no-op; our
[graph adapter](retro_native_graph.py) keeps its selection and attention
kernels while replacing its host-indexed cache append with a device-indexed
append. Its index rebuild would require recapturing the graph; the adapter
explicitly refuses to run across that boundary. The 1,025-output-token
benchmark has 1,024 decode steps and does not cross it.

At 32K and 64K, graph and eager logits were identical at seven sampled
positions on the same cache for dense attention, RetroInfer, and LoD BF16 and
INT4, including a LoD cache-update boundary. At 4K, teacher-forced LoD graph and eager outputs
matched across all 385 tested positions. In a 32K retrieval sanity test,
dense, LoD BF16, and LoD INT4 recovered all three passkeys. These checks do not
substitute for a full quality evaluation.

| Method | Context | Prefill | Graph setup | Decode | Request | State |
|---|---:|---:|---:|---:|---:|---|
| Dense | 32,768 | 1.671 s | 0.115 s | 8.730 ms/token | 10.726 s | complete |
| RetroInfer GPU-only | 32,768 | 1.985 s | 0.124 s | 9.766 ms/token | 12.099 s | complete |
| LoD BF16, three-tier top-two | 32,768 | 1.837 s | 0.145 s | 8.128 ms/token | 10.304 s | corrected rerun complete |
| LoD INT4 KV, three-tier top-two | 32,768 | 1.973 s | 0.146 s | 8.074 ms/token | 10.379 s | corrected rerun complete |
| Dense | 65,536 | 5.314 s | 0.121 s | 10.085 ms/token | 15.748 s | complete |
| RetroInfer GPU-only | 65,536 | 5.862 s | 0.122 s | 10.583 ms/token | 16.829 s | complete |
| LoD BF16, three-tier top-two | 65,536 | 4.657 s | 0.170 s | 8.459 ms/token | 13.495 s | corrected rerun complete |
| LoD INT4 KV, three-tier top-two | 65,536 | 5.044 s | 0.160 s | 8.327 ms/token | 13.731 s | corrected rerun complete |

Each timing is a median of three measured 1,025-token generations after two
warmups, using frozen ProLong prompts and batch one. Graph setup includes a
warmup step and capture after prefill with kernels already compiled; it is
listed separately so one can account for per-request graph setup. Request time
is the median per-run sum of prefill, setup, and decode; it excludes model
startup, initial cache allocation, and first-time compilation.
The three corrected 32K LoD BF16 decode measurements were
7.936, 8.128, and 8.147 ms/token. Normal generation uses the same prompts and
output count but may generate different tokens for different algorithms.

The first adapter sent an entire prompt through one prefill call. INT4
prefill logits became severely distorted beyond 16K, although BF16 passed
the three-passkey test. Explicitly splitting LoD prefill into the production
16K schedule fixed the tested INT4 prefill outputs at 17K and 32K. The LoD
timings above use the corrected schedule.

## H100 routing optimization

The RetroInfer adapter now uses a 32-centroid, one-warp routing tile on
SM90 for batch-one prompts from 32K through 64K, for both BF16 and INT4
three-tier caches. The former setting was 64 centroids and two warps. Page
selection stays at two pages per open region; no retrieval budget was changed.
The general production profile is retained outside this adapter and regime.
The benchmark records the actual route geometry in each new result. The
`LOD_RETRO_ROUTE_GROUP`, `LOD_RETRO_ROUTE_WARPS`,
`LOD_RETRO_ROUTE_REDUCE_WARPS`, and `LOD_RETRO_PAGE_BLOCK` environment
variables are experiment-only overrides, recorded by cluster-run.

The table reports median end-to-end decode time over the same 1,024 decode
steps, two warmups, three measured runs, frozen prompts, and whole-model graph
policy used above. It includes LoD's cache maintenance; graph setup and prefill
are excluded. Differences of a few tenths of a percent should not be read as
conclusive speedups.

| KV cache | Context | Previous | Tuned | Reduction |
|---|---:|---:|---:|---:|
| BF16 | 32,768 | 8.128 ms/token | 8.012 ms/token | 1.42% |
| INT4 | 32,768 | 8.074 ms/token | 7.980 ms/token | 1.16% |
| BF16 | 65,536 | 8.459 ms/token | 8.443 ms/token | 0.19% |
| INT4 | 65,536 | 8.327 ms/token | 8.253 ms/token | 0.89% |

The unmodified dense and RetroInfer whole-model graphs above provide the
matched-runtime references. The H100 GPU trace showed that the BF16 32K
centroid router and page-refinement kernels took roughly 23 and 16 microseconds
per layer respectively; matrix operations dominate much of the model step.
A 64-centroid one-warp router and 128-centroid alternatives were slower in
the short 32K sweep. A page tile of eight failed to compile; 32 had only a
small short-run benefit, so page geometry remains unchanged.

For each cache precision and context, tuned graph and eager logits agreed
exactly at 13 teacher-forced positions, spanning three cache-update boundaries.
Comparing the tuned graph with the **previous geometry** on the same cache,
all 52 sampled next-token argmaxes agreed, but logits were not identical:
the largest absolute differences were 3.8125 and 0.4531 for BF16 at 32K
and 64K, and 1.8359 and 0.8945 for INT4. Different tile reductions and
routes can propagate numerical changes through layers. Both tuned modes
recovered all three 32K passkeys. These are sanity checks, not a full
RULER or LongBench quality evaluation.

Raw records are in `results/h100-baselines/retro-lod-tuned-final-*.json`,
`retro-lod-tuned-check-*.json`, `retro-lod-route-reference-check-*.json`,
and `retro-lod-route-quality-*-32k.json`. The corresponding cluster-run
records include commands, source snapshots, and GPU allocation. The GPU
trace is `/home/ubuntu/comparison/results/retro-lod-32k-gpu.nsys-rep` on
the H100 host.

## QUEST-style attention in the RetroInfer runtime

The [QUEST adapter](retro_quest_adapter.py) runs our independent BF16
QUEST-style page-bound kernels inside the same RetroInfer Llama model used
above, with whole-model CUDA graph capture. The KV cache uses native GQA;
the first two layers use dense FlashAttention, while the remaining layers
select 16-token pages per query head and always include the newest page.
The 25% budget is fixed from the initial prompt length; the full-budget
control includes all prompt and generated-token pages. Page-bound construction
is included in prefill, and page updates and selection are included in decode.
Like the other RetroInfer measurements, initial cache allocation is excluded.

| Method | 32K decode | 64K decode |
|---|---:|---:|
| Dense | 8.730 ms/token | 10.085 ms/token |
| RetroInfer GPU-only | 9.766 ms/token | 10.583 ms/token |
| LoD three-tier BF16, tuned | 8.012 ms/token | 8.443 ms/token |
| QUEST-style BF16, 25% budget | 10.574 ms/token | 13.833 ms/token |
| QUEST-style BF16, full budget | 19.120 ms/token | 30.949 ms/token |

Each is the median of three 1,025-token generations after two warmups, with
the same checkpoint, frozen prompts, batch size, precision, runtime, and graph
policy. Sparse QUEST-style median prefill times were 1.669 s and 5.266 s at
32K and 64K. Generation is not teacher-forced for these timings, so the
methods can generate different tokens and take different execution paths.

The adapter is an independent implementation of the QUEST selection rule,
not the authors' released CUDA kernels. Its timings characterize this port,
not an optimized or authoritative implementation of QUEST. The paper's
released implementation uses FP16 kernels and its controller expects one
KV head per query head. Results using those kernels, including a shared-KV
adaptation, appear in the next section. Earlier eager author-code runs and
the independent vLLM QUEST-style measurements are in
[H100_BASELINES.md](H100_BASELINES.md).

At both lengths and both budgets, graph and eager logits matched exactly at
11 sampled teacher-forced positions spanning page boundaries. Independent GPU
tests checked sparse and full-budget outputs against a direct attention
reference over 18 decode steps crossing a page boundary. At 4K, full-budget
QUEST-style and dense selected the same next token at 11 sampled positions;
their largest absolute logit difference was 0.6875. The 25% path recovered
all three 32K passkeys, but no full RULER or LongBench quality evaluation has
been performed. Results and cluster-run records are in
`results/h100-baselines/retro-quest-*` and `results/h100-baselines/provenance/`.

Double-P's paper says its own attention uses a weighted FlashAttention variant
introduced by RetroInfer; it does not establish that its QUEST baseline ran
in the RetroInfer framework or identify which QUEST kernels it used. Its
baseline settings preserve four sink tokens and a 64-token sliding window at
every layer. This adapter instead retains the newest page and keeps the first
two layers dense, so it is not an exact reproduction of Double-P's evaluation.

## Released QUEST kernels in the RetroInfer runtime

The [author-kernel adapter](retro_quest_author_adapter.py) uses the released
QUEST CUDA kernels for cache append, page-bound scoring, RAFT top-k and paged
attention in the RetroInfer Llama model, with FP16 weights and KV. The first
two layers are dense, matching the released QUEST policy. The remaining layers
use 16-token pages, and the budget includes the current page. The original
author-kernel path repeats KV heads for GQA. Dense attention
uses native GQA. All model decode steps are captured in CUDA graphs.

The released kernels launch on the default stream, which cannot be captured
with PyTorch's CUDA graph API. A [reproducible patch](prepare_quest_author_graph.py)
changes those launches to the current stream, without changing kernel math.
The adapter maintains 16 graphs for partial-page lengths and archives the
completed page outside the graph, included in decode timing. It also masks
padding before selection. These adaptations and their overhead are part of
the measurement. Graph setup captures 16 variants and is reported separately.
An author-controller reference test crosses KV and metadata page boundaries.
For the 25% runs, graph and eager next-token choices matched on all 11 sampled
positions at both lengths, with maximum absolute logit differences of 0.0625
and 0.0347. For the 2,048-token runs, the same 11 choices matched at each
length, but maximum logit differences were 0.2666 and 0.5352. The source of
these FP16 graph/eager differences has not been isolated, so these checks do
not establish numerical equivalence across all inputs.

| Method (FP16) | 32K decode | 64K decode |
|---|---:|---:|
| Dense | 8.660 ms/token | 10.025 ms/token |
| Released QUEST kernels, 25% | 9.841 ms/token | 12.479 ms/token |
| Released QUEST kernels, 2,048 tokens | 8.753 ms/token | 10.030 ms/token |

The median covers three 1,025-token generations after two warmups, using the
same frozen prompts, batch size and runtime. The 25% QUEST graph setup median
was 1.242 s and 1.265 s at 32K and 64K; the corresponding dense setup was
0.101 s and 0.114 s. Prefill and graph setup are excluded from decode time.
The 25% budget means 8,192 and 16,384 tokens respectively, fixed at the
initial context length; the 2,048 budget retains 127 historical pages and
the current page in sparse layers. The 25% setting follows
the [Double-P evaluation](https://arxiv.org/html/2602.05191) setting; the
[QUEST paper](https://arxiv.org/html/2406.10774) also reports experiments with
a fixed 2,048-token budget. The Double-P policy additionally preserves four
sink tokens and a 64-token window, so this is not its exact reproduction.
At both lengths, the 2,048-token author-kernel path and dense each recovered
three passkeys; at 32K the 25% path also recovered three. These are sanity
checks, not full quality evaluations. In particular, the QUEST paper's
2,048-token results on other models and tasks do not establish its accuracy
for Llama-3.1-8B on RULER.

Raw timing, graph-check and retrieval records are in
`results/h100-baselines/retro-author-*` and `retro-dense-fp16-*`; the build
patch and cluster-run metadata are under `results/h100-baselines/provenance/`.

### QUEST with shared GQA storage

A separate [addressing patch](prepare_quest_shared_kv.py) allows the author
scoring and sparse-attention kernels to refer to one physical KV head from
each of its four query heads. Each query head still selects its own pages;
the KV cache and page bounds contain eight heads, rather than 32 copies.
This is a modified version of the author kernels, not the unchanged release.
The adapter selects it with `--quest-shared-kv` and checks that the extension
matches the shared-storage build. The [build script](build_quest_author_graph.sh)
accepts `QUEST_SOURCE` and `QUEST_SHARED_KV=1` for this separate checkout.

The shared-storage CUDA-graph path matched the expanded controller on a
synthetic reference across 34 decode steps, including KV and metadata page
boundaries. This test verifies that both storage pools have eight heads.
On 32K and 64K teacher-forced prompts, shared and expanded paths chose the
same next token at all 65 checked decode positions. For shared storage, graph
and eager next-token choices matched on 11 sampled positions at each length;
the largest logit differences were 0.2598 and 0.5508, respectively. Their
cause remains unresolved, as for the expanded path.

| 2,048-token budget, FP16 | 32K decode | 64K decode |
|---|---:|---:|
| QUEST, expanded KV | 8.753 ms/token | 10.030 ms/token |
| QUEST, shared KV | 8.463 ms/token | 9.691 ms/token |

These are medians of three runs after two warmups using the same setup as the
preceding table. Shared storage reduces decode time by 3.3% and 3.4%,
respectively. The shared version retains the author's separate per-query-head
selection and does not necessarily combine memory loads between query heads.
The KV and bounds pools use one quarter the expanded storage; total model
memory does not fall by that factor. Shared storage recovered all three
passkeys at both lengths, a sanity check rather than a full quality evaluation.
Raw records are in `results/h100-baselines/retro-author-shared-*`, and the
source diff is `results/h100-baselines/provenance/quest-shared-kv.patch`.

## Squeezed Attention in the RetroInfer runtime

The [adapter](retro_squeezed_adapter.py) runs the authors' released Triton
centroid-lookup, compaction, sparse-attention and reduction kernels within the
RetroInfer model. A [kernel-addressing patch](prepare_squeezed_gqa.py) maps
centroid, count, and exact K/V reads to the shared physical KV head while
retaining per-query-head scores and selections. New fixed-prefix indexes
store centroids and labels once per KV head. The adapter also loads older
per-query-head indexes. Cluster counts are computed once when loading an
index.
Generated tokens use exact FlashAttention and are merged with the selected
prefix through the authors' reduction. The graph allocates a fixed upper bound
on sparse blocks to handle variable selection counts. The adapter is a port,
not the authors' optimized generation benchmark.

The comparison uses BF16, 5% centroids, and the 0.9 calibrated threshold
percentile. This threshold is not a fixed 90% token sparsity. At 32K/64K the
median decode times were 10.210/11.607 ms/token, compared with
8.730/10.085 ms/token for BF16 dense. The initially ported 128-row kernel
tile measured 13.359/17.023 ms/token; reducing the decode query tile to 16
rows improved it. Each measurement uses the same frozen prompt, 1,024
decode steps, two warmups and three measured runs as the other rows.
In a separate teacher-forced diagnostic, the final one of 16 decode steps
selected a mean of 9.86%/9.03% of fixed-prefix tokens across layers and
query heads at 32K/64K. This single-step observation is not a run-wide average.

These numbers assume a reusable fixed prefix and prebuilt index. The original
offline clustering and threshold calibration for these indexes took 356 s and
1,101 s at 32K and 64K, excluding the fixed-prefix prefill. On the measured
host, one index deserialization took 0.509/0.985 s; median cache initialization
including index transfer and count preparation took 0.089/0.169 s. Median graph
setup took 0.130/0.135 s. These costs are excluded from the table's prefill and
decode times. Real cold-start and prefix-reuse costs depend on the workload.

The tuned kernels matched the authors' expanded-KV wrapper and direct
attention in a synthetic 20-step test. On each frozen prompt, an eight-step
teacher-forced comparison with the authors' full evaluation model produced
the same next-token choices. Tuned and untuned graph implementations matched
all 65 decode choices at each length. Graph/eager checks agreed on 11 sampled
next-token choices per length but had maximum absolute logit differences of
0.6641/0.2813; the source of those differences has not been isolated. These
checks do not replace RULER or LongBench quality evaluation.

Raw data and run records are `results/h100-baselines/retro-squeezed-*`,
`squeezed-author-*-replay.json`, and the corresponding cluster-run metadata;
the author source change is `provenance/squeezed-gqa-addressing.patch`. The
GPU trace of the initial 32K port is at
`/home/ubuntu/comparison/results/retro-squeezed-32k.nsys-rep` on the H100 host.

## Historical Llama full-model batched 32K comparison

These are synchronous batches of independent requests using
copies of the same frozen 32K prompt, sequential per-request prefill and
whole-model CUDA graphs. Decode covers 256 steps, with one warmup and two
measured repetitions. LoD and Multipole maintenance is included, but the
RetroInfer index rebuild is not reached. The adapter holds
separate KV and cluster state for each request. This is a fixed-batch pilot,
not the earlier vLLM cohort experiment. LoD uses Triton 3.7.1; the Multipole
online clustering requires the RetroInfer environment's Triton 3.1.0. Report
batch-step latency and total generated tokens per second; prefill excludes
initial cache allocation and graph setup. Batched QUEST timing is not yet
available.

| Method | Batch | Prefill | Decode batch step | Decode throughput |
|---|---:|---:|---:|---:|
| Dense BF16 | 4 | 6.934 s | 12.781 ms | 313.0 tokens/s |
| RetroInfer GPU-only BF16 | 4 | 8.182 s | 10.096 ms | 396.2 tokens/s |
| LoD three-tier BF16 | 4 | 7.637 s | 9.330 ms | 428.7 tokens/s |
| LoD three-tier INT4 KV | 4 | 8.151 s | 9.231 ms | 433.3 tokens/s |
| Multipole BF16 | 4 | 9.695 s | 18.113 ms | 220.8 tokens/s |
| Multipole BF16, split lookup | 4 | 9.678 s | 17.616 ms | 227.1 tokens/s |
| Multipole, replacement disabled | 4 | 9.742 s | 16.487 ms | 242.6 tokens/s |
| Dense BF16 | 8 | 14.144 s | 18.383 ms | 435.2 tokens/s |
| RetroInfer GPU-only BF16 | 8 | 16.570 s | 11.109 ms | 720.1 tokens/s |
| LoD three-tier BF16 | 8 | 15.574 s | 11.211 ms | 713.6 tokens/s |
| LoD three-tier INT4 KV | 8 | 16.582 s | 11.189 ms | 715.0 tokens/s |
| Multipole BF16 | 8 | 19.666 s | 22.392 ms | 357.3 tokens/s |
| Multipole BF16, split lookup | 8 | 19.729 s | 22.476 ms | 355.9 tokens/s |
| Multipole, replacement disabled | 8 | 19.675 s | 19.522 ms | 409.8 tokens/s |

All sampled batch-four graph/eager checks agreed on the next token for every
request. Multipole and its replacement-disabled variant had maximum absolute
logit differences of 1.623 and 1.500, respectively, in 11 sampled checks;
dense and both LoD formats were exact in these checks. The RetroInfer
batch-four check was exact at seven sampled steps. For batch eight, all 11
sampled next-token choices agreed per request; dense, RetroInfer and LoD were
exact in sampled logits, while Multipole and the replacement-disabled variant
had maximum absolute logit differences of 1.492 and 1.580, respectively.
Multipole performs two online cluster updates during each 256-step decode; RetroInfer does not reach
its periodic index rebuild in this short run. The 512-token Multipole budget
and LoD configuration have not been matched on task quality. These repeated-
prompt measurements do not establish a production serving throughput. During
the batch-eight Multipole run, GPU memory use reached at least 77,617 MiB;
this is a sampled observation, not a measured peak.

## Provenance and controls

RetroInfer release revision `03f912c6e917c380d9d90c5ec85bb0f161ba53ef` has a
local CUDA build compatibility change in `library/retroinfer/setup.py`. The
experiment uses GPU-only caches, BF16 model weights, and one H100. LoD's BF16
and INT4 KV modes use the same top-two three-tier profile as the H100 vLLM
integration. The adapter uses RetroInfer's FlashAttention 2 for LoD prefill;
the vLLM version uses FlashAttention 3. The author framework has PyTorch 2.5 /
CUDA 12.4; dense, RetroInfer and LoD use the same `uv` Triton 3.7.1 overlay.
Multipole uses Triton 3.1.0 for its compiled clustering, as described in the
[historical adapter details](archive/RETRO_LOD_MULTIPOLE_HISTORY.md). The author's Transformers version stays unchanged;
the adapter loads LoD's projection-free engine without its Hugging Face facade.

For reference, an ungraphed diagnostic at 32K measured 10.301 ms/token dense,
13.139 ms/token RetroInfer, and 20.920 ms/token LoD. It is a separate execution
policy and should not be mixed with the matched graphed results.

The [comparison harness](h100_comparison.py), [LoD adapter](retro_lod_adapter.py),
[graph adapter](retro_native_graph.py), and [retrieval check](retro_lod_quality.py)
are submitted through [cluster-run](h100_cluster_run.py). The QUEST author
graph experiment uses the separate patched source copy described above.

## Historical Llama full-model matched-batch snapshot

The new 128K comparison uses **batch two** for every method and an identical
128,000-token prompt. The prior 32,768-token comparison above uses **batch
eight** for every method. Both use the actual complete Llama-3.1-8B model;
the synthetic single-layer tables above are diagnostic and must not be mixed
with them. The H100 ran out of memory while bringing the full dense batch-four,
128,000-token BF16 cache onto the GPU. QUEST and Squeezed use adapters that
run independent author-kernel calls for each request inside the captured model
step. Graph-versus-eager checks passed, but independent batch-reference
validation and full timing runs remain pending.

The 128K model runs use an 8,192-token MLP chunk during prefill to reduce peak
memory. Graph capture and initial cache allocation are outside the timing;
prefill and 256 decode steps are measured separately. The RetroInfer graph
adapter does not rebuild its index within that 256-step window. Squeezed uses
a prebuilt prefix index; its construction and loading are separate costs.
Different methods use their previously documented budgets and have not been
matched for quality. The remaining short timing runs are stopped; these are
preliminary results, not a comparison of amortized maintenance costs.

| Method | 128K batch-two prefill | Decode batch step | Decode throughput |
|---|---:|---:|---:|
| Dense BF16 | 34.762 s | 18.009 ms | 111.1 tokens/s |
| RetroInfer GPU-only BF16, rebuild excluded | 36.948 s | 12.494 ms | 160.1 tokens/s |
| LoD three-tier BF16 | 22.485 s | 9.586 ms | 208.6 tokens/s |
| LoD three-tier INT4 KV | 24.144 s | 9.350 ms | 213.9 tokens/s |
| Multipole BF16, 512-token budget | 40.881 s | 26.881 ms | 74.4 tokens/s |

The Multipole result includes two measured repetitions and two updates per
decode. The replacement-disabled Multipole variant, QUEST, and Squeezed do
not yet have completed 128K batch-two timing results. Their remaining short
runs were stopped before the longer-window comparison.

## Historical Llama full-model amortized RetroInfer comparison

These historical full-model runs compare dense, RetroInfer GPU-only, and LoD three-tier
BF16 and INT4, using batch eight at 32,768 prompt tokens and batch two at
128,000. Each run generates 3,072 tokens: after the prefill-generated token,
1,023 decode steps warm the state and the following 2,048 steps are timed.
RetroInfer's author index rebuild and the required graph recapture are both
inside this timed window, which must contain exactly two rebuilds. LoD's
updates are included in the same timed window. Initial graph capture and
allocation remain outside decode timing; prefill and decode warmup are recorded
separately. Each method has one warmup repetition and two timed repetitions.

The RetroInfer runs require graph-versus-eager checks at and around the rebuild
boundary at the matching prompt length and batch size. Both the 32K batch-eight
and 128K batch-two checks passed with exact sampled logits. The results below
are full-model measurements using identical prompts copied across the batch.
Methods use their previously documented settings and have not been matched on
task quality.

| Method | 32K batch-eight prefill | 32K decode batch step | 32K output tokens/s | 128K batch-two prefill | 128K decode batch step | 128K output tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Dense BF16 | 14.393 s | 19.071 ms | 419.5 | 34.785 s | 18.162 ms | 110.1 |
| RetroInfer GPU-only BF16 | 16.666 s | 12.114 ms | 660.4 | 36.746 s | 13.459 ms | 148.6 |
| LoD three-tier BF16 | 15.620 s | 11.463 ms | 697.9 | 22.555 s | 9.660 ms | 207.0 |
| LoD three-tier INT4 KV | 16.624 s | 11.421 ms | 700.4 | 24.167 s | 9.471 ms | 211.2 |

Each RetroInfer measured repetition recorded two index updates in the timed
window. The earlier 256-step snapshot did not include a RetroInfer rebuild;
its different window and shorter context growth also affect decode times, so
the difference is not an isolated measurement of rebuild cost. Raw records
are `results/h100-baselines/amortized-*.json` and the graph/eager checks are
`results/h100-baselines/retro-rebuild-*-check.json`.
