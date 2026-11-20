# Multipole and LoD: H100 synthetic attention comparison

## Graph-captured comparison of Multipole's paper components

These are synthetic, single-layer Qwen3-8B-shaped H100 measurements, not
full-model inference or a quality-matched comparison. Both tables use BF16
Q/K/V and a nominal 131,072-token prefix. Neither includes new KV writes or
cache maintenance beyond the isolated update described below.
The [Squeezed variant results](#multipole-derived-squeezed-variant) use the
same synthetic boundaries and are shown separately below.
For batch-one graphs using real Qwen3-8B layer states, see the separate
[real-state comparison](QWEN3_REAL_ATTENTION.md).
The [connected Qwen3 decode and quality report](../results/h100-baselines/qwen-long/REPORT.md)
uses one optimized LoD prefill per prompt and includes ongoing cache updates
in complete-model decode timings for LoD, dense, and Squeezed variants.

### Remote attention only

Both methods exclude local and sink attention in this table. LoD's backing
prefix includes a 256-token local field and sink that are excluded from its
graph; the author's fixture has 131,072 remote tokens.

The primary table captures the same author component closures together, the
author's dense FlashDecoding graph, and LoD's remote attention graph. A
separately measured, amortized isolated update cost is added
outside each sparse graph; dense needs no update addend. The author fixtures are
independent: replacement output is not an input to the sparse-attention
fixture, so its graph is a graph of paper components, not a connected pipeline.
The dense values are medians of two 512-replay measurements; graph and eager
outputs matched, and an independent dense reference was checked for one KV
head in every request. The author sparse function updates its merge buffers
in place, so the graph resets those buffers on every replay. Earlier numbers
did not reset them and accumulated output across replays; the table below
uses corrected runs with repeated-output checks.

| Batch | Author dense BF16 | Multipole, 512 budget | Multipole, 5% budget | Multipole, 10% budget | LoD two-tier BF16 | LoD three-tier BF16 | LoD three-tier INT4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.256 ms | 0.235 ms | 0.279 ms | 0.287 ms | 0.072 ms | 0.060 ms | 0.061 ms |
| 4 | 0.832 ms | 0.335 ms | 0.388 ms | 0.448 ms | 0.140 ms | 0.127 ms | 0.126 ms |
| 16 | 3.014 ms | 0.919 ms | 1.100 ms | 1.286 ms | 0.433 ms | 0.396 ms | 0.393 ms |

### Including local and sink attention

The following graph comparison includes 256 exact local tokens and one exact
sink token in each sparse method, with 130,815 remote entries for a total of
131,072 tokens. One sink matches LoD's configuration; the released Multipole
generation path uses ten. The author local and remote sparse attention outputs
are merged into one softmax using the released sparse kernel. Lookup and
replacement remain independent paper-benchmark fixtures, as above. The LoD
graph includes its local and sink fields in its usual reduction. Neither
graph attends to a separately appended new KV item or includes per-token KV
writes; the same separately amortized isolated updates are added as above.
The author graph resets its in-place merge buffers on each replay, and LoD
uses its fixed state-update kernel.

| Batch | Author dense BF16 | Multipole, 512 budget | Multipole, 5% budget | Multipole, 10% budget | LoD two-tier BF16 | LoD three-tier BF16 | LoD three-tier INT4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.256 ms | 0.282 ms | 0.327 ms | 0.336 ms | 0.074 ms | 0.061 ms | 0.061 ms |
| 4 | 0.832 ms | 0.394 ms | 0.445 ms | 0.503 ms | 0.143 ms | 0.131 ms | 0.128 ms |
| 16 | 3.014 ms | 1.210 ms | 1.397 ms | 1.580 ms | 0.440 ms | 0.404 ms | 0.401 ms |

The 5% and 10% budgets correspond to 6,540 and 13,081 remote entries; 512 is
also a remote-entry budget. Multipole local attention was checked against an
independent dense reference, combined graph and eager outputs matched, and
replays were stable. Perturbing either local or sink values changed each
method's output. These checks are not an end-to-end quality evaluation.

**Shape sensitivity:** The increase from the preceding remote-only table is
not solely local/sink cost, so do not subtract the two tables to estimate
that cost. At batch 1 and 5%, the 131,072-remote graph took 0.252 ms before
the update addend. On the 130,815-remote fixture, the remote-only graph took
0.287 ms and the local/sink graph took 0.300 ms: the matched-fixture increase
was 0.013 ms, rather than the 0.048 ms difference between table entries.
The change in remote shape accounts for much of the table difference; these
measurements do not establish why it changes the runtime. At a percentage
budget it also changes the number of selected remote entries. The added Multipole
path uses a separate invocation of the author's sparse-attention kernel and
merges its result with remote attention.

For three-tier LoD at batch 1, the BF16 graph timings before the same update addend were
0.05584 ms without local/sink and 0.05605 ms with them, from separate runs;
the 0.00022 ms difference is too small to interpret as a precise overhead.
The local and sink perturbation checks confirmed that both affected the output.
LoD incorporates these fields in its attention path rather than invoking a
separate sparse-attention operation for them. Different kernel organization
can therefore produce very different incremental costs; these measurements
do not establish that local/sink attention is free.

At batch 16 and 5%, on the same 130,815-remote
fixture, Multipole's remote graph took 1.108 ms and adding the local/sink
field took it to 1.126 ms, before the update addend. The preceding table's
131,072-remote graph took about 0.829 ms. A separate diagnostic with 130,816
remote entries took 0.841 ms without local attention; its total context with
the local/sink field was 131,073, so it is not used in the matched table.
This establishes sensitivity to the synthetic input shape, not the precise
cause. See [the alignment diagnostic](../results/h100-baselines/multipole-author-local-sink-b16-128k-alignment-check.json)
and [run records](../results/h100-baselines/provenance/multipole-local-sink-graph/).

### Multipole-derived Squeezed variant

This variant uses the Multipole benchmark's centroid lookup and sparse
attention with centroid replacement omitted: unselected entries have no
summary contribution. Its graph is captured and measured directly, with
empty sparse merge buffers reset on every replay. As in the Multipole table,
lookup and sparse attention use independent author fixtures, so this is not
a connected selection-to-attention pipeline. Lookup includes the benchmark's
per-token gather and unsorted top-k selection; the generation path's separate
compaction, KV writes and full cache maintenance are outside this boundary.

The 1:8 columns use one centroid per eight remote tokens, matching the
Squeezed configuration in [the Multipole paper's runtime comparison, Appendix D](https://proceedings.neurips.cc/paper_files/paper/2025/file/359f442022ef31bcd9e41959576107ef-Paper-Conference.pdf).
The 1:16 columns use the Multipole table's centroid density, as an ablation.
Both variants add an independently measured, amortized online cluster update,
as in the Multipole table. The paper's Squeezed baseline also uses Multipole's
online clustering improvements; the original released Squeezed evaluation
uses a fixed offline index and different selection. These figures are not a
measurement of that original end-to-end implementation, nor are they a
reproduction of the paper's A6000 numbers.

| Attention fields | Batch | 1:8, 512 budget | 1:8, 5% budget | 1:8, 10% budget | 1:16, 512 budget | 1:16, 5% budget | 1:16, 10% budget |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Remote only | 1 | 0.162 ms | 0.206 ms | 0.216 ms | 0.149 ms | 0.198 ms | 0.207 ms |
| Remote only | 4 | 0.320 ms | 0.372 ms | 0.431 ms | 0.256 ms | 0.308 ms | 0.368 ms |
| Remote only | 16 | 1.019 ms | 1.198 ms | 1.387 ms | 0.749 ms | 0.931 ms | 1.117 ms |
| Local and sink | 1 | 0.180 ms | 0.225 ms | 0.235 ms | 0.169 ms | 0.217 ms | 0.227 ms |
| Local and sink | 4 | 0.349 ms | 0.401 ms | 0.460 ms | 0.284 ms | 0.335 ms | 0.397 ms |
| Local and sink | 16 | 1.090 ms | 1.267 ms | 1.452 ms | 0.820 ms | 1.003 ms | 1.185 ms |

Local/sink rows use 130,815 remote entries plus 256 local and one sink; the
remote-only rows use 131,072 remote entries, as in the preceding tables.
The 1:8 update fixture has 8K FP32 keys, 1,024 existing centroids and 16 new
centroids for 128 new tokens. Its measured updates were 4.153, 12.646 and
48.874 ms for batches 1, 4 and 16, respectively, each divided by 128. The
1:16 rows use the same 512-centroid update measurements as Multipole, given
below. These are isolated update components rather than full generation
maintenance; this H100 1:8 update fixture is an adaptation of the released
1:16 benchmark.

All 36 measured cases matched eager execution and had stable replay outputs. The
local/sink runs additionally checked local attention against an independent
reference and verified that changing either local or sink values changes the
combined result. These checks do not establish model quality. See the
[raw results](../results/h100-baselines/) and [run records](../results/h100-baselines/provenance/multipole-squeezed/).

The [historical Multipole experiments](archive/RETRO_LOD_MULTIPOLE_HISTORY.md)
contain the earlier eager, connected-generation and mixed-boundary tables.
They are retained for provenance and are not the current comparison.

The update addend mirrors the released `benchmark_kmeans.py` convention: an
isolated 8K fixture, `do_bench` with the same 500/500 settings, divided by
the method's update interval. Multipole's FP32 fixture updates 128 tokens,
512 centroids and eight new centroids, and its median update costs are
3.407, 9.547 and 34.608 ms at batches 1, 4 and 16, divided by 128. LoD's
BF16 fixture updates 256 tokens and starts with 1,448 regions; its isolated
region-summary and owner-assignment costs are 1.165/1.221 ms (batch one,
BF16/INT4), 1.251/1.208 ms (batch four), and 1.234/1.214 ms (batch 16),
divided by 256, for three-tier. The corresponding independently measured
two-tier BF16 costs are 1.192, 1.243 and 1.205 ms, also divided by 256.
The LoD timing clones its input state inside the timed call to
provide a fresh state, and updates both key and value sums; the author test
creates outputs and times key clustering. These are analogous isolated update
components, not identical operations. Neither addend includes the full cache
maintenance required for generation, and INT4 page storage is not touched by
this isolated LoD state update.

In the remote-only table, Multipole's 5% and 10% budgets represent about
6,554 and 13,107 exact tokens; its smallest column is 512. LoD two-tier opens
up to eight whole regions; three-tier opens up to eight regions and two
16-token pages per region (top-two page opening). Two-tier BF16 is the supported two-tier mode in
this implementation. Equal timing methodology
does not imply equal token budgets or equal answer quality. The LoD code uses Triton 3.7.1 while the author
benchmark uses 3.1.0, because the author benchmark did not run with the newer
version in this environment. The very different `do_bench` and graph numbers
must not be mixed: the former clears cache between iterations, and their
launch and measurement behavior differs.

The LoD state-merge kernel used in the two tables above includes a
synchronization fix. An independent scatter-add regression passed 10 repetitions at batches 1, 4 and
16; isolated state updates also conserved counts and matched independently
reconstructed value sums. These checks do not establish model quality. The
older LoD rows in the historical reports used the previous kernel and remain
flagged there. The [run logs](../results/h100-baselines/provenance/multipole-paper-style-fixed/)
include the initial failed reference and passing fixed runs; the raw records
are `multipole-lod-paper-{attention,update}-*-fixed*.json` in
`results/h100-baselines/`; the graph dense results are
`multipole-paper-dense-graph-b*-128k.json`. The [LoD harness](multipole_lod_bench.py),
[author graph harness](multipole_paper_graph.py), and
[author isolated-update harness](multipole_paper_clustering.py) define the
timing boundaries.

The two-tier graphs use the same replay and inclusion checks as the three-tier
graphs; each measured graph matched eager execution. Perturbing local and sink
values changed the local/sink output, while perturbing excluded values left the
remote-only output unchanged. The isolated two-tier updates passed repeatability,
count conservation and value-sum reference checks. Raw measurements are
`multipole-lod-two-tier-{remote,local,update}-b*.json` in
`results/h100-baselines/`; [run records](../results/h100-baselines/provenance/multipole-two-tier/)
contain the commands and source hashes. These are synthetic timing and kernel
checks, not an evaluation of answer quality.

### Why the connected LoD timing was shorter

The old connected and separately summed tables measured different boundaries,
and both used the earlier LoD kernel. A fixed-kernel, synthetic BF16 rerun at
batch 16 and 128K measured **0.449 ms per step** over two 512-step connected
decode windows, including local/sink attention, KV writes and periodic full
cache updates. In a separate experiment, the remote-only graph took 0.391 ms;
adding a separately synchronized full catch-up cost of 0.097 ms per step gives
0.489 ms. That sum is an illustration of the timing difference, not a matched
comparison: the graph excludes local/sink attention and KV writes, while the
connected window grows its context and updates its state.

A diagnostic that inserted GPU events without synchronizing at each update
still measured 0.449 ms per connected step. Its full-update event intervals
were 8.6–9.0 ms, compared with about 24.9 ms for the separately synchronized
full catch-up. A different diagnostic that synchronized each update measured
0.513 ms per step, but also added instrumentation overhead. These observations
show that the timing boundary and queueing matter; the experiments do not
isolate their contribution from differences in state, context and cache
behavior. Event intervals measure elapsed time on the GPU timeline, not pure
active compute. This full catch-up diagnostic is distinct from the smaller
isolated state-update addend used in the paper-component tables above.

See the [connected run](../results/h100-baselines/lod-connected-vs-isolated-three-tier-bf16-b16-128k-fixed.json),
[queued-event diagnostic](../results/h100-baselines/lod-connected-update-async-profile-b16-128k-fixed.json),
[synchronized diagnostic](../results/h100-baselines/lod-connected-update-profile-b16-128k-fixed.json),
[isolated catch-up samples](../results/h100-baselines/multipole-lod-paper-attention-three-tier-bf16-b16-128k-fixed.json),
and [run records](../results/h100-baselines/provenance/lod-connected-update-audit/).
