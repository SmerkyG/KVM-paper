# Historical Multipole experiments

These are superseded research records, retained for provenance. The current
graph-captured comparisons are in [MULTIPOLE.md](../MULTIPOLE.md).

**Validity and scope:** Most historical LoD H100 results below predate a fix
for dropped value-summary updates; the paper-style eager comparison immediately
below is a later fixed-kernel measurement. The old remote-only Multipole graph
did not reset in-place merge buffers on every replay. Some tables also mix
isolated clustering with complete LoD catch-up or compare connected generation
with independently timed paper components. Their figures are not current
validated comparisons and should not be used as evidence of end-to-end speed or quality.

In particular, the connected LoD timings and the older remote-only graph plus
complete catch-up timings came from separate runs with different timing
boundaries and inputs. For example, at batch 16 the connected LoD BF16 result
was 0.450 ms per step, while the separate remote graph plus isolated full
catch-up estimate was 0.491 ms. The latter combines a fixed-cache graph with
separately timed updates; the former times a growing decode window with updates
inside it. Neither experiment establishes why one total was smaller. Both
predate the H100 correctness fix.

## Historical paper-style eager component comparison

For reference, the table below follows the authors' released eager benchmark
method: `triton.testing.do_bench` times each Multipole component separately,
then we sum its lookup, replacement and sparse-attention medians; LoD's
remote attention is timed as one component. Both use 500 ms warmup, 500 ms
repeat, the median, and per-iteration cache clearing in their respective
Triton versions. Each sparse-method cell adds the same isolated update. The
512-token author component result is one run; dense, 5% and 10% columns use
the median of two runs. LoD component results use one run per cell.

| Batch | Author dense BF16 | Multipole, 512 budget | Multipole, 5% budget | Multipole, 10% budget | LoD BF16 | LoD INT4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.261 ms | 0.342 ms | 0.359 ms | 0.392 ms | 0.308 ms | 0.308 ms |
| 4 | 0.835 ms | 0.402 ms | 0.431 ms | 0.507 ms | 0.305 ms | 0.306 ms |
| 16 | 3.016 ms | 0.970 ms | 1.139 ms | 1.326 ms | 0.401 ms | 0.398 ms |

**Batch-one timing interpretation:** The apparent LoD slowdown relative to
dense in the eager table is sensitive to launch timing. In a separate run on
the same LoD cache, attention alone measured 0.302 ms with the eager event
boundary and L2 clearing, versus 0.066 ms for the captured graph with L2
clearing and 0.057 ms for the captured graph without it. Graph and eager
outputs matched exactly. A diagnostic CUDA trace recorded six LoD kernels
whose durations summed to approximately 0.050 ms; profiling changes timing,
so that sum is diagnostic, not a replacement benchmark. The large remaining
difference is consistent with host launch and scheduling gaps in the eager
measurement, not just a cold-cache penalty. The author dense fixture launches
two kernels. See the [diagnostic record](../../results/h100-baselines/multipole-lod-b1-paper-vs-graph-diagnostic.json).


## Multipole Attention in the RetroInfer runtime

The [Multipole adapter](../retro_multipole_adapter.py) ports the authors' released
single-level grouped-KV Triton lookup, sparse attention, replacement, and
compaction kernels into the same Llama runtime. It uses their online blockwise
clustering, 10 sink tokens, a 128-token update interval, one centroid per 16
clustered tokens, windowed RoPE (the query at position 2,048 and unrotated
keys), and a 512-token cluster budget. The same adapter can disable centroid
replacement to test the Multipole paper's Squeezed variant. This differs from
the earlier prebuilt-index Squeezed row, which uses per-query-head selection,
a calibrated threshold and a fixed prefix.

Prefill includes prefix clustering. Decode includes periodic index updates;
its whole-model graph uses preallocated buffers, with the updates executed
outside the graph and charged to decode. Cluster counts are recomputed from
final labels because the released k-means functions return counts from an
earlier assignment. The selection masks empty and padded clusters; when the
budget covers every token, all populated clusters are selected, addressing the
released strict-threshold fallback that can otherwise omit the last cluster.
[Preparation](../prepare_multipole_source.py) also corrects an undefined variable in the author's final-window branch; under its current
window partitioning that branch appears unreachable. The released repository
revision is `db65e51d582b9bcb89807b9d9397c3532d1099ef`. Initial cache
allocation and graph capture remain excluded, as in the other table rows.

The 32K Multipole median of 15.279 ms/token includes eight online updates in
1,024 decode steps; its 2.299 s prefill includes index construction. The initial
full-capacity sparse launch measured 15.301 ms/token; bounding the launch by
the fixed token budget produced only a small change. This is
preliminary performance evidence for the port, not a quality-matched ranking.
A direct grouped-KV reference matched the graph for both replacement settings
and a full-budget case over eight synthetic decode steps. In teacher-forced
32K/64K checks, all 11 sampled graph/eager next-token choices agreed, including
after updates; maximum absolute logit differences were 0.328/0.754 for
Multipole and 0.875/1.391 for the replacement-disabled variant. The released
atomic compaction may reorder selected indices. On a fixed synthetic 64K
input the selected sets stayed identical across 20 graph replays, while the
index order changed and attention output varied by at most 0.0000611. A
separate 64K model diagnostic also found eager/eager logit differences
(maximum 0.406 across sampled positions), so bitwise reproducibility is not
established; the individual sources of model-level differences are not fully
isolated. These comparisons check the port, not full task quality.

Multipole and dense both recovered the key in three 32K retrieval smoke
prompts (depths 0.1, 0.5 and 0.9). This small diagnostic does not substitute
for RULER or LongBench.

A separate 32K diagnostic measured two warmed index updates at 111.3 and
110.8 ms, or approximately 0.87 ms per token amortized over the 128-step
update interval. In a CUDA graph node trace of 256 decode steps, centroid
lookup averaged 62.9 microseconds per layer, centroid replacement 30.6,
centroid-score sort 22.0 and sparse attention 11.9. Across 32 layers these
four operations total approximately 4.1 ms per batch-one decode step, excluding
other work. The released generation lookup currently used here does not split
centroids across blocks: it launches one block for each of Llama's eight KV
heads. The paper's kernel microbenchmark uses a separate split-centroid kernel,
so its attention-only speedups should not be assumed to describe this port.
These are diagnostic trace measurements rather than additive components of the median wall-clock result; the trace includes two updates and
may perturb timings. [Kernel summary](../../results/h100-baselines/retro-multipole-32k-gpu-kernels.csv);
the full trace remains on the H100 host at
`/home/ubuntu/comparison/results/retro-multipole-32k-decode.nsys-rep`.

For reproducibility, prepare the author source with
`uv run --no-project python benchmarks/prepare_multipole_source.py AUTHOR_REPO PATCHED_DIR`,
then submit `benchmarks.h100_comparison` through the H100 `cluster-run` with
`PYTHONPATH=PATCHED_DIR:RETROINFER_DIR:.` and `uv run --no-project` using the
RetroInfer environment plus `pytest==9.2.0` and `scikit-learn==1.7.2`. Select
`--method retro-multipole` or `--method retro-multipole-squeezed` and
`--multipole-graph`; the run metadata records the complete command and source
snapshot. The batch versions use [the batched Multipole adapter](../retro_multipole_batch_adapter.py)
and [the batched LoD adapter](../retro_lod_batch_adapter.py).

An experimental split lookup adapts the kernel from the authors' kernel
benchmark while retaining per-query normalization and whole-cluster selection
from the generation path. It masks empty clusters and stores logits to enable
stable normalization; the [preparation script](../prepare_multipole_split.py)
records those changes. On 32K, its batch-one median was 14.288 ms/token
(one warmup, two measured 1,025-token runs), versus 15.279 ms/token for the
original lookup; at 64K the split version measured 16.404 ms/token versus
19.035. At batch four and 32K it measured 17.616 ms per step versus 18.113; at
batch eight it measured 22.476 ms versus 22.392.
The split and original lookups produced the same teacher-forced next-token
choices at all 140 tested positions for each request at batch one and four
at 32K, and at batch one at 64K. At batch eight and 32K, one of 1,120
teacher-forced choices differed. These checks do not establish task-quality
equivalence. To enable the experimental split lookup, first run
`uv run --no-project python benchmarks/prepare_multipole_split.py AUTHOR_REPO PATCHED_DIR`,
then set `MULTIPOLE_SPLIT_LOOKUP=1` for the run. It is not the default.

The authors' compiled clustering is incompatible with the Triton
3.7 override used by the earlier Squeezed port, so these runs use the
RetroInfer environment's native Triton.


## Multipole split-size and paper-style H100 checks (synthetic single layer at 128K)

For the existing batch-eight, 32K, BF16, 512-exact-token Llama 3.1
RetroInfer comparison, changing only the experimental centroid lookup split
from 256 to 2,048 increased median decode time from 22.476 to 24.016 ms per
batch step (355.9 to 333.1 tokens/s); both runs use a whole-model CUDA graph,
257 output tokens, one warmup, two measured repetitions, and include two
online updates. The sparse-attention kernel already used a 2,048 split.
The 2,048 configuration passed the batch-eight explicit-frontier check and
agreed with eager execution on every request's next token in 11 same-cache
checks (maximum sampled absolute logit difference 0.844). Against the 256
split, one of 1,120 teacher-forced choices differed. These checks do not
establish quality equivalence. The experimental lookup still defaults to 256;
`MULTIPOLE_LOOKUP_SPLIT_SIZE` selects another multiple of 128.
[Timing](../../results/h100-baselines/multipole-split2048-b8-final.json),
[graph check](../../results/h100-baselines/multipole-split2048-b8-check.json).

We also measured the released Multipole paper benchmark's separate attention
components on H100, using Qwen3-8B head geometry, synthetic BF16 tensors,
128K context, one centroid per 16 tokens, and 90%/95% sparsity. We used its
Triton `do_bench` median procedure for the dense, centroid lookup (including
selection), replacement, and sparse kernels. We add the median separately
measured FP32 online cluster update on 8K tokens, divided by 128 steps, as in
the paper. These are sums of isolated measurements rather than a timed
end-to-end pipeline; input tensors and lookup results are generated
independently for the separate stages. The released benchmark selection uses
different scoring and selection from the released model generation path.
The paper reports split 2,048; the released benchmark defaults to 512, so both
are shown. GPU and shapes differ from the full-model test above.

| Batch | Split | 90% sparse: dense / Multipole (ms) | Speedup | 95% sparse: dense / Multipole (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,048 | 0.261 / 0.366 | 0.71x | 0.260 / 0.333 | 0.78x |
| 4 | 2,048 | 0.833 / 0.489 | 1.70x | 0.833 / 0.426 | 1.96x |
| 8 | 2,048 | 1.592 / 0.753 | 2.11x | 1.593 / 0.666 | 2.39x |
| 16 | 2,048 | 3.044 / 1.331 | 2.29x | 3.045 / 1.148 | 2.65x |
| 1 | 512 | 0.443 / 0.313 | 1.42x | 0.444 / 0.259 | 1.72x |
| 4 | 512 | 1.069 / 0.499 | 2.14x | 1.069 / 0.428 | 2.50x |
| 8 | 512 | 1.866 / 0.808 | 2.31x | 1.870 / 0.694 | 2.70x |
| 16 | 512 | 3.529 / 1.418 | 2.49x | 3.529 / 1.207 | 2.92x |

At batch one, the 512-split dense kernel is substantially slower than the
2,048-split version, which inflates the ratio if the same split is forced for
both methods. At batch eight and 95% sparsity with split 2,048, the measured
components are lookup 0.290 ms, replacement 0.108 ms, sparse attention 0.130
ms, and amortized cluster update 0.137 ms. Relative to the paper's A100
results, the H100 results reproduce stronger gains at larger batches but not
the reported batch-one speedup with the paper's split. Neither measurement is
a full-model speedup.

Raw [component timings](../../results/h100-baselines/multipole-paper-h100-panel.json)
and [cluster timings](../../results/h100-baselines/multipole-paper-clustering.json);
benchmark drivers are [attention](../multipole_paper_bench.py) and
[clustering](../multipole_paper_clustering.py). Runs were submitted through the
H100 `cluster-run` adapter with `uv`; author source is MultipoleAttention
commit `db65e51d582b9bcb89807b9d9397c3532d1099ef`.

## Dense-kernel head-to-head (synthetic single layer) and full-model update cost

On H100, compare the dense FlashAttention kernel used in the RetroInfer
runtime against Multipole's benchmark FlashDecoding baseline with matching
random Q/K/V data, Qwen3-8B/Llama 3.1 8B head geometry (32 query heads,
8 KV heads, head dimension 128), and BF16. Each implementation receives its
native KV layout; layout conversion is outside the timed region. Times below
are median single-attention-call CUDA-graph replay times in milliseconds, not
whole-model timings. We also measured the paper's `do_bench` method; those
raw timings and maximum output differences (at most 0.000245) are in the
[result file](../../results/h100-baselines/multipole-dense-head-to-head.json).

| Context | Batch | FlashAttention | Paper dense split 2,048 | Paper dense split 512 |
|---:|---:|---:|---:|---:|
| 32K | 1 | 0.066 | 0.071 | 0.117 |
| 32K | 4 | 0.192 | 0.217 | 0.282 |
| 32K | 8 | 0.364 | 0.406 | 0.488 |
| 32K | 16 | 0.715 | 0.781 | 0.897 |
| 128K | 1 | 0.197 | 0.259 | 0.445 |
| 128K | 4 | 0.710 | 0.833 | 1.061 |
| 128K | 8 | 1.409 | 1.582 | 1.867 |
| 128K | 16 | 2.812 | 3.020 | 3.518 |

Using the 128K batch-eight `do_bench` FlashAttention time (1.419 ms) as the
baseline for the separate paper-style 95%-sparse component sum (0.666 ms)
gives approximately 2.13x, versus 2.39x using the paper dense kernel. These
component timings come from separate synthetic runs, not an end-to-end
pipeline, and differ from the Llama generation path. The head-to-head driver
is [multipole_dense_compare.py](../multipole_dense_compare.py).

To measure the actual update overhead, the batch-eight, 32K, 256-step
whole-model experiment was rerun with explicit synchronization around each
update. Following warmup, four updates took 0.613-0.621 seconds each, about
4.83 ms per batch step amortized across the 128-step interval. Each update
includes per-layer clustering, rebuilding value centroids and counts from the
labels, metadata copies and local-window shifts; the paper microbenchmark
separately times a clustering function, not this complete maintenance path.
The measured total decode time was about 22.41 ms per batch step. The profile
synchronizes around updates and may add diagnostic overhead. It does not
attribute the update time among its components.
[Update profile](../../results/h100-baselines/multipole-b8-update-profile.json).

The paper's lookup measurement already includes its own per-token top-k
selection. Our earlier model trace reports sorting separately because the
released generation path uses normalized cluster scores, sorting and a
whole-cluster cutoff plus index compaction; these are different selection
implementations, not an omitted sort to add to the paper's lookup time. The
paper tests sparse attention with dummy merge inputs and does not measure our
local-attention call, per-step KV/cache bookkeeping, or index-installation
steps as a connected model pipeline. Initial clustering is discussed
separately by the paper and is included in our prefill timings.

## Matched attention and update audit (synthetic single layer)

The 128K paper-style speedups above must not be compared directly with the
32K full-model timing. We timed dense and the complete adapted Multipole
attention path on the same synthetic KV data at batch eight, BF16 and a
512-token Multipole budget. Each one-layer CUDA graph includes its own KV
writes; Multipole includes lookup, selection and compaction, local attention,
replacement and sparse attention. Random cluster labels are used, with counts
and key/value centroids constructed from the same input data. The selected
counts ranged from 497 to 512 at 32K and 495 to 512 at 128K. Input prefix
lengths are 32,768 and 131,072, respectively, with a new KV appended.

Separately, we measured one complete update of the same one-layer cache,
restoring identical state before each update and discarding three warmups to
avoid including `torch.compile` shape compilation. The median of the three
remaining synchronized wall-clock measurements is divided by the 128-step
interval. The summed time is a diagnostic estimate, not a graph timing of a
complete sequence and not a measurement of a 128K full model.

| Prefix | Dense step (ms/layer) | Multipole steady step (ms/layer) | Update / 128 (ms/layer) | Multipole sum (ms/layer) |
|---:|---:|---:|---:|---:|
| 32K | 0.3723 | 0.3253 | 0.1446 | 0.4699 |
| 128K | 1.4192 | 0.8459 | 0.2049 | 1.0507 |

For 32K, the update outweighs the steady attention saving. Extrapolating the
0.1446 ms/layer update across 32 layers gives roughly 4.63 ms per batch step,
close to the 4.83 ms measured in the actual batch-eight model update profile;
the synthetic per-layer estimate also includes synchronization overhead.
At 128K, the same synthetic one-layer comparison predicts a net benefit, but
cannot establish a 128K full-model speedup or quality equivalence. The
outputs are [32K](../../results/h100-baselines/multipole-matched-32k-fixed-update.json)
and [128K](../../results/h100-baselines/multipole-matched-128k-fixed-update.json);
[driver](../multipole_matched_attention.py).

For a separate check of the authors' benchmark at 32K and a 512-token budget,
split 2,048 yielded 0.1388 ms lookup, 0.0921 ms replacement and 0.0273 ms
sparse attention. Adding the independently measured 0.1373 ms amortized
8K-cluster update gives 0.3956 ms, versus 0.4110 ms for the author's dense
kernel or approximately 0.3692 ms for the separately measured FlashAttention
baseline. These are paper-style isolated synthetic measurements with a
different selection path from the model integration, not the connected path
above. [Results](../../results/h100-baselines/multipole-paper-b8-32k-budget512.json).


## Qwen3-8B Figure 5 kernel reproduction on H100

The released Multipole `benchmark/benchmark_kernels.py` is a synthetic,
single-layer microbenchmark with Qwen3-8B's 32 query heads, 8 KV heads and
128-dimensional heads. It does not load Qwen3-8B weights or run full-model
inference. We ran the released script directly on the H100 (its default split
is 512), and ran the same author benchmark functions with split 2,048 to match
the paper's reported setting. We used 131,072 KV tokens, BF16, one centroid
per 16 tokens, batches 1/4/16, and 90%/95% pruning. The two paper-split runs
independently measure each stage with the author's `triton.testing.do_bench`
settings. The table uses the median of the two reported stage medians, then
adds a separately measured median 8K cluster-update cost divided by 128 steps.
The update was measured once with the authors' sequential update function.
The component sum is not a timed end-to-end pipeline; the benchmark inputs
and selection state differ between separately timed components.

| Batch | Pruning | H100 dense (ms) | H100 component sum (ms) | H100 speedup | Figure 5 A6000 speedup | Figure 6 A100 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 90% | 0.259 | 0.392 | 0.66x | 2.4x | 1.2x |
| 1 | 95% | 0.261 | 0.359 | 0.73x | 2.5x | 1.3x |
| 4 | 90% | 0.836 | 0.507 | 1.65x | 3.3x | 1.9x |
| 4 | 95% | 0.835 | 0.431 | 1.93x | 4.1x | 2.1x |
| 16 | 90% | 3.012 | 1.326 | 2.27x | 3.7x | 2.4x |
| 16 | 95% | 3.016 | 1.139 | 2.65x | 4.5x | 2.8x |

At batch one and 90% pruning, the H100 sum consists of lookup 0.197 ms,
replacement 0.093 ms, sparse attention 0.076 ms, and amortized update
0.027 ms. The paper reports A6000 for Figure 5 and Appendix E identifies
Figure 6 as A100. Those numbers are paper-reported values, not H100
measurements. As in the original harness, this comparison uses the authors'
own dense FlashDecoding baseline and synthetic independently prepared inputs;
it is neither a full-model speedup nor a same-hardware reproduction of the
paper. The independently measured cluster update is shared across both repeats.

Raw component results: [first](../../results/h100-baselines/qwen3-multipole-figure5-h100-split2048.json),
[repeat](../../results/h100-baselines/qwen3-multipole-figure5-h100-split2048-repeat.json),
and [cluster update](../../results/h100-baselines/qwen3-multipole-figure5-h100-cluster.json).
The [run records and direct released-harness log](../../results/h100-baselines/provenance/qwen3-multipole-figure5/)
include the `cluster-run` commands. The author checkout was at
`db65e51d582b9bcb89807b9d9397c3532d1099ef`.

## LoD at the Multipole synthetic-layer benchmark boundary

[The LoD benchmark adapter](../multipole_lod_bench.py) constructs the production
three-tier LoD cache from synthetic BF16 K/V with Qwen3-8B head geometry and
times its connected graph-captured decode path. It includes KV append, routing,
page opening, and real cache catch-up. Each row starts with 131,072 tokens,
warms up for 512 decode steps including updates, then measures two 512-step
windows; LoD updates are every 256 tokens. The prefix grows during the timed
windows. Requests have independent synthetic prefixes, but the query and new
K/V are held fixed during decode. No Qwen model weights are loaded.

The following compares LoD with the authors' Figure 5-style H100 benchmark
above and an additional author-kernel run with a 512-token budget. All rows
use the same batch and starting context length. The dense and Multipole rows
are independently timed kernel-component measurements at fixed context,
whereas LoD is a connected-path wall-clock measurement including host cache
bookkeeping and updates. Their inputs are independently generated, and this
is not an accuracy-matched comparison or a full-model result. Multipole's
512-token microbenchmark does not include a separately timed local window.
LoD opens at most eight regions and two 16-token pages per region, alongside
its exact local field and sink; the 95% pruning case selects about 6,554 exact
tokens out of 131,072. Consequently, equal model geometry does not imply equal
exact-token budgets or quality.

| Batch | Author dense (ms) | Multipole 95% prune (ms) | Multipole 512-token budget (ms) | LoD three-tier BF16 (ms) | LoD three-tier INT4 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.261 | 0.359 | 0.342 | 0.0647 | 0.0632 |
| 4 | 0.835 | 0.431 | 0.402 | 0.1474 | 0.1420 |
| 16 | 3.016 | 1.139 | 0.970 | 0.4498 | 0.4471 |

The first two columns of component data come from the paper-style split-2048
panel, with the independently measured clustering update divided by 128; the
512-budget column uses a separate split-2048 author run with the same cluster
update. LoD uses the median of its two timed-window wall times. The graph and
eager LoD outputs matched exactly on the same cache before and after the timed
updates in all six cases. This is a narrow execution check, not an accuracy or
quality evaluation. LoD uses Triton 3.7.1; the author component and cluster
tests use Triton 3.1.0. Attempting the author benchmark with 3.7.1 failed in
PyTorch Inductor because `triton_key` could not be imported, so it is not a
fully common software stack. No timing from the failed run is used.

Results are in `results/h100-baselines/multipole-lod-v2-*.json` and
[Multipole 512-token component timings](../../results/h100-baselines/multipole-paper-128k-small-budget.json).
[Cluster-run records](../../results/h100-baselines/provenance/multipole-lod/)
include both the successful LoD and author-kernel runs and the failed
Triton 3.7.1 compatibility attempt.

## Connected Multipole and LoD graph comparison

**Timing-boundary caveat:** This compares our connected generation adapters,
not the kernel-only operations timed in Multipole Figure 5. It cannot explain
the paper's reported kernel latencies or be used as a direct reproduction of
that experiment. The connected adapters also have different selection and
cache-maintenance implementations.

[This single-layer benchmark](../multipole_connected_bench.py) captures the
connected Multipole generation path through the existing RetroInfer batch
adapter: KV append, centroid scoring and selection, local attention, centroid
replacement, and sparse attention. It runs the full cluster update and related
cache maintenance outside the graph every 128 tokens, inside the timed window.
LoD's connected graph and updates are measured by `multipole_lod_bench.py`.
For both methods: one synthetic Qwen3-8B-shaped BF16 attention layer, 131,072
initial KV tokens, batch 1/4/16, fixed Q and new K/V, 512 warmup decode steps,
and two 512-step timed windows. Multipole's initial cluster labels are random;
centroids, counts, and value means are computed from those labels and K/V.
Inputs differ between methods, context grows during timing, no model weights
are loaded, and accuracy and quality have not been matched.

| Batch | Multipole, 512-token budget | Multipole, 5% budget | Multipole, 10% budget | LoD three-tier BF16 | LoD three-tier INT4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.233 ms | 1.200 ms | 1.233 ms | 0.0647 ms | 0.0632 ms |
| 4 | 1.500 ms | 1.432 ms | 1.482 ms | 0.1474 ms | 0.1420 ms |
| 16 | 2.335 ms* | 2.452 ms | 2.598 ms | 0.4498 ms | 0.4471 ms |

*The original batch-16/512 windows were 2.495 and 6.718 ms. A subsequent
four-window repeat measured 2.303, 2.340, 2.330, and 2.375 ms; the table
reports the latter median. The cause of the original outlier is not established.
All other rows use medians of two windows. The 5%/10% budgets are 6,554/13,107
tokens, respectively. The windowed cluster update ran exactly four times per
Multipole window and LoD's every-256-token updates also occurred in each LoD
window. Initial capture, prefix setup and compilation warmup are excluded.

These timings do not reproduce the separate Figure 5 components: the connected
adapter uses the released generation kernels and generation selection policy,
whereas the paper microbenchmark has a distinct split lookup and times its
components independently. The environment also differs (Triton 3.1 for the
author generation code, 3.7.1 for LoD). In a separate instrumented batch-16
512-budget run, the adapter's update including index maintenance, value
centroids and local-window movement took about 157–184 ms per update after
warmup; profiling adds synchronization. This helps explain the difference from
the isolated clustering kernel but is not an isolated clustering-kernel time.

The 512-token budget is a maximum, not a guaranteed selection count: in a
post-update snapshot some heads selected as few as one token on these synthetic
inputs. Repeated identical new K/V and random initial assignments may affect
this distribution; do not infer quality or a quality-matched speedup. Sampled
graph/eager outputs matched within 0.002 absolute error after updates across the
reported cases, but this is not an end-to-end quality check.

The original batch-16 generation kernel failed with an illegal memory access:
its 32-bit KV head-stride multiplication can overflow for this geometry. The
[overlay generator](../prepare_multipole_wide.py) casts the head index to int64
before the multiplication. The patched kernel passed a dense reference for
all 128 KV heads, including offsets beyond 2^31 elements, as well as the
existing batched attention-frontier tests. Batch 16 uses this explicitly
modified kernel; batches 1 and 4 use the unmodified author code.

Raw records are `results/h100-baselines/multipole-connected-*.json`; the
[cluster-run logs](../../results/h100-baselines/provenance/multipole-connected/)
include the failed original batch-16 run, the reference checks, repeat and
instrumented profile. All rows are synthetic-layer results and must not be
presented as full-model timings.

## Graph-captured paper-component comparison, without local or sink attention

**Update-scope caveat:** The first table below combines the paper-style
isolated Multipole clustering measurement with complete LoD catch-up. For a
comparison that includes complete cache updates on both sides, see the next
section, which supersedes these combined estimates for that purpose.

This H100 comparison captures the same three functions used by the released
Multipole Figure 5 component script: centroid lookup and selection, centroid
replacement, and sparse attention. LoD captures its remote routing, page
selection, exact-page attention, and summary reduction. Both attention graphs
exclude local and sink attention, new-token KV writes, and cache updates.
The LoD prefix still contains a 256-token local field and a sink, but these
entries are excluded from its measured attention; the author's synthetic KV
fixture has 131,072 entries. Thus the attended remote populations differ
slightly even though both use a nominal 128K prefix. No model weights are
loaded. Qwen3-8B head geometry, BF16 inputs and batches match.

Each table cell shows **graph time + amortized update time = combined estimate**,
in milliseconds per batch step. The three Multipole columns select at most
512, about 6,554, and about 13,107 KV entries, respectively. Three-tier LoD
selects at most eight regions and two 16-entry pages per region. These budgets
are not quality matched. The final two columns use BF16 and INT4 LoD page KV.

| Batch | Multipole 512 budget | Multipole 5% budget | Multipole 10% budget | LoD BF16 | LoD INT4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.206 + 0.027 = 0.233 | 0.249 + 0.027 = 0.276 | 0.258 + 0.027 = 0.284 | 0.056 + 0.008 = 0.063 | 0.056 + 0.008 = 0.064 |
| 4 | 0.257 + 0.075 = 0.332 | 0.311 + 0.075 = 0.386 | 0.369 + 0.075 = 0.444 | 0.123 + 0.029 = 0.152 | 0.121 + 0.029 = 0.150 |
| 16 | 0.660 + 0.270 = 0.930 | 0.827 + 0.270 = 1.098 | 1.016 + 0.270 = 1.286 | 0.393 + 0.098 = 0.491 | 0.388 + 0.101 = 0.489 |

Multipole's graph joins the original author closures, but their fixtures remain
independent: the replacement output is not fed into the sparse-attention test,
whose dummy merge state is initialized to an empty state outside timing. This
is a graph of the paper's measured components, not a connected implementation
of the algorithm. Each Multipole graph time is one 512-replay wall-clock sample;
each LoD value is the median of two such samples. Graph replay avoids the
per-iteration cache clearing used in the original component benchmark. These
numbers therefore need not match the earlier sum of separately timed kernels.
LoD graph and eager outputs matched exactly on the same cache, and perturbing
the excluded local and sink KV produced no output change in all six runs.
These checks do not establish full-model accuracy.

The released Figure 5 component script does not time cluster updates. As in
our earlier paper-style estimate, the Multipole addend is the separately timed
released sequential cluster-update function on synthetic inputs, divided by
128. It does not include the complete generation adapter's index and cache
maintenance. For LoD, the addend is the median of four separately timed real
production catch-ups after two update intervals of warmup, divided by 256;
it includes host bookkeeping and cache maintenance. These are different
update scopes and timing methods, so the combined values are component-based
estimates rather than matched end-to-end generation times. Real updates to
the local cache are included in the LoD catch-up cost even though local
attention is excluded from the graph. LoD used Triton 3.7.1 and the Multipole
code used Triton 3.1.0 because the author benchmark did not run with the newer
version in this environment.

Sources: [Multipole graph harness](../multipole_paper_graph.py),
[LoD graph and update harness](../multipole_lod_bench.py),
[separate Multipole update](../../results/h100-baselines/qwen3-multipole-figure5-h100-cluster.json),
and [run records and logs](../../results/h100-baselines/provenance/multipole-paper-remote/).
The raw files are `multipole-author-remote-graph-*.json` and
`multipole-lod-remote-three-tier-*.json` in `results/h100-baselines/`.

## Complete update costs for both methods

We also separately timed complete updates of the synthetic 128K caches used
by each method's adapter: from an update-ready state until its routing,
counts and value summaries are installed for the next attention step. Both
include host work and local-cache movement, but exclude per-token KV append,
attention and model execution. The respective update intervals are 128 tokens
for Multipole and 256 for LoD. The table reports the median of four
synchronized wall-clock samples after two warmup updates, followed by the
amortized cost per batch step.

| Batch | Multipole complete update | Multipole / 128 | LoD BF16 update | LoD BF16 / 256 | LoD INT4 update | LoD INT4 / 256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.683 ms | 0.771 ms | 2.011 ms | 0.0079 ms | 2.136 ms | 0.0083 ms |
| 4 | 109.125 ms | 0.853 ms | 7.353 ms | 0.0287 ms | 7.310 ms | 0.0286 ms |
| 16 | 157.926 ms | 1.234 ms | 25.060 ms | 0.0979 ms | 25.848 ms | 0.1010 ms |

The Multipole measurement calls the authors' online cluster update through
our RetroInfer adapter and includes preparing inputs, installing labels and
counts, recomputing value centroids over the cache, local-cache movement and
host bookkeeping. Its initial clusters were constructed with the authors'
clustering routine; initial clustering and filling in newly generated
synthetic KV were outside the timed samples. The LoD results are the same
production catch-up samples described above. Multipole counts covered every
remote KV entry in every head, and an independently recomputed value summary
matched the stored summary for one head in each run. This validates part of
the updated state, not end-to-end model quality.

Adding the complete-update amortization to the previous graph-component
times gives the following *component-based estimates*, in milliseconds per
batch step. The intervals are each method's defaults; this does not mean
identical internal algorithms or an end-to-end execution comparison.

| Batch | Multipole 512 budget | Multipole 5% budget | Multipole 10% budget | LoD BF16 | LoD INT4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.977 | 1.020 | 1.029 | 0.063 | 0.064 |
| 4 | 1.110 | 1.164 | 1.222 | 0.152 | 0.150 |
| 16 | 1.894 | 2.061 | 2.250 | 0.491 | 0.489 |

The Multipole update numbers describe this adapter, not a claim that every
implementation of the algorithm has that cost. The authors' released Qwen
integration also handles RoPE differently, and the graph-captured Multipole
paper components still use independent fixtures rather than the updated
adapter cache. Synthetic inputs, unequal exact-token budgets, no local or
sink attention in either graph, and the Triton-version difference described
above remain limitations. The separate [author 8K clustering microbenchmark](../../results/h100-baselines/qwen3-multipole-figure5-h100-cluster.json) excludes
the rest of this maintenance and should not be substituted for the complete
adapter update in this table.

The new harness is the `--update-only --author-initial-clusters` mode of
[multipole_connected_bench.py](../multipole_connected_bench.py). Results are
`multipole-matched-author-init-b*-128k.json`; the
[run records](../../results/h100-baselines/provenance/multipole-matched-update/)
also retain the exploratory random-initialization runs and the 8K checks.
