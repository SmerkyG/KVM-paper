# Kimi K3 24-layer MLA-stack proxy

This benchmark uses [`tests/fixtures/kimi-k3-mla-stack`](../../tests/fixtures/kimi-k3-mla-stack),
a dummy-weight model containing K3's 24 MLA layers and no KDA or FFN/MoE
compute. It retains K3's official attention dimensions, 12-layer
attention-residual grouping, DCP/TP execution, vLLM cache lifecycle, and LoD
cross-layer construction. Its purpose is fast systems development; its logits
have no quality meaning.

The model loaded about 2 GiB per rank and completed model loading in roughly
2--9 seconds, versus the multi-minute full K3 startup. Unlike the earlier
one-layer probe, it initially reproduced the full-model symptom of LoD
prefill remaining slower than dense at long context. The older tables below
are historical. See **Allocator reuse: current successful fixture path** at
the end for the completed October 4 comparisons and their limitations.

## Historical prefill results

MI325X, TP8/DCP8, BF16 cache, 16,384-token scheduler budget, one warmup plus
one measured repetition. Times are vLLM scheduled-to-first-token prefill time.

| Batch | Context | Dense (s) | Two-tier LoD (s) | Dense / LoD |
|---:|---:|---:|---:|---:|
| 1 | 16K | 0.189 | 0.204 | 0.924x |
| 1 | 64K | 1.396 | 1.837 | 0.760x |
| 8 | 16K | 1.499 | 2.216 | 0.676x |
| 8 | 64K | 11.128 | 15.603 | 0.713x |

The source JSON files are:

- `full-tp8-dcp8-b1-res12.json`
- `two-tier-tp8-dcp8-b1-res12.json`
- `full-tp8-dcp8-b8-res12.json`
- `two-tier-tp8-dcp8-b8-res12.json`

## Fixed aggregate-prefill budget

To test request concurrency without increasing the 16K activation budget, a
benchmark-only scheduler divided each 16K step among an equal-length cohort.
The table reports aggregate prompt throughput.  The LoD measurements below
construct state after the first scheduler chunk; they therefore test the cost
of chunking, not a proposed policy that would retain an exact prefix and defer
the first LoD construction until the request reaches 16K.

| Cohort and per-request chunk | Dense 16K | LoD 16K | Dense 64K | LoD 64K |
|---:|---:|---:|---:|---:|
| B1 x 16K | 88,990 tok/s | 75,718 tok/s | 47,591 tok/s | 41,685 tok/s |
| B2 x 8K | 83,245 tok/s | 101,378 tok/s | 41,210 tok/s | 49,970 tok/s |
| B4 x 4K | 78,126 tok/s | 118,106 tok/s | 34,480 tok/s | 48,299 tok/s |
| B8 x 2K | 68,151 tok/s | 118,516 tok/s | 27,856 tok/s | 31,826 tok/s |

B2 x 8K is the best 64K LoD point in this sweep.  B8 x 2K is memory-feasible,
but eight scheduler/model passes per 16K request interval erase much of the
attention saving.  Dense attention, which performs no LoD construction, also
falls from 47,591 tok/s at B1 x 16K to 27,856 tok/s at B8 x 2K; consequently,
deferring LoD construction to the 16K boundary cannot remove the dominant
repeated-pass cost.

The fixed-budget source files are named `*-b2x8k.json`, `*-b4x4k.json`, and
`*-b8x2k.json` (with `-incremental` on the LoD files).

## Reproduction

Dense batch 1:

```bash
benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --mode full --lengths 16384 65536 --batch-size 1 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --output results/kimi-k3-mla-stack/full-tp8-dcp8-b1-res12.json
```

Replace `--mode full` with `--mode two-tier` for LoD, and use
`--batch-size 8` for the batch-8 panel. The benchmark sets `load_format` to
`dummy` and does not need K3 checkpoint weights.

## Independent-centroid DCP prefill prototype (2026-10-04)

**Rejected timing evidence:** a subsequent GPU correctness test found that
the cached `...kimi_d192v128_asyncbias_v10` binary was compiled with
`CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1`. It emitted only one winner per tile,
leaving seven candidate channels unwritten. Consequently the prototype
measurements below cannot establish the speed of correct top-eight LoD.
This also disqualifies any earlier v10 comparison that loaded that same
binary. The current source explicitly sets this flag to zero and uses v12
to avoid reusing the incorrect cached module. No release timing table has
been replaced with these experimental values.

The opt-in `LOD_KIMI_DCP_LOCAL_PREFILL=1` path constructs and archives only
each rank's interleaved sequence slice. Each GPU retains a whole-sequence
`16 sqrt(T)` centroid budget, subject to the available local tokens; this is
not a globally divided centroid budget. The current causal 16K chunk remains
exact and head-sharded. For remote attention, all 96 query heads are gathered,
each GPU independently refines eight local centroids, and the disjoint remote
outputs/LSE are combined and head-scattered. No global routing top-k is
performed. Only the owner of global position zero retains a sink.
Every measured length is warmed separately and construction is included in
scheduled-to-first-token timing. This first synchronous-construction prototype
supports aligned initial-prefill chunks, two-tier BF16 and unit DCP interleave.
It does not support arbitrary chat continuations.

| Batch | Context | Previous LoD prefill (s) | Independent local prefill (s) |
|---:|---:|---:|---:|
| 1 | 16K | 0.199 | 0.162 |
| 1 | 32K | 0.564 | 0.921 |
| 1 | 64K | 1.590 | 2.668 |
| 8 | 16K | 1.624 | 1.536 |
| 8 | 32K | 6.515 | 8.391 |
| 8 | 64K | 14.177 | 24.195 |

Previous B1 values: `experimental-packed-argmax-v10-b1-16-64k.json`.
Previous B8 values: `matched-current-two-tier-tp8-dcp8-b8-16-64k.json`, before
the final packed-argmax optimization; these are historical controls, not a
fresh v10 B8 run. New values: `local-dcp-prefill-b1.json` and
`local-dcp-prefill-b8.json`. B8 submits eight requests under the same 16K
aggregate scheduler budget; it is not a true B8 x 16K prefill step. These are
dummy-weight fixture timings, not pretrained-model quality measurements.

Separate warmed kernel comparisons used Q=16K, S=3547, three measured samples:

| Operation | Original geometry | Sequence-sliced geometry |
|---|---:|---:|
| Fused route/coarse, candidate reduction and expansion | H12: 3.251 ms | H96: 20.714 ms |
| Expanded routed leaves, uniform random owners/routes, page size 16 | H12 / 49152 leaves: 2.064 ms | H96 / 6144 leaves: 8.999 ms |

These isolated timings are not an additive end-to-end attention decomposition.
More heads improve efficiency per head but do not offset scoring the same
centroid count for eight times as many heads. Fine attention additionally pays
for minimum 16-token tiles when local centroid lists become short. Correctness
checks passed for refinement LSE and independent-partition LSE combination;
the complete prefill path ran at all six listed fixture points. Defaults are
unchanged. No full-model run was undertaken for this slower prototype.

Reproduce B1 without a cluster runner:

```bash
LOD_KIMI_DCP_LOCAL_PREFILL=1 benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --mode two-tier --lengths 16384 32768 65536 --batch-size 1 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --output results/kimi-k3-mla-stack/local-dcp-prefill-b1.json
```

Use `--batch-size 8` and the corresponding output filename for B8. GPU checks:
`benchmarks/run_kimi_k3_v10_direct.sh -m pytest -q tests/test_kimi_k3_lod.py
-k 'independent or refinement_returns'`.

## Replicated-summary / sharded-leaf hybrid (2026-10-04)

`LOD_KIMI_DCP_SHARED_PREFILL=1` divides the original total centroid budget
among DCP ranks. Each rank constructs its own interleaved 2K-token share of
every global 16K update, then exchanges FP32 centroid sums, counts and leaf
lengths. Centroid scoring runs on each query-head owner (12 heads, not 96).
Exactly eight total regions are selected, with oversized winners left coarse
and singleton winners left coarse because refinement is algebraically
redundant. Queries/routes are gathered for owner-local leaf refinement;
partial fine outputs/LSE are combined and head-scattered. The selected
centroid contributions are removed exactly once on the query owner.
The sink and 256-token old exact tail are replicated and included with the
current causal chunk. Persistent leaves remain 512+64 latent records.

This first version intentionally waits synchronously for cache construction;
it does **not** yet test request-level construction/attention pipelining.
It supports aligned initial-prefill chunks, two-tier BF16 and unit interleave
only. GPU checks additionally cover FP32-sum-to-BF16-mean projection, route
ownership, aggregated fine output and empty refinement sets. State updates
use a separate temporary V workspace so writes cannot overlap its persistent
K-prefix alias.

The first completed runs (`shared-dcp-prefill-b1.json`, job 20755 on node 3;
`shared-dcp-prefill-b8.json`, job 20754 on node 2) loaded the rejected v10
binary described above. Their raw prefill seconds were B1 0.179/0.649/1.726
and B8 1.496/5.434/14.776 at 16K/32K/64K. These are retained solely as
debugging provenance, **not valid top-eight performance results**. Earlier
jobs 20751/20752 failed on FP32/BF16 projection dtype mismatch and produced
no measurements. Correct top-eight fixture timing must use v12 and record
the worker kernel build flags before making performance claims.

Reproduce the corrected B1 path without the cluster runner:

```bash
LOD_KIMI_DCP_SHARED_PREFILL=1 benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --mode two-tier --lengths 16384 32768 65536 --batch-size 1 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --output results/kimi-k3-mla-stack/shared-dcp-prefill-b1-v12.json
```

### Next scheduling experiment

The current 16K aggregate scheduler normally executes one long prefill row at
a time, even when eight requests are queued. `wait_deferred_prefill()` uses
a host-synchronized completion event because vLLM can consume the cache on a
different graph stream; replacing this with a wait on an unrelated current
stream previously corrupted results. Production construction already overlaps
later layers, but pending-cache rows have no request-level readiness queue.

A conservative next step is to rotate 16K chunks among requests and skip a
request until its cache event is ready, while executing another ready row.
Updates retain the global per-request 16K prefill/256 decode cadence, and no
consumer uses stale state. Per-row transient workspaces and collective order
must be isolated before allowing construction/summary exchange to overlap
another request's TP/DCP collectives. Final construction must still be charged
to prefill or the first subsequent consumer, not hidden by benchmark fences.

The benchmark-only `LOD_BENCHMARK_ROTATE_PREFILLS=1` now implements this
rotation without preemption or cache-owner changes. It gates other prefills
for one scheduler step and advances the least-progressed eligible row. The
16K aggregate token budget and per-request global update boundaries are
unchanged. Intermediate builds need not finish at the end of every model
call; the consuming row still waits for its own build event, and final builds
are included in the measured cohort completion time.

The corrected v12 serial control (`rotation-serial-b8-v12.json`, job 20764,
node 4, TP8/DCP8, GPU memory utilization 0.10) measured 1.611 / 6.453 / 14.086
seconds for eight queued 16K / 32K / 64K requests. All worker binary audits
passed. These timings include final cache construction, one exact-shape
warmup and one measurement per point. The first rotation attempt (job 20765)
measured 1.612 seconds at 16K with identical generated token IDs, then failed
during the 32K warmup with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, reporting
zero free GPU memory. No 32K/64K rotation performance claim is valid from
that attempt. The full K3 weight daemon remains resident but idle during
these fixture experiments; it substantially reduces available memory.
Simply reducing `gpu_memory_utilization` to 0.04 (job 20766) failed vLLM's
startup cache-budget check before inference. The next probe uses its explicit
`--kv-cache-memory-bytes 4294967296` override instead, retaining utilization
0.10 and leaving more free space for simultaneous temporary LoD caches.
Native-cache allocation size is recorded in the fixture result JSON.
That probe (job 20767) completed the measured 32K point in 6.500 seconds,
with the same eight generated token IDs as the serial control, then failed
the 64K warmup with another resource-allocation error (126--174 MB reported
free on the failing GPUs). This is a partial log result, not a completed
worker-audited artifact, and the serial control used a larger native cache.
It shows no meaningful scheduling gain at 32K, but does not evaluate the
one-request-per-GPU attention layout or its overlap with MoE. The fixture
runner now saves every completed point with pending audit status so later
capacity failures do not erase prior measurements.

A separate proposed layout places each request's complete attention on one
GPU while retaining distributed MoE. Its output transfer is a pipeline
dependency, **not automatically a serial per-row latency penalty**: other
rows' attention/MoE can run while the transfer completes. Evaluate filled
pipeline throughput including fill/drain and final construction, not the
sum of isolated attention and transfer latencies. Transfer bandwidth,
collective ordering, shared compute resources and live-buffer memory still
constrain throughput. The rotation experiment above is only a scheduling
probe; it does not implement this attention-owner/distributed-MoE layout.

### Corrected top-eight experiments completed on October 4

The following LoD runs loaded the corrected v12 route/coarse binary and passed
the worker audit. They keep TP8/DCP8, eight requests, a 16K aggregate scheduler
chunk, top-eight refinement and the global per-request update cadence. The
weight daemon was resident but idle on node 4. The dense control ran on node 2;
as requested, nodes are treated equivalently unless a stack effect is found.

| Layout | 16K prefill (s) | 32K prefill (s) | 64K prefill (s) | Source |
|:--|--:|--:|--:|:--|
| Dense, original TP8/DCP8 | 1.730 | 4.293 | 11.739 | `current-dense-b8-native4g.json` |
| LoD, original serial, native allocation about 14 GiB | 1.611 | 6.453 | 14.086 | `rotation-serial-b8-v12.json` |
| LoD, shared summaries and sharded prefill leaves | 1.398 | 5.694 | 16.137 | `shared-dcp-prefill-b8-v12-native4g.json` |
| LoD, rotate between two unfinished prefills | — | 6.081 | 13.643 | `rotation-two-prefills-b8-v12-native4g.json` |
| LoD, construction distributed over eight layers/ranks | 1.850 | 7.065 | 14.618 | `distributed-build8-b8-v12-native4g.json` |

All rows except the serial LoD control reserve a 4 GiB native cache. The small
rotation gain therefore still needs a cache-allocation-matched serial control
before being attributed purely to scheduling. None of these alternatives is
a convincing winner at 64K. The distributed construction experiment uses a
separate sibling RCCL communicator so background construction cannot collide
with foreground TP/DCP collective ordering.

The first single-GPU **dense** owner probe (all 96 heads, TP1/DCP1, B1,
`owner-full-b1-32k.json`) measured 0.648 s at 16K and 2.024 s at 32K. These
are not eight-owner throughput results. The corresponding LoD probe exposed
an excessively expensive compilation of the generic 96-head grouped decode
router during vLLM warmup. The new single-GPU path uses the existing 16-head
MLA tiles over one physical latent cache, and advances that cache once per
token. Its completed single-owner prefill was 0.692 s at 16K, 2.305 s at
32K (`owner-lod-tiled-b1-32k.json`) and 6.843 s at 64K
(`owner-lod-b1-64k-profile.json`), with worker audits passing.

`benchmarks.kimi_k3_request_owners` now measures actual concurrent owners,
with a barrier after each exact-shape warmup and a cohort clock enclosing
release and final drain. It explicitly excludes MoE and Q/output transfers;
it must not be reported as a full-model attention-owner pipeline speedup.

```bash
benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_request_owners \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --mode full --lengths 16384 32768 65536 --owners 8 \
  --output results/kimi-k3-mla-stack/eight-owners-dense.json
```

Run the same command with `--mode two-tier` and a separate output path for
the owner-layout LoD comparison. This runner refuses non-fixture checkpoints.

Actual eight-owner results on node 2, one synchronized exact-shape warmup plus
one measured cohort per length, 3 GiB native cache per owner:

| Context | Eight dense owners (s) | Eight LoD owners (s) | Dense / LoD |
|--:|--:|--:|--:|
| 16K | 0.752 | 0.818 | 0.919x |
| 32K | 2.818 | 2.735 | 1.030x |
| 64K | 9.336 | 7.909 | 1.180x |

Source: `eight-owners-dense.json` and `eight-owners-lod.json`, jobs 20780 and
20782. All eight owners passed the correct top-eight binary audit. Each
owner's generated IDs matched the corresponding dense owner at every length,
but these dummy-weight outputs still establish **no model quality result**.
The 32K 3% difference is too small to call a robust win. The owners had a
visible wall-time spread (especially GPUs 4--6), recorded individually in the
JSON; the table deliberately uses the slowest finished owner, not the mean.

The separate opt-in diagnostic profiler in `owner-lod-b1-64k-profile.json`
shows 1.86 s of summed coarse-route GPU work, 1.81 s in the two leaf projection
GEMMs, 1.38 s in fine attention, and 0.33 s in candidate reduction. These are
**not additive wall-time components**: they overlap, and instrumentation has
overhead. The canonical prefill measurement remains the uninstrumented
6.843 s. The profiler is used only to identify optimization targets.

### Owner kernel tuning: completed, not promoted

`benchmarks.kimi_k3_owner_tune` keeps one vLLM model resident across variants;
every measured variant receives an exact-shape warmup. These experiments do
not change the centroid construction, routes, or replacement formula.

| Variant | 32K prefill (s) | 64K prefill (s) | Assessment |
|:--|--:|--:|:--|
| Original 32-query / one-wave leaf tile | 2.300 | 6.682 | Matched control |
| 64-query / two-wave leaf tile | 2.307 | 6.628 | Less than 1% at 64K |
| 128-query / four-wave leaf tile | 2.311 | 6.582 | About 1.5% at 64K |
| Project K/V inside each attention tile | 6.430 | 25.149 | Reject: repeats projection for query tiles |
| Separate projection control | 2.322 | 6.730 | Second matched experiment |
| Combined K/V projection GEMM | 2.324 | 6.695 | About 0.5%; not a meaningful win |

Sources: `owner-prefill-leaf-tune.json` (job 20784, node 3 GPU 0) and
`owner-prefill-fused-kv-tune.json` (job 20786, node 2 GPU 0). The combined
projection's GPU check passed for both paths, including direct-key channels
and the noncontiguous value view. Defaults are unchanged.

The projection-usage diagnostic (`owner-projection-usage-final.json`, job
20788) found 11,364,576 needed leaf/head pairs out of 224,716,032 projected
pairs across the 72 remote-attention calls of a 64K request (5.06%). This
counts the union of actual **post-cap** routes for each head/chunk, not raw
centroid winners. Dummy-weight queries repeat unusually strongly: each head
selected only eight distinct centroids in each chunk. This fraction must not
be generalized to real pretrained K3. The uninstrumented prefill was 6.701 s;
the extra profiler pass is diagnostic only.

Two subsequent prototypes were compared within a single resident model,
with a control measured both before and after (job 20791, node 2 GPU 0):

| Variant | 32K prefill (s) | 64K prefill (s) |
|:--|--:|--:|
| Original, before | 2.315 | 6.708 |
| Project only the union of needed centroid leaves, once per head | 2.287 | 6.479 |
| Reuse each request's projected prefix and project only appended leaves | 2.326 | 6.688 |
| Original, after | 2.325 | 6.718 |

Selective projection saves about 3.4--3.6% at 64K, not the large gain that
its reduced arithmetic alone might suggest. The prototype's tile-list
compaction requires one host synchronization. Incremental reuse needs extra
per-request projected storage and prefix copies at growth; it is effectively
neutral at these lengths. Neither is promoted. GPU checks cover selected
head/centroid pairs, inline and two-level directories, unwritten unselected
pairs, empty routes, prefix growth and isolation between requests (five
checks passed in job 20790). All tuning outputs had the same dummy-model IDs;
this is not a quality evaluation.

```bash
benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --lengths 32768 65536 \
  --variants default sparse incremental default \
  --output results/kimi-k3-mla-stack/owner-prefill-projection-tune.json
```

### Exact tile refinement and copy-free leaf results

The optional `LOD_KIMI_TILE_REFINE=1` path deliberately uses a new **v13**
coarse binary that emits one maximum per 128-centroid tile. It selects eight
tiles, re-scores those tiles, and recovers the global top eight centroids.
This is not the old broken v10 path: it never treats one winner per tile as
eight valid candidates. An excluded tile has eight tile maxima ahead of it,
so it cannot contain a global top-eight centroid. The re-score uses the same
query/key score and count bonus. FP rounding can still change near-ties.

Matched kernel probes at 16K queries and 3,547 centroids:

| Heads | Original route/coarse (ms) | Tile refinement (ms) | Original / new |
|--:|--:|--:|--:|
| 12 | 5.507--5.593 | 3.839--3.876 | 1.43--1.45x |
| 96 | 37.901--38.096 | 24.292--24.347 | 1.56x |

Sources: `route-tile-refine-h12.json` (job 20798) and
`route-tile-refine-h96.json` (job 20800). The 12-head selected sets matched
100%; the 96-head sets matched 99.999875%, with approximately two near-tie
query/head rows differing. Small/partial-tile GPU tests passed (job 20797).

`LOD_KIMI_DIRECT_LEAF_RESULT=1` additionally avoids allocating and copying an
entire second fine-output tensor when all heads fit in one projection group.
The returned workspace is consumed immediately by the replacement merge.
Eight GPU checks passed, including empty/aggregated routes and numerical
coarse/fine checks (job 20801). Neither option changes the construction cadence,
leaf cap, centroid ranking formula, or replacement formula.

| Single-owner variant | 32K prefill (s) | 64K prefill (s) |
|:--|--:|--:|
| Original, before | 2.316 | 6.717 |
| Copy-free leaf result only | 2.290 | 6.647 |
| Exact tile refinement + copy-free result | 2.241 | 6.182 |
| Original, after | 2.330 | 6.728 |

Source: `owner-prefill-direct-result-tune.json` (job 20802). Approximately 8%
at 64K is useful, but is not a full-model result.

The same change in the original TP8/DCP8 B8 layout saved much less:

| Distributed variant | 32K prefill (s) | 64K prefill (s) |
|:--|--:|--:|
| Original, before | 6.739 | 14.335 |
| Exact tile refinement + copy-free result | 6.486 | 14.053 |
| Original, after | 6.474 | 14.377 |

Source: `dcp8-prefill-route-refine-tune.json` (job 20803), native allocation
4 GiB/rank. The 64K gain is about 2%; 32K is within control variation. This
does **not** solve distributed prefill's overhead.

A fresh eight-owner comparison, LoD followed by dense on the same node and
GPUs, measured:

| Context | Eight dense owners (s) | Eight improved LoD owners (s) | Dense / LoD |
|--:|--:|--:|--:|
| 32K | 2.762 | 2.627 | 1.051x |
| 64K | 9.199 | 7.286 | 1.263x |

Sources: `eight-owners-dense-recheck.json` (job 20805) and
`eight-owners-lod-refine-direct.json` (job 20804). All worker audits passed;
generated dummy-weight IDs matched owner by owner. The cohort clock uses the
slowest completion. Faster owners individually improved about 1.10--1.12x at
64K; the larger cohort gain partly reflects slower owners' varying timings.
These results still **exclude MoE and Q/output transfers**. They establish
layout feasibility, not a working distributed-MoE attention-owner pipeline.

```bash
benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --lengths 32768 65536 --variants default refine_direct default \
  --batch-size 8 --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-memory-bytes 4294967296 \
  --output results/kimi-k3-mla-stack/dcp8-prefill-route-refine-tune.json

benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_request_owners \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --mode two-tier --lengths 32768 65536 --owners 8 \
  --tile-refine --direct-leaf-result \
  --output results/kimi-k3-mla-stack/eight-owners-lod-refine-direct.json
```

### Allocator reuse: current successful fixture path

The exact tile-refinement kernel gains alone were insufficient. A distributed
diagnostic (`dcp8-prefill-profile.json`, job 20807) found substantial host
allocation/free work: rank zero spent 3.29 s inside `hipFree` and 8.05 s in
event synchronization in that **instrumented** pass. Synchronization includes
waiting for GPU work, and these are not independent/additive wall components.
The diagnostic's uninstrumented 64K control was 16.176 s on node 2.

First, move the allocator-reclamation interval from 32K to final-only without
changing any cache updates. Matched node-3 results
(`dcp8-prefill-fence-reclaim-tune.json`, job 20806):

| Variant | 32K prefill (s) | 64K prefill (s) |
|:--|--:|--:|
| Refined routing + copy-free result, before | 6.606 | 13.910 |
| Final-only allocator reclamation | 6.596 | 12.038 |
| Remove only intermediate benchmark fences | 6.545 | 13.863 |
| Both changes | 6.614 | 11.979 |
| Refined routing + copy-free result, after | 6.487 | 13.838 |

Second, stop returning idle allocator blocks to HIP after every final build
when enough external-workspace headroom remains. All staging tensors are
still released; mandatory cache-completion fences remain. The opt-in
`LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1` retains reusable allocator blocks only
when at least 8 GiB is globally free, otherwise it reclaims as before.
This is **not** a proof that 8 GiB is sufficient for every full-model/MoE
workspace. Full-model validation is separate.

Matched node-2 results (`dcp8-prefill-allocator-reuse.json`, job 20808):

| Variant | 32K prefill (s) | 64K prefill (s) |
|:--|--:|--:|
| Final-only reclamation, before | 7.403 | 14.827 |
| Reuse idle allocator blocks | 5.376 | 12.622 |
| Final-only reclamation, after | 7.535 | 14.794 |

At 64K, current reserved memory rose from approximately 14.1 to 19.0 GiB/rank;
live allocated memory was unchanged (about 11.2 GiB). Peaks in this earlier
variant run are cumulative engine peaks, not reset per point. The runner now
resets peaks before each point's warmup. Three pressure-guard tests and ten
routing/fine/coarse GPU checks passed in job 20809. All worker audits passed.

The next fresh-process comparison on node 3, identical 4 GiB native allocation,
TP8/DCP8, eight requests, 16K aggregate scheduler budget and global per-request
16K/256 update cadence, finally beat dense:

| Context | Dense (s) | Serial LoD + reuse (s) | Rotate-eight LoD + reuse (s) | Dense / serial LoD |
|--:|--:|--:|--:|--:|
| 16K | 1.606 | 1.276 | — | 1.259x |
| 32K | 3.957 | 3.641 | 3.625 | 1.087x |
| 64K | 10.922 | 8.797 | 8.691 | 1.242x |

Sources: `dcp8-dense-recheck-oct4.json` (job 20810),
`dcp8-prefill-serial-reuse-control.json` (job 20813), and
`dcp8-prefill-rotation8-reuse.json` (job 20812). Final construction is included,
one exact-shape warmup and one measured repetition per point. Every worker
passed the loaded-binary audit, and all dummy-weight output IDs matched dense.
The serial control shows that request rotation contributes only about 0.5%
at 32K and 1.2% at 64K here: **allocator reuse, not rotation, explains most of
the improvement**. Serial reuse is therefore the simpler promising path.
The rotated 64K point peaked at 29.16 GiB live and 43.58 GiB reserved per rank.

The node-2 allocator comparison is slower than these fresh node-3 runs; do not
attribute that spread to an identified stack effect, or mix nodes/variant
orders to manufacture a speedup. Its fresh same-node dense control was
4.301/11.772 s (`dcp8-dense-same-node-recheck.json`, job 20811), so that resident
variant-run candidate still **lost** to dense. The new fixture win needs full
model testing; dummy outputs establish no model quality result.

```bash
LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 \
LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
LOD_KIMI_TILE_REFINE=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack \
  --lengths 16384 32768 65536 --variants reuse_allocator \
  --batch-size 8 --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-memory-bytes 4294967296 \
  --output results/kimi-k3-mla-stack/dcp8-prefill-serial-reuse-control.json
```

The experiment flags remain opt-in on `lod-k3`; no production default has been
promoted and this branch has not been pushed.

A separate same-node 128K comparison also fits and favors LoD:
36.405 s dense versus 30.163 s rotated LoD, **1.207x** (node 2,
`dcp8-dense-128k-recheck.json`, job 20815, and
`dcp8-prefill-rotation8-128k.json`, job 20814). Both worker audits passed and
dummy-weight IDs matched. LoD peaked at 46.72 GiB live and 54.50 GiB reserved
per rank. Do not compare its scaling directly with the node-3 64K row as if
they were a same-node, same-process sweep. Neither point measures amortized
decode, since each request generates only its prefill-produced token.

A fresh serial/reuse process on node 2 (`dcp8-prefill-fresh-serial-reuse-node2.json`,
job 20819) still measured 5.500/13.016 s at 32K/64K. Thus variant ordering alone
does not explain its difference from node 3's 3.641/8.797 s. Worker audits
(including loaded binaries, effective four-layer construction grouping, HIP
version, model settings, and device architecture) matched. The later diagnostic
identified three orphan workers from job **20312** on node 2's GPUs 4--6
(PIDs 659362/659380/659406, parent PID 1, stdout still pointing at that job's
October 2 log). They were consuming about 8 GiB each and spinning, although
the job was absent from active scheduler listings. In job 20823 those ranks'
coarse/leaf kernels were slower, and other ranks spent longer in RCCL waiting
for them. The verified stale workers were terminated on October 4; a clean
same-node rerun is recorded separately. This establishes actual contention,
not a hypothesized ROCm-stack effect. Keep the affected historical records
visible as contaminated diagnostics, not release performance evidence.

The first matched complete-model check is now available in
[the full-model results](../kimi-k3-full-model-current/README.md): 33.635 s dense
versus 33.951 s LoD at B8/32K. This is essentially a tie, not the fixture's
approximately 9% win. It uses real ProLong tokens and the resident full K3
weights, with final cache construction included.

### Construction group tuning with allocator reuse

One resident fixture, before/after four-layer controls, matched global update
cadence and final completion fences (job 20822, node 3):

| Group size (layers) | 32K prefill (s) | 64K prefill (s) |
|--:|--:|--:|
| 4, before | 3.614 | 8.794 |
| 1 | 3.987 | 9.585 |
| 8 | 3.582 | 8.798 |
| 12 | 3.585 | 8.792 |
| 4, after | 3.627 | 8.798 |

Source: `dcp8-prefill-reuse-group-tune.json`. All loaded-binary audits passed.
Eight/twelve layers do not produce a material 64K gain; one layer is worse.
The simpler four-layer construction group remains the candidate default.

After terminating node 2's three verified orphan workers, the fresh four-layer
serial/reuse check measured **3.518/8.675 s** at 32K/64K
(`dcp8-prefill-reuse-clean-node2.json`, job 20827, audited). This resolves most
of the earlier 5.500/13.016 s spread as resource contention, not a difference
in the intended algorithm or loaded kernel revision. A fresh dense control
on that cleaned node is recorded separately.

The matched clean-node dense control is 4.024/11.037 s
(`dcp8-dense-clean-node2.json`, job 20830), versus the above LoD 3.518/8.675 s:
**1.144x at 32K and 1.272x at 64K**. Both loaded-path audits passed. This is
fixture-only evidence, not a full-model speedup.

Node 3 also had verified orphan workers from our old jobs 19892, 19902, 20243,
and 20266. They were terminated in job 20834 after checking each process's
checkout and stdout-log ownership. Earlier node-3 measurements above are
therefore contaminated diagnostics too; the clean node-2 pair is the current
fixture comparison to quote.

Distributing eight construction layers across eight ranks was retested with
the improved allocator policy and before/after controls, not the older
reclamation-heavy policy (job 20826, node 3):

| Variant | 32K prefill (s) | 64K prefill (s) |
|:--|--:|--:|
| Serial four-layer construction, before | 3.623 | 8.803 |
| Eight-rank distributed construction | 3.554 | 8.664 |
| Serial four-layer construction, after | 3.655 | 8.795 |

Source: `dcp8-prefill-reuse-distributed-build.json`; all loaded-binary audits
passed and fixture token IDs matched. About 2--3% at 32K and 1.5% at 64K does
not justify making the separate communicator and distributed builder the
default. The experiment remains opt-in.

The native AITER local branch was checked against the previous CK branch
and an FP32 causal reference, including output and LSE (three GPU tests,
job 20825). Its resident-fixture timing was 3.539/8.798 s versus before/after
controls of 3.605/8.793 and 3.637/8.798 s at 32K/64K
(`dcp8-prefill-reuse-native-local.json`, job 20829). It produces no 64K gain
and remains opt-in. Those node-3 timings precede the orphan cleanup.

### Fixed-shape graph replay experiments

On clean node 3, the 24-MLA fixture was tested with piecewise graphs around
eager attention/cache operations. Both modes captured the same 16,384 and
16,392 token descriptors and exact decode-row sizes. One exact-shape warmup
and one measured pass include final cache construction; TP8/DCP8, eight
requests, 16K scheduler chunks, and four-layer construction groups are retained.

| Context | Dense with graphs (s) | LoD with graphs (s) | Dense / LoD |
|--:|--:|--:|--:|
| 32K | 3.998 | 3.540 | 1.129x |
| 64K | 10.957 | 8.661 | 1.265x |

Sources: `dcp8-fixed-prefill-graph-dense.json` (20841) and
`dcp8-fixed-prefill-graph.json` (20840). All eight binary/path audits pass;
each large prefill descriptor has 25 graph segments and 24 eager attention
breaks. Fixture IDs match across modes. Relative to the prior clean pair
(4.024/3.518 s and 11.037/8.675 s), graphs add no material benefit. These are
dummy-fixture timings, not trained-model quality or full-model speedups.

The separate `benchmarks.kimi_k3_update_graph` experiment captures only a
fixed four-layer, 16K-overflow append/merge stage, including copies into and
out of stable graph storage. It reconstructs centroids from real trained
ProLong leaf membership, not random data; the next overflow cyclically reuses
those captured records. This is a GPU-stage diagnostic, not a natural model
sequence or full cache-construction benchmark.

`trained-fixed-update-graph-check.json` (20847) reports 1.662 ms eager versus
1.713 ms replay including copies. Its strict ownership check fails, but two
ordinary eager executions also disagree (88 ownership entries of 65,536).
It must not be treated as a passed graph-correctness check. With benchmark-only
stable tie ordering, `trained-fixed-update-graph-stable-ties.json` (20850)
passes exact ownership/count/state comparison, including fresh state and
overflow after capture. Eager selection scores repeat exactly. This points
to append tie ordering rather than frozen graph inputs; serving's selection
method is unchanged. The stable-sort diagnostic takes 1.974 ms eager versus
1.819 ms replay, and is not compared to production as a speed improvement.
The graph helper stays under `benchmarks/`, outside serving.

```bash
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_update_graph \
  --input results/kimi-k3-full-model-current/trained-prefill-leaf-input.pt \
  --stable-ties \
  --output results/kimi-k3-mla-stack/trained-fixed-update-graph-stable-ties.json
```

The capture is a local diagnostic, excluded from Git along with weights.
It is generated by the trained-prefill diagnostic described in the full-model
results; its tensors are not a public model or quality result.

### Correct sink scoring and scalar page lookup

The deferred-query path previously used a broadcast shape carrier as the
actual query in the final separate-sink score. The corrected implementation
projects the tiny sink to each head's native D192/V128 space and scores it
with the real query, including all 64 direct-key dimensions. It no longer
materializes the 216 MiB D576 query carrier per 16K/12-head chunk. Sink products
accumulate in FP32, matching the attention score definition rather than
rounding each BF16 product before reduction. Earlier deferred-query LoD
measurements above are historical pre-fix records, not corrected-path evidence.

The BF16 page loop now resolves a 16-token page directory once per page, not
through sixteen duplicate lanes. The trained-input control-before/after test
(`trained-leaf-scalar-directory-controls.json`, 20853) takes 1.395 / 1.405 ms
for the original leaf stage versus 1.291 ms for scalar lookup: **1.084x** for
that leaf stage alone. Every consumed output and LSE is bitwise identical.
Inline and two-level directories, batch/head indexing, partial last pages,
and closed routes pass GPU checks. Unselected per-route scratch is deliberately
unwritten and must not be compared or consumed without the route mask.

Four GPU checks pass in 20855, including a dense FP32 reference for the complete
coarse/refined/local/sink merge with distinct keys for each query head. The
corrected 24-layer fixture, B8 TP8/DCP8, four-layer construction, exact 16K
chunks and global 16K/256 cadence, takes **3.474 s at 32K / 8.467 s at 64K**
(`dcp8-prefill-correct-sink-scalar-directory.json`, 20856). All worker audits
pass. This is 1.159x / 1.303x versus the unchanged clean dense controls
(4.024 / 11.037 s). It is a fixture-only comparison, not trained-model quality
or full-model speed evidence. The full-model corrected path is checked
separately before claiming crossover.

### Shared coarse/route graph experiment (not promoted)

`LOD_KIMI_GRAPH_COARSE=1` captures the fixed 16,384-query coarse/route stage,
including exact top-eight tile refinement. One bounded graph arena is shared
across layers, rather than keeping 24 copies. Query, centroid, count, slot
length and **projection-weight** inputs are refreshed before every replay.
At most four state shapes are captured; other shapes use ordinary attention.
This changes no cadence, routing score, cap or LSE replacement accounting.

An unused stream fork previously made capture finalization crash. The fused
path now forks/joins only its actual work stream. The fresh-input tests vary
both Q/K data and W_UK/W_UV, specifically checking that layer-zero projection
weights cannot become frozen in a shared graph. Smaller-state tests also
exposed a contiguous-value assumption in centroid preparation: latent values
can alias the first 512 channels of a 576-wide key. That preparation kernel
now uses explicit input strides, including the fallback path. The GPU suite
passes 49 tests (20878), including stride-specific, fresh-cache and projection
ordering checks. The focused local CPU suite passes 72 tests (9 GPU skips).

The corrected ordinary/graph/ordinary control with trained geometry
(`trained-coarse-shared-graph-controls.json`, 20864) takes **2.788 / 2.380 /
2.738 ms**, with copies included. Fresh queries and projection weights match
ordinary computation exactly. These are serial single-stage measurements on
reconstructed trained centroids, not full-model quality or latency. The
earlier `trained-coarse-shared-graph.json` (20862) mistakenly labeled replay
as an eager control; it is retained as a correctness diagnostic only. The
corrected benchmark uses genuinely ordinary controls on either side.

The actual 24-layer fixture does **not** gain materially:

| Context | Corrected ordinary LoD (s) | Shared coarse graph (s) |
|--:|--:|--:|
| 32K | 3.474 | 3.466 |
| 64K | 8.467 | 8.522 |

Graph source: `dcp8-prefill-coarse-graph.json` (20863), with all eight audits
passing and three captured 16K-query state shapes (2048, 2896, 3547). Dummy
fixture output IDs match the ordinary run. Capture/replay counters were added
after this process started, so that file records null counters, not zero
replays. No full-model rerun is justified by this negligible fixture effect.

```bash
env LOD_KIMI_GRAPH_COARSE=1 \
  benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack --lengths 32768 65536 \
  --variants reuse_allocator --batch-size 8 --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --kv-cache-memory-bytes 4294967296 \
  --output results/kimi-k3-mla-stack/dcp8-prefill-coarse-graph.json

benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_coarse_graph \
  --input results/kimi-k3-full-model-current/trained-prefill-leaf-input.pt \
  --shared-manager \
  --output results/kimi-k3-mla-stack/trained-coarse-shared-graph-controls.json
```

The captured `.pt` input is a local real-model diagnostic, deliberately not
committed with code or results. It must be regenerated with the documented
trained-input capture diagnostic to reproduce the stage-only test.

### Complete final-cache graph experiment (not promoted)

`LOD_KIMI_GRAPH_FINAL_CACHE=1` captures complete final rank-local DCP cache
construction, including append/merge, page membership and the recent exact
tail. A runtime-wide manager keeps at most four exact shapes. Each arena owns
its update scratch; a caller clearing its ordinary scratch cannot invalidate
a replay. Source records are copied before each replay, and the result is
installed before the arena is reused by another layer group. Unsupported
shapes/policies retain ordinary construction. Global 16K/256 cadences and
spherical assignment are unchanged.

The trained-geometry stage benchmark reconstructs a four-layer, DCP8-owned
64K prefix by cyclically reusing captured records. It is not a new natural
text sequence. In `trained-final-cache-shared-graph.json` (20871), ordinary /
graph / ordinary takes **3.542 / 2.241 / 3.672 ms**, including input copies.
Fresh records, unique complete membership, counts, recent tail and centroid
sums pass the independent checks. BF16 accumulation is compared with FP32
leaf sums, rather than insisting on one arbitrary ordering of tied appends.

The actual fixture again shows no material total-prefill benefit:

| Context | Corrected ordinary LoD (s) | Final-cache graph (s) |
|--:|--:|--:|
| 32K | 3.474 | 3.480 |
| 64K | 8.467 | 8.483 |

Source: `dcp8-prefill-final-cache-graph.json` (20872). All eight audits pass;
each worker captures the local 4096/8192-record shapes, reports 192 replays
and zero fallbacks. Fixture token IDs match. This demonstrates working cache
replay but does not justify a full-model retiming or a default change.

```bash
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_cache_graph \
  --input results/kimi-k3-full-model-current/trained-prefill-leaf-input.pt \
  --shared-manager \
  --output results/kimi-k3-mla-stack/trained-final-cache-shared-graph.json

env LOD_KIMI_GRAPH_FINAL_CACHE=1 \
  benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack --lengths 32768 65536 \
  --variants reuse_allocator --batch-size 8 --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --kv-cache-memory-bytes 4294967296 \
  --output results/kimi-k3-mla-stack/dcp8-prefill-final-cache-graph.json
```

### Overlapping independent leaf projection

Leaf K/V projection depends on the archive and layer weights, not on the
selected routes. `LOD_KIMI_OVERLAP_LEAF_PROJECTION=1` schedules only those
GEMMs before the local/coarse completion waits. Expert packing, leaf attention
and refinement still wait, preserving their scratch lifetimes. Projection
uses its existing separate workspace; routes, cap and attention math do not
change. Other projection modes and multi-head-group layouts retain their
original ordering.

One resident-engine ordinary / candidate / ordinary control (20874,
`dcp8-prefill-overlap-projection-controls.json`) measures:

| Context | Ordinary before (s) | Overlapped projection (s) | Ordinary after (s) |
|--:|--:|--:|--:|
| 32K | 3.475 | 3.441 | 3.474 |
| 64K | 8.492 | 8.302 | 8.482 |

That is approximately 1.010x / 1.022x against the mean controls, not a major
crossover shift. Fixture output IDs match and all worker audits pass. The
opt-in stays separate until a matched full-model check is complete.

```bash
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_owner_tune \
  --checkpoint tests/fixtures/kimi-k3-mla-stack --lengths 32768 65536 \
  --variants reuse_allocator reuse_overlap_projection reuse_allocator \
  --batch-size 8 --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-memory-bytes 4294967296 \
  --output results/kimi-k3-mla-stack/dcp8-prefill-overlap-projection-controls.json
```

### Fixed tile-packing and immutable-weight layouts (not promoted)

Two additional ordinary / candidate / ordinary fixture controls keep the same
model, logical requests, top eight and global cadence:

| Candidate | Context | Ordinary before (s) | Candidate (s) | Ordinary after (s) |
|:--|--:|--:|--:|--:|
| Fixed tile-query ranges | 32K | 3.480 | 3.520 | 3.487 |
| Fixed tile-query ranges | 64K | 8.494 | 8.418 | 8.499 |
| Retain immutable weight layouts per layer | 32K | 3.465 | 3.453 | 3.485 |
| Retain immutable weight layouts per layer | 64K | 8.478 | 8.415 | 8.480 |

Sources: `dcp8-prefill-dense-tile-pack-controls.json` (20881) and
`dcp8-prefill-cached-weights-controls.json` (20883). All worker audits pass
and fixture IDs match. None provides a meaningful crossover improvement;
the first also regresses at 32K. No full-model test or default change is
justified by these results.

`LOD_KIMI_DENSE_TILE_PACK=1` reserves a fixed Q-row range per centroid-score
tile and uses one atomic reservation per tile/query block, rather than one
atomic per route followed by global prefix/block-list construction. Empty
rescoring workgroups exit immediately. It changes organization, not the exact
top-eight result. The trained stage control
`trained-dense-tile-pack.json` (20880) takes 2.753 / 2.649 / 2.714 ms; fresh
queries/keys produce bitwise-identical routes, selected scores and coarse
outputs. Eight direct GPU tests (20879) cover multiple rows/heads, incomplete
tiles, fewer than eight tiles and queries spanning several packing blocks.
The 20880 after-control followed changed-input correctness checks without
restoring its original Q/K. Treat its stage timing as exploratory, not a
matched-control speedup; the complete fixture controls above are valid.

`LOD_KIMI_CACHE_PROJECTION_WEIGHTS=1` retains small immutable flattened
projection weights per source/layer, instead of replacing the last layer's
layout in the runtime-wide scratch dictionary. Source references prevent
address recycling. Local, coarse and leaf names remain separate to avoid
introducing an unordered cross-stream dependency. Graphs with mutable weight
inputs disable this immutable cache. Two CPU/GPU cache tests pass (20882).

Reproduce the fixture controls with the preceding owner-tuner command,
replacing the middle variant with `reuse_dense_tile_pack` or
`reuse_cached_weights` and choosing a separate output. The standalone stage
test is:

```bash
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_route_pack \
  --input results/kimi-k3-full-model-current/trained-prefill-leaf-input.pt \
  --output results/kimi-k3-mla-stack/trained-dense-tile-pack.json
```

### Whole-attention graphs and further tile checks (not promoted)

The fixed projected attention body, including the independent exact-local
stream, route/coarse work, leaf attention and separate-sink merge, captures
successfully. Fresh Q/K, W_UK/W_UV and leaf records match ordinary execution
bitwise. Both paths refresh the same stable input storage, and those copies
are included in timing. The local/sink geometry cyclically reuses captured
trained records; this is not a new natural-text sequence or model-quality run.

The corrected same-input ordinary / graph / ordinary test (20892,
`trained-whole-attention-graph-controls.json`) takes **6.607 / 6.675 /
6.623 ms**. Thus graph capture alone does not speed up this GPU workload.
The earlier 20885 record (`trained-whole-attention-graph.json`) validates
fresh-input correctness, but its after-control used altered input data and
must not be used as a matched timing comparison. The corrected script restores
all original sources before its final control.

```bash
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_attention_graph \
  --input results/kimi-k3-full-model-current/trained-prefill-leaf-input.pt \
  --output results/kimi-k3-mla-stack/trained-whole-attention-graph-controls.json
```

Changing the query tile used for exact centroid-tile rescoring is also slower:

| Query tile | Ordinary 64 before (ms) | Candidate (ms) | Ordinary 64 after (ms) |
|--:|--:|--:|--:|
| 16 | 2.818 | 3.591 | 2.768 |
| 32 | 2.748 | 2.839 | 2.717 |
| 128 | 2.824 | 3.074 | 2.767 |

Sources: `trained-refine16-controls.json` (20891),
`trained-refine32-controls.json` (20889), and
`trained-refine128-controls.json` (20890). All fresh-input results match
bitwise and the before/after controls use restored identical inputs. The
ordinary 64-query tile remains unchanged. Reproduce with the preceding
route-packing stage command and `--candidate refine16`, `refine32` or
`refine128`.

A sequential grouped-leaf-workgroup prototype measured 1.292 / 1.269 /
1.293 / 1.396 ms for 1 / 2 / 4 / 8 query tiles per workgroup
(`trained-grouped-leaf-programs.json`, 20888). Six GPU correctness checks
passed (20887). It provided no meaningful gain and its extra generic kernel
wrapper was removed rather than retained in the serving path. Only 0.993%
of the captured opened routes are singletons, so skipping their redundant
refinement cannot address the main bottleneck in this workload.
