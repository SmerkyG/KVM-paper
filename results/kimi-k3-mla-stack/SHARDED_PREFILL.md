# K3 BF16 prefill archives: sharded storage experiment

This is **opt-in development work on `lod-k3`**, not a production/default change.
The goal is to remove the replicated prefill archive that prevents the trained
model from fitting 512K/1020K B1 contexts, without changing centroid construction,
global top-eight routing, the 1,024-leaf opening cap, or final decode construction.

## Calculation and storage

Every rank still constructs the same global centroids with the original
cross-layer updater. A rank stores only chronological positions `rank::8` in
its persistent BF16 prefill archive. The sink stays separate and the exact
current/recent tail is retained. Cadences remain 16K **global sequence tokens**
per request during prefill and 256 during decode.

Two consumers are implemented:

- **Distributed fine attention:** exchange queries and selected global slots,
  attend to disjoint local leaves, combine their LSEs and reduce-scatter outputs.
  Global coarse contributions are removed only once. This is correct but slower.
- **Transient reconstruction:** retain only a sharded persistent archive and an
  index-only global page directory. Gather the current layer's raw BF16 records
  into a shared temporary workspace, overlapping coarse/local attention. The
  existing projected leaf kernels then run with their original local query
  heads. No per-chunk query exchange or partial-output reduction is necessary.
  There is never a persistent reconstructed archive for each of the 24 layers.

Transient reconstruction is a **memory/latency tradeoff**, not an asymptotic
communication optimization: gathering covered history every fixed 16K chunk adds
quadratic total KV communication as prompt length grows. The attention calculation
and selected exact leaves are unchanged. Do not describe this experiment as a
proof of subquadratic total prefill, or assume it is faster at arbitrary lengths.

After prefill, the original rank-local DCP decode builder consumes the already
sharded history. It must not stride it a second time. These changes do not change
decode's global sequence cadence or turn it into independent per-rank routing.

## Validated short-context timings

Eight MI325X GPUs, TP8/DCP8, 24-MLA-layer attention-only fixture, B1, 16K scheduler
chunks, one exact-shape warmup and one measured pass. Prefill includes final cache
construction before the first token. Runs are isolated on their nodes. The
fixture uses deterministic synthetic token IDs and dummy weights: these are
software/speed checks, **not full-model speed or quality results**.

| Context | Replicated archive | Distributed leaves | Transient workspace | Workspace overhead |
|--:|--:|--:|--:|--:|
| 32K | 0.4075 s | 0.5942 s | 0.4251 s | +4.3% |
| 64K | 0.9732 s | 1.5483 s | 1.0269 s | +5.5% |
| 128K | — | — | 2.5518 s | — |

Sources:

- [Replicated control](oct5-replicated-prefill-b1-control.json).
- [Distributed leaves with tiled LSE/stream collectives](oct5-sharded-stream-collective-prefill-b1.json).
- [Transient workspace](oct5-sharded-workspace-prefill-b1.json).

All workers passed the loaded-binary audit for exact global top-eight routing.
The generated tokens matched the replicated control at both shared lengths.
This does not replace a trained-model quality evaluation.

The first long-context transient implementation exposed an allocator problem:
each layer's separate communication stream retained fresh gather buffers.
At 512K the fixture had 19.52 GiB peak live Torch allocation but **177.29 GiB
reserved**. It then failed 1020K warmup with HSA out-of-resources, reporting no
free device memory. Its incomplete artifact is
`oct5-sharded-workspace-long-prefill-b1.json`; do not use it as a validated sweep.

The corrected implementation uses **one shared communication stream and shared
input/gather/reconstruction workspaces across layers**, growing the scratch
buffers geometrically instead of reallocating them every 16K. The original
measured passes were allowed to finish; no existing measurement was cancelled.
The intermediate shared-buffer run completed 512K but rejected the short final
1020K chunk. It is not a completed sweep. The latest candidate fixes that final
boundary and completes both long contexts:

| Context | Original replicated prefill | Sharded-workspace prefill | Peak live Torch allocation, replicated / sharded | Peak reserved, replicated / sharded |
|--:|--:|--:|--:|--:|
| 512K | 13.7255 s | 16.2028 s | 31.05 / 21.14 GiB | 60.13 / 40.83 GiB |
| 1020K | 36.0332 s | 41.6359 s | 47.60 / 26.48 GiB | 71.01 / 51.85 GiB |

Sources: [original replicated long control](oct5-replicated-long-prefill-b1-memory.json)
for 512K, [matched final-chunk control](oct5-replicated-matched-final-control-b1.json)
for 1020K, and [completed geometric-workspace candidate](oct5-sharded-geometric-workspace-b1.json).
The 512K control reports the warmed measured pass's peak; the other entries
conservatively report the maximum of warmup and measurement. These are **fixture allocations**,
not trained-model or cache-only VRAM. All ranks passed the loaded-binary audit.
Both generated tokens matched at these two lengths. Sharding adds 18.0% / 15.5%
prefill time and reduces peak live allocation by 31.9% / 44.4%, respectively.

The same direct final DCP conversion is now enabled for both archive layouts,
including a final chunk shorter than 16K. The matched control additionally checks
a 45,059-token prompt whose final chunk is not a multiple of 256: replicated
prefill takes 0.6219 s, sharded prefill 0.6650 s (+6.9%), with both generated tokens
matching. The original 45,059-token
control matched the first generated token but differed on the next decode token;
its old final-construction path is not the matched comparison. The original
1020K control also used that old final path (35.3653 s, 60.40 GiB peak allocation)
and is superseded by the matched control for this ablation.

The long-context candidate uses `HSA_NO_SCRATCH_RECLAIM=0`, also used by long
trained-model controls. The gather/reconstruction buffers are shared, so 512K
reserved allocation no longer grows to the failed version's 177.29 GiB.

A trained-model 512K/1020K probe (job `21043`, node 4) used the resident weight
daemon and real ProLong traces. It **failed during 512K warmup**, at a processed
global prefix of 376,832 tokens, with HSA out-of-resources / HIP "too many
resources requested for launch" and reported zero free device memory. It
produced no validated long-context timing. The attention-only fixture fit does
not establish that the trained model fits. The distributed consumer below is
being revisited specifically to avoid the whole-history reconstruction buffers.

## Current distributed-leaf revisit

The distributed consumer now reuses query, route, LSE and output collective
scratch across all layers, including its tiled FP32 LSE combination. It retains
the current projected leaf tiles, exact fused top-eight route/coarse selector,
global centroid assignments and counts, cap, sink separation and final decode
handoff. Unlike temporary reconstruction, it never gathers the covered history.
The per-layer projection maps are gathered once and cached; queries and selected
slots are exchanged for each prefill chunk. Only the rank-owned leaves are
attended to, and a head-owner reduce-scatter combines weighted fine outputs.

Focused tests: **22 passed on GPU**, with **311 passed / 45 skipped** in the
broader CPU regression subset. This includes reusable collective buffers,
B1/B2 batch/head ordering, empty shards, and numerical leaf/LSE equivalence.

The allocation-matched comparisons completed: [B1](oct5-storage-b1-v4/comparison.json)
and [B8](oct5-storage-b8-v4/comparison.json). Each node ran the four variants
sequentially with the same engine capacity and native cache reservation (3 GiB
for B1, 4 GiB for B8). All workers passed the loaded-binary audit. These remain
dummy-weight fixture tests, not trained-model speed or quality evidence.

| Cohort / context | Dense prefill (s) | Replicated LoD (s) | Distributed leaves (s) | Temporary reconstruction (s) |
|--:|--:|--:|--:|--:|
| B1 / 64K | 1.385 | 0.971 | 1.565 | 1.068 |
| B1 / 128K | 4.370 | 2.345 | 3.796 | 2.652 |
| B1 / 512K | 57.229 | 13.736 | 20.138 | 16.551 |
| B8 / 64K | 10.945 | 7.798 | 12.432 | 8.432 |
| B8 / 128K | 34.573 | 18.738 | 30.090 | 20.958 |

Maximum per-rank **peak allocated / peak reserved Torch GiB**, taking the
maximum over warmup and the measured pass. Not cache-only bytes or all-process
physical VRAM. B1 is sized for 512K throughout; B8 for 128K throughout.

| Cohort / context | Dense | Replicated LoD | Distributed leaves | Temporary reconstruction |
|--:|--:|--:|--:|--:|
| B1 / 64K | 7.40 / 7.81 | 12.11 / 14.99 | 14.85 / 17.53 | 10.83 / 12.79 |
| B1 / 128K | 7.89 / 8.57 | 14.84 / 20.22 | 16.09 / 19.85 | 12.24 / 15.89 |
| B1 / 512K | 8.01 / 9.32 | 29.71 / 53.23 | 21.61 / 41.26 | 19.26 / 38.90 |
| B8 / 64K | 8.74 / 10.49 | 17.60 / 21.80 | 20.34 / 23.46 | 16.32 / 19.60 |
| B8 / 128K | 9.24 / 10.78 | 20.33 / 26.75 | 21.58 / 25.79 | 17.73 / 22.42 |

At 512K/B1, distributed leaves save **27.2% peak live allocation** versus
replicated LoD but cost **46.6% more prefill time**. They are still 2.84× faster
than dense *on this fixture*. At 64K/128K, the large fixed query/output exchange
workspaces outweigh the archive savings. Temporary reconstruction is better in
both fixture speed and peak allocation at every measured point, but retains
its whole-history communication cost. No default change is warranted from
these results.

Both generated tokens match replicated LoD at all B1 lengths. Every first
token matches for B8, but some second tokens differ in both sharded consumers.
That is recorded explicitly in the comparison artifacts; no B8 decode or
trained-model quality equivalence is claimed. The B8 scheduler uses a total
16K budget and allows completed requests to leave before the others finish:
these are eight-request prefill-cohort latencies, not constant-B8 decode or
eight simultaneously retained unfinished archives.

A trained-model 512K fit probe on node 4 used the resident weights and real
ProLong trace (`21053-kimi-full-distributed-leaves-b1-512k-v4`). It **failed in
warmup at a 376,832-token processed prefix** with HSA out-of-resources in RCCL,
reporting zero free memory on GPU 2 and 380 MB on GPU 0. This is the same
logical boundary as the reconstruction probe, despite using an engine sized
for 512K rather than 1020K. No completed timing or peak-memory artifact resulted.
The failed job was canceled to clean up surviving workers; the resident weights
daemon was left running. Failure record:
[distributed-leaf trained probe](../kimi-k3-full-model-current/oct5-distributed-leaf-capacity-failure.json).

This establishes that sharding the persistent archive alone is insufficient
for this trained-model configuration. The common remaining transients/cache
allocations need a memory audit; the identical failure boundary is evidence
to investigate those, not proof of a specific allocation or kernel defect.

## October 5: shared centroid budget and request owners

These are fresh **24-layer attention-stack fixture** probes using the current
fused route/coarse kernels, not full trained-K3 or quality measurements. Each
point has one exact-shape warmup and one measured pass. Peak memory is the
maximum per-worker Torch allocation over both passes, including weights,
native KV reservation, LoD state and temporary workspaces. It excludes another
process's allocations and is not cache-only memory. All completed artifacts
below passed their loaded-worker top-eight kernel audits.

### One eighth of the centroids per DCP rank

`LOD_KIMI_DCP_SHARED_PREFILL=1` assigns each rank one eighth of the original
total centroid budget, constructed from its owned chronological slice. It
exchanges the small centroid summaries, routes each rank's twelve query heads
against the complete summary table, selects **eight global regions**, and
distributes their exact leaf work to the owners. It does not select eight
regions per rank. All eight owners' fine fields are combined by LSE before
the selected coarse contributions are replaced once.

This changes clustering relative to the monolithic global builder; it is not
the storage-only variant described above. It retains the global per-request
16K prefill / 256 decode cadences, the total state schedule, the 1,024-leaf
opening cap, and exact current/recent/sink fields. The current prototype is
BF16 two-tier with aligned 16K initial-prefill chunks, including one-token
decode rows in mixed scheduler steps. Arbitrary chat continuation is not yet
implemented for this experiment.

The first revisit used an all-96-head fine workspace. Bounding projection and
route scratch to twelve heads at a time reduced 128K peak allocation from
13.60 to **9.93 GiB**, with 5.5% more candidate prefill time. Routes and scores
are unchanged. Both generated tokens matched the replicated-LoD B1 control
at every measured length, before and after the scratch bound.

| B1 context | Dense prefill (s) | Replicated LoD (s) | Shared-budget LoD, bounded scratch (s) | Peak GiB: dense / replicated / shared |
|--:|--:|--:|--:|--:|
| 16K | 0.209 | 0.153 | 0.181 | 6.97 / 5.22 / 7.81 |
| 32K | 0.507 | 0.409 | 0.624 | 6.97 / 9.14 / 9.43 |
| 64K | 1.383 | 0.975 | 1.651 | 7.33 / 10.61 / 9.63 |
| 128K | 4.371 | 2.348 | 4.128 | 7.83 / 13.35 / 9.93 |

The engines in this B1 comparison are sized for 128K, with the same 3 GiB
native reservation. Sources: [matched initial comparison](oct5-shared-budget-b1-v1/comparison.json)
and [bounded-scratch candidate](oct5-shared-budget-b1-head12-v2/comparison.json).
The latter is still **76% slower than replicated LoD at 128K** and has more
peak allocation than dense. It is not a speed-first winner and is not promoted.

The B8 smoke now completes at 16K/32K, taking 1.611/5.169 s. Its engine is sized
for 32K, so its peak memory must not be compared with the 128K-capacity controls.
Source: [mixed-row smoke](oct5-shared-budget-b8-smoke-v5/comparison.json).
The matched 128K-capacity B8 comparison also completed:

| B8 context | Dense prefill (s) | Replicated LoD (s) | Shared-budget, bounded scratch (s) | Peak GiB: dense / replicated / shared |
|--:|--:|--:|--:|--:|
| 64K | 10.951 | 7.785 | 13.379 | 8.74 / 17.59 / 17.02 |
| 128K | 34.570 | 18.714 | 33.459 | 9.24 / 20.33 / 17.49 |

Sources: [dense](oct5-shared-budget-b8-v1/full.json),
[replicated LoD](oct5-shared-budget-b8-v1/replicated.json),
[shared-budget LoD](oct5-shared-budget-b8-head12-v6/comparison.json).
Native reservation is 4 GiB in all three engines. The first token matches the
replicated-LoD control for every row, but some second tokens differ. Thus this
is prefill timing evidence, not decode/quality equivalence. Shared-budget LoD
does not recover the original speed advantage; the owner layout below remains
the better speed-first candidate.

The smoke exposed two real decode handoff bugs, now fixed: maximum-capacity
metadata was passed to a live-row shape validator, and the leaf-count cap
reducer assumed six virtual MLA head tiles had six separate physical caches.
The latter read outside the cache when a nonzero request row was selected.
Routing now honors batch/head strides, including stride-zero head aliases;
three focused GPU reducer tests passed. These fixes do not change the intended
algorithm or expand/copy the KV cache.

### One GPU per request row: speed-first fallback

The request-owner runner gives each of eight GPUs a **complete 96-head
attention-stack fixture** and one request. It excludes MoE/FFN and the transfers
needed to integrate request-owned attention with TP8/EP8 K3. It replicates the
fixture's attention weights (about 10.44 GiB per GPU versus about 1.95 GiB in
the TP8 fixture). Therefore absolute peaks across these two layouts are not
a cache-only comparison.

All owners are released simultaneously after their warmups; the reported
cohort wall latency includes release and final drain. Each owner has a 16K
chunk budget, so aggregate concurrency is **128K**, not the normal TP8/DCP8
runner's 16K aggregate scheduler budget. This is explicitly a layout probe,
not a demonstrated full-model speedup or an equal-budget comparison between
layouts. Dense versus LoD **within the owner layout** is matched: identical
weights, engine capacity, native 4 GiB reservation and release protocol.

The compact-scratch variant projects/refines twelve heads at a time and
LSE-reduces the eight fine routes before assembling the all-head output.
It keeps all eight regions and all their eligible leaves, the same centroids,
cap and update cadence. Early BF16 output reduction can introduce extra
rounding; this has not replaced a trained-model quality evaluation. All
generated fixture tokens matched the previous owner implementation.

| Context | Owner dense (s) | Owner LoD original scratch (s) | Owner LoD compact scratch (s) | Dense / compact LoD |
|--:|--:|--:|--:|--:|
| 16K | 0.654 | 0.701 | 0.701 | 0.93x |
| 32K | 2.047 | 1.984 | 2.038 | 1.00x |
| 64K | 6.822 | 5.216 | 5.357 | 1.27x |
| 128K | 24.644 | 13.598 | 13.912 | 1.77x |

| Context | Owner dense peak GiB | Owner LoD original peak GiB | Owner LoD compact peak GiB |
|--:|--:|--:|--:|
| 16K | 18.74 | 20.95 | 20.95 |
| 32K | 20.44 | 28.90 | 25.01 |
| 64K | 23.07 | 33.22 | 25.61 |
| 128K | 27.14 | 34.04 | 26.61 |

Sources: [owner dense](oct5-eight-owner-dense-current.json),
[owner LoD original scratch](oct5-eight-owner-lod-current.json),
[owner LoD compact scratch](oct5-eight-owner-lod-compact-scratch.json).
At 128K the scratch change saves **21.8% peak allocation** for 2.3% extra
LoD latency; LoD now has slightly less peak allocation than its matched dense
owner control. At 64K it remains 11% above dense. This is the strongest
speed/memory candidate in this revisit, but it does not establish trained-K3
512K/1020K fit or the cost of Q/output transfers.

Run the owner probe on eight visible GPUs in the prepared K3 image userspace:

```bash
env -u LOD_KIMI_DCP_SHARED_PREFILL -u LOD_KIMI_DCP_LOCAL_PREFILL \
  -u LOD_KIMI_DCP_SHARDED_LEAVES -u LOD_KIMI_DCP_PREFILL_WORKSPACE \
  LOD_KIMI_SUBTILE64=score LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  TRITON_CACHE_AUTOTUNING=1 HSA_NO_SCRATCH_RECLAIM=0 \
  benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_request_owners \
  --checkpoint tests/fixtures/kimi-k3-mla-stack --mode two-tier \
  --lengths 16384 32768 65536 131072 --owners 8 \
  --kv-cache-memory-bytes 4294967296 --tile-refine --direct-leaf-result \
  --fine-scratch-heads 12 --report-memory \
  --output results/kimi-k3-mla-stack/owner-compact-reproduction.json
```

Omit `--fine-scratch-heads 12` for the original LoD scratch. For the matched
dense control, use `--mode full` and omit all three LoD kernel/scratch options
(`--tile-refine`, `--direct-leaf-result`, `--fine-scratch-heads`). Keep lengths,
native reservation, owners, memory reporting and release protocol unchanged.

The matched shared-budget driver resets inherited `LOD_KIMI_*` experiments:

```bash
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_sharded_prefill_compare \
  --lengths 16384 32768 65536 131072 --batch-size 1 \
  --kv-cache-memory-bytes 3221225472 --variants full replicated shared \
  --output-dir results/kimi-k3-mla-stack/shared-budget-reproduction
```

No candidate here changes production defaults; all work remains on `lod-k3`.

Validation for this revisit: **267 passed / 45 skipped** on the host CPU
regression subset, and **34 passed** in the complete focused suite on an
otherwise idle node 2. GPU coverage includes all three cap/stride reducers,
metadata-only directory equivalence, head-grouped and sharded fine attention,
archive interleaving and B1/B2 LSE combination. The node-4 standalone numerical
probes stalled/crashed at test-input generation and were not counted as
passing. No benchmark or resident weights daemon was canceled for those test
retries.

## Memory expectation (not an empirical peak)

### October 6: B1 pool-backed archive and bounded projection

The token-sharded path now reuses the already allocated B1 DCP decode
leaf backing during prefill. It keeps a separate global-centroid directory:
prefill's centroid IDs are not the rank-local IDs rebuilt for decode. Initial
metadata construction allocates no second chronological record array; owned
prefix/tail records are copied into the fixed backing, and later 16K updates
append directly there. Final handoff packs the authoritative rank-owned
records before installing the ordinary DCP decode cache. Sink ownership,
BF16 K/V aliasing, global counts, top-eight/cap policy and global 16K/256
cadences are unchanged. Multi-request storage remains separate.

Distributed fine attention also bounds projection to the native TP head
group (12 heads on full TP8 K3), instead of reserving projected scratch for
up to all 96 query heads. All 96 heads still receive their exact selected
leaves, with LSE-weighted partial outputs returned to their head owners.
This mode does not reconstruct or gather the covered whole history.

Validation: 113 host checks passed, with 14 GPU-only skips; the focused
GPU regression suite passed all 85 tests, including shared-scratch graph
replay. The new actual-kernel storage test covers all eight ownership ranks,
including append, authoritative exact tail, sink, counts and K/V aliasing.
The full TP8/DCP8 attention-stack integration completed:

| Context, B1 fixture | Replicated prefill (s) | Token-sharded prefill (s) | Peak live Torch GiB, replicated / sharded |
|--:|--:|--:|--:|
| 64K | 0.938 | 1.679 | 9.945 / 9.623 |
| 128K | 2.169 | 4.047 | 12.500 / 10.086 |

Both generated tokens match at both lengths; loaded-kernel audits pass on
all eight ranks. The peaks include warmup and measurement, and exclude any
weight daemon (the fixture uses dummy weights). Sources:
[replicated fixture](oct6-pool-backed-sharded-b1/replicated.json),
[token-sharded fixture](oct6-pool-backed-sharded-b1/distributed-r2.json).
The first integrated attempt caught an overly strict capacity check that
added global decode headroom to an already rank-local-sized fixed arena;
the corrected path and regression test use the fixed pool's capacity.
The full-model 512K run now completes warmup and measured generation:
84.443 s prefill versus the existing dense control's 118.730 s, followed by
26.393 ms/step over 1,025 measured decode steps. All eight ranks and all 24
MLA layers pass the four-update and loaded-attention audits. Peak client
Torch allocation is 19.124 GiB/rank, excluding daemon weights and driver
allocations. The 1020K run also completes: 178.122 s prefill versus 344.493 s
dense (1.934x), and 26.327 ms/decode step versus 27.378 ms dense, with all
audits passing. Its peak client allocation is 22.734 GiB/rank. See the full-model
[token-sharded report](../kimi-k3-full-model-current/TOKEN_SHARDED_PREFILL.md).

Enable `LOD_KIMI_DCP_SHARDED_LEAVES=1` with B1, two-tier BF16, TP8/DCP8,
and unit token interleave. Leave `LOD_KIMI_DCP_PREFILL_WORKSPACE` unset.
B1 backing reuse and native-head projection bounds are automatic inside
this opt-in path; no request-owner mode or weight changes are required.

For 24 MLA layers with one 576-dimensional BF16 record per position, the raw
persistent prefill archive alone is:

| Context | Replicated per rank | Sharded per rank | Raw archive reduction |
|--:|--:|--:|--:|
| 512K | 13.500 GiB | 1.688 GiB | 11.812 GiB |
| 1020K | 26.895 GiB | 3.362 GiB | 23.533 GiB |

This excludes state, page directories, native KV reservation, projections,
collective buffers, exact tails and allocator slack. The transient consumer adds
only the current layer's reconstructed field and gather workspace, reused across
layers. Measure **peak** memory, not just retained allocation after decode begins.
The fixture runner's `--report-memory` records peak Torch allocation/reservation
and device free memory after generation, outside the timed region. Torch's peak
counter does not measure IPC weight-daemon or other processes' allocations.

## Tests and reproduction

Focused GPU checks: **18 passed** (`tests/test_kimi_sharded_prefill.py`). They cover
all-rank ownership including the sink, retained exact tails, K/V prefix aliasing,
global counts, geometric scratch reuse, parameter restoration, distributed leaf/LSE equivalence, exact
interleaving for ragged shard lengths and B2, and metadata-only page construction.
The index-only directory stores one placeholder record, never full history.
CPU regression checks: **277 passed, 45 skipped** across the sharded archive,
K3 kernels, decode panel, prefill rotation/collectives and vLLM configuration tests.
GPU-required tests are skipped in that host environment.

Run from the repository root in the prepared K3 v10 image userspace on eight
GPUs. No cluster runner is needed:

```bash
env \
  LOD_KIMI_DCP_SHARDED_LEAVES=1 \
  LOD_KIMI_DCP_PREFILL_WORKSPACE=1 \
  LOD_KIMI_SUBTILE64=score \
  LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_TILE_REFINE=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  TRITON_CACHE_AUTOTUNING=1 HSA_NO_SCRATCH_RECLAIM=0 \
  benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint tests/fixtures/kimi-k3-mla-stack --mode two-tier \
  --lengths 45059 524288 1044480 --batch-size 1 --decode-tokens 2 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --kv-cache-memory-bytes 1073741824 --report-memory \
  --output results/kimi-k3-mla-stack/sharded-workspace-reproduction.json
```

Remove both `LOD_KIMI_DCP_*` variables for the original replicated control.
Keep only `LOD_KIMI_DCP_SHARDED_LEAVES=1` for distributed fine attention.
The prototype currently requires aligned projected prefill chunks (with a short
final chunk allowed), DCP unit interleave, two-tier BF16 and K3's 512+64 geometry;
unsupported paths fail explicitly. Two generated tokens do **not** provide
amortized decode throughput.

The new allocation-matched comparison can be reproduced without cluster-run:

```bash
# Run each variant sequentially on one otherwise idle eight-GPU node.
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_sharded_prefill_compare \
  --lengths 65536 131072 524288 --batch-size 1 \
  --kv-cache-memory-bytes 3221225472 \
  --output-dir results/kimi-k3-mla-stack/storage-b1-reproduction

# Run this on a separate idle node, or after the preceding command finishes.
benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_sharded_prefill_compare \
  --lengths 65536 131072 --batch-size 8 \
  --kv-cache-memory-bytes 4294967296 \
  --output-dir results/kimi-k3-mla-stack/storage-b8-reproduction
```

The driver explicitly sets the current projected-leaf configuration and clears
inherited `LOD_KIMI_*` experiment flags. It verifies completion and loaded-kernel
audits, records generated-ID agreement with replicated LoD, and writes
`comparison.json` with maximum per-rank peak allocation/reservation, including
warmup. The scheduler still uses a total 16K token budget, not true B8×16K
prefill; B8 here is the queued request cohort. Startup/JIT is outside the warmed
prefill measurement.
