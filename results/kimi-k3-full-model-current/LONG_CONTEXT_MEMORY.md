# Fitting 1020K B1 and B8 two-tier LoD

Target: full trained K3, eight MI325X GPUs, BF16 attention storage, complete
top-eight leaf refinement, unchanged global 16K/256-token cadences. No
chronological-token quantization, fewer selected leaves, or fixture timing
is used to claim a memory/speed success.

## Storage lower bound when retaining all archived positions

Each archived K3 position is one BF16 576-channel record (512 latent + 64
direct-key channels). Values alias the latent prefix; expanded per-head K/V
are temporary, not a second persistent history. Across 24 MLA layers:

| 1020K layout | Raw latent history per GPU, approximately |
|:--|--:|
| B1 token-sharded over DCP8 | 3.36 GiB |
| B8, one full request owned by each GPU | 26.90 GiB |

This excludes centroid sums/means, page directories, native cache, KDA state,
graphs, projection/construction workspace, driver allocations and weights.
The daemon reports about 200.9 GiB/rank of exported resident tensors; its
larger allocation accounting must not be mistaken for KV storage. Its loader
already calls `empty_cache()` after export, so doing that again is not a
new proposed fix.

The read-only process inventory on node 4 finds **212.305–213.180 GiB/rank**
of daemon VRAM residency versus 200.888 GiB/rank of exported tensors, with no
additional orphaned GPU clients. The remaining residency is not explained
by this inventory; do not attribute it to specific scratch or fragmentation
without another measurement. Container `ps` PIDs differ from the host PIDs
reported by `rocm-smi --showpids`.

## Current exact-storage measures

1. Keep the existing token-sharded B1 archive for 512K and 1020K. Preserve
   global routing/counts and merge exact distributed LSE partials correctly.
2. Reuse each B8 owner's fixed decode archive during prefill. The parent DCP
   pools are already restricted to a small unused backing; this is not a new
   saving. Keep shared cross-layer decode scratch and shared prefill workspace.
3. Reserve only 1 GiB for native attention in fresh LoD sweeps, versus the
   previous 3 GiB short-owner reservation. Native exact-prefix/tail semantics
   and update cadences remain unchanged; preemption counters must stay zero.
4. Bound temporary per-head leaf projections. Existing geometry reduces
   head groups with context length rather than retaining projections for all
   96 heads. The million-token B8 capacity attempt uses two-head groups;
   four-, six-, and twelve-head grouping have the same selected-set math.
   The initial 512K attempt retained normal grouping and failed in warmup.
   A separate four-head, bounded-MoE retry runs before claiming support, so
   a later 1020K allocation failure cannot erase a completed 512K result.
5. Use expandable client allocations, local compilation caches, and remove
   verified orphaned clients before timing. Do not rematerialize daemon weights
   or assume that unused allocator reservations are necessary live KV memory.
6. Size the temporary update score matrix to the **actual overflow**, rounded
   to 256 tokens, not the 16K capacity of its small max-sim vectors. The old
   layer-batched owner decode retained 7.239 GiB of construction storage at
   a 256K request capacity; the fixed version retains 1.148 GiB. This changes
   allocation only: GPU tests retain exact scores, indices, and cache state.
7. Share the owner-prefill construction workspace across serial MLA layers
   on their current stream. Each layer retains its own complete centroid/leaf
   cache; prepared keys refresh when the source pointers change. Delete the
   shared prefill scratch at the completed-prefix decode handoff. The two-layer
   GPU test uses different keys per layer and matches independent construction
   bitwise. This avoids 24 persistent copies of a near-512-MiB score matrix at
   million-token capacity. No asynchronous DCP construction uses this scope.

The million-token B8 attempt additionally uses two-head temporary projection
groups and the existing tokenwise native MoE wrapper in 4K slices. Attention
scheduler slices remain 8×2K, while centroid updates remain global 16K/256.
These bounds do not drop tokens or change the attention approximation.
ROCm launch scratch reclamation is permitted in the new LoD runner; the failed
short B8 retry with reclamation disabled reached a reported 130 MB free during
an RCCL launch. That failure is logged, not counted as a timing.

### Trained million-token startup failures

The permanent-cache fixture is not enough to certify serving. A real trained
1020K/B8 engine with 8K MoE slices failed during vLLM's dummy forward:
AITER's per-expert MoE output requested **896 MiB** with only 88 MiB free on
rank 0. Reducing just the MoE slice to 4K completed the forward but failed in
the subsequent dummy sampler's RCCL allocation of **6 MiB**. These are startup
failures, not measured generation results:
[8K slice](oct7-lod-b8-million-initial-allocation.json),
[4K slice](oct7-lod-b8-million-initial-allocation-moe4096.json).

A narrowly scoped startup fix now releases unused PyTorch allocator blocks
between that dummy forward and the dummy sampler, only in request-owner K3
engines with an explicit native-KV budget. RCCL's direct allocations cannot
ask PyTorch to release its cached temporary blocks. The hook restores its
profile-only flag even if initialization fails and does not run in serving,
normal sampling, or graph replay. Its focused tests check call ordering,
explicit-budget gating, idempotence and exception cleanup. The trained 4K-slice
[allocation retry](oct7-lod-b8-million-allocation-moe4096-profile-cleanup.json)
now **completes initialization and native model graph capture** on all ranks.
Rank 0 holds 32.843 GiB of live client allocations, including 30.120 GiB of
owner semantic cache, and has 2.023 GiB physically free before releasing its
unused cached profiling blocks. These are startup readings, not measured
generation peaks. The subsequent full-length warmup fails on a 48-MiB KDA
projection allocation with no free device memory:
[4K-slice generation attempt](oct7-current-lod-b8-long-1020k-moe4096.json).
Later scheduler/runtime-length errors in that log are consequences of the
worker OOM, not the original failure.

The next exact-storage trial shards the **prefill AttnRes bank** by token over
the TP group, gathering each native mix's output before the normal TP layers.
For the trained 7,168-channel model, the 16K call's eight bank blocks occupy
1.750 GiB/rank unsharded versus 0.219 GiB/rank sharded: a calculated 1.531 GiB
bank saving, not a measured whole-model peak reduction. Eight-token B8 decode
explicitly bypasses this sharding and adds no new collectives to its graph.
CPU tests cover all eight rank slices and native state updates; three GPU
cases match native bank-mix outputs bitwise. The attention cache is unchanged.

That [sharded-bank 4K-MoE retry](oct7-current-lod-b8-long-1020k-bankshard.json)
passes the former KDA allocation but fails in AITER MoE's per-expert temporary
buffers: stage 1 requests 722 MiB and stage 2 requests 448 MiB. The
[2K MoE-slice retry](oct7-current-lod-b8-long-1020k-bankshard-moe2048.json)
also fails with HIP/HSA device-resource errors under memory pressure.
It halves the tokenwise MoE input slice, not the attention scheduler budget,
global update interval, or retained KV history. None of these failed warmups
supplies a timing cell or establishes 1020K/B8 support.

### Exact compact page-directory trial

`LOD_KIMI_COMPACT_PAGE_DIRECTORY=1` reuses the existing inline-plus-hash
directory for two-tier absorbed MLA, instead of a worst-case root array for
every centroid. Up to 128 pages (2,048 leaves) fit inline; longer lists retain
all their pages in a fixed-capacity overflow table with 32-probe lookup.
Thus every <=1,024-leaf centroid eligible for refinement fits inline. The
closed garbage buckets are still represented completely: nothing is evicted
or quantized, and their coarse sums and mass remain unchanged.

A million-token-capacity GPU test stores 70,017 real BF16 latent records,
including 68,000 in one overflow bucket, then appends incrementally. It checks
every source record and every centroid's membership, aliases values to the
latent prefix, preserves backing addresses, and reports no hash overflow.
Directory storage falls from **87,908,356 to 12,582,912 bytes per layer**, a
measured **1.684 GiB/rank** saving across 24 MLA layers. The actual pool-backed
consumer also passes CUDA-graph capture/replay and bitwise eager/replay output
checks for both old and compact indexing. This is a storage/consumer result,
not a trained serving speed claim or an unconditional default change.

The [trained compact-directory attempt](oct7-current-lod-b8-long-1020k-compact-bankshard-moe2048.json)
initializes, but its first warmup still fails with device-resource errors.
The visible exception surfaces at KDA convolution; the asynchronous HIP error
and preceding queue diagnostics do **not** establish that convolution is the
originating fault. A [native-kernel resource probe](oct7-kda-convolution-resource-probe.json)
finds **zero spills** for channel tiles 256, 128 and 64 at 1,536/1,792 channels,
including 16K B1 and 8x2K strided inputs. Tile 256 is faster (0.047–0.056 ms
for the full 16K call versus 0.082–0.096 ms at 128); output and native state
updates match bitwise. Keep the existing convolution tile. These isolated
probe times are not substituted into full-model serving measurements.

The frozen 16K B8 diagnostic with the complete 1020K reservation and 1K MoE
slices stalls under near-full VRAM, then fails with an engine/worker timeout
and HIP/HSA resource error. A synchronous-launch retry narrows the visible
failure to the sharded residual bank's RCCL all-gather with zero free device
memory; neither diagnostic yields a serving timing. Logs and raw snapshots:
[ordinary diagnostic](oct7-lod-b8-million-reservation-short-moe1024.json),
[synchronous diagnostic](oct7-lod-b8-million-reservation-short-sync.json).
The independent 512K full warmup/measurement uses compact indexing, four-head
projection groups, 4K MoE slices and the sharded prefill residual bank.

### Bounding exact-local projection memory

The fine-leaf head-group limit did **not** constrain the exact-local field:
that field expanded all 96 heads' keys and values concurrently. At a 32K
local field, its K, intermediate K, and V buffers total **2,818,572,288 bytes
(2.625 GiB)**. The new opt-in `LOD_KIMI_LOCAL_PREFILL_HEAD_GROUP=16` executes
all six groups on the same stream, copying their complete output/LSE before
reusing the projection buffers. It retains every query head and every local
key, including the direct-key channels and original causal mask.

The GPU test `tests/test_kimi_local_projection_groups.py` compares the old
96-head path with both the first and a changed-query second grouped call.
Attention output and LSE match within numerical tolerance. Projection storage
is **469,762,048 bytes (0.438 GiB)**, saving **2.188 GiB**; full output/LSE are
still assembled normally. This is a workspace measurement, not a speed claim.
The default remains the old all-head local projection until trained capacity
and timing checks justify changing it.

A frozen 32K B8 diagnostic reserved the entire million-token cache with
this grouped-local path, compact directory, sharded bank and 1K MoE slices.
Its untimed per-chunk memory audit reports actual input lengths and storage
groups. The ROCr async scratch threshold is explicitly bounded to 256 MiB;
[AMD documents](https://rocm.docs.amd.com/en/develop/reference/env-variables.html)
this as a reclaim threshold, not a per-token cache limit. Do not assume it
changes the resident daemon's allocations or guarantees a serving fit.
It passed the first 16K state construction but failed under device-resource
pressure after chunk 8; it does not establish generation support:
[raw failed diagnostic](oct7-lod-b8-million-reservation-grouped-local-32k.json).

### Active-state score storage and grouped centroid projection

The reusable construction GEMM workspace now grows in power-of-two buckets
of the **active** centroid width rather than reserving the final state's
width on its first use. Its output is still the same contiguous, unpadded
GEMM, and existing storage is reused until a larger bucket is required.
In the GPU test with a 16K overflow, 2,048 active centroids and 16K reserved
centroids, storage is **64 MiB instead of 512 MiB**, with bitwise-identical
BF16 scores. This saving shrinks as the state grows; it is not a saving in
permanent history or a claim of lower final-length scratch.
Tests: `tests/test_kimi_state_workspace.py`, cluster job 21529 (14 passed
including the local-projection tests). The trained diagnostic with this
change reaches chunk 12, but still fails with RCCL/HSA resource errors and
zero reported free memory on rank 0:
[raw diagnostic](oct7-lod-b8-million-reservation-active-state-32k.json).
The attempted 1K scheduler-row override was rejected before model startup;
the actual capacity trials retain the supported **8×2K** attention schedule.

An additional opt-in, `LOD_KIMI_COARSE_PREFILL_HEAD_GROUP=16`, bounds
expanded centroid K/V projections in the same way. It retains every head's
top-eight selections, output/LSE and the complete projected centroid means
needed for replacement; it does not change the underlying LoD approximation.
Groups explicitly wait for their route/coarse stream before copying results
and reusing workspace. The GPU check passes both original and changed-query/
changed-weight calls, with exact selected sets and numerically equivalent
attention/LSE. BF16 centroid values can differ by rounding when GEMM's output
width changes; these values are not claimed bitwise identical.
At 2,048 centroids, projection storage plus assembled complete values falls
from **168 MiB to 76 MiB**. At 16K centroids the corresponding calculated
saving is **0.719 GiB**, not an observed whole-model peak reduction.
`tests/test_kimi_local_projection_groups.py`: job 21533, four GPU tests passed.

The trained diagnostic combines grouped centroid projection with
eight-head exact-local projection (calculated local projection workspace
0.219 GiB instead of 2.625 GiB at a 32K exact field). Both remain **capacity
opt-ins**, not changes to the normal timing-panel defaults. A complete short
prompt with million-token reservation is only a preflight; full-length
warmup and the measured 1,025-step generation must still complete.
Neither this diagnostic nor untimed audit/compile durations enter the speed
tables. This [32K diagnostic](oct7-lod-b8-million-reservation-grouped-coarse-32k.json)
**completes prefill and one decode step with the full million-token cache
reservation**. Its rank-0 client peak is 34.107 GiB, with zero device free
memory reported after generation; it is still close to the physical limit.
The input-batch audit passes the supported 8×2K schedule. This establishes
that the early construction failures can be avoided, not that the larger
active-state workspaces at 1020K fit.

The next checks should reuse this exact-storage configuration, first with a
longer live prefix at the same reservation and then full-length warmup plus
measured decode. If pressure returns, bound projection groups further or
serialize/reclaim completed **prefill-only** scratch before direct RCCL
allocations. Do not assume the allocator's cached-but-unused blocks are
available to RCCL. The 26.90-GiB raw-history lower bound remains; removing
it would require a separately validated allocator that reclaims leaves from
permanently closed centroids while preserving their complete sums and mass,
or a different validated cache format—not chronological INT4 quantization.

If permanent-history reclamation is needed, the conservative design is:

- In K3's unit-weight append/merge scheme, a centroid's member count only
  increases until request reset. Once it exceeds the existing 1,024-leaf
  opening cap it cannot become eligible for refinement again. Its key sum,
  value sum and full member count must continue updating normally.
- Keep sink records and the entire current exact local field independently.
  Only reclaim a closed centroid's leaves once they are outside that field
  and all readers from the preceding attention/update have completed.
- Use reusable physical latent blocks plus indexed leaf membership, not a
  fixed chronological allocation of `max_model_len` records. The existing
  chronological allocation cannot return scattered records to the allocator.
- Keep **total members** separate from **resident leaves**: zeroing the
  count used for the cap would accidentally make a closed centroid eligible
  again. Retain stable device directory pointers and update indices at the
  existing global-256 boundaries so decode graph replay needs no allocation.
- Measure the closed-leaf fraction before cleanup, then prove selected sets,
  output/LSE and two-generation replay equivalence before claiming a saving.
  Empty post-cleanup statistics cannot justify this change. No leaf is
  reclaimed by the current implementation.

### Trained measurement of the reclamation opportunity

The [128K B8 real-token capacity check](oct7-lod-b8-closed-centroid-fraction-128k.json)
completes all eight prefills and one decode step. It uses the frozen canonical
128K cohort, compact indexing, sharded residual bank, four-head fine
projection, and 4K MoE slices, with a **128K-sized**, not million-token,
reservation. All 24 MLA layers are inspected on each owner while their caches
are still live. The loaded-attention and 8×2K input-batch audits pass.
This is a storage diagnostic, not an amortized decode timing or a new speed
cell. Its untimed warmup includes compilation.

| Owner / request | Archived leaves in >1,024-member centroids | Reclaimable raw BF16 bytes across 24 MLA layers |
|--:|--:|--:|
| 0 | 3.66% | 0.123 GiB |
| 1 | 1.41% | 0.047 GiB |
| 2 | 14.93% | 0.503 GiB |
| 3 | 0.77% | 0.026 GiB |
| 4 | 5.08% | 0.171 GiB |
| 5 | 3.82% | 0.129 GiB |
| 6 | 2.36% | 0.080 GiB |
| 7 | 0.67% | 0.022 GiB |

The pooled fraction is **4.09%**. Each owner has 3,139,560 archived member
records summed across its 24 layers. The bytes above are calculated as the
measured closed-leaf count × 576 channels × 2 bytes; they are not an actual
reduction in allocator usage, nor do they include directory reclamation.
Closed centroids still retain their full summaries. Their largest individual
member counts vary from 1,889 to 9,119 across owners.

This proves that reclamation has a real but uneven opportunity. It does
**not** justify assuming the same fraction at 1020K or promising that this
allocator alone supplies enough headroom on every rank. A production change
must use reusable physical blocks (so closed records actually reduce peak
allocation) and enforce a capacity bound/fail explicitly if the live frontier
exceeds it. The separate 11–12-GiB difference between exported weight storage
and daemon residency is also worth diagnosing, but is not a demonstrated
removable allocation. No weights were reloaded for these checks.

This is a follow-up design, not an implemented memory saving. The
[256K-prefix diagnostic with full 1020K reservation](oct7-lod-b8-million-reservation-grouped-coarse-256k.json)
failed during warmup after roughly 82K global tokens. Its asynchronous
RCCL/HSA callback reported 4,199 MB available on rank 5; that callback alone
does not identify the originating allocation or prove a zero-free-memory
failure. It provides no timing cell. Separate capacity controls test
reclaiming idle allocator blocks before residual-bank collectives and a
smaller native cache reservation; neither changes the LoD history or routing.

The [512-MiB native-cache allocation audit](oct7-lod-b8-million-native512mb-allocation.json)
completes native graph initialization with the full million-token LoD
reservation. vLLM reports capacity for 9,061,121 native tokens, or 8.67
requests at the million-token limit. Rank 0 has 4.094 GiB physically free
after startup. This saves 512 MiB compared with the 1-GiB native reservation,
but startup capacity is not evidence that all eight requests can complete
prefill/decode without preemption. The longer-prefix serving control must
pass before reducing the reproduction runner's native-cache default. That
[longer-prefix control](oct7-lod-b8-million-reservation-native512mb-256k.json)
fails near 96K global tokens during warmup with an HSA resource error and
zero free memory reported on rank 0. The native reservation remains 1 GiB.

The [residual-collective reclamation trial](oct7-lod-b8-million-reservation-collective-reclaim-256k.json)
also fails: MoE stage 2 cannot allocate 112 MiB on ranks 0, 2 and 3, with
34.53 GiB of live PyTorch allocations and zero device free memory. Several
other ranks wait for the failed workers. This is not a demonstrated
collective deadlock or a serving timing. The new reclamation helper and its
flag were removed after this unsuccessful trial; no per-collective allocator
trimming is promoted to the retained implementation.

The fresh dense 1020K/B8 control also ran out of device resources during its
first warmup, after successful engine initialization with a 31 GiB native
cache. It supplies no speed cell. Dense 512K/B8 did complete both passes:
939.950 s prefill / 61.130 ms per decode step. This control's capacity failure
does not establish whether the smaller native cache used by LoD will fit.

`--report-memory` now uses the existing alias-aware storage accountant outside
generation timing, including parent/owner construction workspaces, and can
record global centroid counts/local leaf counts. This identifies, without evicting anything, how much raw history is
owned by centroids already closed by the 1,024-leaf cap. A possible next step
would reclaim/reuse those exact leaf blocks while continuing to update their
complete coarse sums and mass. This requires a stable paged allocator and
proof that closure is monotonic; it is not enabled or claimed as a saving yet.

## Validation status

B1 completes full-length warmup **and** measured 1,025-step decode at 512K
and 1020K with the already validated sharded algorithm. At 1020K, rank 0's
measured client allocation peak is 22.409 GiB and its post-generation free
space is 4.385 GiB (weights and other process allocations are outside that
client peak). See [current timings](CURRENT_TIMINGS.md) and
[raw B1 long sweep](oct7-current-lod-b1-long.json).

B8 completed both generations through 256K with bounded decode workspace;
that unshared-prefill configuration is preserved in [its log](OCT7_B8_UNSHARED.md).
The new shared-prefill-scratch canonical sweep has now also completed every
point through 512K, including all audited updates. It measures 330.056 s
prefill / 32.571 ms decode there. The initial separate 512K warmup failed
with device-resource errors; its bounded-workspace, compact-directory,
sharded-bank retry **completes both passes**: 849.782 s prefill / 33.163 ms
decode, versus dense's 939.950 s / 61.130 ms. Its rank-0 measured client peak
is 26.837 GiB, with 4.141 GiB physically free after generation.
1020K B8 generation still needs to fit and complete. No failed warmup
is treated as a prefill timing.
Its measured 16K client allocation peak is **21.595 GiB**, compared with
**27.419 GiB** in the superseded engine at the same 256K reservation: 5.823 GiB
less peak client memory. Both are measured-pass allocator peaks, not total
device usage or an attribution of exclusive kernel time. The new canonical
32K point completes in 33.860 s prefill / 31.058 ms decode.
A separate 24-MLA-layer fixture inspects the million-token backing
allocation only; it does not include trained weights/MoE and is not a serving
benchmark. Its legacy scheduler cannot stand in for the trained K3 owner
prefill scheduler, so no fixture prefill speed is promoted.

The storage-only million-token audit reserves **30.120 GiB/rank** of owner
semantic cache (including directories and centroids), versus the 26.90 GiB
raw-history lower bound. Total client allocations are 33.636 GiB/rank, with
a 35.337 GiB setup peak, in a fixture without trained MoE weights or prefill.
Thus the permanent cache is modestly above its unavoidable raw latent size;
this is not evidence that the full-model runtime fits. Source:
[`oct7-million-token-storage-accounting.json`](../kimi-k3-mla-stack/oct7-million-token-storage-accounting.json).

Membership statistics must be sampled **before** request cleanup. The
`--report-memory` warmup-only observer attempts to capture them at pool cleanup and
removes itself before the measured pass. Statistics read after cleanup are
zero and cannot establish a reclamation opportunity. It handles the runtime
attached to either the model runner or its model state. The fresh B1 engines
started before this diagnostic was complete; their timings remain valid,
but cleared/absent membership statistics are not used for memory decisions.
The new B8 engine also reports empty membership dictionaries; do not treat
these as evidence that no centroids are closed or that their leaves can be
reclaimed. Allocation peaks and the alias-aware storage inventory are available
independently of this missing diagnostic.

Earlier B8/256K one-pass capacity success did not survive a second generation.
Require full-length warmup **and** the subsequent 1,025-step measured decode
before marking a context as supported by the current panel.
