# Fitting 1020K B1 and B8 two-tier LoD

Target: full trained K3, eight MI325X GPUs, BF16 attention storage, complete
top-eight leaf refinement, unchanged global 16K/256-token cadences. No
chronological-token quantization, fewer selected leaves, or fixture timing
is used to claim a memory/speed success.

## Storage lower bound

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
   The 512K attempt retains the normal grouping and runs in its own engine
   first, so a later 1020K allocation failure cannot erase that result.
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
buffers: stage 1 requests 722 MiB and stage 2 requests 448 MiB. A 2K MoE-slice
retry is in progress. It halves the tokenwise MoE input slice, not the
attention scheduler budget, global update interval, or retained KV history.
Neither failed warmup supplies a timing cell or establishes 1020K/B8 support.

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
point through 256K, including all audited updates. It measures 330.056 s
prefill / 32.571 ms decode there. The separate 512K warmup/measurement is
running; 1020K generation still needs to fit and complete.
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
