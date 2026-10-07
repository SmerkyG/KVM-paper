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
groups and the existing tokenwise native MoE wrapper in 8K slices. Attention
scheduler slices remain 8×2K, while centroid updates remain global 16K/256.
These bounds do not drop tokens or change the attention approximation.
ROCm launch scratch reclamation is permitted in the new LoD runner; the failed
short B8 retry with reclamation disabled reached a reported 130 MB free during
an RCCL launch. That failure is logged, not counted as a timing.

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

Fresh B8 now completes both generations at 16K with bounded decode workspace.
Its longer points, then 512K/1020K capacity and generation, still need to
complete. A separate 24-MLA-layer fixture inspects the million-token backing
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
`--report-memory` warmup-only observer captures them at pool cleanup and
removes itself before the measured pass. Statistics read after cleanup are
zero and cannot establish a reclamation opportunity. It handles the runtime
attached to either the model runner or its model state. The fresh B1 engines
started before this diagnostic was complete; their timings remain valid,
but cleared/absent membership statistics are not used for memory decisions.

Earlier B8/256K one-pass capacity success did not survive a second generation.
Require full-length warmup **and** the subsequent 1,025-step measured decode
before marking a context as supported by the current panel.
