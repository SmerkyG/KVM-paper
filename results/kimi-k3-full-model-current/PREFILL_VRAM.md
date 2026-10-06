# K3 prefill VRAM investigation (October 6)

Current B1 result: **token-sharded ordinary LoD fits 512K and 1020K**, including warmed
prefill and 1,025 decode steps. It shares each rank's one-eighth BF16 archive
with the fixed B1 decode backing, while keeping separate global-centroid
prefill metadata. Peak client Torch allocation is 19.124 / 22.734 GiB/rank,
respectively, excluding daemon weights and driver allocations. Both measured
passes complete and pass the all-rank real-attention/four-update audits. See
[TOKEN_SHARDED_PREFILL.md](TOKEN_SHARDED_PREFILL.md) for the implementation,
results and commands; the earlier replicated failures below are unchanged.

The request-owner cache backing plus six-head projection groups
**fits B8/256K prefill and one real decode step**. Neither changes top-eight
selection, BF16 leaf data, the 1,024-leaf closure rule, or the global
per-request 16K/256-token update cadences. The earlier compact-projection
experiment did **not** demonstrate a full-model peak allocation saving;
the failed ordinary-DCP capacity probes below are retained as evidence.

## Observed allocation breakdown

The corrected B1/256K decode endpoint was retried on node 2 after the
ordinary multi-length sweep failed during 256K prefill with
`hipErrorLaunchOutOfResources`. The retry completed, using compact projection
during prefill only. Its 1,025-step decode window was 25.374 ms/step and
passed all eight ranks' four-update-per-row audits.

The warmup-only memory observer records unique **backing storage**, not the
sum of logical K/V views. Aliased Kimi values and shared per-layer workspaces
are counted once. It is removed before measured generation and adds no
events or observers to decode graphs. It cannot attribute kernel time or
measure the exact peak inside an attention/MoE kernel.

Before the final 16K prefill chunk, with 245,760 global tokens processed:

| Rank-0 category | GiB |
|:--|--:|
| Persistent DCP-sharded semantic cache | 0.939 |
| Replicated prefill shadow, 24 MLA layers | 7.367 |
| Decode scratch | 0.560 |
| Shared prefill scratch | 2.645 |
| Total live Torch allocation, including other allocations | 15.911 |
| Torch allocator reservation | 29.596 |
| Physically free device memory at this boundary | 0.004 |

These are different scopes, **not additive rows**. Daemon-owned weights and
driver/private kernel scratch are excluded from the Torch categories.
This shows very little launch headroom at that boundary, but does not prove
the cause of the earlier asynchronous launch error.

The archive in this observation belongs to the one currently prefilling
request. Completed requests use the fixed DCP-sharded decode pools; it would
be wrong to multiply this replicated-shadow observation by eight for the
ordinary serial-row scheduler. Persistent cache and reserved decode scratch
do, however, reserve space for every batch row.

Source: [corrected B1/256K endpoint](oct6-lod-b1-decode-power2-correct-dcp-four-updates-remaining.json),
`measurements["262144"].warmup_prefill_batch_audits[0].memory_snapshots[-1]`.

## B8/256K fit test

A fit-only test ran on node 2 with the same frozen dense-control
prompts. It combines the validated compact projection with the already
implemented 64K-slab growth of temporary prefill archives and a 32K allocator
reclamation check (retaining allocations if 4 GiB remains free).
Neither setting changes attention math, model weights or update cadence.
The permanent graph-captured decode pool stays fixed-address and DCP-sharded.

The probe generates two verified prefix tokens from the archived natural
continuation, so all eight requests must survive prefill and exercise one
corrected batched decode step. It runs once with warmup memory accounting;
it is **not** an amortized decode benchmark and produces no serving-speed
claim. It **failed during the first request's prefill**, after the last
observed completed 32K prefix, before reaching any batched decode step.
Consequently that ordinary-DCP probe has no successful B8/256K capacity or
timing result. The later request-owner result is documented below.

| Rank-0 category before the failing third 16K chunk | GiB |
|:--|--:|
| Persistent DCP-sharded semantic cache, capacity for eight rows | 7.510 |
| Replicated prefill shadow, one active row | 2.064 |
| Decode scratch, reserved across 24 MLA layers | 4.482 |
| Shared prefill scratch | 0.955 |
| Total live Torch allocation, including other allocations | 19.517 |
| Torch allocator reservation | 30.197 |
| Physically free device memory at this boundary | 1.813 |

Again these scopes are not additive. The runtime subsequently reported
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, only **86 MB available**, and an
asynchronously surfaced `hipErrorLaunchOutOfResources`. Its reported kernel
was an RCCL communication kernel; this does not identify the initiating
operation. The last successful snapshot and failure are preserved separately
because a dead worker cannot return the normal end-of-warmup audit.

Sources: [failed fit-test artifact](oct6-compact-b8-256k-capacity.json),
[preserved chunk snapshots](oct6-compact-b8-256k-capacity-memory.json).
This is direct evidence of the tight memory headroom, not a claim of a bad
GPU or a node-specific software-stack effect.

The subsequent [shared decode scratch implementation](SHARED_DECODE_SCRATCH.md)
reduces this reservation from 4.482 to 0.380 GiB/rank without changing LoD
math. CPU isolation checks and actual routing/union/Gluon graph-replay
equivalence tests passed. The matched B8/256K retry advanced to a completed
128K prefix of its first request, but still failed before the next chunk
completed and before any batched decode. Enabling ROCm scratch reclamation
in a separate retry did not change that outcome. This saves 4.101 GiB/rank,
but is **not yet a successful B8/256K capacity fix**.

## Ordinary DCP8 B1/512K retry (October 6)

The ordinary two-tier LoD path was retried after shared decode scratch was
implemented. This is **not** the request-owner/per-GPU experiment. It uses
the full trained model, TP8/DCP8/EP8, the archived dense B1/512K ProLong
prompt, BF16 attention cache, and two verified continuation tokens (one real
decode step if prefill completes). Native KV reservation is 1 GiB/rank and
engine capacity is 525,330 tokens. Top-eight routing, 1,024-leaf closure and
global 16K/256-token update cadences are unchanged.

The first retry combines shared decode scratch, compact selected-leaf
projection, 64K growing prefill archives, the expandable PyTorch allocator,
HIP scratch reclamation, and a 32K idle-allocation reclamation interval with
4 GiB headroom. It **failed during prefill after a completed 262,144-token
prefix**, before any decode. HIP/RCCL reported zero physical free memory on
GPU 3. No prefill/decode serving speed is inferred from this capacity probe.

Last completed-prefix snapshot, rank 0:

| Category | GiB |
|:--|--:|
| Persistent DCP-sharded semantic cache | 1.835 |
| Replicated prefill archive, 24 MLA layers | 7.824 |
| Decode scratch, with layer sharing | 0.072 |
| Shared prefill scratch | 2.770 |
| Total live client Torch allocation | 19.086 |
| Torch allocator reservation | 28.812 |
| Physically free device memory at the boundary | 3.018 |

The category rows have different scopes and are not additive. Daemon weights
and driver allocations are excluded from the client Torch counters. The
28.812 minus 19.086 GiB difference is reserved-but-unallocated Torch memory;
it is not automatically guaranteed to be reclaimable at this boundary.
All 24 MLA layers participate in one shared transient decode registry, and
no owner caches, owner rows, or owner transport buffers exist in the snapshot.
Source:
[ordinary shared-workspace 512K fit attempt](oct6-ordinary-lod-shared-expandable-b1-512k-capacity.json).
All 17 compacted rank-0 boundary snapshots are preserved in
[the memory record](oct6-ordinary-lod-shared-expandable-b1-512k-capacity-memory.json)
because failed workers cannot return the final warmup audit.

A bounded retry changes only existing allocator controls: reclaim after every
16K prefill chunk and retain idle blocks only with at least 8 GiB free. It
does not change the sequence-index state-update cadence or attention math.
Source:
[16K-reclamation 512K fit attempt](oct6-ordinary-lod-shared-expandable-reclaim16k-b1-512k-capacity.json).
It also **failed during prefill**, after a completed 278,528-token (272K)
prefix, before decode. The last rank-0 boundary had 21.647 GiB live client
Torch allocation, 30.838 GiB reservation, 0.979 GiB physically free, and a
10.261 GiB replicated archive. The subsequent HIP/RCCL resource error
reported 1,732 MB physically free on GPU 0; free memory at the earlier
snapshot and at the error are different observations. The larger archive
includes directory/capacity overhead, not just token payload.
All 18 compacted boundary snapshots are preserved in
[the retry memory record](oct6-ordinary-lod-shared-expandable-reclaim16k-b1-512k-capacity-memory.json).
Thus neither retry establishes an ordinary 512K/B1 LoD fit, and neither is
an amortized speed benchmark. No attention implementation or default was
changed for these probes.

## Request-owner cache backing

The [captured B8 owner decoder](OWNER_CAPTURED_DECODE.md) avoids distributed
routing/LSE combination and permits the prefill archive to share its already
allocated single-row decode cache. The wider exact prefill tail stays
temporary; installation copies that tail and skips aliased remote storage.
GPU comparisons give identical centroid sums, leaf membership and routed
decode output. This changes storage, not the approximation or update cadence.

At matched 64K/B8, peak client Torch allocation falls from **27.683 to
25.220 GiB/rank**, saving **2.463 GiB/rank**. Prefill remains about 69.2 s
and captured decode about 34.2 ms/batch step. These counters exclude the
resident daemon's weights and driver allocations. Commands, full controls,
audits and caveats are in the linked owner report.

The first backed B8/256K fit probe still ran out of physical device memory
after all eight rows reached an observed 147,456-token prefix. At that
boundary the 7.680 GiB owner cache was shared, with no duplicate full-history
archive, but Torch retained roughly 12 GiB of idle allocation and physically
free memory was only 0.605 GiB. A once-per-prefill-batch idle-allocation
reclamation retry reached 176K on all eight rows but still failed.

Halving the projection head group from twelve to six then **completed
B8/256K prefill and one real decode step**, without allocator reclamation.
GPU tests verify identical fine-attention output/LSE and half-size projection
storage. The full-model capacity pass peaked at **29.814 GiB/rank of client
Torch allocation**, excluding daemon weights and driver allocations. It is
a fit result, not an amortized decode-speed result. The original top-eight,
1,024-leaf closure and global 16K/256-token cadences are unchanged. See the
owner report for the full command and [raw artifact](oct6-owner-backed-head6-b8-256k-capacity.json).

## Reproduction

The memory observer is opt-in and applies only to untimed warmup:

```bash
export LOD_BENCHMARK_PREFILL_MEMORY_AUDIT=1
# With benchmarks.prolong, also pass --report-memory.
# With benchmarks.kimi_k3_prefill_sweep, pass --audit-prefill-batches --report-memory.
```

Each rank returns snapshots with the warmup audit. Rank 0 also prints
`KIMI_PREFILL_MEMORY_CHUNK` lines, preserving evidence if a later chunk kills
a worker before the final audit RPC. The snapshot labels its scope explicitly.
No weight reload, source-fingerprint check, or timing rerun is needed to
inspect these recorded allocations.

The fit-only overrides are:

```bash
export LOD_KIMI_COMPACT_SELECTED_PROJECTION=1
export LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK=65536
export LOD_KIMI_PREFILL_RECLAIM_INTERVAL=32768
export LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1
export LOD_KIMI_PREFILL_MIN_FREE_GIB=4
```

These are not promoted to the production default. Keep compilation artifacts
on local disk through `benchmarks/run_kimi_k3_v10_direct.sh`.
