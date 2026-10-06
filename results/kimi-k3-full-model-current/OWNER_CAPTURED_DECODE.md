# Captured B8 request-owner LoD decode (October 6)

The table below records the one-wave baseline and cache-backing tests.
The latest cached-centroid/32-split follow-up is **30.872 ms/step** at 64K,
versus 31.245 in the preceding pipeline and 34.198 dense (9.7% lower latency).
It passes the same eight-owner graph/four-update audits; see
[cached routing](CACHED_ROUTING.md) and
[the raw result](oct6-cached-means-decode-owner-b8-64k.json).
The same current decode kernels complete **30.134 ms/step at 16K** versus
31.649 dense and **31.938 ms/step at 128K** versus 37.994 dense. The 128K
run uses the documented smaller, pool-backed prefill workspace and allocator
pressure policy after its first attempt ran out of memory. Its measured
prefill is slower than dense (202.915 versus 155.750 s); no prefill speedup
is claimed. All eight owners pass the actual 1,025-replay/four-update and
loaded-geometry audits. See
[the current context panel](CACHED_ROUTING.md#16k--128k-context-follow-up).
The four-wave router improved matched 64K owner decode from
34.148 to **31.403 ms/step**, versus 34.198 ms dense; see
[the router follow-up](ROUTER_DECODE.md). The subsequent parallel-reducer
follow-up measures **31.245 ms/step**, versus 32.536 ms optimized ordinary
DCP8 and 34.198 ms dense. It retains physical B1/64-split attention on each
owner and passes all eight-owner graph/four-update audits. See
[the current pipeline comparison](DECODE_PIPELINE.md) and
[its raw owner result](oct6-parallel-decode-owner-b8-64k.json).

Full trained Kimi K3, eight MI325X GPUs, TP8/EP8. Each GPU owns one of
eight live requests and runs all 96 MLA heads against that request's complete
BF16 latent-plus-direct-key history. Native TP projections, gating, W_O,
MoE, and KDA remain distributed. This is an opt-in development path.

## Validated result

| Context | Dense (ms/batch step) | Ordinary DCP8 LoD | Request-owner LoD | Owner improvement over ordinary LoD |
|--:|--:|--:|--:|--:|
| 16K | 31.649 | 37.933 | 33.699 | 11.2% lower latency |
| 64K | 34.198 | 38.209 | 34.148 | 10.6% lower latency |

The owner remains **6.5% slower than dense at 16K**. This is not a dense
speedup claim. Ordinary LoD uses the shared-scratch rerun; the earlier
corrected private-scratch value was 37.928 ms and is effectively identical.
Dense controls were reused, not rerun solely because development code changed.
At 64K the owner is effectively tied with dense (0.15% lower latency in one
pass, not evidence of a meaningful dense speedup). Its prefill took 69.227 s,
versus 70.504 s in the archived dense control and 72.585 s in the corrected
ordinary-LoD decode benchmark. The ordinary **prefill-only** panel is separate.

Sources:

- [Captured owner, 16K](oct6-owner-captured-b8-16k-decode-r2.json).
- [Captured owner, 64K](oct6-owner-captured-b8-64k-decode.json).
- [Shared-scratch ordinary LoD, 16K](oct6-shared-decode-scratch-b8-16k-decode-r2.json).
- [Corrected ordinary LoD, 64K](oct6-lod-b8-decode-16k64k-correct-dcp-four-updates.json).
- [Archived dense 16K/32K/64K controls](oct4-full-b8-decode-16k64k-four-updates.json).

The owner run records **1,025 actual eight-row CUDA-graph replays on every
rank**, not just instantiated graph handles. Every one of the 24 MLA layers
on every rank reports 1,025 decoded tokens and **four global-256 catch-ups**.
All eight requests are live throughout the timed first-to-last-token window;
first/last token spread, preemptions and prefix-cache hits are all zero.
Prompt and teacher-forced continuation tokens match the archived ProLong
controls. One full-shape warmup precedes one measured pass. Updates are
included in these end-to-end decode times.

Owner prefill took 15.783 s at 16K. Its scheduler processes eight 2K row
slices (16K total token budget plus eight decode-reserve tokens), while
semantic cache construction still occurs at each request's **global 16K**
boundary. That scheduling change is explicit: this prefill observation is
not substituted into the ordinary serial-row prefill panel.

## Implementation and safeguards

- Native replicated queries, or a fixed-buffer head gather when unavailable,
  supply the owner's 96 heads. A fused projection absorbs W_UK into the query.
- The ordinary single-GPU head-tiled LoD decoder runs unchanged. It needs no
  distributed routing selection or LSE merge.
- A fused W_UV projection and one graph-safe reduce-scatter return each
  native TP rank's twelve-head output slice. This is an output transpose,
  not a mixture of partial attention masses.
- Cache installation and cross-layer batched catch-ups run before graph
  replay. Static decode buffers and approved transient scratch sharing are
  reused; layer outputs and cache state remain private.
- Top eight, the 1,024-leaf closure rule, separate sink and global per-request
  16K prefill / 256 decode cadences are unchanged.
- B8/TP8 and eight distinct live owners are required. Prefix reuse and
  arbitrary mixed-size serving batches are not supported by this prototype.

The former [eager 256K owner result](REQUEST_OWNER_PREFILL.md#owner-decode-extension-october-5)
is not equivalent to this captured path. It is retained as history, not used
as the current owner decode baseline.

## Reproduction

Run inside the pinned K3 v10 environment, with the resident transformed
weights daemon already available. The launcher puts compilation artifacts
on local `/tmp/dan-agent` storage. Set `CHECKPOINT` to locally staged weights,
and `WEIGHT_CACHE_ID` to the resident daemon entry. No checkpoint reload or
conversion is performed between these runs.

```bash
env LOD_KIMI_OWNER_QUERY_CHUNK=2048 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_SUBTILE64=score LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_CHUNK_TILE_PACK=1 LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_BENCHMARK_SYNC_PREFILL_CACHE=1 \
  HSA_NO_SCRATCH_RECLAIM=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  TRITON_CACHE_AUTOTUNING=1 VLLM_USE_TRITON_AWQ=1 \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint "$CHECKPOINT" --weight-cache-id "$WEIGHT_CACHE_ID" \
  --mode two-tier --owner-tp-mla --batch-size 8 \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --lengths 16384 --max-model-len 66578 --decode-tokens 1026 \
  --kv-cache-memory-bytes 3221225472 --reference-decode-trace \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct6-lod-b8-decode-16k64k-correct-dcp-four-updates.json \
  --audit-prefill-batches --report-memory --repeats 1 \
  --output results/kimi-k3-full-model-current/owner-captured-reproduction.json
```

The initial full-model attempt used the old owner preset (eight 16K row
slices, 131K total scheduler budget) and failed in startup MoE profiling,
which requested a 12 GiB temporary allocation. It produced **no timing**.
The corrected preset above honors the explicit 2K row slices and leaves
both semantic update cadences unchanged. GPU projection/order, cache-update,
and graph/eager equality tests passed before the full-model measurement.

## Prefill cache reuse

`LOD_KIMI_OWNER_POOL_BACKED_PREFILL=1` makes the completed remote centroids,
directory and leaf archive use the already allocated owner decode pool.
The exact prefill tail stays temporary; only that tail is copied on handoff.
This avoids allocating the whole chronological history a second time without
quantizing sequential tokens, dropping leaves, or changing the algorithm.
The numerical GPU check constructs two complete 16K blocks with both backing
layouts. It verifies identical centroid sums, counts, recent tail, sink and
leaf memberships, then identical routed decode output and unchanged fixed
cache pointers. Physical page IDs can differ because parallel allocation
does not prescribe their order; membership is checked through each centroid's
actual directory rather than comparing arbitrary physical IDs.
CPU tests additionally check exception cleanup and backing-storage accounting.

The matched **64K, B8** full-model follow-up completed on node 2 with the
same frozen prompts, 3 GiB native cache, 16K aggregate scheduler budget,
graphs and four updates as the node-4 unbacked owner control. One warmed
measurement, maximum across the eight ranks:

| Quantity | Separate prefill archive | Reuse owner decode cache |
|:--|--:|--:|
| Peak Torch allocation (GiB/rank) | 27.683 | 25.220 |
| Peak Torch reservation (GiB/rank) | 30.051 | 27.285 |
| Prefill (s/batch) | 69.227 | 69.174 |
| Decode (ms/batch step) | 34.148 | 34.183 |

This saves **2.463 GiB/rank of peak client Torch allocation** (8.9%) without
a meaningful speed change. These counters **exclude daemon-owned weights**
and non-Torch/driver allocations; they are not total model VRAM or a cache
compression ratio. Both runs pass all 1,025 actual replay/four-update audits.
Source: [cache-backed 64K run](oct6-owner-backed-b8-64k-decode-memory.json).

The first cache-backed **B8/256K** fit-only attempt still failed with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES` during prefill, after the last observed
completed 147,456-token prefix of **all eight** requests. The rank-0 boundary
had 7.680 GiB owner cache, 3.490 GiB shared prefill scratch and just 0.605 GiB
physically free; Torch allocated 21.037 GiB and reserved 33.062 GiB.
The error reported zero free memory on ranks 4--6. The reported RCCL kernel
does not prove which asynchronous operation first exhausted resources.
This is **not** a completed 256K result or a decode timing.
Source: [first backed 256K capacity attempt](oct6-owner-backed-b8-256k-capacity.json).

`LOD_KIMI_OWNER_PRESSURE_CHECK=1` now performs the established 4 GiB
idle-allocator headroom check **once before each prefill batch**, instead of
once in every attention layer. It never runs inside captured decode and
never releases live semantic caches or workspaces. Reclamation cost, if
triggered, is included in normal prefill timing. This retry passed the earlier
failure point but still exhausted physical memory after the last observed
180,224-token prefix of all eight requests. It is not a 256K fit result.
Source: [pressure-check capacity retry](oct6-owner-backed-pressure-b8-256k-capacity.json).

The next storage-only test sets `LOD_KIMI_OWNER_PREFILL_HEAD_GROUP=6`, halving
the projected K/V arena relative to twelve heads at a time. All 96 heads,
their top-eight routes and leaf data are still consumed, in more groups.
GPU tests give bitwise-identical output and LSE for both chronological and
compact selected-leaf projection, and verify that the projection arena
halves. The full-model **B8/256K fit test completed on node 4**, with the
pressure check disabled: all eight rows prefilled together through 262,144
tokens, installed their caches, and generated two archived continuation
tokens each (one actual decode step after prefill). All 24 MLA layers per
rank were attached, with top-eight, cap 1,024 and the original global update
cadences. No allocator reclamation calls were recorded.

Peak client Torch allocation was **29.814 GiB/rank**, and peak allocator
reservation **34.211 GiB/rank**. These exclude daemon-owned weights and
driver allocations. The one capacity pass took 343.553 s including untimed
shape warmup/JIT; this is **not** a warmed prefill speed or an amortized
decode measurement. It does not demonstrate four subsequent decode updates
at 256K. The matched 16K/64K speed comparisons above remain the validated
decode timings. Source: [successful six-head 256K fit test](oct6-owner-backed-head6-b8-256k-capacity.json).

To reproduce the capacity test, add the following environment settings to
the command above:

```bash
export LOD_KIMI_OWNER_POOL_BACKED_PREFILL=1
export LOD_KIMI_OWNER_PREFILL_HEAD_GROUP=6
export LOD_BENCHMARK_PREFILL_MEMORY_AUDIT=1
```

Use `--lengths 262144 --max-model-len 263178 --decode-tokens 2 --capacity-only`,
`--kv-cache-memory-bytes 1073741824`, and the
`oct4-full-b8-decode-256k512k-four-updates.json` reference baseline instead of
the shorter-length baseline in the command above. Keep the 2K row slices
and 16K aggregate scheduler budget; no larger token budget is assumed.

### 256K warmed comparison follow-up

Dense full attention already fits B8/256K with DCP8 and BF16 MLA records;
there is no chronological KV quantization. Its archived, warmed full-model
control takes **365.571 s/batch prefill** and **47.405 ms/batch decode step**.
It uses the improved dense Gluon decoder, the same frozen ProLong prompt
and continuation cohort, and 1,025 synchronized eight-row decode steps.
Source: [dense B8/256K control](oct4-full-b8-decode-256k512k-four-updates.json).

The six-head, cache-backed owner follow-up completed its entire 256K/B8
warmup, including all 1,026 continuation tokens, in 381.869 s. However,
the subsequent measured generation failed on its first prefill batch with
`HSA_STATUS_ERROR_OUT_OF_RESOURCES`, reporting just 34 MB physically free
on GPU 1. The failure artifact has **no completed measured timing**; warmup
elapsed time is not substituted for prefill or decode speed.
Source: [first warmed comparison attempt](oct6-owner-backed-head6-b8-256k-warm-decode.json).

The retry enables HIP scratch reclamation (`HSA_NO_SCRATCH_RECLAIM=0`, as
in the dense control) and the existing once-per-prefill-batch idle allocator
pressure check (`LOD_KIMI_OWNER_PRESSURE_CHECK=1`, 4 GiB reserve). Neither
change frees live semantic caches or changes top-eight, leaf closure, head
coverage, or global update boundaries. Any reclamation during measurement
is included in prefill latency. This retry also completed its entire warmup,
in 541.995 s, but failed at the first measured prefill batch with zero free
physical memory on ranks 4, 6 and 7. There is again no completed measured
timing. Source:
[pressure/scratch-reclamation retry](oct6-owner-backed-head6-pressure-b8-256k-warm-decode.json).

The owner handoff had bypassed the ordinary cleanup of each parent's
construction-only state-update workspaces. `prepare_owner_decode_batch`
now drops those parent construction-workspace references once after
installing the completed prefix. Unlike the ordinary cleanup, it deliberately
does not clear runtime-wide dictionaries: those may already be shared by
the children's decode catch-up from a prior generation.
The child semantic cache and graph-captured decode scratch are separate and
remain allocated at unchanged addresses; subsequent decode steps do not run
the cleanup. Tests check this separation and that cleanup runs once, only
after installation. The benchmark now also preserves warmup memory and
allocator-policy records before starting measurement, including if that
measurement later fails.

The cleanup follow-up completed full warmup in 380.797 s but still failed
on the measured generation's first prefill batch. This time the traceback
pinpoints an AITER/FlyDSL **MoE stage-two** workspace allocation of 1.75 GiB:
GPU 2 had 516 MiB physically free, 27.76 GiB of live Torch allocations,
6.09 GiB reserved but unallocated, and only 92 MiB in private graph pools.
Thus the unused reservation is not predominantly a CUDA-graph pool.
The post-warmup `empty_cache()` freed only about 0.2 GiB/rank. Cleanup of
construction scratch alone did **not** resolve repeat-generation capacity.
Source: [cleanup follow-up](oct6-owner-backed-head6-cleanup-b8-256k-warm-decode.json).

The expandable-allocator follow-up (`PYTORCH_ALLOC_CONF=expandable_segments:True`)
completed full warmup in 379.685 s. After releasing idle reservations, live
client Torch allocation was 20.969 GiB/rank, reservation 21.496 GiB/rank,
and physical free memory 10.658--11.908 GiB/rank. This substantially improves
post-warmup headroom, but did **not** by itself make the second generation
run: its first prefill batch exhausted HIP/RCCL launch resources, with the
runtime reporting zero free memory on GPU 2. There is no measured speed
point in this artifact either. Source:
[expandable-allocator follow-up](oct6-owner-backed-head6-expandable-b8-256k-warm-decode.json).

A final owner retry combines the expandable allocator with HIP scratch
reclamation, without the slower per-batch pressure check. It completed full
warmup in 380.185 s, then again failed at the measured generation's first
prefill batch. This time the HIP/RCCL launch-resource failure reported
2,148 MB physically free on GPU 2, despite 10.522--11.908 GiB/rank free
immediately before measurement. Scratch reclamation plus the expandable
allocator is therefore **not sufficient** to fix repeated-generation
capacity. The surviving failed workers were stopped; the weights daemon was
left running. No timing point is promoted from this attempt.
Source:
[owner, expandable allocator plus scratch reclamation](oct6-owner-backed-head6-expandable-reclaim-b8-256k-warm-decode.json).
The fresh dense control with the expandable allocator and scratch retention
**completed**, taking **364.752 s/batch prefill** and **48.090 ms/batch decode
step**. All eight ranks passed the full-model/dense-Gluon/no-dummy audit;
the eight requests had no prefix hits or preemptions and stayed live throughout
the 1,025-step measured decode window. Whole-generation time was 414.092 s,
with only 0.048 s outside the prefill/decode metric windows. Source:
[fresh dense control](oct6-full-expandable-b8-256k-warm-decode.json).
These are distinct scratch policies and must be labelled as such; the archived
dense control already uses HIP scratch reclamation. The fresh dense control
differs from it by -0.22% prefill latency and +1.45% decode latency.
Warmup elapsed times above include prefill and continuation and are never
presented as warmed serving speed.

These are layout comparisons, not identical prefill scheduling: dense uses
serial 16K row chunks, while owner LoD uses eight simultaneous 2K slices.
Both have a 16,392-token aggregate scheduler budget. The reused dense
engine reserves 17 GiB/rank of native KV cache and supports 512K prompts;
the owner engine reserves 1 GiB/rank of native KV cache, has a separate
semantic archive, and is sized to 263,178 tokens. Those reservations are
not total model or cache-memory measurements.

## Sharing projected slices between GPUs

This could reduce owner memory only with streaming consumption and bounded
buffers. Gathering every projected slice before attention would reconstruct
the original owner allocation. Expanded K/V contains 192+128 values per
head, whereas the shared latent/direct-key record contains 576 values total.
For all 96 heads that is about 53 times as many values to transmit as the
single latent record, before accounting for selection. Doing fine attention
on the projecting GPUs and exchanging only output/LSE would reduce that
traffic, but restore the distributed attention-combination overhead avoided
by request ownership. The smaller local head group therefore tests the
storage reduction first, without introducing communication or new math.
