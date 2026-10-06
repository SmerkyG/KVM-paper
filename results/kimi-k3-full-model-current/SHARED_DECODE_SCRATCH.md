# Shared K3 decode workspace (October 6)

K3's ordinary sequential vLLM path now shares **transient decode scratch**
across MLA layers. It does not share KV caches, centroids, final outputs,
returned LSEs, epochs/stamps, or incremental mask state. Attention math,
top-eight selection, the 1,024-leaf closure rule, global sequence-index
cadences, and model weights are unchanged.

## Implementation and safety

One registry per model runner owns temporary tensors, keyed by their name,
exact shape, dtype and device. Different TP/DCP geometries receive separate
storage. The serving runtime reserves them before graph capture; subsequent
layers reuse their fixed addresses without new GPU events, synchronization,
or hot-path allocation. There is no new tuning flag.

Only a reviewed whitelist of fully overwritten intermediates is shared.
New/unrecognized buffers default to private allocation. Outputs and returned
LSEs remain private so native DCP collectives/projections cannot read another
layer's result. The runtime disables sharing for speculative execution and
interleaved microbatches/dual-batch overlap. Other model families retain their
existing private allocation behavior.

Code: `lod_attention/kernels/_decode_scratch.py`,
`new_fused_decode_buffers`, and the vLLM runtime's pool reservation.

## Measured workspace saving

Same B8/256K engine capacity, node 2, full trained model, TP8/DCP8/EP8,
resident weights, frozen ProLong prompts, compact prefill projection and
the same archive growth/reclamation settings as the preceding failed probe.
The warmup observer counts unique backing storage, not repeated tensor views.

| Decode workspace on rank 0 | GiB |
|:--|--:|
| Previous private per-layer allocation | 4.482 |
| Shared workspace plus private layer-specific buffers | 0.380 |
| Saved | 4.101 |

This is **91.5% less decode workspace**, not 91.5% less total model/cache
VRAM. The 7.510 GiB persistent semantic cache is unchanged.
All 24 MLA layers participate in one registry containing 44 geometry-specific
transient tensors; those shared tensors themselves occupy 0.178 GiB.

The initial shared-workspace fit retry advanced past a completed 128K prefix
of its first request, versus 32K previously, but still failed while starting
the next 16K chunk. It did not reach the batched decode barrier. The last
snapshot showed 20.046 GiB live Torch allocation, 30.678 GiB allocator
reservation, and 1.252 GiB physically free; the runtime subsequently reported
zero free memory and an asynchronous resource-allocation error. This remains
**a failed B8/256K fit test**, not a throughput measurement.

Sources: [previous snapshots](oct6-compact-b8-256k-capacity-memory.json),
[shared retry](oct6-shared-decode-scratch-b8-256k-capacity.json),
[preserved retry snapshots](oct6-shared-decode-scratch-b8-256k-capacity-memory.json).

A separate fit-only retry enabled ROCm scratch reclamation
(`HSA_NO_SCRATCH_RECLAIM=0`), keeping all other settings unchanged. It also
failed after a completed 128K prefix of the first request, with 0 MB free
reported by the runtime. See the
[runtime-reclamation probe](oct6-shared-decode-scratch-b8-256k-capacity-reclaim.json)
and [preserved snapshots](oct6-shared-decode-scratch-b8-256k-capacity-reclaim-memory.json).
This is a separate runtime-policy check, not part of the saving attributed to
layer sharing.

## Correctness and speed checks

CPU tests verify exact-name/shape/dtype separation, default-private new
buffers, independent epochs and outputs, and microbatch/speculation guards.
The GPU test uses the actual routing/union/Gluon decoder for two different
MLA layers and two requests, comparing shared and private workspaces through
captured graph replays with changing queries and advancing live tails. Output
and returned-LSE checks passed. The first GPU run passed all five tests. The
focused CPU suite passed 61 tests, with the GPU-only test skipped locally. A
broader Kimi/benchmark/configuration suite passed 329 tests, with 41
GPU-dependent tests skipped locally.

The full-model B8/16K check reuses the corrected panel's archived prompts and
1,026-token natural continuations, its 66,578-token engine capacity and
3 GiB native cache reservation. It warms once and measures once, with four
global-256-token updates per request and no graph-internal timing hooks.
The [first matched attempt](oct6-shared-decode-scratch-b8-16k-decode.json)
completed generation but was rejected by an incorrect benchmark guard that
required identical first-token timestamps. With serial-row chunked prefill,
the first samples are produced at different times, before the synchronized
decode barrier. The sweep now uses the same canonical ProLong timer as the
decode panel: from the latest first sample to the latest last sample. It
still rejects preemption, cached prefixes, split decode waves, incorrect
traces and missing updates. It also saves all per-batch timing validation
fields. Regression tests cover staggered prefill first samples and reject
staggered decode finishes.

The [corrected matched decode check](oct6-shared-decode-scratch-b8-16k-decode-r2.json)
completed successfully on node 4. All eight ranks report
`FULL_DECODE_ONLY`, one shared registry for 24 MLA layers, and the unchanged
top-eight/global-16K/global-256 configuration. Every layer/rank recorded four
catch-up batches and 32 catch-up rows (four for each of eight requests).
There were no preemptions or cached-prefix tokens; last-token spread was zero
and the all-live overlap equaled the entire 38.881 s decode window. Only
5.67 ms of the 55.332 s generation wall time was outside the request-metric
window. The first-token spread was 14.386 s from serial prefill, as expected.

| B8/16K end-to-end decode, TP8/DCP8/EP8 | ms/batch step |
|:--|--:|
| Existing dense control | 31.649 |
| Previous corrected LoD, private scratch | 37.928 |
| LoD, shared scratch | 37.933 |

The 0.012% difference between the LoD runs is negligible in this one-pass
check. Sharing preserves latency; it does **not** fix the existing decode
slowdown versus dense attention. Source for the private LoD and dense
controls: [corrected decode panel](DECODE_POWER2.md). No dense controls or
full context sweep were rerun for this allocation-only change.

## Row-per-GPU decode at B8

This is a plausible next comparison, not a measured improvement. One GPU
could own one request's entire LoD cache and run all 96 attention heads,
eliminating distributed top-eight selection and the attention/LSE merge.
Native TP still needs to distribute queries and return the appropriate
head-output shards. Local routing/attention remains, with less request
parallelism per GPU and a larger cache to search for that row.

The earlier owner decode measurement was eager (157.650 ms at B8/256K),
whereas the dense control was graph-captured (47.405 ms). It cannot settle
whether a graph-captured owner layout is faster than ordinary DCP. Any new
comparison should capture both paths, replay identical ProLong traces and
include the same four global-256-token updates per request. Existing owner
results and limitations are in [the owner experiment](REQUEST_OWNER_PREFILL.md).

## Reproduction

From the repository root with the K3 v10 environment installed:

```bash
bash benchmarks/run_kimi_k3_v10_direct.sh -m pytest -q tests/test_decode_scratch.py
```

Ordinary K3 model-runner startup enables safe sharing automatically. Reuse
the resident weights daemon for full-model runs and keep compilation caches
on local disk. Both probes used the full trained weights already staged in
`/tmp`, with no checkpoint reload. The scripts below are local commands and
do not depend on the cluster runner.

### Matched 16K decode check

This command reuses the node-4 resident daemon and frozen corrected-panel
traces; substitute only a resident daemon ID/path on another machine. It
runs one exact-shape warmup and one measured 1,025-step decode window.

```bash
env \
  LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 \
  LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_SUBTILE64=score \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_BENCHMARK_SYNC_PREFILL_CACHE=1 \
  HSA_NO_SCRATCH_RECLAIM=1 \
  LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  TRITON_CACHE_AUTOTUNING=1 \
  VLLM_USE_TRITON_AWQ=1 \
  AITER_CONFIG_FMOE=/home/dan/subusers/agent/KVM-paper-release/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv \
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier \
  --lengths 16384 \
  --max-model-len 66578 \
  --batch-size 8 \
  --decode-tokens 1026 \
  --reference-decode-trace \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --weight-cache-id kimi-k3-shared-int4-v6 \
  --kv-cache-memory-bytes 3221225472 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct6-lod-b8-decode-16k64k-correct-dcp-four-updates.json \
  --report-memory \
  --repeats 1 \
  --output results/kimi-k3-full-model-current/oct6-shared-decode-scratch-b8-16k-decode-r2.json
```

### B8/256K fit-only probe

This command used the node-2 resident daemon and the existing full-model
checkpoint on local disk. It intentionally replays only two verified tokens
and must not be used to quote amortized decode latency.

```bash
env \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  TRITON_CACHE_AUTOTUNING=1 \
  VLLM_USE_TRITON_AWQ=1 \
  HSA_NO_SCRATCH_RECLAIM=1 \
  LOD_BENCHMARK_SYNC_PREFILL_CACHE=1 \
  AITER_CONFIG_FMOE=/home/dan/subusers/agent/KVM-paper-release/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv \
  LOD_KIMI_SUBTILE64=score \
  LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 \
  LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=32768 \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_TILE_REFINE=1 \
  LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  LOD_KIMI_COMPACT_SELECTED_PROJECTION=1 \
  LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK=65536 \
  LOD_BENCHMARK_PREFILL_MEMORY_AUDIT=1 \
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier \
  --lengths 262144 \
  --max-model-len 263178 \
  --batch-size 8 \
  --decode-tokens 2 \
  --reference-decode-trace \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --weight-cache-id kimi-k3-node2-full-int4-v1 \
  --kv-cache-memory-bytes 1073741824 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct4-full-b8-decode-256k512k-four-updates.json \
  --audit-prefill-batches \
  --report-memory \
  --capacity-only \
  --repeats 1 \
  --output results/kimi-k3-full-model-current/oct6-shared-decode-scratch-b8-256k-capacity.json
```

For the separate runtime-policy retry, change only
`HSA_NO_SCRATCH_RECLAIM=1` to `HSA_NO_SCRATCH_RECLAIM=0` and use a distinct
output path. The archived retry failed as described above.
