# K3 decode with spill-free centroid routing

This is the preceding router-only baseline. The subsequent
[decode-pipeline optimization](DECODE_PIPELINE.md) retains this router and
improves ordinary 64K decode to 22.025 ms at B1 and 32.536 ms at B8.

October 6, 2026. Full trained Kimi K3, eight MI325X GPUs, TP8/EP8,
65,536-token real ProLong prompts and frozen continuations. Dense uses the
improved Gluon decoder. Both modes use the same resident transformed INT4
**MoE weights**; the LoD attention cache is BF16. No weights are reloaded or
reconverted for these tests.

## Full-model results

All times are **end-to-end ms per batched decode step**, including updates.
One complete shape warmup and one measured pass, 1,026 output tokens / 1,025
timed decode steps, four global per-request 256-token catch-ups in all 24
MLA layers on all eight ranks. Prompt and continuation hashes match the
archived controls. All requests remain live throughout the decode window;
no preemptions, cache hits or finish-time spread occur.

| Execution layout | Dense | Old LoD | Four-wave LoD | LoD latency reduction |
|:--|--:|--:|--:|--:|
| B1 ordinary DCP8 | 21.845 | 25.550 | 23.004 | 10.0% |
| B8 ordinary DCP8 | 34.198 | 38.209 | 35.364 | 7.4% |
| B8 one request's attention per GPU | 34.198 | 34.148 | 31.403 | 8.0% |

The optimized ordinary decoder remains **5.3% / 3.4% slower than dense**
at B1/B8. The owner layout is **8.2% lower latency than dense** (1.089x
dense/LoD). This is a modest full-model speedup, not the router's 5--8x
kernel speedup or the attention-only fixture's larger gain.

Four waves replace one wave for the existing 16-head by 64-centroid,
512-latent + 64-direct-key router. This eliminates its 603 VGPR spills and
1,592 bytes/thread of private scratch. Scoring, candidate selection, top
eight, the 1,024-leaf closure rule, separate sink and all update cadences
are unchanged. The default profile applies to ordinary DCP and owner
decode alike; Qwen, K2 and smaller absorbed-MLA geometries are untouched.

The [isolated validation and fixture report](../kimi-k3-mla-stack/ROUTER_OPTIMIZATION.md)
records bit-identical old/new scores and indices, GPU cache/update/output
checks, and captured 12-layer comparisons. The CPU regression subset passes
378 tests (48 device-only skips); three targeted GPU cases pass separately.
These are numerical equivalence checks, not new NIAH or LongBench scores.

## Matched setup and scope

The B1 engine retains the previous power-of-two panel's **263,186-token
capacity** and 1 GiB native cache. B8 retains **66,578-token capacity** and
3 GiB native cache. Ordinary prefill uses one 16K scheduler slice at a time;
owner prefill uses eight 2K slices. Both retain global 16K semantic updates
and global 256-token decode updates. The owner still uses native TP Q/K/V/O,
MoE and KDA; only attention is request-owned. Its numerical transport and
actual B8 graph-replay audits are preserved.

Measured prefill took 8.520 s for ordinary B1, 68.402 s for ordinary B8,
and 69.152 s for owner B8. Prefill code is unchanged by this optimization;
these observations are not a claim of a new prefill speedup.

Dense controls are reused because dense code/math did not change. B1 and
ordinary B8 follow-ups ran on node 2; the owner ran on node 4, matching its
earlier owner control. Treat nodes as equivalent as requested. No other
model workload was active on their GPUs during measurement; idle resident
weight daemons remained in place. Compilation uses local `/tmp/dan-agent`.

Sources:

- [New ordinary B1](oct6-router-optimized-b1-64k.json).
- [New ordinary B8](oct6-router-optimized-dcp-b8-64k.json).
- [New owner B8](oct6-router-optimized-owner-b8-64k.json).
- [Old ordinary B1](oct6-lod-b1-decode-power2-correct-dcp-four-updates.partial.json).
- [Old ordinary B8](oct6-lod-b8-decode-16k64k-correct-dcp-four-updates.json).
- [Old owner B8](oct6-owner-captured-b8-64k-decode.json).
- [B1 dense](oct4-full-b1-decode-power2-four-updates.json).
- [B8 dense](oct4-full-b8-decode-16k64k-four-updates.json).

## Reproduction

Use the pinned K3 v10 environment with the existing transformed-weight
daemon and local checkpoint. Set `K3_CHECKPOINT` and `K3_WEIGHT_CACHE_ID`
to its local path and cache entry. The daemon already owns the weights;
running these commands must not trigger a cold load/conversion.

```bash
export LOD_KIMI_SUBTILE64=score LOD_KIMI_CHUNK_TILE_PACK=1
export LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_KIMI_SORT_LEAF_ROUTES=1
export LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1
export LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1
export LOD_KIMI_PREFILL_MIN_FREE_GIB=4 LOD_KIMI_TILE_REFINE=1
export LOD_KIMI_DIRECT_LEAF_RESULT=1 LOD_BENCHMARK_SYNC_PREFILL_CACHE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_USE_TRITON_AWQ=1
export TRITON_CACHE_AUTOTUNING=1 HSA_NO_SCRATCH_RECLAIM=1
export AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv"

# Ordinary B8. For B1 use batch=1, capacity=263186, native cache=1073741824,
# and oct6-lod-b1-decode-power2-correct-dcp-four-updates.partial.json as reference.
bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint "$K3_CHECKPOINT" --weight-cache-id "$K3_WEIGHT_CACHE_ID" \
  --mode two-tier --batch-size 8 --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 --lengths 65536 --max-model-len 66578 \
  --decode-tokens 1026 --kv-cache-memory-bytes 3221225472 \
  --reference-decode-trace \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct6-lod-b8-decode-16k64k-correct-dcp-four-updates.json \
  --repeats 1 --output router-dcp-b8-64k.json

# Owner B8: same command, adding --owner-tp-mla --audit-prefill-batches,
# with LOD_KIMI_OWNER_QUERY_CHUNK=2048 and a distinct output filename.
```

Do not compare this to the old approximately 20.7 ms LoD decode results:
those omitted the nonreplicated-Q DCP dispatch and used an unwritten tail.
All before/after controls here include the corrected query and tail paths.
