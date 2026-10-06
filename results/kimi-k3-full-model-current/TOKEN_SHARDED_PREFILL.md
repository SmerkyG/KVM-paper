# B1 token-sharded BF16 LoD prefill

October 6, `lod-k3`: opt-in storage/communication implementation for the full
trained K3 model, TP8/DCP8/EP8 on eight MI325X GPUs. This is **not** request-owner
LoD. The original global centroids, top-eight selection, 1,024-leaf closure,
separate sink, and global per-request 16K prefill / 256 decode cadences remain.

## Implementation

Each GPU stores only positions `rank::8` of the chronological BF16 latent
archive, reusing its already allocated B1 decode leaf backing. Prefill retains
separate global-centroid metadata; it cannot reuse decode's rank-local centroid
IDs. All ranks still construct the same global summaries and global counts.

The query-head owner computes coarse attention and selects eight global regions.
Queries and selected IDs are exchanged; each token owner attends to its disjoint
leaves. LSE-weighted partial outputs are summed and returned to the head owners.
Selected coarse terms are replaced once, not once per rank. The sink and exact
local tail are evaluated once per query head. Projected fine scratch is limited
to the native twelve-head TP group, not all 96 heads at once.

No whole-history KV gather is used. At handoff, authoritative owned records are
packed before installing the ordinary fixed-address DCP decode state. Values
remain a view of the first 512 dimensions of the 576-dimensional key record.
No token quantization, head-specific clustering, or attention approximation
change was introduced. Multi-request storage is unchanged.

## Trained-model results

One full-shape warmup and one measured pass, resident IPC weights, frozen real
ProLong tokens. Prefill includes final cache installation before the first output.
LoD generates 1,026 outputs: 1,025 timed decode steps and exactly four catch-ups
in each of 24 MLA layers on all eight ranks. The memory observer runs only in
warmup; measurement contains no instrumentation inside the decode graphs.

| Context, B1 | Dense prefill (s) | Sharded LoD prefill (s) | Dense / LoD | Dense decode (ms/step) | LoD decode (ms/step) |
|--:|--:|--:|--:|--:|--:|
| 512K | 118.730 | 84.443 | 1.406x | 24.686 | 26.393 |
| 1020K | 344.493 | 178.122 | 1.934x | 27.378 | 26.327 |

Both lengths complete warmup and measurement, with real attention, no preemptions,
no prefix hits, matched prompt hash, all 1,026 forced outputs verified, and
four measured updates in every layer/rank. 512K prefill is 1.406x faster than
dense but decode is 6.9% slower; 1020K prefill is 1.934x faster and decode is
3.8% faster. Decode still uses the ordinary captured DCP path, not request owners.

Maximum per-rank client Torch allocation/reservation over warmup and measurement:

| Context | Peak live allocation (GiB) | Peak allocator reservation (GiB) |
|--:|--:|--:|
| 512K | 19.124 | 23.018 |
| 1020K | 22.734 | 26.875 |

These are not total physical VRAM or cache-only bytes: resident daemon weights
and driver/private allocations are excluded. The prompt audit confirms 32 full
16K chunks at 512K, and 63 full 16K chunks plus a final 12K chunk at 1020K on
every rank; DCP does not divide the global sequence-index update cadence.

Dense controls reuse the already validated 1,024-step Gluon-decoder panel.
The entire archived 1,025-token output trace is verified, with one extra token
from the same frozen ProLong stream for the LoD four-update protocol. Both
original and extended hashes are saved. These are single-pass speed comparisons,
not quality tests or confidence intervals. No dense control was rerun.

Sources:

- [512K LoD](oct6-token-sharded-pool-backed-b1-512k-decode1025.json), job `21336`, node 2.
- [1020K LoD](oct6-token-sharded-pool-backed-b1-1020k-decode1025.json), job `21337`, node 4.
- [Dense controls](oct4-full-b1-512k1020k-decode1025.json).

The older replicated 512K retries failed at completed 256K/272K prefixes;
their evidence remains in [PREFILL_VRAM.md](PREFILL_VRAM.md). Successful sharded
512K/1020K generation is a new capacity result, not a reinterpretation of those failures.

## Validation

113 host regression tests passed, with 14 GPU-only skips, followed by 85 GPU
regression tests passing, including scratch-sharing graph replay. The archive
tests cover all eight ownership ranks, append, authoritative sink/prefix/tail,
global counts, separate metadata, capacity headroom, and K/V aliasing.
The B1 24-layer dummy fixture matches both generated tokens at 64K and 128K;
its memory/latency results are in [SHARDED_PREFILL.md](../kimi-k3-mla-stack/SHARDED_PREFILL.md).
Fixture agreement and teacher-forced speed runs do not establish free-generation
quality equivalence at 512K/1020K.

## Reproduction

Run from the repo root on the eight GPUs with a resident full K3 weight daemon.
Compilation artifacts stay on local disk through the image launcher. Substitute
the local checkpoint and resident daemon ID for your machine; do not reload the
weights between modes. Do not enable request-owner or reconstruction-workspace
flags. The 512K run used:

```bash
env LOD_KIMI_DCP_SHARDED_LEAVES=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True HSA_NO_SCRATCH_RECLAIM=0 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_SUBTILE64=score LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_TILE_REFINE=1 LOD_KIMI_PREFILL_MIN_FREE_GIB=8 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 LOD_BENCHMARK_SYNC_PREFILL_CACHE=1 \
  LOD_KIMI_CHUNK_TILE_PACK=1 LOD_KIMI_PREFILL_RECLAIM_INTERVAL=16384 \
  LOD_KIMI_COMPACT_SELECTED_PROJECTION=1 LOD_BENCHMARK_PREFILL_MEMORY_AUDIT=1 \
  VLLM_ALLOW_INSECURE_SERIALIZATION=1 TRITON_CACHE_AUTOTUNING=1 \
  VLLM_USE_TRITON_AWQ=1 \
  AITER_CONFIG_FMOE=results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv \
  bash benchmarks/run_kimi_k3_v10_direct.sh -m benchmarks.kimi_k3_prefill_sweep \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --mode two-tier --lengths 524288 --max-model-len 525330 --batch-size 1 \
  --decode-tokens 1026 --reference-decode-trace \
  --tensor-parallel-size 8 --decode-context-parallel-size 8 \
  --weight-cache-id kimi-k3-node2-full-int4-v1 --kv-cache-memory-bytes 1073741824 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --reference-baselines results/kimi-k3-full-model-current/oct4-full-b1-512k1020k-decode1025.json \
  --audit-prefill-batches --report-memory --repeats 1 \
  --output results/kimi-k3-full-model-current/oct6-token-sharded-pool-backed-b1-512k-decode1025.json
```

For 1020K change `--lengths` to `1044480`, `--max-model-len` to `1045522`,
the output name to `...b1-1020k-decode1025.json`, and the node-4 daemon ID to
`kimi-k3-shared-int4-v6`. These lengths stay below the model context limit
including all outputs. Each engine is sized for its measured length.
