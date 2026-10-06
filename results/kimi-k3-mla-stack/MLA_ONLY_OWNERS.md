# Hybrid K3 prefill: only MLA/LoD on owners

Branch `lod-k3`; not pushed. Prototype:
[kimi_k3_mla_owners.py](../../benchmarks/kimi_k3_mla_owners.py).

The previous two-stage experiment moved both KDA and MLA to owners. This
experiment keeps all KDA layers on native TP8 (12 of 96 heads per rank) and
keeps all feed-forward layers unchanged: native EP8 INT4 MoE, plus the native
dense first-layer FFN. Both attention-side and FFN-side AttnRes execute.
Only MLA's Q/K/V/O projections and LoD attention become request-owner work.

Each rank already has identical hidden inputs from native TP/EP execution.
At B8, each MLA owner processes one complete row concurrently, then all eight
hidden-width outputs are all-gathered back to TP8. At B1, layer ownership
rotates and the owner's hidden-width output is broadcast. These transfers
are timed. There is no query-head exchange, return of per-head values, or
TP output-projection all-reduce for owner MLA. KDA projections, recurrence,
state and output all-reduce are unchanged.

The unchanged full trained checkpoint attaches to the existing INT4 weight
daemon. Cropping to the first 4 or 24 layers occurs only in the worker probe;
it does not create a different daemon entry or modify any shared weights.
The first 24 layers have 18 KDA, six MLA, one dense FFN and 23 MoE layers.
Gathered MLA weight copies are prepared outside timing. This prototype
retains the original TP shards too, so it is not a final memory layout.

## Validation and timing

- Identical-input, same-LoD MLA outputs are checked against native TP
  projections, independently of downstream MoE route changes.
- Every cache audit checks chronological history and completed updates at
  global 16K boundaries. Each row has its own cadence; B and DCP do not
  shorten it. A completed context leaves 256 tokens exact.
- The speed baseline is native AITER full causal expanded MLA under TP8,
  with all native KDA, MoE and residual work included, not a TP8 LoD control.
- Dense and owner modes use identical trained weights, real ProLong input
  token rows, packed KDA/MoE batch shape and scheduler-slice size.
- One exact-shape warmup precedes one synchronized wall-clock pass. The
  reported interval is the maximum across all eight workers. Updates,
  transport, native MoE, KDA, residuals and active-slice embeddings are
  inside; weight assembly, LM head, sampling and scheduling are outside.

This is a partial-model prefill layout test, not full-model serving latency
or a language-quality evaluation. There is no pipeline overlap yet. The
existing TP8/EP8 weights and distributed modules remain installed, so this
test specifically measures the smaller architectural change rather than
another attention-only pipeline.

## Results

Four-layer trained smoke complete, run `21200-kimi-mla-only-owner-smoke`,
node 4 GPUs 0–7. Raw [four-layer smoke](oct5-hybrid-owner-trained-4layer-smoke.json).

| Context | Batch | Row slice | Full TP8 s | MLA-owner LoD s | Full / owner |
|--:|--:|--:|--:|--:|--:|
| 32K | 1 | 2K | 0.198331 | 0.325359 | 0.610x |

Same-input TP/owner MLA relative L2 is 0.4173% in the exact prefix and
0.4157% after LoD begins. Native KDA and all three native MoE layers execute;
cache audits pass. This smoke is **slower**, not a speedup. It uses 2K B1
slices to validate the B8 slice machinery; it is not a best-B1 tuning claim.

The first-24 32K B1/B8 and 64K B1/B8 comparisons completed. The final
64K B8 retry finished just before cancellation was attempted at the user's
request; no further runs were started. The initial submission
`21201` rejected short source documents before vLLM startup and produced
no timings. The corrected `21202` concatenates real documents exactly as
the existing `kimi_k3_prefill_sweep` does (row `r` takes documents
`r, r+B, r+2B, ...`, cyclically), with no synthetic tokens or padding.
Full and owner modes use identical resulting rows.

The precomputed-embedding panel `21202` completed 32K B1/B8 and 64K B1,
then OOMed in native MoE during the 64K B8 dense warmup. It retained all
eight full embedded prompts (7 GiB) unnecessarily. Its completed B8/32K
point was dense 7.987205 s versus owner 7.874205 s (1.014x); it is only a
preliminary diagnostic, not the finalized panel. The failed job was
cancelled to release its stuck workers; the weight daemon was retained.

The `21203` probe embeds only active slices, as normal vLLM does, and
includes that work in both layouts. Its new B8 numerical check caught a
bug in the **probe's untimed TP reference**, not the owner output gather:
`attend_slice` can return a token-major shared-workspace view, so collecting
eight results and concatenating later overwrote earlier rows. The reference
now copies each row immediately; a CPU regression test covers scratch reuse.

Run `21204-kimi-mla-owner-b8-layout-fixed` passes B1 and B8 layout
checks (0.4173/0.4157% and 0.4349/0.4371% relative L2 respectively, exact
prefix/refined prefix). Do not mix its embedding-included timings with the
old embedding-excluded smoke/partial run. No production kernels changed.

### Validated initial 24-layer measurements

[Saved initial partial panel](oct5-hybrid-owner-trained-24layer-partial.json)
and [completed 64K B8 retry](oct5-hybrid-owner-trained-24layer-64k-b8.json).
Each completed measurement has all eight worker records and passing cache
audits. Active-slice embeddings are included, unlike the old smoke.

| Context | Batch | Row slice | Full TP8 s | MLA-owner LoD s | Full / owner |
|--:|--:|--:|--:|--:|--:|
| 32K | 1 | 2K | 1.344899 | 2.100786 | 0.640x |
| 32K | 8 | 2K | 8.010649 | 7.913299 | 1.012x |
| 64K | 1 | 2K | 2.845188 | 4.992677 | 0.570x |
| 64K | 8 | 2K | 17.098147 | 16.639941 | 1.028x |

The B8/64K owner warmup stalled. A read-only device diagnostic (`21205`)
showed 97–99% VRAM allocation and no memory read/write activity; this is
not an accepted latency. The probe unnecessarily kept its dense control's
full chronological archive while running owner LoD. The current version
releases that archive before owner execution and reclaims inactive allocator
blocks at a synchronized phase boundary, outside warmup and measurement.
Run `21206-kimi-mla-owner-64k-b8-no-control-archive` retried only the missing
B8/64K point, without repeating the other timings. Its dense control took
17.098147 s, consistent with 17.091257 s above, and its B8 layout check
passed again. The owner measurement completed at 16.639941 s, with all eight
worker records and passing cache audits. The job exited successfully before
the cancellation attempt, which returned "run is already finished". No
additional benchmark was launched; the weight daemon was left resident
and unchanged.

With 2K row slices, B8/32K is effectively tied and B8/64K shows a modest
1.028x speedup; B1 is slower.
This is not evidence of a major speed win, and no production default was
changed. The full native KDA/FFN work is included; no factor-of-eight
pipeline throughput extrapolation is being made.

## Reproduction

On node 4, with the already resident `kimi-k3-shared-int4-v6` daemon:

```bash
env \
  LOD_KIMI_SUBTILE64=score \
  LOD_KIMI_CHUNK_TILE_PACK=1 \
  LOD_KIMI_TILE_PACK_QUERY_BLOCK=1024 \
  LOD_KIMI_SORT_LEAF_ROUTES=1 \
  LOD_KIMI_LEAF_BLOCK_M=64 LOD_KIMI_LEAF_WARPS=1 \
  LOD_KIMI_PREFILL_RECLAIM_INTERVAL=0 \
  LOD_KIMI_REUSE_PREFILL_ALLOCATOR=1 \
  LOD_KIMI_PREFILL_MIN_FREE_GIB=4 \
  LOD_KIMI_TILE_REFINE=1 LOD_KIMI_DIRECT_LEAF_RESULT=1 \
  HSA_NO_SCRATCH_RECLAIM=0 TRITON_CACHE_AUTOTUNING=1 \
  VLLM_USE_TRITON_AWQ=1 TRITON_CACHE_DIR=/tmp/dan-agent/.triton/cache \
  AITER_CONFIG_FMOE="$PWD/results/kimi-k3-full-model-current/kimik3_i4_tuned_fmoe_b2x16k_merged.csv" \
  bash benchmarks/run_kimi_k3_v10_direct.sh \
  -m benchmarks.kimi_k3_mla_owners \
  --checkpoint /tmp/dan-agent-kimi-k3-f831ab66814297da540d832a5235f8e904f29d06 \
  --weight-cache-id kimi-k3-shared-int4-v6 \
  --real-token-cache results/kimi-k3-full-model-current/prolong-kimi-k3-speed-token-cache.pt \
  --lengths 32768 65536 --batches 1 8 \
  --chunk-size 2048 --layer-count 24 \
  --output results/kimi-k3-mla-stack/oct5-hybrid-owner-trained-24layer-panel.json
```

For the initial smoke use `--layer-count 4 --lengths 32768 --batches 1`.
There is no deterministic sampling seed: no sampling occurs. Exact real
token hashes and all worker timings/cache audits are recorded in the JSON.
