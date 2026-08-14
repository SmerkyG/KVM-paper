# LOD Attention for vLLM

This out-of-tree plugin makes semantic LOD state the authoritative cache for
eligible global-attention layers. Initial and cached prefill build that state
directly, and decode advances the same fixed-address rows. vLLM retains only a
bounded chronological staging window for those layers; recurrent/Mamba state
and attention layers that are not LOD-compatible remain in their native cache.

Completed LOD rows remain available for content-matched prefix reuse. When
vLLM resumes a prompt at a physical block boundary, the plugin verifies the
exact token prefix and rolls the retained row back only within its unclustered
exact tail. It never tries to undo clustered history. The bounded native
staging cache is not a lossless remote-attention fallback.

`VLLM_LOD_CACHE_OWNERSHIP=dual` retains the earlier diagnostic design: native
FP16/BF16 K/V remains authoritative, prefill can run natively, and LOD is a
rebuildable decode shadow. This mode is useful for comparisons but does not
provide the authoritative mode's memory saving.

INT4 is applied only after keys have been assigned to a semantic LOD region.
The plugin never quantizes chronological vLLM cache blocks as if they were LOD
pages.

## Install and run

Install the plugin editable into the environment that provides vLLM:

```bash
uv add --editable /absolute/path/to/code/integrations/vllm_lod
```

Then select the registered custom backend:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_LOD_CACHE_OWNERSHIP=lod \
VLLM_LOD_POOL_SIZE=8 \
VLLM_LOD_KV_BITS=4 \
vllm serve MODEL \
  --attention-backend CUSTOM \
  --kv-cache-dtype bfloat16 \
  --max-num-seqs 8
```

`VLLM_LOD_POOL_SIZE` is the number of simultaneous or retained request rows on
each worker. It defaults to 8 and must be at least `--max-num-seqs`. The pool is
independent of vLLM's stable request indices; a persistent device indirection
table maps active batch rows to LOD rows without changing captured tensor
addresses. Graph padding never borrows a scheduled row. It may temporarily use
an unscheduled or completed row after restoring that row's real tail length;
the dummy append is placed just beyond the retained tail and discarded before
the row is observed again.

The optional `VLLM_LOD_MAX_CONTEXT` caps each LOD row. It defaults to vLLM's
`max_model_len`. Other settings are `VLLM_LOD_CHUNK_SIZE` (256),
`VLLM_LOD_LOCAL_WINDOW` (512), `VLLM_LOD_STATE_FACTOR` (16),
`VLLM_LOD_STATE_MIN` (256), `VLLM_LOD_OPEN_COUNT` (8), and
`VLLM_LOD_QUANT_GROUP_SIZE` (32). Authoritative ownership forces
`VLLM_LOD_PREFILL_MODE=direct`. `VLLM_LOD_NATIVE_STAGING_CHUNK` (1024) controls
the exact chronological window retained by vLLM, while
`VLLM_LOD_NATIVE_CACHE_HEADROOM` (1.5) controls transient block-pool headroom.
`VLLM_LOD_ROUTING_GEOMETRY=auto` selects coherence-aware state routing for
attention modules with normalized keys and spherical routing for unnormalized
keys. `raw`, `spherical`, and `coherence` are explicit diagnostic overrides.

## Execution contract

- The default authoritative path runs direct LOD prefill and keeps only bounded
  native staging for eligible global-attention layers. `dual` ownership retains
  the older native-prefill/rebuild path.
- Dual-mode prefix conversion reads each backend's canonical paged-cache view. In
  particular, ROCm's raw allocation has a nominal token-major shape but is
  written through head-major K/V views; treating the raw shape as semantic
  token order scrambles the reconstructed prefix.
- Pure one-token decode uses fixed-address LOD pools and stable request-row
  indirection, so the Triton decode launches can be captured in CUDA graphs.
- Exact protected/sink K/V lives in a separate side cache rather than consuming
  clustered state slots, and is fused into the final attention reduction.
- State/page pools and scratch for configured decode capture sizes are reserved
  before vLLM computes its native KV-block budget. Decode replay does not
  allocate or change tensor addresses.
- State/page catch-up runs in `ModelState.preprocess_state`, between graph
  replays, in 256-token batches. Decode itself only appends to a fixed local
  tail and advances one integer length per active row.
- A direct-prefill mixed batch uses LOD only when every request can advance an
  exact authoritative prefix. In `lod` ownership, a missing prefix is an error:
  bounded staging cannot reconstruct discarded remote history. In `dual`
  ownership the batch can use native attention and rebuild later.
- Equal-length native prefixes are gathered and rebuilt as one batch per layer;
  ragged prefixes are grouped by length. Ordinary decode tokens update only
  host metadata between state boundaries, so eager state-maintenance launches
  occur once per update interval rather than once per token.
- Sliding-window, encoder, ALiBi, attention-sink, soft-capped, quantized-native
  KV, speculative decode, and DCP paths remain native or are rejected when a
  lossless fallback is not available. Tensor parallelism and hybrid recurrent
  layers do not alter the per-layer LOD contract.

The first integration deliberately uses one uniform attention mode per batch.
Every pure one-token decode batch uses LOD, including short requests, because a
captured vLLM graph cannot dynamically swap its attention implementation.
Length-based native/LOD dispatch needs a scheduler-visible graph key and is a
later integration stage, not a hidden branch in the captured kernel.

## Current memory behavior

Authoritative mode replaces full chronological K/V for eligible global layers.
The vLLM scheduler uses chunk-local cache semantics to free settled staging
blocks after each prefill/decode step, and startup caps that native pool from
the global in-flight token budget rather than the maximum context length.
Recurrent state remains native. Semantic leaves are quantized only after region
assignment; sequential native blocks are never treated as quantizable pages.

`dual` mode intentionally retains both representations and should not be used
to assess memory savings.

## Quality validation

The authoritative integration was checked with Qwen3.5-0.8B on vLLM 0.27.1
for ROCm. NIAH used eight 8K examples at batch size eight; the initial ProLong
check used two 8K documents.

| Evaluation | Native vLLM | LOD backend |
| --- | ---: | ---: |
| ProLong token CE | 3.109570 | 3.113030 |
| ProLong perplexity | 22.411398 | 22.489091 |
| NIAH-S3 exact match | 8/8 | 8/8 |

ProLong prompt log-probabilities exercise direct LOD prefill; its CE increase in
this small check was 0.003461 (0.11%). NIAH-S3 exercises direct prefill and INT4
recursive LOD decode. The final NIAH run used CUDA graphs. The worker recorded
48 real authoritative cache installations: six global-attention layers for
each of eight requests.

## Warm serving performance

The following Qwen3.5-0.8B measurements use batch size eight, CUDA graphs, a
65,536-token scheduler budget, an 8,192-token long-prefill threshold, 1,025
generated tokens, and the authoritative INT4 cache. They report the median of
five runs after one complete warm request. The exact full backend uses
`ROCM_AITER_UNIFIED_ATTN`, the fastest working exact backend tested on this ROCm
7.2 system, and a tightly sized native block pool. The 1,024 measured decode
intervals include four 256-token LOD state updates, so their costs are
amortized rather than represented by a single boundary.

| Context | Full prefill | LOD prefill | Full decode step | LOD decode step |
| --- | ---: | ---: | ---: | ---: |
| 16K | 0.911 s | 1.270 s | 3.19 ms | 3.88 ms |
| 64K | 8.033 s | 6.836 s | 5.10 ms | 4.25 ms |

At 64K this is a 1.18x prefill speedup and a 1.20x decode-step speedup. At 16K,
exact full attention remains 1.39x faster in prefill and 1.22x faster in decode.
At 64K, the persistent cache tensors attributed to the LOD semantic cache and
bounded native staging used 5.804 GB, versus 7.248 GB for the full backend
(19.9% less). At 16K they used 3.986 GB versus 2.265 GB, because fixed LOD pools
and bounded staging outweigh compression at short context. Total device use is
not reported as a cache metric because compiled kernels and runtime workspaces
remain resident and vary with warmup history.

vLLM's automatic ROCm selection chose `ROCM_ATTN` for this hybrid model. Its
native paged kernel supports only 16- and 32-token blocks, while Qwen3.5's
hybrid recurrent cache forces a 544-token attention block. It therefore fell
back to the generic Triton chunked-paged path, inflating the 16K decode step to
15.93 ms and the 64K decode step to 56.78 ms. Those timings are backend fallback
diagnostics, not a fair exact-attention baseline.

Cached-prefill state boundaries are rounded down to the 256-token update grid.
This keeps at most one extra partial chunk exact and prevents arbitrary request
lengths from producing an unbounded family of page-update specializations.
The fused routing, coherence-update, final-reduction, and sparse page-transfer
kernels keep ragged tensor extents and their batch/head strides as runtime
arguments. On the 16K batch-eight diagnostic this reduced inference-time JIT
events from 117 to 56; all remaining compilations completed during the one warm
request. Two clean paired replicas produced LOD prefill medians of 1.270 and
1.279 seconds; the five runs in the reported replica stayed between 1.255 and
1.278 seconds.
An unseen device/kernel configuration still incurs ordinary Triton JIT work;
production images should preserve or pre-populate their Triton compilation
cache. The timings above do not include that one-time compilation.

The paired evaluator is `scripts/eval_vllm_lod_quality.py`. Its LOD NIAH check
requires both an executed decode path and real cache installation, preventing
graph-capture warmups from being mistaken for an LOD result.

Warm batch throughput can be reproduced with:

```bash
python scripts/benchmark_vllm_lod_speed.py \
  --mode lod --length 8192 --batch-size 8 --decode-tokens 1025 --repeats 5 \
  --max-num-batched-tokens 65536 --long-prefill-token-threshold 8192 \
  --output artifacts/vllm_lod_speed/lod_b8_8k_d1025.json
```

Use at least 1,025 decode tokens when comparing amortized serving performance
so the measurement spans four state-update boundaries.
