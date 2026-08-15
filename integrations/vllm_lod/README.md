# LOD Attention for vLLM

This out-of-tree plugin makes semantic LOD state the authoritative cache for
eligible global-attention layers. Initial and cached prefill build that state
directly, and decode advances the same fixed-address rows. With vLLM prefix
caching disabled, the scheduler sees only a tiny logical placeholder and no
chronological attention K/V is written. Prefix caching instead keeps a bounded
chronological staging window so its block hashes remain usable. Recurrent/Mamba
state and attention layers that are not LOD-compatible remain native.

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
  --no-enable-prefix-caching \
  --max-num-seqs 8
```

The `--no-enable-prefix-caching` form gives the smallest cache. Omit that flag
when cross-request vLLM prefix reuse is more important; LOD then retains only
the bounded native staging needed to validate and resume hashed prefixes.

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
the exact chronological window retained when prefix caching is enabled.
`VLLM_LOD_NATIVE_PLACEHOLDER_CACHE` defaults to true and removes chronological
attention K/V when prefix caching is disabled. `VLLM_LOD_NATIVE_CACHE_HEADROOM`
defaults to 1.0; prefix caching enforces a 2.0 minimum for dormant hashed
blocks.
`VLLM_LOD_ROUTING_GEOMETRY=auto` selects coherence-aware state routing for
attention modules with normalized keys and spherical routing for unnormalized
keys. `raw`, `spherical`, and `coherence` are explicit diagnostic overrides.

## Execution contract

- The default authoritative path runs direct LOD prefill. It uses scheduler-only
  placeholder pages when prefix caching is off and bounded native staging when
  prefix caching is on. `dual` ownership retains the older
  native-prefill/rebuild path.
- Dual-mode prefix conversion reads each backend's canonical paged-cache view. In
  particular, ROCm's raw allocation has a nominal token-major shape but is
  written through head-major K/V views; treating the raw shape as semantic
  token order scrambles the reconstructed prefix.
- Pure one-token decode uses fixed-address LOD pools and stable request-row
  indirection, so the Triton decode launches can be captured in CUDA graphs.
- Authoritative mode disables vLLM's asynchronous two-model-batch queue because
  an in-place LOD row must settle before its next update is prepared. Ordinary
  request batching and CUDA graph replay remain enabled.
- Exact protected/sink K/V lives in a separate side cache rather than consuming
  clustered state slots, and is fused into the final attention reduction.
- State/page pools and one maximum-batch decode workspace are reserved before
  vLLM computes its native KV-block budget. Smaller captured batches use stable
  views of that workspace, so decode replay does not allocate or change tensor
  addresses.
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
Without prefix caching, one logical placeholder block per active request is
enough; K/V writes to it are skipped. With prefix caching, chunk-local semantics
free settled staging blocks and startup sizes the pool from active, in-flight,
and dormant-prefix demand rather than maximum context length. Recurrent state
remains native. Semantic leaves are quantized only after region assignment;
sequential native blocks are never treated as quantizable pages.

For Qwen3.5-0.8B at 64K, batch eight, INT4 LOD uses 2.817 GB of semantic cache,
0.221 GB of native scheduler/recurrent state, and a 0.063 GB shared decode
workspace. Persistent cache is therefore 3.037 GB, 55.6% below full attention's
6.845 GB. The earlier Transformers figure of 2.758 GB counted semantic
attention state but not Qwen's recurrent GDN cache; adding the same 0.221 GB
puts it within 2.0% of the vLLM result.

`dual` mode intentionally retains both representations and should not be used
to assess memory savings.

## Quality validation

The authoritative integration was checked with Qwen3.5-0.8B on vLLM 0.27.1
for ROCm. NIAH used eight 8K examples at batch size eight; the initial ProLong
check used two 8K documents.

| Evaluation | Native vLLM | LOD backend |
| --- | ---: | ---: |
| ProLong token CE | 2.125621 | 2.128387 |
| ProLong perplexity | 8.378100 | 8.401305 |
| NIAH-S3 exact match | 8/8 | 8/8 |

ProLong prompt log-probabilities exercise direct LOD prefill; its CE increase in
this paired check was 0.002766 (0.13%). NIAH-S3 exercises direct prefill and INT4
recursive LOD decode. The final NIAH run used CUDA graphs. The worker recorded
48 real authoritative cache installations: six global-attention layers for
each of eight requests.

Full 503-example LongBench v2 runs used identical guided A-D decoding for full
and LOD attention. After fixing cached-prefill finalization, stable page
chronology, and unused INT4-page summaries, Qwen3.5-35B-A3B LOD scored 229/503
(45.53%) versus full attention's 245/503 (48.71%). Qwen3.8-27B-FP8 LOD scored
256/503 (50.89%) versus full attention's 269/503 (53.48%).

| Model and subset | Full attention | LOD attention |
| --- | ---: | ---: |
| Qwen3.5 overall | 48.71% | 45.53% |
| Qwen3.5 short / medium / long | 52.78% / 48.84% / 41.67% | 49.44% / 46.05% / 37.96% |
| Qwen3.8 overall | 53.48% | 50.89% |
| Qwen3.8 short / medium / long | 55.00% / 53.95% / 50.00% | 53.89% / 49.77% / 48.15% |

All four runs truncated the same 205 prompts to the model's 131,072-token
limit. The remaining LOD gaps are 3.18 percentage points on Qwen3.5 and 2.58
points on Qwen3.8; NIAH success alone is not sufficient validation for this
approximation.

## Warm serving performance

The following Qwen3.5-0.8B measurements use batch size eight, CUDA graphs, a
65,536-token scheduler budget, an 8,192-token long-prefill threshold, 1,025
generated tokens, and the speed-oriented BF16 LOD leaf cache
(`VLLM_LOD_KV_BITS=0`). They report warm medians from five to seven runs. The
exact full backend uses `ROCM_AITER_UNIFIED_ATTN`, the fastest working exact
backend tested on this ROCm 7.2 system, and a tightly sized native block pool.
The 1,024 measured decode intervals include four 256-token LOD state updates,
so their costs are amortized rather than represented by a single boundary.

| Context | Full prefill | LOD prefill | Full decode step | LOD decode step |
| --- | ---: | ---: | ---: | ---: |
| 16K | 0.898 s | 1.056 s | 3.09 ms | 3.47 ms |
| 64K | 8.039 s | 6.177 s | 5.11 ms | 4.26 ms |

At 64K this is a 1.30x prefill speedup and a 1.20x decode-step speedup. At 16K,
exact full attention remains 1.18x faster in prefill and 1.12x faster in decode.
INT4 remains the memory-oriented mode. The current low-memory 64K result is
reported in the memory section above. Total device use is not reported as a
cache metric because model weights, compiled kernels, and runtime workspaces
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
arguments. Prefill routing uses 128-query tiles, coarse attention uses 32x64
tiles, recursive page attention uses two-page tiles, and the local branch uses
AITER. The final merge now honors arbitrary output strides, so packed prefill
writes directly into vLLM's token-major output rather than allocating and
copying a second full output tensor. An unseen device/kernel configuration
still incurs ordinary Triton JIT work; production images should preserve or
pre-populate their Triton compilation cache. Warm medians above exclude setup
time but can still contain isolated shape-specific compilation outliers.

The paired evaluator is `scripts/eval_vllm_lod_quality.py`. Its LOD NIAH check
requires both an executed decode path and real cache installation, preventing
graph-capture warmups from being mistaken for an LOD result.

Warm batch throughput can be reproduced with:

```bash
VLLM_LOD_KV_BITS=0 python scripts/benchmark_vllm_lod_speed.py \
  --mode lod --length 8192 --batch-size 8 --decode-tokens 1025 --repeats 5 \
  --max-num-batched-tokens 65536 --long-prefill-token-threshold 8192 \
  --output artifacts/vllm_lod_speed/lod_b8_8k_d1025.json
```

Use at least 1,025 decode tokens when comparing amortized serving performance
so the measurement spans four state-update boundaries.
