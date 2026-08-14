# LOD Attention for vLLM

This out-of-tree plugin keeps vLLM's native FP16/BF16 paged cache as an
authoritative representation. By default it uses native attention for prefill,
prefix-cache hits, mixed prefill/decode batches, and sliding-window layers.
Before a request's first pure decode batch, it gathers the request's native K/V
through the vLLM block table, reconstructs semantic LOD regions, and switches
compatible global-attention layers to recursive LOD decode. The same native
cache continues to receive every K/V update, so a request can safely fall back
to native attention and be reconverted later.

Prefill has two selectable paths. `VLLM_LOD_PREFILL_MODE=rebuild` (the
default) is the native-prefill behavior above. `VLLM_LOD_PREFILL_MODE=direct`
runs LOD attention and incrementally builds the LOD shadow during initial or
cached prefill, while vLLM still writes and retains the ordinary native K/V.
The direct path is used only when every row in a mixed/ragged batch starts at
zero or has an exactly matching LOD prefix. A native-only prefix-cache hit
safely falls back to native attention and is rebuilt before LOD decode.

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
VLLM_LOD_POOL_SIZE=8 \
VLLM_LOD_KV_BITS=4 \
vllm serve MODEL \
  --attention-backend CUSTOM \
  --kv-cache-dtype bfloat16 \
  --max-num-seqs 8
```

`VLLM_LOD_POOL_SIZE` is the maximum simultaneous pure-decode requests on each
worker. It defaults to 8. The pool is independent of vLLM's stable request
indices; a persistent device indirection table maps active batch rows to LOD
rows without changing captured tensor addresses. Set it equal to
`--max-num-seqs` when full CUDA graphs are enabled. Captured padding rows use
distinct unused LOD rows and are reset before replay; mapping padding onto a
live row would race its K/V append. A larger native prefill token batch is
allowed, but a pure decode graph wider than the pool fails with an actionable
error rather than silently aliasing rows.

The optional `VLLM_LOD_MAX_CONTEXT` caps each LOD row. It defaults to vLLM's
`max_model_len`. Other settings are `VLLM_LOD_CHUNK_SIZE` (256),
`VLLM_LOD_LOCAL_WINDOW` (512), `VLLM_LOD_STATE_FACTOR` (16),
`VLLM_LOD_STATE_MIN` (256), `VLLM_LOD_OPEN_COUNT` (8), and
`VLLM_LOD_QUANT_GROUP_SIZE` (32). Set `VLLM_LOD_PREFILL_MODE` to `direct` to
exercise direct LOD prefill; leave it at `rebuild` for native prefill followed
by conversion. `VLLM_LOD_ROUTING_GEOMETRY=auto` selects coherence-aware state
routing for attention modules with normalized keys and spherical routing for
unnormalized keys. `raw`, `spherical`, and `coherence` are available as explicit
diagnostic overrides.

## Execution contract

- The default rebuild path leaves native prefill unchanged. Both prefill modes
  retain the native cache and remain compatible with continuous batching.
- Prefix conversion reads each backend's canonical paged-cache view. In
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
  exact shadow prefix. Otherwise it stays entirely native; affected LOD rows
  are invalidated and reconstructed from the authoritative native cache before
  their next pure decode batch. Already-ready one-token decode rows are kept
  current from that native batch's post-RoPE K/V, avoiding repeated full
  reconstruction under continuous batching.
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

The native cache is deliberately retained in this first integration. It is
the source for prefix sharing, fallback, and reconversion; LOD therefore adds
a second representation rather than immediately reducing total server memory.
INT4 still reduces the added LOD leaf pool substantially. Releasing native
block references after conversion requires scheduler/KV-manager ownership and
is a separate integration stage; it must not be approximated by quantizing
sequential native blocks.

## Quality validation

The integration was checked with Qwen3.5-0.8B on vLLM 0.27.1 for ROCm. Each
mode used eight 8K examples at batch size eight.

| Evaluation | Native vLLM | LOD backend |
| --- | ---: | ---: |
| ProLong token CE | 3.251862 | 3.251862 |
| ProLong perplexity | 25.838413 | 25.838413 |
| NIAH-S3 exact match | 8/8 | 8/8 |

ProLong prompt log-probabilities exercise the unchanged native-prefill path,
so equality verifies that selecting the plugin does not perturb prefill.
NIAH-S3 exercises native BF16 prefill, native-to-LOD conversion, and captured
INT4 recursive LOD decode. The worker recorded 48 real cache installations for
that run: six global-attention layers for each of eight requests.

The paired evaluator is `scripts/eval_vllm_lod_quality.py`. Its LOD NIAH check
requires both an executed decode path and at least one real prefix conversion,
which prevents graph-capture warmups from being mistaken for an LOD result.

Warm batch throughput can be reproduced with:

```bash
python scripts/benchmark_vllm_lod_speed.py \
  --mode lod --length 8192 --batch-size 8 --decode-tokens 300 \
  --output artifacts/vllm_lod_speed/lod_b8_8k_d300.json
```

Use at least 257 decode tokens when comparing amortized serving performance so
the measurement includes a real state-update boundary.
