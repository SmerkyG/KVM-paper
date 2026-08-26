# LOD Attention for vLLM

This out-of-tree plugin makes semantic LOD state the authoritative cache for
eligible global-attention layers. Initial and cached prefill build that state
directly, and decode advances the same fixed-address rows. Authoritative LOD
layers bind no chronological native attention K/V, including when vLLM prefix
caching is enabled. Prefix hashes come from token/block metadata; a hit resumes
an exactly matched retained LOD row. Recurrent/Mamba state and attention layers
that are not LOD-compatible remain native.

Completed LOD rows remain available for content-matched prefix reuse. When
vLLM resumes a prompt at a physical block boundary, the plugin verifies the
exact token prefix. A hit inside the unclustered tail is a metadata-only
rollback. For an older shared-prefix boundary, two-level LOD rebuilds its
semantic centroids from the row's chronological exact-leaf archive; it never
tries to invert clustered history. No native attention-cache fallback is used
for an external LOD layer.

External semantic ownership is the only supported cache mode. The plugin does
not expose the older dual-cache, bounded-staging, or 1x1/1x1x1 placeholder
paths, so a CUSTOM attention run cannot silently select their slower geometry.

The default `VLLM_LOD_LEVELS=2` path is the current flat BF16 implementation
shared with the Hugging Face benchmark. Exact leaves are stored once in a
dense chronological pool; compact per-centroid page tables index that pool.
It opens all exact pages in each of the top-eight centroids. Set
`VLLM_LOD_DENSE_LEAF_STORAGE=0` only for matched comparisons with the older
region-owned physical-page layout. `VLLM_LOD_LEVELS=3` retains the older
recursive page-summary implementation and its INT4/INT8 storage modes.

When GQA-union decode and its AITER path are enabled, one indexed M16/N64
attention call consumes exact leaves, the local window, protected sinks, and
unopened coarse entries from a shared page-size-one arena. Coarse entries carry
an FP16 logit bias. Every centroid occupies a fixed suffix position: opened or
inactive entries receive `-inf`, while unopened entries receive `log(count)`.
This preserves represented token mass without compacting a per-query centroid
list, a separate coarse value pass, or a final branch merge.

`VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1` selects the fixed-list variant. At
each 256-token state-update boundary it stores one persistent list per KV head:
local positions, sinks, every coarse position, then all valid leaves in
centroid-major order. Decode changes only an epoch-stamped opened-centroid mask.
A parallel mask-preparation kernel resolves owners and route epochs into one
byte per list entry and one byte per 64-entry block. The M16/N64 page-size-one
kernel first reads the block byte; a fully inactive tile exits without loading
its lane mask or K/V and without issuing QK or PV MFMA. Partially active tiles
retain the ordinary AITER-shaped online softmax, including `log(count)` on
unopened coarse entries. The default long fixed scan uses 128 independent
segments at the measured batch-8 geometry. The route-dependent compact leaf-list
construction and coarse/leaf correction merge are therefore absent. This
option requires two-level BF16 GQA-union decode with
`VLLM_LOD_DECODE_GQA_UNION_HIP=1` and is mutually exclusive with
`VLLM_LOD_DECODE_GQA_STAGED_FIXED_AITER=1`.
`VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N` selects the experimental fast-fail
tile width (16, 64, or 128; default 64), and
`VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS` selects 8--512 split segments
(default 128).

`VLLM_LOD_DECODE_ROUTE_COHORT=1` restricts dynamic top-k or mass-cutoff
routing to centroids in the same small-posting-list cohort used by the static
variant: an inclusive `max(16, ceil(sqrt(T) / 16))` leaf cap.  This policy
replaces `VLLM_LOD_DECODE_MAX_OPEN_LEAVES`; it is not intersected with that
legacy guard.  `VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP` remains an optional fixed
override for controlled single-length experiments.  `VLLM_LOD_OPEN_COUNT`
selects one through eight routes for top-k experiments.  Predicted-mass decode
instead uses `VLLM_LOD_DECODE_GQA_PREDICTED_MASS=1` and
`VLLM_LOD_DECODE_GQA_MASS_FRACTION`; it applies the current query to the
eligible centroids while reusing only the preceding token's total mass as the
cutoff denominator.

`VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER=1` selects the routing-free compact
variant. At a state update it builds one persistent page-size-one list per KV
head: sinks, every leaf whose centroid count is at most
the active cap, one `log(count)`-biased coarse entry for each larger centroid,
and the fixed local-window suffix. By default the inclusive cap for a request
of length `T` is `max(16, ceil(sqrt(T) / 16))`. Set
`VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP` to use a fixed experimental override, or
`VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP_MIN` to change the default floor. Decode
then performs a single indexed attention scan over that list; it has no
coarse-score, top-k, union, or mask-construction dependency. It shares the
split count selected by `VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS` and requires
two-level BF16 GQA-union decode with
`VLLM_LOD_DECODE_GQA_UNION_HIP=1`.

## Install and run

Install the plugin editable into the environment that provides vLLM:

```bash
uv add --editable /absolute/path/to/code/integrations/vllm_lod
```

### Reuse loaded weights while developing

The package also registers an `ipc_cache` model loader. A long-lived, GPU-light
broker loads and post-processes each exact model/TP/PP configuration on its
first request, then fresh vLLM workers map the retained final parameters,
buffers, and tensor attributes through CUDA/HIP IPC. The client constructs the
module tree on `meta`, so it neither rereads the checkpoint nor briefly
allocates a second copy of the weights.

The first vLLM process using `--load-format ipc_cache` automatically starts the
broker if it is not already running. Startup is serialized by an owner-only
filesystem lock, so concurrent TP ranks share one broker. The broker then stays
alive for subsequent vLLM processes on that node and GPU allocation. To choose
non-default eviction limits up front, it can still be started manually:

```bash
VLLM_PLUGINS=lod_attention \
vllm-weight-cache --cache-id dev
```

Fresh LOD or ordinary vLLM processes on the same node and physical GPUs can
then use it as follows:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_WEIGHT_CACHE_ID=dev \
vllm serve MODEL \
  --tensor-parallel-size 8 \
  --load-format ipc_cache \
  --attention-backend CUSTOM
```

Use `VLLM_PLUGINS=weight_cache` instead when testing normal attention without
installing the LOD hooks. `VLLM_WEIGHT_CACHE_DIR` selects a non-default
owner-only socket directory. Equivalent per-run settings can be passed through
`--model-loader-extra-config '{"cache_id":"dev","cache_dir":"..."}'`. The
`auto_start` setting defaults to true; set it to false or export
`VLLM_WEIGHT_CACHE_AUTO_START=0` when an absent broker should be an error. Broker
startup logs are written to `broker.log` in the selected cache namespace. The
client sends its exact vLLM configuration to the broker; the backing load
defaults to `auto` and can be changed with `backing_load_format` and
`backing_loader_extra_config` in that same object. A fingerprint mismatch or an
incomplete meta mapping is a hard error rather than a silent disk fallback.

The broker tracks live vLLM worker PIDs as leases. Its default cache budget is
60% of each GPU's memory; after a miss it LRU-evicts only resident models whose
workers have exited. Use `--max-cache-fraction` or
`--max-cache-gb-per-gpu` to change that budget. If an uncached model hits OOM
while inactive models are resident, the broker evicts them and retries once.

The daemon must remain alive for the lifetime of every mapped engine. Check or
stop one cache namespace with:

```bash
vllm-weight-cache status --cache-id dev
vllm-weight-cache stop --cache-id dev
```

On `cluster-run`, keep the daemon as a detached job and use `--overlap-own` for
development jobs that must share its allocated GPUs. The `ipc_cache` plugin
accounts for weights that were resident before vLLM's normal memory snapshot,
so ordinary `--gpu-memory-utilization` settings continue to include the mapped
model weights. The broker is single-node and DP=1; TP and PP are supported and
discovered from each requesting vLLM engine.

The repository's current vLLM benchmark, quality, prefix-cache, chat-batch, and
NIAH panel entry points select `ipc_cache` by default. They share the namespace
from `VLLM_WEIGHT_CACHE_ID` (`dev` in the shell launchers). Set
`VLLM_WEIGHT_CACHE_LOAD_FORMAT=auto` only for an explicit uncached control.

Then select the registered custom backend:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_LOD_POOL_SIZE=8 \
VLLM_LOD_LEVELS=2 \
VLLM_LOD_KV_BITS=0 \
vllm serve MODEL \
  --attention-backend CUSTOM \
  --kv-cache-dtype bfloat16 \
  --enable-prefix-caching \
  --max-num-seqs 8
```

Prefix caching does not allocate chronological K/V for authoritative LOD
layers. The retained LOD-row pool must be large enough for the live requests
and prefixes that should remain reusable.

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
`VLLM_LOD_QUANT_GROUP_SIZE` (32). The two-tier path additionally defaults to
4,096-token prefill chunks, a 4,864-token prefill local field, 4,096-token
prefill state updates, expert-layout BF16 leaf attention, and the two-level
page directory. These are configurable with `VLLM_LOD_PREFILL_CHUNK_SIZE`,
`VLLM_LOD_PREFILL_LOCAL_WINDOW`, `VLLM_LOD_PREFILL_STATE_UPDATE_SIZE`,
`VLLM_LOD_LEAF_LAYOUT`, and `VLLM_LOD_LEAF_PAGED_DIRECTORY`.
`VLLM_LOD_PREFILL_STATIC_LEAF_AITER=1` replaces query-dependent prefill top-k
with the static small-centroid cohort. After each 4,096-token state catch-up it
rebuilds one page-size-one AITER list per KV head containing every leaf whose
centroid has at most `max(16, ceil(sqrt(T) / 16))` leaves; the list stays fixed
for that query chunk. Larger centroids remain represented by their biased
coarse entries. `VLLM_LOD_PREFILL_STATIC_LEAF_CAP_MIN` changes the floor. This
experimental prefill mode requires two-level BF16 dense leaf storage.
`VLLM_LOD_LEAF_SEAL_CAPACITY` is an opt-in diagnostic; it is unset by default,
and the reported two-tier benchmarks retain every leaf. The two-tier cache accepts
`VLLM_LOD_KV_BITS=0` (BF16) or `8` (signed INT8 K/V with one BF16 scale per
token); its routing state and page summaries remain BF16.
`VLLM_LOD_DENSE_LEAF_STORAGE` defaults to true and
removes the physical page-fragmentation overhead without discarding leaves.
`VLLM_LOD_PREFILL_INT8_ROUTE_MMA=1` enables the experimental Sage-style
fused centroid QK path: queries and centroid keys are scaled per row, QK uses
INT8 MMA, and route selection plus stable coarse attention consume the scores
without materializing a query-by-centroid tensor. It remains opt-in because
the current Triton value accumulator reduces occupancy on gfx942; the regular
materialized route GEMM is faster in the measured Qwen3.5-0.8B configuration.
For leaf attention, the Sage-style path quantizes Q and uses INT8 QK MMA.
Below 64K it dequantizes V for BF16 PV MMA; at 64K and above it also quantizes
the probabilities for INT8 PV because the longer posting-list scan amortizes
that fixed work. `VLLM_LOD_PREFILL_INT8_PV_MMA=0` or `1` overrides this
automatic crossover. Batch-8 INT8 leaf kernels use two warps by default; one
warp under-occupies the true batched workload.
Recursive three-tier storage accepts 0, 4, or 8 bits;
quantized storage requires the same precision for K and V. External ownership
forces `VLLM_LOD_PREFILL_MODE=direct`.
`VLLM_LOD_PREFIX_ROLLBACK_TOKENS` (1024) controls the exact local tail retained
for inexpensive metadata-only prefix rollback; it is semantic LOD state, not a
native cache. Eligible layers always retain a scheduler-visible virtual
full-history group with the original block geometry and are restored as
worker-only attention groups, so they receive ordinary metadata while binding
no native K/V tensor. With prefix caching enabled, that virtual group stores
chained token hashes in a bounded CPU sentinel table, but it never consumes GPU
tensor bytes or IDs from vLLM's shared physical block pool. Physical cache
sizing therefore uses only the model's remaining native groups, while hybrid
coordination keeps the full-attention topology that vLLM uses to build its fast
local-attention metadata path. This is an unconditional cache-ownership
invariant for every LOD-eligible CUSTOM layer, not a model- or mode-specific
option: scheduler specs and model markers must match exactly, and startup fails
if an eligible layer has an unsupported spec, loses its virtual group, or
appears in any native GPU K/V tensor. The plugin retains and exactly verifies
the corresponding semantic LOD row before accepting a prefix hit.
`VLLM_LOD_ROUTING_GEOMETRY=raw` matches the current two-tier HF configuration.
`auto` selects coherence-aware state routing for
attention modules with normalized keys and spherical routing for unnormalized
keys. `raw`, `spherical`, and `coherence` are explicit diagnostic overrides.

For recursive three-tier decode,
`VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND=fused` retains the existing grouped
state-route kernel. `resplit` selects the allocation-free score-materialization
path. The latter keeps score generation, softmax, split value accumulation,
and value reduction independently testable. It reuses each score-table tile
load for both tile top-eight and tile LSE, then reduces the top-eight candidates
and LSE partials together. This conservative fusion is the default for the
`resplit` backend; the benchmark reproducer can still execute every stage
separately. `resplit` is not a universal speed default: current batch-8
measurements favor it for D>=256 or high KV-head count, while the grouped route
remains faster for shorter D128/KV2 workloads.
The score table is FP32 and the count-corrected probability table is FP16; this
combination removed score-ordering loss and was more accurate against exact
coarse attention than either the initial BF16-probability path or the existing
grouped route in the Qwen control.

`VLLM_LOD_AUG19_COMPAT=1` selects the closest reconstructable execution path
to the August 19, 2026 BF16 LongBench run: fixed eight-way Triton decode,
cooperative GQA/HIP decode disabled, and a four-warp leaf-route reduction.
The historical run used an uncommitted working tree, so this is an execution
compatibility preset rather than a byte-exact source restoration. The current
optimized path remains the default. `VLLM_LOD_LEAF_REDUCE_NUM_WARPS` controls
the route-reduction warp count directly when the compatibility preset is off.

For the supplied LongBench launcher, the seventh argument selects the profile:
`current` (default) or `aug19`. On Qwen3.5-35B-A3B both profiles use the generic
split decoder because its 16 query heads and two KV heads form GQA groups of
eight. The only exercised kernel difference between those two LongBench
profiles is therefore the one- versus four-warp prefill leaf-route reduction.

## Execution contract

- The default authoritative path runs direct LOD prefill. Eligible global
  layers bind no native K/V tensor. Prefix caching keeps only scheduler token
  hashes plus retained semantic LOD rows. There is no alternate cache-ownership
  mode or native rebuild path.
- Pure one-token decode uses fixed-address LOD pools and stable request-row
  indirection, so the Triton decode launches can be captured in CUDA graphs.
- vLLM asynchronous scheduling remains enabled. Model forwards and all LOD
  state mutations are submitted in worker order on the main GPU stream, while
  only sampling-output copies overlap on a separate stream. Consequently the
  large semantic state remains single-buffered and graph addresses stay fixed.
- Use a 16,384-token `long_prefill_token_threshold` per request. The aggregate
  `max_num_batched_tokens` should remain `batch_size * 16,384` (131,072 for
  batch 8); setting the aggregate limit itself to 16,384 serializes the batch.
- Two-tier mode protects the sink inside the state, matching the current HF
  implementation. Recursive compatibility mode retains its separate sink
  branch.
- State/page pools and one maximum-batch decode workspace are reserved before
  vLLM computes its native KV-block budget. Smaller captured batches use stable
  views of that workspace, so decode replay does not allocate or change tensor
  addresses.
- State/page catch-up runs in `ModelState.preprocess_state`, between graph
  replays, in 256-token batches. Decode itself only appends to a fixed local
  tail and advances one integer length per active row.
- Portable two-tier decode uses one fixed eight-way Triton split kernel for
  exact leaves and local tokens, followed by one stable-LSE reduction. This is
  the path for GQA-8 Qwen3.5-35B-A3B and for every unsupported geometry.
- On gfx942 only, H=256/GQA-4 decode may use one specialized HIP kernel that
  loads a routed leaf tile once for the four query heads sharing its KV head.
  A small Triton kernel handles the local branch. Both BF16 and signed INT8
  leaf storage are supported. There is deliberately no second cooperative
  Triton fallback; disabling or missing this specialization selects the
  generic split decoder.
- A direct-prefill mixed batch uses LOD only when every request can advance an
  exact authoritative prefix. A missing prefix is an error because there is
  deliberately no native attention fallback. Retained two-level rows can
  reconstruct an older matched prefix from their chronological leaves.
  Ordinary decode tokens update only host metadata between state boundaries,
  so eager state-maintenance launches occur once per update interval rather
  than once per token.
- Sliding-window, encoder, ALiBi, attention-sink, soft-capped, quantized-native
  KV, speculative decode, and DCP paths remain native or are rejected when a
  lossless fallback is not available. Tensor parallelism and hybrid recurrent
  layers do not alter the per-layer LOD contract.

Every eligible global layer uses external LOD for prefill and decode, including
short requests. A captured graph never swaps in native attention and no native
chronological cache exists for such a layer.

## Paper-oriented kernel surface

The primary two-tier implementation has one prefill route and two decode
backends. Precision changes are compile-time storage specializations, not
different LOD algorithms.

Prefill processes a 4,096-token chunk in three stages:

1. A fused routing/coarse-attention kernel scores state centroids, retains the
   top eight routes, and produces the unopened-centroid residual branch.
2. The expert-major leaf kernel attends each routed query group to the exact
   posting list of its centroid, without compacting or copying K/V.
3. A stable-LSE reduction merges the eight exact-route results with the coarse,
   local-window, and protected-token branches.

Decode first performs the same centroid routing/coarse calculation. It then
uses either the generic split-8 exact/local kernel or the optional gfx942
H=256/GQA-4 HIP specialization described above, and finally performs one
stable-LSE merge with the coarse branch. The compatibility profile does not
duplicate these algorithms; it only freezes older dispatch/reduction settings.

The paper implementation intentionally removed the unused combined
cooperative Triton kernel and the slower cooperative Triton fallback. This
deleted over 900 lines of kernel and fallback-dispatch code. Generic decode
scratch no longer reserves the
specialized GQA partial buffers. A 64K, batch-8 Qwen3.5-0.8B validation after
the deletion produced identical top-1 outputs in both modes; after compilation,
the specialized and generic full-model decode steps were 14.14 ms and 14.55 ms,
respectively. The focused numerical verifier matched specialized BF16 output
to the generic/reference result within 7.4e-4 maximum absolute error and INT8
within 8.6e-4.

## Current memory behavior

External semantic ownership replaces full chronological K/V for eligible global
layers. With prefix caching, an all-global model retains its real logical cache
geometry in the scheduler for token hashes, while the worker removes the GPU
allocation and attention path entirely. Hybrid models reuse an existing native
group's hashes and need no tracker. Without prefix caching, no scheduler group
or worker tensor is needed for those layers. Recurrent and sliding-window caches
remain native. Semantic leaves are
quantized only after region assignment; sequential native blocks are never
treated as quantizable pages.

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

The current flat two-tier BF16 port was separately checked after enabling raw
routing and the 4,096/4,864/4,096 prefill schedule. It also scored 8/8 on the
same 8K NIAH-S3 batch, with direct LOD prefill and captured LOD decode both
exercised. See
`artifacts/vllm_lod_quality/qwen35_0p8b_two_tier_raw_bf16_niah_s3_8k_s8.json`.

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

The flat physical-page INT8 path now allocates the persistent vLLM pool as
signed INT8 and retains its per-token K/V scales. Before this fix, only the
transient prefill cache was INT8: installation copied its integer codes into a
BF16 destination and discarded the scales. Results from that broken path did
not measure valid INT8 attention.

The corrected uncapped Qwen3.5-0.8B batch-8 measurements below use 16K chunks
per request, a 128K aggregate scheduler budget, M=16/N=32 leaf tiles, two
warps, and 4,096-token state updates. The 32K pair disables asynchronous
scheduling so both precisions execute the same 78 direct-prefill calls. The
64K pair has identical scheduler diagnostics. INT8 uses BF16 PV below 64K and
INT8 PV at 64K, selected automatically.

| Context | BF16 prefill | INT8 prefill | INT8 change | BF16 LOD cache | INT8 LOD cache |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 1.791 s | 1.798 s | 0.4% slower | 9.05 GB | 5.03 GB |
| 64K | 4.089 s | 3.952 s | 3.3% faster | 14.56 GB | 7.95 GB |

Thus the remaining 32K difference is effectively noise-sized, while the
longer posting-list scan amortizes probability quantization at 64K. Cache
storage falls by 44.4% and 45.4%, respectively; BF16 routing state and page
metadata prevent the total from reaching exactly 50%. The optimized INT8
kernel retains NIAH-S3 accuracy at 64/64. On eight 8K ProLong examples it has
token CE 1.924265 and perplexity 6.850114, versus 1.923501--1.923557 for the
matched BF16 checks.

Recursive cache-native INT8 uses signed 8-bit page-mean residuals for both K and V, with
one BF16 scale per page and 32 channels. At 8K and batch eight on
Qwen3.5-0.8B, five warm 1,025-token runs measured 0.418 s prefill and 3.677 ms
per decode batch step. That is within 0.3% of BF16 decode, 3.1% faster than
INT4 decode, and 7.6% faster than INT4 prefill. Its 1.341 GB LOD cache was
30.2% smaller than BF16. See
`artifacts/vllm_lod_speed/INT8_8K_B8_20260817.md` for the paired results.

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
