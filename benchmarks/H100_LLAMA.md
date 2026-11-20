# Llama-3.1-8B-Instruct on H100

The CUDA port keeps vLLM's native Llama projections, RoPE, normalization, and
MLP. LoD replaces attention and owns the persistent KV cache for all 32 layers.
It supports two-tier BF16 and three-tier BF16/INT4. Three-tier decode opens two
16-token semantic pages per routed region. Prefill refines all leaves in the
eight selected regions, as in the reference release.

See [the matched timing and validation results](H100_LLAMA_RESULTS.md).

The RetroInfer benchmark adapter also supports batch-one three-tier threshold
decode with BF16 or INT4 pages. It uses uncapped per-region c=16 routing,
persistent top-two-page workers and cached region means maintained at normal
cache-update boundaries. See the [AMD-to-H100 three-tier report](../results/h100-baselines/three-tier-transfer/REPORT.md)
for validation, amortized timings, retrieval checks and enable flags. This
experimental adapter keeps top-eight prefill and the 16,384-token exact initial
chunk; the vLLM serving defaults below retain fixed top-eight routing.

The [adaptive page-refinement investigation](../results/h100-baselines/three-tier-repair/REPORT.md)
also implements BF16 page-mass thresholds in the RetroInfer adapter. It uses
RMS-normalized queries for page selection and native queries for attention,
with no fixed page-count cap or CPU count readback. It recovers the targeted
multikey misses and is faster than fixed four-page refinement, while fixed
two-page decode remains faster. The report records quality, work counts,
amortized timings, precision ablations and the experimental enable flags.

The experimental two-tier fused threshold-prefill adapter also has
[tiled prefix-scan selection/packing](../results/h100-baselines/selectpack/REPORT.md).
For the conservative coarse/correction variant, automatic dispatch uses 32-query × 256-region tiles
and four warps at 4,096 or more active region slots. At 2,048–4,095 slots,
the [prefix-filter selection pass](../results/h100-baselines/selectpack-64k/REPORT.md)
reuses the producer's running routing mass to skip candidates that cannot
qualify; below 2,048 slots it retains the original pass. It opens every
region satisfying the existing c=16 rule;
there is no sorting, selection cap or selected-count CPU readback. Set
both `LOD_H100_FAST_SCAN_SELECT=0` and `LOD_H100_FAST_PREFIX_SELECT=0`
to select the original packing implementation.
This optimization leaves the exact 16,384-token initial prefill and the
vLLM serving configuration unchanged. The separate direct-closed-summary
projection remains experimental. The [normalization follow-up](../results/h100-baselines/closed-offset/REPORT.md)
now recovers aggregate CWE quality while retaining direct projection speed:
one FP16 coarse projection with FP32 output followed by FP32 normalization.
The model and KV storage stay BF16. Enable it with
`LOD_H100_FAST_PREFIX_SELECT=1`, `LOD_H100_FAST_PREFIX_CLOSED=1`,
`LOD_H100_FAST_CLOSED_BLAS=1`, `LOD_H100_CLOSED_NORMALIZE_M=1`,
`LOD_H100_CLOSED_ARITHMETIC=direct`, `LOD_H100_CLOSED_PRECISION=scaled-fp16`
and `LOD_H100_CLOSED_VALUE_CENTER=0`, with producer tiles 64 × 128 and
four warps. Recorded commands include the remaining threshold-adapter flags.
The same configuration passes [32K/128K scaling checks](../results/h100-baselines/closed-offset/REPORT.md#context-length-scaling);
128,000-token prefill fits on the 80GB H100 with the exact initial 16K field
unchanged, peaking at 66.4 GiB in PyTorch allocations.

The [current threshold optimization](../results/h100-baselines/threshold-next/REPORT.md#latest-optimized-threshold-prefill) adds a tensor-core reduction for small exact frontiers, matching cache-key normalization and packed max/index reductions, and adaptive scratch/pipeline settings. Fresh full-prefill medians are **1.663 s at 32,768 tokens**, **4.518 s at 65,536 tokens** and **11.528 s at 128,000 tokens**, versus dense’s **1.553 s / 4.749 s / 15.697 s**. Threshold uses 4.9% less time at 64K and 26.6% less at 128K; it remains 7.1% slower at 32K. All cache updates/completion, local/sink attention and the exact 16K initial field are included; prefill is eager for every arm. Peak PyTorch allocation, including weights, is 64.20/70.68 GiB at 32K/64K; the larger scratch tiles trade memory for speed.

The full 32-example CWE check scores **35.94% versus 34.38%** for the matched previous-selector control; individual scores change in both directions. Six retrieval/QA checks retain 100%. Sampled frontier and same-cache eager/graph checks pass. This is not a new 13-task RULER average. The [optimization retime report](../results/h100-baselines/direct-opt-v3/REPORT.md) records the experimental overrides, independent numerical checks, rejected trials and raw timings. Serving defaults are unchanged. At 128,000 tokens use `LOD_H100_FAST_TILE=4096` throughout for memory headroom; this changes scratch size, retaining the initial exact 16K field. The 128K result is 2.4% faster than the preceding 11.816 s threshold version and peaks at **66.44 GiB** in PyTorch allocations, with 224 sampled frontier checks and exact same-cache eager/graph logits. No new 128K quality panel has been run.

The previous online-compatible two-projection combination remains in the
[arithmetic report](../results/h100-baselines/selectpack-64k/COMBINATION.md)
as a slower numerical ablation. Neither path becomes a serving default.

## Runtime

Use Python 3.12 and the CUDA wheels pinned in `h100-requirements.txt`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r benchmarks/h100-requirements.txt -e .
export LOD_PYTHON="$PWD/.venv/bin/python"
```

Download the official `meta-llama/Llama-3.1-8B-Instruct` checkpoint with an
account that has accepted its model license. Pass its local snapshot directory
as `CHECKPOINT` below. Weights and model computations stay BF16; INT4 refers to
the remote semantic-page KV residuals.
The measurements use model revision `0e9e39f249a16976918f6564b8830bc894c89659`.

On the prepared H100 host, the source is `/home/ubuntu/lod-h100/code`, the
environment is `/home/ubuntu/lod-h100/.venv`, and results are in
`/home/ubuntu/lod-h100/results`. Use the existing environment:

```bash
cd /home/ubuntu/lod-h100/code
export LOD_PYTHON=/home/ubuntu/lod-h100/.venv/bin/python
export CHECKPOINT=/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
```

The tested runtime is vLLM 0.27.1, PyTorch 2.13.0+cu130, Triton 3.7.1,
and NVIDIA driver 580.196.07.

The H100 path uses FlashAttention-3 for exact causal/local attention, with
native GQA4 and bottom-right alignment for suffix queries. A fused Triton
producer streams centroid tiles through tensor cores, produces the coarse
partition, retains eight routing winners in registers, and emits expert-packing
offsets. It does not materialize a query-by-centroid score matrix. Existing
cache, page-refinement, and decode kernels are shared with the reference;
AMD occupancy options are supplied only on HIP.

Llama has no Q/K normalization layers. Semantic assignment therefore uses
cosine similarity, and routing scores use `q/RMS(q)` plus the log count.
Coarse mass, exact attention, and frontier replacement always use native `q`.
Decode recomputes the eight selected native logits before replacing their
coarse mass, so routing normalization never changes attention's softmax.

The H100 defaults use 64-query by 64-centroid prefill tiles with four warps.
Decode uses 64-centroid groups with two warps and a parallel four-warp
partition reduction. These settings apply to BF16 and INT4 without changing
the region or page budgets.

## Prefill kernel tuning

Warm CUDA-graph timings include mean preparation, coarse attention, top-eight
routing, and packing offsets. Shapes are one request, 16,384 queries, 32 query
heads, eight KV heads, and head dimension 128. Every tested tile configuration
matched the verified default's routing set, coarse output, and partition mass.

| Centroids | Initial 32×128 tile | Selected 64×64 tile | Speedup |
|---|---:|---:|---:|
| 2,048 | 25.738 ms | 6.636 ms | 3.88× |
| 4,096 | 51.081 ms | 13.358 ms | 3.82× |

These are kernel speedups within the CUDA port. End-to-end model timings also
include the model's other layers and LoD cache maintenance.

## Serving

The ordinary vLLM OpenAI server loads the same plugin and backend:

```bash
VLLM_PLUGINS=lod_attention VLLM_LOD_MODE=three-tier-int4 VLLM_LOD_POOL_SIZE=1 \
  uv run --no-project --python "$LOD_PYTHON" vllm serve "$CHECKPOINT" \
  --host 127.0.0.1 --port 8000 --attention-config '{"backend":"CUSTOM"}' \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --max-model-len 131072 \
  --max-num-seqs 1 --max-num-batched-tokens 16385 \
  --long-prefill-token-threshold 16384 \
  --scheduler-cls vllm_lod_plugin.scheduler.LODChunkAlignedScheduler \
  --no-enable-prefix-caching --gpu-memory-utilization 0.8
```

Select `three-tier-bf16` or `two-tier` through `VLLM_LOD_MODE`.

## Logged experiments

On the paper fleet, use its existing `cluster-run`. On a dedicated single-GPU
SSH host, `benchmarks/h100_cluster_run.py` provides a small adapter with the same
submission convention. It snapshots source at submission, records SHA256 file
hashes, command and GPU identity, and serializes jobs using a GPU lock. Its
scope is one host and GPU 0; it is not a fleet scheduler.

```bash
alias cluster-run='uv run --no-project --python "$LOD_PYTHON" python -m benchmarks.h100_cluster_run'
cluster-run --help
cluster-run --name llama-smoke --detach --num-gpus 1 -- \
  uv run --no-project --python "$LOD_PYTHON" python -m benchmarks.h100_llama \
  --checkpoint "$CHECKPOINT" --mode three-tier-bf16 --smoke \
  --max-model-len 32768 --output "$PWD/results/h100/smoke.json"
cluster-run status
```

The command runs from the frozen snapshot, so use absolute checkpoint, Python,
and output paths. Logs, source, and `run.json` are retained under
`~/.local/state/lod-cluster-run/<run-id>/`. `cluster-run logs RUN_ID` reads the
log; `cluster-run cancel RUN_ID` stops a job.

For matched timing, run each of `full`, `two-tier`, `three-tier-bf16`, and
`three-tier-int4` with the same capacity, lengths, batch size, and memory target:

```bash
cluster-run --name llama-speed --detach --num-gpus 1 -- \
  uv run --no-project --python "$LOD_PYTHON" python -m benchmarks.h100_llama \
  --checkpoint "$CHECKPOINT" --mode three-tier-bf16 \
  --lengths 8192,32768,65536 --batch-size 1 --decode-tokens 1025 \
  --max-model-len 66577 --warmups 2 --repeats 3 \
  --dataset-file "$PROLONG_SHARD" --prompt-manifest "$PWD/results/h100/prompts-b1.json" \
  --output "$PWD/results/h100/three-tier-bf16-b1.json"
```

`PROLONG_SHARD` is the absolute path to `train_00000.parquet` from ProLong
revision `97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`. This H100 comparison
selects that shard explicitly; it does not reuse the paper fleet's full-stream
shuffle. Omit `--dataset-file` to use the original full stream, or repeat it to
select more pinned local shards. Record the same selection for every mode.

Prompts use the existing ProLong preparation, with exact token counts,
distinct documents, a repetition guard, and recorded token hashes. Decode
timing includes periodic 256-token cache catch-up. Two warmups precede three
measurements per length; startup and first-use compilation are reported
separately from warm timings. `full` disables the LoD plugin and uses the native
vLLM FlashAttention backend with the same scheduling budget.
Prompt preparation finishes before model startup. The manifest freezes the
cohort for every mode and validates its tokenizer, shard hashes, batch size,
token lengths, and token digests before reuse.

To reproduce the window-capacity point, use `--lengths 128000 --max-model-len 131072`
with a separate prompt manifest. This leaves room for all 1,025 output tokens.

After downloading the raw results and provenance, regenerate the comparison:

```bash
uv run --no-project python -m benchmarks.h100_report \
  --results results/h100-llama31/raw --output benchmarks/H100_LLAMA_RESULTS.md
```

`--retrieval --lengths 8192,32768,65536` instead runs deterministic passkey
checks at 10%, 50%, and 90% depth. These are a serving sanity check, not a full
RULER quality evaluation. `--smoke` checks short instruction-following responses.
`--retrieval-cohort --batch-size 8 --lengths 8192,32768` submits eight requests
together with distinct passkeys and mixed lengths; use a 33,809-token capacity
and the same 90% memory target as the batch-eight timing panel.

CUDA correctness checks include independent dense references for coarse mass,
top-eight routing, local causal alignment, and the disjoint top-two page
frontier, including INT4 reconstruction and INT8 summaries:

```bash
cluster-run --name llama-correctness --detach --num-gpus 1 -- \
  env LOD_RUN_GPU_TESTS=1 uv run --no-project --python "$LOD_PYTHON" \
  python -m pytest -q tests/test_cuda_prefill.py \
  'tests/test_page_opening.py::test_top_pages_match_disjoint_frontier[bf16-False-None-128-4]' \
  'tests/test_page_opening.py::test_top_pages_match_disjoint_frontier[bf16-True-None-128-4]' \
  'tests/test_page_opening.py::test_top_pages_match_disjoint_frontier[int8-False-None-128-4]' \
  'tests/test_page_opening.py::test_top_pages_match_disjoint_frontier[int8-True-None-128-4]' \
  tests/test_page_opening.py::test_top2_preserves_ties_and_runner_up_across_scan_blocks
```

Batch and context capacity must fit the H100's 80GB together with model weights,
LoD pools, and prefill workspaces. An eight-request 64K BF16 configuration needs
a separate capacity check; a one-request result does not establish that it fits.
