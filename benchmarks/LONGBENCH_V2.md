# LongBench v2

This benchmark measures end-to-end, prefill-heavy serving quality on all 503
examples in `THUDM/LongBench-v2`. The evaluator talks to an OpenAI-compatible
vLLM server, caps input at 131,072 tokens by retaining the first and last
halves, disables model thinking, and constrains the answer to A/B/C/D.

## Results

These results were freshly collected on 2026-09-23 from commit `e248719a`
with vLLM 0.27.1 on AMD MI325X. Both LoD phases route exactly eight centroids.
The three-tier modes use the same semantic pages, stored either in BF16 or as
centroid-relative INT4 residuals.

| Model | Full attention | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 262/503 (52.09%) | 274/503 (54.47%) | 269/503 (53.48%) | 265/503 (52.68%) |
| K2-Horizon-32B-FP8 | 210/503 (41.75%) | 214/503 (42.54%) | 214/503 (42.54%) | 211/503 (41.95%) |

| Model / mode | Short (180) | Medium (215) | Long (108) |
|---|---:|---:|---:|
| Qwen full | 97 | 111 | 54 |
| Qwen two-tier BF16 | 106 | 116 | 52 |
| Qwen three-tier BF16 | 103 | 109 | 57 |
| Qwen three-tier INT4 | 101 | 110 | 54 |
| K2 full | 89 | 74 | 47 |
| K2 two-tier BF16 | 85 | 83 | 46 |
| K2 three-tier BF16 | 89 | 80 | 45 |
| K2 three-tier INT4 | 86 | 80 | 45 |

All runs contained 503 unique IDs and every response parsed as A/B/C/D. The
Qwen tokenizer produced 205 truncated prompts; K2 produced 181.

The evaluation used eight length-balanced shards on eight GPUs. The table
below reports evaluator wall time per shard as minimum / mean / maximum;
startup and warmup are excluded. Four K2 two-tier shards were rerun cleanly
from scratch after allocator fragmentation invalidated their original resumed
wall times.

| Model | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8 | 23.40 / 26.91 / 32.29 min | 13.26 / 16.31 / 20.66 min | 12.95 / 14.49 / 15.76 min | 12.05 / 13.30 / 14.69 min |
| K2 Horizon | 28.14 / 32.40 / 38.97 min | 23.95 / 27.31 / 32.25 min | 24.25 / 27.76 / 33.30 min | 26.73 / 30.63 / 36.18 min |

All arms were rerun from the current release checkout, but each mode used
independent length-balanced shards rather than an interleaved timing protocol.
Treat this timing as operational context, not as a controlled kernel-speed comparison.
Use the matched sweep in [ProLong](PROLONG.md) for that purpose.

The K2 three-tier INT4 result remains valid for the standardized 256-token
decode update interval. LongBench v2 generates at most 32 tokens per request,
so it never reaches the first cache catch-up under either the former 512-token
interval or the current 256-token interval; its prefill calculation and decoded
outputs are unchanged.

## Reproduce

Install the serving and benchmark dependencies from the repository root:

```bash
uv sync --extra vllm --extra benchmarks
```

Apply and build the AITER patch described in the root README before timing LoD.
Start a Qwen two-tier server:

```bash
VLLM_PLUGINS=lod_attention \
VLLM_LOD_MODE=two-tier \
VLLM_LOD_POOL_SIZE=8 \
uv run vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --model-impl vllm \
  --renderer-num-workers 8 \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131200 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16392 \
  --long-prefill-token-threshold 16384 \
  --scheduler-cls vllm_lod_plugin.scheduler.LODChunkAlignedScheduler \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
  --attention-config '{"backend":"CUSTOM"}'
```

For K2, use `IFM/K2-Horizon-32B-FP8`, set both pool size and
`--max-num-seqs` to 4, set `--max-num-batched-tokens 16388`, use
`--gpu-memory-utilization 0.8`, set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and omit
`--language-model-only`. The higher target is required because K2's larger
model-side 131K LoD pool otherwise leaves vLLM no native cache blocks. To test
a different LoD organization, change only `VLLM_LOD_MODE` to
`three-tier-bf16` or `three-tier-int4`.

Run the complete evaluator against that server:

```bash
uv run python -m benchmarks.longbench_v2 \
  --base-url http://127.0.0.1:8000/v1 \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --output results/longbench-qwen-two-tier.jsonl \
  --max-input-tokens 131072 \
  --max-output-tokens 32 \
  --workers 8 \
  --warmup-batches 1
```

The JSONL is resumable. A compact aggregate is written beside it as
`results/longbench-qwen-two-tier.summary.json`.

Keep vLLM prefix caching disabled for timed runs. The evaluator deliberately
warms the first measured prompts so their exact request shapes are compiled;
prefix caching would instead turn that first measured batch into cache hits.
The LoD server uses a lower native-cache utilization target because its
authoritative semantic cache is allocated separately; the remaining headroom
also covers concurrent cache-maintenance workspaces on ragged long batches.

For the full-attention control, restart the server without `VLLM_LOD_MODE` and
select the same ROCm backend used for the archived result:

```bash
VLLM_PLUGINS=lod_attention \
uv run vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --model-impl vllm \
  --renderer-num-workers 8 \
  --language-model-only \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 131200 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16392 \
  --long-prefill-token-threshold 16384 \
  --scheduler-cls vllm_lod_plugin.scheduler.LODChunkAlignedScheduler \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"backend":"ROCM_AITER_UNIFIED_ATTN"}'
```

To reproduce the eight-GPU protocol, run one server/evaluator pair per GPU and
add `--num-shards 8 --shard-index N` for `N=0,...,7`. Use distinct ports and
output files. Accuracy is identical to an unsharded run; sharding changes only
wall time.

## Reproduction requirements

Use the locked dependency versions (notably vLLM 0.27.1), the dataset revision
recorded above, the same model revision, and the patched AITER build described
in the root README. The evaluator has no randomized sampling or dataset
subsampling: it enumerates all 503 records deterministically, sorts each shard
by tokenized input length, uses `temperature=0`, disables thinking, and retains
the default guided A/B/C/D output constraint. Preserve `--workers 8`, one
warmup batch, the 131,072-token first/last-half truncation rule, and disabled
prefix caching to reproduce the reported timing protocol.

Run modes sequentially on the same otherwise idle MI325X GPU set. Greedy
generation needs no stochastic sampling seed here, but FP8 GEMMs and parallel
GPU reductions are still not guaranteed to be bitwise deterministic. Keep the
per-example JSONL so any changed answer can be distinguished from a timing
change.
