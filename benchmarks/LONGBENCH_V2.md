# LongBench v2

This benchmark measures end-to-end, prefill-heavy serving quality on all 503
examples in `THUDM/LongBench-v2`. The evaluator talks to an OpenAI-compatible
vLLM server, caps input at 131,072 tokens by retaining the first and last
halves, disables model thinking, and constrains the answer to A/B/C/D.

## Results

These results were freshly collected on 2026-09-11 with vLLM 0.27.1 on AMD
MI325X. Both LoD phases routed exactly four centroids. The three-tier modes use
the same semantic pages, stored either in BF16 or as centroid-relative INT4
residuals.

| Model | Full attention | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 265/503 (52.68%) | 270/503 (53.68%) | 262/503 (52.09%) | 267/503 (53.08%) |
| K2-Horizon-32B-FP8 | 207/503 (41.15%) | 209/503 (41.55%) | 208/503 (41.35%) | 212/503 (42.15%) |

| Model / mode | Short (180) | Medium (215) | Long (108) |
|---|---:|---:|---:|
| Qwen full | 100 | 112 | 53 |
| Qwen two-tier BF16 | 97 | 120 | 53 |
| Qwen three-tier BF16 | 98 | 110 | 54 |
| Qwen three-tier INT4 | 102 | 112 | 53 |
| K2 full | 87 | 74 | 46 |
| K2 two-tier BF16 | 86 | 80 | 43 |
| K2 three-tier BF16 | 87 | 76 | 45 |
| K2 three-tier INT4 | 87 | 82 | 43 |

All runs contained 503 unique IDs and every response parsed as A/B/C/D. The
Qwen tokenizer produced 205 truncated prompts; K2 produced 181.

The archived evaluation used eight length-balanced shards on eight GPUs. The
table below reports evaluator wall time per shard as minimum / mean / maximum;
startup and warmup are excluded.

| Model | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8 | 23.38 / 26.88 / 32.22 min | 15.77 / 16.82 / 17.81 min | 16.98 / 19.61 / 22.21 min | 12.25 / 13.35 / 14.94 min |
| K2 Horizon | 27.69 / 31.59 / 38.75 min | 23.19 / 26.34 / 31.29 min | 23.78 / 27.31 / 32.51 min | 25.90 / 29.35 / 34.51 min |

All arms were rerun from the release checkout, but each mode used independent
length-balanced shards rather than an interleaved timing protocol. Treat this
timing as operational context, not as a controlled kernel-speed comparison.
Use the matched sweep in [ProLong](PROLONG.md) for that purpose.

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
  --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 16384 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
  --attention-config '{"backend":"CUSTOM"}'
```

For K2, use `IFM/K2-Horizon-32B-FP8`, set both pool size and
`--max-num-seqs` to 4, use `--gpu-memory-utilization 0.8`, and omit
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
  --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 16384 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --attention-config '{"backend":"ROCM_AITER_UNIFIED_ATTN"}'
```

To reproduce the eight-GPU protocol, run one server/evaluator pair per GPU and
add `--num-shards 8 --shard-index N` for `N=0,...,7`. Use distinct ports and
output files. Accuracy is identical to an unsharded run; sharding changes only
wall time.
