# LongBench v2

This benchmark measures end-to-end, prefill-heavy serving quality on all 503
examples in `THUDM/LongBench-v2`. The evaluator talks to an OpenAI-compatible
vLLM server, caps input at 131,072 tokens by retaining the first and last
halves, disables model thinking, and constrains the answer to A/B/C/D.

## Archived results

These results were collected on 2026-09-09 with vLLM 0.27.1 on AMD MI325X.
Both LoD phases routed exactly four centroids. The three-tier result used BF16
semantic pages. No finalized top-4/top-4 INT4 LongBench run was archived.

| Model | Full attention | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 267/503 (53.08%) | 264/503 (52.49%) | 261/503 (51.89%) | not archived |
| K2-Horizon-32B-FP8 | 209/503 (41.55%) | 208/503 (41.35%) | 213/503 (42.35%) | not archived |

| Model / mode | Short (180) | Medium (215) | Long (108) |
|---|---:|---:|---:|
| Qwen full | 102 | 112 | 53 |
| Qwen two-tier BF16 | 99 | 112 | 53 |
| Qwen three-tier BF16 | 96 | 112 | 53 |
| K2 full | 88 | 74 | 47 |
| K2 two-tier BF16 | 89 | 77 | 42 |
| K2 three-tier BF16 | 84 | 81 | 48 |

All runs contained 503 unique IDs and every response parsed as A/B/C/D. The
Qwen tokenizer produced 205 truncated prompts; K2 produced 181.

The archived evaluation used eight length-balanced shards on eight GPUs. The
table below reports evaluator wall time per shard as minimum / mean / maximum;
startup and warmup are excluded.

| Model | Full | Two-tier BF16 | Three-tier BF16 |
|---|---:|---:|---:|
| Qwen3.8 | 23.5 / 26.9 / 32.0 min | 15.14 / 16.72 / 19.29 min | 10.65 / 11.92 / 13.87 min |
| K2 Horizon | 28.1 / 32.2 / 38.4 min | 23.81 / 27.23 / 33.10 min | 23.17 / 26.29 / 30.99 min |

The full controls were run on 2026-09-06 and the LoD arms on 2026-09-09, not
as an interleaved timing experiment. Treat this timing as operational context,
not as a controlled speedup measurement. Use the matched ProLong sweep for
kernel-speed comparisons.

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
  --gpu-memory-utilization 0.8 \
  --attention-config '{"backend":"CUSTOM"}'
```

For K2, use `IFM/K2-Horizon-32B-FP8`, set both pool size and
`--max-num-seqs` to 4, and omit `--language-model-only`. To test a different
LoD organization, change only `VLLM_LOD_MODE` to `three-tier-bf16` or
`three-tier-int4`.

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
  --gpu-memory-utilization 0.8 \
  --attention-config '{"backend":"ROCM_AITER_UNIFIED_ATTN"}'
```

To reproduce the eight-GPU protocol, run one server/evaluator pair per GPU and
add `--num-shards 8 --shard-index N` for `N=0,...,7`. Use distinct ports and
output files. Accuracy is identical to an unsharded run; sharding changes only
wall time.
