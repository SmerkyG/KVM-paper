# RULER NIAH-S3

NIAH-S3 is the single-key, UUID-valued RULER needle task. The public runner uses
the generator from `lm-eval==0.4.12`, essay haystacks, word keys, UUID values,
the canonical random seeds, each model's chat template, and thinking disabled.
A sample is correct when the generated response contains the target UUID.

## Results

Every row below uses 128 examples at each context length. LoD uses the release's
uniform top-4 policy after 16K. At 8K and 16K, decode instead scans every leaf
in the authoritative LoD cache, matching the exact first-prefill-chunk policy
and removing short-context routing misses.

| Model / mode | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| Qwen3.8 full | 128/128 | 128/128 | 128/128 | 128/128 |
| Qwen3.8 two-tier BF16 | 128/128 | 128/128 | 125/128 | 127/128 |
| Qwen3.8 three-tier BF16 | 128/128 | 128/128 | 120/128 | 126/128 |
| Qwen3.8 three-tier INT4 | 128/128 | 128/128 | 121/128 | 127/128 |
| K2 Horizon full | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon two-tier BF16 | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon three-tier BF16 | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon three-tier INT4 | 128/128 | 128/128 | 128/128 | 128/128 |

For Qwen, all 128 generated responses are byte-for-byte identical to full
attention at both 8K and 16K in every LoD mode. This is stronger than target
matching alone. The nominal 16K samples contain about 16.10K prompt tokens, so
their complete 64-token generations remain inside the 16,384-token exact
decode boundary.

## Reproduce

Install dependencies:

```bash
uv sync --extra vllm --extra benchmarks
```

Run Qwen over 8K through 64K with offline vLLM:

```bash
uv run python -m benchmarks.niah_s3 \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --mode two-tier \
  --lengths 8192,16384,32768,65536 \
  --samples 128 \
  --batch-size 8 \
  --tensor-parallel-size 1 \
  --max-new-tokens 64 \
  --output results/niah-s3-qwen-two-tier.json
```

Run the matched K2 test:

```bash
uv run python -m benchmarks.niah_s3 \
  --checkpoint IFM/K2-Horizon-32B-FP8 \
  --mode two-tier \
  --lengths 8192,16384,32768,65536 \
  --samples 128 \
  --batch-size 4 \
  --tensor-parallel-size 1 \
  --max-new-tokens 64 \
  --output results/niah-s3-k2-two-tier.json
```

Use `--mode full` for the native control, or select `three-tier-bf16` or
`three-tier-int4` for the other release modes. Each invocation loads one model
and one cache organization, then evaluates all requested lengths.
