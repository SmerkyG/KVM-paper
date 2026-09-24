# RULER NIAH-S3

NIAH-S3 is the single-key, UUID-valued RULER needle task. The public runner uses
the generator from `lm-eval==0.4.12`, essay haystacks, word keys, UUID values,
the canonical random seeds, each model's chat template, and thinking disabled.
A sample is correct when the generated response contains the target UUID.

## Results

Every row below uses 128 examples at each context length. Every full-attention
and LoD row was rerun on 2026-09-23 from commit `e248719a`; LoD uses eight
routes in both prefill and decode. The retained-leaf exact decode path ends at
2K, so all displayed lengths exercise routed LoD.

| Model / mode | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| Qwen3.8 full | 128/128 | 128/128 | 128/128 | 128/128 |
| Qwen3.8 two-tier BF16, top-8 | 128/128 | 127/128 | 128/128 | 128/128 |
| Qwen3.8 three-tier BF16, top-8 | 126/128 | 122/128 | 123/128 | 128/128 |
| Qwen3.8 three-tier INT4, top-8 | 128/128 | 122/128 | 126/128 | 125/128 |
| K2 Horizon full | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon two-tier BF16, top-8 | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon three-tier BF16, top-8 | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon three-tier INT4, top-8 | 128/128 | 128/128 | 128/128 | 128/128 |

Every LoD cell uses the current default with the same 2K exact-decode cutoff.
K2 remains 128/128 in every full and LoD cell. Qwen two-tier is near-perfect at
511/512 total, but the current Qwen three-tier rerun regresses at 16K--32K:
BF16 scores 499/512 overall and INT4 scores 501/512. The failures are retained
rather than carrying forward a stronger earlier sample.

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

## Reproduction requirements

Use the locked `lm-eval==0.4.12` RULER generator. The runner fixes Python's
generator seed to `0` and NumPy's generator seed to `1234` before constructing
every length; these are part of the canonical task definition and are not CLI
parameters. Generation is greedy (`temperature=0`), model thinking is disabled,
and the commands above fix the sample count, offset, batch size, and output
limit. Run each cache mode in a fresh process against the same model revision.

As with the ProLong runs, FP8 GPU execution is not guaranteed to be bitwise
deterministic even with fixed prompt-generation seeds. Preserve the JSON sample
records, rather than only the aggregate score, when checking a reproduction.
