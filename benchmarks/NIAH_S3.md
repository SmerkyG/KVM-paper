# RULER NIAH-S3

NIAH-S3 is the single-key, UUID-valued RULER needle task. The public runner uses
the generator from `lm-eval==0.4.12`, essay haystacks, word keys, UUID values,
the canonical random seeds, each model's chat template, and thinking disabled.
A sample is correct when the generated response contains the target UUID.

## Results

Every row below uses 128 examples at each context length. All reported LoD
lengths use the release's uniform top-4 routing policy; the retained-leaf exact
decode path ends at 2K.

| Model / mode | 8K | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| Qwen3.8 full | 128/128 | 128/128 | 128/128 | 128/128 |
| Qwen3.8 two-tier BF16 | 127/128 | 125/128 | 125/128 | 127/128 |
| Qwen3.8 three-tier BF16 | 127/128 | 121/128 | 120/128 | 126/128 |
| Qwen3.8 three-tier INT4 | 123/128 | 118/128 | 121/128 | 127/128 |
| K2 Horizon full | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon two-tier BF16 | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon three-tier BF16 | 128/128 | 128/128 | 128/128 | 128/128 |
| K2 Horizon three-tier INT4 | 128/128 | 127/128 | 128/128 | 128/128 |

The 8K and 16K LoD cells were rerun with exact decode disabled at those
lengths; the final release cutoff is 2K. The 32K and 64K cells already used
routed top-4 LoD and are unchanged.

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
