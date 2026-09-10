# RULER NIAH-S3

NIAH-S3 is the single-key, UUID-valued RULER needle task. The public runner uses
the generator from `lm-eval==0.4.12`, essay haystacks, word keys, UUID values,
the canonical random seeds, each model's chat template, and thinking disabled.
A sample is correct when the generated response contains the target UUID.

## Archived results

The clean 64K archive for the finalized uniform top-4 policy contains the full
and two-tier BF16 controls below.

| Model | Full attention | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 8/8 | 8/8 | not archived | not archived |
| K2-Horizon-32B-FP8 | 8/8 | 8/8 | not archived | not archived |

“Not archived” is not a failed result. Earlier three-tier runs also scored 8/8
at 64K, but they used pre-release top-8 decode routing (and, for Qwen, top-8
prefill routing), so they are not substituted for a finalized top-4/top-4 run.

The Qwen full control additionally scored 8/8 at 128K in the archived panel.
The reported 64K Qwen two-tier run used batch 8; K2 used batch 4 because of its
larger per-request cache allocation.

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
  --samples 8 \
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
  --samples 8 \
  --batch-size 4 \
  --tensor-parallel-size 1 \
  --max-new-tokens 64 \
  --output results/niah-s3-k2-two-tier.json
```

Use `--mode full` for the native control, or select `three-tier-bf16` or
`three-tier-int4` for the other release modes. Each invocation loads one model
and one cache organization, then evaluates all requested lengths.
