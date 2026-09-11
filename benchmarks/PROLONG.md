# ProLong quality and speed

ProLong serves two purposes here. Prompt log-probabilities measure the
token-weighted cross-entropy and perplexity of long natural text. A separate
generation sweep measures prefill wall time and decode latency on exact-length
natural-text prompts.

## Prompt-loss results

The table uses eight 65,536-token documents from the deterministic shuffled
cohort at offset 8 in `Seerkfang/prolong-64k-512-new`, revision
`97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`.

| Model | Cache | Prefill routes | Loss | Perplexity |
|---|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | Full | all | 0.801509 | 2.228902 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 4 | 0.804913 | 2.236501 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 4 | 0.804743 | 2.236123 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 4 | 0.804831 | 2.236319 |
| K2-Horizon-32B-FP8 | Full | all | 0.521601 | 1.684723 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 4 | 0.524196 | 1.689101 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 4 | 0.524244 | 1.689181 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 4 | 0.524302 | 1.689279 |

Prompt loss exercises prefill only. Every LoD row above uses the release's
uniform top-4 prefill policy. Each measurement contains 524,280 predicted
tokens; the loss is token-weighted across all eight documents.

## Matched speed results

These measurements were freshly collected on 2026-09-11 on AMD MI325X with
vLLM 0.27.1. Each cell is
`prefill seconds / decode milliseconds per batch step`; lower is better. One
B8 decode step emits eight tokens concurrently.

Prompts are raw, distinct ProLong token streams, concatenating different
documents when needed and never repeating a document to fill a request. The
older internal summary called these chat-formatted, but its retained prompt
metadata and runner show that raw `prompt_token_ids` were used. The scheduler
chunk is 16,384 tokens. Decode generates 1,025 tokens and measures the final
1,024 steps. This includes four 256-token LoD state updates in the regular
path, or two 512-token updates for K2 INT4. Results are medians after one
warmup and three measured repetitions.

### Qwen3.8, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.998 s / 28.82 ms | 0.913 s / 29.74 ms | 0.935 s / 30.94 ms | 0.922 s / 35.48 ms |
| 16K | 2.172 s / 29.56 ms | 1.854 s / 28.73 ms | 1.893 s / 29.48 ms | 1.944 s / 29.62 ms |
| 32K | 5.214 s / 30.26 ms | 3.855 s / 28.81 ms | 3.923 s / 29.52 ms | 4.054 s / 29.57 ms |
| 64K | 13.941 s / 31.65 ms | 8.020 s / 29.00 ms | 8.151 s / 29.63 ms | 8.468 s / 29.88 ms |
| 128K | 42.227 s / 34.34 ms | 16.716 s / 29.35 ms | 16.973 s / 29.71 ms | 17.718 s / 29.81 ms |

### Qwen3.8, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.500 s / 22.07 ms | 3.376 s / 24.73 ms | 3.557 s / 26.53 ms | 3.638 s / 32.70 ms |
| 16K | 7.681 s / 23.01 ms | 7.007 s / 22.04 ms | 7.466 s / 22.46 ms | 7.578 s / 22.61 ms |
| 32K | 17.365 s / 24.23 ms | 14.703 s / 22.21 ms | 14.838 s / 22.55 ms | 15.071 s / 22.71 ms |
| 64K | 43.194 s / 27.17 ms | 30.027 s / 22.65 ms | 30.137 s / 22.63 ms | 30.692 s / 22.84 ms |
| 128K | 119.801 s / 32.72 ms | 61.561 s / 23.55 ms | 61.838 s / 22.89 ms | 63.410 s / 23.11 ms |

### K2 Horizon, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.155 s / 37.42 ms | 1.192 s / 41.95 ms | 1.290 s / 41.28 ms | 1.314 s / 53.21 ms |
| 16K | 2.603 s / 38.36 ms | 2.550 s / 39.94 ms | 2.694 s / 42.14 ms | 2.870 s / 41.05 ms |
| 32K | 6.310 s / 38.92 ms | 6.600 s / 40.58 ms | 6.816 s / 42.52 ms | 7.320 s / 41.54 ms |
| 64K | 17.095 s / 40.81 ms | 16.316 s / 41.00 ms | 16.832 s / 42.86 ms | 18.198 s / 41.91 ms |
| 128K | 52.783 s / 44.36 ms | 39.077 s / 42.15 ms | 40.511 s / 43.69 ms | 44.506 s / 42.57 ms |

### K2 Horizon, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.816 s / 22.58 ms | 5.001 s / 27.62 ms | 5.208 s / 26.93 ms | 5.290 s / 45.84 ms |
| 16K | 8.393 s / 23.53 ms | 10.765 s / 24.90 ms | 10.827 s / 27.83 ms | 10.964 s / 26.13 ms |
| 32K | 19.223 s / 25.14 ms | 25.036 s / 25.66 ms | 24.458 s / 28.28 ms | 25.245 s / 26.65 ms |
| 64K | 49.130 s / 28.60 ms | 58.143 s / 26.92 ms | 57.535 s / 29.26 ms | 60.419 s / 27.74 ms |
| 128K | 139.249 s / 35.44 ms | 134.425 s / 28.75 ms | 135.645 s / 31.49 ms | 144.221 s / 30.34 ms |

### Qwen3.8 with DFlash2, TP1, batch 1

This panel uses `z-lab/Qwen3.8-27B-DFlash2` with seven proposed tokens. The
prefill column remains target-model prefill; the decode column measures the
complete speculative target-and-draft loop.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.004 s / 8.23 ms | 0.914 s / 11.01 ms | 0.924 s / 9.06 ms | 0.932 s / 8.27 ms |
| 16K | 2.251 s / 6.45 ms | 1.893 s / 6.20 ms | 1.934 s / 6.37 ms | 1.978 s / 6.09 ms |
| 32K | 5.358 s / 7.03 ms | 3.939 s / 8.47 ms | 4.008 s / 7.19 ms | 4.121 s / 8.08 ms |
| 64K | 14.221 s / 10.73 ms | 8.201 s / 10.91 ms | 8.329 s / 7.81 ms | 8.615 s / 7.04 ms |
| 128K | 42.993 s / 12.96 ms | 17.078 s / 9.35 ms | 17.338 s / 7.24 ms | 18.047 s / 6.83 ms |

At 8K, the two BF16 LoD modes use their exact retained-leaf path. DFlash2's
INT4 target keeps routed attention because its one-token and multi-token
verification graphs share one cache pool; this avoids reserving an
incompatible exact-scan graph while preserving the production top-4 policy.

## Reproduce prompt quality

Install dependencies:

```bash
uv sync --extra vllm --extra benchmarks
```

Run the Qwen two-tier measurement:

```bash
uv run python -m benchmarks.prolong \
  --measure quality \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --mode two-tier \
  --length 65536 \
  --samples 8 \
  --sample-offset 8 \
  --batch-size 1 \
  --tensor-parallel-size 1 \
  --output results/prolong-qwen-two-tier-quality.json
```

For K2, replace the checkpoint with `IFM/K2-Horizon-32B-FP8`. Select
`--mode full`, `three-tier-bf16`, or `three-tier-int4` for the other arms. Run
one process per arm so each process owns one unambiguous cache organization.

## Reproduce speed

The TP1, batch-1 sweep is:

```bash
uv run python -m benchmarks.prolong \
  --measure speed \
  --checkpoint Qwen/Qwen3.8-27B-FP8 \
  --mode two-tier \
  --lengths 8192,16384,32768,65536,131072 \
  --batch-size 1 \
  --tensor-parallel-size 1 \
  --decode-tokens 1025 \
  --repeats 3 \
  --gpu-memory-utilization 0.7 \
  --output results/prolong-qwen-two-tier-speed-tp1-b1.json
```

For TP4, batch 8, change `--batch-size` to 8 and
`--tensor-parallel-size` to 4. Full attention defaults to the archived
`ROCM_AITER_UNIFIED_ATTN` control backend. The output records each repetition,
prompt hashes, aggregate prefill throughput, decode batch-step latency, and
decode token throughput.

The LoD command leaves additional VRAM outside vLLM's native-cache allocator
for concurrent cache-maintenance workspaces. Qwen defaults to 0.7. K2 defaults
to 0.8 because its larger model-side 131K LoD pool otherwise leaves no native
cache blocks. Use `--gpu-memory-utilization 0.9` for the full-attention control,
which has no separate LoD pool.

To reproduce the DFlash2 panel, add:

```bash
  --speculative-model z-lab/Qwen3.8-27B-DFlash2 \
  --num-speculative-tokens 7
```

to the Qwen speed command. DFlash2 is supported only for Qwen3.8 in this
release.
