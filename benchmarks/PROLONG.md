# ProLong quality and speed

ProLong serves two purposes here. Prompt log-probabilities measure the
token-weighted cross-entropy and perplexity of long natural text. A separate
generation sweep measures prefill wall time and decode latency on exact-length
natural-text prompts.

## Archived prompt-loss results

The table uses eight 65,536-token documents from the deterministic shuffled
cohort at offset 8 in `Seerkfang/prolong-64k-512-new`, revision
`97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`.

| Model | Cache | Prefill routes | Loss | Perplexity |
|---|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | Full | all | 0.801394 | 2.228646 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 4 | 0.804739 | 2.236113 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 8 (pre-release) | 0.804929 | 2.236537 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 8 (pre-release) | 0.804961 | 2.236610 |
| K2-Horizon-32B-FP8 | Full | all | 0.521465 | 1.684494 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 4 | 0.524221 | 1.689142 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 4 | 0.527488 | 1.694671 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 4 | 0.527762 | 1.695134 |

Prompt loss exercises prefill only. The two Qwen three-tier rows predate the
uniform top-4 lock and are retained as quantization evidence, not mislabeled as
final top-4 measurements. K2's three-tier rows use the final top-4 prefill
calculation but were recorded under the earlier experimental profile name.

## Archived matched speed results

These measurements were collected on AMD MI325X with vLLM 0.27.1. Each cell is
`prefill seconds / decode milliseconds per batch step`; lower is better. One
B8 decode step emits eight tokens concurrently.

Prompts are raw, distinct ProLong token streams, concatenating different
documents when needed and never repeating a document to fill a request. The
older internal summary called these chat-formatted, but its retained prompt
metadata and runner show that raw `prompt_token_ids` were used. The scheduler
chunk is 16,384 tokens. Decode generates 1,025 tokens and measures the final
1,024 steps, thereby including four 256-token LoD state updates. Results are
medians after one warmup and three measured repetitions.

### Qwen3.8, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.968 s / 28.83 ms | 0.918 s / 28.63 ms | 0.930 s / 29.38 ms | 0.970 s / 29.38 ms |
| 16K | 2.159 s / 29.58 ms | 1.871 s / 28.56 ms | 1.885 s / 29.29 ms | 1.924 s / 29.50 ms |
| 32K | 5.185 s / 30.31 ms | 3.894 s / 28.74 ms | 3.896 s / 29.29 ms | 4.012 s / 29.61 ms |
| 64K | 13.868 s / 31.69 ms | 8.114 s / 28.93 ms | 8.092 s / 29.32 ms | 8.395 s / 29.65 ms |
| 128K | 42.275 s / 34.41 ms | 17.163 s / 29.25 ms | 16.866 s / 29.36 ms | 17.584 s / 29.54 ms |

### Qwen3.8, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.530 s / 22.05 ms | 3.388 s / 21.97 ms | 3.624 s / 22.97 ms | 3.670 s / 23.07 ms |
| 16K | 7.663 s / 22.97 ms | 6.966 s / 21.98 ms | 7.449 s / 23.03 ms | 7.516 s / 23.18 ms |
| 32K | 17.291 s / 24.18 ms | 15.184 s / 22.21 ms | 15.409 s / 23.08 ms | 15.572 s / 23.22 ms |
| 64K | 43.076 s / 27.12 ms | 32.335 s / 22.61 ms | 32.676 s / 23.13 ms | 33.060 s / 23.21 ms |
| 128K | 119.626 s / 32.68 ms | 69.634 s / 23.50 ms | 70.133 s / 23.22 ms | 71.089 s / 23.35 ms |

### K2 Horizon, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.156 s / 37.67 ms | 1.161 s / 39.84 ms | 1.300 s / 42.85 ms | 1.405 s / 42.12 ms |
| 16K | 2.598 s / 38.32 ms | 2.514 s / 40.00 ms | 2.721 s / 43.00 ms | 2.771 s / 42.18 ms |
| 32K | 6.347 s / 39.29 ms | 6.534 s / 40.13 ms | 6.876 s / 43.04 ms | 7.177 s / 42.30 ms |
| 64K | 17.181 s / 41.18 ms | 16.160 s / 40.10 ms | 16.903 s / 43.07 ms | 17.904 s / 42.35 ms |
| 128K | 52.876 s / 44.51 ms | 38.792 s / 40.37 ms | 40.701 s / 43.23 ms | 43.599 s / 42.53 ms |

### K2 Horizon, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.851 s / 22.57 ms | 4.995 s / 26.12 ms | 5.538 s / 29.66 ms | not archived |
| 16K | 8.500 s / 23.50 ms | 10.996 s / 26.23 ms | 11.339 s / 29.70 ms | not archived |
| 32K | 19.406 s / 25.14 ms | 26.927 s / 26.35 ms | 26.318 s / 29.79 ms | not archived |
| 64K | 49.478 s / 28.62 ms | 64.505 s / 26.12 ms | 63.016 s / 30.00 ms | not archived |
| 128K | 140.544 s / 35.54 ms | 150.106 s / 26.90 ms | 151.238 s / 30.30 ms | not archived |

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
  --gpu-memory-utilization 0.9 \
  --output results/prolong-qwen-two-tier-speed-tp1-b1.json
```

For TP4, batch 8, change `--batch-size` to 8 and
`--tensor-parallel-size` to 4. Full attention defaults to the archived
`ROCM_AITER_UNIFIED_ATTN` control backend. The output records each repetition,
prompt hashes, aggregate prefill throughput, decode batch-step latency, and
decode token throughput.

