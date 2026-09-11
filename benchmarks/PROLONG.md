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

The 8K and 16K LoD cells were rerun with exact decode disabled at those
lengths; the final release cutoff is 2K. Longer LoD cells already used routed
top-4 attention; the non-speculative full-attention controls are unchanged.
The DFlash2 panel was rerun in full with the diagnostics described below.

### Qwen3.8, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.998 s / 28.82 ms | 0.925 s / 28.53 ms | 0.911 s / 28.98 ms | 0.946 s / 28.89 ms |
| 16K | 2.172 s / 29.56 ms | 1.877 s / 28.58 ms | 1.863 s / 29.00 ms | 1.897 s / 29.09 ms |
| 32K | 5.214 s / 30.26 ms | 3.855 s / 28.81 ms | 3.923 s / 29.52 ms | 4.054 s / 29.57 ms |
| 64K | 13.941 s / 31.65 ms | 8.020 s / 29.00 ms | 8.151 s / 29.63 ms | 8.468 s / 29.88 ms |
| 128K | 42.227 s / 34.34 ms | 16.716 s / 29.35 ms | 16.973 s / 29.71 ms | 17.718 s / 29.81 ms |

### Qwen3.8, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.500 s / 22.07 ms | 3.337 s / 21.72 ms | 3.692 s / 21.75 ms | 3.684 s / 21.91 ms |
| 16K | 7.681 s / 23.01 ms | 6.954 s / 21.87 ms | 7.488 s / 21.83 ms | 7.485 s / 22.01 ms |
| 32K | 17.365 s / 24.23 ms | 14.703 s / 22.21 ms | 14.838 s / 22.55 ms | 15.071 s / 22.71 ms |
| 64K | 43.194 s / 27.17 ms | 30.027 s / 22.65 ms | 30.137 s / 22.63 ms | 30.692 s / 22.84 ms |
| 128K | 119.801 s / 32.72 ms | 61.561 s / 23.55 ms | 61.838 s / 22.89 ms | 63.410 s / 23.11 ms |

### K2 Horizon, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.156 s / 37.58 ms | 1.191 s / 39.60 ms | 1.284 s / 41.50 ms | 1.460 s / 40.30 ms |
| 16K | 2.593 s / 38.16 ms | 2.546 s / 39.99 ms | 2.691 s / 41.95 ms | 2.852 s / 40.77 ms |
| 32K | 6.324 s / 39.17 ms | 6.001 s / 40.45 ms | 6.167 s / 42.31 ms | 6.717 s / 41.18 ms |
| 64K | 17.120 s / 41.06 ms | 15.715 s / 41.04 ms | 16.023 s / 42.75 ms | 17.696 s / 41.68 ms |
| 128K | 52.721 s / 44.46 ms | 34.262 s / 41.99 ms | 35.010 s / 43.36 ms | 39.929 s / 42.40 ms |

### K2 Horizon, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.831 s / 22.58 ms | 5.020 s / 24.81 ms | 5.201 s / 27.31 ms | 5.484 s / 25.45 ms |
| 16K | 8.495 s / 23.54 ms | 10.782 s / 24.84 ms | 10.802 s / 27.83 ms | 10.906 s / 26.09 ms |
| 32K | 19.406 s / 25.17 ms | 25.491 s / 25.60 ms | 24.407 s / 28.27 ms | 25.234 s / 26.64 ms |
| 64K | 49.427 s / 28.62 ms | 59.488 s / 26.84 ms | 57.184 s / 29.18 ms | 61.076 s / 27.66 ms |
| 128K | 140.160 s / 35.46 ms | 138.536 s / 28.72 ms | 134.260 s / 31.43 ms | 147.164 s / 30.15 ms |

All K2 cells above, including the full-attention controls, were rerun after
matching AITER route-candidate storage to the D=128 kernel's native 128-key
tile and increasing K2's exact-leaf query tile from 32 to 64 rows. The selected
routes matched the previous exact scorer, and the 64-row exact-leaf output and
LSE were bit-identical to the 32-row version; these changes affect workspace
and launch geometry rather than LoD attention math.

### Qwen3.8 with DFlash2, TP1, batch 1

This panel uses `z-lab/Qwen3.8-27B-DFlash2` with seven proposed tokens. The
prefill column remains target-model prefill; the decode column measures the
complete speculative target-and-draft loop. Every row and mode was run
sequentially on one otherwise idle MI325X from the final release checkout,
with generation seed 0, so GPU contention does not get conflated across arms.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.979 s / 9.00 ms | 0.915 s / 7.53 ms | 0.926 s / 7.25 ms | 0.977 s / 7.06 ms |
| 16K | 2.204 s / 6.33 ms | 1.907 s / 6.17 ms | 1.930 s / 5.83 ms | 1.975 s / 6.22 ms |
| 32K | 5.253 s / 7.03 ms | 3.966 s / 7.89 ms | 3.994 s / 8.37 ms | 4.118 s / 8.53 ms |
| 64K | 14.017 s / 11.44 ms | 8.263 s / 8.38 ms | 8.310 s / 6.61 ms | 8.608 s / 7.44 ms |
| 128K | 42.601 s / 10.68 ms | 17.218 s / 7.19 ms | 17.316 s / 7.29 ms | 18.013 s / 7.96 ms |

All displayed DFlash2 lengths use routed top-4 LoD. DFlash2 stays routed at
shorter lengths too because its captured verifier and ordinary decode graphs
share one target cache; non-speculative LoD retains the 2K exact path. Decode
latency is end-to-end speculative latency rather than an isolated target-model
attention microbenchmark, so it also varies with the draft acceptance length.

The next table separates those effects. Each cell reports milliseconds per
target verification cycle / mean output tokens produced by that cycle; lower
is better for the first number and higher is better for the second.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 39.91 ms / 4.44 | 39.35 ms / 5.26 | 39.27 ms / 5.42 | 39.51 ms / 5.63 |
| 16K | 41.81 ms / 6.61 | 39.47 ms / 6.44 | 39.29 ms / 6.74 | 39.54 ms / 6.40 |
| 32K | 44.96 ms / 6.44 | 39.82 ms / 5.07 | 39.50 ms / 4.74 | 39.89 ms / 4.69 |
| 64K | 51.38 ms / 4.49 | 40.31 ms / 4.82 | 39.82 ms / 6.03 | 40.34 ms / 5.42 |
| 128K | 63.24 ms / 5.92 | 40.90 ms / 5.70 | 40.12 ms / 5.51 | 40.41 ms / 5.10 |

This distinction matters at 8K. The three measured full-attention repetitions
were 8.22, 9.00, and 9.48 ms per emitted token, while their verification-cycle
costs were 39.91, 39.91, and 39.96 ms. The older 8.23 ms result is therefore
reproduced by the first current repetition; the changed median is an acceptance
trajectory difference, not a slower full-attention kernel. The result JSON now
records the speculative counters and output-token hashes needed to diagnose
this case.

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
  --seed 0 \
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

## Reproduction requirements

The reported speed panel used one AMD MI325X for TP1 and four MI325X GPUs on
one node for TP4, vLLM 0.27.1, the release checkout, and the patched AITER build
described in the root README. Run cache modes as separate processes,
sequentially on the same otherwise idle GPU set. Preserve the full length list
in one invocation: this intentionally gives every arm the same 128K configured
capacity even while measuring 8K. Model startup is excluded, and the runner
performs one unreported warmup at every length before taking three repetitions.

Speed prompts use the fixed dataset shuffle seed `20260824`; prompt-loss
samples use seed `42`. Pass `--seed 0` exactly as shown to seed generation. For
DFlash2, also preserve seven proposed tokens, greedy sampling, and the draft
checkpoint shown above. The runner records all of these inputs, prompt hashes,
output-token hashes, and per-repetition timings in its JSON output.

A fixed seed does not make FP8 GEMMs and parallel GPU reductions bitwise
deterministic. A near-tied token can therefore change the continuation and its
subsequent draft acceptance even when the kernel cost is unchanged. For
DFlash2 comparisons, report both end-to-end emitted-token latency and the
verification-cycle/acceptance diagnostics rather than treating either one in
isolation as kernel speed.
