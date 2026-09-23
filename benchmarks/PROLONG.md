# ProLong quality and speed

ProLong serves two purposes here. Prompt log-probabilities measure the
token-weighted cross-entropy and perplexity of long natural text. A separate
generation sweep measures prefill wall time and decode latency on exact-length
natural-text prompts.

## Prompt-loss results

The table uses 65,536-token prefixes of the same eight raw documents in every
row. They are dataset indices `14, 19, 20, 23, 24, 25, 27, 28` in
`Seerkfang/prolong-64k-512-new`, revision
`97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`. The runner freezes those indices
before tokenization and records both a tokenizer-independent document hash and
a token hash. A tokenizer that cannot fill the requested length now fails
instead of silently substituting a different document.

| Model | Cache | Prefill routes | Loss | Perplexity |
|---|---|---:|---:|---:|
| Qwen3.8-27B-FP8 | Full | all | 0.443036 | 1.557428 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 4 | 0.447485 | 1.564373 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 8 | 0.445344 | 1.561027 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 4 | 0.447462 | 1.564337 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 8 | 0.445109 | 1.560660 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 4 | 0.447536 | 1.564452 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 8 | 0.445317 | 1.560984 |
| K2-Horizon-32B-FP8 | Full | all | 0.495866 | 1.641919 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 4 | 0.498166 | 1.645701 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 8 | 0.496567 | 1.643071 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 4 | 0.497483 | 1.644577 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 8 | 0.496666 | 1.643234 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 4 | 0.498337 | 1.645982 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 8 | 0.496575 | 1.643084 |

Prompt loss exercises prefill only. Each measurement contains 524,280 predicted
tokens; the loss is token-weighted across all eight documents. The top-4 rows
are archived baselines from 2026-09-14; the top-8 rows were rerun on 2026-09-17
using the current unflagged production profile and the same document hashes.
The K2 top-8 rows include the enlarged exact-leaf query tiles.

On this shared cohort, top-8 improves loss over top-4 in every LoD mode on
both models. It remains slightly worse than full attention: about `+0.0021`
for Qwen and `+0.0007` for K2. The earlier apparent K2 reversal came from
evaluating a different tokenizer-selected document set.

## Current routed top-8 speed results

These use the same MI325X, raw prompt cohort, 16,384-token scheduler chunk,
1,025-token decode, and timing definition as the archived top-4 panel below.
Full-attention controls are retained because its code and configuration did
not change. LoD rows were rerun on 2026-09-17 with top-8 in both prefill and
decode. Each cell is `prefill seconds / decode milliseconds per batch step`.

### Qwen3.8, TP1, batch 1, top-8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.980 s / 28.81 ms | 0.940 s / 28.76 ms | 0.955 s / 29.51 ms | 1.054 s / 29.56 ms |
| 16K | 2.163 s / 29.62 ms | 2.003 s / 28.76 ms | 2.025 s / 29.47 ms | 2.125 s / 29.59 ms |
| 32K | 5.205 s / 30.30 ms | 4.128 s / 28.70 ms | 4.181 s / 29.48 ms | 4.367 s / 29.64 ms |
| 64K | 13.920 s / 31.71 ms | 8.508 s / 28.98 ms | 8.628 s / 29.59 ms | 9.049 s / 29.70 ms |
| 128K | 42.236 s / 34.29 ms | 17.862 s / 29.29 ms | 18.054 s / 29.59 ms | 19.052 s / 29.84 ms |

### Qwen3.8, TP1, batch 8, top-8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 7.908 s / 37.84 ms | 7.661 s / 36.40 ms | 7.560 s / 35.63 ms | 8.274 s / 35.92 ms |
| 16K | 17.595 s / 40.78 ms | 16.194 s / 36.61 ms | 16.208 s / 35.80 ms | 17.015 s / 36.13 ms |
| 32K | 42.410 s / 45.85 ms | 33.704 s / 37.53 ms | 33.628 s / 35.89 ms | 35.242 s / 36.22 ms |
| 64K | 113.494 s / 55.72 ms | 69.869 s / 39.33 ms | 69.577 s / 35.99 ms | 72.793 s / 36.31 ms |
| 128K | 344.471 s / 74.74 ms | 147.081 s / 43.15 ms | 146.524 s / 36.32 ms | 154.188 s / 36.71 ms |

### Qwen3.8, TP4, batch 8, top-8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.480 s / 22.07 ms | 3.398 s / 22.13 ms | 3.503 s / 22.41 ms | 3.670 s / 22.47 ms |
| 16K | 7.719 s / 23.03 ms | 7.234 s / 22.15 ms | 7.373 s / 22.50 ms | 7.558 s / 22.65 ms |
| 32K | 17.437 s / 24.17 ms | 14.786 s / 22.36 ms | 15.036 s / 22.54 ms | 15.463 s / 22.69 ms |
| 64K | 43.388 s / 27.10 ms | 30.148 s / 22.81 ms | 30.529 s / 22.66 ms | 31.555 s / 22.82 ms |
| 128K | 120.322 s / 32.75 ms | 62.107 s / 23.70 ms | 62.903 s / 22.87 ms | 65.642 s / 23.11 ms |

### K2 Horizon, TP1, batch 1, top-8

These K2 cells use 128-by-32 BF16 exact-leaf tiles and 256-by-16 INT4 tiles.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.156 s / 37.58 ms | 1.197 s / 40.13 ms | 1.260 s / 41.60 ms | 1.585 s / 40.70 ms |
| 16K | 2.593 s / 38.16 ms | 2.548 s / 40.32 ms | 2.660 s / 42.03 ms | 2.978 s / 41.11 ms |
| 32K | 6.324 s / 39.17 ms | 6.050 s / 40.93 ms | 6.236 s / 42.64 ms | 6.814 s / 41.54 ms |
| 64K | 17.120 s / 41.06 ms | 14.001 s / 41.49 ms | 14.348 s / 42.99 ms | 15.732 s / 41.94 ms |
| 128K | 52.721 s / 44.46 ms | 33.324 s / 42.87 ms | 34.133 s / 43.78 ms | 37.923 s / 42.98 ms |

### K2 Horizon, TP1, batch 8, top-8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 9.227 s / 46.51 ms | 9.622 s / 48.04 ms | 9.938 s / 51.40 ms | 11.373 s / 50.68 ms |
| 16K | 21.084 s / 50.51 ms | 20.410 s / 49.19 ms | 21.625 s / 52.50 ms | 23.106 s / 51.82 ms |
| 32K | 51.810 s / 56.90 ms | 50.890 s / 51.53 ms | 52.517 s / 54.35 ms | 55.921 s / 53.86 ms |
| 64K | 141.656 s / 70.18 ms | 120.959 s / 53.94 ms | 123.473 s / 56.46 ms | 132.990 s / 55.92 ms |
| 128K | Does not fit B8 | Does not fit B8 | Does not fit B8 | 333.850 s / 61.23 ms |

The 128K INT4 cell is a capacity result; BF16 and full-attention caches do
not fit eight K2 requests on one MI325X. It was measured separately with the
same eight prompts and a 0.8 GPU-memory fraction.

### K2 Horizon, TP4, batch 8, top-8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.842 s / 22.66 ms | 4.008 s / 25.00 ms | 4.095 s / 27.38 ms | 4.619 s / 25.99 ms |
| 16K | 8.449 s / 23.59 ms | 8.140 s / 25.39 ms | 8.825 s / 28.54 ms | 9.260 s / 26.89 ms |
| 32K | 19.343 s / 25.21 ms | 20.021 s / 26.34 ms | 20.768 s / 29.13 ms | 21.967 s / 27.62 ms |
| 64K | 49.363 s / 28.67 ms | 46.930 s / 27.41 ms | 48.046 s / 29.96 ms | 51.656 s / 28.43 ms |
| 128K | 139.962 s / 35.51 ms | 113.306 s / 30.74 ms | 115.871 s / 33.38 ms | 128.477 s / 32.63 ms |

### Qwen3.8 with DFlash2, TP1, batch 1, top-8

This uses the same eight-prompt cohort and seven-token draft as the archived
DFlash2 panel. Decode measures the complete speculative loop; its time depends
on draft acceptance as well as target attention. All LoD prompt hashes were
checked against the same online dataset cohort.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.988 s / 7.76 ms | 0.941 s / 7.93 ms | 0.964 s / 8.39 ms | 1.042 s / 7.63 ms |
| 16K | 2.153 s / 9.26 ms | 1.989 s / 8.21 ms | 2.019 s / 8.87 ms | 2.095 s / 8.19 ms |
| 32K | 5.208 s / 10.72 ms | 4.142 s / 8.63 ms | 4.183 s / 9.01 ms | 4.343 s / 8.56 ms |
| 64K | 13.954 s / 11.44 ms | 8.575 s / 8.31 ms | 8.658 s / 8.61 ms | 8.991 s / 9.03 ms |
| 128K | 42.470 s / 19.82 ms | 17.974 s / 10.50 ms | 18.207 s / 8.83 ms | 19.062 s / 8.55 ms |

## Archived matched top-4 speed results

These measurements use AMD MI325X with vLLM 0.27.1. Every LoD cell, the three
non-speculative Qwen full-attention panels, and both K2 batch-8 full-attention
panels were freshly collected on 2026-09-12 or 2026-09-13. The unchanged K2
TP1, batch-1 and DFlash2 full-attention controls are retained from their
preceding matched sweeps because the cleanup does not touch native attention.
Each cell reports `prefill seconds / decode milliseconds per batch step`;
lower is better. One B8 decode step emits eight tokens concurrently.

Prompts are raw, distinct ProLong token streams, concatenating different
documents when needed and never repeating a document to fill a request. The
older internal summary called these chat-formatted, but its retained prompt
metadata and runner show that raw `prompt_token_ids` were used. The scheduler
chunk is 16,384 tokens. Decode generates 1,025 tokens and measures the final
1,024 steps. This includes four 256-token LoD state updates in the regular
path, or two 512-token updates for K2 INT4. Results are medians after one
warmup and three measured repetitions.

All displayed LoD cells in the archived tables below use routed top-4 attention;
the current top-8 panels appear above. The release's exact
decode cutoff is 2K. Full-attention controls use native attention throughout.
The DFlash2 panel was rerun in full with the diagnostics described below.

### Qwen3.8, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.980 s / 28.81 ms | 0.940 s / 28.61 ms | 0.960 s / 29.37 ms | 1.043 s / 29.52 ms |
| 16K | 2.163 s / 29.62 ms | 2.002 s / 28.64 ms | 2.030 s / 29.48 ms | 2.108 s / 29.39 ms |
| 32K | 5.205 s / 30.30 ms | 4.091 s / 28.71 ms | 4.137 s / 29.48 ms | 4.270 s / 29.42 ms |
| 64K | 13.920 s / 31.71 ms | 8.353 s / 28.95 ms | 8.459 s / 29.53 ms | 8.727 s / 29.51 ms |
| 128K | 42.236 s / 34.29 ms | 17.225 s / 29.25 ms | 17.388 s / 29.45 ms | 18.029 s / 29.65 ms |

### Qwen3.8, TP1, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 7.908 s / 37.84 ms | 7.577 s / 35.94 ms | 7.598 s / 35.45 ms | 8.210 s / 35.61 ms |
| 16K | 17.595 s / 40.78 ms | 16.007 s / 36.26 ms | 16.326 s / 35.47 ms | 16.868 s / 35.81 ms |
| 32K | 42.410 s / 45.85 ms | 33.076 s / 37.19 ms | 33.535 s / 35.68 ms | 34.416 s / 35.87 ms |
| 64K | 113.494 s / 55.72 ms | 67.853 s / 38.93 ms | 68.726 s / 35.70 ms | 70.359 s / 35.94 ms |
| 128K | 344.471 s / 74.74 ms | 140.283 s / 42.81 ms | 141.903 s / 35.98 ms | 145.970 s / 36.33 ms |

This panel isolates serving-batch scaling from tensor parallelism. At 128K,
three-tier BF16 delivers a 2.43x prefill speedup and a 2.08x decode speedup
over full attention; three-tier INT4 delivers 2.36x and 2.06x, respectively.

### Qwen3.8, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.480 s / 22.07 ms | 3.510 s / 21.81 ms | 3.472 s / 22.22 ms | 3.687 s / 22.37 ms |
| 16K | 7.719 s / 23.03 ms | 7.315 s / 21.88 ms | 7.449 s / 22.29 ms | 7.541 s / 22.54 ms |
| 32K | 17.437 s / 24.17 ms | 14.808 s / 22.09 ms | 15.029 s / 22.39 ms | 15.281 s / 22.52 ms |
| 64K | 43.388 s / 27.10 ms | 30.036 s / 22.51 ms | 30.415 s / 22.46 ms | 30.964 s / 22.65 ms |
| 128K | 120.322 s / 32.75 ms | 61.201 s / 23.45 ms | 61.930 s / 22.79 ms | 63.409 s / 22.96 ms |

### K2 Horizon, TP1, batch 1

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.156 s / 37.58 ms | 1.200 s / 39.89 ms | 1.284 s / 41.81 ms | 1.600 s / 40.20 ms |
| 16K | 2.593 s / 38.16 ms | 2.556 s / 40.18 ms | 2.704 s / 42.17 ms | 2.981 s / 40.64 ms |
| 32K | 6.324 s / 39.17 ms | 5.860 s / 40.70 ms | 6.086 s / 42.58 ms | 6.661 s / 41.25 ms |
| 64K | 17.120 s / 41.06 ms | 13.205 s / 41.25 ms | 13.682 s / 42.99 ms | 15.167 s / 41.70 ms |
| 128K | 52.721 s / 44.46 ms | 30.113 s / 42.43 ms | 31.216 s / 43.84 ms | 35.616 s / 42.49 ms |

### K2 Horizon, TP1, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 9.227 s / 46.51 ms | 9.708 s / 47.25 ms | 9.852 s / 50.08 ms | 11.262 s / 49.52 ms |
| 16K | 21.084 s / 50.51 ms | 20.446 s / 48.41 ms | 21.535 s / 51.14 ms | 22.891 s / 50.57 ms |
| 32K | 51.810 s / 56.90 ms | 49.033 s / 50.18 ms | 50.236 s / 52.80 ms | 54.082 s / 52.25 ms |
| 64K | 141.656 s / 70.18 ms | 113.604 s / 52.14 ms | 115.453 s / 54.79 ms | 127.186 s / 54.20 ms |
| 128K | Does not fit B8 | Does not fit B8 | Does not fit B8 | 315.758 s / 58.26 ms |

Two-tier crosses over by 16K. At 64K it is 1.25x faster in prefill and
1.35x faster per decode batch step than full attention. Three-tier BF16 is
1.23x and 1.28x faster at 64K, while three-tier INT4 is 1.11x and 1.29x
faster and provides the capacity benefit at 128K.

The unavailable 128K cells are a physical-capacity limit, not missing speed
runs. K2 has 64 layers, 8 KV heads, and 128 channels per head, so uncompressed
BF16 K/V leaves at 128K require exactly 32 GiB per request. Eight requests use
the entire 256 GiB MI325X before the 36.1 GiB model, summaries, or workspaces.
The full-attention engine reported only 5.74x maximum concurrency, and both
BF16 LoD pool allocations ran out of memory. Three-tier INT4 reported 8.00x
maximum concurrency and completed without preemption. Its 128K cell is
therefore a capacity result rather than a cross-mode speed comparison.

### K2 Horizon, TP4, batch 8

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.842 s / 22.66 ms | 3.997 s / 24.61 ms | 4.121 s / 26.89 ms | 4.627 s / 25.41 ms |
| 16K | 8.449 s / 23.59 ms | 8.110 s / 24.99 ms | 8.842 s / 28.02 ms | 9.254 s / 26.30 ms |
| 32K | 19.343 s / 25.21 ms | 19.457 s / 25.92 ms | 20.222 s / 28.60 ms | 21.578 s / 27.01 ms |
| 64K | 49.363 s / 28.67 ms | 44.976 s / 26.73 ms | 46.058 s / 29.41 ms | 50.449 s / 27.84 ms |
| 128K | 139.962 s / 35.51 ms | 106.345 s / 29.11 ms | 108.411 s / 32.42 ms | 125.176 s / 31.34 ms |

Both TP4, batch-8 panels, including their full-attention controls, use the
release's chunk-aligned async scheduler. A fixed 16,384 aggregate token budget
let already-running decode rows remove one to seven tokens from a newly
admitted 16K prompt. That forced LoD to construct an almost-complete prefix and
then process a second tiny cached-prefill fragment. The aligned scheduler
exposes the complete 16K prefill allowance plus only the decode work actually
eligible in that step. Before this scheduler fix, the three K2 16K LoD prefill
figures were 10.782, 10.802, and 10.906 seconds; the fully rerun values are now
8.110, 8.842, and 9.254 seconds. No attention math changed. TP1, batch-1 and
DFlash2 batch-1 cannot co-schedule a waiting prefill with a live request, so
their tables are not affected by this scheduler correction.

All K2 cells above, including the full-attention controls, were rerun after
matching AITER route-candidate storage to the D=128 kernel's native 128-key
tile and increasing K2's exact-leaf query tile from 32 to 64 rows. The selected
routes matched the previous exact scorer, and the 64-row exact-leaf output and
LSE were bit-identical to the 32-row version; these changes affect workspace
and launch geometry rather than LoD attention math.

### Qwen3.8 with DFlash2, TP1, batch 1

This panel uses `z-lab/Qwen3.8-27B-DFlash2` with seven proposed tokens. The
prefill column remains target-model prefill; the decode column measures the
complete speculative target-and-draft loop. At every length it uses the same
eight prompt-token streams as the TP1, batch-8 panel, but submits each request
alone. Each displayed value is the median of three cohort passes, with timing
averaged across the eight isolated B1 requests in each pass.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.988 s / 7.76 ms | 0.944 s / 8.15 ms | 0.963 s / 8.09 ms | 1.036 s / 8.47 ms |
| 16K | 2.153 s / 9.26 ms | 2.003 s / 8.42 ms | 2.013 s / 8.32 ms | 2.088 s / 8.19 ms |
| 32K | 5.208 s / 10.72 ms | 4.123 s / 8.24 ms | 4.141 s / 8.02 ms | 4.269 s / 8.50 ms |
| 64K | 13.954 s / 11.44 ms | 8.462 s / 8.48 ms | 8.499 s / 7.90 ms | 8.738 s / 8.84 ms |
| 128K | 42.470 s / 19.82 ms | 17.463 s / 9.82 ms | 17.545 s / 9.29 ms | 18.116 s / 8.45 ms |

All displayed DFlash2 lengths use routed top-4 LoD. DFlash2 stays routed at
shorter lengths too because its captured verifier and ordinary decode graphs
share one target cache; non-speculative LoD retains the 2K exact path. Decode
latency is end-to-end speculative latency rather than an isolated target-model
attention microbenchmark, so it also varies with the draft acceptance length.

The next table separates those effects. Each cell reports pooled milliseconds
per target verification cycle / pooled mean output tokens produced by that
cycle across the same eight requests; lower is better for the first number and
higher is better for the second.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 39.95 ms / 5.14 | 39.43 ms / 4.85 | 39.28 ms / 4.87 | 39.60 ms / 4.68 |
| 16K | 41.65 ms / 4.52 | 39.68 ms / 4.71 | 39.60 ms / 4.78 | 39.73 ms / 4.87 |
| 32K | 44.88 ms / 4.20 | 39.81 ms / 4.88 | 39.74 ms / 4.97 | 39.94 ms / 4.70 |
| 64K | 51.58 ms / 4.53 | 40.42 ms / 4.78 | 39.84 ms / 5.08 | 40.20 ms / 4.56 |
| 128K | 63.49 ms / 3.21 | 40.96 ms / 4.19 | 40.15 ms / 4.34 | 40.62 ms / 4.82 |

Pooling uses `1 + sum(accepted draft tokens) / sum(target cycles)`, matching
the B8 counter. The JSON additionally records each isolated request's
acceptance and their equal-weight mean. Keeping both matters because a hard
continuation consumes more target cycles and therefore receives more weight in
the pooled statistic. Fixed-seed FP8 generation can still diverge at a
near-tied token, so the eight-request cohort is substantially more stable than
the former single-continuation report.

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
  --repeats 1 \
  --seed 0 \
  --gpu-memory-utilization 0.7 \
  --output results/prolong-qwen-two-tier-speed-tp1-b1.json
```

For TP1, batch 8, change only `--batch-size` to 8. For TP4, batch 8, also
change `--tensor-parallel-size` to 4. Full attention defaults to the archived
`ROCM_AITER_UNIFIED_ATTN` control backend. The output records each repetition,
prompt hashes, aggregate prefill throughput, decode batch-step latency, and
decode token throughput. The benchmark does not run a separate instrumented
attention pass or alter the production CUDA graph.

K2 TP1, batch 8 is the capacity exception. Reproduce its matched 8K–64K
portion with:

```bash
uv run python -m benchmarks.prolong \
  --measure speed \
  --checkpoint IFM/K2-Horizon-32B-FP8 \
  --mode two-tier \
  --lengths 8192,16384,32768,65536 \
  --batch-size 8 \
  --tensor-parallel-size 1 \
  --decode-tokens 1025 \
  --repeats 1 \
  --seed 0 \
  --gpu-memory-utilization 0.9 \
  --output results/prolong-k2-two-tier-speed-tp1-b8-64k.json
```

Repeat that command with `--mode full`, `three-tier-bf16`, and
`three-tier-int4` for the other columns. The displayed 128K INT4 cell uses the
same command with all five lengths through `131072`,
`--gpu-memory-utilization 0.8`, and `--mode three-tier-int4`. The other three
modes cannot run eight 128K requests on one 256 GiB device.

The LoD command leaves additional VRAM outside vLLM's native-cache allocator
for concurrent cache-maintenance workspaces. Qwen defaults to 0.7. K2 defaults
to 0.8 because its larger model-side 131K LoD pool otherwise leaves no native
cache blocks. The matched K2 TP1, batch-8 8K–64K panel explicitly uses 0.9
for every arm; its INT4-only 128K run uses 0.8. Use
`--gpu-memory-utilization 0.9` for the other full-attention controls, which
have no separate LoD pool.

To reproduce the DFlash2 panel, add:

```bash
  --speculative-model z-lab/Qwen3.8-27B-DFlash2 \
  --num-speculative-tokens 7 \
  --speed-samples 8
```

to the Qwen speed command. With `--batch-size 1`, the eight prompts execute
sequentially; with `--batch-size 8`, the identical prompt cohort executes as
one serving batch. DFlash2 is supported only for Qwen3.8 in this release.

## Reproduction requirements

The reported speed panel used one AMD MI325X for TP1 and four MI325X GPUs on
one node for TP4, vLLM 0.27.1, the release checkout, and the patched AITER build
described in the root README. The refreshed Qwen TP1, batch-8 cache modes and
the matched K2 TP1, batch-8 8K–64K modes ran concurrently as separate
processes on otherwise idle MI325X devices; runner settings and prompt hashes
matched within each panel. Other cache modes were run as separate processes,
sequentially on the same otherwise idle GPU set. Except for the documented K2
TP1, batch-8 capacity split, preserve the full length list in one invocation:
this intentionally gives every arm the same 128K configured capacity even
while measuring 8K. For K2 TP1, batch 8, preserve the four-length matched list
through 64K and use the separate five-length INT4 run only for its 128K cell.
Model startup is excluded, and the runner performs one unreported warmup at
every length before taking one measured repetition.

Speed prompts use the fixed dataset shuffle seed `20260824`. Prompt loss uses
the frozen raw-document indices listed above; its default `--sample-offset 8`
selects them from the 16-entry shared release cohort. Pass `--seed 0` exactly
as shown to seed generation. For DFlash2, also preserve seven proposed tokens,
greedy sampling, and the draft checkpoint shown above. The runner records all
of these inputs, document hashes, prompt hashes, output-token hashes, and
per-repetition timings in its JSON output.

Resolve the ProLong dataset online when reproducing a speed panel. With
`HF_HUB_OFFLINE=1`, the cached streaming-dataset shuffle produced a different
document order despite the pinned revision and shuffle seed. Compare the
recorded prompt-token hashes between modes before comparing their timings.

A fixed seed does not make FP8 GEMMs and parallel GPU reductions bitwise
deterministic. A near-tied token can therefore change the continuation and its
subsequent draft acceptance even when the kernel cost is unchanged. For
DFlash2 comparisons, report both end-to-end emitted-token latency and the
verification-cycle/acceptance diagnostics rather than treating either one in
isolation as kernel speed.
