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
| Qwen3.8-27B-FP8 | Full | all | 0.443030 | 1.557420 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 4 | 0.447485 | 1.564373 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 8 | 0.445124 | 1.560683 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 4 | 0.447462 | 1.564337 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 8 | 0.445257 | 1.560891 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 4 | 0.447536 | 1.564452 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 8 | 0.445408 | 1.561127 |
| K2-Horizon-32B-FP8 | Full | all | 0.495866 | 1.641919 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 4 | 0.498166 | 1.645701 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 8 | 0.496713 | 1.643311 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 4 | 0.497483 | 1.644577 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 8 | 0.496475 | 1.642920 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 4 | 0.498337 | 1.645982 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 8 | 0.496903 | 1.643623 |

Prompt loss exercises prefill only. Each measurement contains 524,280 predicted
tokens; the loss is token-weighted across all eight documents. The top-4 rows
are archived baselines from 2026-09-14. Every full-attention and top-8 row was
rerun on 2026-09-23 from commit `94ec960c` with the current unflagged
production profile and identical document and token hashes within each model.

On this shared cohort, top-8 improves loss over top-4 in every LoD mode on
both models. It remains slightly worse than full attention: `+0.0021` to
`+0.0024` for Qwen and `+0.0006` to `+0.0010` for K2. The earlier apparent K2
reversal came from evaluating a different tokenizer-selected document set.

## Current routed top-8 speed results

Every full-attention, dummy-attention, and LoD arm below was freshly measured
on 2026-09-23 from commit `94ec960c` (source fingerprint `4e789f03da16`).
The panel uses MI325X GPUs, the same raw prompt cohort, a 16,384-token scheduler
chunk, one warmup per length, one measured repetition, and a 1,025-token
decode. Top-8 is used in both prefill and decode. Each cell is
`prefill seconds / decode milliseconds per batch step`.

The end-to-end table for each configuration is followed by a matched
attention-core table. Attention core is ordinary real wall time minus a
same-configuration dummy-attention wall time with CUDA graphs enabled. It
includes the attention backend, routing, cache updates, and LoD state
maintenance, while excluding QKV/RoPE, output projection, MLPs, scheduling,
and sampling. See [ATTENTION_TIMING.md](ATTENTION_TIMING.md) for the validated
method and its limitations.

### Qwen3.8, TP1, batch 1, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.992 s / 28.70 ms | 0.915 s / 28.65 ms | 0.930 s / 29.36 ms | 1.012 s / 29.86 ms |
| 16K | 2.185 s / 29.47 ms | 1.880 s / 28.67 ms | 1.886 s / 29.54 ms | 1.964 s / 29.55 ms |
| 32K | 5.243 s / 30.24 ms | 3.854 s / 28.78 ms | 3.899 s / 29.45 ms | 4.050 s / 29.56 ms |
| 64K | 14.054 s / 31.68 ms | 7.993 s / 28.89 ms | 8.084 s / 29.55 ms | 8.422 s / 29.62 ms |
| 128K | 42.568 s / 34.40 ms | 16.539 s / 29.25 ms | 16.777 s / 29.74 ms | 17.611 s / 30.08 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.122 s / 0.93 ms | 0.044 s / 0.89 ms | 0.059 s / 1.59 ms | 0.142 s / 2.10 ms |
| 16K | 0.474 s / 1.72 ms | 0.170 s / 0.91 ms | 0.175 s / 1.78 ms | 0.253 s / 1.80 ms |
| 32K | 1.837 s / 2.49 ms | 0.448 s / 1.04 ms | 0.493 s / 1.70 ms | 0.643 s / 1.81 ms |
| 64K | 7.250 s / 4.10 ms | 1.189 s / 1.31 ms | 1.280 s / 1.97 ms | 1.619 s / 2.04 ms |
| 128K | 28.973 s / 6.78 ms | 2.943 s / 1.63 ms | 3.181 s / 2.11 ms | 4.015 s / 2.46 ms |

### Qwen3.8, TP1, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 7.810 s / 37.79 ms | 7.233 s / 36.41 ms | 7.295 s / 35.57 ms | 7.921 s / 35.80 ms |
| 16K | 17.422 s / 40.73 ms | 14.907 s / 36.67 ms | 15.070 s / 35.97 ms | 15.663 s / 36.21 ms |
| 32K | 42.012 s / 45.94 ms | 31.131 s / 37.36 ms | 31.426 s / 36.25 ms | 32.598 s / 36.44 ms |
| 64K | 112.778 s / 55.71 ms | 64.545 s / 39.37 ms | 65.061 s / 36.74 ms | 67.558 s / 37.05 ms |
| 128K | 340.893 s / 74.66 ms | 134.499 s / 43.05 ms | 135.977 s / 37.62 ms | 142.278 s / 37.83 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.982 s / 5.08 ms | 0.405 s / 3.70 ms | 0.466 s / 2.86 ms | 1.093 s / 3.09 ms |
| 16K | 3.818 s / 8.04 ms | 1.304 s / 3.99 ms | 1.467 s / 3.28 ms | 2.060 s / 3.52 ms |
| 32K | 14.780 s / 13.12 ms | 3.898 s / 4.55 ms | 4.193 s / 3.44 ms | 5.365 s / 3.63 ms |
| 64K | 58.285 s / 22.90 ms | 10.052 s / 6.55 ms | 10.568 s / 3.93 ms | 13.064 s / 4.23 ms |
| 128K | 231.451 s / 42.00 ms | 25.057 s / 10.39 ms | 26.534 s / 4.96 ms | 32.836 s / 5.17 ms |

### Qwen3.8, TP4, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.426 s / 22.09 ms | 3.295 s / 22.14 ms | 3.989 s / 22.52 ms | 3.531 s / 22.66 ms |
| 16K | 7.610 s / 23.04 ms | 6.869 s / 22.00 ms | 6.987 s / 22.67 ms | 7.099 s / 22.85 ms |
| 32K | 17.253 s / 24.28 ms | 14.040 s / 22.33 ms | 14.253 s / 22.69 ms | 14.573 s / 23.01 ms |
| 64K | 42.977 s / 27.20 ms | 28.655 s / 22.84 ms | 29.042 s / 22.91 ms | 29.832 s / 23.05 ms |
| 128K | 119.429 s / 32.85 ms | 58.769 s / 23.76 ms | 59.555 s / 23.28 ms | 61.708 s / 23.58 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.307 s / 1.57 ms | 0.176 s / 1.63 ms | 0.870 s / 2.00 ms | 0.411 s / 2.14 ms |
| 16K | 1.156 s / 2.54 ms | 0.415 s / 1.50 ms | 0.533 s / 2.17 ms | 0.645 s / 2.35 ms |
| 32K | 4.338 s / 3.85 ms | 1.125 s / 1.90 ms | 1.339 s / 2.25 ms | 1.659 s / 2.57 ms |
| 64K | 17.145 s / 6.79 ms | 2.823 s / 2.43 ms | 3.210 s / 2.50 ms | 3.999 s / 2.63 ms |
| 128K | 67.765 s / 12.44 ms | 7.105 s / 3.35 ms | 7.891 s / 2.87 ms | 10.044 s / 3.17 ms |

### K2 Horizon, TP1, batch 1, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.158 s / 37.42 ms | 1.200 s / 38.27 ms | 1.270 s / 39.63 ms | 1.589 s / 39.81 ms |
| 16K | 2.604 s / 38.19 ms | 2.685 s / 38.69 ms | 2.675 s / 39.92 ms | 2.983 s / 40.23 ms |
| 32K | 6.349 s / 39.07 ms | 6.044 s / 38.82 ms | 6.266 s / 40.47 ms | 6.820 s / 40.79 ms |
| 64K | 17.191 s / 40.95 ms | 13.948 s / 39.15 ms | 14.395 s / 41.14 ms | 15.729 s / 41.64 ms |
| 128K | 53.861 s / 44.31 ms | 32.234 s / 39.92 ms | 33.223 s / 42.09 ms | 37.029 s / 42.56 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.160 s / 1.31 ms | 0.202 s / 2.17 ms | 0.273 s / 3.52 ms | 0.592 s / 3.70 ms |
| 16K | 0.622 s / 2.04 ms | 0.703 s / 2.53 ms | 0.693 s / 3.76 ms | 1.001 s / 4.07 ms |
| 32K | 2.385 s / 2.97 ms | 2.080 s / 2.72 ms | 2.302 s / 4.37 ms | 2.855 s / 4.68 ms |
| 64K | 9.016 s / 4.90 ms | 5.773 s / 3.10 ms | 6.220 s / 5.09 ms | 7.554 s / 5.59 ms |
| 128K | 37.987 s / 8.35 ms | 16.360 s / 3.96 ms | 17.349 s / 6.13 ms | 21.155 s / 6.60 ms |

### K2 Horizon, TP1, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 9.272 s / 46.33 ms | 9.757 s / 46.49 ms | 9.955 s / 47.99 ms | 11.383 s / 48.58 ms |
| 16K | 21.121 s / 49.83 ms | 20.703 s / 47.11 ms | 21.669 s / 49.38 ms | 23.087 s / 49.84 ms |
| 32K | 51.895 s / 56.92 ms | 51.497 s / 49.10 ms | 52.538 s / 51.06 ms | 56.023 s / 51.86 ms |
| 64K | 141.965 s / 69.56 ms | 122.662 s / 50.51 ms | 123.301 s / 52.84 ms | 133.197 s / 53.39 ms |
| 128K | Does not fit B8 | Does not fit B8 | Does not fit B8 | 326.523 s / 59.02 ms |

The 128K INT4 cell is a capacity result; BF16 and full-attention caches do
not fit eight K2 requests on one MI325X. It was measured separately with the
same eight prompts and a 0.8 GPU-memory fraction. Because a full-style dummy
also cannot keep B8 live at 128K, no attention-only subtraction is reported
for that capacity-only cell.

Attention core (real minus dummy), matched through 64K:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.255 s / 5.68 ms | 1.741 s / 5.84 ms | 1.939 s / 7.34 ms | 3.367 s / 7.93 ms |
| 16K | 5.083 s / 9.17 ms | 4.664 s / 6.45 ms | 5.630 s / 8.72 ms | 7.049 s / 9.19 ms |
| 32K | 19.821 s / 16.38 ms | 19.422 s / 8.56 ms | 20.463 s / 10.52 ms | 23.948 s / 11.33 ms |
| 64K | 77.422 s / 28.84 ms | 58.120 s / 9.79 ms | 58.759 s / 12.11 ms | 68.655 s / 12.67 ms |

### K2 Horizon, TP4, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.792 s / 22.60 ms | 3.969 s / 23.89 ms | 4.123 s / 24.97 ms | 4.615 s / 25.11 ms |
| 16K | 8.389 s / 23.59 ms | 8.059 s / 24.01 ms | 8.845 s / 26.31 ms | 9.249 s / 26.18 ms |
| 32K | 19.231 s / 25.23 ms | 19.913 s / 24.60 ms | 20.830 s / 27.03 ms | 21.968 s / 27.03 ms |
| 64K | 49.199 s / 28.61 ms | 46.686 s / 25.15 ms | 48.044 s / 28.22 ms | 51.590 s / 28.01 ms |
| 128K | 139.555 s / 35.55 ms | 111.052 s / 26.87 ms | 114.153 s / 31.79 ms | 127.760 s / 32.12 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.426 s / 2.17 ms | 0.603 s / 3.45 ms | 0.757 s / 4.54 ms | 1.249 s / 4.68 ms |
| 16K | 1.704 s / 3.16 ms | 1.374 s / 3.57 ms | 2.160 s / 5.88 ms | 2.565 s / 5.74 ms |
| 32K | 5.865 s / 4.80 ms | 6.546 s / 4.16 ms | 7.464 s / 6.60 ms | 8.602 s / 6.59 ms |
| 64K | 22.473 s / 8.18 ms | 19.961 s / 4.72 ms | 21.319 s / 7.79 ms | 24.864 s / 7.58 ms |
| 128K | 86.105 s / 15.11 ms | 57.602 s / 6.43 ms | 60.703 s / 11.36 ms | 74.310 s / 11.69 ms |

### Qwen3.8 with DFlash2, TP1, batch 1, top-8

This uses the same eight-prompt cohort and seven-token draft as the archived
DFlash2 panel. Decode measures the complete speculative loop; its time depends
on draft acceptance as well as target attention. All LoD prompt hashes were
checked against the same online dataset cohort.

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.996 s / 8.49 ms | 0.910 s / 7.27 ms | 0.920 s / 8.66 ms | 1.008 s / 7.66 ms |
| 16K | 2.166 s / 9.23 ms | 1.856 s / 8.72 ms | 1.860 s / 8.99 ms | 1.939 s / 7.75 ms |
| 32K | 5.200 s / 9.73 ms | 3.840 s / 9.25 ms | 3.877 s / 9.76 ms | 4.033 s / 9.47 ms |
| 64K | 13.934 s / 10.58 ms | 7.936 s / 8.77 ms | 8.040 s / 8.58 ms | 8.365 s / 8.20 ms |
| 128K | 42.366 s / 17.63 ms | 16.426 s / 10.37 ms | 16.758 s / 7.61 ms | 17.595 s / 8.49 ms |

Each next cell is `target verification-cycle milliseconds / pooled mean output
tokens per cycle`. It separates verifier cost from trajectory-dependent draft
acceptance.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 40.03 ms / 4.73 | 39.70 ms / 5.48 | 39.74 ms / 4.60 | 39.98 ms / 5.23 |
| 16K | 41.85 ms / 4.55 | 39.84 ms / 4.58 | 39.81 ms / 4.44 | 39.93 ms / 5.17 |
| 32K | 45.03 ms / 4.65 | 40.06 ms / 4.35 | 39.95 ms / 4.11 | 40.13 ms / 4.24 |
| 64K | 51.51 ms / 4.89 | 40.59 ms / 4.64 | 40.11 ms / 4.69 | 40.69 ms / 4.98 |
| 128K | 63.44 ms / 3.61 | 41.57 ms / 4.03 | 40.69 ms / 5.37 | 41.02 ms / 4.85 |

Attention-core subtraction is not valid for DFlash2: replacing target
attention changes sampled continuations and draft acceptance, so the dummy and
real runs no longer execute matched work. The end-to-end and verification-cycle
tables are the valid speculative measurements.

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
change `--tensor-parallel-size` to 4. Full attention defaults to the native
`ROCM_AITER_UNIFIED_ATTN` control backend. The output records each repetition,
prompt hashes, aggregate prefill throughput, decode batch-step latency, and
decode token throughput. These ordinary speed runs do not instrument kernels
or alter the production CUDA graph.

To reproduce an attention-core table, first run and save every ordinary full
and LoD arm. Then run the same full-attention command with only these changes:

```bash
  --mode full \
  --dummy-attention \
  --output results/prolong-qwen-dummy-speed-tp1-b1.json
```

Validate the pairing and subtract the matched dummy wall time:

```bash
uv run python -m benchmarks.attention_timing \
  --dummy results/prolong-qwen-dummy-speed-tp1-b1.json \
  --real results/prolong-qwen-full-speed-tp1-b1.json \
  --real results/prolong-qwen-two-tier-speed-tp1-b1.json \
  --real results/prolong-qwen-three-tier-bf16-speed-tp1-b1.json \
  --real results/prolong-qwen-three-tier-int4-speed-tp1-b1.json \
  --output results/prolong-qwen-attention-speed-tp1-b1.json
```

The summary command rejects mismatched source/runtime identities,
configurations, context lengths, or prompt hashes. The dummy run changes no
CUDA-graph setting; it replaces only the attention backend. DFlash2 cannot use
this method because dummy target outputs alter its speculative trajectory.

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
same command with `--lengths 131072`,
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
one node for TP4, commit `94ec960cfcb3097037e10d136fb4dd403649e056`, source
fingerprint `4e789f03da16b35908116511ff79a611823783e8b41d43530d75f87d7b7945c6`,
Python 3.12.13, PyTorch `2.11.0+gitd0c8b1f`, Transformers 5.15.0, Triton
3.6.0, vLLM `0.27.1+rocm723`, and the patched AITER build described in the
root README. The refreshed Qwen TP1, batch-8 cache modes and
the matched K2 TP1, batch-8 8K–64K modes ran concurrently as separate
processes on otherwise idle MI325X devices; runner settings and prompt hashes
matched within each panel. Other cache modes were run as separate processes,
sequentially on the same otherwise idle GPU set. Except for the documented K2
TP1, batch-8 capacity split, preserve the full length list in one invocation:
this intentionally gives every arm the same 128K configured capacity even
while measuring 8K. For K2 TP1, batch 8, preserve the four-length matched list
through 64K and use the separate 128K-only INT4 run for its capacity cell.
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
