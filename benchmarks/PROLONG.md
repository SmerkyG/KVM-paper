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
| Qwen3.8-27B-FP8 | Full | all | 0.443006 | 1.557382 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 8 | 0.445054 | 1.560574 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 8 | 0.445347 | 1.561032 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 8 | 0.445263 | 1.560901 |
| K2-Horizon-32B-FP8 | Full | all | 0.495866 | 1.641919 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 8 | 0.496713 | 1.643311 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 8 | 0.496617 | 1.643153 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 8 | 0.496615 | 1.643150 |

Prompt loss exercises prefill only. Each measurement contains 524,280 predicted
tokens; the loss is token-weighted across all eight documents. The Qwen rows were rerun on 2026-09-23 with the current unflagged top-8
production profile and identical document and token hashes. K2 full and
two-tier retain the matched production rerun from commit `94ec960c`; K2
three-tier BF16 and INT4 were rerun on 2026-09-24 from commit `2551051c`
(source fingerprint `201135ea9072`) after enabling cross-layer page
construction. All K2 rows use the same frozen documents and token hashes.

On this shared cohort, LoD remains slightly worse than full attention:
`+0.0021` to `+0.0024` for Qwen and `+0.0007` to `+0.0008` for K2. The earlier
apparent K2 reversal came from evaluating a different tokenizer-selected
document set.

## Current routed top-8 speed results

Every Qwen full-attention, dummy-attention, and LoD arm below was freshly
measured on 2026-09-23 from commit `e248719a` (source fingerprint
`cc735bcc44b4`). The K2 full and two-tier columns retain the matched commit
`94ec960c` controls. K2 TP1 batch-1 and TP4 batch-8 three-tier columns were
refreshed from commit `2551051c` (source fingerprint `201135ea9072`) after
cross-layer page construction was enabled for BF16 and INT4. K2 TP1 batch-8
three-tier columns use the same commit with source fingerprint `7f23540fc95d`
after bounding the dense route workspace. The panels use MI325X GPUs, the
same raw prompt cohort, a 16,384-token scheduler chunk, one warmup per length,
one measured repetition, and a 1,025-token decode. Top-8 is used in both
prefill and decode. Each cell is
`prefill seconds / decode milliseconds per batch step`.

The Qwen TP1 batch-1 two-tier 8K--64K cells were subsequently replaced by one
matched three-repetition sweep from commit `07aa3870` (source fingerprint
`cc735bcc44b4`) with the TP1 process holding an otherwise idle eight-GPU node.
The K2 TP1 batch-8 two-tier column retains its 2026-09-24 exclusive-node
rerun from commit `2551051c` (source fingerprint `f4a180d5730a`). Its
three-tier BF16 and INT4 columns were refreshed under the same exclusive-node
policy from source fingerprint `7f23540fc95d`; the INT4 128K capacity cell was
measured separately with the same prompts. At short lengths, decode attention
is a small residual between two much larger end-to-end measurements, so
hundredths of a millisecond are below the useful resolution.

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
| 8K | 1.085 s / 29.71 ms | 0.904 s / 28.43 ms | 0.916 s / 28.91 ms | 0.933 s / 28.83 ms |
| 16K | 2.316 s / 30.14 ms | 1.847 s / 28.40 ms | 2.061 s / 29.94 ms | 2.209 s / 30.08 ms |
| 32K | 5.291 s / 30.52 ms | 3.817 s / 28.32 ms | 4.089 s / 30.76 ms | 4.304 s / 30.66 ms |
| 64K | 13.886 s / 31.75 ms | 7.884 s / 28.47 ms | 8.223 s / 30.54 ms | 8.645 s / 30.91 ms |
| 128K | 42.472 s / 43.57 ms | 16.542 s / 29.62 ms | 16.818 s / 31.01 ms | 17.885 s / 30.86 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.202 s / 1.03 ms | 0.050 s / 0.87 ms | 0.047 s / 1.11 ms | 0.064 s / 1.03 ms |
| 16K | 0.575 s / 1.56 ms | 0.150 s / 0.85 ms | 0.320 s / 1.36 ms | 0.468 s / 1.50 ms |
| 32K | 1.808 s / 1.89 ms | 0.439 s / 0.74 ms | 0.606 s / 2.14 ms | 0.821 s / 2.03 ms |
| 64K | 7.069 s / 3.06 ms | 1.150 s / 0.73 ms | 1.405 s / 1.84 ms | 1.827 s / 2.21 ms |
| 128K | 28.840 s / 15.08 ms | 2.909 s / 1.13 ms | 3.186 s / 2.52 ms | 4.253 s / 2.37 ms |

### Qwen3.8, TP1, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 7.719 s / 37.94 ms | 7.304 s / 34.49 ms | 7.856 s / 35.55 ms | 7.930 s / 35.89 ms |
| 16K | 17.440 s / 40.86 ms | 14.923 s / 34.79 ms | 14.955 s / 35.89 ms | 15.782 s / 36.26 ms |
| 32K | 41.839 s / 46.45 ms | 31.177 s / 35.17 ms | 31.128 s / 36.25 ms | 32.817 s / 36.56 ms |
| 64K | 112.042 s / 55.77 ms | 64.618 s / 35.79 ms | 64.513 s / 36.65 ms | 67.946 s / 37.00 ms |
| 128K | 342.808 s / 74.66 ms | 134.521 s / 36.67 ms | 135.291 s / 37.62 ms | 143.629 s / 38.02 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.999 s / 4.86 ms | 0.585 s / 1.42 ms | 1.136 s / 2.47 ms | 1.211 s / 2.82 ms |
| 16K | 3.930 s / 8.01 ms | 1.413 s / 1.95 ms | 1.444 s / 3.04 ms | 2.271 s / 3.41 ms |
| 32K | 14.834 s / 13.60 ms | 4.172 s / 2.32 ms | 4.124 s / 3.40 ms | 5.812 s / 3.72 ms |
| 64K | 58.018 s / 22.93 ms | 10.594 s / 2.95 ms | 10.489 s / 3.81 ms | 13.922 s / 4.16 ms |
| 128K | 234.769 s / 41.85 ms | 26.482 s / 3.85 ms | 27.251 s / 4.81 ms | 35.590 s / 5.20 ms |

### Qwen3.8, TP4, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.566 s / 22.07 ms | 3.393 s / 21.31 ms | 3.448 s / 22.51 ms | 3.567 s / 22.57 ms |
| 16K | 7.720 s / 23.06 ms | 6.886 s / 21.39 ms | 7.063 s / 22.58 ms | 7.148 s / 22.76 ms |
| 32K | 17.454 s / 24.26 ms | 14.102 s / 21.61 ms | 14.429 s / 22.78 ms | 14.656 s / 22.79 ms |
| 64K | 43.454 s / 27.20 ms | 28.772 s / 21.71 ms | 29.294 s / 22.81 ms | 30.046 s / 22.94 ms |
| 128K | 120.498 s / 32.77 ms | 59.019 s / 22.00 ms | 59.991 s / 23.24 ms | 62.118 s / 23.48 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.417 s / 1.69 ms | 0.244 s / 0.92 ms | 0.299 s / 2.13 ms | 0.419 s / 2.19 ms |
| 16K | 1.227 s / 2.59 ms | 0.394 s / 0.92 ms | 0.570 s / 2.12 ms | 0.655 s / 2.29 ms |
| 32K | 4.488 s / 3.81 ms | 1.137 s / 1.17 ms | 1.463 s / 2.34 ms | 1.690 s / 2.35 ms |
| 64K | 17.501 s / 6.79 ms | 2.819 s / 1.31 ms | 3.341 s / 2.40 ms | 4.092 s / 2.53 ms |
| 128K | 68.578 s / 12.40 ms | 7.098 s / 1.62 ms | 8.071 s / 2.86 ms | 10.198 s / 3.10 ms |

### K2 Horizon, TP1, batch 1, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.158 s / 37.42 ms | 1.200 s / 38.27 ms | 1.204 s / 39.68 ms | 1.491 s / 39.93 ms |
| 16K | 2.604 s / 38.19 ms | 2.685 s / 38.69 ms | 2.568 s / 40.07 ms | 2.855 s / 40.36 ms |
| 32K | 6.349 s / 39.07 ms | 6.044 s / 38.82 ms | 5.918 s / 40.72 ms | 6.515 s / 41.00 ms |
| 64K | 17.191 s / 40.95 ms | 13.948 s / 39.15 ms | 13.403 s / 41.24 ms | 14.882 s / 41.58 ms |
| 128K | 53.861 s / 44.31 ms | 32.234 s / 39.92 ms | 30.474 s / 42.20 ms | 34.615 s / 42.47 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.160 s / 1.31 ms | 0.202 s / 2.17 ms | 0.206 s / 3.57 ms | 0.493 s / 3.82 ms |
| 16K | 0.622 s / 2.04 ms | 0.703 s / 2.53 ms | 0.586 s / 3.92 ms | 0.873 s / 4.21 ms |
| 32K | 2.385 s / 2.97 ms | 2.080 s / 2.72 ms | 1.954 s / 4.62 ms | 2.551 s / 4.90 ms |
| 64K | 9.016 s / 4.90 ms | 5.773 s / 3.10 ms | 5.228 s / 5.19 ms | 6.707 s / 5.53 ms |
| 128K | 37.987 s / 8.35 ms | 16.360 s / 3.96 ms | 14.600 s / 6.24 ms | 18.741 s / 6.51 ms |

### K2 Horizon, TP1, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 9.272 s / 46.33 ms | 9.697 s / 46.46 ms | 9.767 s / 47.79 ms | 11.293 s / 48.61 ms |
| 16K | 21.121 s / 49.83 ms | 20.429 s / 47.19 ms | 20.597 s / 49.20 ms | 22.066 s / 49.88 ms |
| 32K | 51.895 s / 56.92 ms | 49.740 s / 49.13 ms | 49.410 s / 51.12 ms | 53.419 s / 51.83 ms |
| 64K | 141.965 s / 69.56 ms | 115.998 s / 50.44 ms | 114.730 s / 52.72 ms | 126.000 s / 53.33 ms |
| 128K | Does not fit B8 | Does not fit B8 | Does not fit B8 | 310.522 s / 59.20 ms |

The 128K INT4 cell is a capacity result; BF16 and full-attention caches do
not fit eight K2 requests on one MI325X. It was measured separately with the
same eight prompts and a 0.8 GPU-memory fraction. Because a full-style dummy
also cannot keep B8 live at 128K, no attention-only subtraction is reported
for that capacity-only cell.

Attention core (real minus dummy), matched through 64K:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.255 s / 5.68 ms | 1.681 s / 5.81 ms | 1.750 s / 7.14 ms | 3.276 s / 7.96 ms |
| 16K | 5.083 s / 9.17 ms | 4.391 s / 6.53 ms | 4.559 s / 8.54 ms | 6.028 s / 9.22 ms |
| 32K | 19.821 s / 16.38 ms | 17.666 s / 8.59 ms | 17.336 s / 10.58 ms | 21.345 s / 11.29 ms |
| 64K | 77.422 s / 28.84 ms | 51.455 s / 9.72 ms | 50.187 s / 12.00 ms | 61.457 s / 12.61 ms |

### K2 Horizon, TP4, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.792 s / 22.60 ms | 3.969 s / 23.89 ms | 4.086 s / 24.88 ms | 4.609 s / 25.03 ms |
| 16K | 8.389 s / 23.59 ms | 8.059 s / 24.01 ms | 8.478 s / 26.27 ms | 8.983 s / 26.10 ms |
| 32K | 19.231 s / 25.23 ms | 19.913 s / 24.60 ms | 19.771 s / 27.07 ms | 20.985 s / 26.91 ms |
| 64K | 49.199 s / 28.61 ms | 46.686 s / 25.15 ms | 45.268 s / 28.11 ms | 48.663 s / 28.02 ms |
| 128K | 139.555 s / 35.55 ms | 111.052 s / 26.87 ms | 106.511 s / 31.78 ms | 118.550 s / 31.66 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.426 s / 2.17 ms | 0.603 s / 3.45 ms | 0.720 s / 4.45 ms | 1.243 s / 4.60 ms |
| 16K | 1.704 s / 3.16 ms | 1.374 s / 3.57 ms | 1.793 s / 5.84 ms | 2.298 s / 5.67 ms |
| 32K | 5.865 s / 4.80 ms | 6.546 s / 4.16 ms | 6.405 s / 6.64 ms | 7.619 s / 6.48 ms |
| 64K | 22.473 s / 8.18 ms | 19.961 s / 4.72 ms | 18.542 s / 7.68 ms | 21.937 s / 7.59 ms |
| 128K | 86.105 s / 15.11 ms | 57.602 s / 6.43 ms | 53.061 s / 11.34 ms | 65.100 s / 11.22 ms |

### Qwen3.8 with DFlash2, TP1, batch 1, top-8

This uses the same eight-prompt cohort and seven-token draft as the archived
DFlash2 panel. Decode measures the complete speculative loop; its time depends
on draft acceptance as well as target attention. All LoD prompt hashes were
checked against the same online dataset cohort.

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.978 s / 8.10 ms | 0.922 s / 7.44 ms | 0.929 s / 7.45 ms | 1.016 s / 8.38 ms |
| 16K | 2.149 s / 8.82 ms | 1.852 s / 7.24 ms | 1.872 s / 8.25 ms | 1.955 s / 8.57 ms |
| 32K | 5.204 s / 9.44 ms | 3.843 s / 9.64 ms | 3.894 s / 9.20 ms | 4.062 s / 8.37 ms |
| 64K | 14.138 s / 9.94 ms | 7.936 s / 7.72 ms | 8.071 s / 9.34 ms | 8.438 s / 9.21 ms |
| 128K | 42.817 s / 16.86 ms | 16.430 s / 9.41 ms | 16.831 s / 9.69 ms | 17.700 s / 10.55 ms |

Each next cell is `target verification-cycle milliseconds / pooled mean output
tokens per cycle`. It separates verifier cost from trajectory-dependent draft
acceptance.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 39.88 ms / 4.93 | 39.73 ms / 5.35 | 39.68 ms / 5.35 | 39.97 ms / 4.79 |
| 16K | 41.90 ms / 4.77 | 40.02 ms / 5.54 | 39.77 ms / 4.84 | 40.07 ms / 4.70 |
| 32K | 44.88 ms / 4.77 | 40.21 ms / 4.18 | 39.99 ms / 4.36 | 40.32 ms / 4.83 |
| 64K | 51.79 ms / 5.23 | 40.77 ms / 5.30 | 40.24 ms / 4.32 | 41.04 ms / 4.47 |
| 128K | 63.56 ms / 3.78 | 41.64 ms / 4.43 | 40.62 ms / 4.21 | 41.02 ms / 3.90 |

Attention-core subtraction is not valid for DFlash2: replacing target
attention changes sampled continuations and draft acceptance, so the dummy and
real runs no longer execute matched work. The end-to-end and verification-cycle
tables are the valid speculative measurements.

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

The summary command rejects mismatched runtime package identities,
configurations, context lengths, or prompt hashes. Source fingerprints are
recorded per arm but may differ, allowing an unchanged full-attention or dummy
control to be reused after a LoD-only source change. The dummy run changes no
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
