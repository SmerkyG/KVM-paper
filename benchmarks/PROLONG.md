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
| Qwen3.8-27B-FP8 | Two-tier BF16 | 8 | 0.445288 | 1.560940 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 8 | 0.445161 | 1.560741 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 8 | 0.445557 | 1.561360 |
| K2-Horizon-32B-FP8 | Full | all | 0.495866 | 1.641919 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 8 | 0.496789 | 1.643435 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 8 | 0.496818 | 1.643484 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 8 | 0.496989 | 1.643764 |

Prompt loss exercises prefill only. Each measurement contains 524,280 predicted
tokens; the loss is token-weighted across all eight documents. Every LoD row
was rerun on 2026-09-25--26 from the current release checkout with the
unflagged top-8 production profile and identical document and token hashes.
The LoD speed and quality runs record source fingerprint `9e316f18f0c85ba`;
the K2 two-tier quality run used the same released calculation before the
unrelated three-tier directory change. The full-attention controls are
unchanged. All rows use the same frozen documents and token hashes.
The standardized 256-token decode update interval does not affect this table:
prompt loss ends after prefill, before any decode cache catch-up can occur.

On this shared cohort, LoD remains slightly worse than full attention:
`+0.0022` to `+0.0026` for Qwen and `+0.0009` to `+0.0011` for K2. The earlier
apparent K2 reversal came from evaluating a different tokenizer-selected
document set.

## Current routed top-8 speed results

Every LoD arm was freshly measured on 2026-09-25--26 from the current release
checkout (source fingerprint `9e316f18f0c85ba`) with the standardized
256-token decode update interval. The Qwen TP1/B1 full-attention control and a
single dummy-attention control shared by all four Qwen TP1/B1 columns were
rerun together on 2026-09-26; their strict matched subtraction is preserved in
`results/qwen-tp1-b1-dummy-audit/attention.json`. The panels use MI325X GPUs,
the same raw prompt cohort, a 16,384-token scheduler chunk, one warmup per
length, one measured repetition, and a 1,025-token decode. Top-8 is used in
both prefill and decode. Each cell is
`prefill seconds / decode milliseconds per batch step`.

The refreshed K2 INT4 128K capacity cell was measured separately with the same
prompts and production policy as its 8K--64K sweep. At short lengths, decode attention
is a small residual between two much larger end-to-end measurements, so
hundredths of a millisecond are below the useful resolution.

The end-to-end table for each configuration is followed by a matched
attention-core table. Attention core is ordinary real wall time minus one
configuration-matched dummy-attention control shared by every attention mode
in that panel, with CUDA graphs enabled. It includes the attention backend,
routing, cache updates, and LoD state maintenance, while excluding QKV/RoPE,
output projection, MLPs, scheduling, and sampling. See
[ATTENTION_TIMING.md](ATTENTION_TIMING.md) for the validated method and its
limitations.

### Qwen3.8, TP1, batch 1, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.002 s / 28.84 ms | 0.913 s / 28.43 ms | 0.913 s / 29.44 ms | 0.991 s / 29.58 ms |
| 16K | 2.184 s / 29.61 ms | 1.846 s / 28.32 ms | 1.868 s / 29.47 ms | 1.943 s / 29.68 ms |
| 32K | 5.242 s / 30.33 ms | 3.828 s / 28.35 ms | 3.865 s / 29.50 ms | 4.030 s / 29.72 ms |
| 64K | 13.996 s / 31.63 ms | 7.902 s / 28.49 ms | 7.999 s / 29.52 ms | 8.356 s / 29.75 ms |
| 128K | 42.336 s / 34.29 ms | 16.375 s / 28.71 ms | 16.626 s / 29.66 ms | 17.497 s / 29.96 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.132 s / 1.25 ms | 0.043 s / 0.85 ms | 0.043 s / 1.86 ms | 0.121 s / 2.00 ms |
| 16K | 0.486 s / 2.00 ms | 0.148 s / 0.71 ms | 0.170 s / 1.86 ms | 0.245 s / 2.07 ms |
| 32K | 1.863 s / 2.62 ms | 0.449 s / 0.64 ms | 0.486 s / 1.80 ms | 0.651 s / 2.02 ms |
| 64K | 7.263 s / 4.08 ms | 1.169 s / 0.95 ms | 1.266 s / 1.98 ms | 1.623 s / 2.21 ms |
| 128K | 28.860 s / 6.74 ms | 2.899 s / 1.16 ms | 3.149 s / 2.11 ms | 4.021 s / 2.41 ms |

### Qwen3.8, TP1, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 7.719 s / 37.94 ms | 7.271 s / 34.44 ms | 7.209 s / 35.66 ms | 8.006 s / 35.84 ms |
| 16K | 17.440 s / 40.86 ms | 14.986 s / 34.84 ms | 14.847 s / 36.06 ms | 15.734 s / 36.24 ms |
| 32K | 41.839 s / 46.45 ms | 31.321 s / 35.20 ms | 31.126 s / 36.36 ms | 32.930 s / 36.52 ms |
| 64K | 112.042 s / 55.77 ms | 64.956 s / 35.84 ms | 64.427 s / 36.73 ms | 68.309 s / 37.02 ms |
| 128K | 342.808 s / 74.66 ms | 135.454 s / 36.57 ms | 134.686 s / 37.44 ms | 145.785 s / 37.70 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.999 s / 4.86 ms | 0.552 s / 1.37 ms | 0.489 s / 2.58 ms | 1.287 s / 2.77 ms |
| 16K | 3.930 s / 8.01 ms | 1.476 s / 2.00 ms | 1.336 s / 3.21 ms | 2.223 s / 3.39 ms |
| 32K | 14.834 s / 13.60 ms | 4.316 s / 2.35 ms | 4.122 s / 3.51 ms | 5.925 s / 3.68 ms |
| 64K | 58.018 s / 22.93 ms | 10.932 s / 3.00 ms | 10.403 s / 3.89 ms | 14.285 s / 4.18 ms |
| 128K | 234.769 s / 41.85 ms | 27.415 s / 3.75 ms | 26.646 s / 4.63 ms | 37.746 s / 4.88 ms |

### Qwen3.8, TP4, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.566 s / 22.07 ms | 3.340 s / 21.26 ms | 3.394 s / 22.33 ms | 3.490 s / 22.56 ms |
| 16K | 7.720 s / 23.06 ms | 6.943 s / 21.37 ms | 6.990 s / 22.47 ms | 7.134 s / 22.74 ms |
| 32K | 17.454 s / 24.26 ms | 14.200 s / 21.46 ms | 14.228 s / 22.67 ms | 14.598 s / 22.92 ms |
| 64K | 43.454 s / 27.20 ms | 29.021 s / 21.54 ms | 28.983 s / 22.74 ms | 29.774 s / 22.98 ms |
| 128K | 120.498 s / 32.77 ms | 59.487 s / 21.90 ms | 59.328 s / 23.04 ms | 61.438 s / 23.22 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.417 s / 1.69 ms | 0.191 s / 0.87 ms | 0.245 s / 1.95 ms | 0.342 s / 2.18 ms |
| 16K | 1.227 s / 2.59 ms | 0.451 s / 0.90 ms | 0.497 s / 2.01 ms | 0.641 s / 2.27 ms |
| 32K | 4.488 s / 3.81 ms | 1.235 s / 1.02 ms | 1.262 s / 2.23 ms | 1.632 s / 2.48 ms |
| 64K | 17.501 s / 6.79 ms | 3.068 s / 1.14 ms | 3.030 s / 2.33 ms | 3.820 s / 2.57 ms |
| 128K | 68.578 s / 12.40 ms | 7.566 s / 1.52 ms | 7.408 s / 2.66 ms | 9.518 s / 2.84 ms |

### K2 Horizon, TP1, batch 1, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.158 s / 37.42 ms | 1.188 s / 38.43 ms | 1.204 s / 39.59 ms | 1.497 s / 40.28 ms |
| 16K | 2.604 s / 38.19 ms | 2.530 s / 38.44 ms | 2.571 s / 39.78 ms | 2.874 s / 40.39 ms |
| 32K | 6.349 s / 39.07 ms | 5.889 s / 38.75 ms | 5.965 s / 39.92 ms | 6.568 s / 40.55 ms |
| 64K | 17.191 s / 40.95 ms | 13.262 s / 38.82 ms | 13.472 s / 40.38 ms | 14.870 s / 40.97 ms |
| 128K | 53.861 s / 44.31 ms | 29.707 s / 39.10 ms | 30.287 s / 40.65 ms | 33.709 s / 41.32 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.160 s / 1.31 ms | 0.190 s / 2.33 ms | 0.206 s / 3.48 ms | 0.499 s / 4.17 ms |
| 16K | 0.622 s / 2.04 ms | 0.548 s / 2.28 ms | 0.589 s / 3.63 ms | 0.892 s / 4.24 ms |
| 32K | 2.385 s / 2.97 ms | 1.925 s / 2.65 ms | 2.001 s / 3.82 ms | 2.604 s / 4.45 ms |
| 64K | 9.016 s / 4.90 ms | 5.087 s / 2.77 ms | 5.297 s / 4.33 ms | 6.695 s / 4.92 ms |
| 128K | 37.987 s / 8.35 ms | 13.833 s / 3.14 ms | 14.413 s / 4.69 ms | 17.835 s / 5.36 ms |

### K2 Horizon, TP1, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 9.272 s / 46.33 ms | 9.648 s / 46.50 ms | 9.810 s / 47.96 ms | 11.297 s / 48.88 ms |
| 16K | 21.121 s / 49.83 ms | 20.369 s / 47.23 ms | 20.705 s / 49.05 ms | 22.067 s / 50.09 ms |
| 32K | 51.895 s / 56.92 ms | 49.761 s / 48.67 ms | 49.940 s / 50.84 ms | 53.470 s / 51.86 ms |
| 64K | 141.965 s / 69.56 ms | 115.377 s / 49.54 ms | 115.255 s / 51.54 ms | 125.032 s / 52.53 ms |
| 128K | Does not fit B8 | Does not fit B8 | Does not fit B8 | 290.698 s / 54.53 ms |

The 128K INT4 cell is a capacity result; BF16 and full-attention caches do
not fit eight K2 requests on one MI325X. It was measured separately with the
same eight prompts and a 0.8 GPU-memory fraction. Because a full-style dummy
also cannot keep B8 live at 128K, no attention-only subtraction is reported
for that capacity-only cell.

Attention core (real minus dummy), matched through 64K:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 1.255 s / 5.68 ms | 1.632 s / 5.85 ms | 1.793 s / 7.31 ms | 3.280 s / 8.23 ms |
| 16K | 5.083 s / 9.17 ms | 4.331 s / 6.57 ms | 4.667 s / 8.39 ms | 6.029 s / 9.43 ms |
| 32K | 19.821 s / 16.38 ms | 17.687 s / 8.13 ms | 17.866 s / 10.30 ms | 21.396 s / 11.32 ms |
| 64K | 77.422 s / 28.84 ms | 50.834 s / 8.82 ms | 50.712 s / 10.82 ms | 60.489 s / 11.81 ms |

### K2 Horizon, TP4, batch 8, top-8

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 3.792 s / 22.60 ms | 3.977 s / 23.81 ms | 4.088 s / 24.90 ms | 4.623 s / 25.42 ms |
| 16K | 8.389 s / 23.59 ms | 8.071 s / 23.75 ms | 8.443 s / 25.82 ms | 9.014 s / 26.38 ms |
| 32K | 19.231 s / 25.23 ms | 19.587 s / 24.14 ms | 19.948 s / 26.39 ms | 21.134 s / 26.93 ms |
| 64K | 49.199 s / 28.61 ms | 44.810 s / 24.36 ms | 45.263 s / 26.71 ms | 48.224 s / 27.26 ms |
| 128K | 139.555 s / 35.55 ms | 101.314 s / 24.86 ms | 102.023 s / 27.17 ms | 109.824 s / 28.62 ms |

Attention core (real minus dummy):

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.426 s / 2.17 ms | 0.611 s / 3.37 ms | 0.722 s / 4.47 ms | 1.257 s / 4.99 ms |
| 16K | 1.704 s / 3.16 ms | 1.386 s / 3.31 ms | 1.758 s / 5.39 ms | 2.329 s / 5.95 ms |
| 32K | 5.865 s / 4.80 ms | 6.220 s / 3.70 ms | 6.582 s / 5.96 ms | 7.768 s / 6.50 ms |
| 64K | 22.473 s / 8.18 ms | 18.085 s / 3.93 ms | 18.537 s / 6.28 ms | 21.498 s / 6.83 ms |
| 128K | 86.105 s / 15.11 ms | 47.864 s / 4.42 ms | 48.573 s / 6.73 ms | 56.374 s / 8.18 ms |

### Qwen3.8 with DFlash2, TP1, batch 1, top-8

This uses the same eight-prompt cohort and seven-token draft as the archived
DFlash2 panel. Decode measures the complete speculative loop; its time depends
on draft acceptance as well as target attention. All LoD prompt hashes were
checked against the same online dataset cohort.

End-to-end wall time:

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.978 s / 8.10 ms | 0.923 s / 8.49 ms | 0.921 s / 8.45 ms | 0.999 s / 8.06 ms |
| 16K | 2.149 s / 8.82 ms | 1.846 s / 7.82 ms | 1.855 s / 8.28 ms | 1.931 s / 7.44 ms |
| 32K | 5.204 s / 9.44 ms | 3.842 s / 8.76 ms | 3.898 s / 9.80 ms | 4.056 s / 8.85 ms |
| 64K | 14.138 s / 9.94 ms | 7.942 s / 8.80 ms | 8.051 s / 8.34 ms | 8.431 s / 9.31 ms |
| 128K | 42.817 s / 16.86 ms | 16.544 s / 8.99 ms | 16.772 s / 10.50 ms | 17.620 s / 9.85 ms |

Each next cell is `target verification-cycle milliseconds / pooled mean output
tokens per cycle`. It separates verifier cost from trajectory-dependent draft
acceptance.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 39.88 ms / 4.93 | 39.75 ms / 4.69 | 39.78 ms / 4.73 | 39.83 ms / 4.96 |
| 16K | 41.90 ms / 4.77 | 39.95 ms / 5.12 | 39.93 ms / 4.85 | 40.15 ms / 5.42 |
| 32K | 44.88 ms / 4.77 | 40.26 ms / 4.61 | 40.05 ms / 4.10 | 40.21 ms / 4.56 |
| 64K | 51.79 ms / 5.23 | 40.75 ms / 4.65 | 40.34 ms / 4.86 | 40.59 ms / 4.37 |
| 128K | 63.56 ms / 3.78 | 41.57 ms / 4.64 | 40.66 ms / 3.88 | 40.92 ms / 4.16 |

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
