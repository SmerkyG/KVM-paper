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
| Qwen3.8-27B-FP8 | Full | all | 1.107629 | 3.027173 |
| Qwen3.8-27B-FP8 | Two-tier BF16 | 4 | 1.112470 | 3.041864 |
| Qwen3.8-27B-FP8 | Three-tier BF16 | 4 | 1.112427 | 3.041731 |
| Qwen3.8-27B-FP8 | Three-tier INT4 | 4 | 1.112835 | 3.042973 |
| K2-Horizon-32B-FP8 | Full | all | 0.324007 | 1.382657 |
| K2-Horizon-32B-FP8 | Two-tier BF16 | 4 | 0.323817 | 1.382394 |
| K2-Horizon-32B-FP8 | Three-tier BF16 | 4 | 0.323757 | 1.382312 |
| K2-Horizon-32B-FP8 | Three-tier INT4 | 4 | 0.323581 | 1.382068 |

Prompt loss exercises prefill only. Every LoD row above uses the release's
uniform top-4 prefill policy. Each measurement contains 524,280 predicted
tokens; the loss is token-weighted across all eight documents. All eight rows
were freshly rerun after the release cleanup.

## Matched speed results

These measurements use AMD MI325X with vLLM 0.27.1. Every LoD cell, both Qwen
full-attention panels, and the K2 TP4 full-attention panel were freshly
collected on 2026-09-12. The unchanged K2 TP1 and DFlash2 full-attention
controls are retained from their preceding matched sweeps because the cleanup
does not touch native attention. Each cell is `prefill seconds / decode
milliseconds per batch step`; lower is better. One B8 decode step emits eight
tokens concurrently.

Prompts are raw, distinct ProLong token streams, concatenating different
documents when needed and never repeating a document to fill a request. The
older internal summary called these chat-formatted, but its retained prompt
metadata and runner show that raw `prompt_token_ids` were used. The scheduler
chunk is 16,384 tokens. Decode generates 1,025 tokens and measures the final
1,024 steps. This includes four 256-token LoD state updates in the regular
path, or two 512-token updates for K2 INT4. Results are medians after one
warmup and three measured repetitions.

All displayed LoD cells use routed top-4 attention; the final release's exact
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
complete speculative target-and-draft loop. Every row and mode was run
sequentially on one otherwise idle MI325X from the final release checkout,
with generation seed 0, so GPU contention does not get conflated across arms.

| Context | Full | Two-tier BF16 | Three-tier BF16 | Three-tier INT4 |
|---:|---:|---:|---:|---:|
| 8K | 0.979 s / 9.00 ms | 0.930 s / 8.66 ms | 0.966 s / 8.29 ms | 1.043 s / 7.02 ms |
| 16K | 2.204 s / 6.33 ms | 1.968 s / 8.27 ms | 2.014 s / 7.22 ms | 2.095 s / 7.53 ms |
| 32K | 5.253 s / 7.03 ms | 4.063 s / 6.83 ms | 4.140 s / 7.38 ms | 4.268 s / 7.96 ms |
| 64K | 14.017 s / 11.44 ms | 8.360 s / 7.41 ms | 8.491 s / 7.62 ms | 8.747 s / 8.05 ms |
| 128K | 42.601 s / 10.68 ms | 17.249 s / 10.24 ms | 17.520 s / 10.60 ms | 18.091 s / 8.33 ms |

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
| 8K | 39.91 ms / 4.44 | 39.25 ms / 4.56 | 39.47 ms / 4.77 | 39.48 ms / 5.66 |
| 16K | 41.81 ms / 6.61 | 39.41 ms / 4.76 | 39.35 ms / 5.48 | 39.56 ms / 5.28 |
| 32K | 44.96 ms / 6.44 | 39.74 ms / 5.84 | 39.54 ms / 5.37 | 39.75 ms / 5.01 |
| 64K | 51.38 ms / 4.49 | 40.16 ms / 5.46 | 39.80 ms / 5.24 | 40.03 ms / 4.98 |
| 128K | 63.24 ms / 5.92 | 40.69 ms / 3.98 | 40.06 ms / 3.79 | 40.43 ms / 4.85 |

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
