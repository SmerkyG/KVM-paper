# Qwen3.5-35B-A3B decode crossover (batch 8, 1025 tokens)

Measured on one MI325X per run with vLLM 0.27.1. Full attention uses
`ROCM_AITER_UNIFIED_ATTN`; LOD uses recursive INT4 LOD with eight routed
regions. Both use batch size 8, 16,384 maximum batched tokens, and 4,096 as the
long-prefill threshold.

Each process first ran one unmeasured warm generation, then three measured
generations. A generation requested 1,025 tokens and timed the 1,024 intervals
between the first and last emitted tokens. Thus every measured LOD generation
amortizes four 256-token state-update boundaries.

| Context | Full ms/batch step | LOD ms/batch step | Full / LOD | LOD latency change |
|---:|---:|---:|---:|---:|
| 8,192 | 10.37 | 15.26 | 0.68x | +47.1% |
| 16,384 | 11.13 | 14.87 | 0.75x | +33.6% |
| 32,768 | 13.02 | 15.49 | 0.84x | +19.0% |
| 65,536 | 16.12 | 15.63 | 1.03x | -3.0% |
| 131,072 | 22.56 | 16.33 | 1.38x | -27.6% |

The measured crossover lies between 32k and 64k. Interpolating the latency
difference between those points gives roughly 59k context on a log2-length
axis (60k with linear-length interpolation).

The table reports wall-clock latency for one decode step across the whole
batch: one new token for each of eight requests. Per-request-token latency is
the displayed value divided by eight. See `full_<length>.json` and
`lod_<length>.json` for all three repetitions and memory diagnostics.
