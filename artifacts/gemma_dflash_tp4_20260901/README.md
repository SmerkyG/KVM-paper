# Gemma 4 original DFlash at TP4

This is the matched TP4 panel for `google/gemma-4-26B-A4B-it` with the
original `z-lab/gemma-4-26B-A4B-it-DFlash` drafter. The drafter proposes 15
tokens and the target verifies one anchor plus those proposals in a
16-position block. This is original DFlash, not DFlash2.

The target has five global `D=512` attention layers. At TP4 their replicated
KV geometry is `(QH, KVH, GQA) = (4, 1, 4)` per rank. Full attention uses
`TRITON_ATTN`; the available AITER D=512 configuration exceeds gfx942's LDS
limit. LOD uses recursive three-tier BF16 target attention. The selected TP4
profile opens 16 prefill routes, keeps the normal top-eight decode routing,
and permits 64 speculative rows per verifier chunk.

## Protocol

All runs use four MI325X GPUs, graph capture, vLLM custom TP all-reduce with
PYNCCL fallback, and a 16,384-token aggregate scheduler budget. Prompts are
distinct, non-repeated real documents from
`Seerkfang/prolong-64k-512-new`, chat formatted with a final summarization
request. Full and LOD speed prompt SHA-256 hashes match at every length.
Sampling is greedy. Speed points have one warmup followed by three measured
repetitions producing 256 tokens. NIAH-S3 uses eight chat-formatted examples
and 64 generated tokens.

Verifier-cycle milliseconds divide complete decode wall time by scheduler
iterations that actually verify drafts. The metric includes drafting,
target execution, rejection/sampling, and scheduler work; it does not claim
to isolate an attention kernel. It removes the direct dependence on accepted
tokens. At B8, average cycle latency still reflects the number of active
request rows remaining in each launch, so the raw records retain both launch
and active-row counts.

## NIAH-S3 quality

The selected top-16/64-row LOD profile matches full attention throughout.

| context | full DFlash | recursive LOD DFlash |
|---:|---:|---:|
| 8K | 8/8 | 8/8 |
| 16K | 8/8 | 8/8 |
| 32K | 8/8 | 8/8 |
| 64K | 8/8 | 8/8 |
| 128K | 8/8 | 8/8 |

The small TP4 controls were more route-sensitive than TP1 or TP2. Top four
scored 7/8 at 8K and 32K, 8/8 at 16K and 64K, and 6/8 at 128K. Changing the
speculative row bound did not fix 128K: both 32- and 64-row top-four controls
scored 6/8. Top eight scored 8/8 at 8K, 16K, 32K, and 128K but 7/8 at 64K
under both row bounds. Top 16 scored 8/8 throughout under both row bounds, so
the 64-row variant is retained as the conservative measured profile because
it also performs fewer speculative chunks at B8. Top 16 affects prefill
routing only; decode remains top eight.

These eight-example observations do **not** establish a mathematical TP4
routing requirement. A given query head should rank the same centroids under
TP1, TP2, and TP4. TP4 changes the local execution geometry to GQA-4 because
Gemma's two KV heads are replicated over four ranks, so finite-precision
kernel tiling or a TP-specific implementation discrepancy can perturb a
discontinuous top-k boundary. The matched TP2 panel preserves GQA-8, uses the
TP1 top-four/top-three profiles, and scores 8/8 throughout; see
`../gemma_tp2_20260901/README.md`.

## Batch one

Decode time is milliseconds per emitted token. Large end-to-end decode ratios
can be acceptance-trajectory effects; verifier-cycle speedup is the cleaner
compute comparison.

| context | full prefill (s) | LOD prefill (s) | prefill speedup | full decode (ms) | LOD decode (ms) | decode speedup | full verifier cycle (ms) | LOD verifier cycle (ms) | cycle speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.129 | 0.147 | 0.88x | 1.809 | 1.766 | 1.02x | 12.813 | 9.581 | 1.34x |
| 16K | 0.299 | 0.320 | 0.94x | 4.744 | 2.335 | 2.03x | 17.279 | 9.924 | 1.74x |
| 32K | 0.724 | 0.681 | 1.06x | 8.455 | 3.204 | 2.64x | 26.292 | 10.494 | 2.51x |
| 64K | 1.923 | 1.481 | 1.30x | 15.098 | 1.816 | 8.31x | 44.254 | 11.875 | 3.73x |
| 128K | 5.801 | 3.293 | 1.76x | 39.319 | 1.234 | 31.87x | 80.210 | 14.261 | 5.62x |

## Batch eight

Decode time is milliseconds per emitted batch step.

| context | full prefill (s) | LOD prefill (s) | prefill speedup | full decode (ms) | LOD decode (ms) | decode speedup | full verifier cycle (ms) | LOD verifier cycle (ms) | cycle speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 1.032 | 1.229 | 0.84x | 9.365 | 7.134 | 1.31x | 14.215 | 13.140 | 1.08x |
| 16K | 2.369 | 2.668 | 0.89x | 11.183 | 6.280 | 1.78x | 18.582 | 14.214 | 1.31x |
| 32K | 5.764 | 5.740 | 1.00x | 17.654 | 7.576 | 2.33x | 25.725 | 13.371 | 1.92x |
| 64K | 15.511 | 12.586 | 1.23x | 35.360 | 6.478 | 5.46x | 40.261 | 13.073 | 3.08x |
| 128K | 46.895 | 28.248 | 1.66x | 72.390 | 10.682 | 6.78x | 65.907 | 12.515 | 5.27x |

## TP1 comparison

TP4 strengthens the long-context verifier-cycle result rather than weakening
it. At B1/128K, TP1 measured 72.210 ms full versus 20.243 ms LOD (3.57x),
while TP4 measures 80.210 ms full versus 14.261 ms LOD (5.62x). At B8/128K,
TP1 measured 57.787 ms versus 19.415 ms (2.98x), while TP4 measures 65.907 ms
versus 12.515 ms (5.27x).

The direction is plausible from the per-rank geometry. Splitting Gemma's 16
query heads leaves only four per rank, which makes the long regular D=512 full
attention launch less efficient and adds TP collectives. Full verifier cycles
are therefore slightly slower than TP1. LOD's much shorter scans lose less to
that head split while the rest of the target benefits from model sharding, so
its cycle becomes faster. This is a whole-cycle observation rather than an
isolated kernel attribution.

## Raw records

Selected results:

- Full B1 speed: `full_tp4_b1_8k128k_r3_d256.json` (cluster 12382).
- Full B8 quality and speed:
  `full_tp4_b8_niah8_speed_8k128k_r3_d256.json` (cluster 12384).
- Selected LOD quality: `lod3_top16_rows64_tp4_b8_niah8_8k128k.json`
  (cluster 12392).
- Selected LOD B1 speed:
  `lod3_top16_rows64_tp4_b1_speed_8k128k_r3_d256.json` (cluster 12394).
- Selected LOD B8 speed:
  `lod3_top16_rows64_tp4_b8_speed_8k128k_r3_d256.json` (cluster 12395).

Quality controls:

- TP1-selected top-four/top-three profiles: clusters 12385, 12386, and 12387.
- Top-four 128K row-bound controls: clusters 12388 and 12390.
- Top-eight 32- and 64-row panels: clusters 12389 and 12391.
- Top-sixteen 32-row confirmation: cluster 12393.
- TP4 full/LOD execution-audit smokes: clusters 12380 and 12381.
