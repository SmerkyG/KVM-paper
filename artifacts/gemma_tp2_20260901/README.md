# Gemma 4 TP2 attention and DFlash panel

This is the matched TP2 panel for `google/gemma-4-26B-A4B-it`, with and
without the original `z-lab/gemma-4-26B-A4B-it-DFlash` drafter. TP2 is the
natural tensor-parallel split for Gemma's global attention: each rank has
`(QH, KVH, GQA, D) = (8, 1, 8, 512)`. It therefore preserves the target's
GQA-8 reuse without replicating either of its two global KV heads.

Full attention uses the audited `TRITON_ATTN` implementation on the five
global layers. LOD uses BF16 leaves and the fused recursive three-tier route,
with prefill top three and decode top eight for ordinary generation. The
DFlash profile uses the TP1-validated prefill routes: top four through 64K,
top three at 128K, and decode top eight throughout. The runtime audit confirms
the expected local TP2 geometry and no attention fallback.

## Protocol

All speed runs use two MI325X GPUs, graph capture, vLLM custom TP all-reduce
with PYNCCL fallback, and a 16,384-token aggregate scheduler budget. Prompts
are distinct, non-repeated real documents from
`Seerkfang/prolong-64k-512-new`, chat formatted with a final summarization
request. Full and LOD prompt SHA-256 hashes match at every length. Each point
has one warmup followed by three measured repetitions producing 256 tokens;
tables report medians. Prefill is complete-model cold-prefill wall time.
Ordinary decode is milliseconds per emitted token at B1 and per emitted batch
step at B8.

The DFlash checkpoint is the original 16-position design: one anchor plus 15
proposed tokens. Its emitted-token decode latency depends on acceptance.
Verifier-cycle milliseconds divide complete decode wall time by scheduler
iterations that actually verify drafts and are the cleaner compute
comparison. They still include drafting, target execution, sampling, and
scheduler work. At B8, the average cycle also reflects the active-row mix as
requests finish on different iterations.

## Ordinary greedy generation

### Batch one

The auxiliary two-tier arm is included here as well. At TP2, three-tier is
slightly faster than two-tier in both phases across this panel, so three-tier
is the primary TP2 LOD result.

| context | full prefill (s) | two-tier prefill (s) | three-tier prefill (s) | full decode (ms) | two-tier decode (ms) | three-tier decode (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | **0.223** | 0.243 | 0.236 | **5.119** | 5.488 | 5.465 |
| 16K | 0.515 | 0.502 | **0.493** | 6.356 | 5.528 | **5.483** |
| 32K | 1.267 | 1.055 | **1.046** | 7.350 | 5.612 | **5.501** |
| 64K | 3.422 | 2.223 | **2.198** | 8.677 | 5.760 | **5.537** |
| 128K | 10.355 | 4.757 | **4.701** | 11.337 | 5.954 | **5.569** |

At 128K, three-tier LOD is 2.20x faster than full attention in prefill and
2.04x faster in decode.

### Batch eight

| context | full prefill (s) | three-tier prefill (s) | prefill speedup | full decode (ms) | three-tier decode (ms) | decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | **1.773** | 1.912 | 0.93x | **6.909** | 7.487 | 0.92x |
| 16K | 4.105 | **4.098** | 1.00x | 8.137 | **7.548** | 1.08x |
| 32K | 10.172 | **8.560** | 1.19x | 9.201 | **7.487** | 1.23x |
| 64K | 27.517 | **18.107** | 1.52x | 10.436 | **7.517** | 1.39x |
| 128K | 83.091 | **39.100** | 2.13x | 13.032 | **7.547** | 1.73x |

The optional two-tier B8 arm hit the external-cache mixed-admission retained-
prefix guard before measurement and is excluded. The requested full-versus-
LOD comparison is complete through 128K with the faster three-tier decoder.

## Original DFlash

### Batch one

| context | full prefill (s) | LOD prefill (s) | prefill speedup | full decode (ms) | LOD decode (ms) | full verifier cycle (ms) | LOD verifier cycle (ms) | cycle speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | **0.226** | 0.240 | 0.94x | **1.869** | 3.819 | **12.880** | 13.195 | 0.98x |
| 16K | 0.518 | **0.502** | 1.03x | **4.703** | 5.685 | 16.655 | **13.806** | 1.21x |
| 32K | 1.282 | **1.091** | 1.18x | 8.248 | **2.195** | 24.177 | **14.351** | 1.68x |
| 64K | 3.460 | **2.316** | 1.49x | 23.488 | **2.258** | 39.407 | **15.595** | 2.53x |
| 128K | 10.411 | **4.985** | 2.09x | 52.011 | **1.754** | 69.804 | **17.922** | 3.90x |

### Batch eight

| context | full prefill (s) | LOD prefill (s) | prefill speedup | full decode (ms) | LOD decode (ms) | full verifier cycle (ms) | LOD verifier cycle (ms) | cycle speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | **1.810** | 1.947 | 0.93x | **9.107** | 9.209 | **16.587** | 19.251 | 0.86x |
| 16K | 4.192 | **4.111** | 1.02x | 16.079 | **11.192** | 18.535 | **16.806** | 1.10x |
| 32K | 10.370 | **9.100** | 1.14x | 24.580 | **10.352** | 25.761 | **18.632** | 1.38x |
| 64K | 28.072 | **19.311** | 1.45x | 27.206 | **9.556** | 36.707 | **16.672** | 2.20x |
| 128K | 84.500 | **39.798** | 2.12x | 64.347 | **13.429** | 57.982 | **16.308** | 3.56x |

The very large emitted-token decode ratios at long context are acceptance-
trajectory outcomes. The verifier-cycle ratios are the appropriate claim
about the complete speculative compute loop.

## DFlash quality

The exact TP1 routing profiles transfer to TP2; TP2 does not require the
top-16 prefill routing used as a conservative TP4 control.

| context | TP2 recursive LOD DFlash NIAH-S3 |
|---:|---:|
| 8K | 8/8 |
| 16K | 8/8 |
| 32K | 8/8 |
| 64K | 8/8 |
| 128K | 8/8 |

The 8K/16K profile is top-four with a 32-row verifier bound, 32K/64K is
top-four with a 64-row bound, and 128K is fixed top-three with a 64-row bound.
Decode remains top eight in every case.

## TP interpretation

TP2 behaves as expected from the no-replication geometry. For ordinary B8
decode at 64K, current TP2 full/three-tier LOD are 10.436/7.517 ms. The
previous matched TP1 values are 12.783/10.023 ms, while TP4 measured
12.432/10.812 ms. TP2 is therefore the lowest-latency ordinary decode point;
TP4 loses the GQA-8 reuse and does not help this single-token workload.

DFlash exposes a different tradeoff because its target invocation has 16
positions and enough parallel work to benefit further from model sharding.
At B1/128K, full/LOD verifier cycles are 72.210/20.243 ms at TP1,
69.804/17.922 ms at TP2, and 80.210/14.261 ms at TP4. At B8/128K they are
57.787/19.415 ms at TP1, 57.982/16.308 ms at TP2, and 65.907/12.515 ms at
TP4. Thus TP2 is the best compromise and avoids full attention's TP4
regression, while TP4 still gives the lowest absolute LOD DFlash cycle by
spending twice as many GPUs.

## Raw records

Selected speed records:

- Ordinary full B1/B8: `normal_full_b1_8k128k_r3_d256.json` (cluster 12409)
  and `normal_full_b8_8k128k_r3_d256.json` (cluster 12410).
- Ordinary two-tier B1: `normal_lod2_b1_8k128k_r3_d256.json` (cluster 12408).
- Ordinary three-tier B1/B8: `normal_lod3_b1_8k128k_r3_d256.json` (cluster
  12411) and `normal_lod3_b8_8k128k_r3_d256.json` (cluster 12419).
- DFlash full B1/B8: `dflash_full_b1_8k128k_r3_d256.json` (cluster 12414) and
  `dflash_full_b8_8k128k_r3_d256.json` (cluster 12412).
- DFlash LOD B1: `dflash_lod3_top4_b1_8k128k_r3_d256.json` (cluster 12418).
- DFlash LOD B8: `dflash_lod3_top4_rows32_b8_8k16k_r3_d256.json` (cluster
  12415), `dflash_lod3_top4_rows64_b8_32k64k_r3_d256.json` (cluster 12416),
  and `dflash_lod3_top3_rows64_b8_128k_r3_d256.json` (cluster 12413).

Quality records are `dflash_lod3_top4_rows32_b8_niah8_8k16k.json` (cluster
12420), `dflash_lod3_top4_rows64_b8_niah8_32k64k.json` (cluster 12422), and
`dflash_lod3_top3_rows64_b8_niah8_128k.json` (cluster 12421).
