# H100 Llama 3.1 threshold prefill: initial experiments

BF16 model and KV, batch one, with equal-region c=16 decode held fixed across the prefill variants. The entry rule applies to remote entries; exact local and sink attention remain. The entry-24 variant retains the strict 16 sqrt(n) cap using a deterministic region-order prefix.

The implementation is a correctness-first tiled PyTorch router with the existing grouped paged-leaf consumer, not an optimized prefill kernel. A synthetic H100 frontier check passed with maximum absolute output error 0.0034.

## Retrieval smoke test

One matched example per task. These are smoke tests, not task-quality estimates; each value below is for one example.

| Task | Fixed 8 prefill | Region c=16 | Entry 16 | Entry 24, cap 16 |
|---|---:|---:|---:|---:|
| niah_multikey_2 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_multikey_3 | 100.00% | 100.00% | 100.00% | 100.00% |
| niah_single_3 | 100.00% | 100.00% | 100.00% | 100.00% |

## Whole-model prefill timing

H100, 32,768 prompt tokens, matched teacher-forced prompt and BF16 Llama. All figures are medians of two measured repeats after a warmup. Differences are specific to this prototype.

| Prefill rule | Seconds |
|---|---:|
| Fixed 8 prefill | 1.663 |
| Region c=16 | 5.707 |
| Entry 16 | 14.504 |
| Entry 24, cap 16 | 18.130 |

## Selected multikey quality and remote prefill work

Eight matched examples per task, using first-response scoring. Work averages include every remote prefill query, query head, and layer, excluding exact initial, local, and sink fields. Proxy coverage is not exact attention mass. Maximum fraction uses 16 sqrt(n) for all variants; the region rule does not have that cap.

| Task | Rule | Score | Mean entries/head/query | Mean regions/head/query | Mean proxy mass | Max bound fraction |
|---|---|---:|---:|---:|---:|---:|
| niah_multikey_2 | Fixed 8 prefill | 100.00% | — | — | — | — |
| niah_multikey_2 | Region c=16 | 100.00% | 361.6 | 10.06 | 17.07% | 2.095 |
| niah_multikey_2 | Entry 16 | 100.00% | 215.8 | 69.03 | 24.25% | 0.765 |
| niah_multikey_2 | Entry 24, cap 16 | 100.00% | 421.8 | 116.41 | 32.33% | 0.945 |
| niah_multikey_3 | Fixed 8 prefill | 100.00% | — | — | — | — |
| niah_multikey_3 | Region c=16 | 100.00% | 268.4 | 10.48 | 17.08% | 1.977 |
| niah_multikey_3 | Entry 16 | 100.00% | 201.2 | 71.99 | 26.07% | 0.553 |
| niah_multikey_3 | Entry 24, cap 16 | 100.00% | 378.7 | 115.43 | 33.06% | 0.731 |

Audit outputs: [Region c=16](threshold-prefill-region-16-cap16-ruler32k-audit8.json) · [Entry 16](threshold-prefill-entry-16-cap16-ruler32k-audit8.json) · [Entry 24, cap 16](threshold-prefill-entry-24-cap16-ruler32k-audit8.json).

The entry bound covers logical exact remote entries per query head. Prefill also performs local attention, routing, page packing, and cache construction; no complete-execution subquadratic bound has been established.

Pilot outputs: [Fixed 8 prefill](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json) · [Region c=16](threshold-prefill-region-16-cap16-ruler32k-pilot.json) · [Entry 16](threshold-prefill-entry-16-cap16-ruler32k-pilot.json) · [Entry 24, cap 16](threshold-prefill-entry-24-cap16-ruler32k-pilot.json).
