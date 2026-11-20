# Llama 3.1: looser entry cutoffs with the original 16 sqrt(n) cap

Three selected 32K retrieval tasks, 32 matched examples each, BF16 model and KV, batch one, and identical sparse prefill. Entry B considers a region when p_r/n_r > 1/(B sqrt(n)). Uncapped variants use the strict B sqrt(n) guard; cap-16 variants use a strict 16 sqrt(n) guard. When candidates exceed the cap, the guard opens a deterministic prefix in region order and leaves all later candidates as summaries. This can discard a later candidate even if it would fit. All rules are experimental and these selected tasks are not a full RULER average.

| Task | Region c=16 | Entry 16 | Entry 24 | Entry 32 | Entry 24, cap 16 | Entry 32, cap 16 |
|---|---:|---:|---:|---:|---:|---:|
| niah_single_3 | 100.00 | 96.88 | 100.00 | 100.00 | 100.00 | 100.00 |
| niah_multikey_2 | 100.00 | 68.75 | 93.75 | 93.75 | 93.75 | 96.88 |
| niah_multikey_3 | 100.00 | 71.88 | 93.75 | 100.00 | 93.75 | 100.00 |

Scores use the first response. A single-answer miss changes a task score by 3.125 points.

## Remote work on the final decode step

Means cover all layers and query heads at each example’s final decode step. Proxy mass is not exact attention mass. The region work audit did not include single-needle 3. The bound fraction uses each entry variant’s work cap; the region fraction uses 16 sqrt(n) for comparison.

| Task | Rule | Mean entries/head | Mean opened proxy mass | Max bound fraction |
|---|---|---:|---:|---:|
| niah_single_3 | Entry 16 | 271.7 | 32.76% | 0.399 |
| niah_single_3 | Entry 24 | 493.2 | 39.26% | 0.431 |
| niah_single_3 | Entry 32 | 739.3 | 44.11% | 0.396 |
| niah_single_3 | Entry 24, cap 16 | 490.7 | 38.89% | 0.647 |
| niah_single_3 | Entry 32, cap 16 | 741.5 | 43.84% | 0.793 |
| niah_multikey_2 | Region c=16 | 804.6 | 27.88% | 3.153 |
| niah_multikey_2 | Entry 16 | 277.4 | 26.47% | 0.633 |
| niah_multikey_2 | Entry 24 | 504.2 | 35.43% | 0.462 |
| niah_multikey_2 | Entry 32 | 788.7 | 40.99% | 0.464 |
| niah_multikey_2 | Entry 24, cap 16 | 505.1 | 35.61% | 0.694 |
| niah_multikey_2 | Entry 32, cap 16 | 778.9 | 41.43% | 0.999 |
| niah_multikey_3 | Region c=16 | 524.2 | 27.61% | 2.392 |
| niah_multikey_3 | Entry 16 | 305.1 | 32.93% | 0.528 |
| niah_multikey_3 | Entry 24 | 528.9 | 38.79% | 0.449 |
| niah_multikey_3 | Entry 32 | 774.8 | 45.68% | 0.383 |
| niah_multikey_3 | Entry 24, cap 16 | 526.1 | 39.24% | 0.674 |
| niah_multikey_3 | Entry 32, cap 16 | 783.0 | 45.51% | 0.767 |

## Whole-model timing

H100, batch-one BF16 Llama, graph-captured decode, matched teacher-forced prompts, one warmup and two measured repetitions. Each repetition has 512 untimed and 1,024 timed steps, including cache catch-up. Timings include page padding and all model layers.

| Context | Rule | Decode ms/token | Timed layer update events |
|---|---|---:|---:|
| 32,768 | Region c=16 | 7.817 | 128, 128 |
| 32,768 | Entry 16 | 7.997 | 128, 128 |
| 32,768 | Entry 24 | 8.404 | 128, 128 |
| 32,768 | Entry 32 | 8.821 | 128, 128 |
| 32,768 | Entry 24, cap 16 | 8.412 | 128, 128 |
| 32,768 | Entry 32, cap 16 | 8.774 | 128, 128 |
| 65,536 | Region c=16 | 8.302 | 128, 128 |
| 65,536 | Entry 16 | 8.255 | 128, 128 |
| 65,536 | Entry 24 | 8.714 | 128, 128 |
| 65,536 | Entry 32 | 9.070 | 128, 128 |
| 65,536 | Entry 24, cap 16 | 8.708 | 128, 128 |
| 65,536 | Entry 32, cap 16 | 9.069 | 128, 128 |

The hard cap covers exact remote KV entries per query head, not local/sink work, padded page lanes, summary scoring, prefill, or cache maintenance. The full execution has not been proven subquadratic.

Frozen cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
Outputs: [Region c=16](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json) · [Entry 16](threshold-entry-guard-ruler-32k-full13.json) · [Entry 24](threshold-entry-24-ruler32k-retrieval.json) · [Entry 32](threshold-entry-32-ruler32k-retrieval.json) · [Entry 24, cap 16](entry-24-cap16-ruler32k-retrieval.json) · [Entry 32, cap 16](entry-32-cap16-ruler32k-retrieval.json).
