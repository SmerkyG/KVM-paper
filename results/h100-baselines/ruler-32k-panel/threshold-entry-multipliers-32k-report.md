# Llama 3.1 32K retrieval: entry-bound multiples 16, 24, and 32

Three selected tasks, 32 matched examples per task, BF16 model and KV, batch one, same sparse prefill. An entry multiple B opens a region when proxy mass per entry exceeds 1/(B sqrt(n)), with a strict B sqrt(n) remote-entry guard. Higher B opens more candidates and allows more exact work. The per-region control opens regions above 16 times mean region proxy mass, capped at 1,024 entries per region. These tasks were selected because entry multiple 16 had misses; this is not a full RULER panel.

| Task | Per-region | Entry 16 | Entry 24 | Entry 32 |
|---|---:|---:|---:|---:|
| niah_single_3 | 100.00 | 96.88 | 100.00 | 100.00 |
| niah_multikey_2 | 100.00 | 68.75 | 93.75 | 93.75 |
| niah_multikey_3 | 100.00 | 71.88 | 93.75 | 100.00 |

First-response scoring stops at the first end-of-turn or end-of-text token. One example changes a single-answer score by 3.125 points.

## Last-step remote work and proxy mass

Means cover every layer and query head on the final decode step only. Proxy mass is normalized over remote regions and is not exact attention mass. The region control was not rerun with work instrumentation for single-needle 3.

| Task | Rule | Mean entries/head | Mean regions/head | Mean opened proxy mass | Max fraction of that rule’s bound |
|---|---|---:|---:|---:|---:|
| niah_single_3 | Entry 16 | 271.7 | 84.04 | 32.76% | 0.399 |
| niah_single_3 | Entry 24 | 493.2 | 131.17 | 39.26% | 0.431 |
| niah_single_3 | Entry 32 | 739.3 | 177.07 | 44.11% | 0.396 |
| niah_multikey_2 | Per-region | 804.6 | 17.02 | 27.88% | 3.153 |
| niah_multikey_2 | Entry 16 | 277.4 | 73.15 | 26.47% | 0.633 |
| niah_multikey_2 | Entry 24 | 504.2 | 117.97 | 35.43% | 0.462 |
| niah_multikey_2 | Entry 32 | 788.7 | 167.09 | 40.99% | 0.464 |
| niah_multikey_3 | Per-region | 524.2 | 18.02 | 27.61% | 2.392 |
| niah_multikey_3 | Entry 16 | 305.1 | 76.57 | 32.93% | 0.528 |
| niah_multikey_3 | Entry 24 | 528.9 | 118.90 | 38.79% | 0.449 |
| niah_multikey_3 | Entry 32 | 774.8 | 161.12 | 45.68% | 0.383 |

The per-region fraction is measured against 16 sqrt(n) for reference; the entry fractions use each rule’s own bound.

## Whole-model speed

H100 graph-captured BF16 Llama, batch one, identical teacher-forced prompt tokens. One warmup and two measured repetitions; 512 untimed plus 1,024 timed decode steps. Timed steps include cache catch-up, with layer-event counts shown per repetition.

| Context | Rule | Decode ms/token | Layer update events |
|---|---|---:|---:|
| 32,768 | Per-region | 7.817 | 128, 128 |
| 32,768 | Entry 16 | 7.997 | 128, 128 |
| 32,768 | Entry 24 | 8.404 | 128, 128 |
| 32,768 | Entry 32 | 8.821 | 128, 128 |
| 65,536 | Per-region | 8.302 | 128, 128 |
| 65,536 | Entry 16 | 8.255 | 128, 128 |
| 65,536 | Entry 24 | 8.714 | 128, 128 |
| 65,536 | Entry 32 | 9.070 | 128, 128 |

Each entry rule’s bound covers remote exact KV entries per query head; local/sink attention, page padding, prefill, and cache maintenance are separate. The bound is not a guarantee of retrieval quality.

Frozen cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
[Entry 16](threshold-entry-guard-ruler-32k-full13.json) · [Entry 24](threshold-entry-24-ruler32k-retrieval.json) · [Entry 32](threshold-entry-32-ruler32k-retrieval.json) · [Per-region control](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json).
