# AMD two-tier threshold routing: targeted 64K screening

Follow-up: [AMD threshold kernel optimization and matched top-8 controls](../amd-threshold-fast-v2/REPORT.md). This newer report includes the compact-table implementation, complete speed runs and explicit quality and timing limitations.

Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, batch one, BF16 KV, two-tier LoD and greedy decoding. For each task, the eight fixed examples include four historical two-tier failures and four successes. This deliberately difficult, nonrepresentative subset is useful for screening but is not a full-panel accuracy estimate. Fresh fixed-eight controls and all threshold settings use identical tokenized prompts and answers within each model.

Threshold routing opens eligible regions with normalized-query proxy mass greater than c times the mean over nonempty remote regions; regions over 1,024 entries remain summaries. The existing AMD fixed-eight router uses native query scores, so the comparison changes query normalization as well as the selection rule. Prefill uses the existing fixed-eight LoD path. The last-step region counts below average across the sampled requests, layers, and query heads; they are not averages over all generated tokens.

## QWEN quality screening

| Routing | Multi-key 2 | Multi-key 3 | Mean regions per head at last step | Paired examples improved / declined |
|---|---:|---:|---:|---:|
| Fixed eight | 87.50% | 100.00% | fixed 8 maximum | 0 / 0 |
| c=8 | 75.00% | 100.00% | 55.35 | 0 / 1 |
| c=16 | 87.50% | 100.00% | 26.97 | 0 / 0 |
| c=32 | 75.00% | 100.00% | 12.99 | 0 / 1 |
| c=64 | 87.50% | 62.50% | 6.12 | 0 / 3 |

A single answer changes a task score by 12.5 points in this screen. The fresh fixed-eight controls reproduced historical failures as follows:

| Task | Historical failures still failing | Historical successes newly failing |
|---|---:|---:|
| niah_multikey_2 | 1 / 4 | 0 / 4 |
| niah_multikey_3 | 0 / 4 | 0 / 4 |

Historical and fresh runs differ in runtime and source revisions; the cause of changed outcomes has not been isolated. Qwen and K2 subsets were selected independently.

## K2 quality screening

| Routing | Multi-key 2 | Multi-key 3 | Mean regions per head at last step | Paired examples improved / declined |
|---|---:|---:|---:|---:|
| Fixed eight | 62.50% | 87.50% | fixed 8 maximum | 0 / 0 |
| c=8 | 62.50% | 87.50% | 64.71 | 0 / 0 |
| c=16 | 62.50% | 87.50% | 24.71 | 0 / 0 |
| c=32 | 62.50% | 87.50% | 9.46 | 0 / 0 |
| c=64 | 62.50% | 62.50% | 3.52 | 0 / 2 |
| c=32 GQA union | 62.50% | 87.50% | not recorded | 0 / 0 |

A single answer changes a task score by 12.5 points in this screen. The fresh fixed-eight controls reproduced historical failures as follows:

| Task | Historical failures still failing | Historical successes newly failing |
|---|---:|---:|
| niah_multikey_2 | 2 / 4 | 1 / 4 |
| niah_multikey_3 | 1 / 4 | 0 / 4 |

Historical and fresh runs differ in runtime and source revisions; the cause of changed outcomes has not been isolated. Qwen and K2 subsets were selected independently.

## Paired 32-example-per-task follow-up

These are the first 32 archived examples of each multi-key task, rather than the failure-enriched screen. Every row uses the same tokenized inputs and reference answers within each model.

| Model | Routing | Multi-key 2 | Multi-key 3 | Paired examples improved / declined |
|---|---|---:|---:|---:|
| QWEN | fixed eight | 100.00% | 100.00% | 0 / 0 |
| QWEN | c=16 | 100.00% | 100.00% | 0 / 0 |
| K2 | fixed eight | 84.38% | 90.62% | 0 / 0 |
| K2 | c=32 | 90.62% | 100.00% | 5 / 0 |
| K2 | c=32 GQA union | 90.62% | 96.88% | 5 / 1 |

A single answer changes a task score by 3.125 points. These two tasks are not a full-panel average.

## 64K end-to-end decode timing

The [compact threshold decode follow-up](../amd-threshold-fast/REPORT.md) has fresh 1,024-step top-8 / old c=16 / optimized c=16 comparisons on both models, plus implementation correctness and retrieval checks. The measurements below are the earlier 512-step runs.

The same batch-one ProLong prompt, 512 generated decode steps, one warmup and two timed repetitions on an MI325X. Timed decode includes the serving path and cache updates; prefill is excluded. Generation is autoregressive, so different methods can follow different token trajectories and open different regions; the timings are not quality matched.

| Model | Fixed eight ms/token | c=16 ms/token | c=32 ms/token |
|---|---:|---:|---:|
| QWEN | 28.474 | 29.780 | 28.288 |
| K2 | 38.842 | 44.256 | 42.009 |

Repeat-to-repeat greedy outputs differed in: K2 fixed eight, K2 c=32. This affects interpretation of small quality differences; the fixed-cache synthetic kernel checks were repeatable. The selected-region work differs across methods.

An experimental K2 c=32 GQA-union worker measured 40.485 ms/token on the same prompt, versus 42.009 for the first threshold worker and 38.842 for fixed eight. This worker shares pages across query heads and uses BF16 attention weights; repeated greedy output differed and it remains experimental.

The threshold path is opt-in and currently supports batch-one two-tier BF16. It is not a serving default, and this screening does not establish a best multiplier.

[Synthetic correctness](synthetic-determinism.json), [K2 kernel tuning](k2-synthetic-tuning.json), [Qwen kernel tuning](qwen-synthetic-tuning.json), [K2 GQA-union tuning](k2-union-synthetic-tuning.json), and [Qwen GQA-union tuning](qwen-union-synthetic-tuning.json).

[K2 union timing](k2-c32-union-64k-speed.json), [Qwen c=16 timing](qwen-c16-64k-speed.json), [Qwen c=32 timing](qwen-c32-64k-speed.json), [K2 c=16 timing](k2-c16-64k-speed.json), [K2 c=32 timing](k2-c32-64k-speed.json).

QWEN screening outputs: [fixed eight](qwen-ruler-difficult-top8.json), [c=8](qwen-ruler-difficult-8.json), [c=16](qwen-ruler-difficult-16.json), [c=32](qwen-ruler-difficult-32.json), [c=64](qwen-ruler-difficult-64.json).
K2 screening outputs: [fixed eight](k2-ruler-difficult-top8.json), [c=8](k2-ruler-difficult-8.json), [c=16](k2-ruler-difficult-16.json), [c=32](k2-ruler-difficult-32.json), [c=64](k2-ruler-difficult-64.json).
[K2 c=32 GQA-union screening outputs](k2-ruler-difficult-32-union.json).
QWEN 32-example-per-task outputs: [qwen-ruler-prefix32-top8](qwen-ruler-prefix32-top8.json), [qwen-ruler-prefix32-16](qwen-ruler-prefix32-16.json).
K2 32-example-per-task outputs: [k2-ruler-prefix32-top8](k2-ruler-prefix32-top8.json), [k2-ruler-prefix32-32](k2-ruler-prefix32-32.json), [k2-ruler-prefix32-32-union](k2-ruler-prefix32-32-union.json).
