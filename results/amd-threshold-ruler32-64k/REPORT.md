# Optimized AMD decode: 32-example RULER panel at 64K

All 13 tasks, 32 examples per task and model, 416 paired requests per model. Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8 use batch-one two-tier LoD with BF16 KV and equal-per-region c=16 routing. The new path is the materialized-index/GQA-shared implementation from the [kernel optimization report](../amd-threshold-fast-v2/REPORT.md).

The old and optimized kernels use the same frozen first-32 requests, answers, generation budgets and seed 1234. There is no failure-based selection. Prefill retains top-8 routing and its exact first 16,384-token chunk. Only decode implementation settings differ; optimized settings match the preceding speed measurements. All GPU jobs use uv through cluster-run with frozen source and request hashes.

The archived dataset has a nominal 65,536-token context. Actual model-tokenized prompt ranges are QWEN: 59,700–65,511; K2: 58,771–65,496 tokens. Every generated request is checked against a separate CPU retokenization of its frozen archived chat; prompts are not padded or shortened.

## Mean over all 13 tasks

| Model | Archived dense | Old c=16 | Optimized c=16 | Difference, points | Improved / declined requests |
|---|---:|---:|---:|---:|---:|
| QWEN | 96.15 | 94.99 | 94.67 | -0.32 | 6 / 5 |
| K2 | 93.23 | 92.17 | 92.32 | +0.15 | 4 / 1 |

Scores are percentages; means weight all 13 tasks equally. The original RULER substring-recall metric permits fractional per-example scores. Archived dense scores use the identical requests and answers, checked by hashes and request arguments, but come from a different runtime and scheduling configuration. They provide historical context. The paired implementation comparison is between the two fresh c=16 runs. Thirty-two examples per task can reveal regressions but cannot certify unchanged quality on the 500-example benchmark.

The earlier [500-example threshold versus fresh top-8 comparison](../amd-threshold/full500/REPORT.md) covers four Qwen tasks and five K2 tasks. It evaluates the old c=16 implementation, not the optimized implementation tested here; it is not a complete 13-task threshold panel.

A separate [precision diagnostic](precision-diagnostic/REPORT.md) investigates changed Qwen examples. The [complete Qwen precision follow-up](precision-panel/REPORT.md) evaluates all 13 tasks with `LOD_AMD_THRESHOLD_PRECISE=1`; its results do not replace the original candidate below.

## QWEN task results

| Task | Archived dense | Old c=16 | Optimized c=16 | Difference, points | Improved / declined |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_single_3 | 100.00 | 100.00 | 96.88 | -3.12 | 0 / 1 |
| niah_multikey_1 | 100.00 | 100.00 | 96.88 | -3.12 | 0 / 1 |
| niah_multikey_2 | 100.00 | 96.88 | 100.00 | +3.12 | 1 / 0 |
| niah_multikey_3 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multiquery | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multivalue | 100.00 | 100.00 | 99.22 | -0.78 | 0 / 1 |
| ruler_cwe | 99.69 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| ruler_fwe | 100.00 | 97.92 | 100.00 | +2.08 | 2 / 0 |
| ruler_qa_hotpot | 75.00 | 65.62 | 59.38 | -6.25 | 0 / 2 |
| ruler_qa_squad | 75.26 | 74.48 | 78.39 | +3.91 | 3 / 0 |
| ruler_vt | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |

Paired bootstrap 95% interval for the mean difference: [-1.48, +0.74] points. This describes sampling uncertainty within the observed cohort; it does not establish noninferiority on the full benchmark.

## K2 task results

| Task | Archived dense | Old c=16 | Optimized c=16 | Difference, points | Improved / declined |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_single_2 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_single_3 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multikey_1 | 100.00 | 96.88 | 96.88 | +0.00 | 0 / 0 |
| niah_multikey_2 | 100.00 | 90.62 | 90.62 | +0.00 | 0 / 0 |
| niah_multikey_3 | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multiquery | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |
| niah_multivalue | 98.44 | 99.22 | 99.22 | +0.00 | 0 / 0 |
| ruler_cwe | 92.19 | 95.31 | 95.94 | +0.62 | 2 / 0 |
| ruler_fwe | 86.46 | 85.42 | 87.50 | +2.08 | 2 / 0 |
| ruler_qa_hotpot | 65.62 | 62.50 | 62.50 | +0.00 | 0 / 0 |
| ruler_qa_squad | 69.27 | 68.23 | 67.45 | -0.78 | 0 / 1 |
| ruler_vt | 100.00 | 100.00 | 100.00 | +0.00 | 0 / 0 |

Paired bootstrap 95% interval for the mean difference: [-0.10, +0.44] points. This describes sampling uncertainty within the observed cohort; it does not establish noninferiority on the full benchmark.

## Reproduction and audit

[All commands and jobs](jobs.json), [frozen requests and historical sources](cohort-manifest.json), [independent prompt-token audit](token-audit.json), [optimized settings](settings.json), [source hashes](source-manifest.json), [placement, status and exit codes](provenance.json), [changed predictions](changed-examples.json), [raw summary](REPORT.json).

Bootstrap intervals resample paired differences separately within each task, with 2,000 replicates and seed zero. The report rejects missing tasks, altered snapshots, failed jobs, inconsistent scoring, mismatched tokenized inputs or incorrect decode settings. The implementation remains opt-in; this report does not change serving defaults.
