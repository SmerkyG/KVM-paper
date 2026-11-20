# AMD two-tier threshold: paired 500-example RULER tasks

Qwen3.8-27B-FP8 and K2-Horizon-32B-FP8, batch one, BF16 KV, 64K archived RULER inputs, greedy generation, and two-tier sparse prefill. Each task has all 500 examples, with matched input tokens and answers across fresh fixed-eight and c=16 threshold runs. The threshold uses normalized-query mass; the existing fixed-eight AMD router uses native query scores, so both the routing score and selection rule change.

The archived dense and fixed-eight scores are included for context; those historical runs used a different runtime and code revision. Paired changes compare only the two fresh runs. QA and multi-value use the original RULER substring-recall metric, so an example can have a fractional score.

Archived dense and the fresh runs use the same 500 requests and answers per task: document, prompt, target, and request arguments were validated against the frozen cohort. The dense comparison remains descriptive because runtime and scheduling differ.

| Model | Task | Archived dense | Archived fixed eight | Fresh fixed eight | Threshold c=16 | vs fresh top-8, points | vs dense, points | Improved / declined vs fresh top-8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| QWEN | niah_multikey_2 | 100.00% | 96.20% | 98.60% | 98.60% | +0.00 | -1.40 | 4 / 4 |
| QWEN | niah_multikey_3 | 100.00% | 92.60% | 98.00% | 100.00% | +2.00 | +0.00 | 10 / 0 |
| QWEN | ruler_qa_hotpot | 72.00% | 67.20% | 65.60% | 65.80% | +0.20 | -6.20 | 13 / 12 |
| QWEN | ruler_qa_squad | 78.85% | 74.88% | 75.57% | 75.85% | +0.28 | -3.00 | 24 / 19 |
| K2 | niah_multikey_2 | 99.00% | 93.00% | 93.20% | 94.60% | +1.40 | -4.40 | 9 / 2 |
| K2 | niah_multikey_3 | 100.00% | 98.20% | 96.00% | 99.00% | +3.00 | -1.00 | 15 / 0 |
| K2 | niah_multivalue | 97.70% | 95.95% | 97.10% | 97.55% | +0.45 | -0.15 | 17 / 8 |
| K2 | ruler_qa_hotpot | 60.40% | 57.40% | 58.20% | 58.20% | +0.00 | -2.20 | 7 / 7 |
| K2 | ruler_qa_squad | 75.30% | 72.08% | 71.60% | 72.77% | +1.17 | -2.53 | 9 / 1 |

These are selected tasks, not a 13-task RULER average. Both fresh LoD paths use fixed-eight sparse prefill, so the experiment does not measure a threshold-prefill implementation.

Frozen cohort manifest: `/home/dan/subusers/agent/.codex/worktrees/h100-llama31/KVM-paper/results/amd-threshold/full500-frozen/manifest.json`.
