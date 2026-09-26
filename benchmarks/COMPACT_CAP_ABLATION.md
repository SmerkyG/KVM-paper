# Compact cap ablation

This ablation measures the quality effect of bounding exact refinement of a
routed centroid. The production cap leaves the top-eight centroid ranking
unchanged, but a selected centroid containing more than 1,024 archived KV
tokens remains represented by its coarse summary instead of opening its exact
leaves. “Unbounded” opens every selected centroid regardless of its leaf count.
Both conditions use two-tier BF16 LoD Attention.

## ProLong prompt loss

These measurements use the same eight 65,536-token documents from
`Seerkfang/prolong-64k-512-new` revision
`97295b7d7fe48dc0aa6ba373af3a8b9d945e505b`. The document and token hashes
match between conditions. Each result contains 524,280 predicted tokens, and
loss is token-weighted across the eight documents.

| Model | Unbounded loss | Cap 1,024 loss | Loss change | Unbounded PPL | Cap 1,024 PPL | Relative PPL change |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 0.445054 | 0.445288 | +0.000235 | 1.560574 | 1.560940 | +0.0235% |
| K2-Horizon-32B-FP8 | 0.496713 | 0.496789 | +0.000076 | 1.643311 | 1.643435 | +0.0076% |

The cap therefore changes perplexity by less than 0.025% on either model. See
[PROLONG.md](PROLONG.md) for the complete quality protocol and the comparisons
against full attention.

## LongBench v2

The full LongBench-v2 comparison evaluates all 503 examples per model with the
same deterministic A/B/C/D evaluator, 131,072-token input limit, disabled
thinking, and top-eight two-tier BF16 LoD configuration.

| Model | Unbounded | Cap 1,024 | Accuracy change | Correct-answer change |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 274/503 (54.47%) | 262/503 (52.09%) | -2.39 pp | -12 |
| K2-Horizon-32B-FP8 | 214/503 (42.54%) | 221/503 (43.94%) | +1.39 pp | +7 |
| Pooled | 488/1,006 (48.51%) | 483/1,006 (48.01%) | -0.50 pp | -5 |

The changes are not concentrated in the longest examples:

| Model and condition | Short (180) | Medium (215) | Long (108) |
|---|---:|---:|---:|
| Qwen, unbounded | 106 | 116 | 52 |
| Qwen, cap 1,024 | 100 | 109 | 53 |
| K2, unbounded | 85 | 83 | 46 |
| K2, cap 1,024 | 90 | 85 | 46 |

Across both models, the cap changes pooled accuracy by -0.50 percentage points.
Qwen moves downward while K2 moves upward, and neither model loses correct
answers in the long-example stratum. See [LONGBENCH_V2.md](LONGBENCH_V2.md)
for the full evaluation protocol.

## Comparability and interpretation

The tables above are archived release-before/after comparisons. They preserve
the model, data, evaluator, context limit, LoD organization, and top-eight
routing policy, but the runs have different source fingerprints because other
implementation work landed between the unbounded and capped measurements.
They should therefore be interpreted as a production-policy ablation rather
than a literal one-line source-code A/B.

As a direct isolation check, the cap was also tested with otherwise identical
source trees on the same deterministic 16-example LongBench-v2 subset. Qwen
moved from 8/16 to 9/16 and K2 from 8/16 to 10/16. That subset is too small to
support a standalone accuracy claim, but it provides no evidence that the cap
itself causes the Qwen decrease in the complete before/after comparison.

Taken together, ProLong shows a negligible next-token-prediction change, while
the complete LongBench-v2 results show mixed model-level movement and a small
pooled difference. The evidence supports using the 1,024-leaf cap to bound
exact refinement work without a material aggregate quality loss.
