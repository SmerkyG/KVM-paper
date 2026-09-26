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
| Qwen3.8-27B-FP8 | 0.445205 | 0.445288 | +0.000084 | 1.560810 | 1.560940 | +0.0084% |
| K2-Horizon-32B-FP8 | 0.496648 | 0.496789 | +0.000141 | 1.643203 | 1.643435 | +0.0141% |

The cap therefore changes perplexity by less than 0.015% on either model. See
[PROLONG.md](PROLONG.md) for the complete quality protocol and the comparisons
against full attention.

## LongBench v2

The full LongBench-v2 comparison evaluates all 503 examples per model with the
same deterministic A/B/C/D evaluator, 131,072-token input limit, disabled
thinking, and top-eight two-tier BF16 LoD configuration.

| Model | Unbounded | Cap 1,024 | Cap accuracy change | Cap correct-answer change |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | 271/503 (53.88%) | 262/503 (52.09%) | -1.79 pp | -9 |
| K2-Horizon-32B-FP8 | 209/503 (41.55%) | 221/503 (43.94%) | +2.39 pp | +12 |
| Pooled | 480/1,006 (47.71%) | 483/1,006 (48.01%) | +0.30 pp | +3 |

The changes are not concentrated in the longest examples:

| Model and condition | Short (180) | Medium (215) | Long (108) |
|---|---:|---:|---:|
| Qwen, unbounded | 105 | 114 | 52 |
| Qwen, cap 1,024 | 100 | 109 | 53 |
| K2, unbounded | 90 | 77 | 42 |
| K2, cap 1,024 | 90 | 85 | 46 |

Across both models, the cap changes pooled accuracy by +0.30 percentage points.
Qwen moves downward while K2 moves upward, and both capped runs gain correct
answers in the long-example stratum. See [LONGBENCH_V2.md](LONGBENCH_V2.md)
for the full evaluation protocol.

## Comparability and interpretation

The unbounded condition was freshly rerun on 2026-09-26 from release commit
`4bec54b1` with the production profile changed only from
`max_open_centroid_leaves = 1024` to `None`. These results replace the prior
archived unbounded artifacts whose exact source state was uncertain. The
production-cap columns are the current official results also reported in
[PROLONG.md](PROLONG.md) and [LONGBENCH_V2.md](LONGBENCH_V2.md). All fresh
LongBench-v2 outputs contain 503 unique IDs and 503 parsed A/B/C/D answers per
model. The ProLong conditions use matching document and token hashes and each
contains 524,280 predicted tokens.

Taken together, ProLong shows a negligible next-token-prediction change, while
the complete LongBench-v2 results move in opposite directions by model and are
slightly positive when pooled. The evidence supports using the 1,024-leaf cap
to bound exact refinement work without a material aggregate quality loss.
