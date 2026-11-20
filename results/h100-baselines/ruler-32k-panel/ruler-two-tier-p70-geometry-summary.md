# Adaptive 70%: native versus normalized routing scores

Both use two-tier LoD on the same 32 Llama multi-key 3 examples. The native-query score is the actual centroid attention logit plus log count. The normalized-query score is the existing LoD routing score; its softmax is a routing-proxy distribution, not the actual centroid attention mass. Each opens the smallest prefix covering 70% of its own distribution, then retains all other regions as summaries. Both use FP32 PyTorch decode and original sparse prefill; neither is an optimized timing run.

| 70% distribution | First-response score | Average open regions / query head | Average exact remote entries / query head |
|---|---:|---:|---:|
| Native query, actual centroid mass | 56.25% | 194.63 | 3763.97 |
| Normalized-query routing proxy | 100.00% | 364.26 | 7737.48 |

Normalized minus native: +43.75 percentage points; paired bootstrap 95% interval [+28.12, +62.50]. 14 improved ([68, 156, 193, 202, 236, 246, 302, 306, 327, 384, 390, 442, 447, 448]); 0 worsened ([]).

The native run has 816 decode steps; normalized has 815. Per-head means are weighted by decode steps separately for each run. The [layer-by-head CSV](ruler-two-tier-p70-geometry-layer-head.csv) reports all 1,024 actual heads.

| Head | Native avg | Normalized avg | Head | Native avg | Normalized avg |
|---:|---:|---:|---:|---:|---:|
| 0 | 189.17 | 353.65 | 16 | 190.45 | 377.46 |
| 1 | 192.63 | 354.73 | 17 | 201.10 | 384.78 |
| 2 | 170.77 | 338.77 | 18 | 205.16 | 378.82 |
| 3 | 192.44 | 353.24 | 19 | 231.99 | 394.05 |
| 4 | 187.64 | 370.09 | 20 | 197.97 | 357.52 |
| 5 | 215.77 | 383.54 | 21 | 226.83 | 374.69 |
| 6 | 215.30 | 368.88 | 22 | 235.15 | 375.06 |
| 7 | 217.39 | 382.63 | 23 | 199.03 | 363.75 |
| 8 | 161.86 | 353.61 | 24 | 201.09 | 356.41 |
| 9 | 167.41 | 352.65 | 25 | 186.58 | 350.41 |
| 10 | 180.93 | 358.31 | 26 | 150.77 | 321.41 |
| 11 | 160.82 | 351.59 | 27 | 191.02 | 353.35 |
| 12 | 185.27 | 358.35 | 28 | 235.14 | 396.00 |
| 13 | 168.69 | 347.65 | 29 | 194.55 | 365.97 |
| 14 | 214.94 | 364.62 | 30 | 209.22 | 388.18 |
| 15 | 183.00 | 355.70 | 31 | 167.97 | 370.48 |

Native minimum / maximum individual head-step: 1 / 1135; normalized: 1 / 1218. Mean skipped oversized regions per head-step: 0.000000 native and 0.000000 normalized.

For example, layer 15 / head 30 opened an average of 3.93 regions with the
native-query rule and 106.52 with the normalized-query rule. That is the head
in which earlier traces found an answer token receiving approximately 99% of
dense attention while its region was not selected. Layer 1 / head 2 averaged
1,015.87 regions under the native-query rule. These averages use different
generation trajectories; they do not by themselves establish which openings
caused a particular answer to change.

[Native 70% individual outputs](ruler-two-tier-p70-mk3.json),
[normalized-score 70% outputs](ruler-two-tier-p70-normalized-mk3.json),
[native detailed summary](ruler-two-tier-p70-mk3-summary.md),
[experiment script](../../../benchmarks/retro_ruler_adaptive_regions.py), and
[comparison script](../../../benchmarks/retro_ruler_adaptive_geometry_compare.py).
The [native run](1794114085166345466-ruler-two-tier-p70-mk3-run.json) and
[normalized-score run](1794114303356399874-ruler-two-tier-p70-normalized-mk3-run.json)
used `uv` via `cluster-run`.
