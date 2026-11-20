# Llama 32K multi-key 3: adaptive 70% region mass

All rows use the same 32 frozen prompts and BF16 model/KV. The adaptive run uses original two-tier LoD sparse prefill and cache updates, and the same FP32 PyTorch decode approach as the fixed-8 and fixed-36 controls. For every query head it ranks the remote centroids by their **native-query, count-corrected** scores, opens the smallest prefix covering 70% of their estimated mass, and retains all other regions as summaries. Local and sink entries are excluded from the 70% denominator and receive exact attention. Regions over 1,024 entries stay summarized; any resulting coverage shortfall is reported below. Fixed-8/36 LoD ranks regions with a normalized query, so both the count rule and routing score differ. This is a quality diagnostic, not a speed measurement.

| Method | First-response score | Change from fixed 8, pp (paired 95% interval) |
|---|---:|---:|
| Adaptive 70% centroid mass | 56.25% | -28.12 [-46.88, -12.50] |
| Fixed 8 regions | 84.38% | +0.00 [+0.00, +0.00] |
| Fixed 36 regions | 100.00% | +15.62 [+3.12, +28.12] |
| Dense | 100.00% | +15.62 [+3.12, +28.12] |
| RetroInfer | 100.00% | +15.62 [+3.12, +28.12] |

Adaptive versus fixed 8: 1 improved and 10 worsened; improved source rows [126]; worsened rows [68, 156, 193, 202, 246, 327, 384, 390, 442, 448].

There were 816 decode steps across the 32 answers. Every query head and layer is sampled at every step. The mean is **194.63 open regions per query head**, with 3763.97 exact remote KV entries. Across individual head/layer/step observations, the minimum was 1 regions and the maximum was 1135. Mean selected estimated remote mass was 70.316%; mean skipped oversized regions per head-step was 0.000000.

## Per query head

These are query-head indices 0–31 averaged across all layers and decode steps; head 0 in different layers refers to different learned heads. The [layer-by-head CSV](ruler-two-tier-p70-mk3-layer-head.csv) gives all 1,024 separate heads, including minimum, maximum and exact-entry counts.

| Head | Mean open regions | Head | Mean open regions | Head | Mean open regions | Head | Mean open regions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 189.17 | 8 | 161.86 | 16 | 190.45 | 24 | 201.09 |
| 1 | 192.63 | 9 | 167.41 | 17 | 201.10 | 25 | 186.58 |
| 2 | 170.77 | 10 | 180.93 | 18 | 205.16 | 26 | 150.77 |
| 3 | 192.44 | 11 | 160.82 | 19 | 231.99 | 27 | 191.02 |
| 4 | 187.64 | 12 | 185.27 | 20 | 197.97 | 28 | 235.14 |
| 5 | 215.77 | 13 | 168.69 | 21 | 226.83 | 29 | 194.55 |
| 6 | 215.30 | 14 | 214.94 | 22 | 235.15 | 30 | 209.22 |
| 7 | 217.39 | 15 | 183.00 | 23 | 199.03 | 31 | 167.97 |

The lowest layer-averaged head was 26 (150.77); the highest was 22 (235.15). Layer means range from 70.30 (layer 25) to 345.54 (layer 31).

Nine independent first-step selection/frontier checks (three heads at layers 0, 15 and 31) had maximum absolute output error 0.000000119. These checks use the same optimized cache and do not validate the entire autoregressive run. The other two fixed-budget PyTorch controls were previously compared with the readable reference.

The [individual outputs and routing totals](ruler-two-tier-p70-mk3.json),
[comparison with normalized-score routing](ruler-two-tier-p70-geometry-summary.md),
[experiment script](../../../benchmarks/retro_ruler_adaptive_regions.py), and
[run metadata](1794114085166345466-ruler-two-tier-p70-mk3-run.json) are saved.
The run used `uv` via `cluster-run`.
