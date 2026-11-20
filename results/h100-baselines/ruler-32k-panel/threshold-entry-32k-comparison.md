# Llama 3.1 two-tier threshold: per-entry versus per-region

32K RULER, BF16 model and KV, batch one, 32 matched examples per task. Both use the same sparse top-eight prefill and normalized-query proxy scores in decode. The existing region rule opens eligible regions above 16 times mean proxy mass and leaves regions over 1,024 entries coarse. The entry rule compares proxy mass per remote entry to 1/(16 sqrt(n)), where n is the current remote-entry count; it can open larger regions. The kernel uses a one-part-per-million conservative log threshold and an exact integer work guard, so a numerical boundary can close a region that the ideal rule would open.

## First-response quality

| Task | Per-region c=16 | Per-entry | Entry minus region, pp (paired 95% interval) |
|---|---:|---:|---:|
| niah_single_1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_2 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_single_3 | 100.00 | 96.88 | -3.12 [-9.38, +0.00] |
| niah_multikey_1 | 100.00 | 100.00 | +0.00 [+0.00, +0.00] |
| niah_multikey_2 | 100.00 | 68.75 | -31.25 [-46.88, -15.62] |
| niah_multikey_3 | 100.00 | 71.88 | -28.12 [-43.75, -12.50] |
| niah_multivalue | 99.22 | 99.22 | +0.00 [+0.00, +0.00] |
| niah_multiquery | 98.44 | 97.66 | -0.78 [-2.34, +0.00] |
| vt | 100.00 | 98.75 | -1.25 [-3.75, +0.00] |
| fwe | 88.54 | 89.58 | +1.04 [+0.00, +3.12] |
| cwe | 29.69 | 25.00 | -4.69 [-13.75, +4.06] |
| qa_1 | 68.75 | 65.62 | -3.12 [-9.38, +0.00] |
| qa_2 | 46.88 | 46.88 | +0.00 [+0.00, +0.00] |
| **RULER average** | **87.04** | **81.55** | -5.48 |

Scores stop at the first end-of-turn or end-of-text token; task means are weighted equally. The sample is small and task intervals are paired bootstrap intervals.

## Opened entries and proxy coverage

The following is sampled only on the **last decode step** of each example, across all layers and query heads. Proxy coverage is the sum of normalized proxy masses of opened regions; it is not exact attention mass. Counts exclude exact local and sink entries. The maximum ratio divides opened remote entries by 16 sqrt(n).

| Task | Rule | Mean opened entries/head | Mean opened regions/head | Mean proxy mass opened | Max bound fraction | Empty heads |
|---|---|---:|---:|---:|---:|---:|
| niah_multikey_2 | Per-region | 804.6 | 17.02 | 27.88% | 3.153 | 1.77% |
| niah_multikey_2 | Per-entry | 277.4 | 73.15 | 26.47% | 0.633 | 0.11% |
| niah_multikey_3 | Per-region | 524.2 | 18.02 | 27.61% | 2.392 | 1.61% |
| niah_multikey_3 | Per-entry | 305.1 | 76.57 | 32.93% | 0.528 | 0.02% |
| niah_multivalue | Per-region | 460.0 | 16.05 | 26.66% | 1.225 | 1.57% |
| niah_multivalue | Per-entry | 274.4 | 82.71 | 33.41% | 0.399 | 0.06% |
| niah_multiquery | Per-region | 446.1 | 15.99 | 26.72% | 1.308 | 1.16% |
| niah_multiquery | Per-entry | 282.9 | 87.66 | 35.10% | 0.394 | 0.02% |
| qa_1 | Per-region | 588.9 | 16.93 | 24.76% | 2.270 | 1.39% |
| qa_1 | Per-entry | 302.3 | 74.00 | 29.38% | 0.521 | 0.02% |

The fresh per-region audit differs from the earlier per-region run in 19 of 160 generated token sequences; the quality table uses the earlier full-panel run.

## Whole-model speed

H100, BF16 Llama, batch one, graph-captured decode and identical teacher-forced prompt tokens. Each result uses one warmup plus two measured repeats, 512 untimed decode steps and 1,024 timed steps. Prefill is reported separately. Timed decode includes host-side cache catch-up; the update count is summed over model layers.

| Context | Rule | Decode ms/token | Prefill seconds, median | Layer update events per timed window |
|---|---|---:|---:|---:|
| 32,768 | Per-region | 7.817 | 1.663 | 128, 128 |
| 32,768 | Per-entry | 7.997 | 1.661 | 128, 128 |
| 65,536 | Per-region | 8.302 | 4.195 | 128, 128 |
| 65,536 | Per-entry | 8.255 | 4.203 | 128, 128 |

## Scope of the work bound

The sum of opened remote region sizes is strictly below 16 sqrt(n) per query head: each opened region contributes more than its size divided by that bound to a proxy distribution summing to one. The integer guard checks the same bound against actual page-list lengths. It can conservatively discard a suffix of candidates at a numerical boundary. Padded page lanes and per-page bookkeeping mean this is not an identical bound on hardware operations. Local and sink work, region scoring, prefill, and cache updates are outside this bound.

The packing scan processes fixed 32-element tiles and carries totals forward once, giving linear work in the number of region tiles. The existing prefill still uses top-eight routing and a 1,024-entry per-region cap, rather than this entry rule. A normal catch-up routes each fixed-size overflow block against current centroids and updates the page index; the timed decode includes those updates. The usual unsplit 16 sqrt(T) state schedule suggests subquadratic work for idealized routing and fixed-size updates, but this is not a proof for complete execution: allocation, page-index maintenance, fallback paths, prefill kernels, model layers, and state-capacity behavior must all be included. No end-to-end asymptotic claim is made here.

Frozen cohort SHA-256: `f7ff96d79181b4424bdbd7085c3bbca4877b05ad09b635633cfd82603db538c6`.
[Per-entry RULER output](threshold-entry-guard-ruler-32k-full13.json) · [Prior per-region RULER output](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json) · [Fresh per-region audit](threshold-region-guard-ruler-32k-audit5.json).
