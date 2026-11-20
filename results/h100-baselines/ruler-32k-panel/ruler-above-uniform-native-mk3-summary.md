# Above-uniform centroid mass: Llama 32K multi-key 3

Same 32 frozen examples; BF16 weights/cache, original two-tier sparse prefill and FP32 PyTorch decode. The rule opens a region when its native-query count-weighted centroid probability, normalized over nonempty remote regions, is strictly greater than 1 divided by the number of those regions. Local and sink are always exact; regions larger than 1024 entries remain summaries. This is a quality diagnostic, not an optimized speed test.

| Rule | First-response accuracy | Mean open regions per head | Mean exact remote entries per head | Mean selected estimated remote mass |
|---|---:|---:|---:|---:|
| Above uniform | 100.00% | 421.52 | 7360.02 | 86.71% |
| 70% top-p | 56.25% | 194.63 | 3763.97 | 70.32% |

Above-uniform versus top-p: 14 improved [68, 156, 193, 202, 236, 246, 302, 306, 327, 384, 390, 442, 447, 448]; 0 worsened [].
Above-uniform had 815 decode steps; counts ranged from 1 to 1142 per head-step. Mean skipped oversized candidate regions per head-step was 0.000000. The top-p run had 816 decode steps. Head averages are weighted by each run’s decode steps.
Nine independent checks of selection and explicit frontier had maximum absolute error 0.000000119.

[Per-layer, per-head statistics](ruler-above-uniform-native-mk3-heads.csv).
