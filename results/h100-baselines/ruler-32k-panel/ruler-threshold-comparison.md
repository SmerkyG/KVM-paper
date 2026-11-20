# Sort-free centroid thresholds on 32K multi-key 3

All rows use the same 32 frozen Llama-3.1-8B-Instruct examples, original LoD two-tier sparse prefill, BF16 weights and cache, and FP32 PyTorch decode. Thresholds open each nonempty remote region with native-query centroid probability strictly above the indicated multiple of the mean (1 / number of nonempty remote regions). A threshold selection rule can use a reduction and pointwise comparison without sorting; these tests are not an optimized production implementation. The percentile row is an order-statistic reference, implemented here by sorting. Unopened regions remain summaries; local and sink are always exact. The 1,024-entry cap applies. This is a quality diagnostic, not a latency benchmark.

| Selection | First-response accuracy | Open regions / head | Exact remote entries / head | Selected estimated remote mass |
|---|---:|---:|---:|---:|
| 70% mass top-p | 56.25% | 194.63 | 3763.97 | 70.32% |
| Above 1x mean | 100.00% | 421.52 | 7360.02 | 86.71% |
| Above 4x mean | 100.00% | 120.66 | 2439.18 | 65.69% |
| Above 8x mean | 96.88% | 53.96 | 1187.74 | 52.43% |
| 70th percentile | 100.00% | 859.25 | 13616.06 | 94.00% |

Mean counts weight each run by its own number of decode steps; no oversized candidate regions were skipped in the listed runs (see logs). Each selection and explicit frontier passed nine independent first-step checks. The 70%-mass top-p control and percentile rule use sorting in these diagnostic implementations. Threshold comparisons use a pointwise rule; the 1x experiment used sorting only in its diagnostic implementation, and 4x/8x did not sort in selection.

[Per-layer, per-head counts](ruler-threshold-comparison-heads.csv).

The 8x rule scored 31/32 and failed source row 68; its minimum was zero regions on an individual head-step, leaving those remote entries represented by summaries. The other rules scored 32/32 except 70%-mass top-p, which scored 18/32.

[1x outputs](ruler-above-uniform-native-mk3.json), [4x outputs](ruler-mean4-native-mk3.json), [8x outputs](ruler-mean8-native-mk3.json), [percentile outputs](ruler-percentile70-native-mk3.json), [4x run metadata](1794115246424758114-ruler-mean4-native-run.json), [8x run metadata](1794115252040169465-ruler-mean8-native-run.json), [percentile run metadata](1794115204041550329-ruler-percentile70-native-run.json), and [sort-free runner](../../../benchmarks/retro_ruler_mean_multiple_regions.py).
