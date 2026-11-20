# Native 70% routing: answer-region audit

Four failures from the 32-example Llama-3.1-8B-Instruct 32K multi-key-3 cohort were replayed using the same sparse prefill and FP32 PyTorch native-query 70% decode. Each replay checked the prompt hash and reproduced the recorded generated tokens through the first divergence from the independently generated dense baseline. At that decode step, the audit computed exact attention over the replay's own query and cache, and independently recomputed the adaptive region selection. The comparison isolates the frontier for that actual LoD trajectory; it is not an intervention proving that this head caused the final answer to be wrong.

The row for each example shows the omitted answer-token region with the largest individual answer-token dense attention at that step. Region true mass and the estimated mass use the same *exact dense partition function* as denominator; estimated mass is not its probability in the centroid-only routing distribution. The adaptive selection is based on the 70% prefix of that centroid-only distribution over remote regions. All regions in these traces were within the 1024-entry opening cap.

| Row | First divergence index | Layer / query head | Answer token true attention | Region true attention | Region centroid estimate / dense partition | Entries in region | Region rank | Regions opened |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 68 | 18 | 15 / 30 | 99.894% | 99.897% | 9.57e-10 | 66 | 17 | 2 |
| 156 | 5 | 16 / 20 | 99.024% | 99.030% | 2.42e-7 | 33 | 16 | 14 |
| 193 | 16 | 15 / 30 | 98.750% | 98.750% | 3.31e-9 | 11 | 30 | 5 |
| 202 | 21 | 15 / 30 | 99.162% | 99.948% | 3.92e-10 | 95 | 48 | 4 |

The raw audit also includes outputs from the earlier fixed-eight audit routine. Its `reference_vs_optimized_*` fields compare that fixed-eight reference with the 70% output and are **not** an implementation correctness check for the 70% run. Other `*_region_mass` and `*_relative_l2` fields referring to fixed-eight routes should likewise not be interpreted as 70% routing metrics; the `p70_*` fields and dense and centroid metrics above apply to this experiment. The earlier complete 70% run independently checked its reference frontier.

[Raw trace](ruler-p70-frontier-trace.json), [run metadata](1794114696449046436-ruler-p70-frontier-trace-run.json), [audit script](../../../benchmarks/retro_ruler_p70_frontier_audit.py), [70% aggregate comparison](ruler-two-tier-p70-geometry-summary.md).
