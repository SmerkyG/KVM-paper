# Three-tier 4× mean-mass region selection: Llama 32K multi-key 3

Same 32 frozen Llama-3.1-8B-Instruct prompts, BF16 model and cache. Three-tier selected regions open their two highest-scoring pages, and remaining entries in each opened region are combined in one residual summary; unselected regions remain summaries. The 4× rule opens regions whose native-query estimated mass is greater than four times the mean remote-region mass. LoD rows use their respective sparse-prefill modes; PyTorch rows use FP32 reference decode and existing cache updates. The fixed-eight routes use the original normalized-query score. The optimized baseline uses BF16 decode, and dense uses dense prefill/decode. This is a quality diagnostic, not an optimized latency run.

| Mode | First-response accuracy | Mean open regions per head | Mean exact remote entries per head | Mean exact pages per head |
|---|---:|---:|---:|---:|
| Three-tier 4x mean, PyTorch | 87.50% (28/32) | 120.86 | 1680.65 | 165.35 |
| Three-tier fixed 8, PyTorch | 53.12% (17/32) | 8.00 | 151.14 | 12.68 |
| Three-tier fixed 8, optimized BF16 | 50.00% (16/32) | — | — | — |
| Two-tier 4x mean, PyTorch | 100.00% (32/32) | 120.66 | 2439.18 | — |
| Dense | 100.00% (32/32) | — | — | — |

4× versus fixed-eight FP32: 12 improved [68, 78, 126, 139, 170, 193, 235, 236, 306, 436, 447, 455]; 1 worsened [202]. 4× failed rows: [59, 202, 302, 316].
The 4× run averaged 65.67% estimated remote region mass opened; region counts ranged from 1 to 271 per head-step, across 816 decode steps. Mean skipped oversized regions per head-step was 0.000000.
Nine independent first-step checks per PyTorch run validated selection, two-page selection, and the residual frontier. The fixed-eight reference versus optimized output on these nine heads had maximum absolute component difference 0.001953; this does not validate all heads or later steps.
First generated token mismatches: three-tier 4× vs three-tier fixed-eight FP32 0; three-tier 4× vs two-tier 4× 0.

[Per-layer, per-head counts](ruler-three-c4-heads.csv).

[Three-tier 4× outputs](ruler-three-c4-mk3.json), [fixed-eight reference outputs](ruler-three-fixed8-control.json), [4× run metadata](1794140902142150060-ruler-three-c4-mk3-run.json), [fixed-eight run metadata](1794140939033593185-ruler-three-fixed8-control-run.json), and [3-tier threshold runner](../../../benchmarks/retro_ruler_three_tier_mean_regions.py).
