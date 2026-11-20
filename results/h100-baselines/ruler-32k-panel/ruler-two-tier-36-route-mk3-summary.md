# Two-tier LoD with 36 open regions: Llama multi-key 3

All rows use the same 32 frozen 32K prompts, Llama-3.1-8B-Instruct, BF16 and greedy first-response scoring. The PyTorch runs use the original optimized two-tier sparse prefill and cache updates, but evaluate the two-tier decode frontier in eager FP32/PyTorch. Each of its query heads opens up to the specified number of whole regions; RetroInfer selects 36 clusters shared by its four-query-head GQA group, so equal counts are not equal work. This is a quality experiment, not a latency benchmark.

| Method | First-response score | Change from PyTorch eight-region control, pp (paired 95% interval) |
|---|---:|---:|
| Dense | 100.00% | +15.62 [+3.12, +28.12] |
| Optimized two-tier, 8 regions | 78.12% | -6.25 [-15.62, +0.00] |
| RetroInfer, 36 shared clusters | 100.00% | +15.62 [+3.12, +28.12] |
| PyTorch two-tier control, 8 regions | 84.38% | +0.00 [+0.00, +0.00] |
| PyTorch two-tier, 36 regions | 100.00% | +15.62 [+3.12, +28.12] |

36 versus PyTorch 8: 5 improved, 0 worsened. Improved source rows: [126, 236, 302, 306, 447]; worsened source rows: [].

Optimized 8 versus PyTorch 8: 0 improved, 2 worsened. Improved source rows: []; worsened source rows: [78, 193].

8-region PyTorch first-step check against the independent readable reference: maximum absolute component error 0.015625; average layer-wise mean absolute error 0.000127.

36-region PyTorch first-step check against the independent readable reference: maximum absolute component error 0.031250; average layer-wise mean absolute error 0.000123.

Checks cover the first decode step on one example; later states were not checked against the reference. The PyTorch versions use FP32 accumulation and can choose different tokens from the BF16 optimized kernel, so the 8-region control is included. Dense and RetroInfer use dense prefill. Paired bootstrap intervals are over these 32 examples.

The [36-region outputs](ruler-two-tier-36-route-mk3.json) and [8-region
control outputs](ruler-two-tier-8-route-control-mk3.json) retain the per-example
answers and reference checks. The [runner](../../../benchmarks/retro_ruler_route_budget.py)
and [comparison script](../../../benchmarks/retro_ruler_route_budget_compare.py)
were run with `uv` using `cluster-run`. The recorded runs are
[36-region](1794112808868063220-ruler-two-tier-36-route-mk3-run.json) and
[8-region control](1794112815065420853-ruler-two-tier-8-route-control-mk3-run.json).
