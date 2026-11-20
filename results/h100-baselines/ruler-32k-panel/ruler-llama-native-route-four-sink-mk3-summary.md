# Llama 32K multi-key 3: native routing and four sinks

The same 32 frozen prompts, Llama-3.1-8B-Instruct and BF16. The candidate combines native-query region routing and four exact sink tokens in both sparse prefill and three-tier top-2 decode. The original three-tier run uses normalized-query routing and one sink. This changes both settings at once and is not an isolated attribution or a speed measurement.

| Method | First-response score | Difference from original three-tier, pp (paired 95% interval) |
|---|---:|---:|
| Dense | 100.00% | +50.00 [+34.38, +68.75] |
| Original three-tier LoD | 50.00% | +0.00 [+0.00, +0.00] |
| Native-query routing + four sinks | 25.00% | -25.00 [-43.75, -9.38] |
| Two-tier LoD | 78.12% | +28.12 [+12.50, +43.75] |
| RetroInfer | 100.00% | +50.00 [+34.38, +68.75] |

Paired first responses: 1 improved, 9 worsened, 22 unchanged. Improved source rows: [164]. Worsened source rows: [4, 68, 127, 156, 187, 246, 384, 390, 462]. The first token after prefill changed on 0/32 examples.

First-response scoring stops at end-of-turn or end-of-text. Bootstrap intervals use paired examples. Dense and RetroInfer use dense prefill; the LoD runs use sparse prefill. The source rows and individual full fixed-length outputs are retained in the JSON.
The first graph/eager decode check matched exactly. At every prefill layer and
on every example, the diagnostic checked that all four stored sink K/V entries
match the original first four entries, and that centroid counts, remote leaves,
local entries and sinks cover the prompt. On one failed example, an independent
PyTorch attention calculation using the same modified cache had maximum
first-step per-layer absolute error 0.015625 versus the optimized kernel, with
mean absolute error averaged over layers 0.000135. These checks do not prove
all steps or cache construction correct.

Results: [individual outputs](ruler-llama-native-route-four-sink-mk3.json),
[graph/eager check](ruler-llama-native-route-four-sink-check.json),
[PyTorch check](ruler-llama-native-route-four-sink-torch-check.json), and
[run metadata](1794111167606448382-ruler-llama-native-route-four-sink-mk3-run.json).
The main run was `1794111167606448382-ruler-llama-native-route-four-sink-mk3`;
the summary run was `1794111217735719643-ruler-llama-native-route-four-sink-summary`.
