# Llama 32K multi-key 3: prefill and decode ablation

All four rows use the same 32 inputs and BF16 weights/KV. LoD is three-tier top-two. Dense-decode diagnostic stores and checks all exact keys and values produced by LoD prefill at every layer; it then uses the native dense decode graph. It is not a speed measurement. Matching prefill variants produced the same first generated token on all 32 examples.

## First response

| Prefill and decode | Score | Change from LoD prefill/decode, pp (paired 95% interval) |
|---|---:|---:|
| Dense prefill, dense decode | 100.00 | +50.00 [+31.25, +68.75] |
| LoD prefill, LoD decode | 50.00 | +0.00 [+0.00, +0.00] |
| Dense prefill, LoD decode | 56.25 | +6.25 [-9.38, +21.88] |
| LoD prefill, dense decode | 100.00 | +50.00 [+34.38, +65.62] |

## Full fixed-length output

| Prefill and decode | Score | Change from LoD prefill/decode, pp (paired 95% interval) |
|---|---:|---:|
| Dense prefill, dense decode | 100.00 | +46.88 [+28.12, +65.62] |
| LoD prefill, LoD decode | 53.12 | +0.00 [+0.00, +0.00] |
| Dense prefill, LoD decode | 65.62 | +12.50 [+0.00, +28.12] |
| LoD prefill, dense decode | 100.00 | +46.88 [+31.25, +62.50] |

First-response scoring stops at the first end-of-turn or end-of-text token. Full fixed-length scoring can count text after that token. Paired bootstrap intervals are based on 32 examples.
