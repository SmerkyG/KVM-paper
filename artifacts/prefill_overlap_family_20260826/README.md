# Two-level prefill branch-overlap family panel

This panel tests whether the Muse-Glimmer prefill optimization generalizes to
the other 30B-class model geometries. After current-query route selection, the
stable coarse, exact-leaf, and exact-local branches are independent. The
experimental setting queues them on separate GPU streams and synchronizes at
the final LSE merge.

## Method

- Batch size 8 at 64K context
- Eight distinct real ProLong documents; no repeated synthetic text
- 16K aggregate scheduler chunks and 4K LOD state-update chunks
- Ordinary query-dependent two-level top-three routing, not static-cohort LOD
- BF16 LOD K/V; Qwen uses its published FP8 model weights
- One warmup followed by three timed prefills
- Identical model, GPU allocation, prompts, and production vLLM dispatch for
  each overlap-off/on pair

## Results

| Model | Geometry | overlap off | overlap on | Change |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-FP8 | D=256, GQA=6 | 80.123 s | 79.573 s | **0.69% faster** |
| OLMo-3-1125-32B | D=128, GQA=5 | 73.059 s | 72.430 s | **0.86% faster** |
| Phi-4 TP5 | D=128, GQA=4 | 42.941 s | 47.215 s | **9.95% slower** |
| Gemma-4-26B-A4B, 0.7 memory utilization | D=512, GQA=8 | 18.836 s | 28.380 s | **50.67% slower** |

Qwen's individual off timings were 81.914, 80.123, and 80.053 seconds;
overlap-on timings were 81.414, 79.545, and 79.573 seconds. OLMo's off
timings were 74.267, 73.047, and 73.059 seconds; overlap-on timings were
72.430, 72.902, and 72.180 seconds. Thus all three OLMo overlap measurements
beat all three controls, while Qwen's like-for-like improvement is smaller.

Phi's overlap timings were 47.215, 50.781, and 46.465 seconds versus 42.941,
47.271, and 42.906 seconds without overlap. Gemma is an unambiguous rejection:
at the normal 0.8 memory utilization the concurrent branches exhausted device
headroom, while a matched 0.7-memory pair avoided the fault but produced
79.016, 28.380, and 21.480 seconds with overlap versus 18.836, 19.200, and
18.835 seconds without it. Even Gemma's best overlap timing regressed.

## Dispatch audit and conclusion

The saved production dispatch records verify both overlap flags in every `on`
run and neither in every `off` run. Qwen and OLMo used the fused
`_route_logits_topk_coarse_attention_kernel` route/coarse path. Phi used the
group-candidate top-k series and stable Triton coarse attention. Gemma used the
group-candidate selector followed by the wide-head torch matmul/softmax/matmul
coarse path.

The optimization therefore is not a universal prefill default. Muse's D=128,
GQA=16 geometry has enough independent underoccupied work to hide meaningful
latency. Qwen and OLMo show small opt-in gains, but Phi loses to resource
contention and Gemma's wide concurrent branches add both severe contention and
peak-memory pressure. Automatic enablement remains restricted to Muse's
D=128/GQA=16 geometry.

Raw results are the `*_overlap_{off,on}_64k_b8_r3.json` files in this directory.
