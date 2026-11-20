# H100 two-tier threshold: fused page offsets

This experiment uses Llama-3.1-8B-Instruct, batch one, BF16 weights and KV,
and the experimental two-tier RetroInfer adapter. Normalized-query routing
opens every eligible region whose estimated mass exceeds 16 times the mean
over nonempty remote regions. Regions larger than 1,024 entries remain
summarized. Local and sink attention are exact. The selector does not sort or
impose a fixed number of selected regions.

The optimization computes per-tile page offsets while selecting and packing
regions, eliminating a separate global page-offset scan. The full-model
timings below use the same teacher-forced ProLong prompt and whole-model CUDA
graph on an H100 80 GB. Each run includes 512 untimed decode steps and 1,024
timed steps, including cache catch-up; prefill and graph capture are excluded.
The numbers are medians of two timed repetitions after one warmup.

| 32K method | Decode ms/token | Timed repetitions, ms/token | Prefill s |
|---|---:|---|---:|
| Fixed eight | 8.332 | 8.3278, 8.3354 | 1.653 |
| Threshold c=16, prior | 7.837 | 7.8361, 7.8369 | 1.665 |
| Threshold c=16, fused offsets | 7.745 | 7.7345, 7.7551 | 1.654 |

The fused threshold path is 1.17% faster than the previous threshold path and
7.04% faster than fixed eight on this prompt. The fixed-eight and threshold
methods open different regions and are not quality matched. This is a single
prompt, batch, context length and GPU; it is not evidence of the same gain in
other settings. On an isolated real 32K layer, the same threshold geometry
changed from 48.65 to 45.66 microseconds per call; this layer-only test omits
model and cache-maintenance work.

Synthetic tests checked empty selection, more than eight selected regions,
multiple selection tiles, and regions at and above the 1,024-entry limit. On
one real 32K query, selected regions matched the independent reference across
all 32 layers, with maximum attention-component error 0.00390625. On 32
multi-key-3 RULER examples, generated tokens matched both the previous H100
threshold implementation and the independent FP32 reference exactly.
The [full 13-task RULER comparison](ruler-32k-llama-two-tier-threshold-c16-fused-full13-summary.md)
contains 32 examples per task and records first-response and fixed-length
scores separately. First-response scores average 87.04% for two-tier threshold,
80.58% for the prior three-tier result, and 88.88% for dense. Graph/eager
checks passed on the first example of each task; the full run's 32 multi-key-3
first responses matched the separately checked threshold output tokens.
The [paired fixed-eight comparison](ruler-llama-threshold-vs-fixed8-selected.md)
covers eight selected tasks; for multi-key 3 the threshold scored 100% versus
78.12% for fixed eight on the same 32 examples.

This remains an opt-in experiment, enabled by
`LOD_H100_THRESHOLD_MULTIPLIER=16`; it is not a serving default and currently
supports only two-tier BF16 batch-one Llama geometry.

[Timing data](h100-threshold-fused-c16-32k.json),
[layer tuning](h100-threshold-fused-layer-tuning.json),
[synthetic checks](h100-threshold-fused-synthetic.json),
[real-query checks](h100-threshold-fused-real.json),
[32-example outputs](h100-threshold-fused-ruler-quality.json), and
[output equivalence](h100-threshold-fused-equivalence.json).
The [full-panel outputs](ruler-32k-llama-two-tier-threshold-c16-fused-full13.json)
and [cluster-run record](1794144511786104224-llama-32k-ruler-full13-threshold-c16-fused-run.json)
retain the complete sample outputs and source snapshot hashes.
