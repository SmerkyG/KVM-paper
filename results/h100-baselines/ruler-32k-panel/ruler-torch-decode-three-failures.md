# PyTorch reference decode on three Llama RULER failures

Llama-3.1-8B-Instruct, BF16, approximately 32K context, `niah_multikey_3`,
three-tier top-2. These are the first three examples in the frozen cohort that
the original three-tier run scored incorrect and dense scored correct. Answers
below are the first assistant response, stopping at the end-of-turn token.

| Source row | Expected UUID | Optimized LoD | PyTorch reference decode |
|---:|---|---|---|
| 46 | `86a867a7-6a98-4a02-b065-70a33bafafde` | `86a86703-6a98-4a02-b065-70a33bafafde` | `86a8a7a7-6a98-4a02-b065-70a33bafafde` |
| 59 | `350ec93a-1edc-4e1e-a5b1-c8f78fb4ffc3` | `350ec93a-1edc-4e01-a5b1-c8f78fb4ffc3` | `350ec93a-1edc-4e1e-a5b1-c8f78fb4ff03` |
| 78 | `0cc6a78c-2bf9-4240-bc01-857a0a9457cd` | `0c6a78c-2bf9-4240-bc01-857a0a9457cd` | `0c6a78c-2bf9-4240-bc01-857a0a9457cd` |

All three reference answers are incorrect. The entire first-response token
sequence is identical to the optimized result for row 78; the other two
produce different incorrect answers. This is a deliberately selected set of
failures, not an estimate of overall RULER accuracy.

## Method and limits

The experiment reruns optimized LoD sparse prefill and verifies the prompt
input-token hash and first generated token against the original LoD run. At
each decode layer it reconstructs a `PytorchLODCache` from the optimized
pool's actual BF16 region sums, counts, pages, leaves, sink and local KV; the
existing `PytorchLODAttention._lod_attention` computes the model output using
ordinary PyTorch operations. The optimized decode kernel also runs to append
the current local KV and provide a per-layer comparison, but its attention
output is discarded. The cache update/region construction still use the
optimized implementation, so this is an independent reference **attention
calculation**, not an independent implementation of the full cache pipeline.
Checks verify that all leaf entries appear exactly once, counts agree, and the
reconstructed chronological page order matches the stored pages.

On the first decode step, comparing the two attention outputs for the same
layer inputs, the maximum absolute difference over all layers is 0.015625 in
each example; the average of the 32 layer-wise mean absolute differences is
0.000138, 0.000134, and 0.000133 for rows 46, 59, and 78 respectively. These
are first-step checks only. Different rounding or routing/page decisions can
accumulate as generation proceeds; this experiment does not attribute the
differences or rule out a bug in cache construction or later decode steps.

The [individual results](ruler-torch-decode-three-failures.json) include token
IDs and per-layer differences. The benchmark is
[`benchmarks/retro_ruler_torch_decode_probe.py`](../../../benchmarks/retro_ruler_torch_decode_probe.py).
The H100 `cluster-run` was
`1794109976447863701-ruler-torch-decode-three-failures`; its
[run metadata](1794109976447863701-ruler-torch-decode-three-failures-run.json)
records the command and source snapshot.
