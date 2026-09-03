# Qwen3.5-0.8B LOD K/V precision factorial

> **Superseded conclusion:** a four-repeat Qwen3.5-4B rerun showed that the
> apparent K8/V4 advantage was a noisy-metric artifact. K8/V8 preserved all
> BF16 predictions and had substantially better margin fidelity. See
> `../quant_margin_qwen35_4b_131k_20260817/README.md`.

This is a fast sensitivity experiment, not a LongBench-v2 accuracy estimate.
It uses the same eight-example, fixed-batch, two-repeat choice-margin sentinel
as the calibrated BF16/INT4 comparison. Mixed formats are simulated by
groupwise page-residual quantize/dequantize (QDQ) into BF16 storage; therefore
these runs measure numerical quality, not memory or speed.

Completed 16-token semantic pages use the existing 32-channel groups and
their exact page mean. Quantized K or V page summaries are also rounded through
the existing groupwise INT8 summary format. A zero-bit label below means that
side remains BF16.

## Results

All drift columns compare the average of two format runs with the average of
two BF16 runs. Repeat columns compare the two runs of that format.

| K | V | Mean JS vs BF16 | RMS margin drift | Repeat JS | Repeat RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | BF16 | 0.000000 | 0.000 | 0.000132 | 0.044 |
| INT8 | INT4 | **0.002803** | **0.294** | **0.000431** | **0.088** |
| INT4 | BF16 | 0.002142 | 0.331 | 0.005025 | 0.380 |
| INT4 | INT8 | 0.003977 | 0.395 | 0.017085 | 0.809 |
| INT8 | BF16 | 0.005077 | 0.425 | 0.001559 | 0.272 |
| BF16 | INT8 | 0.005532 | 0.491 | 0.001953 | 0.290 |
| INT4 | INT4 (QDQ) | 0.006566 | 0.537 | 0.000896 | 0.207 |
| INT8 | INT8 | 0.006747 | 0.555 | 0.002121 | 0.283 |
| BF16 | INT4 | 0.008790 | 0.565 | 0.000859 | 0.225 |
| INT4 | INT4 (actual packed) | 0.005250 | 0.514 | 0.004069 | 0.261 |

Raw probability records are `k{K}_v{V}_{a,b}.jsonl`. `factorial.json`
contains the BF16-relative and conditional pairwise comparisons.

## Interpretation

The isolated 4-bit perturbation says V is more precision-sensitive: keeping K
BF16 and quantizing only V produces 4.1x the mean JS divergence of quantizing
only K (0.008790 versus 0.002142), and 1.7x the RMS margin drift (0.565 versus
0.331). The same direction is present but much weaker at INT8.

The mixed interaction is not additive. K=INT8/V=INT4 is the best and most
stable non-BF16 cell, while the structurally tempting K=INT4/V=INT8 cell is
very noisy. Thus this panel supports **testing an actual packed K8/V4 path
first**, but it does not establish a general rule that values need fewer bits.
The isolated result actually favors preserving V precision.

There are two important limits:

1. This model/sentinel is non-monotonic at the final logits; INT8/INT8 did not
   land closer to BF16 than INT4/INT4. Small cache perturbations can change
   later sparse-routing and model trajectories, so final-logit distance is not
   a direct reconstruction-error measurement.
2. Simulated and actual INT4/INT4 differ by mean JS 0.005278. QDQ is useful for
   screening formats, but a mixed winner must be verified in the real packed
   attention path before making a storage decision.

For Qwen3.5's equal K/V dimensions, K8/V4 would use six leaf bits per scalar on
average: 50% more leaf payload than K4/V4, but 62.5% less than BF16 K/V. Coarse
state and metadata memory are unchanged.
