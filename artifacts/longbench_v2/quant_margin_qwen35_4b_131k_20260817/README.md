# Qwen3.5-4B INT8/INT8 versus INT8/INT4 sentinel

This reruns the fixed-batch LongBench-v2 choice-margin sentinel on a larger
Qwen3.5 model after the 0.8B experiment produced a suspiciously favorable
K=INT8/V=INT4 result.

## Configuration

- Model: `Qwen/Qwen3.5-4B`, BF16 weights
- Batch: the same eight prompts in one ordered completions request
- Context cap: 131,072 input tokens for every format
- Four independent passes per format, two passes per server
- Formats: BF16/BF16, simulated INT8/INT8, INT8/INT4, and INT4/INT8
- Completed semantic pages use page-mean residual QDQ with 32-channel groups
- Quantized page summaries use the existing groupwise INT8 summary format

The 131K cap is required because the quality-only simulation retains BF16
backing buffers; this experiment measures numerical quality, not format memory
or speed.

## Four-pass averaged result

| K / V | Correct | Agreement with BF16 | Mean JS vs BF16 | RMS margin drift | Mean margin drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 / BF16 | 3/8 | 100% | 0 | 0 | 0 |
| INT8 / INT8 | **3/8** | **100%** | 0.004261 | **0.234** | **0.045** |
| INT8 / INT4 | 2/8 | 87.5% | **0.001895** | 0.397 | 0.213 |
| INT4 / INT8 | 3/8 | 100% | 0.005516 | 0.515 | 0.153 |

INT8/INT8 is the quality winner. It preserves every averaged BF16 prediction
and correct answer, and its margin distortion is substantially smaller.
INT8/INT4 flips one answer and loses one correct example.

INT4/INT8 also preserves the averaged BF16 predictions and score, so if a
six-bit mixed format is mandatory it is preferable to INT8/INT4 on the
discrete outcomes. It is not competitive with INT8/INT8 on margin fidelity or
stability: its RMS margin drift is 0.515, and pairwise repeat agreement is only
91.7% versus 100% for INT8/INT8.

The lower JS value for INT8/INT4 is not trustworthy here. BF16's mean
pairwise repeat JS is 0.005940, larger than both format-to-BF16 averages and
almost three times the averaged INT8/INT8-to-INT8/INT4 JS (0.002025). The
eight-example JS statistic is therefore being dominated by scheduler/kernel
variation and nonlinear cancellation. It is not precision-monotonic.

Repeat stability supports the same conclusion: INT8/INT8 has 100% pairwise
prediction agreement across its four passes, while INT8/INT4 has 91.7%.

## Conclusion

The larger-model result rejects the apparent 0.8B K8/V4 advantage. There is no
evidence here for reducing V to INT4 while retaining K at INT8. Between the
two asymmetric six-bit formats, preserving V at INT8 gives better discrete
behavior, but neither matches INT8/INT8's margin fidelity. For format
selection, use prediction preservation and margin fidelity, and treat mean JS
as a diagnostic only when it exceeds the measured BF16 repeat-noise floor.

The combined reports are `k8_v8_vs_bf16_4repeat.json`,
`k8_v4_vs_bf16_4repeat.json`, and `k8_v4_vs_k8_v8_4repeat.json`.
The INT4/INT8 reports are `k4_v8_vs_bf16_4repeat.json` and
`k4_v8_vs_k8_v8_4repeat.json`.
