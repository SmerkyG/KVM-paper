# Qwen INT4 quality recovery

Date: 2026-08-27

## Outcome

The recommended recursive INT4 format now uses 16-channel groups for leaf
residuals and INT8 page summaries, plus a least-squares-refined leaf scale for
both prefill conversion and subsequent page appends. The packed K/V codes and
page-wide scale broadcast are unchanged.
This is now the automatic vLLM default when `VLLM_LOD_KV_BITS=4`.

The previous format used one max-absolute scale for 16 tokens by 32 channels,
or 512 residual values. One outlier could therefore consume too much of
INT4's seven positive reconstruction levels. The new group covers 256 values,
then refines the scale against the selected integer codes twice. It retains
the fast one-dimensional scale load during attention.

## Qwen3.5-0.8B results

All LOD rows use recursive three-tier direct prefill, automatic routing, batch
eight, and native packed INT4 K/V. ProLong consists of eight real 8K examples.
The choice panel contains 48 fixed LongBench-v2 examples spanning 13K to 262K
input tokens and compares first-token A-D distributions against the same BF16
LOD reference.

| Storage | ProLong CE | Choice score | Agreement with BF16 | RMS margin drift | Panel wall time |
|---|---:|---:|---:|---:|---:|
| BF16 LOD | 1.925134 | 11/48 | reference | reference | 66.33 s |
| legacy INT4, G32 max | 1.925332 | 10/48 | 89.58% | 0.3385 | 76.80 s |
| INT4, G32 L2 | 1.925203 | 11/48 | 89.58% | 0.3886 | 76.26 s |
| **INT4, G16 L2** | **1.925276** | **12/48** | **91.67%** | **0.3292** | **82.31 s** |

The selected format improves the discrete proxy from 10/48 to 12/48 and has
slightly less RMS margin drift than legacy INT4. Its CE gap to BF16 is
0.000143 (0.0074%), versus 0.000198 (0.0103%) for the legacy format. Panel wall
time is 7.2% above legacy INT4 in this run; the attention consumer retains the
same one-dimensional broadcast-scale shape rather than the much more expensive
token-by-channel scale matrix.

Scale metadata increases each leaf element from 4.03125 to 4.0625 effective
bits: only 0.78% more than legacy INT4, and still 74.61% below a 16-bit leaf
payload. This calculation excludes shared BF16 routing state and page
metadata, which are unchanged.

The selected format also scored 64/64 on Qwen3.5-0.8B NIAH-S3 at 8K, batch
eight. The run exercised direct recursive LOD prefill and decode and completed
in 12.09 seconds after startup.

## Rejected alternatives

- Token-axis groups substantially reduced isolated attention reconstruction
  error, but made long-prompt reads much slower because the scale became a
  token-by-channel matrix rather than a broadcast vector. Four-token/G32
  scored 12/48 but took 115.17 seconds. Token-wise/G128 matched BF16 ProLong CE
  (1.925139) but scored only 10/48 and took 124.71 seconds.
- Just-in-time zero-mean residual correction reduced isolated kernel MSE but
  did not improve ProLong or the choice sentinel, so it was removed rather
  than retained as another runtime knob.
- A broader clipping-ratio search lowered residual reconstruction MSE but was
  worse than the existing L2 refinement on ProLong, so it was also removed.

The 48-example panel is a rapid damage test, not a replacement for the full
503-example LongBench-v2 run. A full run is still required before claiming
that the historical 243/503 BF16 versus 236/503 INT4 gap is closed.

## Configuration

The quality default is selected automatically for recursive INT4:

```bash
VLLM_LOD_LEVELS=3 VLLM_LOD_KV_BITS=4
```

The equivalent explicit settings are:

```bash
VLLM_LOD_QUANT_GROUP_SIZE=16
VLLM_LOD_LEAF_QUANT_SCALE_MODE=l2
VLLM_LOD_LEAF_APPEND_QUANT_SCALE_MODE=l2
VLLM_LOD_QUANT_TOKEN_GROUP_SIZE=16
```

For exact legacy reproduction, set group size 32 and both leaf scale modes to
`max`.
