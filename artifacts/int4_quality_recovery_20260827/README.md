# Qwen INT4 quality recovery

Date: 2026-08-27

## Outcome

The recommended recursive INT4 format now uses four-channel groups for leaf
residuals and INT8 page summaries, plus a least-squares-refined leaf scale for
both prefill conversion and subsequent page appends. The packed K/V codes and
page-wide scale broadcast are unchanged.
This is now the automatic vLLM default when `VLLM_LOD_KV_BITS=4`.

The previous quality default shared one scale across 16 tokens by 16 channels,
or 256 residual values. The new group covers 64 values, then refines the scale
against the selected integer codes twice. It retains the one-dimensional
broadcast scale load during attention; no token-axis scale matrix is added.

## Qwen3.5-0.8B results

All LOD rows use recursive three-tier direct prefill, automatic routing, batch
eight, and native packed INT4 K/V. ProLong consists of eight real 8K examples.
The choice panel contains 48 fixed LongBench-v2 examples spanning 13K to 262K
input tokens and compares first-token A-D distributions against the same BF16
LOD reference.

The LOD prefill/update path has measurable nondeterminism: four BF16 repeats
agree with each other on only 90.28% of pairwise choices. Consequently the
scale decision uses mean A/B/C/D distributions across repeats rather than a
lucky pairing of two individual runs.

| Storage | Runs | Choice score | Agreement with four-run BF16 mean | Mean JS | RMS margin drift |
|---|---:|---:|---:|---:|---:|
| BF16 LOD | 4 | 9/48 | reference | reference | reference |
| former INT4 default, G16 L2 | 1 | 8/48 | 42/48 (87.50%) | 0.004041 | 0.3833 |
| INT4, G4 L2 | 1 | 9/48 | 43/48 (89.58%) | **0.001910** | **0.2626** |
| **INT4, G4 L2** | **3** | **9/48** | **45/48 (93.75%)** | 0.001935 | 0.2720 |

Thus the repeat-audited G4 result exceeds the former 44/48 (91.67%) target
without relying on one favorable run. All three INT4 runs also preserve the
BF16-mean score. The eight-by-8K ProLong CE is 1.925282, a 0.000148 (0.0077%)
increase from BF16's 1.925134.
The compact repeat comparison is
`qwen08_panel48_int4_g4_l2_repeat_audit.json`.

Scale metadata increases each leaf element from 4.0625 to 4.25 effective
bits, 4.62% more than the former G16 format and still 73.44% below a 16-bit
leaf payload. This calculation excludes shared BF16 routing state and page
metadata, which are unchanged.

The selected G4 L2 format also scored 64/64 on Qwen3.5-0.8B NIAH-S3 at 8K,
batch eight. The run exercised direct recursive LOD prefill and decode.

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
VLLM_LOD_QUANT_GROUP_SIZE=4
VLLM_LOD_LEAF_QUANT_SCALE_MODE=l2
VLLM_LOD_LEAF_APPEND_QUANT_SCALE_MODE=l2
VLLM_LOD_QUANT_TOKEN_GROUP_SIZE=16
```

For the former quality default, set group size 16 with both scale modes at
`l2`. For exact legacy reproduction, set group size 32 and both modes to `max`.
