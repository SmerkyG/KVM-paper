# Two-tier INT8 primary path

Batch-8 Qwen3.5-0.8B results on gfx942. The implementation remains strictly
two-tier: it does not add page-summary routing or increase persistent state.

## Design

- Store exact leaves as per-token-scaled INT8 K and V.
- Quantize each decode query once per layer, then reuse it across every routed
  posting-list split.
- Keep K packed and use `sdot4` for QK; do not expand K to BF16.
- Fold each token's V scale into its softmax probability, quantize the 16-value
  probability fragment in parallel across the wave, and use `sdot4` for PV.
- Share each selected posting-list page across the four GQA query heads.
- At eight route splits, cache four packed probability words. Repacking is
  faster at 16/32 splits because the cached words reduce occupancy.
- Reuse existing local-attention scratch for query codes/scales. Persistent LOD
  state size is unchanged.

An experiment that fused query quantization into the Triton route/top-k reducer
was rejected: route-reduce time rose from about 0.26 to 0.90 ms per step.

## HF phase benchmark

| Context | Format | Prefill ms | Decode ms/step | Leaf phase ms/step | Cache GiB |
|---:|:---|---:|---:|---:|---:|
| 32K | BF16 | 2383 | 16.036 | 0.766 | 4.032 |
| 32K | INT8 | 2017 | 14.994 | 0.762 | 2.533 |
| 64K | BF16 | 5763 | 15.652 | 0.857 | 7.617 |
| 64K | INT8 | 4622 | 15.653 | 0.927 | 4.628 |
| 128K | BF16 | 14993 | 15.666 | 3.777 | 14.608 |
| 128K | INT8 | 11663 | 14.983 | 3.844 | 8.643 |

The INT8 leaf scan is tied at 32K and 1.8--8.1% slower at 64--128K, but total HF
decode is tied or faster. Prefill improves by 18--29%, and persistent cache use
falls by 37--41%.

## vLLM benchmark

Median batch-step time over 256 decoded steps; prefill uses the normal vLLM
chunked path.

| Context | Format | Prefill s | Decode ms/batch-step | LOD cache GiB |
|---:|:---|---:|---:|---:|
| 32K | BF16 | 1.700 | 4.531 | 4.458 |
| 32K | INT8 | 1.671 | 4.538 | 2.935 |
| 64K | BF16 | 3.866 | 5.166 | 8.042 |
| 64K | INT8 | 3.799 | 5.277 | 5.031 |
| 128K | BF16 | 9.244 | 6.197 | 15.033 |
| 128K | INT8 | 9.045 | 6.301 | 9.045 |

INT8 vLLM decode is within 0.2% at 32K and 1.7--2.2% at 64--128K. That small,
roughly fixed delta is query/probability quantization overhead rather than a
context-growing dequantization scan, so adding a third tier is not justified by
these measurements.

## Correctness

- Targeted HIP-versus-Triton INT8 decode verifier: max absolute difference 0.0.
- NIAH-S3 at 32K: 8/8 exact.
- The matched 128K BF16 and INT8 runs produced identical top-1 decode traces.

Primary artifacts are in `baseline/`, `matched_bf16_decode/`,
`prequant_parpv_decode/`, `matched_128k/`, `vllm_current/`, and
`vllm_prequant_parpv/`.
