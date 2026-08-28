# LongBench-v2 quantization margin sentinel

This is an eight-example, fixed-batch regression sentinel for LOD KV
quantization. It is not an unbiased estimate of LongBench-v2 accuracy.

The panel contains 1,005,883 prompt tokens, or 1.62% of the 62,210,414 tokens
in the 503-example run. Each example appends `The correct answer is (` to the
normal no-thinking chat prompt, constrains the next token to A/B/C/D, and
records all four log probabilities. All eight prompts are sent in one ordered
completions request so their batch composition and scheduler insertion order
are fixed.

## Qwen3.5-0.8B calibration

BF16 LOD and INT4 LOD were each run twice on the same server. A repeat takes
about 20--25 seconds after server startup.

| Measurement | Mean JS divergence | RMS correct-margin drift |
| --- | ---: | ---: |
| BF16 vs INT4, after averaging repeats | 0.005250 | 0.5144 |
| BF16 repeat noise | 0.000132 | 0.0442 |
| INT4 repeat noise | 0.004069 | 0.2615 |

The cross-format signal is about 40x the BF16 JS noise floor and 12x its RMS
margin noise floor. INT4's high self-noise is itself useful evidence of damage
that a replacement format should reduce.

The primary pass criteria for a new format are lower mean-distribution JS,
lower RMS correct-margin drift, and lower repeat self-noise than INT4. Top-one
accuracy and answer flips are secondary because near-boundary LongBench answers
are much noisier.

Raw outputs and the combined report are in `lod_bf16_{a,b}.jsonl`,
`lod_int4_{a,b}.jsonl`, and `repeated_comparison.json` in this directory.
