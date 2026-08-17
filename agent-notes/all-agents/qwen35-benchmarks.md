# Qwen35 Benchmarks

Topic hints: Separate prefill-heavy LongBench timing from decode speed

## Lessons

- Qwen3.5 speed benchmarks must assert the resolved per-layer FLA delta-rule, fused gated RMSNorm, and causal-conv prefill/decode callables. Transformers' combined fast-path warning is not component-specific, and this repo's FLA compatibility patch can silently no-op when the optional extra is absent; never infer the active path from package declarations or the warning alone.

- For LOD decode benchmarks with 256-token state updates, generate at least 1,025 tokens and time the 1,024 decode intervals so four state updates are amortized; a 300-token run contains only one update and is not representative.

- LongBench v2 runs capped at 32 output tokens are dominated by prefill and cannot resolve modest decode-kernel gains; use a matched 1,025-token decode sweep for decode speed, and use LongBench for end-to-end prefill-heavy timing and quality.
