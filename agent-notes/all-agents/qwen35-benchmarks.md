# Qwen35 Benchmarks

Topic hints: Assert FLA and causal-conv hooks in Qwen3.5 speed runs

## Lessons

- Qwen3.5 speed benchmarks must assert the resolved per-layer FLA delta-rule, fused gated RMSNorm, and causal-conv prefill/decode callables. Transformers' combined fast-path warning is not component-specific, and this repo's FLA compatibility patch can silently no-op when the optional extra is absent; never infer the active path from package declarations or the warning alone.
