# Qwen35 Benchmarks

Topic hints: Amortize multiple LOD update boundaries in decode benchmarks

## Lessons

- Qwen3.5 speed benchmarks must assert the resolved per-layer FLA delta-rule, fused gated RMSNorm, and causal-conv prefill/decode callables. Transformers' combined fast-path warning is not component-specific, and this repo's FLA compatibility patch can silently no-op when the optional extra is absent; never infer the active path from package declarations or the warning alone.

- For LOD decode benchmarks with 256-token state updates, generate at least 1,025 tokens and time the 1,024 decode intervals so four state updates are amortized; a 300-token run contains only one update and is not representative.
