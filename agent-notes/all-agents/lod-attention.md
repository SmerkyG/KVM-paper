# Lod Attention

Topic hints: Separate LOD prefill allocation from decode scan bounds

## Lessons

- Keep Hugging Face model adapters thin: they own projection, RoPE, cache bookkeeping, gating, and output projection only. LOD engines should consume post-QKV/post-RoPE tensors through a model-independent interface, while Triton implementations remain in generic runtime and kernel modules.

- Hugging Face calls Cache.update before AttentionInterface. A model-independent LOD cache must therefore stage the new post-RoPE K/V block, bind that cache layer to the attention module, and let the registered backend consume it and perform causal compression; finalizing the whole prefill inside Cache.update would expose future state to earlier queries.

- When separating a tiny persistent K/V side cache, materialize its slices with `.contiguous()` so views do not pin full prefill allocations, and keep the mergeable state at its scheduled efficient width rather than subtracting side-cache entries and creating irregular route-GEMM dimensions.

- For the kernel LOD zero-route fallback, compute count-corrected state attention with the existing Triton LSE kernel, run the exact causal local field with aten._scaled_dot_product_flash_attention, and LSE-merge the branches; on Qwen3.5-0.8B this removed FlexAttention while improving 8K batch-8 prefill speed.

- Generic HF LOD installation must use decoder config.layer_types (and module sliding_window as a fallback) to replace only full/global attention; mixed full/SWA models need per-module backend dispatch plus a native sliding-cache/LOD-cache bridge.

- When adding physical prefill padding to generic HF LOD, propagate logical_prefill_len through both the engine wrapper and Triton core, and verify the committed snapshot rather than a dirty development worktree.

- Keep the large prefill lookback allocation separate from the logical LOD decode-local scan bound. Pass the configured decode local length to fused decode and assert state catch-up keeps each active local length within it; passing the backing/prefill capacity makes every layer loop over masked rows and can dominate decode latency.
