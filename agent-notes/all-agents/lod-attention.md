# Lod Attention

Topic hints: Preserve compact storage and efficient state widths for side caches

## Lessons

- Keep Hugging Face model adapters thin: they own projection, RoPE, cache bookkeeping, gating, and output projection only. LOD engines should consume post-QKV/post-RoPE tensors through a model-independent interface, while Triton implementations remain in generic runtime and kernel modules.

- Hugging Face calls Cache.update before AttentionInterface. A model-independent LOD cache must therefore stage the new post-RoPE K/V block, bind that cache layer to the attention module, and let the registered backend consume it and perform causal compression; finalizing the whole prefill inside Cache.update would expose future state to earlier queries.

- When separating a tiny persistent K/V side cache, materialize its slices with `.contiguous()` so views do not pin full prefill allocations, and keep the mergeable state at its scheduled efficient width rather than subtracting side-cache entries and creating irregular route-GEMM dimensions.
