# Lod Attention

Topic hints: Stage custom HF cache updates until the attention backend consumes QKV

## Lessons

- Keep Hugging Face model adapters thin: they own projection, RoPE, cache bookkeeping, gating, and output projection only. LOD engines should consume post-QKV/post-RoPE tensors through a model-independent interface, while Triton implementations remain in generic runtime and kernel modules.

- Hugging Face calls Cache.update before AttentionInterface. A model-independent LOD cache must therefore stage the new post-RoPE K/V block, bind that cache layer to the attention module, and let the registered backend consume it and perform causal compression; finalizing the whole prefill inside Cache.update would expose future state to earlier queries.
