# Lod Attention

Topic hints: Keep HF adapters separate from generic LOD engines and kernels

## Lessons

- Keep Hugging Face model adapters thin: they own projection, RoPE, cache bookkeeping, gating, and output projection only. LOD engines should consume post-QKV/post-RoPE tensors through a model-independent interface, while Triton implementations remain in generic runtime and kernel modules.
