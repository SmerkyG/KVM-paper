## Lessons

- Standard LayerNorm after zeroing partial-RoPE state-key coordinates repopulates the masked channels through mean-centering and affine bias. Every state-key normalization used in preparation, routing, or readout must preserve or reapply the mask; bias-free masked RMSNorm or post-normalization remasking fixes the invariant.
- Do not treat split RoPE-window/NoPE-state attention that reuses the same Q/K representations as a credible higher-expressivity alternative: the shared representation must serve conflicting positional and content interactions. Use separate low-rank/LoRA-style adapters from the RoPE representations to NoPE Q/K (optionally V); this parameterized branch has worked well in distilled models.
