# Kvm

Topic hints: Correct the KVM split-attention recommendation while preserving the partial-R...

## Lessons

- Standard LayerNorm after zeroing partial-RoPE state-key coordinates repopulates the masked channels through mean-centering and affine bias. Every state-key normalization used in preparation, routing, or readout must preserve or reapply the mask; bias-free RMSNorm is a compatible alternative, while split NoPE-state/RoPE-window attention avoids the issue in a full-dimensional design.

- Plain split attention that reuses the same Q/K representations in rotated and unrotated branches creates a representation conflict, so it is not a credible higher-expressivity alternative by itself. The relevant parameterized alternative is a separate low-rank/LoRA-style adapter from the RoPE representations to NoPE Q/K (optionally V), which has worked well in distilled models. Preserve the existing guidance that support-preserving or post-normalization-remasked normalization fixes partial-RoPE zeroing.
