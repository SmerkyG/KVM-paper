# Kvm

Topic hints: Partial-RoPE state-key normalization must preserve the NoPE subspace.

## Lessons

- Standard LayerNorm after zeroing partial-RoPE state-key coordinates repopulates the masked channels through mean-centering and affine bias. Every state-key normalization used in preparation, routing, or readout must preserve or reapply the mask; bias-free RMSNorm is a compatible alternative, while split NoPE-state/RoPE-window attention avoids the issue in a full-dimensional design.
