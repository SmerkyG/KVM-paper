# Kimi K3 MLA-stack performance fixture

This dummy-weight fixture contains 24 consecutive MLA layers with the full
Kimi K3 attention geometry: hidden width 7,168, 96 query heads, a 1,536-wide
query low-rank projection, and a 512+64 latent/direct-key cache record.  It is
intended to reproduce the cross-layer scheduling, 12-layer attention-residual
grouping, cache construction, collectives, and GPU occupancy of K3's 24 MLA
layers without loading or executing its 69 KDA layers or any FFN/MoE
sublayers.

`lod_attention_only_fixture` is a benchmark-only opt-in handled by the vLLM
LoD Kimi adapter.  It bypasses each decoder layer's post-attention FFN branch;
the small `intermediate_size` merely lets the upstream model class construct a
structurally valid layer before that branch is discarded.  The vocabulary is
also reduced because this fixture is for attention timing, not model quality.

Use `load_format="dummy"` (as `benchmarks.kimi_k3_prefill_sweep` does).  This
is not a checkpoint and its logits have no quality meaning.
