"""Positional encoding variants of the original KVM attention layer."""

import torch
import torch.nn.functional as F

from .kvm_mixer import SequenceMixer, _lower_right_causal_mask


class RoPEEverywhere(SequenceMixer):
    def _prepare_state_update_k(self, k_block):
        return self.ln_s_k(k_block)


class LocalOnlyRoPE(SequenceMixer):
    def _position_qk(self, q, k, position_embeddings):
        local_q, local_k = super()._position_qk(q, k, position_embeddings)
        rotary = self.rope_partial_dim
        # Share the non-rotary channels and carry the original rotary channels
        # separately, so each region can use its own dot product in one SDPA.
        return (
            torch.cat([local_q, q[..., :rotary]], dim=-1),
            torch.cat([local_k, k[..., :rotary]], dim=-1),
        )

    def _prepare_state_update_k(self, k_block):
        rotary, head_dim = self.rope_partial_dim, self.d_qk_head
        unrotated_key = torch.cat([k_block[..., head_dim:], k_block[..., rotary:head_dim]], dim=-1)
        return self.ln_s_k(unrotated_key)

    def _sdpa_with_repeated_kv(self, q, k, v, **kwargs):
        head_dim = self.d_qk_head
        return super()._sdpa_with_repeated_kv(q[..., :head_dim], k[..., :head_dim], v, **kwargs)

    def _join_attention_keys(self, state_k, local_k):
        rotary, head_dim = self.rope_partial_dim, self.d_qk_head
        state_rotary, state_nonrotary = state_k.split([rotary, head_dim - rotary], dim=-1)
        state_k = torch.cat([torch.zeros_like(state_rotary), state_nonrotary, state_rotary], dim=-1)
        local_k = torch.cat([local_k[..., :head_dim], torch.zeros_like(local_k[..., head_dim:])], dim=-1)
        return super()._join_attention_keys(state_k, local_k)

    def _state_attention(self, q, k_star, v_star):
        causal_mask = _lower_right_causal_mask(q.size(2), k_star.size(2), device=q.device)
        return F.scaled_dot_product_attention(
            q, k_star, v_star, attn_mask=causal_mask, is_causal=False,
            scale=self.d_qk_head ** -0.5,
        )
