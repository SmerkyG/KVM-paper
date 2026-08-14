"""Model-specific latent-cache adapters for Hugging Face MLA decoders.

The generic Hugging Face LOD backend starts after a model has expanded its
keys and values.  That is the wrong boundary for Multi-head Latent Attention:
expanding first duplicates the compressed cache once per attention head.

This module keeps DeepSeek-V2 and GLM-4.7-Flash at their native cache boundary.
It absorbs the per-head key up-projection into the query, attends to the shared
compressed KV latent, and applies the per-head value up-projection to the latent
attention output.  The transformation is algebraically identical to the
model's eager attention because both up-projections are linear.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import nn


def _native_transient_forward(
    module: nn.Module, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the checkpoint's original prefill before LOD is needed.

    Matrix absorption is algebraically exact but changes BF16 rounding order.
    Keeping the original expanded implementation inside the exact local field
    avoids accumulating that drift while preserving the latent LOD cache for
    longer contexts.
    """
    implementation = module.config._attn_implementation
    module.config._attn_implementation = module._hf_lod_native_attn_implementation
    try:
        return module._hf_lod_original_forward(*args, **kwargs)
    finally:
        module.config._attn_implementation = implementation


def _use_native_transient_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    past_key_values: Any | None,
) -> bool:
    settings = module._hf_lod_settings
    return (
        past_key_values is None
        and int(hidden_states.size(1)) <= int(settings.config.local_window)
    )


def is_glm4_moe_lite_mla(module: nn.Module) -> bool:
    """Return whether ``module`` exposes GLM-4.7-Flash's MLA geometry."""
    return (
        type(module).__name__ == "Glm4MoeLiteAttention"
        and type(module).__module__.startswith(
            "transformers.models.glm4_moe_lite."
        )
        and isinstance(getattr(module, "kv_b_proj", None), nn.Linear)
        and isinstance(getattr(module, "kv_a_layernorm", None), nn.Module)
    )


def is_deepseek_v2_mla(module: nn.Module) -> bool:
    """Return whether ``module`` exposes DeepSeek-V2's MLA geometry."""
    return (
        type(module).__name__ == "DeepseekV2Attention"
        and type(module).__module__.startswith(
            "transformers.models.deepseek_v2."
        )
        and isinstance(getattr(module, "kv_b_proj", None), nn.Linear)
        and isinstance(getattr(module, "kv_a_layernorm", None), nn.Module)
    )


def _mla_projection_weights(
    module: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-head content-key and value up-projection weights."""
    weight = module.kv_b_proj.weight.view(
        int(module.num_heads),
        int(module.qk_nope_head_dim) + int(module.v_head_dim),
        int(module.kv_lora_rank),
    )
    return torch.split(
        weight,
        [int(module.qk_nope_head_dim), int(module.v_head_dim)],
        dim=1,
    )


def _project_mla_query_and_cache(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Project the shared portions of DeepSeek-style MLA."""
    batch_size, sequence_length = hidden_states.shape[:-1]
    query_shape = (
        batch_size,
        sequence_length,
        int(module.num_heads),
        int(module.qk_head_dim),
    )
    if module.q_lora_rank is None:
        query_states = module.q_proj(hidden_states)
    else:
        query_states = module.q_b_proj(
            module.q_a_layernorm(module.q_a_proj(hidden_states))
        )
    query_states = query_states.view(query_shape).transpose(1, 2)
    query_nope, query_rope = torch.split(
        query_states,
        [int(module.qk_nope_head_dim), int(module.qk_rope_head_dim)],
        dim=-1,
    )

    compressed_kv = module.kv_a_proj_with_mqa(hidden_states)
    raw_kv_latent, key_rope = torch.split(
        compressed_kv,
        [int(module.kv_lora_rank), int(module.qk_rope_head_dim)],
        dim=-1,
    )
    kv_latent = module.kv_a_layernorm(raw_kv_latent).view(
        batch_size, 1, sequence_length, int(module.kv_lora_rank)
    )
    raw_kv_latent = raw_kv_latent.view(
        batch_size, 1, sequence_length, int(module.kv_lora_rank)
    )
    key_rope = key_rope.view(
        batch_size, 1, sequence_length, int(module.qk_rope_head_dim)
    )
    return query_nope, query_rope, raw_kv_latent, kv_latent, key_rope


def _finish_mla_lod_attention(
    module: nn.Module,
    query_nope: torch.Tensor,
    query_rope: torch.Tensor,
    raw_kv_latent: torch.Tensor,
    kv_latent: torch.Tensor,
    key_rope: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, None]:
    """Run absorbed latent attention and restore head-specific values."""
    from .hf_pytorch_lod_attention import hf_lod_attention_forward

    batch_size, _, sequence_length, _ = query_nope.shape
    key_weight, value_weight = _mla_projection_weights(module)
    latent_query = torch.einsum(
        "bhsc,hcr->bhsr", query_nope, key_weight
    )
    absorbed_query = torch.cat((latent_query, query_rope), dim=-1)
    state_normalization = (
        module._hf_lod_settings.config.mla_state_key_normalization
    )
    key_latent = raw_kv_latent if state_normalization != "none" else kv_latent
    latent_key = torch.cat((key_latent, key_rope), dim=-1)
    latent_value = kv_latent

    if past_key_values is not None:
        latent_key, latent_value = past_key_values.update(
            latent_key, latent_value, int(module.layer_idx)
        )

    latent_output, _ = hf_lod_attention_forward(
        module,
        absorbed_query,
        latent_key,
        latent_value,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        **kwargs,
    )
    value_output = torch.einsum(
        "bshr,hvr->bshv", latent_output, value_weight
    )
    value_output = value_output.reshape(batch_size, sequence_length, -1)
    return module.o_proj(value_output), None


def _glm4_moe_lite_lod_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run GLM-4.7-Flash attention directly on its compressed MLA cache."""
    if _use_native_transient_attention(self, hidden_states, past_key_values):
        return _native_transient_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
        )
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
    )

    query_nope, query_rope, raw_kv_latent, kv_latent, key_rope = (
        _project_mla_query_and_cache(self, hidden_states)
    )

    cos, sin = position_embeddings
    if self.config.rope_interleave:
        query_rope, key_rope = apply_rotary_pos_emb_interleave(
            query_rope, key_rope, cos, sin
        )
    else:
        query_rope, key_rope = apply_rotary_pos_emb(
            query_rope, key_rope, cos, sin
        )

    return _finish_mla_lod_attention(
        self,
        query_nope,
        query_rope,
        raw_kv_latent,
        kv_latent,
        key_rope,
        attention_mask,
        past_key_values,
        kwargs,
    )


def _deepseek_v2_lod_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    position_embeddings: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run DeepSeek-V2 attention directly on its compressed MLA cache."""
    if _use_native_transient_attention(self, hidden_states, past_key_values):
        return _native_transient_forward(
            self,
            hidden_states,
            attention_mask,
            past_key_values,
            position_embeddings,
            **kwargs,
        )
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
        apply_rotary_emb,
    )

    if position_embeddings is None:
        raise ValueError("DeepSeek-V2 MLA requires precomputed RoPE frequencies")
    query_nope, query_rope, raw_kv_latent, kv_latent, key_rope = (
        _project_mla_query_and_cache(self, hidden_states)
    )
    query_rope, key_rope = apply_rotary_emb(
        query_rope,
        key_rope,
        position_embeddings.to(query_rope.device),
    )
    return _finish_mla_lod_attention(
        self,
        query_nope,
        query_rope,
        raw_kv_latent,
        kv_latent,
        key_rope,
        attention_mask,
        past_key_values,
        kwargs,
    )


def install_mla_lod_adapter(module: nn.Module) -> bool:
    """Install a supported pre-expansion MLA forward, if one is required."""
    if is_glm4_moe_lite_mla(module):
        adapter_name = "glm4_moe_lite"
        forward = _glm4_moe_lite_lod_forward
    elif is_deepseek_v2_mla(module):
        adapter_name = "deepseek_v2"
        forward = _deepseek_v2_lod_forward
    else:
        return False
    if not hasattr(module, "_hf_lod_original_forward"):
        module._hf_lod_original_forward = module.forward
        module._hf_lod_native_attn_implementation = (
            module.config._attn_implementation
        )
        module.forward = MethodType(forward, module)
    module._hf_lod_mla_adapter = adapter_name
    return True


__all__ = [
    "install_mla_lod_adapter",
    "is_deepseek_v2_mla",
    "is_glm4_moe_lite_mla",
]
