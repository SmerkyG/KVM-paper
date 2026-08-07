"""Hugging Face attention replacements backed by pluggable LOD engines.

The LOD core deliberately starts after QKV projection and positional encoding.
This file supplies the thin model-specific adapter needed to replace a Hugging
Face attention module while preserving its projections, normalization, RoPE,
output gate, and output projection.

The default engine is the clean PyTorch implementation; ``engine_backend`` can
instead select the inference-only optimized kernel engine.  Currently Qwen3.5
text full-attention layers are supported.  Its interleaved Gated DeltaNet
layers are left untouched.  Inputs must be unpadded causal sequences; ordinary
HF causal masks are accepted but padding entries are not interpreted by the
LOD core.  Beam-cache reordering is not yet supported.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    apply_rotary_pos_emb,
)

from .pytorch_lod_attention import LODCache, LODConfig
from .pytorch_lod_attention_fast import FastTwoLevelLODAttention
from .pytorch_lod_attention_paged import (
    PagedLODCache,
    PagedLODConfig,
    PagedTwoLevelLODAttention,
)
from .triton_lod_engines import (
    KernelCoarseLODAttention,
    KernelLODCache,
    KernelRecursivePagedLODAttention,
    KernelTwoLevelLODAttention,
)


class Qwen3_5FastLODAttention(Qwen3_5Attention):
    """Qwen3.5 attention whose post-RoPE attention field uses LOD."""

    lod_engine: (
        FastTwoLevelLODAttention
        | PagedTwoLevelLODAttention
        | KernelCoarseLODAttention
        | KernelTwoLevelLODAttention
        | KernelRecursivePagedLODAttention
    )
    _lod_cache: LODCache | PagedLODCache | KernelLODCache | None
    _lod_hf_cache_id: int | None

    def reset_lod_cache(self) -> None:
        self._lod_cache = None
        self._lod_hf_cache_id = None
        reset_runtime_cache = getattr(self.lod_engine, "reset_runtime_cache", None)
        if reset_runtime_cache is not None:
            reset_runtime_cache()
        for attribute in (
            "_posting_key",
            "_postings",
            "_region_key",
            "_region_pages",
        ):
            if hasattr(self.lod_engine, attribute):
                setattr(self.lod_engine, attribute, None)

    def _update_qwen_cache_length(
        self,
        past_key_values,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        total_length: int,
    ) -> None:
        """Keep Qwen's hybrid cache length without retaining its duplicate KV."""
        transformer_layers = getattr(past_key_values, "transformer_layers", ())
        if not transformer_layers or self.layer_idx != transformer_layers[0]:
            return
        key_cache = getattr(past_key_values, "key_cache", None)
        value_cache = getattr(past_key_values, "value_cache", None)
        if key_cache is None or value_cache is None:
            raise TypeError("Qwen3.5 cache does not expose key/value cache lists")
        metadata_shape = (int(key_states.size(0)), 0, total_length, 0)
        metadata_key = key_cache[self.layer_idx]
        metadata_value = value_cache[self.layer_idx]
        if metadata_key is None:
            key_cache[self.layer_idx] = key_states.new_empty(metadata_shape)
            value_cache[self.layer_idx] = value_states.new_empty(metadata_shape)
            return
        metadata_key.resize_(metadata_shape)
        if metadata_value is None:
            raise AssertionError("Qwen3.5 metadata value cache is missing")
        metadata_value.resize_(metadata_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, cache_position, kwargs
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        use_cache = past_key_values is not None
        hf_cache_id = id(past_key_values) if use_cache else None
        if not use_cache or hf_cache_id != self._lod_hf_cache_id:
            self.reset_lod_cache()
        lod_cache = self._lod_cache if use_cache else None
        attention_output, next_cache = self.lod_engine(
            query_states,
            key_states,
            value_states,
            cache=lod_cache,
            use_cache=use_cache,
            scale=self.scaling,
        )
        if use_cache:
            if next_cache is None:
                raise AssertionError("LOD attention did not return a decode cache")
            self._lod_cache = next_cache
            self._lod_hf_cache_id = hf_cache_id
            self._update_qwen_cache_length(
                past_key_values,
                key_states,
                value_states,
                next_cache.total_length,
            )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1)
        attention_output = attention_output * torch.sigmoid(gate)
        return self.o_proj(attention_output), None


def _qwen35_text_layers(model: nn.Module):
    text_model = getattr(getattr(model, "model", None), "language_model", None)
    if text_model is None:
        text_model = getattr(model, "model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise TypeError("could not locate Qwen3.5 text decoder layers")
    return layers


def replace_qwen35_attention_with_lod(
    model: nn.Module,
    *,
    config: LODConfig | PagedLODConfig | None = None,
    open_count: int = 8,
    leaf_dtype: torch.dtype | None = None,
    engine_backend: str = "torch",
) -> list[int]:
    """Replace Qwen3.5 full-attention modules in-place, preserving weights.

    ``engine_backend="kernel"`` selects optimized state, routing, coarse,
    page, and leaf kernels.  An ``open_count`` of zero installs its coarse-only
    engine; a positive count uses all routed-region leaves unless a positive
    ``PagedLODConfig.page_size`` requests recursive one-page routing.
    """
    if config is None:
        config = LODConfig()
    if leaf_dtype is not None:
        config = replace(config, leaf_dtype=leaf_dtype)
    if engine_backend not in ("torch", "kernel"):
        raise ValueError("engine_backend must be 'torch' or 'kernel'")
    replaced_layers: list[int] = []
    for layer_index, layer in enumerate(_qwen35_text_layers(model)):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        if not isinstance(attention, Qwen3_5Attention):
            raise TypeError(
                f"layer {layer_index} attention has unexpected type "
                f"{type(attention)!r}"
            )
        if engine_backend == "kernel":
            query_heads = attention.q_proj.out_features // (2 * attention.head_dim)
            key_value_heads = attention.k_proj.out_features // attention.head_dim
            if open_count == 0:
                lod_engine = KernelCoarseLODAttention(
                    config,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    scale=attention.scaling,
                )
            elif isinstance(config, PagedLODConfig) and config.page_size is not None:
                lod_engine = KernelRecursivePagedLODAttention(
                    config,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    scale=attention.scaling,
                    default_open_count=open_count,
                )
            else:
                lod_engine = KernelTwoLevelLODAttention(
                    config,
                    query_heads=query_heads,
                    key_value_heads=key_value_heads,
                    scale=attention.scaling,
                    default_open_count=open_count,
                )
        else:
            engine = (
                PagedTwoLevelLODAttention
                if isinstance(config, PagedLODConfig)
                else FastTwoLevelLODAttention
            )
            lod_engine = engine(config, default_open_count=open_count)
        attention.__class__ = Qwen3_5FastLODAttention
        attention.lod_engine = lod_engine
        attention._lod_cache = None
        attention._lod_hf_cache_id = None
        replaced_layers.append(layer_index)
    if not replaced_layers:
        raise RuntimeError("no Qwen3.5 full-attention layers were found")
    return replaced_layers


def reset_hf_lod_caches(model: nn.Module) -> None:
    """Clear module-local LOD caches before starting an unrelated sequence."""
    for module in model.modules():
        if isinstance(module, Qwen3_5FastLODAttention):
            module.reset_lod_cache()


__all__ = [
    "Qwen3_5FastLODAttention",
    "replace_qwen35_attention_with_lod",
    "reset_hf_lod_caches",
]
