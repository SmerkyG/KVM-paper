"""Legacy Qwen3.5 wrapper around the generic Triton LOD core.

New integrations should use :mod:`hf_pytorch_lod_attention`. This module keeps
the original experimental graft API available for benchmark scripts.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    apply_rotary_pos_emb,
)

from .triton_lod_attention import TritonLODAttentionCore


class Qwen3_5TwoLevelAttention(TritonLODAttentionCore, Qwen3_5Attention):
    """Compatibility wrapper retaining the original direct Qwen graft."""

    def __init__(self, config, layer_idx: int) -> None:
        Qwen3_5Attention.__init__(self, config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, kwargs
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
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

        query_len = int(query_states.size(2))
        is_prefill = query_len > 1 or not hasattr(self, "_lod_state")
        if cache_position is not None and int(cache_position[0].item()) == 0:
            is_prefill = True
        total_len = (
            int(cache_position[-1].item()) + 1
            if cache_position is not None
            else (
                query_len
                if is_prefill
                else int(self._lod_state["total_len"]) + query_len
            )
        )
        if (
            past_key_values is not None
            and self.layer_idx == past_key_values.transformer_layers[0]
        ):
            metadata_shape = (int(key_states.size(0)), 0, total_len, 0)
            metadata_key = past_key_values.key_cache[self.layer_idx]
            metadata_value = past_key_values.value_cache[self.layer_idx]
            if metadata_key is None:
                past_key_values.key_cache[self.layer_idx] = key_states.new_empty(
                    metadata_shape
                )
                past_key_values.value_cache[self.layer_idx] = value_states.new_empty(
                    metadata_shape
                )
            else:
                metadata_key.resize_(metadata_shape)
                if metadata_value is None:
                    raise AssertionError("Qwen3.5 LOD metadata V cache is missing")
                metadata_value.resize_(metadata_shape)
        if is_prefill:
            attention_output = self._prefill_attention(
                query_states, key_states, value_states
            )
        else:
            if query_len != 1:
                raise AssertionError("Qwen3.5 LOD cached decode expects one query")
            attention_output = self._decode_attention(
                query_states,
                key_states,
                value_states,
                total_len=total_len,
            )

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1)
        attention_output = attention_output * torch.sigmoid(gate)
        return self.o_proj(attention_output), None


def pop_qwen35_dynamic_open_statistics(model: nn.Module) -> dict[str, dict]:
    """Collect and clear query counts by number of opened state slots."""
    result: dict[str, dict] = {}
    for phase in ("prefill", "decode"):
        parts = []
        for module in model.modules():
            if not isinstance(module, Qwen3_5TwoLevelAttention):
                continue
            attribute = f"_lod_dynamic_{phase}_histograms"
            parts.extend(getattr(module, attribute, ()))
            if hasattr(module, attribute):
                delattr(module, attribute)
        if not parts:
            continue
        histogram = torch.stack(parts).sum(dim=0)
        rows = int(histogram.sum().item())
        indices = torch.arange(
            int(histogram.numel()), device=histogram.device, dtype=torch.long
        )
        opened = int((indices * histogram).sum().item())
        result[phase] = {
            "histogram": histogram.cpu().tolist(),
            "queries": rows,
            "mean_opened": opened / rows if rows else 0.0,
        }
    return result


def qwen35_page_quantization_statistics(model: nn.Module) -> dict[str, int | float]:
    """Summarize how much of the paged leaf archive has been quantized."""
    layers = active_pages = quantized_pages = 0
    leaf_entries = quantized_leaf_entries = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5TwoLevelAttention):
            continue
        state = getattr(module, "_lod_state", None)
        if not isinstance(state, dict):
            continue
        cache = state.get("page_cache")
        if not isinstance(cache, dict):
            continue
        page_counts = cache.get("page_counts")
        page_quantized = cache.get("page_quantized")
        page_quantized_counts = cache.get("page_quantized_counts")
        if not isinstance(page_counts, torch.Tensor):
            continue
        if isinstance(page_quantized_counts, torch.Tensor):
            quantized_page_mask = page_quantized_counts.gt(0)
            quantized_entries = int(page_quantized_counts.sum().item())
        elif isinstance(page_quantized, torch.Tensor):
            quantized_page_mask = page_quantized
            quantized_entries = int(page_quantized.sum().item()) * int(
                module.leaf_page_size
            )
        else:
            continue
        layers += 1
        active_pages += int(page_counts.gt(0).sum().item())
        quantized_pages += int(quantized_page_mask.sum().item())
        leaf_entries += int(page_counts.sum().item())
        quantized_leaf_entries += quantized_entries
    return {
        "layers": layers,
        "active_pages": active_pages,
        "quantized_pages": quantized_pages,
        "leaf_entries": leaf_entries,
        "quantized_leaf_entries": quantized_leaf_entries,
        "quantized_leaf_fraction": (
            quantized_leaf_entries / leaf_entries if leaf_entries else 0.0
        ),
    }


def graft_qwen35_two_level_attention(
    model: nn.Module,
    *,
    topk: int = 8,
    state_growth_factor: float = 16.0,
    leaf_attention_backend: str = "packed",
) -> list[int]:
    """Replace every Qwen3.5 full-attention layer in-place, preserving weights."""
    text_model = getattr(getattr(model, "model", None), "language_model", None)
    if text_model is None:
        text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise TypeError("could not locate Qwen3.5 text decoder layers")
    if leaf_attention_backend not in {"packed", "paged"}:
        raise ValueError("leaf_attention_backend must be either 'packed' or 'paged'")

    replaced: list[int] = []
    for layer_index, layer in enumerate(text_model.layers):
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        if not isinstance(attention, Qwen3_5Attention):
            raise TypeError(
                f"layer {layer_index} attention has unexpected type "
                f"{type(attention)!r}"
            )
        attention.__class__ = Qwen3_5TwoLevelAttention
        attention.two_level_topk = topk
        attention.state_growth_factor = state_growth_factor
        attention.leaf_attention_backend = leaf_attention_backend
        replaced.append(layer_index)
    if not replaced:
        raise RuntimeError("no Qwen3.5 full-attention layers were found")
    return replaced


__all__ = [
    "Qwen3_5TwoLevelAttention",
    "graft_qwen35_two_level_attention",
    "pop_qwen35_dynamic_open_statistics",
    "qwen35_page_quantization_statistics",
]
