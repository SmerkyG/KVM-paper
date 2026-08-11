"""Mixed native/LOD cache support for Hugging Face hybrid decoders.

Hybrid models such as Qwen3.5 interleave recurrent and full-attention layers.
Their native cache must retain recurrent states, while full-attention layers
should use the same LOD-owned cache as ordinary decoder-only models.  This
module supplies that thin cache bridge without making the attention backend
model-specific.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class HybridHFLODCache:
    """Delegate recurrent state to a native cache and full attention to LOD."""

    is_compileable = False

    def __init__(
        self,
        native_cache: Any,
        lod_layers: dict[int, Any],
        *,
        backend_name: str,
    ) -> None:
        if not lod_layers:
            raise ValueError("a hybrid LOD cache requires at least one LOD layer")
        self.native_cache = native_cache
        self.lod_layers = lod_layers
        self.backend_name = backend_name
        self._padding_plan: Any | None = None
        for layer in lod_layers.values():
            layer._bind_owner(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_cache, name)

    def __len__(self) -> int:
        return len(self.native_cache)

    def _get_padding_plan(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
    ):
        if self._padding_plan is None:
            from .hf_lod_left_padding import build_padding_plan

            self._padding_plan = build_padding_plan(
                attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
        elif (
            self._padding_plan.batch_size != batch_size
            or self._padding_plan.padded_length != sequence_length
        ):
            raise RuntimeError("HF LOD layers received inconsistent prompt batches")
        return self._padding_plan

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.lod_layers.get(layer_idx)
        if layer is not None:
            return layer.update(key_states, value_states, cache_kwargs)
        return self.native_cache.update(
            key_states, value_states, layer_idx, cache_kwargs
        )

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer = self.lod_layers.get(layer_idx) if layer_idx is not None else None
        if layer is not None:
            return int(layer.get_seq_length())
        native_layers = getattr(self.native_cache, "layers", None)
        if isinstance(native_layers, list) and layer_idx is not None:
            native_layer = (
                native_layers[layer_idx]
                if 0 <= layer_idx < len(native_layers)
                else None
            )
            native_length = getattr(native_layer, "get_seq_length", None)
            if callable(native_length):
                return int(native_length())
        # Hybrid models commonly ask a recurrent layer (or layer zero) for the
        # global decoded length.  Native recurrent state has no sequence axis,
        # so use any LOD layer as the authoritative logical length.
        return int(next(iter(self.lod_layers.values())).get_seq_length())

    def get_mask_sizes(
        self, query_length: int | torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        length = (
            int(query_length.shape[0])
            if isinstance(query_length, torch.Tensor)
            else int(query_length)
        )
        layer = self.lod_layers.get(layer_idx)
        if layer is not None:
            return layer.get_mask_sizes(length)
        if isinstance(getattr(self.native_cache, "layers", None), list):
            return self.native_cache.get_mask_sizes(length, layer_idx)
        return self.get_seq_length() + length, 0

    def get_max_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is None:
            return max(
                layer.get_max_length() for layer in self.lod_layers.values()
            )
        layer = self.lod_layers.get(layer_idx)
        if layer is not None:
            return int(layer.get_max_length())
        native = getattr(self.native_cache, "get_max_length", None)
        return int(native(layer_idx)) if callable(native) else -1

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.get_max_length(layer_idx)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        native = getattr(self.native_cache, "reorder_cache", None)
        if callable(native):
            native(beam_idx)
        for layer in self.lod_layers.values():
            layer.reorder_cache(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        native = getattr(self.native_cache, "batch_repeat_interleave", None)
        if callable(native):
            native(repeats)
        elif repeats != 1:
            raise NotImplementedError(
                "this hybrid model's native cache does not support beam expansion"
            )
        for layer in self.lod_layers.values():
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        native = getattr(self.native_cache, "batch_select_indices", None)
        if callable(native):
            native(indices)
        else:
            raise NotImplementedError(
                "this hybrid model's native cache does not support batch selection"
            )
        for layer in self.lod_layers.values():
            layer.batch_select_indices(indices)

    def crop(self, max_length: int) -> None:
        current = self.get_seq_length()
        if max_length < 0:
            max_length = current + max_length
        if max_length >= current:
            return
        if max_length == 0:
            self.reset()
            return
        raise NotImplementedError("partial rollback is not supported by hybrid LOD")

    def reset(self) -> None:
        for layer in self.lod_layers.values():
            layer.reset()
        native_reset = getattr(self.native_cache, "reset", None)
        if callable(native_reset):
            native_reset()
        else:
            for name in (
                "conv_states",
                "recurrent_states",
                "key_cache",
                "value_cache",
            ):
                values = getattr(self.native_cache, name, None)
                if isinstance(values, list):
                    values[:] = [None] * len(values)
        self._padding_plan = None


def _installed_lod_modules(model: nn.Module):
    indexed = {}
    for name, module in model.named_modules():
        settings = getattr(module, "_hf_lod_settings", None)
        if settings is None:
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if not isinstance(layer_idx, int):
            raise TypeError(f"LOD attention module {name!r} has no integer layer_idx")
        if layer_idx in indexed:
            raise ValueError(f"multiple LOD attention modules use layer {layer_idx}")
        indexed[layer_idx] = (module, settings)
    return indexed


def maybe_new_hybrid_hf_lod_cache(model: nn.Module):
    """Return a supported hybrid cache, or ``None`` for ordinary decoders."""
    indexed = _installed_lod_modules(model)
    if not indexed:
        return None
    expected = list(range(max(indexed) + 1))
    if sorted(indexed) == expected:
        return None

    text_config = model.config.get_text_config(decoder=True)
    config_module = type(text_config).__module__
    from .hf_pytorch_lod_attention import HFLODCacheLayer

    backend_names = {settings.backend_name for _, settings in indexed.values()}
    if len(backend_names) != 1:
        raise ValueError("installed LOD modules use different backend names")
    layers = {
        layer_idx: HFLODCacheLayer(module, settings)
        for layer_idx, (module, settings) in indexed.items()
    }
    if config_module.startswith("transformers.models.qwen3_5."):
        from transformers import DynamicCache

        native_cache = DynamicCache(config=text_config)
    else:
        layer_types = getattr(text_config, "layer_types", None)
        lod_indices = set(indexed)
        native_types = {
            layer_type
            for layer_idx, layer_type in enumerate(layer_types or ())
            if layer_idx not in lod_indices
        }
        if not native_types or not native_types.issubset(
            {"sliding_attention", "chunked_attention"}
        ):
            raise NotImplementedError(
                "this hybrid decoder needs a native-cache bridge before it can use HF LOD"
            )
        from transformers import DynamicCache

        native_cache = DynamicCache(config=text_config)
    return HybridHFLODCache(
        native_cache, layers, backend_name=backend_names.pop()
    )


def is_hybrid_hf_lod_cache(value: Any) -> bool:
    return isinstance(value, HybridHFLODCache)


__all__ = [
    "HybridHFLODCache",
    "is_hybrid_hf_lod_cache",
    "maybe_new_hybrid_hf_lod_cache",
]
