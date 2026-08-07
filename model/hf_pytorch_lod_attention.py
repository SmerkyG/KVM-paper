"""Registered Hugging Face backend with an LOD-owned inference cache.

The backend starts at Hugging Face's post-QKV/post-position-encoding attention
interface.  Projections, positional encoding, output gating, and output
projection therefore remain model-owned.  ``HFLODCache`` owns every tensor
used by LOD attention, including exact BF16 or INT4 leaves; Hugging Face's
cache API is used only for lifecycle and generation bookkeeping.

Only unpadded causal decoder self-attention is supported initially.  Models
whose attention modules do not use ``AttentionInterface`` and hybrid recurrent
caches should continue to use a model-specific compatibility adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import weakref
from typing import Any, Callable

import torch
from torch import nn
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.cache_utils import Cache, CacheLayerMixin

from .pytorch_lod_attention import LODConfig
from .pytorch_lod_attention_fast import (
    FastCoarseLODAttention,
    FastTwoLevelLODAttention,
)
from .pytorch_lod_attention_paged import (
    PagedLODConfig,
    PagedTwoLevelLODAttention,
)
from .triton_lod_engines import (
    KernelCoarseLODAttention,
    KernelLODCache,
    KernelRecursivePagedLODAttention,
    KernelTwoLevelLODAttention,
)


@dataclass(frozen=True)
class HFLODSettings:
    """Configuration installed on each compatible HF attention module."""

    config: LODConfig | PagedLODConfig
    open_count: int
    engine_backend: str
    backend_name: str

    def __post_init__(self) -> None:
        if self.engine_backend not in ("torch", "kernel"):
            raise ValueError("engine_backend must be 'torch' or 'kernel'")
        if not 0 <= self.open_count <= self.config.max_routes:
            raise ValueError("open_count must be between zero and max_routes")


def _build_engine(
    settings: HFLODSettings,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale: float | None,
) -> nn.Module:
    config = settings.config
    if settings.engine_backend == "torch":
        if settings.open_count == 0:
            return FastCoarseLODAttention(config)
        if isinstance(config, PagedLODConfig):
            return PagedTwoLevelLODAttention(
                config, default_open_count=settings.open_count
            )
        return FastTwoLevelLODAttention(
            config, default_open_count=settings.open_count
        )

    effective_scale = (
        float(scale)
        if scale is not None
        else float(query.size(-1)) ** -0.5
    )
    geometry = {
        "query_heads": int(query.size(1)),
        "key_value_heads": int(key.size(1)),
        "scale": effective_scale,
    }
    if settings.open_count == 0:
        return KernelCoarseLODAttention(config, **geometry)
    if isinstance(config, PagedLODConfig):
        return KernelRecursivePagedLODAttention(
            config,
            default_open_count=settings.open_count,
            **geometry,
        )
    return KernelTwoLevelLODAttention(
        config,
        default_open_count=settings.open_count,
        **geometry,
    )


def _map_batch_tensors(
    value: Any,
    *,
    batch_size: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Any:
    """Apply a batch transform to all batch-major tensors in an LOD cache."""
    if isinstance(value, torch.Tensor):
        if value.ndim and int(value.size(0)) == batch_size:
            return transform(value)
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _map_batch_tensors(
                    getattr(value, field.name),
                    batch_size=batch_size,
                    transform=transform,
                )
                for field in fields(value)
            }
        )
    if isinstance(value, dict):
        return {
            key: _map_batch_tensors(
                item, batch_size=batch_size, transform=transform
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _map_batch_tensors(
                item, batch_size=batch_size, transform=transform
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return type(value)(
            _map_batch_tensors(
                item, batch_size=batch_size, transform=transform
            )
            for item in value
        )
    return value


def _clear_engine_derived_state(engine: nn.Module | None) -> None:
    if engine is None:
        return
    for attribute in (
        "_posting_key",
        "_postings",
        "_region_key",
        "_region_pages",
    ):
        if hasattr(engine, attribute):
            setattr(engine, attribute, None)
    for attribute in (
        "_lod_decode_attention_buffers",
        "_lod_route_buffers",
        "_lod_state_maxsim_buffers",
        "_lod_state_update_buffers",
    ):
        if hasattr(engine, attribute):
            delattr(engine, attribute)


class HFLODCacheLayer(CacheLayerMixin):
    """One HF cache layer whose persistent storage is entirely LOD-owned."""

    is_compileable = False
    is_sliding = False

    def __init__(self, module: nn.Module, settings: HFLODSettings) -> None:
        super().__init__()
        self._module = weakref.ref(module)
        self.settings = settings
        self.engine: nn.Module | None = None
        self.lod_cache: Any | None = None
        self.pending_key: torch.Tensor | None = None
        self.pending_value: torch.Tensor | None = None
        self.total_length = 0
        self._batch_size = 0

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    @property
    def max_cache_len(self) -> int:
        return -1

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self._batch_size = int(key_states.size(0))
        self.keys = key_states.new_empty(
            int(key_states.size(0)), int(key_states.size(1)), 0, int(key_states.size(-1))
        )
        self.values = value_states.new_empty(
            int(value_states.size(0)),
            int(value_states.size(1)),
            0,
            int(value_states.size(-1)),
        )
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.pending_key is not None or self.pending_value is not None:
            raise RuntimeError("the previous staged LOD cache update was not consumed")
        if int(key_states.size(0)) != self._batch_size:
            raise ValueError("LOD cache batch size changed without a batch operation")
        if key_states.shape[:3] != value_states.shape[:3]:
            raise ValueError("staged LOD key/value shapes disagree")
        module = self._module()
        if module is None:
            raise RuntimeError("the attention module bound to this LOD cache was deleted")
        active = getattr(module, "_hf_lod_active_cache_layer", None)
        if active not in (None, self):
            raise RuntimeError("an attention module is already using another LOD cache")
        self.pending_key = key_states
        self.pending_value = value_states
        module._hf_lod_active_cache_layer = self
        # The registered backend consumes only this new block. Persistent K/V
        # are kept inside lod_cache, never in these HF compatibility sentinels.
        return key_states, value_states

    def consume(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float | None,
    ) -> torch.Tensor:
        if module is not self._module():
            raise RuntimeError("LOD cache layer was consumed by the wrong module")
        if self.pending_key is None or self.pending_value is None:
            raise RuntimeError("LOD attention did not receive a staged cache update")
        if key is not self.pending_key or value is not self.pending_value:
            raise RuntimeError("the model replaced staged K/V before LOD attention")
        if self.engine is None:
            self.engine = _build_engine(
                self.settings, query, key, scale=scale
            )
        previous_length = self.total_length
        try:
            output, next_cache = self.engine(
                query,
                key,
                value,
                cache=self.lod_cache,
                use_cache=True,
                scale=scale,
            )
            if next_cache is None:
                raise RuntimeError("LOD engine did not return its owned cache")
            expected_length = previous_length + int(key.size(2))
            if int(next_cache.total_length) != expected_length:
                raise AssertionError("LOD engine and HF cache lengths diverged")
            self.lod_cache = next_cache
            self.total_length = expected_length
            return output
        finally:
            self.pending_key = None
            self.pending_value = None
            module._hf_lod_active_cache_layer = None

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.total_length + int(cache_position.shape[0]), 0

    def get_seq_length(self) -> int:
        return self.total_length

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self) -> None:
        module = self._module()
        if module is not None and getattr(
            module, "_hf_lod_active_cache_layer", None
        ) is self:
            module._hf_lod_active_cache_layer = None
        self.pending_key = None
        self.pending_value = None
        self.lod_cache = None
        self.total_length = 0
        _clear_engine_derived_state(self.engine)
        if self.engine is not None and hasattr(self.engine, "reset_runtime_cache"):
            self.engine.reset_runtime_cache()
        if self.is_initialized:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]

    def _batch_transform(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        next_batch_size: int,
    ) -> None:
        if self.pending_key is not None or self.pending_value is not None:
            raise RuntimeError("cannot reorder a staged LOD cache update")
        if self.lod_cache is not None:
            self.lod_cache = _map_batch_tensors(
                self.lod_cache,
                batch_size=self._batch_size,
                transform=transform,
            )
        if self.is_initialized:
            self.keys = transform(self.keys)
            self.values = transform(self.values)
        self._batch_size = next_batch_size
        _clear_engine_derived_state(self.engine)
        if (
            self.engine is not None
            and isinstance(self.lod_cache, KernelLODCache)
        ):
            self.engine._lod_state = self.lod_cache.state

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.is_initialized:
            return
        self._batch_transform(
            lambda tensor: tensor.index_select(0, beam_idx.to(tensor.device)),
            next_batch_size=int(beam_idx.numel()),
        )

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError("batch repeats must be positive")
        if not self.is_initialized:
            return
        self._batch_transform(
            lambda tensor: tensor.repeat_interleave(repeats, dim=0),
            next_batch_size=self._batch_size * repeats,
        )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized:
            return
        self._batch_transform(
            lambda tensor: tensor.index_select(0, indices.to(tensor.device)),
            next_batch_size=int(indices.numel()),
        )

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.total_length + max_length
        if max_length >= self.total_length:
            return
        if max_length == 0:
            self.reset()
            return
        raise NotImplementedError(
            "partial rollback is not yet supported by HFLODCache"
        )


class HFLODCache(Cache):
    """HF cache protocol backed exclusively by per-layer LOD caches."""

    def __init__(self, layers: list[HFLODCacheLayer], *, backend_name: str) -> None:
        if not layers:
            raise ValueError("HFLODCache requires at least one attention layer")
        super().__init__(layers=layers)
        self.backend_name = backend_name

    @classmethod
    def for_model(cls, model: nn.Module) -> HFLODCache:
        indexed: dict[int, tuple[str, nn.Module, HFLODSettings]] = {}
        for name, module in model.named_modules():
            settings = getattr(module, "_hf_lod_settings", None)
            if settings is None:
                continue
            layer_index = getattr(module, "layer_idx", None)
            if not isinstance(layer_index, int):
                raise TypeError(f"LOD attention module {name!r} has no integer layer_idx")
            if layer_index in indexed:
                raise ValueError(f"multiple LOD attention modules use layer {layer_index}")
            indexed[layer_index] = (name, module, settings)
        if not indexed:
            raise RuntimeError("the model has no installed LOD attention modules")
        expected = list(range(max(indexed) + 1))
        if sorted(indexed) != expected:
            raise ValueError(
                "HFLODCache currently requires one causal attention module per layer"
            )
        backend_names = {settings.backend_name for _, _, settings in indexed.values()}
        if len(backend_names) != 1:
            raise ValueError("installed LOD modules use different backend names")
        layers = [
            HFLODCacheLayer(indexed[index][1], indexed[index][2])
            for index in expected
        ]
        return cls(layers, backend_name=backend_names.pop())


def hf_lod_attention_mask(*, attention_mask=None, **kwargs):
    """Preserve only the compact user mask; causal structure is internal."""
    del kwargs
    return attention_mask


def _validate_unpadded_mask(attention_mask: torch.Tensor | None) -> None:
    if attention_mask is None:
        return
    if attention_mask.ndim != 2:
        raise NotImplementedError(
            "HF LOD currently supports only an unpadded 2D attention mask"
        )
    torch._assert_async(
        torch.all(attention_mask == 1),
        "HF LOD currently requires unpadded causal sequences",
    )


def hf_lod_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Hugging Face ``AttentionInterface`` entry point for causal LOD."""
    settings = getattr(module, "_hf_lod_settings", None)
    if not isinstance(settings, HFLODSettings):
        raise RuntimeError(
            "the LOD backend reached an attention module that was not installed"
        )
    if kwargs.get("output_attentions", False):
        raise NotImplementedError("HF LOD does not return dense attention weights")
    if module.training and float(dropout) != 0.0:
        raise NotImplementedError("HF LOD does not yet implement attention dropout")
    if (
        settings.engine_backend == "kernel"
        and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in (query, key, value))
    ):
        raise NotImplementedError(
            "the kernel HF LOD backend is inference-only; use engine_backend='torch' for gradients"
        )
    if kwargs.get("softcap") not in (None, 0, 0.0):
        raise NotImplementedError("HF LOD does not yet implement attention soft-capping")
    _validate_unpadded_mask(attention_mask)

    active_layer = getattr(module, "_hf_lod_active_cache_layer", None)
    if active_layer is None:
        if int(query.size(2)) != int(key.size(2)):
            raise RuntimeError(
                "cached HF LOD inference requires an HFLODCache, not the default HF cache"
            )
        engine = getattr(module, "_hf_lod_transient_engine", None)
        if engine is None:
            engine = _build_engine(settings, query, key, scale=scaling)
            module._hf_lod_transient_engine = engine
        output, _ = engine(
            query,
            key,
            value,
            cache=None,
            use_cache=False,
            scale=scaling,
        )
    elif isinstance(active_layer, HFLODCacheLayer):
        output = active_layer.consume(
            module, query, key, value, scale=scaling
        )
    else:
        raise RuntimeError("attention module contains an invalid active LOD cache")
    return output.transpose(1, 2).contiguous(), None


def register_hf_lod_attention(backend_name: str = "lod") -> None:
    """Register the LOD attention and compact-mask functions globally."""
    AttentionInterface.register(backend_name, hf_lod_attention_forward)
    AttentionMaskInterface.register(backend_name, hf_lod_attention_mask)


def _compatible_attention_modules(model: nn.Module):
    for name, module in model.named_modules():
        if not isinstance(getattr(module, "layer_idx", None), int):
            continue
        if not bool(getattr(module, "is_causal", False)):
            continue
        if "attention" not in type(module).__name__.lower():
            continue
        yield name, module


def install_hf_lod_attention(
    model: nn.Module,
    *,
    config: LODConfig | PagedLODConfig | None = None,
    open_count: int = 8,
    leaf_dtype: torch.dtype | None = None,
    engine_backend: str = "torch",
    backend_name: str = "lod",
    submodel_key: str | None = None,
) -> list[str]:
    """Install the registered LOD backend on compatible causal HF layers.

    Cached generation must receive ``HFLODCache.for_model(model)`` as
    ``past_key_values``.  ``submodel_key`` selects only one multimodal
    backbone through Hugging Face's per-subconfig attention dispatch.
    """
    config = LODConfig() if config is None else config
    if leaf_dtype is not None:
        config = replace(config, leaf_dtype=leaf_dtype)
    settings = HFLODSettings(
        config=config,
        open_count=open_count,
        engine_backend=engine_backend,
        backend_name=backend_name,
    )
    register_hf_lod_attention(backend_name)
    installed = []
    for name, module in _compatible_attention_modules(model):
        module._hf_lod_settings = settings
        module._hf_lod_active_cache_layer = None
        installed.append(name)
    if not installed:
        raise RuntimeError("no compatible causal AttentionInterface modules were found")
    implementation: str | dict[str, str] = backend_name
    if submodel_key is not None:
        implementation = {submodel_key: backend_name}
    model.set_attn_implementation(implementation)
    return installed


def new_hf_lod_cache(model: nn.Module) -> HFLODCache:
    """Construct a fresh model-bound LOD cache for prefill and generation."""
    return HFLODCache.for_model(model)


_LEGACY_QWEN_EXPORTS = {
    "Qwen3_5FastLODAttention",
    "replace_qwen35_attention_with_lod",
    "reset_hf_lod_caches",
}


def __getattr__(name: str):
    if name in _LEGACY_QWEN_EXPORTS:
        from . import hf_qwen35_lod_attention

        return getattr(hf_qwen35_lod_attention, name)
    raise AttributeError(name)


__all__ = [
    "HFLODCache",
    "HFLODCacheLayer",
    "HFLODSettings",
    "hf_lod_attention_forward",
    "hf_lod_attention_mask",
    "install_hf_lod_attention",
    "new_hf_lod_cache",
    "register_hf_lod_attention",
    *_LEGACY_QWEN_EXPORTS,
]
