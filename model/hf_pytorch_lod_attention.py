"""Registered Hugging Face backend with an LOD-owned inference cache.

The backend starts at Hugging Face's post-QKV/post-position-encoding attention
interface.  Projections, positional encoding, output gating, and output
projection therefore remain model-owned.  ``HFLODCache`` owns every tensor
used by LOD attention, including exact BF16 or INT4 leaves; Hugging Face's
cache API is used only for lifecycle and generation bookkeeping.

Left-padded batches are partitioned only at the attention boundary, preserving
batching through the rest of the model while keeping padding out of every LOD
state schedule. Models whose attention modules do not use ``AttentionInterface``
and hybrid recurrent caches should continue to use a model-specific
compatibility adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from copy import copy
from types import MethodType
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
    left_padding_mode: str = "chunk_aligned"

    def __post_init__(self) -> None:
        if self.engine_backend not in ("torch", "kernel"):
            raise ValueError("engine_backend must be 'torch' or 'kernel'")
        if self.left_padding_mode not in ("exact", "chunk_aligned"):
            raise ValueError(
                "left_padding_mode must be 'exact' or 'chunk_aligned'"
            )
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
        self._owner_cache: weakref.ReferenceType[HFLODCache] | None = None
        self._padding_runtime: Any | None = None

    def _bind_owner(self, cache: Any) -> None:
        self._owner_cache = weakref.ref(cache)

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
        attention_mask: torch.Tensor | None,
        scale: float | None,
    ) -> torch.Tensor:
        if module is not self._module():
            raise RuntimeError("LOD cache layer was consumed by the wrong module")
        if self.pending_key is None or self.pending_value is None:
            raise RuntimeError("LOD attention did not receive a staged cache update")
        if key is not self.pending_key or value is not self.pending_value:
            raise RuntimeError("the model replaced staged K/V before LOD attention")
        previous_length = self.total_length
        try:
            if previous_length == 0:
                owner = self._owner_cache() if self._owner_cache is not None else None
                if owner is None:
                    raise RuntimeError(
                        "LOD cache layer is not bound to its outer cache"
                    )
                plan = owner._get_padding_plan(
                    attention_mask,
                    batch_size=int(query.size(0)),
                    sequence_length=int(query.size(2)),
                )
                if plan.requires_grouping:
                    from .hf_lod_left_padding import (
                        GroupedHFLODRuntime,
                        chunk_align_padding_plan,
                    )

                    if self.settings.left_padding_mode == "chunk_aligned":
                        plan = chunk_align_padding_plan(
                            plan,
                            chunk_size=self.settings.config.chunk_size,
                            minimum_length=self.settings.config.local_window,
                        )

                    self._padding_runtime = GroupedHFLODRuntime(
                        plan, device=query.device
                    )

            if self._padding_runtime is not None:
                output = self._padding_runtime.consume(
                    self.settings,
                    query,
                    key,
                    value,
                    initial_prefill=previous_length == 0,
                    scale=scale,
                )
            else:
                if self.engine is None:
                    self.engine = _build_engine(
                        self.settings, query, key, scale=scale
                    )
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
            self.total_length = previous_length + int(key.size(2))
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
        if self._padding_runtime is not None:
            self._padding_runtime.reset()
        self._padding_runtime = None
        if self.is_initialized:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]

    def _batch_select(self, indices: torch.Tensor) -> None:
        if self.pending_key is not None or self.pending_value is not None:
            raise RuntimeError("cannot reorder a staged LOD cache update")
        indices = indices.to(self.device)
        if self._padding_runtime is not None:
            self._padding_runtime.select_batch(
                indices, batch_size=self._batch_size
            )
        elif self.lod_cache is not None:
            self.lod_cache = _map_batch_tensors(
                self.lod_cache,
                batch_size=self._batch_size,
                transform=lambda tensor: tensor.index_select(
                    0, indices.to(tensor.device)
                ),
            )
        if self.is_initialized:
            self.keys = self.keys.index_select(0, indices)
            self.values = self.values.index_select(0, indices)
        self._batch_size = int(indices.numel())
        _clear_engine_derived_state(self.engine)
        if self.engine is not None and isinstance(self.lod_cache, KernelLODCache):
            self.engine._lod_state = self.lod_cache.state

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if not self.is_initialized:
            return
        self._batch_select(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError("batch repeats must be positive")
        if not self.is_initialized:
            return
        indices = torch.arange(
            self._batch_size, dtype=torch.long, device=self.device
        ).repeat_interleave(repeats)
        self._batch_select(indices)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized:
            return
        self._batch_select(indices)

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
        self._padding_plan: Any | None = None
        for layer in layers:
            layer._bind_owner(self)

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

    def reset(self) -> None:
        super().reset()
        self._padding_plan = None

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
    active_layer = getattr(module, "_hf_lod_active_cache_layer", None)
    if active_layer is None:
        if int(query.size(2)) != int(key.size(2)):
            raise RuntimeError(
                "cached HF LOD inference requires an HFLODCache, not the default HF cache"
            )
        from .hf_lod_left_padding import (
            build_padding_plan,
            grouped_transient_attention,
        )

        plan = build_padding_plan(
            attention_mask,
            batch_size=int(query.size(0)),
            sequence_length=int(query.size(2)),
        )
        if plan.requires_grouping:
            output = grouped_transient_attention(
                module,
                settings,
                query,
                key,
                value,
                plan,
                scale=scaling,
            )
        else:
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
            module,
            query,
            key,
            value,
            attention_mask=attention_mask,
            scale=scaling,
        )
    else:
        raise RuntimeError("attention module contains an invalid active LOD cache")
    return output.transpose(1, 2).contiguous(), None


def register_hf_lod_attention(backend_name: str = "lod") -> None:
    """Register the LOD attention and compact-mask functions globally."""
    AttentionInterface.register(backend_name, hf_lod_attention_forward)
    AttentionMaskInterface.register(backend_name, hf_lod_attention_mask)


def _install_generation_cache_factory(model: nn.Module) -> None:
    if bool(getattr(model, "_hf_lod_generation_cache_factory_installed", False)):
        return
    if not callable(getattr(model, "_prepare_cache_for_generation", None)):
        return
    model._hf_lod_generation_cache_factory_installed = True

    def prepare_lod_cache_for_generation(
        self,
        generation_config,
        model_kwargs,
        generation_mode,
        batch_size,
        max_cache_length,
    ) -> None:
        del generation_mode, batch_size, max_cache_length
        supplied_cache = model_kwargs.get("past_key_values")
        if supplied_cache is not None:
            from .hf_lod_hybrid_cache import is_hybrid_hf_lod_cache

            if not isinstance(supplied_cache, HFLODCache) and not is_hybrid_hf_lod_cache(
                supplied_cache
            ):
                raise TypeError(
                    "a model with HF LOD installed requires an LOD-owned cache "
                    "for generation"
                )
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    "HF LOD cache ownership cannot be combined with "
                    "cache_implementation"
                )
            return
        if generation_config.use_cache is False:
            return
        if generation_config.cache_implementation is not None:
            raise ValueError(
                "HF LOD cache ownership cannot be combined with cache_implementation"
            )
        model_kwargs["past_key_values"] = new_hf_lod_cache(self)

    model._prepare_cache_for_generation = MethodType(
        prepare_lod_cache_for_generation, model
    )


def _decoder_config(model: nn.Module):
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config(decoder=True) if callable(get_text_config) else config


def _causal_attention_modules(model: nn.Module):
    for name, module in model.named_modules():
        if not isinstance(getattr(module, "layer_idx", None), int):
            continue
        if not bool(getattr(module, "is_causal", False)):
            continue
        if "attention" not in type(module).__name__.lower():
            continue
        yield name, module


def _compatible_attention_modules(model: nn.Module):
    """Select full/global causal attention, never local or sliding layers."""
    decoder_config = _decoder_config(model)
    layer_types = getattr(decoder_config, "layer_types", None)
    for name, module in _causal_attention_modules(model):
        layer_idx = module.layer_idx
        if layer_types is not None:
            if layer_idx >= len(layer_types):
                raise ValueError(
                    f"attention module {name!r} exceeds config.layer_types"
                )
            if layer_types[layer_idx] not in ("full_attention", "attention"):
                continue
        elif getattr(module, "sliding_window", None) is not None:
            continue
        elif getattr(decoder_config, "sliding_window", None) is not None:
            # Without a per-layer pattern, a configured window denotes an
            # all-sliding decoder (for example classic Mistral).
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
    left_padding_mode: str = "chunk_aligned",
) -> list[str]:
    """Install the registered LOD backend on compatible causal HF layers.

    ``model.generate`` automatically creates an ``HFLODCache``. Direct cached
    forward calls must receive ``HFLODCache.for_model(model)`` as
    ``past_key_values``. ``submodel_key`` selects only one multimodal backbone
    through Hugging Face's per-subconfig attention dispatch.
    """
    config = LODConfig() if config is None else config
    if leaf_dtype is not None:
        config = replace(config, leaf_dtype=leaf_dtype)
    settings = HFLODSettings(
        config=config,
        open_count=open_count,
        engine_backend=engine_backend,
        backend_name=backend_name,
        left_padding_mode=left_padding_mode,
    )
    register_hf_lod_attention(backend_name)
    all_attention = list(_causal_attention_modules(model))
    compatible = list(_compatible_attention_modules(model))
    installed = []
    for name, module in compatible:
        module._hf_lod_settings = settings
        module._hf_lod_active_cache_layer = None
        installed.append(name)
    if not installed:
        raise RuntimeError("no compatible causal AttentionInterface modules were found")
    if len(compatible) == len(all_attention):
        implementation: str | dict[str, str] = backend_name
        if submodel_key is not None:
            implementation = {submodel_key: backend_name}
        model.set_attn_implementation(implementation)
    else:
        # Keep the model-level backend and mask construction unchanged for
        # sliding/local layers. Only full-attention modules receive a shallow
        # config copy that dispatches their post-QKV call through LOD.
        for _, module in compatible:
            module.config = copy(module.config)
            module.config._attn_implementation = backend_name
    _install_generation_cache_factory(model)
    return installed


def new_hf_lod_cache(model: nn.Module) -> Any:
    """Construct a fresh model-bound LOD cache for prefill and generation."""
    from .hf_lod_hybrid_cache import maybe_new_hybrid_hf_lod_cache

    hybrid = maybe_new_hybrid_hf_lod_cache(model)
    if hybrid is not None:
        return hybrid
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
