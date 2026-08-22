"""Environment configuration for the out-of-tree vLLM backend."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Any


_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def configured_full_attention_layer(name: str, config: Any) -> bool:
    """Return whether ``name`` is globally attending in a hybrid model."""
    text_config = getattr(
        getattr(config, "model_config", None), "hf_text_config", None
    )
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        return True
    match = _LAYER_INDEX.search(name)
    if match is None:
        return True
    layer_index = int(match.group(1))
    if layer_index >= len(layer_types):
        raise ValueError(
            f"attention layer index {layer_index} exceeds "
            f"{len(layer_types)} configured layer types"
        )
    return layer_types[layer_index] == "full_attention"


def _integer(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return value


def _floating(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    return value


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        options = ", ".join(choices)
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


@dataclass(frozen=True)
class VLLMLODSettings:
    chunk_size: int = 256
    local_window: int = 512
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    protected_prefix: int = 1
    open_count: int = 8
    kv_bits: int = 4
    quant_group_size: int = 32
    pool_size: int = 8
    request_capacity: int | None = None
    prefill_mode: str = "direct"
    routing_geometry: str = "auto"
    cache_ownership: str = "lod"
    native_staging_chunk: int = 1024
    native_cache_headroom: float = 1.5
    native_placeholder_cache: bool = True
    prefill_local_backend: str = "aiter"
    fused_prefill_route_coarse: bool = False
    fused_prefill_stable_recompute: bool = True
    fused_prefill_external_recompute: bool = True
    prefill_coarse_max_grouped_rows: int = 64
    decode_state_update_len: int = 256
    gqa_union_aiter: bool = False
    gqa_union_route_then_coarse: bool = False
    gqa_union_persistent_route: bool = False
    gqa_union_fused_correction: bool = False
    gqa_union_own_route_correction: bool = False
    gqa_union_stage1_reduce: bool = False
    gqa_union_group_size: int = 0
    gqa_union_max_slot_leaves: int = 0

    @classmethod
    def from_environment(cls) -> VLLMLODSettings:
        capacity = _integer("VLLM_LOD_MAX_CONTEXT", 0)
        settings = cls(
            chunk_size=_integer("VLLM_LOD_CHUNK_SIZE", 256),
            local_window=_integer("VLLM_LOD_LOCAL_WINDOW", 512),
            state_growth_factor=_floating("VLLM_LOD_STATE_FACTOR", 16.0),
            state_min_size=_integer("VLLM_LOD_STATE_MIN", 256),
            protected_prefix=_integer("VLLM_LOD_PROTECTED_PREFIX", 1),
            open_count=_integer("VLLM_LOD_OPEN_COUNT", 8),
            kv_bits=_integer("VLLM_LOD_KV_BITS", 4),
            quant_group_size=_integer("VLLM_LOD_QUANT_GROUP_SIZE", 32),
            pool_size=_integer("VLLM_LOD_POOL_SIZE", 8),
            request_capacity=capacity or None,
            prefill_mode=_choice(
                "VLLM_LOD_PREFILL_MODE", "rebuild", ("rebuild", "direct")
            ),
            routing_geometry=_choice(
                "VLLM_LOD_ROUTING_GEOMETRY",
                "auto",
                ("auto", "raw", "spherical", "coherence"),
            ),
            cache_ownership=_choice(
                "VLLM_LOD_CACHE_OWNERSHIP", "lod", ("lod", "dual")
            ),
            native_staging_chunk=_integer(
                "VLLM_LOD_NATIVE_STAGING_CHUNK", 1024
            ),
            native_cache_headroom=_floating(
                "VLLM_LOD_NATIVE_CACHE_HEADROOM", 1.5
            ),
            native_placeholder_cache=bool(
                _integer("VLLM_LOD_NATIVE_PLACEHOLDER_CACHE", 1)
            ),
            prefill_local_backend=_choice(
                "VLLM_LOD_PREFILL_LOCAL_BACKEND",
                "aiter",
                ("torch", "aiter"),
            ),
            fused_prefill_route_coarse=bool(
                _integer("VLLM_LOD_FUSED_PREFILL_ROUTE_COARSE", 0)
            ),
            fused_prefill_stable_recompute=bool(
                _integer("VLLM_LOD_FUSED_PREFILL_STABLE_RECOMPUTE", 1)
            ),
            fused_prefill_external_recompute=bool(
                _integer("VLLM_LOD_FUSED_PREFILL_EXTERNAL_RECOMPUTE", 1)
            ),
            prefill_coarse_max_grouped_rows=_integer(
                "VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS", 64
            ),
            decode_state_update_len=_integer(
                "VLLM_LOD_DECODE_STATE_UPDATE_LEN", 256
            ),
            gqa_union_aiter=bool(
                _integer("VLLM_LOD_GQA_UNION_AITER", 0)
            ),
            gqa_union_route_then_coarse=bool(
                _integer("VLLM_LOD_GQA_UNION_ROUTE_THEN_COARSE", 0)
            ),
            gqa_union_persistent_route=bool(
                _integer("VLLM_LOD_GQA_UNION_PERSISTENT_ROUTE", 0)
            ),
            gqa_union_fused_correction=bool(
                _integer("VLLM_LOD_GQA_UNION_FUSED_CORRECTION", 0)
            ),
            gqa_union_own_route_correction=bool(
                _integer("VLLM_LOD_GQA_UNION_OWN_ROUTE_CORRECTION", 0)
            ),
            gqa_union_stage1_reduce=bool(
                _integer("VLLM_LOD_GQA_UNION_STAGE1_REDUCE", 0)
            ),
            gqa_union_group_size=_integer(
                "VLLM_LOD_GQA_UNION_GROUP_SIZE", 0
            ),
            gqa_union_max_slot_leaves=_integer(
                "VLLM_LOD_GQA_MAX_SLOT_LEAVES", 0
            ),
        )
        if settings.kv_bits not in (0, 4):
            raise ValueError("VLLM_LOD_KV_BITS must be zero or four")
        if not 1 <= settings.open_count <= 8:
            raise ValueError("VLLM_LOD_OPEN_COUNT must be between one and eight")
        if settings.pool_size <= 0:
            raise ValueError("VLLM_LOD_POOL_SIZE must be positive")
        if settings.native_staging_chunk <= 0:
            raise ValueError("VLLM_LOD_NATIVE_STAGING_CHUNK must be positive")
        if settings.native_cache_headroom < 1.0:
            raise ValueError("VLLM_LOD_NATIVE_CACHE_HEADROOM must be at least one")
        if settings.prefill_coarse_max_grouped_rows <= 0:
            raise ValueError(
                "VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS must be positive"
            )
        if settings.decode_state_update_len <= 0:
            raise ValueError("VLLM_LOD_DECODE_STATE_UPDATE_LEN must be positive")
        if settings.gqa_union_group_size < 0:
            raise ValueError("VLLM_LOD_GQA_UNION_GROUP_SIZE must be nonnegative")
        if settings.gqa_union_max_slot_leaves < 0:
            raise ValueError("VLLM_LOD_GQA_MAX_SLOT_LEAVES must be nonnegative")
        if (
            settings.gqa_union_own_route_correction
            and not settings.gqa_union_fused_correction
        ):
            raise ValueError(
                "VLLM_LOD_GQA_UNION_OWN_ROUTE_CORRECTION requires "
                "VLLM_LOD_GQA_UNION_FUSED_CORRECTION=1"
            )
        if settings.gqa_union_aiter and settings.kv_bits != 0:
            raise ValueError(
                "VLLM_LOD_GQA_UNION_AITER currently requires "
                "VLLM_LOD_KV_BITS=0"
            )
        if settings.cache_ownership == "lod" and settings.prefill_mode != "direct":
            settings = replace(settings, prefill_mode="direct")
        return settings
