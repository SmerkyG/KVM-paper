"""Environment configuration for the out-of-tree vLLM backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


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
    prefill_local_backend: str = "aiter"
    fused_prefill_route_coarse: bool = False
    fused_prefill_stable_recompute: bool = True
    fused_prefill_external_recompute: bool = True
    prefill_coarse_max_grouped_rows: int = 64

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
        if settings.cache_ownership == "lod" and settings.prefill_mode != "direct":
            settings = replace(settings, prefill_mode="direct")
        return settings
