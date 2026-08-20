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


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        options = ", ".join(choices)
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


@dataclass(frozen=True)
class VLLMLODSettings:
    aug19_compat: bool = False
    levels: int = 2
    chunk_size: int = 256
    local_window: int = 512
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    protected_prefix: int = 1
    open_count: int = 8
    kv_bits: int = 0
    key_bits: int | None = None
    value_bits: int | None = None
    quant_group_size: int = 32
    pool_size: int = 8
    request_capacity: int | None = None
    prefill_mode: str = "direct"
    routing_geometry: str = "raw"
    cache_ownership: str = "lod"
    native_staging_chunk: int = 1024
    native_cache_headroom: float = 1.0
    native_placeholder_cache: bool = True
    prefill_local_backend: str = "aiter"
    fused_prefill_route_coarse: bool = True
    fused_prefill_stable_recompute: bool = True
    fused_prefill_external_recompute: bool = True
    prefill_coarse_max_grouped_rows: int = 64
    prefill_int8_route_mma: bool = False
    prefill_int8_coarse_mma: bool = True
    prefill_int8_coarse_block_n: int = 64
    prefill_int8_coarse_num_warps: int = 2
    prefill_int8_append_num_warps: int = 4
    prefill_int8_pv_mma: bool | None = None
    prefill_chunk_size: int = 4096
    prefill_local_window: int = 4864
    prefill_state_update_size: int = 4096
    leaf_layout: str = "expert"
    leaf_block_m: int = 16
    leaf_block_n: int = 32
    leaf_num_warps: int = 2
    leaf_reduce_num_warps: int = 1
    prefill_int8_leaf_num_warps: int = 2
    leaf_paged_directory: bool = True
    dense_leaf_storage: bool = True
    leaf_seal_capacity: int | None = None
    decode_split_kv: int = 8
    decode_gqa_cooperative: bool = True
    decode_gqa_cooperative_hip: bool = True
    decode_gqa_route_splits: int | None = None

    @property
    def resolved_key_bits(self) -> int:
        return self.kv_bits if self.key_bits is None else self.key_bits

    @property
    def resolved_value_bits(self) -> int:
        return self.kv_bits if self.value_bits is None else self.value_bits

    @classmethod
    def from_environment(cls) -> VLLMLODSettings:
        capacity = _integer("VLLM_LOD_MAX_CONTEXT", 0)
        settings = cls(
            aug19_compat=_boolean("VLLM_LOD_AUG19_COMPAT", False),
            levels=_integer("VLLM_LOD_LEVELS", 2),
            chunk_size=_integer("VLLM_LOD_CHUNK_SIZE", 256),
            local_window=_integer("VLLM_LOD_LOCAL_WINDOW", 512),
            state_growth_factor=_floating("VLLM_LOD_STATE_FACTOR", 16.0),
            state_min_size=_integer("VLLM_LOD_STATE_MIN", 256),
            protected_prefix=_integer("VLLM_LOD_PROTECTED_PREFIX", 1),
            open_count=_integer("VLLM_LOD_OPEN_COUNT", 8),
            kv_bits=_integer("VLLM_LOD_KV_BITS", 0),
            key_bits=(
                _integer("VLLM_LOD_KEY_BITS", 0)
                if os.getenv("VLLM_LOD_KEY_BITS") is not None
                else None
            ),
            value_bits=(
                _integer("VLLM_LOD_VALUE_BITS", 0)
                if os.getenv("VLLM_LOD_VALUE_BITS") is not None
                else None
            ),
            quant_group_size=_integer("VLLM_LOD_QUANT_GROUP_SIZE", 32),
            pool_size=_integer("VLLM_LOD_POOL_SIZE", 8),
            request_capacity=capacity or None,
            prefill_mode=_choice(
                "VLLM_LOD_PREFILL_MODE", "rebuild", ("rebuild", "direct")
            ),
            routing_geometry=_choice(
                "VLLM_LOD_ROUTING_GEOMETRY",
                "raw",
                ("auto", "raw", "spherical", "coherence"),
            ),
            dense_leaf_storage=_boolean("VLLM_LOD_DENSE_LEAF_STORAGE", True),
            cache_ownership=_choice(
                "VLLM_LOD_CACHE_OWNERSHIP", "lod", ("lod", "dual")
            ),
            native_staging_chunk=_integer(
                "VLLM_LOD_NATIVE_STAGING_CHUNK", 1024
            ),
            native_cache_headroom=_floating(
                "VLLM_LOD_NATIVE_CACHE_HEADROOM", 1.0
            ),
            native_placeholder_cache=_boolean(
                "VLLM_LOD_NATIVE_PLACEHOLDER_CACHE", True
            ),
            prefill_local_backend=_choice(
                "VLLM_LOD_PREFILL_LOCAL_BACKEND",
                "aiter",
                ("torch", "aiter"),
            ),
            fused_prefill_route_coarse=bool(
                _integer("VLLM_LOD_FUSED_PREFILL_ROUTE_COARSE", 1)
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
            prefill_int8_route_mma=_boolean(
                "VLLM_LOD_PREFILL_INT8_ROUTE_MMA", False
            ),
            prefill_int8_coarse_mma=_boolean(
                "VLLM_LOD_PREFILL_INT8_COARSE_MMA", True
            ),
            prefill_int8_coarse_block_n=_integer(
                "VLLM_LOD_PREFILL_INT8_COARSE_BLOCK_N", 64
            ),
            prefill_int8_coarse_num_warps=_integer(
                "VLLM_LOD_PREFILL_INT8_COARSE_NUM_WARPS", 2
            ),
            prefill_int8_append_num_warps=_integer(
                "VLLM_LOD_PREFILL_INT8_APPEND_NUM_WARPS", 4
            ),
            prefill_int8_pv_mma=(
                _boolean("VLLM_LOD_PREFILL_INT8_PV_MMA", False)
                if os.getenv("VLLM_LOD_PREFILL_INT8_PV_MMA") is not None
                else None
            ),
            prefill_chunk_size=_integer("VLLM_LOD_PREFILL_CHUNK_SIZE", 4096),
            prefill_local_window=_integer("VLLM_LOD_PREFILL_LOCAL_WINDOW", 4864),
            prefill_state_update_size=_integer(
                "VLLM_LOD_PREFILL_STATE_UPDATE_SIZE", 4096
            ),
            leaf_layout=_choice(
                "VLLM_LOD_LEAF_LAYOUT", "expert", ("query", "expert")
            ),
            leaf_block_m=_integer("VLLM_LOD_LEAF_BLOCK_M", 16),
            leaf_block_n=_integer("VLLM_LOD_LEAF_BLOCK_N", 32),
            leaf_num_warps=_integer("VLLM_LOD_LEAF_NUM_WARPS", 2),
            leaf_reduce_num_warps=_integer(
                "VLLM_LOD_LEAF_REDUCE_NUM_WARPS", 1
            ),
            prefill_int8_leaf_num_warps=_integer(
                "VLLM_LOD_PREFILL_INT8_LEAF_NUM_WARPS", 2
            ),
            leaf_paged_directory=_boolean(
                "VLLM_LOD_LEAF_PAGED_DIRECTORY", True
            ),
            leaf_seal_capacity=(
                _integer("VLLM_LOD_LEAF_SEAL_CAPACITY", 0) or None
            ),
            decode_split_kv=_integer("VLLM_LOD_DECODE_SPLIT_KV", 8),
            decode_gqa_cooperative=_boolean(
                "VLLM_LOD_DECODE_GQA_COOPERATIVE", True
            ),
            decode_gqa_cooperative_hip=_boolean(
                "VLLM_LOD_DECODE_GQA_COOPERATIVE_HIP", True
            ),
            decode_gqa_route_splits=(
                _integer("VLLM_LOD_DECODE_GQA_ROUTE_SPLITS", 0) or None
            ),
        )
        if settings.aug19_compat:
            # The August 19 LongBench run predates the cooperative GQA/HIP
            # decode path and the one-warp route reduction.  Its exact dirty
            # source tree was not archived, so this is a best-effort execution
            # compatibility preset rather than a byte-for-byte restoration.
            settings = replace(
                settings,
                leaf_reduce_num_warps=4,
                decode_split_kv=8,
                decode_gqa_cooperative=False,
                decode_gqa_cooperative_hip=False,
                decode_gqa_route_splits=None,
            )
        if settings.levels not in (2, 3):
            raise ValueError("VLLM_LOD_LEVELS must be two or three")
        if settings.kv_bits not in (0, 4, 8):
            raise ValueError("VLLM_LOD_KV_BITS must be zero, four, or eight")
        if settings.resolved_key_bits not in (0, 4, 8):
            raise ValueError("VLLM_LOD_KEY_BITS must be zero, four, or eight")
        if settings.resolved_value_bits not in (0, 4, 8):
            raise ValueError("VLLM_LOD_VALUE_BITS must be zero, four, or eight")
        if settings.kv_bits in (4, 8) and (
            settings.resolved_key_bits != settings.kv_bits
            or settings.resolved_value_bits != settings.kv_bits
        ):
            raise ValueError(
                "quantized storage requires matching K and V precision; use "
                "VLLM_LOD_KV_BITS=0 for mixed-precision QDQ analysis"
            )
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
        if (
            settings.prefill_int8_coarse_block_n <= 0
            or settings.prefill_int8_coarse_block_n % 32
        ):
            raise ValueError(
                "VLLM_LOD_PREFILL_INT8_COARSE_BLOCK_N must be a positive "
                "multiple of 32"
            )
        if settings.prefill_int8_coarse_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_PREFILL_INT8_COARSE_NUM_WARPS must be 1, 2, 4, or 8"
            )
        if settings.prefill_int8_append_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_PREFILL_INT8_APPEND_NUM_WARPS must be 1, 2, 4, or 8"
            )
        if settings.prefill_chunk_size <= 0:
            raise ValueError("VLLM_LOD_PREFILL_CHUNK_SIZE must be positive")
        if settings.prefill_local_window < settings.prefill_chunk_size:
            raise ValueError(
                "VLLM_LOD_PREFILL_LOCAL_WINDOW must contain the prefill chunk"
            )
        if settings.prefill_state_update_size <= 0:
            raise ValueError("VLLM_LOD_PREFILL_STATE_UPDATE_SIZE must be positive")
        if settings.leaf_block_m <= 0 or settings.leaf_block_n <= 0:
            raise ValueError("VLLM_LOD leaf block sizes must be positive")
        if settings.leaf_num_warps not in (1, 2, 4, 8):
            raise ValueError("VLLM_LOD_LEAF_NUM_WARPS must be 1, 2, 4, or 8")
        if settings.leaf_reduce_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_LEAF_REDUCE_NUM_WARPS must be 1, 2, 4, or 8"
            )
        if settings.prefill_int8_leaf_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_PREFILL_INT8_LEAF_NUM_WARPS must be 1, 2, 4, or 8"
            )
        if settings.leaf_seal_capacity is not None and settings.leaf_seal_capacity <= 0:
            raise ValueError("VLLM_LOD_LEAF_SEAL_CAPACITY must be positive")
        if settings.decode_split_kv not in (1, 8, 16, 32):
            raise ValueError("VLLM_LOD_DECODE_SPLIT_KV must be 1, 8, 16, or 32")
        if settings.decode_gqa_route_splits not in (None, 4, 8, 16, 32):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_ROUTE_SPLITS must be 4, 8, 16, or 32"
            )
        if settings.levels == 2 and settings.kv_bits not in (0, 8):
            raise ValueError(
                "the two-tier vLLM cache supports BF16 or INT8 K/V storage"
            )
        if settings.cache_ownership == "lod" and settings.prefill_mode != "direct":
            settings = replace(settings, prefill_mode="direct")
        return settings
