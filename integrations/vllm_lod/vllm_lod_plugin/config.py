"""Environment configuration for the out-of-tree vLLM backend."""

from __future__ import annotations

import math
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


def scheduled_static_leaf_cap(
    total_length: int,
    minimum: int = 16,
    divisor: int = 16,
) -> int:
    """Return ``max(minimum, ceil(sqrt(total_length) / divisor))`` exactly."""
    if total_length < 1:
        raise ValueError("static leaf-cap scheduling requires a positive length")
    if minimum < 1:
        raise ValueError("static leaf-cap scheduling requires a positive minimum")
    if divisor < 1:
        raise ValueError("static leaf-cap scheduling requires a positive divisor")
    # For positive integers T, floor(sqrt(T - 1)) // divisor + 1 is exactly
    # ceil(sqrt(T) / divisor), including the perfect-square boundaries.
    scheduled = math.isqrt(total_length - 1) // divisor + 1
    return max(minimum, scheduled)


@dataclass(frozen=True)
class VLLMLODSettings:
    aug19_compat: bool = False
    levels: int = 2
    chunk_size: int = 256
    local_window: int = 512
    state_growth_factor: float = 16.0
    state_premerge_factor: int = 1
    state_min_size: int = 256
    state_split_max_leaves: int | None = None
    protected_prefix: int = 1
    open_count: int = 8
    prefill_open_count: int | None = None
    kv_bits: int = 0
    key_bits: int | None = None
    value_bits: int | None = None
    quant_group_size: int = 32
    quant_token_group_size: int = 16
    leaf_quant_scale_mode: str = "max"
    leaf_append_quant_scale_mode: str = "max"
    page_summary_scale_mode: str = "l2"
    pool_size: int = 8
    request_capacity: int | None = None
    prefill_mode: str = "direct"
    routing_geometry: str = "raw"
    routing_positive_dot_stats: bool = False
    routing_cutoff_stats_min_state: int = 0
    routing_cutoff_stats_route_count: int = 0
    routing_cutoff_stats_normalization: str = "raw"
    prefix_rollback_tokens: int = 1024
    prefill_local_backend: str = "aiter"
    fused_prefill_route_coarse: bool = True
    fused_prefill_stable_recompute: bool = True
    fused_prefill_external_recompute: bool = True
    prefill_hierarchical_route: bool | None = None
    prefill_coarse_max_grouped_rows: int = 64
    prefill_coarse_direct_gqa: bool | None = None
    prefill_coarse_block_n: int = 32
    prefill_coarse_num_warps: int = 8
    prefill_overlap_coarse_leaf: bool | None = None
    prefill_overlap_local_lod: bool | None = None
    prefill_int8_route_mma: bool = False
    prefill_int8_coarse_mma: bool = True
    prefill_int8_coarse_block_n: int = 64
    prefill_int8_coarse_num_warps: int = 2
    prefill_int8_append_num_warps: int = 4
    prefill_int8_pv_mma: bool | None = None
    prefill_chunk_size: int = 4096
    prefill_local_window: int = 4864
    prefill_state_update_size: int = 4096
    recursive_prefill_all_leaves: bool | None = None
    prefill_static_leaf_aiter: bool = False
    prefill_static_leaf_cap_min: int = 16
    prefill_route_cohort: bool = False
    static_leaf_cap_divisor: int = 16
    static_cohort_never_readmit: bool = False
    leaf_layout: str = "expert"
    leaf_union_query_tile: int = 16
    leaf_block_m: int = 16
    leaf_block_n: int = 32
    leaf_num_warps: int = 2
    leaf_geometry_tuning: bool = True
    leaf_reduce_num_warps: int = 1
    prefill_direct_expert_buckets: bool | None = None
    prefill_int8_leaf_num_warps: int = 2
    leaf_paged_directory: bool = True
    dense_leaf_storage: bool = True
    leaf_seal_capacity: int | None = None
    prefill_leaf_visit_cap: int | None = None
    decode_split_kv: int = 8
    decode_geometry_tuning: bool = True
    decode_centroid_major_hip: bool = False
    decode_hierarchical_route: bool | None = None
    decode_gqa_cooperative: bool = True
    decode_gqa_cooperative_hip: bool = True
    decode_gqa_union: bool = False
    decode_gqa_mass_fraction: float | None = None
    decode_gqa_predicted_mass: bool = False
    decode_gqa_pilot_z: bool = False
    decode_gqa_pilot_z_route_count: int = 8
    decode_gqa_pilot_z_margin: float = 0.25
    decode_gqa_union_hip: bool = False
    decode_gqa_staged_fixed_aiter: bool = False
    decode_gqa_fixed_mask_aiter: bool = False
    decode_gqa_overlap_local_sink: bool = False
    decode_gqa_fixed_mask_block_n: int = 64
    decode_gqa_fixed_mask_segments: int = 128
    decode_gqa_fixed_mask_adaptive_segments: bool = False
    decode_gqa_fixed_mask_reduce_block_d: int = 0
    decode_gqa_fixed_mask_direct_routes: bool = True
    decode_gqa_fixed_mask_scan_num_warps: int = 2
    decode_gqa_fixed_mask_scan_waves_per_eu: int = 2
    decode_gqa_fixed_mask_scan_num_stages: int = 2
    decode_gqa_static_leaf_cap: int | None = None
    decode_gqa_static_leaf_cap_min: int = 16
    decode_gqa_static_leaf_aiter: bool = False
    decode_route_cohort: bool = False
    diagnostic_static_preselected: bool = False
    decode_max_open_leaves: int | None = 1024
    decode_gqa_route_splits: int | None = None
    recursive_materialize_page_scores: bool = False
    recursive_page_score_block_n: int = 16
    recursive_page_score_num_warps: int = 2
    recursive_page_select_block_n: int = 64
    recursive_state_route_backend: str = "auto"

    @property
    def resolved_key_bits(self) -> int:
        return self.kv_bits if self.key_bits is None else self.key_bits

    @property
    def resolved_value_bits(self) -> int:
        return self.kv_bits if self.value_bits is None else self.value_bits

    @classmethod
    def from_environment(cls) -> VLLMLODSettings:
        capacity = _integer("VLLM_LOD_MAX_CONTEXT", 0)
        kv_bits = _integer("VLLM_LOD_KV_BITS", 0)
        # INT4 uses page-wide, four-channel groups and refines each scale by
        # least squares.  This preserves the broadcast scale load while
        # limiting how many channels share the range of one outlier. BF16 and
        # INT8 retain their established layouts unless explicitly overridden.
        int4_storage = kv_bits == 4
        settings = cls(
            aug19_compat=_boolean("VLLM_LOD_AUG19_COMPAT", False),
            levels=_integer("VLLM_LOD_LEVELS", 2),
            chunk_size=_integer("VLLM_LOD_CHUNK_SIZE", 256),
            local_window=_integer("VLLM_LOD_LOCAL_WINDOW", 512),
            state_growth_factor=_floating("VLLM_LOD_STATE_FACTOR", 16.0),
            state_premerge_factor=_integer("VLLM_LOD_STATE_PREMERGE_FACTOR", 1),
            state_min_size=_integer("VLLM_LOD_STATE_MIN", 256),
            state_split_max_leaves=(
                _integer("VLLM_LOD_STATE_SPLIT_MAX_LEAVES", 0) or None
            ),
            protected_prefix=_integer("VLLM_LOD_PROTECTED_PREFIX", 1),
            open_count=_integer("VLLM_LOD_OPEN_COUNT", 8),
            prefill_open_count=(
                _integer("VLLM_LOD_PREFILL_OPEN_COUNT", 0) or None
            ),
            kv_bits=kv_bits,
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
            quant_group_size=_integer(
                "VLLM_LOD_QUANT_GROUP_SIZE", 4 if int4_storage else 32
            ),
            quant_token_group_size=_integer(
                "VLLM_LOD_QUANT_TOKEN_GROUP_SIZE", 16
            ),
            leaf_quant_scale_mode=_choice(
                "VLLM_LOD_LEAF_QUANT_SCALE_MODE",
                "l2" if int4_storage else "max",
                ("max", "l2"),
            ),
            leaf_append_quant_scale_mode=_choice(
                "VLLM_LOD_LEAF_APPEND_QUANT_SCALE_MODE",
                "l2" if int4_storage else "max",
                ("max", "l2"),
            ),
            page_summary_scale_mode=_choice(
                "VLLM_LOD_PAGE_SUMMARY_SCALE_MODE", "l2", ("max", "l2")
            ),
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
            routing_positive_dot_stats=_boolean(
                "VLLM_LOD_ROUTING_POSITIVE_DOT_STATS", False
            ),
            routing_cutoff_stats_min_state=_integer(
                "VLLM_LOD_ROUTING_CUTOFF_STATS_MIN_STATE", 0
            ),
            routing_cutoff_stats_route_count=_integer(
                "VLLM_LOD_ROUTING_CUTOFF_STATS_ROUTE_COUNT", 0
            ),
            routing_cutoff_stats_normalization=_choice(
                "VLLM_LOD_ROUTING_CUTOFF_STATS_NORMALIZATION",
                "raw",
                ("raw", "lse", "pilot64_lse", "pilot64_z"),
            ),
            dense_leaf_storage=_boolean("VLLM_LOD_DENSE_LEAF_STORAGE", True),
            prefix_rollback_tokens=_integer(
                "VLLM_LOD_PREFIX_ROLLBACK_TOKENS", 1024
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
            prefill_hierarchical_route=(
                _boolean("VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE", False)
                if os.getenv("VLLM_LOD_PREFILL_HIERARCHICAL_ROUTE") is not None
                else None
            ),
            prefill_coarse_max_grouped_rows=_integer(
                "VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS", 64
            ),
            prefill_coarse_direct_gqa=(
                _boolean("VLLM_LOD_PREFILL_COARSE_DIRECT_GQA", False)
                if os.getenv("VLLM_LOD_PREFILL_COARSE_DIRECT_GQA") is not None
                else None
            ),
            prefill_coarse_block_n=_integer(
                "VLLM_LOD_PREFILL_COARSE_BLOCK_N", 32
            ),
            prefill_coarse_num_warps=_integer(
                "VLLM_LOD_PREFILL_COARSE_NUM_WARPS", 8
            ),
            prefill_overlap_coarse_leaf=(
                _boolean("VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF", False)
                if os.getenv("VLLM_LOD_PREFILL_OVERLAP_COARSE_LEAF") is not None
                else None
            ),
            prefill_overlap_local_lod=(
                _boolean("VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD", False)
                if os.getenv("VLLM_LOD_PREFILL_OVERLAP_LOCAL_LOD") is not None
                else None
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
            recursive_prefill_all_leaves=(
                _boolean("VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES", False)
                if os.getenv(
                    "VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES"
                ) is not None
                else None
            ),
            prefill_static_leaf_aiter=_boolean(
                "VLLM_LOD_PREFILL_STATIC_LEAF_AITER", False
            ),
            prefill_static_leaf_cap_min=_integer(
                "VLLM_LOD_PREFILL_STATIC_LEAF_CAP_MIN", 16
            ),
            prefill_route_cohort=_boolean(
                "VLLM_LOD_PREFILL_ROUTE_COHORT", False
            ),
            static_leaf_cap_divisor=_integer(
                "VLLM_LOD_STATIC_LEAF_CAP_DIVISOR", 16
            ),
            static_cohort_never_readmit=_boolean(
                "VLLM_LOD_STATIC_COHORT_NEVER_READMIT", False
            ),
            leaf_layout=_choice(
                "VLLM_LOD_LEAF_LAYOUT",
                "expert",
                ("query", "expert", "aiter_union", "aiter_masked_union"),
            ),
            leaf_union_query_tile=_integer(
                "VLLM_LOD_LEAF_UNION_QUERY_TILE", 16
            ),
            leaf_block_m=_integer("VLLM_LOD_LEAF_BLOCK_M", 16),
            leaf_block_n=_integer("VLLM_LOD_LEAF_BLOCK_N", 32),
            leaf_num_warps=_integer("VLLM_LOD_LEAF_NUM_WARPS", 2),
            leaf_geometry_tuning=_boolean(
                "VLLM_LOD_LEAF_GEOMETRY_TUNING", True
            ),
            leaf_reduce_num_warps=_integer(
                "VLLM_LOD_LEAF_REDUCE_NUM_WARPS", 1
            ),
            prefill_direct_expert_buckets=(
                _boolean("VLLM_LOD_PREFILL_DIRECT_EXPERT_BUCKETS", False)
                if os.getenv("VLLM_LOD_PREFILL_DIRECT_EXPERT_BUCKETS") is not None
                else None
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
            prefill_leaf_visit_cap=(
                _integer("VLLM_LOD_PREFILL_LEAF_VISIT_CAP", 0) or None
            ),
            decode_split_kv=_integer("VLLM_LOD_DECODE_SPLIT_KV", 8),
            decode_geometry_tuning=_boolean(
                "VLLM_LOD_DECODE_GEOMETRY_TUNING", True
            ),
            decode_centroid_major_hip=_boolean(
                "VLLM_LOD_DECODE_CENTROID_MAJOR_HIP", False
            ),
            decode_hierarchical_route=(
                _boolean("VLLM_LOD_DECODE_HIERARCHICAL_ROUTE", False)
                if os.getenv("VLLM_LOD_DECODE_HIERARCHICAL_ROUTE") is not None
                else None
            ),
            decode_gqa_cooperative=_boolean(
                "VLLM_LOD_DECODE_GQA_COOPERATIVE", True
            ),
            decode_gqa_cooperative_hip=_boolean(
                "VLLM_LOD_DECODE_GQA_COOPERATIVE_HIP", True
            ),
            decode_gqa_union=_boolean("VLLM_LOD_DECODE_GQA_UNION", False),
            decode_gqa_mass_fraction=(
                _floating("VLLM_LOD_DECODE_GQA_MASS_FRACTION", 0.0) or None
            ),
            decode_gqa_predicted_mass=_boolean(
                "VLLM_LOD_DECODE_GQA_PREDICTED_MASS", False
            ),
            decode_gqa_pilot_z=_boolean(
                "VLLM_LOD_DECODE_GQA_PILOT_Z", False
            ),
            decode_gqa_pilot_z_route_count=_integer(
                "VLLM_LOD_DECODE_GQA_PILOT_Z_ROUTE_COUNT", 8
            ),
            decode_gqa_pilot_z_margin=_floating(
                "VLLM_LOD_DECODE_GQA_PILOT_Z_MARGIN", 0.25
            ),
            decode_gqa_union_hip=_boolean(
                "VLLM_LOD_DECODE_GQA_UNION_HIP", False
            ),
            decode_gqa_staged_fixed_aiter=_boolean(
                "VLLM_LOD_DECODE_GQA_STAGED_FIXED_AITER", False
            ),
            decode_gqa_fixed_mask_aiter=_boolean(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER", False
            ),
            decode_gqa_overlap_local_sink=_boolean(
                "VLLM_LOD_DECODE_GQA_OVERLAP_LOCAL_SINK", False
            ),
            decode_gqa_fixed_mask_block_n=_integer(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N", 64
            ),
            decode_gqa_fixed_mask_segments=_integer(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS", 128
            ),
            decode_gqa_fixed_mask_adaptive_segments=_boolean(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS", False
            ),
            decode_gqa_fixed_mask_reduce_block_d=_integer(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D", 0
            ),
            decode_gqa_fixed_mask_direct_routes=_boolean(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_DIRECT_ROUTES", True
            ),
            decode_gqa_fixed_mask_scan_num_warps=_integer(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SCAN_NUM_WARPS", 2
            ),
            decode_gqa_fixed_mask_scan_waves_per_eu=_integer(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SCAN_WAVES_PER_EU", 2
            ),
            decode_gqa_fixed_mask_scan_num_stages=_integer(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SCAN_NUM_STAGES", 2
            ),
            decode_gqa_static_leaf_cap=(
                _integer("VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP", 0) or None
            ),
            decode_gqa_static_leaf_cap_min=_integer(
                "VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP_MIN", 16
            ),
            decode_gqa_static_leaf_aiter=_boolean(
                "VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER", False
            ),
            decode_route_cohort=_boolean(
                "VLLM_LOD_DECODE_ROUTE_COHORT", False
            ),
            diagnostic_static_preselected=_boolean(
                "VLLM_LOD_DIAGNOSTIC_STATIC_PRESELECTED", False
            ),
            decode_max_open_leaves=(
                _integer("VLLM_LOD_DECODE_MAX_OPEN_LEAVES", 1024) or None
            ),
            decode_gqa_route_splits=(
                _integer("VLLM_LOD_DECODE_GQA_ROUTE_SPLITS", 0) or None
            ),
            recursive_materialize_page_scores=_boolean(
                "VLLM_LOD_MATERIALIZE_PAGE_SCORES", False
            ),
            recursive_page_score_block_n=_integer(
                "VLLM_LOD_PAGE_SCORE_BLOCK_N", 16
            ),
            recursive_page_score_num_warps=_integer(
                "VLLM_LOD_PAGE_SCORE_NUM_WARPS", 2
            ),
            recursive_page_select_block_n=_integer(
                "VLLM_LOD_PAGE_SELECT_BLOCK_N", 64
            ),
            recursive_state_route_backend=os.getenv(
                "VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND", "auto"
            ).strip().lower(),
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
                decode_gqa_union=False,
                decode_gqa_mass_fraction=None,
                decode_gqa_predicted_mass=False,
                decode_gqa_pilot_z=False,
                decode_gqa_union_hip=False,
                decode_gqa_staged_fixed_aiter=False,
                decode_gqa_fixed_mask_aiter=False,
                decode_gqa_overlap_local_sink=False,
                decode_gqa_static_leaf_cap=None,
                decode_gqa_static_leaf_aiter=False,
                diagnostic_static_preselected=False,
                decode_gqa_route_splits=None,
            )
        if settings.levels not in (2, 3):
            raise ValueError("VLLM_LOD_LEVELS must be two or three")
        if settings.recursive_prefill_all_leaves and (
            settings.levels != 3 or settings.kv_bits != 0
        ):
            raise ValueError(
                "VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES requires "
                "three-level BF16 LOD"
            )
        if settings.state_split_max_leaves is not None and settings.levels != 2:
            raise ValueError(
                "VLLM_LOD_STATE_SPLIT_MAX_LEAVES requires two-level LOD"
            )
        if settings.decode_gqa_mass_fraction is not None and not (
            0.0 < settings.decode_gqa_mass_fraction < 1.0
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_MASS_FRACTION must lie in (0, 1)"
            )
        if (
            settings.decode_gqa_mass_fraction is not None
            and not settings.decode_gqa_union
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_MASS_FRACTION requires "
                "VLLM_LOD_DECODE_GQA_UNION=1"
            )
        if settings.decode_gqa_union_hip and not settings.decode_gqa_union:
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_UNION_HIP requires "
                "VLLM_LOD_DECODE_GQA_UNION=1"
            )
        if settings.decode_gqa_staged_fixed_aiter and (
            not settings.decode_gqa_union
            or not settings.decode_gqa_union_hip
            or settings.levels != 2
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_STAGED_FIXED_AITER requires two-level "
                "GQA-union page-size-one HIP decode"
            )
        if settings.decode_gqa_fixed_mask_aiter and (
            not settings.decode_gqa_union
            or not settings.decode_gqa_union_hip
            or settings.levels != 2
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER requires two-level "
                "GQA-union page-size-one HIP decode"
            )
        if (
            settings.decode_gqa_fixed_mask_aiter
            and settings.decode_gqa_staged_fixed_aiter
        ):
            raise ValueError(
                "fixed-mask and staged-fixed AITER decode are mutually exclusive"
            )
        if settings.decode_gqa_overlap_local_sink and (
            not settings.decode_gqa_fixed_mask_aiter
            or settings.decode_gqa_predicted_mass
            or settings.decode_gqa_pilot_z
            or settings.decode_gqa_static_leaf_aiter
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_OVERLAP_LOCAL_SINK currently requires "
                "top-k fixed-mask AITER decode"
            )
        if settings.decode_gqa_fixed_mask_block_n not in (16, 64, 128):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_BLOCK_N must be 16, 64, or 128"
            )
        if settings.decode_gqa_fixed_mask_segments not in (
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SEGMENTS must be a supported "
                "power of two from 8 through 512"
            )
        if (
            settings.decode_gqa_fixed_mask_adaptive_segments
            and not settings.decode_gqa_fixed_mask_aiter
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_ADAPTIVE_SEGMENTS requires "
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1"
            )
        if settings.decode_gqa_fixed_mask_reduce_block_d not in (
            0,
            16,
            32,
            64,
            128,
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D must be 0, "
                "16, 32, 64, or 128"
            )
        if (
            settings.decode_gqa_fixed_mask_reduce_block_d
            and not settings.decode_gqa_fixed_mask_aiter
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_REDUCE_BLOCK_D requires "
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_AITER=1"
            )
        if settings.decode_gqa_fixed_mask_scan_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SCAN_NUM_WARPS must be 1, 2, "
                "4, or 8"
            )
        if settings.decode_gqa_fixed_mask_scan_waves_per_eu not in (1, 2, 4):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SCAN_WAVES_PER_EU must be 1, "
                "2, or 4"
            )
        if settings.decode_gqa_fixed_mask_scan_num_stages not in (1, 2, 3, 4):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_FIXED_MASK_SCAN_NUM_STAGES must be 1, 2, "
                "3, or 4"
            )
        if settings.decode_gqa_static_leaf_cap is not None and (
            settings.decode_gqa_static_leaf_cap < 1
            or not (
                settings.decode_gqa_fixed_mask_aiter
                or settings.decode_gqa_static_leaf_aiter
                or settings.decode_route_cohort
            )
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP must be positive and "
                "requires fixed-mask, compact-static AITER, or cohort routing"
            )
        if settings.decode_gqa_static_leaf_cap_min < 1:
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_STATIC_LEAF_CAP_MIN must be positive"
            )
        if settings.static_leaf_cap_divisor < 1:
            raise ValueError(
                "VLLM_LOD_STATIC_LEAF_CAP_DIVISOR must be positive"
            )
        if settings.decode_route_cohort and settings.levels != 2:
            raise ValueError(
                "VLLM_LOD_DECODE_ROUTE_COHORT requires two-level LOD"
            )
        if settings.decode_gqa_static_leaf_aiter and (
            not settings.decode_gqa_union
            or not settings.decode_gqa_union_hip
            or settings.levels != 2
            or settings.decode_gqa_fixed_mask_aiter
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_STATIC_LEAF_AITER requires two-level "
                "GQA-union page-size-one HIP decode and no fixed mask"
            )
        if (
            settings.diagnostic_static_preselected
            and not settings.decode_gqa_static_leaf_aiter
        ):
            raise ValueError(
                "VLLM_LOD_DIAGNOSTIC_STATIC_PRESELECTED requires compact-static "
                "AITER decode"
            )
        if settings.decode_gqa_predicted_mass and (
            settings.decode_gqa_mass_fraction is None
            or not settings.decode_gqa_union
            or not settings.decode_gqa_union_hip
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_PREDICTED_MASS requires mass-fraction, "
                "GQA-union, and page-size-one HIP decode"
            )
        if settings.decode_gqa_pilot_z and (
            not settings.decode_gqa_union
            or not settings.decode_gqa_union_hip
            or settings.levels != 2
            or settings.decode_gqa_predicted_mass
            or settings.decode_gqa_mass_fraction is not None
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_PILOT_Z requires two-level GQA-union "
                "HIP decode and is mutually exclusive with "
                "mass-fraction routing"
            )
        if not math.isfinite(settings.decode_gqa_pilot_z_margin) or (
            settings.decode_gqa_pilot_z_margin < 0.0
        ):
            raise ValueError("pilot-z routing margin must be finite and nonnegative")
        if not 1 <= settings.decode_gqa_pilot_z_route_count <= 128:
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_PILOT_Z_ROUTE_COUNT must be between "
                "one and 128"
            )
        if settings.kv_bits not in (0, 4, 8):
            raise ValueError("VLLM_LOD_KV_BITS must be zero, four, or eight")
        if settings.resolved_key_bits not in (0, 2, 3, 4, 8):
            raise ValueError(
                "VLLM_LOD_KEY_BITS must be zero, two, three, four, or eight"
            )
        if settings.resolved_value_bits not in (0, 2, 3, 4, 8):
            raise ValueError(
                "VLLM_LOD_VALUE_BITS must be zero, two, three, four, or eight"
            )
        if settings.kv_bits in (4, 8) and (
            settings.resolved_key_bits != settings.kv_bits
            or settings.resolved_value_bits != settings.kv_bits
        ):
            raise ValueError(
                "quantized storage requires matching K and V precision; use "
                "VLLM_LOD_KV_BITS=0 for mixed-precision QDQ analysis"
            )
        if settings.quant_token_group_size not in (1, 2, 4, 8, 16):
            raise ValueError(
                "VLLM_LOD_QUANT_TOKEN_GROUP_SIZE must be one, two, four, "
                "eight, or sixteen"
            )
        if not 1 <= settings.open_count <= 8:
            raise ValueError("VLLM_LOD_OPEN_COUNT must be between one and eight")
        if settings.prefill_open_count is not None and not (
            1 <= settings.prefill_open_count <= 128
        ):
            raise ValueError(
                "VLLM_LOD_PREFILL_OPEN_COUNT must be between one and 128"
            )
        if settings.state_premerge_factor not in {1, 2, 4, 8, 16, 32}:
            raise ValueError(
                "VLLM_LOD_STATE_PREMERGE_FACTOR must be one, two, four, eight, "
                "sixteen, or thirty-two"
            )
        if settings.routing_cutoff_stats_min_state < 0:
            raise ValueError(
                "VLLM_LOD_ROUTING_CUTOFF_STATS_MIN_STATE must be nonnegative"
            )
        if settings.routing_cutoff_stats_route_count < 0:
            raise ValueError(
                "VLLM_LOD_ROUTING_CUTOFF_STATS_ROUTE_COUNT must be nonnegative"
            )
        if settings.pool_size <= 0:
            raise ValueError("VLLM_LOD_POOL_SIZE must be positive")
        if settings.prefix_rollback_tokens <= 0:
            raise ValueError("VLLM_LOD_PREFIX_ROLLBACK_TOKENS must be positive")
        if settings.prefill_coarse_max_grouped_rows <= 0:
            raise ValueError(
                "VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS must be positive"
            )
        if settings.prefill_coarse_direct_gqa and (
            settings.prefill_coarse_max_grouped_rows
            & (settings.prefill_coarse_max_grouped_rows - 1)
        ):
            raise ValueError(
                "direct-GQA prefill requires a power-of-two "
                "VLLM_LOD_PREFILL_COARSE_GROUPED_ROWS"
            )
        if settings.prefill_coarse_block_n not in (16, 32, 64, 128):
            raise ValueError(
                "VLLM_LOD_PREFILL_COARSE_BLOCK_N must be 16, 32, 64, or 128"
            )
        if settings.prefill_coarse_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_PREFILL_COARSE_NUM_WARPS must be 1, 2, 4, or 8"
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
        if settings.prefill_static_leaf_cap_min < 1:
            raise ValueError(
                "VLLM_LOD_PREFILL_STATIC_LEAF_CAP_MIN must be positive"
            )
        if settings.prefill_route_cohort and settings.levels != 2:
            raise ValueError(
                "VLLM_LOD_PREFILL_ROUTE_COHORT requires two-level LOD"
            )
        if settings.prefill_static_leaf_aiter and (
            settings.levels != 2
            or settings.kv_bits != 0
            or not settings.dense_leaf_storage
        ):
            raise ValueError(
                "VLLM_LOD_PREFILL_STATIC_LEAF_AITER requires two-level BF16 "
                "dense leaf storage"
            )
        if settings.leaf_block_m <= 0 or settings.leaf_block_n <= 0:
            raise ValueError("VLLM_LOD leaf block sizes must be positive")
        if settings.leaf_union_query_tile not in (1, 2, 4, 8, 16):
            raise ValueError(
                "VLLM_LOD_LEAF_UNION_QUERY_TILE must be 1, 2, 4, 8, or 16"
            )
        if settings.leaf_layout.startswith("aiter_") and (
            settings.levels != 2
            or settings.kv_bits != 0
            or not settings.dense_leaf_storage
        ):
            raise ValueError(
                "AITER union leaf layouts require two-level BF16 dense leaf "
                "storage"
            )
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
        if (
            settings.prefill_leaf_visit_cap is not None
            and settings.prefill_leaf_visit_cap <= 0
        ):
            raise ValueError(
                "VLLM_LOD_PREFILL_LEAF_VISIT_CAP must be positive"
            )
        if settings.decode_split_kv not in (1, 8, 16, 32):
            raise ValueError("VLLM_LOD_DECODE_SPLIT_KV must be 1, 8, 16, or 32")
        if settings.decode_gqa_route_splits not in (None, 4, 8, 16, 32):
            raise ValueError(
                "VLLM_LOD_DECODE_GQA_ROUTE_SPLITS must be 4, 8, 16, or 32"
            )
        if (
            settings.decode_max_open_leaves is not None
            and settings.decode_max_open_leaves < 1
        ):
            raise ValueError(
                "VLLM_LOD_DECODE_MAX_OPEN_LEAVES must be positive or zero "
                "to disable the routing limit"
            )
        valid_page_blocks = (16, 32, 64, 128)
        if settings.recursive_page_score_block_n not in valid_page_blocks:
            raise ValueError(
                "VLLM_LOD_PAGE_SCORE_BLOCK_N must be 16, 32, 64, or 128"
            )
        if settings.recursive_page_select_block_n not in valid_page_blocks:
            raise ValueError(
                "VLLM_LOD_PAGE_SELECT_BLOCK_N must be 16, 32, 64, or 128"
            )
        if settings.recursive_page_score_num_warps not in (1, 2, 4, 8):
            raise ValueError(
                "VLLM_LOD_PAGE_SCORE_NUM_WARPS must be 1, 2, 4, or 8"
            )
        if settings.recursive_materialize_page_scores and settings.levels != 3:
            raise ValueError(
                "VLLM_LOD_MATERIALIZE_PAGE_SCORES requires VLLM_LOD_LEVELS=3"
            )
        if settings.recursive_state_route_backend not in (
            "auto",
            "fused",
            "resplit",
        ):
            raise ValueError(
                "VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND must be auto, fused, "
                "or resplit"
            )
        if settings.recursive_state_route_backend == "resplit" and settings.levels != 3:
            raise ValueError(
                "VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND=resplit requires "
                "VLLM_LOD_LEVELS=3"
            )
        if settings.levels == 2 and settings.kv_bits not in (0, 8):
            raise ValueError(
                "the two-tier vLLM cache supports BF16 or INT8 K/V storage"
            )
        if settings.prefill_mode != "direct":
            settings = replace(settings, prefill_mode="direct")
        return settings
