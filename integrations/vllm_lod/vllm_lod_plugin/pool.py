"""Fixed-address per-layer LOD pools used by captured vLLM decode graphs."""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from model.kernels.paged_leaf_attention import (
    advance_decode_cache_lengths,
    fused_decode_paged_lod_attention,
    materialize_page1_coarse_means,
    materialize_page1_fixed_indices,
    materialize_page1_static_cap_indices,
    new_fused_decode_buffers,
    prepare_speculative_decode_kv,
    rehash_overflow_pages,
    static_cap_page1_decode_attention,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import (
    KernelLODCache,
    KernelRecursivePagedLODAttention,
    KernelTwoLevelLODAttention,
)

from .config import VLLMLODSettings, scheduled_static_leaf_cap


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _power_of_two(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


def _prefill_coarse_direct_gqa_geometry(
    head_dim: int,
    gqa: int,
    kv_heads: int,
) -> tuple[int, int, int] | None:
    """Return a measured direct-GQA coarse geometry, if one exists."""

    return {
        # D, GQA, KV heads: (grouped rows, N, warps)
        (128, 5, 8): (128, 16, 8),  # OLMo-3-32B TP1
        (256, 6, 4): (64, 16, 8),   # Qwen3.8-27B TP1
    }.get((head_dim, gqa, kv_heads))


def _prefill_hierarchical_route_geometry(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
) -> bool:
    """Whether exact two-stage top-k is the measured automatic policy."""

    geometry = (head_dim, gqa, kv_heads)
    return geometry in {
        (128, 16, 2),  # Muse-Glimmer TP1
        (128, 5, 8),   # OLMo-3-32B TP1
        (128, 4, 2),   # Phi-4 TP5
        (256, 6, 4),   # Qwen3.8-27B TP1 (neutral/slightly positive)
        (512, 8, 2),   # Gemma-4-26B-A4B TP1
    } or (levels == 3 and geometry == (256, 4, 2))


def _prefill_overlap_geometry(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
) -> tuple[bool, bool]:
    """Return automatic (coarse/leaf, local/LOD) stream overlap policy."""

    geometry = (head_dim, gqa, kv_heads)
    muse = geometry == (128, 16, 2)
    recursive_qwen = levels == 3 and geometry == (256, 4, 2)
    return levels == 2 and muse, muse or recursive_qwen


def _prefill_direct_expert_bucket_geometry(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
    open_count: int,
) -> bool:
    """Whether measured route density favors histogram/scatter dispatch."""

    return (
        levels == 2
        and open_count == 3
        and (head_dim, gqa, kv_heads) == (256, 4, 2)
    )


def _recursive_prefill_all_leaves_geometry(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
) -> bool:
    """Whether recursive prefill should reuse complete-expert attention."""

    # These geometries can evaluate complete selected experts with regular MMA
    # faster than the query-major one-page residual path.  Phi remains on that
    # consumer for the complete prefill.  Qwen TP1 crosses over later, so its
    # automatic policy is bounded by the total-prompt helper below.  Decode
    # still consumes the recursive page archive normally in both cases.
    return levels == 3 and (head_dim, gqa, kv_heads) in (
        (128, 4, 2),
        (256, 6, 4),
    )


def _recursive_prefill_all_leaves_token_limit(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
) -> int:
    """Largest request that should use complete-expert prefill.

    Zero means that the measured geometry has no request-length cutoff.  With
    the 16*sqrt(T) schedule, average posting length is sqrt(T)/16.  It first
    reaches one 16-token residual page at T=64K, when the state has 4096
    entries.  Past that point page selection can reduce average leaf work.
    """

    if levels == 3 and (head_dim, gqa, kv_heads) == (256, 6, 4):
        return 65536
    return 0


def _recursive_state_route_backend(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
    request_capacity: int,
    requested: str,
) -> str:
    """Resolve the measured recursive coarse-routing implementation."""

    if requested != "auto":
        return requested
    # Re-split has a nearly fixed launch floor, whereas the grouped producer
    # grows with the allocated state field. Keep the measured batch-eight
    # crossover in request-token units for each validated geometry. Muse stays
    # grouped through the largest measured 128K capacity. OLMo and Gemma also
    # stay grouped: re-split was faster at 64K, but each uniquely missed a
    # matched NIAH-S3 case that grouped routing passed. Unmeasured shapes and
    # flat two-tier LOD retain the conservative grouped producer.
    crossover = {
        (128, 4, 2): 0,        # Phi-4 TP5 (re-split also wins at 8K)
        (256, 4, 2): 65_536,   # Qwen3.5-0.8B TP1
        (256, 6, 4): 22_528,   # Qwen3.8-27B TP1
    }.get((head_dim, gqa, kv_heads))
    if levels == 3 and crossover is not None and request_capacity >= crossover:
        return "resplit"
    return "fused"


class VLLMLayerLODPool:
    """One layer's stable request rows and graph-captured decode scratch."""

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        settings: VLLMLODSettings,
        max_requests: int,
        request_capacity: int,
        active_indices: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        has_query_norm: bool = False,
        has_key_norm: bool = False,
        prefix_rollback_tokens: int = 0,
    ) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("vLLM LOD conversion requires a native FP16/BF16 KV cache")
        if request_capacity < settings.local_window:
            raise ValueError("LOD request capacity is shorter than its local window")
        self.layer = layer
        self.settings = settings
        self.max_requests = max_requests
        self.request_capacity = request_capacity
        self.active_indices = active_indices
        self.dtype = dtype
        self.device = device
        self.query_heads = int(layer.num_heads)
        self.kv_heads = int(layer.num_kv_heads)
        self.head_dim = int(layer.head_size)
        self.value_dim = int(layer.head_size_v)
        gqa = self.query_heads // self.kv_heads
        if self.value_dim != self.head_dim:
            raise NotImplementedError(
                "LOD vLLM currently requires equal K and V widths"
            )

        geometry = settings.routing_geometry
        if geometry == "auto":
            geometry = "coherence" if has_key_norm else "spherical"
        state_normalization = "cosine" if geometry == "spherical" else "none"
        centroid_rescale = "coherence" if geometry == "coherence" else "none"
        routing_normalization = (
            "none"
            if geometry == "raw" or has_query_norm
            else "query"
        )
        local_window = settings.local_window
        if prefix_rollback_tokens:
            # Keep the scheduler's usual late-prefix rollback in the exact
            # field. This does not enlarge the two-level pool when the prefill
            # local allocation is already wider; it only avoids rebuilding
            # centroids for the common repeated-prompt hit. Older shared
            # prefixes remain supported by restore_prefix() below.
            local_window = max(local_window, prefix_rollback_tokens)
        config_type = PagedLODConfig if settings.levels == 3 else LODConfig
        recursive_state_route_backend = _recursive_state_route_backend(
            settings.levels,
            self.head_dim,
            gqa,
            self.kv_heads,
            request_capacity,
            settings.recursive_state_route_backend,
        )
        config_kwargs = dict(
            chunk_size=settings.chunk_size,
            local_window=local_window,
            state_growth_factor=settings.state_growth_factor,
            state_premerge_factor=settings.state_premerge_factor,
            state_min_size=settings.state_min_size,
            state_split_max_leaves=settings.state_split_max_leaves,
            protected_prefix=settings.protected_prefix,
            max_routes=max(
                settings.open_count,
                settings.prefill_open_count or 0,
                (
                    settings.decode_gqa_pilot_z_route_count
                    if settings.decode_gqa_pilot_z
                    else 0
                ),
                8,
            ),
            leaf_dtype=self.dtype,
            state_clustering_normalization=state_normalization,
            state_clustering_centroid_rescale=centroid_rescale,
            state_clustering_centroid_rescale_scope="assignment",
            routing_normalization=routing_normalization,
            leaf_paged_directory=settings.leaf_paged_directory,
            leaf_seal_capacity=(
                settings.leaf_seal_capacity if settings.levels == 2 else None
            ),
        )
        if settings.levels == 3:
            config_kwargs.update(
                page_size=16,
                kv_bits=settings.kv_bits,
                quant_group_size=settings.quant_group_size,
                page_summary_quant_bits=(
                    0 if settings.recursive_materialize_page_scores else 8
                ),
                recursive_materialize_page_scores=(
                    settings.recursive_materialize_page_scores
                ),
                recursive_page_score_block_n=(
                    settings.recursive_page_score_block_n
                ),
                recursive_page_score_num_warps=(
                    settings.recursive_page_score_num_warps
                ),
                recursive_page_select_block_n=(
                    settings.recursive_page_select_block_n
                ),
                recursive_state_route_backend=(
                    recursive_state_route_backend
                ),
                # The compatibility pool uses its fixed graph-safe overflow
                # hash rather than the flat two-tier directory allocation.
                leaf_paged_directory=False,
            )
        config = config_type(**config_kwargs)
        engine_type = (
            KernelTwoLevelLODAttention
            if settings.levels == 2
            else KernelRecursivePagedLODAttention
        )
        self.engine = engine_type(
            config,
            query_heads=self.query_heads,
            key_value_heads=self.kv_heads,
            scale=float(layer.impl.scale),
            default_open_count=settings.open_count,
        )
        self.engine.routing_positive_dot_stats = (
            settings.routing_positive_dot_stats
        )
        self.engine.routing_cutoff_stats_min_state = (
            settings.routing_cutoff_stats_min_state
        )
        self.engine.routing_cutoff_stats_route_count = (
            settings.routing_cutoff_stats_route_count
        )
        self.engine.routing_cutoff_stats_normalization = (
            settings.routing_cutoff_stats_normalization
        )
        # Decode auto-dispatch is based on the production fixed-list
        # page-size-one path, not the slower portable flat leaf path. Muse was
        # already using the segmented schedule in the historical fast
        # baseline. Qwen and Gemma regress, while Phi's 0.32% delta is
        # noise-scale, on the matched fixed-mask AITER comparison. OLMo's
        # nominal tuned arm remains a one-tile grouped producer, so it is not
        # evidence for hierarchical routing. An explicit override still
        # enables the diagnostic schedule for any geometry below.
        automatic_hierarchical_decode = bool(
            settings.decode_geometry_tuning
            and (self.head_dim, gqa, self.kv_heads)
            in {
                (128, 16, 2),  # Muse-Glimmer TP1 (existing fast baseline)
            }
        )
        hierarchical_decode_route = (
            automatic_hierarchical_decode
            if settings.decode_hierarchical_route is None
            else settings.decode_hierarchical_route
        )
        if hierarchical_decode_route and request_capacity >= 32_768:
            # Keep several native state tiles behind each independent program,
            # emit only its local top eight, then reduce the much shorter
            # candidate/online-softmax field in parallel.  The schedules below
            # are selected by attention geometry rather than model name and
            # preserve the exact route set on both equal- and variable-count
            # controls at the production 64K/B8 state size.
            route_geometry = {
                # D, GQA, KV heads: (N, tiles per segment)
                (128, 16, 2): (64, 2),
                (128, 5, 8): (64, 1),
                (128, 4, 2): (64, 2),
                (256, 6, 4): (32, 2),
                (512, 8, 2): (32, 4),
            }.get((self.head_dim, gqa, self.kv_heads))
            if route_geometry is not None:
                (
                    self.engine.decode_route_group_size,
                    self.engine.decode_route_segment_tiles,
                ) = route_geometry
                self.engine.decode_route_num_warps = 2
                self.engine.decode_route_reduce_num_warps = 2
                self.engine.decode_route_parallel_reduce = True
                # Preserve the established BF16-rounded centroid means.
                # Post-MFMA normalization is faster for some geometries but
                # changes near-tied routes, so it remains a separate option.
                self.engine.decode_route_post_dot_normalize = False
                self.engine.decode_route_post_pv_normalize = False
        flat_int8 = settings.levels == 2 and settings.kv_bits == 8
        self.engine.leaf_key_quant_bits = (
            0 if flat_int8 else settings.resolved_key_bits
        )
        self.engine.leaf_value_quant_bits = (
            0 if flat_int8 else settings.resolved_value_bits
        )
        self.engine.leaf_quant_token_group_size = settings.quant_token_group_size
        self.engine.leaf_quant_scale_mode = settings.leaf_quant_scale_mode
        self.engine.leaf_append_quant_scale_mode = (
            settings.leaf_append_quant_scale_mode
        )
        self.engine.page_summary_scale_mode = settings.page_summary_scale_mode
        self.engine.prefill_int8_leaf_mma = flat_int8
        int8_pv_mma = settings.prefill_int8_pv_mma
        if int8_pv_mma is None:
            # Probability requantization has a fixed per-tile cost. It loses
            # at 32K but is amortized by longer posting-list scans at 64K.
            int8_pv_mma = request_capacity >= 65_536
        self.engine.prefill_int8_pv_mma = flat_int8 and int8_pv_mma
        self.engine.prefill_int8_coarse_mma = (
            flat_int8 and settings.prefill_int8_coarse_mma
        )
        self.engine.prefill_int8_coarse_block_n = (
            settings.prefill_int8_coarse_block_n
        )
        self.engine.prefill_int8_coarse_num_warps = (
            settings.prefill_int8_coarse_num_warps
        )
        self.engine.prefill_int8_append_num_warps = (
            settings.prefill_int8_append_num_warps
        )
        self.engine.prefill_int8_route_mma = (
            flat_int8 and settings.prefill_int8_route_mma
        )
        self.engine.simulate_leaf_quantization = (
            settings.kv_bits == 0
            and bool(settings.resolved_key_bits or settings.resolved_value_bits)
        )
        self.engine.prefill_local_attention_backend = settings.prefill_local_backend
        self.engine.static_cohort_never_readmit = (
            settings.static_cohort_never_readmit
        )
        # Fold the native GQA ratio directly into the MFMA M dimension when
        # that avoids substantial padded rows.  The selected geometries are
        # production B8/64K wins; power-of-two GQA4/GQA16 already fill the
        # former 64-row layout and do not benefit from this alternative.
        direct_gqa_geometry = _prefill_coarse_direct_gqa_geometry(
            self.head_dim, gqa, self.kv_heads
        )
        if settings.prefill_coarse_direct_gqa is None:
            direct_gqa = direct_gqa_geometry is not None
        else:
            direct_gqa = settings.prefill_coarse_direct_gqa
        if direct_gqa and settings.prefill_coarse_direct_gqa is None:
            (
                coarse_grouped_rows,
                coarse_block_n,
                coarse_num_warps,
            ) = direct_gqa_geometry
        else:
            coarse_grouped_rows = settings.prefill_coarse_max_grouped_rows
            coarse_block_n = settings.prefill_coarse_block_n
            coarse_num_warps = settings.prefill_coarse_num_warps
        self.engine.prefill_coarse_direct_gqa = direct_gqa
        self.engine.prefill_coarse_max_grouped_rows = coarse_grouped_rows
        self.engine.prefill_coarse_route_block_n = coarse_block_n
        self.engine.prefill_coarse_route_num_warps = coarse_num_warps
        # Scheduler chunks are already presented as one contiguous prefill
        # field.  Use the serving-wide update batch for both flat and
        # recursive LOD; the recursive engine's old 5 * chunk_size default
        # needlessly split each 4K archive step into four small state-update
        # launches (1280 + 1280 + 1280 + 256 for the standard 256-token
        # decode chunk).  The target state schedule and exact-local window are
        # unchanged; assignment batching can change centroid/page membership,
        # so this schedule is covered by the matched ProLong quality artifact.
        self.engine.prefill_state_update_len = settings.prefill_state_update_size
        recursive_prefill_all_leaves_auto = (
            settings.recursive_prefill_all_leaves is None
        )
        recursive_prefill_all_leaves = (
            (
                settings.kv_bits == 0
                and _recursive_prefill_all_leaves_geometry(
                    settings.levels, self.head_dim, gqa, self.kv_heads
                )
            )
            if settings.recursive_prefill_all_leaves is None
            else settings.recursive_prefill_all_leaves
        )
        self.engine.recursive_prefill_all_leaves = recursive_prefill_all_leaves
        self.engine.recursive_prefill_all_leaves_token_limit = (
            _recursive_prefill_all_leaves_token_limit(
                settings.levels, self.head_dim, gqa, self.kv_heads
            )
            if (
                recursive_prefill_all_leaves_auto
                and recursive_prefill_all_leaves
            )
            else 0
        )
        direct_expert_buckets = (
            (
                settings.kv_bits == 0
                and settings.leaf_layout == "expert"
                and _prefill_direct_expert_bucket_geometry(
                    settings.levels,
                    self.head_dim,
                    gqa,
                    self.kv_heads,
                    (
                        settings.prefill_open_count
                        if settings.prefill_open_count is not None
                        else min(3, settings.open_count)
                    ),
                )
            )
            if settings.prefill_direct_expert_buckets is None
            else settings.prefill_direct_expert_buckets
        )
        if direct_expert_buckets and settings.kv_bits != 0:
            raise ValueError("direct prefill expert buckets require BF16 LOD leaves")
        if direct_expert_buckets and settings.leaf_layout != "expert":
            raise ValueError("direct prefill expert buckets require expert leaf layout")
        self.engine.prefill_direct_expert_buckets = direct_expert_buckets
        if settings.levels == 3 and recursive_prefill_all_leaves:
            # Reuse the same measured expert geometry as flat two-tier
            # prefill. The recursive branch still owns and updates the page
            # directory; only its prefill exact-attention consumer changes.
            self.engine.leaf_layout = settings.leaf_layout
            self.engine.leaf_union_query_tile = settings.leaf_union_query_tile
            self.engine.leaf_block_m = settings.leaf_block_m
            self.engine.leaf_block_n = settings.leaf_block_n
            self.engine.leaf_num_warps = settings.leaf_num_warps
            self.engine.leaf_geometry_tuning = settings.leaf_geometry_tuning
            self.engine.leaf_reduce_num_warps = settings.leaf_reduce_num_warps
        if settings.levels == 2:
            self.engine.virtual_page_storage = settings.dense_leaf_storage
            self.engine.prefill_chunk_len = settings.prefill_chunk_size
            self.engine.prefill_local_len = settings.prefill_local_window
            self.engine.prefill_static_leaf_aiter = (
                settings.prefill_static_leaf_aiter
            )
            self.engine.prefill_static_leaf_cap_min = (
                settings.prefill_static_leaf_cap_min
            )
            self.engine.prefill_route_leaf_cap_min = (
                settings.prefill_static_leaf_cap_min
                if settings.prefill_route_cohort
                else None
            )
            self.engine.static_leaf_cap_divisor = (
                settings.static_leaf_cap_divisor
            )
            self.engine.prefill_two_level_topk = (
                settings.prefill_open_count
                if settings.prefill_open_count is not None
                else min(3, settings.open_count)
            )
            self.engine.split_prefill_local_attention = True
            self.engine.leaf_layout = settings.leaf_layout
            self.engine.leaf_union_query_tile = settings.leaf_union_query_tile
            self.engine.leaf_block_m = settings.leaf_block_m
            self.engine.leaf_block_n = settings.leaf_block_n
            self.engine.leaf_num_warps = (
                settings.prefill_int8_leaf_num_warps
                if flat_int8
                else settings.leaf_num_warps
            )
            self.engine.leaf_geometry_tuning = settings.leaf_geometry_tuning
            self.engine.leaf_reduce_num_warps = settings.leaf_reduce_num_warps
            self.engine.leaf_paged_directory = settings.leaf_paged_directory
            self.engine.leaf_seal_capacity = settings.leaf_seal_capacity
            self.engine.prefill_leaf_visit_cap = settings.prefill_leaf_visit_cap
            self.engine.decode_split_kv = settings.decode_split_kv
            if (
                settings.decode_gqa_union
                and settings.decode_geometry_tuning
                and request_capacity >= 32_768
                and self.head_dim == 128
                and self.query_heads == self.kv_heads * 16
            ):
                # The shared-union final scan has enough work to benefit from
                # twice the ordinary decode parallelism, but 32 splits adds
                # reduction overhead. Batch-8 Muse geometry is fastest at 16.
                self.engine.decode_split_kv = 16
            self.engine.decode_geometry_tuning = settings.decode_geometry_tuning
            self.engine.decode_centroid_major_hip = (
                settings.decode_centroid_major_hip
            )
            if settings.decode_geometry_tuning and self.head_dim == 512:
                self.engine.decode_route_use_dot = False
            self.engine.decode_gqa_cooperative_leaf = (
                settings.decode_gqa_cooperative
            )
            self.engine.decode_gqa_cooperative_hip = (
                settings.decode_gqa_cooperative_hip
            )
            self.engine.decode_gqa_union_mass_fraction = (
                settings.decode_gqa_mass_fraction
            )
            self.engine.decode_gqa_union_predicted_mass = (
                settings.decode_gqa_predicted_mass
            )
            self.engine.decode_gqa_union_pilot_z = settings.decode_gqa_pilot_z
            self.engine.decode_gqa_union_pilot_z_route_count = (
                settings.decode_gqa_pilot_z_route_count
            )
            self.engine.decode_gqa_union_pilot_z_margin = (
                settings.decode_gqa_pilot_z_margin
            )
            self.engine.decode_gqa_cooperative_route_splits = (
                settings.decode_gqa_route_splits
            )
        self.engine.fused_prefill_route_coarse = (
            settings.fused_prefill_route_coarse
        )
        self.engine.fused_prefill_stable_recompute = (
            settings.fused_prefill_stable_recompute
        )
        self.engine.fused_prefill_external_recompute = (
            settings.fused_prefill_external_recompute
        )
        # Wide tile-local top-three followed by a tiny global reduction is an
        # exact selector replacement, but the extra launch only pays for the
        # geometries where the former selector is a meaningful end-to-end
        # fraction. The explicit environment setting remains a force-on/off
        # override for diagnostics and new architectures.
        hierarchical_prefill_geometry = _prefill_hierarchical_route_geometry(
            settings.levels, self.head_dim, gqa, self.kv_heads
        )
        self.engine.prefill_hierarchical_route = (
            hierarchical_prefill_geometry
            if settings.prefill_hierarchical_route is None
            else settings.prefill_hierarchical_route
        )
        if (
            settings.fused_prefill_route_coarse
            and settings.fused_prefill_stable_recompute
            and not settings.fused_prefill_external_recompute
        ):
            # A 128-row value accumulator spills on the Qwen 3.5 GQA=8
            # geometry. Keep the single-kernel stable path at 64 rows.
            self.engine.fused_prefill_block_m = 8
        if (
            settings.fused_prefill_route_coarse
            and settings.fused_prefill_stable_recompute
            and settings.fused_prefill_external_recompute
        ):
            coarse_leaf_overlap, local_lod_overlap = _prefill_overlap_geometry(
                settings.levels, self.head_dim, gqa, self.kv_heads
            )
            if not settings.prefill_static_leaf_aiter and local_lod_overlap:
                # Muse's routed GQA-16 coarse, exact-leaf, and exact-local
                # branches are independent after route selection. Queue them
                # on three streams and synchronize only at the final LSE
                # merge. Static-cohort prefill keeps a much larger exact-leaf
                # working set live, so overlapping its materialized coarse
                # logits exhausts the normal transient-memory allowance.
                # The recursive page branch is one dependent LOD operation,
                # so only its independent local branch can overlap. Flat LOD
                # additionally overlaps coarse and exact-leaf attention.
                self.engine.prefill_overlap_coarse_leaf = coarse_leaf_overlap
                self.engine.prefill_overlap_local_lod = True
        if settings.prefill_overlap_coarse_leaf is not None:
            self.engine.prefill_overlap_coarse_leaf = (
                settings.prefill_overlap_coarse_leaf
            )
        if settings.prefill_overlap_local_lod is not None:
            self.engine.prefill_overlap_local_lod = (
                settings.prefill_overlap_local_lod
            )
        # The current flat two-tier path keeps the protected token in state,
        # exactly matching the HF benchmark. Retain the older recursive side
        # cache for compatibility with VLLM_LOD_LEVELS=3.
        self.engine.separate_sink_cache = settings.levels == 3
        self.state_capacity = self.engine._state_capacity(
            request_capacity, min(request_capacity, settings.chunk_size)
        )
        self.decode_local_capacity = (
            local_window + int(self.engine.decode_state_update_len)
        )
        self.local_capacity = max(
            self.decode_local_capacity,
            int(self.engine.prefill_local_len),
        )
        self.leaf_capacity = _round_up(request_capacity, settings.chunk_size) + max(
            settings.chunk_size, int(self.engine.decode_cache_headroom)
        )
        self.page_capacity = math.ceil(self.leaf_capacity / 16) + self.state_capacity
        self.hash_capacity = _power_of_two(
            self.page_capacity * int(self.engine.leaf_overflow_hash_factor)
        )
        self.state = self._allocate_state()
        self.local_lens = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )
        self.ready = [False] * max_requests
        self.clean = [True] * max_requests
        self.metadata = [dict[str, int | bool]() for _ in range(max_requests)]
        self.decode_buffer_storage: dict[str, torch.Tensor] | None = None
        self.decode_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self.speculative_decode_buffers: dict[tuple[int, int], dict[str, Any]] = {}
        self.decode_enabled = False
        self.speculative_decode_steps = 0
        self.hybrid_full_decode = False
        self.direct_prefill_plan: tuple[tuple[int, int, int, int], ...] | None = None
        self.direct_prefill_prompt_lengths: dict[int, int] = {}
        self.install_count = 0
        self.batched_install_calls = 0
        self.direct_prefill_calls = 0
        self.batched_cached_prefill_calls = 0
        self.batched_cached_prefill_rows = 0
        self.cached_prefill_packed_calls = 0
        self.cached_prefill_nonpacked_calls = 0
        self.cached_prefill_candidate_calls = 0
        self.cached_prefill_candidate_rows = 0
        self.cached_prefill_nonuniform_lengths = 0
        self.cached_prefill_nonuniform_previous = 0
        self.cached_prefill_unready = 0
        self.cached_prefill_noncontiguous = 0
        self.decode_calls = 0
        self.catch_up_batches = 0
        self.catch_up_rows = 0
        self.retained_reuse_count = 0
        self.retained_restore_attempts = 0
        self.retained_restore_fail_no_row = 0
        self.retained_restore_fail_short = 0
        self.retained_restore_fail_tokens = 0
        self.retained_restore_fail_coverage = 0
        self.retained_restore_rebuilds = 0
        self.retained_restore_rebuild_tokens = 0
        self.retained_restore_last_prefix = 0
        self.retained_restore_last_coverage = 0
        self.retained_restore_last_total = 0
        self.split_state_len_max = 0
        self.split_scheduled_state_len_max = 0
        self.split_posting_len_max = 0

    def _record_split_state(
        self,
        state: dict[str, object],
        page: dict[str, object],
    ) -> None:
        """Record cheap experiment invariants before request rows are reset."""

        if self.settings.state_split_max_leaves is None:
            return
        actual = int(state["state_len"])
        scheduled = int(state.get("scheduled_state_len", actual))
        lengths = page.get("slot_lengths")
        if not isinstance(lengths, torch.Tensor):
            raise RuntimeError("split-state posting lengths are missing")
        posting_max = int(lengths[..., :actual].max().item())
        if posting_max > int(self.settings.state_split_max_leaves):
            raise AssertionError("split-state posting limit was violated")
        self.split_state_len_max = max(self.split_state_len_max, actual)
        self.split_scheduled_state_len_max = max(
            self.split_scheduled_state_len_max, scheduled
        )
        self.split_posting_len_max = max(self.split_posting_len_max, posting_max)

    def _allocate_state(self) -> dict[str, object]:
        r, h, s, d = (
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
            self.head_dim,
        )
        unified_page1 = bool(
            self.settings.levels == 2
            and self.settings.dense_leaf_storage
            and self.settings.kv_bits == 0
            and self.settings.decode_gqa_union
            and self.settings.decode_gqa_union_hip
            and self.dtype == torch.bfloat16
            and 1 < self.query_heads // self.kv_heads <= 16
            and self.query_heads % self.kv_heads == 0
            and self.head_dim in (128, 256, 512)
        )
        sink_capacity = (
            self.settings.protected_prefix
            if self.engine.separate_sink_cache
            else 0
        )
        if unified_page1:
            arena_leaf_offset = 0
            kv_rows = r * h
            arena_local_offset = arena_leaf_offset + kv_rows * self.leaf_capacity
            arena_sink_offset = arena_local_offset + kv_rows * self.local_capacity
            arena_coarse_offset = arena_sink_offset + kv_rows * sink_capacity
            arena_capacity = arena_coarse_offset + kv_rows * self.state_capacity
            unified_page1_k = torch.empty(
                arena_capacity, d, dtype=self.dtype, device=self.device
            )
            unified_page1_v = torch.empty_like(unified_page1_k)
            # Leaves, local tokens, and sinks have no multiplicity bias. Coarse
            # refreshes overwrite only the centroid section with log(count).
            unified_page1_bias = torch.zeros(
                arena_capacity, dtype=torch.float16, device=self.device
            )
            recent_k = unified_page1_k[
                arena_local_offset : arena_local_offset + kv_rows * self.local_capacity
            ].view(r, h, self.local_capacity, d)
            recent_v = unified_page1_v[
                arena_local_offset : arena_local_offset + kv_rows * self.local_capacity
            ].view(r, h, self.local_capacity, d)
            fixed_mask_page1 = bool(
                self.settings.decode_gqa_fixed_mask_aiter
            )
            static_cap_page1 = bool(
                self.settings.decode_gqa_static_leaf_aiter
            )
            if fixed_mask_page1 or static_cap_page1:
                fixed_capacity = (
                    self.leaf_capacity
                    + int(self.engine.local_len)
                    + sink_capacity
                    + self.state_capacity
                )
                # Graph capture exercises decode before a real prefill has
                # materialized the persistent list. Keep its one bootstrap
                # entry in-bounds; the first state refresh overwrites it.
                unified_page1_fixed_indices = torch.zeros(
                    r,
                    h,
                    fixed_capacity,
                    dtype=torch.int32,
                    device=self.device,
                )
                unified_page1_fixed_leaf_owners = (
                    torch.empty(
                        r,
                        h,
                        self.leaf_capacity,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    if fixed_mask_page1
                    else None
                )
                unified_page1_fixed_slot_offsets = torch.empty(
                    r,
                    h,
                    self.state_capacity + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                unified_page1_fixed_lengths = torch.zeros(
                    r, h, dtype=torch.int32, device=self.device
                )
            else:
                unified_page1_fixed_indices = None
                unified_page1_fixed_leaf_owners = None
                unified_page1_fixed_slot_offsets = None
                unified_page1_fixed_lengths = None
        else:
            arena_leaf_offset = 0
            arena_local_offset = 0
            arena_sink_offset = 0
            arena_coarse_offset = 0
            arena_capacity = 0
            unified_page1_k = None
            unified_page1_v = None
            unified_page1_bias = None
            unified_page1_fixed_indices = None
            unified_page1_fixed_leaf_owners = None
            unified_page1_fixed_slot_offsets = None
            unified_page1_fixed_lengths = None
            recent_k = torch.empty(
                r, h, self.local_capacity, d, dtype=self.dtype, device=self.device
            )
            recent_v = torch.empty_like(recent_k)
        state: dict[str, object] = {
            "state_k": torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device),
            "state_v": torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device),
            "counts": torch.zeros(r, h, s, 1, dtype=torch.float32, device=self.device),
            "state_len": s,
            "coverage": 0,
            "state_capacity": s,
            "recent_k": recent_k,
            "recent_v": recent_v,
            "recent_len": 0,
            "total_len": 0,
        }
        if self.engine.separate_sink_cache and self.settings.protected_prefix:
            if unified_page1:
                state["sink_k"] = unified_page1_k[
                    arena_sink_offset : arena_sink_offset + r * h * sink_capacity
                ].view(r, h, sink_capacity, d)
                state["sink_v"] = unified_page1_v[
                    arena_sink_offset : arena_sink_offset + r * h * sink_capacity
                ].view(r, h, sink_capacity, d)
            else:
                state["sink_k"] = torch.empty(
                    r,
                    h,
                    self.settings.protected_prefix,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                )
                state["sink_v"] = torch.empty_like(state["sink_k"])
        if self.engine.state_clustering_centroid_rescale != "none":
            state["key_norm_sums"] = torch.zeros(
                r, h, s, 1, dtype=torch.float32, device=self.device
            )

        if self.settings.levels == 2:
            page_size = 16
            int8_storage = self.settings.kv_bits == 8
            maximum_slot_pages = max(
                1, math.ceil(self.leaf_capacity / page_size)
            )
            root_capacity = max(1, math.ceil(maximum_slot_pages / 64))
            if self.settings.leaf_paged_directory:
                slot_pages = torch.full(
                    (r, h, s, root_capacity),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                overflow_page_keys = torch.full(
                    (r, h, 1), -1, dtype=torch.int32, device=self.device
                )
                overflow_page_values = torch.full(
                    (r, h, self.page_capacity, 64),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                overflow_active = False
                overflow_safe_until = root_capacity * 64 * page_size
            else:
                slot_dtype = (
                    torch.int16
                    if self.page_capacity <= torch.iinfo(torch.int16).max
                    else torch.int32
                )
                slot_pages = torch.full(
                    (r, h, s, int(self.engine.leaf_inline_pages_per_slot)),
                    -1,
                    dtype=slot_dtype,
                    device=self.device,
                )
                overflow_page_keys = torch.full(
                    (r, h, self.hash_capacity),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                overflow_page_values = torch.full_like(overflow_page_keys, -1)
                overflow_active = True
                overflow_safe_until = 0
            if self.settings.dense_leaf_storage:
                if unified_page1:
                    leaf_k = unified_page1_k[
                        arena_leaf_offset : arena_leaf_offset + r * h * self.leaf_capacity
                    ].view(r, h, self.leaf_capacity, d)
                    leaf_v = unified_page1_v[
                        arena_leaf_offset : arena_leaf_offset + r * h * self.leaf_capacity
                    ].view(r, h, self.leaf_capacity, d)
                else:
                    leaf_k = torch.zeros(
                        r,
                        h,
                        self.leaf_capacity,
                        d,
                        dtype=torch.int8 if int8_storage else self.dtype,
                        device=self.device,
                    )
                    leaf_v = torch.zeros_like(leaf_k)
                state["page_cache"] = {
                    "region_owned_pages": True,
                    "dense_leaf_storage": True,
                    "slot_pages": slot_pages,
                    "overflow_page_keys": overflow_page_keys,
                    "overflow_page_values": overflow_page_values,
                    "overflow_hash_capacity": self.hash_capacity,
                    "overflow_flag": torch.zeros(
                        (), dtype=torch.int32, device=self.device
                    ),
                    "overflow_used": torch.zeros(
                        (), dtype=torch.int32, device=self.device
                    ),
                    "overflow_active": overflow_active,
                    "overflow_safe_until": overflow_safe_until,
                    "paged_page_directory": self.settings.leaf_paged_directory,
                    "page_directory_size": 64,
                    "slot_lengths": torch.zeros(
                        r, h, s, dtype=torch.int32, device=self.device
                    ),
                    "static_cohort_status": torch.zeros(
                        r,
                        h,
                        (
                            s
                            if os.getenv(
                                "LOD_STATIC_COHORT_EVICTION_DIAGNOSTICS"
                            )
                            == "1"
                            or self.settings.static_cohort_never_readmit
                            else 0
                        ),
                        dtype=torch.int8,
                        device=self.device,
                    ),
                    "next_page": torch.zeros(
                        r, h, dtype=torch.int32, device=self.device
                    ),
                    "page_size": page_size,
                    "leaf_capacity": self.leaf_capacity,
                    "leaf_count": 0,
                    "page_indices": torch.full(
                        (r, h, self.page_capacity, page_size),
                        -1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "leaf_k": leaf_k,
                    "leaf_v": leaf_v,
                    # The shared virtual-page append primitive maintains these
                    # summaries. Flat two-tier attention does not consume them,
                    # but retaining them keeps append graph-safe and makes the
                    # storage interchangeable with the recursive allocator.
                    "page_sum_k": torch.zeros(
                        r, h, self.page_capacity, d,
                        dtype=self.dtype, device=self.device,
                    ),
                    "page_sum_v": torch.zeros(
                        r, h, self.page_capacity, d,
                        dtype=self.dtype, device=self.device,
                    ),
                    "page_counts": torch.zeros(
                        r, h, self.page_capacity,
                        dtype=torch.int32, device=self.device,
                    ),
                    "quantization_finalized": False,
                    "summary_quantization_finalized": False,
                }
                if unified_page1:
                    state["page_cache"].update(
                        unified_page1_k=unified_page1_k,
                        unified_page1_v=unified_page1_v,
                        unified_page1_bias=unified_page1_bias,
                        unified_page1_capacity=arena_capacity,
                        unified_page1_leaf_offset=arena_leaf_offset,
                        unified_page1_local_offset=arena_local_offset,
                        unified_page1_sink_offset=arena_sink_offset,
                        unified_page1_coarse_offset=arena_coarse_offset,
                    )
                    if isinstance(unified_page1_fixed_indices, torch.Tensor):
                        state["page_cache"].update(
                            unified_page1_fixed_indices=(
                                unified_page1_fixed_indices
                            ),
                            unified_page1_fixed_leaf_owners=(
                                unified_page1_fixed_leaf_owners
                            ),
                            unified_page1_fixed_slot_offsets=(
                                unified_page1_fixed_slot_offsets
                            ),
                            unified_page1_fixed_lengths=(
                                unified_page1_fixed_lengths
                            ),
                        )
                    if self.settings.decode_gqa_predicted_mass:
                        state["page_cache"]["decode_previous_total_lse"] = (
                            torch.full(
                                (r, self.query_heads),
                                float("inf"),
                                dtype=torch.float32,
                                device=self.device,
                            )
                        )
                    if self.settings.decode_gqa_pilot_z:
                        state["page_cache"]["decode_pilot_z_bound"] = torch.full(
                            (r, self.query_heads),
                            float("inf"),
                            dtype=torch.float32,
                            device=self.device,
                        )
                if int8_storage:
                    state["page_cache"].update(
                        page_k_token_scales=torch.zeros(
                            r, h, self.leaf_capacity,
                            dtype=self.dtype, device=self.device,
                        ),
                        page_v_token_scales=torch.zeros(
                            r, h, self.leaf_capacity,
                            dtype=self.dtype, device=self.device,
                        ),
                        prefill_int8_leaf_mma=True,
                    )
                return state
            state["page_cache"] = {
                "region_owned_pages": True,
                "slot_pages": slot_pages,
                "overflow_page_keys": overflow_page_keys,
                "overflow_page_values": overflow_page_values,
                "overflow_hash_capacity": self.hash_capacity,
                "overflow_flag": torch.zeros(
                    (), dtype=torch.int32, device=self.device
                ),
                "overflow_used": torch.zeros(
                    (), dtype=torch.int32, device=self.device
                ),
                "overflow_active": overflow_active,
                "overflow_safe_until": overflow_safe_until,
                "paged_page_directory": self.settings.leaf_paged_directory,
                "page_directory_size": 64,
                "slot_lengths": torch.zeros(
                    r, h, s, dtype=torch.int32, device=self.device
                ),
                "next_page": torch.zeros(
                    r, h, dtype=torch.int32, device=self.device
                ),
                "page_size": page_size,
                "leaf_capacity": self.leaf_capacity,
                "leaf_count": 0,
                "page_k": torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    page_size,
                    d,
                    dtype=torch.int8 if int8_storage else self.dtype,
                    device=self.device,
                ),
                "page_v": torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    page_size,
                    d,
                    dtype=torch.int8 if int8_storage else self.dtype,
                    device=self.device,
                ),
            }
            if int8_storage:
                state["page_cache"].update(
                    page_k_token_scales=torch.zeros(
                        r,
                        h,
                        self.page_capacity,
                        page_size,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    page_v_token_scales=torch.zeros(
                        r,
                        h,
                        self.page_capacity,
                        page_size,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                    prefill_int8_leaf_mma=True,
                )
            return state

        slot_dtype = (
            torch.int16
            if self.page_capacity <= torch.iinfo(torch.int16).max
            else torch.int32
        )
        page: dict[str, object] = {
            "region_owned_pages": True,
            "slot_pages": torch.full(
                (r, h, s, int(self.engine.leaf_inline_pages_per_slot)),
                -1,
                dtype=slot_dtype,
                device=self.device,
            ),
            "overflow_page_keys": torch.full(
                (r, h, self.hash_capacity),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "overflow_page_values": torch.full(
                (r, h, self.hash_capacity),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "overflow_hash_capacity": self.hash_capacity,
            "overflow_flag": torch.zeros((), dtype=torch.int32, device=self.device),
            "overflow_used": torch.zeros((), dtype=torch.int32, device=self.device),
            # A fixed pool cannot specialize graph kernels per request. Always
            # enable the bounded hash lookup; rows without overflow simply miss.
            "overflow_active": True,
            "overflow_safe_until": 0,
            "slot_lengths": torch.zeros(r, h, s, dtype=torch.int32, device=self.device),
            "next_page": torch.zeros(r, h, dtype=torch.int32, device=self.device),
            "page_size": 16,
            "leaf_capacity": self.leaf_capacity,
            "leaf_count": 0,
            "page_indices": torch.full(
                (r, h, self.page_capacity, 16),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "page_counts": torch.zeros(
                r, h, self.page_capacity, dtype=torch.int32, device=self.device
            ),
        }
        groups = d // self.settings.quant_group_size
        token_groups = 16 // self.settings.quant_token_group_size
        if self.settings.kv_bits in (4, 8):
            quant_bits = self.settings.kv_bits
            quant_width = d // 2 if quant_bits == 4 else d
            quant_dtype = torch.uint8 if quant_bits == 4 else torch.int8
            page.update(
                leaf_quant_bits=quant_bits,
                leaf_k=torch.empty(r, h, 1, d, dtype=self.dtype, device=self.device),
                leaf_v=torch.empty(r, h, 1, d, dtype=self.dtype, device=self.device),
                quantized_leaf_k=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    quant_width,
                    dtype=quant_dtype,
                    device=self.device,
                ),
                quantized_leaf_v=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    quant_width,
                    dtype=quant_dtype,
                    device=self.device,
                ),
                page_k_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    token_groups * groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_v_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    token_groups * groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_quantized_counts=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    dtype=torch.int32,
                    device=self.device,
                ),
                page_sum_k=torch.empty(
                    r, h, 1, d, dtype=self.dtype, device=self.device
                ),
                page_sum_v=torch.empty(
                    r, h, 1, d, dtype=self.dtype, device=self.device
                ),
                quantized_page_sum_k=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=torch.int8,
                    device=self.device,
                ),
                quantized_page_sum_v=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=torch.int8,
                    device=self.device,
                ),
                page_sum_k_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_sum_v_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                quantization_finalized=True,
                summary_quantization_finalized=True,
            )
        else:
            page.update(
                leaf_quant_bits=0,
                leaf_k=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                leaf_v=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_sum_k=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_sum_v=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                quantization_finalized=False,
                summary_quantization_finalized=False,
            )
        state["page_cache"] = page
        return state

    def reset(self, slot: int) -> None:
        if not 0 <= slot < self.max_requests:
            raise IndexError("vLLM request slot is outside the LOD pool")
        self._snapshot_static_cohort_eviction(slot)
        self.ready[slot] = False
        self.clean[slot] = True
        self.metadata[slot].clear()
        self.local_lens[slot].zero_()
        self.state["counts"][slot].zero_()
        if "sink_k" in self.state:
            self.state["sink_k"][slot].zero_()
            self.state["sink_v"][slot].zero_()
        if "key_norm_sums" in self.state:
            self.state["key_norm_sums"][slot].zero_()
        page = self.state["page_cache"]
        page["slot_pages"][slot].fill_(-1)
        page["slot_lengths"][slot].zero_()
        if "static_cohort_status" in page:
            page["static_cohort_status"][slot].zero_()
        page["next_page"][slot].zero_()
        if "page_indices" in page:
            page["page_indices"][slot].fill_(-1)
        if "page_counts" in page:
            page["page_counts"][slot].zero_()
        page["overflow_page_keys"][slot].fill_(-1)
        page["overflow_page_values"][slot].fill_(-1)
        if "page_quantized_counts" in page:
            page["page_quantized_counts"][slot].zero_()
        if "decode_previous_total_lse" in page:
            page["decode_previous_total_lse"][slot].fill_(float("inf"))
        if "decode_pilot_z_bound" in page:
            page["decode_pilot_z_bound"][slot].fill_(float("inf"))

    def _reset_range(self, start: int, stop: int) -> None:
        """Reset one contiguous row range with one launch per cache field."""
        if not 0 <= start < stop <= self.max_requests:
            raise IndexError("vLLM request row range is outside the LOD pool")
        for slot in range(start, stop):
            self._snapshot_static_cohort_eviction(slot)
            self.ready[slot] = False
            self.clean[slot] = True
            self.metadata[slot].clear()
        self.local_lens[start:stop].zero_()
        self.state["counts"][start:stop].zero_()
        if "sink_k" in self.state:
            self.state["sink_k"][start:stop].zero_()
            self.state["sink_v"][start:stop].zero_()
        if "key_norm_sums" in self.state:
            self.state["key_norm_sums"][start:stop].zero_()
        page = self.state["page_cache"]
        page["slot_pages"][start:stop].fill_(-1)
        page["slot_lengths"][start:stop].zero_()
        if "static_cohort_status" in page:
            page["static_cohort_status"][start:stop].zero_()
        page["next_page"][start:stop].zero_()
        if "page_indices" in page:
            page["page_indices"][start:stop].fill_(-1)
        if "page_counts" in page:
            page["page_counts"][start:stop].zero_()
        page["overflow_page_keys"][start:stop].fill_(-1)
        page["overflow_page_values"][start:stop].fill_(-1)
        if "page_quantized_counts" in page:
            page["page_quantized_counts"][start:stop].zero_()
        if "decode_previous_total_lse" in page:
            page["decode_previous_total_lse"][start:stop].fill_(float("inf"))
        if "decode_pilot_z_bound" in page:
            page["decode_pilot_z_bound"][start:stop].fill_(float("inf"))

    def _snapshot_static_cohort_eviction(self, slot: int) -> None:
        """Preserve the final monotone-cohort working set before row reuse."""
        if (
            (
                os.getenv("LOD_STATIC_COHORT_EVICTION_DIAGNOSTICS") != "1"
                and not self.settings.static_cohort_never_readmit
            )
            or self.clean[slot]
            or not self.metadata[slot]
        ):
            return
        page = self.state.get("page_cache")
        if not isinstance(page, dict):
            return
        status = page.get("static_cohort_status")
        lengths = page.get("slot_lengths")
        if not isinstance(status, torch.Tensor) or not isinstance(
            lengths, torch.Tensor
        ):
            return
        row_status = status[slot]
        row_lengths = lengths[slot]
        cap = scheduled_static_leaf_cap(
            int(self.metadata[slot].get("total_len", 0)),
            minimum=int(self.settings.prefill_static_leaf_cap_min),
            divisor=int(self.settings.static_leaf_cap_divisor),
        )
        active = row_lengths.gt(0)
        row_status.masked_fill_(row_status.eq(0) & active, 1)
        row_status.masked_fill_(row_status.eq(1) & row_lengths.gt(cap), -1)
        snapshots = getattr(self, "_static_cohort_eviction_snapshots", None)
        if snapshots is None:
            snapshots = {}
            self._static_cohort_eviction_snapshots = snapshots
        snapshots[int(slot)] = (
            row_lengths.detach().to("cpu").clone(),
            row_status.detach().to("cpu").clone(),
            int(cap),
            int(self.metadata[slot].get("total_len", 0)),
        )

    def truncate_recent(self, slot: int, total_length: int) -> None:
        """Roll a retained cache back inside its unclustered exact tail."""
        if not self.ready[slot]:
            raise RuntimeError("cannot truncate an uninitialized LOD cache")
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        old_total = int(metadata["total_len"])
        if not coverage <= total_length <= old_total:
            raise ValueError(
                "retained LOD prefix lies outside the exact recent tail: "
                f"coverage={coverage}, requested={total_length}, total={old_total}"
            )
        recent_len = total_length - coverage
        self.local_lens[slot].fill_(recent_len)
        metadata["recent_len"] = recent_len
        metadata["total_len"] = total_length

    def restore_prefix(self, slot: int, total_length: int) -> None:
        """Restore any retained prefix without consulting native model K/V.

        A prefix inside the live exact tail is a metadata-only rollback.  A
        more distant shared prefix cannot undo centroid updates, but two-level
        LOD already archives every underlying leaf in chronological order.  In
        that case rebuild the much smaller semantic cache from those owned
        leaves.  This preserves general vLLM prefix caching (not just repeated
        whole prompts) while the model's chronological native K/V stays absent.
        """
        if not self.ready[slot]:
            raise RuntimeError("cannot restore an uninitialized LOD cache")
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        old_total = int(metadata["total_len"])
        if not 0 < total_length <= old_total:
            raise ValueError(
                "retained LOD prefix lies outside the cached request: "
                f"requested={total_length}, total={old_total}"
            )
        if coverage <= total_length:
            self.truncate_recent(slot, total_length)
            return
        if self.settings.levels != 2 or not self.settings.dense_leaf_storage:
            raise NotImplementedError(
                "distant authoritative prefix restoration currently requires "
                "two-level chronological dense-leaf storage"
            )
        if self.engine.state_clustering_query_metric != "none":
            raise NotImplementedError(
                "query-conditioned routing cannot rebuild a distant cached prefix"
            )
        if self.engine.mla_key_norm_weight is not None:
            raise NotImplementedError(
                "MLA distant-prefix restoration requires archived raw latents"
            )

        page = self.state["page_cache"]
        leaf_k = page.get("leaf_k")
        leaf_v = page.get("leaf_v")
        if not isinstance(leaf_k, torch.Tensor) or not isinstance(
            leaf_v, torch.Tensor
        ):
            raise RuntimeError("retained LOD row has no chronological leaf archive")
        leaf_count = int(metadata["leaf_count"])
        if leaf_count < total_length:
            raise RuntimeError(
                "retained LOD leaf archive was sealed before the requested prefix: "
                f"leaves={leaf_count}, requested={total_length}"
            )

        key = leaf_k[slot : slot + 1, :, :total_length, :]
        value = leaf_v[slot : slot + 1, :, :total_length, :]
        if key.dtype == torch.int8 or value.dtype == torch.int8:
            if key.dtype != torch.int8 or value.dtype != torch.int8:
                raise TypeError("retained INT8 prefix requires both K and V in INT8")
            key_scales = page.get("page_k_token_scales")
            value_scales = page.get("page_v_token_scales")
            if not isinstance(key_scales, torch.Tensor) or not isinstance(
                value_scales, torch.Tensor
            ):
                raise RuntimeError("retained INT8 leaves are missing token scales")
            key = key.to(self.dtype) * key_scales[
                slot : slot + 1, :, :total_length
            ].unsqueeze(-1)
            value = value.to(self.dtype) * value_scales[
                slot : slot + 1, :, :total_length
            ].unsqueeze(-1)
        else:
            # The converted cache may retain its flat source as virtual leaf
            # storage. Clone before install() resets the old pool row.
            key = key.clone()
            value = value.clone()
        converted = self.engine.build_cache_from_bf16(
            key.contiguous(), value.contiguous()
        )
        self.install(slot, converted)
        self.engine.reset_runtime_cache()
        self.retained_restore_rebuilds += 1
        self.retained_restore_rebuild_tokens += total_length

    @staticmethod
    def _copy_row(
        destination: torch.Tensor,
        source: torch.Tensor,
        slot: int,
        source_slot: int = 0,
    ) -> None:
        source = source[source_slot : source_slot + 1]
        target = destination[slot : slot + 1]
        if source.ndim != target.ndim:
            raise ValueError("converted LOD tensor rank differs from its pool")
        if (
            source.data_ptr() == target.data_ptr()
            and source.shape == target.shape
            and source.stride() == target.stride()
        ):
            return
        slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
        target[slices].copy_(source[slices])

    @staticmethod
    def _copy_range(
        destination: torch.Tensor,
        source: torch.Tensor,
        start: int,
        stop: int,
    ) -> None:
        target = destination[start:stop]
        if source.ndim != target.ndim or int(source.size(0)) != stop - start:
            raise ValueError("converted LOD tensor batch differs from its pool range")
        if (
            source.data_ptr() == target.data_ptr()
            and source.shape == target.shape
            and source.stride() == target.stride()
        ):
            return
        slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
        target[slices].copy_(source[slices])

    @staticmethod
    def _copy_rows(
        destination: torch.Tensor,
        source: torch.Tensor,
        slots: tuple[int, ...],
        indices: torch.Tensor,
    ) -> None:
        if source.ndim != destination.ndim or int(source.size(0)) != len(slots):
            raise ValueError("converted LOD tensor batch differs from its pool rows")
        slices = (slice(None),) + tuple(
            slice(0, min(int(destination.size(axis)), int(source.size(axis))))
            for axis in range(1, source.ndim)
        )
        destination[slices].index_copy_(
            0,
            indices,
            source[slices],
        )

    def _persist_decode_pilot_z_bound(
        self,
        slots: tuple[int, ...],
        *,
        source_slots: tuple[int, ...] | None = None,
    ) -> None:
        """Install the latest prefill calibration into authoritative rows."""

        if not self.settings.decode_gqa_pilot_z:
            return
        destination = self.state["page_cache"].get("decode_pilot_z_bound")
        source = getattr(self.engine, "_lod_decode_pilot_z_bound", None)
        if not isinstance(destination, torch.Tensor) or not isinstance(
            source, torch.Tensor
        ):
            return
        if source.ndim != 2 or int(source.size(1)) != self.query_heads:
            raise ValueError("calibrated pilot-z bounds have the wrong shape")
        if source_slots is None:
            source_slots = tuple(range(len(slots)))
        if len(source_slots) != len(slots):
            raise ValueError("pilot-z source and destination rows do not match")
        if any(
            not 0 <= source_slot < int(source.size(0))
            for source_slot in source_slots
        ):
            # Mixed prefill/decode scheduler steps may finish several
            # independent catch-up groups in one model invocation.  The
            # engine scratch retains only the most recently evaluated group's
            # calibration, while each group's page-cache tensor was already
            # copied by the caller.  There is no valid row mapping from stale
            # scratch in that case.
            return
        for source_slot, destination_slot in zip(
            source_slots, slots, strict=True
        ):
            destination[destination_slot].copy_(source[source_slot])

    def _refresh_unified_page1_coarse(self, slots: tuple[int, ...]) -> None:
        """Materialize centroid means in the persistent AITER K/V arena.

        State updates retain sums because routing and archival mutate them in
        that representation.  Decode attention consumes means, so refresh the
        arena only at install/catch-up boundaries rather than dividing every
        centroid in the hot decode path.
        """
        if not slots:
            return
        page = self.state.get("page_cache")
        if not isinstance(page, dict):
            return
        arena_k = page.get("unified_page1_k")
        arena_v = page.get("unified_page1_v")
        arena_bias = page.get("unified_page1_bias")
        coarse_offset = page.get("unified_page1_coarse_offset")
        if (
            not isinstance(arena_k, torch.Tensor)
            or not isinstance(arena_v, torch.Tensor)
            or not isinstance(arena_bias, torch.Tensor)
        ):
            return
        if not isinstance(coarse_offset, int):
            raise TypeError("unified page-size-one coarse offset is invalid")
        coarse_k = arena_k[
            coarse_offset : coarse_offset
            + self.max_requests * self.kv_heads * self.state_capacity
        ].view(
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
            self.head_dim,
        )
        coarse_v = arena_v[
            coarse_offset : coarse_offset
            + self.max_requests * self.kv_heads * self.state_capacity
        ].view_as(coarse_k)
        coarse_bias = arena_bias[
            coarse_offset : coarse_offset
            + self.max_requests * self.kv_heads * self.state_capacity
        ].view(
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
        )
        ordered = tuple(sorted(slots))
        begin = 0
        while begin < len(ordered):
            end = begin + 1
            while end < len(ordered) and ordered[end] == ordered[end - 1] + 1:
                end += 1
            start_slot = ordered[begin]
            stop_slot = ordered[end - 1] + 1
            materialize_page1_coarse_means(
                self.state["state_k"][start_slot:stop_slot],
                self.state["state_v"][start_slot:stop_slot],
                self.state["counts"][start_slot:stop_slot],
                coarse_k[start_slot:stop_slot],
                coarse_v[start_slot:stop_slot],
                coarse_bias[start_slot:stop_slot],
            )
            begin = end
        self._refresh_unified_page1_fixed(slots)

    def _refresh_unified_page1_fixed(self, slots: tuple[int, ...]) -> None:
        """Rebuild persistent page-size-one lists after state updates."""
        if not slots:
            return
        page = self.state.get("page_cache")
        if not isinstance(page, dict):
            return
        fixed_indices = page.get("unified_page1_fixed_indices")
        fixed_leaf_owners = page.get("unified_page1_fixed_leaf_owners")
        fixed_slot_offsets = page.get("unified_page1_fixed_slot_offsets")
        fixed_lengths = page.get("unified_page1_fixed_lengths")
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (
                fixed_indices,
                fixed_slot_offsets,
                fixed_lengths,
            )
        ):
            return
        static_cap_page1 = bool(self.settings.decode_gqa_static_leaf_aiter)
        if not static_cap_page1 and not isinstance(
            fixed_leaf_owners, torch.Tensor
        ):
            return
        if self.settings.static_cohort_never_readmit:
            status = page.get("static_cohort_status")
            lengths = page.get("slot_lengths")
            if not isinstance(status, torch.Tensor) or not isinstance(
                lengths, torch.Tensor
            ):
                raise RuntimeError("monotone static cohort storage is missing")
            if tuple(status.shape) != tuple(lengths.shape):
                raise ValueError("monotone static cohort storage has the wrong shape")
            for slot in slots:
                row_status = status[slot]
                row_lengths = lengths[slot]
                active = row_lengths.gt(0)
                row_status.masked_fill_(row_status.eq(0) & active, 1)
                row_status.masked_fill_(
                    row_status.eq(1)
                    & row_lengths.gt(self._static_leaf_cap_for_slot(slot)),
                    -1,
                )
        if (
            self.settings.decode_gqa_static_leaf_cap is not None
            or static_cap_page1
        ):
            # Finished-request teardown clears the live GPU counts before the
            # driver asks for diagnostics. Preserve only this experiment's
            # small posting-list histogram source at each update boundary.
            snapshots = getattr(self, "_static_leaf_cap_count_snapshots", None)
            if snapshots is None:
                snapshots = {}
                self._static_leaf_cap_count_snapshots = snapshots
            cap_snapshots = getattr(
                self, "_static_leaf_cap_value_snapshots", None
            )
            if cap_snapshots is None:
                cap_snapshots = {}
                self._static_leaf_cap_value_snapshots = cap_snapshots
            for slot in slots:
                snapshots[int(slot)] = (
                    page["slot_lengths"][slot].detach().to("cpu").clone()
                )
                cap_snapshots[int(slot)] = self._static_leaf_cap_for_slot(slot)
        sink = self.state.get("sink_k")
        sink_len = int(sink.size(2)) if isinstance(sink, torch.Tensor) else 0
        ordered = tuple(sorted(slots))
        begin = 0
        while begin < len(ordered):
            static_leaf_cap = (
                self._static_leaf_cap_for_slot(ordered[begin])
                if static_cap_page1
                else None
            )
            end = begin + 1
            while (
                end < len(ordered)
                and ordered[end] == ordered[end - 1] + 1
                and (
                    not static_cap_page1
                    or self._static_leaf_cap_for_slot(ordered[end])
                    == static_leaf_cap
                )
            ):
                end += 1
            start_slot = ordered[begin]
            stop_slot = ordered[end - 1] + 1
            common = dict(
                row_offset=start_slot,
                arena_leaf_offset=int(page["unified_page1_leaf_offset"]),
                arena_local_offset=int(page["unified_page1_local_offset"]),
                arena_sink_offset=int(page["unified_page1_sink_offset"]),
                arena_coarse_offset=int(page["unified_page1_coarse_offset"]),
                local_capacity=self.local_capacity,
                local_limit=int(self.engine.local_len),
                sink_capacity=sink_len,
                sink_len=sink_len,
                hash_probes=int(self.engine._page_lookup_probes(page)),
            )
            positional = (
                page["page_indices"][start_slot:stop_slot],
                page["slot_pages"][start_slot:stop_slot],
                page["overflow_page_keys"][start_slot:stop_slot],
                page["overflow_page_values"][start_slot:stop_slot],
                page["overflow_used"],
                page["slot_lengths"][start_slot:stop_slot],
                fixed_indices[start_slot:stop_slot],
            )
            if static_cap_page1:
                if static_leaf_cap is None:
                    raise AssertionError("scheduled static leaf cap is missing")
                materialize_page1_static_cap_indices(
                    *positional,
                    fixed_slot_offsets[start_slot:stop_slot],
                    fixed_lengths[start_slot:stop_slot],
                    cohort_status=(
                        page["static_cohort_status"][start_slot:stop_slot]
                        if self.settings.static_cohort_never_readmit
                        else None
                    ),
                    leaf_capacity=self.leaf_capacity,
                    max_exact_leaves=static_leaf_cap,
                    **common,
                )
            else:
                materialize_page1_fixed_indices(
                    *positional,
                    fixed_leaf_owners[start_slot:stop_slot],
                    fixed_slot_offsets[start_slot:stop_slot],
                    fixed_lengths[start_slot:stop_slot],
                    **common,
                )
            begin = end

    def _static_leaf_cap_for_slot(self, slot: int) -> int:
        """Resolve the fixed override or the length-dependent default cap."""
        fixed = self.settings.decode_gqa_static_leaf_cap
        if fixed is not None:
            return int(fixed)
        total_length = int(self.metadata[slot].get("total_len", 0))
        return scheduled_static_leaf_cap(
            total_length,
            minimum=int(self.settings.decode_gqa_static_leaf_cap_min),
            divisor=int(self.settings.static_leaf_cap_divisor),
        )

    def _decode_route_leaf_limit(self) -> int | None:
        """Return the route kernel's exclusive posting-list eligibility limit.

        Cohort routing is an override, not an additional intersection with the
        legacy overfull-centroid guard.  A fixed cohort cap is useful for graph
        captured single-length experiments; otherwise the allocation length
        supplies the schedule until the per-row device limit is refreshed.
        """
        if not self.settings.decode_route_cohort:
            return self.settings.decode_max_open_leaves
        cap = self.settings.decode_gqa_static_leaf_cap
        if cap is None:
            cap = scheduled_static_leaf_cap(
                self.request_capacity,
                minimum=int(self.settings.decode_gqa_static_leaf_cap_min),
                divisor=int(self.settings.static_leaf_cap_divisor),
            )
        # Route kernels use count < limit; the cohort definition is inclusive.
        return int(cap) + 1

    def install_range(
        self, start: int, stop: int, converted: KernelLODCache
    ) -> None:
        self.install_rows(tuple(range(start, stop)), converted)

    def install_rows(
        self, slots: tuple[int, ...], converted: KernelLODCache
    ) -> None:
        """Install one converted batch into a contiguous set of pool rows."""
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("converted LOD row indices must be nonempty and unique")
        start, stop = min(slots), max(slots) + 1
        if tuple(sorted(slots)) != tuple(range(start, stop)):
            raise ValueError("batched LOD installation requires a contiguous row set")
        source = converted.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("converted LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("converted prefix exceeds VLLM_LOD_MAX_CONTEXT")
        if int(source["state_k"].size(0)) != len(slots):
            raise ValueError("converted LOD batch differs from its pool row range")
        self.batched_install_calls += 1
        if not all(self.clean[slot] for slot in slots):
            self._reset_range(start, stop)
        ascending = slots == tuple(range(start, stop))
        slot_indices = (
            None
            if ascending
            else torch.tensor(slots, dtype=torch.long, device=self.device)
        )

        def copy(destination: torch.Tensor, value: torch.Tensor) -> None:
            if ascending:
                self._copy_range(destination, value, start, stop)
            else:
                if slot_indices is None:
                    raise AssertionError("permuted LOD row indices are missing")
                self._copy_rows(destination, value, slots, slot_indices)

        tensor_names = ["state_k", "state_v", "counts", "recent_k", "recent_v"]
        if "sink_k" in source:
            tensor_names.extend(("sink_k", "sink_v"))
        if "key_norm_sums" in source:
            tensor_names.append("key_norm_sums")
        for name in tensor_names:
            copy(self.state[name], source[name])

        destination_page = self.state["page_cache"]
        for name, value in source_page.items():
            if name in ("overflow_page_keys", "overflow_page_values") or (
                name.startswith("unified_page1_")
            ):
                continue
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                copy(destination, value)
        source_keys = source_page["overflow_page_keys"]
        source_values = source_page["overflow_page_values"]
        destination_keys = destination_page["overflow_page_keys"]
        destination_values = destination_page["overflow_page_values"]
        if int(source_keys.size(2)) == int(destination_keys.size(2)):
            copy(destination_keys, source_keys)
            copy(destination_values, source_values)
            destination_page["overflow_used"].logical_or_(source_page["overflow_used"])
        else:
            for source_slot, destination_slot in enumerate(slots):
                rehash_overflow_pages(
                    source_keys,
                    source_values,
                    destination_keys,
                    destination_values,
                    destination_page["overflow_used"],
                    destination_page["overflow_flag"],
                    source_slot=source_slot,
                    destination_slot=destination_slot,
                )
        destination_page["overflow_flag"].logical_or_(source_page["overflow_flag"])
        self._persist_decode_pilot_z_bound(slots)
        recent_len = int(source["recent_len"])
        if ascending:
            self.local_lens[start:stop].fill_(recent_len)
        else:
            if slot_indices is None:
                raise AssertionError("permuted LOD row indices are missing")
            self.local_lens.index_fill_(0, slot_indices, recent_len)
        for slot in slots:
            self.metadata[slot].update(
                state_len=int(source["state_len"]),
                scheduled_state_len=int(
                    source.get("scheduled_state_len", source["state_len"])
                ),
                coverage=int(source["coverage"]),
                total_len=int(source["total_len"]),
                recent_len=recent_len,
                leaf_count=int(source_page["leaf_count"]),
                overflow_safe_until=int(source_page["overflow_safe_until"]),
            )
            self.ready[slot] = True
            self.clean[slot] = False
            self.install_count += 1
        self._record_split_state(source, source_page)
        self._refresh_unified_page1_coarse(slots)

    def install(
        self, slot: int, converted: KernelLODCache, *, source_slot: int = 0
    ) -> None:
        source = converted.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("converted LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("converted prefix exceeds VLLM_LOD_MAX_CONTEXT")
        self.reset(slot)
        tensor_names = ["state_k", "state_v", "counts", "recent_k", "recent_v"]
        if "sink_k" in source:
            tensor_names.extend(("sink_k", "sink_v"))
        for name in tensor_names:
            self._copy_row(
                self.state[name], source[name], slot, source_slot=source_slot
            )
        if "key_norm_sums" in source:
            self._copy_row(
                self.state["key_norm_sums"],
                source["key_norm_sums"],
                slot,
                source_slot=source_slot,
            )

        destination_page = self.state["page_cache"]
        for name, value in source_page.items():
            if name in ("overflow_page_keys", "overflow_page_values") or (
                name.startswith("unified_page1_")
            ):
                continue
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                self._copy_row(
                    destination, value, slot, source_slot=source_slot
                )
        source_keys = source_page["overflow_page_keys"]
        source_values = source_page["overflow_page_values"]
        destination_keys = destination_page["overflow_page_keys"]
        destination_values = destination_page["overflow_page_values"]
        if int(source_keys.size(2)) == int(destination_keys.size(2)):
            self._copy_row(
                destination_keys, source_keys, slot, source_slot=source_slot
            )
            self._copy_row(
                destination_values, source_values, slot, source_slot=source_slot
            )
            destination_page["overflow_used"].logical_or_(
                source_page["overflow_used"]
            )
        else:
            rehash_overflow_pages(
                source_keys,
                source_values,
                destination_keys,
                destination_values,
                destination_page["overflow_used"],
                destination_page["overflow_flag"],
                source_slot=source_slot,
                destination_slot=slot,
            )
        destination_page["overflow_flag"].logical_or_(
            source_page["overflow_flag"]
        )
        self._persist_decode_pilot_z_bound(
            (slot,), source_slots=(source_slot,)
        )
        recent_len = int(source["recent_len"])
        self.local_lens[slot].fill_(recent_len)
        self.metadata[slot].update(
            state_len=int(source["state_len"]),
            scheduled_state_len=int(
                source.get("scheduled_state_len", source["state_len"])
            ),
            coverage=int(source["coverage"]),
            total_len=int(source["total_len"]),
            recent_len=recent_len,
            leaf_count=int(source_page["leaf_count"]),
            overflow_safe_until=int(source_page["overflow_safe_until"]),
        )
        self.ready[slot] = True
        self.clean[slot] = False
        self.install_count += 1
        self._record_split_state(source, source_page)
        self._refresh_unified_page1_coarse((slot,))

    def _synchronize_row(self, slot: int, cache: KernelLODCache) -> None:
        """Persist metadata and any reallocated tensors after cached prefill."""
        self._synchronize_rows((slot,), cache)

    def _synchronize_rows(
        self, slots: tuple[int, ...], cache: KernelLODCache
    ) -> None:
        """Persist an equal-metadata batch after cached prefill."""
        if not slots:
            return
        source = cache.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("updated LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("updated prefix exceeds VLLM_LOD_MAX_CONTEXT")
        if int(source["state_k"].size(0)) != len(slots):
            raise ValueError("updated LOD batch does not match its pool rows")
        tensor_names = ["state_k", "state_v", "counts", "recent_k", "recent_v"]
        if "sink_k" in source:
            tensor_names.extend(("sink_k", "sink_v"))
        for name in tensor_names:
            for source_slot, slot in enumerate(slots):
                self._copy_row(
                    self.state[name], source[name], slot, source_slot=source_slot
                )
        if "key_norm_sums" in source:
            for source_slot, slot in enumerate(slots):
                self._copy_row(
                    self.state["key_norm_sums"],
                    source["key_norm_sums"],
                    slot,
                    source_slot=source_slot,
                )
        destination_page = self.state["page_cache"]
        for name, value in source_page.items():
            if name.startswith("unified_page1_"):
                continue
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                for source_slot, slot in enumerate(slots):
                    self._copy_row(
                        destination, value, slot, source_slot=source_slot
                    )
        self._persist_decode_pilot_z_bound(slots)
        recent_len = int(source["recent_len"])
        for slot in slots:
            self.local_lens[slot].fill_(recent_len)
            self.metadata[slot].update(
                state_len=int(source["state_len"]),
                scheduled_state_len=int(
                    source.get("scheduled_state_len", source["state_len"])
                ),
                coverage=int(source["coverage"]),
                total_len=int(source["total_len"]),
                recent_len=recent_len,
                leaf_count=int(source_page["leaf_count"]),
                overflow_safe_until=int(source_page["overflow_safe_until"]),
            )
            self.ready[slot] = True
        self._record_split_state(source, source_page)
        self._refresh_unified_page1_coarse(slots)

    def _direct_cached_prefill_group(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        plan: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        """Advance one contiguous equal-length/equal-history cache group."""
        length = plan[0][2] - plan[0][1]
        previous_length = plan[0][3]
        slots = tuple(slot for slot, _, _, _ in plan)
        packed_begin = plan[0][1]
        packed_end = packed_begin + len(plan) * length
        packed = packed_end <= int(query.size(0)) and all(
            begin == packed_begin + source_slot * length
            and end == packed_begin + (source_slot + 1) * length
            for source_slot, (_, begin, end, _) in enumerate(plan)
        )
        if packed:
            self.cached_prefill_packed_calls += 1
        else:
            self.cached_prefill_nonpacked_calls += 1
        q = (
            query[packed_begin:packed_end]
            .reshape(len(plan), length, *query.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [query[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        k = (
            key[packed_begin:packed_end]
            .reshape(len(plan), length, *key.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [key[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        v = (
            value[packed_begin:packed_end]
            .reshape(len(plan), length, *value.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [value[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        contiguous_slots = slots == tuple(
            range(slots[0], slots[0] + len(slots))
        )
        cache = (
            self._range_cache(slots[0], slots[-1] + 1)
            if contiguous_slots
            else self._selected_cache(slots)
        )
        if cache.total_length != previous_length:
            raise RuntimeError(
                "batched cached LOD prefill length differs from its prepared plan"
            )
        output_view = (
            output[packed_begin:packed_end]
            .reshape(len(plan), length, *output.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else None
        )
        result, cache = self.engine(
            q,
            k,
            v,
            cache=cache,
            use_cache=True,
            output_buffer=output_view,
            finalize_cache_for_decode=False,
        )
        if cache is None:
            raise AssertionError("batched cached LOD prefill did not return a cache")
        if len(slots) > 1:
            self.batched_cached_prefill_calls += 1
            self.batched_cached_prefill_rows += len(slots)
        self._synchronize_rows(slots, cache)
        self.engine.reset_runtime_cache()
        if packed:
            if output_view is None:
                raise AssertionError("packed cached prefill has no output view")
            if result.data_ptr() != output_view.data_ptr():
                output_view.copy_(result)
        else:
            for source_slot, (_, begin, end, _) in enumerate(plan):
                output[begin:end].copy_(result[source_slot].permute(1, 0, 2))

    def direct_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run ragged initial or cached prefill into authoritative LOD rows."""
        plan = self.direct_prefill_plan
        self.direct_prefill_plan = None
        prompt_lengths = self.direct_prefill_prompt_lengths
        self.direct_prefill_prompt_lengths = {}
        if plan is None:
            raise RuntimeError("direct LOD prefill has no prepared request plan")
        self.direct_prefill_calls += 1
        initial: dict[int, list[tuple[int, int, int, int]]] = {}
        cached: list[tuple[int, int, int, int]] = []
        for item in plan:
            slot, begin, end, previous_length = item
            if end <= begin:
                continue
            if previous_length == 0 and not self.ready[slot]:
                initial.setdefault(end - begin, []).append(item)
            elif previous_length > 0 and self.ready[slot]:
                cached.append(item)
            elif previous_length > 0:
                raise RuntimeError("cached LOD prefill row is not initialized")
            else:
                raise RuntimeError("initial LOD prefill row is already initialized")

        for length, group in initial.items():
            slots = tuple(slot for slot, _, _, _ in group)
            if any(slot not in prompt_lengths for slot in slots):
                raise RuntimeError("direct LOD prefill has no total prompt length")
            self.engine.recursive_prefill_request_total_len = max(
                prompt_lengths[slot] for slot in slots
            )
            packed_begin = group[0][1]
            packed_end = packed_begin + len(group) * length
            packed = packed_end <= int(query.size(0)) and all(
                begin == packed_begin + source_slot * length
                and end == packed_begin + (source_slot + 1) * length
                for source_slot, (_, begin, end, _) in enumerate(group)
            )
            q = (
                query[packed_begin:packed_end]
                .reshape(len(group), length, *query.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [
                        query[begin:end].permute(1, 0, 2)
                        for _, begin, end, _ in group
                    ]
                )
            )
            k = (
                key[packed_begin:packed_end]
                .reshape(len(group), length, *key.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [key[begin:end].permute(1, 0, 2) for _, begin, end, _ in group]
                )
            )
            v = (
                value[packed_begin:packed_end]
                .reshape(len(group), length, *value.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [value[begin:end].permute(1, 0, 2) for _, begin, end, _ in group]
                )
            )
            output_view = (
                output[packed_begin:packed_end]
                .reshape(len(group), length, *output.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else None
            )
            result, cache = self.engine(
                q,
                k,
                v,
                use_cache=True,
                output_buffer=output_view,
                finalize_cache_for_decode=False,
            )
            if cache is None:
                raise AssertionError("direct LOD prefill did not return a cache")
            if tuple(sorted(slots)) == tuple(range(min(slots), max(slots) + 1)):
                self.install_rows(slots, cache)
            else:
                for source_slot, (slot, _, _, _) in enumerate(group):
                    self.install(slot, cache, source_slot=source_slot)
            self.engine.reset_runtime_cache()
            if packed:
                if output_view is None or result.data_ptr() != output_view.data_ptr():
                    raise AssertionError("LOD prefill did not use its output buffer")
            else:
                for source_slot, (_, begin, end, _) in enumerate(group):
                    output[begin:end].copy_(result[source_slot].permute(1, 0, 2))

        if not cached:
            return output
        lengths = {end - begin for _, begin, end, _ in cached}
        previous_lengths = {
            previous_length for _, _, _, previous_length in cached
        }
        slots = tuple(slot for slot, _, _, _ in cached)
        ordered_plan = tuple(sorted(cached, key=lambda item: item[0]))
        ordered_slots = tuple(slot for slot, _, _, _ in ordered_plan)
        contiguous_slots = bool(ordered_slots) and ordered_slots == tuple(
            range(ordered_slots[0], ordered_slots[0] + len(ordered_slots))
        )
        self.cached_prefill_candidate_calls += 1
        self.cached_prefill_candidate_rows += len(cached)
        self.cached_prefill_nonuniform_lengths += int(len(lengths) != 1)
        self.cached_prefill_nonuniform_previous += int(len(previous_lengths) != 1)
        self.cached_prefill_noncontiguous += int(not contiguous_slots)
        groups: dict[
            tuple[int, ...], list[tuple[int, int, int, int]]
        ] = {}
        for item in ordered_plan:
            slot, begin, end, previous_length = item
            metadata = self.metadata[slot]
            signature = (
                end - begin,
                previous_length,
                int(metadata["state_len"]),
                int(metadata.get("scheduled_state_len", metadata["state_len"])),
                int(metadata["coverage"]),
                int(metadata["recent_len"]),
                int(metadata["leaf_count"]),
                int(metadata["overflow_safe_until"]),
            )
            groups.setdefault(signature, []).append(item)
        for group in groups.values():
            group_slots = tuple(slot for slot, _, _, _ in group)
            if any(slot not in prompt_lengths for slot in group_slots):
                raise RuntimeError("cached LOD prefill has no total prompt length")
            self.engine.recursive_prefill_request_total_len = max(
                prompt_lengths[slot] for slot in group_slots
            )
            self._direct_cached_prefill_group(
                query,
                key,
                value,
                output,
                tuple(group),
            )
        return output

    def _row_cache(self, slot: int) -> KernelLODCache:
        return self._range_cache(slot, slot + 1)

    def _selected_cache(self, slots: tuple[int, ...]) -> KernelLODCache:
        """Gather equal-metadata noncontiguous rows for one prefill call."""
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("LOD cache row indices must be nonempty and unique")
        if any(not 0 <= slot < self.max_requests for slot in slots):
            raise IndexError("LOD cache row index is outside the fixed pool")
        metadata = self.metadata[slots[0]]
        scalar_names = (
            "state_len",
            "scheduled_state_len",
            "coverage",
            "recent_len",
            "total_len",
            "leaf_count",
            "overflow_safe_until",
        )
        if any(
            int(self.metadata[slot][name]) != int(metadata[name])
            for slot in slots[1:]
            for name in scalar_names
        ):
            raise ValueError("gathered LOD catch-up rows have different metadata")
        indices = torch.tensor(slots, dtype=torch.long, device=self.device)
        state: dict[str, object] = {
            name: value.index_select(0, indices)
            for name, value in self.state.items()
            if isinstance(value, torch.Tensor) and value.ndim
        }
        state.update(
            state_len=int(metadata["state_len"]),
            scheduled_state_len=int(
                metadata.get("scheduled_state_len", metadata["state_len"])
            ),
            coverage=int(metadata["coverage"]),
            state_capacity=self.state_capacity,
            recent_len=int(metadata["recent_len"]),
            total_len=int(metadata["total_len"]),
        )
        page_pool = self.state["page_cache"]
        page: dict[str, object] = {
            name: (
                value.index_select(0, indices)
                if isinstance(value, torch.Tensor) and value.ndim
                else value
            )
            for name, value in page_pool.items()
            if not name.startswith("unified_page1_")
        }
        page.update(
            leaf_count=int(metadata["leaf_count"]),
            leaf_capacity=self.leaf_capacity,
            overflow_active=True,
            overflow_safe_until=int(metadata["overflow_safe_until"]),
        )
        state["page_cache"] = page
        return KernelLODCache(state)

    def _range_cache(self, start: int, stop: int) -> KernelLODCache:
        if not 0 <= start < stop <= self.max_requests:
            raise IndexError("LOD cache row range is outside the fixed pool")
        metadata = self.metadata[start]
        scalar_names = (
            "state_len",
            "scheduled_state_len",
            "coverage",
            "recent_len",
            "total_len",
            "leaf_count",
            "overflow_safe_until",
        )
        for slot in range(start + 1, stop):
            if any(
                int(self.metadata[slot][name]) != int(metadata[name])
                for name in scalar_names
            ):
                raise ValueError("batched LOD catch-up rows have different metadata")
        state: dict[str, object] = {
            name: value[start:stop]
            for name, value in self.state.items()
            if isinstance(value, torch.Tensor) and value.ndim
        }
        state.update(
            state_len=int(metadata["state_len"]),
            scheduled_state_len=int(
                metadata.get("scheduled_state_len", metadata["state_len"])
            ),
            coverage=int(metadata["coverage"]),
            state_capacity=self.state_capacity,
            recent_len=int(metadata["recent_len"]),
            total_len=int(metadata["total_len"]),
        )
        page_pool = self.state["page_cache"]
        page: dict[str, object] = {}
        for name, value in page_pool.items():
            if name.startswith("unified_page1_"):
                continue
            page[name] = (
                value[start:stop]
                if isinstance(value, torch.Tensor) and value.ndim
                else value
            )
        page.update(
            leaf_count=int(metadata["leaf_count"]),
            leaf_capacity=self.leaf_capacity,
            overflow_active=True,
            overflow_safe_until=int(metadata["overflow_safe_until"]),
        )
        state["page_cache"] = page
        return KernelLODCache(state)

    def _catch_up_target(
        self, slot: int, total_length: int
    ) -> tuple[int, int]:
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        recent_length = total_length - coverage
        if recent_length < 0 or recent_length > self.local_capacity:
            raise ValueError("decode-local length exceeds its fixed cache row")
        update_len = int(self.engine.decode_state_update_len)
        exact_floor = int(self.engine.local_len - self.engine.chunk_len)
        target_coverage = max(min(total_length, self.engine.chunk_len), coverage)
        pending_update = total_length + 1 - target_coverage - exact_floor
        if pending_update > update_len:
            target_coverage += ((pending_update - 1) // update_len) * update_len
        return recent_length, min(target_coverage, total_length)

    def catch_up(self, slot: int, total_length: int) -> None:
        if not self.ready[slot]:
            raise RuntimeError("cannot catch up an uninitialized LOD request row")
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        recent_length, target_coverage = self._catch_up_target(slot, total_length)
        if coverage >= target_coverage:
            # Captured decode already appended K/V and advanced local_lens on
            # device. Most tokens need only this host metadata bookkeeping.
            metadata["total_len"] = total_length
            metadata["recent_len"] = recent_length
            return
        row = self._row_cache(slot)
        self.engine.catch_up_cache(
            row, total_length=total_length, recent_length=recent_length
        )
        page = row.state["page_cache"]
        self.metadata[slot].update(
            state_len=int(row.state["state_len"]),
            scheduled_state_len=int(
                row.state.get("scheduled_state_len", row.state["state_len"])
            ),
            coverage=int(row.state["coverage"]),
            total_len=int(row.state["total_len"]),
            recent_len=int(row.state["recent_len"]),
            leaf_count=int(page["leaf_count"]),
            overflow_safe_until=int(page["overflow_safe_until"]),
        )
        self.local_lens[slot].fill_(int(row.state["recent_len"]))
        self._record_split_state(row.state, page)
        self._refresh_unified_page1_coarse((slot,))

    def catch_up_many(self, requests: list[tuple[int, int]]) -> None:
        """Batch equal-metadata contiguous rows at a state-update boundary."""
        pending: dict[tuple[int, ...], list[int]] = {}
        for slot, total_length in requests:
            if not self.ready[slot]:
                raise RuntimeError("cannot catch up an uninitialized LOD request row")
            metadata = self.metadata[slot]
            recent_length, target_coverage = self._catch_up_target(
                slot, total_length
            )
            if int(metadata["coverage"]) >= target_coverage:
                metadata["total_len"] = total_length
                metadata["recent_len"] = recent_length
                continue
            signature = (
                total_length,
                int(metadata["state_len"]),
                int(metadata.get("scheduled_state_len", metadata["state_len"])),
                int(metadata["coverage"]),
                int(metadata["recent_len"]),
                int(metadata["leaf_count"]),
                int(metadata["overflow_safe_until"]),
            )
            pending.setdefault(signature, []).append(slot)

        for signature, slots in pending.items():
            total_length = signature[0]
            slots.sort()
            begin = 0
            while begin < len(slots):
                end = begin + 1
                while end < len(slots) and slots[end] == slots[end - 1] + 1:
                    end += 1
                start_slot = slots[begin]
                stop_slot = slots[end - 1] + 1
                row = self._range_cache(start_slot, stop_slot)
                recent_length = total_length - int(row.state["coverage"])
                self.engine.catch_up_cache(
                    row,
                    total_length=total_length,
                    recent_length=recent_length,
                )
                self.catch_up_batches += 1
                self.catch_up_rows += stop_slot - start_slot
                page = row.state["page_cache"]
                for slot in range(start_slot, stop_slot):
                    self.metadata[slot].update(
                        state_len=int(row.state["state_len"]),
                        scheduled_state_len=int(
                            row.state.get(
                                "scheduled_state_len", row.state["state_len"]
                            )
                        ),
                        coverage=int(row.state["coverage"]),
                        total_len=int(row.state["total_len"]),
                        recent_len=int(row.state["recent_len"]),
                        leaf_count=int(page["leaf_count"]),
                        overflow_safe_until=int(page["overflow_safe_until"]),
                    )
                self.local_lens[start_slot:stop_slot].fill_(
                    int(row.state["recent_len"])
                )
                self._record_split_state(row.state, page)
                self._refresh_unified_page1_coarse(
                    tuple(range(start_slot, stop_slot))
                )
                begin = end

    def _buffers(self, query: torch.Tensor, rows: int) -> dict[str, torch.Tensor]:
        buffers = self.decode_buffers.get(rows)
        storage = self.decode_buffer_storage
        if storage is None or storage["partial_out"].device != query.device:
            template = query.new_empty(
                self.max_requests,
                self.query_heads,
                1,
                self.head_dim,
            )
            storage = new_fused_decode_buffers(
                template,
                splits=int(self.engine.decode_split_kv),
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
                route_segment_tiles=int(self.engine.decode_route_segment_tiles),
                gqa_route_splits=(
                    self._decode_route_splits()
                    if self._use_cooperative_decode()
                    else None
                ),
                materialized_state_route=(
                    self.engine.recursive_state_route_backend == "resplit"
                ),
                gqa_union_mass_fraction=(
                    self.settings.decode_gqa_mass_fraction
                    if self.settings.decode_gqa_union
                    else None
                ),
                gqa_union_predicted_mass=(
                    self.settings.decode_gqa_predicted_mass
                    if self.settings.decode_gqa_union
                    else False
                ),
                gqa_union_pilot_z=(
                    self.settings.decode_gqa_pilot_z
                    if self.settings.decode_gqa_union
                    else False
                ),
                gqa_union_pilot_z_route_count=(
                    self.settings.decode_gqa_pilot_z_route_count
                    if self.settings.decode_gqa_union
                    else 8
                ),
                gqa_union_kv_heads=(
                    self.kv_heads
                    if self.settings.decode_gqa_union
                    and self.settings.levels == 2
                    and self.query_heads % self.kv_heads == 0
                    and 1 < self.query_heads // self.kv_heads <= 16
                    and self.head_dim in (128, 256, 512)
                    and self.dtype == torch.bfloat16
                    else None
                ),
                gqa_union_index_capacity=(
                    self.leaf_capacity
                    + int(self.engine.local_len)
                    + 1
                    + self.state_capacity
                    + (
                        int(self.state["sink_k"].size(2))
                        if isinstance(self.state.get("sink_k"), torch.Tensor)
                        else 0
                    )
                    if self.settings.decode_gqa_union
                    and self.settings.levels == 2
                    and self.query_heads % self.kv_heads == 0
                    and 1 < self.query_heads // self.kv_heads <= 16
                    and self.head_dim in (128, 256, 512)
                    and self.dtype == torch.bfloat16
                    else None
                ),
                gqa_union_hip=(
                    self.settings.decode_gqa_union_hip
                    if self.settings.decode_gqa_union
                    else False
                ),
                gqa_union_fixed_mask=bool(
                    self.settings.decode_gqa_fixed_mask_aiter
                    and self.settings.decode_gqa_union
                ),
                gqa_union_overlap_local_sink=bool(
                    self.settings.decode_gqa_overlap_local_sink
                    and self.settings.decode_gqa_union
                ),
                gqa_union_static_cap_page1=bool(
                    self.settings.decode_gqa_static_leaf_aiter
                    and self.settings.decode_gqa_union
                ),
                gqa_union_fixed_mask_tile_size=int(
                    self.settings.decode_gqa_fixed_mask_block_n
                ),
                gqa_union_fixed_mask_segments=int(
                    max(
                        self.settings.decode_gqa_fixed_mask_segments,
                        256,
                    )
                    if (
                        self.settings.decode_gqa_fixed_mask_adaptive_segments
                    )
                    else self.settings.decode_gqa_fixed_mask_segments
                ),
            )
            if bool(self.engine.recursive_materialize_page_scores):
                storage["recursive_page_scores"] = torch.empty(
                    self.max_requests,
                    self.query_heads,
                    1,
                    self.page_capacity,
                    dtype=torch.float32,
                    device=self.device,
                )
            if (
                self.head_dim in (128, 256, 512)
                and 1 < self.query_heads // self.kv_heads <= 16
                and not bool(self.engine.recursive_materialize_page_scores)
            ):
                # The wide-head local decoder materializes one regular GQA
                # score field so both QK and PV use MFMA without reloading K/V
                # independently for every query head. Recursive materialized
                # decode reuses its larger page-score field for this earlier,
                # non-overlapping phase rather than reserving another buffer.
                storage["wide_gqa_local_scores"] = torch.empty(
                    self.max_requests,
                    self.query_heads,
                    int(self.engine.local_len) + 1,
                    dtype=torch.float32,
                    device=self.device,
                )
            self.decode_buffer_storage = storage
            self.decode_buffers.clear()
            buffers = None
        if buffers is None:
            buffers = {
                name: (
                    tensor[:rows]
                    if (
                        name != "route_pilot_z_thresholds"
                        and tensor.ndim
                        and int(tensor.size(0)) == self.max_requests
                    )
                    else tensor
                )
                for name, tensor in storage.items()
            }
            self.decode_buffers[rows] = buffers
        return buffers

    def _decode_route_splits(self) -> int:
        configured = self.settings.decode_gqa_route_splits
        if configured is not None:
            return configured
        split_work = max(1, self.request_capacity // 4096)
        return max(8, min(32, 1 << (split_work.bit_length() - 1)))

    def _use_cooperative_decode(self) -> bool:
        if (
            self.settings.levels != 2
            or not self.settings.decode_gqa_cooperative
            or not self.settings.decode_gqa_cooperative_hip
            or self.query_heads != self.kv_heads * 4
            or self.head_dim != 256
            or self.dtype != torch.bfloat16
        ):
            return False
        if self.request_capacity < max(32768, 4096 * self.max_requests):
            return False
        from model.kernels.gqa_cooperative_decode import (
            gqa_cooperative_decode_available,
        )

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return gqa_cooperative_decode_available(device_index)

    def reserve_decode_buffers(self, rows: int) -> None:
        """Reserve graph scratch before vLLM computes its native cache budget."""
        if not 1 <= rows <= self.max_requests:
            raise ValueError("decode scratch rows exceed the fixed LOD pool")
        query = torch.empty(
            rows,
            self.query_heads,
            1,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._buffers(query, rows)

    def reserve_speculative_decode_buffers(self, rows: int, steps: int) -> None:
        """Reserve fixed-address request-major/step-major graph staging.

        vLLM lays uniform speculative verification out request-major, while
        the graph-safe LOD primitive advances one token for every request at a
        time.  These tiny staging tensors make that transpose explicit and,
        crucially, keep every pointer stable across graph replay.
        """
        if not 1 <= rows <= self.max_requests:
            raise ValueError("speculative scratch rows exceed the fixed LOD pool")
        if steps <= 1:
            raise ValueError("speculative decode requires at least two steps")
        signature = (rows, steps)
        if signature in self.speculative_decode_buffers:
            return
        self.reserve_decode_buffers(rows)
        staging: dict[str, Any] = {
            "q": torch.empty(
                steps,
                rows,
                self.query_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            "k": torch.empty(
                steps,
                rows,
                self.kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            "v": torch.empty(
                steps,
                rows,
                self.kv_heads,
                self.value_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            "out": torch.empty(
                steps,
                rows,
                self.query_heads,
                self.value_dim,
                dtype=self.dtype,
                device=self.device,
            ),
        }
        if self._parallel_speculative_decode_eligible(steps):
            total_rows = rows * steps
            parallel_steps = self._parallel_speculative_chunk_steps(steps, rows)
            parallel_rows = rows * parallel_steps
            speculative_route_backend = (
                self._speculative_recursive_state_route_backend()
            )
            template = torch.empty(
                parallel_rows,
                self.query_heads,
                1,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            staging["cache_indices"] = torch.empty(
                total_rows, dtype=torch.long, device=self.device
            )
            staging["local_lens"] = torch.empty(
                total_rows, dtype=torch.int32, device=self.device
            )
            staging["decode_buffers"] = new_fused_decode_buffers(
                template,
                splits=int(self.engine.decode_split_kv),
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
                route_segment_tiles=int(self.engine.decode_route_segment_tiles),
                gqa_route_splits=(
                    int(self.engine.decode_split_kv)
                    if self._speculative_cooperative_leaf_eligible(steps)
                    else None
                ),
                materialized_state_route=bool(
                    self.settings.levels == 3
                    and speculative_route_backend == "resplit"
                ),
                gqa_union_mass_fraction=(
                    self.settings.decode_gqa_mass_fraction
                    if self.settings.decode_gqa_union
                    else None
                ),
                gqa_union_predicted_mass=(
                    self.settings.decode_gqa_predicted_mass
                    if self.settings.decode_gqa_union
                    else False
                ),
                gqa_union_pilot_z=(
                    self.settings.decode_gqa_pilot_z
                    if self.settings.decode_gqa_union
                    else False
                ),
                gqa_union_pilot_z_route_count=(
                    self.settings.decode_gqa_pilot_z_route_count
                    if self.settings.decode_gqa_union
                    else 8
                ),
                gqa_union_kv_heads=(
                    self.kv_heads
                    if self._speculative_fixed_mask_eligible(steps)
                    else None
                ),
                gqa_union_index_capacity=(
                    self.leaf_capacity
                    + int(self.engine.local_len)
                    + 1
                    + self.state_capacity
                    + (
                        int(self.state["sink_k"].size(2))
                        if isinstance(self.state.get("sink_k"), torch.Tensor)
                        else 0
                    )
                    if self._speculative_fixed_mask_eligible(steps)
                    else None
                ),
                gqa_union_hip=(
                    self.settings.decode_gqa_union_hip
                    if self.settings.decode_gqa_union
                    else False
                ),
                gqa_union_fixed_mask=self._speculative_fixed_mask_eligible(
                    steps
                ),
                gqa_union_overlap_local_sink=bool(
                    self.settings.decode_gqa_overlap_local_sink
                    and self._speculative_fixed_mask_eligible(steps)
                ),
                gqa_union_fixed_mask_tile_size=int(
                    self.settings.decode_gqa_fixed_mask_block_n
                ),
                gqa_union_fixed_mask_segments=int(
                    max(
                        self.settings.decode_gqa_fixed_mask_segments,
                        256,
                    )
                    if (
                        self.settings.decode_gqa_fixed_mask_adaptive_segments
                        or self._speculative_fixed_mask_adaptive_segments(steps)
                    )
                    else self.settings.decode_gqa_fixed_mask_segments
                ),
            )
            if self.settings.levels == 3:
                if bool(self.engine.recursive_materialize_page_scores):
                    staging["decode_buffers"]["recursive_page_scores"] = (
                        torch.empty(
                            parallel_rows,
                            self.query_heads,
                            1,
                            self.page_capacity,
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                elif (
                    self.head_dim in (128, 256, 512)
                    and 1 < self.query_heads // self.kv_heads <= 16
                ):
                    staging["decode_buffers"]["wide_gqa_local_scores"] = (
                        torch.empty(
                            parallel_rows,
                            self.query_heads,
                            int(self.engine.local_len) + 1,
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
            staging["decode_buffers"]["speculative_parallel_execution_marker"] = (
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            staging["decode_buffers"]["speculative_local_partial_out"] = (
                torch.empty_like(staging["decode_buffers"]["partial_out"])
            )
            staging["decode_buffers"]["speculative_local_partial_lse"] = (
                torch.empty_like(staging["decode_buffers"]["partial_lse"])
            )
            staging["decode_buffers"]["speculative_route_execution_marker"] = (
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            staging["decode_buffers"]["speculative_local_execution_marker"] = (
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
        self.speculative_decode_buffers[signature] = staging

    def _parallel_speculative_chunk_steps(
        self, steps: int, rows: int = 1
    ) -> int:
        """Bound one recursive flattened verifier launch to a tested depth.

        Original Gemma DFlash supplies sixteen positions. B1 therefore stays
        one launch, while the conservative D=512 B8 default consumes the
        immutable remote state in four four-position chunks rather than
        falling back to sixteen serial verifier calls. All proposal K/V remains
        staged at once. Narrower Qwen DFlash2 retains its validated 64-row
        bound; Gemma can opt into that bound for individually validated
        high-throughput profiles.
        """
        if self.settings.levels != 3:
            return steps
        default_maximum_rows = 32 if self.head_dim >= 512 else 64
        maximum_rows = int(
            os.getenv(
                "VLLM_LOD_SPECULATIVE_PARALLEL_MAX_ROWS",
                str(default_maximum_rows),
            )
        )
        if maximum_rows < 1:
            raise ValueError(
                "VLLM_LOD_SPECULATIVE_PARALLEL_MAX_ROWS must be positive"
            )
        maximum = min(steps, max(1, maximum_rows // rows))
        while steps % maximum:
            maximum -= 1
        return maximum

    def _parallel_speculative_decode_eligible(self, steps: int) -> bool:
        """Whether one flattened launch can verify all proposal positions."""
        recursive = self.settings.levels == 3
        two_level = bool(
            self.settings.levels == 2
            and (
                (steps == 2 and not self.settings.decode_gqa_union)
                or self._speculative_fixed_mask_eligible(steps)
            )
        )
        return bool(
            os.getenv("VLLM_LOD_SPECULATIVE_PARALLEL", "1") != "0"
            and steps >= 2
            and (recursive or two_level)
            and not self.settings.decode_gqa_static_leaf_aiter
            and not self._use_cooperative_decode()
        )

    def _speculative_recursive_state_route_backend(self) -> str:
        """Resolve the recursive route used inside speculative verification.

        The materialized re-split route remains useful for ordinary decode,
        but its long-context score-table pipeline is not yet safe under
        speculative verification (including the serial verifier control).
        Keep the ordinary per-model policy unchanged and default only the
        speculative recursive path to the grouped producer. The override is
        retained for targeted re-split validation.
        """
        if self.settings.levels != 3:
            return str(self.engine.recursive_state_route_backend)
        backend = os.getenv(
            "VLLM_LOD_SPECULATIVE_RECURSIVE_STATE_ROUTE_BACKEND",
            "fused",
        )
        if backend not in ("fused", "resplit"):
            raise ValueError(
                "speculative recursive state-route backend must be fused or resplit"
            )
        return backend

    def _speculative_fixed_mask_eligible(self, steps: int) -> bool:
        """Whether MTP can reuse the persistent masked page-size-one arena."""
        return bool(
            os.getenv("VLLM_LOD_SPECULATIVE_FIXED_MASK_AITER", "1") != "0"
            and steps >= 2
            and self.settings.levels == 2
            and self.settings.decode_gqa_union
            and self.settings.decode_gqa_union_hip
            and self.settings.decode_gqa_fixed_mask_aiter
            and not self.settings.decode_gqa_static_leaf_aiter
            and self.query_heads % self.kv_heads == 0
            and 1 < self.query_heads // self.kv_heads <= 8
            and self.head_dim in (128, 256, 512)
            and self.dtype == torch.bfloat16
        )

    def _speculative_fixed_mask_adaptive_segments(self, steps: int) -> bool:
        """Use the measured low-batch scan geometry for fixed-mask MTP."""
        return bool(
            self._speculative_fixed_mask_eligible(steps)
            and os.getenv(
                "VLLM_LOD_SPECULATIVE_FIXED_MASK_ADAPTIVE_SEGMENTS", "1"
            )
            != "0"
        )

    def _shared_speculative_route_eligible(
        self, steps: int, rows: int = 1
    ) -> bool:
        """Whether proposal positions fit pairwise native grouped route tiles."""
        return bool(
            self._parallel_speculative_decode_eligible(steps)
            and self._parallel_speculative_chunk_steps(steps, rows) == steps
            and steps % 2 == 0
            and os.getenv("VLLM_LOD_SPECULATIVE_SHARED_ROUTE", "1") != "0"
            and bool(self.engine.decode_route_gqa_grouped)
            and int(self.engine.decode_route_segment_tiles) == 1
            and 2 * (self.query_heads // self.kv_heads) <= 16
        )

    def _speculative_cooperative_leaf_eligible(self, steps: int) -> bool:
        """Whether exact pages can be shared by both MTP GQA6 groups."""
        return bool(
            self._parallel_speculative_decode_eligible(steps)
            and steps == 2
            and os.getenv("VLLM_LOD_SPECULATIVE_COOPERATIVE_LEAVES", "0") != "0"
            and self.settings.decode_gqa_cooperative
            and self.settings.decode_gqa_cooperative_hip
            and self.query_heads == self.kv_heads * 6
            and self.head_dim == 256
            and self.dtype == torch.bfloat16
            and self.request_capacity >= 32768
        )

    def _shared_speculative_local_eligible(
        self, steps: int, rows: int = 1
    ) -> bool:
        """Whether pairwise routing can also absorb causal local attention."""
        return bool(
            self._shared_speculative_route_eligible(steps, rows)
            and os.getenv("VLLM_LOD_SPECULATIVE_SHARED_LOCAL", "1") != "0"
            and os.getenv("VLLM_LOD_SPECULATIVE_FUSE_LOCAL_ROUTE", "1") != "0"
            # Fixed-mask page-size-one attention already consumes the causal
            # local prefix in its unified final scan; a separate shared-local
            # launch would be computed and discarded.
            and not self._speculative_fixed_mask_eligible(steps)
        )

    def speculative_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Verify a uniform proposal inside one replayable target graph.

        The proposal queries route concurrently against the immutable remote
        LOD state. Their K/V are staged before that launch, so each query sees
        the exact causal local suffix through its own logical length. The host
        truncates rejected tail entries by resetting ``local_lens`` before the
        next replay; no route is lagged.
        """
        steps = int(self.speculative_decode_steps)
        if steps <= 1:
            raise RuntimeError("speculative LOD decode was not prepared")
        total_tokens = int(query.size(0))
        if total_tokens % steps:
            raise ValueError(
                "uniform speculative tokens do not divide the padded batch"
            )
        rows = total_tokens // steps
        signature = (rows, steps)
        staging = self.speculative_decode_buffers.get(signature)
        if staging is None:
            raise RuntimeError(
                "speculative LOD graph staging was not reserved before forward"
            )

        # One copy per tensor changes vLLM's request-major layout [B, M, H, D]
        # into step-major [M, B, H, D].  The attention calls below then use
        # contiguous M=1 inputs and outputs without per-step packing kernels.
        staging["q"].copy_(
            query[: rows * steps]
            .view(rows, steps, self.query_heads, self.head_dim)
            .permute(1, 0, 2, 3)
        )
        staging["k"].copy_(
            key[: rows * steps]
            .view(rows, steps, self.kv_heads, self.head_dim)
            .permute(1, 0, 2, 3)
        )
        staging["v"].copy_(
            value[: rows * steps]
            .view(rows, steps, self.kv_heads, self.value_dim)
            .permute(1, 0, 2, 3)
        )

        if self._parallel_speculative_decode_eligible(steps):
            staging["decode_buffers"][
                "speculative_parallel_execution_marker"
            ].add_(1)
            flat_q = staging["q"].view(
                rows * steps, self.query_heads, self.head_dim
            )
            flat_k = staging["k"].view(
                rows * steps, self.kv_heads, self.head_dim
            )
            flat_v = staging["v"].view(
                rows * steps, self.kv_heads, self.value_dim
            )
            flat_out = staging["out"].view(
                rows * steps, self.query_heads, self.value_dim
            )
            prepare_speculative_decode_kv(
                self.active_indices[:rows],
                self.local_lens,
                flat_k.unsqueeze(2),
                flat_v.unsqueeze(2),
                self.state["recent_k"],
                self.state["recent_v"],
                staging["cache_indices"],
                staging["local_lens"],
                rows=rows,
                steps=steps,
            )

            parallel_steps = self._parallel_speculative_chunk_steps(steps, rows)

            class _ParallelMetadata:
                num_actual_tokens = rows * parallel_steps

            for step_begin in range(0, steps, parallel_steps):
                begin = step_begin * rows
                end = begin + parallel_steps * rows
                self.decode(
                    flat_q[begin:end],
                    flat_k[begin:end],
                    flat_v[begin:end],
                    _ParallelMetadata(),
                    flat_out[begin:end],
                    cache_indices=staging["cache_indices"][begin:end],
                    local_lens=staging["local_lens"][begin:end],
                    decode_buffers=staging["decode_buffers"],
                    local_lens_are_logical=True,
                    store_new_kv=False,
                    advance_local_lens=False,
                    speculative_steps=(
                        steps
                        if parallel_steps == steps
                        and (
                            self._shared_speculative_route_eligible(steps, rows)
                            or self._speculative_fixed_mask_eligible(steps)
                        )
                        else 1
                    ),
                    recursive_state_route_backend=(
                        self._speculative_recursive_state_route_backend()
                    ),
                )
            advance_decode_cache_lengths(
                self.active_indices[:rows], self.local_lens, increment=steps
            )
            output[: rows * steps].view(
                rows, steps, self.query_heads, self.value_dim
            ).copy_(staging["out"].permute(1, 0, 2, 3))
            return output

        class _Metadata:
            num_actual_tokens = rows

        metadata = _Metadata()
        for step in range(steps):
            self.decode(
                staging["q"][step],
                staging["k"][step],
                staging["v"][step],
                metadata,
                staging["out"][step],
                recursive_state_route_backend=(
                    self._speculative_recursive_state_route_backend()
                ),
            )
        output[: rows * steps].view(
            rows, steps, self.query_heads, self.value_dim
        ).copy_(staging["out"].permute(1, 0, 2, 3))
        return output

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Any,
        output: torch.Tensor,
        *,
        cache_indices: torch.Tensor | None = None,
        local_lens: torch.Tensor | None = None,
        decode_buffers: dict[str, torch.Tensor] | None = None,
        local_lens_are_logical: bool = False,
        store_new_kv: bool = True,
        advance_local_lens: bool = True,
        speculative_steps: int = 1,
        recursive_state_route_backend: str | None = None,
    ) -> torch.Tensor:
        self.decode_calls += 1
        rows = int(metadata.num_actual_tokens)
        if rows == 0:
            return output
        q = query[:rows].unsqueeze(2)
        k = key[:rows].unsqueeze(2)
        v = value[:rows].unsqueeze(2)
        if cache_indices is None:
            cache_indices = self.active_indices[:rows]
        if local_lens is None:
            local_lens = self.local_lens
        if decode_buffers is None:
            decode_buffers = self._buffers(q, rows)
        page = self.state["page_cache"]
        recursive = self.settings.levels == 3
        indexed_flat = (
            not recursive and isinstance(page.get("page_indices"), torch.Tensor)
        )
        page_k = page["leaf_k"] if recursive or indexed_flat else page["page_k"]
        page_v = page["leaf_v"] if recursive or indexed_flat else page["page_v"]
        flat_int8 = not recursive and (
            page_k.dtype == torch.int8 or page_v.dtype == torch.int8
        )
        if self.settings.decode_gqa_static_leaf_aiter and isinstance(
            page.get("unified_page1_fixed_indices"), torch.Tensor
        ):
            result = static_cap_page1_decode_attention(
                q,
                k,
                v,
                cache_indices=self.active_indices[:rows],
                local_lens=self.local_lens,
                fixed_indices=page["unified_page1_fixed_indices"],
                fixed_base_lengths=page["unified_page1_fixed_lengths"],
                arena_k=page["unified_page1_k"],
                arena_v=page["unified_page1_v"],
                arena_bias=page["unified_page1_bias"],
                arena_local_offset=int(page["unified_page1_local_offset"]),
                local_capacity=self.local_capacity,
                local_limit=int(self.engine.local_len),
                kv_heads=self.kv_heads,
                scale=float(self.engine.scaling),
                buffers=self._buffers(q, rows),
                output=output[:rows].unsqueeze(2),
                preselected_only=self.settings.diagnostic_static_preselected,
                timing_events=getattr(
                    self.engine, "_lod_decode_timing_events", None
                ),
            )
            if result.data_ptr() != output.data_ptr():
                raise AssertionError(
                    "static page-size-one decode did not use the vLLM output buffer"
                )
            return output
        result = fused_decode_paged_lod_attention(
            q,
            self.state["state_k"],
            self.state["state_v"],
            self.state["counts"],
            self.state["recent_k"],
            self.state["recent_v"],
            page_k,
            page_v,
            page["slot_pages"],
            page["overflow_page_keys"],
            page["overflow_page_values"],
            page["overflow_used"],
            page["slot_lengths"],
            None,
            sink_k=self.state.get("sink_k"),
            sink_v=self.state.get("sink_v"),
            state_len=self.state_capacity,
            # Prefill keeps a larger exact lookback in the same backing row,
            # but catch-up keeps each decode query's live tail within the
            # normal local window. The extra storage only receives the current
            # token before the next scheduled catch-up.
            local_len=int(self.engine.local_len),
            cache_indices=cache_indices,
            local_lens=local_lens,
            new_k=k,
            new_v=v,
            local_lens_are_logical=local_lens_are_logical,
            store_new_kv=store_new_kv,
            advance_local_lens=advance_local_lens,
            speculative_steps=speculative_steps,
            kv_group_size=self.query_heads // self.kv_heads,
            scale=float(self.engine.scaling),
            hash_probes=int(self.engine._page_lookup_probes(page)),
            block_n=int(self.engine.decode_block_n),
            num_warps=int(self.engine.decode_num_warps),
            waves_per_eu=int(self.engine.leaf_waves_per_eu),
            split_kv=int(self.engine.decode_split_kv),
            buffers=decode_buffers,
            use_dot=bool(self.engine.decode_use_dot),
            fuse_state_route=True,
            route_group_size=int(self.engine.decode_route_group_size),
            route_segment_tiles=int(self.engine.decode_route_segment_tiles),
            route_num_warps=int(self.engine.decode_route_num_warps),
            route_reduce_num_warps=int(self.engine.decode_route_reduce_num_warps),
            route_parallel_reduce=bool(self.engine.decode_route_parallel_reduce),
            route_post_dot_normalize=bool(
                self.engine.decode_route_post_dot_normalize
            ),
            route_post_pv_normalize=bool(
                self.engine.decode_route_post_pv_normalize
            ),
            final_reduce_num_warps=int(self.engine.decode_final_reduce_num_warps),
            fuse_final_reduce=bool(self.engine.decode_fuse_final_reduce),
            route_use_dot=bool(self.engine.decode_route_use_dot),
            route_gqa_grouped=bool(self.engine.decode_route_gqa_grouped),
            route_centroid_major_hip=bool(
                self.settings.decode_centroid_major_hip
            ),
            gqa_cooperative_leaf=(
                self._use_cooperative_decode()
                or self._speculative_cooperative_leaf_eligible(speculative_steps)
            ),
            gqa_cooperative_hip=bool(
                self.settings.decode_gqa_cooperative_hip
            ),
            gqa_union_decode=bool(self.settings.decode_gqa_union),
            gqa_union_mass_fraction=self.settings.decode_gqa_mass_fraction,
            gqa_union_predicted_mass=bool(
                self.settings.decode_gqa_predicted_mass
            ),
            gqa_union_pilot_z=bool(self.settings.decode_gqa_pilot_z),
            gqa_union_pilot_z_margin=float(
                self.settings.decode_gqa_pilot_z_margin
            ),
            gqa_union_hip=bool(self.settings.decode_gqa_union_hip),
            gqa_union_staged_fixed_aiter=bool(
                self.settings.decode_gqa_staged_fixed_aiter
            ),
            gqa_union_fixed_mask_aiter=bool(
                self.settings.decode_gqa_fixed_mask_aiter
            ),
            gqa_union_overlap_local_sink=bool(
                self.settings.decode_gqa_overlap_local_sink
            ),
            gqa_union_fixed_mask_tile_size=int(
                self.settings.decode_gqa_fixed_mask_block_n
            ),
            gqa_union_fixed_mask_adaptive_segments=bool(
                self.settings.decode_gqa_fixed_mask_adaptive_segments
                or self._speculative_fixed_mask_adaptive_segments(
                    speculative_steps
                )
            ),
            gqa_union_fixed_mask_reduce_block_d=int(
                self.settings.decode_gqa_fixed_mask_reduce_block_d
            ),
            gqa_union_fixed_mask_direct_routes=bool(
                self.settings.decode_gqa_fixed_mask_direct_routes
            ),
            gqa_union_fixed_mask_scan_num_warps=int(
                self.settings.decode_gqa_fixed_mask_scan_num_warps
            ),
            gqa_union_fixed_mask_scan_waves_per_eu=int(
                self.settings.decode_gqa_fixed_mask_scan_waves_per_eu
            ),
            gqa_union_fixed_mask_scan_num_stages=int(
                self.settings.decode_gqa_fixed_mask_scan_num_stages
            ),
            gqa_union_static_leaf_cap=(
                self.settings.decode_gqa_static_leaf_cap
                if (
                    self.settings.decode_gqa_fixed_mask_aiter
                    and not self.settings.decode_route_cohort
                )
                else None
            ),
            gqa_union_page1_k=page.get("unified_page1_k"),
            gqa_union_page1_v=page.get("unified_page1_v"),
            gqa_union_page1_bias=page.get("unified_page1_bias"),
            gqa_union_page1_leaf_offset=int(
                page.get("unified_page1_leaf_offset", 0)
            ),
            gqa_union_page1_local_offset=int(
                page.get("unified_page1_local_offset", 0)
            ),
            gqa_union_page1_sink_offset=int(
                page.get("unified_page1_sink_offset", 0)
            ),
            gqa_union_page1_coarse_offset=int(
                page.get("unified_page1_coarse_offset", 0)
            ),
            gqa_union_previous_total_lse=page.get(
                "decode_previous_total_lse"
            ),
            gqa_union_pilot_z_bound=page.get("decode_pilot_z_bound"),
            gqa_union_fixed_indices=page.get(
                "unified_page1_fixed_indices"
            ),
            gqa_union_fixed_leaf_owners=page.get(
                "unified_page1_fixed_leaf_owners"
            ),
            gqa_union_fixed_slot_offsets=page.get(
                "unified_page1_fixed_slot_offsets"
            ),
            gqa_union_fixed_lengths=page.get(
                "unified_page1_fixed_lengths"
            ),
            gqa_cooperative_route_splits=(
                int(self.engine.decode_split_kv)
                if self._speculative_cooperative_leaf_eligible(speculative_steps)
                else self._decode_route_splits()
            ),
            protected_len=self.engine._protected_state_len(self.state_capacity),
            # Routing-only guard: large centroids remain live and keep being
            # updated, but decode represents them by their coarse entry instead
            # of opening an unbounded exact posting list.
            max_leaf_tokens=self._decode_route_leaf_limit(),
            open_count=int(self.settings.open_count),
            recursive_page_cache=page if recursive else None,
            flat_page_indices=page["page_indices"] if indexed_flat else None,
            flat_page_k_scales=(
                page.get("page_k_token_scales") if flat_int8 else None
            ),
            flat_page_v_scales=(
                page.get("page_v_token_scales") if flat_int8 else None
            ),
            recursive_quant_group_size=int(self.engine.leaf_quant_group_size),
            recursive_quant_token_group_size=int(
                self.engine.leaf_quant_token_group_size
            ),
            timing_events=getattr(self.engine, "_lod_decode_timing_events", None),
            recursive_materialize_page_scores=bool(
                self.engine.recursive_materialize_page_scores
            ),
            recursive_page_score_block_n=int(
                self.engine.recursive_page_score_block_n
            ),
            recursive_page_score_num_warps=int(
                self.engine.recursive_page_score_num_warps
            ),
            recursive_page_select_block_n=int(
                self.engine.recursive_page_select_block_n
            ),
            recursive_state_route_backend=(
                self.engine.recursive_state_route_backend
                if recursive_state_route_backend is None
                else recursive_state_route_backend
            ),
            output_buffer=output[:rows].unsqueeze(2),
        )
        if result.data_ptr() != output.data_ptr():
            raise AssertionError("fused LOD decode did not use the vLLM output buffer")
        return output


__all__ = ["VLLMLayerLODPool"]
