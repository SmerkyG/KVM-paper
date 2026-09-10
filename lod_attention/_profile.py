"""One production LoD policy, with geometry-only kernel specialization."""

from __future__ import annotations

from typing import Any

from ._config import (
    LODMode,
    ModelFamily,
    PREFILL_CHUNK_SIZE,
    PREFILL_LOCAL_WINDOW,
    ROUTE_COUNT,
)


def configure_engine(
    engine: Any,
    *,
    family: ModelFamily,
    mode: LODMode,
    request_capacity: int,
    has_query_norm: bool,
    has_key_norm: bool,
) -> None:
    """Apply the measured release profile to a projection-free engine.

    The attention calculation is identical for both families: top-four in
    prefill and decode, count-corrected coarse mass, and exact replacement of
    selected regions. Differences below are only launch geometry or storage.
    """

    gqa = engine.config.num_attention_heads // engine.config.num_key_value_heads
    head_dim = getattr(engine, "head_dim", None)
    if head_dim is None:
        # The engine learns D from its first tensors. Model-family validation
        # already fixes the only supported shapes, so use their known widths.
        head_dim = 256 if family is ModelFamily.QWEN38 else 128
    expected = (256, 6) if family is ModelFamily.QWEN38 else (128, 8)
    if (int(head_dim), int(gqa)) != expected:
        raise ValueError(
            f"{family.value} requires D/GQA={expected}, got {(head_dim, gqa)}"
        )

    engine.two_level_topk = ROUTE_COUNT
    engine.prefill_two_level_topk = ROUTE_COUNT
    engine.separate_sink_cache = True
    engine.prefill_chunk_len = PREFILL_CHUNK_SIZE
    engine.prefill_local_len = PREFILL_LOCAL_WINDOW
    engine.prefill_state_update_len = PREFILL_CHUNK_SIZE
    engine.prefill_exact_first_chunk = True
    engine.split_prefill_local_attention = True
    engine.prefill_local_attention_backend = "aiter"
    engine.fused_prefill_route_coarse = True
    engine.fused_prefill_stable_recompute = True
    engine.fused_prefill_external_recompute = True
    engine.prefill_hierarchical_route = True
    engine.prefill_overlap_coarse_leaf = True
    engine.prefill_overlap_local_lod = False
    engine.fused_state_update = True
    engine.fused_state_maxsim = True

    engine.state_clustering_normalization = "none" if has_key_norm else "cosine"
    engine.state_clustering_centroid_rescale = (
        "coherence" if has_key_norm else "none"
    )
    engine.state_clustering_centroid_rescale_scope = "assignment"
    engine.routing_normalization = "none" if has_query_norm else "query"

    engine.leaf_layout = "expert"
    engine.leaf_block_m = 32
    engine.leaf_block_n = 16
    engine.leaf_num_warps = 2
    engine.leaf_reduce_num_warps = 1
    engine.leaf_geometry_tuning = True

    if family is ModelFamily.QWEN38:
        engine.prefill_coarse_direct_gqa = True
        engine.prefill_coarse_max_grouped_rows = 64
        engine.prefill_coarse_route_block_n = 16
        engine.prefill_coarse_route_num_warps = 8
    else:
        engine.prefill_coarse_direct_gqa = False
        engine.decode_route_group_size = 64
        engine.decode_route_segment_tiles = 1
        engine.decode_route_num_warps = 1
        engine.decode_route_reduce_num_warps = 2

    if mode.levels == 2:
        engine.virtual_page_storage = True
        engine.decode_gqa_cooperative_leaf = family is ModelFamily.QWEN38
        engine.decode_gqa_cooperative_hip = family is ModelFamily.QWEN38
        return

    engine.recursive_prefill_all_leaves = True
    engine.recursive_prefill_all_leaves_token_limit = 0
    engine.recursive_state_route_backend = (
        "resplit"
        if family is ModelFamily.QWEN38 and request_capacity >= 22_528
        else "fused"
    )
    if mode is LODMode.THREE_TIER_INT4:
        engine.leaf_quant_scale_mode = "l2"
        engine.leaf_append_quant_scale_mode = "l2"
        if family is ModelFamily.K2:
            engine.leaf_block_m = 64
            engine.leaf_num_warps = 4
            engine.decode_state_update_len = 512
            engine.decode_route_parallel_reduce = True
            engine.decode_route_parallel_reduce_block_d = 32


__all__ = ["configure_engine"]
