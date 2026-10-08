"""One production LoD policy, with geometry-only kernel specialization."""

from __future__ import annotations

import os
from typing import Any

from ._config import (
    EXACT_DECODE_LIMIT,
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

    The attention calculation is shared across both families: top-eight in
    prefill and decode, count-corrected coarse mass, and exact replacement of
    selected regions. Differences below are launch geometry and storage.
    """

    gqa = engine.config.num_attention_heads // engine.config.num_key_value_heads
    head_dim = getattr(engine, "head_dim", None)
    if head_dim is None:
        # The engine learns D from its first tensors. Model-family validation
        # already fixes the only supported shapes, so use their known widths.
        head_dim = 256 if family is ModelFamily.QWEN38 else 128
    expected = {
        ModelFamily.QWEN38: (256, 6),
        ModelFamily.K2: (128, 8),
    }.get(family)
    if expected is not None and (int(head_dim), int(gqa)) != expected:
        raise ValueError(
            f"{family.value} requires D/GQA={expected}, got {(head_dim, gqa)}"
        )
    if family in (ModelFamily.KIMI_K3, ModelFamily.GLM53_FLASH) and (int(head_dim) <= 0 or int(gqa) <= 0):
        raise ValueError("absorbed MLA requires positive D/GQA geometry")
    # NoPE MLA stores exactly the same latent as both K and V. Preserve that
    # invariant even when cached-prefill concatenations create fresh tensors.
    engine._lod_shared_latent_kv = family is ModelFamily.GLM53_FLASH
    # GLM's K256/V256 local and centroid projections are both faster than
    # their absorbed512 prefill counterparts. Persistent/decode K=V stays512.
    engine._lod_glm_project_local = family is ModelFamily.GLM53_FLASH

    engine.two_level_topk = ROUTE_COUNT
    engine.prefill_two_level_topk = ROUTE_COUNT
    # Keep oversized regions at coarse resolution. This bounds exact leaf work
    # in two-tier mode and page-summary work in three-tier mode.
    engine.max_open_centroid_leaves = 1024
    engine.separate_sink_cache = True
    engine.prefill_chunk_len = PREFILL_CHUNK_SIZE
    engine.decode_state_update_len = 256
    engine.prefill_local_len = PREFILL_LOCAL_WINDOW
    engine.prefill_state_update_len = PREFILL_CHUNK_SIZE
    engine.prefill_exact_first_chunk = True
    engine.exact_decode_limit = EXACT_DECODE_LIMIT
    engine.split_prefill_local_attention = True
    engine.prefill_local_attention_backend = "aiter"
    engine.fused_prefill_route_coarse = True
    engine.fused_prefill_stable_recompute = True
    engine.fused_prefill_external_recompute = True
    engine.prefill_hierarchical_route = True
    engine.prefill_overlap_coarse_leaf = True
    engine.fused_state_update = True
    engine.fused_state_maxsim = True
    engine.prefill_aiter_route_coarse = True

    engine.state_clustering_normalization = "none" if has_key_norm else "cosine"
    engine.state_clustering_centroid_rescale = "coherence" if has_key_norm else "none"
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
    elif family is ModelFamily.K2:
        # Wider K2 query tiles amortize long selected centroids. BF16 uses
        # 128 rows and 32 leaf columns; INT4 uses 256 rows and retains its
        # page-sized 16-column dequantization path below.
        engine.leaf_block_m = 128
        engine.leaf_block_n = 32
        engine.prefill_coarse_direct_gqa = False
        engine.prefill_coarse_max_grouped_rows = 64
        engine.prefill_coarse_route_block_n = 32
        engine.prefill_coarse_route_num_warps = 8
        engine.decode_route_group_size = 64
        engine.decode_route_segment_tiles = 1
        engine.decode_route_num_warps = 1
        engine.decode_route_reduce_num_warps = 2
    else:
        # Kimi's MLA cache has a single latent KV head.  Queries are absorbed
        # through W_UK before entering LoD, so these are ordinary MQA launch
        # settings over [latent, direct-key] vectors.  Keep the calculation
        # identical to the other families; only the tile shape differs.
        # Prefill expands only the routed leaf working set to D192/V128 before
        # expert attention.  The Kimi-specific leaf geometry is tuned below;
        # it remains much wider than the legacy raw-D576 16-row tile.
        engine.leaf_block_m = int(os.getenv("LOD_KIMI_LEAF_BLOCK_M", "32"))
        engine.leaf_block_n = 16
        engine.leaf_num_warps = int(os.getenv("LOD_KIMI_LEAF_WARPS", "2"))
        engine.prefill_coarse_direct_gqa = False
        engine.prefill_coarse_max_grouped_rows = 64
        engine.prefill_coarse_route_block_n = 16
        engine.prefill_coarse_route_num_warps = 8
        # The source-derived MLA route/coarse kernel preserves the asymmetric
        # absorbed cache: 512 latent value channels plus 64 direct-key-only
        # channels.  It follows AITER MLA's one-token/16-head work mapping.
        engine.prefill_aiter_route_coarse = True
        engine.decode_route_group_size = 64
        engine.decode_route_segment_tiles = 1
        # Full K3's 16x64, D512+64 scoring tile spills its keys and sort
        # temporaries with one wave (603 VGPR spills on gfx942). Four waves
        # preserve the same MFMA products/top-eight while keeping the tile in
        # registers. Smaller absorbed geometries retain their existing tile.
        engine.decode_route_num_warps = 4 if int(head_dim) == 576 else 1
        engine.decode_route_reduce_num_warps = 2

        # AMD's K3 v10 image carries a newer AITER host/kernel ABI than the
        # release branch's source-derived MLA prefill specialization.  Keep a
        # correctness-first fallback available while that specialization is
        # ported; decode continues to use the same Gluon LoD kernel.
        if os.getenv("LOD_KIMI_DISABLE_AITER_PREFILL", "0") == "1":
            engine.prefill_local_attention_backend = "torch"
            engine.prefill_aiter_route_coarse = False

    if mode.levels == 2:
        engine.virtual_page_storage = True
        engine.decode_gqa_cooperative_leaf = family is ModelFamily.QWEN38
        engine.decode_gqa_cooperative_hip = family is ModelFamily.QWEN38
        return

    # Keep the common first 32 pages of each centroid inline. Less common
    # pages use the already allocated bounded overflow hash, reducing the
    # persistent per-centroid directory without changing the stored leaves.
    engine.leaf_inline_pages_per_slot = 32
    engine.recursive_prefill_all_leaves = True
    engine.recursive_prefill_all_leaves_token_limit = 0
    # The fused producer is now also the fastest stable path for long Qwen
    # requests.  The old request-capacity crossover selected the legacy
    # materialized re-split route on TP1, whose launch floor and variable
    # decode latency became dominant after the fused route/coarse consumer was
    # optimized.
    engine.recursive_state_route_backend = "fused"
    if mode is LODMode.THREE_TIER_INT4:
        engine.leaf_quant_scale_mode = "l2"
        engine.leaf_append_quant_scale_mode = "l2"
        if family is ModelFamily.K2:
            engine.leaf_block_m = 256
            engine.leaf_block_n = 16
            engine.leaf_num_warps = 4
            engine.decode_route_parallel_reduce = True


__all__ = ["configure_engine"]
