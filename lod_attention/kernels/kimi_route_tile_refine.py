"""Exact top-eight refinement from eight winning 128-centroid tiles.

Each excluded tile has at least eight selected tile maxima ahead of it, so
it cannot contain a global top-eight key. This is an experimental prefill
organization, not a change to the LoD selection rule or leaf cap.
"""

from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from ._paged_common import _pack_route_score_index, _unpack_route_score_index
from .aiter_prefill_attention import _workspace_tensor
from .paged_prefill import _pack_expert_routes


@triton.jit
def _select_centroid_tiles(
    candidates, selected_tiles, Q, TILES,
    BLOCK_TILES: tl.constexpr, BLOCK_M: tl.constexpr,
    FIELDS: tl.constexpr = 16,
):
    head = tl.program_id(0)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    tile = tl.arange(0, BLOCK_TILES)
    score = tl.load(
        candidates + ((head * TILES + tile[None, :]) * FIELDS) * Q + query[:, None],
        mask=(query[:, None] < Q) & (tile[None, :] < TILES),
        other=-float("inf"),
    )
    packed = _pack_route_score_index(score, tile[None, :])
    best = tl.topk(packed, 8, dim=1)
    values, indices = _unpack_route_score_index(best)
    indices = tl.where((indices < TILES) & (values > -float("inf")), indices, -1)
    tl.store(
        selected_tiles + (head * Q + query[:, None]) * 8 + tl.arange(0, 8)[None, :],
        indices, mask=query[:, None] < Q,
    )


@triton.jit
def _pack_dense_tile_queries(
    selected, packed_rows, query_counts, Q, TILES,
    BLOCK_Q: tl.constexpr,
):
    """Reserve once per tile/query block, not once per selected query route.

    A tile occurs at most once in each query's top eight. Fixed Q-row storage
    per expert avoids cumulative-start/block-list construction. Only the
    populated prefix is consumed by the rescoring kernel.
    """
    head = tl.program_id(0)
    tile = tl.program_id(1)
    query = tl.program_id(2) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    route = tl.arange(0, 8)
    chosen = tl.load(selected + (head * Q + query[:, None]) * 8 + route[None, :],
                     mask=query[:, None] < Q, other=-1)
    rank = tl.max(tl.where(chosen == tile, route[None, :], -1), axis=1)
    valid = (query < Q) & (rank >= 0)
    local = tl.cumsum(valid.to(tl.int32)) - 1
    count = tl.sum(valid.to(tl.int32))
    expert = head * TILES + tile
    begin = tl.atomic_add(query_counts + expert, count, sem="relaxed")
    tl.store(packed_rows + expert * Q + begin + local,
             (head * Q + query) * 8 + rank, mask=valid)


@triton.jit
def _rescore_centroid_tiles(
    q, k, log_counts, packed_rows, block_experts, block_starts,
    query_counts, query_starts, active_programs, output,
    Q, TILES, STATES,
    HEADS: tl.constexpr, K_BATCH_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr, K_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr, SCALE_LOG2: tl.constexpr,
    DENSE_EXPERT_LAYOUT: tl.constexpr = False,
    FRAGMENT_CHUNKS: tl.constexpr = 0,
    PACK_Q: tl.constexpr = 256,
    BLOCK_N: tl.constexpr = 128,
):
    if DENSE_EXPERT_LAYOUT:
        expert = tl.program_id(0).to(tl.int64)
        query_block = tl.program_id(1)
        valid_program = query_block * BLOCK_M < tl.load(query_counts + expert)
    else:
        program = tl.program_id(0)
        valid_program = program < tl.load(active_programs)
    if valid_program:
        if not DENSE_EXPERT_LAYOUT:
            fragment = tl.load(block_experts + program).to(tl.int64)
            expert = fragment // FRAGMENT_CHUNKS if FRAGMENT_CHUNKS else fragment
            query_block = program - tl.load(block_starts + fragment)
        local_row = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        if FRAGMENT_CHUNKS:
            valid_query = local_row < tl.load(query_counts + fragment)
            packed_begin = fragment * PACK_Q
        elif DENSE_EXPERT_LAYOUT:
            valid_query = local_row < tl.load(query_counts + expert)
            packed_begin = expert * Q
        else:
            valid_query = local_row < tl.load(query_counts + expert)
            packed_begin = tl.load(query_starts + expert)
        route_row = tl.load(packed_rows + packed_begin + local_row, mask=valid_query, other=0).to(tl.int64)
        query_row = route_row // 8
        head_row = expert // TILES
        tile = expert % TILES
        batch = head_row // HEADS
        head = head_row % HEADS
        main_d = tl.arange(0, 128)
        tail_d = tl.arange(0, 64)
        key_index = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_key = key_index < STATES
        q_main = tl.load(q + query_row[:, None] * 192 + main_d[None, :], mask=valid_query[:, None], other=0.0)
        q_tail = tl.load(q + query_row[:, None] * 192 + 128 + tail_d[None, :], mask=valid_query[:, None], other=0.0)
        key_row = batch * K_BATCH_STRIDE + head * K_HEAD_STRIDE + key_index * K_TOKEN_STRIDE
        k_main = tl.load(k + key_row[None, :] + main_d[:, None], mask=valid_key[None, :], other=0.0)
        k_tail = tl.load(k + key_row[None, :] + 128 + tail_d[:, None], mask=valid_key[None, :], other=0.0)
        scores = tl.dot(q_main, k_main) + tl.dot(q_tail, k_tail)
        count_bias = tl.load(log_counts + batch * COUNT_BATCH_STRIDE + key_index, mask=valid_key, other=-float("inf"))
        scores = scores * SCALE_LOG2 + count_bias[None, :].to(tl.float32) * 1.4426950408889634
        scores = tl.where(valid_query[:, None] & valid_key[None, :], scores, -float("inf"))
        packed = _pack_route_score_index(scores, key_index[None, :])
        best = tl.topk(packed, 8, dim=1)
        best_scores, best_indices = _unpack_route_score_index(best)
        query_index = query_row % Q
        route = route_row % 8
        candidate_base = ((head_row * 8 + route) * 16) * Q + query_index
        rank = tl.arange(0, 8)
        tl.store(output + candidate_base[:, None] + rank[None, :] * Q,
                 best_scores, mask=valid_query[:, None])
        tl.store(output + candidate_base[:, None] + (8 + rank[None, :]) * Q,
                 best_indices.to(tl.float32), mask=valid_query[:, None])


def refine_kimi_centroid_tiles(
    candidates: torch.Tensor,
    q: torch.Tensor,
    expanded_k: torch.Tensor,
    log_counts: torch.Tensor,
    *,
    state_len: int,
    scale: float,
    buffers: dict[str, torch.Tensor] | None = None,
    block_m: int = 64,
    tile_n: int = 128,
) -> torch.Tensor:
    """Return ordinary eight-per-tile candidates for the exact global reducer."""
    block_m = int(os.environ.get("LOD_KIMI_REFINE_BLOCK_M", block_m))
    if block_m not in (16, 32, 64, 128):
        raise ValueError("Kimi refinement query tile must be 16, 32, 64 or 128")
    if q.ndim != 4 or q.size(-1) != 192 or not q.is_contiguous():
        raise ValueError("tile refinement requires contiguous BHQ192 queries")
    batch, heads, queries, _ = q.shape
    tiles = int(candidates.size(2))
    fields = int(candidates.size(3))
    if (fields not in (1, 16)
            or tuple(candidates.shape) != (batch, heads, tiles, fields, queries)):
        raise ValueError("tile-max candidates have incompatible geometry")
    if expanded_k.shape[0] != batch or expanded_k.shape[2:] != (heads, 192):
        raise ValueError("tile refinement keys must be BSH192")
    if tile_n not in (32, 64, 128):
        raise ValueError("Kimi refinement key tile must be 32, 64 or 128")
    required_tiles = triton.cdiv(state_len, tile_n)
    # CK pads coarse K to a native 128-token boundary. Smaller route groups
    # can therefore include additional wholly masked groups at the end.
    padded_tiles = triton.cdiv(state_len, 128) * (128 // tile_n)
    if not required_tiles <= tiles <= padded_tiles:
        raise ValueError("native centroid tile count is inconsistent")
    selected = _workspace_tensor(
        buffers, "tile_refine_selected", (batch, heads, queries, 8),
        dtype=torch.int64, device=q.device,
    )
    _select_centroid_tiles[(batch * heads, triton.cdiv(queries, 32))](
        candidates, selected, queries, tiles,
        BLOCK_TILES=max(8, triton.next_power_of_2(tiles)), BLOCK_M=32,
        FIELDS=fields, num_warps=4,
    )
    dense = os.environ.get("LOD_KIMI_DENSE_TILE_PACK") == "1"
    fragment_chunks = 0
    pack_q = int(os.environ.get("LOD_KIMI_TILE_PACK_QUERY_BLOCK", "256"))
    if pack_q not in (256, 512, 1024):
        raise ValueError("Kimi tile-packing query block must be 256, 512 or 1024")
    chunked = (os.environ.get("LOD_KIMI_CHUNK_TILE_PACK") == "1"
               and batch * heads * tiles * triton.cdiv(queries, pack_q) <= 32768)
    fixed_fragments = chunked and os.environ.get("LOD_KIMI_FIXED_FRAGMENT_RESCORE") == "1"
    if chunked:
        from .kimi_route_chunk_pack import pack_chunked_kimi_tile_queries

        packed, counts, block_starts, block_experts, max_blocks, fragment_chunks = (
            pack_chunked_kimi_tile_queries(selected, tiles=tiles, block_m=block_m,
                                           buffers=buffers, pack_q=pack_q,
                                           compact_work=not fixed_fragments))
        starts = block_starts
        rescore_grid = (max_blocks,)
        dense = False
    elif dense:
        packed = _workspace_tensor(
            buffers, "tile_refine_dense_rows", (batch * heads * tiles * queries,),
            dtype=torch.int32, device=q.device,
        )
        counts = _workspace_tensor(
            buffers, "tile_refine_dense_counts", (batch * heads * tiles,),
            dtype=torch.int32, device=q.device,
        )
        counts.zero_()
        _pack_dense_tile_queries[(batch * heads, tiles, triton.cdiv(queries, 256))](
            selected, packed, counts, queries, tiles, BLOCK_Q=256, num_warps=4,
        )
        # Unused arguments in the fixed expert-major dispatch.
        starts = block_experts = block_starts = counts
        rescore_grid = (batch * heads * tiles, triton.cdiv(queries, block_m))
    else:
        # Here experts are contiguous key tiles, not semantic leaf lists.
        packed, counts, starts, block_experts, block_starts, max_blocks = _pack_expert_routes(
            selected, active_slots=tiles, kv_heads=heads, kv_group_size=1,
            expert_count=batch * heads * tiles, block_m=block_m, buffers=buffers,
        )
        rescore_grid = (max_blocks,)
    output = _workspace_tensor(
        buffers, "tile_refine_candidates", (batch, heads, 8, 16, queries),
        dtype=torch.float32, device=q.device,
    )
    if required_tiles < 8:
        output.fill_(-float("inf"))
        output[:, :, :, 8:].fill_(-1)
    if fixed_fragments:
        from .kimi_route_fragment_rescore import rescore_fixed_fragments

        rescore_fixed_fragments[rescore_grid](
            q, expanded_k, log_counts, packed, counts, output,
            queries, tiles, state_len, HEADS=heads,
            K_BATCH_STRIDE=expanded_k.stride(0), K_HEAD_STRIDE=expanded_k.stride(2),
            K_TOKEN_STRIDE=expanded_k.stride(1), COUNT_BATCH_STRIDE=log_counts.stride(0),
            CHUNKS=fragment_chunks, PACK_Q=pack_q, BLOCK_M=block_m, BLOCK_N=tile_n,
            SCALE_LOG2=scale / math.log(2), num_warps=4)
        return output
    _rescore_centroid_tiles[rescore_grid](
        q, expanded_k, log_counts, packed, block_experts, block_starts[:-1],
        counts, starts[:-1], block_starts[-1:], output,
        queries, tiles, state_len,
        HEADS=heads, K_BATCH_STRIDE=expanded_k.stride(0),
        K_HEAD_STRIDE=expanded_k.stride(2), K_TOKEN_STRIDE=expanded_k.stride(1),
        COUNT_BATCH_STRIDE=log_counts.stride(0),
        BLOCK_M=block_m, SCALE_LOG2=scale / math.log(2), num_warps=4,
        DENSE_EXPERT_LAYOUT=dense,
        FRAGMENT_CHUNKS=fragment_chunks,
        PACK_Q=pack_q,
        BLOCK_N=tile_n,
    )
    return output
