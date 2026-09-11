"""State-summary routing and coarse-attention kernels for paged LoD."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._paged_common import _pack_route_score_index, _unpack_route_score_index


@triton.jit
def _materialize_state_summary_scores_gqa_kernel(
    q,
    state_k,
    counts,
    cache_indices,
    output_scores,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    SCORE_BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    BYPASS_K_L1: tl.constexpr = False,
):
    """Materialize count-corrected centroid scores shared across GQA heads."""
    batch_kv = tl.program_id(0).to(tl.int64)
    state_block = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_lane = tl.arange(0, 16)
    query_valid = query_lane < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_lane
    query_row = batch * QUERY_HEADS + query_head
    dimension = tl.arange(0, HEAD_BLOCK_DIM)
    slot = state_block * SCORE_BLOCK_N + tl.arange(0, SCORE_BLOCK_N)
    slot_valid = slot < state_len
    count = tl.load(
        counts
        + cache_batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=slot_valid,
        other=0.0,
    ).to(tl.float32)
    slot_valid &= count > 0.0
    safe_count = tl.where(slot_valid, count, 1.0)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dimension[None, :],
        mask=query_valid[:, None] & (dimension[None, :] < HEAD_DIM),
        other=0.0,
    )
    key_offsets = (
        state_k
        + cache_batch * STATE_BATCH_STRIDE
        + kv_head * STATE_HEAD_STRIDE
        + slot[:, None] * STATE_TOKEN_STRIDE
        + dimension[None, :]
    )
    if BYPASS_K_L1:
        key_sums = tl.load(
            key_offsets,
            mask=slot_valid[:, None] & (dimension[None, :] < HEAD_DIM),
            other=0.0,
            cache_modifier=".cg",
        )
    else:
        key_sums = tl.load(
            key_offsets,
            mask=slot_valid[:, None] & (dimension[None, :] < HEAD_DIM),
            other=0.0,
        )
    dots = tl.dot(queries, tl.trans(key_sums), out_dtype=tl.float32)
    scores = SCALE * dots / safe_count[None, :] + tl.log(safe_count)[None, :]
    scores = tl.where(
        query_valid[:, None] & slot_valid[None, :],
        scores,
        -float("inf"),
    )
    tl.store(
        output_scores + query_row[:, None] * STATE_CAPACITY + slot[None, :],
        scores,
        mask=query_valid[:, None] & (slot[None, :] < state_len),
    )


@triton.jit
def _materialized_state_tile_top8_lse_kernel(
    scores,
    counts,
    cache_indices,
    candidate_scores,
    candidate_indices,
    partial_lse,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    MAX_TILES: tl.constexpr,
    TOPK_BLOCK_N: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    CANDIDATES_PER_TILE: tl.constexpr = 8,
):
    """Select tile route candidates and reuse their score load for tile LSE."""
    batch_kv = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_lane = tl.arange(0, 16)
    query_valid = query_lane < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_lane
    query_row = batch * QUERY_HEADS + query_head
    slot = tile * TOPK_BLOCK_N + tl.arange(0, TOPK_BLOCK_N)
    table_valid = slot < state_len
    count = tl.load(
        counts
        + cache_batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=table_valid,
        other=0.0,
    ).to(tl.float32)
    route_valid = table_valid & (slot >= PROTECTED_LEN) & (count > 0.0)
    if MAX_LEAF_TOKENS:
        route_valid &= count < MAX_LEAF_TOKENS
    values = tl.load(
        scores + query_row[:, None] * STATE_CAPACITY + slot[None, :],
        mask=query_valid[:, None] & table_valid[None, :],
        other=-float("inf"),
    ).to(tl.float32)

    packed = _pack_route_score_index(
        tl.where(route_valid[None, :], values, -float("inf")),
        slot[None, :],
    )
    best = tl.topk(packed, CANDIDATES_PER_TILE, dim=1)
    best_scores, best_indices = _unpack_route_score_index(best)
    rank = tl.arange(0, CANDIDATES_PER_TILE)
    candidate_base = (query_row * MAX_TILES + tile) * CANDIDATES_PER_TILE
    tl.store(
        candidate_scores + candidate_base[:, None] + rank[None, :],
        best_scores,
        mask=query_valid[:, None],
    )
    tl.store(
        candidate_indices + candidate_base[:, None] + rank[None, :],
        best_indices,
        mask=query_valid[:, None],
    )

    lse_active = table_valid[None, :] & (values > -float("inf"))
    lse_has_mass = tl.sum(lse_active.to(tl.int32), axis=1) > 0
    lse_maximum = tl.where(
        lse_has_mass,
        tl.max(tl.where(lse_active, values, -float("inf")), axis=1),
        0.0,
    )
    lse_denominator = tl.sum(
        tl.where(lse_active, tl.exp(values - lse_maximum[:, None]), 0.0),
        axis=1,
    )
    tile_lse = tl.where(
        lse_has_mass,
        lse_maximum + tl.log(lse_denominator),
        -float("inf"),
    )
    tl.store(
        partial_lse + query_row * MAX_TILES + tile,
        tile_lse,
        mask=query_valid,
    )


@triton.jit
def _reduce_materialized_state_top8_lse_kernel(
    candidate_scores,
    candidate_indices,
    partial_lse,
    top_scores,
    top_indices,
    output_lse,
    active_tiles,
    MAX_TILES: tl.constexpr,
    CANDIDATE_TILE: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    CANDIDATES_PER_TILE: tl.constexpr = 8,
    OPEN_COUNT: tl.constexpr = 8,
    ROUTE_COUNT: tl.constexpr = 8,
):
    """Reduce route candidates and tile LSE values in one row program."""
    row = tl.program_id(0).to(tl.int64)
    candidate_offset = tl.arange(0, CANDIDATE_TILE)
    best_packed = tl.full((OPEN_COUNT,), -9223372036854775807, tl.int64)
    for candidate_begin in tl.range(
        0, active_tiles * CANDIDATES_PER_TILE, CANDIDATE_TILE, num_stages=1
    ):
        candidate = candidate_begin + candidate_offset
        valid = candidate < active_tiles * CANDIDATES_PER_TILE
        values = tl.load(
            candidate_scores + row * MAX_TILES * CANDIDATES_PER_TILE + candidate,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        indices = tl.load(
            candidate_indices + row * MAX_TILES * CANDIDATES_PER_TILE + candidate,
            mask=valid,
            other=0,
        )
        packed = _pack_route_score_index(values, indices)
        block_best = tl.topk(packed, OPEN_COUNT, dim=0)
        best_packed = tl.topk(tl.interleave(best_packed, block_best), OPEN_COUNT, dim=0)
    best_scores, best_indices = _unpack_route_score_index(best_packed)
    rank = tl.arange(0, OPEN_COUNT)
    tl.store(top_scores + row * ROUTE_COUNT + rank, best_scores)
    tl.store(top_indices + row * ROUTE_COUNT + rank, best_indices)
    if OPEN_COUNT < ROUTE_COUNT:
        tail = OPEN_COUNT + tl.arange(0, ROUTE_COUNT - OPEN_COUNT)
        tl.store(top_scores + row * ROUTE_COUNT + tail, -float("inf"))
        tl.store(top_indices + row * ROUTE_COUNT + tail, -1)

    tile = tl.arange(0, TILE_BLOCK)
    tile_valid = tile < active_tiles
    lse_values = tl.load(
        partial_lse + row * MAX_TILES + tile,
        mask=tile_valid,
        other=-float("inf"),
    ).to(tl.float32)
    lse_active = tile_valid & (lse_values > -float("inf"))
    lse_maximum = tl.max(tl.where(lse_active, lse_values, -float("inf")), axis=0)
    lse_denominator = tl.sum(
        tl.where(lse_active, tl.exp(lse_values - lse_maximum), 0.0), axis=0
    )
    tl.store(output_lse + row, lse_maximum + tl.log(lse_denominator))


@triton.jit
def _materialized_state_normalized_pv_split_kernel(
    scores,
    coarse_lse,
    counts,
    cache_indices,
    state_v,
    partial_out,
    STATE_V_BATCH_STRIDE,
    STATE_V_HEAD_STRIDE,
    STATE_V_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PV_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fuse count-corrected normalization into a split MFMA PV tile."""
    batch_kv = tl.program_id(0).to(tl.int64)
    dimension_block = tl.program_id(1).to(tl.int64)
    split = tl.program_id(2).to(tl.int64)
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_lane = tl.arange(0, 16)
    query_valid = query_lane < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_lane
    query_row = batch * QUERY_HEADS + query_head
    dimension = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((16, BLOCK_D), tl.float32)
    token_offset = tl.arange(0, BLOCK_N)
    lse = tl.load(
        coarse_lse + query_row,
        mask=query_valid,
        other=0.0,
    ).to(tl.float32)
    for state_begin in tl.range(
        split * BLOCK_N, state_len, PV_SPLITS * BLOCK_N, num_stages=1
    ):
        slot = state_begin + token_offset
        valid_slot = slot < state_len
        count = tl.load(
            counts
            + cache_batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        active = valid_slot & (count > 0.0)
        score = tl.load(
            scores + query_row[:, None] * STATE_CAPACITY + slot[None, :],
            mask=query_valid[:, None] & active[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        probability = tl.where(
            query_valid[:, None] & active[None, :],
            tl.exp(score - lse[:, None]) / count[None, :],
            0.0,
        ).to(tl.float16)
        value_sums = tl.load(
            state_v
            + cache_batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot[:, None] * STATE_V_TOKEN_STRIDE
            + dimension[None, :],
            mask=valid_slot[:, None] & (dimension[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float16)
        accumulator += tl.dot(
            probability,
            value_sums,
            out_dtype=tl.float32,
        )
    tl.store(
        partial_out
        + (query_row[:, None] * PV_SPLITS + split) * HEAD_DIM
        + dimension[None, :],
        accumulator,
        mask=query_valid[:, None] & (dimension[None, :] < HEAD_DIM),
    )


@triton.jit
def _reduce_materialized_state_pv_kernel(
    partial_out,
    coarse_out,
    QUERY_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Sum separately executed PV state-axis pieces."""
    row = tl.program_id(0).to(tl.int64)
    dimension = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    split = tl.arange(0, PV_SPLITS)
    values = tl.load(
        partial_out
        + (row * PV_SPLITS + split[:, None]) * HEAD_DIM
        + dimension[None, :],
        mask=dimension[None, :] < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        coarse_out + row * HEAD_DIM + dimension,
        tl.sum(values, axis=0),
        mask=dimension < HEAD_DIM,
    )


def materialized_state_route_gqa(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    cache_indices: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    open_count: int = 4,
    protected_len: int = 0,
    max_leaf_tokens: int | None = None,
    waves_per_eu: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the deliberately re-split centroid route and coarse pipeline."""
    batch, query_heads, query_len, head_dim = q.shape
    if query_len != 1:
        raise ValueError("materialized state routing requires decode queries")
    cache_batch, kv_heads, state_capacity, summary_dim = state_k.shape
    if summary_dim != head_dim or tuple(state_v.shape) != tuple(state_k.shape):
        raise ValueError("materialized state routing requires equal K/V geometry")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("materialized state routing has inconsistent GQA geometry")
    if kv_group_size < 1 or kv_group_size > 16:
        raise ValueError("materialized state routing supports GQA groups up to 16")
    if tuple(counts.shape[:3]) != (cache_batch, kv_heads, state_capacity):
        raise ValueError("materialized state routing counts do not match state")
    if not 0 < state_len <= state_capacity:
        raise ValueError("materialized state routing has an invalid active length")
    if tuple(cache_indices.shape) != (batch,):
        raise ValueError("materialized state routing cache indices are invalid")
    scores = buffers.get("route_state_scores")
    expected_table = (batch, query_heads, 1, state_capacity)
    if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected_table:
        raise ValueError(
            "materialized state score buffer is missing or has wrong shape"
        )
    if scores.dtype != torch.float32:
        raise TypeError("materialized state scores must use FP32")
    candidate_scores = buffers["route_candidate_scores"]
    candidate_indices = buffers["route_candidate_indices"]
    partial_lse = buffers["route_group_lse"]
    top_slots = buffers["route_top_slots"]
    top_scores = buffers["route_top_scores"]
    coarse_out = buffers["coarse_out"]
    coarse_lse = buffers["coarse_lse"]
    route_count = int(top_slots.size(-1))
    if route_count < 4:
        raise ValueError("materialized state route buffer is too small")
    if open_count not in (4, 8) or open_count > route_count:
        raise ValueError("materialized state routing supports four or eight routes")
    max_tiles = int(candidate_scores.size(2))
    active_topk_tiles = triton.cdiv(state_len, 128)
    active_lse_tiles = triton.cdiv(state_len, 128)
    if max(active_topk_tiles, active_lse_tiles) > max_tiles:
        raise ValueError("materialized state routing candidate buffers are too small")
    if active_topk_tiles != active_lse_tiles:
        raise ValueError("fused top-k/LSE tiles require a shared tile geometry")
    rows = batch * query_heads
    score_block_n = 64 if head_dim == 128 else 32 if head_dim == 256 else 16

    def begin() -> torch.cuda.Event | None:
        if timing_events is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end(name: str, start: torch.cuda.Event | None) -> None:
        if start is None or timing_events is None:
            return
        finish = torch.cuda.Event(enable_timing=True)
        finish.record()
        timing_events.setdefault(name, []).append((start, finish))
        timing_events.setdefault(f"{name}_b{batch}", []).append((start, finish))

    start = begin()
    _materialize_state_summary_scores_gqa_kernel[
        batch * kv_heads, triton.cdiv(state_len, score_block_n)
    ](
        q.contiguous(),
        state_k,
        counts,
        cache_indices.contiguous(),
        scores,
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        STATE_CAPACITY=state_capacity,
        HEAD_DIM=head_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        SCORE_BLOCK_N=score_block_n,
        SCALE=float(scale),
        BYPASS_K_L1=True,
        num_warps=2,
        num_stages=3,
        waves_per_eu=2,
    )
    end("route_state_scores", start)
    start = begin()
    _materialized_state_tile_top8_lse_kernel[batch * kv_heads, active_topk_tiles](
        scores,
        counts,
        cache_indices.contiguous(),
        candidate_scores,
        candidate_indices,
        partial_lse,
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        STATE_CAPACITY=state_capacity,
        MAX_TILES=max_tiles,
        TOPK_BLOCK_N=128,
        CANDIDATES_PER_TILE=4,
        PROTECTED_LEN=protected_len,
        MAX_LEAF_TOKENS=max_leaf_tokens or 0,
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    end("route_state_topk_lse_tile", start)
    start = begin()
    _reduce_materialized_state_top8_lse_kernel[rows,](
        candidate_scores,
        candidate_indices,
        partial_lse,
        top_scores,
        top_slots,
        coarse_lse,
        active_topk_tiles,
        MAX_TILES=max_tiles,
        CANDIDATE_TILE=triton.next_power_of_2(active_topk_tiles * 4),
        TILE_BLOCK=triton.next_power_of_2(active_lse_tiles),
        CANDIDATES_PER_TILE=4,
        OPEN_COUNT=open_count,
        ROUTE_COUNT=route_count,
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    end("route_state_topk_lse_reduce", start)
    pv_splits = min(8, int(buffers["partial_out"].size(2)))
    start = begin()
    _materialized_state_normalized_pv_split_kernel[
        batch * kv_heads, triton.cdiv(head_dim, 128), pv_splits
    ](
        scores,
        coarse_lse,
        counts,
        cache_indices.contiguous(),
        state_v,
        buffers["partial_out"],
        state_v.stride(0),
        state_v.stride(1),
        state_v.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        STATE_CAPACITY=state_capacity,
        HEAD_DIM=head_dim,
        PV_SPLITS=pv_splits,
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    end("route_state_normalized_pv_split", start)
    start = begin()
    _reduce_materialized_state_pv_kernel[rows, triton.cdiv(head_dim, 128)](
        buffers["partial_out"],
        coarse_out,
        QUERY_HEADS=query_heads,
        HEAD_DIM=head_dim,
        PV_SPLITS=pv_splits,
        BLOCK_D=128,
        num_warps=1,
        waves_per_eu=waves_per_eu,
    )
    end("route_state_pv_reduce", start)
    return (top_slots, top_scores, coarse_out, coarse_lse)


@triton.jit
def _decode_route_coarse_gqa_groups_kernel(
    q,
    state_k,
    state_v,
    counts,
    cache_indices,
    candidate_scores,
    candidate_indices,
    group_out,
    group_lse,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    STATE_V_BATCH_STRIDE,
    STATE_V_HEAD_STRIDE,
    STATE_V_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    state_len,
    state_lens,
    union_counts,
    union_token_counts,
    sequence_epochs,
    log_count_bias,
    local_lens,
    local_k,
    local_v,
    new_k,
    new_v,
    LOG_COUNT_BIAS_BATCH_STRIDE,
    LOG_COUNT_BIAS_HEAD_STRIDE,
    LOG_COUNT_BIAS_TOKEN_STRIDE,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    GROUP_N: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    USE_DOT: tl.constexpr,
    KEYS_ARE_MEANS: tl.constexpr = False,
    SCORE_ONLY: tl.constexpr = False,
    USE_STATE_LENS: tl.constexpr = False,
    CANDIDATES_PER_GROUP: tl.constexpr = 8,
    FLOAT_TOP4: tl.constexpr = False,
    FUSE_UNION_INIT: tl.constexpr = False,
    UNION_SEQUENCE_CAPACITY: tl.constexpr = 1,
    PACKED_CANDIDATES: tl.constexpr = False,
    STORE_ALL_SCORES: tl.constexpr = False,
    USE_LOG_COUNT_BIAS: tl.constexpr = False,
    FUSE_LOCAL: tl.constexpr = False,
    LOCAL_CAPACITY: tl.constexpr = 1,
    LOCAL_LIMIT: tl.constexpr = 0,
    INCLUDE_NEW: tl.constexpr = False,
):
    """Share each state K/V tile across all query heads in one GQA group."""
    batch_kv = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = batch_kv - batch * KV_HEADS
    if FUSE_UNION_INIT and group == 0:
        union_row_valid = batch_kv < UNION_SEQUENCE_CAPACITY
        safe_union_row = tl.where(union_row_valid, batch_kv, 0)
        tl.store(union_counts + safe_union_row, 0, mask=union_row_valid)
        tl.store(
            union_token_counts + safe_union_row,
            0,
            mask=union_row_valid,
        )
        epoch = tl.load(
            sequence_epochs + safe_union_row,
            mask=union_row_valid,
            other=0,
        ).to(tl.int32)
        tl.store(
            sequence_epochs + safe_union_row,
            epoch + 1,
            mask=union_row_valid,
        )
    if USE_STATE_LENS:
        active_state_len = tl.minimum(
            tl.load(state_lens + cache_batch).to(tl.int32), state_len
        )
    else:
        active_state_len = state_len
    if FUSE_LOCAL:
        active_local_len = tl.minimum(
            tl.load(local_lens + cache_batch).to(tl.int32),
            LOCAL_LIMIT,
        )
        active_local_extent = active_local_len + 1 if INCLUDE_NEW else active_local_len
    else:
        active_local_len = 0
        active_local_extent = 0
    # Pad the four GQA rows to a native MFMA M tile. Short-M value matmuls use
    # a different reduction order on MI325X and perturb decode enough to hurt
    # exact string continuation.
    q_offset = tl.arange(0, 16)
    query_valid = q_offset < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + q_offset
    query_row = batch * QUERY_HEADS + query_head
    slot = group * GROUP_N + tl.arange(0, GROUP_N)
    valid = slot < active_state_len
    dim = tl.arange(0, HEAD_DIM)
    if group * GROUP_N >= active_state_len and group * GROUP_N >= active_local_extent:
        rank = tl.arange(0, CANDIDATES_PER_GROUP)
        candidate_base = (query_row * MAX_GROUPS + group) * CANDIDATES_PER_GROUP
        if PACKED_CANDIDATES:
            tl.store(
                candidate_indices + candidate_base[:, None] + rank[None, :],
                -9187343239835811841,
                mask=query_valid[:, None],
            )
        else:
            tl.store(
                candidate_scores + candidate_base[:, None] + rank[None, :],
                -float("inf"),
                mask=query_valid[:, None],
            )
            tl.store(
                candidate_indices + candidate_base[:, None] + rank[None, :],
                -1,
                mask=query_valid[:, None],
            )
        if not SCORE_ONLY:
            group_row = query_row * MAX_GROUPS + group
            tl.store(
                group_out + group_row[:, None] * HEAD_DIM + dim[None, :],
                0.0,
                mask=query_valid[:, None],
            )
            tl.store(group_lse + group_row, -float("inf"), mask=query_valid)
        return
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    count = tl.load(
        counts
        + cache_batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=valid,
        other=1.0,
    ).to(tl.float32)
    valid &= count > 0.0
    count = tl.where(valid, count, 1.0)
    keys = tl.load(
        state_k
        + cache_batch * STATE_BATCH_STRIDE
        + kv_head * STATE_HEAD_STRIDE
        + slot[:, None] * STATE_TOKEN_STRIDE
        + dim[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    if KEYS_ARE_MEANS:
        mean_keys = keys
    else:
        mean_keys = (keys.to(tl.float32) / count[:, None]).to(keys.dtype)
    scores = tl.dot(queries, tl.trans(mean_keys), out_dtype=tl.float32)
    scores *= SCALE
    if STORE_ALL_SCORES:
        # ``group_out`` is otherwise unused by score-only routing and has at
        # least MAX_GROUPS * HEAD_DIM entries per query row. Preserve the
        # already-computed QK term so the attention consumer need not reload
        # every centroid key and repeat the same matrix multiply.
        tl.store(
            group_out + query_row[:, None] * MAX_GROUPS * HEAD_DIM + slot[None, :],
            scores,
            mask=query_valid[:, None] & valid[None, :],
        )
    if USE_LOG_COUNT_BIAS:
        count_bias = tl.load(
            log_count_bias
            + cache_batch * LOG_COUNT_BIAS_BATCH_STRIDE
            + kv_head * LOG_COUNT_BIAS_HEAD_STRIDE
            + slot * LOG_COUNT_BIAS_TOKEN_STRIDE,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
    else:
        count_bias = tl.log(count)
    scores += count_bias[None, :]
    scores = tl.where(query_valid[:, None] & valid[None, :], scores, -float("inf"))
    route_scores = tl.where(slot[None, :] >= PROTECTED_LEN, scores, -float("inf"))
    if MAX_LEAF_TOKENS:
        route_scores = tl.where(
            count[None, :] < MAX_LEAF_TOKENS,
            route_scores,
            -float("inf"),
        )

    candidate_base = (query_row * MAX_GROUPS + group) * CANDIDATES_PER_GROUP
    if FLOAT_TOP4:
        # Four FP32 max/argmax reductions avoid the substantially heavier
        # int64 packed sort while retaining the same exact tie rule (the
        # lowest slot wins equal scores).
        remaining_scores = route_scores
        position = tl.arange(0, GROUP_N)
        for candidate_rank in tl.static_range(0, 4):
            block_score = tl.max(remaining_scores, axis=1)
            block_position = tl.argmax(remaining_scores, axis=1, tie_break_left=True)
            block_index = tl.sum(
                tl.where(
                    position[None, :] == block_position[:, None],
                    slot[None, :],
                    0,
                ),
                axis=1,
            )
            tl.store(
                candidate_scores + candidate_base + candidate_rank,
                block_score,
                mask=query_valid,
            )
            tl.store(
                candidate_indices + candidate_base + candidate_rank,
                block_index,
                mask=query_valid,
            )
            remaining_scores = tl.where(
                position[None, :] == block_position[:, None],
                -float("inf"),
                remaining_scores,
            )
    else:
        packed = _pack_route_score_index(route_scores, slot[None, :])
        block_top = tl.topk(packed, CANDIDATES_PER_GROUP, dim=1)
        rank = tl.arange(0, CANDIDATES_PER_GROUP)
        if PACKED_CANDIDATES:
            tl.store(
                candidate_indices + candidate_base[:, None] + rank[None, :],
                block_top,
                mask=query_valid[:, None],
            )
        else:
            block_scores, block_indices = _unpack_route_score_index(block_top)
            tl.store(
                candidate_scores + candidate_base[:, None] + rank[None, :],
                block_scores,
                mask=query_valid[:, None],
            )
            tl.store(
                candidate_indices + candidate_base[:, None] + rank[None, :],
                block_indices,
                mask=query_valid[:, None],
            )

    if not SCORE_ONLY:
        values = tl.load(
            state_v
            + cache_batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot[:, None] * STATE_V_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        mean_values = (values.to(tl.float32) / count[:, None]).to(values.dtype)
        maximum = tl.max(scores, axis=1)
        weights = tl.exp(scores - maximum[:, None])
        weights = tl.where(query_valid[:, None] & valid[None, :], weights, 0.0)
        denominator = tl.sum(weights, axis=1)
        weighted_values = tl.dot(
            weights.to(mean_values.dtype), mean_values, out_dtype=tl.float32
        )
        if FUSE_LOCAL:
            # Most state groups have no local tokens. Keep the local MFMA and
            # cache traffic behind a program-uniform branch instead of paying
            # for a fully masked dot product in every centroid group.
            if group * GROUP_N < active_local_extent:
                local_token = group * GROUP_N + tl.arange(0, GROUP_N)
                cached_local = local_token < active_local_len
                current_local = INCLUDE_NEW & (local_token == active_local_len)
                valid_local = cached_local | current_local
                local_storage = (
                    cache_batch * KV_HEADS + kv_head
                ) * LOCAL_CAPACITY + local_token
                local_keys = tl.load(
                    local_k + local_storage[:, None] * HEAD_DIM + dim[None, :],
                    mask=cached_local[:, None],
                    other=0.0,
                )
                local_values = tl.load(
                    local_v + local_storage[:, None] * HEAD_DIM + dim[None, :],
                    mask=cached_local[:, None],
                    other=0.0,
                )
                if INCLUDE_NEW:
                    current_key = tl.load(
                        new_k + (batch * KV_HEADS + kv_head) * HEAD_DIM + dim
                    )
                    current_value = tl.load(
                        new_v + (batch * KV_HEADS + kv_head) * HEAD_DIM + dim
                    )
                    local_keys = tl.where(
                        current_local[:, None], current_key[None, :], local_keys
                    )
                    local_values = tl.where(
                        current_local[:, None], current_value[None, :], local_values
                    )
                    tl.store(
                        local_k + local_storage[:, None] * HEAD_DIM + dim[None, :],
                        current_key[None, :],
                        mask=current_local[:, None],
                    )
                    tl.store(
                        local_v + local_storage[:, None] * HEAD_DIM + dim[None, :],
                        current_value[None, :],
                        mask=current_local[:, None],
                    )
                local_scores = (
                    tl.dot(
                        queries,
                        tl.trans(local_keys),
                        out_dtype=tl.float32,
                    )
                    * SCALE
                )
                local_scores = tl.where(
                    query_valid[:, None] & valid_local[None, :],
                    local_scores,
                    -float("inf"),
                )
                local_has_mass = tl.sum(valid_local.to(tl.int32), axis=0) > 0
                state_has_mass = denominator > 0.0
                local_maximum = tl.max(local_scores, axis=1)
                combined_maximum = tl.where(
                    state_has_mass,
                    tl.where(
                        local_has_mass,
                        tl.maximum(maximum, local_maximum),
                        maximum,
                    ),
                    local_maximum,
                )
                state_correction = tl.where(
                    state_has_mass,
                    tl.exp(maximum - combined_maximum),
                    0.0,
                )
                local_weights = tl.where(
                    query_valid[:, None] & valid_local[None, :],
                    tl.exp(local_scores - combined_maximum[:, None]),
                    0.0,
                )
                denominator = denominator * state_correction + tl.sum(
                    local_weights, axis=1
                )
                weighted_values = weighted_values * state_correction[:, None] + tl.dot(
                    local_weights.to(local_values.dtype),
                    local_values,
                    out_dtype=tl.float32,
                )
                maximum = tl.where(
                    state_has_mass | local_has_mass,
                    combined_maximum,
                    maximum,
                )
        group_row = query_row * MAX_GROUPS + group
        tl.store(
            group_out + group_row[:, None] * HEAD_DIM + dim[None, :],
            tl.where(
                denominator[:, None] > 0.0,
                weighted_values / denominator[:, None],
                0.0,
            ),
            mask=query_valid[:, None],
        )
        tl.store(
            group_lse + group_row,
            tl.where(
                denominator > 0.0,
                maximum + tl.log(denominator),
                -float("inf"),
            ),
            mask=query_valid,
        )


@triton.jit
def _decode_route_coarse_gqa_mtp2_groups_kernel(
    q,
    state_k,
    state_v,
    counts,
    cache_indices,
    candidate_scores,
    candidate_indices,
    group_out,
    group_lse,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    STATE_V_BATCH_STRIDE,
    STATE_V_HEAD_STRIDE,
    STATE_V_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    state_len,
    state_lens,
    execution_marker,
    local_execution_marker,
    local_lens,
    local_k,
    local_v,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE,
    LOCAL_V_BATCH_STRIDE,
    LOCAL_V_HEAD_STRIDE,
    LOCAL_V_TOKEN_STRIDE,
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    REQUEST_ROWS: tl.constexpr,
    SPECULATIVE_STEPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    GROUP_N: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    USE_DOT: tl.constexpr,
    FUSE_LOCAL: tl.constexpr,
    USE_STATE_LENS: tl.constexpr = False,
    SCORE_ONLY: tl.constexpr = False,
    CANDIDATES_PER_GROUP: tl.constexpr = 8,
):
    """Route each adjacent proposal pair in one native M=16 state/local tile.

    The flattened speculative batch is step-major.  Treating every two steps
    as an independent pair preserves a distinct current query and route for
    every position while sharing each centroid/local K/V load across the two
    positions.  ``SPECULATIVE_STEPS == 2`` is the original MTP path; larger
    even proposals simply contribute more independent pair groups.
    """
    pair_request_kv = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    pair_request = pair_request_kv // KV_HEADS
    kv_head = pair_request_kv - pair_request * KV_HEADS
    pair_group = pair_request // REQUEST_ROWS
    request = pair_request - pair_group * REQUEST_ROWS
    base_step = pair_group * 2
    base_logical_batch = base_step * REQUEST_ROWS + request
    cache_batch = tl.load(cache_indices + base_logical_batch).to(tl.int64)
    tl.store(
        execution_marker,
        1,
        mask=(pair_request_kv == 0) & (group == 0),
    )
    if FUSE_LOCAL:
        tl.store(
            local_execution_marker,
            1,
            mask=(pair_request_kv == 0) & (group == 0),
        )

    if USE_STATE_LENS:
        active_state_len = tl.minimum(
            tl.load(state_lens + cache_batch).to(tl.int32), state_len
        )
    else:
        active_state_len = state_len
    if FUSE_LOCAL:
        base_local_len = tl.load(local_lens + base_logical_batch).to(tl.int32)
        active_local_extent = tl.minimum(
            base_local_len + 2, local_len + SPECULATIVE_STEPS
        )
    else:
        base_local_len = 0
        active_local_extent = 0

    lane = tl.arange(0, 16)
    query_valid = lane < 2 * KV_GROUP_SIZE
    step = lane // KV_GROUP_SIZE
    query_in_group = lane - step * KV_GROUP_SIZE
    logical_batch = (base_step + step) * REQUEST_ROWS + request
    query_head = kv_head * KV_GROUP_SIZE + query_in_group
    query_row = logical_batch * QUERY_HEADS + query_head
    slot = group * GROUP_N + tl.arange(0, GROUP_N)
    valid = slot < active_state_len
    dim = tl.arange(0, HEAD_DIM)
    if (
        group * GROUP_N >= active_state_len
        and group * GROUP_N >= active_local_extent
    ):
        rank = tl.arange(0, CANDIDATES_PER_GROUP)
        candidate_base = (query_row * MAX_GROUPS + group) * CANDIDATES_PER_GROUP
        tl.store(
            candidate_scores + candidate_base[:, None] + rank[None, :],
            -float("inf"),
            mask=query_valid[:, None],
        )
        tl.store(
            candidate_indices + candidate_base[:, None] + rank[None, :],
            -1,
            mask=query_valid[:, None],
        )
        if not SCORE_ONLY:
            group_row = query_row * MAX_GROUPS + group
            tl.store(
                group_out + group_row[:, None] * HEAD_DIM + dim[None, :],
                0.0,
                mask=query_valid[:, None],
            )
            tl.store(group_lse + group_row, -float("inf"), mask=query_valid)
        return
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    count = tl.load(
        counts
        + cache_batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=valid,
        other=1.0,
    ).to(tl.float32)
    valid &= count > 0.0
    count = tl.where(valid, count, 1.0)
    keys = tl.load(
        state_k
        + cache_batch * STATE_BATCH_STRIDE
        + kv_head * STATE_HEAD_STRIDE
        + slot[:, None] * STATE_TOKEN_STRIDE
        + dim[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    mean_keys = (keys.to(tl.float32) / count[:, None]).to(keys.dtype)
    if USE_DOT:
        scores = tl.dot(queries, tl.trans(mean_keys), out_dtype=tl.float32)
    else:
        scores = tl.sum(
            queries[:, None, :].to(tl.float32) * mean_keys[None, :, :].to(tl.float32),
            axis=2,
        )
    scores = scores * SCALE + tl.log(count)[None, :]
    scores = tl.where(query_valid[:, None] & valid[None, :], scores, -float("inf"))
    route_scores = tl.where(slot[None, :] >= PROTECTED_LEN, scores, -float("inf"))
    if MAX_LEAF_TOKENS:
        route_scores = tl.where(
            count[None, :] < MAX_LEAF_TOKENS,
            route_scores,
            -float("inf"),
        )

    candidate_base = (query_row * MAX_GROUPS + group) * CANDIDATES_PER_GROUP
    packed = _pack_route_score_index(route_scores, slot[None, :])
    block_top = tl.topk(packed, CANDIDATES_PER_GROUP, dim=1)
    block_scores, block_indices = _unpack_route_score_index(block_top)
    rank = tl.arange(0, CANDIDATES_PER_GROUP)
    tl.store(
        candidate_scores + candidate_base[:, None] + rank[None, :],
        block_scores,
        mask=query_valid[:, None],
    )
    tl.store(
        candidate_indices + candidate_base[:, None] + rank[None, :],
        block_indices,
        mask=query_valid[:, None],
    )

    if not SCORE_ONLY:
        values = tl.load(
            state_v
            + cache_batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot[:, None] * STATE_V_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        mean_values = (values.to(tl.float32) / count[:, None]).to(values.dtype)
        maximum = tl.max(scores, axis=1)
        weights = tl.exp(scores - maximum[:, None])
        weights = tl.where(query_valid[:, None] & valid[None, :], weights, 0.0)
        denominator = tl.sum(weights, axis=1)
        weighted_values = tl.dot(
            weights.to(mean_values.dtype), mean_values, out_dtype=tl.float32
        )
        if FUSE_LOCAL:
            # Proposal K/V has already been staged into the physical request
            # row.  Spread the bounded suffix over the first route groups,
            # so the existing M=16 programs share every local load across
            # both proposal positions without adding another launch/barrier.
            # All proposal tokens may temporarily extend past the ordinary
            # local limit before the host advances/catches up the cache.
            if group * GROUP_N < base_local_len + 2:
                local_token = group * GROUP_N + tl.arange(0, GROUP_N)
                shared_local_valid = (local_token < base_local_len + 2) & (
                    local_token < local_len + SPECULATIVE_STEPS
                )
                causal_local_valid = local_token[None, :] < (
                    base_local_len + step[:, None] + 1
                )
                local_valid = (
                    query_valid[:, None]
                    & shared_local_valid[None, :]
                    & causal_local_valid
                )
                local_keys = tl.load(
                    local_k
                    + cache_batch * LOCAL_K_BATCH_STRIDE
                    + kv_head * LOCAL_K_HEAD_STRIDE
                    + local_token[:, None] * LOCAL_K_TOKEN_STRIDE
                    + dim[None, :],
                    mask=shared_local_valid[:, None],
                    other=0.0,
                ).to(queries.dtype)
                local_values = tl.load(
                    local_v
                    + cache_batch * LOCAL_V_BATCH_STRIDE
                    + kv_head * LOCAL_V_HEAD_STRIDE
                    + local_token[:, None] * LOCAL_V_TOKEN_STRIDE
                    + dim[None, :],
                    mask=shared_local_valid[:, None],
                    other=0.0,
                )
                local_scores = (
                    tl.dot(
                        queries,
                        tl.trans(local_keys),
                        out_dtype=tl.float32,
                    )
                    * SCALE
                )
                local_scores = tl.where(local_valid, local_scores, -float("inf"))
                local_has_mass = tl.sum(local_valid.to(tl.int32), axis=1) > 0
                local_maximum = tl.max(local_scores, axis=1)
                combined_maximum = tl.where(
                    local_has_mass,
                    tl.maximum(maximum, local_maximum),
                    maximum,
                )
                state_correction = tl.exp(maximum - combined_maximum)
                local_weights = tl.where(
                    local_valid,
                    tl.exp(local_scores - combined_maximum[:, None]),
                    0.0,
                )
                denominator = denominator * state_correction + tl.sum(
                    local_weights, axis=1
                )
                weighted_values = weighted_values * state_correction[:, None] + tl.dot(
                    local_weights.to(local_values.dtype),
                    local_values,
                    out_dtype=tl.float32,
                )
                maximum = combined_maximum
        group_row = query_row * MAX_GROUPS + group
        tl.store(
            group_out + group_row[:, None] * HEAD_DIM + dim[None, :],
            tl.where(
                denominator[:, None] > 0.0,
                weighted_values / denominator[:, None],
                0.0,
            ),
            mask=query_valid[:, None],
        )
        tl.store(
            group_lse + group_row,
            tl.where(
                denominator > 0.0,
                maximum + tl.log(denominator),
                -float("inf"),
            ),
            mask=query_valid,
        )


@triton.jit
def _decode_route_coarse_gqa_groups_fixed_prepare_kernel(
    q,
    state_k,
    state_v,
    counts,
    cache_indices,
    candidate_scores,
    candidate_indices,
    group_out,
    group_lse,
    local_lens,
    fixed_lengths,
    context_lens,
    launch_lens,
    new_k,
    new_v,
    arena_k,
    arena_v,
    execution_marker,
    previous_cache_rows,
    previous_counts,
    previous_slots,
    fixed_slot_offsets,
    active_mask,
    active_blocks,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    STATE_V_BATCH_STRIDE,
    STATE_V_HEAD_STRIDE,
    STATE_V_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    SLOT_OFFSET_STRIDE,
    MASK_STRIDE,
    BLOCK_STRIDE,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    GROUP_N: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    SCORE_ONLY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    MASK_CAPACITY: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    RESET_BLOCK_N: tl.constexpr,
    RESET_BLOCKS_N: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    SEPARATE_LOCAL_SINK: tl.constexpr,
    KEYS_ARE_MEANS: tl.constexpr,
    REUSE_COARSE: tl.constexpr,
    CANDIDATES_PER_GROUP: tl.constexpr = 8,
):
    """Score current centroids while preparing the persistent fixed mask.

    The score grid already has many independent programs per GQA sequence.
    Those programs distribute local/coarse/block initialization and previous
    union clearing among themselves, so this work adds no separate launch or
    inter-kernel barrier ahead of routing.
    """
    batch_kv = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = batch_kv - batch * KV_HEADS
    physical_sequence = cache_batch * KV_HEADS + kv_head
    lane = tl.arange(0, GROUP_N)
    slot = group * GROUP_N + lane
    valid = slot < state_len

    # Distribute prefix-mask maintenance across the same grid that scores the
    # state. Long-context fixed-mask eligibility always provides enough state
    # groups to cover the bounded local window and prefix block table.
    active_local = (
        tl.minimum(tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT)
        + INCLUDE_NEW
    )
    tl.store(
        active_mask + batch_kv * MASK_STRIDE + slot,
        ((slot < active_local) & (not SEPARATE_LOCAL_SINK)).to(tl.uint8),
        mask=slot < LOCAL_LIMIT,
    )
    tl.store(
        active_mask + batch_kv * MASK_STRIDE + LOCAL_LIMIT + SINK_LEN + slot,
        0 if REUSE_COARSE else 1,
        mask=slot < STATE_CAPACITY,
    )
    prefix_blocks = (LEAF_BEGIN + TILE_SIZE - 1) // TILE_SIZE
    tl.store(
        active_blocks + batch_kv * BLOCK_STRIDE + group,
        (
            ((not SEPARATE_LOCAL_SINK) & (group * TILE_SIZE < LOCAL_LIMIT + SINK_LEN))
            if REUSE_COARSE
            else (
                (not SEPARATE_LOCAL_SINK)
                | (group * TILE_SIZE + TILE_SIZE > LOCAL_LIMIT + SINK_LEN)
            )
        ).to(tl.uint8),
        mask=group < prefix_blocks,
    )

    if group == 0:
        sink_lane = tl.arange(0, 1 if SINK_LEN == 0 else SINK_LEN)
        tl.store(
            active_mask + batch_kv * MASK_STRIDE + LOCAL_LIMIT + sink_lane,
            0 if SEPARATE_LOCAL_SINK else 1,
            mask=sink_lane < SINK_LEN,
        )
        fixed_length = tl.load(fixed_lengths + physical_sequence).to(tl.int32)
        remote_length = tl.maximum(
            fixed_length - (LOCAL_LIMIT if SEPARATE_LOCAL_SINK else 0), 0
        )
        tl.store(context_lens + batch_kv, remote_length)
        tl.store(launch_lens + batch_kv, tl.maximum(remote_length, 1))
        tl.store(execution_marker, 2, mask=batch_kv == 0)
        if INCLUDE_NEW and not SEPARATE_LOCAL_SINK:
            dimension = tl.arange(0, HEAD_DIM)
            current_key = tl.load(
                new_k
                + batch * NEW_K_BATCH_STRIDE
                + kv_head * NEW_K_HEAD_STRIDE
                + dimension
            )
            current_value = tl.load(
                new_v
                + batch * NEW_V_BATCH_STRIDE
                + kv_head * NEW_V_HEAD_STRIDE
                + dimension
            )
            physical_local = (
                LOCAL_OFFSET + physical_sequence * LOCAL_CAPACITY + active_local - 1
            )
            tl.store(
                arena_k + physical_local * HEAD_DIM + dimension,
                current_key,
            )
            tl.store(
                arena_v + physical_local * HEAD_DIM + dimension,
                current_value,
            )

    # One route-score group resets one retained previous-union entry. The
    # reset rank is independent of this program's current centroid tile.
    previous_count = tl.load(previous_counts + batch_kv).to(tl.int32)
    previous_valid = group < previous_count
    previous_slot = tl.load(
        previous_slots + batch_kv * UNION_CAPACITY + group,
        mask=previous_valid,
        other=0,
    ).to(tl.int32)
    previous_valid &= (previous_slot >= 0) & (previous_slot < STATE_CAPACITY)
    safe_previous = tl.where(previous_valid, previous_slot, 0)
    previous_cache_batch = tl.load(previous_cache_rows + batch_kv).to(tl.int64)
    previous_valid &= previous_cache_batch >= 0
    previous_offset_base = (
        tl.maximum(previous_cache_batch, 0) * KV_HEADS + kv_head
    ) * SLOT_OFFSET_STRIDE
    previous_start = tl.load(
        fixed_slot_offsets + previous_offset_base + safe_previous,
        mask=previous_valid,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    previous_stop = tl.load(
        fixed_slot_offsets + previous_offset_base + safe_previous + 1,
        mask=previous_valid,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    previous_leaf_count = tl.where(previous_valid, previous_stop - previous_start, 0)
    reset_lane = tl.arange(0, RESET_BLOCK_N)
    for reset_begin in tl.range(0, previous_leaf_count, RESET_BLOCK_N, num_stages=1):
        reset_offset = reset_begin + reset_lane
        logical_token = LEAF_BEGIN + previous_start + reset_offset
        tl.store(
            active_mask + batch_kv * MASK_STRIDE + logical_token,
            0,
            mask=(
                previous_valid
                & (reset_offset < previous_leaf_count)
                & (logical_token < MASK_CAPACITY)
            ),
        )
    first_block = (LEAF_BEGIN + previous_start) // TILE_SIZE
    last_block = (LEAF_BEGIN + previous_stop + TILE_SIZE - 1) // TILE_SIZE
    reset_block = tl.arange(0, RESET_BLOCKS_N)
    for reset_begin in tl.range(
        0, last_block - first_block, RESET_BLOCKS_N, num_stages=1
    ):
        logical_block = first_block + reset_begin + reset_block
        reset_valid = (
            previous_valid
            & (reset_begin + reset_block < last_block - first_block)
            & (logical_block < (MASK_CAPACITY + TILE_SIZE - 1) // TILE_SIZE)
            & (logical_block * TILE_SIZE >= LEAF_BEGIN)
        )
        tl.store(
            active_blocks + batch_kv * BLOCK_STRIDE + logical_block,
            0,
            mask=reset_valid,
        )

    # Existing GQA-cooperative current-query route scorer.
    q_offset = tl.arange(0, 16)
    query_valid = q_offset < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + q_offset
    query_row = batch * QUERY_HEADS + query_head
    dim = tl.arange(0, HEAD_DIM)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    count = tl.load(
        counts
        + cache_batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=valid,
        other=1.0,
    ).to(tl.float32)
    valid &= count > 0.0
    count = tl.where(valid, count, 1.0)
    keys = tl.load(
        state_k
        + cache_batch * STATE_BATCH_STRIDE
        + kv_head * STATE_HEAD_STRIDE
        + slot[:, None] * STATE_TOKEN_STRIDE
        + dim[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    if KEYS_ARE_MEANS:
        # BF16 cached means are a no-op cast. The same specialization can also
        # benchmark an update-maintained FP8 mean cache promoted immediately
        # before MFMA, isolating its bandwidth benefit from query quantization.
        mean_keys = keys.to(tl.bfloat16)
    else:
        mean_keys = (keys.to(tl.float32) / count[:, None]).to(keys.dtype)
    scores = tl.dot(queries, tl.trans(mean_keys), out_dtype=tl.float32)
    scores = scores * SCALE + tl.log(count)[None, :]
    scores = tl.where(query_valid[:, None] & valid[None, :], scores, -float("inf"))
    route_scores = tl.where(slot[None, :] >= PROTECTED_LEN, scores, -float("inf"))
    if MAX_LEAF_TOKENS:
        route_scores = tl.where(
            count[None, :] < MAX_LEAF_TOKENS,
            route_scores,
            -float("inf"),
        )
    candidate_base = (query_row * MAX_GROUPS + group) * CANDIDATES_PER_GROUP
    packed = _pack_route_score_index(route_scores, slot[None, :])
    block_top = tl.topk(packed, CANDIDATES_PER_GROUP, dim=1)
    block_scores, block_indices = _unpack_route_score_index(block_top)
    rank = tl.arange(0, CANDIDATES_PER_GROUP)
    tl.store(
        candidate_scores + candidate_base[:, None] + rank[None, :],
        block_scores,
        mask=query_valid[:, None],
    )
    tl.store(
        candidate_indices + candidate_base[:, None] + rank[None, :],
        block_indices,
        mask=query_valid[:, None],
    )
    if not SCORE_ONLY:
        values = tl.load(
            state_v
            + cache_batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot[:, None] * STATE_V_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        mean_values = (values.to(tl.float32) / count[:, None]).to(values.dtype)
        maximum = tl.max(scores, axis=1)
        weights = tl.exp(scores - maximum[:, None])
        weights = tl.where(query_valid[:, None] & valid[None, :], weights, 0.0)
        denominator = tl.sum(weights, axis=1)
        weighted_values = tl.dot(
            weights.to(mean_values.dtype), mean_values, out_dtype=tl.float32
        )
        group_row = query_row * MAX_GROUPS + group
        tl.store(
            group_out + group_row[:, None] * HEAD_DIM + dim[None, :],
            tl.where(
                denominator[:, None] > 0.0,
                weighted_values / denominator[:, None],
                0.0,
            ),
            mask=query_valid[:, None],
        )
        tl.store(
            group_lse + group_row,
            tl.where(
                denominator > 0.0,
                maximum + tl.log(denominator),
                -float("inf"),
            ),
            mask=query_valid,
        )


@triton.jit
def _reduce_decode_route_coarse_kernel(
    candidate_scores,
    candidate_indices,
    group_out,
    group_lse,
    top_slots,
    top_scores,
    coarse_out,
    coarse_lse,
    active_groups,
    HEAD_DIM: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    OPEN_COUNT: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    CANDIDATE_TILE: tl.constexpr,
    APPLY_MASS_CUTOFF: tl.constexpr,
    LOG_MASS_FRACTION: tl.constexpr,
    CANDIDATES_PER_GROUP: tl.constexpr = 8,
    EXACT_TOP4: tl.constexpr = False,
):
    query_row = tl.program_id(0).to(tl.int64)
    candidate_offset = tl.arange(0, CANDIDATE_TILE)
    if EXACT_TOP4:
        best_packed = tl.full((4,), -9223372036854775807, tl.int64)
        for candidate_begin in tl.range(
            0,
            active_groups * CANDIDATES_PER_GROUP,
            CANDIDATE_TILE,
            num_stages=1,
        ):
            candidate = candidate_begin + candidate_offset
            valid_candidate = candidate < active_groups * CANDIDATES_PER_GROUP
            scores = tl.load(
                candidate_scores
                + query_row * MAX_GROUPS * CANDIDATES_PER_GROUP
                + candidate,
                mask=valid_candidate,
                other=-float("inf"),
            )
            indices = tl.load(
                candidate_indices
                + query_row * MAX_GROUPS * CANDIDATES_PER_GROUP
                + candidate,
                mask=valid_candidate,
                other=0,
            )
            packed = _pack_route_score_index(scores, indices)
            block_top = tl.topk(packed, 4, dim=0)
            best_packed = tl.topk(tl.interleave(best_packed, block_top), 4, dim=0)
    else:
        best_packed = tl.full((8,), -9223372036854775807, tl.int64)
        for candidate_begin in tl.range(
            0,
            active_groups * CANDIDATES_PER_GROUP,
            CANDIDATE_TILE,
            num_stages=1,
        ):
            candidate = candidate_begin + candidate_offset
            valid_candidate = candidate < active_groups * CANDIDATES_PER_GROUP
            scores = tl.load(
                candidate_scores
                + query_row * MAX_GROUPS * CANDIDATES_PER_GROUP
                + candidate,
                mask=valid_candidate,
                other=-float("inf"),
            )
            indices = tl.load(
                candidate_indices
                + query_row * MAX_GROUPS * CANDIDATES_PER_GROUP
                + candidate,
                mask=valid_candidate,
                other=0,
            )
            packed = _pack_route_score_index(scores, indices)
            block_top = tl.topk(packed, 8, dim=0)
            best_packed = tl.topk(tl.interleave(best_packed, block_top), 8, dim=0)
    best_scores, best_indices = _unpack_route_score_index(best_packed)
    best_valid = (
        (best_scores > -float("inf"))
        & (best_indices >= 0)
        & (best_indices < STATE_CAPACITY)
    )
    if EXACT_TOP4:
        rank = tl.arange(0, 4)
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(best_valid, best_indices, -1),
        )
        tl.store(top_scores + query_row * ROUTE_COUNT + rank, best_scores)
        tl.store(top_slots + query_row * ROUTE_COUNT + 4 + rank, -1)
        tl.store(
            top_scores + query_row * ROUTE_COUNT + 4 + rank,
            -float("inf"),
        )
    elif ROUTE_COUNT == 8:
        rank = tl.arange(0, 8)
        # A routing guard (for example MAX_LEAF_TOKENS) represents excluded
        # centroids with -inf.  Do not let their otherwise arbitrary packed
        # indices leak through when fewer than ROUTE_COUNT finite candidates
        # remain.
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(
                (rank < OPEN_COUNT) & best_valid,
                best_indices,
                -1,
            ),
        )
        tl.store(top_scores + query_row * ROUTE_COUNT + rank, best_scores)

    dim = tl.arange(0, HEAD_DIM)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    for group in tl.range(0, active_groups):
        row = query_row * MAX_GROUPS + group
        current_lse = tl.load(group_lse + row)
        current_out = tl.load(group_out + row * HEAD_DIM + dim)
        new_maximum = tl.maximum(maximum, current_lse)
        old_weight = tl.exp(maximum - new_maximum)
        current_weight = tl.exp(current_lse - new_maximum)
        denominator = denominator * old_weight + current_weight
        accumulator = accumulator * old_weight + current_out * current_weight
        maximum = new_maximum
    full_lse = maximum + tl.log(denominator)
    tl.store(coarse_out + query_row * HEAD_DIM + dim, accumulator / denominator)
    tl.store(
        coarse_lse + query_row,
        full_lse,
    )
    if APPLY_MASS_CUTOFF and ROUTE_COUNT == 8:
        if EXACT_TOP4:
            rank = tl.arange(0, 4)
            tl.store(
                top_slots + query_row * ROUTE_COUNT + rank,
                tl.where(
                    best_valid & (best_scores > full_lse + LOG_MASS_FRACTION),
                    best_indices,
                    -1,
                ),
            )
        else:
            rank = tl.arange(0, 8)
            tl.store(
                top_slots + query_row * ROUTE_COUNT + rank,
                tl.where(
                    (rank < OPEN_COUNT)
                    & best_valid
                    & (best_scores > full_lse + LOG_MASS_FRACTION),
                    best_indices,
                    -1,
                ),
            )


@triton.jit
def _reduce_decode_route_topk_kernel(
    candidate_scores,
    candidate_indices,
    top_slots,
    top_scores,
    active_candidate_groups,
    seen_stamps,
    sequence_epochs,
    union_counts,
    union_slots,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    OPEN_COUNT: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    CANDIDATES_PER_GROUP: tl.constexpr = 8,
    EXACT_TOP4: tl.constexpr = False,
    FLOAT_TOP4: tl.constexpr = False,
    STAMP_SELECTED: tl.constexpr = False,
    FUSE_UNION_BUILD: tl.constexpr = False,
    UNION_SEQUENCE_CAPACITY: tl.constexpr = 1,
    PACKED_CANDIDATES: tl.constexpr = False,
    SORTED_GROUP_MERGE: tl.constexpr = False,
    FLOAT_SCORE_TOP4: tl.constexpr = False,
):
    """Reduce score-only route candidates without serial coarse PV work."""
    query_row = tl.program_id(0).to(tl.int64)
    if EXACT_TOP4:
        if SORTED_GROUP_MERGE:
            # Every producer's four candidates are already sorted. Merge the
            # per-group frontiers instead of sorting the padded concatenation
            # of all 4*groups candidates again.
            group = tl.arange(0, CANDIDATE_BLOCK // CANDIDATES_PER_GROUP)
            group_valid = group < active_candidate_groups
            frontier_rank = tl.zeros(
                (CANDIDATE_BLOCK // CANDIDATES_PER_GROUP,), tl.int32
            )
            output_rank = tl.arange(0, 4)
            best_scores = tl.full((4,), -float("inf"), tl.float32)
            best_indices = tl.full((4,), -1, tl.int64)
            for selected_rank in tl.static_range(0, 4):
                frontier_offset = (
                    query_row * MAX_SEGMENTS * CANDIDATES_PER_GROUP
                    + group * CANDIDATES_PER_GROUP
                    + frontier_rank
                )
                frontier_valid = group_valid & (frontier_rank < CANDIDATES_PER_GROUP)
                if PACKED_CANDIDATES:
                    frontier = tl.load(
                        candidate_indices + frontier_offset,
                        mask=frontier_valid,
                        other=-9187343239835811841,
                    )
                else:
                    frontier_scores = tl.load(
                        candidate_scores + frontier_offset,
                        mask=frontier_valid,
                        other=-float("inf"),
                    )
                    frontier_indices = tl.load(
                        candidate_indices + frontier_offset,
                        mask=frontier_valid,
                        other=0,
                    )
                    frontier = _pack_route_score_index(
                        frontier_scores, frontier_indices
                    )
                selected = tl.max(frontier, axis=0)
                selected_score, selected_index = _unpack_route_score_index(selected)
                best_scores = tl.where(
                    output_rank == selected_rank,
                    selected_score,
                    best_scores,
                )
                best_indices = tl.where(
                    output_rank == selected_rank,
                    selected_index,
                    best_indices,
                )
                selected_group = tl.argmax(frontier, axis=0, tie_break_left=True)
                frontier_rank += (group == selected_group).to(tl.int32)
        else:
            candidate = tl.arange(0, CANDIDATE_BLOCK)
            valid = candidate < active_candidate_groups * CANDIDATES_PER_GROUP
            candidate_offset = (
                query_row * MAX_SEGMENTS * CANDIDATES_PER_GROUP + candidate
            )
            if PACKED_CANDIDATES:
                packed = tl.load(
                    candidate_indices + candidate_offset,
                    mask=valid,
                    other=-9187343239835811841,
                )
            else:
                scores = tl.load(
                    candidate_scores + candidate_offset,
                    mask=valid,
                    other=-float("inf"),
                )
                indices = tl.load(
                    candidate_indices + candidate_offset,
                    mask=valid,
                    other=0,
                )
            if PACKED_CANDIDATES:
                best = tl.topk(packed, 4, dim=0)
                best_scores, best_indices = _unpack_route_score_index(best)
            elif FLOAT_SCORE_TOP4:
                best_scores = tl.topk(scores, 4, dim=0)
                output_rank = tl.arange(0, 4)
                best_indices = tl.full((4,), -1, tl.int64)
                remaining = valid
                indices_i32 = indices.to(tl.int32)
                for selected_rank in tl.static_range(0, 4):
                    selected_score = tl.max(
                        tl.where(
                            output_rank == selected_rank,
                            best_scores,
                            -float("inf"),
                        ),
                        axis=0,
                    )
                    selected_index = tl.min(
                        tl.where(
                            remaining & (scores == selected_score),
                            indices_i32,
                            2_147_483_647,
                        ),
                        axis=0,
                    )
                    best_indices = tl.where(
                        output_rank == selected_rank,
                        selected_index.to(tl.int64),
                        best_indices,
                    )
                    remaining &= indices_i32 != selected_index
            elif FLOAT_TOP4:
                remaining_scores = scores
                candidate_position = tl.arange(0, CANDIDATE_BLOCK)
                best_scores = tl.full((4,), -float("inf"), tl.float32)
                best_indices = tl.full((4,), -1, tl.int64)
                output_rank = tl.arange(0, 4)
                for selected_rank in tl.static_range(0, 4):
                    selected_score = tl.max(remaining_scores, axis=0)
                    selected_position = tl.argmax(
                        remaining_scores, axis=0, tie_break_left=True
                    )
                    selected_index = tl.sum(
                        tl.where(
                            candidate_position == selected_position,
                            indices,
                            0,
                        ),
                        axis=0,
                    ).to(tl.int64)
                    best_scores = tl.where(
                        output_rank == selected_rank,
                        selected_score,
                        best_scores,
                    )
                    best_indices = tl.where(
                        output_rank == selected_rank,
                        selected_index,
                        best_indices,
                    )
                    remaining_scores = tl.where(
                        candidate_position == selected_position,
                        -float("inf"),
                        remaining_scores,
                    )
            else:
                packed = _pack_route_score_index(scores, indices)
                best = tl.topk(packed, 4, dim=0)
                best_scores, best_indices = _unpack_route_score_index(best)
        rank = tl.arange(0, 4)
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(
                (best_scores > -float("inf"))
                & (best_indices >= 0)
                & (best_indices < STATE_CAPACITY),
                best_indices,
                -1,
            ),
        )
        tl.store(top_scores + query_row * ROUTE_COUNT + rank, best_scores)
        tl.store(top_slots + query_row * ROUTE_COUNT + 4 + rank, -1)
        tl.store(
            top_scores + query_row * ROUTE_COUNT + 4 + rank,
            -float("inf"),
        )
    else:
        candidate = tl.arange(0, CANDIDATE_BLOCK)
        valid = candidate < active_candidate_groups * CANDIDATES_PER_GROUP
        candidate_offset = query_row * MAX_SEGMENTS * CANDIDATES_PER_GROUP + candidate
        if PACKED_CANDIDATES:
            packed = tl.load(
                candidate_indices + candidate_offset,
                mask=valid,
                other=-9187343239835811841,
            )
        else:
            scores = tl.load(
                candidate_scores + candidate_offset,
                mask=valid,
                other=-float("inf"),
            )
            indices = tl.load(
                candidate_indices + candidate_offset,
                mask=valid,
                other=0,
            )
        if not PACKED_CANDIDATES:
            packed = _pack_route_score_index(scores, indices)
        best = tl.topk(packed, 8, dim=0)
        best_scores, best_indices = _unpack_route_score_index(best)
        rank = tl.arange(0, ROUTE_COUNT)
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(
                (rank < OPEN_COUNT)
                & (best_scores > -float("inf"))
                & (best_indices >= 0)
                & (best_indices < STATE_CAPACITY),
                best_indices,
                -1,
            ),
        )
        tl.store(top_scores + query_row * ROUTE_COUNT + rank, best_scores)

    if STAMP_SELECTED:
        batch = query_row // QUERY_HEADS
        query_head = query_row - batch * QUERY_HEADS
        kv_head = query_head // KV_GROUP_SIZE
        sequence = batch * KV_HEADS + kv_head
        epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
        selected = (
            (rank < OPEN_COUNT)
            & (best_scores > -float("inf"))
            & (best_indices >= 0)
            & (best_indices < STATE_CAPACITY)
        )
        tl.store(
            seen_stamps + sequence * STATE_CAPACITY + best_indices,
            epoch,
            mask=selected,
        )

    if FUSE_UNION_BUILD:
        batch = query_row // QUERY_HEADS
        query_head = query_row - batch * QUERY_HEADS
        kv_head = query_head // KV_GROUP_SIZE
        sequence = batch * KV_HEADS + kv_head
        union_row_valid = sequence < UNION_SEQUENCE_CAPACITY
        safe_sequence = tl.where(union_row_valid, sequence, 0)
        selected = union_row_valid & (rank < OPEN_COUNT) & (best_scores > -float("inf"))
        slot = tl.where(selected, best_indices, 0).to(tl.int32)
        stamp_pointer = seen_stamps + safe_sequence * STATE_CAPACITY + slot
        epoch = tl.load(
            sequence_epochs + safe_sequence,
            mask=union_row_valid,
            other=0,
        ).to(tl.int32)
        epoch_vector = slot * 0 + epoch
        observed_epoch = tl.load(stamp_pointer, mask=selected, other=epoch_vector)
        old_epoch = tl.atomic_cas(
            stamp_pointer,
            tl.where(selected, observed_epoch, -1),
            epoch_vector,
            sem="relaxed",
        )
        unique = selected & (old_epoch != epoch_vector)
        unique_integer = unique.to(tl.int32)
        local_rank = tl.cumsum(unique_integer, axis=0) - 1
        unique_count = tl.sum(unique_integer, axis=0)
        union_base = tl.atomic_add(
            union_counts + safe_sequence,
            unique_count,
            mask=union_row_valid & (unique_count > 0),
            sem="relaxed",
        ).to(tl.int32)
        union_rank = union_base + local_rank
        tl.store(
            union_slots + safe_sequence * UNION_CAPACITY + union_rank,
            slot,
            mask=unique & (union_rank < UNION_CAPACITY),
        )


@triton.jit
def _reduce_decode_route_coarse_vector_topk_kernel(
    candidate_scores,
    candidate_indices,
    segment_out,
    segment_lse,
    top_slots,
    top_scores,
    coarse_out,
    coarse_lse,
    active_segments,
    active_candidate_groups,
    HEAD_DIM: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    OPEN_COUNT: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    SEGMENT_BLOCK: tl.constexpr,
    APPLY_MASS_CUTOFF: tl.constexpr,
    LOG_MASS_FRACTION: tl.constexpr,
    CANDIDATES_PER_GROUP: tl.constexpr = 8,
    EXACT_TOP4: tl.constexpr = False,
):
    """Reduce route candidates and segment outputs with parallel axes."""
    query_row = tl.program_id(0).to(tl.int64)
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    valid_candidate = candidate < active_candidate_groups * CANDIDATES_PER_GROUP
    scores = tl.load(
        candidate_scores + query_row * MAX_SEGMENTS * CANDIDATES_PER_GROUP + candidate,
        mask=valid_candidate,
        other=-float("inf"),
    )
    indices = tl.load(
        candidate_indices + query_row * MAX_SEGMENTS * CANDIDATES_PER_GROUP + candidate,
        mask=valid_candidate,
        other=0,
    )
    packed = _pack_route_score_index(scores, indices)
    best = tl.topk(packed, 4 if EXACT_TOP4 else 8, dim=0)
    best_scores, best_indices = _unpack_route_score_index(best)
    best_valid = (
        (best_scores > -float("inf"))
        & (best_indices >= 0)
        & (best_indices < STATE_CAPACITY)
    )
    rank = tl.arange(0, 4 if EXACT_TOP4 else 8)
    if ROUTE_COUNT == 8:
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(
                (rank < OPEN_COUNT) & best_valid,
                best_indices,
                -1,
            ),
        )
        tl.store(top_scores + query_row * ROUTE_COUNT + rank, best_scores)
        if EXACT_TOP4:
            tl.store(top_slots + query_row * ROUTE_COUNT + 4 + rank, -1)
            tl.store(
                top_scores + query_row * ROUTE_COUNT + 4 + rank,
                -float("inf"),
            )

    segment = tl.arange(0, SEGMENT_BLOCK)
    valid_segment = segment < active_segments
    segment_scale = tl.load(
        segment_lse + query_row * MAX_SEGMENTS + segment,
        mask=valid_segment,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(segment_scale, axis=0)
    weights = tl.where(valid_segment, tl.exp(segment_scale - maximum), 0.0)
    dim = tl.arange(0, HEAD_DIM)
    partial = tl.load(
        segment_out
        + (query_row * MAX_SEGMENTS + segment[:, None]) * HEAD_DIM
        + dim[None, :],
        mask=valid_segment[:, None],
        other=0.0,
    ).to(tl.float32)
    denominator = tl.sum(weights, axis=0)
    accumulator = tl.sum(partial * weights[:, None], axis=0)
    full_lse = maximum + tl.log(denominator)
    tl.store(coarse_out + query_row * HEAD_DIM + dim, accumulator / denominator)
    tl.store(coarse_lse + query_row, full_lse)
    if APPLY_MASS_CUTOFF and ROUTE_COUNT == 8:
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(
                (rank < OPEN_COUNT)
                & best_valid
                & (best_scores > full_lse + LOG_MASS_FRACTION),
                best_indices,
                -1,
            ),
        )
