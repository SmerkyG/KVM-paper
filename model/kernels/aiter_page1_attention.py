"""Decode-only AITER page-size-one attention with per-KV logit bias.

This is the BF16, paged, segmented decode specialization of AITER's
``kernel_unified_attention_3d``.  It retains the same M=16/N=64 MFMA layout
and online-softmax segmentation, while dropping branches that are constexpr
dead for LOD and adding one bias load per page-size-one K/V row.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def kernel_page1_attention_3d_bias(
    segment_output,
    segment_max,
    segment_exp_sum,
    query,
    key_cache,
    value_cache,
    key_bias,
    block_table,
    cache_indices,
    sequence_lengths,
    scale,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    INDEX_BY_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
):
    """AITER-shaped segmented decode attention over indexed single-token pages."""
    sequence = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(2).to(tl.int64)
    sequence_length = tl.load(sequence_lengths + sequence).to(tl.int32)
    tiles_per_segment = _cdiv(sequence_length, NUM_SEGMENTS * TILE_SIZE)
    tile_begin = segment * tiles_per_segment
    if tile_begin * TILE_SIZE >= sequence_length:
        return

    query_lane = tl.arange(0, BLOCK_M)
    query_valid = query_lane < NUM_QUERY_HEADS
    dimension = tl.arange(0, HEAD_SIZE)
    token_lane = tl.arange(0, TILE_SIZE)
    queries = tl.load(
        query
        + sequence * query_stride_0
        + query_lane[:, None] * query_stride_1
        + dimension[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )

    rcp_ln2: tl.constexpr = 1.4426950408889634
    qk_scale = scale * rcp_ln2
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_SIZE), tl.float32)
    if INDEX_BY_CACHE:
        logical_batch = sequence // KV_HEADS
        kv_head = sequence - logical_batch * KV_HEADS
        cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
        table_sequence = cache_batch * KV_HEADS + kv_head
    else:
        table_sequence = sequence
    table_base = table_sequence * block_table_stride
    tile_count = _cdiv(sequence_length, TILE_SIZE)

    for tile in range(
        tile_begin,
        min((segment + 1) * tiles_per_segment, tile_count),
    ):
        logical_token = tile * TILE_SIZE + token_lane
        token_valid = logical_token < sequence_length
        physical_token = tl.load(
            block_table + table_base + logical_token,
            mask=token_valid,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            key_cache
            + physical_token[None, :] * HEAD_SIZE
            + dimension[:, None],
            mask=token_valid[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(queries.dtype)
        values = tl.load(
            value_cache
            + physical_token[:, None] * HEAD_SIZE
            + dimension[None, :],
            mask=token_valid[:, None],
            other=0.0,
            cache_modifier=".cg",
        ).to(queries.dtype)
        bias = tl.load(
            key_bias + physical_token,
            mask=token_valid,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

        scores = qk_scale * tl.dot(queries, keys)
        scores += bias[None, :] * rcp_ln2
        scores = tl.where(
            query_valid[:, None] & token_valid[None, :],
            scores,
            -float("inf"),
        )
        tile_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, tile_maximum)
        new_maximum = tl.where(new_maximum > -float("inf"), new_maximum, 0.0)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum[:, None])
        denominator = (
            denominator * correction + tl.sum(probabilities, axis=1)
        )
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, acc=accumulator
        )
        maximum = new_maximum

    segment_output_offset = (
        sequence * (NUM_QUERY_HEADS * NUM_SEGMENTS * HEAD_SIZE)
        + query_lane[:, None] * (NUM_SEGMENTS * HEAD_SIZE)
        + segment * HEAD_SIZE
        + dimension[None, :]
    )
    tl.store(
        segment_output + segment_output_offset,
        accumulator,
        mask=query_valid[:, None],
    )
    segment_offset = (
        sequence * (NUM_QUERY_HEADS * NUM_SEGMENTS)
        + query_lane * NUM_SEGMENTS
        + segment
    )
    tl.store(segment_max + segment_offset, maximum, mask=query_valid)
    tl.store(segment_exp_sum + segment_offset, denominator, mask=query_valid)


@triton.jit
def kernel_page1_attention_3d_bias_fixed_mask(
    segment_output,
    segment_max,
    segment_exp_sum,
    query,
    key_cache,
    value_cache,
    key_bias,
    fixed_indices,
    fixed_active_mask,
    fixed_active_blocks,
    fixed_lengths,
    cache_indices,
    scale,
    fixed_index_stride: tl.int64,
    fixed_mask_stride: tl.int64,
    fixed_block_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
):
    """AITER-shaped attention over a persistent, route-masked index list.

    The table is ordered as fixed local capacity, sink, all coarse entries,
    then valid leaves in centroid-major order. A separate, fully parallel
    sparse union-delta kernels reset the previous routes and enable only the
    current selected centroids. A fused context-preparation kernel maintains
    local/sink bytes and prefix block flags, leaving this hot loop with one
    scalar block check and one byte mask load for every surviving tile.
    """
    sequence = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(2).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    physical_sequence = cache_batch * KV_HEADS + kv_head
    sequence_length = tl.load(
        fixed_lengths + physical_sequence
    ).to(tl.int32)
    tiles_per_segment = _cdiv(sequence_length, NUM_SEGMENTS * TILE_SIZE)
    tile_begin = segment * tiles_per_segment
    if tile_begin * TILE_SIZE >= sequence_length:
        return

    query_lane = tl.arange(0, BLOCK_M)
    query_valid = query_lane < NUM_QUERY_HEADS
    dimension = tl.arange(0, HEAD_SIZE)
    token_lane = tl.arange(0, TILE_SIZE)
    queries = tl.load(
        query
        + sequence * query_stride_0
        + query_lane[:, None] * query_stride_1
        + dimension[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )

    rcp_ln2: tl.constexpr = 1.4426950408889634
    qk_scale = scale * rcp_ln2
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_SIZE), tl.float32)
    table_base = physical_sequence * fixed_index_stride
    mask_base = sequence * fixed_mask_stride
    block_base = sequence * fixed_block_stride
    tile_count = _cdiv(sequence_length, TILE_SIZE)

    for tile in range(
        tile_begin,
        min((segment + 1) * tiles_per_segment, tile_count),
    ):
        logical_token = tile * TILE_SIZE + token_lane
        token_valid = logical_token < sequence_length
        tile_has_mass = tl.load(
            fixed_active_blocks + block_base + tile,
            cache_modifier=".cg",
        ).to(tl.int1)

        # This scalar branch is uniform across the workgroup. More than 90%
        # of 64-entry tiles are empty on the measured 64K workloads. The
        # common path reads only the block byte above, avoiding even the lane
        # mask load along with K/V traffic and both MFMA operations.
        if tile_has_mass:
            active = token_valid & tl.load(
                fixed_active_mask + mask_base + logical_token,
                mask=token_valid,
                other=0,
                cache_modifier=".cg",
            ).to(tl.int1)
            physical_token = tl.load(
                fixed_indices + table_base + logical_token,
                # Keep this gather in the native one-dimensional token layout.
                # Newer vLLM Triton lowers ``active`` through the MFMA score
                # layout after the scalar fast-fail reduction, which is not a
                # legal mask for this pointer.  In a surviving tile, reading the
                # inactive INT32 indices is harmless; K/V and bias remain masked.
                mask=token_valid,
                other=0,
                cache_modifier=".cg",
            ).to(tl.int64)
            keys = tl.load(
                key_cache
                + physical_token[None, :] * HEAD_SIZE
                + dimension[:, None],
                mask=active[None, :],
                other=0.0,
                cache_modifier=".cg",
            ).to(queries.dtype)
            values = tl.load(
                value_cache
                + physical_token[:, None] * HEAD_SIZE
                + dimension[None, :],
                mask=active[:, None],
                other=0.0,
                cache_modifier=".cg",
            ).to(queries.dtype)
            bias = tl.load(
                key_bias + physical_token,
                mask=active,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)

            scores = qk_scale * tl.dot(queries, keys)
            scores += bias[None, :] * rcp_ln2
            scores = tl.where(
                query_valid[:, None] & active[None, :],
                scores,
                -float("inf"),
            )
            tile_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, tile_maximum)
            new_maximum = tl.where(
                new_maximum > -float("inf"), new_maximum, 0.0
            )
            correction = tl.math.exp2(maximum - new_maximum)
            probabilities = tl.math.exp2(scores - new_maximum[:, None])
            denominator = (
                denominator * correction + tl.sum(probabilities, axis=1)
            )
            accumulator = accumulator * correction[:, None]
            accumulator = tl.dot(
                probabilities.to(values.dtype), values, acc=accumulator
            )
            maximum = new_maximum

    segment_output_offset = (
        sequence * (NUM_QUERY_HEADS * NUM_SEGMENTS * HEAD_SIZE)
        + query_lane[:, None] * (NUM_SEGMENTS * HEAD_SIZE)
        + segment * HEAD_SIZE
        + dimension[None, :]
    )
    tl.store(
        segment_output + segment_output_offset,
        accumulator,
        mask=query_valid[:, None],
    )
    segment_offset = (
        sequence * (NUM_QUERY_HEADS * NUM_SEGMENTS)
        + query_lane * NUM_SEGMENTS
        + segment
    )
    tl.store(segment_max + segment_offset, maximum, mask=query_valid)
    tl.store(segment_exp_sum + segment_offset, denominator, mask=query_valid)


@triton.jit
def reduce_page1_hip_consumers(
    output,
    previous_total_lse,
    consumer_output,
    consumer_max,
    consumer_exp_sum,
    cache_indices,
    stream_counts,
    opened_counts,
    producer_done,
    overflow_flags,
    query_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    num_consumers: tl.constexpr,
    reduce_consumers: tl.constexpr,
):
    """Reduce persistent HIP consumers directly, then recycle route queues."""
    sequence = tl.program_id(0).to(tl.int64)
    query_in_group = tl.program_id(1).to(tl.int64)
    logical_batch = sequence // kv_heads
    kv_head = sequence - logical_batch * kv_heads
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    query_head = kv_head * (query_heads // kv_heads) + query_in_group
    consumer = tl.arange(0, reduce_consumers)
    valid = consumer < num_consumers
    scalar = (
        (sequence * num_consumers + consumer) * 16 + query_in_group
    )
    maxima = tl.load(
        consumer_max + scalar, mask=valid, other=-float("inf")
    )
    denominators = tl.load(
        consumer_exp_sum + scalar, mask=valid, other=0.0
    )
    maximum = tl.max(maxima, axis=0)
    corrections = tl.where(
        valid & (denominators > 0.0),
        tl.math.exp2(maxima - maximum),
        0.0,
    )
    denominator = tl.sum(denominators * corrections, axis=0)
    dimension = tl.arange(0, head_size)
    partials = tl.load(
        consumer_output
        + scalar[:, None] * head_size
        + dimension[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    numerator = tl.sum(partials * corrections[:, None], axis=0)
    tl.store(
        output
        + (logical_batch * query_heads + query_head) * head_size
        + dimension,
        numerator / denominator,
    )
    ln2: tl.constexpr = 0.6931471805599453
    tl.store(
        previous_total_lse + cache_batch * query_heads + query_head,
        (maximum + tl.log2(denominator)) * ln2,
    )
    if query_in_group == 0:
        tl.store(stream_counts + sequence, 0)
        tl.store(opened_counts + sequence, 0)
        tl.store(producer_done + sequence, 0)
        tl.store(overflow_flags + sequence, 0)


@triton.jit
def init_page1_predicted_mass_union(
    sequence_epochs,
    union_counts,
    union_token_counts,
    SEQUENCES: tl.constexpr,
):
    """Advance the route epoch and clear one GQA-union work queue."""
    sequence = tl.program_id(0)
    if sequence < SEQUENCES:
        epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
        tl.store(sequence_epochs + sequence, epoch + 1)
        tl.store(union_counts + sequence, 0)
        tl.store(union_token_counts + sequence, 0)


@triton.jit
def kernel_page1_predicted_mass_union(
    query,
    coarse_key,
    coarse_bias,
    counts,
    cache_indices,
    previous_total_lse,
    seen_stamps,
    sequence_epochs,
    union_counts,
    union_slots,
    scale,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    count_batch_stride: tl.int64,
    count_head_stride: tl.int64,
    count_token_stride: tl.int64,
    previous_lse_batch_stride: tl.int64,
    previous_lse_head_stride: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    COARSE_OFFSET: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    LOG_MASS_FRACTION: tl.constexpr,
):
    """Route centroids against the preceding token's total attention mass.

    This is the QK half of :func:`kernel_page1_attention_3d_bias`: it keeps
    the same M=16/N=64 page-size-one MFMA layout, but compares each current
    centroid score directly with ``previous_lse + log(mass_fraction)`` and
    compacts the union across the GQA query heads.  No score table or top-k
    reduction is materialized.
    """
    sequence = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head

    query_lane = tl.arange(0, BLOCK_M)
    query_valid = query_lane < NUM_QUERY_HEADS
    dimension = tl.arange(0, HEAD_SIZE)
    token_lane = tl.arange(0, TILE_SIZE)
    token = tile * TILE_SIZE + token_lane
    token_valid = token < STATE_LEN

    queries = tl.load(
        query
        + sequence * query_stride_0
        + query_lane[:, None] * query_stride_1
        + dimension[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
    physical_token = COARSE_OFFSET + kv_row * STATE_CAPACITY + token
    keys = tl.load(
        coarse_key
        + physical_token[None, :] * HEAD_SIZE
        + dimension[:, None],
        mask=token_valid[None, :],
        other=0.0,
        cache_modifier=".cg",
    ).to(queries.dtype)
    bias = tl.load(
        coarse_bias + physical_token,
        mask=token_valid,
        other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    count = tl.load(
        counts
        + cache_batch * count_batch_stride
        + kv_head * count_head_stride
        + token * count_token_stride,
        mask=token_valid,
        other=0.0,
    ).to(tl.float32)

    rcp_ln2: tl.constexpr = 1.4426950408889634
    scores = scale * rcp_ln2 * tl.dot(queries, keys)
    scores += bias[None, :] * rcp_ln2
    previous_lse = tl.load(
        previous_total_lse
        + cache_batch * previous_lse_batch_stride
        + (kv_head * NUM_QUERY_HEADS + query_lane)
            * previous_lse_head_stride,
        mask=query_valid,
        other=float("inf"),
    ).to(tl.float32)
    # The retained mass is stored in natural-log units; scores use log2 to
    # match AITER's exp2 online-softmax path.
    threshold = (previous_lse + LOG_MASS_FRACTION) * rcp_ln2
    selected_by_head = (
        query_valid[:, None]
        & token_valid[None, :]
        & (scores > threshold[:, None])
    )
    selected = tl.sum(selected_by_head.to(tl.int32), axis=0) > 0
    eligible = (
        token_valid
        & (token >= PROTECTED_LEN)
        & (count > 0.0)
        & ((MAX_LEAF_TOKENS <= 0) | (count < MAX_LEAF_TOKENS))
    )
    selected &= eligible

    selected_integer = selected.to(tl.int32)
    destination_in_tile = tl.cumsum(selected_integer, axis=0) - 1
    selected_count = tl.sum(selected_integer, axis=0)
    tile_base = tl.atomic_add(
        union_counts + sequence,
        selected_count,
        sem="relaxed",
    ).to(tl.int32)
    destination = tile_base + destination_in_tile
    epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
    tl.store(
        seen_stamps + sequence * STATE_CAPACITY + token,
        epoch,
        mask=selected & (destination < UNION_CAPACITY),
    )
    tl.store(
        union_slots + sequence * UNION_CAPACITY + destination,
        token,
        mask=selected & (destination < UNION_CAPACITY),
    )


@triton.jit
def kernel_page1_predicted_mass_fixed_prepare(
    query,
    coarse_key,
    coarse_bias,
    counts,
    cache_indices,
    previous_remote_lse,
    seen_stamps,
    sequence_epochs,
    union_counts,
    union_slots,
    remote_group_lse,
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
    previous_union_counts,
    previous_union_slots,
    fixed_slot_offsets,
    active_mask,
    active_blocks,
    scale,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    count_batch_stride: tl.int64,
    count_head_stride: tl.int64,
    count_token_stride: tl.int64,
    previous_lse_batch_stride: tl.int64,
    previous_lse_head_stride: tl.int64,
    remote_lse_row_stride: tl.int64,
    new_k_batch_stride: tl.int64,
    new_k_head_stride: tl.int64,
    new_v_batch_stride: tl.int64,
    new_v_head_stride: tl.int64,
    slot_offset_stride: tl.int64,
    mask_stride: tl.int64,
    block_stride: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    COARSE_OFFSET: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    REMOTE_MAX_GROUPS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    LOG_MASS_FRACTION: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    MASK_CAPACITY: tl.constexpr,
    RESET_BLOCK_N: tl.constexpr,
    RESET_BLOCKS_N: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
):
    """Predicted remote-mass routing plus fixed-mask preparation.

    The threshold denominator is the preceding token's eligible remote-coarse
    LSE, matching the centroid-score numerator. The first token uses one
    current-query winner per N=64 tile, avoiding both an empty bootstrap and a
    global top-k barrier. The same tile grid resets prior leaf ranges and
    prepares the fixed prefix before publishing the current sparse union.
    """
    sequence = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    query_lane = tl.arange(0, BLOCK_M)
    query_valid = query_lane < NUM_QUERY_HEADS
    dimension = tl.arange(0, HEAD_SIZE)
    token_lane = tl.arange(0, TILE_SIZE)
    token = tile * TILE_SIZE + token_lane
    token_valid = token < STATE_LEN

    # Prefix/coarse mask preparation is distributed across route tiles.
    local_length = tl.minimum(
        tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
    )
    active_local = local_length + INCLUDE_NEW
    tl.store(
        active_mask + sequence * mask_stride + token,
        (token < active_local).to(tl.uint8),
        mask=token < LOCAL_LIMIT,
    )
    tl.store(
        active_mask
        + sequence * mask_stride
        + LOCAL_LIMIT
        + SINK_LEN
        + token,
        1,
        mask=token < STATE_CAPACITY,
    )
    route_tiles: tl.constexpr = (STATE_LEN + TILE_SIZE - 1) // TILE_SIZE
    prefix_blocks: tl.constexpr = (LEAF_BEGIN + TILE_SIZE - 1) // TILE_SIZE
    tl.store(
        active_blocks + sequence * block_stride + tile,
        1,
        mask=tile < prefix_blocks,
    )
    second_prefix_block = tile + route_tiles
    tl.store(
        active_blocks + sequence * block_stride + second_prefix_block,
        1,
        mask=second_prefix_block < prefix_blocks,
    )
    if tile == 0:
        sink_lane = tl.arange(0, 1 if SINK_LEN == 0 else SINK_LEN)
        tl.store(
            active_mask
            + sequence * mask_stride
            + LOCAL_LIMIT
            + sink_lane,
            1,
            mask=sink_lane < SINK_LEN,
        )
        fixed_length = tl.load(fixed_lengths + kv_row).to(tl.int32)
        tl.store(context_lens + sequence, fixed_length)
        tl.store(launch_lens + sequence, tl.maximum(fixed_length, 1))
        tl.store(execution_marker, 2, mask=sequence == 0)
        if INCLUDE_NEW:
            current_key = tl.load(
                new_k
                + logical_batch * new_k_batch_stride
                + kv_head * new_k_head_stride
                + dimension
            )
            current_value = tl.load(
                new_v
                + logical_batch * new_v_batch_stride
                + kv_head * new_v_head_stride
                + dimension
            )
            physical_local = (
                LOCAL_OFFSET + kv_row * LOCAL_CAPACITY + local_length
            )
            tl.store(
                arena_k + physical_local * HEAD_SIZE + dimension,
                current_key,
            )
            tl.store(
                arena_v + physical_local * HEAD_SIZE + dimension,
                current_value,
            )

    # Reset one previous-union entry per route tile.
    previous_count = tl.load(previous_union_counts + sequence).to(tl.int32)
    previous_valid = tile < previous_count
    previous_slot = tl.load(
        previous_union_slots + sequence * UNION_CAPACITY + tile,
        mask=previous_valid,
        other=0,
    ).to(tl.int32)
    previous_valid &= (previous_slot >= 0) & (previous_slot < STATE_CAPACITY)
    safe_previous = tl.where(previous_valid, previous_slot, 0)
    previous_cache_batch = tl.load(previous_cache_rows + sequence).to(tl.int64)
    previous_valid &= previous_cache_batch >= 0
    previous_offset_base = (
        tl.maximum(previous_cache_batch, 0) * KV_HEADS + kv_head
    ) * slot_offset_stride
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
    previous_leaf_count = tl.where(
        previous_valid, previous_stop - previous_start, 0
    )
    reset_lane = tl.arange(0, RESET_BLOCK_N)
    for reset_begin in tl.range(
        0, previous_leaf_count, RESET_BLOCK_N, num_stages=1
    ):
        reset_offset = reset_begin + reset_lane
        logical_token = LEAF_BEGIN + previous_start + reset_offset
        tl.store(
            active_mask + sequence * mask_stride + logical_token,
            0,
            mask=(
                previous_valid
                & (reset_offset < previous_leaf_count)
                & (logical_token < MASK_CAPACITY)
            ),
        )
    first_block = (LEAF_BEGIN + previous_start) // TILE_SIZE
    last_block = (
        LEAF_BEGIN + previous_stop + TILE_SIZE - 1
    ) // TILE_SIZE
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
            active_blocks + sequence * block_stride + logical_block,
            0,
            mask=reset_valid,
        )

    # Current-query centroid mass and matching remote-mass denominator.
    queries = tl.load(
        query
        + sequence * query_stride_0
        + query_lane[:, None] * query_stride_1
        + dimension[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
    physical_token = COARSE_OFFSET + kv_row * STATE_CAPACITY + token
    keys = tl.load(
        coarse_key
        + physical_token[None, :] * HEAD_SIZE
        + dimension[:, None],
        mask=token_valid[None, :],
        other=0.0,
        cache_modifier=".cg",
    ).to(queries.dtype)
    bias = tl.load(
        coarse_bias + physical_token,
        mask=token_valid,
        other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    count = tl.load(
        counts
        + cache_batch * count_batch_stride
        + kv_head * count_head_stride
        + token * count_token_stride,
        mask=token_valid,
        other=0.0,
    ).to(tl.float32)
    eligible = (
        token_valid
        & (token >= PROTECTED_LEN)
        & (count > 0.0)
        & ((MAX_LEAF_TOKENS <= 0) | (count < MAX_LEAF_TOKENS))
    )
    rcp_ln2: tl.constexpr = 1.4426950408889634
    scores = scale * rcp_ln2 * tl.dot(queries, keys)
    scores += bias[None, :] * rcp_ln2
    eligible_scores = tl.where(
        query_valid[:, None] & eligible[None, :],
        scores,
        -float("inf"),
    )
    tile_maximum = tl.max(eligible_scores, axis=1)
    tile_denominator = tl.sum(
        tl.where(
            eligible[None, :] & (tile_maximum[:, None] > -float("inf")),
            tl.math.exp2(eligible_scores - tile_maximum[:, None]),
            0.0,
        ),
        axis=1,
    )
    tile_lse = tl.where(
        tile_denominator > 0.0,
        (tile_maximum + tl.log2(tile_denominator)) / rcp_ln2,
        -float("inf"),
    )
    query_head = kv_head * NUM_QUERY_HEADS + query_lane
    remote_row = logical_batch * (KV_HEADS * NUM_QUERY_HEADS) + query_head
    tl.store(
        remote_group_lse + remote_row * remote_lse_row_stride + tile,
        tile_lse,
        mask=query_valid & (tile < REMOTE_MAX_GROUPS),
    )

    previous_lse = tl.load(
        previous_remote_lse
        + cache_batch * previous_lse_batch_stride
        + query_head * previous_lse_head_stride,
        mask=query_valid,
        other=float("inf"),
    ).to(tl.float32)
    previous_valid = (
        query_valid
        & (previous_lse == previous_lse)
        & (previous_lse < float("inf"))
        & (previous_lse > -float("inf"))
    )
    threshold = (previous_lse + LOG_MASS_FRACTION) * rcp_ln2
    selected_by_head = (
        previous_valid[:, None]
        & eligible[None, :]
        & (scores > threshold[:, None])
    )
    selected = tl.sum(selected_by_head.to(tl.int32), axis=0) > 0

    # On an uninitialized first token, retain one current-query winner per
    # tile. This is parallel and conservative without introducing top-k.
    bootstrap = tl.sum((query_valid & ~previous_valid).to(tl.int32), axis=0) > 0
    across_heads = tl.max(eligible_scores, axis=0)
    best_value = tl.max(across_heads, axis=0)
    tied_best = eligible & (across_heads == best_value)
    first_best = tied_best & (tl.cumsum(tied_best.to(tl.int32), axis=0) == 1)
    selected |= bootstrap & first_best & (best_value > -float("inf"))

    selected_integer = selected.to(tl.int32)
    destination_in_tile = tl.cumsum(selected_integer, axis=0) - 1
    selected_count = tl.sum(selected_integer, axis=0)
    tile_base = tl.atomic_add(
        union_counts + sequence,
        selected_count,
        sem="relaxed",
    ).to(tl.int32)
    destination = tile_base + destination_in_tile
    epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
    tl.store(
        seen_stamps + sequence * STATE_CAPACITY + token,
        epoch,
        mask=selected & (destination < UNION_CAPACITY),
    )
    tl.store(
        union_slots + sequence * UNION_CAPACITY + destination,
        token,
        mask=selected & (destination < UNION_CAPACITY),
    )
