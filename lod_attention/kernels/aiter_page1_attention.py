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
    LENGTHS_BY_CACHE: tl.constexpr = False,
    LOCAL_LIMIT: tl.constexpr = 0,
    SINK_LEN: tl.constexpr = 0,
    INCLUDE_NEW: tl.constexpr = False,
):
    """AITER-shaped segmented decode attention over indexed single-token pages."""
    sequence = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(2).to(tl.int64)
    if LENGTHS_BY_CACHE:
        logical_batch = sequence // KV_HEADS
        cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
        local_token_count = (
            tl.minimum(
                tl.load(sequence_lengths + cache_batch).to(tl.int32),
                LOCAL_LIMIT,
            )
            + INCLUDE_NEW
        )
        sequence_length = local_token_count + SINK_LEN
    else:
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
        table_token = logical_token
        if LENGTHS_BY_CACHE:
            # The persistent list reserves LOCAL_LIMIT positions before its
            # fixed sink suffix. Compact the logical local+sink scan without
            # constructing another block table for the current local length.
            table_token = tl.where(
                logical_token < local_token_count,
                logical_token,
                LOCAL_LIMIT + logical_token - local_token_count,
            )
        physical_token = tl.load(
            block_table + table_base + table_token,
            mask=token_valid,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            key_cache + physical_token[None, :] * HEAD_SIZE + dimension[:, None],
            mask=token_valid[None, :],
            other=0.0,
            cache_modifier=".cg",
        ).to(queries.dtype)
        values = tl.load(
            value_cache + physical_token[:, None] * HEAD_SIZE + dimension[None, :],
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
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator)
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
def kernel_exact_tiered_attention_3d(
    segment_output,
    segment_max,
    segment_exp_sum,
    sequence_lengths,
    query,
    sink_k,
    sink_v,
    leaf_k,
    leaf_v,
    local_k,
    local_v,
    new_k,
    new_v,
    cache_indices,
    leaf_lens,
    local_lens,
    scale,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    sink_k_batch_stride: tl.int64,
    sink_k_head_stride: tl.int64,
    sink_k_token_stride: tl.int64,
    sink_v_batch_stride: tl.int64,
    sink_v_head_stride: tl.int64,
    sink_v_token_stride: tl.int64,
    leaf_k_batch_stride: tl.int64,
    leaf_k_head_stride: tl.int64,
    leaf_k_token_stride: tl.int64,
    leaf_v_batch_stride: tl.int64,
    leaf_v_head_stride: tl.int64,
    leaf_v_token_stride: tl.int64,
    local_k_batch_stride: tl.int64,
    local_k_head_stride: tl.int64,
    local_k_token_stride: tl.int64,
    local_v_batch_stride: tl.int64,
    local_v_head_stride: tl.int64,
    local_v_token_stride: tl.int64,
    new_k_batch_stride: tl.int64,
    new_k_head_stride: tl.int64,
    new_v_batch_stride: tl.int64,
    new_v_head_stride: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    STORE_NEW: tl.constexpr,
    LOCAL_LENS_LOGICAL: tl.constexpr,
    MAX_CONTEXT: tl.constexpr = 0,
):
    """Exact decode over the BF16 sink, archived leaves, local tail, and token."""

    sequence = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(2).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    leaf_len = tl.load(leaf_lens + cache_batch).to(tl.int32)
    if LOCAL_LENS_LOGICAL:
        local_len = tl.minimum(
            tl.load(local_lens + logical_batch).to(tl.int32), LOCAL_LIMIT
        )
    else:
        local_len = tl.minimum(
            tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
        )
    sequence_length = SINK_LEN + leaf_len + local_len + INCLUDE_NEW
    use_exact = (MAX_CONTEXT <= 0) | (sequence_length <= MAX_CONTEXT)
    tl.store(sequence_lengths + sequence, tl.where(use_exact, sequence_length, 0))
    if not use_exact:
        return

    dimension = tl.arange(0, HEAD_SIZE)
    if STORE_NEW and segment == 0:
        incoming_key = tl.load(
            new_k
            + logical_batch * new_k_batch_stride
            + kv_head * new_k_head_stride
            + dimension
        )
        incoming_value = tl.load(
            new_v
            + logical_batch * new_v_batch_stride
            + kv_head * new_v_head_stride
            + dimension
        )
        tl.store(
            local_k
            + cache_batch * local_k_batch_stride
            + kv_head * local_k_head_stride
            + local_len * local_k_token_stride
            + dimension,
            incoming_key,
        )
        tl.store(
            local_v
            + cache_batch * local_v_batch_stride
            + kv_head * local_v_head_stride
            + local_len * local_v_token_stride
            + dimension,
            incoming_value,
        )

    tiles_per_segment = _cdiv(sequence_length, NUM_SEGMENTS * TILE_SIZE)
    tile_begin = segment * tiles_per_segment
    if tile_begin * TILE_SIZE >= sequence_length:
        return

    query_lane = tl.arange(0, BLOCK_M)
    query_valid = query_lane < NUM_QUERY_HEADS
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
    tile_count = _cdiv(sequence_length, TILE_SIZE)
    sink_end = SINK_LEN
    leaf_end = sink_end + leaf_len
    local_end = leaf_end + local_len

    for tile in range(
        tile_begin,
        min((segment + 1) * tiles_per_segment, tile_count),
    ):
        token_begin = tile * TILE_SIZE
        token = token_begin + token_lane
        leaf_only = (token_begin >= sink_end) & (
            token_begin + TILE_SIZE <= leaf_end
        )
        if leaf_only:
            # Almost every short-context tile lies wholly inside the
            # chronological leaf archive. Avoid issuing masked loads against
            # the three small suffix stores on this hot path.
            valid = tl.full((TILE_SIZE,), True, tl.int1)
            leaf_token = token - sink_end
            keys = tl.load(
                leaf_k
                + cache_batch * leaf_k_batch_stride
                + kv_head * leaf_k_head_stride
                + leaf_token[None, :] * leaf_k_token_stride
                + dimension[:, None],
                cache_modifier=".cg",
            )
            values = tl.load(
                leaf_v
                + cache_batch * leaf_v_batch_stride
                + kv_head * leaf_v_head_stride
                + leaf_token[:, None] * leaf_v_token_stride
                + dimension[None, :],
                cache_modifier=".cg",
            )
        else:
            valid = token < sequence_length
            is_sink = valid & (token < sink_end)
            is_leaf = valid & (token >= sink_end) & (token < leaf_end)
            is_local = valid & (token >= leaf_end) & (token < local_end)
            is_new = valid & (token >= local_end)
            sink_token = tl.maximum(token, 0)
            leaf_token = tl.maximum(token - sink_end, 0)
            local_token = tl.maximum(token - leaf_end, 0)

            keys = tl.load(
                sink_k
                + cache_batch * sink_k_batch_stride
                + kv_head * sink_k_head_stride
                + sink_token[None, :] * sink_k_token_stride
                + dimension[:, None],
                mask=is_sink[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            values = tl.load(
                sink_v
                + cache_batch * sink_v_batch_stride
                + kv_head * sink_v_head_stride
                + sink_token[:, None] * sink_v_token_stride
                + dimension[None, :],
                mask=is_sink[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            keys += tl.load(
                leaf_k
                + cache_batch * leaf_k_batch_stride
                + kv_head * leaf_k_head_stride
                + leaf_token[None, :] * leaf_k_token_stride
                + dimension[:, None],
                mask=is_leaf[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            values += tl.load(
                leaf_v
                + cache_batch * leaf_v_batch_stride
                + kv_head * leaf_v_head_stride
                + leaf_token[:, None] * leaf_v_token_stride
                + dimension[None, :],
                mask=is_leaf[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            keys += tl.load(
                local_k
                + cache_batch * local_k_batch_stride
                + kv_head * local_k_head_stride
                + local_token[None, :] * local_k_token_stride
                + dimension[:, None],
                mask=is_local[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            values += tl.load(
                local_v
                + cache_batch * local_v_batch_stride
                + kv_head * local_v_head_stride
                + local_token[:, None] * local_v_token_stride
                + dimension[None, :],
                mask=is_local[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            keys += tl.load(
                new_k
                + logical_batch * new_k_batch_stride
                + kv_head * new_k_head_stride
                + dimension[:, None],
                mask=is_new[None, :],
                other=0.0,
            )
            values += tl.load(
                new_v
                + logical_batch * new_v_batch_stride
                + kv_head * new_v_head_stride
                + dimension[None, :],
                mask=is_new[:, None],
                other=0.0,
            )

        scores = qk_scale * tl.dot(queries, keys.to(queries.dtype))
        scores = tl.where(
            query_valid[:, None] & valid[None, :], scores, -float("inf")
        )
        tile_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, tile_maximum)
        new_maximum = tl.where(new_maximum > -float("inf"), new_maximum, 0.0)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
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
def kernel_exact_residual_int4_attention_3d(
    segment_output,
    segment_max,
    segment_exp_sum,
    sequence_lengths,
    query,
    sink_k,
    sink_v,
    quantized_leaf_k,
    quantized_leaf_v,
    page_indices,
    page_counts,
    next_page,
    page_k_scales,
    page_v_scales,
    quantized_page_sum_k,
    quantized_page_sum_v,
    page_sum_k_scales,
    page_sum_v_scales,
    local_k,
    local_v,
    new_k,
    new_v,
    cache_indices,
    leaf_lens,
    local_lens,
    scale,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    sink_k_batch_stride: tl.int64,
    sink_k_head_stride: tl.int64,
    sink_k_token_stride: tl.int64,
    sink_v_batch_stride: tl.int64,
    sink_v_head_stride: tl.int64,
    sink_v_token_stride: tl.int64,
    local_k_batch_stride: tl.int64,
    local_k_head_stride: tl.int64,
    local_k_token_stride: tl.int64,
    local_v_batch_stride: tl.int64,
    local_v_head_stride: tl.int64,
    local_v_token_stride: tl.int64,
    new_k_batch_stride: tl.int64,
    new_k_head_stride: tl.int64,
    new_v_batch_stride: tl.int64,
    new_v_head_stride: tl.int64,
    NUM_QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    STORE_NEW: tl.constexpr,
    LOCAL_LENS_LOGICAL: tl.constexpr,
    MAX_CONTEXT: tl.constexpr,
):
    """Exact short decode over every residual-INT4 semantic page.

    Physical pages are scanned directly rather than routed through centroids.
    Empty lanes in partially filled semantic pages are masked, so the result is
    full chronological attention modulo the cache's documented INT4 residual
    quantization.
    """

    sequence = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(2).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    leaf_len = tl.load(leaf_lens + cache_batch).to(tl.int32)
    if LOCAL_LENS_LOGICAL:
        local_len = tl.minimum(
            tl.load(local_lens + logical_batch).to(tl.int32), LOCAL_LIMIT
        )
    else:
        local_len = tl.minimum(
            tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
        )
    page_count = tl.minimum(tl.load(next_page + kv_row).to(tl.int32), PAGE_CAPACITY)
    page_tokens = page_count * PAGE_SIZE
    actual_context = SINK_LEN + leaf_len + local_len + INCLUDE_NEW
    work_length = page_tokens + SINK_LEN + local_len + INCLUDE_NEW
    use_exact = actual_context <= MAX_CONTEXT
    tl.store(sequence_lengths + sequence, tl.where(use_exact, work_length, 0))
    if not use_exact:
        return

    tiles_per_segment = _cdiv(work_length, NUM_SEGMENTS * TILE_SIZE)
    tile_begin = segment * tiles_per_segment
    if tile_begin * TILE_SIZE >= work_length:
        return

    query_lane = tl.arange(0, BLOCK_M)
    query_valid = query_lane < NUM_QUERY_HEADS
    dimension = tl.arange(0, HEAD_SIZE)
    if STORE_NEW and segment == 0:
        incoming_key = tl.load(
            new_k
            + logical_batch * new_k_batch_stride
            + kv_head * new_k_head_stride
            + dimension
        )
        incoming_value = tl.load(
            new_v
            + logical_batch * new_v_batch_stride
            + kv_head * new_v_head_stride
            + dimension
        )
        tl.store(
            local_k
            + cache_batch * local_k_batch_stride
            + kv_head * local_k_head_stride
            + local_len * local_k_token_stride
            + dimension,
            incoming_key,
        )
        tl.store(
            local_v
            + cache_batch * local_v_batch_stride
            + kv_head * local_v_head_stride
            + local_len * local_v_token_stride
            + dimension,
            incoming_value,
        )
    packed_dimension = dimension // 2
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
    tile_count = _cdiv(work_length, TILE_SIZE)

    for tile in range(
        tile_begin,
        min((segment + 1) * tiles_per_segment, tile_count),
    ):
        token_begin = tile * TILE_SIZE
        page_tile = token_begin < page_tokens
        if page_tile:
            page_id = tile.to(tl.int64)
            page_row = kv_row * PAGE_CAPACITY + page_id
            populated = tl.load(page_counts + page_row).to(tl.int32)
            valid = token_lane < populated
            physical_token = page_row * PAGE_SIZE + token_lane
            leaf_index = tl.load(
                page_indices + physical_token, mask=valid, other=0
            ).to(tl.int64)
            valid &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            storage_token = kv_row * LEAF_CAPACITY + leaf_index

            packed_keys = tl.load(
                quantized_leaf_k
                + storage_token[None, :] * (HEAD_SIZE // 2)
                + packed_dimension[:, None],
                mask=valid[None, :],
                other=0,
                cache_modifier=".cg",
            ).to(tl.int32)
            packed_values = tl.load(
                quantized_leaf_v
                + storage_token[:, None] * (HEAD_SIZE // 2)
                + packed_dimension[None, :],
                mask=valid[:, None],
                other=0,
                cache_modifier=".cg",
            ).to(tl.int32)
            shift = (dimension & 1) * 4
            key_code = ((packed_keys >> shift[:, None]) & 15) - 8
            value_code = ((packed_values >> shift[None, :]) & 15) - 8

            scale_row = page_row * (HEAD_SIZE // QUANT_GROUP_SIZE)
            key_scale = tl.load(
                page_k_scales + scale_row + dimension // QUANT_GROUP_SIZE
            ).to(tl.float32)
            value_scale = tl.load(
                page_v_scales + scale_row + dimension // QUANT_GROUP_SIZE
            ).to(tl.float32)
            key_sum_code = tl.load(
                quantized_page_sum_k + page_row * HEAD_SIZE + dimension
            ).to(tl.float32)
            value_sum_code = tl.load(
                quantized_page_sum_v + page_row * HEAD_SIZE + dimension
            ).to(tl.float32)
            key_sum_scale = tl.load(
                page_sum_k_scales + scale_row + dimension // QUANT_GROUP_SIZE
            ).to(tl.float32)
            value_sum_scale = tl.load(
                page_sum_v_scales + scale_row + dimension // QUANT_GROUP_SIZE
            ).to(tl.float32)
            inverse_count = 1.0 / tl.maximum(populated.to(tl.float32), 1.0)
            key_anchor = key_sum_code * key_sum_scale * inverse_count
            value_anchor = value_sum_code * value_sum_scale * inverse_count
            keys = (
                key_code.to(tl.float32) * key_scale[:, None]
                + key_anchor[:, None]
            ).to(tl.bfloat16)
            values = (
                value_code.to(tl.float32) * value_scale[None, :]
                + value_anchor[None, :]
            ).to(tl.bfloat16)
        else:
            suffix_token = token_begin - page_tokens + token_lane
            suffix_end = SINK_LEN + local_len
            valid = suffix_token < suffix_end + INCLUDE_NEW
            is_sink = valid & (suffix_token < SINK_LEN)
            is_local = (
                valid
                & (suffix_token >= SINK_LEN)
                & (suffix_token < suffix_end)
            )
            is_new = valid & (suffix_token >= suffix_end)
            sink_token = tl.maximum(suffix_token, 0)
            local_token = tl.maximum(suffix_token - SINK_LEN, 0)
            keys = tl.load(
                sink_k
                + cache_batch * sink_k_batch_stride
                + kv_head * sink_k_head_stride
                + sink_token[None, :] * sink_k_token_stride
                + dimension[:, None],
                mask=is_sink[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            values = tl.load(
                sink_v
                + cache_batch * sink_v_batch_stride
                + kv_head * sink_v_head_stride
                + sink_token[:, None] * sink_v_token_stride
                + dimension[None, :],
                mask=is_sink[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            keys += tl.load(
                local_k
                + cache_batch * local_k_batch_stride
                + kv_head * local_k_head_stride
                + local_token[None, :] * local_k_token_stride
                + dimension[:, None],
                mask=is_local[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            values += tl.load(
                local_v
                + cache_batch * local_v_batch_stride
                + kv_head * local_v_head_stride
                + local_token[:, None] * local_v_token_stride
                + dimension[None, :],
                mask=is_local[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            keys += tl.load(
                new_k
                + logical_batch * new_k_batch_stride
                + kv_head * new_k_head_stride
                + dimension[:, None],
                mask=is_new[None, :],
                other=0.0,
            )
            values += tl.load(
                new_v
                + logical_batch * new_v_batch_stride
                + kv_head * new_v_head_stride
                + dimension[None, :],
                mask=is_new[:, None],
                other=0.0,
            )

        scores = qk_scale * tl.dot(queries, keys.to(queries.dtype))
        scores = tl.where(
            query_valid[:, None] & valid[None, :], scores, -float("inf")
        )
        tile_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, tile_maximum)
        new_maximum = tl.where(new_maximum > -float("inf"), new_maximum, 0.0)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum[:, None])
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
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
    PREFIX_SKIP: tl.constexpr = 0,
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
    full_sequence_length = tl.load(fixed_lengths + physical_sequence).to(tl.int32)
    sequence_length = tl.maximum(full_sequence_length - PREFIX_SKIP, 0)
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
        remote_token = tile * TILE_SIZE + token_lane
        token_valid = remote_token < sequence_length
        logical_token = PREFIX_SKIP + remote_token
        tile_has_mass = tl.load(
            fixed_active_blocks + block_base + PREFIX_SKIP // TILE_SIZE + tile,
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
                key_cache + physical_token[None, :] * HEAD_SIZE + dimension[:, None],
                mask=active[None, :],
                other=0.0,
                cache_modifier=".cg",
            ).to(queries.dtype)
            values = tl.load(
                value_cache + physical_token[:, None] * HEAD_SIZE + dimension[None, :],
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
            new_maximum = tl.where(new_maximum > -float("inf"), new_maximum, 0.0)
            correction = tl.math.exp2(maximum - new_maximum)
            probabilities = tl.math.exp2(scores - new_maximum[:, None])
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
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
