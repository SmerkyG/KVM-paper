"""Exact local/leaf attention and output-reduction kernels for LoD decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._paged_common import _lookup_page_id, _online_softmax_update


@triton.jit
def _wide_gqa_local_scores_kernel(
    q,
    cache_indices,
    local_lens,
    local_k,
    new_k,
    scores_out,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    SCORE_BATCH_STRIDE,
    SCORE_HEAD_STRIDE,
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    LOCAL_LENS_LOGICAL: tl.constexpr,
):
    """MFMA QK pass for wide-head local decode, sharing K across GQA heads."""
    batch_kv = tl.program_id(0).to(tl.int64)
    token_block = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    active_len = tl.load(
        local_lens + (batch if LOCAL_LENS_LOGICAL else cache_batch)
    ).to(tl.int32)
    query_offset = tl.arange(0, BLOCK_M)
    query_valid = query_offset < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_offset
    dim = tl.arange(0, HEAD_DIM)
    queries = tl.load(
        q + (batch * QUERY_HEADS + query_head[:, None]) * HEAD_DIM + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    token = token_block * BLOCK_N + tl.arange(0, BLOCK_N)
    scanned_len = tl.minimum(active_len, local_len)
    current = INCLUDE_NEW & (token == local_len)
    keys = tl.load(
        local_k
        + cache_batch * LOCAL_K_BATCH_STRIDE
        + kv_head * LOCAL_K_HEAD_STRIDE
        + token[:, None] * LOCAL_K_TOKEN_STRIDE
        + dim[None, :],
        mask=(token < scanned_len)[:, None],
        other=0.0,
    )
    if INCLUDE_NEW:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
        )
        keys = tl.where(current[:, None], current_key[None, :], keys)
    score = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32) * SCALE
    valid = query_valid[:, None] & ((token < scanned_len)[None, :] | current[None, :])
    score = tl.where(valid, score, -float("inf"))
    tl.store(
        scores_out
        + batch * SCORE_BATCH_STRIDE
        + query_head[:, None] * SCORE_HEAD_STRIDE
        + token[None, :],
        score,
        mask=query_valid[:, None] & (token[None, :] <= local_len),
    )


@triton.jit
def _wide_gqa_local_value_kernel(
    cache_indices,
    local_lens,
    local_k,
    local_v,
    new_k,
    new_v,
    scores,
    out,
    lse_out,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE,
    LOCAL_V_BATCH_STRIDE,
    LOCAL_V_HEAD_STRIDE,
    LOCAL_V_TOKEN_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    SCORE_BATCH_STRIDE,
    SCORE_HEAD_STRIDE,
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    LOCAL_LENS_LOGICAL: tl.constexpr,
):
    """Tiled MFMA PV pass and current-token append for wide local decode."""
    batch_kv = tl.program_id(0).to(tl.int64)
    dim_block = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    active_len = tl.load(
        local_lens + (batch if LOCAL_LENS_LOGICAL else cache_batch)
    ).to(tl.int32)
    query_offset = tl.arange(0, BLOCK_M)
    query_valid = query_offset < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_offset
    dim = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    token_offset = tl.arange(0, BLOCK_K)
    result = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    scanned_len = tl.minimum(active_len, local_len)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    # Each output-dimension tile reconstructs the small row statistics.  This
    # duplicates only score reads/reductions and avoids a separate global
    # softmax launch plus the normalized-probability write/read round trip.
    for begin in tl.range(0, local_len + 1, BLOCK_K):
        token = begin + token_offset
        visible = (token < scanned_len) | (INCLUDE_NEW & (token == local_len))
        score = tl.load(
            scores
            + batch * SCORE_BATCH_STRIDE
            + query_head[:, None] * SCORE_HEAD_STRIDE
            + token[None, :],
            mask=query_valid[:, None] & visible[None, :],
            other=-float("inf"),
        )
        block_maximum = tl.max(score, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        denominator = denominator * tl.exp(maximum - new_maximum) + tl.sum(
            tl.exp(score - new_maximum[:, None]), axis=1
        )
        maximum = new_maximum
    for begin in tl.range(0, local_len + 1, BLOCK_K):
        token = begin + token_offset
        visible = (token < scanned_len) | (INCLUDE_NEW & (token == local_len))
        score = tl.load(
            scores
            + batch * SCORE_BATCH_STRIDE
            + query_head[:, None] * SCORE_HEAD_STRIDE
            + token[None, :],
            mask=query_valid[:, None] & visible[None, :],
            other=-float("inf"),
        )
        probabilities = (tl.exp(score - maximum[:, None]) / denominator[:, None]).to(
            tl.bfloat16
        )
        values = tl.load(
            local_v
            + cache_batch * LOCAL_V_BATCH_STRIDE
            + kv_head * LOCAL_V_HEAD_STRIDE
            + token[:, None] * LOCAL_V_TOKEN_STRIDE
            + dim[None, :],
            mask=(token < scanned_len)[:, None] & (dim[None, :] < HEAD_DIM),
            other=0.0,
        )
        if INCLUDE_NEW:
            current_value = tl.load(
                new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim,
                mask=dim < HEAD_DIM,
                other=0.0,
            )
            values = tl.where(
                (token == local_len)[:, None], current_value[None, :], values
            )
        result += tl.dot(probabilities, values, out_dtype=tl.float32)
    tl.store(
        out + (batch * QUERY_HEADS + query_head[:, None]) * HEAD_DIM + dim[None, :],
        result,
        mask=query_valid[:, None] & (dim[None, :] < HEAD_DIM),
    )
    if dim_block == 0:
        tl.store(
            lse_out + batch * QUERY_HEADS + query_head,
            maximum + tl.log(denominator),
            mask=query_valid,
        )
    if INCLUDE_NEW:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim,
            mask=dim < HEAD_DIM,
            other=0.0,
        )
        current_value = tl.load(
            new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim,
            mask=dim < HEAD_DIM,
            other=0.0,
        )
        tl.store(
            local_k
            + cache_batch * LOCAL_K_BATCH_STRIDE
            + kv_head * LOCAL_K_HEAD_STRIDE
            + active_len * LOCAL_K_TOKEN_STRIDE
            + dim,
            current_key,
            mask=dim < HEAD_DIM,
        )
        tl.store(
            local_v
            + cache_batch * LOCAL_V_BATCH_STRIDE
            + kv_head * LOCAL_V_HEAD_STRIDE
            + active_len * LOCAL_V_TOKEN_STRIDE
            + dim,
            current_value,
            mask=dim < HEAD_DIM,
        )


@triton.jit
def _split_decode_paged_lod_attention_kernel(
    q,
    cache_indices,
    local_lens,
    state_k,
    state_v,
    counts,
    local_k,
    local_v,
    page_k,
    page_v,
    page_indices,
    page_k_scales,
    page_v_scales,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    top_slots,
    new_k,
    new_v,
    partial_out,
    partial_lse,
    top_scores,
    coarse_out,
    coarse_lse,
    output,
    completion,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    STATE_V_BATCH_STRIDE,
    STATE_V_HEAD_STRIDE,
    STATE_V_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE,
    LOCAL_V_BATCH_STRIDE,
    LOCAL_V_HEAD_STRIDE,
    LOCAL_V_TOKEN_STRIDE,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    state_len,
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_DOT: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    STORE_NEW: tl.constexpr,
    LOCAL_LENS_LOGICAL: tl.constexpr,
    SEPARATE_LOCAL: tl.constexpr,
    FUSE_FINAL_REDUCE: tl.constexpr,
    INDEXED: tl.constexpr,
    INT8_STORAGE: tl.constexpr,
    STRIPE_ROUTE_LEAVES: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    batch = query_row // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    active_local_len = tl.load(
        local_lens + (batch if LOCAL_LENS_LOGICAL else cache_batch)
    ).to(tl.int32)
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = cache_batch * KV_HEADS + kv_head

    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    query = tl.load(q + query_row * HEAD_DIM + dim)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_DIM,), tl.float32)

    # Interleave state tiles across splits so their work stays balanced even
    # when the final state tile is partial.
    for state_begin in tl.range(
        split * BLOCK_N, state_len, SPLITS * BLOCK_N, num_stages=1
    ):
        slot = state_begin + token_offset
        valid = slot < state_len
        routed = tl.zeros((BLOCK_N,), tl.int1)
        for route in tl.static_range(0, ROUTE_COUNT):
            selected = tl.load(
                top_slots
                + batch * TOP_BATCH_STRIDE
                + query_head * TOP_HEAD_STRIDE
                + route
            )
            routed |= slot == selected
        valid &= ~routed
        count = tl.load(
            counts
            + cache_batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE,
            mask=valid,
            other=1.0,
        ).to(tl.float32)
        keys = tl.load(
            state_k
            + cache_batch * STATE_BATCH_STRIDE
            + kv_head * STATE_HEAD_STRIDE
            + slot[:, None] * STATE_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        values = tl.load(
            state_v
            + cache_batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot[:, None] * STATE_V_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        mean_keys = (keys.to(tl.float32) / count[:, None]).to(keys.dtype)
        mean_values = (values.to(tl.float32) / count[:, None]).to(values.dtype)
        if USE_DOT:
            scores = tl.dot(query[None, :], tl.trans(mean_keys), out_dtype=tl.float32)
            scores = tl.reshape(scores, (BLOCK_N,))
        else:
            scores = tl.sum(
                mean_keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
            )
        scores *= SCALE_LOG2
        scores += tl.math.log2(count)
        maximum, denominator, accumulator = _online_softmax_update(
            scores,
            mean_values,
            valid,
            maximum,
            denominator,
            accumulator,
            USE_DOT,
        )

    # In the balanced path every routed posting list is striped over every
    # split.  This keeps one unusually large selected centroid from setting
    # the latency of the entire attention call.  The legacy path assigns one
    # complete route to one split and is retained as an explicit benchmark
    # control.
    for route in tl.static_range(0, ROUTE_COUNT):
        routed_slot = tl.load(
            top_slots + batch * TOP_BATCH_STRIDE + query_head * TOP_HEAD_STRIDE + route
        ).to(tl.int64)
        slot_valid = routed_slot >= 0
        slot = tl.where(slot_valid, routed_slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=slot_valid,
            other=0,
        ).to(tl.int32)
        if STRIPE_ROUTE_LEAVES:
            key_begin_offset = split * BLOCK_N
            key_begin_stride = SPLITS * BLOCK_N
        else:
            key_count = tl.where(split == route % SPLITS, key_count, 0)
            key_begin_offset = 0
            key_begin_stride = BLOCK_N
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        for key_begin in tl.range(
            key_begin_offset,
            key_count,
            key_begin_stride,
            num_stages=1,
        ):
            logical_key = key_begin + token_offset
            valid = logical_key < key_count
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(page_table + page_ordinal, mask=valid, other=0).to(
                    tl.int64
                )
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            page_valid = valid & (page_id >= 0) & (page_id < PAGE_CAPACITY)
            page_id = tl.where(page_valid, page_id, 0)
            physical_token = (
                kv_row * PAGE_CAPACITY + page_id
            ) * PAGE_SIZE + within_page
            if INDEXED:
                leaf_index = tl.load(
                    page_indices + physical_token, mask=page_valid, other=0
                ).to(tl.int64)
                valid = page_valid & (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
                # The subsequent K/V loads are masked by ``valid``. Avoid a
                # redundant select here: for BLOCK_N > 16 Triton can assign
                # the page-derived mask and loaded leaf index incompatible
                # layouts, causing RemoveLayoutConversions to fail in vLLM.
                storage_token = kv_row * LEAF_CAPACITY + leaf_index
            else:
                valid = page_valid
                storage_token = physical_token
            keys = tl.load(
                page_k + storage_token[:, None] * HEAD_DIM + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            values = tl.load(
                page_v + storage_token[:, None] * VALUE_DIM + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if INT8_STORAGE:
                key_scale = tl.load(
                    page_k_scales + storage_token, mask=valid, other=0.0
                ).to(tl.float32)
                value_scale = tl.load(
                    page_v_scales + storage_token, mask=valid, other=0.0
                ).to(tl.float32)
                keys = keys.to(tl.float32) * key_scale[:, None]
                values = values.to(tl.float32) * value_scale[:, None]
            if USE_DOT:
                scores = tl.dot(query[None, :], tl.trans(keys), out_dtype=tl.float32)
                scores = tl.reshape(scores, (BLOCK_N,))
            else:
                scores = tl.sum(
                    keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
                )
            scores *= SCALE_LOG2
            maximum, denominator, accumulator = _online_softmax_update(
                scores,
                values,
                valid,
                maximum,
                denominator,
                accumulator,
                USE_DOT,
            )

    if not SEPARATE_LOCAL:
        for local_begin in tl.range(
            split * BLOCK_N, local_len, SPLITS * BLOCK_N, num_stages=1
        ):
            token = local_begin + token_offset
            valid = token < active_local_len
            keys = tl.load(
                local_k
                + cache_batch * LOCAL_K_BATCH_STRIDE
                + kv_head * LOCAL_K_HEAD_STRIDE
                + token[:, None] * LOCAL_K_TOKEN_STRIDE
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            values = tl.load(
                local_v
                + cache_batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + token[:, None] * LOCAL_V_TOKEN_STRIDE
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if USE_DOT:
                scores = tl.dot(query[None, :], tl.trans(keys), out_dtype=tl.float32)
                scores = tl.reshape(scores, (BLOCK_N,))
            else:
                scores = tl.sum(
                    keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
                )
            scores *= SCALE_LOG2
            maximum, denominator, accumulator = _online_softmax_update(
                scores,
                values,
                valid,
                maximum,
                denominator,
                accumulator,
                USE_DOT,
            )

    # Assign the current token to split zero so the final LSE reduction counts
    # it once.  That same split persists the KV into the bounded local cache.
    if INCLUDE_NEW and not SEPARATE_LOCAL:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
        )
        current_value = tl.load(
            new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim
        )
        current_score = SCALE_LOG2 * tl.sum(
            current_key.to(tl.float32) * query.to(tl.float32), axis=0
        )
        current_score = tl.where(split == 0, current_score, -float("inf"))
        new_maximum = tl.maximum(maximum, current_score)
        correction = tl.math.exp2(maximum - new_maximum)
        current_weight = tl.math.exp2(current_score - new_maximum)
        denominator = denominator * correction + current_weight
        accumulator = accumulator * correction + current_weight * current_value.to(
            tl.float32
        )
        maximum = new_maximum
        if STORE_NEW and split == 0:
            if query_head % KV_GROUP_SIZE == 0:
                tl.store(
                    local_k
                    + cache_batch * LOCAL_K_BATCH_STRIDE
                    + kv_head * LOCAL_K_HEAD_STRIDE
                    + active_local_len * LOCAL_K_TOKEN_STRIDE
                    + dim,
                    current_key,
                )
                tl.store(
                    local_v
                    + cache_batch * LOCAL_V_BATCH_STRIDE
                    + kv_head * LOCAL_V_HEAD_STRIDE
                    + active_local_len * LOCAL_V_TOKEN_STRIDE
                    + dim,
                    current_value,
                )

    partial_row = query_row * SPLITS + split
    has_mass = denominator > 0.0
    tl.store(
        partial_out + partial_row * VALUE_DIM + dim,
        tl.where(has_mass, accumulator / denominator, 0.0),
    )
    tl.store(
        partial_lse + partial_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )

    if FUSE_FINAL_REDUCE:
        finished = tl.atomic_add(completion + query_row, 1, sem="acq_rel").to(tl.int32)
        if finished == SPLITS - 1:
            full_coarse_lse = tl.load(coarse_lse + query_row)
            remainder_out = tl.load(coarse_out + query_row * HEAD_DIM + dim).to(
                tl.float32
            )
            selected_mass = tl.zeros((), tl.float32)
            selected_value = tl.zeros((HEAD_DIM,), tl.float32)
            for route in tl.static_range(0, ROUTE_COUNT):
                slot = tl.load(
                    top_slots
                    + batch * TOP_BATCH_STRIDE
                    + query_head * TOP_HEAD_STRIDE
                    + route
                ).to(tl.int64)
                valid_slot = slot >= 0
                slot = tl.where(valid_slot, slot, 0)
                count = tl.load(
                    counts
                    + cache_batch * COUNT_BATCH_STRIDE
                    + kv_head * COUNT_HEAD_STRIDE
                    + slot * COUNT_TOKEN_STRIDE
                ).to(tl.float32)
                value = (
                    tl.load(
                        state_v
                        + cache_batch * STATE_V_BATCH_STRIDE
                        + kv_head * STATE_V_HEAD_STRIDE
                        + slot * STATE_V_TOKEN_STRIDE
                        + dim
                    ).to(tl.float32)
                    / count
                )
                score = tl.load(top_scores + query_row * ROUTE_COUNT + route)
                mass = tl.where(valid_slot, tl.exp(score - full_coarse_lse), 0.0)
                selected_mass += mass
                selected_value += mass * value
            remainder_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
            remainder_out = (remainder_out - selected_value) / remainder_mass
            remainder_lse = full_coarse_lse + tl.log(remainder_mass)

            split_offsets = tl.arange(0, SPLITS)
            split_lse = tl.load(partial_lse + query_row * SPLITS + split_offsets)
            merge_maximum = tl.maximum(remainder_lse, tl.max(split_lse, axis=0))
            remainder_weight = tl.exp(remainder_lse - merge_maximum)
            split_weights = tl.exp(split_lse - merge_maximum)
            merge_denominator = remainder_weight + tl.sum(split_weights, axis=0)
            merge_accumulator = remainder_weight * remainder_out
            for merge_split in tl.static_range(0, SPLITS):
                split_weight = tl.exp(
                    tl.load(partial_lse + query_row * SPLITS + merge_split)
                    - merge_maximum
                )
                split_value = tl.load(
                    partial_out + (query_row * SPLITS + merge_split) * HEAD_DIM + dim
                )
                merge_accumulator += split_weight * split_value
            tl.store(
                output + query_row * HEAD_DIM + dim,
                merge_accumulator / merge_denominator,
            )
            tl.atomic_xchg(completion + query_row, 0, sem="release")


@triton.jit
def _prepare_aiter_fixed_mask_context_kernel(
    cache_indices,
    local_lens,
    fixed_lengths,
    context_lens,
    launch_lens,
    new_k,
    new_v,
    arena_k,
    arena_v,
    execution_marker,
    active_mask,
    active_blocks,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    KV_HEADS: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SINK_LEN: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    LOCAL_MASK_BLOCK: tl.constexpr,
    SINK_MASK_BLOCK: tl.constexpr,
    COARSE_MASK_BLOCK: tl.constexpr,
    PREFIX_BLOCKS_BLOCK: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    SEPARATE_LOCAL_SINK: tl.constexpr,
    REUSE_COARSE: tl.constexpr,
):
    """Map persistent lengths and place the current token in the local arena."""
    sequence = tl.program_id(0).to(tl.int64)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    cache_batch = tl.load(cache_indices + logical_batch).to(tl.int64)
    physical_sequence = cache_batch * KV_HEADS + kv_head
    length = tl.load(fixed_lengths + physical_sequence).to(tl.int32)
    remote_length = tl.maximum(length - (LOCAL_LIMIT if SEPARATE_LOCAL_SINK else 0), 0)
    tl.store(context_lens + sequence, remote_length)
    tl.store(launch_lens + sequence, tl.maximum(remote_length, 1))
    tl.store(execution_marker, 2, mask=sequence == 0)
    local_length = tl.minimum(
        tl.load(local_lens + cache_batch).to(tl.int32), LOCAL_LIMIT
    )
    active_local = local_length + INCLUDE_NEW
    local_token = tl.arange(0, LOCAL_MASK_BLOCK)
    tl.store(
        active_mask + sequence * MASK_STRIDE + local_token,
        ((local_token < active_local) & (not SEPARATE_LOCAL_SINK)).to(tl.uint8),
        mask=local_token < LOCAL_LIMIT,
    )
    sink_token = tl.arange(0, SINK_MASK_BLOCK)
    tl.store(
        active_mask + sequence * MASK_STRIDE + LOCAL_LIMIT + sink_token,
        0 if SEPARATE_LOCAL_SINK else 1,
        mask=sink_token < SINK_LEN,
    )
    # Normally coarse lanes form the always-present residual branch. When the
    # route kernel has retained that branch's output and LSE, keep every coarse
    # lane disabled so this scan contains only local/sink plus exact leaves.
    coarse_lane = tl.arange(0, COARSE_MASK_BLOCK)
    for coarse_begin in tl.range(0, STATE_CAPACITY, COARSE_MASK_BLOCK, num_stages=1):
        coarse_slot = coarse_begin + coarse_lane
        tl.store(
            active_mask + sequence * MASK_STRIDE + LOCAL_LIMIT + SINK_LEN + coarse_slot,
            0 if REUSE_COARSE else 1,
            mask=coarse_slot < STATE_CAPACITY,
        )
    prefix_block = tl.arange(0, PREFIX_BLOCKS_BLOCK)
    tl.store(
        active_blocks + sequence * BLOCK_STRIDE + prefix_block,
        (
            (
                (not SEPARATE_LOCAL_SINK)
                & (prefix_block * TILE_SIZE < LOCAL_LIMIT + SINK_LEN)
            )
            if REUSE_COARSE
            else (
                (not SEPARATE_LOCAL_SINK)
                | (prefix_block * TILE_SIZE + TILE_SIZE > LOCAL_LIMIT + SINK_LEN)
            )
        ).to(tl.uint8),
        mask=prefix_block < (LEAF_BEGIN + TILE_SIZE - 1) // TILE_SIZE,
    )
    if INCLUDE_NEW and not SEPARATE_LOCAL_SINK:
        dimension = tl.arange(0, HEAD_DIM)
        current_key = tl.load(
            new_k
            + logical_batch * NEW_K_BATCH_STRIDE
            + kv_head * NEW_K_HEAD_STRIDE
            + dimension
        )
        current_value = tl.load(
            new_v
            + logical_batch * NEW_V_BATCH_STRIDE
            + kv_head * NEW_V_HEAD_STRIDE
            + dimension
        )
        physical_local = (
            LOCAL_OFFSET + physical_sequence * LOCAL_CAPACITY + local_length
        )
        tl.store(arena_k + physical_local * HEAD_DIM + dimension, current_key)
        tl.store(arena_v + physical_local * HEAD_DIM + dimension, current_value)


@triton.jit
def _materialize_page1_coarse_means_kernel(
    state_k,
    state_v,
    counts,
    coarse_k,
    coarse_v,
    coarse_bias,
    STATE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_row = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    dimension = tl.arange(0, HEAD_DIM)
    active_slot = slot < STATE_CAPACITY
    count = tl.load(
        counts + kv_row * STATE_CAPACITY + slot,
        mask=active_slot,
        other=0.0,
    ).to(tl.float32)
    active = active_slot & (count > 0.0)
    storage = (kv_row * STATE_CAPACITY + slot[:, None]) * HEAD_DIM + dimension
    key_sum = tl.load(state_k + storage, mask=active[:, None], other=0.0)
    value_sum = tl.load(state_v + storage, mask=active[:, None], other=0.0)
    denominator = tl.where(active, count, 1.0)
    tl.store(
        coarse_k + storage,
        key_sum.to(tl.float32) / denominator[:, None],
        mask=active_slot[:, None],
    )
    tl.store(
        coarse_v + storage,
        value_sum.to(tl.float32) / denominator[:, None],
        mask=active_slot[:, None],
    )
    tl.store(
        coarse_bias + kv_row * STATE_CAPACITY + slot,
        tl.where(active, tl.log(count), -float("inf")),
        mask=active_slot,
    )


def materialize_page1_coarse_means(
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    coarse_k: torch.Tensor,
    coarse_v: torch.Tensor,
    coarse_bias: torch.Tensor,
) -> None:
    """Refresh persistent centroid means and their natural-log mass bias."""
    if tuple(state_v.shape) != tuple(state_k.shape) or (
        tuple(coarse_k.shape) != tuple(state_k.shape)
        or tuple(coarse_v.shape) != tuple(state_k.shape)
    ):
        raise ValueError("page-size-one coarse mean tensors have mismatched shapes")
    if tuple(counts.shape) != tuple(state_k.shape[:-1]) + (1,):
        raise ValueError("page-size-one coarse counts have the wrong shape")
    if tuple(coarse_bias.shape) != tuple(state_k.shape[:-1]):
        raise ValueError("page-size-one coarse bias has the wrong shape")
    if coarse_bias.dtype != torch.float16:
        raise TypeError("page-size-one coarse bias must use FP16 storage")
    if not all(
        tensor.is_cuda
        for tensor in (state_k, state_v, counts, coarse_k, coarse_v, coarse_bias)
    ):
        raise ValueError("page-size-one coarse mean refresh requires CUDA tensors")
    if not all(
        tensor.is_contiguous()
        for tensor in (state_k, state_v, counts, coarse_k, coarse_v, coarse_bias)
    ):
        raise ValueError(
            "page-size-one coarse mean refresh requires contiguous tensors"
        )
    batch, kv_heads, state_capacity, head_dim = state_k.shape
    block_n = 8
    _materialize_page1_coarse_means_kernel[
        (batch * kv_heads, triton.cdiv(state_capacity, block_n))
    ](
        state_k,
        state_v,
        counts,
        coarse_k,
        coarse_v,
        coarse_bias,
        STATE_CAPACITY=state_capacity,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        num_warps=4,
    )


@triton.jit
def _reset_aiter_fixed_previous_union_kernel(
    cache_indices,
    previous_cache_rows,
    previous_counts,
    previous_slots,
    fixed_slot_offsets,
    active_mask,
    active_blocks,
    slot_offset_stride: tl.int64,
    mask_stride: tl.int64,
    block_stride: tl.int64,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    MASK_CAPACITY: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCKS_N: tl.constexpr,
):
    """Clear the prior union before current-query routing begins.

    Every leaf byte is cleared when its route leaves the active union.  This
    maintains the invariant that an inactive leaf block contains no stale
    lanes, so the post-route kernel only has to set the current ranges.  The
    reset depends solely on retained previous-token metadata and can overlap
    the current token's coarse scoring and top-k chain.
    """
    sequence = tl.program_id(0).to(tl.int64)
    rank = tl.program_id(1).to(tl.int32)
    previous_count = tl.load(previous_counts + sequence).to(tl.int32)
    previous_valid = rank < previous_count
    previous_slot = tl.load(
        previous_slots + sequence * UNION_CAPACITY + rank,
        mask=previous_valid,
        other=0,
    ).to(tl.int32)
    previous_valid &= (previous_slot >= 0) & (previous_slot < STATE_CAPACITY)
    safe_previous = tl.where(previous_valid, previous_slot, 0)
    logical_batch = sequence // KV_HEADS
    kv_head = sequence - logical_batch * KV_HEADS
    # Keep this load in the dependency chain so graph-captured scheduler row
    # remaps are observed before the reset stream publishes completion.
    _ = tl.load(cache_indices + logical_batch).to(tl.int64)
    previous_cache_batch = tl.load(previous_cache_rows + sequence).to(tl.int64)
    previous_valid &= previous_cache_batch >= 0
    previous_offset_base = (
        tl.maximum(previous_cache_batch, 0) * KV_HEADS + kv_head
    ) * slot_offset_stride

    # Restore the previous union's coarse entries. Empty centroids carry -inf
    # bias, so enabling them is harmless and avoids another count lookup.
    previous_coarse = LEAF_BEGIN - STATE_CAPACITY + safe_previous
    tl.store(
        active_mask + sequence * mask_stride + previous_coarse,
        1,
        mask=previous_valid & (previous_coarse < MASK_CAPACITY),
    )

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
    previous_first_block = (LEAF_BEGIN + previous_start) // TILE_SIZE
    previous_last_block = (LEAF_BEGIN + previous_stop + TILE_SIZE - 1) // TILE_SIZE

    # Clear exact leaf lanes first.  Different centroids can share a physical
    # attention tile, but all stores write zero, so their overlap is benign.
    leaf_count = tl.where(previous_valid, previous_stop - previous_start, 0)
    token = tl.arange(0, BLOCK_N)
    for begin in tl.range(0, leaf_count, BLOCK_N, num_stages=1):
        offset = begin + token
        logical_token = LEAF_BEGIN + previous_start + offset
        tl.store(
            active_mask + sequence * mask_stride + logical_token,
            0,
            mask=(
                previous_valid & (offset < leaf_count) & (logical_token < MASK_CAPACITY)
            ),
        )

    block = tl.arange(0, BLOCKS_N)
    for begin in tl.range(
        0, previous_last_block - previous_first_block, BLOCKS_N, num_stages=1
    ):
        logical_block = previous_first_block + begin + block
        valid = (
            previous_valid
            & (begin + block < previous_last_block - previous_first_block)
            & (logical_block < (MASK_CAPACITY + TILE_SIZE - 1) // TILE_SIZE)
            & (logical_block * TILE_SIZE >= LEAF_BEGIN)
        )
        tl.store(
            active_blocks + sequence * block_stride + logical_block,
            0,
            mask=valid,
        )


@triton.jit
def _apply_aiter_fixed_direct_routes_kernel(
    top_slots,
    cache_indices,
    union_counts,
    union_token_counts,
    sequence_epochs,
    previous_counts,
    previous_slots,
    previous_cache_rows,
    fixed_slot_offsets,
    active_mask,
    active_blocks,
    execution_geometry,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    slot_offset_stride: tl.int64,
    mask_stride: tl.int64,
    block_stride: tl.int64,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    MASK_CAPACITY: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCKS_N: tl.constexpr,
    EFFECTIVE_SEGMENTS: tl.constexpr,
    SPLIT_D_REDUCE: tl.constexpr,
    PRESERVE_UNION_METADATA: tl.constexpr,
):
    """Activate head routes directly when deduplicating would serialize B1.

    Every write is idempotent: duplicate routes only clear the same coarse byte
    and enable the same leaf range. Retaining all head-route pairs also makes
    the next-token reset safe without constructing a compact union first.
    """
    sequence = tl.program_id(0).to(tl.int64)
    candidate = tl.program_id(1).to(tl.int32)
    batch = sequence // KV_HEADS
    kv_head = sequence - batch * KV_HEADS
    query_lane = candidate // ROUTE_COUNT
    route = candidate - query_lane * ROUTE_COUNT
    query_head = kv_head * KV_GROUP_SIZE + query_lane
    slot = tl.load(
        top_slots + batch * TOP_BATCH_STRIDE + query_head * TOP_HEAD_STRIDE + route,
        mask=candidate < KV_GROUP_SIZE * ROUTE_COUNT,
        other=-1,
    ).to(tl.int32)
    slot_valid = (slot >= 0) & (slot < STATE_CAPACITY)
    safe_slot = tl.where(slot_valid, slot, 0)
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    physical_sequence = cache_batch * KV_HEADS + kv_head
    offset_base = physical_sequence * slot_offset_stride
    leaf_start = tl.load(
        fixed_slot_offsets + offset_base + safe_slot,
        mask=slot_valid,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    leaf_stop = tl.load(
        fixed_slot_offsets + offset_base + safe_slot + 1,
        mask=slot_valid,
        other=0,
        cache_modifier=".cg",
    ).to(tl.int32)
    leaf_count = tl.where(slot_valid, leaf_stop - leaf_start, 0)
    logical_start = LEAF_BEGIN + leaf_start
    coarse_token = LEAF_BEGIN - STATE_CAPACITY + safe_slot
    tl.store(
        active_mask + sequence * mask_stride + coarse_token,
        0,
        mask=slot_valid & (coarse_token < MASK_CAPACITY),
    )
    token = tl.arange(0, BLOCK_N)
    for begin in tl.range(0, leaf_count, BLOCK_N, num_stages=1):
        offset = begin + token
        logical_token = logical_start + offset
        tl.store(
            active_mask + sequence * mask_stride + logical_token,
            1,
            mask=(slot_valid & (offset < leaf_count) & (logical_token < MASK_CAPACITY)),
        )
    first_block = logical_start // TILE_SIZE
    last_block = (logical_start + leaf_count + TILE_SIZE - 1) // TILE_SIZE
    block = tl.arange(0, BLOCKS_N)
    for begin in tl.range(0, last_block - first_block, BLOCKS_N, num_stages=1):
        logical_block = first_block + begin + block
        tl.store(
            active_blocks + sequence * block_stride + logical_block,
            1,
            mask=(
                slot_valid
                & (begin + block < last_block - first_block)
                & (logical_block < (MASK_CAPACITY + TILE_SIZE - 1) // TILE_SIZE)
            ),
        )

    tl.store(
        previous_slots + sequence * UNION_CAPACITY + candidate,
        slot,
        mask=candidate < UNION_CAPACITY,
    )
    if candidate == 0:
        tl.store(previous_counts + sequence, UNION_CAPACITY)
        tl.store(previous_cache_rows + sequence, cache_batch)
        if not PRESERVE_UNION_METADATA:
            tl.store(union_counts + sequence, UNION_CAPACITY)
            tl.store(union_token_counts + sequence, 0)
            epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
            tl.store(sequence_epochs + sequence, epoch + 1)
        if sequence == 0:
            tl.store(execution_geometry, EFFECTIVE_SEGMENTS)
            tl.store(execution_geometry + 1, SPLIT_D_REDUCE)
            tl.store(execution_geometry + 2, tl.num_programs(0))
            tl.store(execution_geometry + 3, KV_HEADS)


@triton.jit
def _initialize_page1_fixed_prefix_kernel(
    fixed_indices,
    ROW_OFFSET: tl.constexpr,
    KV_HEADS: tl.constexpr,
    FIXED_CAPACITY: tl.constexpr,
    LOCAL_OFFSET: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    LOCAL_LIMIT: tl.constexpr,
    SINK_OFFSET: tl.constexpr,
    SINK_CAPACITY: tl.constexpr,
    SINK_LEN: tl.constexpr,
    COARSE_OFFSET: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    local_sequence = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    global_sequence = ROW_OFFSET * KV_HEADS + local_sequence
    token = block * BLOCK_N + tl.arange(0, BLOCK_N)
    prefix_length: tl.constexpr = LOCAL_LIMIT + SINK_LEN + STATE_CAPACITY
    valid = token < prefix_length
    local_entry = token < LOCAL_LIMIT
    sink_rank = token - LOCAL_LIMIT
    sink_entry = (sink_rank >= 0) & (sink_rank < SINK_LEN)
    coarse_slot = token - (LOCAL_LIMIT + SINK_LEN)
    physical = tl.where(
        local_entry,
        LOCAL_OFFSET + global_sequence * LOCAL_CAPACITY + token,
        tl.where(
            sink_entry,
            SINK_OFFSET + global_sequence * SINK_CAPACITY + sink_rank,
            COARSE_OFFSET + global_sequence * STATE_CAPACITY + coarse_slot,
        ),
    )
    tl.store(
        fixed_indices + local_sequence * FIXED_CAPACITY + token,
        physical,
        mask=valid,
    )


@triton.jit
def _materialize_page1_fixed_leaves_kernel(
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    slot_offsets,
    fixed_indices,
    fixed_leaf_owners,
    ROW_OFFSET: tl.constexpr,
    KV_HEADS: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    FIXED_CAPACITY: tl.constexpr,
    LEAF_BEGIN: tl.constexpr,
    ARENA_LEAF_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    local_sequence = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1).to(tl.int64)
    global_sequence = ROW_OFFSET * KV_HEADS + local_sequence
    leaf_count = tl.load(slot_lengths + local_sequence * STATE_CAPACITY + slot).to(
        tl.int32
    )
    destination = tl.load(
        slot_offsets + local_sequence * (STATE_CAPACITY + 1) + slot
    ).to(tl.int32)
    token = tl.arange(0, BLOCK_N)
    for begin in tl.range(0, leaf_count, BLOCK_N, num_stages=1):
        logical_token = begin + token
        valid = logical_token < leaf_count
        page_ordinal = logical_token // PAGE_SIZE
        within_page = logical_token % PAGE_SIZE
        page_id = _lookup_page_id(
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            local_sequence,
            slot,
            page_ordinal,
            valid,
            STATE_CAPACITY,
            INLINE_PAGES_PER_SLOT,
            PAGE_CAPACITY,
            HASH_CAPACITY,
            HASH_PROBES,
        ).to(tl.int64)
        page_valid = valid & (page_id >= 0) & (page_id < PAGE_CAPACITY)
        safe_page = tl.where(page_valid, page_id, 0)
        physical_token = (
            local_sequence * PAGE_CAPACITY + safe_page
        ) * PAGE_SIZE + within_page
        leaf_index = tl.load(
            page_indices + physical_token,
            mask=page_valid,
            other=0,
        ).to(tl.int32)
        leaf_valid = page_valid & (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
        leaf_rank = destination + logical_token
        physical_leaf = ARENA_LEAF_OFFSET + global_sequence * LEAF_CAPACITY + leaf_index
        tl.store(
            fixed_indices + local_sequence * FIXED_CAPACITY + LEAF_BEGIN + leaf_rank,
            physical_leaf,
            mask=leaf_valid,
        )
        tl.store(
            fixed_leaf_owners + local_sequence * LEAF_CAPACITY + leaf_rank,
            slot,
            mask=leaf_valid,
        )


def materialize_page1_fixed_indices(
    page_indices: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    fixed_indices: torch.Tensor,
    fixed_leaf_owners: torch.Tensor,
    fixed_slot_offsets: torch.Tensor,
    fixed_lengths: torch.Tensor,
    *,
    row_offset: int,
    arena_leaf_offset: int,
    arena_local_offset: int,
    arena_sink_offset: int,
    arena_coarse_offset: int,
    local_capacity: int,
    local_limit: int,
    sink_capacity: int,
    sink_len: int,
    hash_probes: int,
) -> None:
    """Rebuild the persistent valid-leaf list at a state-update boundary."""
    if slot_lengths.dtype != torch.int32:
        raise TypeError("fixed page-size-one lists require INT32 slot lengths")
    if fixed_indices.dtype != torch.int32 or fixed_leaf_owners.dtype != torch.int32:
        raise TypeError("fixed page-size-one index metadata must use INT32")
    if fixed_slot_offsets.dtype != torch.int32 or fixed_lengths.dtype != torch.int32:
        raise TypeError("fixed page-size-one offsets and lengths must use INT32")
    rows, kv_heads, state_capacity = slot_lengths.shape
    if tuple(fixed_slot_offsets.shape) != (
        rows,
        kv_heads,
        state_capacity + 1,
    ):
        raise ValueError("fixed page-size-one slot offsets have the wrong shape")
    if tuple(fixed_lengths.shape) != (rows, kv_heads):
        raise ValueError("fixed page-size-one lengths have the wrong shape")
    leaf_capacity = int(fixed_leaf_owners.size(2))
    fixed_capacity = int(fixed_indices.size(2))
    leaf_begin = local_limit + sink_len + state_capacity
    if fixed_capacity < leaf_begin + leaf_capacity:
        raise ValueError("fixed page-size-one table has insufficient capacity")
    if tuple(fixed_indices.shape[:2]) != (rows, kv_heads) or tuple(
        fixed_leaf_owners.shape[:2]
    ) != (rows, kv_heads):
        raise ValueError("fixed page-size-one metadata rows do not match")
    if not all(
        tensor.is_cuda
        for tensor in (
            page_indices,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            fixed_indices,
            fixed_leaf_owners,
            fixed_slot_offsets,
            fixed_lengths,
        )
    ):
        raise ValueError("fixed page-size-one materialization requires CUDA tensors")

    fixed_slot_offsets[..., 0].zero_()
    torch.cumsum(
        slot_lengths,
        dim=-1,
        dtype=torch.int32,
        out=fixed_slot_offsets[..., 1:],
    )
    fixed_lengths.copy_(fixed_slot_offsets[..., -1] + leaf_begin)
    sequences = rows * kv_heads
    prefix_block = 64
    _initialize_page1_fixed_prefix_kernel[
        (sequences, triton.cdiv(leaf_begin, prefix_block))
    ](
        fixed_indices,
        ROW_OFFSET=row_offset,
        KV_HEADS=kv_heads,
        FIXED_CAPACITY=fixed_capacity,
        LOCAL_OFFSET=arena_local_offset,
        LOCAL_CAPACITY=local_capacity,
        LOCAL_LIMIT=local_limit,
        SINK_OFFSET=arena_sink_offset,
        SINK_CAPACITY=sink_capacity,
        SINK_LEN=sink_len,
        COARSE_OFFSET=arena_coarse_offset,
        STATE_CAPACITY=state_capacity,
        BLOCK_N=prefix_block,
        num_warps=1,
    )
    leaf_block = 64
    _materialize_page1_fixed_leaves_kernel[(sequences, state_capacity)](
        page_indices,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        fixed_slot_offsets,
        fixed_indices,
        fixed_leaf_owners,
        ROW_OFFSET=row_offset,
        KV_HEADS=kv_heads,
        PAGE_CAPACITY=int(page_indices.size(2)),
        LEAF_CAPACITY=leaf_capacity,
        STATE_CAPACITY=state_capacity,
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=int(page_indices.size(3)),
        FIXED_CAPACITY=fixed_capacity,
        LEAF_BEGIN=leaf_begin,
        ARENA_LEAF_OFFSET=arena_leaf_offset,
        BLOCK_N=leaf_block,
        num_warps=1,
    )


@triton.jit
def _reduce_aiter_page1_segments_with_lse_kernel(
    segment_output,
    segment_max_logits,
    segment_exp_sums,
    context_lens,
    output,
    output_lse,
    sequence_epochs,
    union_counts,
    union_token_counts,
    cache_indices,
    local_lens,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEGMENTS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ADVANCE_QUEUE: tl.constexpr,
    ADVANCE_LOCAL: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SKIP_ZERO_LENGTH: tl.constexpr = False,
):
    """AITER's segment reduction with the branch LSE stored concurrently."""
    sequence = tl.program_id(0).to(tl.int64)
    query_lane = tl.program_id(1).to(tl.int64)
    length = tl.load(context_lens + sequence).to(tl.int32)
    if SKIP_ZERO_LENGTH and length == 0:
        return
    segment = tl.arange(0, SEGMENTS)
    tiles_per_segment = tl.cdiv(tl.maximum(length, 1), SEGMENTS * TILE_SIZE)
    active_segments = tl.cdiv(length, tiles_per_segment * TILE_SIZE)
    valid = segment < active_segments
    scalar_offset = sequence * QUERY_ROWS * SEGMENTS + query_lane * SEGMENTS + segment
    maxima = tl.load(
        segment_max_logits + scalar_offset,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(maxima, axis=0)
    exp_sums = tl.load(segment_exp_sums + scalar_offset, mask=valid, other=0.0).to(
        tl.float32
    )
    corrections = tl.where(valid, tl.math.exp2(maxima - maximum), 0.0)
    denominator = tl.sum(exp_sums * corrections, axis=0)
    dimension = tl.arange(0, HEAD_DIM)
    vector_offset = (
        sequence * QUERY_ROWS * SEGMENTS * HEAD_DIM
        + query_lane * SEGMENTS * HEAD_DIM
        + segment[:, None] * HEAD_DIM
        + dimension[None, :]
    )
    partials = tl.load(
        segment_output + vector_offset,
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(partials * corrections[:, None], axis=0)
    result = tl.where(denominator > 0.0, numerator / denominator, 0.0)
    tl.store(
        output + sequence * OUTPUT_STRIDE_0 + query_lane * OUTPUT_STRIDE_1 + dimension,
        result,
    )
    lse = tl.where(
        denominator > 0.0,
        (maximum + tl.log2(denominator)) * 0.6931471805599453,
        -float("inf"),
    )
    tl.store(output_lse + sequence * QUERY_ROWS + query_lane, lse)
    if ADVANCE_QUEUE and query_lane == 0:
        epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
        tl.store(sequence_epochs + sequence, epoch + 1)
        tl.store(union_counts + sequence, 0)
        tl.store(union_token_counts + sequence, 0)
    if ADVANCE_LOCAL and query_lane == 0 and sequence % KV_HEADS == 0:
        batch = sequence // KV_HEADS
        cache_batch = tl.load(cache_indices + batch).to(tl.int64)
        local_length = tl.load(local_lens + cache_batch)
        tl.store(local_lens + cache_batch, local_length + 1)


@triton.jit
def _reduce_aiter_page1_segments_with_lse_split_d_kernel(
    segment_output,
    segment_max_logits,
    segment_exp_sums,
    context_lens,
    output,
    output_lse,
    sequence_epochs,
    union_counts,
    union_token_counts,
    OUTPUT_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_1: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEGMENTS: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ADVANCE_QUEUE: tl.constexpr,
    SKIP_ZERO_LENGTH: tl.constexpr = False,
):
    """Reduce page1 splits with independent output-dimension programs."""
    sequence = tl.program_id(0).to(tl.int64)
    query_lane = tl.program_id(1).to(tl.int64)
    dimension_block = tl.program_id(2).to(tl.int64)
    length = tl.load(context_lens + sequence).to(tl.int32)
    if SKIP_ZERO_LENGTH and length == 0:
        return
    segment = tl.arange(0, SEGMENTS)
    tiles_per_segment = tl.cdiv(tl.maximum(length, 1), SEGMENTS * TILE_SIZE)
    active_segments = tl.cdiv(length, tiles_per_segment * TILE_SIZE)
    valid = segment < active_segments
    scalar_offset = sequence * QUERY_ROWS * SEGMENTS + query_lane * SEGMENTS + segment
    maxima = tl.load(
        segment_max_logits + scalar_offset,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(maxima, axis=0)
    exp_sums = tl.load(segment_exp_sums + scalar_offset, mask=valid, other=0.0).to(
        tl.float32
    )
    corrections = tl.where(valid, tl.math.exp2(maxima - maximum), 0.0)
    denominator = tl.sum(exp_sums * corrections, axis=0)

    dimension = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_dimension = dimension < HEAD_DIM
    vector_offset = (
        sequence * QUERY_ROWS * SEGMENTS * HEAD_DIM
        + query_lane * SEGMENTS * HEAD_DIM
        + segment[:, None] * HEAD_DIM
        + dimension[None, :]
    )
    partials = tl.load(
        segment_output + vector_offset,
        mask=valid[:, None] & valid_dimension[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(partials * corrections[:, None], axis=0)
    result = tl.where(denominator > 0.0, numerator / denominator, 0.0)
    tl.store(
        output + sequence * OUTPUT_STRIDE_0 + query_lane * OUTPUT_STRIDE_1 + dimension,
        result,
        mask=valid_dimension,
    )
    if dimension_block == 0:
        lse = tl.where(
            denominator > 0.0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        )
        tl.store(output_lse + sequence * QUERY_ROWS + query_lane, lse)
        if ADVANCE_QUEUE and query_lane == 0:
            epoch = tl.load(sequence_epochs + sequence).to(tl.int32)
            tl.store(sequence_epochs + sequence, epoch + 1)
            tl.store(union_counts + sequence, 0)
            tl.store(union_token_counts + sequence, 0)


@triton.jit
def _reduce_routed_split_decode_lod_attention_kernel(
    q,
    sink_k,
    sink_v,
    state_k,
    state_v,
    counts,
    cache_indices,
    local_lens,
    local_k,
    local_v,
    new_k,
    new_v,
    top_slots,
    top_scores,
    coarse_out,
    coarse_lse,
    partial_out,
    partial_lse,
    separate_local_out,
    separate_local_lse,
    out,
    SINK_K_BATCH_STRIDE,
    SINK_K_HEAD_STRIDE,
    SINK_K_TOKEN_STRIDE,
    SINK_V_BATCH_STRIDE,
    SINK_V_HEAD_STRIDE,
    SINK_V_TOKEN_STRIDE,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    STATE_V_BATCH_STRIDE,
    STATE_V_HEAD_STRIDE,
    STATE_V_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE,
    LOCAL_V_BATCH_STRIDE,
    LOCAL_V_HEAD_STRIDE,
    LOCAL_V_TOKEN_STRIDE,
    NEW_K_BATCH_STRIDE,
    NEW_K_HEAD_STRIDE,
    NEW_V_BATCH_STRIDE,
    NEW_V_HEAD_STRIDE,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SPLITS: tl.constexpr,
    ROUTE_SPLITS: tl.constexpr,
    INCLUDE_SEPARATE_LOCAL: tl.constexpr,
    SEPARATE_LOCAL_SPLITS: tl.constexpr,
    FUSE_LOCAL_SCAN: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    INCLUDE_SINK: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LOCAL_BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    USE_DOT: tl.constexpr,
    ADVANCE_LOCAL: tl.constexpr,
    SUBTRACT_ROUTES: tl.constexpr,
):
    """Remove routed summaries, then stream exact branches into one softmax."""
    query_row = tl.program_id(0).to(tl.int64)
    batch = query_row // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    dim = tl.arange(0, HEAD_DIM)
    full_coarse_lse = tl.load(coarse_lse + query_row)
    remainder_out = tl.load(coarse_out + query_row * HEAD_DIM + dim)
    selected_mass = tl.zeros((), tl.float32)
    selected_value = tl.zeros((HEAD_DIM,), tl.float32)
    if SUBTRACT_ROUTES:
        for route in tl.static_range(0, ROUTE_COUNT):
            slot = tl.load(top_slots + query_row * ROUTE_COUNT + route).to(tl.int64)
            score = tl.load(top_scores + query_row * ROUTE_COUNT + route)
            valid_slot = (
                (slot >= 0)
                & (slot < STATE_CAPACITY)
                & (score > -float("inf"))
            )
            slot = tl.where(valid_slot, slot, 0)
            count = tl.load(
                counts
                + cache_batch * COUNT_BATCH_STRIDE
                + kv_head * COUNT_HEAD_STRIDE
                + slot * COUNT_TOKEN_STRIDE,
                mask=valid_slot,
                other=1.0,
            ).to(tl.float32)
            valid_slot &= count > 0.0
            safe_count = tl.where(valid_slot, count, 1.0)
            value = tl.load(
                state_v
                + cache_batch * STATE_V_BATCH_STRIDE
                + kv_head * STATE_V_HEAD_STRIDE
                + slot * STATE_V_TOKEN_STRIDE
                + dim,
                mask=valid_slot,
                other=0.0,
            )
            mean_value = value.to(tl.float32) / safe_count
            mass = tl.where(valid_slot, tl.exp(score - full_coarse_lse), 0.0)
            selected_mass += mass
            selected_value += mass * mean_value
        remainder_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
        remainder_out = (remainder_out - selected_value) / remainder_mass
        remainder_lse = full_coarse_lse + tl.log(remainder_mass)
    else:
        remainder_lse = full_coarse_lse

    # Fold branches sequentially. This keeps only one HEAD_DIM-wide value live
    # instead of materializing SPLITS x HEAD_DIM in the final-reduction program.
    maximum = remainder_lse
    denominator = tl.full((), 1.0, tl.float32)
    numerator = remainder_out.to(tl.float32)
    if INCLUDE_SEPARATE_LOCAL:
        for local_split in tl.static_range(0, SEPARATE_LOCAL_SPLITS):
            local_row = query_row * SEPARATE_LOCAL_SPLITS + local_split
            local_lse = tl.load(separate_local_lse + local_row)
            local_value = tl.load(separate_local_out + local_row * HEAD_DIM + dim).to(
                tl.float32
            )
            new_maximum = tl.maximum(maximum, local_lse)
            old_weight = tl.exp(maximum - new_maximum)
            new_weight = tl.exp(local_lse - new_maximum)
            denominator = denominator * old_weight + new_weight
            numerator = numerator * old_weight + new_weight * local_value
            maximum = new_maximum
    if ROUTE_SPLITS == 1:
        for split_index in tl.static_range(0, SPLITS):
            branch_lse = tl.load(partial_lse + query_row * SPLITS + split_index)
            branch_value = tl.load(
                partial_out + (query_row * SPLITS + split_index) * HEAD_DIM + dim
            ).to(tl.float32)
            new_maximum = tl.maximum(maximum, branch_lse)
            old_weight = tl.exp(maximum - new_maximum)
            new_weight = tl.exp(branch_lse - new_maximum)
            denominator = denominator * old_weight + new_weight
            numerator = numerator * old_weight + new_weight * branch_value
            maximum = new_maximum
    else:
        # Cooperative decode produces one exact partial for every
        # (route, page-list split). Fold those partials directly into the final
        # LOD result instead of first materializing eight route-level outputs.
        for branch_index in tl.range(0, SPLITS * ROUTE_SPLITS, num_stages=1):
            branch_lse = tl.load(
                partial_lse + query_row * SPLITS * ROUTE_SPLITS + branch_index
            )
            branch_value = tl.load(
                partial_out
                + (query_row * SPLITS * ROUTE_SPLITS + branch_index) * HEAD_DIM
                + dim
            ).to(tl.float32)
            new_maximum = tl.maximum(maximum, branch_lse)
            old_weight = tl.exp(maximum - new_maximum)
            new_weight = tl.exp(branch_lse - new_maximum)
            denominator = denominator * old_weight + new_weight
            numerator = numerator * old_weight + new_weight * branch_value
            maximum = new_maximum

    query = tl.load(q + query_row * HEAD_DIM + dim).to(tl.float32)
    active_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
    if FUSE_LOCAL_SCAN:
        token_offset = tl.arange(0, LOCAL_BLOCK_N)
        for local_begin in tl.range(0, active_local_len, LOCAL_BLOCK_N, num_stages=1):
            token = local_begin + token_offset
            valid = token < active_local_len
            keys = tl.load(
                local_k
                + cache_batch * LOCAL_K_BATCH_STRIDE
                + kv_head * LOCAL_K_HEAD_STRIDE
                + token[:, None] * LOCAL_K_TOKEN_STRIDE
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            values = tl.load(
                local_v
                + cache_batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + token[:, None] * LOCAL_V_TOKEN_STRIDE
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if USE_DOT:
                scores = tl.dot(
                    query[None, :].to(keys.dtype),
                    tl.trans(keys),
                    out_dtype=tl.float32,
                )
                scores = tl.reshape(scores, (LOCAL_BLOCK_N,))
            else:
                scores = tl.sum(query[None, :] * keys.to(tl.float32), axis=1)
            scores = tl.where(valid, scores * SCALE, -float("inf"))
            block_maximum = tl.max(scores, axis=0)
            new_maximum = tl.maximum(maximum, block_maximum)
            old_weight = tl.exp(maximum - new_maximum)
            weights = tl.exp(scores - new_maximum)
            denominator = denominator * old_weight + tl.sum(weights, axis=0)
            numerator = numerator * old_weight + tl.sum(
                weights[:, None] * values.to(tl.float32), axis=0
            )
            maximum = new_maximum

        if INCLUDE_NEW:
            current_key = tl.load(
                new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
            )
            current_value = tl.load(
                new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim
            )
            current_score = tl.sum(query * current_key.to(tl.float32), axis=0) * SCALE
            new_maximum = tl.maximum(maximum, current_score)
            old_weight = tl.exp(maximum - new_maximum)
            new_weight = tl.exp(current_score - new_maximum)
            denominator = denominator * old_weight + new_weight
            numerator = numerator * old_weight + new_weight * current_value.to(
                tl.float32
            )
            maximum = new_maximum
            if query_head % KV_GROUP_SIZE == 0:
                tl.store(
                    local_k
                    + cache_batch * LOCAL_K_BATCH_STRIDE
                    + kv_head * LOCAL_K_HEAD_STRIDE
                    + active_local_len * LOCAL_K_TOKEN_STRIDE
                    + dim,
                    current_key,
                )
                tl.store(
                    local_v
                    + cache_batch * LOCAL_V_BATCH_STRIDE
                    + kv_head * LOCAL_V_HEAD_STRIDE
                    + active_local_len * LOCAL_V_TOKEN_STRIDE
                    + dim,
                    current_value,
                )

    if INCLUDE_SINK:
        for sink_index in tl.static_range(0, SINK_LEN):
            key = tl.load(
                sink_k
                + cache_batch * SINK_K_BATCH_STRIDE
                + kv_head * SINK_K_HEAD_STRIDE
                + sink_index * SINK_K_TOKEN_STRIDE
                + dim
            ).to(tl.float32)
            value = tl.load(
                sink_v
                + cache_batch * SINK_V_BATCH_STRIDE
                + kv_head * SINK_V_HEAD_STRIDE
                + sink_index * SINK_V_TOKEN_STRIDE
                + dim
            ).to(tl.float32)
            score = tl.sum(query * key, axis=0) * SCALE
            new_maximum = tl.maximum(maximum, score)
            old_weight = tl.exp(maximum - new_maximum)
            new_weight = tl.exp(score - new_maximum)
            denominator = denominator * old_weight + new_weight
            numerator = numerator * old_weight + value * new_weight
            maximum = new_maximum
    result = numerator / denominator
    tl.store(out + query_row * HEAD_DIM + dim, result)
    if ADVANCE_LOCAL and query_head == 0:
        local_length = tl.load(local_lens + cache_batch)
        tl.store(local_lens + cache_batch, local_length + 1)
