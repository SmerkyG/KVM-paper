"""Uniform all-centroid top-1 LOD attention kernels.

The first kernel is expert-major: for every region it finds the best member
leaf for every query.  It writes a tiled dense table of leaf indices and the
scores already produced by that search, but performs no centroid work.
The second kernel is query-major and replaces every region by two uniform
terms: its exact winning leaf and the centroid summary of its remaining
members.  The low- and high-detail terms therefore share one softmax pass.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .paged_leaf_attention import _lookup_page_id


@triton.jit
def _segmented_argmax_combine(
    left_owner,
    left_score,
    left_leaf,
    right_owner,
    right_score,
    right_leaf,
):
    same_segment = left_owner == right_owner
    take_left = same_segment & (left_score >= right_score)
    return (
        right_owner,
        tl.where(take_left, left_score, right_score),
        tl.where(take_left, left_leaf, right_leaf),
    )


@triton.jit
def _pack_centroid_leaf_stream_kernel(
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    slot_offsets,
    packed_leaf_indices,
    packed_leaf_slots,
    STATE_CAPACITY: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEXED: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    batch_kv = expert // STATE_CAPACITY
    slot = expert - batch_kv * STATE_CAPACITY
    key_count = tl.load(slot_lengths + expert).to(tl.int32)
    output_begin = tl.load(
        slot_offsets + batch_kv * (STATE_CAPACITY + 1) + slot
    ).to(tl.int32)
    lane = tl.arange(0, BLOCK_N)
    for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
        logical_key = key_begin + lane
        valid_leaf = logical_key < key_count
        page_ordinal = logical_key // PAGE_SIZE
        within_page = logical_key % PAGE_SIZE
        if HASH_PROBES == 0:
            page_id = tl.load(
                slot_pages
                + expert * INLINE_PAGES_PER_SLOT
                + page_ordinal,
                mask=valid_leaf,
                other=-1,
            ).to(tl.int64)
        else:
            page_id = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                batch_kv,
                slot,
                page_ordinal,
                valid_leaf,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        valid_leaf &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
        physical_token = page_id * PAGE_SIZE + within_page
        if INDEXED:
            leaf_index = tl.load(
                page_indices
                + batch_kv * PAGE_CAPACITY * PAGE_SIZE
                + physical_token,
                mask=valid_leaf,
                other=-1,
            ).to(tl.int32)
        else:
            leaf_index = physical_token.to(tl.int32)
        output_position = output_begin + logical_key
        valid_leaf &= (
            (leaf_index >= 0)
            & (leaf_index < LEAF_CAPACITY)
            & (output_position < LEAF_CAPACITY)
        )
        tl.store(
            packed_leaf_indices
            + batch_kv * LEAF_CAPACITY
            + output_position,
            leaf_index,
            mask=valid_leaf,
        )
        tl.store(
            packed_leaf_slots + batch_kv * LEAF_CAPACITY + output_position,
            slot,
            mask=valid_leaf,
        )


def build_all_centroid_leaf_stream(
    page_indices: torch.Tensor | None,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    slot_offsets: torch.Tensor,
    *,
    leaf_capacity: int,
    hash_probes: int,
    packed_leaf_indices: torch.Tensor | None = None,
    packed_leaf_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize only the index metadata for a zero-gap centroid stream."""
    batch, kv_heads, state_capacity = slot_lengths.shape
    expected_shape = (batch, kv_heads, leaf_capacity)
    if packed_leaf_indices is None or tuple(packed_leaf_indices.shape) != expected_shape:
        packed_leaf_indices = torch.empty(
            expected_shape, dtype=torch.int32, device=slot_lengths.device
        )
    if packed_leaf_slots is None or tuple(packed_leaf_slots.shape) != expected_shape:
        packed_leaf_slots = torch.empty_like(packed_leaf_indices)
    indexed = page_indices is not None
    page_size = int(page_indices.size(3) if indexed else 16)
    page_capacity = int(
        page_indices.size(2) if indexed else leaf_capacity // page_size
    )
    _pack_centroid_leaf_stream_kernel[(batch * kv_heads * state_capacity,)](
        page_indices if indexed else slot_pages,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        slot_offsets,
        packed_leaf_indices,
        packed_leaf_slots,
        STATE_CAPACITY=state_capacity,
        PAGE_CAPACITY=page_capacity,
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        LEAF_CAPACITY=leaf_capacity,
        PAGE_SIZE=page_size,
        BLOCK_N=128,
        INDEXED=indexed,
        num_warps=4,
    )
    return packed_leaf_indices, packed_leaf_slots


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=["query_len", "state_len"],
)
def _all_centroid_group_stream_fused_kernel(
    q,
    state_k,
    state_v,
    state_counts,
    slot_lengths,
    slot_offsets,
    packed_leaf_indices,
    packed_leaf_slots,
    leaf_k,
    leaf_v,
    scratch_leaf,
    scratch_score,
    local_output,
    local_lse,
    output,
    output_lse,
    query_len,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CENTROIDS_PER_PROGRAM: tl.constexpr,
    LEAF_BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    INCLUDE_LOCAL: tl.constexpr,
    DISJOINT_RESIDUAL: tl.constexpr,
):
    """Packed exact-leaf search and coarse/fine combine in one program."""
    query_program = tl.program_id(0).to(tl.int64)
    query_blocks_per_kv = tl.cdiv(KV_GROUP_SIZE * query_len, BLOCK_M)
    batch_kv = query_program // query_blocks_per_kv
    query_block_in_kv = query_program - batch_kv * query_blocks_per_kv
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    query_in_kv = query_block_in_kv * BLOCK_M + tl.arange(0, BLOCK_M)
    query_group = query_in_kv // query_len
    query_position = query_in_kv - query_group * query_len
    valid_query = query_group < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_group
    query_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query_position
    ).to(tl.int64)

    value_offset = tl.arange(0, VALUE_DIM)
    centroid = tl.arange(0, CENTROIDS_PER_PROGRAM)
    leaf_lane = tl.arange(0, LEAF_BLOCK_N)
    offset_row = batch_kv * (STATE_CAPACITY + 1)
    scratch_row = (
        query_program * BLOCK_M + tl.arange(0, BLOCK_M)
    ) * CENTROIDS_PER_PROGRAM
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)

    for slot_begin in tl.range(
        0, state_len, CENTROIDS_PER_PROGRAM, num_stages=1
    ):
        slot = slot_begin + centroid
        valid_slot = slot < state_len
        count = tl.load(
            state_counts + batch_kv * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        key_count = tl.load(
            slot_lengths + batch_kv * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0,
        ).to(tl.int32)
        group_leaf_begin = tl.load(
            slot_offsets + offset_row + slot_begin
        ).to(tl.int32)
        group_end_slot = tl.minimum(
            slot_begin + CENTROIDS_PER_PROGRAM, state_len
        )
        group_leaf_end = tl.load(
            slot_offsets + offset_row + group_end_slot
        ).to(tl.int32)
        carry_owner = tl.full((), -2, tl.int32)
        carry_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        carry_leaf = tl.full((BLOCK_M,), -1, tl.int32)

        for leaf_begin in tl.range(
            0,
            group_leaf_end - group_leaf_begin,
            LEAF_BLOCK_N,
            num_stages=1,
        ):
            packed_position = group_leaf_begin + leaf_begin + leaf_lane
            valid_leaf = packed_position < group_leaf_end
            owner = tl.load(
                packed_leaf_slots
                + batch_kv * LEAF_CAPACITY
                + packed_position,
                mask=valid_leaf,
                other=-1,
            ).to(tl.int32)
            leaf_index = tl.load(
                packed_leaf_indices
                + batch_kv * LEAF_CAPACITY
                + packed_position,
                mask=valid_leaf,
                other=-1,
            ).to(tl.int32)
            valid_leaf &= (
                (owner >= slot_begin)
                & (owner < group_end_slot)
                & (leaf_index >= 0)
                & (leaf_index < LEAF_CAPACITY)
            )
            safe_leaf = tl.where(valid_leaf, leaf_index, 0)
            scores = tl.zeros((BLOCK_M, LEAF_BLOCK_N), tl.float32)
            for dim_begin in tl.static_range(0, HEAD_DIM, BLOCK_D):
                dim = dim_begin + tl.arange(0, BLOCK_D)
                valid_dim = dim < HEAD_DIM
                query = tl.load(
                    q + query_row[:, None] * HEAD_DIM + dim[None, :],
                    mask=valid_query[:, None] & valid_dim[None, :],
                    other=0.0,
                )
                keys = tl.load(
                    leaf_k
                    + (batch_kv * LEAF_CAPACITY + safe_leaf[None, :])
                    * HEAD_DIM
                    + dim[:, None],
                    mask=valid_dim[:, None] & valid_leaf[None, :],
                    other=0.0,
                )
                scores += tl.dot(query, keys)
            scores = tl.where(
                valid_query[:, None] & valid_leaf[None, :],
                scores,
                -float("inf"),
            )
            scan_owner, scan_score, scan_leaf = tl.associative_scan(
                (
                    owner[None, :]
                    + tl.zeros((BLOCK_M, LEAF_BLOCK_N), tl.int32),
                    scores,
                    safe_leaf[None, :]
                    + tl.zeros((BLOCK_M, LEAF_BLOCK_N), tl.int32),
                ),
                axis=1,
                combine_fn=_segmented_argmax_combine,
            )
            take_carry = (scan_owner == carry_owner) & (
                carry_score[:, None] >= scan_score
            )
            scan_score = tl.where(take_carry, carry_score[:, None], scan_score)
            scan_leaf = tl.where(take_carry, carry_leaf[:, None], scan_leaf)
            next_owner = tl.load(
                packed_leaf_slots
                + batch_kv * LEAF_CAPACITY
                + packed_position
                + 1,
                mask=packed_position + 1 < group_leaf_end,
                other=-1,
            ).to(tl.int32)
            segment_end = valid_leaf & (owner != next_owner)
            owner_offset = owner - slot_begin
            scratch_pointer = scratch_row[:, None] + owner_offset[None, :]
            tl.store(
                scratch_leaf + scratch_pointer,
                scan_leaf,
                mask=valid_query[:, None] & segment_end[None, :],
            )
            tl.store(
                scratch_score + scratch_pointer,
                scan_score,
                mask=valid_query[:, None] & segment_end[None, :],
            )
            block_last = valid_leaf & (
                leaf_lane
                == tl.minimum(
                    LEAF_BLOCK_N - 1,
                    group_leaf_end - group_leaf_begin - leaf_begin - 1,
                )
            )
            carry_owner = tl.max(tl.where(block_last, owner, -1), axis=0)
            carry_score = tl.max(
                tl.where(block_last[None, :], scan_score, -float("inf")),
                axis=1,
            )
            carry_leaf = tl.sum(
                tl.where(block_last[None, :], scan_leaf, 0), axis=1
            ).to(tl.int32)

        tl.debug_barrier()
        scratch_centroid_pointer = scratch_row[:, None] + centroid[None, :]
        best_leaf = tl.load(
            scratch_leaf + scratch_centroid_pointer,
            mask=valid_query[:, None]
            & valid_slot[None, :]
            & (key_count[None, :] > 0),
            other=-1,
        ).to(tl.int64)
        best_score = tl.load(
            scratch_score + scratch_centroid_pointer,
            mask=valid_query[:, None]
            & valid_slot[None, :]
            & (key_count[None, :] > 0),
            other=-float("inf"),
        ).to(tl.float32)

        state_dot = tl.zeros(
            (BLOCK_M, CENTROIDS_PER_PROGRAM), tl.float32
        )
        for dim_begin in tl.static_range(0, HEAD_DIM, BLOCK_D):
            dim = dim_begin + tl.arange(0, BLOCK_D)
            valid_dim = dim < HEAD_DIM
            query = tl.load(
                q + query_row[:, None] * HEAD_DIM + dim[None, :],
                mask=valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            state_key_sum = tl.load(
                state_k
                + (batch_kv * STATE_CAPACITY + slot[:, None]) * HEAD_DIM
                + dim[None, :],
                mask=valid_slot[:, None] & valid_dim[None, :],
                other=0.0,
            )
            state_dot += tl.dot(
                query, tl.trans(state_key_sum), out_dtype=tl.float32
            )

        valid_winner = (
            valid_query[:, None]
            & valid_slot[None, :]
            & (best_leaf >= 0)
            & (best_leaf < LEAF_CAPACITY)
            & (count[None, :] >= 1.0)
        )
        safe_count = tl.where(valid_winner, count[None, :], 1.0)
        winner_score = tl.where(
            valid_winner, SCALE_LOG2 * best_score, -float("inf")
        )
        residual_count = count[None, :] - 1.0
        valid_residual = valid_winner & (residual_count > 0.0)
        safe_residual_count = tl.where(valid_residual, residual_count, 1.0)
        if DISJOINT_RESIDUAL:
            residual_dot = (
                state_dot - best_score
            ) / safe_residual_count
        else:
            residual_dot = state_dot / safe_count
        residual_score = (
            SCALE_LOG2 * residual_dot + tl.log2(safe_residual_count)
        )
        residual_score = tl.where(
            valid_residual, residual_score, -float("inf")
        )
        block_maximum = tl.maximum(
            tl.max(winner_score, axis=1),
            tl.max(residual_score, axis=1),
        )
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        winner_probability = tl.where(
            valid_winner,
            tl.math.exp2(winner_score - new_maximum[:, None]),
            0.0,
        )
        residual_probability = tl.where(
            valid_residual,
            tl.math.exp2(residual_score - new_maximum[:, None]),
            0.0,
        )
        denominator = denominator * correction + tl.sum(
            winner_probability + residual_probability, axis=1
        )
        if DISJOINT_RESIDUAL:
            state_weight = residual_probability / safe_residual_count
            winner_weight = winner_probability - state_weight
        else:
            state_weight = residual_probability / safe_count
            winner_weight = winner_probability
        state_value_sum = tl.load(
            state_v
            + (batch_kv * STATE_CAPACITY + slot[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=valid_slot[:, None],
            other=0.0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            state_weight.to(tl.bfloat16),
            state_value_sum.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        tl.store(
            scratch_score + scratch_centroid_pointer,
            winner_weight,
            mask=valid_query[:, None] & valid_slot[None, :],
        )
        tl.debug_barrier()

        for leaf_begin in tl.range(
            0,
            group_leaf_end - group_leaf_begin,
            LEAF_BLOCK_N,
            num_stages=1,
        ):
            packed_position = group_leaf_begin + leaf_begin + leaf_lane
            valid_leaf = packed_position < group_leaf_end
            owner = tl.load(
                packed_leaf_slots
                + batch_kv * LEAF_CAPACITY
                + packed_position,
                mask=valid_leaf,
                other=-1,
            ).to(tl.int32)
            leaf_index = tl.load(
                packed_leaf_indices
                + batch_kv * LEAF_CAPACITY
                + packed_position,
                mask=valid_leaf,
                other=-1,
            ).to(tl.int32)
            valid_leaf &= (
                (owner >= slot_begin)
                & (owner < group_end_slot)
                & (leaf_index >= 0)
                & (leaf_index < LEAF_CAPACITY)
            )
            safe_owner_offset = tl.where(valid_leaf, owner - slot_begin, 0)
            leaf_scratch_pointer = (
                scratch_row[:, None] + safe_owner_offset[None, :]
            )
            selected_leaf = tl.load(
                scratch_leaf + leaf_scratch_pointer,
                mask=valid_query[:, None] & valid_leaf[None, :],
                other=-1,
            )
            exact_weight = tl.load(
                scratch_score + leaf_scratch_pointer,
                mask=valid_query[:, None] & valid_leaf[None, :],
                other=0.0,
            )
            exact_weight = tl.where(
                selected_leaf == leaf_index[None, :], exact_weight, 0.0
            )
            safe_leaf = tl.where(valid_leaf, leaf_index, 0).to(tl.int64)
            leaf_values = tl.load(
                leaf_v
                + (batch_kv * LEAF_CAPACITY + safe_leaf[:, None])
                * VALUE_DIM
                + value_offset[None, :],
                mask=valid_leaf[:, None],
                other=0.0,
            )
            accumulator += tl.dot(
                exact_weight.to(tl.bfloat16),
                leaf_values.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
        maximum = new_maximum

    if INCLUDE_LOCAL:
        branch_lse = tl.load(
            local_lse + query_row,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
        branch_lse_log2 = branch_lse * 1.4426950408889634
        branch_output = tl.load(
            local_output
            + query_row[:, None] * VALUE_DIM
            + value_offset[None, :],
            mask=valid_query[:, None],
            other=0.0,
        ).to(tl.float32)
        new_maximum = tl.maximum(maximum, branch_lse_log2)
        correction = tl.math.exp2(maximum - new_maximum)
        probability = tl.math.exp2(branch_lse_log2 - new_maximum)
        denominator = denominator * correction + probability
        accumulator = (
            accumulator * correction[:, None]
            + probability[:, None] * branch_output
        )
        maximum = new_maximum

    tl.store(
        output + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(
            denominator[:, None] > 0.0,
            accumulator / denominator[:, None],
            0.0,
        ),
        mask=valid_query[:, None],
    )
    tl.store(
        output_lse + query_row,
        tl.where(
            denominator > 0.0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=valid_query,
    )


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=["query_len", "state_len"],
)
def _all_centroid_top1_leaf_kernel(
    q,
    leaf_k,
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    winner_indices,
    winner_scores,
    query_len,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CENTROIDS_PER_PROGRAM: tl.constexpr,
    INDEXED: tl.constexpr,
):
    """Write one physical leaf index and score per query/centroid pair."""
    expert_group = tl.program_id(0).to(tl.int64)
    query_block = tl.program_id(1).to(tl.int64)
    groups_per_kv = tl.cdiv(STATE_CAPACITY, CENTROIDS_PER_PROGRAM)
    batch_kv = expert_group // groups_per_kv
    slot_begin = (
        expert_group - batch_kv * groups_per_kv
    ) * CENTROIDS_PER_PROGRAM
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS

    query_in_kv = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    query_group = query_in_kv // query_len
    query_position = query_in_kv - query_group * query_len
    valid_query = query_group < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_group
    query_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query_position
    ).to(tl.int64)
    centroid = tl.arange(0, CENTROIDS_PER_PROGRAM)
    slot = slot_begin + centroid
    valid_slot = slot < state_len
    key_count = tl.load(
        slot_lengths + batch_kv * STATE_CAPACITY + slot,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    maximum_key_count = tl.max(key_count, axis=0)
    packed_key = tl.arange(0, CENTROIDS_PER_PROGRAM * BLOCK_N)
    packed_centroid = packed_key // BLOCK_N
    key_offset = packed_key - packed_centroid * BLOCK_N
    within_key = tl.arange(0, BLOCK_N)
    key_slot = slot_begin + packed_centroid
    packed_key_count = tl.load(
        slot_lengths + batch_kv * STATE_CAPACITY + key_slot,
        mask=key_slot < state_len,
        other=0,
    ).to(tl.int32)
    best_score = tl.full(
        (BLOCK_M, CENTROIDS_PER_PROGRAM), -float("inf"), tl.float32
    )
    best_leaf = tl.full(
        (BLOCK_M, CENTROIDS_PER_PROGRAM), -1, tl.int32
    )

    for key_begin in tl.range(0, maximum_key_count, BLOCK_N, num_stages=1):
        logical_key = key_begin + key_offset
        valid_key = logical_key < packed_key_count
        page_ordinal = logical_key // PAGE_SIZE
        within_page = logical_key % PAGE_SIZE
        if HASH_PROBES == 0:
            page_id = tl.load(
                slot_pages
                + (batch_kv * STATE_CAPACITY + key_slot)
                * INLINE_PAGES_PER_SLOT
                + page_ordinal,
                mask=valid_key,
                other=-1,
            ).to(tl.int64)
        else:
            page_id = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                batch_kv,
                key_slot,
                page_ordinal,
                valid_key,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        valid_key &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
        physical_token = page_id * PAGE_SIZE + within_page
        if INDEXED:
            leaf_index = tl.load(
                page_indices
                + batch_kv * PAGE_CAPACITY * PAGE_SIZE
                + physical_token,
                mask=valid_key,
                other=-1,
            ).to(tl.int64)
        else:
            leaf_index = physical_token
        valid_key &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
        safe_leaf = tl.where(valid_key, leaf_index, 0)
        scores = tl.zeros(
            (BLOCK_M, CENTROIDS_PER_PROGRAM * BLOCK_N), tl.float32
        )
        for dim_begin in tl.static_range(0, HEAD_DIM, BLOCK_D):
            dim = dim_begin + tl.arange(0, BLOCK_D)
            valid_dim = dim < HEAD_DIM
            query = tl.load(
                q + query_row[:, None] * HEAD_DIM + dim[None, :],
                mask=valid_query[:, None] & valid_dim[None, :],
                other=0.0,
            )
            keys = tl.load(
                leaf_k
                + (batch_kv * LEAF_CAPACITY + safe_leaf[None, :]) * HEAD_DIM
                + dim[:, None],
                mask=valid_dim[:, None] & valid_key[None, :],
                other=0.0,
            )
            scores += tl.dot(query, keys)
        scores = tl.where(
            valid_query[:, None] & valid_key[None, :],
            scores,
            -float("inf"),
        )
        scores = tl.reshape(
            scores, (BLOCK_M, CENTROIDS_PER_PROGRAM, BLOCK_N)
        )
        block_score = tl.max(scores, axis=2)
        block_offset = tl.argmax(scores, axis=2)
        leaf_by_centroid = tl.reshape(
            safe_leaf, (CENTROIDS_PER_PROGRAM, BLOCK_N)
        )
        block_leaf = tl.sum(
            tl.where(
                within_key[None, None, :] == block_offset[:, :, None],
                leaf_by_centroid[None, :, :],
                0,
            ),
            axis=2,
        ).to(tl.int32)
        better = block_score > best_score
        best_score = tl.where(better, block_score, best_score)
        best_leaf = tl.where(better, block_leaf, best_leaf)

    expert = batch_kv * STATE_CAPACITY + slot
    output_offset = (
        (expert[:, None] * QUERY_BLOCKS + query_block) * BLOCK_M
        + tl.arange(0, BLOCK_M)[None, :]
    )
    valid_output = (
        valid_slot[:, None]
        & (key_count[:, None] > 0)
        & valid_query[None, :]
    )
    tl.store(
        winner_indices + output_offset,
        tl.where(valid_output, tl.trans(best_leaf), -1),
    )
    tl.store(
        winner_scores + output_offset,
        tl.where(valid_output, tl.trans(best_score), -float("inf")),
    )


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=["query_len", "state_len"],
)
def _all_centroid_coarse_fine_kernel(
    q,
    state_k,
    state_v,
    state_counts,
    leaf_v,
    winner_indices,
    winner_scores,
    local_output,
    local_lse,
    output,
    output_lse,
    query_len,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    WINNER_BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    INCLUDE_LOCAL: tl.constexpr,
    DISJOINT_RESIDUAL: tl.constexpr,
):
    """Uniformly combine every centroid's exact winner and residual."""
    query_program = tl.program_id(0).to(tl.int64)
    query_blocks_per_kv = tl.cdiv(KV_GROUP_SIZE * query_len, BLOCK_M)
    batch_kv = query_program // query_blocks_per_kv
    query_block_in_kv = query_program - batch_kv * query_blocks_per_kv
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    query_in_kv = query_block_in_kv * BLOCK_M + tl.arange(0, BLOCK_M)
    query_group = query_in_kv // query_len
    query_position = query_in_kv - query_group * query_len
    valid_query = query_group < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_group
    query_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query_position
    ).to(tl.int64)
    winner_query_block = query_in_kv // WINNER_BLOCK_M
    winner_query_lane = query_in_kv - winner_query_block * WINNER_BLOCK_M

    head_offset = tl.arange(0, HEAD_DIM)
    value_offset = tl.arange(0, VALUE_DIM)
    query = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=valid_query[:, None],
        other=0.0,
    )
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    slot_offset = tl.arange(0, BLOCK_N)

    for slot_begin in tl.range(0, state_len, BLOCK_N, num_stages=1):
        slot = slot_begin + slot_offset
        valid_slot = slot < state_len
        count = tl.load(
            state_counts
            + (batch_kv * STATE_CAPACITY + slot) * 1,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        winner_offset = (
            (
                (batch_kv * STATE_CAPACITY + slot[:, None]) * QUERY_BLOCKS
                + winner_query_block[None, :]
            )
            * WINNER_BLOCK_M
            + winner_query_lane[None, :]
        )
        winner = tl.load(
            winner_indices + winner_offset,
            mask=valid_slot[:, None] & valid_query[None, :],
            other=-1,
        ).to(tl.int64)
        winner_dot = tl.load(
            winner_scores + winner_offset,
            mask=valid_slot[:, None] & valid_query[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        winner = tl.trans(winner)
        winner_dot = tl.trans(winner_dot)
        valid_winner = (
            valid_query[:, None]
            & valid_slot[None, :]
            & (winner >= 0)
            & (winner < LEAF_CAPACITY)
            & (count[None, :] >= 1.0)
        )
        safe_count = tl.where(valid_winner, count[None, :], 1.0)
        safe_winner = tl.where(valid_winner, winner, 0)

        state_key_sum = tl.load(
            state_k
            + (batch_kv * STATE_CAPACITY + slot[:, None]) * HEAD_DIM
            + head_offset[None, :],
            mask=valid_slot[:, None],
            other=0.0,
        ).to(tl.float32)
        state_value_sum = tl.load(
            state_v
            + (batch_kv * STATE_CAPACITY + slot[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=valid_slot[:, None],
            other=0.0,
        ).to(tl.float32)
        winner_value = tl.load(
            leaf_v
            + (batch_kv * LEAF_CAPACITY + safe_winner[:, :, None]) * VALUE_DIM
            + value_offset[None, None, :],
            mask=valid_winner[:, :, None],
            other=0.0,
        ).to(tl.float32)

        state_dot = tl.dot(
            query,
            tl.trans(state_key_sum.to(query.dtype)),
            out_dtype=tl.float32,
        )
        winner_score = SCALE_LOG2 * winner_dot
        winner_score = tl.where(valid_winner, winner_score, -float("inf"))
        residual_count = count[None, :] - 1.0
        valid_residual = valid_winner & (residual_count > 0.0)
        safe_residual_count = tl.where(valid_residual, residual_count, 1.0)
        if DISJOINT_RESIDUAL:
            residual_dot = (state_dot - winner_dot) / safe_residual_count
        else:
            residual_dot = state_dot / safe_count
        residual_score = SCALE_LOG2 * residual_dot + tl.log2(
            safe_residual_count
        )
        residual_score = tl.where(
            valid_residual, residual_score, -float("inf")
        )
        block_maximum = tl.maximum(
            tl.max(winner_score, axis=1),
            tl.max(residual_score, axis=1),
        )
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        winner_probability = tl.math.exp2(
            winner_score - new_maximum[:, None]
        )
        winner_probability = tl.where(
            valid_winner, winner_probability, 0.0
        )
        residual_probability = tl.math.exp2(
            residual_score - new_maximum[:, None]
        )
        residual_probability = tl.where(
            valid_residual, residual_probability, 0.0
        )
        denominator = denominator * correction + tl.sum(
            winner_probability + residual_probability, axis=1
        )
        if DISJOINT_RESIDUAL:
            state_weight = residual_probability / safe_residual_count
            winner_weight = winner_probability - state_weight
        else:
            state_weight = residual_probability / safe_count
            winner_weight = winner_probability
        state_update = tl.dot(
            state_weight.to(tl.bfloat16),
            state_value_sum.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        winner_update = tl.sum(
            winner_weight[:, :, None] * winner_value,
            axis=1,
        )
        accumulator = (
            accumulator * correction[:, None] + state_update + winner_update
        )
        maximum = new_maximum

    if INCLUDE_LOCAL:
        branch_lse = tl.load(
            local_lse + query_row,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
        branch_lse_log2 = branch_lse * 1.4426950408889634
        branch_output = tl.load(
            local_output + query_row[:, None] * VALUE_DIM + value_offset[None, :],
            mask=valid_query[:, None],
            other=0.0,
        ).to(tl.float32)
        new_maximum = tl.maximum(maximum, branch_lse_log2)
        correction = tl.math.exp2(maximum - new_maximum)
        probability = tl.math.exp2(branch_lse_log2 - new_maximum)
        denominator = denominator * correction + probability
        accumulator = (
            accumulator * correction[:, None]
            + probability[:, None] * branch_output
        )
        maximum = new_maximum

    valid_output = denominator > 0.0
    tl.store(
        output + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(
            valid_output[:, None], accumulator / denominator[:, None], 0.0
        ),
        mask=valid_query[:, None],
    )
    tl.store(
        output_lse + query_row,
        tl.where(
            valid_output,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=valid_query,
    )


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=["query_len", "state_len"],
)
def _all_centroid_coarse_fine_split_value_kernel(
    state_v,
    state_counts,
    leaf_v,
    winner_indices,
    winner_scores,
    state_scores,
    local_output,
    local_lse,
    output,
    output_lse,
    query_len,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    WINNER_BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    INCLUDE_LOCAL: tl.constexpr,
):
    """Value-tiled uniform combine using QK results from the top-1 pass."""
    query_program = tl.program_id(0).to(tl.int64)
    value_program = tl.program_id(1).to(tl.int64)
    query_blocks_per_kv = tl.cdiv(KV_GROUP_SIZE * query_len, BLOCK_M)
    batch_kv = query_program // query_blocks_per_kv
    query_block_in_kv = query_program - batch_kv * query_blocks_per_kv
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    query_in_kv = query_block_in_kv * BLOCK_M + tl.arange(0, BLOCK_M)
    query_group = query_in_kv // query_len
    query_position = query_in_kv - query_group * query_len
    valid_query = query_group < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_group
    query_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query_position
    ).to(tl.int64)
    winner_query_block = query_in_kv // WINNER_BLOCK_M
    winner_query_lane = query_in_kv - winner_query_block * WINNER_BLOCK_M

    value_offset = value_program * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_value = value_offset < VALUE_DIM
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    slot_offset = tl.arange(0, BLOCK_N)

    for slot_begin in tl.range(0, state_len, BLOCK_N, num_stages=1):
        slot = slot_begin + slot_offset
        valid_slot = slot < state_len
        count = tl.load(
            state_counts + batch_kv * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        record_offset = (
            (
                (batch_kv * STATE_CAPACITY + slot[:, None]) * QUERY_BLOCKS
                + winner_query_block[None, :]
            )
            * WINNER_BLOCK_M
            + winner_query_lane[None, :]
        )
        winner = tl.trans(
            tl.load(
                winner_indices + record_offset,
                mask=valid_slot[:, None] & valid_query[None, :],
                other=-1,
            ).to(tl.int64)
        )
        winner_dot = tl.trans(
            tl.load(
                winner_scores + record_offset,
                mask=valid_slot[:, None] & valid_query[None, :],
                other=-float("inf"),
            ).to(tl.float32)
        )
        state_dot = tl.trans(
            tl.load(
                state_scores + record_offset,
                mask=valid_slot[:, None] & valid_query[None, :],
                other=0.0,
            ).to(tl.float32)
        )
        valid_winner = (
            valid_query[:, None]
            & valid_slot[None, :]
            & (winner >= 0)
            & (winner < LEAF_CAPACITY)
            & (count[None, :] >= 1.0)
        )
        safe_count = tl.where(valid_winner, count[None, :], 1.0)
        safe_winner = tl.where(valid_winner, winner, 0)

        winner_score = tl.where(
            valid_winner,
            SCALE_LOG2 * winner_dot,
            -float("inf"),
        )
        residual_count = count[None, :] - 1.0
        valid_residual = valid_winner & (residual_count > 0.0)
        safe_residual_count = tl.where(valid_residual, residual_count, 1.0)
        residual_score = (
            SCALE_LOG2 * (state_dot - winner_dot) / safe_residual_count
            + tl.log2(safe_residual_count)
        )
        residual_score = tl.where(
            valid_residual, residual_score, -float("inf")
        )
        block_maximum = tl.maximum(
            tl.max(winner_score, axis=1),
            tl.max(residual_score, axis=1),
        )
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        winner_probability = tl.where(
            valid_winner,
            tl.math.exp2(winner_score - new_maximum[:, None]),
            0.0,
        )
        residual_probability = tl.where(
            valid_residual,
            tl.math.exp2(residual_score - new_maximum[:, None]),
            0.0,
        )
        denominator = denominator * correction + tl.sum(
            winner_probability + residual_probability, axis=1
        )

        state_value_sum = tl.load(
            state_v
            + (batch_kv * STATE_CAPACITY + slot[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=valid_slot[:, None] & valid_value[None, :],
            other=0.0,
        )
        winner_value = tl.load(
            leaf_v
            + (batch_kv * LEAF_CAPACITY + safe_winner[:, :, None]) * VALUE_DIM
            + value_offset[None, None, :],
            mask=valid_winner[:, :, None] & valid_value[None, None, :],
            other=0.0,
        ).to(tl.float32)
        state_weight = residual_probability / safe_residual_count
        winner_weight = winner_probability - state_weight
        state_update = tl.dot(
            state_weight.to(tl.bfloat16),
            state_value_sum.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        winner_update = tl.sum(
            winner_weight[:, :, None] * winner_value,
            axis=1,
        )
        accumulator = (
            accumulator * correction[:, None] + state_update + winner_update
        )
        maximum = new_maximum

    if INCLUDE_LOCAL:
        branch_lse = tl.load(
            local_lse + query_row,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
        branch_lse_log2 = branch_lse * 1.4426950408889634
        branch_output = tl.load(
            local_output
            + query_row[:, None] * VALUE_DIM
            + value_offset[None, :],
            mask=valid_query[:, None] & valid_value[None, :],
            other=0.0,
        ).to(tl.float32)
        new_maximum = tl.maximum(maximum, branch_lse_log2)
        correction = tl.math.exp2(maximum - new_maximum)
        probability = tl.math.exp2(branch_lse_log2 - new_maximum)
        denominator = denominator * correction + probability
        accumulator = (
            accumulator * correction[:, None]
            + probability[:, None] * branch_output
        )
        maximum = new_maximum

    valid_output = valid_query[:, None] & valid_value[None, :]
    tl.store(
        output + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(
            denominator[:, None] > 0.0,
            accumulator / denominator[:, None],
            0.0,
        ),
        mask=valid_output,
    )
    tl.store(
        output_lse + query_row,
        tl.where(
            denominator > 0.0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=valid_query & (value_program == 0),
    )


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=["query_len", "state_len"],
)
def _all_centroid_coarse_fine_packed_value_kernel(
    q,
    state_k,
    state_v,
    state_counts,
    leaf_v,
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    winner_indices,
    winner_scores,
    local_output,
    local_lse,
    output,
    output_lse,
    query_len,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    QUERY_BLOCKS: tl.constexpr,
    WINNER_BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CENTROIDS_PER_PROGRAM: tl.constexpr,
    LEAF_BLOCK_N: tl.constexpr,
    BLOCK_K_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    INCLUDE_LOCAL: tl.constexpr,
    INDEXED: tl.constexpr,
    DISJOINT_RESIDUAL: tl.constexpr,
):
    """Coalesced exact-winner/residual value accumulation.

    Winner values are selected with register-resident one-hot weights over a
    regular centroid/page tile.  This reads every leaf value once per query
    tile instead of gathering one unrelated vector for every query/centroid.
    """
    query_program = tl.program_id(0).to(tl.int64)
    value_program = tl.program_id(1).to(tl.int64)
    query_blocks_per_kv = tl.cdiv(KV_GROUP_SIZE * query_len, BLOCK_M)
    batch_kv = query_program // query_blocks_per_kv
    query_block_in_kv = query_program - batch_kv * query_blocks_per_kv
    batch = batch_kv // KV_HEADS
    kv_head = batch_kv - batch * KV_HEADS
    query_in_kv = query_block_in_kv * BLOCK_M + tl.arange(0, BLOCK_M)
    query_group = query_in_kv // query_len
    query_position = query_in_kv - query_group * query_len
    valid_query = query_group < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_group
    query_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query_position
    ).to(tl.int64)
    winner_query_block = query_in_kv // WINNER_BLOCK_M
    winner_query_lane = query_in_kv - winner_query_block * WINNER_BLOCK_M

    value_offset = value_program * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_value = value_offset < VALUE_DIM
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    centroid = tl.arange(0, CENTROIDS_PER_PROGRAM)
    packed_leaf = tl.arange(0, CENTROIDS_PER_PROGRAM * LEAF_BLOCK_N)
    packed_centroid = packed_leaf // LEAF_BLOCK_N
    leaf_offset = packed_leaf - packed_centroid * LEAF_BLOCK_N

    for slot_begin in tl.range(
        0, state_len, CENTROIDS_PER_PROGRAM, num_stages=1
    ):
        slot = slot_begin + centroid
        valid_slot = slot < state_len
        count = tl.load(
            state_counts + batch_kv * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        record_offset = (
            (
                (batch_kv * STATE_CAPACITY + slot[:, None]) * QUERY_BLOCKS
                + winner_query_block[None, :]
            )
            * WINNER_BLOCK_M
            + winner_query_lane[None, :]
        )
        winner = tl.trans(
            tl.load(
                winner_indices + record_offset,
                mask=valid_slot[:, None] & valid_query[None, :],
                other=-1,
            ).to(tl.int64)
        )
        winner_dot = tl.trans(
            tl.load(
                winner_scores + record_offset,
                mask=valid_slot[:, None] & valid_query[None, :],
                other=-float("inf"),
            ).to(tl.float32)
        )
        valid_winner = (
            valid_query[:, None]
            & valid_slot[None, :]
            & (winner >= 0)
            & (winner < LEAF_CAPACITY)
            & (count[None, :] >= 1.0)
        )
        safe_count = tl.where(valid_winner, count[None, :], 1.0)
        state_dot = tl.zeros(
            (BLOCK_M, CENTROIDS_PER_PROGRAM), tl.float32
        )
        for dim_begin in tl.static_range(0, HEAD_DIM, BLOCK_K_D):
            key_dim = dim_begin + tl.arange(0, BLOCK_K_D)
            valid_key_dim = key_dim < HEAD_DIM
            query = tl.load(
                q + query_row[:, None] * HEAD_DIM + key_dim[None, :],
                mask=valid_query[:, None] & valid_key_dim[None, :],
                other=0.0,
            )
            state_key_sum = tl.load(
                state_k
                + (batch_kv * STATE_CAPACITY + slot[:, None]) * HEAD_DIM
                + key_dim[None, :],
                mask=valid_slot[:, None] & valid_key_dim[None, :],
                other=0.0,
            )
            state_dot += tl.dot(
                query,
                tl.trans(state_key_sum),
                out_dtype=tl.float32,
            )
        winner_score = tl.where(
            valid_winner, SCALE_LOG2 * winner_dot, -float("inf")
        )
        residual_count = count[None, :] - 1.0
        valid_residual = valid_winner & (residual_count > 0.0)
        safe_residual_count = tl.where(valid_residual, residual_count, 1.0)
        if DISJOINT_RESIDUAL:
            residual_dot = (state_dot - winner_dot) / safe_residual_count
        else:
            residual_dot = state_dot / safe_count
        residual_score = (
            SCALE_LOG2 * residual_dot + tl.log2(safe_residual_count)
        )
        residual_score = tl.where(
            valid_residual, residual_score, -float("inf")
        )
        block_maximum = tl.maximum(
            tl.max(winner_score, axis=1),
            tl.max(residual_score, axis=1),
        )
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        winner_probability = tl.where(
            valid_winner,
            tl.math.exp2(winner_score - new_maximum[:, None]),
            0.0,
        )
        residual_probability = tl.where(
            valid_residual,
            tl.math.exp2(residual_score - new_maximum[:, None]),
            0.0,
        )
        denominator = denominator * correction + tl.sum(
            winner_probability + residual_probability, axis=1
        )
        if DISJOINT_RESIDUAL:
            state_weight = residual_probability / safe_residual_count
            winner_weight = winner_probability - state_weight
        else:
            state_weight = residual_probability / safe_count
            winner_weight = winner_probability

        state_value_sum = tl.load(
            state_v
            + (batch_kv * STATE_CAPACITY + slot[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=valid_slot[:, None] & valid_value[None, :],
            other=0.0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            state_weight.to(tl.bfloat16),
            state_value_sum.to(tl.bfloat16),
            out_dtype=tl.float32,
        )

        packed_slot = slot_begin + packed_centroid
        packed_count = tl.load(
            slot_lengths + batch_kv * STATE_CAPACITY + packed_slot,
            mask=packed_slot < state_len,
            other=0,
        ).to(tl.int32)
        maximum_key_count = tl.max(packed_count, axis=0)
        for key_begin in tl.range(
            0, maximum_key_count, LEAF_BLOCK_N, num_stages=1
        ):
            logical_key = key_begin + leaf_offset
            valid_leaf = (
                (packed_slot < state_len) & (logical_key < packed_count)
            )
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    slot_pages
                    + (batch_kv * STATE_CAPACITY + packed_slot)
                    * INLINE_PAGES_PER_SLOT
                    + page_ordinal,
                    mask=valid_leaf,
                    other=-1,
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    batch_kv,
                    packed_slot,
                    page_ordinal,
                    valid_leaf,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            valid_leaf &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
            physical_token = page_id * PAGE_SIZE + within_page
            if INDEXED:
                leaf_index = tl.load(
                    page_indices
                    + batch_kv * PAGE_CAPACITY * PAGE_SIZE
                    + physical_token,
                    mask=valid_leaf,
                    other=-1,
                ).to(tl.int64)
            else:
                leaf_index = physical_token
            valid_leaf &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            safe_leaf = tl.where(valid_leaf, leaf_index, 0)
            leaf_values = tl.load(
                leaf_v
                + (batch_kv * LEAF_CAPACITY + safe_leaf[:, None]) * VALUE_DIM
                + value_offset[None, :],
                mask=valid_leaf[:, None] & valid_value[None, :],
                other=0.0,
            )
            leaf_by_centroid = tl.reshape(
                safe_leaf,
                (CENTROIDS_PER_PROGRAM, LEAF_BLOCK_N),
            )
            valid_leaf_by_centroid = tl.reshape(
                valid_leaf,
                (CENTROIDS_PER_PROGRAM, LEAF_BLOCK_N),
            )
            exact_weight = tl.where(
                valid_winner[:, :, None]
                & valid_leaf_by_centroid[None, :, :]
                & (winner[:, :, None] == leaf_by_centroid[None, :, :]),
                winner_weight[:, :, None],
                0.0,
            )
            exact_weight = tl.reshape(
                exact_weight,
                (BLOCK_M, CENTROIDS_PER_PROGRAM * LEAF_BLOCK_N),
            )
            accumulator += tl.dot(
                exact_weight.to(tl.bfloat16),
                leaf_values.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
        maximum = new_maximum

    if INCLUDE_LOCAL:
        branch_lse = tl.load(
            local_lse + query_row,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
        branch_lse_log2 = branch_lse * 1.4426950408889634
        branch_output = tl.load(
            local_output
            + query_row[:, None] * VALUE_DIM
            + value_offset[None, :],
            mask=valid_query[:, None] & valid_value[None, :],
            other=0.0,
        ).to(tl.float32)
        new_maximum = tl.maximum(maximum, branch_lse_log2)
        correction = tl.math.exp2(maximum - new_maximum)
        probability = tl.math.exp2(branch_lse_log2 - new_maximum)
        denominator = denominator * correction + probability
        accumulator = (
            accumulator * correction[:, None]
            + probability[:, None] * branch_output
        )
        maximum = new_maximum

    valid_output = valid_query[:, None] & valid_value[None, :]
    tl.store(
        output + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(
            denominator[:, None] > 0.0,
            accumulator / denominator[:, None],
            0.0,
        ),
        mask=valid_output,
    )
    tl.store(
        output_lse + query_row,
        tl.where(
            denominator > 0.0,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=valid_query & (value_program == 0),
    )


def all_centroid_top1_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    state_counts: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor | None,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    slot_offsets: torch.Tensor | None = None,
    packed_leaf_indices: torch.Tensor | None = None,
    packed_leaf_slots: torch.Tensor | None = None,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    local_branch: tuple[torch.Tensor, torch.Tensor] | None = None,
    hash_probes: int = 8,
    winner_block_m: int = 64,
    winner_block_n: int = 16,
    winner_block_d: int = 128,
    centroids_per_program: int = 8,
    combine_centroids_per_program: int = 16,
    attention_block_m: int = 64,
    attention_block_n: int = 8,
    attention_block_d: int = 128,
    winner_num_warps: int = 4,
    attention_num_warps: int = 8,
    waves_per_eu: int = 1,
    fused_prefill: bool = False,
    fused_block_m: int | None = None,
    fused_leaf_block_n: int | None = None,
    fused_block_d: int | None = None,
    fused_centroids_per_program: int | None = None,
    fused_num_warps: int | None = None,
    disjoint_residual: bool = False,
    output_buffer: torch.Tensor | None = None,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run uniform exact-winner/residual attention over every centroid."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("all-centroid LOD attention is forward-only")
    if q.ndim != 4 or state_k.ndim != 4 or state_v.ndim != 4:
        raise ValueError("all-centroid attention requires rank-four Q/state tensors")
    batch, query_heads, query_len, head_dim = q.shape
    cache_batch, kv_heads, state_capacity, state_head_dim = state_k.shape
    if cache_batch != batch:
        raise ValueError("all-centroid prototype requires matching cache/query batches")
    if state_head_dim != head_dim or int(leaf_k.size(-1)) != head_dim:
        raise ValueError("query, state, and leaf key dimensions must match")
    value_dim = int(state_v.size(-1))
    if value_dim != head_dim or int(leaf_v.size(-1)) != value_dim:
        raise ValueError("prototype currently requires equal key/value dimensions")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if not 0 < state_len <= state_capacity:
        raise ValueError("active state length must fit state capacity")
    if int(slot_pages.size(2)) != state_capacity:
        raise ValueError("slot page table does not match state capacity")
    if slot_offsets is None:
        slot_offsets = torch.nn.functional.pad(
            torch.cumsum(slot_lengths, dim=-1, dtype=torch.int32),
            (1, 0),
        )
    if tuple(slot_offsets.shape) != (
        batch,
        kv_heads,
        state_capacity + 1,
    ):
        raise ValueError("slot offsets do not match state capacity")
    indexed = page_indices is not None
    if indexed:
        page_capacity = int(page_indices.size(2))
        page_size = int(page_indices.size(3))
        storage_k = leaf_k
        storage_v = leaf_v
    else:
        if leaf_k.ndim != 5 or leaf_v.ndim != 5:
            raise ValueError("physical page storage must have rank five")
        page_capacity = int(leaf_k.size(2))
        page_size = int(leaf_k.size(3))
        storage_k = leaf_k.flatten(2, 3)
        storage_v = leaf_v.flatten(2, 3)
    if page_size != 16:
        raise ValueError("all-centroid prototype requires 16-token pages")
    leaf_capacity = int(storage_k.size(2))
    if fused_prefill and (
        packed_leaf_indices is None or packed_leaf_slots is None
    ):
        packed_leaf_indices, packed_leaf_slots = build_all_centroid_leaf_stream(
            page_indices,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            slot_offsets,
            leaf_capacity=leaf_capacity,
            hash_probes=hash_probes,
        )
    output_shape = (batch, query_heads, query_len, value_dim)
    if output_buffer is None:
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
    else:
        if tuple(output_buffer.shape) != output_shape:
            raise ValueError("all-centroid output buffer has the wrong shape")
        output = output_buffer
    output_lse = torch.empty(output_shape[:-1], dtype=torch.float32, device=q.device)
    if local_branch is None:
        local_output = output
        local_lse = output_lse
    else:
        local_output, local_lse = local_branch
        if tuple(local_output.shape) != output_shape:
            raise ValueError("local output does not match all-centroid queries")
        if tuple(local_lse.shape) != output_shape[:-1]:
            raise ValueError("local LSE does not match all-centroid queries")

    if fused_prefill:
        if packed_leaf_indices is None or packed_leaf_slots is None:
            raise AssertionError("packed all-centroid leaf metadata is missing")
        fused_block_m = fused_block_m or winner_block_m
        fused_leaf_block_n = fused_leaf_block_n or winner_block_n
        fused_block_d = fused_block_d or winner_block_d
        fused_centroids_per_program = (
            fused_centroids_per_program or centroids_per_program
        )
        fused_num_warps = fused_num_warps or attention_num_warps
        packed_keys = fused_centroids_per_program * fused_leaf_block_n
        if packed_keys != 128:
            raise ValueError("fused all-centroid prefill requires 128 packed leaves")
        if (
            fused_centroids_per_program < 16
            or fused_centroids_per_program > 64
            or 64 % fused_centroids_per_program
        ):
            raise ValueError(
                "fused all-centroid prefill requires 16, 32, or 64 centroids"
            )
        value_leaf_block_n = 64 // fused_centroids_per_program
        if fused_leaf_block_n % value_leaf_block_n:
            raise ValueError("winner leaf block must split into 64-leaf value tiles")
        fused_begin = None
        fused_end = None
        if timing_events is not None:
            fused_begin = torch.cuda.Event(enable_timing=True)
            fused_end = torch.cuda.Event(enable_timing=True)
            fused_begin.record()
        fused_query_blocks = triton.cdiv(
            kv_group_size * query_len, fused_block_m
        )
        fused_programs = batch * kv_heads * fused_query_blocks
        scratch_shape = (
            fused_programs,
            fused_block_m,
            fused_centroids_per_program,
        )
        scratch_leaf = torch.empty(
            scratch_shape, dtype=torch.int32, device=q.device
        )
        scratch_score = torch.empty(
            scratch_shape, dtype=torch.float32, device=q.device
        )
        _all_centroid_group_stream_fused_kernel[
            (fused_programs,)
        ](
            q.contiguous(),
            state_k,
            state_v,
            state_counts,
            slot_lengths,
            slot_offsets,
            packed_leaf_indices,
            packed_leaf_slots,
            storage_k,
            storage_v,
            scratch_leaf,
            scratch_score,
            local_output,
            local_lse,
            output,
            output_lse,
            query_len,
            state_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            STATE_CAPACITY=state_capacity,
            LEAF_CAPACITY=leaf_capacity,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            BLOCK_M=fused_block_m,
            BLOCK_D=fused_block_d,
            CENTROIDS_PER_PROGRAM=fused_centroids_per_program,
            LEAF_BLOCK_N=packed_keys,
            SCALE_LOG2=float(scale) * math.log2(math.e),
            INCLUDE_LOCAL=local_branch is not None,
            DISJOINT_RESIDUAL=disjoint_residual,
            num_warps=fused_num_warps,
            waves_per_eu=waves_per_eu,
        )
        if fused_end is not None:
            fused_end.record()
        if timing_events is not None:
            if fused_begin is None or fused_end is None:
                raise AssertionError("fused timing events were not initialized")
            timing_events.setdefault("fused", []).append(
                (fused_begin, fused_end)
            )
        empty_indices = torch.empty(0, dtype=torch.int32, device=q.device)
        empty_scores = torch.empty(0, dtype=torch.float32, device=q.device)
        return output, output_lse, empty_indices, empty_scores, empty_scores

    query_blocks = triton.cdiv(kv_group_size * query_len, winner_block_m)
    winner_indices = torch.empty(
        batch,
        kv_heads,
        state_capacity,
        query_blocks,
        winner_block_m,
        dtype=torch.int32,
        device=q.device,
    )
    winner_scores = torch.empty_like(winner_indices, dtype=torch.float32)
    top1_begin = None
    top1_end = None
    if timing_events is not None:
        top1_begin = torch.cuda.Event(enable_timing=True)
        top1_end = torch.cuda.Event(enable_timing=True)
        top1_begin.record()
    grid = (
        batch * kv_heads * triton.cdiv(state_capacity, centroids_per_program),
        query_blocks,
    )
    _all_centroid_top1_leaf_kernel[grid](
        q.contiguous(),
        storage_k,
        page_indices if indexed else slot_pages,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        winner_indices,
        winner_scores,
        query_len,
        state_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        STATE_CAPACITY=state_capacity,
        PAGE_CAPACITY=page_capacity,
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        LEAF_CAPACITY=leaf_capacity,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        QUERY_BLOCKS=query_blocks,
        BLOCK_M=winner_block_m,
        BLOCK_N=winner_block_n,
        BLOCK_D=winner_block_d,
        CENTROIDS_PER_PROGRAM=centroids_per_program,
        INDEXED=indexed,
        num_warps=winner_num_warps,
        waves_per_eu=waves_per_eu,
    )
    if top1_end is not None:
        top1_end.record()

    combine_begin = None
    combine_end = None
    if timing_events is not None:
        combine_begin = torch.cuda.Event(enable_timing=True)
        combine_end = torch.cuda.Event(enable_timing=True)
        combine_begin.record()
    attention_query_blocks = triton.cdiv(
        kv_group_size * query_len, attention_block_m
    )
    if query_len > 1:
        if (
            combine_centroids_per_program < 16
            or combine_centroids_per_program > 64
            or 128 % combine_centroids_per_program
        ):
            raise ValueError(
                "combine centroid tile must be a divisor of 128 up to 64"
            )
        combine_leaf_block_n = 128 // combine_centroids_per_program
        _all_centroid_coarse_fine_packed_value_kernel[
            (
                batch * kv_heads * attention_query_blocks,
                triton.cdiv(value_dim, attention_block_d),
            )
        ](
            q.contiguous(),
            state_k,
            state_v,
            state_counts,
            storage_v,
            page_indices if indexed else slot_pages,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            winner_indices,
            winner_scores,
            local_output,
            local_lse,
            output,
            output_lse,
            query_len,
            state_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            STATE_CAPACITY=state_capacity,
            PAGE_CAPACITY=page_capacity,
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            HASH_CAPACITY=int(overflow_page_keys.size(2)),
            HASH_PROBES=hash_probes,
            LEAF_CAPACITY=leaf_capacity,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            PAGE_SIZE=page_size,
            QUERY_BLOCKS=query_blocks,
            WINNER_BLOCK_M=winner_block_m,
            BLOCK_M=attention_block_m,
            CENTROIDS_PER_PROGRAM=combine_centroids_per_program,
            LEAF_BLOCK_N=combine_leaf_block_n,
            BLOCK_K_D=winner_block_d,
            BLOCK_D=attention_block_d,
            SCALE_LOG2=float(scale) * math.log2(math.e),
            INCLUDE_LOCAL=local_branch is not None,
            INDEXED=indexed,
            DISJOINT_RESIDUAL=disjoint_residual,
            num_warps=attention_num_warps,
            waves_per_eu=waves_per_eu,
        )
    else:
        _all_centroid_coarse_fine_kernel[
            (batch * kv_heads * attention_query_blocks,)
        ](
            q.contiguous(),
            state_k,
            state_v,
            state_counts,
            storage_v,
            winner_indices,
            winner_scores,
            local_output,
            local_lse,
            output,
            output_lse,
            query_len,
            state_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            STATE_CAPACITY=state_capacity,
            LEAF_CAPACITY=leaf_capacity,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            QUERY_BLOCKS=query_blocks,
            WINNER_BLOCK_M=winner_block_m,
            BLOCK_M=attention_block_m,
            BLOCK_N=max(16, attention_block_n),
            SCALE_LOG2=float(scale) * math.log2(math.e),
            INCLUDE_LOCAL=local_branch is not None,
            DISJOINT_RESIDUAL=disjoint_residual,
            num_warps=attention_num_warps,
            waves_per_eu=waves_per_eu,
        )
    if combine_end is not None:
        combine_end.record()
    if timing_events is not None:
        if top1_begin is None or top1_end is None:
            raise AssertionError("top-1 timing events were not initialized")
        if combine_begin is None or combine_end is None:
            raise AssertionError("combine timing events were not initialized")
        timing_events.setdefault("top1", []).append((top1_begin, top1_end))
        timing_events.setdefault("combine", []).append(
            (combine_begin, combine_end)
        )
    empty_scores = torch.empty(0, dtype=torch.float32, device=q.device)
    return output, output_lse, winner_indices, winner_scores, empty_scores


__all__ = ["all_centroid_top1_attention", "build_all_centroid_leaf_stream"]
