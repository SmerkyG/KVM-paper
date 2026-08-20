"""Forward-only Triton attention over 16-token leaf pages.

Queries are dispatched MoE-style by routed state slot.  Each expert therefore
shares one page list, allowing the attention kernel to retain an ordinary
``BLOCK_M x BLOCK_N`` matrix multiply while gathering K/V directly from the
persistent page pool.  No historical K/V sort or repack is required.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _page_hash_index(key, HASH_CAPACITY: tl.constexpr):
    value = key.to(tl.uint32)
    value ^= value >> 16
    value *= 0x7FEB352D
    value ^= value >> 15
    value *= 0x846CA68B
    value ^= value >> 16
    return value & (HASH_CAPACITY - 1)


@triton.jit
def _lookup_page_id(
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    kv_row,
    slot,
    page_ordinal,
    valid,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
):
    if HASH_PROBES == -1:
        # Two-level page directory.  ``slot_pages`` is the compact root table;
        # every root entry uses one physical K/V page ID as the handle for a
        # 64-entry directory row in ``overflow_page_values``. HASH_CAPACITY is
        # the number of directory rows per KV row in this mode. The sentinel
        # value -1 selects this direct lookup without adding another argument
        # to every attention kernel that consumes page metadata.
        directory_ordinal = page_ordinal // 64
        directory_offset = page_ordinal % 64
        root_valid = valid & (directory_ordinal < INLINE_PAGES_PER_SLOT)
        directory_id = tl.load(
            slot_pages
            + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            + directory_ordinal,
            mask=root_valid,
            other=-1,
        ).to(tl.int32)
        directory_valid = (
            root_valid & (directory_id >= 0) & (directory_id < HASH_CAPACITY)
        )
        page_id = tl.load(
            overflow_page_values
            + (kv_row * HASH_CAPACITY + directory_id) * 64
            + directory_offset,
            mask=directory_valid,
            other=-1,
        ).to(tl.int32)
    else:
        inline = valid & (page_ordinal < INLINE_PAGES_PER_SLOT)
        page_id = tl.load(
            slot_pages
            + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            + page_ordinal,
            mask=inline,
            other=-1,
        ).to(tl.int32)
    if HASH_PROBES > 0:
        if tl.load(overflow_used) != 0:
            # Qwen3.5's 256K context uses at most 16K pages in one posting list.
            # A fixed stride keeps keys valid if the physical pool later grows.
            lookup_key = (slot * 65_536 + page_ordinal).to(tl.int32)
            index = _page_hash_index(lookup_key, HASH_CAPACITY)
            active = valid & ~inline
            for _ in tl.static_range(0, HASH_PROBES):
                stored_key = tl.load(
                    overflow_page_keys + kv_row * HASH_CAPACITY + index,
                    mask=active,
                    other=-2,
                )
                match = active & (stored_key == lookup_key)
                stored_value = tl.load(
                    overflow_page_values + kv_row * HASH_CAPACITY + index,
                    mask=match,
                    other=-1,
                )
                page_id = tl.where(match, stored_value, page_id)
                active &= ~match
                index = (index + 1) & (HASH_CAPACITY - 1)
    return page_id


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _candidate_page_mass_kernel(
    q,
    page_sum_k,
    page_counts,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    candidates,
    output_scores,
    query_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CANDIDATE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE_BLOCK_N: tl.constexpr,
):
    """Estimate candidate-slot log-mass from its existing page centroids."""
    query_row = tl.program_id(0).to(tl.int64)
    candidate_rank = tl.program_id(1).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head
    slot = tl.load(candidates + query_row * CANDIDATE_COUNT + candidate_rank).to(
        tl.int64
    )
    valid_slot = (slot >= 0) & (slot < STATE_CAPACITY)
    slot = tl.where(valid_slot, slot, 0)
    key_count = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + slot,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    slot_page_count = (key_count + PAGE_SIZE - 1) // PAGE_SIZE
    dim = tl.arange(0, HEAD_DIM)
    page_offset = tl.arange(0, PAGE_BLOCK_N)
    query = tl.load(q + query_row * HEAD_DIM + dim)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    if HASH_PROBES == 0:
        page_table = (
            slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
        )

    for page_begin in tl.range(0, slot_page_count, PAGE_BLOCK_N, num_stages=1):
        page_ordinal = page_begin + page_offset
        valid_page = valid_slot & (page_ordinal < slot_page_count)
        if HASH_PROBES == 0:
            page_id = tl.load(
                page_table + page_ordinal,
                mask=valid_page,
                other=0,
            ).to(tl.int64)
        else:
            page_id = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                kv_row,
                slot,
                page_ordinal,
                valid_page,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        valid_page &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
        page_id = tl.where(valid_page, page_id, 0)
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page_id,
            mask=valid_page,
            other=1,
        ).to(tl.float32)
        key_sum = tl.load(
            page_sum_k
            + (kv_row * PAGE_CAPACITY + page_id[:, None]) * HEAD_DIM
            + dim[None, :],
            mask=valid_page[:, None],
            other=0.0,
        ).to(tl.float32)
        score = SCALE * tl.sum(
            (key_sum / count[:, None]) * query[None, :].to(tl.float32),
            axis=1,
        ) + tl.log(count)
        score = tl.where(valid_page, score, -float("inf"))
        block_maximum = tl.max(score, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - new_maximum)
        probability = tl.where(valid_page, tl.exp(score - new_maximum), 0.0)
        denominator = denominator * correction + tl.sum(probability, axis=0)
        maximum = new_maximum

    mass_score = tl.where(
        valid_slot & (denominator > 0.0),
        maximum + tl.log(denominator),
        -float("inf"),
    )
    tl.store(
        output_scores + query_row * CANDIDATE_COUNT + candidate_rank,
        mass_score,
    )


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _candidate_leaf_mass_kernel(
    q,
    page_k,
    leaf_k,
    page_indices,
    page_counts,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    candidates,
    output_scores,
    query_len,
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
    PAGE_SIZE: tl.constexpr,
    CANDIDATE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    VIRTUAL: tl.constexpr,
):
    """Compute exact token-level log-mass for a candidate state slot."""
    query_row = tl.program_id(0).to(tl.int64)
    candidate_rank = tl.program_id(1).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head
    slot = tl.load(candidates + query_row * CANDIDATE_COUNT + candidate_rank).to(
        tl.int64
    )
    valid_slot = (slot >= 0) & (slot < STATE_CAPACITY)
    slot = tl.where(valid_slot, slot, 0)
    key_count = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + slot,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    slot_page_count = (key_count + PAGE_SIZE - 1) // PAGE_SIZE
    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, PAGE_SIZE)
    query = tl.load(q + query_row * HEAD_DIM + dim).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    if HASH_PROBES == 0:
        page_table = (
            slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
        )

    for page_ordinal in tl.range(0, slot_page_count, num_stages=1):
        valid_page = valid_slot & (page_ordinal < slot_page_count)
        if HASH_PROBES == 0:
            page_id = tl.load(
                page_table + page_ordinal,
                mask=valid_page,
                other=0,
            ).to(tl.int64)
        else:
            page_id = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                kv_row,
                slot,
                page_ordinal,
                valid_page,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        valid_page &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
        page_id = tl.where(valid_page, page_id, 0)
        page_count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page_id,
            mask=valid_page,
            other=0,
        ).to(tl.int32)
        valid_token = valid_page & (token_offset < page_count)
        if VIRTUAL:
            leaf_index = tl.load(
                page_indices
                + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
                + token_offset,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            leaf_index = tl.where(valid_token, leaf_index, 0)
            key = tl.load(
                leaf_k
                + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * HEAD_DIM
                + dim[None, :],
                mask=valid_token[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            key = tl.load(
                page_k
                + (
                    (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
                    + token_offset[:, None]
                )
                * HEAD_DIM
                + dim[None, :],
                mask=valid_token[:, None],
                other=0.0,
            ).to(tl.float32)
        score = SCALE * tl.sum(key * query[None, :], axis=1)
        score = tl.where(valid_token, score, -float("inf"))
        block_maximum = tl.max(score, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - new_maximum)
        probability = tl.where(valid_token, tl.exp(score - new_maximum), 0.0)
        denominator = denominator * correction + tl.sum(probability, axis=0)
        maximum = new_maximum

    mass_score = tl.where(
        valid_slot & (denominator > 0.0),
        maximum + tl.log(denominator),
        -float("inf"),
    )
    tl.store(
        output_scores + query_row * CANDIDATE_COUNT + candidate_rank,
        mass_score,
    )


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _candidate_virtual_leaf_target_output_kernel(
    q,
    baseline_output,
    baseline_lse,
    candidate_coarse_lse,
    state_sum_v,
    state_counts,
    leaf_k,
    leaf_v,
    page_indices,
    page_counts,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    candidates,
    target_output,
    query_len,
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
    CANDIDATE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Use all candidates as a local exact-output target for route utility."""
    query_row = tl.program_id(0).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head
    key_dim = tl.arange(0, HEAD_DIM)
    value_dim = tl.arange(0, VALUE_DIM)
    token_offset = tl.arange(0, PAGE_SIZE)
    query = tl.load(q + query_row * HEAD_DIM + key_dim).to(tl.float32)
    closed_lse = tl.load(baseline_lse + query_row).to(tl.float32)
    numerator_adjustment = tl.zeros((VALUE_DIM,), tl.float32)
    denominator_adjustment = tl.zeros((), tl.float32)

    for candidate_rank in tl.static_range(0, CANDIDATE_COUNT):
        slot = tl.load(candidates + query_row * CANDIDATE_COUNT + candidate_rank).to(
            tl.int64
        )
        valid_slot = (slot >= 0) & (slot < STATE_CAPACITY)
        slot = tl.where(valid_slot, slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0,
        ).to(tl.int32)
        slot_page_count = (key_count + PAGE_SIZE - 1) // PAGE_SIZE
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        accumulator = tl.zeros((VALUE_DIM,), tl.float32)
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )

        for page_ordinal in tl.range(0, slot_page_count, num_stages=1):
            valid_page = valid_slot & (page_ordinal < slot_page_count)
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal,
                    mask=valid_page,
                    other=0,
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid_page,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            valid_page &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
            page_id = tl.where(valid_page, page_id, 0)
            page_count = tl.load(
                page_counts + kv_row * PAGE_CAPACITY + page_id,
                mask=valid_page,
                other=0,
            ).to(tl.int32)
            valid_token = valid_page & (token_offset < page_count)
            leaf_index = tl.load(
                page_indices
                + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
                + token_offset,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            leaf_index = tl.where(valid_token, leaf_index, 0)
            key = tl.load(
                leaf_k
                + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * HEAD_DIM
                + key_dim[None, :],
                mask=valid_token[:, None],
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                leaf_v
                + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * VALUE_DIM
                + value_dim[None, :],
                mask=valid_token[:, None],
                other=0.0,
            ).to(tl.float32)
            score = SCALE * tl.sum(key * query[None, :], axis=1)
            score = tl.where(valid_token, score, -float("inf"))
            block_maximum = tl.max(score, axis=0)
            new_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.exp(maximum - new_maximum)
            probability = tl.where(valid_token, tl.exp(score - new_maximum), 0.0)
            denominator = denominator * correction + tl.sum(probability, axis=0)
            accumulator = accumulator * correction + tl.sum(
                probability[:, None] * value, axis=0
            )
            maximum = new_maximum

        exact_lse = maximum + tl.log(denominator)
        exact_value = accumulator / denominator
        coarse_lse = tl.load(
            candidate_coarse_lse + query_row * CANDIDATE_COUNT + candidate_rank
        ).to(tl.float32)
        coarse_relative_mass = tl.exp(coarse_lse - closed_lse)
        exact_relative_mass = tl.exp(exact_lse - closed_lse)
        state_count = tl.load(
            state_counts + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=1.0,
        ).to(tl.float32)
        coarse_value = (
            tl.load(
                state_sum_v + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM + value_dim,
                mask=valid_slot,
                other=0.0,
            ).to(tl.float32)
            / state_count
        )
        denominator_adjustment += exact_relative_mass - coarse_relative_mass
        numerator_adjustment += (
            exact_relative_mass * exact_value - coarse_relative_mass * coarse_value
        )

    closed_output = tl.load(baseline_output + query_row * VALUE_DIM + value_dim).to(
        tl.float32
    )
    candidate_target = (closed_output + numerator_adjustment) / (
        1.0 + denominator_adjustment
    )
    tl.store(target_output + query_row * VALUE_DIM + value_dim, candidate_target)


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _candidate_virtual_leaf_output_utility_kernel(
    q,
    baseline_output,
    target_output,
    baseline_lse,
    candidate_coarse_lse,
    state_sum_v,
    state_counts,
    leaf_k,
    leaf_v,
    page_indices,
    page_counts,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    candidates,
    output_utility,
    query_len,
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
    CANDIDATE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
):
    """Rank a slot by its exact change to the approximate attention output."""
    query_row = tl.program_id(0).to(tl.int64)
    candidate_rank = tl.program_id(1).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head
    slot = tl.load(candidates + query_row * CANDIDATE_COUNT + candidate_rank).to(
        tl.int64
    )
    valid_slot = (slot >= 0) & (slot < STATE_CAPACITY)
    slot = tl.where(valid_slot, slot, 0)
    key_count = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + slot,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    slot_page_count = (key_count + PAGE_SIZE - 1) // PAGE_SIZE
    key_dim = tl.arange(0, HEAD_DIM)
    value_dim = tl.arange(0, VALUE_DIM)
    token_offset = tl.arange(0, PAGE_SIZE)
    query = tl.load(q + query_row * HEAD_DIM + key_dim).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_DIM,), tl.float32)
    if HASH_PROBES == 0:
        page_table = (
            slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
        )

    for page_ordinal in tl.range(0, slot_page_count, num_stages=1):
        valid_page = valid_slot & (page_ordinal < slot_page_count)
        if HASH_PROBES == 0:
            page_id = tl.load(
                page_table + page_ordinal,
                mask=valid_page,
                other=0,
            ).to(tl.int64)
        else:
            page_id = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                kv_row,
                slot,
                page_ordinal,
                valid_page,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        valid_page &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
        page_id = tl.where(valid_page, page_id, 0)
        page_count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page_id,
            mask=valid_page,
            other=0,
        ).to(tl.int32)
        valid_token = valid_page & (token_offset < page_count)
        leaf_index = tl.load(
            page_indices
            + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
            + token_offset,
            mask=valid_token,
            other=0,
        ).to(tl.int64)
        valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
        leaf_index = tl.where(valid_token, leaf_index, 0)
        key = tl.load(
            leaf_k
            + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * HEAD_DIM
            + key_dim[None, :],
            mask=valid_token[:, None],
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            leaf_v
            + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * VALUE_DIM
            + value_dim[None, :],
            mask=valid_token[:, None],
            other=0.0,
        ).to(tl.float32)
        score = SCALE * tl.sum(key * query[None, :], axis=1)
        score = tl.where(valid_token, score, -float("inf"))
        block_maximum = tl.max(score, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - new_maximum)
        probability = tl.where(valid_token, tl.exp(score - new_maximum), 0.0)
        denominator = denominator * correction + tl.sum(probability, axis=0)
        accumulator = accumulator * correction + tl.sum(
            probability[:, None] * value, axis=0
        )
        maximum = new_maximum

    exact_lse = maximum + tl.log(denominator)
    exact_value = accumulator / denominator
    closed_lse = tl.load(baseline_lse + query_row).to(tl.float32)
    coarse_lse = tl.load(
        candidate_coarse_lse + query_row * CANDIDATE_COUNT + candidate_rank
    ).to(tl.float32)
    coarse_relative_mass = tl.exp(coarse_lse - closed_lse)
    exact_relative_mass = tl.exp(exact_lse - closed_lse)
    new_denominator = 1.0 - coarse_relative_mass + exact_relative_mass
    closed_output = tl.load(baseline_output + query_row * VALUE_DIM + value_dim).to(
        tl.float32
    )
    state_count = tl.load(
        state_counts + (kv_row * STATE_CAPACITY + slot),
        mask=valid_slot,
        other=1.0,
    ).to(tl.float32)
    coarse_value = (
        tl.load(
            state_sum_v + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM + value_dim,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32)
        / state_count
    )
    opened_output = (
        closed_output
        - coarse_relative_mass * coarse_value
        + exact_relative_mass * exact_value
    ) / new_denominator
    candidate_target = tl.load(target_output + query_row * VALUE_DIM + value_dim).to(
        tl.float32
    )
    baseline_error = candidate_target - closed_output
    opened_error = candidate_target - opened_output
    utility = tl.sum(
        baseline_error * baseline_error - opened_error * opened_error,
        axis=0,
    )
    utility = tl.where(
        valid_slot & (denominator > 0.0) & (new_denominator > 0.0),
        utility,
        -float("inf"),
    )
    tl.store(
        output_utility + query_row * CANDIDATE_COUNT + candidate_rank,
        utility,
    )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _assign_page_ordinals_kernel(
    owners,
    slot_lengths,
    next_page,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    overflow_flag,
    ordinals,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    token_row = tl.program_id(0).to(tl.int64)
    token = token_row % TOKENS
    kv_row = token_row // TOKENS
    owner = tl.load(owners + token_row).to(tl.int64)
    slot_length_ptr = slot_lengths + kv_row * STATE_CAPACITY + owner
    ordinal = tl.atomic_add(slot_length_ptr, 1, sem="relaxed").to(tl.int32)
    tl.store(ordinals + token_row, ordinal)

    page_ordinal = ordinal // PAGE_SIZE
    starts_page = ordinal % PAGE_SIZE == 0
    page_id = tl.atomic_add(
        next_page + kv_row,
        1,
        sem="relaxed",
        mask=starts_page,
    ).to(tl.int32)
    inline = starts_page & (page_ordinal < INLINE_PAGES_PER_SLOT)
    tl.store(
        slot_pages
        + (kv_row * STATE_CAPACITY + owner) * INLINE_PAGES_PER_SLOT
        + page_ordinal,
        page_id,
        mask=inline,
    )
    lookup_key = (owner * 65_536 + page_ordinal).to(tl.int32)
    index = _page_hash_index(lookup_key, HASH_CAPACITY)
    active = starts_page & ~inline
    tl.atomic_xchg(overflow_used, 1, mask=active, sem="relaxed")
    for _ in tl.static_range(0, HASH_PROBES):
        old_key = tl.atomic_cas(
            overflow_page_keys + kv_row * HASH_CAPACITY + index,
            tl.where(active, -1, -2),
            lookup_key,
            sem="relaxed",
        )
        claimed = active & ((old_key == -1) | (old_key == lookup_key))
        tl.store(
            overflow_page_values + kv_row * HASH_CAPACITY + index,
            page_id,
            mask=claimed,
        )
        active &= ~claimed
        index = (index + 1) & (HASH_CAPACITY - 1)
    tl.atomic_xchg(overflow_flag, 1, mask=active, sem="relaxed")


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _publish_page_ids_kernel(
    owners,
    ordinals,
    slot_lengths,
    next_page,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    overflow_flag,
    TOKENS,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
):
    """Commit counts and publish IDs after stable ordinals are materialized."""
    kv_row = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    valid = token < TOKENS
    token_row = kv_row * TOKENS + token
    owner = tl.load(owners + token_row, mask=valid, other=0).to(tl.int64)
    ordinal = tl.load(ordinals + token_row, mask=valid, other=0).to(tl.int64)
    if MAX_LEAF_TOKENS:
        valid &= (ordinal >= 0) & (ordinal < MAX_LEAF_TOKENS)

    lane = tl.arange(0, BLOCK_TOKENS)
    same_owner = (owner[:, None] == owner[None, :]) & valid[:, None] & valid[None, :]
    earlier = lane[None, :] < lane[:, None]
    first_in_block = tl.sum((same_owner & earlier).to(tl.int32), axis=1) == 0
    block_count = tl.sum(same_owner.to(tl.int32), axis=1)
    tl.atomic_add(
        slot_lengths + kv_row * STATE_CAPACITY + owner,
        block_count,
        mask=valid & first_in_block,
        sem="relaxed",
    )

    starts_page = valid & (ordinal % PAGE_SIZE == 0)
    page_ordinal = ordinal // PAGE_SIZE
    page_rank = tl.cumsum(starts_page.to(tl.int32), axis=0) - 1
    page_count = tl.sum(starts_page.to(tl.int32), axis=0)
    first_page = tl.atomic_add(
        next_page + kv_row,
        page_count,
        mask=page_count > 0,
        sem="relaxed",
    ).to(tl.int32)
    page_id = first_page + page_rank
    if HASH_PROBES == -1:
        directory_ordinal = page_ordinal // 64
        directory_offset = page_ordinal % 64
        root_valid = starts_page & (directory_ordinal < INLINE_PAGES_PER_SLOT)
        safe_directory_ordinal = tl.where(root_valid, directory_ordinal, 0)
        root_pointer = (
            slot_pages
            + (kv_row * STATE_CAPACITY + owner) * INLINE_PAGES_PER_SLOT
            + safe_directory_ordinal
        )
        installed_directory_id = tl.atomic_cas(
            root_pointer,
            tl.where(root_valid, -1, -2),
            page_id.to(tl.int32),
            sem="relaxed",
        ).to(tl.int32)
        directory_id = tl.where(
            installed_directory_id == -1,
            page_id.to(tl.int32),
            installed_directory_id,
        )
        directory_valid = (
            root_valid & (directory_id >= 0) & (directory_id < HASH_CAPACITY)
        )
        tl.store(
            overflow_page_values
            + (kv_row * HASH_CAPACITY + directory_id) * 64
            + directory_offset,
            page_id,
            mask=directory_valid,
        )
        failed = starts_page & ~directory_valid
        ones = tl.full((BLOCK_TOKENS,), 1, tl.int32)
        tl.atomic_xchg(
            overflow_flag + token * 0,
            ones,
            mask=failed,
            sem="relaxed",
        )
    else:
        inline = starts_page & (page_ordinal < INLINE_PAGES_PER_SLOT)
        tl.store(
            slot_pages
            + (kv_row * STATE_CAPACITY + owner) * INLINE_PAGES_PER_SLOT
            + page_ordinal,
            page_id,
            mask=inline,
        )

        lookup_key = (owner * 65_536 + page_ordinal).to(tl.int32)
        index = _page_hash_index(lookup_key, HASH_CAPACITY)
        active = starts_page & ~inline
        ones = tl.full((BLOCK_TOKENS,), 1, tl.int32)
        tl.atomic_xchg(
            overflow_used + token * 0,
            ones,
            mask=active,
            sem="relaxed",
        )
        for _ in tl.static_range(0, HASH_PROBES):
            old_key = tl.atomic_cas(
                overflow_page_keys + kv_row * HASH_CAPACITY + index,
                tl.where(active, -1, -2),
                lookup_key,
                sem="relaxed",
            )
            claimed = active & ((old_key == -1) | (old_key == lookup_key))
            tl.store(
                overflow_page_values + kv_row * HASH_CAPACITY + index,
                page_id,
                mask=claimed,
            )
            active &= ~claimed
            index = (index + 1) & (HASH_CAPACITY - 1)
        tl.atomic_xchg(
            overflow_flag + token * 0,
            ones,
            mask=active,
            sem="relaxed",
        )


def _assign_page_ordinals(
    owners: torch.Tensor,
    slot_lengths: torch.Tensor,
    next_page: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    overflow_flag: torch.Tensor,
    *,
    hash_probes: int,
    page_size: int,
    max_leaf_tokens: int | None = None,
) -> torch.Tensor:
    """Assign stable region-local ordinals and publish new logical pages."""
    batch, kv_heads, tokens = owners.shape
    # Recursive LOD pages are semantic units, so their membership must not
    # depend on the order in which GPU programs happen to reserve slot ranges.
    # Sorting the unique (owner, sequence-position) pair groups equal owners
    # while retaining chronological order inside each group. This produces the
    # exact same ranks as counting all prior equal owners, without its O(T^2)
    # scan. Page IDs may be reserved in any order: the semantic identity is
    # (owner, page ordinal), not the numeric page ID.
    positions = torch.arange(tokens, device=owners.device, dtype=owners.dtype).view(
        1, 1, tokens
    )
    order = torch.argsort(owners * tokens + positions, dim=2)
    sorted_owners = owners.gather(2, order)
    new_group = torch.ones_like(sorted_owners, dtype=torch.bool)
    new_group[..., 1:] = sorted_owners[..., 1:] != sorted_owners[..., :-1]
    sorted_positions = positions.expand(batch, kv_heads, tokens)
    group_starts = torch.where(new_group, sorted_positions, 0).cummax(dim=2).values
    sorted_ranks = sorted_positions - group_starts
    ranks = torch.empty_like(owners)
    ranks.scatter_(2, order, sorted_ranks)
    ordinals = (slot_lengths.gather(2, owners).to(owners.dtype) + ranks).to(torch.int32)
    if max_leaf_tokens is not None:
        if max_leaf_tokens <= 0:
            raise ValueError("maximum archived leaves must be positive")
        ordinals.masked_fill_(ordinals >= max_leaf_tokens, -1)
    block_tokens = 16
    _publish_page_ids_kernel[(batch * kv_heads, triton.cdiv(tokens, block_tokens))](
        owners,
        ordinals,
        slot_lengths,
        next_page,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        TOKENS=tokens,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=page_size,
        BLOCK_TOKENS=block_tokens,
        MAX_LEAF_TOKENS=max_leaf_tokens or 0,
        num_warps=1,
    )
    return ordinals


@triton.jit(do_not_specialize=["source_slot", "destination_slot"])
def _rehash_overflow_pages_kernel(
    source_keys,
    source_values,
    destination_keys,
    destination_values,
    destination_used,
    destination_flag,
    source_slot,
    destination_slot,
    SOURCE_BATCH_STRIDE: tl.constexpr,
    SOURCE_HEAD_STRIDE: tl.constexpr,
    DESTINATION_BATCH_STRIDE: tl.constexpr,
    DESTINATION_HEAD_STRIDE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SOURCE_CAPACITY: tl.constexpr,
    DESTINATION_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
):
    entry = tl.program_id(0).to(tl.int64)
    head = entry // SOURCE_CAPACITY
    source_bucket = entry - head * SOURCE_CAPACITY
    source_offset = (
        source_slot * SOURCE_BATCH_STRIDE + head * SOURCE_HEAD_STRIDE + source_bucket
    )
    key = tl.load(source_keys + source_offset).to(tl.int32)
    value = tl.load(source_values + source_offset).to(tl.int32)
    active = (head < KV_HEADS) & (key >= 0)
    index = _page_hash_index(key, DESTINATION_CAPACITY)
    destination_base = (
        destination_slot * DESTINATION_BATCH_STRIDE + head * DESTINATION_HEAD_STRIDE
    )
    tl.atomic_xchg(destination_used, 1, mask=active, sem="relaxed")
    for _ in tl.static_range(0, HASH_PROBES):
        old_key = tl.atomic_cas(
            destination_keys + destination_base + index,
            tl.where(active, -1, -2),
            key,
            sem="relaxed",
        )
        claimed = active & ((old_key == -1) | (old_key == key))
        tl.store(
            destination_values + destination_base + index,
            value,
            mask=claimed,
        )
        active &= ~claimed
        index = (index + 1) & (DESTINATION_CAPACITY - 1)
    tl.atomic_xchg(destination_flag, 1, mask=active, sem="relaxed")


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _write_paged_kv_kernel(
    k,
    v,
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    overflow_flag,
    page_k,
    page_v,
    K_BATCH_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    V_BATCH_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    token_row = tl.program_id(0).to(tl.int64)
    token = token_row % TOKENS
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    archived = ordinal >= 0
    page_ordinal = ordinal // PAGE_SIZE
    within_page = ordinal % PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        archived,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    valid_page = (page_id >= 0) & (page_id < PAGE_CAPACITY)
    invalid_page = archived & ~valid_page
    tl.atomic_xchg(overflow_flag, 1, mask=invalid_page, sem="relaxed")
    archived &= valid_page

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    source_k = (
        k
        + batch * K_BATCH_STRIDE
        + kv_head * K_HEAD_STRIDE
        + token * K_TOKEN_STRIDE
        + head_offset
    )
    source_v = (
        v
        + batch * V_BATCH_STRIDE
        + kv_head * V_HEAD_STRIDE
        + token * V_TOKEN_STRIDE
        + value_offset
    )
    physical_token = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
    tl.store(
        page_k + physical_token * HEAD_DIM + head_offset,
        tl.load(source_k, mask=head_offset < HEAD_DIM, other=0.0),
        mask=archived & (head_offset < HEAD_DIM),
    )
    tl.store(
        page_v + physical_token * VALUE_DIM + value_offset,
        tl.load(source_v, mask=value_offset < VALUE_DIM, other=0.0),
        mask=archived & (value_offset < VALUE_DIM),
    )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _write_paged_int8_kv_kernel(
    k,
    v,
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    overflow_flag,
    page_k,
    page_v,
    page_k_scales,
    page_v_scales,
    K_BATCH_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    V_BATCH_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    """Quantize each archived token once for matrix-native INT8 prefill."""
    token_row = tl.program_id(0).to(tl.int64)
    token = token_row % TOKENS
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    archived = ordinal >= 0
    page_ordinal = ordinal // PAGE_SIZE
    within_page = ordinal % PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        archived,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    valid_page = (page_id >= 0) & (page_id < PAGE_CAPACITY)
    invalid_page = archived & ~valid_page
    tl.atomic_xchg(overflow_flag, 1, mask=invalid_page, sem="relaxed")
    archived &= valid_page

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    source_k = (
        k
        + batch * K_BATCH_STRIDE
        + kv_head * K_HEAD_STRIDE
        + token * K_TOKEN_STRIDE
        + head_offset
    )
    source_v = (
        v
        + batch * V_BATCH_STRIDE
        + kv_head * V_HEAD_STRIDE
        + token * V_TOKEN_STRIDE
        + value_offset
    )
    key = tl.load(source_k, mask=archived & (head_offset < HEAD_DIM), other=0.0).to(
        tl.float32
    )
    value = tl.load(source_v, mask=archived & (value_offset < VALUE_DIM), other=0.0).to(
        tl.float32
    )
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1.1754943508222875e-38)
    value_scale = tl.maximum(
        tl.max(tl.abs(value), axis=0) / 127.0, 1.1754943508222875e-38
    )
    key_code = tl.maximum(
        tl.minimum(tl.floor(key / key_scale + 0.5), 127.0), -127.0
    ).to(tl.int8)
    value_code = tl.maximum(
        tl.minimum(tl.floor(value / value_scale + 0.5), 127.0), -127.0
    ).to(tl.int8)
    physical_token = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
    tl.store(
        page_k + physical_token * HEAD_DIM + head_offset,
        key_code,
        mask=archived & (head_offset < HEAD_DIM),
    )
    tl.store(
        page_v + physical_token * VALUE_DIM + value_offset,
        value_code,
        mask=archived & (value_offset < VALUE_DIM),
    )
    tl.store(page_k_scales + physical_token, key_scale, mask=archived)
    tl.store(page_v_scales + physical_token, value_scale, mask=archived)


@triton.jit(
    do_not_specialize=["LEAF_OFFSET", "TOKENS"],
    do_not_specialize_on_alignment=["LEAF_OFFSET", "TOKENS"],
)
def _write_virtual_int8_kv_kernel(
    k,
    v,
    owners,
    leaf_k,
    leaf_v,
    leaf_k_scales,
    leaf_v_scales,
    K_BATCH_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    V_BATCH_STRIDE: tl.constexpr,
    V_HEAD_STRIDE: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    LEAF_OFFSET,
    TOKENS,
    KV_HEADS: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    """Quantize chronological virtual leaves using one scale per token."""
    token_row = tl.program_id(0).to(tl.int64)
    token = token_row % TOKENS
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    archived = tl.load(owners + token_row) >= 0
    storage_token = kv_row * LEAF_CAPACITY + LEAF_OFFSET + token
    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    key = tl.load(
        k
        + batch * K_BATCH_STRIDE
        + kv_head * K_HEAD_STRIDE
        + token * K_TOKEN_STRIDE
        + head_offset,
        mask=archived & (head_offset < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        v
        + batch * V_BATCH_STRIDE
        + kv_head * V_HEAD_STRIDE
        + token * V_TOKEN_STRIDE
        + value_offset,
        mask=archived & (value_offset < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1.1754943508222875e-38)
    value_scale = tl.maximum(
        tl.max(tl.abs(value), axis=0) / 127.0, 1.1754943508222875e-38
    )
    key_code = tl.maximum(
        tl.minimum(tl.floor(key / key_scale + 0.5), 127.0), -127.0
    ).to(tl.int8)
    value_code = tl.maximum(
        tl.minimum(tl.floor(value / value_scale + 0.5), 127.0), -127.0
    ).to(tl.int8)
    tl.store(
        leaf_k + storage_token * HEAD_DIM + head_offset,
        key_code,
        mask=archived & (head_offset < HEAD_DIM),
    )
    tl.store(
        leaf_v + storage_token * VALUE_DIM + value_offset,
        value_code,
        mask=archived & (value_offset < VALUE_DIM),
    )
    tl.store(leaf_k_scales + storage_token, key_scale, mask=archived)
    tl.store(leaf_v_scales + storage_token, value_scale, mask=archived)


@triton.jit
def _prepare_int8_attention_queries_kernel(
    q,
    q_codes,
    q_scales,
    top_slots,
    expert_ids,
    ROWS,
    QUERY_LEN: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offset = tl.arange(0, HEAD_BLOCK_DIM)
    valid_row = row < ROWS
    valid = valid_row[:, None] & (offset[None, :] < HEAD_DIM)
    values = tl.load(
        q + row[:, None] * HEAD_DIM + offset[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=1) / 127.0, 1.1754943508222875e-38)
    codes = tl.maximum(
        tl.minimum(tl.floor(values / scale[:, None] + 0.5), 127.0), -127.0
    ).to(tl.int8)
    tl.store(
        q_codes + row[:, None] * HEAD_DIM + offset[None, :],
        codes,
        mask=valid,
    )
    tl.store(q_scales + row, scale, mask=valid_row)
    batch_head = row // QUERY_LEN
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_row = batch * KV_HEADS + query_head // KV_GROUP_SIZE
    route = tl.arange(0, ROUTE_BLOCK)
    valid_route = valid_row[:, None] & (route[None, :] < ROUTE_COUNT)
    slot = tl.load(
        top_slots + row[:, None] * ROUTE_COUNT + route[None, :],
        mask=valid_route,
        other=0,
    ).to(tl.int32)
    tl.store(
        expert_ids + row[:, None] * ROUTE_COUNT + route[None, :],
        kv_row[:, None].to(tl.int32) * STATE_CAPACITY + slot,
        mask=valid_route,
    )


@triton.jit
def _prepare_tiny_expert_sort_keys_kernel(
    top_slots,
    slot_lengths,
    sort_keys,
    ROWS,
    QUERY_LEN: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    EXPERT_CAPACITY: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    TINY_EXPERT_MAX: tl.constexpr,
    LONG_EXPERT_THRESHOLD: tl.constexpr,
    SPLIT_LONG_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Build one compound ``(leaf-count bucket, expert)`` routing key.

    The final optional bucket isolates very long posting lists so they can use
    split-N execution without allocating partials for ordinary experts.
    """
    row = (
        tl.program_id(0).to(tl.int64) * BLOCK_M
        + tl.arange(0, BLOCK_M).to(tl.int64)
    )
    valid_row = row < ROWS
    batch_head = row // QUERY_LEN
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_row = batch * KV_HEADS + query_head // KV_GROUP_SIZE
    route = tl.arange(0, ROUTE_BLOCK)
    valid_route = valid_row[:, None] & (route[None, :] < ROUTE_COUNT)
    slot = tl.load(
        top_slots + row[:, None] * ROUTE_COUNT + route[None, :],
        mask=valid_route,
        other=0,
    ).to(tl.int32)
    expert = kv_row[:, None].to(tl.int32) * STATE_CAPACITY + slot
    leaf_count = tl.load(
        slot_lengths + expert,
        mask=valid_route,
        other=TINY_EXPERT_MAX + 1,
    ).to(tl.int32)
    bucket = tl.minimum(leaf_count, TINY_EXPERT_MAX + 1) - 1
    if SPLIT_LONG_EXPERTS:
        bucket = tl.where(
            leaf_count > LONG_EXPERT_THRESHOLD,
            TINY_EXPERT_MAX + 1,
            bucket,
        )
    sort_key = bucket * EXPERT_CAPACITY + expert
    tl.store(
        sort_keys + row[:, None] * ROUTE_COUNT + route[None, :],
        sort_key,
        mask=valid_route,
    )


@triton.jit(
    do_not_specialize=["EXPERTS"],
    do_not_specialize_on_alignment=["EXPERTS"],
)
def _prepare_tiny_expert_metadata_kernel(
    unique_sort_key,
    q_lengths,
    expert_kv_row,
    expert_slot,
    expert_blocks,
    bucket_block_counts,
    EXPERTS,
    EXPERT_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    TINY_EXPERT_MAX: tl.constexpr,
    BUCKET_COUNT: tl.constexpr,
    TINY_BLOCK_M: tl.constexpr,
    GENERAL_BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Recover expert metadata and count launch blocks in one pass."""
    expert_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = expert_offset < EXPERTS
    sort_key = tl.load(unique_sort_key + expert_offset, mask=valid, other=0).to(
        tl.int32
    )
    bucket = sort_key // EXPERT_CAPACITY
    expert = sort_key - bucket * EXPERT_CAPACITY
    kv_row = expert // STATE_CAPACITY
    slot = expert - kv_row * STATE_CAPACITY
    query_count = tl.load(q_lengths + expert_offset, mask=valid, other=0).to(
        tl.int64
    )
    block_m = tl.where(bucket < TINY_EXPERT_MAX, TINY_BLOCK_M, GENERAL_BLOCK_M)
    blocks = (query_count + block_m - 1) // block_m
    tl.store(expert_kv_row + expert_offset, kv_row, mask=valid)
    tl.store(expert_slot + expert_offset, slot, mask=valid)
    tl.store(expert_blocks + expert_offset, blocks, mask=valid)
    for bucket_index in tl.static_range(0, BUCKET_COUNT):
        bucket_blocks = tl.sum(
            tl.where(valid & (bucket == bucket_index), blocks, 0), axis=0
        )
        tl.atomic_add(bucket_block_counts + bucket_index, bucket_blocks)


@triton.jit(
    do_not_specialize=["PROGRAM_OFFSET"],
    do_not_specialize_on_alignment=["PROGRAM_OFFSET"],
)
def _tiny_leaf_expert_attention_kernel(
    q,
    packed_route_row,
    block_expert,
    block_starts,
    q_lengths,
    cu_q,
    expert_kv_row,
    expert_slot,
    leaf_k,
    leaf_v,
    page_indices,
    slot_pages,
    out,
    lse,
    PROGRAM_OFFSET,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    KEY_COUNT: tl.constexpr,
    KEY_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Exact BF16 attention for a small, compile-time leaf count."""
    program = tl.program_id(0).to(tl.int64) + PROGRAM_OFFSET
    expert = tl.load(block_expert + program)
    query_block = program - tl.load(block_starts + expert)
    query_count = tl.load(q_lengths + expert)
    query_offset = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_query = query_offset < query_count
    packed_begin = tl.load(cu_q + expert).to(tl.int64)
    packed_row = packed_begin + query_offset.to(tl.int64)
    route_row = tl.load(
        packed_route_row + packed_row,
        mask=valid_query,
        other=0,
    ).to(tl.int64)
    query_row = route_row // ROUTE_COUNT

    kv_row = tl.load(expert_kv_row + expert).to(tl.int64)
    slot = tl.load(expert_slot + expert).to(tl.int64)
    page_id = tl.load(
        slot_pages
        + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
    ).to(tl.int64)
    physical_page = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE

    head_offset = tl.arange(0, HEAD_DIM)
    value_offset = tl.arange(0, VALUE_DIM)
    q_block = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=valid_query[:, None],
        other=0.0,
    ).to(tl.float32)

    score_columns = tl.full((BLOCK_M, KEY_BLOCK), -float("inf"), tl.float32)
    key_column = tl.arange(0, KEY_BLOCK)
    for key_index in tl.static_range(0, KEY_COUNT):
        leaf_index = tl.load(page_indices + physical_page + key_index).to(tl.int64)
        storage_token = kv_row * LEAF_CAPACITY + leaf_index
        key = tl.load(
            leaf_k + storage_token * HEAD_DIM + head_offset
        ).to(tl.float32)
        score = tl.sum(q_block * key[None, :], axis=1) * SCALE_LOG2
        score_columns = tl.where(
            key_column[None, :] == key_index,
            score[:, None],
            score_columns,
        )

    maximum = tl.max(score_columns, axis=1)
    probability = tl.math.exp2(score_columns - maximum[:, None])
    denominator = tl.sum(probability, axis=1)
    probability /= denominator[:, None]
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    for key_index in tl.static_range(0, KEY_COUNT):
        leaf_index = tl.load(page_indices + physical_page + key_index).to(tl.int64)
        storage_token = kv_row * LEAF_CAPACITY + leaf_index
        value = tl.load(
            leaf_v + storage_token * VALUE_DIM + value_offset
        ).to(tl.float32)
        key_probability = tl.sum(
            tl.where(
                key_column[None, :] == key_index,
                probability,
                0.0,
            ),
            axis=1,
        )
        accumulator += key_probability[:, None] * value[None, :]

    natural_lse = (
        maximum + tl.math.log2(denominator)
    ) * 0.6931471805599453
    tl.store(
        out + route_row[:, None] * VALUE_DIM + value_offset[None, :],
        accumulator,
        mask=valid_query[:, None],
    )
    tl.store(lse + route_row, natural_lse, mask=valid_query)


@triton.jit
def _reduce_expert_route_attention_kernel(
    route_out,
    route_lse,
    exact_out,
    exact_lse,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    """Merge routed expert outputs with their exact log-sum-exp masses."""
    row = tl.program_id(0).to(tl.int64)
    route = tl.arange(0, ROUTE_BLOCK)
    dimension = tl.arange(0, VALUE_BLOCK_DIM)
    valid_route = route < ROUTE_COUNT
    lse = tl.load(
        route_lse + row * ROUTE_COUNT + route,
        mask=valid_route,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(lse, axis=0)
    weight = tl.where(valid_route, tl.exp(lse - maximum), 0.0)
    denominator = tl.sum(weight, axis=0)
    values = tl.load(
        route_out
        + (row * ROUTE_COUNT + route[:, None]) * VALUE_DIM
        + dimension[None, :],
        mask=valid_route[:, None] & (dimension[None, :] < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    output = tl.sum(values * weight[:, None], axis=0) / denominator
    tl.store(
        exact_out + row * VALUE_DIM + dimension,
        output,
        mask=dimension < VALUE_DIM,
    )
    tl.store(exact_lse + row, maximum + tl.log(denominator))


@triton.jit
def _pack_aiter_expert_queries_kernel(
    q,
    order,
    packed_q,
    ROUTE_ROWS,
    ROUTE_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Gather the unchanged expert-major route order into AITER's Q layout."""
    packed_row = (
        tl.program_id(0).to(tl.int64) * BLOCK_M
        + tl.arange(0, BLOCK_M).to(tl.int64)
    )
    dimension = tl.arange(0, HEAD_BLOCK_DIM)
    valid_row = packed_row < ROUTE_ROWS
    route_row = tl.load(order + packed_row, mask=valid_row, other=0).to(tl.int64)
    query_row = route_row // ROUTE_COUNT
    values = tl.load(
        q + query_row[:, None] * HEAD_DIM + dimension[None, :],
        mask=valid_row[:, None] & (dimension[None, :] < HEAD_DIM),
        other=0.0,
    )
    tl.store(
        packed_q + packed_row[:, None] * HEAD_DIM + dimension[None, :],
        values,
        mask=valid_row[:, None] & (dimension[None, :] < HEAD_DIM),
    )


@triton.jit
def _scatter_aiter_expert_routes_kernel(
    packed_out,
    packed_lse,
    order,
    route_out,
    route_lse,
    ROUTE_ROWS,
    VALUE_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Restore AITER's expert-major results to the existing top-8 layout."""
    packed_row = (
        tl.program_id(0).to(tl.int64) * BLOCK_M
        + tl.arange(0, BLOCK_M).to(tl.int64)
    )
    dimension = tl.arange(0, VALUE_BLOCK_DIM)
    valid_row = packed_row < ROUTE_ROWS
    route_row = tl.load(order + packed_row, mask=valid_row, other=0).to(tl.int64)
    values = tl.load(
        packed_out + packed_row[:, None] * VALUE_DIM + dimension[None, :],
        mask=valid_row[:, None] & (dimension[None, :] < VALUE_DIM),
        other=0.0,
    )
    tl.store(
        route_out + route_row[:, None] * VALUE_DIM + dimension[None, :],
        values,
        mask=valid_row[:, None] & (dimension[None, :] < VALUE_DIM),
    )
    lse = tl.load(packed_lse + packed_row, mask=valid_row, other=-float("inf"))
    tl.store(route_lse + route_row, lse, mask=valid_row)


@triton.jit(
    do_not_specialize=["LEAF_OFFSET", "TOKENS"],
    do_not_specialize_on_alignment=["LEAF_OFFSET", "TOKENS"],
)
def _write_virtual_page_indices_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    page_indices,
    LEAF_OFFSET,
    TOKENS,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Map a logical owner page to leaves in the original sequence cache."""
    token_row = tl.program_id(0).to(tl.int64)
    token = token_row % TOKENS
    kv_row = token_row // TOKENS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    page_ordinal = ordinal // PAGE_SIZE
    within_page = ordinal % PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        True,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    physical_token = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
    tl.store(page_indices + physical_token, LEAF_OFFSET + token)


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _update_page_summaries_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    page_k,
    page_v,
    page_indices,
    leaf_k,
    leaf_v,
    leaf_k_scales,
    leaf_v_scales,
    page_sum_k,
    page_sum_v,
    page_counts,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LEAF_K_BATCH_STRIDE,
    LEAF_K_HEAD_STRIDE,
    LEAF_K_TOKEN_STRIDE: tl.constexpr,
    LEAF_V_BATCH_STRIDE,
    LEAF_V_HEAD_STRIDE,
    LEAF_V_TOKEN_STRIDE: tl.constexpr,
    INDEXED: tl.constexpr,
    INT8_STORAGE: tl.constexpr,
    UPDATE_KEY: tl.constexpr,
):
    """Refresh every completed page and each slot's current partial page."""
    token_row = tl.program_id(0).to(tl.int64)
    dimension_block = tl.program_id(1)
    kv_row = token_row // TOKENS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    slot_length = tl.load(slot_lengths + kv_row * STATE_CAPACITY + owner).to(tl.int64)
    completes_page = ordinal % PAGE_SIZE == PAGE_SIZE - 1
    is_partial_tail = ordinal == slot_length - 1
    refresh = (ordinal >= 0) & (completes_page | is_partial_tail)
    page_ordinal = ordinal // PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        refresh,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    page_count = tl.where(completes_page, PAGE_SIZE, ordinal % PAGE_SIZE + 1)
    page_offset = tl.arange(0, PAGE_SIZE)
    dimension = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_page = refresh & (page_offset < page_count)

    if INDEXED:
        batch = kv_row // KV_HEADS
        kv_head = kv_row - batch * KV_HEADS
        leaf_index = tl.load(
            page_indices + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + page_offset,
            mask=valid_page,
            other=0,
        ).to(tl.int64)
    if UPDATE_KEY:
        key_valid = valid_page[:, None] & (dimension[None, :] < HEAD_DIM)
        if INDEXED:
            keys = tl.load(
                leaf_k
                + batch * LEAF_K_BATCH_STRIDE
                + kv_head * LEAF_K_HEAD_STRIDE
                + leaf_index[:, None] * LEAF_K_TOKEN_STRIDE
                + dimension[None, :],
                mask=key_valid,
                other=0.0,
            ).to(tl.float32)
            if INT8_STORAGE:
                key_scales = tl.load(
                    leaf_k_scales
                    + batch * LEAF_K_BATCH_STRIDE // LEAF_K_TOKEN_STRIDE
                    + kv_head * LEAF_K_HEAD_STRIDE // LEAF_K_TOKEN_STRIDE
                    + leaf_index,
                    mask=valid_page,
                    other=0.0,
                ).to(tl.float32)
                keys *= key_scales[:, None]
        else:
            keys = tl.load(
                page_k
                + (
                    (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
                    + page_offset[:, None]
                )
                * HEAD_DIM
                + dimension[None, :],
                mask=key_valid,
                other=0.0,
            ).to(tl.float32)
        key_sum = tl.sum(keys, axis=0)
        tl.store(
            page_sum_k + (kv_row * PAGE_CAPACITY + page_id) * HEAD_DIM + dimension,
            key_sum,
            mask=refresh & (dimension < HEAD_DIM),
        )

    value_valid = valid_page[:, None] & (dimension[None, :] < VALUE_DIM)
    if INDEXED:
        values = tl.load(
            leaf_v
            + batch * LEAF_V_BATCH_STRIDE
            + kv_head * LEAF_V_HEAD_STRIDE
            + leaf_index[:, None] * LEAF_V_TOKEN_STRIDE
            + dimension[None, :],
            mask=value_valid,
            other=0.0,
        ).to(tl.float32)
        if INT8_STORAGE:
            value_scales = tl.load(
                leaf_v_scales
                + batch * LEAF_V_BATCH_STRIDE // LEAF_V_TOKEN_STRIDE
                + kv_head * LEAF_V_HEAD_STRIDE // LEAF_V_TOKEN_STRIDE
                + leaf_index,
                mask=valid_page,
                other=0.0,
            ).to(tl.float32)
            values *= value_scales[:, None]
    else:
        values = tl.load(
            page_v
            + ((kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + page_offset[:, None])
            * VALUE_DIM
            + dimension[None, :],
            mask=value_valid,
            other=0.0,
        ).to(tl.float32)
    value_sum = tl.sum(values, axis=0)
    tl.store(
        page_sum_v + (kv_row * PAGE_CAPACITY + page_id) * VALUE_DIM + dimension,
        value_sum,
        mask=refresh & (dimension < VALUE_DIM),
    )
    tl.store(
        page_counts + kv_row * PAGE_CAPACITY + page_id,
        page_count,
        mask=refresh & (dimension_block == 0),
    )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _update_raw_page_key_summaries_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    append_k,
    page_sum_k,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_BATCH_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
):
    """Accumulate one append's raw MLA keys directly into its pages."""
    token_row = tl.program_id(0).to(tl.int64)
    dimension_block = tl.program_id(1)
    kv_row = token_row // TOKENS
    token = token_row - kv_row * TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    archived = ordinal >= 0
    page_ordinal = ordinal // PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        archived,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    dimension = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_key = (
        archived & (page_id >= 0) & (page_id < PAGE_CAPACITY) & (dimension < HEAD_DIM)
    )
    append_key = tl.load(
        append_k
        + batch * K_BATCH_STRIDE
        + kv_head * K_HEAD_STRIDE
        + token * K_TOKEN_STRIDE
        + dimension,
        mask=valid_key,
        other=0.0,
    )
    destination = page_sum_k + (kv_row * PAGE_CAPACITY + page_id) * HEAD_DIM + dimension
    tl.atomic_add(
        destination,
        append_key,
        mask=valid_key,
        sem="relaxed",
    )


@triton.jit
def _quantize_virtual_page_tensor(
    source,
    page_sum,
    destination,
    scales,
    leaf_index,
    valid_token,
    refresh,
    page_count,
    batch,
    kv_head,
    kv_row,
    page_id,
    group,
    pair_offset,
    even_dimension,
    odd_dimension,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIMENSION_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    SOURCE_BATCH_STRIDE: tl.constexpr,
    SOURCE_HEAD_STRIDE: tl.constexpr,
    SOURCE_TOKEN_STRIDE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    OPTIMIZE_SCALE: tl.constexpr,
):
    valid_even = valid_token[:, None] & (even_dimension[None, :] < DIMENSION_SIZE)
    valid_odd = valid_token[:, None] & (odd_dimension[None, :] < DIMENSION_SIZE)
    source_base = (
        source
        + batch * SOURCE_BATCH_STRIDE
        + kv_head * SOURCE_HEAD_STRIDE
        + leaf_index[:, None] * SOURCE_TOKEN_STRIDE
    )
    even = tl.load(
        source_base + even_dimension[None, :],
        mask=valid_even,
        other=0.0,
    ).to(tl.float32)
    odd = tl.load(
        source_base + odd_dimension[None, :],
        mask=valid_odd,
        other=0.0,
    ).to(tl.float32)
    sum_base = page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    inverse_count = 1.0 / tl.maximum(page_count.to(tl.float32), 1.0)
    even_anchor = (
        tl.load(
            sum_base + even_dimension,
            mask=refresh & (even_dimension < DIMENSION_SIZE),
            other=0.0,
        ).to(tl.float32)
        * inverse_count
    )
    odd_anchor = (
        tl.load(
            sum_base + odd_dimension,
            mask=refresh & (odd_dimension < DIMENSION_SIZE),
            other=0.0,
        ).to(tl.float32)
        * inverse_count
    )
    even_residual = even - even_anchor[None, :]
    odd_residual = odd - odd_anchor[None, :]
    even_max = tl.max(
        tl.max(tl.where(valid_even, tl.abs(even_residual), 0.0), axis=1),
        axis=0,
    )
    odd_max = tl.max(
        tl.max(tl.where(valid_odd, tl.abs(odd_residual), 0.0), axis=1),
        axis=0,
    )
    quant_max: tl.constexpr = (1 << (QUANT_BITS - 1)) - 1
    scale = tl.maximum(tl.maximum(even_max, odd_max) / quant_max, 1.0e-8)
    even_code_float = tl.maximum(
        tl.minimum(tl.floor(even_residual / scale + 0.5), quant_max),
        -quant_max,
    )
    odd_code_float = tl.maximum(
        tl.minimum(tl.floor(odd_residual / scale + 0.5), quant_max),
        -quant_max,
    )
    if OPTIMIZE_SCALE:
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
        even_code_float = tl.maximum(
            tl.minimum(tl.floor(even_residual / scale + 0.5), quant_max),
            -quant_max,
        )
        odd_code_float = tl.maximum(
            tl.minimum(tl.floor(odd_residual / scale + 0.5), quant_max),
            -quant_max,
        )
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
    if QUANT_BITS == 4:
        even_code = even_code_float.to(tl.int32) + 8
        odd_code = odd_code_float.to(tl.int32) + 8
        packed = (even_code | (odd_code << 4)).to(tl.uint8)
        packed_dimension = DIMENSION_SIZE // 2
        destination_base = (
            destination
            + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * packed_dimension
            + group * (GROUP_SIZE // 2)
            + pair_offset[None, :]
        )
        tl.store(destination_base, packed, mask=valid_even)
    else:
        destination_base = (
            destination
            + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * DIMENSION_SIZE
        )
        tl.store(
            destination_base + even_dimension[None, :],
            even_code_float.to(tl.int8),
            mask=valid_even,
        )
        tl.store(
            destination_base + odd_dimension[None, :],
            odd_code_float.to(tl.int8),
            mask=valid_odd,
        )
    tl.store(
        scales
        + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // GROUP_SIZE)
        + group,
        scale,
        mask=refresh,
    )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _quantize_touched_virtual_pages_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    page_indices,
    leaf_k,
    leaf_v,
    page_sum_k,
    page_sum_v,
    page_counts,
    quantized_leaf_k,
    quantized_leaf_v,
    page_k_scales,
    page_v_scales,
    page_quantized_counts,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    LEAF_K_BATCH_STRIDE: tl.constexpr,
    LEAF_K_HEAD_STRIDE: tl.constexpr,
    LEAF_K_TOKEN_STRIDE: tl.constexpr,
    LEAF_V_BATCH_STRIDE: tl.constexpr,
    LEAF_V_HEAD_STRIDE: tl.constexpr,
    LEAF_V_TOKEN_STRIDE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    OPTIMIZE_SCALE: tl.constexpr,
):
    """Requantize each page changed by this append against its current mean."""
    token_row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    slot_length = tl.load(slot_lengths + kv_row * STATE_CAPACITY + owner).to(tl.int64)
    completes_page = ordinal % PAGE_SIZE == PAGE_SIZE - 1
    is_partial_tail = ordinal == slot_length - 1
    refresh = completes_page | is_partial_tail
    page_ordinal = ordinal // PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        refresh,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    page_count = tl.load(
        page_counts + kv_row * PAGE_CAPACITY + page_id,
        mask=refresh,
        other=0,
    ).to(tl.int32)
    token_offset = tl.arange(0, PAGE_SIZE)
    pair_offset = tl.arange(0, GROUP_SIZE // 2)
    even_dimension = group * GROUP_SIZE + pair_offset * 2
    odd_dimension = even_dimension + 1
    valid_token = refresh & (token_offset < page_count)
    leaf_index = tl.load(
        page_indices + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)

    _quantize_virtual_page_tensor(
        leaf_k,
        page_sum_k,
        quantized_leaf_k,
        page_k_scales,
        leaf_index,
        valid_token,
        refresh,
        page_count,
        batch,
        kv_head,
        kv_row,
        page_id,
        group,
        pair_offset,
        even_dimension,
        odd_dimension,
        PAGE_CAPACITY,
        PAGE_SIZE,
        HEAD_DIM,
        GROUP_SIZE,
        LEAF_CAPACITY,
        LEAF_K_BATCH_STRIDE,
        LEAF_K_HEAD_STRIDE,
        LEAF_K_TOKEN_STRIDE,
        QUANT_BITS,
        OPTIMIZE_SCALE,
    )
    _quantize_virtual_page_tensor(
        leaf_v,
        page_sum_v,
        quantized_leaf_v,
        page_v_scales,
        leaf_index,
        valid_token,
        refresh,
        page_count,
        batch,
        kv_head,
        kv_row,
        page_id,
        group,
        pair_offset,
        even_dimension,
        odd_dimension,
        PAGE_CAPACITY,
        PAGE_SIZE,
        VALUE_DIM,
        GROUP_SIZE,
        LEAF_CAPACITY,
        LEAF_V_BATCH_STRIDE,
        LEAF_V_HEAD_STRIDE,
        LEAF_V_TOKEN_STRIDE,
        QUANT_BITS,
        OPTIMIZE_SCALE,
    )
    tl.store(
        page_quantized_counts + kv_row * PAGE_CAPACITY + page_id,
        page_count,
        mask=refresh & (group == 0),
    )


@triton.jit
def _fake_quantize_virtual_page_tensor(
    source,
    page_sum,
    leaf_index,
    valid_token,
    refresh,
    page_count,
    batch,
    kv_head,
    kv_row,
    page_id,
    group,
    dimension,
    PAGE_CAPACITY: tl.constexpr,
    DIMENSION_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    SOURCE_BATCH_STRIDE: tl.constexpr,
    SOURCE_HEAD_STRIDE: tl.constexpr,
    SOURCE_TOKEN_STRIDE: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    OPTIMIZE_LEAF_SCALE: tl.constexpr,
    QUANTIZE_SUMMARY: tl.constexpr,
    OPTIMIZE_SUMMARY_SCALE: tl.constexpr,
):
    """Round one completed virtual page through groupwise symmetric QDQ."""
    valid_dimension = dimension < DIMENSION_SIZE
    valid = valid_token[:, None] & valid_dimension[None, :]
    source_base = (
        source
        + batch * SOURCE_BATCH_STRIDE
        + kv_head * SOURCE_HEAD_STRIDE
        + leaf_index[:, None] * SOURCE_TOKEN_STRIDE
    )
    values = tl.load(
        source_base + dimension[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    sum_base = page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    inverse_count = 1.0 / tl.maximum(page_count.to(tl.float32), 1.0)
    sums = tl.load(
        sum_base + dimension,
        mask=refresh & valid_dimension,
        other=0.0,
    ).to(tl.float32)
    anchor = sums * inverse_count
    residual = values - anchor[None, :]
    maximum = tl.max(tl.max(tl.where(valid, tl.abs(residual), 0.0), axis=1), axis=0)
    scale = tl.maximum(maximum / QUANT_MAX, 1.0e-8)
    codes = tl.maximum(
        tl.minimum(tl.floor(residual / scale + 0.5), QUANT_MAX), -QUANT_MAX
    )
    if OPTIMIZE_LEAF_SCALE:
        denominator = tl.sum(
            tl.sum(tl.where(valid, codes * codes, 0.0), axis=1), axis=0
        )
        numerator = tl.sum(
            tl.sum(tl.where(valid, residual * codes, 0.0), axis=1), axis=0
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
        codes = tl.maximum(
            tl.minimum(tl.floor(residual / scale + 0.5), QUANT_MAX),
            -QUANT_MAX,
        )
        denominator = tl.sum(
            tl.sum(tl.where(valid, codes * codes, 0.0), axis=1), axis=0
        )
        numerator = tl.sum(
            tl.sum(tl.where(valid, residual * codes, 0.0), axis=1), axis=0
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
    tl.store(
        source_base + dimension[None, :],
        codes * scale + anchor[None, :],
        mask=valid,
    )

    if QUANTIZE_SUMMARY:
        summary_scale = tl.maximum(
            tl.max(tl.where(valid_dimension, tl.abs(sums), 0.0), axis=0) / 127.0,
            1.0e-8,
        )
        summary_codes = tl.maximum(
            tl.minimum(tl.floor(sums / summary_scale + 0.5), 127.0), -127.0
        )
        if OPTIMIZE_SUMMARY_SCALE:
            denominator = tl.sum(
                tl.where(valid_dimension, summary_codes * summary_codes, 0.0),
                axis=0,
            )
            numerator = tl.sum(
                tl.where(valid_dimension, sums * summary_codes, 0.0), axis=0
            )
            summary_scale = tl.where(
                denominator > 0.0,
                tl.maximum(numerator / denominator, 1.0e-8),
                summary_scale,
            )
            summary_codes = tl.maximum(
                tl.minimum(tl.floor(sums / summary_scale + 0.5), 127.0),
                -127.0,
            )
            denominator = tl.sum(
                tl.where(valid_dimension, summary_codes * summary_codes, 0.0),
                axis=0,
            )
            numerator = tl.sum(
                tl.where(valid_dimension, sums * summary_codes, 0.0), axis=0
            )
            summary_scale = tl.where(
                denominator > 0.0,
                tl.maximum(numerator / denominator, 1.0e-8),
                summary_scale,
            )
        tl.store(
            sum_base + dimension,
            summary_codes * summary_scale,
            mask=refresh & valid_dimension,
        )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _fake_quantize_completed_virtual_pages_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    page_indices,
    leaf_k,
    leaf_v,
    page_sum_k,
    page_sum_v,
    page_counts,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    LEAF_K_BATCH_STRIDE: tl.constexpr,
    LEAF_K_HEAD_STRIDE: tl.constexpr,
    LEAF_K_TOKEN_STRIDE: tl.constexpr,
    LEAF_V_BATCH_STRIDE: tl.constexpr,
    LEAF_V_HEAD_STRIDE: tl.constexpr,
    LEAF_V_TOKEN_STRIDE: tl.constexpr,
    KEY_BITS: tl.constexpr,
    VALUE_BITS: tl.constexpr,
    OPTIMIZE_LEAF_SCALE: tl.constexpr,
    QUANTIZE_SUMMARIES: tl.constexpr,
    OPTIMIZE_SUMMARY_SCALE: tl.constexpr,
):
    """Simulate independent K/V storage precision on newly completed pages."""
    token_row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    refresh = ordinal % PAGE_SIZE == PAGE_SIZE - 1
    page_ordinal = ordinal // PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        refresh,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    valid_page = refresh & (page_id >= 0) & (page_id < PAGE_CAPACITY)
    page_id = tl.where(valid_page, page_id, 0)
    page_count = tl.load(
        page_counts + kv_row * PAGE_CAPACITY + page_id,
        mask=valid_page,
        other=0,
    ).to(tl.int32)
    token_offset = tl.arange(0, PAGE_SIZE)
    dimension = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    valid_token = valid_page & (token_offset < page_count)
    leaf_index = tl.load(
        page_indices + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)
    valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
    leaf_index = tl.where(valid_token, leaf_index, 0)

    if KEY_BITS > 0:
        _fake_quantize_virtual_page_tensor(
            leaf_k,
            page_sum_k,
            leaf_index,
            valid_token,
            valid_page,
            page_count,
            batch,
            kv_head,
            kv_row,
            page_id,
            group,
            dimension,
            PAGE_CAPACITY,
            HEAD_DIM,
            GROUP_SIZE,
            LEAF_CAPACITY,
            LEAF_K_BATCH_STRIDE,
            LEAF_K_HEAD_STRIDE,
            LEAF_K_TOKEN_STRIDE,
            (1 << (KEY_BITS - 1)) - 1,
            OPTIMIZE_LEAF_SCALE,
            QUANTIZE_SUMMARIES,
            OPTIMIZE_SUMMARY_SCALE,
        )
    if VALUE_BITS > 0:
        _fake_quantize_virtual_page_tensor(
            leaf_v,
            page_sum_v,
            leaf_index,
            valid_token,
            valid_page,
            page_count,
            batch,
            kv_head,
            kv_row,
            page_id,
            group,
            dimension,
            PAGE_CAPACITY,
            VALUE_DIM,
            GROUP_SIZE,
            LEAF_CAPACITY,
            LEAF_V_BATCH_STRIDE,
            LEAF_V_HEAD_STRIDE,
            LEAF_V_TOKEN_STRIDE,
            (1 << (VALUE_BITS - 1)) - 1,
            OPTIMIZE_LEAF_SCALE,
            QUANTIZE_SUMMARIES,
            OPTIMIZE_SUMMARY_SCALE,
        )


@triton.jit
def _quantize_all_virtual_pages_kernel(
    page_indices,
    leaf_k,
    leaf_v,
    page_sum_k,
    page_sum_v,
    page_counts,
    quantized_leaf_k,
    quantized_leaf_v,
    page_k_scales,
    page_v_scales,
    page_quantized_counts,
    KV_HEADS: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    LEAF_K_BATCH_STRIDE,
    LEAF_K_HEAD_STRIDE,
    LEAF_K_TOKEN_STRIDE: tl.constexpr,
    LEAF_V_BATCH_STRIDE,
    LEAF_V_HEAD_STRIDE,
    LEAF_V_TOKEN_STRIDE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    OPTIMIZE_SCALE: tl.constexpr,
):
    """Quantize every populated virtual page once prefill is complete."""
    page_row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    kv_row = page_row // PAGE_CAPACITY
    page_id = page_row - kv_row * PAGE_CAPACITY
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    page_count = tl.load(page_counts + page_row).to(tl.int32)
    refresh = page_count > 0
    token_offset = tl.arange(0, PAGE_SIZE)
    pair_offset = tl.arange(0, GROUP_SIZE // 2)
    even_dimension = group * GROUP_SIZE + pair_offset * 2
    odd_dimension = even_dimension + 1
    valid_token = refresh & (token_offset < page_count)
    leaf_index = tl.load(
        page_indices + page_row * PAGE_SIZE + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)

    _quantize_virtual_page_tensor(
        leaf_k,
        page_sum_k,
        quantized_leaf_k,
        page_k_scales,
        leaf_index,
        valid_token,
        refresh,
        page_count,
        batch,
        kv_head,
        kv_row,
        page_id,
        group,
        pair_offset,
        even_dimension,
        odd_dimension,
        PAGE_CAPACITY,
        PAGE_SIZE,
        HEAD_DIM,
        GROUP_SIZE,
        LEAF_CAPACITY,
        LEAF_K_BATCH_STRIDE,
        LEAF_K_HEAD_STRIDE,
        LEAF_K_TOKEN_STRIDE,
        QUANT_BITS,
        OPTIMIZE_SCALE,
    )
    _quantize_virtual_page_tensor(
        leaf_v,
        page_sum_v,
        quantized_leaf_v,
        page_v_scales,
        leaf_index,
        valid_token,
        refresh,
        page_count,
        batch,
        kv_head,
        kv_row,
        page_id,
        group,
        pair_offset,
        even_dimension,
        odd_dimension,
        PAGE_CAPACITY,
        PAGE_SIZE,
        VALUE_DIM,
        GROUP_SIZE,
        LEAF_CAPACITY,
        LEAF_V_BATCH_STRIDE,
        LEAF_V_HEAD_STRIDE,
        LEAF_V_TOKEN_STRIDE,
        QUANT_BITS,
        OPTIMIZE_SCALE,
    )
    tl.store(
        page_quantized_counts + page_row,
        page_count,
        mask=refresh & (group == 0),
    )


@triton.jit
def _requantize_appended_virtual_page_tensor(
    source,
    page_sum,
    quantized_page_sum,
    page_sum_scales,
    destination,
    scales,
    leaf_index,
    valid_token,
    old_token,
    refresh,
    old_count,
    new_count,
    leaf_offset,
    batch,
    kv_head,
    kv_row,
    page_id,
    group,
    pair_offset,
    even_dimension,
    odd_dimension,
    PAGE_CAPACITY: tl.constexpr,
    DIMENSION_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    SOURCE_TOKEN_COUNT: tl.constexpr,
    SOURCE_BATCH_STRIDE,
    SOURCE_HEAD_STRIDE,
    SOURCE_TOKEN_STRIDE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    QUANTIZED_SUMMARIES: tl.constexpr,
    OPTIMIZE_SUMMARY_SCALE: tl.constexpr,
    OPTIMIZE_LEAF_SCALE: tl.constexpr,
):
    """Requantize one changed page from old quantized and exact new leaves."""
    valid_even = valid_token[:, None] & (even_dimension[None, :] < DIMENSION_SIZE)
    valid_odd = valid_token[:, None] & (odd_dimension[None, :] < DIMENSION_SIZE)
    old_even_valid = valid_even & old_token[:, None]
    old_odd_valid = valid_odd & old_token[:, None]
    if QUANT_BITS == 4:
        packed_dimension = DIMENSION_SIZE // 2
        destination_base = (
            destination
            + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * packed_dimension
            + group * (GROUP_SIZE // 2)
            + pair_offset[None, :]
        )
        old_packed = tl.load(
            destination_base,
            mask=old_even_valid,
            other=0,
        ).to(tl.int32)
        old_even_code = (old_packed & 15) - 8
        old_odd_code = ((old_packed >> 4) & 15) - 8
    else:
        destination_base = (
            destination
            + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * DIMENSION_SIZE
        )
        old_even_code = tl.load(
            destination_base + even_dimension[None, :],
            mask=old_even_valid,
            other=0,
        ).to(tl.int32)
        old_odd_code = tl.load(
            destination_base + odd_dimension[None, :],
            mask=old_odd_valid,
            other=0,
        ).to(tl.int32)
    sum_base = page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    quantized_sum_base = (
        quantized_page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    )
    old_inverse_count = 1.0 / tl.maximum(old_count.to(tl.float32), 1.0)
    has_old_tokens = refresh & (old_count > 0)
    if QUANTIZED_SUMMARIES:
        old_summary_scale = tl.load(
            page_sum_scales
            + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // GROUP_SIZE)
            + group,
            mask=has_old_tokens,
            other=0.0,
        ).to(tl.float32)
        old_even_sum = (
            tl.load(
                quantized_sum_base + even_dimension,
                mask=has_old_tokens & (even_dimension < DIMENSION_SIZE),
                other=0,
            ).to(tl.float32)
            * old_summary_scale
        )
        old_odd_sum = (
            tl.load(
                quantized_sum_base + odd_dimension,
                mask=has_old_tokens & (odd_dimension < DIMENSION_SIZE),
                other=0,
            ).to(tl.float32)
            * old_summary_scale
        )
    else:
        old_even_sum = tl.load(
            sum_base + even_dimension,
            mask=has_old_tokens & (even_dimension < DIMENSION_SIZE),
            other=0.0,
        ).to(tl.float32)
        old_odd_sum = tl.load(
            sum_base + odd_dimension,
            mask=has_old_tokens & (odd_dimension < DIMENSION_SIZE),
            other=0.0,
        ).to(tl.float32)
    old_scale = tl.load(
        scales
        + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // GROUP_SIZE)
        + group,
        mask=refresh & (old_count > 0),
        other=0.0,
    ).to(tl.float32)
    old_even = old_even_sum * old_inverse_count + old_even_code * old_scale
    old_odd = old_odd_sum * old_inverse_count + old_odd_code * old_scale

    source_index = leaf_index - leaf_offset
    new_token = valid_token & ~old_token
    valid_source = new_token & (source_index >= 0) & (source_index < SOURCE_TOKEN_COUNT)
    source_base = (
        source
        + batch * SOURCE_BATCH_STRIDE
        + kv_head * SOURCE_HEAD_STRIDE
        + source_index[:, None] * SOURCE_TOKEN_STRIDE
    )
    new_even = tl.load(
        source_base + even_dimension[None, :],
        mask=valid_source[:, None] & (even_dimension[None, :] < DIMENSION_SIZE),
        other=0.0,
    ).to(tl.float32)
    new_odd = tl.load(
        source_base + odd_dimension[None, :],
        mask=valid_source[:, None] & (odd_dimension[None, :] < DIMENSION_SIZE),
        other=0.0,
    ).to(tl.float32)
    even = tl.where(old_token[:, None], old_even, new_even)
    odd = tl.where(old_token[:, None], old_odd, new_odd)
    new_even_sum = old_even_sum + tl.sum(
        tl.where(valid_source[:, None], new_even, 0.0), axis=0
    )
    new_odd_sum = old_odd_sum + tl.sum(
        tl.where(valid_source[:, None], new_odd, 0.0), axis=0
    )
    if QUANTIZED_SUMMARIES:
        new_summary_scale = tl.maximum(
            tl.maximum(
                tl.max(tl.abs(new_even_sum), axis=0),
                tl.max(tl.abs(new_odd_sum), axis=0),
            )
            / 127.0,
            1.0e-8,
        )
        new_even_code_float = tl.maximum(
            tl.minimum(tl.floor(new_even_sum / new_summary_scale + 0.5), 127.0),
            -127.0,
        )
        new_odd_code_float = tl.maximum(
            tl.minimum(tl.floor(new_odd_sum / new_summary_scale + 0.5), 127.0),
            -127.0,
        )
        if OPTIMIZE_SUMMARY_SCALE:
            denominator = tl.sum(new_even_code_float * new_even_code_float, axis=0)
            denominator += tl.sum(new_odd_code_float * new_odd_code_float, axis=0)
            numerator = tl.sum(new_even_sum * new_even_code_float, axis=0)
            numerator += tl.sum(new_odd_sum * new_odd_code_float, axis=0)
            new_summary_scale = tl.where(
                denominator > 0.0,
                tl.maximum(numerator / denominator, 1.0e-8),
                new_summary_scale,
            )
            new_even_code_float = tl.maximum(
                tl.minimum(tl.floor(new_even_sum / new_summary_scale + 0.5), 127.0),
                -127.0,
            )
            new_odd_code_float = tl.maximum(
                tl.minimum(tl.floor(new_odd_sum / new_summary_scale + 0.5), 127.0),
                -127.0,
            )
            denominator = tl.sum(new_even_code_float * new_even_code_float, axis=0)
            denominator += tl.sum(new_odd_code_float * new_odd_code_float, axis=0)
            numerator = tl.sum(new_even_sum * new_even_code_float, axis=0)
            numerator += tl.sum(new_odd_sum * new_odd_code_float, axis=0)
            new_summary_scale = tl.where(
                denominator > 0.0,
                tl.maximum(numerator / denominator, 1.0e-8),
                new_summary_scale,
            )
        new_even_code = new_even_code_float.to(tl.int8)
        new_odd_code = new_odd_code_float.to(tl.int8)
        tl.store(
            quantized_sum_base + even_dimension,
            new_even_code,
            mask=refresh & (even_dimension < DIMENSION_SIZE),
        )
        tl.store(
            quantized_sum_base + odd_dimension,
            new_odd_code,
            mask=refresh & (odd_dimension < DIMENSION_SIZE),
        )
        tl.store(
            page_sum_scales
            + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // GROUP_SIZE)
            + group,
            new_summary_scale,
            mask=refresh,
        )
    else:
        tl.store(
            sum_base + even_dimension,
            new_even_sum,
            mask=refresh & (even_dimension < DIMENSION_SIZE),
        )
        tl.store(
            sum_base + odd_dimension,
            new_odd_sum,
            mask=refresh & (odd_dimension < DIMENSION_SIZE),
        )

    inverse_count = 1.0 / tl.maximum(new_count.to(tl.float32), 1.0)
    even_residual = even - new_even_sum[None, :] * inverse_count
    odd_residual = odd - new_odd_sum[None, :] * inverse_count
    even_max = tl.max(
        tl.max(tl.where(valid_even, tl.abs(even_residual), 0.0), axis=1),
        axis=0,
    )
    odd_max = tl.max(
        tl.max(tl.where(valid_odd, tl.abs(odd_residual), 0.0), axis=1),
        axis=0,
    )
    quant_max: tl.constexpr = (1 << (QUANT_BITS - 1)) - 1
    scale = tl.maximum(tl.maximum(even_max, odd_max) / quant_max, 1.0e-8)
    even_code_float = tl.maximum(
        tl.minimum(tl.floor(even_residual / scale + 0.5), quant_max),
        -quant_max,
    )
    odd_code_float = tl.maximum(
        tl.minimum(tl.floor(odd_residual / scale + 0.5), quant_max),
        -quant_max,
    )
    if OPTIMIZE_LEAF_SCALE:
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
        even_code_float = tl.maximum(
            tl.minimum(tl.floor(even_residual / scale + 0.5), quant_max),
            -quant_max,
        )
        odd_code_float = tl.maximum(
            tl.minimum(tl.floor(odd_residual / scale + 0.5), quant_max),
            -quant_max,
        )
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=1,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
    if QUANT_BITS == 4:
        even_code = even_code_float.to(tl.int32) + 8
        odd_code = odd_code_float.to(tl.int32) + 8
        tl.store(
            destination_base,
            (even_code | (odd_code << 4)).to(tl.uint8),
            mask=valid_even,
        )
    else:
        tl.store(
            destination_base + even_dimension[None, :],
            even_code_float.to(tl.int8),
            mask=valid_even,
        )
        tl.store(
            destination_base + odd_dimension[None, :],
            odd_code_float.to(tl.int8),
            mask=valid_odd,
        )
    tl.store(
        scales
        + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // GROUP_SIZE)
        + group,
        scale,
        mask=refresh,
    )


@triton.jit(
    do_not_specialize=["leaf_offset", "TOKENS"],
    do_not_specialize_on_alignment=["leaf_offset", "TOKENS"],
)
def _append_quantized_virtual_pages_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    page_indices,
    append_k,
    append_v,
    page_sum_k,
    page_sum_v,
    quantized_page_sum_k,
    quantized_page_sum_v,
    page_sum_k_scales,
    page_sum_v_scales,
    page_counts,
    quantized_leaf_k,
    quantized_leaf_v,
    page_k_scales,
    page_v_scales,
    page_quantized_counts,
    leaf_offset,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    APPEND_K_BATCH_STRIDE,
    APPEND_K_HEAD_STRIDE,
    APPEND_K_TOKEN_STRIDE: tl.constexpr,
    APPEND_V_BATCH_STRIDE,
    APPEND_V_HEAD_STRIDE,
    APPEND_V_TOKEN_STRIDE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    QUANTIZED_SUMMARIES: tl.constexpr,
    OPTIMIZE_SUMMARY_SCALE: tl.constexpr,
    OPTIMIZE_LEAF_SCALE: tl.constexpr,
):
    token_row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    slot_length = tl.load(slot_lengths + kv_row * STATE_CAPACITY + owner).to(tl.int64)
    completes_page = ordinal % PAGE_SIZE == PAGE_SIZE - 1
    is_partial_tail = ordinal == slot_length - 1
    refresh = completes_page | is_partial_tail
    page_ordinal = ordinal // PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        refresh,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    old_count = tl.load(
        page_counts + kv_row * PAGE_CAPACITY + page_id,
        mask=refresh,
        other=0,
    ).to(tl.int32)
    new_count = tl.where(completes_page, PAGE_SIZE, ordinal % PAGE_SIZE + 1)
    token_offset = tl.arange(0, PAGE_SIZE)
    pair_offset = tl.arange(0, GROUP_SIZE // 2)
    even_dimension = group * GROUP_SIZE + pair_offset * 2
    odd_dimension = even_dimension + 1
    valid_token = refresh & (token_offset < new_count)
    old_token = valid_token & (token_offset < old_count)
    leaf_index = tl.load(
        page_indices + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)

    _requantize_appended_virtual_page_tensor(
        append_k,
        page_sum_k,
        quantized_page_sum_k,
        page_sum_k_scales,
        quantized_leaf_k,
        page_k_scales,
        leaf_index,
        valid_token,
        old_token,
        refresh,
        old_count,
        new_count,
        leaf_offset,
        batch,
        kv_head,
        kv_row,
        page_id,
        group,
        pair_offset,
        even_dimension,
        odd_dimension,
        PAGE_CAPACITY,
        HEAD_DIM,
        GROUP_SIZE,
        LEAF_CAPACITY,
        TOKENS,
        APPEND_K_BATCH_STRIDE,
        APPEND_K_HEAD_STRIDE,
        APPEND_K_TOKEN_STRIDE,
        QUANT_BITS,
        QUANTIZED_SUMMARIES,
        OPTIMIZE_SUMMARY_SCALE,
        OPTIMIZE_LEAF_SCALE,
    )
    _requantize_appended_virtual_page_tensor(
        append_v,
        page_sum_v,
        quantized_page_sum_v,
        page_sum_v_scales,
        quantized_leaf_v,
        page_v_scales,
        leaf_index,
        valid_token,
        old_token,
        refresh,
        old_count,
        new_count,
        leaf_offset,
        batch,
        kv_head,
        kv_row,
        page_id,
        group,
        pair_offset,
        even_dimension,
        odd_dimension,
        PAGE_CAPACITY,
        VALUE_DIM,
        GROUP_SIZE,
        LEAF_CAPACITY,
        TOKENS,
        APPEND_V_BATCH_STRIDE,
        APPEND_V_HEAD_STRIDE,
        APPEND_V_TOKEN_STRIDE,
        QUANT_BITS,
        QUANTIZED_SUMMARIES,
        OPTIMIZE_SUMMARY_SCALE,
        OPTIMIZE_LEAF_SCALE,
    )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=["TOKENS"],
)
def _finalize_appended_virtual_page_counts_kernel(
    owners,
    ordinals,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    page_counts,
    page_quantized_counts,
    TOKENS,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Publish page lengths after every quantization group reads the old ones."""
    token_row = tl.program_id(0).to(tl.int64)
    kv_row = token_row // TOKENS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    slot_length = tl.load(slot_lengths + kv_row * STATE_CAPACITY + owner).to(tl.int64)
    completes_page = ordinal % PAGE_SIZE == PAGE_SIZE - 1
    is_partial_tail = ordinal == slot_length - 1
    refresh = completes_page | is_partial_tail
    page_ordinal = ordinal // PAGE_SIZE
    page_id = _lookup_page_id(
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        kv_row,
        owner,
        page_ordinal,
        refresh,
        STATE_CAPACITY,
        INLINE_PAGES_PER_SLOT,
        PAGE_CAPACITY,
        HASH_CAPACITY,
        HASH_PROBES,
    ).to(tl.int64)
    new_count = tl.where(completes_page, PAGE_SIZE, ordinal % PAGE_SIZE + 1)
    tl.store(
        page_counts + kv_row * PAGE_CAPACITY + page_id,
        new_count,
        mask=refresh,
    )
    tl.store(
        page_quantized_counts + kv_row * PAGE_CAPACITY + page_id,
        new_count,
        mask=refresh,
    )


@triton.jit(
    do_not_specialize=["PROGRAM_OFFSET"],
    do_not_specialize_on_alignment=["PROGRAM_OFFSET"],
)
def _paged_leaf_attention_kernel(
    q,
    q_scales,
    packed_route_row,
    block_expert,
    block_starts,
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
    q_lengths,
    cu_q,
    expert_kv_row,
    expert_slot,
    out,
    lse,
    PROGRAM_OFFSET,
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
    SCALE_LOG2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,
    PARTIAL_OUTPUT: tl.constexpr,
    INT8_MMA: tl.constexpr,
    INT8_PV_MMA: tl.constexpr,
    INDEXED: tl.constexpr,
):
    split_program = tl.program_id(0).to(tl.int64)
    local_program = split_program // SPLIT_N
    split = split_program - local_program * SPLIT_N
    program = local_program + PROGRAM_OFFSET
    expert = tl.load(block_expert + program)
    query_block = program - tl.load(block_starts + expert)
    query_count = tl.load(q_lengths + expert)
    query_offset = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_query = query_offset < query_count
    packed_begin = tl.load(cu_q + expert).to(tl.int64)
    packed_row = packed_begin + query_offset.to(tl.int64)
    route_row = tl.load(
        packed_route_row + packed_row,
        mask=valid_query,
        other=0,
    ).to(tl.int64)
    query_row = route_row // ROUTE_COUNT

    kv_row = tl.load(expert_kv_row + expert).to(tl.int64)
    slot = tl.load(expert_slot + expert).to(tl.int64)

    head_offset = tl.arange(0, HEAD_DIM)
    value_offset = tl.arange(0, VALUE_DIM)
    q_block = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=valid_query[:, None],
        other=0.0,
    )
    if INT8_MMA:
        q_mma = q_block.to(tl.int8)
        q_scale = tl.load(q_scales + query_row, mask=valid_query, other=1.0).to(
            tl.float32
        )
    key_count = tl.load(slot_lengths + kv_row * STATE_CAPACITY + slot).to(tl.int32)
    if HASH_PROBES == 0:
        page_table = (
            slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
        )
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    token_offset = tl.arange(0, BLOCK_N)

    keys_per_split = (key_count + SPLIT_N - 1) // SPLIT_N
    split_begin = split * keys_per_split
    split_count = tl.maximum(tl.minimum(keys_per_split, key_count - split_begin), 0)
    for key_begin in tl.range(0, split_count, BLOCK_N, num_stages=1):
        logical_key = split_begin + key_begin + token_offset
        valid_key = (key_begin + token_offset) < split_count
        page_ordinal = logical_key // PAGE_SIZE
        within_page = logical_key % PAGE_SIZE
        if HASH_PROBES == 0:
            page_id = tl.load(page_table + page_ordinal, mask=valid_key, other=0).to(
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
                valid_key,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        physical_token = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
        if INDEXED:
            leaf_index = tl.load(
                page_indices + physical_token, mask=valid_key, other=0
            ).to(tl.int64)
            storage_token = kv_row * LEAF_CAPACITY + leaf_index
        else:
            storage_token = physical_token
        k_block = tl.load(
            page_k + storage_token[None, :] * HEAD_DIM + head_offset[:, None],
            mask=valid_key[None, :],
            other=0.0,
        )
        v_block = tl.load(
            page_v + storage_token[:, None] * VALUE_DIM + value_offset[None, :],
            mask=valid_key[:, None],
            other=0.0,
        )

        if INT8_MMA:
            scale_token = storage_token if INDEXED else physical_token
            key_scale = tl.load(
                page_k_scales + scale_token,
                mask=valid_key,
                other=1.0,
            ).to(tl.float32)
            scores = tl.dot(q_mma, k_block, out_dtype=tl.int32).to(tl.float32)
            scores *= SCALE_LOG2 * q_scale[:, None] * key_scale[None, :]
        else:
            scores = SCALE_LOG2 * tl.dot(q_block, k_block, out_dtype=tl.float32)
        scores = tl.where(
            valid_query[:, None] & valid_key[None, :],
            scores,
            -float("inf"),
        )
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum[:, None])
        probabilities = tl.where(
            valid_query[:, None] & valid_key[None, :],
            probabilities,
            0.0,
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        accumulator_t = tl.trans(accumulator)
        if INT8_MMA:
            value_scale = tl.load(
                page_v_scales + scale_token,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            if INT8_PV_MMA:
                scaled_probabilities = probabilities * value_scale[None, :]
                probability_scale = tl.maximum(
                    tl.max(tl.abs(scaled_probabilities), axis=1) / 127.0,
                    1.1754943508222875e-38,
                )
                probability_code = tl.maximum(
                    tl.minimum(
                        tl.floor(
                            scaled_probabilities / probability_scale[:, None] + 0.5
                        ),
                        127.0,
                    ),
                    -127.0,
                ).to(tl.int8)
                value_update = tl.dot(
                    tl.trans(v_block),
                    tl.trans(probability_code),
                    out_dtype=tl.int32,
                ).to(tl.float32)
                accumulator_t += value_update * probability_scale[None, :]
            else:
                # V_i = code_i * scale_i.  Fold the per-token scale into P
                # instead of multiplying every value channel by it.  This is
                # algebraically identical before BF16 rounding and changes the
                # scale work from BLOCK_N * VALUE_DIM to BLOCK_M * BLOCK_N.
                # It also leaves the code matrix directly consumable by the
                # BF16 MMA used for the short-context PV path.
                scaled_probabilities = (probabilities * value_scale[None, :]).to(
                    tl.bfloat16
                )
                accumulator_t += tl.dot(
                    tl.trans(v_block.to(tl.bfloat16)),
                    tl.trans(scaled_probabilities),
                    out_dtype=tl.float32,
                )
        else:
            accumulator_t += tl.dot(
                tl.trans(v_block),
                tl.trans(probabilities.to(v_block.dtype)),
                out_dtype=tl.float32,
            )
        accumulator = tl.trans(accumulator_t)
        maximum = new_maximum

    has_mass = denominator > 0.0
    normalized = tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0)
    natural_lse = tl.where(
        has_mass,
        (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
        -float("inf"),
    )
    if PARTIAL_OUTPUT:
        partial_row = (
            (local_program * SPLIT_N + split) * BLOCK_M
            + tl.arange(0, BLOCK_M).to(tl.int64)
        )
        tl.store(
            out + partial_row[:, None] * VALUE_DIM + value_offset[None, :],
            normalized,
            mask=valid_query[:, None],
        )
        tl.store(lse + partial_row, natural_lse, mask=valid_query)
    else:
        tl.store(
            out + route_row[:, None] * VALUE_DIM + value_offset[None, :],
            normalized,
            mask=valid_query[:, None],
        )
        tl.store(lse + route_row, natural_lse, mask=valid_query)


@triton.jit(
    do_not_specialize=["PROGRAM_OFFSET"],
    do_not_specialize_on_alignment=["PROGRAM_OFFSET"],
)
def _reduce_split_expert_attention_kernel(
    packed_route_row,
    block_expert,
    block_starts,
    q_lengths,
    cu_q,
    partial_out,
    partial_lse,
    out,
    lse,
    PROGRAM_OFFSET,
    VALUE_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    """Merge compact split-N partials for one long-expert query block."""
    local_program = tl.program_id(0).to(tl.int64)
    program = local_program + PROGRAM_OFFSET
    expert = tl.load(block_expert + program)
    query_block = program - tl.load(block_starts + expert)
    query_count = tl.load(q_lengths + expert)
    lane = tl.arange(0, BLOCK_M)
    query_offset = query_block * BLOCK_M + lane
    valid_query = query_offset < query_count
    packed_begin = tl.load(cu_q + expert).to(tl.int64)
    packed_row = packed_begin + query_offset.to(tl.int64)
    route_row = tl.load(
        packed_route_row + packed_row,
        mask=valid_query,
        other=0,
    ).to(tl.int64)

    value_offset = tl.arange(0, VALUE_DIM)
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    for split in tl.static_range(0, SPLIT_N):
        partial_row = (local_program * SPLIT_N + split) * BLOCK_M + lane
        split_lse = tl.load(
            partial_lse + partial_row,
            mask=valid_query,
            other=-float("inf"),
        ).to(tl.float32)
        new_maximum = tl.maximum(maximum, split_lse)
        correction = tl.where(
            maximum == -float("inf"),
            0.0,
            tl.exp(maximum - new_maximum),
        )
        split_weight = tl.where(
            split_lse == -float("inf"),
            0.0,
            tl.exp(split_lse - new_maximum),
        )
        split_value = tl.load(
            partial_out + partial_row[:, None] * VALUE_DIM + value_offset[None, :],
            mask=valid_query[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * correction[:, None]
            + split_value * split_weight[:, None]
        )
        denominator = denominator * correction + split_weight
        maximum = new_maximum

    has_mass = denominator > 0.0
    normalized = tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0)
    combined_lse = tl.where(
        has_mass,
        maximum + tl.log(denominator),
        -float("inf"),
    )
    tl.store(
        out + route_row[:, None] * VALUE_DIM + value_offset[None, :],
        normalized,
        mask=valid_query[:, None],
    )
    tl.store(lse + route_row, combined_lse, mask=valid_query)


@triton.jit(
    do_not_specialize=["QUERY_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN"],
)
def _query_major_paged_leaf_attention_kernel(
    q,
    page_k,
    page_v,
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    top_slots,
    out,
    lse,
    QUERY_LEN,
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
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEXED: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    batch_head = query_row // QUERY_LEN
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    query = tl.load(
        q + query_row * HEAD_DIM + head_offset,
        mask=head_offset < HEAD_DIM,
        other=0.0,
    )
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_BLOCK_DIM,), tl.float32)

    for route in tl.static_range(0, ROUTE_COUNT):
        routed_slot = tl.load(top_slots + query_row * ROUTE_COUNT + route).to(tl.int64)
        slot_valid = routed_slot >= 0
        slot = tl.where(slot_valid, routed_slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=slot_valid,
            other=0,
        ).to(tl.int32)
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
            logical_key = key_begin + token_offset
            valid_key = logical_key < key_count
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal, mask=valid_key, other=0
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid_key,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            physical_token = (
                kv_row * PAGE_CAPACITY + page_id
            ) * PAGE_SIZE + within_page
            if INDEXED:
                leaf_index = tl.load(
                    page_indices + physical_token, mask=valid_key, other=0
                ).to(tl.int64)
                storage_token = kv_row * LEAF_CAPACITY + leaf_index
            else:
                storage_token = physical_token
            keys = tl.load(
                page_k + storage_token[:, None] * HEAD_DIM + head_offset[None, :],
                mask=valid_key[:, None] & (head_offset[None, :] < HEAD_DIM),
                other=0.0,
            )
            values = tl.load(
                page_v + storage_token[:, None] * VALUE_DIM + value_offset[None, :],
                mask=valid_key[:, None] & (value_offset[None, :] < VALUE_DIM),
                other=0.0,
            )
            scores = SCALE_LOG2 * tl.sum(
                keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
            )
            scores = tl.where(valid_key, scores, -float("inf"))
            block_maximum = tl.max(scores, axis=0)
            new_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.math.exp2(maximum - new_maximum)
            probabilities = tl.math.exp2(scores - new_maximum)
            probabilities = tl.where(valid_key, probabilities, 0.0)
            denominator = denominator * correction + tl.sum(probabilities, axis=0)
            value_update = tl.sum(probabilities[:, None] * values, axis=0)
            accumulator = accumulator * correction + value_update
            maximum = new_maximum

    has_mass = denominator > 0.0
    tl.store(
        out + query_row * VALUE_DIM + value_offset,
        tl.where(has_mass, accumulator / denominator, 0.0),
        mask=value_offset < VALUE_DIM,
    )
    tl.store(
        lse + query_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )


def query_major_paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    page_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_n: int = 16,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse all routed slots for each query into one online softmax."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("query-major paged leaf attention is forward-only")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(page_k.size(1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    indexed = page_indices is not None
    page_size = int(page_indices.size(3)) if indexed else int(page_k.size(3))
    page_capacity = int(page_indices.size(2)) if indexed else int(page_k.size(2))
    if page_size != 16:
        raise ValueError("query-major leaf attention requires 16-token pages")
    rows = batch * query_heads * query_len
    value_dim = int(page_v.size(-1))
    output = torch.empty(rows, value_dim, dtype=q.dtype, device=q.device)
    lse = torch.empty(rows, dtype=torch.float32, device=q.device)
    begin = None
    if timing_events is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
    _query_major_paged_leaf_attention_kernel[(rows,)](
        q,
        page_k,
        page_v,
        page_indices if indexed else page_k,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        output,
        lse,
        QUERY_LEN=query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=int(page_k.size(1)),
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        LEAF_CAPACITY=int(page_k.size(2)) if indexed else 1,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=page_size,
        ROUTE_COUNT=int(top_slots.size(-1)),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_N=block_n,
        INDEXED=indexed,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if begin is None:
            raise AssertionError("query-major timing start is missing")
        timing_events.setdefault("kernel", []).append((begin, end))
        timing_events.setdefault("total", []).append((begin, end))
    return (
        output.reshape(batch, query_heads, query_len, value_dim),
        lse.reshape(batch, query_heads, query_len),
    )


@triton.jit(
    do_not_specialize=["QUERY_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN"],
)
def _query_tile_masked_paged_leaf_attention_kernel(
    q,
    page_k,
    page_v,
    page_indices,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    top_slots,
    out,
    lse,
    QUERY_LEN,
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
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEXED: tl.constexpr,
):
    """Share routed pages across nearby queries and mask non-selecting rows."""
    batch_head = tl.program_id(0).to(tl.int64)
    query_begin = tl.program_id(1).to(tl.int64) * BLOCK_M
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head

    row = tl.arange(0, BLOCK_M)
    query_offset = query_begin + row
    query_valid = query_offset < QUERY_LEN
    query_row = batch_head * QUERY_LEN + query_offset
    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=query_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
        other=0.0,
    )
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK_DIM), tl.float32)

    route_offset = tl.arange(0, CANDIDATE_BLOCK)
    route_row = route_offset // ROUTE_COUNT
    route_rank = route_offset - route_row * ROUTE_COUNT
    route_valid = (route_row < BLOCK_M) & (query_begin + route_row < QUERY_LEN)
    all_slots = tl.load(
        top_slots
        + (batch_head * QUERY_LEN + query_begin + route_row) * ROUTE_COUNT
        + route_rank,
        mask=route_valid,
        other=-1,
    ).to(tl.int64)

    for candidate in tl.static_range(0, BLOCK_M * ROUTE_COUNT):
        candidate_row = candidate // ROUTE_COUNT
        candidate_rank = candidate - candidate_row * ROUTE_COUNT
        candidate_query_valid = query_begin + candidate_row < QUERY_LEN
        slot = tl.load(
            top_slots
            + (batch_head * QUERY_LEN + query_begin + candidate_row) * ROUTE_COUNT
            + candidate_rank,
            mask=candidate_query_valid,
            other=-1,
        ).to(tl.int64)
        slot_valid = candidate_query_valid & (slot >= 0) & (slot < STATE_CAPACITY)
        seen_before = tl.sum(
            (
                route_valid
                & (route_offset < candidate)
                & (all_slots == slot)
            ).to(tl.int32),
            axis=0,
        ) > 0
        selected = tl.zeros((BLOCK_M,), tl.int1)
        for route in tl.static_range(0, ROUTE_COUNT):
            row_slot = tl.load(
                top_slots + query_row * ROUTE_COUNT + route,
                mask=query_valid,
                other=-1,
            ).to(tl.int64)
            selected |= row_slot == slot
        selected &= query_valid & slot_valid
        candidate_active = slot_valid & ~seen_before & (tl.sum(selected, axis=0) > 0)
        if candidate_active:
            key_count = tl.load(
                slot_lengths + kv_row * STATE_CAPACITY + slot
            ).to(tl.int32)
            for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
                logical_key = key_begin + token_offset
                valid_key = logical_key < key_count
                page_ordinal = logical_key // PAGE_SIZE
                within_page = logical_key % PAGE_SIZE
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid_key,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
                physical_token = (
                    kv_row * PAGE_CAPACITY + page_id
                ) * PAGE_SIZE + within_page
                if INDEXED:
                    leaf_index = tl.load(
                        page_indices + physical_token,
                        mask=valid_key,
                        other=0,
                    ).to(tl.int64)
                    storage_token = kv_row * LEAF_CAPACITY + leaf_index
                else:
                    storage_token = physical_token
                keys = tl.load(
                    page_k
                    + storage_token[None, :] * HEAD_DIM
                    + head_offset[:, None],
                    mask=valid_key[None, :] & (head_offset[:, None] < HEAD_DIM),
                    other=0.0,
                )
                values = tl.load(
                    page_v
                    + storage_token[:, None] * VALUE_DIM
                    + value_offset[None, :],
                    mask=valid_key[:, None] & (value_offset[None, :] < VALUE_DIM),
                    other=0.0,
                )
                score_valid = selected[:, None] & valid_key[None, :]
                scores = SCALE_LOG2 * tl.dot(
                    queries, keys, out_dtype=tl.float32
                )
                scores = tl.where(score_valid, scores, -float("inf"))
                block_maximum = tl.max(scores, axis=1)
                new_maximum = tl.where(
                    selected,
                    tl.maximum(maximum, block_maximum),
                    maximum,
                )
                correction = tl.where(
                    selected,
                    tl.math.exp2(maximum - new_maximum),
                    1.0,
                )
                probabilities = tl.where(
                    score_valid,
                    tl.math.exp2(scores - new_maximum[:, None]),
                    0.0,
                )
                denominator = denominator * correction + tl.sum(
                    probabilities, axis=1
                )
                accumulator = accumulator * correction[:, None] + tl.dot(
                    probabilities.to(values.dtype),
                    values,
                    out_dtype=tl.float32,
                )
                maximum = new_maximum

    has_mass = query_valid & (denominator > 0.0)
    tl.store(
        out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0),
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
    )
    tl.store(
        lse + query_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=query_valid,
    )


def query_tile_masked_paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    page_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_m: int = 4,
    block_n: int = 32,
    num_warps: int = 4,
    waves_per_eu: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked query-tile union without changing the coarse branch."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("masked tile leaf attention is forward-only")
    if block_m not in (2, 4, 8, 16):
        raise ValueError("masked leaf query tile must be 2, 4, 8, or 16")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("masked leaf key tile must be a positive power of two")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(page_k.size(1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("masked leaf query/KV grouping is inconsistent")
    indexed = page_indices is not None
    page_size = int(page_indices.size(3)) if indexed else int(page_k.size(3))
    page_capacity = int(page_indices.size(2)) if indexed else int(page_k.size(2))
    if page_size != 16:
        raise ValueError("masked tile leaf attention requires 16-token pages")
    value_dim = int(page_v.size(-1))
    output = torch.empty(
        batch,
        query_heads,
        query_len,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.empty(
        batch, query_heads, query_len, dtype=torch.float32, device=q.device
    )
    begin = None
    if timing_events is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
    candidate_count = block_m * int(top_slots.size(-1))
    _query_tile_masked_paged_leaf_attention_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        q,
        page_k,
        page_v,
        page_indices if indexed else page_k,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        output,
        lse,
        QUERY_LEN=query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        LEAF_CAPACITY=int(page_k.size(2)) if indexed else 1,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=page_size,
        ROUTE_COUNT=int(top_slots.size(-1)),
        CANDIDATE_BLOCK=triton.next_power_of_2(candidate_count),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        INDEXED=indexed,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if begin is None:
            raise AssertionError("masked tile timing start is missing")
        timing_events.setdefault("kernel", []).append((begin, end))
        timing_events.setdefault("total", []).append((begin, end))
    return output, lse


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _dense_page_summary_attention_kernel(
    q,
    cache_indices,
    page_sum_k,
    page_sum_v,
    page_counts,
    next_page,
    top_pages,
    out,
    lse,
    query_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    TOP_PAGES: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    REMOVE_SELECTED: tl.constexpr,
):
    """Dense page-summary attention with stable exact-page replacement prep.

    A single page scan retains the globally best pages for each query while
    accumulating attention over the complete summary field.  The selected
    summary contributions are then removed in the scan's normalization frame.
    Exact leaves from those pages are evaluated by the companion kernel and
    LSE-merged by the caller.
    """
    batch = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    query_begin = tl.program_id(2).to(tl.int64) * BLOCK_Q
    row = tl.arange(0, BLOCK_M)
    group_head = row // BLOCK_Q
    query_offset = query_begin + row % BLOCK_Q
    query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = (group_head < KV_GROUP_SIZE) & (query_offset < query_len)
    query_row = (batch * QUERY_HEADS + query_head) * query_len + query_offset
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    allocated_pages = tl.load(next_page + kv_row).to(tl.int32)

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    page_offset = tl.arange(0, BLOCK_N)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=query_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
        other=0.0,
    )

    top_rank = tl.arange(0, TOP_PAGES)
    top_packed = tl.full((BLOCK_M, TOP_PAGES), -9223372036854775807, tl.int64)
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK_DIM), tl.float32)
    for page_begin in tl.range(0, allocated_pages, BLOCK_N, num_stages=1):
        page = page_begin + page_offset
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page,
            mask=page < allocated_pages,
            other=0,
        ).to(tl.float32)
        page_valid = (page < allocated_pages) & (count > 0.0)
        safe_count = tl.maximum(count, 1.0)
        key_sums = tl.load(
            page_sum_k
            + (kv_row * PAGE_CAPACITY + page[:, None]) * HEAD_DIM
            + head_offset[None, :],
            mask=page_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
            other=0.0,
        )
        value_sums = tl.load(
            page_sum_v
            + (kv_row * PAGE_CAPACITY + page[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=page_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
            other=0.0,
        )
        page_keys = (key_sums.to(tl.float32) / safe_count[:, None]).to(queries.dtype)
        page_values = (value_sums.to(tl.float32) / safe_count[:, None]).to(
            value_sums.dtype
        )
        block_scores = (
            SCALE_LOG2 * tl.dot(queries, tl.trans(page_keys), out_dtype=tl.float32)
            + tl.log2(safe_count)[None, :]
        )
        block_scores = tl.where(
            query_valid[:, None] & page_valid[None, :],
            block_scores,
            -float("inf"),
        )

        block_maximum = tl.max(block_scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(block_scores - new_maximum[:, None])
        probabilities = tl.where(
            query_valid[:, None] & page_valid[None, :], probabilities, 0.0
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(page_values.dtype),
            page_values,
            out_dtype=tl.float32,
        )
        maximum = new_maximum

        # Pack the exactly ordered FP32 score bits with an inverse page ID.
        # This lets one bitonic network move scores and page IDs together,
        # including deterministic tie-breaking, with no serial argmax loop.
        score_bits = block_scores.to(tl.uint32, bitcast=True)
        negative = (score_bits & 0x80000000) != 0
        ordered_bits = tl.where(
            negative,
            score_bits ^ 0xFFFFFFFF,
            score_bits ^ 0x80000000,
        ).to(tl.int64)
        score_rank = ordered_bits - 2147483648
        inverse_page = 4294967295 - page.to(tl.int64)
        packed_scores = score_rank * 4294967296 + inverse_page[None, :]
        block_top = tl.topk(packed_scores, TOP_PAGES, dim=1)
        top_packed = tl.topk(tl.interleave(top_packed, block_top), TOP_PAGES, dim=1)

    inverse_page = top_packed & 0xFFFFFFFF
    selected_pages = (4294967295 - inverse_page).to(tl.int32)

    if REMOVE_SELECTED:
        # The legacy per-query exact path removes only that row's selections.
        # Tile-union attention defers removal until the shared union is known.
        for rank in tl.static_range(0, TOP_PAGES):
            selected_rank = top_rank[None, :] == rank
            page = tl.sum(tl.where(selected_rank, selected_pages, 0), axis=1).to(
                tl.int64
            )
            selected_score = tl.sum(tl.where(selected_rank, top_packed, 0), axis=1).to(
                tl.int64
            )
            selected_inverse_page = selected_score & 0xFFFFFFFF
            selected_score_rank = (selected_score - selected_inverse_page) // 4294967296
            selected_ordered_bits = (selected_score_rank + 2147483648).to(tl.uint32)
            selected_negative = selected_ordered_bits < 0x80000000
            selected_score_bits = tl.where(
                selected_negative,
                selected_ordered_bits ^ 0xFFFFFFFF,
                selected_ordered_bits ^ 0x80000000,
            )
            selected_score = selected_score_bits.to(tl.float32, bitcast=True)
            selected_valid = query_valid & (page >= 0) & (page < allocated_pages)
            safe_page = tl.where(selected_valid, page, 0)
            count = tl.load(
                page_counts + kv_row * PAGE_CAPACITY + safe_page,
                mask=selected_valid,
                other=0,
            ).to(tl.float32)
            selected_valid &= count > 0.0
            safe_count = tl.maximum(count, 1.0)
            value_sums = tl.load(
                page_sum_v
                + (kv_row * PAGE_CAPACITY + safe_page[:, None]) * VALUE_DIM
                + value_offset[None, :],
                mask=selected_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
                other=0.0,
            )
            page_values = value_sums.to(tl.float32) / safe_count[:, None]
            weight = tl.where(
                selected_valid,
                tl.math.exp2(selected_score - maximum),
                0.0,
            )
            denominator -= weight
            accumulator -= weight[:, None] * page_values

    has_mass = denominator > 0.0
    tl.store(
        out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0),
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
    )
    tl.store(
        lse + query_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=query_valid,
    )
    tl.store(
        top_pages + query_row[:, None] * TOP_PAGES + top_rank[None, :],
        selected_pages,
        mask=query_valid[:, None],
    )


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _dense_page_regular_attention_kernel(
    q,
    cache_indices,
    page_sum_k,
    page_sum_v,
    page_counts,
    next_page,
    out,
    lse,
    query_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Regular dense attention over all page summaries, without routing."""
    batch = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    query_begin = tl.program_id(2).to(tl.int64) * BLOCK_Q
    row = tl.arange(0, BLOCK_M)
    group_head = row // BLOCK_Q
    query_offset = query_begin + row % BLOCK_Q
    query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = (group_head < KV_GROUP_SIZE) & (query_offset < query_len)
    query_row = (batch * QUERY_HEADS + query_head) * query_len + query_offset
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    allocated_pages = tl.load(next_page + kv_row).to(tl.int32)

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    page_offset = tl.arange(0, BLOCK_N)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=query_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
        other=0.0,
    )
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK_DIM), tl.float32)
    for page_begin in tl.range(0, allocated_pages, BLOCK_N, num_stages=1):
        page = page_begin + page_offset
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page,
            mask=page < allocated_pages,
            other=0,
        ).to(tl.float32)
        page_valid = (page < allocated_pages) & (count > 0.0)
        safe_count = tl.maximum(count, 1.0)
        key_sums = tl.load(
            page_sum_k
            + (kv_row * PAGE_CAPACITY + page[:, None]) * HEAD_DIM
            + head_offset[None, :],
            mask=page_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
            other=0.0,
        )
        value_sums = tl.load(
            page_sum_v
            + (kv_row * PAGE_CAPACITY + page[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=page_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
            other=0.0,
        )
        page_keys = (key_sums.to(tl.float32) / safe_count[:, None]).to(queries.dtype)
        page_values = (value_sums.to(tl.float32) / safe_count[:, None]).to(
            value_sums.dtype
        )
        scores = (
            SCALE_LOG2 * tl.dot(queries, tl.trans(page_keys), out_dtype=tl.float32)
            + tl.log2(safe_count)[None, :]
        )
        valid = query_valid[:, None] & page_valid[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.where(
            valid, tl.math.exp2(scores - new_maximum[:, None]), 0.0
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(page_values.dtype),
            page_values,
            out_dtype=tl.float32,
        )
        maximum = new_maximum

    has_mass = denominator > 0.0
    tl.store(
        out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0),
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
    )
    tl.store(
        lse + query_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=query_valid,
    )


@triton.jit(
    do_not_specialize=["query_len"],
    do_not_specialize_on_alignment=["query_len"],
)
def _dense_page_topk_kernel(
    q,
    cache_indices,
    page_sum_k,
    page_counts,
    next_page,
    top_pages,
    top_scores,
    weighted_lse,
    query_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    TOP_PAGES: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """QK-only streaming top-k over every physical page summary."""
    batch = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    query_begin = tl.program_id(2).to(tl.int64) * BLOCK_Q
    row = tl.arange(0, BLOCK_M)
    group_head = row // BLOCK_Q
    query_offset = query_begin + row % BLOCK_Q
    query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = (group_head < KV_GROUP_SIZE) & (query_offset < query_len)
    query_row = (batch * QUERY_HEADS + query_head) * query_len + query_offset
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head
    allocated_pages = tl.load(next_page + kv_row).to(tl.int32)
    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    page_offset = tl.arange(0, BLOCK_N)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=query_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
        other=0.0,
    )
    top_packed = tl.full((BLOCK_M, TOP_PAGES), -9223372036854775807, tl.int64)
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    for page_begin in tl.range(0, allocated_pages, BLOCK_N, num_stages=1):
        page = page_begin + page_offset
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page,
            mask=page < allocated_pages,
            other=0,
        ).to(tl.float32)
        page_valid = (page < allocated_pages) & (count > 0.0)
        safe_count = tl.maximum(count, 1.0)
        key_sums = tl.load(
            page_sum_k
            + (kv_row * PAGE_CAPACITY + page[:, None]) * HEAD_DIM
            + head_offset[None, :],
            mask=page_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
            other=0.0,
        )
        page_keys = (key_sums.to(tl.float32) / safe_count[:, None]).to(queries.dtype)
        scores = (
            SCALE_LOG2 * tl.dot(queries, tl.trans(page_keys), out_dtype=tl.float32)
            + tl.log2(safe_count)[None, :]
        )
        scores = tl.where(
            query_valid[:, None] & page_valid[None, :],
            scores,
            -float("inf"),
        )
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.where(
            query_valid[:, None] & page_valid[None, :],
            tl.math.exp2(scores - new_maximum[:, None]),
            0.0,
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        maximum = new_maximum
        score_bits = scores.to(tl.uint32, bitcast=True)
        negative = (score_bits & 0x80000000) != 0
        ordered_bits = tl.where(
            negative,
            score_bits ^ 0xFFFFFFFF,
            score_bits ^ 0x80000000,
        ).to(tl.int64)
        score_rank = ordered_bits - 2147483648
        inverse_page = 4294967295 - page.to(tl.int64)
        packed = score_rank * 4294967296 + inverse_page[None, :]
        block_top = tl.topk(packed, TOP_PAGES, dim=1)
        top_packed = tl.topk(tl.interleave(top_packed, block_top), TOP_PAGES, dim=1)

    inverse_page = top_packed & 0xFFFFFFFF
    selected_pages = (4294967295 - inverse_page).to(tl.int32)
    selected_score_rank = (top_packed - inverse_page) // 4294967296
    ordered_bits = (selected_score_rank + 2147483648).to(tl.uint32)
    negative = ordered_bits < 0x80000000
    score_bits = tl.where(
        negative,
        ordered_bits ^ 0xFFFFFFFF,
        ordered_bits ^ 0x80000000,
    )
    selected_scores = score_bits.to(tl.float32, bitcast=True)
    rank = tl.arange(0, TOP_PAGES)
    tl.store(
        top_pages + query_row[:, None] * TOP_PAGES + rank[None, :],
        selected_pages,
        mask=query_valid[:, None],
    )
    tl.store(
        top_scores + query_row[:, None] * TOP_PAGES + rank[None, :],
        selected_scores,
        mask=query_valid[:, None],
    )
    tl.store(
        weighted_lse + query_row,
        (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
        mask=query_valid,
    )


@triton.jit
def _dense_page_remove_selected_kernel(
    summary_out,
    summary_lse,
    weighted_lse,
    cache_indices,
    page_sum_v,
    page_counts,
    top_pages,
    top_scores,
    out,
    lse,
    QUERY_LEN: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    TOP_PAGES: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    batch_head = query_row // QUERY_LEN
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = cache_batch * KV_HEADS + kv_head
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    total_output = tl.load(
        summary_out + query_row * VALUE_DIM + value_offset,
        mask=value_offset < VALUE_DIM,
        other=0.0,
    ).to(tl.float32)
    unweighted_lse = tl.load(summary_lse + query_row).to(tl.float32)
    total_lse = tl.load(weighted_lse + query_row).to(tl.float32)
    total_output *= tl.math.exp2((unweighted_lse - total_lse) * 1.4426950408889634)
    selected_mass = tl.zeros((), tl.float32)
    selected_numerator = tl.zeros((VALUE_BLOCK_DIM,), tl.float32)
    for rank in tl.static_range(0, TOP_PAGES):
        page = tl.load(top_pages + query_row * TOP_PAGES + rank).to(tl.int64)
        score = tl.load(top_scores + query_row * TOP_PAGES + rank).to(tl.float32)
        valid = (page >= 0) & (page < PAGE_CAPACITY) & (score > -float("inf"))
        safe_page = tl.where(valid, page, 0)
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + safe_page,
            mask=valid,
            other=0,
        ).to(tl.float32)
        valid &= count > 0.0
        values = tl.load(
            page_sum_v
            + (kv_row * PAGE_CAPACITY + safe_page) * VALUE_DIM
            + value_offset,
            mask=valid & (value_offset < VALUE_DIM),
            other=0.0,
        ).to(tl.float32) / tl.maximum(count, 1.0)
        weight = tl.where(
            valid,
            tl.math.exp2(score - total_lse * 1.4426950408889634),
            0.0,
        )
        selected_mass += weight
        selected_numerator += weight * values
    residual_mass = 1.0 - selected_mass
    has_mass = residual_mass > 1e-7
    residual = (total_output - selected_numerator) / tl.maximum(residual_mass, 1e-7)
    tl.store(
        out + query_row * VALUE_DIM + value_offset,
        tl.where(has_mass, residual, 0.0),
        mask=value_offset < VALUE_DIM,
    )
    tl.store(
        lse + query_row,
        tl.where(has_mass, total_lse + tl.log(residual_mass), -float("inf")),
    )


@triton.jit(
    do_not_specialize=[
        "LEAF_CAPACITY",
        "LEAF_K_BATCH_STRIDE",
        "LEAF_K_HEAD_STRIDE",
        "LEAF_K_TOKEN_STRIDE",
        "LEAF_V_BATCH_STRIDE",
        "LEAF_V_HEAD_STRIDE",
        "LEAF_V_TOKEN_STRIDE",
        "query_len",
    ],
    do_not_specialize_on_alignment=[
        "LEAF_CAPACITY",
        "LEAF_K_BATCH_STRIDE",
        "LEAF_K_HEAD_STRIDE",
        "LEAF_K_TOKEN_STRIDE",
        "LEAF_V_BATCH_STRIDE",
        "LEAF_V_HEAD_STRIDE",
        "LEAF_V_TOKEN_STRIDE",
        "query_len",
    ],
)
def _dense_page_exact_attention_kernel(
    q,
    cache_indices,
    leaf_k,
    leaf_v,
    page_indices,
    page_counts,
    top_pages,
    out,
    lse,
    query_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY,
    LEAF_K_BATCH_STRIDE,
    LEAF_K_HEAD_STRIDE,
    LEAF_K_TOKEN_STRIDE,
    LEAF_V_BATCH_STRIDE,
    LEAF_V_HEAD_STRIDE,
    LEAF_V_TOKEN_STRIDE,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOP_PAGES: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = cache_batch * KV_HEADS + kv_head
    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    token_offset = tl.arange(0, PAGE_SIZE)
    query = tl.load(
        q + query_row * HEAD_DIM + head_offset,
        mask=head_offset < HEAD_DIM,
        other=0.0,
    )
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_BLOCK_DIM,), tl.float32)

    for rank in tl.static_range(0, TOP_PAGES):
        page = tl.load(top_pages + query_row * TOP_PAGES + rank).to(tl.int64)
        valid_page = (page >= 0) & (page < PAGE_CAPACITY)
        page = tl.where(valid_page, page, 0)
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + page,
            mask=valid_page,
            other=0,
        ).to(tl.int32)
        valid_token = valid_page & (token_offset < count)
        leaf_index = tl.load(
            page_indices + (kv_row * PAGE_CAPACITY + page) * PAGE_SIZE + token_offset,
            mask=valid_token,
            other=-1,
        ).to(tl.int64)
        valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
        leaf_index = tl.where(valid_token, leaf_index, 0)
        keys = tl.load(
            leaf_k
            + cache_batch * LEAF_K_BATCH_STRIDE
            + kv_head * LEAF_K_HEAD_STRIDE
            + leaf_index[:, None] * LEAF_K_TOKEN_STRIDE
            + head_offset[None, :],
            mask=valid_token[:, None] & (head_offset[None, :] < HEAD_DIM),
            other=0.0,
        )
        values = tl.load(
            leaf_v
            + cache_batch * LEAF_V_BATCH_STRIDE
            + kv_head * LEAF_V_HEAD_STRIDE
            + leaf_index[:, None] * LEAF_V_TOKEN_STRIDE
            + value_offset[None, :],
            mask=valid_token[:, None] & (value_offset[None, :] < VALUE_DIM),
            other=0.0,
        )
        scores = SCALE_LOG2 * tl.sum(
            keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
        )
        scores = tl.where(valid_token, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum)
        probabilities = tl.where(valid_token, probabilities, 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        maximum = new_maximum

    has_mass = denominator > 0.0
    tl.store(
        out + query_row * VALUE_DIM + value_offset,
        tl.where(has_mass, accumulator / denominator, 0.0),
        mask=value_offset < VALUE_DIM,
    )
    tl.store(
        lse + query_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )


@triton.jit(
    do_not_specialize=["query_len", "union_width"],
    do_not_specialize_on_alignment=["query_len", "union_width"],
)
def _dense_page_remove_union_kernel(
    q,
    cache_indices,
    page_sum_k,
    page_sum_v,
    page_counts,
    union_pages,
    union_counts,
    full_out,
    full_lse,
    residual_out,
    residual_lse,
    query_len,
    union_width,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    TILES_PER_KV_ROW: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    """Remove one shared tile union from every row's summary attention."""
    sequence = tl.program_id(0).to(tl.int64)
    sequences_per_batch = KV_HEADS * TILES_PER_KV_ROW
    batch = sequence // sequences_per_batch
    within_batch = sequence - batch * sequences_per_batch
    kv_head = within_batch // TILES_PER_KV_ROW
    query_tile = within_batch - kv_head * TILES_PER_KV_ROW
    query_begin = query_tile * QUERY_TILE
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_row = cache_batch * KV_HEADS + kv_head

    row = tl.arange(0, BLOCK_M)
    group_head = row // QUERY_TILE
    query_offset = query_begin + row % QUERY_TILE
    query_valid = (group_head < KV_GROUP_SIZE) & (query_offset < query_len)
    query_head = kv_head * KV_GROUP_SIZE + group_head
    query_row = (batch * QUERY_HEADS + query_head) * query_len + query_offset
    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=query_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
        other=0.0,
    )
    total_output = tl.load(
        full_out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    total_lse = tl.load(full_lse + query_row, mask=query_valid, other=0.0).to(
        tl.float32
    )
    total_lse_log2 = total_lse * 1.4426950408889634
    selected_mass = tl.zeros((BLOCK_M,), tl.float32)
    selected_numerator = tl.zeros((BLOCK_M, VALUE_BLOCK_DIM), tl.float32)
    selected_count = tl.load(union_counts + sequence).to(tl.int32)
    page_offset = tl.arange(0, BLOCK_N)
    for page_begin in tl.range(0, union_width, BLOCK_N, num_stages=1):
        rank = page_begin + page_offset
        page = tl.load(
            union_pages + sequence * union_width + rank,
            mask=rank < selected_count,
            other=-1,
        ).to(tl.int64)
        page_valid = (rank < selected_count) & (page >= 0) & (page < PAGE_CAPACITY)
        safe_page = tl.where(page_valid, page, 0)
        count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + safe_page,
            mask=page_valid,
            other=0,
        ).to(tl.float32)
        page_valid &= count > 0.0
        safe_count = tl.maximum(count, 1.0)
        key_sums = tl.load(
            page_sum_k
            + (kv_row * PAGE_CAPACITY + safe_page[:, None]) * HEAD_DIM
            + head_offset[None, :],
            mask=page_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
            other=0.0,
        )
        value_sums = tl.load(
            page_sum_v
            + (kv_row * PAGE_CAPACITY + safe_page[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=page_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
            other=0.0,
        )
        page_keys = (key_sums.to(tl.float32) / safe_count[:, None]).to(queries.dtype)
        page_values = (value_sums.to(tl.float32) / safe_count[:, None]).to(
            value_sums.dtype
        )
        scores = (
            SCALE_LOG2 * tl.dot(queries, tl.trans(page_keys), out_dtype=tl.float32)
            + tl.log2(safe_count)[None, :]
        )
        valid_score = query_valid[:, None] & page_valid[None, :]
        weights = tl.where(
            valid_score,
            tl.math.exp2(scores - total_lse_log2[:, None]),
            0.0,
        )
        selected_mass += tl.sum(weights, axis=1)
        selected_numerator += tl.dot(
            weights.to(page_values.dtype), page_values, out_dtype=tl.float32
        )

    residual_mass = 1.0 - selected_mass
    has_mass = query_valid & (residual_mass > 1.0e-7)
    residual = (total_output - selected_numerator) / tl.maximum(
        residual_mass[:, None], 1.0e-7
    )
    tl.store(
        residual_out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(has_mass[:, None], residual, 0.0),
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
    )
    tl.store(
        residual_lse + query_row,
        tl.where(has_mass, total_lse + tl.log(residual_mass), -float("inf")),
        mask=query_valid,
    )


def _build_dense_page_tile_unions(
    selected: torch.Tensor,
    page_counts: torch.Tensor,
    cache_indices: torch.Tensor,
    *,
    kv_heads: int,
    kv_group_size: int,
    query_tile: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate every GQA/query-tile page set into a dense left-packed table."""
    batch, query_heads, query_len, top_pages = selected.shape
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("tile-union query/KV head grouping is inconsistent")
    tile_count = triton.cdiv(query_len, query_tile)
    padded_query_len = tile_count * query_tile
    if padded_query_len != query_len:
        selected = F.pad(
            selected,
            (0, 0, 0, padded_query_len - query_len),
            value=-1,
        )
    candidates = (
        selected.view(
            batch,
            kv_heads,
            kv_group_size,
            tile_count,
            query_tile,
            top_pages,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(
            batch * kv_heads * tile_count,
            kv_group_size * query_tile * top_pages,
        )
    )
    sequences = int(candidates.size(0))
    sequence = torch.arange(sequences, device=selected.device)
    sequence_batch = torch.div(sequence, kv_heads * tile_count, rounding_mode="floor")
    sequence_kv_head = torch.div(
        sequence % (kv_heads * tile_count),
        tile_count,
        rounding_mode="floor",
    )
    cache_batch = cache_indices.index_select(0, sequence_batch).long()
    sequence_page_counts = page_counts[cache_batch, sequence_kv_head]
    page_capacity = int(page_counts.size(2))
    candidate_valid = (candidates >= 0) & (candidates < page_capacity)
    safe_candidate = candidates.clamp(min=0, max=max(page_capacity - 1, 0)).long()
    candidate_valid &= torch.gather(sequence_page_counts, 1, safe_candidate) > 0
    sortable = torch.where(
        candidate_valid,
        candidates,
        torch.full_like(candidates, page_capacity),
    )
    sorted_pages = sortable.sort(dim=-1).values
    unique = sorted_pages < page_capacity
    unique[:, 1:] &= sorted_pages[:, 1:] != sorted_pages[:, :-1]
    union_counts = unique.sum(dim=-1, dtype=torch.int32)
    max_union = max(1, int(union_counts.max().item()))
    union_pages = torch.full(
        (sequences, max_union),
        -1,
        dtype=torch.int32,
        device=selected.device,
    )
    ranks = unique.cumsum(dim=-1, dtype=torch.int32) - 1
    rows = torch.arange(sequences, device=selected.device)[:, None].expand_as(unique)
    union_pages[rows[unique], ranks[unique].long()] = sorted_pages[unique].to(
        torch.int32
    )
    return union_pages, union_counts


def _aiter_indexed_tile_union_attention(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    cache_indices: torch.Tensor,
    union_pages: torch.Tensor,
    union_counts: torch.Tensor,
    *,
    kv_group_size: int,
    query_tile: int,
    scale: float,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a shared tile union through AITER's one-token paged-KV kernel.

    Every chronological leaf is exposed as a physical page of size one.  The
    logical LoD page table supplies the indexed leaf IDs, so K/V are never
    copied or expanded into inline 16-token pages.  Until the local AITER
    sentinel specialization is available, unused ``-1`` page lanes are
    compacted out of the temporary page table before launch.
    """
    try:
        from aiter.ops.mha import mha_batch_prefill_func
    except ImportError as error:
        raise RuntimeError(
            "indexed tile-union attention requires an AITER installation"
        ) from error

    batch, query_heads, query_len, head_dim = q.shape
    cache_batch_size, kv_heads, leaf_capacity, key_dim = leaf_k.shape
    value_dim = int(leaf_v.size(-1))
    if key_dim != head_dim or value_dim != head_dim:
        raise ValueError("indexed AITER attention requires equal Q/K/V dimensions")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("indexed AITER query/KV head grouping is inconsistent")
    if leaf_k.dtype not in (torch.float16, torch.bfloat16) or (
        leaf_v.dtype != leaf_k.dtype
    ):
        raise TypeError("indexed AITER tile unions currently require FP16/BF16 K/V")
    tile_count = triton.cdiv(query_len, query_tile)
    sequences = batch * kv_heads * tile_count
    if tuple(union_pages.shape[:1]) != (sequences,) or tuple(union_counts.shape) != (
        sequences,
    ):
        raise ValueError("indexed AITER union metadata has the wrong sequence count")

    table_begin = None
    if timing_events is not None:
        table_begin = torch.cuda.Event(enable_timing=True)
        table_begin.record()
    sequence = torch.arange(sequences, device=q.device)
    sequence_batch = torch.div(sequence, kv_heads * tile_count, rounding_mode="floor")
    sequence_kv_head = torch.div(
        sequence % (kv_heads * tile_count),
        tile_count,
        rounding_mode="floor",
    )
    cache_batch = cache_indices.index_select(0, sequence_batch).long()
    union_rank = torch.arange(int(union_pages.size(1)), device=q.device)
    valid_union = union_rank[None, :] < union_counts[:, None]
    safe_page = union_pages.clamp(min=0, max=int(page_indices.size(2)) - 1).long()
    indexed_leaves = page_indices[
        cache_batch[:, None], sequence_kv_head[:, None], safe_page
    ]
    valid_leaf = (
        valid_union[:, :, None]
        & (indexed_leaves >= 0)
        & (indexed_leaves < leaf_capacity)
    )
    token_counts = valid_leaf.sum(dim=(1, 2), dtype=torch.int32)
    physical_base = ((cache_batch * kv_heads + sequence_kv_head) * leaf_capacity).to(
        torch.int64
    )
    physical_leaf = physical_base[:, None, None] + indexed_leaves.to(torch.int64)
    kv_page_indices = physical_leaf[valid_leaf].to(torch.int32).contiguous()
    kv_indptr = F.pad(token_counts.cumsum(0), (1, 0)).to(torch.int32)

    query_lengths = torch.full(
        (tile_count,), query_tile, dtype=torch.int32, device=q.device
    )
    query_lengths[-1] = query_len - (tile_count - 1) * query_tile
    query_lengths = query_lengths.repeat(batch * kv_heads)
    qo_indptr = F.pad(query_lengths.cumsum(0), (1, 0)).to(torch.int32)
    packed_q = (
        q.view(batch, kv_heads, kv_group_size, query_len, head_dim)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch * kv_heads * query_len, kv_group_size, head_dim)
    )
    token_k = leaf_k.reshape(
        cache_batch_size * kv_heads * leaf_capacity, 1, head_dim
    ).unsqueeze(2)
    token_v = leaf_v.reshape(
        cache_batch_size * kv_heads * leaf_capacity, 1, value_dim
    ).unsqueeze(2)
    last_page_lens = torch.ones(sequences, dtype=torch.int32, device=q.device)
    # AITER accepts an upper bound here and uses ``seqlen_k`` for each actual
    # sequence.  The fixed union width avoids a device-to-host synchronization
    # on token_counts for every prefill chunk/layer.
    max_seqlen_k = max(1, int(union_pages.size(1)) * int(page_indices.size(3)))
    if timing_events is not None:
        table_end = torch.cuda.Event(enable_timing=True)
        table_end.record()
        if table_begin is None:
            raise AssertionError("indexed page-table timing start is missing")
        timing_events.setdefault("indexed_table", []).append((table_begin, table_end))
        aiter_begin = torch.cuda.Event(enable_timing=True)
        aiter_begin.record()
    packed_out, packed_lse = mha_batch_prefill_func(
        packed_q,
        token_k,
        token_v,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        query_tile,
        max_seqlen_k,
        softmax_scale=float(scale),
        causal=False,
        return_lse=True,
        kv_last_page_lens=last_page_lens,
        seqlen_k=token_counts,
    )
    if timing_events is not None:
        aiter_end = torch.cuda.Event(enable_timing=True)
        aiter_end.record()
        timing_events.setdefault("indexed_aiter", []).append((aiter_begin, aiter_end))
        unpack_begin = torch.cuda.Event(enable_timing=True)
        unpack_begin.record()
    exact = (
        packed_out.view(batch, kv_heads, query_len, kv_group_size, value_dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, query_heads, query_len, value_dim)
        .contiguous()
    )
    exact_lse = (
        # AITER returns varlen LSE head-major as [query_heads_per_sequence,
        # total_packed_queries], unlike its token-major attention output.
        packed_lse.view(kv_group_size, batch, kv_heads, query_len)
        .permute(1, 2, 0, 3)
        .reshape(batch, query_heads, query_len)
        .contiguous()
    )
    if timing_events is not None:
        unpack_end = torch.cuda.Event(enable_timing=True)
        unpack_end.record()
        timing_events.setdefault("indexed_unpack", []).append(
            (unpack_begin, unpack_end)
        )
    return exact, exact_lse


def dense_page_summary_attention(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    next_page: torch.Tensor,
    *,
    cache_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    top_pages: int = 8,
    block_m: int = 16,
    block_n: int = 64,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    split_kernels: bool = False,
    indexed_aiter_union: bool = False,
    union_query_tile: int = 16,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attend to all page summaries and exactly replace the best pages."""
    tensors = (
        q,
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        next_page,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("dense page-summary attention requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    cache_batch = int(leaf_k.size(0))
    kv_heads = int(leaf_k.size(1))
    value_dim = int(leaf_v.size(-1))
    page_capacity = int(page_indices.size(2))
    page_size = int(page_indices.size(3))
    if cache_indices is None:
        if cache_batch != batch:
            raise ValueError("cache indices are required for a shared page pool")
        cache_indices = torch.arange(batch, dtype=torch.long, device=q.device)
    if tuple(cache_indices.shape) != (batch,):
        raise ValueError("cache indices must contain one row per query batch")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if page_size != 16:
        raise ValueError("dense page-summary attention requires 16-token pages")
    if top_pages not in (1, 2, 4, 8):
        raise ValueError("dense page attention supports 1, 2, 4, or 8 exact pages")
    if union_query_tile <= 0 or union_query_tile & (union_query_tile - 1):
        raise ValueError("dense page union query tile must be a positive power of two")
    if (
        block_m <= 0
        or block_m & (block_m - 1)
        or block_n <= 0
        or block_n & (block_n - 1)
    ):
        raise ValueError("dense page tiles must be positive powers where required")
    group_block = triton.next_power_of_2(kv_group_size)
    if block_m < group_block or block_m % group_block:
        raise ValueError("dense page M tile must contain a complete GQA group")
    block_q = block_m // group_block
    if tuple(page_sum_k.shape) != (cache_batch, kv_heads, page_capacity, head_dim):
        raise ValueError("page key summaries do not match indexed page storage")
    if tuple(page_sum_v.shape) != (cache_batch, kv_heads, page_capacity, value_dim):
        raise ValueError("page value summaries do not match indexed page storage")
    if tuple(page_counts.shape) != (cache_batch, kv_heads, page_capacity):
        raise ValueError("page counts do not match indexed page storage")
    if tuple(next_page.shape) != (cache_batch, kv_heads):
        raise ValueError("next-page counters do not match the cache")
    if int(leaf_k.size(-1)) != head_dim or leaf_k.shape[:3] != leaf_v.shape[:3]:
        raise ValueError("flat exact K/V storage is inconsistent")
    if page_sum_k.dtype != q.dtype or page_sum_v.dtype != leaf_v.dtype:
        raise TypeError("dense page summaries must use the model K/V dtype")

    q = q.contiguous()
    page_indices = page_indices.contiguous()
    if split_kernels and indexed_aiter_union:
        raise ValueError("split dense kernels and indexed AITER unions are exclusive")
    if split_kernels:
        return _dense_page_split_attention_impl(
            q,
            leaf_k,
            leaf_v,
            page_indices,
            page_sum_k,
            page_sum_v,
            page_counts,
            next_page,
            cache_indices=cache_indices.contiguous(),
            kv_group_size=kv_group_size,
            scale=scale,
            top_pages=top_pages,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
            timing_events=timing_events,
        )
    rows = batch * query_heads * query_len
    residual = torch.empty(
        batch, query_heads, query_len, value_dim, dtype=q.dtype, device=q.device
    )
    residual_lse = torch.empty(
        batch, query_heads, query_len, dtype=torch.float32, device=q.device
    )
    selected = torch.empty(
        batch,
        query_heads,
        query_len,
        top_pages,
        dtype=torch.int32,
        device=q.device,
    )
    exact = torch.empty_like(residual)
    exact_lse = torch.empty_like(residual_lse)
    begin = None
    phase_begin = None
    if timing_events is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
        phase_begin = torch.cuda.Event(enable_timing=True)
        phase_begin.record()
    _dense_page_summary_attention_kernel[
        (batch, kv_heads, triton.cdiv(query_len, block_q))
    ](
        q,
        cache_indices.contiguous(),
        page_sum_k,
        page_sum_v,
        page_counts,
        next_page,
        selected,
        residual,
        residual_lse,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        TOP_PAGES=top_pages,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        GROUP_BLOCK=group_block,
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        REMOVE_SELECTED=not indexed_aiter_union,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        phase_end = torch.cuda.Event(enable_timing=True)
        phase_end.record()
        if phase_begin is None:
            raise AssertionError("dense summary/select timing start is missing")
        timing_events.setdefault("summary_select", []).append((phase_begin, phase_end))
        phase_begin = torch.cuda.Event(enable_timing=True)
        phase_begin.record()
    if indexed_aiter_union:
        union_pages, union_counts = _build_dense_page_tile_unions(
            selected,
            page_counts,
            cache_indices,
            kv_heads=kv_heads,
            kv_group_size=kv_group_size,
            query_tile=union_query_tile,
        )
        if timing_events is not None:
            phase_end = torch.cuda.Event(enable_timing=True)
            phase_end.record()
            if phase_begin is None:
                raise AssertionError("dense union-build timing start is missing")
            timing_events.setdefault("union_build", []).append((phase_begin, phase_end))
            phase_begin = torch.cuda.Event(enable_timing=True)
            phase_begin.record()
        tile_count = triton.cdiv(query_len, union_query_tile)
        union_rows = triton.next_power_of_2(kv_group_size * union_query_tile)
        _dense_page_remove_union_kernel[(int(union_pages.size(0)),)](
            q,
            cache_indices,
            page_sum_k,
            page_sum_v,
            page_counts,
            union_pages,
            union_counts,
            residual,
            residual_lse,
            residual,
            residual_lse,
            query_len,
            int(union_pages.size(1)),
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            QUERY_TILE=union_query_tile,
            TILES_PER_KV_ROW=tile_count,
            PAGE_CAPACITY=page_capacity,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
            VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
            BLOCK_M=union_rows,
            BLOCK_N=min(64, triton.next_power_of_2(int(union_pages.size(1)))),
            SCALE_LOG2=float(scale) * math.log2(math.e),
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        if timing_events is not None:
            phase_end = torch.cuda.Event(enable_timing=True)
            phase_end.record()
            if phase_begin is None:
                raise AssertionError("dense union-removal timing start is missing")
            timing_events.setdefault("summary_removal", []).append(
                (phase_begin, phase_end)
            )
            phase_begin = torch.cuda.Event(enable_timing=True)
            phase_begin.record()
        exact, exact_lse = _aiter_indexed_tile_union_attention(
            q,
            leaf_k,
            leaf_v,
            page_indices,
            cache_indices,
            union_pages,
            union_counts,
            kv_group_size=kv_group_size,
            query_tile=union_query_tile,
            scale=scale,
            timing_events=timing_events,
        )
    else:
        _dense_page_exact_attention_kernel[(rows,)](
            q,
            cache_indices.contiguous(),
            leaf_k,
            leaf_v,
            page_indices,
            page_counts,
            selected,
            exact,
            exact_lse,
            query_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            PAGE_CAPACITY=page_capacity,
            LEAF_CAPACITY=int(leaf_k.size(2)),
            LEAF_K_BATCH_STRIDE=leaf_k.stride(0),
            LEAF_K_HEAD_STRIDE=leaf_k.stride(1),
            LEAF_K_TOKEN_STRIDE=leaf_k.stride(2),
            LEAF_V_BATCH_STRIDE=leaf_v.stride(0),
            LEAF_V_HEAD_STRIDE=leaf_v.stride(1),
            LEAF_V_TOKEN_STRIDE=leaf_v.stride(2),
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
            VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
            PAGE_SIZE=page_size,
            TOP_PAGES=top_pages,
            SCALE_LOG2=float(scale) * math.log2(math.e),
            num_warps=2,
            waves_per_eu=waves_per_eu,
        )
    if timing_events is not None:
        phase_end = torch.cuda.Event(enable_timing=True)
        phase_end.record()
        if phase_begin is None:
            raise AssertionError("dense exact timing start is missing")
        timing_events.setdefault("exact_pages", []).append((phase_begin, phase_end))
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if begin is None:
            raise AssertionError("dense page timing start is missing")
        timing_events.setdefault("total", []).append((begin, end))
    return residual, residual_lse, exact, exact_lse, selected

    # Experimental decomposed form retained for phase-level profiling.  It is
    # slower end-to-end because the QK field must be scanned a second time.
    summary = torch.empty(
        batch,
        query_heads,
        query_len,
        value_dim,
        dtype=torch.float32,
        device=q.device,
    )
    selected_scores = torch.empty_like(selected, dtype=torch.float32)
    selected_weighted_lse = torch.empty_like(residual_lse)
    # Keep the regular summary pass at the requested cooperative tile size.
    # A fixed M=16 tile created four times as many programs for Qwen3.5 GQA
    # and left the kernel far below the occupancy/throughput sweet spot.
    attention_block_m = block_m
    attention_block_q = attention_block_m // group_block
    _dense_page_regular_attention_kernel[
        (batch, kv_heads, triton.cdiv(query_len, attention_block_q))
    ](
        q,
        cache_indices.contiguous(),
        page_sum_k,
        page_sum_v,
        page_counts,
        next_page,
        summary,
        residual_lse,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_Q=attention_block_q,
        BLOCK_M=attention_block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    _dense_page_topk_kernel[(batch, kv_heads, triton.cdiv(query_len, block_q))](
        q,
        cache_indices.contiguous(),
        page_sum_k,
        page_counts,
        next_page,
        selected,
        selected_scores,
        selected_weighted_lse,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        HEAD_DIM=head_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        TOP_PAGES=top_pages,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    _dense_page_remove_selected_kernel[(rows,)](
        summary,
        residual_lse,
        selected_weighted_lse,
        cache_indices.contiguous(),
        page_sum_v,
        page_counts,
        selected,
        selected_scores,
        residual,
        residual_lse,
        QUERY_LEN=query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        VALUE_DIM=value_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        TOP_PAGES=top_pages,
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    _dense_page_exact_attention_kernel[(rows,)](
        q,
        cache_indices.contiguous(),
        leaf_k,
        leaf_v,
        page_indices,
        page_counts,
        selected,
        exact,
        exact_lse,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        LEAF_CAPACITY=int(leaf_k.size(2)),
        LEAF_K_BATCH_STRIDE=leaf_k.stride(0),
        LEAF_K_HEAD_STRIDE=leaf_k.stride(1),
        LEAF_K_TOKEN_STRIDE=leaf_k.stride(2),
        LEAF_V_BATCH_STRIDE=leaf_v.stride(0),
        LEAF_V_HEAD_STRIDE=leaf_v.stride(1),
        LEAF_V_TOKEN_STRIDE=leaf_v.stride(2),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=page_size,
        TOP_PAGES=top_pages,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if begin is None:
            raise AssertionError("dense page timing start is missing")
        timing_events.setdefault("total", []).append((begin, end))
    return residual, residual_lse, exact, exact_lse, selected


def _dense_page_split_attention_impl(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    next_page: torch.Tensor,
    *,
    cache_indices: torch.Tensor,
    kv_group_size: int,
    scale: float,
    top_pages: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    waves_per_eu: int,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Profileable decomposed dense-summary, selection, and review path."""
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(leaf_k.size(1))
    value_dim = int(leaf_v.size(-1))
    page_capacity = int(page_indices.size(2))
    page_size = int(page_indices.size(3))
    rows = batch * query_heads * query_len
    group_block = triton.next_power_of_2(kv_group_size)
    block_q = block_m // group_block
    # Use the configured cooperative tile here too. M=16 launches four times
    # as many programs as M=64 for Qwen3.5's four-query-head GQA groups.
    weighted_lse = torch.empty(
        batch, query_heads, query_len, dtype=torch.float32, device=q.device
    )
    residual = torch.empty(
        batch, query_heads, query_len, value_dim, dtype=q.dtype, device=q.device
    )
    residual_lse = torch.empty(
        batch, query_heads, query_len, dtype=torch.float32, device=q.device
    )
    selected = torch.empty(
        batch,
        query_heads,
        query_len,
        top_pages,
        dtype=torch.int32,
        device=q.device,
    )
    selected_scores = torch.empty_like(selected, dtype=torch.float32)
    exact = torch.empty_like(residual)
    exact_lse = torch.empty_like(residual_lse)

    def start_phase() -> torch.cuda.Event | None:
        if timing_events is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def end_phase(name: str, start: torch.cuda.Event | None) -> None:
        if timing_events is None or start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        timing_events.setdefault(name, []).append((start, end))

    total_start = start_phase()
    phase_start = start_phase()
    summary_prepare_start = start_phase()
    mapped_page_sum_k = page_sum_k.index_select(0, cache_indices.long())
    mapped_page_sum_v = page_sum_v.index_select(0, cache_indices.long())
    mapped_page_counts = page_counts.index_select(0, cache_indices.long())
    safe_counts = mapped_page_counts.clamp_min(1).unsqueeze(-1)
    mean_page_k = mapped_page_sum_k.float().div_(safe_counts).to(q.dtype).contiguous()
    flash_page_v = mapped_page_sum_v.contiguous()
    end_phase("summary_prepare", summary_prepare_start)
    summary_flash_start = start_phase()
    summary, summary_lse, *_ = (
        torch.ops.aten._scaled_dot_product_flash_attention.default(
            q,
            mean_page_k,
            flash_page_v,
            0.0,
            False,
            False,
            scale=float(scale),
        )
    )
    end_phase("summary_flash", summary_flash_start)
    end_phase("summary_attention", phase_start)

    phase_start = start_phase()
    _dense_page_topk_kernel[(batch, kv_heads, triton.cdiv(query_len, block_q))](
        q,
        cache_indices,
        page_sum_k,
        page_counts,
        next_page,
        selected,
        selected_scores,
        weighted_lse,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        HEAD_DIM=head_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        TOP_PAGES=top_pages,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_Q=block_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    end_phase("page_topk", phase_start)

    phase_start = start_phase()
    _dense_page_remove_selected_kernel[(rows,)](
        summary,
        summary_lse,
        weighted_lse,
        cache_indices,
        page_sum_v,
        page_counts,
        selected,
        selected_scores,
        residual,
        residual_lse,
        QUERY_LEN=query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        VALUE_DIM=value_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        TOP_PAGES=top_pages,
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    end_phase("summary_removal", phase_start)

    phase_start = start_phase()
    _dense_page_exact_attention_kernel[(rows,)](
        q,
        cache_indices,
        leaf_k,
        leaf_v,
        page_indices,
        page_counts,
        selected,
        exact,
        exact_lse,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        LEAF_CAPACITY=int(leaf_k.size(2)),
        LEAF_K_BATCH_STRIDE=leaf_k.stride(0),
        LEAF_K_HEAD_STRIDE=leaf_k.stride(1),
        LEAF_K_TOKEN_STRIDE=leaf_k.stride(2),
        LEAF_V_BATCH_STRIDE=leaf_v.stride(0),
        LEAF_V_HEAD_STRIDE=leaf_v.stride(1),
        LEAF_V_TOKEN_STRIDE=leaf_v.stride(2),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=page_size,
        TOP_PAGES=top_pages,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        num_warps=2,
        waves_per_eu=waves_per_eu,
    )
    end_phase("exact_pages", phase_start)
    end_phase("total", total_start)
    return residual, residual_lse, exact, exact_lse, selected


@triton.jit(
    do_not_specialize=[
        "LEAF_CAPACITY",
        "LEAF_K_BATCH_STRIDE",
        "LEAF_K_HEAD_STRIDE",
        "LEAF_K_TOKEN_STRIDE",
        "LEAF_V_BATCH_STRIDE",
        "LEAF_V_HEAD_STRIDE",
        "LEAF_V_TOKEN_STRIDE",
        "query_len",
    ],
    do_not_specialize_on_alignment=[
        "LEAF_CAPACITY",
        "LEAF_K_BATCH_STRIDE",
        "LEAF_K_HEAD_STRIDE",
        "LEAF_K_TOKEN_STRIDE",
        "LEAF_V_BATCH_STRIDE",
        "LEAF_V_HEAD_STRIDE",
        "LEAF_V_TOKEN_STRIDE",
        "query_len",
    ],
)
def _query_major_residual_page_attention_kernel(
    q,
    state_k,
    state_v,
    state_counts,
    mla_norm_weight,
    cache_indices,
    page_k,
    page_v,
    page_indices,
    leaf_k,
    leaf_v,
    quantized_leaf_k,
    quantized_leaf_v,
    page_k_scales,
    page_v_scales,
    page_quantized_counts,
    page_sum_k,
    page_sum_v,
    quantized_page_sum_k,
    quantized_page_sum_v,
    page_sum_k_scales,
    page_sum_v_scales,
    page_counts,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    top_slots,
    query_len,
    out,
    lse,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    PAGE_BLOCK_N: tl.constexpr,
    LEAF_K_BATCH_STRIDE,
    LEAF_K_HEAD_STRIDE,
    LEAF_K_TOKEN_STRIDE,
    LEAF_V_BATCH_STRIDE,
    LEAF_V_HEAD_STRIDE,
    LEAF_V_TOKEN_STRIDE,
    LEAF_CAPACITY,
    QUANT_GROUP_SIZE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    QUANTIZED_SUMMARIES: tl.constexpr,
    INDEXED: tl.constexpr,
    ROUTE_PARALLEL: tl.constexpr,
    MLA_LATENT_DIM: tl.constexpr,
    MLA_NORM_EPS: tl.constexpr,
):
    """Open one page per routed slot and summarize its disjoint residual."""
    query_row = tl.program_id(0).to(tl.int64)
    active_route = tl.program_id(1).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = cache_batch * KV_HEADS + kv_head

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    page_offset = tl.arange(0, PAGE_BLOCK_N)
    token_offset = tl.arange(0, PAGE_SIZE)
    query = tl.load(
        q + query_row * HEAD_DIM + head_offset,
        mask=head_offset < HEAD_DIM,
        other=0.0,
    )
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_BLOCK_DIM,), tl.float32)

    route_begin = active_route if ROUTE_PARALLEL else 0
    route_end = active_route + 1 if ROUTE_PARALLEL else ROUTE_COUNT
    for route in tl.range(route_begin, route_end, num_stages=1):
        routed_slot = tl.load(top_slots + query_row * ROUTE_COUNT + route).to(tl.int64)
        valid_slot = (routed_slot >= 0) & (routed_slot < STATE_CAPACITY)
        slot = tl.where(valid_slot, routed_slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0,
        ).to(tl.int32)
        slot_page_count = (key_count + PAGE_SIZE - 1) // PAGE_SIZE
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        selected_score = tl.full((), -float("inf"), tl.float32)
        selected_page = tl.full((), 0, tl.int64)
        single_page = valid_slot & (slot_page_count == 1)
        if HASH_PROBES == 0:
            first_page = tl.load(
                page_table,
                mask=single_page,
                other=0,
            ).to(tl.int64)
        else:
            first_page = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                kv_row,
                slot,
                0,
                single_page,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        single_page &= (first_page >= 0) & (first_page < PAGE_CAPACITY)
        selected_score = tl.where(single_page, float("inf"), selected_score)
        selected_page = tl.where(single_page, first_page, selected_page)
        scan_page_count = tl.where(single_page, 0, slot_page_count)

        for page_begin in tl.range(0, scan_page_count, PAGE_BLOCK_N, num_stages=1):
            page_ordinal = page_begin + page_offset
            valid_page = page_ordinal < scan_page_count
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal, mask=valid_page, other=0
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid_page,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            valid_page &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
            page_id = tl.where(valid_page, page_id, 0)
            count = tl.load(
                page_counts + kv_row * PAGE_CAPACITY + page_id,
                mask=valid_page,
                other=1,
            ).to(tl.float32)
            if QUANTIZED_SUMMARIES:
                key_sum_codes = tl.load(
                    quantized_page_sum_k
                    + (kv_row * PAGE_CAPACITY + page_id[:, None]) * HEAD_DIM
                    + head_offset[None, :],
                    mask=valid_page[:, None] & (head_offset[None, :] < HEAD_DIM),
                    other=0,
                ).to(tl.float32)
                key_sum_scales = tl.load(
                    page_sum_k_scales
                    + (kv_row * PAGE_CAPACITY + page_id[:, None])
                    * (HEAD_DIM // QUANT_GROUP_SIZE)
                    + head_offset[None, :] // QUANT_GROUP_SIZE,
                    mask=valid_page[:, None] & (head_offset[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                key_sums = key_sum_codes * key_sum_scales
            else:
                key_sums = tl.load(
                    page_sum_k
                    + (kv_row * PAGE_CAPACITY + page_id[:, None]) * HEAD_DIM
                    + head_offset[None, :],
                    mask=valid_page[:, None] & (head_offset[None, :] < HEAD_DIM),
                    other=0.0,
                )
            page_keys = key_sums.to(tl.float32) / count[:, None]
            if MLA_LATENT_DIM > 0:
                # Reproduce DeepSeek's latent RMSNorm ordering exactly:
                # average raw compressed latents, round the unit-RMS vector
                # to BF16, then apply the learned gain.  The appended RoPE
                # channels remain an ordinary arithmetic mean.
                page_keys = page_keys.to(tl.bfloat16)
                latent_mask = head_offset < MLA_LATENT_DIM
                latent_values = tl.where(
                    latent_mask[None, :], page_keys.to(tl.float32), 0.0
                )
                inverse_rms = tl.rsqrt(
                    tl.sum(latent_values * latent_values, axis=1) / MLA_LATENT_DIM
                    + MLA_NORM_EPS
                )
                unit_latent = (page_keys.to(tl.float32) * inverse_rms[:, None]).to(
                    tl.bfloat16
                )
                norm_gain = tl.load(
                    mla_norm_weight + head_offset,
                    mask=latent_mask,
                    other=1.0,
                ).to(tl.bfloat16)
                normalized_latent = (unit_latent * norm_gain[None, :]).to(tl.bfloat16)
                page_keys = tl.where(
                    latent_mask[None, :], normalized_latent, page_keys
                ).to(tl.float32)
            page_scores = SCALE_LOG2 * tl.sum(
                page_keys * query[None, :].to(tl.float32),
                axis=1,
            ) + tl.log2(count)
            page_scores = tl.where(valid_page, page_scores, -float("inf"))
            block_score = tl.max(page_scores, axis=0)
            block_page = tl.max(
                tl.where(page_scores == block_score, page_id, -1), axis=0
            ).to(tl.int64)
            better = block_score > selected_score
            selected_score = tl.where(better, block_score, selected_score)
            selected_page = tl.where(better, block_page, selected_page)

        selected_valid = selected_score > -float("inf")
        selected_count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + selected_page,
            mask=selected_valid,
            other=0,
        ).to(tl.float32)
        state_count = tl.load(
            state_counts + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0,
        ).to(tl.float32)
        residual_count = state_count - selected_count
        if QUANTIZED_SUMMARIES:
            selected_key_sum = tl.load(
                quantized_page_sum_k
                + (kv_row * PAGE_CAPACITY + selected_page) * HEAD_DIM
                + head_offset,
                mask=selected_valid & (head_offset < HEAD_DIM),
                other=0,
            ).to(tl.float32) * tl.load(
                page_sum_k_scales
                + (kv_row * PAGE_CAPACITY + selected_page)
                * (HEAD_DIM // QUANT_GROUP_SIZE)
                + head_offset // QUANT_GROUP_SIZE,
                mask=selected_valid & (head_offset < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            selected_value_sum = tl.load(
                quantized_page_sum_v
                + (kv_row * PAGE_CAPACITY + selected_page) * VALUE_DIM
                + value_offset,
                mask=selected_valid & (value_offset < VALUE_DIM),
                other=0,
            ).to(tl.float32) * tl.load(
                page_sum_v_scales
                + (kv_row * PAGE_CAPACITY + selected_page)
                * (VALUE_DIM // QUANT_GROUP_SIZE)
                + value_offset // QUANT_GROUP_SIZE,
                mask=selected_valid & (value_offset < VALUE_DIM),
                other=0.0,
            ).to(tl.float32)
        else:
            selected_key_sum = tl.load(
                page_sum_k
                + (kv_row * PAGE_CAPACITY + selected_page) * HEAD_DIM
                + head_offset,
                mask=selected_valid & (head_offset < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            selected_value_sum = tl.load(
                page_sum_v
                + (kv_row * PAGE_CAPACITY + selected_page) * VALUE_DIM
                + value_offset,
                mask=selected_valid & (value_offset < VALUE_DIM),
                other=0.0,
            ).to(tl.float32)
        state_key_sum = tl.load(
            state_k + (kv_row * STATE_CAPACITY + slot) * HEAD_DIM + head_offset,
            mask=valid_slot & (head_offset < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        state_value_sum = tl.load(
            state_v + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM + value_offset,
            mask=valid_slot & (value_offset < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)

        if residual_count > 0.0:
            residual_key = (state_key_sum - selected_key_sum) / residual_count
            if MLA_LATENT_DIM > 0:
                residual_key = residual_key.to(tl.bfloat16)
                latent_mask = head_offset < MLA_LATENT_DIM
                latent_values = tl.where(latent_mask, residual_key.to(tl.float32), 0.0)
                inverse_rms = tl.rsqrt(
                    tl.sum(latent_values * latent_values, axis=0) / MLA_LATENT_DIM
                    + MLA_NORM_EPS
                )
                unit_latent = (residual_key.to(tl.float32) * inverse_rms).to(
                    tl.bfloat16
                )
                norm_gain = tl.load(
                    mla_norm_weight + head_offset,
                    mask=latent_mask,
                    other=1.0,
                ).to(tl.bfloat16)
                normalized_latent = (unit_latent * norm_gain).to(tl.bfloat16)
                residual_key = tl.where(
                    latent_mask, normalized_latent, residual_key
                ).to(tl.float32)
            residual_value = (state_value_sum - selected_value_sum) / residual_count
            residual_score = SCALE_LOG2 * tl.sum(
                residual_key * query.to(tl.float32), axis=0
            ) + tl.log2(residual_count)
            new_maximum = tl.maximum(maximum, residual_score)
            correction = tl.math.exp2(maximum - new_maximum)
            probability = tl.math.exp2(residual_score - new_maximum)
            denominator = denominator * correction + probability
            accumulator = accumulator * correction + probability * residual_value
            maximum = new_maximum

        valid_token = selected_valid & (token_offset < selected_count)
        physical_token = (
            kv_row * PAGE_CAPACITY + selected_page
        ) * PAGE_SIZE + token_offset
        if INDEXED:
            leaf_index = tl.load(
                page_indices + physical_token,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            leaf_index = tl.where(valid_token, leaf_index, 0)
            if QUANT_BITS:
                quantized_count = tl.load(
                    page_quantized_counts + kv_row * PAGE_CAPACITY + selected_page
                ).to(tl.int32)
                use_quantized = valid_token & (token_offset < quantized_count)
                if QUANT_BITS == 4:
                    packed_head_offset = head_offset // 2
                    packed_value_offset = value_offset // 2
                    packed_keys = tl.load(
                        quantized_leaf_k
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None])
                        * (HEAD_DIM // 2)
                        + packed_head_offset[None, :],
                        mask=use_quantized[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0,
                    ).to(tl.int32)
                    packed_values = tl.load(
                        quantized_leaf_v
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None])
                        * (VALUE_DIM // 2)
                        + packed_value_offset[None, :],
                        mask=use_quantized[:, None]
                        & (value_offset[None, :] < VALUE_DIM),
                        other=0,
                    ).to(tl.int32)
                    key_shift = (head_offset & 1) * 4
                    value_shift = (value_offset & 1) * 4
                    key_code = ((packed_keys >> key_shift[None, :]) & 15) - 8
                    value_code = ((packed_values >> value_shift[None, :]) & 15) - 8
                else:
                    key_code = tl.load(
                        quantized_leaf_k
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * HEAD_DIM
                        + head_offset[None, :],
                        mask=use_quantized[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0,
                    ).to(tl.int32)
                    value_code = tl.load(
                        quantized_leaf_v
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * VALUE_DIM
                        + value_offset[None, :],
                        mask=use_quantized[:, None]
                        & (value_offset[None, :] < VALUE_DIM),
                        other=0,
                    ).to(tl.int32)
                key_scale = tl.load(
                    page_k_scales
                    + (kv_row * PAGE_CAPACITY + selected_page)
                    * (HEAD_DIM // QUANT_GROUP_SIZE)
                    + head_offset // QUANT_GROUP_SIZE,
                    mask=head_offset < HEAD_DIM,
                    other=0.0,
                ).to(tl.float32)
                value_scale = tl.load(
                    page_v_scales
                    + (kv_row * PAGE_CAPACITY + selected_page)
                    * (VALUE_DIM // QUANT_GROUP_SIZE)
                    + value_offset // QUANT_GROUP_SIZE,
                    mask=value_offset < VALUE_DIM,
                    other=0.0,
                ).to(tl.float32)
                quantized_keys = (
                    selected_key_sum / selected_count
                    + key_code.to(tl.float32) * key_scale
                )
                quantized_values = (
                    selected_value_sum / selected_count
                    + value_code.to(tl.float32) * value_scale
                )
            else:
                use_quantized = tl.full((PAGE_SIZE,), False, tl.int1)
                quantized_keys = tl.zeros((HEAD_BLOCK_DIM,), tl.float32)
                quantized_values = tl.zeros((VALUE_BLOCK_DIM,), tl.float32)
            keys = tl.load(
                leaf_k
                + cache_batch * LEAF_K_BATCH_STRIDE
                + kv_head * LEAF_K_HEAD_STRIDE
                + leaf_index[:, None] * LEAF_K_TOKEN_STRIDE
                + head_offset[None, :],
                mask=(valid_token & ~use_quantized)[:, None]
                & (head_offset[None, :] < HEAD_DIM),
                other=0.0,
            )
            values = tl.load(
                leaf_v
                + cache_batch * LEAF_V_BATCH_STRIDE
                + kv_head * LEAF_V_HEAD_STRIDE
                + leaf_index[:, None] * LEAF_V_TOKEN_STRIDE
                + value_offset[None, :],
                mask=(valid_token & ~use_quantized)[:, None]
                & (value_offset[None, :] < VALUE_DIM),
                other=0.0,
            )
            keys = tl.where(use_quantized[:, None], quantized_keys, keys)
            values = tl.where(use_quantized[:, None], quantized_values, values)
        else:
            keys = tl.load(
                page_k + physical_token[:, None] * HEAD_DIM + head_offset[None, :],
                mask=valid_token[:, None] & (head_offset[None, :] < HEAD_DIM),
                other=0.0,
            )
            values = tl.load(
                page_v + physical_token[:, None] * VALUE_DIM + value_offset[None, :],
                mask=valid_token[:, None] & (value_offset[None, :] < VALUE_DIM),
                other=0.0,
            )
        exact_scores = SCALE_LOG2 * tl.sum(
            keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
        )
        exact_scores = tl.where(valid_token, exact_scores, -float("inf"))
        block_maximum = tl.max(exact_scores, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.where(
            selected_valid,
            tl.math.exp2(maximum - new_maximum),
            1.0,
        )
        probabilities = tl.math.exp2(exact_scores - new_maximum)
        probabilities = tl.where(valid_token, probabilities, 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        maximum = tl.where(selected_valid, new_maximum, maximum)

    output_row = query_row * ROUTE_COUNT + active_route if ROUTE_PARALLEL else query_row
    has_mass = denominator > 0.0
    tl.store(
        out + output_row * VALUE_DIM + value_offset,
        tl.where(has_mass, accumulator / denominator, 0.0),
        mask=value_offset < VALUE_DIM,
    )
    tl.store(
        lse + output_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )


def refine_route_candidates_by_page_mass(
    q: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    candidates: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    page_size: int = 16,
    page_block_n: int = 16,
) -> torch.Tensor:
    """Return query-dependent page-centroid log-mass for candidate slots."""
    tensors = (
        q,
        page_sum_k,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("page-mass route refinement requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(page_sum_k.size(1))
    candidate_count = int(candidates.size(-1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if tuple(candidates.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("route candidates have the wrong query shape")
    if not 8 <= candidate_count <= 128:
        raise ValueError("page-mass refinement requires 8 to 128 candidates")
    if int(page_sum_k.size(-1)) != head_dim:
        raise ValueError("page-summary key dimension differs from the query")
    if tuple(page_counts.shape) != tuple(page_sum_k.shape[:3]):
        raise ValueError("page counts differ from page-summary storage")
    if page_size != 16:
        raise ValueError("page-mass refinement currently requires 16-token pages")
    if page_block_n <= 0 or page_block_n & (page_block_n - 1):
        raise ValueError("page-mass block size must be a power of two")
    rows = batch * query_heads * query_len
    scores = torch.empty(
        batch,
        query_heads,
        query_len,
        candidate_count,
        dtype=torch.float32,
        device=q.device,
    )
    _candidate_page_mass_kernel[(rows, candidate_count)](
        q.contiguous(),
        page_sum_k,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates.contiguous(),
        scores,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=int(page_sum_k.size(2)),
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        CANDIDATE_COUNT=candidate_count,
        SCALE=float(scale),
        PAGE_BLOCK_N=page_block_n,
        num_warps=2,
    )
    return scores


def refine_route_candidates_by_leaf_mass(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    candidates: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    page_size: int = 16,
) -> torch.Tensor:
    """Return exact token-level log-mass for candidate state slots."""
    tensors = (
        q,
        page_k,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("leaf-mass route refinement requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(page_k.size(1))
    candidate_count = int(candidates.size(-1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if tuple(candidates.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("route candidates have the wrong query shape")
    if not 8 <= candidate_count <= 128:
        raise ValueError("leaf-mass refinement requires 8 to 128 candidates")
    if tuple(page_k.shape[-2:]) != (page_size, head_dim):
        raise ValueError("leaf-page geometry differs from the query")
    if tuple(page_counts.shape) != tuple(page_k.shape[:3]):
        raise ValueError("page counts differ from leaf-page storage")
    if page_size != 16:
        raise ValueError("leaf-mass refinement currently requires 16-token pages")
    rows = batch * query_heads * query_len
    scores = torch.empty(
        batch,
        query_heads,
        query_len,
        candidate_count,
        dtype=torch.float32,
        device=q.device,
    )
    _candidate_leaf_mass_kernel[(rows, candidate_count)](
        q.contiguous(),
        page_k,
        page_k,
        page_counts,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates.contiguous(),
        scores,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=int(page_k.size(2)),
        LEAF_CAPACITY=1,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        CANDIDATE_COUNT=candidate_count,
        SCALE=float(scale),
        VIRTUAL=False,
        num_warps=4,
    )
    return scores


def refine_route_candidates_by_virtual_leaf_mass(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    page_indices: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    candidates: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    page_size: int = 16,
) -> torch.Tensor:
    """Return exact candidate log-mass from virtual prompt-leaf pages."""
    tensors = (
        q,
        leaf_k,
        page_indices.contiguous(),
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("virtual leaf-mass refinement requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(leaf_k.size(1))
    candidate_count = int(candidates.size(-1))
    page_capacity = int(page_indices.size(2))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if tuple(candidates.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("route candidates have the wrong query shape")
    if not 8 <= candidate_count <= 128:
        raise ValueError("leaf-mass refinement requires 8 to 128 candidates")
    if int(leaf_k.size(-1)) != head_dim:
        raise ValueError("virtual leaf dimension differs from the query")
    if tuple(page_indices.shape[-1:]) != (page_size,):
        raise ValueError("virtual page geometry differs from the page size")
    if tuple(page_counts.shape) != tuple(page_indices.shape[:3]):
        raise ValueError("page counts differ from virtual page storage")
    if page_size != 16:
        raise ValueError("leaf-mass refinement currently requires 16-token pages")
    rows = batch * query_heads * query_len
    scores = torch.empty(
        batch,
        query_heads,
        query_len,
        candidate_count,
        dtype=torch.float32,
        device=q.device,
    )
    _candidate_leaf_mass_kernel[(rows, candidate_count)](
        q.contiguous(),
        leaf_k,
        leaf_k,
        page_indices.contiguous(),
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates.contiguous(),
        scores,
        query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=page_capacity,
        LEAF_CAPACITY=int(leaf_k.size(2)),
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        CANDIDATE_COUNT=candidate_count,
        SCALE=float(scale),
        VIRTUAL=True,
        num_warps=4,
    )
    return scores


def refine_route_candidates_by_virtual_leaf_output(
    q: torch.Tensor,
    baseline_output: torch.Tensor,
    baseline_lse: torch.Tensor,
    candidate_coarse_lse: torch.Tensor,
    state_sum_v: torch.Tensor,
    state_counts: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    candidates: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    page_size: int = 16,
) -> torch.Tensor:
    """Return error reduction toward the all-candidate exact-output target."""
    tensors = (
        q,
        baseline_output,
        baseline_lse,
        candidate_coarse_lse,
        state_sum_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("virtual leaf-output refinement requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(leaf_k.size(1))
    value_dim = int(leaf_v.size(-1))
    candidate_count = int(candidates.size(-1))
    page_capacity = int(page_indices.size(2))
    expected_query_shape = (batch, query_heads, query_len)
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if tuple(candidates.shape[:3]) != expected_query_shape:
        raise ValueError("route candidates have the wrong query shape")
    if tuple(baseline_output.shape) != (*expected_query_shape, value_dim):
        raise ValueError("baseline output has the wrong shape")
    if tuple(baseline_lse.shape) != expected_query_shape:
        raise ValueError("baseline LSE has the wrong shape")
    if tuple(candidate_coarse_lse.shape) != tuple(candidates.shape):
        raise ValueError("candidate coarse scores have the wrong shape")
    if not 8 <= candidate_count <= 32:
        raise ValueError("leaf-output refinement requires 8 to 32 candidates")
    if int(leaf_k.size(-1)) != head_dim or leaf_k.shape[:3] != leaf_v.shape[:3]:
        raise ValueError("virtual leaf K/V geometry is inconsistent")
    if tuple(page_indices.shape[-1:]) != (page_size,):
        raise ValueError("virtual page geometry differs from the page size")
    if tuple(page_counts.shape) != tuple(page_indices.shape[:3]):
        raise ValueError("page counts differ from virtual page storage")
    if tuple(state_counts.shape) != tuple(state_sum_v.shape[:3]):
        raise ValueError("state value sums and counts are inconsistent")
    if page_size != 16:
        raise ValueError("leaf-output refinement currently requires 16-token pages")
    rows = batch * query_heads * query_len
    target_output = torch.empty_like(baseline_output, dtype=torch.float32)
    common_kernel_args = (
        q.contiguous(),
        baseline_output.contiguous(),
        baseline_lse.contiguous(),
        candidate_coarse_lse.contiguous(),
        state_sum_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices.contiguous(),
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates.contiguous(),
    )
    common_meta = {
        "QUERY_HEADS": query_heads,
        "KV_HEADS": kv_heads,
        "KV_GROUP_SIZE": kv_group_size,
        "PAGE_CAPACITY": page_capacity,
        "LEAF_CAPACITY": int(leaf_k.size(2)),
        "STATE_CAPACITY": int(slot_pages.size(2)),
        "INLINE_PAGES_PER_SLOT": int(slot_pages.size(3)),
        "HASH_CAPACITY": int(overflow_page_keys.size(2)),
        "HASH_PROBES": hash_probes,
        "HEAD_DIM": head_dim,
        "VALUE_DIM": value_dim,
        "PAGE_SIZE": page_size,
        "CANDIDATE_COUNT": candidate_count,
        "SCALE": float(scale),
    }
    _candidate_virtual_leaf_target_output_kernel[(rows,)](
        *common_kernel_args,
        target_output,
        query_len,
        **common_meta,
        num_warps=4,
    )
    utility = torch.empty_like(candidate_coarse_lse, dtype=torch.float32)
    _candidate_virtual_leaf_output_utility_kernel[(rows, candidate_count)](
        q.contiguous(),
        baseline_output.contiguous(),
        target_output,
        baseline_lse.contiguous(),
        candidate_coarse_lse.contiguous(),
        state_sum_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices.contiguous(),
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates.contiguous(),
        utility,
        query_len,
        **common_meta,
        num_warps=4,
    )
    return utility


def query_major_residual_page_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    state_counts: torch.Tensor,
    page_k: torch.Tensor | None,
    page_v: torch.Tensor | None,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    cache_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    page_block_n: int = 16,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
    page_indices: torch.Tensor | None = None,
    leaf_k: torch.Tensor | None = None,
    leaf_v: torch.Tensor | None = None,
    quantized_leaf_k: torch.Tensor | None = None,
    quantized_leaf_v: torch.Tensor | None = None,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    page_quantized_counts: torch.Tensor | None = None,
    quantized_page_sum_k: torch.Tensor | None = None,
    quantized_page_sum_v: torch.Tensor | None = None,
    page_sum_k_scales: torch.Tensor | None = None,
    page_sum_v_scales: torch.Tensor | None = None,
    quant_group_size: int = 32,
    quant_bits: int = 4,
    output_buffer: torch.Tensor | None = None,
    lse_buffer: torch.Tensor | None = None,
    route_parallel: bool = False,
    mla_norm_weight: torch.Tensor | None = None,
    mla_norm_epsilon: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top page plus a count-corrected residual for each routed slot."""
    indexed = page_indices is not None
    if indexed != (leaf_k is not None and leaf_v is not None):
        raise ValueError("indexed pages require indices and flat K/V together")
    if not indexed and (page_k is None or page_v is None):
        raise ValueError("physical residual-page attention requires page K/V")
    storage_k = leaf_k if indexed else page_k
    storage_v = leaf_v if indexed else page_v
    if storage_k is None or storage_v is None:
        raise AssertionError("residual-page K/V storage is missing")
    quantization_tensors = (
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
    )
    quantized = any(tensor is not None for tensor in quantization_tensors)
    if quantized and not all(
        isinstance(tensor, torch.Tensor) for tensor in quantization_tensors
    ):
        raise ValueError("indexed quantized tensors must be supplied together")
    if quantized and not indexed:
        raise ValueError("quantized residual pages require indexed virtual storage")
    if quantized and quant_bits not in (4, 8):
        raise ValueError("quantized residual pages support 4 or 8 bits")
    summary_quantization_tensors = (
        quantized_page_sum_k,
        quantized_page_sum_v,
        page_sum_k_scales,
        page_sum_v_scales,
    )
    quantized_summaries = any(
        tensor is not None for tensor in summary_quantization_tensors
    )
    if quantized_summaries and not all(
        isinstance(tensor, torch.Tensor) for tensor in summary_quantization_tensors
    ):
        raise ValueError("INT8 page-summary tensors must be supplied together")
    mla_latent_dim = 0
    if mla_norm_weight is not None:
        if not mla_norm_weight.is_cuda:
            raise ValueError("MLA RMSNorm gain must be a CUDA tensor")
        if quantized or quantized_summaries:
            raise ValueError("raw MLA page summaries do not support quantization")
        mla_latent_dim = int(mla_norm_weight.numel())
    page_shape = page_indices.shape if indexed else page_k.shape
    tensors = (
        q,
        state_k,
        state_v,
        state_counts,
        storage_k,
        storage_v,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        slot_lengths,
        top_slots,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("residual-page attention requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    cache_batch_size = int(storage_k.size(0))
    if cache_indices is None:
        if cache_batch_size != batch:
            raise ValueError(
                "cache indices are required when cache and query batches differ"
            )
        cache_indices = torch.arange(batch, dtype=torch.long, device=q.device)
    elif tuple(cache_indices.shape) != (batch,):
        raise ValueError("cache indices must contain one stable slot per query row")
    kv_heads = int(storage_k.size(1))
    value_dim = int(storage_v.size(-1))
    if mla_latent_dim and not 0 < mla_latent_dim < head_dim:
        raise ValueError("MLA RMSNorm gain does not match the key geometry")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if quantized and (head_dim % quant_group_size or value_dim % quant_group_size):
        raise ValueError("quantization group size must divide K/V dimensions")
    if quantized:
        expected_k_width = head_dim // 2 if quant_bits == 4 else head_dim
        expected_v_width = value_dim // 2 if quant_bits == 4 else value_dim
        expected_dtype = torch.uint8 if quant_bits == 4 else torch.int8
        if quantized_leaf_k.dtype != expected_dtype or (
            quantized_leaf_v.dtype != expected_dtype
        ):
            raise TypeError(
                f"{quant_bits}-bit leaf storage requires {expected_dtype} codes"
            )
        if int(quantized_leaf_k.size(-1)) != expected_k_width or (
            int(quantized_leaf_v.size(-1)) != expected_v_width
        ):
            raise ValueError("quantized leaf widths do not match K/V dimensions")
    if int(page_shape[3]) != 16:
        raise ValueError("residual-page attention requires 16-token pages")
    expected_k_summary = (
        cache_batch_size,
        kv_heads,
        int(page_shape[2]),
        head_dim,
    )
    expected_v_summary = (
        cache_batch_size,
        kv_heads,
        int(page_shape[2]),
        value_dim,
    )
    if quantized_summaries:
        if tuple(quantized_page_sum_k.shape) != expected_k_summary:
            raise ValueError("quantized page K summaries do not match the cache")
        if tuple(quantized_page_sum_v.shape) != expected_v_summary:
            raise ValueError("quantized page V summaries do not match the cache")
        expected_k_scales = expected_k_summary[:-1] + (head_dim // quant_group_size,)
        expected_v_scales = expected_v_summary[:-1] + (value_dim // quant_group_size,)
        if tuple(page_sum_k_scales.shape) != expected_k_scales:
            raise ValueError("page K-summary scales do not match the cache")
        if tuple(page_sum_v_scales.shape) != expected_v_scales:
            raise ValueError("page V-summary scales do not match the cache")
    else:
        if tuple(page_sum_k.shape) != expected_k_summary:
            raise ValueError("page K summaries do not match the page cache")
        if tuple(page_sum_v.shape) != expected_v_summary:
            raise ValueError("page V summaries do not match the page cache")
    rows = batch * query_heads * query_len
    route_count = int(top_slots.size(-1))
    if route_parallel and query_len != 1:
        raise ValueError("route-parallel residual pages require decode queries")
    output_shape = (
        (batch, query_heads, route_count, value_dim)
        if route_parallel
        else (batch, query_heads, query_len, value_dim)
    )
    lse_shape = output_shape[:-1]
    if output_buffer is None:
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
    else:
        if tuple(output_buffer.shape) != output_shape:
            raise ValueError("residual-page output buffer has the wrong shape")
        output = output_buffer
    if lse_buffer is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    else:
        if tuple(lse_buffer.shape) != lse_shape:
            raise ValueError("residual-page LSE buffer has the wrong shape")
        lse = lse_buffer
    begin = None
    if timing_events is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
    grid = (rows, route_count) if route_parallel else (rows, 1)
    _query_major_residual_page_attention_kernel[grid](
        q.contiguous(),
        state_k,
        state_v,
        state_counts,
        mla_norm_weight if mla_norm_weight is not None else page_counts,
        cache_indices.contiguous(),
        storage_k,
        storage_v,
        page_indices if indexed else slot_pages,
        storage_k,
        storage_v,
        quantized_leaf_k if quantized else storage_k,
        quantized_leaf_v if quantized else storage_v,
        page_k_scales if quantized else page_counts,
        page_v_scales if quantized else page_counts,
        page_quantized_counts if quantized else page_counts,
        page_sum_k,
        page_sum_v,
        quantized_page_sum_k if quantized_summaries else page_sum_k,
        quantized_page_sum_v if quantized_summaries else page_sum_v,
        page_sum_k_scales if quantized_summaries else page_counts,
        page_sum_v_scales if quantized_summaries else page_counts,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots.contiguous(),
        query_len,
        output,
        lse,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=int(page_shape[2]),
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=int(page_shape[3]),
        ROUTE_COUNT=int(top_slots.size(-1)),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        PAGE_BLOCK_N=page_block_n,
        LEAF_K_BATCH_STRIDE=int(storage_k.stride(0)) if indexed else 0,
        LEAF_K_HEAD_STRIDE=int(storage_k.stride(1)) if indexed else 0,
        LEAF_K_TOKEN_STRIDE=int(storage_k.stride(2)) if indexed else 0,
        LEAF_V_BATCH_STRIDE=int(storage_v.stride(0)) if indexed else 0,
        LEAF_V_HEAD_STRIDE=int(storage_v.stride(1)) if indexed else 0,
        LEAF_V_TOKEN_STRIDE=int(storage_v.stride(2)) if indexed else 0,
        LEAF_CAPACITY=(
            int(quantized_leaf_k.size(2)) if quantized else int(storage_k.size(2))
        ),
        QUANT_GROUP_SIZE=quant_group_size,
        QUANT_BITS=quant_bits if quantized else 0,
        QUANTIZED_SUMMARIES=quantized_summaries,
        INDEXED=indexed,
        ROUTE_PARALLEL=route_parallel,
        MLA_LATENT_DIM=mla_latent_dim,
        MLA_NORM_EPS=float(mla_norm_epsilon),
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if begin is None:
            raise AssertionError("recursive page timing start is missing")
        timing_events.setdefault("kernel", []).append((begin, end))
        timing_events.setdefault("total", []).append((begin, end))
    return (
        output,
        lse,
    )


def query_major_indexed_residual_page_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    state_counts: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursive page attention over indexed leaves in the original KV cache."""
    return query_major_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        None,
        None,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        page_indices=page_indices,
        leaf_k=leaf_k,
        leaf_v=leaf_v,
        **kwargs,
    )


@triton.jit
def _online_softmax_update(
    scores,
    values,
    valid,
    maximum,
    denominator,
    accumulator,
    USE_DOT: tl.constexpr,
):
    scores = tl.where(valid, scores, -float("inf"))
    block_maximum = tl.max(scores, axis=0)
    new_maximum = tl.maximum(maximum, block_maximum)
    correction = tl.math.exp2(maximum - new_maximum)
    probabilities = tl.math.exp2(scores - new_maximum)
    probabilities = tl.where(valid, probabilities, 0.0)
    denominator = denominator * correction + tl.sum(probabilities, axis=0)
    if USE_DOT:
        value_update = tl.dot(
            probabilities[None, :].to(values.dtype),
            values,
            out_dtype=tl.float32,
        )
        value_update = tl.reshape(value_update, (values.shape[1],))
    else:
        value_update = tl.sum(probabilities[:, None] * values, axis=0)
    accumulator = accumulator * correction + value_update
    return new_maximum, denominator, accumulator


@triton.jit
def _gqa_online_softmax_update(
    scores,
    values,
    valid,
    maximum,
    denominator,
    accumulator,
):
    """Independent short-M softmax updates over one shared K/V tile."""
    scores = tl.where(valid, scores, -float("inf"))
    active = tl.sum(valid.to(tl.int32), axis=1) > 0
    block_maximum = tl.max(scores, axis=1)
    new_maximum = tl.where(active, tl.maximum(maximum, block_maximum), maximum)
    correction = tl.where(active, tl.math.exp2(maximum - new_maximum), 1.0)
    probabilities = tl.where(valid, tl.math.exp2(scores - new_maximum[:, None]), 0.0)
    denominator = denominator * correction + tl.sum(probabilities, axis=1)
    value_update = tl.sum(
        probabilities[:, :, None] * values[None, :, :].to(tl.float32),
        axis=1,
    )
    accumulator = accumulator * correction[:, None] + value_update
    return new_maximum, denominator, accumulator


@triton.jit
def _pack_route_score_index(scores, indices):
    """Pack descending FP32 score and ascending slot index into one int64."""
    score_bits = scores.to(tl.uint32, bitcast=True)
    negative = (score_bits & 0x80000000) != 0
    ordered_bits = tl.where(
        negative,
        score_bits ^ 0xFFFFFFFF,
        score_bits ^ 0x80000000,
    ).to(tl.int64)
    score_rank = ordered_bits - 2147483648
    inverse_index = 4294967295 - indices.to(tl.int64)
    return score_rank * 4294967296 + inverse_index


@triton.jit
def _unpack_route_score_index(packed):
    inverse_index = packed & 0xFFFFFFFF
    indices = (4294967295 - inverse_index).to(tl.int64)
    score_rank = packed >> 32
    ordered_bits = (score_rank + 2147483648).to(tl.uint32)
    negative = (ordered_bits & 0x80000000) == 0
    score_bits = tl.where(
        negative,
        ordered_bits ^ 0xFFFFFFFF,
        ordered_bits ^ 0x80000000,
    )
    scores = score_bits.to(tl.float32, bitcast=True)
    return scores, indices


@triton.jit
def _decode_route_coarse_groups_kernel(
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
):
    """Compute routing candidates and coarse attention from one state read."""
    query_row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    batch = query_row // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    slot = group * GROUP_N + tl.arange(0, GROUP_N)
    valid = slot < state_len
    dim = tl.arange(0, HEAD_DIM)
    query = tl.load(q + query_row * HEAD_DIM + dim)
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
        scores = tl.reshape(scores, (GROUP_N,))
    else:
        scores = tl.sum(
            mean_keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
        )
    scores *= SCALE
    scores += tl.log(count)
    scores = tl.where(valid, scores, -float("inf"))
    route_scores = tl.where(slot >= PROTECTED_LEN, scores, -float("inf"))
    if MAX_LEAF_TOKENS:
        route_scores = tl.where(count < MAX_LEAF_TOKENS, route_scores, -float("inf"))

    position = tl.arange(0, GROUP_N)
    candidate_base = (query_row * MAX_GROUPS + group) * 8
    if GROUP_N == 8:
        # Every entry is a valid global top-8 candidate.  Preserve them in
        # their natural order and let the global reducer perform the only sort.
        tl.store(candidate_scores + candidate_base + position, route_scores)
        tl.store(
            candidate_indices + candidate_base + position,
            group * GROUP_N + position,
        )
    else:
        packed = _pack_route_score_index(route_scores, slot)
        block_top = tl.topk(packed, 8, dim=0)
        block_scores, block_indices = _unpack_route_score_index(block_top)
        rank = tl.arange(0, 8)
        tl.store(candidate_scores + candidate_base + rank, block_scores)
        tl.store(candidate_indices + candidate_base + rank, block_indices)

    maximum = tl.max(scores, axis=0)
    weights = tl.exp(scores - maximum)
    weights = tl.where(valid, weights, 0.0)
    denominator = tl.sum(weights, axis=0)
    weighted_values = tl.dot(
        weights[None, :].to(mean_values.dtype),
        mean_values,
        out_dtype=tl.float32,
    )
    weighted_values = tl.reshape(weighted_values, (HEAD_DIM,))
    group_row = query_row * MAX_GROUPS + group
    tl.store(
        group_out + group_row * HEAD_DIM + dim,
        tl.where(denominator > 0.0, weighted_values / denominator, 0.0),
    )
    tl.store(
        group_lse + group_row,
        tl.where(denominator > 0.0, maximum + tl.log(denominator), -float("inf")),
    )


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
):
    """Share each state K/V tile across all query heads in one GQA group."""
    batch_kv = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = batch_kv - batch * KV_HEADS
    # Pad the four GQA rows to a native MFMA M tile. Short-M value matmuls use
    # a different reduction order on MI325X and perturb decode enough to hurt
    # exact string continuation.
    q_offset = tl.arange(0, 16)
    query_valid = q_offset < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + q_offset
    query_row = batch * QUERY_HEADS + query_head
    slot = group * GROUP_N + tl.arange(0, GROUP_N)
    valid = slot < state_len
    dim = tl.arange(0, HEAD_DIM)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
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

    candidate_base = (query_row * MAX_GROUPS + group) * 8
    packed = _pack_route_score_index(route_scores, slot[None, :])
    block_top = tl.topk(packed, 8, dim=1)
    block_scores, block_indices = _unpack_route_score_index(block_top)
    rank = tl.arange(0, 8)
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
def _decode_route_coarse_scalar_gqa_groups_kernel(
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
):
    """Reuse one state tile while preserving scalar per-query routing math."""
    batch_kv = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = batch_kv - batch * KV_HEADS
    slot = group * GROUP_N + tl.arange(0, GROUP_N)
    valid = slot < state_len
    dim = tl.arange(0, HEAD_DIM)
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
    position = tl.arange(0, GROUP_N)

    for query_group in tl.static_range(0, KV_GROUP_SIZE):
        query_head = kv_head * KV_GROUP_SIZE + query_group
        query_row = batch * QUERY_HEADS + query_head
        query = tl.load(q + query_row * HEAD_DIM + dim)
        scores = tl.sum(
            mean_keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
        )
        scores = scores * SCALE + tl.log(count)
        scores = tl.where(valid, scores, -float("inf"))
        route_scores = tl.where(slot >= PROTECTED_LEN, scores, -float("inf"))
        if MAX_LEAF_TOKENS:
            route_scores = tl.where(
                count < MAX_LEAF_TOKENS, route_scores, -float("inf")
            )

        candidate_base = (query_row * MAX_GROUPS + group) * 8
        if GROUP_N == 8:
            tl.store(candidate_scores + candidate_base + position, route_scores)
            tl.store(
                candidate_indices + candidate_base + position,
                group * GROUP_N + position,
            )
        else:
            packed = _pack_route_score_index(route_scores, slot)
            block_top = tl.topk(packed, 8, dim=0)
            block_scores, block_indices = _unpack_route_score_index(block_top)
            rank = tl.arange(0, 8)
            tl.store(candidate_scores + candidate_base + rank, block_scores)
            tl.store(candidate_indices + candidate_base + rank, block_indices)

        maximum = tl.max(scores, axis=0)
        weights = tl.exp(scores - maximum)
        weights = tl.where(valid, weights, 0.0)
        denominator = tl.sum(weights, axis=0)
        weighted_values = tl.dot(
            weights[None, :].to(mean_values.dtype),
            mean_values,
            out_dtype=tl.float32,
        )
        weighted_values = tl.reshape(weighted_values, (HEAD_DIM,))
        group_row = query_row * MAX_GROUPS + group
        tl.store(
            group_out + group_row * HEAD_DIM + dim,
            tl.where(denominator > 0.0, weighted_values / denominator, 0.0),
        )
        tl.store(
            group_lse + group_row,
            tl.where(
                denominator > 0.0,
                maximum + tl.log(denominator),
                -float("inf"),
            ),
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
    ROUTE_COUNT: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    CANDIDATE_TILE: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    candidate_offset = tl.arange(0, CANDIDATE_TILE)
    best_packed = tl.full((8,), -9223372036854775807, tl.int64)
    for candidate_begin in tl.range(0, active_groups * 8, CANDIDATE_TILE, num_stages=1):
        candidate = candidate_begin + candidate_offset
        valid_candidate = candidate < active_groups * 8
        scores = tl.load(
            candidate_scores + query_row * MAX_GROUPS * 8 + candidate,
            mask=valid_candidate,
            other=-float("inf"),
        )
        indices = tl.load(
            candidate_indices + query_row * MAX_GROUPS * 8 + candidate,
            mask=valid_candidate,
            other=0,
        )
        packed = _pack_route_score_index(scores, indices)
        block_top = tl.topk(packed, 8, dim=0)
        best_packed = tl.topk(tl.interleave(best_packed, block_top), 8, dim=0)
    best_scores, best_indices = _unpack_route_score_index(best_packed)
    rank = tl.arange(0, 8)
    if ROUTE_COUNT == 8:
        tl.store(top_slots + query_row * ROUTE_COUNT + rank, best_indices)
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
    tl.store(coarse_out + query_row * HEAD_DIM + dim, accumulator / denominator)
    tl.store(
        coarse_lse + query_row,
        maximum + tl.log(denominator),
    )


@triton.jit
def _mask_decode_routes_top_p_kernel(
    top_slots,
    top_scores,
    target,
    ROUTE_COUNT: tl.constexpr,
):
    """Keep the smallest sorted route prefix covering ``target`` mass."""
    query_row = tl.program_id(0).to(tl.int64)
    rank = tl.arange(0, ROUTE_COUNT)
    scores = tl.load(top_scores + query_row * ROUTE_COUNT + rank)
    weights = tl.exp(scores - tl.max(scores, axis=0))
    cumulative_before = tl.cumsum(weights, axis=0) - weights
    keep = cumulative_before < target * tl.sum(weights, axis=0)
    slots = tl.load(top_slots + query_row * ROUTE_COUNT + rank)
    tl.store(
        top_slots + query_row * ROUTE_COUNT + rank,
        tl.where(keep, slots, -1),
    )


@triton.jit
def _mask_decode_routes_residual_lse_kernel(
    top_slots,
    top_scores,
    denominator_lse,
    residual_mass,
    ROUTE_COUNT: tl.constexpr,
):
    """Apply a residual-mass cutoff against a supplied global LSE bound."""
    query_row = tl.program_id(0).to(tl.int64)
    rank = tl.arange(0, ROUTE_COUNT)
    scores = tl.load(top_scores + query_row * ROUTE_COUNT + rank)
    full_lse = tl.load(denominator_lse + query_row)
    global_mass = tl.exp(scores - full_lse)
    cumulative_before = tl.cumsum(global_mass, axis=0) - global_mass
    remaining_before = tl.sum(global_mass, axis=0) - cumulative_before
    keep = (rank == 0) | (remaining_before > residual_mass)
    slots = tl.load(top_slots + query_row * ROUTE_COUNT + rank)
    tl.store(
        top_slots + query_row * ROUTE_COUNT + rank,
        tl.where(keep, slots, -1),
    )


@triton.jit
def _mask_decode_routes_residual_mass_kernel(
    q,
    local_k,
    local_v,
    cache_indices,
    local_lens,
    new_k,
    new_v,
    top_slots,
    top_scores,
    coarse_lse,
    local_out,
    local_lse_out,
    residual_mass,
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
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    LOCAL_BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    COMPUTE_LOCAL_OUTPUT: tl.constexpr,
    APPLY_ROUTE_MASK: tl.constexpr,
):
    """Bound unopened routed mass against the complete state+local field."""
    query_row = tl.program_id(0).to(tl.int64)
    batch = query_row // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    active_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    dim = tl.arange(0, HEAD_DIM)
    query = tl.load(q + query_row * HEAD_DIM + dim).to(tl.float32)

    local_maximum = tl.full((), -float("inf"), tl.float32)
    local_denominator = tl.zeros((), tl.float32)
    local_accumulator = tl.zeros((HEAD_DIM,), tl.float32)
    for begin in tl.range(0, local_len, LOCAL_BLOCK_N):
        position = begin + tl.arange(0, LOCAL_BLOCK_N)
        valid = position < active_local_len
        keys = tl.load(
            local_k
            + cache_batch * LOCAL_K_BATCH_STRIDE
            + kv_head * LOCAL_K_HEAD_STRIDE
            + position[:, None] * LOCAL_K_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        if COMPUTE_LOCAL_OUTPUT:
            values = tl.load(
                local_v
                + cache_batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + position[:, None] * LOCAL_V_TOKEN_STRIDE
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        new_maximum = tl.maximum(local_maximum, block_maximum)
        old_weight = tl.exp(local_maximum - new_maximum)
        weights = tl.exp(scores - new_maximum)
        local_denominator = local_denominator * old_weight + tl.sum(weights, axis=0)
        if COMPUTE_LOCAL_OUTPUT:
            local_accumulator = local_accumulator * old_weight + tl.sum(
                weights[:, None] * values, axis=0
            )
        local_maximum = new_maximum
    if INCLUDE_NEW:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
        ).to(tl.float32)
        if COMPUTE_LOCAL_OUTPUT:
            current_value = tl.load(
                new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim
            ).to(tl.float32)
        current_score = tl.sum(current_key * query, axis=0) * SCALE
        new_maximum = tl.maximum(local_maximum, current_score)
        old_weight = tl.exp(local_maximum - new_maximum)
        current_weight = tl.exp(current_score - new_maximum)
        local_denominator = local_denominator * old_weight + current_weight
        if COMPUTE_LOCAL_OUTPUT:
            local_accumulator = (
                local_accumulator * old_weight + current_weight * current_value
            )
        local_maximum = new_maximum
        if COMPUTE_LOCAL_OUTPUT:
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
    local_lse = local_maximum + tl.log(local_denominator)
    if COMPUTE_LOCAL_OUTPUT:
        tl.store(
            local_out + query_row * HEAD_DIM + dim,
            local_accumulator / local_denominator,
        )
        tl.store(local_lse_out + query_row, local_lse)
    if APPLY_ROUTE_MASK:
        state_lse = tl.load(coarse_lse + query_row)
        full_maximum = tl.maximum(state_lse, local_lse)
        full_lse = full_maximum + tl.log(
            tl.exp(state_lse - full_maximum) + tl.exp(local_lse - full_maximum)
        )

        rank = tl.arange(0, ROUTE_COUNT)
        scores = tl.load(top_scores + query_row * ROUTE_COUNT + rank)
        global_mass = tl.exp(scores - full_lse)
        cumulative_before = tl.cumsum(global_mass, axis=0) - global_mass
        remaining_before = tl.sum(global_mass, axis=0) - cumulative_before
        keep = (rank == 0) | (remaining_before > residual_mass)
        slots = tl.load(top_slots + query_row * ROUTE_COUNT + rank)
        tl.store(
            top_slots + query_row * ROUTE_COUNT + rank,
            tl.where(keep, slots, -1),
        )


@triton.jit
def _fused_decode_paged_lod_attention_kernel(
    q,
    state_k,
    state_v,
    counts,
    local_k,
    local_v,
    page_k,
    page_v,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    top_slots,
    new_k,
    new_v,
    out,
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
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_DOT: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
):
    """Evaluate the complete two-level decode softmax in one program."""
    query_row = tl.program_id(0).to(tl.int64)
    batch = query_row // QUERY_HEADS
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head

    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    query = tl.load(q + query_row * HEAD_DIM + dim)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_DIM,), tl.float32)

    # Low-detail remote branch.  Routed summaries are removed because their
    # exact leaves are included below.
    for state_begin in tl.range(0, state_len, BLOCK_N, num_stages=1):
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
            + batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE,
            mask=valid,
            other=1.0,
        ).to(tl.float32)
        keys = tl.load(
            state_k
            + batch * STATE_BATCH_STRIDE
            + kv_head * STATE_HEAD_STRIDE
            + slot[:, None] * STATE_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        values = tl.load(
            state_v
            + batch * STATE_V_BATCH_STRIDE
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

    # High-detail branch: expand all leaves of every routed summary.
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
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
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
            physical_token = (
                kv_row * PAGE_CAPACITY + page_id
            ) * PAGE_SIZE + within_page
            keys = tl.load(
                page_k + physical_token[:, None] * HEAD_DIM + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            values = tl.load(
                page_v + physical_token[:, None] * VALUE_DIM + dim[None, :],
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

    # Exact bounded sliding-window branch, including the current token.
    for local_begin in tl.range(0, local_len, BLOCK_N, num_stages=1):
        token = local_begin + token_offset
        valid = token < local_len
        keys = tl.load(
            local_k
            + batch * LOCAL_K_BATCH_STRIDE
            + kv_head * LOCAL_K_HEAD_STRIDE
            + token[:, None] * LOCAL_K_TOKEN_STRIDE
            + dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        values = tl.load(
            local_v
            + batch * LOCAL_V_BATCH_STRIDE
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
            scores = tl.sum(keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1)
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

    # The current decode token is not yet in the persistent local cache.  Fold
    # it into this softmax exactly once, then let one query head in each GQA
    # group append the shared KV to the cache for the next decode step.
    if INCLUDE_NEW:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
        )
        current_value = tl.load(
            new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim
        )
        current_score = SCALE_LOG2 * tl.sum(
            current_key.to(tl.float32) * query.to(tl.float32), axis=0
        )
        new_maximum = tl.maximum(maximum, current_score)
        correction = tl.math.exp2(maximum - new_maximum)
        current_weight = tl.math.exp2(current_score - new_maximum)
        denominator = denominator * correction + current_weight
        accumulator = accumulator * correction + current_weight * current_value.to(
            tl.float32
        )
        maximum = new_maximum
        if query_head % KV_GROUP_SIZE == 0:
            tl.store(
                local_k
                + batch * LOCAL_K_BATCH_STRIDE
                + kv_head * LOCAL_K_HEAD_STRIDE
                + local_len * LOCAL_K_TOKEN_STRIDE
                + dim,
                current_key,
            )
            tl.store(
                local_v
                + batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + local_len * LOCAL_V_TOKEN_STRIDE
                + dim,
                current_value,
            )

    tl.store(out + query_row * VALUE_DIM + dim, accumulator / denominator)


@triton.jit
def _gqa_cooperative_route_decode_paged_leaf_attention_kernel(
    q,
    cache_indices,
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
    partial_out,
    partial_lse,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CACHE_BATCHES: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEXED: tl.constexpr,
    INT8_STORAGE: tl.constexpr,
):
    """Expand one union centroid for all selecting heads in a GQA group."""
    batch_kv = tl.program_id(0).to(tl.int64)
    candidate = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    cache_valid = (cache_batch >= 0) & (cache_batch < CACHE_BATCHES)
    cache_batch = tl.where(cache_valid, cache_batch, 0)
    kv_head = batch_kv - batch * KV_HEADS
    cache_kv = cache_batch * KV_HEADS + kv_head

    union_offset = tl.arange(0, BLOCK_M * ROUTE_COUNT)
    union_head = union_offset // ROUTE_COUNT
    union_rank = union_offset % ROUTE_COUNT
    union_valid = (union_head < KV_GROUP_SIZE) & cache_valid
    all_slots = tl.load(
        top_slots
        + batch * TOP_BATCH_STRIDE
        + (kv_head * KV_GROUP_SIZE + union_head) * TOP_HEAD_STRIDE
        + union_rank,
        mask=union_valid,
        other=-2,
    ).to(tl.int64)
    candidate_head = candidate // ROUTE_COUNT
    candidate_rank = candidate % ROUTE_COUNT
    routed_slot = tl.load(
        top_slots
        + batch * TOP_BATCH_STRIDE
        + (kv_head * KV_GROUP_SIZE + candidate_head) * TOP_HEAD_STRIDE
        + candidate_rank,
        mask=(candidate_head < KV_GROUP_SIZE) & cache_valid,
        other=-1,
    ).to(tl.int64)
    slot_valid = (
        cache_valid
        & (candidate_head < KV_GROUP_SIZE)
        & (routed_slot >= 0)
        & (routed_slot < STATE_CAPACITY)
    )
    seen_before = (
        tl.sum(
            (union_valid & (union_offset < candidate) & (all_slots == routed_slot)).to(
                tl.int32
            ),
            axis=0,
        )
        > 0
    )
    union_leader = slot_valid & ~seen_before

    if union_leader:
        matched = union_valid & (all_slots == routed_slot)
        matched0 = matched & (union_head == 0)
        selected0 = tl.sum(matched0.to(tl.int32), axis=0) > 0
        selected_rank0 = tl.sum(matched0.to(tl.int32) * union_rank, axis=0)
        if KV_GROUP_SIZE > 1:
            matched1 = matched & (union_head == 1)
            selected1 = tl.sum(matched1.to(tl.int32), axis=0) > 0
            selected_rank1 = tl.sum(matched1.to(tl.int32) * union_rank, axis=0)
        if KV_GROUP_SIZE > 2:
            matched2 = matched & (union_head == 2)
            selected2 = tl.sum(matched2.to(tl.int32), axis=0) > 0
            selected_rank2 = tl.sum(matched2.to(tl.int32) * union_rank, axis=0)
        if KV_GROUP_SIZE > 3:
            matched3 = matched & (union_head == 3)
            selected3 = tl.sum(matched3.to(tl.int32), axis=0) > 0
            selected_rank3 = tl.sum(matched3.to(tl.int32) * union_rank, axis=0)

        query_row = batch * QUERY_HEADS + kv_head * KV_GROUP_SIZE
        dim = tl.arange(0, HEAD_DIM)
        token_offset = tl.arange(0, BLOCK_N)
        query0 = tl.load(q + query_row * HEAD_DIM + dim)
        maximum0 = tl.full((), -float("inf"), tl.float32)
        denominator0 = tl.zeros((), tl.float32)
        accumulator0 = tl.zeros((HEAD_DIM,), tl.float32)
        if KV_GROUP_SIZE > 1:
            query1 = tl.load(q + (query_row + 1) * HEAD_DIM + dim)
            maximum1 = tl.full((), -float("inf"), tl.float32)
            denominator1 = tl.zeros((), tl.float32)
            accumulator1 = tl.zeros((HEAD_DIM,), tl.float32)
        if KV_GROUP_SIZE > 2:
            query2 = tl.load(q + (query_row + 2) * HEAD_DIM + dim)
            maximum2 = tl.full((), -float("inf"), tl.float32)
            denominator2 = tl.zeros((), tl.float32)
            accumulator2 = tl.zeros((HEAD_DIM,), tl.float32)
        if KV_GROUP_SIZE > 3:
            query3 = tl.load(q + (query_row + 3) * HEAD_DIM + dim)
            maximum3 = tl.full((), -float("inf"), tl.float32)
            denominator3 = tl.zeros((), tl.float32)
            accumulator3 = tl.zeros((HEAD_DIM,), tl.float32)

        slot = routed_slot
        key_count = tl.load(slot_lengths + cache_kv * STATE_CAPACITY + slot).to(
            tl.int32
        )
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (cache_kv * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
            logical_key = key_begin + token_offset
            token_valid = logical_key < key_count
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal,
                    mask=token_valid,
                    other=0,
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    cache_kv,
                    slot,
                    page_ordinal,
                    token_valid,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            page_valid = token_valid & (page_id >= 0) & (page_id < PAGE_CAPACITY)
            page_id = tl.where(page_valid, page_id, 0)
            physical_token = (
                cache_kv * PAGE_CAPACITY + page_id
            ) * PAGE_SIZE + within_page
            if INDEXED:
                leaf_index = tl.load(
                    page_indices + physical_token,
                    mask=page_valid,
                    other=0,
                ).to(tl.int64)
                token_valid = (
                    page_valid & (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
                )
                leaf_index = tl.where(token_valid, leaf_index, 0)
                storage_token = cache_kv * LEAF_CAPACITY + leaf_index
            else:
                token_valid = page_valid
                storage_token = physical_token
            keys = tl.load(
                page_k + storage_token[:, None] * HEAD_DIM + dim[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            values = tl.load(
                page_v + storage_token[:, None] * HEAD_DIM + dim[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            if INT8_STORAGE:
                key_scale = tl.load(
                    page_k_scales + storage_token, mask=token_valid, other=0.0
                ).to(tl.float32)
                value_scale = tl.load(
                    page_v_scales + storage_token, mask=token_valid, other=0.0
                ).to(tl.float32)
                keys = keys.to(tl.float32) * key_scale[:, None]
                values = values.to(tl.float32) * value_scale[:, None]
            if selected0:
                scores0 = (
                    tl.sum(
                        keys.to(tl.float32) * query0[None, :].to(tl.float32),
                        axis=1,
                    )
                    * SCALE_LOG2
                )
                maximum0, denominator0, accumulator0 = _online_softmax_update(
                    scores0,
                    values,
                    token_valid,
                    maximum0,
                    denominator0,
                    accumulator0,
                    False,
                )
            if KV_GROUP_SIZE > 1:
                if selected1:
                    scores1 = (
                        tl.sum(
                            keys.to(tl.float32) * query1[None, :].to(tl.float32),
                            axis=1,
                        )
                        * SCALE_LOG2
                    )
                    maximum1, denominator1, accumulator1 = _online_softmax_update(
                        scores1,
                        values,
                        token_valid,
                        maximum1,
                        denominator1,
                        accumulator1,
                        False,
                    )
            if KV_GROUP_SIZE > 2:
                if selected2:
                    scores2 = (
                        tl.sum(
                            keys.to(tl.float32) * query2[None, :].to(tl.float32),
                            axis=1,
                        )
                        * SCALE_LOG2
                    )
                    maximum2, denominator2, accumulator2 = _online_softmax_update(
                        scores2,
                        values,
                        token_valid,
                        maximum2,
                        denominator2,
                        accumulator2,
                        False,
                    )
            if KV_GROUP_SIZE > 3:
                if selected3:
                    scores3 = (
                        tl.sum(
                            keys.to(tl.float32) * query3[None, :].to(tl.float32),
                            axis=1,
                        )
                        * SCALE_LOG2
                    )
                    maximum3, denominator3, accumulator3 = _online_softmax_update(
                        scores3,
                        values,
                        token_valid,
                        maximum3,
                        denominator3,
                        accumulator3,
                        False,
                    )

        partial_row0 = query_row * ROUTE_COUNT + selected_rank0
        has_mass0 = denominator0 > 0.0
        tl.store(
            partial_out + partial_row0 * HEAD_DIM + dim,
            tl.where(has_mass0, accumulator0 / denominator0, 0.0),
            mask=selected0,
        )
        tl.store(
            partial_lse + partial_row0,
            tl.where(
                has_mass0,
                (maximum0 + tl.math.log2(denominator0)) * 0.6931471805599453,
                -float("inf"),
            ),
            mask=selected0,
        )
        if KV_GROUP_SIZE > 1:
            partial_row1 = (query_row + 1) * ROUTE_COUNT + selected_rank1
            has_mass1 = denominator1 > 0.0
            tl.store(
                partial_out + partial_row1 * HEAD_DIM + dim,
                tl.where(has_mass1, accumulator1 / denominator1, 0.0),
                mask=selected1,
            )
            tl.store(
                partial_lse + partial_row1,
                tl.where(
                    has_mass1,
                    (maximum1 + tl.math.log2(denominator1)) * 0.6931471805599453,
                    -float("inf"),
                ),
                mask=selected1,
            )
        if KV_GROUP_SIZE > 2:
            partial_row2 = (query_row + 2) * ROUTE_COUNT + selected_rank2
            has_mass2 = denominator2 > 0.0
            tl.store(
                partial_out + partial_row2 * HEAD_DIM + dim,
                tl.where(has_mass2, accumulator2 / denominator2, 0.0),
                mask=selected2,
            )
            tl.store(
                partial_lse + partial_row2,
                tl.where(
                    has_mass2,
                    (maximum2 + tl.math.log2(denominator2)) * 0.6931471805599453,
                    -float("inf"),
                ),
                mask=selected2,
            )
        if KV_GROUP_SIZE > 3:
            partial_row3 = (query_row + 3) * ROUTE_COUNT + selected_rank3
            has_mass3 = denominator3 > 0.0
            tl.store(
                partial_out + partial_row3 * HEAD_DIM + dim,
                tl.where(has_mass3, accumulator3 / denominator3, 0.0),
                mask=selected3,
            )
            tl.store(
                partial_lse + partial_row3,
                tl.where(
                    has_mass3,
                    (maximum3 + tl.math.log2(denominator3)) * 0.6931471805599453,
                    -float("inf"),
                ),
                mask=selected3,
            )


@triton.jit
def _gqa_cooperative_split_decode_local_attention_kernel(
    q,
    cache_indices,
    local_lens,
    local_k,
    local_v,
    new_k,
    new_v,
    partial_out,
    partial_lse,
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
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOCAL_SPLITS: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
):
    """Share local-window K/V loads across a short GQA query tile."""
    batch_kv = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    active_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
    kv_head = batch_kv - batch * KV_HEADS
    query_offset = tl.arange(0, BLOCK_M)
    query_valid = query_offset < KV_GROUP_SIZE
    query_head = kv_head * KV_GROUP_SIZE + query_offset
    query_row = batch * QUERY_HEADS + query_head
    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for local_begin in tl.range(
        split * BLOCK_N, local_len, LOCAL_SPLITS * BLOCK_N, num_stages=1
    ):
        token = local_begin + token_offset
        token_valid = token < active_local_len
        keys = tl.load(
            local_k
            + cache_batch * LOCAL_K_BATCH_STRIDE
            + kv_head * LOCAL_K_HEAD_STRIDE
            + token[:, None] * LOCAL_K_TOKEN_STRIDE
            + dim[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        values = tl.load(
            local_v
            + cache_batch * LOCAL_V_BATCH_STRIDE
            + kv_head * LOCAL_V_HEAD_STRIDE
            + token[:, None] * LOCAL_V_TOKEN_STRIDE
            + dim[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        valid = query_valid[:, None] & token_valid[None, :]
        scores = (
            tl.sum(
                queries[:, None, :].to(tl.float32) * keys[None, :, :].to(tl.float32),
                axis=2,
            )
            * SCALE_LOG2
        )
        maximum, denominator, accumulator = _gqa_online_softmax_update(
            scores,
            values,
            valid,
            maximum,
            denominator,
            accumulator,
        )

    if INCLUDE_NEW and split == 0:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
        )
        current_value = tl.load(
            new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim
        )
        current_score = SCALE_LOG2 * tl.sum(
            queries.to(tl.float32) * current_key[None, :].to(tl.float32),
            axis=1,
        )
        new_maximum = tl.maximum(maximum, current_score)
        correction = tl.math.exp2(maximum - new_maximum)
        current_weight = tl.math.exp2(current_score - new_maximum)
        denominator = denominator * correction + current_weight
        accumulator = accumulator * correction[:, None] + current_weight[
            :, None
        ] * current_value[None, :].to(tl.float32)
        maximum = new_maximum
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

    partial_row = query_row * LOCAL_SPLITS + split
    has_mass = denominator > 0.0
    tl.store(
        partial_out + partial_row[:, None] * HEAD_DIM + dim[None, :],
        tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0),
        mask=query_valid[:, None],
    )
    tl.store(
        partial_lse + partial_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
        mask=query_valid,
    )


@triton.jit
def _gqa_cooperative_split_decode_paged_leaf_attention_kernel(
    q,
    cache_indices,
    local_k,
    local_v,
    page_k,
    page_v,
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
    local_len,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    SEPARATE_LOCAL: tl.constexpr,
):
    """Decode exact leaves cooperatively across one small GQA group.

    The four query heads sharing a KV head often select many of the same
    centroids.  The query-major decode kernel reloads those pages once per
    query head.  This kernel instead builds the route union inside the GQA
    group, loads every union member's K/V pages once, and masks its M<=4
    independent online softmax rows by the heads that selected that member.
    QK and PV deliberately use vector reductions rather than padding M to a
    16-row MFMA tile.
    """
    batch_kv = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    batch = batch_kv // KV_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    kv_head = batch_kv - batch * KV_HEADS
    kv_row = cache_batch * KV_HEADS + kv_head

    query_row = batch * QUERY_HEADS + kv_head * KV_GROUP_SIZE
    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    query0 = tl.load(q + query_row * HEAD_DIM + dim)
    maximum0 = tl.full((), -float("inf"), tl.float32)
    denominator0 = tl.zeros((), tl.float32)
    accumulator0 = tl.zeros((HEAD_DIM,), tl.float32)
    if KV_GROUP_SIZE > 1:
        query1 = tl.load(q + (query_row + 1) * HEAD_DIM + dim)
        maximum1 = tl.full((), -float("inf"), tl.float32)
        denominator1 = tl.zeros((), tl.float32)
        accumulator1 = tl.zeros((HEAD_DIM,), tl.float32)
    if KV_GROUP_SIZE > 2:
        query2 = tl.load(q + (query_row + 2) * HEAD_DIM + dim)
        maximum2 = tl.full((), -float("inf"), tl.float32)
        denominator2 = tl.zeros((), tl.float32)
        accumulator2 = tl.zeros((HEAD_DIM,), tl.float32)
    if KV_GROUP_SIZE > 3:
        query3 = tl.load(q + (query_row + 3) * HEAD_DIM + dim)
        maximum3 = tl.full((), -float("inf"), tl.float32)
        denominator3 = tl.zeros((), tl.float32)
        accumulator3 = tl.zeros((HEAD_DIM,), tl.float32)

    # Flatten the top-eight lists for this GQA group.  A route is owned by its
    # first occurrence in this vector, so pages shared at different ranks are
    # still read exactly once.  The owner's rank assigns it to one split.
    union_offset = tl.arange(0, BLOCK_M * ROUTE_COUNT)
    union_head = union_offset // ROUTE_COUNT
    union_rank = union_offset % ROUTE_COUNT
    union_valid = union_head < KV_GROUP_SIZE
    all_slots = tl.load(
        top_slots
        + batch * TOP_BATCH_STRIDE
        + (kv_head * KV_GROUP_SIZE + union_head) * TOP_HEAD_STRIDE
        + union_rank,
        mask=union_valid,
        other=-2,
    ).to(tl.int64)

    for candidate in tl.range(
        0, BLOCK_M * ROUTE_COUNT, num_stages=1, loop_unroll_factor=1
    ):
        candidate_head = candidate // ROUTE_COUNT
        candidate_rank = candidate % ROUTE_COUNT
        routed_slot = tl.load(
            top_slots
            + batch * TOP_BATCH_STRIDE
            + (kv_head * KV_GROUP_SIZE + candidate_head) * TOP_HEAD_STRIDE
            + candidate_rank,
            mask=candidate_head < KV_GROUP_SIZE,
            other=-1,
        ).to(tl.int64)
        slot_valid = (candidate_head < KV_GROUP_SIZE) & (routed_slot >= 0)
        seen_before = (
            tl.sum(
                (
                    union_valid
                    & (union_offset < candidate)
                    & (all_slots == routed_slot)
                ).to(tl.int32),
                axis=0,
            )
            > 0
        )
        union_leader = slot_valid & ~seen_before
        selected0 = slot_valid & (
            tl.sum(
                (union_valid & (union_head == 0) & (all_slots == routed_slot)).to(
                    tl.int32
                ),
                axis=0,
            )
            > 0
        )
        if KV_GROUP_SIZE > 1:
            selected1 = slot_valid & (
                tl.sum(
                    (union_valid & (union_head == 1) & (all_slots == routed_slot)).to(
                        tl.int32
                    ),
                    axis=0,
                )
                > 0
            )
        if KV_GROUP_SIZE > 2:
            selected2 = slot_valid & (
                tl.sum(
                    (union_valid & (union_head == 2) & (all_slots == routed_slot)).to(
                        tl.int32
                    ),
                    axis=0,
                )
                > 0
            )
        if KV_GROUP_SIZE > 3:
            selected3 = slot_valid & (
                tl.sum(
                    (union_valid & (union_head == 3) & (all_slots == routed_slot)).to(
                        tl.int32
                    ),
                    axis=0,
                )
                > 0
            )

        slot = tl.where(slot_valid, routed_slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=slot_valid,
            other=0,
        ).to(tl.int32)
        key_count = tl.where(union_leader & (split == candidate % SPLITS), key_count, 0)
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )

        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
            logical_key = key_begin + token_offset
            token_valid = logical_key < key_count
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal,
                    mask=token_valid,
                    other=0,
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    token_valid,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            physical_token = (
                kv_row * PAGE_CAPACITY + page_id
            ) * PAGE_SIZE + within_page
            keys = tl.load(
                page_k + physical_token[:, None] * HEAD_DIM + dim[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            values = tl.load(
                page_v + physical_token[:, None] * HEAD_DIM + dim[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            if selected0:
                scores0 = (
                    tl.sum(
                        keys.to(tl.float32) * query0[None, :].to(tl.float32),
                        axis=1,
                    )
                    * SCALE_LOG2
                )
                maximum0, denominator0, accumulator0 = _online_softmax_update(
                    scores0,
                    values,
                    token_valid,
                    maximum0,
                    denominator0,
                    accumulator0,
                    False,
                )
            if KV_GROUP_SIZE > 1:
                if selected1:
                    scores1 = (
                        tl.sum(
                            keys.to(tl.float32) * query1[None, :].to(tl.float32),
                            axis=1,
                        )
                        * SCALE_LOG2
                    )
                    maximum1, denominator1, accumulator1 = _online_softmax_update(
                        scores1,
                        values,
                        token_valid,
                        maximum1,
                        denominator1,
                        accumulator1,
                        False,
                    )
            if KV_GROUP_SIZE > 2:
                if selected2:
                    scores2 = (
                        tl.sum(
                            keys.to(tl.float32) * query2[None, :].to(tl.float32),
                            axis=1,
                        )
                        * SCALE_LOG2
                    )
                    maximum2, denominator2, accumulator2 = _online_softmax_update(
                        scores2,
                        values,
                        token_valid,
                        maximum2,
                        denominator2,
                        accumulator2,
                        False,
                    )
            if KV_GROUP_SIZE > 3:
                if selected3:
                    scores3 = (
                        tl.sum(
                            keys.to(tl.float32) * query3[None, :].to(tl.float32),
                            axis=1,
                        )
                        * SCALE_LOG2
                    )
                    maximum3, denominator3, accumulator3 = _online_softmax_update(
                        scores3,
                        values,
                        token_valid,
                        maximum3,
                        denominator3,
                        accumulator3,
                        False,
                    )

    # The bounded local window is also shared by the same KV head.  Interleave
    # its tiles over the existing splits and reuse each K/V load across M rows.
    if not SEPARATE_LOCAL:
        for local_begin in tl.range(
            split * BLOCK_N, local_len, SPLITS * BLOCK_N, num_stages=1
        ):
            token = local_begin + token_offset
            token_valid = token < local_len
            keys = tl.load(
                local_k
                + batch * LOCAL_K_BATCH_STRIDE
                + kv_head * LOCAL_K_HEAD_STRIDE
                + token[:, None] * LOCAL_K_TOKEN_STRIDE
                + dim[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            values = tl.load(
                local_v
                + batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + token[:, None] * LOCAL_V_TOKEN_STRIDE
                + dim[None, :],
                mask=token_valid[:, None],
                other=0.0,
            )
            scores0 = (
                tl.sum(
                    keys.to(tl.float32) * query0[None, :].to(tl.float32),
                    axis=1,
                )
                * SCALE_LOG2
            )
            maximum0, denominator0, accumulator0 = _online_softmax_update(
                scores0,
                values,
                token_valid,
                maximum0,
                denominator0,
                accumulator0,
                False,
            )
            if KV_GROUP_SIZE > 1:
                scores1 = (
                    tl.sum(
                        keys.to(tl.float32) * query1[None, :].to(tl.float32),
                        axis=1,
                    )
                    * SCALE_LOG2
                )
                maximum1, denominator1, accumulator1 = _online_softmax_update(
                    scores1,
                    values,
                    token_valid,
                    maximum1,
                    denominator1,
                    accumulator1,
                    False,
                )
            if KV_GROUP_SIZE > 2:
                scores2 = (
                    tl.sum(
                        keys.to(tl.float32) * query2[None, :].to(tl.float32),
                        axis=1,
                    )
                    * SCALE_LOG2
                )
                maximum2, denominator2, accumulator2 = _online_softmax_update(
                    scores2,
                    values,
                    token_valid,
                    maximum2,
                    denominator2,
                    accumulator2,
                    False,
                )
            if KV_GROUP_SIZE > 3:
                scores3 = (
                    tl.sum(
                        keys.to(tl.float32) * query3[None, :].to(tl.float32),
                        axis=1,
                    )
                    * SCALE_LOG2
                )
                maximum3, denominator3, accumulator3 = _online_softmax_update(
                    scores3,
                    values,
                    token_valid,
                    maximum3,
                    denominator3,
                    accumulator3,
                    False,
                )

    # Fold the just-produced shared KV into split zero, and append it once for
    # the whole GQA group rather than once from every query-head program.
    if INCLUDE_NEW and not SEPARATE_LOCAL:
        current_key = tl.load(
            new_k + batch * NEW_K_BATCH_STRIDE + kv_head * NEW_K_HEAD_STRIDE + dim
        )
        current_value = tl.load(
            new_v + batch * NEW_V_BATCH_STRIDE + kv_head * NEW_V_HEAD_STRIDE + dim
        )
        if split == 0:
            current_score0 = SCALE_LOG2 * tl.sum(
                query0.to(tl.float32) * current_key.to(tl.float32), axis=0
            )
            new_maximum0 = tl.maximum(maximum0, current_score0)
            correction0 = tl.math.exp2(maximum0 - new_maximum0)
            current_weight0 = tl.math.exp2(current_score0 - new_maximum0)
            denominator0 = denominator0 * correction0 + current_weight0
            accumulator0 = (
                accumulator0 * correction0
                + current_weight0 * current_value.to(tl.float32)
            )
            maximum0 = new_maximum0
            if KV_GROUP_SIZE > 1:
                current_score1 = SCALE_LOG2 * tl.sum(
                    query1.to(tl.float32) * current_key.to(tl.float32), axis=0
                )
                new_maximum1 = tl.maximum(maximum1, current_score1)
                correction1 = tl.math.exp2(maximum1 - new_maximum1)
                current_weight1 = tl.math.exp2(current_score1 - new_maximum1)
                denominator1 = denominator1 * correction1 + current_weight1
                accumulator1 = (
                    accumulator1 * correction1
                    + current_weight1 * current_value.to(tl.float32)
                )
                maximum1 = new_maximum1
            if KV_GROUP_SIZE > 2:
                current_score2 = SCALE_LOG2 * tl.sum(
                    query2.to(tl.float32) * current_key.to(tl.float32), axis=0
                )
                new_maximum2 = tl.maximum(maximum2, current_score2)
                correction2 = tl.math.exp2(maximum2 - new_maximum2)
                current_weight2 = tl.math.exp2(current_score2 - new_maximum2)
                denominator2 = denominator2 * correction2 + current_weight2
                accumulator2 = (
                    accumulator2 * correction2
                    + current_weight2 * current_value.to(tl.float32)
                )
                maximum2 = new_maximum2
            if KV_GROUP_SIZE > 3:
                current_score3 = SCALE_LOG2 * tl.sum(
                    query3.to(tl.float32) * current_key.to(tl.float32), axis=0
                )
                new_maximum3 = tl.maximum(maximum3, current_score3)
                correction3 = tl.math.exp2(maximum3 - new_maximum3)
                current_weight3 = tl.math.exp2(current_score3 - new_maximum3)
                denominator3 = denominator3 * correction3 + current_weight3
                accumulator3 = (
                    accumulator3 * correction3
                    + current_weight3 * current_value.to(tl.float32)
                )
                maximum3 = new_maximum3
            tl.store(
                local_k
                + batch * LOCAL_K_BATCH_STRIDE
                + kv_head * LOCAL_K_HEAD_STRIDE
                + local_len * LOCAL_K_TOKEN_STRIDE
                + dim,
                current_key,
            )
            tl.store(
                local_v
                + batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + local_len * LOCAL_V_TOKEN_STRIDE
                + dim,
                current_value,
            )

    partial_row = query_row * SPLITS + split
    has_mass0 = denominator0 > 0.0
    tl.store(
        partial_out + partial_row * HEAD_DIM + dim,
        tl.where(has_mass0, accumulator0 / denominator0, 0.0),
    )
    tl.store(
        partial_lse + partial_row,
        tl.where(
            has_mass0,
            (maximum0 + tl.math.log2(denominator0)) * 0.6931471805599453,
            -float("inf"),
        ),
    )
    if KV_GROUP_SIZE > 1:
        has_mass1 = denominator1 > 0.0
        tl.store(
            partial_out + (partial_row + SPLITS) * HEAD_DIM + dim,
            tl.where(has_mass1, accumulator1 / denominator1, 0.0),
        )
        tl.store(
            partial_lse + partial_row + SPLITS,
            tl.where(
                has_mass1,
                (maximum1 + tl.math.log2(denominator1)) * 0.6931471805599453,
                -float("inf"),
            ),
        )
    if KV_GROUP_SIZE > 2:
        has_mass2 = denominator2 > 0.0
        tl.store(
            partial_out + (partial_row + 2 * SPLITS) * HEAD_DIM + dim,
            tl.where(has_mass2, accumulator2 / denominator2, 0.0),
        )
        tl.store(
            partial_lse + partial_row + 2 * SPLITS,
            tl.where(
                has_mass2,
                (maximum2 + tl.math.log2(denominator2)) * 0.6931471805599453,
                -float("inf"),
            ),
        )
    if KV_GROUP_SIZE > 3:
        has_mass3 = denominator3 > 0.0
        tl.store(
            partial_out + (partial_row + 3 * SPLITS) * HEAD_DIM + dim,
            tl.where(has_mass3, accumulator3 / denominator3, 0.0),
        )
        tl.store(
            partial_lse + partial_row + 3 * SPLITS,
            tl.where(
                has_mass3,
                (maximum3 + tl.math.log2(denominator3)) * 0.6931471805599453,
                -float("inf"),
            ),
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
    SEPARATE_LOCAL: tl.constexpr,
    FUSE_FINAL_REDUCE: tl.constexpr,
    INDEXED: tl.constexpr,
    INT8_STORAGE: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    batch = query_row // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    active_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
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

    # Assign whole routed posting lists round-robin to splits.  An inactive
    # route receives key_count=0 and therefore performs no page loop.
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
        key_count = tl.where(split == route % SPLITS, key_count, 0)
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
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
                leaf_index = tl.where(valid, leaf_index, 0)
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
        if split == 0:
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
def _reduce_split_decode_lod_attention_with_lse_kernel(
    partial_out,
    partial_lse,
    out,
    out_lse,
    VALUE_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    split = tl.arange(0, SPLITS)
    dim = tl.arange(0, VALUE_DIM)
    lse = tl.load(partial_lse + query_row * SPLITS + split)
    maximum = tl.max(lse, axis=0)
    has_mass = maximum > -float("inf")
    weights = tl.where(has_mass, tl.exp(lse - maximum), 0.0)
    denominator = tl.sum(weights, axis=0)
    values = tl.load(
        partial_out + (query_row * SPLITS + split[:, None]) * VALUE_DIM + dim[None, :]
    )
    result = tl.where(
        has_mass,
        tl.sum(weights[:, None] * values, axis=0) / denominator,
        0.0,
    )
    tl.store(out + query_row * VALUE_DIM + dim, result)
    tl.store(
        out_lse + query_row,
        tl.where(has_mass, maximum + tl.log(denominator), -float("inf")),
    )


@triton.jit
def _reduce_split_decode_lod_attention_kernel(
    partial_out,
    partial_lse,
    out,
    VALUE_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    split = tl.arange(0, SPLITS)
    dim = tl.arange(0, VALUE_DIM)
    lse = tl.load(partial_lse + query_row * SPLITS + split)
    maximum = tl.max(lse, axis=0)
    weights = tl.exp(lse - maximum)
    denominator = tl.sum(weights, axis=0)
    values = tl.load(
        partial_out + (query_row * SPLITS + split[:, None]) * VALUE_DIM + dim[None, :]
    )
    result = tl.sum(weights[:, None] * values, axis=0) / denominator
    tl.store(out + query_row * VALUE_DIM + dim, result)


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
    ROUTE_COUNT: tl.constexpr,
    SPLITS: tl.constexpr,
    ROUTE_SPLITS: tl.constexpr,
    INCLUDE_SEPARATE_LOCAL: tl.constexpr,
    FUSE_LOCAL_SCAN: tl.constexpr,
    INCLUDE_NEW: tl.constexpr,
    INCLUDE_SINK: tl.constexpr,
    SINK_LEN: tl.constexpr,
    LOCAL_BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
    USE_DOT: tl.constexpr,
    ADVANCE_LOCAL: tl.constexpr,
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
    for route in tl.static_range(0, ROUTE_COUNT):
        slot = tl.load(top_slots + query_row * ROUTE_COUNT + route).to(tl.int64)
        valid_slot = slot >= 0
        slot = tl.where(valid_slot, slot, 0)
        count = tl.load(
            counts
            + cache_batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE
        ).to(tl.float32)
        value = tl.load(
            state_v
            + cache_batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot * STATE_V_TOKEN_STRIDE
            + dim
        )
        mean_value = value.to(tl.float32) / count
        score = tl.load(top_scores + query_row * ROUTE_COUNT + route)
        mass = tl.where(valid_slot, tl.exp(score - full_coarse_lse), 0.0)
        selected_mass += mass
        selected_value += mass * mean_value
    remainder_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
    remainder_out = (remainder_out - selected_value) / remainder_mass
    remainder_lse = full_coarse_lse + tl.log(remainder_mass)

    # Fold branches sequentially. This keeps only one HEAD_DIM-wide value live
    # instead of materializing SPLITS x HEAD_DIM in the final-reduction program.
    maximum = remainder_lse
    denominator = tl.full((), 1.0, tl.float32)
    numerator = remainder_out.to(tl.float32)
    if INCLUDE_SEPARATE_LOCAL:
        local_lse = tl.load(separate_local_lse + query_row)
        local_value = tl.load(separate_local_out + query_row * HEAD_DIM + dim).to(
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
                partial_lse
                + query_row * SPLITS * ROUTE_SPLITS
                + branch_index
            )
            branch_value = tl.load(
                partial_out
                + (
                    query_row * SPLITS * ROUTE_SPLITS
                    + branch_index
                )
                * HEAD_DIM
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


@triton.jit
def _advance_decode_cache_lengths_kernel(
    cache_indices,
    local_lens,
    num_rows,
    increment: tl.constexpr,
):
    row = tl.program_id(0)
    if row < num_rows:
        cache_row = tl.load(cache_indices + row).to(tl.int64)
        length = tl.load(local_lens + cache_row)
        tl.store(local_lens + cache_row, length + increment)


def advance_decode_cache_lengths(
    cache_indices: torch.Tensor,
    local_lens: torch.Tensor,
    *,
    increment: int = 1,
) -> None:
    """Advance fixed-pool local lengths after a fused decode launch.

    ``cache_indices`` must be unique.  Keeping this as a separate launch makes
    every attention program observe the same pre-append length and remains
    safe to capture and replay in a CUDA graph.
    """
    if cache_indices.ndim != 1 or local_lens.ndim != 1:
        raise ValueError("decode cache indices and lengths must be vectors")
    if cache_indices.device != local_lens.device:
        raise ValueError("decode cache indices and lengths must share a device")
    if increment <= 0:
        raise ValueError("decode cache length increment must be positive")
    rows = int(cache_indices.numel())
    if rows:
        _advance_decode_cache_lengths_kernel[(rows,)](
            cache_indices,
            local_lens,
            rows,
            increment=increment,
            num_warps=1,
        )


def new_fused_decode_buffers(
    q: torch.Tensor,
    *,
    splits: int,
    state_capacity: int | None = None,
    route_group_size: int = 64,
    gqa_route_splits: int = 4,
) -> dict[str, torch.Tensor]:
    batch, query_heads, _, value_dim = q.shape
    if gqa_route_splits not in {4, 8, 16, 32}:
        raise ValueError("GQA cooperative route splits must be 4, 8, 16, or 32")
    buffers = {
        "cache_indices": torch.arange(batch, dtype=torch.long, device=q.device),
        "local_lens": torch.empty(batch, dtype=torch.int32, device=q.device),
        "partial_out": torch.empty(
            batch,
            query_heads,
            splits,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        ),
        "partial_lse": torch.empty(
            batch,
            query_heads,
            splits,
            dtype=torch.float32,
            device=q.device,
        ),
        "gqa_local_partial_out": torch.empty(
            batch,
            query_heads,
            32,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        ),
        "gqa_local_partial_lse": torch.empty(
            batch,
            query_heads,
            32,
            dtype=torch.float32,
            device=q.device,
        ),
        "gqa_route_partial_out": torch.empty(
            batch,
            query_heads,
            8,
            gqa_route_splits,
            value_dim,
            dtype=torch.float32,
            device=q.device,
        ),
        "gqa_route_partial_lse": torch.empty(
            batch,
            query_heads,
            8,
            gqa_route_splits,
            dtype=torch.float32,
            device=q.device,
        ),
        "output": torch.empty_like(q),
    }
    if state_capacity is not None:
        max_groups = triton.cdiv(state_capacity, route_group_size)
        buffers.update(
            route_candidate_scores=torch.empty(
                batch,
                query_heads,
                max_groups,
                8,
                dtype=torch.float32,
                device=q.device,
            ),
            route_candidate_indices=torch.empty(
                batch,
                query_heads,
                max_groups,
                8,
                dtype=torch.long,
                device=q.device,
            ),
            route_group_out=torch.empty(
                batch,
                query_heads,
                max_groups,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            route_group_lse=torch.empty(
                batch,
                query_heads,
                max_groups,
                dtype=torch.float32,
                device=q.device,
            ),
            route_top_slots=torch.empty(
                batch,
                query_heads,
                1,
                8,
                dtype=torch.long,
                device=q.device,
            ),
            route_top_scores=torch.empty(
                batch,
                query_heads,
                1,
                8,
                dtype=torch.float32,
                device=q.device,
            ),
            coarse_out=torch.empty(
                batch,
                query_heads,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            coarse_lse=torch.empty(
                batch,
                query_heads,
                dtype=torch.float32,
                device=q.device,
            ),
            route_local_out=torch.empty(
                batch,
                query_heads,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            ),
            route_local_lse=torch.empty(
                batch,
                query_heads,
                dtype=torch.float32,
                device=q.device,
            ),
            completion=torch.zeros(
                batch,
                query_heads,
                dtype=torch.int32,
                device=q.device,
            ),
        )
    return buffers


def fused_decode_paged_lod_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor | None,
    *,
    sink_k: torch.Tensor | None = None,
    sink_v: torch.Tensor | None = None,
    state_len: int,
    local_len: int | None = None,
    cache_indices: torch.Tensor | None = None,
    local_lens: torch.Tensor | None = None,
    new_k: torch.Tensor | None = None,
    new_v: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_n: int = 16,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    split_kv: int = 1,
    buffers: dict[str, torch.Tensor] | None = None,
    use_dot: bool = False,
    fuse_state_route: bool = False,
    route_group_size: int = 64,
    route_num_warps: int = 4,
    route_reduce_num_warps: int = 4,
    final_reduce_num_warps: int = 4,
    fuse_final_reduce: bool = False,
    route_use_dot: bool = False,
    route_gqa_grouped: bool = False,
    gqa_cooperative_leaf: bool = True,
    gqa_cooperative_hip: bool = False,
    gqa_cooperative_route_splits: int = 4,
    gqa_cooperative_adaptive_splits: bool = False,
    gqa_cooperative_fused_reduce: bool = False,
    protected_len: int = 0,
    max_leaf_tokens: int | None = None,
    route_top_p: float | None = None,
    route_residual_mass: float | None = None,
    reuse_residual_local_attention: bool = False,
    route_residual_use_state_bound: bool = False,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
    recursive_page_cache: dict[str, torch.Tensor | int] | None = None,
    flat_page_indices: torch.Tensor | None = None,
    flat_page_k_scales: torch.Tensor | None = None,
    flat_page_v_scales: torch.Tensor | None = None,
    recursive_quant_group_size: int = 32,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse coarse, exact-leaf, local, and branch-merge decode attention."""
    batch, query_heads, query_len, head_dim = q.shape
    if query_len != 1:
        raise ValueError("fused LOD decode attention requires one query token")
    kv_heads = int(state_k.size(1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if int(state_v.size(-1)) != head_dim:
        raise ValueError("fused LOD decode requires equal QK/V dimensions")
    page_shape = (
        recursive_page_cache.get("page_indices")
        if recursive_page_cache is not None
        else (flat_page_indices if flat_page_indices is not None else page_k)
    )
    if not isinstance(page_shape, torch.Tensor) or int(page_shape.size(3)) != 16:
        raise ValueError("fused LOD decode requires 16-token leaf pages")
    flat_int8 = page_k.dtype == torch.int8 or page_v.dtype == torch.int8
    if flat_int8:
        if recursive_page_cache is not None:
            raise ValueError("decode INT8 storage requires flat two-tier leaves")
        if page_k.dtype != torch.int8 or page_v.dtype != torch.int8:
            raise TypeError("decode INT8 storage requires both K and V in INT8")
        if flat_page_k_scales is None or flat_page_v_scales is None:
            raise ValueError("decode INT8 storage requires per-token K/V scales")
        if tuple(flat_page_k_scales.shape) != tuple(page_k.shape[:-1]):
            raise ValueError("decode INT8 K scales do not match leaf storage")
        if tuple(flat_page_v_scales.shape) != tuple(page_v.shape[:-1]):
            raise ValueError("decode INT8 V scales do not match leaf storage")
    if local_len is None:
        local_len = int(local_k.size(2))
    if local_len < 0 or local_len > int(local_k.size(2)):
        raise ValueError("active local length exceeds its allocated cache")
    if cache_indices is None:
        if int(state_k.size(0)) != batch:
            raise ValueError(
                "cache indices are required when cache and query batches differ"
            )
        if buffers is not None and "cache_indices" in buffers:
            cache_indices = buffers["cache_indices"][:batch]
        else:
            cache_indices = torch.arange(batch, dtype=torch.long, device=q.device)
    if tuple(cache_indices.shape) != (batch,):
        raise ValueError("cache indices must contain one stable slot per query row")
    ragged_local_lens = local_lens is not None
    if local_lens is None:
        if (
            buffers is not None
            and "local_lens" in buffers
            and int(buffers["local_lens"].numel()) >= int(state_k.size(0))
        ):
            local_lens = buffers["local_lens"][: int(state_k.size(0))]
        else:
            local_lens = torch.empty(
                int(state_k.size(0)), dtype=torch.int32, device=q.device
            )
        local_lens.fill_(local_len)
    elif tuple(local_lens.shape) != (int(state_k.size(0)),):
        raise ValueError("local lengths must contain one value per cache slot")
    include_new = new_k is not None or new_v is not None
    include_sink = sink_k is not None or sink_v is not None

    def timing_begin() -> torch.cuda.Event | None:
        if timing_events is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def timing_end(name: str, begin: torch.cuda.Event | None) -> None:
        if begin is None or timing_events is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        timing_events.setdefault(name, []).append((begin, end))

    if include_new and (new_k is None or new_v is None):
        raise ValueError("new decode K and V must be provided together")
    if include_sink and (sink_k is None or sink_v is None):
        raise ValueError("separate sink K and V must be provided together")
    if include_sink:
        if tuple(sink_k.shape[:2]) != (int(state_k.size(0)), kv_heads):
            raise ValueError("separate sink K has the wrong batch/head shape")
        if tuple(sink_v.shape[:3]) != tuple(sink_k.shape[:3]):
            raise ValueError("separate sink K/V shapes do not match")
        if not fuse_state_route or split_kv == 1:
            raise ValueError(
                "separate sink fusion requires routed split decode attention"
            )
        # The sink is folded into this existing final reduction. The atomic
        # leaf-kernel reduction cannot see the side cache.
        fuse_final_reduce = False
    else:
        sink_k = state_k[..., :1, :]
        sink_v = state_v[..., :1, :]
    if include_new:
        if tuple(new_k.shape[:3]) != (batch, kv_heads, 1):
            raise ValueError("new decode K has the wrong shape")
        if tuple(new_v.shape[:3]) != (batch, kv_heads, 1):
            raise ValueError("new decode V has the wrong shape")
        if not ragged_local_lens and local_len >= int(local_k.size(2)):
            raise ValueError("local decode cache has no append capacity")
    else:
        # Triton still requires valid pointer arguments for a constexpr-dead
        # branch.  These aliases are never read or written.
        new_k = state_k[..., :1, :]
        new_v = state_v[..., :1, :]
    if split_kv not in {1, 8, 16, 32}:
        raise ValueError("fused LOD decode split count must be 1, 8, 16, or 32")
    if gqa_cooperative_route_splits not in {4, 8, 16, 32}:
        raise ValueError("GQA cooperative route splits must be 4, 8, 16, or 32")
    if fuse_state_route and split_kv == 1:
        raise ValueError("fused state routing requires split decode attention")
    if route_group_size not in {8, 16, 32, 64}:
        raise ValueError("decode route group size must be 8, 16, 32, or 64")
    if protected_len < 0 or protected_len + 8 > state_len:
        raise ValueError("protected state leaves too few decode routing candidates")
    if max_leaf_tokens is not None and max_leaf_tokens <= 0:
        raise ValueError("maximum routed leaves must be positive")
    if route_top_p is not None and not 0.0 < route_top_p <= 1.0:
        raise ValueError("decode route top-p must lie in (0, 1]")
    if route_residual_mass is not None and not 0.0 < route_residual_mass <= 1.0:
        raise ValueError("decode route residual mass must lie in (0, 1]")
    if route_top_p is not None and route_residual_mass is not None:
        raise ValueError("decode route mass criteria are mutually exclusive")
    if (
        route_top_p is not None or route_residual_mass is not None
    ) and not fuse_state_route:
        raise ValueError("dynamic decode routing requires fused state routing")
    if route_residual_mass is not None and fuse_final_reduce:
        raise ValueError("full-mass dynamic routing requires separate final reduction")
    if reuse_residual_local_attention and route_residual_mass is None:
        raise ValueError("local-attention reuse requires residual-mass routing")
    if route_residual_use_state_bound and route_residual_mass is None:
        raise ValueError("state-bound routing requires residual-mass routing")
    if route_residual_use_state_bound and reuse_residual_local_attention:
        raise ValueError("state-bound routing cannot reuse uncomputed local attention")
    expected_output = (batch, query_heads, 1, head_dim)
    if output_buffer is not None and (
        tuple(output_buffer.shape) != expected_output
        or output_buffer.dtype != q.dtype
        or output_buffer.device != q.device
    ):
        raise ValueError("fused LOD decode output buffer has the wrong geometry")
    if split_kv == 1:
        output = (
            output_buffer
            if output_buffer is not None
            else torch.empty(expected_output, dtype=q.dtype, device=q.device)
        )
    else:
        if buffers is None:
            buffers = new_fused_decode_buffers(
                q,
                splits=split_kv,
                state_capacity=(int(state_k.size(2)) if fuse_state_route else None),
                route_group_size=route_group_size,
                gqa_route_splits=gqa_cooperative_route_splits,
            )
        output = buffers["output"] if output_buffer is None else output_buffer
        partial_out = buffers["partial_out"]
        partial_lse = buffers["partial_lse"]
        expected_partial = (batch, query_heads, split_kv, head_dim)
        if tuple(partial_out.shape) != expected_partial:
            raise ValueError("fused LOD decode buffers have the wrong shape")
        if fuse_state_route:
            required = (
                "route_candidate_scores",
                "route_candidate_indices",
                "route_group_out",
                "route_group_lse",
                "route_top_slots",
                "route_top_scores",
                "coarse_out",
                "coarse_lse",
                "route_local_out",
                "route_local_lse",
            )
            if any(name not in buffers for name in required):
                raise ValueError("fused state-routing buffers are missing")
            group_size = route_group_size
            active_groups = triton.cdiv(state_len, group_size)
            max_groups = int(buffers["route_group_lse"].size(2))
            if active_groups > max_groups:
                raise ValueError("fused state-routing buffers are too small")
            if route_gqa_grouped:
                scalar_gqa = not route_use_dot and group_size <= 16
                route_kernel = (
                    _decode_route_coarse_scalar_gqa_groups_kernel
                    if scalar_gqa
                    else _decode_route_coarse_gqa_groups_kernel
                )
            else:
                scalar_gqa = False
                route_kernel = _decode_route_coarse_groups_kernel
            score_use_dot = route_use_dot or (route_gqa_grouped and not scalar_gqa)
            route_rows = batch * kv_heads if route_gqa_grouped else batch * query_heads
            route_groups_begin = timing_begin()
            route_kernel[(route_rows, active_groups)](
                q,
                state_k,
                state_v,
                counts,
                cache_indices,
                buffers["route_candidate_scores"],
                buffers["route_candidate_indices"],
                buffers["route_group_out"],
                buffers["route_group_lse"],
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
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
                HEAD_DIM=head_dim,
                SCALE=float(scale),
                GROUP_N=group_size,
                MAX_GROUPS=max_groups,
                PROTECTED_LEN=protected_len,
                MAX_LEAF_TOKENS=max_leaf_tokens or 0,
                USE_DOT=score_use_dot,
                num_warps=route_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("route_groups", route_groups_begin)
            route_reduce_begin = timing_begin()
            _reduce_decode_route_coarse_kernel[(batch * query_heads,)](
                buffers["route_candidate_scores"],
                buffers["route_candidate_indices"],
                buffers["route_group_out"],
                buffers["route_group_lse"],
                buffers["route_top_slots"],
                buffers["route_top_scores"],
                buffers["coarse_out"],
                buffers["coarse_lse"],
                active_groups,
                HEAD_DIM=head_dim,
                ROUTE_COUNT=8,
                MAX_GROUPS=max_groups,
                CANDIDATE_TILE=1024,
                num_warps=route_reduce_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("route_reduce", route_reduce_begin)
            top_slots = buffers["route_top_slots"]
            if route_residual_mass is not None:
                route_mask_begin = timing_begin()
                if route_residual_use_state_bound:
                    _mask_decode_routes_residual_lse_kernel[(batch * query_heads,)](
                        top_slots,
                        buffers["route_top_scores"],
                        buffers["coarse_lse"],
                        float(route_residual_mass),
                        ROUTE_COUNT=8,
                        num_warps=2,
                        waves_per_eu=waves_per_eu,
                    )
                else:
                    _mask_decode_routes_residual_mass_kernel[(batch * query_heads,)](
                        q,
                        local_k,
                        local_v,
                        cache_indices,
                        local_lens,
                        new_k,
                        new_v,
                        top_slots,
                        buffers["route_top_scores"],
                        buffers["coarse_lse"],
                        buffers["route_local_out"],
                        buffers["route_local_lse"],
                        float(route_residual_mass),
                        local_k.stride(0),
                        local_k.stride(1),
                        local_k.stride(2),
                        local_v.stride(0),
                        local_v.stride(1),
                        local_v.stride(2),
                        new_k.stride(0),
                        new_k.stride(1),
                        new_v.stride(0),
                        new_v.stride(1),
                        local_len,
                        QUERY_HEADS=query_heads,
                        KV_GROUP_SIZE=kv_group_size,
                        HEAD_DIM=head_dim,
                        ROUTE_COUNT=8,
                        LOCAL_BLOCK_N=32,
                        SCALE=float(scale),
                        INCLUDE_NEW=include_new,
                        COMPUTE_LOCAL_OUTPUT=reuse_residual_local_attention,
                        APPLY_ROUTE_MASK=True,
                        num_warps=1,
                        waves_per_eu=waves_per_eu,
                    )
                timing_end("route_mask", route_mask_begin)
            elif route_top_p is not None and route_top_p < 1.0:
                route_mask_begin = timing_begin()
                _mask_decode_routes_top_p_kernel[(batch * query_heads,)](
                    top_slots,
                    buffers["route_top_scores"],
                    float(route_top_p),
                    ROUTE_COUNT=8,
                    num_warps=1,
                    waves_per_eu=waves_per_eu,
                )
                timing_end("route_mask", route_mask_begin)
        if recursive_page_cache is not None:
            if not fuse_state_route:
                raise ValueError("fused recursive decode requires fused state routing")
            if split_kv != int(top_slots.size(-1)):
                raise ValueError("fused recursive decode requires one split per route")

            def cache_tensor(name: str) -> torch.Tensor:
                value = recursive_page_cache.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"fused recursive decode cache is missing {name}")
                return value

            # Residual-mass routing can reuse the local output it had to form
            # while choosing routes. Other modes compute the bounded local
            # branch separately; folding its scan into the final reducer makes
            # that kernel register-bound on the target ROCm geometry.
            reuse_separate_local = bool(
                route_residual_mass is not None and reuse_residual_local_attention
            )
            if not reuse_separate_local:
                local_begin = timing_begin()
                _mask_decode_routes_residual_mass_kernel[(batch * query_heads,)](
                    q,
                    local_k,
                    local_v,
                    cache_indices,
                    local_lens,
                    new_k,
                    new_v,
                    top_slots,
                    buffers["route_top_scores"],
                    buffers["coarse_lse"],
                    buffers["route_local_out"],
                    buffers["route_local_lse"],
                    1.0,
                    local_k.stride(0),
                    local_k.stride(1),
                    local_k.stride(2),
                    local_v.stride(0),
                    local_v.stride(1),
                    local_v.stride(2),
                    new_k.stride(0),
                    new_k.stride(1),
                    new_v.stride(0),
                    new_v.stride(1),
                    local_len,
                    QUERY_HEADS=query_heads,
                    KV_GROUP_SIZE=kv_group_size,
                    HEAD_DIM=head_dim,
                    ROUTE_COUNT=8,
                    LOCAL_BLOCK_N=32,
                    SCALE=float(scale),
                    INCLUDE_NEW=include_new,
                    COMPUTE_LOCAL_OUTPUT=True,
                    APPLY_ROUTE_MASK=False,
                    num_warps=1,
                    waves_per_eu=waves_per_eu,
                )
                timing_end("recursive_local", local_begin)

            quantized_attention = bool(
                recursive_page_cache.get("quantization_finalized", False)
            )
            quantized_summaries = bool(
                recursive_page_cache.get("summary_quantization_finalized", False)
            )
            recursive_begin = timing_begin()
            recursive_out, recursive_lse = query_major_indexed_residual_page_attention(
                q,
                state_k,
                state_v,
                counts,
                cache_tensor("leaf_k"),
                cache_tensor("leaf_v"),
                cache_tensor("page_indices"),
                cache_tensor("page_sum_k"),
                cache_tensor("page_sum_v"),
                cache_tensor("page_counts"),
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                top_slots,
                cache_indices=cache_indices,
                kv_group_size=kv_group_size,
                scale=scale,
                hash_probes=hash_probes,
                page_block_n=block_n,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
                quantized_leaf_k=(
                    cache_tensor("quantized_leaf_k") if quantized_attention else None
                ),
                quantized_leaf_v=(
                    cache_tensor("quantized_leaf_v") if quantized_attention else None
                ),
                page_k_scales=(
                    cache_tensor("page_k_scales") if quantized_attention else None
                ),
                page_v_scales=(
                    cache_tensor("page_v_scales") if quantized_attention else None
                ),
                page_quantized_counts=(
                    cache_tensor("page_quantized_counts")
                    if quantized_attention
                    else None
                ),
                quantized_page_sum_k=(
                    cache_tensor("quantized_page_sum_k")
                    if quantized_summaries
                    else None
                ),
                quantized_page_sum_v=(
                    cache_tensor("quantized_page_sum_v")
                    if quantized_summaries
                    else None
                ),
                page_sum_k_scales=(
                    cache_tensor("page_sum_k_scales") if quantized_summaries else None
                ),
                page_sum_v_scales=(
                    cache_tensor("page_sum_v_scales") if quantized_summaries else None
                ),
                quant_group_size=recursive_quant_group_size,
                quant_bits=int(recursive_page_cache.get("leaf_quant_bits", 4)),
                output_buffer=partial_out,
                lse_buffer=partial_lse,
                route_parallel=True,
            )
            timing_end("recursive_leaf", recursive_begin)
            final_reduce_begin = timing_begin()
            _reduce_routed_split_decode_lod_attention_kernel[(batch * query_heads,)](
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
                buffers["route_top_scores"],
                buffers["coarse_out"],
                buffers["coarse_lse"],
                recursive_out,
                recursive_lse,
                buffers["route_local_out"],
                buffers["route_local_lse"],
                output,
                sink_k.stride(0),
                sink_k.stride(1),
                sink_k.stride(2),
                sink_v.stride(0),
                sink_v.stride(1),
                sink_v.stride(2),
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
                state_v.stride(0),
                state_v.stride(1),
                state_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                QUERY_HEADS=query_heads,
                KV_GROUP_SIZE=kv_group_size,
                HEAD_DIM=head_dim,
                ROUTE_COUNT=int(top_slots.size(-1)),
                SPLITS=split_kv,
                ROUTE_SPLITS=1,
                INCLUDE_SEPARATE_LOCAL=True,
                FUSE_LOCAL_SCAN=False,
                INCLUDE_NEW=False,
                INCLUDE_SINK=include_sink,
                SINK_LEN=int(sink_k.size(2)),
                LOCAL_BLOCK_N=32,
                SCALE=float(scale),
                USE_DOT=score_use_dot,
                ADVANCE_LOCAL=include_new and ragged_local_lens,
                num_warps=final_reduce_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("final_reduce", final_reduce_begin)
            return output
        fused_completion = (
            buffers["completion"]
            if fuse_state_route and fuse_final_reduce
            else partial_lse
        )
        if top_slots is None:
            raise ValueError("fused LOD decode requires routed state slots")
        leaf_begin = timing_begin()
        cooperative_hip_eligible = False
        if gqa_cooperative_hip and q.is_cuda:
            from model.kernels.gqa_cooperative_decode import (
                gqa_cooperative_decode_available,
            )

            device_index = q.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            cooperative_hip_eligible = bool(
                kv_group_size == 4
                and head_dim == 256
                and q.dtype == torch.bfloat16
                and (
                    (
                        page_k.dtype == torch.bfloat16
                        and page_v.dtype == torch.bfloat16
                    )
                    or (
                        flat_int8
                        and flat_page_k_scales is not None
                        and flat_page_v_scales is not None
                    )
                )
                and hash_probes in {-1, 0}
                and gqa_cooperative_decode_available(device_index)
            )
        cooperative_leaf = bool(
            gqa_cooperative_leaf
            and fuse_state_route
            and not fuse_final_reduce
            and not use_dot
            and 1 < kv_group_size <= 4
            and int(top_slots.size(-1)) == 8
            and split_kv == 8
            and "gqa_local_partial_out" in buffers
            and "gqa_local_partial_lse" in buffers
            and "gqa_route_partial_out" in buffers
            and "gqa_route_partial_lse" in buffers
            and (not gqa_cooperative_hip or cooperative_hip_eligible or flat_int8)
        )
        cooperative_separate_local = False
        final_partial_out = partial_out
        final_partial_lse = partial_lse
        final_route_splits = 1
        if cooperative_leaf:
            reuse_separate_local = bool(
                route_residual_mass is not None and reuse_residual_local_attention
            )
            if not reuse_separate_local:
                local_partial_out = buffers["gqa_local_partial_out"]
                local_partial_lse = buffers["gqa_local_partial_lse"]
                _gqa_cooperative_split_decode_local_attention_kernel[
                    (batch * kv_heads, 32)
                ](
                    q,
                    cache_indices,
                    local_lens,
                    local_k,
                    local_v,
                    new_k,
                    new_v,
                    local_partial_out,
                    local_partial_lse,
                    local_k.stride(0),
                    local_k.stride(1),
                    local_k.stride(2),
                    local_v.stride(0),
                    local_v.stride(1),
                    local_v.stride(2),
                    new_k.stride(0),
                    new_k.stride(1),
                    new_v.stride(0),
                    new_v.stride(1),
                    local_len,
                    QUERY_HEADS=query_heads,
                    KV_HEADS=kv_heads,
                    KV_GROUP_SIZE=kv_group_size,
                    BLOCK_M=triton.next_power_of_2(kv_group_size),
                    HEAD_DIM=head_dim,
                    LOCAL_SPLITS=32,
                    SCALE_LOG2=float(scale) * math.log2(math.e),
                    BLOCK_N=block_n,
                    INCLUDE_NEW=include_new,
                    num_warps=num_warps,
                    waves_per_eu=waves_per_eu,
                )
                _reduce_split_decode_lod_attention_with_lse_kernel[
                    (batch * query_heads,)
                ](
                    local_partial_out,
                    local_partial_lse,
                    buffers["route_local_out"],
                    buffers["route_local_lse"],
                    VALUE_DIM=head_dim,
                    SPLITS=32,
                    num_warps=final_reduce_num_warps,
                    waves_per_eu=waves_per_eu,
                )
            use_cooperative_hip = cooperative_hip_eligible
            if use_cooperative_hip:
                from model.kernels.gqa_cooperative_decode import (
                    gqa_cooperative_decode,
                )

                route_partial_out = buffers["gqa_route_partial_out"]
                route_partial_lse = buffers["gqa_route_partial_lse"]
                route_partial_lse.fill_(float("-inf"))
                gqa_cooperative_decode(
                    q,
                    cache_indices,
                    page_k,
                    page_v,
                    slot_pages,
                    overflow_page_values,
                    slot_lengths,
                    top_slots,
                    route_partial_out,
                    route_partial_lse,
                    quantized_q_scratch=buffers["gqa_local_partial_out"],
                    query_scale_scratch=buffers["gqa_local_partial_lse"],
                    page_indices=flat_page_indices,
                    page_k_scales=(flat_page_k_scales if flat_int8 else None),
                    page_v_scales=(flat_page_v_scales if flat_int8 else None),
                    scale_log2=float(scale) * math.log2(math.e),
                    page_lookup_mode=hash_probes,
                    route_splits=gqa_cooperative_route_splits,
                    adaptive_splits=gqa_cooperative_adaptive_splits,
                )
                if (
                    gqa_cooperative_fused_reduce
                    and gqa_cooperative_route_splits <= 8
                ):
                    final_partial_out = route_partial_out
                    final_partial_lse = route_partial_lse
                    final_route_splits = gqa_cooperative_route_splits
                else:
                    partial_lse.fill_(float("-inf"))
                    _reduce_split_decode_lod_attention_with_lse_kernel[
                        (batch * query_heads * 8,)
                    ](
                        route_partial_out,
                        route_partial_lse,
                        partial_out,
                        partial_lse,
                        VALUE_DIM=head_dim,
                        SPLITS=gqa_cooperative_route_splits,
                        num_warps=final_reduce_num_warps,
                        waves_per_eu=waves_per_eu,
                    )
            else:
                partial_lse.fill_(float("-inf"))
                _gqa_cooperative_route_decode_paged_leaf_attention_kernel[
                    (
                        batch * kv_heads,
                        triton.next_power_of_2(kv_group_size) * 8,
                    )
                ](
                    q,
                    cache_indices,
                    page_k,
                    page_v,
                    flat_page_indices if flat_page_indices is not None else page_k,
                    flat_page_k_scales if flat_int8 else page_k,
                    flat_page_v_scales if flat_int8 else page_v,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    slot_lengths,
                    top_slots,
                    partial_out,
                    partial_lse,
                    top_slots.stride(0),
                    top_slots.stride(1),
                    QUERY_HEADS=query_heads,
                    KV_HEADS=kv_heads,
                    KV_GROUP_SIZE=kv_group_size,
                    BLOCK_M=triton.next_power_of_2(kv_group_size),
                    CACHE_BATCHES=int(page_k.size(0)),
                    PAGE_CAPACITY=int(page_shape.size(2)),
                    LEAF_CAPACITY=(
                        int(page_k.size(2)) if flat_page_indices is not None else 1
                    ),
                    STATE_CAPACITY=int(slot_pages.size(2)),
                    INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                    HASH_CAPACITY=int(overflow_page_values.size(2)),
                    HASH_PROBES=hash_probes,
                    HEAD_DIM=head_dim,
                    PAGE_SIZE=int(page_shape.size(3)),
                    ROUTE_COUNT=8,
                    SCALE_LOG2=float(scale) * math.log2(math.e),
                    BLOCK_N=block_n,
                    INDEXED=flat_page_indices is not None,
                    INT8_STORAGE=flat_int8,
                    num_warps=num_warps,
                    waves_per_eu=waves_per_eu,
                )
            cooperative_separate_local = True
        else:
            _split_decode_paged_lod_attention_kernel[(batch * query_heads, split_kv)](
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
                flat_page_indices if flat_page_indices is not None else page_k,
                flat_page_k_scales if flat_int8 else page_k,
                flat_page_v_scales if flat_int8 else page_v,
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
                (buffers["route_top_scores"] if fuse_state_route else partial_lse),
                buffers["coarse_out"] if fuse_state_route else partial_out,
                buffers["coarse_lse"] if fuse_state_route else partial_lse,
                output,
                fused_completion,
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
                state_v.stride(0),
                state_v.stride(1),
                state_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                top_slots.stride(0),
                top_slots.stride(1),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                0 if fuse_state_route else state_len,
                local_len,
                QUERY_HEADS=query_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=kv_group_size,
                PAGE_CAPACITY=int(page_shape.size(2)),
                LEAF_CAPACITY=(
                    int(page_k.size(2)) if flat_page_indices is not None else 1
                ),
                STATE_CAPACITY=int(slot_pages.size(2)),
                INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=hash_probes,
                HEAD_DIM=head_dim,
                VALUE_DIM=head_dim,
                PAGE_SIZE=int(page_shape.size(3)),
                ROUTE_COUNT=int(top_slots.size(-1)),
                SPLITS=split_kv,
                SCALE_LOG2=float(scale) * math.log2(math.e),
                BLOCK_N=block_n,
                USE_DOT=use_dot,
                INCLUDE_NEW=include_new,
                SEPARATE_LOCAL=(
                    route_residual_mass is not None and reuse_residual_local_attention
                ),
                FUSE_FINAL_REDUCE=fuse_state_route and fuse_final_reduce,
                INDEXED=flat_page_indices is not None,
                INT8_STORAGE=flat_int8,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )
        timing_end("leaf_local", leaf_begin)
        if fuse_state_route and not fuse_final_reduce:
            final_reduce_begin = timing_begin()
            _reduce_routed_split_decode_lod_attention_kernel[(batch * query_heads,)](
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
                buffers["route_top_scores"],
                buffers["coarse_out"],
                buffers["coarse_lse"],
                final_partial_out,
                final_partial_lse,
                (
                    buffers["route_local_out"]
                    if reuse_residual_local_attention or cooperative_separate_local
                    else partial_out
                ),
                (
                    buffers["route_local_lse"]
                    if reuse_residual_local_attention or cooperative_separate_local
                    else partial_lse
                ),
                output,
                sink_k.stride(0),
                sink_k.stride(1),
                sink_k.stride(2),
                sink_v.stride(0),
                sink_v.stride(1),
                sink_v.stride(2),
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
                state_v.stride(0),
                state_v.stride(1),
                state_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                QUERY_HEADS=query_heads,
                KV_GROUP_SIZE=kv_group_size,
                HEAD_DIM=head_dim,
                ROUTE_COUNT=int(top_slots.size(-1)),
                SPLITS=split_kv,
                ROUTE_SPLITS=final_route_splits,
                INCLUDE_SEPARATE_LOCAL=(
                    reuse_residual_local_attention or cooperative_separate_local
                ),
                FUSE_LOCAL_SCAN=False,
                INCLUDE_NEW=False,
                INCLUDE_SINK=include_sink,
                SINK_LEN=int(sink_k.size(2)),
                LOCAL_BLOCK_N=32,
                SCALE=float(scale),
                USE_DOT=score_use_dot,
                ADVANCE_LOCAL=include_new and ragged_local_lens,
                num_warps=final_reduce_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("final_reduce", final_reduce_begin)
        elif not fuse_state_route:
            final_reduce_begin = timing_begin()
            _reduce_split_decode_lod_attention_kernel[(batch * query_heads,)](
                partial_out,
                partial_lse,
                output,
                VALUE_DIM=head_dim,
                SPLITS=split_kv,
                num_warps=final_reduce_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("final_reduce", final_reduce_begin)
        return output
    if top_slots is None:
        raise ValueError("fused LOD decode requires routed state slots")
    _fused_decode_paged_lod_attention_kernel[(batch * query_heads,)](
        q,
        state_k,
        state_v,
        counts,
        local_k,
        local_v,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        new_k,
        new_v,
        output,
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        state_v.stride(0),
        state_v.stride(1),
        state_v.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        local_k.stride(0),
        local_k.stride(1),
        local_k.stride(2),
        local_v.stride(0),
        local_v.stride(1),
        local_v.stride(2),
        top_slots.stride(0),
        top_slots.stride(1),
        new_k.stride(0),
        new_k.stride(1),
        new_v.stride(0),
        new_v.stride(1),
        state_len,
        local_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=int(page_k.size(2)),
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=head_dim,
        PAGE_SIZE=int(page_k.size(3)),
        ROUTE_COUNT=int(top_slots.size(-1)),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_N=block_n,
        USE_DOT=use_dot,
        INCLUDE_NEW=include_new,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return output


def paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    page_indices: torch.Tensor | None = None,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    int8_pv_mma: bool = True,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    tiny_expert_max: int = 0,
    tiny_block_m: int = 8,
    tiny_num_warps: int = 1,
    long_expert_threshold: int = 0,
    long_expert_splits: int = 1,
    reduce_num_warps: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to the exact leaves of every routed slot and merge by LSE."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("paged leaf Triton attention is forward-only")
    batch, query_heads, query_len, head_dim = q.shape
    route_count = int(top_slots.size(-1))
    kv_heads = int(page_k.size(1))
    value_dim = int(page_v.size(-1))
    indexed = page_indices is not None
    page_size = int(page_indices.size(3)) if indexed else int(page_k.size(3))
    page_capacity = int(page_indices.size(2)) if indexed else int(page_k.size(2))
    state_capacity = int(slot_pages.size(2))
    if page_size != 16:
        raise ValueError("paged leaf Triton attention requires 16-token pages")
    if head_dim != value_dim:
        raise ValueError("paged leaf Triton attention requires equal QK/V dimensions")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    int8_mma = page_k.dtype == torch.int8 or page_v.dtype == torch.int8
    if int8_mma:
        if page_k.dtype != torch.int8 or page_v.dtype != torch.int8:
            raise TypeError("INT8 leaf MMA requires both K and V in signed INT8")
        if page_k_scales is None or page_v_scales is None:
            raise ValueError("INT8 leaf MMA requires per-token K/V scales")
        expected_scale_shape = tuple(page_k.shape[:-1])
        if tuple(page_k_scales.shape) != expected_scale_shape:
            raise ValueError("INT8 leaf K scales do not match page storage")
        if tuple(page_v_scales.shape) != tuple(page_v.shape[:-1]):
            raise ValueError("INT8 leaf V scales do not match page storage")
        if head_dim % 32 or value_dim % 32 or block_n % 32:
            raise ValueError("INT8 leaf MMA requires dimensions divisible by 32")
    elif page_k_scales is not None or page_v_scales is not None:
        raise ValueError("BF16 leaf attention received INT8 scale tensors")
    if tiny_expert_max:
        if tiny_expert_max not in (4, 8, 16):
            raise ValueError("tiny expert attention supports N<=4, N<=8, or N<=16")
        if not indexed:
            raise ValueError("tiny expert attention requires virtual indexed leaves")
        if int8_mma:
            raise ValueError("tiny expert attention currently requires BF16 K/V")
        if tiny_block_m not in (1, 2, 4, 8, 16):
            raise ValueError("tiny expert BLOCK_M must be one of 1, 2, 4, 8, 16")
    if long_expert_splits not in (1, 2, 4, 8):
        raise ValueError("long expert split count must be 1, 2, 4, or 8")
    split_long_experts = long_expert_splits > 1
    if split_long_experts:
        if long_expert_threshold <= 0:
            raise ValueError("split-N expert attention requires a positive threshold")
        if not indexed:
            raise ValueError("split-N expert attention requires virtual indexed leaves")
        if int8_mma:
            raise ValueError("split-N expert attention currently requires BF16 K/V")
    compound_expert_routing = bool(tiny_expert_max or split_long_experts)
    if reduce_num_warps not in (1, 2, 4, 8):
        raise ValueError("expert route reduction warps must be one of 1, 2, 4, 8")

    boundaries: list[torch.cuda.Event] = []

    def record_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            boundaries.append(event)

    record_boundary()
    dispatch_boundaries: list[torch.cuda.Event] = boundaries[-1:] if boundaries else []

    def record_dispatch_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            dispatch_boundaries.append(event)

    with torch.no_grad():
        rows = batch * query_heads * query_len
        kernel_q = q
        query_scales = q
        if int8_mma:
            source_q = q.contiguous()
            kernel_q = torch.empty_like(source_q, dtype=torch.int8)
            query_scales = torch.empty(rows, dtype=q.dtype, device=q.device)
            expert_id = torch.empty(
                rows * route_count, dtype=torch.int32, device=q.device
            )
            query_prepare_block_m = 16
            _prepare_int8_attention_queries_kernel[
                (triton.cdiv(rows, query_prepare_block_m),)
            ](
                source_q,
                kernel_q,
                query_scales,
                top_slots,
                expert_id,
                ROWS=rows,
                QUERY_LEN=query_len,
                QUERY_HEADS=query_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=kv_group_size,
                STATE_CAPACITY=state_capacity,
                ROUTE_COUNT=route_count,
                ROUTE_BLOCK=triton.next_power_of_2(route_count),
                HEAD_DIM=head_dim,
                HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
                BLOCK_M=query_prepare_block_m,
                num_warps=4,
            )
        elif compound_expert_routing:
            expert_capacity = batch * kv_heads * state_capacity
            bucket_count = tiny_expert_max + 1 + int(split_long_experts)
            if expert_capacity * bucket_count >= 2**31:
                raise ValueError("compound expert routing key exceeds INT32")
            expert_id = torch.empty(
                rows * route_count, dtype=torch.int32, device=q.device
            )
            query_prepare_block_m = 16
            _prepare_tiny_expert_sort_keys_kernel[
                (triton.cdiv(rows, query_prepare_block_m),)
            ](
                top_slots,
                slot_lengths,
                expert_id,
                rows,
                QUERY_LEN=query_len,
                QUERY_HEADS=query_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=kv_group_size,
                STATE_CAPACITY=state_capacity,
                EXPERT_CAPACITY=expert_capacity,
                ROUTE_COUNT=route_count,
                ROUTE_BLOCK=triton.next_power_of_2(route_count),
                TINY_EXPERT_MAX=tiny_expert_max,
                LONG_EXPERT_THRESHOLD=long_expert_threshold,
                SPLIT_LONG_EXPERTS=split_long_experts,
                BLOCK_M=query_prepare_block_m,
                num_warps=1,
            )
        else:
            query_head = torch.arange(query_heads, device=q.device, dtype=torch.int32)
            kv_head_for_query_head = torch.div(
                query_head, kv_group_size, rounding_mode="floor"
            )
            kv_row_for_head = torch.arange(
                batch, device=q.device, dtype=torch.int32
            ).unsqueeze(1) * kv_heads + kv_head_for_query_head.unsqueeze(0)
            expert_id = (
                kv_row_for_head[:, :, None, None] * state_capacity
                + top_slots.to(torch.int32)
            ).reshape(-1)
        record_dispatch_boundary()
        sorted_expert, order = expert_id.sort(stable=False)
        record_dispatch_boundary()
        unique_sort_key, q_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        if compound_expert_routing:
            expert_kv_row = torch.empty_like(unique_sort_key, dtype=torch.int32)
            expert_slot = torch.empty_like(unique_sort_key, dtype=torch.int32)
            expert_blocks = torch.empty_like(q_lengths)
            bucket_block_counts = torch.zeros(
                bucket_count,
                device=q.device,
                dtype=torch.int64,
            )
            metadata_block = 256
            _prepare_tiny_expert_metadata_kernel[
                (triton.cdiv(q_lengths.numel(), metadata_block),)
            ](
                unique_sort_key,
                q_lengths,
                expert_kv_row,
                expert_slot,
                expert_blocks,
                bucket_block_counts,
                q_lengths.numel(),
                EXPERT_CAPACITY=expert_capacity,
                STATE_CAPACITY=state_capacity,
                TINY_EXPERT_MAX=tiny_expert_max,
                BUCKET_COUNT=bucket_count,
                TINY_BLOCK_M=tiny_block_m,
                GENERAL_BLOCK_M=block_m,
                BLOCK=metadata_block,
                num_warps=4,
            )
        else:
            unique_expert = unique_sort_key
            expert_kv_row = torch.div(
                unique_expert, state_capacity, rounding_mode="floor"
            )
            expert_slot = unique_expert % state_capacity
        cu_q = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
        expert_index = torch.arange(
            q_lengths.numel(), device=q.device, dtype=torch.int32
        )
        record_dispatch_boundary()
        if compound_expert_routing:
            block_count_host = bucket_block_counts.cpu().tolist()
            tiny_bucket_blocks = tuple(
                int(count) for count in block_count_host[:tiny_expert_max]
            )
            tiny_total_blocks = sum(tiny_bucket_blocks)
            general_total_blocks = int(block_count_host[tiny_expert_max])
            long_total_blocks = (
                int(block_count_host[tiny_expert_max + 1])
                if split_long_experts
                else 0
            )
            total_blocks = (
                tiny_total_blocks + general_total_blocks + long_total_blocks
            )
            block_expert = torch.repeat_interleave(
                expert_index,
                expert_blocks,
                output_size=total_blocks,
            )
            block_starts = F.pad(expert_blocks.cumsum(0), (1, 0))[:-1].to(
                torch.int32
            )
        else:
            expert_blocks = torch.div(
                q_lengths + block_m - 1,
                block_m,
                rounding_mode="floor",
            )
            total_blocks = int(expert_blocks.sum().item())
            block_expert = torch.repeat_interleave(
                expert_index,
                expert_blocks,
                output_size=total_blocks,
            )
            block_starts = F.pad(expert_blocks.cumsum(0), (1, 0))[:-1].to(
                torch.int32
            )
        q_lengths = q_lengths.to(torch.int32)
        record_dispatch_boundary()

    record_boundary()

    route_out = torch.empty(
        rows * route_count,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    route_lse = torch.empty(rows * route_count, dtype=torch.float32, device=q.device)
    record_boundary()
    tiny_kernel_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    general_kernel_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    split_kernel_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    split_reduce_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
    if compound_expert_routing:
        if timing_events is not None and tiny_total_blocks:
            tiny_begin = torch.cuda.Event(enable_timing=True)
            tiny_end = torch.cuda.Event(enable_timing=True)
            tiny_begin.record()
        program_offset = 0
        assert isinstance(page_indices, torch.Tensor)
        for key_count, bucket_blocks in enumerate(tiny_bucket_blocks, start=1):
            if bucket_blocks:
                _tiny_leaf_expert_attention_kernel[(bucket_blocks,)](
                    kernel_q,
                    order,
                    block_expert,
                    block_starts,
                    q_lengths,
                    cu_q,
                    expert_kv_row,
                    expert_slot,
                    page_k,
                    page_v,
                    page_indices,
                    slot_pages,
                    route_out,
                    route_lse,
                    program_offset,
                    PAGE_CAPACITY=page_capacity,
                    LEAF_CAPACITY=int(page_k.size(2)),
                    STATE_CAPACITY=state_capacity,
                    INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                    HEAD_DIM=head_dim,
                    VALUE_DIM=value_dim,
                    PAGE_SIZE=page_size,
                    ROUTE_COUNT=route_count,
                    SCALE_LOG2=float(scale) * math.log2(math.e),
                    KEY_COUNT=key_count,
                    KEY_BLOCK=triton.next_power_of_2(key_count),
                    BLOCK_M=tiny_block_m,
                    num_warps=tiny_num_warps,
                    waves_per_eu=waves_per_eu,
                )
            program_offset += bucket_blocks
        if timing_events is not None and tiny_total_blocks:
            tiny_end.record()
            tiny_kernel_events = (tiny_begin, tiny_end)

        if timing_events is not None and general_total_blocks:
            general_begin = torch.cuda.Event(enable_timing=True)
            general_end = torch.cuda.Event(enable_timing=True)
            general_begin.record()
        if general_total_blocks:
            _paged_leaf_attention_kernel[(general_total_blocks,)](
                kernel_q,
                query_scales,
                order,
                block_expert,
                block_starts,
                page_k,
                page_v,
                page_indices,
                page_k,
                page_v,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                q_lengths,
                cu_q,
                expert_kv_row,
                expert_slot,
                route_out,
                route_lse,
                tiny_total_blocks,
                PAGE_CAPACITY=page_capacity,
                LEAF_CAPACITY=int(page_k.size(2)),
                STATE_CAPACITY=state_capacity,
                INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=hash_probes,
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                PAGE_SIZE=page_size,
                ROUTE_COUNT=route_count,
                SCALE_LOG2=float(scale) * math.log2(math.e),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                SPLIT_N=1,
                PARTIAL_OUTPUT=False,
                INT8_MMA=False,
                INT8_PV_MMA=False,
                INDEXED=True,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )
        if timing_events is not None and general_total_blocks:
            general_end.record()
            general_kernel_events = (general_begin, general_end)

        if long_total_blocks:
            if timing_events is not None:
                split_begin_event = torch.cuda.Event(enable_timing=True)
                split_end_event = torch.cuda.Event(enable_timing=True)
                split_begin_event.record()
            split_partial_out = torch.empty(
                long_total_blocks,
                long_expert_splits,
                block_m,
                value_dim,
                dtype=torch.float32,
                device=q.device,
            )
            split_partial_lse = torch.empty(
                long_total_blocks,
                long_expert_splits,
                block_m,
                dtype=torch.float32,
                device=q.device,
            )
            long_program_offset = tiny_total_blocks + general_total_blocks
            _paged_leaf_attention_kernel[
                (long_total_blocks * long_expert_splits,)
            ](
                kernel_q,
                query_scales,
                order,
                block_expert,
                block_starts,
                page_k,
                page_v,
                page_indices,
                page_k,
                page_v,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                q_lengths,
                cu_q,
                expert_kv_row,
                expert_slot,
                split_partial_out,
                split_partial_lse,
                long_program_offset,
                PAGE_CAPACITY=page_capacity,
                LEAF_CAPACITY=int(page_k.size(2)),
                STATE_CAPACITY=state_capacity,
                INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=hash_probes,
                HEAD_DIM=head_dim,
                VALUE_DIM=value_dim,
                PAGE_SIZE=page_size,
                ROUTE_COUNT=route_count,
                SCALE_LOG2=float(scale) * math.log2(math.e),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                SPLIT_N=long_expert_splits,
                PARTIAL_OUTPUT=True,
                INT8_MMA=False,
                INT8_PV_MMA=False,
                INDEXED=True,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )
            if timing_events is not None:
                split_end_event.record()
                split_kernel_events = (split_begin_event, split_end_event)
                split_reduce_begin = torch.cuda.Event(enable_timing=True)
                split_reduce_end = torch.cuda.Event(enable_timing=True)
                split_reduce_begin.record()
            _reduce_split_expert_attention_kernel[(long_total_blocks,)](
                order,
                block_expert,
                block_starts,
                q_lengths,
                cu_q,
                split_partial_out,
                split_partial_lse,
                route_out,
                route_lse,
                long_program_offset,
                VALUE_DIM=value_dim,
                ROUTE_COUNT=route_count,
                BLOCK_M=block_m,
                SPLIT_N=long_expert_splits,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )
            if timing_events is not None:
                split_reduce_end.record()
                split_reduce_events = (split_reduce_begin, split_reduce_end)
    else:
        _paged_leaf_attention_kernel[(total_blocks,)](
            kernel_q,
            query_scales,
            order,
            block_expert,
            block_starts,
            page_k,
            page_v,
            page_indices if indexed else page_k,
            page_k_scales if int8_mma else page_k,
            page_v_scales if int8_mma else page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            q_lengths,
            cu_q,
            expert_kv_row,
            expert_slot,
            route_out,
            route_lse,
            0,
            PAGE_CAPACITY=page_capacity,
            LEAF_CAPACITY=int(page_k.size(2)) if indexed else 1,
            STATE_CAPACITY=state_capacity,
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            HASH_CAPACITY=int(overflow_page_values.size(2)),
            HASH_PROBES=hash_probes,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            PAGE_SIZE=page_size,
            ROUTE_COUNT=route_count,
            SCALE_LOG2=float(scale) * math.log2(math.e),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            SPLIT_N=1,
            PARTIAL_OUTPUT=False,
            INT8_MMA=int8_mma,
            INT8_PV_MMA=int8_mma and int8_pv_mma,
            INDEXED=indexed,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )

    record_boundary()

    exact_out = torch.empty(rows, value_dim, dtype=q.dtype, device=q.device)
    exact_lse = torch.empty(rows, dtype=torch.float32, device=q.device)
    _reduce_expert_route_attention_kernel[(rows,)](
        route_out,
        route_lse,
        exact_out,
        exact_lse,
        ROUTE_COUNT=route_count,
        ROUTE_BLOCK=triton.next_power_of_2(route_count),
        VALUE_DIM=value_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        num_warps=reduce_num_warps,
    )
    record_boundary()
    if timing_events is not None:
        for name, begin, end in zip(
            (
                "dispatch_prepare",
                "dispatch_sort",
                "dispatch_group",
                "dispatch_blocks",
            ),
            dispatch_boundaries[:-1],
            dispatch_boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        if tiny_kernel_events is not None:
            timing_events.setdefault("tiny_kernel", []).append(tiny_kernel_events)
        if general_kernel_events is not None:
            timing_events.setdefault("general_kernel", []).append(
                general_kernel_events
            )
        if split_kernel_events is not None:
            timing_events.setdefault("split_n_kernel", []).append(
                split_kernel_events
            )
        if split_reduce_events is not None:
            timing_events.setdefault("split_n_reduce", []).append(
                split_reduce_events
            )
        for name, begin, end in zip(
            ("dispatch", "pack", "kernel", "reduce"),
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        timing_events.setdefault("total", []).append((boundaries[0], boundaries[-1]))
    return (
        exact_out.reshape(batch, query_heads, query_len, value_dim),
        exact_lse.reshape(batch, query_heads, query_len),
    )


def aiter_bucketed_paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    waves_per_eu: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run centroid experts through AITER's paged-KV attention kernel.

    AITER's MI300 varlen paged-KV FMHA currently requires 128-token cache
    blocks, while LOD deliberately uses 16-token pages.  Its dense-query
    paged-KV kernel supports those pages, so active experts are divided into
    power-of-two query-length buckets.  K/V stay in the persistent page pool;
    only routed Q rows are packed and padded within a bucket.
    """
    del overflow_page_keys, overflow_page_values, overflow_used
    del hash_probes, block_m, block_n, num_warps, waves_per_eu
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("AITER bucketed leaf attention is forward-only")
    try:
        from aiter.ops.triton.attention.mha import flash_attn_with_kvcache
    except ImportError as error:
        raise RuntimeError(
            "AITER bucketed leaf attention requires an AITER installation"
        ) from error

    batch, query_heads, query_len, head_dim = q.shape
    route_count = int(top_slots.size(-1))
    kv_heads = int(page_k.size(1))
    value_dim = int(page_v.size(-1))
    page_capacity = int(page_k.size(2))
    page_size = int(page_k.size(3))
    state_capacity = int(slot_pages.size(2))
    inline_pages = int(slot_pages.size(3))
    if page_size != 16:
        raise ValueError("AITER bucketed leaf attention requires 16-token pages")
    if head_dim != value_dim:
        raise ValueError("AITER bucketed leaf attention requires equal QK/V dimensions")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")

    boundaries: list[torch.cuda.Event] = []

    def record_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            boundaries.append(event)

    record_boundary()
    with torch.no_grad():
        rows = batch * query_heads * query_len
        query_head = torch.arange(query_heads, device=q.device, dtype=torch.long)
        kv_head_for_query_head = torch.div(
            query_head, kv_group_size, rounding_mode="floor"
        )
        kv_row_for_head = torch.arange(
            batch, device=q.device, dtype=torch.long
        ).unsqueeze(1) * kv_heads + kv_head_for_query_head.unsqueeze(0)
        expert_id = (
            kv_row_for_head[:, :, None, None] * state_capacity + top_slots
        ).reshape(-1)
        sorted_expert, order = expert_id.sort(stable=False)
        unique_expert, q_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        expert_kv_row = torch.div(unique_expert, state_capacity, rounding_mode="floor")
        expert_slot = unique_expert % state_capacity
        k_lengths = slot_lengths.reshape(-1).index_select(0, unique_expert)
        if bool((k_lengths <= 0).any().item()):
            raise AssertionError("a routed state expert owns no leaves")
        max_q = int(q_lengths.max().item())
        max_k = int(k_lengths.max().item())
        max_pages = (max_k + page_size - 1) // page_size
        if max_pages > inline_pages:
            raise RuntimeError(
                "AITER bucketed leaf attention does not yet materialize overflow "
                "page-table entries"
            )
        cu_q = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int64)
        local_block_table = (
            slot_pages.reshape(-1, inline_pages)
            .index_select(0, unique_expert)[:, :max_pages]
            .to(torch.int32)
        )
        valid_page = local_block_table >= 0
        physical_page_base = (expert_kv_row * page_capacity).to(torch.int32)
        block_table = torch.where(
            valid_page,
            local_block_table + physical_page_base[:, None],
            -1,
        ).contiguous()
        q_flat = q.reshape(rows, head_dim)
        page_k_flat = page_k.reshape(-1, page_size, head_dim).unsqueeze(2)
        page_v_flat = page_v.reshape(-1, page_size, value_dim).unsqueeze(2)

    record_boundary()
    route_out = torch.empty(
        rows * route_count, value_dim, dtype=q.dtype, device=q.device
    )
    route_lse = torch.empty(rows * route_count, dtype=torch.float32, device=q.device)

    bucket_sizes: list[int] = []
    bucket_size = 16
    while bucket_size < max_q:
        bucket_sizes.append(bucket_size)
        bucket_size *= 2
    bucket_sizes.append(bucket_size)
    previous_size = 0
    packed_buckets: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    for bucket_size in bucket_sizes:
        expert_mask = (q_lengths > previous_size) & (q_lengths <= bucket_size)
        bucket_expert = torch.nonzero(expert_mask, as_tuple=False).flatten()
        previous_size = bucket_size
        if int(bucket_expert.numel()) == 0:
            continue
        bucket_lengths = q_lengths.index_select(0, bucket_expert)
        offset = torch.arange(bucket_size, device=q.device, dtype=torch.int64)
        valid_query = offset[None, :] < bucket_lengths[:, None]
        packed_row = cu_q.index_select(0, bucket_expert)[:, None] + offset[None, :]
        packed_row = packed_row.clamp_max(int(order.numel()) - 1)
        route_row = order.index_select(0, packed_row.reshape(-1)).reshape_as(packed_row)
        query_row = torch.div(route_row, route_count, rounding_mode="floor")
        bucket_q = q_flat.index_select(0, query_row.reshape(-1)).reshape(
            int(bucket_expert.numel()), bucket_size, 1, head_dim
        )
        bucket_q = bucket_q.masked_fill(~valid_query[:, :, None, None], 0).contiguous()
        packed_buckets.append(
            (
                bucket_q,
                route_row,
                valid_query,
                k_lengths.index_select(0, bucket_expert).to(torch.int32),
                block_table.index_select(0, bucket_expert),
            )
        )

    record_boundary()
    for (
        bucket_q,
        route_row,
        valid_query,
        bucket_k_lengths,
        bucket_table,
    ) in packed_buckets:
        bucket_out, bucket_lse = flash_attn_with_kvcache(
            bucket_q,
            page_k_flat,
            page_v_flat,
            cache_seqlens=bucket_k_lengths,
            softmax_scale=float(scale),
            causal=False,
            block_table=bucket_table,
            return_softmax_lse=True,
        )
        destination = route_row[valid_query]
        route_out.index_copy_(0, destination, bucket_out[:, :, 0][valid_query])
        route_lse.index_copy_(
            0, destination, bucket_lse.reshape_as(valid_query)[valid_query]
        )

    record_boundary()
    route_out = route_out.reshape(rows, route_count, value_dim)
    route_lse = route_lse.reshape(rows, route_count)
    route_weight = torch.softmax(route_lse, dim=-1).to(route_out.dtype)
    exact_out = (route_out * route_weight.unsqueeze(-1)).sum(dim=1)
    exact_lse = torch.logsumexp(route_lse, dim=-1)
    record_boundary()
    if timing_events is not None:
        for name, begin, end in zip(
            ("dispatch", "pack", "kernel", "reduce"),
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        timing_events.setdefault("total", []).append((boundaries[0], boundaries[-1]))
    return (
        exact_out.reshape(batch, query_heads, query_len, value_dim),
        exact_lse.reshape(batch, query_heads, query_len),
    )


@triton.jit(
    do_not_specialize=["max_k"],
    do_not_specialize_on_alignment=["max_k"],
)
def _materialize_aiter_indexed_expert_table_kernel(
    unique_expert,
    k_lengths,
    cu_k,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    page_indices,
    kv_page_indices,
    max_k,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Expose virtual chronological leaves as AITER page-size-one blocks."""
    expert = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
    expert_id = tl.load(unique_expert + expert).to(tl.int64)
    kv_row = expert_id // STATE_CAPACITY
    slot = expert_id - kv_row * STATE_CAPACITY
    length = tl.load(k_lengths + expert).to(tl.int64)
    valid = (token < length) & (token < max_k)
    page_ordinal = token // PAGE_SIZE
    within_page = token - page_ordinal * PAGE_SIZE
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
    leaf = tl.load(
        page_indices
        + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
        + within_page,
        mask=page_valid,
        other=-1,
    ).to(tl.int64)
    leaf_valid = page_valid & (leaf >= 0) & (leaf < LEAF_CAPACITY)
    destination = tl.load(cu_k + expert).to(tl.int64) + token
    tl.store(
        kv_page_indices + destination,
        kv_row * LEAF_CAPACITY + leaf,
        mask=leaf_valid,
    )


@triton.jit(
    do_not_specialize=["max_k"],
    do_not_specialize_on_alignment=["max_k"],
)
def _copy_aiter_indexed_expert_kv_kernel(
    source_k,
    source_v,
    source_leaf_indices,
    source_indptr,
    destination_page_indptr,
    k_lengths,
    destination_k,
    destination_v,
    max_k,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    COPY_PAGE_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    """Materialize one ideal contiguous K/V posting list per active expert.

    This is a diagnostic layout adapter, not part of the proposed runtime path:
    its cost is deliberately kept outside the timed AITER attention interval.
    """
    expert = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1).to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
    head_dimension = tl.arange(0, HEAD_BLOCK_DIM)
    value_dimension = tl.arange(0, VALUE_BLOCK_DIM)
    length = tl.load(k_lengths + expert).to(tl.int64)
    valid_token = (token < length) & (token < max_k)
    source_token = tl.load(source_indptr + expert).to(tl.int64) + token
    source_leaf = tl.load(
        source_leaf_indices + source_token,
        mask=valid_token,
        other=0,
    ).to(tl.int64)
    destination_page = (
        tl.load(destination_page_indptr + expert).to(tl.int64)
        + token // COPY_PAGE_SIZE
    )
    destination_token = destination_page * COPY_PAGE_SIZE + token % COPY_PAGE_SIZE
    key = tl.load(
        source_k + source_leaf[:, None] * HEAD_DIM + head_dimension[None, :],
        mask=valid_token[:, None] & (head_dimension[None, :] < HEAD_DIM),
        other=0.0,
    )
    value = tl.load(
        source_v + source_leaf[:, None] * VALUE_DIM + value_dimension[None, :],
        mask=valid_token[:, None] & (value_dimension[None, :] < VALUE_DIM),
        other=0.0,
    )
    tl.store(
        destination_k
        + destination_token[:, None] * HEAD_DIM
        + head_dimension[None, :],
        key,
        mask=valid_token[:, None] & (head_dimension[None, :] < HEAD_DIM),
    )
    tl.store(
        destination_v
        + destination_token[:, None] * VALUE_DIM
        + value_dimension[None, :],
        value,
        mask=valid_token[:, None] & (value_dimension[None, :] < VALUE_DIM),
    )


@triton.jit(
    do_not_specialize=["max_slot_length"],
    do_not_specialize_on_alignment=["max_slot_length"],
)
def _materialize_aiter_indexed_union_table_kernel(
    union_experts,
    union_lengths,
    union_prefix,
    kv_indptr,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    page_indices,
    kv_page_indices,
    max_slot_length,
    UNION_WIDTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Concatenate every centroid in one query-tile union for AITER."""
    sequence = tl.program_id(0).to(tl.int64)
    union_rank = tl.program_id(1).to(tl.int64)
    token = tl.program_id(2).to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
    union_offset = sequence * UNION_WIDTH + union_rank
    expert_id = tl.load(union_experts + union_offset).to(tl.int64)
    kv_row = expert_id // STATE_CAPACITY
    slot = expert_id - kv_row * STATE_CAPACITY
    length = tl.load(union_lengths + union_offset).to(tl.int64)
    valid = (token < length) & (token < max_slot_length)
    page_ordinal = token // PAGE_SIZE
    within_page = token - page_ordinal * PAGE_SIZE
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
    leaf = tl.load(
        page_indices
        + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
        + within_page,
        mask=page_valid,
        other=-1,
    ).to(tl.int64)
    leaf_valid = page_valid & (leaf >= 0) & (leaf < LEAF_CAPACITY)
    destination = (
        tl.load(kv_indptr + sequence).to(tl.int64)
        + tl.load(union_prefix + union_offset).to(tl.int64)
        + token
    )
    tl.store(
        kv_page_indices + destination,
        kv_row * LEAF_CAPACITY + leaf,
        mask=leaf_valid,
    )


@triton.jit(
    do_not_specialize=["query_len", "max_slot_length"],
    do_not_specialize_on_alignment=["query_len", "max_slot_length"],
)
def _materialize_aiter_indexed_masked_union_table_kernel(
    union_experts,
    union_lengths,
    union_prefix,
    original_slots,
    kv_indptr,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    page_indices,
    kv_page_indices,
    kv_query_masks,
    query_len,
    max_slot_length,
    UNION_WIDTH: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    TILES_PER_HEAD: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_WIDTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Materialize a tile union and each token's exact 16-query mask."""
    sequence = tl.program_id(0).to(tl.int64)
    union_rank = tl.program_id(1).to(tl.int64)
    token = tl.program_id(2).to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
    union_offset = sequence * UNION_WIDTH + union_rank
    expert_id = tl.load(union_experts + union_offset).to(tl.int64)
    kv_row = expert_id // STATE_CAPACITY
    slot = expert_id - kv_row * STATE_CAPACITY
    length = tl.load(union_lengths + union_offset).to(tl.int64)
    valid = (token < length) & (token < max_slot_length)

    sequences_per_batch = QUERY_HEADS * TILES_PER_HEAD
    batch = sequence // sequences_per_batch
    within_batch = sequence - batch * sequences_per_batch
    query_head = within_batch // TILES_PER_HEAD
    query_tile = within_batch - query_head * TILES_PER_HEAD
    query_offset = tl.arange(0, QUERY_TILE)
    route_rank = tl.arange(0, ROUTE_WIDTH)
    query_position = query_tile * QUERY_TILE + query_offset
    query_valid = query_position < query_len
    query_row = (batch * QUERY_HEADS + query_head) * query_len + query_position
    selected_slots = tl.load(
        original_slots
        + query_row[:, None] * ROUTE_COUNT
        + route_rank[None, :],
        mask=query_valid[:, None] & (route_rank[None, :] < ROUTE_COUNT),
        other=-1,
    ).to(tl.int64)
    query_selected = tl.sum((selected_slots == slot).to(tl.int32), axis=1) > 0
    query_bits = tl.full((QUERY_TILE,), 1, tl.int32) << query_offset
    query_membership = tl.sum(
        tl.where(query_valid & query_selected, query_bits, 0), axis=0
    ).to(tl.int32)

    page_ordinal = token // PAGE_SIZE
    within_page = token - page_ordinal * PAGE_SIZE
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
    leaf = tl.load(
        page_indices
        + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
        + within_page,
        mask=page_valid,
        other=-1,
    ).to(tl.int64)
    leaf_valid = page_valid & (leaf >= 0) & (leaf < LEAF_CAPACITY)
    destination = (
        tl.load(kv_indptr + sequence).to(tl.int64)
        + tl.load(union_prefix + union_offset).to(tl.int64)
        + token
    )
    tl.store(
        kv_page_indices + destination,
        kv_row * LEAF_CAPACITY + leaf,
        mask=leaf_valid,
    )
    tl.store(
        kv_query_masks + destination,
        query_membership,
        mask=leaf_valid,
    )


def query_tile_slot_unions(
    top_slots: torch.Tensor,
    slot_lengths: torch.Tensor,
    *,
    kv_group_size: int,
    query_tile: int,
) -> torch.Tensor:
    """Share each query-head's route union across one contiguous query tile."""
    if query_tile <= 0 or query_tile & (query_tile - 1):
        raise ValueError("AITER leaf union query tile must be a positive power of two")
    batch, query_heads, query_len, route_count = top_slots.shape
    cache_batch, kv_heads, state_capacity = slot_lengths.shape
    if cache_batch != batch or query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER leaf union query/KV geometry is inconsistent")
    tile_count = triton.cdiv(query_len, query_tile)
    padded_query_len = tile_count * query_tile
    padded = top_slots
    if padded_query_len != query_len:
        padded = F.pad(padded, (0, 0, 0, padded_query_len - query_len), value=-1)
    candidates = padded.view(
        batch,
        query_heads,
        tile_count,
        query_tile * route_count,
    ).reshape(batch * query_heads * tile_count, query_tile * route_count)
    sequences = int(candidates.size(0))
    sequence = torch.arange(sequences, device=top_slots.device)
    sequence_batch = torch.div(
        sequence, query_heads * tile_count, rounding_mode="floor"
    )
    sequence_query_head = torch.div(
        sequence % (query_heads * tile_count), tile_count, rounding_mode="floor"
    )
    sequence_kv_head = torch.div(
        sequence_query_head, kv_group_size, rounding_mode="floor"
    )
    sequence_lengths = slot_lengths[sequence_batch, sequence_kv_head]
    candidate_valid = (candidates >= 0) & (candidates < state_capacity)
    safe_candidate = candidates.clamp(min=0, max=max(state_capacity - 1, 0)).long()
    candidate_valid &= torch.gather(sequence_lengths, 1, safe_candidate) > 0
    sortable = torch.where(
        candidate_valid,
        candidates,
        torch.full_like(candidates, state_capacity),
    )
    sorted_slots = sortable.sort(dim=-1).values
    unique = sorted_slots < state_capacity
    unique[:, 1:] &= sorted_slots[:, 1:] != sorted_slots[:, :-1]
    # Keep a fixed-width sparse union instead of synchronizing to discover a
    # batch-dependent maximum and scattering into a second compact buffer.
    # Downstream metadata construction treats every -1 lane as zero length.
    union_slots = torch.where(
        unique,
        sorted_slots,
        torch.full_like(sorted_slots, -1),
    )
    union_width = int(union_slots.size(-1))
    return (
        union_slots.view(batch, query_heads, tile_count, union_width)
        .unsqueeze(3)
        .expand(-1, -1, -1, query_tile, -1)
        .reshape(batch, query_heads, padded_query_len, union_width)
        .narrow(2, 0, query_len)
        .contiguous()
    )


@triton.jit(
    do_not_specialize=["query_len", "union_width"],
    do_not_specialize_on_alignment=["query_len", "union_width"],
)
def _remove_query_tile_union_from_coarse_kernel(
    q,
    state_k,
    state_v,
    counts,
    original_slots,
    union_slots,
    full_out,
    full_lse,
    residual_out,
    residual_lse,
    query_len,
    union_width,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    TILES_PER_HEAD: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    ORIGINAL_COUNT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    """Remove union-only centroids from an already top-k-pruned coarse result."""
    sequence = tl.program_id(0).to(tl.int64)
    sequences_per_batch = QUERY_HEADS * TILES_PER_HEAD
    batch = sequence // sequences_per_batch
    within_batch = sequence - batch * sequences_per_batch
    query_head = within_batch // TILES_PER_HEAD
    query_tile = within_batch - query_head * TILES_PER_HEAD
    query_begin = query_tile * QUERY_TILE
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = batch * KV_HEADS + kv_head

    row = tl.arange(0, QUERY_TILE)
    query_offset = query_begin + row
    query_valid = query_offset < query_len
    query_row = (batch * QUERY_HEADS + query_head) * query_len + query_offset
    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    queries = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=query_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
        other=0.0,
    )
    total_output = tl.load(
        full_out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    total_lse = tl.load(full_lse + query_row, mask=query_valid, other=0.0).to(
        tl.float32
    )
    total_lse_log2 = total_lse * 1.4426950408889634
    selected_mass = tl.zeros((QUERY_TILE,), tl.float32)
    selected_numerator = tl.zeros((QUERY_TILE, VALUE_BLOCK_DIM), tl.float32)
    union_row = (batch * QUERY_HEADS + query_head) * query_len + query_begin
    union_offset = tl.arange(0, BLOCK_N)
    for union_begin in tl.range(0, union_width, BLOCK_N, num_stages=1):
        rank = union_begin + union_offset
        slot = tl.load(
            union_slots + union_row * union_width + rank,
            mask=rank < union_width,
            other=-1,
        ).to(tl.int64)
        slot_valid = (rank < union_width) & (slot >= 0) & (slot < STATE_CAPACITY)
        safe_slot = tl.where(slot_valid, slot, 0)
        count = tl.load(
            counts + (kv_row * STATE_CAPACITY + safe_slot),
            mask=slot_valid,
            other=0.0,
        ).to(tl.float32)
        slot_valid &= count > 0.0
        safe_count = tl.maximum(count, 1.0)
        already_open = tl.zeros((QUERY_TILE, BLOCK_N), tl.int1)
        for original_rank in tl.static_range(0, ORIGINAL_COUNT):
            original = tl.load(
                original_slots + query_row * ORIGINAL_COUNT + original_rank,
                mask=query_valid,
                other=-1,
            ).to(tl.int64)
            already_open |= original[:, None] == safe_slot[None, :]
        extra = query_valid[:, None] & slot_valid[None, :] & ~already_open
        key_sums = tl.load(
            state_k
            + (kv_row * STATE_CAPACITY + safe_slot[:, None]) * HEAD_DIM
            + head_offset[None, :],
            mask=slot_valid[:, None] & (head_offset[None, :] < HEAD_DIM),
            other=0.0,
        )
        value_sums = tl.load(
            state_v
            + (kv_row * STATE_CAPACITY + safe_slot[:, None]) * VALUE_DIM
            + value_offset[None, :],
            mask=slot_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
            other=0.0,
        )
        mean_keys = (key_sums.to(tl.float32) / safe_count[:, None]).to(queries.dtype)
        mean_values = (value_sums.to(tl.float32) / safe_count[:, None]).to(
            value_sums.dtype
        )
        scores = (
            SCALE_LOG2 * tl.dot(queries, tl.trans(mean_keys), out_dtype=tl.float32)
            + tl.log2(safe_count)[None, :]
        )
        weights = tl.where(
            extra,
            tl.math.exp2(scores - total_lse_log2[:, None]),
            0.0,
        )
        selected_mass += tl.sum(weights, axis=1)
        selected_numerator += tl.dot(
            weights.to(mean_values.dtype), mean_values, out_dtype=tl.float32
        )

    remaining_mass = 1.0 - selected_mass
    has_mass = query_valid & (remaining_mass > 1.0e-7)
    residual = (total_output - selected_numerator) / tl.maximum(
        remaining_mass[:, None], 1.0e-7
    )
    tl.store(
        residual_out + query_row[:, None] * VALUE_DIM + value_offset[None, :],
        tl.where(has_mass[:, None], residual, 0.0),
        mask=query_valid[:, None] & (value_offset[None, :] < VALUE_DIM),
    )
    tl.store(
        residual_lse + query_row,
        tl.where(has_mass, total_lse + tl.log(remaining_mass), -float("inf")),
        mask=query_valid,
    )


def remove_query_tile_union_from_coarse(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    original_slots: torch.Tensor,
    union_slots: torch.Tensor,
    coarse_out: torch.Tensor,
    coarse_lse: torch.Tensor,
    *,
    kv_group_size: int,
    query_tile: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extend fused per-query top-k removal to a shared query-tile union."""
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(state_k.size(1))
    state_capacity = int(state_k.size(2))
    value_dim = int(state_v.size(-1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("coarse union correction has inconsistent GQA geometry")
    if tuple(original_slots.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("original coarse routes have the wrong shape")
    if tuple(union_slots.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("union coarse routes have the wrong shape")
    if query_tile <= 0 or query_tile & (query_tile - 1):
        raise ValueError("coarse union query tile must be a positive power of two")
    tile_count = triton.cdiv(query_len, query_tile)
    union_width = int(union_slots.size(-1))
    block_n = min(64, triton.next_power_of_2(union_width))
    _remove_query_tile_union_from_coarse_kernel[
        (batch * query_heads * tile_count,)
    ](
        q.contiguous(),
        state_k.contiguous(),
        state_v.contiguous(),
        counts.contiguous(),
        original_slots.contiguous(),
        union_slots.contiguous(),
        coarse_out,
        coarse_lse,
        coarse_out,
        coarse_lse,
        query_len,
        union_width,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        QUERY_TILE=query_tile,
        TILES_PER_HEAD=tile_count,
        STATE_CAPACITY=state_capacity,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        ORIGINAL_COUNT=int(original_slots.size(-1)),
        BLOCK_N=block_n,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        num_warps=4,
    )
    return coarse_out, coarse_lse


def aiter_query_tile_union_paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    page_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    query_tile: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    waves_per_eu: int = 1,
    mask_queries: bool = False,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend each query tile to its union with one large-tile AITER sequence.

    In legacy unmasked mode, ``top_slots`` already contains the same union for
    every query. In masked mode it contains the original per-query routes; the
    function builds the union and supplies AITER one exact 16-bit membership
    mask per indexed token. The latter keeps the coarse branch per-query too.
    """
    del block_m, block_n, num_warps, waves_per_eu
    if page_indices is None:
        raise ValueError("AITER query-tile unions require virtual indexed pages")
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("AITER query-tile leaf attention is forward-only")
    try:
        from aiter.ops.mha import mha_batch_prefill_func
    except ImportError as error:
        raise RuntimeError(
            "AITER query-tile leaf attention requires an AITER installation"
        ) from error

    batch, query_heads, query_len, head_dim = q.shape
    cache_batch, kv_heads, leaf_capacity, key_dim = page_k.shape
    value_dim = int(page_v.size(-1))
    state_capacity = int(slot_pages.size(2))
    inline_pages = int(slot_pages.size(3))
    page_capacity = int(page_indices.size(2))
    page_size = int(page_indices.size(3))
    if cache_batch != batch or query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER query-tile leaf geometry is inconsistent")
    if key_dim != head_dim or value_dim != head_dim:
        raise ValueError("AITER query-tile leaves require equal Q/K/V dimensions")
    if page_size != 16:
        raise ValueError("AITER query-tile leaves require 16-token logical pages")
    if query_tile <= 0 or query_tile & (query_tile - 1):
        raise ValueError("AITER query tile must be a positive power of two")
    if mask_queries and query_tile > 16:
        raise ValueError("masked AITER query unions support at most 16 queries")
    tile_count = triton.cdiv(query_len, query_tile)
    original_slots = top_slots
    if mask_queries:
        top_slots = query_tile_slot_unions(
            original_slots,
            slot_lengths,
            kv_group_size=kv_group_size,
            query_tile=query_tile,
        )
    union_width = int(top_slots.size(-1))
    if tuple(top_slots.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("AITER query-tile union routes have the wrong shape")

    boundaries: list[torch.cuda.Event] = []

    def record_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            boundaries.append(event)

    record_boundary()
    with torch.no_grad():
        union_slots = top_slots[:, :, ::query_tile, :].reshape(
            batch * query_heads * tile_count, union_width
        )
        sequences = int(union_slots.size(0))
        sequence = torch.arange(sequences, device=q.device)
        sequence_batch = torch.div(
            sequence, query_heads * tile_count, rounding_mode="floor"
        )
        sequence_query_head = torch.div(
            sequence % (query_heads * tile_count),
            tile_count,
            rounding_mode="floor",
        )
        sequence_kv_head = torch.div(
            sequence_query_head, kv_group_size, rounding_mode="floor"
        )
        kv_row = sequence_batch * kv_heads + sequence_kv_head
        safe_slot = union_slots.clamp(min=0, max=state_capacity - 1).long()
        union_lengths = torch.gather(
            slot_lengths[sequence_batch, sequence_kv_head], 1, safe_slot
        ).to(torch.int32)
        union_lengths = torch.where(
            union_slots >= 0, union_lengths, torch.zeros_like(union_lengths)
        )
        union_prefix = union_lengths.cumsum(dim=-1, dtype=torch.int32) - union_lengths
        token_counts = union_lengths.sum(dim=-1, dtype=torch.int32)
        if bool((token_counts <= 0).any().item()):
            raise AssertionError("an AITER query-tile union owns no leaves")
        max_slot_length = max(1, int(union_lengths.max().item()))
        max_k = max(1, int(token_counts.max().item()))
        kv_indptr = F.pad(token_counts.cumsum(0), (1, 0)).to(torch.int32)
        kv_page_indices = torch.empty(
            int(kv_indptr[-1].item()), dtype=torch.int32, device=q.device
        )
        union_experts = (kv_row[:, None] * state_capacity + safe_slot).to(torch.int32)
        table_block_k = 128
        if mask_queries:
            kv_query_masks = torch.empty_like(kv_page_indices)
            _materialize_aiter_indexed_masked_union_table_kernel[
                (sequences, union_width, triton.cdiv(max_slot_length, table_block_k))
            ](
                union_experts,
                union_lengths,
                union_prefix,
                original_slots.contiguous(),
                kv_indptr,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                page_indices,
                kv_page_indices,
                kv_query_masks,
                query_len,
                max_slot_length,
                UNION_WIDTH=union_width,
                QUERY_HEADS=query_heads,
                TILES_PER_HEAD=tile_count,
                QUERY_TILE=query_tile,
                ROUTE_COUNT=int(original_slots.size(-1)),
                ROUTE_WIDTH=triton.next_power_of_2(int(original_slots.size(-1))),
                STATE_CAPACITY=state_capacity,
                INLINE_PAGES_PER_SLOT=inline_pages,
                PAGE_CAPACITY=page_capacity,
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=(-1 if overflow_page_values.ndim == 4 else hash_probes),
                PAGE_SIZE=page_size,
                LEAF_CAPACITY=leaf_capacity,
                BLOCK_K=table_block_k,
                num_warps=1,
            )
        else:
            kv_query_masks = None
            _materialize_aiter_indexed_union_table_kernel[
                (sequences, union_width, triton.cdiv(max_slot_length, table_block_k))
            ](
                union_experts,
                union_lengths,
                union_prefix,
                kv_indptr,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                page_indices,
                kv_page_indices,
                max_slot_length,
                UNION_WIDTH=union_width,
                STATE_CAPACITY=state_capacity,
                INLINE_PAGES_PER_SLOT=inline_pages,
                PAGE_CAPACITY=page_capacity,
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=(-1 if overflow_page_values.ndim == 4 else hash_probes),
                PAGE_SIZE=page_size,
                LEAF_CAPACITY=leaf_capacity,
                BLOCK_K=table_block_k,
                num_warps=1,
            )
        query_lengths = torch.full(
            (tile_count,), query_tile, dtype=torch.int32, device=q.device
        )
        query_lengths[-1] = query_len - (tile_count - 1) * query_tile
        query_lengths = query_lengths.repeat(batch * query_heads)
        qo_indptr = F.pad(query_lengths.cumsum(0), (1, 0)).to(torch.int32)
        packed_q = q.reshape(batch * query_heads * query_len, 1, head_dim)
        token_k = page_k.reshape(
            cache_batch * kv_heads * leaf_capacity, 1, head_dim
        ).unsqueeze(2)
        token_v = page_v.reshape(
            cache_batch * kv_heads * leaf_capacity, 1, value_dim
        ).unsqueeze(2)
        last_page_lens = torch.ones(sequences, dtype=torch.int32, device=q.device)

    record_boundary()
    aiter_metadata = (
        {
            "block_table": torch.empty((1, 4), dtype=torch.int32, device=q.device),
            "seqlen_k": kv_query_masks,
        }
        if mask_queries
        else {
            "kv_last_page_lens": last_page_lens,
            "seqlen_k": token_counts,
        }
    )
    packed_out, packed_lse = mha_batch_prefill_func(
        packed_q,
        token_k,
        token_v,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        query_tile,
        max_k,
        softmax_scale=float(scale),
        causal=False,
        return_lse=True,
        **aiter_metadata,
    )
    record_boundary()
    exact = packed_out[:, 0].reshape(batch, query_heads, query_len, value_dim)
    exact_lse = packed_lse.reshape(batch, query_heads, query_len)
    record_boundary()
    if timing_events is not None:
        for name, begin, end in zip(
            ("union_table", "union_aiter", "union_unpack"),
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        timing_events.setdefault("total", []).append((boundaries[0], boundaries[-1]))
    return exact, exact_lse


def aiter_varlen_paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    page_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    waves_per_eu: int = 1,
    copy_indexed_kv: bool = False,
    copy_page_size: int = 16,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ragged centroid experts with AITER batch paged prefill.

    Virtual-page caches normally use AITER's page-size-one linear layout.  The
    diagnostic ``copy_indexed_kv`` mode instead materializes every active
    expert once into ordinary pages before the measured AITER call.  It gives
    an upper bound for query-indexed AITER without charging the simulated K/V
    layout conversion.
    """
    del block_m, block_n, num_warps, waves_per_eu
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("AITER varlen leaf attention is forward-only")
    try:
        from aiter.ops.mha import mha_batch_prefill_func
    except ImportError as error:
        raise RuntimeError(
            "AITER varlen leaf attention requires an AITER installation"
        ) from error

    batch, query_heads, query_len, head_dim = q.shape
    route_count = int(top_slots.size(-1))
    kv_heads = int(page_k.size(1))
    value_dim = int(page_v.size(-1))
    indexed = page_indices is not None
    page_capacity = int(page_indices.size(2)) if indexed else int(page_k.size(2))
    page_size = int(page_indices.size(3)) if indexed else int(page_k.size(3))
    leaf_capacity = int(page_k.size(2)) if indexed else 0
    state_capacity = int(slot_pages.size(2))
    inline_pages = int(slot_pages.size(3))
    if page_size != 16:
        raise ValueError("AITER varlen leaf attention requires 16-token pages")
    if head_dim != value_dim:
        raise ValueError("AITER varlen leaf attention requires equal QK/V dimensions")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if indexed and (page_k.ndim != 4 or page_v.ndim != 4):
        raise ValueError("indexed AITER leaves require chronological rank-four K/V")
    if indexed and (page_indices.dtype != torch.int32 or page_indices.ndim != 4):
        raise TypeError("indexed AITER page metadata must be rank-four int32")
    if copy_indexed_kv and not indexed:
        raise ValueError("copied AITER experts require virtual indexed leaves")
    if copy_page_size <= 0 or copy_page_size & (copy_page_size - 1):
        raise ValueError("copied AITER page size must be a positive power of two")

    boundaries: list[torch.cuda.Event] = []

    def record_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            boundaries.append(event)

    record_boundary()
    with torch.no_grad():
        rows = batch * query_heads * query_len
        query_head = torch.arange(query_heads, device=q.device, dtype=torch.long)
        kv_head_for_query_head = torch.div(
            query_head, kv_group_size, rounding_mode="floor"
        )
        kv_row_for_head = torch.arange(
            batch, device=q.device, dtype=torch.long
        ).unsqueeze(1) * kv_heads + kv_head_for_query_head.unsqueeze(0)
        expert_id = (
            kv_row_for_head[:, :, None, None] * state_capacity + top_slots
        ).reshape(-1)
        sorted_expert, order = expert_id.sort(stable=False)
        unique_expert, q_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        expert_kv_row = torch.div(unique_expert, state_capacity, rounding_mode="floor")
        k_lengths = slot_lengths.reshape(-1).index_select(0, unique_expert)
        if bool((k_lengths <= 0).any().item()):
            raise AssertionError("a routed state expert owns no leaves")
        max_q = int(q_lengths.max().item())
        max_k = int(k_lengths.max().item())
        cu_q = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
        if indexed:
            source_indptr = F.pad(k_lengths.cumsum(0), (1, 0)).to(torch.int32)
            source_leaf_indices = torch.full(
                (int(source_indptr[-1].item()),),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
            table_block_k = 128
            _materialize_aiter_indexed_expert_table_kernel[
                (
                    int(unique_expert.numel()),
                    triton.cdiv(max_k, table_block_k),
                )
            ](
                unique_expert.to(torch.int32),
                k_lengths.to(torch.int32),
                source_indptr,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                page_indices,
                source_leaf_indices,
                max_k,
                STATE_CAPACITY=state_capacity,
                INLINE_PAGES_PER_SLOT=inline_pages,
                PAGE_CAPACITY=page_capacity,
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=(-1 if overflow_page_values.ndim == 4 else hash_probes),
                PAGE_SIZE=page_size,
                LEAF_CAPACITY=leaf_capacity,
                BLOCK_K=table_block_k,
                num_warps=1,
            )
            if copy_indexed_kv:
                missing_leaves = int((source_leaf_indices < 0).sum().item())
                if missing_leaves:
                    raise AssertionError(
                        f"AITER copied-expert simulation could not resolve "
                        f"{missing_leaves} archived leaves"
                    )
                page_counts = torch.div(
                    k_lengths + copy_page_size - 1,
                    copy_page_size,
                    rounding_mode="floor",
                )
                kv_indptr = F.pad(page_counts.cumsum(0), (1, 0)).to(torch.int32)
                copied_pages = int(kv_indptr[-1].item())
                # AITER's page-size-16 V path can speculatively touch padding
                # even when kv_last_page_lens masks it from the softmax.  Keep
                # those diagnostic-only physical pages zeroed so 0 * garbage
                # cannot manufacture NaNs for one-token experts.
                copied_k = torch.zeros(
                    copied_pages,
                    copy_page_size,
                    1,
                    head_dim,
                    dtype=page_k.dtype,
                    device=q.device,
                )
                copied_v = torch.zeros(
                    copied_pages,
                    copy_page_size,
                    1,
                    value_dim,
                    dtype=page_v.dtype,
                    device=q.device,
                )
                copy_block_k = 16
                _copy_aiter_indexed_expert_kv_kernel[
                    (
                        int(unique_expert.numel()),
                        triton.cdiv(max_k, copy_block_k),
                    )
                ](
                    page_k,
                    page_v,
                    source_leaf_indices,
                    source_indptr,
                    kv_indptr,
                    k_lengths,
                    copied_k,
                    copied_v,
                    max_k,
                    HEAD_DIM=head_dim,
                    VALUE_DIM=value_dim,
                    COPY_PAGE_SIZE=copy_page_size,
                    BLOCK_K=copy_block_k,
                    HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
                    VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
                    num_warps=4,
                )
                kv_page_indices = torch.arange(
                    copied_pages, dtype=torch.int32, device=q.device
                )
                kv_last_page_lens = ((k_lengths - 1) % copy_page_size + 1).to(
                    torch.int32
                )
                page_k_flat = copied_k
                page_v_flat = copied_v
            else:
                kv_indptr = source_indptr
                kv_page_indices = source_leaf_indices
                kv_last_page_lens = torch.ones_like(k_lengths, dtype=torch.int32)
                page_k_flat = page_k.reshape(-1, 1, head_dim).unsqueeze(2)
                page_v_flat = page_v.reshape(-1, 1, value_dim).unsqueeze(2)
        else:
            page_counts = torch.div(
                k_lengths + page_size - 1,
                page_size,
                rounding_mode="floor",
            )
            max_pages = (max_k + page_size - 1) // page_size
            if max_pages > inline_pages:
                raise RuntimeError(
                    "AITER varlen leaf attention does not yet materialize overflow "
                    "page-table entries"
                )
            kv_indptr = F.pad(page_counts.cumsum(0), (1, 0)).to(torch.int32)
            local_pages = (
                slot_pages.reshape(-1, inline_pages)
                .index_select(0, unique_expert)[:, :max_pages]
                .to(torch.int32)
            )
            page_mask = (
                torch.arange(max_pages, device=q.device)[None, :]
                < page_counts[:, None]
            )
            physical_page_base = (expert_kv_row * page_capacity).to(torch.int32)
            physical_pages = local_pages + physical_page_base[:, None]
            kv_page_indices = physical_pages[page_mask].contiguous()
            kv_last_page_lens = ((k_lengths - 1) % page_size + 1).to(torch.int32)
            page_k_flat = page_k.reshape(-1, page_size, head_dim).unsqueeze(2)
            page_v_flat = page_v.reshape(-1, page_size, value_dim).unsqueeze(2)

    record_boundary()
    route_rows = rows * route_count
    packed_q = torch.empty(
        route_rows, 1, head_dim, dtype=q.dtype, device=q.device
    )
    layout_block_m = 16
    _pack_aiter_expert_queries_kernel[
        (triton.cdiv(route_rows, layout_block_m),)
    ](
        q,
        order,
        packed_q,
        route_rows,
        ROUTE_COUNT=route_count,
        HEAD_DIM=head_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        BLOCK_M=layout_block_m,
        num_warps=4,
    )
    route_out = torch.empty(
        route_rows, value_dim, dtype=q.dtype, device=q.device
    )
    route_lse = torch.empty(route_rows, dtype=torch.float32, device=q.device)
    record_boundary()

    packed_out, packed_lse = mha_batch_prefill_func(
        packed_q,
        page_k_flat,
        page_v_flat,
        cu_q,
        kv_indptr,
        kv_page_indices,
        max_q,
        max_k,
        softmax_scale=float(scale),
        causal=False,
        return_lse=True,
        kv_last_page_lens=kv_last_page_lens,
        seqlen_k=k_lengths.to(torch.int32),
    )
    record_boundary()
    _scatter_aiter_expert_routes_kernel[
        (triton.cdiv(route_rows, layout_block_m),)
    ](
        packed_out,
        packed_lse,
        order,
        route_out,
        route_lse,
        route_rows,
        VALUE_DIM=value_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        BLOCK_M=layout_block_m,
        num_warps=4,
    )
    exact_out = torch.empty(rows, value_dim, dtype=q.dtype, device=q.device)
    exact_lse = torch.empty(rows, dtype=torch.float32, device=q.device)
    _reduce_expert_route_attention_kernel[(rows,)](
        route_out,
        route_lse,
        exact_out,
        exact_lse,
        ROUTE_COUNT=route_count,
        ROUTE_BLOCK=triton.next_power_of_2(route_count),
        VALUE_DIM=value_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        num_warps=4,
    )
    record_boundary()
    if copy_indexed_kv and (
        not bool(torch.isfinite(packed_out).all().item())
        or not bool(torch.isfinite(packed_lse).all().item())
    ):
        bad_output = int((~torch.isfinite(packed_out)).sum().item())
        bad_lse = int((~torch.isfinite(packed_lse)).sum().item())
        raise AssertionError(
            "copied-expert AITER returned non-finite values: "
            f"output={bad_output}, lse={bad_lse}, experts={unique_expert.numel()}, "
            f"max_q={max_q}, max_k={max_k}, pages={kv_page_indices.numel()}"
        )
    if copy_indexed_kv and (
        not bool(torch.isfinite(exact_out).all().item())
        or not bool(torch.isfinite(exact_lse).all().item())
    ):
        bad_output = int((~torch.isfinite(exact_out)).sum().item())
        bad_lse = int((~torch.isfinite(exact_lse)).sum().item())
        raise AssertionError(
            "copied-expert route merge returned non-finite values: "
            f"output={bad_output}, lse={bad_lse}"
        )
    if timing_events is not None:
        for name, begin, end in zip(
            ("dispatch", "pack", "kernel", "reduce"),
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        timing_events.setdefault("total", []).append((boundaries[0], boundaries[-1]))
    return (
        exact_out.reshape(batch, query_heads, query_len, value_dim),
        exact_lse.reshape(batch, query_heads, query_len),
    )


def rehash_overflow_pages(
    source_keys: torch.Tensor,
    source_values: torch.Tensor,
    destination_keys: torch.Tensor,
    destination_values: torch.Tensor,
    destination_used: torch.Tensor,
    destination_flag: torch.Tensor,
    *,
    source_slot: int,
    destination_slot: int,
    hash_probes: int = 32,
) -> None:
    """Move one sparse page hash row between differently sized fixed pools."""
    if source_keys.shape != source_values.shape or source_keys.ndim != 3:
        raise ValueError("source overflow page tables must be matching rank three")
    if destination_keys.shape != destination_values.shape or destination_keys.ndim != 3:
        raise ValueError("destination overflow page tables must be matching rank three")
    if int(source_keys.size(1)) != int(destination_keys.size(1)):
        raise ValueError("overflow page tables have different KV head counts")
    destination_capacity = int(destination_keys.size(2))
    if destination_capacity & (destination_capacity - 1):
        raise ValueError("destination overflow hash capacity must be a power of two")
    if not 0 <= source_slot < int(source_keys.size(0)):
        raise IndexError("source overflow hash slot is out of range")
    if not 0 <= destination_slot < int(destination_keys.size(0)):
        raise IndexError("destination overflow hash slot is out of range")
    entries = int(source_keys.size(1)) * int(source_keys.size(2))
    _rehash_overflow_pages_kernel[(entries,)](
        source_keys,
        source_values,
        destination_keys,
        destination_values,
        destination_used,
        destination_flag,
        source_slot,
        destination_slot,
        SOURCE_BATCH_STRIDE=source_keys.stride(0),
        SOURCE_HEAD_STRIDE=source_keys.stride(1),
        DESTINATION_BATCH_STRIDE=destination_keys.stride(0),
        DESTINATION_HEAD_STRIDE=destination_keys.stride(1),
        KV_HEADS=int(source_keys.size(1)),
        SOURCE_CAPACITY=int(source_keys.size(2)),
        DESTINATION_CAPACITY=destination_capacity,
        HASH_PROBES=hash_probes,
        num_warps=1,
    )


def append_paged_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    owners: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    overflow_flag: torch.Tensor,
    slot_lengths: torch.Tensor,
    next_page: torch.Tensor,
    *,
    hash_probes: int = 8,
    page_sum_k: torch.Tensor | None = None,
    page_sum_v: torch.Tensor | None = None,
    page_counts: torch.Tensor | None = None,
    raw_page_summary_k: torch.Tensor | None = None,
    max_leaf_tokens: int | None = None,
) -> None:
    """Assign incoming leaves to pages and write K/V without state-sized work."""
    batch, kv_heads, tokens, head_dim = k.shape
    if owners.shape != (batch, kv_heads, tokens):
        raise ValueError("page owners do not match incoming K/V")
    if slot_lengths.dtype != torch.int32 or next_page.dtype != torch.int32:
        raise TypeError("Triton page counters must use int32")
    token_rows = batch * kv_heads * tokens
    ordinals = _assign_page_ordinals(
        owners,
        slot_lengths,
        next_page,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        hash_probes=hash_probes,
        page_size=int(page_k.size(3)),
        max_leaf_tokens=max_leaf_tokens,
    )
    _write_paged_kv_kernel[(token_rows,)](
        k,
        v,
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        page_k,
        page_v,
        K_BATCH_STRIDE=int(k.stride(0)),
        K_HEAD_STRIDE=int(k.stride(1)),
        K_TOKEN_STRIDE=int(k.stride(2)),
        V_BATCH_STRIDE=int(v.stride(0)),
        V_HEAD_STRIDE=int(v.stride(1)),
        V_TOKEN_STRIDE=int(v.stride(2)),
        TOKENS=tokens,
        KV_HEADS=kv_heads,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_CAPACITY=int(page_k.size(2)),
        PAGE_SIZE=int(page_k.size(3)),
        HEAD_DIM=head_dim,
        VALUE_DIM=int(v.size(-1)),
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(int(v.size(-1))),
        num_warps=4,
    )
    summaries = (page_sum_k, page_sum_v, page_counts)
    if any(summary is not None for summary in summaries):
        if not all(summary is not None for summary in summaries):
            raise ValueError("page summary K, V, and counts must be supplied together")
        if page_sum_k is None or page_sum_v is None or page_counts is None:
            raise AssertionError("page summary tensors are missing")
        if tuple(page_sum_k.shape[:3]) != tuple(page_k.shape[:3]):
            raise ValueError("page summary K shape does not match the page cache")
        if tuple(page_sum_v.shape[:3]) != tuple(page_v.shape[:3]):
            raise ValueError("page summary V shape does not match the page cache")
        if tuple(page_counts.shape) != tuple(page_k.shape[:3]):
            raise ValueError("page summary counts shape does not match the page cache")
        if raw_page_summary_k is not None and raw_page_summary_k.shape != k.shape:
            raise ValueError("raw page-summary K must match the incoming K shape")
        block_d = 64
        summary_blocks = max(
            triton.cdiv(head_dim, block_d),
            triton.cdiv(int(v.size(-1)), block_d),
        )
        _update_page_summaries_kernel[(token_rows, summary_blocks)](
            owners,
            ordinals,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            page_k,
            page_v,
            slot_pages,
            page_k,
            page_v,
            page_k,
            page_v,
            page_sum_k,
            page_sum_v,
            page_counts,
            TOKENS=tokens,
            KV_HEADS=kv_heads,
            STATE_CAPACITY=int(slot_pages.size(2)),
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            PAGE_CAPACITY=int(page_k.size(2)),
            HASH_CAPACITY=int(overflow_page_values.size(2)),
            HASH_PROBES=hash_probes,
            PAGE_SIZE=int(page_k.size(3)),
            HEAD_DIM=head_dim,
            VALUE_DIM=int(v.size(-1)),
            BLOCK_D=block_d,
            LEAF_K_BATCH_STRIDE=0,
            LEAF_K_HEAD_STRIDE=0,
            LEAF_K_TOKEN_STRIDE=0,
            LEAF_V_BATCH_STRIDE=0,
            LEAF_V_HEAD_STRIDE=0,
            LEAF_V_TOKEN_STRIDE=0,
            INDEXED=False,
            INT8_STORAGE=False,
            UPDATE_KEY=raw_page_summary_k is None,
            num_warps=4,
        )
        if raw_page_summary_k is not None:
            _update_raw_page_key_summaries_kernel[
                (token_rows, triton.cdiv(head_dim, block_d))
            ](
                owners,
                ordinals,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                raw_page_summary_k,
                page_sum_k,
                TOKENS=tokens,
                KV_HEADS=kv_heads,
                STATE_CAPACITY=int(slot_pages.size(2)),
                INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                PAGE_CAPACITY=int(page_k.size(2)),
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=hash_probes,
                PAGE_SIZE=int(page_k.size(3)),
                HEAD_DIM=head_dim,
                BLOCK_D=block_d,
                K_BATCH_STRIDE=int(raw_page_summary_k.stride(0)),
                K_HEAD_STRIDE=int(raw_page_summary_k.stride(1)),
                K_TOKEN_STRIDE=int(raw_page_summary_k.stride(2)),
                num_warps=4,
            )


def append_virtual_paged_kv(
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    leaf_offset: int,
    owners: torch.Tensor,
    page_indices: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    overflow_flag: torch.Tensor,
    slot_lengths: torch.Tensor,
    next_page: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    *,
    leaf_k_token_scales: torch.Tensor | None = None,
    leaf_v_token_scales: torch.Tensor | None = None,
    source_k: torch.Tensor | None = None,
    source_v: torch.Tensor | None = None,
    hash_probes: int = 8,
    quantized_leaf_k: torch.Tensor | None = None,
    quantized_leaf_v: torch.Tensor | None = None,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    page_quantized_counts: torch.Tensor | None = None,
    quant_group_size: int = 32,
    quant_bits: int = 4,
    quantize_touched: bool = True,
    optimize_scale: bool = False,
    fake_key_quant_bits: int = 0,
    fake_value_quant_bits: int = 0,
    fake_quantize_summaries: bool = False,
    optimize_summary_scale: bool = False,
    raw_page_summary_k: torch.Tensor | None = None,
) -> None:
    """Build virtual owner pages without copying the original sequence K/V."""
    owners = owners.contiguous()
    batch, kv_heads, leaf_capacity, head_dim = leaf_k.shape
    tokens = int(owners.size(2))
    if owners.shape[:2] != (batch, kv_heads):
        raise ValueError("virtual page owners do not match flat K/V")
    if leaf_offset < 0 or leaf_offset + tokens > leaf_capacity:
        raise ValueError("virtual page append exceeds the flat K/V cache")
    if leaf_v.shape[:3] != leaf_k.shape[:3]:
        raise ValueError("flat K/V cache shapes do not match")
    int8_storage = leaf_k.dtype == torch.int8 or leaf_v.dtype == torch.int8
    if int8_storage:
        if leaf_k.dtype != torch.int8 or leaf_v.dtype != torch.int8:
            raise TypeError("virtual INT8 storage requires both K and V in INT8")
        if leaf_k_token_scales is None or leaf_v_token_scales is None:
            raise ValueError("virtual INT8 storage requires per-token scales")
        if tuple(leaf_k_token_scales.shape) != tuple(leaf_k.shape[:-1]):
            raise ValueError("virtual INT8 K scales do not match leaf storage")
        if tuple(leaf_v_token_scales.shape) != tuple(leaf_v.shape[:-1]):
            raise ValueError("virtual INT8 V scales do not match leaf storage")
        if source_k is None or source_v is None:
            raise ValueError("virtual INT8 append requires source K/V")
        if tuple(source_k.shape) != (batch, kv_heads, tokens, head_dim):
            raise ValueError("virtual INT8 source K has the wrong shape")
        if tuple(source_v.shape) != (batch, kv_heads, tokens, int(leaf_v.size(-1))):
            raise ValueError("virtual INT8 source V has the wrong shape")
    if tuple(page_indices.shape[:2]) != (batch, kv_heads):
        raise ValueError("virtual page index rows do not match flat K/V")
    if int(page_indices.size(3)) != 16:
        raise ValueError("virtual page append requires 16-entry pages")
    if slot_lengths.dtype != torch.int32 or next_page.dtype != torch.int32:
        raise TypeError("Triton page counters must use int32")
    if raw_page_summary_k is not None:
        expected = (batch, kv_heads, tokens, head_dim)
        if tuple(raw_page_summary_k.shape) != expected:
            raise ValueError("raw page-summary K must match the appended K shape")
    token_rows = batch * kv_heads * tokens
    ordinals = _assign_page_ordinals(
        owners,
        slot_lengths,
        next_page,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        hash_probes=hash_probes,
        page_size=int(page_indices.size(3)),
    )
    if int8_storage:
        _write_virtual_int8_kv_kernel[(token_rows,)](
            source_k,
            source_v,
            owners,
            leaf_k,
            leaf_v,
            leaf_k_token_scales,
            leaf_v_token_scales,
            K_BATCH_STRIDE=int(source_k.stride(0)),
            K_HEAD_STRIDE=int(source_k.stride(1)),
            K_TOKEN_STRIDE=int(source_k.stride(2)),
            V_BATCH_STRIDE=int(source_v.stride(0)),
            V_HEAD_STRIDE=int(source_v.stride(1)),
            V_TOKEN_STRIDE=int(source_v.stride(2)),
            LEAF_OFFSET=leaf_offset,
            TOKENS=tokens,
            KV_HEADS=kv_heads,
            LEAF_CAPACITY=leaf_capacity,
            HEAD_DIM=head_dim,
            VALUE_DIM=int(leaf_v.size(-1)),
            HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
            VALUE_BLOCK_DIM=triton.next_power_of_2(int(leaf_v.size(-1))),
            num_warps=4,
        )
    _write_virtual_page_indices_kernel[(token_rows,)](
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        page_indices,
        LEAF_OFFSET=leaf_offset,
        TOKENS=tokens,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_CAPACITY=int(page_indices.size(2)),
        PAGE_SIZE=int(page_indices.size(3)),
        num_warps=1,
    )
    block_d = 64
    summary_blocks = max(
        triton.cdiv(head_dim, block_d),
        triton.cdiv(int(leaf_v.size(-1)), block_d),
    )
    _update_page_summaries_kernel[(token_rows, summary_blocks)](
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        leaf_k,
        leaf_v,
        page_indices,
        leaf_k,
        leaf_v,
        leaf_k_token_scales if int8_storage else leaf_k,
        leaf_v_token_scales if int8_storage else leaf_v,
        page_sum_k,
        page_sum_v,
        page_counts,
        TOKENS=tokens,
        KV_HEADS=kv_heads,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        PAGE_CAPACITY=int(page_indices.size(2)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=int(page_indices.size(3)),
        HEAD_DIM=head_dim,
        VALUE_DIM=int(leaf_v.size(-1)),
        BLOCK_D=block_d,
        LEAF_K_BATCH_STRIDE=int(leaf_k.stride(0)),
        LEAF_K_HEAD_STRIDE=int(leaf_k.stride(1)),
        LEAF_K_TOKEN_STRIDE=int(leaf_k.stride(2)),
        LEAF_V_BATCH_STRIDE=int(leaf_v.stride(0)),
        LEAF_V_HEAD_STRIDE=int(leaf_v.stride(1)),
        LEAF_V_TOKEN_STRIDE=int(leaf_v.stride(2)),
        INDEXED=True,
        INT8_STORAGE=int8_storage,
        UPDATE_KEY=raw_page_summary_k is None,
        num_warps=4,
    )
    if raw_page_summary_k is not None:
        _update_raw_page_key_summaries_kernel[
            (token_rows, triton.cdiv(head_dim, block_d))
        ](
            owners,
            ordinals,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            raw_page_summary_k,
            page_sum_k,
            TOKENS=tokens,
            KV_HEADS=kv_heads,
            STATE_CAPACITY=int(slot_pages.size(2)),
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            PAGE_CAPACITY=int(page_indices.size(2)),
            HASH_CAPACITY=int(overflow_page_values.size(2)),
            HASH_PROBES=hash_probes,
            PAGE_SIZE=int(page_indices.size(3)),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            K_BATCH_STRIDE=int(raw_page_summary_k.stride(0)),
            K_HEAD_STRIDE=int(raw_page_summary_k.stride(1)),
            K_TOKEN_STRIDE=int(raw_page_summary_k.stride(2)),
            num_warps=4,
        )
    if fake_key_quant_bits not in (0, 4, 8) or fake_value_quant_bits not in (
        0,
        4,
        8,
    ):
        raise ValueError("virtual fake quantization supports 0, 4, or 8 bits")
    if fake_key_quant_bits or fake_value_quant_bits:
        value_dim = int(leaf_v.size(-1))
        if fake_key_quant_bits and head_dim % quant_group_size:
            raise ValueError("virtual K group size must divide the key dimension")
        if fake_value_quant_bits and value_dim % quant_group_size:
            raise ValueError("virtual V group size must divide the value dimension")
        group_count = max(
            head_dim // quant_group_size if fake_key_quant_bits else 0,
            value_dim // quant_group_size if fake_value_quant_bits else 0,
        )
        _fake_quantize_completed_virtual_pages_kernel[(token_rows, group_count)](
            owners,
            ordinals,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            page_indices,
            leaf_k,
            leaf_v,
            page_sum_k,
            page_sum_v,
            page_counts,
            TOKENS=tokens,
            KV_HEADS=kv_heads,
            STATE_CAPACITY=int(slot_pages.size(2)),
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            PAGE_CAPACITY=int(page_indices.size(2)),
            HASH_CAPACITY=int(overflow_page_values.size(2)),
            HASH_PROBES=hash_probes,
            PAGE_SIZE=int(page_indices.size(3)),
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            GROUP_SIZE=quant_group_size,
            LEAF_CAPACITY=leaf_capacity,
            LEAF_K_BATCH_STRIDE=int(leaf_k.stride(0)),
            LEAF_K_HEAD_STRIDE=int(leaf_k.stride(1)),
            LEAF_K_TOKEN_STRIDE=int(leaf_k.stride(2)),
            LEAF_V_BATCH_STRIDE=int(leaf_v.stride(0)),
            LEAF_V_HEAD_STRIDE=int(leaf_v.stride(1)),
            LEAF_V_TOKEN_STRIDE=int(leaf_v.stride(2)),
            KEY_BITS=fake_key_quant_bits,
            VALUE_BITS=fake_value_quant_bits,
            OPTIMIZE_LEAF_SCALE=optimize_scale,
            QUANTIZE_SUMMARIES=fake_quantize_summaries,
            OPTIMIZE_SUMMARY_SCALE=optimize_summary_scale,
            num_warps=4,
        )
    quantization_tensors = (
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
    )
    if any(tensor is not None for tensor in quantization_tensors):
        if not all(isinstance(tensor, torch.Tensor) for tensor in quantization_tensors):
            raise ValueError("virtual quantized tensors must be supplied together")
        if quant_bits not in (4, 8):
            raise ValueError("virtual leaf storage supports 4 or 8 bits")
        if head_dim % quant_group_size or int(leaf_v.size(-1)) % quant_group_size:
            raise ValueError(
                "virtual quantization group size must divide K/V dimensions"
            )
        if quant_group_size % 2:
            raise ValueError("virtual quantization group size must be even")
        key_width = head_dim // 2 if quant_bits == 4 else head_dim
        value_width = (
            int(leaf_v.size(-1)) // 2 if quant_bits == 4 else int(leaf_v.size(-1))
        )
        code_dtype = torch.uint8 if quant_bits == 4 else torch.int8
        if (
            tuple(quantized_leaf_k.shape[:2]) != (batch, kv_heads)
            or int(quantized_leaf_k.size(2)) < leaf_capacity
            or int(quantized_leaf_k.size(3)) != key_width
            or quantized_leaf_k.dtype != code_dtype
        ):
            raise ValueError("quantized virtual K does not match the flat cache")
        if (
            tuple(quantized_leaf_v.shape[:2]) != (batch, kv_heads)
            or int(quantized_leaf_v.size(2)) < leaf_capacity
            or int(quantized_leaf_v.size(3)) != value_width
            or quantized_leaf_v.dtype != code_dtype
        ):
            raise ValueError("quantized virtual V does not match the flat cache")
        if quantize_touched:
            group_count = max(
                head_dim // quant_group_size,
                int(leaf_v.size(-1)) // quant_group_size,
            )
            _quantize_touched_virtual_pages_kernel[(token_rows, group_count)](
                owners,
                ordinals,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                page_indices,
                leaf_k,
                leaf_v,
                page_sum_k,
                page_sum_v,
                page_counts,
                quantized_leaf_k,
                quantized_leaf_v,
                page_k_scales,
                page_v_scales,
                page_quantized_counts,
                TOKENS=tokens,
                KV_HEADS=kv_heads,
                STATE_CAPACITY=int(slot_pages.size(2)),
                INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                PAGE_CAPACITY=int(page_indices.size(2)),
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=hash_probes,
                PAGE_SIZE=int(page_indices.size(3)),
                HEAD_DIM=head_dim,
                VALUE_DIM=int(leaf_v.size(-1)),
                GROUP_SIZE=quant_group_size,
                LEAF_CAPACITY=int(quantized_leaf_k.size(2)),
                LEAF_K_BATCH_STRIDE=int(leaf_k.stride(0)),
                LEAF_K_HEAD_STRIDE=int(leaf_k.stride(1)),
                LEAF_K_TOKEN_STRIDE=int(leaf_k.stride(2)),
                LEAF_V_BATCH_STRIDE=int(leaf_v.stride(0)),
                LEAF_V_HEAD_STRIDE=int(leaf_v.stride(1)),
                LEAF_V_TOKEN_STRIDE=int(leaf_v.stride(2)),
                QUANT_BITS=quant_bits,
                OPTIMIZE_SCALE=optimize_scale,
                num_warps=4,
            )


@triton.jit
def _quantize_page_summaries_int8_kernel(
    source,
    codes,
    scales,
    DIMENSION: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    OPTIMIZE_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    offset = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    valid = offset < DIMENSION
    values = tl.load(
        source + row * DIMENSION + offset,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), axis=0) / 127.0, 1.0e-8)
    quantized_float = tl.maximum(
        tl.minimum(tl.floor(values / scale + 0.5), 127.0), -127.0
    )
    if OPTIMIZE_SCALE:
        denominator = tl.sum(quantized_float * quantized_float, axis=0)
        numerator = tl.sum(values * quantized_float, axis=0)
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
        quantized_float = tl.maximum(
            tl.minimum(tl.floor(values / scale + 0.5), 127.0), -127.0
        )
        denominator = tl.sum(quantized_float * quantized_float, axis=0)
        numerator = tl.sum(values * quantized_float, axis=0)
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
    tl.store(
        codes + row * DIMENSION + offset,
        quantized_float.to(tl.int8),
        mask=valid,
    )
    tl.store(scales + row * (DIMENSION // GROUP_SIZE) + group, scale)


def quantize_page_summaries_int8(
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    *,
    quant_group_size: int = 32,
    optimize_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack finalized page sums with symmetric groupwise INT8."""
    if page_sum_k.shape[:3] != page_sum_v.shape[:3]:
        raise ValueError("page K/V summary rows do not match")
    if not page_sum_k.is_cuda or not page_sum_v.is_cuda:
        raise ValueError("page-summary quantization requires CUDA tensors")
    outputs: list[torch.Tensor] = []
    output_scales: list[torch.Tensor] = []
    for source in (page_sum_k, page_sum_v):
        dimension = int(source.size(-1))
        if dimension % quant_group_size:
            raise ValueError("summary group size must divide the summary dimension")
        codes = torch.empty_like(source, dtype=torch.int8)
        scales = torch.empty(
            *source.shape[:-1],
            dimension // quant_group_size,
            dtype=source.dtype,
            device=source.device,
        )
        rows = source.numel() // dimension
        _quantize_page_summaries_int8_kernel[(rows, dimension // quant_group_size)](
            source,
            codes,
            scales,
            DIMENSION=dimension,
            GROUP_SIZE=quant_group_size,
            OPTIMIZE_SCALE=optimize_scale,
            num_warps=1,
        )
        outputs.append(codes)
        output_scales.append(scales)
    return outputs[0], outputs[1], output_scales[0], output_scales[1]


def quantize_virtual_paged_kv(
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    quantized_leaf_k: torch.Tensor,
    quantized_leaf_v: torch.Tensor,
    page_k_scales: torch.Tensor,
    page_v_scales: torch.Tensor,
    page_quantized_counts: torch.Tensor,
    *,
    quant_group_size: int = 32,
    quant_bits: int = 4,
    optimize_scale: bool = False,
) -> None:
    """Quantize every populated page after BF16 prefill attention has finished."""
    batch, kv_heads, _, head_dim = leaf_k.shape
    value_dim = int(leaf_v.size(-1))
    if leaf_v.shape[:3] != leaf_k.shape[:3]:
        raise ValueError("flat K/V cache shapes do not match")
    if head_dim % quant_group_size or value_dim % quant_group_size:
        raise ValueError("virtual quantization group size must divide K/V dimensions")
    if quant_bits not in (4, 8):
        raise ValueError("virtual leaf storage supports 4 or 8 bits")
    if quant_group_size % 2:
        raise ValueError("virtual quantization group size must be even")
    if tuple(page_indices.shape[:2]) != (batch, kv_heads):
        raise ValueError("virtual page index rows do not match flat K/V")
    leaf_capacity = int(quantized_leaf_k.size(2))
    if int(leaf_k.size(2)) > leaf_capacity:
        raise ValueError("quantized virtual cache is smaller than its BF16 source")
    key_width = head_dim // 2 if quant_bits == 4 else head_dim
    value_width = value_dim // 2 if quant_bits == 4 else value_dim
    code_dtype = torch.uint8 if quant_bits == 4 else torch.int8
    if (
        int(quantized_leaf_k.size(-1)) != key_width
        or int(quantized_leaf_v.size(-1)) != value_width
        or quantized_leaf_k.dtype != code_dtype
        or quantized_leaf_v.dtype != code_dtype
    ):
        raise ValueError("quantized virtual cache layout does not match quant_bits")
    page_capacity = int(page_indices.size(2))
    group_count = max(
        head_dim // quant_group_size,
        value_dim // quant_group_size,
    )
    _quantize_all_virtual_pages_kernel[(batch * kv_heads * page_capacity, group_count)](
        page_indices,
        leaf_k,
        leaf_v,
        page_sum_k,
        page_sum_v,
        page_counts,
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
        KV_HEADS=kv_heads,
        PAGE_CAPACITY=page_capacity,
        PAGE_SIZE=int(page_indices.size(3)),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        GROUP_SIZE=quant_group_size,
        LEAF_CAPACITY=leaf_capacity,
        LEAF_K_BATCH_STRIDE=int(leaf_k.stride(0)),
        LEAF_K_HEAD_STRIDE=int(leaf_k.stride(1)),
        LEAF_K_TOKEN_STRIDE=int(leaf_k.stride(2)),
        LEAF_V_BATCH_STRIDE=int(leaf_v.stride(0)),
        LEAF_V_HEAD_STRIDE=int(leaf_v.stride(1)),
        LEAF_V_TOKEN_STRIDE=int(leaf_v.stride(2)),
        QUANT_BITS=quant_bits,
        OPTIMIZE_SCALE=optimize_scale,
        num_warps=4,
    )


def append_quantized_virtual_paged_kv(
    append_k: torch.Tensor,
    append_v: torch.Tensor,
    leaf_offset: int,
    owners: torch.Tensor,
    page_indices: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    overflow_flag: torch.Tensor,
    slot_lengths: torch.Tensor,
    next_page: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    quantized_leaf_k: torch.Tensor,
    quantized_leaf_v: torch.Tensor,
    page_k_scales: torch.Tensor,
    page_v_scales: torch.Tensor,
    page_quantized_counts: torch.Tensor,
    *,
    hash_probes: int = 8,
    quant_group_size: int = 32,
    quant_bits: int = 4,
    quantized_page_sum_k: torch.Tensor | None = None,
    quantized_page_sum_v: torch.Tensor | None = None,
    page_sum_k_scales: torch.Tensor | None = None,
    page_sum_v_scales: torch.Tensor | None = None,
    optimize_summary_scale: bool = False,
    optimize_leaf_scale: bool = False,
) -> None:
    """Append decode leaves by requantizing only pages changed by the append."""
    owners = owners.contiguous()
    append_k = append_k.contiguous()
    append_v = append_v.contiguous()
    batch, kv_heads, tokens, head_dim = append_k.shape
    value_dim = int(append_v.size(-1))
    leaf_capacity = int(quantized_leaf_k.size(2))
    if append_v.shape[:3] != append_k.shape[:3]:
        raise ValueError("append K/V shapes do not match")
    if owners.shape != (batch, kv_heads, tokens):
        raise ValueError("virtual page owners do not match appended K/V")
    if leaf_offset < 0 or leaf_offset + tokens > leaf_capacity:
        raise ValueError("quantized virtual page append exceeds cache capacity")
    if head_dim % quant_group_size or value_dim % quant_group_size:
        raise ValueError("virtual quantization group size must divide K/V dimensions")
    if quant_bits not in (4, 8):
        raise ValueError("virtual leaf storage supports 4 or 8 bits")
    key_width = head_dim // 2 if quant_bits == 4 else head_dim
    value_width = value_dim // 2 if quant_bits == 4 else value_dim
    code_dtype = torch.uint8 if quant_bits == 4 else torch.int8
    if (
        int(quantized_leaf_k.size(-1)) != key_width
        or int(quantized_leaf_v.size(-1)) != value_width
        or quantized_leaf_k.dtype != code_dtype
        or quantized_leaf_v.dtype != code_dtype
    ):
        raise ValueError("quantized append layout does not match quant_bits")
    summary_quantization_tensors = (
        quantized_page_sum_k,
        quantized_page_sum_v,
        page_sum_k_scales,
        page_sum_v_scales,
    )
    quantized_summaries = any(
        tensor is not None for tensor in summary_quantization_tensors
    )
    if quantized_summaries and not all(
        isinstance(tensor, torch.Tensor) for tensor in summary_quantization_tensors
    ):
        raise ValueError("INT8 page-summary tensors must be supplied together")
    token_rows = batch * kv_heads * tokens
    ordinals = _assign_page_ordinals(
        owners,
        slot_lengths,
        next_page,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        hash_probes=hash_probes,
        page_size=int(page_indices.size(3)),
    )
    _write_virtual_page_indices_kernel[(token_rows,)](
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        page_indices,
        LEAF_OFFSET=leaf_offset,
        TOKENS=tokens,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_CAPACITY=int(page_indices.size(2)),
        PAGE_SIZE=int(page_indices.size(3)),
        num_warps=1,
    )
    group_count = max(
        head_dim // quant_group_size,
        value_dim // quant_group_size,
    )
    _append_quantized_virtual_pages_kernel[(token_rows, group_count)](
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        page_indices,
        append_k,
        append_v,
        page_sum_k,
        page_sum_v,
        quantized_page_sum_k if quantized_summaries else page_sum_k,
        quantized_page_sum_v if quantized_summaries else page_sum_v,
        page_sum_k_scales if quantized_summaries else page_counts,
        page_sum_v_scales if quantized_summaries else page_counts,
        page_counts,
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
        leaf_offset,
        TOKENS=tokens,
        KV_HEADS=kv_heads,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        PAGE_CAPACITY=int(page_indices.size(2)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=int(page_indices.size(3)),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        GROUP_SIZE=quant_group_size,
        LEAF_CAPACITY=leaf_capacity,
        APPEND_K_BATCH_STRIDE=int(append_k.stride(0)),
        APPEND_K_HEAD_STRIDE=int(append_k.stride(1)),
        APPEND_K_TOKEN_STRIDE=int(append_k.stride(2)),
        APPEND_V_BATCH_STRIDE=int(append_v.stride(0)),
        APPEND_V_HEAD_STRIDE=int(append_v.stride(1)),
        APPEND_V_TOKEN_STRIDE=int(append_v.stride(2)),
        QUANT_BITS=quant_bits,
        QUANTIZED_SUMMARIES=quantized_summaries,
        OPTIMIZE_SUMMARY_SCALE=optimize_summary_scale,
        OPTIMIZE_LEAF_SCALE=optimize_leaf_scale,
        num_warps=4,
    )
    _finalize_appended_virtual_page_counts_kernel[(token_rows,)](
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        page_counts,
        page_quantized_counts,
        TOKENS=tokens,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        PAGE_CAPACITY=int(page_indices.size(2)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=int(page_indices.size(3)),
        num_warps=1,
    )


def append_paged_int8_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    owners: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    page_k_scales: torch.Tensor,
    page_v_scales: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    overflow_flag: torch.Tensor,
    slot_lengths: torch.Tensor,
    next_page: torch.Tensor,
    *,
    hash_probes: int = 8,
    max_leaf_tokens: int | None = None,
    num_warps: int = 4,
) -> None:
    """Append tokenwise symmetric INT8 K/V for the prefill MMA path."""
    batch, kv_heads, tokens, head_dim = k.shape
    value_dim = int(v.size(-1))
    if owners.shape != (batch, kv_heads, tokens):
        raise ValueError("page owners do not match incoming K/V")
    if page_k.dtype != torch.int8 or page_v.dtype != torch.int8:
        raise TypeError("matrix-native INT8 pages require signed INT8 K/V")
    if tuple(page_k_scales.shape) != tuple(page_k.shape[:-1]):
        raise ValueError("INT8 page K scales do not match page storage")
    if tuple(page_v_scales.shape) != tuple(page_v.shape[:-1]):
        raise ValueError("INT8 page V scales do not match page storage")
    if slot_lengths.dtype != torch.int32 or next_page.dtype != torch.int32:
        raise TypeError("Triton page counters must use int32")
    token_rows = batch * kv_heads * tokens
    ordinals = _assign_page_ordinals(
        owners,
        slot_lengths,
        next_page,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        hash_probes=hash_probes,
        page_size=int(page_k.size(3)),
        max_leaf_tokens=max_leaf_tokens,
    )
    _write_paged_int8_kv_kernel[(token_rows,)](
        k,
        v,
        owners,
        ordinals,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        page_k,
        page_v,
        page_k_scales,
        page_v_scales,
        K_BATCH_STRIDE=int(k.stride(0)),
        K_HEAD_STRIDE=int(k.stride(1)),
        K_TOKEN_STRIDE=int(k.stride(2)),
        V_BATCH_STRIDE=int(v.stride(0)),
        V_HEAD_STRIDE=int(v.stride(1)),
        V_TOKEN_STRIDE=int(v.stride(2)),
        TOKENS=tokens,
        KV_HEADS=kv_heads,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_CAPACITY=int(page_k.size(2)),
        PAGE_SIZE=int(page_k.size(3)),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        num_warps=num_warps,
    )


# Backward-compatible name for callers that predate the cache-format switch.
# It accepts ``quant_bits`` as well, so new code should use the generic name.
quantize_virtual_paged_kv_int4 = quantize_virtual_paged_kv


__all__ = [
    "advance_decode_cache_lengths",
    "append_paged_kv",
    "append_paged_int8_kv",
    "append_quantized_virtual_paged_kv",
    "append_virtual_paged_kv",
    "aiter_bucketed_paged_leaf_attention",
    "aiter_query_tile_union_paged_leaf_attention",
    "aiter_varlen_paged_leaf_attention",
    "paged_leaf_attention",
    "query_major_paged_leaf_attention",
    "query_tile_masked_paged_leaf_attention",
    "query_tile_slot_unions",
    "remove_query_tile_union_from_coarse",
    "refine_route_candidates_by_leaf_mass",
    "refine_route_candidates_by_page_mass",
    "refine_route_candidates_by_virtual_leaf_mass",
    "refine_route_candidates_by_virtual_leaf_output",
    "query_major_indexed_residual_page_attention",
    "query_major_residual_page_attention",
    "quantize_page_summaries_int8",
    "quantize_virtual_paged_kv",
    "quantize_virtual_paged_kv_int4",
    "rehash_overflow_pages",
    "_assign_page_ordinals_kernel",
    "_paged_leaf_attention_kernel",
    "_query_major_paged_leaf_attention_kernel",
    "dense_page_summary_attention",
    "_query_major_residual_page_attention_kernel",
    "_update_page_summaries_kernel",
    "_write_virtual_page_indices_kernel",
    "_write_paged_kv_kernel",
]
