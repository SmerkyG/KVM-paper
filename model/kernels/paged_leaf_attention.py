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
    slot = tl.load(
        candidates + query_row * CANDIDATE_COUNT + candidate_rank
    ).to(tl.int64)
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
            slot_pages
            + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
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
        score = (
            SCALE
            * tl.sum(
                (key_sum / count[:, None]) * query[None, :].to(tl.float32),
                axis=1,
            )
            + tl.log(count)
        )
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
    slot = tl.load(
        candidates + query_row * CANDIDATE_COUNT + candidate_rank
    ).to(tl.int64)
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
            slot_pages
            + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
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
        slot = tl.load(
            candidates + query_row * CANDIDATE_COUNT + candidate_rank
        ).to(tl.int64)
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
                slot_pages
                + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
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
            candidate_coarse_lse
            + query_row * CANDIDATE_COUNT
            + candidate_rank
        ).to(tl.float32)
        coarse_relative_mass = tl.exp(coarse_lse - closed_lse)
        exact_relative_mass = tl.exp(exact_lse - closed_lse)
        state_count = tl.load(
            state_counts + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=1.0,
        ).to(tl.float32)
        coarse_value = tl.load(
            state_sum_v
            + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM
            + value_dim,
            mask=valid_slot,
            other=0.0,
        ).to(tl.float32) / state_count
        denominator_adjustment += exact_relative_mass - coarse_relative_mass
        numerator_adjustment += (
            exact_relative_mass * exact_value
            - coarse_relative_mass * coarse_value
        )

    closed_output = tl.load(
        baseline_output + query_row * VALUE_DIM + value_dim
    ).to(tl.float32)
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
    slot = tl.load(
        candidates + query_row * CANDIDATE_COUNT + candidate_rank
    ).to(tl.int64)
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
            slot_pages
            + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
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
        candidate_coarse_lse
        + query_row * CANDIDATE_COUNT
        + candidate_rank
    ).to(tl.float32)
    coarse_relative_mass = tl.exp(coarse_lse - closed_lse)
    exact_relative_mass = tl.exp(exact_lse - closed_lse)
    new_denominator = 1.0 - coarse_relative_mass + exact_relative_mass
    closed_output = tl.load(
        baseline_output + query_row * VALUE_DIM + value_dim
    ).to(tl.float32)
    state_count = tl.load(
        state_counts + (kv_row * STATE_CAPACITY + slot),
        mask=valid_slot,
        other=1.0,
    ).to(tl.float32)
    coarse_value = tl.load(
        state_sum_v
        + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM
        + value_dim,
        mask=valid_slot,
        other=0.0,
    ).to(tl.float32) / state_count
    opened_output = (
        closed_output
        - coarse_relative_mass * coarse_value
        + exact_relative_mass * exact_value
    ) / new_denominator
    candidate_target = tl.load(
        target_output + query_row * VALUE_DIM + value_dim
    ).to(tl.float32)
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
):
    """Commit counts and publish IDs after stable ordinals are materialized."""
    kv_row = tl.program_id(0).to(tl.int64)
    token = (
        tl.program_id(1).to(tl.int64) * BLOCK_TOKENS
        + tl.arange(0, BLOCK_TOKENS)
    )
    valid = token < TOKENS
    token_row = kv_row * TOKENS + token
    owner = tl.load(owners + token_row, mask=valid, other=0).to(tl.int64)
    ordinal = tl.load(ordinals + token_row, mask=valid, other=0).to(tl.int64)

    lane = tl.arange(0, BLOCK_TOKENS)
    same_owner = (
        (owner[:, None] == owner[None, :])
        & valid[:, None]
        & valid[None, :]
    )
    earlier = lane[None, :] < lane[:, None]
    first_in_block = tl.sum(
        (same_owner & earlier).to(tl.int32), axis=1
    ) == 0
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
    positions = torch.arange(
        tokens, device=owners.device, dtype=owners.dtype
    ).view(1, 1, tokens)
    order = torch.argsort(owners * tokens + positions, dim=2)
    sorted_owners = owners.gather(2, order)
    new_group = torch.ones_like(sorted_owners, dtype=torch.bool)
    new_group[..., 1:] = sorted_owners[..., 1:] != sorted_owners[..., :-1]
    sorted_positions = positions.expand(batch, kv_heads, tokens)
    group_starts = torch.where(new_group, sorted_positions, 0).cummax(dim=2).values
    sorted_ranks = sorted_positions - group_starts
    ranks = torch.empty_like(owners)
    ranks.scatter_(2, order, sorted_ranks)
    ordinals = (
        slot_lengths.gather(2, owners).to(owners.dtype) + ranks
    ).to(torch.int32)
    block_tokens = 16
    _publish_page_ids_kernel[
        (batch * kv_heads, triton.cdiv(tokens, block_tokens))
    ](
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=page_size,
        BLOCK_TOKENS=block_tokens,
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
        source_slot * SOURCE_BATCH_STRIDE
        + head * SOURCE_HEAD_STRIDE
        + source_bucket
    )
    key = tl.load(source_keys + source_offset).to(tl.int32)
    value = tl.load(source_values + source_offset).to(tl.int32)
    active = (head < KV_HEADS) & (key >= 0)
    index = _page_hash_index(key, DESTINATION_CAPACITY)
    destination_base = (
        destination_slot * DESTINATION_BATCH_STRIDE
        + head * DESTINATION_HEAD_STRIDE
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
        mask=head_offset < HEAD_DIM,
    )
    tl.store(
        page_v + physical_token * VALUE_DIM + value_offset,
        tl.load(source_v, mask=value_offset < VALUE_DIM, other=0.0),
        mask=value_offset < VALUE_DIM,
    )


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
    UPDATE_KEY: tl.constexpr,
):
    """Refresh every completed page and each slot's current partial page."""
    token_row = tl.program_id(0).to(tl.int64)
    dimension_block = tl.program_id(1)
    kv_row = token_row // TOKENS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    slot_length = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + owner
    ).to(tl.int64)
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
    page_count = tl.where(completes_page, PAGE_SIZE, ordinal % PAGE_SIZE + 1)
    page_offset = tl.arange(0, PAGE_SIZE)
    dimension = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_page = refresh & (page_offset < page_count)

    if INDEXED:
        batch = kv_row // KV_HEADS
        kv_head = kv_row - batch * KV_HEADS
        leaf_index = tl.load(
            page_indices
            + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
            + page_offset,
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
            page_sum_k
            + (kv_row * PAGE_CAPACITY + page_id) * HEAD_DIM
            + dimension,
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
        page_sum_v
        + (kv_row * PAGE_CAPACITY + page_id) * VALUE_DIM
        + dimension,
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
    page_ordinal = ordinal // PAGE_SIZE
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
    dimension = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_key = (page_id >= 0) & (page_id < PAGE_CAPACITY) & (
        dimension < HEAD_DIM
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
    destination = (
        page_sum_k
        + (kv_row * PAGE_CAPACITY + page_id) * HEAD_DIM
        + dimension
    )
    tl.atomic_add(
        destination,
        append_key,
        mask=valid_key,
        sem="relaxed",
    )


@triton.jit
def _quantize_virtual_page_tensor_int4(
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
    OPTIMIZE_SCALE: tl.constexpr,
):
    valid_even = valid_token[:, None] & (
        even_dimension[None, :] < DIMENSION_SIZE
    )
    valid_odd = valid_token[:, None] & (
        odd_dimension[None, :] < DIMENSION_SIZE
    )
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
    even_anchor = tl.load(
        sum_base + even_dimension,
        mask=refresh & (even_dimension < DIMENSION_SIZE),
        other=0.0,
    ).to(tl.float32) * inverse_count
    odd_anchor = tl.load(
        sum_base + odd_dimension,
        mask=refresh & (odd_dimension < DIMENSION_SIZE),
        other=0.0,
    ).to(tl.float32) * inverse_count
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
    scale = tl.maximum(tl.maximum(even_max, odd_max) / 7.0, 1.0e-8)
    even_code_float = tl.maximum(
        tl.minimum(tl.floor(even_residual / scale + 0.5), 7.0), -7.0
    )
    odd_code_float = tl.maximum(
        tl.minimum(tl.floor(odd_residual / scale + 0.5), 7.0), -7.0
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
            tl.minimum(tl.floor(even_residual / scale + 0.5), 7.0), -7.0
        )
        odd_code_float = tl.maximum(
            tl.minimum(tl.floor(odd_residual / scale + 0.5), 7.0), -7.0
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
def _quantize_touched_virtual_pages_int4_kernel(
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
    slot_length = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + owner
    ).to(tl.int64)
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
        page_indices
        + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
        + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)

    _quantize_virtual_page_tensor_int4(
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
        OPTIMIZE_SCALE,
    )
    _quantize_virtual_page_tensor_int4(
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
        OPTIMIZE_SCALE,
    )
    tl.store(
        page_quantized_counts + kv_row * PAGE_CAPACITY + page_id,
        page_count,
        mask=refresh & (group == 0),
    )


@triton.jit
def _quantize_all_virtual_pages_int4_kernel(
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

    _quantize_virtual_page_tensor_int4(
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
        OPTIMIZE_SCALE,
    )
    _quantize_virtual_page_tensor_int4(
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
        OPTIMIZE_SCALE,
    )
    tl.store(
        page_quantized_counts + page_row,
        page_count,
        mask=refresh & (group == 0),
    )


@triton.jit
def _requantize_appended_virtual_page_tensor_int4(
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
    QUANTIZED_SUMMARIES: tl.constexpr,
    OPTIMIZE_SUMMARY_SCALE: tl.constexpr,
    OPTIMIZE_LEAF_SCALE: tl.constexpr,
):
    """Requantize one changed page from packed old leaves and exact new leaves."""
    valid_even = valid_token[:, None] & (
        even_dimension[None, :] < DIMENSION_SIZE
    )
    valid_odd = valid_token[:, None] & (
        odd_dimension[None, :] < DIMENSION_SIZE
    )
    old_even_valid = valid_even & old_token[:, None]
    old_odd_valid = valid_odd & old_token[:, None]
    packed_dimension = DIMENSION_SIZE // 2
    packed_base = (
        destination
        + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * packed_dimension
        + group * (GROUP_SIZE // 2)
        + pair_offset[None, :]
    )
    old_packed = tl.load(
        packed_base,
        mask=old_even_valid,
        other=0,
    ).to(tl.int32)
    old_even_code = (old_packed & 15) - 8
    old_odd_code = ((old_packed >> 4) & 15) - 8
    sum_base = page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    quantized_sum_base = (
        quantized_page_sum
        + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    )
    old_inverse_count = 1.0 / tl.maximum(old_count.to(tl.float32), 1.0)
    has_old_tokens = refresh & (old_count > 0)
    if QUANTIZED_SUMMARIES:
        old_summary_scale = tl.load(
            page_sum_scales
            + (kv_row * PAGE_CAPACITY + page_id)
            * (DIMENSION_SIZE // GROUP_SIZE)
            + group,
            mask=has_old_tokens,
            other=0.0,
        ).to(tl.float32)
        old_even_sum = tl.load(
            quantized_sum_base + even_dimension,
            mask=has_old_tokens & (even_dimension < DIMENSION_SIZE),
            other=0,
        ).to(tl.float32) * old_summary_scale
        old_odd_sum = tl.load(
            quantized_sum_base + odd_dimension,
            mask=has_old_tokens & (odd_dimension < DIMENSION_SIZE),
            other=0,
        ).to(tl.float32) * old_summary_scale
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
    valid_source = new_token & (source_index >= 0) & (
        source_index < SOURCE_TOKEN_COUNT
    )
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
            tl.minimum(
                tl.floor(new_even_sum / new_summary_scale + 0.5), 127.0
            ),
            -127.0,
        )
        new_odd_code_float = tl.maximum(
            tl.minimum(
                tl.floor(new_odd_sum / new_summary_scale + 0.5), 127.0
            ),
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
                tl.minimum(
                    tl.floor(new_even_sum / new_summary_scale + 0.5), 127.0
                ),
                -127.0,
            )
            new_odd_code_float = tl.maximum(
                tl.minimum(
                    tl.floor(new_odd_sum / new_summary_scale + 0.5), 127.0
                ),
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
            + (kv_row * PAGE_CAPACITY + page_id)
            * (DIMENSION_SIZE // GROUP_SIZE)
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
    scale = tl.maximum(tl.maximum(even_max, odd_max) / 7.0, 1.0e-8)
    even_code_float = tl.maximum(
        tl.minimum(tl.floor(even_residual / scale + 0.5), 7.0), -7.0
    )
    odd_code_float = tl.maximum(
        tl.minimum(tl.floor(odd_residual / scale + 0.5), 7.0), -7.0
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
            tl.minimum(tl.floor(even_residual / scale + 0.5), 7.0), -7.0
        )
        odd_code_float = tl.maximum(
            tl.minimum(tl.floor(odd_residual / scale + 0.5), 7.0), -7.0
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
    even_code = even_code_float.to(tl.int32) + 8
    odd_code = odd_code_float.to(tl.int32) + 8
    tl.store(
        packed_base,
        (even_code | (odd_code << 4)).to(tl.uint8),
        mask=valid_even,
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
def _append_quantized_virtual_pages_int4_kernel(
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
    slot_length = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + owner
    ).to(tl.int64)
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
        page_indices
        + (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE
        + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)

    _requantize_appended_virtual_page_tensor_int4(
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
        QUANTIZED_SUMMARIES,
        OPTIMIZE_SUMMARY_SCALE,
        OPTIMIZE_LEAF_SCALE,
    )
    _requantize_appended_virtual_page_tensor_int4(
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
    slot_length = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + owner
    ).to(tl.int64)
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


@triton.jit
def _paged_leaf_attention_kernel(
    q,
    packed_route_row,
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
    out,
    lse,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert = tl.program_id(0)
    query_block = tl.program_id(1)
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
    key_count = tl.load(
        slot_lengths + kv_row * STATE_CAPACITY + slot
    ).to(tl.int32)
    if HASH_PROBES == 0:
        page_table = (
            slot_pages
            + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
        )
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    token_offset = tl.arange(0, BLOCK_N)

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
            (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
        )
        k_block = tl.load(
            page_k
            + physical_token[None, :] * HEAD_DIM
            + head_offset[:, None],
            mask=valid_key[None, :],
            other=0.0,
        )
        v_block = tl.load(
            page_v
            + physical_token[:, None] * VALUE_DIM
            + value_offset[None, :],
            mask=valid_key[:, None],
            other=0.0,
        )

        scores = SCALE_LOG2 * tl.dot(
            q_block, k_block, out_dtype=tl.float32
        )
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
        accumulator_t += tl.dot(
            tl.trans(v_block),
            tl.trans(probabilities.to(v_block.dtype)),
            out_dtype=tl.float32,
        )
        accumulator = tl.trans(accumulator_t)
        maximum = new_maximum

    normalized = accumulator / denominator[:, None]
    natural_lse = (maximum + tl.math.log2(denominator)) * 0.6931471805599453
    tl.store(
        out + route_row[:, None] * VALUE_DIM + value_offset[None, :],
        normalized,
        mask=valid_query[:, None],
    )
    tl.store(lse + route_row, natural_lse, mask=valid_query)


@triton.jit(
    do_not_specialize=["QUERY_LEN"],
    do_not_specialize_on_alignment=["QUERY_LEN"],
)
def _query_major_paged_leaf_attention_kernel(
    q,
    page_k,
    page_v,
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
        routed_slot = tl.load(
            top_slots + query_row * ROUTE_COUNT + route
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
                slot_pages
                + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
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
                (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
            )
            keys = tl.load(
                page_k
                + physical_token[:, None] * HEAD_DIM
                + head_offset[None, :],
                mask=valid_key[:, None] & (head_offset[None, :] < HEAD_DIM),
                other=0.0,
            )
            values = tl.load(
                page_v
                + physical_token[:, None] * VALUE_DIM
                + value_offset[None, :],
                mask=valid_key[:, None]
                & (value_offset[None, :] < VALUE_DIM),
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
            denominator = (
                denominator * correction + tl.sum(probabilities, axis=0)
            )
            value_update = tl.sum(probabilities[:, None] * values, axis=0)
            accumulator = accumulator * correction + value_update
            maximum = new_maximum

    tl.store(
        out + query_row * VALUE_DIM + value_offset,
        accumulator / denominator,
        mask=value_offset < VALUE_DIM,
    )
    tl.store(
        lse + query_row,
        (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
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
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_n: int = 16,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    timing_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse all routed slots for each query into one online softmax."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("query-major paged leaf attention is forward-only")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(page_k.size(1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if int(page_k.size(3)) != 16:
        raise ValueError("query-major leaf attention requires 16-token pages")
    rows = batch * query_heads * query_len
    value_dim = int(page_v.size(-1))
    output = torch.empty(
        rows, value_dim, dtype=q.dtype, device=q.device
    )
    lse = torch.empty(rows, dtype=torch.float32, device=q.device)
    begin = None
    if timing_events is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
    _query_major_paged_leaf_attention_kernel[(rows,)](
        q,
        page_k,
        page_v,
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
        PAGE_CAPACITY=int(page_k.size(2)),
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=int(page_k.size(3)),
        ROUTE_COUNT=int(top_slots.size(-1)),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_N=block_n,
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
    QUANTIZED: tl.constexpr,
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
        routed_slot = tl.load(
            top_slots + query_row * ROUTE_COUNT + route
        ).to(tl.int64)
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
                slot_pages
                + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
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
        selected_score = tl.where(
            single_page, float("inf"), selected_score
        )
        selected_page = tl.where(single_page, first_page, selected_page)
        scan_page_count = tl.where(single_page, 0, slot_page_count)

        for page_begin in tl.range(
            0, scan_page_count, PAGE_BLOCK_N, num_stages=1
        ):
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
                    mask=valid_page[:, None]
                    & (head_offset[None, :] < HEAD_DIM),
                    other=0,
                ).to(tl.float32)
                key_sum_scales = tl.load(
                    page_sum_k_scales
                    + (kv_row * PAGE_CAPACITY + page_id[:, None])
                    * (HEAD_DIM // QUANT_GROUP_SIZE)
                    + head_offset[None, :] // QUANT_GROUP_SIZE,
                    mask=valid_page[:, None]
                    & (head_offset[None, :] < HEAD_DIM),
                    other=0.0,
                ).to(tl.float32)
                key_sums = key_sum_codes * key_sum_scales
            else:
                key_sums = tl.load(
                    page_sum_k
                    + (kv_row * PAGE_CAPACITY + page_id[:, None]) * HEAD_DIM
                    + head_offset[None, :],
                    mask=valid_page[:, None]
                    & (head_offset[None, :] < HEAD_DIM),
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
                    tl.sum(latent_values * latent_values, axis=1)
                    / MLA_LATENT_DIM
                    + MLA_NORM_EPS
                )
                unit_latent = (
                    page_keys.to(tl.float32) * inverse_rms[:, None]
                ).to(tl.bfloat16)
                norm_gain = tl.load(
                    mla_norm_weight + head_offset,
                    mask=latent_mask,
                    other=1.0,
                ).to(tl.bfloat16)
                normalized_latent = (unit_latent * norm_gain[None, :]).to(
                    tl.bfloat16
                )
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
            state_k
            + (kv_row * STATE_CAPACITY + slot) * HEAD_DIM
            + head_offset,
            mask=valid_slot & (head_offset < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        state_value_sum = tl.load(
            state_v
            + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM
            + value_offset,
            mask=valid_slot & (value_offset < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)

        if residual_count > 0.0:
            residual_key = (state_key_sum - selected_key_sum) / residual_count
            if MLA_LATENT_DIM > 0:
                residual_key = residual_key.to(tl.bfloat16)
                latent_mask = head_offset < MLA_LATENT_DIM
                latent_values = tl.where(
                    latent_mask, residual_key.to(tl.float32), 0.0
                )
                inverse_rms = tl.rsqrt(
                    tl.sum(latent_values * latent_values, axis=0)
                    / MLA_LATENT_DIM
                    + MLA_NORM_EPS
                )
                unit_latent = (
                    residual_key.to(tl.float32) * inverse_rms
                ).to(tl.bfloat16)
                norm_gain = tl.load(
                    mla_norm_weight + head_offset,
                    mask=latent_mask,
                    other=1.0,
                ).to(tl.bfloat16)
                normalized_latent = (unit_latent * norm_gain).to(
                    tl.bfloat16
                )
                residual_key = tl.where(
                    latent_mask, normalized_latent, residual_key
                ).to(tl.float32)
            residual_value = (
                state_value_sum - selected_value_sum
            ) / residual_count
            residual_score = (
                SCALE_LOG2
                * tl.sum(residual_key * query.to(tl.float32), axis=0)
                + tl.log2(residual_count)
            )
            new_maximum = tl.maximum(maximum, residual_score)
            correction = tl.math.exp2(maximum - new_maximum)
            probability = tl.math.exp2(residual_score - new_maximum)
            denominator = denominator * correction + probability
            accumulator = accumulator * correction + probability * residual_value
            maximum = new_maximum

        valid_token = selected_valid & (token_offset < selected_count)
        physical_token = (
            (kv_row * PAGE_CAPACITY + selected_page) * PAGE_SIZE + token_offset
        )
        if INDEXED:
            leaf_index = tl.load(
                page_indices + physical_token,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            leaf_index = tl.where(valid_token, leaf_index, 0)
            if QUANTIZED:
                quantized_count = tl.load(
                    page_quantized_counts
                    + kv_row * PAGE_CAPACITY
                    + selected_page
                ).to(tl.int32)
                use_quantized = valid_token & (token_offset < quantized_count)
                packed_head_offset = head_offset // 2
                packed_value_offset = value_offset // 2
                packed_keys = tl.load(
                    quantized_leaf_k
                    + (kv_row * LEAF_CAPACITY + leaf_index[:, None])
                    * (HEAD_DIM // 2)
                    + packed_head_offset[None, :],
                    mask=use_quantized[:, None]
                    & (head_offset[None, :] < HEAD_DIM),
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
                value_code = (
                    (packed_values >> value_shift[None, :]) & 15
                ) - 8
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
            keys = tl.where(
                use_quantized[:, None], quantized_keys, keys
            )
            values = tl.where(
                use_quantized[:, None], quantized_values, values
            )
        else:
            keys = tl.load(
                page_k
                + physical_token[:, None] * HEAD_DIM
                + head_offset[None, :],
                mask=valid_token[:, None]
                & (head_offset[None, :] < HEAD_DIM),
                other=0.0,
            )
            values = tl.load(
                page_v
                + physical_token[:, None] * VALUE_DIM
                + value_offset[None, :],
                mask=valid_token[:, None]
                & (value_offset[None, :] < VALUE_DIM),
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

    output_row = (
        query_row * ROUTE_COUNT + active_route
        if ROUTE_PARALLEL
        else query_row
    )
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        raise ValueError("indexed INT4 tensors must be supplied together")
    if quantized and not indexed:
        raise ValueError("INT4 residual pages require indexed virtual storage")
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
        isinstance(tensor, torch.Tensor)
        for tensor in summary_quantization_tensors
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
        raise ValueError("INT4 group size must divide K/V dimensions")
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
        expected_k_scales = expected_k_summary[:-1] + (
            head_dim // quant_group_size,
        )
        expected_v_scales = expected_v_summary[:-1] + (
            value_dim // quant_group_size,
        )
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        QUANTIZED=quantized,
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
        scores = tl.dot(
            query[None, :], tl.trans(mean_keys), out_dtype=tl.float32
        )
        scores = tl.reshape(scores, (GROUP_N,))
    else:
        scores = tl.sum(
            mean_keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
        )
    scores *= SCALE
    scores += tl.log(count)
    scores = tl.where(valid, scores, -float("inf"))
    route_scores = tl.where(slot >= PROTECTED_LEN, scores, -float("inf"))

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
    scores = tl.where(
        query_valid[:, None] & valid[None, :], scores, -float("inf")
    )
    route_scores = tl.where(
        slot[None, :] >= PROTECTED_LEN, scores, -float("inf")
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
        group_out
        + group_row[:, None] * HEAD_DIM
        + dim[None, :],
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
    for candidate_begin in tl.range(
        0, active_groups * 8, CANDIDATE_TILE, num_stages=1
    ):
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
        best_packed = tl.topk(
            tl.interleave(best_packed, block_top), 8, dim=0
        )
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
        accumulator = (
            accumulator * old_weight + current_out * current_weight
        )
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
        local_denominator = local_denominator * old_weight + tl.sum(
            weights, axis=0
        )
        if COMPUTE_LOCAL_OUTPUT:
            local_accumulator = local_accumulator * old_weight + tl.sum(
                weights[:, None] * values, axis=0
            )
        local_maximum = new_maximum
    if INCLUDE_NEW:
        current_key = tl.load(
            new_k
            + batch * NEW_K_BATCH_STRIDE
            + kv_head * NEW_K_HEAD_STRIDE
            + dim
        ).to(tl.float32)
        if COMPUTE_LOCAL_OUTPUT:
            current_value = tl.load(
                new_v
                + batch * NEW_V_BATCH_STRIDE
                + kv_head * NEW_V_HEAD_STRIDE
                + dim
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
            scores = tl.dot(
                query[None, :], tl.trans(mean_keys), out_dtype=tl.float32
            )
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
            top_slots
            + batch * TOP_BATCH_STRIDE
            + query_head * TOP_HEAD_STRIDE
            + route
        ).to(tl.int64)
        slot_valid = routed_slot >= 0
        slot = tl.where(slot_valid, routed_slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=slot_valid,
            other=0,
        ).to(tl.int32)
        if HASH_PROBES == 0:
            page_table = slot_pages + (
                kv_row * STATE_CAPACITY + slot
            ) * INLINE_PAGES_PER_SLOT
        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
            logical_key = key_begin + token_offset
            valid = logical_key < key_count
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal, mask=valid, other=0
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
                    valid,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            physical_token = (
                (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
            )
            keys = tl.load(
                page_k
                + physical_token[:, None] * HEAD_DIM
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            values = tl.load(
                page_v
                + physical_token[:, None] * VALUE_DIM
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if USE_DOT:
                scores = tl.dot(
                    query[None, :], tl.trans(keys), out_dtype=tl.float32
                )
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
            scores = tl.dot(
                query[None, :], tl.trans(keys), out_dtype=tl.float32
            )
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

    # The current decode token is not yet in the persistent local cache.  Fold
    # it into this softmax exactly once, then let one query head in each GQA
    # group append the shared KV to the cache for the next decode step.
    if INCLUDE_NEW:
        current_key = tl.load(
            new_k
            + batch * NEW_K_BATCH_STRIDE
            + kv_head * NEW_K_HEAD_STRIDE
            + dim
        )
        current_value = tl.load(
            new_v
            + batch * NEW_V_BATCH_STRIDE
            + kv_head * NEW_V_HEAD_STRIDE
            + dim
        )
        current_score = SCALE_LOG2 * tl.sum(
            current_key.to(tl.float32) * query.to(tl.float32), axis=0
        )
        new_maximum = tl.maximum(maximum, current_score)
        correction = tl.math.exp2(maximum - new_maximum)
        current_weight = tl.math.exp2(current_score - new_maximum)
        denominator = denominator * correction + current_weight
        accumulator = (
            accumulator * correction
            + current_weight * current_value.to(tl.float32)
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
def _split_decode_paged_lod_attention_kernel(
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
):
    query_row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
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
            scores = tl.dot(
                query[None, :], tl.trans(mean_keys), out_dtype=tl.float32
            )
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
            top_slots
            + batch * TOP_BATCH_STRIDE
            + query_head * TOP_HEAD_STRIDE
            + route
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
            page_table = slot_pages + (
                kv_row * STATE_CAPACITY + slot
            ) * INLINE_PAGES_PER_SLOT
        for key_begin in tl.range(0, key_count, BLOCK_N, num_stages=1):
            logical_key = key_begin + token_offset
            valid = logical_key < key_count
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal, mask=valid, other=0
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
                    valid,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            physical_token = (
                (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
            )
            keys = tl.load(
                page_k
                + physical_token[:, None] * HEAD_DIM
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            values = tl.load(
                page_v
                + physical_token[:, None] * VALUE_DIM
                + dim[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if USE_DOT:
                scores = tl.dot(
                    query[None, :], tl.trans(keys), out_dtype=tl.float32
                )
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
                scores = tl.dot(
                    query[None, :], tl.trans(keys), out_dtype=tl.float32
                )
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
            new_k
            + batch * NEW_K_BATCH_STRIDE
            + kv_head * NEW_K_HEAD_STRIDE
            + dim
        )
        current_value = tl.load(
            new_v
            + batch * NEW_V_BATCH_STRIDE
            + kv_head * NEW_V_HEAD_STRIDE
            + dim
        )
        current_score = SCALE_LOG2 * tl.sum(
            current_key.to(tl.float32) * query.to(tl.float32), axis=0
        )
        current_score = tl.where(split == 0, current_score, -float("inf"))
        new_maximum = tl.maximum(maximum, current_score)
        correction = tl.math.exp2(maximum - new_maximum)
        current_weight = tl.math.exp2(current_score - new_maximum)
        denominator = denominator * correction + current_weight
        accumulator = (
            accumulator * correction
            + current_weight * current_value.to(tl.float32)
        )
        maximum = new_maximum
        if split == 0:
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
        finished = tl.atomic_add(
            completion + query_row, 1, sem="acq_rel"
        ).to(tl.int32)
        if finished == SPLITS - 1:
            full_coarse_lse = tl.load(coarse_lse + query_row)
            remainder_out = tl.load(
                coarse_out + query_row * HEAD_DIM + dim
            ).to(tl.float32)
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
                    + batch * COUNT_BATCH_STRIDE
                    + kv_head * COUNT_HEAD_STRIDE
                    + slot * COUNT_TOKEN_STRIDE
                ).to(tl.float32)
                value = tl.load(
                    state_v
                    + batch * STATE_V_BATCH_STRIDE
                    + kv_head * STATE_V_HEAD_STRIDE
                    + slot * STATE_V_TOKEN_STRIDE
                    + dim
                ).to(tl.float32) / count
                score = tl.load(
                    top_scores + query_row * ROUTE_COUNT + route
                )
                mass = tl.where(
                    valid_slot, tl.exp(score - full_coarse_lse), 0.0
                )
                selected_mass += mass
                selected_value += mass * value
            remainder_mass = tl.maximum(1.0 - selected_mass, 1.0e-7)
            remainder_out = (remainder_out - selected_value) / remainder_mass
            remainder_lse = full_coarse_lse + tl.log(remainder_mass)

            split_offsets = tl.arange(0, SPLITS)
            split_lse = tl.load(
                partial_lse + query_row * SPLITS + split_offsets
            )
            merge_maximum = tl.maximum(
                remainder_lse, tl.max(split_lse, axis=0)
            )
            remainder_weight = tl.exp(remainder_lse - merge_maximum)
            split_weights = tl.exp(split_lse - merge_maximum)
            merge_denominator = remainder_weight + tl.sum(
                split_weights, axis=0
            )
            merge_accumulator = remainder_weight * remainder_out
            for merge_split in tl.static_range(0, SPLITS):
                split_weight = tl.exp(
                    tl.load(
                        partial_lse
                        + query_row * SPLITS
                        + merge_split
                    )
                    - merge_maximum
                )
                split_value = tl.load(
                    partial_out
                    + (query_row * SPLITS + merge_split) * HEAD_DIM
                    + dim
                )
                merge_accumulator += split_weight * split_value
            tl.store(
                output + query_row * HEAD_DIM + dim,
                merge_accumulator / merge_denominator,
            )
            tl.atomic_xchg(completion + query_row, 0, sem="release")


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
        partial_out
        + (query_row * SPLITS + split[:, None]) * VALUE_DIM
        + dim[None, :]
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
        local_value = tl.load(
            separate_local_out + query_row * HEAD_DIM + dim
        ).to(tl.float32)
        new_maximum = tl.maximum(maximum, local_lse)
        old_weight = tl.exp(maximum - new_maximum)
        new_weight = tl.exp(local_lse - new_maximum)
        denominator = denominator * old_weight + new_weight
        numerator = numerator * old_weight + new_weight * local_value
        maximum = new_maximum
    for split_index in tl.static_range(0, SPLITS):
        branch_lse = tl.load(
            partial_lse + query_row * SPLITS + split_index
        )
        branch_value = tl.load(
            partial_out
            + (query_row * SPLITS + split_index) * HEAD_DIM
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
        for local_begin in tl.range(
            0, active_local_len, LOCAL_BLOCK_N, num_stages=1
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
                scores = tl.dot(
                    query[None, :].to(keys.dtype),
                    tl.trans(keys),
                    out_dtype=tl.float32,
                )
                scores = tl.reshape(scores, (LOCAL_BLOCK_N,))
            else:
                scores = tl.sum(
                    query[None, :] * keys.to(tl.float32), axis=1
                )
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
                new_k
                + batch * NEW_K_BATCH_STRIDE
                + kv_head * NEW_K_HEAD_STRIDE
                + dim
            )
            current_value = tl.load(
                new_v
                + batch * NEW_V_BATCH_STRIDE
                + kv_head * NEW_V_HEAD_STRIDE
                + dim
            )
            current_score = (
                tl.sum(query * current_key.to(tl.float32), axis=0) * SCALE
            )
            new_maximum = tl.maximum(maximum, current_score)
            old_weight = tl.exp(maximum - new_maximum)
            new_weight = tl.exp(current_score - new_maximum)
            denominator = denominator * old_weight + new_weight
            numerator = (
                numerator * old_weight
                + new_weight * current_value.to(tl.float32)
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
) -> dict[str, torch.Tensor]:
    batch, query_heads, _, value_dim = q.shape
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
    protected_len: int = 0,
    route_top_p: float | None = None,
    route_residual_mass: float | None = None,
    reuse_residual_local_attention: bool = False,
    route_residual_use_state_bound: bool = False,
    timing_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] | None = None,
    recursive_page_cache: dict[str, torch.Tensor | int] | None = None,
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
        else page_k
    )
    if not isinstance(page_shape, torch.Tensor) or int(page_shape.size(3)) != 16:
        raise ValueError("fused LOD decode requires 16-token leaf pages")
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
            cache_indices = torch.arange(
                batch, dtype=torch.long, device=q.device
            )
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
    if fuse_state_route and split_kv == 1:
        raise ValueError("fused state routing requires split decode attention")
    if route_group_size not in {8, 16, 32, 64}:
        raise ValueError("decode route group size must be 8, 16, 32, or 64")
    if protected_len < 0 or protected_len + 8 > state_len:
        raise ValueError("protected state leaves too few decode routing candidates")
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
        raise ValueError(
            "full-mass dynamic routing requires separate final reduction"
        )
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
            else torch.empty(
                expected_output, dtype=q.dtype, device=q.device
            )
        )
    else:
        if buffers is None:
            buffers = new_fused_decode_buffers(
                q,
                splits=split_kv,
                state_capacity=(int(state_k.size(2)) if fuse_state_route else None),
                route_group_size=route_group_size,
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
                    _mask_decode_routes_residual_lse_kernel[
                        (batch * query_heads,)
                    ](
                        top_slots,
                        buffers["route_top_scores"],
                        buffers["coarse_lse"],
                        float(route_residual_mass),
                        ROUTE_COUNT=8,
                        num_warps=2,
                        waves_per_eu=waves_per_eu,
                    )
                else:
                    _mask_decode_routes_residual_mass_kernel[
                        (batch * query_heads,)
                    ](
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
                raise ValueError(
                    "fused recursive decode requires fused state routing"
                )
            if split_kv != int(top_slots.size(-1)):
                raise ValueError(
                    "fused recursive decode requires one split per route"
                )

            def cache_tensor(name: str) -> torch.Tensor:
                value = recursive_page_cache.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"fused recursive decode cache is missing {name}"
                    )
                return value

            # Residual-mass routing can reuse the local output it had to form
            # while choosing routes. Other modes compute the bounded local
            # branch separately; folding its scan into the final reducer makes
            # that kernel register-bound on the target ROCm geometry.
            reuse_separate_local = bool(
                route_residual_mass is not None
                and reuse_residual_local_attention
            )
            if not reuse_separate_local:
                local_begin = timing_begin()
                _mask_decode_routes_residual_mass_kernel[
                    (batch * query_heads,)
                ](
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
                recursive_page_cache.get(
                    "summary_quantization_finalized", False
                )
            )
            recursive_begin = timing_begin()
            recursive_out, recursive_lse = (
                query_major_indexed_residual_page_attention(
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
                        cache_tensor("quantized_leaf_k")
                        if quantized_attention
                        else None
                    ),
                    quantized_leaf_v=(
                        cache_tensor("quantized_leaf_v")
                        if quantized_attention
                        else None
                    ),
                    page_k_scales=(
                        cache_tensor("page_k_scales")
                        if quantized_attention
                        else None
                    ),
                    page_v_scales=(
                        cache_tensor("page_v_scales")
                        if quantized_attention
                        else None
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
                        cache_tensor("page_sum_k_scales")
                        if quantized_summaries
                        else None
                    ),
                    page_sum_v_scales=(
                        cache_tensor("page_sum_v_scales")
                        if quantized_summaries
                        else None
                    ),
                    quant_group_size=recursive_quant_group_size,
                    output_buffer=partial_out,
                    lse_buffer=partial_lse,
                    route_parallel=True,
                )
            )
            timing_end("recursive_leaf", recursive_begin)
            final_reduce_begin = timing_begin()
            _reduce_routed_split_decode_lod_attention_kernel[
                (batch * query_heads,)
            ](
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
        _split_decode_paged_lod_attention_kernel[(batch * query_heads, split_kv)](
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
            partial_out,
            partial_lse,
            (
                buffers["route_top_scores"]
                if fuse_state_route
                else partial_lse
            ),
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
            PAGE_CAPACITY=int(page_k.size(2)),
            STATE_CAPACITY=int(slot_pages.size(2)),
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            HASH_CAPACITY=int(overflow_page_keys.size(2)),
            HASH_PROBES=hash_probes,
            HEAD_DIM=head_dim,
            VALUE_DIM=head_dim,
            PAGE_SIZE=int(page_k.size(3)),
            ROUTE_COUNT=int(top_slots.size(-1)),
            SPLITS=split_kv,
            SCALE_LOG2=float(scale) * math.log2(math.e),
            BLOCK_N=block_n,
            USE_DOT=use_dot,
            INCLUDE_NEW=include_new,
            SEPARATE_LOCAL=(
                route_residual_mass is not None
                and reuse_residual_local_attention
            ),
            FUSE_FINAL_REDUCE=fuse_state_route and fuse_final_reduce,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
        timing_end("leaf_local", leaf_begin)
        if fuse_state_route and not fuse_final_reduce:
            final_reduce_begin = timing_begin()
            _reduce_routed_split_decode_lod_attention_kernel[
                (batch * query_heads,)
            ](
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
                partial_out,
                partial_lse,
                (
                    buffers["route_local_out"]
                    if reuse_residual_local_attention
                    else partial_out
                ),
                (
                    buffers["route_local_lse"]
                    if reuse_residual_local_attention
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
                INCLUDE_SEPARATE_LOCAL=reuse_residual_local_attention,
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    waves_per_eu: int = 1,
    timing_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to the exact leaves of every routed slot and merge by LSE."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("paged leaf Triton attention is forward-only")
    batch, query_heads, query_len, head_dim = q.shape
    route_count = int(top_slots.size(-1))
    kv_heads = int(page_k.size(1))
    value_dim = int(page_v.size(-1))
    page_size = int(page_k.size(3))
    state_capacity = int(slot_pages.size(2))
    if page_size != 16:
        raise ValueError("paged leaf Triton attention requires 16-token pages")
    if head_dim != value_dim:
        raise ValueError("paged leaf Triton attention requires equal QK/V dimensions")
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
        query_row = torch.arange(rows, device=q.device, dtype=torch.long)
        bh_for_row = torch.div(query_row, query_len, rounding_mode="floor")
        batch_for_row = torch.div(
            bh_for_row, query_heads, rounding_mode="floor"
        )
        query_head_for_row = bh_for_row % query_heads
        kv_row_for_query = (
            batch_for_row * kv_heads
            + torch.div(
                query_head_for_row, kv_group_size, rounding_mode="floor"
            )
        )
        route_slot = top_slots.reshape(rows, route_count)
        expert_id = (
            kv_row_for_query.unsqueeze(-1) * state_capacity + route_slot
        ).reshape(-1)
        order = expert_id.argsort(stable=False)
        sorted_expert = expert_id[order]
        unique_expert, q_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        expert_kv_row = torch.div(
            unique_expert, state_capacity, rounding_mode="floor"
        )
        expert_slot = unique_expert % state_capacity
        cu_q = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
        q_lengths = q_lengths.to(torch.int32)
        max_q = int(q_lengths.max().item())

    record_boundary()

    route_out = torch.empty(
        rows * route_count,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    route_lse = torch.empty(
        rows * route_count, dtype=torch.float32, device=q.device
    )
    record_boundary()
    grid = (int(q_lengths.numel()), triton.cdiv(max_q, block_m))
    _paged_leaf_attention_kernel[grid](
        q,
        order,
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
        PAGE_CAPACITY=int(page_k.size(2)),
        STATE_CAPACITY=state_capacity,
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        PAGE_SIZE=page_size,
        ROUTE_COUNT=route_count,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
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
        timing_events.setdefault("total", []).append(
            (boundaries[0], boundaries[-1])
        )
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
        raise ValueError(
            "destination overflow page tables must be matching rank three"
        )
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
            page_sum_k,
            page_sum_v,
            page_counts,
            TOKENS=tokens,
            KV_HEADS=kv_heads,
            STATE_CAPACITY=int(slot_pages.size(2)),
            INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
            PAGE_CAPACITY=int(page_k.size(2)),
            HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
                HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
    hash_probes: int = 8,
    quantized_leaf_k: torch.Tensor | None = None,
    quantized_leaf_v: torch.Tensor | None = None,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    page_quantized_counts: torch.Tensor | None = None,
    quant_group_size: int = 32,
    quantize_touched: bool = True,
    optimize_scale: bool = False,
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        page_sum_k,
        page_sum_v,
        page_counts,
        TOKENS=tokens,
        KV_HEADS=kv_heads,
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        PAGE_CAPACITY=int(page_indices.size(2)),
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
            HASH_CAPACITY=int(overflow_page_keys.size(2)),
            HASH_PROBES=hash_probes,
            PAGE_SIZE=int(page_indices.size(3)),
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            K_BATCH_STRIDE=int(raw_page_summary_k.stride(0)),
            K_HEAD_STRIDE=int(raw_page_summary_k.stride(1)),
            K_TOKEN_STRIDE=int(raw_page_summary_k.stride(2)),
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
            raise ValueError("virtual INT4 tensors must be supplied together")
        if head_dim % quant_group_size or int(leaf_v.size(-1)) % quant_group_size:
            raise ValueError("virtual INT4 group size must divide K/V dimensions")
        if quant_group_size % 2:
            raise ValueError("virtual INT4 group size must be even")
        if (
            tuple(quantized_leaf_k.shape[:2]) != (batch, kv_heads)
            or int(quantized_leaf_k.size(2)) < leaf_capacity
            or int(quantized_leaf_k.size(3)) != head_dim // 2
        ):
            raise ValueError("packed virtual K shape does not match the flat cache")
        if (
            tuple(quantized_leaf_v.shape[:2]) != (batch, kv_heads)
            or int(quantized_leaf_v.size(2)) < leaf_capacity
            or int(quantized_leaf_v.size(3)) != int(leaf_v.size(-1)) // 2
        ):
            raise ValueError("packed virtual V shape does not match the flat cache")
        if quantize_touched:
            group_count = max(
                head_dim // quant_group_size,
                int(leaf_v.size(-1)) // quant_group_size,
            )
            _quantize_touched_virtual_pages_int4_kernel[
                (token_rows, group_count)
            ](
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
                HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        _quantize_page_summaries_int8_kernel[
            (rows, dimension // quant_group_size)
        ](
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


def quantize_virtual_paged_kv_int4(
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
    optimize_scale: bool = False,
) -> None:
    """Pack every populated page after BF16 prefill attention has finished."""
    batch, kv_heads, _, head_dim = leaf_k.shape
    value_dim = int(leaf_v.size(-1))
    if leaf_v.shape[:3] != leaf_k.shape[:3]:
        raise ValueError("flat K/V cache shapes do not match")
    if head_dim % quant_group_size or value_dim % quant_group_size:
        raise ValueError("virtual INT4 group size must divide K/V dimensions")
    if quant_group_size % 2:
        raise ValueError("virtual INT4 group size must be even")
    if tuple(page_indices.shape[:2]) != (batch, kv_heads):
        raise ValueError("virtual page index rows do not match flat K/V")
    leaf_capacity = int(quantized_leaf_k.size(2))
    if int(leaf_k.size(2)) > leaf_capacity:
        raise ValueError("packed virtual cache is smaller than its BF16 source")
    page_capacity = int(page_indices.size(2))
    group_count = max(
        head_dim // quant_group_size,
        value_dim // quant_group_size,
    )
    _quantize_all_virtual_pages_int4_kernel[
        (batch * kv_heads * page_capacity, group_count)
    ](
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
        raise ValueError("quantized virtual page append exceeds packed capacity")
    if head_dim % quant_group_size or value_dim % quant_group_size:
        raise ValueError("virtual INT4 group size must divide K/V dimensions")
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
        isinstance(tensor, torch.Tensor)
        for tensor in summary_quantization_tensors
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_CAPACITY=int(page_indices.size(2)),
        PAGE_SIZE=int(page_indices.size(3)),
        num_warps=1,
    )
    group_count = max(
        head_dim // quant_group_size,
        value_dim // quant_group_size,
    )
    _append_quantized_virtual_pages_int4_kernel[
        (token_rows, group_count)
    ](
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
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
        HASH_CAPACITY=int(overflow_page_keys.size(2)),
        HASH_PROBES=hash_probes,
        PAGE_SIZE=int(page_indices.size(3)),
        num_warps=1,
    )


__all__ = [
    "advance_decode_cache_lengths",
    "append_paged_kv",
    "append_quantized_virtual_paged_kv",
    "append_virtual_paged_kv",
    "paged_leaf_attention",
    "query_major_paged_leaf_attention",
    "refine_route_candidates_by_leaf_mass",
    "refine_route_candidates_by_page_mass",
    "refine_route_candidates_by_virtual_leaf_mass",
    "refine_route_candidates_by_virtual_leaf_output",
    "query_major_indexed_residual_page_attention",
    "query_major_residual_page_attention",
    "quantize_page_summaries_int8",
    "quantize_virtual_paged_kv_int4",
    "rehash_overflow_pages",
    "_assign_page_ordinals_kernel",
    "_paged_leaf_attention_kernel",
    "_query_major_paged_leaf_attention_kernel",
    "_query_major_residual_page_attention_kernel",
    "_update_page_summaries_kernel",
    "_write_virtual_page_indices_kernel",
    "_write_paged_kv_kernel",
]
