"""Triton helpers shared by paged LoD kernels."""

from __future__ import annotations

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
            # The supported context range uses at most 16K pages in one
            # posting list. A fixed stride keeps keys valid if the pool grows.
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
