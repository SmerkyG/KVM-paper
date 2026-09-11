"""Page directory, append, and quantization kernels for LoD caches."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._paged_common import _lookup_page_id, _page_hash_index


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


@triton.jit
def _quantize_virtual_page_tensor_grouped_int4(
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
    group_begin,
    PAGE_CAPACITY: tl.constexpr,
    DIMENSION_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    SOURCE_BATCH_STRIDE: tl.constexpr,
    SOURCE_HEAD_STRIDE: tl.constexpr,
    SOURCE_TOKEN_STRIDE: tl.constexpr,
    OPTIMIZE_SCALE: tl.constexpr,
):
    """Quantize several four-channel, page-wide INT4 groups together."""
    group_lane = tl.arange(0, GROUPS_PER_PROGRAM)
    pair_lane = tl.arange(0, 2)
    groups = group_begin + group_lane
    dimensions = groups[:, None] * 4 + pair_lane[None, :] * 2
    group_valid = groups < DIMENSION_SIZE // 4
    even_dimension = dimensions
    odd_dimension = dimensions + 1
    valid_even = (
        valid_token[:, None, None]
        & group_valid[None, :, None]
        & (even_dimension[None, :, :] < DIMENSION_SIZE)
    )
    valid_odd = valid_even & (odd_dimension[None, :, :] < DIMENSION_SIZE)
    source_base = (
        source
        + batch * SOURCE_BATCH_STRIDE
        + kv_head * SOURCE_HEAD_STRIDE
        + leaf_index[:, None, None] * SOURCE_TOKEN_STRIDE
    )
    even = tl.load(
        source_base + even_dimension[None, :, :],
        mask=valid_even,
        other=0.0,
    ).to(tl.float32)
    odd = tl.load(
        source_base + odd_dimension[None, :, :],
        mask=valid_odd,
        other=0.0,
    ).to(tl.float32)
    sum_base = page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    inverse_count = 1.0 / tl.maximum(page_count.to(tl.float32), 1.0)
    even_anchor = (
        tl.load(
            sum_base + even_dimension,
            mask=refresh & group_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        * inverse_count
    )
    odd_anchor = (
        tl.load(
            sum_base + odd_dimension,
            mask=refresh & group_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        * inverse_count
    )
    even_residual = even - even_anchor[None, :, :]
    odd_residual = odd - odd_anchor[None, :, :]
    even_max = tl.max(
        tl.max(tl.where(valid_even, tl.abs(even_residual), 0.0), axis=2),
        axis=0,
    )
    odd_max = tl.max(
        tl.max(tl.where(valid_odd, tl.abs(odd_residual), 0.0), axis=2),
        axis=0,
    )
    scale = tl.maximum(tl.maximum(even_max, odd_max) / 7.0, 1.0e-8)
    even_code_float = tl.maximum(
        tl.minimum(tl.floor(even_residual / scale[None, :, None] + 0.5), 7.0),
        -7.0,
    )
    odd_code_float = tl.maximum(
        tl.minimum(tl.floor(odd_residual / scale[None, :, None] + 0.5), 7.0),
        -7.0,
    )
    if OPTIMIZE_SCALE:
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
        even_code_float = tl.maximum(
            tl.minimum(tl.floor(even_residual / scale[None, :, None] + 0.5), 7.0),
            -7.0,
        )
        odd_code_float = tl.maximum(
            tl.minimum(tl.floor(odd_residual / scale[None, :, None] + 0.5), 7.0),
            -7.0,
        )
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=2,
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
    destination_base = (
        destination
        + (kv_row * LEAF_CAPACITY + leaf_index[:, None, None]) * (DIMENSION_SIZE // 2)
        + groups[None, :, None] * 2
        + pair_lane[None, None, :]
    )
    tl.store(destination_base, packed, mask=valid_even)
    tl.store(
        scales + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // 4) + groups,
        scale,
        mask=refresh & group_valid,
    )


@triton.jit
def _quantize_all_virtual_pages_grouped_int4_kernel(
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
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    CHANNEL_GROUP_COUNT: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    LEAF_K_BATCH_STRIDE,
    LEAF_K_HEAD_STRIDE,
    LEAF_K_TOKEN_STRIDE: tl.constexpr,
    LEAF_V_BATCH_STRIDE,
    LEAF_V_HEAD_STRIDE,
    LEAF_V_TOKEN_STRIDE: tl.constexpr,
    OPTIMIZE_SCALE: tl.constexpr,
):
    page_row = tl.program_id(0).to(tl.int64)
    group_begin = tl.program_id(1).to(tl.int64) * GROUPS_PER_PROGRAM
    kv_row = page_row // PAGE_CAPACITY
    page_id = page_row - kv_row * PAGE_CAPACITY
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    page_count = tl.load(page_counts + page_row).to(tl.int32)
    refresh = page_count > 0
    token_offset = tl.arange(0, 16)
    valid_token = refresh & (token_offset < page_count)
    leaf_index = tl.load(
        page_indices + page_row * 16 + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)
    _quantize_virtual_page_tensor_grouped_int4(
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
        group_begin,
        PAGE_CAPACITY,
        HEAD_DIM,
        GROUPS_PER_PROGRAM,
        LEAF_CAPACITY,
        LEAF_K_BATCH_STRIDE,
        LEAF_K_HEAD_STRIDE,
        LEAF_K_TOKEN_STRIDE,
        OPTIMIZE_SCALE,
    )
    _quantize_virtual_page_tensor_grouped_int4(
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
        group_begin,
        PAGE_CAPACITY,
        VALUE_DIM,
        GROUPS_PER_PROGRAM,
        LEAF_CAPACITY,
        LEAF_V_BATCH_STRIDE,
        LEAF_V_HEAD_STRIDE,
        LEAF_V_TOKEN_STRIDE,
        OPTIMIZE_SCALE,
    )
    tl.store(
        page_quantized_counts + page_row,
        page_count,
        mask=refresh & (group_begin == 0),
    )


@triton.jit
def _requantize_appended_virtual_page_tensor_grouped_int4(
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
    group_begin,
    PAGE_CAPACITY: tl.constexpr,
    DIMENSION_SIZE: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    SOURCE_TOKEN_COUNT: tl.constexpr,
    SOURCE_BATCH_STRIDE,
    SOURCE_HEAD_STRIDE,
    SOURCE_TOKEN_STRIDE: tl.constexpr,
    QUANTIZED_SUMMARIES: tl.constexpr,
    OPTIMIZE_SUMMARY_SCALE: tl.constexpr,
    OPTIMIZE_LEAF_SCALE: tl.constexpr,
):
    """Requantize several changed four-channel INT4 groups together."""
    group_lane = tl.arange(0, GROUPS_PER_PROGRAM)
    pair_lane = tl.arange(0, 2)
    groups = group_begin + group_lane
    dimensions = groups[:, None] * 4 + pair_lane[None, :] * 2
    group_valid = groups < DIMENSION_SIZE // 4
    even_dimension = dimensions
    odd_dimension = dimensions + 1
    valid_even = valid_token[:, None, None] & group_valid[None, :, None]
    valid_odd = valid_even
    old_even_valid = valid_even & old_token[:, None, None]
    destination_base = (
        destination
        + (kv_row * LEAF_CAPACITY + leaf_index[:, None, None]) * (DIMENSION_SIZE // 2)
        + groups[None, :, None] * 2
        + pair_lane[None, None, :]
    )
    old_packed = tl.load(
        destination_base,
        mask=old_even_valid,
        other=0,
    ).to(tl.int32)
    old_even_code = (old_packed & 15) - 8
    old_odd_code = ((old_packed >> 4) & 15) - 8
    sum_base = page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    quantized_sum_base = (
        quantized_page_sum + (kv_row * PAGE_CAPACITY + page_id) * DIMENSION_SIZE
    )
    old_inverse_count = 1.0 / tl.maximum(old_count.to(tl.float32), 1.0)
    has_old_tokens = refresh & (old_count > 0)
    if QUANTIZED_SUMMARIES:
        old_summary_scale = tl.load(
            page_sum_scales
            + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // 4)
            + groups,
            mask=has_old_tokens & group_valid,
            other=0.0,
        ).to(tl.float32)
        old_even_sum = (
            tl.load(
                quantized_sum_base + even_dimension,
                mask=has_old_tokens & group_valid[:, None],
                other=0,
            ).to(tl.float32)
            * old_summary_scale[:, None]
        )
        old_odd_sum = (
            tl.load(
                quantized_sum_base + odd_dimension,
                mask=has_old_tokens & group_valid[:, None],
                other=0,
            ).to(tl.float32)
            * old_summary_scale[:, None]
        )
    else:
        old_even_sum = tl.load(
            sum_base + even_dimension,
            mask=has_old_tokens & group_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        old_odd_sum = tl.load(
            sum_base + odd_dimension,
            mask=has_old_tokens & group_valid[:, None],
            other=0.0,
        ).to(tl.float32)
    old_scale = tl.load(
        scales + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // 4) + groups,
        mask=has_old_tokens & group_valid,
        other=0.0,
    ).to(tl.float32)
    old_even = (
        old_even_sum[None, :, :] * old_inverse_count
        + old_even_code * old_scale[None, :, None]
    )
    old_odd = (
        old_odd_sum[None, :, :] * old_inverse_count
        + old_odd_code * old_scale[None, :, None]
    )

    source_index = leaf_index - leaf_offset
    new_token = valid_token & ~old_token
    valid_source = new_token & (source_index >= 0) & (source_index < SOURCE_TOKEN_COUNT)
    source_base = (
        source
        + batch * SOURCE_BATCH_STRIDE
        + kv_head * SOURCE_HEAD_STRIDE
        + source_index[:, None, None] * SOURCE_TOKEN_STRIDE
    )
    new_even = tl.load(
        source_base + even_dimension[None, :, :],
        mask=valid_source[:, None, None] & group_valid[None, :, None],
        other=0.0,
    ).to(tl.float32)
    new_odd = tl.load(
        source_base + odd_dimension[None, :, :],
        mask=valid_source[:, None, None] & group_valid[None, :, None],
        other=0.0,
    ).to(tl.float32)
    even = tl.where(old_token[:, None, None], old_even, new_even)
    odd = tl.where(old_token[:, None, None], old_odd, new_odd)
    new_even_sum = old_even_sum + tl.sum(
        tl.where(valid_source[:, None, None], new_even, 0.0), axis=0
    )
    new_odd_sum = old_odd_sum + tl.sum(
        tl.where(valid_source[:, None, None], new_odd, 0.0), axis=0
    )
    if QUANTIZED_SUMMARIES:
        new_summary_scale = tl.maximum(
            tl.maximum(
                tl.max(tl.abs(new_even_sum), axis=1),
                tl.max(tl.abs(new_odd_sum), axis=1),
            )
            / 127.0,
            1.0e-8,
        )
        new_even_code_float = tl.maximum(
            tl.minimum(
                tl.floor(new_even_sum / new_summary_scale[:, None] + 0.5),
                127.0,
            ),
            -127.0,
        )
        new_odd_code_float = tl.maximum(
            tl.minimum(
                tl.floor(new_odd_sum / new_summary_scale[:, None] + 0.5),
                127.0,
            ),
            -127.0,
        )
        if OPTIMIZE_SUMMARY_SCALE:
            denominator = tl.sum(
                new_even_code_float * new_even_code_float, axis=1
            ) + tl.sum(new_odd_code_float * new_odd_code_float, axis=1)
            numerator = tl.sum(new_even_sum * new_even_code_float, axis=1) + tl.sum(
                new_odd_sum * new_odd_code_float, axis=1
            )
            new_summary_scale = tl.where(
                denominator > 0.0,
                tl.maximum(numerator / denominator, 1.0e-8),
                new_summary_scale,
            )
            new_even_code_float = tl.maximum(
                tl.minimum(
                    tl.floor(new_even_sum / new_summary_scale[:, None] + 0.5),
                    127.0,
                ),
                -127.0,
            )
            new_odd_code_float = tl.maximum(
                tl.minimum(
                    tl.floor(new_odd_sum / new_summary_scale[:, None] + 0.5),
                    127.0,
                ),
                -127.0,
            )
            denominator = tl.sum(
                new_even_code_float * new_even_code_float, axis=1
            ) + tl.sum(new_odd_code_float * new_odd_code_float, axis=1)
            numerator = tl.sum(new_even_sum * new_even_code_float, axis=1) + tl.sum(
                new_odd_sum * new_odd_code_float, axis=1
            )
            new_summary_scale = tl.where(
                denominator > 0.0,
                tl.maximum(numerator / denominator, 1.0e-8),
                new_summary_scale,
            )
        tl.store(
            quantized_sum_base + even_dimension,
            new_even_code_float.to(tl.int8),
            mask=refresh & group_valid[:, None],
        )
        tl.store(
            quantized_sum_base + odd_dimension,
            new_odd_code_float.to(tl.int8),
            mask=refresh & group_valid[:, None],
        )
        tl.store(
            page_sum_scales
            + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // 4)
            + groups,
            new_summary_scale,
            mask=refresh & group_valid,
        )
    else:
        tl.store(
            sum_base + even_dimension,
            new_even_sum,
            mask=refresh & group_valid[:, None],
        )
        tl.store(
            sum_base + odd_dimension,
            new_odd_sum,
            mask=refresh & group_valid[:, None],
        )

    inverse_count = 1.0 / tl.maximum(new_count.to(tl.float32), 1.0)
    even_residual = even - new_even_sum[None, :, :] * inverse_count
    odd_residual = odd - new_odd_sum[None, :, :] * inverse_count
    even_max = tl.max(
        tl.max(tl.where(valid_even, tl.abs(even_residual), 0.0), axis=2),
        axis=0,
    )
    odd_max = tl.max(
        tl.max(tl.where(valid_odd, tl.abs(odd_residual), 0.0), axis=2),
        axis=0,
    )
    scale = tl.maximum(tl.maximum(even_max, odd_max) / 7.0, 1.0e-8)
    even_code_float = tl.maximum(
        tl.minimum(tl.floor(even_residual / scale[None, :, None] + 0.5), 7.0),
        -7.0,
    )
    odd_code_float = tl.maximum(
        tl.minimum(tl.floor(odd_residual / scale[None, :, None] + 0.5), 7.0),
        -7.0,
    )
    if OPTIMIZE_LEAF_SCALE:
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
        even_code_float = tl.maximum(
            tl.minimum(tl.floor(even_residual / scale[None, :, None] + 0.5), 7.0),
            -7.0,
        )
        odd_code_float = tl.maximum(
            tl.minimum(tl.floor(odd_residual / scale[None, :, None] + 0.5), 7.0),
            -7.0,
        )
        denominator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_code_float * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        denominator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_code_float * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator = tl.sum(
            tl.sum(
                tl.where(valid_even, even_residual * even_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        numerator += tl.sum(
            tl.sum(
                tl.where(valid_odd, odd_residual * odd_code_float, 0.0),
                axis=2,
            ),
            axis=0,
        )
        scale = tl.where(
            denominator > 0.0,
            tl.maximum(numerator / denominator, 1.0e-8),
            scale,
        )
    packed = (
        (even_code_float.to(tl.int32) + 8) | ((odd_code_float.to(tl.int32) + 8) << 4)
    ).to(tl.uint8)
    tl.store(destination_base, packed, mask=valid_even)
    tl.store(
        scales + (kv_row * PAGE_CAPACITY + page_id) * (DIMENSION_SIZE // 4) + groups,
        scale,
        mask=refresh & group_valid,
    )


@triton.jit(
    do_not_specialize=["leaf_offset", "TOKENS"],
    do_not_specialize_on_alignment=["leaf_offset", "TOKENS"],
)
def _append_quantized_virtual_pages_grouped_int4_kernel(
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
    leaf_offset,
    TOKENS,
    KV_HEADS: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
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
    group_begin = tl.program_id(1).to(tl.int64) * GROUPS_PER_PROGRAM
    kv_row = token_row // TOKENS
    batch = kv_row // KV_HEADS
    kv_head = kv_row - batch * KV_HEADS
    owner = tl.load(owners + token_row).to(tl.int64)
    ordinal = tl.load(ordinals + token_row).to(tl.int64)
    slot_length = tl.load(slot_lengths + kv_row * STATE_CAPACITY + owner).to(tl.int64)
    completes_page = ordinal % 16 == 15
    is_partial_tail = ordinal == slot_length - 1
    refresh = completes_page | is_partial_tail
    page_ordinal = ordinal // 16
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
    new_count = tl.where(completes_page, 16, ordinal % 16 + 1)
    token_offset = tl.arange(0, 16)
    valid_token = refresh & (token_offset < new_count)
    old_token = valid_token & (token_offset < old_count)
    leaf_index = tl.load(
        page_indices + (kv_row * PAGE_CAPACITY + page_id) * 16 + token_offset,
        mask=valid_token,
        other=0,
    ).to(tl.int64)
    _requantize_appended_virtual_page_tensor_grouped_int4(
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
        group_begin,
        PAGE_CAPACITY,
        HEAD_DIM,
        GROUPS_PER_PROGRAM,
        LEAF_CAPACITY,
        TOKENS,
        APPEND_K_BATCH_STRIDE,
        APPEND_K_HEAD_STRIDE,
        APPEND_K_TOKEN_STRIDE,
        QUANTIZED_SUMMARIES,
        OPTIMIZE_SUMMARY_SCALE,
        OPTIMIZE_LEAF_SCALE,
    )
    _requantize_appended_virtual_page_tensor_grouped_int4(
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
        group_begin,
        PAGE_CAPACITY,
        VALUE_DIM,
        GROUPS_PER_PROGRAM,
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
    page_sum_k: torch.Tensor | None,
    page_sum_v: torch.Tensor | None,
    page_counts: torch.Tensor | None,
    *,
    hash_probes: int = 8,
) -> None:
    """Publish BF16 leaves into their centroid-owned 16-token pages."""
    owners = owners.contiguous()
    batch, kv_heads, leaf_capacity, head_dim = leaf_k.shape
    tokens = int(owners.size(2))
    if owners.shape[:2] != (batch, kv_heads):
        raise ValueError("virtual page owners do not match flat K/V")
    if leaf_offset < 0 or leaf_offset + tokens > leaf_capacity:
        raise ValueError("virtual page append exceeds the flat K/V cache")
    if leaf_v.shape[:3] != leaf_k.shape[:3]:
        raise ValueError("flat K/V cache shapes do not match")
    if leaf_k.dtype not in (torch.float16, torch.bfloat16) or leaf_v.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise TypeError("the LoD release stores unquantized leaves in FP16/BF16")
    if tuple(page_indices.shape[:2]) != (batch, kv_heads):
        raise ValueError("virtual page index rows do not match flat K/V")
    if int(page_indices.size(3)) != 16:
        raise ValueError("the LoD release requires 16-token pages")
    if slot_lengths.dtype != torch.int32 or next_page.dtype != torch.int32:
        raise TypeError("Triton page counters must use int32")
    summaries = (page_sum_k, page_sum_v, page_counts)
    maintain_summaries = any(summary is not None for summary in summaries)
    if maintain_summaries and not all(
        isinstance(summary, torch.Tensor) for summary in summaries
    ):
        raise ValueError("page summary K, V, and counts must be supplied together")
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
        page_size=16,
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
        PAGE_SIZE=16,
        num_warps=1,
    )
    if maintain_summaries:
        assert isinstance(page_sum_k, torch.Tensor)
        assert isinstance(page_sum_v, torch.Tensor)
        assert isinstance(page_counts, torch.Tensor)
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
            PAGE_SIZE=16,
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
            INT8_STORAGE=False,
            UPDATE_KEY=True,
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
    quant_group_size: int = 4,
    quant_token_group_size: int = 16,
    quant_bits: int = 4,
    optimize_scale: bool = False,
) -> None:
    """Quantize every populated semantic page with residual INT4."""
    batch, kv_heads, _, head_dim = leaf_k.shape
    value_dim = int(leaf_v.size(-1))
    if leaf_v.shape[:3] != leaf_k.shape[:3]:
        raise ValueError("flat K/V cache shapes do not match")
    if (quant_bits, quant_group_size, quant_token_group_size) != (4, 4, 16):
        raise ValueError("the LoD release uses INT4 groups of 4 channels x 16 tokens")
    if head_dim % quant_group_size or value_dim % quant_group_size:
        raise ValueError("virtual quantization group size must divide K/V dimensions")
    if tuple(page_indices.shape[:2]) != (batch, kv_heads):
        raise ValueError("virtual page index rows do not match flat K/V")
    leaf_capacity = int(quantized_leaf_k.size(2))
    if int(leaf_k.size(2)) > leaf_capacity:
        raise ValueError("quantized virtual cache is smaller than its BF16 source")
    key_width = head_dim // 2
    value_width = value_dim // 2
    if (
        int(quantized_leaf_k.size(-1)) != key_width
        or int(quantized_leaf_v.size(-1)) != value_width
        or quantized_leaf_k.dtype != torch.uint8
        or quantized_leaf_v.dtype != torch.uint8
    ):
        raise ValueError("quantized virtual cache must use packed UINT8 INT4 codes")
    page_capacity = int(page_indices.size(2))
    page_size = int(page_indices.size(3))
    if page_size != 16:
        raise ValueError("the LoD release requires 16-token pages")
    channel_group_count = max(
        head_dim // quant_group_size,
        value_dim // quant_group_size,
    )
    expected_k_scales = (
        batch,
        kv_heads,
        page_capacity,
        head_dim // quant_group_size,
    )
    expected_v_scales = expected_k_scales[:-1] + (value_dim // quant_group_size,)
    if tuple(page_k_scales.shape) != expected_k_scales:
        raise ValueError("virtual K scale layout does not match token groups")
    if tuple(page_v_scales.shape) != expected_v_scales:
        raise ValueError("virtual V scale layout does not match token groups")
    groups_per_program = 4
    _quantize_all_virtual_pages_grouped_int4_kernel[
        (
            batch * kv_heads * page_capacity,
            triton.cdiv(channel_group_count, groups_per_program),
        )
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
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        GROUPS_PER_PROGRAM=groups_per_program,
        CHANNEL_GROUP_COUNT=channel_group_count,
        LEAF_CAPACITY=leaf_capacity,
        LEAF_K_BATCH_STRIDE=int(leaf_k.stride(0)),
        LEAF_K_HEAD_STRIDE=int(leaf_k.stride(1)),
        LEAF_K_TOKEN_STRIDE=int(leaf_k.stride(2)),
        LEAF_V_BATCH_STRIDE=int(leaf_v.stride(0)),
        LEAF_V_HEAD_STRIDE=int(leaf_v.stride(1)),
        LEAF_V_TOKEN_STRIDE=int(leaf_v.stride(2)),
        OPTIMIZE_SCALE=optimize_scale,
        num_warps=1,
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
    quant_group_size: int = 4,
    quant_token_group_size: int = 16,
    quant_bits: int = 4,
    quantized_page_sum_k: torch.Tensor | None = None,
    quantized_page_sum_v: torch.Tensor | None = None,
    page_sum_k_scales: torch.Tensor | None = None,
    page_sum_v_scales: torch.Tensor | None = None,
    optimize_summary_scale: bool = False,
    optimize_leaf_scale: bool = False,
) -> None:
    """Append decode leaves by requantizing only changed residual-INT4 pages."""
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
    if (quant_bits, quant_group_size, quant_token_group_size) != (4, 4, 16):
        raise ValueError("the LoD release uses INT4 groups of 4 channels x 16 tokens")
    if head_dim % quant_group_size or value_dim % quant_group_size:
        raise ValueError("virtual quantization group size must divide K/V dimensions")
    page_size = int(page_indices.size(3))
    if page_size != 16:
        raise ValueError("the LoD release requires 16-token pages")
    expected_k_scales = (
        batch,
        kv_heads,
        int(page_indices.size(2)),
        head_dim // quant_group_size,
    )
    expected_v_scales = expected_k_scales[:-1] + (value_dim // quant_group_size,)
    if tuple(page_k_scales.shape) != expected_k_scales:
        raise ValueError("quantized append K scales do not match token groups")
    if tuple(page_v_scales.shape) != expected_v_scales:
        raise ValueError("quantized append V scales do not match token groups")
    key_width = head_dim // 2
    value_width = value_dim // 2
    if (
        int(quantized_leaf_k.size(-1)) != key_width
        or int(quantized_leaf_v.size(-1)) != value_width
        or quantized_leaf_k.dtype != torch.uint8
        or quantized_leaf_v.dtype != torch.uint8
    ):
        raise ValueError("quantized append must use packed UINT8 INT4 codes")
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
    groups_per_program = 4
    common_args = (
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
    )
    common_meta = {
        "TOKENS": tokens,
        "KV_HEADS": kv_heads,
        "STATE_CAPACITY": int(slot_pages.size(2)),
        "INLINE_PAGES_PER_SLOT": int(slot_pages.size(3)),
        "PAGE_CAPACITY": int(page_indices.size(2)),
        "HASH_CAPACITY": int(overflow_page_values.size(2)),
        "HASH_PROBES": hash_probes,
        "HEAD_DIM": head_dim,
        "VALUE_DIM": value_dim,
        "LEAF_CAPACITY": leaf_capacity,
        "APPEND_K_BATCH_STRIDE": int(append_k.stride(0)),
        "APPEND_K_HEAD_STRIDE": int(append_k.stride(1)),
        "APPEND_K_TOKEN_STRIDE": int(append_k.stride(2)),
        "APPEND_V_BATCH_STRIDE": int(append_v.stride(0)),
        "APPEND_V_HEAD_STRIDE": int(append_v.stride(1)),
        "APPEND_V_TOKEN_STRIDE": int(append_v.stride(2)),
        "QUANTIZED_SUMMARIES": quantized_summaries,
        "OPTIMIZE_SUMMARY_SCALE": optimize_summary_scale,
        "OPTIMIZE_LEAF_SCALE": optimize_leaf_scale,
        "num_warps": 1,
    }
    _append_quantized_virtual_pages_grouped_int4_kernel[
        (token_rows, triton.cdiv(group_count, groups_per_program))
    ](
        *common_args,
        leaf_offset,
        GROUPS_PER_PROGRAM=groups_per_program,
        **common_meta,
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
