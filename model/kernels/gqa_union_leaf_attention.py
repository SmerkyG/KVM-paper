# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode attention over the GQA-wide union of routed centroid leaves."""

from __future__ import annotations

from functools import cache
import math
from pathlib import Path

import torch
import triton
import triton.language as tl


@cache
def _require_wide_head_pa_v1(head_dim: int) -> None:
    if head_dim <= 256:
        return
    from csrc.cpp_itfs.utils import AITER_CORE_DIR

    header = Path(AITER_CORE_DIR) / "csrc/cpp_itfs/pa/pa_kernels.cuh"
    try:
        source = header.read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect AITER PA-v1 source at {header}") from exc
    wide_value_layout = (
        "constexpr int V_SHARED_DEPTH = MAX(4, HEAD_SIZE / 16 / NWARPS);"
    )
    if wide_value_layout not in source:
        raise RuntimeError(
            "AITER PA-v1 needs the LOD wide-head patch for head dimension "
            f"{head_dim}; apply integrations/vllm_lod/patches/"
            "aiter-pa-v1-head-dim-512.patch to the AITER source checkout"
        )


@triton.jit(
    do_not_specialize=["local_begin", "local_len"],
    do_not_specialize_on_alignment=["local_begin", "local_len"],
)
def _compute_gqa_union_metadata_kernel(
    top_slots,  # [B, Hq, 1, K]
    slot_offsets,  # [B, Hkv, C + 1]
    cache_indices,  # [B], logical batch row -> cache row
    local_begins,  # [cache B]
    local_lens,  # [cache B]
    new_k,  # [B, Hkv, 1, D]
    new_v,  # [B, Hkv, 1, D]
    local_k,  # [cache B, Hkv, local capacity, D]
    local_v,  # [cache B, Hkv, local capacity, D]
    archive_k,  # [cache B, Hkv, leaf capacity, D]
    archive_v,  # [cache B, Hkv, leaf capacity, D]
    archive_begins,  # [cache B]
    union_leaf_indices,  # [B, Hlogical, leaf capacity]
    union_lengths,  # [B, Hlogical]
    actual_lengths,  # [B, Hlogical], may alias union_lengths
    unique_slots,  # [B, Hlogical, G*K]
    leaf_begins,  # [B, Hlogical, G*K]
    leaf_counts,  # [B, Hlogical, G*K]
    output_begins,  # [B, Hlogical, G*K]
    local_begin,
    local_len,
    top_stride_b: tl.constexpr,
    top_stride_h: tl.constexpr,
    top_stride_k: tl.constexpr,
    offset_stride_b: tl.constexpr,
    offset_stride_h: tl.constexpr,
    offset_stride_c: tl.constexpr,
    union_stride_b: tl.constexpr,
    union_stride_h: tl.constexpr,
    new_k_stride_b: tl.constexpr,
    new_k_stride_h: tl.constexpr,
    new_v_stride_b: tl.constexpr,
    new_v_stride_h: tl.constexpr,
    local_k_stride_b: tl.constexpr,
    local_k_stride_h: tl.constexpr,
    local_k_stride_n: tl.constexpr,
    local_v_stride_b: tl.constexpr,
    local_v_stride_h: tl.constexpr,
    local_v_stride_n: tl.constexpr,
    archive_k_stride_b: tl.constexpr,
    archive_k_stride_h: tl.constexpr,
    archive_k_stride_n: tl.constexpr,
    archive_v_stride_b: tl.constexpr,
    archive_v_stride_h: tl.constexpr,
    archive_v_stride_n: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_SIZE: tl.constexpr,
    GROUPS_PER_KV: tl.constexpr,
    LOGICAL_GROUPS: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    UNION_COUNT: tl.constexpr,
    UNION_BLOCK: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    MAX_SLOT_LEAVES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_CACHE_INDICES: tl.constexpr,
    USE_LOCAL_RANGES: tl.constexpr,
    CLAMP_EMPTY_LENGTHS: tl.constexpr,
    APPEND_NEW: tl.constexpr,
):
    batch_group = tl.program_id(0).to(tl.int64)
    batch = batch_group // LOGICAL_GROUPS
    logical_group = batch_group - batch * LOGICAL_GROUPS
    kv_head = logical_group // GROUPS_PER_KV
    union_subgroup = logical_group - kv_head * GROUPS_PER_KV
    cache_batch = batch
    if USE_CACHE_INDICES:
        cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    cache_batch_kv = cache_batch * KV_HEADS + kv_head
    row_local_len = local_len
    if USE_LOCAL_RANGES:
        row_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
    if APPEND_NEW:
        dim = tl.arange(0, BLOCK_D)
        write_new = (union_subgroup == 0) & (dim < HEAD_DIM)
        key = tl.load(
            new_k
            + batch * new_k_stride_b
            + kv_head * new_k_stride_h
            + dim,
            mask=write_new,
            other=0.0,
        )
        value = tl.load(
            new_v
            + batch * new_v_stride_b
            + kv_head * new_v_stride_h
            + dim,
            mask=write_new,
            other=0.0,
        )
        tl.store(
            local_k
            + cache_batch * local_k_stride_b
            + kv_head * local_k_stride_h
            + row_local_len * local_k_stride_n
            + dim,
            key,
            mask=write_new,
        )
        tl.store(
            local_v
            + cache_batch * local_v_stride_b
            + kv_head * local_v_stride_h
            + row_local_len * local_v_stride_n
            + dim,
            value,
            mask=write_new,
        )
        archive_begin = tl.load(archive_begins + cache_batch).to(tl.int64)
        archive_position = archive_begin + row_local_len
        tl.store(
            archive_k
            + cache_batch * archive_k_stride_b
            + kv_head * archive_k_stride_h
            + archive_position * archive_k_stride_n
            + dim,
            key,
            mask=write_new,
        )
        tl.store(
            archive_v
            + cache_batch * archive_v_stride_b
            + kv_head * archive_v_stride_h
            + archive_position * archive_v_stride_n
            + dim,
            value,
            mask=write_new,
        )
        row_local_len += 1
    union_lane = tl.arange(0, UNION_BLOCK)
    group_head = union_lane // ROUTE_COUNT
    route = union_lane - group_head * ROUTE_COUNT
    query_head = (
        kv_head * KV_GROUP_SIZE
        + union_subgroup * UNION_GROUP_SIZE
        + group_head
    )
    valid_lane = union_lane < UNION_COUNT
    slots = tl.load(
        top_slots
        + batch * top_stride_b
        + query_head * top_stride_h
        + route * top_stride_k,
        mask=valid_lane & (query_head < QUERY_HEADS),
        other=-1,
    ).to(tl.int64)
    valid_slot = valid_lane & (slots >= 0) & (slots < STATE_CAPACITY)
    safe_slot = tl.where(valid_slot, slots, 0)
    earlier = union_lane[None, :] < union_lane[:, None]
    same_slot = slots[:, None] == slots[None, :]
    duplicate = tl.sum(
        tl.where(earlier & same_slot & valid_slot[None, :], 1, 0), axis=1
    ) > 0
    unique = valid_slot & ~duplicate
    offset_base = cache_batch * offset_stride_b + kv_head * offset_stride_h
    leaf_begin = tl.load(
        slot_offsets + offset_base + safe_slot * offset_stride_c,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    leaf_end = tl.load(
        slot_offsets + offset_base + (safe_slot + 1) * offset_stride_c,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    slot_leaf_count = leaf_end - leaf_begin
    if MAX_SLOT_LEAVES > 0:
        unique &= slot_leaf_count <= MAX_SLOT_LEAVES
    leaf_count = tl.where(unique, slot_leaf_count, 0)
    # A logical route union can include overlapping posting lists from stale
    # metadata while the cache is crossing an update boundary.  Keep both the
    # physical writes and the length consumed by PA-v1 within this row.
    raw_prefix = tl.cumsum(leaf_count, axis=0) - leaf_count
    remaining = tl.maximum(LEAF_CAPACITY - row_local_len - raw_prefix, 0)
    leaf_count = tl.minimum(leaf_count, remaining)
    leaf_prefix = tl.cumsum(leaf_count, axis=0) - leaf_count
    total_leaves = tl.sum(leaf_count, axis=0)
    actual_length = total_leaves + row_local_len
    if CLAMP_EMPTY_LENGTHS:
        # PA-v1 assumes every sequence has at least one block.  Preserve the
        # true zero length for LSE masking, but give AITER one harmless leaf so
        # CUDA-graph dummy rows never dereference an undefined block-table
        # entry.  The final reducer ignores this branch when actual_length=0.
        tl.store(actual_lengths + batch_group, actual_length)
        tl.store(
            union_lengths + batch_group,
            tl.maximum(actual_length, 1),
        )
        tl.store(
            union_leaf_indices
            + batch * union_stride_b
            + logical_group * union_stride_h,
            cache_batch_kv.to(tl.int32) * LEAF_CAPACITY,
            mask=actual_length == 0,
        )
    else:
        tl.store(union_lengths + batch_group, actual_length)

    metadata_offset = batch_group * UNION_COUNT + union_lane
    tl.store(
        unique_slots + metadata_offset,
        tl.where(unique, slots, -1).to(tl.int32),
        mask=valid_lane,
    )
    tl.store(leaf_begins + metadata_offset, leaf_begin, mask=valid_lane)
    tl.store(leaf_counts + metadata_offset, leaf_count, mask=valid_lane)
    tl.store(output_begins + metadata_offset, leaf_prefix, mask=valid_lane)


@triton.jit(
    do_not_specialize=["local_begin", "local_len"],
    do_not_specialize_on_alignment=["local_begin", "local_len"],
)
def _copy_gqa_union_leaves_kernel(
    packed_leaf_indices,  # [B, Hkv, leaf capacity]
    cache_indices,  # [B], logical batch row -> cache row
    local_begins,  # [cache B]
    local_lens,  # [cache B]
    union_leaf_indices,  # [B, Hlogical, leaf capacity]
    union_top_slots,  # [B, Hq, 1, G*K]
    unique_slots,  # [B, Hlogical, G*K]
    leaf_begins,  # [B, Hlogical, G*K]
    leaf_counts,  # [B, Hlogical, G*K]
    output_begins,  # [B, Hlogical, G*K]
    union_lengths,  # [B, Hlogical]
    local_begin,
    local_len,
    packed_stride_b: tl.constexpr,
    packed_stride_h: tl.constexpr,
    packed_stride_n: tl.constexpr,
    union_stride_b: tl.constexpr,
    union_stride_h: tl.constexpr,
    union_stride_n: tl.constexpr,
    route_stride_b: tl.constexpr,
    route_stride_h: tl.constexpr,
    route_stride_k: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_BLOCK: tl.constexpr,
    GROUPS_PER_KV: tl.constexpr,
    LOGICAL_GROUPS: tl.constexpr,
    UNION_COUNT: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    GLOBAL_PAGE_INDICES: tl.constexpr,
    USE_CACHE_INDICES: tl.constexpr,
    USE_LOCAL_RANGES: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    APPEND_NEW: tl.constexpr,
):
    batch_group = tl.program_id(0).to(tl.int64)
    route_lane = tl.program_id(1).to(tl.int64)
    batch = batch_group // LOGICAL_GROUPS
    logical_group = batch_group - batch * LOGICAL_GROUPS
    kv_head = logical_group // GROUPS_PER_KV
    union_subgroup = logical_group - kv_head * GROUPS_PER_KV
    cache_batch = batch
    if USE_CACHE_INDICES:
        cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    cache_batch_kv = cache_batch * KV_HEADS + kv_head
    metadata_offset = batch_group * UNION_COUNT + route_lane
    selected_slot = tl.load(unique_slots + metadata_offset).to(tl.int64)
    selected_unique = selected_slot >= 0
    selected_begin = tl.load(leaf_begins + metadata_offset).to(tl.int32)
    selected_count = tl.load(leaf_counts + metadata_offset).to(tl.int32)
    output_begin = tl.load(output_begins + metadata_offset).to(tl.int32)

    # Triton block dimensions must be powers of two. The logical GQA group
    # need not be (Qwen3.8 uses 6 and OLMo-3 uses 5), so pad and mask lanes.
    output_group = tl.arange(0, UNION_GROUP_BLOCK)
    output_group_valid = output_group < UNION_GROUP_SIZE
    output_head = (
        kv_head * KV_GROUP_SIZE
        + union_subgroup * UNION_GROUP_SIZE
        + output_group
    )
    tl.store(
        union_top_slots
        + batch * route_stride_b
        + output_head * route_stride_h
        + route_lane * route_stride_k,
        tl.where(selected_unique, selected_slot, -1),
        mask=(
            output_group_valid
            & (output_head < QUERY_HEADS)
            & (route_lane < UNION_COUNT)
        ),
    )

    copy_lane = tl.arange(0, COPY_BLOCK)
    for copy_begin in tl.range(0, selected_count, COPY_BLOCK, num_stages=1):
        copy_offset = copy_begin + copy_lane
        valid_copy = selected_unique & (copy_offset < selected_count)
        leaf_index = tl.load(
            packed_leaf_indices
            + cache_batch * packed_stride_b
            + kv_head * packed_stride_h
            + (selected_begin + copy_offset) * packed_stride_n,
            mask=valid_copy,
            other=-1,
        ).to(tl.int32)
        if GLOBAL_PAGE_INDICES:
            leaf_index += cache_batch_kv.to(tl.int32) * LEAF_CAPACITY
        output_position = output_begin + copy_offset
        tl.store(
            union_leaf_indices
            + batch * union_stride_b
            + logical_group * union_stride_h
            + output_position * union_stride_n,
            leaf_index,
            mask=valid_copy & (output_position < LEAF_CAPACITY),
        )

    row_local_begin = local_begin
    row_local_len = local_len
    if USE_LOCAL_RANGES:
        row_local_begin = tl.load(local_begins + cache_batch).to(tl.int32)
        row_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
    if APPEND_NEW:
        row_local_len += 1

    # Virtual leaf storage is indexed by original sequence position, so the
    # exact local window can share this same AITER call without copying K/V.
    # One route lane appends its indices after the deduplicated remote leaves.
    for copy_begin in tl.static_range(0, LOCAL_CAPACITY, COPY_BLOCK):
        copy_offset = copy_begin + copy_lane
        total_leaves = tl.load(union_lengths + batch_group) - row_local_len
        output_position = total_leaves + copy_offset
        valid_copy = (
            (route_lane == 0)
            & (copy_offset < row_local_len)
            & (output_position < LEAF_CAPACITY)
        )
        leaf_index = (
            cache_batch_kv.to(tl.int32) * LEAF_CAPACITY
            + row_local_begin
            + copy_offset
        )
        tl.store(
            union_leaf_indices
            + batch * union_stride_b
            + logical_group * union_stride_h
            + output_position * union_stride_n,
            leaf_index,
            mask=valid_copy,
        )


@triton.jit(
    do_not_specialize=["local_begin", "local_len"],
    do_not_specialize_on_alignment=["local_begin", "local_len"],
)
def _pack_gqa_union_leaves_kernel(
    top_slots,  # [B, Hq, 1, K]
    slot_offsets,  # [cache B, Hkv, C + 1]
    packed_leaf_indices,  # [cache B, Hkv, leaf capacity]
    cache_indices,  # [B]
    local_begins,  # [cache B]
    local_lens,  # [cache B]
    union_leaf_indices,  # [B, Hlogical, leaf capacity]
    union_lengths,  # [B, Hlogical]
    actual_lengths,  # [B, Hlogical], may alias union_lengths
    union_top_slots,  # [B, Hq, 1, G*K]
    local_begin,
    local_len,
    top_stride_b: tl.constexpr,
    top_stride_h: tl.constexpr,
    top_stride_k: tl.constexpr,
    offset_stride_b: tl.constexpr,
    offset_stride_h: tl.constexpr,
    offset_stride_c: tl.constexpr,
    packed_stride_b: tl.constexpr,
    packed_stride_h: tl.constexpr,
    packed_stride_n: tl.constexpr,
    union_stride_b: tl.constexpr,
    union_stride_h: tl.constexpr,
    union_stride_n: tl.constexpr,
    route_stride_b: tl.constexpr,
    route_stride_h: tl.constexpr,
    route_stride_k: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_BLOCK: tl.constexpr,
    GROUPS_PER_KV: tl.constexpr,
    LOGICAL_GROUPS: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    UNION_COUNT: tl.constexpr,
    UNION_BLOCK: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    MAX_SLOT_LEAVES: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
    GLOBAL_PAGE_INDICES: tl.constexpr,
    USE_CACHE_INDICES: tl.constexpr,
    USE_LOCAL_RANGES: tl.constexpr,
    LOCAL_CAPACITY: tl.constexpr,
    CLAMP_EMPTY_LENGTHS: tl.constexpr,
):
    """Deduplicate routes, prefix their leaves, and copy them in one launch.

    The small route union is recomputed independently by each route program.
    This trades a few cached metadata reads for eliminating a launch and four
    intermediate metadata arrays from the decode critical path.
    """
    batch_group = tl.program_id(0).to(tl.int64)
    route_lane = tl.program_id(1).to(tl.int64)
    batch = batch_group // LOGICAL_GROUPS
    logical_group = batch_group - batch * LOGICAL_GROUPS
    kv_head = logical_group // GROUPS_PER_KV
    union_subgroup = logical_group - kv_head * GROUPS_PER_KV
    cache_batch = batch
    if USE_CACHE_INDICES:
        cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    cache_batch_kv = cache_batch * KV_HEADS + kv_head

    union_lane = tl.arange(0, UNION_BLOCK)
    group_head = union_lane // ROUTE_COUNT
    route = union_lane - group_head * ROUTE_COUNT
    query_head = (
        kv_head * KV_GROUP_SIZE
        + union_subgroup * UNION_GROUP_SIZE
        + group_head
    )
    valid_lane = union_lane < UNION_COUNT
    slots = tl.load(
        top_slots
        + batch * top_stride_b
        + query_head * top_stride_h
        + route * top_stride_k,
        mask=valid_lane & (query_head < QUERY_HEADS),
        other=-1,
    ).to(tl.int64)
    valid_slot = valid_lane & (slots >= 0) & (slots < STATE_CAPACITY)
    safe_slot = tl.where(valid_slot, slots, 0)
    earlier = union_lane[None, :] < union_lane[:, None]
    duplicate = tl.sum(
        tl.where(
            earlier
            & (slots[:, None] == slots[None, :])
            & valid_slot[None, :],
            1,
            0,
        ),
        axis=1,
    ) > 0
    unique = valid_slot & ~duplicate
    offset_base = cache_batch * offset_stride_b + kv_head * offset_stride_h
    leaf_begin = tl.load(
        slot_offsets + offset_base + safe_slot * offset_stride_c,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    leaf_end = tl.load(
        slot_offsets + offset_base + (safe_slot + 1) * offset_stride_c,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    slot_leaf_count = leaf_end - leaf_begin
    if MAX_SLOT_LEAVES > 0:
        unique &= slot_leaf_count <= MAX_SLOT_LEAVES
    row_local_begin = local_begin
    row_local_len = local_len
    if USE_LOCAL_RANGES:
        row_local_begin = tl.load(local_begins + cache_batch).to(tl.int32)
        row_local_len = tl.load(local_lens + cache_batch).to(tl.int32)
    leaf_count = tl.where(unique, slot_leaf_count, 0)
    raw_prefix = tl.cumsum(leaf_count, axis=0) - leaf_count
    remaining = tl.maximum(LEAF_CAPACITY - row_local_len - raw_prefix, 0)
    leaf_count = tl.minimum(leaf_count, remaining)
    leaf_prefix = tl.cumsum(leaf_count, axis=0) - leaf_count
    total_leaves = tl.sum(leaf_count, axis=0)
    actual_length = total_leaves + row_local_len
    tl.store(
        actual_lengths + batch_group,
        actual_length,
        mask=route_lane == 0,
    )
    stored_length = actual_length
    if CLAMP_EMPTY_LENGTHS:
        stored_length = tl.maximum(stored_length, 1)
    tl.store(
        union_lengths + batch_group,
        stored_length,
        mask=route_lane == 0,
    )
    if CLAMP_EMPTY_LENGTHS:
        tl.store(
            union_leaf_indices
            + batch * union_stride_b
            + logical_group * union_stride_h,
            cache_batch_kv.to(tl.int32) * LEAF_CAPACITY,
            mask=(route_lane == 0) & (actual_length == 0),
        )

    selected = union_lane == route_lane
    selected_slot = tl.sum(tl.where(selected, slots, 0), axis=0).to(tl.int64)
    selected_unique = tl.sum(tl.where(selected, unique, False), axis=0) > 0
    selected_begin = tl.sum(tl.where(selected, leaf_begin, 0), axis=0).to(tl.int32)
    selected_count = tl.sum(tl.where(selected, leaf_count, 0), axis=0).to(tl.int32)
    output_begin = tl.sum(tl.where(selected, leaf_prefix, 0), axis=0).to(tl.int32)

    output_group = tl.arange(0, UNION_GROUP_BLOCK)
    output_group_valid = output_group < UNION_GROUP_SIZE
    output_head = (
        kv_head * KV_GROUP_SIZE
        + union_subgroup * UNION_GROUP_SIZE
        + output_group
    )
    tl.store(
        union_top_slots
        + batch * route_stride_b
        + output_head * route_stride_h
        + route_lane * route_stride_k,
        tl.where(selected_unique, selected_slot, -1),
        mask=(
            output_group_valid
            & (output_head < QUERY_HEADS)
            & (route_lane < UNION_COUNT)
        ),
    )

    copy_lane = tl.arange(0, COPY_BLOCK)
    for copy_begin in tl.range(0, selected_count, COPY_BLOCK, num_stages=1):
        copy_offset = copy_begin + copy_lane
        valid_copy = selected_unique & (copy_offset < selected_count)
        leaf_index = tl.load(
            packed_leaf_indices
            + cache_batch * packed_stride_b
            + kv_head * packed_stride_h
            + (selected_begin + copy_offset) * packed_stride_n,
            mask=valid_copy,
            other=-1,
        ).to(tl.int32)
        if GLOBAL_PAGE_INDICES:
            leaf_index += cache_batch_kv.to(tl.int32) * LEAF_CAPACITY
        output_position = output_begin + copy_offset
        tl.store(
            union_leaf_indices
            + batch * union_stride_b
            + logical_group * union_stride_h
            + output_position * union_stride_n,
            leaf_index,
            mask=valid_copy & (output_position < LEAF_CAPACITY),
        )

    for copy_begin in tl.static_range(0, LOCAL_CAPACITY, COPY_BLOCK):
        copy_offset = copy_begin + copy_lane
        output_position = total_leaves + copy_offset
        valid_copy = (
            (route_lane == 0)
            & (copy_offset < row_local_len)
            & (output_position < LEAF_CAPACITY)
        )
        leaf_index = (
            cache_batch_kv.to(tl.int32) * LEAF_CAPACITY
            + row_local_begin
            + copy_offset
        )
        tl.store(
            union_leaf_indices
            + batch * union_stride_b
            + logical_group * union_stride_h
            + output_position * union_stride_n,
            leaf_index,
            mask=valid_copy,
        )


@triton.jit
def _pack_query_routed_leaves_kernel(
    top_slots,
    slot_offsets,
    packed_leaf_indices,
    leaf_indices,
    lengths,
    top_stride_b: tl.constexpr,
    top_stride_h: tl.constexpr,
    top_stride_k: tl.constexpr,
    offset_stride_b: tl.constexpr,
    offset_stride_h: tl.constexpr,
    offset_stride_c: tl.constexpr,
    packed_stride_b: tl.constexpr,
    packed_stride_h: tl.constexpr,
    packed_stride_n: tl.constexpr,
    output_stride_b: tl.constexpr,
    output_stride_h: tl.constexpr,
    output_stride_n: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    COPY_BLOCK: tl.constexpr,
):
    query_row = tl.program_id(0).to(tl.int64)
    route_lane = tl.program_id(1).to(tl.int64)
    batch = query_row // QUERY_HEADS
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    batch_kv = batch * KV_HEADS + kv_head
    route = tl.arange(0, ROUTE_BLOCK)
    valid_route = route < ROUTE_COUNT
    slots = tl.load(
        top_slots
        + batch * top_stride_b
        + query_head * top_stride_h
        + route * top_stride_k,
        mask=valid_route,
        other=-1,
    ).to(tl.int64)
    valid_slot = valid_route & (slots >= 0) & (slots < STATE_CAPACITY)
    safe_slot = tl.where(valid_slot, slots, 0)
    offset_base = batch * offset_stride_b + kv_head * offset_stride_h
    leaf_begin = tl.load(
        slot_offsets + offset_base + safe_slot * offset_stride_c,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    leaf_end = tl.load(
        slot_offsets + offset_base + (safe_slot + 1) * offset_stride_c,
        mask=valid_slot,
        other=0,
    ).to(tl.int32)
    leaf_count = tl.where(valid_slot, leaf_end - leaf_begin, 0)
    leaf_prefix = tl.cumsum(leaf_count, axis=0) - leaf_count
    total_leaves = tl.sum(leaf_count, axis=0)
    tl.store(lengths + query_row, total_leaves, mask=route_lane == 0)

    selected_begin = tl.sum(
        tl.where(route == route_lane, leaf_begin, 0), axis=0
    ).to(tl.int32)
    selected_count = tl.sum(
        tl.where(route == route_lane, leaf_count, 0), axis=0
    ).to(tl.int32)
    output_begin = tl.sum(
        tl.where(route == route_lane, leaf_prefix, 0), axis=0
    ).to(tl.int32)
    selected_valid = tl.sum(
        tl.where(route == route_lane, valid_slot, False), axis=0
    ) > 0
    copy_lane = tl.arange(0, COPY_BLOCK)
    for copy_begin in tl.range(0, selected_count, COPY_BLOCK, num_stages=1):
        copy_offset = copy_begin + copy_lane
        valid_copy = selected_valid & (copy_offset < selected_count)
        leaf_index = tl.load(
            packed_leaf_indices
            + batch * packed_stride_b
            + kv_head * packed_stride_h
            + (selected_begin + copy_offset) * packed_stride_n,
            mask=valid_copy,
            other=-1,
        ).to(tl.int32)
        leaf_index += batch_kv.to(tl.int32) * LEAF_CAPACITY
        output_position = output_begin + copy_offset
        tl.store(
            leaf_indices
            + batch * output_stride_b
            + query_head * output_stride_h
            + output_position * output_stride_n,
            leaf_index,
            mask=valid_copy & (output_position < LEAF_CAPACITY),
        )


@triton.jit
def _gqa_union_indexed_attention_kernel(
    q,  # [B, Hq, 1, D]
    leaf_k,  # [B, Hkv, leaf capacity, D]
    leaf_v,  # [B, Hkv, leaf capacity, D]
    union_leaf_indices,  # [B, Hkv, leaf capacity]
    union_lengths,  # [B, Hkv]
    output,  # [B, Hq, 1, D]
    output_lse,  # [B, Hq, 1]
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_d: tl.constexpr,
    index_stride_b: tl.constexpr,
    index_stride_h: tl.constexpr,
    index_stride_n: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_h: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_SIZE: tl.constexpr,
    GROUPS_PER_KV: tl.constexpr,
    LOGICAL_GROUPS: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    TOTAL_LEAVES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    SINGLE_HEAD: tl.constexpr,
):
    batch_group = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, BLOCK_G)
    if SINGLE_HEAD:
        batch = batch_group // QUERY_HEADS
        query_head_base = batch_group - batch * QUERY_HEADS
        logical_group = query_head_base // UNION_GROUP_SIZE
        kv_head = query_head_base // KV_GROUP_SIZE
        query_head = query_head_base + group
    else:
        batch = batch_group // LOGICAL_GROUPS
        logical_group = batch_group - batch * LOGICAL_GROUPS
        kv_head = logical_group // GROUPS_PER_KV
        subgroup = logical_group - kv_head * GROUPS_PER_KV
        query_head = (
            kv_head * KV_GROUP_SIZE + subgroup * UNION_GROUP_SIZE + group
        )
    dim = tl.arange(0, BLOCK_D)
    key_lane = tl.arange(0, BLOCK_N)
    valid_group = group < UNION_GROUP_SIZE
    if SINGLE_HEAD:
        valid_group = group < 1
    valid_dim = dim < HEAD_DIM
    query = tl.load(
        q
        + batch * q_stride_b
        + query_head[:, None] * q_stride_h
        + dim[None, :] * q_stride_d,
        mask=valid_group[:, None] & valid_dim[None, :],
        other=0.0,
    )
    maximum = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_G,), tl.float32)
    accumulator = tl.zeros((BLOCK_G, BLOCK_D), tl.float32)
    leaf_count = tl.load(
        union_lengths
        + batch * LOGICAL_GROUPS
        + logical_group
    ).to(tl.int32)

    for key_begin in tl.range(0, leaf_count, BLOCK_N, num_stages=1):
        key_offset = key_begin + key_lane
        valid_key = key_offset < leaf_count
        leaf_index = tl.load(
            union_leaf_indices
            + batch * index_stride_b
            + logical_group * index_stride_h
            + key_offset * index_stride_n,
            mask=valid_key,
            other=0,
        ).to(tl.int64)
        valid_key &= (leaf_index >= 0) & (leaf_index < TOTAL_LEAVES)
        leaves_per_batch = KV_HEADS * LEAF_CAPACITY
        leaf_batch = leaf_index // leaves_per_batch
        leaf_offset = leaf_index - (leaf_index // LEAF_CAPACITY) * LEAF_CAPACITY
        key = tl.load(
            leaf_k
            + leaf_batch[:, None] * k_stride_b
            + kv_head * k_stride_h
            + leaf_offset[:, None] * k_stride_n
            + dim[None, :] * k_stride_d,
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key), out_dtype=tl.float32) * SCALE_LOG2
        scores = tl.where(
            valid_group[:, None] & valid_key[None, :],
            scores,
            -float("inf"),
        )
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probability = tl.math.exp2(scores - new_maximum[:, None])
        probability = tl.where(
            valid_group[:, None] & valid_key[None, :], probability, 0.0
        )
        value = tl.load(
            leaf_v
            + leaf_batch[:, None] * v_stride_b
            + kv_head * v_stride_h
            + leaf_offset[:, None] * v_stride_n
            + dim[None, :] * v_stride_d,
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(
            probability.to(value.dtype),
            value,
            out_dtype=tl.float32,
            acc=accumulator,
        )
        denominator = denominator * correction + tl.sum(probability, axis=1)
        maximum = new_maximum

    has_mass = denominator > 0.0
    normalized = tl.where(
        has_mass[:, None], accumulator / denominator[:, None], 0.0
    )
    natural_lse = tl.where(
        has_mass,
        (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
        -float("inf"),
    )
    tl.store(
        output
        + batch * out_stride_b
        + query_head[:, None] * out_stride_h
        + dim[None, :] * out_stride_d,
        normalized,
        mask=valid_group[:, None] & valid_dim[None, :],
    )
    tl.store(
        output_lse + batch * lse_stride_b + query_head * lse_stride_h,
        natural_lse,
        mask=valid_group,
    )


@triton.jit
def _gqa_union_indexed_attention_wide_kernel(
    q,
    leaf_k,
    leaf_v,
    union_leaf_indices,
    union_lengths,
    output,
    output_lse,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_n: tl.constexpr,
    v_stride_d: tl.constexpr,
    index_stride_b: tl.constexpr,
    index_stride_h: tl.constexpr,
    index_stride_n: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_h: tl.constexpr,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    UNION_GROUP_SIZE: tl.constexpr,
    LOGICAL_GROUPS: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    TOTAL_LEAVES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    """Register-bounded exact attention for heads wider than 256 channels."""
    query_row = tl.program_id(0).to(tl.int64)
    batch = query_row // QUERY_HEADS
    query_head = query_row - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    logical_group = query_head // UNION_GROUP_SIZE
    dim = tl.arange(0, BLOCK_D)
    valid_dim = dim < HEAD_DIM
    query = tl.load(
        q
        + batch * q_stride_b
        + query_head * q_stride_h
        + dim * q_stride_d,
        mask=valid_dim,
        other=0.0,
    )
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    leaf_count = tl.load(
        union_lengths
        + batch * LOGICAL_GROUPS
        + logical_group
    ).to(tl.int32)
    key_lane = tl.arange(0, BLOCK_N)

    for key_begin in tl.range(0, leaf_count, BLOCK_N, num_stages=1):
        key_offset = key_begin + key_lane
        valid_key = key_offset < leaf_count
        leaf_index = tl.load(
            union_leaf_indices
            + batch * index_stride_b
            + logical_group * index_stride_h
            + key_offset * index_stride_n,
            mask=valid_key,
            other=0,
        ).to(tl.int64)
        valid_key &= (leaf_index >= 0) & (leaf_index < TOTAL_LEAVES)
        leaf_batch = leaf_index // (KV_HEADS * LEAF_CAPACITY)
        leaf_offset = leaf_index - (leaf_index // LEAF_CAPACITY) * LEAF_CAPACITY
        key = tl.load(
            leaf_k
            + leaf_batch[:, None] * k_stride_b
            + kv_head * k_stride_h
            + leaf_offset[:, None] * k_stride_n
            + dim[None, :] * k_stride_d,
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        scores = tl.sum(
            key.to(tl.float32) * query[None, :].to(tl.float32), axis=1
        ) * SCALE_LOG2
        scores = tl.where(valid_key, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probability = tl.where(
            valid_key, tl.math.exp2(scores - new_maximum), 0.0
        )
        value = tl.load(
            leaf_v
            + leaf_batch[:, None] * v_stride_b
            + kv_head * v_stride_h
            + leaf_offset[:, None] * v_stride_n
            + dim[None, :] * v_stride_d,
            mask=valid_key[:, None] & valid_dim[None, :],
            other=0.0,
        )
        accumulator = (
            accumulator * correction
            + tl.sum(
                probability[:, None] * value.to(tl.float32), axis=0
            )
        )
        denominator = denominator * correction + tl.sum(probability, axis=0)
        maximum = new_maximum

    has_mass = denominator > 0.0
    tl.store(
        output
        + batch * out_stride_b
        + query_head * out_stride_h
        + dim * out_stride_d,
        tl.where(has_mass, accumulator / denominator, 0.0),
        mask=valid_dim,
    )
    tl.store(
        output_lse + batch * lse_stride_b + query_head * lse_stride_h,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )


def new_gqa_union_buffers(
    q: torch.Tensor,
    *,
    kv_heads: int,
    leaf_capacity: int,
    route_count: int,
    union_group_size: int | None = None,
) -> dict[str, torch.Tensor]:
    batch, query_heads, query_len, head_dim = q.shape
    if query_len != 1 or query_heads % kv_heads:
        raise ValueError("GQA-union buffers require one decode row and valid GQA")
    kv_group_size = query_heads // kv_heads
    if union_group_size is None:
        union_group_size = kv_group_size
    if union_group_size <= 0 or kv_group_size % union_group_size:
        raise ValueError("GQA union groups must evenly partition each KV group")
    logical_groups = query_heads // union_group_size
    union_count = union_group_size * route_count
    return {
        "leaf_indices": torch.empty(
            batch,
            logical_groups,
            leaf_capacity,
            dtype=torch.int32,
            device=q.device,
        ),
        "lengths": torch.empty(
            batch, logical_groups, dtype=torch.int32, device=q.device
        ),
        "unique_slots": torch.empty(
            batch,
            logical_groups,
            union_count,
            dtype=torch.int32,
            device=q.device,
        ),
        "leaf_begins": torch.empty(
            batch,
            logical_groups,
            union_count,
            dtype=torch.int32,
            device=q.device,
        ),
        "leaf_counts": torch.empty(
            batch,
            logical_groups,
            union_count,
            dtype=torch.int32,
            device=q.device,
        ),
        "output_begins": torch.empty(
            batch,
            logical_groups,
            union_count,
            dtype=torch.int32,
            device=q.device,
        ),
        "top_slots": torch.empty(
            batch,
            query_heads,
            1,
            union_count,
            dtype=torch.int64,
            device=q.device,
        ),
        "output": torch.empty_like(q),
        "lse": torch.empty(
            batch,
            query_heads,
            1,
            dtype=torch.float32,
            device=q.device,
        ),
    }


def new_gqa_union_aiter_buffers(
    q: torch.Tensor,
    *,
    kv_heads: int,
    leaf_capacity: int,
    route_count: int,
    union_group_size: int | None = None,
    partition_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Allocate fixed-address PA-v1 workspace and exposed partition stats."""
    buffers = new_gqa_union_buffers(
        q,
        kv_heads=kv_heads,
        leaf_capacity=leaf_capacity,
        route_count=route_count,
        union_group_size=union_group_size,
    )
    batch, query_heads, _, head_dim = q.shape
    kv_group_size = query_heads // kv_heads
    group_size = kv_group_size if union_group_size is None else union_group_size
    sequences = batch * (query_heads // group_size)
    partitions = math.ceil(leaf_capacity / partition_size)
    statistic_elements = sequences * group_size * partitions
    workspace_bytes = (
        2 * statistic_elements * 4
        + sequences
        * group_size
        * partitions
        * head_dim
        * q.element_size()
    )
    workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=q.device
    )
    statistics = workspace[: 2 * statistic_elements * 4].view(torch.float32)
    partial_output = workspace[2 * statistic_elements * 4 :].view(q.dtype)
    buffers.update(
        actual_lengths=torch.empty(
            batch,
            query_heads // group_size,
            dtype=torch.int32,
            device=q.device,
        ),
        workspace=workspace,
        exp_sums=statistics[:statistic_elements].view(
            sequences, group_size, partitions
        ),
        max_logits=statistics[statistic_elements:].view(
            sequences, group_size, partitions
        ),
        partial_output=partial_output.view(
            sequences, group_size, partitions, head_dim
        ),
        unit_scale=torch.ones(1, dtype=torch.float32, device=q.device),
    )
    return buffers


def new_query_routed_aiter_buffers(
    q: torch.Tensor,
    *,
    leaf_capacity: int,
    partition_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Allocate PA-v1 metadata for one physical leaf list per query head."""
    batch, query_heads, _, head_dim = q.shape
    sequences = batch * query_heads
    partitions = math.ceil(leaf_capacity / partition_size)
    statistic_elements = sequences * partitions
    workspace_bytes = (
        2 * statistic_elements * 4
        + sequences
        * partitions
        * head_dim
        * q.element_size()
    )
    workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=q.device
    )
    statistics = workspace[: 2 * statistic_elements * 4].view(torch.float32)
    return {
        "leaf_indices": torch.empty(
            batch,
            query_heads,
            leaf_capacity,
            dtype=torch.int32,
            device=q.device,
        ),
        "lengths": torch.empty(
            batch, query_heads, dtype=torch.int32, device=q.device
        ),
        "output": torch.empty_like(q),
        "workspace": workspace,
        "exp_sums": statistics[:statistic_elements].view(
            sequences, 1, partitions
        ),
        "max_logits": statistics[statistic_elements:].view(
            sequences, 1, partitions
        ),
        "unit_scale": torch.ones(1, dtype=torch.float32, device=q.device),
    }


def _pack_gqa_union_leaves(
    top_slots: torch.Tensor,
    slot_offsets: torch.Tensor,
    packed_leaf_indices: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    kv_group_size: int,
    union_group_size: int,
    cache_indices: torch.Tensor | None,
    local_begin: int,
    local_len: int,
    local_begins: torch.Tensor | None,
    local_lens: torch.Tensor | None,
    local_capacity: int,
    new_k: torch.Tensor | None = None,
    new_v: torch.Tensor | None = None,
    local_k: torch.Tensor | None = None,
    local_v: torch.Tensor | None = None,
    archive_k: torch.Tensor | None = None,
    archive_v: torch.Tensor | None = None,
    archive_begins: torch.Tensor | None = None,
    max_slot_leaves: int,
    global_page_indices: bool,
    clamp_empty_lengths: bool,
    fused_metadata: bool,
    waves_per_eu: int,
) -> None:
    batch, query_heads, _, route_count = top_slots.shape
    kv_heads = int(slot_offsets.size(1))
    leaf_capacity = int(packed_leaf_indices.size(2))
    groups_per_kv = kv_group_size // union_group_size
    logical_groups = query_heads // union_group_size
    union_count = union_group_size * route_count
    cache_index_arg = (
        buffers["lengths"] if cache_indices is None else cache_indices
    )
    use_local_ranges = local_begins is not None or local_lens is not None
    if use_local_ranges and (
        local_begins is None
        or local_lens is None
        or local_begins.ndim != 1
        or tuple(local_lens.shape) != tuple(local_begins.shape)
        or local_begins.device != top_slots.device
        or local_lens.device != top_slots.device
    ):
        raise ValueError("GQA local leaf ranges are incompatible")
    append_inputs = (
        new_k,
        new_v,
        local_k,
        local_v,
        archive_k,
        archive_v,
        archive_begins,
    )
    append_new = any(tensor is not None for tensor in append_inputs)
    if append_new and not all(
        isinstance(tensor, torch.Tensor) for tensor in append_inputs
    ):
        raise ValueError("fused GQA append requires every K/V cache tensor")
    if append_new and (fused_metadata or not use_local_ranges):
        raise ValueError(
            "fused GQA append requires split metadata and ranged local leaves"
        )
    if append_new:
        head_dim = int(new_k.size(-1))
        if (
            tuple(new_k.shape) != tuple(new_v.shape)
            or tuple(new_k.shape[:3]) != (batch, kv_heads, 1)
            or tuple(local_k.shape[:2]) != tuple(local_v.shape[:2])
            or tuple(local_k.shape[:2]) != (int(local_lens.numel()), kv_heads)
            or int(local_k.size(-1)) != head_dim
            or tuple(archive_k.shape) != tuple(archive_v.shape)
            or tuple(archive_k.shape[:2]) != tuple(local_k.shape[:2])
            or int(archive_k.size(-1)) != head_dim
            or tuple(archive_begins.shape) != tuple(local_lens.shape)
        ):
            raise ValueError("fused GQA append cache geometry is incompatible")
    else:
        head_dim = 1
        new_k = new_v = local_k = local_v = top_slots
        archive_k = archive_v = archive_begins = top_slots
    local_begin_arg = buffers["lengths"] if local_begins is None else local_begins
    local_len_arg = buffers["lengths"] if local_lens is None else local_lens
    if not fused_metadata:
        _compute_gqa_union_metadata_kernel[(batch * logical_groups,)](
            top_slots,
            slot_offsets,
            cache_index_arg,
            local_begin_arg,
            local_len_arg,
            new_k,
            new_v,
            local_k,
            local_v,
            archive_k,
            archive_v,
            archive_begins,
            buffers["leaf_indices"],
            buffers["lengths"],
            buffers.get("actual_lengths", buffers["lengths"]),
            buffers["unique_slots"],
            buffers["leaf_begins"],
            buffers["leaf_counts"],
            buffers["output_begins"],
            local_begin,
            local_len,
            top_slots.stride(0),
            top_slots.stride(1),
            top_slots.stride(3),
            slot_offsets.stride(0),
            slot_offsets.stride(1),
            slot_offsets.stride(2),
            buffers["leaf_indices"].stride(0),
            buffers["leaf_indices"].stride(1),
            new_k.stride(0) if append_new else 0,
            new_k.stride(1) if append_new else 0,
            new_v.stride(0) if append_new else 0,
            new_v.stride(1) if append_new else 0,
            local_k.stride(0) if append_new else 0,
            local_k.stride(1) if append_new else 0,
            local_k.stride(2) if append_new else 0,
            local_v.stride(0) if append_new else 0,
            local_v.stride(1) if append_new else 0,
            local_v.stride(2) if append_new else 0,
            archive_k.stride(0) if append_new else 0,
            archive_k.stride(1) if append_new else 0,
            archive_k.stride(2) if append_new else 0,
            archive_v.stride(0) if append_new else 0,
            archive_v.stride(1) if append_new else 0,
            archive_v.stride(2) if append_new else 0,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            UNION_GROUP_SIZE=union_group_size,
            LOGICAL_GROUPS=logical_groups,
            GROUPS_PER_KV=groups_per_kv,
            ROUTE_COUNT=route_count,
            UNION_COUNT=union_count,
            UNION_BLOCK=triton.next_power_of_2(union_count),
            STATE_CAPACITY=int(slot_offsets.size(2)) - 1,
            LEAF_CAPACITY=leaf_capacity,
            MAX_SLOT_LEAVES=max_slot_leaves,
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            USE_CACHE_INDICES=cache_indices is not None,
            USE_LOCAL_RANGES=use_local_ranges,
            CLAMP_EMPTY_LENGTHS=clamp_empty_lengths,
            APPEND_NEW=append_new,
            num_warps=4,
            waves_per_eu=waves_per_eu,
        )
        _copy_gqa_union_leaves_kernel[(batch * logical_groups, union_count)](
            packed_leaf_indices,
            cache_index_arg,
            local_begin_arg,
            local_len_arg,
            buffers["leaf_indices"],
            buffers["top_slots"],
            buffers["unique_slots"],
            buffers["leaf_begins"],
            buffers["leaf_counts"],
            buffers["output_begins"],
            buffers["lengths"],
            local_begin,
            local_len,
            packed_leaf_indices.stride(0),
            packed_leaf_indices.stride(1),
            packed_leaf_indices.stride(2),
            buffers["leaf_indices"].stride(0),
            buffers["leaf_indices"].stride(1),
            buffers["leaf_indices"].stride(2),
            buffers["top_slots"].stride(0),
            buffers["top_slots"].stride(1),
            buffers["top_slots"].stride(3),
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=kv_group_size,
            UNION_GROUP_SIZE=union_group_size,
            UNION_GROUP_BLOCK=triton.next_power_of_2(union_group_size),
            GROUPS_PER_KV=groups_per_kv,
            LOGICAL_GROUPS=logical_groups,
            UNION_COUNT=union_count,
            LEAF_CAPACITY=leaf_capacity,
            COPY_BLOCK=128,
            GLOBAL_PAGE_INDICES=global_page_indices,
            USE_CACHE_INDICES=cache_indices is not None,
            USE_LOCAL_RANGES=use_local_ranges,
            LOCAL_CAPACITY=local_capacity,
            APPEND_NEW=append_new,
            num_warps=4,
            waves_per_eu=waves_per_eu,
        )
        return
    _pack_gqa_union_leaves_kernel[(batch * logical_groups, union_count)](
        top_slots,
        slot_offsets,
        packed_leaf_indices,
        cache_index_arg,
        local_begin_arg,
        local_len_arg,
        buffers["leaf_indices"],
        buffers["lengths"],
        buffers.get("actual_lengths", buffers["lengths"]),
        buffers["top_slots"],
        local_begin,
        local_len,
        top_slots.stride(0),
        top_slots.stride(1),
        top_slots.stride(3),
        slot_offsets.stride(0),
        slot_offsets.stride(1),
        slot_offsets.stride(2),
        packed_leaf_indices.stride(0),
        packed_leaf_indices.stride(1),
        packed_leaf_indices.stride(2),
        buffers["leaf_indices"].stride(0),
        buffers["leaf_indices"].stride(1),
        buffers["leaf_indices"].stride(2),
        buffers["top_slots"].stride(0),
        buffers["top_slots"].stride(1),
        buffers["top_slots"].stride(3),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        UNION_GROUP_SIZE=union_group_size,
        UNION_GROUP_BLOCK=triton.next_power_of_2(union_group_size),
        LOGICAL_GROUPS=logical_groups,
        GROUPS_PER_KV=groups_per_kv,
        ROUTE_COUNT=route_count,
        UNION_COUNT=union_count,
        UNION_BLOCK=triton.next_power_of_2(union_count),
        STATE_CAPACITY=int(slot_offsets.size(2)) - 1,
        LEAF_CAPACITY=leaf_capacity,
        MAX_SLOT_LEAVES=max_slot_leaves,
        COPY_BLOCK=128,
        GLOBAL_PAGE_INDICES=global_page_indices,
        USE_CACHE_INDICES=cache_indices is not None,
        USE_LOCAL_RANGES=use_local_ranges,
        LOCAL_CAPACITY=local_capacity,
        CLAMP_EMPTY_LENGTHS=clamp_empty_lengths,
        num_warps=4,
        waves_per_eu=waves_per_eu,
    )


def query_routed_aiter_attention(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    top_slots: torch.Tensor,
    slot_offsets: torch.Tensor,
    packed_leaf_indices: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    buffers: dict[str, torch.Tensor] | None = None,
    partition_size: int = 256,
    waves_per_eu: int = 1,
    timing_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ]
    | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Run PA-v1 over each query head's routed slots without copying K/V."""
    if q.ndim != 4 or int(q.size(2)) != 1:
        raise ValueError("query-routed AITER attention supports decode only")
    if leaf_k.ndim == 5:
        leaf_k = leaf_k.flatten(2, 3)
        leaf_v = leaf_v.flatten(2, 3)
    batch, query_heads, _, head_dim = q.shape
    _require_wide_head_pa_v1(head_dim)
    cache_batch, kv_heads, leaf_capacity, key_dim = leaf_k.shape
    route_count = int(top_slots.size(-1))
    if (
        cache_batch != batch
        or key_dim != head_dim
        or query_heads != kv_heads * kv_group_size
        or tuple(leaf_v.shape) != tuple(leaf_k.shape)
    ):
        raise ValueError("query-routed AITER cache geometry is inconsistent")
    if tuple(slot_offsets.shape[:2]) != (batch, kv_heads) or tuple(
        packed_leaf_indices.shape
    ) != (batch, kv_heads, leaf_capacity):
        raise ValueError("query-routed AITER leaf metadata is inconsistent")
    partitions = math.ceil(leaf_capacity / partition_size)
    if (
        buffers is None
        or tuple(buffers["leaf_indices"].shape)
        != (batch, query_heads, leaf_capacity)
        or tuple(buffers["exp_sums"].shape)
        != (batch * query_heads, 1, partitions)
        or buffers["leaf_indices"].device != q.device
    ):
        buffers = new_query_routed_aiter_buffers(
            q,
            leaf_capacity=leaf_capacity,
            partition_size=partition_size,
        )

    pack_begin = None
    if timing_events is not None:
        pack_begin = torch.cuda.Event(enable_timing=True)
        pack_begin.record()
    _pack_query_routed_leaves_kernel[(batch * query_heads, route_count)](
        top_slots,
        slot_offsets,
        packed_leaf_indices,
        buffers["leaf_indices"],
        buffers["lengths"],
        top_slots.stride(0),
        top_slots.stride(1),
        top_slots.stride(3),
        slot_offsets.stride(0),
        slot_offsets.stride(1),
        slot_offsets.stride(2),
        packed_leaf_indices.stride(0),
        packed_leaf_indices.stride(1),
        packed_leaf_indices.stride(2),
        buffers["leaf_indices"].stride(0),
        buffers["leaf_indices"].stride(1),
        buffers["leaf_indices"].stride(2),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        ROUTE_COUNT=route_count,
        ROUTE_BLOCK=triton.next_power_of_2(route_count),
        STATE_CAPACITY=int(slot_offsets.size(2)) - 1,
        LEAF_CAPACITY=leaf_capacity,
        COPY_BLOCK=128,
        num_warps=4,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        pack_end = torch.cuda.Event(enable_timing=True)
        pack_end.record()
        timing_events.setdefault("aiter_routed_pack", []).append(
            (pack_begin, pack_end)
        )

    from csrc.cpp_itfs.pa.pa_v1 import paged_attention_v1

    sequences = batch * query_heads
    aiter_begin = None
    if timing_events is not None:
        aiter_begin = torch.cuda.Event(enable_timing=True)
        aiter_begin.record()
    paged_attention_v1(
        buffers["output"].view(sequences, 1, head_dim),
        buffers["workspace"],
        q.view(sequences, 1, head_dim),
        leaf_k.view(batch * kv_heads * leaf_capacity, 1, 1, head_dim),
        leaf_v.view(batch * kv_heads * leaf_capacity, 1, 1, head_dim),
        float(scale),
        buffers["leaf_indices"].view(sequences, leaf_capacity),
        None,
        buffers["lengths"].view(sequences),
        leaf_capacity,
        None,
        "auto",
        "NHD",
        0.0,
        buffers["unit_scale"],
        buffers["unit_scale"],
        None,
        partition_size,
        1,
        sliding_window=0,
    )
    if timing_events is not None:
        aiter_end = torch.cuda.Event(enable_timing=True)
        aiter_end.record()
        timing_events.setdefault("aiter_routed_exact", []).append(
            (aiter_begin, aiter_end)
        )
    return (
        buffers["output"],
        buffers["exp_sums"],
        buffers["max_logits"],
        buffers["lengths"].view(sequences),
        buffers,
    )


def gqa_union_indexed_attention(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    top_slots: torch.Tensor,
    slot_offsets: torch.Tensor,
    packed_leaf_indices: torch.Tensor,
    *,
    kv_group_size: int,
    union_group_size: int | None = None,
    scale: float,
    cache_indices: torch.Tensor | None = None,
    buffers: dict[str, torch.Tensor] | None = None,
    local_begin: int = 0,
    local_len: int = 0,
    local_begins: torch.Tensor | None = None,
    local_lens: torch.Tensor | None = None,
    local_capacity: int = 0,
    max_slot_leaves: int = 0,
    block_n: int = 128,
    num_warps: int = 4,
    waves_per_eu: int = 1,
    timing_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Attend to the deduplicated union of every GQA head's routed slots."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("GQA-union leaf attention is forward-only")
    if q.ndim != 4 or int(q.size(2)) != 1:
        raise ValueError("GQA-union leaf attention supports decode only")
    if leaf_k.ndim == 5:
        leaf_k = leaf_k.flatten(2, 3)
        leaf_v = leaf_v.flatten(2, 3)
    if leaf_k.ndim != 4 or leaf_v.ndim != 4:
        raise ValueError("leaf K/V must have rank four or paged rank five")
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (q, leaf_k, leaf_v, top_slots, slot_offsets, packed_leaf_indices)
    ):
        raise ValueError("GQA-union inputs must be contiguous CUDA tensors")
    batch, query_heads, _, head_dim = q.shape
    cache_batch, kv_heads, leaf_capacity, key_dim = leaf_k.shape
    route_count = int(top_slots.size(-1))
    if (cache_indices is None and cache_batch != batch) or key_dim != head_dim:
        raise ValueError("GQA-union cache geometry does not match queries")
    if cache_indices is not None and (
        tuple(cache_indices.shape) != (batch,)
        or cache_indices.dtype not in (torch.int32, torch.int64)
        or cache_indices.device != q.device
    ):
        raise ValueError("GQA-union cache indices are incompatible")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("GQA-union query/KV head grouping is inconsistent")
    if tuple(leaf_v.shape) != tuple(leaf_k.shape):
        raise ValueError("GQA-union prototype requires equal K/V geometry")
    if tuple(slot_offsets.shape[:2]) != (cache_batch, kv_heads):
        raise ValueError("slot offsets do not match the GQA cache")
    if tuple(packed_leaf_indices.shape) != (
        cache_batch,
        kv_heads,
        leaf_capacity,
    ):
        raise ValueError("packed leaf indices do not match leaf storage")
    if block_n not in (32, 64, 128):
        raise ValueError("GQA-union block size must be 32, 64, or 128")
    if union_group_size is None:
        union_group_size = kv_group_size
    if union_group_size <= 0 or kv_group_size % union_group_size:
        raise ValueError("GQA union groups must partition each physical group")
    logical_groups = query_heads // union_group_size
    union_count = union_group_size * route_count
    expected_top = (batch, query_heads, 1, union_count)
    if (
        buffers is None
        or tuple(buffers["leaf_indices"].shape)
        != (batch, logical_groups, leaf_capacity)
        or tuple(buffers["top_slots"].shape) != expected_top
        or buffers["leaf_indices"].device != q.device
    ):
        buffers = new_gqa_union_buffers(
            q,
            kv_heads=kv_heads,
            leaf_capacity=leaf_capacity,
            route_count=route_count,
            union_group_size=union_group_size,
        )
    pack_begin = None
    if timing_events is not None:
        pack_begin = torch.cuda.Event(enable_timing=True)
        pack_begin.record()
    _pack_gqa_union_leaves(
        top_slots,
        slot_offsets,
        packed_leaf_indices,
        buffers,
        kv_group_size=kv_group_size,
        union_group_size=union_group_size,
        cache_indices=cache_indices,
        local_begin=local_begin,
        local_len=local_len,
        local_begins=local_begins,
        local_lens=local_lens,
        local_capacity=(local_capacity if local_begins is not None else local_len),
        max_slot_leaves=max_slot_leaves,
        global_page_indices=True,
        clamp_empty_lengths=False,
        fused_metadata=head_dim > 256,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        pack_end = torch.cuda.Event(enable_timing=True)
        pack_end.record()
        timing_events.setdefault("gqa_union_pack", []).append(
            (pack_begin, pack_end)
        )
    block_g = triton.next_power_of_2(union_group_size)
    block_d = triton.next_power_of_2(head_dim)
    exact_begin = None
    if timing_events is not None:
        exact_begin = torch.cuda.Event(enable_timing=True)
        exact_begin.record()
    single_head = head_dim > 256
    common_args = (
        q,
        leaf_k,
        leaf_v,
        buffers["leaf_indices"],
        buffers["lengths"],
        buffers["output"],
        buffers["lse"],
        q.stride(0),
        q.stride(1),
        q.stride(3),
        leaf_k.stride(0),
        leaf_k.stride(1),
        leaf_k.stride(2),
        leaf_k.stride(3),
        leaf_v.stride(0),
        leaf_v.stride(1),
        leaf_v.stride(2),
        leaf_v.stride(3),
        buffers["leaf_indices"].stride(0),
        buffers["leaf_indices"].stride(1),
        buffers["leaf_indices"].stride(2),
        buffers["output"].stride(0),
        buffers["output"].stride(1),
        buffers["output"].stride(3),
        buffers["lse"].stride(0),
        buffers["lse"].stride(1),
    )
    common_constants = dict(
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        UNION_GROUP_SIZE=union_group_size,
        LOGICAL_GROUPS=logical_groups,
        LEAF_CAPACITY=leaf_capacity,
        TOTAL_LEAVES=cache_batch * kv_heads * leaf_capacity,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if single_head:
        _gqa_union_indexed_attention_wide_kernel[(batch * query_heads,)](
            *common_args, **common_constants
        )
    else:
        _gqa_union_indexed_attention_kernel[(batch * logical_groups,)](
            *common_args,
            GROUPS_PER_KV=kv_group_size // union_group_size,
            BLOCK_G=block_g,
            SINGLE_HEAD=False,
            **common_constants,
        )
    if timing_events is not None:
        exact_end = torch.cuda.Event(enable_timing=True)
        exact_end.record()
        timing_events.setdefault("gqa_union_exact", []).append(
            (exact_begin, exact_end)
        )
    return (
        buffers["output"],
        buffers["lse"],
        buffers["top_slots"],
        buffers,
    )


def gqa_union_aiter_attention(
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    top_slots: torch.Tensor,
    slot_offsets: torch.Tensor,
    packed_leaf_indices: torch.Tensor,
    *,
    kv_group_size: int,
    union_group_size: int | None = None,
    scale: float,
    cache_indices: torch.Tensor | None = None,
    buffers: dict[str, torch.Tensor] | None = None,
    local_begin: int = 0,
    local_len: int = 0,
    local_begins: torch.Tensor | None = None,
    local_lens: torch.Tensor | None = None,
    local_capacity: int = 0,
    new_k: torch.Tensor | None = None,
    new_v: torch.Tensor | None = None,
    local_k: torch.Tensor | None = None,
    local_v: torch.Tensor | None = None,
    archive_begins: torch.Tensor | None = None,
    max_slot_leaves: int = 0,
    partition_size: int = 256,
    waves_per_eu: int = 1,
    stage1_only: bool = False,
    timing_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ]
    | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Run actual AITER PA-v1 over the GQA-wide union without forming LSE.

    The returned exponential sums and maxima alias AITER's workspace.  They
    are intentionally left partitioned so LOD's final branch reducer can form
    the exact LSE in registers instead of launching a separate reduction.
    """
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("AITER GQA-union attention is forward-only")
    if q.ndim != 4 or int(q.size(2)) != 1:
        raise ValueError("AITER GQA-union attention supports decode only")
    if leaf_k.ndim == 5:
        leaf_k = leaf_k.flatten(2, 3)
        leaf_v = leaf_v.flatten(2, 3)
    if leaf_k.ndim != 4 or leaf_v.ndim != 4:
        raise ValueError("leaf K/V must have rank four or paged rank five")
    if not all(
        tensor.is_cuda and tensor.is_contiguous()
        for tensor in (
            q,
            leaf_k,
            leaf_v,
            top_slots,
            slot_offsets,
            packed_leaf_indices,
        )
    ):
        raise ValueError("AITER GQA-union inputs must be contiguous CUDA tensors")
    batch, query_heads, _, head_dim = q.shape
    _require_wide_head_pa_v1(head_dim)
    cache_batch, kv_heads, leaf_capacity, key_dim = leaf_k.shape
    route_count = int(top_slots.size(-1))
    if (cache_indices is None and cache_batch != batch) or key_dim != head_dim:
        raise ValueError("AITER GQA-union cache geometry does not match queries")
    if cache_indices is not None and (
        tuple(cache_indices.shape) != (batch,)
        or cache_indices.dtype not in (torch.int32, torch.int64)
        or cache_indices.device != q.device
    ):
        raise ValueError("AITER GQA-union cache indices are incompatible")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("AITER GQA-union query/KV grouping is inconsistent")
    if union_group_size is None:
        union_group_size = kv_group_size
    if union_group_size <= 0 or kv_group_size % union_group_size:
        raise ValueError("AITER union groups must evenly partition each KV group")
    if tuple(leaf_v.shape) != tuple(leaf_k.shape):
        raise ValueError("AITER GQA-union requires equal K/V geometry")
    if tuple(slot_offsets.shape[:2]) != (cache_batch, kv_heads):
        raise ValueError("slot offsets do not match the AITER GQA cache")
    if tuple(packed_leaf_indices.shape) != (
        cache_batch,
        kv_heads,
        leaf_capacity,
    ):
        raise ValueError("packed leaf indices do not match AITER leaf storage")
    if local_begin < 0 or local_len < 0 or local_begin + local_len > leaf_capacity:
        raise ValueError("AITER local leaf range exceeds the virtual archive")
    if local_capacity < 0 or local_capacity > leaf_capacity:
        raise ValueError("AITER local capacity exceeds the virtual archive")
    if max_slot_leaves < 0:
        raise ValueError("AITER maximum slot leaf count must be nonnegative")
    if local_begins is not None and local_capacity == 0:
        raise ValueError("AITER ranged local leaves require a positive capacity")
    logical_groups = query_heads // union_group_size
    union_count = union_group_size * route_count
    expected_top = (batch, query_heads, 1, union_count)
    partitions = math.ceil(leaf_capacity / partition_size)
    if (
        buffers is None
        or tuple(buffers["leaf_indices"].shape)
        != (batch, logical_groups, leaf_capacity)
        or tuple(buffers["top_slots"].shape) != expected_top
        or not isinstance(buffers.get("exp_sums"), torch.Tensor)
        or tuple(buffers["exp_sums"].shape)
        != (batch * logical_groups, union_group_size, partitions)
        or buffers["leaf_indices"].device != q.device
    ):
        buffers = new_gqa_union_aiter_buffers(
            q,
            kv_heads=kv_heads,
            leaf_capacity=leaf_capacity,
            route_count=route_count,
            union_group_size=union_group_size,
            partition_size=partition_size,
        )

    pack_begin = None
    if timing_events is not None:
        pack_begin = torch.cuda.Event(enable_timing=True)
        pack_begin.record()
    _pack_gqa_union_leaves(
        top_slots,
        slot_offsets,
        packed_leaf_indices,
        buffers,
        kv_group_size=kv_group_size,
        union_group_size=union_group_size,
        cache_indices=cache_indices,
        local_begin=local_begin,
        local_len=local_len,
        local_begins=local_begins,
        local_lens=local_lens,
        local_capacity=(local_capacity if local_begins is not None else local_len),
        new_k=new_k,
        new_v=new_v,
        local_k=local_k,
        local_v=local_v,
        archive_k=leaf_k if new_k is not None else None,
        archive_v=leaf_v if new_v is not None else None,
        archive_begins=archive_begins,
        max_slot_leaves=max_slot_leaves,
        global_page_indices=True,
        clamp_empty_lengths=True,
        fused_metadata=False,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        pack_end = torch.cuda.Event(enable_timing=True)
        pack_end.record()
        timing_events.setdefault("gqa_union_pack", []).append(
            (pack_begin, pack_end)
        )

    sequences = batch * logical_groups
    aiter_begin = None
    if timing_events is not None:
        aiter_begin = torch.cuda.Event(enable_timing=True)
        aiter_begin.record()
    if stage1_only:
        from model.kernels.aiter_pa_stage1 import paged_attention_stage1

        paged_attention_stage1(
            buffers["workspace"],
            q.view(sequences, union_group_size, head_dim),
            leaf_k.view(cache_batch * kv_heads * leaf_capacity, 1, 1, head_dim),
            leaf_v.view(cache_batch * kv_heads * leaf_capacity, 1, 1, head_dim),
            float(scale),
            buffers["leaf_indices"].view(sequences, leaf_capacity),
            buffers["lengths"].view(sequences),
            leaf_capacity,
            buffers["unit_scale"],
            buffers["unit_scale"],
            partition_size=partition_size,
        )
    else:
        # Import the PA-v1 source wrapper directly. Importing ``aiter.ops`` also
        # initializes unrelated optional Gluon kernels, which makes this HIP/CK
        # operator needlessly depend on AITER's bundled Triton version.
        from csrc.cpp_itfs.pa.pa_v1 import paged_attention_v1

        paged_attention_v1(
            buffers["output"].view(sequences, union_group_size, head_dim),
            buffers["workspace"],
            q.view(sequences, union_group_size, head_dim),
            leaf_k.view(cache_batch * kv_heads * leaf_capacity, 1, 1, head_dim),
            leaf_v.view(cache_batch * kv_heads * leaf_capacity, 1, 1, head_dim),
            float(scale),
            buffers["leaf_indices"].view(sequences, leaf_capacity),
            None,
            buffers["lengths"].view(sequences),
            leaf_capacity,
            None,
            "auto",
            "NHD",
            0.0,
            buffers["unit_scale"],
            buffers["unit_scale"],
            None,
            partition_size,
            1,
            sliding_window=0,
        )
    if timing_events is not None:
        aiter_end = torch.cuda.Event(enable_timing=True)
        aiter_end.record()
        timing_events.setdefault("gqa_union_exact", []).append(
            (aiter_begin, aiter_end)
        )
    return (
        buffers["output"],
        buffers["exp_sums"],
        buffers["max_logits"],
        buffers["actual_lengths"].view(batch * logical_groups),
        buffers["top_slots"],
        buffers,
    )


__all__ = [
    "gqa_union_aiter_attention",
    "gqa_union_indexed_attention",
    "new_gqa_union_aiter_buffers",
    "new_gqa_union_buffers",
    "new_query_routed_aiter_buffers",
    "query_routed_aiter_attention",
]
