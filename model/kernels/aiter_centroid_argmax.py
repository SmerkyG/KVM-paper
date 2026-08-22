# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Flat page-size-1 QK and centroid-segment argmax.

The QK pass is adapted from AITER's page-size-1 sparse decode traversal.  It
reads one centroid-ordered stream of leaf indices using ordinary dense key
tiles, but omits values, exponentials, softmax statistics, and attention
output accumulation.  A second inexpensive pass reduces the emitted scores
over the contiguous range belonging to each centroid.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
def _flat_page1_qk_kernel(
    q_ptr,  # [B, Hq, D]
    k_ptr,  # [total_pages, Hkv, D]
    leaf_indices_ptr,  # [B, Hkv, N], ordered by centroid
    score_ptr,  # [B, Hkv, N, G]
    total_pages,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_n: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    li_stride_b: tl.constexpr,
    li_stride_h: tl.constexpr,
    li_stride_n: tl.constexpr,
    s_stride_b: tl.constexpr,
    s_stride_h: tl.constexpr,
    s_stride_n: tl.constexpr,
    s_stride_g: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QUERY_GROUP: tl.constexpr,
    LEAF_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One conventional QK tile over the flat page-size-1 index stream."""
    batch_kv = tl.program_id(0).to(tl.int64)
    key_tile = tl.program_id(1).to(tl.int64)
    kv_head = batch_kv % KV_HEADS
    batch = batch_kv // KV_HEADS

    group = tl.arange(0, BLOCK_H)
    dim = tl.arange(0, BLOCK_D)
    key_lane = tl.arange(0, BLOCK_K)
    key_position = key_tile * BLOCK_K + key_lane
    valid_group = group < QUERY_GROUP
    valid_dim = dim < HEAD_DIM
    valid_key = key_position < LEAF_COUNT

    query_head = kv_head * QUERY_GROUP + group
    query = tl.load(
        q_ptr
        + batch * q_stride_b
        + query_head[:, None] * q_stride_h
        + dim[None, :] * q_stride_d,
        mask=valid_group[:, None] & valid_dim[None, :],
        other=0.0,
    )
    leaf_index = tl.load(
        leaf_indices_ptr
        + batch * li_stride_b
        + kv_head * li_stride_h
        + key_position * li_stride_n,
        mask=valid_key,
        other=-1,
    ).to(tl.int64)
    valid_key &= (leaf_index >= 0) & (leaf_index < total_pages)
    key = tl.load(
        k_ptr
        + leaf_index[:, None] * k_stride_n
        + kv_head * k_stride_h
        + dim[None, :] * k_stride_d,
        mask=valid_key[:, None] & valid_dim[None, :],
        other=0.0,
    )
    scores = tl.dot(query, tl.trans(key), out_dtype=tl.float32)
    score_offset = (
        batch * s_stride_b
        + kv_head * s_stride_h
        + key_position[:, None] * s_stride_n
        + group[None, :] * s_stride_g
    )
    tl.store(
        score_ptr + score_offset,
        tl.trans(scores),
        mask=valid_key[:, None] & valid_group[None, :],
    )


@triton.jit
def _flat_centroid_argmax_reduce_kernel(
    score_ptr,  # [B, Hkv, N, G]
    leaf_indices_ptr,  # [B, Hkv, N]
    centroid_offsets_ptr,  # [B, Hkv, C + 1]
    winner_indices_ptr,  # [B, Hkv, C, G]
    winner_scores_ptr,  # [B, Hkv, C, G]
    s_stride_b: tl.constexpr,
    s_stride_h: tl.constexpr,
    s_stride_n: tl.constexpr,
    s_stride_g: tl.constexpr,
    li_stride_b: tl.constexpr,
    li_stride_h: tl.constexpr,
    li_stride_n: tl.constexpr,
    co_stride_b: tl.constexpr,
    co_stride_h: tl.constexpr,
    co_stride_c: tl.constexpr,
    wi_stride_b: tl.constexpr,
    wi_stride_h: tl.constexpr,
    wi_stride_c: tl.constexpr,
    wi_stride_g: tl.constexpr,
    ws_stride_b: tl.constexpr,
    ws_stride_h: tl.constexpr,
    ws_stride_c: tl.constexpr,
    ws_stride_g: tl.constexpr,
    KV_HEADS: tl.constexpr,
    CENTROIDS: tl.constexpr,
    QUERY_GROUP: tl.constexpr,
    CENTROIDS_PER_BLOCK: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce several adjacent centroid segments; no QK work occurs here."""
    centroid_blocks = tl.cdiv(CENTROIDS, CENTROIDS_PER_BLOCK)
    expert_block = tl.program_id(0).to(tl.int64)
    centroid_block = expert_block % centroid_blocks
    batch_kv = expert_block // centroid_blocks
    kv_head = batch_kv % KV_HEADS
    batch = batch_kv // KV_HEADS
    centroid_lane = tl.arange(0, CENTROIDS_PER_BLOCK)
    centroid = centroid_block * CENTROIDS_PER_BLOCK + centroid_lane
    valid_centroid = centroid < CENTROIDS
    group = tl.arange(0, BLOCK_G)
    leaf_lane = tl.arange(0, BLOCK_N)
    valid_group = group < QUERY_GROUP

    offset_base = (
        batch * co_stride_b + kv_head * co_stride_h + centroid * co_stride_c
    )
    leaf_begin = tl.load(
        centroid_offsets_ptr + offset_base,
        mask=valid_centroid,
        other=0,
    ).to(tl.int32)
    leaf_end = tl.load(
        centroid_offsets_ptr + offset_base + co_stride_c,
        mask=valid_centroid,
        other=0,
    ).to(tl.int32)
    leaf_count = leaf_end - leaf_begin
    max_leaf_count = tl.max(leaf_count, axis=0)
    best_score = tl.full(
        (CENTROIDS_PER_BLOCK, BLOCK_G), -float("inf"), tl.float32
    )
    best_leaf = tl.full((CENTROIDS_PER_BLOCK, BLOCK_G), -1, tl.int32)

    for leaf_start in tl.range(0, max_leaf_count, BLOCK_N, num_stages=2):
        leaf_offset = leaf_start + leaf_lane
        valid_leaf = leaf_offset[None, :] < leaf_count[:, None]
        leaf_position = leaf_begin[:, None] + leaf_offset[None, :]
        scores = tl.load(
            score_ptr
            + batch * s_stride_b
            + kv_head * s_stride_h
            + leaf_position[:, None, :] * s_stride_n
            + group[None, :, None] * s_stride_g,
            mask=(
                valid_centroid[:, None, None]
                & valid_group[None, :, None]
                & valid_leaf[:, None, :]
            ),
            other=-float("inf"),
        )
        tile_score = tl.max(scores, axis=2)
        tile_offset = tl.argmax(scores, axis=2, tie_break_left=True).to(tl.int32)
        winner_position = leaf_begin[:, None] + leaf_start + tile_offset
        tile_valid = (
            valid_centroid[:, None]
            & valid_group[None, :]
            & (leaf_start < leaf_count[:, None])
        )
        tile_leaf = tl.load(
            leaf_indices_ptr
            + batch * li_stride_b
            + kv_head * li_stride_h
            + winner_position * li_stride_n,
            mask=tile_valid,
            other=-1,
        ).to(tl.int32)
        better = tile_valid & (tile_score > best_score)
        best_score = tl.where(better, tile_score, best_score)
        best_leaf = tl.where(better, tile_leaf, best_leaf)

    winner_index_offset = (
        batch * wi_stride_b
        + kv_head * wi_stride_h
        + centroid[:, None] * wi_stride_c
        + group[None, :] * wi_stride_g
    )
    winner_score_offset = (
        batch * ws_stride_b
        + kv_head * ws_stride_h
        + centroid[:, None] * ws_stride_c
        + group[None, :] * ws_stride_g
    )
    valid_output = valid_centroid[:, None] & valid_group[None, :]
    tl.store(
        winner_indices_ptr + winner_index_offset,
        tl.where(leaf_count[:, None] > 0, best_leaf, -1),
        mask=valid_output,
    )
    tl.store(
        winner_scores_ptr + winner_score_offset,
        tl.where(leaf_count[:, None] > 0, best_score, -float("inf")),
        mask=valid_output,
    )


@triton.jit
def _flat_centroid_argmax_stream_reduce_kernel(
    score_ptr,  # [B, Hkv, N, G]
    leaf_indices_ptr,  # [B, Hkv, N]
    leaf_owners_ptr,  # [B, Hkv, N]
    centroid_offsets_ptr,  # [B, Hkv, C + 1]
    winner_indices_ptr,  # [B, Hkv, C, G]
    winner_scores_ptr,  # [B, Hkv, C, G]
    s_stride_b: tl.constexpr,
    s_stride_h: tl.constexpr,
    s_stride_n: tl.constexpr,
    s_stride_g: tl.constexpr,
    li_stride_b: tl.constexpr,
    li_stride_h: tl.constexpr,
    li_stride_n: tl.constexpr,
    lo_stride_b: tl.constexpr,
    lo_stride_h: tl.constexpr,
    lo_stride_n: tl.constexpr,
    co_stride_b: tl.constexpr,
    co_stride_h: tl.constexpr,
    co_stride_c: tl.constexpr,
    wi_stride_b: tl.constexpr,
    wi_stride_h: tl.constexpr,
    wi_stride_c: tl.constexpr,
    wi_stride_g: tl.constexpr,
    ws_stride_b: tl.constexpr,
    ws_stride_h: tl.constexpr,
    ws_stride_c: tl.constexpr,
    ws_stride_g: tl.constexpr,
    KV_HEADS: tl.constexpr,
    CENTROIDS: tl.constexpr,
    QUERY_GROUP: tl.constexpr,
    CENTROIDS_PER_BLOCK: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Segmented argmax over a centroid-ordered leaf stream."""
    centroid_blocks = tl.cdiv(CENTROIDS, CENTROIDS_PER_BLOCK)
    expert_block = tl.program_id(0).to(tl.int64)
    centroid_block = expert_block % centroid_blocks
    batch_kv = expert_block // centroid_blocks
    kv_head = batch_kv % KV_HEADS
    batch = batch_kv // KV_HEADS
    centroid_lane = tl.arange(0, CENTROIDS_PER_BLOCK)
    centroid_begin = centroid_block * CENTROIDS_PER_BLOCK
    centroid = centroid_begin + centroid_lane
    centroid_end = tl.minimum(centroid_begin + CENTROIDS_PER_BLOCK, CENTROIDS)
    valid_centroid = centroid < CENTROIDS
    group = tl.arange(0, BLOCK_G)
    valid_group = group < QUERY_GROUP
    leaf_lane = tl.arange(0, BLOCK_N)

    winner_index_offset = (
        batch * wi_stride_b
        + kv_head * wi_stride_h
        + centroid[:, None] * wi_stride_c
        + group[None, :] * wi_stride_g
    )
    winner_score_offset = (
        batch * ws_stride_b
        + kv_head * ws_stride_h
        + centroid[:, None] * ws_stride_c
        + group[None, :] * ws_stride_g
    )
    valid_output = valid_centroid[:, None] & valid_group[None, :]
    tl.store(winner_indices_ptr + winner_index_offset, -1, mask=valid_output)
    tl.store(
        winner_scores_ptr + winner_score_offset,
        -float("inf"),
        mask=valid_output,
    )
    tl.debug_barrier()

    offset_row = batch * co_stride_b + kv_head * co_stride_h
    leaf_begin = tl.load(
        centroid_offsets_ptr + offset_row + centroid_begin * co_stride_c
    ).to(tl.int32)
    leaf_end = tl.load(
        centroid_offsets_ptr + offset_row + centroid_end * co_stride_c
    ).to(tl.int32)
    carry_owner = tl.full((), -2, tl.int32)
    carry_score = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    carry_leaf = tl.full((BLOCK_G,), -1, tl.int32)

    for leaf_start in tl.range(0, leaf_end - leaf_begin, BLOCK_N, num_stages=1):
        leaf_position = leaf_begin + leaf_start + leaf_lane
        valid_leaf = leaf_position < leaf_end
        owner = tl.load(
            leaf_owners_ptr
            + batch * lo_stride_b
            + kv_head * lo_stride_h
            + leaf_position * lo_stride_n,
            mask=valid_leaf,
            other=-1,
        ).to(tl.int32)
        leaf_index = tl.load(
            leaf_indices_ptr
            + batch * li_stride_b
            + kv_head * li_stride_h
            + leaf_position * li_stride_n,
            mask=valid_leaf,
            other=-1,
        ).to(tl.int32)
        valid_leaf &= (owner >= centroid_begin) & (owner < centroid_end)
        scores = tl.load(
            score_ptr
            + batch * s_stride_b
            + kv_head * s_stride_h
            + leaf_position[None, :] * s_stride_n
            + group[:, None] * s_stride_g,
            mask=valid_group[:, None] & valid_leaf[None, :],
            other=-float("inf"),
        )
        scan_owner, scan_score, scan_leaf = tl.associative_scan(
            (
                owner[None, :] + tl.zeros((BLOCK_G, BLOCK_N), tl.int32),
                scores,
                leaf_index[None, :] + tl.zeros((BLOCK_G, BLOCK_N), tl.int32),
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
            leaf_owners_ptr
            + batch * lo_stride_b
            + kv_head * lo_stride_h
            + (leaf_position + 1) * lo_stride_n,
            mask=leaf_position + 1 < leaf_end,
            other=-1,
        ).to(tl.int32)
        segment_end = valid_leaf & (owner != next_owner)
        result_index_offset = (
            batch * wi_stride_b
            + kv_head * wi_stride_h
            + owner[None, :] * wi_stride_c
            + group[:, None] * wi_stride_g
        )
        result_score_offset = (
            batch * ws_stride_b
            + kv_head * ws_stride_h
            + owner[None, :] * ws_stride_c
            + group[:, None] * ws_stride_g
        )
        result_mask = valid_group[:, None] & segment_end[None, :]
        tl.store(
            winner_indices_ptr + result_index_offset,
            scan_leaf,
            mask=result_mask,
        )
        tl.store(
            winner_scores_ptr + result_score_offset,
            scan_score,
            mask=result_mask,
        )
        block_last = valid_leaf & (
            leaf_lane == tl.minimum(BLOCK_N - 1, leaf_end - leaf_begin - leaf_start - 1)
        )
        carry_owner = tl.max(tl.where(block_last, owner, -1), axis=0)
        carry_score = tl.max(
            tl.where(block_last[None, :], scan_score, -float("inf")),
            axis=1,
        )
        carry_leaf = tl.sum(
            tl.where(block_last[None, :], scan_leaf, 0), axis=1
        ).to(tl.int32)


def _key_cache_view(key_cache: torch.Tensor) -> torch.Tensor:
    if key_cache.ndim == 4:
        if key_cache.size(1) != 1:
            raise ValueError("four-dimensional key cache must have page size 1")
        key_cache = key_cache[:, 0]
    if key_cache.ndim != 3:
        raise ValueError("key cache must have shape [pages, Hkv, D]")
    return key_cache


def flat_page1_qk_scores(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    leaf_indices: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
    block_k: int = 128,
) -> torch.Tensor:
    """Emit ``q @ k.T`` scores in centroid-stream order."""
    key_cache = _key_cache_view(key_cache)
    if not query.is_cuda:
        raise RuntimeError("flat page-size-1 QK requires CUDA/HIP tensors")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"query must be bf16/fp16, got {query.dtype}")
    if key_cache.dtype != query.dtype:
        raise TypeError("query and key cache dtypes differ")
    if leaf_indices.ndim != 3 or leaf_indices.dtype != torch.int32:
        raise TypeError("leaf indices must be int32 [B, Hkv, N]")
    if not (
        query.is_contiguous()
        and key_cache.is_contiguous()
        and leaf_indices.is_contiguous()
    ):
        raise ValueError("flat QK inputs must be contiguous")
    if not (query.device == key_cache.device == leaf_indices.device):
        raise ValueError("flat QK inputs must share one device")

    batch, query_heads, head_dim = query.shape
    total_pages, kv_heads, key_dim = key_cache.shape
    if key_dim != head_dim:
        raise ValueError("query and key head dimensions differ")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if tuple(leaf_indices.shape[:2]) != (batch, kv_heads):
        raise ValueError("leaf-index batch/KV-head shape is incompatible")
    if head_dim != triton.next_power_of_2(head_dim):
        raise ValueError("the prototype requires a power-of-two head dimension")
    if block_k not in (64, 128):
        raise ValueError("block_k must be 64 or 128")

    leaf_count = leaf_indices.size(2)
    query_group = query_heads // kv_heads
    expected = (batch, kv_heads, leaf_count, query_group)
    if scores is None:
        scores = torch.empty(expected, dtype=torch.float32, device=query.device)
    elif tuple(scores.shape) != expected or scores.dtype != torch.float32:
        raise ValueError("score workspace has an incompatible shape or dtype")
    elif not scores.is_contiguous() or scores.device != query.device:
        raise ValueError("score workspace must be contiguous and on the query device")

    block_h = max(16, triton.next_power_of_2(query_group))
    _flat_page1_qk_kernel[(batch * kv_heads, triton.cdiv(leaf_count, block_k))](
        query,
        key_cache,
        leaf_indices,
        scores,
        total_pages,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        leaf_indices.stride(0),
        leaf_indices.stride(1),
        leaf_indices.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        KV_HEADS=kv_heads,
        QUERY_GROUP=query_group,
        LEAF_COUNT=leaf_count,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        BLOCK_D=head_dim,
        BLOCK_K=block_k,
        num_warps=4,
        waves_per_eu=1,
    )
    return scores


def flat_centroid_argmax_reduce(
    scores: torch.Tensor,
    leaf_indices: torch.Tensor,
    centroid_offsets: torch.Tensor,
    *,
    leaf_owners: torch.Tensor | None = None,
    winner_indices: torch.Tensor | None = None,
    winner_scores: torch.Tensor | None = None,
    block_n: int = 32,
    centroids_per_block: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce flat QK scores over each contiguous centroid range."""
    if scores.ndim != 4 or scores.dtype != torch.float32:
        raise TypeError("scores must be fp32 [B, Hkv, N, G]")
    if leaf_indices.ndim != 3 or leaf_indices.dtype != torch.int32:
        raise TypeError("leaf indices must be int32 [B, Hkv, N]")
    if centroid_offsets.ndim != 3 or centroid_offsets.dtype != torch.int32:
        raise TypeError("centroid offsets must be int32 [B, Hkv, C + 1]")
    if not (
        scores.is_contiguous()
        and leaf_indices.is_contiguous()
        and centroid_offsets.is_contiguous()
    ):
        raise ValueError("flat reduction inputs must be contiguous")
    if not (scores.device == leaf_indices.device == centroid_offsets.device):
        raise ValueError("flat reduction inputs must share one device")
    if block_n < 16 or block_n != triton.next_power_of_2(block_n):
        raise ValueError("block_n must be a power of two of at least 16")
    if centroids_per_block not in (1, 2, 4, 8, 16):
        raise ValueError("centroids_per_block must be 1, 2, 4, 8, or 16")

    batch, kv_heads, leaf_count, query_group = scores.shape
    if tuple(leaf_indices.shape) != (batch, kv_heads, leaf_count):
        raise ValueError("leaf indices do not match the score stream")
    if tuple(centroid_offsets.shape[:2]) != (batch, kv_heads):
        raise ValueError("centroid offsets do not match score batch/KV heads")
    centroids = centroid_offsets.size(2) - 1
    expected = (batch, kv_heads, centroids, query_group)
    if winner_indices is None:
        winner_indices = torch.empty(
            expected, dtype=torch.int32, device=scores.device
        )
    if winner_scores is None:
        winner_scores = torch.empty(
            expected, dtype=torch.float32, device=scores.device
        )
    if tuple(winner_indices.shape) != expected or winner_indices.dtype != torch.int32:
        raise ValueError("winner-index workspace has an incompatible shape or dtype")
    if tuple(winner_scores.shape) != expected or winner_scores.dtype != torch.float32:
        raise ValueError("winner-score workspace has an incompatible shape or dtype")
    if not (winner_indices.is_contiguous() and winner_scores.is_contiguous()):
        raise ValueError("winner workspaces must be contiguous")
    if not (
        winner_indices.device == scores.device
        and winner_scores.device == scores.device
    ):
        raise ValueError("winner workspaces must be on the score device")

    block_g = triton.next_power_of_2(query_group)
    centroid_blocks = triton.cdiv(centroids, centroids_per_block)
    if leaf_owners is not None:
        if leaf_owners.dtype != torch.int32 or tuple(leaf_owners.shape) != tuple(
            leaf_indices.shape
        ):
            raise TypeError("leaf owners must be int32 with the leaf-index shape")
        if not leaf_owners.is_contiguous() or leaf_owners.device != scores.device:
            raise ValueError("leaf owners must be contiguous and on the score device")
        _flat_centroid_argmax_stream_reduce_kernel[
            (batch * kv_heads * centroid_blocks,)
        ](
            scores,
            leaf_indices,
            leaf_owners,
            centroid_offsets,
            winner_indices,
            winner_scores,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            scores.stride(3),
            leaf_indices.stride(0),
            leaf_indices.stride(1),
            leaf_indices.stride(2),
            leaf_owners.stride(0),
            leaf_owners.stride(1),
            leaf_owners.stride(2),
            centroid_offsets.stride(0),
            centroid_offsets.stride(1),
            centroid_offsets.stride(2),
            winner_indices.stride(0),
            winner_indices.stride(1),
            winner_indices.stride(2),
            winner_indices.stride(3),
            winner_scores.stride(0),
            winner_scores.stride(1),
            winner_scores.stride(2),
            winner_scores.stride(3),
            KV_HEADS=kv_heads,
            CENTROIDS=centroids,
            QUERY_GROUP=query_group,
            CENTROIDS_PER_BLOCK=centroids_per_block,
            BLOCK_G=block_g,
            BLOCK_N=block_n,
            num_warps=4,
            waves_per_eu=1,
        )
        return winner_indices, winner_scores

    _flat_centroid_argmax_reduce_kernel[(batch * kv_heads * centroid_blocks,)](
        scores,
        leaf_indices,
        centroid_offsets,
        winner_indices,
        winner_scores,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        leaf_indices.stride(0),
        leaf_indices.stride(1),
        leaf_indices.stride(2),
        centroid_offsets.stride(0),
        centroid_offsets.stride(1),
        centroid_offsets.stride(2),
        winner_indices.stride(0),
        winner_indices.stride(1),
        winner_indices.stride(2),
        winner_indices.stride(3),
        winner_scores.stride(0),
        winner_scores.stride(1),
        winner_scores.stride(2),
        winner_scores.stride(3),
        KV_HEADS=kv_heads,
        CENTROIDS=centroids,
        QUERY_GROUP=query_group,
        CENTROIDS_PER_BLOCK=centroids_per_block,
        BLOCK_G=block_g,
        BLOCK_N=block_n,
        num_warps=4,
        waves_per_eu=1,
    )
    return winner_indices, winner_scores


def centroid_argmax_page1(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    leaf_indices: torch.Tensor,
    centroid_offsets: torch.Tensor,
    *,
    leaf_owners: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    winner_indices: torch.Tensor | None = None,
    winner_scores: torch.Tensor | None = None,
    block_k: int = 128,
    block_n: int = 32,
    centroids_per_block: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run flat indexed QK followed by exact centroid-segment argmax."""
    scores = flat_page1_qk_scores(
        query, key_cache, leaf_indices, scores=scores, block_k=block_k
    )
    return flat_centroid_argmax_reduce(
        scores,
        leaf_indices,
        centroid_offsets,
        leaf_owners=leaf_owners,
        winner_indices=winner_indices,
        winner_scores=winner_scores,
        block_n=block_n,
        centroids_per_block=centroids_per_block,
    )


__all__ = [
    "centroid_argmax_page1",
    "flat_centroid_argmax_reduce",
    "flat_page1_qk_scores",
]
