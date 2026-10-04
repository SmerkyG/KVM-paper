"""Exact top-eight merge of eight already sorted K3 tile-candidate lists.

Only the current head of each list participates in a reduction. Advancing
the winning list avoids loading and sorting all 64 candidates at once.
Score ties use flattened candidate order, matching the ordinary reducer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .aiter_prefill_attention import _workspace_tensor


@triton.jit
def _merge_sorted_lists(
    candidates, lengths, routes, selected_scores, Q, STATES,
    HEADS: tl.constexpr, LENGTH_BATCH_STRIDE: tl.constexpr,
    LENGTH_HEAD_STRIDE: tl.constexpr, MAX_LEAVES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    head_row = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    rank = tl.arange(0, 8)
    cursor = tl.zeros((BLOCK_M, 8), tl.int32)
    base = ((head_row * 8 + rank[None, :]) * 16) * Q + query[:, None]
    score = tl.load(candidates + base, mask=query[:, None] < Q,
                    other=-float("inf")).to(tl.float32)
    index = tl.load(candidates + base + 8 * Q, mask=query[:, None] < Q,
                    other=-1).to(tl.int64)
    score = tl.where((index >= 0) & (index < STATES), score, -float("inf"))
    winners = tl.full((BLOCK_M, 8), -1, tl.int64)
    winner_scores = tl.full((BLOCK_M, 8), -float("inf"), tl.float32)
    for output_rank in tl.static_range(0, 8):
        best_score = tl.max(score, axis=1)
        ordinal = rank[None, :] * 8 + cursor
        best_ordinal = tl.min(tl.where(score == best_score[:, None], ordinal,
                                     0x7fffffff), axis=1)
        chosen = ordinal == best_ordinal[:, None]
        best_index = tl.sum(tl.where(chosen, index, 0), axis=1)
        winners = tl.where(rank[None, :] == output_rank, best_index[:, None], winners)
        winner_scores = tl.where(rank[None, :] == output_rank, best_score[:, None], winner_scores)
        if output_rank < 7:
            cursor += chosen.to(tl.int32)
            valid = chosen & (query[:, None] < Q) & (cursor < 8)
            next_score = tl.load(candidates + base + cursor * Q,
                                 mask=valid, other=-float("inf")).to(tl.float32)
            next_index = tl.load(candidates + base + (8 + cursor) * Q,
                                 mask=valid, other=-1).to(tl.int64)
            next_score = tl.where((next_index >= 0) & (next_index < STATES),
                                  next_score, -float("inf"))
            score = tl.where(chosen, next_score, score)
            index = tl.where(chosen, next_index, index)
    if MAX_LEAVES:
        batch = head_row // HEADS
        valid = (winners >= 0) & (winners < STATES) & (query[:, None] < Q)
        count = tl.load(lengths + batch * LENGTH_BATCH_STRIDE + winners,
                        mask=valid, other=0)
        winners = tl.where(valid & (count <= MAX_LEAVES), winners, -1)
        winner_scores = tl.where(winners >= 0, winner_scores, -float("inf"))
    # Preserve the expert consumer's boundary-last, ID-sorted convention.
    remaining = tl.where((rank[None, :] < 7) & (winners >= 0),
                         winners, 0x7fffffffffffffff)
    output_base = (head_row * Q + query) * 8
    for output_rank in tl.static_range(0, 8):
        if output_rank < 7:
            selected = tl.min(remaining, axis=1)
            selected = tl.where(selected == 0x7fffffffffffffff, -1, selected)
        else:
            selected = tl.max(tl.where(rank[None, :] == 7, winners, -1), axis=1)
        value = tl.max(tl.where((winners == selected[:, None]) & (selected[:, None] >= 0),
                               winner_scores, -float("inf")), axis=1)
        tl.store(routes + output_base + output_rank, selected, mask=query < Q)
        tl.store(selected_scores + output_base + output_rank,
                 value * 0.6931471805599453, mask=query < Q)
        remaining = tl.where(remaining == selected[:, None],
                             0x7fffffffffffffff, remaining)


def merge_sorted_kimi_candidates(candidates, *, state_len, slot_lengths=None,
                                 max_open_leaf_tokens=None, buffers=None):
    """Match K3's metadata-free reducer, including post-selection closing."""
    if (candidates.ndim != 5 or candidates.shape[2:4] != (8, 16)
            or not candidates.is_cuda or not candidates.is_contiguous()):
        raise ValueError("sorted K3 merge expects CUDA [B,H,8,16,Q] candidates")
    if (slot_lengths is None) != (max_open_leaf_tokens is None):
        raise ValueError("sorted K3 merge needs both slot lengths and a cap")
    batch, heads, _, _, queries = candidates.shape
    if slot_lengths is not None and (slot_lengths.ndim != 3
            or slot_lengths.shape[:2] != (batch, 1) or slot_lengths.size(2) < state_len
            or slot_lengths.stride(-1) != 1):
        raise ValueError("sorted K3 merge requires [B,1,S] lengths with unit slot stride")
    routes = _workspace_tensor(buffers, "kimi_merged_routes", (batch, heads, queries, 8),
                               dtype=torch.int64, device=candidates.device)
    scores = _workspace_tensor(buffers, "kimi_merged_route_scores", routes.shape,
                               dtype=torch.float32, device=candidates.device)
    _merge_sorted_lists[(batch * heads, triton.cdiv(queries, 32))](
        candidates, slot_lengths if slot_lengths is not None else candidates,
        routes, scores, queries, state_len, HEADS=heads,
        LENGTH_BATCH_STRIDE=slot_lengths.stride(0) if slot_lengths is not None else 0,
        LENGTH_HEAD_STRIDE=0, MAX_LEAVES=max_open_leaf_tokens or 0,
        BLOCK_M=32, num_warps=4,
    )
    return routes, None, None, scores
