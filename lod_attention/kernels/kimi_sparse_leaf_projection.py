"""Experimental prefill: project each needed leaf once per query head.

The output retains the original chronological leaf indices. Unselected
head/leaf pairs are deliberately unwritten, and may only be consumed through
the same post-cap route list that selected this projection work.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._paged_common import _lookup_page_id
from .aiter_mla_prefill_attention import _workspace_tensor


@triton.jit
def _project_needed_leaf_tiles(
    source, uk, uv, tile_experts, tile_ordinals,
    slot_pages, overflow_keys, overflow_values, overflow_used,
    page_indices, slot_lengths, output_k, output_v,
    TOKENS: tl.constexpr, HEADS: tl.constexpr, SLOTS: tl.constexpr,
    SOURCE_BATCH_STRIDE: tl.constexpr, SOURCE_TOKEN_STRIDE: tl.constexpr,
    UK_HEAD_STRIDE: tl.constexpr, UK_ROW_STRIDE: tl.constexpr,
    UK_LATENT_STRIDE: tl.constexpr,
    UV_HEAD_STRIDE: tl.constexpr, UV_LATENT_STRIDE: tl.constexpr,
    UV_VALUE_STRIDE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr, PAGE_CAPACITY: tl.constexpr,
    INLINE_PAGES: tl.constexpr, HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr, PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    program = tl.program_id(0)
    expert = tl.load(tile_experts + program).to(tl.int64)
    ordinal = tl.load(tile_ordinals + program)
    batch_head = expert // SLOTS
    slot = expert % SLOTS
    batch = batch_head // HEADS
    head = batch_head % HEADS
    token = ordinal * BLOCK_M + tl.arange(0, BLOCK_M)
    count = tl.load(slot_lengths + batch * STATE_CAPACITY + slot)
    valid = token < count
    page = _lookup_page_id(
        slot_pages, overflow_keys, overflow_values, overflow_used,
        batch, slot, token // PAGE_SIZE, valid,
        STATE_CAPACITY, INLINE_PAGES, PAGE_CAPACITY, HASH_CAPACITY, HASH_PROBES,
    ).to(tl.int64)
    valid &= (page >= 0) & (page < PAGE_CAPACITY)
    safe_page = tl.where(valid, page, 0)
    leaf = tl.load(
        page_indices + (batch * PAGE_CAPACITY + safe_page) * PAGE_SIZE
        + token % PAGE_SIZE,
        mask=valid, other=0,
    ).to(tl.int64)
    valid &= (leaf >= 0) & (leaf < TOKENS)
    leaf = tl.where(valid, leaf, 0)
    source_row = batch * SOURCE_BATCH_STRIDE + leaf * SOURCE_TOKEN_STRIDE
    k_lane = tl.arange(0, 128)
    output_lane = tl.arange(0, 128)
    key = tl.zeros((BLOCK_M, 128), tl.float32)
    value = tl.zeros((BLOCK_M, 128), tl.float32)
    for begin in tl.static_range(0, 512, 128):
        latent_lane = begin + k_lane
        latent = tl.load(
            source + source_row[:, None] + latent_lane[None, :],
            mask=valid[:, None], other=0.0,
        )
        key_weight = tl.load(
            uk + head * UK_HEAD_STRIDE
            + latent_lane[:, None] * UK_LATENT_STRIDE
            + output_lane[None, :] * UK_ROW_STRIDE,
        )
        value_weight = tl.load(
            uv + head * UV_HEAD_STRIDE
            + latent_lane[:, None] * UV_LATENT_STRIDE
            + output_lane[None, :] * UV_VALUE_STRIDE,
        )
        key += tl.dot(latent, key_weight)
        value += tl.dot(latent, value_weight)
    output_row = (batch * TOKENS + leaf) * HEADS + head
    tl.store(
        output_k + output_row[:, None] * 192 + output_lane[None, :],
        key, mask=valid[:, None],
    )
    tl.store(
        output_v + output_row[:, None] * 128 + output_lane[None, :],
        value, mask=valid[:, None],
    )
    direct_lane = tl.arange(0, 64)
    direct = tl.load(
        source + source_row[:, None] + 512 + direct_lane[None, :],
        mask=valid[:, None], other=0.0,
    )
    tl.store(
        output_k + output_row[:, None] * 192 + 128 + direct_lane[None, :],
        direct, mask=valid[:, None],
    )


def project_needed_kimi_leaves(
    key: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    page_cache: dict,
    head_counts: torch.Tensor,
    *,
    active_slots: int,
    hash_probes: int,
    buffers: dict[str, torch.Tensor] | None = None,
    block_m: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the union of centroid leaves selected anywhere in the chunk.

    This initial prototype compacts the tile list with PyTorch (one host
    synchronization). It is prefill-only and not a CUDA-graph decode path.
    The existing leaf packer reuses ``head_counts`` and its route offsets.
    """
    if key.ndim != 4 or key.size(1) != 1 or key.size(-1) != 576:
        raise ValueError("sparse Kimi projection expects [B,1,T,576]")
    batch, tokens = int(key.size(0)), int(key.size(2))
    heads = int(w_uk_t.size(0))
    if tuple(w_uk_t.shape) != (heads, 128, 512) or tuple(w_uv.shape) != (heads, 512, 128):
        raise ValueError("sparse Kimi projection weights have incompatible geometry")
    if head_counts.numel() != batch * heads * active_slots:
        raise ValueError("projection route counts have incompatible geometry")
    lengths = page_cache["slot_lengths"][..., :active_slots]
    counts = head_counts.view(batch, heads, active_slots)
    tile_counts = torch.where(counts > 0, (lengths + block_m - 1) // block_m, 0)
    tile_counts = tile_counts.reshape(-1).long()
    tile_starts = tile_counts.cumsum(0) - tile_counts
    # repeat_interleave would synchronize internally for its output shape;
    # supply that shape explicitly so the cost is visible and occurs once.
    tile_count = int(tile_counts.sum().item())
    experts = torch.repeat_interleave(
        torch.arange(tile_counts.numel(), device=key.device),
        tile_counts, output_size=tile_count,
    )
    ordinals = (torch.arange(tile_count, device=key.device)
                - tile_starts.index_select(0, experts)).int()
    output_k = _workspace_tensor(
        buffers, "kimi_leaf_expanded_k_token_major", (batch, tokens, heads, 192),
        dtype=key.dtype, device=key.device,
    )
    output_v = _workspace_tensor(
        buffers, "kimi_leaf_expanded_v_token_major", (batch, tokens, heads, 128),
        dtype=key.dtype, device=key.device,
    )
    if tile_count:
        slot_pages = page_cache["slot_pages"]
        indices = page_cache["page_indices"]
        overflow_keys = page_cache["overflow_page_keys"]
        _project_needed_leaf_tiles[(tile_count,)](
            key, w_uk_t, w_uv, experts, ordinals,
            slot_pages, overflow_keys, page_cache["overflow_page_values"],
            page_cache["overflow_used"], indices, page_cache["slot_lengths"],
            output_k, output_v,
            TOKENS=tokens, HEADS=heads, SLOTS=active_slots,
            SOURCE_BATCH_STRIDE=key.stride(0), SOURCE_TOKEN_STRIDE=key.stride(2),
            UK_HEAD_STRIDE=w_uk_t.stride(0), UK_ROW_STRIDE=w_uk_t.stride(1),
            UK_LATENT_STRIDE=w_uk_t.stride(2),
            UV_HEAD_STRIDE=w_uv.stride(0), UV_LATENT_STRIDE=w_uv.stride(1),
            UV_VALUE_STRIDE=w_uv.stride(2),
            STATE_CAPACITY=slot_pages.size(2), PAGE_CAPACITY=indices.size(2),
            INLINE_PAGES=slot_pages.size(3),
            HASH_CAPACITY=page_cache["overflow_page_values"].size(2),
            HASH_PROBES=hash_probes, PAGE_SIZE=indices.size(3),
            BLOCK_M=block_m, num_warps=4,
        )
    return output_k.permute(0, 2, 1, 3), output_v.permute(0, 2, 1, 3)
