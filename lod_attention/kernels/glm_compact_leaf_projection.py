"""Transient K256/V256 projection of selected NoPE MLA centroid leaves.

One compact range per (batch, head, selected centroid), reused by all routed
queries. Persistent storage remains shared latent512. Prefix sums and the
live projection size stay on device, including during graph replay.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._mla_projection import combined_kv_weight
from ._paged_common import _lookup_page_id
from .aiter_mla_prefill_attention import _workspace_tensor
from .quantized_latent_load import load_latent
from .kimi_compact_leaf_projection import (
    _selected_row_counts, _head_tile_counts, _projection_workers,
)


@triton.jit
def _project_tiles(
    source, weight, starts, head_tiles, slot_pages, overflow_keys,
    overflow_values, overflow_used, page_indices, output_k, output_v,
    scales, sums, sum_scales, page_counts,
    TOKENS, HEADS: tl.constexpr, SLOTS, OUTPUT_ROWS,
    SOURCE_BATCH_STRIDE: tl.constexpr, SOURCE_TOKEN_STRIDE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr, PAGE_CAPACITY: tl.constexpr,
    INLINE_PAGES: tl.constexpr, HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr, HEAD_ROWS: tl.constexpr,
    HEAD_SEARCH_STEPS: tl.constexpr, SLOT_SEARCH_STEPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    QUANTIZED: tl.constexpr, QUANT_GROUP: tl.constexpr, INT8_SUMS: tl.constexpr,
):
    total_tiles = tl.load(head_tiles + HEAD_ROWS)
    lanes = tl.arange(0, BLOCK_M)
    channels = tl.arange(0, 128)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    for tile in tl.range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        lower = tl.full((), 0, tl.int32)
        upper = HEAD_ROWS
        for _ in tl.static_range(HEAD_SEARCH_STEPS):
            searching = lower < upper
            middle = (lower + upper) // 2
            end = tl.load(head_tiles + middle + 1, mask=searching, other=total_tiles)
            right = searching & (end <= tile)
            lower = tl.where(right, middle + 1, lower)
            upper = tl.where(searching & ~right, middle, upper)
        head_row = lower
        head, batch = head_row % HEADS, head_row // HEADS
        begin = tl.load(starts + head_row * SLOTS)
        end = tl.load(starts + (head_row + 1) * SLOTS)
        row = begin + (tile - tl.load(head_tiles + head_row)) * BLOCK_M + lanes
        valid = (row < end) & (row < OUTPUT_ROWS)
        # Tiles can cross centroid boundaries, never head boundaries. This
        # avoids rounding every short centroid up to a separate GEMM tile.
        lower = tl.full((BLOCK_M,), 0, tl.int32)
        upper = tl.full((BLOCK_M,), SLOTS, tl.int32)
        for _ in tl.static_range(SLOT_SEARCH_STEPS):
            searching = valid & (lower < upper)
            middle = (lower + upper) // 2
            slot_end = tl.load(starts + head_row * SLOTS + middle + 1,
                               mask=searching, other=end)
            right = searching & (slot_end <= row)
            lower = tl.where(right, middle + 1, lower)
            upper = tl.where(searching & ~right, middle, upper)
        slot = tl.where(valid, lower, 0)
        expert_begin = tl.load(starts + head_row * SLOTS + slot, mask=valid, other=0)
        ordinal = row - expert_begin
        page = _lookup_page_id(
            slot_pages, overflow_keys, overflow_values, overflow_used,
            batch, slot, ordinal // 16, valid,
            STATE_CAPACITY, INLINE_PAGES, PAGE_CAPACITY, HASH_CAPACITY, HASH_PROBES,
        ).to(tl.int64)
        valid &= (page >= 0) & (page < PAGE_CAPACITY)
        page = tl.where(valid, page, 0)
        leaf = tl.load(page_indices + (batch * PAGE_CAPACITY + page) * 16 + ordinal % 16,
                       mask=valid, other=0).to(tl.int64)
        valid &= (leaf >= 0) & (leaf < TOKENS)
        leaf = tl.where(valid, leaf, 0)
        source_row = batch * SOURCE_BATCH_STRIDE + leaf * SOURCE_TOKEN_STRIDE
        result = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for step in tl.range(4, num_stages=1):
            latent_channel = step * 128 + channels
            if QUANTIZED:
                latent = load_latent(source, scales, sums, sum_scales, page_counts,
                    batch, page, leaf, latent_channel, valid,
                    TOKENS, PAGE_CAPACITY, 512, QUANT_GROUP, INT8_SUMS)
            else:
                latent = tl.load(source + source_row[:, None] + latent_channel[None, :],
                                 mask=valid[:, None], other=0.)
            weights = tl.load(weight + latent_channel[:, None] * (HEADS * 512)
                              + head * 512 + columns[None, :])
            result += tl.dot(latent, weights)
        # Two contiguous arenas satisfy the expert attention addressing
        # contract, unlike strided views of an interleaved K/V projection.
        if tl.program_id(1) < 256 // BLOCK_N:
            tl.store(output_k + row.to(tl.int64)[:, None] * 256 + columns[None, :],
                     result, mask=valid[:, None])
        else:
            tl.store(output_v + row.to(tl.int64)[:, None] * 256 + columns[None, :] - 256,
                     result, mask=valid[:, None])


def project_compact_glm_leaves(source, uk, uv, cache, head_counts, *,
                               active_slots, hash_probes, buffers=None,
                               block_m=64, block_n=128, num_warps=4):
    """Return contiguous projected K/V and global per-expert row offsets.

    Capacity is a reusable worst-case bound (all leaves for every head), not
    selected-only allocated VRAM. Only the live union is projected/written.
    Counts must describe the exact post-cap routes, not the raw top-eight.
    """
    batch, shared_heads, tokens, dim = source.shape
    quantized = bool(cache.get("quantization_finalized", False))
    quant_sums = bool(cache.get("summary_quantization_finalized", False))
    if quantized:
        tokens = int(cache["quantized_leaf_k"].size(2))
    elif tokens < int(cache.get("leaf_count", 0)):
        raise ValueError("GLM projection cannot read a BF16 sentinel as a live leaf archive")
    heads = uk.size(0)
    if (shared_heads != 1 or dim != 512 or tuple(uk.shape) != (heads, 256, 512)
            or tuple(uv.shape) != (heads, 512, 256)):
        raise ValueError("GLM compact projection requires shared latent512/K256/V256")
    if (not source.is_cuda or source.dtype != torch.bfloat16
            or uk.dtype != source.dtype or uv.dtype != source.dtype
            or uk.device != source.device or uv.device != source.device
            or source.stride(-1) != 1):
        raise ValueError("GLM projection requires same-device BF16 with contiguous latent channels")
    lengths = cache["slot_lengths"]
    experts = batch * heads * active_slots
    if (tuple(lengths.shape[:2]) != (batch, 1) or not 0 < active_slots <= lengths.size(-1)
            or head_counts.numel() != experts or head_counts.dtype != torch.int32
            or head_counts.device != source.device or not head_counts.is_contiguous()):
        raise ValueError("GLM projection route/directory geometry differs")
    if block_m not in (16, 32, 64, 128) or block_n not in (64, 128) or num_warps not in (4, 8):
        raise ValueError("unsupported GLM projection tile")
    capacity = batch * heads * tokens
    if capacity >= 2**31:
        raise ValueError("GLM compact projection exceeds int32 prefix capacity")
    def workspace(name, shape, dtype=torch.int32):
        return _workspace_tensor(buffers, "glm_compact_" + name, shape, dtype=dtype, device=source.device)
    row_counts, starts = workspace("row_counts", (experts + 1,)), workspace("starts", (experts + 1,))
    _selected_row_counts[(triton.cdiv(experts, 256),)](
        head_counts, lengths, row_counts, experts, heads, active_slots,
        STATE_CAPACITY=lengths.size(-1), BLOCK_M=block_m, BLOCK=256, num_warps=4)
    torch.cumsum(row_counts, 0, dtype=torch.int32, out=starts)
    tile_counts = workspace("tile_counts", (batch * heads + 1,))
    head_tiles = workspace("head_tiles", (batch * heads + 1,))
    _head_tile_counts[(triton.cdiv(batch * heads, 256),)](
        starts, tile_counts, batch * heads, active_slots,
        BLOCK_M=block_m, BLOCK=256, num_warps=4)
    torch.cumsum(tile_counts, 0, dtype=torch.int32, out=head_tiles)
    k, v = (workspace(name, (batch, heads, tokens, 256), source.dtype) for name in ("k", "v"))
    directory, indices = cache["slot_pages"], cache["page_indices"]
    workers = min(_projection_workers(source.device.index), batch * heads * triton.cdiv(tokens, block_m))
    _project_tiles[(workers, 512 // block_n)](
        cache["quantized_leaf_k"] if quantized else source,
        combined_kv_weight(uk, uv, buffers), starts, head_tiles, directory,
        cache["overflow_page_keys"], cache["overflow_page_values"], cache["overflow_used"],
        indices, k, v, cache.get("page_k_scales", source),
        cache.get("quantized_page_sum_k" if quant_sums else "page_sum_k", source),
        cache.get("page_sum_k_scales", source), cache.get("page_counts", source),
        tokens, heads, active_slots, capacity,
        SOURCE_BATCH_STRIDE=source.stride(0), SOURCE_TOKEN_STRIDE=source.stride(2),
        STATE_CAPACITY=lengths.size(-1), PAGE_CAPACITY=indices.size(2),
        INLINE_PAGES=directory.size(3), HASH_CAPACITY=cache["overflow_page_values"].size(2),
        HASH_PROBES=hash_probes, HEAD_ROWS=batch * heads,
        HEAD_SEARCH_STEPS=(batch * heads).bit_length(), SLOT_SEARCH_STEPS=active_slots.bit_length(),
        BLOCK_M=block_m, BLOCK_N=block_n,
        QUANTIZED=quantized, QUANT_GROUP=512 // cache["page_k_scales"].size(-1) if quantized else 32,
        INT8_SUMS=quant_sums, num_warps=num_warps, num_stages=1)
    return k, v, starts
