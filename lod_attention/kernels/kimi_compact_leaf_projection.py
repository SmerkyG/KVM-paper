"""Project the union of selected Kimi leaves into contiguous expert ranges.

Routes and centroid ownership are unchanged. A device prefix scan assigns
each selected (batch, head, centroid) one range, shared by all its queries.
Persistent projection workers read the device total; no CPU count/size read
or per-query projection is required. The reusable allocation is a worst-case
bound, not a promise of selected-only VRAM: only its live prefix is written.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from ._paged_common import _lookup_page_id
from .aiter_mla_prefill_attention import _workspace_tensor


def compact_row_capacity(tokens: int, slots: int, block_m: int) -> int:
    """Per-head bound: each leaf occurs once; centroid ranges have no padding."""
    if tokens < 1 or slots < 1 or block_m not in (16, 32, 64):
        raise ValueError("compact projection requires positive sizes and 16/32/64-row tiles")
    return tokens


@triton.jit
def _selected_row_counts(counts, lengths, rows, EXPERTS, HEADS, SLOTS,
                         STATE_CAPACITY: tl.constexpr, BLOCK_M: tl.constexpr,
                         BLOCK: tl.constexpr):
    expert = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = expert < EXPERTS
    head = expert // SLOTS
    slot = expert % SLOTS
    count = tl.load(counts + expert, mask=valid, other=0)
    length = tl.load(lengths + (head // HEADS) * STATE_CAPACITY + slot,
                     mask=valid, other=0)
    tl.store(rows + expert + 1, tl.where(count > 0, length, 0), mask=valid)
    if tl.program_id(0) == 0:
        tl.store(rows, 0)


@triton.jit
def _head_tile_counts(starts, tiles, HEAD_ROWS, SLOTS,
                      BLOCK_M: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = head < HEAD_ROWS
    begin = tl.load(starts + head * SLOTS, mask=valid, other=0)
    end = tl.load(starts + (head + 1) * SLOTS, mask=valid, other=0)
    tl.store(tiles + head + 1, tl.cdiv(end - begin, BLOCK_M), mask=valid)
    if tl.program_id(0) == 0:
        tl.store(tiles, 0)


@triton.jit
def _project_compact_tiles(
    source, uk, uv, starts, head_tiles, slot_pages, overflow_keys, overflow_values,
    overflow_used, page_indices, output_k, output_v,
    TOKENS, HEADS, SLOTS, OUTPUT_ROWS,
    SOURCE_BATCH_STRIDE: tl.constexpr, SOURCE_TOKEN_STRIDE: tl.constexpr,
    UK_HEAD_STRIDE: tl.constexpr, UK_ROW_STRIDE: tl.constexpr,
    UK_LATENT_STRIDE: tl.constexpr, UV_HEAD_STRIDE: tl.constexpr,
    UV_LATENT_STRIDE: tl.constexpr, UV_VALUE_STRIDE: tl.constexpr,
    STATE_CAPACITY: tl.constexpr, PAGE_CAPACITY: tl.constexpr,
    INLINE_PAGES: tl.constexpr, HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr, PAGE_SIZE: tl.constexpr,
    HEAD_ROWS: tl.constexpr, HEAD_SEARCH_STEPS: tl.constexpr,
    SLOT_SEARCH_STEPS: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # A fixed grid consumes just the live compact tiles. Empty experts neither
    # project leaves nor require their own workgroup; zeros in the prefix are
    # resolved by upper-bound search.
    total_tiles = tl.load(head_tiles + HEAD_ROWS)
    programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK_M)
    channels = tl.arange(0, 128)
    output_lane = tl.program_id(1) * 64 + tl.arange(0, 64)
    for tile in tl.range(tl.program_id(0), total_tiles, programs):
        lower = tl.full((), 0, tl.int32)
        upper = HEAD_ROWS
        for _ in tl.static_range(0, HEAD_SEARCH_STEPS):
            searching = lower < upper
            middle = (lower + upper) // 2
            end = tl.load(head_tiles + middle + 1, mask=searching, other=total_tiles)
            right = searching & (end <= tile)
            lower = tl.where(right, middle + 1, lower)
            upper = tl.where(searching & ~right, middle, upper)
        head_row = lower
        head = head_row % HEADS
        batch = head_row // HEADS
        head_begin = tl.load(starts + head_row * SLOTS)
        head_end = tl.load(starts + (head_row + 1) * SLOTS)
        local_tile = tile - tl.load(head_tiles + head_row)
        row = head_begin + local_tile * BLOCK_M + lane
        valid = (row < head_end) & (row < OUTPUT_ROWS)
        # A projection tile may cross centroid boundaries, but never a head
        # boundary. Its rows share one projection matrix, eliminating all
        # per-centroid MFMA padding without changing attention's leaf order.
        lower = tl.full((BLOCK_M,), 0, tl.int32)
        upper = tl.full((BLOCK_M,), SLOTS, tl.int32)
        for _ in tl.static_range(0, SLOT_SEARCH_STEPS):
            searching = valid & (lower < upper)
            middle = (lower + upper) // 2
            end = tl.load(starts + head_row * SLOTS + middle + 1,
                          mask=searching, other=head_end)
            right = searching & (end <= row)
            lower = tl.where(right, middle + 1, lower)
            upper = tl.where(searching & ~right, middle, upper)
        slot = tl.where(valid, lower, 0)
        expert_begin = tl.load(starts + head_row * SLOTS + slot, mask=valid, other=0)
        ordinal = row - expert_begin
        page = _lookup_page_id(
            slot_pages, overflow_keys, overflow_values, overflow_used,
            batch, slot, ordinal // PAGE_SIZE, valid,
            STATE_CAPACITY, INLINE_PAGES, PAGE_CAPACITY, HASH_CAPACITY, HASH_PROBES,
        ).to(tl.int64)
        valid &= (page >= 0) & (page < PAGE_CAPACITY)
        page = tl.where(valid, page, 0)
        leaf = tl.load(page_indices + (batch * PAGE_CAPACITY + page) * PAGE_SIZE
                       + ordinal % PAGE_SIZE, mask=valid, other=0).to(tl.int64)
        valid &= (leaf >= 0) & (leaf < TOKENS)
        leaf = tl.where(valid, leaf, 0)
        source_row = batch * SOURCE_BATCH_STRIDE + leaf * SOURCE_TOKEN_STRIDE
        key = tl.zeros((BLOCK_M, 64), tl.float32)
        value = tl.zeros((BLOCK_M, 64), tl.float32)
        for latent_step in tl.range(0, 4, num_stages=1):
            latent_begin = latent_step * 128
            latent_lane = latent_begin + channels
            latent = tl.load(source + source_row[:, None] + latent_lane[None, :],
                             mask=valid[:, None], other=0.0)
            key_weight = tl.load(uk + head * UK_HEAD_STRIDE
                                 + latent_lane[:, None] * UK_LATENT_STRIDE
                                 + output_lane[None, :] * UK_ROW_STRIDE)
            value_weight = tl.load(uv + head * UV_HEAD_STRIDE
                                   + latent_lane[:, None] * UV_LATENT_STRIDE
                                   + output_lane[None, :] * UV_VALUE_STRIDE)
            key += tl.dot(latent, key_weight)
            value += tl.dot(latent, value_weight)
        row = row.to(tl.int64)
        tl.store(output_k + row[:, None] * 192 + output_lane[None, :], key, mask=valid[:, None])
        tl.store(output_v + row[:, None] * 128 + output_lane[None, :], value, mask=valid[:, None])
        if tl.program_id(1) == 0:
            direct_lane = tl.arange(0, 64)
            direct = tl.load(source + source_row[:, None] + 512 + direct_lane[None, :],
                             mask=valid[:, None], other=0.0)
            tl.store(output_k + row[:, None] * 192 + 128 + direct_lane[None, :],
                     direct, mask=valid[:, None])


@functools.lru_cache(maxsize=16)
def _projection_workers(device_index: int) -> int:
    return 2 * torch.cuda.get_device_properties(device_index).multi_processor_count


def project_compact_kimi_leaves(
    key: torch.Tensor, w_uk_t: torch.Tensor, w_uv: torch.Tensor,
    page_cache: dict, head_counts: torch.Tensor, *, active_slots: int,
    hash_probes: int, buffers: dict[str, torch.Tensor] | None = None,
    block_m: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return projected K/V and device-only per-expert compact row starts.

    The flat K/V allocation is exposed as a BHTD view for the existing expert
    packer. The consumer must use ``compact_leaf_offsets``: its rows are global
    compact addresses, NOT chronological leaf indices or fixed head ranges.
    ``head_counts`` must describe the exact post-cap routes being consumed.
    Each original leaf belongs to at most one centroid in the shared directory.
    """
    if key.ndim != 4 or key.size(1) != 1 or key.size(-1) != 576:
        raise ValueError("compact Kimi projection expects [B,1,T,576]")
    batch, tokens, heads = int(key.size(0)), int(key.size(2)), int(w_uk_t.size(0))
    if tuple(w_uk_t.shape) != (heads, 128, 512) or tuple(w_uv.shape) != (heads, 512, 128):
        raise ValueError("compact Kimi projection weights have incompatible geometry")
    if tuple(page_cache["slot_lengths"].shape[:2]) != (batch, 1):
        raise ValueError("compact Kimi projection requires one shared latent directory")
    state_capacity = int(page_cache["slot_lengths"].size(-1))
    if not 0 < active_slots <= state_capacity or head_counts.numel() != batch * heads * active_slots:
        raise ValueError("compact projection route counts have incompatible geometry")
    if not key.is_cuda or key.dtype != torch.bfloat16:
        raise ValueError("compact projection requires CUDA/ROCm BF16 inputs")
    if (head_counts.dtype != torch.int32 or head_counts.device != key.device
            or not head_counts.is_contiguous()
            or w_uk_t.dtype != key.dtype or w_uv.dtype != key.dtype
            or w_uk_t.device != key.device or w_uv.device != key.device):
        raise ValueError("compact projection inputs must share the GPU; counts must be contiguous int32")
    capacity = compact_row_capacity(tokens, active_slots, block_m)
    experts = batch * heads * active_slots
    if batch * heads * capacity >= 2**31:
        raise ValueError("compact projection exceeds int32 prefix capacity")
    rows = _workspace_tensor(buffers, "kimi_compact_row_counts", (experts + 1,),
                             dtype=torch.int32, device=key.device)
    starts = _workspace_tensor(buffers, "kimi_compact_row_starts", (experts + 1,),
                               dtype=torch.int32, device=key.device)
    _selected_row_counts[(triton.cdiv(experts, 256),)](
        head_counts, page_cache["slot_lengths"], rows, experts, heads, active_slots,
        STATE_CAPACITY=state_capacity, BLOCK_M=block_m, BLOCK=256, num_warps=4,
    )
    torch.cumsum(rows, dim=0, dtype=torch.int32, out=starts)
    tile_counts = _workspace_tensor(buffers, "kimi_compact_head_tile_counts", (batch * heads + 1,),
                                    dtype=torch.int32, device=key.device)
    head_tiles = _workspace_tensor(buffers, "kimi_compact_head_tiles", (batch * heads + 1,),
                                   dtype=torch.int32, device=key.device)
    _head_tile_counts[(triton.cdiv(batch * heads, 256),)](
        starts, tile_counts, batch * heads, active_slots,
        BLOCK_M=block_m, BLOCK=256, num_warps=4)
    torch.cumsum(tile_counts, dim=0, dtype=torch.int32, out=head_tiles)
    output_k = _workspace_tensor(buffers, "kimi_compact_projected_k", (batch, heads, capacity, 192),
                                 dtype=key.dtype, device=key.device)
    output_v = _workspace_tensor(buffers, "kimi_compact_projected_v", (batch, heads, capacity, 128),
                                 dtype=key.dtype, device=key.device)
    directory = page_cache["slot_pages"]
    indices = page_cache["page_indices"]
    workers = min(_projection_workers(key.device.index), batch * heads * triton.cdiv(capacity, block_m))
    _project_compact_tiles[(workers, 2)](
        key, w_uk_t, w_uv, starts, head_tiles, directory, page_cache["overflow_page_keys"],
        page_cache["overflow_page_values"], page_cache["overflow_used"], indices,
        output_k, output_v, tokens, heads, active_slots, batch * heads * capacity,
        SOURCE_BATCH_STRIDE=int(key.stride(0)), SOURCE_TOKEN_STRIDE=int(key.stride(2)),
        UK_HEAD_STRIDE=int(w_uk_t.stride(0)), UK_ROW_STRIDE=int(w_uk_t.stride(1)),
        UK_LATENT_STRIDE=int(w_uk_t.stride(2)), UV_HEAD_STRIDE=int(w_uv.stride(0)),
        UV_LATENT_STRIDE=int(w_uv.stride(1)), UV_VALUE_STRIDE=int(w_uv.stride(2)),
        STATE_CAPACITY=state_capacity, PAGE_CAPACITY=int(indices.size(2)),
        INLINE_PAGES=int(directory.size(3)), HASH_CAPACITY=int(page_cache["overflow_page_values"].size(2)),
        HASH_PROBES=hash_probes, PAGE_SIZE=16, HEAD_ROWS=batch * heads,
        HEAD_SEARCH_STEPS=(batch * heads).bit_length(),
        SLOT_SEARCH_STEPS=active_slots.bit_length(),
        BLOCK_M=block_m, num_warps=4, num_stages=1,
    )
    return output_k, output_v, starts
