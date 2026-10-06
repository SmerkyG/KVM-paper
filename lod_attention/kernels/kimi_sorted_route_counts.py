"""Experimental atomics-free route ordinals from fixed-size local sorts.

Only metadata changes. Every valid route keeps its centroid, and offsets are
unique dense ordinals within each query-head/centroid. The ordinary scatter
and exact leaf attention consume these same counts/offsets unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .paged_prefill import _workspace_tensor


@triton.jit
def _segmented_count(left_slot, left_count, right_slot, right_count):
    return right_slot, tl.where(left_slot == right_slot, left_count + right_count, right_count)


@triton.jit
def _sort_count_chunks(routes, fragment_counts, offsets, ITEMS, STATES, CHUNKS,
                       BLOCK: tl.constexpr, SHIFT: tl.constexpr):
    head, chunk = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    index = chunk * BLOCK + lane
    slot = tl.load(routes + head * ITEMS + index, mask=index < ITEMS, other=-1).to(tl.int32)
    valid = (index < ITEMS) & (slot >= 0) & (slot < STATES)
    packed = tl.where(valid, (slot << SHIFT) | lane, 0x7fffffff)
    ordered = tl.sort(packed, descending=False)
    slot = ordered >> SHIFT
    source = ordered & (BLOCK - 1)
    valid = ordered != 0x7fffffff
    _, rank = tl.associative_scan((slot, tl.full((BLOCK,), 1, tl.int32)),
                                 axis=0, combine_fn=_segmented_count)
    successor = tl.gather(ordered, tl.minimum(lane + 1, BLOCK - 1), axis=0) >> SHIFT
    final = valid & ((lane == BLOCK - 1) | (successor != slot))
    tl.store(fragment_counts + (head * STATES + slot) * CHUNKS + chunk, rank, mask=final)
    tl.store(offsets + head * ITEMS + chunk * BLOCK + source, rank - 1, mask=valid)


@triton.jit
def _prefix_fragments(fragment_counts, fragment_offsets, head_counts, STATES, CHUNKS,
                       BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    head = tl.program_id(0)
    slot = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
    chunk = tl.arange(0, BLOCK_C)
    ptr = fragment_counts + (head * STATES + slot[:, None]) * CHUNKS + chunk[None, :]
    values = tl.load(ptr, mask=(slot[:, None] < STATES) & (chunk[None, :] < CHUNKS), other=0)
    prefix = tl.cumsum(values, axis=1)
    output = fragment_offsets + (head * STATES + slot[:, None]) * CHUNKS + chunk[None, :]
    # Keep raw counts immutable: both the reduction and scan use them, while
    # the ordinal pass consumes the separate prefix output. This avoids an
    # in-place read/write alias in the count reduction.
    tl.store(output, prefix - values, mask=(slot[:, None] < STATES) & (chunk[None, :] < CHUNKS))
    tl.store(head_counts + head * STATES + slot, tl.sum(values, axis=1), mask=slot < STATES)


@triton.jit
def _finish_ordinals(routes, fragment_counts, offsets, ITEMS, STATES, CHUNKS,
                      CHUNK_ITEMS: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ptr = head * ITEMS + index
    slot = tl.load(routes + ptr, mask=index < ITEMS, other=-1).to(tl.int32)
    valid = (index < ITEMS) & (slot >= 0) & (slot < STATES)
    local = tl.load(offsets + ptr, mask=valid, other=0)
    start = tl.load(fragment_counts + (head * STATES + slot) * CHUNKS + index // CHUNK_ITEMS,
                    mask=valid, other=0)
    tl.store(offsets + ptr, local + start, mask=valid)


def count_sorted_kimi_routes(top_slots, *, active_slots, buffers=None, chunk_items=2048):
    if top_slots.ndim != 4 or not top_slots.is_contiguous() or not top_slots.is_cuda:
        raise ValueError("sorted counts require contiguous CUDA BHTR routes")
    if chunk_items not in (512, 1024, 2048):
        raise ValueError("sorted route chunks must contain 512, 1024 or 2048 items")
    if not 0 < active_slots < (1 << (31 - (chunk_items.bit_length() - 1))):
        raise ValueError("centroid count exceeds packed route-sort capacity")
    batch, heads, queries, routes = top_slots.shape
    items = queries * routes
    chunks = triton.cdiv(items, chunk_items)
    counts = _workspace_tensor(buffers, "sorted_leaf_head_counts", (batch * heads * active_slots,),
                               dtype=torch.int32, device=top_slots.device)
    offsets = _workspace_tensor(buffers, "sorted_leaf_offsets", tuple(top_slots.shape),
                                dtype=torch.int32, device=top_slots.device)
    fragments = _workspace_tensor(buffers, "sorted_leaf_fragments",
                                  (batch * heads * active_slots, chunks),
                                  dtype=torch.int32, device=top_slots.device)
    fragment_offsets = _workspace_tensor(buffers, "sorted_leaf_fragment_offsets",
                                         tuple(fragments.shape),
                                         dtype=torch.int32, device=top_slots.device)
    fragments.zero_()
    _sort_count_chunks[(batch * heads, chunks)](
        top_slots, fragments, offsets, items, active_slots, chunks,
        BLOCK=chunk_items, SHIFT=chunk_items.bit_length() - 1, num_warps=4)
    _prefix_fragments[(batch * heads, triton.cdiv(active_slots, 16))](
        fragments, fragment_offsets, counts, active_slots, chunks, BLOCK_S=16,
        BLOCK_C=triton.next_power_of_2(chunks), num_warps=4)
    _finish_ordinals[(batch * heads, triton.cdiv(items, 256))](
        top_slots, fragment_offsets, offsets, items, active_slots, chunks,
        CHUNK_ITEMS=chunk_items, BLOCK=256, num_warps=4)
    return counts, offsets
