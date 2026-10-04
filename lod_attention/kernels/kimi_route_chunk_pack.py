"""Atomics-free fixed-query-fragment packing for K3 centroid-tile rescoring.

Each tile/query fragment owns a fixed scratch range. A device-side prefix
then lists only its populated query workgroups. Selection and attention math
are unchanged; no CPU readback or per-route atomic reservation is needed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .aiter_prefill_attention import _workspace_tensor


@triton.jit
def _pack_fragments(selected, rows, counts, Q, TILES,
                    CHUNKS: tl.constexpr, PACK_Q: tl.constexpr):
    head, tile, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query = chunk * PACK_Q + tl.arange(0, PACK_Q)
    rank = tl.arange(0, 8)
    chosen = tl.load(selected + (head * Q + query[:, None]) * 8 + rank[None, :],
                     mask=query[:, None] < Q, other=-1)
    route = tl.max(tl.where(chosen == tile, rank[None, :], -1), axis=1)
    valid = (query < Q) & (route >= 0)
    local = tl.cumsum(valid.to(tl.int32)) - 1
    fragment = (head * TILES + tile) * CHUNKS + chunk
    tl.store(rows + fragment * PACK_Q + local,
             (head * Q + query) * 8 + route, mask=valid)
    tl.store(counts + fragment, tl.sum(valid.to(tl.int32)))


@triton.jit
def _prefix_fragment_work(counts, starts, FRAGMENTS: tl.constexpr,
                          BLOCK: tl.constexpr, BLOCK_M: tl.constexpr):
    fragment = tl.arange(0, BLOCK)
    count = tl.load(counts + fragment, mask=fragment < FRAGMENTS, other=0)
    blocks = (count + BLOCK_M - 1) // BLOCK_M
    cumulative = tl.cumsum(blocks)
    tl.store(starts + fragment, cumulative - blocks, mask=fragment < FRAGMENTS)
    tl.store(starts + FRAGMENTS, tl.sum(blocks))


@triton.jit
def _list_fragment_work(starts, work, FRAGMENTS: tl.constexpr,
                        MAX_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    fragment = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rank = tl.arange(0, MAX_BLOCKS)
    begin = tl.load(starts + fragment, mask=fragment < FRAGMENTS, other=0)
    end = tl.load(starts + fragment + 1, mask=fragment < FRAGMENTS, other=0)
    program = begin[:, None] + rank[None, :]
    tl.store(work + program, fragment[:, None],
             mask=(fragment[:, None] < FRAGMENTS) & (program < end[:, None]))


def pack_chunked_kimi_tile_queries(selected, *, tiles, block_m, buffers=None, pack_q=256,
                                   compact_work=True):
    """Return fixed query-fragment rows and compact rescoring work metadata."""
    batch, heads, queries, routes = selected.shape
    if routes != 8 or not selected.is_cuda or not selected.is_contiguous():
        raise ValueError("K3 fragment packing requires contiguous CUDA B,H,Q,8 routes")
    if pack_q % block_m or block_m not in (16, 32, 64, 128):
        raise ValueError("fragment query size must be divisible by the rescoring tile")
    chunks = triton.cdiv(queries, pack_q)
    fragments = batch * heads * tiles * chunks
    # The experiment targets K3's small fixed centroid-tile vocabulary. Keep
    # one prefix kernel bounded; larger layouts retain the ordinary packer.
    if fragments > 32768:
        raise ValueError("K3 fragment experiment supports at most 32768 fragments")
    rows = _workspace_tensor(buffers, "kimi_fragment_rows", (fragments * pack_q,),
                             dtype=torch.int32, device=selected.device)
    counts = _workspace_tensor(buffers, "kimi_fragment_counts", (fragments,),
                               dtype=torch.int32, device=selected.device)
    _pack_fragments[(batch * heads, tiles, chunks)](
        selected, rows, counts, queries, tiles, CHUNKS=chunks, PACK_Q=pack_q, num_warps=4)
    if not compact_work:
        return rows, counts, None, None, fragments, chunks
    starts = _workspace_tensor(buffers, "kimi_fragment_starts", (fragments + 1,),
                               dtype=torch.int32, device=selected.device)
    max_blocks = triton.cdiv(selected.numel(), block_m) + fragments
    work = _workspace_tensor(buffers, "kimi_fragment_work", (max_blocks,),
                             dtype=torch.int32, device=selected.device)
    _prefix_fragment_work[(1,)](
        counts, starts, FRAGMENTS=fragments, BLOCK=triton.next_power_of_2(fragments),
        BLOCK_M=block_m, num_warps=8)
    _list_fragment_work[(triton.cdiv(fragments, 64),)](
        starts, work, FRAGMENTS=fragments, MAX_BLOCKS=pack_q // block_m,
        BLOCK=64, num_warps=4)
    return rows, counts, starts, work, max_blocks, chunks
