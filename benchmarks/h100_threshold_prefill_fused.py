"""Experimental fused threshold routing and coarse attention for H100 prefill."""
from __future__ import annotations

import math
import os
import time

import torch
import triton
import triton.language as tl

from lod_attention.kernels.aiter_prefill_attention import _workspace_tensor
from lod_attention.kernels.lod_kernels import merge_attention_branches_with_sink
from lod_attention.kernels._paged_common import _lookup_page_id


@triton.jit
def _prepare_state(K, V, Counts, MeanK, MeanV, S: tl.constexpr,
                   CAP: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    kh = row // S
    slot = row % S
    d = tl.arange(0, D)
    count = tl.load(Counts + kh * CAP + slot).to(tl.bfloat16).to(tl.float32)
    divisor = tl.maximum(count, 1.0)
    k = tl.load(K + (kh * CAP + slot) * D + d).to(tl.float32)
    v = tl.load(V + (kh * CAP + slot) * D + d).to(tl.float32)
    tl.store(MeanK + row * D + d, k / divisor)
    tl.store(MeanV + row * D + d, v / divisor)


@triton.jit
def _region_totals(Lengths, Remote, NRegions, S: tl.constexpr, CAP: tl.constexpr,
                   BLOCK: tl.constexpr):
    kh = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    sizes = tl.load(Lengths + kh * CAP + i, mask=i < S, other=0).to(tl.int64)
    tl.store(Remote + kh, tl.sum(sizes, 0))
    tl.store(NRegions + kh, tl.sum((sizes > 0).to(tl.int32), 0))


def _sample_selected(selected, slots, counts, *, bitmask, direct_pack, h, qlen, n, state_len, width):
    """Recover a small diagnostic selection sample; not used in timed paths."""
    if direct_pack:
        chosen = slots.reshape(h, qlen, width)[:, :n, :].to(torch.int64)
        valid = torch.arange(width, device=slots.device)[None, None, :] < counts[:, :n, None]
        result = torch.zeros((h, n, state_len), dtype=torch.int32, device=slots.device)
        result.scatter_add_(2, torch.where(valid, chosen, 0), valid.to(torch.int32))
        return result.reshape(1, h, n, state_len).bool()
    if bitmask:
        idx = torch.arange(state_len, device=selected.device)
        return ((selected[:, :n, :].to(torch.int32).index_select(-1, idx // 8) >> (idx % 8)) & 1).reshape(1, h, n, state_len).bool()
    return selected[:, :n, :].reshape(1, h, n, state_len).bool()


@triton.jit
def _route_logits_with_lse(Q, MeanK, Lengths, Logits, PartialLSE,
                           QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr,
                           H: tl.constexpr, KV: tl.constexpr, G: tl.constexpr,
                           D: tl.constexpr, SCALE: tl.constexpr,
                           M: tl.constexpr, N: tl.constexpr, TILES: tl.constexpr):
    bh = tl.program_id(0)
    row = tl.program_id(1) * M + tl.arange(0, M)
    tile = tl.program_id(2)
    col = tile * N + tl.arange(0, N)
    d = tl.arange(0, D)
    kh = (bh // H) * KV + (bh % H) // G
    q = tl.load(Q + (bh * QLEN + row[:, None]) * D + d[None, :],
                mask=row[:, None] < QLEN, other=0)
    k = tl.load(MeanK + (kh * S + col[None, :]) * D + d[:, None],
                mask=col[None, :] < S, other=0)
    dot = tl.dot(q, k).to(tl.bfloat16)
    tl.store(Logits + (bh * QLEN + row[:, None]) * S + col[None, :], dot,
             mask=(row[:, None] < QLEN) & (col[None, :] < S))
    sizes = tl.load(Lengths + kh * CAP + col, mask=col < S, other=0).to(tl.float32)
    valid = (col < S) & (sizes > 0)
    scores = tl.where(valid[None, :], dot.to(tl.float32) * SCALE +
                      tl.log(tl.maximum(sizes, 1.0))[None, :], -float('inf'))
    mx = tl.max(scores, 1)
    safe = tl.where(mx != -float('inf'), mx, 0.0)
    mass = tl.sum(tl.where(valid[None, :], tl.exp(scores - safe[:, None]), 0.0), 1)
    lse = tl.where(mass > 0, safe + tl.log(mass), -float('inf'))
    tl.store(PartialLSE + (bh * QLEN + row) * TILES + tile, lse, mask=row < QLEN)


@triton.jit
def _select_and_coarse(Q, RouteLogits, MeanK, MeanV, Lengths, Selected, Slots, RouteCounts, EntryCounts, Out, LSE, DebugScores,
                       PartialLSE, RemoteTotals, RegionTotals, ExpertHeadCounts, ExpertRouteOffsets,
                       QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr,
                       H: tl.constexpr, KV: tl.constexpr, G: tl.constexpr,
                       D: tl.constexpr, SCALE: tl.constexpr,
                       ENTRY: tl.constexpr, MULTIPLIER: tl.constexpr,
                       ENTRY_BOUND: tl.constexpr, WORK_BOUND: tl.constexpr,
                       REGION_CAP: tl.constexpr, M: tl.constexpr, N: tl.constexpr, DEBUG: tl.constexpr,
                       TRACK_ENTRIES: tl.constexpr, FUSE_PACK: tl.constexpr, WIDTH: tl.constexpr,
                       REUSE_SCORES: tl.constexpr, PRECOMPUTED_LSE: tl.constexpr,
                       LSE_TILES: tl.constexpr, LSE_BLOCK: tl.constexpr, BITMASK: tl.constexpr,
                       WRITE_SELECTED: tl.constexpr, MAX_REGIONS: tl.constexpr, FUSE_EXPERT_META: tl.constexpr):
    bh = tl.program_id(0)
    batch = bh // H
    head = bh % H
    kh = batch * KV + head // G
    rows = tl.program_id(1) * M + tl.arange(0, M)
    dims = tl.arange(0, D)
    cols = tl.arange(0, N)
    query = tl.load(Q + (bh * QLEN + rows[:, None]) * D + dims[None, :],
                    mask=rows[:, None] < QLEN, other=0)
    if REUSE_SCORES:
        q_float = query.to(tl.float32)
        inverse_rms = tl.rsqrt(tl.maximum(tl.sum(q_float * q_float, 1) / D, 1.0e-12))
    if PRECOMPUTED_LSE:
        t = tl.arange(0, LSE_BLOCK)
        partial = tl.load(PartialLSE + (bh * QLEN + rows[:, None]) * LSE_TILES + t[None, :],
                          mask=(rows[:, None] < QLEN) & (t[None, :] < LSE_TILES), other=-float('inf'))
        maximum_lse = tl.max(partial, 1)
        safe_lse = tl.where(maximum_lse != -float('inf'), maximum_lse, 0.0)
        mass_lse = tl.sum(tl.exp(partial - safe_lse[:, None]), 1)
        log_z = tl.where(mass_lse > 0, safe_lse + tl.log(mass_lse), float('inf'))
        remote = tl.load(RemoteTotals + kh)
        nregions = tl.load(RegionTotals + kh)
    else:
        max_route = tl.full((M,), -float('inf'), tl.float32)
        sum_route = tl.zeros((M,), tl.float32)
        remote = tl.full((), 0, tl.int64)
        nregions = tl.full((), 0, tl.int32)
        for block in range(tl.cdiv(S, N)):
            idx = block * N + cols
            sizes = tl.load(Lengths + kh * CAP + idx, mask=idx < S, other=0).to(tl.int64)
            valid = (idx < S) & (sizes > 0)
            dot = tl.load(RouteLogits + (bh * QLEN + rows[:, None]) * S + idx[None, :],
                          mask=(rows[:, None] < QLEN) & valid[None, :], other=0).to(tl.float32) * SCALE
            if REUSE_SCORES:
                dot = dot * inverse_rms[:, None]
            score = tl.where(valid[None, :], dot + tl.log(tl.maximum(sizes.to(tl.float32), 1.0))[None, :], -float('inf'))
            mx = tl.maximum(max_route, tl.max(score, 1))
            safe = tl.where(mx != -float('inf'), mx, 0.0)
            correction = tl.where(max_route != -float('inf'), tl.exp(max_route - safe), 0.0)
            sum_route = sum_route * correction + tl.sum(tl.where(valid[None, :], tl.exp(score - safe[:, None]), 0.0), 1)
            max_route = mx
            remote += tl.sum(tl.where(valid, sizes, 0), 0)
            nregions += tl.sum(valid.to(tl.int32), 0)
        log_z = tl.where(sum_route > 0, max_route + tl.log(sum_route), float('inf'))
    if ENTRY:
        cutoff = log_z - tl.log(ENTRY_BOUND * 1.0) - 0.5 * tl.log(tl.maximum(remote.to(tl.float32), 1.0)) + 1.0e-6
    else:
        cutoff = log_z + tl.log(MULTIPLIER / tl.maximum(nregions.to(tl.float32), 1.0))
    candidate_before = tl.zeros((M,), tl.int64)
    selected_count = tl.zeros((M,), tl.int32)
    entry_count = tl.zeros((M,), tl.int32)
    maximum = tl.full((M,), -float('inf'), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    numerator = tl.zeros((M, D), tl.float32)
    for block in range(tl.cdiv(S, N)):
        idx = block * N + cols
        sizes = tl.load(Lengths + kh * CAP + idx, mask=idx < S, other=0).to(tl.int64)
        valid = (idx < S) & (sizes > 0)
        if not REUSE_SCORES:
            keys = tl.load(MeanK + (kh * S + idx[None, :]) * D + dims[:, None],
                           mask=valid[None, :], other=0)
        values = tl.load(MeanV + (kh * S + idx[:, None]) * D + dims[None, :],
                         mask=valid[:, None], other=0)
        log_size = tl.log(tl.maximum(sizes.to(tl.float32), 1.0))
        dot = tl.load(RouteLogits + (bh * QLEN + rows[:, None]) * S + idx[None, :],
                      mask=(rows[:, None] < QLEN) & valid[None, :], other=0).to(tl.float32) * SCALE
        if REUSE_SCORES:
            route = dot * inverse_rms[:, None] + log_size[None, :]
        else:
            route = dot + log_size[None, :]
        if DEBUG:
            tl.store(DebugScores + (bh * QLEN + rows[:, None]) * S + idx[None, :], route,
                     mask=(rows[:, None] < QLEN) & (idx[None, :] < S))
        if ENTRY:
            candidate = valid[None, :] & (route - log_size[None, :] > cutoff[:, None])
            add = tl.where(candidate, sizes[None, :], 0)
            prefix = candidate_before[:, None] + tl.cumsum(add, 1)
            opened = candidate & (prefix * prefix < WORK_BOUND * WORK_BOUND * remote)
            candidate_before += tl.sum(add, 1)
        else:
            opened = valid[None, :] & (sizes[None, :] <= REGION_CAP) & (route > cutoff[:, None])
        if FUSE_PACK or MAX_REGIONS > 0:
            ranks = selected_count[:, None] + tl.cumsum(opened.to(tl.int32), 1) - 1
        if MAX_REGIONS > 0:
            opened &= ranks < MAX_REGIONS
        if FUSE_PACK:
            tl.store(Slots + (bh * QLEN + rows[:, None]) * WIDTH + ranks,
                     idx[None, :].to(tl.int32), mask=opened & (rows[:, None] < QLEN))
            if FUSE_EXPERT_META:
                offset = tl.atomic_add(ExpertHeadCounts + bh * S + idx[None, :] + rows[:, None] * 0, 1,
                                       mask=opened & (rows[:, None] < QLEN), sem='relaxed')
                tl.store(ExpertRouteOffsets + (bh * QLEN + rows[:, None]) * WIDTH + ranks,
                         offset, mask=opened & (rows[:, None] < QLEN))
        selected_count += tl.sum(opened.to(tl.int32), 1)
        if TRACK_ENTRIES:
            entry_count += tl.sum(tl.where(opened, sizes[None, :], 0).to(tl.int32), 1)
        if WRITE_SELECTED and BITMASK:
            bits = tl.arange(0, 8)
            packed_bits = tl.sum(tl.reshape(opened.to(tl.int32), (M, N // 8, 8)) *
                                 (1 << bits[None, None, :]), 2).to(tl.uint8)
            byte_idx = block * (N // 8) + tl.arange(0, N // 8)
            tl.store(Selected + (bh * QLEN + rows[:, None]) * tl.cdiv(S, 8) + byte_idx[None, :],
                     packed_bits, mask=(rows[:, None] < QLEN) & (byte_idx[None, :] < tl.cdiv(S, 8)))
        elif WRITE_SELECTED:
            tl.store(Selected + (bh * QLEN + rows[:, None]) * S + idx[None, :],
                     opened.to(tl.uint8), mask=(rows[:, None] < QLEN) & (idx[None, :] < S))
        if REUSE_SCORES:
            dot_native = dot
        else:
            dot_native = tl.dot(query, keys).to(tl.bfloat16).to(tl.float32) * SCALE
        closed = valid[None, :] & ~opened
        scores = tl.where(closed, dot_native + log_size[None, :], -float('inf'))
        mx = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(mx != -float('inf'), mx, 0.0)
        correction = tl.where(maximum != -float('inf'), tl.exp(maximum - safe), 0.0)
        weight = tl.where(closed, tl.exp(scores - safe[:, None]), 0.0)
        numerator = numerator * correction[:, None] + tl.dot(weight.to(values.dtype), values)
        denominator = denominator * correction + tl.sum(weight, 1)
        maximum = mx
    tl.store(RouteCounts + bh * QLEN + rows, selected_count, mask=rows < QLEN)
    if TRACK_ENTRIES:
        tl.store(EntryCounts + bh * QLEN + rows, entry_count, mask=rows < QLEN)
    tl.store(Out + (bh * QLEN + rows[:, None]) * D + dims[None, :],
             numerator / tl.maximum(denominator[:, None], 1.0e-30), mask=rows[:, None] < QLEN)
    tl.store(LSE + bh * QLEN + rows, tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf')),
             mask=rows < QLEN)


@triton.jit
def _compact(Selected, Slots, Offsets, HeadCounts, QLEN: tl.constexpr,
             S: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    bh = row // QLEN
    idx = tl.arange(0, BLOCK)
    selected = tl.load(Selected + row * S + idx, mask=idx < S, other=0) != 0
    rank = tl.cumsum(selected.to(tl.int32), 0) - 1
    offset = tl.atomic_add(HeadCounts + bh * S + idx, 1, mask=selected, sem='relaxed')
    tl.store(Slots + row * WIDTH + rank, idx.to(tl.int64), mask=selected)
    tl.store(Offsets + row * WIDTH + rank, offset, mask=selected)


@triton.jit
def _compact_grouped(Selected, Slots, Offsets, HeadCounts, QLEN: tl.constexpr,
                     S: tl.constexpr, WIDTH: tl.constexpr,
                     GROUP_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    row = tl.program_id(1) * GROUP_ROWS + tl.arange(0, GROUP_ROWS)
    col = tl.arange(0, BLOCK)
    before = tl.zeros((GROUP_ROWS,), tl.int32)
    for block in range(tl.cdiv(S, BLOCK)):
        idx = block * BLOCK + col
        chosen = tl.load(Selected + (bh * QLEN + row[:, None]) * S + idx[None, :],
                         mask=(row[:, None] < QLEN) & (idx[None, :] < S), other=0) != 0
        integer = chosen.to(tl.int32)
        rank = before[:, None] + tl.cumsum(integer, 1) - 1
        count = tl.sum(integer, 0)
        start = tl.atomic_add(HeadCounts + bh * S + idx, count,
                              mask=(idx < S) & (count > 0), sem='relaxed')
        offset = start[None, :] + tl.cumsum(integer, 0) - 1
        pos = (bh * QLEN + row[:, None]) * WIDTH + rank
        tl.store(Slots + pos, idx[None, :].to(tl.int64), mask=chosen)
        tl.store(Offsets + pos, offset, mask=chosen)
        before += tl.sum(integer, 1)


@triton.jit
def _compact_slots_only(Selected, Slots, QLEN: tl.constexpr, S: tl.constexpr,
                        WIDTH: tl.constexpr, GROUP_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    row = tl.program_id(1) * GROUP_ROWS + tl.arange(0, GROUP_ROWS)
    col = tl.arange(0, BLOCK)
    before = tl.zeros((GROUP_ROWS,), tl.int32)
    for block in range(tl.cdiv(S, BLOCK)):
        idx = block * BLOCK + col
        chosen = tl.load(Selected + (bh * QLEN + row[:, None]) * S + idx[None, :],
                         mask=(row[:, None] < QLEN) & (idx[None, :] < S), other=0) != 0
        integer = chosen.to(tl.int32)
        rank = before[:, None] + tl.cumsum(integer, 1) - 1
        tl.store(Slots + (bh * QLEN + row[:, None]) * WIDTH + rank,
                 idx[None, :].to(tl.int64), mask=chosen)
        before += tl.sum(integer, 1)


@triton.jit
def _compact_slots_packed(Selected, Slots, RouteOffsets, QLEN: tl.constexpr, S: tl.constexpr,
                          GROUP_ROWS: tl.constexpr, BLOCK: tl.constexpr, BITMASK: tl.constexpr):
    bh = tl.program_id(0)
    row = tl.program_id(1) * GROUP_ROWS + tl.arange(0, GROUP_ROWS)
    col = tl.arange(0, BLOCK)
    before = tl.zeros((GROUP_ROWS,), tl.int32)
    base = tl.load(RouteOffsets + bh * QLEN + row, mask=row < QLEN, other=0).to(tl.int64)
    for block in range(tl.cdiv(S, BLOCK)):
        idx = block * BLOCK + col
        if BITMASK:
            byte = tl.load(Selected + (bh * QLEN + row[:, None]) * tl.cdiv(S, 8) + idx[None, :] // 8,
                           mask=(row[:, None] < QLEN) & (idx[None, :] < S), other=0).to(tl.int32)
            chosen = ((byte >> (idx[None, :] % 8)) & 1) != 0
        else:
            chosen = tl.load(Selected + (bh * QLEN + row[:, None]) * S + idx[None, :],
                             mask=(row[:, None] < QLEN) & (idx[None, :] < S), other=0) != 0
        integer = chosen.to(tl.int32)
        rank = before[:, None] + tl.cumsum(integer, 1) - 1
        tl.store(Slots + base[:, None] + rank, idx[None, :].to(tl.int32), mask=chosen)
        before += tl.sum(integer, 1)


@triton.jit
def _direct_leaf_attention(Q, Slots, Counts, Lengths, Roots, HashKeys, Directories, Used,
                           PageIndices, LeafK, LeafV, Out, LSE,
                           QLEN: tl.constexpr, WIDTH: tl.constexpr, S: tl.constexpr,
                           CAP: tl.constexpr, G: tl.constexpr, ROOTS: tl.constexpr,
                           HASH_CAP: tl.constexpr, PAGE_CAP: tl.constexpr, LEAF_CAP: tl.constexpr,
                           PROBES: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                           BLOCK_N: tl.constexpr, SEARCH_STEPS: tl.constexpr):
    row = tl.program_id(0)
    head = row // QLEN
    kh = head // G
    route = tl.arange(0, WIDTH)
    count = tl.load(Counts + row)
    slots = tl.load(Slots + row * WIDTH + route, mask=route < count, other=0).to(tl.int32)
    lengths = tl.load(Lengths + kh * CAP + slots, mask=route < count, other=0).to(tl.int32)
    ends = tl.cumsum(lengths, 0)
    total = tl.sum(lengths, 0)
    d = tl.arange(0, D)
    t = tl.arange(0, BLOCK_N)
    query = tl.load(Q + row * D + d).to(tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((D,), tl.float32)
    for start in range(tl.cdiv(total, BLOCK_N)):
        ordinal = start * BLOCK_N + t
        valid = ordinal < total
        lo = tl.zeros((BLOCK_N,), tl.int32)
        hi = tl.full((BLOCK_N,), WIDTH, tl.int32)
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            end = tl.gather(ends, tl.minimum(mid, WIDTH - 1), 0)
            before = end <= ordinal
            lo = tl.where(before, mid + 1, lo)
            hi = tl.where(before, hi, mid)
        pos = tl.minimum(lo, WIDTH - 1)
        slot = tl.gather(slots, pos, 0)
        previous = tl.where(pos > 0, tl.gather(ends, tl.maximum(pos - 1, 0), 0), 0)
        local = ordinal - previous
        page = _lookup_page_id(Roots, HashKeys, Directories, Used, kh, slot, local // 16, valid,
                               CAP, ROOTS, PAGE_CAP, HASH_CAP, PROBES)
        valid &= (page >= 0) & (page < PAGE_CAP)
        index = tl.load(PageIndices + (kh * PAGE_CAP + tl.maximum(page, 0)) * 16 + local % 16,
                        mask=valid, other=0).to(tl.int32)
        valid &= (index >= 0) & (index < LEAF_CAP)
        index = tl.minimum(tl.maximum(index, 0), LEAF_CAP - 1)
        keys = tl.load(LeafK + (kh * LEAF_CAP + index[:, None]) * D + d[None, :],
                       mask=valid[:, None], other=0).to(tl.float32)
        values = tl.load(LeafV + (kh * LEAF_CAP + index[:, None]) * D + d[None, :],
                         mask=valid[:, None], other=0).to(tl.float32)
        score = tl.where(valid, tl.sum(keys * query[None, :], 1) * SCALE, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(score, 0))
        safe = tl.where(next_max != -float('inf'), next_max, 0.0)
        factor = tl.where(maximum != -float('inf'), tl.exp(maximum - safe), 0.0)
        weights = tl.where(valid, tl.exp(score - safe), 0.0)
        numerator = numerator * factor + tl.sum(values * weights[:, None], 0)
        denominator = denominator * factor + tl.sum(weights, 0)
        maximum = next_max
    tl.store(Out + row * D + d, numerator / tl.maximum(denominator, 1.e-30))
    tl.store(LSE + row, tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf')))


@triton.jit
def _expand_direct_indices(Slots, Counts, Lengths, Roots, HashKeys, Directories, Used,
                           PageIndices, Indices, EntryCounts, RouteOffsets, EntryOffsets,
                           QLEN: tl.constexpr, WIDTH: tl.constexpr, CAP: tl.constexpr,
                           G: tl.constexpr, ROOTS: tl.constexpr, HASH_CAP: tl.constexpr,
                           PAGE_CAP: tl.constexpr, LEAF_CAP: tl.constexpr, PROBES: tl.constexpr,
                           ENTRY_BLOCK: tl.constexpr, SEARCH_STEPS: tl.constexpr,
                           PACKED: tl.constexpr, PADDED_ROUTES: tl.constexpr, EXPAND_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    kh = row // QLEN // G
    routes = tl.arange(0, WIDTH)
    count = tl.load(Counts + row)
    if PACKED and not PADDED_ROUTES:
        route_base = tl.load(RouteOffsets + row).to(tl.int64)
    else:
        route_base = row * WIDTH
    if PACKED:
        entry_base = tl.load(EntryOffsets + row).to(tl.int64)
    else:
        entry_base = row * ENTRY_BLOCK
    slots = tl.load(Slots + route_base + routes, mask=routes < count, other=0).to(tl.int32)
    sizes = tl.load(Lengths + kh * CAP + slots, mask=routes < count, other=0).to(tl.int32)
    ends = tl.cumsum(sizes, 0)
    total = tl.load(EntryCounts + row)
    for chunk in range(tl.cdiv(total, EXPAND_BLOCK)):
        ordinal = chunk * EXPAND_BLOCK + tl.arange(0, EXPAND_BLOCK)
        valid = ordinal < total
        lo = tl.zeros((EXPAND_BLOCK,), tl.int32)
        hi = tl.full((EXPAND_BLOCK,), WIDTH, tl.int32)
        for _ in tl.static_range(0, SEARCH_STEPS):
            mid = (lo + hi) // 2
            end = tl.gather(ends, tl.minimum(mid, WIDTH - 1), 0)
            before = end <= ordinal
            lo = tl.where(before, mid + 1, lo)
            hi = tl.where(before, hi, mid)
        pos = tl.minimum(lo, WIDTH - 1)
        slot = tl.gather(slots, pos, 0)
        previous = tl.where(pos > 0, tl.gather(ends, tl.maximum(pos - 1, 0), 0), 0)
        local = ordinal - previous
        page = _lookup_page_id(Roots, HashKeys, Directories, Used, kh, slot, local // 16, valid,
                               CAP, ROOTS, PAGE_CAP, HASH_CAP, PROBES)
        valid &= (page >= 0) & (page < PAGE_CAP)
        index = tl.load(PageIndices + (kh * PAGE_CAP + tl.maximum(page, 0)) * 16 + local % 16,
                        mask=valid, other=-1).to(tl.int32)
        valid &= (index >= 0) & (index < LEAF_CAP)
        tl.store(Indices + entry_base + ordinal, tl.where(valid, index, -1), mask=ordinal < total)


@triton.jit
def _expand_direct_indices_stream(Slots, Counts, Lengths, Roots, HashKeys, Directories, Used,
                                  PageIndices, Indices, RouteOffsets, EntryOffsets,
                                  QLEN: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr,
                                  ROOTS: tl.constexpr, HASH_CAP: tl.constexpr, PAGE_CAP: tl.constexpr,
                                  LEAF_CAP: tl.constexpr, PROBES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    kh = row // QLEN // G
    count = tl.load(Counts + row)
    route_base = tl.load(RouteOffsets + row).to(tl.int64)
    entry_base = tl.load(EntryOffsets + row).to(tl.int64)
    offset = tl.full((), 0, tl.int32)
    for route in range(count):
        slot = tl.load(Slots + route_base + route).to(tl.int32)
        size = tl.load(Lengths + kh * CAP + slot).to(tl.int32)
        for chunk in range(tl.cdiv(size, BLOCK)):
            local = chunk * BLOCK + tl.arange(0, BLOCK)
            valid = local < size
            page = _lookup_page_id(Roots, HashKeys, Directories, Used, kh, slot, local // 16, valid,
                                   CAP, ROOTS, PAGE_CAP, HASH_CAP, PROBES)
            valid &= (page >= 0) & (page < PAGE_CAP)
            index = tl.load(PageIndices + (kh * PAGE_CAP + tl.maximum(page, 0)) * 16 + local % 16,
                            mask=valid, other=-1).to(tl.int32)
            valid &= (index >= 0) & (index < LEAF_CAP)
            tl.store(Indices + entry_base + offset + local, tl.where(valid, index, -1), mask=local < size)
        offset += size


@triton.jit
def _indexed_direct_attention(Q, Indices, EntryCounts, EntryOffsets, LeafK, LeafV, Out, LSE,
                              QLEN: tl.constexpr, ENTRY_BLOCK: tl.constexpr, G: tl.constexpr,
                              LEAF_CAP: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                              BLOCK_N: tl.constexpr, PACKED: tl.constexpr):
    row = tl.program_id(0)
    kh = row // QLEN // G
    total = tl.load(EntryCounts + row)
    base = tl.load(EntryOffsets + row).to(tl.int64) if PACKED else row * ENTRY_BLOCK
    d = tl.arange(0, D)
    t = tl.arange(0, BLOCK_N)
    query = tl.load(Q + row * D + d).to(tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((D,), tl.float32)
    for start in range(tl.cdiv(total, BLOCK_N)):
        ordinal = start * BLOCK_N + t
        index = tl.load(Indices + base + ordinal, mask=ordinal < total, other=-1)
        valid = (ordinal < total) & (index >= 0) & (index < LEAF_CAP)
        index = tl.minimum(tl.maximum(index, 0), LEAF_CAP - 1)
        keys = tl.load(LeafK + (kh * LEAF_CAP + index[:, None]) * D + d[None, :],
                       mask=valid[:, None], other=0).to(tl.float32)
        values = tl.load(LeafV + (kh * LEAF_CAP + index[:, None]) * D + d[None, :],
                         mask=valid[:, None], other=0).to(tl.float32)
        score = tl.where(valid, tl.sum(keys * query[None, :], 1) * SCALE, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(score, 0))
        safe = tl.where(next_max != -float('inf'), next_max, 0.0)
        factor = tl.where(maximum != -float('inf'), tl.exp(maximum - safe), 0.0)
        weights = tl.where(valid, tl.exp(score - safe), 0.0)
        numerator = numerator * factor + tl.sum(values * weights[:, None], 0)
        denominator = denominator * factor + tl.sum(weights, 0)
        maximum = next_max
    tl.store(Out + row * D + d, numerator / tl.maximum(denominator, 1.e-30))
    tl.store(LSE + row, tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf')))


@triton.jit
def _paired_indexed_attention(Q, Indices, EntryCounts, LeafK, LeafV, Out, LSE,
                              QLEN: tl.constexpr, ENTRY_BLOCK: tl.constexpr, G: tl.constexpr,
                              LEAF_CAP: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                              BLOCK_N: tl.constexpr):
    pair = tl.program_id(0)
    groups_per_head = tl.cdiv(QLEN, 2)
    head = pair // groups_per_head
    q0 = (pair % groups_per_head) * 2
    row0 = head * QLEN + q0
    row1 = row0 + 1
    has_second = q0 + 1 < QLEN
    kh = head // G
    n0 = tl.load(EntryCounts + row0)
    n1 = tl.load(EntryCounts + row1, mask=has_second, other=0)
    d = tl.arange(0, D)
    t = tl.arange(0, BLOCK_N)
    query0 = tl.load(Q + row0 * D + d).to(tl.float32)
    query1 = tl.load(Q + row1 * D + d, mask=has_second, other=0).to(tl.float32)
    mx0 = tl.full((), -float('inf'), tl.float32)
    mx1 = tl.full((), -float('inf'), tl.float32)
    den0 = tl.full((), 0.0, tl.float32)
    den1 = tl.full((), 0.0, tl.float32)
    out0 = tl.zeros((D,), tl.float32)
    out1 = tl.zeros((D,), tl.float32)
    for start in range(tl.cdiv(tl.maximum(n0, n1), BLOCK_N)):
        ordinal = start * BLOCK_N + t
        idx0 = tl.load(Indices + row0 * ENTRY_BLOCK + ordinal, mask=ordinal < n0, other=-1)
        idx1 = tl.load(Indices + row1 * ENTRY_BLOCK + ordinal,
                       mask=has_second & (ordinal < n1), other=-1)
        valid0 = (ordinal < n0) & (idx0 >= 0) & (idx0 < LEAF_CAP)
        valid1 = has_second & (ordinal < n1) & (idx1 >= 0) & (idx1 < LEAF_CAP)
        shared = valid0 & valid1 & (idx0 == idx1)
        safe0 = tl.minimum(tl.maximum(idx0, 0), LEAF_CAP - 1)
        safe1 = tl.minimum(tl.maximum(idx1, 0), LEAF_CAP - 1)
        k0 = tl.load(LeafK + (kh * LEAF_CAP + safe0[:, None]) * D + d[None, :],
                     mask=valid0[:, None], other=0).to(tl.float32)
        v0 = tl.load(LeafV + (kh * LEAF_CAP + safe0[:, None]) * D + d[None, :],
                     mask=valid0[:, None], other=0).to(tl.float32)
        k1 = tl.load(LeafK + (kh * LEAF_CAP + safe1[:, None]) * D + d[None, :],
                     mask=(valid1 & ~shared)[:, None], other=0).to(tl.float32)
        v1 = tl.load(LeafV + (kh * LEAF_CAP + safe1[:, None]) * D + d[None, :],
                     mask=(valid1 & ~shared)[:, None], other=0).to(tl.float32)
        k1 = tl.where(shared[:, None], k0, k1)
        v1 = tl.where(shared[:, None], v0, v1)
        score0 = tl.where(valid0, tl.sum(k0 * query0[None, :], 1) * SCALE, -float('inf'))
        score1 = tl.where(valid1, tl.sum(k1 * query1[None, :], 1) * SCALE, -float('inf'))
        next0 = tl.maximum(mx0, tl.max(score0, 0))
        next1 = tl.maximum(mx1, tl.max(score1, 0))
        safe_mx0 = tl.where(next0 != -float('inf'), next0, 0.0)
        safe_mx1 = tl.where(next1 != -float('inf'), next1, 0.0)
        factor0 = tl.where(mx0 != -float('inf'), tl.exp(mx0 - safe_mx0), 0.0)
        factor1 = tl.where(mx1 != -float('inf'), tl.exp(mx1 - safe_mx1), 0.0)
        weight0 = tl.where(valid0, tl.exp(score0 - safe_mx0), 0.0)
        weight1 = tl.where(valid1, tl.exp(score1 - safe_mx1), 0.0)
        out0 = out0 * factor0 + tl.sum(v0 * weight0[:, None], 0)
        out1 = out1 * factor1 + tl.sum(v1 * weight1[:, None], 0)
        den0 = den0 * factor0 + tl.sum(weight0, 0)
        den1 = den1 * factor1 + tl.sum(weight1, 0)
        mx0 = next0
        mx1 = next1
    tl.store(Out + row0 * D + d, out0 / tl.maximum(den0, 1.e-30))
    tl.store(LSE + row0, tl.where(den0 > 0, mx0 + tl.log(den0), -float('inf')))
    tl.store(Out + row1 * D + d, out1 / tl.maximum(den1, 1.e-30), mask=has_second)
    tl.store(LSE + row1, tl.where(den1 > 0, mx1 + tl.log(den1), -float('inf')), mask=has_second)


@triton.jit
def _stream_route_reduce(RouteOut, RouteLSE, Counts, Out, LSE,
                         WIDTH: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    route = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    count = tl.load(Counts + row)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((D,), tl.float32)
    for block in range(tl.cdiv(count, BLOCK)):
        index = block * BLOCK + route
        valid = index < count
        mass = tl.load(RouteLSE + row * WIDTH + index, mask=valid, other=-float('inf')).to(tl.float32)
        next_max = tl.maximum(maximum, tl.max(mass, 0))
        safe = tl.where(next_max != -float('inf'), next_max, 0.0)
        factor = tl.where(maximum != -float('inf'), tl.exp(maximum - safe), 0.0)
        weight = tl.where(valid & (mass != -float('inf')), tl.exp(mass - safe), 0.0)
        values = tl.load(RouteOut + (row * WIDTH + index[:, None]) * D + d[None, :],
                         mask=(valid & (mass != -float('inf')))[:, None], other=0).to(tl.float32)
        numerator = numerator * factor + tl.sum(values * weight[:, None], 0)
        denominator = denominator * factor + tl.sum(weight, 0)
        maximum = next_max
    tl.store(Out + row * D + d, numerator / tl.maximum(denominator, 1.e-30))
    tl.store(LSE + row, tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf')))


def threshold_prefill_fused(engine, q, local_k, local_v, state_k, state_v, counts,
                            owners, exact_k, exact_v, *, state_len, state_capacity,
                            page_cache, local_branch, sink_k, sink_v, output_buffer=None,
                            rule, multiplier, entry_bound, work_bound, tile_queries,
                            block_m=32, block_n=64, **kwargs):
    del local_k, local_v, owners, exact_k, exact_v, kwargs
    if (q.size(0) != 1 or q.size(1) != 32 or q.size(-1) != 128 or not q.is_cuda
            or torch.version.hip or local_branch is None or page_cache is None
            or sink_k is None or sink_v is None or page_cache.get('quantization_finalized', False)):
        raise ValueError('fused threshold prefill requires batch-one H100 Llama BF16')
    if rule not in ('region', 'entry') or min(entry_bound, work_bound, tile_queries) < 1 or multiplier <= 0:
        raise ValueError('invalid threshold prefill configuration')
    block_m = int(os.environ.get('LOD_H100_PREFILL_SELECT_M', str(block_m)))
    block_n = int(os.environ.get('LOD_H100_PREFILL_SELECT_N', str(block_n)))
    select_warps = int(os.environ.get('LOD_H100_PREFILL_SELECT_WARPS', '4'))
    if block_m not in (16, 32) or block_n not in (64, 128, 256) or select_warps not in (4, 8):
        raise ValueError('unsupported fused selection geometry')
    buffers = getattr(engine, '_lod_prefill_attention_buffers', None)
    if buffers is None:
        buffers = {}
        engine._lod_prefill_attention_buffers = buffers
    device = q.device
    profile = os.environ.get('LOD_H100_PREFILL_PROFILE') == '1'
    if profile and not hasattr(engine, '_lod_leaf_timing_events'):
        engine._lod_leaf_timing_events = {}
    profile_stats = getattr(engine, '_lod_prefill_threshold_profile', None)
    if profile and profile_stats is None:
        profile_stats = {}
        engine._lod_prefill_threshold_profile = profile_stats
    if profile:
        torch.cuda.synchronize(device)
        profile_begin = time.perf_counter()

    def mark(name):
        nonlocal profile_begin
        if profile:
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            profile_stats[name] = profile_stats.get(name, 0.0) + now - profile_begin
            profile_begin = now
    workspace = lambda name, shape, dtype: _workspace_tensor(buffers, 'threshold_' + name, shape, dtype=dtype, device=device)
    h, kv, d, g = int(q.size(1)), int(state_k.size(1)), int(q.size(-1)), engine.num_key_value_groups
    mean_k = workspace('mean_k', (1, kv, state_len, d), state_k.dtype)
    mean_v = workspace('mean_v', (1, kv, state_len, d), state_v.dtype)
    _prepare_state[(kv * state_len,)](state_k, state_v, counts, mean_k, mean_v, state_len, state_capacity, d, num_warps=4)
    reuse_mean = os.environ.get('LOD_H100_PREFILL_REUSE_MEAN') == '1'
    reuse_scores = os.environ.get('LOD_H100_PREFILL_REUSE_SCORES') == '1'
    producer_lse = os.environ.get('LOD_H100_PREFILL_PRODUCER_LSE') == '1'
    bitmask = os.environ.get('LOD_H100_PREFILL_BITMASK') == '1'
    direct_pack = os.environ.get('LOD_H100_PREFILL_PACK_DIRECT') == '1'
    grouped8 = os.environ.get('LOD_H100_PREFILL_GROUPED8') == '1'
    grouped8_fused_counts = os.environ.get('LOD_H100_PREFILL_GROUPED8_FUSED_COUNTS') == '1'
    backend_crosscheck = grouped8 and os.environ.get('LOD_H100_PREFILL_GROUPED8_CROSSCHECK') == '1'
    max_regions = int(os.environ.get('LOD_H100_PREFILL_MAX_REGIONS', '0'))
    if max_regions not in (0, 8) or (grouped8 and max_regions != 8):
        raise ValueError('the grouped fixed-width experiment requires an eight-region cap')
    if grouped8_fused_counts and not grouped8:
        raise ValueError('fused expert route counts require grouped-eight attention')
    if reuse_scores and not reuse_mean:
        raise ValueError('reusing native scores requires cached routing means')
    if producer_lse and (not reuse_mean or reuse_scores):
        raise ValueError('producer LSE requires normalized-query cached-mean routing')
    if reuse_mean:
        if (engine.routing_normalization != 'query' or int(engine.routing_rope_fast_pairs) != 0
                or getattr(engine, 'mla_state_key_normalization', 'none') != 'none'):
            raise ValueError('cached routing means require query-only normalization and ordinary keys')
        if not reuse_scores:
            route_q_full = engine._routing_rms_normalize(q)
    mark('prepare_state')
    output = torch.empty_like(q) if output_buffer is None else output_buffer
    local_out, local_lse = local_branch
    lengths = page_cache['slot_lengths']
    if producer_lse:
        remote_totals = workspace('remote_totals', (kv,), torch.int64)
        region_totals = workspace('region_totals', (kv,), torch.int32)
        _region_totals[(kv,)](lengths, remote_totals, region_totals, state_len, state_capacity,
                              triton.next_power_of_2(state_len), num_warps=4)
    for begin in range(0, int(q.size(2)), tile_queries):
        end = min(begin + tile_queries, int(q.size(2)))
        query = q[..., begin:end, :].contiguous()
        qlen = end - begin
        if reuse_mean:
            route_q = query if reuse_scores else route_q_full[..., begin:end, :].contiguous()
            if producer_lse:
                producer_m = int(os.environ.get('LOD_H100_PREFILL_PRODUCER_M', '32'))
                producer_n = int(os.environ.get('LOD_H100_PREFILL_PRODUCER_N', '128'))
                if producer_m not in (16, 32, 64) or producer_n not in (64, 128, 256):
                    raise ValueError('unsupported route producer geometry')
                lse_tiles = triton.cdiv(state_len, producer_n)
                route_logits = workspace('producer_logits', (1, h, qlen, state_len), q.dtype)
                partial_lse = workspace('producer_lse', (h, qlen, lse_tiles), torch.float32)
                _route_logits_with_lse[(h, triton.cdiv(qlen, producer_m), lse_tiles)](
                    route_q, mean_k, lengths, route_logits, partial_lse, qlen, state_len, state_capacity,
                    h, kv, g, d, float(engine.scaling), producer_m, producer_n, lse_tiles,
                    num_warps=4, num_stages=2)
            elif engine.route_gqa_matmul:
                grouped_q = route_q.reshape(1, kv, g, qlen, d)
                route_logits = torch.matmul(grouped_q, mean_k.transpose(-1, -2).unsqueeze(2)).reshape(1, h, qlen, state_len).contiguous()
            else:
                route_logits = torch.matmul(route_q, engine._repeat_kv(mean_k).transpose(-1, -2)).contiguous()
            if not reuse_scores and os.environ.get('LOD_H100_PREFILL_VERIFY') == '1' and not hasattr(engine, '_lod_prefill_fused_validation'):
                prior_logits = engine._state_routing_logits(query, state_k, counts, state_len=state_len).contiguous()
                if not torch.equal(route_logits, prior_logits):
                    raise RuntimeError(f'cached routing logits differ: max abs {float((route_logits.float() - prior_logits.float()).abs().max().item())}')
        else:
            route_logits = engine._state_routing_logits(query, state_k, counts, state_len=state_len).contiguous()
        mark('route_logits')
        selected = workspace('selected_bits' if bitmask else 'selected',
                             (1,) if (direct_pack or grouped8) else
                             (h, qlen, triton.cdiv(state_len, 8) if bitmask else state_len), torch.uint8)
        route_counts = workspace('route_counts', (h, qlen), torch.int32)
        exact_mode = os.environ.get('LOD_H100_PREFILL_EXACT', 'direct-indexed' if rule == 'entry' else 'expert')
        if grouped8:
            exact_mode = 'expert'
        mapped = exact_mode == 'direct-indexed'
        direct = exact_mode in ('direct', 'direct-indexed')
        packed = os.environ.get('LOD_H100_PREFILL_PACKED') == '1' and not grouped8
        no_host_sync = packed or os.environ.get('LOD_H100_PREFILL_NO_HOST_SYNC') == '1'
        if grouped8:
            no_host_sync = True
        fuse_pack = os.environ.get('LOD_H100_PREFILL_FUSE_PACK') == '1'
        if direct_pack and (not packed or not mapped or bitmask or fuse_pack or grouped8):
            raise ValueError('direct packing requires packed indexed attention without bitmask or legacy fused packing')
        if bitmask and (not packed or fuse_pack):
            raise ValueError('bitmask selection requires packed direct-indexed routing')
        if no_host_sync and (rule != 'entry' or (not mapped and not grouped8)):
            raise ValueError('bounded device-only packing requires indexed per-entry routing')
        if fuse_pack and not no_host_sync:
            raise ValueError('fused packing requires a bounded device-only workspace')
        if packed and (fuse_pack or os.environ.get('LOD_H100_PREFILL_PAIR_REUSE') == '1'):
            raise ValueError('packed workspaces are not yet combined with fused packing or paired attention')
        if no_host_sync:
            leaf_capacity = int(page_cache['leaf_k'].size(2))
            # The remote entries are a disjoint subset of the leaf store. The
            # strict work bound limits both entries and nonempty regions.
            max_entries = max(1, min(leaf_capacity, math.isqrt(work_bound * work_bound * leaf_capacity) + 1))
            entry_block = 1 << (max_entries - 1).bit_length()
            max_routes = max(1, min(state_len, max_entries))
            width = max_regions if max_regions else 1 << (max_routes - 1).bit_length()
            slots = workspace('slots', (1, h, qlen, width), torch.int64 if grouped8 else torch.int32)
            if grouped8:
                slots.fill_(-1)
                if grouped8_fused_counts:
                    fused_head_counts = workspace('grouped8_head_counts', (h * state_len,), torch.int32)
                    fused_route_offsets = workspace('grouped8_route_offsets', (1, h, qlen, width), torch.int32)
                    fused_head_counts.zero_()
        else:
            slots = selected
        entry_counts = workspace('entry_counts', (h, qlen), torch.int32) if (mapped or backend_crosscheck) else route_counts
        coarse_out = workspace('coarse_out', (1, h, qlen, d), q.dtype)
        coarse_lse = workspace('coarse_lse', (1, h, qlen), torch.float32)
        verify = os.environ.get('LOD_H100_PREFILL_VERIFY') == '1' and not hasattr(engine, '_lod_prefill_fused_validation')
        if verify and reuse_scores:
            raise ValueError('exact fused verification assumes normalized-query matmul; use the reuse audit for this experiment')
        debug_scores = (workspace('debug_scores', (h, qlen, state_len), torch.float32) if verify else coarse_lse)
        _select_and_coarse[(h, triton.cdiv(qlen, block_m))](query, route_logits, mean_k, mean_v, lengths, selected, slots, route_counts, entry_counts,
            coarse_out, coarse_lse, debug_scores, partial_lse if producer_lse else coarse_lse,
            remote_totals if producer_lse else route_counts, region_totals if producer_lse else route_counts,
            fused_head_counts if grouped8_fused_counts else route_counts,
            fused_route_offsets if grouped8_fused_counts else route_counts,
            qlen, state_len, state_capacity, h, kv, g, d, float(engine.scaling),
            rule == 'entry', float(multiplier), entry_bound, work_bound,
            int(engine.max_open_centroid_leaves or 2147483647), block_m, block_n, verify, mapped or backend_crosscheck, fuse_pack or direct_pack or grouped8,
            width if no_host_sync else 1, reuse_scores, producer_lse, lse_tiles if producer_lse else 1,
            triton.next_power_of_2(lse_tiles) if producer_lse else 1, bitmask, not (direct_pack or grouped8), max_regions,
            grouped8_fused_counts,
            num_warps=select_warps, num_stages=2)
        if os.environ.get('LOD_H100_OVERLAP_AUDIT') == '1' and begin == 0:
            if not grouped8:
                raise ValueError('overlap audit requires the capped grouped-eight threshold path')
            from benchmarks.h100_prefill_overlap import audit_routes
            audit_routes(engine, slots, lengths, state_len=state_len, group_size=engine.num_key_value_groups,
                         sample_limit=512, label='512_queries', route_counts=route_counts)
        if max_regions and os.environ.get('LOD_H100_PREFILL_CAP_AUDIT') == '1' and begin == 0:
            from benchmarks.h100_threshold_prefill import select_regions
            n = min(64, qlen)
            sample_lengths = engine._repeat_kv(lengths[..., :state_len]).to(torch.int64)
            sample_counts = engine._repeat_kv(counts[..., :state_len, :]).squeeze(-1).float()
            route = route_logits[..., :n, :].float() * engine.scaling + sample_counts.clamp_min(1).log().unsqueeze(2)
            kwargs = dict(rule=rule, multiplier=multiplier, entry_bound=entry_bound,
                          work_bound=work_bound, max_region_entries=engine.max_open_centroid_leaves)
            uncapped, _ = select_regions(route, sample_lengths, **kwargs)
            expected, _ = select_regions(route, sample_lengths, max_regions=max_regions, **kwargs)
            actual = _sample_selected(selected, slots, route_counts, bitmask=bitmask,
                                      direct_pack=direct_pack or grouped8, h=h, qlen=qlen, n=n,
                                      state_len=state_len, width=width)
            uncapped_entries = (uncapped.to(torch.int64) * sample_lengths.unsqueeze(2)).sum(-1)
            actual_entries = (actual.to(torch.int64) * sample_lengths.unsqueeze(2)).sum(-1)
            remote = sample_lengths.sum(-1, keepdim=True)
            if bool((actual.sum(-1) > max_regions).any().item()):
                raise RuntimeError('region cap violated')
            if rule == 'entry' and bool((actual_entries.square() >= work_bound * work_bound * remote).any().item()):
                raise RuntimeError('capped entry bound violated')
            audit = getattr(engine, '_lod_prefill_cap_audit', None)
            if audit is None:
                audit = {'query_heads': 0, 'uncapped_regions': 0, 'capped_regions': 0,
                         'uncapped_entries': 0, 'capped_entries': 0, 'above_cap_heads': 0,
                         'zero_regions_heads': 0, 'selection_mismatches': 0}
                engine._lod_prefill_cap_audit = audit
            audit['query_heads'] += h * n
            audit['uncapped_regions'] += int(uncapped.sum().item())
            audit['capped_regions'] += int(actual.sum().item())
            audit['uncapped_entries'] += int(uncapped_entries.sum().item())
            audit['capped_entries'] += int(actual_entries.sum().item())
            audit['above_cap_heads'] += int((uncapped.sum(-1) > max_regions).sum().item())
            audit['zero_regions_heads'] += int((actual.sum(-1) == 0).sum().item())
            audit['selection_mismatches'] += int((expected != actual).sum().item())
        if reuse_scores and os.environ.get('LOD_H100_PREFILL_REUSE_AUDIT') == '1' and begin == 0:
            # Use the same tile shape as production for the baseline matmul;
            # sampling the query before matmul can change BF16 rounding.
            from benchmarks.h100_threshold_prefill import select_regions
            normalized = engine._routing_rms_normalize(query)
            grouped = normalized.reshape(1, kv, g, qlen, d)
            old_logits = torch.matmul(grouped, mean_k.transpose(-1, -2).unsqueeze(2)).reshape(1, h, qlen, state_len)
            n = min(64, qlen)
            sample_lengths = engine._repeat_kv(lengths[..., :state_len]).to(torch.int64)
            sample_counts = engine._repeat_kv(counts[..., :state_len, :]).squeeze(-1).float()
            old_scores = old_logits[..., :n, :].float() * engine.scaling + sample_counts.clamp_min(1).log().unsqueeze(2)
            expected, _ = select_regions(old_scores, sample_lengths, rule=rule, multiplier=multiplier,
                                          entry_bound=entry_bound, work_bound=work_bound,
                                          max_region_entries=engine.max_open_centroid_leaves)
            actual = _sample_selected(selected, slots, route_counts, bitmask=bitmask, direct_pack=direct_pack,
                                      h=h, qlen=qlen, n=n, state_len=state_len, width=width)
            disagreement = expected != actual
            old_entries = (expected.to(torch.int64) * sample_lengths.unsqueeze(2)).sum(-1)
            new_entries = (actual.to(torch.int64) * sample_lengths.unsqueeze(2)).sum(-1)
            remote = sample_lengths.sum(-1, keepdim=True)
            if rule == 'entry' and bool((new_entries.square() >= work_bound * work_bound * remote).any().item()):
                raise RuntimeError('reused-score entry bound violated')
            audit = getattr(engine, '_lod_prefill_reuse_audit', None)
            if audit is None:
                audit = {'query_heads': 0, 'region_comparisons': 0, 'changed_regions': 0,
                         'changed_query_heads': 0, 'old_opened_entries': 0, 'new_opened_entries': 0,
                         'max_abs_route_logit_difference': 0.0}
                engine._lod_prefill_reuse_audit = audit
            rms = engine._routing_rms(query[..., :n, :])
            reused = route_logits[..., :n, :].float() / rms * engine.scaling
            difference = (reused - old_logits[..., :n, :].float() * engine.scaling).abs()
            audit['query_heads'] += h * n
            audit['region_comparisons'] += int(disagreement.numel())
            audit['changed_regions'] += int(disagreement.sum().item())
            audit['changed_query_heads'] += int(disagreement.any(-1).sum().item())
            audit['old_opened_entries'] += int(old_entries.sum().item())
            audit['new_opened_entries'] += int(new_entries.sum().item())
            audit['max_abs_route_logit_difference'] = max(audit['max_abs_route_logit_difference'], float(difference.max().item()))
        if not no_host_sync:
            summary = [route_counts.max().to(torch.int64), route_counts.sum(dtype=torch.int64)]
            if mapped:
                summary.append(entry_counts.max().to(torch.int64))
            summary = [int(value) for value in torch.stack(summary).tolist()]
            maximum, total_routes = summary[:2]
            maximum = max(1, maximum)
        if packed:
            entry_offsets = workspace('packed_entry_offsets', (h * qlen + 1,), torch.int64)
            entry_offsets[:1].zero_()
            if not direct_pack:
                route_offsets = workspace('packed_route_offsets', (h * qlen + 1,), torch.int64)
                route_offsets[:1].zero_()
                torch.cumsum(route_counts.reshape(-1), 0, dtype=torch.int64, out=route_offsets[1:])
            torch.cumsum(entry_counts.reshape(-1), 0, dtype=torch.int64, out=entry_offsets[1:])
        mark('select_and_coarse')
        if not no_host_sync:
            width = 1 << (maximum - 1).bit_length()
            slots = workspace('slots', (1, h, qlen, width), torch.int64)
        # Direct paths read only the compacted prefix described by
        # route_counts, so clearing padded slots would be wasted bandwidth.
        if not direct and not grouped8:
            slots.fill_(-1)
        group_rows = int(os.environ.get('LOD_H100_PREFILL_COMPACT_GROUP', '16'))
        if packed and not direct_pack:
            _compact_slots_packed[(h, triton.cdiv(qlen, 16))](selected, slots, route_offsets,
                                                            qlen, state_len, 16, 128, bitmask, num_warps=4)
        elif direct and not fuse_pack and not direct_pack:
            _compact_slots_only[(h, triton.cdiv(qlen, 16))](selected, slots, qlen, state_len, width,
                                                          16, 128, num_warps=4)
        elif not direct and not grouped8:
            offsets = workspace('offsets', (1, h, qlen, width), torch.int32)
            head_counts = workspace('head_counts', (h * state_len,), torch.int32)
            head_counts.zero_()
        if not direct and not grouped8 and group_rows == 1:
            _compact[(h * qlen,)](selected, slots, offsets, head_counts, qlen, state_len, width,
                                  triton.next_power_of_2(state_len), num_warps=4)
        elif not direct and not grouped8 and group_rows in (4, 8, 16, 32):
            _compact_grouped[(h, triton.cdiv(qlen, group_rows))](selected, slots, offsets, head_counts,
                qlen, state_len, width, group_rows, 128, num_warps=4)
        elif not direct and not grouped8:
            raise ValueError('invalid grouped compaction size')
        if not direct and not grouped8:
            engine._lod_prefill_route_head_counts = head_counts
            engine._lod_prefill_route_offsets = offsets
            engine._lod_prefill_total_valid_routes = total_routes
        if grouped8_fused_counts:
            engine._lod_prefill_route_head_counts = fused_head_counts
            engine._lod_prefill_route_offsets = fused_route_offsets
        mark('compact')
        if direct:
            roots = page_cache['slot_pages']
            directory = page_cache['overflow_page_values']
            pages = page_cache['page_indices']
            leaf_k, leaf_v = page_cache['leaf_k'], page_cache['leaf_v']
            exact_out = workspace('direct_out', (1, h, qlen, d), q.dtype)
            exact_lse = workspace('direct_lse', (1, h, qlen), torch.float32)
            direct_block = int(os.environ.get('LOD_H100_PREFILL_DIRECT_BLOCK', '128' if mapped else '32'))
            direct_warps = int(os.environ.get('LOD_H100_PREFILL_DIRECT_WARPS', '4'))
            if direct_block not in (16, 32, 64, 128) or direct_warps not in (4, 8):
                raise ValueError('unsupported direct attention geometry')
            if mapped:
                if not no_host_sync:
                    max_entries = max(1, summary[2])
                    entry_block = 1 << (max_entries - 1).bit_length()
                indices = workspace('direct_indices', (h, qlen, entry_block), torch.int32)
                if os.environ.get('LOD_H100_PREFILL_EXPAND_STREAM') == '1':
                    if not packed or direct_pack:
                        raise ValueError('stream expansion requires packed workspaces')
                    _expand_direct_indices_stream[(h * qlen,)](slots, route_counts, lengths, roots,
                        page_cache['overflow_page_keys'], directory, page_cache['overflow_used'], pages,
                        indices, route_offsets, entry_offsets, qlen, state_capacity, g, int(roots.size(3)),
                        int(directory.size(2)), int(pages.size(2)), int(leaf_k.size(2)),
                        engine._page_lookup_probes(page_cache), 128, num_warps=4)
                else:
                    _expand_direct_indices[(h * qlen,)](slots, route_counts, lengths, roots,
                        page_cache['overflow_page_keys'], directory, page_cache['overflow_used'], pages,
                        indices, entry_counts, route_offsets if packed and not direct_pack else route_counts,
                        entry_offsets if packed else entry_counts, qlen, width, state_capacity, g, int(roots.size(3)),
                        int(directory.size(2)), int(pages.size(2)), int(leaf_k.size(2)),
                        engine._page_lookup_probes(page_cache), entry_block, width.bit_length(), packed, direct_pack,
                        256 if packed else entry_block, num_warps=4)
                mark('expand_indices')
                selected_event = None
                if profile and os.environ.get('LOD_H100_PREFILL_GPU_EVENTS') == '1':
                    selected_event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    selected_event[0].record()
                if os.environ.get('LOD_H100_PREFILL_PAIR_REUSE') == '1':
                    pair_block = int(os.environ.get('LOD_H100_PREFILL_PAIR_BLOCK', '32'))
                    if pair_block not in (16, 32, 64):
                        raise ValueError('unsupported paired attention block')
                    _paired_indexed_attention[(h * triton.cdiv(qlen, 2),)](
                        query, indices, entry_counts, leaf_k, leaf_v, exact_out, exact_lse,
                        qlen, entry_block, g, int(leaf_k.size(2)), d, float(engine.scaling),
                        pair_block, num_warps=4)
                else:
                    _indexed_direct_attention[(h * qlen,)](query, indices, entry_counts,
                        entry_offsets if packed else entry_counts, leaf_k, leaf_v,
                        exact_out, exact_lse, qlen, entry_block, g, int(leaf_k.size(2)), d,
                        float(engine.scaling), direct_block, packed, num_warps=direct_warps)
                if selected_event is not None:
                    selected_event[1].record()
                    engine._lod_leaf_timing_events.setdefault('selected_attention', []).append(selected_event)
            else:
                _direct_leaf_attention[(h * qlen,)](query, slots, route_counts, lengths, roots,
                    page_cache['overflow_page_keys'], directory, page_cache['overflow_used'], pages,
                    leaf_k, leaf_v, exact_out, exact_lse, qlen, width, state_len, state_capacity, g,
                    int(roots.size(3)), int(directory.size(2)), int(pages.size(2)), int(leaf_k.size(2)),
                    engine._page_lookup_probes(page_cache), d, float(engine.scaling), direct_block,
                    width.bit_length(), num_warps=direct_warps)
        else:
            # The fixed-eight reducer defaults to one warp. Wide threshold
            # frontiers can exceed its register budget, so expose a separate
            # experimental geometry without changing the serving default.
            old_warps = engine.leaf_reduce_num_warps
            engine.leaf_reduce_num_warps = int(os.environ.get('LOD_H100_PREFILL_REDUCE_WARPS', old_warps))
            streaming = os.environ.get('LOD_H100_PREFILL_REDUCE') == 'stream'
            try:
                exact_out, exact_lse = engine._paged_leaf_attention(query, slots, page_cache,
                                                                    active_slots=state_len, reduce_routes=not streaming)
            finally:
                engine.leaf_reduce_num_warps = old_warps
            if streaming:
                route_out, route_lse = exact_out, exact_lse
                exact_out = workspace('stream_out', (1, h, qlen, d), q.dtype)
                exact_lse = workspace('stream_lse', (1, h, qlen), torch.float32)
                _stream_route_reduce[(h * qlen,)](route_out, route_lse, route_counts, exact_out, exact_lse,
                                                  width, d, 32, num_warps=4)
        if backend_crosscheck and begin == 0:
            # Run both leaf implementations on the same queries, slots, and
            # cache, so earlier numerical differences cannot change the input.
            n = min(16, qlen)
            sample_q = query[..., :n, :].contiguous()
            sample_slots = slots[..., :n, :].contiguous()
            sample_routes = route_counts[:, :n].contiguous()
            sample_entries = entry_counts[:, :n].contiguous()
            offsets = workspace('crosscheck_entry_offsets', (h * n + 1,), torch.int64)
            offsets[:1].zero_()
            torch.cumsum(sample_entries.reshape(-1), 0, dtype=torch.int64, out=offsets[1:])
            indices = workspace('crosscheck_indices', (h, n, entry_block), torch.int32)
            leaf_k, leaf_v = page_cache['leaf_k'], page_cache['leaf_v']
            roots = page_cache['slot_pages']
            directory = page_cache['overflow_page_values']
            pages = page_cache['page_indices']
            _expand_direct_indices[(h * n,)](sample_slots, sample_routes, lengths, roots,
                page_cache['overflow_page_keys'], directory, page_cache['overflow_used'], pages,
                indices, sample_entries, sample_routes, offsets, n, width, state_capacity, g, int(roots.size(3)),
                int(directory.size(2)), int(pages.size(2)), int(leaf_k.size(2)),
                engine._page_lookup_probes(page_cache), entry_block, width.bit_length(), True, True,
                256, num_warps=4)
            indexed_out = workspace('crosscheck_out', (1, h, n, d), q.dtype)
            indexed_lse = workspace('crosscheck_lse', (1, h, n), torch.float32)
            _indexed_direct_attention[(h * n,)](sample_q, indices, sample_entries, offsets, leaf_k, leaf_v,
                indexed_out, indexed_lse, n, entry_block, g, int(leaf_k.size(2)), d,
                float(engine.scaling), 128, True, num_warps=4)
            original_out = merge_attention_branches_with_sink(sample_q, sink_k, sink_v,
                coarse_out[..., :n, :], coarse_lse[..., :n], exact_out[..., :n, :], exact_lse[..., :n],
                local_out[..., begin:begin+n, :], local_lse[..., begin:begin+n], kv_group_size=g, scale=engine.scaling)
            comparison_out = merge_attention_branches_with_sink(sample_q, sink_k, sink_v,
                coarse_out[..., :n, :], coarse_lse[..., :n], indexed_out, indexed_lse,
                local_out[..., begin:begin+n, :], local_lse[..., begin:begin+n], kv_group_size=g, scale=engine.scaling)
            error = (original_out.float() - comparison_out.float()).abs()
            if not bool(torch.isfinite(error).all().item()):
                raise RuntimeError('grouped versus indexed comparison produced non-finite output')
            audit = getattr(engine, '_lod_grouped8_backend_audit', None)
            if audit is None:
                audit = {'query_heads': 0, 'max_abs_output_difference': 0.0,
                         'sum_abs_output_difference': 0.0, 'output_elements': 0}
                engine._lod_grouped8_backend_audit = audit
            audit['query_heads'] += h * n
            audit['max_abs_output_difference'] = max(audit['max_abs_output_difference'], float(error.max().item()))
            audit['sum_abs_output_difference'] += float(error.sum().item())
            audit['output_elements'] += error.numel()
        mark('exact_leaves')
        merge_attention_branches_with_sink(query, sink_k, sink_v, coarse_out, coarse_lse, exact_out, exact_lse,
            local_out[..., begin:end, :], local_lse[..., begin:end], kv_group_size=g, scale=engine.scaling,
            output_buffer=output[..., begin:end, :])
        mark('merge')
        if verify:
            from benchmarks.h100_threshold_prefill import select_regions, threshold_prefill
            n = min(16, qlen)
            sample = query[..., :n, :].contiguous()
            sample_lengths = engine._repeat_kv(lengths[..., :state_len]).to(torch.int64)
            sample_counts = engine._repeat_kv(counts[..., :state_len, :]).squeeze(-1).float()
            # Reuse the actual query-tile matmul: recomputing only this sample
            # can select a different cuBLAS algorithm and BF16 rounding.
            route = route_logits[..., :n, :].float() * engine.scaling
            route += sample_counts.clamp_min(1).log().unsqueeze(2)
            expected, _ = select_regions(route, sample_lengths, rule=rule, multiplier=multiplier,
                                          entry_bound=entry_bound, work_bound=work_bound,
                                          max_region_entries=engine.max_open_centroid_leaves,
                                          max_regions=max_regions or None)
            actual = _sample_selected(selected, slots, route_counts, bitmask=bitmask, direct_pack=direct_pack or grouped8,
                                      h=h, qlen=qlen, n=n, state_len=state_len, width=width)
            if rule == 'entry':
                opened_entries = (actual.to(torch.int64) * sample_lengths.unsqueeze(2)).sum(-1)
                remote_entries = sample_lengths.sum(-1, keepdim=True)
                if bool((opened_entries.square() >= work_bound * work_bound * remote_entries).any().item()):
                    raise RuntimeError('fused real-state entry bound violated')
            mismatch_mask = expected != actual
            mismatches = int(mismatch_mask.sum().item())
            outside_tolerance = 0
            max_margin = 0.0
            if mismatches:
                valid_mask = sample_lengths > 0
                norm = torch.logsumexp(route.masked_fill(~valid_mask.unsqueeze(2), -torch.inf), -1, keepdim=True)
                if rule == 'entry':
                    cutoff_all = norm - math.log(entry_bound) - 0.5 * sample_lengths.sum(-1, keepdim=True).unsqueeze(2).float().log() + 1.e-6
                    value_all = route - sample_lengths.clamp_min(1).float().log().unsqueeze(2)
                else:
                    cutoff_all = norm + torch.log(multiplier / valid_mask.sum(-1, keepdim=True).unsqueeze(2).float())
                    value_all = route
                margins = (value_all - cutoff_all).abs().masked_select(mismatch_mask)
                max_margin = float(margins.max().item())
                outside_tolerance = int((margins > 2.e-5).sum().item())
                i = mismatch_mask.nonzero()[0].tolist()
                b, head, position, slot = i
                valid = sample_lengths[b, head] > 0
                remote_count = int(sample_lengths[b, head].sum().item())
                cutoff = torch.logsumexp(route[b, head, position].masked_fill(~valid, -torch.inf), 0)
                if rule == 'entry':
                    cutoff = cutoff - math.log(entry_bound) - 0.5 * math.log(remote_count) + 1.e-6
                    reference_value = route[b, head, position, slot] - sample_lengths[b, head, slot].float().log()
                else:
                    cutoff = cutoff + math.log(multiplier / int(valid.sum().item()))
                    reference_value = route[b, head, position, slot]
                print({'prefill_route_mismatch': i, 'rule': rule, 'reference_margin': float((reference_value-cutoff).item()),
                       'reference_route': float(route[b, head, position, slot].item()),
                       'fused_route': float(debug_scores[head, position, slot].item()),
                       'reference_selected': bool(expected[b, head, position, slot].item()),
                       'fused_selected': bool(actual[b, head, position, slot].item())}, flush=True)
            result = output[..., begin:begin+n, :].clone()
            reference = threshold_prefill(engine, sample, None, None, state_k, state_v, counts,
                None, None, None, state_len=state_len, state_capacity=state_capacity,
                page_cache=page_cache, local_branch=(local_out[..., begin:begin+n, :],
                                                    local_lse[..., begin:begin+n]),
                sink_k=sink_k, sink_v=sink_v, rule=rule, multiplier=multiplier,
                entry_bound=entry_bound, work_bound=work_bound, tile_queries=n, max_regions=max_regions or None)
            error = float((result.float() - reference.float()).abs().max().item())
            if outside_tolerance or error > 0.025:
                raise RuntimeError(f'fused real-state frontier mismatch: {mismatches=} {outside_tolerance=} {max_margin=} {error=}')
            engine._lod_prefill_fused_validation = {'query_heads': h * n,
                'state_len': state_len, 'selection_mismatches': mismatches,
                'selection_outside_tolerance': outside_tolerance, 'max_log_margin': max_margin, 'max_abs': error}
    return output
