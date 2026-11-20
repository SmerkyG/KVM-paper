"""Exact prefill accumulation without a materialized output per region."""

import torch
import triton
import triton.language as tl

from ._launch import hip_launch_kwargs
from ._paged_common import _lookup_page_id


@triton.jit
def finite_anchor(value):
    return tl.where(tl.abs(value) < float("inf"), value, 0.0)


@triton.jit
def _exact_kv_tile(K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                   kh, region, ordinal, length, dim, D: tl.constexpr, STATE: tl.constexpr,
                   LEAVES: tl.constexpr, PAGES: tl.constexpr, INLINE: tl.constexpr,
                   HASH: tl.constexpr, PROBES: tl.constexpr):
    live = ordinal < length
    page = _lookup_page_id(
        Directory, OverflowKeys, OverflowValues, OverflowUsed,
        kh, region, ordinal // 16, live, STATE, INLINE, PAGES, HASH, PROBES,
    )
    page_live = live & (page >= 0) & (page < PAGES)
    leaf = tl.load(Indices + (kh * PAGES + page) * 16 + ordinal % 16, mask=page_live, other=-1)
    valid = page_live & (leaf >= 0) & (leaf < LEAVES)
    leaf = tl.where(valid, leaf, 0)
    index = (kh * LEAVES + leaf[:, None].to(tl.int64)) * D + dim[None, :]
    key = tl.load(K + index, mask=valid[:, None], other=0)
    value = tl.load(V + index, mask=valid[:, None], other=0)
    return key, value, valid


@triton.jit
def _exact_row(Q, K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                Lengths, Slots, Counts, row, T: tl.constexpr, G: tl.constexpr,
                D: tl.constexpr, WIDTH: tl.constexpr, STATE: tl.constexpr,
                LEAVES: tl.constexpr, PAGES: tl.constexpr, INLINE: tl.constexpr,
                HASH: tl.constexpr, PROBES: tl.constexpr, N: tl.constexpr,
                SCALE: tl.constexpr):
    """Walk the exact entries with one online softmax across all open regions."""
    kh = row // T // G
    dim = tl.arange(0, D)
    token = tl.arange(0, N)
    query = tl.load(Q + row * D + dim).to(tl.float32)
    count = tl.load(Counts + row)
    maximum = tl.full((), -float("inf"), tl.float32)
    mass = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((D,), tl.float32)
    for rank in range(count):
        region = tl.load(Slots + row * WIDTH + rank).to(tl.int64)
        live_region = (region >= 0) & (region < STATE)
        length = tl.load(Lengths + kh * STATE + region, mask=live_region, other=0)
        for begin in range(0, length, N):
            ordinal = begin + token
            key, value, valid = _exact_kv_tile(
                K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                kh, region, ordinal, length, dim, D, STATE, LEAVES, PAGES, INLINE, HASH, PROBES,
            )
            logits = tl.where(valid, tl.sum(key.to(tl.float32) * query[None, :], 1) * SCALE, -float("inf"))
            updated = tl.maximum(maximum, tl.max(logits, 0))
            safe = finite_anchor(updated)
            old_weight = tl.exp(maximum - safe)
            weight = tl.exp(logits - safe)
            numerator = old_weight * numerator + tl.sum(weight[:, None] * value.to(tl.float32), 0)
            mass = old_weight * mass + tl.sum(weight, 0)
            maximum = updated
    return (
        numerator / tl.maximum(mass, 1.0e-30),
        tl.where(mass > 0.0, maximum + tl.log(mass), -float("inf")),
    )


@triton.jit
def _next_region(Slots, row, cursor, count, WIDTH: tl.constexpr, STATE: tl.constexpr):
    slot = tl.load(Slots + row * WIDTH + cursor, mask=cursor < count, other=STATE)
    return tl.where((slot >= 0) & (slot < STATE), slot, STATE).to(tl.int32)


@triton.jit
def _grouped_query(Q, K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                    Lengths, Slots, Counts, Order, Out, LSE, T: tl.constexpr, G: tl.constexpr,
                    D: tl.constexpr, WIDTH: tl.constexpr, STATE: tl.constexpr,
                    LEAVES: tl.constexpr, PAGES: tl.constexpr, INLINE: tl.constexpr,
                    HASH: tl.constexpr, PROBES: tl.constexpr, M: tl.constexpr,
                    N: tl.constexpr, SCALE: tl.constexpr, REORDER: tl.constexpr):
    """Merge the route lists of queries sharing a KV head."""
    head = tl.program_id(1).to(tl.int64)
    qid = tl.program_id(0) * M + tl.arange(0, M)
    if REORDER:
        kh = head
        active = qid < G * T
        offset = tl.load(Order + kh * G * T + qid, mask=active, other=0)
        row = kh * G * T + offset
    else:
        kh = head // G
        active = qid < T
        row = head * T + qid
    dim = tl.arange(0, D)
    token = tl.arange(0, N)
    query = tl.load(Q + row[:, None] * D + dim[None, :], mask=active[:, None], other=0)
    count = tl.load(Counts + row, mask=active, other=0)
    cursor = tl.zeros((M,), tl.int32)
    next_slot = _next_region(Slots, row, cursor, count, WIDTH, STATE)
    region = tl.min(next_slot, 0)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    mass = tl.zeros((M,), tl.float32)
    numerator = tl.zeros((M, D), tl.float32)
    while region < STATE:
        selected = next_slot == region
        length = tl.load(Lengths + kh * STATE + region)
        for begin in range(0, length, N):
            key, value, valid = _exact_kv_tile(
                K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                kh, region, begin + token, length, dim, D, STATE, LEAVES, PAGES, INLINE, HASH, PROBES,
            )
            logits = tl.dot(query, tl.trans(key)) * SCALE
            logits = tl.where(selected[:, None] & valid[None, :], logits, -float("inf"))
            updated = tl.maximum(maximum, tl.max(logits, 1))
            safe = finite_anchor(updated)
            old_weight = tl.exp(maximum - safe)
            weight = tl.exp(logits - safe[:, None])
            numerator = tl.dot(weight.to(value.dtype), value, numerator * old_weight[:, None])
            mass = old_weight * mass + tl.sum(weight, 1)
            maximum = updated
        cursor += selected.to(tl.int32)
        next_slot = _next_region(Slots, row, cursor, count, WIDTH, STATE)
        region = tl.min(next_slot, 0)
    tl.store(Out + row[:, None] * D + dim[None, :], numerator / tl.maximum(mass[:, None], 1.0e-30),
             mask=active[:, None])
    tl.store(LSE + row, tl.where(mass > 0, maximum + tl.log(mass), -float("inf")), mask=active)


@triton.jit
def _query_major(Q, K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                 Lengths, Slots, Counts, Repair, Out, LSE, ROWS: tl.constexpr,
                 T: tl.constexpr, G: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr,
                 STATE: tl.constexpr, LEAVES: tl.constexpr, PAGES: tl.constexpr,
                 INLINE: tl.constexpr, HASH: tl.constexpr, PROBES: tl.constexpr,
                 N: tl.constexpr, SCALE: tl.constexpr, REPAIR_GROUP: tl.constexpr):
    base = tl.program_id(0).to(tl.int64)
    dim = tl.arange(0, D)
    if REPAIR_GROUP:
        base *= REPAIR_GROUP
        lane = tl.arange(0, REPAIR_GROUP)
        pending = tl.load(Repair + base + lane, mask=base + lane < ROWS, other=0) != 0
        first = tl.min(tl.where(pending, lane, REPAIR_GROUP), 0)
        while first < REPAIR_GROUP:
            row = base + first
            value, lse = _exact_row(
                Q, K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
                Lengths, Slots, Counts, row, T, G, D, WIDTH, STATE, LEAVES, PAGES, INLINE, HASH, PROBES, N, SCALE,
            )
            tl.store(Out + row * D + dim, value)
            tl.store(LSE + row, lse)
            first = tl.min(tl.where(pending & (lane > first), lane, REPAIR_GROUP), 0)
    else:
        value, lse = _exact_row(
            Q, K, V, Directory, OverflowKeys, OverflowValues, OverflowUsed, Indices,
            Lengths, Slots, Counts, base, T, G, D, WIDTH, STATE, LEAVES, PAGES, INLINE, HASH, PROBES, N, SCALE,
        )
        tl.store(Out + base * D + dim, value)
        tl.store(LSE + base, lse)


@triton.jit
def _normalize(Numerator, Mass, Anchor, Out, LSE, ROWS: tl.constexpr, D: tl.constexpr,
               M: tl.constexpr, SHARDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * M + tl.arange(0, M)
    dim = tl.arange(0, D)
    valid = row < ROWS
    shard = tl.arange(0, SHARDS)
    index = row[:, None] * SHARDS + shard[None, :]
    mass = tl.sum(tl.load(Mass + index, mask=valid[:, None], other=0), 1)
    anchor = finite_anchor(tl.load(Anchor + row, mask=valid, other=0))
    value = tl.sum(tl.load(Numerator + index[:, :, None] * D + dim[None, None, :],
                           mask=valid[:, None, None], other=0), 1)
    tl.store(Out + row[:, None] * D + dim[None, :], value / tl.maximum(mass[:, None], 1.0e-30),
             mask=valid[:, None])
    tl.store(LSE + row, tl.where(mass > 0.0, anchor + tl.log(mass), -float("inf")), mask=valid)


def query_major_exact_attention(
    q, page_k, page_v, slot_pages, overflow_keys, overflow_values, overflow_used,
    page_indices, slot_lengths, slots, counts, *, group, scale, hash_probes,
    key_tile=32, warps=4, workspace, accumulation=None, query_group=1, query_order="adjacent",
):
    """Emit one exact branch per query. An accumulation supplies sparse repair work."""
    b, h, t, d = q.shape
    rows = b * h * t
    output = workspace('direct_exact_output', q.shape, q.dtype)
    lse = workspace('direct_exact_lse', q.shape[:-1], torch.float32)
    if query_group > 1:
        order = None
        grid = (triton.cdiv(t, query_group), b * h)
        if query_order != "adjacent":
            from .prefill_frontier_order import frontier_keys

            keys, _ = frontier_keys(slots, counts, slot_lengths, group=group, method=query_order)
            order = keys.argsort(dim=-1, stable=True)
            grid = (triton.cdiv(group * t, query_group), b * h // group)
        _grouped_query[grid](
            q.contiguous(), page_k, page_v, slot_pages, overflow_keys, overflow_values, overflow_used,
            page_indices, slot_lengths, slots, counts, order, output, lse,
            t, group, d, slots.size(-1), slot_pages.size(2), page_k.size(2), page_indices.size(2),
            slot_pages.size(3), overflow_values.size(2), hash_probes, query_group, key_tile, float(scale),
            order is not None, num_warps=warps, **hip_launch_kwargs(waves_per_eu=1),
        )
        return output, lse
    repair_group = 0
    repair = counts
    if accumulation is not None:
        numerator, mass, anchor, repair, shards = accumulation
        _normalize[(triton.cdiv(rows, 8),)](numerator, mass, anchor, output, lse, rows, d, 8, shards, num_warps=4)
        repair_group = 32
    grid = (triton.cdiv(rows, repair_group or 1),)
    _query_major[grid](
        q.contiguous(), page_k, page_v, slot_pages, overflow_keys, overflow_values, overflow_used,
        page_indices, slot_lengths, slots, counts, repair, output, lse, rows,
        t, group, d, slots.size(-1), slot_pages.size(2), page_k.size(2),
        page_indices.size(2), slot_pages.size(3), overflow_values.size(2),
        hash_probes, key_tile, float(scale), repair_group,
        num_warps=warps, **hip_launch_kwargs(waves_per_eu=1),
    )
    return output, lse
