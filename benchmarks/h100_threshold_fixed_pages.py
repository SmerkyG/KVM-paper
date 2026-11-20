"""BF16 two-page threshold refinement for Llama decode on H100."""

import math

import triton
import triton.language as tl

from lod_attention.kernels._paged_common import _lookup_page_id


@triton.jit
def _pick_page(score, page, ordinal, BLOCK: tl.constexpr):
    # Match the recursive decoder's tie order: first scan tile, largest page ID.
    best = tl.max(score, 0)
    tile = tl.min(tl.where(score == best, ordinal // BLOCK, 64), 0)
    winner = tl.max(tl.where((score == best) & (ordinal // BLOCK == tile), page, -1), 0)
    return tl.where(best > -float('inf'), winner, -1)


@triton.jit
def _select(PK, PC, Table, HashKeys, HashValues, HashUsed,
            kh, slot, num_pages, query, d,
            D: tl.constexpr, S: tl.constexpr, P: tl.constexpr, ROOTS: tl.constexpr,
            HASH: tl.constexpr, PROBES: tl.constexpr, SCALE: tl.constexpr,
            WIDTH: tl.constexpr, TIE_BLOCK: tl.constexpr):
    ordinal = tl.arange(0, WIDTH)
    active = ordinal < num_pages
    page = _lookup_page_id(Table, HashKeys, HashValues, HashUsed, kh, slot, ordinal,
                           active, S, ROOTS, P, HASH, PROBES).to(tl.int32)
    active = active & (page >= 0) & (page < P)
    count = tl.load(PC + kh * P + page, mask=active, other=1).to(tl.float32)
    key = tl.load(PK + (kh * P + page[:, None]) * D + d[None, :],
                  mask=active[:, None], other=0).to(tl.float32)
    score = SCALE * tl.sum((key / count[:, None]) * query[None, :], 1) + tl.log2(count)
    score = tl.where(active, score, -float('inf'))
    page0 = _pick_page(score, page, ordinal, TIE_BLOCK)
    page1 = _pick_page(tl.where(page != page0, score, -float('inf')), page, ordinal, TIE_BLOCK)
    return page0, page1


@triton.jit
def _fixed_pages(Q, SK, SV, SC, LK, LV, Indices, PK, PV, PC, Table, HashKeys, HashValues, HashUsed,
                 Lengths, Selected, RouteCount, Out, LSE,
                 G: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                 P: tl.constexpr, L: tl.constexpr, ROOTS: tl.constexpr, HASH: tl.constexpr,
                 PROBES: tl.constexpr, WORKERS: tl.constexpr, SCALE: tl.constexpr,
                 SMALL: tl.constexpr, TIE_BLOCK: tl.constexpr):
    head = tl.program_id(0)
    worker = tl.program_id(1)
    kh = head // G
    d = tl.arange(0, D)
    token = tl.arange(0, 32)
    query = tl.load(Q + head * D + d).to(tl.float32)
    maximum = tl.full((), -float('inf'), tl.float32)
    den = tl.full((), 0., tl.float32)
    acc = tl.full((D,), 0., tl.float32)
    nr = tl.load(RouteCount + head)
    for route in tl.range(worker, nr, WORKERS, num_stages=1):
        slot = tl.load(Selected + head * S + route)
        length = tl.load(Lengths + kh * S + slot)
        num_pages = (length + 15) // 16
        # Every page is exact when the region fits in the two-page budget.
        if num_pages <= 2 and num_pages > 0:
            page0 = _lookup_page_id(Table, HashKeys, HashValues, HashUsed, kh, slot, 0,
                                    True, S, ROOTS, P, HASH, PROBES).to(tl.int32)
            page1 = _lookup_page_id(Table, HashKeys, HashValues, HashUsed, kh, slot, 1,
                                    num_pages == 2, S, ROOTS, P, HASH, PROBES).to(tl.int32)
            page1 = tl.where(num_pages == 2, page1, -1)
        elif SMALL > 0 and num_pages <= SMALL:
            page0, page1 = _select(PK, PC, Table, HashKeys, HashValues, HashUsed,
                                   kh, slot, num_pages, query, d, D, S, P, ROOTS,
                                   HASH, PROBES, SCALE, SMALL, TIE_BLOCK)
        else:
            page0, page1 = _select(PK, PC, Table, HashKeys, HashValues, HashUsed,
                                   kh, slot, num_pages, query, d, D, S, P, ROOTS,
                                   HASH, PROBES, SCALE, 64, TIE_BLOCK)
        has0 = (page0 >= 0) & (page0 < P)
        has1 = (page1 >= 0) & (page1 < P)
        p0 = kh * P + tl.maximum(page0, 0)
        p1 = kh * P + tl.maximum(page1, 0)
        n0 = tl.load(PC + p0, mask=has0, other=0)
        n1 = tl.load(PC + p1, mask=has1, other=0)
        left = tl.load(SC + kh * S + slot).to(tl.float32) - n0 - n1
        if left > 0:
            sk = tl.load(SK + (kh * S + slot) * D + d).to(tl.float32)
            sv = tl.load(SV + (kh * S + slot) * D + d).to(tl.float32)
            pk0 = tl.load(PK + p0 * D + d, mask=has0, other=0).to(tl.float32)
            pk1 = tl.load(PK + p1 * D + d, mask=has1, other=0).to(tl.float32)
            pv0 = tl.load(PV + p0 * D + d, mask=has0, other=0).to(tl.float32)
            pv1 = tl.load(PV + p1 * D + d, mask=has1, other=0).to(tl.float32)
            rk = (sk - (pk0 + pk1)) / left
            rv = (sv - (pv0 + pv1)) / left
            score = tl.sum(rk * query, 0) * SCALE + tl.log2(left)
            nm = tl.maximum(maximum, score)
            correction = tl.exp2(maximum - nm)
            weight = tl.exp2(score - nm)
            acc = acc * correction + weight * rv
            den = den * correction + weight
            maximum = nm
        within = token % 16
        physical = tl.where(token < 16, p0, p1)
        count = tl.where(token < 16, n0, n1)
        valid = within < count
        leaf = tl.load(Indices + physical * 16 + within, mask=valid, other=-1)
        valid = valid & (leaf >= 0) & (leaf < L)
        leaf = tl.where(valid, leaf, 0)
        key = tl.load(LK + (kh * L + leaf[:, None]) * D + d[None, :],
                      mask=valid[:, None], other=0).to(tl.float32)
        value = tl.load(LV + (kh * L + leaf[:, None]) * D + d[None, :],
                        mask=valid[:, None], other=0).to(tl.float32)
        score = tl.where(valid, tl.sum(key * query[None, :], 1) * SCALE, -float('inf'))
        nm = tl.maximum(maximum, tl.max(score, 0))
        safe = tl.where(nm != -float('inf'), nm, 0.)
        correction = tl.exp2(maximum - safe)
        weight = tl.where(valid, tl.exp2(score - safe), 0.)
        acc = acc * correction + tl.sum(weight[:, None] * value, 0)
        den = den * correction + tl.sum(weight, 0)
        maximum = nm
    row = head * WORKERS + worker
    tl.store(Out + row * D + d, acc / tl.maximum(den, 1.e-30))
    tl.store(LSE + row, tl.where(den > 0, (maximum + tl.log2(tl.maximum(den, 1.e-30))) * 0.6931471805599453,
                                -float('inf')))


def fixed_two_page_attention(q, state_k, state_v, state_counts, leaf_k, leaf_v, page_indices,
                             page_sum_k, page_sum_v, page_counts, slot_pages, overflow_page_keys,
                             overflow_page_values, overflow_used, slot_lengths, top_slots, *,
                             kv_group_size, scale, hash_probes, route_counts, route_workers,
                             output_buffer, lse_buffer, page_block_n, num_warps, pages_per_slot,
                             small_scan=8, **kwargs):
    """Consume compact region IDs for a full batch in cache row order."""
    batch, heads, _, dim = q.shape
    if (kv_group_size != 4 or dim != 128 or pages_per_slot != 2 or 'quantized_leaf_k' in kwargs
            or state_k.size(0) != batch
            or not leaf_k.is_contiguous() or not leaf_v.is_contiguous()):
        raise ValueError('specialized page refinement requires BF16 Llama KV and two pages')
    if small_scan not in (0, 4, 8, 16, 32):
        raise ValueError('unsupported short page scan width')
    _fixed_pages[(batch * heads, route_workers)](
        q, state_k, state_v, state_counts, leaf_k, leaf_v, page_indices, page_sum_k, page_sum_v, page_counts,
        slot_pages, overflow_page_keys, overflow_page_values, overflow_used, slot_lengths, top_slots, route_counts,
        output_buffer, lse_buffer, kv_group_size, dim, state_k.size(2), page_counts.size(2), leaf_k.size(2),
        slot_pages.size(3), overflow_page_values.size(2), hash_probes, route_workers,
        scale * math.log2(math.e), small_scan, page_block_n, num_warps=num_warps)
    return output_buffer, lse_buffer
