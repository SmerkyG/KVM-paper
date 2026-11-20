"""Opt-in H100 threshold decode with shared GQA tensor-core exact attention.

Routing and closed-summary arithmetic are inherited from the indexed H100
implementation. Per-KV-head union lists contain physical indices and membership
bits, never copied K/V. All scans have linear work for fixed worker counts.
"""
import math
import os

import torch
import triton
import triton.language as tl

from benchmarks.h100_threshold_decode import _local, _merge_append, _scores
from benchmarks.h100_threshold_decode_fast import _threshold_pack, merge_threshold


@triton.jit
def _coarse_union(Native, Allowed, Counts, Values, Lengths, Slots, Ends, Bits,
                  Number, Totals, Coarse, LSE,
                  S: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                  TILE: tl.constexpr, NG: tl.constexpr):
    kh = tl.program_id(0)
    group = tl.program_id(1)
    lane = tl.arange(0, 16)
    head = kh * G + lane
    hm = lane < G
    idx = group * TILE + tl.arange(0, TILE)
    dim = tl.arange(0, D)
    opened = tl.load(Allowed + head[:, None] * S + idx[None, :],
                     mask=hm[:, None] & (idx[None, :] < S), other=0) != 0
    count = tl.load(Counts + kh * S + idx, mask=idx < S, other=0).to(tl.float32)
    valid = (idx < S) & (count > 0)
    member = tl.sum(opened.to(tl.int32) * (1 << lane[:, None]), 0)
    union = valid & (member != 0)
    rank = tl.cumsum(union.to(tl.int32), 0) - 1
    length = tl.load(Lengths + kh * S + idx, mask=idx < S, other=0)
    entries = tl.where(union, length, 0)
    ends = tl.cumsum(entries, 0)
    offset = (kh * NG + group) * TILE + rank
    tl.store(Slots + offset, idx, mask=union)
    tl.store(Ends + offset, ends, mask=union)
    tl.store(Bits + offset, member, mask=union)
    tl.store(Number + kh * NG + group, tl.sum(union.to(tl.int32), 0))
    tl.store(Totals + kh * NG + group, tl.sum(entries, 0))

    score = tl.load(Native + head[:, None] * S + idx[None, :],
                    mask=hm[:, None] & (idx[None, :] < S), other=-float('inf'))
    closed = hm[:, None] & valid[None, :] & ~opened
    score = tl.where(closed, score, -float('inf'))
    mx = tl.max(score, 1)
    safe = tl.where(mx != -float('inf'), mx, 0.)
    w = tl.where(closed, tl.exp(score - safe[:, None]), 0.)
    den = tl.sum(w, 1)
    value = tl.load(Values + (kh * S + idx[:, None]) * D + dim[None, :],
                    mask=valid[:, None], other=0).to(tl.float32)
    mean = (value / tl.maximum(count.to(tl.bfloat16).to(tl.float32)[:, None], 1.)).to(tl.bfloat16)
    result = tl.dot(w.to(tl.bfloat16), mean, out_dtype=tl.float32) / tl.maximum(den[:, None], 1.e-30)
    row = head * NG + group
    tl.store(Coarse + row[:, None] * D + dim[None, :], result, mask=hm[:, None])
    tl.store(LSE + row, tl.where(den > 0, safe + tl.log(den), -float('inf')), mask=hm)


@triton.jit
def _materialize(Slots, Ends, Bits, Number, Totals, Roots, Directory, Indices,
                 Flat, FlatBits, Total,
                 S: tl.constexpr, TILE: tl.constexpr, NG: tl.constexpr,
                 NGB: tl.constexpr, ROOTS: tl.constexpr, PAGECAP: tl.constexpr,
                 LEAFCAP: tl.constexpr, WORKERS: tl.constexpr,
                 SHARDS: tl.constexpr, BLOCK: tl.constexpr):
    kh = tl.program_id(0)
    worker = tl.program_id(1)
    shard = tl.program_id(2)
    groups = tl.arange(0, NGB)
    totals = tl.load(Totals + kh * NG + groups, mask=groups < NG, other=0)
    prefix = tl.cumsum(totals, 0)
    token = tl.arange(0, BLOCK)
    for group in range(worker, NG, WORKERS):
        before = tl.sum(tl.gather(prefix, tl.maximum(group - 1, 0)[None], 0), 0)
        before = tl.where(group > 0, before, 0)
        count = tl.load(Number + kh * NG + group)
        for rank in range(shard, count, SHARDS):
            i = (kh * NG + group) * TILE + rank
            slot = tl.load(Slots + i)
            end = tl.load(Ends + i)
            prev = tl.load(Ends + i - 1, mask=rank > 0, other=0)
            length = end - prev
            bits = tl.load(Bits + i)
            for start in range(0, length, BLOCK):
                ordinal = start + token
                page_ordinal = ordinal // 16
                root = tl.load(Roots + (kh * S + slot) * ROOTS + page_ordinal // 64,
                               mask=(ordinal < length) & (page_ordinal // 64 < ROOTS), other=-1)
                root_ok = (ordinal < length) & (root >= 0) & (root < PAGECAP)
                page = tl.load(Directory + (kh * PAGECAP + tl.maximum(root, 0)) * 64 + page_ordinal % 64,
                               mask=root_ok, other=-1)
                page_ok = root_ok & (page >= 0) & (page < PAGECAP)
                physical = tl.load(Indices + (kh * PAGECAP + tl.maximum(page, 0)) * 16 + ordinal % 16,
                                   mask=page_ok, other=-1)
                physical = tl.where(page_ok & (physical >= 0) & (physical < LEAFCAP), physical, -1)
                dst = kh * LEAFCAP + before + prev + ordinal
                tl.store(Flat + dst, physical, mask=ordinal < length)
                tl.store(FlatBits + dst, bits, mask=ordinal < length)
    if worker == 0 and shard == 0:
        tl.store(Total + kh, tl.sum(totals, 0))


@triton.jit
def _consume(Q, Flat, Bits, Total, K, V, Out, LSE,
             G: tl.constexpr, D: tl.constexpr, LEAFCAP: tl.constexpr,
             N: tl.constexpr, WORKERS: tl.constexpr, SCALE: tl.constexpr,
             PRECISE: tl.constexpr):
    kh = tl.program_id(0)
    worker = tl.program_id(1)
    lane = tl.arange(0, 16)
    head = kh * G + lane
    hm = lane < G
    dim = tl.arange(0, D)
    token = tl.arange(0, N)
    q = tl.load(Q + head[:, None] * D + dim[None, :], mask=hm[:, None], other=0)
    total = tl.load(Total + kh)
    tiles = tl.cdiv(total, N)
    per = tl.cdiv(tiles, WORKERS)
    mx = tl.full((16,), -float('inf'), tl.float32)
    den = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, D), tl.float32)
    for tile in range(worker * per, tl.minimum((worker + 1) * per, tiles)):
        entry = tile * N + token
        physical = tl.load(Flat + kh * LEAFCAP + entry, mask=entry < total, other=-1)
        bits = tl.load(Bits + kh * LEAFCAP + entry, mask=entry < total, other=0).to(tl.int32)
        valid = (entry < total) & (physical >= 0) & (physical < LEAFCAP)
        key = tl.load(K + (kh * LEAFCAP + tl.maximum(physical, 0))[None, :] * D + dim[:, None],
                      mask=valid[None, :], other=0)
        member = hm[:, None] & valid[None, :] & ((bits[None, :] & (1 << lane[:, None])) != 0)
        score = tl.where(member, tl.dot(q, key, out_dtype=tl.float32) * (SCALE * 1.4426950408889634), -float('inf'))
        nm = tl.maximum(mx, tl.max(score, 1))
        safe = tl.where(nm != -float('inf'), nm, 0.)
        adjust = tl.exp2(mx - safe)
        w = tl.where(member, tl.exp2(score - safe[:, None]), 0.)
        value = tl.load(V + (kh * LEAFCAP + tl.maximum(physical, 0))[:, None] * D + dim[None, :],
                        mask=valid[:, None], other=0)
        acc = acc * adjust[:, None] + tl.dot(w.to(tl.bfloat16), value, out_dtype=tl.float32)
        if PRECISE:
            acc += tl.dot((w - w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), value, out_dtype=tl.float32)
        den = den * adjust + tl.sum(w, 1)
        mx = nm
    row = head * WORKERS + worker
    tl.store(Out + row[:, None] * D + dim[None, :], acc / tl.maximum(den[:, None], 1.e-30), mask=hm[:, None])
    tl.store(LSE + row, tl.where(den > 0, (mx + tl.log2(den)) * 0.6931471805599453, -float('inf')), mask=hm)


def threshold_attention_shared(pool, q, k, v, out, multiplier, *, tile=64, splits=32,
                               rule='region', entry_bound=16, work_bound=None):
    if work_bound is None:
        work_bound = entry_bound
    if rule not in ('region', 'entry') or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('invalid threshold rule or multiplier')
    if any(not isinstance(x, int) or x < 1 for x in (entry_bound, work_bound, splits)):
        raise ValueError('bounds and splits must be positive integers')
    if q.shape != (1, 1, 32, 128) or pool.kv_heads != 8 or pool.settings.levels != 2 or pool.dtype != torch.bfloat16:
        raise ValueError('shared H100 path requires batch-one BF16 Llama GQA two-tier')
    st = pool.state
    page = st['page_cache']
    if not page.get('paged_page_directory'):
        raise ValueError('shared H100 path requires directory cache')
    S, H, KV, G, D = pool.state_capacity, 32, 8, 4, 128
    ng = triton.cdiv(S, tile)
    leafcap = page['leaf_k'].size(2)
    localcap = st['recent_k'].size(2)
    limit = int(pool.decode_local_limit)
    sink = st['sink_k'].size(2)
    nl = triton.cdiv(limit + 1 + sink, tile)
    spec = (S, tile, splits, leafcap, nl)
    b = getattr(pool, '_threshold_shared_scratch', None)
    if b is None or b['spec'] != spec:
        def f(*shape): return torch.empty(shape, device=q.device, dtype=torch.float32)
        def i(*shape): return torch.empty(shape, device=q.device, dtype=torch.int32)
        b = dict(spec=spec, native=f(H, S), route=f(H, S), threshold=f(H), allowed=i(H, S),
                 selected=i(H, S), count=i(H), offsets=i(H, S), entry_ends=i(H, S), total_entries=i(H),
                 total_pages=i(H), saved=i(1), co=f(H, ng, D), cl=f(H, ng),
                 eo=f(H, splits, D), el=f(H, splits), lo=f(H, nl, D), ll=f(H, nl),
                 union_slots=i(KV, ng, tile), union_ends=i(KV, ng, tile), union_bits=i(KV, ng, tile),
                 union_count=i(KV, ng), union_total=i(KV, ng), flat=i(KV, leafcap),
                 flat_bits=torch.empty((KV, leafcap), device=q.device, dtype=torch.uint8), flat_total=i(KV))
        pool._threshold_shared_scratch = b
    pool._threshold_scratch = b
    scale = float(pool.engine.scaling)
    _scores[(KV, ng)](q, st['state_k'], st['counts'], pool.state_lens, pool.local_lens,
                      b['native'], b['route'], b['saved'], S, KV, G, D, scale, tile, num_warps=4)
    _threshold_pack[(H,)](b['route'], page['slot_lengths'], st['counts'], b['threshold'], b['allowed'],
                         b['selected'], b['count'], b['offsets'], b['total_pages'], b['entry_ends'], b['total_entries'],
                         S, G, float(multiplier), rule == 'entry', entry_bound, work_bound,
                         triton.next_power_of_2(S), False, num_warps=4)
    _coarse_union[(KV, ng)](b['native'], b['allowed'], st['counts'], st['state_v'], page['slot_lengths'],
                           b['union_slots'], b['union_ends'], b['union_bits'], b['union_count'], b['union_total'],
                           b['co'], b['cl'], S, G, D, tile, ng, num_warps=4)
    workers = int(os.environ.get('LOD_H100_SHARED_PACK_WORKERS', '16'))
    shards = int(os.environ.get('LOD_H100_SHARED_PACK_SHARDS', '4'))
    if workers < 1 or shards < 1:
        raise ValueError('packing worker and shard counts must be positive')
    _materialize[(KV, workers, shards)](b['union_slots'], b['union_ends'], b['union_bits'], b['union_count'],
                                       b['union_total'], page['slot_pages'], page['overflow_page_values'], page['page_indices'],
                                       b['flat'], b['flat_bits'], b['flat_total'], S, tile, ng, triton.next_power_of_2(ng),
                                       page['slot_pages'].size(3), page['page_indices'].size(2), leafcap,
                                       workers, shards, 128, num_warps=4)
    block = int(os.environ.get('LOD_H100_SHARED_ENTRY_BLOCK', '64'))
    _consume[(KV, splits)](q, b['flat'], b['flat_bits'], b['flat_total'], page['leaf_k'], page['leaf_v'], b['eo'], b['el'],
                           G, D, leafcap, block, splits, scale,
                           os.environ.get('LOD_H100_SHARED_PRECISE', '1') == '1',
                           num_warps=4, num_stages=1)
    _local[(H, nl)](q, st['recent_k'], st['recent_v'], st['sink_k'], st['sink_v'], k, v, b['saved'], b['lo'], b['ll'],
                    KV, G, D, localcap, limit, sink, scale, tile, nl, num_warps=4)
    merge_threshold(b, k, v, st, pool.local_lens, out, D, G, KV, localcap, ng, splits, nl)
    return out
