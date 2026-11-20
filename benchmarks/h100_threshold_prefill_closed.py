"""H100 threshold prefill with saved scores and direct closed-summary attention.

Preserves every selected region, including the deterministic entry-work guard.
Packed expert dispatch reads actual route counts. Persistent exact workers and
route-count-aware reduction avoid worst-case padded launches and scans.
"""
import math
import os

import torch
import triton
import triton.language as tl

from lod_attention.kernels.aiter_prefill_attention import _workspace_tensor
from lod_attention.kernels.threshold_prefill import _qk, _pv, _expert_metadata, _expert_prefix
from lod_attention.kernels.paged_prefill import paged_leaf_attention
from lod_attention.kernels.lod_kernels import merge_attention_branches_with_sink


@triton.jit
def _prepare(K, V, C, MK, MV, LogC, S: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    kh, slot = row // S, row % S
    dim = tl.arange(0, D)
    count = tl.load(C + kh * CAP + slot).to(tl.float32)
    divisor = tl.maximum(count.to(tl.bfloat16).to(tl.float32), 1.)
    offset = (kh * CAP + slot) * D + dim
    tl.store(MK + row * D + dim, tl.load(K + offset).to(tl.float32) / divisor)
    tl.store(MV + row * D + dim, tl.load(V + offset).to(tl.float32) / divisor)
    tl.store(LogC + row, tl.log(tl.maximum(count, 1.)))


@triton.jit
def _select(Q, Scores, C, LogC, Probability, LSE, Slots, Counts, Entries, Mask, Ranks, HeadCounts, Offsets,
            T: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            CUTOFF: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr, SCALE: tl.constexpr,
            ENTRY: tl.constexpr, BOUND: tl.constexpr, WORK: tl.constexpr, ATOMIC_METADATA: tl.constexpr,
            ROWS: tl.constexpr, M: tl.constexpr, SKIP_METADATA: tl.constexpr, AGGREGATE: tl.constexpr):
    if AGGREGATE:
        query = tl.program_id(1) * M + tl.arange(0, M)
        row = tl.program_id(0).to(tl.int64) * T + query
        row_valid = query < T
    else:
        row = tl.program_id(0).to(tl.int64) * M + tl.arange(0, M)
        row_valid = row < ROWS
    kh = row // T // G
    row = row[:, None]
    kh = kh[:, None]
    slot = tl.arange(0, BLOCK)[None, :]
    row_valid = row_valid[:, None]
    count = tl.load(C + kh * CAP + slot, mask=row_valid & (slot < S), other=0).to(tl.float32)
    valid = row_valid & (slot < S) & (count > 0)
    dot = tl.load(Scores + row * S + slot, mask=row_valid & (slot < S), other=0) * SCALE
    dim = tl.arange(0, D)[None, :]
    q = tl.load(Q + row * D + dim, mask=row_valid, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.maximum(tl.sum(q * q, 1) / D, 1.e-12))[:, None]
    logc = tl.load(LogC + kh * S + slot, mask=row_valid & (slot < S), other=0)
    route = tl.where(valid, dot * inverse + logc, -float('inf'))
    mx = tl.max(route, 1)[:, None]
    safe = tl.where(mx != -float('inf'), mx, 0.)
    mass = tl.sum(tl.where(valid, tl.exp(route - safe), 0.), 1)[:, None]
    if ENTRY:
        size = count.to(tl.int32)
        remote = tl.sum(tl.where(valid, size, 0), 1)[:, None]
        cutoff = safe + tl.log(mass) - tl.log(BOUND * 1.) - .5 * tl.log(tl.maximum(remote.to(tl.float32), 1.)) + 1.e-6
        candidate = valid & (dot * inverse > cutoff)
        prefix = tl.cumsum(tl.where(candidate, size, 0), 1)
        opened = candidate & (prefix.to(tl.int64) * prefix.to(tl.int64) < WORK * WORK * remote.to(tl.int64))
    else:
        opened = valid & (count <= 1024) & (tl.exp(route - safe) >
                    CUTOFF * mass / tl.maximum(tl.sum(valid.to(tl.int32), 1)[:, None], 1))
    n = tl.sum(opened.to(tl.int32), 1)[:, None]
    tl.store(Counts + row, n, mask=row_valid)
    tl.store(Entries + row, tl.sum(tl.where(opened, count, 0), 1)[:, None], mask=row_valid)
    rank = tl.cumsum(opened.to(tl.int32), 1) - 1
    tl.store(Slots + row * WIDTH + tl.maximum(rank, 0), slot, mask=opened)
    if SKIP_METADATA:
        pass
    elif ATOMIC_METADATA:
        if AGGREGATE:
            total = tl.sum(opened.to(tl.int32), 0)[None, :]
            base = tl.atomic_add(HeadCounts + tl.program_id(0) * S + slot, total,
                                 mask=(slot < S) & (total > 0), sem='relaxed')
            offset = base + tl.cumsum(opened.to(tl.int32), 0) - 1
        else:
            offset = tl.atomic_add(HeadCounts + row // T * S + slot, 1, mask=opened, sem='relaxed')
        tl.store(Offsets + row * WIDTH + tl.maximum(rank, 0), offset, mask=opened)
    else:
        tl.store(Ranks + row * S + slot, rank.to(tl.int16), mask=opened)
        bit = tl.arange(0, 8)
        bits = tl.sum(tl.reshape(opened.to(tl.int32), (M, BLOCK // 8, 8)) * (1 << bit[None, None, :]), 2).to(tl.uint8)
        byte = tl.arange(0, BLOCK // 8)[None, :]
        tl.store(Mask + row * tl.cdiv(S, 8) + byte, bits, mask=row_valid & (byte < tl.cdiv(S, 8)))
    closed = valid & ~opened
    native = tl.where(closed, dot + logc, -float('inf'))
    nm = tl.max(native, 1)[:, None]
    nsafe = tl.where(nm != -float('inf'), nm, 0.)
    w = tl.where(closed, tl.exp(native - nsafe), 0.)
    den = tl.sum(w, 1)[:, None]
    tl.store(LSE + row, tl.where(den > 0, nsafe + tl.log(den), -float('inf')), mask=row_valid)
    tl.store(Probability + row * S + slot, w / tl.maximum(den, 1.e-30), mask=row_valid & (slot < S))


@triton.jit
def _finish(Slots, Counts, Offsets, TileStarts,
            T: tl.constexpr, S: tl.constexpr, WIDTH: tl.constexpr, QTILE: tl.constexpr):
    head = tl.program_id(0)
    query = tl.program_id(1) * 4 + tl.arange(0, 4)
    rank = tl.arange(0, 32)
    count = tl.load(Counts + head * T + query, mask=query < T, other=0)
    for start in range(0, tl.max(count, 0), 32):
        r = start + rank
        valid = (query[:, None] < T) & (r[None, :] < count[:, None])
        position = (head * T + query[:, None]) * WIDTH + r[None, :]
        slot = tl.load(Slots + position, mask=valid, other=0).to(tl.int32)
        local = tl.load(Offsets + position, mask=valid, other=0)
        before = tl.load(TileStarts + (head * S + slot) * tl.cdiv(T, QTILE) + query[:, None] // QTILE,
                         mask=valid, other=0)
        tl.store(Offsets + position, local + before, mask=valid)


@triton.jit
def _reduce_grouped(Values, LSE, Counts, Offsets, Output, OutputLSE,
                    ROWS: tl.constexpr, WIDTH: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
                    PACKED: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * M + tl.arange(0, M)
    count = tl.load(Counts + row, mask=row < ROWS, other=0)
    rank = tl.arange(0, 16)
    dim = tl.arange(0, D)
    mx = tl.full((M,), -float('inf'), tl.float32)
    den = tl.zeros((M,), tl.float32)
    num = tl.zeros((M, D), tl.float32)
    for start in range(0, tl.max(count, 0), 16):
        r = start + rank
        valid = (row[:, None] < ROWS) & (r[None, :] < count[:, None])
        position = row[:, None] * WIDTH + r[None, :]
        if PACKED:
            position = tl.load(Offsets + position, mask=valid, other=0).to(tl.int64)
        score = tl.load(LSE + position, mask=valid, other=-float('inf'))
        value = tl.load(Values + (position[:, :, None] * D + dim[None, None, :]),
                        mask=valid[:, :, None], other=0).to(tl.float32)
        nxt = tl.maximum(mx, tl.max(score, 1))
        safe = tl.where(nxt != -float('inf'), nxt, 0.)
        adjust = tl.where(mx != -float('inf'), tl.exp(mx - safe), 0.)
        w = tl.where(valid, tl.exp(score - safe[:, None]), 0.)
        den = den * adjust + tl.sum(w, 1)
        num = num * adjust[:, None] + tl.sum(w[:, :, None] * value, 1)
        mx = nxt
    tl.store(Output + row[:, None] * D + dim[None, :], num / tl.maximum(den[:, None], 1.e-30), mask=row[:, None] < ROWS)
    tl.store(OutputLSE + row, tl.where(den > 0, mx + tl.log(den), -float('inf')), mask=row < ROWS)


@triton.jit
def _reduce_split_d(Values,LSE,Counts,Output,OutputLSE,
                    WIDTH:tl.constexpr,R:tl.constexpr,D:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0).to(tl.int64);block=tl.program_id(1)
    count=tl.load(Counts+row);r=tl.arange(0,R)
    score=tl.load(LSE+row*WIDTH+r,mask=r<count,other=-float('inf'))
    maximum=tl.max(score,0);safe=tl.where(maximum!=-float('inf'),maximum,0.)
    den=tl.sum(tl.where(r<count,tl.exp(score-safe),0.),0)
    dim=block*BD+tl.arange(0,BD);lane=tl.arange(0,16)
    acc=tl.zeros((BD,),tl.float32)
    for start in range(tl.cdiv(count,16)):
        rank=start*16+lane
        tile_score=tl.load(LSE+row*WIDTH+rank,mask=rank<count,other=-float('inf'))
        weight=tl.where(rank<count,tl.exp(tile_score-safe),0.)
        value=tl.load(Values+(row*WIDTH+rank[:,None])*D+dim[None,:],
            mask=(rank[:,None]<count)&(dim[None,:]<D),other=0).to(tl.float32)
        acc+=tl.sum(weight[:,None]*value,0)
    tl.store(Output+row*D+dim,acc/tl.maximum(den,1.e-30),mask=dim<D)
    if block==0:tl.store(OutputLSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')))


def route_coarse(q, state_k, state_v, counts, *, state_len, scale, group, rule,
                 multiplier=16., entry_bound=16, work_bound=16, leaf_capacity,
                 buffers=None, prepared=None, timing_events=None):
    """Return uncapped route IDs, counts, closed summaries and grouped offsets."""
    h, t, d = q.size(1), q.size(2), q.size(3)
    kv, cap, s = state_k.size(1), state_k.size(2), state_len
    width = min(s, 1 + math.isqrt(work_bound * work_bound * leaf_capacity)) if rule == 'entry' else min(s, math.ceil(s / multiplier) + 1)
    def ws(name, shape, dtype=q.dtype):
        return _workspace_tensor(buffers, 'closed_' + name, shape, dtype=dtype, device=q.device)
    first = None
    if timing_events is not None:
        first = torch.cuda.Event(enable_timing=True)
        first.record()
    def mark(name):
        nonlocal first
        if timing_events is not None:
            last = torch.cuda.Event(enable_timing=True)
            last.record()
            timing_events.setdefault(name, []).append((first, last))
            first = last
    if prepared is None:
        mk, mv = ws('mean_k', (1, kv, s, d)), ws('mean_v', (1, kv, s, d))
        logc = ws('log_count', (kv, s), torch.float32)
        _prepare[(kv * s,)](state_k, state_v, counts, mk, mv, logc, s, cap, d, num_warps=4)
    else:
        mk, mv, logc = prepared
    mark('closed_prepare')
    scores = ws('scores', (h, t, s), torch.float32)
    p = ws('probabilities', (h, t, s))
    coarse = ws('coarse', (1, h, t, d))
    lse = ws('coarse_lse', (1, h, t), torch.float32)
    slots = ws('slots', (1, h, t, width), torch.int64)
    number = ws('counts', (1, h, t), torch.int32)
    entries = ws('entries', (1, h, t), torch.float32)
    atomic = os.environ.get('LOD_H100_CLOSED_METADATA', 'scan') == 'atomic'
    skip_metadata = os.environ.get('LOD_H100_CLOSED_METADATA') == 'none'
    if skip_metadata and os.environ.get('LOD_H100_CLOSED_EXACT') != 'indexed':
        raise ValueError('metadata can be skipped only for per-query indexed attention')
    bits = ws('bits', (1,) if atomic else (h, t, triton.cdiv(s, 8)), torch.uint8)
    ranks = ws('ranks', (1,) if atomic else (h, t, s), torch.int16)
    offsets = ws('offsets', slots.shape, torch.int32)
    qm = int(os.environ.get('LOD_H100_CLOSED_GROUP_M', '64'))
    sn = int(os.environ.get('LOD_H100_CLOSED_GROUP_N', '128'))
    tc = ws('tile_counts', (h * s, triton.cdiv(t, qm)), torch.int32)
    hc = ws('head_counts', (h * s,), torch.int32)
    if atomic:
        hc.zero_()
    backend = os.environ.get('LOD_H100_CLOSED_MATMUL', 'triton')
    if backend == 'blas':
        from benchmarks.h100_blas import qk_fp32
        qk_fp32(q.view(kv, group * t, d), mk.view(kv, s, d), scores.view(kv, group * t, s))
    elif backend == 'triton':
        _qk[(h, triton.cdiv(t, 32), triton.cdiv(s, 64))](q, mk, scores, t, s, h, kv, group, d, scale, 32, 64, num_warps=4)
    else:
        raise ValueError('closed prefill matmul must be triton or blas')
    mark('closed_qk')
    sm = int(os.environ.get('LOD_H100_CLOSED_SELECT_QM', '1'))
    if sm not in (1, 2, 4, 8):
        raise ValueError('selection query group must be one, two, four or eight')
    aggregate = os.environ.get('LOD_H100_PREFILL_AGGREGATE', '0') == '1'
    if aggregate and (not atomic or skip_metadata):
        raise ValueError('aggregated prefill metadata requires atomic expert packing')
    grid = (h, triton.cdiv(t, sm)) if aggregate else (triton.cdiv(h * t, sm),)
    _select[grid](q, scores, counts, logc, p, lse, slots, number, entries, bits, ranks, hc, offsets,
                       t, s, cap, group, d, float(multiplier), width, triton.next_power_of_2(max(s, 8)),
                       float(scale), rule == 'entry', entry_bound, work_bound, atomic, h * t, sm, skip_metadata, aggregate,
                       num_warps=int(os.environ.get('LOD_H100_CLOSED_SELECT_WARPS', '8')))
    mark('closed_select')
    if not atomic and not skip_metadata:
        _expert_metadata[(h, triton.cdiv(t, qm), triton.cdiv(s, sn))](bits, ranks, offsets, tc, t, s, width, qm, sn,
            num_warps=int(os.environ.get('LOD_H100_CLOSED_GROUP_WARPS', '8')))
        _expert_prefix[(h * s,)](tc, hc, triton.cdiv(t, qm), triton.next_power_of_2(triton.cdiv(t, qm)), num_warps=4)
        _finish[(h, triton.cdiv(t, 4))](slots, number, offsets, tc, t, s, width, qm, num_warps=4)
    mark('closed_metadata')
    if backend == 'blas':
        torch.bmm(p.view(kv, group * t, s), mv.view(kv, s, d), out=coarse.view(kv, group * t, d))
    else:
        _pv[(h, triton.cdiv(t, 32), triton.cdiv(d, 64))](p, mv, coarse, t, s, h, kv, group, d, 32, 64, 64, num_warps=4)
    mark('closed_pv')
    return slots, number, entries, coarse, lse, None if skip_metadata else hc, None if skip_metadata else offsets


def exact_attention(engine, q, page, state_len, routed, buffers, *, entry_capacity=None, return_routes=False):
    slots, number, _, _, _, hc, offsets = routed
    backend = os.environ.get('LOD_H100_CLOSED_EXACT', 'grouped')
    if backend == 'indexed':
        from benchmarks.h100_threshold_prefill_fused import _expand_direct_indices, _indexed_direct_attention
        h, t, d = q.shape[1:]
        def ws(name, shape, dtype):
            return _workspace_tensor(buffers, 'closed_indexed_' + name, shape, dtype=dtype, device=q.device)
        width = triton.next_power_of_2(slots.size(-1))
        padded = ws('slots', (*slots.shape[:-1], width), slots.dtype)
        padded[..., :slots.size(-1)].copy_(slots)
        entries = routed[2].to(torch.int32)
        prefix = ws('prefix', (h * t + 1,), torch.int64)
        prefix[:1].zero_()
        torch.cumsum(entries.reshape(-1), 0, dtype=torch.int64, out=prefix[1:])
        lk, lv = page['leaf_k'], page['leaf_v']
        capacity = lk.size(2)
        bound = capacity if entry_capacity is None else min(capacity, entry_capacity)
        indices = ws('indices', (h * t * bound,), torch.int32)
        roots, directory, pages = page['slot_pages'], page['overflow_page_values'], page['page_indices']
        _expand_direct_indices[(h * t,)](padded, number, page['slot_lengths'], roots,
            page['overflow_page_keys'], directory, page['overflow_used'], pages, indices, entries,
            number, prefix, t, width, page['slot_lengths'].size(2), engine.num_key_value_groups,
            roots.size(3), directory.size(2), pages.size(2), capacity, engine._page_lookup_probes(page),
            capacity, width.bit_length(), True, True, 256, num_warps=4)
        output, lse = torch.empty_like(q), torch.empty_like(number, dtype=torch.float32)
        _indexed_direct_attention[(h * t,)](q, indices, entries, prefix, lk, lv, output, lse,
            t, capacity, engine.num_key_value_groups, capacity, d, float(engine.scaling), 128, True, num_warps=4)
        return output, lse
    if backend != 'grouped':
        raise ValueError('closed prefill exact backend must be grouped or indexed')
    reducer = os.environ.get('LOD_H100_CLOSED_REDUCE', 'wide')
    packed = os.environ.get('LOD_H100_CLOSED_PACKED_OUTPUT') == '1'
    if return_routes and packed:
        raise ValueError('fused merge requires query-major route outputs')
    if packed and reducer != 'grouped':
        raise ValueError('packed exact outputs require grouped reduction')
    result = paged_leaf_attention(q, page['leaf_k'], page['leaf_v'], page['slot_pages'],
        page['overflow_page_keys'], page['overflow_page_values'], page['overflow_used'],
        page['slot_lengths'], slots, page_indices=page['page_indices'],
        kv_group_size=engine.num_key_value_groups, active_slots=state_len, scale=engine.scaling,
        route_head_counts=hc, route_offsets=offsets, route_counts=number,
        persistent_workers=int(os.environ.get('LOD_H100_CLOSED_WORKERS', '264')),
        variable_dispatch=True, variable_reduce_backend='wide' if reducer in ('grouped','split-d','tc-wide') else reducer,
        packed_output=packed,
        reduce_routes=not return_routes and reducer not in ('grouped','split-d','bucketed','tc-wide'),
        block_m=int(os.environ.get('LOD_H100_CLOSED_LEAF_M', str(engine.leaf_block_m))),
        block_n=int(os.environ.get('LOD_H100_CLOSED_LEAF_N', str(engine.leaf_block_n))),
        num_warps=int(os.environ.get('LOD_H100_CLOSED_LEAF_WARPS', str(engine.leaf_num_warps))),
        hash_probes=engine._page_lookup_probes(page), buffers=buffers,
        timing_events=getattr(engine, '_lod_leaf_timing_events', None))
    if return_routes:
        return result
    if reducer=='tc-wide':
        if q.dtype!=torch.bfloat16 or q.size(-1)!=128:
            raise ValueError('tensor-core frontier reduction requires BF16 D128 queries')
        from benchmarks.h100_prefill_tc_reduce import reduce
        from benchmarks.h100_prefill_split_reduce import reduce as fallback
        output=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
        events=getattr(engine,'_lod_leaf_timing_events',None)
        if events is not None:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        rows=q.numel()//q.size(-1);m=int(os.environ.get('LOD_H100_TC_REDUCE_ROWS','4'))
        if m not in (4,8,16):raise ValueError('TC reducer supports four, eight, or sixteen queries')
        parts=int(os.environ.get('LOD_H100_TC_REDUCE_PARTS','3'))
        small=int(os.environ.get('LOD_H100_TC_REDUCE_RANKS','32'))
        if small not in (16,32,64):raise ValueError('TC reduction supports sixteen, thirty-two or sixty-four common routes')
        if parts not in (2,3):raise ValueError('TC reduction requires two or three BF16 parts')
        reduce[(triton.cdiv(rows,m),)](result[0],result[1],number,output,lse,rows,
            slots.size(-1),q.size(-1),m,PARTS=parts,RANKS=small,
            num_warps=4,num_stages=1)
        fallback[(rows,)](result[0],result[1],number,output,lse,
            slots.size(-1),triton.next_power_of_2(slots.size(-1)),q.size(-1),False,MIN_COUNT=small,num_warps=4)
        if events is not None:
            end.record();events.setdefault('reduce',[]).append((start,end))
        return output,lse
    if reducer=='bucketed':
        from benchmarks.h100_prefill_reduce_buckets import reduce
        output=torch.empty_like(q);lse=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
        events=getattr(engine,'_lod_leaf_timing_events',None)
        if events is not None:
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        reduce(result[0],result[1],number,output,lse,width=slots.size(-1),d=q.size(-1),buffers=buffers)
        if events is not None:
            end.record();events.setdefault('reduce',[]).append((start,end))
        return output,lse
    if reducer not in ('grouped','split-d'):
        return result
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    rows = q.numel() // q.size(-1)
    if reducer=='split-d':
        bd=int(os.environ.get('LOD_H100_PREFILL_REDUCE_D','32'))
        if bd not in (16,32,64):raise ValueError('split prefill reduction requires 16/32/64 value dimensions')
        _reduce_split_d[(rows,triton.cdiv(q.size(-1),bd))](result[0],result[1],number,output,lse,
            slots.size(-1),triton.next_power_of_2(slots.size(-1)),q.size(-1),bd,num_warps=4)
        return output,lse
    m = int(os.environ.get('LOD_H100_CLOSED_REDUCE_QM', '4'))
    if m not in (1, 2, 4, 8):
        raise ValueError('grouped reduction handles one, two, four or eight queries per CTA')
    events = getattr(engine, '_lod_leaf_timing_events', None)
    if events is not None:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
    _reduce_grouped[(triton.cdiv(rows, m),)](result[0], result[1], number, offsets, output, lse,
        rows, slots.size(-1), q.size(-1), m, packed, num_warps=4)
    if events is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        events.setdefault('reduce', []).append((start, end))
    return output, lse


def verify_frontier(engine, q, counts, page, state_len, prepared, routed, exact, exact_lse,
                    rule, multiplier, entry_bound, work_bound):
    """Untimed opt-in checks on real model states using independent operators."""
    from benchmarks.h100_threshold_prefill_fused import _direct_leaf_attention
    index = torch.tensor(sorted({0, q.size(2) // 3, 2 * q.size(2) // 3, q.size(2) - 1}), device=q.device)
    query = q.index_select(2, index).contiguous()
    slots = routed[0].index_select(2, index).contiguous()
    number = routed[1].index_select(2, index).contiguous()
    size = counts[0, :, :state_len, 0].float().repeat_interleave(engine.num_key_value_groups, 0)[None, :, None, :]
    mk, mv = (v.repeat_interleave(engine.num_key_value_groups, 1).float() for v in prepared[:2])
    native = query.float() @ mk.transpose(-1, -2) * engine.scaling
    inverse = query.float().square().mean(-1, keepdim=True).clamp_min(1.e-12).rsqrt()
    valid = size > 0
    logits = (native * inverse + size.clamp_min(1).log()).masked_fill(~valid, -torch.inf)
    if rule == 'region':
        margin = logits - logits.logsumexp(-1, keepdim=True) - math.log(multiplier) + valid.sum(-1, keepdim=True).clamp_min(1).float().log()
        expected = valid & (size <= 1024) & (margin > 0)
    else:
        remote = size.sum(-1, keepdim=True).long()
        cutoff = logits.logsumexp(-1, keepdim=True) - math.log(entry_bound) - .5 * remote.clamp_min(1).float().log() + 1.e-6
        margin = native * inverse - cutoff
        candidate = valid & (margin > 0)
        expected = candidate & ((candidate * size.long()).cumsum(-1).square() < work_bound * work_bound * remote)
    active = torch.arange(slots.size(-1), device=q.device) < number[..., None]
    actual = torch.zeros_like(expected, dtype=torch.int32)
    actual.scatter_add_(-1, torch.where(active, slots, 0), active.int())
    mismatch = actual.bool() != expected
    mismatches = int(mismatch.sum())
    away = int((mismatch & (margin.abs() > 2.e-5)).sum())
    if away:
        raise AssertionError(f'{away} real-state threshold mismatches away from rounding boundaries')
    closed = (native + size.clamp_min(1).log()).masked_fill(~valid | actual.bool(), -torch.inf)
    reference_coarse = closed.softmax(-1).nan_to_num() @ mv
    torch.testing.assert_close(routed[3].index_select(2, index).float(), reference_coarse, atol=.02, rtol=.03)
    width = triton.next_power_of_2(slots.size(-1))
    padded = torch.full((*slots.shape[:-1], width), -1, device=q.device, dtype=slots.dtype)
    padded[..., :slots.size(-1)] = slots
    eo, el = torch.empty_like(query), torch.empty_like(number, dtype=torch.float32)
    roots, directory, pages = page['slot_pages'], page['overflow_page_values'], page['page_indices']
    lk, lv = page['leaf_k'], page['leaf_v']
    _direct_leaf_attention[(query.size(1) * query.size(2),)](
        query, padded, number, page['slot_lengths'], roots, page['overflow_page_keys'], directory,
        page['overflow_used'], pages, lk, lv, eo, el, query.size(2), width, state_len,
        counts.size(2), engine.num_key_value_groups, roots.size(3), directory.size(2), pages.size(2),
        lk.size(2), engine._page_lookup_probes(page), query.size(-1), float(engine.scaling), 64,
        width.bit_length(), num_warps=4)
    actual_exact = exact.index_select(2, index).float()
    torch.testing.assert_close(actual_exact, eo.float(), atol=.025, rtol=.03)
    torch.testing.assert_close(exact_lse.index_select(2, index), el, atol=.003, rtol=.001)
    entries = (actual * size.long()).sum(-1)
    if rule == 'entry' and not bool((entries.square() < work_bound * work_bound * size.sum(-1)).all()):
        raise AssertionError('entry-work bound violated')
    record = dict(query_heads=number.numel(), state_len=state_len, selection_mismatches=mismatches,
                  mismatches_away_from_boundary=away, exact_max_abs=float((actual_exact - eo.float()).abs().max()),
                  coarse_max_abs=float((routed[3].index_select(2, index).float() - reference_coarse).abs().max()),
                  opened_entries_mean=float(entries.float().mean()), opened_entries_max=int(entries.max()))
    engine._lod_closed_prefill_audit = getattr(engine, '_lod_closed_prefill_audit', []) + [record]


def threshold_prefill_closed(engine, q, local_k, local_v, state_k, state_v, counts,
                             owners, exact_k, exact_v, *, state_len, state_capacity,
                             page_cache, local_branch, sink_k, sink_v, output_buffer=None,
                             rule, multiplier, entry_bound, work_bound, tile_queries, **kwargs):
    if rule not in ('region', 'entry') or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('closed prefill requires region/entry routing and multiplier >= 1')
    if any(not isinstance(x, int) or x < 1 for x in (entry_bound, work_bound, tile_queries)):
        raise ValueError('entry/work bounds and internal query tile must be positive integers')
    if q.shape[:2] != (1, 32) or q.size(-1) != 128 or q.dtype != torch.bfloat16 or torch.version.hip:
        raise ValueError('closed H100 prefill requires batch-one BF16 Llama GQA')
    if local_branch is None or sink_k is None or sink_v is None or page_cache is None or page_cache.get('quantization_finalized'):
        raise ValueError('requires exact local/sink branches and BF16 indexed leaves')
    buffers = getattr(engine, '_lod_prefill_attention_buffers', None)
    if buffers is None:
        engine._lod_prefill_attention_buffers = buffers = {}
    result = torch.empty_like(q) if output_buffer is None else output_buffer
    # This tile bounds scratch only. Local attention and the cache-update/initial
    # exact-prefill schedule remain those of the surrounding native operator.
    block = int(os.environ.get('LOD_H100_CLOSED_TILE', '512' if rule == 'entry' or os.environ.get('LOD_H100_CLOSED_EXACT') == 'indexed' else '4096'))
    if block < 1:
        raise ValueError('closed prefill internal tile must be positive')
    events = None
    if os.environ.get('LOD_H100_PREFILL_PROFILE') == '1':
        if not hasattr(engine, '_lod_leaf_timing_events'):
            engine._lod_leaf_timing_events = {}
        events = engine._lod_leaf_timing_events
    kv, d = state_k.size(1), q.size(-1)
    def ws(name, shape, dtype=q.dtype):
        return _workspace_tensor(buffers, 'closed_' + name, shape, dtype=dtype, device=q.device)
    mk, mv, logc = ws('mean_k', (1, kv, state_len, d)), ws('mean_v', (1, kv, state_len, d)), ws('log_count', (kv, state_len), torch.float32)
    _prepare[(kv * state_len,)](state_k, state_v, counts, mk, mv, logc, state_len, state_k.size(2), d, num_warps=4)
    for begin in range(0, q.size(2), block):
        end = min(begin + block, q.size(2))
        query = q[..., begin:end, :].contiguous()
        routed = route_coarse(query, state_k, state_v, counts, state_len=state_len,
            scale=engine.scaling, group=engine.num_key_value_groups, rule=rule,
            multiplier=multiplier, entry_bound=entry_bound, work_bound=work_bound,
            leaf_capacity=page_cache['leaf_k'].size(2), buffers=buffers,
            prepared=(mk, mv, logc), timing_events=events)
        entry_capacity = (1 + math.isqrt(work_bound * work_bound * page_cache['leaf_k'].size(2))
                          if rule == 'entry' else routed[0].size(-1) * 1024)
        exact, exact_lse = exact_attention(engine, query, page_cache, state_len, routed, buffers,
                                          entry_capacity=entry_capacity)
        if begin == 0 and os.environ.get('LOD_H100_CLOSED_VERIFY') == '1':
            verify_frontier(engine, query, counts, page_cache, state_len, (mk, mv, logc), routed,
                            exact, exact_lse, rule, multiplier, entry_bound, work_bound)
        merge_attention_branches_with_sink(query, sink_k, sink_v, routed[3], routed[4], exact, exact_lse,
            local_branch[0][..., begin:end, :], local_branch[1][..., begin:end],
            kv_group_size=engine.num_key_value_groups, scale=engine.scaling,
            output_buffer=result[..., begin:end, :])
    return result
