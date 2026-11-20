"""H100 threshold prefill using the top-eight producer's score-reuse arithmetic.

Stream native QK once, produce full coarse attention and routing LSE, select
from saved scores, and correct the coarse field in the final local/sink merge.
This is opt-in and preserves the existing prefill/cache update schedule.

The direct scaled-FP16 variant instead combines final threshold decisions and
closed weights in one saved-score pass, then packs exact bits without another
score scan. It projects only closed summaries, with no opened-summary subtraction.
"""
from __future__ import annotations
import math
import os
import torch
import triton
import triton.language as tl
from lod_attention.kernels.aiter_prefill_attention import _workspace_tensor, _prepare_aiter_state_kernel
from benchmarks.h100_threshold_prefill_fused import _expand_direct_indices, _indexed_direct_attention, _region_totals


@triton.jit
def _produce(Q, K, V, Lengths, Scores, Coarse, CoarseLSE, RouteLSE,
             QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr,
             D: tl.constexpr, SCALE: tl.constexpr, M: tl.constexpr, N: tl.constexpr, InvRMS=None, COARSE:tl.constexpr=True,
             CandidateBits=None, NRegions=None, MULTIPLIER:tl.constexpr=16., LogSizes=None):
    head = tl.program_id(0)
    kh = head // G
    row = tl.program_id(1) * M + tl.arange(0, M)
    dim = tl.arange(0, D)
    col = tl.arange(0, N)
    q = tl.load(Q + (head * QLEN + row[:, None]) * D + dim[None, :], mask=row[:, None] < QLEN, other=0)
    qf = q.to(tl.float32)
    inv = tl.rsqrt(tl.maximum(tl.sum(qf * qf, 1) / D, 1.e-12))
    if InvRMS is not None:
        tl.store(InvRMS + head * QLEN + row, inv, mask=row < QLEN)
    mx = tl.full((M,), -float('inf'), tl.float32)
    den = tl.zeros((M,), tl.float32)
    num = tl.zeros((M, D), tl.float32)
    rmx = tl.full((M,), -float('inf'), tl.float32)
    rden = tl.zeros((M,), tl.float32)
    if CandidateBits is not None:
        cutoff_ratio=MULTIPLIER/tl.maximum(tl.load(NRegions+kh).to(tl.float32),1.)
    for start in range(tl.cdiv(S, N)):
        slot = start * N + col
        sizes = tl.load(Lengths + kh * CAP + slot, mask=slot < S, other=0).to(tl.float32)
        valid = (slot < S) & (sizes > 0)
        k = tl.load(K + (kh * S + slot[None, :]) * D + dim[:, None], mask=valid[None, :], other=0)
        dot = tl.dot(q, k).to(tl.float32) * SCALE
        if Scores is not None:
            tl.store(Scores + (head * QLEN + row[:, None]) * S + slot[None, :], dot,
                     mask=(row[:, None] < QLEN) & (slot[None, :] < S))
        logsize = tl.log(tl.maximum(sizes, 1.0)) if LogSizes is None else tl.load(LogSizes+kh*S+slot,mask=slot<S,other=0.)
        if COARSE:
            v = tl.load(V + (kh * S + slot[:, None]) * D + dim[None, :], mask=valid[:, None], other=0)
            score = tl.where(valid[None, :], dot + logsize[None, :], -float('inf'))
            nxt = tl.maximum(mx, tl.max(score, 1))
            safe = tl.where(nxt != -float('inf'), nxt, 0.0)
            adjust = tl.where(mx != -float('inf'), tl.exp(mx - safe), 0.0)
            weight = tl.where(valid[None, :], tl.exp(score - safe[:, None]), 0.0)
            num = num * adjust[:, None] + tl.dot(weight.to(v.dtype), v)
            den = den * adjust + tl.sum(weight, 1)
            mx = nxt
        route = tl.where(valid[None, :], dot * inv[:, None] + logsize[None, :], -float('inf'))
        rnxt = tl.maximum(rmx, tl.max(route, 1))
        rsafe = tl.where(rnxt != -float('inf'), rnxt, 0.0)
        route_weight=tl.where(valid[None,:],tl.exp(route-rsafe[:,None]),0.)
        rden = rden * tl.where(rmx != -float('inf'), tl.exp(rmx - rsafe), 0.0) + tl.sum(route_weight, 1)
        if CandidateBits is not None:
            # Prefix mass is a lower bound on final mass. The guard preserves
            # candidates at FP32 rounding boundaries; the final test is strict.
            candidate=valid[None,:]&(row[:,None]<QLEN)&(route_weight>=cutoff_ratio*rden[:,None]*(1.-1.e-5))
            bit=tl.arange(0,8)
            packed=tl.sum(tl.reshape(candidate.to(tl.int32),(M,N//8,8))*(1<<bit[None,None,:]),2).to(tl.uint8)
            byte=start*(N//8)+tl.arange(0,N//8)
            tl.store(CandidateBits+(head*QLEN+row[:,None])*tl.cdiv(S,8)+byte[None,:],packed,
                mask=(row[:,None]<QLEN)&(byte[None,:]<tl.cdiv(S,8)))
        rmx = rnxt
    if COARSE:
        tl.store(Coarse + (head * QLEN + row[:, None]) * D + dim[None, :], num / tl.maximum(den[:, None], 1.e-30), mask=row[:, None] < QLEN)
        tl.store(CoarseLSE + head * QLEN + row, tl.where(den > 0, mx + tl.log(den), -float('inf')), mask=row < QLEN)
    tl.store(RouteLSE + head * QLEN + row, tl.where(rden > 0, rmx + tl.log(rden), float('inf')), mask=row < QLEN)


@triton.jit
def _select(Q, Scores, RouteLSE, Lengths, Remote, NRegions, Slots, Counts, Entries, HeadCounts, Offsets, Bits,
            QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            WIDTH: tl.constexpr, ENTRY: tl.constexpr, BOUND: tl.constexpr, WORK: tl.constexpr,
            MULTIPLIER: tl.constexpr, MAX_LEAVES: tl.constexpr, MAX_REGIONS: tl.constexpr,
            FUSED_COUNTS: tl.constexpr, BLOCK: tl.constexpr, EARLY_EXIT: tl.constexpr, SAVE_MASK: tl.constexpr):
    row = tl.program_id(0)
    head = row // QLEN
    kh = head // G
    d = tl.arange(0, D)
    q = tl.load(Q + row * D + d).to(tl.float32)
    inv = tl.rsqrt(tl.maximum(tl.sum(q * q, 0) / D, 1.e-12))
    logz = tl.load(RouteLSE + row)
    remote = tl.load(Remote + kh).to(tl.int64)
    if ENTRY:
        cutoff = logz - tl.log(BOUND * 1.0) - 0.5 * tl.log(tl.maximum(remote.to(tl.float32), 1.0)) + 1.e-6
    else:
        nregions = tl.load(NRegions + kh)
        cutoff = logz + tl.log(MULTIPLIER / tl.maximum(nregions.to(tl.float32), 1.0))
    idx = tl.arange(0, BLOCK)
    before = tl.full((), 0, tl.int64)
    selected = tl.full((), 0, tl.int32)
    entries = tl.full((), 0, tl.int32)
    start = tl.full((), 0, tl.int32)
    selection_limit: tl.constexpr = MAX_REGIONS if EARLY_EXIT and MAX_REGIONS else 2147483647
    while (start < tl.cdiv(S, BLOCK)) & (selected < selection_limit):
        slot = start * BLOCK + idx
        size = tl.load(Lengths + kh * CAP + slot, mask=slot < S, other=0).to(tl.int64)
        dot = tl.load(Scores + row * S + slot, mask=slot < S, other=0)
        route = dot * inv + tl.log(tl.maximum(size.to(tl.float32), 1.0))
        if ENTRY:
            candidate = (slot < S) & (size > 0) & (dot * inv > cutoff)
            add = tl.where(candidate, size, 0)
            prefix = before + tl.cumsum(add, 0)
            chosen = candidate & (prefix * prefix < WORK * WORK * remote)
            before += tl.sum(add, 0)
        else:
            chosen = (slot < S) & (size > 0) & (size <= MAX_LEAVES) & (route > cutoff)
        rank = selected + tl.cumsum(chosen.to(tl.int32), 0) - 1
        if MAX_REGIONS:
            chosen &= rank < MAX_REGIONS
        tl.store(Slots + row * WIDTH + rank, slot, mask=chosen)
        if FUSED_COUNTS:
            offset = tl.atomic_add(HeadCounts + head * S + slot, 1, mask=chosen, sem='relaxed')
            tl.store(Offsets + row * WIDTH + rank, offset, mask=chosen)
        selected += tl.sum(chosen.to(tl.int32), 0)
        entries += tl.sum(tl.where(chosen, size, 0), 0).to(tl.int32)
        # Mark opened summaries in-place. A numerical fallback can recompute
        # the closed coarse field without a separate selection mask.
        if SAVE_MASK:
            bit = tl.arange(0, 8)
            packed_bits = tl.sum(tl.reshape(chosen.to(tl.int32), (BLOCK // 8, 8)) * (1 << bit[None, :]), 1).to(tl.uint8)
            byte = start * (BLOCK // 8) + tl.arange(0, BLOCK // 8)
            tl.store(Bits + row * tl.cdiv(S, 8) + byte, packed_bits, mask=byte < tl.cdiv(S, 8))
        else:
            tl.store(Scores + row * S + slot, -float('inf'), mask=chosen)
        start += 1
    tl.store(Counts + row, selected)
    tl.store(Entries + row, entries)


@triton.jit
def _select_region_full(Q, Scores, RouteLSE, Lengths, NRegions, Slots, Counts, Entries, HeadCounts, Offsets, Bits,
                       T: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                       WIDTH: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr,
                       M: tl.constexpr, AGGREGATE: tl.constexpr):
    if AGGREGATE:
        query = tl.program_id(1) * M + tl.arange(0, M)
        head = tl.program_id(0)
        row = head.to(tl.int64) * T + query
        row_valid = query < T
    else:
        row = tl.program_id(0).to(tl.int64) + tl.arange(0, M)
        head = row // T
        row_valid = tl.full((M,), True, tl.int1)
    kh = head // G
    d = tl.arange(0, D)
    q = tl.load(Q + row[:, None] * D + d[None, :], mask=row_valid[:, None], other=0).to(tl.float32)
    inv = tl.rsqrt(tl.maximum(tl.sum(q * q, 1) / D, 1.e-12))
    slot = tl.arange(0, BLOCK)[None, :]
    size = tl.load(Lengths + kh * CAP + slot, mask=slot < S, other=0).to(tl.int32)
    dot = tl.load(Scores + row[:, None] * S + slot, mask=row_valid[:, None] & (slot < S), other=0)
    logz = tl.load(RouteLSE + row, mask=row_valid, other=0)
    cutoff = logz + tl.log(C / tl.maximum(tl.load(NRegions + kh).to(tl.float32), 1.))
    opened = row_valid[:, None] & (slot < S) & (size > 0) & (size <= 1024) & (dot * inv[:, None] + tl.log(tl.maximum(size.to(tl.float32), 1.)) > cutoff[:, None])
    rank = tl.cumsum(opened.to(tl.int32), 1) - 1
    tl.store(Slots + row[:, None] * WIDTH + tl.maximum(rank, 0), slot, mask=opened)
    if AGGREGATE:
        total = tl.sum(opened.to(tl.int32), 0)[None, :]
        base = tl.atomic_add(HeadCounts + head * S + slot, total, mask=(slot < S) & (total > 0), sem='relaxed')
        offset = base + tl.cumsum(opened.to(tl.int32), 0) - 1
    else:
        offset = tl.atomic_add(HeadCounts + head[:, None] * S + slot, 1, mask=opened, sem='relaxed')
    tl.store(Offsets + row[:, None] * WIDTH + tl.maximum(rank, 0), offset, mask=opened)
    tl.store(Counts + row, tl.sum(opened.to(tl.int32), 1), mask=row_valid)
    tl.store(Entries + row, tl.sum(tl.where(opened, size, 0), 1), mask=row_valid)
    bit = tl.arange(0, 8)
    packed = tl.sum(tl.reshape(opened.to(tl.int32), (M, BLOCK // 8, 8)) * (1 << bit[None, None, :]), 2).to(tl.uint8)
    byte = tl.arange(0, BLOCK // 8)[None, :]
    tl.store(Bits + row[:, None] * tl.cdiv(S, 8) + byte, packed, mask=row_valid[:, None] & (byte < tl.cdiv(S, 8)))


@triton.jit
def _merge(Q, Scores, K, V, Lengths, Slots, Counts, Full, FullLSE, Exact, ExactLSE,
           Local, LocalLSE, SinkK, SinkV, Out, Flags, Bits,
           QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
           WIDTH: tl.constexpr, SCALE: tl.constexpr, SINK: tl.constexpr, ROUTEWISE: tl.constexpr,
           LK0: tl.constexpr, LK1: tl.constexpr, LL0: tl.constexpr, LL1: tl.constexpr,
           O0: tl.constexpr, O1: tl.constexpr, BLOCK: tl.constexpr, STABILITY: tl.constexpr, USE_SAVED: tl.constexpr,
           ROW_OVERRIDE=None, LogSizes=None):
    # Variable route outputs can exceed 2^31 elements even at batch one.
    row = tl.program_id(0).to(tl.int64) if ROW_OVERRIDE is None else ROW_OVERRIDE.to(tl.int64)
    head = row // QLEN
    query = row % QLEN
    kh = head // G
    dim = tl.arange(0, D)
    t = tl.arange(0, BLOCK)
    q = tl.load(Q + row * D + dim).to(tl.float32)
    full_lse = tl.load(FullLSE + row)
    full_value = tl.load(Full + row * D + dim).to(tl.float32)
    count = tl.load(Counts + row)
    removed = tl.full((), 0.0, tl.float32)
    removed_value = tl.zeros((D,), tl.float32)
    for start in range(tl.cdiv(count, BLOCK)):
        rank = start * BLOCK + t
        slot = tl.load(Slots + row * WIDTH + rank, mask=rank < count, other=0).to(tl.int32)
        size = tl.load(Lengths + kh * CAP + slot, mask=rank < count, other=1).to(tl.float32)
        if USE_SAVED:
            raw_score = tl.load(Scores + row * S + slot, mask=rank < count, other=0)
        else:
            k = tl.load(K + (kh * S + slot[:, None]) * D + dim[None, :], mask=rank[:, None] < count, other=0).to(tl.float32)
            raw_score = tl.sum(k * q[None, :], 1) * SCALE
        v = tl.load(V + (kh * S + slot[:, None]) * D + dim[None, :], mask=rank[:, None] < count, other=0).to(tl.float32)
        logsize=tl.log(size) if LogSizes is None else tl.load(LogSizes+kh*S+slot,mask=rank<count,other=0.)
        score = raw_score + logsize
        w = tl.where(rank < count, tl.exp(score - full_lse), 0.0)
        removed += tl.sum(w, 0)
        removed_value += tl.sum(w[:, None] * v, 0)
    remaining = tl.maximum(1.0 - removed, 0.0)
    residual = full_value - removed_value
    fallback = (remaining < STABILITY) & (count > 0)
    tl.store(Flags + row, fallback.to(tl.uint8))
    if fallback:
        residual = tl.zeros((D,), tl.float32)
        remaining = tl.full((), 0.0, tl.float32)
        for start in range(tl.cdiv(S, BLOCK)):
            slot = start * BLOCK + t
            size = tl.load(Lengths + kh * CAP + slot, mask=slot < S, other=0).to(tl.float32)
            raw = tl.load(Scores + row * S + slot, mask=slot < S, other=-float('inf'))
            if USE_SAVED:
                byte = tl.load(Bits + row * tl.cdiv(S, 8) + slot // 8, mask=slot < S, other=0).to(tl.int32)
                raw = tl.where(((byte >> (slot % 8)) & 1) != 0, -float('inf'), raw)
            logsize=tl.log(tl.maximum(size,1.)) if LogSizes is None else tl.load(LogSizes+kh*S+slot,mask=slot<S,other=0.)
            w = tl.where((slot < S) & (size > 0), tl.exp(raw + logsize - full_lse), 0.0)
            v = tl.load(V + (kh * S + slot[:, None]) * D + dim[None, :], mask=slot[:, None] < S, other=0).to(tl.float32)
            remaining += tl.sum(w, 0)
            residual += tl.sum(w[:, None] * v, 0)
    local_lse = tl.load(LocalLSE + head * LL0 + query * LL1)
    local = tl.load(Local + head * LK0 + query * LK1 + dim).to(tl.float32)
    if ROUTEWISE:
        erank = tl.arange(0, WIDTH)
        valid = erank < count
        es = tl.load(ExactLSE + row * WIDTH + erank, mask=valid, other=-float('inf'))
        ev = tl.load(Exact + (row * WIDTH + erank[:, None]) * D + dim[None, :], mask=valid[:, None], other=0).to(tl.float32)
        exact_lse = tl.max(es, 0)
        eweight = tl.where(valid, tl.exp(es - tl.where(exact_lse != -float('inf'), exact_lse, 0.0)), 0.0)
        eden = tl.sum(eweight, 0)
        exact = tl.sum(eweight[:, None] * ev, 0) / tl.maximum(eden, 1.e-30)
        exact_lse = tl.where(eden > 0, exact_lse + tl.log(eden), -float('inf'))
    else:
        exact_lse = tl.load(ExactLSE + row)
        exact = tl.load(Exact + row * D + dim).to(tl.float32)
    maximum = tl.maximum(tl.maximum(full_lse, exact_lse), local_lse)
    for i in tl.static_range(SINK):
        sk = tl.load(SinkK + (kh * SINK + i) * D + dim).to(tl.float32)
        maximum = tl.maximum(maximum, tl.sum(q * sk, 0) * SCALE)
    cw, ew, lw = tl.exp(full_lse - maximum), tl.exp(exact_lse - maximum), tl.exp(local_lse - maximum)
    numerator = cw * residual + ew * exact + lw * local
    denominator = cw * remaining + ew + lw
    for i in tl.static_range(SINK):
        sk = tl.load(SinkK + (kh * SINK + i) * D + dim).to(tl.float32)
        sv = tl.load(SinkV + (kh * SINK + i) * D + dim).to(tl.float32)
        w = tl.exp(tl.sum(q * sk, 0) * SCALE - maximum)
        numerator += w * sv
        denominator += w
    tl.store(Out + head * O0 + query * O1 + dim, numerator / tl.maximum(denominator, 1.e-30))


def threshold_prefill_fast(engine, q, local_k, local_v, state_k, state_v, counts,
                            owners, exact_k, exact_v, *, state_len, state_capacity,
                            page_cache, local_branch, sink_k, sink_v, output_buffer=None,
                            rule, multiplier, entry_bound, work_bound, tile_queries, **kwargs):
    del local_k, local_v, owners, exact_k, exact_v, tile_queries, kwargs
    if q.size(0) != 1 or q.size(1) != 32 or q.size(-1) != 128 or local_branch is None or page_cache is None or page_cache.get('quantization_finalized', False):
        raise ValueError('fast experiment requires batch-one Llama BF16 prefill')
    if rule not in ('entry', 'region'):
        raise ValueError('fast experiment implements entry and per-region thresholds')
    buffers = getattr(engine, '_lod_prefill_attention_buffers', None)
    if buffers is None:
        buffers = {}; engine._lod_prefill_attention_buffers = buffers
    def workspace(name, shape, dtype):
        return _workspace_tensor(buffers, 'fast_' + name, shape, dtype=dtype, device=q.device)
    profiling = os.environ.get('LOD_H100_PREFILL_PROFILE') == '1'
    if profiling:
        engine._lod_prefill_threshold_profile = {}
        if not hasattr(engine, '_lod_leaf_timing_events'):
            engine._lod_leaf_timing_events = {}
        phase_begin = torch.cuda.Event(enable_timing=True)
        phase_begin.record()
    def mark(name):
        nonlocal phase_begin
        if profiling:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            engine._lod_leaf_timing_events.setdefault(name, []).append((phase_begin, end))
            phase_begin = end
    h, kv, g, d = int(q.size(1)), int(state_k.size(1)), engine.num_key_value_groups, int(q.size(-1))
    lengths = page_cache['slot_lengths']
    mean_k = workspace('mean_k', (1, kv, state_len, d), q.dtype)
    mean_v = workspace('mean_v', (1, kv, state_len, d), q.dtype)
    coarse_counts = workspace('counts', (1, kv, state_len, 1), counts.dtype)
    bias = workspace('bias', (1, h, 1, state_len), q.dtype)
    _prepare_aiter_state_kernel[(kv * state_len,)](state_k, state_v, counts, mean_k, mean_v, coarse_counts, bias,
        state_len, state_len, STATE_CAPACITY=state_capacity, KV_HEADS=kv, KV_GROUP_SIZE=g, BLOCK_G=triton.next_power_of_2(g), HEAD_DIM=d, BLOCK_D=d, num_warps=4)
    remote = workspace('remote', (kv,), torch.int64)
    nregions = workspace('nregions', (kv,), torch.int32)
    _region_totals[(kv,)](lengths, remote, nregions, state_len, state_capacity, triton.next_power_of_2(state_len), num_warps=4)
    mark('fast_prepare')
    cap = int(os.environ.get('LOD_H100_PREFILL_MAX_REGIONS', '0'))
    if cap not in (0, 8, 16):
        raise ValueError('fast experiment supports uncapped, eight-, or sixteen-region cap')
    from benchmarks.h100_prefill_direct_config import scratch_queries
    block_q=scratch_queries(state_len,512 if rule=='entry'and os.environ.get('LOD_H100_FAST_LEAF')=='native'else 4096)
    if block_q <= 0:
        raise ValueError('fast tile must be positive')
    max_entries = max(1, min(int(page_cache['leaf_k'].size(2)), math.isqrt(work_bound * work_bound * int(page_cache['leaf_k'].size(2))) + 1))
    entry_block = triton.next_power_of_2(max_entries)
    width = cap or triton.next_power_of_2(min(state_len, max_entries) if rule == 'entry'
                                        else min(state_len, math.ceil(state_len / multiplier) + 1))
    native = os.environ.get('LOD_H100_FAST_LEAF') == 'native'
    fused_merge = os.environ.get('LOD_H100_FAST_FUSED_MERGE') == '1'
    prefix_closed=os.environ.get('LOD_H100_FAST_PREFIX_CLOSED')=='1'
    from benchmarks.h100_prefill_direct_config import fused_selection_enabled
    direct_select=fused_selection_enabled()
    direct_merge=os.environ.get('LOD_H100_DIRECT_FUSE_MERGE')=='1'
    if direct_select and not prefix_closed:
        raise ValueError('combined selection/normalization requires direct closed projection')
    if direct_merge and (not prefix_closed or cap or not native):
        raise ValueError('direct route merge requires uncapped native direct-summary prefill')
    recompute=os.environ.get('LOD_H100_FAST_RECOMPUTE_CLOSED')=='1'
    if recompute and (fused_merge or prefix_closed):
        raise ValueError('recomputed closed summaries require reduced exact outputs and no prefix projection')
    if prefix_closed and fused_merge:
        raise ValueError('direct closed-summary projection requires reduced exact outputs')
    if fused_merge and (cap or not native or rule!='region' or os.environ.get('LOD_H100_FAST_SAVED_CORRECTION')!='1'
            or os.environ.get('LOD_H100_FAST_CLOSED_PASS')=='1' or os.environ.get('LOD_H100_CLOSED_EXACT','grouped')!='grouped'):
        raise ValueError('fused route merge requires uncapped per-region native leaves and saved correction')
    output = torch.empty_like(q) if output_buffer is None else output_buffer
    local_out, local_lse = local_branch
    verify = os.environ.get('LOD_H100_FAST_VERIFY') == '1'
    log_sizes=None
    if os.environ.get('LOD_H100_FAST_SHARED_LOG')=='1':
        from benchmarks.h100_threshold_prefill_direct import _prepare_log
        log_sizes=workspace('shared_log_sizes',(h//g,state_len),torch.float32)
        _prepare_log[(h//g,triton.cdiv(state_len,256))](lengths,log_sizes,state_len,state_capacity,256,num_warps=4)
        mark('fast_shared_log')
    prepared_projection=None
    if os.environ.get('LOD_H100_DIRECT_HOIST')=='1':
        from benchmarks.h100_threshold_prefill_direct import prepare_projection
        prepared_projection=prepare_projection(mean_v,lengths,s=state_len,cap=state_capacity,buffers=buffers)
        mark('direct_shared_prepare')
    for begin in range(0, int(q.size(2)), block_q):
        end = min(begin + block_q, int(q.size(2))); n = end - begin
        query = q[..., begin:end, :].contiguous()
        scores = None if recompute and not verify else workspace('scores', (h, n, state_len), torch.float32)
        full = workspace('full', (1, h, n, d), torch.float32)
        full_lse = workspace('full_lse', (1, h, n), torch.float32)
        route_lse = workspace('route_lse', (h, n), torch.float32)
        tiled = os.environ.get('LOD_H100_FAST_TILED_SELECT') == '1'
        biased_select = os.environ.get('LOD_H100_FAST_BIASED_SELECT') == '1'
        closed_pass = os.environ.get('LOD_H100_FAST_CLOSED_PASS') == '1'
        scan_mode=os.environ.get('LOD_H100_FAST_SCAN_SELECT','auto')
        if scan_mode not in ('0','1','auto'):raise ValueError('scan selection must be 0, 1, or auto')
        scan_eligible=(not cap and native and rule=='region' and not tiled and not biased_select and not closed_pass
            and os.environ.get('LOD_H100_FAST_FULL_SELECT')=='1' and os.environ.get('LOD_H100_FAST_SAVED_CORRECTION')=='1')
        # Keep the cheaper full-row path at small region counts. Only the
        # packing algorithm changes; every qualifying region is preserved.
        scan_select=(scan_mode=='1' or (scan_mode=='auto' and scan_eligible)) and state_len>=int(
            os.environ.get('LOD_H100_SCAN_MIN_REGIONS','4096' if scan_mode=='auto' else '0'))
        prefix_mode=os.environ.get('LOD_H100_FAST_PREFIX_SELECT','auto')
        if prefix_mode not in ('0','1','auto'):
            raise ValueError('prefix selection must be 0, 1, or auto')
        prefix_select=(prefix_mode=='1' or (prefix_mode=='auto' and scan_eligible and 2048<=state_len<4096))
        if recompute and (prefix_select or not scan_eligible):
            raise ValueError('recomputed closed summaries require uncapped per-region native leaves')
        if prefix_closed and not prefix_select:
            raise ValueError('direct closed-summary projection requires prefix selection')
        if prefix_select and not scan_eligible:
            raise ValueError('prefix selection requires uncapped per-region native leaves and saved correction')
        candidate_bits=workspace('selected_bits',(h,n,triton.cdiv(state_len,8)),torch.uint8) if prefix_select and not direct_select else None
        inverse_rms = workspace('inverse_rms', (h, n), torch.float32) if tiled or closed_pass or scan_select or biased_select or prefix_select or recompute else None
        pm=int(os.environ.get('LOD_H100_FAST_PRODUCE_M','64'))
        pn=int(os.environ.get('LOD_H100_FAST_PRODUCE_N','64'))
        pw=int(os.environ.get('LOD_H100_FAST_PRODUCE_WARPS','4'))
        from benchmarks.h100_prefill_direct_config import producer_stages
        ps=producer_stages(state_len)
        if pm not in (32,64,128) or pn not in (32,64,128,256) or pw not in (4,8):
            raise ValueError('unsupported threshold producer tile')
        _produce[(h, triton.cdiv(n, pm))](query, mean_k, mean_v, lengths, scores, full, full_lse, route_lse,
            n, state_len, state_capacity, g, d, float(engine.scaling), pm, pn, InvRMS=inverse_rms, COARSE=not closed_pass and not prefix_closed and not recompute,
            CandidateBits=candidate_bits,NRegions=nregions if prefix_select else None,MULTIPLIER=float(multiplier),LogSizes=log_sizes,num_warps=pw, num_stages=ps)
        mark('fast_produce')
        debug = scores[:, :min(n, int(os.environ.get('LOD_H100_FAST_VERIFY_QUERIES','16'))), :].clone() if verify and begin == 0 else None
        slots = workspace('slots', (1, h, n, width), torch.int64 if cap or native else torch.int32)
        if cap:
            slots.fill_(-1)
        route_counts = workspace('route_counts', (h, n), torch.int32)
        saved_correction = not cap and os.environ.get('LOD_H100_FAST_SAVED_CORRECTION', '0') == '1'
        bits = workspace('selected_bits', (h, n, triton.cdiv(state_len, 8)) if saved_correction else (1,), torch.uint8)
        entries = workspace('entry_counts', (h, n), torch.int32)
        head_counts = workspace('head_counts', (h * state_len,), torch.int32)
        offsets = workspace('route_offsets', (1, h, n, width) if cap or native else (1,), torch.int32)
        if (cap or native) and (recompute or prefix_select or not (scan_select or biased_select)):
            head_counts.zero_()
        select_m = int(os.environ.get('LOD_H100_FAST_SELECT_M', '1'))
        if direct_select:
            # The closed-weight pass makes the final strict routing decision
            # from the same saved FP32 scores; no prefix-candidate pass remains.
            pass
        elif recompute:
            from benchmarks.h100_threshold_prefill_recompute import select_closed
            sm=int(os.environ.get('LOD_H100_RECOMPUTE_M','64'));sn=int(os.environ.get('LOD_H100_RECOMPUTE_N','64'))
            select_closed[(h,triton.cdiv(n,sm))](query,mean_k,mean_v,lengths,nregions,inverse_rms,route_lse,
                slots,route_counts,entries,head_counts,offsets,full,full_lse,
                n,state_len,state_capacity,g,d,width,float(engine.scaling),float(multiplier),sm,sn,
                os.environ.get('LOD_H100_RECOMPUTE_AGGREGATE','1')=='1',num_warps=4,num_stages=2)
        elif prefix_select:
            from benchmarks.h100_threshold_prefill_prefix import select
            select(scores,inverse_rms,route_lse,lengths,nregions,slots,route_counts,entries,head_counts,offsets,bits,
                t=n,s=state_len,cap=state_capacity,g=g,multiplier=multiplier,buffers=buffers)
        elif biased_select:
            if cap or not native or rule!='region' or not saved_correction:
                raise ValueError('biased packing requires uncapped per-region native leaves and saved correction')
            from benchmarks.h100_threshold_prefill_bias import select
            select(scores,inverse_rms,route_lse,lengths,nregions,slots,route_counts,entries,head_counts,offsets,bits,
                t=n,s=state_len,cap=state_capacity,g=g,multiplier=multiplier,buffers=buffers,
                warps=int(os.environ.get('LOD_H100_BIAS_WARPS','8')))
        elif scan_select:
            if cap or not native or rule!='region' or not saved_correction:
                raise ValueError('prefix packing requires uncapped per-region native leaves and saved correction')
            if os.environ.get('LOD_H100_FAST_BITMAP_PACK')=='1':
                from benchmarks.h100_threshold_prefill_bitmap import select
            elif os.environ.get('LOD_H100_FAST_SPARSE_PACK')=='1':
                from benchmarks.h100_threshold_prefill_sparse import select
            else:
                from benchmarks.h100_threshold_prefill_scan import select
            select(scores,inverse_rms,route_lse,lengths,nregions,slots,route_counts,entries,head_counts,offsets,bits,
                t=n,s=state_len,cap=state_capacity,g=g,multiplier=multiplier,buffers=buffers,
                m=int(os.environ.get('LOD_H100_SCAN_M','32')),n=int(os.environ.get('LOD_H100_SCAN_N','256')),
                warps=int(os.environ.get('LOD_H100_SCAN_WARPS','4')),q=query,
                atomic_experts=os.environ.get('LOD_H100_SCAN_ATOMIC_EXPERTS')=='1')
        elif closed_pass:
            if cap or not native or rule != 'region':
                raise ValueError('streaming closed prefill requires uncapped per-region native leaves')
            from benchmarks.h100_threshold_prefill_stream import select_closed
            sm=int(os.environ.get('LOD_H100_STREAM_M','32'));sn=int(os.environ.get('LOD_H100_STREAM_N','64'))
            if sm not in (16,32,64) or sn not in (32,64,128):
                raise ValueError('unsupported streaming closed-summary tile')
            select_closed[(h,triton.cdiv(n,sm))](scores,inverse_rms,route_lse,lengths,nregions,mean_v,
                slots,route_counts,entries,head_counts,offsets,full,full_lse,
                n,state_len,state_capacity,g,d,width,float(multiplier),sm,sn,num_warps=4,num_stages=1)
        elif tiled:
            if cap or not native or rule != 'region' or not saved_correction:
                raise ValueError('tiled packing requires uncapped per-region native leaves and saved-score correction')
            from benchmarks.h100_threshold_prefill_tiled import select
            sm = int(os.environ.get('LOD_H100_FAST_TILED_M', '16'))
            sn = int(os.environ.get('LOD_H100_FAST_TILED_N', '128'))
            if sm not in (4, 8, 16, 32) or sn not in (64, 128, 256):
                raise ValueError('unsupported query/region tile for threshold packing')
            route_counts.zero_(); entries.zero_()
            select[(h, triton.cdiv(n, sm), triton.cdiv(state_len, sn))](
                scores, inverse_rms, route_lse, lengths, nregions, slots, route_counts, entries,
                head_counts, offsets, bits, n, state_len, state_capacity, g, width,
                float(multiplier), sm, sn, num_warps=4)
        elif os.environ.get('LOD_H100_FAST_FULL_SELECT') == '1':
            if cap or not native or rule != 'region' or not saved_correction:
                raise ValueError('full selection requires uncapped per-region native leaves and saved-score correction')
            aggregate = os.environ.get('LOD_H100_PREFILL_AGGREGATE', '0') == '1'
            sm = int(os.environ.get('LOD_H100_CLOSED_SELECT_QM', '1')) if aggregate else 1
            if sm not in (1, 2, 4, 8):
                raise ValueError('aggregated selection query group must be one, two, four or eight')
            grid = (h, triton.cdiv(n, sm)) if aggregate else (h * n,)
            _select_region_full[grid](query, scores, route_lse, lengths, nregions, slots, route_counts,
                entries, head_counts, offsets, bits, n, state_len, state_capacity, g, d, width,
                float(multiplier), triton.next_power_of_2(max(state_len, 8)), sm, aggregate,
                num_warps=int(os.environ.get('LOD_H100_CLOSED_SELECT_WARPS', '8')))
        elif not cap and select_m == 16 and rule == 'entry':
            _select_vector[(h, triton.cdiv(n, 16))](query, scores, route_lse, lengths, remote, slots, route_counts, entries, bits,
                n, state_len, state_capacity, g, d, width, entry_bound, work_bound, 16, 128, saved_correction, num_warps=8)
        elif select_m in (1, 16):
            _select[(h * n,)](query, scores, route_lse, lengths, remote, nregions, slots, route_counts, entries, head_counts, offsets, bits,
                n, state_len, state_capacity, g, d, width, rule == 'entry', entry_bound, work_bound, float(multiplier),
                min(int(engine.max_open_centroid_leaves or 1024), 1024), cap, bool(cap or native), 256,
                os.environ.get('LOD_H100_FAST_CAP_EARLY_EXIT', '1') == '1', saved_correction, num_warps=4)
        else:
            raise ValueError('fast selection supports query blocks of one or sixteen')
        mark('fast_select')
        if prefix_closed:
            from benchmarks.h100_threshold_prefill_direct import project
            project(scores,lengths,bits,mean_v,full,full_lse,t=n,s=state_len,cap=state_capacity,g=g,buffers=buffers,
                timing_events=getattr(engine,'_lod_leaf_timing_events',None) if profiling else None,
                prepared=prepared_projection,routing=(inverse_rms,route_lse,nregions,slots,route_counts,entries,head_counts,offsets,multiplier)
                if direct_select else None)
            mark('fast_closed_project')
        if cap:
            engine._lod_prefill_route_head_counts = head_counts
            engine._lod_prefill_route_offsets = offsets
            exact_out, exact_lse = engine._paged_leaf_attention(query, slots, page_cache, active_slots=state_len, reduce_routes=False)
        elif native:
            from benchmarks.h100_threshold_prefill_closed import exact_attention
            exact_out, exact_lse = exact_attention(engine, query, page_cache, state_len,
                (slots, route_counts.view(1, h, n), entries, None, None, head_counts, offsets), buffers,
                return_routes=fused_merge or direct_merge)
        elif os.environ.get('LOD_H100_FAST_LEAF', 'grouped') == 'grouped':
            from benchmarks.h100_threshold_ragged import attend
            exact_out, exact_lse, overflow = attend(engine, query, slots, route_counts, page_cache,
                                                   state_len=state_len, workspace=workspace)
            if verify:
                engine._lod_fast_last_overflow = overflow
        else:
            ep = workspace('entry_offsets', (h * n + 1,), torch.int64)
            ep[:1].zero_()
            torch.cumsum(entries.reshape(-1), 0, dtype=torch.int64, out=ep[1:])
            indices = workspace('indices', (h, n, entry_block), torch.int32)
            roots, directory, pages = page_cache['slot_pages'], page_cache['overflow_page_values'], page_cache['page_indices']
            leaf_k, leaf_v = page_cache['leaf_k'], page_cache['leaf_v']
            _expand_direct_indices[(h * n,)](slots, route_counts, lengths, roots, page_cache['overflow_page_keys'], directory,
                page_cache['overflow_used'], pages, indices, entries, route_counts, ep, n, width, state_capacity, g,
                int(roots.size(3)), int(directory.size(2)), int(pages.size(2)), int(leaf_k.size(2)), engine._page_lookup_probes(page_cache),
                entry_block, width.bit_length(), True, True, 256, num_warps=4)
            exact_out = workspace('exact', (1, h, n, d), q.dtype)
            exact_lse = workspace('exact_lse', (1, h, n), torch.float32)
            _indexed_direct_attention[(h * n,)](query, indices, entries, ep, leaf_k, leaf_v, exact_out, exact_lse,
                n, entry_block, g, int(leaf_k.size(2)), d, float(engine.scaling), 128, True, num_warps=4)
        mark('fast_exact')
        flags = workspace('fallback', (h, n), torch.uint8)
        lo, ll = local_out[..., begin:end, :], local_lse[..., begin:end]
        dest = output[..., begin:end, :]
        if direct_merge:
            from benchmarks.h100_prefill_direct_fused import reduce_merge
            reduce_merge[(h*n,)](query,sink_k,sink_v,full,full_lse,exact_out,exact_lse,route_counts,lo,ll,dest,
                n,g,d,width,triton.next_power_of_2(width),float(engine.scaling),int(sink_k.size(2)),
                sink_k.stride(1),sink_k.stride(2),sink_v.stride(1),sink_v.stride(2),
                lo.stride(1),lo.stride(2),ll.stride(1),ll.stride(2),dest.stride(1),dest.stride(2),num_warps=4)
            if verify:flags.zero_()
        elif fused_merge:
            from benchmarks.h100_threshold_prefill_merge import merge_routes, merge_bucket, repair_routes
            sm=int(os.environ.get('LOD_H100_FUSED_MERGE_M','4'))
            sr=int(os.environ.get('LOD_H100_FUSED_MERGE_R','16'))
            if sm not in (1,2,4,8) or sr not in (8,16,32):
                raise ValueError('unsupported fused route-merge tile')
            stability=float(os.environ.get('LOD_H100_FAST_STABILITY','0.0001'))
            if os.environ.get('LOD_H100_FAST_FUSED_BUCKET')=='1':
                merge_bucket[(h,n)](query,scores,mean_v,lengths,slots,route_counts,full,full_lse,
                    exact_out,exact_lse,lo,ll,sink_k,sink_v,dest,flags,
                    n,state_len,state_capacity,g,d,width,float(engine.scaling),int(sink_k.size(2)),
                    lo.stride(1),lo.stride(2),ll.stride(1),ll.stride(2),dest.stride(1),dest.stride(2),stability,
                    triton.next_power_of_2(width),num_warps=4)
            else:
                merge_routes[(h,triton.cdiv(n,sm))](query,scores,mean_v,lengths,slots,route_counts,full,full_lse,
                    exact_out,exact_lse,lo,ll,sink_k,sink_v,dest,flags,
                    n,state_len,state_capacity,g,d,width,float(engine.scaling),int(sink_k.size(2)),
                    lo.stride(1),lo.stride(2),ll.stride(1),ll.stride(2),dest.stride(1),dest.stride(2),
                    sm,sr,triton.next_power_of_2(width),stability,num_warps=4)
            repair_routes[(triton.cdiv(h*n,128),)](query,scores,mean_k,mean_v,lengths,slots,route_counts,full,full_lse,
                exact_out,exact_lse,lo,ll,sink_k,sink_v,dest,flags,bits,
                n,state_len,state_capacity,g,d,width,float(engine.scaling),int(sink_k.size(2)),
                lo.stride(1),lo.stride(2),ll.stride(1),ll.stride(2),dest.stride(1),dest.stride(2),stability,h*n,num_warps=4)
        elif closed_pass or prefix_closed or recompute:
            from lod_attention.kernels.lod_kernels import merge_attention_branches_with_sink
            merge_attention_branches_with_sink(query,sink_k,sink_v,full,full_lse.view(1,h,n),exact_out,exact_lse,
                lo,ll,kv_group_size=g,scale=engine.scaling,output_buffer=dest)
            if verify:flags.zero_()
        elif cap and os.environ.get('LOD_H100_FAST_VECTOR_MERGE', '1') == '1':
            _merge_capped[(h, triton.cdiv(n, 32))](query, scores, mean_k, mean_v, lengths, slots, full, full_lse,
                exact_out, exact_lse, lo, ll, sink_k, sink_v, dest, flags,
                n, state_len, state_capacity, g, d, float(engine.scaling), int(sink_k.size(2)),
                lo.stride(1), lo.stride(2), ll.stride(1), ll.stride(2), dest.stride(1), dest.stride(2),
                32, 64, cap, float(os.environ.get('LOD_H100_FAST_STABILITY', '0.0001')), num_warps=8)
        else:
            _merge[(h * n,)](query, scores, mean_k, mean_v, lengths, slots, route_counts, full, full_lse,
                exact_out, exact_lse, lo, ll, sink_k, sink_v, dest, flags, bits,
                n, state_len, state_capacity, g, d, width, float(engine.scaling), int(sink_k.size(2)), bool(cap),
                lo.stride(1), lo.stride(2), ll.stride(1), ll.stride(2), dest.stride(1), dest.stride(2), 32,
                float(os.environ.get('LOD_H100_FAST_STABILITY', '0.0001')), saved_correction, LogSizes=log_sizes,num_warps=4)
        mark('fast_merge')
        if verify and begin == 0:
            _verify(engine, query, debug, mean_k, mean_v, lengths, slots, route_counts, entries, dest, flags,
                    local_out=lo, local_lse=ll, sink_k=sink_k, sink_v=sink_v, page_cache=page_cache,
                    state_len=state_len, rule=rule, multiplier=multiplier, entry_bound=entry_bound, work_bound=work_bound, cap=cap)
            engine._lod_fast_prefill_audit[-1]['direct_closed_summaries']=closed_pass or prefix_closed or recompute
            engine._lod_fast_prefill_audit[-1]['selection_backend']=('direct-fused' if direct_select else 'recompute' if recompute else 'prefix' if prefix_select
                else 'bitmap' if scan_select and os.environ.get('LOD_H100_FAST_BITMAP_PACK')=='1'
                else 'sparse' if scan_select and os.environ.get('LOD_H100_FAST_SPARSE_PACK')=='1'
                else 'scan' if scan_select else 'full-row')
            engine._lod_fast_prefill_audit[-1]['active_regions']=state_len
    return output


def _verify(engine, q, raw, mean_k, mean_v, lengths, slots, route_counts, entries, output, flags,
            *, local_out, local_lse, sink_k, sink_v, page_cache, state_len, rule, multiplier, entry_bound, work_bound, cap):
    from benchmarks.h100_threshold_prefill import select_regions
    from lod_attention.kernels.lod_kernels import merge_attention_branches_with_sink
    n = int(raw.size(1)); h = int(q.size(1)); g = engine.num_key_value_groups
    sample = q[..., :n, :].contiguous()
    sizes = lengths[..., :state_len].repeat_interleave(g, 1).to(torch.int64)
    inv = torch.rsqrt(sample.float().square().mean(-1, keepdim=True).clamp_min(1.e-12))
    route = raw.unsqueeze(0) * inv + sizes.clamp_min(1).float().log().unsqueeze(2)
    expected, remote = select_regions(route, sizes, rule=rule, multiplier=multiplier, entry_bound=entry_bound,
                                     work_bound=work_bound, max_regions=cap or None)
    counts_sample = route_counts[:, :n]
    max_count = max(1, int(counts_sample.max().item()))
    w = triton.next_power_of_2(max_count)
    chosen = slots[..., :n, :w].to(torch.int64).contiguous()
    valid = torch.arange(w, device=q.device)[None, None, None, :] < counts_sample[None, :, :, None]
    actual_i = torch.zeros((1, h, n, state_len), dtype=torch.int32, device=q.device)
    actual_i.scatter_add_(-1, torch.where(valid, chosen, 0), valid.to(torch.int32))
    actual = actual_i.bool()
    difference = expected != actual
    mismatch = int(difference.sum().item())
    cutoff = torch.logsumexp(route.masked_fill((sizes == 0).unsqueeze(2), -torch.inf), -1, keepdim=True)
    if rule == 'region':
        margin = (route-cutoff-math.log(multiplier)+(sizes>0).sum(-1,keepdim=True).clamp_min(1).float().log().unsqueeze(2)).abs()
    else:
        cutoff = cutoff - math.log(entry_bound) - 0.5 * remote.clamp_min(1).float().log() + 1.e-6
        margin = (route - sizes.clamp_min(1).float().log().unsqueeze(2) - cutoff).abs()
    outside = int((difference & (margin > 2.e-5)).sum().item())
    if outside:
        raise RuntimeError(f'fast selection differs on {outside} regions outside boundary tolerance; total={mismatch}')
    opened = (actual.to(torch.int64) * sizes.unsqueeze(2)).sum(-1)
    if rule == 'entry' and bool((opened.square() >= work_bound * work_bound * remote.squeeze(-1)).any().item()):
        raise RuntimeError('fast selection violated strict entry bound')
    for name in ('_lod_prefill_route_head_counts', '_lod_prefill_route_offsets', '_lod_prefill_total_valid_routes'):
        if hasattr(engine, name): delattr(engine, name)
    chosen = torch.where(valid, chosen, -1)
    exact, exact_lse = engine._paged_leaf_attention(sample, chosen, page_cache, active_slots=state_len, reduce_routes=True)
    native = raw.unsqueeze(0) + sizes.clamp_min(1).float().log().unsqueeze(2)
    native = native.masked_fill(actual | (sizes == 0).unsqueeze(2), -torch.inf)
    coarse_lse = torch.logsumexp(native, -1)
    weights = torch.exp(native - torch.where(torch.isfinite(coarse_lse), coarse_lse, 0).unsqueeze(-1))
    coarse = torch.matmul(weights, mean_v.repeat_interleave(g, 1).float()).to(q.dtype)
    ref = merge_attention_branches_with_sink(sample, sink_k, sink_v, coarse, coarse_lse, exact, exact_lse,
        local_out[..., :n, :], local_lse[..., :n], kv_group_size=g, scale=engine.scaling)
    error = (output[..., :n, :].float() - ref.float()).abs()
    maxerror = float(error.max().item())
    # Qwen's larger activations can have adjacent BF16 values separated by
    # more than the Llama check's 0.04 absolute tolerance. Permit one adjacent
    # representable BF16 step, not an arbitrary larger absolute error.
    above=error>0.04
    max_steps=0
    bad=above
    if output.dtype==ref.dtype==torch.bfloat16:
        def ordered(value):
            bits=value.view(torch.int16).to(torch.int32)&65535
            return torch.where((bits&32768)!=0,32768-(bits&32767),32768+(bits&32767))
        steps=(ordered(output[...,:n,:])-ordered(ref)).abs()
        max_steps=int(torch.where(above,steps,0).max().item())
        bad=above&(steps>1)
    largest=int(error.reshape(-1).argmax().item())
    largest_pair=[float(output[...,:n,:].reshape(-1)[largest]),float(ref.reshape(-1)[largest])]
    if not bool(torch.isfinite(error).all().item()) or bool(bad.any().item()):
        raise RuntimeError(f'fast residual correction mismatch: abs={maxerror}, BF16 steps above 0.04={max_steps}, values={largest_pair}')
    audit = getattr(engine, '_lod_fast_prefill_audit', None)
    if audit is None:
        audit = []; engine._lod_fast_prefill_audit = audit
    audit.append({'query_heads':h*n, 'selection_mismatches':mismatch, 'selection_outside_tolerance':outside,
                  'max_changed_threshold_margin':float(margin[difference].max().item()) if mismatch else 0.0,
                  'max_abs_output_difference':maxerror,
                  'max_bf16_steps_above_absolute_tolerance':max_steps,
                  'max_abs_difference_output_reference':largest_pair,
                  'sum_abs_output_difference':float(error.sum().item()), 'output_elements':error.numel(),
                  'opened_regions':int(actual.sum().item()),'opened_entries':int(opened.sum().item()),
                  'fallback_queries':int(flags[:,:n].sum().item()),
                  'route_workspace_fallback_queries': int((engine._lod_fast_last_overflow[:, :n] > 0).sum().item())
                      if not cap and hasattr(engine, '_lod_fast_last_overflow') else 0,
                  'full_tile_query_heads': h * int(q.size(2)),
                  'full_tile_opened_regions': int(route_counts.sum().item()),
                  'full_tile_opened_entries': int(entries.sum().item()),
                  'full_tile_max_regions': int(route_counts.max().item()),
                  'full_tile_max_entries': int(entries.max().item()),
                  'full_tile_queries_at_cap': int((route_counts == cap).sum().item()) if cap else 0,
                  'full_tile_numerical_fallbacks': int(flags.sum().item()),
                  'full_tile_route_workspace_fallbacks': int((engine._lod_fast_last_overflow > 0).sum().item())
                      if not cap and hasattr(engine, '_lod_fast_last_overflow') else 0})


@triton.jit
def _merge_capped(Q, Scores, K, V, Lengths, Slots, Full, FullLSE, Exact, ExactLSE,
                  Local, LocalLSE, SinkK, SinkV, Out, Flags,
                  QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                  SCALE: tl.constexpr, SINK: tl.constexpr, LK0: tl.constexpr, LK1: tl.constexpr,
                  LL0: tl.constexpr, LL1: tl.constexpr, O0: tl.constexpr, O1: tl.constexpr,
                  M: tl.constexpr, N: tl.constexpr, ROUTES: tl.constexpr, STABILITY: tl.constexpr):
    head=tl.program_id(0);kh=head//G
    query=tl.program_id(1)*M+tl.arange(0,M);valid=query<QLEN
    row=head*QLEN+query
    dim=tl.arange(0,D)
    q=tl.load(Q+row[:,None]*D+dim[None,:],mask=valid[:,None],other=0).to(tl.float32)
    fl=tl.load(FullLSE+row,mask=valid,other=0)
    fv=tl.load(Full+row[:,None]*D+dim[None,:],mask=valid[:,None],other=0).to(tl.float32)
    removed=tl.zeros((M,),tl.float32);rv=tl.zeros((M,D),tl.float32)
    for rank in tl.static_range(ROUTES):
        slot=tl.load(Slots+row*ROUTES+rank,mask=valid,other=-1).to(tl.int32)
        active=valid&(slot>=0)
        slot=tl.maximum(slot,0)
        size=tl.load(Lengths+kh*CAP+slot,mask=active,other=1).to(tl.float32)
        k=tl.load(K+(kh*S+slot[:,None])*D+dim[None,:],mask=active[:,None],other=0).to(tl.float32)
        v=tl.load(V+(kh*S+slot[:,None])*D+dim[None,:],mask=active[:,None],other=0).to(tl.float32)
        w=tl.where(active,tl.exp(tl.sum(q*k,1)*SCALE+tl.log(size)-fl),0.)
        removed+=w;rv+=w[:,None]*v
    rem=tl.maximum(1.-removed,0.);res=fv-rv
    fallback=valid&(rem<STABILITY)&(removed>0)
    tl.store(Flags+row,fallback.to(tl.uint8),mask=valid)
    if tl.sum(fallback.to(tl.int32),0)>0:
        rn=tl.zeros((M,D),tl.float32);rd=tl.zeros((M,),tl.float32)
        for start in range(tl.cdiv(S,N)):
            fslot=start*N+tl.arange(0,N)
            fsize=tl.load(Lengths+kh*CAP+fslot,mask=fslot<S,other=0).to(tl.float32)
            fraw=tl.load(Scores+row[:,None]*S+fslot[None,:],mask=fallback[:,None]&(fslot[None,:]<S),other=-float('inf'))
            fw=tl.where(fallback[:,None]&(fsize[None,:]>0),tl.exp(fraw+tl.log(tl.maximum(fsize,1.))[None,:]-fl[:,None]),0.)
            fvalue=tl.load(V+(kh*S+fslot[:,None])*D+dim[None,:],mask=fslot[:,None]<S,other=0)
            rn+=tl.dot(fw.to(fvalue.dtype),fvalue);rd+=tl.sum(fw,1)
        rem=tl.where(fallback,rd,rem);res=tl.where(fallback[:,None],rn,res)
    local_lse=tl.load(LocalLSE+head*LL0+query*LL1,mask=valid,other=-float('inf'))
    local=tl.load(Local+head*LK0+query[:,None]*LK1+dim[None,:],mask=valid[:,None],other=0).to(tl.float32)
    maximum=tl.maximum(fl,local_lse)
    for rank in tl.static_range(ROUTES):
        slot=tl.load(Slots+row*ROUTES+rank,mask=valid,other=-1)
        el=tl.load(ExactLSE+row*ROUTES+rank,mask=valid&(slot>=0),other=-float('inf'))
        maximum=tl.maximum(maximum,el)
    for i in tl.static_range(SINK):
        sk=tl.load(SinkK+(kh*SINK+i)*D+dim).to(tl.float32)
        maximum=tl.maximum(maximum,tl.sum(q*sk[None,:],1)*SCALE)
    cw=tl.exp(fl-maximum);lw=tl.exp(local_lse-maximum)
    num=cw[:,None]*res+lw[:,None]*local;den=cw*rem+lw
    for rank in tl.static_range(ROUTES):
        slot=tl.load(Slots+row*ROUTES+rank,mask=valid,other=-1)
        active=valid&(slot>=0)
        el=tl.load(ExactLSE+row*ROUTES+rank,mask=active,other=-float('inf'))
        ev=tl.load(Exact+(row*ROUTES+rank)[:,None]*D+dim[None,:],mask=active[:,None],other=0).to(tl.float32)
        w=tl.where(active,tl.exp(el-maximum),0.)
        num+=w[:,None]*ev;den+=w
    for i in tl.static_range(SINK):
        sk=tl.load(SinkK+(kh*SINK+i)*D+dim).to(tl.float32)
        sv=tl.load(SinkV+(kh*SINK+i)*D+dim).to(tl.float32)
        w=tl.exp(tl.sum(q*sk[None,:],1)*SCALE-maximum)
        num+=w[:,None]*sv[None,:];den+=w
    tl.store(Out+head*O0+query[:,None]*O1+dim[None,:],num/tl.maximum(den[:,None],1.e-30),mask=valid[:,None])


@triton.jit
def _select_vector(Q, Scores, RouteLSE, Lengths, Remote, Slots, Counts, Entries, Bits,
                    QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                    WIDTH: tl.constexpr, BOUND: tl.constexpr, WORK: tl.constexpr,
                    M: tl.constexpr, N: tl.constexpr, SAVE_MASK: tl.constexpr):
    head=tl.program_id(0);kh=head//G
    query=tl.program_id(1)*M+tl.arange(0,M);validq=query<QLEN
    row=head*QLEN+query
    dim=tl.arange(0,D);col=tl.arange(0,N)
    q=tl.load(Q+row[:,None]*D+dim[None,:],mask=validq[:,None],other=0).to(tl.float32)
    inv=tl.rsqrt(tl.maximum(tl.sum(q*q,1)/D,1.e-12))
    lz=tl.load(RouteLSE+row,mask=validq,other=float('inf'))
    remote=tl.load(Remote+kh).to(tl.int64)
    cutoff=lz-tl.log(BOUND*1.)-0.5*tl.log(tl.maximum(remote.to(tl.float32),1.))+1.e-6
    before=tl.zeros((M,),tl.int64);selected=tl.zeros((M,),tl.int32);entries=tl.zeros((M,),tl.int32)
    for start in range(tl.cdiv(S,N)):
        slot=start*N+col
        size=tl.load(Lengths+kh*CAP+slot,mask=slot<S,other=0).to(tl.int64)
        dot=tl.load(Scores+row[:,None]*S+slot[None,:],mask=validq[:,None]&(slot[None,:]<S),other=0)
        candidate=validq[:,None]&(slot[None,:]<S)&(size[None,:]>0)&(dot*inv[:,None]>cutoff[:,None])
        add=tl.where(candidate,size[None,:],0)
        prefix=before[:,None]+tl.cumsum(add,1)
        chosen=candidate&(prefix*prefix<WORK*WORK*remote)
        before+=tl.sum(add,1)
        rank=selected[:,None]+tl.cumsum(chosen.to(tl.int32),1)-1
        tl.store(Slots+row[:,None]*WIDTH+rank,slot[None,:],mask=chosen)
        selected+=tl.sum(chosen.to(tl.int32),1)
        entries+=tl.sum(tl.where(chosen,size[None,:],0),1).to(tl.int32)
        if SAVE_MASK:
            bit=tl.arange(0,8)
            packed=tl.sum(tl.reshape(chosen.to(tl.int32),(M,N//8,8))*(1<<bit[None,None,:]),2).to(tl.uint8)
            byte=start*(N//8)+tl.arange(0,N//8)
            tl.store(Bits+row[:,None]*tl.cdiv(S,8)+byte[None,:],packed,mask=validq[:,None]&(byte[None,:]<tl.cdiv(S,8)))
        else:
            tl.store(Scores+row[:,None]*S+slot[None,:],-float('inf'),mask=chosen)
    tl.store(Counts+row,selected,mask=validq)
    tl.store(Entries+row,entries,mask=validq)
