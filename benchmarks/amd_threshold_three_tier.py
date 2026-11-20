"""Three-tier threshold decode with persistent page workers.

Normalized-query region mass selects every eligible region above c/R. Closed
regions contribute their original summaries. Open regions contribute exact
pages and one disjoint residual, with fixed or adaptive page selection.
Only region IDs are packed; KV storage is shared. No sorting or host counts.
"""
import functools
import math
import os

import torch
import triton
import triton.language as tl

from benchmarks.amd_threshold_decode import _local
from benchmarks.amd_threshold_frontier import route_cutoff
from benchmarks.amd_threshold_merge import merge_split_d
from lod_attention.kernels.paged_prefill import query_major_indexed_residual_page_attention


@triton.jit
def scores(Q, StateK, Counts, StateLen, LocalLen, Native, Route, PartLSE, PartCount, Saved,
           S: tl.constexpr, G: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
           TILE: tl.constexpr, NG: tl.constexpr, CACHED: tl.constexpr = False,
           ROUND_COUNTS: tl.constexpr = False, BATCH_KV_HEADS: tl.constexpr = 0, CacheIndices=None):
    kh = tl.program_id(0)
    request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    group = tl.program_id(1)
    lane = tl.arange(0, 16)
    head = kh * G + lane
    i = group * TILE + tl.arange(0, TILE)
    dim = tl.arange(0, D)
    count = tl.load(Counts + cache_kh * S + i, mask=(i < S) & (i < tl.load(StateLen + cache_row)), other=0).to(tl.float32)
    valid = (i < S) & (count > 0)
    q = tl.load(Q + head[:, None] * D + dim[None, :], mask=lane[:, None] < G, other=0).to(tl.bfloat16)
    key = tl.load(StateK + (cache_kh * S + i[:, None]) * D + dim[None, :], mask=valid[:, None], other=0).to(tl.float32)
    if not CACHED:
        divisor = count
        if ROUND_COUNTS:
            divisor = divisor.to(tl.bfloat16).to(tl.float32)
        key = key / tl.maximum(divisor[:, None], 1.)
    key = key.to(tl.bfloat16)
    dot = tl.dot(q, tl.trans(key), out_dtype=tl.float32) * SCALE
    inv = tl.rsqrt(tl.maximum(tl.sum(q.to(tl.float32) * q.to(tl.float32), 1) / D, 1.e-12))
    logcount = tl.log(tl.maximum(count, 1.))
    native = tl.where((lane[:, None] < G) & valid[None, :], dot + logcount[None, :], -float('inf'))
    route = tl.where((lane[:, None] < G) & valid[None, :], dot * inv[:, None] + logcount[None, :], -float('inf'))
    tl.store(Native + head[:, None] * S + i[None, :], native, mask=(lane[:, None] < G) & (i[None, :] < S))
    tl.store(Route + head[:, None] * S + i[None, :], route, mask=(lane[:, None] < G) & (i[None, :] < S))
    mx = tl.max(route, 1)
    safe = tl.where(mx != -float('inf'), mx, 0.)
    z = tl.sum(tl.exp(route - safe[:, None]), 1)
    tl.store(PartLSE + head * NG + group, tl.where(z > 0, safe + tl.log(z), -float('inf')), mask=lane < G)
    tl.store(PartCount + kh * NG + group, tl.sum(valid.to(tl.int32), 0))
    if (kh % BATCH_KV_HEADS == 0 if BATCH_KV_HEADS else kh == 0) and group == 0:
        tl.store(Saved + request, tl.load(LocalLen + cache_row))


@triton.jit
def coarse_pack(Native, Route, PartLSE, PartCount, Counts, StateLen, StateV, Allowed, Tiles, Numbers, Out, LSE,
                S: tl.constexpr, G: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
                TILE: tl.constexpr, NG: tl.constexpr, NGB: tl.constexpr, PRECISE: tl.constexpr,
                CACHED: tl.constexpr = False, ROUND_COUNTS: tl.constexpr = False,
                BATCH_KV_HEADS: tl.constexpr = 0, Cutoffs=None, CacheIndices=None):
    kh = tl.program_id(0)
    request = kh // BATCH_KV_HEADS if BATCH_KV_HEADS else 0
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * BATCH_KV_HEADS + kh % BATCH_KV_HEADS if BATCH_KV_HEADS else kh
    group = tl.program_id(1)
    lane = tl.arange(0, 16)
    head = kh * G + lane
    i = group * TILE + tl.arange(0, TILE)
    dim = tl.arange(0, D)
    if Cutoffs is not None:
        cut = tl.load(Cutoffs + head, mask=lane < G, other=float('inf'))
    else:
        g = tl.arange(0, NGB)
        part = tl.load(PartLSE + head[:, None] * NG + g[None, :], mask=(lane[:, None] < G) & (g[None, :] < NG), other=-float('inf'))
        mx = tl.max(part, 1)
        safe = tl.where(mx != -float('inf'), mx, 0.)
        z = tl.sum(tl.exp(part - safe[:, None]), 1)
        nonempty = tl.sum(tl.load(PartCount + kh * NG + g, mask=g < NG, other=0), 0)
        cut = tl.where(nonempty > 0, safe + tl.log(z) + tl.log(C / nonempty), float('inf'))
    count = tl.load(Counts + cache_kh * S + i, mask=i < S, other=0).to(tl.float32)
    valid = (i < S) & (i < tl.load(StateLen + cache_row)) & (count > 0)
    route = tl.load(Route + head[:, None] * S + i[None, :], mask=(lane[:, None] < G) & (i[None, :] < S), other=-float('inf'))
    opened = (lane[:, None] < G) & valid[None, :] & (count[None, :] <= 1024) & (route > cut[:, None])
    tl.store(Allowed + head[:, None] * S + i[None, :], opened.to(tl.int32), mask=(lane[:, None] < G) & (i[None, :] < S))
    rank = tl.cumsum(opened.to(tl.int32), 1) - 1
    dst = head[:, None] * NG * TILE + group * TILE + rank
    tl.store(Tiles + dst, i[None, :], mask=opened)
    tl.store(Numbers + head * NG + group, tl.sum(opened.to(tl.int32), 1), mask=lane < G)
    native = tl.load(Native + head[:, None] * S + i[None, :], mask=(lane[:, None] < G) & (i[None, :] < S), other=-float('inf'))
    closed = (lane[:, None] < G) & valid[None, :] & ~opened
    score = tl.where(closed, native, -float('inf'))
    mx = tl.max(score, 1)
    safe = tl.where(mx != -float('inf'), mx, 0.)
    w = tl.where(closed, tl.exp(score - safe[:, None]), 0.)
    den = tl.sum(w, 1)
    value = tl.load(StateV + (cache_kh * S + i[:, None]) * D + dim[None, :], mask=valid[:, None], other=0).to(tl.float32)
    if not CACHED:
        divisor = count
        if ROUND_COUNTS:
            divisor = divisor.to(tl.bfloat16).to(tl.float32)
        value = value / tl.maximum(divisor[:, None], 1.)
    value = value.to(tl.bfloat16)
    acc = tl.dot(w.to(tl.bfloat16), value, out_dtype=tl.float32)
    if PRECISE:
        acc += tl.dot((w - w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), value, out_dtype=tl.float32)
    row = head * NG + group
    tl.store(Out + row[:, None] * D + dim[None, :], acc / tl.maximum(den[:, None], 1.e-30), mask=lane[:, None] < G)
    tl.store(LSE + row, tl.where(den > 0, safe + tl.log(den), -float('inf')), mask=lane < G)


@triton.jit
def local_gqa(Q, K, V, SinkK, SinkV, NewK, NewV, Saved, Out, LSE,
              G:tl.constexpr, D:tl.constexpr, KV_HEADS:tl.constexpr, LOCAL_CAP:tl.constexpr,
              LIMIT:tl.constexpr, SINK:tl.constexpr, SCALE:tl.constexpr, TILE:tl.constexpr, NG:tl.constexpr, CacheIndices=None):
    kh = tl.program_id(0)
    group = tl.program_id(1)
    lane = tl.arange(0,16)
    head = kh*G+lane
    dim = tl.arange(0,D)
    pos = group*TILE+tl.arange(0,TILE)
    request = kh // KV_HEADS
    cache_row = tl.load(CacheIndices + request).to(tl.int32) if CacheIndices is not None else request
    cache_kh = cache_row * KV_HEADS + kh % KV_HEADS
    length = tl.load(Saved + request)
    old = (pos<length)&(pos<LIMIT)
    current = pos==length
    sink = (pos>=LIMIT+1)&(pos<LIMIT+1+SINK)
    key = tl.load(K+(cache_kh*LOCAL_CAP+pos[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.bfloat16)
    value = tl.load(V+(cache_kh*LOCAL_CAP+pos[:,None])*D+dim[None,:],mask=old[:,None],other=0).to(tl.bfloat16)
    nk = tl.load(NewK+kh*D+dim)
    nv = tl.load(NewV+kh*D+dim)
    key = tl.where(current[:,None],nk[None,:],key)
    value = tl.where(current[:,None],nv[None,:],value)
    if SINK:
        addr=(cache_kh*SINK+tl.maximum(pos-LIMIT-1,0)[:,None])*D+dim[None,:]
        key=tl.where(sink[:,None],tl.load(SinkK+addr,mask=sink[:,None],other=0),key)
        value=tl.where(sink[:,None],tl.load(SinkV+addr,mask=sink[:,None],other=0),value)
    query=tl.load(Q+head[:,None]*D+dim[None,:],mask=lane[:,None]<G,other=0).to(tl.bfloat16)
    valid=old|current|sink
    logits=tl.where(valid[None,:],tl.dot(query,tl.trans(key),out_dtype=tl.float32)*SCALE,-float('inf'))
    maximum=tl.max(logits,1)
    safe=tl.where(maximum!=-float('inf'),maximum,0.)
    weight=tl.where(valid[None,:],tl.exp(logits-safe[:,None]),0.)
    den=tl.sum(weight,1)
    low=weight.to(tl.bfloat16)
    numerator=tl.dot(low,value,out_dtype=tl.float32)
    numerator+=tl.dot((weight-low.to(tl.float32)).to(tl.bfloat16),value,out_dtype=tl.float32)
    row=head*NG+group
    tl.store(Out+row[:,None]*D+dim[None,:],numerator/tl.maximum(den[:,None],1.e-30),mask=lane[:,None]<G)
    tl.store(LSE+row,tl.where(den>0,safe+tl.log(den),-float('inf')),mask=lane<G)


@triton.jit
def prefix(Numbers, Offsets, Total, NG: tl.constexpr, NGB: tl.constexpr):
    head = tl.program_id(0)
    g = tl.arange(0, NGB)
    n = tl.load(Numbers + head * NG + g, mask=g < NG, other=0)
    tl.store(Offsets + head * NG + g, tl.cumsum(n, 0) - n, mask=g < NG)
    tl.store(Total + head, tl.sum(n, 0))


@triton.jit
def pack(Tiles, Numbers, Offsets, Selected, S: tl.constexpr, NG: tl.constexpr, TILE: tl.constexpr):
    head = tl.program_id(0)
    group = tl.program_id(1)
    i = tl.arange(0, TILE)
    n = tl.load(Numbers + head * NG + group)
    before = tl.load(Offsets + head * NG + group)
    slot = tl.load(Tiles + (head * NG + group) * TILE + i, mask=i < n, other=-1)
    tl.store(Selected + head * S + before + i, slot, mask=i < n)


@triton.jit
def prefix_pack(Tiles, Numbers, Selected, Total,
                S:tl.constexpr, NG:tl.constexpr, NGB:tl.constexpr, TILE:tl.constexpr):
    head=tl.program_id(0)
    group=tl.arange(0,NGB)
    counts=tl.load(Numbers+head*NG+group,mask=group<NG,other=0)
    offsets=tl.cumsum(counts,0)-counts
    lane=tl.arange(0,TILE)
    active=(group[:,None]<NG)&(lane[None,:]<counts[:,None])
    slot=tl.load(Tiles+(head*NG+group[:,None])*TILE+lane[None,:],mask=active,other=0)
    tl.store(Selected+head*S+offsets[:,None]+lane[None,:],slot,mask=active)
    tl.store(Total+head,tl.sum(counts,0))


def _cached_region_means(state):
    """Read the same BF16 means maintained by the serving pool."""
    page = state['page_cache']
    offset = page.get('unified_page1_coarse_offset')
    arena_k = page.get('unified_page1_k')
    arena_v = page.get('unified_page1_v')
    if offset is None or arena_k is None or arena_v is None:
        raise ValueError('cached threshold means require the existing unified K/V arena')
    shape = state['state_k'].shape
    end = offset + math.prod(shape[:-1])
    return arena_k[offset:end].view(shape), arena_v[offset:end].view(shape)


def adaptive_page_attention(pool, q, workers, multiplier):
    """Resolve the AMD BF16 adaptive page kernel and its per-head counters."""
    from benchmarks.h100_threshold_pages import page_threshold_attention

    if pool.state['page_cache'].get('leaf_quant_bits', 0):
        raise ValueError('adaptive page threshold requires BF16 KV')
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('page threshold multiplier must be finite and positive')
    shape = (q.size(0) * q.size(2), workers, 4)
    buffers = vars(pool).setdefault('_amd_page_threshold_buffers', {})
    if shape not in buffers:
        buffers[shape] = torch.empty(shape, device=q.device, dtype=torch.int32)
    stats = pool._amd_page_threshold_stats = buffers[shape]
    k2_geometry = q.size(-1) == 128 and q.size(2) == 8 * pool.kv_heads
    default_warps = '2' if k2_geometry and multiplier == 0.5 else '4'
    return functools.partial(
        page_threshold_attention, multiplier=multiplier, stats_buffer=stats,
        query_score='normalized', rule='page', residual='region',
        page_threshold_warps=int(os.environ.get('LOD_AMD_THRESHOLD_ADAPTIVE_WARPS', default_warps)),
        fast_bf16=True, exact_pages=int(os.environ.get('LOD_AMD_THRESHOLD_ADAPTIVE_TILE', '2')),
        reuse_directory=True, short_scan=int(os.environ.get('LOD_AMD_THRESHOLD_ADAPTIVE_SCAN', '8')),
        fast_residual_logit=True)


def threshold_attention(pool, q, k, v, out, multiplier, *, tile=64, splits=64, rule='region',
                        workers=None, page_warps=None, page_block_n=None, merge_block=None, precise=None,
                        cached_means=None, round_counts_bf16=False, pages_per_slot=2,
                        page_attention=None, page_multiplier=None, factor_int4_page_scales=False, trim_idle_merge=False, materialize_cutoffs=False, gqa_local=False,
                        fused_pack=False, route_warps=4,
                        local_tile=None, cache_indices=None, **_):
    """Same three-tier frontier as serving, with threshold-selected region counts."""
    if (pool.settings.levels != 3 or pool.dtype != torch.bfloat16 or q.ndim != 4
            or q.size(1) != 1 or rule != 'region'):
        raise ValueError('three-tier thresholds require single-token BF16 queries and the region rule')
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('threshold multiplier must be finite and positive')
    if pages_per_slot not in (1, 2, 4, 8):
        raise ValueError('page refinement budget must be 1, 2, 4, or 8')
    batch, _, heads, D = q.shape
    kv_heads = pool.kv_heads
    G = heads // kv_heads
    if heads % kv_heads or (D, G) not in ((128, 4), (128, 8), (256, 6)):
        raise ValueError('unsupported three-tier threshold head geometry')
    st = pool.state
    if (cache_indices is None and st['state_k'].size(0) != batch
            or cache_indices is not None and cache_indices.shape != (batch,)):
        raise ValueError('threshold decode requires one cache index per query row')
    H, KV = batch * heads, batch * kv_heads
    batch_heads = kv_heads if batch > 1 or cache_indices is not None else 0
    page = st['page_cache']
    if page.get('leaf_quant_bits', 0) not in (0, 4):
        raise ValueError('three-tier thresholds support BF16 or INT4 KV')
    if torch.version.hip is not None and os.environ.get('LOD_AMD_THRESHOLD_CACHED_MEANS') == '1':
        cached_means = _cached_region_means(st)
    fixed_scan = os.environ.get('LOD_AMD_THRESHOLD_FIXED_SCAN') if torch.version.hip is not None else None
    if (fixed_scan is not None and page_multiplier is None and page_attention is None
            and pages_per_slot == 2 and not page.get('leaf_quant_bits', 0)):
        from benchmarks.h100_threshold_fixed_pages import fixed_two_page_attention
        page_attention = functools.partial(fixed_two_page_attention, small_scan=int(fixed_scan))
    S = pool.state_capacity
    NG = triton.cdiv(S, tile)
    workers = int(os.environ.get('LOD_AMD_THRESHOLD_PAGE_WORKERS', str(splits))) if workers is None else int(workers)
    if workers < 1 or workers > S:
        raise ValueError('page worker count must lie in [1, state capacity]')
    if page_multiplier is not None:
        if page_attention is not None:
            raise ValueError('choose an adaptive page multiplier or an explicit page kernel')
        page_attention = adaptive_page_attention(pool, q, workers, page_multiplier)
    localcap = st['recent_k'].size(2)
    limit = int(pool.decode_local_limit)
    sink = st['sink_k'].size(2)
    local_tile = tile if local_tile is None or not gqa_local else local_tile
    localgroups = triton.cdiv(limit + 1 + sink, local_tile)
    spec = (S, tile, workers, localgroups, batch, materialize_cutoffs)
    scratch = vars(pool).setdefault('_threshold_three_tier_scratch_by_shape', {})
    b = scratch.get(spec)
    if b is None:
        def f(*shape): return torch.empty(shape, device=q.device, dtype=torch.float32)
        def i(*shape): return torch.empty(shape, device=q.device, dtype=torch.int32)
        b = dict(spec=spec, mask_only=True, three_tier=True, native=f(H, S), route=f(H, S),
                 part_lse=f(H, NG), part_count=i(KV, NG), allowed=i(H, S), saved=i(batch),
                 tiles=i(H, NG, tile), numbers=i(H, NG), offsets=i(H, NG), count=i(H),
                 selected=i(H, S), co=f(H, NG, D), cl=f(H, NG), eo=f(batch, heads, workers, D),
                 el=f(batch, heads, workers), lo=f(H, localgroups, D), ll=f(H, localgroups),
                 cache_indices=torch.arange(batch, device=q.device, dtype=torch.long))
        if materialize_cutoffs:
            b['cutoffs'] = f(H)
        scratch[spec] = b
    pool._threshold_scratch = b
    if os.environ.get('LOD_AMD_THRESHOLD_CAPTURE_QUERY') == '1':
        pool._amd_threshold_probe = (q.clone(), k.clone(), v.clone(), pool.local_lens.clone(), pool.state_lens.clone())
    scale = float(pool.engine.scaling)
    mean_k, mean_v = (st['state_k'], st['state_v']) if cached_means is None else cached_means
    scores[(KV, NG)](q, mean_k, st['counts'], pool.state_lens, pool.local_lens,
                    b['native'], b['route'], b['part_lse'], b['part_count'], b['saved'], S, G, D, scale, tile, NG,
                    cached_means is not None, round_counts_bf16, BATCH_KV_HEADS=batch_heads, CacheIndices=cache_indices, num_warps=route_warps)
    if materialize_cutoffs:
        route_cutoff[(H,)](b['part_lse'], b['part_count'], b['cutoffs'], NG, triton.next_power_of_2(NG),
                           G, float(multiplier), num_warps=1)
    coarse_pack[(KV, NG)](b['native'], b['route'], b['part_lse'], b['part_count'], st['counts'], pool.state_lens, mean_v,
                         b['allowed'], b['tiles'], b['numbers'], b['co'], b['cl'], S, G, D, float(multiplier),
                         tile, NG, triton.next_power_of_2(NG),
                         os.environ.get('LOD_AMD_THRESHOLD_PRECISE', '1') == '1' if precise is None else precise,
                         cached_means is not None, round_counts_bf16, BATCH_KV_HEADS=batch_heads,
                         Cutoffs=b.get('cutoffs'), CacheIndices=cache_indices, num_warps=route_warps)
    if fused_pack:
        prefix_pack[(H,)](b['tiles'], b['numbers'], b['selected'], b['count'],
                                   S, NG, triton.next_power_of_2(NG), tile, num_warps=4)
    else:
        prefix[(H,)](b['numbers'], b['offsets'], b['count'], NG, triton.next_power_of_2(NG), num_warps=1)
        pack[(H, NG)](b['tiles'], b['numbers'], b['offsets'], b['selected'], S, NG, tile, num_warps=1)
    quantized = page.get('leaf_quant_bits', 0) == 4
    kwargs = {}
    if quantized:
        kwargs = {name: page[name] for name in ('quantized_leaf_k', 'quantized_leaf_v', 'page_k_scales',
                    'page_v_scales', 'page_quantized_counts', 'quantized_page_sum_k', 'quantized_page_sum_v',
                    'page_sum_k_scales', 'page_sum_v_scales')}
        if factor_int4_page_scales:
            kwargs['factor_int4_page_scales'] = True
    page_impl = query_major_indexed_residual_page_attention if page_attention is None else page_attention
    page_impl(
        q.reshape(batch, heads, 1, D), st['state_k'], st['state_v'], st['counts'], page['leaf_k'], page['leaf_v'],
        page['page_indices'], page['page_sum_k'], page['page_sum_v'], page['page_counts'], page['slot_pages'],
        page['overflow_page_keys'], page['overflow_page_values'], page['overflow_used'], page['slot_lengths'],
        b['selected'].view(batch, heads, 1, S), cache_indices=b['cache_indices'] if cache_indices is None else cache_indices, kv_group_size=G, scale=scale,
        hash_probes=int(pool.engine.leaf_hash_probes),
        page_block_n=int(pool.engine.leaf_block_n) if page_block_n is None else page_block_n,
        num_warps=int(os.environ.get('LOD_AMD_THRESHOLD_PAGE_WARPS', '2' if D == 256 else '1')) if page_warps is None else page_warps,
        route_parallel=True, route_counts=b['count'], route_workers=workers, output_buffer=b['eo'],
        lse_buffer=b['el'], quant_group_size=int(pool.settings.quant_group_size),
        pages_per_slot=pages_per_slot, **kwargs)
    if gqa_local:
        local_gqa[(KV, localgroups)](q, st['recent_k'], st['recent_v'], st['sink_k'], st['sink_v'], k, v,
            b['saved'], b['lo'], b['ll'], G, D, kv_heads, localcap, limit, sink, scale, local_tile, localgroups,
            CacheIndices=cache_indices, num_warps=4)
    else:
        _local[(H, localgroups)](q, st['recent_k'], st['recent_v'], st['sink_k'], st['sink_v'], k, v,
            b['saved'], b['lo'], b['ll'], KV, G, D, localcap, limit, sink, scale, tile, localgroups,
            BATCH_KV_HEADS=batch_heads, CacheIndices=cache_indices, num_warps=4)
    bd = int(os.environ.get('LOD_AMD_THRESHOLD_MERGE_BLOCK', '32')) if merge_block is None else merge_block
    merge_split_d[(H, triton.cdiv(D, bd))](b['co'], b['cl'], b['eo'], b['el'], b['lo'], b['ll'], k, v,
        st['recent_k'], st['recent_v'], pool.local_lens, b['saved'], out, D, G, KV, localcap,
        NG, workers, localgroups, triton.next_power_of_2(NG + workers + localgroups), bd,
        BATCH_KV_HEADS=batch_heads,
        RouteCount=b['count'] if trim_idle_merge else None, CacheIndices=cache_indices, num_warps=4)
    return out
