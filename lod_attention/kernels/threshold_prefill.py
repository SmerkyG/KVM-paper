"""Sort-free AMD prefill routing with count-weighted, closed-region attention.

The default path saves QK scores and closed-region attention probabilities for
BLAS. Streaming region routing recomputes scores and accumulates closed values
locally. Both use the existing region-grouped kernel for exact leaves.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from ._launch import hip_launch_kwargs
from .aiter_prefill_attention import AiterPrefillCoarse, _workspace_tensor


def _region_score_stride(regions):
    alignment = int(os.environ.get('LOD_AMD_PREFILL_REGION_ALIGNMENT', '1'))
    if alignment not in (1, 16, 32, 64, 128, 256):
        raise ValueError('AMD prefill alignment must be 1, 16, 32, 64, 128 or 256')
    return triton.cdiv(regions, alignment) * alignment


def region_prefill_query_tile(q, state_len, value_dim):
    """Select a power-of-two scratch tile from the padded route capacity."""
    multiplier = float(os.environ['LOD_AMD_PREFILL_THRESHOLD'])
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('AMD prefill threshold multiplier must be finite and >= 1')
    budget = int(os.environ.get('LOD_AMD_PREFILL_REGION_SCRATCH_MIB', '22528')) << 20
    if budget <= 0:
        raise ValueError('AMD region prefill scratch budget must be positive')
    regions = max(1, state_len)
    width = min(regions, math.ceil(regions / multiplier) + 1)
    if os.environ.get('LOD_AMD_PREFILL_MATMUL') == 'stream':
        from .threshold_prefill_streaming import streaming_scratch_bytes_per_head

        route_bytes = streaming_scratch_bytes_per_head(regions, value_dim)
    else:
        bitmap = os.environ.get('LOD_AMD_PREFILL_BITMAP_RANKS') == '1'
        rank_bytes = triton.cdiv(regions, 32) * 6 if bitmap else 2 * regions + triton.cdiv(regions, 8)
        route_bytes = 7 * _region_score_stride(regions) + rank_bytes
    bytes_per_head = route_bytes + width * (value_dim * q.element_size() + 32)
    queries = max(2, budget // (q.size(0) * q.size(1) * bytes_per_head))
    return min(16384, 1 << (queries.bit_length() - 1))


@triton.jit
def _means(K, V, C, MK, MV, LogC, S: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr,
           STRIDE: tl.constexpr, PACKED_LOG_SIZE: tl.constexpr = False,
           PADDED_LOG: tl.constexpr = False):
    row = tl.program_id(0)
    kh, s = row // STRIDE, row % STRIDE
    d = tl.arange(0, D)
    live = s < S
    c = tl.load(C + kh * CAP + s, mask=live, other=0).to(tl.float32)
    divisor = tl.maximum(c, 1.)
    tl.store(MK + row * D + d, tl.load(K + (kh * CAP + s) * D + d, mask=live, other=0).to(tl.float32) / divisor)
    tl.store(MV + row * D + d, tl.load(V + (kh * CAP + s) * D + d, mask=live, other=0).to(tl.float32) / divisor)
    log_size = tl.log(divisor)
    if PACKED_LOG_SIZE:
        log_size = tl.where(c > 1024, -log_size, log_size)
        log_size = tl.where(c > 0, log_size, -float("inf"))
    if PADDED_LOG:
        tl.store(LogC + row, log_size)
    else:
        tl.store(LogC + kh * S + s, log_size, mask=live)


@triton.jit
def _qk(Q, K, Scores, QLEN: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
        KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr):
    bh = tl.program_id(0)
    m = tl.program_id(1) * M + tl.arange(0, M)
    n = tl.program_id(2) * N + tl.arange(0, N)
    d = tl.arange(0, D)
    kh = bh // H * KV + (bh % H) // G
    q = tl.load(Q + (bh * QLEN + m[:, None]) * D + d[None, :], mask=m[:, None] < QLEN, other=0)
    k = tl.load(K + (kh * S + n[None, :]) * D + d[:, None], mask=n[None, :] < S, other=0)
    score = tl.dot(q, k)
    tl.store(Scores + (bh * QLEN + m[:, None]) * S + n[None, :], score,
             mask=(m[:, None] < QLEN) & (n[None, :] < S))


@triton.jit
def _select(Q, Scores, C, LogC, Prob, LSE, Slots, Counts, Mask, Ranks,
            QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, STRIDE: tl.constexpr,
            H: tl.constexpr, KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            CUTOFF: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            NORMALIZE: tl.constexpr, MAX_LEAVES: tl.constexpr, SCALE: tl.constexpr,
            GROUP_PACK: tl.constexpr, PER_ENTRY: tl.constexpr = False,
            ENTRY_BOUND: tl.constexpr = 16, WORK_BOUND: tl.constexpr = 16,
            PROB_SCALE: tl.constexpr = 1., BITMAP_RANKS: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)
    bh = row // QLEN
    kh = bh // H * KV + (bh % H) // G
    s = tl.arange(0, BLOCK)
    c = tl.load(C + kh * CAP + s, mask=s < S, other=0).to(tl.float32)
    valid = (s < S) & (c > 0)
    dot = tl.load(Scores + row * STRIDE + s, mask=s < S, other=0) * SCALE
    inverse = 1.
    if NORMALIZE:
        d = tl.arange(0, D)
        q = tl.load(Q + row * D + d).to(tl.float32)
        inverse = tl.rsqrt(tl.maximum(tl.sum(q * q, 0) / D, 1.e-12))
    logc = tl.load(LogC+kh*S+s,mask=s<S,other=0)
    route = tl.where(valid, dot * inverse + logc, -float('inf'))
    maximum = tl.max(route, 0)
    safe = tl.where(maximum != -float('inf'), maximum, 0.)
    proxy_weights = tl.exp(route - safe)
    mass = tl.sum(proxy_weights, 0)
    # Comparing normalized masses avoids subtracting nearly equal log values.
    if PER_ENTRY:
        sizes = c.to(tl.int64)
        remote = tl.sum(sizes, 0)
        cutoff = safe + tl.log(mass) - tl.log(ENTRY_BOUND * 1.) - .5 * tl.log(tl.maximum(remote, 1).to(tl.float32))
        candidate = valid & (dot * inverse > cutoff + 1.e-6)
        prefix = tl.cumsum(tl.where(candidate, sizes, 0), 0)
        opened = candidate & (prefix * prefix < WORK_BOUND * WORK_BOUND * remote)
    else:
        opened = valid & (c <= MAX_LEAVES) & (proxy_weights >
                      CUTOFF * mass / tl.maximum(tl.sum(valid.to(tl.int32), 0), 1))
    closed = valid & ~opened
    native = tl.where(closed, dot + logc, -float('inf'))
    mx = tl.max(native, 0)
    native_safe = tl.where(mx != -float('inf'), mx, 0.)
    p = tl.where(closed, tl.exp(native - native_safe), 0.)
    z = tl.sum(p, 0)
    closed_lse = tl.where(z > 0, native_safe + tl.log(z), -float('inf'))
    tl.store(LSE + row, closed_lse)
    tl.store(Prob + row * STRIDE + s, p / tl.maximum(z, 1.e-30) * PROB_SCALE, mask=s < STRIDE)
    rank = tl.cumsum(opened.to(tl.int32), 0) - 1
    tl.store(Slots + row * WIDTH + tl.maximum(rank, 0), s, mask=opened)
    if not BITMAP_RANKS:
        tl.store(Counts + row, tl.sum(opened.to(tl.int32), 0))
    if GROUP_PACK:
        if BITMAP_RANKS:
            bits = tl.arange(0, 32)
            lanes = tl.reshape(opened.to(tl.uint32), (BLOCK // 32, 32))
            encoded = tl.sum(lanes << bits[None, :].to(tl.uint32), 1).to(tl.uint32)
            word_counts = tl.sum(lanes, 1).to(tl.int32)
            tl.store(Counts + row, tl.sum(word_counts, 0))
            base = tl.cumsum(word_counts, 0) - word_counts
            word = tl.arange(0, BLOCK // 32)
            words: tl.constexpr = tl.cdiv(S, 32)
            tl.store(Mask + row * words + word, encoded, mask=word < words)
            tl.store(Ranks + row * words + word, base, mask=word < words)
        else:
            bits = tl.arange(0, 8)
            encoded = tl.sum(tl.reshape(opened.to(tl.int32), (BLOCK//8,8)) * (1 << bits[None,:]),1).to(tl.uint8)
            byte = tl.arange(0, BLOCK//8)
            tl.store(Mask + row * tl.cdiv(S,8) + byte, encoded, mask=byte < tl.cdiv(S,8))
            tl.store(Ranks+row*S+s,rank.to(tl.int16),mask=opened)


@triton.jit
def _select_grouped(Q, Scores, Counts, LogCounts, Prob, LSE, Slots, Number, Bits, LiveCount,
                    QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr, STRIDE: tl.constexpr,
                    H: tl.constexpr, KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                    CUTOFF: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                    NORMALIZE: tl.constexpr, SCALE: tl.constexpr, ROWS: tl.constexpr,
                    BITSET: tl.constexpr):
    """Share a region's size and log count across queries in one workgroup."""
    head = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    row = head * QLEN + query
    kv_head = head // H * KV + (head % H) // G
    region = tl.arange(0, BLOCK)
    size = tl.load(Counts + kv_head * CAP + region, mask=region < S, other=0).to(tl.float32)
    log_size = tl.load(LogCounts + kv_head * S + region, mask=region < S, other=0)
    valid = (region < S) & (size > 0)
    active = query < QLEN
    dot = tl.load(Scores + row[:, None] * STRIDE + region[None, :],
                  mask=active[:, None] & (region[None, :] < S), other=0) * SCALE
    inverse = tl.full((ROWS,), 1., tl.float32)
    if NORMALIZE:
        dim = tl.arange(0, D)
        q = tl.load(Q + row[:, None] * D + dim[None, :], mask=active[:, None], other=0).to(tl.float32)
        inverse = tl.rsqrt(tl.maximum(tl.sum(q * q, 1) / D, 1.e-12))
    route = tl.where(valid[None, :], dot * inverse[:, None] + log_size[None, :], -float('inf'))
    top = tl.max(route, 1)
    proxy = tl.exp(route - tl.where(top != -float("inf"), top, 0.)[:, None])
    proxy_mass = tl.sum(proxy, 1)
    if BITSET:
        regions = tl.maximum(tl.load(LiveCount + kv_head), 1)
    else:
        regions = tl.maximum(tl.sum(valid.to(tl.int32), 0), 1)
    opened = active[:, None] & valid[None, :] & (size[None, :] <= 1024)
    opened &= proxy > CUTOFF * proxy_mass[:, None] / regions
    closed = valid[None, :] & ~opened
    native = tl.where(closed, dot + log_size[None, :], -float('inf'))
    maximum = tl.max(native, 1)
    safe = tl.where(maximum != -float("inf"), maximum, 0.)
    weight = tl.exp(native - safe[:, None])
    mass = tl.sum(weight, 1)
    tl.store(LSE + row, tl.where(mass > 0, safe + tl.log(mass), -float('inf')), mask=active)
    tl.store(Prob + row[:, None] * STRIDE + region[None, :], weight / tl.maximum(mass[:, None], 1.e-30),
             mask=active[:, None] & (region[None, :] < STRIDE))
    if BITSET:
        words = tl.arange(0, BLOCK // 32)
        shift = tl.arange(0, 32).to(tl.uint32)
        hits = tl.reshape(opened.to(tl.uint32), (ROWS, BLOCK // 32, 32))
        encoded = tl.sum(hits << shift[None, None, :], 2)
        bit_words: tl.constexpr = tl.cdiv(S, 32)
        tl.store(Bits + row[:, None] * bit_words + words[None, :], encoded,
                 mask=active[:, None] & (words[None, :] < bit_words))
    else:
        rank = tl.cumsum(opened.to(tl.int32), 1) - 1
        tl.store(Slots + row[:, None] * WIDTH + tl.maximum(rank, 0), region[None, :], mask=opened)
        tl.store(Number + row, tl.sum(opened.to(tl.int32), 1), mask=active)


@triton.jit
def _expert_metadata(Mask, Ranks, Offsets, TileCounts,
        QLEN: tl.constexpr, S: tl.constexpr, WIDTH: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr,
        BITMAP_RANKS: tl.constexpr = False, IS_HIP: tl.constexpr = True):
    bh = tl.program_id(0)
    row = tl.program_id(1) * M + tl.arange(0,M)
    s = tl.program_id(2)*N+tl.arange(0,N)
    tile = tl.program_id(1)
    tiles: tl.constexpr = triton.cdiv(QLEN,M)
    if BITMAP_RANKS:
        words: tl.constexpr = tl.cdiv(S, 32)
        word_pos = (bh * QLEN + row[:, None]) * words + s[None, :] // 32
        shift = (s[None, :] % 32).to(tl.uint32)
        bits = tl.load(Mask + word_pos, mask=(row[:, None] < QLEN) & (s[None, :] < S), other=0)
        chosen = ((bits >> shift) & 1).to(tl.int32)
    else:
        bits = tl.load(Mask + (bh*QLEN+row[:,None])*tl.cdiv(S,8)+s[None,:]//8,
                       mask=(row[:,None]<QLEN)&(s[None,:]<S),other=0).to(tl.int32)
        chosen = ((bits >> (s[None,:]%8)) & 1).to(tl.int32)
    total = tl.sum(chosen,0)
    tl.store(TileCounts+(bh*S+s)*tiles+tile,total,mask=s<S)
    if tl.sum(total,0)>0:
        if BITMAP_RANKS:
            base = tl.load(Ranks + word_pos, mask=chosen != 0, other=0).to(tl.int32)
            preceding = bits & ((tl.full((), 1, tl.uint32) << shift) - 1)
            if IS_HIP:
                within = tl.inline_asm_elementwise(
                    'v_bcnt_u32_b32 $0, $1, 0', constraints='=v,v', args=[preceding],
                    dtype=tl.int32, is_pure=True, pack=1)
            else:
                within = tl.inline_asm_elementwise(
                    'popc.b32 $0, $1;', constraints='=r,r', args=[preceding],
                    dtype=tl.int32, is_pure=True, pack=1)
            rank = base + within
        else:
            rank = tl.load(Ranks+(bh*QLEN+row[:,None])*S+s[None,:],mask=chosen!=0,other=0).to(tl.int32)
        offset = tl.cumsum(chosen,0)-1
        destination=(bh*QLEN+row[:,None])*WIDTH+rank
        tl.store(Offsets+destination,offset,mask=chosen!=0)


@triton.jit
def _expert_prefix(TileCounts, HeadCounts, TILES: tl.constexpr, BLOCK: tl.constexpr):
    expert=tl.program_id(0)
    t=tl.arange(0,BLOCK)
    count=tl.load(TileCounts+expert*TILES+t,mask=t<TILES,other=0)
    prefix=tl.cumsum(count,0)
    tl.store(TileCounts+expert*TILES+t,prefix-count,mask=t<TILES)
    tl.store(HeadCounts+expert,tl.sum(count,0))


@triton.jit
def _finish_offsets(Slots, Offsets, TileStarts, QLEN: tl.constexpr, S: tl.constexpr,
                    WIDTH: tl.constexpr, QTILE: tl.constexpr, N: tl.constexpr):
    bh=tl.program_id(0)
    row=tl.program_id(1)*4+tl.arange(0,4)
    r=tl.arange(0,N)
    position=(bh*QLEN+row[:,None])*WIDTH+r[None,:]
    slot=tl.load(Slots+position,mask=(row[:,None]<QLEN)&(r[None,:]<WIDTH),other=-1)
    valid=(row[:,None]<QLEN)&(r[None,:]<WIDTH)&(slot>=0)
    local=tl.load(Offsets+position,mask=valid,other=0)
    prefix=tl.load(TileStarts+(bh*S+tl.maximum(slot,0))*tl.cdiv(QLEN,QTILE)+row[:,None]//QTILE,mask=valid,other=0)
    tl.store(Offsets+position,local+prefix,mask=valid)


@triton.jit
def _pv(P, V, Out, QLEN: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
        KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr, KD: tl.constexpr):
    bh = tl.program_id(0)
    m = tl.program_id(1) * M + tl.arange(0, M)
    d = tl.program_id(2) * KD + tl.arange(0, KD)
    n = tl.arange(0, N)
    kh = bh // H * KV + (bh % H) // G
    acc = tl.zeros((M, KD), tl.float32)
    for start in range(tl.cdiv(S, N)):
        s = start * N + n
        p = tl.load(P + (bh * QLEN + m[:, None]) * S + s[None, :],
                    mask=(m[:, None] < QLEN) & (s[None, :] < S), other=0)
        v = tl.load(V + (kh * S + s[:, None]) * D + d[None, :],
                    mask=(s[:, None] < S) & (d[None, :] < D), other=0)
        acc += tl.dot(p, v)
    tl.store(Out + (bh * QLEN + m[:, None]) * D + d[None, :], acc,
             mask=(m[:, None] < QLEN) & (d[None, :] < D))


@lru_cache(maxsize=None)
def _coarse_projection_stream(device):
    return torch.cuda.Stream(device=device)


def threshold_prefill_route_coarse_attention(q, state_k, state_v, counts, *, state_len,
        kv_group_size, scale, normalize_route_query, max_open_leaf_tokens=None,
        slot_lengths=None, buffers=None, remote_capacity=None, overlap_projection=False):
    """Route by normalized region mass or per-entry mass.

    The padded route width follows from sum(p)=1, and does not truncate
    selections. Counts and launch bounds stay on device; no host reads occur.
    This experimental entry point is BF16 prefill only.
    """
    c = float(os.environ['LOD_AMD_PREFILL_THRESHOLD'])
    if not math.isfinite(c) or c < 1:
        raise ValueError('AMD prefill threshold multiplier must be finite and >= 1')
    b, h, t, d = q.shape
    kv = state_k.size(1)
    s, cap = int(state_len), state_k.size(2)
    if q.dtype != torch.bfloat16 or not q.is_contiguous() or t <= 1 or d not in (128, 256):
        raise ValueError('AMD threshold prefill requires contiguous BF16 multi-query D128/256')
    if h != kv * kv_group_size or not 0 < s <= cap:
        raise ValueError('invalid AMD threshold prefill geometry')
    if max_open_leaf_tokens is not None or slot_lengths is not None:
        raise ValueError('threshold prefill obtains region sizes from counts')
    rule = os.environ.get('LOD_AMD_PREFILL_RULE', 'region')
    if rule not in ('region', 'entry'):
        raise ValueError('AMD prefill threshold rule must be region or entry')
    entry_bound = int(os.environ.get('LOD_AMD_PREFILL_ENTRY_BOUND', '16'))
    work_bound = int(os.environ.get('LOD_AMD_PREFILL_WORK_BOUND', str(entry_bound)))
    if rule == 'entry':
        if not 0 < entry_bound <= 65535 or not 0 < work_bound <= 65535:
            raise ValueError('AMD prefill entry and work multipliers must lie in [1, 65535]')
        width = s if remote_capacity is None else min(s, max(1, math.isqrt(work_bound**2 * remote_capacity)))
    else:
        width = min(s, int(math.ceil(s / c)) + 1)
    def ws(name, shape, dtype=q.dtype):
        return _workspace_tensor(buffers, 'threshold_prefill_' + name, shape, dtype=dtype, device=q.device)
    backend = os.environ.get('LOD_AMD_PREFILL_MATMUL', 'blas')
    streaming = backend == 'stream'
    stride = s if streaming else _region_score_stride(s)
    if stride != s and backend != 'blas':
        raise ValueError('AMD region padding requires BLAS')
    mk, mv = ws('mean_k', (b, kv, stride, d)), ws('mean_v', (b, kv, stride, d))
    selector = os.environ.get('LOD_AMD_PREFILL_SELECTOR', 'triton')
    packed_log_size = not streaming and selector == 'gluon' and os.environ.get('LOD_AMD_PREFILL_PACK_LOG_SIZE') == '1'
    padded_log = packed_log_size and os.environ.get('LOD_AMD_PREFILL_PAD_LOG_SIZE') == '1'
    logc = ws('log_count', (b, kv, stride if padded_log else s), torch.float32)
    probability_precision = os.environ.get('LOD_AMD_PREFILL_PROBABILITY', 'bf16')
    if probability_precision not in ('bf16', 'scaled-fp16'):
        raise ValueError('AMD coarse probability must be bf16 or scaled-fp16')
    scaled_fp16 = probability_precision == 'scaled-fp16'
    if scaled_fp16 and backend != 'blas':
        raise ValueError('AMD scaled FP16 coarse projection uses BLAS')
    slot_type = torch.int64
    if os.environ.get('LOD_AMD_PREFILL_NARROW_SLOTS') == '1':
        slot_type = torch.int16 if s <= torch.iinfo(torch.int16).max else torch.int32
    slots = ws('slots', (b, h, t, width), slot_type)
    number = ws('route_counts', (b, h, t), torch.int32)
    lse = ws('coarse_lse', (b, h, t), torch.float32)
    out = ws('coarse_output', (b, h, t, d))
    group_pack = os.environ.get('LOD_AMD_PREFILL_GROUP_PACK', '1') == '1'
    exact_starts = None
    if os.environ.get('LOD_AMD_PREFILL_PACK_EXACT_ROWS') == '1':
        if os.environ.get('LOD_AMD_PREFILL_FUSED_MERGE') != '1':
            raise ValueError('compact exact rows require the fused prefill merge')
        if slots.numel() > torch.iinfo(torch.int32).max:
            raise ValueError('compact exact rows exceed 32-bit indexing')
        exact_starts = ws('exact_row_starts', (number.numel(),), torch.int32)

    def prepare_exact_starts():
        if exact_starts is not None:
            # The exclusive prefix addresses each query's live leaf outputs.
            counts_flat = number.view(-1)
            exact_starts[:1].zero_()
            torch.cumsum(counts_flat[:-1], dim=0, out=exact_starts[1:])

    if streaming:
        if group_pack or rule != 'region' or torch.version.hip is None:
            raise ValueError('AMD streaming prefill requires compact region-threshold dispatch on ROCm')
        from .threshold_prefill_streaming import streaming_route_coarse

        slots.fill_(-1)
        _means[(b * kv * s,)](state_k, state_v, counts, mk, mv, logc, s, cap, d, s, num_warps=4)
        streaming_route_coarse(q, mk, mv, counts, logc, slots, number, out, lse,
                               multiplier=c, group=kv_group_size, scale=scale,
                               normalize=normalize_route_query, workspace=ws)
        prepare_exact_starts()
        coarse = AiterPrefillCoarse(out, lse, out, lse, mk, mv, counts, False,
                                   closed_regions_only=True, route_counts=number, exact_row_starts=exact_starts)
        return slots, coarse, None, None
    bitset = os.environ.get('LOD_AMD_PREFILL_COMPACT_BITS') == '1'
    compact_rows = int(os.environ.get('LOD_AMD_PREFILL_COMPACT_ROWS', '0'))
    reuse_scores = os.environ.get('LOD_AMD_PREFILL_REUSE_SCORES') == '1'
    if compact_rows not in (0, 1, 2, 4, 8) or (compact_rows and not bitset):
        raise ValueError('invalid AMD bit compaction tile')
    if reuse_scores and (backend != 'blas' or not bitset or selector != 'gluon' or scaled_fp16):
        raise ValueError('reusing scores requires the BLAS wave selector and BF16 probabilities')
    scores = ws('scores', (b, h, t, stride), torch.float32)
    probability_scale = 32768. if scaled_fp16 else 1.
    if reuse_scores:
        # A selector workgroup reads and replaces only its own score rows.
        p = scores.view(q.dtype)[..., :stride]
    else:
        p = ws('probability', (b, h, t, stride), torch.float16 if scaled_fp16 else q.dtype)
    group_m=int(os.environ.get('LOD_AMD_PREFILL_GROUP_M','64'))
    group_n=int(os.environ.get('LOD_AMD_PREFILL_GROUP_N','128'))
    group_warps=int(os.environ.get('LOD_AMD_PREFILL_GROUP_WARPS','8'))
    bitmap = group_pack and os.environ.get('LOD_AMD_PREFILL_BITMAP_RANKS') == '1'
    if bitmap:
        words = triton.cdiv(s, 32)
        mask = ws('bitmap', (b, h, t, words), torch.uint32)
        ranks = ws('bitmap_base', (b, h, t, words), torch.int32 if width > 65535 else torch.uint16)
    elif group_pack:
        mask = ws('mask', (b, h, t, triton.cdiv(s, 8)), torch.uint8)
        ranks = ws('ranks', (b, h, t, s), torch.int16)
    else:
        mask = ranks = None
    head_counts = ws('head_counts', (b*h*s,), torch.int32) if group_pack else None
    offsets = ws('offsets', slots.shape, torch.int32) if group_pack else None
    tile_counts = ws('tile_counts', (b*h*s,triton.cdiv(t,group_m)),torch.int32) if group_pack else None
    if not compact_rows:
        slots.fill_(-1)
    _means[(b * kv * stride,)](
        state_k, state_v, counts, mk, mv, logc, s, cap, d, stride,
        PACKED_LOG_SIZE=packed_log_size, PADDED_LOG=padded_log, num_warps=4)
    if backend == 'blas':
        # Fold the query group into M; each KV head's storage is read directly.
        qk_backend = os.environ.get('LOD_AMD_PREFILL_QK_BACKEND', 'default')
        q_rows = kv_group_size * t
        if qk_backend == 'triton':
            if torch.version.hip is None:
                raise ValueError('tiled centroid QK requires ROCm')
            from .threshold_prefill_qk import centroid_scores

            centroid_scores[(triton.cdiv(q_rows, 64) * triton.cdiv(stride, 128), b * kv)](
                q, mk, scores, q_rows, stride, d, 64, 128, 64,
                num_warps=4, num_stages=1, matrix_instr_nonkdim=16,
            )
        elif qk_backend == 'default':
            torch.bmm(q.view(b * kv, q_rows, d), mk.view(b * kv, stride, d).transpose(1, 2),
                      out=scores.view(b * kv, q_rows, stride), out_dtype=torch.float32)
        else:
            raise ValueError('unknown AMD prefill QK backend')
    elif backend == 'triton':
        _qk[(b*h, triton.cdiv(t, 32), triton.cdiv(s, 64))](
            q, mk, scores, t, s, h, kv, kv_group_size, d, float(scale), 32, 64,
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
    else:
        raise ValueError('AMD prefill matmul must be triton or blas')
    projection_stream = None

    def project_closed():
        torch.bmm(p.view(b * kv, kv_group_size * t, stride), mv.view(b * kv, stride, d),
                  out=out.view(b * kv, kv_group_size * t, d))

    select_rows = int(os.environ.get('LOD_AMD_PREFILL_SELECT_ROWS', '1'))
    warps = int(os.environ.get('LOD_AMD_PREFILL_SELECT_WARPS', '8'))
    if select_rows != 1 or bitset:
        if (select_rows not in ((1, 2, 4, 8) if bitset else (2, 4)) or group_pack or rule != 'region'
                or scaled_fp16 or (bitset and torch.version.hip is None)):
            raise ValueError('grouped selector requires region routing, BF16 and compact dispatch')
        bits = live = None
        if bitset:
            from .threshold_prefill_streaming import _compact_bits, _prepare_live_count

            words = triton.cdiv(s, 32)
            bits = ws('select_bits', (b * h * t, words), torch.uint32)
            live = ws('select_live', (b * kv,), torch.int32)
            _prepare_live_count[(b * kv,)](counts, live, s, cap, triton.next_power_of_2(s), num_warps=4)
        block = triton.next_power_of_2(max(stride, 32 if bitset else 1))
        grid = (b * h, triton.cdiv(t, select_rows))
        launch = dict(num_warps=warps, **hip_launch_kwargs(waves_per_eu=1))
        if selector == 'gluon':
            if not bitset:
                raise ValueError('explicit-wave selector requires compact bitsets')
            from .threshold_prefill_select import select_region_bits
            query_warps = int(os.environ.get('LOD_AMD_PREFILL_SELECT_QUERY_WARPS', '2'))
            vec = int(os.environ.get('LOD_AMD_PREFILL_SELECT_VECTOR', '4'))
            if (query_warps not in (1, 2, 4, 8) or warps % query_warps
                    or vec not in (1, 2, 4, 8)):
                raise ValueError('invalid explicit-wave selector layout')
            select_region_bits[grid](
                q, scores, counts, logc, live, p, lse, bits, t, s, cap, stride,
                d, kv_group_size, c, float(scale), normalize_route_query, select_rows, block,
                warps, query_warps, vec, packed_log_size, p.stride(-2), PADDED_LOG=padded_log, **launch)
        elif selector == 'triton':
            _select_grouped[grid](
                q, scores, counts, logc, p, lse, slots, number, bits, live,
                t, s, cap, stride, h, kv, kv_group_size, d,
                c, width, block, normalize_route_query, float(scale), select_rows, bitset, **launch)
        else:
            raise ValueError('unknown AMD prefill selector')
        if overlap_projection and backend == 'blas' and bitset:
            projection_stream = _coarse_projection_stream(q.device)
            projection_stream.wait_stream(torch.cuda.current_stream(q.device))
            with torch.cuda.stream(projection_stream):
                project_closed()
                for tensor in (p, mv, out):
                    tensor.record_stream(projection_stream)
        if compact_rows:
            from .threshold_prefill_streaming import _compact_and_pad_bits
            total = b * h * t
            _compact_and_pad_bits[(triton.cdiv(total, compact_rows),)](
                bits, slots, number, total, words, width, compact_rows,
                triton.next_power_of_2(words), triton.next_power_of_2(width), num_warps=4)
        elif bitset:
            _compact_bits[(b * h * t,)](bits, slots, number, words, width, triton.next_power_of_2(words), num_warps=4)
    else:
        if packed_log_size:
            raise ValueError('packed log sizes require the wave selector and bitset dispatch')
        _select[(b*h*t,)](q, scores, counts, logc, p, lse, slots, number, mask, ranks,
            t, s, cap, stride, h, kv, kv_group_size, d, c, width, triton.next_power_of_2(max(stride, 32 if bitmap else 8)),
            normalize_route_query, 1024, float(scale), group_pack, rule == 'entry',
            entry_bound, work_bound, probability_scale, BITMAP_RANKS=bitmap,
            num_warps=warps, **hip_launch_kwargs(waves_per_eu=1))
    if group_pack:
        _expert_metadata[(b*h,triton.cdiv(t,group_m),triton.cdiv(s,group_n))](mask,ranks,offsets,tile_counts,t,s,width,group_m,group_n,
            BITMAP_RANKS=bitmap, IS_HIP=torch.version.hip is not None,
            num_warps=group_warps, **hip_launch_kwargs(waves_per_eu=1))
        _expert_prefix[(b*h*s,)](tile_counts,head_counts,triton.cdiv(t,group_m),triton.next_power_of_2(triton.cdiv(t,group_m)),
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
        _finish_offsets[(b*h,triton.cdiv(t,4))](slots,offsets,tile_counts,t,s,width,group_m,triton.next_power_of_2(width),
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
    if scaled_fp16:
        values_fp16 = ws('mean_v_fp16', mv.shape, torch.float16)
        values_fp16.copy_(mv)
        accum = ws('coarse_fp32', out.shape, torch.float32)
        torch.bmm(p.view(b*kv, kv_group_size*t, stride), values_fp16.view(b*kv, stride, d),
                  out=accum.view(b*kv, kv_group_size*t, d), out_dtype=torch.float32)
        out.copy_(accum.mul_(1. / probability_scale))
    elif backend == 'blas':
        if projection_stream is None:
            project_closed()
    else:
        _pv[(b*h, triton.cdiv(t, 32), triton.cdiv(d, 64))](
            p, mv, out, t, s, h, kv, kv_group_size, d, 32, 64, 64,
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
    prepare_exact_starts()
    coarse = AiterPrefillCoarse(out, lse, out, lse, mk[..., :s, :], mv[..., :s, :], counts, False,
                               closed_regions_only=True, route_counts=number, exact_row_starts=exact_starts,
                               projection_stream=projection_stream)
    return slots, coarse, head_counts, offsets
