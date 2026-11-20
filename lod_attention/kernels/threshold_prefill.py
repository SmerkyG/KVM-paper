"""Sort-free AMD prefill routing with count-weighted, closed-region attention.

QK is evaluated once. A row reduction normalizes the routing distribution,
packs every qualifying region, and writes native attention probabilities with
the opened summaries removed. A second matrix multiplication computes coarse
values. Exact leaves use the existing region-grouped prefill implementation.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl

from ._launch import hip_launch_kwargs
from .aiter_prefill_attention import AiterPrefillCoarse, _workspace_tensor


@triton.jit
def _means(K, V, C, MK, MV, LogC, S: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    kh, s = row // S, row % S
    d = tl.arange(0, D)
    c = tl.load(C + kh * CAP + s).to(tl.float32)
    divisor = tl.maximum(c, 1.)
    tl.store(MK + row * D + d, tl.load(K + (kh * CAP + s) * D + d).to(tl.float32) / divisor)
    tl.store(MV + row * D + d, tl.load(V + (kh * CAP + s) * D + d).to(tl.float32) / divisor)
    tl.store(LogC+row,tl.log(divisor))


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
def _select(Q, Scores, C, LogC, Prob, LSE, Slots, Counts, Entries, Mask, Ranks,
            QLEN: tl.constexpr, S: tl.constexpr, CAP: tl.constexpr,
            H: tl.constexpr, KV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
            CUTOFF: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr,
            NORMALIZE: tl.constexpr, MAX_LEAVES: tl.constexpr, SCALE: tl.constexpr,
            GROUP_PACK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    bh = row // QLEN
    kh = bh // H * KV + (bh % H) // G
    s = tl.arange(0, BLOCK)
    c = tl.load(C + kh * CAP + s, mask=s < S, other=0).to(tl.float32)
    valid = (s < S) & (c > 0)
    dot = tl.load(Scores + row * S + s, mask=s < S, other=0) * SCALE
    inverse = 1.
    if NORMALIZE:
        d = tl.arange(0, D)
        q = tl.load(Q + row * D + d).to(tl.float32)
        inverse = tl.rsqrt(tl.maximum(tl.sum(q * q, 0) / D, 1.e-12))
    logc = tl.load(LogC+kh*S+s,mask=s<S,other=0)
    route = tl.where(valid, dot * inverse + logc, -float('inf'))
    maximum = tl.max(route, 0)
    safe = tl.where(maximum != -float('inf'), maximum, 0.)
    mass = tl.sum(tl.where(valid, tl.exp(route - safe), 0.), 0)
    # Comparing normalized masses avoids subtracting nearly equal log values.
    opened = valid & (c <= MAX_LEAVES) & (tl.exp(route - safe) >
                  CUTOFF * mass / tl.maximum(tl.sum(valid.to(tl.int32), 0), 1))
    tl.store(Counts + row, tl.sum(opened.to(tl.int32), 0))
    tl.store(Entries + row, tl.sum(tl.where(opened, c, 0), 0))
    closed = valid & ~opened
    native = tl.where(closed, dot + logc, -float('inf'))
    mx = tl.max(native, 0)
    native_safe = tl.where(mx != -float('inf'), mx, 0.)
    p = tl.where(closed, tl.exp(native - native_safe), 0.)
    z = tl.sum(p, 0)
    closed_lse = tl.where(z > 0, native_safe + tl.log(z), -float('inf'))
    tl.store(LSE + row, closed_lse)
    tl.store(Prob + row * S + s, p / tl.maximum(z, 1.e-30), mask=s < S)
    rank = tl.cumsum(opened.to(tl.int32), 0) - 1
    tl.store(Slots + row * WIDTH + tl.maximum(rank,0), s, mask=opened)
    if GROUP_PACK:
        bits = tl.arange(0, 8)
        encoded = tl.sum(tl.reshape(opened.to(tl.int32), (BLOCK//8,8)) * (1 << bits[None,:]),1).to(tl.uint8)
        byte = tl.arange(0, BLOCK//8)
        tl.store(Mask + row * tl.cdiv(S,8) + byte, encoded, mask=byte < tl.cdiv(S,8))
        tl.store(Ranks+row*S+s,rank.to(tl.int16),mask=opened)


@triton.jit
def _expert_metadata(Mask, Ranks, Offsets, TileCounts,
        QLEN: tl.constexpr, S: tl.constexpr, WIDTH: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr):
    bh = tl.program_id(0)
    row = tl.program_id(1) * M + tl.arange(0,M)
    s = tl.program_id(2)*N+tl.arange(0,N)
    tile = tl.program_id(1)
    tiles: tl.constexpr = triton.cdiv(QLEN,M)
    bits = tl.load(Mask + (bh*QLEN+row[:,None])*tl.cdiv(S,8)+s[None,:]//8,
                   mask=(row[:,None]<QLEN)&(s[None,:]<S),other=0).to(tl.int32)
    chosen = ((bits >> (s[None,:]%8)) & 1).to(tl.int32)
    total = tl.sum(chosen,0)
    tl.store(TileCounts+(bh*S+s)*tiles+tile,total,mask=s<S)
    if tl.sum(total,0)>0:
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


def threshold_prefill_route_coarse_attention(q, state_k, state_v, counts, *, state_len,
        kv_group_size, scale, normalize_route_query, max_open_leaf_tokens=None,
        slot_lengths=None, buffers=None):
    """Return all eligible regions whose normalized mass exceeds c/R.

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
    width = min(s, int(math.ceil(s / c)) + 1)
    def ws(name, shape, dtype=q.dtype):
        return _workspace_tensor(buffers, 'threshold_prefill_' + name, shape, dtype=dtype, device=q.device)
    mk, mv = ws('mean_k', (b, kv, s, d)), ws('mean_v', (b, kv, s, d))
    logc = ws('log_count',(b,kv,s),torch.float32)
    scores = ws('scores', (b, h, t, s), torch.float32)
    backend = os.environ.get('LOD_AMD_PREFILL_MATMUL', 'blas')
    p = ws('probability', (b, h, t, s))
    slots = ws('slots', (b, h, t, width), torch.int64)
    number = ws('route_counts', (b, h, t), torch.int32)
    entries = ws('entry_counts', (b, h, t), torch.float32)
    lse = ws('coarse_lse', (b, h, t), torch.float32)
    out = ws('coarse_output', (b, h, t, d))
    group_pack = os.environ.get('LOD_AMD_PREFILL_GROUP_PACK', '1') == '1'
    group_m=int(os.environ.get('LOD_AMD_PREFILL_GROUP_M','64'))
    group_n=int(os.environ.get('LOD_AMD_PREFILL_GROUP_N','128'))
    group_warps=int(os.environ.get('LOD_AMD_PREFILL_GROUP_WARPS','8'))
    mask = ws('mask', (b,h,t,triton.cdiv(s,8)), torch.uint8)
    ranks = ws('ranks',(b,h,t,s),torch.int16)
    head_counts = ws('head_counts', (b*h*s,), torch.int32) if group_pack else None
    offsets = ws('offsets', slots.shape, torch.int32) if group_pack else None
    tile_counts = ws('tile_counts', (b*h*s,triton.cdiv(t,group_m)),torch.int32) if group_pack else None
    slots.fill_(-1)
    _means[(b * kv * s,)](state_k, state_v, counts, mk, mv, logc, s, cap, d, num_warps=4)
    if backend == 'blas':
        # Fold the query group into M; each KV head's storage is read directly.
        torch.bmm(q.view(b*kv, kv_group_size*t, d), mk.view(b*kv, s, d).transpose(1,2),
                  out=scores.view(b*kv, kv_group_size*t, s), out_dtype=torch.float32)
    elif backend == 'triton':
        _qk[(b*h, triton.cdiv(t, 32), triton.cdiv(s, 64))](
            q, mk, scores, t, s, h, kv, kv_group_size, d, float(scale), 32, 64,
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
    else:
        raise ValueError('AMD prefill matmul must be triton or blas')
    _select[(b*h*t,)](q, scores, counts, logc, p, lse, slots, number, entries, mask, ranks,
        t, s, cap, h, kv, kv_group_size, d, c, width, triton.next_power_of_2(max(s,8)),
        normalize_route_query, 1024, float(scale), group_pack,
        num_warps=int(os.environ.get('LOD_AMD_PREFILL_SELECT_WARPS','8')), **hip_launch_kwargs(waves_per_eu=1))
    if group_pack:
        _expert_metadata[(b*h,triton.cdiv(t,group_m),triton.cdiv(s,group_n))](mask,ranks,offsets,tile_counts,t,s,width,group_m,group_n,
            num_warps=group_warps, **hip_launch_kwargs(waves_per_eu=1))
        _expert_prefix[(b*h*s,)](tile_counts,head_counts,triton.cdiv(t,group_m),triton.next_power_of_2(triton.cdiv(t,group_m)),
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
        _finish_offsets[(b*h,triton.cdiv(t,4))](slots,offsets,tile_counts,t,s,width,group_m,triton.next_power_of_2(width),
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
    if backend == 'blas':
        torch.bmm(p.view(b*kv, kv_group_size*t, s), mv.view(b*kv, s, d),
                  out=out.view(b*kv, kv_group_size*t, d))
    else:
        _pv[(b*h, triton.cdiv(t, 32), triton.cdiv(d, 64))](
            p, mv, out, t, s, h, kv, kv_group_size, d, 32, 64, 64,
            num_warps=4, **hip_launch_kwargs(waves_per_eu=1))
    coarse = AiterPrefillCoarse(out, lse, out, lse, mk, mv, counts, False,
                               closed_regions_only=True, route_counts=number)
    return slots, coarse, head_counts, offsets
