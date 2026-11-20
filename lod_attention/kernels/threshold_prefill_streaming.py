"""Split, matrix-free region-threshold routing and coarse attention."""

import os

import torch
import triton
import triton.language as tl

from ._launch import hip_launch_kwargs


@triton.jit
def _query(Q, head, query, T: tl.constexpr, D: tl.constexpr, NORMALIZE: tl.constexpr):
    d = tl.arange(0, D)
    row = head.to(tl.int64) * T + query
    q = tl.load(Q + row[:, None] * D + d[None, :], mask=query[:, None] < T, other=0)
    inverse = tl.full(query.shape, 1., tl.float32)
    if NORMALIZE:
        x = q.to(tl.float32)
        inverse = tl.rsqrt(tl.maximum(tl.sum(x * x, 1) / D, 1.e-12))
    return q, inverse, row, d


@triton.jit
def _region_tile(K, Counts, LogCounts, kv_head, region, d, q,
                 S: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr):
    size = tl.load(Counts + kv_head * CAP + region, mask=region < S, other=0)
    valid = (region < S) & (size > 0)
    log_size = tl.load(LogCounts + kv_head * S + region, mask=valid, other=0)
    k = tl.load(K + (kv_head * S + region[None, :]) * D + d[:, None], mask=valid[None, :], other=0)
    return tl.dot(q, k) * SCALE, log_size, size, valid


@triton.jit
def _extend_softmax(logits, maximum, denominator):
    updated_max = tl.maximum(maximum, tl.max(logits, 1))
    safe = tl.where(updated_max != -float('inf'), updated_max, 0.)
    rescale = tl.exp(maximum - safe)
    weights = tl.exp(logits - safe[:, None])
    return updated_max, denominator * rescale + tl.sum(weights, 1), rescale, weights


@triton.jit
def _proxy_stats(K, Counts, LogCounts, kv_head, q, inverse, split,
                 S: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                 M: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr, STAGES: tl.constexpr):
    maximum = tl.full((M,), -float('inf'), tl.float32)
    denominator = tl.full((M,), 0., tl.float32)
    live = tl.full((), 0, tl.int32)
    d = tl.arange(0, D)
    column = tl.arange(0, N)
    for tile in tl.range(split * BLOCKS, tl.minimum((split + 1) * BLOCKS, tl.cdiv(S, N)), num_stages=STAGES):
        dot, log_size, _, valid = _region_tile(K, Counts, LogCounts, kv_head, tile * N + column, d, q,
                                             S, CAP, D, SCALE)
        logits = tl.where(valid[None, :], dot * inverse[:, None] + log_size[None, :], -float('inf'))
        maximum, denominator, _, _ = _extend_softmax(logits, maximum, denominator)
        live += tl.sum(valid.to(tl.int32), 0)
    return maximum, denominator, live


@triton.jit
def _prepare_live_count(Counts, Live, S: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    kh = tl.program_id(0)
    s = tl.arange(0, BLOCK)
    live = tl.load(Counts + kh * CAP + s, mask=s < S, other=0) > 0
    tl.store(Live + kh, tl.sum(live.to(tl.int32), 0))


@triton.jit
def _route_partials(Q, K, Counts, LogCounts, Max, Mass,
                    T: tl.constexpr, BH: tl.constexpr, G: tl.constexpr,
                    S: tl.constexpr, CAP: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                    NORMALIZE: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                    BLOCKS: tl.constexpr, STAGES: tl.constexpr):
    head, split = tl.program_id(1), tl.program_id(2)
    query = tl.program_id(0) * M + tl.arange(0, M)
    q, inverse, row, _ = _query(Q, head, query, T, D, NORMALIZE)
    maximum, mass, _ = _proxy_stats(K, Counts, LogCounts, head // G, q, inverse, split, S, CAP,
                                    D, SCALE, M, N, BLOCKS, STAGES)
    index = split.to(tl.int64) * BH * T + row
    tl.store(Max + index, maximum, mask=query < T)
    tl.store(Mass + index, mass, mask=query < T)


@triton.jit
def _route_cutoff(Max, Mass, Live, RouteMax, Cutoff,
                  T: tl.constexpr, G: tl.constexpr, BH: tl.constexpr,
                  SPLITS: tl.constexpr, SPLIT_BLOCK: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr):
    head = tl.program_id(1)
    query = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row = head.to(tl.int64) * T + query
    split = tl.arange(0, SPLIT_BLOCK)
    index = split[:, None].to(tl.int64) * BH * T + row[None, :]
    valid = (query[None, :] < T) & (split[:, None] < SPLITS)
    maximum = tl.load(Max + index, mask=valid, other=-float('inf'))
    mass = tl.load(Mass + index, mask=valid, other=0.)
    top = tl.max(maximum, 0)
    safe = tl.where(top != -float('inf'), top, 0.)
    denominator = tl.sum(mass * tl.exp(maximum - safe[None, :]), 0)
    live = tl.maximum(tl.load(Live + head // G), 1)
    tl.store(RouteMax + row, safe, mask=query < T)
    tl.store(Cutoff + row, C * denominator / live, mask=query < T)


@triton.jit
def _open_regions(dot, inverse, log_size, size, valid, active, route_max, cutoff):
    route = dot * inverse[:, None] + log_size[None, :]
    return (valid[None, :] & (size[None, :] <= 1024) & active[:, None]
            & (tl.exp(route - route_max[:, None]) > cutoff[:, None]))


@triton.jit
def _store_bits(Bits, opened, row, tile, active, S: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    words_per_row: tl.constexpr = tl.cdiv(S, 32)
    words = tile * (N // 32) + tl.arange(0, N // 32)
    shift = tl.arange(0, 32).to(tl.uint32)
    bits = tl.reshape(opened.to(tl.uint32), (M, N // 32, 32))
    encoded = tl.sum(bits << shift[None, None, :], 2)
    tl.store(Bits + row[:, None] * words_per_row + words[None, :], encoded,
             mask=active[:, None] & (words[None, :] < words_per_row))


@triton.jit
def _coarse_partials(Q, K, V, Counts, LogCounts, RouteMax, Cutoff, Bits,
                      Slots, Number, Out, OutMax, OutMass, OutputLSE,
                      T: tl.constexpr, BH: tl.constexpr, G: tl.constexpr, S: tl.constexpr,
                      CAP: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr,
                      C: tl.constexpr, SCALE: tl.constexpr, NORMALIZE: tl.constexpr,
                      M: tl.constexpr, N: tl.constexpr, BLOCKS: tl.constexpr,
                      STAGES: tl.constexpr, SINGLE: tl.constexpr):
    head, split = tl.program_id(1), tl.program_id(2)
    query = tl.program_id(0) * M + tl.arange(0, M)
    q, inverse, row, d = _query(Q, head, query, T, D, NORMALIZE)
    kh = head // G
    if SINGLE:
        route_max, route_mass, live = _proxy_stats(K, Counts, LogCounts, kh, q, inverse, split,
                                                   S, CAP, D, SCALE, M, N, BLOCKS, STAGES)
        route_max = tl.where(route_max != -float('inf'), route_max, 0.)
        cutoff = C * route_mass / tl.maximum(live, 1)
    else:
        route_max = tl.load(RouteMax + row, mask=query < T, other=0.)
        cutoff = tl.load(Cutoff + row, mask=query < T, other=0.)
    column = tl.arange(0, N)
    maximum = tl.full((M,), -float('inf'), tl.float32)
    denominator = tl.full((M,), 0., tl.float32)
    numerator = tl.full((M, D), 0., tl.float32)
    selected = tl.full((M,), 0, tl.int32)
    for tile in tl.range(split * BLOCKS, tl.minimum((split + 1) * BLOCKS, tl.cdiv(S, N)), num_stages=STAGES):
        region = tile * N + column
        dot, log_size, size, valid = _region_tile(K, Counts, LogCounts, kh, region, d, q, S, CAP, D, SCALE)
        opened = _open_regions(dot, inverse, log_size, size, valid, query < T, route_max, cutoff)
        if SINGLE:
            hit = opened.to(tl.int32)
            rank = selected[:, None] + tl.cumsum(hit, 1) - 1
            tl.store(Slots + row[:, None] * WIDTH + rank, region[None, :], mask=opened)
            selected += tl.sum(hit, 1)
        else:
            _store_bits(Bits, opened, row, tile, query < T, S, M, N)
        logits = tl.where(valid[None, :] & ~opened, dot + log_size[None, :], -float('inf'))
        maximum, denominator, rescale, weights = _extend_softmax(logits, maximum, denominator)
        v = tl.load(V + (kh * S + region[:, None]) * D + d[None, :], mask=valid[:, None], other=0)
        numerator = tl.dot(weights.to(v.dtype), v, numerator * rescale[:, None])
    if SINGLE:
        tl.store(Out + row[:, None] * D + d[None, :], numerator / tl.maximum(denominator[:, None], 1.e-30),
                 mask=query[:, None] < T)
        tl.store(OutputLSE + row, tl.where(denominator > 0, maximum + tl.log(denominator), -float('inf')),
                 mask=query < T)
        tl.store(Number + row, selected, mask=query < T)
    else:
        index = split.to(tl.int64) * BH * T + row
        tl.store(Out + index[:, None] * D + d[None, :], numerator, mask=query[:, None] < T)
        tl.store(OutMax + index, maximum, mask=query < T)
        tl.store(OutMass + index, denominator, mask=query < T)


@triton.jit
def _compact_bits(Bits, Slots, Number, WORDS: tl.constexpr, WIDTH: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    word = tl.arange(0, BLOCK)
    bits = tl.load(Bits + row * WORDS + word, mask=word < WORDS, other=0)
    count = tl.inline_asm_elementwise('v_bcnt_u32_b32 $0, $1, 0', '=v,v', [bits], tl.int32,
                                      is_pure=True, pack=1)
    start = tl.cumsum(count, 0) - count
    tl.store(Number + row, tl.sum(count, 0))
    for i in range(tl.max(count, 0)):
        low = tl.inline_asm_elementwise('v_ffbl_b32 $0, $1', '=v,v', [bits], tl.int32,
                                        is_pure=True, pack=1)
        tl.store(Slots + row * WIDTH + start + i, word * 32 + low, mask=i < count)
        bits &= bits - 1


@triton.jit
def _compact_and_pad_bits(
    Bits, Slots, Number, TOTAL: tl.constexpr, WORDS: tl.constexpr, WIDTH: tl.constexpr,
    ROWS: tl.constexpr, WORD_TILE: tl.constexpr, RANK_TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    word = tl.arange(0, WORD_TILE)
    bits = tl.load(Bits + row[:, None] * WORDS + word[None, :],
                   mask=(row[:, None] < TOTAL) & (word[None, :] < WORDS), other=0)
    count = tl.inline_asm_elementwise('v_bcnt_u32_b32 $0, $1, 0', '=v,v', [bits], tl.int32,
                                      is_pure=True, pack=1)
    number = tl.sum(count, 1)
    rank_base = tl.cumsum(count, 1) - count
    tl.store(Number + row, number, mask=row < TOTAL)
    rank = tl.arange(0, RANK_TILE)
    # Padding occupies only the unwritten suffix. These stores never alias
    # the live ranks written by the bit walk.
    tl.store(Slots + row[:, None] * WIDTH + rank[None, :], -1,
             mask=(row[:, None] < TOTAL) & (rank[None, :] >= number[:, None]) & (rank[None, :] < WIDTH))
    for i in range(tl.max(tl.max(count, 1), 0)):
        low = tl.inline_asm_elementwise('v_ffbl_b32 $0, $1', '=v,v', [bits], tl.int32,
                                         is_pure=True, pack=1)
        tl.store(Slots + row[:, None] * WIDTH + rank_base + i,
                 word[None, :] * 32 + low, mask=(i < count) & (row[:, None] < TOTAL))
        bits &= bits - 1


@triton.jit
def _merge_coarse(Numerator, Max, Mass, Out, LSE, ROWS: tl.constexpr,
                   D: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    split = tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    index = split.to(tl.int64) * ROWS + row
    maximum = tl.load(Max + index, mask=split < SPLITS, other=-float('inf'))
    mass = tl.load(Mass + index, mask=split < SPLITS, other=0)
    numerator = tl.load(Numerator + index[:, None] * D + d[None, :], mask=split[:, None] < SPLITS, other=0)
    top = tl.max(maximum, 0)
    rescale = tl.exp(maximum - tl.where(top != -float('inf'), top, 0.))
    total = tl.sum(mass * rescale, 0)
    value = tl.sum(numerator * rescale[:, None], 0) / tl.maximum(total, 1.e-30)
    tl.store(Out + row * D + d, value)
    tl.store(LSE + row, tl.where(total > 0, top + tl.log(total), -float('inf')))


def _geometry(regions, dim):
    default = '128,64,512,4,1' if dim == 128 else '16,64,512,4,1'
    m, n, partition, warps, stages = map(
        int, os.environ.get('LOD_AMD_PREFILL_STREAM_GEOMETRY', default).split(','))
    if (m not in (16, 32, 64, 128) or n not in (32, 64, 128) or partition < 0 or
            warps not in (4, 8) or stages not in (1, 2)):
        raise ValueError('invalid AMD streaming prefill geometry')
    blocks = triton.cdiv(partition or regions, n)
    return m, n, blocks, triton.cdiv(regions, blocks * n), warps, stages


def streaming_scratch_bytes_per_head(regions, dim):
    """Return the extra split workspace bytes per query head."""
    _, _, _, splits, _, _ = _geometry(regions, dim)
    return 0 if splits == 1 else 4 * triton.cdiv(regions, 32) + splits * (4 * dim + 8) + 8


def streaming_route_coarse(q, mk, mv, counts, log_counts, slots, number, out, lse, *,
                           multiplier, group, scale, normalize, workspace):
    """Recompute QK after the cutoff with FP32 online coarse accumulation."""
    b, h, t, d = q.shape
    kv, s, cap = mk.size(1), mk.size(2), counts.size(2)
    m, n, blocks, splits, warps, stages = _geometry(s, d)
    grid = (triton.cdiv(t, m), b * h, splits)
    rows = b * h * t
    width = slots.size(-1)
    launch = dict(num_warps=warps, **hip_launch_kwargs(waves_per_eu=1))
    if splits > 1:
        part_max = workspace('stream_part_max', (splits, rows), torch.float32)
        part_mass = workspace('stream_part_mass', (splits, rows), torch.float32)
        route_max = workspace('stream_route_max', (rows,), torch.float32)
        cutoff = workspace('stream_cutoff', (rows,), torch.float32)
        live = workspace('stream_live', (b * kv,), number.dtype)
        words = triton.cdiv(s, 32)
        bits = workspace('stream_bits', (rows, words), torch.uint32)
        part_out = workspace('stream_part_out', (splits, rows, d), torch.float32)
        _prepare_live_count[(b * kv,)](counts, live, s, cap, triton.next_power_of_2(s), num_warps=4)
        _route_partials[grid](q, mk, counts, log_counts, part_max, part_mass,
                              t, b*h, group, s, cap, d, scale, normalize, m, n, blocks, stages, **launch)
        _route_cutoff[(triton.cdiv(t, 128), b * h)](
            part_max, part_mass, live, route_max, cutoff, t, group, b*h,
            splits, triton.next_power_of_2(splits), multiplier, 128, num_warps=4)
    else:
        route_max = cutoff = bits = part_max = part_mass = None
        part_out = out
    # The cutoff has consumed proxy statistics. Reuse their storage for the closed statistics.
    _coarse_partials[grid](q, mk, mv, counts, log_counts, route_max, cutoff, bits, slots, number,
                            part_out, part_max, part_mass, lse, t, b*h, group, s, cap, d, width,
                            multiplier, scale, normalize, m, n, blocks, stages, splits == 1, **launch)
    if splits > 1:
        _compact_bits[(rows,)](bits, slots, number, words, width, triton.next_power_of_2(words), num_warps=4)
        _merge_coarse[(rows,)](part_out, part_max, part_mass, out, lse,
                                rows, d, splits, triton.next_power_of_2(splits), num_warps=4)
