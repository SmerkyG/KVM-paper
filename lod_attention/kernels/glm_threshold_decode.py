"""Graph-safe per-head mean-mass routing for GLM's shared 512-channel MLA latent."""

import torch
import triton
import triton.language as tl

from ._decode_scratch import empty_decode_tensor


# A descriptor addresses contiguous entries in the existing centroid directory.
# The physical latent rows themselves can be anywhere in the shared arena.


@triton.jit
def _partial_offset(head, part, dim, parts: tl.constexpr, strip: tl.constexpr):
    return ((head * (512 // strip) + dim // strip) * parts + part) * strip + dim % strip


@triton.jit
def _score_regions(
    Q,
    New,
    Arena,
    Counts,
    StateLen,
    LocalLen,
    Slots,
    Scores,
    TileLSE,
    Active,
    Total,
    Saved,
    S: tl.constexpr,
    H: tl.constexpr,
    GROUPS: tl.constexpr,
    TILES: tl.constexpr,
    T: tl.constexpr,
    COARSE: tl.constexpr,
    LOCAL: tl.constexpr,
    LOCAL_CAP: tl.constexpr,
    SCALE: tl.constexpr,
):
    tile, group = tl.program_id(0), tl.program_id(1)
    row = group // GROUPS
    slot = tl.load(Slots + row).to(tl.int32)
    h = (group % GROUPS) * 16 + tl.arange(0, 16)
    r = tile * T + tl.arange(0, T)
    d = tl.arange(0, 512)
    live = r < tl.load(StateLen + slot)
    n = tl.load(Counts + slot * S + r, mask=live, other=0).to(tl.float32)
    valid = live & (n > 0)
    q = tl.load(Q + (row * H + h[:, None]) * 512 + d[None, :], mask=h[:, None] < H, other=0)
    if T == 128:
        left = tile * T + tl.arange(0, 64)
        half_live = tl.load(StateLen + slot)
        m0 = tl.load(Arena + (COARSE + slot * S + left[:, None]) * 512 + d[None, :],
                     mask=left[:, None] < half_live, other=0)
        m1 = tl.load(Arena + (COARSE + slot * S + left[:, None] + 64) * 512 + d[None, :],
                     mask=left[:, None] + 64 < half_live, other=0)
        x0 = tl.dot(q, tl.trans(m0))
        x1 = tl.dot(q, tl.trans(m1))
        qk = tl.reshape(tl.permute(tl.join(x0, x1), (0, 2, 1)), (16, 128))
        logits = qk * SCALE + tl.log(tl.maximum(n, 1.0))[None, :]
    else:
        mean = tl.load(
            Arena + (COARSE + slot * S + r[:, None]) * 512 + d[None, :], mask=valid[:, None], other=0
        )
        logits = tl.dot(q, tl.trans(mean)) * SCALE + tl.log(tl.maximum(n, 1.0))[None, :]
    logits = tl.where(valid[None, :], logits, -float("inf"))
    tl.store(
        Scores + ((row * H + h[:, None]) * S + r[None, :]),
        logits,
        mask=(r[None, :] < S) & (h[:, None] < H),
    )
    maximum = tl.max(logits, 1)
    safe = tl.where(maximum == -float("inf"), 0.0, maximum)
    lse = safe + tl.log(tl.sum(tl.exp(logits - safe[:, None]), 1))
    tl.store(TileLSE + (row * H + h) * TILES + tile, lse, mask=h < H)
    tl.store(Active + group * TILES + tile, tl.sum(valid.to(tl.int32), 0))
    if tile == 0:
        tl.store(Total + group, 0)
        if group % GROUPS == 0:
            length = tl.load(LocalLen + slot)
            tl.store(Saved + row, length)
            value = tl.load(New + row * 512 + d)
            tl.store(
                Arena + (LOCAL + slot * LOCAL_CAP + length) * 512 + d,
                value,
                mask=length < LOCAL_CAP,
            )


@triton.jit
def _cutoff(TileLSE, Active, row, group, h, H: tl.constexpr, TILES: tl.constexpr, NT: tl.constexpr, C: tl.constexpr):
    t = tl.arange(0, NT)
    lses = tl.load(
        TileLSE + (row * H + h[:, None]) * TILES + t[None, :],
        mask=(t[None, :] < TILES) & (h[:, None] < H),
        other=-float("inf"),
    )
    m = tl.max(lses, 1)
    safe = tl.where(m == -float("inf"), 0.0, m)
    z = tl.sum(tl.exp(lses - safe[:, None]), 1)
    count = tl.sum(tl.load(Active + group * TILES + t, mask=t < TILES, other=0), 0)
    cutoff = safe + tl.log(z) + tl.log(C / tl.maximum(count, 1))
    return cutoff


@triton.jit
def _write_cutoff(TileLSE, Active, Out, H: tl.constexpr, GROUPS: tl.constexpr, TILES: tl.constexpr, NT: tl.constexpr, C: tl.constexpr):
    group = tl.program_id(0)
    row = group // GROUPS
    h = group % GROUPS * 16 + tl.arange(0, 16)
    cutoff = _cutoff(TileLSE, Active, row, group, h, H, TILES, NT, C)
    tl.store(Out + row * H + h, cutoff, mask=h < H)


@triton.jit
def _select_and_pack(
    Scores,
    TileLSE,
    Active,
    Slots,
    Lengths,
    Arena,
    Masks,
    Total,
    Partial,
    LSE,
    S: tl.constexpr,
    H: tl.constexpr,
    GROUPS: tl.constexpr,
    TILES: tl.constexpr,
    NT: tl.constexpr,
    T: tl.constexpr,
    COARSE: tl.constexpr,
    C: tl.constexpr,
    CAP: tl.constexpr,
    PARTS: tl.constexpr,
    Flat,
    FlatBits,
    Offsets,
    DESCRIPTOR_CAP: tl.constexpr,
    PAGE: tl.constexpr,
    Mass=None,
    COMBINE: tl.constexpr=False,
    FlatWriter: tl.constexpr=None, Directory=None, TileStarts=None,
    DIR_STRIDE: tl.constexpr=0, LEAF_START: tl.constexpr=0, COPY: tl.constexpr=0,
    COPY_SHARDS: tl.constexpr=1, STRIP: tl.constexpr=512, PRECISE: tl.constexpr=False,
    SCORE_TILES: tl.constexpr=0, Cutoffs=None, MASK_STRIDE: tl.constexpr=0,
    GUARDED: tl.constexpr=False, UNION: tl.constexpr=False, TileCount=None, OrderedStarts=None,
    PAIR_STATS: tl.constexpr=False,
):
    tile, group = tl.program_id(0), tl.program_id(1)
    row = group // GROUPS
    slot = tl.load(Slots + row).to(tl.int32)
    lane = tl.arange(0, 16)
    h = (group % GROUPS) * 16 + lane
    t = tl.arange(0, NT)
    r = tile * T + tl.arange(0, T)
    d = tl.arange(0, 512)
    if Cutoffs is None:
        cutoff = _cutoff(TileLSE, Active, row, group, h, H, SCORE_TILES if SCORE_TILES else TILES, NT, C)
    else:
        cutoff = tl.load(Cutoffs + row * H + h, mask=h < H, other=float("inf"))
    score = tl.load(
        Scores + (row * H + h[:, None]) * S + r[None, :],
        mask=(h[:, None] < H) & (r[None, :] < S),
        other=-float("inf"),
    )
    length = tl.load(Lengths + slot * S + r, mask=r < S, other=0)
    opened = (score > cutoff[:, None]) & (length[None, :] > 0)
    if CAP > 0:
        opened &= length[None, :] <= CAP
    bits = tl.sum(tl.where(opened, 1 << lane[:, None], 0), 0)
    if UNION:
        opened = (h[:, None] < H) & (bits[None, :] != 0)
        bits = tl.sum(tl.where(opened, 1 << lane[:, None], 0), 0)
    selected = tl.where(bits != 0, length, 0)
    if COPY_SHARDS > 1:
        tl.static_assert(FlatWriter is not None and PAGE == 1)
        width = tl.cdiv(selected, COPY_SHARDS * 16) * 16
        skip = tl.program_id(2) * width
        selected = tl.maximum(0, tl.minimum(width, selected - skip))
    pieces = tl.cdiv(selected, PAGE)
    prefix = tl.cumsum(pieces, 0)
    if Masks is not None:
        if COPY_SHARDS == 1 or tl.program_id(2) == 0:
            if MASK_STRIDE:
                tl.store(Masks + group * MASK_STRIDE + r, bits, mask=r < MASK_STRIDE)
            else:
                tl.store(Masks + group * TILES * T + r, bits)
    # Each tile reserves a disjoint descriptor range. Head masks preserve
    # independent selection even though the latent loads are shared.
    size = tl.max(prefix, 0)
    if TileCount is not None:
        first_offset = tl.load(Offsets + slot * (S + 1) + tile * T)
        start = first_offset // PAGE + tile * T
        tl.store(TileCount + group * TILES + tile, size)
        tl.store(OrderedStarts + group * TILES + tile, start)
    elif GUARDED:
        start = 0
        if size > 0:
            start = tl.atomic_add(Total + group, size, sem="relaxed")
    else:
        start = tl.atomic_add(Total + group, size, sem="relaxed")
    relative = tl.load(Offsets + slot * (S + 1) + r, mask=r < S, other=0)
    if COPY_SHARDS > 1:
        relative += skip
    if FlatWriter is not None:
        tl.static_assert(PAGE == 1)
        FlatWriter(Flat, FlatBits, Directory, TileStarts, group, tile, TILES, T, start,
                   selected, prefix, bits, relative, slot * DIR_STRIDE + LEAF_START,
                   DESCRIPTOR_CAP, COPY)
    else:
        for piece in range(0, tl.max(pieces, 0), 8):
            p = piece + tl.arange(0, 8)
            pos = group * DESCRIPTOR_CAP + start + (prefix - pieces)[:, None] + p[None, :]
            valid = p[None, :] < pieces[:, None]
            offset = relative[:, None] + p[None, :] * PAGE
            remaining = tl.minimum(selected[:, None] - p[None, :] * PAGE, PAGE)
            tl.store(Flat + pos, offset * PAGE + remaining - 1, mask=valid)
            if not UNION or Masks is not None:
                tl.store(FlatBits + pos, bits[:, None], mask=valid)
    if not COMBINE and (COPY_SHARDS == 1 or tl.program_id(2) == 0):
        for sub in tl.static_range(T // 64):
            coarse_tile = tile * (T // 64) + sub
            if T == 64:
                sub_r, sub_open, sub_score = r, opened, score
            else:
                sub_r = coarse_tile * 64 + tl.arange(0, 64)
                ordinal = sub * 64 + tl.arange(0, 64)
                idx = tl.broadcast_to(ordinal[None, :], (16, 64))
                sub_open = tl.gather(opened, idx, 1)
                sub_score = tl.gather(score, idx, 1)
            if T == 64 or coarse_tile < triton.cdiv(S, 64):
                closed = tl.where(sub_open, -float("inf"), sub_score)
                mx = tl.max(closed, 1)
                sm = tl.where(mx == -float("inf"), 0.0, mx)
                w = tl.exp(closed - sm[:, None])
                mass = tl.sum(w, 1)
                mean = tl.load(
                    Arena + (COARSE + slot * S + sub_r[:, None]) * 512 + d[None, :],
                    mask=(sub_r[:, None] < S) & (tl.max(sub_score, 0)[:, None] != -float("inf")),
                    other=0,
                )
                weighted = tl.dot(w.to(tl.bfloat16), mean)
                if PRECISE:
                    weighted += tl.dot((w - w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), mean)
                p = (row * H + h) * PARTS + coarse_tile
                if Mass is not None:
                    tl.store(Partial + _partial_offset((row * H + h)[:, None], coarse_tile, d[None, :], PARTS, STRIP), weighted, mask=h[:, None] < H)
                    stat = (row * H + h) * (2 * PARTS) + coarse_tile if PAIR_STATS else p
                    tl.store(LSE + stat, mx, mask=h < H)
                    tl.store(Mass + stat, mass, mask=h < H)
                else:
                    tl.store(
                        Partial + _partial_offset((row * H + h)[:, None], coarse_tile, d[None, :], PARTS, STRIP),
                        weighted / tl.maximum(mass, 1.0e-30)[:, None],
                        mask=h[:, None] < H,
                    )
                    tl.store(LSE + p, sm + tl.log(mass), mask=h < H)


@triton.jit
def _closed_tile(maximum, mass, acc, MaskBits, Scores, Arena, row, group, slot, h, lane, d, tile,
                 H: tl.constexpr, TILES: tl.constexpr, T: tl.constexpr, S: tl.constexpr,
                 COARSE: tl.constexpr, PRECISE: tl.constexpr):
    r = tile * T + tl.arange(0, T)
    bits = tl.load(MaskBits + group * TILES * T + r)
    summary = tl.load(Scores + (row * H + h[:, None]) * S + r[None, :],
                      mask=(h[:, None] < H) & (r[None, :] < S), other=-float("inf"))
    closed = tl.where((bits[None, :] & (1 << lane[:, None])) == 0, summary, -float("inf"))
    tile_max = tl.max(closed, 1)
    origin = tl.where(tile_max == -float("inf"), 0., tile_max)
    weights = tl.exp(closed - origin[:, None])
    means = tl.load(Arena + (COARSE + slot * S + r[:, None]) * 512 + d[None, :],
                    mask=(tl.max(closed, 0)[:, None] != -float("inf")), other=0)
    weighted = tl.dot(weights.to(tl.bfloat16), means)
    if PRECISE:
        weighted += tl.dot((weights - weights.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), means)
    new_max = tl.maximum(maximum, tile_max)
    safe = tl.where(new_max == -float("inf"), 0., new_max)
    alpha, beta = tl.exp(maximum - safe), tl.exp(tile_max - safe)
    acc = acc * alpha[:, None] + weighted * beta[:, None]
    mass = mass * alpha + tl.sum(weights, 1) * beta
    maximum = new_max
    return maximum, mass, acc


@triton.jit
def _closed_tiles(maximum, mass, acc, MaskBits, Scores, Arena, row, group, slot, h, lane, d, worker,
                  H: tl.constexpr, TILES: tl.constexpr, SPLITS: tl.constexpr, T: tl.constexpr,
                  S: tl.constexpr, COARSE: tl.constexpr, PRECISE: tl.constexpr, STATIC: tl.constexpr):
    if STATIC:
        for bump in tl.static_range(triton.cdiv(TILES, SPLITS)):
            tile = worker + bump * SPLITS
            if tile < TILES:
                maximum, mass, acc = _closed_tile(maximum, mass, acc, MaskBits, Scores, Arena, row,
                    group, slot, h, lane, d, tile, H, TILES, T, S, COARSE, PRECISE)
    else:
        for tile in range(worker, TILES, SPLITS):
            maximum, mass, acc = _closed_tile(maximum, mass, acc, MaskBits, Scores, Arena, row,
                group, slot, h, lane, d, tile, H, TILES, T, S, COARSE, PRECISE)
    return maximum, mass, acc


@triton.jit
def _accumulate_value(w, value, acc, adj, PV: tl.constexpr, PRECISE: tl.constexpr):
    if PV == 2:
        acc = tl.trans(tl.dot(tl.trans(value), tl.trans(w.to(tl.bfloat16)), tl.trans(acc * adj[:, None])))
    elif PV == 1:
        acc = tl.dot(w.to(tl.bfloat16), value, acc * adj[:, None])
    else:
        acc = acc * adj[:, None] + tl.dot(w.to(tl.bfloat16), value)
    if PRECISE:
        acc += tl.dot((w - w.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16), value)
    return acc


@triton.jit
def _attend_exact(
    Q,
    Arena,
    Slots,
    Saved,
    Flat,
    Bits,
    Total,
    Partial,
    LSE,
    H: tl.constexpr,
    GROUPS: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    PARTS: tl.constexpr,
    LOCAL: tl.constexpr,
    LOCAL_CAP: tl.constexpr,
    LIMIT: tl.constexpr,
    SINK: tl.constexpr,
    NSINK: tl.constexpr,
    DESCRIPTOR_CAP: tl.constexpr,
    SCALE: tl.constexpr,
    Fixed,
    INDEX_STRIDE: tl.constexpr,
    LEAF_START: tl.constexpr,
    PAGE: tl.constexpr,
    Mass=None, MaskBits=None, Scores=None,
    S: tl.constexpr=0, COARSE: tl.constexpr=0, T: tl.constexpr=64,
    CONTIGUOUS: tl.constexpr=False, STRIP: tl.constexpr=512, PV: tl.constexpr=0,
    CG: tl.constexpr=False, BASE2: tl.constexpr=False, PRECISE: tl.constexpr=False,
    STATIC_CLOSED: tl.constexpr=False, CLOSED_FIRST: tl.constexpr=False, UNION: tl.constexpr=False,
    SPLIT_DIM: tl.constexpr=False, TileCount=None, OrderedStarts=None,
    SELECT_TILES: tl.constexpr=0, SELECT_PAD: tl.constexpr=1, PAIR_STATS: tl.constexpr=False,
):
    worker, group = tl.program_id(0), tl.program_id(1)
    row = group // GROUPS
    slot = tl.load(Slots + row).to(tl.int32)
    lane = tl.arange(0, 16)
    h = (group % GROUPS) * 16 + lane
    d = tl.arange(0, 512)
    q = tl.load(Q + (row * H + h[:, None]) * 512 + d[None, :], mask=h[:, None] < H, other=0)
    if TileCount is not None:
        g = tl.arange(0, SELECT_PAD)
        count = tl.load(TileCount + group * SELECT_TILES + g, mask=g < SELECT_TILES, other=0)
        ends = tl.cumsum(count, 0)
        starts = tl.load(OrderedStarts + group * SELECT_TILES + g, mask=g < SELECT_TILES, other=0)
        remote = tl.max(ends, 0) * PAGE
    else:
        remote = tl.load(Total + group) * PAGE
    saved = tl.load(Saved + row)
    local = tl.minimum(saved, LIMIT)
    total = remote + local + 1 + NSINK
    maximum = tl.full((16,), -float("inf"), tl.float32)
    mass = tl.full((16,), 0.0, tl.float32)
    acc = tl.full((16, 512), 0.0, tl.float32)
    if MaskBits is not None and CLOSED_FIRST:
        maximum, mass, acc = _closed_tiles(maximum, mass, acc, MaskBits, Scores, Arena, row, group, slot,
                h, lane, d, worker, H, TILES, SPLITS, T, S, COARSE, PRECISE, STATIC_CLOSED)
        if BASE2:
            maximum *= 1.4426950408889634
    if CONTIGUOUS:
        width = tl.cdiv(tl.cdiv(total, N), SPLITS) * N
        first, end, step = worker * width, tl.minimum((worker + 1) * width, total), N
    else:
        first, end, step = worker * N, total, SPLITS * N
    for start in range(first, end, step):
        pos = start + tl.arange(0, N)
        extra = pos - remote
        if Fixed is None:
            tl.static_assert(PAGE == 1)
            physical = tl.load(Flat + group * DESCRIPTOR_CAP + pos, mask=pos < remote, other=0)
            flags = tl.load(Bits + group * DESCRIPTOR_CAP + pos, mask=pos < remote, other=65535)
            valid = pos < total
        else:
            ordinal = pos // PAGE
            if TileCount is not None:
                owner = tl.zeros((N,), tl.int32)
                for bit in tl.static_range(SELECT_PAD.bit_length() - 2, -1, -1):
                    step_size = 1 << bit
                    owner += tl.where(tl.gather(ends, owner + step_size - 1, 0) <= ordinal, step_size, 0)
                ordinal += tl.gather(starts + count - ends, owner, 0)
            page = group * DESCRIPTOR_CAP + ordinal
            desc = tl.load(Flat + page, mask=pos < remote, other=0)
            flags = tl.load(Bits + page, mask=pos < remote, other=65535)
            valid = (pos < total) & ((pos >= remote) | (pos % PAGE <= desc % PAGE))
            physical = tl.load(
                Fixed + slot * INDEX_STRIDE + LEAF_START + desc // PAGE + pos % PAGE,
                mask=valid & (pos < remote),
                other=0,
            )
        local_id = tl.where(extra == local, saved, extra)
        physical = tl.where(extra >= 0, LOCAL + slot * LOCAL_CAP + local_id, physical)
        physical = tl.where(extra > local, SINK + slot * NSINK + extra - local - 1, physical)
        if UNION:
            selected = valid[None, :] & (h[:, None] < H)
        else:
            selected = valid[None, :] & ((flags[None, :] & (1 << lane[:, None])) != 0)
        if SPLIT_DIM:
            half = tl.arange(0, 256)
            q0, q1 = tl.split(tl.permute(tl.reshape(q, (16, 2, 256)), (0, 2, 1)))
            k0 = tl.load(Arena + physical[:, None] * 512 + half[None, :], mask=valid[:, None], other=0, cache_modifier=".cg" if CG else "")
            k1 = tl.load(Arena + physical[:, None] * 512 + 256 + half[None, :], mask=valid[:, None], other=0, cache_modifier=".cg" if CG else "")
            qk = tl.dot(q1, tl.trans(k1), tl.dot(q0, tl.trans(k0)))
        else:
            latent = tl.load(Arena + physical[:, None] * 512 + d[None, :], mask=valid[:, None], other=0,
                             cache_modifier=".cg" if CG else "")
            qk = tl.dot(q, tl.trans(latent))
        scale: tl.constexpr = SCALE * 1.4426950408889634 if BASE2 else SCALE
        scores = tl.where(selected, qk * scale, -float("inf"))
        maximum2 = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(maximum2 == -float("inf"), 0.0, maximum2)
        if BASE2:
            adj = tl.exp2(maximum - safe)
            w = tl.exp2(scores - safe[:, None])
        else:
            adj = tl.exp(maximum - safe)
            w = tl.exp(scores - safe[:, None])
        if SPLIT_DIM:
            a0, a1 = tl.split(tl.permute(tl.reshape(acc, (16, 2, 256)), (0, 2, 1)))
            a0 = _accumulate_value(w, k0, a0, adj, PV, PRECISE)
            a1 = _accumulate_value(w, k1, a1, adj, PV, PRECISE)
            acc = tl.reshape(tl.permute(tl.join(a0, a1), (0, 2, 1)), (16, 512))
        else:
            acc = _accumulate_value(w, latent, acc, adj, PV, PRECISE)
        mass = mass * adj + tl.sum(w, 1)
        maximum = maximum2
    if BASE2:
        maximum *= 0.6931471805599453
    if MaskBits is not None:
        if not CLOSED_FIRST:
            maximum, mass, acc = _closed_tiles(maximum, mass, acc, MaskBits, Scores, Arena, row, group, slot,
                h, lane, d, worker, H, TILES, SPLITS, T, S, COARSE, PRECISE, STATIC_CLOSED)
        p = (row * H + h) * PARTS + worker
    else:
        p = (row * H + h) * PARTS + TILES + worker
    part = worker if MaskBits is not None else TILES + worker
    if Mass is not None:
        off = _partial_offset((row * H + h)[:, None], part, d[None, :], PARTS, STRIP)
        tl.store(Partial + off, acc, mask=h[:, None] < H)
        stat = (row * H + h) * (2 * PARTS) + part if PAIR_STATS else p
        tl.store(LSE + stat, maximum, mask=h < H)
        tl.store(Mass + stat, mass, mask=h < H)
    else:
        tl.store(
            Partial + p[:, None] * 512 + d[None, :],
            acc / tl.maximum(mass, 1.0e-30)[:, None],
            mask=h[:, None] < H,
        )
        tl.store(LSE + p, tl.where(mass > 0, maximum + tl.log(mass), -float("inf")), mask=h < H)


@triton.jit
def _merge_fields(
    Partial,
    LSE,
    Out,
    Final,
    Slots,
    Saved,
    LocalLen,
    H: tl.constexpr,
    P: tl.constexpr,
    BP: tl.constexpr,
    BS: tl.constexpr,
    HS: tl.constexpr,
    DS: tl.constexpr,
    Mass=None, STRIP: tl.constexpr=512, BD: tl.constexpr=32, CG: tl.constexpr=False, PAIR_STATS: tl.constexpr=False,
):
    head, strip = tl.program_id(0), tl.program_id(1)
    p = tl.arange(0, BP)
    d = strip * BD + tl.arange(0, BD)
    offset = head * (2 * P if PAIR_STATS else P) + p
    lse = tl.load(LSE + offset, mask=p < P, other=-float("inf"))
    maximum = tl.max(lse, 0)
    w = tl.exp(lse - maximum)
    if Mass is None:
        z = tl.sum(w, 0)
    else:
        z = tl.sum(w * tl.load(Mass + offset, mask=p < P, other=0.), 0)
    v = tl.load(
        Partial + _partial_offset(head, p[:, None], d[None, :], P, STRIP), mask=p[:, None] < P, other=0.0, cache_modifier=".cg" if CG else ""
    )
    result = tl.sum(w[:, None] * v, 0) / z
    row = head // H
    tl.store(Out + row * BS + head % H * HS + d * DS, result)
    if strip == 0:
        tl.store(Final + head, maximum + tl.log(z))
        if head % H == 0:
            slot = tl.load(Slots + row).to(tl.int32)
            tl.store(LocalLen + slot, tl.load(Saved + row) + 1)


def reserve(pool):
    if hasattr(pool, "_glm_threshold_buffers"):
        return pool._glm_threshold_buffers
    batch, heads, states = pool.max_requests, pool.query_heads, pool.state_capacity
    groups, tiles = triton.cdiv(heads, 16), triton.cdiv(states, 64)
    parts = tiles + 256
    # Sum of ceil(n_r / P) over disjoint regions is at most n/P + number of regions.
    # The smallest supported descriptor covers eight entries.
    capacity = min(pool.leaf_capacity, triton.cdiv(pool.leaf_capacity, 8) + states)

    def alloc(name, dtype, *shape):
        return empty_decode_tensor(
            "glm_threshold_" + name,
            *shape,
            dtype=dtype,
            device=pool.device,
            shared_scratch=pool.shared_decode_scratch,
        )

    f, i = torch.float32, torch.int32
    pool._glm_threshold_buffers = dict(
        scores=alloc("scores", f, batch, heads, states),
        tlse=alloc("tlse", f, batch, heads, tiles),
        active=alloc("active", i, batch, groups, tiles),
        total=alloc("total", i, batch, groups),
        saved=alloc("saved", i, batch),
        flat=alloc("flat", i, batch, groups, capacity),
        bits=alloc("bits", i, batch, groups, capacity),
        partial=alloc("partial", f, batch, heads, parts, 512),
        lse=alloc("lse", f, batch, heads, parts),
        final=torch.empty(batch, heads, device=pool.device, dtype=f),
    )
    return pool._glm_threshold_buffers


# Confirmed MI325X shapes for BF16 GLM MLA with variable-size regions.
AMD_GQA_TUNING = {
    1: dict(split_dim=True, pv=2, entry=64, static=True, first=True, guarded=True, strip=64, merge=16),
    8: dict(split_dim=True, pv=2, entry=64, score_tile=128, selector=128, guarded=True, strip=64),
}


def attention(pool, query, key, output, c, *, mask_audit=None):
    """Select by relative region mass, pack descriptors and merge exact and coarse fields."""
    batch, heads, dim = query.shape
    if dim != 512 or pool.kv_heads != 1 or pool.dtype != torch.bfloat16 or pool.dcp_world_size != 1:
        raise ValueError("GLM threshold decode requires unsharded BF16 shared latent KV")
    q = query.contiguous()
    key = key.contiguous()
    pool._glm_threshold_called = True
    groups, states = triton.cdiv(heads, 16), pool.state_capacity
    tile, ng = 64, triton.cdiv(states, 64)
    workers, page_size = (256, 32) if batch == 1 else (64, 8)
    tune = getattr(pool, "_glm_threshold_tuning", None)
    if tune is None:
        tune = AMD_GQA_TUNING.get(batch, {}) if getattr(pool, "glm_threshold_gqa_ports", False) else {}
    if not hasattr(pool, "_glm_threshold_shapes"):
        pool._glm_threshold_shapes = {}
    pool._glm_threshold_shapes[batch] = tune
    captured = getattr(pool, "glm_threshold_capture_audit", None)
    if captured is not None and batch not in captured and torch.cuda.is_current_stream_capturing():
        captured[batch] = tune
    pool._glm_threshold_last_tuning = tune
    workers = tune.get("splits", workers)
    entry_tile = tune.get("entry", 32)
    score_tile = tune.get("score_tile", tile)
    score_tiles = triton.cdiv(states, score_tile)
    sel_tile = tune.get("selector", tile)
    sel_tiles = triton.cdiv(states, sel_tile)
    strip = tune.get("strip", 512)
    merge_tile = tune.get("merge", 32)
    page = pool.state["page_cache"]
    arena = page["unified_page1_k"]
    fixed = page["unified_page1_fixed_indices"]
    offsets = page["unified_page1_fixed_slot_offsets"]
    local, coarse, sink = (
        int(page["unified_page1_" + x + "_offset"]) for x in ("local", "coarse", "sink")
    )
    nsink = pool.state["sink_k"].size(2)
    slots = pool.active_indices
    buf = reserve(pool)
    deferred = pool.glm_threshold_deferred_merge
    combine, raw, contiguous = getattr(pool, "_glm_threshold_execution", (deferred and batch == 1, deferred, deferred and batch == 1))
    combine = tune.get("combine", combine)
    contiguous = tune.get("contiguous", contiguous)
    parts = workers if combine else ng + workers
    if combine and mask_audit is None:
        if "mask_tiles" not in buf:
            buf["mask_tiles"] = empty_decode_tensor("glm_threshold_mask_tiles", pool.max_requests, groups, ng, tile,
                device=pool.device, dtype=torch.int32, shared_scratch=pool.shared_decode_scratch)
        mask_audit = buf["mask_tiles"]
    if raw and "mass" not in buf:
        buf["mass"] = empty_decode_tensor("glm_threshold_mass", *buf["lse"].shape, device=pool.device,
            dtype=torch.float32, shared_scratch=pool.shared_decode_scratch)
    tile_lse, active = buf["tlse"], buf["active"]
    if score_tiles > ng:
        if "small_score_lse" not in buf:
            buf["small_score_lse"] = empty_decode_tensor("glm_threshold_score32_lse", pool.max_requests, heads,
                triton.cdiv(states, 32), dtype=torch.float32, device=pool.device, shared_scratch=pool.shared_decode_scratch)
            buf["small_score_active"] = empty_decode_tensor("glm_threshold_score32_active", pool.max_requests, groups,
                triton.cdiv(states, 32), dtype=torch.int32, device=pool.device, shared_scratch=pool.shared_decode_scratch)
        tile_lse, active = buf["small_score_lse"], buf["small_score_active"]
    cutoffs = None
    if tune.get("cutoff"):
        if "cutoffs" not in buf:
            buf["cutoffs"] = torch.empty_like(buf["final"])
        cutoffs = buf["cutoffs"]
    mass = buf["mass"] if raw else None
    lse = buf["lse"]
    pair_stats = tune.get("pair_stats", False)
    if pair_stats:
        if not raw:
            raise ValueError("Packed maximum and mass require deferred normalization")
        if "pairs" not in buf:
            buf["pairs"] = torch.empty(2 * lse.numel(), dtype=lse.dtype, device=q.device)
        lse = buf["pairs"]
        mass = lse[parts:]
    dir_ids, dir_bits = buf["flat"], buf["bits"]
    tile_count = ordered_starts = None
    if tune.get("ordered"):
        if "ordered_ids" not in buf:
            cap = triton.cdiv(pool.leaf_capacity, 8) + triton.cdiv(states, 128) * 128
            buf["ordered_ids"] = torch.empty((pool.max_requests, groups, cap), dtype=torch.int32, device=q.device)
            buf["ordered_bits"] = torch.empty_like(buf["ordered_ids"])
            buf["ordered_count"] = torch.empty_like(buf["active"])
            buf["ordered_starts"] = torch.empty_like(buf["active"])
        dir_ids, dir_bits = buf["ordered_ids"], buf["ordered_bits"]
        tile_count, ordered_starts = buf["ordered_count"], buf["ordered_starts"]
    if tune.get("half_masks", False):
        dir_bits = dir_bits.view(torch.int16).flatten()[:dir_bits.numel()].view_as(buf["bits"])
    capacity = dir_ids.size(-1)
    begin = pool.decode_local_limit + nsink + states
    # Each graph uses a compact prefix of shared, fixed-address scratch.
    _score_regions[(score_tiles, batch * groups)](
        q,
        key,
        arena,
        pool.state["counts"],
        pool.state_lens,
        pool.local_lens,
        slots,
        buf["scores"],
        tile_lse,
        active,
        buf["total"],
        buf["saved"],
        states,
        heads,
        groups,
        score_tiles,
        score_tile,
        coarse,
        local,
        pool.local_capacity,
        pool.engine.scaling,
        num_warps=4,
    )
    if cutoffs is not None:
        _write_cutoff[(batch * groups,)](tile_lse, active, cutoffs, heads, groups, score_tiles, triton.next_power_of_2(score_tiles), c, num_warps=4)
    _select_and_pack[(sel_tiles, batch * groups)](
        buf["scores"],
        tile_lse,
        active,
        slots,
        page["slot_lengths"],
        arena,
        mask_audit,
        buf["total"],
        buf["partial"],
        lse,
        states,
        heads,
        groups,
        sel_tiles,
        triton.next_power_of_2(score_tiles),
        sel_tile,
        coarse,
        c,
        pool.engine.max_open_centroid_leaves or 0,
        parts,
        dir_ids,
        dir_bits,
        offsets,
        capacity,
        page_size,
        TileCount=tile_count, OrderedStarts=ordered_starts, SCORE_TILES=score_tiles, MASK_STRIDE=ng * tile, Cutoffs=cutoffs, UNION=tune.get("union", False),
        PAIR_STATS=pair_stats, GUARDED=tune.get("guarded", False), Mass=mass, COMBINE=combine, STRIP=strip, PRECISE=tune.get("precise", False), num_warps=4,
    )
    _attend_exact[(workers, batch * groups)](
        q,
        arena,
        slots,
        buf["saved"],
        dir_ids,
        dir_bits,
        buf["total"],
        buf["partial"],
        lse,
        heads,
        groups,
        entry_tile,
        workers,
        ng,
        parts,
        local,
        pool.local_capacity,
        pool.decode_local_limit,
        sink,
        nsink,
        capacity,
        pool.engine.scaling,
        fixed,
        fixed.size(-1),
        begin,
        page_size,
        Mass=mass, MaskBits=mask_audit if combine else None, Scores=buf["scores"] if combine else None,
        S=states, COARSE=coarse, T=tile, CONTIGUOUS=contiguous,
        PAIR_STATS=pair_stats, TileCount=tile_count, OrderedStarts=ordered_starts, SELECT_TILES=sel_tiles, SELECT_PAD=triton.next_power_of_2(sel_tiles),
        UNION=tune.get("union", False), STRIP=strip, PV=tune.get("pv", 0), CG=tune.get("cg", False), BASE2=tune.get("base2", False),
        PRECISE=tune.get("precise", False), STATIC_CLOSED=tune.get("static", False),
        CLOSED_FIRST=tune.get("first", False), SPLIT_DIM=tune.get("split_dim", False), num_warps=tune.get("warps", 4),
        num_stages=1, **tune.get("backend", {}),
    )
    _merge_fields[(batch * heads, 512 // merge_tile)](
        buf["partial"],
        lse,
        output,
        buf["final"],
        slots,
        buf["saved"],
        pool.local_lens,
        heads,
        parts,
        triton.next_power_of_2(parts),
        *output.stride(),
        PAIR_STATS=pair_stats, Mass=mass, STRIP=strip, BD=merge_tile, CG=tune.get("merge_cg", False),
        num_warps=4,
    )
    return output, buf["final"][:batch]
