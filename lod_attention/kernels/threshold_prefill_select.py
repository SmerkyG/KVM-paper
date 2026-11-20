"""Wave layouts for the AMD materialized region-threshold selector."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def select_region_bits(
    Query, Scores, Counts, LogCounts, LiveCount, Probability, LSE, Bits,
    T: gl.constexpr, S: gl.constexpr, CAP: gl.constexpr, STRIDE: gl.constexpr,
    D: gl.constexpr, GROUP: gl.constexpr, C: gl.constexpr, SCALE: gl.constexpr,
    NORMALIZE: gl.constexpr, ROWS: gl.constexpr, COLS: gl.constexpr,
    WARPS: gl.constexpr, QUERY_WARPS: gl.constexpr, VEC: gl.constexpr,
    PACKED_LOG_SIZE: gl.constexpr, PROB_STRIDE: gl.constexpr,
    PADDED_LOG: gl.constexpr = False,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VEC], [1, 64], [QUERY_WARPS, WARPS // QUERY_WARPS], [1, 0],
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    head = gl.program_id(0).to(gl.int64)
    query = gl.program_id(1) * ROWS + gl.arange(0, ROWS, layout=row_layout)
    row = head * T + query
    kv_head = head // GROUP
    col = gl.arange(0, COLS, layout=col_layout)

    if PADDED_LOG:
        log_size = gl.load(LogCounts + kv_head * STRIDE + col, mask=col < STRIDE, other=-float("inf"))
    else:
        log_size = gl.load(LogCounts + kv_head * S + col, mask=col < S, other=0)
    if PACKED_LOG_SIZE:
        valid = (col < S) & (log_size != -float("inf"))
        eligible = valid & (log_size >= 0)
        log_size = gl.abs(log_size)
    else:
        size = gl.load(Counts + kv_head * CAP + col, mask=col < S, other=0)
        valid = (col < S) & (size > 0)
        eligible = valid & (size <= 1024)
    active = query < T
    positions = row[:, None] * STRIDE + col[None, :]
    score_limit: gl.constexpr = STRIDE if PADDED_LOG else S
    dot = gl.load(Scores + positions, mask=active[:, None] & (col[None, :] < score_limit), other=0) * SCALE
    inv = gl.full((ROWS,), 1., gl.float32, row_layout)
    if NORMALIZE:
        dim = gl.arange(0, D, layout=col_layout)
        q = gl.load(Query + row[:, None] * D + dim[None, :], mask=active[:, None], other=0).to(gl.float32)
        inv = gl.rsqrt(gl.maximum(gl.sum(q * q, 1) / D, 1.e-12))

    route = gl.where(valid[None, :], dot * inv[:, None] + log_size[None, :], -float("inf"))
    native = gl.where(valid[None, :], dot + log_size[None, :], -float("inf"))
    route_max = gl.max(route, 1)
    route_safe = gl.where(route_max != -float("inf"), route_max, 0.)
    proxy = gl.exp(route - route_safe[:, None])
    live = gl.maximum(gl.load(LiveCount + kv_head), 1)
    cutoff = C * gl.sum(proxy, 1) / live
    opened = active[:, None] & eligible[None, :]
    opened &= proxy > cutoff[:, None]

    native = gl.where(~opened, native, -float("inf"))
    native_max = gl.max(native, 1)
    native_safe = gl.where(native_max != -float("inf"), native_max, 0.)
    weight = gl.exp(native - native_safe[:, None])
    mass = gl.sum(weight, 1)
    gl.store(LSE + row, gl.where(mass > 0, native_safe + gl.log(mass), -float("inf")), mask=active)
    gl.store(Probability + row[:, None] * PROB_STRIDE + col[None, :], weight / gl.maximum(mass[:, None], 1.e-30),
             mask=active[:, None] & (col[None, :] < STRIDE))

    shifted = opened.to(gl.uint32) << (col[None, :] % 32).to(gl.uint32)
    encoded = gl.sum(gl.reshape(shifted, (ROWS, COLS // 32, 32)), 2)
    word_layout: gl.constexpr = encoded.type.layout
    bit_rows = gl.convert_layout(row, gl.SliceLayout(1, word_layout))
    bit_active = gl.convert_layout(active, gl.SliceLayout(1, word_layout))
    word = gl.arange(0, COLS // 32, layout=gl.SliceLayout(0, word_layout))
    words_per_row: gl.constexpr = triton.cdiv(S, 32)
    gl.store(Bits + bit_rows[:, None] * words_per_row + word[None, :], encoded,
             mask=bit_active[:, None] & (word[None, :] < words_per_row))
