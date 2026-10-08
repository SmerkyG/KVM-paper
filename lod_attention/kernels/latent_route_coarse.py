"""Layout-controlled, exact top-eight route/coarse kernel for NoPE MLA.

Keep the 512-channel shared latent in MFMA operand layouts, as in the local
attention kernel. Routing uses small max reductions instead of a bitonic sort
of every score tile. Scores and slot IDs retain the reference kernel's exact
FP32 ordering (including ascending-ID tie breaking).
"""

from __future__ import annotations

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _pack(scores, indices):
    bits = scores.to(gl.uint32, bitcast=True)
    ordered = gl.where((bits & 0x80000000) != 0,
                       bits ^ 0xFFFFFFFF, bits ^ 0x80000000).to(gl.int64)
    return (ordered - 2147483648) * 4294967296 + 4294967295 - indices.to(gl.int64)


@gluon.jit
def _unpack(packed):
    slots = (4294967295 - (packed & 0xFFFFFFFF)).to(gl.int64)
    ordered = ((packed >> 32) + 2147483648).to(gl.uint32)
    bits = gl.where((ordered & 0x80000000) == 0,
                    ordered ^ 0xFFFFFFFF, ordered ^ 0x80000000)
    return bits.to(gl.float32, bitcast=True), slots


@gluon.jit(do_not_specialize=["QUERY_LEN", "STATE_LEN"])
def latent_route_coarse_gluon(
    Q, Mean, Counts, Out, Lse, Slots, RouteScores,
    Q_BATCH_STRIDE, Q_HEAD_STRIDE, Q_TOKEN_STRIDE,
    QUERY_LEN, STATE_LEN,
    QUERY_HEADS: gl.constexpr, KV_HEADS: gl.constexpr,
    KV_GROUP_SIZE: gl.constexpr, SCALE: gl.constexpr,
    NORMALIZE_ROUTE_QUERY: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    EARLY_EXIT: gl.constexpr,
):
    batch_head = gl.program_id(0)
    batch = batch_head // QUERY_HEADS
    head = batch_head % QUERY_HEADS
    kv_row = batch * KV_HEADS + head // KV_GROUP_SIZE
    load: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[2, 2])
    op_a: gl.constexpr = gl.DotOperandLayout(0, mfma, 8)
    op_b: gl.constexpr = gl.DotOperandLayout(1, mfma, 8)
    # All eight candidates for a row fit in a single warp. The max reductions
    # below therefore need no cross-warp scratch for the candidate merge.
    select: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    rows = gl.program_id(1) * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load))
    dims = gl.arange(0, 512, gl.SliceLayout(0, load))
    query = gl.load(Q + batch * Q_BATCH_STRIDE + head * Q_HEAD_STRIDE
                    + rows[:, None] * Q_TOKEN_STRIDE + dims[None, :],
                    rows[:, None] < QUERY_LEN, other=0.0)
    if NORMALIZE_ROUTE_QUERY:
        query_fp32 = query.to(gl.float32)
        rms = gl.sqrt(gl.maximum(gl.sum(query_fp32 * query_fp32, 1) / 512, 1.e-12))
        rms = gl.convert_layout(rms, gl.SliceLayout(1, mfma))
    else:
        rms = gl.full((BLOCK_M,), 1.0, gl.float32, gl.SliceLayout(1, mfma))
    query = gl.convert_layout(query, op_a)
    score_rows = gl.program_id(1) * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    score_cols = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma))
    cols = gl.arange(0, BLOCK_N, gl.SliceLayout(1, load))
    maximum = gl.full((BLOCK_M,), -float("inf"), gl.float32, gl.SliceLayout(1, mfma))
    denominator = gl.zeros((BLOCK_M,), gl.float32, gl.SliceLayout(1, mfma))
    accumulator = gl.zeros((BLOCK_M, 512), gl.float32, mfma)
    sentinel: gl.constexpr = -9223372036854775807
    ranks = gl.arange(0, 8, gl.SliceLayout(0, select))
    # Distinct sentinels make the minimum unique before all eight slots fill.
    # Thereafter each packed score/ID is unique by construction.
    best = gl.full((BLOCK_M, 8), sentinel, gl.int64, select) + ranks[None, :]
    for begin in range(0, STATE_LEN, BLOCK_N):
        latent = gl.load(Mean + (kv_row * STATE_LEN + begin + cols[:, None]) * 512
                         + dims[None, :], (begin + cols[:, None]) < STATE_LEN, other=0.0)
        key = gl.convert_layout(latent.permute(1, 0), op_b)
        raw = gl.amd.cdna3.mfma(query, key, gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma)) * SCALE
        count = gl.load(Counts + kv_row * STATE_LEN + begin + score_cols,
                        begin + score_cols < STATE_LEN, other=1.0).to(gl.float32)
        log_count = gl.log(count)
        valid = (score_rows[:, None] < QUERY_LEN) & (begin + score_cols[None, :] < STATE_LEN)
        scores = gl.where(valid, raw + log_count[None, :], -float("inf"))
        route_scores = gl.where(valid, raw + rms[:, None] * log_count[None, :], -float("inf"))
        remaining = _pack(route_scores, begin + score_cols[None, :])
        minimum = gl.convert_layout(gl.min(best, 1), gl.SliceLayout(1, mfma))
        remaining = gl.where(remaining > minimum[:, None], remaining, sentinel)
        rank = 0
        active = gl.max(gl.max(remaining, 1), 0) > sentinel if EARLY_EXIT else True
        while (rank < 8) & active:
            winning = gl.max(remaining, 1)
            remaining = gl.where(remaining == winning[:, None], sentinel, remaining)
            winning = gl.convert_layout(winning, gl.SliceLayout(1, select))
            minimum = gl.min(best, 1)
            replace = (best == minimum[:, None]) & (winning[:, None] > minimum[:, None])
            best = gl.where(replace, winning[:, None], best)
            if EARLY_EXIT:
                minimum = gl.convert_layout(gl.min(best, 1), gl.SliceLayout(1, mfma))
                remaining = gl.where(remaining > minimum[:, None], remaining, sentinel)
                active = gl.max(gl.max(remaining, 1), 0) > sentinel
            rank += 1
        new_maximum = gl.maximum(maximum, gl.max(scores, 1))
        # Masked query rows must not create NaNs in the partial final tile.
        safe_maximum = gl.where(new_maximum == -float("inf"), 0.0, new_maximum)
        correction = gl.exp2((maximum - safe_maximum) * 1.4426950408889634)
        probability = gl.exp2((scores - safe_maximum[:, None]) * 1.4426950408889634)
        denominator = denominator * correction + gl.sum(probability, 1)
        accumulator *= correction[:, None]
        value = gl.convert_layout(latent, op_b)
        accumulator = gl.amd.cdna3.mfma(
            gl.convert_layout(probability.to(Mean.dtype.element_ty), op_a), value, accumulator)
        maximum = new_maximum
    result = gl.convert_layout((accumulator / denominator[:, None]).to(Out.dtype.element_ty), load)
    output_rows = (batch * QUERY_LEN + rows) * QUERY_HEADS + head
    gl.store(Out + output_rows[:, None] * 512 + dims[None, :], result, rows[:, None] < QUERY_LEN)
    gl.store(Lse + batch_head * QUERY_LEN + score_rows,
             maximum + gl.log(denominator), score_rows < QUERY_LEN)
    ordered_best = gl.full((BLOCK_M, 8), sentinel, gl.int64, select)
    for rank in gl.static_range(8):
        winning = gl.max(best, 1)
        best = gl.where(best == winning[:, None], sentinel, best)
        ordered_best = gl.where(ranks[None, :] == rank, winning[:, None], ordered_best)
    selected_scores, selected_slots = _unpack(ordered_best)
    select_rows = gl.program_id(1) * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, select))
    offsets = (batch_head * QUERY_LEN + select_rows[:, None]) * 8 + ranks[None, :]
    gl.store(Slots + offsets, selected_slots, select_rows[:, None] < QUERY_LEN)
    gl.store(RouteScores + offsets, selected_scores, select_rows[:, None] < QUERY_LEN)
