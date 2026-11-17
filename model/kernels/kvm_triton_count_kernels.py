"""Count-readout variants of the Triton KVM state kernels.

These mirror the birth-radius kernels in kvm_triton_training_kernels.
Slots start with count one and gain one per routed merge. Readout divides
the gate-weighted value sum by the count, which has no gradient.
"""

import triton
import triton.language as tl


def adjust_counts_by_route(state_count, best_idx_by_token, state_bound, macro_id, step):
    """Apply or rewind one count contribution for each recorded merge."""
    kv_rows, macro_blocks, macro_block = best_idx_by_token.shape
    token_block = min(64, macro_block)
    _adjust_counts_by_route_kernel[(kv_rows, triton.cdiv(macro_block, token_block))](
        state_count,
        best_idx_by_token,
        state_bound,
        macro_id,
        MAX_STATE_LEN=state_count.shape[1],
        MACRO_BLOCKS=macro_blocks,
        MACRO_BLOCK=macro_block,
        TOKEN_BLOCK=token_block,
        STEP=step,
        num_warps=1,
    )


@triton.jit
def _adjust_counts_by_route_kernel(
    state_count,
    best_idx_by_token,
    state_bound,
    macro_id,
    MAX_STATE_LEN: tl.constexpr,
    MACRO_BLOCKS: tl.constexpr,
    MACRO_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    STEP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = tl.program_id(1) * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    best_idx = tl.load(
        best_idx_by_token + (row * MACRO_BLOCKS + macro_id) * MACRO_BLOCK + token,
        mask=token < MACRO_BLOCK,
        other=-1,
    )
    bound = tl.load(state_bound + macro_id)
    tl.atomic_add(
        state_count + row * MAX_STATE_LEN + best_idx,
        STEP,
        mask=(best_idx >= 0) & (best_idx < bound),
        sem="relaxed",
    )


@triton.jit
def _reduce_oldstate_maxsim_global_append_kernel(
    state_k,
    state_v,
    state_k_attn,
    state_v_attn,
    state_count,
    overflow_k,
    overflow_v,
    append_pos_by_token,
    partial_select_scores,
    before_by_macro,
    n_append_by_macro,
    macro_id,
    ln_weight,
    ln_bias,
    MAX_STATE_LEN: tl.constexpr,
    MAX_STATE_GROUPS: tl.constexpr,
    MACRO_BLOCKS: tl.constexpr,
    MACRO_BLOCK: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
    SUB_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    LN_EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token_offsets = tl.arange(0, MACRO_BLOCK)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)

    state_before = tl.load(before_by_macro + macro_id)
    n_append = tl.load(n_append_by_macro + macro_id)
    overflow_idx = macro_id * MACRO_BLOCK + token_offsets
    sub_id = token_offsets // SUB_BLOCK
    sub_token = token_offsets - sub_id * SUB_BLOCK

    state_k_row = state_k + row * MAX_STATE_LEN * HEAD_DIM
    state_v_row = state_v + row * MAX_STATE_LEN * VALUE_DIM
    state_k_attn_row = state_k_attn + row * MAX_STATE_LEN * HEAD_DIM
    state_v_attn_row = state_v_attn + row * MAX_STATE_LEN * VALUE_DIM
    state_count_row = state_count + row * MAX_STATE_LEN
    overflow_k_row = overflow_k + row * (MACRO_BLOCKS * MACRO_BLOCK) * HEAD_DIM
    overflow_v_row = overflow_v + row * (MACRO_BLOCKS * MACRO_BLOCK) * VALUE_DIM
    append_pos_row = append_pos_by_token + row * MACRO_BLOCKS * MACRO_BLOCK
    partial_select_row = partial_select_scores + row * SUB_BLOCKS * MAX_STATE_GROUPS * SUB_BLOCK

    select_best_score = tl.full((MACRO_BLOCK,), -float("inf"), tl.float32)

    for group_id in tl.range(0, MAX_STATE_GROUPS, 1, num_stages=1):
        offsets = (
            sub_id * MAX_STATE_GROUPS * SUB_BLOCK
            + group_id * SUB_BLOCK
            + sub_token
        )
        select_score = tl.load(partial_select_row + offsets)
        select_best_score = tl.maximum(select_best_score, select_score)

    lower_score = select_best_score[None, :] < select_best_score[:, None]
    tie_before = (select_best_score[None, :] == select_best_score[:, None]) & (
        token_offsets[None, :] < token_offsets[:, None]
    )
    score_rank = tl.sum((lower_score | tie_before).to(tl.int32), axis=1)
    append_token = score_rank < n_append

    append_token_i32 = append_token.to(tl.int32)
    append_rank = tl.cumsum(append_token_i32, axis=0) - append_token_i32
    append_pos = state_before + append_rank
    tl.store(
        append_pos_row + macro_id * MACRO_BLOCK + token_offsets,
        tl.where(append_token, append_pos, -1),
    )
    o_k = tl.load(
        overflow_k_row + overflow_idx[:, None] * HEAD_DIM + key_offsets[None, :]
    )
    o_v = tl.load(
        overflow_v_row
        + overflow_idx[:, None] * VALUE_DIM
        + value_offsets[None, :]
    )
    ln_w = tl.load(ln_weight + key_offsets).to(tl.float32)
    ln_b = tl.load(ln_bias + key_offsets).to(tl.float32)
    o_k_float = o_k.to(tl.float32)
    o_k_mean = tl.sum(o_k_float, axis=1) / HEAD_DIM
    o_k_centered = o_k_float - o_k_mean[:, None]
    o_k_var = tl.sum(o_k_centered * o_k_centered, axis=1) / HEAD_DIM
    o_k_norm = o_k_centered * tl.rsqrt(o_k_var[:, None] + LN_EPS)
    o_k_norm = (o_k_norm * ln_w[None, :] + ln_b[None, :]).to(tl.bfloat16)
    tl.store(
        state_k_row + append_pos[:, None] * HEAD_DIM + key_offsets[None, :],
        o_k,
        mask=append_token[:, None],
    )
    tl.store(
        state_v_row + append_pos[:, None] * VALUE_DIM + value_offsets[None, :],
        o_v,
        mask=append_token[:, None],
    )
    tl.store(
        state_k_attn_row + append_pos[:, None] * HEAD_DIM + key_offsets[None, :],
        o_k_norm,
        mask=append_token[:, None],
    )
    tl.store(
        state_v_attn_row
        + append_pos[:, None] * VALUE_DIM
        + value_offsets[None, :],
        o_v,
        mask=append_token[:, None],
    )
    tl.store(state_count_row + append_pos, 1.0, mask=append_token)


@triton.jit
def _init_state_normcache_kernel(
    state_k,
    state_v,
    state_k_attn,
    state_v_attn,
    state_count,
    ln_weight,
    ln_bias,
    INITIAL_STATE_LEN: tl.constexpr,
    MAX_STATE_LEN: tl.constexpr,
    STATE_CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    LN_EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    state_offsets = tl.arange(0, STATE_CHUNK)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    state_idx = chunk * STATE_CHUNK + state_offsets
    valid = state_idx < INITIAL_STATE_LEN

    state_k_row = state_k + row * MAX_STATE_LEN * HEAD_DIM
    state_v_row = state_v + row * MAX_STATE_LEN * VALUE_DIM
    state_k_attn_row = state_k_attn + row * MAX_STATE_LEN * HEAD_DIM
    state_v_attn_row = state_v_attn + row * MAX_STATE_LEN * VALUE_DIM
    state_count_row = state_count + row * MAX_STATE_LEN

    raw_k = tl.load(
        state_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    raw_v = tl.load(
        state_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    ln_w = tl.load(ln_weight + key_offsets).to(tl.float32)
    ln_b = tl.load(ln_bias + key_offsets).to(tl.float32)
    k_mean = tl.sum(raw_k, axis=1) / HEAD_DIM
    k_centered = raw_k - k_mean[:, None]
    k_var = tl.sum(k_centered * k_centered, axis=1) / HEAD_DIM
    k_norm = k_centered * tl.rsqrt(k_var[:, None] + LN_EPS)
    k_norm = (k_norm * ln_w[None, :] + ln_b[None, :]).to(tl.bfloat16)
    v_attn = raw_v.to(tl.bfloat16)
    tl.store(
        state_k_attn_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        k_norm,
        mask=valid[:, None],
    )
    tl.store(
        state_v_attn_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        v_attn,
        mask=valid[:, None],
    )
    tl.store(state_count_row + state_idx, 1.0, mask=valid)


@triton.jit
def _apply_fp16_delta_normcache_rounded_kernel(
    state_k,
    state_v,
    state_k_attn,
    state_v_attn,
    state_count,
    delta_k,
    delta_v,
    touched_slots,
    after_by_macro,
    macro_id,
    ln_weight,
    ln_bias,
    MAX_STATE_LEN: tl.constexpr,
    STATE_CHUNK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    LN_EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1).to(tl.int64)
    state_offsets = tl.arange(0, STATE_CHUNK)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)

    state_after = tl.load(after_by_macro + macro_id)
    state_idx = chunk * STATE_CHUNK + state_offsets
    valid_state = state_idx < state_after

    state_k_row = state_k + row * MAX_STATE_LEN * HEAD_DIM
    state_v_row = state_v + row * MAX_STATE_LEN * VALUE_DIM
    state_k_attn_row = state_k_attn + row * MAX_STATE_LEN * HEAD_DIM
    state_v_attn_row = state_v_attn + row * MAX_STATE_LEN * VALUE_DIM
    state_count_row = state_count + row * MAX_STATE_LEN
    delta_k_row = delta_k + row * MAX_STATE_LEN * HEAD_DIM
    delta_v_row = delta_v + row * MAX_STATE_LEN * VALUE_DIM
    touched_row = touched_slots + row * MAX_STATE_LEN

    touched = tl.load(touched_row + state_idx, mask=valid_state, other=0) != 0
    update_mask = valid_state & touched
    old_k = tl.load(
        state_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=update_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    old_v = tl.load(
        state_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        mask=update_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    add_k = tl.load(
        delta_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=update_mask[:, None],
        other=0.0,
    )
    add_v = tl.load(
        delta_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        mask=update_mask[:, None],
        other=0.0,
    )
    next_k = (old_k + add_k.to(tl.bfloat16)).to(tl.bfloat16)
    next_v = (old_v + add_v.to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(
        state_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        next_k,
        mask=update_mask[:, None],
    )
    tl.store(
        state_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        next_v,
        mask=update_mask[:, None],
    )
    next_k_f = next_k.to(tl.float32)
    next_v_f = next_v.to(tl.float32)
    ln_w = tl.load(ln_weight + key_offsets).to(tl.float32)
    ln_b = tl.load(ln_bias + key_offsets).to(tl.float32)
    k_mean = tl.sum(next_k_f, axis=1) / HEAD_DIM
    k_centered = next_k_f - k_mean[:, None]
    k_var = tl.sum(k_centered * k_centered, axis=1) / HEAD_DIM
    k_norm = k_centered * tl.rsqrt(k_var[:, None] + LN_EPS)
    k_norm = (k_norm * ln_w[None, :] + ln_b[None, :]).to(tl.bfloat16)
    count = tl.load(state_count_row + state_idx, mask=update_mask, other=1.0)
    v_attn = (next_v_f / count[:, None]).to(tl.bfloat16)
    tl.store(
        state_k_attn_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        k_norm,
        mask=update_mask[:, None],
    )
    tl.store(
        state_v_attn_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        v_attn,
        mask=update_mask[:, None],
    )
    tl.store(
        delta_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        0.0,
        mask=update_mask[:, None],
    )
    tl.store(
        delta_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        0.0,
        mask=update_mask[:, None],
    )
    tl.store(touched_row + state_idx, 0, mask=update_mask)


@triton.jit
def _live_state_dkdv_to_raw_grad_kernel(
    q,
    state_k_attn,
    state_v_attn,
    state_k,
    state_v,
    state_count,
    dout,
    lse,
    delta,
    d_state_k,
    d_state_v,
    d_state_count,
    d_ln_weight,
    d_ln_bias,
    d_state_temperature,
    d_state_temperature_partials,
    active_state_len_by_macro,
    macro_id,
    ln_weight,
    state_temperature,
    MAX_STATE_LEN: tl.constexpr,
    STATE_CHUNK: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    MACRO_BLOCKS: tl.constexpr,
    MACRO_BLOCK: tl.constexpr,
    ATTN_BLOCK: tl.constexpr,
    ATTN_BLOCKS_PER_MACRO: tl.constexpr,
    STATE_CHUNKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    LN_EPS: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    SCALE: tl.constexpr,
    Q_HEAD_LOOP_UNROLL: tl.constexpr,
    COMPUTE_TEMP_GRAD: tl.constexpr,
    STORE_TEMP_GRAD: tl.constexpr,
    WRITE_TEMP_PARTIAL: tl.constexpr,
    BATCH_TEMP_GRAD: tl.constexpr,
    ROUND_AOT_OUTPUT_GRADS: tl.constexpr,
):
    kv_row = tl.program_id(0).to(tl.int64)
    chunk_id = tl.program_id(1).to(tl.int64)
    group_size: tl.constexpr = Q_HEADS // KV_HEADS
    batch_id = kv_row // KV_HEADS
    kv_head = kv_row - batch_id * KV_HEADS

    state_offsets = tl.arange(0, STATE_CHUNK)
    token_offsets = tl.arange(0, ATTN_BLOCK)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    total_len: tl.constexpr = MACRO_BLOCKS * MACRO_BLOCK
    state_idx = chunk_id * STATE_CHUNK + state_offsets
    active_state_len = tl.load(active_state_len_by_macro + macro_id)
    valid_state = state_idx < active_state_len

    state_k_attn_base = state_k_attn + kv_row * MAX_STATE_LEN * HEAD_DIM
    state_v_attn_base = state_v_attn + kv_row * MAX_STATE_LEN * VALUE_DIM
    raw_state_k_base = state_k + kv_row * MAX_STATE_LEN * HEAD_DIM
    raw_state_count_base = state_count + kv_row * MAX_STATE_LEN
    d_state_k_row = d_state_k + kv_row * MAX_STATE_LEN * HEAD_DIM
    d_state_v_row = d_state_v + kv_row * MAX_STATE_LEN * VALUE_DIM

    k_chunk = tl.load(
        state_k_attn_base + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=valid_state[:, None],
        other=0.0,
    )
    v_chunk = tl.load(
        state_v_attn_base + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        mask=valid_state[:, None],
        other=0.0,
    )
    dk_attn = tl.zeros((STATE_CHUNK, HEAD_DIM), tl.float32)
    dv_attn = tl.zeros((STATE_CHUNK, VALUE_DIM), tl.float32)

    for group_i in tl.range(
        0, group_size, 1, num_stages=1, loop_unroll_factor=Q_HEAD_LOOP_UNROLL
    ):
        q_head = kv_head * group_size + group_i
        q_row = batch_id * Q_HEADS + q_head
        q_row_ptr = q + q_row * total_len * HEAD_DIM
        dout_row = dout + q_row * total_len * VALUE_DIM
        lse_row = lse + q_row * total_len
        delta_row = delta + q_row * total_len
        state_temp = tl.load(state_temperature + q_head).to(tl.float32)
        k_eff = (k_chunk.to(tl.float32) * state_temp).to(tl.bfloat16)
        if COMPUTE_TEMP_GRAD:
            temp_grad = tl.full((), 0.0, tl.float32)

        for local_block in tl.range(0, ATTN_BLOCKS_PER_MACRO, 1, num_stages=1):
            q_idx = macro_id * MACRO_BLOCK + local_block * ATTN_BLOCK + token_offsets
            q_block = tl.load(
                q_row_ptr + q_idx[:, None] * HEAD_DIM + key_offsets[None, :]
            )
            do = tl.load(
                dout_row + q_idx[:, None] * VALUE_DIM + value_offsets[None, :]
            ).to(tl.float32)
            do_bf16 = do.to(tl.bfloat16)
            lse_i = tl.load(lse_row + q_idx)
            delta_i = tl.load(delta_row + q_idx)
            scores = tl.dot(q_block, tl.trans(k_eff), out_dtype=tl.float32) * SCALE_LOG2
            scores = tl.where(valid_state[None, :], scores, -float("inf"))
            p = tl.exp2(scores - lse_i[:, None])
            p = tl.where(valid_state[None, :], p, 0.0)
            dp = tl.dot(do_bf16, tl.trans(v_chunk), out_dtype=tl.float32)
            ds = p * (dp - delta_i[:, None])
            dk_eff = tl.dot(
                tl.trans(ds.to(tl.bfloat16)), q_block, out_dtype=tl.float32
            ) * SCALE
            if ROUND_AOT_OUTPUT_GRADS:
                dk_attn += dk_eff
            else:
                dk_attn += dk_eff * state_temp
            if COMPUTE_TEMP_GRAD:
                temp_grad += tl.sum(
                    tl.sum(
                        tl.where(
                            valid_state[:, None], dk_eff * k_chunk.to(tl.float32), 0.0
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
            p_bf16 = p.to(tl.bfloat16)
            dv_block = tl.trans(
                tl.dot(tl.trans(do_bf16), p_bf16, out_dtype=tl.float32)
            )
            dv_attn += dv_block

        if ROUND_AOT_OUTPUT_GRADS:
            # AOT materializes BF16 dK/dV before the temperature and state
            # normalization VJPs. Preserve those rounding boundaries.
            d_effective_k = dk_attn.to(tl.bfloat16)
            state_temp_bf16 = state_temp.to(tl.bfloat16)
            dk_attn = (d_effective_k * state_temp_bf16).to(tl.bfloat16).to(tl.float32)
            dv_attn = dv_attn.to(tl.bfloat16).to(tl.float32)

        if STORE_TEMP_GRAD:
            temp_idx = tl.where(BATCH_TEMP_GRAD, batch_id * Q_HEADS + q_head, q_head)
            tl.atomic_add(d_state_temperature + temp_idx, temp_grad, sem="relaxed")
        if WRITE_TEMP_PARTIAL:
            partial_idx = (
                ((kv_row * MACRO_BLOCKS + macro_id) * STATE_CHUNKS + chunk_id)
                * group_size
                + group_i
            )
            tl.store(d_state_temperature_partials + partial_idx, temp_grad)

    raw_k = tl.load(
        raw_state_k_base + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=valid_state[:, None],
        other=0.0,
    ).to(tl.float32)
    ln_w = tl.load(ln_weight + key_offsets).to(tl.float32)
    mean = tl.sum(raw_k, axis=1) / HEAD_DIM
    centered = raw_k - mean[:, None]
    var = tl.sum(centered * centered, axis=1) / HEAD_DIM
    rstd = tl.rsqrt(var + LN_EPS)
    xhat = centered * rstd[:, None]
    gw = dk_attn * ln_w[None, :]
    mean_gw = tl.sum(gw, axis=1) / HEAD_DIM
    mean_gw_xhat = tl.sum(gw * xhat, axis=1) / HEAD_DIM
    grad_raw_k = (gw - mean_gw[:, None] - xhat * mean_gw_xhat[:, None]) * rstd[:, None]
    old_dk = tl.load(
        d_state_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=valid_state[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        d_state_k_row + state_idx[:, None] * HEAD_DIM + key_offsets[None, :],
        old_dk + grad_raw_k,
        mask=valid_state[:, None],
    )
    part_w = tl.sum(tl.where(valid_state[:, None], dk_attn * xhat, 0.0), axis=0)
    part_b = tl.sum(tl.where(valid_state[:, None], dk_attn, 0.0), axis=0)
    tl.atomic_add(d_ln_weight + key_offsets, part_w, sem="relaxed")
    tl.atomic_add(d_ln_bias + key_offsets, part_b, sem="relaxed")

    # Counts are discrete routing metadata and have no adjoint.
    count = tl.load(raw_state_count_base + state_idx, mask=valid_state, other=1.0)
    grad_raw_v = dv_attn / count[:, None]
    old_dv = tl.load(
        d_state_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        mask=valid_state[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        d_state_v_row + state_idx[:, None] * VALUE_DIM + value_offsets[None, :],
        old_dv + grad_raw_v,
        mask=valid_state[:, None],
    )


@triton.jit
def _restore_refresh_from_undo_routes_kernel(
    state_k,
    state_v,
    state_k_attn,
    state_v_attn,
    state_count,
    undo_k_by_token,
    undo_v_by_token,
    append_pos_by_token,
    best_idx_by_token,
    before_by_macro,
    after_by_macro,
    macro_id,
    ln_weight,
    ln_bias,
    MAX_STATE_LEN: tl.constexpr,
    MACRO_BLOCKS: tl.constexpr,
    MACRO_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    LN_EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token_group = tl.program_id(1).to(tl.int64)
    token_offsets = token_group * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    key_offsets = tl.arange(0, HEAD_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)
    route_offset = macro_id * MACRO_BLOCK + token_offsets

    state_before = tl.load(before_by_macro + macro_id)
    state_after = tl.load(after_by_macro + macro_id)
    append_pos_row = append_pos_by_token + row * MACRO_BLOCKS * MACRO_BLOCK
    best_idx_row = best_idx_by_token + row * MACRO_BLOCKS * MACRO_BLOCK
    append_pos = tl.load(append_pos_row + route_offset)
    best_idx = tl.load(best_idx_row + route_offset)

    append_mask = (append_pos >= 0) & (append_pos < state_after)
    restore_mask = (append_pos < 0) & (best_idx >= 0) & (best_idx < state_before)
    append_i64 = append_pos.to(tl.int64)
    best_i64 = best_idx.to(tl.int64)

    state_k_row = state_k + row * MAX_STATE_LEN * HEAD_DIM
    state_v_row = state_v + row * MAX_STATE_LEN * VALUE_DIM
    state_k_attn_row = state_k_attn + row * MAX_STATE_LEN * HEAD_DIM
    state_v_attn_row = state_v_attn + row * MAX_STATE_LEN * VALUE_DIM
    state_count_row = state_count + row * MAX_STATE_LEN
    undo_k_row = undo_k_by_token + row * MACRO_BLOCKS * MACRO_BLOCK * HEAD_DIM
    undo_v_row = undo_v_by_token + row * MACRO_BLOCKS * MACRO_BLOCK * VALUE_DIM

    old_k = tl.load(
        undo_k_row + route_offset[:, None] * HEAD_DIM + key_offsets[None, :],
        mask=restore_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    old_v = tl.load(
        undo_v_row + route_offset[:, None] * VALUE_DIM + value_offsets[None, :],
        mask=restore_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    tl.store(
        state_k_row + best_i64[:, None] * HEAD_DIM + key_offsets[None, :],
        old_k,
        mask=restore_mask[:, None],
    )
    tl.store(
        state_v_row + best_i64[:, None] * VALUE_DIM + value_offsets[None, :],
        old_v,
        mask=restore_mask[:, None],
    )

    ln_w = tl.load(ln_weight + key_offsets).to(tl.float32)
    ln_b = tl.load(ln_bias + key_offsets).to(tl.float32)
    k_mean = tl.sum(old_k, axis=1) / HEAD_DIM
    k_centered = old_k - k_mean[:, None]
    k_var = tl.sum(k_centered * k_centered, axis=1) / HEAD_DIM
    k_norm = k_centered * tl.rsqrt(k_var[:, None] + LN_EPS)
    k_norm = (k_norm * ln_w[None, :] + ln_b[None, :]).to(tl.bfloat16)

    count = tl.load(state_count_row + best_i64, mask=restore_mask, other=1.0)
    v_attn = (old_v / count[:, None]).to(tl.bfloat16)
    tl.store(
        state_k_attn_row + best_i64[:, None] * HEAD_DIM + key_offsets[None, :],
        k_norm,
        mask=restore_mask[:, None],
    )
    tl.store(
        state_v_attn_row + best_i64[:, None] * VALUE_DIM + value_offsets[None, :],
        v_attn,
        mask=restore_mask[:, None],
    )

    tl.store(
        state_k_row + append_i64[:, None] * HEAD_DIM + key_offsets[None, :],
        0.0,
        mask=append_mask[:, None],
    )
    tl.store(
        state_v_row + append_i64[:, None] * VALUE_DIM + value_offsets[None, :],
        0.0,
        mask=append_mask[:, None],
    )
    tl.store(
        state_k_attn_row + append_i64[:, None] * HEAD_DIM + key_offsets[None, :],
        0.0,
        mask=append_mask[:, None],
    )
    tl.store(
        state_v_attn_row + append_i64[:, None] * VALUE_DIM + value_offsets[None, :],
        0.0,
        mask=append_mask[:, None],
    )
    tl.store(state_count_row + append_i64, 0.0, mask=append_mask)
