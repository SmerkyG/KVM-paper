"""Forward-only Triton kernels for LOD state maintenance.

The kernels mirror the efficient parts of the training KVM implementation:
merge tokens accumulate into persistent FP32 deltas and each touched BF16
state slot is rounded only once.  Query routing scans the compact state in
tiles and retains only the top eight slots instead of materializing the full
query-by-state score tensor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _launch_kwargs(num_warps: int) -> dict[str, int]:
    kwargs = {"num_warps": num_warps, "num_stages": 1}
    if torch.version.hip is not None:
        kwargs["waves_per_eu"] = 1
    return kwargs


@triton.jit(
    do_not_specialize=["QUERY_LEN"],
    do_not_specialize_on_alignment=[
        "Q_BATCH_STRIDE",
        "Q_HEAD_STRIDE",
        "Q_TOKEN_STRIDE",
        "SINK_K_BATCH_STRIDE",
        "SINK_K_HEAD_STRIDE",
        "SINK_K_TOKEN_STRIDE",
        "SINK_V_BATCH_STRIDE",
        "SINK_V_HEAD_STRIDE",
        "SINK_V_TOKEN_STRIDE",
        "PRIMARY_BATCH_STRIDE",
        "PRIMARY_HEAD_STRIDE",
        "PRIMARY_TOKEN_STRIDE",
        "PRIMARY_LSE_BATCH_STRIDE",
        "PRIMARY_LSE_HEAD_STRIDE",
        "PRIMARY_LSE_TOKEN_STRIDE",
        "SECONDARY_BATCH_STRIDE",
        "SECONDARY_HEAD_STRIDE",
        "SECONDARY_TOKEN_STRIDE",
        "SECONDARY_LSE_BATCH_STRIDE",
        "SECONDARY_LSE_HEAD_STRIDE",
        "SECONDARY_LSE_TOKEN_STRIDE",
        "TERTIARY_BATCH_STRIDE",
        "TERTIARY_HEAD_STRIDE",
        "TERTIARY_TOKEN_STRIDE",
        "TERTIARY_LSE_BATCH_STRIDE",
        "TERTIARY_LSE_HEAD_STRIDE",
        "TERTIARY_LSE_TOKEN_STRIDE",
        "OUTPUT_BATCH_STRIDE",
        "OUTPUT_HEAD_STRIDE",
        "OUTPUT_TOKEN_STRIDE",
        "QUERY_LEN",
    ],
)
def _merge_attention_branches_with_sink_kernel(
    q,
    sink_k,
    sink_v,
    primary_out,
    primary_lse,
    secondary_out,
    secondary_lse,
    tertiary_out,
    tertiary_lse,
    output,
    Q_BATCH_STRIDE,
    Q_HEAD_STRIDE,
    Q_TOKEN_STRIDE,
    SINK_K_BATCH_STRIDE,
    SINK_K_HEAD_STRIDE,
    SINK_K_TOKEN_STRIDE,
    SINK_V_BATCH_STRIDE,
    SINK_V_HEAD_STRIDE,
    SINK_V_TOKEN_STRIDE,
    PRIMARY_BATCH_STRIDE,
    PRIMARY_HEAD_STRIDE,
    PRIMARY_TOKEN_STRIDE,
    PRIMARY_LSE_BATCH_STRIDE,
    PRIMARY_LSE_HEAD_STRIDE,
    PRIMARY_LSE_TOKEN_STRIDE,
    SECONDARY_BATCH_STRIDE,
    SECONDARY_HEAD_STRIDE,
    SECONDARY_TOKEN_STRIDE,
    SECONDARY_LSE_BATCH_STRIDE,
    SECONDARY_LSE_HEAD_STRIDE,
    SECONDARY_LSE_TOKEN_STRIDE,
    TERTIARY_BATCH_STRIDE,
    TERTIARY_HEAD_STRIDE,
    TERTIARY_TOKEN_STRIDE,
    TERTIARY_LSE_BATCH_STRIDE,
    TERTIARY_LSE_HEAD_STRIDE,
    TERTIARY_LSE_TOKEN_STRIDE,
    OUTPUT_BATCH_STRIDE,
    OUTPUT_HEAD_STRIDE,
    OUTPUT_TOKEN_STRIDE,
    QUERY_LEN,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    SINK_LEN: tl.constexpr,
    INCLUDE_SECONDARY: tl.constexpr,
    INCLUDE_TERTIARY: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Merge materialized attention branches and an exact side sink once."""
    batch = tl.program_id(0).to(tl.int64)
    query_head = tl.program_id(1).to(tl.int64)
    query = tl.program_id(2).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    query_valid = query < QUERY_LEN
    kv_head = query_head // KV_GROUP_SIZE
    dim = tl.arange(0, BLOCK_DIM)
    dim_valid = dim < HEAD_DIM

    query_value = tl.load(
        q
        + batch * Q_BATCH_STRIDE
        + query_head * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    if SINK_LEN == 1:
        key = tl.load(
            sink_k + batch * SINK_K_BATCH_STRIDE + kv_head * SINK_K_HEAD_STRIDE + dim,
            mask=dim_valid,
            other=0.0,
        ).to(tl.float32)
        sink_output = tl.load(
            sink_v + batch * SINK_V_BATCH_STRIDE + kv_head * SINK_V_HEAD_STRIDE + dim,
            mask=dim_valid,
            other=0.0,
        ).to(tl.float32)
        sink_lse = tl.sum(query_value * key[None, :], axis=1) * SCALE
    else:
        sink_maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        sink_denominator = tl.zeros((BLOCK_M,), tl.float32)
        sink_accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), tl.float32)
        for sink_index in tl.static_range(0, SINK_LEN):
            key = tl.load(
                sink_k
                + batch * SINK_K_BATCH_STRIDE
                + kv_head * SINK_K_HEAD_STRIDE
                + sink_index * SINK_K_TOKEN_STRIDE
                + dim,
                mask=dim_valid,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                sink_v
                + batch * SINK_V_BATCH_STRIDE
                + kv_head * SINK_V_HEAD_STRIDE
                + sink_index * SINK_V_TOKEN_STRIDE
                + dim,
                mask=dim_valid,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(query_value * key[None, :], axis=1) * SCALE
            new_maximum = tl.maximum(sink_maximum, score)
            old_weight = tl.exp(sink_maximum - new_maximum)
            new_weight = tl.exp(score - new_maximum)
            sink_denominator = sink_denominator * old_weight + new_weight
            sink_accumulator = (
                sink_accumulator * old_weight[:, None]
                + value[None, :] * new_weight[:, None]
            )
            sink_maximum = new_maximum
        sink_lse = sink_maximum + tl.log(sink_denominator)
        sink_output = sink_accumulator / sink_denominator[:, None]
    sink_lse = tl.where(query_valid, sink_lse, -float("inf"))

    primary_score = tl.load(
        primary_lse
        + batch * PRIMARY_LSE_BATCH_STRIDE
        + query_head * PRIMARY_LSE_HEAD_STRIDE
        + query * PRIMARY_LSE_TOKEN_STRIDE,
        mask=query_valid,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.maximum(primary_score, sink_lse)
    secondary_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    tertiary_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    if INCLUDE_SECONDARY:
        secondary_score = tl.load(
            secondary_lse
            + batch * SECONDARY_LSE_BATCH_STRIDE
            + query_head * SECONDARY_LSE_HEAD_STRIDE
            + query * SECONDARY_LSE_TOKEN_STRIDE,
            mask=query_valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.maximum(maximum, secondary_score)
    if INCLUDE_TERTIARY:
        tertiary_score = tl.load(
            tertiary_lse
            + batch * TERTIARY_LSE_BATCH_STRIDE
            + query_head * TERTIARY_LSE_HEAD_STRIDE
            + query * TERTIARY_LSE_TOKEN_STRIDE,
            mask=query_valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.maximum(maximum, tertiary_score)

    primary_weight = tl.exp(primary_score - maximum)
    sink_weight = tl.exp(sink_lse - maximum)
    denominator = primary_weight + sink_weight
    primary_value = tl.load(
        primary_out
        + batch * PRIMARY_BATCH_STRIDE
        + query_head * PRIMARY_HEAD_STRIDE
        + query[:, None] * PRIMARY_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = primary_weight[:, None] * primary_value
    if SINK_LEN == 1:
        numerator += sink_weight[:, None] * sink_output[None, :]
    else:
        numerator += sink_weight[:, None] * sink_output
    if INCLUDE_SECONDARY:
        secondary_weight = tl.exp(secondary_score - maximum)
        secondary_value = tl.load(
            secondary_out
            + batch * SECONDARY_BATCH_STRIDE
            + query_head * SECONDARY_HEAD_STRIDE
            + query[:, None] * SECONDARY_TOKEN_STRIDE
            + dim[None, :],
            mask=query_valid[:, None] & dim_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        denominator += secondary_weight
        numerator += secondary_weight[:, None] * secondary_value
    if INCLUDE_TERTIARY:
        tertiary_weight = tl.exp(tertiary_score - maximum)
        tertiary_value = tl.load(
            tertiary_out
            + batch * TERTIARY_BATCH_STRIDE
            + query_head * TERTIARY_HEAD_STRIDE
            + query[:, None] * TERTIARY_TOKEN_STRIDE
            + dim[None, :],
            mask=query_valid[:, None] & dim_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        denominator += tertiary_weight
        numerator += tertiary_weight[:, None] * tertiary_value
    output_offset = (
        batch * OUTPUT_BATCH_STRIDE
        + query_head * OUTPUT_HEAD_STRIDE
        + query[:, None] * OUTPUT_TOKEN_STRIDE
        + dim[None, :]
    )
    tl.store(
        output + output_offset,
        numerator / denominator[:, None],
        mask=query_valid[:, None] & dim_valid[None, :],
    )


@triton.jit(
    do_not_specialize=["slot_count", "state_len"],
    do_not_specialize_on_alignment=[
        "STATE_BATCH_STRIDE",
        "STATE_HEAD_STRIDE",
        "COUNT_BATCH_STRIDE",
        "COUNT_HEAD_STRIDE",
        "KEY_NORM_BATCH_STRIDE",
        "KEY_NORM_HEAD_STRIDE",
        "OUTPUT_BATCH_STRIDE",
        "OUTPUT_HEAD_STRIDE",
        "SCALE_BATCH_STRIDE",
        "SCALE_HEAD_STRIDE",
        "INDEX_BATCH_STRIDE",
        "INDEX_HEAD_STRIDE",
        "slot_count",
        "state_len",
    ],
)
def _prepare_state_clustering_keys_kernel(
    state_k,
    counts,
    key_norm_sums,
    route_k,
    append_k,
    select_scale,
    slot_indices,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    KEY_NORM_BATCH_STRIDE,
    KEY_NORM_HEAD_STRIDE,
    KEY_NORM_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE,
    OUTPUT_HEAD_STRIDE,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    SCALE_BATCH_STRIDE,
    SCALE_HEAD_STRIDE,
    SCALE_TOKEN_STRIDE: tl.constexpr,
    INDEX_BATCH_STRIDE,
    INDEX_HEAD_STRIDE,
    INDEX_TOKEN_STRIDE: tl.constexpr,
    slot_count,
    state_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    COHERENCE: tl.constexpr,
    WRITE_ROUTE: tl.constexpr,
    WRITE_APPEND: tl.constexpr,
    WRITE_SCALE: tl.constexpr,
    INDEXED: tl.constexpr,
):
    """Prepare centroid geometry once per state update, not once per leaf tile."""
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    item = tl.program_id(2).to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = item < slot_count
    if INDEXED:
        slot = tl.load(
            slot_indices
            + batch * INDEX_BATCH_STRIDE
            + head * INDEX_HEAD_STRIDE
            + item * INDEX_TOKEN_STRIDE,
            mask=valid,
            other=0,
        ).to(tl.int64)
    else:
        slot = item
    valid &= slot < state_len
    dim = tl.arange(0, BLOCK_D)
    dim_valid = dim < HEAD_DIM
    count = tl.load(
        counts
        + batch * COUNT_BATCH_STRIDE
        + head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=valid,
        other=1.0,
    )
    valid &= count > 0.5
    key = tl.load(
        state_k
        + batch * STATE_BATCH_STRIDE
        + head * STATE_HEAD_STRIDE
        + slot[:, None] * STATE_TOKEN_STRIDE
        + dim[None, :],
        mask=valid[:, None] & dim_valid[None, :],
        other=0.0,
    )
    mean_key = (key / count.to(key.dtype)[:, None]).to(key.dtype)
    output_offset = (
        batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + slot[:, None] * OUTPUT_TOKEN_STRIDE
        + dim[None, :]
    )
    if WRITE_APPEND or WRITE_SCALE:
        centroid_rms = tl.sqrt(
            tl.sum(mean_key.to(tl.float32) * mean_key.to(tl.float32), axis=1) / HEAD_DIM
        )
        normalized = (
            mean_key.to(tl.float32) / tl.maximum(centroid_rms[:, None], 1e-12)
        ).to(mean_key.dtype)
        if WRITE_APPEND:
            tl.store(
                append_k + output_offset,
                normalized,
                mask=valid[:, None] & dim_valid[None, :],
            )
    if COHERENCE:
        norm_sum = tl.load(
            key_norm_sums
            + batch * KEY_NORM_BATCH_STRIDE
            + head * KEY_NORM_HEAD_STRIDE
            + slot * KEY_NORM_TOKEN_STRIDE,
            mask=valid,
            other=1.0,
        ).to(tl.float32)
        mean_norm = norm_sum / tl.maximum(count.to(tl.float32), 1.0)
        if WRITE_ROUTE:
            routed = (
                mean_key.to(tl.float32) / tl.maximum(mean_norm[:, None], 1e-12)
            ).to(mean_key.dtype)
            tl.store(
                route_k + output_offset,
                routed,
                mask=valid[:, None] & dim_valid[None, :],
            )
        if WRITE_SCALE:
            scale_offset = (
                batch * SCALE_BATCH_STRIDE
                + head * SCALE_HEAD_STRIDE
                + slot * SCALE_TOKEN_STRIDE
            )
            tl.store(
                select_scale + scale_offset,
                centroid_rms / tl.maximum(mean_norm, 1e-12),
                mask=valid,
            )


@triton.jit(
    do_not_specialize=["token_len"],
    do_not_specialize_on_alignment=[
        "KEY_BATCH_STRIDE",
        "KEY_HEAD_STRIDE",
        "OUTPUT_BATCH_STRIDE",
        "OUTPUT_HEAD_STRIDE",
        "token_len",
    ],
)
def _constituent_rms_kernel(
    key,
    output,
    KEY_BATCH_STRIDE,
    KEY_HEAD_STRIDE,
    KEY_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE,
    OUTPUT_HEAD_STRIDE,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    token_len,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    token = tl.program_id(2).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    dim = tl.arange(0, BLOCK_D)
    valid = token < token_len
    value = tl.load(
        key
        + batch * KEY_BATCH_STRIDE
        + head * KEY_HEAD_STRIDE
        + token[:, None] * KEY_TOKEN_STRIDE
        + dim[None, :],
        mask=valid[:, None] & (dim[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    rms = tl.sqrt(tl.sum(value * value, axis=1) / HEAD_DIM)
    tl.store(
        output
        + batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + token * OUTPUT_TOKEN_STRIDE,
        rms,
        mask=valid,
    )


@triton.jit(
    do_not_specialize=["overflow_len", "state_len"],
    do_not_specialize_on_alignment=[
        "SCORE_BATCH_STRIDE",
        "SCORE_HEAD_STRIDE",
        "SCORE_TOKEN_STRIDE",
        "SCALE_BATCH_STRIDE",
        "SCALE_HEAD_STRIDE",
        "COUNT_BATCH_STRIDE",
        "COUNT_HEAD_STRIDE",
        "OUTPUT_BATCH_STRIDE",
        "OUTPUT_HEAD_STRIDE",
        "overflow_len",
        "state_len",
    ],
)
def _scaled_coherence_maxsim_kernel(
    append_scores,
    select_scale,
    counts,
    route_scores,
    route_indices,
    select_scores,
    SCORE_BATCH_STRIDE,
    SCORE_HEAD_STRIDE,
    SCORE_TOKEN_STRIDE,
    SCORE_STATE_STRIDE: tl.constexpr,
    SCALE_BATCH_STRIDE,
    SCALE_HEAD_STRIDE,
    SCALE_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE,
    OUTPUT_HEAD_STRIDE,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    overflow_len,
    state_len,
    SINK_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    token = tl.program_id(2).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_valid = token < overflow_len
    best_select_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    best_route_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    best_route_index = tl.full((BLOCK_M,), -1, tl.int32)
    for state_begin in tl.range(0, state_len, BLOCK_N, num_stages=1):
        slot = state_begin + tl.arange(0, BLOCK_N)
        slot_valid = slot < state_len
        count = tl.load(
            counts
            + batch * COUNT_BATCH_STRIDE
            + head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE,
            mask=slot_valid,
            other=0.0,
        )
        slot_valid &= count > 0.5
        score = tl.load(
            append_scores
            + batch * SCORE_BATCH_STRIDE
            + head * SCORE_HEAD_STRIDE
            + token[:, None] * SCORE_TOKEN_STRIDE
            + slot[None, :] * SCORE_STATE_STRIDE,
            mask=token_valid[:, None] & slot_valid[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        scale = tl.load(
            select_scale
            + batch * SCALE_BATCH_STRIDE
            + head * SCALE_HEAD_STRIDE
            + slot * SCALE_TOKEN_STRIDE,
            mask=slot_valid,
            other=1.0,
        ).to(tl.float32)
        best_select_score = tl.maximum(best_select_score, tl.max(score, axis=1))
        # append_key = mean_key / rms(mean_key), while the coherence route
        # key is mean_key / mean(rms(constituent_key)).  They differ only by
        # this per-centroid ratio, so one append-key GEMM supplies both scans.
        candidate = tl.where(
            token_valid[:, None] & slot_valid[None, :] & (slot[None, :] >= SINK_LEN),
            score * scale[None, :],
            -float("inf"),
        )
        local_score = tl.max(candidate, axis=1)
        local_index = tl.min(
            tl.where(candidate == local_score[:, None], slot[None, :], state_len),
            axis=1,
        ).to(tl.int32)
        take_local = local_score > best_route_score
        best_route_score = tl.where(take_local, local_score, best_route_score)
        best_route_index = tl.where(take_local, local_index, best_route_index)
    output_offset = (
        batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + token * OUTPUT_TOKEN_STRIDE
    )
    tl.store(route_scores + output_offset, best_route_score, mask=token_valid)
    tl.store(route_indices + output_offset, best_route_index, mask=token_valid)
    tl.store(select_scores + output_offset, best_select_score, mask=token_valid)


@triton.jit(
    do_not_specialize=["query_len", "state_len", "local_len", "local_offset"],
    do_not_specialize_on_alignment=[
        "Q_BATCH_STRIDE",
        "Q_HEAD_STRIDE",
        "LOGIT_BATCH_STRIDE",
        "LOGIT_HEAD_STRIDE",
        "LOGIT_QUERY_STRIDE",
        "LOCAL_K_BATCH_STRIDE",
        "LOCAL_K_HEAD_STRIDE",
        "LOCAL_V_BATCH_STRIDE",
        "LOCAL_V_HEAD_STRIDE",
        "TOP_BATCH_STRIDE",
        "TOP_HEAD_STRIDE",
        "query_len",
        "state_len",
        "local_len",
        "local_offset",
    ],
)
def _route_logits_coarse_attention_kernel(
    q,
    route_logits,
    route_logit_scale,
    state_v,
    state_v_scales,
    counts,
    local_k,
    local_v,
    top_slots,
    output,
    lse,
    Q_BATCH_STRIDE,
    Q_HEAD_STRIDE,
    Q_TOKEN_STRIDE: tl.constexpr,
    LOGIT_BATCH_STRIDE,
    LOGIT_HEAD_STRIDE,
    LOGIT_QUERY_STRIDE,
    LOGIT_STATE_STRIDE: tl.constexpr,
    STATE_V_BATCH_STRIDE: tl.constexpr,
    STATE_V_HEAD_STRIDE: tl.constexpr,
    STATE_V_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    LOCAL_K_BATCH_STRIDE,
    LOCAL_K_HEAD_STRIDE,
    LOCAL_K_TOKEN_STRIDE: tl.constexpr,
    LOCAL_V_BATCH_STRIDE,
    LOCAL_V_HEAD_STRIDE,
    LOCAL_V_TOKEN_STRIDE: tl.constexpr,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    TOP_QUERY_STRIDE: tl.constexpr,
    query_len,
    state_len,
    local_len,
    local_offset,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    ROW_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    HEAD_TAIL_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    STATE_V_IS_MEAN: tl.constexpr,
    INT8_STATE_PV: tl.constexpr,
    HAS_ROUTE_LOGIT_SCALE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INCLUDE_LOCAL: tl.constexpr,
):
    """Stream the coarse softmax while reusing precomputed route logits."""
    batch = tl.program_id(0).to(tl.int64)
    head_program = tl.program_id(1).to(tl.int64)
    query_block = tl.program_id(2).to(tl.int64)
    row = tl.arange(0, ROW_COUNT)
    if HEAD_MAJOR:
        kv_head = head_program // KV_GROUP_SIZE
        query_head = head_program + tl.zeros((ROW_COUNT,), tl.int64)
        query = query_block * BLOCK_M + row
        group_head_valid = tl.full((ROW_COUNT,), True, tl.int1)
    else:
        kv_head = head_program
        group_head = row // BLOCK_M
        query = query_block * BLOCK_M + row % BLOCK_M
        query_head = kv_head * KV_GROUP_SIZE + group_head
        # ROW_COUNT pads an irregular GQA group to the next power of two so
        # that Triton's row reductions remain legal.  Mask those padding rows
        # instead of falling back to one program per query head.
        group_head_valid = group_head < KV_GROUP_SIZE
    query_valid = (query < query_len) & group_head_valid
    value_dim = tl.arange(0, VALUE_BLOCK_DIM)
    token_offset = tl.arange(0, BLOCK_N)

    if INCLUDE_LOCAL:
        key_dim = tl.arange(0, HEAD_BLOCK_DIM)
        queries = tl.load(
            q
            + batch * Q_BATCH_STRIDE
            + query_head[:, None] * Q_HEAD_STRIDE
            + query[:, None] * Q_TOKEN_STRIDE
            + key_dim[None, :],
            mask=query_valid[:, None] & (key_dim[None, :] < HEAD_DIM),
            other=0.0,
        )
        if HEAD_TAIL_BLOCK_DIM > 0:
            tail_dim = HEAD_BLOCK_DIM + tl.arange(0, HEAD_TAIL_BLOCK_DIM)
            tail_queries = tl.load(
                q
                + batch * Q_BATCH_STRIDE
                + query_head[:, None] * Q_HEAD_STRIDE
                + query[:, None] * Q_TOKEN_STRIDE
                + tail_dim[None, :],
                mask=query_valid[:, None] & (tail_dim[None, :] < HEAD_DIM),
                other=0.0,
            )
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((ROW_COUNT, VALUE_BLOCK_DIM), tl.float32)
    route_scale = tl.full((ROW_COUNT,), 1.0, tl.float32)
    if HAS_ROUTE_LOGIT_SCALE:
        scale_row = ((batch * QUERY_HEADS + query_head) * query_len + query).to(
            tl.int64
        )
        route_scale = tl.load(
            route_logit_scale + scale_row,
            mask=query_valid,
            other=1.0,
        ).to(tl.float32)
    for state_begin in tl.range(0, state_len, BLOCK_N, num_stages=1):
        slot = state_begin + token_offset
        state_valid = slot < state_len
        count = tl.load(
            counts
            + batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + slot * COUNT_TOKEN_STRIDE,
            mask=state_valid,
            other=1.0,
        ).to(tl.float32)
        values = tl.load(
            state_v
            + batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + slot[:, None] * STATE_V_TOKEN_STRIDE
            + value_dim[None, :],
            mask=state_valid[:, None] & (value_dim[None, :] < VALUE_DIM),
            other=0.0,
        )
        if STATE_V_IS_MEAN:
            mean_values = values
        else:
            mean_values = (values.to(tl.float32) / count[:, None]).to(values.dtype)
        scores = tl.load(
            route_logits
            + batch * LOGIT_BATCH_STRIDE
            + query_head[:, None] * LOGIT_HEAD_STRIDE
            + query[:, None] * LOGIT_QUERY_STRIDE
            + slot[None, :] * LOGIT_STATE_STRIDE,
            mask=query_valid[:, None] & state_valid[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        scores = scores * route_scale[:, None] * SCALE + tl.log(count)[None, :]
        routed = tl.zeros((ROW_COUNT, BLOCK_N), dtype=tl.int1)
        for route in tl.static_range(0, ROUTE_COUNT):
            selected = tl.load(
                top_slots
                + batch * TOP_BATCH_STRIDE
                + query_head * TOP_HEAD_STRIDE
                + query * TOP_QUERY_STRIDE
                + route,
                mask=query_valid,
                other=-1,
            )
            routed |= slot[None, :] == selected[:, None]
        valid = query_valid[:, None] & state_valid[None, :] & ~routed
        scores = tl.where(valid, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - new_maximum)
        probabilities = tl.exp(scores - new_maximum[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        if INT8_STATE_PV:
            value_scale = tl.load(
                state_v_scales
                + (
                    (batch * KV_HEADS + kv_head)
                    * ((state_len + BLOCK_N - 1) // BLOCK_N)
                    + state_begin // BLOCK_N
                )
                * VALUE_DIM
                + value_dim,
                mask=value_dim < VALUE_DIM,
                other=0.0,
            ).to(tl.float32)
            probability_scale = tl.maximum(
                tl.max(tl.abs(probabilities), axis=1) / 127.0,
                1.1754943508222875e-38,
            )
            probability_codes = tl.maximum(
                tl.minimum(
                    tl.floor(probabilities / probability_scale[:, None] + 0.5),
                    127.0,
                ),
                -127.0,
            ).to(tl.int8)
            accumulator += (
                tl.dot(
                    probability_codes,
                    mean_values,
                    out_dtype=tl.int32,
                ).to(tl.float32)
                * probability_scale[:, None]
                * value_scale[None, :]
            )
        else:
            accumulator += tl.dot(
                probabilities.to(mean_values.dtype),
                mean_values,
                out_dtype=tl.float32,
            )
        maximum = new_maximum

    if INCLUDE_LOCAL:
        for local_begin in tl.range(0, local_len, BLOCK_N, num_stages=1):
            token = local_begin + token_offset
            token_valid = token < local_len
            keys = tl.load(
                local_k
                + batch * LOCAL_K_BATCH_STRIDE
                + kv_head * LOCAL_K_HEAD_STRIDE
                + token[:, None] * LOCAL_K_TOKEN_STRIDE
                + key_dim[None, :],
                mask=token_valid[:, None] & (key_dim[None, :] < HEAD_DIM),
                other=0.0,
            )
            if HEAD_TAIL_BLOCK_DIM > 0:
                tail_keys = tl.load(
                    local_k
                    + batch * LOCAL_K_BATCH_STRIDE
                    + kv_head * LOCAL_K_HEAD_STRIDE
                    + token[:, None] * LOCAL_K_TOKEN_STRIDE
                    + tail_dim[None, :],
                    mask=token_valid[:, None] & (tail_dim[None, :] < HEAD_DIM),
                    other=0.0,
                )
            values = tl.load(
                local_v
                + batch * LOCAL_V_BATCH_STRIDE
                + kv_head * LOCAL_V_HEAD_STRIDE
                + token[:, None] * LOCAL_V_TOKEN_STRIDE
                + value_dim[None, :],
                mask=token_valid[:, None] & (value_dim[None, :] < VALUE_DIM),
                other=0.0,
            )
            scores = SCALE * tl.dot(queries, tl.trans(keys), out_dtype=tl.float32)
            if HEAD_TAIL_BLOCK_DIM > 0:
                scores += SCALE * tl.dot(
                    tail_queries, tl.trans(tail_keys), out_dtype=tl.float32
                )
            visible = token[None, :] <= query[:, None] + local_offset
            valid = query_valid[:, None] & token_valid[None, :] & visible
            scores = tl.where(valid, scores, -float("inf"))
            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            denominator = denominator * correction + tl.sum(probabilities, axis=1)
            accumulator = accumulator * correction[:, None] + tl.dot(
                probabilities.to(values.dtype), values, out_dtype=tl.float32
            )
            maximum = new_maximum

    output_row = ((batch * QUERY_HEADS + query_head) * query_len + query).to(tl.int64)
    has_mass = query_valid & (denominator > 0.0)
    tl.store(
        output + output_row[:, None] * VALUE_DIM + value_dim[None, :],
        tl.where(
            has_mass[:, None],
            accumulator / tl.maximum(denominator[:, None], 1.0e-30),
            0.0,
        ),
        mask=query_valid[:, None] & (value_dim[None, :] < VALUE_DIM),
    )
    tl.store(
        lse + output_row,
        tl.where(has_mass, maximum + tl.log(denominator), -float("inf")),
        mask=query_valid,
    )


@triton.jit(
    do_not_specialize=["TOKENS"],
    do_not_specialize_on_alignment=[
        "MERGE_K_ROW_STRIDE",
        "MERGE_V_ROW_STRIDE",
        "OWNER_ROW_STRIDE",
        "DELTA_K_ROW_STRIDE",
        "DELTA_V_ROW_STRIDE",
        "DELTA_SLOT_STRIDE",
        "KEY_NORM_ROW_STRIDE",
        "TOKENS",
    ],
)
def _accumulate_state_deltas_kernel(
    merge_k,
    merge_v,
    merge_counts,
    merge_key_norm_sums,
    merge_indices,
    destinations,
    owners,
    delta_k,
    delta_v,
    delta_counts,
    touched,
    key_norm_sums,
    MERGE_K_ROW_STRIDE,
    MERGE_V_ROW_STRIDE,
    OWNER_ROW_STRIDE,
    DELTA_K_ROW_STRIDE,
    DELTA_V_ROW_STRIDE,
    DELTA_SLOT_STRIDE,
    KEY_NORM_ROW_STRIDE,
    KEY_NORM_SLOT_STRIDE,
    TOKENS,
    TOKEN_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    HAS_KEY_NORMS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token_block = tl.program_id(1).to(tl.int64)
    token = token_block * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    valid = token < TOKENS
    key_dim = tl.arange(0, HEAD_BLOCK_DIM)
    value_dim = tl.arange(0, VALUE_BLOCK_DIM)

    destination = tl.load(destinations + row * TOKENS + token, mask=valid, other=0).to(
        tl.int64
    )
    original_token = tl.load(
        merge_indices + row * TOKENS + token, mask=valid, other=0
    ).to(tl.int64)
    tl.store(
        owners + row * OWNER_ROW_STRIDE + original_token,
        destination,
        mask=valid,
    )

    k = tl.load(
        merge_k
        + row * MERGE_K_ROW_STRIDE
        + token[:, None] * HEAD_DIM
        + key_dim[None, :],
        mask=valid[:, None] & (key_dim[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        merge_v
        + row * MERGE_V_ROW_STRIDE
        + token[:, None] * VALUE_DIM
        + value_dim[None, :],
        mask=valid[:, None] & (value_dim[None, :] < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    merge_count = tl.load(
        merge_counts + row * TOKENS + token, mask=valid, other=0.0
    ).to(tl.float32)
    if HAS_KEY_NORMS:
        merge_key_norm = tl.load(
            merge_key_norm_sums + row * TOKENS + token,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    tl.atomic_or(
        touched + row * DELTA_SLOT_STRIDE + destination,
        1,
        sem="relaxed",
        mask=valid,
    )
    tl.atomic_add(
        delta_counts + row * DELTA_SLOT_STRIDE + destination,
        merge_count,
        sem="relaxed",
        mask=valid,
    )
    if HAS_KEY_NORMS:
        tl.atomic_add(
            key_norm_sums
            + row * KEY_NORM_ROW_STRIDE
            + destination * KEY_NORM_SLOT_STRIDE,
            merge_key_norm,
            sem="relaxed",
            mask=valid,
        )
    tl.atomic_add(
        delta_k
        + row * DELTA_K_ROW_STRIDE
        + destination[:, None] * HEAD_DIM
        + key_dim[None, :],
        k,
        sem="relaxed",
        mask=valid[:, None] & (key_dim[None, :] < HEAD_DIM),
    )
    tl.atomic_add(
        delta_v
        + row * DELTA_V_ROW_STRIDE
        + destination[:, None] * VALUE_DIM
        + value_dim[None, :],
        v,
        sem="relaxed",
        mask=valid[:, None] & (value_dim[None, :] < VALUE_DIM),
    )


@triton.jit
def _apply_state_deltas_kernel(
    state_k,
    state_v,
    counts,
    delta_k,
    delta_v,
    delta_counts,
    touched,
    STATE_K_ROW_STRIDE,
    STATE_V_ROW_STRIDE,
    STATE_K_SLOT_STRIDE,
    STATE_V_SLOT_STRIDE,
    COUNT_ROW_STRIDE,
    COUNT_SLOT_STRIDE,
    DELTA_K_ROW_STRIDE,
    DELTA_V_ROW_STRIDE,
    DELTA_SLOT_STRIDE,
    active_slots,
    STATE_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    state_block = tl.program_id(1).to(tl.int64)
    slot = state_block * STATE_BLOCK + tl.arange(0, STATE_BLOCK)
    valid = slot < active_slots
    is_touched = (
        tl.load(touched + row * DELTA_SLOT_STRIDE + slot, mask=valid, other=0) != 0
    )
    update = valid & is_touched
    key_dim = tl.arange(0, HEAD_BLOCK_DIM)
    value_dim = tl.arange(0, VALUE_BLOCK_DIM)

    old_k = tl.load(
        state_k
        + row * STATE_K_ROW_STRIDE
        + slot[:, None] * STATE_K_SLOT_STRIDE
        + key_dim[None, :],
        mask=update[:, None] & (key_dim[None, :] < HEAD_DIM),
        other=0.0,
    ).to(tl.float32)
    old_v = tl.load(
        state_v
        + row * STATE_V_ROW_STRIDE
        + slot[:, None] * STATE_V_SLOT_STRIDE
        + value_dim[None, :],
        mask=update[:, None] & (value_dim[None, :] < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    add_k = tl.load(
        delta_k
        + row * DELTA_K_ROW_STRIDE
        + slot[:, None] * HEAD_DIM
        + key_dim[None, :],
        mask=update[:, None] & (key_dim[None, :] < HEAD_DIM),
        other=0.0,
    )
    add_v = tl.load(
        delta_v
        + row * DELTA_V_ROW_STRIDE
        + slot[:, None] * VALUE_DIM
        + value_dim[None, :],
        mask=update[:, None] & (value_dim[None, :] < VALUE_DIM),
        other=0.0,
    )
    # As in the regular KVM kernels, accumulate in FP32 and perform one BF16
    # state write per slot rather than one rounding per source token.
    tl.store(
        state_k
        + row * STATE_K_ROW_STRIDE
        + slot[:, None] * STATE_K_SLOT_STRIDE
        + key_dim[None, :],
        old_k + add_k,
        mask=update[:, None] & (key_dim[None, :] < HEAD_DIM),
    )
    tl.store(
        state_v
        + row * STATE_V_ROW_STRIDE
        + slot[:, None] * STATE_V_SLOT_STRIDE
        + value_dim[None, :],
        old_v + add_v,
        mask=update[:, None] & (value_dim[None, :] < VALUE_DIM),
    )
    old_count = tl.load(
        counts + row * COUNT_ROW_STRIDE + slot * COUNT_SLOT_STRIDE,
        mask=update,
        other=0.0,
    )
    add_count = tl.load(
        delta_counts + row * DELTA_SLOT_STRIDE + slot,
        mask=update,
        other=0.0,
    )
    tl.store(
        counts + row * COUNT_ROW_STRIDE + slot * COUNT_SLOT_STRIDE,
        old_count + add_count,
        mask=update,
    )

    tl.store(
        delta_k
        + row * DELTA_K_ROW_STRIDE
        + slot[:, None] * HEAD_DIM
        + key_dim[None, :],
        0.0,
        mask=update[:, None] & (key_dim[None, :] < HEAD_DIM),
    )
    tl.store(
        delta_v
        + row * DELTA_V_ROW_STRIDE
        + slot[:, None] * VALUE_DIM
        + value_dim[None, :],
        0.0,
        mask=update[:, None] & (value_dim[None, :] < VALUE_DIM),
    )
    tl.store(delta_counts + row * DELTA_SLOT_STRIDE + slot, 0.0, mask=update)
    tl.store(touched + row * DELTA_SLOT_STRIDE + slot, 0, mask=update)


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=[
        "LOGIT_BATCH_STRIDE",
        "LOGIT_HEAD_STRIDE",
        "LOGIT_QUERY_STRIDE",
        "COUNT_BATCH_STRIDE",
        "COUNT_HEAD_STRIDE",
        "query_len",
        "state_len",
    ],
)
def _route_logits_tile_topk_kernel(
    route_logits,
    counts,
    candidate_scores,
    candidate_indices,
    LOGIT_BATCH_STRIDE,
    LOGIT_HEAD_STRIDE,
    LOGIT_QUERY_STRIDE,
    LOGIT_STATE_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    query_len,
    state_len,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    MAX_TILES: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    ROUTE_COUNT_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Emit exact local route winners from independent centroid tiles."""
    batch_head = tl.program_id(0).to(tl.int64)
    query_block = tl.program_id(1).to(tl.int64)
    tile = tl.program_id(2).to(tl.int64)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    query_offset = tl.arange(0, BLOCK_M)
    query = query_block * BLOCK_M + query_offset
    query_valid = query < query_len
    token_offset = tl.arange(0, BLOCK_N)
    slot = tile * BLOCK_N + token_offset
    state_valid = slot < state_len
    count = tl.load(
        counts
        + batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=state_valid,
        other=1.0,
    ).to(tl.float32)
    raw_scores = tl.load(
        route_logits
        + batch * LOGIT_BATCH_STRIDE
        + query_head * LOGIT_HEAD_STRIDE
        + query[:, None] * LOGIT_QUERY_STRIDE
        + slot[None, :] * LOGIT_STATE_STRIDE,
        mask=query_valid[:, None] & state_valid[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    # Preserve the established route selector's BF16 scale rounding exactly.
    route_scores = (raw_scores.to(tl.bfloat16) * SCALE).to(tl.bfloat16).to(tl.float32)
    route_scores += ROUTE_COUNT_BIAS * tl.log(count)[None, :]
    route_valid = state_valid & (slot >= PROTECTED_LEN)
    if MAX_LEAF_TOKENS:
        route_valid &= count <= MAX_LEAF_TOKENS
    remaining = tl.where(
        query_valid[:, None] & route_valid[None, :],
        route_scores,
        -float("inf"),
    )
    candidate_base = ((batch_head * query_len + query) * MAX_TILES + tile) * ROUTE_COUNT
    for rank in tl.static_range(0, ROUTE_COUNT):
        best_score = tl.max(remaining, axis=1)
        best_position = tl.min(
            tl.where(
                remaining == best_score[:, None],
                token_offset[None, :],
                BLOCK_N,
            ),
            axis=1,
        )
        valid_best = query_valid & (best_position < BLOCK_N)
        tl.store(
            candidate_scores + candidate_base + rank,
            best_score,
            mask=query_valid,
        )
        tl.store(
            candidate_indices + candidate_base + rank,
            tile * BLOCK_N + best_position,
            mask=valid_best,
        )
        remaining = tl.where(
            token_offset[None, :] == best_position[:, None],
            -float("inf"),
            remaining,
        )


@triton.jit(
    do_not_specialize=["query_len", "active_tiles"],
    do_not_specialize_on_alignment=["query_len", "active_tiles"],
)
def _reduce_route_logits_tile_topk_kernel(
    candidate_scores,
    candidate_indices,
    top_slots,
    query_len,
    active_tiles,
    QUERY_HEADS: tl.constexpr,
    MAX_TILES: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Reduce local centroid-tile winners to the exact global route set."""
    batch_head = tl.program_id(0).to(tl.int64)
    query_block = tl.program_id(1).to(tl.int64)
    query_offset = tl.arange(0, BLOCK_M)
    query = query_block * BLOCK_M + query_offset
    query_valid = query < query_len
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    candidate_valid = candidate < active_tiles * ROUTE_COUNT
    candidate_base = (batch_head * query_len + query) * MAX_TILES * ROUTE_COUNT
    remaining = tl.load(
        candidate_scores + candidate_base[:, None] + candidate[None, :],
        mask=query_valid[:, None] & candidate_valid[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    top_base = (batch_head * query_len + query) * ROUTE_COUNT
    route_rank = tl.arange(0, ROUTE_BLOCK)
    selected_slots = tl.full((BLOCK_M, ROUTE_BLOCK), -1, tl.int32)
    for rank in tl.static_range(0, ROUTE_COUNT):
        best_score = tl.max(remaining, axis=1)
        best_position = tl.min(
            tl.where(
                remaining == best_score[:, None],
                candidate[None, :],
                CANDIDATE_BLOCK,
            ),
            axis=1,
        )
        best_slot = tl.load(
            candidate_indices + candidate_base + best_position,
            mask=query_valid & (best_position < active_tiles * ROUTE_COUNT),
            other=-1,
        )
        selected_slots = tl.where(
            route_rank[None, :] == rank,
            best_slot[:, None],
            selected_slots,
        )
        remaining = tl.where(
            candidate[None, :] == best_position[:, None],
            -float("inf"),
            remaining,
        )
    # Match the established selector's ``reorder_like_torch`` contract: the
    # lowest-scoring boundary winner remains last, while the preceding slots
    # are ordered by centroid index.  Keeping this in the candidate reduction
    # avoids a third launch and preserves exact route order for downstream
    # expert grouping.
    boundary_slot = tl.max(
        tl.where(
            route_rank[None, :] == ROUTE_COUNT - 1,
            selected_slots,
            -1,
        ),
        axis=1,
    )
    remaining_slots = tl.where(
        route_rank[None, :] < ROUTE_COUNT - 1,
        selected_slots,
        0x7FFFFFFF,
    )
    for output_rank in tl.static_range(0, ROUTE_COUNT - 1):
        output_slot = tl.min(remaining_slots, axis=1)
        tl.store(
            top_slots + top_base + output_rank,
            output_slot,
            mask=query_valid,
        )
        remaining_slots = tl.where(
            remaining_slots == output_slot[:, None],
            0x7FFFFFFF,
            remaining_slots,
        )
    tl.store(
        top_slots + top_base + ROUTE_COUNT - 1,
        boundary_slot,
        mask=query_valid,
    )


def new_state_delta_buffers(
    state_k: torch.Tensor, state_v: torch.Tensor, capacity: int
) -> dict[str, torch.Tensor]:
    batch, kv_heads, _, head_dim = state_k.shape
    value_dim = int(state_v.size(-1))
    return {
        "delta_k": torch.zeros(
            batch,
            kv_heads,
            capacity,
            head_dim,
            dtype=torch.float32,
            device=state_k.device,
        ),
        "delta_v": torch.zeros(
            batch,
            kv_heads,
            capacity,
            value_dim,
            dtype=torch.float32,
            device=state_v.device,
        ),
        "delta_counts": torch.zeros(
            batch, kv_heads, capacity, dtype=torch.float32, device=state_k.device
        ),
        "touched": torch.zeros(
            batch, kv_heads, capacity, dtype=torch.int32, device=state_k.device
        ),
    }


def new_state_maxsim_buffers(
    overflow_k: torch.Tensor, token_capacity: int
) -> dict[str, torch.Tensor]:
    batch, kv_heads = overflow_k.shape[:2]
    score_shape = (batch, kv_heads, token_capacity)
    return {
        "route_scores": torch.empty(
            score_shape, dtype=overflow_k.dtype, device=overflow_k.device
        ),
        "route_indices": torch.empty(
            score_shape, dtype=torch.long, device=overflow_k.device
        ),
        "select_scores": torch.empty(
            score_shape, dtype=overflow_k.dtype, device=overflow_k.device
        ),
        "overflow_key_norms": torch.empty(
            score_shape, dtype=torch.float32, device=overflow_k.device
        ),
    }


def constituent_rms(key: torch.Tensor) -> torch.Tensor:
    """Compute one FP32 RMS per key without an intermediate FP32 tensor."""
    if not key.is_cuda or key.ndim != 4 or key.stride(-1) != 1:
        raise ValueError("fused constituent RMS requires rank-four CUDA keys")
    batch, heads, token_len, head_dim = key.shape
    output = torch.empty(
        batch,
        heads,
        token_len,
        1,
        dtype=torch.float32,
        device=key.device,
    )
    block_m = max(1, 1024 // triton.next_power_of_2(head_dim))
    _constituent_rms_kernel[(batch, heads, triton.cdiv(token_len, block_m))](
        key,
        output,
        KEY_BATCH_STRIDE=key.stride(0),
        KEY_HEAD_STRIDE=key.stride(1),
        KEY_TOKEN_STRIDE=key.stride(2),
        OUTPUT_BATCH_STRIDE=output.stride(0),
        OUTPUT_HEAD_STRIDE=output.stride(1),
        OUTPUT_TOKEN_STRIDE=output.stride(2),
        token_len=token_len,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_D=triton.next_power_of_2(head_dim),
        **_launch_kwargs(4),
    )
    return output


def prepare_state_clustering_keys(
    state_k: torch.Tensor,
    counts: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    state_len: int,
    key_norm_sums: torch.Tensor | None = None,
    geometry: str,
    slot_indices: torch.Tensor | None = None,
    block_s: int | None = None,
    num_warps: int = 4,
    prepare_coherence_route: bool = True,
    prepare_coherence_append: bool = True,
    prepare_coherence_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refresh all or selected spherical/coherence centroid route keys."""
    if geometry not in {"spherical", "coherence", "spherical_coherence"}:
        raise ValueError(f"unsupported prepared state geometry: {geometry}")
    if not state_k.is_cuda or not counts.is_cuda:
        raise ValueError("prepared state geometry requires CUDA tensors")
    batch, kv_heads, _, head_dim = state_k.shape
    coherence = geometry in {"coherence", "spherical_coherence"}
    if coherence:
        if key_norm_sums is None or not key_norm_sums.is_cuda:
            raise ValueError("coherence routing requires CUDA key-norm sums")
        if tuple(key_norm_sums.shape[:3]) != tuple(state_k.shape[:3]):
            raise ValueError("key-norm sums have the wrong state shape")
    if coherence and not (prepare_coherence_route or prepare_coherence_append):
        raise ValueError("coherence preparation needs a route or append view")
    selective_coherence = coherence and not (
        prepare_coherence_route and prepare_coherence_append
    )
    prepared_route = buffers.get("prepared_route_state")
    prepared_append = buffers.get("prepared_append_state")
    prepared_scale = buffers.get("prepared_select_scale")
    needs_prepared = (
        prepared_route is None
        or tuple(prepared_route.shape) != tuple(state_k.shape)
        or prepared_route.device != state_k.device
        or prepared_route.dtype != state_k.dtype
        or (coherence and prepared_append is None)
        or (
            coherence
            and (
                tuple(prepared_append.shape) != tuple(state_k.shape)
                or prepared_append.device != state_k.device
                or prepared_append.dtype != state_k.dtype
            )
        )
        or (
            coherence
            and not selective_coherence
            and prepared_append.data_ptr() == prepared_route.data_ptr()
        )
        or (coherence and prepared_scale is None)
        or (
            coherence
            and (
                tuple(prepared_scale.shape) != tuple(state_k.shape[:3])
                or prepared_scale.device != state_k.device
                or prepared_scale.dtype != torch.float32
            )
        )
    )
    if needs_prepared:
        if slot_indices is not None:
            raise ValueError("prepared state geometry is unavailable for refresh")
        if coherence and prepare_coherence_route and prepare_coherence_append:
            prepared_route = torch.empty_like(state_k)
            prepared_append = torch.empty_like(state_k)
        else:
            prepared_route = torch.empty_like(state_k)
            prepared_append = prepared_route
        buffers.pop("prepared_coherence_state", None)
        prepared_scale = (
            torch.empty(state_k.shape[:3], dtype=torch.float32, device=state_k.device)
            if coherence
            else counts
        )
        buffers["prepared_route_state"] = prepared_route
        buffers["prepared_append_state"] = prepared_append
        buffers["prepared_select_scale"] = prepared_scale
    if not coherence:
        key_norm_pointer = counts
        prepared_append = prepared_route
        prepared_scale = counts
    else:
        key_norm_pointer = key_norm_sums
    if block_s is None:
        block_s = min(8, max(1, 1024 // head_dim))
    if block_s <= 0 or block_s & (block_s - 1):
        raise ValueError("state preparation tile must be a positive power of two")
    indexed = slot_indices is not None
    if indexed:
        if (
            not slot_indices.is_cuda
            or slot_indices.ndim != 3
            or tuple(slot_indices.shape[:2]) != (batch, kv_heads)
        ):
            raise ValueError("state refresh indices have the wrong shape")
        index_pointer = slot_indices
        slot_count = int(slot_indices.size(2))
    else:
        index_pointer = counts
        slot_count = state_len
    if slot_count:
        _prepare_state_clustering_keys_kernel[
            (batch, kv_heads, triton.cdiv(slot_count, block_s))
        ](
            state_k,
            counts,
            key_norm_pointer,
            prepared_route,
            prepared_append,
            prepared_scale,
            index_pointer,
            STATE_BATCH_STRIDE=state_k.stride(0),
            STATE_HEAD_STRIDE=state_k.stride(1),
            STATE_TOKEN_STRIDE=state_k.stride(2),
            COUNT_BATCH_STRIDE=counts.stride(0),
            COUNT_HEAD_STRIDE=counts.stride(1),
            COUNT_TOKEN_STRIDE=counts.stride(2),
            KEY_NORM_BATCH_STRIDE=key_norm_pointer.stride(0),
            KEY_NORM_HEAD_STRIDE=key_norm_pointer.stride(1),
            KEY_NORM_TOKEN_STRIDE=key_norm_pointer.stride(2),
            OUTPUT_BATCH_STRIDE=prepared_route.stride(0),
            OUTPUT_HEAD_STRIDE=prepared_route.stride(1),
            OUTPUT_TOKEN_STRIDE=prepared_route.stride(2),
            SCALE_BATCH_STRIDE=prepared_scale.stride(0),
            SCALE_HEAD_STRIDE=prepared_scale.stride(1),
            SCALE_TOKEN_STRIDE=prepared_scale.stride(2),
            INDEX_BATCH_STRIDE=index_pointer.stride(0),
            INDEX_HEAD_STRIDE=index_pointer.stride(1),
            INDEX_TOKEN_STRIDE=index_pointer.stride(2),
            slot_count=slot_count,
            state_len=state_len,
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            BLOCK_S=block_s,
            COHERENCE=coherence,
            WRITE_ROUTE=coherence and prepare_coherence_route,
            WRITE_APPEND=not coherence or prepare_coherence_append,
            WRITE_SCALE=coherence and prepare_coherence_scale,
            INDEXED=indexed,
            **_launch_kwargs(num_warps),
        )
    return prepared_route, prepared_append, prepared_scale


def streaming_state_maxsim(
    overflow_k: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    state_len: int,
    sink_len: int,
    key_norm_sums: torch.Tensor | None = None,
    geometry: str = "raw",
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
    prepare_block_s: int | None = None,
    prepare_num_warps: int = 4,
    prepare_state_geometry: bool = True,
    materialize_prepared_scores: bool = False,
    coherence_single_matmul: bool = False,
    mask_invalid_state: bool = True,
    tiled_prepared_scores: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scan transient leaf keys without materializing leaf-by-state scores."""
    if not all(tensor.is_cuda for tensor in (overflow_k, state_k, counts)):
        raise ValueError("streaming LOD state routing requires CUDA tensors")
    batch, kv_heads, overflow_len, head_dim = overflow_k.shape
    if geometry not in {"raw", "spherical", "coherence", "spherical_coherence"}:
        raise ValueError(f"unsupported streaming state geometry: {geometry}")
    coherence = geometry in {"coherence", "spherical_coherence"}
    if coherence:
        if key_norm_sums is None or not key_norm_sums.is_cuda:
            raise ValueError("coherence routing requires CUDA key-norm sums")
        if tuple(key_norm_sums.shape[:2]) != (batch, kv_heads):
            raise ValueError("key-norm sums have the wrong state prefix")
        if state_len > int(key_norm_sums.size(2)):
            raise ValueError("active state exceeds the key-norm storage")
    if state_len > int(state_k.size(2)) or sink_len >= state_len:
        raise ValueError("invalid active LOD state range")
    route_scores = buffers["route_scores"]
    route_indices = buffers["route_indices"]
    select_scores = buffers["select_scores"]
    expected_prefix = (batch, kv_heads)
    if (
        tuple(route_scores.shape[:2]) != expected_prefix
        or int(route_scores.size(2)) < overflow_len
    ):
        raise ValueError("streaming LOD max-sim buffers are too small")
    # Coherence's two centroid representations differ only by one scalar per
    # slot. Scan the stored K sum once and apply those scalars in the MFMA
    # kernel instead of materializing and refreshing two D-wide key caches.
    fused_coherence = coherence and not materialize_prepared_scores
    prepared = geometry != "raw" and not fused_coherence
    if prepared:
        if prepare_state_geometry:
            prepared_route, prepared_append, prepared_scale = (
                prepare_state_clustering_keys(
                    state_k,
                    counts,
                    buffers,
                    state_len=state_len,
                    key_norm_sums=key_norm_sums,
                    geometry=geometry,
                    block_s=prepare_block_s,
                    num_warps=prepare_num_warps,
                    prepare_coherence_route=not coherence_single_matmul,
                    prepare_coherence_append=True,
                    prepare_coherence_scale=coherence_single_matmul,
                )
            )
        else:
            prepared_route = buffers.get("prepared_route_state")
            prepared_append = buffers.get("prepared_append_state")
            prepared_scale = buffers.get("prepared_select_scale")
            if (
                prepared_route is None
                or prepared_append is None
                or prepared_scale is None
            ):
                raise ValueError("prepared state geometry is unavailable")
        block_m = max(block_m, 32)
        if materialize_prepared_scores:
            # Prepared geometry buffers follow the allocated state capacity,
            # which can be larger than the currently active state. Restrict
            # the dense fallback to active slots just like the streaming
            # kernel does; otherwise inactive capacity both mismatches the
            # validity mask and could win the max reduction uninitialized.
            active_route = prepared_route[..., :state_len, :]
            active_append = prepared_append[..., :state_len, :]
            if coherence and coherence_single_matmul:
                append_scores_dense = torch.matmul(
                    overflow_k, active_append.transpose(-1, -2)
                )
                _scaled_coherence_maxsim_kernel[
                    (batch, kv_heads, triton.cdiv(overflow_len, block_m))
                ](
                    append_scores_dense,
                    prepared_scale,
                    counts,
                    route_scores,
                    route_indices,
                    select_scores,
                    SCORE_BATCH_STRIDE=append_scores_dense.stride(0),
                    SCORE_HEAD_STRIDE=append_scores_dense.stride(1),
                    SCORE_TOKEN_STRIDE=append_scores_dense.stride(2),
                    SCORE_STATE_STRIDE=append_scores_dense.stride(3),
                    SCALE_BATCH_STRIDE=prepared_scale.stride(0),
                    SCALE_HEAD_STRIDE=prepared_scale.stride(1),
                    SCALE_TOKEN_STRIDE=prepared_scale.stride(2),
                    COUNT_BATCH_STRIDE=counts.stride(0),
                    COUNT_HEAD_STRIDE=counts.stride(1),
                    COUNT_TOKEN_STRIDE=counts.stride(2),
                    OUTPUT_BATCH_STRIDE=route_scores.stride(0),
                    OUTPUT_HEAD_STRIDE=route_scores.stride(1),
                    OUTPUT_TOKEN_STRIDE=route_scores.stride(2),
                    overflow_len=overflow_len,
                    state_len=state_len,
                    SINK_LEN=sink_len,
                    BLOCK_M=block_m,
                    BLOCK_N=max(block_n, 64),
                    **_launch_kwargs(num_warps),
                )
                active = (..., slice(None, overflow_len))
                return (
                    route_scores[active],
                    route_indices[active],
                    select_scores[active],
                )
            elif coherence:
                route_scores_dense = torch.matmul(
                    overflow_k, active_route.transpose(-1, -2)
                )
                append_scores_dense = torch.matmul(
                    overflow_k, active_append.transpose(-1, -2)
                )
            else:
                route_scores_dense = torch.matmul(
                    overflow_k, active_route.transpose(-1, -2)
                )
                append_scores_dense = route_scores_dense
            if mask_invalid_state:
                invalid = counts[..., :state_len, 0].le(0.5).unsqueeze(-2)
                route_scores_dense.masked_fill_(invalid, float("-inf"))
                if (
                    append_scores_dense is not None
                    and append_scores_dense is not route_scores_dense
                ):
                    append_scores_dense.masked_fill_(invalid, float("-inf"))
            if append_scores_dense is route_scores_dense and sink_len == 0:
                # Separate-sink spherical routing uses the same logits for
                # append selection and merge assignment. Reduce the 512 MiB
                # score field once instead of launching two identical maxima.
                route_score, route_index = route_scores_dense.max(dim=-1)
                select_score = route_score
            else:
                if append_scores_dense is not None:
                    select_score = append_scores_dense.max(dim=-1).values
                route_scores_dense[..., :sink_len] = float("-inf")
                route_score, route_index = route_scores_dense.max(dim=-1)
            return route_score, route_index, select_score
    raise ValueError(
        "the LoD release requires prepared spherical/coherence state routing"
    )


def route_logits_coarse_attention(
    q: torch.Tensor,
    route_logits: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    block_m: int = 4,
    block_n: int = 32,
    num_warps: int = 4,
    precompute_mean_values: bool = False,
    int8_state_pv: bool = False,
    head_major: bool | None = None,
    max_grouped_rows: int = 8,
    direct_gqa_rows: bool = False,
    route_logit_scale: torch.Tensor | None = None,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the coarse state/local branch while reusing routing logits."""
    tensors = (q, route_logits, state_v, counts, local_k, local_v, top_slots)
    if route_logit_scale is not None:
        tensors = (*tensors, route_logit_scale)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("LOD Triton coarse attention requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("LOD Triton coarse attention requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(state_v.size(1))
    local_len = int(local_k.size(2))
    value_dim = int(state_v.size(-1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query heads do not match the requested GQA grouping")
    if tuple(route_logits.shape) != (batch, query_heads, query_len, state_len):
        raise ValueError("routing logits have the wrong shape")
    if route_logit_scale is not None and tuple(route_logit_scale.shape) != (
        batch,
        query_heads,
        query_len,
        1,
    ):
        raise ValueError("routing-logit scales have the wrong shape")
    if tuple(top_slots.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("top-slot routes have the wrong shape")
    # Routing itself has an optimized top-eight fast path, but this coarse
    # subtraction kernel only consumes an already-selected route tensor.  Its
    # static loop is valid for broader experimental page budgets as well.
    if state_len > int(state_v.size(2)) or state_len > int(counts.size(2)):
        raise ValueError("active state exceeds the supplied storage")
    if local_len != 0 and local_len < query_len:
        raise ValueError("local attention must contain every current query token")
    if int(local_v.size(2)) != local_len:
        raise ValueError("local key/value lengths differ")
    if int(local_k.size(1)) != kv_heads or int(local_v.size(1)) != kv_heads:
        raise ValueError("local and state KV heads differ")
    if int(local_k.size(-1)) != head_dim:
        raise ValueError("local key dimension differs from the query")
    if int(local_v.size(-1)) != value_dim:
        raise ValueError("local and state value dimensions differ")
    if block_m <= 0 or block_n <= 0 or max_grouped_rows <= 0:
        raise ValueError("coarse-attention tile sizes must be positive")

    if head_dim > 512 or value_dim > 256:
        # Absorbed MLA heads (for example 576-wide Q/K and 512-wide V) need
        # much larger feature tiles than conventional attention.  Keep their
        # query tile small and head-major so register/shared-memory pressure
        # does not scale with the GQA group as well.
        block_m = min(block_m, 4)
        num_warps = min(num_warps, 4)
        head_major = True
    # Runtime strides can make Triton spill the per-row value accumulator to
    # shared memory. Keep that tile bounded for high-GQA models. The direct-GQA
    # layout fills a power-of-two matrix tile with the real (possibly
    # irregular) GQA factor: GQA5 with a 128-row tile uses 25 query positions
    # and masks only three tail rows. The compatibility layout pads the group
    # itself to a power of two and therefore wastes 24/64 rows for GQA5.
    padded_group_size = triton.next_power_of_2(kv_group_size)
    if direct_gqa_rows and head_major is not True:
        block_m = max(1, max_grouped_rows // kv_group_size)
        grouped_rows = triton.next_power_of_2(kv_group_size * block_m)
        if grouped_rows > max_grouped_rows and kv_group_size <= max_grouped_rows:
            raise ValueError(
                "direct-GQA coarse attention requires a power-of-two grouped-row cap"
            )
        if head_major is None:
            head_major = False
    else:
        if head_major is not True and padded_group_size * block_m > max_grouped_rows:
            block_m = max(1, max_grouped_rows // padded_group_size)
        if head_major is not True and block_m & (block_m - 1):
            block_m = 1 << (block_m.bit_length() - 1)
        grouped_rows = padded_group_size * block_m
    value_block_dim = triton.next_power_of_2(value_dim)
    if head_major is None:
        # GQA grouping keeps one value accumulator per grouped query row.
        # Large groups with wide values can therefore exceed the device's
        # shared-memory budget even though each individual head is ordinary
        # attention (for example 8 * 16 * 256 * fp32 = 128 KiB).
        # Split those cases by query head. The kernel math is unchanged and
        # the extra programs expose useful parallelism on these larger models.
        grouped_accumulator_bytes = grouped_rows * value_block_dim * 4
        head_major = grouped_accumulator_bytes > 48 * 1024
    # Non-power-of-two GQA groups can shrink BLOCK_M to a non-power-of-two
    # value (for example 64 // 6 == 10).  Head-major execution no longer
    # needs BLOCK_M to absorb the GQA group, so round it down to the largest
    # legal query tile instead of rejecting otherwise supported group sizes.
    if head_major and block_m & (block_m - 1):
        block_m = 1 << (block_m.bit_length() - 1)
    row_count = block_m if head_major else grouped_rows
    if row_count & (row_count - 1):
        raise ValueError(
            "head-major coarse attention requires a power-of-two query tile"
        )

    mean_begin = None
    mean_end = None
    if timing_events is not None:
        mean_begin = torch.cuda.Event(enable_timing=True)
        mean_end = torch.cuda.Event(enable_timing=True)
        mean_begin.record()
    kernel_state_v = state_v
    state_v_scales = state_v
    if int8_state_pv:
        raise ValueError("the LoD release does not quantize coarse state values")
    elif precompute_mean_values:
        active_counts = counts[..., :state_len, :].clamp_min(1.0)
        kernel_state_v = (
            (state_v[..., :state_len, :].float() / active_counts)
            .to(state_v.dtype)
            .contiguous()
        )
    if mean_end is not None:
        mean_end.record()

    output = torch.empty(
        batch,
        query_heads,
        query_len,
        value_dim,
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.empty(
        batch,
        query_heads,
        query_len,
        dtype=torch.float32,
        device=q.device,
    )
    grid = (
        batch,
        query_heads if head_major else kv_heads,
        triton.cdiv(query_len, block_m),
    )
    head_block_dim = min(triton.next_power_of_2(head_dim), 512)
    head_tail_block_dim = (
        0
        if head_dim <= head_block_dim
        else triton.next_power_of_2(head_dim - head_block_dim)
    )
    _route_logits_coarse_attention_kernel[grid](
        q,
        route_logits,
        route_logit_scale if route_logit_scale is not None else counts,
        kernel_state_v,
        state_v_scales,
        counts,
        local_k,
        local_v,
        top_slots,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        route_logits.stride(0),
        route_logits.stride(1),
        route_logits.stride(2),
        route_logits.stride(3),
        kernel_state_v.stride(0),
        kernel_state_v.stride(1),
        kernel_state_v.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        local_k.stride(0),
        local_k.stride(1),
        local_k.stride(2),
        local_v.stride(0),
        local_v.stride(1),
        local_v.stride(2),
        top_slots.stride(0),
        top_slots.stride(1),
        top_slots.stride(2),
        query_len,
        state_len,
        local_len,
        local_len - query_len,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_MAJOR=head_major,
        ROW_COUNT=row_count,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=head_block_dim,
        HEAD_TAIL_BLOCK_DIM=head_tail_block_dim,
        VALUE_BLOCK_DIM=value_block_dim,
        ROUTE_COUNT=int(top_slots.size(-1)),
        STATE_V_IS_MEAN=precompute_mean_values,
        INT8_STATE_PV=int8_state_pv,
        HAS_ROUTE_LOGIT_SCALE=route_logit_scale is not None,
        SCALE=scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        INCLUDE_LOCAL=local_len > 0,
        **_launch_kwargs(num_warps),
    )
    if mean_end is not None:
        kernel_end = torch.cuda.Event(enable_timing=True)
        kernel_end.record()
        timing_events.setdefault("coarse_mean_v", []).append((mean_begin, mean_end))
        timing_events.setdefault("coarse_kernel", []).append((mean_end, kernel_end))
    return output, lse


def route_logits_hierarchical_topk(
    route_logits: torch.Tensor,
    counts: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    route_count_bias: float = 1.0,
    topk: int = 3,
    protected_len: int = 0,
    max_leaf_tokens: int | None = None,
    block_m: int = 16,
    block_n: int = 128,
    tile_num_warps: int = 4,
    reduce_num_warps: int = 4,
) -> torch.Tensor:
    """Select exact routes with centroid-tile parallelism and a small reduction."""
    if not route_logits.is_cuda or not counts.is_cuda:
        raise ValueError("hierarchical route selection requires CUDA tensors")
    if not route_logits.is_contiguous() or not counts.is_contiguous():
        raise ValueError("hierarchical route selection requires contiguous tensors")
    if route_logits.ndim != 4 or counts.ndim != 4 or int(counts.size(-1)) != 1:
        raise ValueError("hierarchical route selection received invalid tensors")
    batch, query_heads, query_len, logit_state_len = route_logits.shape
    kv_heads = int(counts.size(1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query heads do not match the requested GQA grouping")
    if topk not in (2, 3, 4, 8):
        raise ValueError(
            "hierarchical route selection currently supports top-2/top-3/top-4/top-8"
        )
    if not 0 < state_len <= logit_state_len or state_len > int(counts.size(2)):
        raise ValueError("active route state exceeds the supplied storage")
    if protected_len < 0 or protected_len + topk > state_len:
        raise ValueError("protected state leaves too few routing candidates")
    if max_leaf_tokens is not None and max_leaf_tokens <= 0:
        raise ValueError("maximum routed leaf count must be positive")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError("hierarchical route query tile must be a power of two")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError("hierarchical route state tile must be a power of two")

    active_tiles = triton.cdiv(state_len, block_n)
    max_tiles = triton.cdiv(int(counts.size(2)), block_n)
    candidate_scores = torch.empty(
        batch,
        query_heads,
        query_len,
        max_tiles,
        topk,
        dtype=torch.float32,
        device=route_logits.device,
    )
    candidate_indices = torch.empty(
        batch,
        query_heads,
        query_len,
        max_tiles,
        topk,
        dtype=torch.int32,
        device=route_logits.device,
    )
    top_slots = torch.empty(
        batch,
        query_heads,
        query_len,
        topk,
        dtype=torch.long,
        device=route_logits.device,
    )
    _route_logits_tile_topk_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m), active_tiles)
    ](
        route_logits,
        counts,
        candidate_scores,
        candidate_indices,
        route_logits.stride(0),
        route_logits.stride(1),
        route_logits.stride(2),
        route_logits.stride(3),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        query_len,
        state_len,
        QUERY_HEADS=query_heads,
        KV_GROUP_SIZE=kv_group_size,
        MAX_TILES=max_tiles,
        ROUTE_COUNT=topk,
        PROTECTED_LEN=protected_len,
        MAX_LEAF_TOKENS=max_leaf_tokens or 0,
        SCALE=scale,
        ROUTE_COUNT_BIAS=route_count_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **_launch_kwargs(tile_num_warps),
    )
    candidate_block = triton.next_power_of_2(max_tiles * topk)
    _reduce_route_logits_tile_topk_kernel[
        (batch * query_heads, triton.cdiv(query_len, block_m))
    ](
        candidate_scores,
        candidate_indices,
        top_slots,
        query_len,
        active_tiles,
        QUERY_HEADS=query_heads,
        MAX_TILES=max_tiles,
        ROUTE_COUNT=topk,
        ROUTE_BLOCK=triton.next_power_of_2(topk),
        CANDIDATE_BLOCK=candidate_block,
        BLOCK_M=block_m,
        **_launch_kwargs(reduce_num_warps),
    )
    return top_slots


def merge_state_in_place(
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    merge_k: torch.Tensor,
    merge_v: torch.Tensor,
    merge_counts: torch.Tensor | None,
    merge_indices: torch.Tensor,
    destinations: torch.Tensor,
    owners: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    active_slots: int | None = None,
    key_norm_sums: torch.Tensor | None = None,
    merge_key_norm_sums: torch.Tensor | None = None,
) -> None:
    if not all(
        tensor.is_cuda for tensor in (state_k, state_v, counts, merge_k, merge_v)
    ):
        raise ValueError("LOD Triton state update requires CUDA tensors")
    batch, kv_heads, tokens, head_dim = merge_k.shape
    value_dim = int(merge_v.size(-1))
    if (
        state_k.stride(3) != 1
        or state_v.stride(3) != 1
        or counts.stride(3) != 1
        or not merge_k.is_contiguous()
        or not merge_v.is_contiguous()
    ):
        raise ValueError("LOD Triton state update received unsupported strides")
    rows = batch * kv_heads
    if merge_counts is None:
        merge_counts = torch.ones(
            batch,
            kv_heads,
            tokens,
            dtype=torch.float32,
            device=merge_k.device,
        )
    elif tuple(merge_counts.shape) not in {
        (batch, kv_heads, tokens),
        (batch, kv_heads, tokens, 1),
    }:
        raise ValueError("LOD merge counts have the wrong shape")
    merge_counts = merge_counts.reshape(batch, kv_heads, tokens).contiguous()
    has_key_norms = key_norm_sums is not None
    if has_key_norms != (merge_key_norm_sums is not None):
        raise ValueError("state and merge key-norm sums must be supplied together")
    if has_key_norms:
        if not key_norm_sums.is_cuda or not merge_key_norm_sums.is_cuda:
            raise ValueError("LOD key-norm state update requires CUDA tensors")
        if tuple(key_norm_sums.shape[:3]) != tuple(state_k.shape[:3]):
            raise ValueError("state key-norm sums have the wrong shape")
        if tuple(merge_key_norm_sums.shape[:3]) != (batch, kv_heads, tokens):
            raise ValueError("merge key-norm sums have the wrong shape")
        merge_key_norm_sums = merge_key_norm_sums.reshape(
            batch, kv_heads, tokens
        ).contiguous()
    else:
        # These pointers are not read by the constexpr-disabled kernel branch.
        key_norm_sums = counts
        merge_key_norm_sums = merge_counts
    capacity = int(buffers["touched"].size(2))
    if active_slots is None:
        active_slots = int(state_k.size(2))
    if active_slots > int(state_k.size(2)) or active_slots > capacity:
        raise ValueError("LOD state delta capacity is smaller than the active state")
    # Keep each atomic tile at 1024 lanes, matching the proven KVM update
    # shape. A 256-wide head therefore uses four tokens per program.
    token_block = 1 if max(head_dim, value_dim) > 256 else 4
    _accumulate_state_deltas_kernel[(rows, triton.cdiv(tokens, token_block))](
        merge_k,
        merge_v,
        merge_counts,
        merge_key_norm_sums,
        merge_indices,
        destinations,
        owners,
        buffers["delta_k"],
        buffers["delta_v"],
        buffers["delta_counts"],
        buffers["touched"],
        key_norm_sums,
        merge_k.stride(1),
        merge_v.stride(1),
        owners.stride(1),
        buffers["delta_k"].stride(1),
        buffers["delta_v"].stride(1),
        buffers["touched"].stride(1),
        key_norm_sums.stride(1),
        key_norm_sums.stride(2),
        TOKENS=tokens,
        TOKEN_BLOCK=token_block,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        HAS_KEY_NORMS=has_key_norms,
        **_launch_kwargs(8),
    )
    # The KVM apply kernel uses an 8x128 tile. Preserve the same 1024-lane
    # footprint for a 256-wide state rather than doubling register use.
    state_block = 1 if max(head_dim, value_dim) > 256 else 4
    _apply_state_deltas_kernel[(rows, triton.cdiv(active_slots, state_block))](
        state_k,
        state_v,
        counts,
        buffers["delta_k"],
        buffers["delta_v"],
        buffers["delta_counts"],
        buffers["touched"],
        state_k.stride(1),
        state_v.stride(1),
        state_k.stride(2),
        state_v.stride(2),
        counts.stride(1),
        counts.stride(2),
        buffers["delta_k"].stride(1),
        buffers["delta_v"].stride(1),
        buffers["touched"].stride(1),
        active_slots,
        STATE_BLOCK=state_block,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        **_launch_kwargs(2),
    )


def _output_has_internal_overlap(output: torch.Tensor) -> bool:
    """Conservatively reject writable views whose logical elements alias."""
    span = 1
    dimensions = sorted(
        (
            (int(stride), int(size))
            for size, stride in zip(output.shape, output.stride(), strict=True)
            if int(size) > 1
        ),
        key=lambda item: item[0],
    )
    for stride, size in dimensions:
        if stride < span:
            return True
        span += (size - 1) * stride
    return False


def merge_attention_branches_with_sink(
    q: torch.Tensor,
    sink_k: torch.Tensor,
    sink_v: torch.Tensor,
    primary_out: torch.Tensor,
    primary_lse: torch.Tensor,
    secondary_out: torch.Tensor | None = None,
    secondary_lse: torch.Tensor | None = None,
    tertiary_out: torch.Tensor | None = None,
    tertiary_lse: torch.Tensor | None = None,
    *,
    kv_group_size: int,
    scale: float,
    output_buffer: torch.Tensor | None = None,
    block_m: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fuse the final LSE reduction with exact side-sink attention."""
    if not q.is_cuda:
        raise ValueError("fused sink reduction requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    if query_heads != int(sink_k.size(1)) * kv_group_size:
        raise ValueError("query heads do not match the side sink's GQA grouping")
    if tuple(sink_v.shape[:3]) != tuple(sink_k.shape[:3]):
        raise ValueError("side sink K/V shapes differ")
    if int(sink_k.size(0)) != batch:
        raise ValueError("side sink and query batch sizes differ")
    if int(sink_k.size(2)) <= 0:
        raise ValueError("the side sink must contain at least one token")
    if int(sink_k.size(-1)) != head_dim or int(sink_v.size(-1)) != head_dim:
        raise ValueError("fused sink reduction requires equal Q/K/V head sizes")
    expected_output_shape = tuple(q.shape)
    expected_lse_shape = tuple(q.shape[:-1])
    branches = (
        (primary_out, primary_lse, "primary"),
        (secondary_out, secondary_lse, "secondary"),
        (tertiary_out, tertiary_lse, "tertiary"),
    )
    for branch_out, branch_lse, name in branches:
        if (branch_out is None) != (branch_lse is None):
            raise ValueError(f"{name} attention output and LSE must be paired")
        if branch_out is None:
            continue
        if tuple(branch_out.shape) != expected_output_shape:
            raise ValueError(f"{name} attention output has the wrong shape")
        if tuple(branch_lse.shape) != expected_lse_shape:
            raise ValueError(f"{name} attention LSE has the wrong shape")
        if not branch_out.is_cuda or not branch_lse.is_cuda:
            raise ValueError("fused sink reduction requires CUDA branch tensors")
        if int(branch_out.stride(-1)) != 1:
            raise ValueError("fused sink reduction requires contiguous head features")
    if secondary_out is None and tertiary_out is not None:
        raise ValueError("a tertiary attention branch requires a secondary branch")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError("fused sink reduction block size must be a power of two")
    if (
        int(q.stride(-1)) != 1
        or int(sink_k.stride(-1)) != 1
        or int(sink_v.stride(-1)) != 1
    ):
        raise ValueError("fused sink reduction requires contiguous head features")

    # Triton still needs valid typed pointers for compile-time-disabled branches.
    secondary_out = primary_out if secondary_out is None else secondary_out
    secondary_lse = primary_lse if secondary_lse is None else secondary_lse
    tertiary_out = primary_out if tertiary_out is None else tertiary_out
    tertiary_lse = primary_lse if tertiary_lse is None else tertiary_lse
    include_secondary = branches[1][0] is not None
    include_tertiary = branches[2][0] is not None
    output = torch.empty_like(q) if output_buffer is None else output_buffer
    if (
        tuple(output.shape) != expected_output_shape
        or output.dtype != q.dtype
        or output.device != q.device
        or int(output.stride(-1)) != 1
        or _output_has_internal_overlap(output)
    ):
        raise ValueError("fused sink output buffer has incompatible geometry")
    grid = (batch, query_heads, triton.cdiv(query_len, block_m))
    _merge_attention_branches_with_sink_kernel[grid](
        q,
        sink_k,
        sink_v,
        primary_out,
        primary_lse,
        secondary_out,
        secondary_lse,
        tertiary_out,
        tertiary_lse,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sink_k.stride(0),
        sink_k.stride(1),
        sink_k.stride(2),
        sink_v.stride(0),
        sink_v.stride(1),
        sink_v.stride(2),
        primary_out.stride(0),
        primary_out.stride(1),
        primary_out.stride(2),
        primary_lse.stride(0),
        primary_lse.stride(1),
        primary_lse.stride(2),
        secondary_out.stride(0),
        secondary_out.stride(1),
        secondary_out.stride(2),
        secondary_lse.stride(0),
        secondary_lse.stride(1),
        secondary_lse.stride(2),
        tertiary_out.stride(0),
        tertiary_out.stride(1),
        tertiary_out.stride(2),
        tertiary_lse.stride(0),
        tertiary_lse.stride(1),
        tertiary_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        QUERY_LEN=query_len,
        QUERY_HEADS=query_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        SINK_LEN=int(sink_k.size(2)),
        INCLUDE_SECONDARY=include_secondary,
        INCLUDE_TERTIARY=include_tertiary,
        SCALE=scale,
        BLOCK_M=block_m,
        **_launch_kwargs(num_warps),
    )
    return output


__all__ = [
    "constituent_rms",
    "merge_attention_branches_with_sink",
    "merge_state_in_place",
    "new_state_delta_buffers",
    "new_state_maxsim_buffers",
    "prepare_state_clustering_keys",
    "route_logits_coarse_attention",
    "route_logits_hierarchical_topk",
    "streaming_state_maxsim",
]
