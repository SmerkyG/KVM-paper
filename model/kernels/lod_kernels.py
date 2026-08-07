"""Forward-only Triton kernels for LOD state maintenance.

The kernels mirror the efficient parts of the training KVM implementation:
merge tokens accumulate into persistent FP32 deltas and each touched BF16
state slot is rounded only once.  Query routing scans the compact state in
tiles and retains only the top eight slots instead of materializing the full
query-by-state score tensor.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _launch_kwargs(num_warps: int) -> dict[str, int]:
    kwargs = {"num_warps": num_warps, "num_stages": 1}
    if torch.version.hip is not None:
        kwargs["waves_per_eu"] = 1
    return kwargs


@triton.jit
def _streaming_state_maxsim_kernel(
    overflow_k,
    state_k,
    counts,
    route_scores,
    route_indices,
    select_scores,
    OVERFLOW_BATCH_STRIDE: tl.constexpr,
    OVERFLOW_HEAD_STRIDE: tl.constexpr,
    OVERFLOW_TOKEN_STRIDE: tl.constexpr,
    STATE_BATCH_STRIDE: tl.constexpr,
    STATE_HEAD_STRIDE: tl.constexpr,
    STATE_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE: tl.constexpr,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    overflow_len,
    state_len,
    HEAD_DIM: tl.constexpr,
    SINK_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    token_block = tl.program_id(2).to(tl.int64)
    token = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    token_valid = token < overflow_len
    dim = tl.arange(0, HEAD_DIM)
    overflow = tl.load(
        overflow_k
        + batch * OVERFLOW_BATCH_STRIDE
        + head * OVERFLOW_HEAD_STRIDE
        + token[:, None] * OVERFLOW_TOKEN_STRIDE
        + dim[None, :],
        mask=token_valid[:, None],
        other=0.0,
    )

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
            other=1.0,
        ).to(tl.float32)
        key = tl.load(
            state_k
            + batch * STATE_BATCH_STRIDE
            + head * STATE_HEAD_STRIDE
            + slot[:, None] * STATE_TOKEN_STRIDE
            + dim[None, :],
            mask=slot_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        mean_key = (key / count[:, None]).to(overflow.dtype)
        scores = tl.dot(overflow, tl.trans(mean_key), out_dtype=tl.float32)
        # torch.matmul returns BF16 for the reference path.  Preserve that
        # score precision while avoiding its overflow-by-state materialization.
        scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(
            token_valid[:, None] & slot_valid[None, :],
            scores,
            -float("inf"),
        )
        best_select_score = tl.maximum(
            best_select_score, tl.max(scores, axis=1)
        )

        route_valid = slot_valid & (slot >= SINK_LEN)
        route_candidate = tl.where(
            token_valid[:, None] & route_valid[None, :],
            scores,
            -float("inf"),
        )
        local_score = tl.max(route_candidate, axis=1)
        local_index = tl.min(
            tl.where(
                route_candidate == local_score[:, None],
                slot[None, :],
                state_len,
            ),
            axis=1,
        ).to(tl.int32)
        take_local = local_score > best_route_score
        best_route_score = tl.where(
            take_local, local_score, best_route_score
        )
        best_route_index = tl.where(
            take_local, local_index, best_route_index
        )

    output_offset = (
        batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + token * OUTPUT_TOKEN_STRIDE
    )
    tl.store(route_scores + output_offset, best_route_score, mask=token_valid)
    tl.store(route_indices + output_offset, best_route_index, mask=token_valid)
    tl.store(select_scores + output_offset, best_select_score, mask=token_valid)


@triton.jit
def _route_logits_coarse_attention_kernel(
    q,
    route_logits,
    state_v,
    counts,
    local_k,
    local_v,
    top_slots,
    output,
    lse,
    Q_BATCH_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_TOKEN_STRIDE: tl.constexpr,
    LOGIT_BATCH_STRIDE: tl.constexpr,
    LOGIT_HEAD_STRIDE: tl.constexpr,
    LOGIT_QUERY_STRIDE: tl.constexpr,
    LOGIT_STATE_STRIDE: tl.constexpr,
    STATE_V_BATCH_STRIDE: tl.constexpr,
    STATE_V_HEAD_STRIDE: tl.constexpr,
    STATE_V_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    LOCAL_K_BATCH_STRIDE: tl.constexpr,
    LOCAL_K_HEAD_STRIDE: tl.constexpr,
    LOCAL_K_TOKEN_STRIDE: tl.constexpr,
    LOCAL_V_BATCH_STRIDE: tl.constexpr,
    LOCAL_V_HEAD_STRIDE: tl.constexpr,
    LOCAL_V_TOKEN_STRIDE: tl.constexpr,
    TOP_BATCH_STRIDE: tl.constexpr,
    TOP_HEAD_STRIDE: tl.constexpr,
    TOP_QUERY_STRIDE: tl.constexpr,
    query_len,
    state_len,
    local_len,
    local_offset,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Stream the coarse softmax while reusing precomputed route logits."""
    batch = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    query_block = tl.program_id(2).to(tl.int64)
    row = tl.arange(0, KV_GROUP_SIZE * BLOCK_M)
    group_head = row // BLOCK_M
    query = query_block * BLOCK_M + row % BLOCK_M
    query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = query < query_len
    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)

    queries = tl.load(
        q
        + batch * Q_BATCH_STRIDE
        + query_head[:, None] * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((KV_GROUP_SIZE * BLOCK_M, HEAD_DIM), tl.float32)

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
            + dim[None, :],
            mask=state_valid[:, None],
            other=0.0,
        )
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
        scores = scores * SCALE + tl.log(count)[None, :]
        routed = tl.zeros(
            (KV_GROUP_SIZE * BLOCK_M, BLOCK_N), dtype=tl.int1
        )
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
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(mean_values.dtype),
            mean_values,
            out_dtype=tl.float32,
        )
        maximum = new_maximum

    for local_begin in tl.range(0, local_len, BLOCK_N, num_stages=1):
        token = local_begin + token_offset
        token_valid = token < local_len
        keys = tl.load(
            local_k
            + batch * LOCAL_K_BATCH_STRIDE
            + kv_head * LOCAL_K_HEAD_STRIDE
            + token[:, None] * LOCAL_K_TOKEN_STRIDE
            + dim[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        values = tl.load(
            local_v
            + batch * LOCAL_V_BATCH_STRIDE
            + kv_head * LOCAL_V_HEAD_STRIDE
            + token[:, None] * LOCAL_V_TOKEN_STRIDE
            + dim[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        scores = SCALE * tl.dot(
            queries, tl.trans(keys), out_dtype=tl.float32
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

    output_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query
    ).to(tl.int64)
    tl.store(
        output + output_row[:, None] * HEAD_DIM + dim[None, :],
        accumulator / denominator[:, None],
        mask=query_valid[:, None],
    )
    tl.store(
        lse + output_row,
        maximum + tl.log(denominator),
        mask=query_valid,
    )


@triton.jit
def _route_logits_topk_coarse_attention_kernel(
    q,
    route_logits,
    state_v,
    counts,
    local_k,
    local_v,
    top_slots,
    output,
    lse,
    Q_BATCH_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_TOKEN_STRIDE: tl.constexpr,
    LOGIT_BATCH_STRIDE: tl.constexpr,
    LOGIT_HEAD_STRIDE: tl.constexpr,
    LOGIT_QUERY_STRIDE: tl.constexpr,
    LOGIT_STATE_STRIDE: tl.constexpr,
    STATE_V_BATCH_STRIDE: tl.constexpr,
    STATE_V_HEAD_STRIDE: tl.constexpr,
    STATE_V_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    LOCAL_K_BATCH_STRIDE: tl.constexpr,
    LOCAL_K_HEAD_STRIDE: tl.constexpr,
    LOCAL_K_TOKEN_STRIDE: tl.constexpr,
    LOCAL_V_BATCH_STRIDE: tl.constexpr,
    LOCAL_V_HEAD_STRIDE: tl.constexpr,
    LOCAL_V_TOKEN_STRIDE: tl.constexpr,
    TOP_BATCH_STRIDE: tl.constexpr,
    TOP_HEAD_STRIDE: tl.constexpr,
    TOP_QUERY_STRIDE: tl.constexpr,
    query_len,
    state_len,
    local_len,
    local_offset,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Select routes while streaming the complete coarse softmax once."""
    batch = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1).to(tl.int64)
    query_block = tl.program_id(2).to(tl.int64)
    row = tl.arange(0, KV_GROUP_SIZE * BLOCK_M)
    group_head = row // BLOCK_M
    query = query_block * BLOCK_M + row % BLOCK_M
    query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = query < query_len
    dim = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    route_rank = tl.arange(0, ROUTE_COUNT)

    queries = tl.load(
        q
        + batch * Q_BATCH_STRIDE
        + query_head[:, None] * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
    maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((KV_GROUP_SIZE * BLOCK_M, HEAD_DIM), tl.float32)
    top_scores = tl.full(
        (KV_GROUP_SIZE * BLOCK_M, ROUTE_COUNT),
        -float("inf"),
        tl.float32,
    )
    top_indices = tl.full(
        (KV_GROUP_SIZE * BLOCK_M, ROUTE_COUNT), -1, tl.int32
    )

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
            + dim[None, :],
            mask=state_valid[:, None],
            other=0.0,
        )
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
        scores = scores * SCALE + tl.log(count)[None, :]
        valid = query_valid[:, None] & state_valid[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        remaining_scores = scores
        for _ in tl.static_range(0, ROUTE_COUNT):
            candidate_score = tl.max(remaining_scores, axis=1)
            candidate_position = tl.min(
                tl.where(
                    remaining_scores == candidate_score[:, None],
                    token_offset[None, :],
                    BLOCK_N,
                ),
                axis=1,
            )
            worst_score = tl.min(top_scores, axis=1)
            worst_rank = tl.min(
                tl.where(
                    top_scores == worst_score[:, None],
                    route_rank[None, :],
                    ROUTE_COUNT,
                ),
                axis=1,
            )
            replace = query_valid & (candidate_score > worst_score)
            replace_at = replace[:, None] & (
                route_rank[None, :] == worst_rank[:, None]
            )
            top_scores = tl.where(
                replace_at, candidate_score[:, None], top_scores
            )
            top_indices = tl.where(
                replace_at,
                (state_begin + candidate_position)[:, None],
                top_indices,
            )
            remaining_scores = tl.where(
                token_offset[None, :] == candidate_position[:, None],
                -float("inf"),
                remaining_scores,
            )

        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - new_maximum)
        probabilities = tl.exp(scores - new_maximum[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(mean_values.dtype),
            mean_values,
            out_dtype=tl.float32,
        )
        maximum = new_maximum

    for local_begin in tl.range(0, local_len, BLOCK_N, num_stages=1):
        token = local_begin + token_offset
        token_valid = token < local_len
        keys = tl.load(
            local_k
            + batch * LOCAL_K_BATCH_STRIDE
            + kv_head * LOCAL_K_HEAD_STRIDE
            + token[:, None] * LOCAL_K_TOKEN_STRIDE
            + dim[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        values = tl.load(
            local_v
            + batch * LOCAL_V_BATCH_STRIDE
            + kv_head * LOCAL_V_HEAD_STRIDE
            + token[:, None] * LOCAL_V_TOKEN_STRIDE
            + dim[None, :],
            mask=token_valid[:, None],
            other=0.0,
        )
        scores = SCALE * tl.dot(
            queries, tl.trans(keys), out_dtype=tl.float32
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

    # The streamed field included every low-resolution state summary. Remove
    # the selected summaries so their exact leaves can replace them without
    # duplicate attention mass.
    for route in tl.static_range(0, ROUTE_COUNT):
        selected_slot = tl.max(
            tl.where(
                route_rank[None, :] == route,
                top_indices,
                -1,
            ),
            axis=1,
        ).to(tl.int64)
        selected_score = tl.max(
            tl.where(
                route_rank[None, :] == route,
                top_scores,
                -float("inf"),
            ),
            axis=1,
        )
        selected_count = tl.load(
            counts
            + batch * COUNT_BATCH_STRIDE
            + kv_head * COUNT_HEAD_STRIDE
            + selected_slot * COUNT_TOKEN_STRIDE,
            mask=query_valid,
            other=1.0,
        ).to(tl.float32)
        selected_values = tl.load(
            state_v
            + batch * STATE_V_BATCH_STRIDE
            + kv_head * STATE_V_HEAD_STRIDE
            + selected_slot[:, None] * STATE_V_TOKEN_STRIDE
            + dim[None, :],
            mask=query_valid[:, None],
            other=0.0,
        )
        selected_mean = (
            selected_values.to(tl.float32) / selected_count[:, None]
        ).to(selected_values.dtype)
        selected_weight = tl.exp(selected_score - maximum)
        denominator -= selected_weight
        accumulator -= (
            selected_weight.to(selected_mean.dtype).to(tl.float32)[:, None]
            * selected_mean.to(tl.float32)
        )

    output_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query
    ).to(tl.int64)
    tl.store(
        output + output_row[:, None] * HEAD_DIM + dim[None, :],
        accumulator / denominator[:, None],
        mask=query_valid[:, None],
    )
    tl.store(
        lse + output_row,
        maximum + tl.log(denominator),
        mask=query_valid,
    )
    tl.store(
        top_slots
        + batch * TOP_BATCH_STRIDE
        + query_head[:, None] * TOP_HEAD_STRIDE
        + query[:, None] * TOP_QUERY_STRIDE
        + route_rank[None, :],
        top_indices,
        mask=query_valid[:, None],
    )


@triton.jit
def _accumulate_state_deltas_kernel(
    merge_k,
    merge_v,
    merge_indices,
    destinations,
    owners,
    delta_k,
    delta_v,
    delta_counts,
    touched,
    MERGE_K_ROW_STRIDE,
    MERGE_V_ROW_STRIDE,
    OWNER_ROW_STRIDE,
    DELTA_K_ROW_STRIDE,
    DELTA_V_ROW_STRIDE,
    DELTA_SLOT_STRIDE,
    TOKENS: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token_block = tl.program_id(1).to(tl.int64)
    token = token_block * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    valid = token < TOKENS
    key_dim = tl.arange(0, HEAD_DIM)
    value_dim = tl.arange(0, VALUE_DIM)

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
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        merge_v
        + row * MERGE_V_ROW_STRIDE
        + token[:, None] * VALUE_DIM
        + value_dim[None, :],
        mask=valid[:, None],
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
        1.0,
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
        mask=valid[:, None],
    )
    tl.atomic_add(
        delta_v
        + row * DELTA_V_ROW_STRIDE
        + destination[:, None] * VALUE_DIM
        + value_dim[None, :],
        v,
        sem="relaxed",
        mask=valid[:, None],
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
):
    row = tl.program_id(0).to(tl.int64)
    state_block = tl.program_id(1).to(tl.int64)
    slot = state_block * STATE_BLOCK + tl.arange(0, STATE_BLOCK)
    valid = slot < active_slots
    is_touched = (
        tl.load(touched + row * DELTA_SLOT_STRIDE + slot, mask=valid, other=0) != 0
    )
    update = valid & is_touched
    key_dim = tl.arange(0, HEAD_DIM)
    value_dim = tl.arange(0, VALUE_DIM)

    old_k = tl.load(
        state_k
        + row * STATE_K_ROW_STRIDE
        + slot[:, None] * STATE_K_SLOT_STRIDE
        + key_dim[None, :],
        mask=update[:, None],
        other=0.0,
    ).to(tl.float32)
    old_v = tl.load(
        state_v
        + row * STATE_V_ROW_STRIDE
        + slot[:, None] * STATE_V_SLOT_STRIDE
        + value_dim[None, :],
        mask=update[:, None],
        other=0.0,
    ).to(tl.float32)
    add_k = tl.load(
        delta_k
        + row * DELTA_K_ROW_STRIDE
        + slot[:, None] * HEAD_DIM
        + key_dim[None, :],
        mask=update[:, None],
        other=0.0,
    )
    add_v = tl.load(
        delta_v
        + row * DELTA_V_ROW_STRIDE
        + slot[:, None] * VALUE_DIM
        + value_dim[None, :],
        mask=update[:, None],
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
        mask=update[:, None],
    )
    tl.store(
        state_v
        + row * STATE_V_ROW_STRIDE
        + slot[:, None] * STATE_V_SLOT_STRIDE
        + value_dim[None, :],
        old_v + add_v,
        mask=update[:, None],
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
        mask=update[:, None],
    )
    tl.store(
        delta_v
        + row * DELTA_V_ROW_STRIDE
        + slot[:, None] * VALUE_DIM
        + value_dim[None, :],
        0.0,
        mask=update[:, None],
    )
    tl.store(delta_counts + row * DELTA_SLOT_STRIDE + slot, 0.0, mask=update)
    tl.store(touched + row * DELTA_SLOT_STRIDE + slot, 0, mask=update)


@triton.jit
def _route_state_group_candidates_kernel(
    q,
    state_k,
    counts,
    partial_scores,
    partial_indices,
    Q_BATCH_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_TOKEN_STRIDE: tl.constexpr,
    STATE_BATCH_STRIDE,
    STATE_HEAD_STRIDE,
    STATE_TOKEN_STRIDE,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    PARTIAL_BATCH_STRIDE: tl.constexpr,
    PARTIAL_HEAD_STRIDE: tl.constexpr,
    PARTIAL_QUERY_STRIDE: tl.constexpr,
    PARTIAL_GROUP_STRIDE: tl.constexpr,
    active_groups,
    query_len,
    state_len,
    SCALE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    q_head = tl.program_id(1).to(tl.int64)
    query_group = tl.program_id(2).to(tl.int64)
    query_block = query_group // active_groups
    state_group = query_group - query_block * active_groups
    kv_head = q_head // KV_GROUP_SIZE
    query = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_offset = tl.arange(0, BLOCK_N)
    slot = state_group * BLOCK_N + slot_offset
    query_valid = query < query_len
    slot_valid = slot < state_len
    dim = tl.arange(0, HEAD_DIM)

    q_values = tl.load(
        q
        + batch * Q_BATCH_STRIDE
        + q_head * Q_HEAD_STRIDE
        + query[:, None] * Q_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None],
        other=0.0,
    )
    count = tl.load(
        counts
        + batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=slot_valid,
        other=1.0,
    ).to(tl.float32)
    key = tl.load(
        state_k
        + batch * STATE_BATCH_STRIDE
        + kv_head * STATE_HEAD_STRIDE
        + slot[:, None] * STATE_TOKEN_STRIDE
        + dim[None, :],
        mask=slot_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    key = (key / count[:, None]).to(q_values.dtype)
    scores = tl.dot(q_values, tl.trans(key), out_dtype=tl.float32)
    scores = (scores.to(tl.bfloat16) * SCALE).to(tl.bfloat16).to(tl.float32)
    scores += tl.log(count)[None, :]
    scores = tl.where(query_valid[:, None] & slot_valid[None, :], scores, -float("inf"))

    partial_base = (
        batch * PARTIAL_BATCH_STRIDE
        + q_head * PARTIAL_HEAD_STRIDE
        + query * PARTIAL_QUERY_STRIDE
        + state_group * PARTIAL_GROUP_STRIDE
    )
    for rank in tl.static_range(0, 8):
        best_score = tl.max(scores, axis=1)
        best_position = tl.min(
            tl.where(
                scores == best_score[:, None],
                slot_offset[None, :],
                BLOCK_N,
            ),
            axis=1,
        )
        best_slot = state_group * BLOCK_N + best_position
        tl.store(
            partial_scores + partial_base + rank,
            best_score,
            mask=query_valid,
        )
        tl.store(
            partial_indices + partial_base + rank,
            best_slot,
            mask=query_valid,
        )
        scores = tl.where(
            slot_offset[None, :] == best_position[:, None],
            -float("inf"),
            scores,
        )


@triton.jit
def _route_score_group_candidates_kernel(
    logits,
    counts,
    partial_scores,
    partial_indices,
    partial_lse,
    LOGIT_BATCH_STRIDE: tl.constexpr,
    LOGIT_HEAD_STRIDE: tl.constexpr,
    LOGIT_QUERY_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE,
    COUNT_HEAD_STRIDE,
    COUNT_TOKEN_STRIDE,
    PARTIAL_BATCH_STRIDE: tl.constexpr,
    PARTIAL_HEAD_STRIDE: tl.constexpr,
    PARTIAL_QUERY_STRIDE: tl.constexpr,
    PARTIAL_GROUP_STRIDE: tl.constexpr,
    PARTIAL_LSE_BATCH_STRIDE: tl.constexpr,
    PARTIAL_LSE_HEAD_STRIDE: tl.constexpr,
    PARTIAL_LSE_QUERY_STRIDE: tl.constexpr,
    PARTIAL_LSE_GROUP_STRIDE: tl.constexpr,
    active_groups,
    query_len,
    state_len,
    SCALE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    q_head = tl.program_id(1).to(tl.int64)
    query_group = tl.program_id(2).to(tl.int64)
    query_block = query_group // active_groups
    state_group = query_group - query_block * active_groups
    kv_head = q_head // KV_GROUP_SIZE
    query = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    slot_offset = tl.arange(0, BLOCK_N)
    slot = state_group * BLOCK_N + slot_offset
    query_valid = query < query_len
    slot_valid = slot < state_len

    count = tl.load(
        counts
        + batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slot * COUNT_TOKEN_STRIDE,
        mask=slot_valid,
        other=1.0,
    ).to(tl.float32)
    scores = tl.load(
        logits
        + batch * LOGIT_BATCH_STRIDE
        + q_head * LOGIT_HEAD_STRIDE
        + query[:, None] * LOGIT_QUERY_STRIDE
        + slot[None, :],
        mask=query_valid[:, None] & slot_valid[None, :],
        other=-float("inf"),
    )
    # Preserve the eager reference's BF16 matmul output and BF16 scale
    # rounding.  Only the count correction promotes the scores to FP32.
    scores = (scores.to(tl.bfloat16) * SCALE).to(tl.bfloat16).to(tl.float32)
    scores += tl.log(count)[None, :]
    scores = tl.where(query_valid[:, None] & slot_valid[None, :], scores, -float("inf"))

    if STORE_LSE:
        group_max = tl.max(scores, axis=1)
        group_sum = tl.sum(tl.exp(scores - group_max[:, None]), axis=1)
        group_lse = group_max + tl.log(group_sum)
        tl.store(
            partial_lse
            + batch * PARTIAL_LSE_BATCH_STRIDE
            + q_head * PARTIAL_LSE_HEAD_STRIDE
            + query * PARTIAL_LSE_QUERY_STRIDE
            + state_group * PARTIAL_LSE_GROUP_STRIDE,
            group_lse,
            mask=query_valid,
        )

    partial_base = (
        batch * PARTIAL_BATCH_STRIDE
        + q_head * PARTIAL_HEAD_STRIDE
        + query * PARTIAL_QUERY_STRIDE
        + state_group * PARTIAL_GROUP_STRIDE
    )
    for rank in tl.static_range(0, 8):
        best_score = tl.max(scores, axis=1)
        best_position = tl.min(
            tl.where(
                scores == best_score[:, None],
                slot_offset[None, :],
                BLOCK_N,
            ),
            axis=1,
        )
        best_slot = state_group * BLOCK_N + best_position
        tl.store(
            partial_scores + partial_base + rank,
            best_score,
            mask=query_valid,
        )
        tl.store(
            partial_indices + partial_base + rank,
            best_slot,
            mask=query_valid,
        )
        scores = tl.where(
            slot_offset[None, :] == best_position[:, None],
            -float("inf"),
            scores,
        )


@triton.jit
def _reduce_route_group_candidates_kernel(
    partial_scores,
    partial_indices,
    partial_lse,
    output,
    state_lse,
    PARTIAL_BATCH_STRIDE: tl.constexpr,
    PARTIAL_HEAD_STRIDE: tl.constexpr,
    PARTIAL_QUERY_STRIDE: tl.constexpr,
    PARTIAL_LSE_BATCH_STRIDE: tl.constexpr,
    PARTIAL_LSE_HEAD_STRIDE: tl.constexpr,
    PARTIAL_LSE_QUERY_STRIDE: tl.constexpr,
    PARTIAL_LSE_GROUP_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE: tl.constexpr,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    STATE_LSE_BATCH_STRIDE: tl.constexpr,
    STATE_LSE_HEAD_STRIDE: tl.constexpr,
    STATE_LSE_QUERY_STRIDE: tl.constexpr,
    query_len,
    active_groups,
    TOPK: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    q_head = tl.program_id(1).to(tl.int64)
    query_block = tl.program_id(2).to(tl.int64)
    query = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    query_valid = query < query_len
    candidate_valid = candidate < active_groups * 8
    partial_offset = (
        batch * PARTIAL_BATCH_STRIDE
        + q_head * PARTIAL_HEAD_STRIDE
        + query[:, None] * PARTIAL_QUERY_STRIDE
        + candidate[None, :]
    )
    scores = tl.load(
        partial_scores + partial_offset,
        mask=query_valid[:, None] & candidate_valid[None, :],
        other=-float("inf"),
    )
    indices = tl.load(
        partial_indices + partial_offset,
        mask=query_valid[:, None] & candidate_valid[None, :],
        other=-1,
    ).to(tl.int64)
    output_base = (
        output
        + batch * OUTPUT_BATCH_STRIDE
        + q_head * OUTPUT_HEAD_STRIDE
        + query * OUTPUT_TOKEN_STRIDE
    )
    for rank in tl.static_range(0, 8):
        best_score = tl.max(scores, axis=1)
        best_position = tl.min(
            tl.where(
                scores == best_score[:, None],
                candidate[None, :],
                CANDIDATE_BLOCK,
            ),
            axis=1,
        )
        best_index = tl.max(
            tl.where(
                candidate[None, :] == best_position[:, None],
                indices,
                -1,
            ),
            axis=1,
        )
        if rank < TOPK:
            tl.store(output_base + rank, best_index, mask=query_valid)
        scores = tl.where(
            candidate[None, :] == best_position[:, None],
            -float("inf"),
            scores,
        )
    if STORE_LSE:
        group = tl.arange(0, MAX_GROUPS)
        group_values = tl.load(
            partial_lse
            + batch * PARTIAL_LSE_BATCH_STRIDE
            + q_head * PARTIAL_LSE_HEAD_STRIDE
            + query[:, None] * PARTIAL_LSE_QUERY_STRIDE
            + group[None, :] * PARTIAL_LSE_GROUP_STRIDE,
            mask=query_valid[:, None] & (group[None, :] < active_groups),
            other=-float("inf"),
        )
        group_max = tl.max(group_values, axis=1)
        group_sum = tl.sum(tl.exp(group_values - group_max[:, None]), axis=1)
        tl.store(
            state_lse
            + batch * STATE_LSE_BATCH_STRIDE
            + q_head * STATE_LSE_HEAD_STRIDE
            + query * STATE_LSE_QUERY_STRIDE,
            group_max + tl.log(group_sum),
            mask=query_valid,
        )


@triton.jit
def _reorder_topk_like_torch_kernel(
    output,
    OUTPUT_BATCH_STRIDE: tl.constexpr,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    query = tl.program_id(2).to(tl.int64)
    base = (
        output
        + batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + query * OUTPUT_TOKEN_STRIDE
    )
    rank = tl.arange(0, 8)
    selected = tl.load(
        base + rank,
        mask=rank < TOPK,
        other=-1,
    )
    boundary = tl.max(tl.where(rank == TOPK - 1, selected, -1), axis=0)
    remaining = tl.where(rank < TOPK - 1, selected, 0x7FFFFFFF)
    for output_rank in tl.static_range(0, 8):
        if output_rank < TOPK - 1:
            best = tl.min(remaining, axis=0)
            tl.store(base + output_rank, best)
            remaining = tl.where(remaining == best, 0x7FFFFFFF, remaining)
    if TOPK > 0:
        tl.store(
            base + TOPK - 1,
            boundary,
        )


@triton.jit
def _apply_residual_mass_opening_kernel(
    logits,
    counts,
    top_slots,
    state_lse,
    local_lse,
    LOGIT_BATCH_STRIDE: tl.constexpr,
    LOGIT_HEAD_STRIDE: tl.constexpr,
    LOGIT_QUERY_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    TOP_BATCH_STRIDE: tl.constexpr,
    TOP_HEAD_STRIDE: tl.constexpr,
    TOP_QUERY_STRIDE: tl.constexpr,
    LSE_BATCH_STRIDE: tl.constexpr,
    LSE_HEAD_STRIDE: tl.constexpr,
    LSE_QUERY_STRIDE: tl.constexpr,
    LOCAL_LSE_BATCH_STRIDE: tl.constexpr,
    LOCAL_LSE_HEAD_STRIDE: tl.constexpr,
    LOCAL_LSE_QUERY_STRIDE: tl.constexpr,
    query_len,
    state_len,
    RESIDUAL_MASS: tl.constexpr,
    SCALE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    q_head = tl.program_id(1).to(tl.int64)
    query_block = tl.program_id(2).to(tl.int64)
    query = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    rank = tl.arange(0, 8)
    query_valid = query < query_len
    top_offset = (
        batch * TOP_BATCH_STRIDE
        + q_head * TOP_HEAD_STRIDE
        + query[:, None] * TOP_QUERY_STRIDE
        + rank[None, :]
    )
    slots = tl.load(
        top_slots + top_offset,
        mask=query_valid[:, None],
        other=0,
    ).to(tl.int64)
    slot_valid = (slots >= 0) & (slots < state_len)
    kv_head = q_head // KV_GROUP_SIZE
    selected_logits = tl.load(
        logits
        + batch * LOGIT_BATCH_STRIDE
        + q_head * LOGIT_HEAD_STRIDE
        + query[:, None] * LOGIT_QUERY_STRIDE
        + slots,
        mask=query_valid[:, None] & slot_valid,
        other=-float("inf"),
    )
    selected_counts = tl.load(
        counts
        + batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + slots * COUNT_TOKEN_STRIDE,
        mask=query_valid[:, None] & slot_valid,
        other=1.0,
    ).to(tl.float32)
    scores = (
        (selected_logits.to(tl.bfloat16) * SCALE)
        .to(tl.bfloat16)
        .to(tl.float32)
        + tl.log(selected_counts)
    )
    lse_offset = (
        batch * LSE_BATCH_STRIDE
        + q_head * LSE_HEAD_STRIDE
        + query * LSE_QUERY_STRIDE
    )
    remote_lse = tl.load(state_lse + lse_offset, mask=query_valid, other=0.0)
    local_lse_offset = (
        batch * LOCAL_LSE_BATCH_STRIDE
        + q_head * LOCAL_LSE_HEAD_STRIDE
        + query * LOCAL_LSE_QUERY_STRIDE
    )
    exact_lse = tl.load(
        local_lse + local_lse_offset, mask=query_valid, other=0.0
    )
    maximum_lse = tl.maximum(remote_lse, exact_lse)
    full_lse = maximum_lse + tl.log(
        tl.exp(remote_lse - maximum_lse) + tl.exp(exact_lse - maximum_lse)
    )
    masses = tl.exp(scores - full_lse[:, None])
    remaining_mass = tl.sum(masses, axis=1)

    for output_rank in tl.static_range(0, 8):
        best_score = tl.max(scores, axis=1)
        best_position = tl.min(
            tl.where(
                scores == best_score[:, None],
                rank[None, :],
                8,
            ),
            axis=1,
        )
        best_slot = tl.max(
            tl.where(
                rank[None, :] == best_position[:, None],
                slots,
                -1,
            ),
            axis=1,
        )
        best_mass = tl.max(
            tl.where(
                rank[None, :] == best_position[:, None],
                masses,
                0.0,
            ),
            axis=1,
        )
        should_open = (output_rank == 0) | (remaining_mass > RESIDUAL_MASS)
        tl.store(
            top_slots
            + batch * TOP_BATCH_STRIDE
            + q_head * TOP_HEAD_STRIDE
            + query * TOP_QUERY_STRIDE
            + output_rank,
            tl.where(should_open, best_slot, -1),
            mask=query_valid,
        )
        remaining_mass -= best_mass
        scores = tl.where(
            rank[None, :] == best_position[:, None],
            -float("inf"),
            scores,
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
    }


def streaming_state_maxsim(
    overflow_k: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    state_len: int,
    sink_len: int,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return old-state route and append scores without a dense score matrix."""
    if not all(tensor.is_cuda for tensor in (overflow_k, state_k, counts)):
        raise ValueError("streaming LOD state routing requires CUDA tensors")
    batch, kv_heads, overflow_len, head_dim = overflow_k.shape
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
    _streaming_state_maxsim_kernel[
        (batch, kv_heads, triton.cdiv(overflow_len, block_m))
    ](
        overflow_k,
        state_k,
        counts,
        route_scores,
        route_indices,
        select_scores,
        OVERFLOW_BATCH_STRIDE=overflow_k.stride(0),
        OVERFLOW_HEAD_STRIDE=overflow_k.stride(1),
        OVERFLOW_TOKEN_STRIDE=overflow_k.stride(2),
        STATE_BATCH_STRIDE=state_k.stride(0),
        STATE_HEAD_STRIDE=state_k.stride(1),
        STATE_TOKEN_STRIDE=state_k.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0),
        COUNT_HEAD_STRIDE=counts.stride(1),
        COUNT_TOKEN_STRIDE=counts.stride(2),
        OUTPUT_BATCH_STRIDE=route_scores.stride(0),
        OUTPUT_HEAD_STRIDE=route_scores.stride(1),
        OUTPUT_TOKEN_STRIDE=route_scores.stride(2),
        overflow_len=overflow_len,
        state_len=state_len,
        HEAD_DIM=head_dim,
        SINK_LEN=sink_len,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **_launch_kwargs(num_warps),
    )
    active = (..., slice(None, overflow_len))
    return (
        route_scores[active],
        route_indices[active],
        select_scores[active],
    )


def new_route_buffers(
    q: torch.Tensor,
    *,
    state_capacity: int,
    query_capacity: int = 256,
    include_lse: bool = False,
) -> dict[str, torch.Tensor]:
    batch, q_heads = q.shape[:2]
    query_capacity = max(query_capacity, int(q.size(2)))
    max_groups = triton.cdiv(state_capacity, 64)
    return {
        "partial_scores": torch.empty(
            batch,
            q_heads,
            query_capacity,
            max_groups,
            8,
            dtype=torch.float32,
            device=q.device,
        ),
        "partial_indices": torch.empty(
            batch,
            q_heads,
            query_capacity,
            max_groups,
            8,
            dtype=torch.long,
            device=q.device,
        ),
        "partial_lse": torch.empty(
            (batch, q_heads, query_capacity, max_groups)
            if include_lse
            else (1, 1, 1, 1),
            dtype=torch.float32,
            device=q.device,
        ),
        "state_lse": torch.empty(
            (batch, q_heads, query_capacity) if include_lse else (1, 1, 1),
            dtype=torch.float32,
            device=q.device,
        ),
        "output": torch.empty(
            batch,
            q_heads,
            query_capacity,
            8,
            dtype=torch.long,
            device=q.device,
        ),
    }


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the coarse state/local branch while reusing routing logits."""
    tensors = (q, route_logits, state_v, counts, local_k, local_v, top_slots)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("LOD Triton coarse attention requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("LOD Triton coarse attention requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(state_v.size(1))
    local_len = int(local_k.size(2))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query heads do not match the requested GQA grouping")
    if tuple(route_logits.shape) != (batch, query_heads, query_len, state_len):
        raise ValueError("routing logits have the wrong shape")
    if tuple(top_slots.shape[:3]) != (batch, query_heads, query_len):
        raise ValueError("top-slot routes have the wrong shape")
    if int(top_slots.size(-1)) > 8:
        raise ValueError("the fused coarse kernel supports at most eight routes")
    if state_len > int(state_v.size(2)) or state_len > int(counts.size(2)):
        raise ValueError("active state exceeds the supplied storage")
    if local_len != 0 and local_len < query_len:
        raise ValueError("local attention must contain every current query token")
    if int(local_v.size(2)) != local_len:
        raise ValueError("local key/value lengths differ")
    if int(local_k.size(1)) != kv_heads or int(local_v.size(1)) != kv_heads:
        raise ValueError("local and state KV heads differ")
    if int(local_k.size(-1)) != head_dim or int(local_v.size(-1)) != head_dim:
        raise ValueError("the fused coarse kernel requires equal Q/K/V head sizes")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("coarse-attention tile sizes must be positive")

    output = torch.empty_like(q)
    lse = torch.empty(
        batch,
        query_heads,
        query_len,
        dtype=torch.float32,
        device=q.device,
    )
    grid = (batch, kv_heads, triton.cdiv(query_len, block_m))
    _route_logits_coarse_attention_kernel[grid](
        q,
        route_logits,
        state_v,
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
        state_v.stride(0),
        state_v.stride(1),
        state_v.stride(2),
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
        HEAD_DIM=head_dim,
        ROUTE_COUNT=int(top_slots.size(-1)),
        SCALE=scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **_launch_kwargs(num_warps),
    )
    return output, lse


def route_logits_topk_coarse_attention(
    q: torch.Tensor,
    route_logits: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    *,
    state_len: int,
    kv_group_size: int,
    scale: float,
    topk: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select top-k routes and form their coarse remainder in one scan."""
    tensors = (q, route_logits, state_v, counts, local_k, local_v)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused LOD prefill routing requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused LOD prefill routing requires contiguous tensors")
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(state_v.size(1))
    local_len = int(local_k.size(2))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query heads do not match the requested GQA grouping")
    if tuple(route_logits.shape) != (batch, query_heads, query_len, state_len):
        raise ValueError("routing logits have the wrong shape")
    if topk != 8:
        raise ValueError("fused LOD prefill routing currently requires top-8")
    if state_len < topk:
        raise ValueError("active state is smaller than the requested route count")
    if state_len > int(state_v.size(2)) or state_len > int(counts.size(2)):
        raise ValueError("active state exceeds the supplied storage")
    if local_len < query_len or int(local_v.size(2)) != local_len:
        raise ValueError("local attention has an invalid length")
    if int(local_k.size(1)) != kv_heads or int(local_v.size(1)) != kv_heads:
        raise ValueError("local and state KV heads differ")
    if int(local_k.size(-1)) != head_dim or int(local_v.size(-1)) != head_dim:
        raise ValueError("fused LOD prefill requires equal Q/K/V head sizes")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("coarse-attention tile sizes must be positive")

    top_slots = torch.empty(
        batch,
        query_heads,
        query_len,
        topk,
        dtype=torch.long,
        device=q.device,
    )
    output = torch.empty_like(q)
    lse = torch.empty(
        batch,
        query_heads,
        query_len,
        dtype=torch.float32,
        device=q.device,
    )
    grid = (batch, kv_heads, triton.cdiv(query_len, block_m))
    _route_logits_topk_coarse_attention_kernel[grid](
        q,
        route_logits,
        state_v,
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
        state_v.stride(0),
        state_v.stride(1),
        state_v.stride(2),
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
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        ROUTE_COUNT=topk,
        SCALE=scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **_launch_kwargs(num_warps),
    )
    return top_slots, output, lse


def merge_state_in_place(
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    merge_k: torch.Tensor,
    merge_v: torch.Tensor,
    merge_indices: torch.Tensor,
    destinations: torch.Tensor,
    owners: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    active_slots: int | None = None,
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
    capacity = int(buffers["touched"].size(2))
    if active_slots is None:
        active_slots = int(state_k.size(2))
    if active_slots > int(state_k.size(2)) or active_slots > capacity:
        raise ValueError("LOD state delta capacity is smaller than the active state")
    # Keep each atomic tile at 1024 lanes, matching the proven KVM update
    # shape. A 256-wide head therefore uses four tokens per program.
    token_block = 4
    _accumulate_state_deltas_kernel[(rows, triton.cdiv(tokens, token_block))](
        merge_k,
        merge_v,
        merge_indices,
        destinations,
        owners,
        buffers["delta_k"],
        buffers["delta_v"],
        buffers["delta_counts"],
        buffers["touched"],
        merge_k.stride(1),
        merge_v.stride(1),
        owners.stride(1),
        buffers["delta_k"].stride(1),
        buffers["delta_v"].stride(1),
        buffers["touched"].stride(1),
        TOKENS=tokens,
        TOKEN_BLOCK=token_block,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        **_launch_kwargs(8),
    )
    # The KVM apply kernel uses an 8x128 tile. Preserve the same 1024-lane
    # footprint for a 256-wide state rather than doubling register use.
    state_block = 4
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
        **_launch_kwargs(2),
    )


def route_top8_state_grouped(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    kv_group_size: int,
    scale: float,
    topk: int,
    state_len: int | None = None,
    reorder_like_torch: bool = True,
) -> torch.Tensor:
    if not 1 <= topk <= 8:
        raise ValueError("grouped LOD routing supports topk from 1 through 8")
    if not q.is_cuda or not state_k.is_cuda or not counts.is_cuda:
        raise ValueError("grouped LOD routing requires CUDA tensors")
    batch, q_heads, query_len, head_dim = q.shape
    if state_len is None:
        state_len = int(state_k.size(2))
    if state_len > int(state_k.size(2)):
        raise ValueError("active LOD state exceeds its allocated capacity")
    block_m = 16 if query_len > 1 else 1
    block_n = 64
    active_groups = triton.cdiv(state_len, block_n)
    partial_scores = buffers["partial_scores"]
    partial_indices = buffers["partial_indices"]
    partial_lse = buffers["partial_lse"]
    output = buffers["output"]
    state_lse = buffers["state_lse"]
    max_groups = int(partial_scores.size(3))
    if active_groups > max_groups or query_len > int(partial_scores.size(2)):
        raise ValueError("grouped LOD routing buffers are too small")
    _route_state_group_candidates_kernel[
        (
            batch,
            q_heads,
            triton.cdiv(query_len, block_m) * active_groups,
        )
    ](
        q,
        state_k,
        counts,
        partial_scores,
        partial_indices,
        Q_BATCH_STRIDE=q.stride(0),
        Q_HEAD_STRIDE=q.stride(1),
        Q_TOKEN_STRIDE=q.stride(2),
        STATE_BATCH_STRIDE=state_k.stride(0),
        STATE_HEAD_STRIDE=state_k.stride(1),
        STATE_TOKEN_STRIDE=state_k.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0),
        COUNT_HEAD_STRIDE=counts.stride(1),
        COUNT_TOKEN_STRIDE=counts.stride(2),
        PARTIAL_BATCH_STRIDE=partial_scores.stride(0),
        PARTIAL_HEAD_STRIDE=partial_scores.stride(1),
        PARTIAL_QUERY_STRIDE=partial_scores.stride(2),
        PARTIAL_GROUP_STRIDE=partial_scores.stride(3),
        active_groups=active_groups,
        query_len=query_len,
        state_len=state_len,
        SCALE=scale,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **_launch_kwargs(4),
    )
    candidate_block = triton.next_power_of_2(max_groups * 8)
    lse_group_block = triton.next_power_of_2(max_groups)
    _reduce_route_group_candidates_kernel[
        (batch, q_heads, triton.cdiv(query_len, block_m))
    ](
        partial_scores,
        partial_indices,
        partial_lse,
        output,
        state_lse,
        PARTIAL_BATCH_STRIDE=partial_scores.stride(0),
        PARTIAL_HEAD_STRIDE=partial_scores.stride(1),
        PARTIAL_QUERY_STRIDE=partial_scores.stride(2),
        PARTIAL_LSE_BATCH_STRIDE=partial_lse.stride(0),
        PARTIAL_LSE_HEAD_STRIDE=partial_lse.stride(1),
        PARTIAL_LSE_QUERY_STRIDE=partial_lse.stride(2),
        PARTIAL_LSE_GROUP_STRIDE=partial_lse.stride(3),
        OUTPUT_BATCH_STRIDE=output.stride(0),
        OUTPUT_HEAD_STRIDE=output.stride(1),
        OUTPUT_TOKEN_STRIDE=output.stride(2),
        STATE_LSE_BATCH_STRIDE=state_lse.stride(0),
        STATE_LSE_HEAD_STRIDE=state_lse.stride(1),
        STATE_LSE_QUERY_STRIDE=state_lse.stride(2),
        query_len=query_len,
        active_groups=active_groups,
        TOPK=topk,
        MAX_GROUPS=lse_group_block,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        STORE_LSE=False,
        **_launch_kwargs(4 if query_len > 1 else 2),
    )
    if reorder_like_torch:
        _reorder_topk_like_torch_kernel[(batch, q_heads, query_len)](
            output,
            OUTPUT_BATCH_STRIDE=output.stride(0),
            OUTPUT_HEAD_STRIDE=output.stride(1),
            OUTPUT_TOKEN_STRIDE=output.stride(2),
            TOPK=topk,
            **_launch_kwargs(1),
        )
    return output[..., :query_len, :topk]


def route_top8_scores_grouped(
    logits: torch.Tensor,
    counts: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    kv_group_size: int,
    scale: float,
    topk: int,
    state_len: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Select top state slots without changing the reference GEMM scores."""
    if not 1 <= topk <= 8:
        raise ValueError("grouped LOD routing supports topk from 1 through 8")
    if not logits.is_cuda or not counts.is_cuda:
        raise ValueError("grouped LOD score routing requires CUDA tensors")
    batch, q_heads, query_len, allocated_state = logits.shape
    if state_len is None:
        state_len = allocated_state
    if state_len > allocated_state:
        raise ValueError("active LOD state exceeds the routing score width")
    block_m = 16 if query_len > 1 else 1
    block_n = 64
    active_groups = triton.cdiv(state_len, block_n)
    partial_scores = buffers["partial_scores"]
    partial_indices = buffers["partial_indices"]
    partial_lse = buffers["partial_lse"]
    output = buffers["output"]
    state_lse = buffers["state_lse"]
    max_groups = int(partial_scores.size(3))
    if active_groups > max_groups or query_len > int(partial_scores.size(2)):
        raise ValueError("grouped LOD routing buffers are too small")
    _route_score_group_candidates_kernel[
        (
            batch,
            q_heads,
            triton.cdiv(query_len, block_m) * active_groups,
        )
    ](
        logits,
        counts,
        partial_scores,
        partial_indices,
        partial_lse,
        LOGIT_BATCH_STRIDE=logits.stride(0),
        LOGIT_HEAD_STRIDE=logits.stride(1),
        LOGIT_QUERY_STRIDE=logits.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0),
        COUNT_HEAD_STRIDE=counts.stride(1),
        COUNT_TOKEN_STRIDE=counts.stride(2),
        PARTIAL_BATCH_STRIDE=partial_scores.stride(0),
        PARTIAL_HEAD_STRIDE=partial_scores.stride(1),
        PARTIAL_QUERY_STRIDE=partial_scores.stride(2),
        PARTIAL_GROUP_STRIDE=partial_scores.stride(3),
        PARTIAL_LSE_BATCH_STRIDE=partial_lse.stride(0),
        PARTIAL_LSE_HEAD_STRIDE=partial_lse.stride(1),
        PARTIAL_LSE_QUERY_STRIDE=partial_lse.stride(2),
        PARTIAL_LSE_GROUP_STRIDE=partial_lse.stride(3),
        active_groups=active_groups,
        query_len=query_len,
        state_len=state_len,
        SCALE=scale,
        KV_GROUP_SIZE=kv_group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        STORE_LSE=return_lse,
        **_launch_kwargs(4),
    )
    candidate_block = triton.next_power_of_2(max_groups * 8)
    lse_group_block = triton.next_power_of_2(max_groups)
    _reduce_route_group_candidates_kernel[
        (batch, q_heads, triton.cdiv(query_len, block_m))
    ](
        partial_scores,
        partial_indices,
        partial_lse,
        output,
        state_lse,
        PARTIAL_BATCH_STRIDE=partial_scores.stride(0),
        PARTIAL_HEAD_STRIDE=partial_scores.stride(1),
        PARTIAL_QUERY_STRIDE=partial_scores.stride(2),
        PARTIAL_LSE_BATCH_STRIDE=partial_lse.stride(0),
        PARTIAL_LSE_HEAD_STRIDE=partial_lse.stride(1),
        PARTIAL_LSE_QUERY_STRIDE=partial_lse.stride(2),
        PARTIAL_LSE_GROUP_STRIDE=partial_lse.stride(3),
        OUTPUT_BATCH_STRIDE=output.stride(0),
        OUTPUT_HEAD_STRIDE=output.stride(1),
        OUTPUT_TOKEN_STRIDE=output.stride(2),
        STATE_LSE_BATCH_STRIDE=state_lse.stride(0),
        STATE_LSE_HEAD_STRIDE=state_lse.stride(1),
        STATE_LSE_QUERY_STRIDE=state_lse.stride(2),
        query_len=query_len,
        active_groups=active_groups,
        TOPK=topk,
        MAX_GROUPS=lse_group_block,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        STORE_LSE=return_lse,
        **_launch_kwargs(4 if query_len > 1 else 2),
    )
    _reorder_topk_like_torch_kernel[(batch, q_heads, query_len)](
        output,
        OUTPUT_BATCH_STRIDE=output.stride(0),
        OUTPUT_HEAD_STRIDE=output.stride(1),
        OUTPUT_TOKEN_STRIDE=output.stride(2),
        TOPK=topk,
        **_launch_kwargs(1),
    )
    routed = output[..., :query_len, :topk]
    if return_lse:
        return routed, state_lse[..., :query_len]
    return routed


def apply_residual_mass_opening(
    logits: torch.Tensor,
    counts: torch.Tensor,
    top_slots: torch.Tensor,
    state_lse: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    kv_group_size: int,
    scale: float,
    residual_mass: float,
    block_m: int = 16,
) -> torch.Tensor:
    """Apply the full-field residual-mass route cutoff in one GPU pass."""
    if not 0.0 < residual_mass <= 1.0:
        raise ValueError("residual mass must lie in (0, 1]")
    batch, q_heads, query_len, _ = logits.shape
    if tuple(top_slots.shape) != (batch, q_heads, query_len, 8):
        raise ValueError("fused residual opening requires exactly eight routes")
    if tuple(state_lse.shape) != tuple(local_lse.shape) or tuple(
        state_lse.shape
    ) != (batch, q_heads, query_len):
        raise ValueError("fused residual opening received incompatible LSE shapes")
    _apply_residual_mass_opening_kernel[
        (batch, q_heads, triton.cdiv(query_len, block_m))
    ](
        logits,
        counts,
        top_slots,
        state_lse,
        local_lse,
        LOGIT_BATCH_STRIDE=logits.stride(0),
        LOGIT_HEAD_STRIDE=logits.stride(1),
        LOGIT_QUERY_STRIDE=logits.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0),
        COUNT_HEAD_STRIDE=counts.stride(1),
        COUNT_TOKEN_STRIDE=counts.stride(2),
        TOP_BATCH_STRIDE=top_slots.stride(0),
        TOP_HEAD_STRIDE=top_slots.stride(1),
        TOP_QUERY_STRIDE=top_slots.stride(2),
        LSE_BATCH_STRIDE=state_lse.stride(0),
        LSE_HEAD_STRIDE=state_lse.stride(1),
        LSE_QUERY_STRIDE=state_lse.stride(2),
        LOCAL_LSE_BATCH_STRIDE=local_lse.stride(0),
        LOCAL_LSE_HEAD_STRIDE=local_lse.stride(1),
        LOCAL_LSE_QUERY_STRIDE=local_lse.stride(2),
        query_len=query_len,
        state_len=int(logits.size(3)),
        RESIDUAL_MASS=residual_mass,
        SCALE=scale,
        KV_GROUP_SIZE=kv_group_size,
        BLOCK_M=block_m,
        **_launch_kwargs(4),
    )
    return top_slots


__all__ = [
    "apply_residual_mass_opening",
    "merge_state_in_place",
    "new_route_buffers",
    "new_state_delta_buffers",
    "new_state_maxsim_buffers",
    "route_top8_scores_grouped",
    "route_top8_state_grouped",
    "route_logits_coarse_attention",
    "streaming_state_maxsim",
]
