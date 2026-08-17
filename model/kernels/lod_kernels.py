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


@triton.jit(
    do_not_specialize=["QUERY_LEN"],
    do_not_specialize_on_alignment=[
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
def _merge_attention_branches_kernel(
    primary_out,
    primary_lse,
    secondary_out,
    secondary_lse,
    tertiary_out,
    tertiary_lse,
    output,
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
    HEAD_DIM: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    INCLUDE_TERTIARY: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Merge two or three already-normalized attention branches once."""
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    query = tl.program_id(2).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    query_valid = query < QUERY_LEN
    dim = tl.arange(0, BLOCK_DIM)
    dim_valid = dim < HEAD_DIM

    primary_score = tl.load(
        primary_lse
        + batch * PRIMARY_LSE_BATCH_STRIDE
        + head * PRIMARY_LSE_HEAD_STRIDE
        + query * PRIMARY_LSE_TOKEN_STRIDE,
        mask=query_valid,
        other=-float("inf"),
    ).to(tl.float32)
    secondary_score = tl.load(
        secondary_lse
        + batch * SECONDARY_LSE_BATCH_STRIDE
        + head * SECONDARY_LSE_HEAD_STRIDE
        + query * SECONDARY_LSE_TOKEN_STRIDE,
        mask=query_valid,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.maximum(primary_score, secondary_score)
    tertiary_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    if INCLUDE_TERTIARY:
        tertiary_score = tl.load(
            tertiary_lse
            + batch * TERTIARY_LSE_BATCH_STRIDE
            + head * TERTIARY_LSE_HEAD_STRIDE
            + query * TERTIARY_LSE_TOKEN_STRIDE,
            mask=query_valid,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.maximum(maximum, tertiary_score)

    primary_weight = tl.exp(primary_score - maximum)
    secondary_weight = tl.exp(secondary_score - maximum)
    denominator = primary_weight + secondary_weight
    primary_value = tl.load(
        primary_out
        + batch * PRIMARY_BATCH_STRIDE
        + head * PRIMARY_HEAD_STRIDE
        + query[:, None] * PRIMARY_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    secondary_value = tl.load(
        secondary_out
        + batch * SECONDARY_BATCH_STRIDE
        + head * SECONDARY_HEAD_STRIDE
        + query[:, None] * SECONDARY_TOKEN_STRIDE
        + dim[None, :],
        mask=query_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = (
        primary_weight[:, None] * primary_value
        + secondary_weight[:, None] * secondary_value
    )
    if INCLUDE_TERTIARY:
        tertiary_weight = tl.exp(tertiary_score - maximum)
        tertiary_value = tl.load(
            tertiary_out
            + batch * TERTIARY_BATCH_STRIDE
            + head * TERTIARY_HEAD_STRIDE
            + query[:, None] * TERTIARY_TOKEN_STRIDE
            + dim[None, :],
            mask=query_valid[:, None] & dim_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        denominator += tertiary_weight
        numerator += tertiary_weight[:, None] * tertiary_value
    tl.store(
        output
        + batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + query[:, None] * OUTPUT_TOKEN_STRIDE
        + dim[None, :],
        numerator / denominator[:, None],
        mask=query_valid[:, None] & dim_valid[None, :],
    )


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
            sink_k
            + batch * SINK_K_BATCH_STRIDE
            + kv_head * SINK_K_HEAD_STRIDE
            + dim,
            mask=dim_valid,
            other=0.0,
        ).to(tl.float32)
        sink_output = tl.load(
            sink_v
            + batch * SINK_V_BATCH_STRIDE
            + kv_head * SINK_V_HEAD_STRIDE
            + dim,
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


@triton.jit
def _bipartite_reduce_overflow_kernel(
    overflow_k,
    overflow_v,
    reduced_k,
    reduced_v,
    reduced_counts,
    membership,
    K_ROW_STRIDE,
    V_ROW_STRIDE,
    MEMBERSHIP_ROW_STRIDE,
    OVERFLOW_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BALANCED: tl.constexpr,
    SALT: tl.constexpr,
):
    """Contract each overflow block by routing one partition to the other."""

    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    block_begin = block * BLOCK_SIZE
    reduced_begin = block * HALF_BLOCK
    anchor = tl.arange(0, HALF_BLOCK)
    source = tl.arange(0, HALF_BLOCK)
    key_dim = tl.arange(0, HEAD_DIM)
    value_dim = tl.arange(0, VALUE_DIM)
    if BALANCED:
        swap = (anchor + row + block + SALT) & 1
        anchor_token = 2 * anchor + swap
        source_token = 2 * source + 1 - swap
    else:
        anchor_token = anchor
        source_token = HALF_BLOCK + source

    anchor_k = tl.load(
        overflow_k
        + row * K_ROW_STRIDE
        + (block_begin + anchor_token)[:, None] * HEAD_DIM
        + key_dim[None, :]
    )
    source_k = tl.load(
        overflow_k
        + row * K_ROW_STRIDE
        + (block_begin + source_token)[:, None] * HEAD_DIM
        + key_dim[None, :]
    )
    similarity = tl.dot(source_k, tl.trans(anchor_k), out_dtype=tl.float32)
    similarity = similarity.to(tl.bfloat16).to(tl.float32)
    best_score = tl.max(similarity, axis=1)
    destination = tl.min(
        tl.where(
            similarity == best_score[:, None],
            anchor[None, :],
            HALF_BLOCK,
        ),
        axis=1,
    ).to(tl.int32)

    assignment = (anchor[:, None] == destination[None, :]).to(source_k.dtype)
    reduced_key = anchor_k.to(tl.float32) + tl.dot(
        assignment, source_k, out_dtype=tl.float32
    )
    anchor_v = tl.load(
        overflow_v
        + row * V_ROW_STRIDE
        + (block_begin + anchor_token)[:, None] * VALUE_DIM
        + value_dim[None, :]
    )
    source_v = tl.load(
        overflow_v
        + row * V_ROW_STRIDE
        + (block_begin + source_token)[:, None] * VALUE_DIM
        + value_dim[None, :]
    )
    reduced_value = anchor_v.to(tl.float32) + tl.dot(
        assignment.to(source_v.dtype), source_v, out_dtype=tl.float32
    )
    count = 1.0 + tl.sum(assignment.to(tl.float32), axis=1)

    tl.store(
        reduced_k
        + row * (OVERFLOW_LEN // 2) * HEAD_DIM
        + (reduced_begin + anchor)[:, None] * HEAD_DIM
        + key_dim[None, :],
        reduced_key,
    )
    tl.store(
        reduced_v
        + row * (OVERFLOW_LEN // 2) * VALUE_DIM
        + (reduced_begin + anchor)[:, None] * VALUE_DIM
        + value_dim[None, :],
        reduced_value,
    )
    tl.store(
        reduced_counts + row * (OVERFLOW_LEN // 2) + reduced_begin + anchor,
        count,
    )
    tl.store(
        membership + row * MEMBERSHIP_ROW_STRIDE + block_begin + anchor_token,
        reduced_begin + anchor,
    )
    tl.store(
        membership
        + row * MEMBERSHIP_ROW_STRIDE
        + block_begin
        + source_token,
        reduced_begin + destination,
    )


@triton.jit
def _balanced_bipartite_route_kernel(
    key_sum,
    counts,
    destination,
    membership,
    KEY_ROW_STRIDE: tl.constexpr,
    COUNT_ROW_STRIDE: tl.constexpr,
    DEST_ROW_STRIDE: tl.constexpr,
    MEMBER_ROW_STRIDE: tl.constexpr,
    TOKEN_LEN: tl.constexpr,
    ANCHOR_COUNT: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SALT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    source = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    source_valid = source < SOURCE_COUNT
    source_swap = (source + row + SALT) & 1
    source_token = 2 * source + 1 - source_swap
    dim = tl.arange(0, HEAD_DIM)
    source_count = tl.load(
        counts + row * COUNT_ROW_STRIDE + source_token,
        mask=source_valid,
        other=1.0,
    )
    source_key = tl.load(
        key_sum
        + row * KEY_ROW_STRIDE
        + source_token[:, None] * HEAD_DIM
        + dim[None, :],
        mask=source_valid[:, None],
        other=0.0,
    )
    source_key = (source_key / source_count[:, None]).to(source_key.dtype)
    best_score = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    best_anchor = tl.zeros((BLOCK_M,), tl.int32)
    for anchor_begin in range(0, ANCHOR_COUNT, BLOCK_N):
        anchor = anchor_begin + tl.arange(0, BLOCK_N)
        anchor_valid = anchor < ANCHOR_COUNT
        anchor_swap = tl.where(
            2 * anchor + 1 < TOKEN_LEN, (anchor + row + SALT) & 1, 0
        )
        anchor_token = 2 * anchor + anchor_swap
        anchor_valid = anchor_valid & (anchor_token < TOKEN_LEN)
        anchor_count = tl.load(
            counts + row * COUNT_ROW_STRIDE + anchor_token,
            mask=anchor_valid,
            other=1.0,
        )
        anchor_key = tl.load(
            key_sum
            + row * KEY_ROW_STRIDE
            + anchor_token[:, None] * HEAD_DIM
            + dim[None, :],
            mask=anchor_valid[:, None],
            other=0.0,
        )
        anchor_key = (anchor_key / anchor_count[:, None]).to(anchor_key.dtype)
        score = tl.dot(source_key, tl.trans(anchor_key), out_dtype=tl.float32)
        # torch.matmul returns BF16 for the reference routing path.  Match its
        # rounding before argmax; near-ties become common at realistic block
        # sizes and otherwise select different owners despite close scores.
        score = score.to(tl.bfloat16).to(tl.float32)
        score = tl.where(
            source_valid[:, None] & anchor_valid[None, :],
            score,
            float("-inf"),
        )
        tile_score = tl.max(score, axis=1)
        tile_anchor = anchor_begin + tl.argmax(score, axis=1)
        improve = tile_score > best_score
        best_score = tl.where(improve, tile_score, best_score)
        best_anchor = tl.where(improve, tile_anchor, best_anchor)
    tl.store(
        destination + row * DEST_ROW_STRIDE + source,
        best_anchor,
        mask=source_valid,
    )
    tl.store(
        membership + row * MEMBER_ROW_STRIDE + source_token,
        best_anchor,
        mask=source_valid,
    )


@triton.jit
def _balanced_bipartite_reduce_feature_kernel(
    values,
    destination,
    output,
    VALUE_ROW_STRIDE: tl.constexpr,
    DEST_ROW_STRIDE: tl.constexpr,
    OUTPUT_ROW_STRIDE: tl.constexpr,
    TOKEN_LEN: tl.constexpr,
    ANCHOR_COUNT: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    FEATURE_DIM: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SALT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tiles_d = tl.cdiv(FEATURE_DIM, BLOCK_D)
    tile = tl.program_id(1)
    anchor_begin = (tile // tiles_d) * BLOCK_A
    dim_begin = (tile % tiles_d) * BLOCK_D
    anchor = anchor_begin + tl.arange(0, BLOCK_A)
    dim = dim_begin + tl.arange(0, BLOCK_D)
    anchor_valid = anchor < ANCHOR_COUNT
    dim_valid = dim < FEATURE_DIM
    anchor_swap = tl.where(
        2 * anchor + 1 < TOKEN_LEN, (anchor + row + SALT) & 1, 0
    )
    anchor_token = 2 * anchor + anchor_swap
    anchor_valid = anchor_valid & (anchor_token < TOKEN_LEN)
    accumulator = tl.load(
        values
        + row * VALUE_ROW_STRIDE
        + anchor_token[:, None] * FEATURE_DIM
        + dim[None, :],
        mask=anchor_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    for source_begin in range(0, SOURCE_COUNT, BLOCK_M):
        source = source_begin + tl.arange(0, BLOCK_M)
        source_valid = source < SOURCE_COUNT
        source_swap = (source + row + SALT) & 1
        source_token = 2 * source + 1 - source_swap
        owner = tl.load(
            destination + row * DEST_ROW_STRIDE + source,
            mask=source_valid,
            other=-1,
        )
        source_value = tl.load(
            values
            + row * VALUE_ROW_STRIDE
            + source_token[:, None] * FEATURE_DIM
            + dim[None, :],
            mask=source_valid[:, None] & dim_valid[None, :],
            other=0.0,
        )
        assignment = (
            anchor[:, None] == owner[None, :]
        ) & anchor_valid[:, None] & source_valid[None, :]
        accumulator += tl.dot(
            assignment.to(source_value.dtype),
            source_value,
            out_dtype=tl.float32,
        )
    tl.store(
        output
        + row * OUTPUT_ROW_STRIDE
        + anchor[:, None] * FEATURE_DIM
        + dim[None, :],
        accumulator,
        mask=anchor_valid[:, None] & dim_valid[None, :],
    )


@triton.jit
def _balanced_bipartite_reduce_count_kernel(
    counts,
    destination,
    output_counts,
    membership,
    COUNT_ROW_STRIDE: tl.constexpr,
    DEST_ROW_STRIDE: tl.constexpr,
    OUTPUT_ROW_STRIDE: tl.constexpr,
    MEMBER_ROW_STRIDE: tl.constexpr,
    TOKEN_LEN: tl.constexpr,
    ANCHOR_COUNT: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SALT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    anchor = tl.program_id(1) * BLOCK_A + tl.arange(0, BLOCK_A)
    anchor_valid = anchor < ANCHOR_COUNT
    anchor_swap = tl.where(
        2 * anchor + 1 < TOKEN_LEN, (anchor + row + SALT) & 1, 0
    )
    anchor_token = 2 * anchor + anchor_swap
    anchor_valid = anchor_valid & (anchor_token < TOKEN_LEN)
    accumulator = tl.load(
        counts + row * COUNT_ROW_STRIDE + anchor_token,
        mask=anchor_valid,
        other=0.0,
    ).to(tl.float32)
    for source_begin in range(0, SOURCE_COUNT, BLOCK_M):
        source = source_begin + tl.arange(0, BLOCK_M)
        source_valid = source < SOURCE_COUNT
        source_swap = (source + row + SALT) & 1
        source_token = 2 * source + 1 - source_swap
        owner = tl.load(
            destination + row * DEST_ROW_STRIDE + source,
            mask=source_valid,
            other=-1,
        )
        source_count = tl.load(
            counts + row * COUNT_ROW_STRIDE + source_token,
            mask=source_valid,
            other=0.0,
        )
        accumulator += tl.sum(
            tl.where(
                (anchor[:, None] == owner[None, :])
                & anchor_valid[:, None]
                & source_valid[None, :],
                source_count[None, :],
                0.0,
            ),
            axis=1,
        )
    tl.store(
        output_counts + row * OUTPUT_ROW_STRIDE + anchor,
        accumulator,
        mask=anchor_valid,
    )
    tl.store(
        membership + row * MEMBER_ROW_STRIDE + anchor_token,
        anchor,
        mask=anchor_valid,
    )


@triton.jit
def _balanced_bipartite_route_atomic_kernel(
    key_sum,
    value_sum,
    counts,
    reduced_k_fp32,
    reduced_v_fp32,
    reduced_counts,
    membership,
    KEY_ROW_STRIDE: tl.constexpr,
    VALUE_ROW_STRIDE: tl.constexpr,
    COUNT_ROW_STRIDE: tl.constexpr,
    OUTPUT_K_ROW_STRIDE: tl.constexpr,
    OUTPUT_V_ROW_STRIDE: tl.constexpr,
    OUTPUT_COUNT_ROW_STRIDE: tl.constexpr,
    MEMBER_ROW_STRIDE: tl.constexpr,
    TOKEN_LEN: tl.constexpr,
    ANCHOR_COUNT: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SALT: tl.constexpr,
):
    """Route once, then scatter source and paired-anchor sums with low fan-in."""

    row = tl.program_id(0).to(tl.int64)
    source = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    source_valid = source < SOURCE_COUNT
    source_swap = (source + row + SALT) & 1
    source_token = 2 * source + 1 - source_swap
    paired_anchor_token = 2 * source + source_swap
    key_dim = tl.arange(0, HEAD_DIM)
    value_dim = tl.arange(0, VALUE_DIM)

    source_count = tl.load(
        counts + row * COUNT_ROW_STRIDE + source_token,
        mask=source_valid,
        other=1.0,
    )
    source_key_sum = tl.load(
        key_sum
        + row * KEY_ROW_STRIDE
        + source_token[:, None] * HEAD_DIM
        + key_dim[None, :],
        mask=source_valid[:, None],
        other=0.0,
    )
    source_key = (source_key_sum / source_count[:, None]).to(source_key_sum.dtype)
    best_score = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    best_anchor = tl.zeros((BLOCK_M,), tl.int32)
    for anchor_begin in range(0, ANCHOR_COUNT, BLOCK_N):
        anchor = anchor_begin + tl.arange(0, BLOCK_N)
        anchor_valid = anchor < ANCHOR_COUNT
        anchor_swap = (anchor + row + SALT) & 1
        anchor_token = 2 * anchor + anchor_swap
        anchor_count = tl.load(
            counts + row * COUNT_ROW_STRIDE + anchor_token,
            mask=anchor_valid,
            other=1.0,
        )
        anchor_key_sum = tl.load(
            key_sum
            + row * KEY_ROW_STRIDE
            + anchor_token[:, None] * HEAD_DIM
            + key_dim[None, :],
            mask=anchor_valid[:, None],
            other=0.0,
        )
        anchor_key = (anchor_key_sum / anchor_count[:, None]).to(
            anchor_key_sum.dtype
        )
        score = tl.dot(source_key, tl.trans(anchor_key), out_dtype=tl.float32)
        score = score.to(tl.bfloat16).to(tl.float32)
        score = tl.where(
            source_valid[:, None] & anchor_valid[None, :],
            score,
            float("-inf"),
        )
        tile_score = tl.max(score, axis=1)
        tile_anchor = anchor_begin + tl.argmax(score, axis=1)
        improve = tile_score > best_score
        best_score = tl.where(improve, tile_score, best_score)
        best_anchor = tl.where(improve, tile_anchor, best_anchor)

    paired_anchor_count = tl.load(
        counts + row * COUNT_ROW_STRIDE + paired_anchor_token,
        mask=source_valid,
        other=0.0,
    )
    paired_anchor_key = tl.load(
        key_sum
        + row * KEY_ROW_STRIDE
        + paired_anchor_token[:, None] * HEAD_DIM
        + key_dim[None, :],
        mask=source_valid[:, None],
        other=0.0,
    )
    source_value = tl.load(
        value_sum
        + row * VALUE_ROW_STRIDE
        + source_token[:, None] * VALUE_DIM
        + value_dim[None, :],
        mask=source_valid[:, None],
        other=0.0,
    )
    paired_anchor_value = tl.load(
        value_sum
        + row * VALUE_ROW_STRIDE
        + paired_anchor_token[:, None] * VALUE_DIM
        + value_dim[None, :],
        mask=source_valid[:, None],
        other=0.0,
    )

    tl.atomic_add(
        reduced_k_fp32
        + row * OUTPUT_K_ROW_STRIDE
        + source[:, None] * HEAD_DIM
        + key_dim[None, :],
        paired_anchor_key.to(tl.float32),
        mask=source_valid[:, None],
    )
    tl.atomic_add(
        reduced_k_fp32
        + row * OUTPUT_K_ROW_STRIDE
        + best_anchor[:, None] * HEAD_DIM
        + key_dim[None, :],
        source_key_sum.to(tl.float32),
        mask=source_valid[:, None],
    )
    tl.atomic_add(
        reduced_v_fp32
        + row * OUTPUT_V_ROW_STRIDE
        + source[:, None] * VALUE_DIM
        + value_dim[None, :],
        paired_anchor_value.to(tl.float32),
        mask=source_valid[:, None],
    )
    tl.atomic_add(
        reduced_v_fp32
        + row * OUTPUT_V_ROW_STRIDE
        + best_anchor[:, None] * VALUE_DIM
        + value_dim[None, :],
        source_value.to(tl.float32),
        mask=source_valid[:, None],
    )
    tl.atomic_add(
        reduced_counts + row * OUTPUT_COUNT_ROW_STRIDE + source,
        paired_anchor_count.to(tl.float32),
        mask=source_valid,
    )
    tl.atomic_add(
        reduced_counts + row * OUTPUT_COUNT_ROW_STRIDE + best_anchor,
        source_count.to(tl.float32),
        mask=source_valid,
    )
    tl.store(
        membership + row * MEMBER_ROW_STRIDE + paired_anchor_token,
        source,
        mask=source_valid,
    )
    tl.store(
        membership + row * MEMBER_ROW_STRIDE + source_token,
        best_anchor,
        mask=source_valid,
    )


@triton.jit
def _balanced_bipartite_finalize_kernel(
    reduced_k_fp32,
    reduced_v_fp32,
    reduced_k,
    reduced_v,
    K_ROW_STRIDE: tl.constexpr,
    V_ROW_STRIDE: tl.constexpr,
    ANCHOR_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    anchor = tl.program_id(1) * BLOCK_A + tl.arange(0, BLOCK_A)
    dim = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    anchor_valid = anchor < ANCHOR_COUNT
    k_mask = anchor_valid[:, None] & (dim[None, :] < HEAD_DIM)
    v_mask = anchor_valid[:, None] & (dim[None, :] < VALUE_DIM)
    k_offset = row * K_ROW_STRIDE + anchor[:, None] * HEAD_DIM + dim[None, :]
    v_offset = row * V_ROW_STRIDE + anchor[:, None] * VALUE_DIM + dim[None, :]
    tl.store(
        reduced_k + k_offset,
        tl.load(reduced_k_fp32 + k_offset, mask=k_mask, other=0.0),
        mask=k_mask,
    )
    tl.store(
        reduced_v + v_offset,
        tl.load(reduced_v_fp32 + v_offset, mask=v_mask, other=0.0),
        mask=v_mask,
    )


@triton.jit
def _balanced_bipartite_atomic_reduce_kernel(
    key_sum,
    value_sum,
    counts,
    destination,
    reduced_k_fp32,
    reduced_v_fp32,
    reduced_counts,
    membership,
    KEY_ROW_STRIDE: tl.constexpr,
    VALUE_ROW_STRIDE: tl.constexpr,
    COUNT_ROW_STRIDE: tl.constexpr,
    DEST_ROW_STRIDE: tl.constexpr,
    OUTPUT_K_ROW_STRIDE: tl.constexpr,
    OUTPUT_V_ROW_STRIDE: tl.constexpr,
    OUTPUT_COUNT_ROW_STRIDE: tl.constexpr,
    MEMBER_ROW_STRIDE: tl.constexpr,
    SOURCE_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SALT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    source = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_begin = tl.program_id(2) * BLOCK_D
    source_valid = source < SOURCE_COUNT
    swap = (source + row + SALT) & 1
    source_token = 2 * source + 1 - swap
    anchor_token = 2 * source + swap
    owner = tl.load(
        destination + row * DEST_ROW_STRIDE + source,
        mask=source_valid,
        other=0,
    )
    feature = feature_begin + tl.arange(0, BLOCK_D)
    key_valid = feature < HEAD_DIM
    value_valid = feature < VALUE_DIM
    source_key = tl.load(
        key_sum
        + row * KEY_ROW_STRIDE
        + source_token[:, None] * HEAD_DIM
        + feature[None, :],
        mask=source_valid[:, None] & key_valid[None, :],
        other=0.0,
    )
    anchor_key = tl.load(
        key_sum
        + row * KEY_ROW_STRIDE
        + anchor_token[:, None] * HEAD_DIM
        + feature[None, :],
        mask=source_valid[:, None] & key_valid[None, :],
        other=0.0,
    )
    source_value = tl.load(
        value_sum
        + row * VALUE_ROW_STRIDE
        + source_token[:, None] * VALUE_DIM
        + feature[None, :],
        mask=source_valid[:, None] & value_valid[None, :],
        other=0.0,
    )
    anchor_value = tl.load(
        value_sum
        + row * VALUE_ROW_STRIDE
        + anchor_token[:, None] * VALUE_DIM
        + feature[None, :],
        mask=source_valid[:, None] & value_valid[None, :],
        other=0.0,
    )
    source_count = tl.load(
        counts + row * COUNT_ROW_STRIDE + source_token,
        mask=source_valid,
        other=0.0,
    )
    anchor_count = tl.load(
        counts + row * COUNT_ROW_STRIDE + anchor_token,
        mask=source_valid,
        other=0.0,
    )
    tl.atomic_add(
        reduced_k_fp32
        + row * OUTPUT_K_ROW_STRIDE
        + source[:, None] * HEAD_DIM
        + feature[None, :],
        anchor_key.to(tl.float32),
        mask=source_valid[:, None] & key_valid[None, :],
    )
    tl.atomic_add(
        reduced_k_fp32
        + row * OUTPUT_K_ROW_STRIDE
        + owner[:, None] * HEAD_DIM
        + feature[None, :],
        source_key.to(tl.float32),
        mask=source_valid[:, None] & key_valid[None, :],
    )
    tl.atomic_add(
        reduced_v_fp32
        + row * OUTPUT_V_ROW_STRIDE
        + source[:, None] * VALUE_DIM
        + feature[None, :],
        anchor_value.to(tl.float32),
        mask=source_valid[:, None] & value_valid[None, :],
    )
    tl.atomic_add(
        reduced_v_fp32
        + row * OUTPUT_V_ROW_STRIDE
        + owner[:, None] * VALUE_DIM
        + feature[None, :],
        source_value.to(tl.float32),
        mask=source_valid[:, None] & value_valid[None, :],
    )
    tl.atomic_add(
        reduced_counts + row * OUTPUT_COUNT_ROW_STRIDE + source,
        anchor_count.to(tl.float32),
        mask=source_valid & (feature_begin == 0),
    )
    tl.atomic_add(
        reduced_counts + row * OUTPUT_COUNT_ROW_STRIDE + owner,
        source_count.to(tl.float32),
        mask=source_valid & (feature_begin == 0),
    )
    tl.store(
        membership + row * MEMBER_ROW_STRIDE + anchor_token,
        source,
        mask=source_valid & (feature_begin == 0),
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
    ]
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
            tl.sum(mean_key.to(tl.float32) * mean_key.to(tl.float32), axis=1)
            / HEAD_DIM
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
    do_not_specialize_on_alignment=["overflow_len", "state_len"],
)
def _streaming_state_maxsim_kernel(
    overflow_k,
    state_k,
    review_state_k,
    select_scale,
    counts,
    route_scores,
    route_indices,
    select_scores,
    overflow_norms,
    OVERFLOW_BATCH_STRIDE: tl.constexpr,
    OVERFLOW_HEAD_STRIDE: tl.constexpr,
    OVERFLOW_TOKEN_STRIDE: tl.constexpr,
    STATE_BATCH_STRIDE: tl.constexpr,
    STATE_HEAD_STRIDE: tl.constexpr,
    STATE_TOKEN_STRIDE: tl.constexpr,
    REVIEW_STATE_BATCH_STRIDE: tl.constexpr,
    REVIEW_STATE_HEAD_STRIDE: tl.constexpr,
    REVIEW_STATE_TOKEN_STRIDE: tl.constexpr,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    SCALE_BATCH_STRIDE: tl.constexpr,
    SCALE_HEAD_STRIDE: tl.constexpr,
    SCALE_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_BATCH_STRIDE: tl.constexpr,
    OUTPUT_HEAD_STRIDE: tl.constexpr,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    overflow_len,
    state_len,
    HEAD_DIM: tl.constexpr,
    SINK_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREPARED: tl.constexpr,
    REVIEW_ROUTE: tl.constexpr,
    FUSED_COHERENCE: tl.constexpr,
    STORE_OVERFLOW_NORMS: tl.constexpr,
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
    if STORE_OVERFLOW_NORMS:
        overflow_rms = tl.sqrt(
            tl.sum(overflow.to(tl.float32) * overflow.to(tl.float32), axis=1)
            / HEAD_DIM
        )
    # Geometry 0 is raw dot product, 1 is spherical construction, 2 is
    # coherence-aware assignment, and 3 is the spherical-coherence diagnostic.
    # ``overflow_k`` is already in transient leaf geometry; fusing the much
    # larger centroid scan here avoids materializing overflow-by-state scores.

    best_select_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    best_route_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    best_route_index = tl.full((BLOCK_M,), -1, tl.int32)
    if REVIEW_ROUTE:
        top_packed = tl.full(
            (BLOCK_M, 4), -9223372036854775807, tl.int64
        )
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
        slot_valid = slot_valid & (count > 0.5)
        key = tl.load(
            state_k
            + batch * STATE_BATCH_STRIDE
            + head * STATE_HEAD_STRIDE
            + slot[:, None] * STATE_TOKEN_STRIDE
            + dim[None, :],
            mask=slot_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        if PREPARED or FUSED_COHERENCE:
            route_key = key.to(overflow.dtype)
        else:
            route_key = (key / count[:, None]).to(overflow.dtype)
        scores = tl.dot(overflow, tl.trans(route_key), out_dtype=tl.float32)
        if FUSED_COHERENCE:
            # Assignment-only coherence has an exact cancellation:
            #
            #   mean(K) / mean(rms(K)) == sum(K) / sum(rms(K)).
            #
            # Likewise, the spherical append key is sum(K) / rms(sum(K)).
            # Applying those two scalar denominators after one MFMA removes
            # both prepared D-wide centroid caches and their refresh kernels.
            norm_sum = tl.load(
                select_scale
                + batch * SCALE_BATCH_STRIDE
                + head * SCALE_HEAD_STRIDE
                + slot * SCALE_TOKEN_STRIDE,
                mask=slot_valid,
                other=1.0,
            ).to(tl.float32)
            state_rms = tl.sqrt(
                tl.sum(key * key, axis=1) / HEAD_DIM
            )
            append_scores = (
                scores / tl.maximum(state_rms[None, :], 1e-12)
            ).to(tl.bfloat16).to(tl.float32)
            route_scores_tile = (
                scores / tl.maximum(norm_sum[None, :], 1e-12)
            ).to(tl.bfloat16).to(tl.float32)
            append_scores = tl.where(
                token_valid[:, None] & slot_valid[None, :],
                append_scores,
                -float("inf"),
            )
            route_scores_tile = tl.where(
                token_valid[:, None] & slot_valid[None, :],
                route_scores_tile,
                -float("inf"),
            )
        else:
            # torch.matmul returns BF16 for the reference path. Preserve that
            # score precision while avoiding its overflow-by-state materialization.
            scores = scores.to(tl.bfloat16).to(tl.float32)
            scores = tl.where(
                token_valid[:, None] & slot_valid[None, :],
                scores,
                -float("inf"),
            )
            append_scores = scores
            route_scores_tile = scores
        if REVIEW_ROUTE:
            scale = tl.load(
                select_scale
                + batch * SCALE_BATCH_STRIDE
                + head * SCALE_HEAD_STRIDE
                + slot * SCALE_TOKEN_STRIDE,
                mask=slot_valid,
                other=0.0,
            )
            approximate_route_scores = scores * scale[None, :]
        best_select_score = tl.maximum(
            best_select_score, tl.max(append_scores, axis=1)
        )

        route_valid = slot_valid & (slot >= SINK_LEN)
        if REVIEW_ROUTE:
            route_candidate = tl.where(
                token_valid[:, None] & route_valid[None, :],
                approximate_route_scores,
                -float("inf"),
            )
            score_bits = route_candidate.to(tl.uint32, bitcast=True)
            negative = (score_bits & 0x80000000) != 0
            ordered_bits = tl.where(
                negative,
                score_bits ^ 0xFFFFFFFF,
                score_bits ^ 0x80000000,
            ).to(tl.int64)
            score_rank = ordered_bits - 2147483648
            inverse_slot = 4294967295 - slot.to(tl.int64)
            packed = score_rank * 4294967296 + inverse_slot[None, :]
            block_top = tl.topk(packed, 4, dim=1)
            top_packed = tl.topk(
                tl.interleave(top_packed, block_top), 4, dim=1
            )
        else:
            route_candidate = tl.where(
                token_valid[:, None] & route_valid[None, :],
                route_scores_tile,
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

    if REVIEW_ROUTE:
        inverse_slot = top_packed & 0xFFFFFFFF
        candidate_indices = (4294967295 - inverse_slot).to(tl.int32)
        candidate_ranks = tl.arange(0, 4)
        for candidate_rank in tl.static_range(0, 4):
            candidate = tl.max(
                tl.where(
                    candidate_ranks[None, :] == candidate_rank,
                    candidate_indices,
                    -1,
                ),
                axis=1,
            ).to(tl.int64)
            candidate_valid = token_valid & (candidate < state_len)
            review_key = tl.load(
                review_state_k
                + batch * REVIEW_STATE_BATCH_STRIDE
                + head * REVIEW_STATE_HEAD_STRIDE
                + candidate[:, None] * REVIEW_STATE_TOKEN_STRIDE
                + dim[None, :],
                mask=candidate_valid[:, None],
                other=0.0,
            )
            exact_score = tl.sum(
                overflow.to(tl.float32) * review_key.to(tl.float32), axis=1
            ).to(tl.bfloat16).to(tl.float32)
            exact_score = tl.where(
                candidate_valid, exact_score, -float("inf")
            )
            take_candidate = (exact_score > best_route_score) | (
                (exact_score == best_route_score)
                & (candidate < best_route_index)
            )
            best_route_score = tl.where(
                take_candidate, exact_score, best_route_score
            )
            best_route_index = tl.where(
                take_candidate, candidate.to(tl.int32), best_route_index
            )

    output_offset = (
        batch * OUTPUT_BATCH_STRIDE
        + head * OUTPUT_HEAD_STRIDE
        + token * OUTPUT_TOKEN_STRIDE
    )
    tl.store(route_scores + output_offset, best_route_score, mask=token_valid)
    tl.store(route_indices + output_offset, best_route_index, mask=token_valid)
    tl.store(select_scores + output_offset, best_select_score, mask=token_valid)
    if STORE_OVERFLOW_NORMS:
        tl.store(overflow_norms + output_offset, overflow_rms, mask=token_valid)


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
            token_valid[:, None]
            & slot_valid[None, :]
            & (slot[None, :] >= SINK_LEN),
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
    state_v,
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
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    else:
        kv_head = head_program
        group_head = row // BLOCK_M
        query = query_block * BLOCK_M + row % BLOCK_M
        query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = query < query_len
    key_dim = tl.arange(0, HEAD_BLOCK_DIM)
    value_dim = tl.arange(0, VALUE_BLOCK_DIM)
    token_offset = tl.arange(0, BLOCK_N)

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
            mean_values = (
                values.to(tl.float32) / count[:, None]
            ).to(values.dtype)
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
        scores = SCALE * tl.dot(
            queries, tl.trans(keys), out_dtype=tl.float32
        )
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

    output_row = (
        (batch * QUERY_HEADS + query_head) * query_len + query
    ).to(tl.int64)
    tl.store(
        output + output_row[:, None] * VALUE_DIM + value_dim[None, :],
        accumulator / denominator[:, None],
        mask=query_valid[:, None] & (value_dim[None, :] < VALUE_DIM),
    )
    tl.store(
        lse + output_row,
        maximum + tl.log(denominator),
        mask=query_valid,
    )


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
        "RESIDUAL_LSE_BATCH_STRIDE",
        "RESIDUAL_LSE_HEAD_STRIDE",
        "TOP_BATCH_STRIDE",
        "TOP_HEAD_STRIDE",
        "query_len",
        "state_len",
        "local_len",
        "local_offset",
    ],
)
def _route_logits_topk_coarse_attention_kernel(
    q,
    route_logits,
    state_v,
    counts,
    local_k,
    local_v,
    residual_local_lse,
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
    RESIDUAL_LSE_BATCH_STRIDE,
    RESIDUAL_LSE_HEAD_STRIDE,
    RESIDUAL_LSE_TOKEN_STRIDE: tl.constexpr,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    TOP_QUERY_STRIDE: tl.constexpr,
    query_len,
    state_len,
    local_len,
    local_offset,
    QUERY_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    ROW_COUNT: tl.constexpr,
    STABLE_RECOMPUTE: tl.constexpr,
    ROUTE_ONLY: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    HEAD_TAIL_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    OPEN_COUNT: tl.constexpr,
    MAX_LEAF_TOKENS: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    RESIDUAL_MASS: tl.constexpr,
    USE_EXTERNAL_LOCAL_LSE: tl.constexpr,
    SCALE: tl.constexpr,
    ROUTE_COUNT_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Select routes while streaming the complete coarse softmax once."""
    batch = tl.program_id(0).to(tl.int64)
    head_program = tl.program_id(1).to(tl.int64)
    query_block = tl.program_id(2).to(tl.int64)
    row = tl.arange(0, ROW_COUNT)
    if HEAD_MAJOR:
        kv_head = head_program // KV_GROUP_SIZE
        query_head = head_program + tl.zeros((ROW_COUNT,), tl.int64)
        query = query_block * BLOCK_M + row
    else:
        kv_head = head_program
        group_head = row // BLOCK_M
        query = query_block * BLOCK_M + row % BLOCK_M
        query_head = kv_head * KV_GROUP_SIZE + group_head
    query_valid = query < query_len
    key_dim = tl.arange(0, HEAD_BLOCK_DIM)
    value_dim = tl.arange(0, VALUE_BLOCK_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    route_rank = tl.arange(0, ROUTE_COUNT)

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
    if ROUTE_COUNT == 8:
        # Sort scores and prefer the lower slot index on exact ties in one
        # packed scalar, so Triton's bitonic top-k can retain both fields.
        top_packed = tl.full(
            (ROW_COUNT, ROUTE_COUNT),
            -9223372036854775807,
            tl.int64,
        )
    else:
        top_scores = tl.full(
            (ROW_COUNT, ROUTE_COUNT),
            -float("inf"),
            tl.float32,
        )
        top_indices = tl.full(
            (ROW_COUNT, ROUTE_COUNT), -1, tl.int32
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
        if not STABLE_RECOMPUTE and not ROUTE_ONLY:
            values = tl.load(
                state_v
                + batch * STATE_V_BATCH_STRIDE
                + kv_head * STATE_V_HEAD_STRIDE
                + slot[:, None] * STATE_V_TOKEN_STRIDE
                + value_dim[None, :],
                mask=state_valid[:, None]
                & (value_dim[None, :] < VALUE_DIM),
                other=0.0,
            )
            mean_values = (
                values.to(tl.float32) / count[:, None]
            ).to(values.dtype)
        scores = tl.load(
            route_logits
            + batch * LOGIT_BATCH_STRIDE
            + query_head[:, None] * LOGIT_HEAD_STRIDE
            + query[:, None] * LOGIT_QUERY_STRIDE
            + slot[None, :] * LOGIT_STATE_STRIDE,
            mask=query_valid[:, None] & state_valid[None, :],
            other=-float("inf"),
        ).to(tl.float32)
        # Match the standalone route kernel's BF16 scale rounding exactly;
        # coarse mass still uses the unrounded FP32 score below.
        route_dot_scores = (
            scores.to(tl.bfloat16) * SCALE
        ).to(tl.bfloat16).to(tl.float32)
        dot_scores = scores * SCALE
        scores = dot_scores + tl.log(count)[None, :]
        route_scores = (
            route_dot_scores + ROUTE_COUNT_BIAS * tl.log(count)[None, :]
        )
        valid = query_valid[:, None] & state_valid[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        route_scores = tl.where(valid, route_scores, -float("inf"))

        route_valid = slot >= PROTECTED_LEN
        if MAX_LEAF_TOKENS:
            remaining_scores = tl.where(
                route_valid[None, :] & (count[None, :] <= MAX_LEAF_TOKENS),
                route_scores,
                -float("inf"),
            )
        else:
            remaining_scores = tl.where(
                route_valid[None, :], route_scores, -float("inf")
            )
        if ROUTE_COUNT == 8:
            score_bits = remaining_scores.to(tl.uint32, bitcast=True)
            negative = (score_bits & 0x80000000) != 0
            ordered_bits = tl.where(
                negative,
                score_bits ^ 0xFFFFFFFF,
                score_bits ^ 0x80000000,
            ).to(tl.int64)
            score_rank = ordered_bits - 2147483648
            global_slot = (state_begin + token_offset).to(tl.int64)
            inverse_slot = 4294967295 - global_slot
            packed_scores = (
                score_rank * 4294967296 + inverse_slot[None, :]
            )
            block_top = tl.topk(packed_scores, ROUTE_COUNT, dim=1)
            top_packed = tl.topk(
                tl.interleave(top_packed, block_top), ROUTE_COUNT, dim=1
            )
        else:
            for _ in tl.static_range(0, OPEN_COUNT):
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

        if (
            (not STABLE_RECOMPUTE and not ROUTE_ONLY)
            or RESIDUAL_MASS > 0.0
        ):
            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            denominator = denominator * correction + tl.sum(
                probabilities, axis=1
            )
            if not STABLE_RECOMPUTE:
                accumulator = accumulator * correction[:, None] + tl.dot(
                    probabilities.to(mean_values.dtype),
                    mean_values,
                    out_dtype=tl.float32,
                )
            maximum = new_maximum

    if ROUTE_COUNT == 8:
        inverse_slot = top_packed & 0xFFFFFFFF
        top_indices = (4294967295 - inverse_slot).to(tl.int32)
    # Routing may use a different count prior, but removing the selected
    # centroids from coarse attention must always use the true mass score.
    selected_valid = query_valid[:, None] & (top_indices < state_len)
    selected_counts = tl.load(
        counts
        + batch * COUNT_BATCH_STRIDE
        + kv_head * COUNT_HEAD_STRIDE
        + top_indices * COUNT_TOKEN_STRIDE,
        mask=selected_valid,
        other=1.0,
    ).to(tl.float32)
    selected_logits = tl.load(
        route_logits
        + batch * LOGIT_BATCH_STRIDE
        + query_head[:, None] * LOGIT_HEAD_STRIDE
        + query[:, None] * LOGIT_QUERY_STRIDE
        + top_indices * LOGIT_STATE_STRIDE,
        mask=selected_valid,
        other=-float("inf"),
    ).to(tl.float32)
    top_route_scores = (
        selected_logits.to(tl.bfloat16) * SCALE
    ).to(tl.bfloat16).to(tl.float32)
    top_route_scores += ROUTE_COUNT_BIAS * tl.log(selected_counts)
    top_scores = selected_logits * SCALE + tl.log(selected_counts)
    top_indices = tl.where(selected_valid, top_indices, -1)

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
        scores = SCALE * tl.dot(
            queries, tl.trans(keys), out_dtype=tl.float32
        )
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

    if (
        RESIDUAL_MASS > 0.0
        or OPEN_COUNT < ROUTE_COUNT
        or OPEN_COUNT <= 4
    ):
        if RESIDUAL_MASS > 0.0:
            full_lse = maximum + tl.log(denominator)
            if USE_EXTERNAL_LOCAL_LSE:
                external_local_lse = tl.load(
                    residual_local_lse
                    + batch * RESIDUAL_LSE_BATCH_STRIDE
                    + query_head * RESIDUAL_LSE_HEAD_STRIDE
                    + query * RESIDUAL_LSE_TOKEN_STRIDE,
                    mask=query_valid,
                    other=-float("inf"),
                ).to(tl.float32)
                combined_maximum = tl.maximum(full_lse, external_local_lse)
                full_lse = combined_maximum + tl.log(
                    tl.exp(full_lse - combined_maximum)
                    + tl.exp(external_local_lse - combined_maximum)
                )
            remaining_mass = tl.sum(
                tl.exp(top_scores - full_lse[:, None]), axis=1
            )
        if RESIDUAL_MASS > 0.0:
            remaining_scores = top_scores
        else:
            # Padded top-k storage may contain one extra candidate (for
            # example top-3 uses a four-wide vector). Select the requested
            # routes with the same BF16-rounded objective as the standalone
            # routing kernel, while retaining exact coarse scores below.
            remaining_scores = top_route_scores
        opened_indices = tl.full(
            (ROW_COUNT, ROUTE_COUNT), -1, tl.int32
        )
        opened_scores = tl.full(
            (ROW_COUNT, ROUTE_COUNT),
            -float("inf"),
            tl.float32,
        )
        for route in tl.static_range(0, OPEN_COUNT):
            candidate_score = tl.max(remaining_scores, axis=1)
            candidate_index = tl.min(
                tl.where(
                    remaining_scores == candidate_score[:, None],
                    top_indices,
                    0x7FFFFFFF,
                ),
                axis=1,
            )
            candidate_rank = tl.min(
                tl.where(
                    (remaining_scores == candidate_score[:, None])
                    & (top_indices == candidate_index[:, None]),
                    route_rank[None, :],
                    ROUTE_COUNT,
                ),
                axis=1,
            )
            candidate_coarse_score = tl.max(
                tl.where(
                    route_rank[None, :] == candidate_rank[:, None],
                    top_scores,
                    -float("inf"),
                ),
                axis=1,
            )
            if RESIDUAL_MASS > 0.0:
                if route == 0:
                    opened = query_valid
                else:
                    opened = (
                        query_valid
                        & (route < OPEN_COUNT)
                        & (remaining_mass > RESIDUAL_MASS)
                    )
            else:
                opened = query_valid & (route < OPEN_COUNT)
            destination = route_rank[None, :] == route
            opened_indices = tl.where(
                destination,
                tl.where(opened, candidate_index, -1)[:, None],
                opened_indices,
            )
            opened_scores = tl.where(
                destination,
                tl.where(opened, candidate_coarse_score, -float("inf"))[
                    :, None
                ],
                opened_scores,
            )
            if RESIDUAL_MASS > 0.0:
                remaining_mass -= tl.exp(candidate_coarse_score - full_lse)
            remaining_scores = tl.where(
                route_rank[None, :] == candidate_rank[:, None],
                -float("inf"),
                remaining_scores,
            )
        top_indices = opened_indices
        top_scores = opened_scores

    if RESIDUAL_MASS == 0.0 and OPEN_COUNT > 1:
        # Match route_top8_scores_grouped(..., reorder_like_torch=True): keep
        # the boundary candidate last and sort the preceding selected slot
        # indices. Recursive leaf reductions consume routes in this order, so
        # set equality alone is not sufficient for numerical parity.
        boundary_index = tl.max(
            tl.where(
                route_rank[None, :] == OPEN_COUNT - 1,
                top_indices,
                -1,
            ),
            axis=1,
        )
        boundary_score = tl.max(
            tl.where(
                route_rank[None, :] == OPEN_COUNT - 1,
                top_scores,
                -float("inf"),
            ),
            axis=1,
        )
        remaining_indices = tl.where(
            route_rank[None, :] < OPEN_COUNT - 1,
            top_indices,
            0x7FFFFFFF,
        )
        reordered_indices = tl.full(
            (ROW_COUNT, ROUTE_COUNT), -1, tl.int32
        )
        reordered_scores = tl.full(
            (ROW_COUNT, ROUTE_COUNT), -float("inf"), tl.float32
        )
        for output_rank in tl.static_range(0, OPEN_COUNT - 1):
            best_index = tl.min(remaining_indices, axis=1)
            best_score = tl.max(
                tl.where(
                    top_indices == best_index[:, None],
                    top_scores,
                    -float("inf"),
                ),
                axis=1,
            )
            destination = route_rank[None, :] == output_rank
            reordered_indices = tl.where(
                destination, best_index[:, None], reordered_indices
            )
            reordered_scores = tl.where(
                destination, best_score[:, None], reordered_scores
            )
            remaining_indices = tl.where(
                remaining_indices == best_index[:, None],
                0x7FFFFFFF,
                remaining_indices,
            )
        boundary_destination = route_rank[None, :] == OPEN_COUNT - 1
        top_indices = tl.where(
            boundary_destination,
            boundary_index[:, None],
            reordered_indices,
        )
        top_scores = tl.where(
            boundary_destination,
            boundary_score[:, None],
            reordered_scores,
        )

    if STABLE_RECOMPUTE:
        # When selected states dominate the softmax, subtracting their mass
        # from the complete field catastrophically cancels to zero. Re-stream
        # the same logits in this kernel while masking selected states so the
        # coarse remainder stays well-conditioned without another launch.
        maximum = tl.where(query_valid, -float("inf"), 0.0).to(tl.float32)
        denominator = tl.where(query_valid, 0.0, 1.0).to(tl.float32)
        accumulator = tl.zeros((ROW_COUNT, VALUE_BLOCK_DIM), tl.float32)
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
            mean_values = (
                values.to(tl.float32) / count[:, None]
            ).to(values.dtype)
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
            routed = tl.zeros((ROW_COUNT, BLOCK_N), dtype=tl.int1)
            for route in tl.static_range(0, OPEN_COUNT):
                selected_slot = tl.max(
                    tl.where(
                        route_rank[None, :] == route,
                        top_indices,
                        -1,
                    ),
                    axis=1,
                )
                routed |= slot[None, :] == selected_slot[:, None]
            valid = query_valid[:, None] & state_valid[None, :] & ~routed
            scores = tl.where(valid, scores, -float("inf"))
            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            denominator = denominator * correction + tl.sum(
                probabilities, axis=1
            )
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
                    mask=token_valid[:, None]
                    & (tail_dim[None, :] < HEAD_DIM),
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
            scores = SCALE * tl.dot(
                queries, tl.trans(keys), out_dtype=tl.float32
            )
            if HEAD_TAIL_BLOCK_DIM > 0:
                scores += SCALE * tl.dot(
                    tail_queries,
                    tl.trans(tail_keys),
                    out_dtype=tl.float32,
                )
            visible = token[None, :] <= query[:, None] + local_offset
            valid = query_valid[:, None] & token_valid[None, :] & visible
            scores = tl.where(valid, scores, -float("inf"))
            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, block_maximum)
            correction = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            denominator = denominator * correction + tl.sum(
                probabilities, axis=1
            )
            accumulator = accumulator * correction[:, None] + tl.dot(
                probabilities.to(values.dtype), values, out_dtype=tl.float32
            )
            maximum = new_maximum
    elif not ROUTE_ONLY:
        # The streamed field included every low-resolution state summary.
        # Remove selected summaries so exact leaves replace them once.
        for route in tl.static_range(0, OPEN_COUNT):
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
            selected_valid = query_valid & (selected_slot >= 0)
            selected_count = tl.load(
                counts
                + batch * COUNT_BATCH_STRIDE
                + kv_head * COUNT_HEAD_STRIDE
                + selected_slot * COUNT_TOKEN_STRIDE,
                mask=selected_valid,
                other=1.0,
            ).to(tl.float32)
            selected_values = tl.load(
                state_v
                + batch * STATE_V_BATCH_STRIDE
                + kv_head * STATE_V_HEAD_STRIDE
                + selected_slot[:, None] * STATE_V_TOKEN_STRIDE
                + value_dim[None, :],
                mask=selected_valid[:, None]
                & (value_dim[None, :] < VALUE_DIM),
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

    if not ROUTE_ONLY:
        output_row = (
            (batch * QUERY_HEADS + query_head) * query_len + query
        ).to(tl.int64)
        has_mass = denominator > 0.0
        tl.store(
            output + output_row[:, None] * VALUE_DIM + value_dim[None, :],
            tl.where(
                has_mass[:, None], accumulator / denominator[:, None], 0.0
            ),
            mask=query_valid[:, None] & (value_dim[None, :] < VALUE_DIM),
        )
        tl.store(
            lse + output_row,
            tl.where(
                has_mass,
                maximum + tl.log(denominator),
                -float("inf"),
            ),
            mask=query_valid,
        )
    tl.store(
        top_slots
        + batch * TOP_BATCH_STRIDE
        + query_head[:, None] * TOP_HEAD_STRIDE
        + query[:, None] * TOP_QUERY_STRIDE
        + route_rank[None, :],
        top_indices,
        mask=query_valid[:, None] & (route_rank[None, :] < OPEN_COUNT),
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
    do_not_specialize=["active_groups", "query_len", "state_len"],
    do_not_specialize_on_alignment=[
        "Q_BATCH_STRIDE",
        "Q_HEAD_STRIDE",
        "STATE_BATCH_STRIDE",
        "STATE_HEAD_STRIDE",
        "COUNT_BATCH_STRIDE",
        "COUNT_HEAD_STRIDE",
        "active_groups",
        "query_len",
        "state_len",
    ],
)
def _route_state_group_candidates_kernel(
    q,
    state_k,
    counts,
    partial_scores,
    partial_indices,
    Q_BATCH_STRIDE,
    Q_HEAD_STRIDE,
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
    COUNT_BIAS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    TOPK: tl.constexpr,
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
    scores += COUNT_BIAS * tl.log(count)[None, :]
    scores = tl.where(
        query_valid[:, None]
        & slot_valid[None, :]
        & (slot[None, :] >= PROTECTED_LEN),
        scores,
        -float("inf"),
    )

    partial_base = (
        batch * PARTIAL_BATCH_STRIDE
        + q_head * PARTIAL_HEAD_STRIDE
        + query * PARTIAL_QUERY_STRIDE
        + state_group * PARTIAL_GROUP_STRIDE
    )
    for rank in tl.static_range(0, TOPK):
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


@triton.jit(
    do_not_specialize=["active_groups", "query_len", "state_len"],
    do_not_specialize_on_alignment=[
        "LOGIT_BATCH_STRIDE",
        "LOGIT_HEAD_STRIDE",
        "LOGIT_QUERY_STRIDE",
        "COUNT_BATCH_STRIDE",
        "COUNT_HEAD_STRIDE",
        "active_groups",
        "query_len",
        "state_len",
    ],
)
def _route_score_group_candidates_kernel(
    logits,
    counts,
    partial_scores,
    partial_indices,
    partial_lse,
    LOGIT_BATCH_STRIDE,
    LOGIT_HEAD_STRIDE,
    LOGIT_QUERY_STRIDE,
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
    COUNT_BIAS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PROTECTED_LEN: tl.constexpr,
    TOPK: tl.constexpr,
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
    scores += COUNT_BIAS * tl.log(count)[None, :]
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

    scores = tl.where(slot[None, :] >= PROTECTED_LEN, scores, -float("inf"))
    partial_base = (
        batch * PARTIAL_BATCH_STRIDE
        + q_head * PARTIAL_HEAD_STRIDE
        + query * PARTIAL_QUERY_STRIDE
        + state_group * PARTIAL_GROUP_STRIDE
    )
    for rank in tl.static_range(0, TOPK):
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


@triton.jit(
    do_not_specialize=["query_len", "active_groups"],
    do_not_specialize_on_alignment=["query_len", "active_groups"],
)
def _reduce_route_group_candidates_kernel(
    partial_scores,
    partial_indices,
    partial_lse,
    output,
    state_lse,
    PARTIAL_BATCH_STRIDE: tl.constexpr,
    PARTIAL_HEAD_STRIDE: tl.constexpr,
    PARTIAL_QUERY_STRIDE: tl.constexpr,
    PARTIAL_GROUP_STRIDE: tl.constexpr,
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
    GROUP_CANDIDATES: tl.constexpr,
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
    candidate_valid = candidate < active_groups * GROUP_CANDIDATES
    candidate_group = candidate // GROUP_CANDIDATES
    candidate_rank = candidate - candidate_group * GROUP_CANDIDATES
    partial_offset = (
        batch * PARTIAL_BATCH_STRIDE
        + q_head * PARTIAL_HEAD_STRIDE
        + query[:, None] * PARTIAL_QUERY_STRIDE
        + candidate_group[None, :] * PARTIAL_GROUP_STRIDE
        + candidate_rank[None, :]
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
    for rank in tl.static_range(0, TOPK):
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


@triton.jit(
    do_not_specialize=["query_len", "state_len"],
    do_not_specialize_on_alignment=[
        "LOGIT_BATCH_STRIDE",
        "LOGIT_HEAD_STRIDE",
        "LOGIT_QUERY_STRIDE",
        "TOP_BATCH_STRIDE",
        "TOP_HEAD_STRIDE",
        "LSE_BATCH_STRIDE",
        "LSE_HEAD_STRIDE",
        "LOCAL_LSE_BATCH_STRIDE",
        "LOCAL_LSE_HEAD_STRIDE",
        "query_len",
        "state_len",
    ],
)
def _apply_residual_mass_opening_kernel(
    logits,
    counts,
    top_slots,
    state_lse,
    local_lse,
    LOGIT_BATCH_STRIDE,
    LOGIT_HEAD_STRIDE,
    LOGIT_QUERY_STRIDE,
    COUNT_BATCH_STRIDE: tl.constexpr,
    COUNT_HEAD_STRIDE: tl.constexpr,
    COUNT_TOKEN_STRIDE: tl.constexpr,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    TOP_QUERY_STRIDE: tl.constexpr,
    LSE_BATCH_STRIDE,
    LSE_HEAD_STRIDE,
    LSE_QUERY_STRIDE: tl.constexpr,
    LOCAL_LSE_BATCH_STRIDE,
    LOCAL_LSE_HEAD_STRIDE,
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
        "overflow_key_norms": torch.empty(
            score_shape, dtype=torch.float32, device=overflow_k.device
        ),
    }


def bipartite_reduce_overflow(
    overflow_k: torch.Tensor,
    overflow_v: torch.Tensor,
    *,
    block_size: int = 32,
    balanced: bool = False,
    salt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce fixed overflow blocks 2:1 and return original-to-rep membership.

    The first half of each block supplies representatives. Every key in the
    second half routes to its most similar representative. Returned K/V are
    sums and ``counts`` records their multiplicity, preserving coarse mass.
    """

    if not overflow_k.is_cuda or not overflow_v.is_cuda:
        raise ValueError("bipartite overflow reduction requires CUDA tensors")
    if overflow_k.ndim != 4 or overflow_v.ndim != 4:
        raise ValueError("bipartite overflow tensors must be rank four")
    if overflow_k.shape[:3] != overflow_v.shape[:3]:
        raise ValueError("bipartite overflow K/V shapes differ")
    if not overflow_k.is_contiguous() or not overflow_v.is_contiguous():
        raise ValueError("bipartite overflow K/V must be contiguous")
    if block_size <= 0 or block_size % 2:
        raise ValueError("bipartite block size must be positive and even")
    overflow_len = int(overflow_k.size(2))
    if overflow_len == 0 or overflow_len % block_size:
        raise ValueError("overflow length must be a nonzero block-size multiple")
    half_block = block_size // 2
    batch, kv_heads, _, head_dim = overflow_k.shape
    value_dim = int(overflow_v.size(-1))
    reduced_len = overflow_len // 2
    reduced_k = torch.empty(
        batch,
        kv_heads,
        reduced_len,
        head_dim,
        dtype=overflow_k.dtype,
        device=overflow_k.device,
    )
    reduced_v = torch.empty(
        batch,
        kv_heads,
        reduced_len,
        value_dim,
        dtype=overflow_v.dtype,
        device=overflow_v.device,
    )
    counts = torch.empty(
        batch,
        kv_heads,
        reduced_len,
        1,
        dtype=torch.float32,
        device=overflow_k.device,
    )
    membership = torch.empty(
        batch,
        kv_heads,
        overflow_len,
        dtype=torch.long,
        device=overflow_k.device,
    )
    rows = batch * kv_heads
    _bipartite_reduce_overflow_kernel[(rows, overflow_len // block_size)](
        overflow_k,
        overflow_v,
        reduced_k,
        reduced_v,
        counts,
        membership,
        overflow_k.stride(1),
        overflow_v.stride(1),
        membership.stride(1),
        OVERFLOW_LEN=overflow_len,
        BLOCK_SIZE=block_size,
        HALF_BLOCK=half_block,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        BALANCED=balanced,
        SALT=salt,
        **_launch_kwargs(4),
    )
    return reduced_k, reduced_v, counts, membership


def balanced_bipartite_reduce_2to1(
    key_sum: torch.Tensor,
    value_sum: torch.Tensor,
    counts: torch.Tensor,
    *,
    salt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contract one or more rows 2:1 using a balanced interleaved partition."""

    if not all(tensor.is_cuda for tensor in (key_sum, value_sum, counts)):
        raise ValueError("balanced bipartite reduction requires CUDA tensors")
    if key_sum.ndim != 4 or value_sum.ndim != 4 or counts.ndim != 4:
        raise ValueError("balanced bipartite inputs must be rank four")
    if key_sum.shape[:3] != value_sum.shape[:3] or counts.shape[:3] != key_sum.shape[:3]:
        raise ValueError("balanced bipartite input prefixes differ")
    if int(counts.size(-1)) != 1:
        raise ValueError("balanced bipartite counts must have a singleton feature")
    if not all(tensor.is_contiguous() for tensor in (key_sum, value_sum, counts)):
        raise ValueError("balanced bipartite inputs must be contiguous")
    batch, heads, token_len, head_dim = key_sum.shape
    if token_len < 2:
        raise ValueError("balanced bipartite reduction needs at least two tokens")
    value_dim = int(value_sum.size(-1))
    anchor_count = (token_len + 1) // 2
    source_count = token_len // 2
    rows = batch * heads
    destination = torch.empty(
        batch,
        heads,
        source_count,
        dtype=torch.int32,
        device=key_sum.device,
    )
    membership = torch.empty(
        batch,
        heads,
        token_len,
        dtype=torch.long,
        device=key_sum.device,
    )
    reduced_k = torch.empty(
        batch,
        heads,
        anchor_count,
        head_dim,
        dtype=key_sum.dtype,
        device=key_sum.device,
    )
    reduced_v = torch.empty(
        batch,
        heads,
        anchor_count,
        value_dim,
        dtype=value_sum.dtype,
        device=value_sum.device,
    )
    reduced_counts = torch.empty(
        batch,
        heads,
        anchor_count,
        1,
        dtype=torch.float32,
        device=key_sum.device,
    )
    if token_len % 2 == 0:
        reduced_k_fp32 = torch.zeros_like(reduced_k, dtype=torch.float32)
        reduced_v_fp32 = torch.zeros_like(reduced_v, dtype=torch.float32)
        reduced_counts.zero_()
        _balanced_bipartite_route_kernel[
            (rows, triton.cdiv(source_count, 16))
        ](
            key_sum,
            counts,
            destination,
            membership,
            key_sum.stride(1),
            counts.stride(1),
            destination.stride(1),
            membership.stride(1),
            TOKEN_LEN=token_len,
            ANCHOR_COUNT=anchor_count,
            SOURCE_COUNT=source_count,
            HEAD_DIM=head_dim,
            BLOCK_M=16,
            BLOCK_N=32,
            SALT=salt,
            **_launch_kwargs(4),
        )
        _balanced_bipartite_atomic_reduce_kernel[
            (
                rows,
                triton.cdiv(source_count, 4),
                triton.cdiv(max(head_dim, value_dim), 32),
            )
        ](
            key_sum,
            value_sum,
            counts,
            destination,
            reduced_k_fp32,
            reduced_v_fp32,
            reduced_counts,
            membership,
            key_sum.stride(1),
            value_sum.stride(1),
            counts.stride(1),
            destination.stride(1),
            reduced_k_fp32.stride(1),
            reduced_v_fp32.stride(1),
            reduced_counts.stride(1),
            membership.stride(1),
            SOURCE_COUNT=source_count,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            BLOCK_M=4,
            BLOCK_D=32,
            SALT=salt,
            **_launch_kwargs(4),
        )
        _balanced_bipartite_finalize_kernel[
            (
                rows,
                triton.cdiv(anchor_count, 8),
                triton.cdiv(max(head_dim, value_dim), 32),
            )
        ](
            reduced_k_fp32,
            reduced_v_fp32,
            reduced_k,
            reduced_v,
            reduced_k_fp32.stride(1),
            reduced_v_fp32.stride(1),
            ANCHOR_COUNT=anchor_count,
            HEAD_DIM=head_dim,
            VALUE_DIM=value_dim,
            BLOCK_A=8,
            BLOCK_D=32,
            **_launch_kwargs(4),
        )
        return reduced_k, reduced_v, reduced_counts, membership

    _balanced_bipartite_route_kernel[
        (rows, triton.cdiv(source_count, 16))
    ](
        key_sum,
        counts,
        destination,
        membership,
        key_sum.stride(1),
        counts.stride(1),
        destination.stride(1),
        membership.stride(1),
        TOKEN_LEN=token_len,
        ANCHOR_COUNT=anchor_count,
        SOURCE_COUNT=source_count,
        HEAD_DIM=head_dim,
        BLOCK_M=16,
        BLOCK_N=32,
        SALT=salt,
        **_launch_kwargs(4),
    )
    for values, output, feature_dim in (
        (key_sum, reduced_k, head_dim),
        (value_sum, reduced_v, value_dim),
    ):
        feature_tiles = triton.cdiv(feature_dim, 32)
        _balanced_bipartite_reduce_feature_kernel[
            (rows, triton.cdiv(anchor_count, 8) * feature_tiles)
        ](
            values,
            destination,
            output,
            values.stride(1),
            destination.stride(1),
            output.stride(1),
            TOKEN_LEN=token_len,
            ANCHOR_COUNT=anchor_count,
            SOURCE_COUNT=source_count,
            FEATURE_DIM=feature_dim,
            BLOCK_A=8,
            BLOCK_M=32,
            BLOCK_D=32,
            SALT=salt,
            **_launch_kwargs(4),
        )
    _balanced_bipartite_reduce_count_kernel[
        (rows, triton.cdiv(anchor_count, 64))
    ](
        counts,
        destination,
        reduced_counts,
        membership,
        counts.stride(1),
        destination.stride(1),
        reduced_counts.stride(1),
        membership.stride(1),
        TOKEN_LEN=token_len,
        ANCHOR_COUNT=anchor_count,
        SOURCE_COUNT=source_count,
        BLOCK_A=64,
        BLOCK_M=32,
        SALT=salt,
        **_launch_kwargs(4),
    )
    return reduced_k, reduced_v, reduced_counts, membership


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
    _constituent_rms_kernel[
        (batch, heads, triton.cdiv(token_len, block_m))
    ](
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
            torch.empty(
                state_k.shape[:3], dtype=torch.float32, device=state_k.device
            )
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
    overflow_norms = buffers["overflow_key_norms"]
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
        scan_state = prepared_append if coherence else prepared_route
        review_state = prepared_route
        scan_scale = prepared_scale
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
            invalid = counts[..., :state_len, 0].le(0.5).unsqueeze(-2)
            route_scores_dense.masked_fill_(invalid, float("-inf"))
            if append_scores_dense is not None:
                append_scores_dense.masked_fill_(invalid, float("-inf"))
                select_score = append_scores_dense.max(dim=-1).values
            route_scores_dense[..., :sink_len] = float("-inf")
            route_score, route_index = route_scores_dense.max(dim=-1)
            return route_score, route_index, select_score
    elif fused_coherence:
        scan_state = state_k
        review_state = state_k
        scan_scale = key_norm_sums
        block_m = max(block_m, 32)
    else:
        scan_state = state_k
        review_state = state_k
        scan_scale = counts
    _streaming_state_maxsim_kernel[
        (batch, kv_heads, triton.cdiv(overflow_len, block_m))
    ](
        overflow_k,
        scan_state,
        review_state,
        scan_scale,
        counts,
        route_scores,
        route_indices,
        select_scores,
        overflow_norms,
        OVERFLOW_BATCH_STRIDE=overflow_k.stride(0),
        OVERFLOW_HEAD_STRIDE=overflow_k.stride(1),
        OVERFLOW_TOKEN_STRIDE=overflow_k.stride(2),
        STATE_BATCH_STRIDE=scan_state.stride(0),
        STATE_HEAD_STRIDE=scan_state.stride(1),
        STATE_TOKEN_STRIDE=scan_state.stride(2),
        REVIEW_STATE_BATCH_STRIDE=review_state.stride(0),
        REVIEW_STATE_HEAD_STRIDE=review_state.stride(1),
        REVIEW_STATE_TOKEN_STRIDE=review_state.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0),
        COUNT_HEAD_STRIDE=counts.stride(1),
        COUNT_TOKEN_STRIDE=counts.stride(2),
        SCALE_BATCH_STRIDE=scan_scale.stride(0),
        SCALE_HEAD_STRIDE=scan_scale.stride(1),
        SCALE_TOKEN_STRIDE=scan_scale.stride(2),
        OUTPUT_BATCH_STRIDE=route_scores.stride(0),
        OUTPUT_HEAD_STRIDE=route_scores.stride(1),
        OUTPUT_TOKEN_STRIDE=route_scores.stride(2),
        overflow_len=overflow_len,
        state_len=state_len,
        HEAD_DIM=head_dim,
        SINK_LEN=sink_len,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PREPARED=prepared,
        REVIEW_ROUTE=coherence and not fused_coherence,
        FUSED_COHERENCE=fused_coherence,
        STORE_OVERFLOW_NORMS=fused_coherence,
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
    precompute_mean_values: bool = False,
    head_major: bool | None = None,
    max_grouped_rows: int = 8,
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
    value_dim = int(state_v.size(-1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query heads do not match the requested GQA grouping")
    if tuple(route_logits.shape) != (batch, query_heads, query_len, state_len):
        raise ValueError("routing logits have the wrong shape")
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
    if (
        block_m <= 0
        or block_n <= 0
        or max_grouped_rows <= 0
    ):
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
    # shared memory.  Keep that tile bounded for high-GQA models instead of
    # relying on stride specialization to keep the accumulator in registers.
    if head_major is not True and kv_group_size * block_m > max_grouped_rows:
        block_m = max(1, max_grouped_rows // kv_group_size)
    grouped_rows = kv_group_size * block_m
    value_block_dim = triton.next_power_of_2(value_dim)
    if head_major is None:
        # GQA grouping keeps one value accumulator per grouped query row.
        # Large groups with wide values can therefore exceed the device's
        # shared-memory budget even though each individual head is ordinary
        # attention (for example Qwen3.5-35B: 8 * 16 * 256 * fp32 = 128 KiB).
        # Split those cases by query head. The kernel math is unchanged and
        # the extra programs expose useful parallelism on these larger models.
        grouped_accumulator_bytes = grouped_rows * value_block_dim * 4
        head_major = (
            grouped_rows & (grouped_rows - 1) != 0
            or grouped_accumulator_bytes > 48 * 1024
        )
    row_count = block_m if head_major else grouped_rows
    if row_count & (row_count - 1):
        raise ValueError(
            "head-major coarse attention requires a power-of-two query tile"
        )

    kernel_state_v = state_v
    if precompute_mean_values:
        active_counts = counts[..., :state_len, :].clamp_min(1.0)
        kernel_state_v = (
            state_v[..., :state_len, :].float() / active_counts
        ).to(state_v.dtype).contiguous()

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
        kernel_state_v,
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
    route_count_bias: float = 1.0,
    topk: int = 8,
    protected_len: int = 0,
    max_leaf_tokens: int | None = None,
    residual_local_lse: torch.Tensor | None = None,
    residual_mass: float | None = None,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 8,
    head_major: bool | None = None,
    stable_recompute: bool = True,
    route_only: bool = False,
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
    value_dim = int(state_v.size(-1))
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query heads do not match the requested GQA grouping")
    if tuple(route_logits.shape) != (batch, query_heads, query_len, state_len):
        raise ValueError("routing logits have the wrong shape")
    if not 0 < topk <= 8:
        raise ValueError("fused LOD prefill routing requires top-k in [1, 8]")
    if max_leaf_tokens is not None and max_leaf_tokens <= 0:
        raise ValueError("maximum routed leaf count must be positive")
    if residual_mass is not None and not 0.0 < residual_mass < 1.0:
        raise ValueError("residual route mass must lie strictly between zero and one")
    if route_only and residual_mass is not None:
        raise ValueError("route-only fusion does not support residual-mass opening")
    if state_len < topk:
        raise ValueError("active state is smaller than the requested route count")
    if protected_len < 0 or protected_len + topk > state_len:
        raise ValueError("protected state leaves too few routing candidates")
    if state_len > int(state_v.size(2)) or state_len > int(counts.size(2)):
        raise ValueError("active state exceeds the supplied storage")
    if (local_len and local_len < query_len) or int(local_v.size(2)) != local_len:
        raise ValueError("local attention has an invalid length")
    if int(local_k.size(1)) != kv_heads or int(local_v.size(1)) != kv_heads:
        raise ValueError("local and state KV heads differ")
    if int(local_k.size(-1)) != head_dim:
        raise ValueError("local key dimension differs from the query")
    if int(local_v.size(-1)) != value_dim:
        raise ValueError("local and state value dimensions differ")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("coarse-attention tile sizes must be positive")

    if head_dim > 512 or value_dim > 256:
        block_m = min(block_m, 4)
        num_warps = min(num_warps, 4)
        head_major = True
    grouped_rows = kv_group_size * block_m
    if head_major is None:
        head_major = grouped_rows & (grouped_rows - 1) != 0
    row_count = block_m if head_major else grouped_rows
    if row_count & (row_count - 1):
        raise ValueError(
            "head-major fused routing requires a power-of-two query tile"
        )

    padded_topk = 1 << (topk - 1).bit_length()
    top_slots = torch.empty(
        batch,
        query_heads,
        query_len,
        topk,
        dtype=torch.long,
        device=q.device,
    )
    if route_only:
        output = torch.empty(1, dtype=q.dtype, device=q.device)
        lse = torch.empty(
            batch,
            query_heads,
            1,
            dtype=torch.float32,
            device=q.device,
        )
        local_k = local_k[..., :0, :].contiguous()
        local_v = local_v[..., :0, :].contiguous()
        local_len = 0
    else:
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
    use_external_local_lse = residual_local_lse is not None and local_len == 0
    if residual_local_lse is not None:
        if residual_mass is None:
            raise ValueError("a residual local LSE requires a residual-mass threshold")
        if tuple(residual_local_lse.shape) != (batch, query_heads, query_len):
            raise ValueError("residual local LSE has the wrong shape")
        if not residual_local_lse.is_cuda or not residual_local_lse.is_contiguous():
            raise ValueError("residual local LSE must be contiguous on the GPU")
        residual_lse = residual_local_lse if use_external_local_lse else lse
    else:
        if residual_mass is not None and local_len == 0:
            raise ValueError(
                "residual-mass routing requires either local KV or its external LSE"
            )
        residual_lse = lse
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
    _route_logits_topk_coarse_attention_kernel[grid](
        q,
        route_logits,
        state_v,
        counts,
        local_k,
        local_v,
        residual_lse,
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
        residual_lse.stride(0),
        residual_lse.stride(1),
        residual_lse.stride(2),
        top_slots.stride(0),
        top_slots.stride(1),
        top_slots.stride(2),
        query_len,
        state_len,
        local_len,
        local_len - query_len,
        QUERY_HEADS=query_heads,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_MAJOR=head_major,
        ROW_COUNT=row_count,
        STABLE_RECOMPUTE=stable_recompute and not route_only,
        ROUTE_ONLY=route_only,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=head_block_dim,
        HEAD_TAIL_BLOCK_DIM=head_tail_block_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        ROUTE_COUNT=padded_topk,
        OPEN_COUNT=topk,
        MAX_LEAF_TOKENS=max_leaf_tokens or 0,
        PROTECTED_LEN=protected_len,
        RESIDUAL_MASS=residual_mass or 0.0,
        USE_EXTERNAL_LOCAL_LSE=use_external_local_lse,
        SCALE=scale,
        ROUTE_COUNT_BIAS=route_count_bias,
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


def route_top8_state_grouped(
    q: torch.Tensor,
    state_k: torch.Tensor,
    counts: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    *,
    kv_group_size: int,
    scale: float,
    count_bias: float = 1.0,
    topk: int,
    state_len: int | None = None,
    protected_len: int = 0,
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
    if protected_len < 0 or protected_len + topk > state_len:
        raise ValueError("protected state leaves too few routing candidates")
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
        COUNT_BIAS=count_bias,
        KV_GROUP_SIZE=kv_group_size,
        HEAD_DIM=head_dim,
        PROTECTED_LEN=protected_len,
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        **_launch_kwargs(4),
    )
    candidate_block = triton.next_power_of_2(max_groups * topk)
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
        PARTIAL_GROUP_STRIDE=partial_scores.stride(3),
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
        GROUP_CANDIDATES=topk,
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
    count_bias: float = 1.0,
    topk: int,
    state_len: int | None = None,
    protected_len: int = 0,
    return_lse: bool = False,
    block_m: int | None = None,
    num_warps: int = 4,
    reorder_like_torch: bool = True,
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
    if protected_len < 0 or protected_len + topk > state_len:
        raise ValueError("protected state leaves too few routing candidates")
    if block_m is None:
        block_m = 16 if query_len > 1 else 1
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError("grouped routing query block must be a power of two")
    if query_len == 1 and block_m != 1:
        raise ValueError("decode grouped routing requires a one-row query block")
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
        COUNT_BIAS=count_bias,
        KV_GROUP_SIZE=kv_group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PROTECTED_LEN=protected_len,
        TOPK=topk,
        STORE_LSE=return_lse,
        **_launch_kwargs(num_warps),
    )
    candidate_block = triton.next_power_of_2(max_groups * topk)
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
        PARTIAL_GROUP_STRIDE=partial_scores.stride(3),
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
        GROUP_CANDIDATES=topk,
        MAX_GROUPS=lse_group_block,
        BLOCK_M=block_m,
        CANDIDATE_BLOCK=candidate_block,
        STORE_LSE=return_lse,
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


def merge_attention_branches(
    primary_out: torch.Tensor,
    primary_lse: torch.Tensor,
    secondary_out: torch.Tensor,
    secondary_lse: torch.Tensor,
    tertiary_out: torch.Tensor | None = None,
    tertiary_lse: torch.Tensor | None = None,
    *,
    output_buffer: torch.Tensor | None = None,
    block_m: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fuse the final LSE reduction of two or three materialized branches."""
    expected_output_shape = tuple(primary_out.shape)
    expected_lse_shape = tuple(primary_lse.shape)
    if (
        len(expected_output_shape) != 4
        or expected_output_shape[:-1] != expected_lse_shape
    ):
        raise ValueError("primary attention output and LSE shapes differ")
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
            raise ValueError("fused branch reduction requires CUDA tensors")
        if int(branch_out.stride(-1)) != 1:
            raise ValueError("fused branch outputs require contiguous head features")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError("fused branch reduction block size must be a power of two")

    include_tertiary = tertiary_out is not None
    tertiary_out = primary_out if tertiary_out is None else tertiary_out
    tertiary_lse = primary_lse if tertiary_lse is None else tertiary_lse
    output = torch.empty_like(primary_out) if output_buffer is None else output_buffer
    if (
        tuple(output.shape) != expected_output_shape
        or output.dtype != primary_out.dtype
        or output.device != primary_out.device
        or int(output.stride(-1)) != 1
        or _output_has_internal_overlap(output)
    ):
        raise ValueError("fused branch output buffer has incompatible geometry")
    batch, heads, query_len, head_dim = expected_output_shape
    _merge_attention_branches_kernel[
        (batch, heads, triton.cdiv(query_len, block_m))
    ](
        primary_out,
        primary_lse,
        secondary_out,
        secondary_lse,
        tertiary_out,
        tertiary_lse,
        output,
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
        HEAD_DIM=head_dim,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        INCLUDE_TERTIARY=include_tertiary,
        BLOCK_M=block_m,
        **_launch_kwargs(num_warps),
    )
    return output


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
    if int(q.stride(-1)) != 1 or int(sink_k.stride(-1)) != 1 or int(
        sink_v.stride(-1)
    ) != 1:
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
    "apply_residual_mass_opening",
    "balanced_bipartite_reduce_2to1",
    "constituent_rms",
    "merge_attention_branches",
    "merge_attention_branches_with_sink",
    "merge_state_in_place",
    "new_route_buffers",
    "new_state_delta_buffers",
    "new_state_maxsim_buffers",
    "prepare_state_clustering_keys",
    "route_top8_scores_grouped",
    "route_top8_state_grouped",
    "route_logits_coarse_attention",
    "streaming_state_maxsim",
]
