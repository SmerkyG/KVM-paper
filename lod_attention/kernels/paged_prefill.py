"""Exact routed leaf attention used during LoD prefill."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ._paged_common import _lookup_page_id


@triton.jit
def _reduce_expert_route_attention_kernel(
    route_out,
    route_lse,
    top_slots,
    exact_out,
    exact_lse,
    ROUTE_COUNT: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
):
    """Merge routed expert outputs with their exact log-sum-exp masses."""
    row = tl.program_id(0).to(tl.int64)
    route = tl.arange(0, ROUTE_BLOCK)
    dimension = tl.arange(0, VALUE_BLOCK_DIM)
    valid_route = route < ROUTE_COUNT
    slot = tl.load(
        top_slots + row * ROUTE_COUNT + route,
        mask=valid_route,
        other=-1,
    )
    valid_route &= slot >= 0
    lse = tl.load(
        route_lse + row * ROUTE_COUNT + route,
        mask=valid_route,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(lse, axis=0)
    # A mass cutoff is allowed to open no experts for a query.  In that case
    # maximum == -inf, and evaluating ``-inf - -inf`` would otherwise poison
    # the reduction even though every route has zero weight.  Form the
    # all-empty identity explicitly; this also prevents uninitialized output
    # lanes for compacted-away routes from being multiplied by NaNs.
    route_has_mass = valid_route & (lse != -float("inf"))
    safe_maximum = tl.where(maximum != -float("inf"), maximum, 0.0)
    weight = tl.where(
        route_has_mass,
        tl.exp(lse - safe_maximum),
        0.0,
    )
    denominator = tl.sum(weight, axis=0)
    values = tl.load(
        route_out
        + (row * ROUTE_COUNT + route[:, None]) * VALUE_DIM
        + dimension[None, :],
        mask=route_has_mass[:, None] & (dimension[None, :] < VALUE_DIM),
        other=0.0,
    ).to(tl.float32)
    safe_denominator = tl.maximum(denominator, 1.0)
    output = tl.sum(values * weight[:, None], axis=0) / safe_denominator
    tl.store(
        exact_out + row * VALUE_DIM + dimension,
        output,
        mask=dimension < VALUE_DIM,
    )
    tl.store(
        exact_lse + row,
        tl.where(
            denominator > 0.0,
            safe_maximum + tl.log(denominator),
            -float("inf"),
        ),
    )


@triton.jit(
    do_not_specialize=["PROGRAM_OFFSET"],
    do_not_specialize_on_alignment=["PROGRAM_OFFSET"],
)
def _paged_leaf_attention_kernel(
    q,
    q_scales,
    packed_route_row,
    block_expert,
    block_starts,
    page_k,
    page_v,
    page_indices,
    page_k_scales,
    page_v_scales,
    quantized_leaf_k,
    quantized_leaf_v,
    page_sum_k,
    page_sum_v,
    quantized_page_sum_k,
    quantized_page_sum_v,
    page_sum_k_scales,
    page_sum_v_scales,
    page_counts,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    q_lengths,
    cu_q,
    expert_kv_row,
    expert_slot,
    out,
    lse,
    PROGRAM_OFFSET,
    program_limit,
    experts,
    PAGE_CAPACITY: tl.constexpr,
    LEAF_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_N: tl.constexpr,
    PARTIAL_OUTPUT: tl.constexpr,
    INT8_MMA: tl.constexpr,
    INT8_PV_MMA: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    QUANT_TOKEN_GROUP_SIZE: tl.constexpr,
    QUANTIZED_SUMMARIES: tl.constexpr,
    INDEXED: tl.constexpr,
    PROGRAMS_POINTER: tl.constexpr,
    SEARCH_BLOCKS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    split_program = tl.program_id(0).to(tl.int64)
    local_program = split_program // SPLIT_N
    split = split_program - local_program * SPLIT_N
    if PROGRAMS_POINTER:
        active_programs = tl.load(program_limit).to(tl.int64)
    else:
        active_programs = program_limit
    valid_program = local_program < active_programs
    program = local_program + PROGRAM_OFFSET
    if SEARCH_BLOCKS:
        lower = tl.full((), 0, tl.int64)
        upper = experts.to(tl.int64)
        for _ in tl.static_range(0, SEARCH_STEPS):
            searching = lower < upper
            middle = (lower + upper) // 2
            boundary = tl.load(
                block_starts + middle + 1,
                mask=valid_program & searching,
                other=active_programs,
            ).to(tl.int64)
            move_right = searching & (local_program >= boundary)
            lower = tl.where(move_right, middle + 1, lower)
            upper = tl.where(searching & ~move_right, middle, upper)
        expert = tl.where(valid_program, lower, 0)
        query_block = local_program - tl.load(
            block_starts + expert, mask=valid_program, other=0
        ).to(tl.int64)
    else:
        expert = tl.load(block_expert + program)
        query_block = program - tl.load(block_starts + expert)
    query_count = tl.load(q_lengths + expert, mask=valid_program, other=0)
    query_offset = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_query = valid_program & (query_offset < query_count)
    packed_begin = tl.load(cu_q + expert).to(tl.int64)
    packed_row = packed_begin + query_offset.to(tl.int64)
    route_row = tl.load(
        packed_route_row + packed_row,
        mask=valid_query,
        other=0,
    ).to(tl.int64)
    query_row = route_row // ROUTE_COUNT

    kv_row = tl.load(expert_kv_row + expert).to(tl.int64)
    slot = tl.load(expert_slot + expert).to(tl.int64)

    head_offset = tl.arange(0, HEAD_DIM)
    value_offset = tl.arange(0, VALUE_DIM)
    q_block = tl.load(
        q + query_row[:, None] * HEAD_DIM + head_offset[None, :],
        mask=valid_query[:, None],
        other=0.0,
    )
    if INT8_MMA:
        q_mma = q_block.to(tl.int8)
        q_scale = tl.load(q_scales + query_row, mask=valid_query, other=1.0).to(
            tl.float32
        )
    key_count = tl.load(slot_lengths + kv_row * STATE_CAPACITY + slot).to(tl.int32)
    if HASH_PROBES == 0:
        page_table = (
            slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
        )
    maximum = tl.where(valid_query, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.where(valid_query, 0.0, 1.0).to(tl.float32)
    accumulator = tl.zeros((BLOCK_M, VALUE_DIM), tl.float32)
    token_offset = tl.arange(0, BLOCK_N)

    keys_per_split = (key_count + SPLIT_N - 1) // SPLIT_N
    split_begin = split * keys_per_split
    split_count = tl.maximum(tl.minimum(keys_per_split, key_count - split_begin), 0)
    for key_begin in tl.range(0, split_count, BLOCK_N, num_stages=1):
        logical_key = split_begin + key_begin + token_offset
        valid_key = (key_begin + token_offset) < split_count
        page_aligned_quant = (
            QUANT_BITS
            and BLOCK_N == PAGE_SIZE
            and QUANT_TOKEN_GROUP_SIZE == PAGE_SIZE
            and SPLIT_N == 1
        )
        if page_aligned_quant:
            # Residual INT4 always visits one complete virtual page at a time.
            # Resolve its directory entry and page metadata once, rather than
            # issuing the same loads independently for all sixteen leaf lanes.
            page_ordinal_scalar = key_begin // PAGE_SIZE
            within_page = token_offset.to(tl.int64)
            valid_page = key_begin < split_count
            if HASH_PROBES == 0:
                page_id_scalar = tl.load(
                    page_table + page_ordinal_scalar, mask=valid_page, other=0
                ).to(tl.int64)
            else:
                page_id_scalar = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal_scalar,
                    valid_page,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            page_id = page_id_scalar + tl.zeros((BLOCK_N,), tl.int64)
        else:
            page_ordinal = logical_key // PAGE_SIZE
            within_page = logical_key % PAGE_SIZE
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal, mask=valid_key, other=0
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid_key,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
        physical_token = (kv_row * PAGE_CAPACITY + page_id) * PAGE_SIZE + within_page
        if INDEXED:
            leaf_index = tl.load(
                page_indices + physical_token, mask=valid_key, other=0
            ).to(tl.int64)
            storage_token = kv_row * LEAF_CAPACITY + leaf_index
        else:
            storage_token = physical_token
        if QUANT_BITS:
            packed_head_offset = head_offset // 2
            packed_value_offset = value_offset // 2
            packed_keys = tl.load(
                quantized_leaf_k
                + storage_token[None, :] * (HEAD_DIM // 2)
                + packed_head_offset[:, None],
                mask=valid_key[None, :],
                other=0,
            ).to(tl.int32)
            packed_values = tl.load(
                quantized_leaf_v
                + storage_token[:, None] * (VALUE_DIM // 2)
                + packed_value_offset[None, :],
                mask=valid_key[:, None],
                other=0,
            ).to(tl.int32)
            key_shift = (head_offset & 1) * 4
            value_shift = (value_offset & 1) * 4
            key_code = ((packed_keys >> key_shift[:, None]) & 15) - 8
            value_code = ((packed_values >> value_shift[None, :]) & 15) - 8
            if page_aligned_quant:
                page_valid_scalar = key_begin < split_count
                page_row = kv_row * PAGE_CAPACITY + tl.sum(
                    tl.where(token_offset == 0, page_id, 0), axis=0
                )
                key_scale_row_scalar = page_row * (HEAD_DIM // QUANT_GROUP_SIZE)
                value_scale_row_scalar = page_row * (VALUE_DIM // QUANT_GROUP_SIZE)
                key_scale_vector = tl.load(
                    page_k_scales
                    + key_scale_row_scalar
                    + head_offset // QUANT_GROUP_SIZE,
                    mask=page_valid_scalar,
                    other=0.0,
                ).to(tl.float32)
                value_scale_vector = tl.load(
                    page_v_scales
                    + value_scale_row_scalar
                    + value_offset // QUANT_GROUP_SIZE,
                    mask=page_valid_scalar,
                    other=0.0,
                ).to(tl.float32)
                page_count_scalar = tl.load(
                    page_counts + page_row,
                    mask=page_valid_scalar,
                    other=1,
                ).to(tl.float32)
                if QUANTIZED_SUMMARIES:
                    key_sum_code_vector = tl.load(
                        quantized_page_sum_k + page_row * HEAD_DIM + head_offset,
                        mask=page_valid_scalar,
                        other=0,
                    ).to(tl.float32)
                    value_sum_code_vector = tl.load(
                        quantized_page_sum_v + page_row * VALUE_DIM + value_offset,
                        mask=page_valid_scalar,
                        other=0,
                    ).to(tl.float32)
                    key_sum_scale_vector = tl.load(
                        page_sum_k_scales
                        + page_row * (HEAD_DIM // QUANT_GROUP_SIZE)
                        + head_offset // QUANT_GROUP_SIZE,
                        mask=page_valid_scalar,
                        other=0.0,
                    ).to(tl.float32)
                    value_sum_scale_vector = tl.load(
                        page_sum_v_scales
                        + page_row * (VALUE_DIM // QUANT_GROUP_SIZE)
                        + value_offset // QUANT_GROUP_SIZE,
                        mask=page_valid_scalar,
                        other=0.0,
                    ).to(tl.float32)
                    key_sum_vector = key_sum_code_vector * key_sum_scale_vector
                    value_sum_vector = value_sum_code_vector * value_sum_scale_vector
                else:
                    key_sum_vector = tl.load(
                        page_sum_k + page_row * HEAD_DIM + head_offset,
                        mask=page_valid_scalar,
                        other=0.0,
                    ).to(tl.float32)
                    value_sum_vector = tl.load(
                        page_sum_v + page_row * VALUE_DIM + value_offset,
                        mask=page_valid_scalar,
                        other=0.0,
                    ).to(tl.float32)
                k_block = (
                    key_code.to(tl.float32) * key_scale_vector[:, None]
                    + key_sum_vector[:, None] / page_count_scalar
                ).to(tl.bfloat16)
                v_block = (
                    value_code.to(tl.float32) * value_scale_vector[None, :]
                    + value_sum_vector[None, :] / page_count_scalar
                ).to(tl.bfloat16)
            else:
                token_group = within_page // QUANT_TOKEN_GROUP_SIZE
                key_scale_row = (
                    (kv_row * PAGE_CAPACITY + page_id)
                    * (PAGE_SIZE // QUANT_TOKEN_GROUP_SIZE)
                    + token_group
                ) * (HEAD_DIM // QUANT_GROUP_SIZE)
                value_scale_row = (
                    (kv_row * PAGE_CAPACITY + page_id)
                    * (PAGE_SIZE // QUANT_TOKEN_GROUP_SIZE)
                    + token_group
                ) * (VALUE_DIM // QUANT_GROUP_SIZE)
                key_scale = tl.load(
                    page_k_scales
                    + key_scale_row[None, :]
                    + head_offset[:, None] // QUANT_GROUP_SIZE,
                    mask=valid_key[None, :],
                    other=0.0,
                ).to(tl.float32)
                value_scale = tl.load(
                    page_v_scales
                    + value_scale_row[:, None]
                    + value_offset[None, :] // QUANT_GROUP_SIZE,
                    mask=valid_key[:, None],
                    other=0.0,
                ).to(tl.float32)
                page_count = tl.load(
                    page_counts + kv_row * PAGE_CAPACITY + page_id,
                    mask=valid_key,
                    other=1,
                ).to(tl.float32)
                if QUANTIZED_SUMMARIES:
                    key_sum_code = tl.load(
                        quantized_page_sum_k
                        + (kv_row * PAGE_CAPACITY + page_id)[None, :] * HEAD_DIM
                        + head_offset[:, None],
                        mask=valid_key[None, :],
                        other=0,
                    ).to(tl.float32)
                    value_sum_code = tl.load(
                        quantized_page_sum_v
                        + (kv_row * PAGE_CAPACITY + page_id)[:, None] * VALUE_DIM
                        + value_offset[None, :],
                        mask=valid_key[:, None],
                        other=0,
                    ).to(tl.float32)
                    key_sum_scale = tl.load(
                        page_sum_k_scales
                        + (kv_row * PAGE_CAPACITY + page_id)[None, :]
                        * (HEAD_DIM // QUANT_GROUP_SIZE)
                        + head_offset[:, None] // QUANT_GROUP_SIZE,
                        mask=valid_key[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    value_sum_scale = tl.load(
                        page_sum_v_scales
                        + (kv_row * PAGE_CAPACITY + page_id)[:, None]
                        * (VALUE_DIM // QUANT_GROUP_SIZE)
                        + value_offset[None, :] // QUANT_GROUP_SIZE,
                        mask=valid_key[:, None],
                        other=0.0,
                    ).to(tl.float32)
                    key_sum = key_sum_code * key_sum_scale
                    value_sum = value_sum_code * value_sum_scale
                else:
                    key_sum = tl.load(
                        page_sum_k
                        + (kv_row * PAGE_CAPACITY + page_id)[None, :] * HEAD_DIM
                        + head_offset[:, None],
                        mask=valid_key[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    value_sum = tl.load(
                        page_sum_v
                        + (kv_row * PAGE_CAPACITY + page_id)[:, None] * VALUE_DIM
                        + value_offset[None, :],
                        mask=valid_key[:, None],
                        other=0.0,
                    ).to(tl.float32)
                k_block = (
                    key_code.to(tl.float32) * key_scale + key_sum / page_count[None, :]
                ).to(tl.bfloat16)
                v_block = (
                    value_code.to(tl.float32) * value_scale
                    + value_sum / page_count[:, None]
                ).to(tl.bfloat16)
        else:
            k_block = tl.load(
                page_k + storage_token[None, :] * HEAD_DIM + head_offset[:, None],
                mask=valid_key[None, :],
                other=0.0,
            )
            v_block = tl.load(
                page_v + storage_token[:, None] * VALUE_DIM + value_offset[None, :],
                mask=valid_key[:, None],
                other=0.0,
            )

        if INT8_MMA:
            scale_token = storage_token if INDEXED else physical_token
            key_scale = tl.load(
                page_k_scales + scale_token,
                mask=valid_key,
                other=1.0,
            ).to(tl.float32)
            scores = tl.dot(q_mma, k_block, out_dtype=tl.int32).to(tl.float32)
            scores *= SCALE_LOG2 * q_scale[:, None] * key_scale[None, :]
        else:
            scores = SCALE_LOG2 * tl.dot(q_block, k_block, out_dtype=tl.float32)
        scores = tl.where(
            valid_query[:, None] & valid_key[None, :],
            scores,
            -float("inf"),
        )
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum[:, None])
        probabilities = tl.where(
            valid_query[:, None] & valid_key[None, :],
            probabilities,
            0.0,
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        accumulator_t = tl.trans(accumulator)
        if INT8_MMA:
            value_scale = tl.load(
                page_v_scales + scale_token,
                mask=valid_key,
                other=0.0,
            ).to(tl.float32)
            if INT8_PV_MMA:
                scaled_probabilities = probabilities * value_scale[None, :]
                probability_scale = tl.maximum(
                    tl.max(tl.abs(scaled_probabilities), axis=1) / 127.0,
                    1.1754943508222875e-38,
                )
                probability_code = tl.maximum(
                    tl.minimum(
                        tl.floor(
                            scaled_probabilities / probability_scale[:, None] + 0.5
                        ),
                        127.0,
                    ),
                    -127.0,
                ).to(tl.int8)
                value_update = tl.dot(
                    tl.trans(v_block),
                    tl.trans(probability_code),
                    out_dtype=tl.int32,
                ).to(tl.float32)
                accumulator_t += value_update * probability_scale[None, :]
            else:
                # V_i = code_i * scale_i.  Fold the per-token scale into P
                # instead of multiplying every value channel by it.  This is
                # algebraically identical before BF16 rounding and changes the
                # scale work from BLOCK_N * VALUE_DIM to BLOCK_M * BLOCK_N.
                # It also leaves the code matrix directly consumable by the
                # BF16 MMA used for the short-context PV path.
                scaled_probabilities = (probabilities * value_scale[None, :]).to(
                    tl.bfloat16
                )
                accumulator_t += tl.dot(
                    tl.trans(v_block.to(tl.bfloat16)),
                    tl.trans(scaled_probabilities),
                    out_dtype=tl.float32,
                )
        else:
            accumulator_t += tl.dot(
                tl.trans(v_block),
                tl.trans(probabilities.to(v_block.dtype)),
                out_dtype=tl.float32,
            )
        accumulator = tl.trans(accumulator_t)
        maximum = new_maximum

    has_mass = denominator > 0.0
    normalized = tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0)
    natural_lse = tl.where(
        has_mass,
        (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
        -float("inf"),
    )
    if PARTIAL_OUTPUT:
        partial_row = (local_program * SPLIT_N + split) * BLOCK_M + tl.arange(
            0, BLOCK_M
        ).to(tl.int64)
        tl.store(
            out + partial_row[:, None] * VALUE_DIM + value_offset[None, :],
            normalized,
            mask=valid_query[:, None],
        )
        tl.store(lse + partial_row, natural_lse, mask=valid_query)
    else:
        tl.store(
            out + route_row[:, None] * VALUE_DIM + value_offset[None, :],
            normalized,
            mask=valid_query[:, None],
        )
        tl.store(lse + route_row, natural_lse, mask=valid_query)


@triton.jit(
    do_not_specialize=[
        "LEAF_CAPACITY",
        "LEAF_K_BATCH_STRIDE",
        "LEAF_K_HEAD_STRIDE",
        "LEAF_K_TOKEN_STRIDE",
        "LEAF_V_BATCH_STRIDE",
        "LEAF_V_HEAD_STRIDE",
        "LEAF_V_TOKEN_STRIDE",
        "query_len",
    ],
    do_not_specialize_on_alignment=[
        "LEAF_CAPACITY",
        "LEAF_K_BATCH_STRIDE",
        "LEAF_K_HEAD_STRIDE",
        "LEAF_K_TOKEN_STRIDE",
        "LEAF_V_BATCH_STRIDE",
        "LEAF_V_HEAD_STRIDE",
        "LEAF_V_TOKEN_STRIDE",
        "query_len",
    ],
)
def _query_major_residual_page_attention_kernel(
    q,
    state_k,
    state_v,
    state_counts,
    mla_norm_weight,
    cache_indices,
    page_k,
    page_v,
    page_indices,
    leaf_k,
    leaf_v,
    quantized_leaf_k,
    quantized_leaf_v,
    page_k_scales,
    page_v_scales,
    page_quantized_counts,
    page_sum_k,
    page_sum_v,
    quantized_page_sum_k,
    quantized_page_sum_v,
    page_sum_k_scales,
    page_sum_v_scales,
    page_counts,
    materialized_page_scores,
    slot_pages,
    overflow_page_keys,
    overflow_page_values,
    overflow_used,
    slot_lengths,
    top_slots,
    query_len,
    out,
    lse,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    INLINE_PAGES_PER_SLOT: tl.constexpr,
    HASH_CAPACITY: tl.constexpr,
    HASH_PROBES: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HEAD_BLOCK_DIM: tl.constexpr,
    VALUE_BLOCK_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROUTE_COUNT: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    PAGE_BLOCK_N: tl.constexpr,
    LEAF_K_BATCH_STRIDE,
    LEAF_K_HEAD_STRIDE,
    LEAF_K_TOKEN_STRIDE,
    LEAF_V_BATCH_STRIDE,
    LEAF_V_HEAD_STRIDE,
    LEAF_V_TOKEN_STRIDE,
    LEAF_CAPACITY,
    QUANT_GROUP_SIZE: tl.constexpr,
    QUANT_TOKEN_GROUP_SIZE: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    QUANTIZED_SUMMARIES: tl.constexpr,
    MATERIALIZED_PAGE_SCORES: tl.constexpr,
    INDEXED: tl.constexpr,
    ROUTE_PARALLEL: tl.constexpr,
    MLA_LATENT_DIM: tl.constexpr,
    MLA_NORM_EPS: tl.constexpr,
):
    """Open one page per routed slot and summarize its disjoint residual."""
    query_row = tl.program_id(0).to(tl.int64)
    active_route = tl.program_id(1).to(tl.int64)
    batch_head = query_row // query_len
    batch = batch_head // QUERY_HEADS
    cache_batch = tl.load(cache_indices + batch).to(tl.int64)
    query_head = batch_head - batch * QUERY_HEADS
    kv_head = query_head // KV_GROUP_SIZE
    kv_row = cache_batch * KV_HEADS + kv_head

    head_offset = tl.arange(0, HEAD_BLOCK_DIM)
    value_offset = tl.arange(0, VALUE_BLOCK_DIM)
    page_offset = tl.arange(0, PAGE_BLOCK_N)
    token_offset = tl.arange(0, PAGE_SIZE)
    query = tl.load(
        q + query_row * HEAD_DIM + head_offset,
        mask=head_offset < HEAD_DIM,
        other=0.0,
    )
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.zeros((), tl.float32)
    accumulator = tl.zeros((VALUE_BLOCK_DIM,), tl.float32)

    route_begin = active_route if ROUTE_PARALLEL else 0
    route_end = active_route + 1 if ROUTE_PARALLEL else ROUTE_COUNT
    for route in tl.range(route_begin, route_end, num_stages=1):
        routed_slot = tl.load(top_slots + query_row * ROUTE_COUNT + route).to(tl.int64)
        valid_slot = (routed_slot >= 0) & (routed_slot < STATE_CAPACITY)
        slot = tl.where(valid_slot, routed_slot, 0)
        key_count = tl.load(
            slot_lengths + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0,
        ).to(tl.int32)
        slot_page_count = (key_count + PAGE_SIZE - 1) // PAGE_SIZE
        if HASH_PROBES == 0:
            page_table = (
                slot_pages + (kv_row * STATE_CAPACITY + slot) * INLINE_PAGES_PER_SLOT
            )
        selected_score = tl.full((), -float("inf"), tl.float32)
        selected_page = tl.full((), 0, tl.int64)
        # A singleton centroid has only one possible detail page, so avoid a
        # redundant page-summary score.
        single_page = valid_slot & (slot_page_count == 1)
        if HASH_PROBES == 0:
            first_page = tl.load(
                page_table,
                mask=single_page,
                other=0,
            ).to(tl.int64)
        else:
            first_page = _lookup_page_id(
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                kv_row,
                slot,
                0,
                single_page,
                STATE_CAPACITY,
                INLINE_PAGES_PER_SLOT,
                PAGE_CAPACITY,
                HASH_CAPACITY,
                HASH_PROBES,
            ).to(tl.int64)
        single_page &= (first_page >= 0) & (first_page < PAGE_CAPACITY)
        selected_score = tl.where(single_page, float("inf"), selected_score)
        selected_page = tl.where(single_page, first_page, selected_page)
        scan_page_count = tl.where(single_page, 0, slot_page_count)

        for page_begin in tl.range(0, scan_page_count, PAGE_BLOCK_N, num_stages=1):
            page_ordinal = page_begin + page_offset
            valid_page = page_ordinal < scan_page_count
            if HASH_PROBES == 0:
                page_id = tl.load(
                    page_table + page_ordinal, mask=valid_page, other=0
                ).to(tl.int64)
            else:
                page_id = _lookup_page_id(
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    kv_row,
                    slot,
                    page_ordinal,
                    valid_page,
                    STATE_CAPACITY,
                    INLINE_PAGES_PER_SLOT,
                    PAGE_CAPACITY,
                    HASH_CAPACITY,
                    HASH_PROBES,
                ).to(tl.int64)
            valid_page &= (page_id >= 0) & (page_id < PAGE_CAPACITY)
            page_id = tl.where(valid_page, page_id, 0)
            if MATERIALIZED_PAGE_SCORES:
                page_scores = tl.load(
                    materialized_page_scores + query_row * PAGE_CAPACITY + page_id,
                    mask=valid_page,
                    other=-float("inf"),
                ).to(tl.float32)
            else:
                count = tl.load(
                    page_counts + kv_row * PAGE_CAPACITY + page_id,
                    mask=valid_page,
                    other=1,
                ).to(tl.float32)
                if QUANTIZED_SUMMARIES:
                    key_sum_codes = tl.load(
                        quantized_page_sum_k
                        + (kv_row * PAGE_CAPACITY + page_id[:, None]) * HEAD_DIM
                        + head_offset[None, :],
                        mask=valid_page[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0,
                    ).to(tl.float32)
                    key_sum_scales = tl.load(
                        page_sum_k_scales
                        + (kv_row * PAGE_CAPACITY + page_id[:, None])
                        * (HEAD_DIM // QUANT_GROUP_SIZE)
                        + head_offset[None, :] // QUANT_GROUP_SIZE,
                        mask=valid_page[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0.0,
                    ).to(tl.float32)
                    key_sums = key_sum_codes * key_sum_scales
                else:
                    key_sums = tl.load(
                        page_sum_k
                        + (kv_row * PAGE_CAPACITY + page_id[:, None]) * HEAD_DIM
                        + head_offset[None, :],
                        mask=valid_page[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0.0,
                    )
                page_keys = key_sums.to(tl.float32) / count[:, None]
                if MLA_LATENT_DIM > 0:
                    # Reproduce DeepSeek's latent RMSNorm ordering exactly:
                    # average raw compressed latents, round the unit-RMS vector
                    # to BF16, then apply the learned gain.  The appended RoPE
                    # channels remain an ordinary arithmetic mean.
                    page_keys = page_keys.to(tl.bfloat16)
                    latent_mask = head_offset < MLA_LATENT_DIM
                    latent_values = tl.where(
                        latent_mask[None, :], page_keys.to(tl.float32), 0.0
                    )
                    inverse_rms = tl.rsqrt(
                        tl.sum(latent_values * latent_values, axis=1) / MLA_LATENT_DIM
                        + MLA_NORM_EPS
                    )
                    unit_latent = (page_keys.to(tl.float32) * inverse_rms[:, None]).to(
                        tl.bfloat16
                    )
                    norm_gain = tl.load(
                        mla_norm_weight + head_offset,
                        mask=latent_mask,
                        other=1.0,
                    ).to(tl.bfloat16)
                    normalized_latent = (unit_latent * norm_gain[None, :]).to(
                        tl.bfloat16
                    )
                    page_keys = tl.where(
                        latent_mask[None, :], normalized_latent, page_keys
                    ).to(tl.float32)
                page_scores = SCALE_LOG2 * tl.sum(
                    page_keys * query[None, :].to(tl.float32),
                    axis=1,
                ) + tl.log2(count)
            page_scores = tl.where(valid_page, page_scores, -float("inf"))
            block_score = tl.max(page_scores, axis=0)
            block_page = tl.max(
                tl.where(page_scores == block_score, page_id, -1), axis=0
            ).to(tl.int64)
            better = block_score > selected_score
            selected_score = tl.where(better, block_score, selected_score)
            selected_page = tl.where(better, block_page, selected_page)

        selected_valid = selected_score > -float("inf")
        selected_count = tl.load(
            page_counts + kv_row * PAGE_CAPACITY + selected_page,
            mask=selected_valid,
            other=0,
        ).to(tl.float32)
        state_count = tl.load(
            state_counts + kv_row * STATE_CAPACITY + slot,
            mask=valid_slot,
            other=0,
        ).to(tl.float32)
        residual_count = state_count - selected_count
        if QUANTIZED_SUMMARIES:
            selected_key_sum = tl.load(
                quantized_page_sum_k
                + (kv_row * PAGE_CAPACITY + selected_page) * HEAD_DIM
                + head_offset,
                mask=selected_valid & (head_offset < HEAD_DIM),
                other=0,
            ).to(tl.float32) * tl.load(
                page_sum_k_scales
                + (kv_row * PAGE_CAPACITY + selected_page)
                * (HEAD_DIM // QUANT_GROUP_SIZE)
                + head_offset // QUANT_GROUP_SIZE,
                mask=selected_valid & (head_offset < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            selected_value_sum = tl.load(
                quantized_page_sum_v
                + (kv_row * PAGE_CAPACITY + selected_page) * VALUE_DIM
                + value_offset,
                mask=selected_valid & (value_offset < VALUE_DIM),
                other=0,
            ).to(tl.float32) * tl.load(
                page_sum_v_scales
                + (kv_row * PAGE_CAPACITY + selected_page)
                * (VALUE_DIM // QUANT_GROUP_SIZE)
                + value_offset // QUANT_GROUP_SIZE,
                mask=selected_valid & (value_offset < VALUE_DIM),
                other=0.0,
            ).to(tl.float32)
        else:
            selected_key_sum = tl.load(
                page_sum_k
                + (kv_row * PAGE_CAPACITY + selected_page) * HEAD_DIM
                + head_offset,
                mask=selected_valid & (head_offset < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            selected_value_sum = tl.load(
                page_sum_v
                + (kv_row * PAGE_CAPACITY + selected_page) * VALUE_DIM
                + value_offset,
                mask=selected_valid & (value_offset < VALUE_DIM),
                other=0.0,
            ).to(tl.float32)
        state_key_sum = tl.load(
            state_k + (kv_row * STATE_CAPACITY + slot) * HEAD_DIM + head_offset,
            mask=valid_slot & (head_offset < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        state_value_sum = tl.load(
            state_v + (kv_row * STATE_CAPACITY + slot) * VALUE_DIM + value_offset,
            mask=valid_slot & (value_offset < VALUE_DIM),
            other=0.0,
        ).to(tl.float32)

        if residual_count > 0.0:
            residual_key = (state_key_sum - selected_key_sum) / residual_count
            if MLA_LATENT_DIM > 0:
                residual_key = residual_key.to(tl.bfloat16)
                latent_mask = head_offset < MLA_LATENT_DIM
                latent_values = tl.where(latent_mask, residual_key.to(tl.float32), 0.0)
                inverse_rms = tl.rsqrt(
                    tl.sum(latent_values * latent_values, axis=0) / MLA_LATENT_DIM
                    + MLA_NORM_EPS
                )
                unit_latent = (residual_key.to(tl.float32) * inverse_rms).to(
                    tl.bfloat16
                )
                norm_gain = tl.load(
                    mla_norm_weight + head_offset,
                    mask=latent_mask,
                    other=1.0,
                ).to(tl.bfloat16)
                normalized_latent = (unit_latent * norm_gain).to(tl.bfloat16)
                residual_key = tl.where(
                    latent_mask, normalized_latent, residual_key
                ).to(tl.float32)
            residual_value = (state_value_sum - selected_value_sum) / residual_count
            residual_score = SCALE_LOG2 * tl.sum(
                residual_key * query.to(tl.float32), axis=0
            ) + tl.log2(residual_count)
            new_maximum = tl.maximum(maximum, residual_score)
            correction = tl.math.exp2(maximum - new_maximum)
            probability = tl.math.exp2(residual_score - new_maximum)
            denominator = denominator * correction + probability
            accumulator = accumulator * correction + probability * residual_value
            maximum = new_maximum

        valid_token = selected_valid & (token_offset < selected_count)
        physical_token = (
            kv_row * PAGE_CAPACITY + selected_page
        ) * PAGE_SIZE + token_offset
        if INDEXED:
            leaf_index = tl.load(
                page_indices + physical_token,
                mask=valid_token,
                other=0,
            ).to(tl.int64)
            valid_token &= (leaf_index >= 0) & (leaf_index < LEAF_CAPACITY)
            leaf_index = tl.where(valid_token, leaf_index, 0)
            if QUANT_BITS:
                # Final conversion and every quantized append publish complete
                # changed pages before attention can observe them.  The old
                # mixed BF16 fallback kept a full-width conditional load and
                # select live in this already register-heavy kernel even
                # though finalized vLLM caches never exercised it.
                use_quantized = valid_token
                if QUANT_BITS == 4:
                    packed_head_offset = head_offset // 2
                    packed_value_offset = value_offset // 2
                    packed_keys = tl.load(
                        quantized_leaf_k
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None])
                        * (HEAD_DIM // 2)
                        + packed_head_offset[None, :],
                        mask=use_quantized[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0,
                    ).to(tl.int32)
                    packed_values = tl.load(
                        quantized_leaf_v
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None])
                        * (VALUE_DIM // 2)
                        + packed_value_offset[None, :],
                        mask=use_quantized[:, None]
                        & (value_offset[None, :] < VALUE_DIM),
                        other=0,
                    ).to(tl.int32)
                    key_shift = (head_offset & 1) * 4
                    value_shift = (value_offset & 1) * 4
                    key_code = ((packed_keys >> key_shift[None, :]) & 15) - 8
                    value_code = ((packed_values >> value_shift[None, :]) & 15) - 8
                else:
                    key_code = tl.load(
                        quantized_leaf_k
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * HEAD_DIM
                        + head_offset[None, :],
                        mask=use_quantized[:, None] & (head_offset[None, :] < HEAD_DIM),
                        other=0,
                    ).to(tl.int32)
                    value_code = tl.load(
                        quantized_leaf_v
                        + (kv_row * LEAF_CAPACITY + leaf_index[:, None]) * VALUE_DIM
                        + value_offset[None, :],
                        mask=use_quantized[:, None]
                        & (value_offset[None, :] < VALUE_DIM),
                        other=0,
                    ).to(tl.int32)
                if QUANT_TOKEN_GROUP_SIZE == PAGE_SIZE:
                    # Keep the original broadcast load for the legacy layout.
                    # Besides avoiding redundant scale traffic, this preserves
                    # the exact reduction numerics of the established kernel.
                    key_scale = tl.load(
                        page_k_scales
                        + (kv_row * PAGE_CAPACITY + selected_page)
                        * (HEAD_DIM // QUANT_GROUP_SIZE)
                        + head_offset // QUANT_GROUP_SIZE,
                        mask=head_offset < HEAD_DIM,
                        other=0.0,
                    ).to(tl.float32)
                    value_scale = tl.load(
                        page_v_scales
                        + (kv_row * PAGE_CAPACITY + selected_page)
                        * (VALUE_DIM // QUANT_GROUP_SIZE)
                        + value_offset // QUANT_GROUP_SIZE,
                        mask=value_offset < VALUE_DIM,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    key_scale_row = (
                        (kv_row * PAGE_CAPACITY + selected_page)
                        * (PAGE_SIZE // QUANT_TOKEN_GROUP_SIZE)
                        + token_offset // QUANT_TOKEN_GROUP_SIZE
                    ) * (HEAD_DIM // QUANT_GROUP_SIZE)
                    value_scale_row = (
                        (kv_row * PAGE_CAPACITY + selected_page)
                        * (PAGE_SIZE // QUANT_TOKEN_GROUP_SIZE)
                        + token_offset // QUANT_TOKEN_GROUP_SIZE
                    ) * (VALUE_DIM // QUANT_GROUP_SIZE)
                    if QUANT_GROUP_SIZE == HEAD_DIM:
                        # Token-wise, whole-vector INT4 has one scale per key.
                        # Load it once and broadcast in registers rather than
                        # issuing HEAD_DIM identical scale loads per token.
                        key_scale = tl.load(
                            page_k_scales + key_scale_row,
                            mask=valid_token,
                            other=0.0,
                        ).to(tl.float32)[:, None]
                    else:
                        key_scale = tl.load(
                            page_k_scales
                            + key_scale_row[:, None]
                            + head_offset[None, :] // QUANT_GROUP_SIZE,
                            mask=valid_token[:, None]
                            & (head_offset[None, :] < HEAD_DIM),
                            other=0.0,
                        ).to(tl.float32)
                    if QUANT_GROUP_SIZE == VALUE_DIM:
                        value_scale = tl.load(
                            page_v_scales + value_scale_row,
                            mask=valid_token,
                            other=0.0,
                        ).to(tl.float32)[:, None]
                    else:
                        value_scale = tl.load(
                            page_v_scales
                            + value_scale_row[:, None]
                            + value_offset[None, :] // QUANT_GROUP_SIZE,
                            mask=valid_token[:, None]
                            & (value_offset[None, :] < VALUE_DIM),
                            other=0.0,
                        ).to(tl.float32)
                key_residual = key_code.to(tl.float32) * key_scale
                value_residual = value_code.to(tl.float32) * value_scale
                inverse_selected_count = 1.0 / tl.maximum(selected_count, 1.0)
                key_anchor = selected_key_sum * inverse_selected_count
                value_anchor = selected_value_sum * inverse_selected_count
                # The page mean is shared by all sixteen leaves. Keep it out
                # of the token-by-channel residual matrices: adding it after
                # the QK/PV reductions is algebraically identical, avoids two
                # page-wide broadcasts, and materially lowers register
                # pressure in the INT4 specialization.
                quantized_exact_scores = SCALE_LOG2 * (
                    tl.sum(key_residual * query[None, :].to(tl.float32), axis=1)
                    + tl.sum(key_anchor * query.to(tl.float32), axis=0)
                )
            else:
                keys = tl.load(
                    leaf_k
                    + cache_batch * LEAF_K_BATCH_STRIDE
                    + kv_head * LEAF_K_HEAD_STRIDE
                    + leaf_index[:, None] * LEAF_K_TOKEN_STRIDE
                    + head_offset[None, :],
                    mask=valid_token[:, None] & (head_offset[None, :] < HEAD_DIM),
                    other=0.0,
                )
                values = tl.load(
                    leaf_v
                    + cache_batch * LEAF_V_BATCH_STRIDE
                    + kv_head * LEAF_V_HEAD_STRIDE
                    + leaf_index[:, None] * LEAF_V_TOKEN_STRIDE
                    + value_offset[None, :],
                    mask=valid_token[:, None] & (value_offset[None, :] < VALUE_DIM),
                    other=0.0,
                )
        else:
            keys = tl.load(
                page_k + physical_token[:, None] * HEAD_DIM + head_offset[None, :],
                mask=valid_token[:, None] & (head_offset[None, :] < HEAD_DIM),
                other=0.0,
            )
            values = tl.load(
                page_v + physical_token[:, None] * VALUE_DIM + value_offset[None, :],
                mask=valid_token[:, None] & (value_offset[None, :] < VALUE_DIM),
                other=0.0,
            )
        if INDEXED and QUANT_BITS:
            exact_scores = quantized_exact_scores
        else:
            exact_scores = SCALE_LOG2 * tl.sum(
                keys.to(tl.float32) * query[None, :].to(tl.float32), axis=1
            )
        exact_scores = tl.where(valid_token, exact_scores, -float("inf"))
        block_maximum = tl.max(exact_scores, axis=0)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.where(
            selected_valid,
            tl.math.exp2(maximum - new_maximum),
            1.0,
        )
        probabilities = tl.math.exp2(exact_scores - new_maximum)
        probabilities = tl.where(valid_token, probabilities, 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        if INDEXED and QUANT_BITS:
            probability_sum = tl.sum(probabilities, axis=0)
            value_update = (
                tl.sum(probabilities[:, None] * value_residual, axis=0)
                + probability_sum * value_anchor
            )
        else:
            value_update = tl.sum(probabilities[:, None] * values, axis=0)
        accumulator = accumulator * correction + value_update
        maximum = tl.where(selected_valid, new_maximum, maximum)

    output_row = query_row * ROUTE_COUNT + active_route if ROUTE_PARALLEL else query_row
    has_mass = denominator > 0.0
    tl.store(
        out + output_row * VALUE_DIM + value_offset,
        tl.where(has_mass, accumulator / denominator, 0.0),
        mask=value_offset < VALUE_DIM,
    )
    tl.store(
        lse + output_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )


def query_major_residual_page_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    state_counts: torch.Tensor,
    page_k: torch.Tensor | None,
    page_v: torch.Tensor | None,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    cache_indices: torch.Tensor | None = None,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    page_block_n: int = 16,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
    page_indices: torch.Tensor | None = None,
    leaf_k: torch.Tensor | None = None,
    leaf_v: torch.Tensor | None = None,
    quantized_leaf_k: torch.Tensor | None = None,
    quantized_leaf_v: torch.Tensor | None = None,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    page_quantized_counts: torch.Tensor | None = None,
    quantized_page_sum_k: torch.Tensor | None = None,
    quantized_page_sum_v: torch.Tensor | None = None,
    page_sum_k_scales: torch.Tensor | None = None,
    page_sum_v_scales: torch.Tensor | None = None,
    quant_group_size: int = 32,
    quant_token_group_size: int = 16,
    quant_bits: int = 4,
    output_buffer: torch.Tensor | None = None,
    lse_buffer: torch.Tensor | None = None,
    route_parallel: bool = False,
    mla_norm_weight: torch.Tensor | None = None,
    mla_norm_epsilon: float = 0.0,
    materialized_page_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact top page plus a count-corrected residual for each routed slot."""
    indexed = page_indices is not None
    if indexed != (leaf_k is not None and leaf_v is not None):
        raise ValueError("indexed pages require indices and flat K/V together")
    if not indexed and (page_k is None or page_v is None):
        raise ValueError("physical residual-page attention requires page K/V")
    storage_k = leaf_k if indexed else page_k
    storage_v = leaf_v if indexed else page_v
    if storage_k is None or storage_v is None:
        raise AssertionError("residual-page K/V storage is missing")
    quantization_tensors = (
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
    )
    quantized = any(tensor is not None for tensor in quantization_tensors)
    if quantized and not all(
        isinstance(tensor, torch.Tensor) for tensor in quantization_tensors
    ):
        raise ValueError("indexed quantized tensors must be supplied together")
    if quantized and not indexed:
        raise ValueError("quantized residual pages require indexed virtual storage")
    if quantized and quant_bits not in (4, 8):
        raise ValueError("quantized residual pages support 4 or 8 bits")
    summary_quantization_tensors = (
        quantized_page_sum_k,
        quantized_page_sum_v,
        page_sum_k_scales,
        page_sum_v_scales,
    )
    quantized_summaries = any(
        tensor is not None for tensor in summary_quantization_tensors
    )
    if quantized_summaries and not all(
        isinstance(tensor, torch.Tensor) for tensor in summary_quantization_tensors
    ):
        raise ValueError("INT8 page-summary tensors must be supplied together")
    mla_latent_dim = 0
    if mla_norm_weight is not None:
        if not mla_norm_weight.is_cuda:
            raise ValueError("MLA RMSNorm gain must be a CUDA tensor")
        if quantized or quantized_summaries:
            raise ValueError("raw MLA page summaries do not support quantization")
        mla_latent_dim = int(mla_norm_weight.numel())
    if materialized_page_scores is not None:
        if quantized_summaries:
            raise ValueError(
                "materialized page scores currently require BF16 page summaries"
            )
        if mla_latent_dim:
            raise ValueError("materialized page scores do not yet support MLA")
    page_shape = page_indices.shape if indexed else page_k.shape
    tensors = (
        q,
        state_k,
        state_v,
        state_counts,
        storage_k,
        storage_v,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        slot_lengths,
        top_slots,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("residual-page attention requires CUDA tensors")
    batch, query_heads, query_len, head_dim = q.shape
    cache_batch_size = int(storage_k.size(0))
    if cache_indices is None:
        if cache_batch_size != batch:
            raise ValueError(
                "cache indices are required when cache and query batches differ"
            )
        cache_indices = torch.arange(batch, dtype=torch.long, device=q.device)
    elif tuple(cache_indices.shape) != (batch,):
        raise ValueError("cache indices must contain one stable slot per query row")
    kv_heads = int(storage_k.size(1))
    value_dim = int(storage_v.size(-1))
    if mla_latent_dim and not 0 < mla_latent_dim < head_dim:
        raise ValueError("MLA RMSNorm gain does not match the key geometry")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if quantized and (head_dim % quant_group_size or value_dim % quant_group_size):
        raise ValueError("quantization group size must divide K/V dimensions")
    if quantized and int(page_indices.size(3)) % quant_token_group_size:
        raise ValueError("token quantization group size must divide the page size")
    if quantized:
        expected_k_width = head_dim // 2 if quant_bits == 4 else head_dim
        expected_v_width = value_dim // 2 if quant_bits == 4 else value_dim
        expected_dtype = torch.uint8 if quant_bits == 4 else torch.int8
        if quantized_leaf_k.dtype != expected_dtype or (
            quantized_leaf_v.dtype != expected_dtype
        ):
            raise TypeError(
                f"{quant_bits}-bit leaf storage requires {expected_dtype} codes"
            )
        if int(quantized_leaf_k.size(-1)) != expected_k_width or (
            int(quantized_leaf_v.size(-1)) != expected_v_width
        ):
            raise ValueError("quantized leaf widths do not match K/V dimensions")
        token_group_count = int(page_indices.size(3)) // quant_token_group_size
        expected_k_scales = (
            cache_batch_size,
            kv_heads,
            int(page_indices.size(2)),
            token_group_count * (head_dim // quant_group_size),
        )
        expected_v_scales = expected_k_scales[:-1] + (
            token_group_count * (value_dim // quant_group_size),
        )
        if tuple(page_k_scales.shape) != expected_k_scales:
            raise ValueError("quantized leaf K scales do not match the cache")
        if tuple(page_v_scales.shape) != expected_v_scales:
            raise ValueError("quantized leaf V scales do not match the cache")
    if int(page_shape[3]) not in (4, 16):
        raise ValueError("residual-page attention requires 4- or 16-token pages")
    expected_k_summary = (
        cache_batch_size,
        kv_heads,
        int(page_shape[2]),
        head_dim,
    )
    expected_v_summary = (
        cache_batch_size,
        kv_heads,
        int(page_shape[2]),
        value_dim,
    )
    if quantized_summaries:
        if tuple(quantized_page_sum_k.shape) != expected_k_summary:
            raise ValueError("quantized page K summaries do not match the cache")
        if tuple(quantized_page_sum_v.shape) != expected_v_summary:
            raise ValueError("quantized page V summaries do not match the cache")
        expected_k_scales = expected_k_summary[:-1] + (head_dim // quant_group_size,)
        expected_v_scales = expected_v_summary[:-1] + (value_dim // quant_group_size,)
        if tuple(page_sum_k_scales.shape) != expected_k_scales:
            raise ValueError("page K-summary scales do not match the cache")
        if tuple(page_sum_v_scales.shape) != expected_v_scales:
            raise ValueError("page V-summary scales do not match the cache")
    else:
        if tuple(page_sum_k.shape) != expected_k_summary:
            raise ValueError("page K summaries do not match the page cache")
        if tuple(page_sum_v.shape) != expected_v_summary:
            raise ValueError("page V summaries do not match the page cache")
    rows = batch * query_heads * query_len
    if materialized_page_scores is not None:
        expected_scores = (batch, query_heads, query_len, int(page_shape[2]))
        if tuple(materialized_page_scores.shape) != expected_scores:
            raise ValueError("materialized page scores have the wrong shape")
        if materialized_page_scores.dtype != torch.float32:
            raise TypeError("materialized page scores must use FP32")
    route_count = int(top_slots.size(-1))
    if route_parallel and query_len != 1:
        raise ValueError("route-parallel residual pages require decode queries")
    output_shape = (
        (batch, query_heads, route_count, value_dim)
        if route_parallel
        else (batch, query_heads, query_len, value_dim)
    )
    lse_shape = output_shape[:-1]
    if output_buffer is None:
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
    else:
        if tuple(output_buffer.shape) != output_shape:
            raise ValueError("residual-page output buffer has the wrong shape")
        output = output_buffer
    if lse_buffer is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    else:
        if tuple(lse_buffer.shape) != lse_shape:
            raise ValueError("residual-page LSE buffer has the wrong shape")
        lse = lse_buffer
    begin = None
    if timing_events is not None:
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
    grid = (rows, route_count) if route_parallel else (rows, 1)
    _query_major_residual_page_attention_kernel[grid](
        q.contiguous(),
        state_k,
        state_v,
        state_counts,
        mla_norm_weight if mla_norm_weight is not None else page_counts,
        cache_indices.contiguous(),
        storage_k,
        storage_v,
        page_indices if indexed else slot_pages,
        storage_k,
        storage_v,
        quantized_leaf_k if quantized else storage_k,
        quantized_leaf_v if quantized else storage_v,
        page_k_scales if quantized else page_counts,
        page_v_scales if quantized else page_counts,
        page_quantized_counts if quantized else page_counts,
        page_sum_k,
        page_sum_v,
        quantized_page_sum_k if quantized_summaries else page_sum_k,
        quantized_page_sum_v if quantized_summaries else page_sum_v,
        page_sum_k_scales if quantized_summaries else page_counts,
        page_sum_v_scales if quantized_summaries else page_counts,
        page_counts,
        materialized_page_scores
        if materialized_page_scores is not None
        else page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots.contiguous(),
        query_len,
        output,
        lse,
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=kv_group_size,
        PAGE_CAPACITY=int(page_shape[2]),
        STATE_CAPACITY=int(slot_pages.size(2)),
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        HEAD_BLOCK_DIM=triton.next_power_of_2(head_dim),
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        PAGE_SIZE=int(page_shape[3]),
        ROUTE_COUNT=int(top_slots.size(-1)),
        SCALE_LOG2=float(scale) * math.log2(math.e),
        PAGE_BLOCK_N=page_block_n,
        LEAF_K_BATCH_STRIDE=int(storage_k.stride(0)) if indexed else 0,
        LEAF_K_HEAD_STRIDE=int(storage_k.stride(1)) if indexed else 0,
        LEAF_K_TOKEN_STRIDE=int(storage_k.stride(2)) if indexed else 0,
        LEAF_V_BATCH_STRIDE=int(storage_v.stride(0)) if indexed else 0,
        LEAF_V_HEAD_STRIDE=int(storage_v.stride(1)) if indexed else 0,
        LEAF_V_TOKEN_STRIDE=int(storage_v.stride(2)) if indexed else 0,
        LEAF_CAPACITY=(
            int(quantized_leaf_k.size(2)) if quantized else int(storage_k.size(2))
        ),
        QUANT_GROUP_SIZE=quant_group_size,
        QUANT_TOKEN_GROUP_SIZE=quant_token_group_size,
        QUANT_BITS=quant_bits if quantized else 0,
        QUANTIZED_SUMMARIES=quantized_summaries,
        MATERIALIZED_PAGE_SCORES=materialized_page_scores is not None,
        INDEXED=indexed,
        ROUTE_PARALLEL=route_parallel,
        MLA_LATENT_DIM=mla_latent_dim,
        MLA_NORM_EPS=float(mla_norm_epsilon),
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    if timing_events is not None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if begin is None:
            raise AssertionError("recursive page timing start is missing")
        timing_events.setdefault("kernel", []).append((begin, end))
        timing_events.setdefault("total", []).append((begin, end))
    return (
        output,
        lse,
    )


def query_major_indexed_residual_page_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    state_counts: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursive page attention over indexed leaves in the original KV cache."""
    return query_major_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        None,
        None,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        page_indices=page_indices,
        leaf_k=leaf_k,
        leaf_v=leaf_v,
        **kwargs,
    )


def paged_leaf_attention(
    q: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    page_indices: torch.Tensor,
    page_k_scales: torch.Tensor | None = None,
    page_v_scales: torch.Tensor | None = None,
    quantized_leaf_k: torch.Tensor | None = None,
    quantized_leaf_v: torch.Tensor | None = None,
    page_sum_k: torch.Tensor | None = None,
    page_sum_v: torch.Tensor | None = None,
    quantized_page_sum_k: torch.Tensor | None = None,
    quantized_page_sum_v: torch.Tensor | None = None,
    page_sum_k_scales: torch.Tensor | None = None,
    page_sum_v_scales: torch.Tensor | None = None,
    page_counts: torch.Tensor | None = None,
    quant_group_size: int = 32,
    quant_token_group_size: int = 16,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_m: int = 16,
    block_n: int = 32,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    reduce_num_warps: int = 1,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to indexed BF16 or residual-INT4 leaves and merge by LSE."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError("paged leaf Triton attention is forward-only")
    batch, query_heads, query_len, head_dim = q.shape
    route_count = int(top_slots.size(-1))
    kv_heads = int(page_k.size(1))
    value_dim = int(page_v.size(-1))
    page_size = int(page_indices.size(3))
    page_capacity = int(page_indices.size(2))
    state_capacity = int(slot_pages.size(2))
    if page_size != 16:
        raise ValueError("the LoD release requires 16-token pages")
    if head_dim != value_dim:
        raise ValueError("paged leaf Triton attention requires equal QK/V dimensions")
    if query_heads != kv_heads * kv_group_size:
        raise ValueError("query/KV head grouping is inconsistent")
    if page_k.dtype == torch.int8 or page_v.dtype == torch.int8:
        raise TypeError("the LoD release does not use flat INT8 leaf storage")
    residual_quantized = isinstance(quantized_leaf_k, torch.Tensor) or isinstance(
        quantized_leaf_v, torch.Tensor
    )
    if residual_quantized:
        required_quantized = (
            quantized_leaf_k,
            quantized_leaf_v,
            page_k_scales,
            page_v_scales,
            page_counts,
        )
        if not all((isinstance(value, torch.Tensor) for value in required_quantized)):
            raise ValueError("residual INT4 leaf attention metadata is incomplete")
        if block_n != 16:
            raise ValueError(
                "residual INT4 expert attention requires page-sized blocks"
            )
        if head_dim % quant_group_size or value_dim % quant_group_size:
            raise ValueError("residual INT4 quantization groups must divide K/V")
        if 16 % quant_token_group_size:
            raise ValueError("residual INT4 token groups must divide the page")
        quantized_summaries = isinstance(
            quantized_page_sum_k, torch.Tensor
        ) or isinstance(quantized_page_sum_v, torch.Tensor)
        if quantized_summaries:
            required_summaries = (
                quantized_page_sum_k,
                quantized_page_sum_v,
                page_sum_k_scales,
                page_sum_v_scales,
            )
        else:
            required_summaries = (page_sum_k, page_sum_v)
        if not all((isinstance(value, torch.Tensor) for value in required_summaries)):
            raise ValueError("residual INT4 page-summary metadata is incomplete")
    else:
        quantized_summaries = False
    if not residual_quantized and (
        page_k_scales is not None or page_v_scales is not None
    ):
        raise ValueError("BF16 leaf attention received INT8 scale tensors")
    if reduce_num_warps not in (1, 2, 4, 8):
        raise ValueError("expert route reduction warps must be one of 1, 2, 4, 8")
    boundaries: list[torch.cuda.Event] = []

    def record_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            boundaries.append(event)

    record_boundary()
    dispatch_boundaries: list[torch.cuda.Event] = boundaries[-1:] if boundaries else []

    def record_dispatch_boundary() -> None:
        if timing_events is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            dispatch_boundaries.append(event)

    with torch.no_grad():
        rows = batch * query_heads * query_len
        query_head = torch.arange(query_heads, device=q.device, dtype=torch.int32)
        kv_head_for_query_head = torch.div(
            query_head, kv_group_size, rounding_mode="floor"
        )
        kv_row_for_head = torch.arange(
            batch, device=q.device, dtype=torch.int32
        ).unsqueeze(1) * kv_heads + kv_head_for_query_head.unsqueeze(0)
        expert_id = (
            kv_row_for_head[:, :, None, None] * state_capacity
            + top_slots.clamp_min(0).to(torch.int32)
        ).reshape(-1)
        record_dispatch_boundary()
        sorted_expert, order = expert_id.sort(stable=False)
        record_dispatch_boundary()
        unique_expert, q_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        expert_kv_row = torch.div(unique_expert, state_capacity, rounding_mode="floor")
        expert_slot = unique_expert % state_capacity
        cu_q = F.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
        expert_index = torch.arange(
            q_lengths.numel(), device=q.device, dtype=torch.int32
        )
        record_dispatch_boundary()
        expert_blocks = torch.div(
            q_lengths + block_m - 1, block_m, rounding_mode="floor"
        )
        cumulative_blocks = F.pad(expert_blocks.cumsum(0), (1, 0)).to(torch.int32)
        total_blocks = int(cumulative_blocks[-1].item())
        block_expert = torch.repeat_interleave(
            expert_index, expert_blocks, output_size=total_blocks
        )
        block_starts = cumulative_blocks[:-1]
        q_lengths = q_lengths.to(torch.int32)
        record_dispatch_boundary()
    record_boundary()
    route_out = torch.empty(
        rows * route_count, value_dim, dtype=q.dtype, device=q.device
    )
    route_lse = torch.empty(rows * route_count, dtype=torch.float32, device=q.device)
    leaf_capacity = (
        int(quantized_leaf_k.size(2)) if residual_quantized else int(page_k.size(2))
    )
    quantized_leaf_k_arg = quantized_leaf_k if residual_quantized else page_k
    quantized_leaf_v_arg = quantized_leaf_v if residual_quantized else page_v
    page_sum_k_arg = page_sum_k if isinstance(page_sum_k, torch.Tensor) else page_k
    page_sum_v_arg = page_sum_v if isinstance(page_sum_v, torch.Tensor) else page_v
    quantized_page_sum_k_arg = quantized_page_sum_k if quantized_summaries else page_k
    quantized_page_sum_v_arg = quantized_page_sum_v if quantized_summaries else page_v
    page_sum_k_scales_arg = page_sum_k_scales if quantized_summaries else page_k
    page_sum_v_scales_arg = page_sum_v_scales if quantized_summaries else page_v
    page_counts_arg = page_counts if residual_quantized else slot_lengths
    record_boundary()
    _paged_leaf_attention_kernel[total_blocks,](
        q,
        q,
        order,
        block_expert,
        block_starts,
        page_k,
        page_v,
        page_indices,
        page_k_scales if residual_quantized else page_k,
        page_v_scales if residual_quantized else page_v,
        quantized_leaf_k_arg,
        quantized_leaf_v_arg,
        page_sum_k_arg,
        page_sum_v_arg,
        quantized_page_sum_k_arg,
        quantized_page_sum_v_arg,
        page_sum_k_scales_arg,
        page_sum_v_scales_arg,
        page_counts_arg,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        q_lengths,
        cu_q,
        expert_kv_row,
        expert_slot,
        route_out,
        route_lse,
        0,
        total_blocks,
        int(q_lengths.numel()),
        PAGE_CAPACITY=page_capacity,
        LEAF_CAPACITY=leaf_capacity,
        STATE_CAPACITY=state_capacity,
        INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
        HASH_CAPACITY=int(overflow_page_values.size(2)),
        HASH_PROBES=hash_probes,
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        PAGE_SIZE=page_size,
        ROUTE_COUNT=route_count,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLIT_N=1,
        PARTIAL_OUTPUT=False,
        INT8_MMA=False,
        INT8_PV_MMA=False,
        QUANT_BITS=4 if residual_quantized else 0,
        QUANT_GROUP_SIZE=quant_group_size if residual_quantized else 1,
        QUANT_TOKEN_GROUP_SIZE=quant_token_group_size if residual_quantized else 1,
        QUANTIZED_SUMMARIES=quantized_summaries,
        INDEXED=True,
        PROGRAMS_POINTER=False,
        SEARCH_BLOCKS=False,
        SEARCH_STEPS=1,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    record_boundary()
    exact_out = torch.empty(rows, value_dim, dtype=q.dtype, device=q.device)
    exact_lse = torch.empty(rows, dtype=torch.float32, device=q.device)
    _reduce_expert_route_attention_kernel[rows,](
        route_out,
        route_lse,
        top_slots,
        exact_out,
        exact_lse,
        ROUTE_COUNT=route_count,
        ROUTE_BLOCK=triton.next_power_of_2(route_count),
        VALUE_DIM=value_dim,
        VALUE_BLOCK_DIM=triton.next_power_of_2(value_dim),
        num_warps=reduce_num_warps,
    )
    record_boundary()
    if timing_events is not None:
        for name, begin, end in zip(
            ("dispatch_prepare", "dispatch_sort", "dispatch_group", "dispatch_blocks"),
            dispatch_boundaries[:-1],
            dispatch_boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        for name, begin, end in zip(
            ("dispatch", "pack", "kernel", "reduce"),
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            timing_events.setdefault(name, []).append((begin, end))
        timing_events.setdefault("total", []).append((boundaries[0], boundaries[-1]))
    return (
        exact_out.reshape(batch, query_heads, query_len, value_dim),
        exact_lse.reshape(batch, query_heads, query_len),
    )
