#!/usr/bin/env python3
"""Upper-bound benchmark for top-8 GQA-union two-tier decode.

The exact-leaf attention uses a physically compact K/V buffer on purpose.  Its
timing answers whether a regular GQA-16 attention kernel over the union can pay
off before implementing the indexed layout adapter.  K/V compaction is not
included in the reported attention time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

from model.kernels.paged_leaf_attention import (
    _decode_route_coarse_gqa_segments_kernel,
    _reduce_decode_route_coarse_vector_topk_kernel,
    new_fused_decode_buffers,
)


@triton.jit
def _top8_gqa_union_append_kernel(
    top_slots,
    seen_stamps,
    sequence_epochs,
    union_counts,
    union_token_counts,
    union_slots,
    TOP_BATCH_STRIDE,
    TOP_HEAD_STRIDE,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    GQA: tl.constexpr,
    ROUTES: tl.constexpr,
    STATE_LEN: tl.constexpr,
    LOCAL_LEN: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
):
    """Append the unique top-8 routes of one GQA group without sorting."""
    sequence = tl.program_id(0).to(tl.int64)
    batch = sequence // KV_HEADS
    kv_head = sequence - batch * KV_HEADS
    candidate = tl.arange(0, CANDIDATE_BLOCK)
    epoch = tl.load(sequence_epochs + sequence).to(tl.int32) + 1
    tl.store(sequence_epochs + sequence, epoch)
    candidate_valid = candidate < GQA * ROUTES
    query_lane = candidate // ROUTES
    route = candidate - query_lane * ROUTES
    query_head = kv_head * GQA + query_lane
    slot = tl.load(
        top_slots
        + batch * TOP_BATCH_STRIDE
        + query_head * TOP_HEAD_STRIDE
        + route,
        mask=candidate_valid,
        other=-1,
    ).to(tl.int32)
    valid = candidate_valid & (slot >= 0) & (slot < STATE_LEN)
    safe_slot = tl.where(valid, slot, 0)
    stamp_pointer = seen_stamps + sequence * STATE_LEN + safe_slot
    epoch_vector = tl.full((CANDIDATE_BLOCK,), 0, tl.int32) + epoch
    observed_epoch = tl.load(stamp_pointer, mask=valid, other=epoch_vector)
    old_epoch = tl.atomic_cas(
        stamp_pointer,
        tl.where(valid, observed_epoch, -1),
        epoch_vector,
        sem="relaxed",
    )
    unique = valid & (old_epoch != epoch_vector)
    unique_integer = unique.to(tl.int32)
    destination = tl.cumsum(unique_integer, axis=0) - 1
    tl.store(
        union_counts + sequence + candidate * 0,
        tl.sum(unique_integer, axis=0),
        mask=candidate == 0,
    )
    tl.store(
        union_token_counts + sequence + candidate * 0,
        LOCAL_LEN,
        mask=candidate == 0,
    )
    tl.store(
        union_slots + sequence * (GQA * ROUTES) + destination,
        slot,
        mask=unique,
    )


@triton.jit(
    do_not_specialize=["union_count"],
    do_not_specialize_on_alignment=["union_count"],
)
def _expand_equal_top8_union_indices_kernel(
    union_slots,
    union_counts,
    union_token_counts,
    token_indices,
    union_count,
    PHYSICAL_TOKENS: tl.constexpr,
    UNION_CAPACITY: tl.constexpr,
    LOCAL_LEN: tl.constexpr,
    LEAVES_PER_CENTROID: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Expand selected centroids plus local tokens into one indexed sequence."""
    sequence = tl.program_id(0).to(tl.int64)
    work = tl.program_id(1).to(tl.int64)
    token_offset = tl.arange(0, BLOCK_K)
    if work == 0:
        for begin in tl.range(0, LOCAL_LEN, BLOCK_K, num_stages=1):
            token = begin + token_offset
            valid = token < LOCAL_LEN
            physical = PHYSICAL_TOKENS - LOCAL_LEN + token
            tl.store(
                token_indices + sequence * union_count + token,
                physical,
                mask=valid,
            )
    else:
        rank = work - 1
        selected_count = tl.load(union_counts + sequence).to(tl.int32)
        valid_rank = rank < selected_count
        slot = tl.load(
            union_slots + sequence * UNION_CAPACITY + rank,
            mask=valid_rank,
            other=0,
        ).to(tl.int32)
        destination = tl.atomic_add(
            union_token_counts + sequence,
            LEAVES_PER_CENTROID,
            mask=valid_rank,
            sem="relaxed",
        ).to(tl.int32)
        token = token_offset
        valid = valid_rank & (token < LEAVES_PER_CENTROID)
        physical = slot * LEAVES_PER_CENTROID + token
        tl.store(
            token_indices + sequence * union_count + destination + token,
            physical,
            mask=valid,
        )


@triton.jit
def _indexed_gqa_split_attention_kernel(
    q,
    k,
    v,
    indices,
    lengths,
    partial_out,
    partial_lse,
    max_k,
    PHYSICAL_TOKENS: tl.constexpr,
    QUERY_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    """M16 indexed attention, split across N for decode occupancy."""
    sequence = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    query_row = tl.arange(0, QUERY_ROWS)
    dimension = tl.arange(0, HEAD_DIM)
    token_offset = tl.arange(0, BLOCK_N)
    queries = tl.load(
        q
        + (sequence * QUERY_ROWS + query_row[:, None]) * HEAD_DIM
        + dimension[None, :]
    )
    length = tl.load(lengths + sequence).to(tl.int32)
    maximum = tl.full((QUERY_ROWS,), -float("inf"), tl.float32)
    denominator = tl.zeros((QUERY_ROWS,), tl.float32)
    accumulator = tl.zeros((QUERY_ROWS, HEAD_DIM), tl.float32)

    for token_begin in tl.range(
        split * BLOCK_N, max_k, SPLITS * BLOCK_N, num_stages=1
    ):
        logical_token = token_begin + token_offset
        valid = logical_token < length
        physical_token = tl.load(
            indices + sequence * max_k + logical_token,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid &= (physical_token >= 0) & (physical_token < PHYSICAL_TOKENS)
        safe_token = tl.where(valid, physical_token, 0)
        keys = tl.load(
            k
            + (sequence * PHYSICAL_TOKENS + safe_token[:, None]) * HEAD_DIM
            + dimension[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        values = tl.load(
            v
            + (sequence * PHYSICAL_TOKENS + safe_token[:, None]) * HEAD_DIM
            + dimension[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(queries, tl.trans(keys), out_dtype=tl.float32)
        scores *= SCALE_LOG2
        scores = tl.where(valid[None, :], scores, -float("inf"))
        block_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.math.exp2(maximum - new_maximum)
        probabilities = tl.math.exp2(scores - new_maximum[:, None])
        probabilities = tl.where(valid[None, :], probabilities, 0.0)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(values.dtype), values, out_dtype=tl.float32
        )
        maximum = new_maximum

    has_mass = denominator > 0.0
    partial_row = (
        (sequence * QUERY_ROWS + query_row) * SPLITS + split
    ).to(tl.int64)
    tl.store(
        partial_out + partial_row[:, None] * HEAD_DIM + dimension[None, :],
        tl.where(has_mass[:, None], accumulator / denominator[:, None], 0.0),
    )
    tl.store(
        partial_lse + partial_row,
        tl.where(
            has_mass,
            (maximum + tl.math.log2(denominator)) * 0.6931471805599453,
            -float("inf"),
        ),
    )


@triton.jit
def _reduce_indexed_gqa_split_attention_kernel(
    partial_out,
    partial_lse,
    output,
    output_lse,
    QUERY_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.arange(0, SPLIT_BLOCK)
    dimension = tl.arange(0, HEAD_DIM)
    valid = split < SPLITS
    lse = tl.load(
        partial_lse + row * SPLITS + split,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(lse, axis=0)
    weight = tl.where(valid, tl.exp(lse - maximum), 0.0)
    denominator = tl.sum(weight, axis=0)
    values = tl.load(
        partial_out
        + (row * SPLITS + split[:, None]) * HEAD_DIM
        + dimension[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    result = tl.sum(values * weight[:, None], axis=0) / denominator
    tl.store(output + row * HEAD_DIM + dimension, result)
    tl.store(
        output_lse + row + dimension * 0,
        maximum + tl.log(denominator),
        mask=dimension == 0,
    )


def elapsed_ms(
    call: Callable[[], object], *, warmups: int, repeats: int
) -> float:
    result = None
    for _ in range(warmups):
        result = call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        result = call()
    end.record()
    end.synchronize()
    if result is None:
        raise AssertionError("benchmark did not execute")
    return float(begin.elapsed_time(end)) / repeats


def unified_attention_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Prepare AITER's production decode kernel for an ordinary KV field."""
    from aiter.ops.triton.unified_attention import unified_attention

    batch, kv_heads, length, head_dim = k.shape
    query_heads = int(q.size(1))
    block_size = 64
    blocks = triton.cdiv(length, block_size)
    padded = blocks * block_size
    if padded != length:
        k = torch.nn.functional.pad(k, (0, 0, 0, padded - length))
        v = torch.nn.functional.pad(v, (0, 0, 0, padded - length))

    def paged(source: torch.Tensor) -> torch.Tensor:
        return (
            source.permute(0, 2, 1, 3)
            .reshape(batch, blocks, block_size, kv_heads, head_dim)
            .reshape(batch * blocks, block_size, kv_heads, head_dim)
            .contiguous()
        )

    page_k = paged(k)
    page_v = paged(v)
    block_table = torch.arange(
        batch * blocks, dtype=torch.int32, device=q.device
    ).reshape(batch, blocks)
    lengths = torch.full(
        (batch,), length, dtype=torch.int32, device=q.device
    )
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    q3 = q[:, :, 0].contiguous()
    output = torch.empty(
        batch, query_heads, head_dim, dtype=q.dtype, device=q.device
    )

    def run() -> torch.Tensor:
        unified_attention(
            q=q3,
            k=page_k,
            v=page_v,
            out=output,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=lengths,
            max_seqlen_k=length,
            softmax_scale=head_dim**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )
        return output

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=65_536)
    parser.add_argument("--local-length", type=int, default=512)
    parser.add_argument("--state-factor", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument("--leaf-tokens-per-centroid", type=int, default=16)
    parser.add_argument(
        "--union-token-sweep",
        type=int,
        nargs="+",
        default=(512, 1024, 2048, 4096, 8192),
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.gqa != 16 or args.head_dim != 128:
        raise ValueError("this first prototype targets Muse's D=128, GQA-16 geometry")

    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    batch = args.batch_size
    kv_heads = args.kv_heads
    gqa = args.gqa
    query_heads = kv_heads * gqa
    head_dim = args.head_dim
    state_len = round(args.state_factor * math.sqrt(args.context_length))
    route_group_n = 64
    route_segment_tiles = 2
    active_segments = triton.cdiv(
        state_len, route_group_n * route_segment_tiles
    )

    q = torch.randn(
        batch,
        query_heads,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    mean_k = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=dtype
    )
    mean_v = torch.randn_like(mean_k)
    counts = torch.full(
        (batch, kv_heads, state_len, 1),
        args.leaf_tokens_per_centroid,
        device=device,
        dtype=torch.float32,
    )
    # Production state stores sums and reconstructs BF16 means while routing.
    state_k = (mean_k.float() * counts).to(dtype)
    state_v = (mean_v.float() * counts).to(dtype)
    cache_indices = torch.arange(batch, device=device, dtype=torch.int64)
    buffers = new_fused_decode_buffers(
        q,
        splits=args.routes,
        state_capacity=state_len,
        route_group_size=route_group_n,
        route_segment_tiles=route_segment_tiles,
    )
    max_segments = int(buffers["route_group_lse"].size(2))

    def route_top8_and_coarse() -> torch.Tensor:
        _decode_route_coarse_gqa_segments_kernel[
            (batch * kv_heads, active_segments)
        ](
            q,
            state_k,
            state_v,
            counts,
            cache_indices,
            buffers["route_candidate_scores"],
            buffers["route_candidate_indices"],
            buffers["route_group_out"],
            buffers["route_group_lse"],
            state_k.stride(0),
            state_k.stride(1),
            state_k.stride(2),
            state_v.stride(0),
            state_v.stride(1),
            state_v.stride(2),
            counts.stride(0),
            counts.stride(1),
            counts.stride(2),
            state_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=gqa,
            HEAD_DIM=head_dim,
            SCALE=head_dim**-0.5,
            GROUP_N=route_group_n,
            SEGMENT_TILES=route_segment_tiles,
            MAX_GROUPS=max_segments,
            PROTECTED_LEN=0,
            MAX_LEAF_TOKENS=0,
            MERGE_TILE_TOPK=True,
            BYPASS_KV_L1=False,
            POST_DOT_NORMALIZE=False,
            POST_PV_NORMALIZE=False,
            num_warps=2,
            num_stages=3,
            waves_per_eu=1,
        )
        _reduce_decode_route_coarse_vector_topk_kernel[
            (batch * query_heads,)
        ](
            buffers["route_candidate_scores"],
            buffers["route_candidate_indices"],
            buffers["route_group_out"],
            buffers["route_group_lse"],
            buffers["route_top_slots"],
            buffers["route_top_scores"],
            buffers["coarse_out"],
            buffers["coarse_lse"],
            active_segments,
            active_segments,
            HEAD_DIM=head_dim,
            ROUTE_COUNT=args.routes,
            MAX_SEGMENTS=max_segments,
            CANDIDATE_BLOCK=triton.next_power_of_2(
                max(16, active_segments * args.routes)
            ),
            SEGMENT_BLOCK=triton.next_power_of_2(active_segments),
            APPLY_MASS_CUTOFF=False,
            LOG_MASS_FRACTION=0.0,
            num_warps=2,
            waves_per_eu=1,
        )
        return buffers["route_top_slots"]

    route_top8_and_coarse()
    torch.cuda.synchronize()
    top_slots = buffers["route_top_slots"][..., 0, :]
    grouped_routes = top_slots.reshape(batch, kv_heads, gqa * args.routes)
    union_counts = torch.tensor(
        [
            torch.unique(row[row >= 0]).numel()
            for row in grouped_routes.reshape(-1, gqa * args.routes)
        ],
        device=device,
        dtype=torch.int32,
    )
    union_leaf_counts = union_counts * args.leaf_tokens_per_centroid
    route_ms = elapsed_ms(
        route_top8_and_coarse, warmups=args.warmups, repeats=args.repeats
    )

    # This intentionally uses ordinary tensor operations as a reference cost;
    # a production implementation should replace it with one fixed-width GPU
    # union/list kernel and must not synchronize to obtain a dynamic shape.
    def reference_union_metadata() -> tuple[torch.Tensor, torch.Tensor]:
        sorted_slots = grouped_routes.sort(dim=-1).values
        unique = torch.ones_like(sorted_slots, dtype=torch.bool)
        unique[..., 1:] = sorted_slots[..., 1:] != sorted_slots[..., :-1]
        valid = unique & (sorted_slots >= 0)
        fixed_union = torch.where(valid, sorted_slots, -1)
        return fixed_union, valid.sum(dim=-1, dtype=torch.int32)

    metadata_ms = elapsed_ms(
        reference_union_metadata, warmups=args.warmups, repeats=args.repeats
    )

    sequence_count = batch * kv_heads
    seen_stamps = torch.zeros(
        sequence_count,
        state_len,
        device=device,
        dtype=torch.int32,
    )
    sequence_epochs = torch.zeros(
        sequence_count, device=device, dtype=torch.int32
    )
    fast_union_counts = torch.empty(
        sequence_count, device=device, dtype=torch.int32
    )
    fast_union_token_counts = torch.empty(
        sequence_count, device=device, dtype=torch.int32
    )
    fast_union_slots = torch.empty(
        sequence_count,
        gqa * args.routes,
        device=device,
        dtype=torch.int32,
    )

    def fast_union_metadata() -> tuple[torch.Tensor, torch.Tensor]:
        _top8_gqa_union_append_kernel[(sequence_count,)](
            top_slots,
            seen_stamps,
            sequence_epochs,
            fast_union_counts,
            fast_union_token_counts,
            fast_union_slots,
            top_slots.stride(0),
            top_slots.stride(1),
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            GQA=gqa,
            ROUTES=args.routes,
            STATE_LEN=state_len,
            LOCAL_LEN=args.local_length,
            CANDIDATE_BLOCK=triton.next_power_of_2(gqa * args.routes),
            num_warps=4,
            waves_per_eu=1,
        )
        return fast_union_slots, fast_union_counts

    fast_union_metadata()
    torch.cuda.synchronize()
    expected_sets = [
        set(row[row >= 0].tolist())
        for row in grouped_routes.reshape(-1, gqa * args.routes).cpu()
    ]
    actual_counts = fast_union_counts.cpu().tolist()
    actual_slots = fast_union_slots.cpu()
    for sequence, (expected, actual_count) in enumerate(
        zip(expected_sets, actual_counts, strict=True)
    ):
        actual = set(actual_slots[sequence, :actual_count].tolist())
        if actual != expected:
            raise AssertionError(
                f"fast GQA union differs in sequence {sequence}: "
                f"{len(actual)} != {len(expected)}"
            )
    fast_metadata_ms = elapsed_ms(
        fast_union_metadata, warmups=args.warmups, repeats=args.repeats
    )

    full_k = torch.randn(
        batch,
        kv_heads,
        args.context_length,
        head_dim,
        device=device,
        dtype=dtype,
    )
    full_v = torch.randn_like(full_k)
    full_call = unified_attention_call(q, full_k, full_v)
    full_ms = elapsed_ms(
        full_call, warmups=args.warmups, repeats=args.repeats
    )

    local_k = full_k[..., : args.local_length, :].contiguous()
    local_v = full_v[..., : args.local_length, :].contiguous()
    local_call = unified_attention_call(q, local_k, local_v)
    local_ms = elapsed_ms(
        local_call, warmups=args.warmups, repeats=args.repeats
    )

    grouped_q = (
        q.view(batch, kv_heads, gqa, 1, head_dim)
        .reshape(batch * kv_heads, gqa, 1, head_dim)
        .contiguous()
    )
    max_union_tokens = max(
        max(args.union_token_sweep), int(union_leaf_counts.max().item())
    )
    compact_k = torch.randn(
        batch * kv_heads,
        1,
        max_union_tokens,
        head_dim,
        device=device,
        dtype=dtype,
    )
    compact_v = torch.randn_like(compact_k)
    union_records: list[dict[str, float | int]] = []
    sweep = sorted(
        set(args.union_token_sweep)
        | {
            int(round(float(union_leaf_counts.float().mean().item()) / 64.0) * 64),
            int(union_leaf_counts.max().item()),
        }
    )
    for union_tokens in sweep:
        if union_tokens <= 0:
            continue
        union_call = unified_attention_call(
            grouped_q,
            compact_k[..., :union_tokens, :].contiguous(),
            compact_v[..., :union_tokens, :].contiguous(),
        )
        union_ms = elapsed_ms(
            union_call, warmups=args.warmups, repeats=args.repeats
        )
        sequential_ms = route_ms + fast_metadata_ms + local_ms + union_ms
        union_records.append(
            {
                "union_tokens": union_tokens,
                "compact_union_attention_ms": union_ms,
                "route_metadata_local_union_ms": sequential_ms,
                "speedup_vs_full_attention_before_final_reduce": (
                    full_ms / sequential_ms
                ),
            }
        )

    # Materialize only integer token indices for the actual top-8 union. K/V
    # remain in their original physical cache order for the indexed kernel.
    actual_union_slots = actual_slots
    indexed_rows: list[torch.Tensor] = []
    for sequence, actual_count in enumerate(actual_counts):
        slots = actual_union_slots[sequence, :actual_count].to(device=device).long()
        leaf_offset = torch.arange(
            args.leaf_tokens_per_centroid, device=device, dtype=torch.long
        )
        leaf_indices = (
            slots[:, None] * args.leaf_tokens_per_centroid + leaf_offset[None, :]
        ).reshape(-1)
        # Fold the entire 512-token local branch into the same exact attention
        # sequence. A tagged two-source index is needed in production; using
        # the tail of this synthetic physical pool is equivalent for timing.
        local_indices = torch.arange(
            args.context_length - args.local_length,
            args.context_length,
            device=device,
            dtype=torch.long,
        )
        indexed_rows.append(torch.cat((leaf_indices, local_indices)))
    expected_indexed_lengths = torch.tensor(
        [row.numel() for row in indexed_rows], device=device, dtype=torch.int32
    )
    indexed_max_k = int(expected_indexed_lengths.max().item())
    indexed_token_table = torch.empty(
        sequence_count, indexed_max_k, device=device, dtype=torch.int32
    )

    def expand_union_indices() -> torch.Tensor:
        _expand_equal_top8_union_indices_kernel[
            (sequence_count, gqa * args.routes + 1)
        ](
            fast_union_slots,
            fast_union_counts,
            fast_union_token_counts,
            indexed_token_table,
            indexed_max_k,
            PHYSICAL_TOKENS=args.context_length,
            UNION_CAPACITY=gqa * args.routes,
            LOCAL_LEN=args.local_length,
            LEAVES_PER_CENTROID=args.leaf_tokens_per_centroid,
            BLOCK_K=128,
            num_warps=1,
            waves_per_eu=1,
        )
        return indexed_token_table

    def fast_union_and_indices() -> torch.Tensor:
        fast_union_metadata()
        return expand_union_indices()

    fast_union_and_indices()
    torch.cuda.synchronize()
    indexed_lengths = fast_union_token_counts
    if not torch.equal(indexed_lengths, expected_indexed_lengths):
        raise AssertionError("GPU union token counts differ from the reference")
    for sequence, reference_row in enumerate(indexed_rows):
        actual_row = indexed_token_table[
            sequence, : int(indexed_lengths[sequence].item())
        ].long()
        if set(actual_row.cpu().tolist()) != set(reference_row.cpu().tolist()):
            raise AssertionError(
                f"GPU union token list differs in sequence {sequence}"
            )
    fast_union_and_indices_ms = elapsed_ms(
        fast_union_and_indices,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    physical_k = full_k.reshape(
        sequence_count, args.context_length, head_dim
    )
    physical_v = full_v.reshape_as(physical_k)
    indexed_q = grouped_q[:, :, 0].contiguous()

    combined_compact_k = torch.randn(
        sequence_count,
        1,
        indexed_max_k,
        head_dim,
        device=device,
        dtype=dtype,
    )
    combined_compact_v = torch.randn_like(combined_compact_k)
    combined_compact_call = unified_attention_call(
        grouped_q, combined_compact_k, combined_compact_v
    )
    combined_compact_ms = elapsed_ms(
        combined_compact_call, warmups=args.warmups, repeats=args.repeats
    )

    indexed_records: list[dict[str, float | int]] = []
    for block_n, splits in (
        (16, 16),
        (16, 32),
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 16),
        (64, 32),
        (128, 16),
        (128, 32),
    ):
        partial_out = torch.empty(
            sequence_count,
            gqa,
            splits,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            sequence_count,
            gqa,
            splits,
            device=device,
            dtype=torch.float32,
        )
        indexed_output = torch.empty(
            sequence_count, gqa, head_dim, device=device, dtype=dtype
        )
        indexed_lse = torch.empty(
            sequence_count, gqa, device=device, dtype=torch.float32
        )

        def indexed_attention_body() -> torch.Tensor:
            _indexed_gqa_split_attention_kernel[(sequence_count, splits)](
                indexed_q,
                physical_k,
                physical_v,
                indexed_token_table,
                indexed_lengths,
                partial_out,
                partial_lse,
                indexed_max_k,
                PHYSICAL_TOKENS=args.context_length,
                QUERY_ROWS=gqa,
                HEAD_DIM=head_dim,
                SPLITS=splits,
                BLOCK_N=block_n,
                SCALE_LOG2=(head_dim**-0.5) * math.log2(math.e),
                num_warps=4,
                waves_per_eu=1,
            )
            return partial_out

        def indexed_attention_reduce() -> torch.Tensor:
            _reduce_indexed_gqa_split_attention_kernel[
                (sequence_count * gqa,)
            ](
                partial_out,
                partial_lse,
                indexed_output,
                indexed_lse,
                QUERY_ROWS=gqa,
                HEAD_DIM=head_dim,
                SPLITS=splits,
                SPLIT_BLOCK=triton.next_power_of_2(splits),
                num_warps=4,
                waves_per_eu=1,
            )
            return indexed_output

        def indexed_attention() -> torch.Tensor:
            indexed_attention_body()
            return indexed_attention_reduce()

        indexed_attention()
        torch.cuda.synchronize()
        reference_length = int(indexed_lengths[0].item())
        reference_indices = indexed_token_table[0, :reference_length].long()
        reference_k = physical_k[0].index_select(0, reference_indices).float()
        reference_v = physical_v[0].index_select(0, reference_indices).float()
        reference_scores = (
            indexed_q[0].float() @ reference_k.transpose(0, 1)
        ) * (head_dim**-0.5)
        reference_output = torch.softmax(reference_scores, dim=-1) @ reference_v
        reference_lse = torch.logsumexp(reference_scores, dim=-1)
        output_error = float(
            (indexed_output[0].float() - reference_output).abs().max().item()
        )
        lse_error = float(
            (indexed_lse[0] - reference_lse).abs().max().item()
        )
        body_ms = elapsed_ms(
            indexed_attention_body,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        reduce_ms = elapsed_ms(
            indexed_attention_reduce,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        total_ms = elapsed_ms(
            indexed_attention,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        indexed_records.append(
            {
                "block_n": block_n,
                "splits": splits,
                "body_ms": body_ms,
                "reduce_ms": reduce_ms,
                "total_ms": total_ms,
                "output_max_abs": output_error,
                "lse_max_abs": lse_error,
                "total_vs_compact": total_ms / combined_compact_ms,
                "estimated_pipeline_ms": (
                    route_ms + fast_union_and_indices_ms + total_ms
                ),
                "estimated_speedup_vs_full_before_coarse_merge": (
                    full_ms
                    / (route_ms + fast_union_and_indices_ms + total_ms)
                ),
            }
        )

    properties = torch.cuda.get_device_properties(device)
    result = {
        "device": properties.name,
        "geometry": {
            "batch": batch,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "gqa": gqa,
            "head_dim": head_dim,
            "context_length": args.context_length,
            "state_length": state_len,
            "local_length": args.local_length,
            "topk": args.routes,
            "leaf_tokens_per_centroid": args.leaf_tokens_per_centroid,
        },
        "route_union": {
            "mean_union_centroids": float(union_counts.float().mean().item()),
            "maximum_union_centroids": int(union_counts.max().item()),
            "mean_union_leaf_tokens": float(
                union_leaf_counts.float().mean().item()
            ),
            "maximum_union_leaf_tokens": int(union_leaf_counts.max().item()),
        },
        "timings_ms": {
            "full_attention": full_ms,
            "top8_and_coarse": route_ms,
            "reference_torch_union_metadata": metadata_ms,
            "fixed_gpu_union_metadata": fast_metadata_ms,
            "fixed_gpu_union_and_token_indices": fast_union_and_indices_ms,
            "local_attention": local_ms,
            "compact_combined_local_union_attention": combined_compact_ms,
        },
        "union_sweep": union_records,
        "indexed_combined_local_union": {
            "mean_tokens": float(indexed_lengths.float().mean().item()),
            "maximum_tokens": indexed_max_k,
            "records": indexed_records,
        },
        "notes": [
            "Compact union attention excludes K/V compaction and indexed layout costs.",
            "The fixed GPU union uses persistent generation stamps and atomic CAS.",
            "The union-and-index timing also expands selected centroids and local tokens.",
            "The sequential estimate excludes coarse/leaf/local final LSE reduction.",
            "The indexed sweep folds the 512 local tokens into the leaf union call.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
