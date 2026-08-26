#!/usr/bin/env python3
"""Time production three-tier routing against rectangular re-split pieces."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl

from model.kernels.paged_leaf_attention import (
    _decode_route_coarse_gqa_groups_kernel,
    _decode_route_coarse_gqa_segments_kernel,
    _materialize_state_summary_scores_gqa_kernel,
    _materialized_state_pv_dequant_split_kernel,
    _materialized_state_pv_int8_split_kernel,
    _materialized_state_pv_kernel,
    _materialized_state_normalized_pv_split_kernel,
    _materialized_state_pv_split_kernel,
    _materialized_state_softmax_kernel,
    _reduce_decode_route_coarse_kernel,
    _reduce_decode_route_coarse_vector_kernel,
    _reduce_decode_route_coarse_vector_topk_kernel,
    _reduce_decode_route_coarse_vector_topk_splitd_kernel,
    _reduce_materialized_state_top8_kernel,
    _reduce_materialized_state_pv_kernel,
    materialized_state_route_gqa,
    materialize_page_summary_scores_gqa,
    new_fused_decode_buffers,
)
from model.kernels.lod_kernels import _quantize_state_mean_values_int8_kernel


@dataclass(frozen=True)
class Geometry:
    name: str
    head_dim: int
    kv_heads: int
    gqa: int


GEOMETRIES = {
    item.name: item
    for item in (
        Geometry("muse", 128, 2, 16),
        Geometry("olmo", 128, 8, 5),
        Geometry("phi", 128, 2, 4),
        Geometry("qwen", 256, 4, 6),
        Geometry("gemma", 512, 2, 8),
    )
}


@triton.jit
def _pack_score_index(scores, indices):
    score_bits = scores.to(tl.uint32, bitcast=True)
    negative = (score_bits & 0x80000000) != 0
    ordered_bits = tl.where(
        negative,
        score_bits ^ 0xFFFFFFFF,
        score_bits ^ 0x80000000,
    ).to(tl.int64)
    score_rank = ordered_bits - 2147483648
    inverse_index = 4294967295 - indices.to(tl.int64)
    return score_rank * 4294967296 + inverse_index


@triton.jit
def _unpack_score_index(packed):
    inverse_index = packed & 0xFFFFFFFF
    indices = (4294967295 - inverse_index).to(tl.int64)
    score_rank = packed >> 32
    ordered_bits = (score_rank + 2147483648).to(tl.uint32)
    negative = (ordered_bits & 0x80000000) == 0
    score_bits = tl.where(
        negative,
        ordered_bits ^ 0xFFFFFFFF,
        ordered_bits ^ 0x80000000,
    )
    return score_bits.to(tl.float32, bitcast=True), indices


@triton.jit
def _score_table_tile_top8_kernel(
    scores,
    candidate_scores,
    candidate_indices,
    length,
    TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    offset = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offset < length
    values = tl.load(
        scores + row * length + offset,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    packed = _pack_score_index(values, offset)
    best = tl.topk(packed, 8, dim=0)
    best_scores, best_indices = _unpack_score_index(best)
    rank = tl.arange(0, 8)
    output_offset = (row * TILES + tile) * 8 + rank
    tl.store(candidate_scores + output_offset, best_scores)
    tl.store(candidate_indices + output_offset, best_indices)


@triton.jit
def _score_table_reduce_top8_kernel(
    candidate_scores,
    candidate_indices,
    top_scores,
    top_indices,
    candidate_count,
    CANDIDATE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, CANDIDATE_BLOCK)
    valid = offset < candidate_count
    values = tl.load(
        candidate_scores + row * candidate_count + offset,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    indices = tl.load(
        candidate_indices + row * candidate_count + offset,
        mask=valid,
        other=0,
    )
    packed = _pack_score_index(values, indices)
    best = tl.topk(packed, 8, dim=0)
    best_scores, best_indices = _unpack_score_index(best)
    rank = tl.arange(0, 8)
    tl.store(top_scores + row * 8 + rank, best_scores)
    tl.store(top_indices + row * 8 + rank, best_indices)


def elapsed(call: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def aiter_attention(
    q: torch.Tensor, mean_k: torch.Tensor, mean_v: torch.Tensor
) -> Callable[[], None]:
    """Return an AITER decode call over an ordinary equal-count summary field."""
    from aiter.ops.triton.unified_attention import unified_attention

    batch, kv_heads, length, dim = mean_k.shape
    q_heads = int(q.size(1))
    block_size = 64
    blocks = triton.cdiv(length, block_size)
    padded = blocks * block_size
    if padded > length:
        mean_k = torch.nn.functional.pad(mean_k, (0, 0, 0, padded - length))
        mean_v = torch.nn.functional.pad(mean_v, (0, 0, 0, padded - length))

    def paged(source: torch.Tensor) -> torch.Tensor:
        return (
            source.permute(0, 2, 1, 3)
            .reshape(batch, blocks, block_size, kv_heads, dim)
            .reshape(batch * blocks, block_size, kv_heads, dim)
            .contiguous()
        )

    keys, values = paged(mean_k), paged(mean_v)
    table = torch.arange(
        batch * blocks, dtype=torch.int32, device=q.device
    ).reshape(batch, blocks)
    lengths = torch.full((batch,), length, dtype=torch.int32, device=q.device)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    q3 = q[:, :, 0].contiguous()
    out = torch.empty(batch, q_heads, dim, dtype=q.dtype, device=q.device)

    def run() -> None:
        unified_attention(
            q=q3,
            k=keys,
            v=values,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=lengths,
            max_seqlen_k=length,
            softmax_scale=dim**-0.5,
            # Unified attention interprets a one-token decode query as the
            # final position, so its required causal mode still sees all K.
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

    return run


def benchmark(geometry: Geometry, args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cuda")
    batch, kv_heads, gqa, dim = (
        args.batch_size,
        geometry.kv_heads,
        geometry.gqa,
        geometry.head_dim,
    )
    q_heads = kv_heads * gqa
    state_len = round(args.state_growth_factor * math.sqrt(args.context_length))
    live_page_len = args.live_page_length or math.ceil(
        args.context_length / args.page_size
    )
    page_len = args.page_capacity or live_page_len
    if page_len < live_page_len:
        raise ValueError("page capacity cannot be shorter than the live page field")
    group_n = (
        args.route_group_size
        if args.route_group_size > 0
        else (64 if dim == 128 else 32)
    )
    groups = triton.cdiv(state_len, group_n)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + dim + kv_heads)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    q = randn((batch, q_heads, 1, dim))
    raw_mean_k = randn((batch, kv_heads, state_len, dim))
    raw_mean_v = randn((batch, kv_heads, state_len, dim))
    count = max(1, round(args.context_length / state_len))
    counts = (
        torch.randint(
            1,
            max(3, 2 * count + 1),
            (batch, kv_heads, state_len, 1),
            dtype=torch.int32,
            device=device,
            generator=generator,
        ).float()
        if args.random_counts
        else torch.full(
            (batch, kv_heads, state_len, 1),
            count,
            dtype=torch.float32,
            device=device,
        )
    )
    sum_k = (raw_mean_k.float() * counts).to(torch.bfloat16)
    sum_v = (raw_mean_v.float() * counts).to(torch.bfloat16)
    # These are exactly what the current route reconstructs just in time.
    mean_k = (sum_k.float() / counts).to(torch.bfloat16)
    mean_v = (sum_v.float() / counts).to(torch.bfloat16)
    ones = torch.ones_like(counts)
    cache_indices = torch.arange(batch, dtype=torch.int64, device=device)
    buffers = new_fused_decode_buffers(
        q,
        splits=args.routes,
        state_capacity=state_len,
        route_group_size=group_n,
        materialized_state_route=True,
    )

    def route_groups(
        keys: torch.Tensor, values: torch.Tensor, input_counts: torch.Tensor
    ) -> None:
        _decode_route_coarse_gqa_groups_kernel[(batch * kv_heads, groups)](
            q,
            keys,
            values,
            input_counts,
            cache_indices,
            buffers["route_candidate_scores"],
            buffers["route_candidate_indices"],
            buffers["route_group_out"],
            buffers["route_group_lse"],
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            input_counts.stride(0),
            input_counts.stride(1),
            input_counts.stride(2),
            state_len,
            QUERY_HEADS=q_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=gqa,
            HEAD_DIM=dim,
            SCALE=dim**-0.5,
            GROUP_N=group_n,
            MAX_GROUPS=groups,
            PROTECTED_LEN=0,
            MAX_LEAF_TOKENS=0,
            USE_DOT=True,
            num_warps=1 if dim == 128 else 2,
            waves_per_eu=1,
        )

    def route_reduce() -> None:
        _reduce_decode_route_coarse_kernel[(batch * q_heads,)](
            buffers["route_candidate_scores"],
            buffers["route_candidate_indices"],
            buffers["route_group_out"],
            buffers["route_group_lse"],
            buffers["route_top_slots"],
            buffers["route_top_scores"],
            buffers["coarse_out"],
            buffers["coarse_lse"],
            groups,
            HEAD_DIM=dim,
            ROUTE_COUNT=args.routes,
            OPEN_COUNT=args.routes,
            MAX_GROUPS=groups,
            CANDIDATE_TILE=1024,
            APPLY_MASS_CUTOFF=False,
            LOG_MASS_FRACTION=0.0,
            num_warps=2 if dim == 128 else 4,
            waves_per_eu=1,
        )

    route_groups(sum_k, sum_v, counts)
    route_reduce()
    old_top_slots = buffers["route_top_slots"].detach().clone()
    old_coarse_out = buffers["coarse_out"].detach().clone()
    old_coarse_lse = buffers["coarse_lse"].detach().clone()

    def segmented_route_record(
        segment_tiles: int,
        warps: int,
        waves: int = 1,
        stages: int = 3,
        bypass_l1: bool = False,
        merge_tile_topk: bool = True,
        post_dot_normalize: bool = False,
        post_pv_normalize: bool = False,
    ) -> dict[str, float]:
        segment_width = group_n * segment_tiles
        segments = triton.cdiv(state_len, segment_width)
        segment_buffers = new_fused_decode_buffers(
            q,
            splits=args.routes,
            state_capacity=state_len,
            route_group_size=(segment_width if merge_tile_topk else group_n),
        )
        max_segments = int(segment_buffers["route_group_lse"].size(2))
        candidate_groups = (
            segments if merge_tile_topk else triton.cdiv(state_len, group_n)
        )

        def route_segments() -> None:
            _decode_route_coarse_gqa_segments_kernel[
                (batch * kv_heads, segments)
            ](
                q,
                sum_k,
                sum_v,
                counts,
                cache_indices,
                segment_buffers["route_candidate_scores"],
                segment_buffers["route_candidate_indices"],
                segment_buffers["route_group_out"],
                segment_buffers["route_group_lse"],
                sum_k.stride(0),
                sum_k.stride(1),
                sum_k.stride(2),
                sum_v.stride(0),
                sum_v.stride(1),
                sum_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                HEAD_DIM=dim,
                SCALE=dim**-0.5,
                GROUP_N=group_n,
                SEGMENT_TILES=segment_tiles,
                MAX_GROUPS=max_segments,
                PROTECTED_LEN=0,
                MAX_LEAF_TOKENS=0,
                MERGE_TILE_TOPK=merge_tile_topk,
                BYPASS_KV_L1=bypass_l1,
                POST_DOT_NORMALIZE=post_dot_normalize,
                POST_PV_NORMALIZE=post_pv_normalize,
                num_warps=warps,
                num_stages=stages,
                waves_per_eu=waves,
            )

        candidate_tile = triton.next_power_of_2(max(16, candidate_groups * 8))

        def reduce_candidates() -> None:
            _reduce_materialized_state_top8_kernel[(batch * q_heads,)](
                segment_buffers["route_candidate_scores"],
                segment_buffers["route_candidate_indices"],
                segment_buffers["route_top_scores"],
                segment_buffers["route_top_slots"],
                candidate_groups,
                MAX_TILES=max_segments,
                CANDIDATE_TILE=candidate_tile,
                num_warps=2,
                waves_per_eu=1,
            )

        def reduce_output() -> None:
            _reduce_decode_route_coarse_vector_kernel[(batch * q_heads,)](
                segment_buffers["route_group_out"],
                segment_buffers["route_group_lse"],
                segment_buffers["coarse_out"],
                segment_buffers["coarse_lse"],
                segments,
                HEAD_DIM=dim,
                MAX_SEGMENTS=max_segments,
                SEGMENT_BLOCK=triton.next_power_of_2(segments),
                num_warps=2,
                waves_per_eu=1,
            )

        def reduce_combined() -> None:
            _reduce_decode_route_coarse_vector_topk_kernel[(batch * q_heads,)](
                segment_buffers["route_candidate_scores"],
                segment_buffers["route_candidate_indices"],
                segment_buffers["route_group_out"],
                segment_buffers["route_group_lse"],
                segment_buffers["route_top_slots"],
                segment_buffers["route_top_scores"],
                segment_buffers["coarse_out"],
                segment_buffers["coarse_lse"],
                segments,
                candidate_groups,
                HEAD_DIM=dim,
                ROUTE_COUNT=args.routes,
                OPEN_COUNT=args.routes,
                MAX_SEGMENTS=max_segments,
                CANDIDATE_BLOCK=candidate_tile,
                SEGMENT_BLOCK=triton.next_power_of_2(segments),
                APPLY_MASS_CUTOFF=False,
                LOG_MASS_FRACTION=0.0,
                num_warps=2,
                waves_per_eu=1,
            )

        def reduce_splitd(block_d: int) -> None:
            _reduce_decode_route_coarse_vector_topk_splitd_kernel[
                (batch * q_heads, triton.cdiv(dim, block_d))
            ](
                segment_buffers["route_candidate_scores"],
                segment_buffers["route_candidate_indices"],
                segment_buffers["route_group_out"],
                segment_buffers["route_group_lse"],
                segment_buffers["route_top_slots"],
                segment_buffers["route_top_scores"],
                segment_buffers["coarse_out"],
                segment_buffers["coarse_lse"],
                segments,
                candidate_groups,
                HEAD_DIM=dim,
                ROUTE_COUNT=args.routes,
                MAX_SEGMENTS=max_segments,
                CANDIDATE_BLOCK=candidate_tile,
                SEGMENT_BLOCK=triton.next_power_of_2(segments),
                BLOCK_D=block_d,
                num_warps=2,
                waves_per_eu=1,
            )

        def complete() -> None:
            route_segments()
            reduce_candidates()
            reduce_output()

        def combined_complete() -> None:
            route_segments()
            reduce_combined()

        def splitd_complete(block_d: int) -> None:
            route_segments()
            reduce_splitd(block_d)

        complete()
        torch.cuda.synchronize()
        actual_slots = segment_buffers["route_top_slots"].reshape(
            batch, q_heads, args.routes
        )
        reference_slots = old_top_slots.reshape(batch, q_heads, args.routes)
        top8_fraction = float(
            (
                actual_slots.sort(dim=-1).values
                == reference_slots.sort(dim=-1).values
            )
            .all(dim=-1)
            .float()
            .mean()
            .item()
        )
        resplit_top8_fraction = float(
            (
                actual_slots.sort(dim=-1).values
                == actual_top_slots.sort(dim=-1).values
            )
            .all(dim=-1)
            .float()
            .mean()
            .item()
        )
        output_error = float(
            (segment_buffers["coarse_out"] - old_coarse_out).abs().max().item()
        )
        lse_error = float(
            (segment_buffers["coarse_lse"] - old_coarse_lse).abs().max().item()
        )
        return {
            "segments": segments,
            "candidate_groups": candidate_groups,
            "group_ms": time(route_segments),
            "candidate_reduce_ms": time(reduce_candidates),
            "output_reduce_ms": time(reduce_output),
            "total_ms": time(complete),
            "combined_reduce_ms": time(reduce_combined),
            "combined_total_ms": time(combined_complete),
            "splitd_reduce_ms": {
                str(block_d): time(lambda block_d=block_d: reduce_splitd(block_d))
                for block_d in (16, 32, 64)
                if block_d <= dim
            },
            "splitd_total_ms": {
                str(block_d): time(
                    lambda block_d=block_d: splitd_complete(block_d)
                )
                for block_d in (16, 32, 64)
                if block_d <= dim
            },
            "top8_set_fraction": top8_fraction,
            "resplit_top8_set_fraction": resplit_top8_fraction,
            "coarse_output_max_abs": output_error,
            "coarse_lse_max_abs": lse_error,
            "exact_pv_max_abs": float(
                (segment_buffers["coarse_out"] - exact_pv).abs().max().item()
            ),
            "reference_lse_max_abs": float(
                (segment_buffers["coarse_lse"] - reference_lse).abs().max().item()
            ),
        }

    def production_resplit(
        timing_events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ]
        | None = None,
        *,
        fuse_tile_topk_lse: bool = False,
        fuse_reduce_topk_lse: bool = False,
        fuse_normalized_pv: bool = False,
        pv_splits: int | None = None,
        pv_block_n: int = 128,
        pv_block_d: int = 32,
        pv_num_warps: int = 2,
    ) -> None:
        materialized_state_route_gqa(
            q,
            sum_k,
            sum_v,
            counts,
            cache_indices,
            buffers,
            state_len=state_len,
            kv_group_size=gqa,
            scale=dim**-0.5,
            waves_per_eu=1,
            fuse_tile_topk_lse=fuse_tile_topk_lse,
            fuse_reduce_topk_lse=fuse_reduce_topk_lse,
            fuse_normalized_pv=fuse_normalized_pv,
            pv_splits=pv_splits,
            pv_block_n=pv_block_n,
            pv_block_d=pv_block_d,
            pv_num_warps=pv_num_warps,
            timing_events=timing_events,
        )

    production_resplit()
    score_reference = buffers["route_state_scores"][..., 0, :state_len].float()
    reference_top_slots = torch.topk(
        score_reference, args.routes, dim=-1, sorted=False
    ).indices
    actual_top_slots = buffers["route_top_slots"][..., 0, :]
    resplit_top8_exact = bool(
        (
            reference_top_slots.sort(dim=-1).values
            == actual_top_slots.sort(dim=-1).values
        )
        .all()
        .item()
    )
    if not resplit_top8_exact:
        raise AssertionError("production re-split top-8 differs from its score table")
    reference_lse = torch.logsumexp(score_reference, dim=-1)
    resplit_lse_max_abs = float(
        (buffers["coarse_lse"] - reference_lse).abs().max().item()
    )
    reference_pv = torch.bmm(
        buffers["route_state_probabilities"][..., 0, :].reshape(
            batch * kv_heads, gqa, state_len
        ),
        sum_v.reshape(batch * kv_heads, state_len, dim).to(
            buffers["route_state_probabilities"].dtype
        ),
        out_dtype=torch.float32,
    ).reshape(batch, q_heads, dim)
    resplit_pv_max_abs = float(
        (buffers["coarse_out"] - reference_pv).abs().max().item()
    )
    exact_probability = (
        torch.softmax(score_reference, dim=-1)
        / counts[..., 0]
        .unsqueeze(2)
        .expand(batch, kv_heads, gqa, state_len)
        .reshape(batch, q_heads, state_len)
    )
    exact_pv = torch.bmm(
        exact_probability.reshape(batch * kv_heads, gqa, state_len),
        sum_v.float().reshape(batch * kv_heads, state_len, dim),
    ).reshape(batch, q_heads, dim)
    resplit_exact_pv_max_abs = float(
        (buffers["coarse_out"] - exact_pv).abs().max().item()
    )
    old_exact_pv_max_abs = float(
        (old_coarse_out - exact_pv).abs().max().item()
    )
    old_top8_set_fraction = float(
        (
            old_top_slots[..., 0, :].sort(dim=-1).values
            == actual_top_slots.sort(dim=-1).values
        )
        .all(dim=-1)
        .float()
        .mean()
        .item()
    )
    old_coarse_out_max_abs = float(
        (old_coarse_out - buffers["coarse_out"]).abs().max().item()
    )
    old_coarse_lse_max_abs = float(
        (old_coarse_lse - buffers["coarse_lse"]).abs().max().item()
    )
    resplit_top_slots = actual_top_slots.detach().clone()
    resplit_coarse_out = buffers["coarse_out"].detach().clone()
    resplit_coarse_lse = buffers["coarse_lse"].detach().clone()

    grouped_q = q[:, :, 0].reshape(batch * kv_heads, gqa, dim)
    grouped_k = mean_k.reshape(batch * kv_heads, state_len, dim)
    score_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.score_dtype]
    score_q = grouped_q if score_dtype == torch.float32 else grouped_q.to(score_dtype)
    score_k = grouped_k if score_dtype == torch.float32 else grouped_k.to(score_dtype)
    dense_scores = torch.empty(
        batch * kv_heads, gqa, state_len, dtype=score_dtype, device=device
    )
    dense_probabilities = torch.empty(
        batch * kv_heads, gqa, state_len, dtype=torch.bfloat16, device=device
    )
    dense_output = torch.empty(
        batch * kv_heads, gqa, dim, dtype=torch.float32, device=device
    )
    grouped_v = mean_v.reshape(batch * kv_heads, state_len, dim)
    top_values = torch.empty(
        batch * kv_heads, gqa, args.routes, dtype=score_dtype, device=device
    )
    top_indices = torch.empty(
        batch * kv_heads, gqa, args.routes, dtype=torch.int64, device=device
    )

    def dense_qk() -> None:
        if score_dtype == torch.float32:
            torch.bmm(
                grouped_q,
                grouped_k.transpose(1, 2),
                out_dtype=torch.float32,
                out=dense_scores,
            )
        else:
            # torch.bmm cannot request an FP16 output from BF16 operands.  For
            # timing, FP16 operands model a custom BF16-MFMA kernel that rounds
            # its FP32 accumulator only at the score-table store.  The actual
            # route-set diagnostic casts production FP32 scores and therefore
            # does not rely on this approximation.
            torch.bmm(
                score_q,
                score_k.transpose(1, 2),
                out=dense_scores,
            )
        dense_scores.mul_(dim**-0.5)

    def dense_topk() -> None:
        torch.topk(
            dense_scores,
            args.routes,
            dim=-1,
            sorted=True,
            out=(top_values, top_indices),
        )

    def dense_softmax() -> None:
        torch.softmax(
            dense_scores,
            dim=-1,
            dtype=torch.bfloat16,
            out=dense_probabilities,
        )

    def dense_pv() -> None:
        torch.bmm(
            dense_probabilities,
            grouped_v,
            out_dtype=torch.float32,
            out=dense_output,
        )

    def benchmark_score_table_topk(
        block_n: int, num_warps: int
    ) -> dict[str, float]:
        rows = batch * kv_heads * gqa
        tiles = triton.cdiv(state_len, block_n)
        candidate_count = tiles * 8
        candidate_block = triton.next_power_of_2(candidate_count)
        candidate_scores = torch.empty(
            rows, tiles, 8, dtype=torch.float16, device=device
        )
        candidate_indices = torch.empty(
            rows, tiles, 8, dtype=torch.int32, device=device
        )
        selected_scores = torch.empty(
            rows, 8, dtype=torch.float16, device=device
        )
        selected_indices = torch.empty(
            rows, 8, dtype=torch.int64, device=device
        )

        def tile_top8() -> None:
            _score_table_tile_top8_kernel[(rows, tiles)](
                dense_scores,
                candidate_scores,
                candidate_indices,
                state_len,
                TILES=tiles,
                BLOCK_N=block_n,
                num_warps=num_warps,
                waves_per_eu=1,
            )

        def reduce_top8() -> None:
            _score_table_reduce_top8_kernel[(rows,)](
                candidate_scores,
                candidate_indices,
                selected_scores,
                selected_indices,
                candidate_count,
                CANDIDATE_BLOCK=candidate_block,
                num_warps=2,
                waves_per_eu=1,
            )

        def total_top8() -> None:
            tile_top8()
            reduce_top8()

        def route_pipeline() -> None:
            dense_qk()
            tile_top8()
            reduce_top8()
            dense_softmax()
            dense_pv()

        total_top8()
        reference = top_indices.reshape(rows, 8).sort(dim=-1).values
        actual = selected_indices.sort(dim=-1).values
        if not bool((actual == reference).all().item()):
            mismatch = int((actual != reference).any(dim=-1).sum().item())
            raise AssertionError(
                f"score-table top-8 differs from torch.topk in {mismatch} rows"
            )
        tile_ms = time(tile_top8)
        reduce_ms = time(reduce_top8)
        total_ms = time(total_top8)
        return {
            "tile_top8_ms": tile_ms,
            "global_reduce_ms": reduce_ms,
            "sum_of_isolated_ms": tile_ms + reduce_ms,
            "combined_ms": total_ms,
            "full_route_pipeline_ms": time(route_pipeline),
        }

    dense_qk()
    dense_topk()
    dense_softmax()
    dense_pv()
    # The installed unified-attention specialization exceeds LDS capacity at
    # D=512.  Keep the remaining D=512 pieces measurable instead of replacing
    # the production backend with an unrelated fallback.
    aiter_import_error = None
    if dim <= 256:
        try:
            coarse_attention = aiter_attention(q, mean_k, mean_v)
        except ModuleNotFoundError as error:
            coarse_attention = None
            aiter_import_error = str(error)
    else:
        coarse_attention = None

    page_mean_k = randn((batch, kv_heads, page_len, dim))
    page_mean_v = randn((batch, kv_heads, page_len, dim))
    page_counts = torch.full(
        (batch, kv_heads, page_len),
        args.page_size,
        dtype=torch.int32,
        device=device,
    )
    if live_page_len < page_len:
        page_counts[:, :, live_page_len:].zero_()
    page_sum_k = (page_mean_k.float() * args.page_size).to(torch.bfloat16)
    page_scores = torch.empty(
        batch, q_heads, 1, page_len, dtype=torch.float32, device=device
    )
    page_top_values = torch.empty(
        batch, q_heads, 1, args.routes, dtype=torch.float32, device=device
    )
    page_top_indices = torch.empty(
        batch, q_heads, 1, args.routes, dtype=torch.int64, device=device
    )
    page_probabilities = torch.empty(
        batch * kv_heads,
        gqa,
        page_len,
        dtype=torch.bfloat16,
        device=device,
    )
    page_output = torch.empty(
        batch * kv_heads, gqa, dim, dtype=torch.float32, device=device
    )
    grouped_page_v = page_mean_v.reshape(batch * kv_heads, page_len, dim)

    def page_qk() -> None:
        materialize_page_summary_scores_gqa(
            q,
            page_sum_k,
            page_counts,
            cache_indices=cache_indices,
            kv_group_size=gqa,
            scale=dim**-0.5,
            output=page_scores,
            page_block_n=32,
            num_warps=2 if dim == 128 else 4,
        )

    def page_qk_variant(page_block_n: int, num_warps: int) -> None:
        materialize_page_summary_scores_gqa(
            q,
            page_sum_k,
            page_counts,
            cache_indices=cache_indices,
            kv_group_size=gqa,
            scale=dim**-0.5,
            output=page_scores,
            page_block_n=page_block_n,
            num_warps=num_warps,
        )

    def page_topk() -> None:
        torch.topk(
            page_scores,
            args.routes,
            dim=-1,
            sorted=True,
            out=(page_top_values, page_top_indices),
        )

    def page_softmax() -> None:
        torch.softmax(
            page_scores.reshape(batch * kv_heads, gqa, page_len),
            dim=-1,
            dtype=torch.bfloat16,
            out=page_probabilities,
        )

    def page_pv() -> None:
        torch.bmm(
            page_probabilities,
            grouped_page_v,
            out_dtype=torch.float32,
            out=page_output,
        )

    page_qk()
    page_topk()
    page_softmax()
    page_pv()
    if dim <= 256 and aiter_import_error is None:
        page_attention = aiter_attention(q, page_mean_k, page_mean_v)
    else:
        page_attention = None
    exact_len = args.routes * args.page_size
    exact_attention = (
        aiter_attention(
            q,
            page_mean_k[:, :, :exact_len].contiguous(),
            page_mean_v[:, :, :exact_len].contiguous(),
        )
        if dim <= 256 and aiter_import_error is None
        else None
    )

    time = lambda call: elapsed(call, args.warmup, args.repeats)  # noqa: E731
    result = {
        "route_groups_sum_count_ms": time(lambda: route_groups(sum_k, sum_v, counts)),
        # This still executes division/log in the old kernel.  It is a
        # conservative control for storing means, not a dedicated mean kernel.
        "route_groups_cached_mean_control_ms": time(
            lambda: route_groups(mean_k, mean_v, ones)
        ),
        "route_reduce_ms": time(route_reduce),
        "production_resplit_route_ms": time(production_resplit),
        "production_resplit_top8_exact": resplit_top8_exact,
        "production_resplit_lse_reference_max_abs": resplit_lse_max_abs,
        "production_resplit_pv_reference_max_abs": resplit_pv_max_abs,
        "production_resplit_exact_pv_max_abs": resplit_exact_pv_max_abs,
        "current_route_exact_pv_max_abs": old_exact_pv_max_abs,
        "production_resplit_vs_old_top8_set_fraction": old_top8_set_fraction,
        "production_resplit_vs_old_coarse_out_max_abs": old_coarse_out_max_abs,
        "production_resplit_vs_old_coarse_lse_max_abs": old_coarse_lse_max_abs,
        "aiter_coarse_attention_ms": (
            time(coarse_attention) if coarse_attention is not None else None
        ),
        "dense_centroid_qk_ms": time(dense_qk),
        "dense_centroid_topk_ms": time(dense_topk),
        "dense_centroid_softmax_ms": time(dense_softmax),
        "dense_centroid_pv_ms": time(dense_pv),
        "dense_page_qk_ms": time(page_qk),
        "dense_page_global_topk_ms": time(page_topk),
        "dense_page_softmax_ms": time(page_softmax),
        "dense_page_pv_ms": time(page_pv),
        "aiter_page_attention_ms": (
            time(page_attention) if page_attention is not None else None
        ),
        "aiter_fixed_exact_pages_ms": (
            time(exact_attention) if exact_attention is not None else None
        ),
    }
    if args.sweep_production_score:
        score_table = buffers["route_state_scores"]

        def production_score(
            block_n: int,
            warps: int,
            *,
            stages: int | None = None,
            waves_per_eu: int = 1,
            bypass_k_l1: bool = False,
        ) -> None:
            launch = {
                "num_warps": warps,
                "waves_per_eu": waves_per_eu,
            }
            if stages is not None:
                launch["num_stages"] = stages
            _materialize_state_summary_scores_gqa_kernel[
                (batch * kv_heads, triton.cdiv(state_len, block_n))
            ](
                q,
                sum_k,
                counts,
                cache_indices,
                score_table,
                sum_k.stride(0),
                sum_k.stride(1),
                sum_k.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                HEAD_DIM=dim,
                HEAD_BLOCK_DIM=triton.next_power_of_2(dim),
                SCORE_BLOCK_N=block_n,
                SCALE=dim**-0.5,
                BYPASS_K_L1=bypass_k_l1,
                **launch,
            )

        score_sweep = {}
        for block_n in (16, 32, 64, 128):
            for warps in (1, 2, 4, 8):
                name = f"n{block_n}_w{warps}"
                try:
                    production_score(block_n, warps)
                    score_sweep[name] = {"score_ms": time(
                        lambda block_n=block_n, warps=warps: production_score(
                            block_n, warps
                        )
                    )}
                except Exception as error:
                    score_sweep[name] = {"error": f"{type(error).__name__}: {error}"}
        result["production_score_sweep"] = score_sweep
        aiter_config_sweep = {}
        for block_n in (16, 32, 64, 128):
            for stages, waves_per_eu, bypass_k_l1 in (
                (1, 1, False),
                (2, 1, False),
                (3, 2, False),
                (3, 2, True),
            ):
                name = (
                    f"n{block_n}_w2_s{stages}_we{waves_per_eu}"
                    f"_cg{int(bypass_k_l1)}"
                )
                try:
                    production_score(
                        block_n,
                        2,
                        stages=stages,
                        waves_per_eu=waves_per_eu,
                        bypass_k_l1=bypass_k_l1,
                    )
                    aiter_config_sweep[name] = {"score_ms": time(
                        lambda block_n=block_n,
                        stages=stages,
                        waves_per_eu=waves_per_eu,
                        bypass_k_l1=bypass_k_l1: production_score(
                            block_n,
                            2,
                            stages=stages,
                            waves_per_eu=waves_per_eu,
                            bypass_k_l1=bypass_k_l1,
                        )
                    )}
                except Exception as error:
                    aiter_config_sweep[name] = {
                        "error": f"{type(error).__name__}: {error}"
                    }
        result["production_score_aiter_config_sweep"] = aiter_config_sweep
    if args.sweep_fusion:
        fusion_sweep = {}
        for name, fuse_tile, fuse_reduce in (
            ("none", False, False),
            ("tile", True, False),
            ("reduce", False, True),
            ("both", True, True),
        ):
            call = lambda fuse_tile=fuse_tile, fuse_reduce=fuse_reduce: (
                production_resplit(
                    fuse_tile_topk_lse=fuse_tile,
                    fuse_reduce_topk_lse=fuse_reduce,
                )
            )
            call()
            variant_top_slots = buffers["route_top_slots"][..., 0, :]
            fusion_sweep[name] = {
                "route_ms": time(call),
                "top8_exact": bool(
                    (
                        variant_top_slots.sort(dim=-1).values
                        == resplit_top_slots.sort(dim=-1).values
                    )
                    .all()
                    .item()
                ),
                "coarse_lse_max_abs": float(
                    (buffers["coarse_lse"] - resplit_coarse_lse)
                    .abs()
                    .max()
                    .item()
                ),
                "coarse_out_max_abs": float(
                    (buffers["coarse_out"] - resplit_coarse_out)
                    .abs()
                    .max()
                    .item()
                ),
            }
        result["production_resplit_fusion_sweep"] = fusion_sweep
    if args.sweep_route_pv_warps:
        result["production_resplit_pv_warps_sweep"] = {
            f"w{warps}": time(
                lambda warps=warps: production_resplit(pv_num_warps=warps)
            )
            for warps in (1, 2, 4)
        }
    if args.sweep_route_pv_splits:
        result["production_fused_pv_split_sweep"] = {
            f"s{splits}_w{warps}": time(
                lambda splits=splits, warps=warps: production_resplit(
                    fuse_tile_topk_lse=True,
                    fuse_reduce_topk_lse=True,
                    pv_splits=splits,
                    pv_num_warps=warps,
                )
            )
            for splits in (1, 2, 4, 8)
            for warps in (1, 2, 4)
        }
    if args.sweep_route_pv_tiles:
        tile_variants = (
            (8, 128, 32, 2),
            (2, 64, 64, 1),
            (4, 64, 64, 1),
            (8, 64, 64, 1),
            (2, 128, 64, 1),
            (4, 128, 64, 1),
            (8, 128, 64, 1),
            (2, 64, 128, 2),
            (4, 64, 128, 2),
            (8, 64, 128, 2),
            (2, 128, 128, 4),
            (4, 128, 128, 4),
            (8, 128, 128, 4),
        )
        result["production_fused_pv_tile_sweep"] = {
            f"s{splits}_n{block_n}_d{block_d}_w{warps}": time(
                lambda splits=splits, block_n=block_n, block_d=block_d, warps=warps: (
                    production_resplit(
                        fuse_tile_topk_lse=True,
                        fuse_reduce_topk_lse=True,
                        pv_splits=splits,
                        pv_block_n=block_n,
                        pv_block_d=block_d,
                        pv_num_warps=warps,
                    )
                )
            )
            for splits, block_n, block_d, warps in tile_variants
            if block_d <= dim
        }
    if args.sweep_int8_pv:
        int8_block_n = 128
        state_blocks = triton.cdiv(state_len, int8_block_n)
        int8_partials = {
            splits: torch.empty(
                batch,
                q_heads,
                splits,
                dim,
                dtype=torch.float32,
                device=device,
            )
            for splits in (2, 4, 8)
        }
        state_v_codes = torch.empty_like(sum_v, dtype=torch.int8)
        state_v_scales = torch.empty(
            batch,
            kv_heads,
            state_blocks,
            dim,
            dtype=sum_v.dtype,
            device=device,
        )
        _quantize_state_mean_values_int8_kernel[
            (batch * kv_heads * state_blocks, triton.cdiv(dim, 32))
        ](
            sum_v,
            counts,
            state_v_codes,
            state_v_scales,
            STATE_V_BATCH_STRIDE=sum_v.stride(0),
            STATE_V_HEAD_STRIDE=sum_v.stride(1),
            STATE_V_TOKEN_STRIDE=sum_v.stride(2),
            COUNT_BATCH_STRIDE=counts.stride(0),
            COUNT_HEAD_STRIDE=counts.stride(1),
            COUNT_TOKEN_STRIDE=counts.stride(2),
            KV_HEADS=kv_heads,
            state_len=state_len,
            state_blocks=state_blocks,
            VALUE_DIM=dim,
            BLOCK_N=int8_block_n,
            BLOCK_D=32,
            num_warps=2,
            waves_per_eu=1,
        )
        ordinary_probabilities = torch.empty_like(
            buffers["route_state_probabilities"]
        )
        ordinary_probabilities[..., 0, :].copy_(
            (
                buffers["route_state_probabilities"][..., 0, :].reshape(
                    batch, kv_heads, gqa, state_len
                )
                * counts[..., 0].unsqueeze(2)
            ).reshape(batch, q_heads, state_len)
        )

        def int8_pv_split(splits: int, block_d: int, warps: int) -> None:
            _materialized_state_pv_int8_split_kernel[
                (batch * kv_heads, triton.cdiv(dim, block_d), splits)
            ](
                ordinary_probabilities,
                state_v_codes,
                state_v_scales,
                cache_indices,
                int8_partials[splits],
                state_v_codes.stride(0),
                state_v_codes.stride(1),
                state_v_codes.stride(2),
                state_v_scales.stride(0),
                state_v_scales.stride(1),
                state_v_scales.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                HEAD_DIM=dim,
                PV_SPLITS=splits,
                BLOCK_N=int8_block_n,
                BLOCK_D=block_d,
                num_warps=warps,
                waves_per_eu=1,
            )

        def int8_pv_reduce(splits: int, block_d: int) -> None:
            _reduce_materialized_state_pv_kernel[
                (batch * q_heads, triton.cdiv(dim, block_d))
            ](
                int8_partials[splits],
                buffers["coarse_out"],
                QUERY_HEADS=q_heads,
                HEAD_DIM=dim,
                PV_SPLITS=splits,
                BLOCK_D=block_d,
                num_warps=1,
                waves_per_eu=1,
            )

        def dequant_pv_split(splits: int, block_d: int, warps: int) -> None:
            _materialized_state_pv_dequant_split_kernel[
                (batch * kv_heads, triton.cdiv(dim, block_d), splits)
            ](
                ordinary_probabilities,
                state_v_codes,
                state_v_scales,
                cache_indices,
                int8_partials[splits],
                state_v_codes.stride(0),
                state_v_codes.stride(1),
                state_v_codes.stride(2),
                state_v_scales.stride(0),
                state_v_scales.stride(1),
                state_v_scales.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                HEAD_DIM=dim,
                PV_SPLITS=splits,
                BLOCK_N=int8_block_n,
                BLOCK_D=block_d,
                num_warps=warps,
                waves_per_eu=1,
            )

        int8_records = {}
        for splits, block_d, warps in (
            (2, 32, 1),
            (2, 32, 2),
            (4, 32, 1),
            (4, 32, 2),
            (8, 32, 1),
            (8, 32, 2),
            (2, 64, 1),
            (4, 64, 1),
            (8, 64, 1),
        ):
            split_call = lambda splits=splits, block_d=block_d, warps=warps: (
                int8_pv_split(splits, block_d, warps)
            )
            reduce_call = lambda splits=splits, block_d=block_d: int8_pv_reduce(
                splits, block_d
            )
            total_call = lambda: (split_call(), reduce_call())
            total_call()
            int8_records[f"s{splits}_d{block_d}_w{warps}"] = {
                "split_ms": time(split_call),
                "reduce_ms": time(reduce_call),
                "total_ms": time(total_call),
                "exact_pv_max_abs": float(
                    (buffers["coarse_out"] - exact_pv).abs().max().item()
                ),
            }
        result["production_int8_pv_sweep"] = int8_records
        dequant_records = {}
        for splits, block_d, warps in (
            (2, 32, 1),
            (2, 32, 2),
            (4, 32, 1),
            (4, 32, 2),
            (8, 32, 1),
            (8, 32, 2),
            (2, 64, 1),
            (4, 64, 1),
            (8, 64, 1),
        ):
            split_call = lambda splits=splits, block_d=block_d, warps=warps: (
                dequant_pv_split(splits, block_d, warps)
            )
            reduce_call = lambda splits=splits, block_d=block_d: int8_pv_reduce(
                splits, block_d
            )
            total_call = lambda: (split_call(), reduce_call())
            total_call()
            dequant_records[f"s{splits}_d{block_d}_w{warps}"] = {
                "split_ms": time(split_call),
                "reduce_ms": time(reduce_call),
                "total_ms": time(total_call),
                "exact_pv_max_abs": float(
                    (buffers["coarse_out"] - exact_pv).abs().max().item()
                ),
            }
        result["production_dequant_pv_sweep"] = dequant_records
    if args.sweep_normalized_pv:
        def baseline_softmax() -> None:
            _materialized_state_softmax_kernel[
                (batch * q_heads, triton.cdiv(state_len, 128))
            ](
                buffers["route_state_scores"],
                buffers["coarse_lse"],
                counts,
                cache_indices,
                buffers["route_state_probabilities"],
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                BLOCK_N=128,
                num_warps=1,
                waves_per_eu=1,
            )

        def baseline_pv_split() -> None:
            _materialized_state_pv_split_kernel[
                (batch * kv_heads, triton.cdiv(dim, 32), 8)
            ](
                buffers["route_state_probabilities"],
                sum_v,
                cache_indices,
                buffers["partial_out"],
                sum_v.stride(0),
                sum_v.stride(1),
                sum_v.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                HEAD_DIM=dim,
                PV_SPLITS=8,
                BLOCK_N=128,
                BLOCK_D=32,
                num_warps=2,
                waves_per_eu=1,
            )

        def reduce_pv(partials: torch.Tensor, splits: int) -> None:
            _reduce_materialized_state_pv_kernel[
                (batch * q_heads, triton.cdiv(dim, 32))
            ](
                partials,
                buffers["coarse_out"],
                QUERY_HEADS=q_heads,
                HEAD_DIM=dim,
                PV_SPLITS=splits,
                BLOCK_D=32,
                num_warps=1,
                waves_per_eu=1,
            )

        def baseline_reduce() -> None:
            reduce_pv(buffers["partial_out"], 8)

        def baseline_total() -> None:
            baseline_softmax()
            baseline_pv_split()
            baseline_reduce()

        baseline_total()
        normalized_pv_records = {
            "baseline": {
                "softmax_ms": time(baseline_softmax),
                "pv_split_ms": time(baseline_pv_split),
                "pv_reduce_ms": time(baseline_reduce),
                "total_ms": time(baseline_total),
                "exact_pv_max_abs": float(
                    (buffers["coarse_out"] - exact_pv).abs().max().item()
                ),
            }
        }
        triton_partials = {
            splits: torch.empty(
                batch,
                q_heads,
                splits,
                dim,
                dtype=torch.float32,
                device=device,
            )
            for splits in (4, 8)
        }
        for splits, block_n, block_d, warps in (
            (4, 64, 32, 1),
            (8, 64, 32, 1),
            (4, 128, 32, 2),
            (8, 128, 32, 2),
            (4, 64, 64, 1),
            (8, 64, 64, 1),
            (4, 128, 64, 2),
            (8, 128, 64, 2),
            (4, 64, 128, 2),
            (8, 64, 128, 2),
            (4, 128, 128, 4),
            (8, 128, 128, 4),
        ):
            if block_d > dim:
                continue
            partials = triton_partials[splits]

            def normalized_call(
                splits=splits,
                block_n=block_n,
                block_d=block_d,
                warps=warps,
                partials=partials,
            ) -> None:
                _materialized_state_normalized_pv_split_kernel[
                    (
                        batch * kv_heads,
                        triton.cdiv(dim, block_d),
                        splits,
                    )
                ](
                    buffers["route_state_scores"],
                    buffers["coarse_lse"],
                    counts,
                    cache_indices,
                    sum_v,
                    partials,
                    sum_v.stride(0),
                    sum_v.stride(1),
                    sum_v.stride(2),
                    counts.stride(0),
                    counts.stride(1),
                    counts.stride(2),
                    state_len,
                    QUERY_HEADS=q_heads,
                    KV_HEADS=kv_heads,
                    KV_GROUP_SIZE=gqa,
                    STATE_CAPACITY=state_len,
                    HEAD_DIM=dim,
                    PV_SPLITS=splits,
                    BLOCK_N=block_n,
                    BLOCK_D=block_d,
                    num_warps=warps,
                    waves_per_eu=1,
                )

            reduce_call = lambda splits=splits, partials=partials: reduce_pv(
                partials, splits
            )
            total_call = lambda normalized_call=normalized_call, reduce_call=reduce_call: (
                normalized_call(),
                reduce_call(),
            )
            total_call()
            name = f"triton_s{splits}_n{block_n}_d{block_d}_w{warps}"
            normalized_pv_records[name] = {
                "normalized_pv_ms": time(normalized_call),
                "pv_reduce_ms": time(reduce_call),
                "total_ms": time(total_call),
                "exact_pv_max_abs": float(
                    (buffers["coarse_out"] - exact_pv).abs().max().item()
                ),
            }
        result["normalized_pv_sweep"] = normalized_pv_records
        fused_route_call = lambda: production_resplit(  # noqa: E731
            fuse_tile_topk_lse=True,
            fuse_reduce_topk_lse=True,
            fuse_normalized_pv=True,
        )
        fused_route_call()
        result["normalized_pv_route"] = {
            "route_ms": time(fused_route_call),
            "top8_exact": bool(
                (
                    buffers["route_top_slots"][..., 0, :]
                    .sort(dim=-1)
                    .values
                    == resplit_top_slots.sort(dim=-1).values
                )
                .all()
                .item()
            ),
            "coarse_lse_max_abs": float(
                (buffers["coarse_lse"] - resplit_coarse_lse).abs().max().item()
            ),
            "exact_pv_max_abs": float(
                (buffers["coarse_out"] - exact_pv).abs().max().item()
            ),
        }
    if args.profile_fused_stages:
        from torch.profiler import ProfilerActivity, profile

        fused_call = lambda: production_resplit(  # noqa: E731
            fuse_tile_topk_lse=True,
            fuse_reduce_topk_lse=True,
        )
        for _ in range(args.warmup):
            fused_call()
        torch.cuda.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as stage_profile:
            for _ in range(args.profile_fused_stages):
                fused_call()
            torch.cuda.synchronize()
        stage_patterns = {
            "score_materialization": "_materialize_state_summary_scores_gqa_kernel",
            "tile_top8_lse": "_materialized_state_tile_top8_lse_kernel",
            "top8_lse_reduce": "_reduce_materialized_state_top8_lse_kernel",
            "softmax": "_materialized_state_softmax_kernel",
            "pv_split": "_materialized_state_pv_split_kernel",
            "pv_reduce": "_reduce_materialized_state_pv_kernel",
        }
        profile_rows = {}
        for stage_name, pattern in stage_patterns.items():
            matching = [
                event
                for event in stage_profile.key_averages()
                if pattern in event.key
            ]
            device_us = sum(
                float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                )
                for event in matching
            )
            profile_rows[stage_name] = {
                "milliseconds_per_route": device_us
                / (1000.0 * args.profile_fused_stages),
                "calls": sum(int(event.count) for event in matching),
                "keys": [event.key for event in matching],
            }
        result["production_fused_profile"] = {
            "profiled_routes": args.profile_fused_stages,
            "whole_route_ms": time(fused_call),
            "stages": profile_rows,
        }
    stage_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = {}
    for _ in range(args.warmup):
        production_resplit()
    for _ in range(args.repeats):
        production_resplit(stage_events)
    torch.cuda.synchronize()
    result["production_resplit_stages_ms"] = {
        name: sum(float(begin.elapsed_time(end)) for begin, end in pairs)
        / args.repeats
        for name, pairs in stage_events.items()
        if not name.endswith(f"_b{batch}")
    }
    if args.sweep_production_pv or args.sweep_split_pv:
        sweep_partials = {
            splits: torch.empty(
                batch,
                q_heads,
                splits,
                dim,
                dtype=torch.float32,
                device=device,
            )
            for splits in (2, 4, 8, 16)
        }

        def production_pv(block_n: int, block_d: int, warps: int) -> None:
            _materialized_state_pv_kernel[
                (batch * kv_heads, triton.cdiv(dim, block_d))
            ](
                buffers["route_state_probabilities"],
                sum_v,
                cache_indices,
                buffers["coarse_out"],
                sum_v.stride(0),
                sum_v.stride(1),
                sum_v.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                HEAD_DIM=dim,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                num_warps=warps,
                waves_per_eu=1,
            )

        def production_split_pv(
            splits: int,
            block_n: int,
            block_d: int,
            warps: int,
        ) -> None:
            _materialized_state_pv_split_kernel[
                (batch * kv_heads, triton.cdiv(dim, block_d), splits)
            ](
                buffers["route_state_probabilities"],
                sum_v,
                cache_indices,
                sweep_partials[splits],
                sum_v.stride(0),
                sum_v.stride(1),
                sum_v.stride(2),
                state_len,
                QUERY_HEADS=q_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=gqa,
                STATE_CAPACITY=state_len,
                HEAD_DIM=dim,
                PV_SPLITS=splits,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                num_warps=warps,
                waves_per_eu=1,
            )

        def production_split_reduce(splits: int, block_d: int) -> None:
            _reduce_materialized_state_pv_kernel[
                (batch * q_heads, triton.cdiv(dim, block_d))
            ](
                sweep_partials[splits],
                buffers["coarse_out"],
                QUERY_HEADS=q_heads,
                HEAD_DIM=dim,
                PV_SPLITS=splits,
                BLOCK_D=block_d,
                num_warps=1,
                waves_per_eu=1,
            )

        def split_pv_record(
            splits: int,
            block_n: int,
            block_d: int,
            warps: int,
        ) -> dict[str, float]:
            split_call = lambda: production_split_pv(
                splits, block_n, block_d, warps
            )
            reduce_call = lambda: production_split_reduce(splits, block_d)
            total_call = lambda: (split_call(), reduce_call())
            split_call()
            reduce_call()
            reference = buffers["coarse_out"].clone()
            production_split_pv(splits, block_n, block_d, warps)
            production_split_reduce(splits, block_d)
            maximum_error = float(
                (buffers["coarse_out"] - reference).abs().max().item()
            )
            return {
                "split_ms": time(split_call),
                "reduce_ms": time(reduce_call),
                "total_ms": time(total_call),
                "repeat_max_abs": maximum_error,
            }

        if args.sweep_production_pv:
            result["production_pv_sweep_ms"] = {
                f"n{block_n}_d{block_d}_w{warps}": time(
                    lambda block_n=block_n, block_d=block_d, warps=warps: (
                        production_pv(block_n, block_d, warps)
                    )
                )
                for block_n in (32, 64, 128)
                for block_d in (32, 64, 128)
                if block_d <= dim
                for warps in (2, 4, 8)
            }
        if args.sweep_split_pv:
            result["production_split_pv_count_sweep"] = {
                f"s{splits}_n128_d32_w2": split_pv_record(
                    splits, 128, 32, 2
                )
                for splits in (2, 4, 8, 16)
            }
            result["production_split_pv_tile_sweep"] = {
                f"s8_n{block_n}_d{block_d}_w{warps}": split_pv_record(
                    8, block_n, block_d, warps
                )
                for block_n in (64, 128, 256)
                for block_d in (32, 64)
                if block_d <= dim
                for warps in (1, 2, 4)
            }
    if args.sweep_page_score:
        result["dense_page_qk_sweep_ms"] = {
            f"n{block_n}_w{warps}": time(
                lambda block_n=block_n, warps=warps: page_qk_variant(
                    block_n, warps
                )
            )
            for block_n in (16, 32, 64, 128)
            for warps in (1, 2, 4)
        }
    if args.sweep_segment_route:
        result["segmented_route_sweep"] = {
            f"t{segment_tiles}_w{warps}_wave{waves}_stage{stages}_cg{int(bypass_l1)}_merge{int(merge_tile_topk)}_postqk{int(post_dot_normalize)}_postpv{int(post_pv_normalize)}": (
                segmented_route_record(
                    segment_tiles,
                    warps,
                    waves,
                    stages,
                    bypass_l1,
                    merge_tile_topk,
                    post_dot_normalize,
                    post_pv_normalize,
                )
            )
            for segment_tiles, warps, waves, stages, bypass_l1, merge_tile_topk, post_dot_normalize, post_pv_normalize in (
                (1, 2, 1, 3, False, True, False, False),
                (2, 2, 1, 3, False, True, False, False),
                (4, 2, 1, 3, False, True, False, False),
                (2, 2, 1, 3, False, True, False, True),
                (2, 2, 1, 3, False, True, True, False),
                (2, 2, 1, 3, False, True, True, True),
            )
        }
    if args.sweep_score_topk:
        if score_dtype != torch.float16:
            raise ValueError("score top-k sweep requires --score-dtype fp16")
        result["fp16_score_table_topk_sweep_ms"] = {
            f"n{block_n}_w{warps}": benchmark_score_table_topk(block_n, warps)
            for block_n in (128, 256)
            for warps in (1, 2, 4)
        }
    result["current_route_total_ms"] = (
        result["route_groups_sum_count_ms"] + result["route_reduce_ms"]
    )
    result["resplit_route_total_ms"] = (
        result["aiter_coarse_attention_ms"]
        + result["dense_centroid_qk_ms"]
        + result["dense_centroid_topk_ms"]
        if result["aiter_coarse_attention_ms"] is not None
        else None
    )
    # Keep score materialization, top-k, softmax, and PV independent so their
    # costs remain attributable before considering any later fusion.
    result["exposed_score_coarse_total_ms"] = (
        result["dense_centroid_qk_ms"]
        + result["dense_centroid_topk_ms"]
        + result["dense_centroid_softmax_ms"]
        + result["dense_centroid_pv_ms"]
    )
    result["flat_page_total_before_correction_ms"] = (
        result["dense_page_qk_ms"]
        + result["dense_page_global_topk_ms"]
        + result["aiter_page_attention_ms"]
        + result["aiter_fixed_exact_pages_ms"]
        if result["aiter_page_attention_ms"] is not None
        and result["aiter_fixed_exact_pages_ms"] is not None
        else None
    )
    result["flat_page_exposed_score_total_before_correction_ms"] = (
        result["dense_page_qk_ms"]
        + result["dense_page_global_topk_ms"]
        + result["dense_page_softmax_ms"]
        + result["dense_page_pv_ms"]
        + result["aiter_fixed_exact_pages_ms"]
        if result["aiter_fixed_exact_pages_ms"] is not None
        else None
    )
    return {
        "geometry": asdict(geometry),
        "batch_size": batch,
        "context_length": args.context_length,
        "state_length": state_len,
        "live_page_length": live_page_len,
        "page_capacity": page_len,
        "controlled_equal_count": None if args.random_counts else count,
        "random_counts": args.random_counts,
        "route_group_size": group_n,
        "score_dtype": args.score_dtype,
        "aiter_import_error": aiter_import_error,
        "times_ms": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=(*GEOMETRIES, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--page-capacity",
        type=int,
        help="Allocated page slots; the tail beyond T/page-size is left empty",
    )
    parser.add_argument(
        "--live-page-length",
        type=int,
        help="Override the number of nonempty page slots (for fragmentation controls)",
    )
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument(
        "--route-group-size",
        type=int,
        default=0,
        help="0 selects 64 for D=128 and 32 for D=256/D=512",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--random-counts", action="store_true")
    parser.add_argument(
        "--score-dtype", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--sweep-page-score", action="store_true")
    parser.add_argument("--sweep-segment-route", action="store_true")
    parser.add_argument("--sweep-score-topk", action="store_true")
    parser.add_argument("--sweep-production-pv", action="store_true")
    parser.add_argument("--sweep-split-pv", action="store_true")
    parser.add_argument("--sweep-route-pv-warps", action="store_true")
    parser.add_argument("--sweep-route-pv-splits", action="store_true")
    parser.add_argument("--sweep-route-pv-tiles", action="store_true")
    parser.add_argument("--sweep-production-score", action="store_true")
    parser.add_argument("--sweep-int8-pv", action="store_true")
    parser.add_argument("--sweep-normalized-pv", action="store_true")
    parser.add_argument("--sweep-fusion", action="store_true")
    parser.add_argument(
        "--profile-fused-stages",
        type=int,
        default=0,
        metavar="ROUTES",
        help="Profile this many fused routes and aggregate device time by stage",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile_fused_stages < 0:
        raise ValueError("--profile-fused-stages must be nonnegative")
    torch.cuda.set_device(0)
    selected = (
        GEOMETRIES.values()
        if args.geometry == "all"
        else (GEOMETRIES[args.geometry],)
    )
    records = []
    for geometry in selected:
        record = benchmark(geometry, args)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        torch.cuda.empty_cache()
    payload = {
        "device": torch.cuda.get_device_name(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
