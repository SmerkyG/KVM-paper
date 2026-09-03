#!/usr/bin/env python3
"""Compare the AITER-layout HIP QK control with production Triton routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

from model.kernels.gqa16_coarse_score import (
    gqa16_coarse_candidates,
    gqa16_coarse_score,
    reduce_route_top8,
)
from model.kernels.paged_leaf_attention import (
    _decode_route_coarse_gqa_segments_kernel,
    _materialize_state_summary_scores_gqa_kernel,
    _materialized_state_tile_top8_kernel,
    _reduce_decode_route_topk_kernel,
    new_fused_decode_buffers,
)


def time_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    batch = args.batch_size
    kv_heads = 2
    gqa = 16
    query_heads = kv_heads * gqa
    head_dim = 128
    state_len = args.state_length
    scale = head_dim**-0.5

    q = torch.randn(
        batch, query_heads, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    means = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=torch.bfloat16
    )
    counts = torch.randint(
        1, 257, (batch, kv_heads, state_len, 1), device=device, dtype=torch.int32
    ).float()
    state_k = (means.float() * counts).to(torch.bfloat16)
    # Values are only needed by the production score-only kernel signature.
    state_v = torch.zeros_like(state_k)
    cache_indices = torch.arange(batch, device=device, dtype=torch.int64)
    hip_scores = torch.empty(
        batch, query_heads, 1, state_len, device=device, dtype=torch.float32
    )
    triton_scores = torch.empty_like(hip_scores)

    buffers = new_fused_decode_buffers(
        q,
        splits=8,
        state_capacity=state_len,
        route_group_size=256,
        route_segment_tiles=4,
    )
    active_segments = triton.cdiv(state_len, 256)
    max_segments = int(buffers["route_group_lse"].size(2))
    hip_candidate_scores = torch.empty(
        batch * query_heads,
        active_segments,
        8,
        device=device,
        dtype=torch.float32,
    )
    hip_candidate_indices = torch.empty(
        batch * query_heads,
        active_segments,
        8,
        device=device,
        dtype=torch.int64,
    )
    hip_top_slots = torch.empty(
        batch, query_heads, 1, 8, device=device, dtype=torch.int64
    )
    hip_top_scores = torch.empty(
        batch, query_heads, 1, 8, device=device, dtype=torch.float32
    )

    def hip_qk() -> None:
        gqa16_coarse_score(
            q,
            state_k,
            counts,
            cache_indices,
            hip_scores,
            state_len=state_len,
            scale=scale,
        )

    def triton_materialized_qk() -> None:
        _materialize_state_summary_scores_gqa_kernel[
            (batch * kv_heads, triton.cdiv(state_len, 64))
        ](
            q,
            state_k,
            counts,
            cache_indices,
            triton_scores,
            state_k.stride(0),
            state_k.stride(1),
            state_k.stride(2),
            counts.stride(0),
            counts.stride(1),
            counts.stride(2),
            state_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=gqa,
            STATE_CAPACITY=state_len,
            HEAD_DIM=head_dim,
            HEAD_BLOCK_DIM=head_dim,
            SCORE_BLOCK_N=64,
            SCALE=scale,
            BYPASS_K_L1=True,
            num_warps=2,
            num_stages=3,
            waves_per_eu=2,
        )

    def hip_candidates() -> None:
        gqa16_coarse_candidates(
            q,
            state_k,
            counts,
            cache_indices,
            hip_candidate_scores,
            hip_candidate_indices,
            state_len=state_len,
            scale=scale,
        )

    def hip_reduce() -> None:
        reduce_route_top8(
            hip_candidate_scores,
            hip_candidate_indices,
            hip_top_slots,
            hip_top_scores,
            active_segments=active_segments,
        )

    def hip_fused_candidate_pipeline() -> None:
        hip_candidates()
        hip_reduce()

    def production_score_candidates() -> None:
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
            SCALE=scale,
            GROUP_N=64,
            SEGMENT_TILES=4,
            MAX_GROUPS=max_segments,
            PROTECTED_LEN=0,
            MAX_LEAF_TOKENS=0,
            MERGE_TILE_TOPK=True,
            BYPASS_KV_L1=False,
            POST_DOT_NORMALIZE=False,
            POST_PV_NORMALIZE=False,
            SCORE_ONLY=True,
            num_warps=2,
            num_stages=3,
            waves_per_eu=1,
        )

    tile_count = triton.cdiv(state_len, 128)
    materialized_candidates = torch.empty(
        batch * query_heads, tile_count, 8, device=device, dtype=torch.float32
    )
    materialized_indices = torch.empty(
        batch * query_heads, tile_count, 8, device=device, dtype=torch.int64
    )

    def materialized_tile_top8() -> None:
        _materialized_state_tile_top8_kernel[(batch * kv_heads, tile_count)](
            hip_scores,
            counts,
            cache_indices,
            materialized_candidates,
            materialized_indices,
            counts.stride(0),
            counts.stride(1),
            counts.stride(2),
            state_len,
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            KV_GROUP_SIZE=gqa,
            STATE_CAPACITY=state_len,
            MAX_TILES=tile_count,
            TOPK_BLOCK_N=128,
            PROTECTED_LEN=0,
            MAX_LEAF_TOKENS=0,
            num_warps=4,
            waves_per_eu=1,
        )

    def materialized_global_top8() -> None:
        _reduce_decode_route_topk_kernel[(batch * query_heads,)](
            materialized_candidates,
            materialized_indices,
            buffers["route_top_slots"],
            buffers["route_top_scores"],
            tile_count,
            ROUTE_COUNT=8,
            MAX_SEGMENTS=tile_count,
            CANDIDATE_BLOCK=triton.next_power_of_2(tile_count * 8),
            num_warps=2,
            waves_per_eu=1,
        )

    def hip_pipeline() -> None:
        hip_qk()
        materialized_tile_top8()
        materialized_global_top8()

    hip_qk()
    triton_materialized_qk()
    hip_fused_candidate_pipeline()
    torch.cuda.synchronize()
    difference = (hip_scores - triton_scores).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    reference_top = torch.topk(triton_scores, 8, dim=-1).indices.sort(dim=-1).values
    hip_top = torch.topk(hip_scores, 8, dim=-1).indices.sort(dim=-1).values
    top8_set_fraction = float((reference_top == hip_top).all(dim=-1).float().mean().item())
    reduced_hip_top = hip_top_slots.sort(dim=-1).values
    hip_candidate_top8_set_fraction = float(
        (reference_top == reduced_hip_top).all(dim=-1).float().mean().item()
    )

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "state_length": state_len,
        "geometry": {"head_dim": head_dim, "kv_heads": kv_heads, "gqa": gqa},
        "correctness": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "top8_set_fraction": top8_set_fraction,
            "candidate_top8_set_fraction": hip_candidate_top8_set_fraction,
        },
        "times_us": {
            "hip_aiter_layout_qk_materialize": 1000.0
            * time_ms(hip_qk, args.warmup, args.repeats),
            "triton_qk_materialize": 1000.0
            * time_ms(triton_materialized_qk, args.warmup, args.repeats),
            "triton_production_score_plus_segment_top8": 1000.0
            * time_ms(production_score_candidates, args.warmup, args.repeats),
            "triton_tile_top8_from_materialized": 1000.0
            * time_ms(materialized_tile_top8, args.warmup, args.repeats),
            "triton_global_top8_from_candidates": 1000.0
            * time_ms(materialized_global_top8, args.warmup, args.repeats),
            "hip_qk_plus_triton_top8_pipeline": 1000.0
            * time_ms(hip_pipeline, args.warmup, args.repeats),
            "hip_score_plus_partition_top8": 1000.0
            * time_ms(hip_candidates, args.warmup, args.repeats),
            "hip_global_top8": 1000.0
            * time_ms(hip_reduce, args.warmup, args.repeats),
            "hip_fused_candidate_pipeline": 1000.0
            * time_ms(hip_fused_candidate_pipeline, args.warmup, args.repeats),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
