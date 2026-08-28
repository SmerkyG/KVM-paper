#!/usr/bin/env python3
"""Compare low-row centroid scorers with identical exact top-k reduction."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from model.kernels.paged_leaf_attention import (
    _decode_route_coarse_gqa_groups_kernel,
    _decode_route_coarse_query_major_score_kernel,
    _decode_route_coarse_scalar_gqa_groups_kernel,
    _reduce_decode_route_topk_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260828)
    device = torch.device("cuda")
    query_heads = args.kv_heads * args.gqa
    query_rows = args.batch_size * query_heads
    shape = (args.batch_size, args.kv_heads, args.slots, args.head_dim)
    q = torch.randn(
        args.batch_size,
        query_heads,
        1,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    state_k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    state_v = torch.empty(shape, device=device, dtype=torch.bfloat16)
    counts = torch.randint(
        1,
        257,
        (args.batch_size, args.kv_heads, args.slots, 1),
        device=device,
        dtype=torch.int32,
    ).to(torch.float32)
    cache_indices = torch.arange(args.batch_size, device=device, dtype=torch.int32)
    max_groups = triton.cdiv(args.slots, 8)
    candidate_scores = torch.empty(
        query_rows, max_groups, 8, device=device, dtype=torch.float32
    )
    candidate_indices = torch.empty(
        query_rows, max_groups, 8, device=device, dtype=torch.int32
    )
    unused_out = torch.empty(
        query_rows, max_groups, args.head_dim, device=device, dtype=torch.float32
    )
    unused_lse = torch.empty(
        query_rows, max_groups, device=device, dtype=torch.float32
    )
    top_slots = torch.empty(query_rows, 8, device=device, dtype=torch.int32)
    top_scores = torch.empty(query_rows, 8, device=device, dtype=torch.float32)

    arguments = (
        q,
        state_k,
        state_v,
        counts,
        cache_indices,
        candidate_scores,
        candidate_indices,
        unused_out,
        unused_lse,
        state_k.stride(0),
        state_k.stride(1),
        state_k.stride(2),
        state_v.stride(0),
        state_v.stride(1),
        state_v.stride(2),
        counts.stride(0),
        counts.stride(1),
        counts.stride(2),
        args.slots,
    )
    configurations = []
    for group_n in (16, 32, 64):
        for warps in (1, 2, 4):
            configurations.append(("mfma_gqa", group_n, warps))
            configurations.append(("query_major_vector", group_n, warps))
        if group_n == 16:
            for warps in (1, 2, 4):
                configurations.append(("scalar_gqa", group_n, warps))

    measurements: list[dict[str, object]] = []
    reference_slots: torch.Tensor | None = None
    for axis, group_n, warps in configurations:
        active_groups = triton.cdiv(args.slots, group_n)
        if axis == "mfma_gqa":
            kernel = _decode_route_coarse_gqa_groups_kernel
            grid = (args.batch_size * args.kv_heads, active_groups)
        elif axis == "query_major_vector":
            kernel = _decode_route_coarse_query_major_score_kernel
            grid = (query_rows, active_groups)
        else:
            kernel = _decode_route_coarse_scalar_gqa_groups_kernel
            grid = (args.batch_size * args.kv_heads, active_groups)

        def launch() -> None:
            kernel[grid](
                *arguments,
                QUERY_HEADS=query_heads,
                KV_HEADS=args.kv_heads,
                KV_GROUP_SIZE=args.gqa,
                HEAD_DIM=args.head_dim,
                SCALE=args.head_dim**-0.5,
                GROUP_N=group_n,
                MAX_GROUPS=max_groups,
                PROTECTED_LEN=1,
                MAX_LEAF_TOKENS=1024,
                USE_DOT=True,
                SCORE_ONLY=True,
                num_warps=warps,
                waves_per_eu=1,
            )
            _reduce_decode_route_topk_kernel[(query_rows,)](
                candidate_scores,
                candidate_indices,
                top_slots,
                top_scores,
                active_groups,
                ROUTE_COUNT=8,
                OPEN_COUNT=8,
                MAX_SEGMENTS=max_groups,
                CANDIDATE_BLOCK=triton.next_power_of_2(active_groups * 8),
                num_warps=4,
                waves_per_eu=1,
            )

        launch()
        torch.cuda.synchronize()
        score_samples: list[float] = []
        reduce_samples: list[float] = []
        total_samples: list[float] = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            scored = torch.cuda.Event(enable_timing=True)
            reduced = torch.cuda.Event(enable_timing=True)
            begin.record()
            kernel[grid](
                *arguments,
                QUERY_HEADS=query_heads,
                KV_HEADS=args.kv_heads,
                KV_GROUP_SIZE=args.gqa,
                HEAD_DIM=args.head_dim,
                SCALE=args.head_dim**-0.5,
                GROUP_N=group_n,
                MAX_GROUPS=max_groups,
                PROTECTED_LEN=1,
                MAX_LEAF_TOKENS=1024,
                USE_DOT=True,
                SCORE_ONLY=True,
                num_warps=warps,
                waves_per_eu=1,
            )
            scored.record()
            _reduce_decode_route_topk_kernel[(query_rows,)](
                candidate_scores,
                candidate_indices,
                top_slots,
                top_scores,
                active_groups,
                ROUTE_COUNT=8,
                OPEN_COUNT=8,
                MAX_SEGMENTS=max_groups,
                CANDIDATE_BLOCK=triton.next_power_of_2(active_groups * 8),
                num_warps=4,
                waves_per_eu=1,
            )
            reduced.record()
            reduced.synchronize()
            score_samples.append(begin.elapsed_time(scored) * 1000.0)
            reduce_samples.append(scored.elapsed_time(reduced) * 1000.0)
            total_samples.append(begin.elapsed_time(reduced) * 1000.0)
        actual_slots = top_slots.detach().clone()
        if reference_slots is None:
            reference_slots = actual_slots
        slot_agreement = float(
            (actual_slots == reference_slots).to(torch.float32).mean().item()
        )
        measurement = {
            "axis": axis,
            "group_n": group_n,
            "num_warps": warps,
            "programs": grid[0] * grid[1],
            "score_us_median": statistics.median(score_samples),
            "reduce_us_median": statistics.median(reduce_samples),
            "total_us_median": statistics.median(total_samples),
            "top_slot_agreement": slot_agreement,
        }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)

    result = {
        "batch_size": args.batch_size,
        "slots": args.slots,
        "head_dim": args.head_dim,
        "kv_heads": args.kv_heads,
        "gqa": args.gqa,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
