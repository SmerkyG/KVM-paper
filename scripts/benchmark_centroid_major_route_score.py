#!/usr/bin/env python3
"""Compare centroid-major, one-wave/query, and production MFMA routing."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton

from model.kernels.centroid_major_route_score import (
    centroid_major_route_score,
    query_wave_route_score,
)
from model.kernels.paged_leaf_attention import (
    _decode_route_coarse_gqa_groups_kernel,
    _reduce_decode_route_topk_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, choices=(4, 6), default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260828)
    device = torch.device("cuda")
    head_dim = 256
    gqa = args.gqa
    query_heads = args.kv_heads * gqa
    query_rows = args.batch_size * query_heads
    shape = (args.batch_size, args.kv_heads, args.slots, head_dim)
    q = torch.randn(
        args.batch_size,
        query_heads,
        1,
        head_dim,
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
    cache_indices = torch.arange(args.batch_size, device=device, dtype=torch.int64)
    group_n = 32
    active_groups = triton.cdiv(args.slots, group_n)
    max_groups = active_groups
    candidate_shape = (query_rows, max_groups, 8)
    hip_scores = torch.empty(candidate_shape, device=device, dtype=torch.float32)
    hip_indices = torch.empty(candidate_shape, device=device, dtype=torch.int64)
    wave_scores = torch.empty_like(hip_scores)
    wave_indices = torch.empty_like(hip_indices)
    mfma_scores = torch.empty_like(hip_scores)
    mfma_indices = torch.empty_like(hip_indices)
    unused_out = torch.empty(
        query_rows, max_groups, head_dim, device=device, dtype=torch.float32
    )
    unused_lse = torch.empty(query_rows, max_groups, device=device, dtype=torch.float32)
    hip_top_slots = torch.empty(query_rows, 8, device=device, dtype=torch.int32)
    hip_top_scores = torch.empty(query_rows, 8, device=device, dtype=torch.float32)
    wave_top_slots = torch.empty_like(hip_top_slots)
    wave_top_scores = torch.empty_like(hip_top_scores)
    mfma_top_slots = torch.empty_like(hip_top_slots)
    mfma_top_scores = torch.empty_like(hip_top_scores)
    candidate_block = triton.next_power_of_2(active_groups * 8)

    def hip_score(mean_before_dot: bool = True) -> None:
        centroid_major_route_score(
            q,
            state_k,
            counts,
            cache_indices,
            hip_scores,
            hip_indices,
            state_len=args.slots,
            protected_len=1,
            max_leaf_tokens=1024,
            mean_before_dot=mean_before_dot,
            scale=head_dim**-0.5,
        )

    def mfma_score(warps: int = 2) -> None:
        _decode_route_coarse_gqa_groups_kernel[
            (args.batch_size * args.kv_heads, active_groups)
        ](
            q,
            state_k,
            state_v,
            counts,
            cache_indices,
            mfma_scores,
            mfma_indices,
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
            QUERY_HEADS=query_heads,
            KV_HEADS=args.kv_heads,
            KV_GROUP_SIZE=gqa,
            HEAD_DIM=head_dim,
            SCALE=head_dim**-0.5,
            GROUP_N=group_n,
            MAX_GROUPS=max_groups,
            PROTECTED_LEN=1,
            MAX_LEAF_TOKENS=1024,
            USE_DOT=True,
            SCORE_ONLY=True,
            num_warps=warps,
            waves_per_eu=1,
        )

    def query_wave_score(mean_before_dot: bool = True) -> None:
        query_wave_route_score(
            q,
            state_k,
            counts,
            cache_indices,
            wave_scores,
            wave_indices,
            state_len=args.slots,
            protected_len=1,
            max_leaf_tokens=1024,
            mean_before_dot=mean_before_dot,
            scale=head_dim**-0.5,
        )

    def reduce(
        candidate_scores: torch.Tensor,
        candidate_indices: torch.Tensor,
        top_slots: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> None:
        _reduce_decode_route_topk_kernel[(query_rows,)](
            candidate_scores,
            candidate_indices,
            top_slots,
            top_scores,
            active_groups,
            ROUTE_COUNT=8,
            OPEN_COUNT=8,
            MAX_SEGMENTS=max_groups,
            CANDIDATE_BLOCK=candidate_block,
            num_warps=4,
            waves_per_eu=1,
        )

    methods = {
        "hip_centroid_major": (
            lambda: hip_score(True),
            lambda: reduce(hip_scores, hip_indices, hip_top_slots, hip_top_scores),
        ),
        "hip_centroid_major_postdot_mean": (
            lambda: hip_score(False),
            lambda: reduce(hip_scores, hip_indices, hip_top_slots, hip_top_scores),
        ),
        "triton_mfma_w1": (
            lambda: mfma_score(1),
            lambda: reduce(mfma_scores, mfma_indices, mfma_top_slots, mfma_top_scores),
        ),
        "triton_mfma_w2": (
            lambda: mfma_score(2),
            lambda: reduce(mfma_scores, mfma_indices, mfma_top_slots, mfma_top_scores),
        ),
        "triton_mfma_w4": (
            lambda: mfma_score(4),
            lambda: reduce(mfma_scores, mfma_indices, mfma_top_slots, mfma_top_scores),
        ),
    }
    if gqa == 4:
        methods["hip_one_wave_per_query"] = (
            lambda: query_wave_score(True),
            lambda: reduce(
                wave_scores, wave_indices, wave_top_slots, wave_top_scores
            ),
        )

    # Compile all paths outside the timed region and establish the MFMA result.
    for score, finish in methods.values():
        score()
        finish()
    torch.cuda.synchronize()
    mfma_score(2)
    reduce(mfma_scores, mfma_indices, mfma_top_slots, mfma_top_scores)
    hip_score(True)
    reduce(hip_scores, hip_indices, hip_top_slots, hip_top_scores)
    if gqa == 4:
        query_wave_score(True)
        reduce(wave_scores, wave_indices, wave_top_slots, wave_top_scores)
    torch.cuda.synchronize()
    sorted_reference = mfma_top_slots.sort(dim=-1).values
    sorted_hip = hip_top_slots.sort(dim=-1).values
    top8_row_agreement = float(
        (sorted_reference == sorted_hip).all(dim=-1).float().mean().item()
    )
    top8_slot_agreement = float(
        (sorted_reference == sorted_hip).float().mean().item()
    )
    query_wave_top8_row_agreement = None
    query_wave_top8_slot_agreement = None
    if gqa == 4:
        sorted_wave = wave_top_slots.sort(dim=-1).values
        query_wave_top8_row_agreement = float(
            (sorted_reference == sorted_wave).all(dim=-1).float().mean().item()
        )
        query_wave_top8_slot_agreement = float(
            (sorted_reference == sorted_wave).float().mean().item()
        )

    measurements: dict[str, dict[str, float]] = {}
    for name, (score, finish) in methods.items():
        for _ in range(args.warmup):
            score()
            finish()
        torch.cuda.synchronize()
        score_samples: list[float] = []
        reduce_samples: list[float] = []
        total_samples: list[float] = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            scored = torch.cuda.Event(enable_timing=True)
            completed = torch.cuda.Event(enable_timing=True)
            begin.record()
            score()
            scored.record()
            finish()
            completed.record()
            completed.synchronize()
            score_samples.append(begin.elapsed_time(scored) * 1000.0)
            reduce_samples.append(scored.elapsed_time(completed) * 1000.0)
            total_samples.append(begin.elapsed_time(completed) * 1000.0)
        measurements[name] = {
            "score_us_median": statistics.median(score_samples),
            "reduce_us_median": statistics.median(reduce_samples),
            "total_us_median": statistics.median(total_samples),
        }
        print(json.dumps({"method": name, **measurements[name]}), flush=True)

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "slots": args.slots,
        "head_dim": head_dim,
        "kv_heads": args.kv_heads,
        "gqa": gqa,
        "active_groups": active_groups,
        "correctness": {
            "top8_row_agreement": top8_row_agreement,
            "top8_slot_agreement": top8_slot_agreement,
            "query_wave_top8_row_agreement": query_wave_top8_row_agreement,
            "query_wave_top8_slot_agreement": query_wave_top8_slot_agreement,
        },
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
