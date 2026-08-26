#!/usr/bin/env python3
"""Benchmark exact two-pass mass routing for Muse decode geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.gqa16_coarse_score import (
    gqa16_coarse_scores_lse,
    mass_cutoff_union,
    reduce_partition_lse,
)


def time_us(function, warmup: int, repeats: int) -> float:
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
    return 1000.0 * float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-length", type=int, default=4096)
    parser.add_argument("--mass-fraction", type=float, default=1.0 / 16.0)
    parser.add_argument("--max-leaf-tokens", type=int, default=1024)
    parser.add_argument(
        "--live-centroids",
        type=int,
        default=0,
        help="positive live prefix; zero uses the complete state length",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = args.batch_size
    kv_heads = 2
    gqa = 16
    query_heads = kv_heads * gqa
    head_dim = 128
    state_len = args.state_length
    segments = (state_len + 255) // 256
    q = torch.randn(
        batch, query_heads, 1, head_dim, device=device, dtype=dtype
    )
    means = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=dtype
    )
    counts = torch.randint(
        1, 2049, (batch, kv_heads, state_len, 1), device=device, dtype=torch.int32
    ).float()
    live_centroids = args.live_centroids or state_len
    if not 0 < live_centroids <= state_len:
        raise ValueError("live centroids must lie in [1, state length]")
    counts[:, :, live_centroids:].zero_()
    state_k = (means.float() * counts).to(dtype)
    cache_indices = torch.arange(batch, device=device, dtype=torch.int64)
    scores = torch.empty(
        batch, query_heads, 1, state_len, device=device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        batch * query_heads, segments, device=device, dtype=torch.float32
    )
    full_lse = torch.empty(
        batch, query_heads, 1, device=device, dtype=torch.float32
    )
    sequence_count = batch * kv_heads
    sequence_epochs = torch.zeros(sequence_count, device=device, dtype=torch.int32)
    union_counts = torch.zeros_like(sequence_epochs)
    union_token_counts = torch.zeros_like(sequence_epochs)
    # A strict mass threshold permits at most ceil(1/f)-1 entries per head.
    union_capacity = min(
        state_len, 16 * max(1, math.ceil(1.0 / args.mass_fraction) - 1)
    )
    union_slots = torch.empty(
        sequence_count, union_capacity, device=device, dtype=torch.int32
    )
    seen_stamps = torch.zeros(
        sequence_count, state_len, device=device, dtype=torch.int32
    )

    def score_lse() -> None:
        gqa16_coarse_scores_lse(
            q,
            state_k,
            counts,
            cache_indices,
            scores,
            partial_lse,
            state_len=state_len,
            max_leaf_tokens=args.max_leaf_tokens,
            scale=head_dim**-0.5,
        )

    def reduce_lse() -> None:
        reduce_partition_lse(
            partial_lse,
            full_lse,
            sequence_epochs,
            union_counts,
            union_token_counts,
            query_heads=query_heads,
            kv_heads=kv_heads,
            active_segments=segments,
        )

    def compact_union() -> None:
        mass_cutoff_union(
            scores,
            full_lse,
            counts,
            cache_indices,
            seen_stamps,
            sequence_epochs,
            union_counts,
            union_slots,
            state_len=state_len,
            mass_fraction=args.mass_fraction,
            max_leaf_tokens=args.max_leaf_tokens,
        )

    def route() -> None:
        score_lse()
        reduce_lse()
        compact_union()

    route()
    torch.cuda.synchronize()
    reference_lse = torch.logsumexp(scores, dim=-1)
    lse_max_abs = float((reference_lse - full_lse).abs().max().item())
    selected = scores > (
        reference_lse[..., None]
        + torch.tensor(args.mass_fraction, device=device).log()
    )
    expanded_counts = (
        counts[..., 0][:, :, None, :]
        .expand(batch, kv_heads, gqa, state_len)
        .reshape(batch, query_heads, 1, state_len)
    )
    eligible = (expanded_counts > 0) & (
        args.max_leaf_tokens <= 0
        or expanded_counts < args.max_leaf_tokens
    )
    selected &= eligible
    reference_union = selected.reshape(
        batch, kv_heads, gqa, state_len
    ).any(dim=2)
    reference_counts = reference_union.sum(dim=-1).reshape(-1).to(torch.int32)
    count_exact = bool((reference_counts == union_counts).all().item())
    set_exact = True
    for sequence in range(sequence_count):
        count = int(union_counts[sequence].item())
        actual = union_slots[sequence, :count].sort().values
        expected = torch.nonzero(
            reference_union.reshape(sequence_count, state_len)[sequence],
            as_tuple=False,
        )[:, 0].to(torch.int32)
        if actual.numel() != expected.numel() or not bool(
            (actual == expected).all().item()
        ):
            set_exact = False
            break

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "state_length": state_len,
        "mass_fraction": args.mass_fraction,
        "max_leaf_tokens": args.max_leaf_tokens,
        "correctness": {
            "full_lse_max_abs": lse_max_abs,
            "union_count_exact": count_exact,
            "union_set_exact": set_exact,
        },
        "selected_union_centroids": {
            "minimum": int(union_counts.min().item()),
            "mean": float(union_counts.float().mean().item()),
            "maximum": int(union_counts.max().item()),
            "capacity": union_capacity,
        },
        "times_us": {
            "hip_score_and_partition_lse": time_us(
                score_lse, args.warmup, args.repeats
            ),
            "hip_lse_reduce_and_union_init": time_us(
                reduce_lse, args.warmup, args.repeats
            ),
            "hip_mass_union_compaction": time_us(
                compact_union, args.warmup, args.repeats
            ),
            "hip_route_total": time_us(route, args.warmup, args.repeats),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
