#!/usr/bin/env python3
"""Sweep the score-scan geometry used by unordered LOD mass routing."""

from __future__ import annotations

import argparse
import json

import torch

from model.kernels.lod_kernels import route_mass_fraction_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--state-length", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(19)
    device = torch.device("cuda", 0)
    query_heads = 8
    kv_heads = 2
    groups = query_heads // kv_heads
    logits = torch.randn(
        args.batch_size,
        query_heads,
        args.query_length,
        args.state_length,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    counts = torch.randint(
        1,
        65,
        (args.batch_size, kv_heads, args.state_length, 1),
        device=device,
    ).float().contiguous()
    route_lengths = torch.randint(
        0,
        65,
        (args.batch_size, kv_heads, args.state_length),
        device=device,
        dtype=torch.int32,
    ).contiguous()
    local_lse = torch.randn(
        args.batch_size,
        query_heads,
        args.query_length,
        device=device,
        dtype=torch.float32,
    ).add_(8.0).contiguous()

    reference = None
    results = []
    geometries = [(False, 16, 4)] + [
        (True, block_m, warps)
        for block_m in (4, 8, 16)
        for warps in (1, 2, 4)
    ]
    for grouped, block_m, warps in geometries:
        kwargs = dict(
            route_lengths=route_lengths,
            kv_group_size=groups,
            scale=0.125,
            mass_fraction=1.0 / 128.0,
            max_routes=16,
            state_len=args.state_length,
            protected_len=1,
            local_lse=local_lse,
            block_m=block_m,
            block_n=128,
            num_warps=warps,
            group_gqa=grouped,
        )
        actual = route_mass_fraction_scores(logits, counts, **kwargs)
        torch.cuda.synchronize(device)
        if reference is None:
            reference = tuple(value.clone() for value in actual)
        elif any(not torch.equal(left, right) for left, right in zip(actual, reference)):
            raise AssertionError(
                f"mass routing changed at grouped={grouped}, M={block_m}, warps={warps}"
            )
        events = []
        for _ in range(args.repeats):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            route_mass_fraction_scores(logits, counts, **kwargs)
            end.record()
            events.append((begin, end))
        torch.cuda.synchronize(device)
        results.append(
            {
                "group_gqa": grouped,
                "block_m": block_m,
                "num_warps": warps,
                "milliseconds": sum(a.elapsed_time(b) for a, b in events)
                / args.repeats,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
