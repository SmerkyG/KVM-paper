#!/usr/bin/env python3
"""Benchmark GQA-shared versus query-major routed-centroid correction."""

from __future__ import annotations

import argparse
import json

import torch

from model.kernels.lod_kernels import remove_state_slots_from_attention


def elapsed_ms(function, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--state-len", type=int, default=2048)
    parser.add_argument("--route-group-size", type=int)
    parser.add_argument(
        "--implementation",
        choices=("both", "gqa", "query"),
        default="both",
    )
    parser.add_argument("--union-counts", type=int, nargs="+", default=[8, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if max(args.union_counts) > args.state_len:
        raise ValueError("union count exceeds state length")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    group_size = args.query_heads // args.kv_heads
    route_group_size = args.route_group_size or group_size
    if group_size % route_group_size:
        raise ValueError("route group size must divide the physical GQA group")
    q = torch.randn(
        args.batch,
        args.query_heads,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    state_k = torch.randn(
        args.batch,
        args.kv_heads,
        args.state_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    state_v = torch.randn_like(state_k)
    counts = torch.ones(
        args.batch,
        args.kv_heads,
        args.state_len,
        1,
        device=device,
        dtype=torch.float32,
    )
    attention_out = torch.randn_like(q)
    # A deliberately loose LSE keeps the subtraction numerically well-defined;
    # timing is insensitive to the concrete attention distribution.
    attention_lse = torch.full(
        q.shape[:-1], 20.0, device=device, dtype=torch.float32
    )
    results = []
    for union_count in args.union_counts:
        shared = torch.arange(
            union_count, device=device, dtype=torch.int64
        ).view(1, 1, 1, union_count)
        slots = shared.expand(
            args.batch, args.query_heads, 1, union_count
        ).contiguous()
        corrected: dict[bool, tuple[torch.Tensor, torch.Tensor]] = {}
        implementations = {
            "both": (True, False),
            "gqa": (True,),
            "query": (False,),
        }[args.implementation]
        for gqa_aware in implementations:
            output = attention_out.clone()
            lse = attention_lse.clone()

            def run() -> None:
                output.copy_(attention_out)
                lse.copy_(attention_lse)
                remove_state_slots_from_attention(
                    q,
                    state_k,
                    state_v,
                    counts,
                    slots,
                    output,
                    lse,
                    kv_group_size=group_size,
                    route_group_size=route_group_size,
                    scale=args.head_dim**-0.5,
                    gqa_aware=gqa_aware,
                )

            milliseconds = elapsed_ms(
                run, warmup=args.warmup, iterations=args.iterations
            )
            corrected[gqa_aware] = (output.clone(), lse.clone())
            result = {
                "gqa_aware": gqa_aware,
                "union_count": union_count,
                "milliseconds": milliseconds,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        if args.implementation == "both":
            torch.testing.assert_close(
                corrected[True][0].float(),
                corrected[False][0].float(),
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                corrected[True][1], corrected[False][1], rtol=2e-5, atol=2e-5
            )
            print(json.dumps({"union_count": union_count, "parity": "passed"}))


if __name__ == "__main__":
    main()
