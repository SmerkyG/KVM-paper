#!/usr/bin/env python3
"""Compare split and fused AITER-stat branch reductions under graph replay."""

from __future__ import annotations

import argparse
import json

import torch

from model.kernels.lod_kernels import (
    aiter_partition_lse,
    merge_attention_branches_with_aiter_stats,
    merge_attention_branches_with_sink,
)


def capture(function):
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    return graph


def elapsed_ms(graph, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--partitions", type=int, nargs="+", default=[68, 261])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--num-warps", type=int, default=1)
    args = parser.parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    group_size = args.query_heads // args.kv_heads
    q = torch.randn(
        args.batch, args.query_heads, 1, args.head_dim, device=device, dtype=dtype
    )
    primary_out = torch.randn_like(q)
    primary_lse = torch.randn(q.shape[:-1], device=device, dtype=torch.float32)
    exact_out = torch.randn_like(q)
    sink_k = torch.randn(
        args.batch, args.kv_heads, 1, args.head_dim, device=device, dtype=dtype
    )
    sink_v = torch.randn_like(sink_k)
    split_output = torch.empty_like(q)
    fused_output = torch.empty_like(q)

    for partitions in args.partitions:
        shape = (args.batch * args.kv_heads, group_size, partitions)
        exp_sums = torch.rand(shape, device=device, dtype=torch.float32) + 0.1
        max_logits = torch.randn(shape, device=device, dtype=torch.float32)
        lengths = torch.full(
            (args.batch * args.kv_heads,),
            partitions * 256,
            device=device,
            dtype=torch.int32,
        )

        def split() -> None:
            exact_lse = aiter_partition_lse(
                exact_out,
                exp_sums,
                max_logits,
                lengths,
                kv_group_size=group_size,
            )
            merge_attention_branches_with_sink(
                q,
                sink_k,
                sink_v,
                primary_out,
                primary_lse,
                exact_out,
                exact_lse,
                kv_group_size=group_size,
                scale=args.head_dim**-0.5,
                output_buffer=split_output,
                block_m=1,
                num_warps=args.num_warps,
            )

        def fused() -> None:
            merge_attention_branches_with_aiter_stats(
                q,
                primary_out,
                primary_lse,
                exact_out,
                exp_sums,
                max_logits,
                lengths,
                kv_group_size=group_size,
                scale=args.head_dim**-0.5,
                sink_k=sink_k,
                sink_v=sink_v,
                output_buffer=fused_output,
                num_warps=args.num_warps,
            )

        split()
        fused()
        torch.testing.assert_close(
            fused_output.float(), split_output.float(), rtol=2e-2, atol=2e-2
        )
        split_graph = capture(split)
        fused_graph = capture(fused)
        row = {
            "partitions": partitions,
            "num_warps": args.num_warps,
            "split_ms": elapsed_ms(
                split_graph, warmup=args.warmup, iterations=args.iterations
            ),
            "fused_ms": elapsed_ms(
                fused_graph, warmup=args.warmup, iterations=args.iterations
            ),
        }
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
