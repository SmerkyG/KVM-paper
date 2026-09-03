#!/usr/bin/env python3
"""Profile a chunked GEMM formulation of dense page attention and selection."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--key-length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--query-tiles", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_ms(function, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats


def main() -> None:
    args = parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be a multiple of KV heads")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    group = args.query_heads // args.kv_heads
    scale = args.head_dim**-0.5
    q = torch.randn(
        args.batch_size,
        args.query_heads,
        args.query_length,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.key_length,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn_like(k)
    counts = torch.randint(
        1,
        17,
        (args.batch_size, args.kv_heads, args.key_length),
        dtype=torch.int32,
        device=device,
    )
    log_counts = counts.float().log().unsqueeze(2).unsqueeze(2)
    grouped_q = q.view(
        args.batch_size,
        args.kv_heads,
        group,
        args.query_length,
        args.head_dim,
    )
    grouped_k_t = k.unsqueeze(2).transpose(-1, -2)
    grouped_v = v.unsqueeze(2)

    records = []
    for query_tile in args.query_tiles:
        if args.query_length % query_tile:
            continue

        def pipeline():
            output_tiles = []
            for query_begin in range(0, args.query_length, query_tile):
                query = grouped_q[..., query_begin : query_begin + query_tile, :]
                logits = torch.matmul(query, grouped_k_t)
                scores = logits.float().mul_(scale).add_(log_counts)
                torch.topk(scores, args.topk, dim=-1)
                probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
                output_tiles.append(torch.matmul(probabilities, grouped_v))
            return torch.cat(output_tiles, dim=-2)

        torch.cuda.reset_peak_memory_stats(device)
        whole_ms = timed_ms(pipeline, args.warmups, args.repeats)
        output = pipeline()
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device)

        phase_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list)
        )
        output_tiles = []
        for query_begin in range(0, args.query_length, query_tile):
            query = grouped_q[..., query_begin : query_begin + query_tile, :]
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            logits = torch.matmul(query, grouped_k_t)
            end.record()
            phase_pairs["qk_gemm"].append((begin, end))

            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            scores = logits.float().mul_(scale).add_(log_counts)
            end.record()
            phase_pairs["score_cast_scale_bias"].append((begin, end))

            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            torch.topk(scores, args.topk, dim=-1)
            end.record()
            phase_pairs["topk"].append((begin, end))

            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16)
            end.record()
            phase_pairs["softmax_and_cast"].append((begin, end))

            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output_tiles.append(torch.matmul(probabilities, grouped_v))
            end.record()
            phase_pairs["pv_gemm"].append((begin, end))
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        torch.cat(output_tiles, dim=-2)
        end.record()
        phase_pairs["output_concat"].append((begin, end))
        torch.cuda.synchronize()
        phase_ms = {
            name: sum(start.elapsed_time(stop) for start, stop in pairs)
            for name, pairs in phase_pairs.items()
        }
        records.append(
            {
                "query_tile": query_tile,
                "whole_ms": whole_ms,
                "phase_ms": phase_ms,
                "phase_sum_ms": sum(phase_ms.values()),
                "peak_allocated_bytes": peak_bytes,
                "finite": bool(torch.isfinite(output).all().item()),
            }
        )

    result = {
        "batch_size": args.batch_size,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "query_length": args.query_length,
        "key_length": args.key_length,
        "head_dim": args.head_dim,
        "topk": args.topk,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
