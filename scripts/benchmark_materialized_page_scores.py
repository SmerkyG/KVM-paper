#!/usr/bin/env python3
"""Validate and time GQA-shared decode scoring over all page summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.paged_leaf_attention import materialize_page_summary_scores_gqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--page-block-n", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def elapsed_ms(function, repetitions: int) -> float:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)) / repetitions


def main() -> None:
    args = parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    group_size = args.query_heads // args.kv_heads
    page_capacity = math.ceil(args.context_length / args.page_size)
    scale = args.head_dim**-0.5
    torch.manual_seed(1234)
    q = torch.randn(
        args.batch_size,
        args.query_heads,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    page_sum_k = torch.randn(
        args.batch_size,
        args.kv_heads,
        page_capacity,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    counts = torch.randint(
        1,
        args.page_size + 1,
        (args.batch_size, args.kv_heads, page_capacity),
        device=device,
        dtype=torch.int32,
    )
    cache_indices = torch.arange(args.batch_size, device=device, dtype=torch.long)
    output = torch.empty(
        args.batch_size,
        args.query_heads,
        1,
        page_capacity,
        device=device,
        dtype=torch.float32,
    )

    def run() -> torch.Tensor:
        return materialize_page_summary_scores_gqa(
            q,
            page_sum_k,
            counts,
            cache_indices=cache_indices,
            kv_group_size=group_size,
            scale=scale,
            output=output,
            page_block_n=args.page_block_n,
            num_warps=args.num_warps,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    kernel_ms = elapsed_ms(run, args.repetitions)

    grouped_q = q[:, :, 0].reshape(
        args.batch_size, args.kv_heads, group_size, args.head_dim
    ).float()
    reference = torch.einsum("bkgd,bkpd->bkgp", grouped_q, page_sum_k.float())
    reference = (
        scale * math.log2(math.e) * reference / counts.float().unsqueeze(2)
        + torch.log2(counts.float()).unsqueeze(2)
    ).reshape(args.batch_size, args.query_heads, 1, page_capacity)
    difference = (output - reference).abs()
    bytes_read = (
        page_sum_k.numel() * page_sum_k.element_size()
        + counts.numel() * counts.element_size()
        + q.numel() * q.element_size()
    )
    bytes_written = output.numel() * output.element_size()
    record = {
        "batch_size": args.batch_size,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "gqa_group_size": group_size,
        "head_dim": args.head_dim,
        "context_length": args.context_length,
        "page_size": args.page_size,
        "page_capacity": page_capacity,
        "page_block_n": args.page_block_n,
        "num_warps": args.num_warps,
        "kernel_ms": kernel_ms,
        "effective_io_gbps": (bytes_read + bytes_written) / (kernel_ms * 1e6),
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "finite": bool(torch.isfinite(output).all().item()),
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
