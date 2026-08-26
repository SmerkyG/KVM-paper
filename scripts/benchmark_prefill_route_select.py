#!/usr/bin/env python3
"""Benchmark exact prefill route selection at high-GQA geometries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from model.kernels.lod_kernels import (
    new_route_buffers,
    route_logits_hierarchical_topk,
    route_top8_scores_grouped,
    route_logits_topk_coarse_attention,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=512)
    parser.add_argument("--state-len", type=int, default=4352)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-leaf-tokens", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hier-only", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _run(
    q: torch.Tensor,
    logits: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    *,
    topk: int,
    block_m: int,
    block_n: int,
    num_warps: int,
    head_major: bool | None,
    warmup: int,
    repeats: int,
    max_leaf_tokens: int | None,
) -> tuple[torch.Tensor, list[float]]:
    empty = state_v[..., :0, :].contiguous()

    def invoke() -> torch.Tensor:
        slots, _, _ = route_logits_topk_coarse_attention(
            q,
            logits,
            state_v,
            counts,
            empty,
            empty,
            state_len=int(logits.size(-1)),
            kv_group_size=int(q.size(1) // state_v.size(1)),
            scale=float(q.size(-1) ** -0.5),
            topk=topk,
            protected_len=1,
            max_leaf_tokens=max_leaf_tokens,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            head_major=head_major,
            stable_recompute=True,
            route_only=True,
            hierarchical_route_only=False,
        )
        return slots

    slots = invoke()
    for _ in range(warmup - 1):
        slots = invoke()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        slots = invoke()
        end.record()
    torch.cuda.synchronize()
    return slots, [start.elapsed_time(end) * 1_000.0 for start, end in zip(starts, ends, strict=True)]


def _run_grouped(
    logits: torch.Tensor,
    counts: torch.Tensor,
    *,
    scale: float,
    topk: int,
    warmup: int,
    repeats: int,
    max_leaf_tokens: int | None,
) -> tuple[torch.Tensor, list[float]]:
    if max_leaf_tokens is not None:
        raise ValueError("the production grouped selector benchmark has no leaf cap")
    buffers = new_route_buffers(
        logits[..., :1],
        state_capacity=int(logits.size(-1)),
        query_capacity=int(logits.size(2)),
    )

    def invoke() -> torch.Tensor:
        return route_top8_scores_grouped(
            logits,
            counts,
            buffers,
            kv_group_size=int(logits.size(1) // counts.size(1)),
            scale=scale,
            topk=topk,
            state_len=int(logits.size(-1)),
            protected_len=1,
            reorder_like_torch=True,
        )

    slots = invoke()
    for _ in range(warmup - 1):
        slots = invoke()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        slots = invoke()
        end.record()
    torch.cuda.synchronize()
    return slots, [
        start.elapsed_time(end) * 1_000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def main() -> None:
    args = _parse_args()
    if args.query_heads % args.kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    q = torch.randn(
        args.batch,
        args.query_heads,
        args.query_len,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    logits = torch.randn(
        args.batch,
        args.query_heads,
        args.query_len,
        args.state_len,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    state_v = torch.randn(
        args.batch,
        args.kv_heads,
        args.state_len,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).contiguous()
    counts = torch.randint(
        1,
        1025,
        (args.batch, args.kv_heads, args.state_len, 1),
        dtype=torch.int32,
        device=device,
    ).to(torch.float32).contiguous()

    configs = [
        ("production_auto_m16_n32_w8", 16, 32, 8, None),
        ("group_m8_n32_w4", 8, 32, 4, False),
        ("group_m8_n32_w8", 8, 32, 8, False),
        ("group_m4_n32_w2", 4, 32, 2, False),
        ("group_m4_n32_w4", 4, 32, 4, False),
        ("group_m4_n32_w8", 4, 32, 8, False),
        ("head_m16_n32_w1", 16, 32, 1, True),
        ("head_m16_n32_w2", 16, 32, 2, True),
        ("head_m16_n32_w4", 16, 32, 4, True),
        ("head_m16_n32_w8", 16, 32, 8, True),
        ("head_m32_n32_w2", 32, 32, 2, True),
        ("head_m32_n32_w4", 32, 32, 4, True),
        ("head_m32_n32_w8", 32, 32, 8, True),
        ("head_m64_n32_w4", 64, 32, 4, True),
        ("head_m64_n32_w8", 64, 32, 8, True),
        ("head_m16_n64_w2", 16, 64, 2, True),
        ("head_m16_n64_w4", 16, 64, 4, True),
        ("group_m4_n64_w4", 4, 64, 4, False),
    ]
    results: list[dict[str, object]] = []
    reference = None
    reference_order = None
    for label, block_m, block_n, num_warps, head_major in (
        configs[:1] if args.hier_only else configs
    ):
        try:
            slots, times_us = _run(
                q,
                logits,
                state_v,
                counts,
                topk=args.topk,
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                head_major=head_major,
                warmup=args.warmup,
                repeats=args.repeats,
                max_leaf_tokens=args.max_leaf_tokens,
            )
        except ValueError as error:
            result = {"label": label, "unsupported": str(error)}
            results.append(result)
            print(json.dumps(result), flush=True)
            continue
        normalized = slots.sort(dim=-1).values
        if reference is None:
            reference = normalized
            reference_order = slots
        exact = bool(torch.equal(normalized, reference))
        result = {
            "label": label,
            "block_m": block_m,
            "block_n": block_n,
            "num_warps": num_warps,
            "head_major": (
                head_major
                if head_major is not None
                else bool(
                    (block_m * (args.query_heads // args.kv_heads))
                    & (block_m * (args.query_heads // args.kv_heads) - 1)
                )
            ),
            "logical_rows_per_program": (
                block_m
                if head_major
                or (
                    head_major is None
                    and (
                        block_m * (args.query_heads // args.kv_heads)
                        & (block_m * (args.query_heads // args.kv_heads) - 1)
                    )
                )
                else block_m * (args.query_heads // args.kv_heads)
            ),
            "median_us": statistics.median(times_us),
            "min_us": min(times_us),
            "max_us": max(times_us),
            "route_set_exact": exact,
            "route_order_exact": bool(torch.equal(slots, reference_order)),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    grouped_slots, grouped_times_us = _run_grouped(
        logits,
        counts,
        scale=float(args.head_dim**-0.5),
        topk=args.topk,
        warmup=args.warmup,
        repeats=args.repeats,
        max_leaf_tokens=args.max_leaf_tokens,
    )
    grouped_result = {
        "label": "production_grouped_m16_n64_w4",
        "block_m": 16,
        "block_n": 64,
        "num_warps": 4,
        "median_us": statistics.median(grouped_times_us),
        "min_us": min(grouped_times_us),
        "max_us": max(grouped_times_us),
        "route_set_exact": bool(
            torch.equal(grouped_slots.sort(dim=-1).values, reference)
        ),
        "route_order_exact": bool(torch.equal(grouped_slots, reference_order)),
    }
    results.append(grouped_result)
    print(json.dumps(grouped_result), flush=True)
    hierarchical_configs = (
        (8, 64, 4, 4),
        (16, 64, 4, 4),
        (8, 128, 4, 4),
        (16, 128, 2, 2),
        (16, 128, 4, 2),
        (16, 128, 4, 4),
        (16, 128, 8, 4),
        (32, 128, 4, 4),
        (8, 256, 4, 4),
        (16, 256, 2, 2),
        (16, 256, 4, 2),
        (16, 256, 4, 4),
        (16, 256, 8, 4),
        (32, 256, 4, 4),
        (8, 512, 4, 4),
        (16, 512, 4, 2),
        (16, 512, 4, 4),
        (8, 1024, 2, 2),
        (16, 1024, 1, 2),
        (16, 1024, 2, 2),
        (16, 1024, 4, 4),
        (16, 1024, 8, 4),
        (32, 1024, 2, 2),
        (8, 2048, 2, 2),
        (16, 2048, 1, 2),
        (16, 2048, 2, 2),
        (16, 2048, 4, 4),
        (16, 4096, 2, 2),
        (16, 4096, 4, 4),
    )
    if args.focused:
        hierarchical_configs = (
            (8, 256, 2, 2),
            (16, 256, 2, 2),
            (8, 512, 2, 2),
            (8, 512, 4, 4),
            (8, 1024, 2, 2),
        )
    for block_m, block_n, tile_warps, reduce_warps in hierarchical_configs:
        label = f"hier_m{block_m}_n{block_n}_tw{tile_warps}_rw{reduce_warps}"

        def invoke_hierarchical() -> torch.Tensor:
            return route_logits_hierarchical_topk(
                logits,
                counts,
                state_len=args.state_len,
                kv_group_size=args.query_heads // args.kv_heads,
                scale=float(args.head_dim**-0.5),
                topk=args.topk,
                protected_len=1,
                max_leaf_tokens=args.max_leaf_tokens,
                block_m=block_m,
                block_n=block_n,
                tile_num_warps=tile_warps,
                reduce_num_warps=reduce_warps,
            )

        slots = invoke_hierarchical()
        for _ in range(args.warmup - 1):
            slots = invoke_hierarchical()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            slots = invoke_hierarchical()
            end.record()
        torch.cuda.synchronize()
        times_us = [
            start.elapsed_time(end) * 1_000.0
            for start, end in zip(starts, ends, strict=True)
        ]
        normalized = slots.sort(dim=-1).values
        exact = bool(torch.equal(normalized, reference))
        result = {
            "label": label,
            "block_m": block_m,
            "block_n": block_n,
            "tile_num_warps": tile_warps,
            "reduce_num_warps": reduce_warps,
            "centroid_tiles": (args.state_len + block_n - 1) // block_n,
            "median_us": statistics.median(times_us),
            "min_us": min(times_us),
            "max_us": max(times_us),
            "route_set_exact": exact,
            "route_order_exact": bool(torch.equal(slots, reference_order)),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "geometry": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": sorted(
            results,
            key=lambda item: float(item.get("median_us", float("inf"))),
        ),
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)


if __name__ == "__main__":
    main()
