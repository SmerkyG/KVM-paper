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
    route_top8_state_grouped,
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
    parser.add_argument("--grouped-hier-only", action="store_true")
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
    torch_scores = (
        (logits * float(args.head_dim**-0.5)).to(torch.bfloat16).float()
        + counts.clamp_min(1).log()
        .unsqueeze(2)
        .expand(-1, -1, args.query_heads // args.kv_heads, -1, -1)
        .reshape(args.batch, args.query_heads, 1, args.state_len)
    )
    torch_scores[..., 0] = float("-inf")
    if args.max_leaf_tokens is not None:
        valid_counts = (
            counts.le(args.max_leaf_tokens)
            .unsqueeze(2)
            .expand(-1, -1, args.query_heads // args.kv_heads, -1, -1)
            .reshape(args.batch, args.query_heads, 1, args.state_len)
        )
        torch_scores.masked_fill_(~valid_counts, float("-inf"))
    torch_reference = torch_scores.topk(args.topk, dim=-1).indices.sort(dim=-1).values
    del torch_scores

    def set_row_fraction(actual: torch.Tensor, expected: torch.Tensor) -> float:
        return float(actual.eq(expected).all(dim=-1).float().mean().item())

    def set_member_recall(actual: torch.Tensor, expected: torch.Tensor) -> float:
        matches = actual.unsqueeze(-1).eq(expected.unsqueeze(-2)).any(dim=-1)
        return float(matches.float().mean().item())
    state_k = route_q = state_reference = None
    if not args.hier_only:
        state_k = torch.randn(
            args.batch,
            args.kv_heads,
            args.state_len,
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        ).mul_(counts.to(torch.bfloat16)).contiguous()
        route_q = (
            q.float()
            / q.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        ).to(torch.bfloat16).contiguous()
        state_logits = torch.matmul(
            route_q.reshape(
                args.batch,
                args.kv_heads,
                args.query_heads // args.kv_heads,
                args.query_len,
                args.head_dim,
            ),
            (state_k / counts.to(torch.bfloat16)).transpose(-1, -2).unsqueeze(2),
        ).reshape(args.batch, args.query_heads, args.query_len, args.state_len)
        state_scores = (
            (state_logits.to(torch.bfloat16) * float(args.head_dim**-0.5))
            .to(torch.bfloat16)
            .float()
            + counts.clamp_min(1).log()
            .unsqueeze(2)
            .expand(-1, -1, args.query_heads // args.kv_heads, -1, -1)
            .reshape(args.batch, args.query_heads, 1, args.state_len)
        )
        state_scores[..., 0] = float("-inf")
        state_reference = state_scores.topk(args.topk, dim=-1).indices

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
        () if args.grouped_hier_only else configs[:1] if args.hier_only else configs
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
            "route_set_torch_exact": bool(torch.equal(normalized, torch_reference)),
            "route_set_torch_row_fraction": set_row_fraction(
                normalized, torch_reference
            ),
            "route_set_torch_member_recall": set_member_recall(
                normalized, torch_reference
            ),
            "route_order_exact": bool(torch.equal(slots, reference_order)),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    grouped_slots = None
    if args.topk <= 8 and not args.hier_only:
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
            "route_set_exact": (
                bool(torch.equal(grouped_slots.sort(dim=-1).values, reference))
                if reference is not None
                else None
            ),
            "route_set_torch_exact": bool(
                torch.equal(grouped_slots.sort(dim=-1).values, torch_reference)
            ),
            "route_set_torch_row_fraction": set_row_fraction(
                grouped_slots.sort(dim=-1).values, torch_reference
            ),
            "route_set_torch_member_recall": set_member_recall(
                grouped_slots.sort(dim=-1).values, torch_reference
            ),
            "route_order_exact": (
                bool(torch.equal(grouped_slots, reference_order))
                if reference_order is not None
                else None
            ),
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
            (16, 1024, 1, 2),
            (32, 1024, 2, 2),
            (4, 2048, 2, 2),
            (8, 2048, 2, 2),
            (16, 2048, 1, 2),
            (2, 4096, 2, 2),
            (4, 4096, 2, 2),
        )
    if args.grouped_hier_only:
        hierarchical_configs = ((8, 1024, 2, 2),)
    for block_m, block_n, tile_warps, reduce_warps in (
        hierarchical_configs if args.topk in (2, 3, 4, 8) else ()
    ):
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
        exact = (
            bool(torch.equal(normalized, reference))
            if reference is not None
            else None
        )
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
            "route_set_torch_exact": bool(torch.equal(normalized, torch_reference)),
            "route_set_reference_row_fraction": (
                set_row_fraction(normalized, reference)
                if reference is not None
                else None
            ),
            "route_set_reference_member_recall": (
                set_member_recall(normalized, reference)
                if reference is not None
                else None
            ),
            "route_set_grouped_exact": bool(
                grouped_slots is not None
                and torch.equal(
                    normalized,
                    grouped_slots.sort(dim=-1).values,
                )
            ),
            "route_set_grouped_row_fraction": (
                set_row_fraction(
                    normalized,
                    grouped_slots.sort(dim=-1).values,
                )
                if grouped_slots is not None
                else None
            ),
            "route_set_grouped_member_recall": (
                set_member_recall(
                    normalized,
                    grouped_slots.sort(dim=-1).values,
                )
                if grouped_slots is not None
                else None
            ),
            "route_order_grouped_exact": bool(
                grouped_slots is not None and torch.equal(slots, grouped_slots)
            ),
            "route_set_torch_row_fraction": set_row_fraction(
                normalized, torch_reference
            ),
            "route_set_torch_member_recall": set_member_recall(
                normalized, torch_reference
            ),
            "route_order_exact": (
                bool(torch.equal(slots, reference_order))
                if reference_order is not None
                else None
            ),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    direct_buffers = (
        new_route_buffers(
            route_q,
            state_capacity=args.state_len,
            query_capacity=args.query_len,
        )
        if not args.hier_only
        else None
    )
    direct_configs = (
        (16, 64, 4, 4),
        (16, 128, 4, 4),
        (16, 256, 4, 4),
        (32, 64, 4, 2),
        (32, 128, 4, 2),
        (32, 256, 4, 2),
        (64, 64, 4, 2),
        (64, 128, 4, 2),
        (64, 256, 4, 2),
        (128, 64, 8, 2),
        (128, 128, 8, 2),
        (8, 256, 4, 2),
        (8, 512, 4, 2),
    )
    for block_m, block_n, tile_warps, reduce_warps in (
        direct_configs if not (args.hier_only or args.grouped_hier_only) else ()
    ):
        label = f"direct_state_m{block_m}_n{block_n}_tw{tile_warps}_rw{reduce_warps}"

        def invoke_direct() -> torch.Tensor:
            return route_top8_state_grouped(
                route_q,
                state_k,
                counts,
                direct_buffers,
                kv_group_size=args.query_heads // args.kv_heads,
                scale=float(args.head_dim**-0.5),
                topk=args.topk,
                state_len=args.state_len,
                protected_len=1,
                reorder_like_torch=True,
                block_m=block_m,
                block_n=block_n,
                num_warps=tile_warps,
                reduce_num_warps=reduce_warps,
            )

        direct_slots = invoke_direct()
        for _ in range(args.warmup - 1):
            direct_slots = invoke_direct()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            direct_slots = invoke_direct()
            end.record()
        torch.cuda.synchronize()
        times_us = [
            start.elapsed_time(end) * 1_000.0
            for start, end in zip(starts, ends, strict=True)
        ]
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
            "route_set_exact": bool(
                torch.equal(
                    direct_slots.sort(dim=-1).values,
                    state_reference.sort(dim=-1).values,
                )
            ),
            "route_order_exact": bool(torch.equal(direct_slots, state_reference)),
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
