#!/usr/bin/env python3
"""Microbenchmark compact Kimi routed-leaf prefill geometry."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from lod_attention.kernels.paged_cache import append_virtual_paged_kv
from lod_attention.kernels.paged_prefill import paged_leaf_attention
from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=16384)
    parser.add_argument("--leaves", type=int, default=98304)
    parser.add_argument("--states", type=int, default=5016)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--block-m", type=int, nargs="+", default=(4, 8, 16))
    parser.add_argument("--block-n", type=int, nargs="+", default=(16,))
    parser.add_argument("--warps", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--routed-projection", action="store_true")
    parser.add_argument(
        "--preexpanded",
        action="store_true",
        help="time the production Kimi path that expands every leaf once",
    )
    parser.add_argument("--identity-projection", action="store_true")
    args = parser.parse_args()
    if args.routed_projection and args.preexpanded:
        raise ValueError("choose either routed or preexpanded projection")

    torch.manual_seed(29)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = 16
    page_capacity = math.ceil(args.leaves / page_size) + args.states
    maximum_slot_pages = math.ceil(args.leaves / page_size)
    root_capacity = math.ceil(maximum_slot_pages / 64)

    query_dim = 192 if args.routed_projection or args.preexpanded else 576
    q = torch.randn(1, args.heads, args.queries, query_dim, dtype=dtype, device=device)
    leaf_k = torch.randn(1, 1, args.leaves, 576, dtype=dtype, device=device)
    leaf_v = leaf_k[..., :512]
    owners = torch.randint(
        args.states,
        (1, 1, args.leaves),
        dtype=torch.int32,
        device=device,
    )
    top_slots = torch.randint(
        args.states,
        (1, args.heads, args.queries, args.routes),
        dtype=torch.int64,
        device=device,
    )
    page_indices = torch.full(
        (1, 1, page_capacity, page_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    slot_pages = torch.full(
        (1, 1, args.states, root_capacity),
        -1,
        dtype=torch.int32,
        device=device,
    )
    overflow_page_keys = torch.full(
        (1, 1, 1), -1, dtype=torch.int32, device=device
    )
    overflow_page_values = torch.full(
        (1, 1, page_capacity, 64), -1, dtype=torch.int32, device=device
    )
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    overflow_flag = torch.zeros((), dtype=torch.int32, device=device)
    slot_lengths = torch.zeros(
        (1, 1, args.states), dtype=torch.int32, device=device
    )
    next_page = torch.zeros((1, 1), dtype=torch.int32, device=device)
    append_virtual_paged_kv(
        leaf_k,
        leaf_v,
        0,
        owners,
        page_indices,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        slot_lengths,
        next_page,
        None,
        None,
        None,
        hash_probes=-1,
    )
    torch.cuda.synchronize()
    if int(overflow_flag.item()):
        raise RuntimeError("synthetic page directory overflowed")

    w_uk_t = torch.randn(
        args.heads, 128, 512, dtype=dtype, device=device
    ) / math.sqrt(512)
    w_uv = torch.randn(
        args.heads, 512, 128, dtype=dtype, device=device
    ) / math.sqrt(512)
    if args.identity_projection:
        w_uk_t.zero_()
        w_uv.zero_()
        diagonal = torch.arange(128, device=device)
        w_uk_t[:, diagonal, diagonal] = 1
        w_uv[:, diagonal, diagonal] = 1
    routed_leaf_k = (
        leaf_k.expand(-1, args.heads, -1, -1)
        if args.routed_projection
        else leaf_k
    )
    routed_leaf_v = (
        leaf_v.expand(-1, args.heads, -1, -1)
        if args.routed_projection
        else leaf_v
    )

    results: list[dict[str, float | int]] = []
    for block_m in args.block_m:
        for block_n in args.block_n:
            for warps in args.warps:
                results.append(
                    run_configuration(
                        args=args,
                        block_m=block_m,
                        block_n=block_n,
                        warps=warps,
                        q=q,
                        leaf_k=leaf_k,
                        leaf_v=leaf_v,
                        routed_leaf_k=routed_leaf_k,
                        routed_leaf_v=routed_leaf_v,
                        w_uk_t=w_uk_t,
                        w_uv=w_uv,
                        slot_pages=slot_pages,
                        overflow_page_keys=overflow_page_keys,
                        overflow_page_values=overflow_page_values,
                        overflow_used=overflow_used,
                        slot_lengths=slot_lengths,
                        top_slots=top_slots,
                        page_indices=page_indices,
                        query_dim=query_dim,
                    )
                )
    print(json.dumps(sorted(results, key=lambda item: item["median_ms"]), indent=2))


def run_configuration(
    *,
    args: argparse.Namespace,
    block_m: int,
    block_n: int,
    warps: int,
    q: torch.Tensor,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    routed_leaf_k: torch.Tensor,
    routed_leaf_v: torch.Tensor,
    w_uk_t: torch.Tensor,
    w_uv: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor,
    page_indices: torch.Tensor,
    query_dim: int,
) -> dict[str, float | int | None | list[float]]:
            buffers: dict[str, torch.Tensor] = {}

            def run() -> None:
                if args.preexpanded:
                    attention_k, attention_v = expand_kimi_leaf_kv(
                        leaf_k, w_uk_t, w_uv, buffers=buffers
                    )
                else:
                    attention_k, attention_v = routed_leaf_k, routed_leaf_v
                return paged_leaf_attention(
                    q,
                    attention_k,
                    attention_v,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    slot_lengths,
                    top_slots,
                    page_indices=page_indices,
                    kimi_w_uk_t=(w_uk_t if args.routed_projection else None),
                    kimi_w_uv=(w_uv if args.routed_projection else None),
                    kv_group_size=(
                        1
                        if args.routed_projection or args.preexpanded
                        else args.heads
                    ),
                    active_slots=args.states,
                    scale=query_dim**-0.5,
                    hash_probes=-1,
                    block_m=block_m,
                    block_n=block_n,
                    num_warps=warps,
                    reduce_num_warps=1,
                    reduce_routes=True,
                    buffers=buffers,
                )

            run()
            torch.cuda.synchronize()
            maximum_error = None
            mean_error = None
            per_head_mean_error = None
            if args.preexpanded:
                actual_k, actual_v = expand_kimi_leaf_kv(
                    leaf_k, w_uk_t, w_uv, buffers=buffers
                )
                latent = leaf_k[:, 0, :, :512].reshape(args.leaves, 512)
                expected_nope = torch.mm(
                    latent,
                    w_uk_t.permute(2, 0, 1).reshape(512, args.heads * 128),
                ).view(1, args.leaves, args.heads, 128)
                expected_v = torch.mm(
                    latent,
                    w_uv.permute(1, 0, 2).reshape(512, args.heads * 128),
                ).view(1, args.leaves, args.heads, 128)
                expected_k = torch.cat(
                    (
                        expected_nope,
                        leaf_k[:, 0, :, None, 512:].expand(
                            -1, -1, args.heads, -1
                        ),
                    ),
                    dim=-1,
                ).permute(0, 2, 1, 3)
                torch.testing.assert_close(actual_k, expected_k)
                torch.testing.assert_close(
                    actual_v, expected_v.permute(0, 2, 1, 3)
                )
            elif args.routed_projection:
                reference_k, reference_v = expand_kimi_leaf_kv(
                    leaf_k, w_uk_t, w_uv
                )
                reference_output, _ = paged_leaf_attention(
                    q,
                    reference_k,
                    reference_v,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    slot_lengths,
                    top_slots,
                    page_indices=page_indices,
                    kv_group_size=1,
                    active_slots=args.states,
                    scale=query_dim**-0.5,
                    hash_probes=-1,
                    block_m=block_m,
                    block_n=block_n,
                    num_warps=warps,
                    reduce_num_warps=1,
                    reduce_routes=True,
                )
                projected_output, _ = run()
                maximum_error = float(
                    (projected_output.float() - reference_output.float()).abs().max()
                )
                mean_error = float(
                    (projected_output.float() - reference_output.float()).abs().mean()
                )
                per_head_mean_error = (
                    (projected_output.float() - reference_output.float())
                    .abs()
                    .mean(dim=(0, 2, 3))
                    .tolist()
                )
            timings = []
            for _ in range(args.iterations):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                run()
                end.record()
                end.synchronize()
                timings.append(begin.elapsed_time(end))
            return {
                "block_m": block_m,
                "block_n": block_n,
                "warps": warps,
                "median_ms": statistics.median(timings),
                "minimum_ms": min(timings),
                "maximum_error": maximum_error,
                "mean_error": mean_error,
                "per_head_mean_error": per_head_mean_error,
            }


if __name__ == "__main__":
    main()
