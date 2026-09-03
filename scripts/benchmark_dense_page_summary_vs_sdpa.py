#!/usr/bin/env python3
"""Compare dense page-summary attention with SDPA at identical geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import triton

from model.kernels.paged_leaf_attention import (
    _dense_page_regular_attention_kernel,
    _dense_page_summary_attention_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--key-lengths", type=int, nargs="+", default=(256, 512, 768))
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-ms", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, nargs="+", default=(4,))
    parser.add_argument("--include-fused-topk", action="store_true")
    parser.add_argument("--include-normalize-and-sdpa", action="store_true")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
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
    cache_indices = torch.arange(args.batch_size, dtype=torch.long, device=device)
    records = []
    for key_length in args.key_lengths:
        k = torch.randn(
            args.batch_size,
            args.kv_heads,
            key_length,
            args.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        v = torch.randn_like(k)
        counts = torch.ones(
            args.batch_size,
            args.kv_heads,
            key_length,
            dtype=torch.int32,
            device=device,
        )
        next_page = torch.full(
            (args.batch_size, args.kv_heads),
            key_length,
            dtype=torch.int32,
            device=device,
        )
        def sdpa():
            output, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
                q,
                k,
                v,
                0.0,
                False,
                False,
                scale=scale,
            )
            return output

        sdpa_ms = timed_ms(sdpa, args.warmups, args.repeats)
        sdpa_out = sdpa()
        torch.cuda.synchronize()
        pair_flops = (
            4
            * args.batch_size
            * args.query_heads
            * args.query_length
            * key_length
            * args.head_dim
        )
        records.append(
            {
                "backend": "sdpa",
                "key_length": key_length,
                "milliseconds": sdpa_ms,
                "effective_tflops": pair_flops / sdpa_ms / 1e9,
            }
        )
        if args.include_normalize_and_sdpa:
            full_counts = torch.full_like(counts, 16)
            key_sums = k * 16
            value_sums = v * 16

            def normalize_summaries():
                divisor = full_counts.unsqueeze(-1)
                return (
                    (key_sums.float() / divisor).to(torch.bfloat16),
                    (value_sums.float() / divisor).to(torch.bfloat16),
                )

            normalize_ms = timed_ms(
                normalize_summaries, args.warmups, args.repeats
            )

            def normalize_and_sdpa():
                mean_k, mean_v = normalize_summaries()
                return torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q,
                    mean_k,
                    mean_v,
                    0.0,
                    False,
                    False,
                    scale=scale,
                )[0]

            normalized_sdpa_ms = timed_ms(
                normalize_and_sdpa, args.warmups, args.repeats
            )
            records.extend(
                (
                    {
                        "backend": "normalize_page_summaries",
                        "key_length": key_length,
                        "milliseconds": normalize_ms,
                    },
                    {
                        "backend": "normalize_and_sdpa",
                        "key_length": key_length,
                        "milliseconds": normalized_sdpa_ms,
                        "effective_tflops_attention_only": (
                            pair_flops / normalized_sdpa_ms / 1e9
                        ),
                    },
                )
            )
        for block_m in args.block_ms:
            for num_warps in args.num_warps:
                group_block = triton.next_power_of_2(group)
                if block_m < group_block or block_m % group_block:
                    continue
                block_q = block_m // group_block
                out = torch.empty_like(q)
                lse = torch.empty(
                    args.batch_size,
                    args.query_heads,
                    args.query_length,
                    dtype=torch.float32,
                    device=device,
                )

                def summary():
                    return _dense_page_regular_attention_kernel[
                        (
                            args.batch_size,
                            args.kv_heads,
                            triton.cdiv(args.query_length, block_q),
                        )
                    ](
                        q,
                        cache_indices,
                        k,
                        v,
                        counts,
                        next_page,
                        out,
                        lse,
                        args.query_length,
                        QUERY_HEADS=args.query_heads,
                        KV_HEADS=args.kv_heads,
                        KV_GROUP_SIZE=group,
                        PAGE_CAPACITY=key_length,
                        HEAD_DIM=args.head_dim,
                        VALUE_DIM=args.head_dim,
                        HEAD_BLOCK_DIM=triton.next_power_of_2(args.head_dim),
                        VALUE_BLOCK_DIM=triton.next_power_of_2(args.head_dim),
                        SCALE_LOG2=scale * math.log2(math.e),
                        BLOCK_Q=block_q,
                        BLOCK_M=block_m,
                        BLOCK_N=args.block_n,
                        num_warps=num_warps,
                        waves_per_eu=1,
                    )

                summary_ms = timed_ms(summary, args.warmups, args.repeats)
                compiled = summary()
                torch.cuda.synchronize()
                metadata = getattr(compiled, "metadata", None)
                records.append(
                    {
                        "backend": "dense_page_summary",
                        "block_m": block_m,
                        "block_n": args.block_n,
                        "num_warps": num_warps,
                        "key_length": key_length,
                        "milliseconds": summary_ms,
                        "effective_tflops": pair_flops / summary_ms / 1e9,
                        "max_abs_vs_sdpa": float(
                            (out.float() - sdpa_out.float()).abs().max().item()
                        ),
                        "registers_per_thread": getattr(compiled, "n_regs", None),
                        "spills_per_thread": getattr(compiled, "n_spills", None),
                        "shared_memory_bytes": getattr(metadata, "shared", None),
                    }
                )
                if args.include_fused_topk:
                    selected = torch.empty(
                        args.batch_size,
                        args.query_heads,
                        args.query_length,
                        8,
                        dtype=torch.int32,
                        device=device,
                    )

                    def fused_summary_topk():
                        return _dense_page_summary_attention_kernel[
                            (
                                args.batch_size,
                                args.kv_heads,
                                triton.cdiv(args.query_length, block_q),
                            )
                        ](
                            q,
                            cache_indices,
                            k,
                            v,
                            counts,
                            next_page,
                            selected,
                            out,
                            lse,
                            args.query_length,
                            QUERY_HEADS=args.query_heads,
                            KV_HEADS=args.kv_heads,
                            KV_GROUP_SIZE=group,
                            PAGE_CAPACITY=key_length,
                            HEAD_DIM=args.head_dim,
                            VALUE_DIM=args.head_dim,
                            HEAD_BLOCK_DIM=triton.next_power_of_2(args.head_dim),
                            VALUE_BLOCK_DIM=triton.next_power_of_2(args.head_dim),
                            TOP_PAGES=8,
                            SCALE_LOG2=scale * math.log2(math.e),
                            GROUP_BLOCK=group_block,
                            BLOCK_Q=block_q,
                            BLOCK_M=block_m,
                            BLOCK_N=args.block_n,
                            num_warps=num_warps,
                            waves_per_eu=1,
                        )

                    fused_ms = timed_ms(
                        fused_summary_topk, args.warmups, args.repeats
                    )
                    compiled = fused_summary_topk()
                    torch.cuda.synchronize()
                    metadata = getattr(compiled, "metadata", None)
                    records.append(
                        {
                            "backend": "dense_page_summary_fused_top8",
                            "block_m": block_m,
                            "block_n": args.block_n,
                            "num_warps": num_warps,
                            "key_length": key_length,
                            "milliseconds": fused_ms,
                            "effective_tflops": pair_flops / fused_ms / 1e9,
                            "registers_per_thread": getattr(
                                compiled, "n_regs", None
                            ),
                            "spills_per_thread": getattr(
                                compiled, "n_spills", None
                            ),
                            "shared_memory_bytes": getattr(
                                metadata, "shared", None
                            ),
                        }
                    )
    result = {
        "batch_size": args.batch_size,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "query_length": args.query_length,
        "head_dim": args.head_dim,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
