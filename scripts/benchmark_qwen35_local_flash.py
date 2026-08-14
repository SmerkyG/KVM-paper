#!/usr/bin/env python3
"""Measure the dense causal local-attention primitive for LOD prefill."""

from __future__ import annotations

import argparse
import json
import math

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--query-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--native-gqa", action="store_true")
    parser.add_argument("--aiter-varlen", action="store_true")
    parser.add_argument("--aiter-dense", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    batch = args.batch_size
    query_heads, kv_heads, dim = 8, 2, 256
    length = args.sequence_length
    query_length = length if args.query_length is None else args.query_length
    if not 0 < query_length <= length:
        raise ValueError("query length must lie in (0, sequence length]")
    query = torch.randn(
        batch, query_heads, query_length, dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, kv_heads, length, dim, device=device, dtype=torch.bfloat16
    )
    if not args.native_gqa:
        k = k.repeat_interleave(query_heads // kv_heads, dim=1)
    v = torch.randn_like(k)

    if args.aiter_varlen and args.aiter_dense:
        raise ValueError("select at most one AITER attention path")
    if args.aiter_dense:
        from aiter.ops.mha import flash_attn_func

        dense_q = query.permute(0, 2, 1, 3).contiguous()
        dense_k = k.permute(0, 2, 1, 3).contiguous()
        dense_v = v.permute(0, 2, 1, 3).contiguous()

        def attention():
            return flash_attn_func(
                dense_q,
                dense_k,
                dense_v,
                softmax_scale=1.0 / math.sqrt(dim),
                causal=True,
                return_lse=True,
            )

    elif args.aiter_varlen:
        from aiter.ops.mha import flash_attn_varlen_func

        cumulative_q = torch.arange(
            batch + 1, device=device, dtype=torch.int32
        ) * query_length
        cumulative_k = torch.arange(
            batch + 1, device=device, dtype=torch.int32
        ) * length

        def attention():
            packed_q = query.permute(0, 2, 1, 3).contiguous().flatten(0, 1)
            packed_k = k.permute(0, 2, 1, 3).contiguous().flatten(0, 1)
            packed_v = v.permute(0, 2, 1, 3).contiguous().flatten(0, 1)
            output, lse = flash_attn_varlen_func(
                packed_q,
                packed_k,
                packed_v,
                cumulative_q,
                cumulative_k,
                query_length,
                length,
                softmax_scale=1.0 / math.sqrt(dim),
                causal=True,
                return_lse=True,
            )
            return output, lse

    else:
        prefix = length - query_length
        padded_query = torch.cat(
            (
                query.new_zeros(batch, query_heads, prefix, dim),
                query,
            ),
            dim=2,
        )

        def attention():
            output, lse, *_ = (
                torch.ops.aten._scaled_dot_product_flash_attention.default(
                    padded_query,
                    k,
                    v,
                    0.0,
                    True,
                    False,
                    scale=1.0 / math.sqrt(dim),
                )
            )
            return output[..., prefix:, :], lse[..., prefix:]

    for _ in range(3):
        output, lse = attention()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(args.repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output, lse = attention()
        end.record()
        end.synchronize()
        elapsed.append(float(begin.elapsed_time(end)))
    elapsed.sort()
    parity = {}
    if args.aiter_varlen or args.aiter_dense:
        prefix = length - query_length
        padded_query = torch.cat(
            (
                query.new_zeros(batch, query_heads, prefix, dim),
                query,
            ),
            dim=2,
        )
        reference_output, reference_lse, *_ = (
            torch.ops.aten._scaled_dot_product_flash_attention.default(
                padded_query,
                k,
                v,
                0.0,
                True,
                False,
                scale=1.0 / math.sqrt(dim),
            )
        )
        if args.aiter_dense:
            actual_output = output.permute(0, 2, 1, 3)
            actual_lse = lse
        else:
            actual_output = output.view(
                batch, query_length, query_heads, dim
            ).permute(0, 2, 1, 3)
            actual_lse = lse.view(query_heads, batch, query_length).permute(1, 0, 2)
        parity = {
            "output_max_abs": float(
                (actual_output - reference_output[..., prefix:, :]).abs().max()
            ),
            "output_mean_abs": float(
                (actual_output - reference_output[..., prefix:, :]).abs().mean()
            ),
            "lse_max_abs": float(
                (actual_lse - reference_lse[..., prefix:]).abs().max()
            ),
            "lse_mean_abs": float(
                (actual_lse - reference_lse[..., prefix:]).abs().mean()
            ),
        }
    print(
        json.dumps(
            {
                "batch_size": batch,
                "sequence_length": length,
                "query_length": query_length,
                "native_gqa": args.native_gqa,
                "aiter_varlen": args.aiter_varlen,
                "aiter_dense": args.aiter_dense,
                "median_ms": elapsed[len(elapsed) // 2],
                "min_ms": elapsed[0],
                "output_shape": list(output.shape),
                "lse_shape": list(lse.shape),
                "finite": bool(torch.isfinite(output).all() and torch.isfinite(lse).all()),
                "parity": parity,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
