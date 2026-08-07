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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--native-gqa", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    batch = args.batch_size
    query_heads, kv_heads, dim = 8, 2, 128
    length = args.sequence_length
    q = torch.randn(
        batch, query_heads, length, dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, kv_heads, length, dim, device=device, dtype=torch.bfloat16
    )
    if not args.native_gqa:
        k = k.repeat_interleave(query_heads // kv_heads, dim=1)
    v = torch.randn_like(k)

    def attention():
        return torch.ops.aten._scaled_dot_product_flash_attention.default(
            q,
            k,
            v,
            0.0,
            True,
            False,
            scale=1.0 / math.sqrt(dim),
        )

    for _ in range(3):
        output, lse, *_ = attention()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(args.repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output, lse, *_ = attention()
        end.record()
        end.synchronize()
        elapsed.append(float(begin.elapsed_time(end)))
    elapsed.sort()
    print(
        json.dumps(
            {
                "batch_size": batch,
                "sequence_length": length,
                "native_gqa": args.native_gqa,
                "median_ms": elapsed[len(elapsed) // 2],
                "min_ms": elapsed[0],
                "output_shape": list(output.shape),
                "lse_shape": list(lse.shape),
                "finite": bool(torch.isfinite(output).all() and torch.isfinite(lse).all()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
