#!/usr/bin/env python3
"""Benchmark one AITER GQA16 decode call over an indexed page-size-1 list."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=(4096, 8192, 16384, 32768, 65536)
    )
    parser.add_argument("--page-sizes", type=int, nargs="+", default=(1, 64))
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from aiter.ops.triton.unified_attention import unified_attention

    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = args.batch_size
    query_heads = args.query_heads
    kv_heads = args.kv_heads
    head_dim = args.head_dim
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    maximum_length = max(args.lengths)
    q = torch.randn(batch, query_heads, head_dim, device=device, dtype=dtype)
    out = torch.empty_like(q)
    cu_q = torch.arange(batch + 1, device=device, dtype=torch.int32)
    records = []

    for page_size in args.page_sizes:
        maximum_pages = (maximum_length + page_size - 1) // page_size
        k = torch.randn(
            batch * maximum_pages,
            page_size,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v = torch.randn_like(k)
        full_table = torch.arange(
            batch * maximum_pages, device=device, dtype=torch.int32
        ).reshape(batch, maximum_pages)
        for length in args.lengths:
            pages = (length + page_size - 1) // page_size
            table = full_table[:, :pages]
            lengths = torch.full(
                (batch,), length, device=device, dtype=torch.int32
            )

            def attention() -> None:
                unified_attention(
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    cu_seqlens_q=cu_q,
                    max_seqlen_q=1,
                    seqused_k=lengths,
                    max_seqlen_k=length,
                    softmax_scale=head_dim**-0.5,
                    causal=True,
                    window_size=(-1, -1),
                    block_table=table,
                    softcap=0.0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                )

            for _ in range(args.warmup):
                attention()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                attention()
                end.record()
                end.synchronize()
                samples.append(float(begin.elapsed_time(end)))
            records.append(
                {
                    "page_size": page_size,
                    "length": length,
                    "median_us": 1000.0 * statistics.median(samples),
                    "minimum_us": 1000.0 * min(samples),
                }
            )
        del k, v
        torch.cuda.empty_cache()

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "geometry": {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "gqa": query_heads // kv_heads,
            "head_dim": head_dim,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
