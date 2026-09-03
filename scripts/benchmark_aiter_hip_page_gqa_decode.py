#!/usr/bin/env python3
"""Benchmark AITER's HIP MFMA paged-decode kernel for GQA16 index lists."""

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
    parser.add_argument("--page-sizes", type=int, nargs="+", default=(1, 16, 64))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from aiter.ops.attention import paged_attention_rocm

    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = args.batch_size
    query_heads = 32
    kv_heads = 2
    gqa = query_heads // kv_heads
    head_dim = 128
    vector = 16 // torch.empty((), dtype=dtype).element_size()
    q = torch.randn(batch, query_heads, head_dim, device=device, dtype=dtype)
    out = torch.empty_like(q)
    maximum_length = max(args.lengths)
    partitions = (maximum_length + 255) // 256
    exp_sums = torch.empty(
        batch, query_heads, partitions, device=device, dtype=torch.float32
    )
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(
        batch,
        query_heads,
        partitions,
        head_dim,
        device=device,
        dtype=dtype,
    )
    scale_tensor = torch.ones(1, device=device, dtype=torch.float32)
    records = []

    for page_size in args.page_sizes:
        maximum_pages = (maximum_length + page_size - 1) // page_size
        key_cache = torch.randn(
            batch * maximum_pages,
            kv_heads,
            head_dim // vector,
            page_size,
            vector,
            device=device,
            dtype=dtype,
        )
        value_cache = torch.randn(
            batch * maximum_pages,
            kv_heads,
            head_dim,
            page_size,
            device=device,
            dtype=dtype,
        )
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
                paged_attention_rocm(
                    out=out,
                    exp_sums=exp_sums,
                    max_logits=max_logits,
                    tmp_out=tmp_out,
                    query=q,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    num_kv_heads=kv_heads,
                    scale=head_dim**-0.5,
                    block_tables=table,
                    context_lens=lengths,
                    block_size=page_size,
                    max_context_len=length,
                    alibi_slopes=None,
                    kv_cache_dtype="auto",
                    k_scale=scale_tensor,
                    v_scale=scale_tensor,
                    fp8_out_scale=None,
                    partition_size=256,
                    mtp=1,
                    q_scale=None,
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
            if not bool(torch.isfinite(out).all().item()):
                raise AssertionError("AITER HIP paged attention returned non-finite output")
            records.append(
                {
                    "page_size": page_size,
                    "length": length,
                    "median_us": 1000.0 * statistics.median(samples),
                    "minimum_us": 1000.0 * min(samples),
                }
            )
        del key_cache, value_cache
        torch.cuda.empty_cache()

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "geometry": {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "gqa": gqa,
            "head_dim": head_dim,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
