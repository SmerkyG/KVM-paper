#!/usr/bin/env python3
"""Measure state-only BF16 cache conversion against ordinary LOD prefill."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import KernelRecursivePagedLODAttention


def _engine(config: PagedLODConfig, query_heads: int, kv_heads: int, dim: int):
    return KernelRecursivePagedLODAttention(
        config,
        query_heads=query_heads,
        key_value_heads=kv_heads,
        scale=dim**-0.5,
        default_open_count=8,
    ).cuda()


def _measure(function, repetitions: int) -> list[float]:
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA or ROCm GPU")

    query_heads = 8
    kv_heads = 2
    dimension = 128
    config = PagedLODConfig(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16.0,
        state_min_size=256,
        protected_prefix=1,
        max_routes=8,
        page_size=16,
        kv_bits=args.kv_bits,
        quant_group_size=32,
    )
    query = torch.randn(
        args.batch_size,
        query_heads,
        args.length,
        dimension,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.batch_size,
        kv_heads,
        args.length,
        dimension,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    prefill_engine = _engine(config, query_heads, kv_heads, dimension)
    conversion_engine = _engine(config, query_heads, kv_heads, dimension)

    # Compile and allocate both paths before timing.
    prefill_engine(query, key, value, use_cache=True)
    conversion_engine.build_cache_from_bf16(key, value)
    prefill_ms = _measure(
        lambda: prefill_engine(query, key, value, use_cache=True),
        args.repetitions,
    )
    conversion_ms = _measure(
        lambda: conversion_engine.build_cache_from_bf16(key, value),
        args.repetitions,
    )
    result = {
        "length": args.length,
        "batch_size": args.batch_size,
        "kv_bits": args.kv_bits,
        "repetitions": args.repetitions,
        "ordinary_lod_prefill_ms": prefill_ms,
        "cache_conversion_ms": conversion_ms,
        "ordinary_lod_prefill_median_ms": statistics.median(prefill_ms),
        "cache_conversion_median_ms": statistics.median(conversion_ms),
        "conversion_fraction": (
            statistics.median(conversion_ms) / statistics.median(prefill_ms)
        ),
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
