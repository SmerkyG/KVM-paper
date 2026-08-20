#!/usr/bin/env python3
"""Measure AITER's MFMA paged-decode kernel as an LOD expert kernel.

One LOD expert is represented as one MQA sequence: routed query rows become
query heads and the expert's leaf list becomes the single paged KV stream.
This is deliberately a kernel-feasibility probe; it does not include the
packing needed for variable-width experts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


VENDOR_AITER = Path("/home/dan/subusers/agent/vendor/aiter")
if str(VENDOR_AITER) not in sys.path:
    # Keep the installed AITER package first, but expose its source-only csrc
    # package used by the ragged paged-attention JIT wrapper.
    sys.path.append(str(VENDOR_AITER))

from csrc.cpp_itfs.pa.pa_ragged import paged_attention_ragged  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-rows", type=int, nargs="+", default=[16])
    parser.add_argument("--key-lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--target-pairs", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--min-experts", type=int, default=128)
    parser.add_argument("--max-experts", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_case(
    *,
    query_rows: int,
    key_length: int,
    target_pairs: int,
    min_experts: int,
    max_experts: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    head_dim = 128
    page_size = 16
    partition_size = 256
    pages_per_expert = math.ceil(key_length / page_size)
    experts = max(min_experts, target_pairs // (query_rows * key_length))
    experts = min(max_experts, experts)
    total_pages = experts * pages_per_expert

    generator = torch.Generator(device=device).manual_seed(17 + key_length + query_rows)
    query = torch.randn(
        experts,
        query_rows,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    key_cache = torch.randn(
        total_pages,
        page_size,
        1,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    value_cache = torch.randn(
        total_pages,
        page_size,
        1,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    output = torch.empty_like(query)
    page_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_indptr = (
        torch.arange(experts + 1, dtype=torch.int32, device=device)
        * pages_per_expert
    )
    last_page_lens = torch.full(
        (experts,),
        key_length - (pages_per_expert - 1) * page_size,
        dtype=torch.int32,
        device=device,
    )
    max_partitions = math.ceil(key_length / partition_size)
    workspace_bytes = (
        experts
        * query_rows
        * max_partitions
        * (2 * torch.tensor([], dtype=torch.float32).element_size() + head_dim * 2)
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    unit_scale = torch.ones(1, dtype=torch.float32, device=device)
    scale = head_dim**-0.5

    def invoke() -> None:
        paged_attention_ragged(
            output,
            workspace,
            query,
            key_cache,
            value_cache,
            scale,
            kv_indptr,
            page_indices,
            last_page_lens,
            page_size,
            max_partitions,
            None,
            "auto",
            "NHD",
            0.0,
            unit_scale,
            unit_scale,
            partition_size=partition_size,
        )

    # This first invocation may JIT-compile a GQA-ratio specialization.
    invoke()
    torch.cuda.synchronize(device)

    # Validate one expert before timing the bulk launch.
    q_ref = query[0].float()
    k_ref = key_cache[:pages_per_expert, :, 0].reshape(-1, head_dim)[:key_length].float()
    v_ref = value_cache[:pages_per_expert, :, 0].reshape(-1, head_dim)[:key_length].float()
    reference = torch.softmax((q_ref @ k_ref.T) * scale, dim=-1) @ v_ref
    maximum_error = float((output[0].float() - reference).abs().max().item())

    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize(device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        invoke()
    end.record()
    torch.cuda.synchronize(device)
    milliseconds = float(begin.elapsed_time(end)) / repeats

    useful_pairs = experts * query_rows * key_length
    useful_flops = 4 * useful_pairs * head_dim
    return {
        "experts": experts,
        "query_rows": query_rows,
        "key_length": key_length,
        "max_partitions": max_partitions,
        "milliseconds": milliseconds,
        "useful_qk_pairs": useful_pairs,
        "useful_pair_giga_per_second": useful_pairs / milliseconds / 1.0e6,
        "useful_tflops": useful_flops / (milliseconds * 1.0e9),
        "maximum_output_error": maximum_error,
    }


def main() -> None:
    args = parse_args()
    records = [
        run_case(
            query_rows=query_rows,
            key_length=key_length,
            target_pairs=args.target_pairs,
            min_experts=args.min_experts,
            max_experts=args.max_experts,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for query_rows in args.query_rows
        for key_length in args.key_lengths
    ]
    result = {
        "device": torch.cuda.get_device_name(0),
        "records": records,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
