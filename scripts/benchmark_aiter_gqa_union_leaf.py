#!/usr/bin/env python3
"""Verify and time actual AITER page-size-1 attention for GQA leaf unions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import triton

from csrc.cpp_itfs.pa.pa_v1 import paged_attention_v1

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.kernels.gqa_union_leaf_attention import (
    _gqa_union_indexed_attention_kernel,
)


PARTITION_SIZE = 256


def _elapsed_ms(fn: Callable[[], object], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def _make_case(
    *,
    batch: int,
    kv_heads: int,
    group_size: int,
    context: int,
    max_context: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sequences = batch * kv_heads
    blocks = sequences * context
    keys = torch.randn(blocks, head_dim, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    # AITER PA-v1's NHD layout keeps K and V identical and supports a one-token
    # physical page: [physical block, page token, KV head, head dimension].
    key_cache = keys.reshape(blocks, 1, 1, head_dim)
    value_cache = values.reshape(blocks, 1, 1, head_dim)
    block_tables = torch.zeros(
        sequences, max_context, device=device, dtype=torch.int32
    )
    block_tables[:, :context] = (
        torch.arange(context, device=device, dtype=torch.int32)[None, :]
        + torch.arange(sequences, device=device, dtype=torch.int32)[:, None]
        * context
    )
    context_lens = torch.full(
        (sequences,), context, device=device, dtype=torch.int32
    )
    query = torch.randn(
        sequences, group_size, head_dim, device=device, dtype=dtype
    )
    partitions = math.ceil(max_context / PARTITION_SIZE)
    statistics_elements = sequences * group_size * partitions
    tmp_output_bytes = sequences * group_size * partitions * head_dim * 2
    workspace = torch.empty(
        2 * statistics_elements * 4 + tmp_output_bytes,
        device=device,
        dtype=torch.uint8,
    )
    statistics = workspace[: 2 * statistics_elements * 4].view(torch.float32)
    return {
        "query": query,
        "keys": keys,
        "values": values,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "context_lens": context_lens,
        "output": torch.empty_like(query),
        "workspace": workspace,
        # PA-v1 documents this workspace prefix even though LSE is not a
        # return value: first per-partition exp sums, then maxima, both FP32.
        "exp_sums": statistics[:statistics_elements].view(
            sequences, group_size, partitions
        ),
        "max_logits": statistics[statistics_elements:].view(
            sequences, group_size, partitions
        ),
        "scale": torch.tensor(1.0, device=device, dtype=torch.float32),
        "triton_query": query.view(batch, kv_heads * group_size, 1, head_dim),
        "triton_keys": keys.view(batch, kv_heads, context, head_dim),
        "triton_values": values.view(batch, kv_heads, context, head_dim),
        "triton_indices": (
            torch.arange(context, device=device, dtype=torch.int32)[None, None, :]
            + torch.arange(
                batch * kv_heads, device=device, dtype=torch.int32
            ).view(batch, kv_heads, 1)
            * context
        ).contiguous(),
        "triton_lengths": context_lens.view(batch, kv_heads),
        "triton_output": torch.empty(
            batch,
            kv_heads * group_size,
            1,
            head_dim,
            device=device,
            dtype=dtype,
        ),
        "triton_lse": torch.empty(
            batch,
            kv_heads * group_size,
            1,
            device=device,
            dtype=torch.float32,
        ),
    }


def _run_aiter(case: dict[str, torch.Tensor], scale: float) -> torch.Tensor:
    return paged_attention_v1(
        case["output"],
        case["workspace"],
        case["query"],
        case["key_cache"],
        case["value_cache"],
        scale,
        case["block_tables"],
        None,
        case["context_lens"],
        int(case["block_tables"].size(1)),
        None,
        "auto",
        "NHD",
        0.0,
        case["scale"],
        case["scale"],
        None,
        PARTITION_SIZE,
        1,
        sliding_window=0,
    )


def _lse_from_partitions(case: dict[str, torch.Tensor]) -> torch.Tensor:
    max_logits = case["max_logits"]
    exp_sums = case["exp_sums"]
    valid_partitions = (
        torch.arange(max_logits.size(-1), device=max_logits.device)[None, :]
        < (case["context_lens"][:, None] + PARTITION_SIZE - 1) // PARTITION_SIZE
    )
    valid_partitions = valid_partitions[:, None, :]
    masked_max_logits = torch.where(
        valid_partitions, max_logits, torch.full_like(max_logits, -torch.inf)
    )
    maximum = masked_max_logits.max(dim=-1).values
    denominator = (
        torch.where(valid_partitions, exp_sums, torch.zeros_like(exp_sums))
        * torch.exp(masked_max_logits - maximum.unsqueeze(-1))
    ).sum(dim=-1)
    return maximum + torch.log(denominator)


def _run_triton(case: dict[str, torch.Tensor], scale: float) -> None:
    q = case["triton_query"]
    keys = case["triton_keys"]
    values = case["triton_values"]
    indices = case["triton_indices"]
    lengths = case["triton_lengths"]
    output = case["triton_output"]
    lse = case["triton_lse"]
    batch, query_heads, _, head_dim = q.shape
    kv_heads = int(keys.size(1))
    group_size = query_heads // kv_heads
    _gqa_union_indexed_attention_kernel[(batch * kv_heads,)](
        q,
        keys,
        values,
        indices,
        lengths,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(3),
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        lse.stride(0),
        lse.stride(1),
        QUERY_HEADS=query_heads,
        KV_HEADS=kv_heads,
        KV_GROUP_SIZE=group_size,
        UNION_GROUP_SIZE=group_size,
        GROUPS_PER_KV=1,
        LOGICAL_GROUPS=kv_heads,
        LEAF_CAPACITY=int(keys.size(2)),
        TOTAL_LEAVES=int(keys.numel() // head_dim),
        HEAD_DIM=head_dim,
        BLOCK_G=triton.next_power_of_2(group_size),
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_N=128,
        SCALE_LOG2=float(scale) * math.log2(math.e),
        num_warps=4,
        waves_per_eu=1,
    )


def _verify(case: dict[str, torch.Tensor], scale: float) -> dict[str, float]:
    _run_aiter(case, scale)
    sequence = 0
    length = int(case["context_lens"][sequence])
    indices = case["block_tables"][sequence, :length].long()
    keys = case["keys"][indices].float()
    values = case["values"][indices].float()
    scores = torch.einsum("hd,nd->hn", case["query"][sequence].float(), keys)
    scores *= scale
    reference_output = torch.softmax(scores, dim=-1) @ values
    reference_lse = torch.logsumexp(scores, dim=-1)
    output = case["output"][sequence].float()
    lse = _lse_from_partitions(case)[sequence]
    return {
        "output_max_abs": float((output - reference_output).abs().max()),
        "output_mean_abs": float((output - reference_output).abs().mean()),
        "lse_max_abs": float((lse - reference_lse).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1400, 2500])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="AITER launch/cache capacity; defaults to each logical context",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-triton", action="store_true")
    args = parser.parse_args()
    scale = args.head_dim**-0.5
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "batch": args.batch,
                "kv_heads": args.kv_heads,
                "group_size": args.group_size,
                "head_dim": args.head_dim,
            }
        ),
        flush=True,
    )
    results = []
    for context in args.contexts:
        max_context = context if args.max_context is None else args.max_context
        if max_context < context:
            raise ValueError("--max-context cannot be shorter than --contexts")
        case = _make_case(
            batch=args.batch,
            kv_heads=args.kv_heads,
            group_size=args.group_size,
            context=context,
            max_context=max_context,
            head_dim=args.head_dim,
        )
        parity = _verify(case, scale)
        kernel_ms = _elapsed_ms(
            lambda: _run_aiter(case, scale), args.warmup, args.iterations
        )
        triton_ms = None
        if not args.skip_triton:
            triton_ms = _elapsed_ms(
                lambda: _run_triton(case, scale), args.warmup, args.iterations
            )
        lse_ms = _elapsed_ms(
            lambda: _lse_from_partitions(case), args.warmup, args.iterations
        )
        row = {
            "context": context,
            "max_context": max_context,
            "sequences": args.batch * args.kv_heads,
            "query_heads": args.batch * args.kv_heads * args.group_size,
            "partitions": math.ceil(max_context / PARTITION_SIZE),
            "aiter_ms": kernel_ms,
            "triton_ms": triton_ms,
            "triton_over_aiter": (
                None if triton_ms is None else triton_ms / kernel_ms
            ),
            "pytorch_lse_ms": lse_ms,
            **parity,
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    print(json.dumps({"results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
