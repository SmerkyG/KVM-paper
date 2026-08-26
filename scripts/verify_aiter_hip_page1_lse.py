#!/usr/bin/env python3
"""Verify that AITER HIP paged-decode exposes a usable page-size-1 LSE."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--length", type=int, default=4864)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--value-shuffled", action="store_true")
    parser.add_argument(
        "--kernel",
        choices=("legacy", "ragged"),
        default="legacy",
        help="AITER HIP page-size-1 implementation to verify.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from aiter.ops.attention import paged_attention_ragged, paged_attention_rocm

    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    sequences = args.batch_size
    query_heads = 16
    kv_heads = 1
    head_dim = 128
    page_size = 1
    partition_size = 256
    vector = 16 // torch.empty((), dtype=dtype).element_size()
    length = args.length
    max_context_length = args.max_context_length or length
    if max_context_length < length:
        raise ValueError("max-context-length cannot be shorter than length")
    partitions = (max_context_length + partition_size - 1) // partition_size

    q = torch.randn(sequences, query_heads, head_dim, device=device, dtype=dtype)
    flat_k = torch.randn(
        sequences, length, head_dim, device=device, dtype=dtype
    )
    flat_v = torch.randn_like(flat_k)
    key_cache = flat_k.view(
        sequences * length, kv_heads, head_dim // vector, page_size, vector
    )
    value_cache = (
        flat_v.view(
            sequences * length,
            kv_heads,
            head_dim // vector,
            page_size,
            vector,
        )
        if args.value_shuffled
        else flat_v.view(sequences * length, kv_heads, head_dim, page_size)
    )
    block_table = torch.zeros(
        sequences, max_context_length, device=device, dtype=torch.int32
    )
    block_table[:, :length] = torch.arange(
        sequences * length, device=device, dtype=torch.int32
    ).view(sequences, length)
    context_lens = torch.full(
        (sequences,), length, device=device, dtype=torch.int32
    )
    out = torch.empty_like(q)
    exp_sums = torch.empty(
        sequences, query_heads, partitions, device=device, dtype=torch.float32
    )
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(
        sequences,
        query_heads,
        partitions,
        head_dim,
        device=device,
        dtype=dtype,
    )
    kv_indptr = torch.arange(
        0,
        (sequences + 1) * length,
        length,
        device=device,
        dtype=torch.int32,
    )
    kv_page_indices = block_table[:, :length].reshape(-1).contiguous()
    kv_last_page_lens = torch.ones(
        sequences, device=device, dtype=torch.int32
    )
    ragged_key_cache = flat_k.reshape(
        sequences * length, page_size, kv_heads, head_dim
    )
    ragged_value_cache = flat_v.reshape(
        sequences * length, page_size, kv_heads, head_dim
    )
    workspace_bytes = (
        sequences * query_heads * partitions * head_dim * dtype.itemsize
        + 2 * sequences * query_heads * partitions * torch.float32.itemsize
    )
    ragged_workspace = torch.empty(
        workspace_bytes, device=device, dtype=torch.uint8
    )
    scale_tensor = torch.ones(1, device=device, dtype=torch.float32)
    def run() -> None:
        if args.kernel == "ragged":
            paged_attention_ragged(
                out=out,
                workspace_buffer=ragged_workspace,
                query=q,
                key_cache=ragged_key_cache,
                value_cache=ragged_value_cache,
                scale=head_dim**-0.5,
                kv_indptr=kv_indptr,
                kv_page_indices=kv_page_indices,
                kv_last_page_lens=kv_last_page_lens,
                block_size=page_size,
                max_num_partitions=partitions,
                alibi_slopes=None,
                kv_cache_dtype="auto",
                kv_cache_layout="NHD",
                logits_soft_cap=0.0,
                k_scale=scale_tensor,
                v_scale=scale_tensor,
                fp8_out_scale=None,
                partition_size=partition_size,
                mtp=1,
            )
        else:
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
                block_tables=block_table,
                context_lens=context_lens,
                block_size=page_size,
                max_context_len=max_context_length,
                alibi_slopes=None,
                kv_cache_dtype="auto",
                k_scale=scale_tensor,
                v_scale=scale_tensor,
                fp8_out_scale=None,
                partition_size=partition_size,
                mtp=1,
                q_scale=None,
            )

    run()
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        run()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))

    if args.kernel == "ragged":
        workspace_fp32 = ragged_workspace.view(torch.float32)
        score_elements = sequences * query_heads * partitions
        exp_sums = workspace_fp32[:score_elements].view(
            sequences, query_heads, partitions
        )
        max_logits = workspace_fp32[
            score_elements : 2 * score_elements
        ].view(sequences, query_heads, partitions)

    # Each partition stores its local exp sum and maximum.  Reconstruct the
    # global LSE with the same max trick used by the kernel's output reducer.
    hip_lse = torch.logsumexp(torch.log(exp_sums) + max_logits, dim=-1)
    scores = (
        q[0].float() @ flat_k[0].float().transpose(0, 1)
    ) * (head_dim**-0.5)
    reference_out = torch.softmax(scores, dim=-1) @ flat_v[0].float()
    reference_lse = torch.logsumexp(scores, dim=-1)
    result = {
        "geometry": {
            "sequences": sequences,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "length": length,
            "max_context_length": max_context_length,
            "page_size": page_size,
            "partitions": partitions,
            "value_shuffled": args.value_shuffled,
            "kernel": args.kernel,
        },
        "output_max_abs": float(
            (out[0].float() - reference_out).abs().max().item()
        ),
        "lse_max_abs": float((hip_lse[0] - reference_lse).abs().max().item()),
        "exp_sums_finite": bool(torch.isfinite(exp_sums).all().item()),
        "max_logits_finite": bool(torch.isfinite(max_logits).all().item()),
        "median_us": 1000.0 * statistics.median(samples),
        "minimum_us": 1000.0 * min(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
