#!/usr/bin/env python3
"""Benchmark retained-mass page-size-one decode routing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import triton

from model.kernels.aiter_page1_attention import (
    init_page1_predicted_mass_union,
    kernel_page1_predicted_mass_union,
)


def time_us(function, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return 1000.0 * float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--state-length", type=int, default=4352)
    parser.add_argument("--mass-fraction", type=float, default=1.0 / 16.0)
    parser.add_argument("--max-leaf-tokens", type=int, default=1024)
    parser.add_argument("--query-noise", type=float, default=0.01)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.gqa > 16 or args.head_dim not in (128, 256):
        raise ValueError("the page-size-one diagnostic supports GQA<=16 and D=128/256")
    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch = args.batch_size
    kv_heads = args.kv_heads
    gqa = args.gqa
    query_heads = kv_heads * gqa
    head_dim = args.head_dim
    state_len = args.state_length
    sequences = batch * kv_heads
    scale = head_dim**-0.5

    q0 = torch.randn(
        batch, query_heads, head_dim, device=device, dtype=dtype
    )
    q1 = (q0.float() + args.query_noise * torch.randn_like(q0.float())).to(dtype)
    coarse = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=dtype
    )
    counts = torch.randint(
        1,
        args.max_leaf_tokens + 128,
        (batch, kv_heads, state_len, 1),
        device=device,
        dtype=torch.int32,
    ).float()
    bias = counts[..., 0].log().to(torch.float16)
    cache_indices = torch.arange(batch, device=device, dtype=torch.int64)
    epochs = torch.zeros(sequences, device=device, dtype=torch.int32)
    union_counts = torch.zeros_like(epochs)
    token_counts = torch.zeros_like(epochs)
    stamps = torch.zeros(
        sequences, state_len, device=device, dtype=torch.int32
    )
    union_slots = torch.empty(
        sequences, state_len, device=device, dtype=torch.int32
    )

    def scores(query: torch.Tensor) -> torch.Tensor:
        grouped = query.reshape(batch, kv_heads, gqa, head_dim)
        result = torch.einsum("bkgd,bksd->bkgs", grouped.float(), coarse.float())
        result.mul_(scale).add_(bias.float().unsqueeze(2))
        eligible = (
            counts[..., 0].gt(0)
            & counts[..., 0].lt(args.max_leaf_tokens)
        )
        eligible[..., 0] = False
        return result.masked_fill(~eligible.unsqueeze(2), float("-inf"))

    previous_lse = torch.logsumexp(scores(q0), dim=-1).reshape(batch, query_heads)
    current_scores = scores(q1)
    threshold = previous_lse.reshape(batch, kv_heads, gqa, 1)
    threshold = threshold + math.log(args.mass_fraction)
    reference = (current_scores > threshold).any(dim=2)

    q_sequence = q1.reshape(sequences, gqa, head_dim).contiguous()
    coarse_flat = coarse.reshape(-1, head_dim)
    bias_flat = bias.reshape(-1)

    def route() -> None:
        init_page1_predicted_mass_union[(triton.cdiv(sequences, 1),)](
            epochs,
            union_counts,
            token_counts,
            SEQUENCES=sequences,
            num_warps=1,
        )
        kernel_page1_predicted_mass_union[
            (sequences, triton.cdiv(state_len, 64))
        ](
            q_sequence,
            coarse_flat,
            bias_flat,
            counts,
            cache_indices,
            previous_lse,
            stamps,
            epochs,
            union_counts,
            union_slots,
            scale,
            q_sequence.stride(0),
            q_sequence.stride(1),
            counts.stride(0),
            counts.stride(1),
            counts.stride(2),
            previous_lse.stride(0),
            previous_lse.stride(1),
            NUM_QUERY_HEADS=gqa,
            KV_HEADS=kv_heads,
            STATE_LEN=state_len,
            STATE_CAPACITY=state_len,
            COARSE_OFFSET=0,
            UNION_CAPACITY=state_len,
            TILE_SIZE=64,
            HEAD_SIZE=head_dim,
            BLOCK_M=16,
            PROTECTED_LEN=1,
            MAX_LEAF_TOKENS=args.max_leaf_tokens,
            LOG_MASS_FRACTION=math.log(args.mass_fraction),
            num_warps=2,
            waves_per_eu=2,
            num_stages=2,
        )

    route()
    torch.cuda.synchronize()
    exact = True
    expected_counts = reference.sum(dim=-1).reshape(-1).to(torch.int32)
    for sequence in range(sequences):
        count = int(union_counts[sequence].item())
        actual = union_slots[sequence, :count].sort().values
        expected = torch.nonzero(
            reference.reshape(sequences, state_len)[sequence], as_tuple=False
        )[:, 0].to(torch.int32)
        if actual.numel() != expected.numel() or not bool((actual == expected).all()):
            exact = False
            break

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "kv_heads": kv_heads,
        "gqa": gqa,
        "head_dim": head_dim,
        "state_length": state_len,
        "mass_fraction": args.mass_fraction,
        "query_noise": args.query_noise,
        "correctness": {
            "union_counts_exact": bool((union_counts == expected_counts).all()),
            "union_sets_exact": exact,
        },
        "selected_union_centroids": {
            "minimum": int(union_counts.min().item()),
            "mean": float(union_counts.float().mean().item()),
            "maximum": int(union_counts.max().item()),
        },
        "route_us": time_us(route, args.warmups, args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
