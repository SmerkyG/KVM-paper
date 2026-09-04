#!/usr/bin/env python3
"""Microbenchmark the long batch-one LOD state catch-up path."""

from __future__ import annotations

import argparse
import itertools
import json

import torch

from model.pytorch_lod_attention import LODConfig
from model.triton_lod_engines import KernelTwoLevelLODAttention


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-m", type=parse_ints, default=(8, 16, 32, 64))
    parser.add_argument("--block-n", type=parse_ints, default=(16, 32, 64, 128))
    parser.add_argument("--warps", type=parse_ints, default=(2, 4, 8))
    parser.add_argument("--context-length", type=int, default=8448)
    parser.add_argument("--available-context", type=int, default=8192)
    parser.add_argument("--state-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, heads, dimension = args.batch_size, 8, 128
    state_len = args.state_len
    available_context = args.available_context
    if not 0 < state_len < available_context <= args.context_length:
        raise ValueError(
            "state-len must be below available-context, which must not exceed "
            "context-length"
        )
    overflow_len = available_context - state_len
    config = LODConfig(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16,
        state_min_size=256,
        max_routes=8,
        state_clustering_centroid_rescale="coherence",
        state_clustering_centroid_rescale_scope="assignment",
    )
    key = torch.randn(batch, heads, available_context, dimension, device=device)
    key *= key.float().square().mean(-1, keepdim=True).rsqrt().to(key.dtype)
    value = torch.randn_like(key)
    state_capacity = max(
        state_len,
        int(config.state_growth_factor * args.context_length**0.5),
    )

    rows = []
    for block_m, block_n, warps in itertools.product(
        args.block_m, args.block_n, args.warps
    ):
        engine = KernelTwoLevelLODAttention(
            config,
            query_heads=64,
            key_value_heads=heads,
            scale=dimension**-0.5,
            default_open_count=2,
        ).to(device)
        engine.fused_state_update = True
        engine.fused_state_maxsim = True
        engine.state_maxsim_block_m = block_m
        engine.state_maxsim_block_n = block_n
        engine.state_maxsim_num_warps = warps
        state_k = torch.zeros(
            batch, heads, state_capacity, dimension,
            dtype=key.dtype, device=device,
        )
        state_v = torch.zeros_like(state_k)
        counts = torch.zeros(
            batch, heads, state_capacity, 1,
            dtype=torch.float32, device=device,
        )
        key_norm_sums = torch.zeros_like(counts)
        state_k[..., :state_len, :].copy_(key[..., :state_len, :])
        state_v[..., :state_len, :].copy_(value[..., :state_len, :])
        counts[..., :state_len, :].fill_(1)
        key_norm_sums[..., :state_len, :].fill_(1)

        def update() -> None:
            engine._update_state(
                state_k,
                state_v,
                counts,
                key_norm_sums,
                key[..., state_len:, :],
                value[..., state_len:, :],
                state_len=state_len,
                ctx_len=args.context_length,
                available_context=available_context,
                state_capacity=state_capacity,
            )

        update()
        update()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.repeats):
            update()
        end.record()
        end.synchronize()
        rows.append(
            {
                "block_m": block_m,
                "block_n": block_n,
                "warps": warps,
                "milliseconds": begin.elapsed_time(end) / args.repeats,
            }
        )
    result = {
        "available_context": available_context,
        "context_length": args.context_length,
        "geometry": [batch, heads, overflow_len, dimension],
        "state_len": state_len,
        "results": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
