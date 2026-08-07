#!/usr/bin/env python3
"""Synthetic prefill-chunk benchmark for the pure-PyTorch LOD backends."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from model.pytorch_lod_attention import LODState, two_level_lod_attention
from model.pytorch_lod_attention_fast import (
    _posting_lists,
    fast_two_level_lod_attention,
)


def timed(callable_, iterations: int) -> float:
    for _ in range(2):
        callable_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        callable_()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--local", type=int, default=512)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires a CUDA or ROCm GPU")
    device = torch.device("cuda")
    torch.manual_seed(70)
    state_size = args.state or math.floor(16.0 * math.sqrt(args.history))
    dtype = torch.bfloat16

    query = torch.randn(
        1, args.query_heads, args.queries, args.dimension,
        device=device, dtype=dtype,
    )
    leaf_key = torch.randn(
        1, args.kv_heads, args.history, args.dimension,
        device=device, dtype=dtype,
    )
    leaf_value = torch.randn_like(leaf_key)
    local_key = torch.randn(
        1, args.kv_heads, args.local, args.dimension,
        device=device, dtype=dtype,
    )
    local_value = torch.randn_like(local_key)
    owner = torch.arange(args.history, device=device).remainder(state_size)
    owner = owner.view(1, 1, -1).expand(1, args.kv_heads, -1).clone()
    for head in range(args.kv_heads):
        owner[:, head] = owner[:, head, torch.randperm(args.history, device=device)]
    count = torch.zeros(
        1, args.kv_heads, state_size, device=device, dtype=torch.float32
    )
    count.scatter_add_(2, owner, torch.ones_like(owner, dtype=torch.float32))
    mean_key = torch.randn(
        1, args.kv_heads, state_size, args.dimension,
        device=device, dtype=dtype,
    )
    mean_value = torch.randn_like(mean_key)
    state = LODState(
        key_sum=mean_key * count.to(dtype).unsqueeze(-1),
        value_sum=mean_value * count.to(dtype).unsqueeze(-1),
        count=count,
    )
    postings = _posting_lists(owner, state)
    query_offset = args.local - args.queries

    def reference():
        return two_level_lod_attention(
            query, local_key, local_value, state, owner, leaf_key, leaf_value,
            max_routes=args.routes,
            open_count=args.routes,
            query_offset=query_offset,
        ).output

    def fast_cached():
        return fast_two_level_lod_attention(
            query, local_key, local_value, state, owner, leaf_key, leaf_value,
            max_routes=args.routes,
            open_count=args.routes,
            query_offset=query_offset,
            postings=postings,
        ).output

    def fast_rebuild():
        return fast_two_level_lod_attention(
            query, local_key, local_value, state, owner, leaf_key, leaf_value,
            max_routes=args.routes,
            open_count=args.routes,
            query_offset=query_offset,
        ).output

    expected = reference()
    actual = fast_cached()
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=5e-2, rtol=5e-2
    )
    fast_cached_ms = timed(fast_cached, args.iterations)
    fast_rebuild_ms = timed(fast_rebuild, args.iterations)
    reference_ms = timed(reference, max(3, args.iterations // 2))
    print(
        f"shape: history={args.history} state={state_size} "
        f"queries={args.queries} q_heads={args.query_heads} "
        f"kv_heads={args.kv_heads} dim={args.dimension} routes={args.routes}"
    )
    print(f"reference:            {reference_ms:.3f} ms")
    print(f"fast, cached postings:{fast_cached_ms:8.3f} ms")
    print(f"fast, rebuild posts:  {fast_rebuild_ms:8.3f} ms")
    print(f"speedup cached:       {reference_ms / fast_cached_ms:8.2f}x")
    print(f"speedup with rebuild: {reference_ms / fast_rebuild_ms:8.2f}x")


if __name__ == "__main__":
    main()
