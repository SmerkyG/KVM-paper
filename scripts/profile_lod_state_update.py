#!/usr/bin/env python3
"""Benchmark one LOD state update independently of leaf attention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.triton_lod_attention import TritonLODAttentionCore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--update-length", type=int, default=1024)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--union-bipartite-state", action="store_true")
    parser.add_argument("--state-precompact-direct-append", action="store_true")
    parser.add_argument("--overflow-bipartite-block-size", type=int, default=0)
    parser.add_argument("--overflow-bipartite-positional-halves", action="store_true")
    parser.add_argument("--overflow-bipartite-keep-ratio", type=float, default=0.5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scheduled_length(context: int, factor: float) -> int:
    return max(math.floor(factor * math.sqrt(max(context, 0))), 256)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    previous_context = args.context_length - args.update_length
    if previous_context <= 0:
        raise ValueError("context length must exceed the update length")
    state_len = scheduled_length(previous_context, args.state_growth_factor)
    target_len = scheduled_length(args.context_length, args.state_growth_factor)
    shape = (args.batch_size, args.kv_heads)
    counts_base = torch.randint(
        1,
        17,
        (*shape, target_len, 1),
        device=device,
        dtype=torch.int32,
    ).float()
    mean_k = torch.randn(
        *shape, target_len, args.head_dim, device=device, dtype=torch.bfloat16
    )
    mean_v = torch.randn_like(mean_k)
    state_k_base = mean_k * counts_base.to(mean_k.dtype)
    state_v_base = mean_v * counts_base.to(mean_v.dtype)
    overflow_k = torch.randn(
        *shape,
        args.update_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    overflow_v = torch.randn_like(overflow_k)
    state_k = torch.empty_like(state_k_base)
    state_v = torch.empty_like(state_v_base)
    counts = torch.empty_like(counts_base)

    core = TritonLODAttentionCore()
    core.state_growth_factor = args.state_growth_factor
    core.state_min_len = 256
    core.chunk_len = 256
    core.sink_len = 1
    core.leaf_attention_backend = "packed"
    core.state_union_bipartite = args.union_bipartite_state
    core.state_precompact_direct_append = args.state_precompact_direct_append
    core.overflow_bipartite_merge = args.overflow_bipartite_block_size > 0
    if core.overflow_bipartite_merge:
        core.overflow_bipartite_block_size = args.overflow_bipartite_block_size
        core.overflow_bipartite_positional_halves = (
            args.overflow_bipartite_positional_halves
        )
        core.overflow_bipartite_keep_ratio = args.overflow_bipartite_keep_ratio

    def run_once() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        state_k.copy_(state_k_base)
        state_v.copy_(state_v_base)
        counts.copy_(counts_base)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = core._update_state(
            state_k,
            state_v,
            counts,
            overflow_k,
            overflow_v,
            state_len=state_len,
            ctx_len=args.context_length,
            available_context=args.context_length,
            state_capacity=target_len,
        )
        end.record()
        if result[3] != target_len:
            raise AssertionError("state update returned the wrong target length")
        return begin, end

    for _ in range(args.warmups):
        run_once()
    torch.cuda.synchronize()
    events = [run_once() for _ in range(args.repeats)]
    torch.cuda.synchronize()
    elapsed_ms = [float(begin.elapsed_time(end)) for begin, end in events]
    result = {
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "elapsed_ms": elapsed_ms,
        "head_dim": args.head_dim,
        "kv_heads": args.kv_heads,
        "mean_ms": sum(elapsed_ms) / len(elapsed_ms),
        "median_ms": float(torch.tensor(elapsed_ms).median()),
        "overflow_bipartite_block_size": args.overflow_bipartite_block_size,
        "overflow_bipartite_positional_halves": (
            args.overflow_bipartite_positional_halves
        ),
        "overflow_bipartite_keep_ratio": args.overflow_bipartite_keep_ratio,
        "repeats": args.repeats,
        "state_growth_factor": args.state_growth_factor,
        "state_len": state_len,
        "target_len": target_len,
        "union_bipartite_state": args.union_bipartite_state,
        "state_precompact_direct_append": args.state_precompact_direct_append,
        "update_length": args.update_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
