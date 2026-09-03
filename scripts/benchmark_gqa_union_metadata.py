#!/usr/bin/env python3
"""Microbenchmark the fixed top-k GQA-union metadata kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton

from model.kernels.paged_leaf_attention import _decode_topk_gqa_union_kernel


def graph_elapsed_ms(call, repeats: int) -> float:
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 2, 8))
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=4)
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument("--state-len", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    results: list[dict[str, float | int]] = []
    for batch in args.batch_sizes:
        query_heads = args.kv_heads * args.gqa
        sequences = batch * args.kv_heads
        top_slots = torch.randint(
            0,
            args.state_len,
            (batch, query_heads, args.routes),
            dtype=torch.int32,
            device=device,
        )
        capacity = args.gqa * args.routes
        for warps in (1, 2, 4, 8):
            seen = torch.zeros(
                sequences, args.state_len, dtype=torch.int32, device=device
            )
            epochs = torch.zeros(sequences, dtype=torch.int32, device=device)
            counts = torch.empty(sequences, dtype=torch.int32, device=device)
            token_counts = torch.empty_like(counts)
            slots = torch.empty(
                sequences, capacity, dtype=torch.int32, device=device
            )

            def call() -> None:
                _decode_topk_gqa_union_kernel[(sequences,)](
                    top_slots,
                    seen,
                    epochs,
                    counts,
                    token_counts,
                    slots,
                    args.state_len,
                    top_slots.stride(0),
                    top_slots.stride(1),
                    QUERY_HEADS=query_heads,
                    KV_HEADS=args.kv_heads,
                    KV_GROUP_SIZE=args.gqa,
                    ROUTE_COUNT=args.routes,
                    STATE_CAPACITY=args.state_len,
                    CANDIDATE_BLOCK=triton.next_power_of_2(capacity),
                    num_warps=warps,
                    waves_per_eu=1,
                )

            call()
            torch.cuda.synchronize()
            if int(counts.min().item()) <= 0:
                raise AssertionError("union metadata did not produce routes")
            results.append(
                {
                    "batch_size": batch,
                    "sequence_count": sequences,
                    "num_warps": warps,
                    "milliseconds": graph_elapsed_ms(call, args.repeats),
                }
            )

    payload = {
        "kv_heads": args.kv_heads,
        "gqa": args.gqa,
        "routes": args.routes,
        "state_len": args.state_len,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
