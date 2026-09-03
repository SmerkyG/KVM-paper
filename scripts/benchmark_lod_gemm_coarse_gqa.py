#!/usr/bin/env python3
"""Compare broadcast and GQA-packed coarse PV GEMMs."""

from __future__ import annotations

import argparse
import json

import torch


def time_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=512)
    parser.add_argument("--state-len", type=int, default=4352)
    parser.add_argument("--value-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    probabilities = torch.softmax(
        torch.randn(
            args.batch,
            args.kv_heads,
            args.gqa,
            args.query_len,
            args.state_len,
            dtype=torch.float32,
            device="cuda",
        ),
        dim=-1,
    ).to(torch.bfloat16)
    values = torch.randn(
        args.batch,
        args.kv_heads,
        args.state_len,
        args.value_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )

    def broadcast():
        return torch.matmul(probabilities, values.unsqueeze(2))

    def packed():
        return torch.matmul(
            probabilities.reshape(
                args.batch,
                args.kv_heads,
                args.gqa * args.query_len,
                args.state_len,
            ),
            values,
        ).reshape(
            args.batch,
            args.kv_heads,
            args.gqa,
            args.query_len,
            args.value_dim,
        )

    broadcast_output = broadcast()
    packed_output = packed()
    payload = {
        "batch": args.batch,
        "kv_heads": args.kv_heads,
        "gqa": args.gqa,
        "query_len": args.query_len,
        "state_len": args.state_len,
        "value_dim": args.value_dim,
        "broadcast_ms": time_ms(
            broadcast, warmup=args.warmup, repeats=args.repeats
        ),
        "packed_ms": time_ms(packed, warmup=args.warmup, repeats=args.repeats),
        "output_max_abs": float(
            (broadcast_output.float() - packed_output.float()).abs().max().item()
        ),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
