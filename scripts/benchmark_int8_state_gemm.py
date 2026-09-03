#!/usr/bin/env python3
"""Microbenchmark dense INT8 GEMM shapes used by materialized LOD coarse PV."""

from __future__ import annotations

import argparse
import json

import torch


def timed(callable_, repeats: int) -> float:
    for _ in range(5):
        callable_()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        callable_()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--state", type=int, default=3072)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    shape_a = (args.rows, args.state) if args.batch == 1 else (
        args.batch,
        args.rows,
        args.state,
    )
    shape_b = (args.state, args.value_dim) if args.batch == 1 else (
        args.batch,
        args.state,
        args.value_dim,
    )
    a8 = torch.randint(
        -127,
        128,
        shape_a,
        dtype=torch.int8,
        device="cuda",
    )
    b8 = torch.randint(
        -127,
        128,
        shape_b,
        dtype=torch.int8,
        device="cuda",
    )
    abf = a8.to(torch.bfloat16)
    bbf = b8.to(torch.bfloat16)
    result = {"shape": [args.batch, args.rows, args.state, args.value_dim]}
    if args.batch == 1:
        result["int8_int_mm_ms"] = timed(
            lambda: torch._int_mm(a8, b8), args.repeats
        )
        result["bf16_mm_ms"] = timed(lambda: torch.mm(abf, bbf), args.repeats)
    else:
        import aiter
        from aiter.ops.triton.gemm.batched.batched_gemm_a8w8 import (
            batched_gemm_a8w8 as triton_batched_gemm_a8w8,
        )

        x_scale = torch.ones(
            args.batch,
            args.rows,
            1,
            dtype=torch.float32,
            device="cuda",
        )
        w_scale = torch.ones(
            args.batch,
            1,
            args.value_dim,
            dtype=torch.float32,
            device="cuda",
        )
        weights = b8.transpose(1, 2).contiguous()
        result["bf16_bmm_ms"] = timed(
            lambda: torch.bmm(abf, bbf), args.repeats
        )
        result["aiter_ck_int8_bmm_ms"] = timed(
            lambda: aiter.batched_gemm_a8w8_CK(
                a8,
                weights,
                x_scale,
                w_scale,
            ),
            args.repeats,
        )
        result["aiter_triton_int8_bmm_ms"] = timed(
            lambda: triton_batched_gemm_a8w8(
                a8,
                weights,
                x_scale,
                w_scale,
            ),
            args.repeats,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
