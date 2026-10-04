#!/usr/bin/env python3
"""Time Kimi's expanded-query to absorbed-latent projection."""

from __future__ import annotations

import argparse
import statistics

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="16384,32768")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--heads", type=int, default=96)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    weight = torch.randn(args.heads, 128, 512, device=device, dtype=dtype)

    for length in (int(value) for value in args.lengths.split(",")):
        query = torch.randn(length, args.heads, 192, device=device, dtype=dtype)
        query_nope, query_direct = query.split([128, 64], dim=-1)

        def project() -> torch.Tensor:
            latent = torch.bmm(
                query_nope.transpose(0, 1), weight
            ).transpose(0, 1)
            return torch.cat((latent, query_direct), dim=-1)

        for _ in range(2):
            output = project()
        torch.cuda.synchronize()
        timings = []
        for _ in range(args.iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = project()
            end.record()
            torch.cuda.synchronize()
            timings.append(begin.elapsed_time(end))
        print(
            f"length={length} median_ms={statistics.median(timings):.6f} "
            f"timings_ms={timings}",
            flush=True,
        )
        del query, query_nope, query_direct, output
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
