#!/usr/bin/env python3
"""Probe whether AITER's K3 MLA prefill assembly can emit split LSEs."""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--splits", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.heads != 16:
        raise ValueError("the gfx942 K3 assembly probe requires 16 heads")

    from aiter.ops.attention import mla_prefill_asm_fwd

    device = torch.device("cuda")
    dtype = torch.bfloat16
    key_len = args.queries + args.lookback
    torch.manual_seed(11)
    query = torch.randn(
        args.queries, args.heads, 576, dtype=dtype, device=device
    )
    key = torch.randn(key_len, 1, 1, 576, dtype=dtype, device=device)
    qo_indptr = torch.tensor([0, args.queries], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, key_len], dtype=torch.int32, device=device)
    indices = torch.arange(key_len, dtype=torch.int32, device=device)
    last_page_lens = torch.ones(1, dtype=torch.int32, device=device)
    scale = 576**-0.5

    def allocate(splits: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(
                args.queries,
                splits,
                args.heads,
                512,
                dtype=dtype,
                device=device,
            ),
            torch.empty(
                args.queries,
                splits,
                args.heads,
                1,
                dtype=torch.float32,
                device=device,
            ),
        )

    def invoke(output: torch.Tensor, lse: torch.Tensor) -> None:
        mla_prefill_asm_fwd(
            query,
            key,
            qo_indptr,
            kv_indptr,
            indices,
            last_page_lens,
            args.queries,
            scale,
            output,
            lse,
        )

    reference, reference_lse = allocate(1)
    split_output, split_lse = allocate(args.splits)
    invoke(reference, reference_lse)
    invoke(split_output, split_lse)
    torch.cuda.synchronize()

    weights = torch.softmax(split_lse.float(), dim=1)
    merged = (split_output.float() * weights).sum(dim=1)
    reference_value = reference[:, 0].float()
    difference = (merged - reference_value).abs()

    timings: dict[str, list[float]] = {"one": [], "split": []}
    for name, output, lse in (
        ("one", reference, reference_lse),
        ("split", split_output, split_lse),
    ):
        for _ in range(args.iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            invoke(output, lse)
            end.record()
            end.synchronize()
            timings[name].append(begin.elapsed_time(end))

    print(
        json.dumps(
            {
                "queries": args.queries,
                "key_len": key_len,
                "splits": args.splits,
                "reference_lse_finite_fraction": float(
                    torch.isfinite(reference_lse).float().mean()
                ),
                "split_lse_finite_fraction": float(
                    torch.isfinite(split_lse).float().mean()
                ),
                "maximum_output_error": float(difference.max()),
                "mean_output_error": float(difference.mean()),
                "one_split_median_ms": statistics.median(timings["one"]),
                "multi_split_median_ms": statistics.median(timings["split"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
