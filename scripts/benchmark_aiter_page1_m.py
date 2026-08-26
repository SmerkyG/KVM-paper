#!/usr/bin/env python3
"""Measure AITER page-size-one attention throughput as M changes.

Each sequence owns a distinct KV stream.  This makes the M comparison include
the real reuse obtained when more query rows attend the same keys, rather than
accidentally sharing one physical cache across otherwise independent rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch


VENDOR_AITER = Path("/home/dan/subusers/agent/vendor/aiter")
if str(VENDOR_AITER) not in sys.path:
    sys.path.append(str(VENDOR_AITER))
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")


def measure_ms(function, *, warmups: int, trials: int, iterations: int) -> list[float]:
    result = None
    for _ in range(warmups):
        result = function()
    torch.cuda.synchronize()
    elapsed: list[float] = []
    for _ in range(trials):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            result = function()
        end.record()
        torch.cuda.synchronize()
        elapsed.append(begin.elapsed_time(end) / iterations)
    if result is None:
        raise AssertionError("benchmark did not execute")
    output, lse = result
    if not bool(torch.isfinite(output).all().item()):
        raise AssertionError("AITER returned a non-finite output")
    if not bool(torch.isfinite(lse).all().item()):
        raise AssertionError("AITER returned a non-finite LSE")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", type=int, nargs="+", default=(16, 64, 128))
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=(4096, 16384, 65536)
    )
    parser.add_argument(
        "--sequence-counts", type=int, nargs="+", default=(16, 64)
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in (*args.m_values, *args.k_values, *args.sequence_counts)):
        raise ValueError("M, K, and sequence counts must be positive")
    if args.head_dim <= 0:
        raise ValueError("head dimension must be positive")

    from aiter.ops.mha import mha_batch_prefill_func

    torch.manual_seed(args.seed)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    max_sequences = max(args.sequence_counts)
    max_k = max(args.k_values)
    physical_tokens = max_sequences * max_k
    token_k = torch.empty(
        physical_tokens,
        1,
        1,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    token_v = torch.empty_like(token_k)
    token_k.normal_(mean=0.0, std=0.125)
    token_v.normal_(mean=0.0, std=0.125)
    scale = 1.0 / math.sqrt(args.head_dim)
    records: list[dict[str, object]] = []

    # Run the largest shapes first so clock and allocator warm-up do not
    # systematically favor M=128 in the comparison.
    shapes = [
        (sequences, k, m)
        for sequences in args.sequence_counts
        for k in args.k_values
        for m in args.m_values
    ]
    shapes.sort(key=lambda item: (item[1], item[0], item[2]), reverse=True)
    for sequences, k, m in shapes:
        packed_q = torch.empty(
            sequences * m,
            1,
            args.head_dim,
            device=device,
            dtype=dtype,
        )
        packed_q.normal_(mean=0.0, std=0.125)
        qo_indptr = (
            torch.arange(sequences + 1, device=device, dtype=torch.int32) * m
        )
        kv_indptr = (
            torch.arange(sequences + 1, device=device, dtype=torch.int32) * k
        )
        sequence = torch.arange(sequences, device=device, dtype=torch.int32)
        token = torch.arange(k, device=device, dtype=torch.int32)
        page_indices = (sequence[:, None] * max_k + token[None, :]).reshape(-1)
        page_indices = page_indices.contiguous()
        kv_last_page_lens = torch.ones(
            sequences, device=device, dtype=torch.int32
        )
        seqlen_k = torch.full(
            (sequences,), k, device=device, dtype=torch.int32
        )

        def attention():
            return mha_batch_prefill_func(
                packed_q,
                token_k,
                token_v,
                qo_indptr,
                kv_indptr,
                page_indices,
                m,
                k,
                softmax_scale=scale,
                causal=False,
                return_lse=True,
                kv_last_page_lens=kv_last_page_lens,
                seqlen_k=seqlen_k,
            )

        samples_ms = measure_ms(
            attention,
            warmups=args.warmups,
            trials=args.trials,
            iterations=args.iterations,
        )
        median_ms = statistics.median(samples_ms)
        mean_ms = statistics.fmean(samples_ms)
        pairs = sequences * m * k
        flops = 4 * pairs * args.head_dim
        unique_kv_bytes = sequences * k * args.head_dim * 2 * 2
        records.append(
            {
                "sequences": sequences,
                "m": m,
                "k": k,
                "head_dim": args.head_dim,
                "samples_ms": samples_ms,
                "median_ms": median_ms,
                "mean_ms": mean_ms,
                "min_ms": min(samples_ms),
                "max_ms": max(samples_ms),
                "query_rows": sequences * m,
                "qk_pairs": pairs,
                "gpair_per_s": pairs / (median_ms * 1.0e6),
                "estimated_tflop_per_s": flops / (median_ms * 1.0e9),
                "unique_kv_tb_per_s": unique_kv_bytes / (median_ms * 1.0e9),
            }
        )

    for sequences in args.sequence_counts:
        for k in args.k_values:
            group = [
                record
                for record in records
                if record["sequences"] == sequences and record["k"] == k
            ]
            baseline = next(record for record in group if record["m"] == 16)
            baseline_throughput = float(baseline["gpair_per_s"])
            for record in group:
                record["throughput_vs_m16"] = (
                    float(record["gpair_per_s"]) / baseline_throughput
                )

    properties = torch.cuda.get_device_properties(device)
    result = {
        "device": {
            "name": properties.name,
            "total_memory_gib": properties.total_memory / (1024**3),
        },
        "dtype": str(dtype),
        "page_size": 1,
        "head_dim": args.head_dim,
        "m_values": args.m_values,
        "k_values": args.k_values,
        "sequence_counts": args.sequence_counts,
        "warmups": args.warmups,
        "trials": args.trials,
        "iterations": args.iterations,
        "physical_cache_gib": (
            token_k.numel() * token_k.element_size()
            + token_v.numel() * token_v.element_size()
        )
        / (1024**3),
        "records": sorted(
            records, key=lambda record: (record["sequences"], record["k"], record["m"])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
