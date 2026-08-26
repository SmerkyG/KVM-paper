#!/usr/bin/env python3
"""Microbenchmark the prefill coarse-attention GQA layouts."""

from __future__ import annotations

import argparse
import json

import torch

from model.kernels.lod_kernels import route_logits_coarse_attention


def _time_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=512)
    parser.add_argument("--state-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--sweep-muse", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    results: dict[str, object] = {}
    # These reproduce the sparse-layer geometries of OLMo, Muse, and Phi.
    for label, kv_heads, group_size in (
        ("olmo_g5", 8, 5),
        ("muse_g16", 2, 16),
        ("phi_g4", 2, 4),
    ):
        head_dim = 128
        query_heads = kv_heads * group_size
        q = torch.randn(
            args.batch,
            query_heads,
            args.query_len,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        route_logits = torch.randn(
            args.batch,
            query_heads,
            args.query_len,
            args.state_len,
            dtype=torch.bfloat16,
            device=device,
        )
        state_v = torch.randn(
            args.batch,
            kv_heads,
            args.state_len,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        counts = torch.randint(
            1,
            32,
            (args.batch, kv_heads, args.state_len, 1),
            dtype=torch.int32,
            device=device,
        ).float()
        empty = torch.empty(
            args.batch,
            kv_heads,
            0,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        top_slots = torch.randint(
            0,
            args.state_len,
            (args.batch, query_heads, args.query_len, 8),
            dtype=torch.int32,
            device=device,
        )

        def run(head_major: bool, *, precompute_mean_values: bool):
            return route_logits_coarse_attention(
                q,
                route_logits,
                state_v,
                counts,
                empty,
                empty,
                top_slots,
                state_len=args.state_len,
                kv_group_size=group_size,
                scale=head_dim**-0.5,
                precompute_mean_values=precompute_mean_values,
                max_grouped_rows=64,
                head_major=head_major,
            )

        grouped_output, grouped_lse = run(False, precompute_mean_values=True)
        head_output, head_lse = run(True, precompute_mean_values=True)
        dynamic_output, dynamic_lse = run(
            False, precompute_mean_values=False
        )
        results[label] = {
            "grouped_ms": _time_ms(
                lambda: run(False, precompute_mean_values=True),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "head_major_ms": _time_ms(
                lambda: run(True, precompute_mean_values=True),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "divide_in_kernel_ms": _time_ms(
                lambda: run(False, precompute_mean_values=False),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "output_max_abs": float(
                (grouped_output.float() - head_output.float()).abs().max().item()
            ),
            "lse_max_abs": float((grouped_lse - head_lse).abs().max().item()),
            "divide_in_kernel_output_max_abs": float(
                (grouped_output.float() - dynamic_output.float()).abs().max().item()
            ),
            "divide_in_kernel_lse_max_abs": float(
                (grouped_lse - dynamic_lse).abs().max().item()
            ),
        }
        if label == "muse_g16" and args.sweep_muse:
            sweep: dict[str, float] = {}
            for block_n in (32, 64, 128):
                for max_grouped_rows in (8, 16, 32, 64):
                    for num_warps in (2, 4):
                        name = (
                            f"n{block_n}_rows{max_grouped_rows}_w{num_warps}"
                        )

                        def run_sweep(
                            block_n: int = block_n,
                            max_grouped_rows: int = max_grouped_rows,
                            num_warps: int = num_warps,
                        ):
                            return route_logits_coarse_attention(
                                q,
                                route_logits,
                                state_v,
                                counts,
                                empty,
                                empty,
                                top_slots,
                                state_len=args.state_len,
                                kv_group_size=group_size,
                                scale=head_dim**-0.5,
                                block_m=16,
                                block_n=block_n,
                                num_warps=num_warps,
                                precompute_mean_values=True,
                                max_grouped_rows=max_grouped_rows,
                                head_major=False,
                            )

                        sweep[name] = _time_ms(
                            run_sweep,
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
            results[label]["sweep_ms"] = sweep
        del q, route_logits, state_v, counts, empty, top_slots
        torch.cuda.empty_cache()

    payload = {
        "batch": args.batch,
        "query_len": args.query_len,
        "state_len": args.state_len,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
