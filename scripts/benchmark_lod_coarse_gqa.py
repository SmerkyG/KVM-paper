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
    parser.add_argument("--sweep-direct", action="store_true")
    parser.add_argument(
        "--only",
        choices=("olmo_g5", "muse_g16", "phi_g4", "qwen_g6_d256"),
        default=None,
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    results: dict[str, object] = {}
    # These reproduce the sparse-layer geometries of OLMo, Muse, and Phi.
    for label, kv_heads, group_size, head_dim in (
        ("olmo_g5", 8, 5, 128),
        ("muse_g16", 2, 16, 128),
        ("phi_g4", 2, 4, 128),
        ("qwen_g6_d256", 4, 6, 256),
    ):
        if args.only is not None and label != args.only:
            continue
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

        def run(
            head_major: bool,
            *,
            precompute_mean_values: bool,
            block_m: int = 16,
            block_n: int = 32,
            num_warps: int = 4,
            grouped_rows: int = 64,
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
                block_m=block_m,
                block_n=block_n,
                num_warps=num_warps,
                precompute_mean_values=precompute_mean_values,
                max_grouped_rows=grouped_rows,
                head_major=head_major,
            )

        def run_direct(*, grouped_rows: int, block_n: int = 64, num_warps: int = 4):
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
                block_n=block_n,
                num_warps=num_warps,
                precompute_mean_values=True,
                max_grouped_rows=grouped_rows,
                direct_gqa_rows=True,
            )

        grouped_output, grouped_lse = run(False, precompute_mean_values=True)
        head_output, head_lse = run(True, precompute_mean_values=True)
        dynamic_output, dynamic_lse = run(
            False, precompute_mean_values=False
        )
        direct64_output, direct64_lse = run_direct(grouped_rows=64)
        direct128_output, direct128_lse = run_direct(grouped_rows=128)
        results[label] = {
            "grouped_ms": _time_ms(
                lambda: run(False, precompute_mean_values=True),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "production_m64_n32_w8_ms": _time_ms(
                lambda: run(
                    False,
                    precompute_mean_values=True,
                    block_n=32,
                    num_warps=8,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "grouped_m64_n16_w8_ms": _time_ms(
                lambda: run(
                    False,
                    precompute_mean_values=True,
                    block_n=16,
                    num_warps=8,
                ),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "grouped_m128_n16_w8_ms": _time_ms(
                lambda: run(
                    False,
                    precompute_mean_values=True,
                    block_n=16,
                    num_warps=8,
                    grouped_rows=128,
                ),
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
            "direct_gqa_m64_ms": _time_ms(
                lambda: run_direct(grouped_rows=64),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "direct_gqa_m128_ms": _time_ms(
                lambda: run_direct(grouped_rows=128),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "direct_gqa_m64_n16_w8_ms": _time_ms(
                lambda: run_direct(grouped_rows=64, block_n=16, num_warps=8),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "direct_gqa_m128_n16_w8_ms": _time_ms(
                lambda: run_direct(grouped_rows=128, block_n=16, num_warps=8),
                warmup=args.warmup,
                repeats=args.repeats,
            ),
            "direct_gqa_m64_output_max_abs": float(
                (grouped_output.float() - direct64_output.float()).abs().max().item()
            ),
            "direct_gqa_m64_lse_max_abs": float(
                (grouped_lse - direct64_lse).abs().max().item()
            ),
            "direct_gqa_m128_output_max_abs": float(
                (grouped_output.float() - direct128_output.float()).abs().max().item()
            ),
            "direct_gqa_m128_lse_max_abs": float(
                (grouped_lse - direct128_lse).abs().max().item()
            ),
        }
        if args.sweep_direct:
            direct_sweep: dict[str, float] = {}
            for grouped_rows in (32, 64, 128):
                if grouped_rows < group_size:
                    continue
                for block_n in (16, 32, 64, 128):
                    for num_warps in (2, 4, 8):
                        name = f"m{grouped_rows}_n{block_n}_w{num_warps}"
                        direct_sweep[name] = _time_ms(
                            lambda grouped_rows=grouped_rows,
                            block_n=block_n,
                            num_warps=num_warps: run_direct(
                                grouped_rows=grouped_rows,
                                block_n=block_n,
                                num_warps=num_warps,
                            ),
                            warmup=args.warmup,
                            repeats=args.repeats,
                        )
            results[label]["direct_sweep_ms"] = direct_sweep
        if label == "muse_g16" and args.sweep_muse:
            sweep: dict[str, float] = {}
            for block_n in (32, 64, 128):
                for max_grouped_rows in (8, 16, 32, 64):
                    for num_warps in (2, 4, 8):
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
            tuned_output, tuned_lse = route_logits_coarse_attention(
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
                block_n=64,
                num_warps=8,
                precompute_mean_values=True,
                max_grouped_rows=16,
                head_major=False,
            )
            results[label]["tuned_output_max_abs"] = float(
                (grouped_output.float() - tuned_output.float()).abs().max().item()
            )
            results[label]["tuned_lse_max_abs"] = float(
                (grouped_lse - tuned_lse).abs().max().item()
            )
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
