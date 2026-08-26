#!/usr/bin/env python3
"""Compare exact-current and retained-threshold GQA16 routing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.gqa16_coarse_score import (
    gqa16_coarse_scores_lse,
    gqa16_predicted_mass_union,
    init_mass_union,
    mass_cutoff_union,
    reduce_partition_lse,
)


def time_us(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    end.synchronize()
    return 1000.0 * float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-length", type=int, default=4352)
    parser.add_argument("--mass-fraction", type=float, default=1.0 / 256.0)
    parser.add_argument(
        "--threshold-mode", choices=("mass", "rank"), default="mass"
    )
    parser.add_argument("--target-routes", type=int, default=8)
    parser.add_argument("--max-leaf-tokens", type=int, default=1024)
    parser.add_argument(
        "--query-noise", type=float, nargs="+", default=(0.0, 0.01, 0.03, 0.1)
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260824)
    device = torch.device("cuda")
    batch = args.batch_size
    kv_heads, gqa, head_dim = 2, 16, 128
    query_heads = kv_heads * gqa
    state_len = args.state_length
    segments = (state_len + 255) // 256
    sequences = batch * kv_heads
    union_capacity = (
        min(state_len, gqa * max(1, math.ceil(1.0 / args.mass_fraction) - 1))
        if args.threshold_mode == "mass"
        # Retained rank thresholds can temporarily over-select after a query
        # shift. Keep the diagnostic untruncated so precision and list-size
        # measurements expose that behavior faithfully.
        else state_len
    )

    q0 = torch.randn(
        batch, query_heads, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    means = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=torch.bfloat16
    )
    counts = torch.randint(
        1, 1025, (batch, kv_heads, state_len, 1), device=device
    ).float()
    state_k = (means.float() * counts).to(torch.bfloat16)
    cache_indices = torch.arange(batch, device=device, dtype=torch.int64)

    def workspace():
        return {
            "scores": torch.empty(
                batch,
                query_heads,
                1,
                state_len,
                device=device,
                dtype=torch.float32,
            ),
            "partial": torch.empty(
                batch * query_heads, segments, device=device, dtype=torch.float32
            ),
            "lse": torch.empty(
                batch, query_heads, 1, device=device, dtype=torch.float32
            ),
            "epochs": torch.zeros(sequences, device=device, dtype=torch.int32),
            "counts": torch.zeros(sequences, device=device, dtype=torch.int32),
            "token_counts": torch.zeros(
                sequences, device=device, dtype=torch.int32
            ),
            "slots": torch.empty(
                sequences, union_capacity, device=device, dtype=torch.int32
            ),
            "stamps": torch.zeros(
                sequences, state_len, device=device, dtype=torch.int32
            ),
        }

    previous = workspace()
    exact = workspace()
    predicted = workspace()

    def score_current(q, work) -> None:
        gqa16_coarse_scores_lse(
            q,
            state_k,
            counts,
            cache_indices,
            work["scores"],
            work["partial"],
            state_len=state_len,
            max_leaf_tokens=args.max_leaf_tokens,
            scale=head_dim**-0.5,
        )

    def exact_route(q, work) -> None:
        score_current(q, work)
        reduce_partition_lse(
            work["partial"],
            work["lse"],
            work["epochs"],
            work["counts"],
            work["token_counts"],
            query_heads=query_heads,
            kv_heads=kv_heads,
            active_segments=segments,
        )
        if args.threshold_mode == "mass":
            mass_cutoff_union(
                work["scores"],
                work["lse"],
                counts,
                cache_indices,
                work["stamps"],
                work["epochs"],
                work["counts"],
                work["slots"],
                state_len=state_len,
                mass_fraction=args.mass_fraction,
                max_leaf_tokens=args.max_leaf_tokens,
            )

    exact_route(q0, previous)
    if args.threshold_mode == "mass":
        retained_threshold = previous["lse"] + math.log(args.mass_fraction)
    else:
        if not 0 < args.target_routes < state_len:
            raise ValueError("target routes must lie in (0, state length)")
        boundary = torch.topk(
            previous["scores"], args.target_routes + 1, dim=-1
        ).values
        retained_threshold = 0.5 * (
            boundary[..., args.target_routes - 1]
            + boundary[..., args.target_routes]
        )
    records = []
    for noise in args.query_noise:
        torch.manual_seed(20260825)
        q1 = (q0.float() + noise * torch.randn_like(q0.float())).to(
            torch.bfloat16
        )
        exact_route(q1, exact)
        exact_rank_union = None
        if args.threshold_mode == "rank":
            rank_slots = torch.topk(
                exact["scores"], args.target_routes, dim=-1
            ).indices.reshape(batch, kv_heads, gqa, args.target_routes)
            exact_rank_union = []
            for sequence in range(sequences):
                batch_index = sequence // kv_heads
                kv_head = sequence % kv_heads
                exact_rank_union.append(
                    set(
                        rank_slots[batch_index, kv_head]
                        .reshape(-1)
                        .cpu()
                        .tolist()
                    )
                )

        def predicted_route(*, emit_lse: bool = True) -> None:
            init_mass_union(
                predicted["epochs"],
                predicted["counts"],
                predicted["token_counts"],
            )
            gqa16_predicted_mass_union(
                q1,
                state_k,
                counts,
                cache_indices,
                retained_threshold,
                predicted["partial"],
                predicted["stamps"],
                predicted["epochs"],
                predicted["counts"],
                predicted["slots"],
                state_len=state_len,
                max_leaf_tokens=args.max_leaf_tokens,
                scale=head_dim**-0.5,
                emit_lse=emit_lse,
            )

        predicted_route()
        torch.cuda.synchronize()
        intersections = 0
        exact_total = 0
        predicted_total = 0
        exact_match = True
        for sequence in range(sequences):
            predicted_count = int(predicted["counts"][sequence].item())
            if exact_rank_union is None:
                exact_count = int(exact["counts"][sequence].item())
                exact_set = set(
                    exact["slots"][sequence, :exact_count].cpu().tolist()
                )
            else:
                exact_set = exact_rank_union[sequence]
            predicted_set = set(
                predicted["slots"][sequence, :predicted_count].cpu().tolist()
            )
            intersections += len(exact_set & predicted_set)
            exact_total += len(exact_set)
            predicted_total += len(predicted_set)
            exact_match &= exact_set == predicted_set
        records.append(
            {
                "query_noise": noise,
                "exact_union_mean": exact_total / sequences,
                "predicted_union_mean": float(
                    predicted["counts"].float().mean().item()
                ),
                "precision": intersections / max(1, predicted_total),
                "recall": intersections / max(1, exact_total),
                "all_union_sets_exact": exact_match,
                "predicted_route_us": time_us(
                    predicted_route, args.warmup, args.repeats
                ),
                "predicted_route_no_lse_us": time_us(
                    lambda: predicted_route(emit_lse=False),
                    args.warmup,
                    args.repeats,
                ),
            }
        )

    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "state_length": state_len,
        "threshold_mode": args.threshold_mode,
        "target_routes": args.target_routes,
        "mass_fraction": args.mass_fraction,
        "union_capacity": union_capacity,
        "exact_route_us": time_us(
            lambda: exact_route(q0, exact), args.warmup, args.repeats
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
