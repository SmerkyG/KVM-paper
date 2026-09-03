#!/usr/bin/env python3
"""Check unordered mass-fraction centroid routing against PyTorch."""

from __future__ import annotations

import argparse

import torch

from model.kernels.lod_kernels import (
    route_logits_coarse_attention,
    route_mass_fraction_scores,
    subtract_selected_coarse_from_full,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-length", type=int, default=65)
    parser.add_argument("--state-length", type=int, default=257)
    parser.add_argument("--fraction", type=float, default=0.0625)
    parser.add_argument("--max-routes", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(17)
    device = torch.device("cuda", 0)
    kv_heads = 2
    groups = 4
    query_heads = kv_heads * groups
    scale = 0.0625
    logits = torch.randn(
        args.batch_size,
        query_heads,
        args.query_length,
        args.state_length,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    counts = torch.randint(
        1,
        65,
        (args.batch_size, kv_heads, args.state_length, 1),
        device=device,
        dtype=torch.int32,
    ).float().contiguous()
    route_lengths = torch.randint(
        0,
        5,
        (args.batch_size, kv_heads, args.state_length),
        device=device,
        dtype=torch.int32,
    ).contiguous()
    local_lse = (
        torch.randn(
            args.batch_size,
            query_heads,
            args.query_length,
            device=device,
            dtype=torch.float32,
        )
        + 5.0
    ).contiguous()

    actual, actual_count, actual_overflow, actual_partition_lse = (
        route_mass_fraction_scores(
        logits,
        counts,
        route_lengths=route_lengths,
        kv_group_size=groups,
        scale=scale,
        mass_fraction=args.fraction,
        max_routes=args.max_routes,
        state_len=args.state_length,
        protected_len=1,
        local_lse=local_lse,
        return_partition_lse=True,
        )
    )
    torch.cuda.synchronize(device)

    query_counts = counts.squeeze(-1).repeat_interleave(groups, dim=1)
    scores = (logits.bfloat16() * scale).bfloat16().float()
    scores += query_counts.log().unsqueeze(2)
    state_lse = torch.logsumexp(scores, dim=-1)
    total_lse = torch.logaddexp(state_lse, local_lse)
    selected = scores > total_lse.unsqueeze(-1) + torch.tensor(
        args.fraction, device=device
    ).log()
    selected &= route_lengths.repeat_interleave(groups, dim=1).unsqueeze(2) > 0
    selected[..., 0] = False
    expected_count = selected.sum(dim=-1, dtype=torch.int32)
    expected_overflow = (expected_count - args.max_routes).clamp_min(0)
    state_index = torch.arange(args.state_length, device=device)
    expected = torch.where(
        selected,
        state_index,
        torch.full_like(state_index, args.state_length),
    ).sort(dim=-1).values[..., : args.max_routes]
    expected = torch.where(
        expected < args.state_length, expected, torch.full_like(expected, -1)
    )

    if not torch.equal(actual, expected):
        mismatch = actual.ne(expected)
        raise AssertionError(f"route mismatch in {int(mismatch.sum().item())} entries")
    if not torch.equal(actual_count, expected_count):
        raise AssertionError("selected-count mismatch")
    if not torch.equal(actual_overflow, expected_overflow):
        raise AssertionError("overflow-count mismatch")
    precomputed = route_mass_fraction_scores(
        logits,
        counts,
        route_lengths=route_lengths,
        state_lse=actual_partition_lse,
        kv_group_size=groups,
        scale=scale,
        mass_fraction=args.fraction,
        max_routes=args.max_routes,
        state_len=args.state_length,
        protected_len=1,
        return_partition_lse=True,
    )
    if any(
        not torch.equal(left, right)
        for left, right in zip(
            precomputed[:3], (actual, actual_count, actual_overflow)
        )
    ):
        raise AssertionError("precomputed-state-LSE routing mismatch")

    dim = 64
    q = torch.randn(
        args.batch_size,
        query_heads,
        args.query_length,
        dim,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    state_v = torch.randn(
        args.batch_size,
        kv_heads,
        args.state_length,
        dim,
        device=device,
        dtype=torch.bfloat16,
    ).contiguous()
    local = torch.empty(
        args.batch_size, kv_heads, 0, dim, device=device, dtype=torch.bfloat16
    )
    test_routes = torch.randint(
        0,
        args.state_length,
        (args.batch_size, query_heads, args.query_length, 8),
        device=device,
    ).contiguous()
    test_routes[..., -2:] = -1
    no_routes = test_routes[..., :0].contiguous()
    direct_output, direct_lse = route_logits_coarse_attention(
        q,
        logits,
        state_v,
        counts,
        local,
        local,
        test_routes,
        state_len=args.state_length,
        kv_group_size=groups,
        scale=scale,
        precompute_mean_values=True,
    )
    full_output, full_lse = route_logits_coarse_attention(
        q,
        logits,
        state_v,
        counts,
        local,
        local,
        no_routes,
        state_len=args.state_length,
        kv_group_size=groups,
        scale=scale,
        precompute_mean_values=True,
    )
    corrected_output, corrected_lse = subtract_selected_coarse_from_full(
        logits,
        state_v,
        counts,
        test_routes,
        full_output,
        full_lse,
        state_len=args.state_length,
        kv_group_size=groups,
        scale=scale,
    )
    torch.cuda.synchronize(device)
    output_error = float(
        (corrected_output.float() - direct_output.float()).abs().max().item()
    )
    lse_error = float((corrected_lse - direct_lse).abs().max().item())
    if output_error > 0.02 or lse_error > 0.01:
        raise AssertionError(
            f"coarse subtraction mismatch: output={output_error}, lse={lse_error}"
        )
    print(
        {
            "status": "ok",
            "mean_selected": float(actual_count.float().mean().item()),
            "max_selected": int(actual_count.max().item()),
            "overflow_rows": int(actual_overflow.gt(0).sum().item()),
            "coarse_output_max_abs": output_error,
            "coarse_lse_max_abs": lse_error,
        }
    )


if __name__ == "__main__":
    main()
