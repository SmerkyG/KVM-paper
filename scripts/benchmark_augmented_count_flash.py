#!/usr/bin/env python3
"""Verify and time homogeneous-coordinate page-count bias in FlashAttention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def timed_ms(function, warmups: int, repeats: int) -> float:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / repeats


def flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float):
    return torch.ops.aten._scaled_dot_product_flash_attention.default(
        q, k, v, 0.0, False, False, scale=scale
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--query-length", type=int, default=4096)
    parser.add_argument("--key-length", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--padded-dim", type=int, default=264)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    scale = args.head_dim**-0.5
    q = torch.randn(
        args.batch_size,
        args.query_heads,
        args.query_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    mean_k = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.key_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    mean_v = torch.randn_like(mean_k)
    counts = torch.randint(
        1,
        17,
        (args.batch_size, args.kv_heads, args.key_length),
        device=device,
        dtype=torch.int32,
    )
    key_sums = mean_k * counts.unsqueeze(-1)
    value_sums = mean_v * counts.unsqueeze(-1)

    q_aug = F.pad(q, (0, args.padded_dim - args.head_dim))
    q_aug[..., args.head_dim] = 1.0
    k_aug = F.pad(mean_k, (0, args.padded_dim - args.head_dim))
    k_aug[..., args.head_dim] = counts.float().log().div(scale).to(k_aug.dtype)
    v_aug = F.pad(mean_v, (0, args.padded_dim - args.head_dim))

    base_ms = timed_ms(
        lambda: flash(q, mean_k, mean_v, scale), args.warmups, args.repeats
    )
    augmented_ms = timed_ms(
        lambda: flash(q_aug, k_aug, v_aug, scale), args.warmups, args.repeats
    )

    count_values = counts.to(torch.bfloat16).unsqueeze(-1).expand_as(mean_v).contiguous()

    def two_pass_count_weighted():
        weighted_numerator, unweighted_lse, *_ = flash(
            q, mean_k, value_sums, scale
        )
        expected_count, *_ = flash(q, mean_k, count_values, scale)
        denominator_ratio = expected_count[..., :1].float()
        output = weighted_numerator.float() / denominator_ratio
        weighted_lse = unweighted_lse + denominator_ratio.squeeze(-1).log()
        return output, weighted_lse

    two_pass_ms = timed_ms(
        two_pass_count_weighted, args.warmups, args.repeats
    )

    def prepare_augmented():
        divisor = counts.unsqueeze(-1)
        normalized_k = key_sums.float().div(divisor).to(torch.bfloat16)
        normalized_v = value_sums.float().div(divisor).to(torch.bfloat16)
        prepared_q = F.pad(q, (0, args.padded_dim - args.head_dim))
        prepared_q[..., args.head_dim] = 1.0
        prepared_k = F.pad(
            normalized_k, (0, args.padded_dim - args.head_dim)
        )
        prepared_k[..., args.head_dim] = (
            counts.float().log().div(scale).to(prepared_k.dtype)
        )
        prepared_v = F.pad(
            normalized_v, (0, args.padded_dim - args.head_dim)
        )
        return prepared_q, prepared_k, prepared_v

    prepare_ms = timed_ms(prepare_augmented, args.warmups, args.repeats)

    def prepare_and_flash():
        prepared_q, prepared_k, prepared_v = prepare_augmented()
        return flash(prepared_q, prepared_k, prepared_v, scale)

    prepared_flash_ms = timed_ms(
        prepare_and_flash, args.warmups, args.repeats
    )

    # Small explicit reference avoids materializing the full benchmark score field.
    check_q = q[:1, :, :32].float()
    check_k = mean_k[:1, :, :64].float()
    check_v = mean_v[:1, :, :64].float()
    check_count = counts[:1, :, :64].float()
    group = args.query_heads // args.kv_heads
    scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        check_q,
        check_k.repeat_interleave(group, dim=1),
    ).mul_(scale)
    scores.add_(check_count.log().repeat_interleave(group, dim=1).unsqueeze(2))
    reference = torch.einsum(
        "bhqk,bhkd->bhqd",
        torch.softmax(scores, dim=-1),
        check_v.repeat_interleave(group, dim=1),
    )
    check_q_aug = q_aug[:1, :, :32]
    check_k_aug = k_aug[:1, :, :64]
    check_v_aug = v_aug[:1, :, :64]
    flash_output, flash_lse, *_ = flash(
        check_q_aug, check_k_aug, check_v_aug, scale
    )
    check_weighted_numerator, check_unweighted_lse, *_ = flash(
        q[:1, :, :32],
        mean_k[:1, :, :64],
        value_sums[:1, :, :64],
        scale,
    )
    check_expected_count, *_ = flash(
        q[:1, :, :32],
        mean_k[:1, :, :64],
        count_values[:1, :, :64],
        scale,
    )
    check_denominator_ratio = check_expected_count[..., :1].float()
    two_pass_output = check_weighted_numerator.float() / check_denominator_ratio
    two_pass_lse = check_unweighted_lse + check_denominator_ratio.squeeze(-1).log()
    result = {
        "shape": {
            "batch": args.batch_size,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "query_length": args.query_length,
            "key_length": args.key_length,
            "head_dim": args.head_dim,
            "padded_dim": args.padded_dim,
        },
        "milliseconds": {
            "native_d256_without_count_bias": base_ms,
            "native_d264_prebuilt_augmented": augmented_ms,
            "native_d256_two_pass_count_weighted": two_pass_ms,
            "prepare_augmented": prepare_ms,
            "prepare_and_d264_flash": prepared_flash_ms,
        },
        "correctness": {
            "max_abs_output": float(
                (flash_output[..., : args.head_dim].float() - reference)
                .abs()
                .max()
                .item()
            ),
            "two_pass_max_abs_output": float(
                (two_pass_output - reference).abs().max().item()
            ),
            "two_pass_finite_lse": bool(torch.isfinite(two_pass_lse).all().item()),
            "finite_lse": bool(torch.isfinite(flash_lse).all().item()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
