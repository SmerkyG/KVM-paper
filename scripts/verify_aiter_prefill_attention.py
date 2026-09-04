#!/usr/bin/env python3
"""Numerical smoke test for AITER weighted coarse prefill attention."""

from __future__ import annotations

import math

import torch

from model.kernels.aiter_prefill_attention import (
    aiter_prefill_coarse_attention,
)
from model.kernels.lod_kernels import route_logits_coarse_attention


def _grouped_logits(
    query: torch.Tensor, key: torch.Tensor, groups: int
) -> torch.Tensor:
    batch, query_heads, query_len, head_dim = query.shape
    kv_heads = int(key.size(1))
    return torch.matmul(
        query.reshape(batch, kv_heads, groups, query_len, head_dim),
        key.transpose(-1, -2).unsqueeze(2),
    ).reshape(batch, query_heads, query_len, int(key.size(2)))


def _reference(
    query: torch.Tensor,
    mean_key: torch.Tensor,
    mean_value: torch.Tensor,
    counts: torch.Tensor,
    slots: torch.Tensor,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(float(query.size(-1)))
    scores = _grouped_logits(query, mean_key, groups).float() * scale
    query_counts = counts[..., 0].repeat_interleave(groups, dim=1)
    scores += query_counts.log().unsqueeze(2)
    excluded = torch.zeros_like(scores, dtype=torch.bool)
    excluded.scatter_(-1, slots, True)
    scores.masked_fill_(excluded, float("-inf"))
    probabilities = scores.softmax(-1).to(mean_value.dtype)
    batch, query_heads, query_len, state_len = probabilities.shape
    kv_heads = int(mean_value.size(1))
    output = torch.matmul(
        probabilities.reshape(
            batch, kv_heads, groups, query_len, state_len
        ),
        mean_value.unsqueeze(2),
    ).reshape(batch, query_heads, query_len, int(mean_value.size(-1)))
    return output, scores.logsumexp(-1)


def _run_case(key_dim: int, value_dim: int, query_normalized: bool) -> None:
    torch.manual_seed(1 + key_dim + value_dim + int(query_normalized))
    batch, kv_heads, groups, query_len, state_len = 1, 2, 2, 32, 64
    query_heads = kv_heads * groups
    query = torch.randn(
        batch,
        query_heads,
        query_len,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    mean_key = torch.randn(
        batch,
        kv_heads,
        state_len,
        key_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    mean_value = torch.randn(
        batch,
        kv_heads,
        state_len,
        value_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    counts = torch.randint(
        1,
        33,
        (batch, kv_heads, state_len, 1),
        device="cuda",
    ).float().contiguous()
    state_value = (mean_value * counts.to(mean_value.dtype)).contiguous()
    query_rms = query.float().square().mean(-1, keepdim=True).sqrt()
    route_query = (
        (query.float() / query_rms).to(query.dtype)
        if query_normalized
        else query
    )
    route_logits = _grouped_logits(route_query, mean_key, groups).contiguous()
    route_scores = route_logits.float() / math.sqrt(float(key_dim))
    route_scores += counts[..., 0].repeat_interleave(groups, 1).log().unsqueeze(2)
    slots = route_scores.topk(4, dim=-1).indices.contiguous()
    expected_out, expected_lse = _reference(
        query, mean_key, mean_value, counts, slots, groups
    )
    if key_dim <= 256 and value_dim == key_dim:
        aiter_out, aiter_lse = aiter_prefill_coarse_attention(
            query,
            mean_key,
            state_value,
            counts,
            slots,
            state_len=state_len,
            kv_group_size=groups,
            scale=1.0 / math.sqrt(float(key_dim)),
        )
        torch.cuda.synchronize()
        aiter_out_error = (aiter_out.float() - expected_out.float()).abs()
        aiter_lse_error = (aiter_lse.float() - expected_lse.float()).abs()
        print(
            "AITER kernel: "
            f"out max={aiter_out_error.max().item():.6f} "
            f"mean={aiter_out_error.mean().item():.6f}; "
            f"lse max={aiter_lse_error.max().item():.6f}"
        )
        assert aiter_out_error.max().item() < 0.07
        assert aiter_out_error.mean().item() < 0.004
        assert aiter_lse_error.max().item() < 0.04

    empty_key = mean_key[..., :0, :].contiguous()
    empty_value = mean_value[..., :0, :].contiguous()
    triton_out, triton_lse = route_logits_coarse_attention(
        query,
        route_logits,
        state_value,
        counts,
        empty_key,
        empty_value,
        slots,
        state_len=state_len,
        kv_group_size=groups,
        scale=1.0 / math.sqrt(float(key_dim)),
        precompute_mean_values=True,
        route_logit_scale=(
            query_rms.contiguous() if query_normalized else None
        ),
    )
    torch.cuda.synchronize()
    triton_out_error = (triton_out.float() - expected_out.float()).abs()
    triton_lse_error = (triton_lse.float() - expected_lse.float()).abs()
    print(
        f"route-logit kernel K={key_dim} V={value_dim} "
        f"normalized={query_normalized}: "
        f"out max={triton_out_error.max().item():.6f} "
        f"mean={triton_out_error.mean().item():.6f}; "
        f"lse max={triton_lse_error.max().item():.6f}"
    )
    assert triton_out_error.max().item() < 0.07
    assert triton_out_error.mean().item() < 0.004
    assert triton_lse_error.max().item() < 0.04


def _verify_strided_local_inputs() -> None:
    from aiter.ops.mha import flash_attn_func

    torch.manual_seed(19)
    batch, query_heads, kv_heads, length, head_dim = 2, 8, 2, 96, 128
    query = torch.randn(
        batch, query_heads, length, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch, kv_heads, length, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    strided = (
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
    )
    packed = tuple(tensor.contiguous() for tensor in strided)
    strided_out, strided_lse = flash_attn_func(
        *strided, causal=True, return_lse=True,
    )
    packed_out, packed_lse = flash_attn_func(
        *packed, causal=True, return_lse=True,
    )
    torch.cuda.synchronize()
    out_error = (strided_out.float() - packed_out.float()).abs().max().item()
    lse_error = (strided_lse.float() - packed_lse.float()).abs().max().item()
    print(
        "AITER strided local inputs: "
        f"out max={out_error:.6f}; lse max={lse_error:.6f}"
    )
    assert out_error < 1e-6
    assert lse_error < 1e-6


def main() -> None:
    _run_case(128, 128, False)
    _run_case(128, 128, True)
    _verify_strided_local_inputs()


if __name__ == "__main__":
    main()
