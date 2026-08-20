#!/usr/bin/env python3
"""Verify tile-union correction of an already top-k-pruned coarse branch."""

from __future__ import annotations

import json

import torch

from model.kernels.paged_leaf_attention import (
    query_tile_slot_unions,
    remove_query_tile_union_from_coarse,
)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    batch, query_heads, kv_heads = 2, 4, 2
    query_len, state_capacity, dimension = 9, 11, 32
    kv_group_size, query_tile, route_count = 2, 4, 3
    scale = dimension**-0.5
    q = torch.randn(
        batch, query_heads, query_len, dimension, device=device, dtype=torch.bfloat16
    )
    key_sum = torch.randn(
        batch, kv_heads, state_capacity, dimension, device=device, dtype=torch.bfloat16
    )
    value_sum = torch.randn_like(key_sum)
    counts = torch.randint(
        1,
        9,
        (batch, kv_heads, state_capacity, 1),
        device=device,
        dtype=torch.int32,
    )
    original = torch.randint(
        0,
        state_capacity,
        (batch, query_heads, query_len, route_count),
        device=device,
    )
    union = query_tile_slot_unions(
        original,
        counts.squeeze(-1),
        kv_group_size=kv_group_size,
        query_tile=query_tile,
    )
    mean_key = key_sum.float() / counts.float()
    mean_value = value_sum.float() / counts.float()
    repeated_key = mean_key.repeat_interleave(kv_group_size, dim=1)
    repeated_value = mean_value.repeat_interleave(kv_group_size, dim=1)
    repeated_count = counts.squeeze(-1).repeat_interleave(kv_group_size, dim=1)
    scores = (
        torch.matmul(q.float(), repeated_key.transpose(-1, -2)) * scale
        + repeated_count.float().log().unsqueeze(2)
    )

    def branch(excluded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_slot = torch.arange(state_capacity, device=device)
        mask = (
            (excluded.unsqueeze(-1) == state_slot)
            & (excluded.unsqueeze(-1) >= 0)
        ).any(dim=-2)
        branch_scores = scores.masked_fill(mask, float("-inf"))
        lse = torch.logsumexp(branch_scores, dim=-1)
        out = torch.matmul(torch.softmax(branch_scores, dim=-1), repeated_value)
        return out.to(torch.bfloat16).contiguous(), lse.contiguous()

    original_out, original_lse = branch(original)
    reference_out, reference_lse = branch(union)
    actual_out, actual_lse = remove_query_tile_union_from_coarse(
        q,
        key_sum,
        value_sum,
        counts,
        original,
        union,
        original_out.clone(),
        original_lse.clone(),
        kv_group_size=kv_group_size,
        query_tile=query_tile,
        scale=scale,
    )
    torch.cuda.synchronize(device)
    result = {
        "max_output_error": float(
            (actual_out.float() - reference_out.float()).abs().max().item()
        ),
        "max_lse_error": float((actual_lse - reference_lse).abs().max().item()),
        "union_width": int(union.size(-1)),
    }
    if result["max_output_error"] > 0.02 or result["max_lse_error"] > 0.005:
        raise AssertionError(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
