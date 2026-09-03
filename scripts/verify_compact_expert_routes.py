#!/usr/bin/env python3
"""Compare fixed-width and physically compacted expert route dispatch."""

from __future__ import annotations

import torch

from model.kernels.paged_leaf_attention import (
    paged_leaf_attention,
    query_major_paged_leaf_attention,
)


def main() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda", 0)
    batch, query_heads, kv_heads, query_len = 1, 8, 2, 65
    slots, routes, head_dim = 32, 16, 64
    page_size, pages_per_slot = 16, 2
    group_size = query_heads // kv_heads
    lengths = torch.randint(
        1, pages_per_slot * page_size + 1, (slots,), device=device
    )
    slot_lengths = lengths.view(1, 1, slots).expand(batch, kv_heads, -1).to(
        torch.int32
    ).contiguous()
    slot_pages = torch.arange(
        slots * pages_per_slot, device=device, dtype=torch.int32
    ).view(1, 1, slots, pages_per_slot).expand(batch, kv_heads, -1, -1).contiguous()
    page_k = torch.randn(
        batch,
        kv_heads,
        slots * pages_per_slot,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.randn_like(page_k)
    overflow_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_values = torch.full_like(overflow_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    top_slots = torch.rand(
        batch, query_heads, query_len, slots, device=device
    ).topk(routes, dim=-1, sorted=False).indices
    open_count = torch.arange(query_len, device=device).remainder(routes + 1)
    top_slots = torch.where(
        torch.arange(routes, device=device).view(1, 1, 1, routes)
        < open_count.view(1, 1, query_len, 1),
        top_slots,
        torch.full_like(top_slots, -1),
    ).contiguous()
    kwargs = dict(
        kv_group_size=group_size,
        scale=head_dim**-0.5,
        block_m=16,
        block_n=16,
        num_warps=2,
    )
    fixed_out, fixed_lse = paged_leaf_attention(
        q,
        page_k,
        page_v,
        slot_pages,
        overflow_keys,
        overflow_values,
        overflow_used,
        slot_lengths,
        top_slots,
        **kwargs,
    )
    compact_out, compact_lse = paged_leaf_attention(
        q,
        page_k,
        page_v,
        slot_pages,
        overflow_keys,
        overflow_values,
        overflow_used,
        slot_lengths,
        top_slots,
        compact_invalid_routes=True,
        **kwargs,
    )
    query_out, query_lse = query_major_paged_leaf_attention(
        q,
        page_k,
        page_v,
        slot_pages,
        overflow_keys,
        overflow_values,
        overflow_used,
        slot_lengths,
        top_slots,
        kv_group_size=group_size,
        scale=head_dim**-0.5,
        block_n=16,
        num_warps=2,
        hash_probes=0,
    )
    has_route = top_slots.ge(0).any(dim=-1)
    torch.cuda.synchronize(device)
    output_error = (
        compact_out.float() - fixed_out.float()
    ).abs().masked_fill(~has_route.unsqueeze(-1), 0.0)
    lse_error = (
        compact_lse.float() - fixed_lse.float()
    ).abs().masked_fill(~has_route, 0.0)
    result = {
        "output_max_abs": float(output_error.max().item()),
        "lse_max_abs": float(lse_error.max().item()),
        "query_output_max_abs": float(
            (compact_out.float() - query_out.float())
            .abs()
            .masked_fill(~has_route.unsqueeze(-1), 0.0)
            .max()
            .item()
        ),
        "query_lse_max_abs": float(
            (compact_lse.float() - query_lse.float())
            .abs()
            .masked_fill(~has_route, 0.0)
            .max()
            .item()
        ),
        "compact_finite": bool(compact_out[has_route].isfinite().all().item()),
    }
    print(result)
    if (
        result["output_max_abs"] > 0.02
        or result["lse_max_abs"] > 0.02
        or result["query_output_max_abs"] > 0.03
        or result["query_lse_max_abs"] > 0.03
        or not result["compact_finite"]
    ):
        raise AssertionError("compacted expert dispatch differs from fixed width")


if __name__ == "__main__":
    main()
