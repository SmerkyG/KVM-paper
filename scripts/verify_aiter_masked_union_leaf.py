#!/usr/bin/env python3
"""Verify the large-tile masked AITER union against query-major attention."""

from __future__ import annotations

import json

import torch

from model.kernels.paged_leaf_attention import (
    aiter_query_tile_union_paged_leaf_attention,
    query_major_paged_leaf_attention,
)


def main() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda", 0)
    batch, query_heads, kv_heads = 1, 4, 1
    query_len, slots, routes, dimension = 35, 12, 3, 128
    page_size, pages_per_slot = 16, 4
    page_capacity = slots * pages_per_slot
    leaf_capacity = page_capacity * page_size
    lengths = torch.randint(
        1,
        pages_per_slot * page_size + 1,
        (batch, kv_heads, slots),
        device=device,
        dtype=torch.int32,
    )
    lengths[..., :routes] = 0
    slot_pages = torch.arange(
        page_capacity, device=device, dtype=torch.int32
    ).view(batch, kv_heads, slots, pages_per_slot)
    page_indices = torch.arange(
        leaf_capacity, device=device, dtype=torch.int32
    ).view(batch, kv_heads, page_capacity, page_size)
    leaf_k = torch.randn(
        batch,
        kv_heads,
        leaf_capacity,
        dimension,
        device=device,
        dtype=torch.bfloat16,
    )
    leaf_v = torch.randn_like(leaf_k)
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        dimension,
        device=device,
        dtype=torch.bfloat16,
    )
    top_slots = torch.rand(
        batch, query_heads, query_len, slots, device=device
    ).topk(routes, dim=-1, sorted=False).indices
    # Exercise the valid top-3 case where one query has no exact leaves. The
    # coarse branch still owns that row, so exact attention must return 0/-inf.
    top_slots[:, :, 0, :] = torch.arange(routes, device=device)
    overflow_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_values = torch.full_like(overflow_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    scale = dimension**-0.5

    reference_out, reference_lse = query_major_paged_leaf_attention(
        q,
        leaf_k,
        leaf_v,
        slot_pages,
        overflow_keys,
        overflow_values,
        overflow_used,
        lengths,
        top_slots,
        page_indices=page_indices,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        block_n=32,
        num_warps=1,
    )
    actual_out, actual_lse = aiter_query_tile_union_paged_leaf_attention(
        q,
        leaf_k,
        leaf_v,
        slot_pages,
        overflow_keys,
        overflow_values,
        overflow_used,
        lengths,
        top_slots,
        page_indices=page_indices,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        query_tile=16,
        mask_queries=True,
    )
    torch.cuda.synchronize(device)
    finite_lse = torch.isfinite(reference_lse)
    result = {
        "max_output_error": float(
            (actual_out.float() - reference_out.float()).abs().max().item()
        ),
        "max_lse_error": float(
            (actual_lse[finite_lse] - reference_lse[finite_lse]).abs().max().item()
        ),
        "lse_finiteness_matches": bool(
            torch.equal(torch.isfinite(actual_lse), finite_lse)
        ),
    }
    if (
        result["max_output_error"] > 0.03
        or result["max_lse_error"] > 0.01
        or not result["lse_finiteness_matches"]
    ):
        raise AssertionError(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
