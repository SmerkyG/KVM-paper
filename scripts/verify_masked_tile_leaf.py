#!/usr/bin/env python3
"""Compare masked query-tile leaf attention with the query-major reference."""

from __future__ import annotations

import json

import torch

from model.kernels.paged_leaf_attention import (
    query_major_paged_leaf_attention,
    query_tile_masked_paged_leaf_attention,
)


def main() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda", 0)
    batch, query_heads, kv_heads = 1, 4, 1
    query_len, slots, routes, dimension = 17, 8, 3, 64
    page_size, pages_per_slot = 16, 4
    page_capacity = slots * pages_per_slot
    lengths = torch.randint(
        1,
        pages_per_slot * page_size + 1,
        (batch, kv_heads, slots),
        device=device,
        dtype=torch.int32,
    )
    slot_pages = torch.arange(
        page_capacity, device=device, dtype=torch.int32
    ).view(1, 1, slots, pages_per_slot)
    page_k = torch.randn(
        batch,
        kv_heads,
        page_capacity,
        page_size,
        dimension,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.randn_like(page_k)
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
    overflow_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_values = torch.full_like(overflow_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    scale = dimension**-0.5
    reference_out, reference_lse = query_major_paged_leaf_attention(
        q,
        page_k,
        page_v,
        slot_pages,
        overflow_keys,
        overflow_values,
        overflow_used,
        lengths,
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        block_n=32,
        num_warps=1,
    )
    results = {}
    for query_tile in (2, 4, 8):
        actual_out, actual_lse = query_tile_masked_paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_keys,
            overflow_values,
            overflow_used,
            lengths,
            top_slots,
            kv_group_size=query_heads // kv_heads,
            scale=scale,
            hash_probes=0,
            block_m=query_tile,
            block_n=32,
            num_warps=2,
        )
        torch.cuda.synchronize(device)
        output_error = float(
            (actual_out.float() - reference_out.float()).abs().max().item()
        )
        lse_error = float((actual_lse - reference_lse).abs().max().item())
        results[str(query_tile)] = {
            "max_output_error": output_error,
            "max_lse_error": lse_error,
        }
        if output_error > 0.02 or lse_error > 0.005:
            raise AssertionError(results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
