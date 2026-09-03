#!/usr/bin/env python3
"""Check split-N long-expert attention against the unsplit expert kernel."""

from __future__ import annotations

import torch

from model.kernels.paged_leaf_attention import (
    append_virtual_paged_kv,
    paged_leaf_attention,
)


def main() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(19)
    batch, kv_heads, query_heads = 2, 2, 8
    tokens, query_len, head_dim = 4096, 37, 128
    slots, routes, page_size = 8, 4, 16
    # Leave room for one partially filled tail page per slot.
    page_capacity = tokens // page_size + slots

    leaf_k = torch.randn(
        batch, kv_heads, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    leaf_v = torch.randn_like(leaf_k)
    owners = torch.zeros(batch, kv_heads, tokens, device=device, dtype=torch.long)
    owners[..., 3072:] = (
        torch.arange(tokens - 3072, device=device).remainder(slots - 1) + 1
    )
    page_indices = torch.full(
        (batch, kv_heads, page_capacity, page_size),
        -1,
        device=device,
        dtype=torch.int32,
    )
    slot_pages = torch.full(
        (batch, kv_heads, slots, page_capacity),
        -1,
        device=device,
        dtype=torch.int16,
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    slot_lengths = torch.zeros(
        batch, kv_heads, slots, device=device, dtype=torch.int32
    )
    next_page = torch.zeros(batch, kv_heads, device=device, dtype=torch.int32)
    page_sum_k = torch.zeros(
        batch,
        kv_heads,
        page_capacity,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_sum_v = torch.zeros_like(page_sum_k)
    page_counts = torch.zeros(
        batch, kv_heads, page_capacity, device=device, dtype=torch.int32
    )
    for begin in range(0, tokens, 256):
        append_virtual_paged_kv(
            leaf_k,
            leaf_v,
            begin,
            owners[..., begin : begin + 256],
            page_indices,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            overflow_flag,
            slot_lengths,
            next_page,
            page_sum_k,
            page_sum_v,
            page_counts,
            hash_probes=0,
        )

    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    top_slots = torch.randint(
        slots,
        (batch, query_heads, query_len, routes),
        device=device,
        dtype=torch.long,
    )
    # Ensure every query opens the deliberately long posting list.
    top_slots[..., 0] = 0
    common = dict(
        page_indices=page_indices,
        kv_group_size=query_heads // kv_heads,
        scale=head_dim**-0.5,
        hash_probes=0,
        block_m=16,
        block_n=32,
        num_warps=2,
        reduce_num_warps=1,
    )
    expected_out, expected_lse = paged_leaf_attention(
        q,
        leaf_k,
        leaf_v,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        **common,
    )
    result: dict[str, float] = {}
    for splits in (2, 4, 8):
        actual_out, actual_lse = paged_leaf_attention(
            q,
            leaf_k,
            leaf_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            long_expert_threshold=256,
            long_expert_splits=splits,
            **common,
        )
        output_error = float(
            (actual_out.float() - expected_out.float()).abs().max().item()
        )
        lse_error = float((actual_lse - expected_lse).abs().max().item())
        result[f"split_{splits}_output_max_abs"] = output_error
        result[f"split_{splits}_lse_max_abs"] = lse_error
        if output_error > 0.03 or lse_error > 0.03:
            raise AssertionError(f"split-{splits} disagrees with unsplit attention")
    print(result)


if __name__ == "__main__":
    main()
