#!/usr/bin/env python3
"""Compare AITER bucketed experts with the ragged Triton implementation."""

from __future__ import annotations

import torch

from model.kernels.paged_leaf_attention import (
    aiter_bucketed_paged_leaf_attention,
    aiter_varlen_paged_leaf_attention,
    paged_leaf_attention,
)


def main() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch, query_heads, kv_heads = 2, 8, 2
    query_len, slots, routes, head_dim = 513, 64, 8, 128
    page_size, pages_per_slot = 16, 8
    group_size = query_heads // kv_heads

    lengths = torch.randint(
        1,
        pages_per_slot * page_size + 1,
        (batch, kv_heads, slots),
        device=device,
        dtype=torch.int32,
    )
    slot_pages = torch.arange(
        slots * pages_per_slot, device=device, dtype=torch.int32
    ).view(1, 1, slots, pages_per_slot).expand(
        batch, kv_heads, slots, pages_per_slot
    ).clone()
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
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
    scale = head_dim**-0.5

    with torch.inference_mode():
        expected_out, expected_lse = paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            lengths,
            top_slots,
            kv_group_size=group_size,
            scale=scale,
            hash_probes=0,
            block_m=16,
            block_n=16,
            num_warps=2,
        )
        actual_out, actual_lse = aiter_bucketed_paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            lengths,
            top_slots,
            kv_group_size=group_size,
            scale=scale,
        )
        varlen_out, varlen_lse = aiter_varlen_paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            lengths,
            top_slots,
            kv_group_size=group_size,
            scale=scale,
        )
        leaf_k = page_k.reshape(batch, kv_heads, -1, head_dim)
        leaf_v = page_v.reshape(batch, kv_heads, -1, head_dim)
        page_indices = torch.arange(
            slots * pages_per_slot * page_size,
            device=device,
            dtype=torch.int32,
        ).view(1, 1, slots * pages_per_slot, page_size)
        page_indices = page_indices.expand(batch, kv_heads, -1, -1).contiguous()
        indexed_out, indexed_lse = aiter_varlen_paged_leaf_attention(
            q,
            leaf_k,
            leaf_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            lengths,
            top_slots,
            page_indices=page_indices,
            kv_group_size=group_size,
            scale=scale,
        )
        copied_out, copied_lse = aiter_varlen_paged_leaf_attention(
            q,
            leaf_k,
            leaf_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            lengths,
            top_slots,
            page_indices=page_indices,
            kv_group_size=group_size,
            scale=scale,
            copy_indexed_kv=True,
            copy_page_size=16,
        )
        torch.cuda.synchronize(device)

    output_error = float(
        (actual_out.float() - expected_out.float()).abs().max().item()
    )
    lse_error = float((actual_lse - expected_lse).abs().max().item())
    varlen_output_error = float(
        (varlen_out.float() - expected_out.float()).abs().max().item()
    )
    varlen_lse_error = float((varlen_lse - expected_lse).abs().max().item())
    indexed_output_error = float(
        (indexed_out.float() - expected_out.float()).abs().max().item()
    )
    indexed_lse_error = float((indexed_lse - expected_lse).abs().max().item())
    copied_output_error = float(
        (copied_out.float() - expected_out.float()).abs().max().item()
    )
    copied_lse_error = float((copied_lse - expected_lse).abs().max().item())
    print(f"output_max_abs_error={output_error:.8f}")
    print(f"lse_max_abs_error={lse_error:.8f}")
    print(f"varlen_output_max_abs_error={varlen_output_error:.8f}")
    print(f"varlen_lse_max_abs_error={varlen_lse_error:.8f}")
    print(f"indexed_output_max_abs_error={indexed_output_error:.8f}")
    print(f"indexed_lse_max_abs_error={indexed_lse_error:.8f}")
    print(f"copied_output_max_abs_error={copied_output_error:.8f}")
    print(f"copied_lse_max_abs_error={copied_lse_error:.8f}")
    if output_error > 0.02 or lse_error > 0.02:
        raise AssertionError("AITER bucketed attention disagrees with Triton")
    if varlen_output_error > 0.02 or varlen_lse_error > 0.02:
        raise AssertionError("AITER varlen attention disagrees with Triton")
    if indexed_output_error > 0.02 or indexed_lse_error > 0.02:
        raise AssertionError("AITER indexed varlen attention disagrees with Triton")
    if copied_output_error > 0.02 or copied_lse_error > 0.02:
        raise AssertionError("AITER copied varlen attention disagrees with Triton")


if __name__ == "__main__":
    main()
