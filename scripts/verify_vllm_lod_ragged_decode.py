#!/usr/bin/env python3
"""Verify stable-request indexing for CUDA-graph-shaped LOD decode."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.kernels.paged_leaf_attention import (
    advance_decode_cache_lengths,
    fused_decode_paged_lod_attention,
    new_fused_decode_buffers,
)


def _slice_cache(value: torch.Tensor, row: int) -> torch.Tensor:
    return value[row : row + 1].clone()


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    cache_rows, query_rows = 3, 2
    query_heads, kv_heads, head_dim = 8, 2, 128
    groups = query_heads // kv_heads
    state_capacity = state_len = 16
    page_size = 16
    page_capacity = state_len
    leaf_capacity = page_capacity * page_size
    local_capacity = 32

    state_k = torch.randn(
        cache_rows,
        kv_heads,
        state_capacity,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    state_v = torch.randn_like(state_k)
    counts = torch.randint(
        1,
        9,
        (cache_rows, kv_heads, state_capacity, 1),
        dtype=torch.int32,
        device=device,
    ).float()
    leaf_k = torch.randn(
        cache_rows,
        kv_heads,
        leaf_capacity,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    leaf_v = torch.randn_like(leaf_k)
    page_indices = (
        torch.arange(leaf_capacity, dtype=torch.int32, device=device)
        .view(1, 1, page_capacity, page_size)
        .expand(cache_rows, kv_heads, -1, -1)
        .clone()
    )
    page_view_k = leaf_k.view(cache_rows, kv_heads, page_capacity, page_size, head_dim)
    page_view_v = leaf_v.view_as(page_view_k)
    page_sum_k = page_view_k.float().sum(dim=3).to(torch.bfloat16)
    page_sum_v = page_view_v.float().sum(dim=3).to(torch.bfloat16)
    page_counts = torch.full(
        (cache_rows, kv_heads, page_capacity),
        page_size,
        dtype=torch.int32,
        device=device,
    )
    slot_pages = (
        torch.arange(state_len, dtype=torch.int32, device=device)
        .view(1, 1, state_len, 1)
        .expand(cache_rows, kv_heads, -1, -1)
        .clone()
    )
    slot_lengths = torch.full(
        (cache_rows, kv_heads, state_len),
        page_size,
        dtype=torch.int32,
        device=device,
    )
    overflow_page_keys = torch.full(
        (cache_rows, kv_heads, 1), -1, dtype=torch.int32, device=device
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    local_k = torch.randn(
        cache_rows,
        kv_heads,
        local_capacity,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    local_v = torch.randn_like(local_k)
    local_lens = torch.tensor([5, 7, 11], dtype=torch.int32, device=device)
    cache_indices = torch.tensor([2, 0], dtype=torch.long, device=device)
    query = torch.randn(
        query_rows,
        query_heads,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    new_k = torch.randn(
        query_rows,
        kv_heads,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    new_v = torch.randn_like(new_k)
    dummy_pages = leaf_k.new_empty(cache_rows, kv_heads, 1, page_size, head_dim)
    recursive = {
        "leaf_k": leaf_k,
        "leaf_v": leaf_v,
        "page_indices": page_indices,
        "page_sum_k": page_sum_k,
        "page_sum_v": page_sum_v,
        "page_counts": page_counts,
        "quantization_finalized": False,
        "summary_quantization_finalized": False,
    }
    common = {
        "state_len": state_len,
        "kv_group_size": groups,
        "scale": head_dim**-0.5,
        "hash_probes": 0,
        "split_kv": 8,
        "fuse_state_route": True,
        "route_group_size": 16,
        "recursive_page_cache": recursive,
    }

    expected = []
    expected_append_k = []
    expected_append_v = []
    for query_row, cache_row in enumerate(cache_indices.tolist()):
        row_local_k = _slice_cache(local_k, cache_row)
        row_local_v = _slice_cache(local_v, cache_row)
        row_recursive = {
            name: _slice_cache(value, cache_row)
            if isinstance(value, torch.Tensor) and value.ndim
            else value
            for name, value in recursive.items()
        }
        row_buffers = new_fused_decode_buffers(
            query[query_row : query_row + 1],
            splits=8,
            state_capacity=state_capacity,
            route_group_size=16,
        )
        expected.append(
            fused_decode_paged_lod_attention(
                query[query_row : query_row + 1],
                _slice_cache(state_k, cache_row),
                _slice_cache(state_v, cache_row),
                _slice_cache(counts, cache_row),
                row_local_k,
                row_local_v,
                _slice_cache(dummy_pages, cache_row),
                _slice_cache(dummy_pages, cache_row),
                _slice_cache(slot_pages, cache_row),
                _slice_cache(overflow_page_keys, cache_row),
                _slice_cache(overflow_page_values, cache_row),
                overflow_used,
                _slice_cache(slot_lengths, cache_row),
                None,
                local_len=int(local_lens[cache_row].item()),
                new_k=new_k[query_row : query_row + 1],
                new_v=new_v[query_row : query_row + 1],
                buffers=row_buffers,
                **{**common, "recursive_page_cache": row_recursive},
            )
        )
        position = int(local_lens[cache_row].item())
        expected_append_k.append(row_local_k[..., position : position + 1, :])
        expected_append_v.append(row_local_v[..., position : position + 1, :])

    mapped_buffers = new_fused_decode_buffers(
        query,
        splits=8,
        state_capacity=state_capacity,
        route_group_size=16,
    )
    actual = fused_decode_paged_lod_attention(
        query,
        state_k,
        state_v,
        counts,
        local_k,
        local_v,
        dummy_pages,
        dummy_pages,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        None,
        local_len=local_capacity,
        local_lens=local_lens,
        cache_indices=cache_indices,
        new_k=new_k,
        new_v=new_v,
        buffers=mapped_buffers,
        **common,
    )
    torch.testing.assert_close(
        actual.float(), torch.cat(expected).float(), rtol=3e-2, atol=1.5e-2
    )
    for query_row, cache_row in enumerate(cache_indices.tolist()):
        position = int(local_lens[cache_row].item())
        torch.testing.assert_close(
            local_k[cache_row : cache_row + 1, ..., position : position + 1, :],
            expected_append_k[query_row],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            local_v[cache_row : cache_row + 1, ..., position : position + 1, :],
            expected_append_v[query_row],
            rtol=0,
            atol=0,
        )
    advance_decode_cache_lengths(cache_indices, local_lens)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        local_lens.cpu(), torch.tensor([6, 7, 12], dtype=torch.int32)
    )
    print("vLLM stable-slot ragged decode parity: PASS")


if __name__ == "__main__":
    main()
