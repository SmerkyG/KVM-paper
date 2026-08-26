#!/usr/bin/env python3
"""Verify query-independent static-cap prefill against dense PyTorch attention."""

from __future__ import annotations

import torch

from model.kernels.paged_leaf_attention import (
    append_virtual_paged_kv,
    static_cap_aiter_paged_leaf_attention,
)


def verify_inline_pages(head_dim: int = 128) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, query_heads, kv_heads = 2, 8, 2
    query_len, slots = 65, 16
    page_size, pages_per_slot, cap = 16, 3, 16
    group_size = query_heads // kv_heads
    page_capacity = slots * pages_per_slot
    leaf_capacity = page_capacity * page_size

    lengths = torch.randint(
        1,
        pages_per_slot * page_size + 1,
        (batch, kv_heads, slots),
        dtype=torch.int32,
        device=device,
    )
    # Guarantee a mixture of included, excluded, and underfull-page experts.
    lengths[..., 0] = 1
    lengths[..., 1] = cap
    lengths[..., 2] = cap + 1
    slot_pages = (
        torch.arange(page_capacity, dtype=torch.int32, device=device)
        .view(slots, pages_per_slot)
        .view(1, 1, slots, pages_per_slot)
        .expand(batch, kv_heads, -1, -1)
        .contiguous()
    )
    page_indices = (
        torch.arange(leaf_capacity, dtype=torch.int32, device=device)
        .view(1, 1, page_capacity, page_size)
        .expand(batch, kv_heads, -1, -1)
        .contiguous()
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, dtype=torch.int32, device=device
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    leaf_k = torch.randn(
        batch,
        kv_heads,
        leaf_capacity,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    leaf_v = torch.randn_like(leaf_k)
    scale = head_dim**-0.5

    with torch.inference_mode():
        actual, actual_lse, token_counts = static_cap_aiter_paged_leaf_attention(
            q,
            leaf_k,
            leaf_v,
            page_indices,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            lengths,
            kv_group_size=group_size,
            scale=scale,
            max_exact_leaves=cap,
            hash_probes=0,
        )

        expected = torch.empty_like(actual)
        expected_lse = torch.empty_like(actual_lse)
        expected_counts: list[int] = []
        for batch_index in range(batch):
            for kv_head in range(kv_heads):
                selected: list[torch.Tensor] = []
                for slot in range(slots):
                    length = int(lengths[batch_index, kv_head, slot].item())
                    if 0 < length <= cap:
                        begin = slot * pages_per_slot * page_size
                        selected.append(
                            torch.arange(begin, begin + length, device=device)
                        )
                indices = torch.cat(selected)
                expected_counts.append(int(indices.numel()))
                keys = leaf_k[batch_index, kv_head].index_select(0, indices).float()
                values = leaf_v[batch_index, kv_head].index_select(0, indices).float()
                for group_head in range(group_size):
                    query_head = kv_head * group_size + group_head
                    scores = (
                        q[batch_index, query_head].float() @ keys.transpose(0, 1)
                    ) * scale
                    expected_lse[batch_index, query_head] = torch.logsumexp(
                        scores, dim=-1
                    )
                    expected[batch_index, query_head] = (
                        torch.softmax(scores, dim=-1) @ values
                    ).to(expected.dtype)
        torch.cuda.synchronize(device)

    count_error = int(
        (
            token_counts
            - torch.tensor(expected_counts, dtype=torch.int32, device=device)
        )
        .abs()
        .max()
        .item()
    )
    output_error = float((actual.float() - expected.float()).abs().max().item())
    lse_error = float((actual_lse - expected_lse).abs().max().item())
    print(f"d{head_dim}_count_max_abs_error={count_error}")
    print(f"d{head_dim}_output_max_abs_error={output_error:.8f}")
    print(f"d{head_dim}_lse_max_abs_error={lse_error:.8f}")
    if count_error or output_error > 0.03 or lse_error > 0.03:
        raise AssertionError("static-cap AITER prefill disagrees with dense attention")


def verify_paged_directory() -> None:
    """Exercise the directory-backed singleton layout used at first catch-up."""
    torch.manual_seed(11)
    device = torch.device("cuda")
    batch, query_heads, kv_heads = 3, 8, 2
    tokens, query_len, slots, head_dim = 256, 17, 1536, 256
    group_size = query_heads // kv_heads
    page_size, leaf_capacity, page_capacity = 16, 4352, 1808
    root_capacity = (leaf_capacity // page_size + 63) // 64

    q = torch.randn(
        batch, query_heads, query_len, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    source_k = torch.randn(
        batch, kv_heads, tokens, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    source_v = torch.randn_like(source_k)
    leaf_k = torch.empty(
        batch, kv_heads, leaf_capacity, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    leaf_v = torch.empty_like(leaf_k)
    leaf_k[..., :tokens, :].copy_(source_k)
    leaf_v[..., :tokens, :].copy_(source_v)
    owners = torch.arange(tokens, dtype=torch.int64, device=device).view(1, 1, -1)
    owners = owners.expand(batch, kv_heads, -1).contiguous()
    page_indices = torch.full(
        (batch, kv_heads, page_capacity, page_size),
        -1, dtype=torch.int32, device=device,
    )
    slot_pages = torch.full(
        (batch, kv_heads, slots, root_capacity),
        -1, dtype=torch.int32, device=device,
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, dtype=torch.int32, device=device,
    )
    overflow_page_values = torch.full(
        (batch, kv_heads, page_capacity, 64),
        -1, dtype=torch.int32, device=device,
    )
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    overflow_flag = torch.zeros_like(overflow_used)
    slot_lengths = torch.zeros(
        batch, kv_heads, slots, dtype=torch.int32, device=device,
    )
    next_page = torch.zeros(batch, kv_heads, dtype=torch.int32, device=device)
    page_sum_k = torch.zeros(
        batch, kv_heads, page_capacity, head_dim,
        dtype=torch.float32, device=device,
    )
    page_sum_v = torch.zeros_like(page_sum_k)
    page_counts = torch.zeros(
        batch, kv_heads, page_capacity, dtype=torch.int32, device=device,
    )
    append_virtual_paged_kv(
        leaf_k, leaf_v, 0, owners, page_indices, slot_pages,
        overflow_page_keys, overflow_page_values, overflow_used, overflow_flag,
        slot_lengths, next_page, page_sum_k, page_sum_v, page_counts,
        hash_probes=-1,
    )
    torch.cuda.synchronize(device)
    written = page_indices.ge(0).sum(dim=(-1, -2))
    if not bool(written.eq(tokens).all().item()):
        raise AssertionError(f"directory append wrote {written.tolist()}, expected {tokens}")

    # Mirror vLLM's transition from a pool-sized catch-up batch to a smaller
    # initial-prefill batch.  Narrowing these trailing dimensions must not pass
    # a strided view to kernels whose row stride is the logical state capacity.
    buffers = {
        "static_prefill_exact_lengths": torch.empty(
            batch + 1, slots * 4, dtype=torch.int32, device=device
        ),
        "static_prefill_slot_offsets": torch.empty(
            batch + 1, slots * 4 + 1, dtype=torch.int32, device=device
        ),
    }
    actual, actual_lse, token_counts = static_cap_aiter_paged_leaf_attention(
        q, leaf_k, leaf_v, page_indices, slot_pages,
        overflow_page_keys, overflow_page_values, overflow_used, slot_lengths,
        kv_group_size=group_size,
        scale=head_dim**-0.5,
        max_exact_leaves=16,
        hash_probes=-1,
        buffers=buffers,
    )
    repeated_k = source_k.repeat_interleave(group_size, dim=1).float()
    repeated_v = source_v.repeat_interleave(group_size, dim=1).float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), repeated_k) * head_dim**-0.5
    expected_lse = torch.logsumexp(scores, dim=-1)
    expected = torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), repeated_v)
    torch.cuda.synchronize(device)
    output_error = float((actual.float() - expected).abs().max().item())
    lse_error = float((actual_lse - expected_lse).abs().max().item())
    count_error = int(token_counts.sub(tokens).abs().max().item())
    print(f"directory_count_max_abs_error={count_error}")
    print(f"directory_output_max_abs_error={output_error:.8f}")
    print(f"directory_lse_max_abs_error={lse_error:.8f}")
    if count_error or output_error > 0.03 or lse_error > 0.03:
        raise AssertionError("directory-backed static prefill disagrees with dense attention")


def main() -> None:
    verify_inline_pages()
    verify_inline_pages(512)
    verify_paged_directory()


if __name__ == "__main__":
    main()
