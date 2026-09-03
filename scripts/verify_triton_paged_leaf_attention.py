#!/usr/bin/env python3
"""Compare the forward-only Triton page kernel with packed exact attention."""

from __future__ import annotations

import math
import os

import torch

from model.kernels.paged_leaf_attention import (
    append_paged_int8_kv,
    append_paged_kv,
    append_quantized_virtual_paged_kv,
    append_virtual_paged_kv,
    dense_page_summary_attention,
    fused_decode_paged_lod_attention,
    materialize_page_summary_scores_gqa,
    new_fused_decode_buffers,
    paged_leaf_attention,
    query_major_paged_leaf_attention,
    query_major_indexed_residual_page_attention,
    query_major_residual_page_attention,
    quantize_page_summaries_int8,
    quantize_virtual_paged_kv,
    quantize_virtual_paged_kv_int4,
)
from model.kernels.lod_kernels import merge_attention_branches_with_sink
from model.kvm_two_level_mixer import _expert_leaf_attention, _merge_lse_branches


def verify_fused_sink_branch_merge(device: torch.device) -> None:
    batch, query_heads, kv_heads = 2, 8, 2
    query_len, head_dim, sink_len = 17, 128, 3
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    sink_k = torch.randn(
        batch,
        kv_heads,
        sink_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    sink_v = torch.randn_like(sink_k)
    outputs = [torch.randn_like(q) for _ in range(2)]
    lses = [torch.randn(batch, query_heads, query_len, device=device) for _ in range(2)]
    # Exercise non-contiguous query slices from split local prefill attention.
    tertiary_storage = torch.randn(
        batch,
        query_heads,
        query_len + 3,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    tertiary_lse_storage = torch.randn(batch, query_heads, query_len + 3, device=device)
    tertiary_output = tertiary_storage[..., 3:, :]
    tertiary_lse = tertiary_lse_storage[..., 3:]
    scale = head_dim**-0.5
    repeated_k = sink_k.repeat_interleave(query_heads // kv_heads, dim=1)
    repeated_v = sink_v.repeat_interleave(query_heads // kv_heads, dim=1)
    sink_scores = torch.matmul(q.float(), repeated_k.float().transpose(-1, -2))
    sink_scores.mul_(scale)
    sink_lse = torch.logsumexp(sink_scores, dim=-1)
    sink_output = torch.matmul(sink_scores.softmax(dim=-1), repeated_v.float())
    branch_lse = torch.stack((*lses, tertiary_lse, sink_lse), dim=-1)
    weights = branch_lse.softmax(dim=-1)
    expected = (
        outputs[0].float() * weights[..., 0, None]
        + outputs[1].float() * weights[..., 1, None]
        + tertiary_output.float() * weights[..., 2, None]
        + sink_output * weights[..., 3, None]
    ).to(q.dtype)
    actual = merge_attention_branches_with_sink(
        q,
        sink_k,
        sink_v,
        outputs[0],
        lses[0],
        outputs[1],
        lses[1],
        tertiary_output,
        tertiary_lse,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=8e-3)


def verify_dense_page_summary_attention(device: torch.device) -> None:
    batch, query_heads, kv_heads = 1, 4, 1
    query_len, pages, page_size, head_dim = 17, 64, 16, 128
    topk = 4
    union_query_tile = 32
    tokens = pages * page_size
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    leaf_k = torch.randn(
        batch, kv_heads, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    leaf_v = torch.randn_like(leaf_k)
    counts = torch.full(
        (batch, kv_heads, pages), page_size, device=device, dtype=torch.int32
    )
    counts[..., -1] = 7
    page_view_k = leaf_k.view(batch, kv_heads, pages, page_size, head_dim)
    page_view_v = leaf_v.view_as(page_view_k)
    valid = torch.arange(page_size, device=device).view(
        1, 1, 1, page_size, 1
    ) < counts.unsqueeze(-1).unsqueeze(-1)
    page_sum_k = torch.where(valid, page_view_k, 0).float().sum(3).to(leaf_k.dtype)
    page_sum_v = torch.where(valid, page_view_v, 0).float().sum(3).to(leaf_v.dtype)
    page_indices = torch.arange(tokens, device=device, dtype=torch.int32).view(
        1, 1, pages, page_size
    )
    page_indices = torch.where(valid.squeeze(-1), page_indices, -1)
    next_page = torch.full((batch, kv_heads), pages, device=device, dtype=torch.int32)
    scale = head_dim**-0.5
    residual, residual_lse, exact, exact_lse, selected = dense_page_summary_attention(
        q,
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        counts,
        next_page,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        top_pages=topk,
        block_m=16,
        block_n=16,
        num_warps=4,
    )

    repeated_counts = counts.repeat_interleave(query_heads, dim=1)
    means_k = (
        page_sum_k.float() / counts.clamp_min(1).unsqueeze(-1)
    ).repeat_interleave(query_heads, dim=1)
    means_v = (
        page_sum_v.float() / counts.clamp_min(1).unsqueeze(-1)
    ).repeat_interleave(query_heads, dim=1)
    page_scores = torch.matmul(q.float(), means_k.transpose(-1, -2)) * scale
    page_scores += repeated_counts.float().log().unsqueeze(2)
    reference_scores, reference_pages = page_scores.topk(topk, dim=-1)
    residual_scores = page_scores.clone()
    residual_scores.scatter_(-1, reference_pages, float("-inf"))
    reference_residual_lse = torch.logsumexp(residual_scores, dim=-1)
    reference_residual = torch.matmul(torch.softmax(residual_scores, dim=-1), means_v)

    selected_counts = torch.gather(
        repeated_counts.unsqueeze(2).expand(-1, -1, query_len, -1),
        -1,
        reference_pages,
    )
    token_offset = torch.arange(page_size, device=device)
    token_index = reference_pages.unsqueeze(-1) * page_size + token_offset
    token_valid = token_offset < selected_counts.unsqueeze(-1)
    flat_index = token_index.flatten(-2)
    repeated_k = leaf_k.repeat_interleave(query_heads, dim=1)
    repeated_v = leaf_v.repeat_interleave(query_heads, dim=1)
    expanded_k = repeated_k.unsqueeze(2).expand(-1, -1, query_len, -1, -1)
    expanded_v = repeated_v.unsqueeze(2).expand_as(expanded_k)
    gathered_k = torch.gather(
        expanded_k,
        3,
        flat_index.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim),
    )
    gathered_v = torch.gather(
        expanded_v,
        3,
        flat_index.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim),
    )
    exact_scores = (q.float().unsqueeze(-2) * gathered_k.float()).sum(-1) * scale
    exact_scores.masked_fill_(~token_valid.flatten(-2), float("-inf"))
    reference_exact_lse = torch.logsumexp(exact_scores, dim=-1)
    reference_exact = torch.matmul(
        torch.softmax(exact_scores, dim=-1).unsqueeze(-2), gathered_v.float()
    ).squeeze(-2)

    torch.cuda.synchronize(device)
    if not torch.equal(
        selected.long().sort(dim=-1).values,
        reference_pages.sort(dim=-1).values,
    ):
        raise AssertionError("dense page attention selected the wrong pages")
    torch.testing.assert_close(
        residual.float(), reference_residual, rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        residual_lse.float(), reference_residual_lse, rtol=0.002, atol=0.01
    )
    torch.testing.assert_close(exact.float(), reference_exact, rtol=0.02, atol=0.02)
    torch.testing.assert_close(
        exact_lse.float(), reference_exact_lse, rtol=0.002, atol=0.01
    )

    (
        union_residual,
        union_residual_lse,
        union_exact,
        union_exact_lse,
        union_selected,
    ) = dense_page_summary_attention(
        q,
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        counts,
        next_page,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        top_pages=topk,
        block_m=16,
        block_n=16,
        num_warps=4,
        indexed_aiter_union=True,
        union_query_tile=union_query_tile,
    )
    reference_union_residual = torch.empty_like(reference_residual)
    reference_union_residual_lse = torch.empty_like(reference_residual_lse)
    reference_union_exact = torch.empty_like(reference_exact)
    reference_union_exact_lse = torch.empty_like(reference_exact_lse)
    for query_begin in range(0, query_len, union_query_tile):
        query_end = min(query_begin + union_query_tile, query_len)
        union = torch.unique(reference_pages[:, :, query_begin:query_end])
        union_page_scores = page_scores[:, :, query_begin:query_end, union]
        kept_scores = page_scores[:, :, query_begin:query_end].clone()
        kept_scores[..., union] = float("-inf")
        reference_union_residual_lse[:, :, query_begin:query_end] = torch.logsumexp(
            kept_scores, dim=-1
        )
        reference_union_residual[:, :, query_begin:query_end] = torch.matmul(
            torch.softmax(kept_scores, dim=-1), means_v
        )
        union_counts = counts[..., union]
        union_offsets = torch.arange(page_size, device=device)
        union_token_indices = (
            union[:, None] * page_size + union_offsets[None, :]
        ).reshape(-1)
        union_token_valid = (
            union_offsets[None, :] < union_counts.reshape(-1, 1)
        ).reshape(-1)
        union_token_indices = union_token_indices[union_token_valid]
        union_keys = leaf_k[..., union_token_indices, :].repeat_interleave(
            query_heads, dim=1
        )
        union_values = leaf_v[..., union_token_indices, :].repeat_interleave(
            query_heads, dim=1
        )
        tile_q = q[:, :, query_begin:query_end]
        union_scores = (
            torch.matmul(tile_q.float(), union_keys.float().transpose(-1, -2)) * scale
        )
        reference_union_exact_lse[:, :, query_begin:query_end] = torch.logsumexp(
            union_scores, dim=-1
        )
        reference_union_exact[:, :, query_begin:query_end] = torch.matmul(
            torch.softmax(union_scores, dim=-1), union_values.float()
        )
    torch.cuda.synchronize(device)
    if not torch.equal(
        union_selected.long().sort(dim=-1).values,
        reference_pages.sort(dim=-1).values,
    ):
        raise AssertionError("tile-union dense attention selected the wrong pages")
    torch.testing.assert_close(
        union_residual_lse.float(),
        reference_union_residual_lse,
        rtol=0.003,
        atol=0.015,
    )
    torch.testing.assert_close(
        union_exact.float(), reference_union_exact, rtol=0.025, atol=0.025
    )
    torch.testing.assert_close(
        union_exact_lse.float(),
        reference_union_exact_lse,
        rtol=0.002,
        atol=0.01,
    )
    # Union removal reconstructs the residual numerator from the BF16 full
    # summary output.  Its normalized value is ill-conditioned when the union
    # owns almost all summary mass, but that branch is downweighted by exactly
    # the same small mass during LSE merging.  Check the mass-weighted
    # numerator and the observable merged output instead of magnifying that
    # harmless cancellation in the standalone residual value.
    page_normalizer = torch.maximum(
        reference_union_residual_lse, reference_union_exact_lse
    )
    actual_residual_numerator = union_residual.float() * torch.exp(
        union_residual_lse.float().unsqueeze(-1) - page_normalizer.unsqueeze(-1)
    )
    reference_residual_numerator = reference_union_residual * torch.exp(
        reference_union_residual_lse.unsqueeze(-1) - page_normalizer.unsqueeze(-1)
    )
    torch.testing.assert_close(
        actual_residual_numerator,
        reference_residual_numerator,
        rtol=0.04,
        atol=0.025,
    )
    actual_page_weights = torch.softmax(
        torch.stack((union_residual_lse, union_exact_lse), dim=-1), dim=-1
    )
    actual_page_output = (
        union_residual.float() * actual_page_weights[..., 0, None]
        + union_exact.float() * actual_page_weights[..., 1, None]
    )
    reference_page_weights = torch.softmax(
        torch.stack((reference_union_residual_lse, reference_union_exact_lse), dim=-1),
        dim=-1,
    )
    reference_page_output = (
        reference_union_residual * reference_page_weights[..., 0, None]
        + reference_union_exact * reference_page_weights[..., 1, None]
    )
    torch.testing.assert_close(
        actual_page_output, reference_page_output, rtol=0.035, atol=0.025
    )

    split_residual, split_residual_lse, split_exact, split_exact_lse, split_selected = (
        dense_page_summary_attention(
            q,
            leaf_k,
            leaf_v,
            page_indices,
            page_sum_k,
            page_sum_v,
            counts,
            next_page,
            kv_group_size=query_heads // kv_heads,
            scale=scale,
            top_pages=topk,
            block_m=16,
            block_n=16,
            num_warps=4,
            split_kernels=True,
        )
    )
    torch.cuda.synchronize(device)
    if not torch.equal(
        split_selected.long().sort(dim=-1).values,
        reference_pages.sort(dim=-1).values,
    ):
        raise AssertionError("split dense page attention selected the wrong pages")
    torch.testing.assert_close(
        split_residual.float(), reference_residual, rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        split_residual_lse.float(), reference_residual_lse, rtol=0.002, atol=0.01
    )
    torch.testing.assert_close(
        split_exact.float(), reference_exact, rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        split_exact_lse.float(), reference_exact_lse, rtol=0.002, atol=0.01
    )


def verify_page_append(device: torch.device) -> None:
    batch, kv_heads, slots, tokens, head_dim = 1, 4, 64, 256, 256
    page_size, inline_pages_per_slot, page_capacity = 16, 2, 256
    page_k = torch.zeros(
        batch,
        kv_heads,
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.zeros_like(page_k)
    slot_pages = torch.full(
        (batch, kv_heads, slots, inline_pages_per_slot),
        -1,
        device=device,
        dtype=torch.int16,
    )
    slot_lengths = torch.zeros(batch, kv_heads, slots, device=device, dtype=torch.int32)
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1024), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    next_page = torch.zeros(batch, kv_heads, device=device, dtype=torch.int32)
    chunks = []
    for chunk in range(2):
        owners = torch.randint(
            4, (batch, kv_heads, tokens), device=device, dtype=torch.long
        )
        k = torch.randn(
            batch,
            kv_heads,
            tokens,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        token_id = torch.arange(tokens, device=device)
        k[..., 0] = (token_id % 128).to(k.dtype)
        k[..., 1] = (token_id // 128 + chunk * 2).to(k.dtype)
        v = torch.randn_like(k)
        append_paged_kv(
            k,
            v,
            owners,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            overflow_flag,
            slot_lengths,
            next_page,
        )
        chunks.append((owners, k, v))
    torch.cuda.synchronize(device)
    if int(overflow_flag.item()) != 0:
        raise AssertionError("sparse overflow page hash ran out of probes")

    def hash_index(key: int, capacity: int) -> int:
        value = key & 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        value = (value * 0x846CA68B) & 0xFFFFFFFF
        value ^= value >> 16
        return value & (capacity - 1)

    def page_id_for(head: int, slot: int, page_ordinal: int) -> int:
        if page_ordinal < inline_pages_per_slot:
            return int(slot_pages[0, head, slot, page_ordinal].item())
        key = slot * 65_536 + page_ordinal
        index = hash_index(key, int(overflow_page_keys.size(2)))
        for _ in range(8):
            if int(overflow_page_keys[0, head, index].item()) == key:
                return int(overflow_page_values[0, head, index].item())
            index = (index + 1) & (int(overflow_page_keys.size(2)) - 1)
        return -1

    for head in range(kv_heads):
        expected_counts = torch.cat(
            [owners[0, head] for owners, _, _ in chunks]
        ).bincount(minlength=slots)
        if not torch.equal(slot_lengths[0, head].long(), expected_counts):
            raise AssertionError("Triton page append produced incorrect slot lengths")
        expected_pages = int(((expected_counts + page_size - 1) // page_size).sum())
        if int(next_page[0, head].item()) != expected_pages:
            raise AssertionError("Triton page append produced incorrect page count")
        for slot in range(slots):
            count = int(expected_counts[slot].item())
            pages = (count + page_size - 1) // page_size
            page_ids = torch.tensor(
                [page_id_for(head, slot, ordinal) for ordinal in range(pages)],
                device=device,
                dtype=torch.long,
            )
            if bool((page_ids < 0).any().item()):
                raise AssertionError("Triton page append left a missing page")
            actual_k = page_k[0, head].index_select(0, page_ids).flatten(0, 1)[:count]
            expected_k = torch.cat(
                [k[0, head][owners[0, head] == slot] for owners, k, _ in chunks]
            )
            if not torch.equal(actual_k, expected_k):
                raise AssertionError(
                    "Triton page append did not preserve chronological region order"
                )


def verify_two_level_page_directory(device: torch.device) -> None:
    """Cross multiple directory pages without a hash or oversized root."""
    batch, kv_heads, slots, tokens, head_dim = 1, 2, 4, 2080, 64
    page_size, directory_size = 16, 64
    page_capacity = 256
    page_k = torch.zeros(
        batch,
        kv_heads,
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.zeros_like(page_k)
    # Four root entries address 256 physical pages, but every populated root
    # entry occupies only one int32 per centroid.
    slot_pages = torch.full(
        (batch, kv_heads, slots, 4),
        -1,
        device=device,
        dtype=torch.int32,
    )
    page_directory = torch.full(
        (batch, kv_heads, page_capacity, directory_size),
        -1,
        device=device,
        dtype=torch.int32,
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    directory_next_page = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    slot_lengths = torch.zeros(batch, kv_heads, slots, device=device, dtype=torch.int32)
    next_page = torch.zeros(batch, kv_heads, device=device, dtype=torch.int32)
    owners = torch.zeros(batch, kv_heads, tokens, device=device, dtype=torch.long)
    k = torch.randn(
        batch,
        kv_heads,
        tokens,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    append_paged_kv(
        k,
        v,
        owners,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        page_directory,
        directory_next_page,
        overflow_flag,
        slot_lengths,
        next_page,
        hash_probes=-1,
    )
    torch.cuda.synchronize(device)
    pages = math.ceil(tokens / page_size)
    if int(overflow_flag.item()) != 0:
        raise AssertionError("two-level page directory exhausted its hard capacity")
    for head in range(kv_heads):
        physical_pages = []
        for page_ordinal in range(pages):
            directory_id = int(
                slot_pages[0, head, 0, page_ordinal // directory_size].item()
            )
            physical_pages.append(
                int(
                    page_directory[
                        0,
                        head,
                        directory_id,
                        page_ordinal % directory_size,
                    ].item()
                )
            )
        page_ids = torch.tensor(physical_pages, device=device, dtype=torch.long)
        restored_k = page_k[0, head].index_select(0, page_ids).flatten(0, 1)[:tokens]
        restored_v = page_v[0, head].index_select(0, page_ids).flatten(0, 1)[:tokens]
        torch.testing.assert_close(restored_k, k[0, head])
        torch.testing.assert_close(restored_v, v[0, head])

    query_len = 8
    q = torch.randn(
        batch,
        kv_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    top_slots = torch.zeros(
        batch, kv_heads, query_len, 1, device=device, dtype=torch.long
    )
    scale = head_dim**-0.5
    actual, actual_lse = paged_leaf_attention(
        q,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        page_directory,
        directory_next_page,
        slot_lengths,
        top_slots,
        kv_group_size=1,
        scale=scale,
        hash_probes=-1,
        block_m=16,
        block_n=32,
        num_warps=2,
    )
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    expected = torch.matmul(scores.softmax(dim=-1), v.float()).to(q.dtype)
    expected_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-2, atol=2e-2)


def verify_sealed_page_append(device: torch.device) -> None:
    cap = 32
    page_size = 16
    page_k = torch.zeros(1, 1, 8, page_size, 64, device=device, dtype=torch.bfloat16)
    page_v = torch.zeros_like(page_k)
    slot_pages = torch.full(
        (1, 1, 4, cap // page_size),
        -1,
        device=device,
        dtype=torch.int16,
    )
    slot_lengths = torch.zeros(1, 1, 4, device=device, dtype=torch.int32)
    overflow_page_keys = torch.full((1, 1, 1), -1, device=device, dtype=torch.int32)
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    next_page = torch.zeros(1, 1, device=device, dtype=torch.int32)
    expected = []
    for chunk in range(2):
        tokens = 24
        owners = torch.zeros(1, 1, tokens, device=device, dtype=torch.long)
        k = torch.zeros(1, 1, tokens, 64, device=device, dtype=torch.bfloat16)
        k[..., 0] = torch.arange(
            chunk * tokens,
            (chunk + 1) * tokens,
            device=device,
            dtype=k.dtype,
        )
        v = -k
        append_paged_kv(
            k,
            v,
            owners,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            overflow_flag,
            slot_lengths,
            next_page,
            hash_probes=0,
            max_leaf_tokens=cap,
        )
        expected.append(k)
    torch.cuda.synchronize(device)
    if int(slot_lengths[0, 0, 0].item()) != cap:
        raise AssertionError("sealed page append did not stop at its leaf cap")
    if int(next_page[0, 0].item()) != cap // page_size:
        raise AssertionError("sealed page append allocated pages beyond its cap")
    page_ids = slot_pages[0, 0, 0].long()
    actual = page_k[0, 0].index_select(0, page_ids).flatten(0, 1)
    reference = torch.cat(expected, dim=2)[0, 0, :cap]
    if not torch.equal(actual, reference):
        raise AssertionError("sealed page append did not retain the earliest leaves")


def verify_int8_mma_page_attention(device: torch.device) -> None:
    # Use several batches and a non-tile-aligned query length so blocked INT8
    # query preparation exercises row-to-(batch, head, token) indexing and its
    # masked tail, not only the trivial single-batch case.
    batch, query_heads, kv_heads = 3, 4, 1
    tokens, query_len, head_dim = 256, 35, 256
    slots, routes, page_size = 4, 2, 16
    page_capacity = tokens // page_size + slots

    def metadata(dtype: torch.dtype):
        page_k = torch.zeros(
            batch,
            kv_heads,
            page_capacity,
            page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
        page_v = torch.zeros_like(page_k)
        slot_pages = torch.full(
            (batch, kv_heads, slots, tokens // page_size),
            -1,
            device=device,
            dtype=torch.int16,
        )
        slot_lengths = torch.zeros(
            batch, kv_heads, slots, device=device, dtype=torch.int32
        )
        overflow_keys = torch.full(
            (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
        )
        return (
            page_k,
            page_v,
            slot_pages,
            overflow_keys,
            torch.full_like(overflow_keys, -1),
            torch.zeros((), device=device, dtype=torch.int32),
            torch.zeros((), device=device, dtype=torch.int32),
            slot_lengths,
            torch.zeros(batch, kv_heads, device=device, dtype=torch.int32),
        )

    torch.manual_seed(7)
    k = torch.randn(
        batch, kv_heads, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    owners = (
        torch.arange(tokens, device=device)
        .remainder(slots)
        .view(1, 1, -1)
        .expand(batch, kv_heads, tokens)
        .contiguous()
    )
    bf16 = metadata(torch.bfloat16)
    int8 = metadata(torch.int8)
    append_paged_kv(k, v, owners, *bf16, hash_probes=0)
    k_scales = torch.zeros(*int8[0].shape[:-1], device=device, dtype=torch.bfloat16)
    v_scales = torch.zeros_like(k_scales)
    append_paged_int8_kv(
        k,
        v,
        owners,
        int8[0],
        int8[1],
        k_scales,
        v_scales,
        *int8[2:],
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
    scale = head_dim**-0.5
    expected, expected_lse = paged_leaf_attention(
        q,
        *bf16[:2],
        *bf16[2:6],
        bf16[7],
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        block_m=16,
        block_n=32,
        num_warps=2,
    )
    actual, actual_lse = paged_leaf_attention(
        q,
        *int8[:2],
        *int8[2:6],
        int8[7],
        top_slots,
        page_k_scales=k_scales,
        page_v_scales=v_scales,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        block_m=16,
        block_n=32,
        num_warps=2,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.03)
    torch.testing.assert_close(
        actual_lse.float(), expected_lse.float(), rtol=0.01, atol=0.03
    )


def verify_virtual_page_append(device: torch.device) -> None:
    batch, kv_heads, slots, tokens, head_dim = 1, 2, 64, 512, 256
    page_size, inline_pages_per_slot, page_capacity = 16, 32, 128
    leaf_k = torch.randn(
        batch, kv_heads, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    leaf_v = torch.randn_like(leaf_k)
    owners = torch.randint(
        4, (batch, kv_heads, tokens), device=device, dtype=torch.long
    )
    page_indices = torch.full(
        (batch, kv_heads, page_capacity, page_size),
        -1,
        device=device,
        dtype=torch.int32,
    )
    page_sum_k = torch.zeros(
        batch,
        kv_heads,
        page_capacity,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_sum_v = torch.zeros_like(page_sum_k)
    decode_tokens = 64
    packed_capacity = tokens + decode_tokens
    quantized_leaf_k = torch.empty(
        batch,
        kv_heads,
        packed_capacity,
        head_dim // 2,
        device=device,
        dtype=torch.uint8,
    )
    quantized_leaf_v = torch.empty_like(quantized_leaf_k)
    l2_quantized_leaf_k = torch.empty_like(quantized_leaf_k)
    l2_quantized_leaf_v = torch.empty_like(quantized_leaf_v)
    page_k_scales = torch.empty(
        batch,
        kv_heads,
        page_capacity,
        head_dim // 32,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v_scales = torch.empty_like(page_k_scales)
    l2_page_k_scales = torch.empty_like(page_k_scales)
    l2_page_v_scales = torch.empty_like(page_v_scales)
    page_counts = torch.zeros(
        batch, kv_heads, page_capacity, device=device, dtype=torch.int32
    )
    page_quantized_counts = torch.zeros_like(page_counts)
    l2_page_quantized_counts = torch.zeros_like(page_counts)
    slot_pages = torch.full(
        (batch, kv_heads, slots, inline_pages_per_slot),
        -1,
        device=device,
        dtype=torch.int16,
    )
    slot_lengths = torch.zeros(batch, kv_heads, slots, device=device, dtype=torch.int32)
    overflow_page_keys = torch.full(
        (batch, kv_heads, 16), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    next_page = torch.zeros(batch, kv_heads, device=device, dtype=torch.int32)
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
            quantized_leaf_k=quantized_leaf_k,
            quantized_leaf_v=quantized_leaf_v,
            page_k_scales=page_k_scales,
            page_v_scales=page_v_scales,
            page_quantized_counts=page_quantized_counts,
            quantize_touched=False,
        )
    if bool(page_quantized_counts.any().item()):
        raise AssertionError("deferred virtual INT4 quantized before finalization")
    quantize_virtual_paged_kv_int4(
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
    )
    quantize_virtual_paged_kv_int4(
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        l2_quantized_leaf_k,
        l2_quantized_leaf_v,
        l2_page_k_scales,
        l2_page_v_scales,
        l2_page_quantized_counts,
        optimize_scale=True,
    )
    torch.cuda.synchronize(device)
    max_leaf_squared_errors = []
    l2_leaf_squared_errors = []
    for head in range(kv_heads):
        expected_counts = owners[0, head].bincount(minlength=slots)
        torch.testing.assert_close(slot_lengths[0, head].long(), expected_counts)
        for slot in range(4):
            count = int(expected_counts[slot].item())
            pages = (count + page_size - 1) // page_size
            page_ids = slot_pages[0, head, slot, :pages].long()
            actual_indices = (
                page_indices[0, head].index_select(0, page_ids).flatten()[:count].long()
            )
            torch.testing.assert_close(
                page_quantized_counts[0, head].index_select(0, page_ids).long(),
                page_counts[0, head].index_select(0, page_ids).long(),
            )
            expected_indices = torch.nonzero(
                owners[0, head] == slot, as_tuple=False
            ).flatten()
            if not torch.equal(actual_indices, expected_indices):
                raise AssertionError(
                    "virtual page append changed chronological leaf order"
                )
            if not torch.equal(
                actual_indices.sort().values, expected_indices.sort().values
            ):
                raise AssertionError("virtual page append lost a leaf index")
            actual_key_sum = page_sum_k[0, head].index_select(0, page_ids).sum(0)
            actual_value_sum = page_sum_v[0, head].index_select(0, page_ids).sum(0)
            torch.testing.assert_close(
                actual_key_sum.float(),
                leaf_k[0, head].index_select(0, expected_indices).float().sum(0),
                rtol=2e-2,
                atol=2.5e-1,
            )
            torch.testing.assert_close(
                actual_value_sum.float(),
                leaf_v[0, head].index_select(0, expected_indices).float().sum(0),
                rtol=2e-2,
                atol=2.5e-1,
            )
            for page_id_tensor in page_ids:
                page_id = int(page_id_tensor.item())
                page_count = int(page_counts[0, head, page_id].item())
                indices = page_indices[0, head, page_id, :page_count].long()
                for (
                    source,
                    packed_cache,
                    scales_cache,
                    l2_packed_cache,
                    l2_scales_cache,
                    page_sum,
                ) in (
                    (
                        leaf_k,
                        quantized_leaf_k,
                        page_k_scales,
                        l2_quantized_leaf_k,
                        l2_page_k_scales,
                        page_sum_k,
                    ),
                    (
                        leaf_v,
                        quantized_leaf_v,
                        page_v_scales,
                        l2_quantized_leaf_v,
                        l2_page_v_scales,
                        page_sum_v,
                    ),
                ):
                    values = source[0, head].index_select(0, indices).float()
                    anchor = page_sum[0, head, page_id].float() / page_count
                    residual = (values - anchor).reshape(page_count, -1, 32)
                    scale = residual.abs().amax(dim=(0, 2)) / 7
                    scale = scale.clamp_min(1.0e-8)
                    code = torch.floor(residual / scale[None, :, None] + 0.5)
                    code = code.clamp(-7, 7).to(torch.int32) + 8
                    expected_packed = (
                        (code[..., 0::2] | (code[..., 1::2] << 4))
                        .reshape(page_count, -1)
                        .to(torch.uint8)
                    )
                    actual_packed = packed_cache[0, head].index_select(0, indices)
                    actual_low = (actual_packed.to(torch.int32) & 15) - 8
                    actual_high = (actual_packed.to(torch.int32) >> 4) - 8
                    actual_code = torch.stack(
                        (actual_low, actual_high), dim=-1
                    ).reshape(page_count, -1, 32)
                    if int((actual_code - (code - 8)).abs().max().item()) > 1:
                        raise AssertionError(
                            "virtual INT4 code differs by more than one"
                        )
                    torch.testing.assert_close(
                        scales_cache[0, head, page_id].float(),
                        scale,
                        rtol=8e-3,
                        atol=1e-5,
                    )
                    l2_packed = l2_packed_cache[0, head].index_select(0, indices)
                    l2_low = (l2_packed.to(torch.int32) & 15) - 8
                    l2_high = (l2_packed.to(torch.int32) >> 4) - 8
                    l2_code = torch.stack((l2_low, l2_high), dim=-1).reshape(
                        page_count, -1, 32
                    )
                    max_reconstruction = actual_code.float() * scales_cache[
                        0, head, page_id
                    ].float()[None, :, None] + anchor.reshape(1, -1, 32)
                    l2_reconstruction = l2_code.float() * l2_scales_cache[
                        0, head, page_id
                    ].float()[None, :, None] + anchor.reshape(1, -1, 32)
                    target = values.reshape(page_count, -1, 32)
                    max_leaf_squared_errors.append(
                        (max_reconstruction - target).square().flatten()
                    )
                    l2_leaf_squared_errors.append(
                        (l2_reconstruction - target).square().flatten()
                    )

    max_leaf_mse = torch.cat(max_leaf_squared_errors).mean()
    l2_leaf_mse = torch.cat(l2_leaf_squared_errors).mean()
    if l2_leaf_mse > max_leaf_mse:
        raise AssertionError("L2 INT4 scale increased leaf reconstruction error")
    print(
        {
            "virtual_int4_max_mse": float(max_leaf_mse.item()),
            "virtual_int4_l2_mse": float(l2_leaf_mse.item()),
            "virtual_int4_l2_mse_ratio": float((l2_leaf_mse / max_leaf_mse).item()),
        }
    )

    max_summary_tensors = quantize_page_summaries_int8(page_sum_k, page_sum_v)
    (
        quantized_page_sum_k,
        quantized_page_sum_v,
        page_sum_k_scales,
        page_sum_v_scales,
    ) = quantize_page_summaries_int8(page_sum_k, page_sum_v, optimize_scale=True)
    for source, max_codes, max_scales, l2_codes, l2_scales in (
        (
            page_sum_k,
            max_summary_tensors[0],
            max_summary_tensors[2],
            quantized_page_sum_k,
            page_sum_k_scales,
        ),
        (
            page_sum_v,
            max_summary_tensors[1],
            max_summary_tensors[3],
            quantized_page_sum_v,
            page_sum_v_scales,
        ),
    ):
        max_reconstruction = (
            max_codes.float() * max_scales.repeat_interleave(32, dim=-1).float()
        )
        l2_reconstruction = (
            l2_codes.float() * l2_scales.repeat_interleave(32, dim=-1).float()
        )
        assert torch.mean((l2_reconstruction - source.float()) ** 2) <= torch.mean(
            (max_reconstruction - source.float()) ** 2
        )
    # Fixed-capacity serving pools deliberately reuse storage. A page with no
    # leaves must ignore whatever summary bytes its previous owner left behind.
    unused_pages = torch.arange(page_capacity, device=device).view(1, 1, -1) >= (
        next_page.unsqueeze(-1)
    )
    quantized_page_sum_k.masked_fill_(unused_pages.unsqueeze(-1), 127)
    quantized_page_sum_v.masked_fill_(unused_pages.unsqueeze(-1), -127)
    page_sum_k_scales.masked_fill_(unused_pages.unsqueeze(-1), 8)
    page_sum_v_scales.masked_fill_(unused_pages.unsqueeze(-1), 8)
    append_k = torch.randn(
        batch,
        kv_heads,
        decode_tokens,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    append_v = torch.randn_like(append_k)
    append_owners = torch.randint(
        4,
        (batch, kv_heads, decode_tokens),
        device=device,
        dtype=torch.long,
    )
    append_quantized_virtual_paged_kv(
        append_k,
        append_v,
        tokens,
        append_owners,
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
        l2_quantized_leaf_k,
        l2_quantized_leaf_v,
        l2_page_k_scales,
        l2_page_v_scales,
        l2_page_quantized_counts,
        hash_probes=0,
        quantized_page_sum_k=quantized_page_sum_k,
        quantized_page_sum_v=quantized_page_sum_v,
        page_sum_k_scales=page_sum_k_scales,
        page_sum_v_scales=page_sum_v_scales,
        optimize_summary_scale=True,
        optimize_leaf_scale=True,
    )
    torch.cuda.synchronize(device)
    torch.testing.assert_close(l2_page_quantized_counts, page_counts)
    for head in range(kv_heads):
        combined_owners = torch.cat((owners[0, head], append_owners[0, head]))
        expected_counts = combined_owners.bincount(minlength=slots)
        torch.testing.assert_close(slot_lengths[0, head].long(), expected_counts)
        for slot in range(4):
            count = int(expected_counts[slot].item())
            page_ids = slot_pages[0, head, slot, : math.ceil(count / page_size)].long()
            expected_key_sum = (
                torch.cat(
                    (
                        leaf_k[0, head][owners[0, head] == slot],
                        append_k[0, head][append_owners[0, head] == slot],
                    )
                )
                .float()
                .sum(0)
            )
            actual_key_sum = (
                quantized_page_sum_k[0, head].index_select(0, page_ids).float()
                * page_sum_k_scales[0, head]
                .index_select(0, page_ids)
                .repeat_interleave(32, dim=-1)
                .float()
            ).sum(0)
            torch.testing.assert_close(
                actual_key_sum,
                expected_key_sum,
                rtol=2e-2,
                atol=3.0e-1,
            )
            expected_value_sum = (
                torch.cat(
                    (
                        leaf_v[0, head][owners[0, head] == slot],
                        append_v[0, head][append_owners[0, head] == slot],
                    )
                )
                .float()
                .sum(0)
            )
            actual_value_sum = (
                quantized_page_sum_v[0, head].index_select(0, page_ids).float()
                * page_sum_v_scales[0, head]
                .index_select(0, page_ids)
                .repeat_interleave(32, dim=-1)
                .float()
            ).sum(0)
            torch.testing.assert_close(
                actual_value_sum,
                expected_value_sum,
                rtol=2e-2,
                atol=3.0e-1,
            )


def verify_direct_page_append(device: torch.device) -> None:
    """Exercise the compile-time zero-probe path used before inline overflow."""
    batch, kv_heads, slots, tokens, head_dim = 1, 2, 64, 256, 256
    page_size, inline_pages_per_slot, page_capacity = 16, 32, 256
    page_k = torch.zeros(
        batch,
        kv_heads,
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.zeros_like(page_k)
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
    slot_pages = torch.full(
        (batch, kv_heads, slots, inline_pages_per_slot),
        -1,
        device=device,
        dtype=torch.int32,
    )
    slot_lengths = torch.zeros(batch, kv_heads, slots, device=device, dtype=torch.int32)
    overflow_page_keys = torch.full(
        (batch, kv_heads, 16), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    next_page = torch.zeros(batch, kv_heads, device=device, dtype=torch.int32)
    owners = torch.randint(
        slots, (batch, kv_heads, tokens), device=device, dtype=torch.long
    )
    k = torch.randn(
        batch, kv_heads, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    token_id = torch.arange(tokens, device=device)
    k[..., 0] = (token_id % 128).to(k.dtype)
    k[..., 1] = (token_id // 128).to(k.dtype)
    v = torch.randn_like(k)
    append_paged_kv(
        k,
        v,
        owners,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        slot_lengths,
        next_page,
        hash_probes=0,
        page_sum_k=page_sum_k,
        page_sum_v=page_sum_v,
        page_counts=page_counts,
    )
    torch.cuda.synchronize(device)
    if int(overflow_flag.item()) != 0 or int(overflow_used.item()) != 0:
        raise AssertionError("direct page append unexpectedly used overflow storage")

    for head in range(kv_heads):
        expected_counts = owners[0, head].bincount(minlength=slots)
        if not torch.equal(slot_lengths[0, head].long(), expected_counts):
            raise AssertionError("direct page append produced incorrect slot lengths")
        expected_pages = int(((expected_counts + page_size - 1) // page_size).sum())
        actual_pages = int(next_page[0, head].item())
        if actual_pages != expected_pages:
            raise AssertionError(
                "direct page append produced incorrect page count: "
                f"head={head}, expected={expected_pages}, actual={actual_pages}"
            )
        for slot in range(slots):
            count = int(expected_counts[slot].item())
            pages = (count + page_size - 1) // page_size
            page_ids = slot_pages[0, head, slot, :pages].long()
            if bool((page_ids < 0).any().item()):
                raise AssertionError("direct page append left a missing page")
            actual_k = page_k[0, head].index_select(0, page_ids).flatten(0, 1)[:count]
            expected_k = k[0, head][owners[0, head] == slot]
            if not torch.equal(actual_k, expected_k):
                raise AssertionError(
                    "direct page append did not preserve chronological region order"
                )
            if (
                int(page_counts[0, head].index_select(0, page_ids).sum().item())
                != count
            ):
                raise AssertionError("direct page summaries have incorrect counts")
            actual_key_sum = page_sum_k[0, head].index_select(0, page_ids).sum(0)
            actual_value_sum = page_sum_v[0, head].index_select(0, page_ids).sum(0)
            expected_value = v[0, head][owners[0, head] == slot]
            torch.testing.assert_close(
                actual_key_sum.float(),
                expected_k.float().sum(0),
                rtol=2e-2,
                atol=2.5e-1,
            )
            torch.testing.assert_close(
                actual_value_sum.float(),
                expected_value.float().sum(0),
                rtol=2e-2,
                atol=2.5e-1,
            )


def verify_residual_page_attention(
    device: torch.device, kv_group_size: int = 4, head_dim: int = 256
) -> None:
    batch, kv_heads = 1, 1
    query_heads = kv_heads * kv_group_size
    slots, tokens, query_len = 4, 256, 8
    page_size, inline_pages_per_slot, page_capacity = 16, 16, 64
    page_k = torch.zeros(
        batch,
        kv_heads,
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.zeros_like(page_k)
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
    slot_pages = torch.full(
        (batch, kv_heads, slots, inline_pages_per_slot),
        -1,
        device=device,
        dtype=torch.int32,
    )
    slot_lengths = torch.zeros(batch, kv_heads, slots, device=device, dtype=torch.int32)
    overflow_page_keys = torch.full(
        (batch, kv_heads, 16), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    overflow_flag = torch.zeros((), device=device, dtype=torch.int32)
    next_page = torch.zeros(batch, kv_heads, device=device, dtype=torch.int32)
    # Keep slot zero within one page while the other slots span several pages.
    # This covers the no-residual fast path in the same mixed routing call.
    owners = torch.cat(
        (
            torch.zeros(8, device=device, dtype=torch.long),
            torch.arange(tokens - 8, device=device, dtype=torch.long)
            .remainder(slots - 1)
            .add(1),
        )
    ).view(1, 1, tokens)
    k = torch.randn(
        batch, kv_heads, tokens, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    append_paged_kv(
        k,
        v,
        owners,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        overflow_flag,
        slot_lengths,
        next_page,
        hash_probes=0,
        page_sum_k=page_sum_k,
        page_sum_v=page_sum_v,
        page_counts=page_counts,
    )
    torch.cuda.synchronize(device)

    state_k = torch.zeros(
        batch, kv_heads, slots, head_dim, device=device, dtype=torch.bfloat16
    )
    state_v = torch.zeros_like(state_k)
    state_counts = slot_lengths.unsqueeze(-1).float()
    for slot in range(slots):
        pages = (int(slot_lengths[0, 0, slot].item()) + page_size - 1) // page_size
        page_ids = slot_pages[0, 0, slot, :pages].long()
        state_k[0, 0, slot] = page_sum_k[0, 0].index_select(0, page_ids).sum(0)
        state_v[0, 0, slot] = page_sum_v[0, 0].index_select(0, page_ids).sum(0)

    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    top_slots = (
        torch.rand(batch, query_heads, query_len, slots, device=device)
        .topk(2, dim=-1, sorted=False)
        .indices
    )
    scale = head_dim**-0.5
    expected_out = torch.empty_like(q, dtype=torch.float32)
    expected_lse = torch.empty(
        batch,
        query_heads,
        query_len,
        device=device,
        dtype=torch.float32,
    )
    for head in range(query_heads):
        for query_index in range(query_len):
            query = q[0, head, query_index].float()
            scores = []
            values = []
            for slot_tensor in top_slots[0, head, query_index]:
                slot = int(slot_tensor.item())
                count = int(slot_lengths[0, 0, slot].item())
                pages = (count + page_size - 1) // page_size
                page_ids = slot_pages[0, 0, slot, :pages].long()
                counts = page_counts[0, 0].index_select(0, page_ids).float()
                sums = page_sum_k[0, 0].index_select(0, page_ids).float()
                page_scores = (sums / counts[:, None]) @ query * scale + counts.log()
                selected = int(page_ids[int(page_scores.argmax().item())].item())
                selected_count = int(page_counts[0, 0, selected].item())
                residual_count = count - selected_count
                if residual_count:
                    residual_key = (
                        state_k[0, 0, slot].float() - page_sum_k[0, 0, selected].float()
                    ) / residual_count
                    residual_value = (
                        state_v[0, 0, slot].float() - page_sum_v[0, 0, selected].float()
                    ) / residual_count
                    scores.append(
                        query.dot(residual_key) * scale + math.log(residual_count)
                    )
                    values.append(residual_value.unsqueeze(0))
                exact_keys = page_k[0, 0, selected, :selected_count].float()
                exact_values = page_v[0, 0, selected, :selected_count].float()
                scores.extend((exact_keys @ query * scale).unbind())
                values.append(exact_values)
            score_tensor = torch.stack(scores)
            value_tensor = torch.cat(values, dim=0)
            expected_out[0, head, query_index] = (
                score_tensor.softmax(0).unsqueeze(0) @ value_tensor
            ).squeeze(0)
            expected_lse[0, head, query_index] = score_tensor.logsumexp(0)

    for page_block_n in (1, 2, 4, 8, 16, 32):
        actual_out, actual_lse = query_major_residual_page_attention(
            q,
            state_k,
            state_v,
            state_counts,
            page_k,
            page_v,
            page_sum_k,
            page_sum_v,
            page_counts,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            kv_group_size=query_heads // kv_heads,
            scale=scale,
            hash_probes=0,
            page_block_n=page_block_n,
        )
        torch.testing.assert_close(
            actual_out.float(), expected_out, rtol=2e-2, atol=8e-3
        )
        torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-4, atol=2e-4)

    prefix_slots = top_slots.clone()
    prefix_slots[..., 1:] = -1
    prefix_out, prefix_lse = query_major_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        page_k,
        page_v,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        prefix_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=2,
    )
    one_route_out, one_route_lse = query_major_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        page_k,
        page_v,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots[..., :1],
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=2,
    )
    torch.testing.assert_close(prefix_out, one_route_out)
    torch.testing.assert_close(prefix_lse, one_route_lse)

    (
        quantized_page_sum_k,
        quantized_page_sum_v,
        page_sum_k_scales,
        page_sum_v_scales,
    ) = quantize_page_summaries_int8(page_sum_k, page_sum_v, optimize_scale=True)
    quantized_summary_out, quantized_summary_lse = query_major_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        page_k,
        page_v,
        page_sum_k[..., :1, :],
        page_sum_v[..., :1, :],
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=16,
        quantized_page_sum_k=quantized_page_sum_k,
        quantized_page_sum_v=quantized_page_sum_v,
        page_sum_k_scales=page_sum_k_scales,
        page_sum_v_scales=page_sum_v_scales,
    )
    torch.testing.assert_close(
        quantized_summary_out.float(), expected_out, rtol=3e-2, atol=1.5e-2
    )
    torch.testing.assert_close(
        quantized_summary_lse, expected_lse, rtol=2e-3, atol=3e-3
    )

    flat_tokens = page_capacity * page_size
    permutation = torch.randperm(flat_tokens, device=device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(flat_tokens, device=device)
    leaf_k = page_k.flatten(2, 3).index_select(2, permutation).contiguous()
    leaf_v = page_v.flatten(2, 3).index_select(2, permutation).contiguous()
    page_indices = inverse.to(torch.int32).view(1, 1, page_capacity, page_size)
    indexed_out, indexed_lse = query_major_indexed_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=2,
    )
    torch.testing.assert_close(indexed_out.float(), expected_out, rtol=2e-2, atol=8e-3)
    torch.testing.assert_close(indexed_lse, expected_lse, rtol=2e-4, atol=2e-4)
    decode_q = q[..., :1, :].contiguous()
    decode_slots = top_slots[..., :1, :].contiguous()
    materialized_scores = materialize_page_summary_scores_gqa(
        decode_q,
        page_sum_k,
        page_counts,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        page_block_n=16,
    )
    materialized_out, materialized_lse = (
        query_major_indexed_residual_page_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            leaf_k,
            leaf_v,
            page_indices,
            page_sum_k,
            page_sum_v,
            page_counts,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            decode_slots,
            kv_group_size=query_heads // kv_heads,
            scale=scale,
            hash_probes=0,
            page_block_n=16,
            materialized_page_scores=materialized_scores,
        )
    )
    torch.testing.assert_close(
        materialized_out.float(), indexed_out[..., :1, :].float(), rtol=2e-2, atol=8e-3
    )
    torch.testing.assert_close(
        materialized_lse, indexed_lse[..., :1], rtol=2e-4, atol=2e-4
    )
    indexed_quantized_out, indexed_quantized_lse = (
        query_major_indexed_residual_page_attention(
            q,
            state_k,
            state_v,
            state_counts,
            leaf_k,
            leaf_v,
            page_indices,
            page_sum_k[..., :1, :],
            page_sum_v[..., :1, :],
            page_counts,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            kv_group_size=query_heads // kv_heads,
            scale=scale,
            hash_probes=0,
            page_block_n=16,
            quantized_page_sum_k=quantized_page_sum_k,
            quantized_page_sum_v=quantized_page_sum_v,
            page_sum_k_scales=page_sum_k_scales,
            page_sum_v_scales=page_sum_v_scales,
        )
    )
    torch.testing.assert_close(
        indexed_quantized_out.float(), expected_out, rtol=3e-2, atol=1.5e-2
    )
    torch.testing.assert_close(
        indexed_quantized_lse, expected_lse, rtol=2e-3, atol=3e-3
    )

    quantized_leaf_k = torch.empty(
        batch,
        kv_heads,
        flat_tokens,
        head_dim // 2,
        device=device,
        dtype=torch.uint8,
    )
    quantized_leaf_v = torch.empty_like(quantized_leaf_k)
    page_k_scales = torch.empty(
        batch,
        kv_heads,
        page_capacity,
        head_dim // 32,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v_scales = torch.empty_like(page_k_scales)
    page_quantized_counts = torch.zeros_like(page_counts)
    quantize_virtual_paged_kv_int4(
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        quantized_leaf_k,
        quantized_leaf_v,
        page_k_scales,
        page_v_scales,
        page_quantized_counts,
    )
    int4_out, int4_lse = query_major_indexed_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        leaf_k[..., :1, :],
        leaf_v[..., :1, :],
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=2,
        quantized_leaf_k=quantized_leaf_k,
        quantized_leaf_v=quantized_leaf_v,
        page_k_scales=page_k_scales,
        page_v_scales=page_v_scales,
        page_quantized_counts=page_quantized_counts,
    )
    int4_mse = (int4_out.float() - expected_out).square().mean()
    print(
        {
            "virtual_int4_attention_mse": float(int4_mse.item()),
        }
    )
    token_group_size = 1
    token_group_count = page_size // token_group_size
    token_group_leaf_k = torch.empty_like(quantized_leaf_k)
    token_group_leaf_v = torch.empty_like(quantized_leaf_v)
    token_group_k_scales = torch.empty(
        batch,
        kv_heads,
        page_capacity,
        token_group_count * (head_dim // 32),
        device=device,
        dtype=torch.bfloat16,
    )
    token_group_v_scales = torch.empty_like(token_group_k_scales)
    token_group_counts = torch.zeros_like(page_counts)
    quantize_virtual_paged_kv_int4(
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        token_group_leaf_k,
        token_group_leaf_v,
        token_group_k_scales,
        token_group_v_scales,
        token_group_counts,
        quant_token_group_size=token_group_size,
        optimize_scale=True,
    )
    token_group_out, token_group_lse = query_major_indexed_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        leaf_k[..., :1, :],
        leaf_v[..., :1, :],
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=2,
        quantized_leaf_k=token_group_leaf_k,
        quantized_leaf_v=token_group_leaf_v,
        page_k_scales=token_group_k_scales,
        page_v_scales=token_group_v_scales,
        page_quantized_counts=token_group_counts,
        quant_token_group_size=token_group_size,
    )
    token_group_mse = (token_group_out.float() - expected_out).square().mean()
    if token_group_mse >= int4_mse:
        raise AssertionError("token-group INT4 did not reduce attention output error")
    print(
        {
            "virtual_int4_token1_attention_mse": float(token_group_mse.item()),
            "virtual_int4_token1_attention_mse_ratio": float(
                (token_group_mse / int4_mse).item()
            ),
        }
    )
    torch.testing.assert_close(token_group_lse, expected_lse, rtol=2e-2, atol=3e-2)
    # Packed INT4 has a small number of legitimate high-relative-error output
    # elements around zero; guard aggregate accuracy above and retain a loose
    # absolute outlier bound here.  The former 0.12 bound failed the unchanged
    # page-wide implementation (7 / 8192 elements, max 0.236).
    torch.testing.assert_close(int4_out.float(), expected_out, rtol=2e-1, atol=2.5e-1)
    torch.testing.assert_close(int4_lse, expected_lse, rtol=2e-2, atol=3e-2)

    int8_leaf_k = torch.empty(
        batch,
        kv_heads,
        flat_tokens,
        head_dim,
        device=device,
        dtype=torch.int8,
    )
    int8_leaf_v = torch.empty_like(int8_leaf_k)
    int8_page_k_scales = torch.empty_like(page_k_scales)
    int8_page_v_scales = torch.empty_like(page_v_scales)
    int8_page_quantized_counts = torch.zeros_like(page_counts)
    quantize_virtual_paged_kv(
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        int8_leaf_k,
        int8_leaf_v,
        int8_page_k_scales,
        int8_page_v_scales,
        int8_page_quantized_counts,
        quant_bits=8,
    )
    int8_out, int8_lse = query_major_indexed_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        leaf_k[..., :1, :],
        leaf_v[..., :1, :],
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=2,
        quantized_leaf_k=int8_leaf_k,
        quantized_leaf_v=int8_leaf_v,
        page_k_scales=int8_page_k_scales,
        page_v_scales=int8_page_v_scales,
        page_quantized_counts=int8_page_quantized_counts,
        quant_bits=8,
    )
    torch.testing.assert_close(int8_out.float(), expected_out, rtol=3e-2, atol=2e-2)
    torch.testing.assert_close(int8_lse, expected_lse, rtol=3e-3, atol=4e-3)

    invalid_slots = torch.full_like(top_slots, -1)
    invalid_out, invalid_lse = query_major_indexed_residual_page_attention(
        q,
        state_k,
        state_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        invalid_slots,
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=16,
    )
    if bool(invalid_out.ne(0).any().item()):
        raise AssertionError("invalid recursive routes produced nonzero output")
    if bool((invalid_lse != -torch.inf).any().item()):
        raise AssertionError("invalid recursive routes produced finite mass")

    parallel_out, parallel_lse = query_major_indexed_residual_page_attention(
        q[..., :1, :].contiguous(),
        state_k,
        state_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices,
        page_sum_k,
        page_sum_v,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        top_slots[..., :1, :].contiguous(),
        kv_group_size=query_heads // kv_heads,
        scale=scale,
        hash_probes=0,
        page_block_n=16,
        route_parallel=True,
    )
    for route in range(int(top_slots.size(-1))):
        route_out, route_lse = query_major_indexed_residual_page_attention(
            q[..., :1, :].contiguous(),
            state_k,
            state_v,
            state_counts,
            leaf_k,
            leaf_v,
            page_indices,
            page_sum_k,
            page_sum_v,
            page_counts,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots[..., :1, route : route + 1].contiguous(),
            kv_group_size=query_heads // kv_heads,
            scale=scale,
            hash_probes=0,
            page_block_n=16,
        )
        torch.testing.assert_close(
            parallel_out[..., route : route + 1, :].float(),
            route_out.float(),
            rtol=2e-2,
            atol=8e-3,
        )
        torch.testing.assert_close(
            parallel_lse[..., route : route + 1],
            route_lse,
            rtol=2e-4,
            atol=2e-4,
        )
    parallel_weights = parallel_lse.softmax(dim=-1).to(parallel_out.dtype)
    parallel_merged = (parallel_out * parallel_weights.unsqueeze(-1)).sum(
        dim=-2, keepdim=True
    )
    torch.testing.assert_close(
        parallel_merged.float(),
        indexed_out[..., :1, :].float(),
        rtol=2e-2,
        atol=8e-3,
    )
    torch.testing.assert_close(
        parallel_lse.logsumexp(dim=-1, keepdim=True),
        indexed_lse[..., :1],
        rtol=2e-4,
        atol=2e-4,
    )


def compare_large_gqa_route(device: torch.device) -> dict[str, float]:
    batch, query_heads, kv_heads, state_len, head_dim = 1, 8, 2, 2048, 256
    group_size = query_heads // kv_heads
    q = torch.randn(
        batch, query_heads, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    state_k = torch.randn(
        batch, kv_heads, state_len, head_dim, device=device, dtype=torch.bfloat16
    )
    state_v = torch.randn_like(state_k)
    counts = torch.randint(
        1, 64, (batch, kv_heads, state_len, 1), device=device
    ).float()
    local_k = torch.randn(
        batch, kv_heads, 512, head_dim, device=device, dtype=torch.bfloat16
    )
    local_v = torch.randn_like(local_k)
    page_k = torch.zeros(
        batch, kv_heads, 1, 16, head_dim, device=device, dtype=torch.bfloat16
    )
    page_v = torch.zeros_like(page_k)
    slot_pages = torch.full(
        (batch, kv_heads, state_len, 1), -1, device=device, dtype=torch.int32
    )
    slot_lengths = torch.zeros(
        batch, kv_heads, state_len, device=device, dtype=torch.int32
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 16), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    scalar_buffers = new_fused_decode_buffers(
        q, splits=8, state_capacity=state_len, route_group_size=64
    )
    gqa_buffers = new_fused_decode_buffers(
        q,
        splits=8,
        state_capacity=state_len,
        route_group_size=64,
        gqa_route_splits=4,
    )
    if any(name.startswith("gqa_") for name in scalar_buffers):
        raise AssertionError("generic decode allocated cooperative GQA scratch")
    if "gqa_route_partial_out" not in gqa_buffers:
        raise AssertionError("cooperative decode did not allocate GQA scratch")
    common = dict(
        state_len=state_len,
        kv_group_size=group_size,
        scale=head_dim**-0.5,
        split_kv=8,
        fuse_state_route=True,
        route_group_size=64,
    )
    scalar_output = fused_decode_paged_lod_attention(
        q,
        state_k,
        state_v,
        counts,
        local_k,
        local_v,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        None,
        buffers=scalar_buffers,
        route_use_dot=True,
        **common,
    )
    gqa_output = fused_decode_paged_lod_attention(
        q,
        state_k,
        state_v,
        counts,
        local_k,
        local_v,
        page_k,
        page_v,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        None,
        buffers=gqa_buffers,
        route_gqa_grouped=True,
        **common,
    )
    torch.cuda.synchronize(device)
    scalar_slots = scalar_buffers["route_top_slots"].sort(dim=-1).values
    gqa_slots = gqa_buffers["route_top_slots"].sort(dim=-1).values
    return {
        "top8_set_fraction": float(
            (scalar_slots == gqa_slots).all(dim=-1).float().mean().item()
        ),
        "top_score_max_abs": float(
            (
                scalar_buffers["route_top_scores"].sort(dim=-1).values
                - gqa_buffers["route_top_scores"].sort(dim=-1).values
            )
            .abs()
            .max()
            .item()
        ),
        "output_max_abs": float(
            (scalar_output.float() - gqa_output.float()).abs().max().item()
        ),
    }


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    if os.environ.get("VERIFY_RESIDUAL_PAGE_ONLY") == "1":
        verify_residual_page_attention(device)
        return
    cooperative_hip = os.environ.get("VERIFY_GQA_COOPERATIVE_HIP") == "1"
    cooperative_route_splits = int(
        os.environ.get("VERIFY_GQA_COOPERATIVE_ROUTE_SPLITS", "4")
    )
    if os.environ.get("VERIFY_GQA_ONLY") != "1":
        verify_fused_sink_branch_merge(device)
        verify_dense_page_summary_attention(device)
        verify_direct_page_append(device)
        verify_virtual_page_append(device)
        verify_residual_page_attention(device)
        verify_page_append(device)
        verify_two_level_page_directory(device)
        verify_sealed_page_append(device)
        verify_int8_mma_page_attention(device)
    batch, query_heads, kv_heads = 1, 16, 4
    query_len, slots, routes, head_dim = 256, 64, 8, 256
    page_size, pages_per_slot = 16, 4
    group_size = query_heads // kv_heads

    lengths = torch.randint(
        1,
        pages_per_slot * page_size + 1,
        (slots,),
        device=device,
        dtype=torch.long,
    )
    slot_lengths = (
        lengths.view(1, 1, slots).expand(batch, kv_heads, slots).clone().to(torch.int32)
    )
    slot_pages = (
        torch.arange(slots * pages_per_slot, device=device, dtype=torch.int32)
        .view(1, 1, slots, pages_per_slot)
        .expand(batch, kv_heads, slots, pages_per_slot)
        .clone()
    )
    overflow_page_keys = torch.full(
        (batch, kv_heads, 16), -1, device=device, dtype=torch.int32
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
    page_k_scales = (
        page_k.float().abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    ).to(torch.bfloat16)
    page_v_scales = (
        page_v.float().abs().amax(dim=-1).clamp_min(1.0e-8) / 127.0
    ).to(torch.bfloat16)
    int8_page_k = (
        page_k.float() / page_k_scales.float().unsqueeze(-1)
    ).round().clamp(-127, 127).to(torch.int8)
    int8_page_v = (
        page_v.float() / page_v_scales.float().unsqueeze(-1)
    ).round().clamp(-127, 127).to(torch.int8)
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    top_slots = (
        torch.rand(batch, query_heads, query_len, slots, device=device)
        .topk(routes, dim=-1, sorted=False)
        .indices
    )

    exact_k_parts = []
    exact_v_parts = []
    owner_parts = []
    for slot in range(slots):
        length = int(lengths[slot].item())
        exact_k_parts.append(
            page_k[:, :, slot * pages_per_slot : (slot + 1) * pages_per_slot].flatten(
                2, 3
            )[:, :, :length]
        )
        exact_v_parts.append(
            page_v[:, :, slot * pages_per_slot : (slot + 1) * pages_per_slot].flatten(
                2, 3
            )[:, :, :length]
        )
        owner_parts.append(
            torch.full((batch, kv_heads, length), slot, device=device, dtype=torch.long)
        )
    exact_k = torch.cat(exact_k_parts, dim=2)
    exact_v = torch.cat(exact_v_parts, dim=2)
    owners = torch.cat(owner_parts, dim=2)
    state_counts = slot_lengths.unsqueeze(-1).float()
    scale = head_dim**-0.5
    dynamic_open_counts = torch.arange(query_len, device=device).remainder(routes) + 1
    dynamic_top_slots = torch.where(
        torch.arange(routes, device=device).view(1, 1, 1, routes)
        < dynamic_open_counts.view(1, 1, query_len, 1),
        top_slots,
        torch.full_like(top_slots, -1),
    )

    with torch.inference_mode():
        expected_out, expected_lse = _expert_leaf_attention(
            q,
            exact_k,
            exact_v,
            owners,
            state_counts,
            top_slots,
            kv_group_size=group_size,
            head_temperature=torch.ones(query_heads, device=device, dtype=q.dtype),
            scale=scale,
        )
        actual_out, actual_lse = paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            kv_group_size=group_size,
            scale=scale,
            block_n=16,
            num_warps=2,
        )
        query_out, query_lse = query_major_paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            kv_group_size=group_size,
            scale=scale,
            hash_probes=0,
        )
        repeated_exact_k = exact_k.repeat_interleave(group_size, dim=1)
        repeated_exact_v = exact_v.repeat_interleave(group_size, dim=1)
        repeated_owners = owners.repeat_interleave(group_size, dim=1)
        dynamic_selected = (
            repeated_owners.unsqueeze(2).unsqueeze(-1) == dynamic_top_slots.unsqueeze(3)
        ).any(dim=-1)
        dynamic_scores = (
            torch.matmul(q.float(), repeated_exact_k.float().transpose(-1, -2)) * scale
        )
        dynamic_scores.masked_fill_(~dynamic_selected, float("-inf"))
        expected_dynamic_lse = torch.logsumexp(dynamic_scores, dim=-1)
        expected_dynamic_out = torch.matmul(
            dynamic_scores.softmax(dim=-1), repeated_exact_v.float()
        )
        dynamic_query_out, dynamic_query_lse = query_major_paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            dynamic_top_slots,
            kv_group_size=group_size,
            scale=scale,
            hash_probes=0,
        )
        dynamic_compact_out, dynamic_compact_lse = paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            dynamic_top_slots,
            kv_group_size=group_size,
            scale=scale,
            block_n=16,
            num_warps=2,
            compact_invalid_routes=True,
        )

        decode_q = q[..., :1, :].contiguous()
        decode_top_slots = top_slots[..., :1, :].contiguous()
        state_k = torch.stack([part.sum(dim=2) for part in exact_k_parts], dim=2)
        state_v = torch.stack([part.sum(dim=2) for part in exact_v_parts], dim=2)
        local_k = torch.randn(
            batch,
            kv_heads,
            257,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        local_v = torch.randn_like(local_k)
        query_counts = state_counts.repeat_interleave(group_size, dim=1)
        mean_k = (state_k / state_counts.to(state_k.dtype)).repeat_interleave(
            group_size, dim=1
        )
        mean_v = (state_v / state_counts.to(state_v.dtype)).repeat_interleave(
            group_size, dim=1
        )
        repeated_local_k = local_k.repeat_interleave(group_size, dim=1)
        repeated_local_v = local_v.repeat_interleave(group_size, dim=1)
        coarse_scores = torch.matmul(decode_q, mean_k.transpose(-1, -2)) * scale
        coarse_scores += query_counts.squeeze(-1).log().unsqueeze(2)
        coarse_scores.scatter_(-1, decode_top_slots, float("-inf"))
        local_scores = (
            torch.matmul(decode_q, repeated_local_k.transpose(-1, -2)) * scale
        )
        combined_scores = torch.cat((coarse_scores, local_scores), dim=-1)
        combined_values = torch.cat((mean_v, repeated_local_v), dim=2)
        coarse_weight = torch.softmax(combined_scores.float(), dim=-1).to(
            decode_q.dtype
        )
        coarse_out = torch.matmul(coarse_weight, combined_values)
        coarse_lse = torch.logsumexp(combined_scores.float(), dim=-1)
        decode_exact_out, decode_exact_lse = _expert_leaf_attention(
            decode_q,
            exact_k,
            exact_v,
            owners,
            state_counts,
            decode_top_slots,
            kv_group_size=group_size,
            head_temperature=torch.ones(query_heads, device=device, dtype=q.dtype),
            scale=scale,
        )
        expected_decode = _merge_lse_branches(
            coarse_out,
            coarse_lse,
            decode_exact_out,
            decode_exact_lse,
        )
        fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            decode_top_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
        )
        split_fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            decode_top_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )
        dynamic_decode_top_slots = decode_top_slots.clone()
        dynamic_decode_top_slots[..., 2:] = -1
        dynamic_split_fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            dynamic_decode_top_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )
        two_route_split_fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            decode_top_slots[..., :2].contiguous(),
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )
        dot_split_fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            decode_top_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            use_dot=True,
        )
        routed_scores = (
            decode_q.float().unsqueeze(-2) * mean_k.float().unsqueeze(2)
        ).sum(dim=-1) * scale
        routed_scores += query_counts.squeeze(-1).log().unsqueeze(2)
        fused_route_top = routed_scores.topk(routes, dim=-1, sorted=False).indices
        routed_coarse_scores = routed_scores.clone()
        routed_coarse_scores.scatter_(-1, fused_route_top, float("-inf"))
        routed_combined_scores = torch.cat(
            (routed_coarse_scores, local_scores.float()), dim=-1
        )
        routed_combined_values = torch.cat((mean_v, repeated_local_v), dim=2)
        routed_weight = torch.softmax(routed_combined_scores, dim=-1).to(decode_q.dtype)
        routed_coarse_out = torch.matmul(routed_weight, routed_combined_values)
        routed_coarse_lse = torch.logsumexp(routed_combined_scores, dim=-1)
        routed_exact_out, routed_exact_lse = _expert_leaf_attention(
            decode_q,
            exact_k,
            exact_v,
            owners,
            state_counts,
            fused_route_top,
            kv_group_size=group_size,
            head_temperature=torch.ones(query_heads, device=device, dtype=q.dtype),
            scale=scale,
        )
        expected_fused_route = _merge_lse_branches(
            routed_coarse_out,
            routed_coarse_lse,
            routed_exact_out,
            routed_exact_lse,
        )
        fused_route_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
        )
        sink_k = torch.randn_like(state_k[..., :1, :])
        sink_v = torch.randn_like(state_v[..., :1, :])
        repeated_sink_k = sink_k.repeat_interleave(group_size, dim=1)
        repeated_sink_v = sink_v.repeat_interleave(group_size, dim=1)
        sink_scores = (
            torch.matmul(decode_q.float(), repeated_sink_k.float().transpose(-1, -2))
            * scale
        )
        sink_lse = torch.logsumexp(sink_scores, dim=-1)
        sink_out = torch.matmul(
            torch.softmax(sink_scores, dim=-1).to(decode_q.dtype), repeated_sink_v
        )
        expected_separate_sink = _merge_lse_branches(
            expected_fused_route,
            torch.logaddexp(routed_coarse_lse, routed_exact_lse),
            sink_out,
            sink_lse,
        )
        separate_sink_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            sink_k=sink_k,
            sink_v=sink_v,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
        )
        fused_route_group8_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
            route_group_size=8,
        )
        fused_route_gqa_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
            route_group_size=32,
            route_gqa_grouped=True,
            hash_probes=0 if cooperative_hip else 8,
            gqa_cooperative_hip=cooperative_hip,
            gqa_cooperative_route_splits=cooperative_route_splits,
            gqa_cooperative_adaptive_splits=os.environ.get(
                "VERIFY_GQA_COOPERATIVE_ADAPTIVE_SPLITS", "0"
            )
            == "1",
        )
        fused_route_gqa_combined_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
            route_group_size=32,
            route_gqa_grouped=True,
            hash_probes=0 if cooperative_hip else 8,
            gqa_cooperative_hip=cooperative_hip,
            gqa_cooperative_route_splits=cooperative_route_splits,
            gqa_cooperative_fused_reduce=True,
        )
        int8_route_generic_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            int8_page_k,
            int8_page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
            route_group_size=32,
            route_gqa_grouped=True,
            hash_probes=0,
            gqa_cooperative_hip=False,
            flat_page_k_scales=page_k_scales,
            flat_page_v_scales=page_v_scales,
        )
        int8_route_gqa_hip_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            int8_page_k,
            int8_page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            fuse_state_route=True,
            route_group_size=32,
            route_gqa_grouped=True,
            hash_probes=0,
            gqa_cooperative_hip=cooperative_hip,
            gqa_cooperative_route_splits=cooperative_route_splits,
            flat_page_k_scales=page_k_scales,
            flat_page_v_scales=page_v_scales,
        )
        dynamic_route_buffers = new_fused_decode_buffers(
            decode_q,
            splits=8,
            state_capacity=slots,
            route_group_size=32,
        )
        dynamic_fused_route_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            buffers=dynamic_route_buffers,
            fuse_state_route=True,
            route_group_size=32,
            route_top_p=0.5,
        )
        dynamic_fused_slots = dynamic_route_buffers["route_top_slots"].clone()
        dynamic_explicit_route_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            dynamic_fused_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )
        residual_route_buffers = new_fused_decode_buffers(
            decode_q,
            splits=8,
            state_capacity=slots,
            route_group_size=32,
        )
        residual_fused_route_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            buffers=residual_route_buffers,
            fuse_state_route=True,
            route_group_size=32,
            route_gqa_grouped=True,
            route_residual_mass=0.05,
            reuse_residual_local_attention=True,
        )
        residual_fused_slots = residual_route_buffers["route_top_slots"].clone()
        residual_scores, residual_slots = routed_scores.topk(
            routes, dim=-1, sorted=True
        )
        residual_full_lse = torch.logaddexp(
            torch.logsumexp(routed_scores, dim=-1),
            torch.logsumexp(local_scores.float(), dim=-1),
        )
        residual_global_mass = torch.exp(
            residual_scores - residual_full_lse.unsqueeze(-1)
        )
        residual_cumulative_before = (
            residual_global_mass.cumsum(dim=-1) - residual_global_mass
        )
        residual_remaining_before = (
            residual_global_mass.sum(dim=-1, keepdim=True) - residual_cumulative_before
        )
        residual_keep = (torch.arange(routes, device=device) == 0) | (
            residual_remaining_before > 0.05
        )
        expected_residual_slots = torch.where(
            residual_keep, residual_slots, torch.full_like(residual_slots, -1)
        )
        residual_explicit_route_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            residual_fused_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )
        state_bound_buffers = new_fused_decode_buffers(
            decode_q,
            splits=8,
            state_capacity=slots,
            route_group_size=32,
        )
        state_bound_fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            buffers=state_bound_buffers,
            fuse_state_route=True,
            route_group_size=32,
            route_residual_mass=0.05,
            route_residual_use_state_bound=True,
        )
        state_bound_slots = state_bound_buffers["route_top_slots"].clone()
        state_bound_explicit_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            state_bound_slots,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )
        atomic_final_buffers = new_fused_decode_buffers(
            decode_q,
            splits=8,
            state_capacity=slots,
            route_group_size=32,
        )
        atomic_final_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            buffers=atomic_final_buffers,
            fuse_state_route=True,
            fuse_final_reduce=True,
            route_group_size=32,
        )
        scalar_dot_buffers = new_fused_decode_buffers(
            decode_q,
            splits=8,
            state_capacity=slots,
            route_group_size=32,
        )
        gqa_buffers = new_fused_decode_buffers(
            decode_q,
            splits=8,
            state_capacity=slots,
            route_group_size=32,
            gqa_route_splits=4,
        )
        scalar_dot_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            buffers=scalar_dot_buffers,
            fuse_state_route=True,
            route_group_size=32,
            route_use_dot=True,
        )
        gqa_compare_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            local_k,
            local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            None,
            state_len=slots,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
            buffers=gqa_buffers,
            fuse_state_route=True,
            route_group_size=32,
            route_gqa_grouped=True,
        )
        new_k = torch.randn(
            batch, kv_heads, 1, head_dim, device=device, dtype=torch.bfloat16
        )
        new_v = torch.randn_like(new_k)
        repeated_new_k = new_k.repeat_interleave(group_size, dim=1)
        repeated_new_v = new_v.repeat_interleave(group_size, dim=1)
        new_scores = torch.matmul(decode_q, repeated_new_k.transpose(-1, -2)) * scale
        appended_scores = torch.cat((coarse_scores, local_scores, new_scores), dim=-1)
        appended_values = torch.cat((mean_v, repeated_local_v, repeated_new_v), dim=2)
        appended_weight = torch.softmax(appended_scores.float(), dim=-1).to(
            decode_q.dtype
        )
        appended_coarse_out = torch.matmul(appended_weight, appended_values)
        appended_coarse_lse = torch.logsumexp(appended_scores.float(), dim=-1)
        expected_appended_decode = _merge_lse_branches(
            appended_coarse_out,
            appended_coarse_lse,
            decode_exact_out,
            decode_exact_lse,
        )
        buffered_local_k = torch.empty(
            batch,
            kv_heads,
            int(local_k.size(2)) + 1,
            head_dim,
            device=device,
            dtype=local_k.dtype,
        )
        buffered_local_v = torch.empty_like(buffered_local_k)
        buffered_local_k[..., :-1, :].copy_(local_k)
        buffered_local_v[..., :-1, :].copy_(local_v)
        appended_fused_decode = fused_decode_paged_lod_attention(
            decode_q,
            state_k,
            state_v,
            state_counts,
            buffered_local_k,
            buffered_local_v,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            decode_top_slots,
            state_len=slots,
            local_len=int(local_k.size(2)),
            new_k=new_k,
            new_v=new_v,
            kv_group_size=group_size,
            scale=scale,
            split_kv=8,
        )

    output_error = (actual_out.float() - expected_out.float()).abs()
    lse_error = (actual_lse.float() - expected_lse.float()).abs()
    result = {
        "output_max_abs": float(output_error.max().item()),
        "output_mean_abs": float(output_error.mean().item()),
        "lse_max_abs": float(lse_error.max().item()),
        "lse_mean_abs": float(lse_error.mean().item()),
        "actual_finite": bool(torch.isfinite(actual_out).all().item()),
        "actual_lse_finite": bool(torch.isfinite(actual_lse).all().item()),
        "query_output_max_abs": float(
            (query_out.float() - expected_out.float()).abs().max().item()
        ),
        "query_output_mean_abs": float(
            (query_out.float() - expected_out.float()).abs().mean().item()
        ),
        "query_lse_max_abs": float(
            (query_lse.float() - expected_lse.float()).abs().max().item()
        ),
        "dynamic_query_output_max_abs": float(
            (dynamic_query_out.float() - expected_dynamic_out).abs().max().item()
        ),
        "dynamic_query_lse_max_abs": float(
            (dynamic_query_lse.float() - expected_dynamic_lse).abs().max().item()
        ),
        "dynamic_compact_output_max_abs": float(
            (dynamic_compact_out.float() - expected_dynamic_out).abs().max().item()
        ),
        "dynamic_compact_lse_max_abs": float(
            (dynamic_compact_lse.float() - expected_dynamic_lse).abs().max().item()
        ),
        "fused_decode_max_abs": float(
            (fused_decode.float() - expected_decode.float()).abs().max().item()
        ),
        "fused_decode_mean_abs": float(
            (fused_decode.float() - expected_decode.float()).abs().mean().item()
        ),
        "split_fused_decode_max_abs": float(
            (split_fused_decode.float() - expected_decode.float()).abs().max().item()
        ),
        "split_fused_decode_mean_abs": float(
            (split_fused_decode.float() - expected_decode.float()).abs().mean().item()
        ),
        "dynamic_split_decode_max_abs": float(
            (dynamic_split_fused_decode.float() - two_route_split_fused_decode.float())
            .abs()
            .max()
            .item()
        ),
        "dot_split_fused_decode_max_abs": float(
            (dot_split_fused_decode.float() - expected_decode.float())
            .abs()
            .max()
            .item()
        ),
        "appended_fused_decode_max_abs": float(
            (appended_fused_decode.float() - expected_appended_decode.float())
            .abs()
            .max()
            .item()
        ),
        "appended_k_max_abs": float(
            (buffered_local_k[..., -1:, :].float() - new_k.float()).abs().max().item()
        ),
        "appended_v_max_abs": float(
            (buffered_local_v[..., -1:, :].float() - new_v.float()).abs().max().item()
        ),
        "fused_route_decode_max_abs": float(
            (fused_route_decode.float() - expected_fused_route.float())
            .abs()
            .max()
            .item()
        ),
        "separate_sink_decode_max_abs": float(
            (separate_sink_decode.float() - expected_separate_sink.float())
            .abs()
            .max()
            .item()
        ),
        "fused_route_group8_decode_max_abs": float(
            (fused_route_group8_decode.float() - expected_fused_route.float())
            .abs()
            .max()
            .item()
        ),
        "fused_route_gqa_decode_max_abs": float(
            (fused_route_gqa_decode.float() - expected_fused_route.float())
            .abs()
            .max()
            .item()
        ),
        "dynamic_fused_route_decode_max_abs": float(
            (dynamic_fused_route_decode.float() - dynamic_explicit_route_decode.float())
            .abs()
            .max()
            .item()
        ),
        "dynamic_fused_route_mean_opened": float(
            (dynamic_fused_slots >= 0).sum(dim=-1).float().mean().item()
        ),
        "residual_fused_route_decode_max_abs": float(
            (
                residual_fused_route_decode.float()
                - residual_explicit_route_decode.float()
            )
            .abs()
            .max()
            .item()
        ),
        "residual_fused_route_mean_opened": float(
            (residual_fused_slots >= 0).sum(dim=-1).float().mean().item()
        ),
        "residual_fused_route_slots_match": bool(
            (residual_fused_slots == expected_residual_slots).all().item()
        ),
        "state_bound_fused_decode_max_abs": float(
            (state_bound_fused_decode.float() - state_bound_explicit_decode.float())
            .abs()
            .max()
            .item()
        ),
        "state_bound_mean_opened": float(
            (state_bound_slots >= 0).sum(dim=-1).float().mean().item()
        ),
        "atomic_final_decode_max_abs": float(
            (atomic_final_decode.float() - fused_route_decode.float())
            .abs()
            .max()
            .item()
        ),
        "gqa_vs_scalar_dot_decode_max_abs": float(
            (gqa_compare_decode.float() - scalar_dot_decode.float()).abs().max().item()
        ),
        "gqa_combined_reduce_max_abs": float(
            (
                fused_route_gqa_combined_decode.float()
                - fused_route_gqa_decode.float()
            )
            .abs()
            .max()
            .item()
        ),
        "int8_gqa_hip_vs_generic_max_abs": float(
            (
                int8_route_gqa_hip_decode.float()
                - int8_route_generic_decode.float()
            )
            .abs()
            .max()
            .item()
        ),
        "gqa_vs_scalar_dot_top8_set_fraction": float(
            (
                gqa_buffers["route_top_slots"].sort(dim=-1).values
                == scalar_dot_buffers["route_top_slots"].sort(dim=-1).values
            )
            .all(dim=-1)
            .float()
            .mean()
            .item()
        ),
        "gqa_vs_scalar_dot_top_score_max_abs": float(
            (
                gqa_buffers["route_top_scores"].sort(dim=-1).values
                - scalar_dot_buffers["route_top_scores"].sort(dim=-1).values
            )
            .abs()
            .max()
            .item()
        ),
        "large_gqa_route": compare_large_gqa_route(device),
    }
    print(result)
    if not result["actual_finite"] or not result["actual_lse_finite"]:
        raise AssertionError("Triton paged attention returned a non-finite result")
    if result["output_max_abs"] > 0.03 or result["lse_max_abs"] > 0.03:
        raise AssertionError("Triton paged attention disagrees with packed attention")
    if result["query_output_max_abs"] > 0.03 or result["query_lse_max_abs"] > 0.03:
        raise AssertionError("query-major attention disagrees with packed attention")
    if (
        result["dynamic_query_output_max_abs"] > 0.03
        or result["dynamic_query_lse_max_abs"] > 0.03
    ):
        raise AssertionError("masked query-major attention disagrees with reference")
    if (
        result["dynamic_compact_output_max_abs"] > 0.03
        or result["dynamic_compact_lse_max_abs"] > 0.03
    ):
        raise AssertionError("compacted expert attention disagrees with reference")
    if result["fused_decode_max_abs"] > 0.03:
        raise AssertionError("fused decode attention disagrees with two branches")
    if result["split_fused_decode_max_abs"] > 0.03:
        raise AssertionError("split fused decode disagrees with two branches")
    if result["dynamic_split_decode_max_abs"] > 0.03:
        raise AssertionError("masked split decode disagrees with a shorter route list")
    if result["dot_split_fused_decode_max_abs"] > 0.03:
        raise AssertionError("dot split fused decode disagrees with two branches")
    if result["appended_fused_decode_max_abs"] > 0.03:
        raise AssertionError("appending fused decode disagrees with two branches")
    if result["appended_k_max_abs"] or result["appended_v_max_abs"]:
        raise AssertionError("fused decode did not append the current KV exactly")
    if result["fused_route_decode_max_abs"] > 0.03:
        raise AssertionError("fused route/coarse decode disagrees with reference")
    if result["separate_sink_decode_max_abs"] > 0.03:
        raise AssertionError("fused separate sink disagrees with exact LSE merge")
    if result["fused_route_group8_decode_max_abs"] > 0.03:
        raise AssertionError("group-8 fused route/coarse disagrees with reference")
    if result["fused_route_gqa_decode_max_abs"] > 0.03:
        raise AssertionError("GQA fused route/coarse decode disagrees with reference")
    if result["gqa_combined_reduce_max_abs"] > 0.03:
        raise AssertionError("combined GQA split/final reduction changed decode output")
    if result["int8_gqa_hip_vs_generic_max_abs"] > 0.03:
        raise AssertionError("INT8 HIP cooperative decode disagrees with generic")
    if result["dynamic_fused_route_decode_max_abs"] > 0.03:
        raise AssertionError("dynamic fused routing disagrees with explicit routing")
    if result["dynamic_fused_route_mean_opened"] >= 8.0:
        raise AssertionError("dynamic fused routing did not mask any routes")
    if result["residual_fused_route_decode_max_abs"] > 0.03:
        raise AssertionError("full-mass fused routing disagrees with explicit routing")
    if not result["residual_fused_route_slots_match"]:
        raise AssertionError("full-mass fused routing chose the wrong route prefix")
    if result["state_bound_fused_decode_max_abs"] > 0.03:
        raise AssertionError(
            "state-bound fused routing disagrees with explicit routing"
        )
    if result["atomic_final_decode_max_abs"] > 0.03:
        raise AssertionError("atomic final decode disagrees with separate reduction")


if __name__ == "__main__":
    main()
