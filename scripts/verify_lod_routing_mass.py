#!/usr/bin/env python3
"""Verify exact candidate-slot routing mass for materialized and virtual pages."""

from __future__ import annotations

import torch

from model.kernels.paged_leaf_attention import (
    refine_route_candidates_by_leaf_mass,
    refine_route_candidates_by_virtual_leaf_mass,
    refine_route_candidates_by_virtual_leaf_output,
)


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, kv_heads, query_heads = 1, 2, 4
    query_len, slots, pages_per_slot = 8, 144, 2
    page_size, head_dim, candidates_count = 16, 64, 8
    page_capacity = slots * pages_per_slot
    group_size = query_heads // kv_heads
    lengths = torch.randint(
        1, pages_per_slot * page_size + 1, (slots,), device=device
    )
    page_k = torch.randn(
        batch,
        kv_heads,
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    page_v = torch.randn_like(page_k)
    page_counts = torch.zeros(
        batch, kv_heads, page_capacity, device=device, dtype=torch.int32
    )
    for slot in range(slots):
        for page in range(pages_per_slot):
            page_counts[..., slot * pages_per_slot + page] = max(
                min(int(lengths[slot].item()) - page * page_size, page_size), 0
            )
    slot_pages = torch.arange(
        page_capacity, device=device, dtype=torch.int32
    ).view(1, 1, slots, pages_per_slot).expand(
        batch, kv_heads, -1, -1
    ).contiguous()
    slot_lengths = lengths.view(1, 1, slots).expand(
        batch, kv_heads, -1
    ).to(torch.int32).contiguous()
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, device=device, dtype=torch.int32
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), device=device, dtype=torch.int32)
    q = torch.randn(
        batch,
        query_heads,
        query_len,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    candidates = torch.rand(
        batch, query_heads, query_len, slots, device=device
    ).topk(candidates_count, dim=-1, sorted=False).indices
    scale = head_dim**-0.5

    actual_page = refine_route_candidates_by_leaf_mass(
        q,
        page_k,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
        kv_group_size=group_size,
        scale=scale,
        hash_probes=0,
    )
    leaf_k = page_k.flatten(2, 3)
    page_indices = torch.arange(
        page_capacity * page_size, device=device, dtype=torch.int32
    ).view(1, 1, page_capacity, page_size).expand(
        batch, kv_heads, -1, -1
    ).contiguous()
    actual_virtual = refine_route_candidates_by_virtual_leaf_mass(
        q,
        leaf_k,
        page_indices,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
        kv_group_size=group_size,
        scale=scale,
        hash_probes=0,
    )
    leaf_v = page_v.flatten(2, 3)
    state_sum_k = torch.zeros(
        batch, kv_heads, slots, head_dim, device=device, dtype=torch.float32
    )
    state_sum_v = torch.zeros_like(state_sum_k)
    state_counts = slot_lengths.clone()
    for slot in range(slots):
        length = int(lengths[slot])
        state_sum_k[..., slot, :] = page_k[
            ..., slot * pages_per_slot : (slot + 1) * pages_per_slot, :, :
        ].flatten(-3, -2)[..., :length, :].float().sum(-2)
        state_sum_v[..., slot, :] = page_v[
            ..., slot * pages_per_slot : (slot + 1) * pages_per_slot, :, :
        ].flatten(-3, -2)[..., :length, :].float().sum(-2)
    repeated_sum_k = state_sum_k.repeat_interleave(group_size, dim=1)
    repeated_sum_v = state_sum_v.repeat_interleave(group_size, dim=1)
    repeated_counts = state_counts.repeat_interleave(group_size, dim=1)
    state_scores = torch.einsum(
        "bhtd,bhsd->bhts",
        q.float(),
        repeated_sum_k / repeated_counts[..., None].float(),
    ) * scale + repeated_counts.float().log().unsqueeze(2)
    baseline_lse = torch.logsumexp(state_scores, dim=-1)
    baseline_output = torch.einsum(
        "bhts,bhsd->bhtd",
        state_scores.softmax(-1),
        repeated_sum_v / repeated_counts[..., None].float(),
    )
    candidate_coarse_lse = torch.gather(state_scores, -1, candidates)
    actual_output_utility = refine_route_candidates_by_virtual_leaf_output(
        q,
        baseline_output,
        baseline_lse,
        candidate_coarse_lse,
        state_sum_v,
        state_counts,
        leaf_k,
        leaf_v,
        page_indices,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        candidates,
        kv_group_size=group_size,
        scale=scale,
        hash_probes=0,
    )
    expected = torch.empty_like(actual_page)
    expected_output_utility = torch.empty_like(actual_output_utility)
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        for query_index in range(query_len):
            query = q[0, query_head, query_index].float()
            closed_lse = baseline_lse[0, query_head, query_index]
            closed_output = baseline_output[0, query_head, query_index]
            candidate_terms = []
            for candidate_rank in range(candidates_count):
                slot = int(candidates[0, query_head, query_index, candidate_rank])
                length = int(lengths[slot])
                keys = page_k[
                    0,
                    kv_head,
                    slot * pages_per_slot : (slot + 1) * pages_per_slot,
                ].flatten(0, 1)[:length].float()
                expected[0, query_head, query_index, candidate_rank] = (
                    torch.logsumexp(keys @ query * scale, dim=0)
                )
                values = page_v[
                    0,
                    kv_head,
                    slot * pages_per_slot : (slot + 1) * pages_per_slot,
                ].flatten(0, 1)[:length].float()
                exact_scores = keys @ query * scale
                exact_lse = torch.logsumexp(exact_scores, dim=0)
                exact_value = exact_scores.softmax(0) @ values
                coarse_lse = candidate_coarse_lse[
                    0, query_head, query_index, candidate_rank
                ]
                coarse_mass = torch.exp(coarse_lse - closed_lse)
                exact_mass = torch.exp(exact_lse - closed_lse)
                coarse_value = state_sum_v[0, kv_head, slot] / float(length)
                candidate_terms.append(
                    (coarse_mass, exact_mass, coarse_value, exact_value)
                )
            target_denominator = 1.0 + sum(
                exact_mass - coarse_mass
                for coarse_mass, exact_mass, _, _ in candidate_terms
            )
            target_output = (
                closed_output
                + sum(
                    exact_mass * exact_value - coarse_mass * coarse_value
                    for coarse_mass, exact_mass, coarse_value, exact_value
                    in candidate_terms
                )
            ) / target_denominator
            baseline_error = (target_output - closed_output).square().sum()
            for candidate_rank, terms in enumerate(candidate_terms):
                coarse_mass, exact_mass, coarse_value, exact_value = terms
                denominator = 1.0 - coarse_mass + exact_mass
                opened_output = (
                    closed_output
                    - coarse_mass * coarse_value
                    + exact_mass * exact_value
                ) / denominator
                expected_output_utility[
                    0, query_head, query_index, candidate_rank
                ] = baseline_error - (target_output - opened_output).square().sum()

    torch.testing.assert_close(actual_page, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_virtual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(
        actual_output_utility,
        expected_output_utility,
        rtol=1e-3,
        atol=1e-4,
    )

    wide_candidates = torch.rand(
        batch, query_heads, query_len, slots, device=device
    ).topk(128, dim=-1, sorted=False).indices
    actual_wide = refine_route_candidates_by_virtual_leaf_mass(
        q,
        leaf_k,
        page_indices,
        page_counts,
        slot_pages,
        overflow_page_keys,
        overflow_page_values,
        overflow_used,
        slot_lengths,
        wide_candidates,
        kv_group_size=group_size,
        scale=scale,
        hash_probes=0,
    )
    expected_wide = torch.empty_like(actual_wide)
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        for query_index in range(query_len):
            query = q[0, query_head, query_index].float()
            for candidate_rank in range(128):
                slot = int(wide_candidates[0, query_head, query_index, candidate_rank])
                length = int(lengths[slot])
                keys = page_k[
                    0,
                    kv_head,
                    slot * pages_per_slot : (slot + 1) * pages_per_slot,
                ].flatten(0, 1)[:length].float()
                expected_wide[0, query_head, query_index, candidate_rank] = (
                    torch.logsumexp(keys @ query * scale, dim=0)
                )
    torch.testing.assert_close(actual_wide, expected_wide, rtol=2e-4, atol=2e-4)
    print(
        {
            "page_max_abs": float((actual_page - expected).abs().max()),
            "virtual_max_abs": float((actual_virtual - expected).abs().max()),
            "output_utility_max_abs": float(
                (actual_output_utility - expected_output_utility).abs().max()
            ),
            "wide_virtual_max_abs": float(
                (actual_wide - expected_wide).abs().max()
            ),
        }
    )


if __name__ == "__main__":
    main()
