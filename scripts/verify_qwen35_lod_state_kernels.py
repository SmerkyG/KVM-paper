#!/usr/bin/env python3
"""Check fused Qwen3.5 LOD state update and compact top-k routing."""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from model.kernels.lod_kernels import (
    apply_residual_mass_opening,
    merge_state_in_place,
    new_route_buffers,
    new_state_delta_buffers,
    new_state_maxsim_buffers,
    prepare_state_clustering_keys,
    route_logits_coarse_attention,
    route_logits_topk_coarse_attention,
    route_top8_scores_grouped,
    route_top8_state_grouped,
    streaming_state_maxsim,
)


def verify_route_logits_coarse_attention(
    kv_group_size: int = 4,
    *,
    batch: int = 2,
    dim: int = 256,
    query_len: int = 7,
    state_len: int = 37,
    local_len: int = 11,
) -> dict[str, float]:
    kv_heads = 2
    query_heads = kv_heads * kv_group_size
    scale = dim**-0.5
    q = torch.randn(
        batch, query_heads, query_len, dim, device="cuda", dtype=torch.bfloat16
    )
    state_k = torch.randn(
        batch, kv_heads, state_len, dim, device="cuda", dtype=torch.bfloat16
    )
    state_v = torch.randn_like(state_k)
    counts = torch.randint(
        1, 17, (batch, kv_heads, state_len, 1), device="cuda"
    ).float()
    local_k = torch.randn(
        batch, kv_heads, local_len, dim, device="cuda", dtype=torch.bfloat16
    )
    local_v = torch.randn_like(local_k)
    mean_state_k = (state_k.float() / counts).to(torch.bfloat16)
    repeated_state_k = mean_state_k.repeat_interleave(kv_group_size, dim=1)
    route_logits = torch.matmul(q, repeated_state_k.transpose(-1, -2))
    repeated_counts = counts.repeat_interleave(kv_group_size, dim=1).squeeze(-1)
    corrected_scores = route_logits.float() * scale
    corrected_scores += repeated_counts.log().unsqueeze(2)
    top_slots = corrected_scores.topk(8, dim=-1, sorted=False).indices.contiguous()

    actual_output, actual_lse = route_logits_coarse_attention(
        q.contiguous(),
        route_logits.contiguous(),
        state_v.contiguous(),
        counts.contiguous(),
        local_k.contiguous(),
        local_v.contiguous(),
        top_slots,
        state_len=state_len,
        kv_group_size=kv_group_size,
        scale=scale,
    )
    zero_route_output, zero_route_lse = route_logits_coarse_attention(
        q.contiguous(),
        route_logits.contiguous(),
        state_v.contiguous(),
        counts.contiguous(),
        local_k.contiguous(),
        local_v.contiguous(),
        top_slots[..., :0].contiguous(),
        state_len=state_len,
        kv_group_size=kv_group_size,
        scale=scale,
    )
    remote_output, remote_lse = route_logits_coarse_attention(
        q.contiguous(),
        route_logits.contiguous(),
        state_v.contiguous(),
        counts.contiguous(),
        local_k[..., :0, :].contiguous(),
        local_v[..., :0, :].contiguous(),
        top_slots,
        state_len=state_len,
        kv_group_size=kv_group_size,
        scale=scale,
    )
    prefix_q = torch.randn(
        batch,
        query_heads,
        local_len - query_len,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    local_q = torch.cat((prefix_q, q), dim=2)
    local_output, local_lse, *_ = (
        torch.ops.aten._scaled_dot_product_flash_attention.default(
            local_q.contiguous(),
            local_k.contiguous(),
            local_v.contiguous(),
            0.0,
            True,
            False,
            scale=scale,
        )
    )
    local_output = local_output[..., -query_len:, :]
    local_lse = local_lse[..., -query_len:]
    direct_local_output, direct_local_lse, *_ = (
        torch.ops.aten._scaled_dot_product_flash_attention.default(
            q.contiguous(),
            local_k.contiguous(),
            local_v.contiguous(),
            0.0,
            True,
            False,
            scale=scale,
        )
    )
    dynamic_buffers = new_route_buffers(
        q, state_capacity=state_len, include_lse=True
    )
    dynamic_slots, state_full_lse = route_top8_scores_grouped(
        route_logits,
        counts,
        dynamic_buffers,
        kv_group_size=kv_group_size,
        scale=scale,
        topk=8,
        state_len=state_len,
        return_lse=True,
    )
    selected_logits = torch.gather(route_logits, -1, dynamic_slots)
    expanded_counts = repeated_counts.unsqueeze(2).expand(
        -1, -1, query_len, -1
    )
    selected_counts = torch.gather(expanded_counts, -1, dynamic_slots)
    selected_scores = (selected_logits * scale).float() + selected_counts.log()
    sorted_scores, dynamic_order = selected_scores.sort(dim=-1, descending=True)
    sorted_slots = torch.gather(dynamic_slots, -1, dynamic_order)
    full_lse = torch.logaddexp(state_full_lse, local_lse)
    selected_mass = torch.exp(sorted_scores - full_lse.unsqueeze(-1))
    cumulative_before = selected_mass.cumsum(dim=-1) - selected_mass
    remaining_before = selected_mass.sum(dim=-1, keepdim=True) - cumulative_before
    rank = torch.arange(8, device=q.device)
    expected_dynamic_slots = torch.where(
        (rank == 0) | (remaining_before > 0.05),
        sorted_slots,
        torch.full_like(sorted_slots, -1),
    )
    actual_dynamic_slots = apply_residual_mass_opening(
        route_logits,
        counts,
        dynamic_slots,
        state_full_lse,
        local_lse,
        kv_group_size=kv_group_size,
        scale=scale,
        residual_mass=0.05,
    )
    split_lse = torch.logaddexp(remote_lse, local_lse)
    split_weights = torch.softmax(
        torch.stack((remote_lse, local_lse), dim=-1), dim=-1
    ).to(actual_output.dtype)
    split_output = (
        remote_output * split_weights[..., :1]
        + local_output * split_weights[..., 1:].to(local_output.dtype)
    )

    state_scores = corrected_scores.clone()
    state_scores.scatter_(-1, top_slots, float("-inf"))
    grouped_q = q.float().reshape(
        batch, kv_heads, kv_group_size, query_len, dim
    )
    local_scores = torch.matmul(
        grouped_q, local_k.float().transpose(-1, -2).unsqueeze(2)
    ).reshape(batch, query_heads, query_len, local_len)
    local_scores *= scale
    query_index = torch.arange(query_len, device="cuda").unsqueeze(-1)
    local_index = torch.arange(local_len, device="cuda").unsqueeze(0)
    local_scores.masked_fill_(
        ~(local_index <= query_index + local_len - query_len).view(
            1, 1, query_len, local_len
        ),
        float("-inf"),
    )
    scores = torch.cat((state_scores, local_scores), dim=-1)
    expected_lse = torch.logsumexp(scores, dim=-1)
    weights = torch.softmax(scores, dim=-1)
    mean_state_v = (state_v.float() / counts).repeat_interleave(
        kv_group_size, dim=1
    )
    repeated_local_v = local_v.float().repeat_interleave(kv_group_size, dim=1)
    values = torch.cat((mean_state_v, repeated_local_v), dim=2)
    expected_output = torch.matmul(weights, values)
    zero_route_scores = torch.cat((corrected_scores, local_scores), dim=-1)
    zero_route_expected_lse = torch.logsumexp(zero_route_scores, dim=-1)
    zero_route_expected_output = torch.matmul(
        torch.softmax(zero_route_scores, dim=-1), values
    )
    torch.cuda.synchronize()
    return {
        "output_max_abs": float(
            (actual_output.float() - expected_output).abs().max().item()
        ),
        "output_mean_abs": float(
            (actual_output.float() - expected_output).abs().mean().item()
        ),
        "lse_max_abs": float((actual_lse - expected_lse).abs().max().item()),
        "lse_mean_abs": float((actual_lse - expected_lse).abs().mean().item()),
        "zero_route_output_max_abs": float(
            (zero_route_output.float() - zero_route_expected_output)
            .abs()
            .max()
            .item()
        ),
        "zero_route_lse_max_abs": float(
            (zero_route_lse - zero_route_expected_lse).abs().max().item()
        ),
        "split_output_max_abs": float(
            (actual_output.float() - split_output.float()).abs().max().item()
        ),
        "split_lse_max_abs": float((actual_lse - split_lse).abs().max().item()),
        "lower_right_local_output_max_abs": float(
            (direct_local_output - local_output).abs().max().item()
        ),
        "lower_right_local_lse_max_abs": float(
            (direct_local_lse - local_lse).abs().max().item()
        ),
        "residual_opening_exact_fraction": float(
            (actual_dynamic_slots == expected_dynamic_slots).float().mean().item()
        ),
    }


def verify_route_logits_topk_coarse_attention(
    kv_group_size: int = 4,
    *,
    batch: int = 2,
    dim: int = 256,
    query_len: int = 7,
    state_len: int = 37,
    local_len: int = 11,
) -> dict[str, float]:
    kv_heads = 2
    query_heads = kv_heads * kv_group_size
    scale = dim**-0.5
    q = torch.randn(
        batch, query_heads, query_len, dim, device="cuda", dtype=torch.bfloat16
    )
    state_k = torch.randn(
        batch, kv_heads, state_len, dim, device="cuda", dtype=torch.bfloat16
    )
    state_v = torch.randn_like(state_k)
    counts = torch.randint(
        1, 17, (batch, kv_heads, state_len, 1), device="cuda"
    ).float()
    counts[..., :8, :] = torch.arange(1, 9, device="cuda").view(1, 1, 8, 1)
    local_k = torch.randn(
        batch, kv_heads, local_len, dim, device="cuda", dtype=torch.bfloat16
    )
    local_v = torch.randn_like(local_k)
    mean_state_k = (state_k.float() / counts).to(torch.bfloat16)
    repeated_state_k = mean_state_k.repeat_interleave(kv_group_size, dim=1)
    route_logits = torch.matmul(q, repeated_state_k.transpose(-1, -2))
    repeated_counts = counts.repeat_interleave(kv_group_size, dim=1).squeeze(-1)
    corrected_scores = route_logits.float() * scale
    corrected_scores += repeated_counts.log().unsqueeze(2)
    grouped_q = q.float().reshape(
        batch, kv_heads, kv_group_size, query_len, dim
    )
    local_scores = torch.matmul(
        grouped_q, local_k.float().transpose(-1, -2).unsqueeze(2)
    ).reshape(batch, query_heads, query_len, local_len)
    local_scores *= scale
    query_index = torch.arange(query_len, device="cuda").unsqueeze(-1)
    local_index = torch.arange(local_len, device="cuda").unsqueeze(0)
    local_scores.masked_fill_(
        ~(local_index <= query_index + local_len - query_len).view(
            1, 1, query_len, local_len
        ),
        float("-inf"),
    )
    mean_state_v = (state_v.float() / counts).repeat_interleave(
        kv_group_size, dim=1
    )
    repeated_local_v = local_v.float().repeat_interleave(kv_group_size, dim=1)
    values = torch.cat((mean_state_v, repeated_local_v), dim=2)

    def compare(
        topk: int,
        max_leaf_tokens: int | None,
        residual_mass: float | None = None,
        route_count_bias: float = 1.0,
    ) -> dict[str, float]:
        candidate_scores = (
            route_logits.float() * scale
            + route_count_bias * repeated_counts.log().unsqueeze(2)
        )
        if max_leaf_tokens is not None:
            candidate_scores = candidate_scores.masked_fill(
                repeated_counts.unsqueeze(2) > max_leaf_tokens,
                float("-inf"),
            )
        expected_slots = candidate_scores.topk(
            topk, dim=-1, sorted=False
        ).indices
        if residual_mass is not None:
            selected_scores = torch.gather(corrected_scores, -1, expected_slots)
            sorted_scores, order = selected_scores.sort(dim=-1, descending=True)
            expected_slots = torch.gather(expected_slots, -1, order)
            full_lse = torch.logsumexp(
                torch.cat((corrected_scores, local_scores), dim=-1), dim=-1
            )
            selected_mass = torch.exp(sorted_scores - full_lse.unsqueeze(-1))
            cumulative_before = selected_mass.cumsum(dim=-1) - selected_mass
            remaining_before = selected_mass.sum(dim=-1, keepdim=True) - (
                cumulative_before
            )
            rank = torch.arange(topk, device="cuda")
            open_routes = (rank == 0) | (remaining_before > residual_mass)
            expected_slots = torch.where(
                open_routes, expected_slots, torch.full_like(expected_slots, -1)
            )
        actual_slots, actual_output, actual_lse = (
            route_logits_topk_coarse_attention(
                q.contiguous(),
                route_logits.contiguous(),
                state_v.contiguous(),
                counts.contiguous(),
                local_k.contiguous(),
                local_v.contiguous(),
                state_len=state_len,
                kv_group_size=kv_group_size,
                scale=scale,
                route_count_bias=route_count_bias,
                topk=topk,
                max_leaf_tokens=max_leaf_tokens,
                residual_local_lse=(
                    torch.logsumexp(local_scores, dim=-1).contiguous()
                    if residual_mass is not None
                    else None
                ),
                residual_mass=residual_mass,
            )
        )
        state_scores = corrected_scores.clone()
        opened = expected_slots >= 0
        actual_opened = actual_slots >= 0
        opened_slots = F.one_hot(
            expected_slots.clamp_min(0), num_classes=state_len
        ).bool()
        opened_slots &= opened.unsqueeze(-1)
        state_scores.masked_fill_(opened_slots.any(dim=-2), float("-inf"))
        scores = torch.cat((state_scores, local_scores), dim=-1)
        expected_lse = torch.logsumexp(scores, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        expected_output = torch.matmul(weights, values)
        return {
            "route_set_exact_fraction": float(
                (
                    actual_slots.sort(dim=-1).values
                    == expected_slots.sort(dim=-1).values
                )
                .all(dim=-1)
                .float()
                .mean()
                .item()
            ),
            "output_max_abs": float(
                (actual_output.float() - expected_output).abs().max().item()
            ),
            "output_mean_abs": float(
                (actual_output.float() - expected_output).abs().mean().item()
            ),
            "lse_max_abs": float((actual_lse - expected_lse).abs().max().item()),
            "lse_mean_abs": float((actual_lse - expected_lse).abs().mean().item()),
            "mean_opened": float(opened.float().sum(dim=-1).mean().item()),
            "actual_mean_opened": float(
                actual_opened.float().sum(dim=-1).mean().item()
            ),
            "max_open_count_difference": int(
                (
                    actual_opened.sum(dim=-1) - opened.sum(dim=-1)
                ).abs().max().item()
            ),
        }

    result = {
        "top8": compare(8, None),
        "top8_count2": compare(8, None, route_count_bias=2.0),
        "top3_cap8": compare(3, 8),
        "top2_cap8": compare(2, 8),
        "top8_residual05_cap8": compare(8, 8, 0.05),
    }
    torch.cuda.synchronize()
    return result


def verify_concentrated_irregular_gqa_coarse_attention(
    kv_group_size: int,
) -> dict[str, float | int]:
    """Exercise the cancellation case that previously produced all-NaN rows."""
    batch, kv_heads, query_len, state_len, dim = 1, 2, 7, 37, 128
    query_heads = kv_heads * kv_group_size
    scale = dim**-0.5
    q = torch.randn(
        batch, query_heads, query_len, dim,
        device="cuda", dtype=torch.bfloat16,
    )
    route_logits = torch.zeros(
        batch, query_heads, query_len, state_len,
        device="cuda", dtype=torch.bfloat16,
    )
    route_logits[..., 1:4] = 1000.0
    state_v = torch.randn(
        batch, kv_heads, state_len, dim,
        device="cuda", dtype=torch.bfloat16,
    )
    counts = torch.ones(
        batch, kv_heads, state_len, 1, device="cuda", dtype=torch.float32
    )
    local_k = torch.empty(
        batch, kv_heads, 0, dim, device="cuda", dtype=torch.bfloat16
    )
    local_v = torch.empty_like(local_k)
    top_slots, output, lse = route_logits_topk_coarse_attention(
        q.contiguous(),
        route_logits.contiguous(),
        state_v.contiguous(),
        counts.contiguous(),
        local_k,
        local_v,
        state_len=state_len,
        kv_group_size=kv_group_size,
        scale=scale,
        topk=3,
        protected_len=1,
    )
    remaining = torch.ones(state_len, dtype=torch.bool, device="cuda")
    remaining[1:4] = False
    expected_output = state_v[:, :, remaining].float().mean(dim=2)
    expected_output = expected_output.repeat_interleave(
        kv_group_size, dim=1
    ).unsqueeze(2).expand_as(output)
    expected_lse = torch.full_like(lse, math.log(state_len - 3))
    expected_slots = torch.tensor([1, 2, 3], device="cuda")
    slot_sets_match = (
        top_slots.sort(dim=-1).values == expected_slots.sort().values
    ).all(dim=-1)
    torch.cuda.synchronize()
    return {
        "nonfinite_output": int((~torch.isfinite(output)).sum().item()),
        "nonfinite_lse": int((~torch.isfinite(lse)).sum().item()),
        "route_set_exact_fraction": float(slot_sets_match.float().mean().item()),
        "output_max_abs": float(
            (output.float() - expected_output).abs().max().item()
        ),
        "lse_max_abs": float((lse - expected_lse).abs().max().item()),
    }


def verify_streaming_state_geometries() -> dict[str, dict[str, float]]:
    """Compare fused spherical/coherence scans with transient torch geometry."""
    batch, heads, overflow_len, state_len, state_capacity, dim = (
        8,
        3,
        127,
        301,
        384,
        128,
    )
    overflow = torch.randn(
        batch, heads, overflow_len, dim, device="cuda", dtype=torch.bfloat16
    )
    # Allocate spare state capacity without consuming additional random values;
    # later randomized checks in this script retain their established inputs.
    state = torch.zeros(
        batch, heads, state_capacity, dim, device="cuda", dtype=torch.bfloat16
    )
    state[..., :state_len, :] = torch.randn(
        batch, heads, state_len, dim, device="cuda", dtype=torch.bfloat16
    )
    counts = torch.ones(
        batch, heads, state_capacity, 1, device="cuda", dtype=torch.float32
    )
    counts[..., :state_len, :] = torch.randint(
        1, 65, (batch, heads, state_len, 1), device="cuda"
    ).float()
    mean_key = state[..., :state_len, :] / counts[..., :state_len, :].to(
        state.dtype
    )
    mean_norm = (
        mean_key.float().square().mean(dim=-1, keepdim=True).sqrt()
        * torch.empty_like(counts[..., :state_len, :]).uniform_(1.0, 1.8)
    )
    key_norm_sums = torch.zeros_like(counts)
    key_norm_sums[..., :state_len, :] = (
        mean_norm * counts[..., :state_len, :]
    )
    result = {}
    # Reuse buffers across geometries as a live module does; in particular,
    # coherence must split the single spherical scratch into route/append keys.
    buffers = new_state_maxsim_buffers(overflow, overflow_len)
    for geometry in ("spherical", "coherence", "spherical_coherence"):
        leaf = overflow
        if geometry in {"spherical", "spherical_coherence"}:
            leaf = (
                leaf.float()
                * torch.rsqrt(
                    leaf.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
                )
            ).to(leaf.dtype)
        centroid_rms = mean_key.float().square().mean(dim=-1, keepdim=True).sqrt()
        append_key = (
            mean_key.float() / centroid_rms.clamp_min(1e-12)
        ).to(mean_key.dtype)
        route_key = (
            (mean_key.float() / mean_norm.clamp_min(1e-12)).to(mean_key.dtype)
            if geometry in {"coherence", "spherical_coherence"}
            else append_key
        )
        route_scores = torch.matmul(leaf, route_key.transpose(-1, -2))
        append_scores = torch.matmul(leaf, append_key.transpose(-1, -2))
        expected_select = append_scores.max(dim=-1).values
        route_scores[..., :16] = float("-inf")
        expected_route, expected_index = route_scores.max(dim=-1)
        append_base_index_fraction = 1.0
        append_base_candidate_recall = {2: 1.0, 4: 1.0, 8: 1.0}
        if geometry in {"coherence", "spherical_coherence"}:
            coherence = centroid_rms / mean_norm.clamp_min(1e-12)
            append_base_route_scores = (
                append_scores.float() * coherence.squeeze(-1).unsqueeze(-2)
            )
            append_base_route_scores[..., :16] = float("-inf")
            append_base_index = append_base_route_scores.argmax(dim=-1)
            append_base_index_fraction = float(
                (append_base_index == expected_index).float().mean().item()
            )
            append_base_candidate_recall = {
                candidate_count: float(
                    (
                        append_base_route_scores
                        .topk(candidate_count, dim=-1)
                        .indices
                        == expected_index.unsqueeze(-1)
                    )
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
                for candidate_count in (2, 4, 8)
            }
        actual_route, actual_index, actual_select = streaming_state_maxsim(
            leaf,
            state,
            counts,
            buffers,
            state_len=state_len,
            sink_len=16,
            key_norm_sums=key_norm_sums,
            geometry=geometry,
            materialize_prepared_scores=True,
        )
        result[geometry] = {
            "route_index_exact_fraction": float(
                (actual_index == expected_index).float().mean().item()
            ),
            "route_score_max_abs": float(
                (actual_route - expected_route).abs().max().item()
            ),
            "select_score_max_abs": float(
                (actual_select - expected_select).abs().max().item()
            ),
            "append_top16_set_exact_fraction": float(
                (
                    actual_select.topk(16, dim=-1).indices.sort(dim=-1).values
                    == expected_select.topk(16, dim=-1)
                    .indices.sort(dim=-1).values
                )
                .all(dim=-1)
                .float()
                .mean()
                .item()
            ),
            "append_base_route_index_exact_fraction": append_base_index_fraction,
            **{
                f"append_base_top{candidate_count}_route_recall": recall
                for candidate_count, recall in append_base_candidate_recall.items()
            },
        }
    return result


def verify_incremental_state_geometries() -> dict[str, dict[str, float]]:
    """Check sparse refreshes against a full dense geometry reconstruction."""
    batch, heads, overflow_len, state_len, dim = 8, 2, 127, 301, 128
    overflow = torch.randn(
        batch, heads, overflow_len, dim, device="cuda", dtype=torch.bfloat16
    )
    base_state = torch.randn(
        batch, heads, state_len, dim, device="cuda", dtype=torch.bfloat16
    )
    base_counts = torch.randint(
        1, 65, (batch, heads, state_len, 1), device="cuda"
    ).float()
    result = {}
    refresh_begin, refresh_end = 32, 64
    refresh_slots = (
        torch.arange(refresh_begin, refresh_end, device="cuda")
        .view(1, 1, -1)
        .expand(batch, heads, -1)
    )
    for geometry in ("spherical", "coherence"):
        state = base_state.clone()
        counts = base_counts.clone()
        mean_key = state / counts.to(state.dtype)
        mean_norm = (
            mean_key.float().square().mean(dim=-1, keepdim=True).sqrt()
            * torch.empty_like(counts).uniform_(1.0, 1.8)
        )
        key_norm_sums = mean_norm * counts
        leaf = overflow
        if geometry == "spherical":
            leaf = (
                leaf.float()
                * torch.rsqrt(
                    leaf.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
                )
            ).to(leaf.dtype)
        buffers = new_state_maxsim_buffers(leaf, overflow_len)
        streaming_state_maxsim(
            leaf,
            state,
            counts,
            buffers,
            state_len=state_len,
            sink_len=16,
            key_norm_sums=(key_norm_sums if geometry == "coherence" else None),
            geometry=geometry,
            materialize_prepared_scores=True,
        )
        state[..., refresh_begin:refresh_end, :].add_(
            torch.randn_like(state[..., refresh_begin:refresh_end, :])
        )
        counts[..., refresh_begin:refresh_end, :].add_(1.0)
        if geometry == "coherence":
            key_norm_sums[..., refresh_begin:refresh_end, :].add_(
                torch.rand_like(
                    key_norm_sums[..., refresh_begin:refresh_end, :]
                )
            )
        prepare_state_clustering_keys(
            state,
            counts,
            buffers,
            state_len=state_len,
            key_norm_sums=(key_norm_sums if geometry == "coherence" else None),
            geometry=geometry,
            slot_indices=refresh_slots,
            prepare_coherence_route=True,
        )
        actual_route, actual_index, actual_select = streaming_state_maxsim(
            leaf,
            state,
            counts,
            buffers,
            state_len=state_len,
            sink_len=16,
            key_norm_sums=(key_norm_sums if geometry == "coherence" else None),
            geometry=geometry,
            prepare_state_geometry=False,
            materialize_prepared_scores=True,
        )
        mean_key = state / counts.to(state.dtype)
        centroid_rms = mean_key.float().square().mean(dim=-1, keepdim=True).sqrt()
        append_key = (
            mean_key.float() / centroid_rms.clamp_min(1e-12)
        ).to(mean_key.dtype)
        route_key = append_key
        if geometry == "coherence":
            mean_norm = key_norm_sums / counts
            route_key = (
                mean_key.float() / mean_norm.clamp_min(1e-12)
            ).to(mean_key.dtype)
        route_scores = torch.matmul(leaf, route_key.transpose(-1, -2))
        append_scores = torch.matmul(leaf, append_key.transpose(-1, -2))
        expected_select = append_scores.max(dim=-1).values
        route_scores[..., :16] = float("-inf")
        expected_route, expected_index = route_scores.max(dim=-1)
        result[geometry] = {
            "route_index_exact_fraction": float(
                (actual_index == expected_index).float().mean().item()
            ),
            "route_score_max_abs": float(
                (actual_route - expected_route).abs().max().item()
            ),
            "append_top16_set_exact_fraction": float(
                (
                    actual_select.topk(16, dim=-1).indices.sort(dim=-1).values
                    == expected_select.topk(16, dim=-1)
                    .indices.sort(dim=-1).values
                )
                .all(dim=-1)
                .float()
                .mean()
                .item()
            ),
        }
    return result


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, kv_heads, q_heads = 1, 2, 8
    state_len, overflow_len, merge_len, dim = 333, 256, 173, 256

    state_k = torch.randn(
        batch, kv_heads, state_len, dim, device=device, dtype=torch.bfloat16
    )
    state_v = torch.randn_like(state_k)
    counts = torch.randint(
        1, 17, (batch, kv_heads, state_len, 1), device=device
    ).float()
    merge_k = torch.randn(
        batch, kv_heads, merge_len, dim, device=device, dtype=torch.bfloat16
    )
    merge_v = torch.randn_like(merge_k)
    merge_counts = (
        torch.arange(merge_len, device=device).remainder(4).add(1).float()
        .view(1, 1, merge_len, 1)
        .expand(batch, kv_heads, -1, -1)
        .contiguous()
    )
    merge_indices = (
        torch.argsort(torch.rand(batch, kv_heads, overflow_len, device=device), dim=-1)[
            ..., :merge_len
        ]
        .sort(dim=-1)
        .values
    )
    destinations = torch.randint(
        1, state_len, (batch, kv_heads, merge_len), device=device
    )
    owners = torch.full(
        (batch, kv_heads, overflow_len), -1, dtype=torch.long, device=device
    )

    assignment = F.one_hot(destinations, num_classes=state_len).to(merge_k.dtype)
    assignment_t = assignment.transpose(-1, -2)
    expected_k = state_k + torch.matmul(assignment_t, merge_k)
    expected_v = state_v + torch.matmul(assignment_t, merge_v)
    expected_counts = counts + torch.matmul(assignment_t.float(), merge_counts)
    expected_owners = owners.clone().scatter(2, merge_indices, destinations)
    destination_counts = assignment_t.float().sum(dim=-1)
    singleton_slots = destination_counts == 1
    collision_slots = destination_counts > 1
    fp32_delta_k = torch.zeros_like(state_k, dtype=torch.float32).scatter_add_(
        2,
        destinations.unsqueeze(-1).expand_as(merge_k),
        merge_k.float(),
    )
    fp32_delta_v = torch.zeros_like(state_v, dtype=torch.float32).scatter_add_(
        2,
        destinations.unsqueeze(-1).expand_as(merge_v),
        merge_v.float(),
    )
    fp32_expected_k = (state_k.float() + fp32_delta_k).to(torch.bfloat16)
    fp32_expected_v = (state_v.float() + fp32_delta_v).to(torch.bfloat16)

    actual_k = state_k.clone()
    actual_v = state_v.clone()
    actual_counts = counts.clone()
    buffers = new_state_delta_buffers(actual_k, actual_v, 512)
    merge_state_in_place(
        actual_k,
        actual_v,
        actual_counts,
        merge_k,
        merge_v,
        merge_counts,
        merge_indices.contiguous(),
        destinations.contiguous(),
        owners,
        buffers,
    )
    torch.cuda.synchronize()
    delta_buffers_cleared = all(
        not torch.count_nonzero(buffers[name])
        for name in ("delta_k", "delta_v", "delta_counts", "touched")
    )

    state_key_norm_sums = torch.rand_like(counts).mul_(counts)
    merge_key_norm_sums = torch.rand_like(merge_counts).mul_(merge_counts)
    expected_key_norm_sums = state_key_norm_sums + torch.matmul(
        assignment_t.float(), merge_key_norm_sums
    )
    actual_key_norm_sums = state_key_norm_sums.clone()
    norm_actual_k = state_k.clone()
    norm_actual_v = state_v.clone()
    norm_actual_counts = counts.clone()
    norm_owners = torch.full_like(owners, -1)
    norm_buffers = new_state_delta_buffers(norm_actual_k, norm_actual_v, 512)
    merge_state_in_place(
        norm_actual_k,
        norm_actual_v,
        norm_actual_counts,
        merge_k,
        merge_v,
        merge_counts,
        merge_indices.contiguous(),
        destinations.contiguous(),
        norm_owners,
        norm_buffers,
        key_norm_sums=actual_key_norm_sums,
        merge_key_norm_sums=merge_key_norm_sums,
    )
    torch.cuda.synchronize()
    key_norm_update_max_abs = float(
        (actual_key_norm_sums - expected_key_norm_sums).abs().max().item()
    )

    route_state_k = torch.randn_like(state_k)
    route_counts = torch.randint(
        1, 33, (batch, kv_heads, state_len, 1), device=device
    ).float()
    route_results = {}

    overflow_route_k = torch.randn(
        batch, kv_heads, overflow_len, dim, device=device, dtype=torch.bfloat16
    )
    dense_maxsim = torch.matmul(
        overflow_route_k,
        (route_state_k / route_counts.to(torch.bfloat16)).transpose(-1, -2),
    )
    expected_select_scores = dense_maxsim.max(dim=-1).values
    dense_maxsim[..., 0] = float("-inf")
    expected_route_scores, expected_route_indices = dense_maxsim.max(dim=-1)
    maxsim_buffers = new_state_maxsim_buffers(overflow_route_k, overflow_len)
    (
        actual_route_scores,
        actual_route_indices,
        actual_select_scores,
    ) = streaming_state_maxsim(
        overflow_route_k,
        route_state_k,
        route_counts,
        maxsim_buffers,
        state_len=state_len,
        sink_len=1,
    )
    maxsim_result = {
        "route_index_exact_fraction": float(
            (actual_route_indices == expected_route_indices).float().mean().item()
        ),
        "route_score_max_abs": float(
            (actual_route_scores - expected_route_scores).abs().max().item()
        ),
        "select_score_max_abs": float(
            (actual_select_scores - expected_select_scores).abs().max().item()
        ),
    }
    for query_len in (1, 257):
        q = torch.randn(
            batch, q_heads, query_len, dim, device=device, dtype=torch.bfloat16
        )
        mean_k = (route_state_k / route_counts.to(torch.bfloat16)).repeat_interleave(
            q_heads // kv_heads, dim=1
        )
        logits = torch.matmul(q, mean_k.transpose(-1, -2))
        logits = logits * (dim**-0.5)
        log_counts = (
            route_counts.repeat_interleave(q_heads // kv_heads, dim=1).squeeze(-1).log()
        )
        expected_lse = torch.logsumexp(logits + log_counts.unsqueeze(2), dim=-1)
        for topk in (3, 8):
            expected_route = (
                (logits + log_counts.unsqueeze(2))
                .topk(topk, dim=-1, sorted=False)
                .indices
            )
            route_buffers = new_route_buffers(
                q, state_capacity=512, include_lse=True
            )
            actual_route = route_top8_state_grouped(
                q,
                F.pad(route_state_k, (0, 0, 0, 512 - state_len)),
                F.pad(route_counts, (0, 0, 0, 512 - state_len)),
                route_buffers,
                kv_group_size=q_heads // kv_heads,
                scale=dim**-0.5,
                topk=topk,
                state_len=state_len,
            ).clone()
            score_route, score_lse = route_top8_scores_grouped(
                torch.matmul(q, mean_k.transpose(-1, -2)),
                route_counts,
                route_buffers,
                kv_group_size=q_heads // kv_heads,
                scale=dim**-0.5,
                topk=topk,
                state_len=state_len,
                return_lse=True,
            )
            expected_sorted = expected_route.sort(dim=-1).values
            actual_sorted = actual_route.sort(dim=-1).values
            route_results[f"{query_len}_top{topk}"] = {
                "route_set_exact_fraction": float(
                    (expected_sorted == actual_sorted)
                    .all(dim=-1)
                    .float()
                    .mean()
                    .item()
                ),
                "slot_recall": float(
                    (actual_route.unsqueeze(-1) == expected_route.unsqueeze(-2))
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                ),
                "score_route_set_exact_fraction": float(
                    (expected_sorted == score_route.sort(dim=-1).values)
                    .all(dim=-1)
                    .float()
                    .mean()
                    .item()
                ),
                "score_route_order_exact_fraction": float(
                    (expected_route == score_route)
                    .all(dim=-1)
                    .float()
                    .mean()
                    .item()
                ),
                "score_lse_max_abs": float(
                    (expected_lse - score_lse).abs().max().item()
                ),
                "example_reference": [
                    int(value) for value in expected_route[0, 0, 0].tolist()
                ],
                "example_grouped": [
                    int(value) for value in score_route[0, 0, 0].tolist()
                ],
            }

    k_error = (actual_k - fp32_expected_k).abs().float()
    flat_worst_k = int(k_error.argmax().item())
    worst_k = []
    remainder = flat_worst_k
    for size in reversed(actual_k.shape):
        worst_k.append(remainder % size)
        remainder //= size
    worst_k.reverse()
    kb, kh, ks, kd = worst_k
    contributing_k = merge_k[kb, kh][destinations[kb, kh] == ks, kd]
    contributing_tokens = torch.nonzero(
        destinations[kb, kh] == ks, as_tuple=False
    ).squeeze(-1)

    result = {
        "state_k_max_abs": float((actual_k - expected_k).abs().max().item()),
        "state_k_mean_abs": float((actual_k - expected_k).abs().float().mean().item()),
        "state_k_fp32_max_abs": float((actual_k - fp32_expected_k).abs().max().item()),
        "state_k_fp32_mean_abs": float(
            (actual_k - fp32_expected_k).abs().float().mean().item()
        ),
        "state_k_singleton_mean_abs": float(
            (actual_k - fp32_expected_k)
            .abs()
            .float()[singleton_slots.unsqueeze(-1).expand_as(actual_k)]
            .mean()
            .item()
        ),
        "state_k_collision_mean_abs": float(
            (actual_k - fp32_expected_k)
            .abs()
            .float()[collision_slots.unsqueeze(-1).expand_as(actual_k)]
            .mean()
            .item()
        ),
        "state_k_worst": {
            "index": worst_k,
            "old": float(state_k[kb, kh, ks, kd].item()),
            "actual": float(actual_k[kb, kh, ks, kd].item()),
            "expected": float(fp32_expected_k[kb, kh, ks, kd].item()),
            "contributors": [float(value) for value in contributing_k.tolist()],
            "contributing_tokens": [
                int(value) for value in contributing_tokens.tolist()
            ],
        },
        "state_v_max_abs": float((actual_v - expected_v).abs().max().item()),
        "state_v_mean_abs": float((actual_v - expected_v).abs().float().mean().item()),
        "state_v_fp32_max_abs": float((actual_v - fp32_expected_v).abs().max().item()),
        "state_v_fp32_mean_abs": float(
            (actual_v - fp32_expected_v).abs().float().mean().item()
        ),
        "counts_exact": bool(torch.equal(actual_counts, expected_counts)),
        "owners_exact": bool(torch.equal(owners, expected_owners)),
        "buffers_cleared": bool(delta_buffers_cleared),
        "routing": route_results,
        "streaming_state_maxsim": maxsim_result,
        "streaming_state_geometries": verify_streaming_state_geometries(),
        "incremental_state_geometries": verify_incremental_state_geometries(),
        "key_norm_update_max_abs": key_norm_update_max_abs,
        "route_logits_coarse_attention": verify_route_logits_coarse_attention(),
        "route_logits_topk_coarse_attention": (
            verify_route_logits_topk_coarse_attention()
        ),
        "irregular_gqa": {
            str(kv_group_size): {
                "route_logits_coarse_attention": (
                    verify_route_logits_coarse_attention(kv_group_size)
                ),
                "route_logits_topk_coarse_attention": (
                    verify_route_logits_topk_coarse_attention(kv_group_size)
                ),
                "concentrated_top3": (
                    verify_concentrated_irregular_gqa_coarse_attention(
                        kv_group_size
                    )
                ),
            }
            for kv_group_size in (5, 6)
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["counts_exact"] or not result["owners_exact"]:
        raise AssertionError("fused state metadata differs from the reference")
    if min(item["slot_recall"] for item in route_results.values()) < 0.999:
        raise AssertionError(
            "fused top-8 routing differs materially from the reference"
        )
    if maxsim_result["route_index_exact_fraction"] < 0.99:
        raise AssertionError("streaming state max-sim differs materially")
    for geometry, geometry_result in result["streaming_state_geometries"].items():
        if geometry_result["route_index_exact_fraction"] < 0.999:
            raise AssertionError(f"streaming {geometry} state routing differs")
        if max(
            geometry_result["route_score_max_abs"],
            geometry_result["select_score_max_abs"],
        ) > 1e-4:
            raise AssertionError(f"streaming {geometry} state scores differ")
    for geometry, geometry_result in result["incremental_state_geometries"].items():
        if geometry_result["route_index_exact_fraction"] < 0.999:
            raise AssertionError(f"incremental {geometry} state routing differs")
        if geometry_result["append_top16_set_exact_fraction"] < 0.999:
            raise AssertionError(f"incremental {geometry} append selection differs")
    if key_norm_update_max_abs > 1e-4:
        raise AssertionError("fused key-norm state update differs")
    if max(item["score_lse_max_abs"] for item in route_results.values()) > 1e-4:
        raise AssertionError("fused routing LSE differs materially from reference")
    if result["route_logits_coarse_attention"]["output_max_abs"] > 0.02:
        raise AssertionError("fused route-logit coarse output differs materially")
    if result["route_logits_coarse_attention"]["lse_max_abs"] > 0.02:
        raise AssertionError("fused route-logit coarse LSE differs materially")
    if result["route_logits_coarse_attention"]["split_output_max_abs"] > 0.02:
        raise AssertionError("split local/coarse output differs materially")
    if result["route_logits_coarse_attention"]["split_lse_max_abs"] > 0.02:
        raise AssertionError("split local/coarse LSE differs materially")
    if (
        result["route_logits_coarse_attention"][
            "lower_right_local_output_max_abs"
        ]
        > 0.002
    ):
        raise AssertionError("lower-right FlashAttention output differs materially")
    if (
        result["route_logits_coarse_attention"][
            "lower_right_local_lse_max_abs"
        ]
        > 1e-4
    ):
        raise AssertionError("lower-right FlashAttention LSE differs materially")
    if (
        result["route_logits_coarse_attention"][
            "residual_opening_exact_fraction"
        ]
        < 0.999
    ):
        raise AssertionError("fused residual-mass opening differs materially")
    for label, fused_topk in result[
        "route_logits_topk_coarse_attention"
    ].items():
        adaptive = "residual" in label
        minimum_route_agreement = 0.97 if adaptive else 0.999
        maximum_output_error = 0.04 if adaptive else 0.03
        maximum_lse_error = 0.04 if adaptive else 0.03
        if fused_topk["route_set_exact_fraction"] < minimum_route_agreement:
            raise AssertionError(f"fused coarse {label} routing differs materially")
        if fused_topk["max_open_count_difference"] > (1 if adaptive else 0):
            raise AssertionError(f"fused coarse {label} opening count differs materially")
        if fused_topk["output_max_abs"] > maximum_output_error:
            raise AssertionError(f"fused {label} coarse output differs materially")
        if fused_topk["lse_max_abs"] > maximum_lse_error:
            raise AssertionError(f"fused {label} coarse LSE differs materially")
    for kv_group_size, irregular in result["irregular_gqa"].items():
        coarse = irregular["route_logits_coarse_attention"]
        if coarse["output_max_abs"] > 0.02 or coarse["lse_max_abs"] > 0.02:
            raise AssertionError(
                f"irregular GQA group {kv_group_size} coarse attention differs"
            )
        for label, fused_topk in irregular[
            "route_logits_topk_coarse_attention"
        ].items():
            adaptive = "residual" in label
            minimum_route_agreement = 0.85 if adaptive else 0.999
            maximum_output_error = 0.04 if adaptive else 0.03
            maximum_lse_error = 0.04 if adaptive else 0.03
            if fused_topk["route_set_exact_fraction"] < minimum_route_agreement:
                raise AssertionError(
                    f"irregular GQA group {kv_group_size} {label} routes differ"
                )
            if fused_topk["max_open_count_difference"] > (1 if adaptive else 0):
                raise AssertionError(
                    f"irregular GQA group {kv_group_size} {label} counts differ"
                )
            if fused_topk["output_max_abs"] > maximum_output_error:
                raise AssertionError(
                    f"irregular GQA group {kv_group_size} {label} output differs"
                )
            if fused_topk["lse_max_abs"] > maximum_lse_error:
                raise AssertionError(
                    f"irregular GQA group {kv_group_size} {label} LSE differs"
                )
        concentrated = irregular["concentrated_top3"]
        if concentrated["nonfinite_output"] or concentrated["nonfinite_lse"]:
            raise AssertionError(
                f"irregular GQA group {kv_group_size} concentrated coarse is non-finite"
            )
        if concentrated["route_set_exact_fraction"] < 0.999:
            raise AssertionError(
                f"irregular GQA group {kv_group_size} concentrated routes differ"
            )
        if concentrated["output_max_abs"] > 0.02:
            raise AssertionError(
                f"irregular GQA group {kv_group_size} concentrated output differs"
            )
        if concentrated["lse_max_abs"] > 0.02:
            raise AssertionError(
                f"irregular GQA group {kv_group_size} concentrated LSE differs"
            )


if __name__ == "__main__":
    main()
