#!/usr/bin/env python3
"""Check fused Qwen3.5 LOD state update and compact top-k routing."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from model.kernels.qwen35_lod_kernels import (
    apply_residual_mass_opening,
    merge_state_in_place,
    new_route_buffers,
    new_state_delta_buffers,
    new_state_maxsim_buffers,
    route_logits_coarse_attention,
    route_logits_topk_coarse_attention,
    route_top8_scores_grouped,
    route_top8_state_grouped,
    streaming_state_maxsim,
)


def verify_route_logits_coarse_attention() -> dict[str, float]:
    batch, kv_heads, kv_group_size = 2, 2, 4
    query_heads = kv_heads * kv_group_size
    query_len, state_len, local_len, dim = 7, 37, 11, 256
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


def verify_route_logits_topk_coarse_attention() -> dict[str, float]:
    batch, kv_heads, kv_group_size = 2, 2, 4
    query_heads = kv_heads * kv_group_size
    query_len, state_len, local_len, dim = 7, 37, 11, 256
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
    expected_slots = corrected_scores.topk(8, dim=-1, sorted=False).indices

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
        )
    )

    state_scores = corrected_scores.clone()
    state_scores.scatter_(-1, expected_slots, float("-inf"))
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
    torch.cuda.synchronize()
    return {
        "top8_set_exact_fraction": float(
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
    }


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
    expected_counts = counts + assignment_t.float().sum(dim=-1, keepdim=True)
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
        expected_route = (
            (logits + log_counts.unsqueeze(2)).topk(8, dim=-1, sorted=False).indices
        )
        route_buffers = new_route_buffers(q, state_capacity=512, include_lse=True)
        actual_route = route_top8_state_grouped(
            q,
            F.pad(route_state_k, (0, 0, 0, 512 - state_len)),
            F.pad(route_counts, (0, 0, 0, 512 - state_len)),
            route_buffers,
            kv_group_size=q_heads // kv_heads,
            scale=dim**-0.5,
            topk=8,
            state_len=state_len,
        ).clone()
        score_route, score_lse = route_top8_scores_grouped(
            torch.matmul(q, mean_k.transpose(-1, -2)),
            route_counts,
            route_buffers,
            kv_group_size=q_heads // kv_heads,
            scale=dim**-0.5,
            topk=8,
            state_len=state_len,
            return_lse=True,
        )
        expected_lse = torch.logsumexp(logits + log_counts.unsqueeze(2), dim=-1)
        expected_sorted = expected_route.sort(dim=-1).values
        actual_sorted = actual_route.sort(dim=-1).values
        route_results[str(query_len)] = {
            "top8_set_exact_fraction": float(
                (expected_sorted == actual_sorted).all(dim=-1).float().mean().item()
            ),
            "slot_recall": float(
                (actual_route.unsqueeze(-1) == expected_route.unsqueeze(-2))
                .any(dim=-1)
                .float()
                .mean()
                .item()
            ),
            "score_top8_set_exact_fraction": float(
                (expected_sorted == score_route.sort(dim=-1).values)
                .all(dim=-1)
                .float()
                .mean()
                .item()
            ),
            "score_top8_order_exact_fraction": float(
                (expected_route == score_route).all(dim=-1).float().mean().item()
            ),
            "score_lse_max_abs": float(
                (expected_lse - score_lse).abs().max().item()
            ),
            "example_reference": [
                int(value) for value in expected_route[0, 0, 0].tolist()
            ],
            "example_grouped": [int(value) for value in score_route[0, 0, 0].tolist()],
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
        "route_logits_coarse_attention": verify_route_logits_coarse_attention(),
        "route_logits_topk_coarse_attention": (
            verify_route_logits_topk_coarse_attention()
        ),
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
    fused_topk = result["route_logits_topk_coarse_attention"]
    if fused_topk["top8_set_exact_fraction"] < 0.999:
        raise AssertionError("fused coarse top-8 routing differs materially")
    if fused_topk["output_max_abs"] > 0.03:
        raise AssertionError("fused top-8 coarse output differs materially")
    if fused_topk["lse_max_abs"] > 0.03:
        raise AssertionError("fused top-8 coarse LSE differs materially")


if __name__ == "__main__":
    main()
