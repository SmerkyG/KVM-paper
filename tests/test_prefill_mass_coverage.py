from __future__ import annotations

import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_mass_coverage_counts_local_sink_and_opened_regions() -> None:
    from lod_attention.kernels.aiter_prefill_attention import (
        _select_routes_by_mass_coverage_kernel,
    )
    from lod_attention.kernels.paged_prefill import _pack_expert_routes

    device = torch.device("cuda")
    q = torch.zeros((1, 2, 2, 128), device=device, dtype=torch.bfloat16)
    q[..., 0] = 1
    mean_k = torch.zeros((1, 1, 8, 128), device=device, dtype=torch.bfloat16)
    mean_k[0, 0, :, 0] = torch.tensor(
        [0.0, -0.2, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6], device=device
    )
    masses = torch.ones((1, 1, 8, 1), device=device)
    scores = mean_k[0, 0, :, 0].float()
    remote_lse = torch.logsumexp(scores, 0).expand(1, 2, 2).contiguous()
    local_lse = torch.full((1, 2, 2), -float("inf"), device=device)
    sink_k = torch.zeros((1, 1, 1, 128), device=device, dtype=torch.bfloat16)

    def run(local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = torch.arange(8, device=device).view(1, 1, 1, 8).repeat(1, 2, 2, 1)
        head_counts = torch.zeros(16, device=device, dtype=torch.int32)
        route_offsets = torch.empty(slots.shape, device=device, dtype=torch.int32)
        _select_routes_by_mass_coverage_kernel[(2, 1)](
            q,
            mean_k,
            masses,
            remote_lse,
            remote_lse,
            local,
            sink_k,
            slots,
            head_counts,
            route_offsets,
            2,
            8,
            8,
            LOCAL_LSE_BATCH_STRIDE=local.stride(0),
            LOCAL_LSE_HEAD_STRIDE=local.stride(1),
            LOCAL_LSE_TOKEN_STRIDE=local.stride(2),
            SINK_K_BATCH_STRIDE=sink_k.stride(0),
            SINK_K_HEAD_STRIDE=sink_k.stride(1),
            SINK_K_TOKEN_STRIDE=sink_k.stride(2),
            QUERY_HEADS=2,
            KV_HEADS=1,
            KV_GROUP_SIZE=2,
            HEAD_DIM=128,
            ROUTE_COUNT=8,
            SINK_LEN=1,
            HAS_SECOND_PARTITION=False,
            SCALE=1.0,
            COVERAGE=0.75,
            BLOCK_M=16,
            BLOCK_D=128,
            num_warps=4,
        )
        _, q_lengths, _, _, _, _ = _pack_expert_routes(
            slots,
            active_slots=8,
            kv_heads=1,
            kv_group_size=2,
            expert_count=8,
            block_m=32,
            head_counts=head_counts,
            route_offsets=route_offsets,
        )
        return slots, q_lengths

    slots, q_lengths = run(local_lse)
    exact_base = 1.0  # sink, with no local mass
    target_mass = 0.75 * (1.0 + float(scores.exp().sum()))
    expected_open = 0
    while exact_base < target_mass:
        exact_base += math.exp(float(scores[expected_open]))
        expected_open += 1
    assert slots[0, 0, 0].tolist() == list(range(expected_open)) + [-1] * (8 - expected_open)
    assert int(q_lengths.sum()) == 4 * expected_open

    local_lse.fill_(math.log(100.0))
    slots, q_lengths = run(local_lse)
    assert slots.eq(-1).all()
    assert int(q_lengths.sum()) == 0
