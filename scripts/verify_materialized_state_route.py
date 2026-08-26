#!/usr/bin/env python3
"""Check the separated centroid route against explicit tensor references."""

from __future__ import annotations

import json

import torch

from model.kernels.paged_leaf_attention import (
    materialized_state_route_gqa,
    new_fused_decode_buffers,
)


GEOMETRIES = (
    ("muse", 128, 2, 16),
    ("olmo", 128, 8, 5),
    ("phi", 128, 2, 4),
    ("qwen", 256, 4, 6),
    ("gemma", 512, 2, 8),
)


def verify(name: str, head_dim: int, kv_heads: int, gqa: int) -> dict[str, object]:
    device = torch.device("cuda")
    batch, cache_batch = 3, 4
    state_len, state_capacity = 243, 256
    query_heads = kv_heads * gqa
    generator = torch.Generator(device=device)
    generator.manual_seed(20260820 + head_dim + kv_heads)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    q = randn((batch, query_heads, 1, head_dim))
    state_k = randn((cache_batch, kv_heads, state_capacity, head_dim))
    state_v = randn((cache_batch, kv_heads, state_capacity, head_dim))
    counts = torch.randint(
        1,
        49,
        (cache_batch, kv_heads, state_capacity, 1),
        dtype=torch.int32,
        device=device,
        generator=generator,
    ).float()
    counts[..., state_len - 13 :, :].zero_()
    # The first selected cache/head pair has an entirely empty second score
    # tile. This exercises the fused tile-LSE no-mass path, not only a short
    # underfull tail.
    counts[3, 0, 128:state_len, :].zero_()
    cache_indices = torch.tensor((3, 0, 2), dtype=torch.long, device=device)
    buffers = new_fused_decode_buffers(
        q,
        splits=8,
        state_capacity=state_capacity,
        route_group_size=32,
        materialized_state_route=True,
    )
    materialized_state_route_gqa(
        q,
        state_k,
        state_v,
        counts,
        cache_indices,
        buffers,
        state_len=state_len,
        kv_group_size=gqa,
        scale=head_dim**-0.5,
        protected_len=3,
        max_leaf_tokens=40,
    )
    torch.cuda.synchronize()

    selected_counts = counts.index_select(0, cache_indices)[..., :state_len, 0]
    selected_k = state_k.index_select(0, cache_indices)[..., :state_len, :]
    grouped_q = q[..., 0, :].reshape(batch, kv_heads, gqa, head_dim)
    dots = torch.matmul(
        grouped_q.float(), selected_k.float().transpose(-1, -2)
    )
    safe_counts = selected_counts.clamp_min(1.0)
    expected_scores = (
        dots * (head_dim**-0.5) / safe_counts[:, :, None, :]
        + safe_counts.log()[:, :, None, :]
    )
    valid = selected_counts > 0.0
    expected_scores = torch.where(
        valid[:, :, None, :],
        expected_scores,
        torch.full_like(expected_scores, -float("inf")),
    ).to(buffers["route_state_scores"].dtype)
    actual_scores = buffers["route_state_scores"][..., 0, :state_len].reshape(
        batch, kv_heads, gqa, state_len
    )
    finite = torch.isfinite(expected_scores)
    score_max_abs = float(
        (actual_scores.float() - expected_scores.float())[finite].abs().max().item()
    )

    slot = torch.arange(state_len, device=device)
    route_valid = valid & (slot[None, None, :] >= 3) & (selected_counts < 40)
    route_scores = torch.where(
        route_valid[:, :, None, :],
        actual_scores,
        torch.full_like(actual_scores, -float("inf")),
    )
    expected_top = route_scores.topk(8, dim=-1, sorted=False).indices.reshape(
        batch, query_heads, 8
    )
    actual_top = buffers["route_top_slots"][..., 0, :]
    top8_exact = bool(
        (
            expected_top.sort(dim=-1).values
            == actual_top.sort(dim=-1).values
        )
        .all()
        .item()
    )

    score_rows = actual_scores.float().reshape(batch, query_heads, state_len)
    expected_lse = torch.logsumexp(score_rows, dim=-1)
    lse_max_abs = float(
        (buffers["coarse_lse"] - expected_lse).abs().max().item()
    )
    query_counts = safe_counts[:, :, None, :].expand(-1, -1, gqa, -1)
    scaled_probability = (
        torch.exp(
            score_rows.reshape(batch, kv_heads, gqa, state_len)
            - expected_lse.reshape(batch, kv_heads, gqa, 1)
        )
        / query_counts
    ).to(buffers["route_state_probabilities"].dtype)
    expected_out = torch.bmm(
        scaled_probability.reshape(batch * kv_heads, gqa, state_len),
        state_v.index_select(0, cache_indices)[..., :state_len, :].reshape(
            batch * kv_heads, state_len, head_dim
        ).to(scaled_probability.dtype),
        out_dtype=torch.float32,
    ).reshape(batch, query_heads, head_dim)
    output_max_abs = float(
        (buffers["coarse_out"] - expected_out).abs().max().item()
    )
    if score_max_abs > 2.0e-2:
        raise AssertionError(f"{name}: score materialization differs from reference")
    if not top8_exact:
        raise AssertionError(f"{name}: top-8 differs from materialized score table")
    if lse_max_abs > 2.0e-5:
        raise AssertionError(f"{name}: LSE reduction differs from reference")
    if output_max_abs > 2.0e-5:
        raise AssertionError(f"{name}: PV differs from reference")
    return {
        "geometry": name,
        "score_max_abs": score_max_abs,
        "top8_exact": top8_exact,
        "lse_max_abs": lse_max_abs,
        "output_max_abs": output_max_abs,
        "cache_indices": cache_indices.tolist(),
        "protected_len": 3,
        "max_leaf_tokens": 40,
    }


def main() -> None:
    results = [verify(*geometry) for geometry in GEOMETRIES]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
