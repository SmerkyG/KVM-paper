#!/usr/bin/env python3
"""CPU verification for the model-agnostic pure-PyTorch LOD attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from model.pytorch_lod_attention import (
    CoarseLODAttention,
    LODConfig,
    LODState,
    TwoLevelLODAttention,
    coarse_lod_attention,
    two_level_lod_attention,
)


def repeat_kv(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    return tensor.repeat_interleave(query_heads // int(tensor.size(1)), dim=1)


def attention_from_scores(
    scores: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = torch.softmax(scores.float(), dim=-1)
    return torch.matmul(probability.to(value.dtype), value), torch.logsumexp(
        scores.float(), dim=-1
    )


def dense_causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    query_heads = int(query.size(1))
    key = repeat_kv(key, query_heads)
    value = repeat_kv(value, query_heads)
    scale = 1.0 / math.sqrt(float(query.size(-1)))
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    query_index = torch.arange(int(query.size(2))).unsqueeze(-1)
    key_index = torch.arange(int(key.size(2))).unsqueeze(0)
    scores = scores.masked_fill(
        key_index > query_index,
        -torch.inf,
    )
    return attention_from_scores(scores, value)[0]


def state_from_leaves(
    key: torch.Tensor,
    value: torch.Tensor,
    owner: torch.Tensor,
    slots: int,
) -> LODState:
    assignment = F.one_hot(owner, num_classes=slots).float().transpose(-1, -2)
    return LODState(
        key_sum=torch.matmul(assignment.to(key.dtype), key),
        value_sum=torch.matmul(assignment.to(value.dtype), value),
        count=assignment.sum(dim=-1),
    )


def local_scores(
    query: torch.Tensor, key: torch.Tensor, query_offset: int
) -> torch.Tensor:
    key = repeat_kv(key, int(query.size(1)))
    scale = 1.0 / math.sqrt(float(query.size(-1)))
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
    query_index = torch.arange(int(query.size(2))).unsqueeze(-1)
    key_index = torch.arange(int(key.size(2))).unsqueeze(0)
    return scores.masked_fill(key_index > query_index + query_offset, -torch.inf)


def verify_low_level_lse_math() -> None:
    torch.manual_seed(10)
    batch, query_heads, kv_heads = 2, 4, 2
    query_length, history_length, local_length = 3, 7, 4
    key_dim, value_dim, slots = 5, 6, 4
    query = torch.randn(batch, query_heads, query_length, key_dim)
    leaf_key = torch.randn(batch, kv_heads, history_length, key_dim)
    leaf_value = torch.randn(batch, kv_heads, history_length, value_dim)
    local_key = torch.randn(batch, kv_heads, local_length, key_dim)
    local_value = torch.randn(batch, kv_heads, local_length, value_dim)
    owner_pattern = torch.tensor([0, 0, 1, 2, 2, 3, 3])
    owner = owner_pattern.view(1, 1, -1).expand(batch, kv_heads, -1).clone()
    state = state_from_leaves(leaf_key, leaf_value, owner, slots)
    scale = 1.0 / math.sqrt(float(key_dim))
    query_offset = local_length - query_length

    state_key = repeat_kv(state.mean_key, query_heads)
    state_value = repeat_kv(state.mean_value, query_heads)
    state_count = repeat_kv(state.count, query_heads)
    state_score = (
        torch.matmul(query.float(), state_key.float().transpose(-1, -2)) * scale
        + state_count.log().unsqueeze(2)
    )
    exact_local_score = local_scores(query, local_key, query_offset)
    exact_local_value = repeat_kv(local_value, query_heads)
    coarse_score = torch.cat((state_score, exact_local_score), dim=-1)
    coarse_value = torch.cat((state_value, exact_local_value), dim=2)
    expected_output, expected_lse = attention_from_scores(
        coarse_score, coarse_value
    )
    actual = coarse_lod_attention(
        query,
        local_key,
        local_value,
        state,
        query_offset=query_offset,
    )
    torch.testing.assert_close(actual.output, expected_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        actual.logsumexp, expected_lse, atol=1e-6, rtol=1e-6
    )

    full = two_level_lod_attention(
        query,
        local_key,
        local_value,
        state,
        owner,
        leaf_key,
        leaf_value,
        max_routes=slots,
        open_count=slots,
        query_offset=query_offset,
    )
    repeated_leaf_key = repeat_kv(leaf_key, query_heads)
    repeated_leaf_value = repeat_kv(leaf_value, query_heads)
    exact_leaf_score = (
        torch.matmul(
            query.float(), repeated_leaf_key.float().transpose(-1, -2)
        )
        * scale
    )
    expected_score = torch.cat((exact_leaf_score, exact_local_score), dim=-1)
    expected_value = torch.cat((repeated_leaf_value, exact_local_value), dim=2)
    expected_output, expected_lse = attention_from_scores(
        expected_score, expected_value
    )
    torch.testing.assert_close(full.output, expected_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        full.logsumexp, expected_lse, atol=1e-6, rtol=1e-6
    )

    open_count = torch.tensor(
        [
            [[[0], [1], [2]], [[3], [2], [1]]],
            [[[2], [1], [0]], [[1], [2], [3]]],
        ]
    ).reshape(batch, kv_heads, query_length)
    open_count = open_count.repeat_interleave(query_heads // kv_heads, dim=1)
    partial = two_level_lod_attention(
        query,
        local_key,
        local_value,
        state,
        owner,
        leaf_key,
        leaf_value,
        max_routes=slots,
        open_count=open_count,
        query_offset=query_offset,
    )
    if partial.top_slots is None or partial.open_mask is None:
        raise AssertionError("two-level attention did not return its routes")
    selected_state = torch.zeros_like(state_score, dtype=torch.bool)
    for route_index in range(slots):
        selected_state.scatter_(
            -1,
            partial.top_slots[..., route_index : route_index + 1],
            partial.open_mask[..., route_index : route_index + 1],
        )
    repeated_owner = repeat_kv(owner, query_heads)
    selected_leaf = selected_state.gather(
        -1,
        repeated_owner.unsqueeze(2).expand(
            batch, query_heads, query_length, history_length
        ),
    )
    effective_score = torch.cat(
        (
            state_score.masked_fill(selected_state, -torch.inf),
            exact_leaf_score.masked_fill(~selected_leaf, -torch.inf),
            exact_local_score,
        ),
        dim=-1,
    )
    effective_value = torch.cat(
        (state_value, repeated_leaf_value, exact_local_value), dim=2
    )
    expected_output, expected_lse = attention_from_scores(
        effective_score, effective_value
    )
    torch.testing.assert_close(
        partial.output, expected_output, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        partial.logsumexp, expected_lse, atol=1e-6, rtol=1e-6
    )


def verify_dense_exact_limit() -> None:
    torch.manual_seed(20)
    config = LODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=100.0,
        state_min_size=0,
        protected_prefix=0,
        max_routes=8,
        leaf_dtype=torch.float32,
    )
    attention = TwoLevelLODAttention(config, default_open_count=8)
    query = torch.randn(1, 4, 16, 4)
    key = torch.randn(1, 2, 16, 4)
    value = torch.randn(1, 2, 16, 4)
    actual, cache = attention(query, key, value, use_cache=True)
    expected = dense_causal_attention(query, key, value)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    if cache is None or cache.owner is None:
        raise AssertionError("two-level prefill did not return its leaf cache")
    if int(cache.owner.size(2)) != cache.coverage:
        raise AssertionError("owner archive and state coverage disagree")


def verify_prefill_decode_equivalence() -> None:
    torch.manual_seed(30)
    config = LODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=2.0,
        state_min_size=4,
        protected_prefix=0,
        max_routes=4,
        leaf_dtype=torch.float32,
    )
    query = torch.randn(1, 4, 18, 4)
    key = torch.randn(1, 2, 18, 4)
    value = torch.randn(1, 2, 18, 3)

    full_attention = TwoLevelLODAttention(config, default_open_count=2)
    full, _ = full_attention(query, key, value, open_count=2)

    cached_attention = TwoLevelLODAttention(config, default_open_count=2)
    prefix, cache = cached_attention(
        query[..., :12, :],
        key[..., :12, :],
        value[..., :12, :],
        open_count=2,
        use_cache=True,
    )
    if cache is None:
        raise AssertionError("prefill did not return a cache")
    suffix, cache = cached_attention(
        query[..., 12:, :],
        key[..., 12:, :],
        value[..., 12:, :],
        cache=cache,
        open_count=torch.tensor([[[2]]]),
        use_cache=True,
    )
    torch.testing.assert_close(prefix, full[..., :12, :], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(suffix, full[..., 12:, :], atol=1e-6, rtol=1e-6)
    if cache is None or cache.total_length != 18 or cache.coverage != 12:
        raise AssertionError("decode cache length was not updated")

    coarse = CoarseLODAttention(config)
    coarse_prefix, coarse_cache = coarse(
        query[..., :12, :],
        key[..., :12, :],
        value[..., :12, :],
        use_cache=True,
    )
    if coarse_cache is None:
        raise AssertionError("coarse prefill did not return a cache")
    if (
        coarse_cache.owner is not None
        or coarse_cache.leaf_key is not None
        or coarse_cache.leaf_value is not None
    ):
        raise AssertionError("coarse-only cache retained exact leaves")
    coarse_suffix, _ = coarse(
        query[..., 12:, :],
        key[..., 12:, :],
        value[..., 12:, :],
        cache=coarse_cache,
    )
    if coarse_prefix.shape != (1, 4, 12, 3) or coarse_suffix.shape != (1, 4, 6, 3):
        raise AssertionError("coarse attention returned an unexpected shape")


def verify_default_leaf_storage() -> None:
    torch.manual_seed(40)
    config = LODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=2.0,
        state_min_size=4,
        protected_prefix=0,
        max_routes=2,
    )
    attention = TwoLevelLODAttention(config, default_open_count=1)
    query = torch.randn(1, 2, 12, 4)
    key = torch.randn(1, 1, 12, 4)
    value = torch.randn(1, 1, 12, 3)
    output, cache = attention(query, key, value, use_cache=True)
    if output.shape != (1, 2, 12, 3) or cache is None:
        raise AssertionError("default two-level attention returned invalid shapes")
    if cache.leaf_key is None or cache.leaf_value is None:
        raise AssertionError("two-level cache did not retain exact leaves")
    if cache.leaf_key.dtype != torch.bfloat16 or cache.leaf_value.dtype != torch.bfloat16:
        raise AssertionError("default exact leaves were not stored in BF16")


def main() -> None:
    verify_low_level_lse_math()
    print("low-level coarse/top-k/LSE/GQA parity passed")
    verify_dense_exact_limit()
    print("all-regions-open dense causal parity passed")
    verify_prefill_decode_equivalence()
    print("prefill/decode cache parity and no-leaf cache checks passed")
    verify_default_leaf_storage()
    print("default BF16 leaf storage check passed")


if __name__ == "__main__":
    main()
