#!/usr/bin/env python3
"""GPU parity checks for the fast pure-PyTorch LOD attention backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.pytorch_lod_attention import LODConfig, LODState, two_level_lod_attention
from model.pytorch_lod_attention_fast import (
    FastTwoLevelLODAttention,
    fast_coarse_lod_attention,
    fast_two_level_lod_attention,
)


def state_from_leaves(
    key: torch.Tensor,
    value: torch.Tensor,
    owner: torch.Tensor,
    slots: int,
) -> LODState:
    assignment = F.one_hot(owner, num_classes=slots).to(key.dtype).transpose(-1, -2)
    return LODState(
        key_sum=torch.matmul(assignment, key),
        value_sum=torch.matmul(assignment, value),
        count=assignment.float().sum(dim=-1),
    )


@torch.inference_mode()
def verify_low_level(device: torch.device) -> None:
    torch.manual_seed(50)
    batch, query_heads, key_value_heads = 1, 4, 2
    query_length, history_length, local_length = 5, 16, 8
    dimension, slots = 16, 4
    query = torch.randn(
        batch, query_heads, query_length, dimension, device=device
    ).bfloat16()
    leaf_key = torch.randn(
        batch, key_value_heads, history_length, dimension, device=device
    ).bfloat16()
    leaf_value = torch.randn_like(leaf_key)
    local_key = torch.randn(
        batch, key_value_heads, local_length, dimension, device=device
    ).bfloat16()
    local_value = torch.randn_like(local_key)
    owner = torch.arange(history_length, device=device).remainder(slots)
    owner = owner.view(1, 1, -1).expand(batch, key_value_heads, -1).clone()
    state = state_from_leaves(leaf_key, leaf_value, owner, slots)
    query_offset = local_length - query_length
    open_count = torch.tensor([0, 1, 2, 3, 4], device=device).view(1, 1, -1)

    expected_coarse = two_level_lod_attention(
        query,
        local_key,
        local_value,
        state,
        owner,
        leaf_key,
        leaf_value,
        max_routes=slots,
        open_count=0,
        query_offset=query_offset,
    )
    actual_coarse = fast_coarse_lod_attention(
        query,
        local_key,
        local_value,
        state,
        query_offset=query_offset,
    )
    torch.testing.assert_close(
        actual_coarse.output.float(),
        expected_coarse.output.float(),
        atol=3e-2,
        rtol=3e-2,
    )

    expected = two_level_lod_attention(
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
    actual = fast_two_level_lod_attention(
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
    torch.testing.assert_close(
        actual.output.float(), expected.output.float(), atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        actual.logsumexp.float(),
        expected.logsumexp.float(),
        atol=3e-2,
        rtol=3e-2,
    )


@torch.inference_mode()
def verify_module(device: torch.device) -> None:
    torch.manual_seed(60)
    config = LODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=4.0,
        state_min_size=4,
        protected_prefix=0,
        max_routes=4,
    )
    attention = FastTwoLevelLODAttention(config, default_open_count=2).to(device)
    query = torch.randn(1, 4, 18, 16, device=device).bfloat16()
    key = torch.randn(1, 2, 18, 16, device=device).bfloat16()
    value = torch.randn_like(key)
    full, _ = attention(query, key, value, open_count=2)
    prefix, cache = attention(
        query[..., :12, :],
        key[..., :12, :],
        value[..., :12, :],
        open_count=2,
        use_cache=True,
    )
    if cache is None:
        raise AssertionError("fast prefill did not return a cache")
    suffix, cache = attention(
        query[..., 12:, :],
        key[..., 12:, :],
        value[..., 12:, :],
        cache=cache,
        open_count=torch.tensor([[[2]]], device=device),
        use_cache=True,
    )
    torch.testing.assert_close(prefix.float(), full[..., :12, :].float())
    torch.testing.assert_close(suffix.float(), full[..., 12:, :].float())
    if cache is None or cache.coverage != 12:
        raise AssertionError("fast decode did not cross the compression boundary")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("fast LOD verification requires a CUDA or ROCm GPU")
    device = torch.device("cuda")
    verify_low_level(device)
    print("fast coarse/leaf/LSE/GQA/dynamic-open parity passed")
    verify_module(device)
    print("fast prefill/decode cache parity passed")


if __name__ == "__main__":
    main()
