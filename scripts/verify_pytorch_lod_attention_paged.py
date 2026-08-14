#!/usr/bin/env python3
"""Focused checks for paged pure-PyTorch LOD attention."""

from __future__ import annotations

import argparse

import torch

from model.pytorch_lod_attention import LODConfig, LODState, TwoLevelLODAttention
from model.pytorch_lod_attention_paged import (
    PagedKVCache,
    PagedLODConfig,
    PagedTensor,
    PagedTwoLevelLODAttention,
    build_region_pages,
    paged_two_level_lod_attention,
)


def _base_config(**extra) -> dict:
    config = dict(
        chunk_size=4,
        local_window=8,
        state_growth_factor=2.0,
        state_min_size=8,
        protected_prefix=1,
        max_routes=4,
    )
    config.update(extra)
    return config


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def check_page_storage(device: torch.device) -> None:
    torch.manual_seed(10)
    original = torch.randn(2, 2, 11, 32, device=device, dtype=torch.float32)
    paged = PagedTensor.from_tensor(
        original,
        page_size=4,
        bits=0,
        group_size=8,
        dtype=torch.float32,
    )
    _assert_close(paged.materialize(), original)

    position = torch.randint(0, 11, (2, 6, 3, 5), device=device)
    expected = original.repeat_interleave(3, dim=1).gather(
        2,
        position.flatten(2).unsqueeze(-1).expand(-1, -1, -1, 32),
    ).reshape(*position.shape, 32)
    _assert_close(paged.gather(position, query_heads=6), expected)

    addition = torch.randn(2, 2, 6, 32, device=device)
    paged = paged.append(addition)
    _assert_close(paged.materialize(), torch.cat((original, addition), dim=2))
    assert paged.complete_pages == 4
    assert int(paged.tail.size(2)) == 1


def check_chronological_int4_rejected(device: torch.device) -> None:
    torch.manual_seed(20)
    original = torch.randn(
        1, 2, 64, 128, device=device, dtype=torch.bfloat16
    )
    try:
        PagedTensor.from_tensor(
            original,
            page_size=16,
            bits=4,
            group_size=32,
            dtype=torch.bfloat16,
        )
    except NotImplementedError as error:
        if "region-owned INT4" not in str(error):
            raise
    else:
        raise AssertionError("pure-PyTorch chronological INT4 was accepted")


def check_recursive_page_semantics(device: torch.device) -> None:
    dtype = torch.float32
    key_x = torch.tensor(
        [-1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0], device=device
    )
    key = torch.stack((key_x, torch.zeros_like(key_x)), dim=-1).view(1, 1, 8, 2)
    value = torch.arange(16, device=device, dtype=dtype).view(1, 1, 8, 2)
    owner = torch.zeros(1, 1, 8, dtype=torch.long, device=device)
    state = LODState(
        key_sum=key.sum(dim=2, keepdim=True),
        value_sum=value.sum(dim=2, keepdim=True),
        count=torch.full((1, 1, 1), 8.0, device=device),
    )
    query = torch.tensor([[[[1.0, 0.0]]]], device=device)
    local_key = torch.tensor([[[[-2.0, 0.0]]]], device=device)
    local_value = torch.tensor([[[[-2.0, 3.0]]]], device=device)
    config = PagedLODConfig(page_size=4, leaf_dtype=dtype)
    leaves = PagedKVCache.from_tensors(key, value, config)
    pages = build_region_pages(owner, state, leaves, page_size=4)
    assert pages.slot_count.tolist() == [[[2]]]

    result = paged_two_level_lod_attention(
        query,
        local_key,
        local_value,
        state,
        owner,
        leaves,
        max_routes=1,
        open_count=1,
        route_protected_prefix=0,
        scale=1.0,
        query_offset=0,
        region_pages=pages,
    )
    # Page 1 wins. Page 0 is represented by its count-corrected mean, while
    # the local token remains exact in the outer coarse branch.
    expected_score = torch.cat(
        (
            key_x[4:],
            key_x[:4].mean().view(1) + torch.tensor(4.0, device=device).log(),
            torch.tensor([-2.0], device=device),
        )
    )
    expected_value = torch.cat(
        (value[0, 0, 4:], value[0, 0, :4].mean(0, keepdim=True), local_value[0, 0])
    )
    expected_probability = expected_score.softmax(0)
    expected_output = (expected_probability.unsqueeze(-1) * expected_value).sum(0)
    _assert_close(result.output[0, 0, 0], expected_output, atol=1e-6, rtol=1e-6)
    _assert_close(
        result.logsumexp[0, 0, 0],
        expected_score.logsumexp(0),
        atol=1e-6,
        rtol=1e-6,
    )

    full_score = torch.cat((key_x, torch.tensor([-2.0], device=device)))
    full_value = torch.cat((value[0, 0], local_value[0, 0]))
    full_output = (full_score.softmax(0).unsqueeze(-1) * full_value).sum(0)
    assert float((result.output[0, 0, 0] - full_output).abs().max()) > 1e-3

    # Exercise selection across more than one bounded 16-page scan block.
    long_x = torch.arange(20, device=device, dtype=dtype).repeat_interleave(4)
    long_key = torch.stack((long_x, torch.zeros_like(long_x)), -1).view(1, 1, 80, 2)
    long_value = torch.arange(160, device=device, dtype=dtype).view(1, 1, 80, 2)
    long_owner = torch.zeros(1, 1, 80, dtype=torch.long, device=device)
    long_state = LODState(
        key_sum=long_key.sum(2, keepdim=True),
        value_sum=long_value.sum(2, keepdim=True),
        count=torch.full((1, 1, 1), 80.0, device=device),
    )
    long_leaves = PagedKVCache.from_tensors(long_key, long_value, config)
    long_result = paged_two_level_lod_attention(
        query,
        local_key,
        local_value,
        long_state,
        long_owner,
        long_leaves,
        max_routes=1,
        open_count=1,
        route_protected_prefix=0,
        scale=1.0,
        query_offset=0,
    )
    long_score = torch.cat(
        (
            long_x[-4:],
            long_x[:-4].mean().view(1) + torch.tensor(76.0, device=device).log(),
            torch.tensor([-2.0], device=device),
        )
    )
    long_expected_value = torch.cat(
        (
            long_value[0, 0, -4:],
            long_value[0, 0, :-4].mean(0, keepdim=True),
            local_value[0, 0],
        )
    )
    long_expected = (long_score.softmax(0).unsqueeze(-1) * long_expected_value).sum(0)
    _assert_close(
        long_result.output[0, 0, 0], long_expected, atol=2e-5, rtol=2e-5
    )


def check_protected_sink_routing(device: torch.device) -> None:
    dtype = torch.float32
    key_x = torch.tensor([10.0, 3.0, 1.0, 0.0], device=device)
    key = torch.stack((key_x, torch.zeros_like(key_x)), dim=-1).view(1, 1, 4, 2)
    value_x = torch.tensor([10.0, 2.0, 4.0, -1.0], device=device)
    value = torch.stack((value_x, torch.zeros_like(value_x)), dim=-1).view(
        1, 1, 4, 2
    )
    owner = torch.tensor([[[0, 1, 1, 2]]], device=device)
    state = LODState(
        key_sum=torch.tensor(
            [[[[10.0, 0.0], [4.0, 0.0], [0.0, 0.0]]]], device=device
        ),
        value_sum=torch.tensor(
            [[[[10.0, 0.0], [6.0, 0.0], [-1.0, 0.0]]]], device=device
        ),
        count=torch.tensor([[[1.0, 2.0, 1.0]]], device=device),
    )
    query = torch.tensor([[[[1.0, 0.0]]]], device=device)
    local_key = torch.tensor([[[[-5.0, 0.0]]]], device=device)
    local_value = torch.tensor([[[[7.0, 0.0]]]], device=device)
    config = PagedLODConfig(page_size=2, leaf_dtype=dtype)
    leaves = PagedKVCache.from_tensors(key, value, config)
    result = paged_two_level_lod_attention(
        query,
        local_key,
        local_value,
        state,
        owner,
        leaves,
        max_routes=1,
        open_count=1,
        scale=1.0,
        query_offset=0,
    )
    if result.top_slots is None or result.top_slots.item() != 1:
        raise AssertionError("paged routing opened a protected sink slot")
    scores = torch.tensor([10.0, 3.0, 1.0, 0.0, -5.0], device=device)
    values = torch.tensor(
        [[10.0, 0.0], [2.0, 0.0], [4.0, 0.0], [-1.0, 0.0], [7.0, 0.0]],
        device=device,
    )
    expected = (scores.softmax(0).unsqueeze(-1) * values).sum(0)
    _assert_close(result.output[0, 0, 0], expected, atol=1e-6, rtol=1e-6)
    _assert_close(
        result.logsumexp[0, 0, 0], scores.logsumexp(0), atol=1e-6, rtol=1e-6
    )


def check_module(device: torch.device) -> None:
    torch.manual_seed(30)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    shape = (1, 4, 25, 32)
    query = torch.randn(*shape, device=device, dtype=dtype)
    key = torch.randn(1, 2, 25, 32, device=device, dtype=dtype)
    value = torch.randn_like(key)

    flat = TwoLevelLODAttention(
        LODConfig(**_base_config(leaf_dtype=dtype)), default_open_count=4
    )
    paged = PagedTwoLevelLODAttention(
        PagedLODConfig(
            **_base_config(
                leaf_dtype=dtype,
                page_size=4,
                kv_bits=0,
                quant_group_size=8,
            )
        ),
        default_open_count=4,
    )
    no_pages = PagedTwoLevelLODAttention(
        PagedLODConfig(
            **_base_config(leaf_dtype=dtype, page_size=None, kv_bits=0)
        ),
        default_open_count=4,
    )
    with torch.inference_mode():
        flat_output, flat_cache = flat(query, key, value, use_cache=True)
        page_output, page_cache = paged(query, key, value, use_cache=True)
        no_page_output, no_page_cache = no_pages(
            query, key, value, use_cache=True
        )
    tolerance = 1e-5 if device.type == "cpu" else 2e-2
    _assert_close(page_output, flat_output, atol=tolerance, rtol=tolerance)
    _assert_close(no_page_output, flat_output, atol=tolerance, rtol=tolerance)

    next_query = torch.randn(1, 4, 1, 32, device=device, dtype=dtype)
    next_key = torch.randn(1, 2, 1, 32, device=device, dtype=dtype)
    next_value = torch.randn_like(next_key)
    with torch.inference_mode():
        flat_decode, _ = flat(
            next_query,
            next_key,
            next_value,
            cache=flat_cache,
            use_cache=True,
        )
        page_decode, next_page_cache = paged(
            next_query,
            next_key,
            next_value,
            cache=page_cache,
            use_cache=True,
        )
        no_page_decode, _ = no_pages(
            next_query,
            next_key,
            next_value,
            cache=no_page_cache,
            use_cache=True,
        )
    _assert_close(page_decode, flat_decode, atol=tolerance, rtol=tolerance)
    _assert_close(no_page_decode, flat_decode, atol=tolerance, rtol=tolerance)
    assert next_page_cache.leaves.length == 26

    try:
        PagedTwoLevelLODAttention(
            PagedLODConfig(
                **_base_config(
                    leaf_dtype=torch.bfloat16,
                    page_size=4,
                    kv_bits=4,
                    quant_group_size=8,
                )
            ),
            default_open_count=4,
        )
    except NotImplementedError:
        pass
    else:
        raise AssertionError("pure-PyTorch INT4 attention was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    check_page_storage(device)
    check_chronological_int4_rejected(device)
    check_recursive_page_semantics(device)
    check_protected_sink_routing(device)
    check_module(device)
    print(f"paged LOD verification passed on {device}")


if __name__ == "__main__":
    main()
