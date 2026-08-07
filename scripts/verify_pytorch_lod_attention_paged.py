#!/usr/bin/env python3
"""Focused checks for paged and INT4 pure-PyTorch LOD attention."""

from __future__ import annotations

import argparse

import torch

from model.pytorch_lod_attention import LODConfig, TwoLevelLODAttention
from model.pytorch_lod_attention_paged import (
    PagedLODConfig,
    PagedTensor,
    PagedTwoLevelLODAttention,
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


def check_int4_storage(device: torch.device) -> None:
    torch.manual_seed(20)
    original = torch.randn(
        1, 2, 64, 128, device=device, dtype=torch.bfloat16
    )
    paged = PagedTensor.from_tensor(
        original,
        page_size=16,
        bits=4,
        group_size=32,
        dtype=torch.bfloat16,
    )
    restored = paged.materialize()
    error = (restored.float() - original.float()).abs()
    assert float(error.mean()) < 0.12
    assert paged.storage_bytes / (original.numel() * original.element_size()) < 0.33

    position = torch.randint(0, 64, (1, 8, 2, 7), device=device)
    gathered = paged.gather(position, query_heads=8)
    expected = restored.repeat_interleave(4, dim=1).gather(
        2,
        position.flatten(2).unsqueeze(-1).expand(-1, -1, -1, 128),
    ).reshape(*position.shape, 128)
    _assert_close(gathered, expected)

    addition = torch.randn(
        1, 2, 17, 128, device=device, dtype=torch.bfloat16
    )
    appended = paged.append(addition)
    assert appended.length == 81
    assert appended.complete_pages == 5
    assert int(appended.tail.size(2)) == 1
    _assert_close(appended.materialize()[..., -1:, :], addition[..., -1:, :])


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
    tolerance = 0.0 if device.type == "cpu" else 2e-2
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

    int4 = PagedTwoLevelLODAttention(
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
    with torch.inference_mode():
        int4_output, int4_cache = int4(query, key, value, use_cache=True)
    assert torch.isfinite(int4_output).all()
    mean_error = (int4_output.float() - page_output.float()).abs().mean()
    assert float(mean_error) < 0.12
    assert int4_cache.leaves.storage_bytes < page_cache.leaves.storage_bytes


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
    check_int4_storage(device)
    check_module(device)
    print(f"paged LOD verification passed on {device}")


if __name__ == "__main__":
    main()
