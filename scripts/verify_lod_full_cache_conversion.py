#!/usr/bin/env python3
"""Parity and semantic-page checks for BF16 full-cache conversion."""

from __future__ import annotations

import argparse
import os
import sys
from types import MethodType

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.hf_pytorch_lod_attention import (
    convert_hf_full_cache_to_lod,
    install_hf_lod_attention,
)
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import KernelRecursivePagedLODAttention


def _config(kv_bits: int) -> PagedLODConfig:
    return PagedLODConfig(
        chunk_size=16,
        local_window=32,
        state_growth_factor=4.0,
        state_min_size=16,
        protected_prefix=1,
        max_routes=2,
        page_size=16,
        kv_bits=kv_bits,
        quant_group_size=32,
    )


def _engine(config: PagedLODConfig) -> KernelRecursivePagedLODAttention:
    return KernelRecursivePagedLODAttention(
        config,
        query_heads=2,
        key_value_heads=1,
        scale=32**-0.5,
        default_open_count=2,
    ).cuda()


def _fail_if_called(name: str):
    def fail(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError(f"cache conversion unexpectedly called {name}")

    return fail


def _slot_positions(page_cache: dict[str, object], slot: int) -> torch.Tensor:
    slot_lengths = page_cache["slot_lengths"]
    slot_pages = page_cache["slot_pages"]
    page_indices = page_cache["page_indices"]
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (slot_lengths, slot_pages, page_indices)
    ):
        raise TypeError("semantic virtual-page metadata is incomplete")
    length = int(slot_lengths[0, 0, slot].item())
    page_count = (length + 15) // 16
    if page_count > int(slot_pages.size(-1)):
        raise AssertionError("test unexpectedly reached overflow page postings")
    pages = slot_pages[0, 0, slot, :page_count].long()
    positions = page_indices[0, 0, pages].flatten()
    return positions[positions.ge(0)].sort().values


@torch.inference_mode()
def check_interleaved_region_pages(config: PagedLODConfig) -> None:
    engine = _engine(config)
    key = torch.randn(1, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    owners = torch.arange(32, device="cuda").remainder(2).view(1, 1, 32)
    cache = engine._new_page_cache(
        key,
        value,
        owners,
        state_capacity=2,
        sequence_capacity=32,
        virtual_k=key,
        virtual_v=value,
    )
    if not bool(cache.get("region_owned_pages", False)):
        raise AssertionError("page cache did not declare semantic ownership")
    slot_zero = _slot_positions(cache, 0)
    slot_one = _slot_positions(cache, 1)
    torch.testing.assert_close(
        slot_zero,
        torch.arange(0, 32, 2, device="cuda", dtype=slot_zero.dtype),
    )
    torch.testing.assert_close(
        slot_one,
        torch.arange(1, 32, 2, device="cuda", dtype=slot_one.dtype),
    )
    first_page = cache["page_indices"][0, 0, 0]
    if torch.equal(first_page, torch.arange(16, device="cuda")):
        raise AssertionError("semantic page was replaced by a chronological block")


@torch.inference_mode()
def check_conversion_parity(config: PagedLODConfig) -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 2, 384, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 1, 384, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)

    direct_engine = _engine(config)
    _, direct_cache = direct_engine(query, key, value, use_cache=True)
    if direct_cache is None:
        raise AssertionError("ordinary LOD prefill did not return a cache")

    converted_engine = _engine(config)
    converted_engine._exact_attention = MethodType(
        _fail_if_called("exact attention"), converted_engine
    )
    converted_engine._two_level_attention = MethodType(
        _fail_if_called("two-level attention"), converted_engine
    )
    converted_cache = converted_engine.build_cache_from_bf16(key, value)
    del converted_engine._exact_attention
    del converted_engine._two_level_attention

    direct = direct_cache.state
    converted = converted_cache.state
    for name in ("state_len", "coverage", "recent_len", "total_len"):
        if int(direct[name]) != int(converted[name]):
            raise AssertionError(f"converted {name} differs from ordinary prefill")
    state_len = int(direct["state_len"])
    recent_len = int(direct["recent_len"])
    for name in ("state_k", "state_v", "counts"):
        torch.testing.assert_close(
            direct[name][..., :state_len, :],
            converted[name][..., :state_len, :],
            atol=0,
            rtol=0,
        )
    for name in ("recent_k", "recent_v"):
        torch.testing.assert_close(
            direct[name][..., :recent_len, :],
            converted[name][..., :recent_len, :],
            atol=0,
            rtol=0,
        )

    direct_pages = direct["page_cache"]
    converted_pages = converted["page_cache"]
    if not bool(converted_pages.get("region_owned_pages", False)):
        raise AssertionError("converted page cache is not region-owned")
    torch.testing.assert_close(
        direct_pages["slot_lengths"], converted_pages["slot_lengths"]
    )
    for slot in range(state_len):
        torch.testing.assert_close(
            _slot_positions(direct_pages, slot),
            _slot_positions(converted_pages, slot),
        )

    next_query = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.bfloat16)
    next_key = torch.randn(1, 1, 1, 32, device="cuda", dtype=torch.bfloat16)
    next_value = torch.randn_like(next_key)
    direct_output, _ = direct_engine(
        next_query, next_key, next_value, cache=direct_cache, use_cache=True
    )
    converted_output, _ = converted_engine(
        next_query, next_key, next_value, cache=converted_cache, use_cache=True
    )
    torch.testing.assert_close(
        direct_output.float(), converted_output.float(), atol=2e-2, rtol=2e-2
    )


@torch.inference_mode()
def check_missing_query_rejected() -> None:
    config = PagedLODConfig(
        chunk_size=16,
        local_window=32,
        state_min_size=16,
        page_size=16,
        state_clustering_query_metric="diagonal",
    )
    engine = _engine(config)
    key = torch.empty(1, 1, 32, 32, device="cuda", dtype=torch.bfloat16)
    try:
        engine.build_cache_from_bf16(key, key)
    except ValueError as error:
        if "prefill queries" not in str(error):
            raise
    else:
        raise AssertionError("query-dependent conversion accepted missing queries")


@torch.inference_mode()
def check_hf_cache_bridge(config: PagedLODConfig) -> None:
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                max_position_embeddings=512,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    input_ids = torch.randint(0, 64, (1, 384), device="cuda")
    native = model(input_ids=input_ids, use_cache=True)
    source_layer = native.past_key_values.layers[0]
    source_key = source_layer.keys.clone()
    source_value = source_layer.values.clone()
    install_hf_lod_attention(
        model,
        config=config,
        open_count=2,
        engine_backend="kernel",
    )
    converted = convert_hf_full_cache_to_lod(model, native.past_key_values)
    torch.testing.assert_close(source_layer.keys, source_key, atol=0, rtol=0)
    torch.testing.assert_close(source_layer.values, source_value, atol=0, rtol=0)
    if converted.get_seq_length() != 384:
        raise AssertionError("HF conversion lost the logical prefix length")
    continuation = model(
        input_ids=torch.randint(0, 64, (1, 1), device="cuda"),
        past_key_values=converted,
        use_cache=True,
    )
    if not torch.isfinite(continuation.logits).all():
        raise AssertionError("HF decode from a converted cache is not finite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this check requires a CUDA or ROCm GPU")
    config = _config(args.kv_bits)
    check_interleaved_region_pages(config)
    check_conversion_parity(config)
    check_missing_query_rejected()
    check_hf_cache_bridge(config)
    print(f"BF16-to-LOD conversion checks passed (kv_bits={args.kv_bits})")


if __name__ == "__main__":
    main()
