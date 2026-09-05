#!/usr/bin/env python3
"""Verify that vLLM LOD has one locked production configuration."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "integrations" / "vllm_lod"), str(ROOT)]

from vllm_lod_plugin.config import (  # noqa: E402
    EXPERIMENTAL_PROFILE,
    PRODUCTION_PROFILE,
    VLLMLODSettings,
    validate_production_scheduler,
)
from vllm_lod_plugin.pool import (  # noqa: E402
    VLLMLayerLODPool,
    _production_geometry_overrides,
)


def verify_default_profile() -> None:
    with patch.dict(os.environ, {}, clear=True):
        settings = VLLMLODSettings.from_environment()
    assert settings.profile == PRODUCTION_PROFILE
    assert settings.levels == 2
    assert settings.kv_bits == 0
    assert settings.routing_geometry == "auto"
    assert settings.open_count == 8
    assert settings.prefill_open_count is None
    assert settings.prefill_mode == "direct"
    assert settings.prefill_local_backend == "aiter"
    assert settings.decode_gqa_union
    assert settings.decode_gqa_union_hip
    assert settings.decode_gqa_fixed_mask_aiter
    assert settings.decode_gqa_fixed_mask_adaptive_segments


def verify_operational_settings() -> None:
    environment = {
        "VLLM_LOD_POOL_SIZE": "3",
        "VLLM_LOD_MAX_CONTEXT": "65536",
        "VLLM_LOD_WEIGHT_CACHE_BACKING": "1",
        "VLLM_LOD_PANEL_BATCH_SIZE": "8",
    }
    with patch.dict(os.environ, environment, clear=True):
        settings = VLLMLODSettings.from_environment()
    assert settings.pool_size == 3
    assert settings.request_capacity == 65_536


def verify_tuning_requires_explicit_override() -> None:
    with patch.dict(
        os.environ,
        {"VLLM_LOD_PREFILL_CHUNK_SIZE": "4096"},
        clear=True,
    ):
        try:
            VLLMLODSettings.from_environment()
        except ValueError as exc:
            assert "VLLM_LOD_PROFILE=experimental" in str(exc)
            assert "VLLM_LOD_PREFILL_CHUNK_SIZE" in str(exc)
        else:
            raise AssertionError("production tuning did not fail closed")

    with patch.dict(
        os.environ,
        {
            "VLLM_LOD_PROFILE": EXPERIMENTAL_PROFILE,
            "VLLM_LOD_PREFILL_CHUNK_SIZE": "8192",
            "VLLM_LOD_PREFILL_LOCAL_WINDOW": "8448",
        },
        clear=True,
    ):
        settings = VLLMLODSettings.from_environment()
    assert settings.profile == EXPERIMENTAL_PROFILE
    assert settings.prefill_chunk_size == 8192
    assert settings.prefill_local_window == 8448

    with patch.dict(
        os.environ,
        {
            "VLLM_LOD_PROFILE": EXPERIMENTAL_PROFILE,
            "VLLM_LOD_LEVELS": "3",
            "VLLM_LOD_KV_BITS": "4",
            "VLLM_LOD_RECURSIVE_PREFILL_ALL_LEAVES": "1",
        },
        clear=True,
    ):
        settings = VLLMLODSettings.from_environment()
    assert settings.recursive_prefill_all_leaves
    assert settings.quant_group_size == 4
    assert settings.leaf_quant_scale_mode == "l2"


def verify_model_geometries() -> None:
    qwen38 = _production_geometry_overrides(256, 6)
    assert qwen38["prefill_open_count"] == 3
    assert qwen38["prefill_chunk_size"] == 16_384
    assert qwen38["prefill_exact_first_chunk"] is True
    assert qwen38["prefill_overlap_coarse_leaf"] is True
    assert qwen38["decode_gqa_fixed_mask_aiter"] is True
    assert qwen38["decode_gqa_fixed_mask_segments"] == 256
    assert qwen38["decode_gqa_fixed_mask_reduce_block_d"] == 64

    qwen35 = _production_geometry_overrides(256, 4)
    assert qwen35["prefill_open_count"] == 3
    assert qwen35["prefill_chunk_size"] == 16_384
    assert qwen35["decode_gqa_cooperative"] is False
    assert qwen35["decode_gqa_cooperative_hip"] is False
    assert qwen35["decode_gqa_fixed_mask_aiter"] is False

    gemma = _production_geometry_overrides(512, 8)
    assert gemma["prefill_open_count"] == 3
    assert gemma["prefill_chunk_size"] == 4_096
    assert gemma["prefill_local_window"] == 4_864
    assert gemma["prefill_exact_first_chunk"] is False
    assert gemma["prefill_defer_cache_updates"] is False
    assert gemma["prefill_overlap_coarse_leaf"] is False
    assert gemma["decode_gqa_fixed_mask_segments"] == 128

    k2 = _production_geometry_overrides(128, 8)
    assert k2["prefill_open_count"] == 3
    assert k2["prefill_chunk_size"] == 16_384
    assert k2["prefill_exact_first_chunk"] is True
    assert k2["prefill_overlap_coarse_leaf"] is True
    assert k2["decode_gqa_cooperative"] is False
    assert k2["decode_gqa_cooperative_hip"] is False
    assert k2["decode_gqa_fixed_mask_aiter"] is False
    assert k2["decode_gqa_fixed_mask_segments"] == 128
    assert k2["decode_gqa_fixed_mask_scan_num_warps"] == 1


def verify_scheduler_guard() -> None:
    for max_batched, long_threshold in ((8192, 0), (16_384, 4096)):
        try:
            validate_production_scheduler(
                max_model_len=65_536,
                max_num_batched_tokens=max_batched,
                long_prefill_token_threshold=long_threshold,
                required_prefill=16_384,
            )
        except RuntimeError as exc:
            assert "Smaller scheduler slices" in str(exc)
        else:
            raise AssertionError("production accepted a short scheduler slice")

    validate_production_scheduler(
        max_model_len=65_536,
        max_num_batched_tokens=16_384,
        long_prefill_token_threshold=0,
        required_prefill=16_384,
    )
    validate_production_scheduler(
        max_model_len=65_536,
        max_num_batched_tokens=8_192,
        long_prefill_token_threshold=4_096,
        required_prefill=4_096,
    )


def verify_pool_startup_audit() -> None:
    geometries = (
        ("qwen35", 8, 2, 256, 3, 16_384, 256),
        ("qwen38", 24, 4, 256, 3, 16_384, 256),
        ("gemma", 16, 2, 512, 3, 4_096, 128),
        ("k2", 64, 8, 128, 3, 16_384, 128),
    )
    for name, query_heads, kv_heads, head_dim, topk, chunk, segments in geometries:
        normalized_keys = name in {"qwen35", "qwen38", "gemma"}
        layer = torch.nn.Module()
        layer.num_heads = query_heads
        layer.num_kv_heads = kv_heads
        layer.head_size = head_dim
        layer.head_size_v = head_dim
        layer.impl = SimpleNamespace(scale=head_dim**-0.5)
        pool = VLLMLayerLODPool(
            layer,
            settings=VLLMLODSettings.production(
                pool_size=1,
                request_capacity=512,
            ),
            max_requests=1,
            request_capacity=512,
            active_indices=torch.zeros(1, dtype=torch.int64),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            has_query_norm=normalized_keys,
            has_key_norm=normalized_keys,
        )
        assert pool.settings.prefill_open_count == topk
        assert pool.engine.prefill_chunk_len == chunk
        assert pool.settings.decode_gqa_fixed_mask_segments == segments
        if name in {"qwen35", "k2"}:
            assert pool.settings.decode_gqa_union
            assert pool.settings.decode_gqa_union_hip
            assert not pool.settings.decode_gqa_cooperative
            assert not pool.settings.decode_gqa_cooperative_hip
            assert not pool.settings.decode_gqa_fixed_mask_aiter
            assert pool.engine.decode_route_group_size == 64
            assert pool.engine.decode_route_segment_tiles == 1
            assert pool.engine.decode_route_num_warps == (
                1 if name == "k2" else 2
            )
            assert pool.engine.decode_route_reduce_num_warps == 2
        assert (
            pool.engine.state_clustering_normalization,
            pool.engine.state_clustering_centroid_rescale,
            pool.engine.routing_normalization,
        ) == (
            ("none", "coherence", "none")
            if normalized_keys
            else ("cosine", "none", "query")
        )


def verify_recursive_prefill_overrides() -> None:
    layer = torch.nn.Module()
    layer.num_heads = 64
    layer.num_kv_heads = 8
    layer.head_size = 128
    layer.head_size_v = 128
    layer.impl = SimpleNamespace(scale=128**-0.5)
    settings = VLLMLODSettings(
        levels=3,
        pool_size=1,
        request_capacity=512,
        prefill_chunk_size=128,
        prefill_local_window=384,
        prefill_state_update_size=128,
        prefill_open_count=4,
        kv_bits=4,
    )
    pool = VLLMLayerLODPool(
        layer,
        settings=settings,
        max_requests=1,
        request_capacity=512,
        active_indices=torch.zeros(1, dtype=torch.int64),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert pool.engine.prefill_chunk_len == 128
    assert pool.engine.prefill_local_len == 384
    assert pool.engine.prefill_state_update_len == 128
    assert pool.engine.prefill_two_level_topk == 4
    assert pool.engine.recursive_prefill_all_leaves
    assert pool.engine.recursive_prefill_all_leaves_token_limit == 0
    assert pool.engine.leaf_block_m == 64
    assert pool.engine.leaf_block_n == 16
    assert pool.engine.leaf_num_warps == 4
    assert pool.engine.decode_state_update_len == 512
    assert pool.engine.decode_route_parallel_reduce
    assert pool.engine.decode_route_parallel_reduce_block_d == 32
    assert pool.decode_local_limit == 768

    bf16_pool = VLLMLayerLODPool(
        layer,
        settings=replace(settings, kv_bits=0),
        max_requests=1,
        request_capacity=512,
        active_indices=torch.zeros(1, dtype=torch.int64),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert bf16_pool.engine.decode_state_update_len == 256
    assert not bf16_pool.engine.decode_route_parallel_reduce
    assert bf16_pool.decode_local_limit == 512
    assert bf16_pool.engine.recursive_prefill_all_leaves
    assert bf16_pool.engine.recursive_prefill_all_leaves_token_limit == 0

    # Qwen3.8 and K2 use the same recursive INT4 attention organization. Only
    # their exact kernel tile geometry and route backend differ.
    qwen_layer = torch.nn.Module()
    qwen_layer.num_heads = 24
    qwen_layer.num_kv_heads = 4
    qwen_layer.head_size = 256
    qwen_layer.head_size_v = 256
    qwen_layer.impl = SimpleNamespace(scale=256**-0.5)
    qwen_pool = VLLMLayerLODPool(
        qwen_layer,
        settings=replace(settings, prefill_open_count=3),
        max_requests=1,
        request_capacity=512,
        active_indices=torch.zeros(1, dtype=torch.int64),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        has_query_norm=True,
        has_key_norm=True,
    )
    assert qwen_pool.engine.prefill_hierarchical_route
    assert qwen_pool.engine.recursive_prefill_all_leaves
    assert qwen_pool.engine.prefill_overlap_coarse_leaf
    assert not qwen_pool.engine.prefill_overlap_local_lod


def main() -> None:
    verify_default_profile()
    verify_operational_settings()
    verify_tuning_requires_explicit_override()
    verify_model_geometries()
    verify_scheduler_guard()
    verify_pool_startup_audit()
    verify_recursive_prefill_overrides()
    print("vLLM LOD production profile verification passed")


if __name__ == "__main__":
    main()
