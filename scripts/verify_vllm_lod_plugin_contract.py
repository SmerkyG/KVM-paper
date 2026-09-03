#!/usr/bin/env python3
"""Check the external-cache vLLM plugin API and pool lifecycle."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "integrations",
        "vllm_lod",
    ),
)

os.environ.update(
    VLLM_LOD_CHUNK_SIZE="16",
    VLLM_LOD_LOCAL_WINDOW="32",
    VLLM_LOD_STATE_FACTOR="4",
    VLLM_LOD_STATE_MIN="16",
    VLLM_LOD_POOL_SIZE="4",
    VLLM_LOD_MAX_CONTEXT="64",
    VLLM_LOD_KV_BITS="0",
    VLLM_LOD_PREFILL_MODE="direct",
    # Removed controls must not be able to reactivate native/placeholder K/V.
    VLLM_LOD_CACHE_OWNERSHIP="dual",
    VLLM_LOD_EXTERNAL_KV_CACHE="0",
    VLLM_LOD_NATIVE_PLACEHOLDER_CACHE="1",
)

from vllm_lod_plugin.backend import (
    LODAttentionBackend,
    LODAttentionImpl,
)
from vllm_lod_plugin.config import scheduled_static_leaf_cap
from vllm_lod_plugin.plugin import register
from vllm_lod_plugin.pool import (
    _prefill_coarse_direct_gqa_geometry,
    _prefill_hierarchical_route_geometry,
    _prefill_overlap_geometry,
    _recursive_prefill_all_leaves_geometry,
    _recursive_prefill_all_leaves_token_limit,
    _recursive_state_route_backend,
)
from vllm_lod_plugin.runtime import VLLMLODRuntime


def verify_static_leaf_cap_schedule() -> None:
    expected = {
        1: 16,
        8_192: 16,
        16_384: 16,
        32_768: 16,
        65_536: 16,
        65_537: 17,
        131_072: 23,
        262_144: 32,
    }
    assert {
        length: scheduled_static_leaf_cap(length) for length in expected
    } == expected
    proposed = {
        1: 32,
        65_536: 32,
        65_537: 33,
        131_072: 46,
        262_144: 64,
    }
    assert {
        length: scheduled_static_leaf_cap(length, minimum=32, divisor=8)
        for length in proposed
    } == proposed


def verify_prefill_geometry_policy() -> None:
    # Direct GQA packing is shared by flat and recursive prefill; only measured
    # irregular ratios have automatic geometries.
    assert _prefill_coarse_direct_gqa_geometry(128, 5, 8) == (128, 16, 8)
    assert _prefill_coarse_direct_gqa_geometry(256, 6, 4) == (64, 16, 8)
    assert _prefill_coarse_direct_gqa_geometry(256, 4, 2) is None

    # Qwen3.5-0.8B's extra selector launch loses in flat two-tier prefill but
    # wins in recursive three-tier prefill, so the level is part of policy.
    assert not _prefill_hierarchical_route_geometry(2, 256, 4, 2)
    assert _prefill_hierarchical_route_geometry(3, 256, 4, 2)
    assert _prefill_hierarchical_route_geometry(2, 128, 16, 2)

    # Recursive prefill can overlap only its independent local branch. Muse's
    # flat path can additionally overlap coarse and exact-leaf attention.
    assert _prefill_overlap_geometry(2, 128, 16, 2) == (True, True)
    assert _prefill_overlap_geometry(3, 128, 16, 2) == (False, True)
    assert _prefill_overlap_geometry(3, 256, 4, 2) == (False, True)
    assert _prefill_overlap_geometry(2, 256, 4, 2) == (False, False)

    # Phi uses complete experts throughout prefill. Qwen TP1 uses them only
    # through its measured short-context crossover; both retain page-routed
    # decode and other geometries remain conservative.
    assert _recursive_prefill_all_leaves_geometry(3, 128, 4, 2)
    assert _recursive_prefill_all_leaves_geometry(3, 256, 6, 4)
    assert _recursive_prefill_all_leaves_token_limit(3, 128, 4, 2) == 0
    assert _recursive_prefill_all_leaves_token_limit(3, 256, 6, 4) == 65536
    assert not _recursive_prefill_all_leaves_geometry(2, 128, 4, 2)
    assert not _recursive_prefill_all_leaves_geometry(3, 128, 5, 8)
    assert not _recursive_prefill_all_leaves_geometry(3, 256, 4, 2)

    # Re-split preserves the recursive route arithmetic and is measurably
    # faster on these batch-eight production geometries. Muse and unmeasured
    # geometries retain the grouped kernel, and either implementation remains
    # available as an explicit override.
    assert _recursive_state_route_backend(3, 128, 4, 2, 8_192, "auto") == "resplit"
    assert _recursive_state_route_backend(3, 256, 4, 2, 65_536, "auto") == "resplit"
    assert _recursive_state_route_backend(3, 256, 6, 4, 22_528, "auto") == "resplit"
    assert _recursive_state_route_backend(3, 256, 4, 2, 32_768, "auto") == "fused"
    assert _recursive_state_route_backend(2, 256, 4, 2, 131_072, "auto") == "fused"
    assert _recursive_state_route_backend(3, 128, 16, 2, 131_072, "auto") == "fused"
    assert _recursive_state_route_backend(3, 128, 5, 8, 131_072, "auto") == "fused"
    assert _recursive_state_route_backend(3, 512, 8, 2, 131_072, "auto") == "fused"
    assert _recursive_state_route_backend(3, 128, 8, 4, 131_072, "auto") == "fused"
    assert _recursive_state_route_backend(3, 256, 4, 2, 8_192, "fused") == "fused"


def verify_metadata_only_scheduler_cache() -> None:
    from vllm.v1.core.block_pool import BlockPool
    from vllm_lod_plugin.metadata_cache import (
        LODMetadataOnlyFullAttentionManager,
        LODMetadataOnlyFullAttentionSpec,
    )

    spec = LODMetadataOnlyFullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    block_pool = BlockPool(
        num_gpu_blocks=32, enable_caching=True, hash_block_size=16
    )
    manager = LODMetadataOnlyFullAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=16,
    )
    blocks = manager.allocate_new_blocks("first", 64, 64)
    assert len(blocks) == 4 and all(block.block_id == 0 for block in blocks)
    request = SimpleNamespace(request_id="first", block_hashes=[11, 22, 33, 44])
    manager.cache_blocks(request, 64)
    manager.free("first")
    hits, hit_length = manager.find_longest_cache_hit(
        [11, 22, 33, 99], 64, [0], block_pool, spec, False, 16
    )
    assert hit_length == 48 and len(hits[0]) == 3
    manager.add_local_computed_blocks("second", hits[0], hit_length, 0)
    blocks = manager.allocate_new_blocks("second", 64, 64)
    assert len(blocks) == 1 and blocks[0].block_id == 0
    manager.free("second")
    # Null is the only unavailable native block: virtual rows never touched
    # the shared physical pool.
    assert block_pool.get_num_free_blocks() == 31


def verify_cache_ownership_invariant() -> None:
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )
    from vllm_lod_plugin.cache_ownership import (
        _externalize_worker_cache_config,
        _metadata_layer_groups,
        _physical_groups,
    )
    from vllm_lod_plugin.metadata_cache import (
        LODMetadataOnlyFullAttentionSpec,
    )

    common = dict(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    external_spec = LODMetadataOnlyFullAttentionSpec(**common)
    native_spec = FullAttentionSpec(**common)

    # All-global models retain one scheduler group but own no physical group or
    # tensor. This covers the layout used by dense Qwen/Gemma/OLMo/Phi models.
    all_global = [KVCacheGroupSpec(["global.0"], external_spec)]
    assert not _physical_groups(all_global)
    assert _metadata_layer_groups(all_global) == {"global.0": 0}
    config = KVCacheConfig(4, [], all_global)
    logical = _externalize_worker_cache_config(config, {"global.0"})
    assert logical == {"global.0": 0}
    assert config.kv_cache_groups[0].layer_names == []
    assert config.kv_cache_tensors == []

    # Uniform-type grouping is allowed to mix native and external layers. The
    # scheduler retains both; physical planning and the worker retain only the
    # native layer. This is the shape most likely to regress on a new family.
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "global.0": external_spec,
            "native.0": native_spec,
        },
    )
    hybrid = [KVCacheGroupSpec(["global.0", "native.0"], uniform)]
    physical = _physical_groups(hybrid)
    assert len(physical) == 1
    assert physical[0].layer_names == ["native.0"]
    assert set(physical[0].kv_cache_spec.kv_cache_specs) == {"native.0"}
    config = KVCacheConfig(
        32,
        [KVCacheTensor(size=4096, shared_by=["native.0"])],
        hybrid,
    )
    logical = _externalize_worker_cache_config(config, {"global.0"})
    assert logical == {"global.0": 0}
    assert config.kv_cache_groups[0].layer_names == ["native.0"]
    assert set(config.kv_cache_groups[0].kv_cache_spec.kv_cache_specs) == {
        "native.0"
    }
    assert config.kv_cache_tensors[0].shared_by == ["native.0"]

    # Selecting the CUSTOM backend for a model/layer that is not LOD-eligible
    # must leave its ordinary cache untouched.
    native_only = KVCacheConfig(
        32,
        [KVCacheTensor(size=4096, shared_by=["native.0"])],
        [KVCacheGroupSpec(["native.0"], native_spec)],
    )
    assert _externalize_worker_cache_config(native_only, set()) == {}
    assert native_only.kv_cache_groups[0].layer_names == ["native.0"]
    assert native_only.kv_cache_tensors[0].shared_by == ["native.0"]

    # Fail closed if core sizing allocates an external tensor, or if the model
    # marker and scheduler spec ever disagree due to hook ordering/versioning.
    leaking = KVCacheConfig(
        32,
        [KVCacheTensor(size=4096, shared_by=["global.0"])],
        all_global,
    )
    try:
        _externalize_worker_cache_config(leaking, {"global.0"})
    except RuntimeError as error:
        assert "leaked into native GPU K/V tensors" in str(error)
    else:
        raise AssertionError("external GPU K/V allocation was not rejected")
    mismatched = KVCacheConfig(32, [], [KVCacheGroupSpec(["native.0"], native_spec)])
    try:
        _externalize_worker_cache_config(mismatched, {"native.0"})
    except RuntimeError as error:
        assert "marker/spec mismatch" in str(error)
    else:
        raise AssertionError("missing metadata-only spec was not rejected")
    try:
        LODMetadataOnlyFullAttentionSpec.merge([external_spec, native_spec])
    except ValueError:
        pass
    else:
        raise AssertionError("native and external specs were merged")


def main() -> None:
    verify_static_leaf_cap_schedule()
    verify_prefill_geometry_policy()
    register()
    register()
    verify_metadata_only_scheduler_cache()
    verify_cache_ownership_invariant()
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert AttentionBackendEnum.CUSTOM.get_class() is LODAttentionBackend

    implementation = object.__new__(LODAttentionImpl)
    implementation.scale = 128**-0.5
    implementation.lod_eligible = True
    decode_called = False

    class _Pool:
        decode_enabled = True

        def decode(self, query, key, value, metadata, output):
            nonlocal decode_called
            del key, value, metadata
            decode_called = True
            output.copy_(query)
            return output

    layer = SimpleNamespace(
        impl=implementation,
        num_heads=8,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        _vllm_lod_external_kv_cache=True,
        _vllm_lod_pool=_Pool(),
    )
    query = torch.randn(2, 8, 128)
    output = torch.empty_like(query)
    implementation.forward(
        layer,
        query,
        torch.empty(2, 2, 128),
        torch.empty(2, 2, 128),
        torch.empty(0),
        SimpleNamespace(max_query_len=1, num_actual_tokens=2),
        output,
    )
    assert decode_called
    torch.testing.assert_close(output, query)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        num_speculative_tokens=0,
        compilation_config=SimpleNamespace(
            static_forward_context={"layers.0.attn": layer}
        ),
    )
    state = SimpleNamespace(
        vllm_config=config,
        max_num_reqs=8,
        max_model_len=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    runtime = VLLMLODRuntime(state)
    assert len(runtime.pools) == 1
    assert runtime.pool_size == 4
    group = SimpleNamespace(
        layer_names=["layers.0.attn"],
        kv_cache_spec=SimpleNamespace(block_size=16),
    )
    runtime.initialize(SimpleNamespace(kv_cache_groups=[group]))
    assert runtime.settings.prefill_mode == "direct"

    first = runtime._lod_row(7)
    second = runtime._lod_row(3)
    assert first != second
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens_after_padding=4,
        req_ids=["a", "b"],
        idx_mapping_np=np.asarray([7, 3], dtype=np.intp),
        is_prefilling_np=np.zeros(2, dtype=np.bool_),
        max_query_len=1,
        num_computed_tokens_np=np.asarray([32, 32], dtype=np.int32),
    )
    for pool in runtime.pools.values():
        pool.ready[first] = True
        pool.ready[second] = True
        pool.metadata[first]["coverage"] = 16
        pool.metadata[second]["coverage"] = 16
        pool.local_lens.fill_(9)
        pool.catch_up_many = lambda *_args, **_kwargs: None
    runtime.preprocess(batch, (), SimpleNamespace(kv_cache_groups=[group]))
    mapped = runtime.active_indices[:4].tolist()
    assert mapped[:2] == [first, second]
    assert sorted(mapped) == [0, 1, 2, 3]
    for pool in runtime.pools.values():
        assert pool.local_lens[mapped[2:]].tolist() == [0, 0]

    assert runtime._prepare_direct_prefill(
        ["direct-a", "direct-b"],
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([0, 3, 8], dtype=np.int32),
        np.asarray([3, 5], dtype=np.int32),
    )
    for pool in runtime.pools.values():
        assert pool.direct_prefill_plan is not None
        assert [item[2] - item[1] for item in pool.direct_prefill_plan] == [3, 5]
    assert not runtime._prepare_direct_prefill(
        ["direct-a"],
        np.asarray([3], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([4], dtype=np.int32),
    ), "a prefix-cache hit without a semantic LOD row must be rejected"
    runtime._release_lod_row("direct-a")
    runtime._release_lod_row("direct-b")

    dummy_runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=[]))
    for pool in runtime.pools.values():
        pool.decode_enabled = False
        pool.local_lens.fill_(9)
    runtime.prepare_legacy_runner(
        dummy_runner,
        num_reqs=4,
        num_reqs_padded=4,
        max_query_len=1,
        for_capture=False,
    )
    assert runtime.active_indices[:4].tolist() == [0, 1, 2, 3]
    for pool in runtime.pools.values():
        assert pool.decode_enabled
        assert not bool(pool.local_lens.any())

    runtime._release_lod_row(7)
    assert runtime._lod_row(5) == first
    print("vLLM LOD external-cache plugin contract: PASS")


if __name__ == "__main__":
    main()
