#!/usr/bin/env python3
"""Check the current vLLM plugin API, pool lifecycle, and native K/V gather."""

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
    VLLM_LOD_CACHE_OWNERSHIP="dual",
)

import vllm_lod_plugin.runtime as runtime_module
from vllm_lod_plugin.backend import (
    NATIVE_LAYOUT,
    LODAttentionBackend,
    LODAttentionImpl,
)
from vllm_lod_plugin.plugin import register
from vllm_lod_plugin.runtime import VLLMLODRuntime


def _check_native_gather(
    runtime: VLLMLODRuntime,
    layer: SimpleNamespace,
    layout: str,
) -> None:
    original_layout = runtime_module.NATIVE_LAYOUT
    runtime_module.NATIVE_LAYOUT = layout
    try:
        blocks = 4
        if layout == "flash":
            layer.kv_cache = (
                torch.arange(blocks * 2 * 16 * 256, dtype=torch.float32)
                .to(torch.bfloat16)
                .view(blocks, 2, 16, 256)
            )
            def reference(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                selected = layer.kv_cache.index_select(0, ids)
                expected_k, expected_v = selected.split(128, dim=-1)
                expected_k = expected_k.permute(1, 0, 2, 3).reshape(
                    1, 2, 32, 128
                )
                expected_v = expected_v.permute(1, 0, 2, 3).reshape(
                    1, 2, 32, 128
                )
                return expected_k, expected_v
        else:
            from vllm.v1.attention.ops.paged_attn import PagedAttention

            layer.kv_cache = torch.empty(2, blocks, 16, 2, 128, dtype=torch.bfloat16)
            semantic_k = (
                torch.arange(blocks * 16 * 2 * 128, dtype=torch.float32)
                .to(torch.bfloat16)
                .view(blocks, 16, 2, 128)
            )
            semantic_v = semantic_k + 1024
            key_cache, value_cache = PagedAttention.split_kv_cache(
                layer.kv_cache, 2, 128
            )
            key_cache.copy_(
                semantic_k.permute(0, 2, 1, 3)
                .reshape(blocks, 2, 16, 16, 8)
                .permute(0, 1, 3, 2, 4)
            )
            value_cache.copy_(semantic_v.permute(0, 2, 3, 1))
            def reference(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                selected_k = semantic_k.index_select(0, ids)
                selected_v = semantic_v.index_select(0, ids)
                expected_k = selected_k.permute(2, 0, 1, 3).reshape(
                    1, 2, 32, 128
                )
                expected_v = selected_v.permute(2, 0, 1, 3).reshape(
                    1, 2, 32, 128
                )
                return expected_k, expected_v
        expected_k, expected_v = reference(torch.tensor([2, 0]))
        key, value = runtime._gather_native_prefix(
            layer, torch.tensor([2, 0]), length=19, block_size=16
        )
        torch.testing.assert_close(key, expected_k[..., :19, :])
        torch.testing.assert_close(value, expected_v[..., :19, :])
        tables = torch.tensor([[2, 0], [1, 3]])
        key, value = runtime._gather_native_prefix_batch(
            layer, tables, length=19, block_size=16
        )
        alternate_k, alternate_v = reference(tables[1])
        expected_k = torch.cat((expected_k, alternate_k), dim=0)
        expected_v = torch.cat((expected_v, alternate_v), dim=0)
        torch.testing.assert_close(key, expected_k[..., :19, :])
        torch.testing.assert_close(value, expected_v[..., :19, :])
    finally:
        runtime_module.NATIVE_LAYOUT = original_layout


def main() -> None:
    register()
    register()
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

    _check_native_gather(runtime, layer, "flash")
    _check_native_gather(runtime, layer, "rocm")

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
    )
    for pool in runtime.pools.values():
        assert pool.direct_prefill_plan is not None
        assert [item[2] - item[1] for item in pool.direct_prefill_plan] == [3, 5]
    runtime._use_native_attention(["direct-a", "direct-b"])
    assert not runtime._prepare_direct_prefill(
        ["direct-a"],
        np.asarray([3], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    ), "a native prefix-cache hit without an LOD shadow must fall back"
    runtime._release_lod_row("direct-a")
    runtime._release_lod_row("direct-b")

    preserved = runtime._lod_row("preserved-decode")
    for pool in runtime.pools.values():
        pool.ready[preserved] = True
        pool.metadata[preserved]["total_len"] = 10
        pool.catch_up = lambda *_args, **_kwargs: None
    runtime._prepare_native_attention(
        ["preserved-decode", "new-prefill"],
        np.asarray([10, 0], dtype=np.int32),
        np.asarray([0, 1, 5], dtype=np.int32),
    )
    for pool in runtime.pools.values():
        assert pool.native_append_plan == ((preserved, 0, 1, 10),)

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
    print(f"vLLM LOD plugin contract ({NATIVE_LAYOUT}; flash+rocm gather): PASS")


if __name__ == "__main__":
    main()
