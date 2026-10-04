from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from lod_attention._config import LODMode, ModelFamily
from lod_attention._profile import configure_engine
from lod_attention.kernels.paged_leaf_attention import (
    dequantize_owned_virtual_paged_keys,
)
from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv
import vllm_lod_plugin.models.kimi_k3 as kimi_k3
from vllm_lod_plugin.models.kimi_k3 import (
    absorb_query,
    pack_latent_record,
)
from vllm_lod_plugin.pool import VLLMLayerLODPool


def test_growing_dcp_shadow_capacity_changes_storage_only(monkeypatch) -> None:
    from vllm_lod_plugin.pool import _dcp_prefill_archive_capacity

    kwargs = dict(total_len=16_384, prompt_capacity=524_288,
                  chunk_len=256, headroom=256)
    monkeypatch.delenv("LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK", raising=False)
    assert _dcp_prefill_archive_capacity(**kwargs) == (524_544, 524_544, 0)
    monkeypatch.setenv("LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK", "65536")
    assert _dcp_prefill_archive_capacity(**kwargs) == (65_536, 524_544, 65_536)
    kwargs["total_len"] = 524_288
    assert _dcp_prefill_archive_capacity(**kwargs) == (524_288, 524_544, 65_536)
    monkeypatch.setenv("LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK", "-1")
    with pytest.raises(ValueError, match="must be nonnegative"):
        _dcp_prefill_archive_capacity(**kwargs)


@pytest.mark.parametrize("shared_latent", (False, True))
def test_growing_virtual_archive_preserves_records_and_latent_alias(
    monkeypatch, shared_latent
) -> None:
    import lod_attention._core as core
    from lod_attention._engines import KernelTwoLevelLODAttention

    # CPU test of real allocation/copy logic; page listing is tested on GPU.
    monkeypatch.setattr(core, "append_virtual_paged_kv", lambda *args, **kwargs: None)
    engine = KernelTwoLevelLODAttention(query_heads=1, key_value_heads=1, scale=1)
    engine.virtual_page_storage = True
    key = torch.arange(16, dtype=torch.float32).view(1, 1, 2, 8)
    value = key[..., :4] if shared_latent else key[..., :4].clone() + 1
    cache = engine._new_page_cache(
        key, value, torch.zeros(1, 1, 2, dtype=torch.long),
        state_capacity=2, sequence_capacity=2, virtual_k=key, virtual_v=value,
    )
    cache["leaf_growth_chunk"] = 4
    cache["leaf_capacity_limit"] = 7
    new_key = torch.arange(24, dtype=torch.float32).view(1, 1, 3, 8) + 30
    new_value = new_key[..., :4] if shared_latent else new_key[..., :4].clone() + 1
    engine._append_page_cache(cache, new_key, new_value,
                              torch.zeros(1, 1, 3, dtype=torch.long))
    assert cache["leaf_capacity"] == 7  # slab rounding must respect the limit
    torch.testing.assert_close(cache["leaf_k"][..., :5, :],
                               torch.cat((key, new_key), dim=2))
    torch.testing.assert_close(cache["leaf_v"][..., :5, :],
                               torch.cat((value, new_value), dim=2))
    shares = (cache["leaf_k"].untyped_storage().data_ptr()
              == cache["leaf_v"].untyped_storage().data_ptr())
    assert shares is shared_latent
    engine._append_page_cache(cache, key, value,
                              torch.zeros(1, 1, 2, dtype=torch.long))
    assert cache["leaf_count"] == 7
    with pytest.raises(ValueError, match="exceed their prompt capacity"):
        engine._append_page_cache(cache, key[..., :1, :], value[..., :1, :],
                                  torch.zeros(1, 1, 1, dtype=torch.long))
    assert cache["leaf_count"] == 7


def test_growing_latent_page_lists_match_preallocated_gpu_archive() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU page-list allocation and insertion")
    from lod_attention._config import LODConfig
    from lod_attention._engines import KernelTwoLevelLODAttention

    engine = KernelTwoLevelLODAttention(
        LODConfig(leaf_paged_directory=True),
        query_heads=1, key_value_heads=1, scale=1,
    )
    engine.virtual_page_storage = True
    torch.manual_seed(23)
    key = torch.randn(1, 1, 158, 576, dtype=torch.bfloat16, device="cuda")
    value = key[..., :512]
    owners = torch.arange(158, device="cuda").view(1, 1, -1) % 3
    caches = []
    for capacity in (64, 256):
        cache = engine._new_page_cache(
            key[..., :37, :], value[..., :37, :], owners[..., :37],
            state_capacity=3, sequence_capacity=capacity,
            virtual_k=key[..., :37, :], virtual_v=value[..., :37, :],
        )
        if capacity == 64:
            cache["leaf_growth_chunk"] = 64
            cache["leaf_capacity_limit"] = 256
        engine._append_page_cache(cache, key[..., 37:, :], value[..., 37:, :],
                                  owners[..., 37:])
        caches.append(cache)
    torch.cuda.synchronize()
    for cache in caches:
        torch.testing.assert_close(cache["leaf_k"][..., :158, :], key)
        torch.testing.assert_close(cache["leaf_v"][..., :158, :], value)
        assert (cache["leaf_k"].untyped_storage().data_ptr()
                == cache["leaf_v"].untyped_storage().data_ptr())
    for name in ("slot_lengths", "next_page", "leaf_lens"):
        torch.testing.assert_close(caches[0][name], caches[1][name])
    pages = int(caches[0]["next_page"].item())
    torch.testing.assert_close(caches[0]["page_indices"][..., :pages, :],
                               caches[1]["page_indices"][..., :pages, :])


def test_prefill_allocator_audit_does_not_change_retention_policy(monkeypatch):
    from vllm_lod_plugin import prefill_allocator as runtime

    audit = {"calls": 0, "retained": 0, "reclaimed": 0,
             "minimum_free_bytes_at_check": None}
    monkeypatch.setattr(runtime, "_PREFILL_ALLOCATOR_AUDIT", audit)
    monkeypatch.setenv("LOD_KIMI_REUSE_PREFILL_ALLOCATOR", "1")
    monkeypatch.delenv("LOD_KIMI_PREFILL_MIN_FREE_GIB", raising=False)
    memory = iter(((9 * 1024**3, 256 * 1024**3), (3 * 1024**3, 256 * 1024**3)))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: next(memory))
    reclaimed = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: reclaimed.append(True))
    runtime._reclaim_prefill_allocator(torch.device("cuda"))
    runtime._reclaim_prefill_allocator(torch.device("cuda"))
    assert audit == {"calls": 2, "retained": 1, "reclaimed": 1,
                     "minimum_free_bytes_at_check": 3 * 1024**3}
    assert len(reclaimed) == 1
    monkeypatch.setenv("LOD_KIMI_REUSE_PREFILL_ALLOCATOR", "0")
    runtime._reclaim_prefill_allocator(torch.device("cuda"))
    assert audit["calls"] == 3 and audit["reclaimed"] == 2
    assert audit["minimum_free_bytes_at_check"] == 3 * 1024**3
    assert len(reclaimed) == 2


@pytest.mark.parametrize("reserve_gib", (4, 8))
def test_prefill_allocator_keeps_bounded_headroom(monkeypatch, reserve_gib):
    from vllm_lod_plugin import prefill_allocator

    audit = {"calls": 0, "retained": 0, "reclaimed": 0,
             "minimum_free_bytes_at_check": None}
    monkeypatch.setattr(prefill_allocator, "_PREFILL_ALLOCATOR_AUDIT", audit)
    monkeypatch.setenv("LOD_KIMI_REUSE_PREFILL_ALLOCATOR", "1")
    monkeypatch.setenv("LOD_KIMI_PREFILL_MIN_FREE_GIB", str(reserve_gib))
    memory = iter((reserve_gib * 1024**3, reserve_gib * 1024**3 - 1))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (next(memory), 256 * 1024**3))
    reclaimed = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: reclaimed.append(True))
    prefill_allocator._reclaim_prefill_allocator(torch.device("cuda"))
    prefill_allocator._reclaim_prefill_allocator(torch.device("cuda"))
    assert audit["retained"] == audit["reclaimed"] == 1
    assert len(reclaimed) == 1


@pytest.mark.parametrize("reserve", ("0", "-1", "65", "not-a-number"))
def test_prefill_allocator_rejects_invalid_headroom(monkeypatch, reserve):
    from vllm_lod_plugin import prefill_allocator

    monkeypatch.setenv("LOD_KIMI_REUSE_PREFILL_ALLOCATOR", "1")
    monkeypatch.setenv("LOD_KIMI_PREFILL_MIN_FREE_GIB", reserve)
    with pytest.raises(ValueError):
        prefill_allocator._reclaim_prefill_allocator(torch.device("cuda"))


def test_fixed_state_update_graph_refreshes_sources_and_owns_membership(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU fixed-shape state update")
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention.kernels.kimi_prefill_graph import KimiStateUpdateGraphs

    monkeypatch.setenv("LOD_KIMI_SHARED_LATENT_MERGE", "0")

    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=1)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=16384, has_query_norm=True, has_key_norm=False)
    engine.state_growth_factor = 2
    engine.state_min_len = 4
    capacity, initial_len, overflow_len = 32, 8, 129
    options = dict(state_len=initial_len, ctx_len=192, available_context=192,
                   state_capacity=capacity, scheduled_state_len=initial_len,
                   clustering_query_scale=None, retain_prepared_geometry=False)
    manager = KimiStateUpdateGraphs(max_shapes=1)
    initial = torch.zeros(2, 1, capacity, 576, dtype=torch.bfloat16, device="cuda")
    initial[..., :initial_len, :].normal_()
    keys = initial.clone()
    initial_counts = torch.zeros(2, 1, capacity, 1, device="cuda")
    initial_counts[..., :initial_len, :].fill_(1)
    counts = initial_counts.clone()
    overflow = torch.randn(2, 1, overflow_len, 576, device="cuda").bfloat16()

    def check():
        expected_counts = initial_counts.clone()
        expected_counts.scatter_add_(2, result[4][..., None], torch.ones_like(result[4][..., None]).float())
        torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
        expected_keys = initial.float().clone()
        expected_keys.scatter_add_(2, result[4][..., None].expand_as(overflow), overflow.float())
        torch.testing.assert_close(keys, expected_keys.bfloat16(), atol=0.04, rtol=0.02)

    result = manager.run(engine, keys, keys[..., :512], counts, None,
                         overflow, overflow[..., :512], **options)
    check()
    old_owners = result[4].clone()
    old_storage = result[4]
    keys.copy_(initial)
    counts.copy_(initial_counts)
    overflow.mul_(-1.25)
    result = manager.run(engine, keys, keys[..., :512], counts, None,
                         overflow, overflow[..., :512], **options)
    check()
    torch.testing.assert_close(old_storage, old_owners, atol=0, rtol=0)
    assert len(manager.entries) == 1 and manager.replay_count == 2
    assert manager.fallback_count == 0
    assert not hasattr(engine, "_lod_state_update_buffers")
    assert not hasattr(engine, "_lod_state_maxsim_buffers")
    # Changing the host boundary cannot silently replay the old append count.
    keys.copy_(initial)
    counts.copy_(initial_counts)
    result = manager.run(engine, keys, keys[..., :512], counts, None,
                         overflow, overflow[..., :512], **(options | {"ctx_len": 191}))
    check()
    assert manager.fallback_count == 1 and manager.replay_count == 2
    # A graph also freezes the alias/merge geometry. A changed experimental
    # setting must not silently replay the old variant under the original
    # boundary. With a full bounded cache this uses ordinary construction.
    monkeypatch.setenv("LOD_KIMI_SHARED_LATENT_MERGE", "1")
    keys.copy_(initial)
    counts.copy_(initial_counts)
    result = manager.run(engine, keys, keys[..., :512], counts, None,
                         overflow, overflow[..., :512], **options)
    check()
    assert manager.fallback_count == 2 and manager.replay_count == 2


def test_direct_update_graph_requires_stable_storage_and_refreshes_data():
    if not torch.cuda.is_available():
        pytest.skip("GPU direct-workspace replay")
    from benchmarks._kimi_k3_update_graph import DirectStateUpdateGraph
    from lod_attention._engines import KernelTwoLevelLODAttention

    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=1)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=16384, has_query_norm=True, has_key_norm=False)
    engine.state_growth_factor, engine.state_min_len = 2, 4
    initial = torch.zeros(2, 1, 32, 576, dtype=torch.bfloat16, device="cuda")
    initial[..., :8, :].normal_()
    initial_counts = torch.zeros(2, 1, 32, 1, device="cuda")
    initial_counts[..., :8, :].fill_(1)
    keys, counts = initial.clone(), initial_counts.clone()
    overflow = torch.randn(2, 1, 129, 576, device="cuda").bfloat16()
    inputs = keys, keys[..., :512], counts, None, overflow, overflow[..., :512]
    options = dict(state_len=8, ctx_len=192, available_context=192,
                   state_capacity=32, scheduled_state_len=8,
                   clustering_query_scale=None, retain_prepared_geometry=False)
    graph = DirectStateUpdateGraph(engine._update_state, inputs, options)
    torch.testing.assert_close(keys, initial, atol=0, rtol=0)
    torch.testing.assert_close(counts, initial_counts, atol=0, rtol=0)
    old_storage = old_owners = None
    for scale in (1, -1.25):
        keys.copy_(initial)
        counts.copy_(initial_counts)
        overflow.mul_(scale)
        result = graph(*inputs)
        expected_counts = initial_counts.clone()
        expected_counts.scatter_add_(2, result[4][..., None],
                                     torch.ones_like(result[4][..., None]).float())
        expected_keys = initial.float().clone()
        expected_keys.scatter_add_(2, result[4][..., None].expand_as(overflow), overflow.float())
        torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
        torch.testing.assert_close(keys, expected_keys.bfloat16(), atol=0.04, rtol=0.02)
        if old_storage is not None:
            torch.testing.assert_close(old_storage, old_owners, atol=0, rtol=0)
        old_storage, old_owners = result[4], result[4].clone()
    with pytest.raises(ValueError, match="same input storage"):
        graph(*(inputs[:2] + (counts.clone(),) + inputs[3:]))


def test_shared_weight_cache_retains_distinct_layer_layouts(monkeypatch) -> None:
    from lod_attention.kernels.aiter_mla_prefill_attention import _cached_flat_weight

    monkeypatch.setenv("LOD_KIMI_CACHE_PROJECTION_WEIGHTS", "1")
    first = torch.randn(3, 128, 512)
    second = torch.randn_like(first)
    buffers = {}
    first_flat = _cached_flat_weight(buffers, "coarse", first, (2, 0, 1), (512, 384))
    second_flat = _cached_flat_weight(buffers, "coarse", second, (2, 0, 1), (512, 384))
    repeated = _cached_flat_weight(buffers, "coarse", first, (2, 0, 1), (512, 384))
    assert repeated.data_ptr() == first_flat.data_ptr()
    torch.testing.assert_close(first_flat, first.permute(2, 0, 1).reshape(512, 384))
    torch.testing.assert_close(second_flat, second.permute(2, 0, 1).reshape(512, 384))
    # Do not create a cross-stream dependency between independently prepared
    # local/leaf and coarse layouts by aliasing their output allocation.
    local_flat = _cached_flat_weight(buffers, "local", first, (2, 0, 1), (512, 384))
    assert local_flat.data_ptr() != first_flat.data_ptr()
    assert sum(value is first for value in buffers.values()) == 2


def test_cached_leaf_projection_refreshes_each_layer(monkeypatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU leaf projection")
    monkeypatch.setenv("LOD_KIMI_CACHE_PROJECTION_WEIGHTS", "1")
    source = torch.randn(2, 1, 31, 576, device="cuda").bfloat16()
    weights = [(torch.randn(3, 128, 512, device="cuda").bfloat16(),
                torch.randn(3, 512, 128, device="cuda").bfloat16()) for _ in range(2)]
    buffers = {}
    for uk, uv in (*weights, weights[0]):
        output = expand_kimi_leaf_kv(source, uk, uv, buffers=buffers)
        reference = expand_kimi_leaf_kv(source, uk, uv)
        for actual, expected in zip(output, reference, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _dcp_schedule_fixture(rank: int = 0) -> VLLMLayerLODPool:
    pool = VLLMLayerLODPool.__new__(VLLMLayerLODPool)
    pool.dcp_world_size = 8
    pool.dcp_rank = rank
    pool.dcp_interleave_size = 1
    pool.engine = SimpleNamespace(
        state_growth_factor=16.0,
        state_min_len=256,
        chunk_len=256,
        local_len=512,
        prefill_chunk_len=16_384,
        prefill_local_len=16_640,
        prefill_state_update_len=16_384,
        decode_state_update_len=256,
        decode_cache_headroom=256,
    )
    pool._dcp_global_state_growth_factor = 16.0
    pool._dcp_global_state_min_len = 256
    pool._dcp_global_lengths = {
        name: int(getattr(pool.engine, name))
        for name in (
            "chunk_len",
            "local_len",
            "prefill_chunk_len",
            "prefill_local_len",
            "prefill_state_update_len",
            "decode_state_update_len",
            "decode_cache_headroom",
        )
    }
    return pool


def test_dcp_local_state_schedule_scales_every_token_count() -> None:
    pool = _dcp_schedule_fixture()
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_growth_factor == 16.0 / math.sqrt(8)
        assert pool.engine.state_min_len == 32
        assert (
            pool.engine.chunk_len,
            pool.engine.local_len,
            pool.engine.prefill_chunk_len,
            pool.engine.prefill_local_len,
            pool.engine.prefill_state_update_len,
            pool.engine.decode_state_update_len,
            pool.engine.decode_cache_headroom,
        ) == (32, 64, 2_048, 2_080, 2_048, 32, 32)

    assert pool.engine.state_growth_factor == 16.0
    assert pool.engine.state_min_len == 256
    assert (
        pool.engine.chunk_len,
        pool.engine.local_len,
        pool.engine.prefill_chunk_len,
        pool.engine.prefill_local_len,
        pool.engine.prefill_state_update_len,
        pool.engine.decode_state_update_len,
        pool.engine.decode_cache_headroom,
    ) == (256, 512, 16_384, 16_640, 16_384, 256, 256)


def test_independent_dcp_prefill_keeps_global_centroid_budget_per_rank() -> None:
    pool = _dcp_schedule_fixture()
    pool.kimi_local_dcp_prefill = True
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_min_len == 256
        assert pool.engine.prefill_state_update_len == 2_048
        assert pool.engine.decode_state_update_len == 32
        for global_length in (16_384, 65_536, 262_144):
            local_length = pool._dcp_local_length(global_length)
            local_budget = pool.engine.state_growth_factor * math.sqrt(local_length)
            global_budget = 16 * math.sqrt(global_length)
            assert math.isclose(local_budget, global_budget)
    assert pool.engine.state_growth_factor == 16.0


def test_shared_dcp_prefill_divides_centroid_budget_not_token_cadence() -> None:
    pool = _dcp_schedule_fixture()
    pool.kimi_local_dcp_prefill = True
    pool.kimi_shared_dcp_prefill = True
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_growth_factor == 16.0 / math.sqrt(8)
        assert pool.engine.state_min_len == 32
        assert pool.engine.prefill_state_update_len == 2048
        assert pool.engine.decode_state_update_len == 32


def test_shared_global_routes_are_owned_exactly_once() -> None:
    from vllm_lod_plugin.models.kimi_k3_dcp_prefill import owned_routes

    slots = torch.tensor([[-1, 0, 31, 32, 88, 200, 254, 255]])
    mappings = [owned_routes(slots, rank=rank, local_states=32) for rank in range(8)]
    ownership = torch.stack(mappings).ge(0).sum(0)
    torch.testing.assert_close(ownership, slots.ge(0).long())
    for rank, mapping in enumerate(mappings):
        valid = mapping.ge(0)
        torch.testing.assert_close((mapping[valid] + 32 * rank).long(), slots[valid])


def test_fused_kimi_build_explicitly_emits_eight_candidates(monkeypatch) -> None:
    from lod_attention.kernels.aiter_prefill_attention import _specialized_kimi_coarse_mha_fwd

    builders = []

    def compile_ops(*args, **kwargs):
        builders.append(kwargs["gen_func"])
        return lambda function: function

    core = SimpleNamespace(
        compile_ops=compile_ops,
        get_args_of_build=lambda _: {"flags_extra_hip": [
            "-O3", "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1",
            "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=1",
        ]},
    )
    mha = SimpleNamespace(cmdGenFunc_mha_fwd=lambda *a, **kw: {
        "md_name": "mha_fwd", "blob_gen_cmd": ["generate --receipt 100 --output_dir /tmp"],
    })
    monkeypatch.setitem(sys.modules, "aiter.jit.core", core)
    monkeypatch.setitem(sys.modules, "aiter.ops.mha", mha)
    _specialized_kimi_coarse_mha_fwd.cache_clear()
    try:
        _specialized_kimi_coarse_mha_fwd(async_bias=True, fused_route=True)
        args = builders[0]()
        flags = args["flags_extra_hip"]
        assert "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=0" in flags
        assert "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1" not in flags
        assert "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=0" in flags
        assert "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=1" not in flags
        assert "-DCK_TILE_FMHA_ROUTE_TOPK=8" in flags
        assert args["md_name"].endswith("_asyncbias_v12")
    finally:
        _specialized_kimi_coarse_mha_fwd.cache_clear()


def test_independent_dcp_lse_merge_matches_concatenated_hybrid_field() -> None:
    from vllm_lod_plugin.models.kimi_k3_dcp_prefill import merge_partitions

    torch.manual_seed(19)
    # Exact promoted entries and closed coarse entries simply form a disjoint
    # softmax field. Independent choices on eight ranks need no global top-k.
    scores = torch.randn(8, 3, 5, 17)
    values = torch.randn(8, 3, 5, 17, 11)
    partial_lse = torch.logsumexp(scores, dim=-1)
    partial = (scores.softmax(-1)[..., None] * values).sum(-2)
    combined, lse = partial[0], partial_lse[0]
    for rank in range(1, 8):
        combined = merge_partitions(combined, lse, partial[rank], partial_lse[rank])
        lse = torch.logaddexp(lse, partial_lse[rank])
    total = torch.logsumexp(scores.permute(1, 2, 0, 3).flatten(-2), dim=-1)
    expected = (
        torch.exp(scores - total[None, ..., None])[..., None] * values
    ).sum((0, 3))
    torch.testing.assert_close(combined, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(lse, total)


def test_mla_refinement_returns_the_replaced_field_lse() -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        merge_aiter_mla_prefill_refinement,
    )
    from lod_attention.kernels.aiter_prefill_attention import AiterPrefillCoarse

    torch.manual_seed(23)
    heads, queries, states, width = 2, 19, 16, 128
    scores = torch.randn(1, heads, queries, states, device="cuda")
    means = torch.randn(1, heads, states, width, device="cuda").bfloat16()
    coarse_lse = scores.logsumexp(-1)
    coarse_out = torch.einsum("bhqs,bhsv->bqhv", scores.softmax(-1), means.float())
    coarse_out = coarse_out.bfloat16().contiguous()
    selected_scores, slots = scores.topk(8, dim=-1)
    fine_lse = torch.randn(1, heads, queries, 8, device="cuda")
    fine_out = torch.randn(1, heads, queries, 8, width, device="cuda").bfloat16()
    local_lse = torch.randn(1, heads, queries, device="cuda")
    local_out = torch.randn(1, heads, queries, width, device="cuda").bfloat16()
    mean_k = torch.zeros(1, 1, states, 576, device="cuda").bfloat16()
    coarse = AiterPrefillCoarse(
        output_0=coarse_out, lse_0=coarse_lse,
        output_1=coarse_out, lse_1=coarse_lse,
        mean_k=mean_k, mean_v=means,
        counts=torch.ones(1, 1, states, 1, device="cuda"),
        has_second_partition=False, selected_route_scores=selected_scores,
    )
    carrier = mean_k[..., :1, :].expand(1, heads, queries, 576)
    output, lse = merge_aiter_mla_prefill_refinement(
        carrier, mean_k[..., :0, :], means[..., :0, :], coarse, slots,
        fine_out, fine_lse, local_out, local_lse,
        kv_group_size=heads, scale=0.01, return_lse=True,
    )
    remaining = scores.clone().scatter_(-1, slots, -float("inf"))
    all_scores = torch.cat((remaining, fine_lse, local_lse[..., None]), dim=-1)
    expected_lse = all_scores.logsumexp(-1)
    expected = (
        torch.einsum("bhqs,bhsv->bhqv", torch.exp(remaining - expected_lse[..., None]), means.float())
        + (torch.exp(fine_lse - expected_lse[..., None])[..., None] * fine_out.float()).sum(-2)
        + torch.exp(local_lse - expected_lse)[..., None] * local_out.float()
    )
    torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)
    # Coarse subtraction starts with a BF16 accumulated output, so compare
    # with the appropriate rounding tolerance rather than claiming bit equality.
    torch.testing.assert_close(output.float(), expected, atol=0.012, rtol=0.03)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("direct_result", [False, True])
def test_projected_kimi_leaves_support_aggregated_and_empty_routes(monkeypatch, sparse, direct_result) -> None:
    monkeypatch.setenv("LOD_KIMI_SPARSE_LEAF_PROJECTION", "1" if sparse else "0")
    monkeypatch.setenv("LOD_KIMI_DIRECT_LEAF_RESULT", "1" if direct_result else "0")
    if not torch.cuda.is_available():
        return
    from lod_attention._config import LODConfig
    from lod_attention._engines import KernelTwoLevelLODAttention

    torch.manual_seed(41)
    heads, queries = 4, 17
    engine = KernelTwoLevelLODAttention(
        LODConfig(), query_heads=heads, key_value_heads=1, scale=192**-0.5,
    )
    engine.head_dim = 576
    configure_engine(
        engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
        request_capacity=512, has_query_norm=True, has_key_norm=False,
    )
    key = torch.randn(1, 1, 512, 576, device="cuda").bfloat16()
    cache = engine.build_cache_from_bf16(key, key[..., :512], final_cache_coverage=256).state
    assert cache["state_len"] == 255
    query = torch.randn(1, heads, queries, 192, device="cuda").bfloat16()
    w_uk_t = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    w_uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    engine._lod_kimi_expanded_prefill_chunk = query
    engine._lod_kimi_w_uk_t = w_uk_t
    engine._lod_kimi_w_uv = w_uv
    slots = torch.full((1, heads, queries, 8), -1, device="cuda", dtype=torch.int32)
    slots[:, 0, :8] = torch.arange(8, device="cuda", dtype=torch.int32)
    carrier = key[..., :1, :].expand(1, heads, queries, 576)
    output, lse = engine._paged_leaf_attention(
        carrier, slots, cache["page_cache"], active_slots=255, reduce_routes=True,
    )
    assert output.shape == (1, heads, queries, 128)
    assert output.dtype == query.dtype
    assert torch.isfinite(output).all()
    assert output[:, 1:].count_nonzero() == 0
    assert lse[:, 1:].isneginf().all()
    expanded_k, expanded_v = expand_kimi_leaf_kv(key[..., 1:9, :], w_uk_t, w_uv)
    scores = torch.einsum("bqd,bkd->bqk", query[:, 0, :8].float(), expanded_k[:, 0].float()) * engine.scaling
    expected = scores.softmax(-1) @ expanded_v[:, 0].float()
    torch.testing.assert_close(output[:, 0, :8].float(), expected, atol=0.012, rtol=0.03)
    torch.testing.assert_close(lse[:, 0, :8], scores.logsumexp(-1), atol=0.005, rtol=0.003)


@pytest.mark.parametrize("paged_directory", [False, True])
def test_scalar_leaf_directory_lookup_matches_partial_and_closed_routes(paged_directory) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU leaf kernels")
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention.kernels.paged_prefill import paged_leaf_attention

    torch.manual_seed(79)
    batch, heads, tokens, slots, queries = 2, 2, 73, 4, 19
    engine = KernelTwoLevelLODAttention(
        query_heads=heads, key_value_heads=1, scale=192**-0.5,
    )
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=512, has_query_norm=True, has_key_norm=False)
    engine.leaf_paged_directory = paged_directory
    source = torch.randn(batch, 1, tokens, 576, device="cuda").bfloat16()
    owners = (torch.arange(tokens, device="cuda") % slots).view(1, 1, -1)
    owners = owners.expand(batch, 1, tokens).contiguous()
    cache = engine._new_page_cache(
        source, source[..., :512], owners, state_capacity=slots,
        sequence_capacity=512, virtual_k=source, virtual_v=source[..., :512],
    )
    w_uk = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    w_uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    keys, values = expand_kimi_leaf_kv(source, w_uk, w_uv)
    query = torch.randn(batch, heads, queries, 192, device="cuda").bfloat16()
    routes = torch.arange(queries * 8, device="cuda", dtype=torch.int32)
    routes = routes.remainder(slots).view(1, 1, queries, 8)
    routes = routes.expand(batch, heads, queries, 8).contiguous()
    routes[..., 3:] = -1
    routes[:, 1, -1] = -1
    arguments = dict(page_indices=cache["page_indices"], kv_group_size=1,
                     active_slots=slots, scale=engine.scaling,
                     hash_probes=engine._page_lookup_probes(cache),
                     block_m=32, block_n=16, num_warps=1, reduce_routes=False)
    inputs = (query, keys, values, cache["slot_pages"], cache["overflow_page_keys"],
              cache["overflow_page_values"], cache["overflow_used"],
              cache["slot_lengths"], routes)
    expected = paged_leaf_attention(*inputs, **arguments)
    actual = paged_leaf_attention(*inputs, scalar_page_lookup=True, **arguments)
    # Per-route scratch for a closed route is intentionally unwritten; the
    # final merge masks it. Compare every actually consumed route, not garbage.
    mask = routes.ge(0)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left[mask], right[mask], atol=0, rtol=0)


@pytest.mark.parametrize("paged_directory", [False, True])
def test_sparse_leaf_projection_matches_only_selected_head_centroid_pairs(paged_directory) -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention._config import LODConfig
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention.kernels.kimi_sparse_leaf_projection import project_needed_kimi_leaves

    torch.manual_seed(67)
    batch, heads, tokens, slots = 2, 4, 257, 7
    engine = KernelTwoLevelLODAttention(
        LODConfig(), query_heads=heads, key_value_heads=1, scale=192**-0.5,
    )
    engine.head_dim = 576
    configure_engine(
        engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
        request_capacity=512, has_query_norm=True, has_key_norm=False,
    )
    engine.leaf_paged_directory = paged_directory
    source = torch.randn(batch, 1, tokens, 576, device="cuda").bfloat16()
    owners = (torch.arange(tokens, device="cuda") % slots).view(1, 1, -1)
    owners = owners.expand(batch, 1, tokens).contiguous()
    cache = engine._new_page_cache(
        source, source[..., :512], owners, state_capacity=slots,
        sequence_capacity=512, virtual_k=source, virtual_v=source[..., :512],
    )
    weights_k = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    weights_v = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    counts = torch.zeros(batch, heads, slots, device="cuda", dtype=torch.int32)
    counts[0, 0, 1] = 3
    counts[0, 2, 3] = 5
    counts[1, 1, 0] = 9
    counts[1, 3, 6] = 1
    buffers = {
        "kimi_leaf_expanded_k_token_major": source.new_full((batch * tokens * heads * 192,), -77),
        "kimi_leaf_expanded_v_token_major": source.new_full((batch * tokens * heads * 128,), -77),
    }
    actual_k, actual_v = project_needed_kimi_leaves(
        source, weights_k, weights_v, cache, counts,
        active_slots=slots, hash_probes=engine._page_lookup_probes(cache), buffers=buffers,
    )
    expected_k, expected_v = expand_kimi_leaf_kv(source, weights_k, weights_v)
    for b in range(batch):
        for h in range(heads):
            mask = counts[b, h].index_select(0, owners[b, 0].long()) > 0
            torch.testing.assert_close(actual_k[b, h, mask], expected_k[b, h, mask], atol=0.016, rtol=0.02)
            torch.testing.assert_close(actual_v[b, h, mask], expected_v[b, h, mask], atol=0.016, rtol=0.02)
            assert (actual_k[b, h, ~mask] == -77).all()
            assert (actual_v[b, h, ~mask] == -77).all()


def test_incremental_leaf_projection_preserves_prefix_and_resets_per_request(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.kimi_incremental_leaf_projection import expand_incremental_kimi_leaves
    import lod_attention.kernels.kimi_incremental_leaf_projection as module

    torch.manual_seed(73)
    source = torch.randn(1, 1, 73, 576, device="cuda").bfloat16()
    weights_k = (torch.randn(3, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    weights_v = (torch.randn(3, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    projected_lengths = []
    original = module.expand_kimi_leaf_kv

    def counted(key, uk, uv, **kwargs):
        projected_lengths.append(int(key.size(2)))
        return original(key, uk, uv, **kwargs)

    monkeypatch.setattr(module, "expand_kimi_leaf_kv", counted)
    cache = {}
    for length in (19, 33, 73, 73):
        actual = expand_incremental_kimi_leaves(
            source[..., :length, :], weights_k, weights_v, cache, head_begin=0,
        )
        expected = original(source[..., :length, :], weights_k, weights_v)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0.016, rtol=0.02)
    assert projected_lengths == [19, 14, 40]
    # A new request can reuse buffers and lengths, but not the previous
    # request's projected prefix, even if allocator pointers are recycled.
    other_source = -source
    actual = expand_incremental_kimi_leaves(
        other_source, weights_k, weights_v, {}, head_begin=0,
    )
    expected = original(other_source, weights_k, weights_v)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.016, rtol=0.02)
    assert projected_lengths == [19, 14, 40, 73]


@pytest.mark.parametrize("queries", [19, 128])
@pytest.mark.parametrize("tile_refine", [False, True])
@pytest.mark.parametrize("key_step", [32, 64])
def test_kimi_coarse_projects_fp32_sums_after_dividing(queries: int, monkeypatch, tile_refine, key_step) -> None:
    monkeypatch.setenv("LOD_KIMI_TILE_REFINE", "1" if tile_refine else "0")
    monkeypatch.setenv("LOD_KIMI_COARSE_KEY_STEP", str(key_step))
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        aiter_kimi_expanded_prefill_route_coarse_attention,
    )

    torch.manual_seed(47)
    heads, states = 4, 16
    query = torch.randn(1, heads, queries, 192, device="cuda").bfloat16()
    means = torch.randn(1, 1, states, 576, device="cuda").bfloat16()
    counts = torch.arange(1, states + 1, device="cuda").float().view(1, 1, states, 1)
    sums = means.float() * counts
    w_uk_t = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    w_uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    slots, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
        query, sums, sums[..., :512].contiguous(), counts, w_uk_t, w_uv,
        state_len=states, scale=192**-0.5, normalize_route_query=False,
    )
    if coarse.ready_stream is not None:
        torch.cuda.current_stream().wait_stream(coarse.ready_stream)
    expanded_k, expanded_v = expand_kimi_leaf_kv(means, w_uk_t, w_uv)
    scores = torch.einsum("bhqd,bhsd->bhqs", query.float(), expanded_k.float()) * 192**-0.5
    scores += counts.log().view(1, 1, 1, states).bfloat16().float()
    expected = scores.softmax(-1) @ expanded_v.float()
    assert coarse.output_0.dtype == torch.bfloat16
    assert coarse.mean_v.dtype == torch.bfloat16
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3).float(), expected, atol=0.012, rtol=0.03)
    torch.testing.assert_close(coarse.lse_0, scores.logsumexp(-1), atol=0.005, rtol=0.003)
    # Compare sets: ties/last-bit rounding need not preserve their order.
    torch.testing.assert_close(
        slots.sort(-1).values.long(), scores.topk(8, -1).indices.sort(-1).values,
        msg=f"routes={slots[0, 0, :2].tolist()}, expected={scores.topk(8, -1).indices[0, 0, :2].tolist()}",
    )


@pytest.mark.parametrize("offset,return_lse", [(0, True), (17, True), (0, False)])
def test_kimi_native_local_matches_ck_and_causal_reference(monkeypatch, offset, return_lse):
    if not torch.cuda.is_available():
        pytest.skip("GPU local-attention comparison")
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_kimi_local_prefill_attention

    torch.manual_seed(89)
    heads, tokens = 3, 257
    record = torch.randn(1, 1, tokens, 576, device="cuda").bfloat16()
    q = torch.randn(1, heads, tokens - offset, 192, device="cuda").bfloat16()
    uk = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    carrier = record[..., offset:, :].expand(1, heads, -1, -1)
    kwargs = dict(query_offset=offset, scale=192**-0.5, expanded_q=q,
                  w_uk_t=uk, w_uv=uv, return_lse=return_lse)
    monkeypatch.setenv("LOD_KIMI_NATIVE_LOCAL_PREFILL", "0")
    ck_output, ck_lse = aiter_kimi_local_prefill_attention(carrier, record, **kwargs)
    monkeypatch.setenv("LOD_KIMI_NATIVE_LOCAL_PREFILL", "1")
    output, lse = aiter_kimi_local_prefill_attention(carrier, record, **kwargs)
    torch.testing.assert_close(output, ck_output, atol=0.006, rtol=0.03)
    k, v = expand_kimi_leaf_kv(record, uk, uv)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * 192**-0.5
    causal = torch.arange(tokens, device="cuda")[None, :] <= (
        torch.arange(tokens - offset, device="cuda")[:, None] + offset)
    scores.masked_fill_(~causal, -float("inf"))
    torch.testing.assert_close(output.float(), scores.softmax(-1) @ v.float(), atol=0.009, rtol=0.04)
    if return_lse:
        torch.testing.assert_close(lse, ck_lse, atol=0.005, rtol=0.003)
        torch.testing.assert_close(lse, scores.logsumexp(-1), atol=0.005, rtol=0.003)
    else:
        assert lse.numel() == 0


def test_dcp_catch_up_uses_rank_local_share_of_global_update() -> None:
    # The trigger is the same global per-request boundary on every rank.  At
    # DCP8 each rank then archives its 32 owned leaves: 256 global leaves in
    # aggregate.  Batch size never enters this calculation, so B8 performs
    # 8 * 256 source-token work at the aligned update.
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        pool.dcp_sharded = [True]
        pool.local_capacity = 96
        pool.metadata = [{"coverage": 8_160, "dcp_global_coverage": 65_280}]

        recent, target = pool._catch_up_target(0, 65_791)
        assert recent in (63, 64)
        assert target == 8_160

        recent, target = pool._catch_up_target(0, 65_792)
        assert recent == 64
        assert target == 8_192


def test_dcp_prefill_boundary_is_global_not_rank_local() -> None:
    # Immediately before the next global boundary, ranks own unequal local
    # lengths.  None may advance early merely because its local count rounded
    # up to another 32-token block.
    expected = []
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        expected.append(pool._dcp_local_length(65_280))
        assert pool._dcp_global_decode_coverage(65_791) == 65_280
        assert pool._dcp_global_decode_coverage(65_792) == 65_536
    assert expected == [8_160] * 8


def test_dcp_batch_does_not_divide_the_per_request_cadence() -> None:
    local_work = 0
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        pool.dcp_sharded = [True] * 8
        pool.local_capacity = 96
        pool.metadata = [
            {"coverage": 8_160, "dcp_global_coverage": 65_280}
            for _ in range(8)
        ]
        for slot in range(8):
            _recent, target = pool._catch_up_target(slot, 65_792)
            local_work += target - int(pool.metadata[slot]["coverage"])
    assert local_work == 8 * 256


def test_full_k3_profile_uses_launchable_unmasked_mla_geometry(monkeypatch) -> None:
    # This checks the profile default, not an explicit tuning override used by
    # the GPU suite. Geometry overrides have separate execution checks.
    monkeypatch.delenv("LOD_KIMI_LEAF_BLOCK_M", raising=False)
    engine = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=96, num_key_value_heads=1),
        head_dim=576,
    )
    configure_engine(
        engine,
        family=ModelFamily.KIMI_K3,
        mode=LODMode.THREE_TIER_BF16,
        request_capacity=16_640,
        has_query_norm=False,
        has_key_norm=True,
    )
    assert (engine.leaf_block_m, engine.leaf_block_n) == (32, 16)
    assert engine.prefill_aiter_route_coarse is True


def test_absorbed_latent_logits_equal_expanded_mla_logits() -> None:
    torch.manual_seed(7)
    tokens, heads, nope, latent_dim, direct = 11, 8, 64, 128, 32
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)

    absorbed = absorb_query(query, w_uk_t, nope_dim=nope)
    key = pack_latent_record(latent, direct_key)
    value = key[..., :latent_dim]

    expanded_nope = torch.einsum("tl,hpl->thp", latent, w_uk_t)
    expanded_key = torch.cat(
        (expanded_nope, direct_key.expand(-1, heads, -1)), dim=-1
    )
    expanded_logits = torch.einsum("thd,shd->hts", query, expanded_key)
    latent_logits = torch.einsum("thd,shd->hts", absorbed, key)

    # The two equivalent expressions use different contraction orders, so
    # float32 roundoff grows slightly with the latent width.
    torch.testing.assert_close(latent_logits, expanded_logits, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(value, latent.unsqueeze(1))
    assert value.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()


def test_projected_prefill_defers_absorbed_query_materialization(monkeypatch) -> None:
    # Query-deferred projected prefill is a full-K3 specialization.  The small
    # K3-for-All geometry intentionally uses the generic absorbed-query path.
    tokens, heads, nope, latent_dim, direct, value_dim = 5, 8, 128, 512, 64, 128
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)
    w_uv = torch.randn(heads, latent_dim, value_dim)
    observed: dict[str, object] = {}

    class Pool:
        direct_prefill_plan = ((0, 0, tokens, 0),)
        decode_enabled = False

        def direct_prefill(
            self,
            absorbed_shape_carrier,
            key,
            value,
            output,
            **kwargs,
        ):
            observed["carrier_stride"] = absorbed_shape_carrier.stride()
            observed["key_shape"] = tuple(key.shape)
            observed["value_aliases_key"] = (
                value.untyped_storage().data_ptr()
                == key.untyped_storage().data_ptr()
            )
            observed.update(kwargs)
            output.fill_(3)
            return output

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(),
        W_UK_T=w_uk_t,
        W_UV=w_uv,
        qk_nope_head_dim=nope,
        num_heads=heads,
        v_head_dim=value_dim,
    )

    def unexpected_absorption(*_args, **_kwargs):
        raise AssertionError("projected prefill must not materialize absorbed Q")

    monkeypatch.setattr(kimi_k3, "absorb_query", unexpected_absorption)
    result = kimi_k3._run_lod_mla(
        layer,
        query,
        latent,
        direct_key,
        (tokens, heads * value_dim),
        None,
    )

    assert result.shape == (tokens, heads * value_dim)
    assert torch.all(result == 3)
    assert observed["carrier_stride"][1] == 0
    assert observed["key_shape"] == (tokens, 1, latent_dim + direct)
    assert observed["value_aliases_key"] is True
    assert observed["mla_query"] is query
    assert observed["mla_w_uk_t"] is w_uk_t
    assert observed["mla_w_uv"] is w_uv
    assert observed["defer_mla_query_absorption"] is True


def test_graph_warmup_rows_do_not_enter_smaller_decode_pool() -> None:
    tokens, heads, nope, latent_dim, direct = 16, 2, 4, 8, 2
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)

    class Pool:
        direct_prefill_plan = None
        decode_enabled = True
        max_requests = 1

        def decode(self, *_args, **_kwargs):
            raise AssertionError("synthetic graph rows must not enter decode")

    def copy_output(attention_output, output):
        output.copy_(attention_output.reshape(tokens, heads * latent_dim))

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(),
        W_UK_T=torch.randn(heads, nope, latent_dim),
        qk_nope_head_dim=nope,
        num_heads=heads,
        v_head_dim=latent_dim,
        _v_up_proj=copy_output,
    )
    result = kimi_k3._run_lod_mla(
        layer,
        query,
        latent,
        direct_key,
        (tokens, heads * latent_dim),
        None,
    )
    assert torch.count_nonzero(result) == 0


def test_breakable_prefill_writes_static_output_from_fresh_inputs() -> None:
    tokens, heads, nope, latent_dim, direct, value_dim = 5, 2, 128, 512, 64, 128

    class Pool:
        direct_prefill_plan = ((0, 0, tokens, 0),)
        decode_enabled = False
        gain = 1.0

        def direct_prefill(self, carrier, key, value, output, **kwargs):
            # Model the eager cache/attention portion: both the new query and
            # the live host cache plan must be observed on every replay.
            query = kwargs["mla_query"]
            output.copy_(query[..., :value_dim] * self.gain)
            return output

    pool = Pool()
    layer = SimpleNamespace(
        _vllm_lod_pool=pool,
        W_UK_T=torch.randn(heads, nope, latent_dim),
        W_UV=torch.randn(heads, latent_dim, value_dim),
        qk_nope_head_dim=nope, num_heads=heads, v_head_dim=value_dim,
    )
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    output = torch.empty(tokens, heads * value_dim)
    address = output.data_ptr()
    kimi_k3._run_lod_mla_with_output(layer, query, latent, direct_key, output, None)
    torch.testing.assert_close(output, query[..., :value_dim].flatten(1))
    query.add_(2)
    pool.gain = 3.0
    kimi_k3._run_lod_mla_with_output(layer, query, latent, direct_key, output, None)
    assert output.data_ptr() == address
    torch.testing.assert_close(output, (query[..., :value_dim] * 3).flatten(1))
    with pytest.raises(ValueError, match="output buffer"):
        kimi_k3._run_lod_mla(
            layer, query, latent, direct_key, output.shape, None,
            output_buffer=torch.empty(tokens + 1, heads * value_dim),
        )


@pytest.mark.parametrize("graph_prefill", [False, True])
def test_decode_capture_sizes_only_keep_large_prefill_shapes_when_requested(
    monkeypatch, graph_prefill,
) -> None:
    from vllm_lod_plugin.config import _ensure_exact_lod_decode_capture_sizes

    monkeypatch.setenv("VLLM_LOD_POOL_SIZE", "8")
    monkeypatch.setenv("LOD_KIMI_GRAPH_PREFILL", "1" if graph_prefill else "0")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=SimpleNamespace(name="CUSTOM")),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16_384, 16_392],
            max_cudagraph_capture_size=16_392,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        num_speculative_tokens=0,
    )
    _ensure_exact_lod_decode_capture_sizes(config)
    expected = list(range(1, 9)) + ([16_384, 16_392] if graph_prefill else [])
    assert config.compilation_config.cudagraph_capture_sizes == expected
    assert config.compilation_config.max_cudagraph_capture_size == max(expected)


def test_single_owner_decode_dispatches_full_heads_in_mla_tiles() -> None:
    tokens, heads, nope, latent_dim, direct = 1, 96, 128, 512, 64
    observed = {}

    class Pool:
        direct_prefill_plan = None
        decode_enabled = True
        kimi_head_tiled_decode = True
        max_requests = 1

        def decode_dcp(self, query, key, value, output):
            observed["query_shape"] = tuple(query.shape)
            observed["key_shape"] = tuple(key.shape)
            observed["value_is_latent_view"] = (
                key.untyped_storage().data_ptr()
                == value.untyped_storage().data_ptr()
            )
            output.fill_(2)
            return output, torch.zeros(tokens, heads)

        def decode(self, *_args, **_kwargs):
            raise AssertionError("do not compile a 96-head generic route tile")

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(), W_UK_T=torch.randn(heads, nope, latent_dim),
        qk_nope_head_dim=nope, num_heads=heads, v_head_dim=latent_dim,
        _v_up_proj=lambda source, target: target.copy_(source.flatten(1)),
    )
    output = kimi_k3._run_lod_mla(
        layer, torch.randn(tokens, heads, nope + direct),
        torch.randn(tokens, latent_dim), torch.randn(tokens, 1, direct),
        (tokens, heads * latent_dim), None,
    )
    assert observed == {
        "query_shape": (1, 96, 576), "key_shape": (1, 1, 576),
        "value_is_latent_view": True,
    }
    assert torch.all(output == 2)


def test_update_graph_reconstructs_real_archive_membership() -> None:
    from benchmarks.kimi_k3_update_graph import reconstruct_centroids
    from benchmarks._kimi_k3_update_graph import _prefix_alias

    leaves = torch.arange(8 * 6, dtype=torch.float32).reshape(1, 1, 8, 6)
    page_indices = torch.full((1, 1, 2, 16), -1, dtype=torch.int32)
    page_indices[0, 0, 0, :3] = torch.tensor([1, 3, 5])
    page_indices[0, 0, 1, :5] = torch.tensor([0, 2, 4, 6, 7])
    directories = torch.full((1, 1, 2, 64), -1, dtype=torch.int32)
    directories[0, 0, :, 0] = torch.tensor([0, 1])
    payload = {
        "hash_probes": -1, "active_slots": 2,
        "cache": {"leaf_k": leaves, "slot_lengths": torch.tensor([[[3, 5]]]),
                  "slot_pages": torch.tensor([[[[0], [1]]]]),
                  "page_indices": page_indices, "overflow_page_values": directories},
    }
    sums, lengths, archive = reconstruct_centroids(payload)
    torch.testing.assert_close(sums, torch.stack([
        leaves[0, 0, [1, 3, 5]].sum(0), leaves[0, 0, [0, 2, 4, 6, 7]].sum(0),
    ]))
    torch.testing.assert_close(lengths, torch.tensor([[3.], [5.]]))
    assert _prefix_alias(leaves, leaves[..., :4])
    assert not _prefix_alias(leaves, leaves[..., :4].clone())
    page_indices[0, 0, 1, 0] = 1
    with pytest.raises(ValueError, match="duplicated"):
        reconstruct_centroids(payload)


def test_latent_centroid_is_exact_expanded_kv_centroid() -> None:
    torch.manual_seed(11)
    count, heads, nope, latent_dim, direct, value_dim = 13, 8, 64, 128, 32, 64
    latent = torch.randn(count, latent_dim)
    direct_key = torch.randn(count, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)
    w_uv = torch.randn(heads, latent_dim, value_dim)
    query = torch.randn(heads, nope + direct)

    q_absorbed = absorb_query(query.unsqueeze(0), w_uk_t, nope_dim=nope)[0]
    summed_key = torch.cat((latent.sum(0), direct_key.sum(0)))
    latent_score = torch.einsum("hd,d->h", q_absorbed, summed_key)

    expanded_key_sum = torch.cat(
        (
            torch.einsum("l,hpl->hp", latent.sum(0), w_uk_t),
            direct_key.sum(0).expand(heads, -1),
        ),
        dim=-1,
    )
    expanded_score = torch.einsum("hd,hd->h", query, expanded_key_sum)
    torch.testing.assert_close(latent_score, expanded_score, atol=3e-5, rtol=3e-5)

    latent_mean = latent.mean(0).expand(heads, -1)
    projected_mean = torch.einsum("hl,hlv->hv", latent_mean, w_uv)
    expanded_values = torch.einsum("tl,hlv->thv", latent, w_uv)
    torch.testing.assert_close(
        projected_mean,
        expanded_values.mean(0),
        atol=2e-5 * math.sqrt(count),
        rtol=2e-5,
    )


@pytest.mark.parametrize("fused", [False, True])
def test_multihead_leaf_expansion_writes_projected_key_prefix(monkeypatch, fused) -> None:
    monkeypatch.setenv("LOD_KIMI_FUSED_LEAF_KV", "1" if fused else "0")
    if not torch.cuda.is_available():
        return
    torch.manual_seed(13)
    tokens, heads = 19, 12
    latent_key = torch.randn(
        1, 1, tokens, 576, dtype=torch.bfloat16, device="cuda"
    )
    w_uk_t = torch.randn(
        heads, 128, 512, dtype=torch.bfloat16, device="cuda"
    )
    w_uv = torch.randn(heads, 512, 128, dtype=torch.bfloat16, device="cuda")
    expanded_k, expanded_v = expand_kimi_leaf_kv(latent_key, w_uk_t, w_uv)

    latent = latent_key[0, 0, :, :512].float()
    expected_nope = torch.einsum("tl,hpl->htp", latent, w_uk_t.float())
    expected_direct = latent_key[0, 0, :, 512:].expand(heads, -1, -1)
    expected_v = torch.einsum("tl,hlv->htv", latent, w_uv.float())
    torch.testing.assert_close(
        expanded_k[0, ..., :128].float(), expected_nope, atol=0.5, rtol=0.02
    )
    torch.testing.assert_close(expanded_k[0, ..., 128:], expected_direct)
    torch.testing.assert_close(
        expanded_v[0].float(), expected_v, atol=0.5, rtol=0.02
    )


def test_count_channel_equals_centroid_mass_bias() -> None:
    """A padded dot-product channel exactly represents ``+ log(count)``."""
    torch.manual_seed(19)
    queries, states, heads = 7, 11, 3
    scale = 192**-0.5
    query = torch.randn(queries, heads, 192)
    key = torch.randn(states, heads, 192)
    counts = torch.randint(1, 128, (states,), dtype=torch.int64).float()

    expected = torch.einsum("qhd,khd->hqk", query, key) * scale
    expected += counts.log()[None, None, :]

    augmented_query = torch.zeros(queries, heads, 256)
    augmented_key = torch.zeros(states, heads, 256)
    augmented_query[..., :192] = query
    augmented_key[..., :192] = key
    augmented_query[..., 192] = 1.0 / scale
    augmented_key[..., 192] = counts.log()[:, None]
    actual = torch.einsum(
        "qhd,khd->hqk", augmented_query, augmented_key
    ) * scale
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_projected_sink_merge_uses_real_query_and_each_heads_key() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU MLA merge kernels")
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        expand_kimi_sink_kv, merge_aiter_mla_prefill_refinement,
    )

    torch.manual_seed(89)
    batch, heads, queries, states = 2, 3, 19, 12
    query = torch.randn(batch, heads, queries, 192, device="cuda").bfloat16()
    sink = torch.randn(batch, 1, 1, 576, device="cuda").bfloat16()
    uk = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    projected_key, projected_value = expand_kimi_sink_kv(sink, uk, uv, buffers={})
    # Verify the actual inference projection, including all 64 direct channels.
    expected_k = torch.einsum("bsl,hpl->bhsp", sink[:, 0, :, :512].float(), uk.float())
    torch.testing.assert_close(projected_key[..., :128].float(), expected_k,
                               atol=0.025, rtol=0.025)
    torch.testing.assert_close(projected_key[..., 128:], sink[..., 512:].expand(-1, heads, -1, -1))
    mean_v = torch.randn(batch, heads, states, 128, device="cuda").bfloat16()
    baseline = mean_v.float().mean(2).unsqueeze(1).expand(-1, queries, -1, -1).contiguous().bfloat16()
    slots = torch.arange(8, device="cuda", dtype=torch.int32).view(1, 1, 1, 8)
    slots = slots.expand(batch, heads, queries, 8).contiguous()
    route_out = torch.randn(batch, heads, queries, 8, 128, device="cuda").bfloat16()
    route_lse = torch.randn(batch, heads, queries, 8, device="cuda") * 0.1
    local_out = torch.randn(batch, heads, queries, 128, device="cuda").bfloat16()
    local_lse = torch.full((batch, heads, queries), 0.2, device="cuda")
    coarse = SimpleNamespace(
        mean_k=torch.zeros(batch, 1, states, 576, device="cuda", dtype=torch.bfloat16),
        mean_v=mean_v, counts=torch.ones(batch, 1, states, 1, device="cuda"),
        output_0=baseline, lse_0=torch.full((batch, heads, queries), math.log(states), device="cuda"),
        has_second_partition=False,
        selected_route_scores=torch.zeros_like(route_lse),
    )
    scale = 192**-0.5
    output, lse = merge_aiter_mla_prefill_refinement(
        query, projected_key, projected_value, coarse, slots, route_out, route_lse,
        local_out, local_lse, kv_group_size=heads, scale=scale, return_lse=True,
    )
    sink_score = (query.float() * projected_key.float()).sum(-1) * scale
    weights = route_lse.exp()
    numerator = mean_v[:, :, 8:].float().sum(2).unsqueeze(2).expand(-1, -1, queries, -1)
    numerator = numerator + (weights[..., None] * route_out.float()).sum(3)
    numerator += local_lse.exp()[..., None] * local_out.float()
    numerator += sink_score.exp()[..., None] * projected_value.float()
    denominator = 4 + weights.sum(3) + local_lse.exp() + sink_score.exp()
    torch.testing.assert_close(output.float(), numerator / denominator[..., None],
                               atol=0.004, rtol=0.01)
    torch.testing.assert_close(lse, denominator.log(), atol=0.00001, rtol=0.00001)


@pytest.mark.parametrize("overlap_projection", [False, True])
def test_projected_prefill_never_copies_or_scores_the_query_carrier(monkeypatch, overlap_projection) -> None:
    from lod_attention._engines import KernelTwoLevelLODAttention
    import lod_attention.kernels.aiter_mla_prefill_attention as mla

    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=0.1)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=32768, has_query_norm=True, has_key_norm=False)
    real_query = torch.ones(1, 12, 3, 192)
    carrier = torch.full((1, 1, 3, 576), float("nan")).expand(1, 12, 3, 576)
    engine._lod_kimi_expanded_prefill_chunk = real_query
    engine._lod_kimi_w_uk_t = torch.zeros(12, 128, 512)
    engine._lod_kimi_w_uv = torch.zeros(12, 512, 128)
    keys = torch.zeros(1, 1, 256, 576)
    counts = torch.ones(1, 1, 256, 1)
    sink = torch.ones(1, 1, 1, 576)
    projected_sink = torch.ones(1, 12, 1, 192)
    projected_value = torch.ones(1, 12, 1, 128)
    top = torch.zeros(1, 12, 3, 8, dtype=torch.int32)
    stages = []
    monkeypatch.setenv("LOD_KIMI_OVERLAP_LEAF_PROJECTION", "1" if overlap_projection else "0")
    cache = {"page_indices": torch.zeros(1, 1, 1, 16), "leaf_count": 256, "leaf_k": keys}

    def route(query, *_args, **_kwargs):
        assert query.data_ptr() == carrier.data_ptr() and query.stride(1) == 0
        engine._lod_prefill_selected_route_cap_applied = True
        engine._lod_prefill_aiter_coarse = SimpleNamespace(
            ready_stream=object(), output_0=torch.zeros(1, 3, 12, 128),
            mean_v=torch.zeros(1, 12, 256, 128),
        )
        return top

    def merge(query, key, value, *_args, **_kwargs):
        assert query is real_query and key is projected_sink and value is projected_value
        return torch.ones(1, 12, 3, 128)

    def project(key, *_args, **_kwargs):
        assert key.data_ptr() == keys.data_ptr()
        stages.append("project")
        return torch.zeros(1, 12, 256, 192), torch.zeros(1, 12, 256, 128)

    def leaves(*_args, **_kwargs):
        stages.append("leaves")
        if overlap_projection:
            prepared = engine._lod_kimi_preprojected_leaves
            assert prepared[:2] == (id(cache), 256)
            del engine._lod_kimi_preprojected_leaves
        return torch.zeros(1, 12, 3, 8, 128), torch.zeros(1, 12, 3, 8)

    monkeypatch.setattr(engine, "_route_top_slots", route)
    monkeypatch.setattr(engine, "_paged_leaf_attention", leaves)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: SimpleNamespace(wait_stream=lambda *_: stages.append("wait")))
    monkeypatch.setattr(mla, "expand_kimi_leaf_kv", project)
    monkeypatch.setattr(mla, "expand_kimi_sink_kv", lambda *_args, **_kwargs: (projected_sink, projected_value))
    monkeypatch.setattr(mla, "merge_aiter_mla_prefill_refinement", merge)
    result = engine._two_level_attention(
        carrier, keys, keys[..., :512], keys, keys[..., :512], counts,
        None, keys, keys[..., :512], state_len=256, state_capacity=256,
        page_cache=cache,
        local_branch=(torch.zeros(1, 12, 3, 128), torch.zeros(1, 12, 3)),
        sink_k=sink, sink_v=sink[..., :512],
    )
    assert result.shape == (1, 12, 3, 128) and torch.isfinite(result).all()
    if overlap_projection:
        assert stages[:3] == ["project", "wait", "leaves"]


def test_coarse_graph_falls_back_for_nonfixed_inputs(monkeypatch) -> None:
    import lod_attention.kernels.kimi_prefill_graph as graphs

    with pytest.raises(ValueError, match="positive"):
        graphs.KimiPrefillCoarseGraphs(max_shapes=0)
    sentinel = object()
    calls = []

    def ordinary(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(graphs, "aiter_kimi_expanded_prefill_route_coarse_attention", ordinary)
    manager = graphs.KimiPrefillCoarseGraphs(max_shapes=1)
    q, k = torch.zeros(1, 1, 3, 192), torch.zeros(1, 1, 8, 576)
    counts = torch.ones(1, 1, 8, 1)
    uk, uv = torch.zeros(1, 128, 512), torch.zeros(1, 512, 128)
    assert manager.run(q, k, k[..., :512], counts, uk, uv, state_len=8,
                       scale=0.1, normalize_route_query=False) is sentinel
    assert manager.entries == {} and manager.replay_count == 0 and manager.fallback_count == 1
    assert calls[0][0][0] is q and calls[0][1]["state_len"] == 8


def test_coarse_graph_replays_new_layer_weights_and_bounds_shape_count() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU K3 coarse graph capture")
    from lod_attention.kernels.kimi_prefill_graph import KimiPrefillCoarseGraphs
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        aiter_kimi_expanded_prefill_route_coarse_attention as ordinary,
    )

    torch.manual_seed(91)
    q = torch.randn(1, 12, 16384, 192, device="cuda").bfloat16()
    k = torch.randn(1, 1, 32, 576, device="cuda").bfloat16()
    counts = torch.ones(1, 1, 32, 1, device="cuda")
    lengths = torch.full((1, 1, 32), 2, device="cuda", dtype=torch.int32)
    uk = (torch.randn(12, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    uv = (torch.randn(12, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    manager = KimiPrefillCoarseGraphs(max_shapes=1)

    def run(states, use_graph):
        result = (manager.run if use_graph else ordinary)(
            q, k, k[..., :512], counts, uk, uv, state_len=states, scale=192**-0.5,
            normalize_route_query=False, slot_lengths=lengths,
            max_open_leaf_tokens=1024, buffers={},
        )
        slots, coarse, _, _ = result
        if coarse.ready_stream is not None:
            torch.cuda.current_stream().wait_stream(coarse.ready_stream)
        return tuple(t.clone() for t in (slots, coarse.output_0, coarse.lse_0))

    with torch.inference_mode():
        for states in (16, 16, 32):
            # Shared graph inputs must not reuse layer zero's flattened weights.
            uk.mul_(0.75)
            uv.mul_(-0.5)
            q.mul_(-1)
            actual, expected = run(states, True), run(states, False)
            for a, e in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
    assert len(manager.entries) == 1
    assert manager.replay_count == 2 and manager.fallback_count == 1


def test_coarse_state_preparation_respects_latent_value_and_batch_strides() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU K3 state preparation")
    from lod_attention.kernels.aiter_mla_prefill_attention import _prepare_expanded_mla_state_kernel

    torch.manual_seed(93)
    key = torch.randn(2, 1, 64, 576, device="cuda").bfloat16()[..., ::2, :]
    value = key[..., :512]  # Alias: token stride is 1152, not the value width.
    counts = torch.randint(1, 12, (2, 1, 64, 1), device="cuda").float()[..., ::2, :]
    active, dispatch = 17, 128
    mean_k = torch.empty(2, 1, dispatch, 576, device="cuda", dtype=key.dtype)
    mean_v = torch.empty(2, 1, dispatch, 512, device="cuda", dtype=key.dtype)
    padded_counts = torch.empty(2, 1, dispatch, 1, device="cuda")
    logs = torch.empty(2, dispatch, device="cuda", dtype=key.dtype)
    _prepare_expanded_mla_state_kernel[(2 * dispatch,)](
        key, value, counts, mean_k, mean_v, padded_counts, logs, active, dispatch,
        KEY_BATCH_STRIDE=key.stride(0), KEY_TOKEN_STRIDE=key.stride(2),
        VALUE_BATCH_STRIDE=value.stride(0), VALUE_TOKEN_STRIDE=value.stride(2),
        COUNT_BATCH_STRIDE=counts.stride(0), COUNT_TOKEN_STRIDE=counts.stride(2),
        LATENT_DIM=512, DIRECT_DIM=64, LATENT_BLOCK=512, DIRECT_BLOCK=64, num_warps=8,
    )
    expected = (key[..., :active, :].float() / counts[..., :active, :]).bfloat16()
    torch.testing.assert_close(mean_k[..., :active, :], expected, atol=0, rtol=0)
    torch.testing.assert_close(mean_v[..., :active, :], expected[..., :512], atol=0, rtol=0)
    torch.testing.assert_close(padded_counts[..., :active, :], counts[..., :active, :], atol=0, rtol=0)
    assert mean_k[..., active:, :].eq(0).all() and mean_v[..., active:, :].eq(0).all()
    assert logs[..., active:].isneginf().all()


def test_final_cache_graph_refreshes_data_and_owns_private_update_scratch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU K3 cache graph capture")
    from benchmarks.kimi_k3_cache_graph import check_cache
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention.kernels.kimi_prefill_graph import KimiFinalCacheGraphs

    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=192**-0.5)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=32768, has_query_norm=True, has_key_norm=False)
    engine.state_growth_factor /= math.sqrt(8)
    engine.state_min_len = math.ceil(engine.state_min_len / 8)
    for name in ("chunk_len", "local_len", "prefill_chunk_len", "prefill_local_len",
                 "prefill_state_update_len", "decode_state_update_len"):
        setattr(engine, name, math.ceil(getattr(engine, name) / 8))
    torch.manual_seed(97)
    key = torch.randn(2, 1, 4096, 576, device="cuda").bfloat16()
    manager = KimiFinalCacheGraphs(max_shapes=1)
    # These caller-owned dictionaries must be restored, not captured and then
    # invalidated by runtime._release_cross_layer_state_workspaces().
    caller_scratch = {}
    engine._lod_state_maxsim_buffers = caller_scratch
    with torch.inference_mode():
        for length, coverage in ((4096, 4064), (4096, 4064), (2048, 2016)):
            key.mul_(-0.75)
            source = key[..., :length, :]
            cache = manager.run(engine, source, source[..., :512], final_cache_coverage=coverage)
            check_cache(cache, source)
            if hasattr(engine, "_lod_state_maxsim_buffers") and length == 4096:
                assert engine._lod_state_maxsim_buffers is caller_scratch
                caller_scratch.clear()
                del engine._lod_state_maxsim_buffers
    assert len(manager.entries) == 1
    assert manager.replay_count == 2 and manager.fallback_count == 1


@pytest.mark.parametrize("states", [177, 385, 1039])
@pytest.mark.parametrize("queries", [37, 513])
@pytest.mark.parametrize("pack", ["ordinary", "dense", "chunk256", "chunk512", "chunk1024"])
@pytest.mark.parametrize("tile_n", [32, 64, 128])
@pytest.mark.parametrize("fields", [1, 16])
def test_tile_max_refinement_recovers_exact_global_top_eight(states, queries, pack, tile_n, fields, monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    monkeypatch.setenv("LOD_KIMI_DENSE_TILE_PACK", "1" if pack == "dense" else "0")
    monkeypatch.setenv("LOD_KIMI_CHUNK_TILE_PACK", "1" if pack.startswith("chunk") else "0")
    monkeypatch.setenv("LOD_KIMI_TILE_PACK_QUERY_BLOCK", pack.removeprefix("chunk")
                      if pack.startswith("chunk") else "256")
    from lod_attention.kernels.kimi_route_tile_refine import refine_kimi_centroid_tiles
    from lod_attention.kernels.aiter_prefill_attention import _reduce_route_candidates

    torch.manual_seed(79)
    batch, heads = 2, 3
    q = torch.randn(batch, heads, queries, 192, device="cuda").bfloat16()
    k = torch.randn(batch, states, heads, 192, device="cuda").bfloat16()
    logs = torch.randint(1, 50, (batch, states), device="cuda").float().log().bfloat16()
    scale = 192**-0.5
    scores = torch.einsum("bhqd,bshd->bhqs", q.float(), k.float()) * scale
    scores += logs.float()[:, None, None, :]
    tiles = math.ceil(states / 128) * (128 // tile_n)
    padded = torch.nn.functional.pad(scores, (0, tiles * tile_n - states), value=-float("inf"))
    maxima = padded.view(batch, heads, queries, tiles, tile_n).amax(-1)
    candidates = torch.full((batch, heads, tiles, fields, queries), float("nan"), device="cuda")
    # No other candidate channel is valid or may be read by the tile selector.
    candidates[:, :, :, 0] = maxima.transpose(-1, -2) / math.log(2)
    refined = refine_kimi_centroid_tiles(candidates, q, k, logs, state_len=states,
                                          scale=scale, tile_n=tile_n)
    actual_slots, _, _, actual_scores = _reduce_route_candidates(
        refined, state_len=states, head_dim=192, emit_metadata=False,
    )
    expected_slots = scores.topk(8, dim=-1).indices
    torch.testing.assert_close(actual_slots.sort(-1).values, expected_slots.sort(-1).values)
    expected_scores = scores.gather(-1, actual_slots)
    torch.testing.assert_close(actual_scores, expected_scores, atol=0.002, rtol=0.002)


def test_dcp_int4_page_dequantization_restores_owned_records() -> None:
    if not torch.cuda.is_available():
        return
    batch, heads, pages, page_size, head_dim = 1, 2, 2, 16, 8
    leaves = pages * page_size
    page_indices = (
        torch.arange(leaves, dtype=torch.int32, device="cuda")
        .view(batch, 1, pages, page_size)
        .expand(batch, heads, pages, page_size)
        .contiguous()
    )
    page_counts = torch.full(
        (batch, heads, pages), page_size, dtype=torch.int32, device="cuda"
    )
    next_page = torch.full(
        (batch, heads), pages, dtype=torch.int32, device="cuda"
    )
    # Low nibbles reconstruct to the page centroid; high nibbles add one.
    packed_keys = torch.full(
        (batch, heads, leaves, head_dim // 2),
        0x98,
        dtype=torch.uint8,
        device="cuda",
    )
    page_scales = torch.ones(
        batch, heads, pages, head_dim // 4, dtype=torch.bfloat16, device="cuda"
    )
    quantized_sums = torch.full(
        (batch, heads, pages, head_dim),
        32,
        dtype=torch.int8,
        device="cuda",
    )
    summary_scales = torch.full(
        (batch, heads, pages, head_dim // 4),
        0.5,
        dtype=torch.bfloat16,
        device="cuda",
    )

    for rank in range(2):
        actual = dequantize_owned_virtual_paged_keys(
            page_indices,
            page_counts,
            next_page,
            packed_keys,
            page_scales,
            quantized_sums,
            summary_scales,
            source_slot=0,
            sink_len=0,
            local_length=leaves // 2,
            dcp_rank=rank,
            dcp_world_size=2,
            dcp_interleave_size=4,
        )
        expected = torch.empty_like(actual)
        expected[..., 0::2] = 1.0
        expected[..., 1::2] = 2.0
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cap", [None, 17])
@pytest.mark.parametrize("queries", [37, 513])
def test_kimi_sorted_candidate_merge_preserves_routes_ties_and_closing(cap, queries) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU candidate reducer")
    from lod_attention.kernels.aiter_prefill_attention import _reduce_route_candidates
    from lod_attention.kernels.kimi_route_candidate_merge import merge_sorted_kimi_candidates

    torch.manual_seed(731)
    batch, heads, states = 2, 3, 1024
    # Deliberately tie scores across and within the already sorted lists.
    scores = torch.randint(-8, 11, (batch, heads, 8, 8, queries), device="cuda").float()
    scores = scores.sort(dim=3, descending=True, stable=True).values
    indices = (torch.arange(8, device="cuda")[None, None, :, None, None] * 128
               + torch.arange(8, device="cuda")[None, None, None, :, None])
    candidates = torch.cat((scores, indices.expand_as(scores).float()), dim=3).contiguous()
    candidates[:, :, 7, 4:8] = -float("inf")
    candidates[:, :, 7, 12:16] = -1
    lengths = torch.randint(1, 50, (batch, 1, states), device="cuda").int()
    kwargs = dict(state_len=states, slot_lengths=lengths if cap is not None else None,
                  max_open_leaf_tokens=cap)
    expected = _reduce_route_candidates(candidates, head_dim=192, emit_metadata=False,
                                         close_selected_above_limit=cap is not None, **kwargs)
    actual = merge_sorted_kimi_candidates(candidates, **kwargs)
    assert actual[1:3] == (None, None)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)


@pytest.mark.parametrize("states", [53, 1039])
@pytest.mark.parametrize("direct_only", [False, True])
def test_kimi_tiled_state_assignment_uses_all_channels_and_partial_tiles(states, direct_only) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU state assignment")
    from lod_attention.kernels.lod_kernels import new_state_maxsim_buffers, tiled_dot_maxsim

    torch.manual_seed(57)
    leaves = torch.randn(2, 2, 137, 576, device="cuda").bfloat16()
    centroids = torch.randn(2, 2, states, 576, device="cuda").bfloat16()
    if direct_only:
        leaves[..., :512] = 0
        centroids[..., :512] = 0
    centroids[..., 1, :] = centroids[..., 0, :]
    buffers = new_state_maxsim_buffers(leaves, 137)
    actual = tiled_dot_maxsim(leaves, centroids, buffers, prefix="route")
    dense = leaves @ centroids.transpose(-1, -2)
    expected = dense.max(-1)
    torch.testing.assert_close(actual[0], expected.values, atol=0, rtol=0)
    # Explicit BF16 rounding and smallest-index tie behavior must match BLAS.
    torch.testing.assert_close(actual[1], expected.indices, atol=0, rtol=0)


@pytest.mark.parametrize("states", [177, 1039])
@pytest.mark.parametrize("queries", [1, 37])
@pytest.mark.parametrize("key_major", [False, True])
def test_kimi_fullscore_probe_selector_preserves_ties_and_partial_tiles(states, queries, key_major) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU score selector")
    from benchmarks.kimi_k3_score_output import select_full_scores

    torch.manual_seed(92)
    stride = math.ceil(states / 128) * 128
    scores = torch.randint(-16, 16, (2, 3, queries, stride), device="cuda").float()
    # Padding is deliberately larger than valid scores; it must not win.
    scores[..., states:] = 1000
    buffers = {}
    for fresh in range(2):
        if fresh:
            scores[..., :states].neg_()
        source = scores.transpose(-1, -2).contiguous() if key_major else scores
        actual = select_full_scores(source, None, None, None, state_len=states,
                                    scale=1, buffers=buffers, key_major=key_major)
        indices = scores[..., :states].argsort(dim=-1, descending=True, stable=True)[..., :8]
        values = scores[..., :states].gather(-1, indices)
        torch.testing.assert_close(actual[:, :, 0, :8].transpose(-1, -2), values,
                                   atol=0, rtol=0)
        torch.testing.assert_close(actual[:, :, 0, 8:].transpose(-1, -2).long(), indices,
                                   atol=0, rtol=0)


@pytest.mark.parametrize("chunk_items", [512, 1024, 2048])
@pytest.mark.parametrize("queries,states", [(37, 23), (513, 1039)])
def test_kimi_sorted_route_ordinals_are_dense_and_unique(chunk_items, queries, states) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires GPU route sorting")
    from lod_attention.kernels.kimi_sorted_route_counts import count_sorted_kimi_routes

    torch.manual_seed(83)
    routes = torch.randint(-1, states, (2, 3, queries, 8), device="cuda")
    routes[0, 0] = 7  # Extreme contention is still an exact dense range.
    routes[1, 2] = -1  # Empty fragments cannot retain stale counts.
    buffers = {}
    items = queries * 8
    for fresh in range(2):
        if fresh:
            routes.copy_(torch.where(routes >= 0, (routes + 3) % states, -1))
        counts, offsets = count_sorted_kimi_routes(routes, active_slots=states,
                                                   buffers=buffers, chunk_items=chunk_items)
        for slots, ordinals, actual_counts in zip(routes.flatten(0, 1).flatten(1),
                                                 offsets.flatten(0, 1).flatten(1),
                                                 counts.view(6, states), strict=True):
            valid = slots >= 0
            expected_counts = torch.bincount(slots[valid], minlength=states).int()
            torch.testing.assert_close(actual_counts, expected_counts)
            actual_pairs = slots[valid] * items + ordinals[valid]
            sorted_slots = torch.repeat_interleave(torch.arange(states, device="cuda"),
                                                    expected_counts.long())
            starts = expected_counts.cumsum(0) - expected_counts
            sorted_ordinals = (torch.arange(sorted_slots.numel(), device="cuda")
                               - torch.repeat_interleave(starts, expected_counts.long()))
            expected_pairs = sorted_slots * items + sorted_ordinals
            torch.testing.assert_close(actual_pairs.sort().values, expected_pairs)
