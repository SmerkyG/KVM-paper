from types import SimpleNamespace as NS

import pytest
import torch

from lod_attention.kernels._decode_scratch import (
    TRANSIENT_DECODE_BUFFERS, can_share_decode_scratch, empty_decode_tensor,
)
from lod_attention.kernels.paged_decode_buffers import new_fused_decode_buffers


def make_buffers(registry=None, *, device="cpu", batch=2, heads=32):
    query = torch.empty(batch, heads, 1, 576, dtype=torch.bfloat16, device=device)
    return new_fused_decode_buffers(
        query, splits=8, value_dim=512, state_capacity=64, route_group_size=32,
        gqa_union_kv_heads=heads // 16, gqa_union_index_capacity=512,
        gqa_union_hip=True, shared_scratch=registry,
    )


def test_only_transient_buffers_alias_across_layers():
    registry = {}
    left, right = make_buffers(registry), make_buffers(registry)
    for name in left:
        assert (left[name] is right[name]) == (name in TRANSIENT_DECODE_BUFFERS), name
    # These are deliberately private, even though their shapes are equal.
    for name in ("output", "kimi_gluon_final_lse", "gqa_union_seen_stamps",
                 "gqa_union_epochs", "completion", "local_lens",
                 "gqa_union_hip_block_table"):
        assert left[name].data_ptr() != right[name].data_ptr()
    left["gqa_union_epochs"].add_(3)
    assert right["gqa_union_epochs"].eq(1).all()
    left["route_group_out"].fill_(17)
    assert right["route_group_out"].eq(17).all()


def test_no_registry_preserves_private_buffers():
    left, right = make_buffers(), make_buffers()
    assert all(left[name].data_ptr() != right[name].data_ptr() for name in left)


@pytest.mark.parametrize("fields,speculative,allowed", [
    ({}, 0, True), ({"ubatch_size": 1}, 0, True),
    ({"use_ubatching": True}, 0, False), ({"enable_dbo": True}, 0, False),
    ({"ubatch_size": 2}, 0, False), ({}, 1, False),
])
def test_concurrent_or_speculative_calls_keep_private_scratch(fields, speculative, allowed):
    assert can_share_decode_scratch(NS(**fields), speculative) is allowed


def test_registry_separates_names_shapes_and_dtypes():
    registry = {}
    base = empty_decode_tensor("partial_out", 3, 4, dtype=torch.float32,
                               device="cpu", shared_scratch=registry)
    assert empty_decode_tensor("partial_out", 3, 4, dtype=torch.float32,
                               device="cpu", shared_scratch=registry) is base
    for name, shape, dtype in [("partial_lse", (3, 4), torch.float32),
                               ("partial_out", (4, 3), torch.float32),
                               ("partial_out", (3, 4), torch.bfloat16)]:
        other = empty_decode_tensor(name, *shape, dtype=dtype, device="cpu",
                                    shared_scratch=registry)
        assert other.data_ptr() != base.data_ptr()
    # Unrecognized names default to private; adding a buffer cannot silently
    # make persistent state shareable.
    first = empty_decode_tensor("future_state", 3, dtype=torch.float32,
                                device="cpu", shared_scratch=registry)
    second = empty_decode_tensor("future_state", 3, dtype=torch.float32,
                                 device="cpu", shared_scratch=registry)
    assert first is not second and len(registry) == 4


def test_24_layer_kimi_reservation_shrinks_without_aliasing_results():
    query = torch.empty(8, 96, 1, 576, dtype=torch.bfloat16, device="meta")
    registry = {}
    layers = [new_fused_decode_buffers(
        query, splits=8, value_dim=512, state_capacity=1088, route_group_size=32,
        gqa_union_kv_heads=6, gqa_union_index_capacity=35000, gqa_union_hip=True,
        shared_scratch=registry,
    ) for _ in range(24)]
    private_bytes = 24 * sum(t.numel() * t.element_size() for t in layers[0].values())
    unique = {id(t): t for layer in layers for t in layer.values()}
    shared_bytes = sum(t.numel() * t.element_size() for t in unique.values())
    assert shared_bytes < private_bytes * .1
    assert len({id(layer["output"]) for layer in layers}) == 24
    assert len({id(layer["kimi_gluon_final_lse"]) for layer in layers}) == 24


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Kimi GPU graph equivalence")
def test_two_kimi_layers_share_scratch_under_graph_replay():
    """Exercise the actual route/union/Gluon decoder, not a dummy operation."""
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    torch.manual_seed(197)
    batch, heads, prefix = 2, 32, 1024
    registry = {}
    shared, private, queries, records = [], [], [], []
    for layer_index in range(2):
        layer = NS(num_heads=heads, num_kv_heads=1, head_size=576,
                   kv_lora_rank=512, scale=.072, _vllm_lod_absorbed_mla=True)
        engine = KernelTwoLevelLODAttention(LODConfig(), query_heads=heads,
                                           key_value_heads=1, scale=.072)
        engine.head_dim = 576
        configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                         request_capacity=2048, has_query_norm=True, has_key_norm=False)
        key = (torch.randn(batch, 1, prefix, 576, device="cuda") * .2).bfloat16()
        cache = engine.build_cache_from_bf16(key, key[..., :512])
        tail = int(cache.state["recent_len"])
        cache.state["recent_k"] = cache.state["recent_k"][..., :tail, :]
        cache.state["recent_v"] = cache.state["recent_k"][..., :512]
        pair = []
        for scratch in (registry, None):
            pool = VLLMLayerLODPool(
                layer, settings=VLLMLODSettings(), max_requests=batch,
                request_capacity=2048, active_indices=torch.arange(batch, device="cuda"),
                dtype=torch.bfloat16, device=torch.device("cuda"),
                request_owner_prefill=False, shared_decode_scratch=scratch,
            )
            pool.install_rows(tuple(range(batch)), cache)
            pool.active_decode_rows = tuple(range(batch))
            pool.reserve_decode_buffers(batch)
            pair.append(pool)
        shared.append(pair[0])
        private.append(pair[1])
        queries.append((torch.randn(batch, heads, 576, device="cuda") * .03).bfloat16())
        records.append((torch.randn(batch, 1, 576, device="cuda") * .2).bfloat16())

    assert shared[0].dcp_decode_buffer_storage["route_group_out"] is shared[1].dcp_decode_buffer_storage["route_group_out"]
    outputs = [[torch.empty(batch, heads, 512, dtype=torch.bfloat16, device="cuda")
                for _ in range(2)] for _ in range(2)]
    def execute(pools, targets):
        return [pool.decode_dcp(q, key, key[..., :512], out)[1]
                for pool, q, key, out in zip(pools, queries, records, targets)]

    # Warm kernels, then capture both layers in order in each graph.
    for _ in range(3):
        execute(shared, outputs[0])
        execute(private, outputs[1])
    graphs, lses = [], []
    for pools, targets in ((shared, outputs[0]), (private, outputs[1])):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            lse = execute(pools, targets)
        graphs.append(graph)
        lses.append(lse)
    pointers = {key: value.data_ptr() for key, value in registry.items()}
    # Different queries and advancing live tails must not pick up another
    # layer's output/LSE or stale union scratch on subsequent graph replays.
    for step in range(8):
        for q in queries:
            q.mul_(.9).add_(.001 * (step + 1))
        graphs[0].replay()
        graphs[1].replay()
        for layer_index in range(2):
            torch.testing.assert_close(outputs[0][layer_index], outputs[1][layer_index],
                                       atol=1e-4, rtol=.002)
            torch.testing.assert_close(lses[0][layer_index], lses[1][layer_index],
                                       atol=2e-6, rtol=2e-6)
        assert pointers == {key: value.data_ptr() for key, value in registry.items()}
