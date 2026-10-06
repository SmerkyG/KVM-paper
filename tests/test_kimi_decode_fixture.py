"""Fixture diagnostics must trace live graphs without changing serving timing."""

from types import SimpleNamespace as NS

import pytest

from benchmarks.kimi_k3_decode_fixture import arm_decode_trace, audit_fixture, fixture_overrides, summarize_trace


def test_layer_override_preserves_geometry_and_removes_other_layers():
    override = fixture_overrides(12)
    assert override["num_hidden_layers"] == override["first_k_dense_replace"] == 12
    assert override["linear_attn_config"]["full_attn_layers"] == list(range(1, 13))
    assert override["linear_attn_config"]["kda_layers"] == []
    assert "num_attention_heads" not in override
    with pytest.raises(ValueError):
        fixture_overrides(0)


def test_fixture_audit_rejects_retained_ffn():
    import torch

    Decoder = type("KimiDecoderLayer", (), {})
    layer = Decoder()
    layer._vllm_lod_attention_only_fixture = True
    layer.mlp = torch.nn.Identity()
    worker = NS(rank=0, model_runner=NS(model=NS(modules=lambda: [layer])))
    audit = audit_fixture(worker, 1)
    assert audit["ffn_layers"] == 0
    assert audit["router_geometry"] == {}
    layer.mlp = torch.nn.Linear(1, 1)
    with pytest.raises(RuntimeError, match="attention-only"):
        audit_fixture(worker, 1)


def test_trace_excludes_gpu_annotation_and_does_not_sum_overlaps_as_wall():
    def gpu(name, start, end, annotation=False):
        return NS(device_type="GPU", name=name, is_user_annotation=annotation,
                  device_time_total=end-start, time_range=NS(start=start, end=end))
    summary = summarize_trace([gpu("route", 0, 1000), gpu("comm", 500, 1500),
        gpu("wait", 0, 2000, annotation=True), gpu("fine", 2000, 2500)], gpu_device="GPU")
    assert summary["gpu_timeline_span_ms"] == 2.5
    assert summary["gpu_activity_union_ms"] == 2
    assert sum(row["summed_gpu_us"] for row in summary["kernels"]) == 2500
    assert "wait" not in {row["name"] for row in summary["kernels"]}


def test_trace_counter_is_on_actual_graph_replay_and_checks_batch():
    calls = []
    manager = NS(run_fullgraph=lambda desc: calls.append(desc.num_tokens))
    worker = NS(rank=1, model_runner=NS(cudagraph_manager=manager))
    arm_decode_trace(worker, 8)
    assert worker._kimi_fixture_decode_trace["profiler"] is None
    manager.run_fullgraph(NS(num_tokens=8))
    assert calls == [8]
    assert worker._kimi_fixture_decode_trace["replays"] == 1
    with pytest.raises(RuntimeError, match="live batch"):
        manager.run_fullgraph(NS(num_tokens=1))


def test_live_population_is_snapshotted_before_request_release(monkeypatch):
    import torch
    import benchmarks.kimi_k3_decode_fixture as fixture

    manager = NS(run_fullgraph=lambda descriptor: "output")
    worker = NS(rank=1, model_runner=NS(cudagraph_manager=manager))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(fixture, "route_population", lambda worker: {"live": True})
    arm_decode_trace(worker, 1, steps=2)
    manager.run_fullgraph(NS(num_tokens=1))
    assert worker._kimi_fixture_decode_trace["population"] is None
    assert manager.run_fullgraph(NS(num_tokens=1)) == "output"
    assert worker._kimi_fixture_decode_trace["population"] == {"live": True}


def test_router_resource_report_reads_spills_not_only_allocated_registers():
    from benchmarks.kimi_k3_decode_route_tune import compiler_resources

    kernel = NS(asm={"amdgcn": ".vgpr_count: 512\n.vgpr_spill_count: 603\n"
                     ".sgpr_spill_count: 130\n.private_segment_fixed_size: 1592\n"})
    assert compiler_resources(kernel) == dict(vgpr_count=512, vgpr_spill_count=603,
        sgpr_spill_count=130, private_segment_fixed_size=1592)


@pytest.mark.parametrize("heads", [12, 96])
@pytest.mark.parametrize("dimension,waves", [(576, 4), (192, 1)])
def test_k3_router_profile_preserves_algorithm_for_dcp_and_owner(heads, dimension, waves):
    from lod_attention._config import LODMode, ModelFamily
    from lod_attention._profile import configure_engine

    engine = NS(config=NS(num_attention_heads=heads, num_key_value_heads=1), head_dim=dimension)
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=65536, has_query_norm=True, has_key_norm=False)
    assert engine.decode_route_num_warps == waves
    assert engine.decode_route_group_size == 64
    assert engine.two_level_topk == engine.prefill_two_level_topk == 8
    assert engine.max_open_centroid_leaves == 1024
    assert engine.decode_state_update_len == 256
    assert engine.prefill_state_update_len == 16384


def test_compact_consumer_splits_depend_on_live_batch_not_request_capacity():
    from lod_attention.kernels.kimi_gluon_decode import kimi_lod_decode_splits
    assert [kimi_lod_decode_splits(b, head_tiled_metadata=True) for b in (1, 2, 3, 4, 8)] == [32, 64, 64, 16, 16]
    assert [kimi_lod_decode_splits(b, head_tiled_metadata=False) for b in (1, 8)] == [64, 64]
