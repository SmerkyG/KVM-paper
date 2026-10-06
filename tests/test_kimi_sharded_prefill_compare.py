"""The storage benchmark must compare sequential, allocation-matched engines."""

import json
import sys

import pytest

from benchmarks import kimi_k3_sharded_prefill_compare as benchmark


def test_comparison_preserves_capacity_resets_experiment_flags_and_checks_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("LOD_KIMI_DCP_PREFILL_WORKSPACE", "1")
    monkeypatch.setenv("LOD_KIMI_DCP_LOCAL_PREFILL", "1")
    monkeypatch.setenv("LOD_KIMI_CROSS_LAYER_PREFILL_GROUP", "1")
    monkeypatch.setattr(sys, "argv", ["compare", "--lengths", "65536", "--batch-size", "8",
                                      "--kv-cache-memory-bytes", "4294967296", "--output-dir", str(tmp_path)])
    calls = []

    def run(command, *, env, check):
        assert check
        assert env["LOD_KIMI_TILE_REFINE"] == "1"
        assert "LOD_KIMI_DCP_LOCAL_PREFILL" not in env
        assert "LOD_KIMI_CROSS_LAYER_PREFILL_GROUP" not in env
        assert command[command.index("--batch-size") + 1] == "8"
        assert command[command.index("--kv-cache-memory-bytes") + 1] == "4294967296"
        mode = command[command.index("--mode") + 1]
        path = tmp_path / command[-1].split("/")[-1]
        calls.append((path.stem, mode, dict(env)))
        path.write_text(json.dumps({
            "measurement_status": "complete", "worker_attention_audit_status": "passed",
            "measurements": {"65536": {
                "prefill_seconds": 1.0, "generated_token_ids": [[13, 17]],
                "worker_memory": [{"peak_torch_allocated_bytes": 2**30, "peak_torch_reserved_bytes": 2**31}],
                "warmup_worker_memory": [{"peak_torch_allocated_bytes": 2**31, "peak_torch_reserved_bytes": 2**32}],
            }},
        }))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    benchmark.main()
    assert [c[0] for c in calls] == ["full", "replicated", "distributed", "reconstructed"]
    assert [c[1] for c in calls] == ["full", "two-tier", "two-tier", "two-tier"]
    assert [c[2].get("LOD_KIMI_DCP_SHARDED_LEAVES") for c in calls] == [None, None, "1", "1"]
    assert [c[2].get("LOD_KIMI_DCP_PREFILL_WORKSPACE") for c in calls] == [None, None, None, "1"]
    data = json.loads((tmp_path / "comparison.json").read_text())
    for item in data["measurements"]["65536"].values():
        assert item["peak_allocated_gib"] == 2
        assert item["peak_reserved_gib"] == 4
    assert data["measurements"]["65536"]["distributed"]["generated_ids_match_replicated"]


def test_incomplete_or_unaudited_results_cannot_be_reported_as_comparison(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["compare", "--lengths", "65536", "--batch-size", "1",
                                      "--kv-cache-memory-bytes", "1073741824", "--output-dir", str(tmp_path)])

    def run(command, **kwargs):
        (tmp_path / "full.json").write_text(json.dumps({
            "measurement_status": "in_progress", "worker_attention_audit_status": "pending"}))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="did not complete"):
        benchmark.main()
    assert not (tmp_path / "comparison.json").exists()


def test_shared_budget_variant_does_not_enable_global_centroid_sharding(monkeypatch, tmp_path):
    monkeypatch.setenv("LOD_KIMI_DCP_SHARDED_LEAVES", "1")
    monkeypatch.setattr(sys, "argv", ["compare", "--lengths", "16384", "--batch-size", "1",
                                      "--kv-cache-memory-bytes", "1073741824", "--output-dir", str(tmp_path),
                                      "--variants", "shared"])

    def run(command, *, env, check):
        assert env["LOD_KIMI_DCP_SHARED_PREFILL"] == "1"
        assert "LOD_KIMI_DCP_SHARDED_LEAVES" not in env
        assert "LOD_KIMI_DCP_LOCAL_PREFILL" not in env
        assert "LOD_KIMI_DCP_PREFILL_WORKSPACE" not in env
        assert command[command.index("--mode") + 1] == "two-tier"
        (tmp_path / "shared.json").write_text(json.dumps({
            "measurement_status": "complete", "worker_attention_audit_status": "passed",
            "measurements": {"16384": {
                "prefill_seconds": 1.0, "generated_token_ids": [[13, 17]],
                "worker_memory": [{"peak_torch_allocated_bytes": 2**30, "peak_torch_reserved_bytes": 2**31}],
                "warmup_worker_memory": [{"peak_torch_allocated_bytes": 2**30, "peak_torch_reserved_bytes": 2**31}],
            }},
        }))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    benchmark.main()
    data = json.loads((tmp_path / "comparison.json").read_text())
    assert set(data["measurements"]["16384"]) == {"shared"}


def test_owner_scratch_configuration_only_changes_workspace_and_route_reduction():
    from types import SimpleNamespace
    from benchmarks.kimi_k3_request_owners import configure_owner_scratch

    engine = SimpleNamespace(two_level_topk=8, max_open_centroid_leaves=1024,
                             prefill_state_update_len=16384, decode_state_update_len=256)
    pool = SimpleNamespace(query_heads=96, engine=engine)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model_state=SimpleNamespace(
        _vllm_lod_runtime=SimpleNamespace(pools={"layer": pool}))))
    result = configure_owner_scratch(worker, 12)
    assert result == dict(layers=1, head_group_limit=12, reduced_fine_routes=True)
    assert engine._lod_kimi_prefill_head_group_limit == 12
    assert engine._lod_kimi_reduce_prefill_routes
    assert engine.two_level_topk == 8 and engine.max_open_centroid_leaves == 1024
    assert engine.prefill_state_update_len == 16384 and engine.decode_state_update_len == 256
    with pytest.raises(ValueError, match="divide"):
        configure_owner_scratch(worker, 7)


def test_trained_prolong_peak_memory_rpcs_are_outside_generation(monkeypatch):
    from types import SimpleNamespace
    from benchmarks import prolong

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: object()))
    monkeypatch.setattr(prolong, "make_speed_prompts", lambda *args, **kwargs: (
        [{"prompt_token_ids": [1]}], [{"tokens": 16}]))
    calls = []
    memory = [{"rank": 0, "peak_torch_allocated_bytes": 1024}]

    def rpc(function, **kwargs):
        calls.append(function.__name__)
        return memory if function.__name__ == "peak_memory" else []

    def generate(*args, **kwargs):
        calls.append("generation")
        return 3.0, 2.0, 1.0, ((1, 2),), {}, [{}], [{}]

    monkeypatch.setattr(prolong, "timed_generate_cohort", generate)
    result = prolong.evaluate_speed(SimpleNamespace(collective_rpc=rpc), object(),
                                   lengths=[16], batch_size=1, samples=1,
                                   decode_tokens=2, repeats=1, seed=0, report_memory=True)["16"]
    assert calls == ["reset_peak_memory", "generation", "peak_memory", "release_worker_allocator_cache",
                     "read_decode_update_counters", "reset_peak_memory", "generation",
                     "read_decode_update_counters", "peak_memory"]
    assert result["warmup_worker_memory"] == memory
    assert result["measured_worker_memory"] == [memory]
    assert result["prefill_seconds"] == 2
