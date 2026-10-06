"""The K3 panel must include all four decode updates, not infer them."""

from copy import deepcopy
import json

import pytest

from benchmarks.kimi_k3_decode_power2 import command, load_result, validate_result
from benchmarks import kimi_k3_decode_power2 as panel


def result(mode="two-tier", batch=8):
    timing = {
        "request_num_preemptions": [0] * batch,
        "request_num_cached_tokens": [0] * batch,
        "last_token_spread_seconds": 0,
        "all_requests_live_overlap_seconds": 1.025,
        "decode_window_seconds": 1.025,
    }
    counts = {f"layer.{i}": {"catch_up_batches": 4, "catch_up_rows": 4 * batch}
              for i in range(24)}
    return {
        "decode_tokens": 1026, "tensor_parallel_size": 8,
        "decode_context_parallel_size": 8, "dummy_attention": False,
        "mode": mode, "batch_size": batch,
        "worker_attention_audit": [{
            "cudagraph_mode": "FULL_DECODE_ONLY", "dummy_attention_layers": 0,
            "dense_gluon_decode_installed": mode == "full",
        } for _ in range(8)],
        "measurements": {"16384": {
            "prompts": [{"trace_tokens": 1026} for _ in range(batch)],
            "greedy_output_identical": True, "decode_timings_seconds": [1.025],
            "decode_ms_per_batch_step": 1.0,
            "measured_batch_timings": [[timing]],
            "measured_decode_update_counters": [[deepcopy(counts) for _ in range(8)]],
        }},
    }


@pytest.mark.parametrize("mode,batch", [("full", 1), ("two-tier", 1), ("two-tier", 8)])
def test_four_update_panel_accepts_verified_work(mode, batch):
    validate_result(result(mode, batch))


def test_four_update_panel_rejects_missing_update():
    data = result()
    data["measurements"]["16384"]["measured_decode_update_counters"][0][0]["layer.0"]["catch_up_batches"] = 3
    with pytest.raises(AssertionError, match="four updates per row"):
        validate_result(data)


def test_four_update_panel_rejects_batch_shared_instead_of_per_request_updates():
    data = result()
    data["measurements"]["16384"]["measured_decode_update_counters"][0][0]["layer.0"]["catch_up_rows"] = 4
    with pytest.raises(AssertionError, match="four updates per row"):
        validate_result(data)


def test_four_update_panel_command_preserves_1025_timed_steps():
    args = command("two-tier", 8, (16384, 32768, 65536), 3)
    assert args[args.index("--decode-tokens") + 1] == "1026"
    assert args[args.index("--batch-size") + 1] == "8"
    assert args[args.index("--kv-cache-memory-bytes") + 1] == str(3 << 30)
    assert "--synchronized-decode" in args
    assert "--fixed-decode-trace" in args


def test_router_followup_normalizes_only_verified_matching_cohort(tmp_path):
    from benchmarks.prolong import token_digest

    reference = result(batch=1)
    tokens = list(range(1026))
    archived = reference["measurements"]["16384"]
    archived["prompts"][0].update(token_sha256="prompt", trace_token_sha256=token_digest(tokens))
    data = deepcopy(reference)
    data["measurements"] = {"16384": {
        "measurement_status": "complete", "worker_attention_audit_status": "passed",
        "prompt_token_sha256": ["prompt"], "generated_token_ids": [tokens],
        "decode_ms_per_batch_step": 1.0, "decode_samples_seconds_per_batch_step": [.001],
        "measured_batch_timings": archived["measured_batch_timings"][0],
        "measured_decode_update_counters": archived["measured_decode_update_counters"],
    }}
    path = tmp_path / "router.json"
    path.write_text(json.dumps(data))
    normalized = panel.load_router_result(path, reference)
    assert normalized["router_four_waves"]
    assert normalized["measurements"]["16384"]["prompts"] == archived["prompts"]
    data["measurements"]["16384"]["generated_token_ids"][0][-1] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        panel.load_router_result(path, reference)


def test_router_followup_ignores_failure_artifact_without_measurements(tmp_path):
    path = tmp_path / "oct6-current-b1-decode-16k.failure.json"
    path.write_text(json.dumps({"status": "failed", "error": "capacity"}))
    assert panel.load_router_result(path, result(batch=1)) is None


def test_multi_context_followup_renders_after_single_context_overlay(tmp_path, monkeypatch):
    from benchmarks.prolong import token_digest

    monkeypatch.setattr(panel, "RESULTS", tmp_path)
    lengths = (16384, 65536, 131072)
    monkeypatch.setattr(panel, "PLANS", [("two-tier", 1, lengths, 1)])
    reference = result(batch=1)
    tokens = list(range(1026))
    original = reference["measurements"]["16384"]
    reference["measurements"] = {str(length): deepcopy(original) for length in lengths}
    for context, point in reference["measurements"].items():
        point["prompts"][0].update(token_sha256=context, trace_token_sha256=token_digest(tokens))
    panel.output_path("two-tier", 1, lengths).write_text(json.dumps(reference))
    for filename, contexts, ms in (
        ("oct6-cached-means-decode-b1-64k.json", (65536,), 2.),
        ("oct6-cached-means-decode-b1-16k128k.json", (16384, 131072), 3.),
    ):
        followup = deepcopy(reference)
        followup["measurements"] = {}
        for length in contexts:
            archived = reference["measurements"][str(length)]
            timing = deepcopy(archived["measured_batch_timings"][0][0])
            timing["decode_window_seconds"] = timing["all_requests_live_overlap_seconds"] = ms * 1.025
            followup["measurements"][str(length)] = dict(
                measurement_status="complete", worker_attention_audit_status="passed",
                prompt_token_sha256=[str(length)], generated_token_ids=[tokens],
                decode_ms_per_batch_step=ms, decode_samples_seconds_per_batch_step=[ms / 1000],
                measured_batch_timings=[timing],
                measured_decode_update_counters=archived["measured_decode_update_counters"],
            )
        (tmp_path / filename).write_text(json.dumps(followup))
    panel.render()
    rendered = (tmp_path / "DECODE_POWER2.md").read_text()
    assert "| 16K | — | 3.000 |" in rendered
    assert "| 64K | — | 2.000 |" in rendered
    assert "| 128K | — | 3.000 |" in rendered


def owner_followup(reference):
    from benchmarks.prolong import token_digest

    tokens = list(range(1026))
    point = reference["measurements"]["16384"]
    for prompt in point["prompts"]:
        prompt.update(token_sha256="prompt", trace_token_sha256=token_digest(tokens))
    data = deepcopy(reference)
    data.update(request_owner_prefill=True, owner_tp_mla=True)
    data["measurements"]["16384"] = dict(
        measurement_status="complete", worker_attention_audit_status="passed",
        prompt_token_sha256=["prompt"]*8, generated_token_ids=[tokens]*8,
        decode_ms_per_batch_step=1., decode_samples_seconds_per_batch_step=[.001],
        measured_batch_timings=point["measured_batch_timings"][0],
        measured_decode_update_counters=point["measured_decode_update_counters"],
        active_attention_owner_ranks=list(range(8)),
        owner_decode_graph_replay_deltas=[[1025]*8],
        owner_decode_update_deltas=[[{f"layer.{i}": dict(updates=4,tokens=1025)
                                     for i in range(24)} for _ in range(8)]],
        owner_decode_graph_audits=[dict(owner_layer_count=24,local_decode_heads=[96]*24,
            local_decode_world_sizes=[1]*24,active_global_cadences=[256]*24,
            captured_graphs=[dict(num_tokens=8,graph_instantiated=True)]) for _ in range(8)],
    )
    return data


def test_owner_panel_validates_live_graphs_and_four_global_updates(tmp_path):
    reference = result(batch=8)
    data = owner_followup(reference)
    path = tmp_path / "owner.json"
    path.write_text(json.dumps(data))
    normalized = panel.load_router_result(path, reference, owner=True)
    assert normalized["request_owner_prefill"]
    data["measurements"]["16384"]["owner_decode_update_deltas"][0][0]["layer.0"]["updates"] = 3
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="four owner updates"):
        panel.load_router_result(path, reference, owner=True)


def test_render_removes_old_lod_rows_and_labels_current_owners(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "RESULTS", tmp_path)
    monkeypatch.setattr(panel, "PLANS", [("two-tier",8,(16384,32768),3)])
    reference = result(batch=8)
    owner = owner_followup(reference)
    reference["measurements"]["32768"] = deepcopy(reference["measurements"]["16384"])
    panel.output_path("two-tier",8,(16384,32768)).write_text(json.dumps(reference))
    (tmp_path / "oct6-cached-means-decode-owner-b8-16k.json").write_text(json.dumps(owner))
    panel.render()
    rendered = (tmp_path / "DECODE_POWER2.md").read_text()
    assert "B8 row-per-GPU LoD" in rendered
    assert "| 16K | — | — | — | — | 1.000 | — |" in rendered
    assert "| 32K | — | — | — | — | — | — |" in rendered
    assert "correct-dcp-four-updates.json](" not in rendered


def test_completed_partial_points_are_preserved(tmp_path):
    data = result(batch=1)
    data["argv"] = command("two-tier", 1, (16384,), 1)
    for name in ("decode_tokens", "tensor_parallel_size", "decode_context_parallel_size",
                 "batch_size", "mode", "dummy_attention"):
        data.pop(name)
    data["status"] = "in-progress"
    partial = tmp_path / "run.partial.json"
    partial.write_text(json.dumps(data))
    loaded = load_result(tmp_path / "run.json")
    assert loaded["_source_file"] == "run.partial.json"
    assert loaded["decode_tokens"] == 1026
    assert loaded["measurements"] == data["measurements"]


def test_corrected_decode_never_resumes_invalid_old_lod_results(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "RESULTS", tmp_path)
    old = tmp_path / "oct4-lod-b1-decode-power2-four-updates.json"
    old.write_text(json.dumps(result(batch=1)))
    lengths = (16384, 32768, 65536, 131072, 262144)
    assert "correct-dcp" in panel.output_path("two-tier", 1, lengths).name
    assert list(panel.sources_for("two-tier", 1, lengths)) == []


def test_corrected_decode_reuses_unaffected_dense_control(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "RESULTS", tmp_path)
    lengths = (16384, 32768, 65536, 131072, 262144)
    path = panel.output_path("full", 1, lengths)
    assert path.name == "oct4-full-b1-decode-power2-four-updates.json"
    path.write_text(json.dumps(result(mode="full", batch=1)))
    assert len(list(panel.sources_for("full", 1, lengths))) == 1


def sharded_result():
    from benchmarks.prolong import token_digest

    dense = result(mode="full", batch=1)
    archived = {"trace_tokens": 1025, "token_sha256": "same-prompt",
                "trace_token_sha256": token_digest(list(range(1025)))}
    dense["measurements"]["16384"]["prompts"] = [archived]
    data = result(batch=1)
    data.update(global_centroid_sharded_leaf_prefill=True, reference_trace_extensions={
        "16384": [{"archived_outputs": 1025, "measured_outputs": 1026,
                   "archived_trace_sha256": archived["trace_token_sha256"],
                   "extended_trace_sha256": token_digest(list(range(1026)))}]})
    data["measurements"]["16384"].update(
        measurement_status="complete", worker_attention_audit_status="passed",
        prompt_token_sha256=["same-prompt"], generated_token_ids=[list(range(1026))],
        decode_samples_seconds_per_batch_step=[0.001])
    data["measurements"]["16384"]["measured_batch_timings"] = (
        data["measurements"]["16384"]["measured_batch_timings"][0])
    return data, dense


def test_sharded_panel_verifies_archived_trace_and_extra_step(tmp_path):
    data, dense = sharded_result()
    path = tmp_path / "sharded.json"
    path.write_text(json.dumps(data))
    loaded = panel.load_sharded_result(path, dense)
    assert loaded["measurements"]["16384"]["decode_ms_per_batch_step"] == 1.0
    assert json.loads(path.read_text()) == data  # raw artifact is not normalized in place


@pytest.mark.parametrize("fault", ["prompt", "trace", "update", "extra_token"])
def test_sharded_panel_rejects_unmatched_or_incomplete_work(tmp_path, fault):
    data, dense = sharded_result()
    point = data["measurements"]["16384"]
    if fault == "prompt":
        point["prompt_token_sha256"] = ["different-prompt"]
    elif fault == "trace":
        point["generated_token_ids"][0][0] += 1
    elif fault == "extra_token":
        point["generated_token_ids"][0][-1] += 1
    else:
        point["measured_decode_update_counters"][0][0]["layer.0"]["catch_up_batches"] = 3
    path = tmp_path / "sharded.json"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        panel.load_sharded_result(path, dense)


def test_sharded_panel_does_not_render_warmup_or_failed_point(tmp_path):
    data, dense = sharded_result()
    data["measurements"]["16384"]["measurement_status"] = "failed"
    path = tmp_path / "sharded.json"
    path.write_text(json.dumps(data))
    assert panel.load_sharded_result(path, dense) is None
