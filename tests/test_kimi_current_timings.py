from copy import deepcopy

import pytest

from benchmarks.kimi_k3_current_timings import matched, render, validate_current_point, validate_point


def fixture():
    data = dict(decode_tokens=1026, repeats=1, kda_prefill="gluon_paged",
                weight_cache_id="trained", real_token_cache="frozen.pt", batch_size=8,
                mode="two-tier", request_owner_prefill=True)
    point = dict(measurement_status="complete", worker_attention_audit_status="passed",
        generated_token_ids=[[4] * 1026 for _ in range(8)],
        prefill_samples_seconds=[16.0], decode_samples_seconds_per_batch_step=[.03],
        prompt_token_sha256=["frozen"] * 8,
        measured_batch_timings=[dict(request_num_preemptions=[0] * 8,
            request_num_cached_tokens=[0] * 8, last_token_spread_seconds=0,
            all_requests_live_overlap_seconds=30, decode_window_seconds=30)],
        owner_decode_graph_replay_deltas=[[1025] * 8],
        owner_decode_update_deltas=[[{str(layer): dict(updates=4, tokens=1025)
                                     for layer in range(24)} for _ in range(8)]],
        prefill_seconds=16., decode_ms_per_batch_step=30.,
        worker_attention_audit=[dict(expandable_eager_allocator=True, dense_live_splits=True,
            graph_collectives=dict(tp=dict(registered_capture=True, isolated_graph_allocator=True)))
            for _ in range(8)])
    return data, point


def test_current_owner_point_requires_real_cohort_graphs_and_updates():
    data, point = fixture()
    assert validate_point(data, point)
    point["owner_decode_graph_replay_deltas"][0][3] = 1024
    with pytest.raises(ValueError, match="model graph"):
        validate_point(data, point)


@pytest.mark.parametrize("field,value", [("decode_tokens", 1025), ("repeats", 3),
    ("kda_prefill", "current"), ("weight_cache_id", None)])
def test_renderer_does_not_mix_old_or_fixture_baselines(field, value):
    data, point = fixture()
    data[field] = value
    with pytest.raises(ValueError, match="current trained"):
        validate_point(data, point)


def test_failed_capacity_and_unfinished_results_never_become_timing_cells():
    data, point = fixture()
    data["capacity_only"] = True
    assert not validate_point(data, point)
    data["capacity_only"] = False
    point["measurement_status"] = "warming"
    assert not validate_point(data, point)


def test_speedup_requires_both_prompts_and_entire_continuation_to_match():
    _, point = fixture()
    other = deepcopy(point)
    matched(point, other)
    other["generated_token_ids"][0][-1] = 5
    with pytest.raises(ValueError, match="continuations differ"):
        matched(point, other)


def test_renderer_rejects_five_live_requests_in_an_eight_request_panel():
    data, point = fixture()
    point["measured_batch_timings"][0]["all_requests_live_overlap_seconds"] = 29
    with pytest.raises(ValueError, match="entire cohort"):
        validate_point(data, point)


@pytest.mark.parametrize("field", ["expandable_eager_allocator", "dense_live_splits"])
def test_current_dense_requires_live_splits_and_eager_memory_policy(field):
    data, point = fixture()
    data["mode"] = "full"
    point["worker_attention_audit"][0][field] = False
    with pytest.raises(ValueError):
        validate_current_point(data, point)


def test_current_requires_registered_isolated_graph_communication():
    data, point = fixture()
    point["worker_attention_audit"][0]["graph_collectives"]["tp"]["registered_capture"] = False
    with pytest.raises(ValueError, match="IPC-safe"):
        validate_current_point(data, point)


def test_renderer_excludes_old_sweep_and_accepts_validated_sources(tmp_path):
    import json
    from benchmarks.kimi_k3_current_timings import VERIFIED_SOURCES

    data, point = fixture()
    data.update(measurement_status="complete", measurements={"16384": point})
    (tmp_path / "oct7-current-lod-b8-short.json").write_text(json.dumps(data))
    assert "30.000" not in render(tmp_path).read_text()
    (tmp_path / VERIFIED_SOURCES[2]).write_text(json.dumps(data))
    assert "30.000" in render(tmp_path).read_text()
    (tmp_path / "oct7-fixed-lod-b8-short.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ambiguous"):
        render(tmp_path)


def test_contended_points_stay_raw_but_require_quiet_replacements(tmp_path):
    import json
    from benchmarks.kimi_k3_refresh_panel import completed_lengths

    data, point = fixture()
    data["batch_size"] = 1
    # A full-mode point avoids the B8 owner-cohort checks for this source test.
    data["mode"] = "full"
    point["generated_token_ids"] = [[4] * 1026]
    data.update(measurements={str(n): point for n in (32768, 65536, 131072)})
    source = tmp_path / "oct7-fixed-lod-b1-short.json"
    source.write_text(json.dumps(data))
    assert completed_lengths("full", 1, tmp_path) == {131072}
    text = render(tmp_path).read_text()
    assert "| B1 | 32K | — |" in text
    assert "| B1 | 128K | 16.000 |" in text
    assert len(json.loads(source.read_text())["measurements"]) == 3
