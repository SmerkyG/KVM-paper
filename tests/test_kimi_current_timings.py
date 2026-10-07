from copy import deepcopy

import pytest

from benchmarks.kimi_k3_current_timings import matched, validate_point


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
                                     for layer in range(24)} for _ in range(8)]])
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
