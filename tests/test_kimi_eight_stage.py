import pytest

from benchmarks.kimi_k3_eight_stage import completion_summary, service_rounds, stage_ranges


@pytest.mark.parametrize("layers", [24, 93, 96])
def test_eight_stage_schedule_visits_every_layer_and_preserves_cache_order(layers):
    rounds = service_rounds(layers, 32, pipelined=True)
    seen = set()
    by_layer = {i:[] for i in range(layers)}
    for tasks in rounds:
        assert len({t.stage for t in tasks}) == len(tasks)
        assert [t.stage for t in tasks] == sorted((t.stage for t in tasks), reverse=True)
        for t in tasks:
            assert (t.micro, t.layer) not in seen
            if t.layer:
                assert (t.micro, t.layer-1) in seen
            seen.add((t.micro, t.layer))
            by_layer[t.layer].append(t.micro)
    assert len(seen) == layers*32
    assert all(order == list(range(32)) for order in by_layer.values())
    assert max(map(len, rounds)) == 8


def test_full_stage_boundaries_match_attnres_blocks():
    assert stage_ranges(93) == tuple((i*12, min((i+1)*12, 93)) for i in range(8))
    assert stage_ranges(24) == tuple((i*3, (i+1)*3) for i in range(8))
    with pytest.raises(ValueError):
        stage_ranges(7)


def test_serial_control_has_identical_work_without_overlap():
    a = service_rounds(24, 16, pipelined=True)
    b = service_rounds(24, 16, pipelined=False)
    assert all(len(r) == 1 for r in b)
    assert {t for r in a for t in r} == {t for r in b for t in r}


def test_steady_rate_excludes_drain_and_does_not_extrapolate_short_stream():
    times = list(range(1, 17))
    p = completion_summary(times, chunk=16384, stages=8, pipelined=True)
    assert p["steady_completion_count"] == 9
    assert p["steady_tokens_per_second"] == 16384
    p = completion_summary(times[:8], chunk=16384, stages=8, pipelined=True)
    assert p["steady_tokens_per_second"] is None
