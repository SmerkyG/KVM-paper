from types import SimpleNamespace

import pytest

from benchmarks._decode_update_audit import (
    decode_update_deltas,
    read_decode_update_counters,
)


def test_dense_worker_has_no_lod_update_counters():
    assert read_decode_update_counters(SimpleNamespace()) == {}
    assert decode_update_deltas([{}, {}], [{}, {}]) == [{}, {}]


def test_existing_counters_are_snapshotted_without_mutating_pools():
    pool = SimpleNamespace(catch_up_batches=3, catch_up_rows=24)
    worker = SimpleNamespace(model_runner=SimpleNamespace(model_state=SimpleNamespace(
        _vllm_lod_runtime=SimpleNamespace(pools={"layer.0": pool})
    )))
    before = read_decode_update_counters(worker)
    pool.catch_up_batches += 3
    pool.catch_up_rows += 24
    after = read_decode_update_counters(worker)
    assert before["layer.0"] == {"catch_up_batches": 3, "catch_up_rows": 24}
    assert decode_update_deltas([before], [after]) == [{
        "layer.0": {"catch_up_batches": 3, "catch_up_rows": 24}
    }]
    assert pool.catch_up_batches == 6


@pytest.mark.parametrize("before,after,message", [
    ([{}], [], "worker count"),
    ([{}], [{"layer.0": {}}], "layer set"),
    ([{"layer.0": {"catch_up_batches": 2}}],
     [{"layer.0": {"catch_up_batches": 1}}], "counters reset"),
])
def test_changed_runtime_or_reset_counters_are_rejected(before, after, message):
    with pytest.raises(RuntimeError, match=message):
        decode_update_deltas(before, after)


def test_minimal_benchmark_test_doubles_can_omit_counter_audit():
    assert decode_update_deltas(None, None) is None
