from types import SimpleNamespace as NS

import pytest
import numpy as np

from benchmarks._prefill_batch_audit import (
    arm_prefill_batch_audit, finish_prefill_batch_audit, prefill_lengths,
    validate_prefill_batch_audits,
)


def test_live_runner_offsets_ignore_decode_and_unused_capacity():
    batch = NS(num_reqs=3,
               query_start_loc_np=np.array([0, 16384, 32768, 32769, 99999]))
    assert prefill_lengths(batch) == [16384, 16384]
    batch.query_start_loc_np[:] = [0, 1, 2, 3, 99999]
    assert prefill_lengths(batch) == []


def test_audit_wraps_v2_real_batch_preparation_and_restores_before_measurement():
    calls = []
    batch = NS(num_reqs=1, query_start_loc_np=np.array([0, 16384]))
    original = lambda value: calls.append(value) or batch
    runner = NS(prepare_inputs=original, model=NS(forward=lambda: None))
    worker = NS(rank=0, model_runner=runner)
    arm_prefill_batch_audit(worker)
    with pytest.raises(RuntimeError):
        arm_prefill_batch_audit(worker)
    assert runner.prepare_inputs("warmup") is batch
    audit = finish_prefill_batch_audit(worker)
    assert audit["chunks"] == [[16384]]
    assert audit["prefill_tokens"] == 16384
    assert runner.prepare_inputs is original
    assert not hasattr(runner, "_prefill_batch_audit_chunks")
    runner.prepare_inputs("measurement")
    assert calls == ["warmup", "measurement"]


def test_batch_audit_requires_real_token_count_cohort_and_matching_ranks():
    audits = [{"rank": r, "chunks": [[16384, 16384]] * 16,
               "prefill_tokens": 65536 * 8} for r in range(8)]
    kwargs = dict(length=65536, batch_size=8, row_chunk=16384, cohort=2, world_size=8)
    validate_prefill_batch_audits(audits, **kwargs)
    with pytest.raises(RuntimeError):
        validate_prefill_batch_audits(audits, **(kwargs | {"cohort": 1}))
    with pytest.raises(RuntimeError):
        validate_prefill_batch_audits(audits[:-1], **kwargs)
    audits[0] = audits[0] | {"prefill_tokens": 65536 * 7}
    with pytest.raises(RuntimeError):
        validate_prefill_batch_audits(audits, **kwargs)


def test_optional_memory_snapshot_runs_only_in_warmup_prefill(monkeypatch):
    from benchmarks import _kimi_prefill_memory as memory

    calls = []
    monkeypatch.setenv("LOD_BENCHMARK_PREFILL_MEMORY_AUDIT", "1")
    monkeypatch.setattr(memory, "snapshot_prefill_memory",
                        lambda worker: calls.append(worker.rank) or {"bytes": 12})
    batch = NS(num_reqs=1, query_start_loc_np=np.array([0, 16384]))
    original = lambda: batch
    worker = NS(rank=3, model_runner=NS(prepare_inputs=original))
    arm_prefill_batch_audit(worker)
    worker.model_runner.prepare_inputs()
    batch.query_start_loc_np[:] = [0, 1]
    worker.model_runner.prepare_inputs()
    audit = finish_prefill_batch_audit(worker)
    assert calls == [3]
    assert audit["memory_snapshots"] == [{"bytes": 12}]
    assert worker.model_runner.prepare_inputs is original
    worker.model_runner.prepare_inputs()
    assert calls == [3]
