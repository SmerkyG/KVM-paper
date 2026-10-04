"""Counter RPCs stay outside timed generation and exclude warmup updates."""

import sys
from types import SimpleNamespace

from benchmarks._decode_update_audit import read_decode_update_counters


def test_speed_counter_audit_excludes_warmup_and_preserves_timing(monkeypatch):
    import benchmarks.prolong as prolong

    events = []
    counts = {"catch_up_batches": 0, "catch_up_rows": 0}

    class LLM:
        def collective_rpc(self, function):
            if function is read_decode_update_counters:
                events.append("snapshot")
                return [{"layer.0": dict(counts)}]
            events.append("release")
            return None

    def generate(*args, **kwargs):
        events.append("generate")
        counts["catch_up_batches"] += 3
        counts["catch_up_rows"] += 24
        return 32.0, 8.0, 24.0, ((7,),), {}, [{}], [{}]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)
    ))
    monkeypatch.setattr(prolong, "make_speed_prompts", lambda *args, **kwargs: (
        [{"prompt_token_ids": [1]}], [{"request_index": 0}]
    ))
    monkeypatch.setattr(prolong, "timed_generate_cohort", generate)
    measured = prolong.evaluate_speed(
        LLM(), object(), lengths=[65536], batch_size=8, samples=8,
        decode_tokens=1025, repeats=1, seed=0,
    )["65536"]

    assert events == ["generate", "release", "snapshot", "generate", "snapshot"]
    assert measured["measured_decode_update_counters"] == [[{
        "layer.0": {"catch_up_batches": 3, "catch_up_rows": 24}
    }]]
    assert measured["prefill_seconds"] == 8.0
    assert measured["decode_ms_per_batch_step"] == 24.0 / 1024 * 1000
    assert measured["cohort_wall_timings_seconds"] == [32.0]
