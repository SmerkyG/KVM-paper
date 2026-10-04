"""The K3 panel must include all four decode updates, not infer them."""

from copy import deepcopy
import json

import pytest

from benchmarks.kimi_k3_decode_power2 import command, load_result, validate_result


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
