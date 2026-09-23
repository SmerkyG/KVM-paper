from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks._identity import source_identity
from benchmarks.attention_timing import summarize_attention_time


def _result(*, mode: str, dummy: bool, prefill: float, decode: float) -> dict:
    return {
        "benchmark_identity": {
            "schema": 1,
            "source": {
                "git_commit": "deadbeef",
                "source_sha256": "source-hash",
                "source_file_count": 42,
            },
            "runtime": {
                "python": "3.test",
                "packages": {"torch": "test"},
            },
        },
        "checkpoint": "test/model",
        "dataset": "test/data",
        "dataset_revision": "revision",
        "mode": mode,
        "measure": "speed",
        "dummy_attention": dummy,
        "batch_size": 8,
        "speed_samples": 8,
        "tensor_parallel_size": 1,
        "decode_tokens": 1025,
        "seed": 0,
        "scheduler_chunk_tokens": 16_384,
        "scheduler_budget_tokens": 16_392,
        "scheduler_cls": "test.Scheduler",
        "speculative_model": None,
        "num_speculative_tokens": None,
        "measurements": {
            "131072": {
                "prefill_seconds": prefill,
                "decode_ms_per_batch_step": decode,
                "prompts": [
                    {"token_sha256": "first"},
                    {"token_sha256": "second"},
                ],
            }
        },
    }


def test_attention_timing_subtracts_matched_dummy_control() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    full = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)

    result = summarize_attention_time(dummy, [("full", full), ("lod", lod)])

    rows = result["measurements"]["131072"]["runs"]
    assert rows[0]["attention_prefill_seconds"] == 240.0
    assert rows[0]["attention_decode_ms_per_batch_step"] == 42.0
    assert rows[1]["attention_prefill_seconds"] == 40.0
    assert rows[1]["attention_decode_ms_per_batch_step"] == 10.0
    assert rows[1]["attention_prefill_speedup_vs_full"] == 6.0
    assert rows[1]["attention_decode_speedup_vs_full"] == 4.2
    assert rows[1]["end_to_end_prefill_speedup_vs_full"] == 340.0 / 140.0
    assert rows[1]["end_to_end_decode_speedup_vs_full"] == 72.0 / 40.0


def test_attention_timing_rejects_different_prompt_tokens() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real = deepcopy(real)
    real["measurements"]["131072"]["prompts"][1]["token_sha256"] = "different"

    with pytest.raises(ValueError, match="prompt token hashes differ"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_different_source_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real = deepcopy(real)
    real["benchmark_identity"]["source"]["source_sha256"] = "different"

    with pytest.raises(ValueError, match="benchmark_identity"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_legacy_result_without_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    del real["benchmark_identity"]

    with pytest.raises(ValueError, match="legacy results cannot be validated"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_legacy_dummy_without_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    del dummy["benchmark_identity"]

    with pytest.raises(ValueError, match="legacy results cannot be validated"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_nonpositive_difference() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=90.0, decode=29.0)

    with pytest.raises(ValueError, match="not slower than the dummy control"):
        summarize_attention_time(dummy, [("lod", real)])


def test_source_identity_includes_uncommitted_contents(tmp_path: Path) -> None:
    for directory in ("benchmarks", "integrations/vllm_lod", "lod_attention"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    source = tmp_path / "lod_attention" / "kernel.py"
    source.write_text("VALUE = 1\n")

    first = source_identity(tmp_path)
    source.write_text("VALUE = 2\n")
    second = source_identity(tmp_path)

    assert first["source_file_count"] == second["source_file_count"] == 3
    assert first["source_sha256"] != second["source_sha256"]
