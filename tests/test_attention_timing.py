from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks._identity import source_identity
from benchmarks.attention_timing import summarize_attention_time
from benchmarks.speed_comparison import summarize_speed_comparison


def _result(*, mode: str, dummy: bool, prefill: float, decode: float) -> dict:
    decode_seconds = decode * 1024 / 1000
    residual = 0.001
    wall = prefill + decode_seconds + residual
    is_lod = mode != "full"
    return {
        "benchmark_identity": {
            "schema": 1,
            "source": {
                "git_commit": "deadbeef",
                "source_sha256": "source-hash",
                "source_file_count": 42,
            },
            "runtime": {
                "executable": "/test/python",
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
        "decode_context_parallel_size": 1,
        "dcp_comm_backend": "ag_rs",
        "decode_tokens": 1025,
        "decode_input_policy": "prolong-natural-trace-replay",
        "synchronized_decode": True,
        "seed": 0,
        "scheduler_chunk_tokens": 16_384,
        "scheduler_budget_tokens": 16_392,
        "scheduler_cls": "test.Scheduler",
        "hostname": "test-host",
        "gpu_memory_utilization": 0.8,
        "kv_cache_memory_bytes": 268_435_456,
        "full_attention_backend": "test.backend",
        "max_model_len": 132_113,
        "max_num_seqs": 8,
        "enable_prefix_caching": False,
        "kimi_gfx942_int4_moe": False,
        "weight_cache": True,
        "weight_cache_id": "test-cache",
        "timing_protocol": {
            "schema": 10,
            "clock": "vllm-request-metrics-plus-perf-counter-wall-check",
            "decode_inputs": "prolong-natural-trace-replay",
            "measured_repetitions": 3,
            "prefill_completion": "final-kimi-cache-event-before-first-token",
            "cohort_prefill_window": "earliest-scheduled-to-latest-first-token",
            "cohort_decode_window": "latest-first-token-to-latest-last-token",
            "warmup_generations_per_length": 1,
            "decode_steps": "generated_tokens_minus_one",
            "preemptions": "recorded-and-rejected",
            "prefix_cache_hits": "recorded-and-rejected",
            "context_panel": "shared-max-length-prefix-cohort",
        },
        "allow_experimental_environment": False,
        "speculative_model": None,
        "num_speculative_tokens": None,
        "benchmark_environment": {
            "AITER_CONFIG_FMOE": "test.csv",
            "LOD_BENCHMARK_SYNC_PREFILL_CACHE": "1",
            "LOD_BENCHMARK_SYNCHRONIZED_DECODE": "1",
            "LOD_BENCHMARK_ADMISSION_COHORT": "8",
            "VLLM_LOD_ENABLED": "1" if is_lod else "0",
            "VLLM_LOD_MODE": "two-tier",
            **(
                {"LOD_BENCHMARK_DUMMY_ATTENTION": "1"}
                if dummy
                else {}
            ),
        },
        "worker_attention_audit": [
            {
                "lod_runtime": is_lod,
                "lod_pool_count": 24 if is_lod else 0,
                "attached_lod_layers": 24 if is_lod else 0,
                "lod_engine_configurations": (
                    [
                        {
                            "family": "test",
                            "mode": mode,
                            "levels": 2,
                            "kv_bits": 0,
                            "prefill_routes": 8,
                            "decode_routes": 8,
                            "prefill_chunk_len": 16_384,
                            "prefill_state_update_len": 16_384,
                            "decode_state_update_len": 256,
                            "local_len": 512,
                            "leaf_page_size": 16,
                            "leaf_block_m": 16,
                            "leaf_block_n": 32,
                            "leaf_num_warps": 2,
                            "prefill_fused_route_coarse": True,
                            "decode_fused_state_route": True,
                            "max_open_centroid_leaves": None,
                            "leaf_seal_capacity": 1_024,
                        }
                    ]
                    if is_lod
                    else []
                ),
                "lod_cross_layer_prefill_group_size": 24 if is_lod else None,
                "attention_layer_count": 24,
                "dummy_attention_layers": 24 if dummy else 0,
                "dynamic_moe_layer_count": 0,
                "attention_impl_classes": (
                    ["DummyAttentionImpl"] * 24
                    if dummy
                    else ["TestAttentionImpl"] * 24
                ),
                "dense_gluon_decode_installed": False,
                "aiter_route_source_sha256": "test-aiter-route-source",
                "device_name": "test-gpu",
                "device_arch": "test-arch",
                "device_total_memory_bytes": 1234,
                "torch_hip_version": "test-hip",
            }
        ],
        "measurements": {
            "131072": {
                "prefill_seconds": prefill,
                "decode_ms_per_batch_step": decode,
                "cohort_prefill_timings_seconds": [prefill] * 3,
                "cohort_decode_timings_seconds": [decode_seconds] * 3,
                "cohort_wall_timings_seconds": [wall] * 3,
                "cohort_unattributed_wall_seconds": [residual] * 3,
                "prefill_timings_seconds": [prefill] * 3,
                "decode_timings_seconds": [decode_seconds] * 3,
                "measured_batch_timings": [
                    [
                        {
                            "wall_elapsed_seconds": wall,
                            "metric_window_seconds": prefill + decode_seconds,
                            "prefill_window_seconds": prefill,
                            "decode_window_seconds": decode_seconds,
                            "unattributed_wall_seconds": residual,
                            "first_token_spread_seconds": 10.0,
                            "last_token_spread_seconds": 0.001,
                            "all_requests_live_overlap_seconds": decode_seconds,
                            "request_num_preemptions": [0] * 8,
                            "request_num_cached_tokens": [0] * 8,
                            "request_prefill_seconds": [prefill] * 8,
                            "request_decode_seconds": [decode_seconds] * 8,
                        }
                    ]
                ] * 3,
                "greedy_output_identical": True,
                "warmup_output_token_sha256": ["fixed-output"] * 8,
                "warmup_batch_timings": [
                    {
                        "wall_elapsed_seconds": wall,
                        "metric_window_seconds": prefill + decode_seconds,
                        "prefill_window_seconds": prefill,
                        "decode_window_seconds": decode_seconds,
                        "unattributed_wall_seconds": residual,
                        "first_token_spread_seconds": 10.0,
                        "last_token_spread_seconds": 0.001,
                        "all_requests_live_overlap_seconds": decode_seconds,
                        "request_num_preemptions": [0] * 8,
                        "request_num_cached_tokens": [0] * 8,
                        "request_prefill_seconds": [prefill] * 8,
                        "request_decode_seconds": [decode_seconds] * 8,
                    }
                ],
                "output_token_sha256": [["fixed-output"] * 8] * 3,
                "prompts": [
                    {
                        "token_sha256": f"prompt-{index}",
                        "trace_token_sha256": f"trace-{index}",
                        "panel_token_sha256": f"panel-{index}",
                    }
                    for index in range(8)
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


def test_end_to_end_speed_comparison_requires_identical_source() -> None:
    full = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)
    lod["benchmark_identity"]["source"]["source_sha256"] = "different-source"

    with pytest.raises(ValueError, match="exact same source identity"):
        summarize_speed_comparison(full, lod)


def test_attention_timing_rejects_different_prompt_tokens() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real = deepcopy(real)
    real["measurements"]["131072"]["prompts"][1]["token_sha256"] = "different"

    with pytest.raises(ValueError, match="prompt token hashes differ"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_allows_different_source_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real = deepcopy(real)
    real["benchmark_identity"]["source"]["source_sha256"] = "different"

    result = summarize_attention_time(dummy, [("full", real)])

    row = result["measurements"]["131072"]["runs"][0]
    assert row["benchmark_identity"]["source"]["source_sha256"] == "different"


def test_attention_timing_rejects_different_aiter_route_source() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["worker_attention_audit"][0]["aiter_route_source_sha256"] = "different"

    with pytest.raises(ValueError, match="AITER route sources differ"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_kimi_lod_without_loaded_binary_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)
    for result in (dummy, lod):
        result["checkpoint"] = "test/kimi-k3"
        result["worker_attention_audit"][0]["model_class"] = "KimiK3ForCausalLM"

    with pytest.raises(ValueError, match="loaded Kimi AITER module"):
        summarize_attention_time(dummy, [("lod", lod)])


def test_attention_timing_rejects_incomplete_loaded_binary_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)
    for result in (dummy, lod):
        result["checkpoint"] = "test/kimi-k3"
        result["worker_attention_audit"][0]["model_class"] = "KimiK3ForCausalLM"
    lod["worker_attention_audit"][0]["loaded_kimi_lod_modules"] = [
        {
            "module": "mha_fwd_lod_kimi",
            "path": "/cache/mha_fwd_lod_kimi.so",
            "sha256": "binary-digest",
            "build_manifest_sha256": None,
        }
    ]

    with pytest.raises(ValueError, match="incomplete Kimi AITER module identity"):
        summarize_attention_time(dummy, [("lod", lod)])


def test_attention_timing_requires_fused_kimi_binary_beyond_exact_prefix() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)
    for result in (dummy, lod):
        result["checkpoint"] = "test/kimi-k3"
        result["worker_attention_audit"][0]["model_class"] = "KimiK3ForCausalLM"
    lod["worker_attention_audit"][0]["loaded_kimi_lod_modules"] = [
        {
            "module": "mha_fwd_lod_kimi_d192v128_v3",
            "path": "/cache/mha_fwd_lod_kimi_d192v128_v3.so",
            "sha256": "binary-digest",
            "build_manifest_sha256": "manifest-digest",
            "route_build_flags": {},
        }
    ]

    with pytest.raises(ValueError, match="fused Kimi prefill route/coarse binary"):
        summarize_attention_time(dummy, [("lod", lod)])


def test_attention_timing_rejects_wrong_fused_kimi_build_flags() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)
    for result in (dummy, lod):
        result["checkpoint"] = "test/kimi-k3"
        result["worker_attention_audit"][0]["model_class"] = "KimiK3ForCausalLM"
    lod["worker_attention_audit"][0]["loaded_kimi_lod_modules"] = [
        {
            "module": "mha_fwd_lod_kimi_d192v128_asyncbias_v9",
            "path": "/cache/mha_fwd_lod_kimi_d192v128_asyncbias_v9.so",
            "sha256": "binary-digest",
            "build_manifest_sha256": "manifest-digest",
            "route_build_flags": {
                "CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
                "CK_TILE_FMHA_ROUTE_TOPK": "8",
                "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "1",
                "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "0",
            },
        }
    ]

    with pytest.raises(ValueError, match="wrong route flags"):
        summarize_attention_time(dummy, [("lod", lod)])


def test_attention_timing_rejects_different_loaded_binaries_across_ranks() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    lod = _result(mode="two-tier", dummy=False, prefill=140.0, decode=40.0)
    for result in (dummy, lod):
        result["checkpoint"] = "test/kimi-k3"
        result["tensor_parallel_size"] = 2
        result["worker_attention_audit"] = [
            deepcopy(result["worker_attention_audit"][0]),
            deepcopy(result["worker_attention_audit"][0]),
        ]
        for audit in result["worker_attention_audit"]:
            audit["model_class"] = "KimiK3ForCausalLM"
    for rank, audit in enumerate(lod["worker_attention_audit"]):
        audit["loaded_kimi_lod_modules"] = [
            {
                "module": "mha_fwd_lod_kimi_d192v128_asyncbias_v9",
                "path": "/cache/mha_fwd_lod_kimi_d192v128_asyncbias_v9.so",
                "sha256": f"binary-digest-{rank}",
                "build_manifest_sha256": "manifest-digest",
                "route_build_flags": {
                    "CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
                    "CK_TILE_FMHA_ROUTE_TOPK": "8",
                    "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0",
                    "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "0",
                },
            }
        ]

    with pytest.raises(ValueError, match="different Kimi AITER modules"):
        summarize_attention_time(dummy, [("lod", lod)])


def test_attention_timing_rejects_different_runtime_identity() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real = deepcopy(real)
    real["benchmark_identity"]["runtime"]["packages"]["torch"] = "different"

    with pytest.raises(ValueError, match="runtime identities differ"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_different_runtime_environment() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["benchmark_environment"]["AITER_CONFIG_FMOE"] = "different.csv"

    with pytest.raises(ValueError, match="benchmark environments differ"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_different_dense_decode_backend() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["benchmark_environment"]["VLLM_KIMI_DENSE_GLUON"] = "1"
    real["worker_attention_audit"][0]["dense_gluon_decode_installed"] = True

    with pytest.raises(ValueError, match="dense decode backends differ"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_unverified_dummy_backend() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    dummy["worker_attention_audit"][0]["dummy_attention_layers"] = 0

    with pytest.raises(ValueError, match="dummy-attention audit"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_dynamic_moe_subtraction() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    dummy["worker_attention_audit"][0]["dynamic_moe_layer_count"] = 1
    real["worker_attention_audit"][0]["dynamic_moe_layer_count"] = 1

    with pytest.raises(ValueError, match="not valid for dynamic-MoE"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_incorrect_reported_median() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["decode_ms_per_batch_step"] = 73.0

    with pytest.raises(ValueError, match="incorrect median"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_shared_but_wrong_protocol() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    dummy["timing_protocol"]["decode_steps"] = "generated-tokens"
    real["timing_protocol"]["decode_steps"] = "generated-tokens"

    with pytest.raises(ValueError, match="wrong timing protocol"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_batch_totals_not_reconstructed() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["measured_batch_timings"][0][0][
        "prefill_window_seconds"
    ] += 1.0

    with pytest.raises(ValueError, match="inconsistent batch split"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_incomplete_batch_admission_barrier() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    del real["benchmark_environment"]["LOD_BENCHMARK_ADMISSION_COHORT"]

    with pytest.raises(ValueError, match="complete batched cohort barrier"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_preempted_request() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["measured_batch_timings"][0][0][
        "request_num_preemptions"
    ][3] = 1

    with pytest.raises(ValueError, match="preempted request"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_prefix_cache_hit() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["measured_batch_timings"][0][0][
        "request_num_cached_tokens"
    ][3] = 16

    with pytest.raises(ValueError, match="prefix-cache hit"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_single_repetition() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["timing_protocol"]["measured_repetitions"] = 1

    with pytest.raises(ValueError, match="at least three measured repetitions"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_unstable_repetitions() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["prefill_timings_seconds"] = [300.0, 340.0, 380.0]

    with pytest.raises(ValueError, match="unstable prefill timings"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_unexplained_wall_time() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["cohort_wall_timings_seconds"][0] += 10.0
    real["measurements"]["131072"]["cohort_unattributed_wall_seconds"][0] += 10.0

    with pytest.raises(ValueError, match="unattributed wall time"):
        summarize_attention_time(dummy, [("full", real)])


def test_attention_timing_rejects_fixed_trace_output_mismatch() -> None:
    dummy = _result(mode="full", dummy=True, prefill=100.0, decode=30.0)
    real = _result(mode="full", dummy=False, prefill=340.0, decode=72.0)
    real["measurements"]["131072"]["output_token_sha256"][0][0] = "different"

    with pytest.raises(ValueError, match="exact warmup trace"):
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
    launcher = tmp_path / "benchmarks" / "run.sh"
    launcher.write_text("#!/bin/sh\nexec python \"$@\"\n")

    first = source_identity(tmp_path)
    source.write_text("VALUE = 2\n")
    second = source_identity(tmp_path)

    assert first["source_file_count"] == second["source_file_count"] == 4
    assert first["source_sha256"] != second["source_sha256"]
