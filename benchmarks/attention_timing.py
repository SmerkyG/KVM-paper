"""Estimate attention-core time with a matched dummy-attention control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any


MATCHED_FIELDS = (
    "checkpoint",
    "dataset",
    "dataset_revision",
    "batch_size",
    "speed_samples",
    "tensor_parallel_size",
    "decode_context_parallel_size",
    "dcp_comm_backend",
    "decode_tokens",
    "decode_input_policy",
    "synchronized_decode",
    "seed",
    "scheduler_chunk_tokens",
    "scheduler_budget_tokens",
    "scheduler_cls",
    "hostname",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "full_attention_backend",
    "max_model_len",
    "max_num_seqs",
    "enable_prefix_caching",
    "kimi_gfx942_int4_moe",
    "weight_cache",
    "weight_cache_id",
    "timing_protocol",
    "allow_experimental_environment",
    "speculative_model",
    "num_speculative_tokens",
)

_MODE_SPECIFIC_ENVIRONMENT = {
    "LOD_BENCHMARK_DUMMY_ATTENTION",
    "VLLM_KIMI_DENSE_GLUON",
    "VLLM_LOD_ENABLED",
    "VLLM_LOD_MODE",
}


def _runtime_identity(result: dict[str, Any], *, name: str) -> dict[str, Any]:
    identity = result.get("benchmark_identity")
    if not identity or not identity.get("runtime"):
        raise ValueError(
            f"{name} must contain benchmark_identity.runtime; "
            "legacy results cannot be validated for subtraction"
        )
    return identity["runtime"]


def _prompt_hashes(measurement: dict[str, Any]) -> list[str]:
    return [
        ":".join(
            filter(
                None,
                (prompt["token_sha256"], prompt.get("trace_token_sha256")),
            )
        )
        for prompt in measurement["prompts"]
    ]


def _matched_environment(result: dict[str, Any], *, name: str) -> dict[str, str]:
    environment = result.get("benchmark_environment")
    if not isinstance(environment, dict):
        raise ValueError(f"{name} does not record benchmark_environment")
    return {
        key: value
        for key, value in environment.items()
        if key not in _MODE_SPECIFIC_ENVIRONMENT
    }


def _validate_worker_mode(result: dict[str, Any], *, name: str) -> None:
    audits = result.get("worker_attention_audit")
    if not isinstance(audits, list) or not audits:
        raise ValueError(f"{name} does not record a worker attention audit")
    if len(audits) != int(result.get("tensor_parallel_size", 0)):
        raise ValueError(f"{name} does not audit every tensor-parallel worker")
    expected_lod = result.get("mode") != "full"
    expected_dummy = bool(result.get("dummy_attention", False))
    engine_audits: list[tuple[str, int | None]] = []
    kimi_module_audits: list[str] = []
    for audit in audits:
        actual = (
            bool(audit.get("lod_runtime")),
            int(audit.get("lod_pool_count", 0)) > 0,
            int(audit.get("attached_lod_layers", 0)) > 0,
        )
        if actual != (expected_lod, expected_lod, expected_lod):
            raise ValueError(
                f"{name} worker attention audit disagrees with its mode: {audit!r}"
            )
        attention_layers = int(audit.get("attention_layer_count", 0))
        dummy_layers = int(audit.get("dummy_attention_layers", 0))
        if attention_layers <= 0:
            raise ValueError(f"{name} did not audit any attention layers: {audit!r}")
        actual_dummy = attention_layers > 0 and dummy_layers == attention_layers
        if actual_dummy != expected_dummy:
            raise ValueError(
                f"{name} worker dummy-attention audit disagrees with its mode: "
                f"{audit!r}"
            )
        configurations = audit.get("lod_engine_configurations")
        group_size = audit.get("lod_cross_layer_prefill_group_size")
        if expected_lod:
            if not isinstance(configurations, list) or len(configurations) != 1:
                raise ValueError(
                    f"{name} does not record one uniform LoD engine geometry: "
                    f"{audit!r}"
                )
            configuration = configurations[0]
            if (
                configuration.get("mode") != result.get("mode")
                or int(configuration.get("prefill_routes", -1)) != 8
                or int(configuration.get("decode_routes", -1)) != 8
                or int(configuration.get("prefill_chunk_len", -1)) != 16_384
                or int(configuration.get("prefill_state_update_len", -1))
                != 16_384
                or int(configuration.get("decode_state_update_len", -1)) != 256
            ):
                raise ValueError(
                    f"{name} did not execute the canonical LoD geometry: "
                    f"{configuration!r}"
                )
            if not isinstance(group_size, int) or group_size < 1:
                raise ValueError(
                    f"{name} does not record its cross-layer prefill group"
                )
        elif configurations != [] or group_size is not None:
            raise ValueError(f"{name} dense worker unexpectedly owns LoD engines")
        loaded_kimi_modules = audit.get("loaded_kimi_lod_modules", [])
        is_kimi = (
            "kimi" in str(audit.get("model_class", "")).lower()
            or "kimi" in str(result.get("checkpoint", "")).lower()
        )
        if expected_lod and is_kimi:
            if not isinstance(loaded_kimi_modules, list) or not loaded_kimi_modules:
                raise ValueError(
                    f"{name} does not identify its loaded Kimi AITER module"
                )
            for module in loaded_kimi_modules:
                if not isinstance(module, dict) or not all(
                    module.get(field)
                    for field in ("module", "path", "sha256", "build_manifest_sha256")
                ):
                    raise ValueError(
                        f"{name} has an incomplete Kimi AITER module identity"
                    )
            requires_fused_prefill = max(
                map(int, result.get("measurements", {"0": None}))
            ) > int(configuration["prefill_chunk_len"])
            if requires_fused_prefill:
                fused_modules = [
                    module
                    for module in loaded_kimi_modules
                    if "_asyncbias_" in str(module.get("module", ""))
                ]
                if not fused_modules:
                    raise ValueError(
                        f"{name} did not load its fused Kimi prefill route/coarse "
                        "binary"
                    )
                expected_route_flags = {
                    "CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0",
                    "CK_TILE_FMHA_ROUTE_TOPK": "8",
                    "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0",
                    "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "0",
                }
                if any(
                    module.get("route_build_flags") != expected_route_flags
                    for module in fused_modules
                ):
                    raise ValueError(
                        f"{name} fused Kimi prefill binary was built with the "
                        f"wrong route flags: {fused_modules!r}"
                    )
        kimi_module_audits.append(
            json.dumps(loaded_kimi_modules, sort_keys=True, separators=(",", ":"))
        )
        engine_audits.append(
            (
                json.dumps(configurations, sort_keys=True, separators=(",", ":")),
                group_size,
            )
        )
        expected_dense_gluon = (
            result.get("benchmark_environment", {}).get(
                "VLLM_KIMI_DENSE_GLUON", "0"
            )
            == "1"
        )
        if (
            bool(audit.get("dense_gluon_decode_installed", False))
            != expected_dense_gluon
        ):
            raise ValueError(
                f"{name} worker dense-decoder audit disagrees with its "
                f"environment: {audit!r}"
            )
    if len(set(engine_audits)) != 1:
        raise ValueError(f"{name} workers executed different LoD geometries")
    if len(set(kimi_module_audits)) != 1:
        raise ValueError(f"{name} workers loaded different Kimi AITER modules")


def _worker_hardware(result: dict[str, Any], *, name: str) -> list[tuple[Any, ...]]:
    audits = result.get("worker_attention_audit")
    if not isinstance(audits, list) or not audits:
        raise ValueError(f"{name} does not record a worker attention audit")
    fields = (
        "device_name",
        "device_arch",
        "device_total_memory_bytes",
        "torch_hip_version",
    )
    rows = [tuple(audit.get(field) for field in fields) for audit in audits]
    if any(any(value is None for value in row) for row in rows):
        raise ValueError(f"{name} does not record complete worker hardware identity")
    return sorted(rows, key=repr)


def _worker_kernel_identity(
    result: dict[str, Any], *, name: str
) -> list[tuple[Any, ...]]:
    audits = result.get("worker_attention_audit")
    if not isinstance(audits, list) or not audits:
        raise ValueError(f"{name} does not record a worker attention audit")
    rows = [
        (audit.get("aiter_route_source_sha256"),)
        for audit in audits
    ]
    if any(value is None for row in rows for value in row):
        raise ValueError(f"{name} does not record its AITER route source identity")
    if len(set(rows)) != 1:
        raise ValueError(f"{name} workers use different AITER route sources")
    return sorted(rows, key=repr)


def _validate_timing_windows(result: dict[str, Any], *, name: str) -> None:
    if result.get("measure") != "speed":
        raise ValueError(f"{name} is not a speed benchmark")
    protocol = result.get("timing_protocol")
    if not isinstance(protocol, dict) or int(protocol.get("schema", 0)) < 8:
        raise ValueError(f"{name} does not use timing protocol schema 8 or newer")
    required_protocol = {
        "clock": "vllm-request-metrics-plus-perf-counter-wall-check",
        "prefill_completion": "final-kimi-cache-event-before-first-token",
        "cohort_prefill_window": "earliest-scheduled-to-latest-first-token",
        "cohort_decode_window": "latest-first-token-to-latest-last-token",
        "warmup_generations_per_length": 1,
        "decode_steps": "generated_tokens_minus_one",
        "decode_inputs": "prolong-natural-trace-replay",
    }
    if int(protocol["schema"]) >= 9:
        required_protocol["preemptions"] = "recorded-and-rejected"
    if int(protocol["schema"]) >= 10:
        required_protocol.update(
            prefix_cache_hits="recorded-and-rejected",
            context_panel="shared-max-length-prefix-cohort",
        )
    wrong_protocol = {
        field: (protocol.get(field), expected)
        for field, expected in required_protocol.items()
        if protocol.get(field) != expected
    }
    if wrong_protocol:
        raise ValueError(
            f"{name} records the wrong timing protocol: {wrong_protocol!r}"
        )
    repeats = int(protocol.get("measured_repetitions", 0))
    if repeats < 3:
        raise ValueError(f"{name} must contain at least three measured repetitions")
    batch_size = int(result.get("batch_size", 0))
    speed_samples = int(result.get("speed_samples", 0))
    decode_tokens = int(result.get("decode_tokens", 0))
    if (
        batch_size < 1
        or speed_samples < batch_size
        or speed_samples % batch_size
        or decode_tokens < 1_025
    ):
        raise ValueError(f"{name} records an invalid speed cohort geometry")
    if result.get("decode_input_policy") != "prolong-natural-trace-replay":
        raise ValueError(f"{name} does not use the fixed natural decode trace")
    if batch_size > 1 and not result.get("synchronized_decode"):
        raise ValueError(f"{name} does not synchronize its batched decode cohort")
    if batch_size == 1 and result.get("synchronized_decode"):
        raise ValueError(f"{name} unnecessarily enables the batched decode barrier")
    environment = result.get("benchmark_environment", {})
    if batch_size > 1 and (
        environment.get("LOD_BENCHMARK_SYNCHRONIZED_DECODE") != "1"
        or environment.get("LOD_BENCHMARK_ADMISSION_COHORT") != str(batch_size)
    ):
        raise ValueError(f"{name} lacks the complete batched cohort barrier")
    if environment.get("LOD_BENCHMARK_SYNC_PREFILL_CACHE") != "1":
        raise ValueError(f"{name} does not fence final prefill cache construction")
    cohort_batches = speed_samples // batch_size

    for length, measurement in result["measurements"].items():
        walls = measurement.get("cohort_wall_timings_seconds")
        prefills = measurement.get("cohort_prefill_timings_seconds")
        decodes = measurement.get("cohort_decode_timings_seconds")
        residuals = measurement.get("cohort_unattributed_wall_seconds")
        per_batch_prefills = measurement.get("prefill_timings_seconds")
        per_batch_decodes = measurement.get("decode_timings_seconds")
        if not all(isinstance(values, list) and values for values in (
            walls,
            prefills,
            decodes,
            residuals,
            per_batch_prefills,
            per_batch_decodes,
        )):
            raise ValueError(f"{name} at {length} lacks complete wall-clock checks")
        timing_lists = (
            walls,
            prefills,
            decodes,
            residuals,
            per_batch_prefills,
            per_batch_decodes,
        )
        if {len(values) for values in timing_lists} != {repeats}:
            raise ValueError(f"{name} at {length} has mismatched timing samples")
        for label, values in (
            ("prefill", per_batch_prefills),
            ("decode", per_batch_decodes),
        ):
            numeric = list(map(float, values))
            center = statistics.median(numeric)
            if center <= 0.0 or (max(numeric) - min(numeric)) / center > 0.10:
                raise ValueError(
                    f"{name} at {length} has unstable {label} timings: {numeric!r}"
                )
        for wall, prefill, decode, residual, per_prefill, per_decode in zip(
            *timing_lists, strict=True
        ):
            recomputed = float(wall) - float(prefill) - float(decode)
            if abs(recomputed - float(residual)) > 1e-6:
                raise ValueError(f"{name} at {length} has an inconsistent wall clock")
            if not math.isclose(
                float(per_prefill),
                float(prefill) / cohort_batches,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ) or not math.isclose(
                float(per_decode),
                float(decode) / cohort_batches,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"{name} at {length} has inconsistent per-batch timings"
                )
            tolerance = max(0.05, 0.01 * float(wall))
            if abs(float(residual)) > tolerance:
                raise ValueError(
                    f"{name} at {length} has {float(residual):.6f}s of "
                    "unattributed wall time"
                )
        reported_prefill = float(measurement.get("prefill_seconds", float("nan")))
        expected_prefill = statistics.median(map(float, per_batch_prefills))
        reported_decode = float(
            measurement.get("decode_ms_per_batch_step", float("nan"))
        )
        expected_decode = (
            1_000.0
            * statistics.median(map(float, per_batch_decodes))
            / (decode_tokens - 1)
        )
        if not math.isclose(
            reported_prefill, expected_prefill, rel_tol=1e-9, abs_tol=1e-9
        ) or not math.isclose(
            reported_decode, expected_decode, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"{name} at {length} reports an incorrect median")
        prompts = measurement.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != speed_samples:
            raise ValueError(f"{name} at {length} records the wrong prompt count")
        outputs = measurement.get("output_token_sha256")
        if not isinstance(outputs, list) or any(
            not isinstance(repeat, list) or len(repeat) != speed_samples
            for repeat in outputs
        ) or len(outputs) != repeats:
            raise ValueError(f"{name} at {length} records incomplete output hashes")
        warmup_outputs = measurement.get("warmup_output_token_sha256")
        if (
            not isinstance(warmup_outputs, list)
            or len(warmup_outputs) != speed_samples
            or any(repeat != warmup_outputs for repeat in outputs)
        ):
            raise ValueError(
                f"{name} at {length} did not reproduce the exact warmup trace"
            )
        warmup_batches = measurement.get("warmup_batch_timings")
        batches = measurement.get("measured_batch_timings")
        if (
            not isinstance(warmup_batches, list)
            or len(warmup_batches) != cohort_batches
            or not isinstance(batches, list)
            or len(batches) != repeats
            or any(
                not isinstance(repeat_batches, list)
                or len(repeat_batches) != cohort_batches
                for repeat_batches in batches
            )
        ):
            raise ValueError(
                f"{name} at {length} lacks complete execution-batch timings"
            )
        for repeat_index, repeat_batches in enumerate(batches):
            batch_walls = 0.0
            batch_prefills = 0.0
            batch_decodes = 0.0
            for batch in repeat_batches:
                wall = float(batch.get("wall_elapsed_seconds", float("nan")))
                metric = float(batch.get("metric_window_seconds", float("nan")))
                residual = float(
                    batch.get("unattributed_wall_seconds", float("nan"))
                )
                prefill_window = float(
                    batch.get("prefill_window_seconds", float("nan"))
                )
                decode_window = float(
                    batch.get("decode_window_seconds", float("nan"))
                )
                request_prefills = batch.get("request_prefill_seconds")
                request_decodes = batch.get("request_decode_seconds")
                request_num_preemptions = batch.get("request_num_preemptions")
                request_num_cached_tokens = batch.get("request_num_cached_tokens")
                if (
                    not math.isfinite(wall)
                    or not math.isfinite(metric)
                    or not math.isfinite(residual)
                    or not math.isfinite(prefill_window)
                    or not math.isfinite(decode_window)
                    or wall <= 0.0
                    or metric <= 0.0
                    or prefill_window <= 0.0
                    or decode_window <= 0.0
                    or not isinstance(request_prefills, list)
                    or len(request_prefills) != batch_size
                    or not isinstance(request_decodes, list)
                    or len(request_decodes) != batch_size
                    or (
                        int(protocol["schema"]) >= 9
                        and (
                            not isinstance(request_num_preemptions, list)
                            or len(request_num_preemptions) != batch_size
                        )
                    )
                    or (
                        int(protocol["schema"]) >= 10
                        and (
                            not isinstance(request_num_cached_tokens, list)
                            or len(request_num_cached_tokens) != batch_size
                        )
                    )
                ):
                    raise ValueError(
                        f"{name} at {length} has an invalid execution-batch record"
                    )
                if int(protocol["schema"]) >= 9 and any(
                    int(value) != 0 for value in request_num_preemptions
                ):
                    raise ValueError(
                        f"{name} at {length} contains a preempted request"
                    )
                if int(protocol["schema"]) >= 10 and any(
                    int(value) != 0 for value in request_num_cached_tokens
                ):
                    raise ValueError(
                        f"{name} at {length} contains a prefix-cache hit"
                    )
                if abs((wall - metric) - residual) > 1e-6:
                    raise ValueError(
                        f"{name} at {length} has an inconsistent batch wall clock"
                    )
                if abs((prefill_window + decode_window) - metric) > 1e-6:
                    raise ValueError(
                        f"{name} at {length} has an inconsistent batch split"
                    )
                batch_walls += wall
                batch_prefills += prefill_window
                batch_decodes += decode_window
            if not math.isclose(
                batch_walls,
                float(walls[repeat_index]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ) or not math.isclose(
                batch_prefills,
                float(prefills[repeat_index]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ) or not math.isclose(
                batch_decodes,
                float(decodes[repeat_index]),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"{name} at {length} execution batches do not reconstruct "
                    "the reported cohort timing"
                )
        if result.get("decode_input_policy") == "prolong-natural-trace-replay":
            if not measurement.get("greedy_output_identical", False):
                raise ValueError(
                    f"{name} at {length} did not reproduce its fixed decode trace"
                )
        if result.get("synchronized_decode") and int(result.get("batch_size", 1)) > 1:
            for repeat in batches:
                for batch in repeat:
                    spread = float(
                        batch.get("last_token_spread_seconds", float("inf"))
                    )
                    overlap = float(
                        batch.get(
                            "all_requests_live_overlap_seconds",
                            float("-inf"),
                        )
                    )
                    decode_window = overlap + spread
                    tolerance = max(0.05, 0.02 * decode_window)
                    if overlap <= 0 or spread > tolerance:
                        raise ValueError(
                            f"{name} at {length} did not sustain synchronized "
                            f"decode; all-live overlap was {overlap:.6f}s and "
                            f"last-token spread was {spread:.6f}s"
                        )
    if int(protocol["schema"]) >= 10:
        panel_hashes = [
            [prompt.get("panel_token_sha256") for prompt in measurement["prompts"]]
            for measurement in result["measurements"].values()
        ]
        if not panel_hashes or any(
            hashes != panel_hashes[0] or any(value is None for value in hashes)
            for hashes in panel_hashes
        ):
            raise ValueError(f"{name} does not use one shared context panel")


def validate_pair(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_name: str,
) -> None:
    _validate_worker_mode(reference, name="reference")
    _validate_worker_mode(candidate, name=candidate_name)
    _validate_timing_windows(reference, name="reference")
    _validate_timing_windows(candidate, name=candidate_name)
    reference_runtime = _runtime_identity(reference, name="reference")
    candidate_runtime = _runtime_identity(candidate, name=candidate_name)
    if reference_runtime != candidate_runtime:
        raise ValueError(
            f"reference and {candidate_name} runtime identities differ: "
            f"{reference_runtime!r} != {candidate_runtime!r}"
        )
    reference_hardware = _worker_hardware(reference, name="reference")
    candidate_hardware = _worker_hardware(candidate, name=candidate_name)
    if reference_hardware != candidate_hardware:
        raise ValueError(
            f"reference and {candidate_name} worker hardware differs: "
            f"{reference_hardware!r} != {candidate_hardware!r}"
        )
    reference_kernels = _worker_kernel_identity(reference, name="reference")
    candidate_kernels = _worker_kernel_identity(candidate, name=candidate_name)
    if reference_kernels != candidate_kernels:
        raise ValueError(
            f"reference and {candidate_name} AITER route sources differ: "
            f"{reference_kernels!r} != {candidate_kernels!r}"
        )
    reference_layer_counts = sorted(
        int(audit.get("attention_layer_count", -1))
        for audit in reference["worker_attention_audit"]
    )
    candidate_layer_counts = sorted(
        int(audit.get("attention_layer_count", -1))
        for audit in candidate["worker_attention_audit"]
    )
    if reference_layer_counts != candidate_layer_counts:
        raise ValueError(
            f"reference and {candidate_name} attention-layer topology differs: "
            f"{reference_layer_counts!r} != {candidate_layer_counts!r}"
        )
    mismatches = {
        field: (reference.get(field), candidate.get(field))
        for field in MATCHED_FIELDS
        if reference.get(field) != candidate.get(field)
    }
    if mismatches:
        details = ", ".join(
            f"{field}={dummy_value!r}/{real_value!r}"
            for field, (dummy_value, real_value) in mismatches.items()
        )
        raise ValueError(
            f"reference and {candidate_name} configurations differ: {details}"
        )
    reference_environment = _matched_environment(reference, name="reference")
    candidate_environment = _matched_environment(candidate, name=candidate_name)
    if reference_environment != candidate_environment:
        raise ValueError(
            f"reference and {candidate_name} benchmark environments differ: "
            f"{reference_environment!r} != {candidate_environment!r}"
        )
    # The dummy control and real dense run must use the same dense attention
    # implementation.  LoD runs do not execute that decoder, so this setting
    # is intentionally allowed to differ only when one side is non-dense.
    if reference.get("mode") == candidate.get("mode") == "full":
        dense_backend = "VLLM_KIMI_DENSE_GLUON"
        reference_dense = reference.get("benchmark_environment", {}).get(
            dense_backend, "0"
        )
        candidate_dense = candidate.get("benchmark_environment", {}).get(
            dense_backend, "0"
        )
        if reference_dense != candidate_dense:
            raise ValueError(
                f"reference and {candidate_name} dense decode backends differ: "
                f"{reference_dense!r} != {candidate_dense!r}"
            )

    reference_lengths = set(reference["measurements"])
    candidate_lengths = set(candidate["measurements"])
    if reference_lengths != candidate_lengths:
        raise ValueError(
            f"reference and {candidate_name} context lengths differ: "
            f"{sorted(reference_lengths)} != {sorted(candidate_lengths)}"
        )
    for length in sorted(reference_lengths, key=int):
        reference_hashes = _prompt_hashes(reference["measurements"][length])
        candidate_hashes = _prompt_hashes(candidate["measurements"][length])
        if reference_hashes != candidate_hashes:
            raise ValueError(
                f"reference and {candidate_name} prompt token hashes differ "
                f"at {length}"
            )
        if reference.get("decode_input_policy") == "prolong-natural-trace-replay":
            reference_outputs = reference["measurements"][length].get(
                "output_token_sha256"
            )
            candidate_outputs = candidate["measurements"][length].get(
                "output_token_sha256"
            )
            if reference_outputs != candidate_outputs:
                raise ValueError(
                    f"reference and {candidate_name} fixed decode outputs differ "
                    f"at {length}"
                )


def summarize_attention_time(
    dummy: dict[str, Any],
    reals: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Return strict matched real-minus-dummy timing estimates."""
    if not dummy.get("dummy_attention"):
        raise ValueError("dummy result was not recorded with --dummy-attention")
    if dummy.get("measure") != "speed" or dummy.get("mode") != "full":
        raise ValueError("dummy result must be a full-mode speed benchmark")
    if not reals:
        raise ValueError("at least one real result is required")
    _runtime_identity(dummy, name="dummy")
    all_results = [dummy, *(value for _, value in reals)]
    if any(
        result.get("kimi_gfx942_int4_moe", False)
        or any(
            int(audit.get("dynamic_moe_layer_count", 0)) > 0
            for audit in result.get("worker_attention_audit", ())
        )
        for result in all_results
    ):
        raise ValueError(
            "real-minus-dummy attention timing is not valid for dynamic-MoE "
            "models because the dummy changes downstream expert routing; use "
            "matched end-to-end timing and diagnostic kernel profiles instead"
        )

    result: dict[str, Any] = {
        "method": "matched-real-minus-dummy-attention",
        "benchmark_identity": dummy["benchmark_identity"],
        "checkpoint": dummy["checkpoint"],
        "batch_size": dummy["batch_size"],
        "tensor_parallel_size": dummy["tensor_parallel_size"],
        "decode_tokens": dummy["decode_tokens"],
        "measurements": {},
    }
    for real_name, real in reals:
        if real.get("dummy_attention", False):
            raise ValueError(f"{real_name} is another dummy result")
        validate_pair(dummy, real, candidate_name=real_name)

    full_runs = [(name, value) for name, value in reals if value["mode"] == "full"]
    if len(full_runs) != 1:
        raise ValueError("exactly one real full-attention result is required")

    for length in sorted(dummy["measurements"], key=int):
        dummy_measurement = dummy["measurements"][length]
        dummy_prefill = float(dummy_measurement["prefill_seconds"])
        dummy_decode = float(dummy_measurement["decode_ms_per_batch_step"])
        dummy_prefill_samples = list(
            map(float, dummy_measurement["prefill_timings_seconds"])
        )
        dummy_decode_samples = [
            1_000.0 * float(value) / (int(dummy["decode_tokens"]) - 1)
            for value in dummy_measurement["decode_timings_seconds"]
        ]
        rows: list[dict[str, Any]] = []
        for real_name, real in reals:
            measurement = real["measurements"][length]
            real_prefill = float(measurement["prefill_seconds"])
            real_decode = float(measurement["decode_ms_per_batch_step"])
            attention_prefill = real_prefill - dummy_prefill
            attention_decode = real_decode - dummy_decode
            real_prefill_samples = list(
                map(float, measurement["prefill_timings_seconds"])
            )
            real_decode_samples = [
                1_000.0 * float(value) / (int(real["decode_tokens"]) - 1)
                for value in measurement["decode_timings_seconds"]
            ]
            prefill_bounds = (
                min(real_prefill_samples) - max(dummy_prefill_samples),
                max(real_prefill_samples) - min(dummy_prefill_samples),
            )
            decode_bounds = (
                min(real_decode_samples) - max(dummy_decode_samples),
                max(real_decode_samples) - min(dummy_decode_samples),
            )
            if (
                attention_prefill <= 0.0
                or attention_decode <= 0.0
                or prefill_bounds[0] <= 0.0
                or decode_bounds[0] <= 0.0
            ):
                raise ValueError(
                    f"{real_name} at {length} is not slower than the dummy control "
                    f"with a positive observed margin: prefill delta={attention_prefill}, "
                    f"observed bounds={prefill_bounds}; decode delta={attention_decode}, "
                    f"observed bounds={decode_bounds}"
                )
            rows.append(
                {
                    "name": real_name,
                    "mode": real["mode"],
                    "benchmark_identity": real["benchmark_identity"],
                    "real_prefill_seconds": real_prefill,
                    "real_decode_ms_per_batch_step": real_decode,
                    "attention_prefill_seconds": attention_prefill,
                    "attention_prefill_observed_bounds_seconds": list(
                        prefill_bounds
                    ),
                    "attention_decode_ms_per_batch_step": attention_decode,
                    "attention_decode_observed_bounds_ms_per_batch_step": list(
                        decode_bounds
                    ),
                }
            )
        full_row = next(row for row in rows if row["mode"] == "full")
        for row in rows:
            row["attention_prefill_speedup_vs_full"] = (
                full_row["attention_prefill_seconds"]
                / row["attention_prefill_seconds"]
            )
            row["attention_prefill_speedup_observed_bounds_vs_full"] = [
                full_row["attention_prefill_observed_bounds_seconds"][0]
                / row["attention_prefill_observed_bounds_seconds"][1],
                full_row["attention_prefill_observed_bounds_seconds"][1]
                / row["attention_prefill_observed_bounds_seconds"][0],
            ]
            row["attention_decode_speedup_vs_full"] = (
                full_row["attention_decode_ms_per_batch_step"]
                / row["attention_decode_ms_per_batch_step"]
            )
            row["attention_decode_speedup_observed_bounds_vs_full"] = [
                full_row["attention_decode_observed_bounds_ms_per_batch_step"][0]
                / row["attention_decode_observed_bounds_ms_per_batch_step"][1],
                full_row["attention_decode_observed_bounds_ms_per_batch_step"][1]
                / row["attention_decode_observed_bounds_ms_per_batch_step"][0],
            ]
            row["end_to_end_prefill_speedup_vs_full"] = (
                full_row["real_prefill_seconds"] / row["real_prefill_seconds"]
            )
            row["end_to_end_decode_speedup_vs_full"] = (
                full_row["real_decode_ms_per_batch_step"]
                / row["real_decode_ms_per_batch_step"]
            )
        result["measurements"][length] = {
            "dummy_prefill_seconds": dummy_prefill,
            "dummy_decode_ms_per_batch_step": dummy_decode,
            "runs": rows,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", type=Path, required=True)
    parser.add_argument(
        "--real",
        type=Path,
        action="append",
        required=True,
        help="matched real benchmark JSON; repeat for each attention mode",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dummy = json.loads(args.dummy.read_text())
    reals = [(path.stem, json.loads(path.read_text())) for path in args.real]
    result = summarize_attention_time(dummy, reals)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output)


if __name__ == "__main__":
    main()
