from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks._vllm import (
    LOD_SCHEDULER,
    MODES,
    default_gpu_memory_utilization,
    is_kimi_k3,
    llm_kwargs,
    scheduler_budget,
)
from benchmarks.longbench_v2 import (
    extract_answer,
    summarize,
    truncate_prompt,
)
from benchmarks.prolong import (
    QUALITY_DOCUMENT_INDICES,
    comma_separated_ints,
    configure_synchronized_decode_environment,
    document_digest,
    make_speed_trace_panel,
    select_quality_prompts,
    speculative_counters,
    timed_generate_cohort,
    validate_release_environment,
    validate_worker_attention_mode,
)


@pytest.mark.parametrize(
    "retain,free_gib,reclaimed",
    [(False, 100, True), (True, 100, False), (True, 8, False), (True, 7, True)],
)
def test_prolong_warmup_allocator_policy(monkeypatch, retain, free_gib, reclaimed):
    import torch
    from benchmarks.prolong import release_worker_allocator_cache

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("wait"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("reclaim"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (free_gib * 1024**3, 256 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 2)
    result = release_worker_allocator_cache(object(), retain=retain)
    assert calls == (["wait", "reclaim"] if reclaimed else ["wait"])
    assert result["retention_requested"] is retain
    assert result["reclaimed"] is reclaimed
    assert result["free_bytes_before"] == free_gib * 1024**3


def test_prolong_synchronized_cohort_requires_resident_cache_capacity():
    from benchmarks.prolong import validate_cohort_capacity

    validate_cohort_capacity([{"max_concurrent_requests": 8.0}], 8)
    with pytest.raises(RuntimeError, match="exceeds native cache capacity"):
        validate_cohort_capacity([{"max_concurrent_requests": 7.21}], 8)
    with pytest.raises(RuntimeError, match="exceeds native cache capacity"):
        validate_cohort_capacity([{"max_concurrent_requests": 9},
                                  {"max_concurrent_requests": 7.99}], 8)


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(prompt.split())))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return ",".join(map(str, token_ids))


class _Metric:
    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value


class _LLMWithMetrics:
    def get_metrics(self) -> list[_Metric]:
        return [
            _Metric("vllm:spec_decode_num_drafts", 12),
            _Metric("vllm:spec_decode_num_drafts", 3),
            _Metric("vllm:spec_decode_num_accepted_tokens", 44),
            _Metric("vllm:unrelated", 99),
        ]


def test_longbench_middle_truncation_and_answer_parsing() -> None:
    text, original, truncated = truncate_prompt(
        _Tokenizer(),
        "zero one two three four five",
        4,
    )
    assert (text, original, truncated) == ("0,1,4,5", 6, True)
    assert extract_answer("The correct answer is (**c**)") == "C"
    assert extract_answer("no parseable choice") is None


def test_longbench_summary_groups_results() -> None:
    records = [
        {
            "correct": True,
            "difficulty": "easy",
            "length": "short",
            "domain": "qa",
        },
        {
            "correct": False,
            "difficulty": "easy",
            "length": "long",
            "domain": "qa",
        },
    ]
    metrics = summarize(records)
    assert metrics["overall"] == {"correct": 1, "count": 2, "accuracy": 0.5}
    assert metrics["domain:qa"]["count"] == 2


def test_offline_benchmark_uses_only_fixed_release_modes() -> None:
    kwargs = llm_kwargs(
        checkpoint="Qwen/Qwen3.8-27B-FP8",
        mode="three-tier-int4",
        max_model_len=65_616,
        batch_size=8,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    assert MODES == (
        "full",
        "two-tier",
        "three-tier-bf16",
        "three-tier-int4",
    )
    assert kwargs["attention_config"] == {"backend": "CUSTOM"}
    assert kwargs["model_impl"] == "vllm"
    assert kwargs["max_num_batched_tokens"] == 16_392
    assert kwargs["long_prefill_token_threshold"] == 16_384
    assert kwargs["scheduler_cls"] == LOD_SCHEDULER
    assert kwargs["language_model_only"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert os.environ["VLLM_LOD_ENABLED"] == "1"

    llm_kwargs(
        checkpoint="Qwen/Qwen3.8-27B-FP8",
        mode="full",
        max_model_len=65_616,
        batch_size=8,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    assert os.environ["VLLM_LOD_ENABLED"] == "0"


@pytest.mark.parametrize(
    "checkpoint",
    ("Qwen/Qwen3.8-27B-FP8", "IFM/K2-Horizon-32B-FP8"),
)
@pytest.mark.parametrize("mode", MODES)
def test_chunk_aligned_scheduler_applies_to_every_release_path(
    checkpoint: str,
    mode: str,
) -> None:
    kwargs = llm_kwargs(
        checkpoint=checkpoint,
        mode=mode,
        max_model_len=131_072,
        batch_size=8,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.8,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
    )
    assert kwargs["scheduler_cls"] == LOD_SCHEDULER
    assert kwargs["max_num_batched_tokens"] == scheduler_budget(8)


def test_memory_targets_cover_each_model_side_lod_pool() -> None:
    assert default_gpu_memory_utilization("Qwen/Qwen3.8-27B-FP8", "two-tier") == 0.7
    assert default_gpu_memory_utilization("IFM/K2-Horizon-32B-FP8", "two-tier") == 0.8
    assert default_gpu_memory_utilization("IFM/K2-Horizon-32B-FP8", "full") == 0.9
    assert (
        default_gpu_memory_utilization(
            "IFM/K2-Horizon-32B-FP8",
            "three-tier-int4",
            quality=True,
        )
        == 0.65
    )


def test_local_kimi_checkpoint_is_identified_from_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "content-addressed-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "kimi_linear"}')

    assert is_kimi_k3(str(checkpoint))
    llm_kwargs(
        checkpoint=str(checkpoint),
        mode="full",
        max_model_len=65_536,
        batch_size=1,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.8,
        full_attention_backend="TRITON_MLA",
        decode_context_parallel_size=8,
    )
    assert os.environ["VLLM_KIMI_DENSE_GLUON"] == "1"


def test_qwen_dflash2_configuration_is_explicit_and_model_limited() -> None:
    kwargs = llm_kwargs(
        checkpoint="Qwen/Qwen3.8-27B-FP8",
        mode="three-tier-bf16",
        max_model_len=65_808,
        batch_size=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
        speculative_model="z-lab/Qwen3.8-27B-DFlash2",
    )
    assert kwargs["speculative_config"] == {
        "method": "dflash",
        "model": "z-lab/Qwen3.8-27B-DFlash2",
        "num_speculative_tokens": 7,
        "attention_backend": "TRITON_ATTN",
    }
    assert kwargs["max_num_batched_tokens"] == 16_448
    assert scheduler_budget(8) == 16_392
    assert scheduler_budget(8, 7) == 16_448
    with pytest.raises(ValueError, match="only with Qwen3.8"):
        llm_kwargs(
            checkpoint="IFM/K2-Horizon-32B-FP8",
            mode="two-tier",
            max_model_len=65_808,
            batch_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            full_attention_backend="ROCM_AITER_UNIFIED_ATTN",
            speculative_model="z-lab/Qwen3.8-27B-DFlash2",
        )


@pytest.mark.parametrize("mode", ["full", "two-tier"])
def test_kimi_graph_prefill_requests_only_fixed_large_shapes(monkeypatch, mode) -> None:
    monkeypatch.setenv("LOD_KIMI_GRAPH_PREFILL", "1")
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    kwargs = llm_kwargs(
        checkpoint="tests/fixtures/kimi-k3-mla-stack", mode=mode,
        max_model_len=65_546, batch_size=8, tensor_parallel_size=8,
        decode_context_parallel_size=8, gpu_memory_utilization=0.8,
        full_attention_backend="TRITON_MLA",
    )
    assert kwargs["compilation_config"] == {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": list(range(1, 9)) + [16_384],
        "max_cudagraph_capture_size": 16_384,
    }
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "1"


def test_benchmark_cli_and_docs_are_public() -> None:
    assert comma_separated_ints("8192,16384") == [8_192, 16_384]
    with pytest.raises(argparse.ArgumentTypeError):
        comma_separated_ints("8192,nope")
    for name, module in (
        ("LONGBENCH_V2.md", "benchmarks.longbench_v2"),
        ("PROLONG.md", "benchmarks.prolong"),
        ("NIAH_S3.md", "benchmarks.niah_s3"),
    ):
        document = (ROOT / "benchmarks" / name).read_text()
        assert f"python -m {module}" in document
        assert "cluster-run" not in document
        assert "## Reproduction requirements" in document
    assert "--seed 0" in (ROOT / "benchmarks" / "PROLONG.md").read_text()
    assert (
        "NumPy's generator seed to `1234`"
        in (ROOT / "benchmarks" / "NIAH_S3.md").read_text()
    )
    assert (
        "no randomized sampling"
        in (ROOT / "benchmarks" / "LONGBENCH_V2.md").read_text()
    )


def test_prolong_collects_speculative_counter_totals() -> None:
    assert speculative_counters(_LLMWithMetrics()) == {
        "vllm:spec_decode_num_drafts": 15,
        "vllm:spec_decode_num_accepted_tokens": 44,
    }


def test_prolong_rejects_worker_attention_mode_mismatch() -> None:
    dense = {
        "lod_runtime": False,
        "lod_pool_count": 0,
        "attached_lod_layers": 0,
        "dense_gluon_decode_installed": False,
    }
    lod = {
        "lod_runtime": True,
        "lod_pool_count": 61,
        "attached_lod_layers": 61,
        "dense_gluon_decode_installed": False,
        "lod_engine_configurations": [
            {
                "mode": "two-tier",
                "prefill_routes": 8,
                "decode_routes": 8,
                "prefill_chunk_len": 16_384,
                "prefill_state_update_len": 16_384,
                "decode_state_update_len": 256,
            }
        ],
    }
    validate_worker_attention_mode([dense], mode="full")
    validate_worker_attention_mode([lod], mode="two-tier")
    with pytest.raises(RuntimeError, match="does not match worker attention"):
        validate_worker_attention_mode([lod], mode="full")


def test_prolong_requires_loaded_kimi_binary_after_warmup() -> None:
    lod = {
        "lod_runtime": True,
        "lod_pool_count": 61,
        "attached_lod_layers": 61,
        "model_class": "KimiK3ForCausalLM",
        "loaded_kimi_lod_modules": [],
        "dense_gluon_decode_installed": False,
        "lod_engine_configurations": [
            {
                "mode": "two-tier",
                "prefill_routes": 8,
                "decode_routes": 8,
                "prefill_chunk_len": 16_384,
                "prefill_state_update_len": 16_384,
                "decode_state_update_len": 256,
            }
        ],
    }

    with pytest.raises(RuntimeError, match="loaded AITER JIT module identity"):
        validate_worker_attention_mode(
            [lod],
            mode="two-tier",
            require_loaded_kimi_lod=True,
        )


def test_prolong_runs_fixed_speed_cohort_in_execution_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.prolong as prolong

    calls: list[list[int]] = []

    def fake_timed_generate(
        llm: object,
        prompts: list[dict[str, list[int]]],
        params: object,
    ) -> tuple[
        float,
        float,
        float,
        tuple[tuple[int, ...], ...],
        dict[str, int],
        dict[str, object],
    ]:
        del llm, params
        prompt_ids = [prompt["prompt_token_ids"][0] for prompt in prompts]
        calls.append(prompt_ids)
        count = len(prompts)
        return (
            1.0,
            2.0,
            3.0,
            tuple((prompt_id,) for prompt_id in prompt_ids),
            {
                "vllm:spec_decode_num_drafts": count,
                "vllm:spec_decode_num_draft_tokens": 7 * count,
                "vllm:spec_decode_num_accepted_tokens": 2 * count,
            },
            {"wall_elapsed_seconds": 1.0},
        )

    monkeypatch.setattr(prolong, "timed_generate", fake_timed_generate)
    result = timed_generate_cohort(
        object(),
        [{"prompt_token_ids": [index]} for index in range(4)],
        object(),
        batch_size=2,
    )

    elapsed, prefill, decode, token_ids, counters, batch_counters, batch_timings = result
    assert calls == [[0, 1], [2, 3]]
    assert (elapsed, prefill, decode) == (2.0, 4.0, 6.0)
    assert token_ids == ((0,), (1,), (2,), (3,))
    assert counters == {
        "vllm:spec_decode_num_drafts": 4,
        "vllm:spec_decode_num_draft_tokens": 28,
        "vllm:spec_decode_num_accepted_tokens": 8,
    }
    assert len(batch_counters) == 2
    assert len(batch_timings) == 2


def test_prolong_trace_panel_uses_shared_nested_request_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.prolong as prolong

    complete = [
        {"prompt_token_ids": list(range(24))},
        {"prompt_token_ids": list(range(100, 124))},
    ]
    metadata = [
        {
            "request_index": index,
            "source_stream_indices": [index],
            "token_sha256": f"panel-{index}",
            "unique_16_token_block_ratio": 1.0,
        }
        for index in range(2)
    ]

    def fake_prompts(
        tokenizer: object,
        *,
        length: int,
        batch_size: int,
        validation_lengths: tuple[int, ...] | None = None,
    ) -> tuple[list[dict[str, list[int]]], list[dict[str, object]]]:
        del tokenizer
        assert (length, batch_size, validation_lengths) == (24, 2, (8, 16))
        return complete, metadata

    monkeypatch.setattr(prolong, "make_speed_prompts", fake_prompts)
    panel = make_speed_trace_panel(
        object(), lengths=[8, 16], batch_size=2, decode_tokens=8
    )

    short_prompts, short_records, short_traces = panel[8]
    long_prompts, long_records, long_traces = panel[16]
    for short, long in zip(short_prompts, long_prompts, strict=True):
        assert long["prompt_token_ids"][:8] == short["prompt_token_ids"]
    assert short_traces[0] == list(range(8, 16))
    assert long_traces[0] == list(range(16, 24))
    assert [record["panel_token_sha256"] for record in short_records] == [
        record["panel_token_sha256"] for record in long_records
    ]


def test_prolong_speed_cohort_reports_pooled_and_equal_weight_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.prolong as prolong

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.max_tokens = kwargs["max_tokens"]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    prompts = [{"prompt_token_ids": [0]}, {"prompt_token_ids": [1]}]
    metadata = [{"request_index": 0}, {"request_index": 1}]
    monkeypatch.setattr(
        prolong,
        "make_speed_prompts",
        lambda tokenizer, *, length, batch_size: (prompts, metadata),
    )

    def fake_timed_generate(
        llm: object,
        batch: list[dict[str, list[int]]],
        params: object,
    ) -> tuple[
        float,
        float,
        float,
        tuple[tuple[int, ...], ...],
        dict[str, int],
        dict[str, object],
    ]:
        del llm, params
        prompt_id = batch[0]["prompt_token_ids"][0]
        drafts, accepted = ((2, 4), (3, 3))[prompt_id]
        return (
            5.0,
            2.0,
            4.0,
            ((prompt_id,),),
            {
                "vllm:spec_decode_num_drafts": drafts,
                "vllm:spec_decode_num_draft_tokens": 7 * drafts,
                "vllm:spec_decode_num_accepted_tokens": accepted,
            },
            {"wall_elapsed_seconds": 5.0},
        )

    monkeypatch.setattr(prolong, "timed_generate", fake_timed_generate)
    fake_llm = SimpleNamespace(collective_rpc=lambda _function: None)
    progress = []
    result = prolong.evaluate_speed(
        fake_llm,
        object(),
        lengths=[100],
        batch_size=1,
        samples=2,
        decode_tokens=5,
        repeats=1,
        seed=0,
        progress_callback=progress.append,
    )["100"]

    assert result["prefill_seconds"] == 2.0
    assert result["decode_ms_per_batch_step"] == 1_000.0
    assert result["cohort_wall_timings_seconds"] == [10.0]
    assert result["speculative_target_cycle_ms"] == 1_600.0
    assert result["speculative_mean_acceptance_length"] == 2.4
    assert result["speculative_equal_weight_request_mean_acceptance_length"] == 2.5
    assert progress == [{"100": result}]


def test_prolong_speed_preserves_progress_before_later_shape_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.prolong as prolong

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: object())
    )
    monkeypatch.setattr(
        prolong,
        "make_speed_prompts",
        lambda tokenizer, *, length, batch_size: (
            [{"prompt_token_ids": [length]}], [{"tokens": length}]
        ),
    )

    def generate(llm, prompts, params, *, batch_size):
        if prompts[0]["prompt_token_ids"] == [8]:
            raise RuntimeError("later shape does not fit")
        return 3.0, 2.0, 1.0, ((7, 9),), {}, [{}], [{}]

    monkeypatch.setattr(prolong, "timed_generate_cohort", generate)
    progress = []
    with pytest.raises(RuntimeError, match="later shape does not fit"):
        prolong.evaluate_speed(
            SimpleNamespace(collective_rpc=lambda function: None),
            object(),
            lengths=[4, 8],
            batch_size=1,
            samples=1,
            decode_tokens=2,
            repeats=1,
            seed=0,
            progress_callback=progress.append,
        )
    assert len(progress) == 1
    assert set(progress[0]) == {"4"}
    assert progress[0]["4"]["prefill_seconds"] == 2.0


def test_prolong_quality_uses_frozen_raw_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    documents = [
        {"text": f"document-{index}", "length": 4}
        for index in range(max(QUALITY_DOCUMENT_INDICES) + 1)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: documents)

    class Tokenizer:
        def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
            assert kwargs["max_length"] == 4
            return {"input_ids": [len(text), 1, 2, 3]}

    prompts, metadata = select_quality_prompts(
        Tokenizer(),
        length=4,
        samples=2,
        sample_offset=8,
    )
    expected_indices = list(QUALITY_DOCUMENT_INDICES[8:10])
    assert [item["dataset_index"] for item in metadata] == expected_indices
    assert [item["document_sha256"] for item in metadata] == [
        document_digest(documents[index]["text"]) for index in expected_indices
    ]
    assert [prompt["prompt_token_ids"] for prompt in prompts] == [
        [len(documents[index]["text"]), 1, 2, 3] for index in expected_indices
    ]


def test_prolong_quality_rejects_samples_outside_frozen_cohort() -> None:
    with pytest.raises(ValueError, match="frozen shared document cohort"):
        select_quality_prompts(
            object(),
            length=65_536,
            samples=9,
            sample_offset=8,
        )


def test_synchronized_decode_configures_initial_admission_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOD_BENCHMARK_SYNCHRONIZED_DECODE", raising=False)
    monkeypatch.delenv("LOD_BENCHMARK_ADMISSION_COHORT", raising=False)

    configure_synchronized_decode_environment(enabled=True, batch_size=8)

    assert os.environ["LOD_BENCHMARK_SYNCHRONIZED_DECODE"] == "1"
    assert os.environ["LOD_BENCHMARK_ADMISSION_COHORT"] == "8"

    configure_synchronized_decode_environment(enabled=False, batch_size=8)
    assert "LOD_BENCHMARK_SYNCHRONIZED_DECODE" not in os.environ
    assert "LOD_BENCHMARK_ADMISSION_COHORT" not in os.environ


def test_release_speed_rejects_every_hidden_kimi_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOD_KIMI_CROSS_LAYER_PREFILL_GROUP", "12")

    with pytest.raises(ValueError, match="non-release Kimi benchmark environment"):
        validate_release_environment(allow_experimental=False)

    validate_release_environment(allow_experimental=True)
