from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks._vllm import (
    LOD_SCHEDULER,
    MODES,
    default_gpu_memory_utilization,
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
    document_digest,
    select_quality_prompts,
    speculative_counters,
    timed_generate_cohort,
)

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
    assert "NumPy's generator seed to `1234`" in (
        ROOT / "benchmarks" / "NIAH_S3.md"
    ).read_text()
    assert "no randomized sampling" in (
        ROOT / "benchmarks" / "LONGBENCH_V2.md"
    ).read_text()


def test_prolong_collects_speculative_counter_totals() -> None:
    assert speculative_counters(_LLMWithMetrics()) == {
        "vllm:spec_decode_num_drafts": 15,
        "vllm:spec_decode_num_accepted_tokens": 44,
    }


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
        )

    monkeypatch.setattr(prolong, "timed_generate", fake_timed_generate)
    result = timed_generate_cohort(
        object(),
        [{"prompt_token_ids": [index]} for index in range(4)],
        object(),
        batch_size=2,
    )

    elapsed, prefill, decode, token_ids, counters, batch_counters = result
    assert calls == [[0, 1], [2, 3]]
    assert (elapsed, prefill, decode) == (2.0, 4.0, 6.0)
    assert token_ids == ((0,), (1,), (2,), (3,))
    assert counters == {
        "vllm:spec_decode_num_drafts": 4,
        "vllm:spec_decode_num_draft_tokens": 28,
        "vllm:spec_decode_num_accepted_tokens": 8,
    }
    assert len(batch_counters) == 2


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
        )

    monkeypatch.setattr(prolong, "timed_generate", fake_timed_generate)
    result = prolong.evaluate_speed(
        object(),
        object(),
        lengths=[100],
        batch_size=1,
        samples=2,
        decode_tokens=5,
        repeats=1,
        seed=0,
    )["100"]

    assert result["prefill_seconds"] == 2.0
    assert result["decode_ms_per_batch_step"] == 1_000.0
    assert result["speculative_target_cycle_ms"] == 1_600.0
    assert result["speculative_mean_acceptance_length"] == 2.4
    assert (
        result["speculative_equal_weight_request_mean_acceptance_length"] == 2.5
    )


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
