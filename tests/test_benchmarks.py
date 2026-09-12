from __future__ import annotations

import argparse
from pathlib import Path

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
from benchmarks.prolong import comma_separated_ints, speculative_counters

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
