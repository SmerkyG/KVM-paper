from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from benchmarks._vllm import MODES, llm_kwargs
from benchmarks.longbench_v2 import extract_answer, summarize, truncate_prompt
from benchmarks.prolong import comma_separated_ints

ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(range(len(prompt.split())))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return ",".join(map(str, token_ids))


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
    assert kwargs["max_num_batched_tokens"] == 16_384
    assert kwargs["long_prefill_token_threshold"] == 16_384
    assert kwargs["language_model_only"] is True
    assert kwargs["enable_prefix_caching"] is False


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
