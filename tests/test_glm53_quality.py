"""CPU checks for the matched GLM ProLong/LongBench smoke protocol."""
from types import SimpleNamespace
import sys

import pytest

from benchmarks._glm53_quality import (
    common_quality_prefix, evaluate_longbench, prepare_longbench, select_longbench_items)


def items():
    return [dict(_id=f"{band}-{i}", length=band, domain=f"d{i % 4}",
        sub_domain="s", difficulty="easy", answer="A", context="text",
        question="question", choice_A="a", choice_B="b", choice_C="c", choice_D="d")
        for band in ("short", "medium", "long") for i in range(12)]


def test_stratified_panel_is_deterministic_unique_and_mode_independent():
    selected = select_longbench_items(items(), 16)
    assert selected == select_longbench_items(list(reversed(items())), 16)
    assert len({row["_id"] for row in selected}) == 16
    assert {row["length"] for row in selected} == {"short", "medium", "long"}
    assert len({row["domain"] for row in selected}) == 4
    with pytest.raises(ValueError):
        select_longbench_items(items(), 2)


def test_prolong_keeps_documents_without_padding_or_substitution():
    prompts = [{"prompt_token_ids": [1, 2, 3, 4]}, {"prompt_token_ids": [5, 6, 7]}]
    metadata = [dict(dataset_index=14, tokens=4), dict(dataset_index=19, tokens=3)]
    (clipped, records), length = common_quality_prefix((prompts, metadata))
    assert length == 3 and clipped == [{"prompt_token_ids": [1, 2, 3]},
                                      {"prompt_token_ids": [5, 6, 7]}]
    assert [r["dataset_index"] for r in records] == [14, 19]
    assert [r["available_prefix_tokens"] for r in records] == [4, 3]
    assert all(r["tokens"] == 3 for r in records) and prompts[0]["prompt_token_ids"][-1] == 4


def test_generic_prolong_selector_stays_strict_unless_explicitly_allowed(monkeypatch):
    from benchmarks.prolong import select_quality_prompts
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        load_dataset=lambda *a, **k:[{"text": "short raw document"}]))
    tokenizer = lambda *a, **k: {"input_ids": [1, 2]}
    with pytest.raises(RuntimeError, match="only 2 tokens"):
        select_quality_prompts(tokenizer, length=3, samples=1, sample_offset=0)
    prompts, metadata = select_quality_prompts(tokenizer, length=3, samples=1,
        sample_offset=0, allow_short_documents=True)
    assert prompts == [{"prompt_token_ids": [1, 2]}] and metadata[0]["tokens"] == 2


def test_prepare_uses_native_closed_thinking_template_and_token_digests(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k:items()))
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text)))
        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1] == dict(role="assistant", content="", think="")
            assert kwargs == dict(tokenize=True, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False)
            return {"input_ids": [1, 2, 3]}
        def decode(self, ids, **kwargs):
            return "<think></think>"
    rows = prepare_longbench(Tokenizer(), max_input_tokens=1000, samples=8)
    assert len(rows) == 8 and all(row["input_tokens"] == 3 for row in rows)
    assert all(len(row["token_sha256"]) == 64 for row in rows)


def test_quality_scores_the_existing_answer_grammar_in_complete_cohorts(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **k:k))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        StructuredOutputsParams=lambda **k:k))
    rows = [dict(row, prompt_token_ids=[1, 2, 3]) for row in items()[:8]]
    def generate(prompts, params, **kwargs):
        assert len(prompts) == 8 and params["seed"] == 1234 and params["max_tokens"] == 32
        assert len(params["structured_outputs"]["choice"]) == 4
        return [SimpleNamespace(outputs=[SimpleNamespace(text="The correct answer is (A)",
            token_ids=[1], finish_reason="stop")]) for row in prompts]
    progress = []
    result = evaluate_longbench(SimpleNamespace(generate=generate), rows,
        batch_size=8, progress=progress.append)
    assert result["correct"] == result["total"] == 8
    assert result["accuracy"] == 1 and len(progress) == 1
    assert all("prompt_token_ids" not in row for row in result["samples"])
    with pytest.raises(ValueError, match="complete batches"):
        evaluate_longbench(None, rows[:7], batch_size=8, progress=progress.append)


def test_runner_preflights_both_manifests_without_loading_weights(monkeypatch, tmp_path):
    import json
    from benchmarks import glm53_flash_full, prolong
    import benchmarks._glm53_quality as quality
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(dict(
        model_type="glm5_next_text", num_hidden_layers=45, n_routed_experts=288)))
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["glm", "--checkpoint", str(checkpoint),
        "--mode", "full", "--batch-size", "8", "--measure", "quality",
        "--preflight-only", "--output", str(output)])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **k:object())))
    def select(*args, **kwargs):
        assert kwargs == dict(length=65536, samples=8, sample_offset=8, allow_short_documents=True)
        return [{"prompt_token_ids": [1, 2]}] * 8, [{"dataset_index": 14, "tokens": 2}] * 8
    monkeypatch.setattr(prolong, "select_quality_prompts", select)
    monkeypatch.setattr(quality, "prepare_longbench", lambda *a, **k:[dict(
        prompt_token_ids=[1, 2], input_tokens=2, _id=str(i)) for i in range(16)])
    glm53_flash_full.main()
    result = json.loads(output.read_text())
    assert result["status"] == "quality-preflight-complete"
    assert len(result["prompt_manifest"]["longbench-v2"]) == 16
    assert not result["prolong_chat_template"] and result["longbench_chat_template"]
    assert result["measurements"] == {}
