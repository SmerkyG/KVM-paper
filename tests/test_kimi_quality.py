from __future__ import annotations

from types import ModuleType, SimpleNamespace
import json
import sys

import pytest

from benchmarks.kimi_k3_quality import chat_ids, engine_kwargs, evenly_spaced_panel
from benchmarks.kimi_k3_quality import evaluate, prepare_niah, public_rows


def test_kimi_chat_uses_open_response_and_native_thinking_flag():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs == dict(tokenize=True, add_generation_prompt=True, thinking=False)
            return [101, 102]

        def encode(self, text, *, add_special_tokens):
            assert text == "Answer: " and not add_special_tokens
            return [201, 202]

    assert chat_ids(Tokenizer(), "question", "Answer: ") == [101, 102, 201, 202]


def test_kimi_smoke_panel_spans_lengths_and_manifest_omits_tokens():
    assert evenly_spaced_panel(list(range(10)), 4) == [0, 3, 6, 9]
    assert evenly_spaced_panel(list(range(10)), 1) == [5]
    assert evenly_spaced_panel([1, 2], None) == [1, 2]
    with pytest.raises(ValueError):
        evenly_spaced_panel([1, 2], 0)
    assert public_rows([dict(input_tokens=2, prompt_token_ids=[5, 6])]) == [dict(input_tokens=2)]


def test_kimi_quality_matches_full_model_prolong_geometry(monkeypatch):
    import benchmarks.kimi_k3_quality as module

    calls = []
    monkeypatch.setattr(module, "llm_kwargs", lambda **kwargs: calls.append(kwargs) or {})
    args = SimpleNamespace(checkpoint="kimi-k3", mode="two-tier", batch_size=2,
                           kv_cache_memory_bytes=2**31, weight_cache_id="existing")
    kwargs = engine_kwargs(args, max_model_len=131200)
    assert calls[0]["tensor_parallel_size"] == calls[0]["decode_context_parallel_size"] == 8
    assert kwargs["enable_expert_parallel"]
    assert not kwargs["disable_custom_all_reduce"]
    assert kwargs["quantization_config"] == {"moe": {"weight": "int4_per_group_32"}}
    assert kwargs["load_format"] == "ipc_cache"
    assert kwargs["model_loader_extra_config"]["cache_id"] == "existing"
    assert kwargs["kv_cache_memory_bytes"] == 2**31


def test_kimi_quality_eager_diagnostic_is_explicit_and_keeps_geometry(monkeypatch):
    import benchmarks.kimi_k3_quality as module

    monkeypatch.setattr(module, "llm_kwargs", lambda **kwargs: {"geometry": kwargs})
    args = SimpleNamespace(checkpoint="kimi-k3", mode="two-tier", batch_size=1,
        kv_cache_memory_bytes=2**31, weight_cache_id="existing", enforce_eager=False)
    ordinary = engine_kwargs(args, max_model_len=8300)
    assert "enforce_eager" not in ordinary
    args.enforce_eager = True
    eager = engine_kwargs(args, max_model_len=8300)
    assert eager.pop("enforce_eager") is True
    assert eager == ordinary


def test_dense_shadow_reference_includes_archive_tail_and_current_token():
    import torch
    from benchmarks._kimi_decode_reference import dense_pool_partial

    # Rank zero of a two-way shard has already appended position eight.
    keys = torch.randn(5, 6)
    query = torch.randn(1, 3, 6)
    # Poison unused capacity: the diagnostic may read only authoritative rows.
    archive = torch.full((2, 1, 10, 6), float("nan"))
    recent = torch.full_like(archive, float("nan"))
    archive[1, 0, :2] = keys[1:3]
    recent[1, 0, :2] = keys[3:]
    sink = torch.zeros(2, 1, 1, 6)
    sink[1, 0, 0] = keys[0]
    pool = SimpleNamespace(active_decode_rows=(1,), value_dim=4,
        engine=SimpleNamespace(scaling=.2), metadata=[{}, dict(coverage=3)],
        local_lens=torch.tensor([0, 2]), dcp_global_lens=torch.tensor([0, 9]),
        _dcp_local_length=lambda n: (n + 1) // 2,
        state=dict(sink_k=sink, recent_k=recent, page_cache=dict(leaf_k=archive)))
    out, lse = dense_pool_partial(pool, query)
    scores = query @ keys.T * .2
    torch.testing.assert_close(out, torch.softmax(scores, -1) @ keys[:, :4])
    torch.testing.assert_close(lse, torch.logsumexp(scores, -1))


def _fake_vllm(monkeypatch):
    vllm = ModuleType("vllm")
    vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    params = ModuleType("vllm.sampling_params")
    params.StructuredOutputsParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", params)


def test_kimi_quality_scores_and_reports_each_batch(monkeypatch):
    _fake_vllm(monkeypatch)

    class LLM:
        def generate(self, prompts, params, **kwargs):
            assert params.max_tokens == 32 and params.temperature == 0 and params.seed == 0
            assert params.structured_outputs.choice == [f"The correct answer is ({x})" for x in "ABCD"]
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text="The correct answer is (B)", token_ids=[1, 2], finish_reason="stop")])
                for _ in prompts]

    rows = [dict(_id=str(i), prompt_token_ids=[5], answer=answer,
                 domain="qa", difficulty="easy", length="short")
            for i, answer in enumerate(("A", "B", "B"))]
    progress = []
    result = evaluate(LLM(), rows, task="longbench-v2", batch_size=2, progress=progress.append)
    assert result["correct"] == 2 and result["total"] == 3
    assert len(progress[0]["samples"]) == 2 and len(progress[1]["samples"]) == 3
    assert all("prompt_token_ids" not in x for x in result["samples"])


def test_kimi_niah_generation_preserves_canonical_prefix_and_seeds(monkeypatch):
    import benchmarks.kimi_k3_quality as module
    import random
    import numpy as np

    utils = ModuleType("lm_eval.tasks.ruler.niah_utils")
    utils.TEMPLATE = "canonical"
    monkeypatch.setitem(sys.modules, "lm_eval.tasks.ruler.niah_utils", utils)

    def generate(haystack, **kwargs):
        assert haystack == "essay" and kwargs["max_seq_length"] == 32768
        assert kwargs["type_needle_k"] == "words" and kwargs["type_needle_v"] == "uuids"
        assert kwargs["num_samples"] == 1 and kwargs["template"] == "canonical"
        assert random.random() == 0.8444218515250481
        assert np.random.random() == 0.1915194503788923
        return [dict(index=0, input="question", gen_prefix="Answer: ", outputs=["uuid"])]

    monkeypatch.setattr(module, "_load_ruler_generator", lambda: (
        generate, lambda **kwargs: "essay", "0.4.13"))
    monkeypatch.setattr(module, "chat_ids", lambda t, user, prefix: (
        [10, 20] if (user, prefix) == ("question", "Answer: ") else []))
    result = prepare_niah(object(), length=32768, samples=1)
    assert result[0]["target"] == "uuid" and result[0]["input_tokens"] == 2
    assert len(result[0]["token_sha256"]) == 64


def test_kimi_quality_main_preflights_nested_full_model_without_loading(monkeypatch, tmp_path):
    import benchmarks.kimi_k3_quality as module

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 93}}))
    args = SimpleNamespace(checkpoint=str(checkpoint), mode="full", batch_size=2,
        kv_cache_memory_bytes=2**31, weight_cache_id="existing", tasks="niah-s3",
        niah_lengths=[32768], niah_samples=8, max_input_tokens=131072,
        longbench_limit=None, preflight_only=True, output=tmp_path / "preflight.json")
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(module, "_load_ruler_generator", lambda: (None, None, "0.4.13"))
    monkeypatch.setattr(module, "prepare_niah", lambda *args, **kwargs: [
        dict(index=0, prompt_token_ids=[10, 11], input_tokens=2, token_sha256="hash")])
    monkeypatch.setattr(module, "benchmark_identity", lambda: {})
    monkeypatch.setattr(module, "engine_kwargs", lambda *args, **kwargs: {})
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    # No vLLM import/weight loading may occur on the preflight-only branch.
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    module.main()
    manifest = json.loads(args.output.with_suffix(".prompts.json").read_text())
    assert manifest["thinking"] is False and manifest["lm_eval_version"] == "0.4.13"
    assert manifest["prompt_manifest"]["niah-s3:32768"][0]["input_tokens"] == 2
    assert "prompt_token_ids" not in manifest["prompt_manifest"]["niah-s3:32768"][0]
    assert not args.output.exists()
