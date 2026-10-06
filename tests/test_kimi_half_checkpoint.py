from copy import deepcopy

import pytest

from benchmarks.kimi_k3_prepare_half_checkpoint import retained_weight, selected_weight_map, stage_config


def test_stage_config_retains_real_geometry_and_first_48_layers():
    source = {"text_config":{"num_hidden_layers":93, "num_nextn_predict_layers":0,
        "attn_res_block_size":12, "hidden_size":7168,
        "linear_attn_config":{"full_attn_layers":[4, 48, 52, 93], "kda_layers":[1, 47, 49, 91]}},
        "vision_config":{"num_hidden_layers":27}}
    previous = deepcopy(source)
    actual = stage_config(source, 48)
    assert source == previous
    assert actual["text_config"]["num_hidden_layers"] == 48
    assert actual["text_config"]["hidden_size"] == 7168
    assert actual["text_config"]["linear_attn_config"] == {
        "full_attn_layers":[4, 48], "kda_layers":[1, 47]}
    assert actual["vision_config"] == source["vision_config"]


@pytest.mark.parametrize("layers", [0, 47, 93, 96])
def test_invalid_stage_boundary_rejected(layers):
    with pytest.raises(ValueError):
        stage_config({"num_hidden_layers":93, "attn_res_block_size":12}, layers)


def test_weight_filter_does_not_drop_vision_layers_or_globals():
    assert retained_weight("language_model.model.layers.47.self_attn.weight", 48)
    assert not retained_weight("language_model.model.layers.48.self_attn.weight", 48)
    assert not retained_weight("model.layers.92.block_sparse_moe.weight", 48)
    assert retained_weight("vision_tower.layers.50.weight", 48)
    assert retained_weight("language_model.model.output_attn_res_proj.weight", 48)
    assert retained_weight("language_model.lm_head.weight", 48)


def test_index_must_have_every_retained_layer():
    index = {"weight_map":{f"language_model.model.layers.{i}.weight":f"layer{i}.safetensors"
                          for i in range(93)}}
    selected = selected_weight_map(index, 48)
    assert len(selected) == 48
    assert "language_model.model.layers.48.weight" not in selected
    del index["weight_map"]["language_model.model.layers.12.weight"]
    with pytest.raises(ValueError):
        selected_weight_map(index, 48)


def test_half_preload_uses_native_kimi_backend_and_separate_daemon(monkeypatch):
    from benchmarks.kimi_k3_half_preload import preload_kwargs

    for name in ("LOD_KIMI_REQUEST_OWNER_PREFILL", "LOD_KIMI_GRAPH_PREFILL"):
        monkeypatch.delenv(name, raising=False)
    actual = preload_kwargs("/tmp/kimi-k3-first48", "half-stage-test")
    assert actual["load_format"] == "ipc_cache"
    assert actual["model_loader_extra_config"]["cache_id"] == "half-stage-test"
    assert actual["model_loader_extra_config"]["auto_start"] is False
    assert actual["tensor_parallel_size"] == actual["decode_context_parallel_size"] == 8
    assert actual["enable_expert_parallel"] is True
    assert actual["language_model_only"] is True
    assert actual["attention_config"] == {
        "backend": None, "backend_per_kind": {"mla_attention": "TRITON_MLA"}}
    assert actual["quantization_config"] == {"moe":{"weight":"int4_per_group_32"}}


def test_extended_prefill_artifacts_do_not_overwrite_original_64k_points():
    from benchmarks.kimi_k3_half_prefill import length_suffix

    assert length_suffix([65536]) == ""
    assert length_suffix([131072]) == "-128k"
    assert length_suffix([131072, 262144]) == "-128k-256k"


@pytest.mark.parametrize("cohort", [4, 8])
@pytest.mark.parametrize("mode", ["full", "two-tier"])
def test_higher_prefill_cohorts_keep_global_16k_row_chunks(monkeypatch, cohort, mode):
    from benchmarks._vllm import llm_kwargs

    monkeypatch.setenv("LOD_BENCHMARK_PREFILL_COHORT", str(cohort))
    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_PREFILL", "0")
    actual = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-mla-stack", mode=mode,
        max_model_len=131081, batch_size=8, tensor_parallel_size=8,
        decode_context_parallel_size=8, gpu_memory_utilization=0.8,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    assert actual["max_num_batched_tokens"] == cohort * 16384 + 8
    assert actual["long_prefill_token_threshold"] == 16384
    assert actual["max_num_seqs"] == 8
    assert actual["kv_cache_dtype"] == "bfloat16"
