#!/usr/bin/env python3
"""CPU checks for the registered, model-independent HF LOD backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import torch
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    SmolLM3Config,
    SmolLM3ForCausalLM,
)

from model.hf_lod_hybrid_cache import HybridHFLODCache
from model.hf_pytorch_lod_attention import (
    HFLODCache,
    _resolved_rope_route_geometry,
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODCache, PagedLODConfig
from model.triton_lod_engines import KernelRecursivePagedLODAttention


def _lod_config() -> LODConfig:
    return LODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=2.0,
        state_min_size=4,
        protected_prefix=1,
        max_routes=2,
    )


def _common_config() -> dict[str, int]:
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
    }


def _check_qk_norm_aware_routing() -> None:
    common = _common_config()
    bounded_tokens = {
        **common,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    requested = LODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=2.0,
        state_min_size=4,
        max_routes=2,
        routing_normalization="qk_norm_aware",
    )
    unnormalized = (
        LlamaForCausalLM(LlamaConfig(**common)).eval(),
        Qwen2ForCausalLM(Qwen2Config(**common)).eval(),
        SmolLM3ForCausalLM(SmolLM3Config(**bounded_tokens)).eval(),
        Phi3ForCausalLM(Phi3Config(**bounded_tokens)).eval(),
    )
    qwen = Qwen3ForCausalLM(Qwen3Config(**common)).eval()
    for model in unnormalized:
        install_hf_lod_attention(model, config=requested, open_count=2)
    install_hf_lod_attention(qwen, config=requested, open_count=2)
    unnormalized_modes = [
        model.model.layers[0].self_attn._hf_lod_settings.config.routing_normalization
        for model in unnormalized
    ]
    qwen_mode = (
        qwen.model.layers[0]
        .self_attn._hf_lod_settings.config.routing_normalization
    )
    if unnormalized_modes != ["query"] * len(unnormalized) or qwen_mode != "none":
        raise AssertionError(
            "Q/K-norm-aware routing did not resolve from module architecture"
        )
    print("Q/K-norm-aware routing architecture dispatch passed")


def _check_state_clustering_radial_metric() -> None:
    config = PagedLODConfig(
        chunk_size=16,
        local_window=32,
        state_min_size=16,
        state_clustering_normalization="cosine",
        state_clustering_radial_bias=1.0,
    )
    engine = KernelRecursivePagedLODAttention(
        config,
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    leaf = torch.tensor([[[[2.0, 0.0], [1.0, 1.0]]]])
    centroid = torch.tensor([[[[1.0, 0.0], [2.0, 2.0]]]])
    routed_leaf = engine._state_clustering_key(leaf)
    routed_centroid = engine._state_clustering_key(centroid, role="centroid")
    actual = engine._state_clustering_similarity(
        routed_leaf, routed_centroid
    ) / leaf.size(-1)
    cosine = torch.tensor([[[[1.0, 2**-0.5], [2**-0.5, 1.0]]]])
    leaf_rms = leaf.square().mean(-1).sqrt()
    centroid_rms = centroid.square().mean(-1).sqrt()
    expected = cosine - (
        leaf_rms.unsqueeze(-1).log() - centroid_rms.unsqueeze(-2).log()
    ).abs()
    torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)
    if routed_leaf.size(-1) != leaf.size(-1) + 1:
        raise AssertionError("radial coordinate was not transiently appended")
    engine.state_clustering_radial_scope = "append"
    torch.testing.assert_close(
        engine._state_clustering_similarity(
            routed_leaf, routed_centroid, purpose="assignment"
        )
        / leaf.size(-1),
        cosine,
        atol=8e-3,
        rtol=8e-3,
    )
    torch.testing.assert_close(
        engine._state_clustering_similarity(
            routed_leaf, routed_centroid, purpose="append"
        )
        / leaf.size(-1),
        expected,
        atol=8e-3,
        rtol=8e-3,
    )
    try:
        LODConfig(state_clustering_radial_bias=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("radial bias without spherical routing was accepted")
    mean_leaf_config = PagedLODConfig(
        chunk_size=16,
        local_window=32,
        state_growth_factor=0,
        state_min_size=2,
        protected_prefix=0,
        state_clustering_centroid_rescale="mean_leaf_norm",
    )
    mean_leaf_engine = KernelRecursivePagedLODAttention(
        mean_leaf_config,
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    state_key = torch.zeros(1, 1, 6, 2)
    state_value = torch.zeros_like(state_key)
    state_key[..., :2, :].copy_(torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]]))
    state_value[..., :2, :].copy_(state_key[..., :2, :])
    counts = torch.zeros(1, 1, 6, 1)
    counts[..., :2, :].fill_(1)
    key_norm_sums = torch.zeros_like(counts)
    key_norm_sums[..., :2, :].copy_(
        state_key[..., :2, :].square().mean(-1, keepdim=True).sqrt()
    )
    mean_key = state_key[..., :2, :] / counts[..., :2, :]
    target_rms = key_norm_sums[..., :2, :] / counts[..., :2, :]
    rescaled_centroid = mean_leaf_engine._state_clustering_key(
        mean_key, role="centroid", radial_rms=target_rms
    )
    torch.testing.assert_close(
        rescaled_centroid.square().mean(-1, keepdim=True).sqrt(), target_rms
    )
    torch.testing.assert_close(
        torch.nn.functional.normalize(rescaled_centroid.float(), dim=-1),
        torch.nn.functional.normalize(mean_key.float(), dim=-1),
    )
    coherence_config = replace(
        mean_leaf_config, state_clustering_centroid_rescale="coherence"
    )
    coherence_engine = KernelRecursivePagedLODAttention(
        coherence_config,
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    coherence_centroid = coherence_engine._state_clustering_key(
        mean_key, role="centroid", radial_rms=target_rms
    )
    expected_coherence = (
        mean_key.square().mean(-1, keepdim=True).sqrt() / target_rms
    )
    torch.testing.assert_close(
        coherence_centroid.square().mean(-1, keepdim=True).sqrt(),
        expected_coherence,
    )
    torch.testing.assert_close(
        torch.nn.functional.normalize(coherence_centroid.float(), dim=-1),
        torch.nn.functional.normalize(mean_key.float(), dim=-1),
    )
    scoped_coherence_engine = KernelRecursivePagedLODAttention(
        replace(
            coherence_config,
            state_clustering_centroid_rescale_scope="assignment",
        ),
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    assignment_centroid = scoped_coherence_engine._state_clustering_key(
        mean_key,
        role="centroid",
        radial_rms=target_rms,
        purpose="assignment",
    )
    append_centroid = scoped_coherence_engine._state_clustering_key(
        mean_key,
        role="centroid",
        radial_rms=target_rms,
        purpose="append",
    )
    torch.testing.assert_close(assignment_centroid, coherence_centroid)
    torch.testing.assert_close(
        append_centroid.square().mean(-1, keepdim=True).sqrt(),
        torch.ones_like(target_rms),
    )
    spherical_coherence_engine = KernelRecursivePagedLODAttention(
        replace(
            coherence_config,
            state_clustering_centroid_rescale="spherical_coherence",
            state_clustering_centroid_rescale_scope="assignment",
        ),
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    unequal_leaf = torch.tensor([[[[3.0, 4.0], [0.0, 2.0]]]])
    spherical_leaf = spherical_coherence_engine._state_clustering_key(
        unequal_leaf, role="leaf"
    )
    torch.testing.assert_close(
        spherical_leaf.square().mean(-1, keepdim=True).sqrt(),
        torch.ones_like(spherical_leaf[..., :1]),
    )
    spherical_assignment_centroid = (
        spherical_coherence_engine._state_clustering_key(
            mean_key,
            role="centroid",
            radial_rms=target_rms,
            purpose="assignment",
        )
    )
    spherical_append_centroid = spherical_coherence_engine._state_clustering_key(
        mean_key,
        role="centroid",
        radial_rms=target_rms,
        purpose="append",
    )
    torch.testing.assert_close(
        spherical_assignment_centroid, coherence_centroid
    )
    torch.testing.assert_close(
        spherical_append_centroid.square().mean(-1, keepdim=True).sqrt(),
        torch.ones_like(target_rms),
    )
    partial_mean = torch.tensor([[[[2.0, 0.0, 3.0, 4.0]]]])
    partial_constituents = torch.tensor(
        [[[[2.0, 0.0, 3.0, 4.0], [0.0, 2.0, 3.0, 4.0]]]]
    )
    rope_coherence_engine = KernelRecursivePagedLODAttention(
        replace(
            mean_leaf_config,
            state_clustering_centroid_rescale="rope_coherence",
            state_clustering_rope_dim=2,
        ),
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    constituent_rope_rms = (
        rope_coherence_engine._state_clustering_constituent_rms(
            partial_constituents
        ).mean(dim=-2, keepdim=True)
    )
    partial_centroid = rope_coherence_engine._state_clustering_key(
        partial_mean,
        role="centroid",
        radial_rms=constituent_rope_rms,
    )
    expected_rope_coherence = (
        partial_mean[..., :2].square().mean(-1, keepdim=True).sqrt()
        / constituent_rope_rms
    )
    torch.testing.assert_close(
        partial_centroid.square().mean(-1, keepdim=True).sqrt(),
        expected_rope_coherence,
    )
    torch.testing.assert_close(
        torch.nn.functional.normalize(partial_centroid.float(), dim=-1),
        torch.nn.functional.normalize(partial_mean.float(), dim=-1),
    )
    overflow_key = torch.tensor(
        [[[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]]]]
    )
    direction_l2_engine = KernelRecursivePagedLODAttention(
        replace(
            mean_leaf_config,
            state_clustering_centroid_rescale="direction_l2",
        ),
        query_heads=1,
        key_value_heads=1,
        scale=1.0,
        default_open_count=1,
    )
    routed_leaf = direction_l2_engine._state_clustering_key(overflow_key)
    routed_mean = direction_l2_engine._state_clustering_key(
        mean_key, role="centroid", radial_rms=target_rms
    )
    actual_l2_score = direction_l2_engine._state_clustering_similarity(
        routed_leaf, routed_mean
    )
    direction_dim = int(routed_leaf.size(-1))
    expected_l2_score = torch.matmul(
        routed_leaf.float(), routed_mean.float().transpose(-1, -2)
    ) - (
        0.5
        * direction_dim
        * routed_mean.float().square().mean(-1).unsqueeze(-2)
    )
    torch.testing.assert_close(actual_l2_score, expected_l2_score)
    torch.testing.assert_close(
        routed_leaf.float().square().mean(-1),
        torch.ones_like(routed_leaf[..., 0]),
    )
    original_norm_sums = key_norm_sums.clone()
    _, _, _, _, owners, _ = mean_leaf_engine._update_state(
        state_key,
        state_value,
        counts,
        key_norm_sums,
        overflow_key,
        overflow_key,
        state_len=2,
        ctx_len=6,
        available_context=6,
        state_capacity=6,
    )
    overflow_norms = overflow_key.square().mean(-1, keepdim=True).sqrt()
    expected_norm_sums = original_norm_sums[..., :2, :].clone()
    expected_norm_sums.scatter_add_(
        2, owners.unsqueeze(-1), overflow_norms
    )
    torch.testing.assert_close(
        key_norm_sums[..., :2, :], expected_norm_sums
    )
    print("angular-plus-log-radial state clustering metric passed")


def _check_rope_frequency_routing() -> None:
    partial_config = SimpleNamespace(
        model_type="qwen3_5_text",
        rope_parameters={
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    partial_module = SimpleNamespace(layer_idx=3, head_dim=256, layer_type=None)
    rope_dim, fast_pairs = _resolved_rope_route_geometry(
        partial_config, partial_module, 512
    )
    if (rope_dim, fast_pairs) != (64, 9):
        raise AssertionError("partial-RoPE wavelength cutoff is incorrect")

    smol_config = SimpleNamespace(
        model_type="smollm3",
        no_rope_layers=[1, 1, 1, 0],
        rope_theta=5_000_000.0,
        rope_parameters=None,
    )
    no_rope_module = SimpleNamespace(layer_idx=3, head_dim=128, layer_type=None)
    if _resolved_rope_route_geometry(smol_config, no_rope_module, 512) != (0, 0):
        raise AssertionError("NoPE layer received a rotary route filter")

    llama = LlamaForCausalLM(LlamaConfig(**_common_config())).eval()
    install_hf_lod_attention(
        llama,
        config=LODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=2.0,
            state_min_size=4,
            max_routes=2,
            routing_leaf_mass_candidates=16,
            routing_leaf_mass_objective="fast_rope_jensen",
        ),
        open_count=2,
    )
    routed = llama.model.layers[0].self_attn._hf_lod_settings.config
    if (
        routed.routing_rope_dim != 8
        or routed.routing_rope_fast_pairs != 0
        or routed.routing_rope_jensen_pairs != 1
    ):
        raise AssertionError(
            "fast-band Jensen geometry changed the ordinary routing mask"
        )
    print("RoPE wavelength routing geometry passed")


def _check_rope_aware_state_clustering() -> None:
    common = {
        **_common_config(),
        "num_hidden_layers": 4,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    smol = SmolLM3ForCausalLM(SmolLM3Config(**common)).eval()
    install_hf_lod_attention(
        smol,
        config=replace(_lod_config(), state_clustering_policy="rope_aware"),
        open_count=2,
        engine_backend="kernel",
    )
    resolved = [
        layer.self_attn._hf_lod_settings.config
        for layer in smol.model.layers
    ]
    for layer_config in resolved:
        if (
            layer_config.state_clustering_policy != "manual"
            or layer_config.state_clustering_normalization != "cosine"
            or layer_config.state_clustering_centroid_rescale != "none"
        ):
            raise AssertionError(
                "RNoPE model did not resolve to consistent spherical clustering"
            )

    inverse = SmolLM3ForCausalLM(SmolLM3Config(**common)).eval()
    install_hf_lod_attention(
        inverse,
        config=replace(
            _lod_config(), state_clustering_policy="rnope_rope_spherical"
        ),
        open_count=2,
        engine_backend="kernel",
    )
    inverse_resolved = [
        layer.self_attn._hf_lod_settings.config
        for layer in inverse.model.layers
    ]
    for layer_config in inverse_resolved[:3]:
        if (
            layer_config.state_clustering_normalization != "cosine"
            or layer_config.state_clustering_centroid_rescale != "none"
        ):
            raise AssertionError("inverse RNoPE RoPE layer was not spherical")
    inverse_nope = inverse_resolved[3]
    if (
        inverse_nope.state_clustering_normalization != "none"
        or inverse_nope.state_clustering_centroid_rescale != "coherence"
        or inverse_nope.state_clustering_centroid_rescale_scope != "assignment"
    ):
        raise AssertionError("inverse RNoPE NoPE layer lost coherence routing")

    llama = LlamaForCausalLM(LlamaConfig(**_common_config())).eval()
    install_hf_lod_attention(
        llama,
        config=replace(_lod_config(), state_clustering_policy="rope_aware"),
        open_count=2,
        engine_backend="kernel",
    )
    rope_config = llama.model.layers[0].self_attn._hf_lod_settings.config
    if (
        rope_config.state_clustering_policy != "manual"
        or rope_config.state_clustering_normalization != "none"
        or rope_config.state_clustering_centroid_rescale != "coherence"
        or rope_config.state_clustering_centroid_rescale_scope != "assignment"
    ):
        raise AssertionError("RoPE model did not retain centroid coherence")

    qwen_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    qwen = Qwen3_5ForCausalLM(qwen_config).eval()
    install_hf_lod_attention(
        qwen,
        config=replace(_lod_config(), state_clustering_policy="rope_aware"),
        open_count=2,
        engine_backend="kernel",
    )
    partial_config = qwen.model.layers[1].self_attn._hf_lod_settings.config
    if (
        partial_config.state_clustering_centroid_rescale != "coherence"
        or partial_config.state_clustering_rope_dim != 2
    ):
        raise AssertionError("partial-RoPE model lost whole-key coherence")
    print("RoPE-aware state-clustering policy dispatch passed")


def _check_qk_norm_aware_state_clustering() -> None:
    common = {
        **_common_config(),
        "num_hidden_layers": 4,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    smol = SmolLM3ForCausalLM(SmolLM3Config(**common)).eval()
    qwen = Qwen3ForCausalLM(Qwen3Config(**common)).eval()
    requested = replace(
        _lod_config(), state_clustering_policy="qk_norm_aware"
    )
    install_hf_lod_attention(
        smol, config=requested, open_count=2, engine_backend="kernel"
    )
    install_hf_lod_attention(
        qwen, config=requested, open_count=2, engine_backend="kernel"
    )
    smol_configs = [
        layer.self_attn._hf_lod_settings.config for layer in smol.model.layers
    ]
    qwen_configs = [
        layer.self_attn._hf_lod_settings.config for layer in qwen.model.layers
    ]
    if any(
        config.state_clustering_normalization != "cosine"
        or config.state_clustering_centroid_rescale != "none"
        for config in smol_configs
    ):
        raise AssertionError("non-K-normalized layers were not spherical")
    if any(
        config.state_clustering_normalization != "none"
        or config.state_clustering_centroid_rescale != "coherence"
        or config.state_clustering_centroid_rescale_scope != "assignment"
        for config in qwen_configs
    ):
        raise AssertionError("K-normalized layers lost coherence routing")
    print("Q/K-norm-aware state-clustering policy dispatch passed")


@torch.no_grad()
def _check_model_family(model) -> None:
    torch.manual_seed(12)
    token = torch.randint(0, model.config.vocab_size, (1, 14))
    baseline = model(token[:, :4], use_cache=False).logits
    original_state_keys = tuple(model.state_dict())
    installed = install_hf_lod_attention(
        model, config=_lod_config(), open_count=2
    )
    if len(installed) != model.config.num_hidden_layers:
        raise AssertionError("the generic installer missed an attention layer")
    local_lod = model(token[:, :4], use_cache=False).logits
    torch.testing.assert_close(local_lod, baseline, atol=1e-5, rtol=1e-5)
    if tuple(model.state_dict()) != original_state_keys:
        raise AssertionError("LOD installation changed model state-dict keys")

    monolithic = model(token, use_cache=False).logits
    cache = new_hf_lod_cache(model)
    attention_mask = torch.ones(1, 10, dtype=torch.long)
    prefill = model(
        token[:, :10],
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    cached_outputs = [prefill.logits]
    for position in range(10, token.size(1)):
        attention_mask = torch.ones(1, position + 1, dtype=torch.long)
        result = model(
            token[:, position : position + 1],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cached_outputs.append(result.logits)
    cached = torch.cat(cached_outputs, dim=1)
    torch.testing.assert_close(cached, monolithic, atol=1e-5, rtol=1e-5)
    if cache.get_seq_length() != token.size(1):
        raise AssertionError("HF LOD cache length did not advance")
    if any(layer.keys.numel() or layer.values.numel() for layer in cache.layers):
        raise AssertionError("HF compatibility sentinels retained duplicate K/V")
    if any(layer.lod_cache is None for layer in cache.layers):
        raise AssertionError("LOD-owned layer storage was not populated")
    print(f"{type(model).__name__}: registered prefill/decode parity passed")


@torch.inference_mode()
def _check_varied_left_padding(model) -> None:
    torch.manual_seed(18)
    lengths = (5, 8, 8, 12)
    padded_length = max(lengths)
    token = torch.zeros(len(lengths), padded_length, dtype=torch.long)
    attention_mask = torch.zeros_like(token)
    for row, length in enumerate(lengths):
        token[row, -length:] = torch.randint(
            3, model.config.vocab_size, (length,)
        )
        attention_mask[row, -length:] = 1

    transient = model(
        token, attention_mask=attention_mask, use_cache=False
    ).logits
    transient_references = []
    for row, length in enumerate(lengths):
        valid_token = token[row : row + 1, -length:]
        transient_references.append(
            model(
                valid_token,
                attention_mask=torch.ones_like(valid_token),
                use_cache=False,
            ).logits
        )
        torch.testing.assert_close(
            transient[row : row + 1, -length:],
            transient_references[-1],
            atol=5e-4,
            rtol=5e-4,
        )

    cache = new_hf_lod_cache(model)
    prefill = model(
        token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    ).logits
    reference_caches = []
    for row, length in enumerate(lengths):
        valid_token = token[row : row + 1, -length:]
        reference_cache = new_hf_lod_cache(model)
        reference_caches.append(reference_cache)
        reference = model(
            valid_token,
            attention_mask=torch.ones_like(valid_token),
            past_key_values=reference_cache,
            use_cache=True,
        ).logits
        torch.testing.assert_close(
            prefill[row : row + 1, -length:],
            reference,
            atol=5e-4,
            rtol=5e-4,
        )

    next_token = torch.randint(
        3, model.config.vocab_size, (len(lengths), 1)
    )
    decode_mask = torch.cat(
        (attention_mask, torch.ones(len(lengths), 1, dtype=torch.long)), dim=1
    )
    decoded = model(
        next_token,
        attention_mask=decode_mask,
        past_key_values=cache,
        use_cache=True,
    ).logits
    for row, (length, reference_cache) in enumerate(
        zip(lengths, reference_caches, strict=True)
    ):
        reference = model(
            next_token[row : row + 1],
            attention_mask=torch.ones(1, length + 1, dtype=torch.long),
            past_key_values=reference_cache,
            use_cache=True,
        ).logits
        torch.testing.assert_close(
            decoded[row : row + 1], reference, atol=5e-4, rtol=5e-4
        )

    if cache.get_seq_length() != padded_length + 1:
        raise AssertionError("padded HF cache lost its physical generation length")
    chunk_size = _lod_config().chunk_size
    expected_group_lengths = sorted(
        {
            padded_length
            - ((padded_length - length) // chunk_size) * chunk_size
            + 1
            for length in lengths
        }
    )
    for layer in cache.layers:
        padding_runtime = layer._padding_runtime
        if padding_runtime is None:
            raise AssertionError("varied padding did not create grouped LOD state")
        actual_group_lengths = sorted(
            int(runtime.lod_cache.total_length)
            for runtime in padding_runtime.runtimes
        )
        if actual_group_lengths != expected_group_lengths:
            raise AssertionError("left padding entered a grouped LOD schedule")

    cache.batch_repeat_interleave(2)
    repeated_token = next_token.repeat_interleave(2, dim=0)
    repeated_mask = torch.cat(
        (decode_mask.repeat_interleave(2, dim=0), torch.ones(8, 1, dtype=torch.long)),
        dim=1,
    )
    model(
        repeated_token,
        attention_mask=repeated_mask,
        past_key_values=cache,
        use_cache=True,
    )
    beam_idx = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6])
    cache.reorder_cache(beam_idx)
    model(
        repeated_token.index_select(0, beam_idx),
        attention_mask=torch.cat(
            (
                repeated_mask.index_select(0, beam_idx),
                torch.ones(8, 1, dtype=torch.long),
            ),
            dim=1,
        ),
        past_key_values=cache,
        use_cache=True,
    )
    print(f"{type(model).__name__}: varied left-padding lifecycle passed")


@torch.no_grad()
def _check_leaf_storage_modes() -> None:
    cache_types = []
    for bits in (0, 4):
        torch.manual_seed(20)
        model = LlamaForCausalLM(LlamaConfig(**_common_config())).eval()
        config = PagedLODConfig(
            chunk_size=4,
            local_window=8,
            state_growth_factor=2.0,
            state_min_size=4,
            protected_prefix=1,
            max_routes=2,
            page_size=2,
            kv_bits=bits,
            quant_group_size=4,
        )
        install_hf_lod_attention(model, config=config, open_count=2)
        cache = new_hf_lod_cache(model)
        token = torch.randint(0, model.config.vocab_size, (1, 14))
        model(token[:, :12], past_key_values=cache, use_cache=True)
        result = model(token[:, 12:], past_key_values=cache, use_cache=True)
        if not bool(torch.isfinite(result.logits).all()):
            raise AssertionError("paged LOD produced non-finite logits")
        inner_cache = cache.layers[0].lod_cache
        if not isinstance(inner_cache, PagedLODCache):
            raise AssertionError("paged storage did not use a paged LOD cache")
        if inner_cache.leaves.key.bits != bits:
            raise AssertionError("paged storage used the wrong leaf encoding")
        cache_types.append(type(cache))
    if cache_types != [HFLODCache, HFLODCache]:
        raise AssertionError("BF16 and INT4 used different outer cache types")
    print("BF16 and INT4 share one HFLODCache lifecycle")


@torch.no_grad()
def _check_beam_reorder() -> None:
    torch.manual_seed(30)
    config = LlamaConfig(
        **_common_config(), pad_token_id=0, bos_token_id=1, eos_token_id=2
    )
    model = LlamaForCausalLM(config).eval()
    install_hf_lod_attention(model, config=_lod_config(), open_count=2)
    cache = new_hf_lod_cache(model)
    token = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    output = model.generate(
        token,
        attention_mask=torch.ones_like(token),
        past_key_values=cache,
        max_new_tokens=3,
        num_beams=2,
        do_sample=False,
    )
    if output.shape != (1, 13) or cache.batch_size != 2:
        raise AssertionError("beam generation did not reorder the LOD cache")
    print("beam expansion and cache reordering passed")


@torch.no_grad()
def _check_automatic_padded_generation() -> None:
    torch.manual_seed(32)
    config = LlamaConfig(
        **_common_config(), pad_token_id=0, bos_token_id=1, eos_token_id=2
    )
    model = LlamaForCausalLM(config).eval()
    install_hf_lod_attention(model, config=_lod_config(), open_count=2)
    token = torch.tensor(
        [
            [0, 0, 0, 1, 5, 6, 7, 8],
            [1, 9, 10, 11, 12, 13, 14, 15],
        ]
    )
    attention_mask = token.ne(0).long()
    output = model.generate(
        token,
        attention_mask=attention_mask,
        max_new_tokens=2,
        do_sample=False,
    )
    if output.shape != (2, 10):
        raise AssertionError("automatic padded HF LOD generation returned wrong shape")
    print("automatic varied-padding generation cache passed")


@torch.no_grad()
def _check_hybrid_cache() -> None:
    torch.manual_seed(35)
    # The optional causal-conv package is GPU-only, but Transformers resolves it
    # eagerly when installed.  Keep this CPU protocol test on its torch fallback.
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        function = getattr(modeling_qwen3_5, name)
        setattr(modeling_qwen3_5, name, getattr(function, "__wrapped__", function))
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    installed = install_hf_lod_attention(
        model, config=_lod_config(), open_count=2
    )
    if installed != ["model.layers.1.self_attn"]:
        raise AssertionError("hybrid installer selected the wrong attention layers")
    cache = new_hf_lod_cache(model)
    if not isinstance(cache, HybridHFLODCache):
        raise AssertionError("hybrid decoder did not receive a mixed LOD cache")
    token = torch.tensor(
        [[0, 0, 1, 3, 4, 5, 6, 7], [1, 8, 9, 10, 11, 12, 13, 14]]
    )
    attention_mask = token.ne(0).long()
    prefill = model(
        token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    decode_token = torch.tensor([[8], [15]])
    decode_mask = torch.cat(
        (attention_mask, torch.ones(2, 1, dtype=torch.long)), dim=1
    )
    decoded = model(
        decode_token,
        attention_mask=decode_mask,
        past_key_values=cache,
        use_cache=True,
    )
    if not bool(torch.isfinite(prefill.logits).all()):
        raise AssertionError("hybrid LOD prefill produced non-finite logits")
    if not bool(torch.isfinite(decoded.logits).all()):
        raise AssertionError("hybrid LOD decode produced non-finite logits")
    has_previous_state = cache.has_previous_state
    if callable(has_previous_state):
        has_previous_state = has_previous_state()
    if cache.get_seq_length() != 9 or not has_previous_state:
        raise AssertionError("hybrid recurrent/LOD cache lengths diverged")
    native_attention_values = []
    for layer_idx, layer in enumerate(cache.native_cache.layers):
        if layer_idx in cache.lod_layers:
            native_attention_values.extend(
                (getattr(layer, "keys", None), getattr(layer, "values", None))
            )
    if any(item is not None for item in native_attention_values):
        raise AssertionError("hybrid native cache retained duplicate attention K/V")
    generated = model.generate(
        token,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=0,
    )
    if generated.shape != (2, 9):
        raise AssertionError("automatic hybrid LOD generation returned wrong shape")
    print("Qwen3.5 mixed recurrent/LOD cache passed")


@torch.no_grad()
def _check_partial_dense_cache() -> None:
    """A dense native/LOD split must remain exact inside the local window."""
    torch.manual_seed(39)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    native = Qwen3ForCausalLM(config).eval()
    partial = deepcopy(native)
    installed = install_hf_lod_attention(
        partial,
        config=_lod_config(),
        open_count=2,
        layer_indices={1},
    )
    if installed != ["model.layers.1.self_attn"]:
        raise AssertionError("partial dense installer selected the wrong layer")
    cache = new_hf_lod_cache(partial)
    if not isinstance(cache, HybridHFLODCache):
        raise AssertionError("partial dense decoder did not receive a mixed cache")

    token = torch.tensor([[1, 3, 4, 5, 6, 7, 8]])
    reference = native(token, use_cache=False).logits
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    prefill = partial(
        token[:, :5],
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    cached_outputs = [prefill.logits]
    for position in range(5, token.size(1)):
        attention_mask = torch.ones(1, position + 1, dtype=torch.long)
        decoded = partial(
            token[:, position : position + 1],
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cached_outputs.append(decoded.logits)
    cached = torch.cat(cached_outputs, dim=1)
    torch.testing.assert_close(cached, reference, atol=1e-5, rtol=1e-5)
    if cache.get_seq_length() != token.size(1):
        raise AssertionError("partial dense cache length did not advance")
    print("partial dense native/LOD cache parity passed")


@torch.no_grad()
def _check_sliding_layer_selection() -> None:
    torch.manual_seed(38)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Gemma3ForCausalLM(config).eval()
    installed = install_hf_lod_attention(
        model, config=_lod_config(), open_count=2
    )
    if installed != ["model.layers.1.self_attn"]:
        raise AssertionError("LOD selected a Gemma sliding-attention layer")
    attention_backends = [
        layer.self_attn.config._attn_implementation for layer in model.model.layers
    ]
    if attention_backends != ["sdpa", "lod"]:
        raise AssertionError("mixed full/sliding attention dispatch is incorrect")
    cache = new_hf_lod_cache(model)
    if not isinstance(cache, HybridHFLODCache):
        raise AssertionError("mixed full/sliding model did not receive a mixed cache")
    token = torch.tensor(
        [[0, 0, 1, 3, 4, 5, 6, 7], [1, 8, 9, 10, 11, 12, 13, 14]]
    )
    attention_mask = token.ne(0).long()
    model(
        token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    decoded = model(
        torch.tensor([[8], [15]]),
        attention_mask=torch.cat(
            (attention_mask, torch.ones(2, 1, dtype=torch.long)), dim=1
        ),
        past_key_values=cache,
        use_cache=True,
    )
    if not bool(torch.isfinite(decoded.logits).all()):
        raise AssertionError("mixed full/sliding LOD decode produced non-finite logits")
    native_initialized = [layer.is_initialized for layer in cache.native_cache.layers]
    if native_initialized != [True, False]:
        raise AssertionError("full-attention K/V leaked into the native sliding cache")

    sliding_only = MistralForCausalLM(
        MistralConfig(**_common_config(), sliding_window=8)
    ).eval()
    try:
        install_hf_lod_attention(
            sliding_only, config=_lod_config(), open_count=2
        )
    except RuntimeError as error:
        if "no compatible causal" not in str(error):
            raise
    else:
        raise AssertionError("an all-sliding decoder was incorrectly replaced")
    print("full/sliding attention selection and mixed cache passed")


def main() -> None:
    _check_qk_norm_aware_routing()
    _check_state_clustering_radial_metric()
    _check_rope_frequency_routing()
    _check_rope_aware_state_clustering()
    _check_qk_norm_aware_state_clustering()
    common = _common_config()
    models = (
        LlamaForCausalLM(LlamaConfig(**common)).eval(),
        MistralForCausalLM(MistralConfig(**common, sliding_window=None)).eval(),
        Qwen3ForCausalLM(Qwen3Config(**common)).eval(),
    )
    for model in models:
        _check_model_family(model)
        _check_varied_left_padding(model)
    _check_leaf_storage_modes()
    _check_beam_reorder()
    _check_automatic_padded_generation()
    _check_hybrid_cache()
    _check_partial_dense_cache()
    _check_sliding_layer_selection()


if __name__ == "__main__":
    main()
