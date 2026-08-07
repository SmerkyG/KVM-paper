#!/usr/bin/env python3
"""CPU checks for the registered, model-independent HF LOD backend."""

from __future__ import annotations

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from model.hf_pytorch_lod_attention import (
    HFLODCache,
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODCache, PagedLODConfig


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
    if output.shape != (1, 13) or cache.max_batch_size != 2:
        raise AssertionError("beam generation did not reorder the LOD cache")
    print("beam expansion and cache reordering passed")


def main() -> None:
    common = _common_config()
    models = (
        LlamaForCausalLM(LlamaConfig(**common)).eval(),
        MistralForCausalLM(MistralConfig(**common, sliding_window=None)).eval(),
        Qwen3ForCausalLM(Qwen3Config(**common)).eval(),
    )
    for model in models:
        _check_model_family(model)
    _check_leaf_storage_modes()
    _check_beam_reorder()


if __name__ == "__main__":
    main()
