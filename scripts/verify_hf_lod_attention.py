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
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)

from model.hf_lod_hybrid_cache import HybridHFLODCache
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
    expected_group_lengths = sorted({length + 1 for length in lengths})
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
    if output.shape != (1, 13) or cache.max_batch_size != 2:
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
    if cache.get_seq_length() != 9 or not cache.has_previous_state:
        raise AssertionError("hybrid recurrent/LOD cache lengths diverged")
    if any(item is not None for item in cache.key_cache + cache.value_cache):
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


def main() -> None:
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


if __name__ == "__main__":
    main()
