#!/usr/bin/env python3
"""Focused CPU checks for the slabbed PyTorch LOD prototype."""

from __future__ import annotations

from dataclasses import replace
import math

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention import _local_attention
from model.slabbed_lod_attention import (
    SlabbedLODConfig,
    SlabbedTwoLevelLODAttention,
)


def _config() -> SlabbedLODConfig:
    return SlabbedLODConfig(
        chunk_size=4,
        local_window=8,
        protected_prefix=1,
        max_routes=2,
        leaf_dtype=torch.float32,
        slab_size=8,
        slots_per_slab=4,
        routing_chunk_size=2,
        query_chunk_size=2,
    )


def _check_engine() -> None:
    torch.manual_seed(17)
    query = torch.randn(2, 4, 17, 6)
    key = torch.randn(2, 2, 17, 6)
    value = torch.randn(2, 2, 17, 5)
    scale = 1.0 / math.sqrt(6)

    first_two_slabs, _ = _local_attention(
        query[..., :16, :],
        key[..., :16, :],
        value[..., :16, :],
        scale=scale,
        query_offset=0,
    )
    configs = [
        replace(_config(), seed_selection="prefix"),
        replace(_config(), seed_selection="strided"),
        replace(
            _config(), seed_selection="strided", coarse_variance_bias=1.0
        ),
        replace(
            _config(),
            seed_selection="strided",
            exact_closed_mass_oracle=True,
        ),
        replace(
            _config(),
            seed_selection="strided",
            routing_leaf_mass_candidates=16,
        ),
    ]
    for config in configs:
        attention = SlabbedTwoLevelLODAttention(
            config, default_open_count=2
        )
        output, cache = attention(query, key, value, use_cache=True, scale=scale)
        if cache is None:
            raise AssertionError("slabbed attention did not return a cache")
        if cache.state.slot_count != 8 or int(cache.active_key.size(2)) != 1:
            raise AssertionError(
                "complete and active slabs were partitioned incorrectly"
            )
        if not bool(torch.all(cache.state.count.sum(dim=-1) == 16).item()):
            raise AssertionError("slab reduction lost or duplicated leaves")
        torch.testing.assert_close(output[..., :16, :], first_two_slabs)

        for split in (1, 7, 8, 9, 15, 16):
            incremental = SlabbedTwoLevelLODAttention(
                config, default_open_count=2
            )
            prefix, split_cache = incremental(
                query[..., :split, :],
                key[..., :split, :],
                value[..., :split, :],
                use_cache=True,
                scale=scale,
            )
            suffix, split_cache = incremental(
                query[..., split:, :],
                key[..., split:, :],
                value[..., split:, :],
                cache=split_cache,
                use_cache=True,
                scale=scale,
            )
            torch.testing.assert_close(
                torch.cat((prefix, suffix), dim=2),
                output,
                atol=2e-6,
                rtol=2e-6,
            )
            if split_cache is None or not torch.equal(
                split_cache.owner, cache.owner
            ):
                raise AssertionError(
                    "incremental slab routing differs from prefill"
                )

    merged_config = replace(
        _config(),
        local_slabs=1,
        seed_selection="strided",
        merge_group_slabs=2,
        merged_slots_per_group=4,
    )
    merged = SlabbedTwoLevelLODAttention(
        merged_config, default_open_count=2
    )
    merged_output, merged_cache = merged(
        query, key, value, use_cache=True, scale=scale
    )
    if merged_cache is None:
        raise AssertionError("delayed slab merge did not return a cache")
    merged_state, merged_owner = merged._remote_view(
        merged_cache.state, merged_cache.owner, remote_slabs=2
    )
    if merged_state.slot_count != 4:
        raise AssertionError("old slab group was not reduced to its target")
    if not bool(merged_state.count.sum(dim=-1).eq(16).all().item()):
        raise AssertionError("delayed slab merge lost leaves")
    if bool(
        (merged_owner.lt(0) | merged_owner.ge(merged_state.slot_count))
        .any()
        .item()
    ):
        raise AssertionError("delayed slab merge produced invalid owners")
    budgeted = SlabbedTwoLevelLODAttention(
        replace(merged_config, merge_budget_growth_factor=16.0),
        default_open_count=2,
    )
    budgeted_state, _ = budgeted._remote_view(
        merged_cache.state, merged_cache.owner, remote_slabs=2
    )
    if budgeted_state.slot_count != 8:
        raise AssertionError("delayed merge ran before exceeding its budget")
    prefix, merged_cache = merged(
        query[..., :13, :],
        key[..., :13, :],
        value[..., :13, :],
        use_cache=True,
        scale=scale,
    )
    suffix, _ = merged(
        query[..., 13:, :],
        key[..., 13:, :],
        value[..., 13:, :],
        cache=merged_cache,
        use_cache=True,
        scale=scale,
    )
    torch.testing.assert_close(
        torch.cat((prefix, suffix), dim=2),
        merged_output,
        atol=2e-6,
        rtol=2e-6,
    )
    print("slabbed engine exact-local and cache-boundary checks passed")


def _qwen() -> Qwen3_5ForCausalLM:
    # Avoid optional fused recurrent functions being compiled during this CPU
    # protocol check. LOD is installed only on the full-attention layer.
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
    return Qwen3_5ForCausalLM(config).eval()


def _check_hf_adapter() -> None:
    torch.manual_seed(23)
    model = _qwen()
    tokens = torch.randint(3, 64, (2, 12))
    native_first = model(tokens[:, :8], use_cache=False).logits
    installed = install_hf_lod_attention(
        model,
        config=_config(),
        open_count=2,
        engine_backend="torch",
        left_padding_mode="exact",
    )
    if installed != ["model.layers.1.self_attn"]:
        raise AssertionError(f"unexpected Qwen installation: {installed}")
    slabbed_first = model(tokens[:, :8], use_cache=False).logits
    torch.testing.assert_close(slabbed_first, native_first, atol=2e-5, rtol=2e-4)

    one_shot = model(tokens, use_cache=False).logits
    native_all = _qwen()
    native_all.load_state_dict(model.state_dict())
    native_all_logits = native_all(tokens, use_cache=False).logits
    torch.testing.assert_close(one_shot, native_all_logits, atol=2e-5, rtol=2e-4)
    cache = new_hf_lod_cache(model)
    prefix = model(
        tokens[:, :7], past_key_values=cache, use_cache=True
    ).logits
    suffix = model(
        tokens[:, 7:], past_key_values=cache, use_cache=True
    ).logits
    torch.testing.assert_close(
        torch.cat((prefix, suffix), dim=1), one_shot, atol=3e-5, rtol=3e-4
    )
    if cache.get_seq_length() != 12:
        raise AssertionError("HF and slabbed cache lengths diverged")
    print("generic HF Qwen3.5 slabbed-cache checks passed")


def main() -> None:
    _check_engine()
    _check_hf_adapter()


if __name__ == "__main__":
    main()
