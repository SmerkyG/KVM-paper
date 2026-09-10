from __future__ import annotations

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from lod_attention import LODMode, install


def test_qwen38_text_model_installs_only_its_global_attention() -> None:
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=6_144,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=2,
        layer_types=["full_attention"],
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config).eval()

    assert install(model, mode="three-tier-int4") == [
        "model.layers.0.self_attn"
    ]
    settings = model.model.layers[0].self_attn._hf_lod_settings
    assert settings.mode is LODMode.THREE_TIER_INT4
    assert settings.config.kv_bits == 4
    assert settings.config.max_routes == 4
