#!/usr/bin/env python3
"""CPU smoke test for the external DiffusionGemma LOD adapter.

The repository currently pins a Transformers release from before
DiffusionGemma.  Run this check without changing the project lockfile:

    uv run --with transformers==5.11.0 \
        python -m scripts.verify_hf_diffusion_gemma_lod
"""

from __future__ import annotations

import copy

import torch

try:
    from transformers import (
        DiffusionGemmaConfig,
        DiffusionGemmaTextConfig,
        SiglipVisionConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaModel,
    )
except ImportError as exc:
    raise SystemExit(
        "This smoke test requires Transformers 5.11 or newer; see its module docstring."
    ) from exc

from model.hf_diffusion_gemma_lod_attention import (
    install_diffusion_gemma_lod_attention,
    uninstall_diffusion_gemma_lod_attention,
)
from model.pytorch_lod_attention_paged import PagedLODConfig


def _tiny_model() -> DiffusionGemmaModel:
    text = DiffusionGemmaTextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=16,
        max_position_embeddings=128,
    )
    # The text-only test never calls the vision tower.  A tiny registered
    # vision config is still needed because DiffusionGemma constructs it.
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=8,
        patch_size=4,
    )
    vision.rms_norm_eps = 1e-6
    vision.output_proj_dims = 16
    config = DiffusionGemmaConfig(
        text_config=text,
        vision_config=vision,
        canvas_length=4,
    )
    return DiffusionGemmaModel(config).eval()


def _lod_config() -> PagedLODConfig:
    return PagedLODConfig(
        chunk_size=4,
        local_window=8,
        state_growth_factor=8,
        state_min_size=8,
        protected_prefix=1,
        max_routes=8,
        leaf_dtype=torch.float32,
        page_size=4,
        kv_bits=0,
        quant_group_size=4,
    )


def _decode(
    model: DiffusionGemmaModel,
    canvas: torch.Tensor,
    cache,
    mask: torch.Tensor,
) -> torch.Tensor:
    return model.decoder(
        decoder_input_ids=canvas,
        past_key_values=cache,
        decoder_attention_mask=mask,
    ).last_hidden_state


def main() -> None:
    torch.manual_seed(7)
    native = _tiny_model()
    lod = copy.deepcopy(native)
    original_encoder_forward = lod.encoder.language_model.forward
    original_attention_forward = lod.encoder.language_model.layers[1].self_attn.forward

    patched = install_diffusion_gemma_lod_attention(
        lod,
        config=_lod_config(),
        open_count=8,
        engine_backend="torch",
    )
    assert patched == [1], patched
    # Installation with identical settings is intentionally idempotent.
    assert install_diffusion_gemma_lod_attention(
        lod,
        config=_lod_config(),
        open_count=8,
        engine_backend="torch",
    ) == [1]

    prompt = torch.randint(3, 128, (2, 12))
    prompt_mask = torch.ones(2, 12, dtype=torch.long)
    prompt_mask[1, :3] = 0
    prompt[1, :3] = 0
    first_canvas = torch.randint(3, 128, (2, 4))
    first_decoder_mask = torch.cat(
        (prompt_mask, torch.ones(2, 4, dtype=torch.long)), dim=1
    )

    with torch.no_grad():
        native_encoder = native.encoder.language_model(
            input_ids=prompt, attention_mask=prompt_mask
        )
        lod_encoder = lod.encoder.language_model(
            input_ids=prompt, attention_mask=prompt_mask
        )
        native_first = _decode(
            native,
            first_canvas,
            native_encoder.past_key_values,
            first_decoder_mask,
        )
        lod_first = _decode(
            lod, first_canvas, lod_encoder.past_key_values, first_decoder_mask
        )
        lod_repeat = _decode(
            lod, first_canvas, lod_encoder.past_key_values, first_decoder_mask
        )

    torch.testing.assert_close(lod_first, native_first, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lod_repeat, lod_first, atol=0, rtol=0)

    context = lod_encoder.past_key_values._diffusion_gemma_lod_context[1]
    assert [runtime.prompt_length for runtime in context.grouped.runtimes] == [9, 12]
    assert [runtime.indices.tolist() for runtime in context.grouped.runtimes] == [[1], [0]]

    # Finalizing a canvas causally appends it to both the native cache and each
    # logical-length LOD group.  The next canvas again reads the state without
    # mutating it.
    next_canvas = torch.randint(3, 128, (2, 4))
    extended_mask = first_decoder_mask
    next_decoder_mask = torch.cat(
        (extended_mask, torch.ones(2, 4, dtype=torch.long)), dim=1
    )
    with torch.no_grad():
        native.encoder.language_model(
            input_ids=first_canvas,
            attention_mask=extended_mask,
            past_key_values=native_encoder.past_key_values,
        )
        lod.encoder.language_model(
            input_ids=first_canvas,
            attention_mask=extended_mask,
            past_key_values=lod_encoder.past_key_values,
        )
        native_next = _decode(
            native,
            next_canvas,
            native_encoder.past_key_values,
            next_decoder_mask,
        )
        lod_next = _decode(
            lod,
            next_canvas,
            lod_encoder.past_key_values,
            next_decoder_mask,
        )
    torch.testing.assert_close(lod_next, native_next, atol=2e-6, rtol=2e-6)
    assert [
        runtime.lod_cache.total_length for runtime in context.grouped.runtimes
    ] == [13, 16]

    restored = uninstall_diffusion_gemma_lod_attention(lod)
    assert restored == [1], restored
    assert lod.encoder.language_model.forward.__func__ is original_encoder_forward.__func__
    assert (
        lod.encoder.language_model.layers[1].self_attn.forward.__func__
        is original_attention_forward.__func__
    )
    print("DiffusionGemma LOD adapter smoke test passed")


if __name__ == "__main__":
    main()
