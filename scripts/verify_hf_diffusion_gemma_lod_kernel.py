#!/usr/bin/env python3
"""GPU smoke test for DiffusionGemma's recursive Triton LOD adapter."""

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
    raise SystemExit("This check requires Transformers 5.11 or newer") from exc

from model.hf_diffusion_gemma_lod_attention import (
    install_diffusion_gemma_lod_attention,
)
from model.pytorch_lod_attention_paged import PagedLODConfig


def _model() -> DiffusionGemmaModel:
    text = DiffusionGemmaTextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=512,
        global_head_dim=512,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=32,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=64,
        max_position_embeddings=256,
    )
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
    return DiffusionGemmaModel(
        DiffusionGemmaConfig(
            text_config=text,
            vision_config=vision,
            canvas_length=16,
        )
    ).eval()


def _decode(model, canvas, cache, mask):
    return model.decoder(
        decoder_input_ids=canvas,
        past_key_values=cache,
        decoder_attention_mask=mask,
    ).last_hidden_state


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("DiffusionGemma kernel smoke requires a CUDA or ROCm GPU")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(11)
    native = _model().to(device=device, dtype=dtype)
    lod = copy.deepcopy(native)
    assert install_diffusion_gemma_lod_attention(
        lod,
        config=PagedLODConfig(
            chunk_size=16,
            local_window=32,
            state_growth_factor=16,
            state_min_size=32,
            protected_prefix=1,
            max_routes=8,
            leaf_dtype=dtype,
            page_size=16,
            kv_bits=0,
            quant_group_size=32,
        ),
        open_count=8,
        engine_backend="kernel",
    ) == [1]

    prompt = torch.randint(3, 256, (2, 48), device=device)
    prompt_mask = torch.ones(2, 48, dtype=torch.long, device=device)
    prompt_mask[1, :7] = 0
    prompt[1, :7] = 0
    canvas = torch.randint(3, 256, (2, 16), device=device)
    decoder_mask = torch.cat(
        (prompt_mask, torch.ones(2, 16, dtype=torch.long, device=device)), dim=1
    )

    with torch.inference_mode():
        native_encoder = native.encoder.language_model(
            input_ids=prompt, attention_mask=prompt_mask
        )
        lod_encoder = lod.encoder.language_model(
            input_ids=prompt, attention_mask=prompt_mask
        )
        native_output = _decode(
            native, canvas, native_encoder.past_key_values, decoder_mask
        )
        lod_output = _decode(lod, canvas, lod_encoder.past_key_values, decoder_mask)
        repeated = _decode(lod, canvas, lod_encoder.past_key_values, decoder_mask)

    torch.testing.assert_close(lod_output, native_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(repeated, lod_output, atol=0, rtol=0)

    next_canvas = torch.randint(3, 256, (2, 16), device=device)
    next_decoder_mask = torch.cat(
        (decoder_mask, torch.ones(2, 16, dtype=torch.long, device=device)), dim=1
    )
    with torch.inference_mode():
        native.encoder.language_model(
            input_ids=canvas,
            attention_mask=decoder_mask,
            past_key_values=native_encoder.past_key_values,
        )
        lod.encoder.language_model(
            input_ids=canvas,
            attention_mask=decoder_mask,
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
    torch.testing.assert_close(lod_next, native_next, atol=2e-2, rtol=2e-2)

    context = lod_encoder.past_key_values._diffusion_gemma_lod_context[1]
    assert [runtime.prompt_length for runtime in context.grouped.runtimes] == [41, 48]
    assert [runtime.lod_cache.total_length for runtime in context.grouped.runtimes] == [57, 64]
    print(
        "DiffusionGemma kernel smoke passed; max native delta:",
        max(
            float((lod_output - native_output).abs().max()),
            float((lod_next - native_next).abs().max()),
        ),
    )


if __name__ == "__main__":
    main()
