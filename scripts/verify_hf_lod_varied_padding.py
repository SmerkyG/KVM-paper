#!/usr/bin/env python3
"""GPU smoke for varied left padding through the generic HF LOD backend."""

from __future__ import annotations

import argparse

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
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention import LODConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-family", choices=("llama", "mistral", "qwen3"), default="llama"
    )
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="kernel"
    )
    return parser.parse_args()


def build_model(family: str):
    common = {
        "vocab_size": 128,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 128,
    }
    if family == "llama":
        return LlamaForCausalLM(LlamaConfig(**common))
    if family == "mistral":
        return MistralForCausalLM(MistralConfig(**common, sliding_window=None))
    return Qwen3ForCausalLM(Qwen3Config(**common))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the varied-padding kernel smoke requires a GPU")
    torch.manual_seed(41)
    device = torch.device("cuda")
    model = build_model(args.model_family).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    install_hf_lod_attention(
        model,
        config=LODConfig(
            chunk_size=8,
            local_window=16,
            state_growth_factor=2.0,
            state_min_size=8,
            protected_prefix=1,
            max_routes=2,
        ),
        open_count=2,
        engine_backend=args.engine_backend,
    )

    # Keep enough remote state for the optimized decode kernel's eight routing
    # candidates while retaining varied, non-chunk-aligned logical lengths.
    lengths = (33, 40, 40, 48)
    padded_length = max(lengths)
    token = torch.zeros(len(lengths), padded_length, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(token)
    for row, length in enumerate(lengths):
        token[row, -length:] = torch.randint(
            3, model.config.vocab_size, (length,), device=device
        )
        attention_mask[row, -length:] = 1

    cache = new_hf_lod_cache(model)
    prefill = model(
        token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    ).logits
    if not bool(torch.isfinite(prefill).all()):
        raise AssertionError("varied-padding prefill produced non-finite logits")

    next_token = torch.randint(
        3, model.config.vocab_size, (len(lengths), 1), device=device
    )
    decode_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=1)
    decoded = model(
        next_token,
        attention_mask=decode_mask,
        past_key_values=cache,
        use_cache=True,
    ).logits
    if not bool(torch.isfinite(decoded).all()):
        raise AssertionError("varied-padding decode produced non-finite logits")
    for row, length in enumerate(lengths):
        valid_token = token[row : row + 1, -length:]
        reference_cache = new_hf_lod_cache(model)
        model(
            valid_token,
            attention_mask=torch.ones_like(valid_token),
            past_key_values=reference_cache,
            use_cache=True,
        )
        reference = model(
            next_token[row : row + 1],
            attention_mask=torch.ones(
                1, length + 1, dtype=torch.long, device=device
            ),
            past_key_values=reference_cache,
            use_cache=True,
        ).logits
        torch.testing.assert_close(
            decoded[row : row + 1].float(),
            reference.float(),
            atol=5e-2,
            rtol=5e-2,
        )

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
            raise AssertionError("left padding entered the LOD state schedule")
    generated = model.generate(
        token,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=0,
    )
    if generated.shape != (len(lengths), padded_length + 1):
        raise AssertionError("automatic varied-padding generation returned wrong shape")
    print(
        f"{args.model_family} {args.engine_backend} varied-padding smoke passed: "
        f"batch={len(lengths)} physical={padded_length + 1} "
        f"logical={expected_group_lengths}"
    )


if __name__ == "__main__":
    main()
