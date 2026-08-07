#!/usr/bin/env python3
"""Compare one-shot split-full logits with prefill-plus-one-token decode."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[511, 512, 513])
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("cache parity diagnosis requires a CUDA/ROCm GPU")

    checkpoint = args.checkpoint.resolve()
    config = MixerConfig.from_pretrained(checkpoint)
    model = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    max_length = max(args.lengths)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, max_length),
        generator=generator,
        device="cuda",
    )

    capture: dict[int, torch.Tensor] | None = None

    def capture_layer(layer_idx: int):
        def hook(_module, _inputs, output):
            if capture is not None:
                capture[layer_idx] = output[:, -1].detach().float().cpu()

        return hook

    hooks = [
        block.register_forward_hook(capture_layer(layer_idx))
        for layer_idx, block in enumerate(model.transformer.h)
    ]

    try:
        with torch.inference_mode():
            for length in args.lengths:
                input_ids = tokens[:, :length]

                full_layers: dict[int, torch.Tensor] = {}
                capture = full_layers
                full = model(input_ids=input_ids, use_cache=False).logits[:, -1]

                capture = None
                prefix = model(
                    input_ids=input_ids[:, :-1],
                    cache_position=torch.arange(length - 1, device="cuda"),
                    use_cache=True,
                )

                cached_layers: dict[int, torch.Tensor] = {}
                capture = cached_layers
                cached = model(
                    input_ids=input_ids[:, -1:],
                    past_key_values=prefix.past_key_values,
                    cache_position=torch.tensor([length - 1], device="cuda"),
                    use_cache=True,
                ).logits[:, -1]
                capture = None

                logit_delta = (full - cached).abs()
                layer_max = {
                    layer_idx: float(
                        (full_layers[layer_idx] - cached_layers[layer_idx])
                        .abs()
                        .max()
                        .item()
                    )
                    for layer_idx in full_layers
                }
                print(
                    {
                        "length": length,
                        "logit_max_abs": float(logit_delta.max().item()),
                        "logit_mean_abs": float(logit_delta.mean().item()),
                        "top1_equal": bool(full.argmax(-1).eq(cached.argmax(-1)).all()),
                        "first_layer_max_abs": layer_max[0],
                        "last_layer_max_abs": layer_max[len(layer_max) - 1],
                        "layer_max_abs": layer_max,
                    },
                    flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()
