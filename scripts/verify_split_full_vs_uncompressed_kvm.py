#!/usr/bin/env python3
"""Compare split-full attention with classic KVM when every token is appended."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from model.rwkv7_backbone import MixerConfig, RWKV7BackboneForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def run_model(
    checkpoint: Path,
    config: MixerConfig,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    model, loading_info = RWKV7BackboneForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
    )
    relevant_loading_info = {key: value for key, value in loading_info.items() if value}
    if relevant_loading_info:
        raise AssertionError(f"checkpoint mismatch: {relevant_loading_info}")
    model = model.cuda().eval()
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = [
        block.attn.c_proj.register_forward_hook(
            lambda _module, _inputs, output, layer_idx=layer_idx: layer_outputs.__setitem__(
                layer_idx, output[:, -1].detach().float().cpu()
            )
        )
        for layer_idx, block in enumerate(model.transformer.h)
    ]
    try:
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, use_cache=False, logits_to_keep=1
            ).logits.detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
        del model
        torch.cuda.empty_cache()
    return logits, layer_outputs


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("equivalence verification requires a CUDA/ROCm GPU")
    checkpoint = args.checkpoint.resolve()
    split_config = MixerConfig.from_pretrained(checkpoint)
    kvm_config = copy.deepcopy(split_config)
    kvm_config.token_mixer_class_path = "model.kvm_mixer.SequenceMixer"
    kvm_config.state_budget_mode = "fixed"
    kvm_config.state_min_len = args.length
    kvm_config.n_max_d_chunks = max(
        int(kvm_config.n_max_d_chunks),
        (args.length + int(kvm_config.chunk_len) - 1) // int(kvm_config.chunk_len),
    )
    kvm_config.kvm_use_merge_gate_keys = 0
    kvm_config.kvm_use_merge_gate_values = 0
    kvm_config.kvm_use_vlens = 0

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        split_config.vocab_size,
        (1, args.length),
        generator=generator,
        device="cuda",
    )
    split_logits, split_layers = run_model(checkpoint, split_config, input_ids)
    kvm_logits, kvm_layers = run_model(checkpoint, kvm_config, input_ids)

    layer_stats = {
        layer_idx: {
            "max_abs": float(
                (split_layers[layer_idx] - kvm_layers[layer_idx]).abs().max().item()
            ),
            "mean_abs": float(
                (split_layers[layer_idx] - kvm_layers[layer_idx]).abs().mean().item()
            ),
        }
        for layer_idx in split_layers
    }
    logit_delta = (split_logits - kvm_logits).abs()
    print(
        {
            "length": args.length,
            "logit_max_abs": float(logit_delta.max().item()),
            "logit_mean_abs": float(logit_delta.mean().item()),
            "last_token_top1_equal": bool(
                split_logits[:, -1].argmax(-1).eq(kvm_logits[:, -1].argmax(-1)).all()
            ),
            "layer_attention_output": layer_stats,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
