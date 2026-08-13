#!/usr/bin/env python3
"""Measure native DiffusionGemma global-attention concentration on ProLong."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DiffusionGemmaForBlockDiffusion

from model.hf_diffusion_gemma_lod_attention import _project_qkv
from scripts.compare_diffusion_gemma_lod_loss import _corrupt_canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="google/diffusiongemma-26B-A4B-it"
    )
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--exact-prefix", type=int, default=512)
    parser.add_argument(
        "--corruption-rates", type=float, nargs="+", default=(0.25, 1.0)
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_sequence(
    tokenizer: AutoTokenizer, dataset_name: str, sequence_length: int
) -> torch.Tensor:
    dataset = load_dataset(dataset_name, split="train", streaming=True).shuffle(
        seed=42, buffer_size=1_000
    )
    for document in dataset:
        if int(document.get("length", sequence_length)) < sequence_length:
            continue
        tokens = tokenizer(
            document["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length,
            return_attention_mask=False,
        )["input_ids"]
        if len(tokens) == sequence_length:
            return torch.tensor(tokens, dtype=torch.long)
    raise RuntimeError("no sufficiently long ProLong document found")


def attention_statistics(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    scale: float,
    prefix_length: int,
    exact_prefix: int,
) -> dict[str, object]:
    groups = int(query.size(1)) // int(key.size(1))
    key = key.repeat_interleave(groups, dim=1)
    scores = torch.matmul(
        query.float(), key.float().transpose(-1, -2)
    ) * float(scale)
    probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
    top_probability, top_position = probability.max(dim=-1)
    top8_mass = probability.topk(8, dim=-1).values.sum(dim=-1)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(dim=-1)
    exact_start = max(prefix_length - exact_prefix, 0)
    exact_mass = probability[..., exact_start:].sum(dim=-1)
    prefix_mass = probability[..., :prefix_length].sum(dim=-1)
    position_mass = probability.mean(dim=(0, 1, 2))
    position_top1 = torch.bincount(
        top_position.flatten(), minlength=int(probability.size(-1))
    ).float() / top_position.numel()
    mass_values, mass_positions = position_mass.topk(10)
    top1_values, top1_positions = position_top1.topk(10)

    def per_head(value: torch.Tensor) -> list[float]:
        return value.mean(dim=(0, 2)).cpu().tolist()

    return {
        "query_rms": float(query.float().square().mean().sqrt()),
        "key_rms": float(key.float().square().mean().sqrt()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "top1_mass": float(top_probability.mean()),
        "top8_mass": float(top8_mass.mean()),
        "entropy": float(entropy.mean()),
        "effective_keys": float(entropy.mean().exp()),
        "prefix_mass": float(prefix_mass.mean()),
        "exact_field_mass": float(exact_mass.mean()),
        "first_token_mass": float(probability[..., 0].mean()),
        "first_256_mass": float(probability[..., :256].sum(dim=-1).mean()),
        "top1_at_first_token_rate": float(top_position.eq(0).float().mean()),
        "top1_in_first_256_rate": float(top_position.lt(256).float().mean()),
        "top1_in_exact_field_rate": float(
            top_position.ge(exact_start).float().mean()
        ),
        "per_head_top1_mass": per_head(top_probability),
        "per_head_top8_mass": per_head(top8_mass),
        "per_head_entropy": per_head(entropy),
        "per_head_exact_field_mass": per_head(exact_mass),
        "per_head_first_token_mass": probability[..., 0].mean(dim=(0, 2)).cpu().tolist(),
        "highest_mean_mass_positions": [
            {"position": int(position), "mass": float(mass)}
            for mass, position in zip(mass_values, mass_positions, strict=True)
        ],
        "most_common_top1_positions": [
            {"position": int(position), "rate": float(rate)}
            for rate, position in zip(top1_values, top1_positions, strict=True)
        ],
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    sequence = select_sequence(tokenizer, args.dataset, args.sequence_length).to(
        device
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device).eval()
    canvas_length = int(model.config.canvas_length)
    prefix_length = args.sequence_length - canvas_length
    prefix = sequence[:prefix_length].unsqueeze(0)
    target = sequence[prefix_length:].unsqueeze(0)
    attention_mask = torch.ones_like(prefix)
    decoder_mask = torch.ones(
        1, args.sequence_length, dtype=torch.long, device=device
    )
    current_rate = {"value": None}
    records: dict[str, dict[str, object]] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(module, positional, keyword):
            hidden_states = (
                keyword["hidden_states"]
                if "hidden_states" in keyword
                else positional[0]
            )
            position_embeddings = (
                keyword["position_embeddings"]
                if "position_embeddings" in keyword
                else positional[1]
            )
            cache = keyword.get("past_key_values")
            query, canvas_key, _ = _project_qkv(
                module, hidden_states, position_embeddings
            )
            prefix_key = cache.layers[layer_index].keys[..., :prefix_length, :]
            key = torch.cat((prefix_key, canvas_key), dim=2)
            rate = str(current_rate["value"])
            records.setdefault(rate, {})[str(layer_index)] = attention_statistics(
                query,
                key,
                scale=float(module.scaling),
                prefix_length=prefix_length,
                exact_prefix=args.exact_prefix,
            )

        return hook

    for layer_index, layer in enumerate(model.model.decoder.layers):
        if not bool(getattr(layer.self_attn, "is_sliding", False)):
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    make_hook(layer_index), with_kwargs=True
                )
            )

    with torch.inference_mode():
        encoder = model.model.encoder(
            input_ids=prefix, attention_mask=attention_mask
        )
        for rate_index, rate in enumerate(args.corruption_rates):
            canvas, _ = _corrupt_canvas(
                target,
                rate,
                vocabulary_size=int(model.config.text_config.vocab_size),
                sample=0,
                rate_index=rate_index,
            )
            current_rate["value"] = rate
            model(
                past_key_values=encoder.past_key_values,
                decoder_input_ids=canvas,
                decoder_attention_mask=decoder_mask,
            )

    for handle in handles:
        handle.remove()
    output = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "prefix_length": prefix_length,
        "canvas_length": canvas_length,
        "exact_prefix": args.exact_prefix,
        "rates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
