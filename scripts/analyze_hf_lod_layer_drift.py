#!/usr/bin/env python3
"""Measure how a fixed LOD attention perturbation propagates by architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch
from transformers import AutoTokenizer

from model.hf_pytorch_lod_attention import install_hf_lod_attention
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument(
        "--routing-normalization",
        choices=("none", "query", "key", "both", "qk_norm_aware"),
        default="none",
    )
    parser.add_argument(
        "--routing-rope-filter",
        choices=("none", "local_window"),
        default="none",
    )
    parser.add_argument("--routing-rope-cutoff-factor", type=float, default=1.0)
    parser.add_argument("--routing-rope-jensen", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _last_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output:
        tensor = output[0]
    else:
        raise TypeError(f"cannot extract tensor from {type(output)!r}")
    return tensor.detach()[0, -1].float().cpu()


def _comparison(full: torch.Tensor, lod: torch.Tensor) -> dict[str, float]:
    difference = lod - full
    full_norm = torch.linalg.vector_norm(full)
    return {
        "full_rms": float(full.square().mean().sqrt()),
        "lod_rms": float(lod.square().mean().sqrt()),
        "error_rms": float(difference.square().mean().sqrt()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference) / full_norm.clamp_min(1e-12)
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(full, lod, dim=0)
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    _, sequence = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        samples=1,
        rank=0,
        world_size=1,
    )[0]
    sequence = sequence.unsqueeze(0).to(device)
    model, acceleration = load_model(args.checkpoint, device)

    decoder_layers: dict[int, torch.nn.Module] = {}
    attention_layers: dict[int, torch.nn.Module] = {}
    for module_name, module in model.named_modules():
        layer_idx = getattr(module, "layer_idx", None)
        if not isinstance(layer_idx, int):
            match = re.search(r"(?:^|\.)layers\.(\d+)$", module_name)
            if match is not None:
                layer_idx = int(match.group(1))
        if not isinstance(layer_idx, int):
            continue
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            attention_layers[layer_idx] = module
        elif hasattr(module, "input_layernorm"):
            decoder_layers[layer_idx] = module

    current: dict[str, dict[int, torch.Tensor]] = {}
    phase = "full"

    def record(kind: str, layer_idx: int):
        def hook(_module, _inputs, output):
            current.setdefault(f"{phase}_{kind}", {})[layer_idx] = _last_tensor(
                output
            )

        return hook

    handles = []
    for layer_idx, module in decoder_layers.items():
        handles.append(module.register_forward_hook(record("hidden", layer_idx)))
    for layer_idx, module in attention_layers.items():
        handles.append(module.register_forward_hook(record("attention", layer_idx)))

    try:
        with torch.inference_mode():
            full_logits = model(input_ids=sequence, use_cache=False).logits[
                0, -1
            ].detach().float().cpu()
        installed = install_hf_lod_attention(
            model,
            config=PagedLODConfig(
                chunk_size=256,
                local_window=512,
                state_growth_factor=16,
                state_min_size=256,
                protected_prefix=1,
                page_size=16,
                routing_normalization=args.routing_normalization,
                routing_rope_filter=args.routing_rope_filter,
                routing_rope_cutoff_factor=args.routing_rope_cutoff_factor,
                routing_rope_jensen=args.routing_rope_jensen,
            ),
            open_count=8,
            engine_backend="kernel",
        )
        phase = "lod"
        with torch.inference_mode():
            lod_logits = model(input_ids=sequence, use_cache=False).logits[
                0, -1
            ].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    installed_indices = {
        int(name.split(".")[-2])
        for name in installed
        if name.split(".")[-2].isdigit()
    }
    layer_records = []
    for layer_idx in sorted(decoder_layers):
        decoder = decoder_layers[layer_idx]
        attention = attention_layers.get(layer_idx)
        record_payload: dict[str, Any] = {
            "layer": layer_idx,
            "block_type": getattr(decoder, "block_type", None),
            "sandwich_attention_norm": hasattr(
                decoder, "pre_feedforward_layernorm"
            ),
        }
        if attention is not None:
            expected_query_width = int(
                getattr(attention, "head_dim", 0)
                * getattr(attention.config, "num_attention_heads", 0)
            )
            record_payload.update(
                attention_modified=layer_idx in installed_indices,
                q_norm=hasattr(attention, "q_norm"),
                k_norm=hasattr(attention, "k_norm"),
                output_gate=(
                    expected_query_width > 0
                    and int(attention.q_proj.out_features)
                    == 2 * expected_query_width
                ),
                use_rope=getattr(attention, "use_rope", None),
            )
        if layer_idx in current.get("full_attention", {}):
            record_payload["attention_output"] = _comparison(
                current["full_attention"][layer_idx],
                current["lod_attention"][layer_idx],
            )
        record_payload["hidden_output"] = _comparison(
            current["full_hidden"][layer_idx],
            current["lod_hidden"][layer_idx],
        )
        layer_records.append(record_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "sequence_length": args.sequence_length,
                "routing_normalization": args.routing_normalization,
                "routing_rope_filter": args.routing_rope_filter,
                "routing_rope_cutoff_factor": args.routing_rope_cutoff_factor,
                "routing_rope_jensen": args.routing_rope_jensen,
                "acceleration": acceleration,
                "lod_attention_layers": installed,
                "logits": {
                    **_comparison(full_logits, lod_logits),
                    "full_argmax": int(full_logits.argmax()),
                    "lod_argmax": int(lod_logits.argmax()),
                },
                "layers": layer_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
