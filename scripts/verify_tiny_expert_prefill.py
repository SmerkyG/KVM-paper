#!/usr/bin/env python3
"""Compare raw and small-N-specialized expert prefill model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.probe_qwen35_lod_niah import load_text_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--prefill-two-level-topk", type=int, default=8)
    parser.add_argument("--leaf-block-m", type=int, default=16)
    parser.add_argument("--leaf-block-n", type=int, default=32)
    parser.add_argument("--leaf-num-warps", type=int, default=2)
    parser.add_argument("--tiny-expert-max", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--tiny-max-context", type=int, default=65_536)
    parser.add_argument("--reduce-num-warps", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_text_model(
        args.checkpoint,
        "two_level",
        8,
        args.state_growth_factor,
        device,
        "paged",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    sequence = select_sequences(
        tokenizer, args.dataset, args.sequence_length, 1, 0, 1
    )[0][1].unsqueeze(0).expand(args.batch_size, -1).contiguous().to(device)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    for module in modules:
        module.prefill_two_level_topk = args.prefill_two_level_topk
        module.virtual_page_storage = True
        module.leaf_block_m = args.leaf_block_m
        module.leaf_block_n = args.leaf_block_n
        module.leaf_num_warps = args.leaf_num_warps
        module.leaf_tiny_block_m = 8
        module.leaf_tiny_num_warps = 1
        module.leaf_tiny_expert_max = args.tiny_expert_max
        module.leaf_tiny_max_context = args.tiny_max_context
        module.leaf_reduce_num_warps = args.reduce_num_warps

    def run(layout: str) -> torch.Tensor:
        for module in modules:
            module.leaf_layout = layout
            if hasattr(module, "_lod_state"):
                delattr(module, "_lod_state")
        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        return result.logits.detach().float()

    with torch.inference_mode():
        raw = run("expert")
        tiny = run("expert_tiny")

    difference = (tiny - raw).abs()
    record = {
        "checkpoint": args.checkpoint,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "attention_layers": len(modules),
        "state_growth_factor": args.state_growth_factor,
        "prefill_two_level_topk": args.prefill_two_level_topk,
        "leaf_block_m": args.leaf_block_m,
        "leaf_block_n": args.leaf_block_n,
        "leaf_num_warps": args.leaf_num_warps,
        "tiny_expert_max": args.tiny_expert_max,
        "tiny_max_context": args.tiny_max_context,
        "reduce_num_warps": args.reduce_num_warps,
        "raw_finite": bool(torch.isfinite(raw).all().item()),
        "tiny_finite": bool(torch.isfinite(tiny).all().item()),
        "logit_max_abs_error": float(difference.max().item()),
        "logit_mean_abs_error": float(difference.mean().item()),
        "logit_root_mean_square_error": float(
            torch.sqrt(torch.mean(difference.square())).item()
        ),
        "raw_top1": raw.argmax(dim=-1).cpu().tolist(),
        "tiny_top1": tiny.argmax(dim=-1).cpu().tolist(),
        "top1_match": bool(torch.equal(raw.argmax(dim=-1), tiny.argmax(dim=-1))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
