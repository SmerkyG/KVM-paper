#!/usr/bin/env python3
"""Break the paged exact-leaf branch into dispatch, kernel, and reduction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--state-growth-factor", type=float, default=8.0)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--layout", choices=("expert", "query"), default="expert")
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
        tokenizer,
        args.dataset,
        args.sequence_length,
        1,
        0,
        1,
    )[0][1].unsqueeze(0).expand(args.batch_size, -1).contiguous().to(device)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    for module in modules:
        module.leaf_block_m = args.block_m
        module.leaf_block_n = args.block_n
        module.leaf_num_warps = args.num_warps
        module.leaf_layout = args.layout

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
        for module in modules:
            if hasattr(module, "_lod_state"):
                delattr(module, "_lod_state")

        events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = defaultdict(list)
        for module in modules:
            module._lod_leaf_timing_events = events
        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)

    phase_ms = {
        phase: sum(float(begin.elapsed_time(end)) for begin, end in pairs)
        for phase, pairs in events.items()
    }
    total_ms = phase_ms.pop("total")
    record = {
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "attention_layers": len(modules),
        "block_m": args.block_m,
        "block_n": args.block_n,
        "num_warps": args.num_warps,
        "layout": args.layout,
        "exact_leaf_total_ms": total_ms,
        "phase_ms": phase_ms,
        "phase_fraction": {
            phase: milliseconds / total_ms
            for phase, milliseconds in phase_ms.items()
        },
        "calls": len(events["kernel"]),
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
