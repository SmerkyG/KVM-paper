#!/usr/bin/env python3
"""Separate Qwen3.5's dense attention kernel time from model prefill time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from scripts.probe_qwen35_lod_niah import load_text_model
from scripts.profile_qwen35_prefill_total import select_profile_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_text_model(
        args.checkpoint,
        "full",
        8,
        16.0,
        device,
        require_fla_fast_path=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sequence = (
        select_profile_sequence(tokenizer, args.dataset, args.sequence_length)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
        .to(device)
    )

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm

        original = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
        if original is None:
            raise RuntimeError("the Qwen full-attention SDPA interface is unavailable")
        attention_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        def profiled_attention(*call_args, **call_kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = original(*call_args, **call_kwargs)
            end.record()
            attention_events.append((begin, end))
            return result

        totals: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        result = None
        ALL_ATTENTION_FUNCTIONS["sdpa"] = profiled_attention
        try:
            for _ in range(args.repeats):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
                end.record()
                totals.append((begin, end))
            torch.cuda.synchronize(device)
        finally:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original

    if result is None or args.repeats <= 0:
        raise ValueError("profile repeats must be positive")
    total_ms = sum(begin.elapsed_time(end) for begin, end in totals) / args.repeats
    attention_ms = (
        sum(begin.elapsed_time(end) for begin, end in attention_events) / args.repeats
    )
    record = {
        "checkpoint": args.checkpoint,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "attention_calls_per_repeat": len(attention_events) // args.repeats,
        "full_attention_kernel_ms": attention_ms,
        "total_ms": total_ms,
        "non_attention_ms": total_ms - attention_ms,
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
