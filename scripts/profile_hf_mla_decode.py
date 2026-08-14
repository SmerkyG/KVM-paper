#!/usr/bin/env python3
"""Measure batch decode after memory-bounded cached MLA prefill."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import types

import torch
from transformers import AutoTokenizer

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--decode-warmup-steps", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--open-count", type=int, default=16)
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(
        args.sequence_length,
        args.batch_size,
        args.prefill_chunk_size,
        args.decode_steps,
        args.repetitions,
    ) < 1:
        raise ValueError("benchmark sizes must be positive")
    device = torch.device("cuda")
    model, _ = load_model(args.checkpoint, device, use_upstream_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
    sequence = select_sequences(
        tokenizer,
        "Seerkfang/prolong-64k-512-new",
        args.sequence_length,
        1,
        0,
        1,
    )[0][1]
    input_ids = sequence.unsqueeze(0).expand(args.batch_size, -1).contiguous().to(device)

    if args.mode == "lod":
        install_hf_lod_attention(
            model,
            config=PagedLODConfig(
                chunk_size=256,
                local_window=512,
                state_growth_factor=16.0,
                state_min_size=256,
                protected_prefix=1,
                max_routes=args.open_count,
                page_size=16,
                kv_bits=0,
                mla_state_key_normalization="latent",
                mla_recursive_page_key_normalization=True,
                state_clustering_policy="qk_norm_aware",
                routing_normalization="qk_norm_aware",
            ),
            open_count=args.open_count,
            engine_backend="kernel",
        )

    phase_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = defaultdict(list)

    def run_once(*, profile: bool = False) -> tuple[float, float]:
        cache = new_hf_lod_cache(model) if args.mode == "lod" else None
        prefill_begin = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_begin.record()
        result = None
        for begin in range(0, args.sequence_length, args.prefill_chunk_size):
            result = model(
                input_ids=input_ids[:, begin : begin + args.prefill_chunk_size],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = result.past_key_values
        prefill_end.record()
        torch.cuda.synchronize(device)
        if result is None:
            raise AssertionError("cached prefill produced no output")
        if profile and args.mode == "lod":
            for layer in cache.layers:
                engine = layer.engine
                if engine is None:
                    raise RuntimeError("LOD cache did not create its decode engine")
                engine._lod_leaf_timing_events = phase_events
                for name in (
                    "_decode_attention",
                    "_two_level_attention",
                    "_mla_normalize_key",
                    "_route_top_slots",
                    "_coarse_attention",
                    "_update_state",
                ):
                    original = getattr(engine, name)

                    def timed(self, *call_args, __name=name, __original=original, **call_kwargs):
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        value = __original(*call_args, **call_kwargs)
                        end.record()
                        phase_events[__name].append((begin, end))
                        return value

                    setattr(engine, name, types.MethodType(timed, engine))
        next_token = result.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.decode_warmup_steps):
            result = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = result.past_key_values
            next_token = result.logits[:, -1:].argmax(dim=-1)
        if profile:
            phase_events.clear()
        decode_begin = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_begin.record()
        for _ in range(args.decode_steps):
            result = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = result.past_key_values
            next_token = result.logits[:, -1:].argmax(dim=-1)
        decode_end.record()
        torch.cuda.synchronize(device)
        return (
            float(prefill_begin.elapsed_time(prefill_end)),
            float(decode_begin.elapsed_time(decode_end)),
        )

    run_once()
    measurements = [
        run_once(profile=args.profile_phases)
        for _ in range(args.repetitions)
    ]
    prefill_ms = [item[0] for item in measurements]
    decode_ms = [item[1] for item in measurements]
    median_decode_ms = statistics.median(decode_ms)
    payload = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "decode_steps": args.decode_steps,
        "decode_warmup_steps": args.decode_warmup_steps,
        "repetitions": args.repetitions,
        "open_count": args.open_count,
        "phase_ms_per_step": {
            name: sum(float(begin.elapsed_time(end)) for begin, end in events)
            / (args.repetitions * args.decode_steps)
            for name, events in phase_events.items()
        },
        "prefill_elapsed_ms": prefill_ms,
        "decode_elapsed_ms": decode_ms,
        "median_decode_step_ms": median_decode_ms / args.decode_steps,
        "decode_tokens_per_second": (
            args.batch_size * args.decode_steps / (median_decode_ms / 1000.0)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
