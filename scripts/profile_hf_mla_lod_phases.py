#!/usr/bin/env python3
"""Attribute warm native-MLA LOD prefill time to engine phases."""

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
    parser.add_argument(
        "--checkpoint", default="deepseek-ai/DeepSeek-V2-Lite-Chat"
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("full", "lod"), default="lod")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--decode-warmup-steps", type=int, default=4)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--prefill-chunk-factor", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _timed_method(
    owner,
    name: str,
    phase: str,
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> None:
    original = getattr(owner, name)

    def timed(self, *args, **kwargs):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original(*args, **kwargs)
        end.record()
        events[phase].append((begin, end))
        return result

    setattr(owner, name, types.MethodType(timed, owner))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if (
        args.sequence_length < 512
        or args.batch_size < 1
        or args.warmups < 1
        or args.repetitions < 1
        or args.decode_steps < 0
        or args.decode_warmup_steps < 0
    ):
        raise ValueError("sequence length and batch size are too small")
    device = torch.device("cuda")
    model, _ = load_model(
        args.checkpoint, device, use_upstream_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=False
    )
    sequence = select_sequences(
        tokenizer,
        "Seerkfang/prolong-64k-512-new",
        args.sequence_length,
        1,
        0,
        1,
    )[0][1]
    input_ids = sequence.unsqueeze(0).expand(args.batch_size, -1).contiguous()
    input_ids = input_ids.to(device)
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

    for _ in range(args.warmups):
        warm = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
    attention_modules = [
        module
        for module in model.modules()
        if getattr(module, "_hf_lod_mla_adapter", None) is not None
    ]
    engines = [
        module._hf_lod_transient_engine for module in attention_modules
    ]
    expected_engines = 27 if args.mode == "lod" else 0
    if len(attention_modules) != expected_engines or len(engines) != expected_engines:
        raise RuntimeError("DeepSeek-V2-Lite MLA engine count is wrong")

    if args.prefill_chunk_factor is not None:
        if args.mode != "lod" or args.prefill_chunk_factor < 1:
            raise ValueError("prefill chunk factor requires LOD and must be positive")
        for engine in engines:
            exact_lookback = int(engine.prefill_local_len) - int(
                engine.prefill_chunk_len
            )
            engine.prefill_chunk_len = (
                args.prefill_chunk_factor * int(engine.chunk_len)
            )
            engine.prefill_local_len = engine.prefill_chunk_len + exact_lookback
            if hasattr(engine, "_lod_state"):
                delattr(engine, "_lod_state")
        # Compile and warm the experimental chunk geometry before timing it.
        warm = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm

    def reset_engines() -> None:
        for engine in engines:
            if hasattr(engine, "_lod_state"):
                delattr(engine, "_lod_state")

    elapsed_ms = []
    result = None
    for _ in range(args.repetitions):
        reset_engines()
        total_begin = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_begin.record()
        result = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        total_end.record()
        torch.cuda.synchronize(device)
        elapsed_ms.append(float(total_begin.elapsed_time(total_end)))

    events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = defaultdict(list)
    phase_methods = {
        "route": "_route_top_slots",
        "exact_leaf": "_paged_leaf_attention",
        "coarse": "_coarse_attention",
        "state_update": "_update_state",
        "page_append": "_append_page_cache",
        "local": "_prefill_local_attention",
    }
    profiled_total_ms = None
    if args.mode == "lod":
        for module, engine in zip(attention_modules, engines, strict=True):
            _timed_method(module, "forward", "attention_total", events)
            _timed_method(engine, "forward", "engine_total", events)
            for phase, method in phase_methods.items():
                _timed_method(engine, method, phase, events)
        reset_engines()
        total_begin = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_begin.record()
        result = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        total_end.record()
        torch.cuda.synchronize(device)
        profiled_total_ms = float(total_begin.elapsed_time(total_end))
    phase_ms = {
        phase: sum(float(begin.elapsed_time(end)) for begin, end in pairs)
        for phase, pairs in events.items()
    }

    def timed_decode() -> float:
        cache = new_hf_lod_cache(model) if args.mode == "lod" else None
        prefill = model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = prefill.past_key_values
        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.decode_warmup_steps):
            decoded = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = decoded.past_key_values
            next_token = decoded.logits[:, -1:].argmax(dim=-1)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(args.decode_steps):
            decoded = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = decoded.past_key_values
            next_token = decoded.logits[:, -1:].argmax(dim=-1)
        end.record()
        torch.cuda.synchronize(device)
        return float(begin.elapsed_time(end))

    decode_elapsed_ms = (
        [timed_decode() for _ in range(args.repetitions)]
        if args.decode_steps else []
    )
    median_decode_ms = (
        statistics.median(decode_elapsed_ms) if decode_elapsed_ms else None
    )
    payload = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "attention_layers": len(attention_modules),
        "repetitions": args.repetitions,
        "open_count": args.open_count,
        "decode_steps": args.decode_steps,
        "decode_warmup_steps": args.decode_warmup_steps,
        "decode_elapsed_ms": decode_elapsed_ms,
        "median_decode_step_ms": (
            median_decode_ms / args.decode_steps
            if median_decode_ms is not None else None
        ),
        "decode_tokens_per_second": (
            args.batch_size * args.decode_steps / (median_decode_ms / 1000.0)
            if median_decode_ms is not None else None
        ),
        "prefill_chunk_factor": args.prefill_chunk_factor,
        "elapsed_ms": elapsed_ms,
        "median_ms": statistics.median(elapsed_ms),
        "profiled_total_ms": profiled_total_ms,
        "phase_ms": phase_ms,
        "phase_fraction": {
            phase: milliseconds / profiled_total_ms
            for phase, milliseconds in phase_ms.items()
        } if profiled_total_ms is not None else {},
        "adapter_outside_engine_ms": (
            phase_ms["attention_total"] - phase_ms["engine_total"]
            if profiled_total_ms is not None else None
        ),
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
