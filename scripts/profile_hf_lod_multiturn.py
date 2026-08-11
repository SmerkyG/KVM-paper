#!/usr/bin/env python3
"""Profile a cached multi-token turn against token-at-a-time ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.probe_qwen35_lod_niah import (
    enable_fla_fast_path,
    require_qwen35_acceleration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--initial-length", type=int, default=8192)
    parser.add_argument("--turn-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=4)
    parser.add_argument("--mode", choices=("lod", "full"), default="lod")
    parser.add_argument("--skip-serial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def elapsed_ms(run) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    run()
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if min(args.initial_length, args.turn_length, args.batch_size) <= 0:
        raise ValueError("profile lengths and batch size must be positive")
    if args.turn_length <= 1:
        raise ValueError("turn length must exercise cached prefill")
    enable_fla_fast_path(required=True)
    composite = AutoConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    config = composite.get_text_config(decoder=True)
    config._attn_implementation = "sdpa"
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to("cuda")
        .eval()
    )
    acceleration = require_qwen35_acceleration(model)
    if args.mode == "lod":
        install_hf_lod_attention(
            model,
            config=PagedLODConfig(
                chunk_size=256,
                local_window=512,
                state_growth_factor=16.0,
                state_min_size=256,
                protected_prefix=1,
                max_routes=8,
                page_size=16,
                kv_bits=args.kv_bits,
            ),
            open_count=8,
            engine_backend="kernel",
        )
    generator = torch.Generator(device="cuda").manual_seed(203)
    tokens = torch.randint(
        3,
        model.config.vocab_size,
        (
            args.batch_size,
            args.initial_length + args.turn_length + 1,
        ),
        generator=generator,
        device="cuda",
    )
    initial = tokens[:, : args.initial_length]
    turn = tokens[
        :, args.initial_length : args.initial_length + args.turn_length
    ]
    next_token = tokens[:, -1:]

    def start_cache():
        if args.mode == "lod":
            cache = new_hf_lod_cache(model)
            output = model(
                initial,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        else:
            output = model(initial, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
        return cache, output

    # Warm initial prefill, cached-prefill, and post-turn decode shapes.
    warm_cache, warm_output = start_cache()
    warm_output = model(
        turn,
        past_key_values=warm_cache,
        use_cache=True,
        logits_to_keep=1,
    )
    warm_output = model(
        next_token,
        past_key_values=warm_cache,
        use_cache=True,
        logits_to_keep=1,
    )
    torch.cuda.synchronize()
    del warm_cache, warm_output
    torch.cuda.empty_cache()

    fast_cache, fast_output = start_cache()

    def fast_turn() -> None:
        nonlocal fast_output
        fast_output = model(
            turn,
            past_key_values=fast_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    fast_ms = elapsed_ms(fast_turn)
    fast_last_logits = fast_output.logits[:, -1].float()
    fast_decode_output = None

    def fast_decode() -> None:
        nonlocal fast_decode_output
        fast_decode_output = model(
            next_token,
            past_key_values=fast_cache,
            use_cache=True,
            logits_to_keep=1,
        )

    post_turn_decode_ms = elapsed_ms(fast_decode)
    if args.mode == "lod" and fast_cache.get_seq_length() != (
        args.initial_length + args.turn_length + 1
    ):
        raise AssertionError("fast turn cache length did not advance")

    serial_ms = None
    cosine = None
    last_token_match = None
    if args.mode == "lod" and not args.skip_serial:
        serial_cache, serial_output = start_cache()

        def serial_turn() -> None:
            nonlocal serial_output
            for token in range(args.turn_length):
                serial_output = model(
                    turn[:, token : token + 1],
                    past_key_values=serial_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )

        serial_ms = elapsed_ms(serial_turn)
        serial_last_logits = serial_output.logits[:, -1].float()
        if serial_cache.get_seq_length() != args.initial_length + args.turn_length:
            raise AssertionError("serial turn cache length did not advance")
        cosine = float(
            torch.nn.functional.cosine_similarity(
                fast_last_logits, serial_last_logits, dim=-1
            ).mean().item()
        )
        last_token_match = float(
            fast_last_logits.argmax(-1)
            .eq(serial_last_logits.argmax(-1))
            .float()
            .mean()
            .item()
        )
    result = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "initial_length": args.initial_length,
        "turn_length": args.turn_length,
        "kv_bits": args.kv_bits,
        "fast_turn_ms": fast_ms,
        "serial_turn_ms": serial_ms,
        "speedup": serial_ms / fast_ms if serial_ms is not None else None,
        "fast_ms_per_turn_token": fast_ms / args.turn_length,
        "serial_ms_per_turn_token": (
            serial_ms / args.turn_length if serial_ms is not None else None
        ),
        "post_turn_decode_ms": post_turn_decode_ms,
        "last_logit_cosine": cosine,
        "last_token_match": last_token_match,
        "finite": bool(
            torch.isfinite(fast_last_logits).all().item()
            and torch.isfinite(fast_decode_output.logits).all().item()
        ),
        "qwen35_acceleration": acceleration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
