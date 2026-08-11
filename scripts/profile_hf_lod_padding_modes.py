#!/usr/bin/env python3
"""Compare exact and chunk-aligned HF LOD left-padding execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer

from model.hf_pytorch_lod_attention import (
    HFLODSettings,
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.probe_qwen35_lod_niah import (
    enable_fla_fast_path,
    require_qwen35_acceleration,
)
from scripts.compare_qwen35_lod_loss import select_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--prolong", action="store_true")
    parser.add_argument("--recursive-pages", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def set_padding_mode(model, mode: str) -> None:
    for module in model.modules():
        settings = getattr(module, "_hf_lod_settings", None)
        if isinstance(settings, HFLODSettings):
            module._hf_lod_settings = replace(settings, left_padding_mode=mode)


@torch.inference_mode()
def run_once(
    model,
    token,
    attention_mask,
    *,
    decode_tokens: int,
    collect_quality: bool,
):
    cache = new_hf_lod_cache(model)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    output = model(
        token,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - begin
    if collect_quality:
        token_losses = F.cross_entropy(
            output.logits[:, :-1].float().flatten(0, 1),
            token[:, 1:].flatten(),
            reduction="none",
        ).view_as(token[:, 1:])
        valid_prediction = attention_mask[:, :-1].bool()
        prefill_ce = token_losses[valid_prediction].mean().item()
        prefill_last_logits = output.logits[:, -1].float().cpu()
    else:
        prefill_ce = None
        prefill_last_logits = None

    decode_mask = attention_mask
    next_token = output.logits[:, -1:].argmax(dim=-1)
    begin = time.perf_counter()
    for _ in range(decode_tokens):
        decode_mask = torch.cat(
            (decode_mask, torch.ones_like(next_token)), dim=1
        )
        output = model(
            next_token,
            attention_mask=decode_mask,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = output.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - begin
    return {
        "prefill_seconds": prefill_seconds,
        "decode_ms_per_token": decode_seconds * 1000.0 / decode_tokens,
        "prefill_ce": prefill_ce,
        "prefill_last_logits": prefill_last_logits,
        "final_logits": output.logits[:, -1].float().cpu(),
    }


def main() -> None:
    args = parse_args()
    if args.length < 512 or args.batch_size < 1 or args.decode_tokens < 1:
        raise ValueError("length, batch size, and decode count are too small")
    enable_fla_fast_path(required=True)
    composite = AutoConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    config = composite.get_text_config(decoder=True)
    config._attn_implementation = "sdpa"
    device = torch.device("cuda")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    acceleration = require_qwen35_acceleration(model)
    config_kwargs = {
        "chunk_size": 256,
        "local_window": 512,
        "state_growth_factor": 16.0,
        "state_min_size": 256,
        "protected_prefix": 1,
        "max_routes": 8,
    }
    lod_config = (
        PagedLODConfig(**config_kwargs, page_size=16, kv_bits=0)
        if args.recursive_pages
        else LODConfig(**config_kwargs)
    )
    install_hf_lod_attention(
        model,
        config=lod_config,
        open_count=8,
        engine_backend="kernel",
    )

    lengths = [
        args.length - (255 * row // max(args.batch_size - 1, 1))
        for row in range(args.batch_size)
    ]
    token = torch.zeros(
        args.batch_size, args.length, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(token)
    if args.prolong:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True
        )
        sequences = select_sequences(
            tokenizer,
            "Seerkfang/prolong-64k-512-new",
            args.length,
            args.batch_size,
            0,
            1,
        )
    else:
        generator = torch.Generator(device=device).manual_seed(91)
        sequences = [
            (
                row,
                torch.randint(
                    3,
                    model.config.vocab_size,
                    (args.length,),
                    generator=generator,
                    device=device,
                ),
            )
            for row in range(args.batch_size)
        ]
    for row, length in enumerate(lengths):
        sequence = sequences[row][1][:length].to(device)
        token[row, -length:] = sequence
        attention_mask[row, -length:] = 1

    measurements = {}
    final_logits = {}
    prefill_logits = {}
    for mode in ("exact", "chunk_aligned", "exact_repeat"):
        actual_mode = "exact" if mode == "exact_repeat" else mode
        set_padding_mode(model, actual_mode)
        run_once(
            model,
            token,
            attention_mask,
            decode_tokens=min(args.decode_tokens, 4),
            collect_quality=False,
        )
        result = run_once(
            model,
            token,
            attention_mask,
            decode_tokens=args.decode_tokens,
            collect_quality=True,
        )
        final_logits[mode] = result.pop("final_logits")
        prefill_logits[mode] = result.pop("prefill_last_logits")
        measurements[mode] = result

    aligned_delta = final_logits["chunk_aligned"] - final_logits["exact_repeat"]
    prefill_delta = (
        prefill_logits["chunk_aligned"] - prefill_logits["exact_repeat"]
    )
    payload = {
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "lengths": lengths,
        "decode_tokens": args.decode_tokens,
        "prolong": args.prolong,
        "recursive_pages": args.recursive_pages,
        "acceleration": acceleration,
        "measurements": measurements,
        "aligned_vs_exact": {
            "mean_absolute_logit_delta": aligned_delta.abs().mean().item(),
            "max_absolute_logit_delta": aligned_delta.abs().max().item(),
            "top1_agreement": (
                final_logits["chunk_aligned"].argmax(dim=-1)
                == final_logits["exact_repeat"].argmax(dim=-1)
            )
            .float()
            .mean()
            .item(),
            "prefill_last_mean_absolute_logit_delta": (
                prefill_delta.abs().mean().item()
            ),
            "prefill_last_max_absolute_logit_delta": (
                prefill_delta.abs().max().item()
            ),
            "prefill_last_top1_agreement": (
                prefill_logits["chunk_aligned"].argmax(dim=-1)
                == prefill_logits["exact_repeat"].argmax(dim=-1)
            )
            .float()
            .mean()
            .item(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
