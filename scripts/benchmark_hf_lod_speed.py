#!/usr/bin/env python3
"""Measure matched Hugging Face full-attention and LOD prefill/decode speed."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.hf_pytorch_lod_attention import (
    install_hf_lod_attention,
    new_hf_lod_cache,
)
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=1025)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-decode-tokens", type=int)
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def elapsed(run):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    return time.perf_counter() - started, result


@torch.inference_mode()
def run_once(model, token: torch.Tensor, *, mode: str, decode_tokens: int):
    cache = new_hf_lod_cache(model) if mode == "lod" else None

    def prefill():
        kwargs = {
            "input_ids": token,
            "use_cache": True,
            "logits_to_keep": 1,
        }
        if cache is not None:
            kwargs["past_key_values"] = cache
        return model(**kwargs)

    prefill_seconds, output = elapsed(prefill)
    if mode == "full":
        cache = output.past_key_values
    next_token = output.logits[:, -1:].argmax(dim=-1)

    def decode():
        nonlocal output, next_token
        for _ in range(decode_tokens):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            next_token = output.logits[:, -1:].argmax(dim=-1)
        return output

    decode_seconds, output = elapsed(decode)
    expected_length = int(token.size(1)) + decode_tokens
    if cache.get_seq_length() != expected_length:
        raise AssertionError(
            f"cache length {cache.get_seq_length()} != {expected_length}"
        )
    if not bool(torch.isfinite(output.logits).all()):
        raise AssertionError("non-finite logits in speed benchmark")
    return {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_ms_per_step": decode_seconds * 1000.0 / decode_tokens,
        "final_tokens": next_token.squeeze(-1).tolist(),
    }


def main() -> None:
    args = parse_args()
    if min(args.length, args.batch_size, args.decode_tokens, args.repeats) < 1:
        raise ValueError("lengths, batch size, decode tokens, and repeats must be positive")
    warmup_decode_tokens = (
        args.decode_tokens
        if args.warmup_decode_tokens is None
        else args.warmup_decode_tokens
    )
    if warmup_decode_tokens < 1:
        raise ValueError("warmup decode tokens must be positive")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model, acceleration = load_model(args.checkpoint, device)
    installed = []
    if args.mode == "lod":
        installed = install_hf_lod_attention(
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
                state_clustering_policy="qk_norm_aware",
            ),
            open_count=8,
            engine_backend="kernel",
        )

    seed = tokenizer(
        "LOD attention retains precise high-mass regions and summarizes the rest. ",
        add_special_tokens=False,
    )["input_ids"]
    rows = []
    for row in range(args.batch_size):
        prefix = tokenizer(
            f"K2 Horizon benchmark request {row}: ", add_special_tokens=False
        )["input_ids"]
        row_seed = prefix + seed
        rows.append(
            (row_seed * ((args.length + len(row_seed) - 1) // len(row_seed)))[
                : args.length
            ]
        )
    token = torch.tensor(rows, dtype=torch.long, device=device)

    run_once(
        model,
        token,
        mode=args.mode,
        decode_tokens=warmup_decode_tokens,
    )
    torch.cuda.empty_cache()
    records = []
    torch.cuda.reset_peak_memory_stats(device)
    for repeat in range(args.repeats):
        record = run_once(
            model,
            token,
            mode=args.mode,
            decode_tokens=args.decode_tokens,
        )
        record["repeat"] = repeat
        records.append(record)
        print(json.dumps(record), flush=True)

    policy = []
    for module in model.modules():
        settings = getattr(module, "_hf_lod_settings", None)
        if settings is None:
            continue
        resolved = (
            settings.config.state_clustering_normalization,
            settings.config.state_clustering_centroid_rescale,
            settings.config.state_clustering_centroid_rescale_scope,
        )
        if resolved not in policy:
            policy.append(resolved)
    payload = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "length": args.length,
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "warmup_decode_tokens": warmup_decode_tokens,
        "repeats": args.repeats,
        "kv_bits": args.kv_bits if args.mode == "lod" else None,
        "attention_layers": installed,
        "resolved_state_policies": policy,
        "acceleration": acceleration,
        "median_prefill_seconds": statistics.median(
            record["prefill_seconds"] for record in records
        ),
        "median_decode_ms_per_step": statistics.median(
            record["decode_ms_per_step"] for record in records
        ),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
