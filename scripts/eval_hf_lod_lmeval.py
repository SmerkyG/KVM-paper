#!/usr/bin/env python3
"""Run lm-eval with the model-independent Hugging Face LOD backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from model.hf_pytorch_lod_attention import install_hf_lod_attention
from model.pytorch_lod_attention import LODConfig
from model.pytorch_lod_attention_paged import PagedLODConfig
from scripts.eval_qwen35_lod_lmeval import patch_hotpotqa_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ruler-length", type=int)
    parser.add_argument("--open-count", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--local-window", type=int, default=512)
    parser.add_argument("--state-min-size", type=int, default=256)
    parser.add_argument("--protected-prefix", type=int, default=1)
    parser.add_argument(
        "--engine-backend", choices=("torch", "kernel"), default="kernel"
    )
    parser.add_argument("--recursive-pages", action="store_true")
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument(
        "--left-padding-mode",
        choices=("exact", "chunk_aligned"),
        default="chunk_aligned",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(checkpoint: str, device: torch.device):
    composite_config = AutoConfig.from_pretrained(
        checkpoint, trust_remote_code=True
    )
    config = composite_config.get_text_config(decoder=True)
    is_qwen35 = type(config).__module__.startswith(
        "transformers.models.qwen3_5."
    )
    if is_qwen35:
        from scripts.probe_qwen35_lod_niah import enable_fla_fast_path

        enable_fla_fast_path(required=True)
    config._attn_implementation = "sdpa"
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    if is_qwen35:
        from scripts.probe_qwen35_lod_niah import require_qwen35_acceleration

        acceleration = require_qwen35_acceleration(model)
    else:
        acceleration = None
    return model, acceleration


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.ruler_length is not None and args.ruler_length < 1:
        raise ValueError("RULER length must be positive")
    if not 0 <= args.open_count <= 8:
        raise ValueError("open count must be in [0, 8]")
    if args.kv_bits and not args.recursive_pages:
        raise ValueError("KV quantization requires --recursive-pages")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model, acceleration = load_model(args.checkpoint, device)
    config_kwargs = {
        "chunk_size": args.chunk_size,
        "local_window": args.local_window,
        "state_growth_factor": args.state_growth_factor,
        "state_min_size": args.state_min_size,
        "protected_prefix": args.protected_prefix,
        "max_routes": 8,
    }
    config = (
        PagedLODConfig(**config_kwargs, page_size=16, kv_bits=args.kv_bits)
        if args.recursive_pages
        else LODConfig(**config_kwargs)
    )
    installed = install_hf_lod_attention(
        model,
        config=config,
        open_count=args.open_count,
        engine_backend=args.engine_backend,
        left_padding_mode=args.left_padding_mode,
    )
    if local_rank == 0:
        print(f"installed HF LOD on {len(installed)} attention layers")
        if acceleration is not None:
            print(
                "Qwen3.5 acceleration: "
                + json.dumps(acceleration, sort_keys=True)
            )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device=str(device),
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import Accelerator

        accelerator = Accelerator()
        lm.accelerator = accelerator
        lm._rank = accelerator.process_index
        lm._world_size = accelerator.num_processes
    patch_hotpotqa_download()

    metadata = {
        "pretrained": args.checkpoint,
        "tokenizer": args.checkpoint,
    }
    if args.ruler_length is not None:
        metadata["max_seq_lengths"] = [args.ruler_length]
    evaluation_start = time.perf_counter()
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        bootstrap_iters=0,
        log_samples=False,
        metadata=metadata,
        confirm_run_unsafe_code=True,
    )
    evaluation_seconds = time.perf_counter() - evaluation_start
    if results is None:
        return

    payload = dict(results)
    payload["lod_evaluation"] = {
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "ruler_length": args.ruler_length,
        "open_count": args.open_count,
        "state_growth_factor": args.state_growth_factor,
        "chunk_size": args.chunk_size,
        "local_window": args.local_window,
        "state_min_size": args.state_min_size,
        "protected_prefix": args.protected_prefix,
        "engine_backend": args.engine_backend,
        "recursive_pages": args.recursive_pages,
        "kv_bits": args.kv_bits,
        "left_padding_mode": args.left_padding_mode,
        "attention_layers": installed,
        "evaluation_seconds": evaluation_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(make_table(results))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
