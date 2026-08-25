#!/usr/bin/env python3
"""Measure a fresh vLLM process using disk or daemon-backed model loading."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--load-format", choices=("auto", "ipc_cache"), default="auto")
    parser.add_argument("--cache-id", default="default")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from vllm import LLM, SamplingParams

    loader_extra = None
    if args.load_format == "ipc_cache":
        loader_extra = {"cache_id": args.cache_id}
        if args.cache_dir:
            loader_extra["cache_dir"] = args.cache_dir

    started = time.perf_counter()
    llm_kwargs = {
        "model": args.checkpoint,
        "load_format": args.load_format,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "max_num_seqs": 1,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if loader_extra is not None:
        llm_kwargs["model_loader_extra_config"] = loader_extra
    llm = LLM(**llm_kwargs)
    initialized = time.perf_counter()
    outputs = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=8),
        use_tqdm=False,
    )
    finished = time.perf_counter()
    result = {
        "checkpoint": args.checkpoint,
        "load_format": args.load_format,
        "startup_seconds": initialized - started,
        "tensor_parallel_size": args.tensor_parallel_size,
        "first_request_seconds": finished - initialized,
        "text": outputs[0].outputs[0].text,
        "token_ids": list(outputs[0].outputs[0].token_ids),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
