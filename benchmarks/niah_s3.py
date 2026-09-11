"""Run RULER NIAH-S3 against full attention or a release LoD mode."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from ._vllm import (
    MODES,
    close_llm,
    default_gpu_memory_utilization,
    llm_kwargs,
    write_json,
)
from .prolong import comma_separated_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        type=comma_separated_ints,
        default=[8_192, 16_384, 32_768, 65_536],
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--engine-max-model-len",
        type=int,
        default=None,
        help="Allocate a larger engine context than the evaluated lengths.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help=(
            "vLLM native-cache memory fraction "
            "(default: 0.70 for Qwen LoD, 0.80 for K2 LoD, or 0.90 for full)"
        ),
    )
    parser.add_argument(
        "--full-attention-backend",
        default="ROCM_AITER_UNIFIED_ATTN",
    )
    return parser.parse_args()


def _lm_eval_package() -> tuple[Path, str]:
    try:
        distribution = importlib.metadata.distribution("lm-eval")
    except importlib.metadata.PackageNotFoundError as exc:
        configured_root = os.environ.get("LM_EVAL_PACKAGE_ROOT")
        if configured_root is None:
            raise RuntimeError(
                "NIAH-S3 requires the benchmark extra: "
                "uv sync --extra vllm --extra benchmarks"
            ) from exc
        package_root = Path(configured_root).resolve()
        version = os.environ.get("LM_EVAL_VERSION", "unknown")
    else:
        package_root = Path(distribution.locate_file("lm_eval")).resolve()
        version = distribution.version
    return package_root, version


def _load_ruler_generator() -> tuple[Any, Any, str]:
    """Import only RULER's prompt generator, not the full task registry."""

    package_root, version = _lm_eval_package()
    ruler_root = package_root / "tasks" / "ruler"
    if not (ruler_root / "prepare_niah.py").is_file():
        raise RuntimeError("installed lm-eval does not contain the RULER tasks")
    dependency_root = str(package_root.parent)
    added_dependency_root = dependency_root not in sys.path
    if added_dependency_root:
        # Append (rather than prepend), then remove before vLLM imports. This
        # exposes RULER's optional dependencies and package metadata without
        # allowing an unrelated environment's ray/torch to shadow serving.
        sys.path.append(dependency_root)
    try:
        for name, path in (
            ("lm_eval", package_root),
            ("lm_eval.tasks", package_root / "tasks"),
            ("lm_eval.tasks.ruler", ruler_root),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack
    finally:
        if added_dependency_root:
            sys.path.remove(dependency_root)

    return generate_samples, get_haystack, version


def make_samples(
    tokenizer: Any,
    *,
    length: int,
    samples: int,
    sample_offset: int,
) -> list[dict[str, Any]]:
    generate_samples, get_haystack, _version = _load_ruler_generator()
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE

    random.seed(0)
    np.random.seed(1234)
    documents = generate_samples(
        get_haystack(type_haystack="essay"),
        max_seq_length=length,
        template=TEMPLATE,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="uuids",
        num_samples=sample_offset + samples,
        TOKENIZER=tokenizer,
    )
    selected = []
    for document in documents[sample_offset:]:
        encoded = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": document["input"]},
                {
                    "role": "assistant",
                    "content": document["gen_prefix"],
                    "think": "",
                },
            ],
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=False,
        )
        token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        selected.append(
            {
                "index": int(document["index"]),
                "prompt_token_ids": token_ids,
                "target": str(document["outputs"][0]),
            }
        )
    return selected


def evaluate_length(
    llm: Any,
    tokenizer: Any,
    *,
    length: int,
    samples: int,
    sample_offset: int,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    from vllm import SamplingParams

    documents = make_samples(
        tokenizer,
        length=length,
        samples=samples,
        sample_offset=sample_offset,
    )
    prompts = [
        {"prompt_token_ids": document["prompt_token_ids"]} for document in documents
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        detokenize=True,
    )
    started = time.perf_counter()
    outputs = []
    for begin in range(0, len(prompts), batch_size):
        outputs.extend(
            llm.generate(prompts[begin : begin + batch_size], params, use_tqdm=True)
        )
    elapsed = time.perf_counter() - started
    records = []
    for document, output in zip(documents, outputs, strict=True):
        response = output.outputs[0].text
        target = document["target"]
        records.append(
            {
                "index": document["index"],
                "input_tokens": len(document["prompt_token_ids"]),
                "target": target,
                "response": response,
                "exact": target.lower() in response.lower(),
            }
        )
    correct = sum(record["exact"] for record in records)
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "elapsed_seconds": elapsed,
        "input_tokens": sum(record["input_tokens"] for record in records),
        "samples": records,
    }


def main() -> None:
    args = parse_args()
    if min(args.samples, args.batch_size, args.max_new_tokens) < 1:
        raise ValueError("samples, batch-size, and max-new-tokens must be positive")
    if args.sample_offset < 0:
        raise ValueError("sample-offset must be nonnegative")
    minimum_model_len = max(args.lengths) + args.max_new_tokens + 16
    if (
        args.engine_max_model_len is not None
        and args.engine_max_model_len < minimum_model_len
    ):
        raise ValueError("engine-max-model-len is shorter than the requested test")

    # Validate the optional RULER dependencies before paying model-startup cost.
    _, _, lm_eval_version = _load_ruler_generator()

    from transformers import AutoTokenizer

    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None:
        gpu_memory_utilization = default_gpu_memory_utilization(
            args.checkpoint,
            args.mode,
        )
    kwargs = llm_kwargs(
        checkpoint=args.checkpoint,
        mode=args.mode,
        max_model_len=args.engine_max_model_len or minimum_model_len,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        full_attention_backend=args.full_attention_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
    )
    from vllm import LLM

    llm = LLM(**kwargs)
    result = {
        "benchmark": "ruler_niah_s3",
        "generator": "lm_eval.tasks.ruler.niah_single_3",
        "lm_eval_version": lm_eval_version,
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "lengths": args.lengths,
        "requested_samples": args.samples,
        "sample_offset": args.sample_offset,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "engine_max_model_len": args.engine_max_model_len or minimum_model_len,
        "scheduler_chunk_tokens": 16_384,
        "results": {},
    }
    try:
        for length in args.lengths:
            result["results"][str(length)] = evaluate_length(
                llm,
                tokenizer,
                length=length,
                samples=args.samples,
                sample_offset=args.sample_offset,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            write_json(args.output, result)
            row = result["results"][str(length)]
            print(f"{length}: {row['correct']}/{row['total']}", flush=True)
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
