#!/usr/bin/env python3
"""Evaluate native or LOD-backed vLLM on ProLong and NIAH-S3."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--eval", choices=("prolong", "niah_s3"), required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--lod-prefill-route-block-m", type=int)
    parser.add_argument("--lod-prefill-route-num-warps", type=int)
    parser.add_argument("--lod-recursive-page-block-n", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_prolong_tokens(tokenizer, length: int, samples: int) -> list[list[int]]:
    dataset = load_dataset(
        "Seerkfang/prolong-64k-512-new", split="train", streaming=True
    ).shuffle(seed=42, buffer_size=1_000)
    selected: list[list[int]] = []
    for document in dataset:
        token_count = document.get("length")
        if token_count is not None and int(token_count) < length:
            continue
        token_ids = tokenizer(
            document["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=length,
            return_attention_mask=False,
        )["input_ids"]
        if len(token_ids) == length:
            selected.append(token_ids)
        if len(selected) == samples:
            break
    if len(selected) != samples:
        raise RuntimeError(f"found only {len(selected)} sufficiently long documents")
    return selected


def select_niah_s3(tokenizer, checkpoint: str, length: int, samples: int) -> list[dict]:
    random.seed(0)
    np.random.seed(1234)
    try:
        from lm_eval.tasks.ruler.niah_utils import niah_single_3
    except ModuleNotFoundError as error:
        package_root_value = os.environ.get("LMEVAL_PACKAGE_ROOT")
        if not package_root_value:
            raise RuntimeError(
                "NIAH-S3 requires lm_eval, or LMEVAL_PACKAGE_ROOT must point "
                "to its lm_eval package directory"
            ) from error
        package_root = Path(package_root_value).resolve()
        if not (package_root / "tasks" / "ruler" / "niah_utils.py").is_file():
            raise RuntimeError(
                f"LMEVAL_PACKAGE_ROOT is not an lm_eval package: {package_root}"
            ) from error
        dependency_root = str(package_root.parent)
        if dependency_root not in sys.path:
            # Append rather than prepend so serving-environment builds of
            # torch, transformers, and vLLM keep precedence. This path only
            # supplies optional RULER helpers such as wonderwords and nltk.
            sys.path.append(dependency_root)
        # The NIAH generator itself has light dependencies, but importing it
        # normally executes lm_eval.tasks.__init__, which pulls the full eval
        # stack into the serving environment. Namespace packages load only the
        # three RULER modules needed to create prompts.
        for name in tuple(sys.modules):
            if name == "lm_eval" or name.startswith("lm_eval."):
                del sys.modules[name]
        for name, path in (
            ("lm_eval", package_root),
            ("lm_eval.tasks", package_root / "tasks"),
            ("lm_eval.tasks.ruler", package_root / "tasks" / "ruler"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        from lm_eval.tasks.ruler.niah_utils import niah_single_3

    standard_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    lengths = [value for value in standard_lengths if value <= length]
    if length not in lengths:
        lengths.append(length)
    dataset = niah_single_3(max_seq_lengths=lengths, pretrained=checkpoint)["test"]
    documents = [doc for doc in dataset if int(doc["max_length"]) == length]
    if len(documents) < samples:
        raise RuntimeError(f"NIAH-S3 provides only {len(documents)} documents")
    selected = []
    for document in documents[:samples]:
        prompt = document["input"] + " " + document["gen_prefix"]
        token_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        selected.append(
            {
                "index": int(document["index"]),
                "prompt_token_ids": token_ids,
                "target": str(document["outputs"][0]),
            }
        )
    return selected


def make_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs = {
        "model": args.checkpoint,
        "dtype": "bfloat16",
        "max_model_len": args.length + args.max_new_tokens,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": (
            args.max_num_batched_tokens
            or max(args.length, args.batch_size * 2048)
        ),
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
    }
    if args.mode == "lod":
        kwargs["attention_config"] = {"backend": "CUSTOM"}
    return LLM(**kwargs)


def inspect_lod_model(model) -> dict[str, object]:
    """Run in the vLLM worker and prove that LOD pools are attached."""
    custom_layers = []
    pooled_layers = []
    install_count = 0
    direct_prefill_calls = 0
    native_append_calls = 0
    decode_calls = 0
    catch_up_batches = 0
    catch_up_rows = 0
    batched_cached_prefills = 0
    batched_cached_prefill_rows = 0
    routing_geometries = set()
    for name, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if type(impl).__module__ != "vllm_lod_plugin.backend":
            continue
        custom_layers.append(name)
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None:
            pooled_layers.append(name)
            install_count += int(pool.install_count)
            direct_prefill_calls += int(pool.direct_prefill_calls)
            native_append_calls += int(pool.native_append_calls)
            decode_calls += int(pool.decode_calls)
            catch_up_batches += int(pool.catch_up_batches)
            catch_up_rows += int(pool.catch_up_rows)
            batched_cached_prefills += int(pool.batched_cached_prefill_calls)
            batched_cached_prefill_rows += int(pool.batched_cached_prefill_rows)
            engine = pool.engine
            routing_geometries.add(
                (
                    str(engine.state_clustering_normalization),
                    str(engine.state_clustering_centroid_rescale),
                    str(engine.routing_normalization),
                )
            )
    return {
        "custom_layers": custom_layers,
        "pooled_layers": pooled_layers,
        "install_count": install_count,
        "direct_prefill_calls": direct_prefill_calls,
        "native_append_calls": native_append_calls,
        "decode_calls": decode_calls,
        "catch_up_batches": catch_up_batches,
        "catch_up_rows": catch_up_rows,
        "batched_cached_prefills": batched_cached_prefills,
        "batched_cached_prefill_rows": batched_cached_prefill_rows,
        "routing_geometries": sorted(routing_geometries),
    }


def configure_lod_model(
    model,
    *,
    route_block_m: int | None,
    route_num_warps: int | None,
    page_block_n: int | None,
) -> int:
    """Apply evaluation-only kernel ablations to installed LOD engines."""
    configured = 0
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        if route_block_m is not None:
            pool.engine.prefill_route_block_m = route_block_m
        if route_num_warps is not None:
            pool.engine.prefill_route_num_warps = route_num_warps
        if page_block_n is not None:
            pool.engine.recursive_page_block_n = page_block_n
        configured += 1
    return configured


def evaluate_prolong(args: argparse.Namespace, tokenizer, llm) -> dict:
    from vllm import SamplingParams

    sequences = select_prolong_tokens(tokenizer, args.length, args.samples)
    prompts = [{"prompt_token_ids": token_ids} for token_ids in sequences]
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        flat_logprobs=True,
        detokenize=False,
    )
    started = time.perf_counter()
    outputs = []
    for begin in range(0, len(prompts), args.batch_size):
        outputs.extend(
            llm.generate(
                prompts[begin : begin + args.batch_size],
                params,
                use_tqdm=True,
            )
        )
    elapsed = time.perf_counter() - started
    sample_records = []
    total_nll = 0.0
    total_tokens = 0
    for sample, (token_ids, output) in enumerate(zip(sequences, outputs)):
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
            raise RuntimeError("vLLM returned incomplete prompt log probabilities")
        nll = 0.0
        for token_id, candidates in zip(token_ids[1:], prompt_logprobs[1:]):
            if candidates is None or token_id not in candidates:
                raise RuntimeError("target token missing from vLLM prompt logprobs")
            nll -= float(candidates[token_id].logprob)
        prediction_tokens = len(token_ids) - 1
        total_nll += nll
        total_tokens += prediction_tokens
        sample_records.append(
            {
                "sample": sample,
                "tokens": len(token_ids),
                "loss": nll / prediction_tokens,
                "perplexity": math.exp(nll / prediction_tokens),
            }
        )
    loss = total_nll / total_tokens
    return {
        "eval": "prolong",
        "loss": loss,
        "perplexity": math.exp(loss),
        "prediction_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "samples": sample_records,
        "attention_exercised": (
            "direct_lod_prefill"
            if args.mode == "lod"
            and os.environ.get("VLLM_LOD_PREFILL_MODE", "direct") == "direct"
            else "native_prefill"
        ),
    }


def evaluate_niah_s3(args: argparse.Namespace, tokenizer, llm) -> dict:
    from vllm import SamplingParams

    documents = select_niah_s3(tokenizer, args.checkpoint, args.length, args.samples)
    prompts = [
        {"prompt_token_ids": document["prompt_token_ids"]} for document in documents
    ]
    params = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        detokenize=True,
    )
    started = time.perf_counter()
    # Keep each submission within the authoritative LOD pool. Passing all 64
    # prompts to vLLM at once lets the scheduler admit replacement requests as
    # earlier rows finish, which can evict semantic cache rows still needed by
    # in-flight decode and spuriously requests an impossible native fallback.
    outputs = []
    for begin in range(0, len(prompts), args.batch_size):
        outputs.extend(
            llm.generate(
                prompts[begin : begin + args.batch_size],
                params,
                use_tqdm=True,
            )
        )
    elapsed = time.perf_counter() - started
    records = []
    for document, output in zip(documents, outputs):
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
        "eval": "niah_s3",
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "elapsed_seconds": elapsed,
        "samples": records,
        "attention_exercised": (
            (
                "direct_lod_prefill_and_decode"
                if os.environ.get("VLLM_LOD_PREFILL_MODE", "direct") == "direct"
                else "native_prefill_then_lod_decode"
            )
            if args.mode == "lod"
            else "native_prefill_and_decode"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch size must be positive")
    if args.mode == "lod" and args.batch_size > int(
        os.environ.get("VLLM_LOD_POOL_SIZE", "8")
    ):
        raise ValueError("batch size exceeds VLLM_LOD_POOL_SIZE")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    llm = make_llm(args)
    tuning_requested = any(
        value is not None
        for value in (
            args.lod_prefill_route_block_m,
            args.lod_prefill_route_num_warps,
            args.lod_recursive_page_block_n,
        )
    )
    if args.mode == "lod" and tuning_requested:
        configured = llm.apply_model(
            functools.partial(
                configure_lod_model,
                route_block_m=args.lod_prefill_route_block_m,
                route_num_warps=args.lod_prefill_route_num_warps,
                page_block_n=args.lod_recursive_page_block_n,
            )
        )
        if not configured or not all(value > 0 for value in configured):
            raise RuntimeError("LOD evaluation tuning found no installed layers")
    lod_diagnostics_before = None
    if args.mode == "lod":
        diagnostics = llm.apply_model(inspect_lod_model)
        lod_diagnostics_before = diagnostics[0]
        if not lod_diagnostics_before["custom_layers"]:
            raise RuntimeError("vLLM did not construct any CUSTOM attention layers")
        if (
            lod_diagnostics_before["pooled_layers"]
            != lod_diagnostics_before["custom_layers"]
        ):
            raise RuntimeError(
                "LOD pools are not attached to every CUSTOM attention layer: "
                f"{lod_diagnostics_before}"
            )
    if args.eval == "prolong":
        result = evaluate_prolong(args, tokenizer, llm)
    else:
        result = evaluate_niah_s3(args, tokenizer, llm)
    lod_diagnostics_after = None
    if args.mode == "lod":
        lod_diagnostics_after = llm.apply_model(inspect_lod_model)[0]
        if (
            os.environ.get("VLLM_LOD_PREFILL_MODE", "direct") == "direct"
            and lod_diagnostics_after["direct_prefill_calls"]
            <= lod_diagnostics_before["direct_prefill_calls"]
        ):
            raise RuntimeError("evaluation completed without invoking direct LOD prefill")
        if args.eval == "niah_s3":
            if not lod_diagnostics_after["decode_calls"]:
                raise RuntimeError("NIAH-S3 completed without invoking LOD decode")
            if (
                lod_diagnostics_after["install_count"]
                <= lod_diagnostics_before["install_count"]
            ):
                raise RuntimeError(
                    "NIAH-S3 completed without installing an LOD cache"
                )
    result.update(
        checkpoint=args.checkpoint,
        mode=args.mode,
        length=args.length,
        requested_samples=args.samples,
        batch_size=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enforce_eager=args.enforce_eager,
        lod_prefill_route_block_m=args.lod_prefill_route_block_m,
        lod_prefill_route_num_warps=args.lod_prefill_route_num_warps,
        lod_recursive_page_block_n=args.lod_recursive_page_block_n,
        lod_diagnostics_before=lod_diagnostics_before,
        lod_diagnostics_after=lod_diagnostics_after,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))


if __name__ == "__main__":
    main()
