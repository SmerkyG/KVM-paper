#!/usr/bin/env python3
"""Measure warm offline vLLM prefill and decode throughput for native or LOD."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _collect_cuda_storages(
    value: Any, storages: dict[tuple[str, int | None, int], int]
) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.device.type != "cuda":
            return
        storage = value.untyped_storage()
        key = (value.device.type, value.device.index, storage.data_ptr())
        storages[key] = max(storages.get(key, 0), int(storage.nbytes()))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_cuda_storages(item, storages)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_cuda_storages(item, storages)


def inspect_attention_memory(model) -> dict[str, int]:
    """Count unique persistent cache storages inside the worker process."""
    import torch

    native: dict[tuple[str, int | None, int], int] = {}
    lod: dict[tuple[str, int | None, int], int] = {}
    scratch: dict[tuple[str, int | None, int], int] = {}
    for module in model.modules():
        _collect_cuda_storages(getattr(module, "kv_cache", None), native)
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        _collect_cuda_storages(pool.state, lod)
        _collect_cuda_storages(pool.local_lens, lod)
        _collect_cuda_storages(pool.active_indices, lod)
        _collect_cuda_storages(pool.decode_buffers, scratch)
    for key in lod:
        scratch.pop(key, None)
    allocated = int(torch.cuda.memory_allocated())
    reserved_before = int(torch.cuda.memory_reserved())
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.empty_cache()
    reserved_after = int(torch.cuda.memory_reserved())
    free_after, _ = torch.cuda.mem_get_info()
    return {
        "native_cache_bytes": sum(native.values()),
        "lod_cache_bytes": sum(lod.values()),
        "lod_decode_scratch_bytes": sum(scratch.values()),
        "torch_allocated_bytes": allocated,
        "torch_reserved_bytes_before_reclaim": reserved_before,
        "torch_reserved_bytes_after_reclaim": reserved_after,
        "torch_reclaimable_cached_bytes": reserved_before - reserved_after,
        "device_used_bytes_before_reclaim": int(total - free_before),
        "device_used_bytes_after_reclaim": int(total - free_after),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--jit-monitor-verbose", action="store_true")
    parser.add_argument("--attention-backend")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_generate(llm, prompts, params) -> tuple[float, float, float]:
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - started
    expected = int(params.max_tokens)
    if any(len(output.outputs[0].token_ids) != expected for output in outputs):
        raise RuntimeError("a benchmark request stopped before max_tokens")
    metrics = [output.metrics for output in outputs]
    if any(metric is None for metric in metrics):
        raise RuntimeError("vLLM did not return per-request timing metrics")
    scheduled = min(float(metric.scheduled_ts) for metric in metrics)
    first_token = max(float(metric.first_token_ts) for metric in metrics)
    last_token = max(float(metric.last_token_ts) for metric in metrics)
    return elapsed, first_token - scheduled, last_token - first_token


def inspect_lod_model(model) -> dict[str, int]:
    diagnostics = {
        "layers": 0,
        "installs": 0,
        "batched_install_calls": 0,
        "direct_prefills": 0,
        "batched_cached_prefills": 0,
        "batched_cached_prefill_rows": 0,
        "cached_prefill_packed_calls": 0,
        "cached_prefill_nonpacked_calls": 0,
        "cached_prefill_candidate_calls": 0,
        "cached_prefill_candidate_rows": 0,
        "cached_prefill_nonuniform_lengths": 0,
        "cached_prefill_nonuniform_previous": 0,
        "cached_prefill_unready": 0,
        "cached_prefill_noncontiguous": 0,
        "decode_calls": 0,
        "catch_up_batches": 0,
        "catch_up_rows": 0,
    }
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        diagnostics["layers"] += 1
        diagnostics["installs"] += int(pool.install_count)
        diagnostics["batched_install_calls"] += int(pool.batched_install_calls)
        diagnostics["direct_prefills"] += int(pool.direct_prefill_calls)
        diagnostics["batched_cached_prefills"] += int(
            pool.batched_cached_prefill_calls
        )
        diagnostics["batched_cached_prefill_rows"] += int(
            pool.batched_cached_prefill_rows
        )
        diagnostics["cached_prefill_packed_calls"] += int(
            pool.cached_prefill_packed_calls
        )
        diagnostics["cached_prefill_nonpacked_calls"] += int(
            pool.cached_prefill_nonpacked_calls
        )
        diagnostics["cached_prefill_candidate_calls"] += int(
            pool.cached_prefill_candidate_calls
        )
        diagnostics["cached_prefill_candidate_rows"] += int(
            pool.cached_prefill_candidate_rows
        )
        diagnostics["cached_prefill_nonuniform_lengths"] += int(
            pool.cached_prefill_nonuniform_lengths
        )
        diagnostics["cached_prefill_nonuniform_previous"] += int(
            pool.cached_prefill_nonuniform_previous
        )
        diagnostics["cached_prefill_unready"] += int(pool.cached_prefill_unready)
        diagnostics["cached_prefill_noncontiguous"] += int(
            pool.cached_prefill_noncontiguous
        )
        diagnostics["decode_calls"] += int(pool.decode_calls)
        diagnostics["catch_up_batches"] += int(pool.catch_up_batches)
        diagnostics["catch_up_rows"] += int(pool.catch_up_rows)
    return diagnostics


def main() -> None:
    args = parse_args()
    if (
        args.length < 2
        or args.batch_size < 1
        or args.decode_tokens < 2
        or args.repeats < 1
    ):
        raise ValueError(
            "length >= 2, batch size >= 1, decode tokens >= 2, and repeats >= 1 required"
        )

    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    seed = tokenizer(
        "LOD attention retains precise high-mass regions and summarizes the rest. ",
        add_special_tokens=False,
    )["input_ids"]
    tokens = (seed * ((args.length + len(seed) - 1) // len(seed)))[: args.length]
    prompts = [{"prompt_token_ids": tokens} for _ in range(args.batch_size)]
    max_batched = args.max_num_batched_tokens or args.batch_size * args.length
    kwargs = {
        "model": args.checkpoint,
        "dtype": "bfloat16",
        "max_model_len": args.length + args.decode_tokens + 16,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": max_batched,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "jit_monitor_verbose": args.jit_monitor_verbose,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
    }
    if args.num_gpu_blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.attention_backend is not None:
        if args.mode == "lod":
            raise ValueError("--attention-backend is only valid with --mode full")
        kwargs["attention_config"] = {"backend": args.attention_backend}
    elif args.mode == "lod":
        kwargs["attention_config"] = {"backend": "CUSTOM"}
    llm = LLM(**kwargs)

    many = SamplingParams(
        temperature=0,
        max_tokens=args.decode_tokens,
        detokenize=False,
        ignore_eos=True,
    )
    # Warm the full prefill -> optional conversion -> decode path. Rebuild and
    # INT4 otherwise defer several Triton compilations until the measured run.
    timed_generate(llm, prompts, many)
    prefill_timings = []
    total_timings = []
    decode_timings = []
    for _ in range(args.repeats):
        elapsed, prefill_elapsed, decode_elapsed = timed_generate(
            llm, prompts, many
        )
        total_timings.append(elapsed)
        prefill_timings.append(prefill_elapsed)
        decode_timings.append(decode_elapsed)
    prefill_elapsed = statistics.median(prefill_timings)
    total_elapsed = statistics.median(total_timings)
    marginal_decode = statistics.median(decode_timings)
    decode_interval = args.decode_tokens - 1
    marginal_tokens = args.batch_size * decode_interval
    result = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "length": args.length,
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "decode_interval_tokens": decode_interval,
        "repeats": args.repeats,
        "max_num_batched_tokens": max_batched,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "enforce_eager": args.enforce_eager,
        "jit_monitor_verbose": args.jit_monitor_verbose,
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "attention_backend": args.attention_backend,
        "prefill_seconds": prefill_elapsed,
        "prefill_timings_seconds": prefill_timings,
        "decode_timings_seconds": decode_timings,
        "total_timings_seconds": total_timings,
        "prefill_prompt_tokens_per_second": (
            args.batch_size * args.length / prefill_elapsed
        ),
        "prefill_plus_decode_seconds": total_elapsed,
        "marginal_decode_ms_per_token": 1000.0 * marginal_decode / marginal_tokens,
        "marginal_decode_ms_per_batch_step": (
            1000.0 * marginal_decode / decode_interval
        ),
        "marginal_decode_tokens_per_second": (
            marginal_tokens / marginal_decode if marginal_decode else None
        ),
    }
    if args.mode == "lod":
        result["lod_diagnostics"] = llm.apply_model(inspect_lod_model)[0]
    result["attention_memory"] = llm.apply_model(inspect_attention_memory)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
