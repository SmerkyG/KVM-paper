#!/usr/bin/env python3
"""Measure warm offline vLLM prefill and decode throughput for native or LOD."""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def muse_native_text_config(config):
    """Select the native vLLM adapter for Muse-Glimmer's text tower."""
    text_config = getattr(config, "text_config", config)
    text_config.architectures = ["MuseGlimmerForCausalLM"]
    return text_config


def allow_heterogeneous_global_config(config):
    """Expose Gemma 4's nested text model as a heterogeneous causal LM."""
    text_config = getattr(config, "text_config", config)
    # vLLM probes this callback once with a skeletal config while resolving
    # the architecture.
    if not hasattr(text_config, "layer_types"):
        return config
    text_config.allow_global_per_layer_attribute_access = True
    full_layers = [
        text_config.per_layer_config[index]
        for index, layer_type in enumerate(text_config.layer_types)
        if layer_type == "full_attention"
    ]
    if full_layers:
        text_config.global_head_dim = max(
            int(layer.head_dim) for layer in full_layers
        )
        text_config.num_global_key_value_heads = min(
            int(layer.num_key_value_heads) for layer in full_layers
        )
    if getattr(text_config, "top_k", None) is None:
        top_k_experts = getattr(text_config, "top_k_experts", None)
        if top_k_experts is not None:
            text_config.top_k = int(top_k_experts)
    text_config.architectures = ["Gemma4ForCausalLM"]
    return text_config


def _iter_lod_pools(model):
    """Yield pools even when the HF modeling backend hides attention modules."""
    seen: set[int] = set()
    runtime = getattr(model, "_vllm_lod_runtime", None)
    for pool in getattr(runtime, "pools", {}).values():
        seen.add(id(pool))
        yield pool
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is not None and id(pool) not in seen:
            seen.add(id(pool))
            yield pool


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
    for pool in _iter_lod_pools(model):
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


def inspect_native_cache_layouts(model) -> list[dict[str, Any]]:
    """Report physical cache layouts used by native ROCm attention layers."""
    import torch
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
        has_native_kv_cache_layout,
    )
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    layouts = []
    for module in model.modules():
        impl = getattr(module, "impl", None)
        caches = getattr(module, "kv_cache", None)
        if impl is None or caches is None:
            continue
        if isinstance(caches, (list, tuple)):
            if not caches:
                continue
            cache = caches[0]
        else:
            cache = caches
        if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
            continue
        try:
            split = getattr(impl, "_split_kv_cache", None)
            if split is not None:
                key_cache, value_cache = split(cache)
            else:
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    cache,
                    int(impl.num_kv_heads),
                    int(impl.head_size),
                )
        except (AssertionError, RuntimeError, ValueError):
            continue
        layouts.append(
            {
                "layer": str(getattr(module, "layer_name", "")),
                "impl": type(impl).__name__,
                "head_size": int(impl.head_size),
                "num_kv_heads": int(impl.num_kv_heads),
                "sliding_window": getattr(impl, "sliding_window", None),
                "cache_shape": list(cache.shape),
                "cache_stride": list(cache.stride()),
                "key_shape": list(key_cache.shape),
                "key_stride": list(key_cache.stride()),
                "value_shape": list(value_cache.shape),
                "value_stride": list(value_cache.stride()),
                "native_layout": bool(
                    has_native_kv_cache_layout(key_cache, value_cache)
                ),
            }
        )
    return layouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "lod"), required=True)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument(
        "--lengths",
        type=lambda value: [int(item) for item in value.split(",")],
        help="Comma-separated per-request prompt lengths for a ragged batch",
    )
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
    parser.add_argument("--muse-native-text-config", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument(
        "--allow-heterogeneous-global-config", action="store_true"
    )
    parser.add_argument("--lod-leaf-num-warps", type=int)
    parser.add_argument("--lod-recursive-page-block-n", type=int)
    parser.add_argument("--lod-prefill-chunk-len", type=int)
    parser.add_argument("--lod-prefill-state-update-len", type=int)
    parser.add_argument("--lod-direct-prefill-route", action="store_true")
    parser.add_argument("--lod-decode-route-group-size", type=int)
    parser.add_argument("--lod-decode-state-update-len", type=int)
    parser.add_argument("--lod-decode-route-num-warps", type=int)
    parser.add_argument("--lod-decode-route-reduce-num-warps", type=int)
    parser.add_argument("--lod-decode-final-reduce-num-warps", type=int)
    parser.add_argument("--lod-decode-block-n", type=int)
    parser.add_argument("--lod-decode-num-warps", type=int)
    parser.add_argument("--lod-gqa-union-aiter", action="store_true")
    parser.add_argument("--lod-gqa-union-group-size", type=int)
    parser.add_argument("--lod-gqa-max-slot-leaves", type=int)
    parser.add_argument(
        "--lod-gqa-route-then-coarse", action="store_true"
    )
    parser.add_argument("--lod-gqa-persistent-route", action="store_true")
    parser.add_argument("--lod-gqa-fused-correction", action="store_true")
    parser.add_argument("--lod-gqa-stage1-reduce", action="store_true")
    parser.add_argument(
        "--lod-decode-use-dot", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--profile-lod-phases", action="store_true")
    parser.add_argument("--inspect-cache-layouts", action="store_true")
    parser.add_argument("--torch-profile-dir", type=Path)
    parser.add_argument("--torch-profile-delay-iterations", type=int, default=0)
    parser.add_argument("--torch-profile-max-iterations", type=int, default=0)
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
    import torch

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
        "gqa_union_sequences": 0,
        "gqa_union_regions": 0,
        "gqa_union_leaves": 0,
        "gqa_union_max_leaves": 0,
        "gqa_union_stage1_reduce": False,
        "gqa_union_fused_correction": False,
        "gqa_union_own_route_correction": False,
        "gqa_union_group_size": 0,
    }
    for pool in _iter_lod_pools(model):
        diagnostics["gqa_union_stage1_reduce"] = bool(
            pool.engine.gqa_union_stage1_reduce
        )
        diagnostics["gqa_union_fused_correction"] = bool(
            pool.engine.gqa_union_fused_correction
        )
        diagnostics["gqa_union_own_route_correction"] = bool(
            pool.engine.gqa_union_own_route_correction
        )
        diagnostics["gqa_union_group_size"] = int(
            pool.engine.gqa_union_group_size
        )
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
        for buffers in pool.gqa_union_buffers.values():
            lengths = buffers.get("lengths")
            top_slots = buffers.get("top_slots")
            if not isinstance(lengths, torch.Tensor) or not isinstance(
                top_slots, torch.Tensor
            ):
                continue
            diagnostics["gqa_union_sequences"] += int(lengths.numel())
            diagnostics["gqa_union_leaves"] += int(lengths.sum().item())
            diagnostics["gqa_union_max_leaves"] = max(
                diagnostics["gqa_union_max_leaves"],
                int(lengths.max().item()),
            )
            group_size = int(pool.engine.gqa_union_group_size)
            diagnostics["gqa_union_regions"] += int(
                top_slots[:, ::group_size, 0, :].ge(0).sum().item()
            )
    return diagnostics


def install_lod_phase_timers(model) -> int:
    """Record attention-phase GPU events inside each vLLM worker."""
    import torch

    methods = {
        "two_level": "_two_level_attention",
        "route": "_route_top_slots",
        "exact_leaf": "_paged_leaf_attention",
        "coarse": "_coarse_attention",
        "state_update": "_update_state",
        "page_append": "_append_page_cache",
        "local": "_prefill_local_attention",
    }
    installed = 0
    native_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    model._lod_native_phase_timing_events = native_events
    for module in model.modules():
        impl = getattr(module, "impl", None)
        native = getattr(impl, "_triton_swa", None)
        if native is None or getattr(native, "_lod_phase_timed", False):
            continue
        original = native.forward

        def timed_native(*args, __original=original, **kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = __original(*args, **kwargs)
            end.record()
            native_events.append((begin, end))
            return result

        native.forward = timed_native
        native._lod_phase_timed = True
    for pool in _iter_lod_pools(model):
        engine = pool.engine
        if hasattr(engine, "_lod_phase_timing_events"):
            continue
        events = {name: [] for name in methods}
        engine._lod_phase_timing_events = events
        # The fused recursive decode path has its own phase boundaries inside
        # one engine call.  Share the same event sink so its route, local,
        # exact-leaf, and final-reduction costs appear in the profile.
        engine._lod_decode_timing_events = events
        if getattr(engine, "recursive_page_lod", False):
            # Recursive prefill calls the residual-page function directly
            # instead of passing through ``_paged_leaf_attention``.
            engine._lod_leaf_timing_events = {"total": events["exact_leaf"]}
        for phase, method_name in methods.items():
            original = getattr(engine, method_name)

            def timed(
                *args,
                __original=original,
                __phase=phase,
                __events=events,
                **kwargs,
            ):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = __original(*args, **kwargs)
                end.record()
                __events[__phase].append((begin, end))
                return result

            setattr(engine, method_name, timed)
        installed += 1
    return installed


def summarize_lod_phase_timers(model) -> dict[str, dict[str, float | int]]:
    """Synchronize and aggregate phase events from one vLLM worker."""
    import torch

    torch.cuda.synchronize()
    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    native_pairs = getattr(model, "_lod_native_phase_timing_events", ())
    if native_pairs:
        totals["native_fallthrough"] = sum(
            float(begin.elapsed_time(end)) for begin, end in native_pairs
        )
        calls["native_fallthrough"] = len(native_pairs)
    for pool in _iter_lod_pools(model):
        events = getattr(pool.engine, "_lod_phase_timing_events", {})
        for phase, pairs in events.items():
            totals[phase] = totals.get(phase, 0.0) + sum(
                float(begin.elapsed_time(end)) for begin, end in pairs
            )
            calls[phase] = calls.get(phase, 0) + len(pairs)
    return {
        phase: {"milliseconds": totals[phase], "calls": calls[phase]}
        for phase in sorted(totals)
    }


def configure_lod_model(
    model,
    *,
    leaf_num_warps: int | None,
    recursive_page_block_n: int | None,
    prefill_chunk_len: int | None,
    prefill_state_update_len: int | None,
    direct_prefill_route: bool,
    decode_route_group_size: int | None,
    decode_route_num_warps: int | None,
    decode_route_reduce_num_warps: int | None,
    decode_final_reduce_num_warps: int | None,
    decode_block_n: int | None,
    decode_num_warps: int | None,
    decode_use_dot: bool | None,
    gqa_route_then_coarse: bool,
    gqa_persistent_route: bool,
    gqa_fused_correction: bool,
    gqa_union_group_size: int | None,
    gqa_max_slot_leaves: int | None,
) -> int:
    """Apply benchmark-only kernel tuning before the warmup request."""
    configured = 0
    for pool in _iter_lod_pools(model):
        if leaf_num_warps is not None:
            pool.engine.leaf_num_warps = int(leaf_num_warps)
        if recursive_page_block_n is not None:
            pool.engine.recursive_page_block_n = int(recursive_page_block_n)
        if prefill_chunk_len is not None:
            pool.engine.prefill_chunk_len = int(prefill_chunk_len)
            pool.engine.prefill_local_len = (
                int(prefill_chunk_len)
                + int(pool.engine.local_len)
                + int(pool.engine.chunk_len)
            )
        if prefill_state_update_len is not None:
            pool.engine.prefill_state_update_len = int(
                prefill_state_update_len
            )
        if direct_prefill_route:
            pool.engine.reuse_route_logits_for_coarse = False
        if decode_route_group_size is not None:
            pool.engine.decode_route_group_size = int(decode_route_group_size)
        if decode_route_num_warps is not None:
            pool.engine.decode_route_num_warps = int(decode_route_num_warps)
        if decode_route_reduce_num_warps is not None:
            pool.engine.decode_route_reduce_num_warps = int(
                decode_route_reduce_num_warps
            )
        if decode_final_reduce_num_warps is not None:
            pool.engine.decode_final_reduce_num_warps = int(
                decode_final_reduce_num_warps
            )
        if decode_block_n is not None:
            pool.engine.decode_block_n = int(decode_block_n)
        if decode_num_warps is not None:
            pool.engine.decode_num_warps = int(decode_num_warps)
        if decode_use_dot is not None:
            pool.engine.decode_use_dot = bool(decode_use_dot)
        if gqa_route_then_coarse:
            pool.engine.gqa_union_route_then_coarse = True
        if gqa_persistent_route:
            pool.engine.gqa_union_persistent_route = True
        if gqa_fused_correction:
            pool.engine.gqa_union_fused_correction = True
        if gqa_union_group_size is not None:
            pool.engine.gqa_union_group_size = int(gqa_union_group_size)
        if gqa_max_slot_leaves is not None:
            pool.engine.gqa_union_max_slot_leaves = int(gqa_max_slot_leaves)
        configured += 1
    return configured


def main() -> None:
    args = parse_args()
    # This benchmark intentionally sends its local diagnostic callbacks to
    # the colocated worker process after each run.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if args.lod_gqa_union_aiter:
        if args.mode != "lod":
            raise ValueError("--lod-gqa-union-aiter requires --mode lod")
        os.environ["VLLM_LOD_GQA_UNION_AITER"] = "1"
        os.environ["VLLM_LOD_KV_BITS"] = "0"
    if args.lod_gqa_persistent_route:
        if not args.lod_gqa_union_aiter:
            raise ValueError(
                "--lod-gqa-persistent-route requires --lod-gqa-union-aiter"
            )
        os.environ["VLLM_LOD_GQA_UNION_PERSISTENT_ROUTE"] = "1"
    if args.lod_gqa_route_then_coarse:
        if not args.lod_gqa_union_aiter:
            raise ValueError(
                "--lod-gqa-route-then-coarse requires --lod-gqa-union-aiter"
            )
        os.environ["VLLM_LOD_GQA_UNION_ROUTE_THEN_COARSE"] = "1"
    if args.lod_gqa_fused_correction:
        if not args.lod_gqa_union_aiter:
            raise ValueError(
                "--lod-gqa-fused-correction requires --lod-gqa-union-aiter"
            )
        os.environ["VLLM_LOD_GQA_UNION_FUSED_CORRECTION"] = "1"
    if args.lod_gqa_stage1_reduce:
        if not args.lod_gqa_union_aiter:
            raise ValueError(
                "--lod-gqa-stage1-reduce requires --lod-gqa-union-aiter"
            )
        os.environ["VLLM_LOD_GQA_UNION_STAGE1_REDUCE"] = "1"
    if args.lod_gqa_union_group_size is not None:
        if args.lod_gqa_union_group_size <= 0:
            raise ValueError("--lod-gqa-union-group-size must be positive")
        os.environ["VLLM_LOD_GQA_UNION_GROUP_SIZE"] = str(
            args.lod_gqa_union_group_size
        )
    if args.lod_gqa_max_slot_leaves is not None:
        if args.lod_gqa_max_slot_leaves < 0:
            raise ValueError("--lod-gqa-max-slot-leaves must be nonnegative")
        os.environ["VLLM_LOD_GQA_MAX_SLOT_LEAVES"] = str(
            args.lod_gqa_max_slot_leaves
        )
    if args.lod_decode_state_update_len is not None:
        if args.mode != "lod" or args.lod_decode_state_update_len <= 0:
            raise ValueError(
                "--lod-decode-state-update-len requires LOD mode and a positive value"
            )
        os.environ["VLLM_LOD_DECODE_STATE_UPDATE_LEN"] = str(
            args.lod_decode_state_update_len
        )
    prompt_lengths = args.lengths or [args.length] * args.batch_size
    if (
        args.batch_size < 1
        or min(prompt_lengths) < 2
        or args.decode_tokens < 2
        or args.repeats < 1
    ):
        raise ValueError(
            "length >= 2, batch size >= 1, decode tokens >= 2, and repeats >= 1 required"
        )
    if len(prompt_lengths) != args.batch_size:
        raise ValueError("--lengths must contain exactly --batch-size entries")
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    seed = tokenizer(
        "LOD attention retains precise high-mass regions and summarizes the rest. ",
        add_special_tokens=False,
    )["input_ids"]
    prompts = [
        {
            "prompt_token_ids": (
                seed * ((length + len(seed) - 1) // len(seed))
            )[:length]
        }
        for length in prompt_lengths
    ]
    prompt_tokens = sum(prompt_lengths)
    max_batched = args.max_num_batched_tokens or prompt_tokens
    kwargs = {
        "model": args.checkpoint,
        "dtype": "bfloat16",
        "max_model_len": max(prompt_lengths) + args.decode_tokens + 16,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": max_batched,
        "long_prefill_token_threshold": args.long_prefill_token_threshold,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "jit_monitor_verbose": args.jit_monitor_verbose,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "tensor_parallel_size": args.tensor_parallel_size,
        "language_model_only": args.language_model_only,
    }
    if args.muse_native_text_config:
        kwargs["hf_overrides"] = muse_native_text_config
    elif args.allow_heterogeneous_global_config:
        kwargs["hf_overrides"] = allow_heterogeneous_global_config
    if args.num_gpu_blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.attention_backend is not None:
        if args.mode == "lod":
            raise ValueError("--attention-backend is only valid with --mode full")
        kwargs["attention_config"] = {"backend": args.attention_backend}
    elif args.mode == "lod":
        kwargs["attention_config"] = {"backend": "CUSTOM"}
    if args.torch_profile_dir is not None:
        profile_dir = args.torch_profile_dir.resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "delay_iterations": args.torch_profile_delay_iterations,
            "max_iterations": args.torch_profile_max_iterations,
        }
    llm = LLM(**kwargs)
    if args.mode == "lod" and any(
        value is not None
        for value in (
            args.lod_leaf_num_warps,
            args.lod_recursive_page_block_n,
            args.lod_prefill_chunk_len,
            args.lod_prefill_state_update_len,
            args.lod_direct_prefill_route or None,
            args.lod_decode_route_group_size,
            args.lod_decode_route_num_warps,
            args.lod_decode_route_reduce_num_warps,
            args.lod_decode_final_reduce_num_warps,
            args.lod_decode_block_n,
            args.lod_decode_num_warps,
            args.lod_decode_use_dot,
            # GQA-union options above are constructor settings supplied via
            # environment variables. Avoid an unnecessary apply_model call,
            # which is especially important for tensor-parallel workers.
        )
    ):
        configured = llm.apply_model(
            functools.partial(
                configure_lod_model,
                leaf_num_warps=args.lod_leaf_num_warps,
                recursive_page_block_n=args.lod_recursive_page_block_n,
                prefill_chunk_len=args.lod_prefill_chunk_len,
                prefill_state_update_len=args.lod_prefill_state_update_len,
                direct_prefill_route=args.lod_direct_prefill_route,
                decode_route_group_size=args.lod_decode_route_group_size,
                decode_route_num_warps=args.lod_decode_route_num_warps,
                decode_route_reduce_num_warps=(
                    args.lod_decode_route_reduce_num_warps
                ),
                decode_final_reduce_num_warps=(
                    args.lod_decode_final_reduce_num_warps
                ),
                decode_block_n=args.lod_decode_block_n,
                decode_num_warps=args.lod_decode_num_warps,
                decode_use_dot=args.lod_decode_use_dot,
                gqa_route_then_coarse=args.lod_gqa_route_then_coarse,
                gqa_persistent_route=args.lod_gqa_persistent_route,
                gqa_fused_correction=args.lod_gqa_fused_correction,
                gqa_union_group_size=args.lod_gqa_union_group_size,
                gqa_max_slot_leaves=None,
            )
        )
        if not configured or not all(value > 0 for value in configured):
            raise RuntimeError("LOD benchmark tuning found no installed layers")

    many = SamplingParams(
        temperature=0,
        max_tokens=args.decode_tokens,
        detokenize=False,
        ignore_eos=True,
    )
    # Warm the full prefill -> optional conversion -> decode path. Rebuild and
    # INT4 otherwise defer several Triton compilations until the measured run.
    timed_generate(llm, prompts, many)
    if args.profile_lod_phases:
        if args.mode != "lod":
            raise ValueError("--profile-lod-phases requires --mode lod")
        installed = llm.apply_model(install_lod_phase_timers)
        if not installed or not all(value > 0 for value in installed):
            raise RuntimeError("LOD phase profiler found no installed layers")
    if args.torch_profile_dir is not None:
        llm.start_profile("decode_benchmark")
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
    if args.torch_profile_dir is not None:
        llm.stop_profile()
    prefill_elapsed = statistics.median(prefill_timings)
    total_elapsed = statistics.median(total_timings)
    marginal_decode = statistics.median(decode_timings)
    decode_interval = args.decode_tokens - 1
    marginal_tokens = args.batch_size * decode_interval
    result = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "length": args.length if args.lengths is None else None,
        "lengths": prompt_lengths,
        "prompt_tokens": prompt_tokens,
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
        "muse_native_text_config": args.muse_native_text_config,
        "tensor_parallel_size": args.tensor_parallel_size,
        "language_model_only": args.language_model_only,
        "allow_heterogeneous_global_config": (
            args.allow_heterogeneous_global_config
        ),
        "lod_leaf_num_warps": args.lod_leaf_num_warps,
        "lod_recursive_page_block_n": args.lod_recursive_page_block_n,
        "lod_prefill_chunk_len": args.lod_prefill_chunk_len,
        "lod_prefill_state_update_len": args.lod_prefill_state_update_len,
        "lod_direct_prefill_route": args.lod_direct_prefill_route,
        "lod_decode_route_group_size": args.lod_decode_route_group_size,
        "lod_decode_state_update_len": args.lod_decode_state_update_len,
        "lod_decode_route_num_warps": args.lod_decode_route_num_warps,
        "lod_decode_route_reduce_num_warps": (
            args.lod_decode_route_reduce_num_warps
        ),
        "lod_decode_final_reduce_num_warps": (
            args.lod_decode_final_reduce_num_warps
        ),
        "lod_decode_block_n": args.lod_decode_block_n,
        "lod_decode_num_warps": args.lod_decode_num_warps,
        "lod_gqa_union_aiter": args.lod_gqa_union_aiter,
        "lod_gqa_union_group_size": args.lod_gqa_union_group_size,
        "lod_gqa_stage1_reduce": args.lod_gqa_stage1_reduce,
        "lod_gqa_max_slot_leaves": args.lod_gqa_max_slot_leaves,
        "lod_gqa_route_then_coarse": args.lod_gqa_route_then_coarse,
        "lod_gqa_persistent_route": args.lod_gqa_persistent_route,
        "lod_decode_use_dot": args.lod_decode_use_dot,
        "torch_profile_dir": (
            str(args.torch_profile_dir.resolve())
            if args.torch_profile_dir is not None
            else None
        ),
        "prefill_seconds": prefill_elapsed,
        "prefill_timings_seconds": prefill_timings,
        "decode_timings_seconds": decode_timings,
        "total_timings_seconds": total_timings,
        "prefill_prompt_tokens_per_second": (
            prompt_tokens / prefill_elapsed
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
    # vLLM's TP worker RPC pickles callbacks. Functions from an executed
    # __main__ module are not reliably importable in worker subprocesses, so
    # keep these optional diagnostics out of multi-rank timing runs.
    collect_worker_diagnostics = args.tensor_parallel_size == 1
    if args.mode == "lod" and collect_worker_diagnostics:
        result["lod_diagnostics"] = llm.apply_model(inspect_lod_model)[0]
    if args.profile_lod_phases and collect_worker_diagnostics:
        result["lod_phase_profile"] = llm.apply_model(
            summarize_lod_phase_timers
        )[0]
    if collect_worker_diagnostics:
        result["attention_memory"] = llm.apply_model(inspect_attention_memory)[0]
    if args.inspect_cache_layouts and collect_worker_diagnostics:
        result["native_cache_layouts"] = llm.apply_model(
            inspect_native_cache_layouts
        )[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
