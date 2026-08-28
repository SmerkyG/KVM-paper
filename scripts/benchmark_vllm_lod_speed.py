#!/usr/bin/env python3
"""Measure warm offline vLLM prefill and decode throughput for native or LOD."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from vllm_engine_lifecycle import register_llm_shutdown, shutdown_registered_llms


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
    external_lod_native: dict[tuple[str, int | None, int], int] = {}
    lod: dict[tuple[str, int | None, int], int] = {}
    scratch: dict[tuple[str, int | None, int], int] = {}
    external_lod_layers = 0
    external_lod_layers_with_native_cache = 0
    for module in model.modules():
        _collect_cuda_storages(getattr(module, "kv_cache", None), native)
        if bool(getattr(module, "_vllm_lod_external_kv_cache", False)):
            external_lod_layers += 1
            layer_native: dict[tuple[str, int | None, int], int] = {}
            _collect_cuda_storages(getattr(module, "kv_cache", None), layer_native)
            external_lod_layers_with_native_cache += int(bool(layer_native))
            external_lod_native.update(layer_native)
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
        "external_lod_native_cache_bytes": sum(external_lod_native.values()),
        "external_lod_layers": external_lod_layers,
        "external_lod_layers_with_native_cache": (
            external_lod_layers_with_native_cache
        ),
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
    parser.add_argument(
        "--prompt-source",
        choices=("synthetic", "prolong"),
        default="synthetic",
        help=(
            "Prompt construction. 'prolong' uses distinct shuffled real "
            "documents without repeating a document to fill a request."
        ),
    )
    parser.add_argument(
        "--lengths",
        type=lambda value: [int(item) for item in value.split(",")],
        help="Comma-separated per-request prompt lengths for a ragged batch",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--allow-output-mismatch",
        action="store_true",
        help=(
            "Continue a speed-only run if repeated greedy outputs differ; "
            "the result records the mismatch count."
        ),
    )
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-async-scheduling", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--jit-monitor-verbose", action="store_true")
    parser.add_argument("--attention-backend")
    parser.add_argument("--lod-leaf-num-warps", type=int)
    parser.add_argument("--lod-recursive-page-block-n", type=int)
    parser.add_argument(
        "--lod-recursive-state-route-backend",
        choices=("fused", "resplit"),
    )
    parser.add_argument("--lod-prefill-chunk-len", type=int)
    parser.add_argument("--lod-prefill-state-update-len", type=int)
    parser.add_argument("--lod-direct-prefill-route", action="store_true")
    parser.add_argument("--lod-decode-route-group-size", type=int)
    parser.add_argument("--lod-decode-route-num-warps", type=int)
    parser.add_argument("--lod-decode-route-reduce-num-warps", type=int)
    parser.add_argument("--lod-decode-final-reduce-num-warps", type=int)
    parser.add_argument("--lod-decode-block-n", type=int)
    parser.add_argument("--lod-decode-num-warps", type=int)
    parser.add_argument(
        "--lod-decode-use-dot", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--profile-lod-phases", action="store_true")
    parser.add_argument("--torch-profile-dir", type=Path)
    parser.add_argument("--torch-profile-delay-iterations", type=int, default=0)
    parser.add_argument("--torch-profile-max-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed_generate(
    llm, prompts, params, *, return_token_ids: bool = False
) -> tuple[float, float, float] | tuple[
    float, float, float, tuple[tuple[int, ...], ...]
]:
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
    timing = (elapsed, first_token - scheduled, last_token - first_token)
    if not return_token_ids:
        return timing
    token_ids = tuple(
        tuple(map(int, output.outputs[0].token_ids)) for output in outputs
    )
    return (*timing, token_ids)


def inspect_lod_model(model) -> dict[str, object]:
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
        "retained_prefix_reuses": 0,
        "retained_restore_attempts": 0,
        "retained_restore_fail_no_row": 0,
        "retained_restore_fail_short": 0,
        "retained_restore_fail_tokens": 0,
        "retained_restore_fail_coverage": 0,
        "retained_restore_rebuilds": 0,
        "retained_restore_rebuild_tokens": 0,
        "retained_restore_last_prefix": 0,
        "retained_restore_last_coverage": 0,
        "retained_restore_last_total": 0,
        "static_prefill_layers_executed": 0,
        "static_prefill_calls": 0,
        "static_prefill_cap_min_observed": None,
        "static_prefill_cap_max_observed": None,
        "static_prefill_exact_tokens_min": None,
        "static_prefill_exact_tokens_max": None,
        "static_prefill_permanent_exact_tokens": 0,
        "static_prefill_archive_tokens": 0,
        "static_prefill_permanent_exact_fraction": None,
    }
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        if diagnostics["layers"] == 0:
            # Record the resolved worker-side configuration, rather than only
            # benchmark CLI overrides.  In particular this makes an uncapped
            # run and the automatic short/long INT8 PV choice auditable from
            # the artifact itself.
            diagnostics.update(
                levels=int(pool.settings.levels),
                kv_bits=int(pool.settings.kv_bits),
                page_summary_quant_bits=(
                    int(page_summary_bits)
                    if (
                        page_summary_bits := getattr(
                            pool.engine, "page_summary_quant_bits", None
                        )
                    )
                    is not None
                    else None
                ),
                dense_leaf_storage=bool(pool.settings.dense_leaf_storage),
                external_kv_cache=True,
                leaf_seal_capacity=pool.settings.leaf_seal_capacity,
                prefill_chunk_len=(
                    int(pool.engine.prefill_chunk_len)
                    if pool.engine.prefill_chunk_len is not None
                    else None
                ),
                prefill_state_update_len=(
                    int(pool.engine.prefill_state_update_len)
                    if pool.engine.prefill_state_update_len is not None
                    else None
                ),
                prefill_int8_leaf_mma=bool(pool.engine.prefill_int8_leaf_mma),
                prefill_int8_pv_mma=bool(pool.engine.prefill_int8_pv_mma),
                prefill_int8_coarse_mma=bool(pool.engine.prefill_int8_coarse_mma),
                prefill_static_leaf_aiter=bool(
                    pool.engine.prefill_static_leaf_aiter
                ),
                prefill_static_leaf_cap_min=int(
                    pool.engine.prefill_static_leaf_cap_min
                ),
                static_leaf_cap_divisor=int(
                    pool.engine.static_leaf_cap_divisor
                ),
                static_cohort_never_readmit=bool(
                    pool.settings.static_cohort_never_readmit
                ),
                leaf_num_warps=int(pool.engine.leaf_num_warps),
                decode_gqa_cooperative=bool(pool._use_cooperative_decode()),
                open_count=int(pool.settings.open_count),
                decode_route_cohort=bool(pool.settings.decode_route_cohort),
                effective_decode_route_leaf_limit=(
                    pool._decode_route_leaf_limit()
                ),
                recursive_materialize_page_scores=bool(
                    pool.engine.recursive_materialize_page_scores
                ),
                recursive_page_score_block_n=int(
                    pool.engine.recursive_page_score_block_n
                ),
                recursive_page_score_num_warps=int(
                    pool.engine.recursive_page_score_num_warps
                ),
                recursive_page_select_block_n=int(
                    pool.engine.recursive_page_select_block_n
                ),
                recursive_state_route_backend=(
                    pool.engine.recursive_state_route_backend
                ),
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
        diagnostics["retained_prefix_reuses"] += int(pool.retained_reuse_count)
        diagnostics["retained_restore_attempts"] += int(
            pool.retained_restore_attempts
        )
        diagnostics["retained_restore_fail_no_row"] += int(
            pool.retained_restore_fail_no_row
        )
        diagnostics["retained_restore_fail_short"] += int(
            pool.retained_restore_fail_short
        )
        diagnostics["retained_restore_fail_tokens"] += int(
            pool.retained_restore_fail_tokens
        )
        diagnostics["retained_restore_fail_coverage"] += int(
            pool.retained_restore_fail_coverage
        )
        diagnostics["retained_restore_rebuilds"] += int(
            pool.retained_restore_rebuilds
        )
        diagnostics["retained_restore_rebuild_tokens"] += int(
            pool.retained_restore_rebuild_tokens
        )
        diagnostics["retained_restore_last_prefix"] = max(
            int(diagnostics["retained_restore_last_prefix"]),
            int(pool.retained_restore_last_prefix),
        )
        diagnostics["retained_restore_last_coverage"] = max(
            int(diagnostics["retained_restore_last_coverage"]),
            int(pool.retained_restore_last_coverage),
        )
        diagnostics["retained_restore_last_total"] = max(
            int(diagnostics["retained_restore_last_total"]),
            int(pool.retained_restore_last_total),
        )
        eviction_snapshots = getattr(
            pool, "_static_cohort_eviction_snapshots", None
        )
        if isinstance(eviction_snapshots, dict):
            for lengths, status, _, _ in eviction_snapshots.values():
                if not isinstance(lengths, torch.Tensor) or not isinstance(
                    status, torch.Tensor
                ):
                    continue
                diagnostics["static_prefill_archive_tokens"] += int(
                    lengths.sum().item()
                )
                diagnostics["static_prefill_permanent_exact_tokens"] += int(
                    lengths[status.eq(1)].sum().item()
                )
        engine = pool.engine
        if bool(getattr(engine, "_lod_prefill_static_leaf_cap_executed", False)):
            diagnostics["static_prefill_layers_executed"] = (
                int(diagnostics["static_prefill_layers_executed"]) + 1
            )
        cap_history = getattr(engine, "_lod_prefill_static_leaf_cap_history", ())
        diagnostics["static_prefill_calls"] = (
            int(diagnostics["static_prefill_calls"]) + len(cap_history)
        )
        if cap_history:
            cap_low = min(map(int, cap_history))
            cap_high = max(map(int, cap_history))
            previous_low = diagnostics["static_prefill_cap_min_observed"]
            previous_high = diagnostics["static_prefill_cap_max_observed"]
            diagnostics["static_prefill_cap_min_observed"] = (
                cap_low if previous_low is None else min(int(previous_low), cap_low)
            )
            diagnostics["static_prefill_cap_max_observed"] = (
                cap_high
                if previous_high is None
                else max(int(previous_high), cap_high)
            )
        exact_counts = getattr(engine, "_lod_prefill_static_exact_token_counts", None)
        if isinstance(exact_counts, torch.Tensor) and exact_counts.numel():
            exact_low = int(exact_counts.min().item())
            exact_high = int(exact_counts.max().item())
            previous_low = diagnostics["static_prefill_exact_tokens_min"]
            previous_high = diagnostics["static_prefill_exact_tokens_max"]
            diagnostics["static_prefill_exact_tokens_min"] = (
                exact_low
                if previous_low is None
                else min(int(previous_low), exact_low)
            )
            diagnostics["static_prefill_exact_tokens_max"] = (
                exact_high
                if previous_high is None
                else max(int(previous_high), exact_high)
            )
    archived = int(diagnostics["static_prefill_archive_tokens"])
    if archived:
        diagnostics["static_prefill_permanent_exact_fraction"] = (
            int(diagnostics["static_prefill_permanent_exact_tokens"]) / archived
        )
    return diagnostics


def inspect_lod_dispatch(model) -> dict[str, object]:
    """Describe the kernels and launch geometry selected by the vLLM pool."""
    import torch

    records: dict[str, dict[str, object]] = {}
    seen: set[int] = set()
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None or id(pool) in seen:
            continue
        seen.add(id(pool))
        engine = pool.engine
        page = pool.state["page_cache"]
        recursive = int(pool.settings.levels) == 3
        static_prefill = bool(
            getattr(pool.settings, "prefill_static_leaf_aiter", False)
        )
        kv_group_size = int(pool.query_heads // pool.kv_heads)
        gqa_union = bool(
            getattr(pool.settings, "decode_gqa_union", False)
            and not recursive
            and 1 < kv_group_size <= 16
            and int(pool.head_dim) in (128, 256, 512)
            and pool.dtype == torch.bfloat16
        )
        gqa_union_hip = bool(
            gqa_union and getattr(pool.settings, "decode_gqa_union_hip", False)
        )
        gqa_union_aiter_final = bool(
            gqa_union_hip
            and isinstance(page.get("unified_page1_k"), torch.Tensor)
        )
        gqa_union_staged_fixed = bool(
            gqa_union_aiter_final
            and getattr(
                pool.settings, "decode_gqa_staged_fixed_aiter", False
            )
        )
        gqa_union_fixed_mask = bool(
            gqa_union_aiter_final
            and getattr(
                pool.settings, "decode_gqa_fixed_mask_aiter", False
            )
        )
        gqa_union_static_cap = bool(
            gqa_union_aiter_final
            and getattr(
                pool.settings, "decode_gqa_static_leaf_aiter", False
            )
            and isinstance(
                page.get("unified_page1_fixed_indices"), torch.Tensor
            )
        )
        gqa_union_predicted_mass = bool(
            gqa_union_aiter_final
            and getattr(pool.settings, "decode_gqa_predicted_mass", False)
        )
        context_len = max(
            (int(row.get("total_len", 0)) for row in pool.metadata),
            default=0,
        )
        # The ordinary engine decode method has this context-dependent
        # override, but the vLLM pool bypasses that method and calls
        # fused_decode_paged_lod_attention directly with the configured engine
        # attributes. Keep the bypass explicit in the dispatch manifest.
        engine_long_d128_override_exists = bool(
            engine.decode_geometry_tuning
            and int(pool.head_dim) == 128
        )
        effective_route_group_size = int(engine.decode_route_group_size)
        effective_route_segment_tiles = int(engine.decode_route_segment_tiles)
        effective_route_num_warps = int(engine.decode_route_num_warps)
        effective_route_reduce_num_warps = int(
            engine.decode_route_reduce_num_warps
        )
        effective_route_use_dot = bool(engine.decode_route_use_dot)
        centroid_major_hip = bool(
            getattr(pool.settings, "decode_centroid_major_hip", False)
            and gqa_union_aiter_final
            and effective_route_group_size == 32
            and effective_route_segment_tiles == 1
            and int(pool.head_dim) == 256
            and kv_group_size in (4, 6)
            and pool.dtype == torch.bfloat16
        )
        if gqa_union_static_cap:
            state_route = [
                "none (persistent exact-small/coarse-large index list)"
            ]
            state_route_math = "none"
        elif recursive and engine.recursive_state_route_backend == "resplit":
            state_route = [
                "_materialize_state_summary_scores_gqa_kernel",
                "_materialized_state_tile_top8_lse_kernel",
                "_reduce_materialized_state_top8_lse_kernel",
                "_materialized_state_normalized_pv_split_kernel",
                "_reduce_materialized_state_pv_kernel",
            ]
            state_route_math = "materialized_gqa_mfma"
        elif gqa_union_predicted_mass:
            state_route = (
                [
                    (
                        "kernel_page1_predicted_mass_fixed_prepare "
                        "(AITER-shaped M16/N64 QK + direct GQA-union "
                        "compaction + fixed-mask reset/preparation)"
                    )
                ]
                if gqa_union_fixed_mask
                else [
                    "init_page1_predicted_mass_union",
                    (
                        "kernel_page1_predicted_mass_union "
                        "(AITER-shaped M16/N64 QK + direct GQA-union "
                        "compaction)"
                    ),
                ]
            )
            state_route_math = "page1_mfma_m16_retained_mass"
        elif centroid_major_hip:
            state_route = [
                (
                    "centroid_major_route_score_fixed_prepare "
                    "(HIP vector QK, one K load/all GQA scores, LDS query "
                    "reuse, block top-8, fixed-mask maintenance)"
                    if gqa_union_fixed_mask
                    else (
                        "centroid_major_route_score "
                        "(HIP vector QK, one K load/all GQA scores, "
                        "LDS query reuse, block top-8)"
                    )
                ),
                "_reduce_decode_route_topk_kernel",
            ]
            state_route_math = "centroid_major_vector"
        elif bool(engine.decode_route_gqa_grouped):
            scalar_gqa = (
                not effective_route_use_dot
                and effective_route_group_size <= 16
            )
            if effective_route_segment_tiles > 1:
                state_route = [
                    (
                        "_decode_route_coarse_gqa_segments_kernel "
                        "(SCORE_ONLY=True)"
                        if gqa_union
                        else "_decode_route_coarse_gqa_segments_kernel"
                    ),
                    (
                        "_reduce_decode_route_topk_kernel"
                        if gqa_union
                        else "_reduce_decode_route_coarse_vector_topk_kernel"
                    ),
                ]
            else:
                state_route = [
                    (
                        (
                            "_decode_route_coarse_scalar_gqa_groups_kernel "
                            "(SCORE_ONLY=True)"
                            if scalar_gqa
                            else "_decode_route_coarse_gqa_groups_kernel "
                            "(SCORE_ONLY=True)"
                        )
                        if gqa_union_aiter_final
                        else (
                            "_decode_route_coarse_scalar_gqa_groups_kernel"
                            if scalar_gqa
                            else "_decode_route_coarse_gqa_groups_kernel"
                        )
                    ),
                    (
                        "_reduce_decode_route_topk_kernel"
                        if gqa_union_aiter_final
                        else (
                            "_reduce_decode_route_coarse_vector_topk_kernel"
                            if engine.decode_route_parallel_reduce
                            else "_reduce_decode_route_coarse_kernel"
                        )
                    ),
                ]
            # The grouped kernel keeps USE_DOT in its API, but its current
            # implementation always executes a 16-row tl.dot. Only the
            # separately selected scalar kernel avoids that MFMA tile.
            state_route_math = "scalar" if scalar_gqa else "mfma_m16"
        else:
            state_route = [
                "_decode_route_coarse_groups_kernel",
                "_reduce_decode_route_coarse_kernel",
            ]
            state_route_math = "per_query"
        wide_local = bool(
            recursive
            and int(pool.head_dim) in (128, 256, 512)
            and 1 < kv_group_size <= 16
        )
        local = (
            ["included in compact static page-size-one attention"]
            if gqa_union_static_cap
            else
            ["_wide_gqa_local_scores_kernel", "_wide_gqa_local_value_kernel"]
            if wide_local
            else ["_mask_decode_routes_residual_mass_kernel"]
        )
        materialized_pages = bool(engine.recursive_materialize_page_scores)
        page_route = (
            ["_materialize_page_summary_scores_gqa_kernel"]
            if materialized_pages
            else ["inline page-summary scoring in exact-leaf kernel"]
        )
        effective_leaf_page_block_n = (
            int(engine.recursive_page_select_block_n)
            if materialized_pages
            else int(engine.decode_block_n)
        )
        if gqa_union and gqa_union_aiter_final:
            if gqa_union_static_cap:
                exact_leaf_kernel = (
                    "wide_gqa_indexed_page1_attention (split QK/PV D=512)"
                    if int(pool.head_dim) == 512
                    else (
                        "kernel_page1_attention_3d_bias (one compact persistent "
                        "sink + exact-small + coarse-large + local scan; no "
                        "routing or mask)"
                    )
                )
            elif gqa_union_fixed_mask:
                exact_leaf_kernel = (
                    "wide_gqa_indexed_page1_attention (split QK/PV D=512, "
                    "persistent route-prepared mask)"
                    if int(pool.head_dim) == 512
                    else (
                        "kernel_page1_attention_3d_bias_fixed_mask (persistent "
                        "centroid-major index list, M16/N"
                        f"{int(getattr(pool.settings, 'decode_gqa_fixed_mask_block_n', 64))}, "
                        "route-prepared byte mask, block fast-fail before lane "
                        "mask, K/V, and MFMA)"
                    )
                )
            elif gqa_union_staged_fixed:
                exact_leaf_kernel = (
                    "two concurrent kernel_page1_attention_3d_bias calls: "
                    "fixed local + sink + all centroids, and exact leaves; "
                    "_subtract_union_and_merge_aiter_exact_kernel removes "
                    "opened coarse entries and merges the branches"
                )
            else:
                exact_leaf_kernel = (
                    "kernel_page1_attention_3d_bias (AITER M16/N64 "
                    "page-size-1 leaves + local + sink + fixed all-centroid "
                    "suffix; opened/inactive entries use -inf and unopened "
                    "entries use log(count))"
                )
        elif gqa_union:
            exact_leaf_kernel = (
                "AITER kernel_unified_attention_3d (page-size-1 indices, "
                "TILE_SIZE=64 exact union) + "
                "_indexed_topk_gqa_split_decode_attention_kernel (closed "
                "coarse + local)"
                if gqa_union_hip
                else "_indexed_topk_gqa_split_decode_attention_kernel "
                "(closed coarse + shared-union leaves + local)"
            )
        elif recursive:
            exact_leaf_kernel = (
                "_query_major_residual_page_attention_kernel (INDEXED=True)"
            )
        else:
            exact_leaf_kernel = "flat two-tier leaf dispatch"

        # Track the production direct-prefill branch as well as decode. Both
        # flat and recursive vLLM pools use the configured serving update
        # batch; recursive LOD retains its separately derived query/local
        # schedule.
        fused_prefill_route = bool(
            not static_prefill
            and
            engine.fused_prefill_route_coarse
            and int(pool.head_dim) <= 512
            and int(pool.value_dim) <= 256
            and engine.routing_normalization == "none"
            and int(engine.routing_rope_fast_pairs) == 0
            and not bool(engine.routing_rope_jensen)
            and float(engine.routing_count_bias) == 1.0
            and float(engine.routing_variance_bias) == 0.0
            and int(engine.prefill_two_level_topk or engine.two_level_topk) <= 16
        )
        if static_prefill:
            prefill_route_kernels = [
                "_state_route_logits (all centroids)",
                (
                    "_route_logits_coarse_attention_kernel "
                    "(only centroids above the static leaf cap)"
                ),
            ]
        elif fused_prefill_route:
            effective_prefill_topk = int(
                engine.prefill_two_level_topk
                if engine.prefill_two_level_topk is not None
                else engine.two_level_topk
            )
            hierarchical_prefill_route = bool(
                engine.fused_prefill_stable_recompute
                and engine.fused_prefill_external_recompute
                and engine.prefill_hierarchical_route
                and effective_prefill_topk == 3
            )
            prefill_route_kernels = (
                [
                    "_route_logits_tile_topk_kernel",
                    "_reduce_route_logits_tile_topk_kernel",
                ]
                if hierarchical_prefill_route
                else ["_route_logits_topk_coarse_attention_kernel"]
            )
            if (
                engine.fused_prefill_stable_recompute
                and engine.fused_prefill_external_recompute
            ):
                prefill_route_kernels.append("_route_logits_coarse_attention_kernel")
        else:
            effective_prefill_topk = int(
                engine.prefill_two_level_topk
                if engine.prefill_two_level_topk is not None
                else engine.two_level_topk
            )
            if (
                engine.prefill_hierarchical_route
                and effective_prefill_topk == 3
            ):
                prefill_route_kernels = [
                    "_route_logits_tile_topk_kernel",
                    "_reduce_route_logits_tile_topk_kernel",
                ]
            else:
                prefill_route_kernels = [
                    "_route_score_group_candidates_kernel",
                    "_reduce_route_group_candidates_kernel",
                    "_reorder_topk_like_torch_kernel",
                ]
            prefill_route_kernels.append(
                "torch.matmul/softmax/matmul coarse path"
                if int(pool.value_dim) > 256
                else "_route_logits_coarse_attention_kernel"
            )
        if int(pool.head_dim) >= 512:
            prefill_local_kernel = "tiled torch.matmul/softmax/matmul"
        elif engine.prefill_local_attention_backend == "aiter":
            prefill_local_kernel = "aiter.ops.mha.flash_attn_func (CK FMHA v3)"
        else:
            prefill_local_kernel = "aten._scaled_dot_product_flash_attention"
        record = {
            "levels": int(pool.settings.levels),
            "query_heads": int(pool.query_heads),
            "kv_heads": int(pool.kv_heads),
            "kv_group_size": kv_group_size,
            "head_dim": int(pool.head_dim),
            "state_capacity": int(pool.state_capacity),
            "page_capacity": int(pool.page_capacity),
            "local_capacity": int(pool.local_capacity),
            "local_window": int(engine.local_len),
            "leaf_dtype": str(page["leaf_k"].dtype),
            "state_route_backend": engine.recursive_state_route_backend,
            "state_route_kernels": state_route,
            "state_route_math": state_route_math,
            "configured_route_group_size": int(engine.decode_route_group_size),
            "configured_route_segment_tiles": int(
                engine.decode_route_segment_tiles
            ),
            "configured_route_num_warps": int(engine.decode_route_num_warps),
            "configured_route_reduce_num_warps": int(
                engine.decode_route_reduce_num_warps
            ),
            "configured_route_use_dot": bool(engine.decode_route_use_dot),
            "configured_route_parallel_reduce": bool(
                engine.decode_route_parallel_reduce
            ),
            "configured_decode_hierarchical_route": bool(
                engine.decode_route_parallel_reduce
            ),
            "requested_decode_hierarchical_route": getattr(
                pool.settings, "decode_hierarchical_route", None
            ),
            "configured_decode_geometry_tuning": bool(
                pool.settings.decode_geometry_tuning
            ),
            "configured_decode_centroid_major_hip": bool(
                getattr(pool.settings, "decode_centroid_major_hip", False)
            ),
            "configured_gqa_union_decode": bool(
                getattr(pool.settings, "decode_gqa_union", False)
            ),
            "configured_gqa_union_hip": bool(
                getattr(pool.settings, "decode_gqa_union_hip", False)
            ),
            "configured_gqa_staged_fixed_aiter": bool(
                getattr(
                    pool.settings,
                    "decode_gqa_staged_fixed_aiter",
                    False,
                )
            ),
            "configured_gqa_static_leaf_aiter": bool(
                getattr(
                    pool.settings, "decode_gqa_static_leaf_aiter", False
                )
            ),
            "configured_diagnostic_static_preselected": bool(
                getattr(
                    pool.settings, "diagnostic_static_preselected", False
                )
            ),
            "configured_gqa_fixed_mask_aiter": bool(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_aiter",
                    False,
                )
            ),
            "configured_gqa_fixed_mask_block_n": int(
                getattr(pool.settings, "decode_gqa_fixed_mask_block_n", 64)
            ),
            "configured_gqa_fixed_mask_segments": int(
                getattr(pool.settings, "decode_gqa_fixed_mask_segments", 128)
            ),
            "configured_gqa_fixed_mask_adaptive_segments": bool(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_adaptive_segments",
                    False,
                )
            ),
            "configured_gqa_fixed_mask_reduce_block_d": int(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_reduce_block_d",
                    0,
                )
            ),
            "configured_gqa_fixed_mask_direct_routes": bool(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_direct_routes",
                    False,
                )
            ),
            "configured_gqa_fixed_mask_scan_num_warps": int(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_scan_num_warps",
                    2,
                )
            ),
            "configured_gqa_fixed_mask_scan_waves_per_eu": int(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_scan_waves_per_eu",
                    2,
                )
            ),
            "configured_gqa_fixed_mask_scan_num_stages": int(
                getattr(
                    pool.settings,
                    "decode_gqa_fixed_mask_scan_num_stages",
                    2,
                )
            ),
            "configured_gqa_predicted_mass": bool(
                getattr(pool.settings, "decode_gqa_predicted_mass", False)
            ),
            "configured_gqa_mass_fraction": getattr(
                pool.settings, "decode_gqa_mass_fraction", None
            ),
            "configured_decode_max_open_leaves": getattr(
                pool.settings, "decode_max_open_leaves", None
            ),
            "configured_open_count": int(pool.settings.open_count),
            "configured_state_premerge_factor": int(
                pool.settings.state_premerge_factor
            ),
            "effective_prefill_open_count": int(
                engine.prefill_two_level_topk
                if engine.prefill_two_level_topk is not None
                else engine.two_level_topk
            ),
            "configured_prefill_coarse_block_m": int(
                engine.coarse_route_block_m
            ),
            "configured_prefill_coarse_block_n": int(
                engine.coarse_route_block_n
            ),
            "configured_prefill_coarse_num_warps": int(
                engine.coarse_route_num_warps
            ),
            "configured_prefill_coarse_max_grouped_rows": int(
                engine.prefill_coarse_max_grouped_rows
            ),
            "configured_prefill_overlap_coarse_leaf": bool(
                engine.prefill_overlap_coarse_leaf
            ),
            "configured_prefill_overlap_local_lod": bool(
                engine.prefill_overlap_local_lod
            ),
            "configured_prefill_hierarchical_route": bool(
                engine.prefill_hierarchical_route
            ),
            "configured_prefill_leaf_visit_cap": getattr(
                pool.settings, "prefill_leaf_visit_cap", None
            ),
            "configured_decode_route_cohort": bool(
                getattr(pool.settings, "decode_route_cohort", False)
            ),
            "effective_decode_route_leaf_limit": (
                pool._decode_route_leaf_limit()
            ),
            "configured_route_post_dot_normalize": bool(
                engine.decode_route_post_dot_normalize
            ),
            "configured_route_post_pv_normalize": bool(
                engine.decode_route_post_pv_normalize
            ),
            "effective_route_group_size": effective_route_group_size,
            "effective_route_segment_tiles": effective_route_segment_tiles,
            "effective_route_segment_width": (
                effective_route_group_size * effective_route_segment_tiles
            ),
            "effective_route_num_warps": effective_route_num_warps,
            "effective_route_reduce_num_warps": (
                effective_route_reduce_num_warps
            ),
            "effective_route_use_dot": effective_route_use_dot,
            "engine_long_d128_override_exists": engine_long_d128_override_exists,
            "engine_long_d128_override_bypassed_by_vllm_pool": (
                engine_long_d128_override_exists
            ),
            "observed_context_len": context_len,
            "pool_request_capacity": int(pool.request_capacity),
            "route_gqa_grouped": bool(engine.decode_route_gqa_grouped),
            "state_score_block_n": (
                64
                if int(pool.head_dim) == 128
                else 32
                if int(pool.head_dim) == 256
                else 16
            ),
            "state_score_num_warps": 2,
            "state_score_num_stages": 3,
            "state_topk_block_n": 128,
            "state_topk_num_warps": 2,
            "state_pv_block_n": 64,
            "state_pv_block_d": 128,
            "state_pv_splits": min(8, int(engine.decode_split_kv)),
            "local_kernels": local,
            "local_score_block_m": 16,
            "local_score_block_n": 32,
            "local_score_num_warps": 4,
            "local_value_block_m": 16,
            "local_value_block_k": 32,
            "local_value_block_d": 32,
            "local_value_num_warps": 4,
            "materialize_page_scores": materialized_pages,
            "page_route_kernels": page_route,
            "page_score_block_n": int(engine.recursive_page_score_block_n),
            "page_score_num_warps": int(engine.recursive_page_score_num_warps),
            "page_select_block_n": int(engine.recursive_page_select_block_n),
            "exact_leaf_kernel": exact_leaf_kernel,
            "configured_leaf_block_n": int(engine.decode_block_n),
            "effective_leaf_page_block_n": effective_leaf_page_block_n,
            "leaf_num_warps": int(engine.decode_num_warps),
            "hash_probes": int(engine._page_lookup_probes(page)),
            "final_reduce_kernel": (
                "_reduce_wide_gqa_indexed_segments_kernel"
                if gqa_union_aiter_final and int(pool.head_dim) == 512
                else
                (
                    (
                        (
                            "_reduce_aiter_page1_segments_with_predicted_lse_kernel "
                            "(writes output, retains remote LSE, and prepares "
                            "the next route epoch)"
                            if gqa_union_predicted_mass
                            else "_reduce_aiter_page1_segments_with_lse_kernel "
                            "(writes final output)"
                        )
                        if gqa_union_fixed_mask
                        else (
                            "two fused segment/LSE reductions + "
                            "_subtract_union_and_merge_aiter_exact_kernel"
                            if gqa_union_staged_fixed
                            else "AITER reduce_segments (writes final output)"
                        )
                    )
                    if gqa_union_aiter_final
                    else (
                        "_reduce_split_decode_lod_attention_with_aiter_exact_kernel"
                        if gqa_union_hip
                        else "_reduce_split_decode_lod_attention_kernel"
                    )
                )
                if gqa_union
                else "_reduce_routed_split_decode_lod_attention_kernel"
            ),
            "final_reduce_num_warps": int(engine.decode_final_reduce_num_warps),
            "decode_split_kv": int(engine.decode_split_kv),
            "prefill_mode": pool.settings.prefill_mode,
            "configured_prefill_static_leaf_aiter": static_prefill,
            "configured_prefill_static_leaf_cap_min": int(
                getattr(pool.settings, "prefill_static_leaf_cap_min", 16)
            ),
            "configured_static_leaf_cap_divisor": int(
                getattr(pool.settings, "static_leaf_cap_divisor", 16)
            ),
            "configured_static_cohort_never_readmit": bool(
                getattr(pool.settings, "static_cohort_never_readmit", False)
            ),
            "configured_prefill_local_backend": (
                engine.prefill_local_attention_backend
            ),
            "effective_prefill_local_kernel": prefill_local_kernel,
            "prefill_fused_route_eligible": fused_prefill_route,
            "prefill_route_kernels": prefill_route_kernels,
            "prefill_leaf_kernel": (
                (
                    (
                        "_static_cap_wide_indexed_prefill_kernel, Triton "
                        "M16/N16 regular indexed attention over every centroid "
                        "at or below the static cap"
                    )
                    if int(pool.head_dim) == 512
                    else (
                        "AITER mha_batch_prefill_func, page-size-one indexed "
                        "concatenation of every centroid at or below the static cap"
                    )
                )
                if static_prefill
                else (
                    (
                        "paged_leaf_attention expert/MFMA over complete selected "
                        "centroids (recursive page archive retained for decode)"
                        if bool(
                            getattr(engine, "recursive_prefill_all_leaves", False)
                        )
                        else "_query_major_residual_page_attention_kernel "
                        "(INDEXED=True, recursive residual pages)"
                    )
                    if recursive
                    else str(engine.leaf_layout)
                )
            ),
            "requested_recursive_prefill_all_leaves": getattr(
                pool.settings, "recursive_prefill_all_leaves", None
            ),
            "configured_recursive_prefill_all_leaves": bool(
                getattr(engine, "recursive_prefill_all_leaves", False)
            ),
            "prefill_leaf_page_block_n": (
                int(engine.recursive_page_block_n)
                if recursive
                else int(engine.leaf_block_n)
            ),
            "prefill_leaf_num_warps": int(engine.leaf_num_warps),
            "prefill_chunk_len": int(engine.prefill_chunk_len),
            "prefill_state_update_len": int(engine.prefill_state_update_len),
            "prefill_schedule_source": (
                "KernelRecursivePagedLODAttention derived from chunk_size"
                if recursive
                else "VLLM_LOD_PREFILL_* settings"
            ),
        }
        key = (
            f"q{record['query_heads']}_kv{record['kv_heads']}_"
            f"d{record['head_dim']}_{record['state_route_backend']}_"
            f"pages{int(materialized_pages)}"
        )
        existing = records.get(key)
        if existing is None:
            record["layers"] = 1
            records[key] = record
        else:
            existing["layers"] = int(existing["layers"]) + 1
    return {"schema_version": 6, "records": list(records.values())}


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
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        engine = pool.engine
        if hasattr(engine, "_lod_phase_timing_events"):
            # Multi-length panels install the profiler once per length.  The
            # wrappers already close over this dictionary, so reuse it and
            # clear the completed length instead of reporting that no LOD
            # layers were found on the second install.
            for pairs in engine._lod_phase_timing_events.values():
                pairs.clear()
            leaf_events = getattr(engine, "_lod_leaf_timing_events", {})
            for pairs in leaf_events.values():
                pairs.clear()
            installed += 1
            continue
        events = {name: [] for name in methods}
        # The outer method wrappers see both prefill and decode calls.  Keep
        # shape-classified totals as well so an inclusive ``two_level`` timer
        # can be compared with the disjoint fused-decode subphases without
        # mixing in the much larger prefill invocation.
        for name in methods:
            events[f"{name}_decode"] = []
            events[f"{name}_prefill"] = []
        engine._lod_phase_timing_events = events
        # The fused recursive decode path has its own phase boundaries inside
        # one engine call.  Share the same event sink so its route, local,
        # exact-leaf, and final-reduction costs appear in the profile.
        engine._lod_decode_timing_events = events
        if getattr(engine, "recursive_page_lod", False):
            # Recursive prefill calls the residual-page function directly
            # instead of passing through ``_paged_leaf_attention``.
            engine._lod_leaf_timing_events = {"total": events["exact_leaf"]}
        else:
            # The expert leaf kernel exposes dispatch, query quantization/pack,
            # attention, and reduction boundaries. Keep those separate from
            # the outer exact-leaf phase so INT8 overhead is attributable.
            engine._lod_leaf_timing_events = {}
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
                sequence = next(
                    (
                        value
                        for value in args
                        if isinstance(value, torch.Tensor)
                        and value.ndim >= 3
                    ),
                    None,
                )
                suffix = (
                    "decode"
                    if sequence is not None and int(sequence.size(2)) == 1
                    else "prefill"
                )
                __events[f"{__phase}_{suffix}"].append((begin, end))
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
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        events = getattr(pool.engine, "_lod_phase_timing_events", {})
        for phase, pairs in events.items():
            totals[phase] = totals.get(phase, 0.0) + sum(
                float(begin.elapsed_time(end)) for begin, end in pairs
            )
            calls[phase] = calls.get(phase, 0) + len(pairs)
        if not getattr(pool.engine, "recursive_page_lod", False):
            leaf_events = getattr(pool.engine, "_lod_leaf_timing_events", {})
            for leaf_phase, pairs in leaf_events.items():
                phase = f"exact_leaf_{leaf_phase}"
                totals[phase] = totals.get(phase, 0.0) + sum(
                    float(begin.elapsed_time(end)) for begin, end in pairs
                )
                calls[phase] = calls.get(phase, 0) + len(pairs)
    return {
        phase: {"milliseconds": totals[phase], "calls": calls[phase]}
        for phase in sorted(totals)
    }


def install_full_attention_timers(model) -> int:
    """Time real native attention calls on global-attention layers.

    This deliberately wraps each layer's installed attention implementation
    instead of benchmarking a contiguous-KV surrogate.  The latter can choose
    a materially different kernel from vLLM's production paged-cache path.
    Existing ``all`` and batch-specific events remain decode-only; prefill is
    accumulated separately so adding this audit does not change decode-profile
    semantics.
    """
    import torch

    installed = 0
    for module in model.modules():
        impl = getattr(module, "impl", None)
        if impl is None or getattr(module, "_vllm_lod_pool", None) is not None:
            continue
        if not all(
            hasattr(impl, name)
            for name in ("forward", "num_heads", "num_kv_heads", "head_size")
        ):
            continue
        # vLLM represents an unbounded decoder window as (-1, -1).  Restrict
        # this comparison to the layers that LOD would replace.
        window = getattr(impl, "sliding_window", None)
        if window is not None and (
            not isinstance(window, (tuple, list)) or tuple(window) != (-1, -1)
        ):
            continue
        if hasattr(impl, "_full_attention_timing_events"):
            for pairs in impl._full_attention_timing_events.values():
                pairs.clear()
            installed += 1
            continue

        events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {"all": [], "prefill": []}
        impl._full_attention_timing_events = events
        impl._full_attention_dispatch_observations = {}
        original = impl.forward

        def timed(
            *args,
            __original=original,
            __events=events,
            __impl=impl,
            **kwargs,
        ):
            metadata = kwargs.get("attn_metadata")
            if metadata is None and len(args) > 5:
                metadata = args[5]
            is_decode = bool(
                metadata is not None
                and int(getattr(metadata, "max_query_len", 0)) == 1
            )
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = __original(*args, **kwargs)
            end.record()
            if not is_decode:
                __events["prefill"].append((begin, end))
                return result
            batch = int(getattr(metadata, "num_actual_tokens", 0))
            pair = (begin, end)
            __events["all"].append(pair)
            __events.setdefault(f"b{batch}", []).append(pair)
            observations = __impl._full_attention_dispatch_observations
            key = (
                f"b{batch}_q{int(getattr(metadata, 'max_query_len', 0))}_"
                f"kv{int(getattr(metadata, 'max_seq_len', 0))}"
            )
            observations[key] = int(observations.get(key, 0)) + 1
            return result

        impl.forward = timed
        installed += 1
    return installed


def inspect_full_attention_dispatch(model) -> dict[str, object]:
    """Describe the installed native implementation on global layers."""
    records: dict[str, dict[str, object]] = {}
    for module in model.modules():
        impl = getattr(module, "impl", None)
        if impl is None or getattr(module, "_vllm_lod_pool", None) is not None:
            continue
        if not all(
            hasattr(impl, name)
            for name in ("num_heads", "num_kv_heads", "head_size")
        ):
            continue
        window = getattr(impl, "sliding_window", None)
        if window is not None and (
            not isinstance(window, (tuple, list)) or tuple(window) != (-1, -1)
        ):
            continue
        attention_callable = getattr(impl, "unified_attention", None)
        callable_name = (
            f"{getattr(attention_callable, '__module__', '<unknown>')}."
            f"{getattr(attention_callable, '__name__', type(attention_callable).__name__)}"
            if attention_callable is not None
            else "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
        )
        record = {
            "implementation_class": (
                f"{type(impl).__module__}.{type(impl).__name__}"
            ),
            "attention_callable": callable_name,
            "query_heads": int(impl.num_heads),
            "kv_heads": int(impl.num_kv_heads),
            "kv_group_size": int(impl.num_heads // impl.num_kv_heads),
            "head_dim": int(impl.head_size),
            "sliding_window": window,
            "kv_cache_dtype": str(getattr(impl, "kv_cache_dtype", "unknown")),
        }
        key = (
            f"{record['implementation_class']}_q{record['query_heads']}_"
            f"kv{record['kv_heads']}_d{record['head_dim']}"
        )
        if key not in records:
            record["layers"] = 1
            records[key] = record
        else:
            records[key]["layers"] = int(records[key]["layers"]) + 1
    return {"schema_version": 1, "records": list(records.values())}


def summarize_full_attention_timers(model) -> dict[str, object]:
    """Aggregate production global-attention prefill and decode events."""
    import torch

    torch.cuda.synchronize()
    total_ms = 0.0
    calls = 0
    prefill_ms = 0.0
    prefill_calls = 0
    geometries: dict[str, int] = {}
    implementations: dict[str, int] = {}
    callables: dict[str, int] = {}
    observations: dict[str, int] = {}
    by_batch: dict[str, dict[str, float | int]] = {}
    for module in model.modules():
        impl = getattr(module, "impl", None)
        events = getattr(impl, "_full_attention_timing_events", None)
        if events is None:
            continue
        layer_total = sum(
            float(begin.elapsed_time(end)) for begin, end in events["all"]
        )
        total_ms += layer_total
        calls += len(events["all"])
        prefill_ms += sum(
            float(begin.elapsed_time(end)) for begin, end in events["prefill"]
        )
        prefill_calls += len(events["prefill"])
        for batch, pairs in events.items():
            if batch in ("all", "prefill"):
                continue
            row = by_batch.setdefault(batch, {"milliseconds": 0.0, "calls": 0})
            row["milliseconds"] = float(row["milliseconds"]) + sum(
                float(begin.elapsed_time(end)) for begin, end in pairs
            )
            row["calls"] = int(row["calls"]) + len(pairs)
        geometry = (
            f"q{int(impl.num_heads)}_kv{int(impl.num_kv_heads)}_"
            f"d{int(impl.head_size)}"
        )
        geometries[geometry] = geometries.get(geometry, 0) + 1
        implementation = f"{type(impl).__module__}.{type(impl).__name__}"
        implementations[implementation] = implementations.get(implementation, 0) + 1
        attention_callable = getattr(impl, "unified_attention", None)
        if attention_callable is not None:
            callable_name = (
                f"{getattr(attention_callable, '__module__', '<unknown>')}."
                f"{getattr(attention_callable, '__name__', type(attention_callable).__name__)}"
            )
        else:
            callable_name = "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
        callables[callable_name] = callables.get(callable_name, 0) + 1
        for key, count in getattr(
            impl, "_full_attention_dispatch_observations", {}
        ).items():
            observations[key] = observations.get(key, 0) + int(count)
    for row in by_batch.values():
        row["microseconds_per_call"] = (
            1000.0 * float(row["milliseconds"]) / int(row["calls"])
        )
    return {
        "milliseconds": total_ms,
        "calls": calls,
        "microseconds_per_call": 1000.0 * total_ms / calls if calls else None,
        "prefill_milliseconds": prefill_ms,
        "prefill_calls": prefill_calls,
        "prefill_microseconds_per_call": (
            1000.0 * prefill_ms / prefill_calls if prefill_calls else None
        ),
        "layers_by_geometry": geometries,
        "implementation_classes": implementations,
        "attention_callables": callables,
        "dispatch_observations": observations,
        "by_batch": by_batch,
    }


def install_lod_total_timers(model) -> int:
    """Time one complete production LOD decode call without inner events."""
    import torch

    installed = 0
    seen: set[int] = set()
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None or id(pool) in seen:
            continue
        seen.add(id(pool))
        if hasattr(pool, "_lod_total_timing_events"):
            for pairs in pool._lod_total_timing_events.values():
                pairs.clear()
            installed += 1
            continue
        events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {"all": []}
        pool._lod_total_timing_events = events
        original = pool.decode

        def timed(*args, __original=original, __events=events, **kwargs):
            metadata = kwargs.get("metadata")
            if metadata is None and len(args) > 3:
                metadata = args[3]
            batch = int(getattr(metadata, "num_actual_tokens", 0))
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = __original(*args, **kwargs)
            end.record()
            pair = (begin, end)
            __events["all"].append(pair)
            __events.setdefault(f"b{batch}", []).append(pair)
            return result

        pool.decode = timed
        installed += 1
    return installed


def summarize_lod_total_timers(model) -> dict[str, object]:
    """Aggregate clean, complete LOD decode calls on one worker."""
    import torch

    torch.cuda.synchronize()
    totals: dict[str, float] = {}
    calls: dict[str, int] = {}
    seen: set[int] = set()
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None or id(pool) in seen:
            continue
        seen.add(id(pool))
        for batch, pairs in getattr(pool, "_lod_total_timing_events", {}).items():
            if not pairs:
                continue
            totals[batch] = totals.get(batch, 0.0) + sum(
                float(begin.elapsed_time(end)) for begin, end in pairs
            )
            calls[batch] = calls.get(batch, 0) + len(pairs)
    rows = {
        batch: {
            "milliseconds": totals[batch],
            "calls": calls[batch],
            "microseconds_per_call": 1000.0 * totals[batch] / calls[batch],
        }
        for batch in sorted(totals)
        if calls[batch]
    }
    return rows


def configure_lod_model(
    model,
    *,
    leaf_num_warps: int | None,
    recursive_page_block_n: int | None,
    recursive_state_route_backend: str | None,
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
) -> int:
    """Apply benchmark-only kernel tuning before the warmup request."""
    configured = 0
    for module in model.modules():
        pool = getattr(module, "_vllm_lod_pool", None)
        if pool is None:
            continue
        if leaf_num_warps is not None:
            pool.engine.leaf_num_warps = int(leaf_num_warps)
        if recursive_page_block_n is not None:
            pool.engine.recursive_page_block_n = int(recursive_page_block_n)
        if recursive_state_route_backend is not None:
            pool.engine.recursive_state_route_backend = recursive_state_route_backend
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
        configured += 1
    return configured


def main() -> None:
    args = parse_args()
    if args.lod_recursive_state_route_backend is not None:
        # This backend changes graph-reserved scratch, so select it before
        # vLLM constructs the model rather than mutating it after graph capture.
        os.environ["VLLM_LOD_RECURSIVE_STATE_ROUTE_BACKEND"] = (
            args.lod_recursive_state_route_backend
        )
    prompt_lengths = args.lengths or [args.length] * args.batch_size
    if (
        args.batch_size < 1
        or min(prompt_lengths) < 2
        or args.decode_tokens < 1
        or args.repeats < 1
    ):
        raise ValueError(
            "length >= 2, batch size >= 1, decode tokens >= 1, and repeats >= 1 required"
        )
    if len(prompt_lengths) != args.batch_size:
        raise ValueError("--lengths must contain exactly --batch-size entries")
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    prompt_metadata: dict[str, Any] | None = None
    if args.prompt_source == "prolong":
        if len(set(prompt_lengths)) != 1:
            raise ValueError("ProLong speed prompts currently require uniform lengths")
        # Reuse the guarded prompt builder from the quality/speed panel: each
        # batch row consumes distinct shuffled documents and naturally short
        # documents are concatenated, never repeated.
        from eval_vllm_lod_niah_speed_panel import make_speed_prompts

        prompts, prompt_metadata = make_speed_prompts(
            tokenizer,
            prompt_lengths[0],
            args.batch_size,
            streaming=False,
        )
    else:
        seed = tokenizer(
            "LOD attention retains precise high-mass regions and summarizes the rest. ",
            add_special_tokens=False,
        )["input_ids"]
        # Give every request a distinct leading token pattern. Identical synthetic
        # prompts exercise the LOD pool's content-matched completed-prefix reuse,
        # which changes the number and batch shape of measured prefill calls based
        # on scheduler timing and obscures kernel occupancy comparisons.
        request_seeds = [
            tokenizer(
                f"LOD benchmark request {request_index}: ",
                add_special_tokens=False,
            )["input_ids"]
            + seed
            for request_index in range(args.batch_size)
        ]
        prompts = [
            {
                "prompt_token_ids": (
                    request_seed
                    * ((length + len(request_seed) - 1) // len(request_seed))
                )[:length]
            }
            for request_seed, length in zip(request_seeds, prompt_lengths)
        ]
    prompt_tokens = sum(prompt_lengths)
    # Cap each request at a consistent 16K scheduler chunk while retaining the
    # actual request batch. ``max_num_batched_tokens`` is an aggregate budget;
    # setting it to only 16K serializes a nominal batch of eight and gives
    # misleading kernel-occupancy results. This 8 x 16K default also stays far
    # below the ROCm Qwen3.5 GDN fault seen for one 8 x 64K (524K-token) step.
    per_request_prefill_chunk = min(max(prompt_lengths), 16_384)
    max_batched = args.max_num_batched_tokens or min(
        prompt_tokens,
        args.batch_size * per_request_prefill_chunk,
    )
    long_prefill_token_threshold = (
        args.long_prefill_token_threshold or per_request_prefill_chunk
    )
    kwargs = {
        "model": args.checkpoint,
        "load_format": os.getenv("VLLM_WEIGHT_CACHE_LOAD_FORMAT", "ipc_cache"),
        "dtype": "bfloat16",
        "max_model_len": max(prompt_lengths) + args.decode_tokens + 16,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": max_batched,
        "long_prefill_token_threshold": long_prefill_token_threshold,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": not args.disable_async_scheduling,
        "jit_monitor_verbose": args.jit_monitor_verbose,
        "enable_prefix_caching": args.enable_prefix_caching,
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
    llm = register_llm_shutdown(LLM(**kwargs))
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
        )
    ):
        configured = llm.apply_model(
            functools.partial(
                configure_lod_model,
                leaf_num_warps=args.lod_leaf_num_warps,
                recursive_page_block_n=args.lod_recursive_page_block_n,
                recursive_state_route_backend=(
                    args.lod_recursive_state_route_backend
                ),
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
    *_, reference_token_ids = timed_generate(
        llm, prompts, many, return_token_ids=True
    )
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
    output_mismatch_runs = 0
    for _ in range(args.repeats):
        elapsed, prefill_elapsed, decode_elapsed, token_ids = timed_generate(
            llm, prompts, many, return_token_ids=True
        )
        if token_ids != reference_token_ids:
            output_mismatch_runs += 1
            if not args.allow_output_mismatch:
                raise RuntimeError(
                    "deterministic benchmark output changed across identical runs"
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
        "load_format": kwargs["load_format"],
        "length": args.length if args.lengths is None else None,
        "lengths": prompt_lengths,
        "prompt_source": args.prompt_source,
        "prompt_metadata": prompt_metadata,
        "prompt_tokens": prompt_tokens,
        "batch_size": args.batch_size,
        "decode_tokens": args.decode_tokens,
        "decode_interval_tokens": decode_interval,
        "repeats": args.repeats,
        "output_mismatch_runs": output_mismatch_runs,
        "max_num_batched_tokens": max_batched,
        "long_prefill_token_threshold": long_prefill_token_threshold,
        "enforce_eager": args.enforce_eager,
        "async_scheduling_requested": not args.disable_async_scheduling,
        "prefix_caching_requested": args.enable_prefix_caching,
        "token_ids_sha256": hashlib.sha256(
            json.dumps(reference_token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "jit_monitor_verbose": args.jit_monitor_verbose,
        "num_gpu_blocks_override": args.num_gpu_blocks_override,
        "attention_backend": args.attention_backend,
        "lod_leaf_num_warps": args.lod_leaf_num_warps,
        "lod_recursive_page_block_n": args.lod_recursive_page_block_n,
        "lod_recursive_state_route_backend": (
            args.lod_recursive_state_route_backend
        ),
        "lod_prefill_chunk_len": args.lod_prefill_chunk_len,
        "lod_prefill_state_update_len": args.lod_prefill_state_update_len,
        "lod_direct_prefill_route": args.lod_direct_prefill_route,
        "lod_decode_route_group_size": args.lod_decode_route_group_size,
        "lod_decode_route_num_warps": args.lod_decode_route_num_warps,
        "lod_decode_route_reduce_num_warps": (
            args.lod_decode_route_reduce_num_warps
        ),
        "lod_decode_final_reduce_num_warps": (
            args.lod_decode_final_reduce_num_warps
        ),
        "lod_decode_block_n": args.lod_decode_block_n,
        "lod_decode_num_warps": args.lod_decode_num_warps,
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
        "marginal_decode_ms_per_token": (
            1000.0 * marginal_decode / marginal_tokens
            if marginal_tokens
            else None
        ),
        "marginal_decode_ms_per_batch_step": (
            1000.0 * marginal_decode / decode_interval
            if decode_interval
            else None
        ),
        "marginal_decode_tokens_per_second": (
            marginal_tokens / marginal_decode
            if marginal_decode and marginal_tokens
            else None
        ),
    }
    if args.mode == "lod":
        result["lod_diagnostics"] = llm.apply_model(inspect_lod_model)[0]
        result["lod_dispatch"] = llm.apply_model(inspect_lod_dispatch)[0]
    else:
        result["full_attention_dispatch"] = llm.apply_model(
            inspect_full_attention_dispatch
        )[0]
    if args.profile_lod_phases:
        result["lod_phase_profile"] = llm.apply_model(
            summarize_lod_phase_timers
        )[0]
    result["attention_memory"] = llm.apply_model(inspect_attention_memory)[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_registered_llms()
