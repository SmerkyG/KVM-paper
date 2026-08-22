#!/usr/bin/env python3
"""Measure warm end-to-end Qwen3.5 prefill and cached decode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.hf_pytorch_lod_attention import (
    Qwen3_5FastLODAttention,
    replace_qwen35_attention_with_lod,
    reset_hf_lod_caches,
)
from model.pytorch_lod_attention import LODConfig
from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.probe_qwen35_lod_niah import (
    load_text_model,
    require_qwen35_acceleration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument(
        "--mode", choices=("full", "two_level", "pytorch_lod"), required=True
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-microbatch-size", type=int)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--separate-sink-cache", action="store_true")
    parser.add_argument("--prefill-two-level-topk", type=int, default=3)
    parser.add_argument("--prefill-max-leaf-tokens", type=int)
    parser.add_argument("--dynamic-open-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-residual-mass", type=float)
    parser.add_argument("--dynamic-open-decode-top-p", type=float)
    parser.add_argument("--dynamic-open-decode-residual-mass", type=float)
    parser.add_argument("--reuse-dynamic-local-attention", action="store_true")
    parser.add_argument("--dynamic-open-residual-state-bound", action="store_true")
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument(
        "--all-centroid-top1",
        action="store_true",
        help="Use the experimental uniform exact-winner/residual path.",
    )
    parser.add_argument("--all-centroid-mean-residual", action="store_true")
    parser.add_argument(
        "--all-centroid-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--all-centroid-fused-block-m", type=int, default=16)
    parser.add_argument(
        "--all-centroid-fused-centroids", type=int, default=32
    )
    parser.add_argument(
        "--leaf-attention-backend",
        choices=("packed", "paged"),
        default="paged",
    )
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument("--recursive-page-block-n", type=int, default=4)
    parser.add_argument("--leaf-num-warps", type=int, default=1)
    parser.add_argument("--leaf-key-quant-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--leaf-value-quant-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--leaf-quant-group-size", type=int, default=32)
    parser.add_argument(
        "--leaf-quant-scale-mode", choices=("max", "l2"), default="max"
    )
    parser.add_argument(
        "--leaf-append-quant-scale-mode", choices=("max", "l2"), default="max"
    )
    parser.add_argument("--page-summary-quant-bits", type=int, choices=(0, 8), default=8)
    parser.add_argument(
        "--page-summary-scale-mode", choices=("max", "l2"), default="l2"
    )
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument(
        "--state-clustering-geometry",
        choices=("raw", "coherence"),
        default="raw",
        help="Centroid geometry used by the two-level LOD state updater.",
    )
    parser.add_argument(
        "--exact-coherence-matmul",
        action="store_true",
        help="Use the slower two-GEMM BF16 reference for coherence routing.",
    )
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument("--prefill-local-length", type=int)
    parser.add_argument("--prefill-state-update-length", type=int)
    parser.add_argument("--overflow-bipartite-merge", action="store_true")
    parser.add_argument("--overflow-bipartite-block-size", type=int, default=32)
    parser.add_argument("--overflow-bipartite-keep-ratio", type=float, default=0.5)
    parser.add_argument("--merge-before-append", action="store_true")
    parser.add_argument("--append-subblock-size", type=int, default=0)
    parser.add_argument("--union-bipartite-state", action="store_true")
    parser.add_argument("--state-precompact-direct-append", action="store_true")
    parser.add_argument("--leaf-inline-pages-per-slot", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--decode-warmup-steps", type=int, default=16)
    parser.add_argument("--decode-state-update-length", type=int, default=256)
    parser.add_argument("--decode-cache-headroom", type=int, default=256)
    parser.add_argument("--profile-decode-kernels", action="store_true")
    parser.add_argument("--profile-prefill-kernels", action="store_true")
    parser.add_argument("--disable-fused-decode", action="store_true")
    parser.add_argument("--disable-fused-decode-state-route", action="store_true")
    parser.add_argument("--no-clone-decode-routes", action="store_true")
    parser.add_argument("--decode-route-group-size", type=int, default=32)
    parser.add_argument("--decode-route-num-warps", type=int, default=2)
    parser.add_argument("--decode-route-reduce-num-warps", type=int, default=4)
    parser.add_argument("--decode-final-reduce-num-warps", type=int, default=4)
    parser.add_argument("--decode-split-kv", type=int)
    parser.add_argument("--decode-fuse-final-reduce", action="store_true")
    parser.add_argument("--decode-use-dot", action="store_true")
    parser.add_argument(
        "--decode-route-use-dot",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--decode-route-gqa-grouped",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--enable-fused-state-update", action="store_true")
    parser.add_argument(
        "--fused-state-routing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--fused-state-maxsim", action="store_true")
    parser.add_argument("--state-maxsim-block-m", type=int, default=16)
    parser.add_argument("--state-maxsim-block-n", type=int, default=32)
    parser.add_argument("--state-maxsim-num-warps", type=int, default=4)
    parser.add_argument("--disable-coarse-gqa", action="store_true")
    parser.add_argument("--disable-compact-coarse-bias", action="store_true")
    parser.add_argument(
        "--reuse-route-logits",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fused-prefill-route-coarse",
        "--enable-fused-prefill-route-coarse",
        dest="enable_fused_prefill_route_coarse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--split-prefill-local-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fused-prefill-residual-opening", action="store_true")
    parser.add_argument("--coarse-route-block-m", type=int, default=16)
    parser.add_argument("--coarse-route-block-n", type=int, default=32)
    parser.add_argument("--coarse-route-num-warps", type=int, default=4)
    parser.add_argument("--route-gqa-matmul", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def clear_lod_state(model: torch.nn.Module) -> None:
    reset_hf_lod_caches(model)
    for module in model.modules():
        if isinstance(module, Qwen3_5TwoLevelAttention) and hasattr(
            module, "_lod_state"
        ):
            delattr(module, "_lod_state")


def select_profile_sequence(tokenizer, dataset: str, sequence_length: int) -> torch.Tensor:
    """Build long natural-text profiles from deterministic 64K ProLong pieces."""
    piece_length = min(sequence_length, 65_536)
    piece_count = math.ceil(sequence_length / piece_length)
    pieces = select_sequences(
        tokenizer,
        dataset,
        piece_length,
        piece_count,
        0,
        1,
    )
    return torch.cat([tokens for _, tokens in pieces])[:sequence_length]


def cache_memory_breakdown(model: torch.nn.Module, past_key_values) -> dict[str, float]:
    """Report live persistent cache tensors, excluding transient activations."""

    def gib(tensors) -> float:
        return sum(t.numel() * t.element_size() for t in tensors) / (1024**3)

    lod_groups: dict[str, list[torch.Tensor]] = {}
    for module in model.modules():
        if isinstance(module, Qwen3_5FastLODAttention):
            cache = module._lod_cache
            if cache is None:
                continue
            for name in (
                "recent_key",
                "recent_value",
                "owner",
                "leaf_key",
                "leaf_value",
            ):
                value = getattr(cache, name)
                if isinstance(value, torch.Tensor):
                    lod_groups.setdefault(f"pytorch_lod.{name}", []).append(value)
            for name in ("key_sum", "value_sum", "count"):
                lod_groups.setdefault(f"pytorch_lod.state.{name}", []).append(
                    getattr(cache.state, name)
                )
            continue
        if not isinstance(module, Qwen3_5TwoLevelAttention):
            continue
        cache = getattr(module, "_lod_state", {})
        for name, value in cache.items():
            if name == "page_cache":
                for page_name, page_value in value.items():
                    if isinstance(page_value, torch.Tensor):
                        lod_groups.setdefault(f"lod_page_cache.{page_name}", []).append(
                            page_value
                        )
            elif isinstance(value, torch.Tensor):
                lod_groups.setdefault(f"lod_state.{name}", []).append(value)

    standard_groups: dict[str, list[torch.Tensor]] = {}
    if past_key_values is not None:
        for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
            values = getattr(past_key_values, name, ())
            standard_groups[f"qwen_cache.{name}"] = [
                value for value in values if isinstance(value, torch.Tensor)
            ]

    breakdown = {
        name: gib(tensors)
        for name, tensors in {**lod_groups, **standard_groups}.items()
        if tensors
    }
    breakdown["persistent_cache_total"] = sum(breakdown.values())
    return breakdown


def lod_cache_statistics(model: torch.nn.Module) -> dict[str, int | float]:
    max_slot_tokens = 0
    max_used_pages = 0
    overflow_failures = 0
    maximum_count_length_error = 0.0
    positive_count_empty_slots = 0
    for module in model.modules():
        if not isinstance(module, Qwen3_5TwoLevelAttention):
            continue
        cache = getattr(module, "_lod_state", {}).get("page_cache", {})
        slot_lengths = cache.get("slot_lengths")
        next_page = cache.get("next_page")
        overflow_flag = cache.get("overflow_flag")
        if isinstance(slot_lengths, torch.Tensor):
            max_slot_tokens = max(max_slot_tokens, int(slot_lengths.max().item()))
            state = getattr(module, "_lod_state", {})
            counts = state.get("counts")
            state_len = int(state.get("state_len", 0))
            if isinstance(counts, torch.Tensor) and state_len:
                active_counts = counts[..., :state_len, 0].float()
                active_lengths = slot_lengths[..., :state_len].float()
                maximum_count_length_error = max(
                    maximum_count_length_error,
                    float((active_counts - active_lengths).abs().max().item()),
                )
                positive_count_empty_slots += int(
                    ((active_counts > 0.5) & (active_lengths == 0)).sum().item()
                )
        if isinstance(next_page, torch.Tensor):
            max_used_pages = max(max_used_pages, int(next_page.max().item()))
        if isinstance(overflow_flag, torch.Tensor):
            overflow_failures += int(overflow_flag.item())
    return {
        "max_slot_tokens": max_slot_tokens,
        "max_used_pages_per_batch_head": max_used_pages,
        "overflow_hash_failures": overflow_failures,
        "maximum_count_length_error": maximum_count_length_error,
        "positive_count_empty_slots": positive_count_empty_slots,
    }


def snapshot_lod_states(
    model: torch.nn.Module,
) -> dict[Qwen3_5TwoLevelAttention, dict]:
    snapshots = {}
    for module in model.modules():
        if isinstance(module, Qwen3_5TwoLevelAttention):
            state = dict(module._lod_state)
            if "page_cache" in state:
                state["page_cache"] = dict(state["page_cache"])
            snapshots[module] = state
    return snapshots


def merge_batch_values(values, name: str | None = None):
    first = values[0]
    if name == "overflow_active":
        return any(bool(value) for value in values)
    if name == "overflow_safe_until":
        return min(int(value) for value in values)
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return torch.stack(values).max()
        return torch.cat(values, dim=0)
    if isinstance(first, dict):
        return {
            child_name: merge_batch_values(
                [value[child_name] for value in values], child_name
            )
            for child_name in first
        }
    if any(value != first for value in values[1:]):
        raise ValueError("microbatch LOD cache metadata differs")
    return first


def merge_batch_dicts(dicts: list[dict]) -> dict:
    """Merge cache dictionaries while releasing each source field promptly."""
    merged = {}
    for name in list(dicts[0]):
        values = [source.pop(name) for source in dicts]
        if isinstance(values[0], dict):
            merged[name] = merge_batch_dicts(values)
        else:
            merged[name] = merge_batch_values(values, name)
    return merged


def merge_qwen_caches(caches):
    merged = caches[0]
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        destination = getattr(merged, name)
        for layer in range(len(destination)):
            values = [getattr(cache, name)[layer] for cache in caches]
            tensors = [value for value in values if isinstance(value, torch.Tensor)]
            destination[layer] = torch.cat(tensors, dim=0) if tensors else None
    return merged


def avoid_oversized_fla_autotune(sequence_length: int, effective_batch: int) -> None:
    """Keep FLA l2norm indexing below its 32-bit flattened-row limit."""
    if sequence_length * effective_batch < 1_048_576 or effective_batch <= 2:
        return
    import importlib

    from fla.modules.l2norm import l2norm_fwd, l2norm_fwd_kernel

    safe_configs = [
        config
        for config in l2norm_fwd_kernel.configs
        if config.kwargs.get("BT") == 64 and config.num_warps == 4
    ]
    if len(safe_configs) != 1:
        raise RuntimeError("could not identify the safe FLA l2norm configuration")
    l2norm_fwd_kernel.configs = safe_configs

    def chunked_l2norm_fwd(
        x: torch.Tensor,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
    ):
        original_shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        max_rows = max((1 << 29) // int(flat.size(1)), 1)
        if int(flat.size(0)) <= max_rows:
            return l2norm_fwd(x, eps=eps, output_dtype=output_dtype)
        output = torch.empty_like(
            flat, dtype=(output_dtype if output_dtype is not None else flat.dtype)
        )
        rstd = torch.empty(flat.size(0), dtype=torch.float32, device=flat.device)
        for begin in range(0, int(flat.size(0)), max_rows):
            end = min(int(flat.size(0)), begin + max_rows)
            chunk_output, chunk_rstd = l2norm_fwd(
                flat[begin:end], eps=eps, output_dtype=output_dtype
            )
            output[begin:end].copy_(chunk_output)
            rstd[begin:end].copy_(chunk_rstd)
        return output.view(original_shape), rstd.view(original_shape[:-1])

    chunk_module = importlib.import_module("fla.ops.gated_delta_rule.chunk")
    chunk_module.l2norm_fwd = chunked_l2norm_fwd


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.prefill_microbatch_size is not None and not (
        0 < args.prefill_microbatch_size <= args.batch_size
    ):
        raise ValueError("prefill microbatch size must be in [1, batch size]")
    if args.repeats <= 0:
        raise ValueError("profile repeats must be positive")
    if args.prefill_two_level_topk is not None and not (
        0 <= args.prefill_two_level_topk <= 8
    ):
        raise ValueError("prefill top-k must be in [0, 8]")
    if args.prefill_max_leaf_tokens is not None and args.prefill_max_leaf_tokens <= 0:
        raise ValueError("maximum prefill leaf count must be positive")
    if args.decode_steps < 0:
        raise ValueError("decode steps must be non-negative")
    if args.decode_warmup_steps < 0:
        raise ValueError("decode warmup steps must be non-negative")
    if args.decode_state_update_length <= 0:
        raise ValueError("decode state update length must be positive")
    if args.decode_cache_headroom <= 0:
        raise ValueError("decode cache headroom must be positive")
    if args.prefill_chunk_length is not None and args.prefill_chunk_length <= 0:
        raise ValueError("prefill chunk length must be positive")
    if args.prefill_local_length is not None and args.prefill_local_length <= 0:
        raise ValueError("prefill local length must be positive")
    if (
        args.prefill_state_update_length is not None
        and args.prefill_state_update_length <= 0
    ):
        raise ValueError("prefill state update length must be positive")
    effective_prefill_chunk = (
        args.prefill_chunk_length or Qwen3_5TwoLevelAttention.prefill_chunk_len
    )
    effective_prefill_local = (
        args.prefill_local_length or Qwen3_5TwoLevelAttention.prefill_local_len
    )
    if effective_prefill_chunk > effective_prefill_local:
        raise ValueError("prefill chunk length cannot exceed prefill local length")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    avoid_oversized_fla_autotune(
        args.sequence_length,
        min(args.prefill_microbatch_size or args.batch_size, args.batch_size),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sequence = (
        select_profile_sequence(tokenizer, args.dataset, args.sequence_length)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
        .to(device)
    )
    model = load_text_model(
        args.checkpoint,
        "full" if args.mode == "pytorch_lod" else args.mode,
        args.two_level_topk,
        args.state_growth_factor,
        device,
        args.leaf_attention_backend,
        require_fla_fast_path=True,
    )
    acceleration = require_qwen35_acceleration(model)
    print("Qwen3.5 acceleration: " + json.dumps(acceleration, sort_keys=True))
    if args.mode == "pytorch_lod":
        replace_qwen35_attention_with_lod(
            model,
            config=LODConfig(
                chunk_size=256,
                local_window=512,
                state_growth_factor=args.state_growth_factor,
                state_min_size=256,
                protected_prefix=1,
                max_routes=8,
            ),
            open_count=args.two_level_topk,
        )
    dynamic_prefill_top_p = (
        args.dynamic_open_prefill_top_p
        if args.dynamic_open_prefill_top_p is not None
        else args.dynamic_open_top_p
    )
    dynamic_decode_top_p = (
        args.dynamic_open_decode_top_p
        if args.dynamic_open_decode_top_p is not None
        else args.dynamic_open_top_p
    )
    if (
        dynamic_decode_top_p is not None
        and args.dynamic_open_decode_residual_mass is not None
    ):
        raise ValueError(
            "decode top-p and full-mass residual opening are mutually exclusive"
        )
    if (
        dynamic_prefill_top_p is not None
        and args.dynamic_open_prefill_residual_mass is not None
    ):
        raise ValueError(
            "prefill top-p and full-mass residual opening are mutually exclusive"
        )
    if args.mode == "two_level":
        for module in model.modules():
            if isinstance(module, Qwen3_5TwoLevelAttention):
                module.prefill_two_level_topk = args.prefill_two_level_topk
                module.separate_sink_cache = args.separate_sink_cache
                module.prefill_max_leaf_tokens = args.prefill_max_leaf_tokens
                module.decode_state_update_len = args.decode_state_update_length
                module.decode_cache_headroom = args.decode_cache_headroom
                module.fused_decode_attention = not args.disable_fused_decode
                module.recursive_page_lod = args.recursive_page_lod
                module.all_centroid_top1 = args.all_centroid_top1
                module.all_centroid_disjoint_residual = not (
                    args.all_centroid_mean_residual
                )
                module.all_centroid_fused = args.all_centroid_fused
                module.all_centroid_fused_block_m = (
                    args.all_centroid_fused_block_m
                )
                module.all_centroid_fused_centroids_per_program = (
                    args.all_centroid_fused_centroids
                )
                module.all_centroid_fused_leaf_block_n = (
                    128 // args.all_centroid_fused_centroids
                )
                module.virtual_page_storage = args.virtual_page_storage
                module.recursive_page_block_n = args.recursive_page_block_n
                module.leaf_num_warps = args.leaf_num_warps
                module.leaf_key_quant_bits = args.leaf_key_quant_bits
                module.leaf_value_quant_bits = args.leaf_value_quant_bits
                module.leaf_quant_group_size = args.leaf_quant_group_size
                module.leaf_quant_scale_mode = args.leaf_quant_scale_mode
                module.leaf_append_quant_scale_mode = (
                    args.leaf_append_quant_scale_mode
                )
                module.page_summary_quant_bits = args.page_summary_quant_bits
                module.page_summary_scale_mode = args.page_summary_scale_mode
                module.state_clustering_normalization = "none"
                module.state_clustering_centroid_rescale = (
                    "coherence"
                    if args.state_clustering_geometry == "coherence"
                    else "none"
                )
                module.state_clustering_centroid_rescale_scope = (
                    "assignment"
                    if args.state_clustering_geometry == "coherence"
                    else "all"
                )
                module.coherence_single_matmul = not args.exact_coherence_matmul
                if args.prefill_chunk_length is not None:
                    module.prefill_chunk_len = args.prefill_chunk_length
                if args.prefill_local_length is not None:
                    module.prefill_local_len = args.prefill_local_length
                if args.prefill_state_update_length is not None:
                    module.prefill_state_update_len = (
                        args.prefill_state_update_length
                    )
                module.overflow_bipartite_merge = args.overflow_bipartite_merge
                module.overflow_bipartite_block_size = (
                    args.overflow_bipartite_block_size
                )
                module.overflow_bipartite_keep_ratio = (
                    args.overflow_bipartite_keep_ratio
                )
                module.state_merge_before_append = args.merge_before_append
                module.state_append_subblock_size = args.append_subblock_size
                module.state_union_bipartite = args.union_bipartite_state
                module.state_precompact_direct_append = (
                    args.state_precompact_direct_append
                )
                module.dynamic_open_prefill_top_p = dynamic_prefill_top_p
                module.dynamic_open_prefill_residual_mass = (
                    args.dynamic_open_prefill_residual_mass
                )
                module.dynamic_open_decode_top_p = dynamic_decode_top_p
                module.dynamic_open_decode_residual_mass = (
                    args.dynamic_open_decode_residual_mass
                )
                module.reuse_dynamic_local_attention = (
                    args.reuse_dynamic_local_attention
                )
                module.dynamic_open_residual_use_state_bound = (
                    args.dynamic_open_residual_state_bound
                )
                if args.leaf_inline_pages_per_slot is not None:
                    module.leaf_inline_pages_per_slot = (
                        args.leaf_inline_pages_per_slot
                    )
                module.fused_decode_state_route = (
                    not args.disable_fused_decode_state_route
                )
                module.fused_state_maxsim = args.fused_state_maxsim
                module.state_maxsim_block_m = args.state_maxsim_block_m
                module.state_maxsim_block_n = args.state_maxsim_block_n
                module.state_maxsim_num_warps = args.state_maxsim_num_warps
                module.clone_decode_routes = not args.no_clone_decode_routes
                module.decode_route_group_size = args.decode_route_group_size
                module.decode_route_num_warps = args.decode_route_num_warps
                module.decode_route_reduce_num_warps = (
                    args.decode_route_reduce_num_warps
                )
                module.decode_final_reduce_num_warps = (
                    args.decode_final_reduce_num_warps
                )
                if args.decode_split_kv is not None:
                    module.decode_split_kv = args.decode_split_kv
                module.decode_fuse_final_reduce = args.decode_fuse_final_reduce
                module.decode_use_dot = args.decode_use_dot
                if args.decode_route_use_dot is not None:
                    module.decode_route_use_dot = args.decode_route_use_dot
                if args.decode_route_gqa_grouped is not None:
                    module.decode_route_gqa_grouped = args.decode_route_gqa_grouped
                if args.enable_fused_state_update:
                    module.fused_state_update = True
                if args.fused_state_routing is not None:
                    module.fused_state_routing = args.fused_state_routing
                module.coarse_enable_gqa = not args.disable_coarse_gqa
                module.coarse_compact_bias = not args.disable_compact_coarse_bias
                if args.reuse_route_logits is not None:
                    module.reuse_route_logits_for_coarse = args.reuse_route_logits
                module.fused_prefill_route_coarse = (
                    args.enable_fused_prefill_route_coarse
                )
                module.split_prefill_local_attention = (
                    args.split_prefill_local_attention
                )
                module.fused_prefill_residual_opening = (
                    args.fused_prefill_residual_opening
                )
                module.coarse_route_block_m = args.coarse_route_block_m
                module.coarse_route_block_n = args.coarse_route_block_n
                module.coarse_route_num_warps = args.coarse_route_num_warps
                module.route_gqa_matmul = args.route_gqa_matmul

    def prefill():
        microbatch = args.prefill_microbatch_size
        if microbatch is None or microbatch == args.batch_size:
            return model(
                input_ids=sequence,
                use_cache=args.decode_steps > 0,
                logits_to_keep=1,
            )

        if not args.decode_steps:
            result = None
            for begin in range(0, args.batch_size, microbatch):
                clear_lod_state(model)
                result = model(
                    input_ids=sequence[begin : begin + microbatch],
                    use_cache=False,
                    logits_to_keep=1,
                )
            if result is None:
                raise AssertionError("prefill microbatch loop produced no output")
            return result

        results = []
        caches = []
        snapshots = []
        for begin in range(0, args.batch_size, microbatch):
            result = model(
                input_ids=sequence[begin : begin + microbatch],
                use_cache=True,
                logits_to_keep=1,
            )
            results.append(result)
            caches.append(result.past_key_values)
            snapshots.append(snapshot_lod_states(model))
        modules = list(snapshots[0])
        # The final microbatch is still referenced by the modules. Unlink it
        # before destructively consuming snapshots so completed source fields
        # can be released as soon as their merged tensor has been allocated.
        for module in modules:
            module._lod_state = {}
        for module in modules:
            module._lod_state = merge_batch_dicts(
                [snapshot.pop(module) for snapshot in snapshots]
            )
        result = results[-1]
        result.past_key_values = merge_qwen_caches(caches)
        return result

    def decode(past_key_values, steps: int, *, position_offset: int = 0):
        next_token = sequence[:, -1:]
        output = None
        for step in range(steps):
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                cache_position=torch.tensor(
                    [args.sequence_length + position_offset + step],
                    dtype=torch.long,
                    device=device,
                ),
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = output.past_key_values
        return output, past_key_values

    with torch.inference_mode():
        warm = prefill()
        warm_cache = warm.past_key_values
        if args.decode_steps:
            warm_decode, warm_cache = decode(warm_cache, min(args.decode_steps, 2))
            del warm_decode
        torch.cuda.synchronize(device)
        del warm, warm_cache
        clear_lod_state(model)
        torch.cuda.empty_cache()

        prefill_elapsed_ms = []
        decode_elapsed_ms = []
        decode_state_update_tokens_per_layer = []
        decode_state_update_intervals_per_layer = []
        cache_memory_gib = None
        cache_statistics = None
        finite = True
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.repeats):
            clear_lod_state(model)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = prefill()
            end.record()
            torch.cuda.synchronize(device)
            prefill_elapsed_ms.append(float(begin.elapsed_time(end)))
            finite = finite and bool(torch.isfinite(result.logits).all().item())
            cache = result.past_key_values
            if args.decode_steps:
                if args.decode_warmup_steps:
                    decode_warmup, cache = decode(
                        cache,
                        args.decode_warmup_steps,
                    )
                    del decode_warmup
                lod_modules = [
                    module
                    for module in model.modules()
                    if isinstance(module, Qwen3_5TwoLevelAttention)
                ]
                coverage_before = [
                    int(module._lod_state["coverage"]) for module in lod_modules
                ]
                decode_begin = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)
                decode_begin.record()
                decode_result, cache = decode(
                    cache,
                    args.decode_steps,
                    position_offset=args.decode_warmup_steps,
                )
                decode_end.record()
                torch.cuda.synchronize(device)
                decode_elapsed_ms.append(float(decode_begin.elapsed_time(decode_end)))
                coverage_after = [
                    int(module._lod_state["coverage"]) for module in lod_modules
                ]
                state_update_tokens = [
                    after - before
                    for before, after in zip(coverage_before, coverage_after)
                ]
                if args.mode == "two_level":
                    if not state_update_tokens or min(state_update_tokens) <= 0:
                        raise RuntimeError(
                            "timed decode interval did not include a state update; "
                            "increase --decode-steps so steady-state update cost "
                            "is amortized"
                        )
                    state_update_intervals = []
                    for module, update_tokens in zip(
                        lod_modules, state_update_tokens
                    ):
                        if update_tokens % module.decode_state_update_len:
                            raise AssertionError(
                                "decode state coverage advanced off schedule"
                            )
                        state_update_intervals.append(
                            update_tokens // module.decode_state_update_len
                        )
                    decode_state_update_tokens_per_layer.append(
                        state_update_tokens
                    )
                    decode_state_update_intervals_per_layer.append(
                        state_update_intervals
                    )
                if decode_result is None:
                    raise AssertionError("decode produced no output")
                finite = finite and bool(
                    torch.isfinite(decode_result.logits).all().item()
                )
                del decode_result
            if cache_memory_gib is None:
                cache_memory_gib = cache_memory_breakdown(model, cache)
                cache_statistics = lod_cache_statistics(model)
            del result, cache

        prefill_kernel_profile = []
        if args.profile_prefill_kernels:
            clear_lod_state(model)
            with torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                )
            ) as profiler:
                profile_prefill = prefill()
                torch.cuda.synchronize(device)
            del profile_prefill
            for event in profiler.key_averages():
                device_us = float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                )
                if device_us > 0.0:
                    prefill_kernel_profile.append(
                        {
                            "name": event.key,
                            "calls": int(event.count),
                            "self_device_time_us": device_us,
                        }
                    )
            prefill_kernel_profile.sort(
                key=lambda entry: entry["self_device_time_us"], reverse=True
            )
            prefill_kernel_profile = prefill_kernel_profile[:40]

        kernel_profile = []
        if args.profile_decode_kernels:
            if not args.decode_steps:
                raise ValueError("decode kernel profiling requires --decode-steps")
            clear_lod_state(model)
            profile_prefill = prefill()
            profile_cache = profile_prefill.past_key_values
            del profile_prefill
            if args.decode_warmup_steps:
                profile_warmup, profile_cache = decode(
                    profile_cache,
                    args.decode_warmup_steps,
                )
                del profile_warmup
            with torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                )
            ) as profiler:
                profile_decode, profile_cache = decode(
                    profile_cache,
                    args.decode_steps,
                    position_offset=args.decode_warmup_steps,
                )
                torch.cuda.synchronize(device)
            del profile_decode, profile_cache
            for event in profiler.key_averages():
                device_us = float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                )
                if device_us > 0.0:
                    kernel_profile.append(
                        {
                            "name": event.key,
                            "calls": int(event.count),
                            "self_device_time_us": device_us,
                        }
                    )
            kernel_profile.sort(
                key=lambda entry: entry["self_device_time_us"], reverse=True
            )
            kernel_profile = kernel_profile[:40]

    mean_prefill_ms = sum(prefill_elapsed_ms) / len(prefill_elapsed_ms)
    mean_decode_ms = (
        sum(decode_elapsed_ms) / len(decode_elapsed_ms)
        if decode_elapsed_ms
        else None
    )
    mean_decode_step_ms = (
        mean_decode_ms / args.decode_steps if mean_decode_ms is not None else None
    )
    record = {
        "checkpoint": args.checkpoint,
        "qwen35_acceleration": acceleration,
        "mode": args.mode,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "prefill_microbatch_size": args.prefill_microbatch_size,
        "two_level_topk": (
            args.two_level_topk
            if args.mode in ("two_level", "pytorch_lod")
            else None
        ),
        "separate_sink_cache": (
            args.separate_sink_cache if args.mode == "two_level" else None
        ),
        "prefill_two_level_topk": (
            args.prefill_two_level_topk if args.mode == "two_level" else None
        ),
        "prefill_max_leaf_tokens": (
            args.prefill_max_leaf_tokens if args.mode == "two_level" else None
        ),
        "dynamic_open_top_p": (
            args.dynamic_open_top_p if args.mode == "two_level" else None
        ),
        "dynamic_open_prefill_top_p": (
            dynamic_prefill_top_p if args.mode == "two_level" else None
        ),
        "dynamic_open_prefill_residual_mass": (
            args.dynamic_open_prefill_residual_mass
            if args.mode == "two_level"
            else None
        ),
        "dynamic_open_decode_top_p": (
            dynamic_decode_top_p if args.mode == "two_level" else None
        ),
        "dynamic_open_decode_residual_mass": (
            args.dynamic_open_decode_residual_mass
            if args.mode == "two_level"
            else None
        ),
        "reuse_dynamic_local_attention": (
            args.reuse_dynamic_local_attention
            if args.mode == "two_level"
            else None
        ),
        "dynamic_open_residual_state_bound": (
            args.dynamic_open_residual_state_bound
            if args.mode == "two_level"
            else None
        ),
        "recursive_page_lod": (
            args.recursive_page_lod if args.mode == "two_level" else None
        ),
        "all_centroid_top1": (
            args.all_centroid_top1 if args.mode == "two_level" else None
        ),
        "all_centroid_fused": (
            args.all_centroid_fused if args.mode == "two_level" else None
        ),
        "all_centroid_fused_block_m": (
            args.all_centroid_fused_block_m
            if args.mode == "two_level"
            else None
        ),
        "all_centroid_fused_centroids": (
            args.all_centroid_fused_centroids
            if args.mode == "two_level"
            else None
        ),
        "leaf_attention_backend": (
            args.leaf_attention_backend if args.mode == "two_level" else None
        ),
        "virtual_page_storage": (
            args.virtual_page_storage if args.mode == "two_level" else None
        ),
        "leaf_key_quant_bits": (
            args.leaf_key_quant_bits if args.mode == "two_level" else None
        ),
        "leaf_value_quant_bits": (
            args.leaf_value_quant_bits if args.mode == "two_level" else None
        ),
        "leaf_quant_group_size": (
            args.leaf_quant_group_size if args.mode == "two_level" else None
        ),
        "leaf_quant_scale_mode": (
            args.leaf_quant_scale_mode if args.mode == "two_level" else None
        ),
        "leaf_append_quant_scale_mode": (
            args.leaf_append_quant_scale_mode
            if args.mode == "two_level"
            else None
        ),
        "page_summary_quant_bits": (
            args.page_summary_quant_bits if args.mode == "two_level" else None
        ),
        "page_summary_scale_mode": (
            args.page_summary_scale_mode if args.mode == "two_level" else None
        ),
        "recursive_page_block_n": (
            args.recursive_page_block_n if args.mode == "two_level" else None
        ),
        "leaf_num_warps": (
            args.leaf_num_warps if args.mode == "two_level" else None
        ),
        "state_growth_factor": (
            args.state_growth_factor
            if args.mode in ("two_level", "pytorch_lod")
            else None
        ),
        "state_clustering_geometry": (
            args.state_clustering_geometry if args.mode == "two_level" else None
        ),
        "coherence_single_matmul": (
            not args.exact_coherence_matmul if args.mode == "two_level" else None
        ),
        "prefill_chunk_length": (
            256
            if args.mode == "pytorch_lod"
            else effective_prefill_chunk if args.mode == "two_level" else None
        ),
        "prefill_local_length": (
            512
            if args.mode == "pytorch_lod"
            else effective_prefill_local if args.mode == "two_level" else None
        ),
        "prefill_state_update_length": (
            256
            if args.mode == "pytorch_lod"
            else (
                args.prefill_state_update_length
                or Qwen3_5TwoLevelAttention.prefill_state_update_len
            )
            if args.mode == "two_level"
            else None
        ),
        "overflow_bipartite_merge": args.overflow_bipartite_merge,
        "overflow_bipartite_block_size": args.overflow_bipartite_block_size,
        "overflow_bipartite_keep_ratio": args.overflow_bipartite_keep_ratio,
        "merge_before_append": args.merge_before_append,
        "append_subblock_size": args.append_subblock_size,
        "union_bipartite_state": args.union_bipartite_state,
        "state_precompact_direct_append": args.state_precompact_direct_append,
        "leaf_inline_pages_per_slot": (
            args.leaf_inline_pages_per_slot
            if args.leaf_inline_pages_per_slot is not None
            else Qwen3_5TwoLevelAttention.leaf_inline_pages_per_slot
        ),
        "repeats": args.repeats,
        "decode_steps": args.decode_steps,
        "decode_warmup_steps": args.decode_warmup_steps,
        "decode_state_update_length": (
            args.decode_state_update_length if args.mode == "two_level" else None
        ),
        "decode_cache_headroom": (
            args.decode_cache_headroom if args.mode == "two_level" else None
        ),
        "decode_state_update_tokens_per_layer": (
            decode_state_update_tokens_per_layer
            if args.mode == "two_level" and args.decode_steps
            else None
        ),
        "decode_state_update_intervals_per_layer": (
            decode_state_update_intervals_per_layer
            if args.mode == "two_level" and args.decode_steps
            else None
        ),
        "fused_decode_attention": (
            not args.disable_fused_decode if args.mode == "two_level" else None
        ),
        "clone_decode_routes": (
            not args.no_clone_decode_routes if args.mode == "two_level" else None
        ),
        "fused_decode_state_route": (
            not args.disable_fused_decode_state_route
            if args.mode == "two_level"
            else None
        ),
        "decode_route_group_size": (
            args.decode_route_group_size if args.mode == "two_level" else None
        ),
        "decode_route_num_warps": (
            args.decode_route_num_warps if args.mode == "two_level" else None
        ),
        "decode_route_reduce_num_warps": (
            args.decode_route_reduce_num_warps
            if args.mode == "two_level"
            else None
        ),
        "decode_final_reduce_num_warps": (
            args.decode_final_reduce_num_warps
            if args.mode == "two_level"
            else None
        ),
        "decode_split_kv": (
            (
                args.decode_split_kv
                if args.decode_split_kv is not None
                else Qwen3_5TwoLevelAttention.decode_split_kv
            )
            if args.mode == "two_level"
            else None
        ),
        "decode_fuse_final_reduce": (
            args.decode_fuse_final_reduce if args.mode == "two_level" else None
        ),
        "decode_route_use_dot": (
            (
                args.decode_route_use_dot
                if args.decode_route_use_dot is not None
                else Qwen3_5TwoLevelAttention.decode_route_use_dot
            )
            if args.mode == "two_level"
            else None
        ),
        "decode_use_dot": (
            args.decode_use_dot if args.mode == "two_level" else None
        ),
        "decode_route_gqa_grouped": (
            (
                args.decode_route_gqa_grouped
                if args.decode_route_gqa_grouped is not None
                else Qwen3_5TwoLevelAttention.decode_route_gqa_grouped
            )
            if args.mode == "two_level"
            else None
        ),
        "fused_state_update": (
            bool(args.enable_fused_state_update)
            if args.mode == "two_level"
            else None
        ),
        "fused_state_maxsim": (
            args.fused_state_maxsim if args.mode == "two_level" else None
        ),
        "state_maxsim_block_m": (
            args.state_maxsim_block_m if args.mode == "two_level" else None
        ),
        "state_maxsim_block_n": (
            args.state_maxsim_block_n if args.mode == "two_level" else None
        ),
        "state_maxsim_num_warps": (
            args.state_maxsim_num_warps if args.mode == "two_level" else None
        ),
        "coarse_enable_gqa": (
            not args.disable_coarse_gqa if args.mode == "two_level" else None
        ),
        "coarse_compact_bias": (
            not args.disable_compact_coarse_bias
            if args.mode == "two_level"
            else None
        ),
        "reuse_route_logits_for_coarse": (
            (
                args.reuse_route_logits
                if args.reuse_route_logits is not None
                else Qwen3_5TwoLevelAttention.reuse_route_logits_for_coarse
            )
            if args.mode == "two_level"
            else None
        ),
        "fused_prefill_route_coarse": (
            args.enable_fused_prefill_route_coarse
            if args.mode == "two_level"
            else None
        ),
        "split_prefill_local_attention": (
            args.split_prefill_local_attention
            if args.mode == "two_level"
            else None
        ),
        "fused_prefill_residual_opening": (
            args.fused_prefill_residual_opening
            if args.mode == "two_level"
            else None
        ),
        "coarse_route_block_m": (
            args.coarse_route_block_m if args.mode == "two_level" else None
        ),
        "coarse_route_block_n": (
            args.coarse_route_block_n if args.mode == "two_level" else None
        ),
        "coarse_route_num_warps": (
            args.coarse_route_num_warps if args.mode == "two_level" else None
        ),
        "route_gqa_matmul": (
            args.route_gqa_matmul if args.mode == "two_level" else None
        ),
        "prefill_elapsed_ms": prefill_elapsed_ms,
        "mean_prefill_ms": mean_prefill_ms,
        "prefill_tokens_per_second": (
            args.batch_size * args.sequence_length / (mean_prefill_ms / 1000.0)
        ),
        "decode_elapsed_ms": decode_elapsed_ms,
        "mean_decode_ms": mean_decode_ms,
        "mean_decode_step_ms": mean_decode_step_ms,
        "decode_tokens_per_second": (
            args.batch_size / (mean_decode_step_ms / 1000.0)
            if mean_decode_step_ms is not None
            else None
        ),
        "decode_kernel_profile": kernel_profile,
        "prefill_kernel_profile": prefill_kernel_profile,
        "cache_memory_gib": cache_memory_gib,
        "cache_statistics": cache_statistics,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "logit_finite": finite,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
