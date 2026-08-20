#!/usr/bin/env python3
"""Attribute warm Qwen LOD prefill time to its attention phases."""

from __future__ import annotations

import argparse
import json
import types
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.probe_qwen35_lod_niah import load_text_model
from scripts.profile_qwen35_prefill_total import select_profile_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--prefill-two-level-topk", type=int, default=3)
    parser.add_argument("--prefill-max-leaf-tokens", type=int)
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument("--recursive-page-block-n", type=int, default=4)
    parser.add_argument(
        "--leaf-layout",
        choices=(
            "expert",
            "query",
            "aiter_varlen",
            "aiter_union",
            "aiter_masked_union",
        ),
        default="query",
    )
    parser.add_argument(
        "--leaf-union-query-tile", type=int, choices=(2, 4, 8, 16, 32), default=8
    )
    parser.add_argument("--leaf-block-m", type=int, default=16)
    parser.add_argument("--leaf-block-n", type=int, default=32)
    parser.add_argument("--leaf-num-warps", type=int, default=1)
    parser.add_argument("--page-summary-quant-bits", type=int, choices=(0, 8), default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument("--prefill-local-length", type=int)
    parser.add_argument("--prefill-state-update-length", type=int)
    parser.add_argument("--overflow-bipartite-merge", action="store_true")
    parser.add_argument("--overflow-bipartite-block-size", type=int, default=32)
    parser.add_argument("--merge-before-append", action="store_true")
    parser.add_argument("--append-subblock-size", type=int, default=0)
    parser.add_argument("--union-bipartite-state", action="store_true")
    parser.add_argument(
        "--split-prefill-local-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dynamic-open-prefill-residual-mass", type=float)
    parser.add_argument("--fused-prefill-residual-opening", action="store_true")
    parser.add_argument(
        "--fused-prefill-route-coarse",
        "--enable-fused-prefill-route-coarse",
        dest="enable_fused_prefill_route_coarse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--disable-fused-state-update", action="store_true")
    parser.add_argument("--enable-fused-state-update", action="store_true")
    parser.add_argument("--disable-fused-state-routing", action="store_true")
    parser.add_argument(
        "--direct-fused-state-routing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--disable-shared-update-similarity", action="store_true")
    parser.add_argument("--disable-coarse-gqa", action="store_true")
    parser.add_argument("--disable-compact-coarse-bias", action="store_true")
    parser.add_argument("--reuse-route-logits", action="store_true")
    parser.add_argument("--coarse-route-block-m", type=int, default=16)
    parser.add_argument("--coarse-route-block-n", type=int, default=32)
    parser.add_argument("--coarse-route-num-warps", type=int, default=4)
    parser.add_argument("--route-gqa-matmul", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def attention_modules(model: torch.nn.Module) -> list[Qwen3_5TwoLevelAttention]:
    return [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]


def install_timers(
    modules: list[Qwen3_5TwoLevelAttention],
) -> dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]:
    pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(
        list
    )
    phases = {
        "route": "_route_top_slots",
        "exact_leaf": "_paged_leaf_attention",
        "coarse": "_coarse_attention",
        "state_update": "_update_state",
        "page_append": "_append_page_cache",
        "local": "_prefill_local_attention",
    }
    for module in modules:
        if module.recursive_page_lod:
            module._lod_leaf_timing_events = {"total": pairs["exact_leaf"]}
        for phase, method_name in phases.items():
            original = getattr(module, method_name)

            def timed(self, *args, __original=original, __phase=phase, **kwargs):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = __original(*args, **kwargs)
                end.record()
                pairs[__phase].append((begin, end))
                return result

            setattr(module, method_name, types.MethodType(timed, module))
    return pairs


def main() -> None:
    args = parse_args()
    if args.prefill_two_level_topk is not None and not (
        0 <= args.prefill_two_level_topk <= 8
    ):
        raise ValueError("prefill top-k must be in [0, 8]")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_text_model(
        args.checkpoint,
        "two_level",
        args.two_level_topk,
        args.state_growth_factor,
        device,
        "paged",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sequence = (
        select_profile_sequence(tokenizer, args.dataset, args.sequence_length)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
        .to(device)
    )
    modules = attention_modules(model)
    if not modules or not all(
        isinstance(module, Qwen3_5Attention) for module in modules
    ):
        raise RuntimeError("Qwen LOD attention modules were not installed")
    if args.disable_fused_state_update and args.enable_fused_state_update:
        raise ValueError("state update cannot be both enabled and disabled")
    for module in modules:
        module.prefill_two_level_topk = args.prefill_two_level_topk
        module.prefill_max_leaf_tokens = args.prefill_max_leaf_tokens
        module.recursive_page_lod = args.recursive_page_lod
        module.virtual_page_storage = args.virtual_page_storage
        module.recursive_page_block_n = args.recursive_page_block_n
        module.leaf_layout = args.leaf_layout
        module.leaf_union_query_tile = args.leaf_union_query_tile
        module.leaf_block_m = args.leaf_block_m
        module.leaf_block_n = args.leaf_block_n
        module.leaf_num_warps = args.leaf_num_warps
        module.page_summary_quant_bits = args.page_summary_quant_bits
        if args.prefill_chunk_length is not None:
            module.prefill_chunk_len = args.prefill_chunk_length
        if args.prefill_local_length is not None:
            module.prefill_local_len = args.prefill_local_length
        if args.prefill_state_update_length is not None:
            module.prefill_state_update_len = args.prefill_state_update_length
        module.overflow_bipartite_merge = args.overflow_bipartite_merge
        module.overflow_bipartite_block_size = args.overflow_bipartite_block_size
        module.state_merge_before_append = args.merge_before_append
        module.state_append_subblock_size = args.append_subblock_size
        module.state_union_bipartite = args.union_bipartite_state
        module.split_prefill_local_attention = args.split_prefill_local_attention
        module.dynamic_open_prefill_residual_mass = (
            args.dynamic_open_prefill_residual_mass
        )
        module.fused_prefill_residual_opening = (
            args.fused_prefill_residual_opening
        )
        module.fused_prefill_route_coarse = args.enable_fused_prefill_route_coarse
        if args.disable_fused_state_update:
            module.fused_state_update = False
            module.auto_fused_state_update = False
        elif args.enable_fused_state_update:
            module.fused_state_update = True
        module.fused_state_routing = not args.disable_fused_state_routing
        if args.direct_fused_state_routing is not None:
            module.direct_fused_state_routing = args.direct_fused_state_routing
        module.reuse_state_update_similarity = not args.disable_shared_update_similarity
        module.coarse_enable_gqa = not args.disable_coarse_gqa
        module.coarse_compact_bias = not args.disable_compact_coarse_bias
        module.reuse_route_logits_for_coarse = args.reuse_route_logits
        module.coarse_route_block_m = args.coarse_route_block_m
        module.coarse_route_block_n = args.coarse_route_block_n
        module.coarse_route_num_warps = args.coarse_route_num_warps
        module.route_gqa_matmul = args.route_gqa_matmul

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
        for module in modules:
            if hasattr(module, "_lod_state"):
                delattr(module, "_lod_state")
        torch.cuda.empty_cache()

        pairs = install_timers(modules)
        totals = []
        result = None
        for _ in range(args.repeats):
            for module in modules:
                if hasattr(module, "_lod_state"):
                    delattr(module, "_lod_state")
            total_begin = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_begin.record()
            result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
            total_end.record()
            totals.append((total_begin, total_end))
        torch.cuda.synchronize(device)

    if result is None:
        raise ValueError("profile repeats must be positive")
    total_ms = (
        sum(float(begin.elapsed_time(end)) for begin, end in totals) / args.repeats
    )
    phase_ms = {
        phase: sum(float(begin.elapsed_time(end)) for begin, end in events)
        / args.repeats
        for phase, events in pairs.items()
    }
    measured_ms = sum(phase_ms.values())
    record = {
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "state_growth_factor": args.state_growth_factor,
        "prefill_chunk_length": args.prefill_chunk_length,
        "prefill_local_length": args.prefill_local_length,
        "prefill_state_update_length": args.prefill_state_update_length,
        "overflow_bipartite_merge": args.overflow_bipartite_merge,
        "overflow_bipartite_block_size": args.overflow_bipartite_block_size,
        "merge_before_append": args.merge_before_append,
        "append_subblock_size": args.append_subblock_size,
        "union_bipartite_state": args.union_bipartite_state,
        "split_prefill_local_attention": args.split_prefill_local_attention,
        "dynamic_open_prefill_residual_mass": (
            args.dynamic_open_prefill_residual_mass
        ),
        "fused_prefill_residual_opening": args.fused_prefill_residual_opening,
        "fused_prefill_route_coarse": args.enable_fused_prefill_route_coarse,
        "two_level_topk": args.two_level_topk,
        "prefill_two_level_topk": args.prefill_two_level_topk,
        "prefill_max_leaf_tokens": args.prefill_max_leaf_tokens,
        "recursive_page_lod": args.recursive_page_lod,
        "virtual_page_storage": args.virtual_page_storage,
        "recursive_page_block_n": args.recursive_page_block_n,
        "leaf_layout": args.leaf_layout,
        "leaf_union_query_tile": (
            args.leaf_union_query_tile
            if args.leaf_layout in ("aiter_union", "aiter_masked_union")
            else None
        ),
        "leaf_block_m": args.leaf_block_m,
        "leaf_block_n": args.leaf_block_n,
        "leaf_num_warps": args.leaf_num_warps,
        "page_summary_quant_bits": args.page_summary_quant_bits,
        "fused_state_update": bool(modules[0].fused_state_update),
        "auto_fused_state_update": bool(modules[0].auto_fused_state_update),
        "fused_state_routing": not args.disable_fused_state_routing,
        "direct_fused_state_routing": bool(modules[0].direct_fused_state_routing),
        "reuse_state_update_similarity": not args.disable_shared_update_similarity,
        "coarse_enable_gqa": not args.disable_coarse_gqa,
        "coarse_compact_bias": not args.disable_compact_coarse_bias,
        "reuse_route_logits_for_coarse": args.reuse_route_logits,
        "coarse_route_block_m": args.coarse_route_block_m,
        "coarse_route_block_n": args.coarse_route_block_n,
        "coarse_route_num_warps": args.coarse_route_num_warps,
        "route_gqa_matmul": args.route_gqa_matmul,
        "repeats": args.repeats,
        "attention_layers": len(modules),
        "total_ms": total_ms,
        "phase_ms": phase_ms,
        "phase_fraction": {
            phase: milliseconds / total_ms for phase, milliseconds in phase_ms.items()
        },
        "other_ms": total_ms - measured_ms,
        "other_fraction": (total_ms - measured_ms) / total_ms,
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
