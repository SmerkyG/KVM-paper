#!/usr/bin/env python3
"""Break the paged exact-leaf branch into dispatch, kernel, and reduction."""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.probe_qwen35_lod_niah import load_text_model


def select_profile_sequence(
    tokenizer, dataset: str, sequence_length: int
) -> torch.Tensor:
    """Construct contexts longer than the source documents from 64K pieces."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--state-growth-factor", type=float, default=8.0)
    parser.add_argument("--prefill-two-level-topk", type=int, default=3)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--tiny-expert-max", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--tiny-max-context", type=int, default=65_536)
    parser.add_argument("--tiny-block-m", type=int, default=8)
    parser.add_argument("--tiny-num-warps", type=int, default=1)
    parser.add_argument("--long-expert-threshold", type=int, default=0)
    parser.add_argument(
        "--long-expert-splits", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument("--reduce-num-warps", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument(
        "--layout",
        choices=(
            "expert",
            "expert_tiny",
            "query",
            "aiter_varlen",
            "aiter_copy",
            "aiter_union",
            "aiter_masked_union",
        ),
        default="expert",
    )
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument("--aiter-copy-page-size", type=int, default=16)
    parser.add_argument(
        "--leaf-union-query-tile", type=int, choices=(2, 4, 8, 16, 32), default=8
    )
    parser.add_argument("--collect-occupancy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def install_occupancy_collector(
    modules: list[Qwen3_5TwoLevelAttention],
    *,
    block_m: int,
    block_n: int,
) -> dict[str, int]:
    totals = defaultdict(int)
    for module in modules:
        original = module._paged_leaf_attention

        def measured(self, q, top_slots, cache, __original=original):
            batch, query_heads, query_len, _ = q.shape
            route_count = int(top_slots.size(-1))
            key_cache = cache.get("page_k", cache.get("leaf_k"))
            if not isinstance(key_cache, torch.Tensor):
                raise TypeError("paged leaf cache has no K storage")
            kv_heads = int(key_cache.size(1))
            state_capacity = int(cache["slot_lengths"].size(-1))
            rows = batch * query_heads * query_len
            query_row = torch.arange(rows, device=q.device, dtype=torch.long)
            batch_head = torch.div(query_row, query_len, rounding_mode="floor")
            batch_for_row = torch.div(
                batch_head, query_heads, rounding_mode="floor"
            )
            query_head = batch_head % query_heads
            kv_group_size = query_heads // kv_heads
            kv_row = batch_for_row * kv_heads + torch.div(
                query_head, kv_group_size, rounding_mode="floor"
            )
            expert_id = (
                kv_row.unsqueeze(-1) * state_capacity
                + top_slots.reshape(rows, route_count)
            ).reshape(-1)
            unique_expert, query_counts = torch.unique(
                expert_id, return_counts=True
            )
            key_counts = cache["slot_lengths"].reshape(-1)[unique_expert]
            query_blocks = torch.div(
                query_counts + block_m - 1,
                block_m,
                rounding_mode="floor",
            )
            key_blocks = torch.div(
                key_counts + block_n - 1,
                block_n,
                rounding_mode="floor",
            )
            legacy_query_blocks = torch.div(
                query_counts.max() + block_m - 1,
                block_m,
                rounding_mode="floor",
            )
            totals["calls"] += 1
            totals["routes"] += int(query_counts.sum().item())
            totals["active_experts"] += int(query_counts.numel())
            totals["max_routes_per_expert"] = max(
                totals["max_routes_per_expert"], int(query_counts.max().item())
            )
            totals["max_leaves_per_expert"] = max(
                totals["max_leaves_per_expert"], int(key_counts.max().item())
            )
            totals["ragged_programs"] += int(query_blocks.sum().item())
            totals["legacy_programs"] += int(
                legacy_query_blocks.item() * query_counts.numel()
            )
            totals["useful_qk_pairs"] += int(
                (query_counts * key_counts).sum().item()
            )
            totals["ragged_padded_qk_pairs"] += int(
                (query_blocks * block_m * key_blocks * block_n).sum().item()
            )
            totals["legacy_padded_qk_pairs"] += int(
                (
                    legacy_query_blocks
                    * block_m
                    * key_blocks
                    * block_n
                ).sum().item()
            )
            for limit in (
                1,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
            ):
                totals[f"experts_m_le_{limit}"] += int(
                    (query_counts <= limit).sum().item()
                )
                totals[f"experts_k_le_{limit}"] += int(
                    (key_counts <= limit).sum().item()
                )
                totals[f"programs_k_le_{limit}"] += int(
                    query_blocks[key_counts <= limit].sum().item()
                )
            return __original(q, top_slots, cache)

        module._paged_leaf_attention = types.MethodType(measured, module)
    return totals


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = load_text_model(
        args.checkpoint,
        "two_level",
        8,
        args.state_growth_factor,
        device,
        "paged",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    sequence = (
        select_profile_sequence(tokenizer, args.dataset, args.sequence_length)
        .unsqueeze(0)
        .expand(args.batch_size, -1)
        .contiguous()
        .to(device)
    )
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    for module in modules:
        module.prefill_two_level_topk = args.prefill_two_level_topk
        module.leaf_block_m = args.block_m
        module.leaf_block_n = args.block_n
        module.leaf_num_warps = args.num_warps
        module.leaf_tiny_expert_max = args.tiny_expert_max
        module.leaf_tiny_max_context = args.tiny_max_context
        module.leaf_tiny_block_m = args.tiny_block_m
        module.leaf_tiny_num_warps = args.tiny_num_warps
        module.leaf_long_expert_threshold = args.long_expert_threshold
        module.leaf_long_expert_splits = args.long_expert_splits
        module.leaf_reduce_num_warps = args.reduce_num_warps
        module.leaf_layout = args.layout
        module.leaf_union_query_tile = args.leaf_union_query_tile
        module.leaf_aiter_copy_page_size = args.aiter_copy_page_size
        module.virtual_page_storage = args.virtual_page_storage

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
        for module in modules:
            if hasattr(module, "_lod_state"):
                delattr(module, "_lod_state")

        occupancy = (
            install_occupancy_collector(
                modules,
                block_m=args.block_m,
                block_n=args.block_n,
            )
            if args.collect_occupancy
            else None
        )
        events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = defaultdict(list)
        for module in modules:
            module._lod_leaf_timing_events = events
        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)

    phase_ms = {
        phase: sum(float(begin.elapsed_time(end)) for begin, end in pairs)
        for phase, pairs in events.items()
    }
    phase_call_ms = {
        phase: [float(begin.elapsed_time(end)) for begin, end in pairs]
        for phase, pairs in events.items()
    }
    total_ms = phase_ms.pop("total")
    total_call_ms = phase_call_ms.pop("total")
    record = {
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "attention_layers": len(modules),
        "prefill_two_level_topk": args.prefill_two_level_topk,
        "block_m": args.block_m,
        "block_n": args.block_n,
        "num_warps": args.num_warps,
        "tiny_expert_max": (
            args.tiny_expert_max if args.layout == "expert_tiny" else None
        ),
        "tiny_max_context": (
            args.tiny_max_context if args.layout == "expert_tiny" else None
        ),
        "tiny_block_m": args.tiny_block_m if args.layout == "expert_tiny" else None,
        "tiny_num_warps": (
            args.tiny_num_warps if args.layout == "expert_tiny" else None
        ),
        "long_expert_threshold": args.long_expert_threshold,
        "long_expert_splits": args.long_expert_splits,
        "reduce_num_warps": args.reduce_num_warps,
        "layout": args.layout,
        "leaf_union_query_tile": (
            args.leaf_union_query_tile
            if args.layout in ("aiter_union", "aiter_masked_union")
            else None
        ),
        "virtual_page_storage": args.virtual_page_storage,
        "aiter_copy_page_size": (
            args.aiter_copy_page_size if args.layout == "aiter_copy" else None
        ),
        "exact_leaf_total_ms": total_ms,
        "phase_ms": phase_ms,
        "phase_call_ms": phase_call_ms,
        "exact_leaf_call_ms": total_call_ms,
        "phase_fraction": {
            phase: milliseconds / total_ms
            for phase, milliseconds in phase_ms.items()
        },
        "calls": len(events["total"]),
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    if occupancy is not None:
        active_experts = occupancy["active_experts"]
        record["expert_occupancy"] = {
            **occupancy,
            "mean_routes_per_expert": occupancy["routes"] / active_experts,
            "ragged_program_fraction_of_legacy": (
                occupancy["ragged_programs"] / occupancy["legacy_programs"]
            ),
            "ragged_qk_tile_utilization": (
                occupancy["useful_qk_pairs"]
                / occupancy["ragged_padded_qk_pairs"]
            ),
            "legacy_qk_tile_utilization": (
                occupancy["useful_qk_pairs"]
                / occupancy["legacy_padded_qk_pairs"]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
