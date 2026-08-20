#!/usr/bin/env python3
"""Measure whether two-tier LOD routes form useful GQA attention experts.

The proposed INT8-MMA layout groups decode queries by
``(request, KV head, routed centroid)``.  Queries in one such group share the
same exact leaf list, so their QK and PV products can be evaluated as a small
matrix multiplication rather than one vector product per query.  This script
captures real decode routes and reports the resulting M and N occupancy.
"""

from __future__ import annotations

import argparse
import json
import math
import types
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.probe_qwen35_lod_niah import (
    load_text_model,
    require_qwen35_acceleration,
)
from scripts.profile_qwen35_prefill_total import select_profile_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--leaf-layout", choices=("query", "expert"), default="query")
    parser.add_argument("--leaf-inline-pages-per-slot", type=int, default=128)
    parser.add_argument("--mma-block-m", type=int, default=16)
    parser.add_argument("--mma-block-n", type=int, default=32)
    parser.add_argument(
        "--repeat-one-sequence",
        action="store_true",
        help="repeat one prompt across the batch instead of using distinct prompts",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor,
        torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90), dtype=torch.float64),
    ).tolist()
    return {
        "min": float(tensor.min().item()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "max": float(tensor.max().item()),
    }


def _summarize_records(
    top_slots: list[torch.Tensor],
    slot_lengths: torch.Tensor,
    *,
    mma_block_m: int,
    mma_block_n: int,
) -> dict[str, Any]:
    """Aggregate experts across decode steps for one attention layer."""
    if not top_slots:
        raise RuntimeError("no decode routes were captured")
    if slot_lengths.ndim != 3:
        raise ValueError(f"unexpected slot-length shape {slot_lengths.shape}")

    batch, kv_heads, _ = slot_lengths.shape
    query_heads = int(top_slots[0].size(1))
    if query_heads % kv_heads:
        raise ValueError(
            f"query heads {query_heads} are not divisible by KV heads {kv_heads}"
        )
    gqa_group_size = query_heads // kv_heads

    multiplicity_histogram: Counter[int] = Counter()
    expert_leaf_counts: list[int] = []
    pair_leaf_counts: list[int] = []
    unique_experts_per_group: list[int] = []
    useful_interactions = 0
    expert_leaf_reads = 0
    padded_m_interactions = 0
    padded_mn_interactions = 0
    total_route_pairs = 0
    total_query_rows = 0

    for routed in top_slots:
        if routed.ndim != 4 or int(routed.size(2)) != 1:
            raise ValueError(f"unexpected decode route shape {routed.shape}")
        if int(routed.size(0)) != batch or int(routed.size(1)) != query_heads:
            raise ValueError("decode route shape changed across records")
        total_query_rows += batch * query_heads
        routed = routed[..., 0, :]
        for batch_idx in range(batch):
            for kv_head in range(kv_heads):
                first_query_head = kv_head * gqa_group_size
                group_routes = routed[
                    batch_idx,
                    first_query_head : first_query_head + gqa_group_size,
                ].reshape(-1)
                group_routes = group_routes[group_routes.ge(0)]
                if not int(group_routes.numel()):
                    unique_experts_per_group.append(0)
                    continue
                unique_slots, multiplicities = torch.unique(
                    group_routes, sorted=False, return_counts=True
                )
                lengths = slot_lengths[batch_idx, kv_head].index_select(
                    0, unique_slots.to(torch.long)
                )
                unique_experts_per_group.append(int(unique_slots.numel()))
                total_route_pairs += int(multiplicities.sum().item())
                for multiplicity, leaf_count in zip(
                    multiplicities.tolist(), lengths.tolist()
                ):
                    multiplicity = int(multiplicity)
                    leaf_count = int(leaf_count)
                    multiplicity_histogram[multiplicity] += 1
                    expert_leaf_counts.append(leaf_count)
                    pair_leaf_counts.extend([leaf_count] * multiplicity)
                    useful_interactions += multiplicity * leaf_count
                    expert_leaf_reads += leaf_count
                    padded_m_interactions += (
                        math.ceil(multiplicity / mma_block_m)
                        * mma_block_m
                        * leaf_count
                    )
                    padded_mn_interactions += (
                        math.ceil(multiplicity / mma_block_m)
                        * mma_block_m
                        * math.ceil(leaf_count / mma_block_n)
                        * mma_block_n
                    )

    expert_count = sum(multiplicity_histogram.values())
    if total_route_pairs != sum(
        multiplicity * count
        for multiplicity, count in multiplicity_histogram.items()
    ):
        raise AssertionError("route-pair accounting is inconsistent")

    def expert_fraction_at_least(threshold: int) -> float:
        return _ratio(
            sum(
                count
                for multiplicity, count in multiplicity_histogram.items()
                if multiplicity >= threshold
            ),
            expert_count,
        )

    def pair_fraction_at_least(threshold: int) -> float:
        return _ratio(
            sum(
                multiplicity * count
                for multiplicity, count in multiplicity_histogram.items()
                if multiplicity >= threshold
            ),
            total_route_pairs,
        )

    selected_leaf_tokens_per_query = _ratio(useful_interactions, total_query_rows)
    return {
        "decode_events": len(top_slots),
        "batch_size": batch,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "gqa_group_size": gqa_group_size,
        "route_pairs": total_route_pairs,
        "query_rows": total_query_rows,
        "experts": expert_count,
        "mean_query_multiplicity_per_expert": _ratio(total_route_pairs, expert_count),
        "multiplicity_histogram": {
            str(key): multiplicity_histogram[key]
            for key in sorted(multiplicity_histogram)
        },
        "expert_fraction_m_at_least_2": expert_fraction_at_least(2),
        "expert_fraction_m_at_least_4": expert_fraction_at_least(4),
        "expert_fraction_m_at_least_8": expert_fraction_at_least(8),
        "route_pair_fraction_m_at_least_2": pair_fraction_at_least(2),
        "route_pair_fraction_m_at_least_4": pair_fraction_at_least(4),
        "route_pair_fraction_m_at_least_8": pair_fraction_at_least(8),
        "unique_experts_per_request_kv_head": _percentiles(
            unique_experts_per_group
        ),
        "leaf_tokens_per_expert": _percentiles(expert_leaf_counts),
        "leaf_tokens_per_route_pair": _percentiles(pair_leaf_counts),
        "mean_selected_leaf_tokens_per_query": selected_leaf_tokens_per_query,
        "three_tier_fixed_leaf_tokens_per_query": 128,
        "two_tier_vs_three_tier_leaf_interaction_ratio": (
            selected_leaf_tokens_per_query / 128.0
        ),
        "ideal_expert_leaf_reuse_factor": _ratio(
            useful_interactions, expert_leaf_reads
        ),
        "ideal_expert_leaf_read_reduction": 1.0
        - _ratio(expert_leaf_reads, useful_interactions),
        "int8_mma_block_m": mma_block_m,
        "int8_mma_block_n": mma_block_n,
        "int8_mma_m_only_useful_fraction": _ratio(
            useful_interactions, padded_m_interactions
        ),
        "int8_mma_mn_useful_fraction": _ratio(
            useful_interactions, padded_mn_interactions
        ),
        "useful_qk_token_interactions": useful_interactions,
        "expert_leaf_rows": expert_leaf_reads,
        "padded_m_interactions": padded_m_interactions,
        "padded_mn_interactions": padded_mn_interactions,
    }


def _combine_layer_records(layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine raw counters from per-layer summaries without losing weights."""
    histogram: Counter[int] = Counter()
    for layer in layers:
        histogram.update(
            {int(key): int(value) for key, value in layer["multiplicity_histogram"].items()}
        )
    route_pairs = sum(int(layer["route_pairs"]) for layer in layers)
    experts = sum(int(layer["experts"]) for layer in layers)
    useful = sum(int(layer["useful_qk_token_interactions"]) for layer in layers)
    leaf_rows = sum(int(layer["expert_leaf_rows"]) for layer in layers)
    padded_m = sum(int(layer["padded_m_interactions"]) for layer in layers)
    padded_mn = sum(int(layer["padded_mn_interactions"]) for layer in layers)

    def expert_fraction(threshold: int) -> float:
        return _ratio(
            sum(count for mult, count in histogram.items() if mult >= threshold),
            experts,
        )

    def pair_fraction(threshold: int) -> float:
        return _ratio(
            sum(mult * count for mult, count in histogram.items() if mult >= threshold),
            route_pairs,
        )

    query_rows = sum(int(layer["query_rows"]) for layer in layers)
    selected_per_query = _ratio(useful, query_rows)
    return {
        "layers": len(layers),
        "route_pairs": route_pairs,
        "query_rows": query_rows,
        "experts": experts,
        "mean_query_multiplicity_per_expert": _ratio(route_pairs, experts),
        "multiplicity_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "expert_fraction_m_at_least_2": expert_fraction(2),
        "expert_fraction_m_at_least_4": expert_fraction(4),
        "expert_fraction_m_at_least_8": expert_fraction(8),
        "route_pair_fraction_m_at_least_2": pair_fraction(2),
        "route_pair_fraction_m_at_least_4": pair_fraction(4),
        "route_pair_fraction_m_at_least_8": pair_fraction(8),
        "mean_selected_leaf_tokens_per_query": selected_per_query,
        "three_tier_fixed_leaf_tokens_per_query": 128,
        "two_tier_vs_three_tier_leaf_interaction_ratio": selected_per_query / 128.0,
        "ideal_expert_leaf_reuse_factor": _ratio(useful, leaf_rows),
        "ideal_expert_leaf_read_reduction": 1.0 - _ratio(leaf_rows, useful),
        "int8_mma_m_only_useful_fraction": _ratio(useful, padded_m),
        "int8_mma_mn_useful_fraction": _ratio(useful, padded_mn),
        "useful_qk_token_interactions": useful,
        "expert_leaf_rows": leaf_rows,
        "padded_m_interactions": padded_m,
        "padded_mn_interactions": padded_mn,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("batch size and decode steps must be positive")
    if args.two_level_topk <= 0 or args.two_level_topk > 8:
        raise ValueError("two-level top-k must be in [1, 8]")
    if args.mma_block_m <= 0 or args.mma_block_n <= 0:
        raise ValueError("MMA block sizes must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if args.repeat_one_sequence:
        sequence = select_profile_sequence(
            tokenizer, args.dataset, args.sequence_length
        ).unsqueeze(0).expand(args.batch_size, -1)
    else:
        selected = select_sequences(
            tokenizer,
            args.dataset,
            args.sequence_length,
            args.batch_size,
            0,
            1,
        )
        sequence = torch.stack([sample[1] for sample in selected])
    sequence = sequence.contiguous().to(device)

    model = load_text_model(
        args.checkpoint,
        "two_level",
        args.two_level_topk,
        args.state_growth_factor,
        device,
        "paged",
        require_fla_fast_path=True,
    )
    acceleration = require_qwen35_acceleration(model)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    if not modules:
        raise RuntimeError("Qwen3.5 LOD attention modules were not installed")
    for module in modules:
        module.leaf_layout = args.leaf_layout
        module.leaf_inline_pages_per_slot = args.leaf_inline_pages_per_slot
        # The production fused kernel computes routes internally.  Disabling
        # only that fusion exposes the identical route operation without
        # changing the state, page archive, or attention result.
        module.fused_decode_state_route = False

    records: list[dict[str, Any]] = [
        {"top_slots": [], "slot_lengths": None} for _ in modules
    ]
    with torch.inference_mode():
        prefill = model(input_ids=sequence, use_cache=True, logits_to_keep=1)
        cache = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for layer_index, module in enumerate(modules):
            state = getattr(module, "_lod_state", None)
            page_cache = state.get("page_cache") if isinstance(state, dict) else None
            slot_lengths = (
                page_cache.get("slot_lengths")
                if isinstance(page_cache, dict)
                else None
            )
            if not isinstance(slot_lengths, torch.Tensor):
                raise RuntimeError(f"layer {layer_index} has no paged slot lengths")
            records[layer_index]["slot_lengths"] = slot_lengths.detach().clone()
            original = module._route_top_slots

            def captured_route(
                self,
                *method_args,
                __original=original,
                __layer_index=layer_index,
                **method_kwargs,
            ):
                routed = __original(*method_args, **method_kwargs)
                if int(method_args[0].size(2)) == 1:
                    records[__layer_index]["top_slots"].append(
                        routed.detach().clone()
                    )
                return routed

            module._route_top_slots = types.MethodType(captured_route, module)

        position = args.sequence_length
        output = None
        for _ in range(args.steps):
            output = model(
                input_ids=next_token,
                past_key_values=cache,
                cache_position=torch.tensor(
                    [position], dtype=torch.long, device=device
                ),
                use_cache=True,
                logits_to_keep=1,
            )
            cache = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            position += 1
        torch.cuda.synchronize(device)
        if output is None or not bool(torch.isfinite(output.logits).all().item()):
            raise RuntimeError("decode did not produce finite logits")

    layers = []
    for layer_index, record in enumerate(records):
        tops = [tensor.cpu() for tensor in record["top_slots"]]
        lengths = record["slot_lengths"]
        if not isinstance(lengths, torch.Tensor):
            raise AssertionError("missing slot-length snapshot")
        layer_summary = _summarize_records(
            tops,
            lengths.cpu(),
            mma_block_m=args.mma_block_m,
            mma_block_n=args.mma_block_n,
        )
        layer_summary["layer_index"] = layer_index
        layer_summary["model_layer_index"] = int(modules[layer_index].layer_idx)
        layers.append(layer_summary)

    result = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "decode_steps": args.steps,
        "two_level_topk": args.two_level_topk,
        "state_growth_factor": args.state_growth_factor,
        "repeat_one_sequence": args.repeat_one_sequence,
        "mma_block_m": args.mma_block_m,
        "mma_block_n": args.mma_block_n,
        "qwen35_acceleration": acceleration,
        "aggregate": _combine_layer_records(layers),
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
