#!/usr/bin/env python3
"""Profile routed leaf-set sizes and imbalance in Qwen3.5 LOD attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.probe_qwen35_lod_niah import load_text_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--state-growth-factor", type=float, default=8.0)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    quantiles = torch.quantile(
        values, torch.tensor((0.5, 0.9, 0.95, 0.99), dtype=torch.float32)
    )
    return {
        "mean": float(values.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "max": float(values.max().item()),
    }


def page_layout(counts: torch.Tensor) -> dict[str, dict[str, float | int]]:
    counts = counts.round().to(torch.long).flatten()
    result = {}
    for page_size in (8, 16, 32, 64):
        pages_per_slot = torch.div(
            counts + page_size - 1, page_size, rounding_mode="floor"
        )
        pages = int(pages_per_slot.sum().item())
        capacity = pages * page_size
        result[str(page_size)] = {
            "pages": pages,
            "utilization": float(counts.sum().item()) / float(capacity),
            "max_pages_per_slot": int(pages_per_slot.max().item()),
        }
    return result


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    _, sequence = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        samples=1,
        rank=0,
        world_size=1,
    )[0]
    model = load_text_model(
        args.checkpoint,
        "two_level",
        args.two_level_topk,
        args.state_growth_factor,
        device,
    )
    text_model = getattr(model.model, "language_model", model.model)
    lod_layers = [
        attention
        for layer in text_model.layers
        if isinstance(
            (attention := getattr(layer, "self_attn", None)),
            Qwen3_5TwoLevelAttention,
        )
    ]
    for attention in lod_layers:
        attention._lod_collect_stats = True

    with torch.inference_mode():
        model.model(input_ids=sequence.unsqueeze(0).to(device), use_cache=False)

    layer_records = []
    all_selected = []
    all_selected_fraction = []
    all_union_fraction = []
    all_unique_fraction = []
    all_counts = []
    for attention in lod_layers:
        route_stats = attention._lod_route_stats
        selected = torch.cat(
            [item["selected_leaf_count"].float().cpu().flatten() for item in route_stats]
        )
        selected_fraction = torch.cat(
            [
                item["selected_leaf_fraction"].float().cpu().flatten()
                for item in route_stats
            ]
        )
        union_fraction = torch.cat(
            [item["union_leaf_fraction"].float().cpu() for item in route_stats]
        )
        unique_fraction = torch.cat(
            [item["unique_slot_fraction"].float().cpu() for item in route_stats]
        )
        counts = attention._lod_state["counts"].float().cpu().flatten()
        layer_records.append(
            {
                "layer": attention.layer_idx,
                "state_slots": int(attention._lod_state["counts"].size(2)),
                "history_leaves_per_kv_head": int(
                    attention._lod_state["counts"][0, 0].sum().item()
                ),
                "slot_leaf_count": summarize(counts),
                "page_layout": page_layout(counts),
                "largest_slot_over_mean": float(
                    (counts.max() / counts.mean()).item()
                ),
                "selected_leaves_per_query": summarize(selected),
                "selected_history_fraction": summarize(selected_fraction),
                "current_repacked_history_fraction_per_chunk_head": summarize(
                    union_fraction
                ),
                "unique_slots_selected_per_chunk_head_fraction": summarize(
                    unique_fraction
                ),
            }
        )
        all_selected.append(selected)
        all_selected_fraction.append(selected_fraction)
        all_union_fraction.append(union_fraction)
        all_unique_fraction.append(unique_fraction)
        all_counts.append(counts)

    record = {
        "checkpoint": args.checkpoint,
        "sequence_length": args.sequence_length,
        "input_tokens": int(sequence.numel()),
        "state_growth_factor": args.state_growth_factor,
        "two_level_topk": args.two_level_topk,
        "layers": layer_records,
        "all_layers": {
            "selected_leaves_per_query": summarize(torch.cat(all_selected)),
            "selected_history_fraction": summarize(
                torch.cat(all_selected_fraction)
            ),
            "current_repacked_history_fraction_per_chunk_head": summarize(
                torch.cat(all_union_fraction)
            ),
            "unique_slots_selected_per_chunk_head_fraction": summarize(
                torch.cat(all_unique_fraction)
            ),
            "page_layout": page_layout(torch.cat(all_counts)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["all_layers"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
