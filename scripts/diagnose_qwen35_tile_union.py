#!/usr/bin/env python3
"""Measure route overlap and unmasked tile-union work for LOD prefill."""

from __future__ import annotations

import argparse
import json
import math
import types
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=8.0)
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=(4, 8, 16, 32))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile_from_histogram(histogram: list[int], fraction: float) -> int:
    total = sum(histogram)
    target = max(1, math.ceil(total * fraction))
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return value
    return len(histogram) - 1


def main() -> None:
    args = parse_args()
    if not args.tile_sizes or any(size <= 0 for size in args.tile_sizes):
        raise ValueError("tile sizes must be positive")
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
        select_sequences(
            tokenizer,
            args.dataset,
            args.sequence_length,
            1,
            0,
            1,
        )[0][1]
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
        module.leaf_layout = "query"

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
        for module in modules:
            if hasattr(module, "_lod_state"):
                delattr(module, "_lod_state")

        totals: dict[int, dict[str, torch.Tensor | int]] = {}
        for module in modules:
            original = module._paged_leaf_attention

            def measured(self, q, top_slots, cache, __original=original):
                batch, query_heads, query_len, _ = q.shape
                route_count = int(top_slots.size(-1))
                kv_heads = int(cache["slot_lengths"].size(1))
                kv_group_size = query_heads // kv_heads
                slot_lengths = cache["slot_lengths"].repeat_interleave(
                    kv_group_size, dim=1
                )
                safe_top_slots = top_slots.clamp_min(0)
                selected_lengths = torch.gather(
                    slot_lengths.unsqueeze(2).expand(
                        batch, query_heads, query_len, -1
                    ),
                    3,
                    safe_top_slots,
                )
                selected_lengths = torch.where(
                    top_slots >= 0,
                    selected_lengths,
                    torch.zeros_like(selected_lengths),
                )
                selected_leaf_work = selected_lengths.sum(dtype=torch.int64)

                for tile_size in args.tile_sizes:
                    tile_count = (query_len + tile_size - 1) // tile_size
                    padded_len = tile_count * tile_size
                    padded_slots = torch.nn.functional.pad(
                        top_slots,
                        (0, 0, 0, padded_len - query_len),
                        value=-1,
                    )
                    candidates = padded_slots.reshape(
                        batch,
                        query_heads,
                        tile_count,
                        tile_size * route_count,
                    )
                    sorted_slots = candidates.sort(dim=-1).values
                    unique = sorted_slots >= 0
                    unique[..., 1:] &= (
                        sorted_slots[..., 1:] != sorted_slots[..., :-1]
                    )
                    union_centroids = unique.sum(dim=-1, dtype=torch.int64)
                    safe_sorted_slots = sorted_slots.clamp_min(0)
                    union_lengths = torch.gather(
                        slot_lengths.unsqueeze(2).expand(
                            batch, query_heads, tile_count, -1
                        ),
                        3,
                        safe_sorted_slots,
                    )
                    union_leaf_count = torch.where(
                        unique,
                        union_lengths,
                        torch.zeros_like(union_lengths),
                    ).sum(dim=-1, dtype=torch.int64)
                    queries_per_tile = torch.clamp(
                        query_len
                        - torch.arange(tile_count, device=q.device) * tile_size,
                        min=0,
                        max=tile_size,
                    ).to(torch.int64)
                    query_weight = queries_per_tile.view(1, 1, tile_count)

                    if tile_size not in totals:
                        totals[tile_size] = {
                            "calls": 0,
                            "histogram": torch.zeros(
                                tile_size * route_count + 1,
                                dtype=torch.int64,
                                device=q.device,
                            ),
                            "tiles": torch.zeros((), dtype=torch.int64, device=q.device),
                            "queries": torch.zeros((), dtype=torch.int64, device=q.device),
                            "union_centroids": torch.zeros(
                                (), dtype=torch.int64, device=q.device
                            ),
                            "exposed_centroid_work": torch.zeros(
                                (), dtype=torch.int64, device=q.device
                            ),
                            "selected_centroid_work": torch.zeros(
                                (), dtype=torch.int64, device=q.device
                            ),
                            "union_leaf_work": torch.zeros(
                                (), dtype=torch.int64, device=q.device
                            ),
                            "selected_leaf_work": torch.zeros(
                                (), dtype=torch.int64, device=q.device
                            ),
                        }
                    entry = totals[tile_size]
                    entry["calls"] = int(entry["calls"]) + 1
                    histogram = entry["histogram"]
                    if not isinstance(histogram, torch.Tensor):
                        raise TypeError("tile histogram is not a tensor")
                    histogram.add_(
                        torch.bincount(
                            union_centroids.reshape(-1),
                            minlength=tile_size * route_count + 1,
                        )
                    )
                    for name, value in (
                        ("tiles", union_centroids.numel()),
                        ("queries", batch * query_heads * query_len),
                        ("union_centroids", union_centroids.sum()),
                        (
                            "exposed_centroid_work",
                            (union_centroids * query_weight).sum(),
                        ),
                        (
                            "selected_centroid_work",
                            batch * query_heads * query_len * route_count,
                        ),
                        ("union_leaf_work", (union_leaf_count * query_weight).sum()),
                        ("selected_leaf_work", selected_leaf_work),
                    ):
                        target = entry[name]
                        if not isinstance(target, torch.Tensor):
                            raise TypeError(f"tile total {name} is not a tensor")
                        target.add_(value)
                return __original(q, top_slots, cache)

            module._paged_leaf_attention = types.MethodType(measured, module)

        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)

    summaries = {}
    for tile_size, entry in totals.items():
        histogram_tensor = entry["histogram"]
        if not isinstance(histogram_tensor, torch.Tensor):
            raise TypeError("tile histogram is not a tensor")
        histogram = [int(value) for value in histogram_tensor.cpu().tolist()]
        scalar = {
            name: int(value.item())
            for name, value in entry.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        tiles = scalar["tiles"]
        selected_centroid_work = scalar["selected_centroid_work"]
        selected_leaf_work = scalar["selected_leaf_work"]
        summaries[str(tile_size)] = {
            "calls": int(entry["calls"]),
            "tiles": tiles,
            "mean_union_centroids": scalar["union_centroids"] / tiles,
            "p50_union_centroids": percentile_from_histogram(histogram, 0.50),
            "p90_union_centroids": percentile_from_histogram(histogram, 0.90),
            "p99_union_centroids": percentile_from_histogram(histogram, 0.99),
            "max_union_centroids": max(
                index for index, count in enumerate(histogram) if count
            ),
            "centroid_exposure_inflation": (
                scalar["exposed_centroid_work"] / selected_centroid_work
            ),
            "leaf_compute_inflation": (
                scalar["union_leaf_work"] / selected_leaf_work
            ),
            "histogram": histogram,
        }
    record = {
        "checkpoint": args.checkpoint,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "state_growth_factor": args.state_growth_factor,
        "route_count": int(next(iter(modules)).prefill_two_level_topk),
        "attention_layers": len(modules),
        "tile_sizes": list(args.tile_sizes),
        "summaries": summaries,
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
