#!/usr/bin/env python3
"""Measure how often prefill queries share the same complete top-k slot set."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument("--state-growth-factor", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize_group_counts(group_counts: list[torch.Tensor]) -> dict[str, float]:
    counts = torch.cat(group_counts).long()
    weighted = torch.repeat_interleave(counts, counts)
    quantiles = torch.quantile(
        weighted.float(), torch.tensor((0.5, 0.9, 0.95, 0.99))
    )
    total = int(counts.sum().item())
    result = {
        "queries": total,
        "unique_tuples": int(counts.numel()),
        "mean_group_size_per_tuple": float(counts.float().mean().item()),
        "query_weighted_p50": float(quantiles[0].item()),
        "query_weighted_p90": float(quantiles[1].item()),
        "query_weighted_p95": float(quantiles[2].item()),
        "query_weighted_p99": float(quantiles[3].item()),
        "max_group_size": int(counts.max().item()),
    }
    for threshold in (2, 4, 8, 16, 32, 64):
        result[f"query_fraction_in_group_ge_{threshold}"] = float(
            counts[counts >= threshold].sum().item()
        ) / float(total)
    return result


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
    sequence = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        1,
        0,
        1,
    )[0][1].unsqueeze(0).to(device)
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    counts_by_layer: dict[int, list[torch.Tensor]] = {
        module.layer_idx: [] for module in modules
    }
    for module in modules:
        original = module._route_top_slots

        def collect(self, q, state_k, counts, *, __original=original):
            top_slots = __original(q, state_k, counts)
            batch, query_heads, query_len, route_count = top_slots.shape
            canonical = top_slots.sort(dim=-1).values.reshape(
                batch * query_heads * query_len, route_count
            )
            kv_head = torch.div(
                torch.arange(query_heads, device=q.device),
                self.num_key_value_groups,
                rounding_mode="floor",
            )
            kv_head = kv_head.view(1, query_heads, 1).expand(
                batch, query_heads, query_len
            ).reshape(-1, 1)
            signature = torch.cat((kv_head, canonical), dim=-1).cpu()
            _, group_counts = torch.unique(
                signature, dim=0, return_counts=True
            )
            counts_by_layer[self.layer_idx].append(group_counts)
            return top_slots

        module._route_top_slots = types.MethodType(collect, module)

    with torch.inference_mode():
        result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)

    layer_records = {
        str(layer): summarize_group_counts(group_counts)
        for layer, group_counts in counts_by_layer.items()
    }
    record = {
        "sequence_length": args.sequence_length,
        "attention_layers": len(modules),
        "all_layers": summarize_group_counts(
            [counts for values in counts_by_layer.values() for counts in values]
        ),
        "layers": layer_records,
        "logit_finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["all_layers"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
