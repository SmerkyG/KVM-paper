#!/usr/bin/env python3
"""Measure block-list expansion for a block-sparse LOD detail branch.

Each archived KV position belongs to one KVM/LOD state slot.  This probe maps
each slot to the chronological KV blocks containing any of its leaves, routes
every prefill query to its top-k slots, and measures the union of those block
lists.  It reports masks kept separate by query head as well as masks OR-ed
across the GQA query heads sharing one KV head.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from model import hf_pytorch_lod_attention as hf_lod
from model.hf_pytorch_lod_attention import install_hf_lod_attention
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_attention import TritonLODAttentionCore
from scripts.compare_qwen35_lod_loss import select_sequences
from scripts.eval_hf_lod_lmeval import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--block-sizes", type=int, nargs="+", default=(16, 32, 64, 128)
    )
    parser.add_argument(
        "--query-tile-sizes", type=int, nargs="+", default=(1, 4, 8, 16)
    )
    parser.add_argument(
        "--state-clustering-policy",
        choices=("manual", "qk_norm_aware", "rope_aware"),
        default="qk_norm_aware",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class StreamingSummary:
    """Exact moments plus a bounded deterministic sample for quantiles."""

    def __init__(self, sample_per_update: int = 4096) -> None:
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.sample_per_update = sample_per_update
        self.samples: list[torch.Tensor] = []

    def add(self, values: torch.Tensor) -> None:
        values = values.detach().float().flatten()
        if not values.numel():
            return
        self.count += int(values.numel())
        self.total += float(values.double().sum().item())
        self.total_square += float(values.double().square().sum().item())
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))
        if values.numel() > self.sample_per_update:
            index = torch.linspace(
                0,
                int(values.numel()) - 1,
                self.sample_per_update,
                device=values.device,
            ).long()
            values = values[index]
        self.samples.append(values.cpu())

    def result(self) -> dict[str, float | int]:
        if not self.count:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        sample = torch.cat(self.samples)
        quantiles = torch.quantile(
            sample, torch.tensor((0.1, 0.5, 0.9), dtype=sample.dtype)
        )
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "p10": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p90": float(quantiles[2]),
            "min": self.minimum,
            "max": self.maximum,
        }


class MaskMetrics:
    def __init__(self) -> None:
        self.block_count = StreamingSummary()
        self.block_density = StreamingSummary()
        self.union_inflation = StreamingSummary()
        self.useful_token_density = StreamingSummary()
        self.useful_cells = 0.0
        self.scheduled_cells = 0.0

    def add(
        self,
        block_count: torch.Tensor,
        *,
        available_blocks: int,
        baseline_block_count: torch.Tensor,
        useful_cells: torch.Tensor,
        scheduled_cells: torch.Tensor,
    ) -> None:
        block_count = block_count.float()
        scheduled_cells = scheduled_cells.float().clamp_min(1)
        self.block_count.add(block_count)
        self.block_density.add(block_count / available_blocks)
        self.union_inflation.add(
            block_count / baseline_block_count.float().clamp_min(1)
        )
        self.useful_token_density.add(useful_cells.float() / scheduled_cells)
        self.useful_cells += float(useful_cells.double().sum().item())
        self.scheduled_cells += float(scheduled_cells.double().sum().item())

    def result(self) -> dict[str, Any]:
        return {
            "block_count": self.block_count.result(),
            "block_density": self.block_density.result(),
            "union_inflation_vs_mean_row": self.union_inflation.result(),
            "useful_token_density": self.useful_token_density.result(),
            "global_useful_token_fraction": (
                self.useful_cells / self.scheduled_cells
                if self.scheduled_cells
                else 0.0
            ),
        }


MetricsKey = tuple[int, int, int, str]


def _metric_key(
    layer: int, block_size: int, tile_size: int, scope: str
) -> MetricsKey:
    return layer, block_size, tile_size, scope


def _slot_block_membership(
    owners: torch.Tensor,
    *,
    state_len: int,
    position_offset: int,
    block_size: int,
) -> tuple[torch.Tensor, int]:
    batch, kv_heads, leaf_count = owners.shape
    block_index = (
        torch.arange(leaf_count, device=owners.device) + position_offset
    ) // block_size
    available_blocks = int(block_index[-1].item()) + 1
    membership = torch.zeros(
        batch,
        kv_heads,
        state_len,
        available_blocks,
        dtype=torch.bool,
        device=owners.device,
    )
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            owner = owners[batch_index, kv_head].long()
            valid = owner.ge(0) & owner.lt(state_len)
            flat_index = owner[valid] * available_blocks + block_index[valid]
            membership[batch_index, kv_head].view(-1)[flat_index] = True
    return membership, available_blocks


def _row_block_masks(
    membership: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    gqa_ratio: int,
    row_batch: int = 32,
) -> torch.Tensor:
    batch, query_heads, query_len, _ = top_slots.shape
    kv_head = torch.arange(query_heads, device=top_slots.device) // gqa_ratio
    batch_index = torch.arange(batch, device=top_slots.device)
    rows = []
    for begin in range(0, query_len, row_batch):
        routes = top_slots[..., begin : begin + row_batch, :]
        valid = routes.ge(0)
        route_blocks = membership[
            batch_index[:, None, None, None],
            kv_head[None, :, None, None],
            routes.clamp(min=0, max=int(membership.size(2)) - 1),
        ]
        route_blocks &= valid.unsqueeze(-1)
        rows.append(route_blocks.any(dim=-2))
    return torch.cat(rows, dim=2)


def _selected_leaf_counts(
    owners: torch.Tensor,
    top_slots: torch.Tensor,
    *,
    state_len: int,
    gqa_ratio: int,
) -> torch.Tensor:
    batch, kv_heads, _ = owners.shape
    query_heads = int(top_slots.size(1))
    slot_counts = torch.zeros(
        batch,
        kv_heads,
        state_len,
        dtype=torch.int32,
        device=owners.device,
    )
    for batch_index in range(batch):
        for kv_head in range(kv_heads):
            owner = owners[batch_index, kv_head].long()
            valid = owner.ge(0) & owner.lt(state_len)
            slot_counts[batch_index, kv_head] = torch.bincount(
                owner[valid], minlength=state_len
            ).to(torch.int32)
    kv_head = torch.arange(query_heads, device=owners.device) // gqa_ratio
    batch_index = torch.arange(batch, device=owners.device)
    valid_route = top_slots.ge(0)
    selected = slot_counts[
        batch_index[:, None, None, None],
        kv_head[None, :, None, None],
        top_slots.clamp(min=0, max=state_len - 1),
    ]
    return selected.masked_fill(~valid_route, 0).sum(dim=-1)


def _pad_rows(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    missing = padded_length - int(tensor.size(2))
    if not missing:
        return tensor
    return torch.nn.functional.pad(tensor, (0, 0, 0, missing))


def _record_masks(
    metrics: dict[MetricsKey, MaskMetrics],
    *,
    layer: int,
    owners: torch.Tensor,
    top_slots: torch.Tensor,
    state_len: int,
    position_offset: int,
    block_sizes: tuple[int, ...],
    query_tile_sizes: tuple[int, ...],
    gqa_ratio: int,
) -> dict[str, Any]:
    batch, query_heads, query_len, _ = top_slots.shape
    kv_heads = int(owners.size(1))
    if query_heads != kv_heads * gqa_ratio:
        raise AssertionError("query/KV head geometry changed during routing")
    useful_per_row = _selected_leaf_counts(
        owners,
        top_slots,
        state_len=state_len,
        gqa_ratio=gqa_ratio,
    )
    route_summary: dict[str, Any] = {
        "remote_tokens": int(owners.size(2)),
        "state_slots": state_len,
        "query_rows": query_len,
    }
    for block_size in block_sizes:
        membership, available_blocks = _slot_block_membership(
            owners,
            state_len=state_len,
            position_offset=position_offset,
            block_size=block_size,
        )
        row_mask = _row_block_masks(
            membership, top_slots, gqa_ratio=gqa_ratio
        )
        row_count = row_mask.sum(dim=-1)
        block_summary: dict[str, Any] = {"available_blocks": available_blocks}
        for tile_size in query_tile_sizes:
            tile_count = math.ceil(query_len / tile_size)
            padded_length = tile_count * tile_size
            valid_rows = torch.arange(
                padded_length, device=owners.device
            ).lt(query_len).reshape(tile_count, tile_size)
            valid_per_tile = valid_rows.sum(dim=1)

            padded_mask = _pad_rows(row_mask, padded_length)
            head_tile_mask = padded_mask.reshape(
                batch,
                query_heads,
                tile_count,
                tile_size,
                available_blocks,
            ).any(dim=3)
            head_block_count = head_tile_mask.sum(dim=-1)
            padded_row_count = _pad_rows(row_count.unsqueeze(-1), padded_length)[
                ..., 0
            ]
            head_baseline = padded_row_count.reshape(
                batch, query_heads, tile_count, tile_size
            ).sum(dim=-1) / valid_per_tile
            padded_useful = _pad_rows(
                useful_per_row.unsqueeze(-1), padded_length
            )[..., 0]
            head_useful = padded_useful.reshape(
                batch, query_heads, tile_count, tile_size
            ).sum(dim=-1)
            head_scheduled = (
                head_block_count * block_size * valid_per_tile
            )
            head_metric = metrics[
                _metric_key(layer, block_size, tile_size, "query_head")
            ]
            head_metric.add(
                head_block_count,
                available_blocks=available_blocks,
                baseline_block_count=head_baseline,
                useful_cells=head_useful,
                scheduled_cells=head_scheduled,
            )

            gqa_tile_mask = head_tile_mask.reshape(
                batch,
                kv_heads,
                gqa_ratio,
                tile_count,
                available_blocks,
            ).any(dim=2)
            gqa_block_count = gqa_tile_mask.sum(dim=-1)
            gqa_baseline = head_baseline.reshape(
                batch, kv_heads, gqa_ratio, tile_count
            ).sum(dim=2) / gqa_ratio
            gqa_useful = head_useful.reshape(
                batch, kv_heads, gqa_ratio, tile_count
            ).sum(dim=2)
            gqa_scheduled = (
                gqa_block_count * block_size * valid_per_tile * gqa_ratio
            )
            gqa_metric = metrics[
                _metric_key(layer, block_size, tile_size, "gqa_kv_head")
            ]
            gqa_metric.add(
                gqa_block_count,
                available_blocks=available_blocks,
                baseline_block_count=gqa_baseline,
                useful_cells=gqa_useful,
                scheduled_cells=gqa_scheduled,
            )
            block_summary[f"tile_{tile_size}"] = {
                "query_head_mean_blocks": float(
                    head_block_count.float().mean().item()
                ),
                "query_head_mean_density": float(
                    head_block_count.float().mean().item() / available_blocks
                ),
                "gqa_mean_blocks": float(gqa_block_count.float().mean().item()),
                "gqa_mean_density": float(
                    gqa_block_count.float().mean().item() / available_blocks
                ),
            }
        route_summary[f"block_{block_size}"] = block_summary
        del membership, row_mask
    return route_summary


def _nested_results(
    metrics: dict[MetricsKey, MaskMetrics], layer: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (metric_layer, block_size, tile_size, scope), value in metrics.items():
        if metric_layer != layer:
            continue
        result.setdefault(f"block_{block_size}", {}).setdefault(
            f"tile_{tile_size}", {}
        )[scope] = value.result()
    return result


def _combine_layer_metrics(
    metrics: dict[MetricsKey, MaskMetrics], layers: list[int]
) -> dict[str, Any]:
    combined: dict[tuple[int, int, str], MaskMetrics] = defaultdict(MaskMetrics)
    for (layer, block_size, tile_size, scope), source in metrics.items():
        if layer not in layers:
            continue
        target = combined[block_size, tile_size, scope]
        for name in (
            "block_count",
            "block_density",
            "union_inflation",
            "useful_token_density",
        ):
            source_summary = getattr(source, name)
            target_summary = getattr(target, name)
            target_summary.count += source_summary.count
            target_summary.total += source_summary.total
            target_summary.total_square += source_summary.total_square
            target_summary.minimum = min(
                target_summary.minimum, source_summary.minimum
            )
            target_summary.maximum = max(
                target_summary.maximum, source_summary.maximum
            )
            target_summary.samples.extend(source_summary.samples)
        target.useful_cells += source.useful_cells
        target.scheduled_cells += source.scheduled_cells
    result: dict[str, Any] = {}
    for (block_size, tile_size, scope), value in combined.items():
        result.setdefault(f"block_{block_size}", {}).setdefault(
            f"tile_{tile_size}", {}
        )[scope] = value.result()
    return result


def main() -> None:
    args = parse_args()
    if args.sample_index < 0:
        raise ValueError("sample index must be nonnegative")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    block_sizes = tuple(sorted(set(args.block_sizes)))
    query_tile_sizes = tuple(sorted(set(args.query_tile_sizes)))
    if any(value <= 0 for value in block_sizes + query_tile_sizes):
        raise ValueError("block and query tile sizes must be positive")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    selected = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        samples=args.sample_index + 1,
        rank=args.sample_index,
        world_size=args.sample_index + 1,
    )
    sample_id, sequence = selected[-1]
    model, acceleration = load_model(args.checkpoint, device)
    text_config = AutoConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    ).get_text_config(decoder=True)
    installed = install_hf_lod_attention(
        model,
        config=PagedLODConfig(
            chunk_size=256,
            local_window=512,
            state_growth_factor=16,
            state_min_size=256,
            protected_prefix=1,
            state_clustering_policy=args.state_clustering_policy,
            page_size=16,
        ),
        open_count=8,
        engine_backend="kernel",
    )

    metrics: dict[MetricsKey, MaskMetrics] = defaultdict(MaskMetrics)
    original_build = hf_lod._build_engine
    original_route = TritonLODAttentionCore._route_top_slots
    original_new = TritonLODAttentionCore._new_page_cache
    original_append = TritonLODAttentionCore._append_page_cache

    def recording_build(*build_args, **build_kwargs):
        engine = original_build(*build_args, **build_kwargs)
        owner = build_kwargs.get("stats_owner")
        engine._analysis_layer = int(getattr(owner, "layer_idx", -1))
        return engine

    def recording_new(self, k, v, owners, **kwargs):
        # ``_new_page_cache`` records the initial leaves by calling the same
        # append method used by later state catch-up steps.
        self._analysis_owner_parts = []
        self._analysis_last_route = None
        return original_new(self, k, v, owners, **kwargs)

    def recording_append(self, cache, key, value, owners):
        self._analysis_owner_parts.append(owners.detach().clone())
        return original_append(self, cache, key, value, owners)

    def recording_route(self, q, state_k, state_v, counts, **kwargs):
        configured = (
            self.prefill_two_level_topk
            if int(q.size(2)) > 1 and self.prefill_two_level_topk is not None
            else self.two_level_topk
        )
        if int(q.size(2)) <= 1:
            return original_route(self, q, state_k, state_v, counts, **kwargs)
        normal_slots = original_route(self, q, state_k, state_v, counts, **kwargs)
        if args.top_k == configured:
            analysis_slots = normal_slots
        else:
            self.prefill_two_level_topk = args.top_k
            try:
                analysis_slots = original_route(
                    self, q, state_k, state_v, counts, **kwargs
                )
            finally:
                self.prefill_two_level_topk = configured

        page_cache = kwargs.get("page_cache")
        if not isinstance(page_cache, dict):
            raise AssertionError("block sparsity probe requires a paged owner cache")
        leaf_count = int(page_cache["leaf_count"])
        owners = torch.cat(self._analysis_owner_parts, dim=2)
        if int(owners.size(2)) != leaf_count:
            raise AssertionError(
                f"captured {owners.size(2)} owners for {leaf_count} leaves"
            )
        position_offset = self.sink_len if self.separate_sink_cache else 0
        self._analysis_last_route = _record_masks(
            metrics,
            layer=int(self._analysis_layer),
            owners=owners,
            top_slots=analysis_slots,
            state_len=int(kwargs["state_len"]),
            position_offset=position_offset,
            block_sizes=block_sizes,
            query_tile_sizes=query_tile_sizes,
            gqa_ratio=self.num_key_value_groups,
        )
        return normal_slots

    hf_lod._build_engine = recording_build
    TritonLODAttentionCore._route_top_slots = recording_route
    TritonLODAttentionCore._new_page_cache = recording_new
    TritonLODAttentionCore._append_page_cache = recording_append
    try:
        with torch.inference_mode():
            model(
                input_ids=sequence.unsqueeze(0).to(device),
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        hf_lod._build_engine = original_build
        TritonLODAttentionCore._route_top_slots = original_route
        TritonLODAttentionCore._new_page_cache = original_new
        TritonLODAttentionCore._append_page_cache = original_append

    layer_records = []
    for module in model.modules():
        engine = getattr(module, "_hf_lod_transient_engine", None)
        layer = getattr(module, "layer_idx", None)
        if engine is None or not isinstance(layer, int):
            continue
        last_route = getattr(engine, "_analysis_last_route", None)
        if last_route is None:
            continue
        layer_records.append(
            {
                "layer": layer,
                "gqa_ratio": int(module.num_key_value_groups),
                "last_route": last_route,
                "all_routes": _nested_results(metrics, layer),
            }
        )
    layers = [record["layer"] for record in layer_records]
    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "sample_id": sample_id,
        "sequence_length": args.sequence_length,
        "top_k": args.top_k,
        "block_sizes": block_sizes,
        "query_tile_sizes": query_tile_sizes,
        "state_clustering_policy": args.state_clustering_policy,
        "installed_attention_modules": installed,
        "attention_layers": len(layer_records),
        "architecture": {
            "model_type": text_config.model_type,
            "query_heads": int(text_config.num_attention_heads),
            "kv_heads": int(text_config.num_key_value_heads),
            "gqa_ratio": int(
                text_config.num_attention_heads // text_config.num_key_value_heads
            ),
        },
        "acceleration": acceleration,
        "all_layers": _combine_layer_metrics(metrics, layers),
        "layers": layer_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["architecture"], sort_keys=True))
    for block_size in block_sizes:
        summary = payload["all_layers"][f"block_{block_size}"]["tile_16"]
        print(
            json.dumps(
                {"block_size": block_size, "query_tile_16": summary},
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
