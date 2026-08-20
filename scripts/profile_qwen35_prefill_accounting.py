#!/usr/bin/env python3
"""Reconcile whole-model Qwen3.5 prefill time with attention subcomponents."""

from __future__ import annotations

import argparse
import json
import types
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from model.qwen35_two_level_attention import Qwen3_5TwoLevelAttention
from scripts.probe_qwen35_lod_niah import (
    load_text_model,
    require_qwen35_acceleration,
)
from scripts.profile_qwen35_prefill_total import (
    clear_lod_state,
    select_profile_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--mode", choices=("full", "two_level"), required=True)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dense-page-prefill", action="store_true")
    parser.add_argument("--dense-page-split-kernels", action="store_true")
    parser.add_argument("--dense-page-topk", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--dense-page-block-m", type=int, default=64)
    parser.add_argument("--dense-page-block-n", type=int, default=64)
    parser.add_argument(
        "--dense-page-indexed-aiter-union",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dense-page-union-query-tile", type=int, choices=(4, 8, 16, 32), default=32
    )
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument(
        "--page-summary-quant-bits", type=int, choices=(0, 8), default=0
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def event() -> torch.cuda.Event:
    result = torch.cuda.Event(enable_timing=True)
    result.record()
    return result


def install_module_hooks(
    model: torch.nn.Module,
    pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles = []
    starts: dict[tuple[int, str], list[torch.cuda.Event]] = defaultdict(list)

    def add_hooks(module: torch.nn.Module, label: str) -> None:
        key = (id(module), label)

        def before(_module, _args):
            starts[key].append(event())

        def after(_module, _args, _output):
            pairs[label].append((starts[key].pop(), event()))

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))

    attention = [
        module for module in model.modules() if isinstance(module, Qwen3_5Attention)
    ]
    for module in model.modules():
        name = module.__class__.__name__
        if isinstance(module, Qwen3_5Attention):
            add_hooks(module, "attention_modules")
        elif name.endswith("GatedDeltaNet"):
            add_hooks(module, "gdn_modules")
        elif name == "Qwen3_5MLP":
            add_hooks(module, "mlp_modules")
    for module in attention:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add_hooks(getattr(module, name), "attention_projections")
    return handles


def install_lod_method_timers(
    modules: list[Qwen3_5TwoLevelAttention],
    pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> None:
    phases = {
        "front_exact_attention": "_exact_attention",
        "local_attention": "_prefill_local_attention",
        "remote_attention": "_two_level_attention",
        "state_update": "_update_state",
        "page_append": "_append_page_cache",
    }
    dense_names = (
        "total",
        "summary_select",
        "summary_attention",
        "summary_prepare",
        "summary_flash",
        "page_topk",
        "summary_removal",
        "union_build",
        "indexed_table",
        "indexed_aiter",
        "indexed_unpack",
        "exact_pages",
    )
    for module in modules:
        module._lod_dense_page_timing_events = {
            name: pairs[f"dense_{name}"] for name in dense_names
        }
        for phase, method_name in phases.items():
            original = getattr(module, method_name)

            def timed(self, *args, __original=original, __phase=phase, **kwargs):
                begin = event()
                result = __original(*args, **kwargs)
                pairs[__phase].append((begin, event()))
                return result

            setattr(module, method_name, types.MethodType(timed, module))


def mean_event_ms(
    pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
    repeats: int,
) -> dict[str, float]:
    return {
        name: sum(begin.elapsed_time(end) for begin, end in entries) / repeats
        for name, entries in pairs.items()
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
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
        args.mode,
        8,
        16.0,
        device,
        "paged",
        require_fla_fast_path=True,
    )
    acceleration = require_qwen35_acceleration(model)
    lod_modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    for module in lod_modules:
        module.recursive_page_lod = args.recursive_page_lod
        module.virtual_page_storage = args.virtual_page_storage
        module.page_summary_quant_bits = args.page_summary_quant_bits
        module.dense_page_prefill = args.dense_page_prefill
        module.dense_page_split_kernels = args.dense_page_split_kernels
        module.dense_page_topk = args.dense_page_topk
        module.dense_page_block_m = args.dense_page_block_m
        module.dense_page_block_n = args.dense_page_block_n
        module.dense_page_indexed_aiter_union = args.dense_page_indexed_aiter_union
        module.dense_page_union_query_tile = args.dense_page_union_query_tile

    with torch.inference_mode():
        warm = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize(device)
        del warm
        clear_lod_state(model)

        pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(
            list
        )
        handles = install_module_hooks(model, pairs)
        if lod_modules:
            install_lod_method_timers(lod_modules, pairs)
        elif args.mode == "full":
            original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

            def timed_sdpa(*call_args, **call_kwargs):
                begin = event()
                result = original_sdpa(*call_args, **call_kwargs)
                pairs["native_sdpa"].append((begin, event()))
                return result

            ALL_ATTENTION_FUNCTIONS["sdpa"] = timed_sdpa

        totals = []
        result = None
        for _ in range(args.repeats):
            clear_lod_state(model)
            begin = event()
            result = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
            totals.append((begin, event()))
        torch.cuda.synchronize(device)
        for handle in handles:
            handle.remove()

    if result is None:
        raise RuntimeError("no profile result")
    phases = mean_event_ms(pairs, args.repeats)
    total_ms = sum(begin.elapsed_time(end) for begin, end in totals) / args.repeats
    page_count_histogram: dict[str, int] = defaultdict(int)
    for module in lod_modules:
        page_cache = module._lod_state.get("page_cache")
        if not isinstance(page_cache, dict):
            continue
        counts = page_cache.get("page_counts")
        next_page = page_cache.get("next_page")
        if not isinstance(counts, torch.Tensor) or not isinstance(
            next_page, torch.Tensor
        ):
            continue
        page = torch.arange(counts.size(-1), device=counts.device)
        allocated = counts[page < next_page.unsqueeze(-1)]
        unique, frequencies = torch.unique(allocated, return_counts=True)
        for count, frequency in zip(unique.cpu().tolist(), frequencies.cpu().tolist()):
            page_count_histogram[str(int(count))] += int(frequency)
    whole_disjoint = sum(
        phases.get(name, 0.0)
        for name in ("attention_modules", "gdn_modules", "mlp_modules")
    )
    attention_disjoint_names = (
        "attention_projections",
        "native_sdpa",
        "front_exact_attention",
        "local_attention",
        "remote_attention",
        "state_update",
        "page_append",
    )
    attention_disjoint = sum(phases.get(name, 0.0) for name in attention_disjoint_names)
    record = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "attention_layers": sum(
            isinstance(module, Qwen3_5Attention) for module in model.modules()
        ),
        "dense_page_prefill": args.dense_page_prefill,
        "dense_page_split_kernels": args.dense_page_split_kernels,
        "dense_page_topk": args.dense_page_topk,
        "dense_page_indexed_aiter_union": args.dense_page_indexed_aiter_union,
        "dense_page_union_query_tile": args.dense_page_union_query_tile,
        "acceleration": acceleration,
        "total_ms": total_ms,
        "phase_ms": phases,
        "page_count_histogram": dict(
            sorted(page_count_histogram.items(), key=lambda item: int(item[0]))
        ),
        "whole_model_reconciliation": {
            "measured_module_ms": whole_disjoint,
            "other_ms": total_ms - whole_disjoint,
            "closure_error_fraction": (total_ms - whole_disjoint) / total_ms,
        },
        "attention_reconciliation": {
            "measured_internal_ms": attention_disjoint,
            "other_ms": phases.get("attention_modules", 0.0) - attention_disjoint,
        },
        "finite": bool(torch.isfinite(result.logits).all().item()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
