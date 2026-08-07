#!/usr/bin/env python3
"""Tune and attribute fused Qwen LOD decode at batch-size eight."""

from __future__ import annotations

import argparse
import json
import random
import types
from collections import defaultdict
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
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--state-growth-factor", type=float, default=8.0)
    parser.add_argument("--dynamic-open-decode-residual-mass", type=float)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--config-order-seed", type=int)
    parser.add_argument("--split-kv", type=int, nargs="+", default=(8,))
    parser.add_argument("--block-n", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--num-warps", type=int, nargs="+", default=(2, 4))
    parser.add_argument("--use-dot", action="store_true")
    parser.add_argument("--sweep-use-dot", action="store_true")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--no-clone-decode-routes", action="store_true")
    parser.add_argument("--disable-fused-decode-state-route", action="store_true")
    parser.add_argument(
        "--decode-route-group-size", type=int, nargs="+", default=(16,)
    )
    parser.add_argument(
        "--decode-route-num-warps", type=int, nargs="+", default=(2,)
    )
    parser.add_argument(
        "--decode-route-reduce-num-warps", type=int, nargs="+", default=(4,)
    )
    parser.add_argument(
        "--decode-final-reduce-num-warps", type=int, nargs="+", default=(4,)
    )
    parser.add_argument("--decode-route-use-dot", action="store_true")
    parser.add_argument("--sweep-decode-route-use-dot", action="store_true")
    parser.add_argument("--decode-route-gqa-grouped", action="store_true")
    parser.add_argument("--decode-fuse-final-reduce", action="store_true")
    parser.add_argument("--sweep-decode-fuse-final-reduce", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
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
    model = load_text_model(
        args.checkpoint,
        "two_level",
        8,
        args.state_growth_factor,
        device,
        "paged",
    )
    modules = [
        module
        for module in model.modules()
        if isinstance(module, Qwen3_5TwoLevelAttention)
    ]
    if not modules:
        raise RuntimeError("Qwen LOD attention modules were not installed")
    for module in modules:
        module.dynamic_open_decode_residual_mass = (
            args.dynamic_open_decode_residual_mass
        )
        module.clone_decode_routes = not args.no_clone_decode_routes
        module.fused_decode_state_route = not args.disable_fused_decode_state_route
        module.decode_route_gqa_grouped = args.decode_route_gqa_grouped
        module.decode_fuse_final_reduce = args.decode_fuse_final_reduce

    phase_events: dict[
        str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
    ] = defaultdict(list)
    if args.profile_phases:
        for module in modules:
            module._lod_decode_timing_events = phase_events
            for phase, method_name in (
                ("route", "_route_top_slots"),
                ("two_level", "_two_level_attention"),
            ):
                original = getattr(module, method_name)

                def timed(
                    self,
                    *method_args,
                    __original=original,
                    __phase=phase,
                    **kwargs,
                ):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    result = __original(*method_args, **kwargs)
                    end.record()
                    phase_events[__phase].append((begin, end))
                    return result

                setattr(module, method_name, types.MethodType(timed, module))

    next_token = sequence[:, -1:]
    def decode_one(cache, token_position: int):
        return model(
            input_ids=next_token,
            past_key_values=cache,
            cache_position=torch.tensor(
                [token_position], dtype=torch.long, device=device
            ),
            use_cache=True,
            logits_to_keep=1,
        )

    tile_configs = [
        (block_n, num_warps)
        for block_n in args.block_n
        for num_warps in args.num_warps
    ]
    leaf_dot_values = (False, True) if args.sweep_use_dot else (args.use_dot,)
    route_dot_values = (
        (False, True)
        if args.sweep_decode_route_use_dot
        else (args.decode_route_use_dot,)
    )
    fuse_final_values = (
        (False, True)
        if args.sweep_decode_fuse_final_reduce
        else (args.decode_fuse_final_reduce,)
    )
    configs = [
        (
            split_kv,
            block_n,
            num_warps,
            route_group_size,
            route_num_warps,
            route_reduce_num_warps,
            final_reduce_num_warps,
            leaf_use_dot,
            route_use_dot,
            fuse_final_reduce,
        )
        for split_kv in args.split_kv
        for block_n, num_warps in tile_configs
        for route_group_size in args.decode_route_group_size
        for route_num_warps in args.decode_route_num_warps
        for route_reduce_num_warps in args.decode_route_reduce_num_warps
        for final_reduce_num_warps in args.decode_final_reduce_num_warps
        for leaf_use_dot in leaf_dot_values
        for route_use_dot in route_dot_values
        for fuse_final_reduce in fuse_final_values
    ]
    if args.config_order_seed is not None:
        random.Random(args.config_order_seed).shuffle(configs)
    records = []
    with torch.inference_mode():
        for (
            split_kv,
            block_n,
            num_warps,
            route_group_size,
            route_num_warps,
            route_reduce_num_warps,
            final_reduce_num_warps,
            leaf_use_dot,
            route_use_dot,
            fuse_final_reduce,
        ) in configs:
            for module in modules:
                if hasattr(module, "_lod_state"):
                    delattr(module, "_lod_state")
            prefill = model(input_ids=sequence, use_cache=True, logits_to_keep=1)
            cache = prefill.past_key_values
            del prefill
            position = args.sequence_length
            for module in modules:
                module.decode_block_n = block_n
                module.decode_num_warps = num_warps
                module.decode_split_kv = split_kv
                module.decode_use_dot = leaf_use_dot
                module.decode_route_group_size = route_group_size
                module.decode_route_num_warps = route_num_warps
                module.decode_route_reduce_num_warps = route_reduce_num_warps
                module.decode_final_reduce_num_warps = final_reduce_num_warps
                module.decode_route_use_dot = route_use_dot
                module.decode_fuse_final_reduce = fuse_final_reduce
            # Compile this specialization and bring the GPU out of its idle
            # power state outside the measured interval.
            warm = None
            for _ in range(args.warmup_steps):
                warm = decode_one(cache, position)
                cache = warm.past_key_values
                position += 1
            if warm is None:
                raise ValueError("decode sweep requires at least one warmup step")
            del warm
            torch.cuda.synchronize(device)
            phase_events.clear()

            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            output = None
            for _ in range(args.steps):
                output = decode_one(cache, position)
                cache = output.past_key_values
                position += 1
            end.record()
            torch.cuda.synchronize(device)
            if output is None:
                raise ValueError("decode sweep requires at least one step")
            total_ms = float(begin.elapsed_time(end)) / args.steps
            route_ms = sum(
                float(start.elapsed_time(stop))
                for start, stop in phase_events["route"]
            ) / args.steps
            two_level_ms = sum(
                float(start.elapsed_time(stop))
                for start, stop in phase_events["two_level"]
            ) / args.steps
            records.append(
                {
                    "block_n": block_n,
                    "num_warps": num_warps,
                    "split_kv": split_kv,
                    "route_group_size": route_group_size,
                    "route_num_warps": route_num_warps,
                    "route_reduce_num_warps": route_reduce_num_warps,
                    "final_reduce_num_warps": final_reduce_num_warps,
                    "use_dot": leaf_use_dot,
                    "route_use_dot": route_use_dot,
                    "fuse_final_reduce": fuse_final_reduce,
                    "mean_decode_step_ms": total_ms,
                    "route_ms_per_step": route_ms,
                    "fused_attention_ms_per_step": two_level_ms - route_ms,
                    "two_level_ms_per_step": two_level_ms,
                    "kernel_ms_per_step": {
                        name: sum(
                            float(start.elapsed_time(stop))
                            for start, stop in events
                        )
                        / args.steps
                        for name, events in phase_events.items()
                        if name not in {"route", "two_level"}
                    },
                    "profile_phases": args.profile_phases,
                    "logit_finite": bool(torch.isfinite(output.logits).all().item()),
                }
            )
            del output

    record = {
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "dynamic_open_decode_residual_mass": (
            args.dynamic_open_decode_residual_mass
        ),
        "state_growth_factor": args.state_growth_factor,
        "steps": args.steps,
        "config_order_seed": args.config_order_seed,
        "attention_layers": len(modules),
        "clone_decode_routes": not args.no_clone_decode_routes,
        "fused_decode_state_route": not args.disable_fused_decode_state_route,
        "decode_route_group_sizes": args.decode_route_group_size,
        "decode_route_num_warps": args.decode_route_num_warps,
        "decode_route_reduce_num_warps": args.decode_route_reduce_num_warps,
        "decode_final_reduce_num_warps": args.decode_final_reduce_num_warps,
        "decode_route_use_dot": args.decode_route_use_dot,
        "decode_route_gqa_grouped": args.decode_route_gqa_grouped,
        "decode_fuse_final_reduce": args.decode_fuse_final_reduce,
        "configs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
