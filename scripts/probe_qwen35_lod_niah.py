#!/usr/bin/env python3
"""Paired NIAH evaluation of Qwen3.5 full and top-k LOD attention."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_modeling
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from model.qwen35_two_level_attention import (
    Qwen3_5TwoLevelAttention,
    graft_qwen35_two_level_attention,
    pop_qwen35_dynamic_open_statistics,
    qwen35_page_quantization_statistics,
)


def enable_fla_fast_path(*, required: bool = False) -> bool:
    """Work around Transformers 5.3 checking the obsolete `fla` dist name."""
    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError as error:
        if required:
            raise RuntimeError(
                "Qwen3.5's FLA fast path is required for this run; install the "
                "project's `rwkv-kernels` extra"
            ) from error
        return False
    qwen35_modeling.FusedRMSNormGated = FusedRMSNormGated
    qwen35_modeling.chunk_gated_delta_rule = chunk_gated_delta_rule
    qwen35_modeling.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    return True


def generate_documents(task: str, checkpoint: str, length: int) -> list[dict]:
    random.seed(0)
    np.random.seed(1234)
    from lm_eval.tasks.ruler.niah_utils import (
        niah_single_1,
        niah_single_2,
        niah_single_3,
    )

    generator = {
        "niah_single_1": niah_single_1,
        "niah_single_2": niah_single_2,
        "niah_single_3": niah_single_3,
    }[task]
    standard_lengths = [4096, 8192, 16384, 32768, 65536]
    lengths = [value for value in standard_lengths if value <= length]
    dataset = generator(max_seq_lengths=lengths, pretrained=checkpoint)["test"]
    return [doc for doc in dataset if int(doc["max_length"]) == length]


def load_text_model(
    checkpoint: str,
    mode: str,
    topk: int,
    state_growth_factor: float,
    device: torch.device,
    leaf_attention_backend: str = "packed",
    require_fla_fast_path: bool = False,
) -> Qwen3_5ForCausalLM:
    enable_fla_fast_path(required=require_fla_fast_path)
    composite_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    config = composite_config.text_config
    config._attn_implementation = "sdpa"
    model = (
        Qwen3_5ForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    if mode == "two_level":
        replaced = graft_qwen35_two_level_attention(
            model,
            topk=topk,
            state_growth_factor=state_growth_factor,
            leaf_attention_backend=leaf_attention_backend,
        )
        expected = [
            index
            for index, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
        ]
        if replaced != expected:
            raise RuntimeError(
                f"replaced Qwen attention layers {replaced}, expected {expected}"
            )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "two_level"), required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("niah_single_1", "niah_single_2", "niah_single_3"),
        default=("niah_single_1", "niah_single_2", "niah_single_3"),
    )
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument("--dynamic-open-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-residual-mass", type=float)
    parser.add_argument("--dynamic-open-decode-top-p", type=float)
    parser.add_argument("--dynamic-open-decode-residual-mass", type=float)
    parser.add_argument("--disable-dynamic-open-stats", action="store_true")
    parser.add_argument("--reuse-dynamic-local-attention", action="store_true")
    parser.add_argument("--dynamic-open-residual-state-bound", action="store_true")
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument("--recursive-page-block-n", type=int, default=16)
    parser.add_argument("--leaf-num-warps", type=int, default=2)
    parser.add_argument("--leaf-key-quant-bits", type=int, choices=(0, 4, 8), default=0)
    parser.add_argument("--leaf-value-quant-bits", type=int, choices=(0, 4, 8), default=0)
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
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument("--prefill-local-length", type=int)
    parser.add_argument("--prefill-state-update-length", type=int)
    parser.add_argument("--split-prefill-local-attention", action="store_true")
    parser.add_argument(
        "--coarse-compact-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--leaf-attention-backend",
        choices=("packed", "paged"),
        default="packed",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--disable-fused-state-routing", action="store_true")
    parser.add_argument("--no-clone-decode-routes", action="store_true")
    parser.add_argument("--disable-fused-decode-state-route", action="store_true")
    parser.add_argument("--decode-route-group-size", type=int, default=16)
    parser.add_argument("--decode-route-num-warps", type=int, default=2)
    parser.add_argument("--decode-split-kv", type=int)
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
    parser.add_argument("--compare-state-routing", action="store_true")
    parser.add_argument(
        "--direct-fused-state-routing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--disable-shared-update-similarity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = load_text_model(
        args.checkpoint,
        args.mode,
        args.two_level_topk,
        args.state_growth_factor,
        device,
        args.leaf_attention_backend,
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
                module.recursive_page_lod = args.recursive_page_lod
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
                module.virtual_page_storage = args.virtual_page_storage
                if args.prefill_chunk_length is not None:
                    module.prefill_chunk_len = args.prefill_chunk_length
                if args.prefill_local_length is not None:
                    module.prefill_local_len = args.prefill_local_length
                if args.prefill_state_update_length is not None:
                    module.prefill_state_update_len = (
                        args.prefill_state_update_length
                    )
                module.split_prefill_local_attention = (
                    args.split_prefill_local_attention
                )
                module.coarse_compact_bias = args.coarse_compact_bias
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
                module.collect_dynamic_open_stats = (
                    not args.disable_dynamic_open_stats
                    and (
                        dynamic_prefill_top_p is not None
                        or dynamic_decode_top_p is not None
                        or args.dynamic_open_prefill_residual_mass is not None
                        or args.dynamic_open_decode_residual_mass is not None
                    )
                )
                module.fused_state_routing = not args.disable_fused_state_routing
                module.clone_decode_routes = not args.no_clone_decode_routes
                module.fused_decode_state_route = (
                    not args.disable_fused_decode_state_route
                )
                module.decode_route_group_size = args.decode_route_group_size
                module.decode_route_num_warps = args.decode_route_num_warps
                if args.decode_split_kv is not None:
                    module.decode_split_kv = args.decode_split_kv
                module.decode_use_dot = args.decode_use_dot
                if args.decode_route_use_dot is not None:
                    module.decode_route_use_dot = args.decode_route_use_dot
                if args.decode_route_gqa_grouped is not None:
                    module.decode_route_gqa_grouped = args.decode_route_gqa_grouped
                if args.direct_fused_state_routing is not None:
                    module.direct_fused_state_routing = args.direct_fused_state_routing
                module._lod_compare_state_routing = args.compare_state_routing
                module.reuse_state_update_similarity = (
                    not args.disable_shared_update_similarity
                )

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.mode}_rank_{rank:02d}.jsonl"
    with output_path.open("w") as handle, torch.inference_mode():
        for task in args.tasks:
            documents = generate_documents(task, args.checkpoint, args.length)
            if args.indices is None:
                selected_documents = documents[: args.samples]
            else:
                selected_indices = set(args.indices)
                selected_documents = [
                    doc for doc in documents if int(doc["index"]) in selected_indices
                ]
                missing_indices = selected_indices - {
                    int(doc["index"]) for doc in selected_documents
                }
                if missing_indices:
                    raise ValueError(
                        f"requested document indices are absent: {sorted(missing_indices)}"
                    )
            for doc in selected_documents[rank::world_size]:
                pop_qwen35_dynamic_open_statistics(model)
                prompt = doc["input"] + " " + doc["gen_prefix"]
                input_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device)
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(
                    generated[0, input_ids.size(1) :],
                    skip_special_tokens=True,
                )
                target = str(doc["outputs"][0])
                dynamic_open_statistics = pop_qwen35_dynamic_open_statistics(model)
                page_quantization_statistics = (
                    qwen35_page_quantization_statistics(model)
                )
                record = {
                    "checkpoint": args.checkpoint,
                    "mode": args.mode,
                    "task": task,
                    "index": int(doc["index"]),
                    "length": int(doc["max_length"]),
                    "input_tokens": int(input_ids.size(1)),
                    "two_level_topk": (
                        args.two_level_topk if args.mode == "two_level" else None
                    ),
                    "dynamic_open_top_p": (
                        args.dynamic_open_top_p
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_prefill_top_p": (
                        dynamic_prefill_top_p
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_prefill_residual_mass": (
                        args.dynamic_open_prefill_residual_mass
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_decode_top_p": (
                        dynamic_decode_top_p
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_decode_residual_mass": (
                        args.dynamic_open_decode_residual_mass
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_statistics": dynamic_open_statistics,
                    "page_quantization_statistics": page_quantization_statistics,
                    "dynamic_open_stats_enabled": (
                        not args.disable_dynamic_open_stats
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
                        args.recursive_page_lod
                        if args.mode == "two_level"
                        else None
                    ),
                    "recursive_page_block_n": (
                        args.recursive_page_block_n
                        if args.mode == "two_level"
                        else None
                    ),
                    "leaf_num_warps": (
                        args.leaf_num_warps if args.mode == "two_level" else None
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
                    "virtual_page_storage": (
                        args.virtual_page_storage if args.mode == "two_level" else None
                    ),
                    "state_growth_factor": (
                        args.state_growth_factor if args.mode == "two_level" else None
                    ),
                    "prefill_chunk_length": (
                        args.prefill_chunk_length
                        if args.mode == "two_level"
                        else None
                    ),
                    "prefill_local_length": (
                        args.prefill_local_length
                        if args.mode == "two_level"
                        else None
                    ),
                    "prefill_state_update_length": (
                        args.prefill_state_update_length
                        if args.mode == "two_level"
                        else None
                    ),
                    "split_prefill_local_attention": (
                        args.split_prefill_local_attention
                        if args.mode == "two_level"
                        else None
                    ),
                    "coarse_compact_bias": (
                        args.coarse_compact_bias
                        if args.mode == "two_level"
                        else None
                    ),
                    "leaf_attention_backend": (
                        args.leaf_attention_backend
                        if args.mode == "two_level"
                        else None
                    ),
                    "target": target,
                    "response": response,
                    "exact": target.lower() in response.lower(),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"rank={rank} mode={args.mode} task={task} "
                    f"index={doc['index']} tokens={input_ids.size(1)} "
                    f"exact={record['exact']}",
                    flush=True,
                )
                if args.compare_state_routing:
                    compared = mismatched = boundary_ties = 0
                    for module in model.modules():
                        if isinstance(module, Qwen3_5TwoLevelAttention):
                            compared += getattr(module, "_lod_route_compared_rows", 0)
                            mismatched += getattr(
                                module, "_lod_route_mismatched_rows", 0
                            )
                            boundary_ties += getattr(
                                module, "_lod_route_boundary_ties", 0
                            )
                            module._lod_route_compared_rows = 0
                            module._lod_route_mismatched_rows = 0
                            module._lod_route_boundary_ties = 0
                    print(
                        "route_compare "
                        f"rows={compared} mismatched={mismatched} "
                        f"boundary_ties={boundary_ties}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
