#!/usr/bin/env python3
"""Paired NIAH evaluation of Qwen3.5 full and top-k LOD attention."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_modeling
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen35_moe_modeling
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from model.qwen35_two_level_attention import (
    Qwen3_5TwoLevelAttention,
    graft_qwen35_two_level_attention,
    pop_qwen35_dynamic_open_statistics,
    qwen35_page_quantization_statistics,
)


def enable_fla_fast_path(*, required: bool = False) -> bool:
    """Install FLA callables at the Qwen3.5 hook points used by Transformers."""
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
                "project's `qwen35-fast-path` extra"
            ) from error
        return False

    def compatible(function):
        parameters = frozenset(inspect.signature(function).parameters)

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            return function(
                *args,
                **{name: value for name, value in kwargs.items() if name in parameters},
            )

        return wrapped

    chunk_gated_delta_rule = compatible(chunk_gated_delta_rule)
    fused_recurrent_gated_delta_rule = compatible(fused_recurrent_gated_delta_rule)
    for modeling, norm_name in (
        (qwen35_modeling, "Qwen3_5RMSNormGated"),
        (qwen35_moe_modeling, "Qwen3_5MoeRMSNormGated"),
    ):
        setattr(modeling, norm_name, FusedRMSNormGated)
        modeling.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
        modeling.torch_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    return True


def require_qwen35_acceleration(model: torch.nn.Module) -> dict[str, bool | int]:
    """Verify the FLA and causal-convolution callables used by speed runs."""

    def uses_module(function, prefix: str) -> bool:
        if getattr(function, "__module__", "").startswith(prefix):
            return True
        return any(
            getattr(cell.cell_contents, "__module__", "").startswith(prefix)
            for cell in (getattr(function, "__closure__", None) or ())
        )

    linear_layers = [
        module
        for module in model.modules()
        if module.__class__.__name__.endswith("GatedDeltaNet")
    ]
    modeling = (
        qwen35_moe_modeling
        if any(
            module.__class__.__module__.startswith("transformers.models.qwen3_5_moe.")
            for module in linear_layers
        )
        else qwen35_modeling
    )
    has_linear_layers = bool(linear_layers)
    acceleration: dict[str, bool | int] = {
        "linear_layer_count": len(linear_layers),
        "fla_gated_delta_rule": has_linear_layers
        and uses_module(modeling.torch_chunk_gated_delta_rule, "fla."),
        "fla_recurrent_gated_delta_rule": has_linear_layers
        and uses_module(modeling.torch_recurrent_gated_delta_rule, "fla."),
        "fla_fused_rms_norm_gated": has_linear_layers
        and all(
            module.norm.__class__.__module__.startswith("fla.")
            for module in linear_layers
        ),
        "causal_conv1d_prefill": has_linear_layers
        and uses_module(modeling.causal_conv1d_fn, "causal_conv1d"),
        "causal_conv1d_decode": has_linear_layers
        and uses_module(modeling.causal_conv1d_update, "causal_conv1d"),
    }
    required = (
        "fla_gated_delta_rule",
        "fla_recurrent_gated_delta_rule",
        "fla_fused_rms_norm_gated",
        "causal_conv1d_prefill",
        "causal_conv1d_decode",
    )
    missing = [name for name in required if not acceleration[name]]
    if missing:
        raise RuntimeError(
            "Qwen3.5 benchmark is missing required acceleration: "
            + ", ".join(missing)
            + "; install the project's `qwen35-fast-path` extra"
        )
    return acceleration


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
    dataset = generator(max_seq_lengths=[length], pretrained=checkpoint)["test"]
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
    parser.add_argument(
        "--exclude-sink-from-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--separate-sink-cache", action="store_true")
    parser.add_argument("--prefill-two-level-topk", type=int, default=3)
    parser.add_argument("--prefill-max-leaf-tokens", type=int)
    parser.add_argument("--leaf-seal-capacity", type=int)
    parser.add_argument("--dynamic-open-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-residual-mass", type=float)
    parser.add_argument("--dynamic-open-decode-top-p", type=float)
    parser.add_argument("--dynamic-open-decode-residual-mass", type=float)
    parser.add_argument("--disable-dynamic-open-stats", action="store_true")
    parser.add_argument("--reuse-dynamic-local-attention", action="store_true")
    parser.add_argument("--dynamic-open-residual-state-bound", action="store_true")
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument("--dense-page-prefill", action="store_true")
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
    parser.add_argument("--recursive-page-block-n", type=int, default=4)
    parser.add_argument("--leaf-num-warps", type=int, default=1)
    parser.add_argument(
        "--leaf-layout",
        choices=("expert", "expert_tiny", "query"),
        default="query",
    )
    parser.add_argument("--tiny-expert-max", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--tiny-max-context", type=int, default=65_536)
    parser.add_argument("--tiny-block-m", type=int, default=8)
    parser.add_argument("--tiny-num-warps", type=int, default=1)
    parser.add_argument("--reduce-num-warps", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--prefill-int8-leaf-mma", action="store_true")
    parser.add_argument("--prefill-int8-coarse-mma", action="store_true")
    parser.add_argument(
        "--prefill-int8-pv-mma",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--leaf-key-quant-bits", type=int, choices=(0, 4, 8), default=0)
    parser.add_argument(
        "--leaf-value-quant-bits", type=int, choices=(0, 4, 8), default=0
    )
    parser.add_argument("--leaf-quant-group-size", type=int, default=32)
    parser.add_argument("--leaf-quant-scale-mode", choices=("max", "l2"), default="max")
    parser.add_argument(
        "--leaf-append-quant-scale-mode", choices=("max", "l2"), default="max"
    )
    parser.add_argument(
        "--page-summary-quant-bits", type=int, choices=(0, 8), default=8
    )
    parser.add_argument(
        "--page-summary-scale-mode", choices=("max", "l2"), default="l2"
    )
    parser.add_argument("--virtual-page-storage", action="store_true")
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument(
        "--state-clustering-geometry",
        choices=("raw", "coherence"),
        default="raw",
    )
    parser.add_argument(
        "--exact-coherence-matmul",
        action="store_true",
        help="Use the slower two-GEMM BF16 reference for coherence routing.",
    )
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument("--prefill-local-length", type=int)
    parser.add_argument("--prefill-state-update-length", type=int)
    parser.add_argument("--overflow-bipartite-merge", action="store_true")
    parser.add_argument("--overflow-bipartite-block-size", type=int, default=32)
    parser.add_argument("--overflow-bipartite-positional-halves", action="store_true")
    parser.add_argument("--overflow-bipartite-keep-ratio", type=float, default=0.5)
    parser.add_argument("--merge-before-append", action="store_true")
    parser.add_argument("--append-subblock-size", type=int, default=0)
    parser.add_argument("--union-bipartite-state", action="store_true")
    parser.add_argument("--state-precompact-direct-append", action="store_true")
    parser.add_argument(
        "--leaf-paged-directory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--split-prefill-local-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-prefill-route-coarse",
        "--enable-fused-prefill-route-coarse",
        dest="enable_fused_prefill_route_coarse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--decode-route-group-size", type=int, default=32)
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
    if args.prefill_two_level_topk is not None and not (
        0 <= args.prefill_two_level_topk <= 8
    ):
        raise ValueError("prefill top-k must be in [0, 8]")
    if args.prefill_max_leaf_tokens is not None and args.prefill_max_leaf_tokens <= 0:
        raise ValueError("maximum prefill leaf count must be positive")
    if args.leaf_seal_capacity is not None and args.leaf_seal_capacity <= 0:
        raise ValueError("leaf seal capacity must be positive")
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
                module.prefill_two_level_topk = args.prefill_two_level_topk
                module.exclude_sink_from_routes = args.exclude_sink_from_routes
                module.separate_sink_cache = args.separate_sink_cache
                module.prefill_max_leaf_tokens = args.prefill_max_leaf_tokens
                module.leaf_seal_capacity = args.leaf_seal_capacity
                module.leaf_paged_directory = args.leaf_paged_directory
                module.recursive_page_lod = args.recursive_page_lod
                module.dense_page_prefill = args.dense_page_prefill
                module.dense_page_topk = args.dense_page_topk
                module.dense_page_block_m = args.dense_page_block_m
                module.dense_page_block_n = args.dense_page_block_n
                module.dense_page_indexed_aiter_union = (
                    args.dense_page_indexed_aiter_union
                )
                module.dense_page_union_query_tile = args.dense_page_union_query_tile
                module.recursive_page_block_n = args.recursive_page_block_n
                module.leaf_num_warps = args.leaf_num_warps
                module.leaf_layout = args.leaf_layout
                module.leaf_tiny_expert_max = args.tiny_expert_max
                module.leaf_tiny_max_context = args.tiny_max_context
                module.leaf_tiny_block_m = args.tiny_block_m
                module.leaf_tiny_num_warps = args.tiny_num_warps
                module.leaf_reduce_num_warps = args.reduce_num_warps
                module.prefill_int8_leaf_mma = args.prefill_int8_leaf_mma
                module.prefill_int8_coarse_mma = args.prefill_int8_coarse_mma
                module.prefill_int8_pv_mma = args.prefill_int8_pv_mma
                module.leaf_key_quant_bits = args.leaf_key_quant_bits
                module.leaf_value_quant_bits = args.leaf_value_quant_bits
                module.leaf_quant_group_size = args.leaf_quant_group_size
                module.leaf_quant_scale_mode = args.leaf_quant_scale_mode
                module.leaf_append_quant_scale_mode = args.leaf_append_quant_scale_mode
                module.page_summary_quant_bits = args.page_summary_quant_bits
                module.page_summary_scale_mode = args.page_summary_scale_mode
                module.virtual_page_storage = args.virtual_page_storage
                module.state_clustering_normalization = "none"
                module.state_clustering_centroid_rescale = (
                    "coherence"
                    if args.state_clustering_geometry == "coherence"
                    else "none"
                )
                module.state_clustering_centroid_rescale_scope = (
                    "assignment"
                    if args.state_clustering_geometry == "coherence"
                    else "all"
                )
                module.coherence_single_matmul = not args.exact_coherence_matmul
                if args.prefill_chunk_length is not None:
                    module.prefill_chunk_len = args.prefill_chunk_length
                if args.prefill_local_length is not None:
                    module.prefill_local_len = args.prefill_local_length
                if args.prefill_state_update_length is not None:
                    module.prefill_state_update_len = args.prefill_state_update_length
                module.overflow_bipartite_merge = args.overflow_bipartite_merge
                module.overflow_bipartite_block_size = (
                    args.overflow_bipartite_block_size
                )
                module.overflow_bipartite_positional_halves = (
                    args.overflow_bipartite_positional_halves
                )
                module.overflow_bipartite_keep_ratio = (
                    args.overflow_bipartite_keep_ratio
                )
                module.state_merge_before_append = args.merge_before_append
                module.state_append_subblock_size = args.append_subblock_size
                module.state_union_bipartite = args.union_bipartite_state
                module.state_precompact_direct_append = (
                    args.state_precompact_direct_append
                )
                module.split_prefill_local_attention = (
                    args.split_prefill_local_attention
                )
                module.fused_prefill_route_coarse = (
                    args.enable_fused_prefill_route_coarse
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
                page_quantization_statistics = qwen35_page_quantization_statistics(
                    model
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
                    "exclude_sink_from_routes": (
                        args.exclude_sink_from_routes
                        if args.mode == "two_level"
                        else None
                    ),
                    "separate_sink_cache": (
                        args.separate_sink_cache if args.mode == "two_level" else None
                    ),
                    "prefill_two_level_topk": (
                        args.prefill_two_level_topk
                        if args.mode == "two_level"
                        else None
                    ),
                    "prefill_max_leaf_tokens": (
                        args.prefill_max_leaf_tokens
                        if args.mode == "two_level"
                        else None
                    ),
                    "leaf_seal_capacity": (
                        args.leaf_seal_capacity if args.mode == "two_level" else None
                    ),
                    "leaf_paged_directory": (
                        args.leaf_paged_directory if args.mode == "two_level" else None
                    ),
                    "fused_prefill_route_coarse": (
                        args.enable_fused_prefill_route_coarse
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_top_p": (
                        args.dynamic_open_top_p if args.mode == "two_level" else None
                    ),
                    "dynamic_open_prefill_top_p": (
                        dynamic_prefill_top_p if args.mode == "two_level" else None
                    ),
                    "dynamic_open_prefill_residual_mass": (
                        args.dynamic_open_prefill_residual_mass
                        if args.mode == "two_level"
                        else None
                    ),
                    "dynamic_open_decode_top_p": (
                        dynamic_decode_top_p if args.mode == "two_level" else None
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
                        args.recursive_page_lod if args.mode == "two_level" else None
                    ),
                    "dense_page_prefill": (
                        args.dense_page_prefill if args.mode == "two_level" else None
                    ),
                    "dense_page_topk": (
                        args.dense_page_topk if args.mode == "two_level" else None
                    ),
                    "dense_page_block_m": (
                        args.dense_page_block_m if args.mode == "two_level" else None
                    ),
                    "dense_page_block_n": (
                        args.dense_page_block_n if args.mode == "two_level" else None
                    ),
                    "dense_page_indexed_aiter_union": (
                        args.dense_page_indexed_aiter_union
                        if args.mode == "two_level"
                        else None
                    ),
                    "dense_page_union_query_tile": (
                        args.dense_page_union_query_tile
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
                    "leaf_layout": (
                        args.leaf_layout if args.mode == "two_level" else None
                    ),
                    "tiny_expert_max": (
                        args.tiny_expert_max
                        if args.mode == "two_level" and args.leaf_layout == "expert_tiny"
                        else None
                    ),
                    "tiny_max_context": (
                        args.tiny_max_context
                        if args.mode == "two_level" and args.leaf_layout == "expert_tiny"
                        else None
                    ),
                    "reduce_num_warps": (
                        args.reduce_num_warps if args.mode == "two_level" else None
                    ),
                    "prefill_int8_leaf_mma": (
                        args.prefill_int8_leaf_mma if args.mode == "two_level" else None
                    ),
                    "prefill_int8_coarse_mma": (
                        args.prefill_int8_coarse_mma
                        if args.mode == "two_level"
                        else None
                    ),
                    "prefill_int8_coarse_block_n": (
                        Qwen3_5TwoLevelAttention.prefill_int8_coarse_block_n
                        if args.mode == "two_level" and args.prefill_int8_coarse_mma
                        else None
                    ),
                    "prefill_int8_coarse_num_warps": (
                        Qwen3_5TwoLevelAttention.prefill_int8_coarse_num_warps
                        if args.mode == "two_level" and args.prefill_int8_coarse_mma
                        else None
                    ),
                    "prefill_int8_pv_mma": (
                        args.prefill_int8_pv_mma if args.mode == "two_level" else None
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
                        args.page_summary_quant_bits
                        if args.mode == "two_level"
                        else None
                    ),
                    "page_summary_scale_mode": (
                        args.page_summary_scale_mode
                        if args.mode == "two_level"
                        else None
                    ),
                    "virtual_page_storage": (
                        args.virtual_page_storage if args.mode == "two_level" else None
                    ),
                    "state_growth_factor": (
                        args.state_growth_factor if args.mode == "two_level" else None
                    ),
                    "prefill_chunk_length": (
                        args.prefill_chunk_length if args.mode == "two_level" else None
                    ),
                    "prefill_local_length": (
                        args.prefill_local_length if args.mode == "two_level" else None
                    ),
                    "prefill_state_update_length": (
                        args.prefill_state_update_length
                        if args.mode == "two_level"
                        else None
                    ),
                    "overflow_bipartite_merge": (
                        args.overflow_bipartite_merge
                        if args.mode == "two_level"
                        else None
                    ),
                    "overflow_bipartite_block_size": (
                        args.overflow_bipartite_block_size
                        if args.mode == "two_level"
                        else None
                    ),
                    "overflow_bipartite_positional_halves": (
                        args.overflow_bipartite_positional_halves
                        if args.mode == "two_level"
                        else None
                    ),
                    "overflow_bipartite_keep_ratio": (
                        args.overflow_bipartite_keep_ratio
                        if args.mode == "two_level"
                        else None
                    ),
                    "merge_before_append": (
                        args.merge_before_append if args.mode == "two_level" else None
                    ),
                    "append_subblock_size": (
                        args.append_subblock_size if args.mode == "two_level" else None
                    ),
                    "union_bipartite_state": (
                        args.union_bipartite_state if args.mode == "two_level" else None
                    ),
                    "state_precompact_direct_append": (
                        args.state_precompact_direct_append
                        if args.mode == "two_level"
                        else None
                    ),
                    "split_prefill_local_attention": (
                        args.split_prefill_local_attention
                        if args.mode == "two_level"
                        else None
                    ),
                    "coarse_compact_bias": (
                        args.coarse_compact_bias if args.mode == "two_level" else None
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
