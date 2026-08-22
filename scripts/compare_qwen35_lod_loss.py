#!/usr/bin/env python3
"""Compare Qwen3.5 full attention and its inference-only LOD graft on ProLong."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model.qwen35_two_level_attention import (
    Qwen3_5TwoLevelAttention,
    pop_qwen35_dynamic_open_statistics,
    qwen35_page_quantization_statistics,
)
from scripts.probe_qwen35_lod_niah import load_text_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--mode", choices=("full", "two_level"), required=True)
    parser.add_argument("--dataset", default="Seerkfang/prolong-64k-512-new")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--decode-tail-tokens", type=int, default=0)
    parser.add_argument("--two-level-topk", type=int, default=8)
    parser.add_argument(
        "--exclude-sink-from-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--separate-sink-cache", action="store_true")
    parser.add_argument("--prefill-two-level-topk", type=int, default=3)
    parser.add_argument("--prefill-max-leaf-tokens", type=int)
    parser.add_argument("--dynamic-open-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-top-p", type=float)
    parser.add_argument("--dynamic-open-prefill-residual-mass", type=float)
    parser.add_argument("--dynamic-open-decode-top-p", type=float)
    parser.add_argument("--recursive-page-lod", action="store_true")
    parser.add_argument(
        "--all-centroid-top1",
        action="store_true",
        help="Use the experimental uniform exact-winner/residual path.",
    )
    parser.add_argument("--all-centroid-mean-residual", action="store_true")
    parser.add_argument("--recursive-page-block-n", type=int, default=4)
    parser.add_argument("--leaf-num-warps", type=int, default=1)
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
    parser.add_argument("--overflow-bipartite-merge", action="store_true")
    parser.add_argument("--overflow-bipartite-block-size", type=int, default=32)
    parser.add_argument("--overflow-bipartite-positional-halves", action="store_true")
    parser.add_argument("--overflow-bipartite-keep-ratio", type=float, default=0.5)
    parser.add_argument("--merge-before-append", action="store_true")
    parser.add_argument("--append-subblock-size", type=int, default=0)
    parser.add_argument("--union-bipartite-state", action="store_true")
    parser.add_argument("--state-precompact-direct-append", action="store_true")
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
        "--leaf-attention-backend",
        choices=("packed", "paged"),
        default="packed",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def select_sequences(
    tokenizer,
    dataset_name: str,
    sequence_length: int,
    samples: int,
    rank: int,
    world_size: int,
) -> list[tuple[int, torch.Tensor]]:
    # A bounded deterministic shuffle avoids materializing a global index for
    # the 100+ GB local ProLong dataset merely to select a small paired sample.
    dataset = load_dataset(dataset_name, split="train", streaming=True).shuffle(
        seed=42, buffer_size=1_000
    )
    selected: list[tuple[int, torch.Tensor]] = []
    selected_count = 0
    for document in dataset:
        token_count = document.get("length")
        if token_count is not None and int(token_count) < sequence_length:
            continue
        input_ids = tokenizer(
            document["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length,
            return_attention_mask=False,
        )["input_ids"]
        if len(input_ids) != sequence_length:
            continue
        sample = selected_count
        if sample % world_size == rank:
            selected.append((sample, torch.tensor(input_ids, dtype=torch.long)))
        selected_count += 1
        if selected_count == samples:
            break
    if selected_count != samples:
        raise RuntimeError(f"found only {selected_count} sufficiently long documents")
    return selected


def main() -> None:
    args = parse_args()
    if args.prefill_two_level_topk is not None and not (
        0 <= args.prefill_two_level_topk <= 8
    ):
        raise ValueError("prefill top-k must be in [0, 8]")
    if args.prefill_max_leaf_tokens is not None and args.prefill_max_leaf_tokens <= 0:
        raise ValueError("maximum prefill leaf count must be positive")
    if args.decode_tail_tokens < 0 or args.decode_tail_tokens >= args.sequence_length:
        raise ValueError("decode tail must be nonnegative and shorter than the sequence")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    sequences = select_sequences(
        tokenizer,
        args.dataset,
        args.sequence_length,
        args.samples,
        rank,
        world_size,
    )
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
                module.recursive_page_lod = args.recursive_page_lod
                module.all_centroid_top1 = args.all_centroid_top1
                module.all_centroid_disjoint_residual = not (
                    args.all_centroid_mean_residual
                )
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
                module.dynamic_open_prefill_top_p = dynamic_prefill_top_p
                module.dynamic_open_prefill_residual_mass = (
                    args.dynamic_open_prefill_residual_mass
                )
                module.dynamic_open_decode_top_p = dynamic_decode_top_p
                module.collect_dynamic_open_stats = (
                    dynamic_prefill_top_p is not None
                    or dynamic_decode_top_p is not None
                    or args.dynamic_open_prefill_residual_mass is not None
                )

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.mode}_rank_{rank:02d}.jsonl"
    with output_path.open("w") as handle, torch.inference_mode():
        for sample, sequence in sequences:
            pop_qwen35_dynamic_open_statistics(model)
            input_ids = sequence.unsqueeze(0).to(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if args.decode_tail_tokens:
                prefix = input_ids[..., : -args.decode_tail_tokens]
                tail = input_ids[..., -args.decode_tail_tokens :]
                result = model(input_ids=prefix, use_cache=True)
                losses = [
                    F.cross_entropy(result.logits[:, -1].float(), tail[:, 0])
                ]
                past_key_values = result.past_key_values
                for tail_index in range(args.decode_tail_tokens - 1):
                    result = model(
                        input_ids=tail[:, tail_index : tail_index + 1],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = result.past_key_values
                    losses.append(
                        F.cross_entropy(
                            result.logits[:, -1].float(),
                            tail[:, tail_index + 1],
                        )
                    )
                loss = torch.stack(losses).mean()
                prediction_tokens = args.decode_tail_tokens
            else:
                result = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                )
                loss = result.loss
                prediction_tokens = args.sequence_length - 1
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - started
            dynamic_open_statistics = pop_qwen35_dynamic_open_statistics(model)
            page_quantization_statistics = qwen35_page_quantization_statistics(model)
            record = {
                "checkpoint": args.checkpoint,
                "mode": args.mode,
                "sample": sample,
                "tokens": args.sequence_length,
                "prediction_tokens": prediction_tokens,
                "decode_tail_tokens": args.decode_tail_tokens,
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
                "dynamic_open_statistics": dynamic_open_statistics,
                "page_quantization_statistics": page_quantization_statistics,
                "recursive_page_lod": (
                    args.recursive_page_lod if args.mode == "two_level" else None
                ),
                "all_centroid_top1": (
                    args.all_centroid_top1 if args.mode == "two_level" else None
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
                    args.union_bipartite_state
                    if args.mode == "two_level"
                    else None
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
                "leaf_attention_backend": (
                    args.leaf_attention_backend
                    if args.mode == "two_level"
                    else None
                ),
                "loss": float(loss.item()),
                "elapsed_seconds": elapsed_seconds,
                "tokens_per_second": args.sequence_length / elapsed_seconds,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"rank={rank} mode={args.mode} sample={sample} "
                f"loss={record['loss']:.6f} "
                f"seconds={elapsed_seconds:.3f} "
                f"tokens_per_second={record['tokens_per_second']:.1f}",
                flush=True,
            )
            del result, input_ids, loss


if __name__ == "__main__":
    main()
