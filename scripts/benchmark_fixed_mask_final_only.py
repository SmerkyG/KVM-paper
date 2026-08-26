#!/usr/bin/env python3
"""Benchmark only the fixed-list masked decode attention and its reducer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.aiter_page1_attention import (
    kernel_page1_attention_3d_bias,
    kernel_page1_attention_3d_bias_fixed_mask,
)
from model.kernels.paged_leaf_attention import (
    _reduce_aiter_page1_segments_with_lse_kernel,
)


def elapsed_ms(call, repeats: int) -> float:
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=70_144)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--gqa", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--tile-size", type=int, choices=(16, 64), default=64)
    parser.add_argument("--segments", type=int, default=128)
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Cycle through this many disjoint K/V arenas per timed graph.",
    )
    parser.add_argument(
        "--kernel", choices=("fixed-mask", "page1"), default="fixed-mask"
    )
    parser.add_argument(
        "--index-order", choices=("contiguous", "random"), default="contiguous"
    )
    parser.add_argument(
        "--arena-length",
        type=int,
        help="Physical K/V rows per sequence; defaults to --length.",
    )
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    batch = args.batch_size
    kv_heads = args.kv_heads
    gqa = args.gqa
    head_dim = args.head_dim
    sequences = batch * kv_heads
    length = args.length
    tile_size = args.tile_size
    segments = args.segments
    layers = args.layers
    if layers <= 0:
        raise ValueError("--layers must be positive")
    block_count = math.ceil(length / tile_size)
    physical_length = args.arena_length or length
    if physical_length < length:
        raise ValueError("--arena-length cannot be shorter than --length")
    arena_length_per_layer = sequences * physical_length
    arena_length = layers * arena_length_per_layer
    state_capacity = 4352
    local_limit = 512
    sink_len = 0
    leaf_begin = local_limit + sink_len + state_capacity

    query = torch.randn(
        layers, sequences, gqa, head_dim, device=device, dtype=dtype
    )
    # Distinct K/V storage per sequence preserves the production HBM working
    # set instead of allowing every sequence to reuse the same cache lines.
    key = torch.randn(arena_length, head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    bias = torch.zeros(arena_length, device=device, dtype=torch.float16)
    sequence_offsets = (
        torch.arange(sequences, device=device, dtype=torch.int32)
        * physical_length
    )[:, None]
    if args.index_order == "contiguous":
        logical_indices = torch.arange(
            length, device=device, dtype=torch.int32
        )[None, :].expand(sequences, -1)
    else:
        logical_indices = torch.stack(
            [
                torch.randperm(physical_length, device=device)[:length]
                for _ in range(sequences)
            ]
        ).to(torch.int32)
    indices = sequence_offsets + logical_indices
    fixed_lengths = torch.full(
        (sequences,), length, device=device, dtype=torch.int32
    )
    cache_indices = torch.arange(batch, device=device, dtype=torch.int64)
    context_lens = fixed_lengths.clone()
    active_mask = torch.empty(
        sequences, length, device=device, dtype=torch.uint8
    )
    active_blocks = torch.empty(
        sequences, block_count, device=device, dtype=torch.uint8
    )
    segment_out = torch.empty(
        sequences, gqa, segments, head_dim, device=device, dtype=torch.float32
    )
    segment_max = torch.empty(
        sequences, gqa, segments, device=device, dtype=torch.float32
    )
    segment_sum = torch.empty_like(segment_max)
    output = torch.empty(
        sequences, gqa, head_dim, device=device, dtype=torch.float32
    )
    output_lse = torch.empty(
        sequences, gqa, device=device, dtype=torch.float32
    )

    def attention_layer(layer: int) -> None:
        layer_begin = layer * arena_length_per_layer
        layer_end = layer_begin + arena_length_per_layer
        layer_key = key[layer_begin:layer_end]
        layer_value = value[layer_begin:layer_end]
        layer_bias = bias[layer_begin:layer_end]
        if args.kernel == "page1":
            kernel_page1_attention_3d_bias[(sequences, 1, segments)](
                segment_out,
                segment_max,
                segment_sum,
                query[layer],
                layer_key,
                layer_value,
                layer_bias,
                indices,
                cache_indices,
                fixed_lengths,
                head_dim**-0.5,
                indices.stride(0),
                query[layer].stride(0),
                query[layer].stride(1),
                NUM_QUERY_HEADS=gqa,
                KV_HEADS=kv_heads,
                INDEX_BY_CACHE=False,
                TILE_SIZE=tile_size,
                HEAD_SIZE=head_dim,
                BLOCK_M=16,
                NUM_SEGMENTS=segments,
                num_warps=2,
                waves_per_eu=2,
                num_stages=2,
            )
        else:
            kernel_page1_attention_3d_bias_fixed_mask[
                (sequences, 1, segments)
            ](
                segment_out,
                segment_max,
                segment_sum,
                query[layer],
                layer_key,
                layer_value,
                layer_bias,
                indices,
                active_mask,
                active_blocks,
                fixed_lengths,
                cache_indices,
                head_dim**-0.5,
                indices.stride(0),
                active_mask.stride(0),
                active_blocks.stride(0),
                query[layer].stride(0),
                query[layer].stride(1),
                NUM_QUERY_HEADS=gqa,
                KV_HEADS=kv_heads,
                STATE_CAPACITY=state_capacity,
                LOCAL_LIMIT=local_limit,
                SINK_LEN=sink_len,
                LEAF_BEGIN=leaf_begin,
                TILE_SIZE=tile_size,
                HEAD_SIZE=head_dim,
                BLOCK_M=16,
                NUM_SEGMENTS=segments,
                INCLUDE_NEW=False,
                num_warps=2,
                waves_per_eu=2,
                num_stages=2,
            )

    def attention() -> None:
        for layer in range(layers):
            attention_layer(layer)

    def reduce() -> None:
        _reduce_aiter_page1_segments_with_lse_kernel[(sequences, gqa)](
            segment_out,
            segment_max,
            segment_sum,
            context_lens,
            output,
            output_lse,
            OUTPUT_STRIDE_0=output.stride(0),
            OUTPUT_STRIDE_1=output.stride(1),
            QUERY_ROWS=gqa,
            HEAD_DIM=head_dim,
            SEGMENTS=segments,
            TILE_SIZE=tile_size,
            num_warps=2,
            waves_per_eu=1,
        )

    def attention_and_reduce() -> None:
        for layer in range(layers):
            attention_layer(layer)
            reduce()

    results: dict[str, dict[str, float]] = {}
    token = torch.arange(length, device=device)
    token_block = torch.div(token, tile_size, rounding_mode="floor")
    block = torch.arange(block_count, device=device)
    for label, divisor in (("zero", 0), ("quarter", 4), ("all", 1)):
        if divisor == 0:
            block_values = torch.zeros_like(block, dtype=torch.uint8)
            lane_values = torch.zeros_like(token, dtype=torch.uint8)
        else:
            block_values = ((block % divisor) == 0).to(torch.uint8)
            lane_values = ((token_block % divisor) == 0).to(torch.uint8)
        active_blocks.copy_(block_values[None, :].expand_as(active_blocks))
        active_mask.copy_(lane_values[None, :].expand_as(active_mask))
        attention_ms = elapsed_ms(attention, args.repeats)
        total_ms = elapsed_ms(attention_and_reduce, args.repeats)
        results[label] = {
            "active_fraction": float(lane_values.float().mean().item()),
            "attention_ms": attention_ms,
            "attention_and_reduce_ms": total_ms,
            "reduction_increment_ms": total_ms - attention_ms,
            "attention_ms_per_layer": attention_ms / layers,
            "attention_and_reduce_ms_per_layer": total_ms / layers,
            "reduction_increment_ms_per_layer": (
                total_ms - attention_ms
            ) / layers,
        }

    payload = {
        "geometry": {
            "batch_size": batch,
            "kv_heads": kv_heads,
            "gqa": gqa,
            "head_dim": head_dim,
            "fixed_length": length,
            "physical_length": physical_length,
            "kernel": args.kernel,
            "index_order": args.index_order,
            "tile_size": tile_size,
            "segments": segments,
            "layers": layers,
            "arena_bytes": int((key.numel() + value.numel()) * key.element_size()),
            "repeats": args.repeats,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
