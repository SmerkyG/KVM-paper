#!/usr/bin/env python3
"""Correctness and speed probe for the single-launch N=256 LOD decoder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from model.kernels.unified_lod_decode import (
    new_producer_consumer_lod_decode_buffers,
    new_unified_lod_decode_buffers,
    producer_consumer_lod_decode,
    unified_lod_decode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--state-len", type=int, default=4352)
    parser.add_argument("--local-len", type=int, default=512)
    parser.add_argument("--leaf-capacity", type=int, default=65536)
    parser.add_argument("--selected-centroids", type=int, default=8)
    parser.add_argument("--selected-leaves", type=int, default=128)
    parser.add_argument("--default-leaves", type=int, default=15)
    parser.add_argument("--mass-fraction", type=float, default=1 / 16)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--consumer-segments", type=int, nargs="+", default=[4, 8, 12, 16]
    )
    parser.add_argument("--sequential-consumer-debug", action="store_true")
    parser.add_argument("--producer-priority", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def time_cuda(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end)) * 1000.0 / repeats


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch = args.batch_size
    kv_heads = args.kv_heads
    gqa = 4
    query_heads = kv_heads * gqa
    head_dim = 256
    cache_batches = batch
    state_len = args.state_len
    state_capacity = state_len
    local_capacity = args.local_len + 1
    sink_capacity = 0
    local_len = args.local_len
    page_size = 16
    scale = head_dim**-0.5

    slot_lengths_cpu = torch.full(
        (cache_batches, kv_heads, state_capacity),
        args.default_leaves,
        dtype=torch.int32,
    )
    slot_lengths_cpu[:, :, : args.selected_centroids] = args.selected_leaves
    required_leaves = int(slot_lengths_cpu[0, 0].sum())
    leaf_capacity = max(args.leaf_capacity, required_leaves)
    max_slot_pages = math.ceil(args.selected_leaves / page_size)
    root_capacity = math.ceil(max_slot_pages / 64)
    pages_per_row = sum(
        math.ceil(int(length) / page_size)
        for length in slot_lengths_cpu[0, 0]
    )
    page_capacity = pages_per_row
    directory_capacity = state_capacity

    kv_rows = cache_batches * kv_heads
    leaf_offset = 0
    local_offset = kv_rows * leaf_capacity
    sink_offset = local_offset + kv_rows * local_capacity
    coarse_offset = sink_offset + kv_rows * sink_capacity
    arena_capacity = coarse_offset + kv_rows * state_capacity
    arena_k = torch.randn(
        arena_capacity, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.05
    arena_v = torch.randn_like(arena_k) * 0.05
    arena_bias = torch.zeros(arena_capacity, dtype=torch.float16, device=device)

    q = torch.randn(
        batch, query_heads, 1, head_dim, dtype=torch.bfloat16, device=device
    )
    new_k = torch.randn(
        batch, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.05
    new_v = torch.randn_like(new_k) * 0.05
    counts = slot_lengths_cpu.to(torch.float32).unsqueeze(-1).to(device)
    slot_lengths = slot_lengths_cpu.to(device)
    cache_indices = torch.arange(batch, dtype=torch.int64, device=device)
    local_lens = torch.full(
        (cache_batches,), local_len, dtype=torch.int32, device=device
    )

    slot_pages = torch.full(
        (cache_batches, kv_heads, state_capacity, root_capacity),
        -1,
        dtype=torch.int32,
    )
    directory_values = torch.full(
        (cache_batches, kv_heads, directory_capacity, 64),
        -1,
        dtype=torch.int32,
    )
    page_indices = torch.full(
        (cache_batches, kv_heads, page_capacity, page_size),
        -1,
        dtype=torch.int32,
    )
    expected_leaf_indices: list[list[list[int]]] = []
    for cache_batch in range(cache_batches):
        batch_rows: list[list[int]] = []
        for kv_head in range(kv_heads):
            page = 0
            leaf = 0
            row_slots: list[list[int]] = []
            for slot in range(state_capacity):
                length = int(slot_lengths_cpu[cache_batch, kv_head, slot])
                directory = slot
                slot_pages[cache_batch, kv_head, slot, 0] = directory
                slot_leaves = list(range(leaf, leaf + length))
                row_slots.append(slot_leaves)
                for ordinal in range(math.ceil(length / page_size)):
                    directory_values[
                        cache_batch, kv_head, directory, ordinal
                    ] = page
                    begin = ordinal * page_size
                    values = slot_leaves[begin : begin + page_size]
                    page_indices[
                        cache_batch, kv_head, page, : len(values)
                    ] = torch.tensor(values, dtype=torch.int32)
                    page += 1
                leaf += length
            batch_rows.append(row_slots)
        expected_leaf_indices.append(batch_rows)
    slot_pages = slot_pages.to(device)
    directory_values = directory_values.to(device)
    page_indices = page_indices.to(device)

    # Materialize coherent centroid means. The selected prefix is aligned to
    # query head zero, while all other centroid keys are zero. A fixed score
    # threshold therefore opens exactly the requested prefix on every row.
    for cache_batch in range(cache_batches):
        for kv_head in range(kv_heads):
            kv_row = cache_batch * kv_heads + kv_head
            coarse_base = coarse_offset + kv_row * state_capacity
            arena_k[coarse_base : coarse_base + state_capacity].zero_()
            selected_query = q[cache_batch, kv_head * gqa, 0]
            arena_k[
                coarse_base : coarse_base + args.selected_centroids
            ] = selected_query
            row_counts = counts[cache_batch, kv_head, :, 0]
            arena_bias[
                coarse_base : coarse_base + state_capacity
            ] = torch.log(row_counts).to(torch.float16)
    effective_threshold = 10.0
    previous_lse = torch.full(
        (cache_batches, query_heads),
        effective_threshold - math.log(args.mass_fraction),
        dtype=torch.float32,
        device=device,
    )

    index_capacity = leaf_capacity + local_capacity + state_capacity
    buffers = new_unified_lod_decode_buffers(
        sequences=batch * kv_heads,
        index_capacity=index_capacity,
        device=device,
    )
    output = torch.empty_like(q)

    def run_mode(execution_mode: int) -> None:
        # Timing iterations retain the preceding output LSE, as production
        # decode does. The sparse selection remains stable by construction.
        unified_lod_decode(
            q,
            new_k,
            new_v,
            cache_indices,
            local_lens,
            counts,
            slot_pages,
            directory_values,
            slot_lengths,
            page_indices,
            arena_k,
            arena_v,
            arena_bias,
            previous_lse,
            output,
            buffers,
            state_len=state_len,
            local_limit=local_len,
            sink_len=0,
            protected_len=0,
            max_leaf_tokens=1024,
            open_capacity=max(128, args.selected_centroids * 2),
            leaf_offset=leaf_offset,
            local_offset=local_offset,
            sink_offset=sink_offset,
            coarse_offset=coarse_offset,
            scale=scale,
            mass_fraction=args.mass_fraction,
            execution_mode=execution_mode,
        )

    def run() -> None:
        run_mode(0)

    retained_seed = previous_lse.clone()
    run()
    torch.cuda.synchronize()
    observed = output.float().clone()
    first_lse = previous_lse.clone()

    references = torch.empty_like(observed)
    for logical_batch in range(batch):
        for kv_head in range(kv_heads):
            kv_row = logical_batch * kv_heads + kv_head
            physical: list[int] = []
            for slot in range(state_capacity):
                if slot < args.selected_centroids:
                    physical.extend(
                        leaf_offset
                        + kv_row * leaf_capacity
                        + leaf
                        for leaf in expected_leaf_indices[logical_batch][kv_head][slot]
                    )
                else:
                    physical.append(coarse_offset + kv_row * state_capacity + slot)
            physical.extend(
                local_offset + kv_row * local_capacity + token
                for token in range(local_len + 1)
            )
            index = torch.tensor(physical, dtype=torch.long, device=device)
            keys = arena_k[index].float()
            values = arena_v[index].float()
            bias = arena_bias[index].float()
            queries = q[
                logical_batch, kv_head * gqa : (kv_head + 1) * gqa, 0
            ].float()
            score = queries @ keys.T * scale + bias
            reference = torch.softmax(score, dim=-1) @ values
            references[
                logical_batch, kv_head * gqa : (kv_head + 1) * gqa, 0
            ] = reference
    difference = observed - references
    correctness = {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                observed.flatten(), references.flatten(), dim=0
            )
        ),
        "finite": bool(torch.isfinite(observed).all()),
        "first_lse_finite": bool(torch.isfinite(first_lse).all()),
    }
    if not correctness["finite"] or correctness["max_abs"] > 0.05:
        raise AssertionError(f"unified LOD decode is incorrect: {correctness}")

    previous_lse.copy_(retained_seed)
    latency_us = time_cuda(run, args.warmup, args.repeats)
    producer_stream = (
        torch.cuda.current_stream()
        if args.sequential_consumer_debug
        else torch.cuda.Stream(priority=args.producer_priority)
    )
    producer_ready_event = torch.cuda.Event()
    producer_consumer_results: dict[str, dict[str, object]] = {}
    for consumer_segments in args.consumer_segments:
        pc_buffers = new_producer_consumer_lod_decode_buffers(
            sequences=batch * kv_heads,
            index_capacity=index_capacity,
            query_heads_per_kv=gqa,
            consumer_segments=consumer_segments,
            device=device,
        )
        pc_output = torch.empty_like(q)

        def run_pc() -> None:
            producer_consumer_lod_decode(
                q,
                new_k,
                new_v,
                cache_indices,
                local_lens,
                counts,
                slot_pages,
                directory_values,
                slot_lengths,
                page_indices,
                arena_k,
                arena_v,
                arena_bias,
                previous_lse,
                pc_output,
                pc_buffers,
                producer_stream,
                producer_ready_event,
                state_len=state_len,
                local_limit=local_len,
                sink_len=0,
                protected_len=0,
                max_leaf_tokens=1024,
                open_capacity=max(128, args.selected_centroids * 2),
                leaf_offset=leaf_offset,
                local_offset=local_offset,
                sink_offset=sink_offset,
                coarse_offset=coarse_offset,
                scale=scale,
                mass_fraction=args.mass_fraction,
                consumer_segments=consumer_segments,
            )

        previous_lse.copy_(retained_seed)
        run_pc()
        torch.cuda.synchronize()
        pc_difference = pc_output.float() - references
        pc_correctness = {
            "max_abs": float(pc_difference.abs().max()),
            "mean_abs": float(pc_difference.abs().mean()),
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    pc_output.float().flatten(), references.flatten(), dim=0
                )
            ),
            "finite": bool(torch.isfinite(pc_output).all()),
        }
        if not pc_correctness["finite"] or pc_correctness["max_abs"] > 0.05:
            raise AssertionError(
                f"producer/consumer decode is incorrect: {pc_correctness}"
            )
        previous_lse.copy_(retained_seed)
        pc_latency = time_cuda(run_pc, args.warmup, args.repeats)
        producer_consumer_results[str(consumer_segments)] = {
            "latency_us": pc_latency,
            "correctness": pc_correctness,
            "scratch_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in pc_buffers.values()
            ),
        }
    result = {
        "device": torch.cuda.get_device_name(),
        "batch_size": batch,
        "kv_heads": kv_heads,
        "gqa": gqa,
        "head_dim": head_dim,
        "state_len": state_len,
        "local_len": local_len,
        "leaf_capacity": leaf_capacity,
        "selected_centroids": args.selected_centroids,
        "selected_leaves": args.selected_leaves,
        "logical_tile_n": 128,
        "latency_us": latency_us,
        "producer_consumer": producer_consumer_results,
        "correctness": correctness,
        "scratch_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in buffers.values()
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
