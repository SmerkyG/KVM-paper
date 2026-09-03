#!/usr/bin/env python3
"""Isolate page choice from exact-page/residual attention in three-tier decode.

The production recursive decode kernel combines these operations.  This probe
times two calls with identical selected pages and outputs:

* ``scan`` exposes N pages in every selected centroid and chooses the final
  (optionally underfull) page from an already-materialized score table.
* ``preselected`` exposes only that same winning page to the page selector,
  while retaining the original centroid count and sums.  It therefore keeps
  exact leaf attention, centroid-minus-page residual correction, and output/LSE
  generation, but removes the variable-length page-list scan.

The scan-minus-preselected delta is the incremental cost of page choice inside
the existing fused kernel.  Uniform page-list lengths make the scaling explicit
instead of hiding it behind a synthetic occupancy distribution.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from model.kernels.paged_leaf_attention import (
    query_major_indexed_residual_page_attention,
)


@dataclass(frozen=True)
class Geometry:
    name: str
    head_dim: int
    kv_heads: int
    gqa: int


GEOMETRIES = {
    item.name: item
    for item in (
        Geometry("muse", 128, 2, 16),
        Geometry("olmo", 128, 8, 5),
        Geometry("phi", 128, 2, 4),
        Geometry("qwen", 256, 4, 6),
        Geometry("gemma", 512, 2, 8),
    )
}


def elapsed(call: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        call()
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end)) / repeats


def benchmark(
    geometry: Geometry, args: argparse.Namespace
) -> dict[str, object]:
    device = torch.device("cuda")
    batch = args.batch_size
    kv_heads = geometry.kv_heads
    q_heads = kv_heads * geometry.gqa
    dim = geometry.head_dim
    routes = args.routes
    state_len = round(args.state_growth_factor * math.sqrt(args.context_length))
    max_pages = max(args.pages_per_selected_slot)
    # Only the routed slots need long page lists.  All remaining state slots
    # receive one page, keeping the allocation close to a real T/16 page field.
    page_capacity = routes * max_pages + (state_len - routes)
    leaf_capacity = page_capacity * args.page_size
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + dim + kv_heads)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )

    q = randn((batch, q_heads, 1, dim))
    state_k = randn((batch, kv_heads, state_len, dim))
    state_v = randn((batch, kv_heads, state_len, dim))
    state_counts = torch.full(
        (batch, kv_heads, state_len, 1),
        args.page_size,
        dtype=torch.float32,
        device=device,
    )
    leaf_k = randn((batch, kv_heads, leaf_capacity, dim))
    leaf_v = randn((batch, kv_heads, leaf_capacity, dim))
    page_indices = torch.arange(
        leaf_capacity, dtype=torch.int32, device=device
    ).reshape(1, 1, page_capacity, args.page_size)
    page_indices = page_indices.expand(batch, kv_heads, -1, -1).contiguous()
    page_sum_k = randn((batch, kv_heads, page_capacity, dim))
    page_sum_v = randn((batch, kv_heads, page_capacity, dim))
    page_counts = torch.full(
        (batch, kv_heads, page_capacity),
        args.page_size,
        dtype=torch.int32,
        device=device,
    )

    slot_pages_cpu = torch.full(
        (state_len, max_pages), -1, dtype=torch.int32
    )
    next_page = 0
    for slot in range(routes):
        slot_pages_cpu[slot] = torch.arange(
            next_page, next_page + max_pages, dtype=torch.int32
        )
        next_page += max_pages
    for slot in range(routes, state_len):
        slot_pages_cpu[slot, 0] = next_page
        next_page += 1
    if next_page != page_capacity:
        raise AssertionError("page directory construction is inconsistent")
    slot_pages = (
        slot_pages_cpu[None, None]
        .expand(batch, kv_heads, -1, -1)
        .contiguous()
        .to(device)
    )
    preselected_slot_lengths = torch.full(
        (batch, kv_heads, state_len),
        args.page_size,
        dtype=torch.int32,
        device=device,
    )
    scanned_slot_lengths = preselected_slot_lengths.clone()
    top_slots = torch.arange(routes, dtype=torch.int64, device=device)
    top_slots = top_slots.reshape(1, 1, 1, routes).expand(
        batch, q_heads, 1, routes
    ).contiguous()
    overflow_page_keys = torch.full(
        (batch, kv_heads, 1), -1, dtype=torch.int32, device=device
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    cache_indices = torch.arange(batch, dtype=torch.int64, device=device)

    # The winner and last-page count are changed per measurement below.  Keep a
    # stable background table so winners from one case cannot leak into the
    # next one.
    base_page_scores = torch.randn(
        batch,
        q_heads,
        1,
        page_capacity,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    page_scores = base_page_scores.clone()
    output = torch.empty(
        batch, q_heads, routes, dim, dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(batch, q_heads, routes, dtype=torch.float32, device=device)

    common = {
        "q": q,
        "state_k": state_k,
        "state_v": state_v,
        "state_counts": state_counts,
        "leaf_k": leaf_k,
        "leaf_v": leaf_v,
        "page_indices": page_indices,
        "page_sum_k": page_sum_k,
        "page_sum_v": page_sum_v,
        "page_counts": page_counts,
        "overflow_page_keys": overflow_page_keys,
        "overflow_page_values": overflow_page_values,
        "overflow_used": overflow_used,
        "top_slots": top_slots,
        "cache_indices": cache_indices,
        "kv_group_size": geometry.gqa,
        "scale": dim**-0.5,
        "hash_probes": 0,
        "num_warps": args.num_warps,
        "waves_per_eu": 1,
        "output_buffer": output,
        "lse_buffer": lse,
        "route_parallel": True,
        "materialized_page_scores": page_scores,
    }

    results = []
    for page_block_n in args.page_block_n:
        for pages in args.pages_per_selected_slot:
            for last_page_count in args.last_page_counts:
                logical_count = (pages - 1) * args.page_size + last_page_count
                winning_pages = slot_pages_cpu[:routes, pages - 1].to(
                    device=device, dtype=torch.long
                )
                preselected_slot_pages = slot_pages.clone()
                preselected_slot_pages[:, :, :routes, 0] = winning_pages
                scanned_slot_lengths[:, :, :routes].fill_(logical_count)
                preselected_slot_lengths[:, :, :routes].fill_(last_page_count)
                state_counts[:, :, :routes, :].fill_(logical_count)
                page_counts.fill_(args.page_size)
                page_counts[..., winning_pages] = last_page_count
                page_scores.copy_(base_page_scores)
                page_scores[..., winning_pages] = 32.0

                def scan() -> tuple[torch.Tensor, torch.Tensor]:
                    return query_major_indexed_residual_page_attention(
                        slot_lengths=scanned_slot_lengths,
                        slot_pages=slot_pages,
                        page_block_n=page_block_n,
                        **common,
                    )

                def preselected() -> tuple[torch.Tensor, torch.Tensor]:
                    return query_major_indexed_residual_page_attention(
                        slot_lengths=preselected_slot_lengths,
                        slot_pages=preselected_slot_pages,
                        page_block_n=page_block_n,
                        **common,
                    )

                scan_out, scan_lse = scan()
                reference_out = scan_out.detach().clone()
                reference_lse = scan_lse.detach().clone()
                preselected_out, preselected_lse = preselected()
                output_error = float(
                    (preselected_out.float() - reference_out.float()).abs().max().item()
                )
                lse_error = float(
                    (preselected_lse - reference_lse).abs().max().item()
                )
                if output_error != 0.0 or lse_error != 0.0:
                    raise AssertionError(
                        "scan and preselected controls did not choose the same page: "
                        f"output={output_error}, lse={lse_error}"
                    )

                # Unused positions of a partial page must not affect either QK
                # or PV.  Poison them and require bit-identical output/LSE.
                poison_output_error = 0.0
                poison_lse_error = 0.0
                if last_page_count < args.page_size:
                    unused_leaf_indices = page_indices[
                        0, 0, winning_pages, last_page_count:
                    ].reshape(-1).long()
                    saved_leaf_k = leaf_k.index_select(2, unused_leaf_indices)
                    saved_leaf_v = leaf_v.index_select(2, unused_leaf_indices)
                    leaf_k.index_fill_(2, unused_leaf_indices, 0.0)
                    leaf_v.index_fill_(2, unused_leaf_indices, 2048.0)
                    poison_out, poison_lse = scan()
                    poison_output_error = float(
                        (poison_out.float() - reference_out.float()).abs().max().item()
                    )
                    poison_lse_error = float(
                        (poison_lse - reference_lse).abs().max().item()
                    )
                    leaf_k.index_copy_(2, unused_leaf_indices, saved_leaf_k)
                    leaf_v.index_copy_(2, unused_leaf_indices, saved_leaf_v)
                    if poison_output_error != 0.0 or poison_lse_error != 0.0:
                        raise AssertionError(
                            "unused entries in an underfull page affected attention: "
                            f"output={poison_output_error}, lse={poison_lse_error}"
                        )

                scan_ms = elapsed(scan, args.warmup, args.repeats)
                preselected_ms = elapsed(preselected, args.warmup, args.repeats)
                results.append(
                    {
                        "page_block_n": page_block_n,
                        "pages_per_selected_slot": pages,
                        "last_page_count": last_page_count,
                        "logical_slot_count": logical_count,
                        "scan_plus_exact_residual_ms": scan_ms,
                        "preselected_exact_residual_ms": preselected_ms,
                        "incremental_page_choice_ms": scan_ms - preselected_ms,
                        "output_max_abs": output_error,
                        "lse_max_abs": lse_error,
                        "poison_output_max_abs": poison_output_error,
                        "poison_lse_max_abs": poison_lse_error,
                    }
                )
    return {
        "geometry": asdict(geometry),
        "batch_size": batch,
        "context_length": args.context_length,
        "state_length": state_len,
        "page_capacity": page_capacity,
        "leaf_capacity": leaf_capacity,
        "routes": routes,
        "num_warps": args.num_warps,
        "measurements": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=(*GEOMETRIES, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=65536)
    parser.add_argument("--state-growth-factor", type=float, default=16.0)
    parser.add_argument("--page-size", type=int, choices=(16,), default=16)
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument(
        "--pages-per-selected-slot",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 128),
    )
    parser.add_argument(
        "--last-page-counts",
        type=int,
        nargs="+",
        default=(16,),
        help="Live token counts to test in the final selected page.",
    )
    parser.add_argument("--page-block-n", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--num-warps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(count < 1 or count > args.page_size for count in args.last_page_counts):
        raise ValueError("last-page counts must be in [1, page-size]")
    if args.routes > round(args.state_growth_factor * math.sqrt(args.context_length)):
        raise ValueError("route count exceeds state length")
    torch.cuda.set_device(0)
    selected = (
        GEOMETRIES.values()
        if args.geometry == "all"
        else (GEOMETRIES[args.geometry],)
    )
    records = []
    for geometry in selected:
        record = benchmark(geometry, args)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    payload = {"device": torch.cuda.get_device_name(), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
