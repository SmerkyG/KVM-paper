#!/usr/bin/env python3
"""Controlled geometry benchmark for the production fused two-tier decoder."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from model.kernels.paged_leaf_attention import (
    fused_decode_paged_lod_attention,
    new_fused_decode_buffers,
)
from scripts.benchmark_lod_geometry_matrix import (
    DIAGNOSTIC_GEOMETRIES,
    Geometry,
)


@dataclass(frozen=True)
class DecodeConfig:
    block_n: int = 16
    num_warps: int = 2
    route_group_size: int = 32
    route_num_warps: int = 2
    route_reduce_num_warps: int = 4
    final_reduce_num_warps: int = 4
    use_dot: bool = False
    route_use_dot: bool = True
    fuse_final_reduce: bool = False
    cooperative: bool = False
    cooperative_route_splits: int = 8

    @property
    def name(self) -> str:
        return (
            f"n{self.block_n}_w{self.num_warps}_rg{self.route_group_size}"
            f"_rw{self.route_num_warps}_rr{self.route_reduce_num_warps}"
            f"_fr{self.final_reduce_num_warps}_ld{int(self.use_dot)}"
            f"_rd{int(self.route_use_dot)}_ff{int(self.fuse_final_reduce)}"
            f"_coop{int(self.cooperative)}"
        )


BASELINE = DecodeConfig()
COOPERATIVE_CONFIGS = (
    DecodeConfig(cooperative=True, cooperative_route_splits=4),
    DecodeConfig(cooperative=True, cooperative_route_splits=8),
    DecodeConfig(cooperative=True, cooperative_route_splits=16),
)
TUNING_CONFIGS = (
    BASELINE,
    DecodeConfig(block_n=32),
    DecodeConfig(block_n=64),
    DecodeConfig(num_warps=4),
    DecodeConfig(block_n=32, num_warps=4),
    DecodeConfig(block_n=64, num_warps=4),
    DecodeConfig(route_group_size=16),
    DecodeConfig(route_group_size=64),
    DecodeConfig(route_num_warps=1),
    DecodeConfig(route_num_warps=4),
    DecodeConfig(route_reduce_num_warps=1),
    DecodeConfig(route_reduce_num_warps=2),
    DecodeConfig(route_reduce_num_warps=8),
    DecodeConfig(final_reduce_num_warps=1),
    DecodeConfig(final_reduce_num_warps=2),
    DecodeConfig(final_reduce_num_warps=8),
    DecodeConfig(use_dot=True),
    DecodeConfig(route_use_dot=False),
    DecodeConfig(fuse_final_reduce=True),
    DecodeConfig(
        block_n=32,
        route_group_size=64,
        route_num_warps=1,
        route_reduce_num_warps=2,
    ),
    DecodeConfig(
        block_n=32,
        route_group_size=64,
        route_num_warps=1,
        route_reduce_num_warps=2,
        route_use_dot=False,
    ),
    DecodeConfig(
        block_n=64,
        route_group_size=64,
        route_num_warps=1,
        route_reduce_num_warps=2,
        route_use_dot=False,
    ),
)

LOW_ROW_ROUTE_CONFIGS = (
    BASELINE,
    DecodeConfig(route_group_size=16),
    DecodeConfig(route_group_size=16, route_num_warps=1),
    DecodeConfig(route_group_size=16, route_use_dot=False),
    DecodeConfig(
        route_group_size=16, route_num_warps=1, route_use_dot=False
    ),
    DecodeConfig(
        route_group_size=16, route_num_warps=4, route_use_dot=False
    ),
    DecodeConfig(route_group_size=8, route_num_warps=1),
    DecodeConfig(route_group_size=8),
    DecodeConfig(route_group_size=8, route_use_dot=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=(
            "diagnostic",
            "cooperative",
            "tune128",
            "tune512",
            "lowrow256",
            "all",
        ),
        default="diagnostic"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument(
        "--context-len",
        type=int,
        help="Use 16*sqrt(T) slots and posting lists summing to this context",
    )
    parser.add_argument("--geometry", help="Run only geometry names containing this")
    parser.add_argument("--local-len", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_inputs(
    geometry: Geometry,
    *,
    batch_size: int,
    slots: int,
    local_len: int,
    seed: int,
    device: torch.device,
    context_len: int | None = None,
) -> dict[str, torch.Tensor]:
    page_size = 16
    if context_len is None:
        lengths_1d = torch.tensor(
            [page_size * (1 + ((slot * 7) % 16)) for slot in range(slots)],
            dtype=torch.int32,
        )
    else:
        weights = torch.tensor(
            [(1 + ((slot * 7) % 16)) ** 2 for slot in range(slots)],
            dtype=torch.float64,
        )
        lengths_1d = torch.floor(weights * context_len / weights.sum()).to(
            torch.int32
        ).clamp_min_(1)
        difference = context_len - int(lengths_1d.sum().item())
        step = 1 if difference >= 0 else -1
        for index in range(abs(difference)):
            slot = index % slots
            if step > 0 or lengths_1d[slot] > 1:
                lengths_1d[slot] += step
    pages_per_slot = (lengths_1d + page_size - 1) // page_size
    page_capacity = int(pages_per_slot.sum().item())
    inline_pages = int(pages_per_slot.max().item())
    slot_pages_1d = torch.full((slots, inline_pages), -1, dtype=torch.int16)
    page = 0
    for slot, count in enumerate(pages_per_slot.tolist()):
        slot_pages_1d[slot, :count] = torch.arange(
            page, page + count, dtype=torch.int16
        )
        page += count
    slot_pages = slot_pages_1d[None, None].expand(
        batch_size, geometry.kv_heads, -1, -1
    ).contiguous().to(device)
    slot_lengths = lengths_1d[None, None].expand(
        batch_size, geometry.kv_heads, -1
    ).contiguous().to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + geometry.head_dim)
    kv_shape = (batch_size, geometry.kv_heads, slots, geometry.head_dim)
    page_shape = (
        batch_size,
        geometry.kv_heads,
        page_capacity,
        page_size,
        geometry.head_dim,
    )
    local_shape = (
        batch_size,
        geometry.kv_heads,
        local_len,
        geometry.head_dim,
    )
    query_heads = geometry.kv_heads * geometry.group_size
    return {
        "q": torch.randn(
            batch_size, query_heads, 1, geometry.head_dim,
            dtype=torch.bfloat16, device=device, generator=generator,
        ),
        "state_k": torch.randn(
            kv_shape, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "state_v": torch.randn(
            kv_shape, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "counts": torch.randint(
            1, 257, (batch_size, geometry.kv_heads, slots, 1),
            dtype=torch.int32, device=device, generator=generator,
        ).float(),
        "local_k": torch.randn(
            local_shape, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "local_v": torch.randn(
            local_shape, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "page_k": torch.randn(
            page_shape, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "page_v": torch.randn(
            page_shape, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "slot_pages": slot_pages,
        "overflow_page_keys": torch.full(
            (batch_size, geometry.kv_heads, 1), -1,
            dtype=torch.int32, device=device,
        ),
        "overflow_page_values": torch.full(
            (batch_size, geometry.kv_heads, 1), -1,
            dtype=torch.int32, device=device,
        ),
        "overflow_used": torch.zeros((), dtype=torch.int32, device=device),
        "slot_lengths": slot_lengths,
    }


def phase_times(
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
) -> dict[str, float]:
    return {
        name: sum(float(begin.elapsed_time(end)) for begin, end in pairs)
        for name, pairs in events.items()
    }


def benchmark(
    tensors: dict[str, torch.Tensor], geometry: Geometry, config: DecodeConfig,
    *, slots: int, local_len: int, repeats: int,
) -> tuple[dict[str, object], torch.Tensor]:
    q = tensors["q"]
    buffers = new_fused_decode_buffers(
        q, splits=8, state_capacity=slots,
        route_group_size=config.route_group_size,
        gqa_route_splits=(
            config.cooperative_route_splits if config.cooperative else None
        ),
    )
    common = dict(
        **tensors,
        top_slots=None,
        state_len=slots,
        local_len=local_len,
        kv_group_size=geometry.group_size,
        scale=geometry.head_dim**-0.5,
        hash_probes=0,
        block_n=config.block_n,
        num_warps=config.num_warps,
        split_kv=8,
        buffers=buffers,
        use_dot=config.use_dot,
        fuse_state_route=True,
        route_group_size=config.route_group_size,
        route_num_warps=config.route_num_warps,
        route_reduce_num_warps=config.route_reduce_num_warps,
        final_reduce_num_warps=config.final_reduce_num_warps,
        fuse_final_reduce=config.fuse_final_reduce,
        route_use_dot=config.route_use_dot,
        route_gqa_grouped=True,
        gqa_cooperative_leaf=config.cooperative,
        gqa_cooperative_hip=config.cooperative,
        gqa_cooperative_route_splits=config.cooperative_route_splits,
    )
    output = fused_decode_paged_lod_attention(**common)
    torch.cuda.synchronize()
    wall: list[float] = []
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        torch.cuda.synchronize()
        begin = time.perf_counter()
        output = fused_decode_paged_lod_attention(
            **common, timing_events=events
        )
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - begin) * 1000.0)
        for name, value in phase_times(events).items():
            samples.setdefault(name, []).append(value)
    return {
        "config": config.name,
        "wall_ms_median": statistics.median(wall),
        "phase_ms_median": {
            name: statistics.median(values) for name, values in samples.items()
        },
    }, output.detach().clone()


def run_geometry(
    geometry: Geometry, configs: tuple[DecodeConfig, ...], args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    tensors = make_inputs(
        geometry, batch_size=args.batch_size, slots=args.slots,
        local_len=args.local_len, seed=args.seed, device=device,
        context_len=args.context_len,
    )
    baseline: torch.Tensor | None = None
    measurements = []
    for config in configs:
        result, output = benchmark(
            tensors, geometry, config, slots=args.slots,
            local_len=args.local_len, repeats=args.repeats,
        )
        if baseline is None:
            baseline = output
            result["output_max_abs_vs_baseline"] = 0.0
        else:
            absolute_error = (output.float() - baseline.float()).abs()
            result["output_max_abs_vs_baseline"] = float(
                absolute_error.max().item()
            )
            result["output_mean_abs_vs_baseline"] = float(
                absolute_error.mean().item()
            )
        measurements.append(result)
        print(
            geometry.name, config.name,
            f"wall={result['wall_ms_median']:.3f} ms",
            json.dumps(result["phase_ms_median"], sort_keys=True),
            flush=True,
        )
    best = min(measurements, key=lambda item: float(item["wall_ms_median"]))
    return {
        "name": geometry.name,
        "head_dim": geometry.head_dim,
        "kv_heads": geometry.kv_heads,
        "group_size": geometry.group_size,
        "query_heads": geometry.kv_heads * geometry.group_size,
        "best_config": best["config"],
        "measurements": measurements,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if args.context_len is not None:
        args.slots = max(1, round(16 * math.sqrt(args.context_len)))
    jobs: list[tuple[Geometry, tuple[DecodeConfig, ...]]] = []
    if args.suite in ("diagnostic", "all"):
        jobs.extend((geometry, (BASELINE,)) for geometry in DIAGNOSTIC_GEOMETRIES)
    if args.suite in ("cooperative", "all"):
        jobs.extend(
            (geometry, (BASELINE, *COOPERATIVE_CONFIGS))
            for geometry in DIAGNOSTIC_GEOMETRIES
        )
    if args.suite in ("tune128", "all"):
        jobs.extend(
            (geometry, TUNING_CONFIGS)
            for geometry in DIAGNOSTIC_GEOMETRIES if geometry.head_dim == 128
        )
    if args.suite in ("tune512", "all"):
        jobs.append((Geometry("D512_KV2_G8_gemma", 512, 2, 8), TUNING_CONFIGS))
    if args.suite in ("lowrow256", "all"):
        jobs.append(
            (Geometry("D256_KV2_G4_qwen", 256, 2, 4), LOW_ROW_ROUTE_CONFIGS)
        )
    if args.geometry:
        jobs = [job for job in jobs if args.geometry in job[0].name]
    output: dict[str, object] = {
        "suite": args.suite,
        "batch_size": args.batch_size,
        "slots": args.slots,
        "local_len": args.local_len,
        "context_len": args.context_len,
        "device": torch.cuda.get_device_name(device),
        "geometries": [],
    }
    for geometry, configs in jobs:
        output["geometries"].append(run_geometry(geometry, configs, args, device))
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
