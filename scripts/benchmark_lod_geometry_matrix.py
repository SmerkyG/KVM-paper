#!/usr/bin/env python3
"""Isolate two-tier exact-leaf cost across head dimensions and GQA shapes.

This is intentionally a synthetic benchmark of the production expert-major
kernel, not a toy GEMM.  Every case receives the same centroid posting-list
lengths and the same deterministic top-eight route distribution.  The timed
path still includes production dispatch, sorting, expert grouping, exact leaf
attention, and the route/LSE reduction.
"""

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

from model.kernels.paged_leaf_attention import paged_leaf_attention


@dataclass(frozen=True)
class Geometry:
    name: str
    head_dim: int
    kv_heads: int
    group_size: int


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    num_warps: int
    waves_per_eu: int = 1

    @property
    def name(self) -> str:
        suffix = f"_wave{self.waves_per_eu}" if self.waves_per_eu != 1 else ""
        return f"m{self.block_m}_n{self.block_n}_w{self.num_warps}{suffix}"


DIAGNOSTIC_GEOMETRIES = (
    # Isolate D while preserving Qwen's per-rank KV/GQA geometry.
    Geometry("D128_KV4_G6", 128, 4, 6),
    Geometry("D256_KV2_G4_qwen_native", 256, 2, 4),
    Geometry("D256_KV4_G6_qwen", 256, 4, 6),
    Geometry("D512_KV4_G6", 512, 4, 6),
    # Actual per-rank/model geometries from the cross-family panel.
    Geometry("D128_KV2_G16_muse", 128, 2, 16),
    Geometry("D128_KV8_G5_olmo", 128, 8, 5),
    Geometry("D128_KV2_G4_phi_tp5", 128, 2, 4),
    Geometry("D512_KV2_G8_gemma", 512, 2, 8),
)


D128_CONFIGS = (
    KernelConfig(16, 32, 2),
    KernelConfig(16, 16, 2),
    KernelConfig(16, 16, 4),
    KernelConfig(16, 32, 4),
    KernelConfig(16, 64, 2),
    KernelConfig(16, 64, 4),
    KernelConfig(32, 16, 2),
    KernelConfig(32, 16, 4),
    KernelConfig(32, 32, 2),
    KernelConfig(32, 32, 4),
    KernelConfig(32, 64, 4),
    KernelConfig(64, 16, 4),
    KernelConfig(64, 32, 4),
    KernelConfig(64, 64, 4),
    KernelConfig(64, 64, 8),
)


D512_CONFIGS = (
    KernelConfig(16, 32, 2),
    KernelConfig(16, 32, 2, 2),
    KernelConfig(16, 32, 2, 3),
    KernelConfig(16, 32, 2, 4),
    # D=512 keeps both Q and the FP32 output accumulator live.  Smaller M
    # tiles deliberately trade some MFMA utilization for substantially lower
    # register pressure.
    KernelConfig(4, 16, 2),
    KernelConfig(4, 32, 2),
    KernelConfig(4, 64, 2),
    KernelConfig(8, 16, 2),
    KernelConfig(8, 16, 4),
    KernelConfig(8, 32, 2),
    KernelConfig(8, 32, 4),
    KernelConfig(8, 64, 2),
    KernelConfig(8, 64, 4),
    KernelConfig(16, 16, 2),
    KernelConfig(16, 16, 4),
    KernelConfig(16, 32, 4),
    KernelConfig(16, 32, 8),
    KernelConfig(16, 64, 4),
    KernelConfig(16, 64, 8),
    KernelConfig(32, 16, 4),
    KernelConfig(32, 32, 4),
    KernelConfig(32, 32, 8),
    KernelConfig(32, 64, 8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("diagnostic", "tune128", "tune512", "all"),
        default="diagnostic",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-len", type=int, default=256)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--routes", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_inputs(
    geometry: Geometry,
    *,
    batch_size: int,
    query_len: int,
    slots: int,
    routes: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if routes > slots:
        raise ValueError("route count cannot exceed state slots")
    page_size = 16
    # The permutation produces identical 16..256-token posting-list lengths
    # for every batch/KV row, while avoiding correlation with adjacent slots.
    slot_lengths_1d = torch.tensor(
        [page_size * (1 + ((slot * 7) % 16)) for slot in range(slots)],
        dtype=torch.int32,
    )
    pages_per_slot = torch.div(
        slot_lengths_1d + page_size - 1, page_size, rounding_mode="floor"
    )
    page_capacity = int(pages_per_slot.sum().item())
    inline_pages = int(pages_per_slot.max().item())

    slot_pages_1d = torch.full(
        (slots, inline_pages), -1, dtype=torch.int16
    )
    next_page = 0
    for slot, page_count in enumerate(pages_per_slot.tolist()):
        slot_pages_1d[slot, :page_count] = torch.arange(
            next_page, next_page + page_count, dtype=torch.int16
        )
        next_page += page_count
    slot_pages = (
        slot_pages_1d[None, None]
        .expand(batch_size, geometry.kv_heads, -1, -1)
        .contiguous()
        .to(device)
    )
    slot_lengths = (
        slot_lengths_1d[None, None]
        .expand(batch_size, geometry.kv_heads, -1)
        .contiguous()
        .to(device)
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + geometry.head_dim)
    storage_shape = (
        batch_size,
        geometry.kv_heads,
        page_capacity,
        page_size,
        geometry.head_dim,
    )
    page_k = torch.randn(
        storage_shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    page_v = torch.randn(
        storage_shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    query_heads = geometry.kv_heads * geometry.group_size
    q = torch.randn(
        batch_size,
        query_heads,
        query_len,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    # Each query opens eight distinct slots.  The formula is independent of D
    # and evenly distributes routes across the same posting-list population.
    route_offsets = torch.tensor(
        (0, 1, 3, 7, 11, 15, 23, 29)[:routes], device=device
    )
    token = torch.arange(query_len, device=device)[None, None, :, None]
    head = torch.arange(query_heads, device=device)[None, :, None, None]
    batch = torch.arange(batch_size, device=device)[:, None, None, None]
    base = token * 5 + head * 3 + batch * 7
    top_slots = ((base + route_offsets) % slots).to(torch.int64)

    overflow_page_keys = torch.full(
        (batch_size, geometry.kv_heads, 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    overflow_page_values = torch.full_like(overflow_page_keys, -1)
    overflow_used = torch.zeros((), dtype=torch.int32, device=device)
    return {
        "q": q,
        "page_k": page_k,
        "page_v": page_v,
        "slot_pages": slot_pages,
        "overflow_page_keys": overflow_page_keys,
        "overflow_page_values": overflow_page_values,
        "overflow_used": overflow_used,
        "slot_lengths": slot_lengths,
        "top_slots": top_slots,
    }


def elapsed_phases(
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
) -> dict[str, float]:
    return {
        name: sum(float(begin.elapsed_time(end)) for begin, end in pairs)
        for name, pairs in events.items()
    }


def benchmark_config(
    tensors: dict[str, torch.Tensor],
    geometry: Geometry,
    config: KernelConfig,
    *,
    repeats: int,
) -> tuple[dict[str, object], tuple[torch.Tensor, torch.Tensor]]:
    common = dict(
        kv_group_size=geometry.group_size,
        scale=geometry.head_dim**-0.5,
        hash_probes=0,
        block_m=config.block_m,
        block_n=config.block_n,
        num_warps=config.num_warps,
        waves_per_eu=config.waves_per_eu,
        reduce_num_warps=1,
    )
    # Compile and warm all paths before collecting either GPU or wall time.
    warm = paged_leaf_attention(**tensors, **common)
    torch.cuda.synchronize()
    phase_samples: dict[str, list[float]] = {}
    wall_samples: list[float] = []
    latest = warm
    for _ in range(repeats):
        events: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}
        torch.cuda.synchronize()
        begin = time.perf_counter()
        latest = paged_leaf_attention(**tensors, timing_events=events, **common)
        torch.cuda.synchronize()
        wall_samples.append((time.perf_counter() - begin) * 1000.0)
        for name, value in elapsed_phases(events).items():
            phase_samples.setdefault(name, []).append(value)
    phase_medians = {
        name: statistics.median(values) for name, values in phase_samples.items()
    }
    q = tensors["q"]
    query_rows = int(math.prod(q.shape[:-1]))
    useful_pairs = query_rows * int(tensors["top_slots"].size(-1)) * int(
        tensors["slot_lengths"].float().mean().item()
    )
    result: dict[str, object] = {
        "config": config.name,
        "block_m": config.block_m,
        "block_n": config.block_n,
        "num_warps": config.num_warps,
        "waves_per_eu": config.waves_per_eu,
        "wall_ms_median": statistics.median(wall_samples),
        "wall_ms_min": min(wall_samples),
        "phase_ms_median": phase_medians,
        "useful_qk_pairs": useful_pairs,
        "useful_qk_gpairs_per_s": useful_pairs
        / max(phase_medians.get("kernel", 0.0), 1e-9)
        / 1e6,
    }
    return result, latest


def run_geometry(
    geometry: Geometry,
    configs: tuple[KernelConfig, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    tensors = make_inputs(
        geometry,
        batch_size=args.batch_size,
        query_len=args.query_len,
        slots=args.slots,
        routes=args.routes,
        seed=args.seed,
        device=device,
    )
    baseline_output: tuple[torch.Tensor, torch.Tensor] | None = None
    results: list[dict[str, object]] = []
    for config in configs:
        measured, output = benchmark_config(
            tensors, geometry, config, repeats=args.repeats
        )
        if baseline_output is None:
            baseline_output = tuple(value.detach().clone() for value in output)
            measured["output_max_abs_vs_baseline"] = 0.0
            measured["lse_max_abs_vs_baseline"] = 0.0
        else:
            measured["output_max_abs_vs_baseline"] = float(
                (output[0].float() - baseline_output[0].float()).abs().max().item()
            )
            measured["lse_max_abs_vs_baseline"] = float(
                (output[1] - baseline_output[1]).abs().max().item()
            )
        results.append(measured)
        print(
            geometry.name,
            config.name,
            f"wall={measured['wall_ms_median']:.3f} ms",
            f"kernel={measured['phase_ms_median'].get('kernel', float('nan')):.3f} ms",
            flush=True,
        )
    best = min(results, key=lambda result: float(result["wall_ms_median"]))
    return {
        "name": geometry.name,
        "head_dim": geometry.head_dim,
        "kv_heads": geometry.kv_heads,
        "group_size": geometry.group_size,
        "query_heads": geometry.kv_heads * geometry.group_size,
        "best_config": best["config"],
        "measurements": results,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if args.routes > 8:
        raise ValueError("the deterministic route offsets support at most eight routes")
    suites: list[tuple[Geometry, tuple[KernelConfig, ...]]] = []
    if args.suite in ("diagnostic", "all"):
        suites.extend(
            (geometry, (KernelConfig(16, 32, 2),))
            for geometry in DIAGNOSTIC_GEOMETRIES
        )
    if args.suite in ("tune128", "all"):
        suites.extend(
            (geometry, D128_CONFIGS)
            for geometry in DIAGNOSTIC_GEOMETRIES
            if geometry.head_dim == 128
        )
    if args.suite in ("tune512", "all"):
        suites.append((Geometry("D512_KV2_G8_gemma", 512, 2, 8), D512_CONFIGS))

    output: dict[str, object] = {
        "suite": args.suite,
        "batch_size": args.batch_size,
        "query_len": args.query_len,
        "slots": args.slots,
        "routes": args.routes,
        "posting_list_lengths": [16 * (1 + ((slot * 7) % 16)) for slot in range(args.slots)],
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "geometries": [],
    }
    for geometry, configs in suites:
        output["geometries"].append(run_geometry(geometry, configs, args, device))
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
