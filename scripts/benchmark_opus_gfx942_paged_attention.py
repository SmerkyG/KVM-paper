#!/usr/bin/env python3
"""Benchmark the gfx942 OPUS short-bucket kernel on LOD-style pages."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.kernels.paged_leaf_attention import _paged_leaf_attention_kernel
from scripts.benchmark_opus_gfx942_attention import build_library, time_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experts", type=int, nargs="+", default=(256, 1024, 4096, 16384)
    )
    parser.add_argument("--blocks-per-expert", type=int, nargs="+", default=(1,))
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


class OpusPagedAttention:
    def __init__(self, library: Path):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.launch_opus_gfx942_paged_attention
        self.function.restype = ctypes.c_int
        self.function.argtypes = [ctypes.c_void_p] * 14 + [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
        ]

    def __call__(self, case: dict[str, torch.Tensor | int]) -> None:
        error = self.function(
            case["q"].data_ptr(),
            case["packed_route_row"].data_ptr(),
            case["block_expert"].data_ptr(),
            case["block_starts"].data_ptr(),
            case["page_k"].data_ptr(),
            case["page_v"].data_ptr(),
            case["slot_pages"].data_ptr(),
            case["slot_lengths"].data_ptr(),
            case["q_lengths"].data_ptr(),
            case["cu_q"].data_ptr(),
            case["expert_kv_row"].data_ptr(),
            case["expert_slot"].data_ptr(),
            case["opus_output"].data_ptr(),
            case["opus_lse"].data_ptr(),
            case["programs"],
            case["page_capacity"],
            case["experts"],
            4,
            128**-0.5,
            torch.cuda.current_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"HIP kernel launch failed with error {error}")


def make_case(
    programs: int,
    *,
    variable: bool,
    seed: int,
    blocks_per_expert: int = 1,
) -> dict:
    if programs % blocks_per_expert:
        raise ValueError("program count must be divisible by blocks per expert")
    if variable and blocks_per_expert != 1:
        raise ValueError("variable validation uses one block per expert")
    experts = programs // blocks_per_expert
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    page_size = 16
    head_dim = 128
    route_count = 3
    pages_per_expert = 4
    page_capacity = experts * pages_per_expert
    q = torch.randn(
        programs * 16,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    page_k = torch.randn(
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    page_v = torch.randn(
        page_capacity,
        page_size,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    slot_pages = torch.randperm(
        page_capacity, device=device, dtype=torch.int64, generator=generator
    ).to(torch.int32).reshape(experts, pages_per_expert)
    if variable:
        q_lengths = torch.randint(
            1, 17, (experts,), device=device, dtype=torch.int32, generator=generator
        )
        slot_lengths = torch.randint(
            1, 65, (experts,), device=device, dtype=torch.int32, generator=generator
        )
    else:
        q_lengths = torch.full(
            (experts,),
            16 * blocks_per_expert,
            device=device,
            dtype=torch.int32,
        )
        slot_lengths = torch.full((experts,), 64, device=device, dtype=torch.int32)
    cu_q = torch.nn.functional.pad(q_lengths.cumsum(0), (1, 0)).to(torch.int32)
    query_row_grid = torch.arange(
        programs * 16, device=device, dtype=torch.int64
    ).reshape(experts, 16 * blocks_per_expert)
    local_row = torch.arange(16 * blocks_per_expert, device=device)
    packed_query_rows = query_row_grid[local_row.unsqueeze(0) < q_lengths.unsqueeze(1)]
    packed_route_row = packed_query_rows * route_count
    route_rows = programs * 16 * route_count
    return {
        "experts": experts,
        "programs": programs,
        "blocks_per_expert": blocks_per_expert,
        "page_capacity": page_capacity,
        "q": q,
        "packed_route_row": packed_route_row,
        "block_expert": torch.arange(
            experts, device=device, dtype=torch.int32
        ).repeat_interleave(blocks_per_expert),
        "block_starts": (
            torch.arange(experts, device=device, dtype=torch.int32)
            * blocks_per_expert
        ),
        "page_k": page_k,
        "page_v": page_v,
        "slot_pages": slot_pages,
        "slot_lengths": slot_lengths,
        "q_lengths": q_lengths,
        "cu_q": cu_q,
        "expert_kv_row": torch.zeros(experts, device=device, dtype=torch.int64),
        "expert_slot": torch.arange(experts, device=device, dtype=torch.int64),
        "opus_output": torch.full(
            (route_rows, head_dim),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        ),
        "opus_lse": torch.full(
            (route_rows,), float("nan"), device=device, dtype=torch.float32
        ),
        "triton_output": torch.full(
            (route_rows, head_dim),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        ),
        "triton_lse": torch.full(
            (route_rows,), float("nan"), device=device, dtype=torch.float32
        ),
    }


def run_triton(case: dict) -> None:
    dummy = torch.empty(1, device="cuda", dtype=torch.int32)
    _paged_leaf_attention_kernel[(case["programs"],)](
        case["q"],
        case["packed_route_row"],
        case["block_expert"],
        case["block_starts"],
        case["page_k"],
        case["page_v"],
        case["slot_pages"],
        dummy,
        dummy,
        dummy,
        case["slot_lengths"],
        case["q_lengths"],
        case["cu_q"],
        case["expert_kv_row"],
        case["expert_slot"],
        case["triton_output"],
        case["triton_lse"],
        PAGE_CAPACITY=case["page_capacity"],
        STATE_CAPACITY=case["experts"],
        INLINE_PAGES_PER_SLOT=4,
        HASH_CAPACITY=1,
        HASH_PROBES=0,
        HEAD_DIM=128,
        VALUE_DIM=128,
        PAGE_SIZE=16,
        ROUTE_COUNT=3,
        SCALE_LOG2=128**-0.5 * 1.4426950408889634,
        BLOCK_M=16,
        BLOCK_N=64,
        num_warps=4,
        waves_per_eu=1,
    )


def validate(opus: OpusPagedAttention) -> dict[str, float]:
    case = make_case(32, variable=True, seed=31)
    opus(case)
    run_triton(case)
    torch.cuda.synchronize()
    route_rows = case["packed_route_row"]
    opus_output = case["opus_output"].index_select(0, route_rows).float()
    triton_output = case["triton_output"].index_select(0, route_rows).float()
    opus_lse = case["opus_lse"].index_select(0, route_rows)
    triton_lse = case["triton_lse"].index_select(0, route_rows)
    metrics = {
        "output_max_abs_error": float((opus_output - triton_output).abs().max()),
        "output_mean_abs_error": float((opus_output - triton_output).abs().mean()),
        "lse_max_abs_error": float((opus_lse - triton_lse).abs().max()),
    }
    if metrics["output_max_abs_error"] > 0.02 or metrics["lse_max_abs_error"] > 1e-4:
        raise AssertionError(f"paged OPUS result is incorrect: {metrics}")
    return metrics


def main() -> None:
    args = parse_args()
    opus = OpusPagedAttention(build_library(args.force_build))
    correctness = validate(opus)
    results = []
    for blocks_per_expert in args.blocks_per_expert:
        for programs in args.experts:
            case = make_case(
                programs,
                variable=False,
                seed=41 + programs + blocks_per_expert,
                blocks_per_expert=blocks_per_expert,
            )
            opus(case)
            run_triton(case)
            torch.cuda.synchronize()
            repeats = max(10, min(200, 131072 // programs))
            opus_ms = time_cuda(lambda: opus(case), repeats)
            triton_ms = time_cuda(lambda: run_triton(case), repeats)
            useful_pairs = programs * 16 * 64
            results.append(
                {
                    "programs": programs,
                    "active_experts": case["experts"],
                    "blocks_per_expert": blocks_per_expert,
                    "repeats": repeats,
                    "opus_ms": opus_ms,
                    "triton_ms": triton_ms,
                    "opus_speedup": triton_ms / opus_ms,
                    "opus_gpair_per_s": useful_pairs / opus_ms / 1e6,
                    "triton_gpair_per_s": useful_pairs / triton_ms / 1e6,
                }
            )
    record = {"correctness": correctness, "results": results}
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
