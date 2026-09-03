#!/usr/bin/env python3
"""Check and time grouped page-wide INT4 conversion on a real cache geometry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from model.kernels.paged_leaf_attention import quantize_virtual_paged_kv


def allocate_outputs(
    leaf: torch.Tensor,
    page_capacity: int,
) -> tuple[torch.Tensor, ...]:
    batch, heads, leaf_capacity, dimension = leaf.shape
    codes = torch.zeros(
        batch,
        heads,
        leaf_capacity,
        dimension // 2,
        dtype=torch.uint8,
        device=leaf.device,
    )
    scales = torch.zeros(
        batch,
        heads,
        page_capacity,
        dimension // 4,
        dtype=leaf.dtype,
        device=leaf.device,
    )
    counts = torch.zeros(
        batch,
        heads,
        page_capacity,
        dtype=torch.int32,
        device=leaf.device,
    )
    return codes, codes.clone(), scales, scales.clone(), counts


def run(
    *,
    groups_per_program: int,
    leaf_k: torch.Tensor,
    leaf_v: torch.Tensor,
    page_indices: torch.Tensor,
    page_sum_k: torch.Tensor,
    page_sum_v: torch.Tensor,
    page_counts: torch.Tensor,
    repeats: int,
) -> tuple[tuple[torch.Tensor, ...], float]:
    os.environ["VLLM_LOD_INT4_QUANT_GROUPS_PER_PROGRAM"] = str(
        groups_per_program
    )
    outputs = allocate_outputs(leaf_k, int(page_indices.size(2)))

    def launch() -> None:
        quantize_virtual_paged_kv(
            leaf_k,
            leaf_v,
            page_indices,
            page_sum_k,
            page_sum_v,
            page_counts,
            *outputs,
            quant_group_size=4,
            quant_token_group_size=16,
            quant_bits=4,
            optimize_scale=True,
        )

    launch()
    launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        launch()
    end.record()
    torch.cuda.synchronize()
    return outputs, start.elapsed_time(end) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-capacity", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch = 1
    heads = 2
    page_size = 16
    dimension = 256
    leaf_capacity = args.page_capacity * page_size
    leaf_k = torch.randn(
        batch,
        heads,
        leaf_capacity,
        dimension,
        dtype=torch.bfloat16,
        device=device,
    )
    leaf_v = torch.randn_like(leaf_k)
    page_indices = torch.arange(
        leaf_capacity,
        dtype=torch.int32,
        device=device,
    ).reshape(1, 1, args.page_capacity, page_size).repeat(batch, heads, 1, 1)
    # Exercise both full and underfull pages without leaving invalid data in
    # the reference sums.
    page_counts = torch.full(
        (batch, heads, args.page_capacity),
        page_size,
        dtype=torch.int32,
        device=device,
    )
    page_counts[..., 7::17] = 9
    token = torch.arange(page_size, device=device)
    valid = token.view(1, 1, 1, page_size, 1) < page_counts[..., None, None]
    page_sum_k = torch.where(
        valid,
        leaf_k.reshape(batch, heads, args.page_capacity, page_size, dimension),
        0,
    ).sum(dim=3, dtype=torch.float32).to(torch.bfloat16)
    page_sum_v = torch.where(
        valid,
        leaf_v.reshape(batch, heads, args.page_capacity, page_size, dimension),
        0,
    ).sum(dim=3, dtype=torch.float32).to(torch.bfloat16)

    reference, reference_ms = run(
        groups_per_program=1,
        leaf_k=leaf_k,
        leaf_v=leaf_v,
        page_indices=page_indices,
        page_sum_k=page_sum_k,
        page_sum_v=page_sum_v,
        page_counts=page_counts,
        repeats=args.repeats,
    )
    result: dict[str, object] = {
        "geometry": {
            "batch": batch,
            "kv_heads": heads,
            "page_capacity": args.page_capacity,
            "page_size": page_size,
            "dimension": dimension,
        },
        "variants": {
            "1": {"milliseconds": reference_ms, "speedup": 1.0}
        },
    }
    for groups_per_program in (2, 4, 8):
        candidate, milliseconds = run(
            groups_per_program=groups_per_program,
            leaf_k=leaf_k,
            leaf_v=leaf_v,
            page_indices=page_indices,
            page_sum_k=page_sum_k,
            page_sum_v=page_sum_v,
            page_counts=page_counts,
            repeats=args.repeats,
        )
        codes_equal = bool(
            torch.equal(reference[0], candidate[0])
            and torch.equal(reference[1], candidate[1])
        )
        counts_equal = bool(torch.equal(reference[4], candidate[4]))
        scale_max_abs = max(
            float((reference[2].float() - candidate[2].float()).abs().max()),
            float((reference[3].float() - candidate[3].float()).abs().max()),
        )
        result["variants"][str(groups_per_program)] = {
            "milliseconds": milliseconds,
            "speedup": reference_ms / milliseconds,
            "codes_equal": codes_equal,
            "counts_equal": counts_equal,
            "scale_max_abs": scale_max_abs,
        }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


if __name__ == "__main__":
    main()
