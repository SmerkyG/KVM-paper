#!/usr/bin/env python3
"""Measure exact global LoD routing over DCP rank-local top-eight results."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from lod_attention.kernels.distributed_topk import distributed_global_topk


class _TorchDistributedGroup:
    def __init__(self) -> None:
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()

    def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += value.ndim
        source = value.movedim(dim, 0).contiguous()
        output = torch.empty(
            (source.size(0) * self.world_size, *source.shape[1:]),
            dtype=source.dtype,
            device=source.device,
        )
        dist.all_gather_into_tensor(output, source)
        return output.movedim(0, dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--local-slots", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = _TorchDistributedGroup()
    device = torch.device("cuda", local_rank)
    generator = torch.Generator(device=device).manual_seed(20261001 + local_rank)
    dense_scores = torch.randn(
        args.batch_size,
        args.heads,
        args.local_slots,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    local_scores, local_slots = torch.topk(
        dense_scores, args.topk, dim=-1, sorted=True
    )
    local_slots = local_slots.to(torch.int32)

    output = None
    for _ in range(args.warmup):
        output = distributed_global_topk(local_scores, local_slots, group)
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if args.cudagraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = distributed_global_topk(local_scores, local_slots, group)
        torch.cuda.synchronize()
        dist.barrier()
    begin = time.perf_counter()
    for _ in range(args.repetitions):
        if graph is None:
            output = distributed_global_topk(local_scores, local_slots, group)
        else:
            graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - begin

    assert output is not None
    selected_scores, owners, selected_slots = output
    # Validate against the gathered dense field once, outside timing.
    dense_gathered = group.all_gather(dense_scores, dim=-1)
    reference_scores, reference_indices = torch.topk(
        dense_gathered, args.topk, dim=-1, sorted=True
    )
    reference_owners = reference_indices // args.local_slots
    reference_slots = reference_indices % args.local_slots
    torch.testing.assert_close(selected_scores, reference_scores, rtol=0, atol=0)
    torch.testing.assert_close(owners, reference_owners.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        selected_slots, reference_slots.to(torch.int32), rtol=0, atol=0
    )
    if group.rank_in_group == 0:
        print(
            json.dumps(
                {
                    "world_size": group.world_size,
                    "batch_size": args.batch_size,
                    "heads": args.heads,
                    "local_slots": args.local_slots,
                    "topk": args.topk,
                    "cudagraph": args.cudagraph,
                    "candidate_bytes_per_rank": (
                        args.batch_size * args.heads * args.topk * 2 * 4
                    ),
                    "milliseconds": elapsed * 1000 / args.repetitions,
                },
                sort_keys=True,
            )
        )
    if graph is not None:
        graph.reset()
        torch.cuda.synchronize()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
