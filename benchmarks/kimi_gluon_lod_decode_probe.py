#!/usr/bin/env python3
"""Correctness and latency probe for compact two-tier Kimi LoD decode.

This constructs the same logical sequence used by the production two-tier
consumer: an exact/local plus coarse prefix addressed through ``fixed_indices``
and an exact suffix described by packed 16-token leaf pages.  Coarse rows carry
``log(count)`` attention bias, while exact rows carry zero bias.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from lod_attention.kernels.kimi_gluon_decode import absorbed_mla_lod_decode_gfx942


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--coarse", type=int, default=2048)
    parser.add_argument("--local", type=int, default=256)
    parser.add_argument("--exact-pages", type=int, default=128)
    parser.add_argument("--splits", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--head-tiled-metadata", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not 1 <= args.heads <= 128:
        raise ValueError("heads must be in [1,128]")
    if args.local < 0 or args.coarse < 0 or args.exact_pages < 0:
        raise ValueError("sequence component counts must be nonnegative")

    torch.manual_seed(23)
    device = torch.device("cuda")
    page_size = 16
    exact = args.exact_pages * page_size
    prefix = args.local + args.coarse
    logical_len = prefix + exact
    capacity = logical_len + 64

    batch = args.batch_size
    q = torch.randn(
        batch, args.heads, 576, device=device, dtype=torch.bfloat16
    ) / math.sqrt(576)
    kv = torch.randn(
        batch * capacity, 576, device=device, dtype=torch.bfloat16
    )
    # Give coarse rows realistic positive masses and exact rows unit mass.
    bias = torch.zeros(batch * capacity, device=device, dtype=torch.float32)
    if args.coarse:
        counts = torch.randint(
            2,
            257,
            (batch, args.coarse),
            device=device,
            dtype=torch.int32,
        )
        bias.view(batch, capacity)[:, args.local : prefix] = counts.float().log()

    # Keep the synthetic list simple and deterministic while still exercising
    # both levels of indirection used by the production compact consumer.
    fixed_indices = torch.arange(
        batch * capacity, device=device, dtype=torch.int32
    ).view(batch, capacity)
    page_bases = prefix + torch.arange(
        args.exact_pages, device=device, dtype=torch.int32
    ) * page_size
    descriptors = page_bases | (page_size << 24)
    head_tiles = math.ceil(args.heads / 16)
    metadata_rows = batch * head_tiles if args.head_tiled_metadata else batch
    descriptors = descriptors[None].expand(metadata_rows, -1).contiguous()
    cache_indices = torch.arange(batch, device=device, dtype=torch.int32)
    seq_lens = torch.full(
        (metadata_rows,), logical_len, device=device, dtype=torch.int32
    )
    exact_counts = torch.full(
        (metadata_rows,), exact, device=device, dtype=torch.int32
    )
    local_lens = torch.full(
        (batch,), args.local, device=device, dtype=torch.int32
    )
    out = torch.empty(
        batch, args.heads, 512, device=device, dtype=torch.bfloat16
    )
    final_lse = torch.empty(
        batch, args.heads, device=device, dtype=torch.float32
    )
    scale = 576**-0.5
    opened_stamps = None
    sequence_epochs = None
    opened_by_tile: list[torch.Tensor] = []
    if args.head_tiled_metadata:
        opened_stamps = torch.zeros(
            metadata_rows, args.coarse, device=device, dtype=torch.int32
        )
        sequence_epochs = torch.arange(
            1, metadata_rows + 1, device=device, dtype=torch.int32
        )
        for tile in range(metadata_rows):
            opened = (
                torch.arange(8, device=device, dtype=torch.int64) * metadata_rows
                + tile
            ) % max(1, args.coarse)
            opened_by_tile.append(opened)
            if args.coarse:
                opened_stamps[tile, opened] = sequence_epochs[tile]

    def run() -> torch.Tensor:
        return absorbed_mla_lod_decode_gfx942(
            q,
            kv,
            bias,
            out,
            descriptors,
            fixed_indices,
            cache_indices,
            seq_lens,
            exact_counts,
            local_lens,
            scale,
            local_limit=args.local,
            opened_stamps=opened_stamps,
            sequence_epochs=sequence_epochs,
            state_capacity=args.coarse,
            sink_len=0,
            head_tiled_metadata=args.head_tiled_metadata,
            include_new=False,
            num_splits=args.splits,
            final_lse=final_lse,
        )

    run()
    torch.cuda.synchronize()
    logical_rows = fixed_indices[:, :logical_len].long()
    scores = torch.einsum(
        "bhd,bnd->bhn", q.float(), kv[logical_rows].float()
    ) * scale
    scores += bias[logical_rows][:, None]
    if args.head_tiled_metadata and args.coarse:
        for batch_index in range(batch):
            for head in range(args.heads):
                tile = batch_index * head_tiles + head // 16
                scores[
                    batch_index,
                    head,
                    args.local + opened_by_tile[tile],
                ] = -float("inf")
    ref = torch.einsum(
        "bhn,bnv->bhv",
        scores.softmax(dim=-1),
        kv[logical_rows, :512].float(),
    )
    actual = out.float()
    latency_ms = _time_ms(run, args.warmup, args.iterations)
    print(
        json.dumps(
            {
                "batch_size": batch,
                "heads": args.heads,
                "logical_tokens": logical_len,
                "coarse_rows": args.coarse,
                "local_rows": args.local,
                "exact_pages": args.exact_pages,
                "exact_rows": exact,
                "splits": args.splits,
                "head_tiled_metadata": args.head_tiled_metadata,
                "latency_ms": latency_ms,
                "max_abs_error": (actual - ref).abs().max().item(),
                "mean_abs_error": (actual - ref).abs().mean().item(),
                "max_lse_error": (
                    final_lse.float() - scores.logsumexp(dim=-1)
                ).abs().max().item(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
