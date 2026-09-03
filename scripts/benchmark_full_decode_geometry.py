#!/usr/bin/env python3
"""Benchmark one batch-eight full-attention decode layer by head geometry."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Geometry:
    name: str
    query_heads: int
    kv_heads: int
    head_dim: int


GEOMETRIES = (
    Geometry("muse", 32, 2, 128),
    Geometry("olmo", 40, 8, 128),
    Geometry("phi", 8, 2, 128),
    Geometry("qwen", 24, 4, 256),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lengths", type=int, nargs="+",
        default=(8192, 16384, 32768, 65536, 131072),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--geometry")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--window-size",
        type=int,
        default=-1,
        help="AITER left sliding-window size; -1 selects full attention.",
    )
    parser.add_argument(
        "--random-block-table",
        action="store_true",
        help="Permute physical page IDs to model a recycled sliding cache.",
    )
    parser.add_argument(
        "--interleaved-block-table",
        action="store_true",
        help="Allocate consecutive physical pages round-robin across requests.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def benchmark(
    geometry: Geometry,
    length: int,
    *,
    batch_size: int,
    warmup: int,
    repeats: int,
    block_size: int,
    window_size: int,
    random_block_table: bool = False,
    interleaved_block_table: bool = False,
) -> dict[str, object]:
    from aiter.ops.triton.unified_attention import unified_attention

    device = torch.device("cuda", 0)
    blocks = (length + block_size - 1) // block_size
    generator = torch.Generator(device=device)
    generator.manual_seed(20260821 + geometry.head_dim + length)
    q = torch.randn(
        batch_size,
        geometry.query_heads,
        geometry.head_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cache_shape = (
        batch_size * blocks,
        block_size,
        geometry.kv_heads,
        geometry.head_dim,
    )
    k = torch.randn(
        cache_shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    v = torch.randn(
        cache_shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    out = torch.empty_like(q)
    block_ids = torch.arange(batch_size * blocks, dtype=torch.int32, device=device)
    if random_block_table:
        block_ids = block_ids[
            torch.randperm(
                batch_size * blocks, device=device, generator=generator
            )
        ]
    if interleaved_block_table:
        block_table = block_ids.reshape(blocks, batch_size).T.contiguous()
    else:
        block_table = block_ids.reshape(batch_size, blocks)
    cu_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    lengths = torch.full(
        (batch_size,), length, dtype=torch.int32, device=device
    )

    def run() -> None:
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=lengths,
            max_seqlen_k=length,
            softmax_scale=geometry.head_dim**-0.5,
            causal=True,
            window_size=(window_size - 1, 0) if window_size > 0 else (-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        run()
        end.record()
        end.synchronize()
        samples.append(float(begin.elapsed_time(end)))
    return {
        "geometry": asdict(geometry),
        "length": length,
        "block_size": block_size,
        "window_size": window_size,
        "random_block_table": random_block_table,
        "interleaved_block_table": interleaved_block_table,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    geometries = [
        geometry for geometry in GEOMETRIES
        if not args.geometry or args.geometry in geometry.name
    ]
    records = []
    for geometry in geometries:
        for length in args.lengths:
            record = benchmark(
                geometry,
                length,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
                block_size=args.block_size,
                window_size=args.window_size,
                random_block_table=args.random_block_table,
                interleaved_block_table=args.interleaved_block_table,
            )
            records.append(record)
            print(
                geometry.name,
                length,
                f"{record['median_ms'] * 1000:.1f} us",
                flush=True,
            )
    output = {
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
