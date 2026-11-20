"""Deterministic two-pass compaction for the Squeezed GQA decode port."""
import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _counts(Mask, TileCounts, N: tl.constexpr, Tiles: tl.constexpr, B: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    pos = tile * B + tl.arange(0, B)
    active = tl.load(Mask + head * N + pos, mask=pos < N, other=False)
    tl.store(TileCounts + head * Tiles + tile, tl.sum(active.to(tl.int32), 0))


@triton.jit
def _pack(Mask, TileCounts, Indices, Counts, N: tl.constexpr, Tiles: tl.constexpr, B: tl.constexpr, TB: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    tiles = tl.arange(0, TB)
    counts = tl.load(TileCounts + head * Tiles + tiles, mask=tiles < Tiles, other=0)
    start = tl.sum(tl.where(tiles < tile, counts, 0), 0)
    pos = tile * B + tl.arange(0, B)
    active = tl.load(Mask + head * N + pos, mask=pos < N, other=False)
    offset = start + tl.cumsum(active.to(tl.int32), 0) - 1
    tl.store(Indices + head * N + offset, pos, mask=active != 0)
    if tile == 0:
        tl.store(Counts + head, tl.sum(counts, 0))


def compact(mask, indices, counts, tiles):
    """Retain the author's selected set and order it by prefix position."""
    _, heads, n = mask.shape
    b = 4096
    blocks = triton.cdiv(n, b)
    _counts[(heads, blocks)](mask, tiles, n, blocks, b, num_warps=4)
    _pack[(heads, blocks)](mask, tiles, indices, counts, n, blocks, b,
                            triton.next_power_of_2(blocks), num_warps=4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(2)
    cases = []
    for n in (4109, 32768, 128000):
        mask = torch.rand((1, 32, n), device='cuda') > .9
        mask[0, 0].zero_()
        mask[0, 1].fill_(True)
        indices = torch.empty_like(mask, dtype=torch.int64)
        counts = torch.empty((1, 32), dtype=torch.int32, device='cuda')
        tiles = torch.empty((32, triton.cdiv(n, 4096)), dtype=torch.int32, device='cuda')
        compact(mask, indices, counts, tiles)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            compact(mask, indices, counts, tiles)
        g.replay()
        for h in range(32):
            expected = mask[0, h].nonzero().flatten()
            assert int(counts[0, h]) == expected.numel()
            assert torch.equal(indices[0, h, :expected.numel()], expected)
        cases.append(dict(length=n, heads=32, exact=True))
    a.output.write_text(json.dumps(dict(stage='complete', cases=cases), indent=2) + '\n')


if __name__ == '__main__':
    main()
