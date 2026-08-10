from __future__ import annotations

import argparse

import torch

from model.kernels.lod_kernels import (
    balanced_bipartite_reduce_2to1,
    bipartite_reduce_overflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--fused", action="store_true")
    return parser.parse_args()


def reference(
    key_sum: torch.Tensor,
    value_sum: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, tokens, dim = key_sum.shape
    rows = batch * heads
    anchors = (tokens + 1) // 2
    sources = tokens // 2
    row = torch.arange(rows, device=key_sum.device).view(rows, 1)
    anchor = torch.arange(anchors, device=key_sum.device).view(1, anchors)
    source = torch.arange(sources, device=key_sum.device).view(1, sources)
    anchor_swap = (anchor + row) & 1
    anchor_swap = torch.where(2 * anchor + 1 < tokens, anchor_swap, 0)
    source_swap = (source + row) & 1
    anchor_token = 2 * anchor + anchor_swap
    source_token = 2 * source + 1 - source_swap
    flat_k = key_sum.reshape(rows, tokens, dim)
    flat_v = value_sum.reshape(rows, tokens, dim)
    flat_c = counts.reshape(rows, tokens, 1)

    def gather(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values,
            1,
            index.unsqueeze(-1).expand(rows, index.size(1), values.size(-1)),
        )

    anchor_k = gather(flat_k, anchor_token)
    source_k = gather(flat_k, source_token)
    anchor_c = gather(flat_c, anchor_token)
    source_c = gather(flat_c, source_token)
    score = torch.matmul(
        (source_k / source_c).to(torch.bfloat16),
        (anchor_k / anchor_c).to(torch.bfloat16).transpose(-1, -2),
    )
    destination = score.argmax(dim=-1)
    assignment = torch.nn.functional.one_hot(
        anchor_token, num_classes=tokens
    ).float()
    assignment.scatter_add_(
        2,
        source_token.unsqueeze(1).expand(rows, anchors, sources),
        torch.nn.functional.one_hot(destination, num_classes=anchors)
        .transpose(1, 2)
        .float(),
    )
    reduced_k = torch.matmul(assignment.to(flat_k.dtype), flat_k)
    reduced_v = torch.matmul(assignment.to(flat_v.dtype), flat_v)
    reduced_c = torch.matmul(assignment, flat_c.float())
    membership = torch.empty(rows, tokens, dtype=torch.long, device=key_sum.device)
    membership.scatter_(1, anchor_token, anchor.expand(rows, anchors))
    membership.scatter_(1, source_token, destination)
    return (
        reduced_k.reshape(batch, heads, anchors, dim),
        reduced_v.reshape(batch, heads, anchors, dim),
        reduced_c.reshape(batch, heads, anchors, 1),
        membership.reshape(batch, heads, tokens),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    shape = (args.batch_size, args.heads, args.tokens, args.dim)
    key_sum = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value_sum = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    counts = torch.randint(
        1,
        8,
        shape[:-1] + (1,),
        device="cuda",
        dtype=torch.int32,
    ).float()
    expected = reference(key_sum, value_sum, counts)
    if args.fused:
        if not torch.all(counts == 1):
            counts.fill_(1)
            expected = reference(key_sum, value_sum, counts)
        actual = bipartite_reduce_overflow(
            key_sum, value_sum, block_size=args.tokens, balanced=True
        )
    else:
        actual = balanced_bipartite_reduce_2to1(key_sum, value_sum, counts)
    torch.cuda.synchronize()
    for name, got, want in zip(
        ("key", "value", "count", "membership"), actual, expected, strict=True
    ):
        if got.dtype.is_floating_point:
            error = (got.float() - want.float()).abs().max().item()
            mismatched = int(
                (~torch.isclose(got.float(), want.float(), atol=0.125, rtol=0.01))
                .sum()
                .item()
            )
        else:
            error = 0.0
            mismatched = int((got != want).sum().item())
        print(
            f"{name}: max_abs_error={error:.6f} mismatched={mismatched}/{got.numel()}"
        )
    for got, want in zip(actual, expected, strict=True):
        if got.dtype.is_floating_point:
            torch.testing.assert_close(got, want, atol=0.125, rtol=0.01)
        else:
            torch.testing.assert_close(got, want)


if __name__ == "__main__":
    main()
