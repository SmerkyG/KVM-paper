#!/usr/bin/env python3
"""Microbenchmark dense versus routed single-query key search on GPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _routed_block_max(
    query,
    keys,
    positions,
    block_scores,
    block_indices,
    sequence_len: tl.constexpr,
    candidate_count: tl.constexpr,
    head_dim: tl.constexpr,
    block_candidates: tl.constexpr,
):
    head = tl.program_id(0)
    candidate_block = tl.program_id(1)
    candidate_offsets = (
        candidate_block * block_candidates + tl.arange(0, block_candidates)
    )
    dimensions = tl.arange(0, head_dim)
    valid = candidate_offsets < candidate_count
    key_positions = tl.load(
        positions + head * candidate_count + candidate_offsets,
        mask=valid,
        other=0,
    )
    query_values = tl.load(query + head * head_dim + dimensions)
    key_values = tl.load(
        keys
        + head * sequence_len * head_dim
        + key_positions[:, None] * head_dim
        + dimensions[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    scores = tl.sum(key_values * query_values[None, :], axis=1)
    scores = tl.where(valid, scores, -float("inf"))
    local_index = tl.argmax(scores, axis=0)
    output_offset = head * tl.cdiv(candidate_count, block_candidates) + candidate_block
    tl.store(block_scores + output_offset, tl.max(scores, axis=0))
    tl.store(
        block_indices + output_offset,
        tl.max(
            tl.where(
                tl.arange(0, block_candidates) == local_index,
                key_positions,
                -1,
            ),
            axis=0,
        ),
    )


@triton.jit
def _routed_final_max(
    block_scores,
    block_indices,
    output_indices,
    block_count: tl.constexpr,
    reduction_size: tl.constexpr,
):
    head = tl.program_id(0)
    offsets = tl.arange(0, reduction_size)
    valid = offsets < block_count
    scores = tl.load(
        block_scores + head * block_count + offsets,
        mask=valid,
        other=-float("inf"),
    )
    winner = tl.argmax(scores, axis=0)
    output = tl.load(block_indices + head * block_count + winner)
    tl.store(output_indices + head, output)


@triton.jit
def _routed_single_max(
    query,
    keys,
    positions,
    output_indices,
    sequence_len: tl.constexpr,
    candidate_count: tl.constexpr,
    head_dim: tl.constexpr,
    candidate_block: tl.constexpr,
):
    head = tl.program_id(0)
    candidate_offsets = tl.arange(0, candidate_block)
    dimensions = tl.arange(0, head_dim)
    valid = candidate_offsets < candidate_count
    key_positions = tl.load(
        positions + head * candidate_count + candidate_offsets,
        mask=valid,
        other=0,
    )
    query_values = tl.load(query + head * head_dim + dimensions)
    key_values = tl.load(
        keys
        + head * sequence_len * head_dim
        + key_positions[:, None] * head_dim
        + dimensions[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    scores = tl.sum(key_values * query_values[None, :], axis=1)
    scores = tl.where(valid, scores, -float("inf"))
    winner = tl.argmax(scores, axis=0)
    output = tl.max(
        tl.where(candidate_offsets == winner, key_positions, -1), axis=0
    )
    tl.store(output_indices + head, output)


def triton_routed_search(
    query: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    heads, sequence_len, head_dim = keys.shape
    candidate_count = int(positions.size(1))
    block_candidates = 16
    block_count = triton.cdiv(candidate_count, block_candidates)
    block_scores = torch.empty(
        heads, block_count, device=keys.device, dtype=torch.float32
    )
    block_indices = torch.empty(
        heads, block_count, device=keys.device, dtype=torch.int32
    )
    output_indices = torch.empty(heads, device=keys.device, dtype=torch.int32)
    _routed_block_max[(heads, block_count)](
        query,
        keys,
        positions,
        block_scores,
        block_indices,
        sequence_len=sequence_len,
        candidate_count=candidate_count,
        head_dim=head_dim,
        block_candidates=block_candidates,
        num_warps=4,
    )
    _routed_final_max[(heads,)](
        block_scores,
        block_indices,
        output_indices,
        block_count=block_count,
        reduction_size=triton.next_power_of_2(block_count),
        num_warps=1,
    )
    return output_indices


def triton_single_routed_search(
    query: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    heads, sequence_len, head_dim = keys.shape
    candidate_count = int(positions.size(1))
    output_indices = torch.empty(heads, device=keys.device, dtype=torch.int32)
    _routed_single_max[(heads,)](
        query,
        keys,
        positions,
        output_indices,
        sequence_len=sequence_len,
        candidate_count=candidate_count,
        head_dim=head_dim,
        candidate_block=triton.next_power_of_2(candidate_count),
        num_warps=8,
    )
    return output_indices


def benchmark(fn, *, warmup: int = 100, repeats: int = 1000) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(begin.elapsed_time(end)) * 1000.0 / repeats


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    heads, sequence_len, head_dim = 6, 15616, 128
    keys = torch.randn(
        heads, sequence_len, head_dim, device=device, dtype=torch.bfloat16
    )
    query = torch.randn(heads, head_dim, device=device, dtype=torch.bfloat16)

    def dense_search() -> torch.Tensor:
        return torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1).argmax(dim=-1)

    print(f"dense_n={sequence_len} time_us={benchmark(dense_search):.3f}")
    state_keys = keys[:, :2031, :]

    def state_top4() -> torch.Tensor:
        return torch.bmm(
            query.unsqueeze(1), state_keys.transpose(1, 2)
        ).squeeze(1).topk(4, dim=-1).indices

    print(f"state_top4_n=2031 time_us={benchmark(state_top4):.3f}")
    for candidate_count in (64, 128, 230, 512, 1500, 4589):
        positions = torch.randint(
            sequence_len,
            (heads, candidate_count),
            device=device,
            dtype=torch.long,
        )

        def routed_search() -> torch.Tensor:
            selected = keys.gather(
                1,
                positions.unsqueeze(-1).expand(-1, -1, head_dim),
            )
            return (selected * query.unsqueeze(1)).sum(dim=-1).argmax(dim=-1)

        routed_us = benchmark(routed_search)
        triton_output = triton_routed_search(query, keys, positions)
        expected_output = positions.gather(
            1, routed_search().unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(
            triton_output.to(expected_output.dtype), expected_output
        )
        triton_us = benchmark(
            lambda: triton_routed_search(query, keys, positions)
        )
        single_label = ""
        if candidate_count <= 512:
            single_output = triton_single_routed_search(query, keys, positions)
            torch.testing.assert_close(
                single_output.to(expected_output.dtype), expected_output
            )
            single_us = benchmark(
                lambda: triton_single_routed_search(query, keys, positions)
            )
            single_label = f" single_triton_us={single_us:.3f}"
        print(
            f"routed_c={candidate_count} time_us={routed_us:.3f} "
            f"triton_us={triton_us:.3f} "
            f"dense_over_triton={benchmark(dense_search) / triton_us:.3f}"
            f"{single_label}"
        )

    for longer_length in (32768, 65536):
        longer_keys = torch.randn(
            heads,
            longer_length,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        def longer_dense_search() -> torch.Tensor:
            return torch.bmm(
                query.unsqueeze(1), longer_keys.transpose(1, 2)
            ).squeeze(1).argmax(dim=-1)

        print(
            f"dense_n={longer_length} "
            f"time_us={benchmark(longer_dense_search):.3f}"
        )


if __name__ == "__main__":
    main()
