from __future__ import annotations

import torch

from lod_attention.kernels.distributed_topk import (
    distributed_global_topk,
    localize_global_topk,
    pack_local_topk,
    reduce_gathered_rank_topk,
)


class _GatheredGroup:
    def __init__(self, candidates: torch.Tensor, rank: int = 0) -> None:
        self.candidates = candidates
        self.world_size = int(candidates.size(-2) // 8)
        self.rank_in_group = rank

    def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert value.shape[-2:] == (8, 2)
        assert dim in (-2, value.ndim - 2)
        return self.candidates


def _local_candidates(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    local_scores, local_slots = torch.topk(scores, k, dim=-1, sorted=True)
    return local_scores, local_slots.to(torch.int32)


def test_rank_local_topk_union_contains_exact_global_topk() -> None:
    generator = torch.Generator().manual_seed(20261001)
    world, batch, heads, slots, k = 8, 3, 12, 97, 8
    scores = torch.randn(world, batch, heads, slots, generator=generator)
    local_scores, local_slots = _local_candidates(scores, k)
    gathered = torch.cat(
        [pack_local_topk(local_scores[r], local_slots[r]) for r in range(world)],
        dim=-2,
    )
    selected_scores, owners, selected_slots = reduce_gathered_rank_topk(
        gathered, local_k=k
    )

    dense = scores.permute(1, 2, 0, 3).reshape(batch, heads, world * slots)
    expected_scores, expected_indices = torch.topk(dense, k, dim=-1, sorted=True)
    assert torch.equal(selected_scores, expected_scores)
    assert torch.equal(owners, (expected_indices // slots).to(torch.int32))
    assert torch.equal(selected_slots, (expected_indices % slots).to(torch.int32))


def test_global_topk_invalid_candidates_and_local_ownership() -> None:
    local_scores = torch.tensor(
        [
            [[9.0, 7.0, -float("inf"), -float("inf")]],
            [[8.0, 6.0, 5.0, -float("inf")]],
        ]
    )
    local_slots = torch.tensor(
        [
            [[3, 1, -1, -1]],
            [[4, 2, 0, -1]],
        ],
        dtype=torch.int32,
    )
    gathered = torch.cat(
        [pack_local_topk(local_scores[r], local_slots[r]) for r in range(2)],
        dim=-2,
    )
    scores, owners, slots = reduce_gathered_rank_topk(
        gathered, local_k=4, output_k=4
    )
    assert scores.tolist() == [[9.0, 8.0, 7.0, 6.0]]
    assert owners.tolist() == [[0, 1, 0, 1]]
    assert slots.tolist() == [[3, 4, 1, 2]]

    local_scores_1, local_slots_1 = localize_global_topk(
        scores, owners, slots, local_rank=1
    )
    assert local_slots_1.tolist() == [[-1, 4, -1, 2]]
    assert torch.isneginf(local_scores_1[..., 0]).all()
    assert torch.isneginf(local_scores_1[..., 2]).all()


def test_global_topk_ties_are_rank_major_deterministic() -> None:
    gathered = torch.tensor(
        [[[[5.0, 7.0], [5.0, 2.0], [5.0, 9.0], [4.0, 0.0]]]]
    )
    scores, owners, slots = reduce_gathered_rank_topk(
        gathered, local_k=2, output_k=3
    )
    assert scores.tolist() == [[[5.0, 5.0, 5.0]]]
    assert owners.tolist() == [[[0, 0, 1]]]
    assert slots.tolist() == [[[7, 2, 9]]]


def test_distributed_entry_point_uses_one_packed_collective() -> None:
    scores = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)
    slots = torch.arange(64, dtype=torch.int32).remainder(8).reshape(1, 1, 64)
    gathered = torch.stack((scores, slots.float()), dim=-1)
    group = _GatheredGroup(gathered)
    selected_scores, owners, selected_slots = distributed_global_topk(
        scores[..., :8], slots[..., :8], group
    )
    assert selected_scores.tolist() == [[[63.0, 62.0, 61.0, 60.0, 59.0, 58.0, 57.0, 56.0]]]
    assert owners.tolist() == [[[7, 7, 7, 7, 7, 7, 7, 7]]]
    assert selected_slots.tolist() == [[[7, 6, 5, 4, 3, 2, 1, 0]]]
