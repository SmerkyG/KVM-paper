"""Exact global routing from fixed-size rank-local top-k candidates.

Decode context parallelism (DCP) shards the chronological KV cache across
ranks.  LoD therefore routes against rank-local centroids first.  If every
rank contributes its local top ``k``, the global top ``k`` is guaranteed to
be in their union: a candidate omitted by its owner rank already has at
least ``k`` better candidates on that rank alone.

The communication record is deliberately fixed-size and graph friendly.
Each rank contributes ``[score, local_slot]`` pairs; after one all-gather a
small reducer returns both the owner rank and its local slot.  Leaves never
move between ranks.  The owner refines the selected centroid locally and an
ordinary LSE-aware DCP reduction combines the rank-local attention results.
"""

from __future__ import annotations

from typing import Protocol

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only documentation installs.
    triton = None
    tl = None


class AllGatherGroup(Protocol):
    """The narrow vLLM process-group interface needed by routing."""

    world_size: int
    rank_in_group: int

    def all_gather(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor: ...


if triton is not None:

    @triton.jit
    def _reduce_gathered_rank_topk_kernel(
        gathered_ptr,
        output_scores_ptr,
        output_owners_ptr,
        output_slots_ptr,
        gathered_stride_row: tl.constexpr,
        output_stride_row: tl.constexpr,
        CANDIDATES: tl.constexpr,
        LOCAL_K: tl.constexpr,
        OUTPUT_K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        valid_candidate = offsets < CANDIDATES
        base = row * gathered_stride_row + offsets * 2
        scores = tl.load(
            gathered_ptr + base,
            mask=valid_candidate,
            other=-float("inf"),
        ).to(tl.float32)
        slots = tl.load(
            gathered_ptr + base + 1,
            mask=valid_candidate,
            other=-1.0,
        ).to(tl.int32)
        valid_candidate &= slots >= 0
        scores = tl.where(valid_candidate, scores, -float("inf"))
        sentinel = CANDIDATES

        # OUTPUT_K is eight in the release path.  Repeated max selection over
        # 64 candidates is cheaper than a general sort and gives deterministic
        # rank-major tie breaking.
        for output_index in tl.static_range(0, OUTPUT_K):
            best_score = tl.max(scores, axis=0)
            tied = valid_candidate & (scores == best_score)
            best_candidate = tl.min(
                tl.where(tied, offsets, sentinel), axis=0
            )
            has_winner = best_candidate < sentinel
            best_slot = tl.sum(
                tl.where(offsets == best_candidate, slots, 0), axis=0
            )
            output_offset = row * output_stride_row + output_index
            tl.store(
                output_scores_ptr + output_offset,
                tl.where(has_winner, best_score, -float("inf")),
            )
            tl.store(
                output_owners_ptr + output_offset,
                tl.where(has_winner, best_candidate // LOCAL_K, -1),
            )
            tl.store(
                output_slots_ptr + output_offset,
                tl.where(has_winner, best_slot, -1),
            )
            selected = has_winner & (offsets == best_candidate)
            scores = tl.where(selected, -float("inf"), scores)
            valid_candidate &= ~selected


def pack_local_topk(
    local_scores: torch.Tensor,
    local_slots: torch.Tensor,
) -> torch.Tensor:
    """Pack fixed local candidates for one collective.

    Slot indices are represented exactly as fp32 integers.  LoD state slots
    are many orders of magnitude below fp32's exact-integer limit (2**24).
    """

    if local_scores.shape != local_slots.shape:
        raise ValueError("local route scores and slots must have equal shapes")
    if local_scores.ndim < 1:
        raise ValueError("local route candidates require a top-k dimension")
    if local_slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("local route slots must be integer tensors")
    return torch.stack(
        (local_scores.to(torch.float32), local_slots.to(torch.float32)), dim=-1
    )


def reduce_gathered_rank_topk(
    gathered: torch.Tensor,
    *,
    local_k: int,
    output_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select exact global top-k from rank-major gathered candidates.

    ``gathered`` has shape ``[..., world_size * local_k, 2]``.  The final
    coordinate contains ``(score, local_slot)``.  Returns fp32 scores and
    int32 owner ranks/local slots, each shaped ``[..., output_k]``.
    """

    if gathered.ndim < 2 or gathered.size(-1) != 2:
        raise ValueError("gathered candidates must end in [candidate, 2]")
    candidate_count = int(gathered.size(-2))
    if local_k <= 0 or candidate_count % local_k:
        raise ValueError("candidate count must be divisible by local_k")
    if output_k is None:
        output_k = local_k
    if output_k <= 0 or output_k > candidate_count:
        raise ValueError("output_k is outside the gathered candidate range")
    if candidate_count > 128:
        raise ValueError("the decode reducer supports at most 128 candidates")

    leading_shape = gathered.shape[:-2]
    scores_out = torch.empty(
        (*leading_shape, output_k),
        dtype=torch.float32,
        device=gathered.device,
    )
    owners_out = torch.empty_like(scores_out, dtype=torch.int32)
    slots_out = torch.empty_like(scores_out, dtype=torch.int32)

    if gathered.device.type == "cuda" and triton is not None:
        contiguous = gathered.contiguous()
        rows = contiguous.numel() // (candidate_count * 2)
        block = triton.next_power_of_2(candidate_count)
        _reduce_gathered_rank_topk_kernel[(rows,)](
            contiguous,
            scores_out,
            owners_out,
            slots_out,
            candidate_count * 2,
            output_k,
            CANDIDATES=candidate_count,
            LOCAL_K=local_k,
            OUTPUT_K=output_k,
            BLOCK=block,
            num_warps=1 if block <= 64 else 2,
            waves_per_eu=1,
        )
        return scores_out, owners_out, slots_out

    scores = gathered[..., 0]
    slots = gathered[..., 1].to(torch.int32)
    scores = scores.masked_fill(slots < 0, -float("inf"))
    # Stable rank-major ordering supplies deterministic ties on CPU and on
    # the simple PyTorch reference path.
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    order = order[..., :output_k]
    scores_out.copy_(torch.gather(scores, -1, order))
    slots_out.copy_(torch.gather(slots, -1, order))
    owners_out.copy_((order // local_k).to(torch.int32))
    invalid = ~torch.isfinite(scores_out)
    owners_out.masked_fill_(invalid, -1)
    slots_out.masked_fill_(invalid, -1)
    return scores_out, owners_out, slots_out


def distributed_global_topk(
    local_scores: torch.Tensor,
    local_slots: torch.Tensor,
    group: AllGatherGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-gather rank-local top-k and recover the exact global top-k."""

    local_k = int(local_scores.size(-1))
    packed = pack_local_topk(local_scores, local_slots)
    gathered = group.all_gather(packed, dim=-2)
    expected = local_k * int(group.world_size)
    if int(gathered.size(-2)) != expected:
        raise RuntimeError(
            f"DCP route all-gather returned {gathered.size(-2)} candidates; "
            f"expected {expected}"
        )
    return reduce_gathered_rank_topk(gathered, local_k=local_k)


def localize_global_topk(
    global_scores: torch.Tensor,
    global_owners: torch.Tensor,
    global_slots: torch.Tensor,
    *,
    local_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep globally selected routes owned by this rank and mask the rest."""

    if not (
        global_scores.shape == global_owners.shape == global_slots.shape
    ):
        raise ValueError("global route score, owner, and slot shapes must match")
    owned = global_owners == int(local_rank)
    local_scores = torch.where(
        owned, global_scores, torch.full_like(global_scores, -float("inf"))
    )
    local_slots = torch.where(
        owned, global_slots, torch.full_like(global_slots, -1)
    )
    return local_scores, local_slots


__all__ = [
    "distributed_global_topk",
    "localize_global_topk",
    "pack_local_topk",
    "reduce_gathered_rank_topk",
]
