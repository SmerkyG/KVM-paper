"""Small tensor helpers shared by LoD state construction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_by_index(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expanded = index.unsqueeze(-1).expand(-1, -1, -1, int(x.size(-1)))
    return x.gather(2, expanded)


def all_indices(x: torch.Tensor, length: int) -> torch.Tensor:
    return (
        torch.arange(length, device=x.device, dtype=torch.long)
        .view(1, 1, length)
        .expand(int(x.size(0)), int(x.size(1)), length)
    )


def split_append_merge_indices(
    block_keys: torch.Tensor,
    append_count: int,
    reference_keys: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the keys least similar to the existing state; merge the rest."""

    length = int(block_keys.size(2))
    append_count = min(max(int(append_count), 0), length)
    merge_count = length - append_count
    every = all_indices(block_keys, length)
    empty = every[:, :, :0]
    if length == 0:
        return empty, empty
    if append_count == 0:
        return empty, every
    if merge_count == 0:
        return every, empty
    if int(reference_keys.size(2)) == 0:
        return every[:, :, merge_count:], every[:, :, :merge_count]
    with torch.no_grad():
        similarity = torch.matmul(
            block_keys, reference_keys.transpose(-1, -2)
        ).float().amax(dim=-1)
        ordered = torch.argsort(similarity, dim=-1, descending=False)
        appended, _ = torch.sort(ordered[..., :append_count], dim=-1)
        merged, _ = torch.sort(ordered[..., append_count:], dim=-1)
    return appended, merged


def sum_adjacent_groups(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return tensor
    length = int(tensor.size(2))
    groups = (length + factor - 1) // factor
    padded = groups * factor
    if padded != length:
        tensor = F.pad(tensor, (0, 0, 0, padded - length))
    return tensor.reshape(
        *tensor.shape[:2], groups, factor, int(tensor.size(-1))
    ).sum(dim=3)


def premerge_adjacent_state_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if factor not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("adjacent premerge factor must be a power of two through 32")
    length = int(key.size(2))
    if length != int(value.size(2)):
        raise ValueError("adjacent state K/V lengths differ")
    grouped_key = sum_adjacent_groups(key, factor)
    grouped_value = sum_adjacent_groups(value, factor)
    groups = int(grouped_key.size(2))
    count = torch.full(
        (*key.shape[:2], groups, 1),
        float(factor),
        dtype=torch.float32,
        device=key.device,
    )
    if groups and length % factor:
        count[..., -1, 0] = float(length % factor)
    membership = (
        torch.arange(length, device=key.device, dtype=torch.long)
        .div(factor, rounding_mode="floor")
        .view(1, 1, length)
        .expand(*key.shape[:2], length)
    )
    return grouped_key, grouped_value, count, membership

