"""Recursive paged, optionally INT4, pure-PyTorch LOD attention.

This is a small storage-focused variant of :mod:`pytorch_lod_attention`.
State updates, local attention, and LSE merging are reused unchanged.  With a
positive page size, each routed state region is split into logical pages.  The
highest-mass page is attended exactly and the rest of that region is retained
as one count-corrected residual summary.  Thus a recursive route reads one page
per region without dropping or double-counting its remaining leaves.

Physical K/V storage is independent of those logical region pages:

* ``page_size=None`` uses the existing flat BF16 archive;
* a positive ``page_size`` recursively opens region pages while storing
  completed chronological K/V pages separately from the unfinished BF16 tail;
* ``kv_bits=4`` packs completed pages into two signed nibbles per byte.

INT4 pages use one BF16 mean anchor per page and one BF16 symmetric residual
scale per channel group.  The unfinished page stays BF16, so cached decode does
not requantize the whole archive for every token.  Small routed leaf sets are
gathered and dequantized directly.  Large prefill sets fall back to the existing
packed variable-length attention after materializing the archive once.

As with the fast flat implementation, this module is intended for inference.
Inputs and outputs use the same post-QKV/post-RoPE head-separated layout as
``TwoLevelLODAttention``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .pytorch_lod_attention import (
    LODAttentionResult,
    LODConfig,
    LODState,
    TwoLevelLODAttention,
    _attention_from_scores,
    _repeat_kv,
)
from .pytorch_lod_attention_fast import (
    _FastLocalMixin,
    _attention_needs_grad,
    _fast_coarse_attention,
    _merge_two_branches,
    _posting_lists,
    _route_state,
    fast_two_level_lod_attention,
)


@dataclass(frozen=True)
class PagedLODConfig(LODConfig):
    """LOD settings plus optional chronological pages and INT4 storage."""

    page_size: int | None = 16
    kv_bits: int = 0
    quant_group_size: int = 32

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.page_size is not None and self.page_size <= 0:
            raise ValueError("page_size must be positive or None")
        if self.page_size is not None and self.chunk_size % self.page_size:
            raise ValueError("page_size must divide chunk_size for causal prefill")
        if self.kv_bits not in (0, 4):
            raise ValueError("kv_bits must be zero or four")
        if self.kv_bits and self.page_size is None:
            raise ValueError("INT4 storage requires pages")
        if self.quant_group_size <= 0:
            raise ValueError("quant_group_size must be positive")


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _quantize_pages_int4(
    pages: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``[B,H,P,L,D]`` pages using page means and groupwise scales."""
    dimension = int(pages.size(-1))
    if dimension % 2:
        raise ValueError("INT4 K/V dimensions must be even")
    if dimension % group_size:
        raise ValueError("quant_group_size must divide each K/V dimension")
    if int(pages.size(2)) == 0:
        shape = (*pages.shape[:-1], dimension // 2)
        groups = dimension // group_size
        return (
            torch.empty(shape, dtype=torch.uint8, device=pages.device),
            pages.new_empty(*pages.shape[:3], dimension),
            pages.new_empty(*pages.shape[:3], groups),
        )

    working = pages.float()
    anchor = working.mean(dim=3)
    residual = working - anchor.unsqueeze(3)
    groups = dimension // group_size
    grouped = residual.reshape(*residual.shape[:-1], groups, group_size)
    scale = grouped.abs().amax(dim=(3, 5)).div(7.0)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    restored_scale = scale.unsqueeze(3).unsqueeze(-1)
    quantized = (grouped / restored_scale).round().clamp(-7, 7).to(torch.int16)
    quantized = quantized.reshape_as(residual) + 8
    packed = (
        quantized[..., 0::2] | (quantized[..., 1::2] << 4)
    ).to(torch.uint8)
    return packed, anchor.to(pages.dtype), scale.to(pages.dtype)


@dataclass
class PagedTensor:
    """Completed pages plus an unfinished BF16/FP16 tail for one K or V."""

    pages: torch.Tensor
    tail: torch.Tensor
    length: int
    page_size: int
    bits: int
    group_size: int
    dimension: int
    anchor: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        page_size: int,
        bits: int,
        group_size: int,
        dtype: torch.dtype,
    ) -> PagedTensor:
        tensor = tensor.to(dtype)
        length = int(tensor.size(2))
        complete = length // page_size * page_size
        page = tensor[..., :complete, :].reshape(
            *tensor.shape[:2], complete // page_size, page_size, int(tensor.size(-1))
        )
        tail = tensor[..., complete:, :]
        if bits == 4:
            packed, anchor, scale = _quantize_pages_int4(page, group_size)
            page = packed
        else:
            anchor = scale = None
        return cls(
            pages=page,
            tail=tail,
            length=length,
            page_size=page_size,
            bits=bits,
            group_size=group_size,
            dimension=int(tensor.size(-1)),
            anchor=anchor,
            scale=scale,
        )

    @property
    def complete_pages(self) -> int:
        return int(self.pages.size(2))

    @property
    def complete_tokens(self) -> int:
        return self.complete_pages * self.page_size

    @property
    def dtype(self) -> torch.dtype:
        return self.tail.dtype

    @property
    def storage_bytes(self) -> int:
        return sum(
            _tensor_bytes(tensor)
            for tensor in (self.pages, self.tail, self.anchor, self.scale)
        )

    def detached(self) -> PagedTensor:
        return PagedTensor(
            pages=self.pages.detach(),
            tail=self.tail.detach(),
            length=self.length,
            page_size=self.page_size,
            bits=self.bits,
            group_size=self.group_size,
            dimension=self.dimension,
            anchor=None if self.anchor is None else self.anchor.detach(),
            scale=None if self.scale is None else self.scale.detach(),
        )

    def append(self, tensor: torch.Tensor) -> PagedTensor:
        if (
            tensor.shape[:2] != self.tail.shape[:2]
            or int(tensor.size(-1)) != self.dimension
        ):
            raise ValueError("appended paged tensor has an incompatible shape")
        tensor = tensor.to(self.dtype)
        combined = torch.cat((self.tail, tensor), dim=2)
        complete = int(combined.size(2)) // self.page_size * self.page_size
        if complete:
            new_page = combined[..., :complete, :].reshape(
                *combined.shape[:2],
                complete // self.page_size,
                self.page_size,
                self.dimension,
            )
            if self.bits == 4:
                packed, anchor, scale = _quantize_pages_int4(
                    new_page, self.group_size
                )
                pages = torch.cat((self.pages, packed), dim=2)
                if self.anchor is None or self.scale is None:
                    raise AssertionError("INT4 page metadata is missing")
                next_anchor = torch.cat((self.anchor, anchor), dim=2)
                next_scale = torch.cat((self.scale, scale), dim=2)
            else:
                pages = torch.cat((self.pages, new_page), dim=2)
                next_anchor = next_scale = None
        else:
            pages = self.pages
            next_anchor = self.anchor
            next_scale = self.scale
        return PagedTensor(
            pages=pages,
            tail=combined[..., complete:, :],
            length=self.length + int(tensor.size(2)),
            page_size=self.page_size,
            bits=self.bits,
            group_size=self.group_size,
            dimension=self.dimension,
            anchor=next_anchor,
            scale=next_scale,
        )

    def _gather_complete(
        self, position: torch.Tensor, kv_head: torch.Tensor
    ) -> torch.Tensor:
        batch, query_heads = position.shape[:2]
        flat_position = position.reshape(batch, query_heads, -1)
        if self.bits == 0:
            source = self.pages[:, kv_head].flatten(2, 3)
            selected = source.gather(
                2,
                flat_position.unsqueeze(-1).expand(
                    batch, query_heads, flat_position.size(-1), self.dimension
                ),
            )
            return selected.reshape(*position.shape, self.dimension)

        if self.anchor is None or self.scale is None:
            raise AssertionError("INT4 page metadata is missing")
        packed_dimension = self.dimension // 2
        source = self.pages[:, kv_head].flatten(2, 3)
        packed = source.gather(
            2,
            flat_position.unsqueeze(-1).expand(
                batch, query_heads, flat_position.size(-1), packed_dimension
            ),
        )
        unpacked = torch.stack((packed & 15, packed >> 4), dim=-1)
        unpacked = unpacked.reshape(
            batch, query_heads, flat_position.size(-1), self.dimension
        ).float() - 8.0

        page = torch.div(flat_position, self.page_size, rounding_mode="floor")
        anchor = self.anchor[:, kv_head].gather(
            2,
            page.unsqueeze(-1).expand(
                batch, query_heads, flat_position.size(-1), self.dimension
            ),
        )
        groups = self.dimension // self.group_size
        scale = self.scale[:, kv_head].gather(
            2,
            page.unsqueeze(-1).expand(
                batch, query_heads, flat_position.size(-1), groups
            ),
        )
        scale = scale.repeat_interleave(self.group_size, dim=-1)
        restored = unpacked * scale.float() + anchor.float()
        return restored.to(self.dtype).reshape(*position.shape, self.dimension)

    def gather(self, position: torch.Tensor, query_heads: int) -> torch.Tensor:
        """Gather arbitrary chronological positions without expanding all pages."""
        if position.ndim < 3 or int(position.size(0)) != int(self.tail.size(0)):
            raise ValueError("positions must begin with [batch, query_heads]")
        if int(position.size(1)) != query_heads:
            raise ValueError("position query-head count is inconsistent")
        # Recursive GPU callers construct positions from the cache itself;
        # avoid a device synchronization solely for defensive bounds checks.
        if position.device.type == "cpu" and (
            bool((position < 0).any().item())
            or bool((position >= self.length).any().item())
        ):
            raise ValueError("paged gather position is outside the archive")
        key_value_heads = int(self.tail.size(1))
        if query_heads % key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        groups = query_heads // key_value_heads
        kv_head = torch.div(
            torch.arange(query_heads, device=position.device),
            groups,
            rounding_mode="floor",
        )

        complete = self.complete_tokens
        if complete:
            complete_position = position.clamp_max(complete - 1)
            selected_complete = self._gather_complete(complete_position, kv_head)
        else:
            selected_complete = None
        tail_length = int(self.tail.size(2))
        if tail_length:
            batch = int(position.size(0))
            flat = position.reshape(batch, query_heads, -1)
            tail_position = (flat - complete).clamp(0, tail_length - 1)
            source = self.tail[:, kv_head]
            selected_tail = source.gather(
                2,
                tail_position.unsqueeze(-1).expand(
                    batch, query_heads, flat.size(-1), self.dimension
                ),
            ).reshape(*position.shape, self.dimension)
        else:
            selected_tail = None
        if selected_complete is None:
            if selected_tail is None:
                raise AssertionError("cannot gather an empty paged tensor")
            return selected_tail
        if selected_tail is None:
            return selected_complete
        return torch.where(
            (position < complete).unsqueeze(-1), selected_complete, selected_tail
        )

    def materialize(self) -> torch.Tensor:
        """Restore the flat archive, used only by reference/large-set fallbacks."""
        if self.complete_pages:
            if self.bits == 0:
                complete = self.pages.flatten(2, 3)
            else:
                position = torch.arange(
                    self.complete_tokens, device=self.tail.device
                )
                position = position.view(1, 1, -1).expand(
                    int(self.tail.size(0)), int(self.tail.size(1)), -1
                )
                complete = self.gather(position, int(self.tail.size(1)))
        else:
            complete = self.tail[..., :0, :]
        return torch.cat((complete, self.tail), dim=2)


@dataclass
class PagedKVCache:
    """Paged K and V archives with matching chronological positions."""

    key: PagedTensor
    value: PagedTensor

    @classmethod
    def from_tensors(
        cls, key: torch.Tensor, value: torch.Tensor, config: PagedLODConfig
    ) -> PagedKVCache:
        if config.page_size is None:
            raise ValueError("paged cache requires a page size")
        return cls(
            key=PagedTensor.from_tensor(
                key,
                page_size=config.page_size,
                bits=config.kv_bits,
                group_size=config.quant_group_size,
                dtype=config.leaf_dtype,
            ),
            value=PagedTensor.from_tensor(
                value,
                page_size=config.page_size,
                bits=config.kv_bits,
                group_size=config.quant_group_size,
                dtype=config.leaf_dtype,
            ),
        )

    @property
    def length(self) -> int:
        if self.key.length != self.value.length:
            raise AssertionError("paged K/V lengths differ")
        return self.key.length

    @property
    def storage_bytes(self) -> int:
        return self.key.storage_bytes + self.value.storage_bytes

    def append(self, key: torch.Tensor, value: torch.Tensor) -> PagedKVCache:
        return PagedKVCache(self.key.append(key), self.value.append(value))

    def detached(self) -> PagedKVCache:
        return PagedKVCache(self.key.detached(), self.value.detached())

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key.materialize(), self.value.materialize()


@dataclass
class PagedLODCache:
    """State, local window, ownership, and paged exact leaves for decode."""

    state: LODState
    coverage: int
    recent_key: torch.Tensor
    recent_value: torch.Tensor
    total_length: int
    owner: torch.Tensor
    leaves: PagedKVCache

    def detached(self) -> PagedLODCache:
        return PagedLODCache(
            state=self.state.detached(),
            coverage=self.coverage,
            recent_key=self.recent_key.detach(),
            recent_value=self.recent_value.detach(),
            total_length=self.total_length,
            owner=self.owner.detach(),
            leaves=self.leaves.detached(),
        )


@dataclass
class RegionPages:
    """Logical, region-owned pages over positions in the physical KV archive."""

    position: torch.Tensor
    key_sum: torch.Tensor
    value_sum: torch.Tensor
    count: torch.Tensor
    slot_start: torch.Tensor
    slot_count: torch.Tensor


def build_region_pages(
    owner: torch.Tensor,
    state: LODState,
    leaves: PagedKVCache,
    page_size: int,
) -> RegionPages:
    """Stably partition each state region's postings into logical pages."""
    batch, key_value_heads, history_length = owner.shape
    slot_leaf_count = state.count.round().to(torch.long)
    slot_page_count = torch.div(
        slot_leaf_count + page_size - 1, page_size, rounding_mode="floor"
    )
    slot_page_start = slot_page_count.cumsum(-1) - slot_page_count
    page_capacity = int(slot_page_count.sum(-1).max().item())
    position = torch.full(
        (batch, key_value_heads, page_capacity, page_size),
        -1,
        dtype=torch.long,
        device=owner.device,
    )
    key_sum = state.key_sum.new_zeros(
        batch, key_value_heads, page_capacity, int(state.key_sum.size(-1))
    )
    value_sum = state.value_sum.new_zeros(
        batch, key_value_heads, page_capacity, int(state.value_sum.size(-1))
    )
    page_count = torch.zeros(
        batch,
        key_value_heads,
        page_capacity,
        dtype=torch.long,
        device=owner.device,
    )
    if history_length == 0:
        return RegionPages(
            position,
            key_sum,
            value_sum,
            page_count,
            slot_page_start,
            slot_page_count,
        )

    with torch.no_grad():
        order = owner.argsort(dim=-1, stable=True)
        sorted_slot = owner.gather(-1, order)
        slot_leaf_start = slot_leaf_count.cumsum(-1) - slot_leaf_count
        rank = torch.arange(history_length, device=owner.device)
        rank = rank.view(1, 1, -1) - slot_leaf_start.gather(-1, sorted_slot)
        page = slot_page_start.gather(-1, sorted_slot) + torch.div(
            rank, page_size, rounding_mode="floor"
        )
        offset = rank.remainder(page_size)
        batch_index = torch.arange(batch, device=owner.device).view(-1, 1, 1)
        head_index = torch.arange(
            key_value_heads, device=owner.device
        ).view(1, -1, 1)
        position[batch_index, head_index, page, offset] = order

    sorted_key = leaves.key.gather(order, key_value_heads).to(key_sum.dtype)
    sorted_value = leaves.value.gather(order, key_value_heads).to(value_sum.dtype)
    key_sum.scatter_add_(
        2, page.unsqueeze(-1).expand_as(sorted_key), sorted_key
    )
    value_sum.scatter_add_(
        2, page.unsqueeze(-1).expand_as(sorted_value), sorted_value
    )
    page_count.scatter_add_(2, page, torch.ones_like(page))
    return RegionPages(
        position,
        key_sum,
        value_sum,
        page_count,
        slot_page_start,
        slot_page_count,
    )


def _empty_recursive_attention(
    query: torch.Tensor, value_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    output = query.new_zeros(*query.shape[:-1], value_dim)
    lse = torch.full(
        query.shape[:-1], -torch.inf, dtype=torch.float32, device=query.device
    )
    return output, lse


def _gather_page_vectors(
    tensor: torch.Tensor,
    page: torch.Tensor,
    kv_head: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,KVH,P,D]`` rows for arbitrary ``[B,QH,...]`` IDs."""
    batch, query_heads = page.shape[:2]
    dimension = int(tensor.size(-1))
    extra_dimensions = page.ndim - 2
    source = tensor[:, kv_head].view(
        batch,
        query_heads,
        *([1] * extra_dimensions),
        int(tensor.size(2)),
        dimension,
    ).expand(
        *page.shape, int(tensor.size(2)), dimension
    )
    index = page.unsqueeze(-1).unsqueeze(-1).expand(*page.shape, 1, dimension)
    return source.gather(-2, index).squeeze(-2)


def _gather_page_counts(
    count: torch.Tensor,
    page: torch.Tensor,
    kv_head: torch.Tensor,
) -> torch.Tensor:
    batch, query_heads = page.shape[:2]
    source = count[:, kv_head].view(
        batch,
        query_heads,
        *([1] * (page.ndim - 2)),
        int(count.size(2)),
    ).expand(*page.shape, int(count.size(2)))
    return source.gather(-1, page.unsqueeze(-1)).squeeze(-1)


def _select_region_pages(
    query: torch.Tensor,
    pages: RegionPages,
    top_slots: torch.Tensor,
    open_mask: torch.Tensor,
    kv_head: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Find the best page per route with bounded intermediate storage."""
    query_length = int(query.size(2))
    starts = pages.slot_start[:, kv_head].unsqueeze(2).expand(
        -1, -1, query_length, -1
    ).gather(-1, top_slots)
    counts = pages.slot_count[:, kv_head].unsqueeze(2).expand(
        -1, -1, query_length, -1
    ).gather(-1, top_slots)
    counts = torch.where(open_mask, counts, torch.zeros_like(counts))
    route_max_pages = counts.amax(dim=(0, 1, 2)).tolist()
    selected = []
    for route in range(int(top_slots.size(-1))):
        route_count = counts[..., route]
        route_start = starts[..., route]
        best_score = torch.full_like(route_count, -torch.inf, dtype=torch.float32)
        best_page = route_start
        max_pages = int(route_max_pages[route])
        for begin in range(0, max_pages, 16):
            width = min(16, max_pages - begin)
            offset = begin + torch.arange(width, device=query.device)
            valid = offset < route_count.unsqueeze(-1)
            page = (route_start.unsqueeze(-1) + offset).clamp_max(
                int(pages.count.size(2)) - 1
            )
            page_count = _gather_page_counts(
                pages.count, page, kv_head
            ).clamp_min(1)
            key_sum = _gather_page_vectors(pages.key_sum, page, kv_head)
            score = (
                query.float().unsqueeze(-2)
                * (key_sum.float() / page_count.unsqueeze(-1))
            ).sum(-1) * scale + page_count.float().log()
            score.masked_fill_(~valid, -torch.inf)
            block_score, block_offset = score.max(dim=-1)
            block_page = page.gather(-1, block_offset.unsqueeze(-1)).squeeze(-1)
            better = block_score > best_score
            best_score = torch.where(better, block_score, best_score)
            best_page = torch.where(better, block_page, best_page)
        selected.append(best_page)
    return torch.stack(selected, dim=-1)


def _recursive_page_attention(
    query: torch.Tensor,
    state: LODState,
    leaves: PagedKVCache,
    pages: RegionPages,
    top_slots: torch.Tensor,
    open_mask: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Open one page per routed region and summarize its disjoint residual."""
    batch, query_heads, query_length, _ = query.shape
    key_value_heads = int(pages.position.size(1))
    groups = query_heads // key_value_heads
    route_count = int(top_slots.size(-1))
    value_dim = leaves.value.dimension
    if route_count == 0:
        return _empty_recursive_attention(query, value_dim)

    kv_head = torch.div(
        torch.arange(query_heads, device=query.device),
        groups,
        rounding_mode="floor",
    )
    selected_page = _select_region_pages(
        query, pages, top_slots, open_mask, kv_head, scale=scale
    )
    selected_count = _gather_page_counts(pages.count, selected_page, kv_head)
    selected_key_sum = _gather_page_vectors(
        pages.key_sum, selected_page, kv_head
    )
    selected_value_sum = _gather_page_vectors(
        pages.value_sum, selected_page, kv_head
    )
    selected_position = _gather_page_vectors(
        pages.position, selected_page, kv_head
    )
    token_valid = open_mask.unsqueeze(-1) & selected_position.ge(0)
    safe_position = selected_position.clamp_min(0)
    selected_key = leaves.key.gather(safe_position, query_heads)
    selected_value = leaves.value.gather(safe_position, query_heads)
    token_score = (
        query.float().unsqueeze(-2).unsqueeze(-2) * selected_key.float()
    ).sum(-1) * scale
    token_score.masked_fill_(~token_valid, -torch.inf)

    state_key = state.key_sum[:, kv_head].unsqueeze(2).expand(
        -1, -1, query_length, -1, -1
    ).gather(
        3,
        top_slots.unsqueeze(-1).expand(
            -1, -1, -1, -1, int(state.key_sum.size(-1))
        ),
    )
    state_value = state.value_sum[:, kv_head].unsqueeze(2).expand(
        -1, -1, query_length, -1, -1
    ).gather(
        3,
        top_slots.unsqueeze(-1).expand(
            -1, -1, -1, -1, int(state.value_sum.size(-1))
        ),
    )
    state_count = state.count[:, kv_head].unsqueeze(2).expand(
        -1, -1, query_length, -1
    ).gather(-1, top_slots)
    residual_count = state_count - selected_count
    residual_valid = open_mask & residual_count.gt(0)
    safe_residual_count = residual_count.clamp_min(1)
    residual_key = (
        state_key.float() - selected_key_sum.float()
    ) / safe_residual_count.unsqueeze(-1)
    residual_value = (
        state_value.float() - selected_value_sum.float()
    ) / safe_residual_count.unsqueeze(-1)
    residual_score = (
        query.float().unsqueeze(-2) * residual_key
    ).sum(-1) * scale + safe_residual_count.float().log()
    residual_score.masked_fill_(~residual_valid, -torch.inf)

    scores = torch.cat((residual_score.unsqueeze(-1), token_score), dim=-1)
    scores = scores.flatten(-2)
    values = torch.cat(
        (residual_value.to(selected_value.dtype).unsqueeze(-2), selected_value),
        dim=-2,
    ).flatten(-3, -2)
    exact_lse = torch.logsumexp(scores, dim=-1)
    finite = torch.isfinite(exact_lse)
    safe_scores = torch.where(
        finite.unsqueeze(-1), scores, torch.zeros_like(scores)
    )
    probability = torch.softmax(safe_scores, dim=-1)
    probability = torch.where(
        finite.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    output = torch.matmul(
        probability.to(values.dtype).unsqueeze(-2), values
    ).squeeze(-2)
    return output, exact_lse


def _reference_coarse_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    top_slots: torch.Tensor,
    open_mask: torch.Tensor,
    *,
    scale: float,
    query_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense fallback for the coarse/local branch with routed slots removed."""
    query_heads = int(query.size(1))
    query_length = int(query.size(2))
    state_key = _repeat_kv(state.mean_key, query_heads)
    state_value = _repeat_kv(state.mean_value, query_heads)
    local_key = _repeat_kv(local_key, query_heads)
    local_value = _repeat_kv(local_value, query_heads)
    key = torch.cat((state_key, local_key), dim=2)
    value = torch.cat((state_value, local_value), dim=2)
    scores = torch.matmul(
        query.float(), key.float().transpose(-1, -2)
    ) * float(scale)
    state_bias = _repeat_kv(state.count.clamp_min(1).log(), query_heads)
    state_bias = state_bias.unsqueeze(2).expand(-1, -1, query_length, -1).clone()
    for route in range(int(top_slots.size(-1))):
        state_bias.scatter_(
            -1,
            top_slots[..., route : route + 1],
            torch.where(
                open_mask[..., route : route + 1],
                torch.full_like(state_bias[..., :1], -torch.inf),
                state_bias.gather(-1, top_slots[..., route : route + 1]),
            ),
        )
    query_index = torch.arange(query_length, device=query.device).unsqueeze(-1)
    local_index = torch.arange(int(local_key.size(2)), device=query.device)
    local_bias = torch.zeros(
        query_length, int(local_key.size(2)), device=query.device
    ).masked_fill(local_index > query_index + query_offset, -torch.inf)
    bias = torch.cat(
        (
            state_bias,
            local_bias.view(1, 1, query_length, -1).expand(
                int(query.size(0)), query_heads, -1, -1
            ),
        ),
        dim=-1,
    )
    return _attention_from_scores(scores + bias, value)


def paged_two_level_lod_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    owner: torch.Tensor,
    leaves: PagedKVCache,
    *,
    max_routes: int = 8,
    open_count: int | torch.Tensor = 8,
    scale: float | None = None,
    query_offset: int | None = None,
    region_pages: RegionPages | None = None,
) -> LODAttentionResult:
    """Apply recursive region-to-page LOD to a paged leaf archive."""
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))
    if query_offset is None:
        query_offset = int(local_key.size(2)) - int(query.size(2))
    if int(owner.size(2)) > leaves.length:
        raise ValueError("owner archive is longer than paged K/V")
    if leaves.key.dimension != int(query.size(-1)):
        raise ValueError("paged key dimension does not match the query")
    if leaves.value.dimension != int(local_value.size(-1)):
        raise ValueError("paged and local value dimensions differ")

    fast_supported = (
        query.device.type != "cpu"
        and query.dtype in (torch.float16, torch.bfloat16)
        and leaves.key.dtype == query.dtype
        and leaves.value.dtype == query.dtype
        and int(query.size(-1)) == int(local_value.size(-1))
        and int(query.size(-1)) == leaves.value.dimension
        and not _attention_needs_grad(query, local_key, local_value)
    )
    top_slots, open_mask = _route_state(
        query,
        state,
        max_routes=max_routes,
        open_count=open_count,
        scale=scale,
    )
    if fast_supported:
        coarse_output, coarse_lse = _fast_coarse_attention(
            query,
            local_key,
            local_value,
            state,
            top_slots=top_slots,
            open_mask=open_mask,
            scale=scale,
            query_offset=query_offset,
        )
    else:
        coarse_output, coarse_lse = _reference_coarse_attention(
            query,
            local_key,
            local_value,
            state,
            top_slots,
            open_mask,
            scale=scale,
            query_offset=query_offset,
        )
    if region_pages is None:
        # Low-level callers can pass persistent summaries; the module caches
        # them until either ownership or physical page quantization changes.
        region_pages = build_region_pages(
            owner, state, leaves, leaves.key.page_size
        )
    exact_output, exact_lse = _recursive_page_attention(
        query,
        state,
        leaves,
        region_pages,
        top_slots,
        open_mask,
        scale=scale,
    )
    output = _merge_two_branches(
        coarse_output, coarse_lse, exact_output, exact_lse
    )
    return LODAttentionResult(
        output=output,
        logsumexp=torch.logaddexp(coarse_lse.float(), exact_lse.float()),
        top_slots=top_slots,
        open_mask=open_mask,
    )


class PagedTwoLevelLODAttention(_FastLocalMixin, TwoLevelLODAttention):
    """Flat LOD or recursive paged LOD with optional INT4 physical K/V."""

    config: PagedLODConfig

    def __init__(
        self,
        config: PagedLODConfig | None = None,
        *,
        default_open_count: int = 8,
    ) -> None:
        config = PagedLODConfig() if config is None else config
        super().__init__(config, default_open_count=default_open_count)
        self._posting_key: tuple[int, tuple[int, ...], int] | None = None
        self._postings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._region_key: tuple[int, tuple[int, ...], int, int] | None = None
        self._region_pages: RegionPages | None = None

    @property
    def uses_pages(self) -> bool:
        return self.config.page_size is not None

    def _cached_postings(
        self, owner: torch.Tensor, state: LODState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (owner.data_ptr(), tuple(owner.shape), state.slot_count)
        if key != self._posting_key or self._postings is None:
            self._postings = _posting_lists(owner, state)
            self._posting_key = key
        return self._postings

    def _cached_region_pages(
        self, owner: torch.Tensor, state: LODState, leaves: PagedKVCache
    ) -> RegionPages:
        key = (
            owner.data_ptr(),
            tuple(owner.shape),
            state.slot_count,
            leaves.key.complete_pages,
        )
        if key != self._region_key or self._region_pages is None:
            if self.config.page_size is None:
                raise AssertionError("recursive page cache has no page size")
            self._region_pages = build_region_pages(
                owner, state, leaves, self.config.page_size
            )
            self._region_key = key
        return self._region_pages

    def _attend(self, query, local_key, local_value, state, **kwargs):
        owner = kwargs["owner"]
        leaf_key = kwargs["leaf_key"]
        leaf_value = kwargs["leaf_value"]
        if owner is None or leaf_key is None or leaf_value is None:
            raise ValueError("flat two-level attention requires exact leaves")
        return fast_two_level_lod_attention(
            query,
            local_key,
            local_value,
            state,
            owner,
            leaf_key,
            leaf_value,
            max_routes=self.config.max_routes,
            open_count=kwargs["open_count"],
            scale=kwargs["scale"],
            query_offset=int(local_key.size(2)) - int(query.size(2)),
            postings=self._cached_postings(owner, state),
        ).output

    def _attend_paged(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        state: LODState,
        owner: torch.Tensor,
        leaves: PagedKVCache,
        *,
        open_count: int | torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        return paged_two_level_lod_attention(
            query,
            local_key,
            local_value,
            state,
            owner,
            leaves,
            max_routes=self.config.max_routes,
            open_count=open_count,
            scale=scale,
            query_offset=int(local_key.size(2)) - int(query.size(2)),
            region_pages=self._cached_region_pages(owner, state, leaves),
        ).output

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        open_count: int | torch.Tensor,
        scale: float,
        use_cache: bool,
    ):
        if not self.uses_pages:
            return super()._prefill(
                query,
                key,
                value,
                open_count=open_count,
                scale=scale,
                use_cache=use_cache,
            )

        self._region_key = None
        self._region_pages = None
        sequence_length = int(query.size(2))
        front_length = min(sequence_length, self.config.local_window)
        outputs = [
            self._local(
                query[..., :front_length, :],
                key[..., :front_length, :],
                value[..., :front_length, :],
                scale=scale,
                query_offset=0,
            )
        ]
        state = self._empty_state(key, value)
        coverage = 0
        owner = torch.empty(
            *key.shape[:2], 0, dtype=torch.long, device=key.device
        )
        leaves = PagedKVCache.from_tensors(key, value, self.config)
        exact_lookback = self.config.local_window - self.config.chunk_size

        for query_begin in range(
            front_length, sequence_length, self.config.chunk_size
        ):
            query_end = min(sequence_length, query_begin + self.config.chunk_size)
            local_begin = max(0, query_begin - exact_lookback)
            state, next_owner, coverage = self._compress_to(
                state,
                owner,
                key,
                value,
                coverage=coverage,
                target=local_begin,
                context_length=query_begin,
            )
            if next_owner is None:
                raise AssertionError("paged state update dropped leaf ownership")
            owner = next_owner
            outputs.append(
                self._attend_paged(
                    query[..., query_begin:query_end, :],
                    key[..., local_begin:query_end, :],
                    value[..., local_begin:query_end, :],
                    state,
                    owner,
                    leaves,
                    open_count=self._slice_open_count(
                        open_count, query_begin, query_end
                    ),
                    scale=scale,
                )
            )

        if use_cache:
            decode_coverage = self._bswa_begin(sequence_length + 1)
            state, next_owner, coverage = self._compress_to(
                state,
                owner,
                key,
                value,
                coverage=coverage,
                target=decode_coverage,
                context_length=sequence_length,
            )
            if next_owner is None:
                raise AssertionError("paged cache has no ownership archive")
            cache = PagedLODCache(
                state=state,
                coverage=coverage,
                recent_key=key[..., coverage:, :],
                recent_value=value[..., coverage:, :],
                total_length=sequence_length,
                owner=next_owner,
                leaves=leaves,
            ).detached()
        else:
            cache = None
        return torch.cat(outputs, dim=2), cache

    def _decode_one(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache,
        *,
        open_count: int | torch.Tensor,
        scale: float,
    ):
        if not self.uses_pages:
            return super()._decode_one(
                query,
                key,
                value,
                cache,
                open_count=open_count,
                scale=scale,
            )
        if not isinstance(cache, PagedLODCache):
            raise TypeError("paged decode requires a PagedLODCache")
        if int(query.size(2)) != 1 or int(key.size(2)) != 1:
            raise ValueError("paged decode consumes one token at a time")

        recent_key = torch.cat((cache.recent_key, key), dim=2)
        recent_value = torch.cat((cache.recent_value, value), dim=2)
        leaves = cache.leaves.append(key, value)
        total_length = cache.total_length + 1
        target_coverage = self._bswa_begin(total_length)
        overflow_length = target_coverage - cache.coverage
        if overflow_length < 0 or overflow_length > int(recent_key.size(2)):
            raise AssertionError("paged LOD decode coverage drifted")
        state, owner, relative_coverage = self._compress_to(
            cache.state,
            cache.owner,
            recent_key,
            recent_value,
            coverage=0,
            target=overflow_length,
            context_length=cache.total_length,
            coverage_offset=cache.coverage,
        )
        if owner is None:
            raise AssertionError("paged decode dropped leaf ownership")
        coverage = cache.coverage + relative_coverage
        recent_key = recent_key[..., overflow_length:, :]
        recent_value = recent_value[..., overflow_length:, :]
        if state.slot_count == 0:
            output = self._local(
                query,
                recent_key,
                recent_value,
                scale=scale,
                query_offset=int(recent_key.size(2)) - 1,
            )
        else:
            output = self._attend_paged(
                query,
                recent_key,
                recent_value,
                state,
                owner,
                leaves,
                open_count=open_count,
                scale=scale,
            )
        return output, PagedLODCache(
            state=state,
            coverage=coverage,
            recent_key=recent_key,
            recent_value=recent_value,
            total_length=total_length,
            owner=owner,
            leaves=leaves,
        )


__all__ = [
    "PagedKVCache",
    "PagedLODCache",
    "PagedLODConfig",
    "PagedTensor",
    "PagedTwoLevelLODAttention",
    "RegionPages",
    "build_region_pages",
    "paged_two_level_lod_attention",
]
