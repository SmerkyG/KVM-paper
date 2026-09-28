"""Readable, model-independent PyTorch implementation of LLM LoD Attention.

This is a reference implementation of the algorithm described in the paper,
not a performance implementation.  It deliberately expresses routing,
frontier construction, and log-sum-exp merging with ordinary PyTorch tensor
operations.  In particular, the exact-leaf branch materializes scores for the
remote archive and masks them by region ownership.  The optimized kernels in
this package avoid that extra work, but implement the same attention frontier.

Inputs and outputs are post-projection, post-position-encoding head tensors:

    query: [batch, query_heads, query_length, key_dimension]
    key:   [batch, key_value_heads, query_length, key_dimension]
    value: [batch, key_value_heads, query_length, value_dimension]

The engine supports GQA, cached decoding, left-padded Hugging Face batches,
the paper's two-tier BF16 mode, and its recursive three-tier BF16 mode.  INT4
page storage is intentionally left to the optimized engine: it is a cache
encoding optimization rather than part of the LoD frontier definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ._config import (
    CHUNK_SIZE,
    EXACT_DECODE_LIMIT,
    LOCAL_WINDOW,
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    ROUTE_COUNT,
    LODMode,
)


@dataclass(frozen=True)
class PytorchLODConfig:
    """Paper defaults, exposed as fields so small CPU examples remain practical."""

    chunk_size: int = CHUNK_SIZE
    local_window: int = LOCAL_WINDOW
    prefill_block_size: int = PREFILL_CHUNK_SIZE
    prefill_lookback: int = CHUNK_SIZE
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    route_count: int = ROUTE_COUNT
    max_region_size: int = 1_024
    page_size: int = PAGE_SIZE
    exact_decode_limit: int = EXACT_DECODE_LIMIT

    def __post_init__(self) -> None:
        positive = (
            self.chunk_size,
            self.local_window,
            self.prefill_block_size,
            self.state_min_size,
            self.route_count,
            self.max_region_size,
            self.page_size,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("PyTorch LoD sizes and route count must be positive")
        if self.local_window < self.chunk_size:
            raise ValueError("the decode-local window cannot be shorter than one chunk")
        if self.prefill_lookback < 0:
            raise ValueError("the prefill lookback cannot be negative")
        if self.state_growth_factor <= 0:
            raise ValueError("the state growth factor must be positive")
        if self.exact_decode_limit < 0:
            raise ValueError("the exact decode limit cannot be negative")


@dataclass
class PytorchLODState:
    """Region key/value sums, populations, and constituent key norms."""

    key_sum: torch.Tensor
    value_sum: torch.Tensor
    count: torch.Tensor
    key_rms_sum: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.key_sum.size(2))

    @property
    def mean_key(self) -> torch.Tensor:
        return self.key_sum / self.count.clamp_min(1).to(self.key_sum.dtype).unsqueeze(
            -1
        )

    @property
    def mean_value(self) -> torch.Tensor:
        return self.value_sum / self.count.clamp_min(1).to(
            self.value_sum.dtype
        ).unsqueeze(-1)

    def detached(self) -> PytorchLODState:
        return PytorchLODState(
            self.key_sum.detach(),
            self.value_sum.detach(),
            self.count.detach(),
            self.key_rms_sum.detach(),
        )


@dataclass
class PytorchLODCache:
    """Authoritative reference cache used by Hugging Face generation."""

    state: PytorchLODState
    owner: torch.Tensor
    archive_key: torch.Tensor
    archive_value: torch.Tensor
    archive_valid: torch.Tensor
    coverage: int
    sink_key: torch.Tensor
    sink_value: torch.Tensor
    total_length: int

    def detached(self) -> PytorchLODCache:
        return PytorchLODCache(
            state=self.state.detached(),
            owner=self.owner.detach(),
            archive_key=self.archive_key.detach(),
            archive_value=self.archive_value.detach(),
            archive_valid=self.archive_valid.detach(),
            coverage=self.coverage,
            sink_key=self.sink_key.detach(),
            sink_value=self.sink_value.detach(),
            total_length=self.total_length,
        )


@dataclass
class PytorchLODResult:
    """Low-level result, including routes for tests and educational use."""

    output: torch.Tensor
    logsumexp: torch.Tensor
    routes: torch.Tensor


def _validate_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be rank-four tensors")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("query, key, and value sequence geometry differs")
    if query.size(2) != key.size(2) or query.size(-1) != key.size(-1):
        raise ValueError("query and key sequence/head dimensions differ")
    query_heads = int(query.size(1))
    key_value_heads = int(key.size(1))
    if key_value_heads == 0 or query_heads % key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    return query_heads, key_value_heads


def _repeat_kv(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    groups = query_heads // int(tensor.size(1))
    return tensor if groups == 1 else tensor.repeat_interleave(groups, dim=1)


def _scores(query: torch.Tensor, key: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale


def _rms_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() * torch.rsqrt(
        tensor.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    )


def _attention(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one possibly empty/masked attention branch."""

    if int(scores.size(-1)) == 0:
        output = value.new_zeros(*scores.shape[:-1], int(value.size(-1)))
        lse = scores.new_full(scores.shape[:-1], -torch.inf)
        return output, lse
    valid = torch.isfinite(scores).any(dim=-1)
    safe = torch.where(valid.unsqueeze(-1), scores, torch.zeros_like(scores))
    probability = torch.softmax(safe, dim=-1)
    probability = torch.where(
        valid.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    output = torch.matmul(probability.to(value.dtype), value)
    return output, torch.logsumexp(scores, dim=-1)


def _merge_branches(
    outputs: list[torch.Tensor],
    logsumexp: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized branches into one shared softmax."""

    branch_lse = torch.stack(logsumexp, dim=-1).float()
    total_lse = torch.logsumexp(branch_lse, dim=-1)
    valid = torch.isfinite(total_lse)
    safe_total = torch.where(valid, total_lse, torch.zeros_like(total_lse))
    weight = torch.exp(branch_lse - safe_total.unsqueeze(-1))
    weight = torch.where(valid.unsqueeze(-1), weight, torch.zeros_like(weight))
    stacked = torch.stack(outputs, dim=-2)
    output = (stacked * weight.to(stacked.dtype).unsqueeze(-1)).sum(dim=-2)
    return output, total_lse


def _local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    query_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads = int(query.size(1))
    repeated_key = _repeat_kv(key, query_heads)
    repeated_value = _repeat_kv(value, query_heads)
    query_positions = torch.arange(int(query.size(2)), device=query.device)
    key_positions = torch.arange(int(key.size(2)), device=query.device)
    visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1) + query_offset
    score = _scores(query, repeated_key, scale).masked_fill(
        ~visible.view(1, 1, int(query.size(2)), int(key.size(2))), -torch.inf
    )
    return _attention(score, repeated_value)


def _exact_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    valid_starts: torch.Tensor,
) -> torch.Tensor:
    query_heads = int(query.size(1))
    repeated_key = _repeat_kv(key, query_heads)
    repeated_value = _repeat_kv(value, query_heads)
    length = int(key.size(2))
    position = torch.arange(length, device=query.device)
    visible = position.view(1, 1, 1, length) >= valid_starts.view(-1, 1, 1, 1)
    visible = visible & (
        position.view(1, 1, 1, length) <= position.view(1, 1, length, 1)
    )
    score = _scores(query, repeated_key, scale).masked_fill(~visible, -torch.inf)
    output, _ = _attention(score, repeated_value)
    valid_query = position.view(1, 1, length, 1) >= valid_starts.view(-1, 1, 1, 1)
    return torch.where(valid_query, output, torch.zeros_like(output))


class PytorchLODAttention(nn.Module):
    """Clear two-tier/recursive BF16 LoD engine for post-RoPE Q/K/V tensors."""

    def __init__(
        self,
        config: PytorchLODConfig | None = None,
        *,
        mode: str | LODMode = LODMode.TWO_TIER,
        normalized_keys: bool = False,
        normalize_routing_query: bool = True,
    ) -> None:
        super().__init__()
        self.config = PytorchLODConfig() if config is None else config
        self.mode = LODMode.parse(mode)
        if self.mode is LODMode.THREE_TIER_INT4:
            raise ValueError(
                "the pure PyTorch reference supports BF16 frontiers, not INT4 storage"
            )
        self.normalized_keys = bool(normalized_keys)
        self.normalize_routing_query = bool(normalize_routing_query)

    @staticmethod
    def _empty_state(key: torch.Tensor, value: torch.Tensor) -> PytorchLODState:
        shape = (*key.shape[:2], 0)
        return PytorchLODState(
            key[..., :0, :],
            value[..., :0, :],
            torch.empty(shape, dtype=torch.float32, device=key.device),
            torch.empty(shape, dtype=torch.float32, device=key.device),
        )

    def _state_target(self, context_length: int, available: int) -> int:
        scheduled = max(
            self.config.state_min_size,
            math.floor(self.config.state_growth_factor * math.sqrt(context_length)),
        )
        return min(scheduled, available)

    def _assignment_scores(
        self,
        key: torch.Tensor,
        state: PytorchLODState,
    ) -> torch.Tensor:
        """Architecture-aware semantic assignment from the paper."""

        leaf_direction = _rms_normalize(key)
        centroid_direction = _rms_normalize(state.mean_key)
        similarity = torch.matmul(
            leaf_direction, centroid_direction.transpose(-1, -2)
        ) / float(key.size(-1))
        if self.normalized_keys:
            mean_leaf_rms = state.key_rms_sum / state.count.clamp_min(1)
            coherence = state.mean_key.float().square().mean(
                -1
            ).sqrt() / mean_leaf_rms.clamp_min(1e-12)
            similarity = similarity * coherence.unsqueeze(-2)
        return similarity.masked_fill(state.count.le(0).unsqueeze(-2), -torch.inf)

    def _append_state(
        self,
        state: PytorchLODState,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
        *,
        context_length: int,
        available: int,
    ) -> tuple[PytorchLODState, torch.Tensor]:
        """Append novel entries up to 16*sqrt(T), then merge the remainder."""

        length = int(key.size(2))
        owner = torch.full(key.shape[:3], -1, dtype=torch.long, device=key.device)
        if length == 0:
            return state, owner

        if state.size == 0:
            seed_length = min(self.config.chunk_size, length)
            seed_valid = valid[..., :seed_length]
            seed_rms = key[..., :seed_length, :].float().square().mean(-1).sqrt()
            state = PytorchLODState(
                key[..., :seed_length, :] * seed_valid.unsqueeze(-1).to(key.dtype),
                value[..., :seed_length, :] * seed_valid.unsqueeze(-1).to(value.dtype),
                seed_valid.float(),
                seed_rms * seed_valid.float(),
            )
            seed_owner = torch.arange(seed_length, device=key.device).view(
                1, 1, seed_length
            )
            owner[..., :seed_length] = torch.where(
                seed_valid,
                seed_owner,
                torch.full_like(seed_owner, -1),
            )
            if seed_length == length:
                return state, owner
            tail_state, tail_owner = self._append_state(
                state,
                key[..., seed_length:, :],
                value[..., seed_length:, :],
                valid[..., seed_length:],
                context_length=context_length,
                available=available,
            )
            owner[..., seed_length:] = tail_owner
            return tail_state, owner

        current_size = state.size
        target_size = self._state_target(context_length, available)
        append_count = min(max(target_size - current_size, 0), length)
        append_mask = torch.zeros_like(valid)
        if append_count:
            with torch.no_grad():
                novelty = self._assignment_scores(key, state).amax(dim=-1)
                novelty = novelty.masked_fill(~valid, torch.inf)
                append_index = novelty.topk(
                    append_count, dim=-1, largest=False, sorted=False
                ).indices
            gathered_valid = valid.gather(2, append_index)
            append_mask.scatter_(2, append_index, gathered_valid)
            gather_index = append_index.unsqueeze(-1).expand(
                *append_index.shape, int(key.size(-1))
            )
            value_index = append_index.unsqueeze(-1).expand(
                *append_index.shape, int(value.size(-1))
            )
            appended_key = key.gather(2, gather_index)
            appended_value = value.gather(2, value_index)
            appended_rms = appended_key.float().square().mean(-1).sqrt()
            state = PytorchLODState(
                torch.cat(
                    (
                        state.key_sum,
                        appended_key
                        * gathered_valid.unsqueeze(-1).to(appended_key.dtype),
                    ),
                    dim=2,
                ),
                torch.cat(
                    (
                        state.value_sum,
                        appended_value
                        * gathered_valid.unsqueeze(-1).to(appended_value.dtype),
                    ),
                    dim=2,
                ),
                torch.cat((state.count, gathered_valid.float()), dim=2),
                torch.cat(
                    (state.key_rms_sum, appended_rms * gathered_valid.float()),
                    dim=2,
                ),
            )
            append_slots = torch.arange(
                current_size,
                current_size + append_count,
                device=key.device,
            ).view(1, 1, append_count)
            owner.scatter_(
                2,
                append_index,
                torch.where(
                    gathered_valid,
                    append_slots,
                    torch.full_like(append_slots, -1),
                ),
            )

        merge_valid = valid & ~append_mask
        if not bool(merge_valid.any().item()):
            return state, owner
        with torch.no_grad():
            destination = self._assignment_scores(key, state).argmax(dim=-1)
        owner = torch.where(merge_valid, destination, owner)

        destination_key = destination.unsqueeze(-1).expand(
            *destination.shape, int(key.size(-1))
        )
        destination_value = destination.unsqueeze(-1).expand(
            *destination.shape, int(value.size(-1))
        )
        key_add = torch.zeros_like(state.key_sum).scatter_add(
            2,
            destination_key,
            key * merge_valid.unsqueeze(-1).to(key.dtype),
        )
        value_add = torch.zeros_like(state.value_sum).scatter_add(
            2,
            destination_value,
            value * merge_valid.unsqueeze(-1).to(value.dtype),
        )
        count_add = torch.zeros_like(state.count).scatter_add(
            2, destination, merge_valid.float()
        )
        leaf_rms = key.float().square().mean(-1).sqrt()
        rms_add = torch.zeros_like(state.key_rms_sum).scatter_add(
            2, destination, leaf_rms * merge_valid.float()
        )
        return PytorchLODState(
            state.key_sum + key_add,
            state.value_sum + value_add,
            state.count + count_add,
            state.key_rms_sum + rms_add,
        ), owner

    def _compress_to(
        self,
        cache: PytorchLODCache,
        target: int,
        *,
        context_length: int,
    ) -> PytorchLODCache:
        if not cache.coverage <= target <= int(cache.archive_key.size(2)):
            raise ValueError("invalid PyTorch LoD state catch-up target")
        if target == cache.coverage:
            return cache
        begin = cache.coverage
        state, new_owner = self._append_state(
            cache.state,
            cache.archive_key[..., begin:target, :],
            cache.archive_value[..., begin:target, :],
            cache.archive_valid[:, None, begin:target].expand(
                -1, int(cache.archive_key.size(1)), -1
            ),
            context_length=context_length,
            available=target,
        )
        cache.state = state
        cache.owner = torch.cat((cache.owner, new_owner), dim=2)
        cache.coverage = target
        return cache

    def _routes(
        self,
        query: torch.Tensor,
        state: PytorchLODState,
        *,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_heads = int(query.size(1))
        mean_key = _repeat_kv(state.mean_key, query_heads)
        mean_value = _repeat_kv(state.mean_value, query_heads)
        count = _repeat_kv(state.count, query_heads)
        coarse_score = (
            _scores(query, mean_key, scale) + count.clamp_min(1).log()[:, :, None, :]
        )
        coarse_score = coarse_score.masked_fill(count[:, :, None, :].le(0), -torch.inf)

        route_query = _rms_normalize(query) if self.normalize_routing_query else query
        route_score = (
            _scores(route_query, mean_key, scale)
            + count.clamp_min(1).log()[:, :, None, :]
        )
        route_score = route_score.masked_fill(
            ~count.gt(0)[:, :, None, :], -torch.inf
        )
        route_count = min(self.config.route_count, state.size)
        routes = route_score.topk(route_count, dim=-1).indices
        selected_count = torch.gather(
            count[:, :, None, :].expand(-1, -1, int(query.size(2)), -1),
            -1,
            routes,
        )
        active = route_score.gather(-1, routes).isfinite() & selected_count.le(
            self.config.max_region_size
        )
        return (
            coarse_score,
            mean_value,
            torch.where(active, routes, torch.full_like(routes, -1)),
        )

    def _full_region_branches(
        self,
        query: torch.Tensor,
        cache: PytorchLODCache,
        routes: torch.Tensor,
        *,
        scale: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        query_heads = int(query.size(1))
        leaf_key = _repeat_kv(cache.archive_key[..., : cache.coverage, :], query_heads)
        leaf_value = _repeat_kv(
            cache.archive_value[..., : cache.coverage, :], query_heads
        )
        owner = _repeat_kv(cache.owner, query_heads)
        leaf_score = _scores(query, leaf_key, scale)
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        for route_index in range(int(routes.size(-1))):
            route = routes[..., route_index]
            selected = owner[:, :, None, :] == route.unsqueeze(-1)
            selected = selected & route.ge(0).unsqueeze(-1)
            output, lse = _attention(
                leaf_score.masked_fill(~selected, -torch.inf), leaf_value
            )
            outputs.append(output)
            lses.append(lse)
        return outputs, lses

    def _recursive_region_branches(
        self,
        query: torch.Tensor,
        cache: PytorchLODCache,
        routes: torch.Tensor,
        *,
        scale: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Open one best chronological region-local page and its residual."""

        batch, query_heads, query_length, _ = query.shape
        value_dimension = int(cache.archive_value.size(-1))
        groups = query_heads // int(cache.archive_key.size(1))
        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        for route_index in range(int(routes.size(-1))):
            route_output = cache.archive_value.new_zeros(
                batch, query_heads, query_length, value_dimension
            )
            route_lse = torch.full(
                (batch, query_heads, query_length),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
            for batch_index in range(batch):
                for query_head in range(query_heads):
                    kv_head = query_head // groups
                    region_owner = cache.owner[batch_index, kv_head]
                    for query_index in range(query_length):
                        slot = int(
                            routes[
                                batch_index, query_head, query_index, route_index
                            ].item()
                        )
                        if slot < 0:
                            continue
                        positions = torch.nonzero(
                            region_owner.eq(slot), as_tuple=False
                        ).flatten()
                        if not int(positions.numel()):
                            continue
                        region_key = cache.archive_key[
                            batch_index, kv_head
                        ].index_select(0, positions)
                        region_value = cache.archive_value[
                            batch_index, kv_head
                        ].index_select(0, positions)
                        page_scores = []
                        pages = []
                        for page_begin in range(
                            0, int(positions.numel()), self.config.page_size
                        ):
                            page_key = region_key[
                                page_begin : page_begin + self.config.page_size
                            ]
                            page_value = region_value[
                                page_begin : page_begin + self.config.page_size
                            ]
                            pages.append((page_key, page_value))
                            page_scores.append(
                                _scores(
                                    query[
                                        batch_index : batch_index + 1,
                                        query_head : query_head + 1,
                                        query_index : query_index + 1,
                                    ],
                                    page_key[None, None],
                                    scale,
                                ).mean(dim=-1)
                                + math.log(int(page_key.size(0)))
                            )
                        best_page = int(
                            torch.stack(page_scores).reshape(-1).argmax().item()
                        )
                        exact_key, exact_value = pages[best_page]
                        exact_score = _scores(
                            query[
                                batch_index : batch_index + 1,
                                query_head : query_head + 1,
                                query_index : query_index + 1,
                            ],
                            exact_key[None, None],
                            scale,
                        )
                        frontier_scores = [exact_score]
                        frontier_values = [exact_value]
                        residual_count = int(region_key.size(0)) - int(
                            exact_key.size(0)
                        )
                        if residual_count:
                            residual_key = (
                                region_key.sum(dim=0) - exact_key.sum(dim=0)
                            ) / residual_count
                            residual_value = (
                                region_value.sum(dim=0) - exact_value.sum(dim=0)
                            ) / residual_count
                            residual_score = _scores(
                                query[
                                    batch_index : batch_index + 1,
                                    query_head : query_head + 1,
                                    query_index : query_index + 1,
                                ],
                                residual_key[None, None, None],
                                scale,
                            ) + math.log(residual_count)
                            frontier_scores.append(residual_score)
                            frontier_values.append(residual_value[None])
                        score = torch.cat(frontier_scores, dim=-1)
                        value = torch.cat(frontier_values, dim=0)[None, None]
                        output, lse = _attention(score, value)
                        route_output[batch_index, query_head, query_index] = output[
                            0, 0, 0
                        ]
                        route_lse[batch_index, query_head, query_index] = lse[0, 0, 0]
            outputs.append(route_output)
            lses.append(route_lse)
        return outputs, lses

    def _lod_attention(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        cache: PytorchLODCache,
        *,
        scale: float,
        query_offset: int,
        full_regions: bool,
    ) -> PytorchLODResult:
        coarse_score, coarse_value, routes = self._routes(
            query, cache.state, scale=scale
        )
        opened = torch.zeros_like(coarse_score, dtype=torch.bool)
        for route_index in range(int(routes.size(-1))):
            route = routes[..., route_index : route_index + 1]
            route_opened = torch.zeros_like(opened)
            route_opened.scatter_(-1, route.clamp_min(0), route.ge(0))
            opened |= route_opened
        closed_score = coarse_score.masked_fill(opened, -torch.inf)
        coarse_output, coarse_lse = _attention(closed_score, coarse_value)
        local_output, local_lse = _local_attention(
            query,
            local_key,
            local_value,
            scale=scale,
            query_offset=query_offset,
        )
        sink_key = _repeat_kv(cache.sink_key, int(query.size(1)))
        sink_value = _repeat_kv(cache.sink_value, int(query.size(1)))
        sink_output, sink_lse = _attention(_scores(query, sink_key, scale), sink_value)
        outputs = [coarse_output, local_output, sink_output]
        lses = [coarse_lse, local_lse, sink_lse]
        if full_regions or self.mode is LODMode.TWO_TIER:
            exact_outputs, exact_lses = self._full_region_branches(
                query, cache, routes, scale=scale
            )
        else:
            exact_outputs, exact_lses = self._recursive_region_branches(
                query, cache, routes, scale=scale
            )
        output, total_lse = _merge_branches(outputs + exact_outputs, lses + exact_lses)
        return PytorchLODResult(output, total_lse, routes)

    @staticmethod
    def _archive_without_sink(
        key: torch.Tensor,
        value: torch.Tensor,
        valid_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        length = int(key.size(2))
        sink_position = valid_starts[:, None, None, None].expand(
            -1, int(key.size(1)), 1, int(key.size(-1))
        )
        sink_value_position = valid_starts[:, None, None, None].expand(
            -1, int(value.size(1)), 1, int(value.size(-1))
        )
        sink_key = key.gather(2, sink_position)
        sink_value = value.gather(2, sink_value_position)
        archive_position = torch.arange(length - 1, device=key.device)
        source = (
            archive_position.unsqueeze(0)
            + (archive_position.unsqueeze(0) >= valid_starts.unsqueeze(1)).long()
        )
        key_index = source[:, None, :, None].expand(
            -1, int(key.size(1)), -1, int(key.size(-1))
        )
        value_index = source[:, None, :, None].expand(
            -1, int(value.size(1)), -1, int(value.size(-1))
        )
        archive_key = key.gather(2, key_index)
        archive_value = value.gather(2, value_index)
        archive_valid = archive_position.unsqueeze(0) >= valid_starts.unsqueeze(1)
        return sink_key, sink_value, archive_key, archive_value, archive_valid

    def _new_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_starts: torch.Tensor,
        total_length: int,
    ) -> PytorchLODCache:
        (
            sink_key,
            sink_value,
            archive_key,
            archive_value,
            archive_valid,
        ) = self._archive_without_sink(key, value, valid_starts)
        owner = torch.empty(*key.shape[:2], 0, dtype=torch.long, device=key.device)
        return PytorchLODCache(
            self._empty_state(key, value),
            owner,
            archive_key,
            archive_value,
            archive_valid,
            0,
            sink_key,
            sink_value,
            total_length,
        )

    @staticmethod
    def _archive_target(physical_position: int) -> int:
        """Translate a physical prefix boundary past the separately stored sink."""

        return max(0, physical_position - 1)

    def _decode_target(self, total_length: int) -> int:
        rounded = (
            (total_length + self.config.chunk_size - 1) // self.config.chunk_size
        ) * self.config.chunk_size
        return self._archive_target(max(0, rounded - self.config.local_window))

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
        use_cache: bool,
        logical_length: int,
        valid_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, PytorchLODCache | None]:
        query = query[..., :logical_length, :]
        key = key[..., :logical_length, :]
        value = value[..., :logical_length, :]
        front_length = min(logical_length, self.config.prefill_block_size)
        outputs = [
            _exact_causal_attention(
                query[..., :front_length, :],
                key[..., :front_length, :],
                value[..., :front_length, :],
                scale=scale,
                valid_starts=valid_starts,
            )
        ]
        cache = self._new_cache(key, value, valid_starts, logical_length)

        for begin in range(
            self.config.prefill_block_size,
            logical_length,
            self.config.prefill_block_size,
        ):
            end = min(logical_length, begin + self.config.prefill_block_size)
            local_begin = max(0, begin - self.config.prefill_lookback)
            cache = self._compress_to(
                cache,
                self._archive_target(local_begin),
                context_length=begin,
            )
            result = self._lod_attention(
                query[..., begin:end, :],
                key[..., local_begin:end, :],
                value[..., local_begin:end, :],
                cache,
                scale=scale,
                query_offset=begin - local_begin,
                full_regions=True,
            )
            outputs.append(result.output)

        if use_cache:
            cache = self._compress_to(
                cache,
                self._decode_target(logical_length),
                context_length=logical_length,
            ).detached()
        else:
            cache = None
        return torch.cat(outputs, dim=2), cache

    def _decode_one(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: PytorchLODCache,
        *,
        scale: float,
    ) -> tuple[torch.Tensor, PytorchLODCache]:
        cache.archive_key = torch.cat((cache.archive_key, key), dim=2)
        cache.archive_value = torch.cat((cache.archive_value, value), dim=2)
        cache.archive_valid = torch.cat(
            (
                cache.archive_valid,
                torch.ones(int(key.size(0)), 1, dtype=torch.bool, device=key.device),
            ),
            dim=1,
        )
        cache.total_length += 1
        cache = self._compress_to(
            cache,
            self._decode_target(cache.total_length),
            context_length=cache.total_length,
        )
        local_key = cache.archive_key[..., cache.coverage :, :]
        local_value = cache.archive_value[..., cache.coverage :, :]
        if (
            cache.total_length <= self.config.exact_decode_limit
            or cache.state.size == 0
        ):
            exact_key = torch.cat((cache.sink_key, cache.archive_key), dim=2)
            exact_value = torch.cat((cache.sink_value, cache.archive_value), dim=2)
            exact_key = _repeat_kv(exact_key, int(query.size(1)))
            exact_value = _repeat_kv(exact_value, int(query.size(1)))
            exact_valid = torch.cat(
                (
                    torch.ones(
                        int(query.size(0)),
                        1,
                        dtype=torch.bool,
                        device=query.device,
                    ),
                    cache.archive_valid,
                ),
                dim=1,
            )
            exact_score = _scores(query, exact_key, scale).masked_fill(
                ~exact_valid[:, None, None, :], -torch.inf
            )
            output, _ = _attention(exact_score, exact_value)
        else:
            output = self._lod_attention(
                query,
                local_key,
                local_value,
                cache,
                scale=scale,
                query_offset=int(local_key.size(2)) - 1,
                full_regions=False,
            ).output
        return output, cache

    @torch.inference_mode()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache: PytorchLODCache | None = None,
        use_cache: bool = False,
        scale: float | None = None,
        logical_prefill_len: int | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
        output_buffer: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
    ) -> tuple[torch.Tensor, PytorchLODCache | None]:
        """Apply causal LoD and return an optional explicit generation cache."""

        del finalize_cache_for_decode
        _validate_qkv(query, key, value)
        scale = float(query.size(-1)) ** -0.5 if scale is None else float(scale)
        expected_output_shape = (*query.shape[:-1], int(value.size(-1)))
        if (
            output_buffer is not None
            and tuple(output_buffer.shape) != expected_output_shape
        ):
            raise ValueError("PyTorch LoD output buffer has incompatible shape")

        if cache is None:
            attention_length = int(query.size(2))
            logical_length = (
                attention_length
                if logical_prefill_len is None
                else int(logical_prefill_len)
            )
            if not 0 < logical_length <= attention_length:
                raise ValueError("logical prefill length must fit the input")
            if prefill_valid_starts is None:
                valid_starts = torch.zeros(
                    int(query.size(0)), dtype=torch.long, device=query.device
                )
            else:
                valid_starts = prefill_valid_starts.to(
                    device=query.device, dtype=torch.long
                )
            if tuple(valid_starts.shape) != (int(query.size(0)),):
                raise ValueError("prefill valid starts must have one entry per row")
            if bool(
                (valid_starts.lt(0) | valid_starts.ge(logical_length)).any().item()
            ):
                raise ValueError("every prefill row must contain its protected sink")
            output, next_cache = self._prefill(
                query,
                key,
                value,
                scale=scale,
                use_cache=use_cache,
                logical_length=logical_length,
                valid_starts=valid_starts,
            )
            if logical_length != attention_length:
                padded = query.new_zeros(
                    *query.shape[:-2], attention_length, int(value.size(-1))
                )
                padded[..., :logical_length, :].copy_(output)
                output = padded
        else:
            if logical_prefill_len is not None or prefill_valid_starts is not None:
                raise ValueError("padding metadata is valid only for initial prefill")
            outputs = []
            next_cache = cache
            for token in range(int(query.size(2))):
                output, next_cache = self._decode_one(
                    query[..., token : token + 1, :],
                    key[..., token : token + 1, :],
                    value[..., token : token + 1, :],
                    next_cache,
                    scale=scale,
                )
                outputs.append(output)
            output = torch.cat(outputs, dim=2)
            next_cache = next_cache.detached() if use_cache else None

        if output_buffer is not None:
            output_buffer.copy_(output)
            output = output_buffer
        return output, next_cache


__all__ = [
    "PytorchLODAttention",
    "PytorchLODCache",
    "PytorchLODConfig",
    "PytorchLODResult",
    "PytorchLODState",
]
