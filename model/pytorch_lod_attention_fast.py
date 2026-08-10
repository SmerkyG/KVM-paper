"""Fast, inference-oriented PyTorch backend for model-independent LOD attention.

This module reuses the readable cache and state-update lifecycle from
``pytorch_lod_attention`` while replacing its explicit score matrices with:

* SDPA for exact local attention;
* separately compiled FlexAttention for the count-corrected coarse field; and
* direct indexed attention for small selected leaf sets, with PyTorch's packed
  variable-length FlashAttention operator as the large-set fallback.

There are no model-specific or custom-kernel dependencies.  The input and
output contract is identical to ``CoarseLODAttention`` and
``TwoLevelLODAttention``.  This is an inference backend; use the reference
module when gradients through attention are required.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from torch.nn.attention.flex_attention import AuxRequest

from utils.flex_attention import separately_compiled_flex_attention

from .pytorch_lod_attention import (
    CoarseLODAttention,
    LODAttentionResult,
    LODConfig,
    LODState,
    TwoLevelLODAttention,
    _attention_from_scores,
    _normalize_open_count,
    _repeat_kv,
    coarse_lod_attention,
    two_level_lod_attention,
)


def _attention_needs_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def _fast_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    query_offset: int,
) -> torch.Tensor:
    query_length = int(query.size(2))
    key_length = int(key.size(2))
    if query_offset != key_length - query_length:
        raise ValueError("fast local attention requires suffix-aligned queries")
    enable_gqa = int(query.size(1)) != int(key.size(1))
    if query_length == key_length:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            enable_gqa=enable_gqa,
            scale=scale,
        )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_lower_right(query_length, key_length),
        enable_gqa=enable_gqa,
        scale=scale,
    )


def _route_state(
    query: torch.Tensor,
    state: LODState,
    *,
    max_routes: int,
    open_count: int | torch.Tensor,
    route_protected_prefix: int = 1,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads = int(query.size(1))
    key_value_heads = int(state.key_sum.size(1))
    groups = query_heads // key_value_heads
    mean_key = state.mean_key
    count = state.count
    if groups != 1:
        mean_key = mean_key.repeat_interleave(groups, dim=1)
        count = count.repeat_interleave(groups, dim=1)
    state_scores = torch.matmul(
        query.float(), mean_key.float().transpose(-1, -2)
    ) * scale
    state_scores = state_scores + count.clamp_min(1).log().float().unsqueeze(2)
    if route_protected_prefix < 0:
        raise ValueError("route_protected_prefix cannot be negative")
    protected = min(route_protected_prefix, state.slot_count)
    route_count = min(max_routes, state.slot_count - protected)
    open_counts = _normalize_open_count(
        open_count,
        shape=(int(query.size(0)), int(query.size(1)), int(query.size(2))),
        route_count=route_count,
        device=query.device,
    )
    if route_count == 0:
        empty_shape = (*query.shape[:-1], 0)
        return (
            torch.empty(empty_shape, dtype=torch.long, device=query.device),
            torch.empty(empty_shape, dtype=torch.bool, device=query.device),
        )
    with torch.no_grad():
        route_scores = state_scores.detach()
        if protected:
            route_scores = route_scores.clone()
            route_scores[..., :protected] = -torch.inf
        top_slots = route_scores.topk(
            route_count, dim=-1, largest=True, sorted=True
        ).indices
    rank = torch.arange(route_count, device=query.device)
    open_mask = rank.view(1, 1, 1, route_count) < open_counts.unsqueeze(-1)
    return top_slots, open_mask


def _fast_coarse_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    *,
    top_slots: torch.Tensor | None,
    open_mask: torch.Tensor | None,
    scale: float,
    query_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_key = state.mean_key
    state_value = state.mean_value
    coarse_key = torch.cat((state_key, local_key), dim=2).contiguous()
    coarse_value = torch.cat((state_value, local_value), dim=2).contiguous()
    query = query.contiguous()
    query_length = int(query.size(2))
    local_length = int(local_key.size(2))
    query_heads = int(query.size(1))
    key_value_heads = int(coarse_key.size(1))
    groups = query_heads // key_value_heads
    route_count = 0 if top_slots is None else int(top_slots.size(-1))

    log_count = state.count.clamp_min(1).log().float()
    if groups != 1:
        log_count = log_count.repeat_interleave(groups, dim=1)
    state_bias = log_count.unsqueeze(2).expand(
        -1, -1, query_length, -1
    ).clone()
    if route_count:
        excluded = torch.zeros_like(state_bias, dtype=torch.bool)
        for route_index in range(route_count):
            excluded.scatter_(
                -1,
                top_slots[..., route_index : route_index + 1],
                open_mask[..., route_index : route_index + 1],
            )
        state_bias.masked_fill_(excluded, -torch.inf)
    query_index = torch.arange(query_length, device=query.device).unsqueeze(-1)
    local_index = torch.arange(local_length, device=query.device).unsqueeze(0)
    local_visible = local_index <= query_index + query_offset
    local_bias = torch.zeros(
        query_length,
        local_length,
        dtype=state_bias.dtype,
        device=query.device,
    ).masked_fill(~local_visible, -torch.inf)
    local_bias = local_bias.view(1, 1, query_length, local_length).expand(
        int(query.size(0)), query_heads, query_length, local_length
    )
    attention_bias = torch.cat((state_bias, local_bias), dim=-1)

    # Inductor's FlexAttention decoding specialization is brittle for the
    # many different short tail lengths produced by chunked prefill.  The
    # score matrix is small in this regime, and a direct PyTorch matmul avoids
    # both repeated compilation and occasional no-valid-config failures.
    if query_length < 128:
        repeated_key = _repeat_kv(coarse_key, query_heads)
        repeated_value = _repeat_kv(coarse_value, query_heads)
        scores = torch.matmul(
            query.float(), repeated_key.float().transpose(-1, -2)
        ) * float(scale)
        return _attention_from_scores(scores + attention_bias, repeated_value)

    def score_mod(score, batch, head, q_idx, kv_idx):
        return score + attention_bias[batch, head, q_idx, kv_idx]

    output, auxiliary = separately_compiled_flex_attention(
        query,
        coarse_key,
        coarse_value,
        score_mod=score_mod,
        enable_gqa=groups != 1,
        scale=scale,
        return_aux=AuxRequest(lse=True),
    )
    if auxiliary.lse is None:
        raise RuntimeError("FlexAttention did not return coarse LSE")
    return output, auxiliary.lse


def _posting_lists(
    owner: torch.Tensor, state: LODState
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        counts = state.count.round().to(torch.long)
        starts = counts.cumsum(-1) - counts
        order = owner.argsort(dim=-1, stable=False)
    return order, starts


def _packed_leaf_attention(
    query: torch.Tensor,
    leaf_key: torch.Tensor,
    leaf_value: torch.Tensor,
    owner: torch.Tensor,
    state: LODState,
    top_slots: torch.Tensor,
    open_mask: torch.Tensor,
    posting_order: torch.Tensor,
    posting_starts: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch routed queries to exact leaf sets and run one packed attention."""
    batch, query_heads, query_length, head_dim = query.shape
    key_value_heads = int(leaf_key.size(1))
    groups = query_heads // key_value_heads
    state_size = state.slot_count
    route_count = int(top_slots.size(-1))
    history_length = int(owner.size(2))
    value_dim = int(leaf_value.size(-1))
    rows = batch * query_heads * query_length

    dense_output = query.new_zeros(rows, route_count, value_dim)
    dense_lse = torch.full(
        (rows, route_count),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    if route_count == 0 or not bool(open_mask.any().item()):
        exact_lse = torch.full(
            (batch, query_heads, query_length),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        return dense_output.sum(dim=1).reshape(
            batch, query_heads, query_length, value_dim
        ), exact_lse

    with torch.no_grad():
        row = torch.arange(rows, device=query.device, dtype=torch.long)
        route_row = row.unsqueeze(-1).expand(rows, route_count).reshape(-1)
        route_rank = torch.arange(route_count, device=query.device)
        route_rank = route_rank.view(1, route_count).expand(rows, route_count)
        route_rank = route_rank.reshape(-1)
        selected = open_mask.reshape(-1)
        route_row = route_row[selected]
        route_rank = route_rank[selected]
        route_slot = top_slots.reshape(-1)[selected]

        query_head = torch.div(route_row, query_length, rounding_mode="floor")
        query_head = query_head % query_heads
        query_batch = torch.div(
            route_row, query_heads * query_length, rounding_mode="floor"
        )
        key_value_head = torch.div(query_head, groups, rounding_mode="floor")
        leaf_row = query_batch * key_value_heads + key_value_head
        expert = leaf_row * state_size + route_slot
        expert_order = expert.argsort(stable=False)
        sorted_expert = expert[expert_order]
        unique_expert, query_lengths = torch.unique_consecutive(
            sorted_expert, return_counts=True
        )
        packed_query_row = route_row[expert_order]
        packed_route_rank = route_rank[expert_order]
        expert_leaf_row = torch.div(
            unique_expert, state_size, rounding_mode="floor"
        )
        expert_slot = unique_expert % state_size

        counts = state.count.round().to(torch.long).flatten(0, 1)
        starts = posting_starts.flatten(0, 1)
        order = posting_order.flatten(0, 1)
        key_lengths = counts[expert_leaf_row, expert_slot]
        cumulative_query = F.pad(query_lengths.cumsum(0), (1, 0)).to(torch.int32)
        cumulative_key = F.pad(key_lengths.cumsum(0), (1, 0)).to(torch.int32)
        max_query = int(query_lengths.max().item())
        max_key = int(key_lengths.max().item())

        expert_for_leaf = torch.repeat_interleave(
            torch.arange(
                int(key_lengths.numel()), device=query.device, dtype=torch.long
            ),
            key_lengths,
        )
        leaf_begin = (key_lengths.cumsum(0) - key_lengths)[expert_for_leaf]
        leaf_offset = torch.arange(
            int(key_lengths.sum().item()), device=query.device
        ) - leaf_begin
        packed_leaf_row = expert_leaf_row[expert_for_leaf]
        posting_rank = (
            starts[packed_leaf_row, expert_slot[expert_for_leaf]] + leaf_offset
        )
        leaf_position = order[packed_leaf_row, posting_rank]
        packed_route = packed_query_row * route_count + packed_route_rank

    packed_query = query.reshape(rows, head_dim).index_select(
        0, packed_query_row
    ).unsqueeze(1)
    flat_key = leaf_key[..., :history_length, :].flatten(0, 1)
    flat_value = leaf_value[..., :history_length, :].flatten(0, 1)
    packed_key = flat_key[packed_leaf_row, leaf_position].unsqueeze(1)
    packed_value = flat_value[packed_leaf_row, leaf_position].unsqueeze(1)

    packed_output, padded_lse, _, _, _ = torch.ops.aten._flash_attention_forward(
        packed_query,
        packed_key,
        packed_value,
        cumulative_query,
        cumulative_key,
        max_query,
        max_key,
        0.0,
        False,
        False,
        scale=scale,
    )
    expert_for_query = torch.repeat_interleave(
        torch.arange(
            int(query_lengths.numel()), device=query.device, dtype=torch.long
        ),
        query_lengths,
    )
    query_offset = torch.arange(
        int(packed_query.size(0)), device=query.device
    ) - cumulative_query[:-1].long()[expert_for_query]
    packed_lse = padded_lse[expert_for_query, 0, query_offset]
    dense_output.view(-1, value_dim).index_copy_(
        0, packed_route, packed_output.squeeze(1)
    )
    dense_lse.view(-1).index_copy_(0, packed_route, packed_lse)

    exact_lse = torch.logsumexp(dense_lse, dim=-1)
    valid = torch.isfinite(exact_lse)
    safe_lse = torch.where(valid, exact_lse, torch.zeros_like(exact_lse))
    weight = torch.exp(dense_lse - safe_lse.unsqueeze(-1))
    weight = torch.where(valid.unsqueeze(-1), weight, torch.zeros_like(weight))
    exact_output = (
        dense_output * weight.to(dense_output.dtype).unsqueeze(-1)
    ).sum(dim=1)
    return exact_output.reshape(
        batch, query_heads, query_length, value_dim
    ), exact_lse.reshape(batch, query_heads, query_length)


def _gathered_leaf_attention(
    query: torch.Tensor,
    leaf_key: torch.Tensor,
    leaf_value: torch.Tensor,
    owner: torch.Tensor,
    state: LODState,
    top_slots: torch.Tensor,
    open_mask: torch.Tensor,
    posting_order: torch.Tensor,
    posting_starts: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Directly gather small routed leaf sets; this is fastest for decode."""
    batch, query_heads, query_length, _ = query.shape
    key_value_heads = int(leaf_key.size(1))
    groups = query_heads // key_value_heads
    route_count = int(top_slots.size(-1))
    history_length = int(owner.size(2))
    value_dim = int(leaf_value.size(-1))
    if route_count == 0:
        return (
            query.new_zeros(batch, query_heads, query_length, value_dim),
            torch.full(
                (batch, query_heads, query_length),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            ),
        )

    kv_head = torch.div(
        torch.arange(query_heads, device=query.device),
        groups,
        rounding_mode="floor",
    )
    counts = state.count[:, kv_head]
    starts = posting_starts[:, kv_head]
    order = posting_order[:, kv_head]
    expanded_counts = counts.unsqueeze(2).expand(-1, -1, query_length, -1)
    expanded_starts = starts.unsqueeze(2).expand(-1, -1, query_length, -1)
    selected_counts = expanded_counts.gather(-1, top_slots)
    selected_starts = expanded_starts.gather(-1, top_slots)
    selected_counts = torch.where(
        open_mask, selected_counts, torch.zeros_like(selected_counts)
    )
    max_count = int(selected_counts.max().item())
    if max_count == 0:
        return (
            query.new_zeros(batch, query_heads, query_length, value_dim),
            torch.full(
                (batch, query_heads, query_length),
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            ),
        )

    offset = torch.arange(max_count, device=query.device)
    posting_rank = selected_starts.unsqueeze(-1) + offset
    valid = offset < selected_counts.unsqueeze(-1)
    posting_rank = posting_rank.clamp_max(history_length - 1)
    position = order.unsqueeze(2).unsqueeze(3).expand(
        -1, -1, query_length, route_count, -1
    ).gather(-1, posting_rank)

    rows = batch * query_heads
    positions_per_row = query_length * route_count * max_count
    flat_position = position.reshape(rows, positions_per_row)
    query_leaf_key = leaf_key[:, kv_head, :history_length].reshape(
        rows, history_length, -1
    )
    query_leaf_value = leaf_value[:, kv_head, :history_length].reshape(
        rows, history_length, value_dim
    )
    selected_key = query_leaf_key.gather(
        1,
        flat_position.unsqueeze(-1).expand(
            rows, positions_per_row, int(leaf_key.size(-1))
        ),
    ).reshape(
        batch, query_heads, query_length, route_count, max_count, -1
    )
    selected_value = query_leaf_value.gather(
        1,
        flat_position.unsqueeze(-1).expand(
            rows, positions_per_row, value_dim
        ),
    ).reshape(
        batch, query_heads, query_length, route_count * max_count, value_dim
    )
    scores = (
        query.float().unsqueeze(-2).unsqueeze(-2)
        * selected_key.float()
    ).sum(dim=-1) * scale
    scores = scores.masked_fill(~valid, -torch.inf).flatten(-2)
    exact_lse = torch.logsumexp(scores, dim=-1)
    finite = torch.isfinite(exact_lse)
    safe_scores = torch.where(
        finite.unsqueeze(-1), scores, torch.zeros_like(scores)
    )
    probability = torch.softmax(safe_scores, dim=-1)
    probability = torch.where(
        finite.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    exact_output = torch.matmul(
        probability.to(selected_value.dtype).unsqueeze(-2), selected_value
    ).squeeze(-2)
    return exact_output, exact_lse


def _prefer_gathered_leaves(
    query: torch.Tensor,
    state: LODState,
    top_slots: torch.Tensor,
    open_mask: torch.Tensor,
) -> bool:
    if int(top_slots.size(-1)) == 0:
        return True
    query_heads = int(query.size(1))
    key_value_heads = int(state.count.size(1))
    groups = query_heads // key_value_heads
    kv_head = torch.div(
        torch.arange(query_heads, device=query.device),
        groups,
        rounding_mode="floor",
    )
    counts = state.count[:, kv_head].unsqueeze(2).expand(
        -1, -1, int(query.size(2)), -1
    )
    selected_count = counts.gather(-1, top_slots)
    selected_count = torch.where(
        open_mask, selected_count, torch.zeros_like(selected_count)
    )
    max_count = int(selected_count.max().item())
    gathered_elements = (
        int(query.size(0))
        * query_heads
        * int(query.size(2))
        * int(top_slots.size(-1))
        * max_count
        * int(query.size(-1))
    )
    return gathered_elements <= 32_000_000


def _merge_two_branches(
    coarse_output: torch.Tensor,
    coarse_lse: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
) -> torch.Tensor:
    branch_lse = torch.stack((coarse_lse, exact_lse), dim=-1).float()
    weight = torch.softmax(branch_lse, dim=-1).to(coarse_output.dtype)
    return (
        coarse_output * weight[..., 0].unsqueeze(-1)
        + exact_output * weight[..., 1].unsqueeze(-1)
    )


def fast_coarse_lod_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    *,
    scale: float | None = None,
    query_offset: int | None = None,
) -> LODAttentionResult:
    """Fused coarse-state plus exact-local attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))
    if query_offset is None:
        query_offset = int(local_key.size(2)) - int(query.size(2))
    if (
        query.device.type == "cpu"
        or int(query.size(-1)) != int(local_value.size(-1))
        or _attention_needs_grad(query, local_key, local_value)
    ):
        return coarse_lod_attention(
            query,
            local_key,
            local_value,
            state,
            scale=scale,
            query_offset=query_offset,
        )
    output, lse = _fast_coarse_attention(
        query,
        local_key,
        local_value,
        state,
        top_slots=None,
        open_mask=None,
        scale=scale,
        query_offset=query_offset,
    )
    return LODAttentionResult(output=output, logsumexp=lse)


def fast_two_level_lod_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    owner: torch.Tensor,
    leaf_key: torch.Tensor,
    leaf_value: torch.Tensor,
    *,
    max_routes: int = 8,
    open_count: int | torch.Tensor = 8,
    route_protected_prefix: int = 1,
    scale: float | None = None,
    query_offset: int | None = None,
    postings: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> LODAttentionResult:
    """Fast packed exact leaves plus a fused coarse/local remainder."""
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))
    if query_offset is None:
        query_offset = int(local_key.size(2)) - int(query.size(2))
    fast_supported = (
        query.device.type != "cpu"
        and query.dtype in (torch.float16, torch.bfloat16)
        and leaf_key.dtype == query.dtype
        and leaf_value.dtype == query.dtype
        and int(query.size(-1)) == int(leaf_value.size(-1))
        and int(query.size(-1)) == int(local_value.size(-1))
        and not _attention_needs_grad(
            query, local_key, local_value, leaf_key, leaf_value
        )
    )
    if not fast_supported:
        return two_level_lod_attention(
            query,
            local_key,
            local_value,
            state,
            owner,
            leaf_key,
            leaf_value,
            max_routes=max_routes,
            open_count=open_count,
            route_protected_prefix=route_protected_prefix,
            scale=scale,
            query_offset=query_offset,
        )

    top_slots, open_mask = _route_state(
        query,
        state,
        max_routes=max_routes,
        open_count=open_count,
        route_protected_prefix=route_protected_prefix,
        scale=scale,
    )
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
    if postings is None:
        postings = _posting_lists(owner, state)
    leaf_attention = (
        _gathered_leaf_attention
        if _prefer_gathered_leaves(query, state, top_slots, open_mask)
        else _packed_leaf_attention
    )
    exact_output, exact_lse = leaf_attention(
        query,
        leaf_key,
        leaf_value,
        owner,
        state,
        top_slots,
        open_mask,
        postings[0],
        postings[1],
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


class _FastLocalMixin:
    def _local(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
        query_offset: int,
    ) -> torch.Tensor:
        if (
            query.device.type == "cpu"
            or int(query.size(-1)) != int(value.size(-1))
            or _attention_needs_grad(query, key, value)
        ):
            return super()._local(
                query,
                key,
                value,
                scale=scale,
                query_offset=query_offset,
            )
        return _fast_local_attention(
            query,
            key,
            value,
            scale=scale,
            query_offset=query_offset,
        )


class FastCoarseLODAttention(_FastLocalMixin, CoarseLODAttention):
    """Fast no-leaf LOD attention using SDPA and FlexAttention."""

    def _attend(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        state: LODState,
        **kwargs,
    ) -> torch.Tensor:
        scale = kwargs["scale"]
        query_offset = int(local_key.size(2)) - int(query.size(2))
        return fast_coarse_lod_attention(
            query,
            local_key,
            local_value,
            state,
            scale=scale,
            query_offset=query_offset,
        ).output


class FastTwoLevelLODAttention(_FastLocalMixin, TwoLevelLODAttention):
    """Fast top-k exact-leaf LOD attention with cached posting lists."""

    def __init__(
        self,
        config: LODConfig | None = None,
        *,
        default_open_count: int = 8,
    ) -> None:
        super().__init__(config, default_open_count=default_open_count)
        self._posting_key: tuple[int, tuple[int, ...], int] | None = None
        self._postings: tuple[torch.Tensor, torch.Tensor] | None = None

    def _cached_postings(
        self, owner: torch.Tensor, state: LODState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (owner.data_ptr(), tuple(owner.shape), state.slot_count)
        if key != self._posting_key or self._postings is None:
            self._postings = _posting_lists(owner, state)
            self._posting_key = key
        return self._postings

    def _attend(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        state: LODState,
        **kwargs,
    ) -> torch.Tensor:
        owner = kwargs["owner"]
        leaf_key = kwargs["leaf_key"]
        leaf_value = kwargs["leaf_value"]
        if owner is None or leaf_key is None or leaf_value is None:
            raise ValueError("fast two-level attention requires exact leaves")
        query_offset = int(local_key.size(2)) - int(query.size(2))
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
            route_protected_prefix=self.config.protected_prefix,
            scale=kwargs["scale"],
            query_offset=query_offset,
            postings=self._cached_postings(owner, state),
        ).output


__all__ = [
    "FastCoarseLODAttention",
    "FastTwoLevelLODAttention",
    "fast_coarse_lod_attention",
    "fast_two_level_lod_attention",
]
