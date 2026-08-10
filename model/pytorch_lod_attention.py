"""Model-agnostic, pure-PyTorch levels-of-detail attention.

The public modules in this file consume query, key, and value tensors *after*
the model has applied its QKV projections, normalizers, and positional
encoding.  They therefore contain no Hugging Face or model-specific code.

``CoarseLODAttention`` keeps only count-corrected state summaries plus an exact
local window.  ``TwoLevelLODAttention`` additionally archives the original KV
leaves (BF16 by default), opens up to eight routed state regions, computes one
exact attention and log-sum-exp per opened region, and LSE-merges those regions
with the coarse remainder and local window.

Tensor layout follows current Hugging Face attention implementations:

    query: [batch, query_heads, query_length, key_head_dim]
    key:   [batch, key_value_heads, key_length, key_head_dim]
    value: [batch, key_value_heads, key_length, value_head_dim]

GQA is supported when ``query_heads`` is divisible by ``key_value_heads``.
Inputs are assumed to be unpadded causal sequences.  The returned output is
still head-separated; the caller remains responsible for transposing,
flattening, gating, and applying the output projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LODConfig:
    """Chunk, state-growth, and routing settings for LOD attention.

    Protected prefix slots remain exact in the coarse field and are excluded
    from both state merging and detailed-region routing.
    """

    chunk_size: int = 256
    local_window: int = 512
    state_growth_factor: float = 16.0
    state_min_size: int = 256
    protected_prefix: int = 1
    max_routes: int = 8
    leaf_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.local_window < 2 * self.chunk_size:
            raise ValueError("local_window must contain at least two chunks")
        if self.local_window % self.chunk_size:
            raise ValueError("local_window must be a multiple of chunk_size")
        if self.state_growth_factor < 0:
            raise ValueError("state_growth_factor cannot be negative")
        if self.state_min_size < 0:
            raise ValueError("state_min_size cannot be negative")
        if self.protected_prefix < 0:
            raise ValueError("protected_prefix cannot be negative")
        if not 0 <= self.max_routes <= 8:
            raise ValueError("max_routes must be between zero and eight")


@dataclass
class LODState:
    """Low-LOD state. Keys and values are sums; counts recover their means."""

    key_sum: torch.Tensor
    value_sum: torch.Tensor
    count: torch.Tensor

    @property
    def slot_count(self) -> int:
        return int(self.key_sum.size(2))

    @property
    def mean_key(self) -> torch.Tensor:
        return self.key_sum / self.count.to(self.key_sum.dtype).clamp_min(1).unsqueeze(-1)

    @property
    def mean_value(self) -> torch.Tensor:
        return self.value_sum / self.count.to(self.value_sum.dtype).clamp_min(1).unsqueeze(-1)

    def detached(self) -> LODState:
        return LODState(
            key_sum=self.key_sum.detach(),
            value_sum=self.value_sum.detach(),
            count=self.count.detach(),
        )


@dataclass
class LODCache:
    """Explicit cache returned to an HF attention wrapper for later decode."""

    state: LODState
    coverage: int
    recent_key: torch.Tensor
    recent_value: torch.Tensor
    total_length: int
    owner: torch.Tensor | None = None
    leaf_key: torch.Tensor | None = None
    leaf_value: torch.Tensor | None = None

    def detached(self) -> LODCache:
        return LODCache(
            state=self.state.detached(),
            coverage=self.coverage,
            recent_key=self.recent_key.detach(),
            recent_value=self.recent_value.detach(),
            total_length=self.total_length,
            owner=None if self.owner is None else self.owner.detach(),
            leaf_key=None if self.leaf_key is None else self.leaf_key.detach(),
            leaf_value=None if self.leaf_value is None else self.leaf_value.detach(),
        )


@dataclass
class LODAttentionResult:
    """Attention result and optional two-level routing diagnostics."""

    output: torch.Tensor
    logsumexp: torch.Tensor
    top_slots: torch.Tensor | None = None
    open_mask: torch.Tensor | None = None


def _validate_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must all be rank-four tensors")
    if query.size(0) != key.size(0) or key.shape[:3] != value.shape[:3]:
        raise ValueError("query, key, and value batch/sequence shapes disagree")
    if query.size(-1) != key.size(-1):
        raise ValueError("query and key head dimensions must match")
    query_heads = int(query.size(1))
    key_value_heads = int(key.size(1))
    if key_value_heads <= 0 or query_heads % key_value_heads:
        raise ValueError(
            "query head count must be divisible by key/value head count"
        )
    return query_heads, key_value_heads


def _repeat_kv(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    key_value_heads = int(tensor.size(1))
    if query_heads % key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    groups = query_heads // key_value_heads
    return tensor if groups == 1 else tensor.repeat_interleave(groups, dim=1)


def _empty_attention(
    query: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = value.new_zeros(*query.shape[:-1], int(value.size(-1)))
    lse = torch.full(
        query.shape[:-1],
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )
    return output, lse


def _attention_from_scores(
    scores: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized attention and FP32 LSE, including all-masked rows."""
    if int(scores.size(-1)) == 0:
        return _empty_attention(scores, value)
    valid = torch.isfinite(scores).any(dim=-1)
    safe_scores = torch.where(valid.unsqueeze(-1), scores, torch.zeros_like(scores))
    probability = torch.softmax(safe_scores, dim=-1)
    probability = torch.where(
        valid.unsqueeze(-1), probability, torch.zeros_like(probability)
    )
    output = torch.matmul(probability.to(value.dtype), value)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


def _scaled_scores(
    query: torch.Tensor, key: torch.Tensor, scale: float
) -> torch.Tensor:
    # FP32 score/LSE math makes this file a stable reference implementation;
    # values and returned outputs retain their original dtype.
    return torch.matmul(
        query.float(), key.float().transpose(-1, -2)
    ) * float(scale)


def _local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    query_offset: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads, _ = _validate_qkv(query, key, value)
    key = _repeat_kv(key, query_heads)
    value = _repeat_kv(value, query_heads)
    query_length = int(query.size(2))
    key_length = int(key.size(2))
    if query_offset is None:
        query_offset = key_length - query_length
    if query_offset < 0 or query_offset + query_length > key_length:
        raise ValueError("query_offset does not place queries inside local keys")
    scores = _scaled_scores(query, key, scale)
    query_index = torch.arange(query_length, device=query.device).unsqueeze(-1)
    key_index = torch.arange(key_length, device=query.device).unsqueeze(0)
    visible = key_index <= query_index + query_offset
    scores = scores.masked_fill(~visible.view(1, 1, query_length, key_length), -torch.inf)
    return _attention_from_scores(scores, value)


def _state_scores_and_value(
    query: torch.Tensor, state: LODState, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.slot_count == 0:
        value = state.value_sum.new_empty(
            int(query.size(0)), int(query.size(1)), 0, int(state.value_sum.size(-1))
        )
        scores = torch.empty(
            *query.shape[:-1], 0, device=query.device, dtype=torch.float32
        )
        return scores, value
    query_heads = int(query.size(1))
    mean_key = _repeat_kv(state.mean_key, query_heads)
    mean_value = _repeat_kv(state.mean_value, query_heads)
    count = _repeat_kv(state.count, query_heads)
    scores = _scaled_scores(query, mean_key, scale)
    scores = scores + count.clamp_min(1).log().float().unsqueeze(2)
    return scores, mean_value


def _state_attention(
    query: torch.Tensor,
    state: LODState,
    *,
    scale: float,
    excluded_slots: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores, value = _state_scores_and_value(query, state, scale)
    if excluded_slots is not None:
        if excluded_slots.shape != scores.shape:
            raise ValueError("excluded state-slot mask has the wrong shape")
        scores = scores.masked_fill(excluded_slots, -torch.inf)
    return _attention_from_scores(scores, value)


def _merge_lse_branches(
    outputs: list[torch.Tensor], lses: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not outputs or len(outputs) != len(lses):
        raise ValueError("outputs and LSEs must contain the same nonzero branches")
    branch_lse = torch.stack([lse.float() for lse in lses], dim=-1)
    total_lse = torch.logsumexp(branch_lse, dim=-1)
    valid = torch.isfinite(total_lse)
    safe_total = torch.where(valid, total_lse, torch.zeros_like(total_lse))
    weight = torch.exp(branch_lse - safe_total.unsqueeze(-1))
    weight = torch.where(valid.unsqueeze(-1), weight, torch.zeros_like(weight))
    output = torch.stack(outputs, dim=-2)
    merged = (output * weight.to(output.dtype).unsqueeze(-1)).sum(dim=-2)
    return merged, total_lse


def coarse_lod_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    state: LODState,
    *,
    scale: float | None = None,
    query_offset: int | None = None,
) -> LODAttentionResult:
    """Attend to count-corrected state summaries and an exact local field."""
    _validate_qkv(query, local_key, local_value)
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))
    state_output, state_lse = _state_attention(query, state, scale=scale)
    local_output, local_lse = _local_attention(
        query,
        local_key,
        local_value,
        scale=scale,
        query_offset=query_offset,
    )
    output, lse = _merge_lse_branches(
        [state_output, local_output], [state_lse, local_lse]
    )
    return LODAttentionResult(output=output, logsumexp=lse)


def _normalize_open_count(
    open_count: int | torch.Tensor,
    *,
    shape: tuple[int, int, int],
    route_count: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(open_count, int):
        result = torch.full(shape, open_count, dtype=torch.long, device=device)
    elif isinstance(open_count, torch.Tensor):
        if open_count.is_floating_point() and bool(
            (open_count != open_count.round()).any().item()
        ):
            raise ValueError("open_count tensor must contain integers")
        try:
            result = torch.broadcast_to(open_count.to(device=device, dtype=torch.long), shape)
        except RuntimeError as exc:
            raise ValueError(
                "open_count must be scalar or broadcastable to [batch, heads, queries]"
            ) from exc
    else:
        raise TypeError("open_count must be an integer or tensor")
    if bool((result < 0).any().item()):
        raise ValueError("open_count cannot be negative")
    return result.clamp_max(route_count)


def two_level_lod_attention(
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
) -> LODAttentionResult:
    """Replace opened coarse regions with independently normalized exact leaves.

    ``max_routes`` controls how many ranked regions are identified (at most
    eight). ``route_protected_prefix`` leaves exact protected prefix entries in
    the coarse softmax but prevents them from consuming detailed routes.
    ``open_count`` selects how many of that ordered list are actually opened
    and may be either a scalar or a per-query tensor broadcastable to
    ``[batch, query_heads, query_length]``.
    """
    query_heads, key_value_heads = _validate_qkv(query, local_key, local_value)
    _validate_qkv(query, leaf_key, leaf_value)
    if int(leaf_key.size(1)) != key_value_heads:
        raise ValueError("local and leaf archives use different KV-head counts")
    if owner.ndim != 3 or owner.shape[:2] != leaf_key.shape[:2]:
        raise ValueError("owner must have shape [batch, key_value_heads, leaves]")
    history_length = int(owner.size(2))
    if history_length > int(leaf_key.size(2)):
        raise ValueError("owner archive is longer than the leaf archive")
    if history_length and (
        bool((owner < 0).any().item())
        or bool((owner >= state.slot_count).any().item())
    ):
        raise ValueError("owner archive contains an invalid state slot")
    if not 0 <= max_routes <= 8:
        raise ValueError("max_routes must be between zero and eight")
    if route_protected_prefix < 0:
        raise ValueError("route_protected_prefix cannot be negative")
    if scale is None:
        scale = 1.0 / math.sqrt(float(query.size(-1)))

    state_scores, state_value = _state_scores_and_value(query, state, scale)
    protected = min(route_protected_prefix, state.slot_count)
    route_count = min(max_routes, state.slot_count - protected)
    open_counts = _normalize_open_count(
        open_count,
        shape=(int(query.size(0)), query_heads, int(query.size(2))),
        route_count=route_count,
        device=query.device,
    )
    if route_count:
        with torch.no_grad():
            route_scores = state_scores.detach()
            if protected:
                route_scores = route_scores.clone()
                route_scores[..., :protected] = -torch.inf
            top_slots = route_scores.topk(
                route_count, dim=-1, largest=True, sorted=True
            ).indices
        route_rank = torch.arange(route_count, device=query.device)
        open_mask = route_rank.view(1, 1, 1, route_count) < open_counts.unsqueeze(-1)
        excluded = torch.zeros_like(state_scores, dtype=torch.bool)
        for route_index in range(route_count):
            excluded.scatter_(
                -1,
                top_slots[..., route_index : route_index + 1],
                open_mask[..., route_index : route_index + 1],
            )
    else:
        top_slots = torch.empty(
            *query.shape[:-1], 0, dtype=torch.long, device=query.device
        )
        open_mask = torch.empty(
            *query.shape[:-1], 0, dtype=torch.bool, device=query.device
        )
        excluded = torch.zeros_like(state_scores, dtype=torch.bool)

    coarse_scores = state_scores.masked_fill(excluded, -torch.inf)
    coarse_output, coarse_lse = _attention_from_scores(coarse_scores, state_value)
    local_output, local_lse = _local_attention(
        query,
        local_key,
        local_value,
        scale=scale,
        query_offset=query_offset,
    )
    outputs = [coarse_output, local_output]
    lses = [coarse_lse, local_lse]

    if route_count:
        repeated_key = _repeat_kv(leaf_key[..., :history_length, :], query_heads)
        repeated_value = _repeat_kv(
            leaf_value[..., :history_length, :], query_heads
        )
        repeated_owner = _repeat_kv(owner, query_heads)
        leaf_scores = _scaled_scores(query, repeated_key, scale)
        for route_index in range(route_count):
            slot = top_slots[..., route_index]
            selected = repeated_owner.unsqueeze(2) == slot.unsqueeze(-1)
            selected = selected & open_mask[..., route_index].unsqueeze(-1)
            route_scores = leaf_scores.masked_fill(~selected, -torch.inf)
            route_output, route_lse = _attention_from_scores(
                route_scores, repeated_value
            )
            outputs.append(route_output)
            lses.append(route_lse)

    output, lse = _merge_lse_branches(outputs, lses)
    return LODAttentionResult(
        output=output,
        logsumexp=lse,
        top_slots=top_slots,
        open_mask=open_mask,
    )


def _gather_sequence(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return tensor.gather(
        2, index.unsqueeze(-1).expand(*index.shape, int(tensor.size(-1)))
    )


class _PytorchLODAttention(nn.Module):
    """Shared causal state/cache lifecycle for the two public variants."""

    def __init__(
        self,
        config: LODConfig | None,
        *,
        store_leaves: bool,
        default_open_count: int,
    ) -> None:
        super().__init__()
        self.config = LODConfig() if config is None else config
        self.store_leaves = store_leaves
        self.default_open_count = default_open_count
        if not store_leaves and default_open_count:
            raise ValueError("coarse-only attention cannot open leaf regions")
        if not 0 <= default_open_count <= self.config.max_routes:
            raise ValueError("default_open_count exceeds configured max_routes")

    @staticmethod
    def _empty_state(key: torch.Tensor, value: torch.Tensor) -> LODState:
        return LODState(
            key_sum=key[..., :0, :],
            value_sum=value[..., :0, :],
            count=torch.empty(
                *key.shape[:2], 0, dtype=torch.float32, device=key.device
            ),
        )

    def _desired_state_size(
        self, context_length: int, available_context: int, current_size: int
    ) -> int:
        target = max(
            math.floor(
                self.config.state_growth_factor
                * math.sqrt(max(context_length, 0))
            ),
            self.config.state_min_size,
        )
        return max(current_size, min(target, available_context))

    def _bswa_begin(self, total_length: int) -> int:
        rounded_end = (
            (total_length + self.config.chunk_size - 1)
            // self.config.chunk_size
            * self.config.chunk_size
        )
        return max(0, rounded_end - self.config.local_window)

    def _initialize_state(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[LODState, torch.Tensor]:
        length = int(key.size(2))
        count = torch.ones(
            *key.shape[:2], length, dtype=torch.float32, device=key.device
        )
        owner = (
            torch.arange(length, dtype=torch.long, device=key.device)
            .view(1, 1, length)
            .expand(int(key.size(0)), int(key.size(1)), length)
        )
        return LODState(key_sum=key, value_sum=value, count=count), owner

    def _update_state(
        self,
        state: LODState,
        overflow_key: torch.Tensor,
        overflow_value: torch.Tensor,
        *,
        context_length: int,
        available_context: int,
    ) -> tuple[LODState, torch.Tensor]:
        overflow_length = int(overflow_key.size(2))
        if overflow_length == 0:
            owner = torch.empty(
                *overflow_key.shape[:3], dtype=torch.long, device=overflow_key.device
            )
            return state, owner
        current_size = state.slot_count
        desired_size = self._desired_state_size(
            context_length, available_context, current_size
        )
        append_count = min(max(desired_size - current_size, 0), overflow_length)

        with torch.no_grad():
            similarity = torch.matmul(
                overflow_key.detach(), state.mean_key.detach().transpose(-1, -2)
            )
            max_similarity = similarity.max(dim=-1).values
            order = max_similarity.argsort(dim=-1, descending=False)
            append_index = torch.sort(order[..., :append_count], dim=-1).values
            merge_index = torch.sort(order[..., append_count:], dim=-1).values

        owner = torch.full(
            overflow_key.shape[:3], -1, dtype=torch.long, device=overflow_key.device
        )
        if append_count:
            append_key = _gather_sequence(overflow_key, append_index)
            append_value = _gather_sequence(overflow_value, append_index)
            append_count_tensor = torch.ones(
                *append_key.shape[:3], dtype=torch.float32, device=append_key.device
            )
            state = LODState(
                key_sum=torch.cat((state.key_sum, append_key), dim=2),
                value_sum=torch.cat((state.value_sum, append_value), dim=2),
                count=torch.cat((state.count, append_count_tensor), dim=2),
            )
            append_slot = (
                torch.arange(
                    current_size,
                    current_size + append_count,
                    dtype=torch.long,
                    device=overflow_key.device,
                )
                .view(1, 1, append_count)
                .expand_as(append_index)
            )
            owner.scatter_(2, append_index, append_slot)

        merge_key = _gather_sequence(overflow_key, merge_index)
        merge_value = _gather_sequence(overflow_value, merge_index)
        if int(merge_key.size(2)) == 0:
            return state, owner

        protected = min(self.config.protected_prefix, state.slot_count)
        if protected >= state.slot_count:
            raise ValueError("all state slots are protected from merging")
        with torch.no_grad():
            route_score = torch.matmul(
                merge_key.detach(), state.mean_key.detach().transpose(-1, -2)
            )
            route_score[..., :protected] = -torch.inf
            destination = route_score.argmax(dim=-1)
        assignment = F.one_hot(destination, num_classes=state.slot_count)
        assignment_key = assignment.to(merge_key.dtype).transpose(-1, -2)
        assignment_value = assignment.to(merge_value.dtype).transpose(-1, -2)
        state = LODState(
            key_sum=state.key_sum + torch.matmul(assignment_key, merge_key),
            value_sum=state.value_sum + torch.matmul(
                assignment_value, merge_value
            ),
            count=state.count
            + assignment.float().transpose(-1, -2).sum(dim=-1),
        )
        owner.scatter_(2, merge_index, destination)
        return state, owner

    def _compress_to(
        self,
        state: LODState,
        owner: torch.Tensor | None,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        coverage: int,
        target: int,
        context_length: int,
        coverage_offset: int = 0,
    ) -> tuple[LODState, torch.Tensor | None, int]:
        if target < coverage or target > int(key.size(2)):
            raise ValueError("invalid LOD compression target")
        while coverage < target:
            block_end = min(target, coverage + self.config.chunk_size)
            block_key = key[..., coverage:block_end, :]
            block_value = value[..., coverage:block_end, :]
            if state.slot_count == 0:
                state, block_owner = self._initialize_state(block_key, block_value)
            else:
                state, block_owner = self._update_state(
                    state,
                    block_key,
                    block_value,
                    context_length=context_length,
                    available_context=coverage_offset + block_end,
                )
            if owner is not None:
                owner = torch.cat((owner, block_owner), dim=2)
            coverage = block_end
        return state, owner, coverage

    @staticmethod
    def _slice_open_count(
        open_count: int | torch.Tensor, begin: int, end: int
    ) -> int | torch.Tensor:
        if (
            isinstance(open_count, int)
            or open_count.ndim == 0
            or int(open_count.size(-1)) == 1
        ):
            return open_count
        return open_count[..., begin:end]

    def _attend(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        state: LODState,
        *,
        owner: torch.Tensor | None,
        leaf_key: torch.Tensor | None,
        leaf_value: torch.Tensor | None,
        open_count: int | torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        query_offset = int(local_key.size(2)) - int(query.size(2))
        if not self.store_leaves:
            if isinstance(open_count, int):
                nonzero = open_count != 0
            else:
                nonzero = bool((open_count != 0).any().item())
            if nonzero:
                raise ValueError("coarse-only attention requires open_count=0")
            return coarse_lod_attention(
                query,
                local_key,
                local_value,
                state,
                scale=scale,
                query_offset=query_offset,
            ).output
        if owner is None or leaf_key is None or leaf_value is None:
            raise ValueError("two-level attention requires an exact leaf archive")
        return two_level_lod_attention(
            query,
            local_key,
            local_value,
            state,
            owner,
            leaf_key,
            leaf_value,
            max_routes=self.config.max_routes,
            open_count=open_count,
            route_protected_prefix=self.config.protected_prefix,
            scale=scale,
            query_offset=query_offset,
        ).output

    def _local(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
        query_offset: int,
    ) -> torch.Tensor:
        return _local_attention(
            query,
            key,
            value,
            scale=scale,
            query_offset=query_offset,
        )[0]

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        open_count: int | torch.Tensor,
        scale: float,
        use_cache: bool,
    ) -> tuple[torch.Tensor, LODCache | None]:
        sequence_length = int(query.size(2))
        front_length = min(sequence_length, self.config.local_window)
        front_output = self._local(
            query[..., :front_length, :],
            key[..., :front_length, :],
            value[..., :front_length, :],
            scale=scale,
            query_offset=0,
        )
        outputs = [front_output]
        state = self._empty_state(key, value)
        coverage = 0
        owner = (
            torch.empty(
                *key.shape[:2], 0, dtype=torch.long, device=key.device
            )
            if self.store_leaves
            else None
        )
        leaf_key = key.to(self.config.leaf_dtype) if self.store_leaves else None
        leaf_value = value.to(self.config.leaf_dtype) if self.store_leaves else None
        exact_lookback = self.config.local_window - self.config.chunk_size

        for query_begin in range(
            front_length, sequence_length, self.config.chunk_size
        ):
            query_end = min(sequence_length, query_begin + self.config.chunk_size)
            local_begin = max(0, query_begin - exact_lookback)
            state, owner, coverage = self._compress_to(
                state,
                owner,
                key,
                value,
                coverage=coverage,
                target=local_begin,
                context_length=query_begin,
            )
            outputs.append(
                self._attend(
                    query[..., query_begin:query_end, :],
                    key[..., local_begin:query_end, :],
                    value[..., local_begin:query_end, :],
                    state,
                    owner=owner,
                    leaf_key=leaf_key,
                    leaf_value=leaf_value,
                    open_count=self._slice_open_count(
                        open_count, query_begin, query_end
                    ),
                    scale=scale,
                )
            )

        if use_cache:
            decode_coverage = self._bswa_begin(sequence_length + 1)
            state, owner, coverage = self._compress_to(
                state,
                owner,
                key,
                value,
                coverage=coverage,
                target=decode_coverage,
                context_length=sequence_length,
            )
            cache = LODCache(
                state=state,
                coverage=coverage,
                recent_key=key[..., coverage:, :],
                recent_value=value[..., coverage:, :],
                total_length=sequence_length,
                owner=owner,
                leaf_key=leaf_key,
                leaf_value=leaf_value,
            ).detached()
        else:
            cache = None
        return torch.cat(outputs, dim=2), cache

    def _decode_one(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: LODCache,
        *,
        open_count: int | torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, LODCache]:
        if int(query.size(2)) != 1 or int(key.size(2)) != 1:
            raise ValueError("_decode_one requires one query/key/value token")
        recent_key = torch.cat((cache.recent_key, key), dim=2)
        recent_value = torch.cat((cache.recent_value, value), dim=2)
        leaf_key = cache.leaf_key
        leaf_value = cache.leaf_value
        if self.store_leaves:
            if leaf_key is None or leaf_value is None:
                raise ValueError("two-level decode cache has no leaf archive")
            leaf_key = torch.cat((leaf_key, key.to(self.config.leaf_dtype)), dim=2)
            leaf_value = torch.cat(
                (leaf_value, value.to(self.config.leaf_dtype)), dim=2
            )

        total_length = cache.total_length + 1
        target_coverage = self._bswa_begin(total_length)
        overflow_length = target_coverage - cache.coverage
        if overflow_length < 0 or overflow_length > int(recent_key.size(2)):
            raise AssertionError("LOD decode coverage drifted")
        state, owner, coverage = self._compress_to(
            cache.state,
            cache.owner,
            recent_key,
            recent_value,
            coverage=0,
            target=overflow_length,
            context_length=cache.total_length,
            coverage_offset=cache.coverage,
        )
        # _compress_to used coordinates relative to the recent buffer.
        coverage = cache.coverage + coverage
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
            output = self._attend(
                query,
                recent_key,
                recent_value,
                state,
                owner=owner,
                leaf_key=leaf_key,
                leaf_value=leaf_value,
                open_count=open_count,
                scale=scale,
            )
        next_cache = LODCache(
            state=state,
            coverage=coverage,
            recent_key=recent_key,
            recent_value=recent_value,
            total_length=total_length,
            owner=owner,
            leaf_key=leaf_key,
            leaf_value=leaf_value,
        )
        return output, next_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache: LODCache | None = None,
        use_cache: bool = False,
        open_count: int | torch.Tensor | None = None,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, LODCache | None]:
        """Apply causal LOD attention directly to post-RoPE Q/K/V tensors."""
        _validate_qkv(query, key, value)
        if int(query.size(2)) != int(key.size(2)):
            raise ValueError("query and new key/value lengths must match")
        if scale is None:
            scale = 1.0 / math.sqrt(float(query.size(-1)))
        if open_count is None:
            open_count = self.default_open_count
        if cache is None:
            return self._prefill(
                query,
                key,
                value,
                open_count=open_count,
                scale=scale,
                use_cache=use_cache,
            )

        outputs = []
        next_cache = cache
        for token_index in range(int(query.size(2))):
            token_open_count = self._slice_open_count(
                open_count, token_index, token_index + 1
            )
            output, next_cache = self._decode_one(
                query[..., token_index : token_index + 1, :],
                key[..., token_index : token_index + 1, :],
                value[..., token_index : token_index + 1, :],
                next_cache,
                open_count=token_open_count,
                scale=scale,
            )
            outputs.append(output)
        return torch.cat(outputs, dim=2), next_cache.detached() if use_cache else None


class CoarseLODAttention(_PytorchLODAttention):
    """Simple low-LOD state plus exact-local attention with no leaf archive."""

    def __init__(self, config: LODConfig | None = None) -> None:
        super().__init__(config, store_leaves=False, default_open_count=0)


class TwoLevelLODAttention(_PytorchLODAttention):
    """Low LOD plus independently normalized exact BF16 leaf regions."""

    def __init__(
        self,
        config: LODConfig | None = None,
        *,
        default_open_count: int = 8,
    ) -> None:
        super().__init__(
            config, store_leaves=True, default_open_count=default_open_count
        )


__all__ = [
    "CoarseLODAttention",
    "LODAttentionResult",
    "LODCache",
    "LODConfig",
    "LODState",
    "TwoLevelLODAttention",
    "coarse_lod_attention",
    "two_level_lod_attention",
]
