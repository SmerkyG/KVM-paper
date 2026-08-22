"""Independent fixed-size slab construction for two-level LOD attention.

This is an inference-oriented reference engine for the slabbed LOD design.  A
completed ``slab_size`` token block is reduced independently to
``slots_per_slab`` semantic regions.  All completed slabs in an initial
prefill are reduced in parallel by folding the slab axis into the batch axis.
Queries use exact causal attention inside the current and immediately preceding
slab and top-k LOD attention over older regions. This matches classic KVM's
two-chunk BSWA geometry while retaining independent slab construction.

By default this first level retains every completed-slab summary.  An optional
delayed merge can reduce fixed groups of sufficiently old slab summaries while
preserving their original leaf ownership.  That is the first stage of the
recursive hierarchy needed to make the coarse field subquadratic.

The implementation deliberately uses ordinary PyTorch operations.  It is a
quality and architecture prototype; fixed-shape fused kernels can replace its
compression and sparse-leaf gathers without changing the cache contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .pytorch_lod_attention import (
    LODConfig,
    LODState,
    _attention_from_scores,
    _local_attention,
    _merge_lse_branches,
    _repeat_kv,
    _routing_state_scores,
    _scaled_scores,
    _state_scores_and_value,
    _validate_qkv,
)


@dataclass(frozen=True)
class SlabbedLODConfig(LODConfig):
    """Fixed-ratio state construction settings for slabbed LOD.

    Strided seeding spreads immutable initial region keys across a completed
    slab. Prefix seeding is retained as an ablation. Neither leaks future
    information because a slab summary is visible only to later slabs.
    """

    slab_size: int = 4096
    slots_per_slab: int = 256
    local_slabs: int = 2
    routing_chunk_size: int = 4096
    query_chunk_size: int = 4096
    seed_selection: str = "strided"
    merge_group_slabs: int = 0
    merged_slots_per_group: int = 0
    merge_budget_growth_factor: float = 0.0
    coarse_variance_bias: float = 0.0
    exact_closed_mass_oracle: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.slab_size <= 0:
            raise ValueError("slab_size must be positive")
        if not 0 < self.slots_per_slab <= self.slab_size:
            raise ValueError("slots_per_slab must lie in [1, slab_size]")
        if self.local_slabs <= 0:
            raise ValueError("local_slabs must be positive")
        if self.routing_chunk_size <= 0:
            raise ValueError("routing_chunk_size must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.merge_group_slabs == 0:
            if (
                self.merged_slots_per_group != 0
                or self.merge_budget_growth_factor != 0
            ):
                raise ValueError(
                    "delayed merge settings require merge_group_slabs"
                )
        elif self.merge_group_slabs <= 1:
            raise ValueError("merge_group_slabs must be zero or greater than one")
        elif not 0 < self.merged_slots_per_group <= (
            self.merge_group_slabs * self.slots_per_slab
        ):
            raise ValueError(
                "merged_slots_per_group must fit the grouped slab state"
            )
        if (
            self.merge_group_slabs
            and self.protected_prefix > self.merged_slots_per_group
        ):
            raise ValueError("protected_prefix exceeds the merged slab state")
        if self.merge_budget_growth_factor < 0:
            raise ValueError("merge_budget_growth_factor cannot be negative")
        if self.coarse_variance_bias < 0:
            raise ValueError("coarse_variance_bias cannot be negative")
        if self.exact_closed_mass_oracle and self.routing_leaf_mass_candidates:
            raise ValueError(
                "exact_closed_mass_oracle and candidate leaf mass conflict"
            )
        if self.routing_page_mass_candidates:
            raise ValueError("slabbed attention does not support page-mass routing")
        if (
            self.routing_leaf_mass_candidates
            and self.routing_leaf_mass_objective != "exact"
        ):
            raise ValueError(
                "slabbed candidate leaf-mass routing requires objective=exact"
            )
        if (
            self.routing_leaf_mass_top_p is not None
            or self.routing_leaf_mass_review_top_p is not None
        ):
            raise ValueError(
                "slabbed candidate leaf-mass routing currently uses fixed counts"
            )
        if self.seed_selection not in {"prefix", "strided"}:
            raise ValueError("seed_selection must be 'prefix' or 'strided'")
        if self.protected_prefix > self.slots_per_slab:
            raise ValueError("protected_prefix exceeds the first slab state")
        if (
            self.protected_prefix == self.slots_per_slab
            and self.slab_size > self.slots_per_slab
        ):
            raise ValueError(
                "slab compression requires an unprotected destination slot"
            )
        unsupported = (
            self.state_clustering_normalization != "none"
            or self.state_clustering_radial_bias != 0
            or self.state_clustering_centroid_rescale != "none"
            or self.state_clustering_query_metric != "none"
            or self.state_clustering_rope_filter != "none"
        )
        if unsupported:
            raise ValueError(
                "the slabbed PyTorch prototype currently supports raw-key "
                "state clustering only"
            )


@dataclass
class SlabbedLODCache:
    """Frozen slab summaries plus one exact active slab."""

    state: LODState
    owner: torch.Tensor
    leaf_key: torch.Tensor
    leaf_value: torch.Tensor
    postings: torch.Tensor
    active_key: torch.Tensor
    active_value: torch.Tensor
    total_length: int

    def detached(self) -> "SlabbedLODCache":
        return SlabbedLODCache(
            state=self.state.detached(),
            owner=self.owner.detach(),
            leaf_key=self.leaf_key.detach(),
            leaf_value=self.leaf_value.detach(),
            postings=self.postings.detach(),
            active_key=self.active_key.detach(),
            active_value=self.active_value.detach(),
            total_length=self.total_length,
        )


class SlabbedTwoLevelLODAttention(nn.Module):
    """Post-QKV Hugging Face engine for independent slabbed LOD."""

    def __init__(
        self,
        config: SlabbedLODConfig | None = None,
        *,
        default_open_count: int = 8,
    ) -> None:
        super().__init__()
        self.config = SlabbedLODConfig() if config is None else config
        if not 0 <= default_open_count <= self.config.max_routes:
            raise ValueError("default_open_count exceeds configured max_routes")
        self.default_open_count = default_open_count
        self.last_max_region_leaves = 0
        self.last_region_leaf_statistics: dict[str, float] = {}

    def _record_region_statistics(self, state: LODState) -> None:
        count = state.count.detach().float().flatten()
        if not int(count.numel()):
            self.last_max_region_leaves = 0
            self.last_region_leaf_statistics = {}
            return
        self.last_max_region_leaves = int(count.max().item())
        self.last_region_leaf_statistics = {
            "mean": float(count.mean().item()),
            "median": float(count.median().item()),
            "p95": float(torch.quantile(count, 0.95).item()),
            "p99": float(torch.quantile(count, 0.99).item()),
            "max": float(self.last_max_region_leaves),
        }

    @staticmethod
    def _empty_state(key: torch.Tensor, value: torch.Tensor) -> LODState:
        return LODState(
            key_sum=key[..., :0, :],
            value_sum=value[..., :0, :],
            count=torch.empty(
                *key.shape[:2], 0, dtype=torch.float32, device=key.device
            ),
        )

    @staticmethod
    def _block_local_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scale: float,
        query_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact blockwise BSWA using flash branches when on the GPU."""
        if query.device.type == "cpu":
            return _local_attention(
                query,
                key,
                value,
                scale=scale,
                query_offset=query_offset,
            )
        query_heads, _ = _validate_qkv(query, key, value)
        key = _repeat_kv(key, query_heads)
        value = _repeat_kv(value, query_heads)
        query_length = int(query.size(2))
        if query_offset < 0 or query_offset + query_length != int(key.size(2)):
            raise ValueError(
                "blockwise local keys must contain prior blocks followed by "
                "the current query block"
            )

        outputs = []
        lses = []
        if query_offset:
            previous_output, previous_lse, *_ = (
                torch.ops.aten._scaled_dot_product_flash_attention.default(
                    query.contiguous(),
                    key[..., :query_offset, :].contiguous(),
                    value[..., :query_offset, :].contiguous(),
                    0.0,
                    False,
                    False,
                    scale=scale,
                )
            )
            outputs.append(previous_output)
            lses.append(previous_lse)
        current_output, current_lse, *_ = (
            torch.ops.aten._scaled_dot_product_flash_attention.default(
                query.contiguous(),
                key[..., query_offset:, :].contiguous(),
                value[..., query_offset:, :].contiguous(),
                0.0,
                True,
                False,
                scale=scale,
            )
        )
        outputs.append(current_output)
        lses.append(current_lse)
        return _merge_lse_branches(outputs, lses)

    def _compress_rows(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        protect_prefix_rows: torch.Tensor,
    ) -> tuple[LODState, torch.Tensor]:
        """Reduce a batch of complete slabs with identical fixed-shape steps."""
        slab_size = self.config.slab_size
        slot_count = self.config.slots_per_slab
        if int(key.size(2)) != slab_size:
            raise ValueError("slab compression requires complete slabs")
        if tuple(protect_prefix_rows.shape) != (int(key.size(0)),):
            raise ValueError("protected-row mask has the wrong shape")

        if self.config.seed_selection == "prefix":
            seed_position = torch.arange(
                slot_count, dtype=torch.long, device=key.device
            )
        else:
            seed_position = torch.div(
                torch.arange(slot_count, dtype=torch.long, device=key.device)
                * slab_size,
                slot_count,
                rounding_mode="floor",
            )
        seed_slot = torch.arange(
            slot_count, dtype=torch.long, device=key.device
        )
        key_sum = key.index_select(2, seed_position).clone()
        value_sum = value.index_select(2, seed_position).clone()
        count = torch.ones(
            *key.shape[:2], slot_count, dtype=torch.float32, device=key.device
        )
        owner = torch.full(
            key.shape[:3], -1, dtype=torch.long, device=key.device
        )
        owner.scatter_(
            2,
            seed_position.view(1, 1, slot_count).expand(
                int(key.size(0)), int(key.size(1)), slot_count
            ),
            seed_slot.view(1, 1, slot_count).expand(
                int(key.size(0)), int(key.size(1)), slot_count
            ),
        )
        merge_mask = torch.ones(slab_size, dtype=torch.bool, device=key.device)
        merge_mask[seed_position] = False
        merge_position = torch.nonzero(merge_mask, as_tuple=False).flatten()
        protected = min(self.config.protected_prefix, slot_count)

        for begin in range(
            0, int(merge_position.numel()), self.config.routing_chunk_size
        ):
            end = min(
                int(merge_position.numel()),
                begin + self.config.routing_chunk_size,
            )
            block_position = merge_position[begin:end]
            block_key = key.index_select(2, block_position)
            block_value = value.index_select(2, block_position)
            with torch.no_grad():
                mean_key = key_sum / count.clamp_min(1).unsqueeze(-1).to(
                    key_sum.dtype
                )
                score = torch.matmul(
                    block_key.detach(), mean_key.detach().transpose(-1, -2)
                )
                if protected:
                    protected_mask = protect_prefix_rows.view(-1, 1, 1, 1)
                    protected_slot = (
                        torch.arange(slot_count, device=key.device)
                        .view(1, 1, 1, slot_count)
                        .lt(protected)
                    )
                    score.masked_fill_(protected_mask & protected_slot, -torch.inf)
                destination = score.argmax(dim=-1)

            key_delta = torch.zeros_like(key_sum)
            value_delta = torch.zeros_like(value_sum)
            key_delta.scatter_add_(
                2,
                destination.unsqueeze(-1).expand(
                    *destination.shape, int(block_key.size(-1))
                ),
                block_key,
            )
            value_delta.scatter_add_(
                2,
                destination.unsqueeze(-1).expand(
                    *destination.shape, int(block_value.size(-1))
                ),
                block_value,
            )
            count_delta = torch.zeros_like(count)
            count_delta.scatter_add_(
                2, destination, torch.ones_like(destination, dtype=count.dtype)
            )
            key_sum = key_sum + key_delta
            value_sum = value_sum + value_delta
            count = count + count_delta
            owner.scatter_(
                2,
                block_position.view(1, 1, -1).expand_as(destination),
                destination,
            )

        if bool(owner.lt(0).any().item()):
            raise AssertionError("slab owner construction lost leaves")
        return LODState(key_sum, value_sum, count), owner

    def _compress_complete_slabs(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slab_count: int,
    ) -> tuple[LODState, torch.Tensor]:
        """Compress every complete prefill slab concurrently."""
        if slab_count == 0:
            return self._empty_state(key, value), torch.empty(
                *key.shape[:2], 0, dtype=torch.long, device=key.device
            )
        batch, heads, _, key_dim = key.shape
        value_dim = int(value.size(-1))
        slab_size = self.config.slab_size
        frozen_length = slab_count * slab_size
        blocked_key = (
            key[..., :frozen_length, :]
            .reshape(batch, heads, slab_count, slab_size, key_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * slab_count, heads, slab_size, key_dim)
        )
        blocked_value = (
            value[..., :frozen_length, :]
            .reshape(batch, heads, slab_count, slab_size, value_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * slab_count, heads, slab_size, value_dim)
        )
        slab_index = torch.arange(
            slab_count, device=key.device, dtype=torch.long
        ).repeat(batch)
        blocked_state, blocked_owner = self._compress_rows(
            blocked_key,
            blocked_value,
            protect_prefix_rows=slab_index.eq(0),
        )
        slots = self.config.slots_per_slab
        key_sum = (
            blocked_state.key_sum.reshape(
                batch, slab_count, heads, slots, key_dim
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, heads, slab_count * slots, key_dim)
        )
        value_sum = (
            blocked_state.value_sum.reshape(
                batch, slab_count, heads, slots, value_dim
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, heads, slab_count * slots, value_dim)
        )
        count = (
            blocked_state.count.reshape(batch, slab_count, heads, slots)
            .permute(0, 2, 1, 3)
            .reshape(batch, heads, slab_count * slots)
        )
        owner = (
            blocked_owner.reshape(batch, slab_count, heads, slab_size)
            .permute(0, 2, 1, 3)
            + (
                torch.arange(slab_count, device=key.device, dtype=torch.long)
                .view(1, 1, slab_count, 1)
                * slots
            )
        ).reshape(batch, heads, frozen_length)
        return LODState(key_sum, value_sum, count), owner

    def _merge_state_group(
        self,
        state: LODState,
        *,
        protect_prefix: bool,
    ) -> tuple[LODState, torch.Tensor]:
        """Reduce one group of old slab summaries without revisiting its leaves."""
        source_slots = state.slot_count
        target_slots = self.config.merged_slots_per_group
        if source_slots != (
            self.config.merge_group_slabs * self.config.slots_per_slab
        ):
            raise ValueError("delayed merge requires one complete slab group")
        seed_position = torch.div(
            torch.arange(target_slots, device=state.key_sum.device)
            * source_slots,
            target_slots,
            rounding_mode="floor",
        )
        seed_slot = torch.arange(
            target_slots, dtype=torch.long, device=state.key_sum.device
        )
        key_sum = state.key_sum.index_select(2, seed_position).clone()
        value_sum = state.value_sum.index_select(2, seed_position).clone()
        count = state.count.index_select(2, seed_position).clone()
        owner = torch.full(
            state.count.shape,
            -1,
            dtype=torch.long,
            device=state.key_sum.device,
        )
        owner.scatter_(
            2,
            seed_position.view(1, 1, target_slots).expand(
                int(state.count.size(0)), int(state.count.size(1)), -1
            ),
            seed_slot.view(1, 1, target_slots).expand(
                int(state.count.size(0)), int(state.count.size(1)), -1
            ),
        )
        merge_mask = torch.ones(
            source_slots, dtype=torch.bool, device=state.key_sum.device
        )
        merge_mask[seed_position] = False
        merge_position = torch.nonzero(merge_mask, as_tuple=False).flatten()
        source_mean_key = state.mean_key
        protected = self.config.protected_prefix if protect_prefix else 0

        for begin in range(
            0, int(merge_position.numel()), self.config.routing_chunk_size
        ):
            block_position = merge_position[
                begin : begin + self.config.routing_chunk_size
            ]
            block_key_sum = state.key_sum.index_select(2, block_position)
            block_value_sum = state.value_sum.index_select(2, block_position)
            block_count = state.count.index_select(2, block_position)
            with torch.no_grad():
                centroid = key_sum / count.clamp_min(1).unsqueeze(-1).to(
                    key_sum.dtype
                )
                score = torch.matmul(
                    source_mean_key.index_select(2, block_position).detach(),
                    centroid.detach().transpose(-1, -2),
                )
                if protected:
                    score[..., :protected] = -torch.inf
                destination = score.argmax(dim=-1)

            key_delta = torch.zeros_like(key_sum)
            key_delta.scatter_add_(
                2,
                destination.unsqueeze(-1).expand_as(block_key_sum),
                block_key_sum,
            )
            value_delta = torch.zeros_like(value_sum)
            value_delta.scatter_add_(
                2,
                destination.unsqueeze(-1).expand_as(block_value_sum),
                block_value_sum,
            )
            count_delta = torch.zeros_like(count)
            count_delta.scatter_add_(2, destination, block_count)
            key_sum = key_sum + key_delta
            value_sum = value_sum + value_delta
            count = count + count_delta
            owner.scatter_(
                2,
                block_position.view(1, 1, -1).expand_as(destination),
                destination,
            )

        if bool(owner.lt(0).any().item()):
            raise AssertionError("delayed slab merge lost child summaries")
        return LODState(key_sum, value_sum, count), owner

    def _remote_view(
        self,
        state: LODState,
        owner: torch.Tensor,
        remote_slabs: int,
    ) -> tuple[LODState, torch.Tensor]:
        """Return the mixed merged-group and recent-slab state for a query."""
        slab_slots = self.config.slots_per_slab
        slab_size = self.config.slab_size
        remote_slots = remote_slabs * slab_slots
        remote_leaves = remote_slabs * slab_size
        if not self.config.merge_group_slabs:
            return (
                LODState(
                    state.key_sum[..., :remote_slots, :],
                    state.value_sum[..., :remote_slots, :],
                    state.count[..., :remote_slots],
                ),
                owner[..., :remote_leaves],
            )

        group_slabs = self.config.merge_group_slabs
        available_groups = remote_slabs // group_slabs
        complete_groups = available_groups
        if self.config.merge_budget_growth_factor:
            target_slots = max(
                math.floor(
                    self.config.merge_budget_growth_factor
                    * math.sqrt(remote_leaves)
                ),
                self.config.state_min_size,
            )
            slots_saved_per_group = (
                group_slabs * slab_slots
                - self.config.merged_slots_per_group
            )
            if slots_saved_per_group <= 0 or remote_slots <= target_slots:
                complete_groups = 0
            else:
                complete_groups = min(
                    available_groups,
                    math.ceil(
                        (remote_slots - target_slots)
                        / slots_saved_per_group
                    ),
                )
        grouped_slabs = complete_groups * group_slabs
        states: list[LODState] = []
        owners: list[torch.Tensor] = []
        output_offset = 0
        for group_index in range(complete_groups):
            slab_begin = group_index * group_slabs
            slot_begin = slab_begin * slab_slots
            slot_end = slot_begin + group_slabs * slab_slots
            leaf_begin = slab_begin * slab_size
            leaf_end = leaf_begin + group_slabs * slab_size
            grouped_state, child_owner = self._merge_state_group(
                LODState(
                    state.key_sum[..., slot_begin:slot_end, :],
                    state.value_sum[..., slot_begin:slot_end, :],
                    state.count[..., slot_begin:slot_end],
                ),
                protect_prefix=group_index == 0,
            )
            leaf_child = owner[..., leaf_begin:leaf_end] - slot_begin
            grouped_owner = child_owner.gather(2, leaf_child) + output_offset
            states.append(grouped_state)
            owners.append(grouped_owner)
            output_offset += grouped_state.slot_count

        remainder_slot_begin = grouped_slabs * slab_slots
        if remainder_slot_begin < remote_slots:
            remainder_state = LODState(
                state.key_sum[..., remainder_slot_begin:remote_slots, :],
                state.value_sum[..., remainder_slot_begin:remote_slots, :],
                state.count[..., remainder_slot_begin:remote_slots],
            )
            remainder_leaf_begin = grouped_slabs * slab_size
            remainder_owner = (
                owner[..., remainder_leaf_begin:remote_leaves]
                - remainder_slot_begin
                + output_offset
            )
            states.append(remainder_state)
            owners.append(remainder_owner)

        if not states:
            return self._empty_state(state.key_sum, state.value_sum), owner[..., :0]
        return (
            LODState(
                torch.cat([item.key_sum for item in states], dim=2),
                torch.cat([item.value_sum for item in states], dim=2),
                torch.cat([item.count for item in states], dim=2),
            ),
            torch.cat(owners, dim=2),
        )

    @staticmethod
    def _build_postings(owner: torch.Tensor, slot_count: int) -> torch.Tensor:
        """Pack each region's chronological leaf indices into padded rows."""
        batch, heads, leaf_count = owner.shape
        if slot_count == 0 or leaf_count == 0:
            return torch.empty(
                batch,
                heads,
                slot_count,
                0,
                dtype=torch.long,
                device=owner.device,
            )
        rows = batch * heads
        flat_owner = owner.reshape(rows, leaf_count)
        counts = torch.zeros(
            rows, slot_count, dtype=torch.long, device=owner.device
        )
        counts.scatter_add_(1, flat_owner, torch.ones_like(flat_owner))
        max_members = int(counts.max().item())
        order = flat_owner.argsort(dim=-1)
        sorted_owner = flat_owner.gather(1, order)
        starts = counts.cumsum(dim=-1) - counts
        ordinal = (
            torch.arange(leaf_count, device=owner.device).view(1, leaf_count)
            - starts.gather(1, sorted_owner)
        )
        posting = torch.full(
            (rows, slot_count, max_members),
            -1,
            dtype=torch.long,
            device=owner.device,
        )
        row = torch.arange(rows, device=owner.device).view(rows, 1)
        posting[row, sorted_owner, ordinal] = order
        return posting.reshape(batch, heads, slot_count, max_members)

    def _leaf_attention(
        self,
        query: torch.Tensor,
        slot: torch.Tensor,
        postings: torch.Tensor,
        leaf_key: torch.Tensor,
        leaf_value: torch.Tensor,
        *,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend once over the union of all routed regions per query."""
        query_heads = int(query.size(1))
        posting = _repeat_kv(postings, query_heads)
        key = _repeat_kv(leaf_key, query_heads)
        value = _repeat_kv(leaf_value, query_heads)
        members = int(posting.size(-1))
        if members == 0:
            output = value.new_zeros(
                *query.shape[:-1], int(value.size(-1))
            )
            lse = torch.full(
                query.shape[:-1],
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
            return output, lse
        flat_slot = slot.flatten(2, 3)
        selected = posting.gather(
            2, flat_slot.unsqueeze(-1).expand(*flat_slot.shape, members)
        ).reshape(*slot.shape[:-1], int(slot.size(-1)) * members)
        valid = selected.ge(0)
        flat_index = selected.clamp_min(0).flatten(2, 3)
        selected_key = key.gather(
            2,
            flat_index.unsqueeze(-1).expand(
                *flat_index.shape, int(key.size(-1))
            ),
        ).reshape(*selected.shape, int(key.size(-1)))
        selected_value = value.gather(
            2,
            flat_index.unsqueeze(-1).expand(
                *flat_index.shape, int(value.size(-1))
            ),
        ).reshape(*selected.shape, int(value.size(-1)))
        score = (
            query.float().unsqueeze(-2) * selected_key.float()
        ).sum(dim=-1) * float(scale)
        score.masked_fill_(~valid, -torch.inf)
        valid_row = valid.any(dim=-1)
        safe_score = torch.where(
            valid_row.unsqueeze(-1), score, torch.zeros_like(score)
        )
        probability = torch.softmax(safe_score, dim=-1)
        probability = torch.where(
            valid_row.unsqueeze(-1), probability, torch.zeros_like(probability)
        )
        output = (
            probability.to(selected_value.dtype).unsqueeze(-1) * selected_value
        ).sum(dim=-2)
        return output, torch.logsumexp(score, dim=-1)

    def _leaf_region_lse(
        self,
        query: torch.Tensor,
        slot: torch.Tensor,
        postings: torch.Tensor,
        leaf_key: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Return exact key mass for one selected region without loading values."""
        query_heads = int(query.size(1))
        posting = _repeat_kv(postings, query_heads)
        key = _repeat_kv(leaf_key, query_heads)
        members = int(posting.size(-1))
        if members == 0:
            return torch.full(
                query.shape[:-1],
                -torch.inf,
                dtype=torch.float32,
                device=query.device,
            )
        selected = posting.gather(
            2, slot.unsqueeze(-1).expand(*slot.shape, members)
        )
        valid = selected.ge(0)
        flat_index = selected.clamp_min(0).flatten(2, 3)
        selected_key = key.gather(
            2,
            flat_index.unsqueeze(-1).expand(
                *flat_index.shape, int(key.size(-1))
            ),
        ).reshape(*selected.shape, int(key.size(-1)))
        score = (
            query.float().unsqueeze(-2) * selected_key.float()
        ).sum(dim=-1) * float(scale)
        score.masked_fill_(~valid, -torch.inf)
        return torch.logsumexp(score, dim=-1)

    @staticmethod
    def _fused_route_coarse_attention(
        query: torch.Tensor,
        state: LODState,
        *,
        route_count: int,
        protected: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route top-k and form the coarse remainder in one Triton scan."""
        from .kernels.lod_kernels import route_logits_topk_coarse_attention

        batch, query_heads, query_length, head_dim = query.shape
        key_value_heads = int(state.key_sum.size(1))
        groups = query_heads // key_value_heads
        grouped_query = query.detach().reshape(
            batch,
            key_value_heads,
            groups,
            query_length,
            head_dim,
        )
        route_logits = torch.matmul(
            grouped_query,
            state.mean_key.detach().transpose(-1, -2).unsqueeze(2),
        ).reshape(batch, query_heads, query_length, state.slot_count)
        empty_key = state.key_sum[..., :0, :].contiguous()
        empty_value = state.value_sum[..., :0, :].contiguous()
        top_slot, coarse_output, coarse_lse = (
            route_logits_topk_coarse_attention(
                query.contiguous(),
                route_logits.contiguous(),
                state.value_sum.contiguous(),
                state.count.unsqueeze(-1).contiguous(),
                empty_key,
                empty_value,
                state_len=state.slot_count,
                kv_group_size=groups,
                scale=scale,
                route_count_bias=1.0,
                topk=route_count,
                protected_len=protected,
                stable_recompute=True,
            )
        )
        return top_slot, coarse_output, coarse_lse

    def _attend(
        self,
        query: torch.Tensor,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
        state: LODState,
        owner: torch.Tensor,
        postings: torch.Tensor,
        leaf_key: torch.Tensor,
        leaf_value: torch.Tensor,
        *,
        query_offset: int,
        open_count: int,
        scale: float,
    ) -> torch.Tensor:
        local_output, local_lse = self._block_local_attention(
            query,
            local_key,
            local_value,
            scale=scale,
            query_offset=query_offset,
        )
        if state.slot_count == 0:
            return local_output

        protected = min(self.config.protected_prefix, state.slot_count)
        route_count = min(
            open_count, self.config.max_routes, state.slot_count - protected
        )
        fused_route_coarse = (
            query.is_cuda
            and 0 < route_count <= 8
            and not self.config.exact_closed_mass_oracle
            and not self.config.coarse_variance_bias
            and not self.config.routing_leaf_mass_candidates
            and self.config.routing_normalization == "none"
            and self.config.routing_rope_fast_pairs == 0
            and self.config.routing_count_bias == 1.0
            and self.config.routing_variance_bias == 0.0
            and not self.config.routing_rope_jensen
        )
        if fused_route_coarse:
            top_slot, coarse_output, coarse_lse = (
                self._fused_route_coarse_attention(
                    query,
                    state,
                    route_count=route_count,
                    protected=protected,
                    scale=scale,
                )
            )
        else:
            state_score, state_value = _state_scores_and_value(
                query, state, scale
            )
            if self.config.exact_closed_mass_oracle:
                state_score = self._exact_region_mass_scores(
                    query, state, owner, leaf_key, scale=scale
                )
            elif self.config.coarse_variance_bias:
                state_score = state_score + self._coarse_variance_correction(
                    query, state, owner, leaf_key, scale=scale
                )
        if route_count and not fused_route_coarse:
            route_score = (
                state_score.detach()
                if (
                    self.config.routing_normalization == "none"
                    and self.config.routing_rope_fast_pairs == 0
                    and self.config.routing_count_bias == 1.0
                    and self.config.routing_variance_bias == 0.0
                    and not self.config.routing_rope_jensen
                )
                else _routing_state_scores(
                    query,
                    state,
                    scale,
                    self.config.routing_normalization,
                    self.config.routing_count_bias,
                    local_key,
                    self.config.routing_variance_bias,
                    self.config.routing_rope_dim,
                    self.config.routing_rope_fast_pairs,
                    self.config.routing_rope_jensen,
                )
            )
            if protected:
                route_score = route_score.clone()
                route_score[..., :protected] = -torch.inf
            candidate_count = min(
                self.config.routing_leaf_mass_candidates,
                state.slot_count - protected,
            )
            if candidate_count:
                candidate_slot = route_score.topk(
                    candidate_count, dim=-1
                ).indices
                candidate_lse = torch.stack(
                    [
                        self._leaf_region_lse(
                            query,
                            candidate_slot[..., candidate_index],
                            postings,
                            leaf_key,
                            scale=scale,
                        )
                        for candidate_index in range(candidate_count)
                    ],
                    dim=-1,
                )
                state_score = state_score.clone().scatter(
                    -1, candidate_slot, candidate_lse
                )
                chosen_candidate = candidate_lse.topk(
                    route_count, dim=-1
                ).indices
                top_slot = candidate_slot.gather(-1, chosen_candidate)
            else:
                top_slot = route_score.topk(route_count, dim=-1).indices
            excluded = torch.zeros_like(state_score, dtype=torch.bool)
            excluded.scatter_(-1, top_slot, True)
        else:
            if not fused_route_coarse:
                top_slot = torch.empty(
                    *query.shape[:-1], 0, dtype=torch.long, device=query.device
                )
                excluded = torch.zeros_like(state_score, dtype=torch.bool)

        if not fused_route_coarse:
            coarse_output, coarse_lse = _attention_from_scores(
                state_score.masked_fill(excluded, -torch.inf), state_value
            )
        outputs = [coarse_output, local_output]
        lses = [coarse_lse, local_lse]
        if route_count:
            if query.is_cuda:
                from .pytorch_lod_attention_fast import (
                    _packed_leaf_attention,
                    _posting_lists,
                )

                posting_order, posting_starts = _posting_lists(owner, state)
                route_output, route_lse = _packed_leaf_attention(
                    query,
                    leaf_key,
                    leaf_value,
                    owner,
                    state,
                    top_slot,
                    torch.ones_like(top_slot, dtype=torch.bool),
                    posting_order,
                    posting_starts,
                    scale=scale,
                )
            else:
                route_output, route_lse = self._leaf_attention(
                    query,
                    top_slot,
                    postings,
                    leaf_key,
                    leaf_value,
                    scale=scale,
                )
            outputs.append(route_output)
            lses.append(route_lse)
        return _merge_lse_branches(outputs, lses)[0]

    def _coarse_variance_correction(
        self,
        query: torch.Tensor,
        state: LODState,
        owner: torch.Tensor,
        leaf_key: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Second-order diagonal correction for omitted within-region mass."""
        query_heads = int(query.size(1))
        leaf_square = leaf_key.detach().float().square()
        squared_sum = torch.zeros_like(state.key_sum, dtype=torch.float32)
        squared_sum.scatter_add_(
            2, owner.unsqueeze(-1).expand_as(leaf_square), leaf_square
        )
        mean_square = squared_sum / state.count.clamp_min(1).unsqueeze(-1)
        variance = (
            mean_square - state.mean_key.detach().float().square()
        ).clamp_min(0)
        variance.masked_fill_(state.count.le(1).unsqueeze(-1), 0)
        variance = _repeat_kv(variance, query_heads)
        query_square = query.detach().float().square()
        return (
            0.5
            * float(self.config.coarse_variance_bias)
            * float(scale) ** 2
            * torch.einsum("bhqd,bhsd->bhqs", query_square, variance)
        )

    @staticmethod
    def _exact_region_mass_scores(
        query: torch.Tensor,
        state: LODState,
        owner: torch.Tensor,
        leaf_key: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """FP32 oracle for each region's exact constituent-key log mass."""
        query_heads = int(query.size(1))
        owner = _repeat_kv(owner, query_heads)
        leaf_key = _repeat_kv(leaf_key, query_heads)
        leaf_score = _scaled_scores(query, leaf_key, scale)
        leaf_owner = owner.unsqueeze(2).expand(
            int(query.size(0)),
            query_heads,
            int(query.size(2)),
            int(owner.size(2)),
        )
        maxima = torch.full(
            (*query.shape[:-1], state.slot_count),
            -torch.inf,
            device=query.device,
            dtype=torch.float32,
        )
        maxima.scatter_reduce_(
            -1, leaf_owner, leaf_score, reduce="amax", include_self=True
        )
        centered = leaf_score - maxima.gather(-1, leaf_owner)
        mass = torch.zeros_like(maxima)
        mass.scatter_add_(-1, leaf_owner, centered.exp())
        return maxima + mass.clamp_min(1e-30).log()

    def _attend_slab(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: LODState,
        owner: torch.Tensor,
        postings: torch.Tensor,
        leaf_key: torch.Tensor,
        leaf_value: torch.Tensor,
        *,
        query_offset: int = 0,
        open_count: int,
        scale: float,
    ) -> torch.Tensor:
        outputs = []
        for begin in range(0, int(query.size(2)), self.config.query_chunk_size):
            end = min(int(query.size(2)), begin + self.config.query_chunk_size)
            outputs.append(
                self._attend(
                    query[..., begin:end, :],
                    key,
                    value,
                    state,
                    owner,
                    postings,
                    leaf_key,
                    leaf_value,
                    query_offset=query_offset + begin,
                    open_count=open_count,
                    scale=scale,
                )
            )
        return torch.cat(outputs, dim=2)

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        open_count: int,
        scale: float,
        use_cache: bool,
    ) -> tuple[torch.Tensor, SlabbedLODCache | None]:
        sequence_length = int(query.size(2))
        slab_size = self.config.slab_size
        slab_count = sequence_length // slab_size
        frozen_length = slab_count * slab_size
        state, owner = self._compress_complete_slabs(
            key, value, slab_count
        )
        leaf_key = key[..., :frozen_length, :].to(self.config.leaf_dtype)
        leaf_value = value[..., :frozen_length, :].to(self.config.leaf_dtype)
        postings = self._build_postings(owner, state.slot_count)
        self._record_region_statistics(state)

        outputs = []
        total_slabs = math.ceil(sequence_length / slab_size)
        for slab_index in range(total_slabs):
            begin = slab_index * slab_size
            end = min(sequence_length, begin + slab_size)
            local_slab_begin = max(
                0, slab_index - self.config.local_slabs + 1
            )
            local_begin = local_slab_begin * slab_size
            remote_leaves = local_begin
            remote_state, remote_owner = self._remote_view(
                state, owner, local_slab_begin
            )
            remote_postings = (
                self._build_postings(remote_owner, remote_state.slot_count)
                if self.config.merge_group_slabs
                else postings[..., : remote_state.slot_count, :]
            )
            outputs.append(
                self._attend_slab(
                    query[..., begin:end, :],
                    key[..., local_begin:end, :],
                    value[..., local_begin:end, :],
                    remote_state,
                    remote_owner,
                    remote_postings,
                    leaf_key[..., :remote_leaves, :],
                    leaf_value[..., :remote_leaves, :],
                    query_offset=begin - local_begin,
                    open_count=open_count,
                    scale=scale,
                )
            )

        cache = None
        if use_cache:
            cache = SlabbedLODCache(
                state=state,
                owner=owner,
                leaf_key=leaf_key,
                leaf_value=leaf_value,
                postings=postings,
                active_key=key[..., frozen_length:, :],
                active_value=value[..., frozen_length:, :],
                total_length=sequence_length,
            ).detached()
        return torch.cat(outputs, dim=2), cache

    def _decode_one(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: SlabbedLODCache,
        *,
        open_count: int,
        scale: float,
    ) -> tuple[torch.Tensor, SlabbedLODCache]:
        active_key = torch.cat((cache.active_key, key), dim=2)
        active_value = torch.cat((cache.active_value, value), dim=2)
        if int(active_key.size(2)) > self.config.slab_size:
            raise AssertionError("active slab exceeded its fixed capacity")
        slots = self.config.slots_per_slab
        if cache.state.slot_count % slots:
            raise AssertionError("slab state is not aligned to complete slabs")
        complete_slabs = cache.state.slot_count // slots
        local_complete_slabs = min(
            self.config.local_slabs - 1, complete_slabs
        )
        remote_slabs = complete_slabs - local_complete_slabs
        remote_leaves = remote_slabs * self.config.slab_size
        exact_key = torch.cat(
            (cache.leaf_key[..., remote_leaves:, :], active_key), dim=2
        )
        exact_value = torch.cat(
            (cache.leaf_value[..., remote_leaves:, :], active_value), dim=2
        )
        remote_state, remote_owner = self._remote_view(
            cache.state, cache.owner, remote_slabs
        )
        remote_postings = (
            self._build_postings(remote_owner, remote_state.slot_count)
            if self.config.merge_group_slabs
            else cache.postings[..., : remote_state.slot_count, :]
        )
        output = self._attend_slab(
            query,
            exact_key,
            exact_value,
            remote_state,
            remote_owner,
            remote_postings,
            cache.leaf_key[..., :remote_leaves, :],
            cache.leaf_value[..., :remote_leaves, :],
            query_offset=int(exact_key.size(2)) - int(query.size(2)),
            open_count=open_count,
            scale=scale,
        )

        state = cache.state
        owner = cache.owner
        leaf_key = cache.leaf_key
        leaf_value = cache.leaf_value
        postings = cache.postings
        if int(active_key.size(2)) == self.config.slab_size:
            first_slab = state.slot_count == 0
            slab_state, slab_owner = self._compress_rows(
                active_key,
                active_value,
                protect_prefix_rows=torch.full(
                    (int(active_key.size(0)),),
                    first_slab,
                    dtype=torch.bool,
                    device=active_key.device,
                ),
            )
            slot_offset = state.slot_count
            state = LODState(
                torch.cat((state.key_sum, slab_state.key_sum), dim=2),
                torch.cat((state.value_sum, slab_state.value_sum), dim=2),
                torch.cat((state.count, slab_state.count), dim=2),
            )
            owner = torch.cat((owner, slab_owner + slot_offset), dim=2)
            leaf_key = torch.cat(
                (leaf_key, active_key.to(self.config.leaf_dtype)), dim=2
            )
            leaf_value = torch.cat(
                (leaf_value, active_value.to(self.config.leaf_dtype)), dim=2
            )
            postings = self._build_postings(owner, state.slot_count)
            self._record_region_statistics(state)
            active_key = active_key[..., :0, :]
            active_value = active_value[..., :0, :]

        next_cache = SlabbedLODCache(
            state=state,
            owner=owner,
            leaf_key=leaf_key,
            leaf_value=leaf_value,
            postings=postings,
            active_key=active_key,
            active_value=active_value,
            total_length=cache.total_length + 1,
        )
        return output, next_cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache: SlabbedLODCache | None = None,
        use_cache: bool = False,
        open_count: int | None = None,
        scale: float | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SlabbedLODCache | None]:
        """Apply slabbed LOD to post-projection, post-RoPE Q/K/V."""
        _validate_qkv(query, key, value)
        if int(query.size(2)) != int(key.size(2)):
            raise ValueError("query and new key/value lengths must match")
        if prefill_valid_starts is not None:
            raise NotImplementedError(
                "the slabbed prototype currently requires unpadded equal-length rows"
            )
        if open_count is None:
            open_count = self.default_open_count
        if not isinstance(open_count, int):
            raise TypeError("the slabbed prototype requires an integer open_count")
        if not 0 <= open_count <= self.config.max_routes:
            raise ValueError("open_count exceeds configured max_routes")
        if scale is None:
            scale = 1.0 / math.sqrt(float(query.size(-1)))
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
        for token in range(int(query.size(2))):
            output, next_cache = self._decode_one(
                query[..., token : token + 1, :],
                key[..., token : token + 1, :],
                value[..., token : token + 1, :],
                next_cache,
                open_count=open_count,
                scale=scale,
            )
            outputs.append(output)
        return (
            torch.cat(outputs, dim=2),
            next_cache.detached() if use_cache else None,
        )


__all__ = [
    "SlabbedLODCache",
    "SlabbedLODConfig",
    "SlabbedTwoLevelLODAttention",
]
