"""Inference-only, model-independent Triton LOD attention core.

This module consumes head-separated query, key, and value tensors after their
model-specific projections, normalization, and positional encoding. Old KV
leaves are partitioned into a ``16*sqrt(T)`` state; a query expands the leaves
of its top-k state slots and uses count-corrected mean KV summaries for every
other slot. The exact and coarse branches are combined with their log-sum-exp
statistics.

The exact-leaf archive is stored in 16-token pages, while decode keeps only a
bounded recent KV window. Model adapters live in separate modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .kernels.paged_leaf_attention import (
    append_paged_kv,
    append_quantized_virtual_paged_kv,
    append_virtual_paged_kv,
    fused_decode_paged_lod_attention,
    new_fused_decode_buffers,
    paged_leaf_attention,
    query_major_paged_leaf_attention,
    query_major_indexed_residual_page_attention,
    query_major_residual_page_attention,
    quantize_page_summaries_int8,
    quantize_virtual_paged_kv_int4,
)
from .kernels.lod_kernels import (
    apply_residual_mass_opening,
    bipartite_reduce_overflow,
    merge_attention_branches,
    merge_attention_branches_with_sink,
    merge_state_in_place,
    new_route_buffers,
    new_state_delta_buffers,
    new_state_maxsim_buffers,
    route_logits_coarse_attention,
    route_logits_topk_coarse_attention,
    route_top8_scores_grouped,
    route_top8_state_grouped,
    streaming_state_maxsim,
)
from .kvm_mixer import _all_idx, _gather_by_idx, _split_append_merge_idx_by_maxsim


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_sequence(x: torch.Tensor, length: int) -> torch.Tensor:
    missing = length - int(x.size(2))
    if missing < 0:
        raise ValueError(f"cannot pad sequence of length {x.size(2)} to {length}")
    return x if missing == 0 else F.pad(x, (0, 0, 0, missing))


def _merge_lse_branches(
    left_output: torch.Tensor,
    left_lse: torch.Tensor,
    right_output: torch.Tensor,
    right_lse: torch.Tensor,
) -> torch.Tensor:
    branch_lse = torch.stack((left_lse, right_lse), dim=-1).float()
    weights = torch.softmax(branch_lse, dim=-1).to(left_output.dtype)
    return (
        left_output * weights[..., 0].unsqueeze(-1)
        + right_output * weights[..., 1].unsqueeze(-1)
    )


class TritonLODAttentionCore(nn.Module):
    """Projection-free mass-corrected top-k LOD attention implementation."""

    chunk_len = 256
    local_len = 512
    prefill_chunk_len = 256
    prefill_local_len = 512
    prefill_state_update_len = 256
    decode_state_update_len = 256
    decode_cache_headroom = 256
    state_growth_factor = 16.0
    state_min_len = 256
    sink_len = 1
    # A protected singleton is already exact in the coarse branch, so opening
    # its one-token leaf would only consume a detailed-region route.
    exclude_sink_from_routes = True
    # Keep the exact sink outside the centroid state and leaf archive. Its
    # attention contribution is merged as a separate exact branch.
    separate_sink_cache = False
    two_level_topk = 8
    prefill_two_level_topk: int | None = None
    prefill_max_leaf_tokens: int | None = None
    leaf_attention_backend = "packed"
    leaf_page_size = 16
    leaf_inline_pages_per_slot = 128
    leaf_overflow_hash_factor = 4
    leaf_hash_probes = 8
    leaf_block_m = 16
    leaf_block_n = 32
    leaf_short_block_n = 16
    leaf_short_context = 16384
    leaf_num_warps = 2
    leaf_waves_per_eu = 1
    leaf_layout = "query"
    leaf_key_quant_bits = 0
    leaf_value_quant_bits = 0
    leaf_quant_group_size = 32
    leaf_quant_scale_mode = "max"
    leaf_append_quant_scale_mode = "max"
    page_summary_quant_bits = 8
    page_summary_scale_mode = "l2"
    virtual_page_storage = False
    recursive_page_lod = False
    recursive_page_block_n = 16
    dynamic_open_prefill_top_p: float | None = None
    dynamic_open_decode_top_p: float | None = None
    dynamic_open_prefill_residual_mass: float | None = None
    dynamic_open_decode_residual_mass: float | None = None
    dynamic_open_residual_use_state_bound = False
    reuse_dynamic_local_attention = False
    collect_dynamic_open_stats = False
    fused_decode_attention = True
    fused_decode_state_route = True
    decode_split_kv = 8
    decode_use_dot = False
    decode_block_n = 16
    decode_num_warps = 2
    decode_route_group_size = 32
    decode_route_num_warps = 2
    decode_route_reduce_num_warps = 4
    decode_final_reduce_num_warps = 4
    decode_fuse_final_reduce = False
    decode_route_use_dot = True
    decode_route_gqa_grouped = True
    clone_decode_routes = False
    # Dense BF16 state aggregation is faster for batch-one inference;
    # the KVM-style FP32 delta path remains available for larger batches.
    fused_state_update = False
    auto_fused_state_update = True
    reuse_state_update_similarity = True
    fused_state_maxsim = False
    state_maxsim_block_m = 16
    state_maxsim_block_n = 32
    state_maxsim_num_warps = 4
    fused_state_routing = True
    direct_fused_state_routing = True
    route_gqa_matmul = False
    coarse_enable_gqa = True
    coarse_compact_bias = True
    reuse_route_logits_for_coarse = True
    coarse_route_block_m = 16
    coarse_route_block_n = 32
    coarse_route_num_warps = 8
    fused_prefill_route_coarse = False
    split_prefill_local_attention = False
    fused_prefill_residual_opening = False
    overflow_bipartite_merge = False
    overflow_bipartite_block_size = 32
    overflow_bipartite_positional_halves = False
    overflow_bipartite_keep_ratio = 0.5
    state_merge_before_append = False
    state_append_subblock_size = 0
    state_union_bipartite = False
    state_precompact_direct_append = False

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(self.num_key_value_groups, dim=1)

    def _desired_state_len(
        self, ctx_len: int, available_context: int, current_state_len: int
    ) -> int:
        separated = self.sink_len if self.separate_sink_cache else 0
        # Masked left-padding slots deliberately consume part of the shared
        # schedule. Reserving replacements here can round state capacity up
        # for every row in the batch and make recursive decode much slower.
        target = max(
            math.floor(self.state_growth_factor * math.sqrt(max(ctx_len, 0))),
            self.state_min_len,
        )
        available = max(available_context - separated, 0)
        return max(current_state_len, min(target, available))

    def _protected_state_len(self, state_len: int) -> int:
        if self.separate_sink_cache:
            return 0
        return min(self.sink_len, state_len)

    def _state_capacity(self, total_len: int, current_state_len: int) -> int:
        # One chunk of headroom avoids recompilation during short generation.
        target = self._desired_state_len(
            total_len + self.chunk_len, total_len + self.chunk_len, current_state_len
        )
        return _round_up(target, self.chunk_len)

    def _bswa_begin(self, total_len: int) -> int:
        bswa_end = _round_up(total_len, self.chunk_len)
        return max(0, bswa_end - self.local_len)

    @staticmethod
    def _mean(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        return x / counts.to(x.dtype).clamp_min(1)

    def _split_append_merge_indices(
        self,
        scores: torch.Tensor,
        n_append: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split overflow indices globally or with balanced local quotas."""

        overflow_len = int(scores.size(-1))
        subblock_size = self.state_append_subblock_size
        if subblock_size <= 0:
            sorted_idx = torch.argsort(scores.float(), dim=-1, descending=False)
            return (
                torch.sort(sorted_idx[..., :n_append], dim=-1).values,
                torch.sort(sorted_idx[..., n_append:], dim=-1).values,
            )
        if overflow_len % subblock_size:
            raise ValueError(
                "overflow length must be divisible by the append subblock size"
            )
        subblocks = overflow_len // subblock_size
        base_quota, remainder = divmod(n_append, subblocks)
        if base_quota + int(remainder > 0) > subblock_size:
            raise ValueError("append quota exceeds its subblock size")
        block_scores = scores.float().reshape(*scores.shape[:-1], subblocks, subblock_size)
        block_order = torch.argsort(block_scores, dim=-1, descending=False)
        block_offset = (
            torch.arange(subblocks, device=scores.device, dtype=torch.long)
            * subblock_size
        ).view(*([1] * (scores.ndim - 1)), subblocks, 1)
        block_indices = block_order + block_offset
        high_quota = base_quota + int(remainder > 0)
        append_parts = []
        merge_parts = []
        if remainder:
            append_parts.append(
                block_indices[..., :remainder, :high_quota].flatten(-2)
            )
            merge_parts.append(
                block_indices[..., :remainder, high_quota:].flatten(-2)
            )
        if remainder < subblocks:
            append_parts.append(
                block_indices[..., remainder:, :base_quota].flatten(-2)
            )
            merge_parts.append(
                block_indices[..., remainder:, base_quota:].flatten(-2)
            )
        append_idx = (
            append_parts[0]
            if len(append_parts) == 1
            else torch.cat(append_parts, dim=-1)
        )
        merge_idx = (
            merge_parts[0]
            if len(merge_parts) == 1
            else torch.cat(merge_parts, dim=-1)
        )
        return (
            torch.sort(append_idx, dim=-1).values,
            torch.sort(merge_idx, dim=-1).values,
        )

    def _union_bipartite_round(
        self,
        key_sum: torch.Tensor,
        value_sum: torch.Tensor,
        counts: torch.Tensor,
        target_len: int,
        round_index: int,
        *,
        protected_len: int | None = None,
        positional_halves: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one balanced ToMe-style contraction to a state union."""

        batch, heads, current_len, key_dim = key_sum.shape
        if protected_len is None:
            protected_len = self._protected_state_len(current_len)
        protected = min(protected_len, current_len, target_len)
        matchable = current_len - protected
        left_count = (matchable + 1) // 2
        right_count = matchable // 2
        merges = current_len - target_len
        if merges < 0 or merges > left_count or not right_count:
            raise ValueError("invalid bipartite union contraction target")

        rows = batch * heads
        if positional_halves:
            left_slots = torch.arange(
                protected,
                protected + left_count,
                device=key_sum.device,
                dtype=torch.long,
            ).view(1, left_count).expand(rows, left_count)
            right_slots = torch.arange(
                protected + left_count,
                current_len,
                device=key_sum.device,
                dtype=torch.long,
            ).view(1, right_count).expand(rows, right_count)
        else:
            pair_position = torch.arange(
                right_count, device=key_sum.device, dtype=torch.long
            ).view(1, right_count)
            row = torch.arange(rows, device=key_sum.device, dtype=torch.long).view(
                rows, 1
            )
            salt = int(getattr(self, "layer_idx", 0)) + 131 * round_index
            hashed = (
                pair_position * 0x9E3779B1
                + row * 0x85EBCA77
                + current_len * 0xC2B2AE3D
                + salt * 0x27D4EB2F
            )
            hashed = (hashed ^ (hashed >> 16)) * 0x45D9F3B
            swap = (hashed ^ (hashed >> 16)) & 1
            pair_base = protected + 2 * pair_position
            paired_left = pair_base + swap
            right_slots = pair_base + (1 - swap)
            if left_count > right_count:
                left_slots = torch.cat(
                    (
                        paired_left,
                        torch.full(
                            (rows, 1),
                            protected + 2 * right_count,
                            dtype=torch.long,
                            device=key_sum.device,
                        ),
                    ),
                    dim=-1,
                )
            else:
                left_slots = paired_left

        mean_key = self._mean(key_sum, counts).reshape(rows, current_len, key_dim)
        left_key = torch.gather(
            mean_key,
            1,
            left_slots.unsqueeze(-1).expand(rows, left_count, key_dim),
        ).to(torch.bfloat16)
        right_key = torch.gather(
            mean_key,
            1,
            right_slots.unsqueeze(-1).expand(rows, right_count, key_dim),
        ).to(torch.bfloat16)
        similarity = torch.matmul(left_key, right_key.transpose(-1, -2))
        nearest_score, nearest_position = similarity.max(dim=-1)
        nearest_slot = torch.gather(right_slots, 1, nearest_position)

        if (
            protected == 0
            and current_len % 2 == 0
            and target_len * 2 == current_len
        ):
            def exact_half_sum(values: torch.Tensor) -> torch.Tensor:
                feature_dim = int(values.size(-1))
                output = torch.zeros(
                    batch,
                    heads,
                    target_len,
                    feature_dim,
                    dtype=values.dtype,
                    device=values.device,
                )
                return output.scatter_add_(
                    2,
                    assignment.reshape(batch, heads, current_len, 1).expand_as(
                        values
                    ),
                    values,
                )

            assignment = torch.empty(
                rows,
                current_len,
                dtype=torch.long,
                device=key_sum.device,
            )
            compact = torch.arange(
                target_len, dtype=torch.long, device=key_sum.device
            ).view(1, target_len).expand(rows, target_len)
            assignment.scatter_(1, right_slots, compact)
            assignment.scatter_(1, left_slots, nearest_position)
            return (
                exact_half_sum(key_sum),
                exact_half_sum(value_sum),
                exact_half_sum(counts),
                assignment.reshape(batch, heads, current_len),
            )

        slot = torch.arange(
            current_len, device=key_sum.device, dtype=torch.long
        ).expand(rows, current_len)
        destination = slot.clone()
        active = torch.ones_like(slot, dtype=torch.bool)
        if merges:
            selected = torch.topk(
                nearest_score.float(),
                k=merges,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            selected_source = torch.gather(left_slots, 1, selected)
            selected_destination = torch.gather(nearest_slot, 1, selected)
            destination.scatter_(1, selected_source, selected_destination)
            active.scatter_(
                1, selected_source, torch.zeros_like(selected_source, dtype=torch.bool)
            )
        compact_slot = torch.cumsum(active.to(torch.long), dim=-1) - 1
        assignment = torch.gather(compact_slot, 1, destination).reshape(
            batch, heads, current_len
        )

        def cluster_sum(values: torch.Tensor) -> torch.Tensor:
            output = torch.zeros(
                *values.shape[:2],
                target_len,
                int(values.size(-1)),
                dtype=values.dtype,
                device=values.device,
            )
            return output.scatter_add_(
                2, assignment.unsqueeze(-1).expand_as(values), values
            )

        return (
            cluster_sum(key_sum),
            cluster_sum(value_sum),
            cluster_sum(counts),
            assignment,
        )

    def _reduce_overflow_balanced(
        self,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        input_counts: torch.Tensor | None = None,
        *,
        keep_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce fixed-size overflow blocks while preserving leaf membership."""

        block_size = self.overflow_bipartite_block_size
        if keep_ratio is None:
            keep_ratio = self.overflow_bipartite_keep_ratio
        if block_size <= 0 or block_size % 2:
            raise ValueError("overflow block size must be positive and even")
        if not 0.5 <= keep_ratio <= 1.0:
            raise ValueError("overflow keep ratio must be in [0.5, 1.0]")
        overflow_len = int(overflow_k.size(2))
        batch, heads, _, key_dim = overflow_k.shape
        value_dim = int(overflow_v.size(-1))
        if (
            input_counts is None
            and keep_ratio == 0.5
            and not self.overflow_bipartite_positional_halves
            and block_size <= 256
            and overflow_len % block_size == 0
        ):
            return bipartite_reduce_overflow(
                overflow_k.contiguous(),
                overflow_v.contiguous(),
                block_size=block_size,
                balanced=True,
                salt=int(getattr(self, "layer_idx", 0)),
            )
        full_blocks, tail_len = divmod(overflow_len, block_size)
        reduced_block = max(block_size // 2, math.ceil(block_size * keep_ratio))
        reduced_k_parts = []
        reduced_v_parts = []
        reduced_count_parts = []
        membership_parts = []
        reduced_offset = 0
        if full_blocks:
            full_len = full_blocks * block_size

            def blocked(values: torch.Tensor, dim: int) -> torch.Tensor:
                return (
                    values[..., :full_len, :]
                    .reshape(batch, heads, full_blocks, block_size, dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(batch * full_blocks, heads, block_size, dim)
                )

            blocked_k = blocked(overflow_k, key_dim)
            blocked_v = blocked(overflow_v, value_dim)
            blocked_counts = (
                torch.ones(
                    batch * full_blocks,
                    heads,
                    block_size,
                    1,
                    dtype=torch.float32,
                    device=overflow_k.device,
                )
                if input_counts is None
                else blocked(input_counts, 1)
            )
            if (
                keep_ratio == 0.5
                and not self.overflow_bipartite_positional_halves
                and block_size <= 256
                and input_counts is None
            ):
                blocked_k, blocked_v, blocked_counts, blocked_membership = (
                    bipartite_reduce_overflow(
                        blocked_k.contiguous(),
                        blocked_v.contiguous(),
                        block_size=block_size,
                        balanced=True,
                        salt=int(getattr(self, "layer_idx", 0)),
                    )
                )
            else:
                blocked_k, blocked_v, blocked_counts, blocked_membership = (
                    self._union_bipartite_round(
                        blocked_k,
                        blocked_v,
                        blocked_counts,
                        reduced_block,
                        0,
                        protected_len=0,
                        positional_halves=(
                            self.overflow_bipartite_positional_halves
                        ),
                    )
                )

            def unblocked(values: torch.Tensor, dim: int) -> torch.Tensor:
                return (
                    values.reshape(
                        batch, full_blocks, heads, reduced_block, dim
                    )
                    .permute(0, 2, 1, 3, 4)
                    .reshape(batch, heads, full_blocks * reduced_block, dim)
                )

            reduced_k_parts.append(unblocked(blocked_k, key_dim))
            reduced_v_parts.append(unblocked(blocked_v, value_dim))
            reduced_count_parts.append(unblocked(blocked_counts, 1))
            block_offset = (
                torch.arange(
                    full_blocks,
                    device=overflow_k.device,
                    dtype=blocked_membership.dtype,
                )
                * reduced_block
            ).view(1, full_blocks, 1, 1)
            membership_parts.append(
                (
                    blocked_membership.reshape(
                        batch, full_blocks, heads, block_size
                    )
                    + block_offset
                )
                .permute(0, 2, 1, 3)
                .reshape(batch, heads, full_len)
            )
            reduced_offset = full_blocks * reduced_block
        if tail_len:
            tail_k = overflow_k[..., -tail_len:, :]
            tail_v = overflow_v[..., -tail_len:, :]
            tail_counts = (
                torch.ones(
                    batch,
                    heads,
                    tail_len,
                    1,
                    dtype=torch.float32,
                    device=overflow_k.device,
                )
                if input_counts is None
                else input_counts[..., -tail_len:, :]
            )
            tail_target = max((tail_len + 1) // 2, math.ceil(tail_len * keep_ratio))
            if (
                tail_target == (tail_len + 1) // 2
                and not self.overflow_bipartite_positional_halves
                and input_counts is None
                and tail_len % 2 == 0
                and tail_len <= 256
            ):
                tail_k, tail_v, tail_counts, tail_membership = (
                    bipartite_reduce_overflow(
                        tail_k.contiguous(),
                        tail_v.contiguous(),
                        block_size=tail_len,
                        balanced=True,
                        salt=int(getattr(self, "layer_idx", 0)) + full_blocks,
                    )
                )
            elif tail_target < tail_len:
                tail_k, tail_v, tail_counts, tail_membership = (
                    self._union_bipartite_round(
                        tail_k,
                        tail_v,
                        tail_counts,
                        tail_target,
                        full_blocks,
                        protected_len=0,
                        positional_halves=self.overflow_bipartite_positional_halves,
                    )
                )
            else:
                tail_membership = (
                    torch.arange(
                        tail_len, device=overflow_k.device, dtype=torch.long
                    )
                    .view(1, 1, tail_len)
                    .expand(batch, heads, tail_len)
                )
            reduced_k_parts.append(tail_k)
            reduced_v_parts.append(tail_v)
            reduced_count_parts.append(tail_counts)
            membership_parts.append(tail_membership + reduced_offset)
        return (
            torch.cat(reduced_k_parts, dim=2),
            torch.cat(reduced_v_parts, dim=2),
            torch.cat(reduced_count_parts, dim=2),
            torch.cat(membership_parts, dim=2),
        )

    def _update_state_union_bipartite(
        self,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        *,
        state_len: int,
        ctx_len: int,
        available_context: int,
        state_capacity: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Optionally precontract overflow blocks, then contract the state union."""

        if self.leaf_attention_backend == "paged":
            raise ValueError(
                "union bipartite state updates currently require packed leaves"
            )
        current_state_len = state_len
        overflow_membership = None
        if self.overflow_bipartite_merge:
            block_size = self.overflow_bipartite_block_size
            overflow_len = int(overflow_k.size(2))
            if block_size <= 0 or block_size % 2:
                raise ValueError(
                    "union overflow block size must be positive and even"
                )
            batch, heads, _, key_dim = overflow_k.shape
            value_dim = int(overflow_v.size(-1))
            full_blocks, tail_len = divmod(overflow_len, block_size)
            reduced_k_parts = []
            reduced_v_parts = []
            reduced_count_parts = []
            membership_parts = []
            reduced_offset = 0
            if full_blocks:
                full_len = full_blocks * block_size

                def blocked(values: torch.Tensor, dim: int) -> torch.Tensor:
                    return (
                        values[..., :full_len, :]
                        .reshape(batch, heads, full_blocks, block_size, dim)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(batch * full_blocks, heads, block_size, dim)
                    )

                blocked_k = blocked(overflow_k, key_dim)
                blocked_v = blocked(overflow_v, value_dim)
                blocked_counts = torch.ones(
                    batch * full_blocks,
                    heads,
                    block_size,
                    1,
                    dtype=torch.float32,
                    device=overflow_k.device,
                )
                (
                    blocked_k,
                    blocked_v,
                    blocked_counts,
                    blocked_membership,
                ) = self._union_bipartite_round(
                    blocked_k,
                    blocked_v,
                    blocked_counts,
                    block_size // 2,
                    0,
                    protected_len=0,
                    positional_halves=self.overflow_bipartite_positional_halves,
                )
                reduced_block = block_size // 2

                def unblocked(values: torch.Tensor, dim: int) -> torch.Tensor:
                    return (
                        values.reshape(
                            batch, full_blocks, heads, reduced_block, dim
                        )
                        .permute(0, 2, 1, 3, 4)
                        .reshape(batch, heads, full_blocks * reduced_block, dim)
                    )

                reduced_k_parts.append(unblocked(blocked_k, key_dim))
                reduced_v_parts.append(unblocked(blocked_v, value_dim))
                reduced_count_parts.append(unblocked(blocked_counts, 1))
                block_offset = (
                    torch.arange(
                        full_blocks,
                        device=overflow_k.device,
                        dtype=blocked_membership.dtype,
                    )
                    * reduced_block
                ).view(1, full_blocks, 1, 1)
                membership_parts.append(
                    (
                        blocked_membership.reshape(
                            batch, full_blocks, heads, block_size
                        )
                        + block_offset
                    )
                    .permute(0, 2, 1, 3)
                    .reshape(batch, heads, full_len)
                )
                reduced_offset = full_blocks * reduced_block
            if tail_len:
                tail_k = overflow_k[..., -tail_len:, :]
                tail_v = overflow_v[..., -tail_len:, :]
                tail_counts = torch.ones(
                    batch,
                    heads,
                    tail_len,
                    1,
                    dtype=torch.float32,
                    device=overflow_k.device,
                )
                if tail_len > 1:
                    tail_k, tail_v, tail_counts, tail_membership = (
                        self._union_bipartite_round(
                            tail_k,
                            tail_v,
                            tail_counts,
                            (tail_len + 1) // 2,
                            full_blocks,
                            protected_len=0,
                            positional_halves=(
                                self.overflow_bipartite_positional_halves
                            ),
                        )
                    )
                else:
                    tail_membership = torch.zeros(
                        batch,
                        heads,
                        1,
                        dtype=torch.long,
                        device=overflow_k.device,
                    )
                reduced_k_parts.append(tail_k)
                reduced_v_parts.append(tail_v)
                reduced_count_parts.append(tail_counts)
                membership_parts.append(tail_membership + reduced_offset)
            overflow_k = torch.cat(reduced_k_parts, dim=2)
            overflow_v = torch.cat(reduced_v_parts, dim=2)
            overflow_counts = torch.cat(reduced_count_parts, dim=2)
            overflow_membership = torch.cat(membership_parts, dim=2)
        else:
            overflow_counts = torch.ones(
                *overflow_k.shape[:3],
                1,
                dtype=torch.float32,
                device=overflow_k.device,
            )
        target_len = self._desired_state_len(
            ctx_len, available_context, current_state_len
        )
        expanded_k = torch.cat(
            (state_k[..., :current_state_len, :], overflow_k), dim=2
        )
        expanded_v = torch.cat(
            (state_v[..., :current_state_len, :], overflow_v), dim=2
        )
        expanded_counts = torch.cat(
            (
                counts[..., :current_state_len, :],
                overflow_counts,
            ),
            dim=2,
        )
        expanded_len = int(expanded_k.size(2))
        target_len = min(target_len, expanded_len)
        if target_len > state_capacity:
            raise ValueError("union target exceeds state capacity")
        union_assignment = (
            torch.arange(expanded_len, device=overflow_k.device, dtype=torch.long)
            .view(1, 1, expanded_len)
            .expand(*overflow_k.shape[:2], expanded_len)
        )
        round_index = 0
        while int(expanded_k.size(2)) > target_len:
            current_len = int(expanded_k.size(2))
            protected = min(self._protected_state_len(current_len), target_len)
            minimum_next = protected + (current_len - protected) // 2
            next_len = max(target_len, minimum_next)
            (
                expanded_k,
                expanded_v,
                expanded_counts,
                round_assignment,
            ) = self._union_bipartite_round(
                expanded_k,
                expanded_v,
                expanded_counts,
                next_len,
                round_index,
            )
            union_assignment = torch.gather(
                round_assignment, 2, union_assignment
            )
            round_index += 1

        state_k[..., :target_len, :].copy_(expanded_k)
        state_v[..., :target_len, :].copy_(expanded_v)
        counts[..., :target_len, :].copy_(expanded_counts)
        new_owners = union_assignment[..., current_state_len:]
        if overflow_membership is not None:
            new_owners = torch.gather(new_owners, 2, overflow_membership)
        return (
            state_k,
            state_v,
            counts,
            target_len,
            new_owners,
            union_assignment[..., :current_state_len],
        )

    def _update_state_precompact_direct_append(
        self,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        *,
        state_len: int,
        ctx_len: int,
        available_context: int,
        state_capacity: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor | None,
    ] | None:
        """Compact old state first, then append all locally reduced overflow."""

        if not self.overflow_bipartite_merge:
            raise ValueError("state precompaction requires local overflow reduction")
        if self.leaf_attention_backend == "paged":
            raise ValueError("state precompaction currently requires packed leaves")
        overflow_k, overflow_v, overflow_counts, membership = (
            self._reduce_overflow_balanced(overflow_k, overflow_v)
        )
        overflow_len = int(overflow_k.size(2))
        desired_state_len = self._desired_state_len(
            ctx_len, available_context, state_len
        )
        desired_state_len = min(desired_state_len, state_len + overflow_len)
        retained_state_len = desired_state_len - overflow_len
        protected = min(self._protected_state_len(state_len), desired_state_len)
        if desired_state_len > state_capacity:
            raise ValueError("precompacted state target exceeds state capacity")

        old_slot_remap = None
        if retained_state_len < state_len:
            body_len = state_len - protected
            target_body_len = retained_state_len - protected
            block_size = self.overflow_bipartite_block_size
            full_blocks, tail_len = divmod(body_len, block_size)

            def reduced_body_len(ratio: float) -> int:
                full_target = max(
                    block_size // 2, math.ceil(block_size * ratio)
                )
                tail_target = (
                    max((tail_len + 1) // 2, math.ceil(tail_len * ratio))
                    if tail_len
                    else 0
                )
                return full_blocks * full_target + tail_target

            if target_body_len < reduced_body_len(0.5):
                return None
            low, high = 0.5, 1.0
            for _ in range(32):
                middle = (low + high) * 0.5
                if reduced_body_len(middle) <= target_body_len:
                    low = middle
                else:
                    high = middle
            compact_body_k, compact_body_v, compact_body_counts, body_membership = (
                self._reduce_overflow_balanced(
                    state_k[..., protected:state_len, :],
                    state_v[..., protected:state_len, :],
                    counts[..., protected:state_len, :],
                    keep_ratio=low,
                )
            )
            compact_body_len = int(compact_body_k.size(2))
            retained_state_len = protected + compact_body_len
            state_k[..., protected:retained_state_len, :].copy_(compact_body_k)
            state_v[..., protected:retained_state_len, :].copy_(compact_body_v)
            counts[..., protected:retained_state_len, :].copy_(compact_body_counts)
            protected_membership = (
                torch.arange(
                    protected, device=state_k.device, dtype=torch.long
                )
                .view(1, 1, protected)
                .expand(*state_k.shape[:2], protected)
            )
            old_slot_remap = torch.cat(
                (protected_membership, body_membership + protected), dim=2
            )
        elif retained_state_len != state_len:
            return None

        output_state_len = retained_state_len + overflow_len
        state_k[..., retained_state_len:output_state_len, :].copy_(overflow_k)
        state_v[..., retained_state_len:output_state_len, :].copy_(overflow_v)
        counts[..., retained_state_len:output_state_len, :].copy_(overflow_counts)
        reduced_owners = membership + retained_state_len
        return (
            state_k,
            state_v,
            counts,
            output_state_len,
            reduced_owners,
            old_slot_remap,
        )

    def _update_state(
        self,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        *,
        state_len: int,
        ctx_len: int,
        available_context: int,
        state_capacity: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if self.state_union_bipartite:
            return self._update_state_union_bipartite(
                state_k,
                state_v,
                counts,
                overflow_k,
                overflow_v,
                state_len=state_len,
                ctx_len=ctx_len,
                available_context=available_context,
                state_capacity=state_capacity,
            )
        if self.state_precompact_direct_append:
            precompacted = self._update_state_precompact_direct_append(
                state_k,
                state_v,
                counts,
                overflow_k,
                overflow_v,
                state_len=state_len,
                ctx_len=ctx_len,
                available_context=available_context,
                state_capacity=state_capacity,
            )
            if precompacted is not None:
                return precompacted
        original_overflow_len = int(overflow_k.size(2))
        membership = None
        if self.overflow_bipartite_merge:
            overflow_k, overflow_v, overflow_counts, membership = (
                self._reduce_overflow_balanced(overflow_k, overflow_v)
            )
            overflow_select_k = self._mean(overflow_k, overflow_counts)
        else:
            overflow_select_k = overflow_k
            overflow_counts = torch.ones(
                *overflow_k.shape[:3],
                1,
                dtype=torch.float32,
                device=overflow_k.device,
            )
        overflow_len = int(overflow_k.size(2))
        current_state_len = state_len
        desired_state_len = self._desired_state_len(
            ctx_len, available_context, current_state_len
        )
        n_append = min(max(desired_state_len - current_state_len, 0), overflow_len)
        owners = torch.full(
            overflow_k.shape[:-1], -1, dtype=torch.long, device=overflow_k.device
        )
        use_fused_state_update = self.fused_state_update or (
            self.auto_fused_state_update and int(state_k.size(0)) > 1
        )

        buffers = None
        old_route_scores = None
        old_route_indices = None
        if use_fused_state_update and state_k.is_cuda:
            buffers = getattr(self, "_lod_state_update_buffers", None)
            expected_prefix = (
                int(state_k.size(0)),
                int(state_k.size(1)),
            )
            needs_buffers = (
                buffers is None
                or tuple(buffers["touched"].shape[:2]) != expected_prefix
                or int(buffers["touched"].size(2)) < state_capacity
                or buffers["touched"].device != state_k.device
            )
            if needs_buffers:
                buffers = new_state_delta_buffers(state_k, state_v, state_capacity)
                self._lod_state_update_buffers = buffers

        if (
            self.reuse_state_update_similarity
            and self.fused_state_maxsim
            and state_k.is_cuda
        ):
            maxsim_buffers = getattr(self, "_lod_state_maxsim_buffers", None)
            needs_maxsim_buffers = (
                maxsim_buffers is None
                or tuple(maxsim_buffers["route_scores"].shape[:2])
                != tuple(overflow_k.shape[:2])
                or int(maxsim_buffers["route_scores"].size(2)) < overflow_len
                or maxsim_buffers["route_scores"].device != overflow_k.device
            )
            if needs_maxsim_buffers:
                maxsim_buffers = new_state_maxsim_buffers(
                    overflow_select_k,
                    max(
                        overflow_len,
                        self.chunk_len,
                        self.prefill_state_update_len,
                    ),
                )
                self._lod_state_maxsim_buffers = maxsim_buffers
            (
                old_route_scores,
                old_route_indices,
                append_select_scores,
            ) = streaming_state_maxsim(
                overflow_select_k,
                state_k,
                counts,
                maxsim_buffers,
                state_len=current_state_len,
                sink_len=self._protected_state_len(current_state_len),
                block_m=self.state_maxsim_block_m,
                block_n=self.state_maxsim_block_n,
                num_warps=self.state_maxsim_num_warps,
            )
            append_idx, merge_idx = self._split_append_merge_indices(
                append_select_scores, n_append
            )
        elif self.reuse_state_update_similarity and state_k.is_cuda:
            with torch.no_grad():
                old_similarity = torch.matmul(
                    overflow_select_k,
                    self._mean(
                        state_k.detach()[..., :current_state_len, :],
                        counts[..., :current_state_len, :],
                    ).transpose(-1, -2),
                )
                old_similarity.masked_fill_(
                    counts[..., :current_state_len, 0]
                    .le(0.5)
                    .unsqueeze(-2),
                    float("-inf"),
                )
                protected_slots = self._protected_state_len(current_state_len)
                if protected_slots:
                    protected_scores = (
                        old_similarity[..., :protected_slots]
                        .float()
                        .max(dim=-1)
                        .values
                    )
                    old_similarity[..., :protected_slots] = float("-inf")
                else:
                    protected_scores = torch.full_like(
                        old_similarity[..., 0].float(), float("-inf")
                    )
                old_route_scores, old_route_indices = old_similarity.max(dim=-1)
                append_select_scores = torch.maximum(
                    old_route_scores.float(), protected_scores
                )
                append_idx, merge_idx = self._split_append_merge_indices(
                    append_select_scores, n_append
                )
        elif n_append and self.state_append_subblock_size <= 0:
            append_idx, merge_idx = _split_append_merge_idx_by_maxsim(
                overflow_select_k,
                n_append,
                self._mean(
                    state_k.detach()[..., :current_state_len, :],
                    counts[..., :current_state_len, :],
                ),
            )
        elif not n_append:
            merge_idx = _all_idx(overflow_k, overflow_len)
            append_idx = merge_idx[..., :0]
        else:
            with torch.no_grad():
                append_select_scores = torch.matmul(
                    overflow_select_k,
                    self._mean(
                        state_k.detach()[..., :current_state_len, :],
                        counts[..., :current_state_len, :],
                    ).transpose(-1, -2),
                ).max(dim=-1).values
                append_idx, merge_idx = self._split_append_merge_indices(
                    append_select_scores, n_append
                )

        if n_append:
            append_k = _gather_by_idx(overflow_k, append_idx)
            append_v = _gather_by_idx(overflow_v, append_idx)
            append_select_k = _gather_by_idx(overflow_select_k, append_idx)
            append_counts = _gather_by_idx(overflow_counts, append_idx)
            append_slots = (
                torch.arange(
                    current_state_len,
                    current_state_len + n_append,
                    dtype=torch.long,
                    device=overflow_k.device,
                )
                .view(1, 1, n_append)
                .expand_as(append_idx)
            )
            if not self.state_merge_before_append:
                state_k[..., current_state_len:desired_state_len, :].copy_(append_k)
                state_v[..., current_state_len:desired_state_len, :].copy_(append_v)
                counts[..., current_state_len:desired_state_len, :].copy_(
                    append_counts
                )
                owners.scatter_(2, append_idx, append_slots)
            merge_k = _gather_by_idx(overflow_k, merge_idx)
            merge_v = _gather_by_idx(overflow_v, merge_idx)
            merge_select_k = _gather_by_idx(overflow_select_k, merge_idx)
            merge_counts = _gather_by_idx(overflow_counts, merge_idx)
        else:
            merge_k = overflow_k
            merge_v = overflow_v
            merge_select_k = overflow_select_k
            merge_counts = overflow_counts

        if int(merge_k.size(2)) == 0:
            if n_append and self.state_merge_before_append:
                state_k[..., current_state_len:desired_state_len, :].copy_(append_k)
                state_v[..., current_state_len:desired_state_len, :].copy_(append_v)
                counts[..., current_state_len:desired_state_len, :].copy_(
                    append_counts
                )
                owners.scatter_(2, append_idx, append_slots)
            if membership is not None:
                owners = owners.gather(2, membership)
            return state_k, state_v, counts, desired_state_len, owners, None

        with torch.no_grad():
            if old_route_scores is not None and old_route_indices is not None:
                merge_old_scores = old_route_scores.gather(2, merge_idx)
                destination = old_route_indices.gather(2, merge_idx)
                if n_append and not self.state_merge_before_append:
                    appended_logits = torch.matmul(
                        merge_select_k,
                        append_select_k.detach().transpose(-1, -2),
                    )
                    appended_scores, appended_relative = appended_logits.max(dim=-1)
                    appended_destination = appended_relative + current_state_len
                    use_appended = appended_scores > merge_old_scores
                    destination = torch.where(
                        use_appended, appended_destination, destination
                    )
            else:
                route_state_len = (
                    current_state_len
                    if self.state_merge_before_append
                    else desired_state_len
                )
                protected_slots = self._protected_state_len(route_state_len)
                route_logits = torch.matmul(
                    merge_select_k,
                    self._mean(
                        state_k.detach()[..., :route_state_len, :],
                        counts[..., :route_state_len, :],
                    ).transpose(-1, -2),
                )
                route_logits.masked_fill_(
                    counts[..., :route_state_len, 0]
                    .le(0.5)
                    .unsqueeze(-2),
                    float("-inf"),
                )
                route_logits[..., :protected_slots] = float("-inf")
                destination = route_logits.argmax(dim=-1)

        if use_fused_state_update and state_k.is_cuda:
            if buffers is None:
                raise AssertionError("LOD state-update buffers are missing")
            merge_state_in_place(
                state_k,
                state_v,
                counts,
                merge_k.contiguous(),
                merge_v.contiguous(),
                merge_counts.contiguous(),
                merge_idx.contiguous(),
                destination.contiguous(),
                owners,
                buffers,
                active_slots=(
                    current_state_len
                    if self.state_merge_before_append
                    else desired_state_len
                ),
            )
        else:
            route_state_len = (
                current_state_len
                if self.state_merge_before_append
                else desired_state_len
            )
            assignment = F.one_hot(destination, num_classes=route_state_len).to(
                merge_k.dtype
            )
            assignment_t = assignment.transpose(-1, -2)
            state_k[..., :route_state_len, :].add_(
                torch.matmul(assignment_t, merge_k)
            )
            state_v[..., :route_state_len, :].add_(
                torch.matmul(assignment_t.to(merge_v.dtype), merge_v)
            )
            counts[..., :route_state_len, :].add_(
                torch.matmul(assignment_t.float(), merge_counts.float())
            )
            owners.scatter_(2, merge_idx, destination)
        if n_append and self.state_merge_before_append:
            state_k[..., current_state_len:desired_state_len, :].copy_(append_k)
            state_v[..., current_state_len:desired_state_len, :].copy_(append_v)
            counts[..., current_state_len:desired_state_len, :].copy_(append_counts)
            owners.scatter_(2, append_idx, append_slots)
        if membership is not None:
            owners = owners.gather(2, membership)
        return state_k, state_v, counts, desired_state_len, owners, None

    def _state_route_logits(
        self,
        q: torch.Tensor,
        state_k: torch.Tensor,
        counts: torch.Tensor,
        *,
        state_len: int,
    ) -> torch.Tensor:
        mean_k = self._mean(
            state_k.detach()[..., :state_len, :],
            counts[..., :state_len, :],
        )
        if self.route_gqa_matmul:
            batch, query_heads, query_len, head_dim = q.shape
            kv_heads = int(mean_k.size(1))
            grouped_q = q.detach().reshape(
                batch,
                kv_heads,
                self.num_key_value_groups,
                query_len,
                head_dim,
            )
            grouped_k_t = mean_k.transpose(-1, -2).unsqueeze(2)
            return torch.matmul(grouped_q, grouped_k_t).reshape(
                batch, query_heads, query_len, state_len
            )
        return torch.matmul(
            q.detach(), self._repeat_kv(mean_k).transpose(-1, -2)
        )

    def _dynamic_open_target(self, query_len: int) -> float | None:
        return (
            self.dynamic_open_decode_top_p
            if query_len == 1
            else self.dynamic_open_prefill_top_p
        )

    def _dynamic_open_residual(self, query_len: int) -> float | None:
        return (
            self.dynamic_open_decode_residual_mass
            if query_len == 1
            else self.dynamic_open_prefill_residual_mass
        )

    def _dynamic_decode_local_lse(
        self,
        q: torch.Tensor,
        local_k: torch.Tensor,
        *,
        local_len: int | None,
        new_k: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the exact local-field LSE used by full-mass route opening."""
        active_local_len = (
            int(local_k.size(2)) if local_len is None else int(local_len)
        )
        local_scores = torch.matmul(
            q.detach(),
            self._repeat_kv(
                local_k.detach()[..., :active_local_len, :]
            ).transpose(-1, -2),
        ).float() * self.scaling
        local_lse = torch.logsumexp(local_scores, dim=-1)
        if new_k is not None:
            new_score = (
                q.detach()
                * self._repeat_kv(new_k.detach())
            ).sum(dim=-1).float() * self.scaling
            local_lse = torch.logaddexp(local_lse, new_score)
        return local_lse

    def _route_top_slots(
        self,
        q: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        *,
        state_len: int,
        state_capacity: int,
        local_k: torch.Tensor | None = None,
        local_v: torch.Tensor | None = None,
        dynamic_local_lse: torch.Tensor | None = None,
    ) -> torch.Tensor:
        configured_topk = (
            self.prefill_two_level_topk
            if int(q.size(2)) > 1 and self.prefill_two_level_topk is not None
            else self.two_level_topk
        )
        protected_len = (
            self._protected_state_len(state_len)
            if self.exclude_sink_from_routes
            else 0
        )
        route_count = min(configured_topk, state_len - protected_len)
        dynamic_target = self._dynamic_open_target(int(q.size(2)))
        dynamic_residual = self._dynamic_open_residual(int(q.size(2)))
        with torch.no_grad():
            if (
                self.fused_prefill_route_coarse
                and self.reuse_route_logits_for_coarse
                and q.is_cuda
                and int(q.size(2)) > 1
                and 0 < route_count <= 8
                and dynamic_target is None
                and not getattr(self, "_lod_collect_stats", False)
            ):
                if local_k is None or local_v is None:
                    raise AssertionError("fused prefill routing has no local KV")
                logits = self._state_route_logits(
                    q,
                    state_k,
                    counts,
                    state_len=state_len,
                )
                include_local = not self.split_prefill_local_attention
                coarse_local_k = local_k if include_local else local_k[..., :0, :]
                coarse_local_v = local_v if include_local else local_v[..., :0, :]
                routed, coarse_output, coarse_lse = (
                    route_logits_topk_coarse_attention(
                        q,
                        logits.contiguous(),
                        state_v.contiguous(),
                        counts.contiguous(),
                        coarse_local_k.contiguous(),
                        coarse_local_v.contiguous(),
                        state_len=state_len,
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        topk=route_count,
                        protected_len=protected_len,
                        max_leaf_tokens=self.prefill_max_leaf_tokens,
                        residual_local_lse=(
                            dynamic_local_lse.contiguous()
                            if dynamic_local_lse is not None
                            else None
                        ),
                        residual_mass=dynamic_residual,
                        block_m=self.coarse_route_block_m,
                        block_n=self.coarse_route_block_n,
                        num_warps=self.coarse_route_num_warps,
                    )
                )
                if self.collect_dynamic_open_stats and dynamic_residual is not None:
                    open_counts = (routed >= 0).sum(dim=-1)
                    histogram = torch.bincount(
                        open_counts.reshape(-1), minlength=route_count + 1
                    )
                    if not hasattr(self, "_lod_dynamic_prefill_histograms"):
                        self._lod_dynamic_prefill_histograms = []
                    self._lod_dynamic_prefill_histograms.append(histogram)
                self._lod_prefill_fused_coarse = (
                    coarse_output,
                    coarse_lse,
                    include_local,
                )
                return routed
            if (
                self.fused_state_routing
                and q.is_cuda
                and route_count <= 8
                and not getattr(self, "_lod_collect_stats", False)
            ):
                buffers = getattr(self, "_lod_route_buffers", None)
                required_groups = (state_capacity + 63) // 64
                needs_lse_buffers = dynamic_residual is not None
                needs_buffers = (
                    buffers is None
                    or tuple(buffers["output"].shape[:2]) != tuple(q.shape[:2])
                    or int(buffers["output"].size(2)) < int(q.size(2))
                    or int(buffers["partial_scores"].size(3)) < required_groups
                    or (
                        needs_lse_buffers
                        and (
                            tuple(buffers["state_lse"].shape[:2])
                            != tuple(q.shape[:2])
                            or int(buffers["state_lse"].size(2)) < int(q.size(2))
                            or int(buffers["partial_lse"].size(3))
                            < required_groups
                        )
                    )
                    or buffers["output"].device != q.device
                )
                if needs_buffers:
                    buffers = new_route_buffers(
                        q,
                        state_capacity=state_capacity,
                        query_capacity=max(self.chunk_len, self.prefill_chunk_len),
                        include_lse=needs_lse_buffers,
                    )
                    self._lod_route_buffers = buffers
                logits = None
                if (
                    self.direct_fused_state_routing
                    and int(q.size(0)) == 1
                    and dynamic_target is None
                    and dynamic_residual is None
                    and not (
                        self.reuse_route_logits_for_coarse
                        and int(q.size(2)) > 1
                    )
                ):
                    routed = route_top8_state_grouped(
                        q.detach(),
                        state_k.detach(),
                        counts.detach(),
                        buffers,
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        topk=route_count,
                        state_len=state_len,
                        protected_len=protected_len,
                        reorder_like_torch=True,
                    )
                else:
                    logits = self._state_route_logits(
                        q,
                        state_k,
                        counts,
                        state_len=state_len,
                    )
                    if (
                        self.reuse_route_logits_for_coarse
                        and int(q.size(2)) > 1
                    ):
                        self._lod_prefill_route_logits = logits
                    route_result = route_top8_scores_grouped(
                        logits,
                        counts.detach(),
                        buffers,
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        topk=route_count,
                        state_len=state_len,
                        protected_len=protected_len,
                        return_lse=dynamic_residual is not None,
                    )
                    if dynamic_residual is not None:
                        routed, routed_state_lse = route_result
                    else:
                        routed = route_result
                        routed_state_lse = None
                if getattr(self, "_lod_compare_state_routing", False):
                    if logits is None:
                        mean_k = self._repeat_kv(
                            self._mean(
                                state_k.detach()[..., :state_len, :],
                                counts[..., :state_len, :],
                            )
                        )
                        logits = torch.matmul(q.detach(), mean_k.transpose(-1, -2))
                    query_counts = self._repeat_kv(
                        counts.detach()[..., :state_len, :]
                    ).squeeze(-1)
                    corrected = logits * self.scaling + query_counts.log().unsqueeze(2)
                    corrected[..., :protected_len] = float("-inf")
                    reference = corrected.topk(
                        route_count, dim=-1, sorted=False
                    ).indices
                    same_set = (
                        routed.sort(dim=-1).values == reference.sort(dim=-1).values
                    ).all(dim=-1)
                    top9 = corrected.topk(
                        min(route_count + 1, state_len), dim=-1
                    ).values
                    if int(top9.size(-1)) > route_count:
                        boundary_tied = (
                            top9[..., route_count - 1] == top9[..., route_count]
                        )
                    else:
                        boundary_tied = torch.zeros_like(same_set)
                    self._lod_route_compared_rows = getattr(
                        self, "_lod_route_compared_rows", 0
                    ) + int(same_set.numel())
                    self._lod_route_mismatched_rows = getattr(
                        self, "_lod_route_mismatched_rows", 0
                    ) + int((~same_set).sum().item())
                    self._lod_route_boundary_ties = getattr(
                        self, "_lod_route_boundary_ties", 0
                    ) + int(boundary_tied.sum().item())
                if (
                    dynamic_residual is not None
                    and self.fused_prefill_residual_opening
                    and int(q.size(2)) > 1
                    and not self.collect_dynamic_open_stats
                ):
                    if (
                        logits is None
                        or routed_state_lse is None
                        or dynamic_local_lse is None
                    ):
                        raise AssertionError(
                            "fused residual opening is missing routing statistics"
                        )
                    routed = apply_residual_mass_opening(
                        logits,
                        counts.detach(),
                        routed,
                        routed_state_lse,
                        dynamic_local_lse,
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        residual_mass=dynamic_residual,
                    )
                    return routed.clone()
                if dynamic_target is not None or dynamic_residual is not None:
                    if logits is None:
                        raise AssertionError(
                            "dynamic LOD opening requires state routing logits"
                        )
                    query_counts = self._repeat_kv(
                        counts.detach()[..., :state_len, :]
                    ).squeeze(-1)
                    selected_logits = torch.gather(logits, -1, routed)
                    selected_counts = torch.gather(
                        query_counts.unsqueeze(2).expand(
                            -1, -1, int(q.size(2)), -1
                        ),
                        -1,
                        routed,
                    )
                    selected_scores = (
                        selected_logits * self.scaling
                    ).float() + selected_counts.log()
                    full_lse = None
                    if dynamic_residual is not None:
                        if routed_state_lse is None:
                            raise AssertionError("dynamic routing did not return state LSE")
                        state_lse = routed_state_lse
                        if self.dynamic_open_residual_use_state_bound:
                            full_lse = state_lse
                        elif dynamic_local_lse is None:
                            raise AssertionError(
                                "full-mass dynamic opening requires local LSE"
                            )
                        else:
                            full_lse = torch.logaddexp(
                                state_lse,
                                dynamic_local_lse,
                            )
                    routed = self._apply_dynamic_opening(
                        routed,
                        selected_scores,
                        target=dynamic_target,
                        residual_mass=dynamic_residual,
                        full_lse=full_lse,
                    )
                if int(q.size(2)) == 1 and not self.clone_decode_routes:
                    return routed
                return routed.clone()
            logits = self._state_route_logits(
                q,
                state_k,
                counts,
                state_len=state_len,
            )
            if (
                self.reuse_route_logits_for_coarse
                and int(q.size(2)) > 1
                and route_count <= 8
            ):
                self._lod_prefill_route_logits = logits
            logits = logits * self.scaling
            query_counts = self._repeat_kv(counts.detach()[..., :state_len, :]).squeeze(
                -1
            )
            log_counts = query_counts.log()
            logits = logits + log_counts.unsqueeze(2)
            logits[..., :protected_len] = float("-inf")
            top_slots = logits.topk(route_count, dim=-1, sorted=False).indices
            if getattr(self, "_lod_collect_stats", False):
                expanded_counts = query_counts.unsqueeze(2).expand(
                    -1, -1, int(q.size(2)), -1
                )
                selected_counts = torch.gather(expanded_counts, -1, top_slots).sum(-1)
                history_counts = query_counts.sum(-1).clamp_min(1)
                union_leaf_fraction = []
                unique_slot_fraction = []
                for batch_idx in range(int(q.size(0))):
                    for head_idx in range(int(q.size(1))):
                        unique_slots = torch.unique(top_slots[batch_idx, head_idx])
                        union_leaf_fraction.append(
                            query_counts[batch_idx, head_idx]
                            .index_select(0, unique_slots)
                            .sum()
                            / history_counts[batch_idx, head_idx]
                        )
                        unique_slot_fraction.append(
                            unique_slots.new_tensor(
                                float(unique_slots.numel()) / float(state_len),
                                dtype=torch.float32,
                            )
                        )
                self._lod_route_stats.append(
                    {
                        "selected_leaf_count": selected_counts.detach(),
                        "selected_leaf_fraction": (
                            selected_counts / history_counts.unsqueeze(-1)
                        ).detach(),
                        "union_leaf_fraction": torch.stack(
                            union_leaf_fraction
                        ).detach(),
                        "unique_slot_fraction": torch.stack(
                            unique_slot_fraction
                        ).detach(),
                    }
                )
            if dynamic_target is not None or dynamic_residual is not None:
                full_lse = None
                if dynamic_residual is not None:
                    state_lse = torch.logsumexp(logits.float(), dim=-1)
                    if self.dynamic_open_residual_use_state_bound:
                        full_lse = state_lse
                    elif dynamic_local_lse is None:
                        raise AssertionError(
                            "full-mass dynamic opening requires local LSE"
                        )
                    else:
                        full_lse = torch.logaddexp(
                            state_lse,
                            dynamic_local_lse,
                        )
                top_slots = self._apply_dynamic_opening(
                    top_slots,
                    torch.gather(logits.float(), -1, top_slots),
                    target=dynamic_target,
                    residual_mass=dynamic_residual,
                    full_lse=full_lse,
                )
            return top_slots

    def _apply_dynamic_opening(
        self,
        top_slots: torch.Tensor,
        selected_scores: torch.Tensor,
        *,
        target: float | None,
        residual_mass: float | None,
        full_lse: torch.Tensor | None,
    ) -> torch.Tensor:
        """Open a route prefix using conditional or full-field mass."""
        if (target is None) == (residual_mass is None):
            raise ValueError("choose exactly one dynamic LOD mass criterion")
        if target is not None and not 0.0 < target <= 1.0:
            raise ValueError("dynamic LOD top-p must lie in (0, 1]")
        if residual_mass is not None and not 0.0 < residual_mass <= 1.0:
            raise ValueError("dynamic LOD residual mass must lie in (0, 1]")
        route_count = int(top_slots.size(-1))
        if target == 1.0:
            open_counts = torch.full(
                top_slots.shape[:-1],
                route_count,
                dtype=torch.long,
                device=top_slots.device,
            )
            opened_slots = top_slots
        else:
            sorted_scores, order = selected_scores.sort(dim=-1, descending=True)
            rank = torch.arange(route_count, device=top_slots.device)
            if target is not None:
                probabilities = sorted_scores.softmax(dim=-1)
                cumulative = probabilities.cumsum(dim=-1)
                open_counts = (cumulative < target).sum(dim=-1) + 1
                open_counts = open_counts.clamp_max(route_count)
                open_sorted = rank < open_counts.unsqueeze(-1)
            else:
                if full_lse is None or residual_mass is None:
                    raise AssertionError("full-mass dynamic opening has no full LSE")
                global_mass = torch.exp(sorted_scores - full_lse.unsqueeze(-1))
                cumulative_before = global_mass.cumsum(dim=-1) - global_mass
                remaining_before = global_mass.sum(dim=-1, keepdim=True) - (
                    cumulative_before
                )
                open_sorted = (rank == 0) | (
                    remaining_before > residual_mass
                )
                open_counts = open_sorted.sum(dim=-1)
            open_original = torch.zeros_like(open_sorted).scatter_(
                -1, order, open_sorted
            )
            opened_slots = torch.where(
                open_original, top_slots, torch.full_like(top_slots, -1)
            )
        if self.collect_dynamic_open_stats:
            histogram = torch.bincount(
                open_counts.reshape(-1), minlength=route_count + 1
            )
            phase = "decode" if int(top_slots.size(2)) == 1 else "prefill"
            attribute = f"_lod_dynamic_{phase}_histograms"
            if not hasattr(self, attribute):
                setattr(self, attribute, [])
            getattr(self, attribute).append(histogram)
        return opened_slots

    def _new_page_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        owners: torch.Tensor,
        *,
        state_capacity: int,
        sequence_capacity: int,
        virtual_k: torch.Tensor | None = None,
        virtual_v: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        batch, kv_heads, _, head_dim = k.shape
        page_size = self.leaf_page_size
        page_capacity = (
            sequence_capacity + page_size - 1
        ) // page_size + state_capacity
        hash_capacity = 1 << max(
            1,
            (page_capacity * self.leaf_overflow_hash_factor - 1).bit_length(),
        )
        slot_page_dtype = (
            torch.int16
            if page_capacity <= torch.iinfo(torch.int16).max
            else torch.int32
        )
        cache: dict[str, torch.Tensor | int] = {
            "slot_pages": torch.full(
                (
                    batch,
                    kv_heads,
                    state_capacity,
                    self.leaf_inline_pages_per_slot,
                ),
                -1,
                dtype=slot_page_dtype,
                device=k.device,
            ),
            # Most contexts never overflow the compact inline posting lists.
            # Keep only a sentinel allocation until a slot actually needs the
            # hash table; kernels with HASH_PROBES=0 never dereference it.
            "overflow_page_keys": torch.full(
                (batch, kv_heads, 1),
                -1,
                dtype=torch.int32,
                device=k.device,
            ),
            "overflow_page_values": torch.full(
                (batch, kv_heads, 1),
                -1,
                dtype=torch.int32,
                device=k.device,
            ),
            "overflow_hash_capacity": hash_capacity,
            "overflow_flag": torch.zeros((), dtype=torch.int32, device=k.device),
            "overflow_used": torch.zeros((), dtype=torch.int32, device=k.device),
            "overflow_active": False,
            "overflow_safe_until": self.leaf_inline_pages_per_slot * page_size,
            "slot_lengths": torch.zeros(
                batch,
                kv_heads,
                state_capacity,
                dtype=torch.int32,
                device=k.device,
            ),
            "next_page": torch.zeros(
                batch, kv_heads, dtype=torch.int32, device=k.device
            ),
            "page_size": page_size,
            "leaf_capacity": sequence_capacity,
            "leaf_count": 0,
        }
        if self.virtual_page_storage:
            if not self.recursive_page_lod:
                raise ValueError("virtual pages require recursive page LOD")
            virtual_quantized = bool(
                self.leaf_key_quant_bits or self.leaf_value_quant_bits
            )
            if virtual_quantized and (
                self.leaf_key_quant_bits != 4 or self.leaf_value_quant_bits != 4
            ):
                raise ValueError("virtual pages currently require INT4 for both K and V")
            if virtual_k is None or virtual_v is None:
                raise ValueError("virtual pages require the original prompt K/V")
            if virtual_k.shape[:3] != virtual_v.shape[:3]:
                raise ValueError("virtual prompt K/V shapes do not match")
            if virtual_quantized:
                flat_leaf_k = virtual_k.detach()
                flat_leaf_v = virtual_v.detach()
                flat_leaf_capacity = sequence_capacity
            else:
                flat_leaf_k = virtual_k.new_empty(
                    batch, kv_heads, sequence_capacity, head_dim
                )
                flat_leaf_v = virtual_v.new_empty(
                    batch,
                    kv_heads,
                    sequence_capacity,
                    int(virtual_v.size(-1)),
                )
                flat_leaf_k[..., : virtual_k.size(2), :].copy_(virtual_k)
                flat_leaf_v[..., : virtual_v.size(2), :].copy_(virtual_v)
                flat_leaf_capacity = sequence_capacity
            cache.update(
                leaf_k=flat_leaf_k,
                leaf_v=flat_leaf_v,
                page_indices=torch.full(
                    (batch, kv_heads, page_capacity, page_size),
                    -1,
                    dtype=torch.int32,
                    device=k.device,
                ),
                leaf_capacity=flat_leaf_capacity,
                quantization_finalized=False,
            )
            if virtual_quantized:
                group_size = self.leaf_quant_group_size
                value_dim = int(virtual_v.size(-1))
                if head_dim % group_size or value_dim % group_size:
                    raise ValueError("virtual INT4 group size must divide K/V dimensions")
                cache.update(
                    quantized_leaf_k=torch.empty(
                        batch,
                        kv_heads,
                        sequence_capacity,
                        head_dim // 2,
                        dtype=torch.uint8,
                        device=k.device,
                    ),
                    quantized_leaf_v=torch.empty(
                        batch,
                        kv_heads,
                        sequence_capacity,
                        value_dim // 2,
                        dtype=torch.uint8,
                        device=v.device,
                    ),
                    page_k_scales=torch.empty(
                        batch,
                        kv_heads,
                        page_capacity,
                        head_dim // group_size,
                        dtype=k.dtype,
                        device=k.device,
                    ),
                    page_v_scales=torch.empty(
                        batch,
                        kv_heads,
                        page_capacity,
                        value_dim // group_size,
                        dtype=v.dtype,
                        device=v.device,
                    ),
                    page_quantized_counts=torch.zeros(
                        batch,
                        kv_heads,
                        page_capacity,
                        dtype=torch.int32,
                        device=k.device,
                    ),
                )
        else:
            # Zero padding keeps unused lanes deterministic while the last
            # page of each slot is only partially occupied.
            cache.update(
                page_k=torch.zeros(
                    batch,
                    kv_heads,
                    page_capacity,
                    page_size,
                    head_dim,
                    dtype=k.dtype,
                    device=k.device,
                ),
                page_v=torch.zeros(
                    batch,
                    kv_heads,
                    page_capacity,
                    page_size,
                    int(v.size(-1)),
                    dtype=v.dtype,
                    device=v.device,
                ),
            )
        needs_page_summaries = self.recursive_page_lod or bool(
            self.leaf_key_quant_bits or self.leaf_value_quant_bits
        )
        if needs_page_summaries:
            cache.update(
                page_sum_k=torch.zeros(
                    batch,
                    kv_heads,
                    page_capacity,
                    head_dim,
                    dtype=k.dtype,
                    device=k.device,
                ),
                page_sum_v=torch.zeros(
                    batch,
                    kv_heads,
                    page_capacity,
                    int(v.size(-1)),
                    dtype=v.dtype,
                    device=v.device,
                ),
                page_counts=torch.zeros(
                    batch,
                    kv_heads,
                    page_capacity,
                    dtype=torch.int32,
                    device=k.device,
                ),
            )
        if (
            self.leaf_key_quant_bits or self.leaf_value_quant_bits
        ) and not self.virtual_page_storage:
            cache["page_quantized"] = torch.zeros(
                batch,
                kv_heads,
                page_capacity,
                dtype=torch.bool,
                device=k.device,
            )
        self._append_page_cache(cache, k, v, owners)
        return cache

    @staticmethod
    def _grow_page_pool(
        cache: dict[str, torch.Tensor | int], required_pages: int
    ) -> None:
        page_indices = cache.get("page_indices")
        page_k = cache.get("page_k")
        page_v = cache.get("page_v")
        if isinstance(page_indices, torch.Tensor):
            current = int(page_indices.size(2))
        else:
            if not isinstance(page_k, torch.Tensor):
                raise TypeError("page cache K tensor is missing")
            if not isinstance(page_v, torch.Tensor):
                raise TypeError("page cache V tensor is missing")
            current = int(page_k.size(2))
        if required_pages <= current:
            return
        target = max(required_pages, current * 2)
        missing = target - current
        slot_pages = cache.get("slot_pages")
        if (
            isinstance(slot_pages, torch.Tensor)
            and slot_pages.dtype == torch.int16
            and target > torch.iinfo(torch.int16).max
        ):
            cache["slot_pages"] = slot_pages.to(torch.int32)
        if isinstance(page_indices, torch.Tensor):
            cache["page_indices"] = F.pad(
                page_indices, (0, 0, 0, missing), value=-1
            )
        else:
            cache["page_k"] = F.pad(page_k, (0, 0, 0, 0, 0, missing))
            cache["page_v"] = F.pad(page_v, (0, 0, 0, 0, 0, missing))
        page_sum_k = cache.get("page_sum_k")
        page_sum_v = cache.get("page_sum_v")
        page_counts = cache.get("page_counts")
        summaries_finalized = bool(cache.get("summary_quantization_finalized", False))
        if isinstance(page_sum_k, torch.Tensor) and not summaries_finalized:
            cache["page_sum_k"] = F.pad(page_sum_k, (0, 0, 0, missing))
        if isinstance(page_sum_v, torch.Tensor) and not summaries_finalized:
            cache["page_sum_v"] = F.pad(page_sum_v, (0, 0, 0, missing))
        for name in ("quantized_page_sum_k", "quantized_page_sum_v"):
            tensor = cache.get(name)
            if isinstance(tensor, torch.Tensor):
                cache[name] = F.pad(tensor, (0, 0, 0, missing))
        for name in ("page_sum_k_scales", "page_sum_v_scales"):
            tensor = cache.get(name)
            if isinstance(tensor, torch.Tensor):
                cache[name] = F.pad(tensor, (0, 0, 0, missing))
        if isinstance(page_counts, torch.Tensor):
            cache["page_counts"] = F.pad(page_counts, (0, missing))
        page_quantized = cache.get("page_quantized")
        if isinstance(page_quantized, torch.Tensor):
            cache["page_quantized"] = F.pad(page_quantized, (0, missing))
        for name in ("page_k_scales", "page_v_scales"):
            tensor = cache.get(name)
            if isinstance(tensor, torch.Tensor):
                cache[name] = F.pad(tensor, (0, 0, 0, missing))
        page_quantized_counts = cache.get("page_quantized_counts")
        if isinstance(page_quantized_counts, torch.Tensor):
            cache["page_quantized_counts"] = F.pad(
                page_quantized_counts, (0, missing)
            )

    @staticmethod
    def _fake_quantize_page_tensor(
        tensor: torch.Tensor,
        page_sum: torch.Tensor,
        selected: torch.Tensor,
        *,
        bits: int,
        group_size: int,
    ) -> None:
        if not bits:
            return
        pages = tensor[selected].float()
        if not pages.numel():
            return
        page_size = int(pages.size(1))
        dimension = int(pages.size(2))
        padded_dimension = _round_up(dimension, group_size)
        anchors = (page_sum[selected].float() / page_size).unsqueeze(1)
        residuals = pages - anchors
        if padded_dimension != dimension:
            residuals = F.pad(residuals, (0, padded_dimension - dimension))
        groups = padded_dimension // group_size
        grouped = residuals.reshape(-1, page_size, groups, group_size)
        quant_max = (1 << (bits - 1)) - 1
        scales = grouped.abs().amax(dim=(1, 3), keepdim=True) / quant_max
        scales = scales.clamp_min(torch.finfo(torch.float32).tiny)
        restored = (
            (grouped / scales).round().clamp(-quant_max, quant_max) * scales
        ).reshape(-1, page_size, padded_dimension)[..., :dimension]
        tensor[selected] = (restored + anchors).to(tensor.dtype)

    def _fake_quantize_completed_pages(
        self, cache: dict[str, torch.Tensor | int]
    ) -> None:
        page_counts = cache.get("page_counts")
        page_quantized = cache.get("page_quantized")
        page_sum_k = cache.get("page_sum_k")
        page_sum_v = cache.get("page_sum_v")
        page_k = cache.get("page_k")
        page_v = cache.get("page_v")
        tensors = (
            page_counts,
            page_quantized,
            page_sum_k,
            page_sum_v,
            page_k,
            page_v,
        )
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise RuntimeError("quantized page cache metadata is incomplete")
        selected = page_counts.eq(self.leaf_page_size) & ~page_quantized
        self._fake_quantize_page_tensor(
            page_k,
            page_sum_k,
            selected,
            bits=self.leaf_key_quant_bits,
            group_size=self.leaf_quant_group_size,
        )
        self._fake_quantize_page_tensor(
            page_v,
            page_sum_v,
            selected,
            bits=self.leaf_value_quant_bits,
            group_size=self.leaf_quant_group_size,
        )
        page_quantized.logical_or_(selected)

    @staticmethod
    def _grow_slot_page_table(
        cache: dict[str, torch.Tensor | int],
        *,
        required_slots: int,
    ) -> None:
        slot_pages = cache["slot_pages"]
        slot_lengths = cache["slot_lengths"]
        if not isinstance(slot_pages, torch.Tensor):
            raise TypeError("slot page table is missing")
        if not isinstance(slot_lengths, torch.Tensor):
            raise TypeError("slot length tensor is missing")
        missing_slots = max(required_slots - int(slot_pages.size(2)), 0)
        if missing_slots:
            slot_pages = F.pad(slot_pages, (0, 0, 0, missing_slots), value=-1)
            slot_lengths = F.pad(slot_lengths, (0, missing_slots))
        cache["slot_pages"] = slot_pages
        cache["slot_lengths"] = slot_lengths

    @staticmethod
    def _ensure_overflow_page_table(
        cache: dict[str, torch.Tensor | int],
    ) -> None:
        overflow_page_keys = cache.get("overflow_page_keys")
        overflow_page_values = cache.get("overflow_page_values")
        if not isinstance(overflow_page_keys, torch.Tensor) or not isinstance(
            overflow_page_values, torch.Tensor
        ):
            raise TypeError("overflow page table is missing")
        hash_capacity = int(cache["overflow_hash_capacity"])
        if int(overflow_page_keys.size(2)) == hash_capacity:
            return
        shape = (
            int(overflow_page_keys.size(0)),
            int(overflow_page_keys.size(1)),
            hash_capacity,
        )
        cache["overflow_page_keys"] = torch.full(
            shape,
            -1,
            dtype=torch.int32,
            device=overflow_page_keys.device,
        )
        cache["overflow_page_values"] = torch.full(
            shape,
            -1,
            dtype=torch.int32,
            device=overflow_page_values.device,
        )

    def _append_page_cache(
        self,
        cache: dict[str, torch.Tensor | int],
        k: torch.Tensor,
        v: torch.Tensor,
        owners: torch.Tensor,
    ) -> None:
        append_len = int(owners.size(2))
        if append_len == 0:
            return
        page_size = int(cache["page_size"])
        slot_lengths = cache["slot_lengths"]
        next_page = cache["next_page"]
        if not isinstance(slot_lengths, torch.Tensor):
            raise TypeError("slot length tensor is missing")
        if not isinstance(next_page, torch.Tensor):
            raise TypeError("next-page tensor is missing")
        leaf_offset = int(cache["leaf_count"])
        leaf_count = leaf_offset + append_len
        leaf_capacity = int(cache["leaf_capacity"])
        if leaf_count > leaf_capacity:
            leaf_capacity = max(leaf_count, leaf_capacity * 2)
            required_slots = max(
                int(slot_lengths.size(2)), int(owners.max().item()) + 1
            )
            self._grow_slot_page_table(
                cache,
                required_slots=required_slots,
            )
            required_pages = required_slots + (
                leaf_capacity + page_size - 1
            ) // page_size
            self._grow_page_pool(cache, required_pages)
            if isinstance(cache.get("page_indices"), torch.Tensor):
                quantized_leaf_k = cache.get("quantized_leaf_k")
                quantized_leaf_v = cache.get("quantized_leaf_v")
                if isinstance(quantized_leaf_k, torch.Tensor) and isinstance(
                    quantized_leaf_v, torch.Tensor
                ):
                    missing = leaf_capacity - int(quantized_leaf_k.size(2))
                    cache["quantized_leaf_k"] = F.pad(
                        quantized_leaf_k, (0, 0, 0, missing)
                    )
                    cache["quantized_leaf_v"] = F.pad(
                        quantized_leaf_v, (0, 0, 0, missing)
                    )
                else:
                    leaf_k = cache.get("leaf_k")
                    leaf_v = cache.get("leaf_v")
                    if not isinstance(leaf_k, torch.Tensor) or not isinstance(
                        leaf_v, torch.Tensor
                    ):
                        raise RuntimeError("virtual page backing K/V are missing")
                    missing = leaf_capacity - int(leaf_k.size(2))
                    cache["leaf_k"] = F.pad(leaf_k, (0, 0, 0, missing))
                    cache["leaf_v"] = F.pad(leaf_v, (0, 0, 0, missing))
            cache["leaf_capacity"] = leaf_capacity
            slot_lengths = cache["slot_lengths"]
            if not isinstance(slot_lengths, torch.Tensor):
                raise TypeError("slot length tensor is missing")
        cache["leaf_count"] = leaf_count
        slot_pages = cache["slot_pages"]
        next_page = cache["next_page"]
        if not isinstance(slot_pages, torch.Tensor):
            raise TypeError("slot page table is missing")
        if not isinstance(next_page, torch.Tensor):
            raise TypeError("next-page tensor is missing")
        page_k = cache.get("page_k")
        page_v = cache.get("page_v")
        page_indices = cache.get("page_indices")
        overflow_used = cache["overflow_used"]
        overflow_flag = cache["overflow_flag"]
        virtual_pages = isinstance(page_indices, torch.Tensor)
        if not virtual_pages and not isinstance(page_k, torch.Tensor):
            raise TypeError("page cache K tensor is missing")
        if not virtual_pages and not isinstance(page_v, torch.Tensor):
            raise TypeError("page cache V tensor is missing")
        if not isinstance(overflow_used, torch.Tensor):
            raise TypeError("overflow page-used flag is missing")
        if not isinstance(overflow_flag, torch.Tensor):
            raise TypeError("overflow page-table flag is missing")
        append_hash_probes = 0
        if bool(cache["overflow_active"]):
            append_hash_probes = self.leaf_hash_probes
        elif leaf_count > int(cache["overflow_safe_until"]):
            inline_token_capacity = self.leaf_inline_pages_per_slot * page_size
            additions = torch.zeros_like(slot_lengths).scatter_add_(
                2,
                owners.long(),
                torch.ones_like(owners, dtype=slot_lengths.dtype),
            )
            projected_max = int((slot_lengths + additions).max().item())
            cache["overflow_active"] = projected_max > inline_token_capacity
            if bool(cache["overflow_active"]):
                append_hash_probes = self.leaf_hash_probes
            else:
                cache["overflow_safe_until"] = leaf_count + (
                    inline_token_capacity - projected_max
                )
        if append_hash_probes:
            self._ensure_overflow_page_table(cache)
        overflow_page_keys = cache["overflow_page_keys"]
        overflow_page_values = cache["overflow_page_values"]
        if not isinstance(overflow_page_keys, torch.Tensor):
            raise TypeError("overflow page-key tensor is missing")
        if not isinstance(overflow_page_values, torch.Tensor):
            raise TypeError("overflow page-value tensor is missing")
        page_sum_k = cache.get("page_sum_k")
        page_sum_v = cache.get("page_sum_v")
        page_counts = cache.get("page_counts")
        if virtual_pages:
            leaf_k = cache.get("leaf_k")
            leaf_v = cache.get("leaf_v")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (leaf_k, leaf_v, page_sum_k, page_sum_v, page_counts)
            ):
                raise TypeError("virtual page cache tensors are incomplete")
            if bool(cache.get("quantization_finalized", False)):
                quantized_names = (
                    "quantized_leaf_k",
                    "quantized_leaf_v",
                    "page_k_scales",
                    "page_v_scales",
                    "page_quantized_counts",
                )
                quantized_tensors = tuple(cache.get(name) for name in quantized_names)
                if not all(
                    isinstance(value, torch.Tensor) for value in quantized_tensors
                ):
                    raise RuntimeError("finalized virtual INT4 cache is incomplete")
                append_quantized_virtual_paged_kv(
                    k,
                    v,
                    leaf_offset,
                    owners.contiguous(),
                    page_indices,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    overflow_flag,
                    slot_lengths,
                    next_page,
                    page_sum_k,
                    page_sum_v,
                    page_counts,
                    *quantized_tensors,
                    hash_probes=append_hash_probes,
                    quant_group_size=self.leaf_quant_group_size,
                    quantized_page_sum_k=cache.get("quantized_page_sum_k"),
                    quantized_page_sum_v=cache.get("quantized_page_sum_v"),
                    page_sum_k_scales=cache.get("page_sum_k_scales"),
                    page_sum_v_scales=cache.get("page_sum_v_scales"),
                    optimize_summary_scale=(
                        self.page_summary_scale_mode == "l2"
                    ),
                    optimize_leaf_scale=(
                        self.leaf_append_quant_scale_mode == "l2"
                    ),
                )
            else:
                leaf_k[..., leaf_offset:leaf_count, :].copy_(k)
                leaf_v[..., leaf_offset:leaf_count, :].copy_(v)
                append_virtual_paged_kv(
                    leaf_k,
                    leaf_v,
                    leaf_offset,
                    owners.contiguous(),
                    page_indices,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    overflow_flag,
                    slot_lengths,
                    next_page,
                    page_sum_k,
                    page_sum_v,
                    page_counts,
                    hash_probes=append_hash_probes,
                    quantized_leaf_k=cache.get("quantized_leaf_k"),
                    quantized_leaf_v=cache.get("quantized_leaf_v"),
                    page_k_scales=cache.get("page_k_scales"),
                    page_v_scales=cache.get("page_v_scales"),
                    page_quantized_counts=cache.get("page_quantized_counts"),
                    quant_group_size=self.leaf_quant_group_size,
                    quantize_touched=False,
                    optimize_scale=(self.leaf_quant_scale_mode == "l2"),
                )
        else:
            append_paged_kv(
                k,
                v,
                owners.contiguous(),
                page_k,
                page_v,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                overflow_flag,
                slot_lengths,
                next_page,
                hash_probes=append_hash_probes,
                page_sum_k=(page_sum_k if isinstance(page_sum_k, torch.Tensor) else None),
                page_sum_v=(page_sum_v if isinstance(page_sum_v, torch.Tensor) else None),
                page_counts=(page_counts if isinstance(page_counts, torch.Tensor) else None),
            )
        if (
            self.leaf_key_quant_bits or self.leaf_value_quant_bits
        ) and not virtual_pages:
            self._fake_quantize_completed_pages(cache)
    def _paged_leaf_attention(
        self,
        q: torch.Tensor,
        top_slots: torch.Tensor,
        cache: dict[str, torch.Tensor | int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_pages = cache["slot_pages"]
        slot_lengths = cache["slot_lengths"]
        page_k = cache["page_k"]
        page_v = cache["page_v"]
        overflow_page_keys = cache["overflow_page_keys"]
        overflow_page_values = cache["overflow_page_values"]
        overflow_used = cache["overflow_used"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                page_k,
                page_v,
            )
        ):
            raise TypeError("paged LOD cache is incomplete")
        leaf_function = (
            query_major_paged_leaf_attention
            if self.leaf_layout == "query"
            else paged_leaf_attention
        )
        leaf_kwargs = {}
        if self.leaf_layout == "query":
            leaf_count = int(cache["leaf_count"])
            block_n = (
                self.leaf_short_block_n
                if leaf_count <= self.leaf_short_context
                else self.leaf_block_n
            )
            leaf_kwargs = {
                "block_n": block_n,
                "hash_probes": (
                    self.leaf_hash_probes if cache["overflow_active"] else 0
                ),
                "num_warps": self.leaf_num_warps,
                "waves_per_eu": self.leaf_waves_per_eu,
                "timing_events": getattr(self, "_lod_leaf_timing_events", None),
            }
        elif self.leaf_layout == "expert":
            leaf_kwargs = {
                "block_m": self.leaf_block_m,
                "block_n": self.leaf_block_n,
                "hash_probes": (
                    self.leaf_hash_probes if cache["overflow_active"] else 0
                ),
                "num_warps": self.leaf_num_warps,
                "waves_per_eu": self.leaf_waves_per_eu,
                "timing_events": getattr(self, "_lod_leaf_timing_events", None),
            }
        else:
            raise ValueError(f"unknown leaf attention layout {self.leaf_layout!r}")
        return leaf_function(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            kv_group_size=self.num_key_value_groups,
            scale=self.scaling,
            **leaf_kwargs,
        )

    def _coarse_attention(
        self,
        q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        top_slots: torch.Tensor,
        *,
        state_len: int,
        state_capacity: int,
        include_local: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_len = int(q.size(2))
        fused_prefill = getattr(self, "_lod_prefill_fused_coarse", None)
        if fused_prefill is not None:
            del self._lod_prefill_fused_coarse
            coarse_output, coarse_lse, fused_includes_local = fused_prefill
            if include_local != fused_includes_local:
                raise AssertionError("fused prefill local-branch mode drifted")
            if tuple(coarse_output.shape) != tuple(q.shape):
                raise AssertionError("fused prefill coarse output shape drifted")
            return coarse_output, coarse_lse
        route_logits = getattr(self, "_lod_prefill_route_logits", None)
        if route_logits is not None:
            del self._lod_prefill_route_logits
            if not self.reuse_route_logits_for_coarse:
                raise AssertionError("stale LOD prefill route logits")
            if tuple(route_logits.shape) != (
                int(q.size(0)),
                int(q.size(1)),
                query_len,
                state_len,
            ):
                raise AssertionError("LOD prefill route-logit shape drifted")
            coarse_local_k = (
                local_k if include_local else local_k[..., :0, :].contiguous()
            )
            coarse_local_v = (
                local_v if include_local else local_v[..., :0, :].contiguous()
            )
            return route_logits_coarse_attention(
                q.contiguous(),
                route_logits.contiguous(),
                state_v.contiguous(),
                counts.contiguous(),
                coarse_local_k.contiguous(),
                coarse_local_v.contiguous(),
                top_slots.contiguous(),
                state_len=state_len,
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                block_m=self.coarse_route_block_m,
                block_n=self.coarse_route_block_n,
                num_warps=self.coarse_route_num_warps,
            )
        route_logits = self._state_route_logits(
            q,
            state_k,
            counts,
            state_len=state_len,
        )
        coarse_output, coarse_lse = route_logits_coarse_attention(
            q.contiguous(),
            route_logits.contiguous(),
            state_v.contiguous(),
            counts.contiguous(),
            local_k[..., :0, :].contiguous(),
            local_v[..., :0, :].contiguous(),
            top_slots.contiguous(),
            state_len=state_len,
            kv_group_size=self.num_key_value_groups,
            scale=self.scaling,
            block_m=self.coarse_route_block_m,
            block_n=self.coarse_route_block_n,
            num_warps=self.coarse_route_num_warps,
        )
        if not include_local or int(local_k.size(2)) == 0:
            return coarse_output, coarse_lse
        local_output, local_lse, *_ = (
            torch.ops.aten._scaled_dot_product_flash_attention.default(
                q.contiguous(),
                local_k.contiguous(),
                local_v.contiguous(),
                0.0,
                True,
                False,
                scale=self.scaling,
            )
        )
        return (
            _merge_lse_branches(
                coarse_output,
                coarse_lse,
                local_output,
                local_lse,
            ),
            torch.logaddexp(coarse_lse, local_lse),
        )

    def _two_level_attention(
        self,
        q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        owners: torch.Tensor | None,
        exact_k: torch.Tensor,
        exact_v: torch.Tensor,
        *,
        state_len: int,
        state_capacity: int,
        page_cache: dict[str, torch.Tensor | int] | None = None,
        local_len: int | None = None,
        new_k: torch.Tensor | None = None,
        new_v: torch.Tensor | None = None,
        local_branch: tuple[torch.Tensor, torch.Tensor] | None = None,
        sink_k: torch.Tensor | None = None,
        sink_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = q.contiguous()
        if (sink_k is None) != (sink_v is None):
            raise ValueError("separate sink K and V must be provided together")
        configured_topk = (
            self.prefill_two_level_topk
            if int(q.size(2)) > 1 and self.prefill_two_level_topk is not None
            else self.two_level_topk
        )
        dynamic_target = self._dynamic_open_target(int(q.size(2)))
        dynamic_residual = self._dynamic_open_residual(int(q.size(2)))
        if dynamic_target is not None and dynamic_residual is not None:
            raise ValueError(
                "decode top-p and full-mass residual opening are mutually exclusive"
            )
        if configured_topk == 0:
            if dynamic_target is not None or dynamic_residual is not None:
                raise ValueError("dynamic LOD opening requires at least one leaf route")
            coarse_local_k = local_k
            coarse_local_v = local_v
            if new_k is not None or new_v is not None:
                if new_k is None or new_v is None or local_len is None:
                    raise ValueError("buffered low-LOD decode requires current K/V")
                local_k[..., local_len : local_len + 1, :].copy_(new_k)
                local_v[..., local_len : local_len + 1, :].copy_(new_v)
                coarse_local_k = local_k[..., : local_len + 1, :].contiguous()
                coarse_local_v = local_v[..., : local_len + 1, :].contiguous()
            no_slots = torch.empty(
                *q.shape[:3], 0, dtype=torch.long, device=q.device
            )
            coarse_output, coarse_lse = self._coarse_attention(
                q,
                coarse_local_k,
                coarse_local_v,
                state_k,
                state_v,
                counts,
                no_slots,
                state_len=state_len,
                state_capacity=state_capacity,
                include_local=True,
            )
            if sink_k is not None and sink_v is not None:
                return merge_attention_branches_with_sink(
                    q,
                    sink_k,
                    sink_v,
                    coarse_output,
                    coarse_lse,
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                )
            return coarse_output
        dynamic_active = dynamic_target is not None or dynamic_residual is not None
        if dynamic_active:
            if self.leaf_attention_backend != "paged" or self.leaf_layout != "query":
                raise ValueError(
                    "dynamic LOD opening requires query-major paged leaf attention"
                )
            if self.recursive_page_lod and int(q.size(2)) > 1:
                raise ValueError(
                    "dynamic LOD opening is not implemented for recursive page prefill"
                )
        indexed_recursive_decode = bool(
            page_cache is not None
            and self.recursive_page_lod
            and isinstance(page_cache.get("page_indices"), torch.Tensor)
        )
        fuse_decode_route = (
            self.fused_decode_attention
            and self.fused_decode_state_route
            and int(q.size(2)) == 1
            and self.leaf_attention_backend == "paged"
            and (
                indexed_recursive_decode
                or not (
                    page_cache is not None
                    and isinstance(
                        page_cache.get("page_indices"), torch.Tensor
                    )
                )
            )
            and self.two_level_topk <= 8
            and (
                not dynamic_active
                or not self.collect_dynamic_open_stats
            )
        )
        top_slots = None
        if not fuse_decode_route:
            dynamic_local_lse = None
            if (
                dynamic_residual is not None
                and not self.dynamic_open_residual_use_state_bound
            ):
                if int(q.size(2)) > 1:
                    if local_branch is None:
                        raise ValueError(
                            "full-mass prefill opening requires split local attention"
                        )
                    dynamic_local_lse = local_branch[1]
                else:
                    dynamic_local_lse = self._dynamic_decode_local_lse(
                        q,
                        local_k,
                        local_len=local_len,
                        new_k=new_k,
                    )
            top_slots = self._route_top_slots(
                q,
                state_k,
                state_v,
                counts,
                state_len=state_len,
                state_capacity=state_capacity,
                local_k=local_k,
                local_v=local_v,
                dynamic_local_lse=dynamic_local_lse,
            )
            if getattr(self, "_lod_padding_state_reserve", 0):
                query_counts = self._repeat_kv(
                    counts[..., :state_len, :]
                ).squeeze(-1)
                safe_slots = top_slots.clamp_min(0)
                selected_counts = torch.gather(
                    query_counts.unsqueeze(2).expand(
                        -1, -1, int(q.size(2)), -1
                    ),
                    -1,
                    safe_slots,
                )
                top_slots = torch.where(
                    top_slots.ge(0) & selected_counts.gt(0.5),
                    top_slots,
                    torch.full_like(top_slots, -1),
                )
        if (
            self.fused_decode_attention
            and int(q.size(2)) == 1
            and self.leaf_attention_backend == "paged"
            and (
                indexed_recursive_decode
                or not (
                    page_cache is not None
                    and isinstance(
                        page_cache.get("page_indices"), torch.Tensor
                    )
                )
            )
            and (sink_k is None or fuse_decode_route)
        ):
            if page_cache is None:
                raise RuntimeError("paged LOD attention has no leaf page cache")
            page_k = page_cache[
                "leaf_k" if indexed_recursive_decode else "page_k"
            ]
            page_v = page_cache[
                "leaf_v" if indexed_recursive_decode else "page_v"
            ]
            slot_pages = page_cache["slot_pages"]
            overflow_page_keys = page_cache["overflow_page_keys"]
            overflow_page_values = page_cache["overflow_page_values"]
            overflow_used = page_cache["overflow_used"]
            slot_lengths = page_cache["slot_lengths"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    page_k,
                    page_v,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    slot_lengths,
                )
            ):
                raise TypeError("paged LOD cache is incomplete")
            decode_buffers = getattr(self, "_lod_decode_attention_buffers", None)
            expected_partial = (
                int(q.size(0)),
                int(q.size(1)),
                self.decode_split_kv,
                int(q.size(-1)),
            )
            if (
                self.decode_split_kv > 1
                and (
                    decode_buffers is None
                    or tuple(decode_buffers["partial_out"].shape)
                    != expected_partial
                    or decode_buffers["partial_out"].device != q.device
                    or (
                        fuse_decode_route
                        and (
                            "route_group_lse" not in decode_buffers
                            or int(decode_buffers["route_group_lse"].size(2))
                            < math.ceil(
                                state_capacity / self.decode_route_group_size
                            )
                        )
                    )
                )
            ):
                decode_buffers = new_fused_decode_buffers(
                    q,
                    splits=self.decode_split_kv,
                    state_capacity=(state_capacity if fuse_decode_route else None),
                    route_group_size=self.decode_route_group_size,
                )
                self._lod_decode_attention_buffers = decode_buffers
            return fused_decode_paged_lod_attention(
                q,
                state_k,
                state_v,
                counts,
                local_k,
                local_v,
                page_k,
                page_v,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                top_slots,
                state_len=state_len,
                local_len=local_len,
                new_k=new_k,
                new_v=new_v,
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                hash_probes=(
                    self.leaf_hash_probes
                    if page_cache["overflow_active"]
                    else 0
                ),
                block_n=self.decode_block_n,
                num_warps=self.decode_num_warps,
                waves_per_eu=self.leaf_waves_per_eu,
                split_kv=self.decode_split_kv,
                buffers=decode_buffers,
                use_dot=self.decode_use_dot,
                fuse_state_route=fuse_decode_route,
                route_group_size=self.decode_route_group_size,
                route_num_warps=self.decode_route_num_warps,
                route_reduce_num_warps=self.decode_route_reduce_num_warps,
                final_reduce_num_warps=self.decode_final_reduce_num_warps,
                fuse_final_reduce=self.decode_fuse_final_reduce,
                route_use_dot=self.decode_route_use_dot,
                route_gqa_grouped=self.decode_route_gqa_grouped,
                protected_len=(
                    self._protected_state_len(state_len)
                    if self.exclude_sink_from_routes
                    else 0
                ),
                sink_k=sink_k,
                sink_v=sink_v,
                route_top_p=(dynamic_target if fuse_decode_route else None),
                route_residual_mass=(
                    dynamic_residual if fuse_decode_route else None
                ),
                reuse_residual_local_attention=(
                    self.reuse_dynamic_local_attention and fuse_decode_route
                ),
                route_residual_use_state_bound=(
                    self.dynamic_open_residual_use_state_bound
                    and fuse_decode_route
                ),
                timing_events=getattr(self, "_lod_decode_timing_events", None),
                recursive_page_cache=(
                    page_cache if indexed_recursive_decode else None
                ),
                recursive_quant_group_size=self.leaf_quant_group_size,
            )
        if top_slots is None:
            raise AssertionError("LOD routing did not produce slots")
        if self.leaf_attention_backend == "paged":
            if page_cache is None:
                raise RuntimeError("paged LOD attention has no leaf page cache")
            if self.recursive_page_lod:
                summary_names = ("page_sum_k", "page_sum_v", "page_counts")
                if not all(
                    isinstance(page_cache.get(name), torch.Tensor)
                    for name in summary_names
                ):
                    raise RuntimeError("recursive LOD page summaries are missing")
                residual_page_function = (
                    query_major_indexed_residual_page_attention
                    if isinstance(page_cache.get("page_indices"), torch.Tensor)
                    else query_major_residual_page_attention
                )
                page_storage_args = (
                    (
                        page_cache["leaf_k"],
                        page_cache["leaf_v"],
                        page_cache["page_indices"],
                    )
                    if residual_page_function
                    is query_major_indexed_residual_page_attention
                    else (page_cache["page_k"], page_cache["page_v"])
                )
                quantized_attention = bool(
                    page_cache.get("quantization_finalized", False)
                )
                quantized_summaries = bool(
                    page_cache.get("summary_quantization_finalized", False)
                )
                exact_output, exact_lse = residual_page_function(
                    q,
                    state_k,
                    state_v,
                    counts,
                    *page_storage_args,
                    page_cache["page_sum_k"],
                    page_cache["page_sum_v"],
                    page_cache["page_counts"],
                    page_cache["slot_pages"],
                    page_cache["overflow_page_keys"],
                    page_cache["overflow_page_values"],
                    page_cache["overflow_used"],
                    page_cache["slot_lengths"],
                    top_slots,
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                    hash_probes=(
                        self.leaf_hash_probes
                        if page_cache["overflow_active"]
                        else 0
                    ),
                    page_block_n=self.recursive_page_block_n,
                    num_warps=self.leaf_num_warps,
                    waves_per_eu=self.leaf_waves_per_eu,
                    timing_events=getattr(
                        self, "_lod_leaf_timing_events", None
                    ),
                    quantized_leaf_k=(
                        page_cache.get("quantized_leaf_k")
                        if quantized_attention
                        else None
                    ),
                    quantized_leaf_v=(
                        page_cache.get("quantized_leaf_v")
                        if quantized_attention
                        else None
                    ),
                    page_k_scales=(
                        page_cache.get("page_k_scales")
                        if quantized_attention
                        else None
                    ),
                    page_v_scales=(
                        page_cache.get("page_v_scales")
                        if quantized_attention
                        else None
                    ),
                    page_quantized_counts=(
                        page_cache.get("page_quantized_counts")
                        if quantized_attention
                        else None
                    ),
                    quantized_page_sum_k=(
                        page_cache.get("quantized_page_sum_k")
                        if quantized_summaries
                        else None
                    ),
                    quantized_page_sum_v=(
                        page_cache.get("quantized_page_sum_v")
                        if quantized_summaries
                        else None
                    ),
                    page_sum_k_scales=(
                        page_cache.get("page_sum_k_scales")
                        if quantized_summaries
                        else None
                    ),
                    page_sum_v_scales=(
                        page_cache.get("page_sum_v_scales")
                        if quantized_summaries
                        else None
                    ),
                    quant_group_size=self.leaf_quant_group_size,
                )
            else:
                exact_output, exact_lse = self._paged_leaf_attention(
                    q, top_slots, page_cache
                )
        elif self.leaf_attention_backend == "packed":
            if owners is None:
                raise RuntimeError("packed LOD attention has no leaf owners")
            from .kvm_two_level_mixer import _expert_leaf_attention

            exact_output, exact_lse = _expert_leaf_attention(
                q,
                exact_k,
                exact_v,
                owners,
                counts,
                top_slots,
                kv_group_size=self.num_key_value_groups,
                head_temperature=q.new_ones(self.config.num_attention_heads),
                scale=self.scaling,
            )
        else:
            raise ValueError(
                f"unknown LOD leaf backend {self.leaf_attention_backend!r}"
            )
        if top_slots is not None:
            has_exact = top_slots.ge(0).any(dim=-1)
            exact_output = torch.where(
                has_exact.unsqueeze(-1),
                exact_output,
                torch.zeros_like(exact_output),
            )
            exact_lse = torch.where(
                has_exact,
                exact_lse,
                torch.full_like(exact_lse, float("-inf")),
            )
        coarse_output, coarse_lse = self._coarse_attention(
            q,
            local_k,
            local_v,
            state_k,
            state_v,
            counts,
            top_slots,
            state_len=state_len,
            state_capacity=state_capacity,
            include_local=local_branch is None,
        )
        if local_branch is not None:
            local_output, local_lse = local_branch
            if sink_k is not None and sink_v is not None:
                return merge_attention_branches_with_sink(
                    q,
                    sink_k,
                    sink_v,
                    coarse_output,
                    coarse_lse,
                    exact_output,
                    exact_lse,
                    local_output,
                    local_lse,
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                )
            return merge_attention_branches(
                coarse_output,
                coarse_lse,
                exact_output,
                exact_lse,
                local_output,
                local_lse,
            )
        output = merge_attention_branches(
            coarse_output, coarse_lse, exact_output, exact_lse
        )
        if sink_k is None or sink_v is None:
            return output
        return merge_attention_branches_with_sink(
            q,
            sink_k,
            sink_v,
            coarse_output,
            coarse_lse,
            exact_output,
            exact_lse,
            kv_group_size=self.num_key_value_groups,
            scale=self.scaling,
        )

    def _exact_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        valid_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_starts is not None:
            query_len = int(q.size(2))
            key_len = int(k.size(2))
            query_position = torch.arange(query_len, device=q.device)
            key_position = torch.arange(key_len, device=q.device)
            attention_mask = key_position.view(1, 1, 1, key_len) >= (
                valid_starts.view(-1, 1, 1, 1)
            )
            if causal:
                attention_mask = attention_mask & (
                    key_position.view(1, 1, 1, key_len)
                    <= query_position.view(1, 1, query_len, 1)
                )
            output = F.scaled_dot_product_attention(
                q,
                self._repeat_kv(k),
                self._repeat_kv(v),
                attn_mask=attention_mask,
                is_causal=False,
                scale=self.scaling,
            )
            query_valid = query_position.view(1, 1, query_len, 1) >= (
                valid_starts.view(-1, 1, 1, 1)
            )
            return torch.where(query_valid, output, torch.zeros_like(output))
        return F.scaled_dot_product_attention(
            q,
            self._repeat_kv(k),
            self._repeat_kv(v),
            is_causal=causal,
            scale=self.scaling,
        )

    def _prefill_local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        query_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            0.0,
            True,
            False,
            scale=self.scaling,
        )
        return output[..., query_offset:, :], lse[..., query_offset:]

    @torch.compiler.disable
    def _prefill_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        logical_prefill_len: int | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, attention_len, _ = q.shape
        prefill_chunk_len = self.prefill_chunk_len
        prefill_local_len = self.prefill_local_len
        prefill_state_update_len = self.prefill_state_update_len
        if prefill_chunk_len > prefill_local_len:
            raise ValueError("prefill chunk length cannot exceed its local field")
        if prefill_state_update_len <= 0:
            raise ValueError("prefill state update length must be positive")
        prefill_len = (
            attention_len
            if logical_prefill_len is None
            else int(logical_prefill_len)
        )
        if prefill_len <= 0 or prefill_len > attention_len:
            raise ValueError("logical prefill length must fit the attention field")
        if attention_len - prefill_len >= prefill_chunk_len:
            raise ValueError("prefill padding must be confined to the final chunk")
        if prefill_valid_starts is not None:
            prefill_valid_starts = prefill_valid_starts.to(
                device=q.device, dtype=torch.long
            )
            if tuple(prefill_valid_starts.shape) != (batch_size,):
                raise ValueError("prefill valid starts must have one entry per row")
            if bool(
                (
                    prefill_valid_starts.lt(0)
                    | prefill_valid_starts.ge(self.chunk_len)
                ).any().item()
            ):
                raise ValueError(
                    "chunk-aligned padding must fit entirely in the first chunk"
                )
            if self.separate_sink_cache:
                raise NotImplementedError(
                    "chunk-aligned padding does not yet support a separate sink cache"
                )
            self._lod_padding_state_reserve = int(
                prefill_valid_starts.max().item()
            )
        else:
            self._lod_padding_state_reserve = 0
        exact_lookback = prefill_local_len - prefill_chunk_len
        if getattr(self, "_lod_collect_stats", False):
            self._lod_route_stats = []
        front_len = min(attention_len, exact_lookback + self.chunk_len)
        outputs = [
            self._exact_attention(
                q[..., :front_len, :],
                k[..., :front_len, :],
                v[..., :front_len, :],
                causal=True,
                valid_starts=prefill_valid_starts,
            )
        ]

        initial_len = min(prefill_len, self.chunk_len)
        separated_sink_len = (
            min(self.sink_len, initial_len) if self.separate_sink_cache else 0
        )
        sink_k = (
            k[..., :separated_sink_len, :].detach().contiguous()
            if separated_sink_len
            else None
        )
        sink_v = (
            v[..., :separated_sink_len, :].detach().contiguous()
            if separated_sink_len
            else None
        )
        archive_k = k[..., separated_sink_len:, :]
        archive_v = v[..., separated_sink_len:, :]
        initial_state_len = initial_len - separated_sink_len
        state_capacity = self._state_capacity(prefill_len, initial_state_len)
        if prefill_valid_starts is None:
            initial_state_k = archive_k[..., :initial_state_len, :]
            initial_state_v = archive_v[..., :initial_state_len, :]
            initial_valid = None
            owners = (
                torch.arange(
                    initial_state_len, dtype=torch.long, device=k.device
                )
                .view(1, 1, initial_state_len)
                .expand(
                    batch_size,
                    self.config.num_key_value_heads,
                    initial_state_len,
                )
            )
        else:
            slot = torch.arange(initial_state_len, device=k.device)
            valid_count = initial_state_len - prefill_valid_starts
            initial_valid = slot.unsqueeze(0) < valid_count.unsqueeze(1)
            source = prefill_valid_starts.unsqueeze(1) + slot.unsqueeze(0)
            source = source.clamp_max(initial_state_len - 1)
            gather_key_index = source[:, None, :, None].expand(
                -1,
                self.config.num_key_value_heads,
                -1,
                int(k.size(-1)),
            )
            gather_value_index = source[:, None, :, None].expand(
                -1,
                self.config.num_key_value_heads,
                -1,
                int(v.size(-1)),
            )
            initial_state_k = torch.gather(
                k[..., :initial_state_len, :], 2, gather_key_index
            ).masked_fill(~initial_valid[:, None, :, None], 0)
            initial_state_v = torch.gather(
                v[..., :initial_state_len, :], 2, gather_value_index
            ).masked_fill(~initial_valid[:, None, :, None], 0)
            physical = slot.unsqueeze(0).expand(batch_size, -1)
            compact_owner = physical - prefill_valid_starts.unsqueeze(1)
            dummy_owner = valid_count.unsqueeze(1)
            owners = torch.where(
                physical >= prefill_valid_starts.unsqueeze(1),
                compact_owner,
                dummy_owner,
            )[:, None, :].expand(
                -1, self.config.num_key_value_heads, -1
            )
        state_k = _pad_sequence(initial_state_k, state_capacity).clone()
        state_v = _pad_sequence(initial_state_v, state_capacity).clone()
        counts = torch.zeros(
            batch_size,
            self.config.num_key_value_heads,
            state_capacity,
            1,
            dtype=torch.float32,
            device=k.device,
        )
        if initial_valid is None:
            counts[..., :initial_state_len, :].fill_(1.0)
        else:
            counts[..., :initial_state_len, :].fill_(
                torch.finfo(torch.float32).tiny
            )
            counts[..., :initial_state_len, :].masked_fill_(
                initial_valid[:, None, :, None], 1.0
            )
        state_len = initial_state_len
        state_coverage = initial_len
        page_cache = None
        if self.leaf_attention_backend == "paged":
            sequence_capacity = _round_up(prefill_len, self.chunk_len) + max(
                self.chunk_len, self.decode_cache_headroom
            )
            page_cache = self._new_page_cache(
                archive_k[..., :initial_state_len, :],
                archive_v[..., :initial_state_len, :],
                owners,
                state_capacity=state_capacity,
                sequence_capacity=sequence_capacity,
                virtual_k=archive_k if self.virtual_page_storage else None,
                virtual_v=archive_v if self.virtual_page_storage else None,
            )
            owners = None

        for query_begin in range(front_len, attention_len, prefill_chunk_len):
            query_end = min(attention_len, query_begin + prefill_chunk_len)
            bswa_begin = max(0, query_begin - exact_lookback)
            if state_coverage != bswa_begin:
                raise AssertionError("LOD prefill state coverage drifted")
            local_branch = (
                self._prefill_local_attention(
                    q[..., bswa_begin:query_end, :],
                    k[..., bswa_begin:query_end, :],
                    v[..., bswa_begin:query_end, :],
                    query_offset=query_begin - bswa_begin,
                )
                if self.split_prefill_local_attention
                else None
            )
            outputs.append(
                self._two_level_attention(
                    q[..., query_begin:query_end, :],
                    k[..., bswa_begin:query_end, :],
                    v[..., bswa_begin:query_end, :],
                    state_k,
                    state_v,
                    counts,
                    owners,
                    archive_k,
                    archive_v,
                    state_len=state_len,
                    state_capacity=state_capacity,
                    page_cache=page_cache,
                    local_branch=local_branch,
                    sink_k=sink_k,
                    sink_v=sink_v,
                )
            )

            next_bswa_begin = (
                max(0, query_begin + prefill_chunk_len - exact_lookback)
                if query_end < attention_len
                else bswa_begin
            )
            while state_coverage < next_bswa_begin:
                update_end = min(
                    next_bswa_begin, state_coverage + prefill_state_update_len
                )
                (
                    state_k,
                    state_v,
                    counts,
                    state_len,
                    new_owners,
                    old_slot_remap,
                ) = self._update_state(
                    state_k,
                    state_v,
                    counts,
                    k[..., state_coverage:update_end, :],
                    v[..., state_coverage:update_end, :],
                    state_len=state_len,
                    ctx_len=query_begin + update_end - bswa_begin,
                    available_context=update_end,
                    state_capacity=state_capacity,
                )
                if page_cache is not None:
                    if old_slot_remap is not None:
                        raise AssertionError("paged state remapping is unsupported")
                    self._append_page_cache(
                        page_cache,
                        k[..., state_coverage:update_end, :],
                        v[..., state_coverage:update_end, :],
                        new_owners,
                    )
                else:
                    if owners is None:
                        raise AssertionError("packed LOD owner archive is missing")
                    if old_slot_remap is not None:
                        owners = torch.gather(old_slot_remap, 2, owners)
                    owners = torch.cat((owners, new_owners), dim=2)
                state_coverage = update_end

        # Prepare the state boundary required by the first decode token after
        # all prefill outputs have been computed.  Otherwise prompts ending on
        # a chunk boundary pay a full 256-token state update on token one.
        decode_coverage = max(initial_len, self._bswa_begin(prefill_len + 1))
        while decode_coverage > state_coverage:
            update_end = min(
                decode_coverage, state_coverage + prefill_state_update_len
            )
            (
                state_k,
                state_v,
                counts,
                state_len,
                new_owners,
                old_slot_remap,
            ) = self._update_state(
                state_k,
                state_v,
                counts,
                k[..., state_coverage:update_end, :],
                v[..., state_coverage:update_end, :],
                state_len=state_len,
                ctx_len=min(prefill_len, update_end + self.local_len),
                available_context=update_end,
                state_capacity=state_capacity,
            )
            if page_cache is not None:
                if old_slot_remap is not None:
                    raise AssertionError("paged state remapping is unsupported")
                self._append_page_cache(
                    page_cache,
                    k[..., state_coverage:update_end, :],
                    v[..., state_coverage:update_end, :],
                    new_owners,
                )
            else:
                if owners is None:
                    raise AssertionError("packed LOD owner archive is missing")
                if old_slot_remap is not None:
                    owners = torch.gather(old_slot_remap, 2, owners)
                owners = torch.cat((owners, new_owners), dim=2)
            state_coverage = update_end
        # Right padding exists only to reuse compiled final-chunk shapes. It
        # must never become persistent state or enter the decode-local field.
        recent_k = k[..., state_coverage:prefill_len, :]
        recent_v = v[..., state_coverage:prefill_len, :]
        recent_len = int(recent_k.size(2))
        if page_cache is not None:
            recent_capacity = self.local_len + self.decode_state_update_len
            buffered_k = k.new_empty(
                *k.shape[:2], recent_capacity, int(k.size(-1))
            )
            buffered_v = v.new_empty(
                *v.shape[:2], recent_capacity, int(v.size(-1))
            )
            buffered_k[..., :recent_len, :].copy_(recent_k)
            buffered_v[..., :recent_len, :].copy_(recent_v)
            recent_k = buffered_k
            recent_v = buffered_v
            quantized_counts = page_cache.get("page_quantized_counts")
            if isinstance(quantized_counts, torch.Tensor):
                quantization_names = (
                    "leaf_k",
                    "leaf_v",
                    "page_indices",
                    "page_sum_k",
                    "page_sum_v",
                    "page_counts",
                    "quantized_leaf_k",
                    "quantized_leaf_v",
                    "page_k_scales",
                    "page_v_scales",
                )
                quantization_tensors = tuple(
                    page_cache.get(name) for name in quantization_names
                )
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in quantization_tensors
                ):
                    raise RuntimeError("virtual INT4 prefill cache is incomplete")
                if self.leaf_quant_scale_mode not in ("max", "l2"):
                    raise ValueError("leaf quantization scale mode must be max or l2")
                if self.leaf_append_quant_scale_mode not in ("max", "l2"):
                    raise ValueError(
                        "leaf append quantization scale mode must be max or l2"
                    )
                quantize_virtual_paged_kv_int4(
                    *quantization_tensors,
                    quantized_counts,
                    quant_group_size=self.leaf_quant_group_size,
                    optimize_scale=self.leaf_quant_scale_mode == "l2",
                )
                page_cache["quantization_finalized"] = True
                if self.page_summary_quant_bits not in (0, 8):
                    raise ValueError("page-summary quantization supports 0 or 8 bits")
                if self.page_summary_scale_mode not in ("max", "l2"):
                    raise ValueError("page-summary scale mode must be max or l2")
                if self.page_summary_quant_bits == 8:
                    (
                        page_cache["quantized_page_sum_k"],
                        page_cache["quantized_page_sum_v"],
                        page_cache["page_sum_k_scales"],
                        page_cache["page_sum_v_scales"],
                    ) = quantize_page_summaries_int8(
                        page_cache["page_sum_k"],
                        page_cache["page_sum_v"],
                        quant_group_size=self.leaf_quant_group_size,
                        optimize_scale=self.page_summary_scale_mode == "l2",
                    )
                    page_cache["summary_quantization_finalized"] = True
                    page_cache["page_sum_k"] = k.new_empty(
                        *k.shape[:2], 1, int(k.size(-1))
                    )
                    page_cache["page_sum_v"] = v.new_empty(
                        *v.shape[:2], 1, int(v.size(-1))
                    )
                # All archived leaves now live in the packed flat tensors.  Keep
                # only typed pointer sentinels for the compile-time BF16 fallback.
                page_cache["leaf_k"] = k.new_empty(
                    *k.shape[:2], 1, int(k.size(-1))
                )
                page_cache["leaf_v"] = v.new_empty(
                    *v.shape[:2], 1, int(v.size(-1))
                )
        self._lod_state = {
            "state_k": state_k.detach(),
            "state_v": state_v.detach(),
            "counts": counts.detach(),
            "state_len": state_len,
            "coverage": state_coverage,
            "state_capacity": state_capacity,
            "recent_k": recent_k.detach(),
            "recent_v": recent_v.detach(),
            "recent_len": recent_len,
            "total_len": prefill_len,
        }
        if sink_k is not None and sink_v is not None:
            self._lod_state["sink_k"] = sink_k
            self._lod_state["sink_v"] = sink_v
        if page_cache is not None:
            self._lod_state["page_cache"] = page_cache
        else:
            if owners is None:
                raise AssertionError("packed LOD owner archive is missing")
            self._lod_state["owners"] = owners.detach()
            self._lod_state["exact_k"] = archive_k.detach()
            self._lod_state["exact_v"] = archive_v.detach()
        return torch.cat(outputs, dim=2)

    @torch.compiler.disable
    def _cached_prefill_attention(
        self,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> torch.Tensor:
        """Append a causal multi-token turn without replaying decode kernels."""
        if not hasattr(self, "_lod_state"):
            raise RuntimeError("cached LOD prefill did not receive a prior state")
        if int(q.size(2)) <= 1:
            raise ValueError("cached LOD prefill requires multiple query tokens")
        cache = self._lod_state
        page_cache = cache.get("page_cache")
        if not isinstance(page_cache, dict):
            raise NotImplementedError(
                "fast cached prefill currently requires the paged LOD backend"
            )
        if not self.split_prefill_local_attention:
            raise NotImplementedError(
                "fast cached prefill requires split local attention"
            )

        state_k = cache["state_k"]
        state_v = cache["state_v"]
        counts = cache["counts"]
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (state_k, state_v, counts)
        ):
            raise TypeError("cached LOD state tensors are missing")
        state_len = int(cache["state_len"])
        state_coverage = int(cache["coverage"])
        initial_coverage = state_coverage
        previous_total_len = int(cache["total_len"])
        turn_len = int(q.size(2))
        total_len = previous_total_len + turn_len
        recent_k = cache["recent_k"]
        recent_v = cache["recent_v"]
        if not isinstance(recent_k, torch.Tensor) or not isinstance(
            recent_v, torch.Tensor
        ):
            raise TypeError("cached LOD recent tensors are missing")
        recent_len = int(cache.get("recent_len", recent_k.size(2)))
        if previous_total_len - initial_coverage != recent_len:
            raise AssertionError("cached LOD recent coverage drifted")
        working_k = torch.cat(
            (recent_k[..., :recent_len, :], new_k), dim=2
        ).contiguous()
        working_v = torch.cat(
            (recent_v[..., :recent_len, :], new_v), dim=2
        ).contiguous()

        state_capacity = max(
            int(cache["state_capacity"]),
            self._state_capacity(total_len, state_len),
        )
        if int(state_k.size(2)) < state_capacity:
            state_k = _pad_sequence(state_k, state_capacity).clone()
            state_v = _pad_sequence(state_v, state_capacity).clone()
            counts = _pad_sequence(counts, state_capacity).clone()
            self._grow_slot_page_table(
                page_cache, required_slots=state_capacity
            )

        sink_k = cache.get("sink_k")
        sink_v = cache.get("sink_v")
        if (sink_k is None) != (sink_v is None):
            raise RuntimeError("LOD separate sink cache is incomplete")
        owners = cache.get("owners")
        exact_k = cache.get("exact_k", recent_k)
        exact_v = cache.get("exact_v", recent_v)
        prefill_chunk_len = int(self.prefill_chunk_len)
        prefill_state_update_len = int(self.prefill_state_update_len)
        exact_lookback = int(self.prefill_local_len) - prefill_chunk_len
        if prefill_chunk_len <= 0 or prefill_state_update_len <= 0:
            raise ValueError("cached prefill lengths must be positive")
        if exact_lookback < 0:
            raise ValueError("prefill local length cannot be shorter than its chunk")

        def update_to(target_coverage: int, *, context_length_for) -> None:
            nonlocal state_k, state_v, counts, state_len, state_coverage, owners
            while state_coverage < target_coverage:
                update_end = min(
                    target_coverage,
                    state_coverage + prefill_state_update_len,
                )
                source_begin = state_coverage - initial_coverage
                source_end = update_end - initial_coverage
                overflow_k = working_k[..., source_begin:source_end, :]
                overflow_v = working_v[..., source_begin:source_end, :]
                (
                    state_k,
                    state_v,
                    counts,
                    state_len,
                    new_owners,
                    old_slot_remap,
                ) = self._update_state(
                    state_k,
                    state_v,
                    counts,
                    overflow_k,
                    overflow_v,
                    state_len=state_len,
                    ctx_len=context_length_for(update_end),
                    available_context=update_end,
                    state_capacity=state_capacity,
                )
                if old_slot_remap is not None:
                    raise AssertionError("paged state remapping is unsupported")
                self._append_page_cache(
                    page_cache, overflow_k, overflow_v, new_owners
                )
                state_coverage = update_end

        outputs = []
        for query_begin in range(0, turn_len, prefill_chunk_len):
            query_end = min(turn_len, query_begin + prefill_chunk_len)
            absolute_query_begin = previous_total_len + query_begin
            desired_coverage = max(
                state_coverage,
                absolute_query_begin - exact_lookback,
            )
            update_to(
                desired_coverage,
                context_length_for=lambda update_end: (
                    absolute_query_begin + update_end - desired_coverage
                ),
            )
            local_begin = state_coverage - initial_coverage
            local_end = previous_total_len + query_end - initial_coverage
            local_k = working_k[..., local_begin:local_end, :]
            local_v = working_v[..., local_begin:local_end, :]
            query_prefix_len = absolute_query_begin - state_coverage
            local_query = torch.cat(
                (
                    q.new_zeros(
                        *q.shape[:2], query_prefix_len, int(q.size(-1))
                    ),
                    q[..., query_begin:query_end, :],
                ),
                dim=2,
            )
            local_branch = self._prefill_local_attention(
                local_query,
                local_k,
                local_v,
                query_offset=query_prefix_len,
            )
            outputs.append(
                self._two_level_attention(
                    q[..., query_begin:query_end, :],
                    local_k,
                    local_v,
                    state_k,
                    state_v,
                    counts,
                    owners,
                    exact_k,
                    exact_v,
                    state_len=state_len,
                    state_capacity=state_capacity,
                    page_cache=page_cache,
                    local_branch=local_branch,
                    sink_k=sink_k,
                    sink_v=sink_v,
                )
            )

        decode_coverage = max(
            state_coverage,
            self._bswa_begin(total_len + 1),
        )
        update_to(
            decode_coverage,
            context_length_for=lambda update_end: min(
                total_len, update_end + self.local_len
            ),
        )
        tail_begin = state_coverage - initial_coverage
        tail_k = working_k[..., tail_begin:, :]
        tail_v = working_v[..., tail_begin:, :]
        tail_len = int(tail_k.size(2))
        if tail_len > int(recent_k.size(2)):
            recent_capacity = max(
                tail_len,
                self.local_len + self.decode_state_update_len,
            )
            recent_k = new_k.new_empty(
                *new_k.shape[:2], recent_capacity, int(new_k.size(-1))
            )
            recent_v = new_v.new_empty(
                *new_v.shape[:2], recent_capacity, int(new_v.size(-1))
            )
        recent_k[..., :tail_len, :].copy_(tail_k)
        recent_v[..., :tail_len, :].copy_(tail_v)
        cache.update(
            state_k=state_k.detach(),
            state_v=state_v.detach(),
            counts=counts.detach(),
            state_len=state_len,
            coverage=state_coverage,
            state_capacity=state_capacity,
            recent_k=recent_k.detach(),
            recent_v=recent_v.detach(),
            recent_len=tail_len,
            total_len=total_len,
        )
        return torch.cat(outputs, dim=2)

    @torch.compiler.disable
    def _decode_attention(
        self,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        *,
        total_len: int,
    ) -> torch.Tensor:
        if not hasattr(self, "_lod_state"):
            raise RuntimeError("LOD decode did not receive a prefill state")
        cache = self._lod_state
        state_k = cache["state_k"]
        state_v = cache["state_v"]
        counts = cache["counts"]
        state_len = int(cache["state_len"])
        page_cache = cache.get("page_cache")
        owners = cache.get("owners")
        sink_k = cache.get("sink_k")
        sink_v = cache.get("sink_v")
        if (sink_k is None) != (sink_v is None):
            raise RuntimeError("LOD separate sink cache is incomplete")
        state_coverage = int(cache["coverage"])
        recent_k = cache["recent_k"]
        recent_v = cache["recent_v"]
        if not isinstance(recent_k, torch.Tensor) or not isinstance(
            recent_v, torch.Tensor
        ):
            raise TypeError("LOD recent cache is missing")
        previous_total_len = int(cache["total_len"])
        if total_len != previous_total_len + int(new_k.size(2)):
            raise AssertionError("LOD decode position drifted")
        recent_len = int(cache.get("recent_len", recent_k.size(2)))
        if page_cache is None:
            recent_k = torch.cat((recent_k, new_k), dim=2)
            recent_v = torch.cat((recent_v, new_v), dim=2)
            recent_len = int(recent_k.size(2))
            exact_k = torch.cat((cache["exact_k"], new_k), dim=2)
            exact_v = torch.cat((cache["exact_v"], new_v), dim=2)
        else:
            # Unused by the paged backend; keep the call signature uniform.
            exact_k = recent_k
            exact_v = recent_v
        new_bswa_begin = self._bswa_begin(total_len)
        decode_update_len = int(self.decode_state_update_len)
        if decode_update_len <= 0:
            raise ValueError("decode state update length must be positive")
        exact_floor = self.local_len - self.chunk_len
        if exact_floor < 0:
            raise ValueError("LOD local length cannot be shorter than one chunk")
        # Only tokens from prior calls can move into persistent state.  In
        # particular, a prompt shorter than one chunk has no local tail yet,
        # so archiving the just-arrived decode token would underflow recent_k.
        target_coverage = max(
            min(previous_total_len, self.chunk_len), state_coverage
        )
        pending_update = total_len - target_coverage - exact_floor
        if pending_update > decode_update_len:
            target_coverage += (
                (pending_update - 1) // decode_update_len
            ) * decode_update_len

        if state_coverage < target_coverage:
            overflow_len = target_coverage - state_coverage
            overflow_k = recent_k[..., :overflow_len, :]
            overflow_v = recent_v[..., :overflow_len, :]
            update_state_capacity = max(
                int(cache["state_capacity"]),
                self._state_capacity(total_len, state_len),
            )
            (
                state_k,
                state_v,
                counts,
                state_len,
                new_owners,
                old_slot_remap,
            ) = self._update_state(
                state_k,
                state_v,
                counts,
                overflow_k,
                overflow_v,
                state_len=state_len,
                ctx_len=exact_floor + target_coverage,
                available_context=target_coverage,
                state_capacity=update_state_capacity,
            )
            if page_cache is not None:
                if old_slot_remap is not None:
                    raise AssertionError("paged state remapping is unsupported")
                self._append_page_cache(page_cache, overflow_k, overflow_v, new_owners)
            else:
                if owners is None:
                    raise AssertionError("packed LOD owner archive is missing")
                if old_slot_remap is not None:
                    owners = torch.gather(old_slot_remap, 2, owners)
                owners = torch.cat((owners, new_owners), dim=2)
            state_coverage = target_coverage
            if page_cache is None:
                recent_k = recent_k[..., overflow_len:, :]
                recent_v = recent_v[..., overflow_len:, :]
                recent_len = int(recent_k.size(2))
            else:
                remaining = recent_len - overflow_len
                if remaining < 0:
                    raise AssertionError("LOD local cache underflowed")
                if remaining:
                    recent_k[..., :remaining, :].copy_(
                        recent_k[..., overflow_len:recent_len, :]
                    )
                    recent_v[..., :remaining, :].copy_(
                        recent_v[..., overflow_len:recent_len, :]
                    )
                recent_len = remaining
            cache["state_capacity"] = update_state_capacity

        if new_bswa_begin == 0:
            if page_cache is None:
                current_k = recent_k
                current_v = recent_v
            else:
                current_k = torch.cat(
                    (recent_k[..., :recent_len, :], new_k), dim=2
                )
                current_v = torch.cat(
                    (recent_v[..., :recent_len, :], new_v), dim=2
                )
                recent_k[..., recent_len : recent_len + 1, :].copy_(new_k)
                recent_v[..., recent_len : recent_len + 1, :].copy_(new_v)
                recent_len += 1
            output = self._exact_attention(q, current_k, current_v, causal=False)
        else:
            state_capacity = max(
                int(cache["state_capacity"]),
                self._state_capacity(total_len, state_len),
            )
            buffered_decode = (
                page_cache is not None
                and self.fused_decode_attention
                and not isinstance(page_cache.get("page_indices"), torch.Tensor)
            )
            if page_cache is None:
                local_k = recent_k
                local_v = recent_v
                append_k = None
                append_v = None
                active_local_len = None
            elif buffered_decode:
                if recent_len >= int(recent_k.size(2)):
                    raise AssertionError("LOD local cache overflowed")
                local_k = recent_k
                local_v = recent_v
                append_k = new_k
                append_v = new_v
                active_local_len = recent_len
            else:
                local_k = torch.cat(
                    (recent_k[..., :recent_len, :], new_k), dim=2
                )
                local_v = torch.cat(
                    (recent_v[..., :recent_len, :], new_v), dim=2
                )
                append_k = None
                append_v = None
                active_local_len = None
            output = self._two_level_attention(
                q,
                local_k,
                local_v,
                state_k,
                state_v,
                counts,
                owners,
                exact_k,
                exact_v,
                state_len=state_len,
                state_capacity=state_capacity,
                page_cache=page_cache,
                local_len=active_local_len,
                new_k=append_k,
                new_v=append_v,
                sink_k=sink_k,
                sink_v=sink_v,
            )
            if buffered_decode:
                recent_len += 1
            elif page_cache is not None:
                recent_k[..., recent_len : recent_len + 1, :].copy_(new_k)
                recent_v[..., recent_len : recent_len + 1, :].copy_(new_v)
                recent_len += 1
            cache["state_capacity"] = state_capacity

        cache.update(
            state_k=state_k.detach(),
            state_v=state_v.detach(),
            counts=counts.detach(),
            state_len=state_len,
            coverage=state_coverage,
            recent_k=recent_k.detach(),
            recent_v=recent_v.detach(),
            recent_len=recent_len,
            total_len=total_len,
        )
        if owners is not None:
            cache["owners"] = owners.detach()
            cache["exact_k"] = exact_k.detach()
            cache["exact_v"] = exact_v.detach()
        return output

__all__ = ["TritonLODAttentionCore"]
