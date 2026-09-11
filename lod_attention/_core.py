"""Inference-only, model-independent Triton LOD attention core.

This module consumes head-separated query, key, and value tensors after their
model-specific projections, normalization, and positional encoding. Old KV
leaves are partitioned into a ``16*sqrt(T)`` state; a query expands the leaves
of its top-four state slots and uses count-corrected mean KV summaries for every
other slot. The exact and coarse branches are combined with their log-sum-exp
statistics.

The exact-leaf archive is stored in 16-token pages, while decode keeps only a
bounded recent KV window. Model adapters live in separate modules.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from .kernels.paged_leaf_attention import (
    append_quantized_virtual_paged_kv,
    append_virtual_paged_kv,
    fused_decode_paged_lod_attention,
    new_fused_decode_buffers,
    paged_leaf_attention,
    query_major_indexed_residual_page_attention,
    quantize_page_summaries_int8,
    quantize_virtual_paged_kv,
)
from .kernels.lod_kernels import (
    constituent_rms,
    merge_attention_branches_with_sink,
    merge_state_in_place,
    new_state_delta_buffers,
    new_state_maxsim_buffers,
    prepare_state_clustering_keys,
    route_logits_coarse_attention,
    route_logits_hierarchical_topk,
    streaming_state_maxsim,
)
from ._tensor_ops import (
    all_indices as _all_idx,
    gather_by_index as _gather_by_idx,
    premerge_adjacent_state_inputs as _premerge_adjacent_state_inputs,
    split_append_merge_indices as _split_append_merge_idx_by_maxsim,
    sum_adjacent_groups as _sum_adjacent_groups,
)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


_PAGE_DIRECTORY_SIZE = 64


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
    return left_output * weights[..., 0].unsqueeze(-1) + right_output * weights[
        ..., 1
    ].unsqueeze(-1)


class TritonLODAttentionCore(nn.Module):
    """Projection-free mass-corrected top-four LOD implementation."""

    chunk_len = 256
    local_len = 512
    prefill_chunk_len = 256
    prefill_local_len = 512
    prefill_state_update_len = 256
    prefill_exact_first_chunk = False
    prefill_overlap_exact_state = False
    exact_decode_limit = 0
    prefill_local_attention_backend = "torch"
    decode_state_update_len = 256
    decode_cache_headroom = 256
    state_growth_factor = 16.0
    state_min_len = 256
    state_size_offset = 0
    state_premerge_factor = 1
    state_split_max_leaves: int | None = None
    sink_len = 1
    # A protected singleton is already exact in the coarse branch, so opening
    # its one-token leaf would only consume a detailed-region route.
    exclude_sink_from_routes = True
    # Keep the exact sink outside the centroid state and leaf archive. Its
    # attention contribution is merged as a separate exact branch.
    separate_sink_cache = False
    two_level_topk = 4
    prefill_two_level_topk: int | None = None
    # Stop archiving exact leaves once a centroid reaches this many members.
    # Its K/V sums and count still receive every assignment, so a sealed
    # centroid remains available as coarse residual mass.
    leaf_seal_capacity: int | None = None
    leaf_attention_backend = "packed"
    leaf_page_size = 16
    # Use a two-level direct page directory instead of spilling long centroid
    # posting lists into an open-addressed hash table.  One compact root entry
    # addresses 64 physical-page IDs, so the root remains small while lookup
    # has a fixed two-load cost and cannot run out of probes.
    leaf_paged_directory = True
    leaf_inline_pages_per_slot = 128
    leaf_overflow_hash_factor = 4
    leaf_hash_probes = 8
    leaf_block_m = 16
    leaf_block_n = 32
    # The gfx942 BF16 expert kernel has a materially better D=128 operating
    # point at M64/N64.  Apply it only to the untouched legacy defaults so
    # explicit experiment settings keep their exact meaning; D=256 and D=512
    # retain the established M16/N32 path.
    leaf_geometry_tuning = True
    leaf_num_warps = 2
    leaf_waves_per_eu = 1
    leaf_layout = "query"
    # Each row merges only the small fixed route list. One wave is sufficient and
    # avoids the synchronization/occupancy overhead of a four-wave reduction.
    leaf_reduce_num_warps = 1
    # Store flat exact leaves as tokenwise symmetric INT8 and execute both QK
    # and quantized-probability PV with INT8 MMA during expert-layout prefill.
    prefill_int8_leaf_mma = False
    leaf_key_quant_bits = 0
    leaf_value_quant_bits = 0
    leaf_quant_group_size = 32
    leaf_quant_token_group_size = 16
    leaf_quant_scale_mode = "max"
    leaf_append_quant_scale_mode = "max"
    page_summary_quant_bits = 8
    page_summary_scale_mode = "l2"
    virtual_page_storage = False
    recursive_page_lod = False
    recursive_page_block_n = 16
    # Keep the query-major residual-page launch independent from the regular
    # expert/MFMA consumer.  Adaptive recursive prefill may enable the latter
    # for short prompts, but long prompts must retain the measured one-wave
    # residual-page geometry.
    recursive_page_attention_num_warps = 1
    recursive_materialize_page_scores = False
    recursive_page_score_block_n = 16
    recursive_page_score_num_warps = 2
    recursive_page_select_block_n = 64
    recursive_state_route_backend = "fused"
    # Three-tier prefill opens every leaf in each selected centroid; recursive
    # page refinement is used only for decode.
    recursive_prefill_all_leaves = False
    recursive_prefill_all_leaves_token_limit = 0
    recursive_prefill_request_total_len = 0
    prefill_route_mass_fraction: float | None = None
    prefill_route_block_m = 16
    prefill_route_num_warps = 4
    prefill_mass_include_local_lse = True
    prefill_overlap_local_lod = False
    prefill_overlap_coarse_leaf = False
    fused_decode_attention = True
    fused_decode_state_route = True
    decode_split_kv = 8
    decode_use_dot = False
    decode_block_n = 16
    decode_num_warps = 2
    decode_route_group_size = 32
    decode_route_segment_tiles = 1
    decode_route_num_warps = 2
    decode_route_reduce_num_warps = 4
    decode_route_parallel_reduce = False
    decode_route_parallel_reduce_block_d = 0
    decode_final_reduce_num_warps = 4
    decode_fuse_final_reduce = False
    # D=512 route tiles are faster with the scalar accumulation path on
    # gfx942; D=128/256 retain MFMA routing.
    decode_geometry_tuning = True
    decode_route_gqa_grouped = True
    decode_gqa_cooperative_leaf = True
    decode_gqa_cooperative_hip = True
    # Auto-tune page-list parallelism from context. An explicit 4/8/16/32
    # value and experimental per-centroid adaptation remain available.
    decode_gqa_cooperative_route_splits: int | None = None
    decode_gqa_cooperative_adaptive_splits = False
    # Directly fold the fixed route partials into the final branch reduction.
    # The dispatch automatically retains the two-stage tree for 16/32 splits.
    decode_gqa_cooperative_fused_reduce = True
    # Dense BF16 state aggregation is faster for batch-one inference;
    # the KVM-style FP32 delta path remains available for larger batches.
    fused_state_update = False
    auto_fused_state_update = True
    reuse_state_update_similarity = True
    fused_state_maxsim = False
    state_maxsim_block_m = 16
    state_maxsim_block_n = 32
    state_maxsim_num_warps = 4
    route_gqa_matmul = False
    state_clustering_normalization = "none"
    state_clustering_radial_bias = 0.0
    state_clustering_radial_scope = "all"
    state_clustering_centroid_rescale = "none"
    state_clustering_centroid_rescale_scope = "all"
    state_clustering_query_metric = "none"
    state_clustering_rope_dim = 0
    state_clustering_rope_fast_pairs = 0
    coherence_single_matmul = True
    routing_normalization = "none"
    routing_rope_dim = 0
    routing_rope_fast_pairs = 0
    routing_page_mass_candidates = 0
    coarse_route_block_m = 16
    coarse_route_block_n = 32
    coarse_route_num_warps = 8
    coarse_max_grouped_rows = 8
    prefill_coarse_max_grouped_rows = 8
    prefill_coarse_direct_gqa = False
    prefill_coarse_route_block_n = 32
    prefill_coarse_route_num_warps = 8
    # Use patched AITER FMHA to produce count-corrected coarse attention and
    # compact per-tile route candidates in one native-GQA pass.
    prefill_aiter_route_coarse = False
    fused_prefill_route_coarse = False
    fused_prefill_stable_recompute = True
    fused_prefill_external_recompute = True
    # VLLM's pool enables the exact two-stage selector only for measured
    # geometries. Standalone/HF callers retain the established selector unless
    # they opt in explicitly.
    prefill_hierarchical_route = False
    split_prefill_local_attention = False
    overflow_bipartite_merge = False
    overflow_bipartite_block_size = 32
    overflow_bipartite_positional_halves = False
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
        target = (
            max(
                math.floor(self.state_growth_factor * math.sqrt(max(ctx_len, 0))),
                self.state_min_len,
            )
            + self.state_size_offset
        )
        available = max(available_context - separated, 0)
        return max(current_state_len, min(target, available))

    def _protected_state_len(self, state_len: int) -> int:
        if self.separate_sink_cache:
            return 0
        return min(self.sink_len, state_len)

    def _protected_route_len(self, state_len: int) -> int:
        # Protected singleton slots need no exact-leaf route. Fixed groups do:
        # their coarse K/V mean is no longer identical to either constituent.
        if self.state_premerge_factor > 1:
            return 0
        return self._protected_state_len(state_len)

    def _state_capacity(self, total_len: int, current_state_len: int) -> int:
        # One chunk of headroom avoids recompilation during short generation.
        capacity_len = total_len + self.chunk_len
        if self.state_split_max_leaves is not None:
            # Scheduled entries and split children are independent. The latter
            # can add at most one slot per ``max_leaves`` archived tokens.
            scheduled_target = self._desired_state_len(capacity_len, capacity_len, 0)
            # Children are full except for the tail produced by each parent
            # posting list. Reserve one additional scheduled-state population
            # for those tails; the simple T/max_leaves term alone assumes
            # perfectly packed children and is slightly too small in practice.
            target = max(
                current_state_len,
                2 * scheduled_target
                + math.ceil(capacity_len / self.state_split_max_leaves),
            )
        else:
            target = self._desired_state_len(
                capacity_len, capacity_len, current_state_len
            )
        return _round_up(target, self.chunk_len)

    def _next_scheduled_state_len(
        self,
        scheduled_state_len: int,
        *,
        ctx_len: int,
        available_context: int,
        overflow_len: int,
    ) -> int:
        """Advance only the ordinary sqrt(T) centroid-growth schedule."""

        target = self._desired_state_len(
            ctx_len, available_context, scheduled_state_len
        )
        return scheduled_state_len + min(
            max(target - scheduled_state_len, 0), overflow_len
        )

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
        block_scores = scores.float().reshape(
            *scores.shape[:-1], subblocks, subblock_size
        )
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
            append_parts.append(block_indices[..., :remainder, :high_quota].flatten(-2))
            merge_parts.append(block_indices[..., :remainder, high_quota:].flatten(-2))
        if remainder < subblocks:
            append_parts.append(block_indices[..., remainder:, :base_quota].flatten(-2))
            merge_parts.append(block_indices[..., remainder:, base_quota:].flatten(-2))
        append_idx = (
            append_parts[0]
            if len(append_parts) == 1
            else torch.cat(append_parts, dim=-1)
        )
        merge_idx = (
            merge_parts[0] if len(merge_parts) == 1 else torch.cat(merge_parts, dim=-1)
        )
        return (
            torch.sort(append_idx, dim=-1).values,
            torch.sort(merge_idx, dim=-1).values,
        )

    def _split_overfull_state_destinations(
        self,
        destination: torch.Tensor,
        destination_scores: torch.Tensor,
        destination_keys: torch.Tensor,
        counts: torch.Tensor,
        *,
        state_len: int,
        state_capacity: int,
    ) -> tuple[torch.Tensor, int]:
        """Redirect over-cap assignments into affinity-stratified children.

        Existing leaves remain in place. Within each destination, the new
        assignments most similar to its current center fill the remaining
        room.  For every overfull posting list, its least parent-like key is
        then used as a directional pivot and the overflow is gathered into
        bounded bands by similarity to that pivot.  Thus siblings differ in
        key direction, rather than merely being arbitrary arrival-order (or
        scalar parent-affinity) partitions.
        """

        max_leaves = self.state_split_max_leaves
        if max_leaves is None or int(destination.size(-1)) == 0:
            return destination, state_len
        if destination_keys.shape[:-1] != destination.shape:
            raise ValueError("split destination keys have the wrong shape")
        if self.state_premerge_factor != 1:
            raise ValueError(
                "state posting-list splitting requires unmerged token leaves"
            )
        active_counts = counts[..., :state_len, 0].round().to(torch.long)
        if int(active_counts.max().item()) > max_leaves:
            raise AssertionError(
                "an existing centroid exceeded the configured split limit"
            )

        # Sort by affinity first, then stably group by destination. The second
        # sort retains descending affinity inside each posting list.
        affinity_order = torch.argsort(
            destination_scores.float(), dim=-1, descending=True, stable=True
        )
        affinity_destinations = destination.gather(-1, affinity_order)
        destination_order = torch.argsort(affinity_destinations, dim=-1, stable=True)
        order = affinity_order.gather(-1, destination_order)
        sorted_destination = destination.gather(-1, order)

        positions = torch.arange(
            int(destination.size(-1)),
            dtype=torch.long,
            device=destination.device,
        ).view(1, 1, -1)
        starts = torch.ones_like(sorted_destination, dtype=torch.bool)
        starts[..., 1:] = sorted_destination[..., 1:] != sorted_destination[..., :-1]
        start_positions = torch.where(starts, positions, 0)
        group_starts = torch.cummax(start_positions, dim=-1).values
        rank_in_destination = positions - group_starts
        prior_count = active_counts.gather(-1, sorted_destination)
        remaining = (max_leaves - prior_count).clamp_min(0)
        overflow_rank = rank_in_destination - remaining
        split_assignment = overflow_rank >= 0

        # Parent affinity is only one scalar projection and does not separate
        # tangential directions.  Take the least parent-like assignment in
        # each posting list as a deterministic far-point pivot, then order the
        # overflow by similarity to that pivot.  Stable sorting keeps the
        # parent-fill prefix in its original affinity order.  The second
        # stable destination sort is a vectorized segmented sort.
        if bool(split_assignment.any().item()):
            sorted_keys = destination_keys.gather(
                2,
                order.unsqueeze(-1).expand_as(destination_keys),
            )
            ends = torch.ones_like(sorted_destination, dtype=torch.bool)
            ends[..., :-1] = sorted_destination[..., :-1] != sorted_destination[..., 1:]
            sentinel = int(destination.size(-1))
            end_candidates = torch.where(ends, positions, sentinel)
            group_ends = torch.flip(
                torch.cummin(torch.flip(end_candidates, dims=(-1,)), dim=-1).values,
                dims=(-1,),
            )
            pivot_keys = sorted_keys.gather(
                2,
                group_ends.unsqueeze(-1).expand_as(sorted_keys),
            )
            pivot_similarity = (sorted_keys * pivot_keys).sum(
                dim=-1, dtype=torch.float32
            )
            # All parent-fill assignments sort before overflow and retain
            # their existing affinity order through the stable sort.
            secondary_score = torch.where(
                split_assignment,
                pivot_similarity,
                torch.full_like(pivot_similarity, float("inf")),
            )
            secondary_order = torch.argsort(
                secondary_score, dim=-1, descending=True, stable=True
            )
            secondary_destination = sorted_destination.gather(-1, secondary_order)
            regroup = torch.argsort(secondary_destination, dim=-1, stable=True)
            segmented_order = secondary_order.gather(-1, regroup)
            order = order.gather(-1, segmented_order)
            sorted_destination = sorted_destination.gather(-1, segmented_order)

            starts = torch.ones_like(sorted_destination, dtype=torch.bool)
            starts[..., 1:] = (
                sorted_destination[..., 1:] != sorted_destination[..., :-1]
            )
            start_positions = torch.where(starts, positions, 0)
            group_starts = torch.cummax(start_positions, dim=-1).values
            rank_in_destination = positions - group_starts
            prior_count = active_counts.gather(-1, sorted_destination)
            remaining = (max_leaves - prior_count).clamp_min(0)
            overflow_rank = rank_in_destination - remaining
            split_assignment = overflow_rank >= 0

        child_start = split_assignment & overflow_rank.remainder(max_leaves).eq(0)
        child_ordinal = child_start.to(torch.long).cumsum(dim=-1) - 1
        extra_slots = int(child_start.sum(dim=-1).max().item())
        if extra_slots == 0:
            return destination, state_len
        output_state_len = state_len + extra_slots
        if output_state_len > state_capacity:
            raise RuntimeError(
                "split centroid state exceeded its overcomplete allocation: "
                f"required {output_state_len}, capacity {state_capacity}"
            )
        child_destination = state_len + child_ordinal
        sorted_output = torch.where(
            split_assignment, child_destination, sorted_destination
        )
        output = torch.empty_like(destination)
        output.scatter_(-1, order, sorted_output)
        return output, output_state_len

    def _state_clustering_query_scale(
        self,
        query: torch.Tensor,
        *,
        valid_starts: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return a square-root transform for each KV head's query metric.

        For keys sharing one KV head, the expected squared attention-logit
        error is ``(k - mean).T E[q q.T] (k - mean)``.  The diagonal square
        root maps keys into that metric without storing any additional
        centroid data. The diagonal mode is cheap; the full mode retains
        cross-channel covariance. A scalar normalization per KV head keeps
        either mode purely directional, since globally rescaling a head cannot
        change its assignments.
        """
        if self.state_clustering_query_metric == "none":
            return None
        if self.state_clustering_query_metric not in {"diagonal", "full"}:
            raise ValueError(
                "state clustering query metric must be none, diagonal, or full"
            )
        batch_size, query_heads, query_len, head_dim = query.shape
        key_value_heads = int(self.config.num_key_value_heads)
        groups = int(self.num_key_value_groups)
        if query_heads != key_value_heads * groups:
            raise ValueError("query heads do not match the configured GQA geometry")
        grouped = (
            query.detach()
            .float()
            .reshape(batch_size, key_value_heads, groups, query_len, head_dim)
        )
        valid = None
        if valid_starts is not None:
            if tuple(valid_starts.shape) != (batch_size,):
                raise ValueError("valid query starts must have one entry per row")
            position = torch.arange(query_len, device=query.device)
            valid = position.unsqueeze(0) >= valid_starts.unsqueeze(1)
        if self.state_clustering_query_metric == "diagonal":
            if valid is None:
                mean_square = grouped.square().mean(dim=(2, 3), keepdim=False)
            else:
                denominator = valid.sum(dim=1).clamp_min(1).view(batch_size, 1, 1)
                mean_square = (grouped.square() * valid[:, None, None, :, None]).sum(
                    dim=(2, 3)
                ) / (denominator * groups)
            scale = mean_square.clamp_min(1e-12).sqrt()
            scale = scale * torch.rsqrt(
                scale.square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
            )
            return scale.unsqueeze(2)

        if valid is None:
            covariance = torch.einsum("bkgtd,bkgte->bkde", grouped, grouped) / float(
                groups * query_len
            )
        else:
            masked = grouped * valid[:, None, None, :, None]
            denominator = (valid.sum(dim=1).clamp_min(1).float() * groups).view(
                batch_size, 1, 1, 1
            )
            covariance = torch.einsum("bkgtd,bkgte->bkde", masked, masked) / denominator
        mean_variance = covariance.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
        covariance = covariance / mean_variance.clamp_min(1e-12).unsqueeze(-1)
        identity = torch.eye(head_dim, dtype=covariance.dtype, device=covariance.device)
        return torch.linalg.cholesky(covariance + 1e-4 * identity)

    def _mla_normalize_key(
        self,
        key: torch.Tensor,
        *,
        state_centroid: bool,
    ) -> torch.Tensor:
        """Normalize a raw MLA latent either per-token or after aggregation."""
        mode = getattr(self, "mla_state_key_normalization", "none")
        if mode == "none":
            return key
        weight = getattr(self, "mla_key_norm_weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError("raw MLA keys are missing their RMSNorm gain")
        latent_dim = int(weight.numel())
        if latent_dim <= 0 or latent_dim >= int(key.size(-1)):
            raise ValueError("raw MLA key has the wrong latent/RoPE geometry")
        key_float = key.detach().float()
        epsilon = float(getattr(self, "mla_key_norm_epsilon", 0.0))
        if state_centroid and mode == "raw":
            return key
        if state_centroid and mode == "whole":
            inverse_rms = torch.rsqrt(
                key_float.square().mean(dim=-1, keepdim=True) + epsilon
            )
            normalized = (key_float * inverse_rms).to(key.dtype)
            normalized[..., :latent_dim] *= weight.detach().to(key.dtype)
        else:
            latent = key_float[..., :latent_dim]
            inverse_rms = torch.rsqrt(
                latent.square().mean(dim=-1, keepdim=True) + epsilon
            )
            # Match the model RMSNorm exactly: DeepSeek rounds the unit-RMS
            # activation back to its input dtype before applying the learned
            # gain.  Reversing those two operations is measurably different
            # on the key-similarity edge cases this experiment targets.
            normalized_latent = (latent * inverse_rms).to(key.dtype)
            normalized_latent = normalized_latent * weight.detach().to(key.dtype)
            normalized = torch.cat((normalized_latent, key[..., latent_dim:]), dim=-1)
        return normalized

    def _mla_state_key_sum_for_attention(
        self,
        state_k: torch.Tensor,
        counts: torch.Tensor,
        *,
        state_len: int,
    ) -> torch.Tensor:
        """Return transient normalized key sums for coarse attention kernels."""
        if getattr(self, "mla_state_key_normalization", "none") == "none":
            return state_k
        active_counts = counts[..., :state_len, :]
        mean_key = self._mean(state_k[..., :state_len, :], active_counts)
        normalized_mean = self._mla_normalize_key(mean_key, state_centroid=True)
        normalized_sum = normalized_mean * active_counts.to(normalized_mean.dtype)
        if state_len == int(state_k.size(2)):
            return normalized_sum
        output = torch.zeros_like(state_k)
        output[..., :state_len, :].copy_(normalized_sum)
        return output

    def _state_clustering_key(
        self,
        key: torch.Tensor,
        query_scale: torch.Tensor | None = None,
        *,
        role: str = "leaf",
        radial_rms: torch.Tensor | None = None,
        purpose: str = "assignment",
    ) -> torch.Tensor:
        """Map stored keys into the transient geometry used for clustering.

        The attention state continues to hold exact sums in its native key
        space.  This mapping only affects leaf-to-centroid assignment, so it
        can make clusters more unimodal without increasing persistent state or
        changing closed-centroid attention arithmetic.
        """
        clustering_key = self._mla_normalize_key(
            key,
            state_centroid=role == "centroid",
        )
        if role not in {"leaf", "centroid"}:
            raise ValueError("state clustering role must be leaf or centroid")
        if purpose not in {"append", "assignment"}:
            raise ValueError("state clustering purpose must be append or assignment")
        if query_scale is not None:
            if int(query_scale.size(-2)) == int(key.size(-1)):
                clustering_key = torch.matmul(
                    clustering_key.float(), query_scale.float()
                )
            else:
                clustering_key = clustering_key * query_scale
        fast_pairs = int(self.state_clustering_rope_fast_pairs)
        if fast_pairs:
            rope_dim = int(self.state_clustering_rope_dim)
            if rope_dim > int(clustering_key.size(-1)):
                raise ValueError(
                    "state-clustering RoPE dimension exceeds the attention head"
                )
            half = rope_dim // 2
            clustering_key = clustering_key.clone()
            clustering_key[..., :fast_pairs] = 0
            clustering_key[..., half : half + fast_pairs] = 0
        centroid_rescale = self.state_clustering_centroid_rescale
        if centroid_rescale == "direction_l2" and role == "leaf":
            leaf_rms = (
                clustering_key.float()
                .square()
                .mean(dim=-1, keepdim=True)
                .sqrt()
                .clamp_min(1e-12)
            )
            clustering_key = clustering_key.float() / leaf_rms
        if centroid_rescale != "none" and role == "centroid":
            if radial_rms is None:
                raise ValueError(
                    "centroid rescaling is missing its mean constituent RMS"
                )
            if centroid_rescale == "mean_leaf_norm":
                centroid_rms = (
                    clustering_key.float()
                    .square()
                    .mean(dim=-1, keepdim=True)
                    .clamp_min(1e-12)
                    .sqrt()
                )
                clustering_key = (
                    clustering_key.float()
                    / centroid_rms
                    * radial_rms.float().clamp_min(1e-12)
                )
            elif centroid_rescale in {
                "coherence",
                "spherical_coherence",
                "rope_coherence",
                "direction_l2",
            }:
                # RMS(mean key) / mean(RMS(key)) is the directional
                # resultant length.  This removes genuine per-slot radial
                # scale without discarding centroid representativeness.
                use_coherence = (
                    centroid_rescale == "direction_l2"
                    or self.state_clustering_centroid_rescale_scope == "all"
                    or self.state_clustering_centroid_rescale_scope == purpose
                )
                centroid_rms = (
                    clustering_key.float()
                    .square()
                    .mean(dim=-1, keepdim=True)
                    .sqrt()
                    .clamp_min(1e-12)
                )
                if centroid_rescale == "rope_coherence" and use_coherence:
                    rope_dim = int(self.state_clustering_rope_dim)
                    if not 0 < rope_dim < int(clustering_key.size(-1)):
                        raise ValueError(
                            "rope_coherence requires a nonempty partial-RoPE band"
                        )
                    rope_centroid_rms = (
                        clustering_key[..., :rope_dim]
                        .float()
                        .square()
                        .mean(dim=-1, keepdim=True)
                        .sqrt()
                    )
                    coherence = rope_centroid_rms / radial_rms.float().clamp_min(1e-12)
                    clustering_key = clustering_key.float() / centroid_rms * coherence
                else:
                    denominator = radial_rms.float() if use_coherence else centroid_rms
                    clustering_key = clustering_key.float() / denominator.clamp_min(
                        1e-12
                    )
            else:
                raise ValueError(
                    f"unsupported centroid rescaling mode: {centroid_rescale}"
                )
        normalize = self.state_clustering_normalization in {
            "cosine",
            f"{role}_cosine",
        } or (centroid_rescale == "spherical_coherence" and role == "leaf")
        if normalize:
            # Ordinary MHA has an independently normalized key space for each
            # head.  MLA shares one latent state across all query heads, so use
            # the mean of those per-head cosine objectives rather than one
            # global norm that lets high-norm projected heads dominate.
            rms = (
                clustering_key.float()
                .square()
                .mean(dim=-1, keepdim=True)
                .clamp_min(1e-12)
                .sqrt()
            )
            inverse_rms = rms.reciprocal()
            clustering_key = clustering_key.float() * inverse_rms
            if self.state_clustering_radial_bias:
                # The extra coordinate is transient: stored state remains the
                # exact raw key sum.  log(RMS) makes radial separation
                # dimensionless and symmetric under reciprocal norm changes.
                if radial_rms is None:
                    route_rms = rms
                else:
                    if tuple(radial_rms.shape) != tuple(rms.shape):
                        raise ValueError(
                            "state-clustering radial RMS has the wrong shape"
                        )
                    route_rms = radial_rms.float().clamp_min(1e-12)
                clustering_key = torch.cat((clustering_key, route_rms.log()), dim=-1)
        elif self.state_clustering_normalization not in {
            "none",
            "leaf_cosine",
            "centroid_cosine",
            "l2",
        }:
            raise ValueError(
                "state clustering normalization must be none, leaf_cosine, "
                "centroid_cosine, cosine, or l2"
            )
        return clustering_key.to(key.dtype)

    def _state_clustering_similarity(
        self,
        leaf_key: torch.Tensor,
        centroid_key: torch.Tensor,
        *,
        purpose: str = "assignment",
    ) -> torch.Tensor:
        if purpose not in {"append", "assignment"}:
            raise ValueError("state clustering purpose must be append or assignment")
        radial_scope = self.state_clustering_radial_scope
        use_radial = bool(self.state_clustering_radial_bias) and (
            radial_scope == "all" or radial_scope == purpose
        )
        if use_radial:
            direction_dim = int(leaf_key.size(-1)) - 1
            similarity = torch.matmul(
                leaf_key[..., :direction_dim],
                centroid_key[..., :direction_dim].transpose(-1, -2),
            )
            log_norm_distance = (
                leaf_key[..., direction_dim].float().unsqueeze(-1)
                - centroid_key[..., direction_dim].float().unsqueeze(-2)
            ).abs()
            # RMS-normalized vectors have squared norm direction_dim, so this
            # is direction_dim * (cosine - bias * abs(log(norm ratio))).
            similarity = similarity.float() - (
                float(self.state_clustering_radial_bias)
                * direction_dim
                * log_norm_distance
            )
        elif self.state_clustering_radial_bias:
            direction_dim = int(leaf_key.size(-1)) - 1
            similarity = torch.matmul(
                leaf_key[..., :direction_dim],
                centroid_key[..., :direction_dim].transpose(-1, -2),
            )
        else:
            similarity = torch.matmul(leaf_key, centroid_key.transpose(-1, -2))
        if self.state_clustering_centroid_rescale == "direction_l2":
            direction_dim = int(centroid_key.size(-1))
            centroid_squared_radius = (
                centroid_key.float().square().mean(dim=-1).unsqueeze(-2)
            )
            # With RMS-normalized leaves and m=sum(k)/sum(RMS(k)), this is
            # d * (u dot m - ||m||^2/2), hence exactly nearest-centroid
            # assignment in normalized-key space up to a leaf-only constant.
            similarity = similarity.float() - (
                0.5 * direction_dim * centroid_squared_radius
            )
        if self.state_clustering_normalization == "l2":
            # Negative squared distance is the exact assignment objective for
            # centroids that remain arithmetic means. With a query-metric
            # transform this is Mahalanobis distance in expected logit space.
            similarity = (
                2 * similarity
                - leaf_key.float().square().sum(-1, keepdim=True)
                - centroid_key.float().square().sum(-1).unsqueeze(-2)
            )
        return similarity

    def _state_clustering_constituent_rms(self, key: torch.Tensor) -> torch.Tensor:
        """Return the one scalar accumulated per clustering constituent."""
        radial_key = key.detach()
        if self.state_clustering_centroid_rescale == "rope_coherence":
            rope_dim = int(self.state_clustering_rope_dim)
            if not 0 < rope_dim < int(key.size(-1)):
                raise ValueError("rope_coherence requires a nonempty partial-RoPE band")
            radial_key = radial_key[..., :rope_dim]
        if radial_key.is_cuda and radial_key.ndim == 4 and radial_key.stride(-1) == 1:
            return constituent_rms(radial_key)
        return radial_key.float().square().mean(dim=-1, keepdim=True).sqrt()

    def _streaming_state_geometry(self) -> str | None:
        """Return a geometry supported by the fused centroid-scan kernel."""
        if (
            self.state_clustering_radial_bias
            or self.state_clustering_query_metric != "none"
            or getattr(self, "mla_state_key_normalization", "none") != "none"
            or self.state_clustering_rope_fast_pairs
        ):
            return None
        normalization = self.state_clustering_normalization
        rescale = self.state_clustering_centroid_rescale
        if normalization == "cosine" and rescale == "none":
            return "spherical"
        if (
            normalization == "none"
            and rescale in {"coherence", "spherical_coherence"}
            and self.state_clustering_centroid_rescale_scope == "assignment"
        ):
            return rescale
        if normalization == "none" and rescale == "none":
            return "raw"
        return None

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
            left_slots = (
                torch.arange(
                    protected,
                    protected + left_count,
                    device=key_sum.device,
                    dtype=torch.long,
                )
                .view(1, left_count)
                .expand(rows, left_count)
            )
            right_slots = (
                torch.arange(
                    protected + left_count,
                    current_len,
                    device=key_sum.device,
                    dtype=torch.long,
                )
                .view(1, right_count)
                .expand(rows, right_count)
            )
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

        if protected == 0 and current_len % 2 == 0 and target_len * 2 == current_len:

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
                    assignment.reshape(batch, heads, current_len, 1).expand_as(values),
                    values,
                )

            assignment = torch.empty(
                rows,
                current_len,
                dtype=torch.long,
                device=key_sum.device,
            )
            compact = (
                torch.arange(target_len, dtype=torch.long, device=key_sum.device)
                .view(1, target_len)
                .expand(rows, target_len)
            )
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
                raise ValueError("union overflow block size must be positive and even")
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
                        values.reshape(batch, full_blocks, heads, reduced_block, dim)
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
        expanded_k = torch.cat((state_k[..., :current_state_len, :], overflow_k), dim=2)
        expanded_v = torch.cat((state_v[..., :current_state_len, :], overflow_v), dim=2)
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
            union_assignment = torch.gather(round_assignment, 2, union_assignment)
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
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            torch.Tensor,
            torch.Tensor | None,
        ]
        | None
    ):
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
                full_target = max(block_size // 2, math.ceil(block_size * ratio))
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
                torch.arange(protected, device=state_k.device, dtype=torch.long)
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
        key_norm_sums: torch.Tensor | None,
        overflow_k: torch.Tensor,
        overflow_v: torch.Tensor,
        *,
        state_len: int,
        ctx_len: int,
        available_context: int,
        state_capacity: int,
        clustering_query_scale: torch.Tensor | None = None,
        scheduled_state_len: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if self.state_premerge_factor not in {1, 2, 4, 8, 16, 32}:
            raise ValueError(
                "state premerge factor must be one, two, four, eight, sixteen, or "
                "thirty-two"
            )
        if self.state_premerge_factor > 1 and (
            self.overflow_bipartite_merge
            or self.state_union_bipartite
            or self.state_precompact_direct_append
        ):
            raise ValueError(
                "adjacent state premerge cannot be combined with another "
                "state precompaction policy"
            )
        if self.state_split_max_leaves is not None:
            if scheduled_state_len is None:
                raise ValueError(
                    "split state updates require the independent scheduled length"
                )
            if (
                self.overflow_bipartite_merge
                or self.state_union_bipartite
                or self.state_precompact_direct_append
                or self.state_merge_before_append
            ):
                raise ValueError(
                    "state posting-list splitting cannot be combined with state "
                    "precompaction or merge-before-append"
                )
        use_constituent_norms = self.state_clustering_centroid_rescale != "none"
        if use_constituent_norms:
            if key_norm_sums is None:
                raise ValueError("centroid rescaling is missing key-norm sums")
            if self.state_union_bipartite or self.state_precompact_direct_append:
                raise NotImplementedError(
                    "centroid rescaling does not support state precompaction"
                )
            if self.overflow_bipartite_merge:
                raise NotImplementedError(
                    "centroid rescaling does not support overflow precompaction"
                )
        elif key_norm_sums is not None:
            raise ValueError("unexpected key-norm sums for centroid radial routing")
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
        overflow_key_norm_sums = (
            self._state_clustering_constituent_rms(overflow_k)
            if use_constituent_norms
            else None
        )
        membership = None
        premerge_input_len = None
        if self.state_premerge_factor > 1:
            if overflow_k.is_cuda and overflow_v.is_cuda:
                premerge_input_len = int(overflow_k.size(2))
                overflow_k, overflow_v, overflow_counts = (
                    self._premerge_state_inputs_cuda(overflow_k, overflow_v)
                )
            else:
                overflow_k, overflow_v, overflow_counts, membership = (
                    _premerge_adjacent_state_inputs(
                        overflow_k,
                        overflow_v,
                        self.state_premerge_factor,
                    )
                )
            if overflow_key_norm_sums is not None:
                overflow_key_norm_sums = _sum_adjacent_groups(
                    overflow_key_norm_sums,
                    self.state_premerge_factor,
                )
        elif self.overflow_bipartite_merge:
            overflow_k, overflow_v, overflow_counts, membership = (
                self._reduce_overflow_balanced(overflow_k, overflow_v)
            )
        else:
            overflow_counts = torch.ones(
                *overflow_k.shape[:3],
                1,
                dtype=torch.float32,
                device=overflow_k.device,
            )

        def expand_premerged_owners(owners: torch.Tensor) -> torch.Tensor:
            if premerge_input_len is not None:
                return self._expand_premerged_owners_cuda(owners, premerge_input_len)
            if membership is not None:
                return owners.gather(2, membership)
            return owners

        overflow_len = int(overflow_k.size(2))
        current_state_len = state_len
        schedule_base = (
            current_state_len
            if self.state_split_max_leaves is None
            else int(scheduled_state_len)
        )
        scheduled_target = self._desired_state_len(
            ctx_len, available_context, schedule_base
        )
        n_append = min(max(scheduled_target - schedule_base, 0), overflow_len)
        # A fixed premerge can expose fewer atomic routing inputs than the
        # raw-token growth schedule asks us to append in this update.  The
        # state can only grow by the number of available groups; subsequent
        # updates will continue toward the scheduled target.
        desired_state_len = current_state_len + n_append
        if desired_state_len > state_capacity:
            raise RuntimeError(
                "scheduled centroid append exceeded state capacity: "
                f"required {desired_state_len}, capacity {state_capacity}"
            )
        if overflow_len and n_append == overflow_len:
            # Every premerged group is appended in its original order.  Avoid
            # constructing routing keys, gathering the complete overflow, and
            # scattering an identity ownership map merely to reproduce those
            # same tensors.  This is the common initial-cache build through
            # 64K for the fixed-adjacent premerge schedule.
            destination = slice(current_state_len, desired_state_len)
            state_k[..., destination, :].copy_(overflow_k)
            state_v[..., destination, :].copy_(overflow_v)
            counts[..., destination, :].copy_(overflow_counts)
            if key_norm_sums is not None:
                if overflow_key_norm_sums is None:
                    raise AssertionError("LOD append is missing key-norm sums")
                key_norm_sums[..., destination, :].copy_(overflow_key_norm_sums)
            owners = (
                torch.arange(
                    current_state_len,
                    desired_state_len,
                    dtype=torch.long,
                    device=overflow_k.device,
                )
                .view(1, 1, overflow_len)
                .expand(overflow_k.size(0), overflow_k.size(1), overflow_len)
            )
            return (
                state_k,
                state_v,
                counts,
                desired_state_len,
                expand_premerged_owners(owners),
                None,
            )
        overflow_select_k = (
            self._mean(overflow_k, overflow_counts)
            if self.state_premerge_factor > 1 or self.overflow_bipartite_merge
            else overflow_k
        )
        owners = torch.full(
            overflow_k.shape[:-1], -1, dtype=torch.long, device=overflow_k.device
        )
        overflow_route_k = self._state_clustering_key(
            overflow_select_k,
            clustering_query_scale,
        )
        streaming_geometry = self._streaming_state_geometry()
        # Geometry preparation is amortized by the batch dimension.  Keep the
        # dense BLAS path for batch one unless the fused scan was requested
        # explicitly; the kernel path wins for the batched serving workload.
        use_streaming_state_scan = bool(
            self.reuse_state_update_similarity
            and state_k.is_cuda
            and streaming_geometry is not None
            and (
                self.fused_state_maxsim
                or (streaming_geometry != "raw" and int(state_k.size(0)) > 1)
            )
        )
        current_state_route_k = None
        current_state_append_route_k = None
        if not use_streaming_state_scan:
            current_state_mean_k = self._mean(
                state_k.detach()[..., :current_state_len, :],
                counts[..., :current_state_len, :],
            )
            current_state_mean_rms = (
                self._mean(
                    key_norm_sums[..., :current_state_len, :],
                    counts[..., :current_state_len, :],
                )
                if key_norm_sums is not None
                else None
            )
            current_state_route_k = self._state_clustering_key(
                current_state_mean_k,
                clustering_query_scale,
                role="centroid",
                radial_rms=current_state_mean_rms,
                purpose="assignment",
            )
            current_state_append_route_k = self._state_clustering_key(
                current_state_mean_k,
                clustering_query_scale,
                role="centroid",
                radial_rms=current_state_mean_rms,
                purpose="append",
            )
        use_fused_state_update = self.fused_state_update or (
            self.auto_fused_state_update and int(state_k.size(0)) > 1
        )

        buffers = None
        old_route_scores = None
        old_route_indices = None
        maxsim_buffers = None
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

        if use_streaming_state_scan:
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
            prepared_identity = (
                int(state_k.data_ptr()),
                int(counts.data_ptr()),
                (int(key_norm_sums.data_ptr()) if key_norm_sums is not None else 0),
                current_state_len,
                streaming_geometry,
            )
            prepare_state_geometry = maxsim_buffers.get(
                "_prepared_identity"
            ) != prepared_identity or ctx_len <= int(
                maxsim_buffers.get("_prepared_context_len", -1)
            )
            state_maxsim_block_m = self.state_maxsim_block_m
            state_maxsim_block_n = self.state_maxsim_block_n
            state_maxsim_num_warps = self.state_maxsim_num_warps
            if (
                int(overflow_select_k.size(0)) == 1
                and overflow_len >= 8192
                and int(overflow_select_k.size(-1)) == 128
            ):
                # Long scheduler chunks have ample token-axis work but only a
                # handful of KV heads. This exact geometry is about five
                # percent faster for the D128 catch-up update.
                state_maxsim_block_m = 8
                state_maxsim_block_n = 32
                state_maxsim_num_warps = 8
            (
                old_route_scores,
                old_route_indices,
                append_select_scores,
            ) = streaming_state_maxsim(
                overflow_route_k,
                state_k,
                counts,
                maxsim_buffers,
                state_len=current_state_len,
                sink_len=self._protected_state_len(current_state_len),
                key_norm_sums=key_norm_sums,
                geometry=streaming_geometry,
                block_m=state_maxsim_block_m,
                block_n=state_maxsim_block_n,
                num_warps=state_maxsim_num_warps,
                prepare_state_geometry=prepare_state_geometry,
                # Prepared spherical/coherence geometry is fastest as a dense
                # MFMA scan at batch 8. ``fused_state_maxsim`` remains useful
                # for raw geometry, but its long state-axis loop is slower for
                # these prepared views.
                materialize_prepared_scores=(streaming_geometry != "raw"),
                coherence_single_matmul=(
                    self.coherence_single_matmul
                    and streaming_geometry in {"coherence", "spherical_coherence"}
                ),
                # Ordinary unpadded construction keeps every slot below
                # ``current_state_len`` live by construction. Avoid writing
                # the full overflow-by-state score matrix merely to apply an
                # all-false mask; padded prefill retains the exact mask.
                mask_invalid_state=bool(getattr(self, "_lod_padding_state_reserve", 0)),
            )
            append_idx, merge_idx = self._split_append_merge_indices(
                append_select_scores, n_append
            )
        elif self.reuse_state_update_similarity and state_k.is_cuda:
            with torch.no_grad():
                if (
                    current_state_route_k is None
                    or current_state_append_route_k is None
                ):
                    raise AssertionError("LOD state route geometry is missing")
                old_similarity = self._state_clustering_similarity(
                    overflow_route_k,
                    current_state_route_k,
                    purpose="assignment",
                )
                invalid_state = counts[..., :current_state_len, 0].le(0.5).unsqueeze(-2)
                old_similarity.masked_fill_(invalid_state, float("-inf"))
                protected_slots = self._protected_state_len(current_state_len)
                if protected_slots:
                    protected_scores = (
                        old_similarity[..., :protected_slots].float().max(dim=-1).values
                    )
                    old_similarity[..., :protected_slots] = float("-inf")
                else:
                    protected_scores = torch.full_like(
                        old_similarity[..., 0].float(), float("-inf")
                    )
                old_route_scores, old_route_indices = old_similarity.max(dim=-1)
                radial_scope = self.state_clustering_radial_scope
                append_uses_radial = bool(self.state_clustering_radial_bias) and (
                    radial_scope in {"all", "append"}
                )
                assignment_uses_radial = bool(
                    self.state_clustering_radial_bias
                ) and radial_scope in {"all", "assignment"}
                centroid_scope = self.state_clustering_centroid_rescale_scope
                use_scoped_coherence = self.state_clustering_centroid_rescale in {
                    "coherence",
                    "spherical_coherence",
                    "rope_coherence",
                }
                append_uses_coherence = use_scoped_coherence and centroid_scope in {
                    "all",
                    "append",
                }
                assignment_uses_coherence = use_scoped_coherence and centroid_scope in {
                    "all",
                    "assignment",
                }
                if (
                    append_uses_radial == assignment_uses_radial
                    and append_uses_coherence == assignment_uses_coherence
                ):
                    append_route_scores = old_route_scores.float()
                    append_protected_scores = protected_scores
                else:
                    append_similarity = self._state_clustering_similarity(
                        overflow_route_k,
                        current_state_append_route_k,
                        purpose="append",
                    )
                    append_similarity.masked_fill_(invalid_state, float("-inf"))
                    if protected_slots:
                        append_protected_scores = (
                            append_similarity[..., :protected_slots]
                            .float()
                            .max(dim=-1)
                            .values
                        )
                        append_similarity[..., :protected_slots] = float("-inf")
                    else:
                        append_protected_scores = torch.full_like(
                            append_similarity[..., 0].float(), float("-inf")
                        )
                    append_route_scores = append_similarity.max(dim=-1).values
                append_select_scores = torch.maximum(
                    append_route_scores.float(), append_protected_scores
                )
                append_idx, merge_idx = self._split_append_merge_indices(
                    append_select_scores, n_append
                )
        elif (
            n_append
            and self.state_append_subblock_size <= 0
            and self.state_clustering_normalization != "l2"
            and not self.state_clustering_radial_bias
        ):
            append_idx, merge_idx = _split_append_merge_idx_by_maxsim(
                overflow_route_k,
                n_append,
                current_state_append_route_k,
            )
        elif n_append and self.state_append_subblock_size <= 0:
            append_select_scores = (
                self._state_clustering_similarity(
                    overflow_route_k,
                    current_state_append_route_k,
                    purpose="append",
                )
                .max(dim=-1)
                .values
            )
            append_idx, merge_idx = self._split_append_merge_indices(
                append_select_scores, n_append
            )
        elif not n_append:
            merge_idx = _all_idx(overflow_k, overflow_len)
            append_idx = merge_idx[..., :0]
        else:
            with torch.no_grad():
                append_select_scores = (
                    self._state_clustering_similarity(
                        overflow_route_k,
                        current_state_append_route_k,
                        purpose="append",
                    )
                    .max(dim=-1)
                    .values
                )
                append_idx, merge_idx = self._split_append_merge_indices(
                    append_select_scores, n_append
                )

        def refresh_prepared_geometry(
            changed_slots: torch.Tensor,
            active_state_len: int,
        ) -> None:
            if (
                not use_streaming_state_scan
                or streaming_geometry == "raw"
                or maxsim_buffers is None
            ):
                return
            prepare_state_clustering_keys(
                state_k,
                counts,
                maxsim_buffers,
                state_len=active_state_len,
                key_norm_sums=key_norm_sums,
                geometry=streaming_geometry,
                slot_indices=changed_slots,
                prepare_coherence_route=not (
                    self.coherence_single_matmul
                    and streaming_geometry in {"coherence", "spherical_coherence"}
                ),
                prepare_coherence_append=True,
                prepare_coherence_scale=(
                    self.coherence_single_matmul
                    and streaming_geometry in {"coherence", "spherical_coherence"}
                ),
            )
            maxsim_buffers["_prepared_identity"] = (
                int(state_k.data_ptr()),
                int(counts.data_ptr()),
                (int(key_norm_sums.data_ptr()) if key_norm_sums is not None else 0),
                active_state_len,
                streaming_geometry,
            )
            maxsim_buffers["_prepared_context_len"] = ctx_len

        if n_append:
            append_k = _gather_by_idx(overflow_k, append_idx)
            append_v = _gather_by_idx(overflow_v, append_idx)
            append_counts = _gather_by_idx(overflow_counts, append_idx)
            append_key_norm_sums = (
                _gather_by_idx(overflow_key_norm_sums, append_idx)
                if overflow_key_norm_sums is not None
                else None
            )
            append_select_k = self._state_clustering_key(
                self._mean(append_k, append_counts),
                clustering_query_scale,
                role="centroid",
                radial_rms=(
                    self._mean(append_key_norm_sums, append_counts)
                    if append_key_norm_sums is not None
                    else None
                ),
                purpose="assignment",
            )
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
                counts[..., current_state_len:desired_state_len, :].copy_(append_counts)
                if key_norm_sums is not None:
                    key_norm_sums[..., current_state_len:desired_state_len, :].copy_(
                        append_key_norm_sums
                    )
                owners.scatter_(2, append_idx, append_slots)
            merge_k = _gather_by_idx(overflow_k, merge_idx)
            merge_v = _gather_by_idx(overflow_v, merge_idx)
            merge_select_k = _gather_by_idx(overflow_route_k, merge_idx)
            merge_counts = _gather_by_idx(overflow_counts, merge_idx)
            merge_key_norm_sums = (
                _gather_by_idx(overflow_key_norm_sums, merge_idx)
                if overflow_key_norm_sums is not None
                else None
            )
        else:
            merge_k = overflow_k
            merge_v = overflow_v
            merge_select_k = overflow_route_k
            merge_counts = overflow_counts
            merge_key_norm_sums = overflow_key_norm_sums

        if int(merge_k.size(2)) == 0:
            if n_append and self.state_merge_before_append:
                state_k[..., current_state_len:desired_state_len, :].copy_(append_k)
                state_v[..., current_state_len:desired_state_len, :].copy_(append_v)
                counts[..., current_state_len:desired_state_len, :].copy_(append_counts)
                if key_norm_sums is not None:
                    key_norm_sums[..., current_state_len:desired_state_len, :].copy_(
                        append_key_norm_sums
                    )
                owners.scatter_(2, append_idx, append_slots)
            refresh_prepared_geometry(
                (append_slots if n_append else owners[..., :0]),
                desired_state_len,
            )
            owners = expand_premerged_owners(owners)
            return state_k, state_v, counts, desired_state_len, owners, None

        with torch.no_grad():
            if old_route_scores is not None and old_route_indices is not None:
                merge_old_scores = old_route_scores.gather(2, merge_idx)
                destination = old_route_indices.gather(2, merge_idx)
                destination_scores = merge_old_scores
                if n_append and not self.state_merge_before_append:
                    appended_logits = self._state_clustering_similarity(
                        merge_select_k,
                        append_select_k.detach(),
                        purpose="assignment",
                    )
                    appended_scores, appended_relative = appended_logits.max(dim=-1)
                    appended_destination = appended_relative + current_state_len
                    use_appended = appended_scores > merge_old_scores
                    destination = torch.where(
                        use_appended, appended_destination, destination
                    )
                    destination_scores = torch.where(
                        use_appended, appended_scores, destination_scores
                    )
            else:
                route_state_len = (
                    current_state_len
                    if self.state_merge_before_append
                    else desired_state_len
                )
                protected_slots = self._protected_state_len(route_state_len)
                route_logits = self._state_clustering_similarity(
                    merge_select_k,
                    self._state_clustering_key(
                        self._mean(
                            state_k.detach()[..., :route_state_len, :],
                            counts[..., :route_state_len, :],
                        ),
                        clustering_query_scale,
                        role="centroid",
                        radial_rms=(
                            self._mean(
                                key_norm_sums[..., :route_state_len, :],
                                counts[..., :route_state_len, :],
                            )
                            if key_norm_sums is not None
                            else None
                        ),
                        purpose="assignment",
                    ),
                    purpose="assignment",
                )
                route_logits.masked_fill_(
                    counts[..., :route_state_len, 0].le(0.5).unsqueeze(-2),
                    float("-inf"),
                )
                route_logits[..., :protected_slots] = float("-inf")
                destination_scores, destination = route_logits.max(dim=-1)

        route_state_len = (
            current_state_len if self.state_merge_before_append else desired_state_len
        )
        destination, updated_state_len = self._split_overfull_state_destinations(
            destination,
            destination_scores,
            merge_select_k,
            counts,
            state_len=route_state_len,
            state_capacity=state_capacity,
        )
        if updated_state_len > route_state_len:
            state_k[..., route_state_len:updated_state_len, :].zero_()
            state_v[..., route_state_len:updated_state_len, :].zero_()
            counts[..., route_state_len:updated_state_len, :].zero_()
            if key_norm_sums is not None:
                key_norm_sums[..., route_state_len:updated_state_len, :].zero_()
        assignment_t = (
            F.one_hot(destination, num_classes=updated_state_len)
            .float()
            .transpose(-1, -2)
            if not (use_fused_state_update and state_k.is_cuda)
            else None
        )
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
                active_slots=updated_state_len,
                key_norm_sums=key_norm_sums,
                merge_key_norm_sums=merge_key_norm_sums,
            )
        else:
            if assignment_t is None:
                raise AssertionError("LOD dense state assignment is missing")
            state_k[..., :updated_state_len, :].add_(
                torch.matmul(assignment_t.to(merge_k.dtype), merge_k)
            )
            state_v[..., :updated_state_len, :].add_(
                torch.matmul(assignment_t.to(merge_v.dtype), merge_v)
            )
            counts[..., :updated_state_len, :].add_(
                torch.matmul(assignment_t.float(), merge_counts.float())
            )
            owners.scatter_(2, merge_idx, destination)
        if key_norm_sums is not None:
            if not (use_fused_state_update and state_k.is_cuda):
                if assignment_t is None or merge_key_norm_sums is None:
                    raise AssertionError("LOD key-norm assignment is missing")
                key_norm_sums[..., :updated_state_len, :].add_(
                    torch.matmul(assignment_t.float(), merge_key_norm_sums.float())
                )
        if n_append and self.state_merge_before_append:
            state_k[..., current_state_len:desired_state_len, :].copy_(append_k)
            state_v[..., current_state_len:desired_state_len, :].copy_(append_v)
            counts[..., current_state_len:desired_state_len, :].copy_(append_counts)
            if key_norm_sums is not None:
                key_norm_sums[..., current_state_len:desired_state_len, :].copy_(
                    append_key_norm_sums
                )
            owners.scatter_(2, append_idx, append_slots)
        changed_slots = (
            torch.cat((destination, append_slots), dim=-1) if n_append else destination
        )
        refresh_prepared_geometry(changed_slots, updated_state_len)
        owners = expand_premerged_owners(owners)
        return state_k, state_v, counts, updated_state_len, owners, None

    def _state_route_logits(
        self,
        q: torch.Tensor,
        state_k: torch.Tensor,
        counts: torch.Tensor,
        *,
        state_len: int,
    ) -> torch.Tensor:
        timing_events = getattr(self, "_lod_phase_timing_events", None)
        profile_prefill = bool(
            isinstance(timing_events, dict) and q.is_cuda and int(q.size(2)) > 1
        )
        mean_begin = None
        mean_end = None
        if profile_prefill:
            mean_begin = torch.cuda.Event(enable_timing=True)
            mean_end = torch.cuda.Event(enable_timing=True)
            mean_begin.record()
        mean_k = self._mean(
            state_k.detach()[..., :state_len, :],
            counts[..., :state_len, :],
        )
        mean_k = self._mla_normalize_key(mean_k, state_centroid=True)
        if mean_end is not None:
            mean_end.record()
        qk_end = None
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
            logits = torch.matmul(grouped_q, grouped_k_t).reshape(
                batch, query_heads, query_len, state_len
            )
        else:
            logits = torch.matmul(q.detach(), self._repeat_kv(mean_k).transpose(-1, -2))
        if mean_end is not None:
            qk_end = torch.cuda.Event(enable_timing=True)
            qk_end.record()
            timing_events.setdefault("route_mean_k", []).append((mean_begin, mean_end))
            timing_events.setdefault("route_qk", []).append((mean_end, qk_end))
        return logits

    @staticmethod
    def _routing_rms(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.detach()
            .float()
            .square()
            .mean(dim=-1, keepdim=True)
            .clamp_min(1e-12)
            .sqrt()
        )

    def _routing_rms_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        inverse_rms = self._routing_rms(tensor).reciprocal()
        # Keep route logits in the same dtype as the normal fused path.  The
        # normalization statistics themselves are computed in FP32.
        return (tensor.detach().float() * inverse_rms).to(tensor.dtype)

    def _state_routing_logits(
        self,
        q: torch.Tensor,
        state_k: torch.Tensor,
        counts: torch.Tensor,
        *,
        state_len: int,
    ) -> torch.Tensor:
        normalization = self.routing_normalization
        if normalization == "qk_norm_aware":
            raise ValueError(
                "qk_norm_aware routing must be resolved from the attention "
                "module by the Hugging Face installer"
            )
        rope_fast_pairs = int(self.routing_rope_fast_pairs)
        if normalization == "none" and rope_fast_pairs == 0:
            return self._state_route_logits(q, state_k, counts, state_len=state_len)
        if normalization not in {"none", "query", "key", "both"}:
            raise ValueError("routing normalization must be none, query, key, or both")
        route_q = q.detach()
        # For query-only normalization, ranking
        #   scale * (q / rms(q)) @ mean_k + log(count)
        # is exactly equivalent to ranking
        #   scale * q @ mean_k + rms(q) * log(count).
        # This makes routing invariant to the query's attention temperature
        # and replaces a model-wide count-bias sweep with a per-query value.
        mean_k = self._mean(
            state_k.detach()[..., :state_len, :],
            counts[..., :state_len, :],
        )
        mean_k = self._mla_normalize_key(mean_k, state_centroid=True)
        original_query_rms = None
        if rope_fast_pairs:
            rope_dim = int(self.routing_rope_dim)
            if rope_dim > int(q.size(-1)) or rope_fast_pairs > rope_dim // 2:
                raise ValueError("routing RoPE filter exceeds the head geometry")
            if normalization not in {"query", "both"}:
                original_query_rms = (
                    route_q.detach().float().square().mean(-1, keepdim=True).sqrt()
                )
            half = rope_dim // 2
            route_q = route_q.clone()
            mean_k = mean_k.clone()
            route_q[..., :rope_fast_pairs] = 0
            route_q[..., half : half + rope_fast_pairs] = 0
            mean_k[..., :rope_fast_pairs] = 0
            mean_k[..., half : half + rope_fast_pairs] = 0
        if normalization in {"query", "both"}:
            route_q = self._routing_rms_normalize(route_q)
        elif original_query_rms is not None:
            route_q = (self._routing_rms_normalize(route_q) * original_query_rms).to(
                route_q.dtype
            )
        if normalization in {"key", "both"}:
            mean_k = self._routing_rms_normalize(mean_k)
        if self.route_gqa_matmul:
            batch, query_heads, query_len, head_dim = route_q.shape
            kv_heads = int(mean_k.size(1))
            grouped_q = route_q.reshape(
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
        return torch.matmul(route_q, self._repeat_kv(mean_k).transpose(-1, -2))

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
        local_len: int | None = None,
        new_k: torch.Tensor | None = None,
        page_cache: dict[str, torch.Tensor | int] | None = None,
        dynamic_local_lse: torch.Tensor | None = None,
        context_len: int | None = None,
    ) -> torch.Tensor:
        """Select the fixed top-four release routes.

        Prefill uses the exact hierarchical selector and preserves its logits
        for the stable coarse-state recomputation. Decode ordinarily routes
        inside the fused paged kernel; the scalar fallback below keeps the same
        score definition for unsupported launch shapes.
        """
        del (
            state_capacity,
            local_k,
            local_v,
            local_len,
            new_k,
            page_cache,
            dynamic_local_lse,
            context_len,
        )
        query_len = int(q.size(2))
        configured_topk = (
            self.prefill_two_level_topk
            if query_len > 1 and self.prefill_two_level_topk is not None
            else self.two_level_topk
        )
        protected_len = (
            self._protected_route_len(state_len) if self.exclude_sink_from_routes else 0
        )
        route_count = min(int(configured_topk), state_len - protected_len)
        if route_count <= 0:
            return torch.empty(*q.shape[:3], 0, dtype=torch.long, device=q.device)
        if route_count not in (4, 8):
            raise RuntimeError("the LoD release requires exactly four routes")
        if self.routing_normalization not in {"none", "query"}:
            raise RuntimeError(
                "the LoD release supports only raw or query-normalized routing"
            )

        with torch.no_grad():
            if query_len > 1 and self.prefill_aiter_route_coarse:
                if protected_len != 0:
                    raise RuntimeError(
                        "AITER route/coarse prefill requires the separate sink cache"
                    )
                if route_count != 4 or int(q.size(-1)) > 256:
                    raise RuntimeError(
                        "AITER route/coarse prefill requires top-four equal-width "
                        "heads no wider than 256"
                    )
                if int(state_v.size(-1)) != int(q.size(-1)):
                    raise RuntimeError(
                        "AITER route/coarse prefill requires equal K/V head widths"
                    )
                if not self.split_prefill_local_attention:
                    raise RuntimeError(
                        "AITER route/coarse prefill requires split local attention"
                    )
                if getattr(self, "mla_state_key_normalization", "none") != "none":
                    raise RuntimeError(
                        "AITER route/coarse prefill does not support MLA key "
                        "normalization"
                    )
                from .kernels.aiter_prefill_attention import (
                    aiter_prefill_route_coarse_attention,
                )

                active_counts = counts[..., :state_len, :]
                mean_k = self._mean(
                    state_k.detach()[..., :state_len, :], active_counts
                ).contiguous()
                routed, coarse_output, coarse_lse = (
                    aiter_prefill_route_coarse_attention(
                        q.contiguous(),
                        mean_k,
                        state_v.contiguous(),
                        counts.contiguous(),
                        state_len=state_len,
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        normalize_route_query=self.routing_normalization == "query",
                    )
                )
                self._lod_prefill_fused_coarse = (
                    coarse_output,
                    coarse_lse,
                    False,
                )
                return routed

            logits = self._state_routing_logits(
                q,
                state_k,
                counts,
                state_len=state_len,
            )
            query_rms = (
                self._routing_rms(q) if self.routing_normalization == "query" else None
            )
            if query_len > 1:
                if not (
                    self.fused_prefill_route_coarse
                    and self.fused_prefill_stable_recompute
                    and self.fused_prefill_external_recompute
                    and self.prefill_hierarchical_route
                ):
                    raise RuntimeError(
                        "the LoD release requires hierarchical prefill routing "
                        "with stable external coarse recomputation"
                    )
                hierarchical_block_n = min(
                    1024,
                    max(256, 1 << (state_len - 1).bit_length()),
                )
                routed = route_logits_hierarchical_topk(
                    logits.contiguous(),
                    counts.detach().contiguous(),
                    state_len=state_len,
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                    route_count_bias=1.0,
                    topk=route_count,
                    protected_len=protected_len,
                    max_leaf_tokens=None,
                    block_m=8,
                    block_n=hierarchical_block_n,
                    tile_num_warps=2,
                    reduce_num_warps=2,
                )
                self._lod_prefill_route_logits = (
                    (logits, query_rms) if query_rms is not None else logits
                )
                return routed

            self._lod_decode_route_logits = (logits, query_rms)
            query_counts = self._repeat_kv(counts.detach()[..., :state_len, :]).squeeze(
                -1
            )
            scores = logits.float() * self.scaling + query_counts.log().unsqueeze(2)
            scores[..., :protected_len] = float("-inf")
            return scores.topk(route_count, dim=-1, sorted=False).indices

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
        destination: dict[str, torch.Tensor | int] | None = None,
    ) -> dict[str, torch.Tensor | int]:
        batch, kv_heads, _, head_dim = k.shape
        raw_page_key_summaries = bool(
            getattr(self, "mla_recursive_page_key_normalization", False)
        )
        flat_int8_mma = bool(self.prefill_int8_leaf_mma)
        if flat_int8_mma:
            if self.recursive_page_lod:
                raise ValueError(
                    "prefill INT8 leaf MMA does not support recursive leaves"
                )
            if self.leaf_layout != "expert":
                raise ValueError("prefill INT8 leaf MMA requires expert leaf layout")
            if self.leaf_key_quant_bits or self.leaf_value_quant_bits:
                raise ValueError(
                    "prefill INT8 leaf MMA is a native storage format and cannot "
                    "be combined with simulated leaf quantization"
                )
            if head_dim % 32 or int(v.size(-1)) % 32:
                raise ValueError(
                    "prefill INT8 leaf MMA requires dimensions divisible by 32"
                )
        if raw_page_key_summaries:
            if not self.recursive_page_lod:
                raise ValueError(
                    "recursive MLA page normalization requires recursive page LOD"
                )
            if self.leaf_key_quant_bits or self.leaf_value_quant_bits:
                raise ValueError(
                    "recursive raw MLA page summaries do not yet support quantization"
                )
            if self.routing_page_mass_candidates:
                raise ValueError(
                    "page-mass route refinement does not yet support raw MLA summaries"
                )
        page_size = self.leaf_page_size
        page_capacity = (
            sequence_capacity + page_size - 1
        ) // page_size + state_capacity
        direct_native_quant = bool(
            self.leaf_key_quant_bits in (4, 8)
            and self.leaf_value_quant_bits == self.leaf_key_quant_bits
        )
        if destination is not None and (
            not self.virtual_page_storage
            or flat_int8_mma
            or (
                bool(self.leaf_key_quant_bits or self.leaf_value_quant_bits)
                and not direct_native_quant
            )
        ):
            raise ValueError(
                "direct prefill storage requires BF16 or native quantized virtual pages"
            )

        def destination_tensor(name: str) -> torch.Tensor:
            if destination is None:
                raise AssertionError("direct prefill page destination is missing")
            value = destination.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"direct prefill page destination lacks {name}")
            if value.ndim and tuple(value.shape[:2]) != (batch, kv_heads):
                raise ValueError(
                    f"direct prefill page destination {name} has wrong batch/head shape"
                )
            return value

        paged_page_directory = bool(self.leaf_paged_directory)
        if (
            paged_page_directory
            and self.leaf_layout
            in ("aiter_varlen", "aiter_union", "aiter_masked_union")
            and not self.virtual_page_storage
        ):
            raise ValueError(
                "physical-page AITER varlen attention does not consume the "
                "two-level page directory; use virtual-page storage"
            )
        hash_capacity = 1 << max(
            1,
            (page_capacity * self.leaf_overflow_hash_factor - 1).bit_length(),
        )
        if paged_page_directory:
            maximum_slot_pages = max(
                1, (sequence_capacity + page_size - 1) // page_size
            )
            slot_page_capacity = max(
                1,
                (maximum_slot_pages + _PAGE_DIRECTORY_SIZE - 1) // _PAGE_DIRECTORY_SIZE,
            )
            # A root entry uses one of its physical K/V page IDs as the handle
            # for the corresponding directory page.  Matching the physical
            # page-pool capacity therefore gives a collision-free hard bound
            # without a separate directory allocator.
            directory_page_capacity = max(1, page_capacity)
            slot_page_dtype = torch.int32
            if destination is None:
                overflow_page_keys = torch.full(
                    (batch, kv_heads, 1),
                    -1,
                    dtype=torch.int32,
                    device=k.device,
                )
                overflow_page_values = torch.full(
                    (
                        batch,
                        kv_heads,
                        directory_page_capacity,
                        _PAGE_DIRECTORY_SIZE,
                    ),
                    -1,
                    dtype=torch.int32,
                    device=k.device,
                )
                overflow_used = torch.zeros((), dtype=torch.int32, device=k.device)
            else:
                overflow_page_keys = destination_tensor("overflow_page_keys")
                overflow_page_values = destination_tensor("overflow_page_values")
                overflow_used = destination_tensor("overflow_used")
        else:
            slot_page_capacity = self.leaf_inline_pages_per_slot
            slot_page_dtype = (
                torch.int16
                if page_capacity <= torch.iinfo(torch.int16).max
                else torch.int32
            )
            # Most contexts never overflow the compact inline posting lists.
            # Keep only a sentinel allocation until a slot actually needs the
            # hash table; kernels with HASH_PROBES=0 never dereference it.
            if destination is None:
                overflow_page_keys = torch.full(
                    (batch, kv_heads, 1),
                    -1,
                    dtype=torch.int32,
                    device=k.device,
                )
                overflow_page_values = torch.full(
                    (batch, kv_heads, 1),
                    -1,
                    dtype=torch.int32,
                    device=k.device,
                )
                overflow_used = torch.zeros((), dtype=torch.int32, device=k.device)
            else:
                overflow_page_keys = destination_tensor("overflow_page_keys")
                overflow_page_values = destination_tensor("overflow_page_values")
                overflow_used = destination_tensor("overflow_used")
        destination_slot_pages = (
            destination_tensor("slot_pages") if destination is not None else None
        )
        destination_slot_capacity = (
            int(destination_slot_pages.size(-1))
            if destination_slot_pages is not None
            else slot_page_capacity
        )
        cache: dict[str, torch.Tensor | int] = {
            # These pages are allocated from per-region postings below.  This
            # is a semantic contract, not a description of the physical flat
            # leaf backing used by virtual-page storage.
            "region_owned_pages": True,
            "slot_pages": (
                destination_slot_pages
                if destination_slot_pages is not None
                else torch.full(
                    (
                        batch,
                        kv_heads,
                        state_capacity,
                        slot_page_capacity,
                    ),
                    -1,
                    dtype=slot_page_dtype,
                    device=k.device,
                )
            ),
            "overflow_page_keys": overflow_page_keys,
            "overflow_page_values": overflow_page_values,
            "overflow_hash_capacity": (
                int(destination.get("overflow_hash_capacity", hash_capacity))
                if destination is not None
                else hash_capacity
            ),
            "overflow_flag": (
                destination_tensor("overflow_flag")
                if destination is not None
                else torch.zeros((), dtype=torch.int32, device=k.device)
            ),
            "overflow_used": overflow_used,
            "overflow_active": False,
            "overflow_safe_until": destination_slot_capacity * page_size,
            "paged_page_directory": paged_page_directory,
            "page_directory_size": _PAGE_DIRECTORY_SIZE,
            "slot_lengths": (
                destination_tensor("slot_lengths")
                if destination is not None
                else torch.zeros(
                    batch,
                    kv_heads,
                    state_capacity,
                    dtype=torch.int32,
                    device=k.device,
                )
            ),
            "next_page": (
                destination_tensor("next_page")
                if destination is not None
                else torch.zeros(batch, kv_heads, dtype=torch.int32, device=k.device)
            ),
            "page_size": page_size,
            "leaf_capacity": sequence_capacity,
            "leaf_count": 0,
            "leaf_lens": torch.zeros(
                batch, dtype=torch.int32, device=k.device
            ),
            "mla_raw_page_key_summaries": raw_page_key_summaries,
        }
        if self.virtual_page_storage:
            matching_native_quant = (
                self.leaf_key_quant_bits in (4, 8)
                and self.leaf_value_quant_bits == self.leaf_key_quant_bits
            )
            virtual_native_quant = matching_native_quant
            if self.leaf_key_quant_bits not in (0, 2, 3, 4, 8) or (
                self.leaf_value_quant_bits not in (0, 2, 3, 4, 8)
            ):
                raise ValueError(
                    "virtual page quantization supports 0, 2, 3, 4, or 8 bits"
                )
            if virtual_k is None or virtual_v is None:
                raise ValueError("virtual pages require the original prompt K/V")
            if virtual_k.shape[:3] != virtual_v.shape[:3]:
                raise ValueError("virtual prompt K/V shapes do not match")
            # Raw MLA latents are accumulated in the coarse state, but page
            # leaves remain exact model keys.  Materialize the model's
            # per-token latent normalization once when the virtual backing
            # store is created rather than on every leaf lookup.
            virtual_k = self._mla_normalize_key(virtual_k, state_centroid=False)
            if flat_int8_mma:
                flat_leaf_k = torch.empty(
                    batch,
                    kv_heads,
                    sequence_capacity,
                    head_dim,
                    dtype=torch.int8,
                    device=k.device,
                )
                flat_leaf_v = torch.empty(
                    batch,
                    kv_heads,
                    sequence_capacity,
                    int(virtual_v.size(-1)),
                    dtype=torch.int8,
                    device=v.device,
                )
                flat_leaf_capacity = sequence_capacity
            elif virtual_native_quant:
                # Native residual quantization consumes the current prompt K/V
                # directly and writes its persistent codes into the pool below.
                # Keeping these as views avoids creating a BF16 leaf shadow.
                flat_leaf_k = virtual_k.detach()
                flat_leaf_v = virtual_v.detach()
                flat_leaf_capacity = sequence_capacity
            elif destination is not None:
                flat_leaf_k = destination_tensor("leaf_k")
                flat_leaf_v = destination_tensor("leaf_v")
                if (
                    int(flat_leaf_k.size(2)) < sequence_capacity
                    or int(flat_leaf_v.size(2)) < sequence_capacity
                ):
                    raise ValueError("direct prefill leaf destination is too small")
                flat_leaf_capacity = int(flat_leaf_k.size(2))
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
                page_indices=(
                    destination_tensor("page_indices")
                    if destination is not None
                    else torch.full(
                        (batch, kv_heads, page_capacity, page_size),
                        -1,
                        dtype=torch.int32,
                        device=k.device,
                    )
                ),
                leaf_capacity=flat_leaf_capacity,
                leaf_quant_bits=(
                    self.leaf_key_quant_bits if virtual_native_quant else 0
                ),
                quantization_finalized=False,
            )
            if flat_int8_mma:
                cache.update(
                    page_k_token_scales=torch.empty(
                        batch,
                        kv_heads,
                        sequence_capacity,
                        dtype=k.dtype,
                        device=k.device,
                    ),
                    page_v_token_scales=torch.empty(
                        batch,
                        kv_heads,
                        sequence_capacity,
                        dtype=v.dtype,
                        device=v.device,
                    ),
                    prefill_int8_leaf_mma=True,
                )
            elif virtual_native_quant:
                group_size = self.leaf_quant_group_size
                token_groups = page_size // self.leaf_quant_token_group_size
                value_dim = int(virtual_v.size(-1))
                if head_dim % group_size or value_dim % group_size:
                    raise ValueError(
                        "virtual quantization group size must divide K/V dimensions"
                    )
                quant_bits = self.leaf_key_quant_bits
                quant_dtype = torch.uint8 if quant_bits == 4 else torch.int8
                key_width = head_dim // 2 if quant_bits == 4 else head_dim
                value_width = value_dim // 2 if quant_bits == 4 else value_dim
                cache.update(
                    quantized_leaf_k=(
                        destination_tensor("quantized_leaf_k")
                        if destination is not None
                        else torch.empty(
                            batch,
                            kv_heads,
                            sequence_capacity,
                            key_width,
                            dtype=quant_dtype,
                            device=k.device,
                        )
                    ),
                    quantized_leaf_v=(
                        destination_tensor("quantized_leaf_v")
                        if destination is not None
                        else torch.empty(
                            batch,
                            kv_heads,
                            sequence_capacity,
                            value_width,
                            dtype=quant_dtype,
                            device=v.device,
                        )
                    ),
                    page_k_scales=(
                        destination_tensor("page_k_scales")
                        if destination is not None
                        else torch.empty(
                            batch,
                            kv_heads,
                            page_capacity,
                            token_groups * (head_dim // group_size),
                            dtype=k.dtype,
                            device=k.device,
                        )
                    ),
                    page_v_scales=(
                        destination_tensor("page_v_scales")
                        if destination is not None
                        else torch.empty(
                            batch,
                            kv_heads,
                            page_capacity,
                            token_groups * (value_dim // group_size),
                            dtype=v.dtype,
                            device=v.device,
                        )
                    ),
                    page_quantized_counts=(
                        destination_tensor("page_quantized_counts")
                        if destination is not None
                        else torch.zeros(
                            batch,
                            kv_heads,
                            page_capacity,
                            dtype=torch.int32,
                            device=k.device,
                        )
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
                    dtype=torch.int8 if flat_int8_mma else k.dtype,
                    device=k.device,
                ),
                page_v=torch.zeros(
                    batch,
                    kv_heads,
                    page_capacity,
                    page_size,
                    int(v.size(-1)),
                    dtype=torch.int8 if flat_int8_mma else v.dtype,
                    device=v.device,
                ),
            )
            if flat_int8_mma:
                cache.update(
                    page_k_token_scales=torch.zeros(
                        batch,
                        kv_heads,
                        page_capacity,
                        page_size,
                        dtype=k.dtype,
                        device=k.device,
                    ),
                    page_v_token_scales=torch.zeros(
                        batch,
                        kv_heads,
                        page_capacity,
                        page_size,
                        dtype=v.dtype,
                        device=v.device,
                    ),
                    prefill_int8_leaf_mma=True,
                )
        needs_page_summaries = self.recursive_page_lod or bool(
            self.leaf_key_quant_bits or self.leaf_value_quant_bits
        )
        if needs_page_summaries:
            summary_page_capacity = (
                int(cache["page_indices"].size(2))
                if isinstance(cache.get("page_indices"), torch.Tensor)
                else page_capacity
            )
            cache.update(
                page_sum_k=(
                    destination_tensor("page_sum_k")
                    if destination is not None and not direct_native_quant
                    else torch.zeros(
                        batch,
                        kv_heads,
                        summary_page_capacity,
                        head_dim,
                        dtype=k.dtype,
                        device=k.device,
                    )
                ),
                page_sum_v=(
                    destination_tensor("page_sum_v")
                    if destination is not None and not direct_native_quant
                    else torch.zeros(
                        batch,
                        kv_heads,
                        summary_page_capacity,
                        int(v.size(-1)),
                        dtype=v.dtype,
                        device=v.device,
                    )
                ),
                page_counts=(
                    destination_tensor("page_counts")
                    if destination is not None
                    else torch.zeros(
                        batch,
                        kv_heads,
                        summary_page_capacity,
                        dtype=torch.int32,
                        device=k.device,
                    )
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
        if destination is not None and self.leaf_layout == "aiter_hilo":
            for name in (
                "unified_page1_k",
                "unified_page1_v",
                "unified_page1_bias",
                "unified_page1_leaf_offset",
                "unified_page1_coarse_offset",
                "unified_page1_row_offset",
            ):
                if name not in destination:
                    raise TypeError(f"fused AITER HiLo page destination lacks {name}")
                cache[name] = destination[name]
        self._append_page_cache(cache, k, v, owners)
        return cache

    @staticmethod
    def _grow_paged_page_directory(
        cache: dict[str, torch.Tensor | int],
        *,
        required_slots: int,
        required_pages: int,
    ) -> None:
        if not bool(cache.get("paged_page_directory", False)):
            return
        slot_pages = cache.get("slot_pages")
        directory = cache.get("overflow_page_values")
        if not isinstance(slot_pages, torch.Tensor) or not isinstance(
            directory, torch.Tensor
        ):
            raise TypeError("paged page-directory metadata is missing")
        if directory.ndim != 4 or int(directory.size(3)) != _PAGE_DIRECTORY_SIZE:
            raise ValueError("paged page directory has the wrong geometry")
        required_root_entries = max(
            1,
            (required_pages + _PAGE_DIRECTORY_SIZE - 1) // _PAGE_DIRECTORY_SIZE,
        )
        missing_root_entries = max(required_root_entries - int(slot_pages.size(3)), 0)
        if missing_root_entries:
            cache["slot_pages"] = F.pad(slot_pages, (0, missing_root_entries), value=-1)
        # Directory-page handles are physical page IDs, so matching the page
        # pool is an exact collision-free capacity bound.
        required_directory_pages = max(1, required_pages)
        missing_directory_pages = max(
            required_directory_pages - int(directory.size(2)), 0
        )
        if missing_directory_pages:
            cache["overflow_page_values"] = F.pad(
                directory,
                (0, 0, 0, missing_directory_pages),
                value=-1,
            )

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
        if bool(cache.get("paged_page_directory", False)):
            if not isinstance(slot_pages, torch.Tensor):
                raise TypeError("slot page-directory root is missing")
            TritonLODAttentionCore._grow_paged_page_directory(
                cache,
                required_slots=int(slot_pages.size(2)),
                required_pages=target,
            )
            slot_pages = cache.get("slot_pages")
        elif (
            isinstance(slot_pages, torch.Tensor)
            and slot_pages.dtype == torch.int16
            and target > torch.iinfo(torch.int16).max
        ):
            cache["slot_pages"] = slot_pages.to(torch.int32)
        if isinstance(page_indices, torch.Tensor):
            cache["page_indices"] = F.pad(page_indices, (0, 0, 0, missing), value=-1)
        else:
            cache["page_k"] = F.pad(page_k, (0, 0, 0, 0, 0, missing))
            cache["page_v"] = F.pad(page_v, (0, 0, 0, 0, 0, missing))
            for name in ("page_k_token_scales", "page_v_token_scales"):
                tensor = cache.get(name)
                if isinstance(tensor, torch.Tensor):
                    cache[name] = F.pad(tensor, (0, 0, 0, missing))
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
            cache["page_quantized_counts"] = F.pad(page_quantized_counts, (0, missing))

    @staticmethod
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
        if bool(cache.get("paged_page_directory", False)):
            page_indices = cache.get("page_indices")
            page_k = cache.get("page_k")
            page_storage = (
                page_indices if isinstance(page_indices, torch.Tensor) else page_k
            )
            if not isinstance(page_storage, torch.Tensor):
                raise TypeError("paged K/V storage is missing")
            TritonLODAttentionCore._grow_paged_page_directory(
                cache,
                required_slots=required_slots,
                required_pages=int(page_storage.size(2)),
            )

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

    def _page_lookup_probes(self, cache: dict[str, torch.Tensor | int]) -> int:
        if bool(cache.get("paged_page_directory", False)):
            return -1
        return self.leaf_hash_probes if bool(cache["overflow_active"]) else 0

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
        # Page leaves are exact tokens.  Only coarse state entries defer MLA
        # normalization until after their raw latent sum is averaged.
        k = self._mla_normalize_key(k, state_centroid=False)
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
            required_pages = (
                required_slots + (leaf_capacity + page_size - 1) // page_size
            )
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
        leaf_lens = cache.get("leaf_lens")
        if not isinstance(leaf_lens, torch.Tensor):
            raise TypeError("leaf-length tensor is missing")
        leaf_lens.fill_(leaf_count)
        slot_pages = cache["slot_pages"]
        next_page = cache["next_page"]
        if not isinstance(slot_pages, torch.Tensor):
            raise TypeError("slot page table is missing")
        if not isinstance(next_page, torch.Tensor):
            raise TypeError("next-page tensor is missing")
        page_indices = cache.get("page_indices")
        overflow_used = cache["overflow_used"]
        overflow_flag = cache["overflow_flag"]
        virtual_pages = isinstance(page_indices, torch.Tensor)
        if not virtual_pages:
            raise RuntimeError("the LoD release requires virtual indexed leaf pages")
        if virtual_pages and self.leaf_seal_capacity is not None:
            raise NotImplementedError(
                "sealed leaf archival is currently implemented for flat paged leaves"
            )
        if not isinstance(overflow_used, torch.Tensor):
            raise TypeError("overflow page-used flag is missing")
        if not isinstance(overflow_flag, torch.Tensor):
            raise TypeError("overflow page-table flag is missing")
        append_hash_probes = self._page_lookup_probes(cache)
        if append_hash_probes == 0 and leaf_count > int(cache["overflow_safe_until"]):
            inline_token_capacity = self.leaf_inline_pages_per_slot * page_size
            additions = torch.zeros_like(slot_lengths).scatter_add_(
                2,
                owners.long(),
                torch.ones_like(owners, dtype=slot_lengths.dtype),
            )
            projected_max = int((slot_lengths + additions).max().item())
            if self.leaf_seal_capacity is not None:
                projected_max = min(projected_max, self.leaf_seal_capacity)
            cache["overflow_active"] = projected_max > inline_token_capacity
            if bool(cache["overflow_active"]):
                append_hash_probes = self.leaf_hash_probes
            else:
                cache["overflow_safe_until"] = leaf_count + (
                    inline_token_capacity - projected_max
                )
        if append_hash_probes > 0:
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
            if not all(isinstance(value, torch.Tensor) for value in (leaf_k, leaf_v)):
                raise TypeError("virtual page cache tensors are incomplete")
            summaries = (page_sum_k, page_sum_v, page_counts)
            if any(isinstance(value, torch.Tensor) for value in summaries) and not all(
                isinstance(value, torch.Tensor) for value in summaries
            ):
                raise TypeError("virtual page summary tensors are incomplete")
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
                    raise RuntimeError(
                        "finalized virtual quantized cache is incomplete"
                    )
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
                    quant_token_group_size=self.leaf_quant_token_group_size,
                    quant_bits=int(cache.get("leaf_quant_bits", 4)),
                    quantized_page_sum_k=cache.get("quantized_page_sum_k"),
                    quantized_page_sum_v=cache.get("quantized_page_sum_v"),
                    page_sum_k_scales=cache.get("page_sum_k_scales"),
                    page_sum_v_scales=cache.get("page_sum_v_scales"),
                    optimize_summary_scale=(self.page_summary_scale_mode == "l2"),
                    optimize_leaf_scale=(self.leaf_append_quant_scale_mode == "l2"),
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
                )

    def _paged_leaf_attention(
        self,
        q: torch.Tensor,
        top_slots: torch.Tensor,
        cache: dict[str, torch.Tensor | int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_pages = cache["slot_pages"]
        slot_lengths = cache["slot_lengths"]
        page_indices = cache.get("page_indices")
        indexed = isinstance(page_indices, torch.Tensor)
        if not indexed:
            raise RuntimeError("the LoD release requires virtual indexed leaf pages")
        page_k = cache["leaf_k" if indexed else "page_k"]
        page_v = cache["leaf_v" if indexed else "page_v"]
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
        if self.leaf_layout != "expert":
            raise RuntimeError("the LoD release supports only expert leaf layout")
        leaf_kwargs = {
            "block_m": self.leaf_block_m,
            "block_n": self.leaf_block_n,
            "hash_probes": self._page_lookup_probes(cache),
            "num_warps": self.leaf_num_warps,
            "waves_per_eu": self.leaf_waves_per_eu,
            "timing_events": getattr(self, "_lod_leaf_timing_events", None),
            "reduce_num_warps": self.leaf_reduce_num_warps,
        }
        if bool(cache.get("quantization_finalized", False)):
            leaf_kwargs.update(
                quantized_leaf_k=cache.get("quantized_leaf_k"),
                quantized_leaf_v=cache.get("quantized_leaf_v"),
                page_k_scales=cache.get("page_k_scales"),
                page_v_scales=cache.get("page_v_scales"),
                page_sum_k=cache.get("page_sum_k"),
                page_sum_v=cache.get("page_sum_v"),
                quantized_page_sum_k=cache.get("quantized_page_sum_k"),
                quantized_page_sum_v=cache.get("quantized_page_sum_v"),
                page_sum_k_scales=cache.get("page_sum_k_scales"),
                page_sum_v_scales=cache.get("page_sum_v_scales"),
                page_counts=cache.get("page_counts"),
                quant_group_size=self.leaf_quant_group_size,
                quant_token_group_size=self.leaf_quant_token_group_size,
            )
        return paged_leaf_attention(
            q,
            page_k,
            page_v,
            slot_pages,
            overflow_page_keys,
            overflow_page_values,
            overflow_used,
            slot_lengths,
            top_slots,
            page_indices=page_indices,
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
        """Evaluate the release coarse field, excluding refined centroids."""
        del state_capacity
        query_len = int(q.size(2))
        if int(q.size(-1)) > 512 or int(state_v.size(-1)) > 256:
            raise RuntimeError("the LoD release supports Dq<=512 and Dv<=256")

        fused_prefill = getattr(self, "_lod_prefill_fused_coarse", None)
        if fused_prefill is not None:
            del self._lod_prefill_fused_coarse
            coarse_output, coarse_lse, fused_includes_local = fused_prefill
            if include_local != fused_includes_local:
                raise AssertionError("fused prefill local-branch mode drifted")
            expected_output_shape = (*q.shape[:-1], int(state_v.size(-1)))
            if tuple(coarse_output.shape) != expected_output_shape:
                raise AssertionError("fused prefill coarse output shape drifted")
            if tuple(coarse_lse.shape) != tuple(q.shape[:-1]):
                raise AssertionError("fused prefill coarse LSE shape drifted")
            return coarse_output, coarse_lse

        route_payload = getattr(self, "_lod_prefill_route_logits", None)
        if route_payload is not None:
            del self._lod_prefill_route_logits
        else:
            route_payload = getattr(self, "_lod_decode_route_logits", None)
            if route_payload is not None:
                del self._lod_decode_route_logits

        route_logit_scale = None
        if route_payload is None:
            route_logits = self._state_route_logits(
                q,
                state_k,
                counts,
                state_len=state_len,
            )
        elif isinstance(route_payload, tuple):
            route_logits, route_logit_scale = route_payload
        else:
            route_logits = route_payload
        expected_shape = (
            int(q.size(0)),
            int(q.size(1)),
            query_len,
            state_len,
        )
        if tuple(route_logits.shape) != expected_shape:
            raise AssertionError("LoD route-logit shape drifted")

        coarse_local_k = local_k if include_local else local_k[..., :0, :].contiguous()
        coarse_local_v = local_v if include_local else local_v[..., :0, :].contiguous()
        prefill = query_len > 1
        direct_gqa = bool(prefill and self.prefill_coarse_direct_gqa)
        if direct_gqa:
            self._lod_prefill_coarse_direct_gqa_executed = (
                self.prefill_coarse_max_grouped_rows,
                self.prefill_coarse_route_block_n,
                self.prefill_coarse_route_num_warps,
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
            block_n=(
                self.prefill_coarse_route_block_n
                if prefill
                else self.coarse_route_block_n
            ),
            num_warps=(
                self.prefill_coarse_route_num_warps
                if prefill
                else self.coarse_route_num_warps
            ),
            precompute_mean_values=prefill,
            int8_state_pv=False,
            max_grouped_rows=(
                self.prefill_coarse_max_grouped_rows
                if prefill
                else self.coarse_max_grouped_rows
            ),
            direct_gqa_rows=direct_gqa,
            route_logit_scale=(
                route_logit_scale.contiguous()
                if route_logit_scale is not None
                else None
            ),
            timing_events=getattr(self, "_lod_phase_timing_events", None),
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
        output_buffer: torch.Tensor | None = None,
        context_len: int | None = None,
    ) -> torch.Tensor:
        del owners, exact_k, exact_v
        q = q.contiguous()
        # The persistent MLA key buffers intentionally hold raw compressed
        # latents in the experimental modes.  Normalize exact token keys at
        # consumption time; values are already the model-normalized latent.
        local_k = self._mla_normalize_key(local_k, state_centroid=False)
        if new_k is not None:
            new_k = self._mla_normalize_key(new_k, state_centroid=False)
        if sink_k is not None:
            sink_k = self._mla_normalize_key(sink_k, state_centroid=False)
        if output_buffer is not None and (
            tuple(output_buffer.shape) != tuple(q.shape)
            or output_buffer.dtype != q.dtype
            or output_buffer.device != q.device
            or int(output_buffer.stride(-1)) != 1
        ):
            raise ValueError("LOD output buffer has incompatible geometry")
        if sink_k is None or sink_v is None:
            raise RuntimeError("the LoD release requires its separate sink cache")
        query_len = int(q.size(2))
        configured_topk = (
            self.prefill_two_level_topk
            if query_len > 1 and self.prefill_two_level_topk is not None
            else self.two_level_topk
        )
        if configured_topk not in (4, 8):
            raise RuntimeError("the LoD release requires exactly four routes")
        if self.leaf_attention_backend != "paged":
            raise RuntimeError("the LoD release requires paged leaf storage")
        indexed_recursive_decode = bool(
            page_cache is not None
            and self.recursive_page_lod
            and isinstance(page_cache.get("page_indices"), torch.Tensor)
        )
        indexed_flat_decode = bool(
            page_cache is not None
            and not self.recursive_page_lod
            and isinstance(page_cache.get("page_indices"), torch.Tensor)
        )
        fuse_decode_route = (
            self.fused_decode_attention
            and int(state_v.size(-1)) == int(q.size(-1))
            and self.fused_decode_state_route
            and self.routing_normalization == "none"
            and getattr(self, "mla_state_key_normalization", "none") == "none"
            and query_len == 1
            and (
                indexed_recursive_decode
                or indexed_flat_decode
                or not (
                    page_cache is not None
                    and isinstance(page_cache.get("page_indices"), torch.Tensor)
                )
            )
        )
        top_slots = None
        if not fuse_decode_route:
            top_slots = self._route_top_slots(
                q,
                state_k,
                state_v,
                counts,
                state_len=state_len,
                state_capacity=state_capacity,
            )
            if getattr(self, "_lod_padding_state_reserve", 0):
                query_counts = self._repeat_kv(counts[..., :state_len, :]).squeeze(-1)
                safe_slots = top_slots.clamp_min(0)
                selected_counts = torch.gather(
                    query_counts.unsqueeze(2).expand(-1, -1, query_len, -1),
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
            and int(state_v.size(-1)) == int(q.size(-1))
            and query_len == 1
            and (
                indexed_recursive_decode
                or indexed_flat_decode
                or not (
                    page_cache is not None
                    and isinstance(page_cache.get("page_indices"), torch.Tensor)
                )
            )
            and fuse_decode_route
        ):
            if page_cache is None:
                raise RuntimeError("paged LOD attention has no leaf page cache")
            page_k = page_cache[
                "leaf_k"
                if indexed_recursive_decode or indexed_flat_decode
                else "page_k"
            ]
            page_v = page_cache[
                "leaf_v"
                if indexed_recursive_decode or indexed_flat_decode
                else "page_v"
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
            cooperative_route_splits = self.decode_gqa_cooperative_route_splits
            # The retained cooperative decoder is the gfx942 HIP specialization:
            # H=256 with exactly four query heads per KV head. Other geometries
            # use the portable split decoder directly.
            cooperative_leaf = bool(
                self.decode_gqa_cooperative_leaf
                and self.decode_gqa_cooperative_hip
                and self.num_key_value_groups == 4
                and int(q.size(-1)) == 256
            )
            cooperative_adaptive_splits = bool(
                self.decode_gqa_cooperative_adaptive_splits
                and cooperative_route_splits is None
            )
            if cooperative_route_splits is None:
                # Bound fixed page-list parallelism and its partial workspace
                # by per-sequence context. Experimental adaptive mode may use
                # fewer splits for short routes within that bound.
                split_work = max(1, max(context_len or 0, 1) // 4096)
                cooperative_route_splits = max(
                    8,
                    min(32, 1 << (split_work.bit_length() - 1)),
                )
                # Scalar paging already has batch-proportional occupancy. The
                # cooperative launch pays off only after 32K per sequence and,
                # beyond batch eight, after roughly 4K tokens per batch row.
                cooperative_context_threshold = max(32768, 4096 * int(q.size(0)))
                if (context_len or 0) < cooperative_context_threshold:
                    cooperative_leaf = False
            self._last_decode_gqa_cooperative_dispatch = {
                "enabled": bool(cooperative_leaf),
                "route_splits": int(cooperative_route_splits),
                "adaptive_splits": bool(cooperative_adaptive_splits),
                "fused_reduce": bool(
                    cooperative_leaf
                    and self.decode_gqa_cooperative_fused_reduce
                    and cooperative_route_splits <= 8
                ),
            }
            expected_partial = (
                int(q.size(0)),
                int(q.size(1)),
                self.decode_split_kv,
                int(q.size(-1)),
            )
            expected_gqa_route_partial = (
                int(q.size(0)),
                int(q.size(1)),
                8,
                cooperative_route_splits,
                int(q.size(-1)),
            )
            wide_local_required = bool(
                int(q.size(-1)) in (128, 256, 512)
                and 1 < self.num_key_value_groups <= 16
            )
            wide_local_len = (
                int(local_k.size(2)) if local_len is None else int(local_len)
            )
            exact_decode = bool(
                (indexed_recursive_decode or indexed_flat_decode)
                and self.exact_decode_limit > 0
            )
            if self.decode_split_kv > 1 and (
                decode_buffers is None
                or tuple(decode_buffers["partial_out"].shape) != expected_partial
                or decode_buffers["partial_out"].device != q.device
                or (
                    wide_local_required
                    and not (
                        (
                            "wide_gqa_local_scores" in decode_buffers
                            and int(decode_buffers["wide_gqa_local_scores"].size(-1))
                            >= wide_local_len + 1
                        )
                        or (
                            indexed_recursive_decode
                            and self.recursive_materialize_page_scores
                            and "recursive_page_scores" in decode_buffers
                            and int(decode_buffers["recursive_page_scores"].size(-1))
                            >= wide_local_len + 1
                        )
                    )
                )
                or (
                    cooperative_leaf
                    and self.decode_gqa_cooperative_hip
                    and (
                        "gqa_route_partial_out" not in decode_buffers
                        or tuple(decode_buffers["gqa_route_partial_out"].shape)
                        != expected_gqa_route_partial
                    )
                )
                or (
                    fuse_decode_route
                    and (
                        "route_group_lse" not in decode_buffers
                        or int(decode_buffers["route_group_lse"].size(2))
                        < math.ceil(
                            state_capacity
                            / (
                                self.decode_route_group_size
                                * self.decode_route_segment_tiles
                            )
                        )
                    )
                )
            ):
                decode_buffers = new_fused_decode_buffers(
                    q,
                    splits=self.decode_split_kv,
                    exact_kv_heads=(int(state_k.size(1)) if exact_decode else None),
                    state_capacity=(state_capacity if fuse_decode_route else None),
                    route_group_size=self.decode_route_group_size,
                    route_segment_tiles=self.decode_route_segment_tiles,
                    gqa_route_splits=(
                        cooperative_route_splits if cooperative_leaf else None
                    ),
                    materialized_state_route=(
                        self.recursive_state_route_backend == "resplit"
                    ),
                )
                if wide_local_required:
                    if (
                        indexed_recursive_decode
                        and self.recursive_materialize_page_scores
                    ):
                        decode_buffers["recursive_page_scores"] = torch.empty(
                            int(q.size(0)),
                            int(q.size(1)),
                            1,
                            int(page_cache["page_counts"].size(2)),
                            dtype=torch.float32,
                            device=q.device,
                        )
                    else:
                        decode_buffers["wide_gqa_local_scores"] = torch.empty(
                            int(q.size(0)),
                            int(q.size(1)),
                            wide_local_len + 1,
                            dtype=torch.float32,
                            device=q.device,
                        )
                self._lod_decode_attention_buffers = decode_buffers
            if exact_decode and "exact_context_lens" not in decode_buffers:
                decode_buffers["exact_context_lens"] = torch.empty(
                    int(q.size(0)) * int(state_k.size(1)),
                    dtype=torch.int32,
                    device=q.device,
                )
                decode_buffers["exact_exp_sums"] = torch.empty_like(
                    decode_buffers["partial_lse"]
                )
            # Real 32K posting lists favor wider scalar tiles and fewer route
            # reduction groups for every D=128 family tested. Keep the legacy
            # geometry below 32K, where the extra width only adds tail work.
            long_d128_decode = bool(
                self.decode_geometry_tuning
                and int(q.size(-1)) == 128
                and (context_len or 0) >= 32768
            )
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
                hash_probes=self._page_lookup_probes(page_cache),
                block_n=(32 if long_d128_decode else self.decode_block_n),
                num_warps=self.decode_num_warps,
                waves_per_eu=self.leaf_waves_per_eu,
                split_kv=self.decode_split_kv,
                buffers=decode_buffers,
                use_dot=self.decode_use_dot,
                fuse_state_route=fuse_decode_route,
                route_group_size=(
                    64 if long_d128_decode else self.decode_route_group_size
                ),
                route_segment_tiles=self.decode_route_segment_tiles,
                route_num_warps=(
                    1 if long_d128_decode else self.decode_route_num_warps
                ),
                route_reduce_num_warps=(
                    2 if long_d128_decode else self.decode_route_reduce_num_warps
                ),
                route_parallel_reduce=self.decode_route_parallel_reduce,
                route_parallel_reduce_block_d=(
                    self.decode_route_parallel_reduce_block_d
                ),
                final_reduce_num_warps=self.decode_final_reduce_num_warps,
                fuse_final_reduce=self.decode_fuse_final_reduce,
                route_gqa_grouped=self.decode_route_gqa_grouped,
                gqa_cooperative_leaf=cooperative_leaf,
                gqa_cooperative_hip=self.decode_gqa_cooperative_hip,
                protected_len=(
                    self._protected_route_len(state_len)
                    if self.exclude_sink_from_routes
                    else 0
                ),
                max_leaf_tokens=self.leaf_seal_capacity,
                open_count=int(self.two_level_topk),
                sink_k=sink_k,
                sink_v=sink_v,
                route_top_p=None,
                route_residual_mass=None,
                route_mass_fraction=None,
                reuse_residual_local_attention=False,
                timing_events=getattr(self, "_lod_decode_timing_events", None),
                recursive_page_cache=(page_cache if indexed_recursive_decode else None),
                flat_page_indices=(
                    page_cache["page_indices"] if indexed_flat_decode else None
                ),
                flat_page_k_scales=None,
                flat_page_v_scales=None,
                recursive_quant_group_size=self.leaf_quant_group_size,
                recursive_quant_token_group_size=(self.leaf_quant_token_group_size),
                recursive_page_select_block_n=self.recursive_page_select_block_n,
                recursive_state_route_backend=(self.recursive_state_route_backend),
                exact_decode_threshold=(
                    int(self.exact_decode_limit) if exact_decode else 0
                ),
                exact_all_rows=bool(
                    exact_decode
                    and context_len is not None
                    and context_len + int(new_k is not None)
                    <= int(self.exact_decode_limit)
                ),
                exact_leaf_lens=(
                    page_cache.get("leaf_lens") if exact_decode else None
                ),
                precomputed_route_scores=None,
                precomputed_coarse_out=None,
                precomputed_coarse_lse=None,
            )
        if top_slots is None:
            raise AssertionError("LOD routing did not produce slots")
        overlapped_coarse: tuple[torch.Tensor, torch.Tensor] | None = None
        coarse_stream: torch.cuda.Stream | None = None
        foreground_stream: torch.cuda.Stream | None = None
        if (
            self.prefill_overlap_coarse_leaf
            and int(q.size(2)) > 1
            and local_branch is not None
            and self.leaf_attention_backend == "paged"
            and not self.recursive_page_lod
        ):
            coarse_stream = getattr(self, "_lod_prefill_coarse_stream", None)
            if coarse_stream is None:
                coarse_stream = torch.cuda.Stream(device=q.device)
                self._lod_prefill_coarse_stream = coarse_stream
            foreground_stream = torch.cuda.current_stream(q.device)
            coarse_stream.wait_stream(foreground_stream)
            with torch.cuda.stream(coarse_stream):
                overlapped_coarse = self._coarse_attention(
                    q,
                    local_k,
                    local_v,
                    state_k,
                    state_v,
                    counts,
                    top_slots,
                    state_len=state_len,
                    state_capacity=state_capacity,
                    include_local=False,
                )
        if page_cache is None:
            raise RuntimeError("paged LoD attention has no leaf page cache")
        if self.recursive_page_lod and query_len == 1:
            required = (
                "leaf_k",
                "leaf_v",
                "page_indices",
                "page_sum_k",
                "page_sum_v",
                "page_counts",
                "slot_pages",
                "overflow_page_keys",
                "overflow_page_values",
                "overflow_used",
                "slot_lengths",
            )
            if not all(
                isinstance(page_cache.get(name), torch.Tensor) for name in required
            ):
                raise RuntimeError("recursive LoD page cache is incomplete")
            if bool(page_cache.get("mla_raw_page_key_summaries", False)):
                raise RuntimeError("the LoD release does not use raw MLA page keys")
            quantized_attention = bool(page_cache.get("quantization_finalized", False))
            quantized_summaries = bool(
                page_cache.get("summary_quantization_finalized", False)
            )
            recursive_state_k = self._mla_state_key_sum_for_attention(
                state_k,
                counts,
                state_len=state_len,
            )
            exact_output, exact_lse = query_major_indexed_residual_page_attention(
                q,
                recursive_state_k,
                state_v,
                counts,
                page_cache["leaf_k"],
                page_cache["leaf_v"],
                page_cache["page_indices"],
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
                hash_probes=self._page_lookup_probes(page_cache),
                page_block_n=self.recursive_page_block_n,
                num_warps=self.recursive_page_attention_num_warps,
                waves_per_eu=self.leaf_waves_per_eu,
                timing_events=getattr(self, "_lod_leaf_timing_events", None),
                quantized_leaf_k=(
                    page_cache.get("quantized_leaf_k") if quantized_attention else None
                ),
                quantized_leaf_v=(
                    page_cache.get("quantized_leaf_v") if quantized_attention else None
                ),
                page_k_scales=(
                    page_cache.get("page_k_scales") if quantized_attention else None
                ),
                page_v_scales=(
                    page_cache.get("page_v_scales") if quantized_attention else None
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
                    page_cache.get("page_sum_k_scales") if quantized_summaries else None
                ),
                page_sum_v_scales=(
                    page_cache.get("page_sum_v_scales") if quantized_summaries else None
                ),
                quant_group_size=self.leaf_quant_group_size,
                quant_token_group_size=self.leaf_quant_token_group_size,
                quant_bits=int(page_cache.get("leaf_quant_bits", 4)),
            )
        else:
            if self.recursive_page_lod and not self.recursive_prefill_all_leaves:
                raise RuntimeError(
                    "the LoD release opens complete routed centroids during prefill"
                )
            exact_output, exact_lse = self._paged_leaf_attention(
                q, top_slots, page_cache
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
        if overlapped_coarse is None:
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
        else:
            if coarse_stream is None or foreground_stream is None:
                raise AssertionError("coarse/leaf overlap stream state is incomplete")
            foreground_stream.wait_stream(coarse_stream)
            coarse_output, coarse_lse = overlapped_coarse
        if local_branch is not None:
            local_stream = getattr(self, "_lod_prefill_local_stream_pending", None)
            if local_stream is not None:
                del self._lod_prefill_local_stream_pending
                torch.cuda.current_stream(q.device).wait_stream(local_stream)
            local_output, local_lse = local_branch
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
                output_buffer=output_buffer,
            )
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
            output_buffer=output_buffer,
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
        if valid_starts is None and self.prefill_local_attention_backend == "aiter":
            # The exact front is ordinary causal GQA. Reuse the native-GQA CK
            # path used by the local branch instead of physically repeating
            # K/V once per query head for PyTorch SDPA.
            output, _ = self._prefill_local_attention(q, k, v, query_offset=0)
            return output
        k = self._mla_normalize_key(k, state_centroid=False)
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
        k = self._mla_normalize_key(k, state_centroid=False)
        if int(q.size(-1)) >= 512:
            # The practical AITER path rejects 512-wide Q/K (and wider), while
            # absorbed DeepSeek-style MLA is 512 latent + 64 RoPE dimensions. The
            # generic coarse kernel can represent this geometry, but its
            # 512-wide value accumulator forces a four-row tile and is much
            # slower than GEMM for the dense exact-local branch.  Materialize
            # only the target-chunk score tile (not lookback-query rows), then
            # use optimized GEMMs on either side of the softmax.
            key_len = int(k.size(2))
            target_len = key_len - query_offset
            supplied_query_len = int(q.size(2))
            if supplied_query_len == key_len:
                actual_q = q[..., query_offset:, :]
            elif supplied_query_len == target_len:
                actual_q = q
            else:
                raise ValueError(
                    "wide local-attention query length must equal either the "
                    f"key length ({key_len}) or target length ({target_len}), got "
                    f"{supplied_query_len}"
                )
            if target_len <= 0:
                return (
                    v.new_empty(int(q.size(0)), int(q.size(1)), 0, int(v.size(-1))),
                    torch.empty(
                        int(q.size(0)),
                        int(q.size(1)),
                        0,
                        dtype=torch.float32,
                        device=q.device,
                    ),
                )
            # A single large causal GEMM computes the entire upper triangle
            # only to mask it away.  Tile target queries and stop each key
            # field at that tile's end.  This preserves the exact factor-16
            # routing/state schedule while avoiding most masked MLA work.
            target_tile = 4 * self.chunk_len
            output_tiles = []
            lse_tiles = []
            for target_begin in range(0, target_len, target_tile):
                target_end = min(target_len, target_begin + target_tile)
                local_begin = query_offset + target_begin
                local_end = query_offset + target_end
                target_q = actual_q[..., target_begin:target_end, :]
                target_k = self._repeat_kv(k[..., :local_end, :])
                target_v = self._repeat_kv(v[..., :local_end, :])
                scores = torch.matmul(target_q, target_k.transpose(-1, -2)).float()
                scores.mul_(self.scaling)
                query_positions = torch.arange(local_begin, local_end, device=q.device)
                key_positions = torch.arange(local_end, device=q.device)
                visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                scores.masked_fill_(~visible, float("-inf"))
                lse_tiles.append(torch.logsumexp(scores, dim=-1))
                probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
                output_tiles.append(torch.matmul(probabilities, target_v))
            return (
                torch.cat(output_tiles, dim=2),
                torch.cat(lse_tiles, dim=2),
            )
        if self.prefill_local_attention_backend == "aiter":
            original_dlopen_flags = sys.getdlopenflags()
            deepbind = getattr(os, "RTLD_DEEPBIND", 0)
            if deepbind:
                # TileLang exposes its lazy HIP stubs through TVM's global
                # symbol scope.  Bind AITER's CK extension to its own real
                # libamdhip64 dependency so the stub cannot intercept the
                # versioned hipGetDevicePropertiesR0600 entry point.
                sys.setdlopenflags(original_dlopen_flags | deepbind)
            try:
                from aiter.ops.mha import flash_attn_func

                batch, query_heads, supplied_query_len, head_dim = q.shape
                key_len = int(k.size(2))
                query_len = key_len - query_offset
                if supplied_query_len == key_len:
                    actual_q = q[..., query_offset:, :]
                elif supplied_query_len == query_len:
                    actual_q = q
                else:
                    raise ValueError(
                        "AITER local attention requires a full local query field "
                        "or its suffix queries: "
                        f"supplied_query_len={supplied_query_len}, "
                        f"key_len={key_len}, query_offset={query_offset}, "
                        f"suffix_query_len={query_len}"
                    )
                # CK consumes arbitrary batch/token/head strides and only
                # requires the feature dimension to be contiguous.  Keeping
                # these as views avoids copying the full local Q/K/V fields
                # once per attention layer and prefill chunk.
                dense_q = actual_q.permute(0, 2, 1, 3)
                dense_k = k.permute(0, 2, 1, 3)
                dense_v = v.permute(0, 2, 1, 3)
                try:
                    output, lse = flash_attn_func(
                        dense_q,
                        dense_k,
                        dense_v,
                        softmax_scale=self.scaling,
                        causal=True,
                        return_lse=True,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "AITER local attention rejected geometry "
                        f"batch={batch}, query_heads={query_heads}, "
                        f"kv_heads={int(k.size(1))}, query_len={query_len}, "
                        f"key_len={key_len}, head_dim={head_dim}"
                    ) from exc
                return output.permute(0, 2, 1, 3), lse
            finally:
                sys.setdlopenflags(original_dlopen_flags)
        if self.prefill_local_attention_backend not in ("torch", "aiter"):
            raise ValueError("prefill local attention backend must be torch or aiter")
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
        output_buffer: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
    ) -> torch.Tensor:
        output = self._run_prefill(
            q,
            k,
            v,
            logical_prefill_len=logical_prefill_len,
            prefill_valid_starts=prefill_valid_starts,
            build_cache_only=False,
            output_buffer=output_buffer,
            finalize_cache_for_decode=finalize_cache_for_decode,
        )
        if output is None:
            raise AssertionError("attention prefill did not produce an output")
        return output

    @torch.compiler.disable
    def _build_cache_from_bf16(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        clustering_query: torch.Tensor | None = None,
        logical_prefill_len: int | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
    ) -> dict[str, object]:
        """Construct LOD state from an existing post-RoPE BF16 K/V prefix.

        This replays only state updates and semantic region-page construction.
        It deliberately skips query attention and output materialization.  A
        clustering query is needed only by the optional query-metric clustering
        modes; ordinary key-only spherical/coherence routing needs K/V alone.
        """
        self._run_prefill(
            clustering_query,
            k,
            v,
            logical_prefill_len=logical_prefill_len,
            prefill_valid_starts=prefill_valid_starts,
            build_cache_only=True,
            finalize_cache_for_decode=finalize_cache_for_decode,
        )
        state = getattr(self, "_lod_state", None)
        if not isinstance(state, dict):
            raise AssertionError("LOD cache conversion did not produce state")
        return state

    def _run_prefill(
        self,
        q: torch.Tensor | None,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        logical_prefill_len: int | None,
        prefill_valid_starts: torch.Tensor | None,
        build_cache_only: bool,
        output_buffer: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
    ) -> torch.Tensor | None:
        if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
            raise ValueError("prefill K/V must be matching rank-four tensors")
        batch_size, _, attention_len, _ = k.shape
        if q is not None and (
            q.ndim != 4
            or int(q.size(0)) != batch_size
            or int(q.size(2)) != attention_len
            or int(q.size(-1)) != int(k.size(-1))
        ):
            raise ValueError("prefill query geometry does not match cached K/V")
        if not build_cache_only and q is None:
            raise ValueError("attention prefill requires query states")
        if output_buffer is not None and (
            build_cache_only
            or q is None
            or tuple(output_buffer.shape) != tuple(q.shape)
            or output_buffer.dtype != q.dtype
            or output_buffer.device != q.device
            or int(output_buffer.stride(-1)) != 1
        ):
            raise ValueError("prefill output buffer has incompatible geometry")
        prefill_storage = getattr(self, "_lod_prefill_storage", None)
        if prefill_storage is not None and not isinstance(prefill_storage, dict):
            raise TypeError("direct prefill storage must be a tensor dictionary")
        prefill_chunk_len = self.prefill_chunk_len
        prefill_local_len = self.prefill_local_len
        prefill_state_update_len = self.prefill_state_update_len
        if hasattr(self, "_lod_prefill_previous_partition_lse"):
            del self._lod_prefill_previous_partition_lse
        if hasattr(self, "_lod_prefill_previous_state_log_mass"):
            del self._lod_prefill_previous_state_log_mass
        if prefill_chunk_len > prefill_local_len:
            raise ValueError("prefill chunk length cannot exceed its local field")
        if prefill_state_update_len <= 0:
            raise ValueError("prefill state update length must be positive")
        prefill_len = (
            attention_len if logical_prefill_len is None else int(logical_prefill_len)
        )
        if prefill_len <= 0 or prefill_len > attention_len:
            raise ValueError("logical prefill length must fit the attention field")
        if attention_len - prefill_len >= prefill_chunk_len:
            raise ValueError("prefill padding must be confined to the final chunk")
        if prefill_valid_starts is not None:
            prefill_valid_starts = prefill_valid_starts.to(
                device=k.device, dtype=torch.long
            )
            if tuple(prefill_valid_starts.shape) != (batch_size,):
                raise ValueError("prefill valid starts must have one entry per row")
            if bool(
                (prefill_valid_starts.lt(0) | prefill_valid_starts.ge(self.chunk_len))
                .any()
                .item()
            ):
                raise ValueError(
                    "chunk-aligned padding must fit entirely in the first chunk"
                )
            if self.separate_sink_cache:
                raise NotImplementedError(
                    "chunk-aligned padding does not yet support a separate sink cache"
                )
            self._lod_padding_state_reserve = int(prefill_valid_starts.max().item())
        else:
            self._lod_padding_state_reserve = 0
        if self.state_clustering_query_metric != "none" and q is None:
            raise ValueError(
                "full-cache conversion cannot reconstruct query-metric "
                "clustering unless prefill queries are supplied"
            )
        clustering_query_scale = (
            self._state_clustering_query_scale(
                q[..., :prefill_len, :], valid_starts=prefill_valid_starts
            )
            if q is not None
            else None
        )
        exact_lookback = prefill_local_len - prefill_chunk_len
        if getattr(self, "_lod_collect_stats", False):
            self._lod_route_stats = []
        front_len = min(
            attention_len,
            exact_lookback
            + (prefill_chunk_len if self.prefill_exact_first_chunk else self.chunk_len),
        )
        outputs = []
        defer_exact_front = bool(
            self.prefill_exact_first_chunk
            and self.prefill_overlap_exact_state
            and not build_cache_only
        )
        exact_front_stream = None
        if not build_cache_only:
            if q is None:
                raise AssertionError("attention prefill query is missing")
            if defer_exact_front:
                exact_front_stream = getattr(
                    self, "_lod_prefill_exact_front_stream", None
                )
                if exact_front_stream is None:
                    exact_front_stream = torch.cuda.Stream(device=k.device)
                    self._lod_prefill_exact_front_stream = exact_front_stream
                foreground_stream = torch.cuda.current_stream(k.device)
                exact_front_stream.wait_stream(foreground_stream)
                exact_context = torch.cuda.stream(exact_front_stream)
            else:
                exact_context = torch.cuda.stream(torch.cuda.current_stream(k.device))
            with exact_context:
                exact_front = self._exact_attention(
                    q[..., :front_len, :],
                    k[..., :front_len, :],
                    v[..., :front_len, :],
                    causal=True,
                    valid_starts=prefill_valid_starts,
                )
                if output_buffer is None:
                    outputs.append(exact_front)
                else:
                    output_buffer[..., :front_len, :].copy_(exact_front)

        initial_len = min(prefill_len, self.chunk_len)
        separated_sink_len = (
            min(self.sink_len, initial_len) if self.separate_sink_cache else 0
        )
        sink_k = None
        sink_v = None
        if separated_sink_len:
            if prefill_storage is None:
                sink_k = k[..., :separated_sink_len, :].detach().contiguous()
                sink_v = v[..., :separated_sink_len, :].detach().contiguous()
            else:
                sink_k = prefill_storage.get("sink_k")
                sink_v = prefill_storage.get("sink_v")
                if not isinstance(sink_k, torch.Tensor) or not isinstance(
                    sink_v, torch.Tensor
                ):
                    raise TypeError("direct prefill storage lacks sink K/V")
                if (
                    tuple(sink_k.shape[:2])
                    != (batch_size, self.config.num_key_value_heads)
                    or tuple(sink_v.shape[:2]) != tuple(sink_k.shape[:2])
                    or int(sink_k.size(2)) != separated_sink_len
                    or int(sink_v.size(2)) != separated_sink_len
                ):
                    raise ValueError(
                        "direct prefill sink K/V storage has incompatible shape"
                    )
                sink_k.copy_(k[..., :separated_sink_len, :])
                sink_v.copy_(v[..., :separated_sink_len, :])
        archive_k = k[..., separated_sink_len:, :]
        archive_v = v[..., separated_sink_len:, :]
        initial_leaf_len = initial_len - separated_sink_len
        initial_state_len = initial_leaf_len
        if prefill_valid_starts is None:
            initial_state_k = archive_k[..., :initial_state_len, :]
            initial_state_v = archive_v[..., :initial_state_len, :]
            initial_valid = None
            owners = (
                torch.arange(initial_state_len, dtype=torch.long, device=k.device)
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
            )[:, None, :].expand(-1, self.config.num_key_value_heads, -1)
        initial_key_norm_sums = (
            self._state_clustering_constituent_rms(initial_state_k)
            if self.state_clustering_centroid_rescale != "none"
            else None
        )
        if self.state_premerge_factor > 1:
            if initial_state_k.is_cuda and initial_state_v.is_cuda:
                initial_state_k, initial_state_v, initial_counts = (
                    self._premerge_state_inputs_cuda(initial_state_k, initial_state_v)
                )
                grouped_owners = torch.div(
                    owners,
                    self.state_premerge_factor,
                    rounding_mode="floor",
                )
            else:
                (
                    initial_state_k,
                    initial_state_v,
                    initial_counts,
                    grouped_owners,
                ) = _premerge_adjacent_state_inputs(
                    initial_state_k,
                    initial_state_v,
                    self.state_premerge_factor,
                )
            if initial_key_norm_sums is not None:
                initial_key_norm_sums = _sum_adjacent_groups(
                    initial_key_norm_sums,
                    self.state_premerge_factor,
                )
            if initial_valid is None:
                owners = grouped_owners
            else:
                grouped_counts = _sum_adjacent_groups(
                    initial_valid[:, None, :, None].float(),
                    self.state_premerge_factor,
                ).expand(-1, self.config.num_key_value_heads, -1, -1)
                initial_counts = torch.where(
                    grouped_counts > 0,
                    grouped_counts,
                    torch.full_like(grouped_counts, torch.finfo(torch.float32).tiny),
                )
                grouped_slots = int(initial_state_k.size(2))
                initial_state_k = torch.cat(
                    (
                        initial_state_k,
                        torch.zeros_like(initial_state_k[..., :1, :]),
                    ),
                    dim=2,
                )
                initial_state_v = torch.cat(
                    (
                        initial_state_v,
                        torch.zeros_like(initial_state_v[..., :1, :]),
                    ),
                    dim=2,
                )
                initial_counts = torch.cat(
                    (
                        initial_counts,
                        torch.full_like(
                            initial_counts[..., :1, :],
                            torch.finfo(torch.float32).tiny,
                        ),
                    ),
                    dim=2,
                )
                if initial_key_norm_sums is not None:
                    initial_key_norm_sums = torch.cat(
                        (
                            initial_key_norm_sums,
                            torch.zeros_like(initial_key_norm_sums[..., :1, :]),
                        ),
                        dim=2,
                    )
                grouped_valid = grouped_counts[..., 0].gt(0).any(dim=1)
                initial_valid = torch.cat(
                    (
                        grouped_valid,
                        torch.zeros(
                            batch_size,
                            1,
                            dtype=torch.bool,
                            device=k.device,
                        ),
                    ),
                    dim=1,
                )
                physical = slot.unsqueeze(0).expand(batch_size, -1)
                owners = torch.where(
                    physical >= prefill_valid_starts.unsqueeze(1),
                    torch.div(
                        physical - prefill_valid_starts.unsqueeze(1),
                        self.state_premerge_factor,
                        rounding_mode="floor",
                    ),
                    torch.full_like(physical, grouped_slots),
                )[:, None, :].expand(-1, self.config.num_key_value_heads, -1)
            initial_state_len = int(initial_state_k.size(2))
        elif initial_valid is None:
            initial_counts = torch.ones(
                *initial_state_k.shape[:3],
                1,
                dtype=torch.float32,
                device=k.device,
            )
        else:
            initial_counts = torch.full(
                (*initial_state_k.shape[:3], 1),
                torch.finfo(torch.float32).tiny,
                dtype=torch.float32,
                device=k.device,
            )
            initial_counts.masked_fill_(initial_valid[:, None, :, None], 1.0)
        state_capacity = self._state_capacity(prefill_len, initial_state_len)
        if getattr(self, "_lod_collect_stats", False):
            self._lod_ever_selected_slots = torch.zeros(
                batch_size,
                self.config.num_attention_heads,
                state_capacity,
                dtype=torch.bool,
                device=k.device,
            )
        if prefill_storage is None:
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
        else:
            state_k = prefill_storage.get("state_k")
            state_v = prefill_storage.get("state_v")
            counts = prefill_storage.get("counts")
            if not all(
                isinstance(value, torch.Tensor) for value in (state_k, state_v, counts)
            ):
                raise TypeError("direct prefill storage lacks state tensors")
            if (
                tuple(state_k.shape[:2])
                != (batch_size, self.config.num_key_value_heads)
                or tuple(state_v.shape[:2]) != tuple(state_k.shape[:2])
                or tuple(counts.shape[:2]) != tuple(state_k.shape[:2])
                or int(state_k.size(2)) < state_capacity
                or int(state_v.size(2)) < state_capacity
                or int(counts.size(2)) < state_capacity
            ):
                raise ValueError("direct prefill state storage has incompatible shape")
            state_k[..., :initial_state_len, :].copy_(initial_state_k)
            state_v[..., :initial_state_len, :].copy_(initial_state_v)
        counts[..., :initial_state_len, :].copy_(initial_counts)
        key_norm_sums = None
        if self.state_clustering_centroid_rescale != "none":
            if prefill_storage is None:
                key_norm_sums = torch.zeros_like(counts)
            else:
                key_norm_sums = prefill_storage.get("key_norm_sums")
                if not isinstance(key_norm_sums, torch.Tensor):
                    raise TypeError("direct prefill storage lacks key-norm sums")
            if initial_key_norm_sums is None:
                raise AssertionError("initial constituent norms are missing")
            if initial_valid is not None:
                initial_key_norm_sums = initial_key_norm_sums.masked_fill(
                    ~initial_valid[:, None, :, None], 0
                )
            key_norm_sums[..., :initial_state_len, :].copy_(initial_key_norm_sums)
        state_len = initial_state_len
        scheduled_state_len = initial_state_len
        state_coverage = initial_len
        page_cache = None
        if self.leaf_attention_backend == "paged":
            sequence_capacity = _round_up(prefill_len, self.chunk_len) + max(
                self.chunk_len, self.decode_cache_headroom
            )
            page_cache = self._new_page_cache(
                archive_k[..., :initial_leaf_len, :],
                archive_v[..., :initial_leaf_len, :],
                owners,
                state_capacity=state_capacity,
                sequence_capacity=sequence_capacity,
                virtual_k=archive_k if self.virtual_page_storage else None,
                virtual_v=archive_v if self.virtual_page_storage else None,
                destination=(
                    prefill_storage.get("page_cache")
                    if prefill_storage is not None
                    else None
                ),
            )
            owners = None

        # The first large prefill block has no genuinely remote history: its
        # entire causal prefix can be handled by the same exact attention call
        # with only ``exact_lookback`` additional keys.  Build the state needed
        # by the following block after that exact call instead of paying a
        # route/coarse/leaf pass over the nearly empty first remote partition.
        if self.prefill_exact_first_chunk:
            first_remote_begin = front_len
            first_remote_state_coverage = max(
                initial_len, first_remote_begin - exact_lookback
            )
            append_begin = state_coverage
            append_owner_parts = []
            while state_coverage < first_remote_state_coverage:
                update_end = min(
                    first_remote_state_coverage,
                    state_coverage + prefill_state_update_len,
                )
                update_ctx_len = exact_lookback + update_end
                next_scheduled_state_len = (
                    self._next_scheduled_state_len(
                        scheduled_state_len,
                        ctx_len=update_ctx_len,
                        available_context=update_end,
                        overflow_len=update_end - state_coverage,
                    )
                    if self.state_split_max_leaves is not None
                    else scheduled_state_len
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
                    key_norm_sums,
                    k[..., state_coverage:update_end, :],
                    v[..., state_coverage:update_end, :],
                    state_len=state_len,
                    ctx_len=update_ctx_len,
                    available_context=update_end,
                    state_capacity=state_capacity,
                    clustering_query_scale=clustering_query_scale,
                    scheduled_state_len=scheduled_state_len,
                )
                scheduled_state_len = (
                    next_scheduled_state_len
                    if self.state_split_max_leaves is not None
                    else state_len
                )
                if page_cache is not None:
                    if old_slot_remap is not None:
                        raise AssertionError("paged state remapping is unsupported")
                    append_owner_parts.append(
                        new_owners.clone()
                        if self.state_premerge_factor > 1
                        else new_owners
                    )
                else:
                    if owners is None:
                        raise AssertionError("packed LOD owner archive is missing")
                    if old_slot_remap is not None:
                        owners = torch.gather(old_slot_remap, 2, owners)
                    owners = torch.cat((owners, new_owners), dim=2)
                state_coverage = update_end
            if page_cache is not None and append_owner_parts:
                self._append_page_cache(
                    page_cache,
                    k[..., append_begin:state_coverage, :],
                    v[..., append_begin:state_coverage, :],
                    (
                        append_owner_parts[0]
                        if len(append_owner_parts) == 1
                        else torch.cat(append_owner_parts, dim=2)
                    ),
                )
        if exact_front_stream is not None:
            torch.cuda.current_stream(k.device).wait_stream(exact_front_stream)

        for query_begin in range(front_len, attention_len, prefill_chunk_len):
            query_end = min(attention_len, query_begin + prefill_chunk_len)
            bswa_begin = max(0, query_begin - exact_lookback)
            if state_coverage != bswa_begin:
                raise AssertionError("LOD prefill state coverage drifted")
            if not build_cache_only:
                if q is None:
                    raise AssertionError("attention prefill query is missing")
                if self.split_prefill_local_attention:
                    local_args = (
                        q[..., bswa_begin:query_end, :],
                        k[..., bswa_begin:query_end, :],
                        v[..., bswa_begin:query_end, :],
                    )
                    if self.prefill_overlap_local_lod:
                        if (
                            self.prefill_route_mass_fraction is not None
                            and self.prefill_mass_include_local_lse
                        ):
                            raise ValueError(
                                "local/LOD overlap requires a remote-state mass cutoff"
                            )
                        local_stream = getattr(self, "_lod_prefill_local_stream", None)
                        if local_stream is None:
                            local_stream = torch.cuda.Stream(device=k.device)
                            self._lod_prefill_local_stream = local_stream
                        foreground_stream = torch.cuda.current_stream(k.device)
                        local_stream.wait_stream(foreground_stream)
                        with torch.cuda.stream(local_stream):
                            local_branch = self._prefill_local_attention(
                                *local_args,
                                query_offset=query_begin - bswa_begin,
                            )
                        self._lod_prefill_local_stream_pending = local_stream
                    else:
                        local_branch = self._prefill_local_attention(
                            *local_args,
                            query_offset=query_begin - bswa_begin,
                        )
                else:
                    local_branch = None
                chunk_output = self._two_level_attention(
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
                    context_len=query_end,
                    output_buffer=(
                        output_buffer[..., query_begin:query_end, :]
                        if output_buffer is not None
                        else None
                    ),
                )
                if output_buffer is None:
                    outputs.append(chunk_output)

            next_bswa_begin = (
                max(0, query_begin + prefill_chunk_len - exact_lookback)
                if query_end < attention_len
                else bswa_begin
            )
            append_begin = state_coverage
            append_owner_parts = []
            while state_coverage < next_bswa_begin:
                update_end = min(
                    next_bswa_begin, state_coverage + prefill_state_update_len
                )
                update_ctx_len = query_begin + update_end - bswa_begin
                next_scheduled_state_len = (
                    self._next_scheduled_state_len(
                        scheduled_state_len,
                        ctx_len=update_ctx_len,
                        available_context=update_end,
                        overflow_len=update_end - state_coverage,
                    )
                    if self.state_split_max_leaves is not None
                    else scheduled_state_len
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
                    key_norm_sums,
                    k[..., state_coverage:update_end, :],
                    v[..., state_coverage:update_end, :],
                    state_len=state_len,
                    ctx_len=update_ctx_len,
                    available_context=update_end,
                    state_capacity=state_capacity,
                    clustering_query_scale=clustering_query_scale,
                    scheduled_state_len=scheduled_state_len,
                )
                scheduled_state_len = (
                    next_scheduled_state_len
                    if self.state_split_max_leaves is not None
                    else state_len
                )
                if page_cache is not None:
                    if old_slot_remap is not None:
                        raise AssertionError("paged state remapping is unsupported")
                    # Adjacent premerge expands owners through a persistent
                    # workspace.  A 4K archive block can contain several
                    # state-update batches, so retaining the returned view
                    # would let the following update overwrite every earlier
                    # owner segment before the batched page append.
                    append_owner_parts.append(
                        new_owners.clone()
                        if self.state_premerge_factor > 1
                        else new_owners
                    )
                else:
                    if owners is None:
                        raise AssertionError("packed LOD owner archive is missing")
                    if old_slot_remap is not None:
                        owners = torch.gather(old_slot_remap, 2, owners)
                    owners = torch.cat((owners, new_owners), dim=2)
                state_coverage = update_end
            if page_cache is not None and append_owner_parts:
                self._append_page_cache(
                    page_cache,
                    k[..., append_begin:state_coverage, :],
                    v[..., append_begin:state_coverage, :],
                    (
                        append_owner_parts[0]
                        if len(append_owner_parts) == 1
                        else torch.cat(append_owner_parts, dim=2)
                    ),
                )

        # Prepare the state boundary required by the first decode token after
        # all prefill outputs have been computed.  Otherwise prompts ending on
        # a chunk boundary pay a full 256-token state update on token one.
        if finalize_cache_for_decode:
            decode_coverage = max(initial_len, self._bswa_begin(prefill_len + 1))
        else:
            # Scheduler chunks are not semantic LOD query blocks. Preserve the
            # exact field until the current logical prefill block is complete.
            completed_blocks = max(
                0,
                (prefill_len - front_len) // prefill_chunk_len,
            )
            next_query_begin = front_len + completed_blocks * prefill_chunk_len
            decode_coverage = max(initial_len, next_query_begin - exact_lookback)
        append_begin = state_coverage
        append_owner_parts = []
        while decode_coverage > state_coverage:
            update_end = min(decode_coverage, state_coverage + prefill_state_update_len)
            update_ctx_len = min(prefill_len, update_end + self.local_len)
            next_scheduled_state_len = (
                self._next_scheduled_state_len(
                    scheduled_state_len,
                    ctx_len=update_ctx_len,
                    available_context=update_end,
                    overflow_len=update_end - state_coverage,
                )
                if self.state_split_max_leaves is not None
                else scheduled_state_len
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
                key_norm_sums,
                k[..., state_coverage:update_end, :],
                v[..., state_coverage:update_end, :],
                state_len=state_len,
                ctx_len=update_ctx_len,
                available_context=update_end,
                state_capacity=state_capacity,
                clustering_query_scale=clustering_query_scale,
                scheduled_state_len=scheduled_state_len,
            )
            scheduled_state_len = (
                next_scheduled_state_len
                if self.state_split_max_leaves is not None
                else state_len
            )
            if page_cache is not None:
                if old_slot_remap is not None:
                    raise AssertionError("paged state remapping is unsupported")
                append_owner_parts.append(
                    new_owners.clone() if self.state_premerge_factor > 1 else new_owners
                )
            else:
                if owners is None:
                    raise AssertionError("packed LOD owner archive is missing")
                if old_slot_remap is not None:
                    owners = torch.gather(old_slot_remap, 2, owners)
                owners = torch.cat((owners, new_owners), dim=2)
            state_coverage = update_end
        if page_cache is not None and append_owner_parts:
            self._append_page_cache(
                page_cache,
                k[..., append_begin:state_coverage, :],
                v[..., append_begin:state_coverage, :],
                (
                    append_owner_parts[0]
                    if len(append_owner_parts) == 1
                    else torch.cat(append_owner_parts, dim=2)
                ),
            )
        # Right padding exists only to reuse compiled final-chunk shapes. It
        # must never become persistent state or enter the decode-local field.
        recent_k = k[..., state_coverage:prefill_len, :]
        recent_v = v[..., state_coverage:prefill_len, :]
        recent_len = int(recent_k.size(2))
        if page_cache is not None:
            recent_capacity = max(
                self.local_len + self.decode_state_update_len,
                (0 if self.prefill_exact_first_chunk else self.prefill_local_len)
                if not finalize_cache_for_decode
                else 0,
                recent_len,
            )
            if prefill_storage is None:
                buffered_k = k.new_empty(*k.shape[:2], recent_capacity, int(k.size(-1)))
                buffered_v = v.new_empty(*v.shape[:2], recent_capacity, int(v.size(-1)))
            else:
                buffered_k = prefill_storage.get("recent_k")
                buffered_v = prefill_storage.get("recent_v")
                if not isinstance(buffered_k, torch.Tensor) or not isinstance(
                    buffered_v, torch.Tensor
                ):
                    raise TypeError("direct prefill storage lacks recent K/V")
                if (
                    int(buffered_k.size(2)) < recent_capacity
                    or int(buffered_v.size(2)) < recent_capacity
                ):
                    raise ValueError("direct prefill recent K/V storage is too small")
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
                    isinstance(value, torch.Tensor) for value in quantization_tensors
                ):
                    raise RuntimeError("virtual quantized prefill cache is incomplete")
                if self.leaf_quant_scale_mode not in ("max", "l2"):
                    raise ValueError("leaf quantization scale mode must be max or l2")
                if self.leaf_append_quant_scale_mode not in ("max", "l2"):
                    raise ValueError(
                        "leaf append quantization scale mode must be max or l2"
                    )
                quantize_virtual_paged_kv(
                    *quantization_tensors,
                    quantized_counts,
                    quant_group_size=self.leaf_quant_group_size,
                    quant_token_group_size=self.leaf_quant_token_group_size,
                    quant_bits=int(page_cache.get("leaf_quant_bits", 4)),
                    optimize_scale=self.leaf_quant_scale_mode == "l2",
                )
                page_cache["quantization_finalized"] = True
                destination_page = None
                if prefill_storage is not None:
                    destination_page = prefill_storage.get("page_cache")
                    if not isinstance(destination_page, dict):
                        raise TypeError(
                            "direct quantized prefill storage lacks its page cache"
                        )
                if self.page_summary_quant_bits not in (0, 8):
                    raise ValueError("page-summary quantization supports 0 or 8 bits")
                if self.page_summary_scale_mode not in ("max", "l2"):
                    raise ValueError("page-summary scale mode must be max or l2")
                if self.page_summary_quant_bits == 8:
                    quantized_summaries = quantize_page_summaries_int8(
                        page_cache["page_sum_k"],
                        page_cache["page_sum_v"],
                        quant_group_size=self.leaf_quant_group_size,
                        optimize_scale=self.page_summary_scale_mode == "l2",
                    )
                    summary_names = (
                        "quantized_page_sum_k",
                        "quantized_page_sum_v",
                        "page_sum_k_scales",
                        "page_sum_v_scales",
                    )
                    if prefill_storage is None:
                        for name, value in zip(summary_names, quantized_summaries):
                            page_cache[name] = value
                    else:
                        assert isinstance(destination_page, dict)
                        for name, value in zip(summary_names, quantized_summaries):
                            destination_value = destination_page.get(name)
                            if not isinstance(destination_value, torch.Tensor):
                                raise TypeError(
                                    f"direct quantized prefill storage lacks {name}"
                                )
                            if (
                                destination_value.shape[:2] != value.shape[:2]
                                or destination_value.shape[-1] != value.shape[-1]
                                or int(destination_value.size(2)) < int(value.size(2))
                            ):
                                raise ValueError(
                                    f"direct quantized prefill storage {name} is too small"
                                )
                            destination_value[..., : value.size(2), :].copy_(value)
                            page_cache[name] = destination_value
                    page_cache["summary_quantization_finalized"] = True
                    if prefill_storage is None:
                        page_cache["page_sum_k"] = k.new_empty(
                            *k.shape[:2], 1, int(k.size(-1))
                        )
                        page_cache["page_sum_v"] = v.new_empty(
                            *v.shape[:2], 1, int(v.size(-1))
                        )
                    else:
                        assert isinstance(destination_page, dict)
                        page_cache["page_sum_k"] = destination_page["page_sum_k"]
                        page_cache["page_sum_v"] = destination_page["page_sum_v"]
                # All archived leaves now live in quantized flat tensors. Keep
                # only typed pointer sentinels for the compile-time BF16 fallback.
                if prefill_storage is None:
                    page_cache["leaf_k"] = k.new_empty(*k.shape[:2], 1, int(k.size(-1)))
                    page_cache["leaf_v"] = v.new_empty(*v.shape[:2], 1, int(v.size(-1)))
                else:
                    assert isinstance(destination_page, dict)
                    page_cache["leaf_k"] = destination_page["leaf_k"]
                    page_cache["leaf_v"] = destination_page["leaf_v"]
        self._lod_state = {
            "state_k": state_k.detach(),
            "state_v": state_v.detach(),
            "counts": counts.detach(),
            "state_len": state_len,
            "scheduled_state_len": scheduled_state_len,
            "coverage": state_coverage,
            "state_capacity": state_capacity,
            "recent_k": recent_k.detach(),
            "recent_v": recent_v.detach(),
            "recent_len": recent_len,
            "total_len": prefill_len,
        }
        if prefill_storage is not None:
            self._lod_state["pool_backed"] = True
        if key_norm_sums is not None:
            self._lod_state["key_norm_sums"] = key_norm_sums.detach()
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
        if build_cache_only:
            return None
        if output_buffer is not None:
            return output_buffer
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=2)

    @torch.compiler.disable
    def _cached_prefill_attention(
        self,
        q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
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
        key_norm_sums = cache.get("key_norm_sums")
        if not all(
            isinstance(tensor, torch.Tensor) for tensor in (state_k, state_v, counts)
        ):
            raise TypeError("cached LOD state tensors are missing")
        if key_norm_sums is not None and not isinstance(key_norm_sums, torch.Tensor):
            raise TypeError("LOD key-norm sum cache is invalid")
        state_len = int(cache["state_len"])
        scheduled_state_len = int(cache.get("scheduled_state_len", state_len))
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
            if key_norm_sums is not None:
                key_norm_sums = _pad_sequence(key_norm_sums, state_capacity).clone()
            self._grow_slot_page_table(page_cache, required_slots=state_capacity)

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

        clustering_query_scale = self._state_clustering_query_scale(q)

        def update_to(target_coverage: int, *, context_length_for) -> None:
            nonlocal state_k, state_v, counts, state_len, scheduled_state_len
            nonlocal state_coverage
            append_source_begin = state_coverage - initial_coverage
            append_owner_parts = []
            while state_coverage < target_coverage:
                update_end = min(
                    target_coverage,
                    state_coverage + prefill_state_update_len,
                )
                source_begin = state_coverage - initial_coverage
                source_end = update_end - initial_coverage
                overflow_k = working_k[..., source_begin:source_end, :]
                overflow_v = working_v[..., source_begin:source_end, :]
                update_ctx_len = context_length_for(update_end)
                next_scheduled_state_len = (
                    self._next_scheduled_state_len(
                        scheduled_state_len,
                        ctx_len=update_ctx_len,
                        available_context=update_end,
                        overflow_len=source_end - source_begin,
                    )
                    if self.state_split_max_leaves is not None
                    else scheduled_state_len
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
                    key_norm_sums,
                    overflow_k,
                    overflow_v,
                    state_len=state_len,
                    ctx_len=update_ctx_len,
                    available_context=update_end,
                    state_capacity=state_capacity,
                    clustering_query_scale=clustering_query_scale,
                    scheduled_state_len=scheduled_state_len,
                )
                scheduled_state_len = (
                    next_scheduled_state_len
                    if self.state_split_max_leaves is not None
                    else state_len
                )
                if old_slot_remap is not None:
                    raise AssertionError("paged state remapping is unsupported")
                append_owner_parts.append(new_owners)
                state_coverage = update_end
            if append_owner_parts:
                append_source_end = state_coverage - initial_coverage
                self._append_page_cache(
                    page_cache,
                    working_k[..., append_source_begin:append_source_end, :],
                    working_v[..., append_source_begin:append_source_end, :],
                    (
                        append_owner_parts[0]
                        if len(append_owner_parts) == 1
                        else torch.cat(append_owner_parts, dim=2)
                    ),
                )

        outputs = []
        query_begin = 0
        # Exact-first initial prefill ends at the configured prefill-block
        # boundary.  Continue cached scheduler chunks from that same boundary;
        # using the legacy ``lookback + decode chunk`` origin here splits an
        # otherwise aligned 16K continuation into a tiny leading block plus a
        # second routed block, doubling route/leaf launch overhead.
        front_len = (
            prefill_chunk_len
            if self.prefill_exact_first_chunk
            else exact_lookback + self.chunk_len
        )
        if previous_total_len < front_len:
            front_query_end = min(turn_len, front_len - previous_total_len)
            initial_state_k = self._mean(
                state_k[..., :state_len, :], counts[..., :state_len, :]
            )
            initial_state_v = self._mean(
                state_v[..., :state_len, :], counts[..., :state_len, :]
            )
            exact_key_parts = []
            exact_value_parts = []
            if isinstance(sink_k, torch.Tensor) and isinstance(sink_v, torch.Tensor):
                exact_key_parts.append(sink_k)
                exact_value_parts.append(sink_v)
            exact_tail_end = previous_total_len + front_query_end - initial_coverage
            exact_key_parts.extend(
                (initial_state_k, working_k[..., :exact_tail_end, :])
            )
            exact_value_parts.extend(
                (initial_state_v, working_v[..., :exact_tail_end, :])
            )
            exact_k = torch.cat(exact_key_parts, dim=2)
            exact_v = torch.cat(exact_value_parts, dim=2)
            suffix_query = q[..., :front_query_end, :]
            # The initial exact field can contain premerged coarse entries.
            # Its causal prefix is therefore the number of materialized keys
            # before these suffix queries, not the uncompressed token offset.
            # They are equal for the ordinary learned state, but differ for
            # fixed adjacent-token groups such as T/16.
            exact_query_offset = int(exact_k.size(2)) - front_query_end
            exact_q = (
                suffix_query
                if self.prefill_local_attention_backend == "aiter"
                else torch.cat(
                    (
                        q.new_zeros(*q.shape[:2], exact_query_offset, int(q.size(-1))),
                        suffix_query,
                    ),
                    dim=2,
                )
            )
            exact_output, _ = self._prefill_local_attention(
                exact_q,
                exact_k,
                exact_v,
                query_offset=exact_query_offset,
            )
            if output_buffer is not None:
                output_buffer[..., :front_query_end, :].copy_(exact_output)
                outputs.append(output_buffer[..., :front_query_end, :])
            else:
                outputs.append(exact_output)
            query_begin = front_query_end
        while query_begin < turn_len:
            absolute_query_begin = previous_total_len + query_begin
            block_index = (absolute_query_begin - front_len) // prefill_chunk_len
            block_begin = front_len + block_index * prefill_chunk_len
            block_end = block_begin + prefill_chunk_len
            query_end = min(
                turn_len,
                block_end - previous_total_len,
            )
            desired_coverage = max(
                state_coverage,
                block_begin - exact_lookback,
            )
            update_to(
                desired_coverage,
                context_length_for=lambda update_end: (
                    block_begin + update_end - desired_coverage
                ),
            )
            local_begin = state_coverage - initial_coverage
            local_end = previous_total_len + query_end - initial_coverage
            local_k = working_k[..., local_begin:local_end, :]
            local_v = working_v[..., local_begin:local_end, :]
            query_prefix_len = absolute_query_begin - state_coverage
            suffix_query = q[..., query_begin:query_end, :]
            if self.prefill_local_attention_backend == "aiter":
                local_query = suffix_query
            else:
                local_query = torch.cat(
                    (
                        q.new_zeros(*q.shape[:2], query_prefix_len, int(q.size(-1))),
                        suffix_query,
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
                    context_len=previous_total_len + query_end,
                    output_buffer=(
                        output_buffer[..., query_begin:query_end, :]
                        if output_buffer is not None
                        else None
                    ),
                )
            )
            query_begin = query_end

        deferred_update_stream = getattr(
            self, "_lod_prefill_deferred_update_stream", None
        )
        if deferred_update_stream is not None:
            # The routed output above depends only on the pre-update cache.
            # Preserve its K/V temporaries until the deferred final update has
            # consumed them, without making the foreground stream wait.
            foreground_stream = torch.cuda.current_stream(q.device)
            deferred_update_stream.wait_stream(foreground_stream)
            working_k.record_stream(deferred_update_stream)
            working_v.record_stream(deferred_update_stream)
            if clustering_query_scale is not None:
                clustering_query_scale.record_stream(deferred_update_stream)
            update_context = torch.cuda.stream(deferred_update_stream)
        else:
            update_context = torch.cuda.stream(torch.cuda.current_stream(q.device))
        with update_context:
            if finalize_cache_for_decode:
                decode_coverage = max(
                    state_coverage,
                    self._bswa_begin(total_len + 1),
                )
            else:
                # A scheduler chunk may stop anywhere inside a logical LOD
                # query block.  Keeping that whole unfinished block exact can
                # exceed the fixed per-request local row by many thousands of
                # tokens.  Advance through the partial block and retain only
                # the configured exact lookback; its earlier tokens are now
                # represented by state/pages for the next continuation.
                decode_coverage = max(
                    state_coverage,
                    total_len - exact_lookback,
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
                    (0 if self.prefill_exact_first_chunk else self.prefill_local_len)
                    if not finalize_cache_for_decode
                    else 0,
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
                scheduled_state_len=scheduled_state_len,
                coverage=state_coverage,
                state_capacity=state_capacity,
                recent_k=recent_k.detach(),
                recent_v=recent_v.detach(),
                recent_len=tail_len,
                total_len=total_len,
            )
            if key_norm_sums is not None:
                cache["key_norm_sums"] = key_norm_sums.detach()
        if output_buffer is not None:
            return output_buffer
        if len(outputs) == 1:
            return outputs[0]
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
        key_norm_sums = cache.get("key_norm_sums")
        if key_norm_sums is not None and not isinstance(key_norm_sums, torch.Tensor):
            raise TypeError("LOD key-norm sum cache is invalid")
        state_len = int(cache["state_len"])
        scheduled_state_len = int(cache.get("scheduled_state_len", state_len))
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
        target_coverage = max(min(previous_total_len, self.chunk_len), state_coverage)
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
            if int(state_k.size(2)) < update_state_capacity:
                state_k = _pad_sequence(state_k, update_state_capacity).clone()
                state_v = _pad_sequence(state_v, update_state_capacity).clone()
                counts = _pad_sequence(counts, update_state_capacity).clone()
                if key_norm_sums is not None:
                    key_norm_sums = _pad_sequence(
                        key_norm_sums, update_state_capacity
                    ).clone()
                if page_cache is not None:
                    self._grow_slot_page_table(
                        page_cache, required_slots=update_state_capacity
                    )
            clustering_query_scale = self._state_clustering_query_scale(q)
            update_ctx_len = exact_floor + target_coverage
            next_scheduled_state_len = (
                self._next_scheduled_state_len(
                    scheduled_state_len,
                    ctx_len=update_ctx_len,
                    available_context=target_coverage,
                    overflow_len=overflow_len,
                )
                if self.state_split_max_leaves is not None
                else scheduled_state_len
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
                key_norm_sums,
                overflow_k,
                overflow_v,
                state_len=state_len,
                ctx_len=update_ctx_len,
                available_context=target_coverage,
                state_capacity=update_state_capacity,
                clustering_query_scale=clustering_query_scale,
                scheduled_state_len=scheduled_state_len,
            )
            scheduled_state_len = (
                next_scheduled_state_len
                if self.state_split_max_leaves is not None
                else state_len
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
                current_k = torch.cat((recent_k[..., :recent_len, :], new_k), dim=2)
                current_v = torch.cat((recent_v[..., :recent_len, :], new_v), dim=2)
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
                and not self.prefill_int8_leaf_mma
                and int(new_v.size(-1)) == int(q.size(-1))
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
                local_k = torch.cat((recent_k[..., :recent_len, :], new_k), dim=2)
                local_v = torch.cat((recent_v[..., :recent_len, :], new_v), dim=2)
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
                context_len=total_len,
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
            scheduled_state_len=scheduled_state_len,
            coverage=state_coverage,
            recent_k=recent_k.detach(),
            recent_v=recent_v.detach(),
            recent_len=recent_len,
            total_len=total_len,
        )
        if key_norm_sums is not None:
            cache["key_norm_sums"] = key_norm_sums.detach()
        if owners is not None:
            cache["owners"] = owners.detach()
            cache["exact_k"] = exact_k.detach()
            cache["exact_v"] = exact_v.detach()
        return output


__all__ = ["TritonLODAttentionCore"]
