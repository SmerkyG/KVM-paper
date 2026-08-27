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
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from .kernels.paged_leaf_attention import (
    append_paged_int8_kv,
    append_paged_kv,
    append_quantized_virtual_paged_kv,
    append_virtual_paged_kv,
    aiter_query_tile_union_paged_leaf_attention,
    aiter_varlen_paged_leaf_attention,
    dense_page_summary_attention,
    fused_decode_paged_lod_attention,
    materialize_page_summary_scores_gqa,
    new_fused_decode_buffers,
    paged_leaf_attention,
    query_major_paged_leaf_attention,
    query_major_indexed_residual_page_attention,
    query_major_residual_page_attention,
    query_tile_slot_unions,
    remove_query_tile_union_from_coarse,
    static_cap_aiter_paged_leaf_attention,
    refine_route_candidates_by_leaf_mass,
    refine_route_candidates_by_page_mass,
    refine_route_candidates_by_virtual_leaf_mass,
    refine_route_candidates_by_virtual_leaf_output,
    quantize_page_summaries_int8,
    quantize_virtual_paged_kv,
)
from .kernels.lod_kernels import (
    apply_residual_mass_opening,
    bipartite_reduce_overflow,
    constituent_rms,
    expand_adjacent_group_owners,
    merge_attention_branches,
    merge_attention_branches_with_sink,
    merge_state_in_place,
    new_route_buffers,
    new_state_delta_buffers,
    new_state_maxsim_buffers,
    prepare_state_clustering_keys,
    premerge_adjacent_kv,
    route_mass_fraction_scores,
    route_mass_fraction_state_decode,
    route_logits_coarse_attention,
    route_logits_hierarchical_topk,
    route_logits_topk_coarse_attention,
    route_top8_scores_grouped,
    route_top8_state_grouped,
    streaming_state_maxsim,
    subtract_selected_coarse_from_full,
)
from .kvm_mixer import _all_idx, _gather_by_idx, _split_append_merge_idx_by_maxsim
from .pytorch_lod_attention import (
    _premerge_adjacent_state_inputs,
    _sum_adjacent_groups,
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
    """Projection-free mass-corrected top-k LOD attention implementation."""

    chunk_len = 256
    local_len = 512
    prefill_chunk_len = 256
    prefill_local_len = 512
    prefill_state_update_len = 256
    # Query-independent prefill alternative: every 4K catch-up rebuild opens
    # all exact leaves of centroids in the scheduled small-posting cohort and
    # leaves larger centroids in the count-corrected coarse branch.
    prefill_static_leaf_aiter = False
    prefill_static_leaf_cap_min = 16
    # Optional dynamic-routing cohort: top-k may only open centroids whose
    # posting list is at most max(minimum, ceil(sqrt(T) / divisor)). Larger
    # centroids remain in the mass-corrected coarse residual.
    prefill_route_leaf_cap_min: int | None = None
    static_leaf_cap_divisor = 16
    static_cohort_never_readmit = False
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
    two_level_topk = 8
    prefill_two_level_topk: int | None = None
    prefill_max_leaf_tokens: int | None = None
    # Speed diagnostic: leave routing unchanged, but truncate each selected
    # expert's exact posting-list scan.  This intentionally does not restore
    # the unvisited leaves to the coarse branch and is therefore not a
    # quality-valid attention mode.
    prefill_leaf_visit_cap: int | None = None
    # Quality diagnostic: during decode, open every non-protected centroid
    # whose exact posting list is no larger than this cap. Larger centroids
    # remain represented by their coarse mean/count entry. This deliberately
    # bypasses query-dependent routing and is not a speed-optimized path.
    decode_open_all_under_leaf_cap: int | None = None
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
    leaf_short_block_n = 16
    leaf_short_context = 16384
    leaf_num_warps = 2
    leaf_waves_per_eu = 1
    leaf_layout = "query"
    # Exact scalar-vector attention wins through eight leaves on gfx942.  At
    # sixteen leaves the unrolled kernel loses to the regular MMA path.
    leaf_tiny_expert_max = 8
    # As the state grows, large posting lists dominate and the extra bucketed
    # launches stop paying for themselves. Fall back to the same optimized
    # generic expert kernel after this many archived leaves.
    leaf_tiny_max_context: int | None = 65_536
    leaf_tiny_block_m = 8
    leaf_tiny_num_warps = 1
    # Split only exceptionally long centroid posting lists across programs,
    # then merge their normalized output/LSE pairs. Zero/one disables it.
    leaf_long_expert_threshold = 0
    leaf_long_expert_splits = 1
    # Each row merges only the top-eight routes. One wave is sufficient and
    # avoids the synchronization/occupancy overhead of a four-wave reduction.
    leaf_reduce_num_warps = 1
    leaf_union_query_tile = 8
    leaf_aiter_copy_page_size = 16
    # Store flat exact leaves as tokenwise symmetric INT8 and execute both QK
    # and quantized-probability PV with INT8 MMA during expert-layout prefill.
    prefill_int8_leaf_mma = False
    prefill_int8_pv_mma = True
    prefill_int8_coarse_mma = False
    prefill_int8_route_mma = False
    # The INT8 coarse PV kernel benefits from a wider state tile and fewer
    # warps than its BF16 counterpart on gfx942.  Keep this separate so
    # enabling INT8 does not silently retune the existing BF16 path.
    prefill_int8_coarse_block_n = 64
    prefill_int8_coarse_num_warps = 2
    prefill_int8_append_num_warps = 4
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
    recursive_materialize_page_scores = False
    recursive_page_score_block_n = 16
    recursive_page_score_num_warps = 2
    recursive_page_select_block_n = 64
    recursive_state_route_backend = "fused"
    # Prefill-only hybrid used to reuse the regular MoE-style expert/MFMA
    # exact-leaf kernel while still constructing the recursive page archive
    # required by decode.  Selected centroids are evaluated completely during
    # prefill; decode continues to select one page and retain the centroid
    # residual. vLLM automatically selects it only for the measured Phi
    # D128/GQA4/KVH2 BF16 geometry; other installers retain the conservative
    # false default unless they explicitly validate and enable it.
    recursive_prefill_all_leaves = False
    # Prefill-only quadratic/constant-factor alternative: scan every physical
    # page summary as one dense field and exactly replace its best pages.
    # Decode retains ordinary recursive centroid->page routing.
    dense_page_prefill = False
    dense_page_split_kernels = False
    dense_page_topk = 8
    dense_page_block_m = 64
    dense_page_block_n = 64
    dense_page_num_warps = 4
    dense_page_indexed_aiter_union = True
    dense_page_union_query_tile = 32
    dynamic_open_prefill_top_p: float | None = None
    dynamic_open_decode_top_p: float | None = None
    dynamic_open_prefill_residual_mass: float | None = None
    dynamic_open_decode_residual_mass: float | None = None
    # Open every centroid above this share of the estimated attention
    # partition.  Decode uses the preceding token's observed state LSE as its
    # steady-state threshold, allowing one state scan to route and form the
    # complete coarse branch without an ordered top-k reduction.
    prefill_route_mass_fraction: float | None = None
    prefill_route_mass_max_routes = 16
    prefill_route_block_m = 16
    prefill_route_num_warps = 4
    prefill_mass_group_gqa = False
    prefill_mass_full_coarse_first = False
    prefill_mass_previous_chunk_lse = False
    prefill_mass_include_local_lse = True
    prefill_overlap_local_lod = False
    prefill_overlap_coarse_leaf = False
    prefill_mass_streaming_coarse = False
    decode_route_mass_fraction: float | None = None
    decode_route_mass_max_routes = 16
    decode_route_mass_block_n = 64
    decode_route_mass_num_warps = 4
    decode_mass_overlap_coarse = False
    decode_mass_top8_filter = False
    decode_gqa_union_mass_fraction: float | None = None
    decode_gqa_union_predicted_mass = False
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
    decode_route_segment_tiles = 1
    decode_route_num_warps = 2
    decode_route_reduce_num_warps = 4
    decode_route_parallel_reduce = False
    decode_route_post_dot_normalize = False
    decode_route_post_pv_normalize = False
    decode_final_reduce_num_warps = 4
    decode_fuse_final_reduce = False
    decode_route_use_dot = True
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
    # Directly fold the eight route partials into the final branch reduction.
    # The dispatch automatically retains the two-stage tree for 16/32 splits.
    decode_gqa_cooperative_fused_reduce = True
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
    routing_rope_jensen_pairs = 0
    routing_rope_jensen = False
    routing_count_bias = 1.0
    routing_variance_bias = 0.0
    routing_page_mass_candidates = 0
    routing_leaf_mass_candidates = 0
    routing_leaf_mass_objective = "exact"
    routing_leaf_mass_review_top_p: float | None = None
    routing_leaf_mass_top_p: float | None = None
    routing_leaf_mass_min_routes = 1
    coarse_enable_gqa = True
    coarse_compact_bias = True
    reuse_route_logits_for_coarse = True
    coarse_route_block_m = 16
    coarse_route_block_n = 32
    coarse_route_num_warps = 8
    coarse_max_grouped_rows = 8
    prefill_coarse_max_grouped_rows = 8
    prefill_coarse_direct_gqa = False
    prefill_coarse_route_block_n = 32
    prefill_coarse_route_num_warps = 8
    fused_prefill_route_coarse = False
    fused_prefill_stable_recompute = True
    fused_prefill_external_recompute = True
    # VLLM's pool enables the exact two-stage selector only for measured
    # geometries. Standalone/HF callers retain the established selector unless
    # they opt in explicitly.
    prefill_hierarchical_route = False
    fused_prefill_block_m = 16
    fused_prefill_block_n = 32
    fused_prefill_num_warps = 8
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
            scheduled_target = self._desired_state_len(
                capacity_len, capacity_len, 0
            )
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
        destination_order = torch.argsort(
            affinity_destinations, dim=-1, stable=True
        )
        order = affinity_order.gather(-1, destination_order)
        sorted_destination = destination.gather(-1, order)

        positions = torch.arange(
            int(destination.size(-1)),
            dtype=torch.long,
            device=destination.device,
        ).view(1, 1, -1)
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
            ends[..., :-1] = (
                sorted_destination[..., :-1] != sorted_destination[..., 1:]
            )
            sentinel = int(destination.size(-1))
            end_candidates = torch.where(ends, positions, sentinel)
            group_ends = torch.flip(
                torch.cummin(
                    torch.flip(end_candidates, dims=(-1,)), dim=-1
                ).values,
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
            secondary_destination = sorted_destination.gather(
                -1, secondary_order
            )
            regroup = torch.argsort(
                secondary_destination, dim=-1, stable=True
            )
            segmented_order = secondary_order.gather(-1, regroup)
            order = order.gather(-1, segmented_order)
            sorted_destination = sorted_destination.gather(
                -1, segmented_order
            )

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
                        positional_halves=(self.overflow_bipartite_positional_halves),
                    )
                )

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
                    blocked_membership.reshape(batch, full_blocks, heads, block_size)
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
                    torch.arange(tail_len, device=overflow_k.device, dtype=torch.long)
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

    def _premerge_state_inputs_cuda(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use persistent workspaces for fixed-group state inputs."""
        factor = int(self.state_premerge_factor)
        groups = (int(key.size(2)) + factor - 1) // factor
        prefix = (int(key.size(0)), int(key.size(1)))
        buffers = getattr(self, "_lod_premerge_buffers", None)
        needs_buffers = (
            not isinstance(buffers, dict)
            or tuple(buffers["key"].shape[:2]) != prefix
            or int(buffers["key"].size(2)) < groups
            or int(buffers["key"].size(3)) != int(key.size(3))
            or int(buffers["value"].size(3)) != int(value.size(3))
            or buffers["key"].dtype != key.dtype
            or buffers["value"].dtype != value.dtype
            or buffers["key"].device != key.device
        )
        if needs_buffers:
            capacity = _round_up(max(groups, 1), 128)
            buffers = {
                "key": torch.empty(
                    *prefix,
                    capacity,
                    int(key.size(3)),
                    dtype=key.dtype,
                    device=key.device,
                ),
                "value": torch.empty(
                    *prefix,
                    capacity,
                    int(value.size(3)),
                    dtype=value.dtype,
                    device=value.device,
                ),
                "count": torch.empty(
                    *prefix,
                    capacity,
                    1,
                    dtype=torch.float32,
                    device=key.device,
                ),
            }
            self._lod_premerge_buffers = buffers
        return premerge_adjacent_kv(
            key,
            value,
            factor,
            grouped_key=buffers["key"],
            grouped_value=buffers["value"],
            grouped_count=buffers["count"],
        )

    def _expand_premerged_owners_cuda(
        self,
        group_owners: torch.Tensor,
        output_len: int,
    ) -> torch.Tensor:
        output = getattr(self, "_lod_premerge_owner_buffer", None)
        prefix = (int(group_owners.size(0)), int(group_owners.size(1)))
        if (
            not isinstance(output, torch.Tensor)
            or tuple(output.shape[:2]) != prefix
            or int(output.size(2)) < output_len
            or output.dtype != group_owners.dtype
            or output.device != group_owners.device
        ):
            output = torch.empty(
                *prefix,
                _round_up(max(output_len, 1), 256),
                dtype=group_owners.dtype,
                device=group_owners.device,
            )
            self._lod_premerge_owner_buffer = output
        return expand_adjacent_group_owners(
            group_owners,
            output_len,
            int(self.state_premerge_factor),
            output=output,
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
        if self.state_premerge_factor not in {1, 2, 4}:
            raise ValueError("state premerge factor must be one, two, or four")
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
            overflow_select_k = self._mean(overflow_k, overflow_counts)
        elif self.overflow_bipartite_merge:
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
                block_m=self.state_maxsim_block_m,
                block_n=self.state_maxsim_block_n,
                num_warps=self.state_maxsim_num_warps,
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
            logits = torch.matmul(
                q.detach(), self._repeat_kv(mean_k).transpose(-1, -2)
            )
        if mean_end is not None:
            qk_end = torch.cuda.Event(enable_timing=True)
            qk_end.record()
            timing_events.setdefault("route_mean_k", []).append(
                (mean_begin, mean_end)
            )
            timing_events.setdefault("route_qk", []).append((mean_end, qk_end))
        return logits

    @staticmethod
    def _routing_rms_normalize(tensor: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(
            tensor.detach().float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
        )
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

    def _apply_routing_variance_correction(
        self,
        logits: torch.Tensor,
        q: torch.Tensor,
        state_k: torch.Tensor,
        counts: torch.Tensor,
        reference_k: torch.Tensor | None,
        *,
        state_len: int,
        reference_len: int | None = None,
        new_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Approximate omitted within-slot log-mass without growing state."""
        coefficient = float(self.routing_variance_bias)
        if coefficient == 0.0:
            return logits
        if self.routing_normalization != "none":
            raise ValueError(
                "routing variance correction requires unnormalized routing"
            )
        state_count = counts.detach()[..., :state_len, :]
        mean_k = self._mean(
            state_k.detach()[..., :state_len, :],
            state_count,
        )
        mean_sq = mean_k.float().square().sum(-1)
        valid_reference_len = 0
        if reference_k is not None:
            valid_reference_len = int(reference_k.size(2))
            if reference_len is not None:
                valid_reference_len = min(reference_len, valid_reference_len)
        reference_sum = mean_sq.new_zeros(mean_sq.shape[:2])
        if valid_reference_len:
            reference = reference_k.detach()[..., :valid_reference_len, :]
            reference_sum = reference.float().square().sum(dim=-1).sum(dim=-1)
        reference_count = valid_reference_len
        if new_k is not None:
            reference_sum = reference_sum + new_k.detach().float().square().sum(-1).sum(
                -1
            )
            reference_count += int(new_k.size(2))
        if reference_count:
            reference_sq = reference_sum / float(reference_count)
        else:
            singleton = state_count.squeeze(-1).eq(1)
            singleton_count = singleton.sum(-1)
            singleton_mean = mean_sq.masked_fill(~singleton, 0).sum(
                -1
            ) / singleton_count.clamp_min(1)
            # A max-norm centroid is the least-cancelled available proxy when
            # the state has no singleton and there is no exact local field.
            reference_sq = torch.where(
                singleton_count.gt(0), singleton_mean, mean_sq.max(-1).values
            )
        variance_trace = (reference_sq.unsqueeze(-1) - mean_sq).clamp_min(0)
        variance_trace.masked_fill_(state_count.squeeze(-1).le(1), 0)
        variance_trace = self._repeat_kv(variance_trace)
        query_sq = q.detach().float().square().sum(-1)
        # route kernels apply `scaling` after these raw dot-product logits, so
        # divide the desired 0.5 * scaling^2 correction by one scaling here.
        raw_correction = (
            0.5
            * coefficient
            * float(self.scaling)
            * query_sq.unsqueeze(-1)
            * variance_trace.unsqueeze(2)
            / float(q.size(-1))
        )
        return logits + raw_correction.to(logits.dtype)

    def _apply_routing_rope_jensen_correction(
        self,
        logits: torch.Tensor,
        q: torch.Tensor,
        state_k: torch.Tensor,
        counts: torch.Tensor,
        reference_k: torch.Tensor | None,
        *,
        state_len: int,
        reference_len: int | None = None,
        new_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add a per-RoPE-plane second-order omitted-mass estimate."""
        if not self.routing_rope_jensen:
            return logits
        if self.routing_normalization in {"key", "both"}:
            raise ValueError("RoPE Jensen routing does not support key normalization")
        rope_dim = int(self.routing_rope_dim)
        if rope_dim == 0:
            return logits
        half = rope_dim // 2
        route_q = q.detach()
        if self.routing_normalization == "query":
            route_q = self._routing_rms_normalize(route_q)
        elif self.routing_normalization != "none":
            raise ValueError("unsupported normalization for RoPE Jensen routing")
        query_pairs = (
            route_q[..., :half].float().square()
            + route_q[..., half:rope_dim].float().square()
        )

        state_count = counts.detach()[..., :state_len, :]
        mean_k = self._mean(state_k.detach()[..., :state_len, :], state_count).float()
        mean_pairs = mean_k[..., :half].square() + mean_k[..., half:rope_dim].square()
        reference_sum = mean_pairs.new_zeros(*mean_pairs.shape[:2], half)
        valid_reference_len = 0
        if reference_k is not None:
            valid_reference_len = int(reference_k.size(2))
            if reference_len is not None:
                valid_reference_len = min(valid_reference_len, reference_len)
        if valid_reference_len:
            reference = reference_k.detach()[..., :valid_reference_len, :].float()
            reference_sum = reference_sum + (
                reference[..., :half].square() + reference[..., half:rope_dim].square()
            ).sum(-2)
        reference_count = valid_reference_len
        if new_k is not None:
            new_reference = new_k.detach().float()
            reference_sum = reference_sum + (
                new_reference[..., :half].square()
                + new_reference[..., half:rope_dim].square()
            ).sum(-2)
            reference_count += int(new_reference.size(2))
        if reference_count:
            reference_pairs = reference_sum / float(reference_count)
        else:
            singleton = state_count.squeeze(-1).eq(1)
            singleton_count = singleton.sum(-1)
            singleton_pairs = mean_pairs.masked_fill(~singleton.unsqueeze(-1), 0).sum(
                -2
            ) / singleton_count.clamp_min(1).unsqueeze(-1)
            reference_pairs = torch.where(
                singleton_count.gt(0).unsqueeze(-1),
                singleton_pairs,
                mean_pairs.max(-2).values,
            )
        pair_variance = (reference_pairs.unsqueeze(-2) - mean_pairs).clamp_min(0)
        pair_variance.masked_fill_(state_count.le(1), 0)
        pair_variance = self._repeat_kv(pair_variance)
        # The attention path multiplies these raw logits by `scaling` later.
        # Within each rotary plane, covariance trace is the pair-energy
        # deficit, so 0.5 Var(score) becomes 0.25 * scaling here.
        raw_correction = (
            0.25
            * float(self.scaling)
            * torch.matmul(query_pairs, pair_variance.transpose(-1, -2))
        )
        return logits + raw_correction.to(logits.dtype)

    def _closed_state_local_attention(
        self,
        q: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        local_k: torch.Tensor | None,
        local_v: torch.Tensor | None,
        *,
        state_len: int,
        local_len: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the all-closed approximation used by output-error routing."""
        query_counts = self._repeat_kv(counts.detach()[..., :state_len, :]).squeeze(-1)
        state_logits = self._state_route_logits(q, state_k, counts, state_len=state_len)
        state_scores = state_logits.float() * self.scaling + query_counts.clamp_min(
            1
        ).log().unsqueeze(2)
        state_scores.masked_fill_(query_counts.le(0).unsqueeze(2), float("-inf"))
        mean_v = self._repeat_kv(
            self._mean(
                state_v.detach()[..., :state_len, :],
                counts.detach()[..., :state_len, :],
            )
        ).float()
        active_local_len = 0
        if local_k is not None and local_v is not None:
            active_local_len = int(local_k.size(2))
            if local_len is not None:
                active_local_len = min(active_local_len, int(local_len))
        if active_local_len:
            repeated_local_k = self._repeat_kv(
                local_k.detach()[..., :active_local_len, :]
            )
            repeated_local_v = self._repeat_kv(
                local_v.detach()[..., :active_local_len, :]
            ).float()
            local_scores = (
                torch.matmul(q.detach(), repeated_local_k.transpose(-1, -2)).float()
                * self.scaling
            )
            if int(q.size(2)) > 1:
                query_position = torch.arange(
                    int(q.size(2)), device=q.device
                ).unsqueeze(-1)
                key_position = torch.arange(
                    active_local_len, device=q.device
                ).unsqueeze(0)
                query_offset = active_local_len - int(q.size(2))
                local_scores.masked_fill_(
                    key_position > query_offset + query_position,
                    float("-inf"),
                )
            scores = torch.cat((state_scores, local_scores), dim=-1)
            values = torch.cat((mean_v, repeated_local_v), dim=2)
        else:
            scores = state_scores
            values = mean_v
        lse = torch.logsumexp(scores, dim=-1)
        output = torch.matmul(scores.softmax(dim=-1), values)
        return output, lse, state_scores

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
        active_local_len = int(local_k.size(2)) if local_len is None else int(local_len)
        local_scores = (
            torch.matmul(
                q.detach(),
                self._repeat_kv(local_k.detach()[..., :active_local_len, :]).transpose(
                    -1, -2
                ),
            ).float()
            * self.scaling
        )
        local_lse = torch.logsumexp(local_scores, dim=-1)
        if new_k is not None:
            new_score = (q.detach() * self._repeat_kv(new_k.detach())).sum(
                dim=-1
            ).float() * self.scaling
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
        local_len: int | None = None,
        new_k: torch.Tensor | None = None,
        page_cache: dict[str, torch.Tensor | int] | None = None,
        dynamic_local_lse: torch.Tensor | None = None,
        context_len: int | None = None,
    ) -> torch.Tensor:
        configured_topk = (
            self.prefill_route_mass_max_routes
            if int(q.size(2)) > 1 and self.prefill_route_mass_fraction is not None
            else self.decode_route_mass_max_routes
            if int(q.size(2)) == 1 and self.decode_route_mass_fraction is not None
            else self.prefill_two_level_topk
            if int(q.size(2)) > 1 and self.prefill_two_level_topk is not None
            else self.two_level_topk
        )
        protected_len = (
            self._protected_route_len(state_len) if self.exclude_sink_from_routes else 0
        )
        route_count = min(configured_topk, state_len - protected_len)
        seal_capacity = self.leaf_seal_capacity
        if seal_capacity is not None and seal_capacity <= 0:
            raise ValueError("leaf seal capacity must be positive")
        dynamic_target = self._dynamic_open_target(int(q.size(2)))
        dynamic_residual = self._dynamic_open_residual(int(q.size(2)))
        if (
            self.routing_normalization != "none"
            or self.routing_rope_fast_pairs != 0
            or self.routing_rope_jensen
            or self.routing_count_bias != 1.0
            or self.routing_variance_bias != 0.0
            or self.routing_page_mass_candidates != 0
            or self.routing_leaf_mass_candidates != 0
        ) and (dynamic_target is not None or dynamic_residual is not None):
            raise ValueError(
                "routing normalization is not calibrated for dynamic opening"
            )
        with torch.no_grad():
            prefill_route_leaf_cap = self.prefill_max_leaf_tokens
            if int(q.size(2)) > 1 and self.prefill_route_leaf_cap_min is not None:
                if context_len is None or context_len <= 0:
                    raise ValueError(
                        "scheduled prefill cohort routing requires context length"
                    )
                scheduled_cap = max(
                    int(self.prefill_route_leaf_cap_min),
                    math.ceil(math.sqrt(context_len) / self.static_leaf_cap_divisor),
                )
                prefill_route_leaf_cap = (
                    min(prefill_route_leaf_cap, scheduled_cap)
                    if prefill_route_leaf_cap is not None
                    else scheduled_cap
                )
            static_leaf_cap = (
                self.decode_open_all_under_leaf_cap
                if int(q.size(2)) == 1
                else None
            )
            if static_leaf_cap is not None:
                if static_leaf_cap <= 0:
                    raise ValueError("static decode leaf cap must be positive")
                if page_cache is None or not isinstance(
                    page_cache.get("slot_lengths"), torch.Tensor
                ):
                    raise ValueError(
                        "static decode leaf-cap routing requires paged slot lengths"
                    )
                route_lengths = page_cache["slot_lengths"].detach()[
                    ..., :state_len
                ]
                if tuple(route_lengths.shape) != (
                    int(counts.size(0)),
                    int(counts.size(1)),
                    state_len,
                ):
                    raise ValueError(
                        "static decode leaf-cap routing has mismatched slot lengths"
                    )
                query_lengths = self._repeat_kv(route_lengths).to(torch.float32)
                active = query_lengths.gt(0)
                if protected_len:
                    active[..., :protected_len] = False
                opened = active & query_lengths.le(float(static_leaf_cap))
                slots = torch.arange(
                    state_len, device=q.device, dtype=torch.long
                ).view(1, 1, 1, state_len)
                routed = torch.where(
                    opened.unsqueeze(2),
                    slots,
                    torch.full_like(slots, -1),
                ).expand(
                    int(q.size(0)), int(q.size(1)), int(q.size(2)), state_len
                ).contiguous()

                # Measure the ordinary mass-corrected coarse distribution for
                # the same query, without using it to choose routes. Keeping
                # these reductions on-device avoids a synchronization per
                # layer/token; the probe drains them once per generated sample.
                logits = self._state_routing_logits(
                    q,
                    state_k,
                    counts,
                    state_len=state_len,
                )
                logits = self._apply_routing_variance_correction(
                    logits,
                    q,
                    state_k,
                    counts,
                    local_k,
                    state_len=state_len,
                    reference_len=local_len,
                    new_k=new_k,
                )
                logits = self._apply_routing_rope_jensen_correction(
                    logits,
                    q,
                    state_k,
                    counts,
                    local_k,
                    state_len=state_len,
                    reference_len=local_len,
                    new_k=new_k,
                )
                scores = logits.float() * self.scaling
                scores = scores + self.routing_count_bias * query_lengths.clamp_min(
                    1
                ).log().unsqueeze(2)
                scores.masked_fill_(~active.unsqueeze(2), float("-inf"))
                probabilities = torch.softmax(scores, dim=-1)
                # Bins are 1, 2, 3-4, 5-8, ..., 513-1024, and >1024 leaves.
                upper_bounds = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
                mass_parts = []
                centroid_parts = []
                leaf_parts = []
                lower = 0
                for upper in upper_bounds:
                    in_bin = active & query_lengths.gt(lower) & query_lengths.le(upper)
                    mass_parts.append(
                        (probabilities * in_bin.unsqueeze(2)).sum(dim=-1)
                    )
                    centroid_parts.append(in_bin.sum(dim=-1))
                    leaf_parts.append((query_lengths * in_bin).sum(dim=-1))
                    lower = upper
                in_bin = active & query_lengths.gt(upper_bounds[-1])
                mass_parts.append((probabilities * in_bin.unsqueeze(2)).sum(dim=-1))
                centroid_parts.append(in_bin.sum(dim=-1))
                leaf_parts.append((query_lengths * in_bin).sum(dim=-1))
                statistic = {
                    "mass": torch.stack(mass_parts, dim=-1).sum(dim=(0, 1, 2)),
                    "mass_rows": probabilities.new_tensor(
                        int(q.size(0)) * int(q.size(1)) * int(q.size(2))
                    ),
                    "centroids": torch.stack(centroid_parts, dim=-1).sum(dim=(0, 1)),
                    "leaves": torch.stack(leaf_parts, dim=-1).sum(dim=(0, 1)),
                    "opened_leaves": (query_lengths * opened).sum(),
                    "total_leaves": (query_lengths * active).sum(),
                    "opened_centroids": opened.sum(),
                    "total_centroids": active.sum(),
                }
                statistics = getattr(self, "_lod_static_leaf_cap_stats", None)
                if statistics is None:
                    statistics = []
                    self._lod_static_leaf_cap_stats = statistics
                statistics.append(statistic)
                return routed
            if int(q.size(2)) == 1 and self.decode_route_mass_fraction is not None:
                if self.leaf_attention_backend != "paged":
                    raise ValueError(
                        "decode mass-fraction routing requires paged leaf attention"
                    )
                if dynamic_target is not None or dynamic_residual is not None:
                    raise ValueError(
                        "decode mass-fraction routing cannot use dynamic opening"
                    )
                if not 0.0 < float(self.decode_route_mass_fraction) < 1.0:
                    raise ValueError("decode route mass fraction must be in (0, 1)")
                if route_count <= 0 or route_count & (route_count - 1):
                    raise ValueError(
                        "decode mass-fraction routing requires a power-of-two route cap"
                    )
                if (
                    self.routing_normalization != "none"
                    or self.routing_rope_fast_pairs != 0
                    or self.routing_rope_jensen
                    or self.routing_count_bias != 1.0
                    or self.routing_variance_bias != 0.0
                    or self.routing_page_mass_candidates != 0
                    or self.routing_leaf_mass_candidates != 0
                ):
                    raise ValueError(
                        "decode mass-fraction routing supports ordinary centroid scores"
                    )
                state = getattr(self, "_lod_state", None)
                if not isinstance(state, dict):
                    raise RuntimeError("decode mass routing requires an initialized cache")
                buffers = state.get("decode_mass_route_buffers")
                if not isinstance(buffers, dict):
                    buffers = {}
                    state["decode_mass_route_buffers"] = buffers
                predicted_lse = (
                    buffers.get("partition_lse")
                    if bool(state.get("decode_mass_route_seeded", False))
                    else None
                )
                coarse_stream = None
                if self.decode_mass_overlap_coarse and predicted_lse is not None:
                    candidate_stream = state.get("decode_mass_coarse_stream")
                    if not isinstance(candidate_stream, torch.cuda.Stream):
                        candidate_stream = torch.cuda.Stream(device=q.device)
                        state["decode_mass_coarse_stream"] = candidate_stream
                    coarse_stream = candidate_stream
                route_lengths = (
                    page_cache.get("slot_lengths")
                    if page_cache is not None
                    and isinstance(page_cache.get("slot_lengths"), torch.Tensor)
                    else None
                )
                timing_events = getattr(self, "_lod_decode_timing_events", None)
                route_begin = None
                if isinstance(timing_events, dict):
                    route_begin = torch.cuda.Event(enable_timing=True)
                    route_begin.record()
                (
                    routed,
                    route_scores,
                    selected_counts,
                    overflow_counts,
                    observed_lse,
                    coarse_out,
                ) = route_mass_fraction_state_decode(
                    q.contiguous(),
                    state_k.contiguous(),
                    state_v.contiguous(),
                    counts.detach().contiguous(),
                    route_lengths=(
                        route_lengths.detach().contiguous()
                        if isinstance(route_lengths, torch.Tensor)
                        else None
                    ),
                    predicted_lse=(
                        predicted_lse
                        if isinstance(predicted_lse, torch.Tensor)
                        else None
                    ),
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                    count_bias=self.routing_count_bias,
                    mass_fraction=float(self.decode_route_mass_fraction),
                    max_routes=route_count,
                    state_len=state_len,
                    protected_len=protected_len,
                    max_leaf_tokens=(
                        seal_capacity - 1 if seal_capacity is not None else None
                    ),
                    block_n=int(self.decode_route_mass_block_n),
                    num_warps=int(self.decode_route_mass_num_warps),
                    buffers=buffers,
                    timing_events=(
                        timing_events if isinstance(timing_events, dict) else None
                    ),
                    coarse_stream=coarse_stream,
                )
                if route_begin is not None:
                    route_end = torch.cuda.Event(enable_timing=True)
                    route_end.record()
                    timing_events.setdefault("mass_route_coarse", []).append(
                        (route_begin, route_end)
                    )
                    timing_events.setdefault(
                        f"mass_route_coarse_b{int(q.size(0))}", []
                    ).append((route_begin, route_end))
                state["decode_mass_route_seeded"] = True
                self._lod_decode_mass_route_data = (
                    route_scores,
                    coarse_out,
                    observed_lse,
                    coarse_stream,
                )
                self._lod_decode_mass_fraction_stats = (
                    selected_counts,
                    overflow_counts,
                )
                return routed
            mass_fraction = (
                self.prefill_route_mass_fraction if int(q.size(2)) > 1 else None
            )
            if mass_fraction is not None:
                if dynamic_target is not None or dynamic_residual is not None:
                    raise ValueError(
                        "mass-fraction routing cannot be combined with dynamic opening"
                    )
                if not 0.0 < float(mass_fraction) < 1.0:
                    raise ValueError("prefill route mass fraction must be in (0, 1)")
                if route_count <= 0 or route_count & (route_count - 1):
                    raise ValueError(
                        "mass-fraction routing currently requires a power-of-two route cap"
                    )
                if (
                    self.routing_normalization != "none"
                    or self.routing_rope_fast_pairs != 0
                    or self.routing_rope_jensen
                    or self.routing_variance_bias != 0.0
                    or self.routing_page_mass_candidates != 0
                    or self.routing_leaf_mass_candidates != 0
                ):
                    raise ValueError(
                        "the first mass-fraction prefill path supports ordinary centroid scores"
                    )
                logits = self._state_route_logits(
                    q,
                    state_k,
                    counts,
                    state_len=state_len,
                ).contiguous()
                precomputed_state_lse = None
                current_state_log_mass = None
                if self.prefill_mass_previous_chunk_lse:
                    current_state_log_mass = (
                        counts.detach()[..., :state_len, 0]
                        .sum(dim=2)
                        .clamp_min(1.0)
                        .log()
                    )
                    previous_lse = getattr(
                        self, "_lod_prefill_previous_partition_lse", None
                    )
                    previous_state_log_mass = getattr(
                        self, "_lod_prefill_previous_state_log_mass", None
                    )
                    if (
                        isinstance(previous_lse, torch.Tensor)
                        and isinstance(previous_state_log_mass, torch.Tensor)
                        and tuple(previous_lse.shape[:2]) == tuple(q.shape[:2])
                        and int(previous_lse.size(2)) >= int(q.size(2))
                    ):
                        log_mass_growth = self._repeat_kv(
                            current_state_log_mass - previous_state_log_mass
                        ).unsqueeze(-1)
                        precomputed_state_lse = previous_lse[
                            ..., : int(q.size(2))
                        ].add(log_mass_growth).contiguous()
                if self.prefill_mass_full_coarse_first:
                    if self.prefill_mass_previous_chunk_lse:
                        raise ValueError(
                            "previous-LSE routing cannot use full-coarse-first"
                        )
                    if self.prefill_overlap_coarse_leaf:
                        raise ValueError(
                            "full-coarse-first mass routing is a serial diagnostic"
                        )
                    if not self.split_prefill_local_attention:
                        raise ValueError(
                            "full-coarse-first routing requires split local attention"
                        )
                    no_slots = torch.empty(
                        *q.shape[:3], 0, dtype=torch.long, device=q.device
                    )
                    if self.prefill_coarse_direct_gqa:
                        self._lod_prefill_coarse_direct_gqa_executed = (
                            self.prefill_coarse_max_grouped_rows,
                            self.prefill_coarse_route_block_n,
                            self.prefill_coarse_route_num_warps,
                        )
                    full_output, precomputed_state_lse = (
                        route_logits_coarse_attention(
                            q.contiguous(),
                            logits,
                            state_v.contiguous(),
                            counts.contiguous(),
                            local_k[..., :0, :].contiguous(),
                            local_v[..., :0, :].contiguous(),
                            no_slots,
                            state_len=state_len,
                            kv_group_size=self.num_key_value_groups,
                            scale=self.scaling,
                            block_m=self.coarse_route_block_m,
                            block_n=self.prefill_coarse_route_block_n,
                            num_warps=self.prefill_coarse_route_num_warps,
                            precompute_mean_values=True,
                            max_grouped_rows=self.prefill_coarse_max_grouped_rows,
                            direct_gqa_rows=self.prefill_coarse_direct_gqa,
                        )
                    )
                    self._lod_prefill_full_coarse = (
                        full_output,
                        precomputed_state_lse,
                        logits,
                        None,
                    )
                elif (
                    self.prefill_overlap_coarse_leaf
                    and self.prefill_mass_streaming_coarse
                ):
                    if not self.split_prefill_local_attention:
                        raise ValueError(
                            "streaming mass routing requires split local attention"
                        )
                    coarse_stream = getattr(
                        self, "_lod_prefill_coarse_stream", None
                    )
                    if coarse_stream is None:
                        coarse_stream = torch.cuda.Stream(device=q.device)
                        self._lod_prefill_coarse_stream = coarse_stream
                    foreground_stream = torch.cuda.current_stream(q.device)
                    coarse_stream.wait_stream(foreground_stream)
                    no_slots = torch.empty(
                        *q.shape[:3], 0, dtype=torch.long, device=q.device
                    )
                    with torch.cuda.stream(coarse_stream):
                        if self.prefill_coarse_direct_gqa:
                            self._lod_prefill_coarse_direct_gqa_executed = (
                                self.prefill_coarse_max_grouped_rows,
                                self.prefill_coarse_route_block_n,
                                self.prefill_coarse_route_num_warps,
                            )
                        full_coarse = route_logits_coarse_attention(
                            q.contiguous(),
                            logits,
                            state_v.contiguous(),
                            counts.contiguous(),
                            local_k[..., :0, :].contiguous(),
                            local_v[..., :0, :].contiguous(),
                            no_slots,
                            state_len=state_len,
                            kv_group_size=self.num_key_value_groups,
                            scale=self.scaling,
                            block_m=self.coarse_route_block_m,
                            block_n=self.prefill_coarse_route_block_n,
                            num_warps=self.prefill_coarse_route_num_warps,
                            precompute_mean_values=True,
                            max_grouped_rows=self.prefill_coarse_max_grouped_rows,
                            direct_gqa_rows=self.prefill_coarse_direct_gqa,
                        )
                    self._lod_prefill_full_coarse = (
                        *full_coarse,
                        logits,
                        coarse_stream,
                    )
                mass_result = route_mass_fraction_scores(
                    logits,
                    counts.detach().contiguous(),
                    route_lengths=(
                        page_cache["slot_lengths"].detach().contiguous()
                        if page_cache is not None
                        and isinstance(page_cache.get("slot_lengths"), torch.Tensor)
                        and tuple(page_cache["slot_lengths"].shape[:3])
                        == tuple(counts.shape[:3])
                        else None
                    ),
                    state_lse=(
                        precomputed_state_lse.contiguous()
                        if precomputed_state_lse is not None
                        else None
                    ),
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                    count_bias=self.routing_count_bias,
                    mass_fraction=float(mass_fraction),
                    max_routes=route_count,
                    state_len=state_len,
                    protected_len=protected_len,
                    max_leaf_tokens=(
                        min(prefill_route_leaf_cap, seal_capacity - 1)
                        if prefill_route_leaf_cap is not None
                        and seal_capacity is not None
                        else prefill_route_leaf_cap
                        if seal_capacity is None
                        else seal_capacity - 1
                    ),
                    local_lse=(
                        dynamic_local_lse.contiguous()
                        if dynamic_local_lse is not None
                        and not self.prefill_mass_previous_chunk_lse
                        else None
                    ),
                    block_m=int(getattr(self, "prefill_route_block_m", 16)),
                    block_n=128,
                    num_warps=int(getattr(self, "prefill_route_num_warps", 4)),
                    mask_selected_logits=not self.prefill_mass_full_coarse_first,
                    group_gqa=self.prefill_mass_group_gqa,
                    return_partition_lse=(
                        self.prefill_mass_previous_chunk_lse
                        and precomputed_state_lse is None
                    ),
                )
                if (
                    self.prefill_mass_previous_chunk_lse
                    and precomputed_state_lse is None
                ):
                    (
                        routed,
                        selected_counts,
                        overflow_counts,
                        observed_partition_lse,
                    ) = mass_result
                    self._lod_prefill_previous_partition_lse = (
                        observed_partition_lse.detach()
                    )
                    if current_state_log_mass is None:
                        raise AssertionError("state mass predictor was not initialized")
                    self._lod_prefill_previous_state_log_mass = (
                        current_state_log_mass.detach()
                    )
                else:
                    routed, selected_counts, overflow_counts = mass_result
                # The coarse branch consumes the same dense scores, avoiding a
                # second centroid QK pass. The tensors stay on-device so stats
                # collection introduces no synchronization in the timed path.
                if (
                    not self.prefill_mass_full_coarse_first
                    and not (
                        self.prefill_overlap_coarse_leaf
                        and self.prefill_mass_streaming_coarse
                    )
                ):
                    self._lod_prefill_route_logits = logits
                    self._lod_prefill_route_logits_masked = True
                self._lod_mass_fraction_stats = (
                    selected_counts,
                    overflow_counts,
                )
                return routed
            page_mass_candidates = int(self.routing_page_mass_candidates)
            leaf_mass_candidates = int(self.routing_leaf_mass_candidates)
            if (
                (page_mass_candidates or leaf_mass_candidates)
                and page_cache is not None
                and state_len - protected_len > route_count
            ):
                if not self.recursive_page_lod:
                    raise ValueError("page-mass routing requires recursive page LOD")
                if page_mass_candidates and bool(
                    page_cache.get("summary_quantization_finalized", False)
                ):
                    raise ValueError(
                        "page-mass routing does not yet support quantized summaries"
                    )
                candidate_count = min(
                    max(page_mass_candidates, leaf_mass_candidates),
                    state_len - protected_len,
                )
                logits = self._state_routing_logits(
                    q,
                    state_k,
                    counts,
                    state_len=state_len,
                )
                logits = self._apply_routing_variance_correction(
                    logits,
                    q,
                    state_k,
                    counts,
                    local_k,
                    state_len=state_len,
                    reference_len=local_len,
                    new_k=new_k,
                )
                logits = self._apply_routing_rope_jensen_correction(
                    logits,
                    q,
                    state_k,
                    counts,
                    local_k,
                    state_len=state_len,
                    reference_len=local_len,
                    new_k=new_k,
                )
                query_counts = self._repeat_kv(
                    counts.detach()[..., :state_len, :]
                ).squeeze(-1)
                candidate_scores = (
                    logits.float() * self.scaling
                    + self.routing_count_bias
                    * query_counts.clamp_min(1).log().unsqueeze(2)
                )
                candidate_scores.masked_fill_(
                    query_counts.le(0).unsqueeze(2), float("-inf")
                )
                if seal_capacity is not None:
                    candidate_scores.masked_fill_(
                        query_counts.ge(seal_capacity).unsqueeze(2),
                        float("-inf"),
                    )
                candidate_scores[..., :protected_len] = float("-inf")
                candidate_base_scores, candidates = candidate_scores.topk(
                    candidate_count,
                    dim=-1,
                    sorted=self.routing_leaf_mass_review_top_p is not None,
                )
                if self.routing_leaf_mass_review_top_p is not None:
                    # Adapt the transient centroid-review width, not the final
                    # exact-attention route count. Blurry coarse routing has a
                    # diffuse mass distribution and therefore causes more
                    # centroids to be inspected by the finer page hierarchy.
                    complete_coarse_lse = torch.logsumexp(candidate_scores, dim=-1)
                    reviewed_fraction = torch.exp(
                        candidate_base_scores - complete_coarse_lse.unsqueeze(-1)
                    ).cumsum(dim=-1)
                    review_counts = (
                        reviewed_fraction < float(self.routing_leaf_mass_review_top_p)
                    ).sum(dim=-1) + 1
                    review_counts.clamp_(
                        min=route_count,
                        max=candidate_count,
                    )
                    candidate_rank = torch.arange(
                        candidate_count, device=candidates.device
                    )
                    review_mask = candidate_rank < review_counts.unsqueeze(-1)
                    candidates = torch.where(
                        review_mask,
                        candidates,
                        torch.full_like(candidates, -1),
                    )
                    candidate_base_scores = candidate_base_scores.masked_fill(
                        ~review_mask, float("-inf")
                    )
                    self._record_dynamic_open_counts(
                        review_counts,
                        route_count=candidate_count,
                        statistic="review",
                    )
                virtual_leaf_mass = bool(
                    leaf_mass_candidates and self.virtual_page_storage
                )
                if virtual_leaf_mass:
                    page_names = (
                        ("leaf_k", "leaf_v")
                        if self.routing_leaf_mass_objective == "output"
                        else ("leaf_k",)
                    ) + (
                        "page_indices",
                        "page_counts",
                        "slot_pages",
                        "overflow_page_keys",
                        "overflow_page_values",
                        "overflow_used",
                        "slot_lengths",
                    )
                else:
                    page_names = (
                        "page_k" if leaf_mass_candidates else "page_sum_k",
                        "page_counts",
                        "slot_pages",
                        "overflow_page_keys",
                        "overflow_page_values",
                        "overflow_used",
                        "slot_lengths",
                    )
                page_tensors = tuple(page_cache.get(name) for name in page_names)
                if not all(isinstance(value, torch.Tensor) for value in page_tensors):
                    raise RuntimeError("page-mass routing cache is incomplete")
                refinement_kwargs = {
                    "kv_group_size": self.num_key_value_groups,
                    "scale": self.scaling,
                    "hash_probes": self._page_lookup_probes(page_cache),
                    "page_size": int(page_cache["page_size"]),
                }
                if (
                    leaf_mass_candidates
                    and self.routing_leaf_mass_objective == "output"
                ):
                    if not virtual_leaf_mass:
                        raise ValueError(
                            "output-error routing requires virtual page storage"
                        )
                    closed_output, closed_lse, closed_state_scores = (
                        self._closed_state_local_attention(
                            q,
                            state_k,
                            state_v,
                            counts,
                            local_k,
                            local_v,
                            state_len=state_len,
                            local_len=local_len,
                        )
                    )
                    candidate_coarse_scores = torch.gather(
                        closed_state_scores, -1, candidates.clamp_min(0)
                    )
                    mass_scores = refine_route_candidates_by_virtual_leaf_output(
                        q,
                        closed_output,
                        closed_lse,
                        candidate_coarse_scores,
                        state_v.detach(),
                        counts.detach()[..., 0],
                        *page_tensors,
                        candidates,
                        **refinement_kwargs,
                    )
                elif leaf_mass_candidates:
                    refine_leaf_mass = (
                        refine_route_candidates_by_virtual_leaf_mass
                        if virtual_leaf_mass
                        else refine_route_candidates_by_leaf_mass
                    )
                    if self.routing_leaf_mass_objective in {
                        "rope_jensen",
                        "fast_rope_jensen",
                        "slow_rope_jensen",
                    }:
                        rope_dim = int(self.routing_rope_dim)
                        rope_pairs = rope_dim // 2
                        cutoff_pairs = int(self.routing_rope_jensen_pairs)
                        if self.routing_leaf_mass_objective == "fast_rope_jensen":
                            jensen_pair_start = 0
                            jensen_pairs = cutoff_pairs
                        elif self.routing_leaf_mass_objective == "slow_rope_jensen":
                            jensen_pair_start = cutoff_pairs
                            jensen_pairs = rope_pairs - cutoff_pairs
                        else:
                            jensen_pair_start = 0
                            jensen_pairs = rope_pairs
                        if rope_dim == 0 or jensen_pairs == 0:
                            mass_scores = candidate_base_scores
                        else:
                            if self.routing_normalization in {"key", "both"}:
                                raise ValueError(
                                    "RoPE Jensen routing does not support key "
                                    "normalization"
                                )
                            route_q = q.detach()
                            if self.routing_normalization == "query":
                                route_q = self._routing_rms_normalize(route_q)
                            elif self.routing_normalization != "none":
                                raise ValueError(
                                    "unsupported normalization for RoPE Jensen routing"
                                )
                            rope_half = rope_dim // 2
                            jensen_dim = 2 * jensen_pairs
                            # The paged mass kernel vectorizes over a
                            # power-of-two head width. Zero padding this
                            # transient band-only view preserves its dot
                            # products while allowing architecture-derived
                            # cutoffs such as Gemma's 82 dimensions.
                            kernel_dim = 1 << (jensen_dim - 1).bit_length()

                            def select_jensen_pairs(
                                tensor: torch.Tensor,
                            ) -> torch.Tensor:
                                selected = torch.cat(
                                    (
                                        tensor[
                                            ...,
                                            jensen_pair_start : jensen_pair_start
                                            + jensen_pairs,
                                        ],
                                        tensor[
                                            ...,
                                            rope_half + jensen_pair_start : rope_half
                                            + jensen_pair_start
                                            + jensen_pairs,
                                        ],
                                    ),
                                    dim=-1,
                                )
                                return F.pad(
                                    selected, (0, kernel_dim - jensen_dim)
                                ).contiguous()

                            jensen_query = select_jensen_pairs(route_q)
                            jensen_page_tensors = (
                                select_jensen_pairs(page_tensors[0]),
                                *page_tensors[1:],
                            )
                            exact_jensen_mass = refine_leaf_mass(
                                jensen_query,
                                *jensen_page_tensors,
                                candidates,
                                **refinement_kwargs,
                            )
                            state_count = counts.detach()[..., :state_len, :]
                            mean_k = self._mean(
                                state_k.detach()[..., :state_len, :],
                                state_count,
                            )
                            jensen_logits = torch.matmul(
                                jensen_query,
                                self._repeat_kv(select_jensen_pairs(mean_k)).transpose(
                                    -1, -2
                                ),
                            )
                            jensen_scores = (
                                jensen_logits.float() * self.scaling
                                + query_counts.clamp_min(1).log().unsqueeze(2)
                            )
                            candidate_jensen_scores = torch.gather(
                                jensen_scores, -1, candidates.clamp_min(0)
                            )
                            jensen_gap = (
                                exact_jensen_mass - candidate_jensen_scores
                            ).clamp_min(0)
                            mass_scores = candidate_base_scores + jensen_gap
                    else:
                        mass_scores = refine_leaf_mass(
                            q, *page_tensors, candidates, **refinement_kwargs
                        )
                else:
                    mass_scores = refine_route_candidates_by_page_mass(
                        q,
                        *page_tensors,
                        candidates,
                        page_block_n=self.recursive_page_block_n,
                        **refinement_kwargs,
                    )
                if leaf_mass_candidates and self.routing_leaf_mass_objective not in {
                    "exact",
                    "output",
                    "rope_jensen",
                    "fast_rope_jensen",
                    "slow_rope_jensen",
                }:
                    exact_coarse_logits = self._state_route_logits(
                        q,
                        state_k,
                        counts,
                        state_len=state_len,
                    )
                    exact_coarse_scores = (
                        exact_coarse_logits.float() * self.scaling
                        + query_counts.clamp_min(1).log().unsqueeze(2)
                    )
                    candidate_coarse_scores = torch.gather(
                        exact_coarse_scores, -1, candidates.clamp_min(0)
                    )
                    log_mass_deficit = (
                        mass_scores - candidate_coarse_scores
                    ).clamp_min(0.0)
                    if self.routing_leaf_mass_objective == "deficit":
                        mass_scores = log_mass_deficit
                    elif self.routing_leaf_mass_objective == "additional":
                        # log(Z_exact - Z_centroid), evaluated stably.  This
                        # ranks the mass that the closed centroid omits rather
                        # than the slot's total popularity.
                        mass_scores = mass_scores + torch.log(
                            -torch.expm1(-log_mass_deficit)
                        )
                    else:
                        raise ValueError(
                            "leaf-mass objective must be exact, additional, deficit, "
                            "output, rope_jensen, fast_rope_jensen, or "
                            "slow_rope_jensen"
                        )
                selected_scores, selected_candidates = mass_scores.topk(
                    route_count,
                    dim=-1,
                    sorted=self.routing_leaf_mass_top_p is not None,
                )
                routed = torch.gather(candidates, -1, selected_candidates)
                if self.routing_leaf_mass_top_p is not None:
                    # Correct the complete remote-field partition estimate by
                    # replacing each reviewed centroid's coarse mass with its
                    # page-refined mass. This leaves unreviewed centroids in
                    # the denominator instead of normalizing top-p over only
                    # the shortlisted routes.
                    coarse_max = candidate_scores.amax(dim=-1)
                    refined_max = mass_scores.amax(dim=-1)
                    anchor = torch.maximum(coarse_max, refined_max)
                    coarse_mass = torch.exp(
                        candidate_scores - anchor.unsqueeze(-1)
                    ).sum(dim=-1)
                    reviewed_coarse_mass = torch.exp(
                        candidate_base_scores - anchor.unsqueeze(-1)
                    ).sum(dim=-1)
                    reviewed_refined_mass = torch.exp(
                        mass_scores - anchor.unsqueeze(-1)
                    ).sum(dim=-1)
                    corrected_total_mass = (
                        coarse_mass - reviewed_coarse_mass + reviewed_refined_mass
                    ).clamp_min(torch.finfo(torch.float32).tiny)
                    corrected_total_lse = anchor + corrected_total_mass.log()
                    cumulative_fraction = torch.exp(
                        selected_scores - corrected_total_lse.unsqueeze(-1)
                    ).cumsum(dim=-1)
                    open_counts = (
                        cumulative_fraction < float(self.routing_leaf_mass_top_p)
                    ).sum(dim=-1) + 1
                    open_counts.clamp_(
                        min=min(int(self.routing_leaf_mass_min_routes), route_count),
                        max=route_count,
                    )
                    rank = torch.arange(route_count, device=routed.device)
                    routed = torch.where(
                        rank < open_counts.unsqueeze(-1),
                        routed,
                        torch.full_like(routed, -1),
                    )
                    self._record_dynamic_open_counts(
                        open_counts, route_count=route_count
                    )
                return routed if int(q.size(2)) == 1 else routed.clone()
            if (
                self.fused_prefill_route_coarse
                and self.reuse_route_logits_for_coarse
                and int(q.size(-1)) <= 512
                and int(state_v.size(-1)) <= 256
                and self.routing_normalization == "none"
                and self.routing_rope_fast_pairs == 0
                and not self.routing_rope_jensen
                and self.routing_count_bias == 1.0
                and self.routing_variance_bias == 0.0
                and q.is_cuda
                and int(q.size(2)) > 1
                and 0 < route_count <= 8
                and dynamic_target is None
                and not getattr(self, "_lod_collect_stats", False)
                and (
                    self.leaf_layout not in ("aiter_union", "aiter_masked_union")
                    or getattr(self, "mla_state_key_normalization", "none") == "none"
                )
            ):
                if local_k is None or local_v is None:
                    raise AssertionError("fused prefill routing has no local KV")
                fused_state_qk = bool(
                    self.prefill_int8_route_mma
                    and getattr(self, "mla_state_key_normalization", "none") == "none"
                )
                logits = (
                    q
                    if fused_state_qk
                    else self._state_route_logits(
                        q,
                        state_k,
                        counts,
                        state_len=state_len,
                    )
                )
                include_local = not self.split_prefill_local_attention
                coarse_local_k = local_k if include_local else local_k[..., :0, :]
                coarse_local_v = local_v if include_local else local_v[..., :0, :]
                route_select_begin = None
                timing_events = getattr(self, "_lod_phase_timing_events", None)
                if isinstance(timing_events, dict):
                    route_select_begin = torch.cuda.Event(enable_timing=True)
                    route_select_begin.record()
                routed, coarse_output, coarse_lse = route_logits_topk_coarse_attention(
                    q,
                    logits.contiguous(),
                    state_v.contiguous(),
                    counts.contiguous(),
                    coarse_local_k.contiguous(),
                    coarse_local_v.contiguous(),
                    state_len=state_len,
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                    route_count_bias=self.routing_count_bias,
                    topk=route_count,
                    protected_len=protected_len,
                    max_leaf_tokens=(
                        min(prefill_route_leaf_cap, seal_capacity - 1)
                        if prefill_route_leaf_cap is not None
                        and seal_capacity is not None
                        else prefill_route_leaf_cap
                        if seal_capacity is None
                        else seal_capacity - 1
                    ),
                    residual_local_lse=(
                        dynamic_local_lse.contiguous()
                        if dynamic_local_lse is not None
                        else None
                    ),
                    residual_mass=dynamic_residual,
                    block_m=(8 if fused_state_qk else self.fused_prefill_block_m),
                    block_n=self.fused_prefill_block_n,
                    num_warps=self.fused_prefill_num_warps,
                    stable_recompute=self.fused_prefill_stable_recompute,
                    route_only=(
                        not fused_state_qk
                        and self.fused_prefill_stable_recompute
                        and self.fused_prefill_external_recompute
                    ),
                    state_k=(state_k.contiguous() if fused_state_qk else None),
                    int8_qk=fused_state_qk,
                    hierarchical_route_only=(
                        self.prefill_hierarchical_route
                        and not fused_state_qk
                        and self.fused_prefill_stable_recompute
                        and self.fused_prefill_external_recompute
                        and route_count == 3
                    ),
                )
                if route_select_begin is not None:
                    route_select_end = torch.cuda.Event(enable_timing=True)
                    route_select_end.record()
                    timing_events.setdefault("route_select", []).append(
                        (route_select_begin, route_select_end)
                    )
                if self.collect_dynamic_open_stats and dynamic_residual is not None:
                    open_counts = (routed >= 0).sum(dim=-1)
                    histogram = torch.bincount(
                        open_counts.reshape(-1), minlength=route_count + 1
                    )
                    if not hasattr(self, "_lod_dynamic_prefill_histograms"):
                        self._lod_dynamic_prefill_histograms = []
                    self._lod_dynamic_prefill_histograms.append(histogram)
                if (
                    not fused_state_qk
                    and self.fused_prefill_stable_recompute
                    and self.fused_prefill_external_recompute
                ):
                    # The fused first scan selects routes without temporary
                    # group buffers. Reuse those logits in the established
                    # coarse kernel so stable mode remains numerically
                    # identical to the quality baseline.
                    self._lod_prefill_route_logits = logits
                else:
                    self._lod_prefill_fused_coarse = (
                        coarse_output,
                        coarse_lse,
                        include_local,
                    )
                return routed
            if (
                self.fused_state_routing
                and seal_capacity is None
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
                            tuple(buffers["state_lse"].shape[:2]) != tuple(q.shape[:2])
                            or int(buffers["state_lse"].size(2)) < int(q.size(2))
                            or int(buffers["partial_lse"].size(3)) < required_groups
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
                    and self.routing_normalization == "none"
                    and self.routing_rope_fast_pairs == 0
                    and not self.routing_rope_jensen
                    and self.routing_variance_bias == 0.0
                    and dynamic_target is None
                    and dynamic_residual is None
                    and not (self.reuse_route_logits_for_coarse and int(q.size(2)) > 1)
                ):
                    routed = route_top8_state_grouped(
                        q.detach(),
                        state_k.detach(),
                        counts.detach(),
                        buffers,
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        count_bias=self.routing_count_bias,
                        topk=route_count,
                        state_len=state_len,
                        protected_len=protected_len,
                        reorder_like_torch=True,
                    )
                else:
                    logits = self._state_routing_logits(
                        q,
                        state_k,
                        counts,
                        state_len=state_len,
                    )
                    logits = self._apply_routing_variance_correction(
                        logits,
                        q,
                        state_k,
                        counts,
                        local_k,
                        state_len=state_len,
                        reference_len=local_len,
                        new_k=new_k,
                    )
                    logits = self._apply_routing_rope_jensen_correction(
                        logits,
                        q,
                        state_k,
                        counts,
                        local_k,
                        state_len=state_len,
                        reference_len=local_len,
                        new_k=new_k,
                    )
                    if (
                        self.routing_normalization == "none"
                        and self.routing_rope_fast_pairs == 0
                        and not self.routing_rope_jensen
                        and self.routing_count_bias == 1.0
                        and self.routing_variance_bias == 0.0
                        and self.reuse_route_logits_for_coarse
                        and int(q.size(2)) > 1
                    ):
                        self._lod_prefill_route_logits = logits
                    if (
                        self.prefill_hierarchical_route
                        and int(q.size(2)) > 1
                        and route_count == 3
                        and dynamic_residual is None
                    ):
                        hierarchical_block_n = min(
                            1024,
                            max(256, 1 << (state_len - 1).bit_length()),
                        )
                        route_result = route_logits_hierarchical_topk(
                            logits.contiguous(),
                            counts.detach().contiguous(),
                            state_len=state_len,
                            kv_group_size=self.num_key_value_groups,
                            scale=self.scaling,
                            route_count_bias=self.routing_count_bias,
                            topk=route_count,
                            protected_len=protected_len,
                            # Match the established grouped fallback, which
                            # applies no per-centroid leaf cap during routing.
                            max_leaf_tokens=None,
                            block_m=8,
                            block_n=hierarchical_block_n,
                            tile_num_warps=2,
                            reduce_num_warps=2,
                        )
                    else:
                        route_result = route_top8_scores_grouped(
                            logits,
                            counts.detach(),
                            buffers,
                            kv_group_size=self.num_key_value_groups,
                            scale=self.scaling,
                            count_bias=self.routing_count_bias,
                            topk=route_count,
                            state_len=state_len,
                            protected_len=protected_len,
                            return_lse=dynamic_residual is not None,
                            block_m=(
                                int(getattr(self, "prefill_route_block_m", 16))
                                if int(q.size(2)) > 1
                                else 1
                            ),
                            num_warps=(
                                int(getattr(self, "prefill_route_num_warps", 4))
                                if int(q.size(2)) > 1
                                else 4
                            ),
                            # Prefill may open fewer than the returned route count,
                            # so the selected prefix must remain score ordered.
                            reorder_like_torch=True,
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
                    corrected = (
                        logits * self.scaling
                        + self.routing_count_bias * query_counts.log().unsqueeze(2)
                    )
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
                        query_counts.unsqueeze(2).expand(-1, -1, int(q.size(2)), -1),
                        -1,
                        routed,
                    )
                    selected_scores = (
                        selected_logits * self.scaling
                    ).float() + selected_counts.log()
                    full_lse = None
                    if dynamic_residual is not None:
                        if routed_state_lse is None:
                            raise AssertionError(
                                "dynamic routing did not return state LSE"
                            )
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
            logits = self._state_routing_logits(
                q,
                state_k,
                counts,
                state_len=state_len,
            )
            logits = self._apply_routing_variance_correction(
                logits,
                q,
                state_k,
                counts,
                local_k,
                state_len=state_len,
                reference_len=local_len,
                new_k=new_k,
            )
            logits = self._apply_routing_rope_jensen_correction(
                logits,
                q,
                state_k,
                counts,
                local_k,
                state_len=state_len,
                reference_len=local_len,
                new_k=new_k,
            )
            if (
                self.reuse_route_logits_for_coarse
                and int(q.size(2)) == 1
                and self.routing_normalization in {"none", "query"}
                and self.routing_rope_fast_pairs == 0
                and not self.routing_rope_jensen
                and self.routing_variance_bias == 0.0
            ):
                # Decode used to scan the MLA state twice: once with the
                # routing query and again with the attention query.  For
                # query-only routing, recover the raw attention logits with
                # the scalar query RMS; no state-sized tensor is added.
                query_rms = (
                    q.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
                    if self.routing_normalization == "query"
                    else None
                )
                self._lod_decode_route_logits = (logits, query_rms)
            if (
                self.routing_normalization == "none"
                and self.routing_rope_fast_pairs == 0
                and not self.routing_rope_jensen
                and self.routing_count_bias == 1.0
                and self.routing_variance_bias == 0.0
                and self.reuse_route_logits_for_coarse
                and int(q.size(2)) > 1
                and route_count <= 8
            ):
                self._lod_prefill_route_logits = logits
            logits = logits * self.scaling
            query_counts = self._repeat_kv(counts.detach()[..., :state_len, :]).squeeze(
                -1
            )
            log_counts = query_counts.log()
            logits = logits + self.routing_count_bias * log_counts.unsqueeze(2)
            if seal_capacity is not None:
                logits.masked_fill_(
                    query_counts.ge(seal_capacity).unsqueeze(2),
                    float("-inf"),
                )
            logits[..., :protected_len] = float("-inf")
            top_slots = logits.topk(route_count, dim=-1, sorted=False).indices
            if getattr(self, "_lod_collect_stats", False):
                self._lod_ever_selected_slots.scatter_(
                    -1,
                    top_slots.reshape(
                        int(top_slots.size(0)),
                        int(top_slots.size(1)),
                        -1,
                    ),
                    True,
                )
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
                open_sorted = (rank == 0) | (remaining_before > residual_mass)
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

    def _record_dynamic_open_counts(
        self,
        open_counts: torch.Tensor,
        *,
        route_count: int,
        statistic: str = "open",
    ) -> None:
        """Accumulate dynamic route counts on the engine and its HF owner."""
        if not self.collect_dynamic_open_stats:
            return
        histogram = torch.bincount(open_counts.reshape(-1), minlength=route_count + 1)
        phase = "decode" if int(open_counts.size(2)) == 1 else "prefill"
        if statistic not in {"open", "review"}:
            raise ValueError("dynamic route statistic must be open or review")
        statistic_prefix = "" if statistic == "open" else f"{statistic}_"
        attribute = f"_lod_dynamic_{statistic_prefix}{phase}_histogram"
        current = getattr(self, attribute, None)
        setattr(self, attribute, histogram if current is None else current + histogram)
        owner_reference = getattr(self, "_lod_dynamic_stats_owner", None)
        owner = owner_reference() if callable(owner_reference) else None
        if owner is not None:
            owner_current = getattr(owner, attribute, None)
            setattr(
                owner,
                attribute,
                histogram if owner_current is None else owner_current + histogram,
            )

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
            slot_page_capacity = self.leaf_inline_pages_per_slot
            slot_page_dtype = (
                torch.int16
                if page_capacity <= torch.iinfo(torch.int16).max
                else torch.int32
            )
            # Most contexts never overflow the compact inline posting lists.
            # Keep only a sentinel allocation until a slot actually needs the
            # hash table; kernels with HASH_PROBES=0 never dereference it.
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
        cache: dict[str, torch.Tensor | int] = {
            # These pages are allocated from per-region postings below.  This
            # is a semantic contract, not a description of the physical flat
            # leaf backing used by virtual-page storage.
            "region_owned_pages": True,
            "slot_pages": torch.full(
                (
                    batch,
                    kv_heads,
                    state_capacity,
                    slot_page_capacity,
                ),
                -1,
                dtype=slot_page_dtype,
                device=k.device,
            ),
            "overflow_page_keys": overflow_page_keys,
            "overflow_page_values": overflow_page_values,
            "overflow_hash_capacity": hash_capacity,
            "overflow_flag": torch.zeros((), dtype=torch.int32, device=k.device),
            "overflow_used": overflow_used,
            "overflow_active": False,
            "overflow_safe_until": slot_page_capacity * page_size,
            "paged_page_directory": paged_page_directory,
            "page_directory_size": _PAGE_DIRECTORY_SIZE,
            "slot_lengths": torch.zeros(
                batch,
                kv_heads,
                state_capacity,
                dtype=torch.int32,
                device=k.device,
            ),
            # Diagnostic state for the monotone static cohort. Zero is an
            # unseen centroid slot, one is still exact-leaf eligible, and
            # minus one means its leaves crossed a prior schedule limit and
            # must never be re-admitted. The compact allocator can consume
            # this field directly once eviction is enabled.
            "static_cohort_status": torch.zeros(
                batch,
                kv_heads,
                (
                    state_capacity
                    if (
                        os.getenv("LOD_STATIC_COHORT_EVICTION_DIAGNOSTICS") == "1"
                        or self.static_cohort_never_readmit
                    )
                    else 0
                ),
                dtype=torch.int8,
                device=k.device,
            ),
            "next_page": torch.zeros(
                batch, kv_heads, dtype=torch.int32, device=k.device
            ),
            "page_size": page_size,
            "leaf_capacity": sequence_capacity,
            "leaf_count": 0,
            "mla_raw_page_key_summaries": raw_page_key_summaries,
        }
        if self.virtual_page_storage:
            matching_native_quant = (
                self.leaf_key_quant_bits in (4, 8)
                and self.leaf_value_quant_bits == self.leaf_key_quant_bits
            )
            simulate_quantization = bool(
                getattr(self, "simulate_leaf_quantization", False)
            )
            virtual_native_quant = matching_native_quant and not simulate_quantization
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
                    quantized_leaf_k=torch.empty(
                        batch,
                        kv_heads,
                        sequence_capacity,
                        key_width,
                        dtype=quant_dtype,
                        device=k.device,
                    ),
                    quantized_leaf_v=torch.empty(
                        batch,
                        kv_heads,
                        sequence_capacity,
                        value_width,
                        dtype=quant_dtype,
                        device=v.device,
                    ),
                    page_k_scales=torch.empty(
                        batch,
                        kv_heads,
                        page_capacity,
                        token_groups * (head_dim // group_size),
                        dtype=k.dtype,
                        device=k.device,
                    ),
                    page_v_scales=torch.empty(
                        batch,
                        kv_heads,
                        page_capacity,
                        token_groups * (value_dim // group_size),
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
        needs_page_summaries = (
            self.virtual_page_storage
            or self.recursive_page_lod
            or bool(self.leaf_key_quant_bits or self.leaf_value_quant_bits)
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
        raw_page_summary_k = (
            k if bool(cache.get("mla_raw_page_key_summaries", False)) else None
        )
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
        if virtual_pages and self.leaf_seal_capacity is not None:
            raise NotImplementedError(
                "sealed leaf archival is currently implemented for flat paged leaves"
            )
        if not virtual_pages and not isinstance(page_k, torch.Tensor):
            raise TypeError("page cache K tensor is missing")
        if not virtual_pages and not isinstance(page_v, torch.Tensor):
            raise TypeError("page cache V tensor is missing")
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
            if not all(
                isinstance(value, torch.Tensor)
                for value in (leaf_k, leaf_v, page_sum_k, page_sum_v, page_counts)
            ):
                raise TypeError("virtual page cache tensors are incomplete")
            if bool(cache.get("prefill_int8_leaf_mma", False)):
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
                    leaf_k_token_scales=cache.get("page_k_token_scales"),
                    leaf_v_token_scales=cache.get("page_v_token_scales"),
                    source_k=k,
                    source_v=v,
                    raw_page_summary_k=raw_page_summary_k,
                )
            elif bool(cache.get("quantization_finalized", False)):
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
                    quantized_leaf_k=cache.get("quantized_leaf_k"),
                    quantized_leaf_v=cache.get("quantized_leaf_v"),
                    page_k_scales=cache.get("page_k_scales"),
                    page_v_scales=cache.get("page_v_scales"),
                    page_quantized_counts=cache.get("page_quantized_counts"),
                    quant_group_size=self.leaf_quant_group_size,
                    quant_token_group_size=self.leaf_quant_token_group_size,
                    quant_bits=int(cache.get("leaf_quant_bits", 4)),
                    quantize_touched=False,
                    optimize_scale=(self.leaf_quant_scale_mode == "l2"),
                    fake_key_quant_bits=(
                        self.leaf_key_quant_bits
                        if not isinstance(cache.get("quantized_leaf_k"), torch.Tensor)
                        else 0
                    ),
                    fake_value_quant_bits=(
                        self.leaf_value_quant_bits
                        if not isinstance(cache.get("quantized_leaf_v"), torch.Tensor)
                        else 0
                    ),
                    fake_quantize_summaries=(
                        bool(self.leaf_key_quant_bits or self.leaf_value_quant_bits)
                        and self.page_summary_quant_bits == 8
                    ),
                    optimize_summary_scale=(self.page_summary_scale_mode == "l2"),
                    raw_page_summary_k=raw_page_summary_k,
                )
        else:
            if bool(cache.get("prefill_int8_leaf_mma", False)):
                page_k_token_scales = cache.get("page_k_token_scales")
                page_v_token_scales = cache.get("page_v_token_scales")
                if not isinstance(page_k_token_scales, torch.Tensor) or not isinstance(
                    page_v_token_scales, torch.Tensor
                ):
                    raise RuntimeError("INT8 flat page scales are missing")
                append_paged_int8_kv(
                    k,
                    v,
                    owners.contiguous(),
                    page_k,
                    page_v,
                    page_k_token_scales,
                    page_v_token_scales,
                    slot_pages,
                    overflow_page_keys,
                    overflow_page_values,
                    overflow_used,
                    overflow_flag,
                    slot_lengths,
                    next_page,
                    hash_probes=append_hash_probes,
                    max_leaf_tokens=self.leaf_seal_capacity,
                    num_warps=self.prefill_int8_append_num_warps,
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
                    page_sum_k=(
                        page_sum_k if isinstance(page_sum_k, torch.Tensor) else None
                    ),
                    page_sum_v=(
                        page_sum_v if isinstance(page_sum_v, torch.Tensor) else None
                    ),
                    page_counts=(
                        page_counts if isinstance(page_counts, torch.Tensor) else None
                    ),
                    raw_page_summary_k=raw_page_summary_k,
                    max_leaf_tokens=self.leaf_seal_capacity,
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
        page_indices = cache.get("page_indices")
        indexed = isinstance(page_indices, torch.Tensor)
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
        leaf_function = {
            "query": query_major_paged_leaf_attention,
            "expert": paged_leaf_attention,
            "expert_tiny": paged_leaf_attention,
            "aiter_varlen": aiter_varlen_paged_leaf_attention,
            "aiter_copy": aiter_varlen_paged_leaf_attention,
            "aiter_union": aiter_query_tile_union_paged_leaf_attention,
            "aiter_masked_union": aiter_query_tile_union_paged_leaf_attention,
        }.get(self.leaf_layout)
        if leaf_function is None:
            raise ValueError(f"unknown leaf attention layout {self.leaf_layout!r}")
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
                "hash_probes": self._page_lookup_probes(cache),
                "num_warps": self.leaf_num_warps,
                "waves_per_eu": self.leaf_waves_per_eu,
                "timing_events": getattr(self, "_lod_leaf_timing_events", None),
            }
        elif self.leaf_layout in (
            "expert",
            "expert_tiny",
            "aiter_varlen",
            "aiter_copy",
            "aiter_union",
            "aiter_masked_union",
        ):
            block_m = self.leaf_block_m
            block_n = self.leaf_block_n
            num_warps = self.leaf_num_warps
            if (
                self.leaf_geometry_tuning
                and int(q.size(-1)) == 128
                and int(q.size(2)) > 1
                and page_k.dtype != torch.int8
                and (block_m, block_n, num_warps) == (16, 32, 2)
            ):
                block_m, block_n, num_warps = 64, 64, 4
            leaf_kwargs = {
                "block_m": block_m,
                "block_n": block_n,
                "hash_probes": self._page_lookup_probes(cache),
                "num_warps": num_warps,
                "waves_per_eu": self.leaf_waves_per_eu,
                "timing_events": getattr(self, "_lod_leaf_timing_events", None),
            }
            if self.leaf_layout in ("expert", "expert_tiny"):
                leaf_kwargs["reduce_num_warps"] = self.leaf_reduce_num_warps
                if (
                    int(q.size(2)) > 1
                    and self.prefill_leaf_visit_cap is not None
                ):
                    leaf_kwargs["max_leaves_per_expert"] = int(
                        self.prefill_leaf_visit_cap
                    )
                leaf_kwargs["compact_invalid_routes"] = bool(
                    self.prefill_route_mass_fraction is not None
                    and int(q.size(2)) > 1
                )
                leaf_kwargs.update(
                    long_expert_threshold=self.leaf_long_expert_threshold,
                    long_expert_splits=self.leaf_long_expert_splits,
                )
            if self.leaf_layout in ("aiter_union", "aiter_masked_union"):
                leaf_kwargs["query_tile"] = self.leaf_union_query_tile
            tiny_expert_eligible = self.leaf_layout == "expert_tiny" and (
                self.leaf_tiny_max_context is None
                or int(cache["leaf_count"]) <= self.leaf_tiny_max_context
            )
            if tiny_expert_eligible:
                leaf_kwargs.update(
                    tiny_expert_max=self.leaf_tiny_expert_max,
                    tiny_block_m=self.leaf_tiny_block_m,
                    tiny_num_warps=self.leaf_tiny_num_warps,
                )
            if self.leaf_layout == "aiter_copy":
                leaf_kwargs.update(
                    copy_indexed_kv=True,
                    copy_page_size=self.leaf_aiter_copy_page_size,
                )
            if self.leaf_layout == "aiter_masked_union":
                leaf_kwargs["mask_queries"] = True
            if bool(cache.get("prefill_int8_leaf_mma", False)):
                if self.leaf_layout not in ("expert", "expert_tiny"):
                    raise ValueError("INT8 leaf MMA is implemented for expert layout")
                if self.leaf_layout == "expert_tiny":
                    raise ValueError("tiny expert specialization currently requires BF16")
                leaf_kwargs.update(
                    page_k_scales=cache.get("page_k_token_scales"),
                    page_v_scales=cache.get("page_v_token_scales"),
                    int8_pv_mma=self.prefill_int8_pv_mma,
                )
        if indexed:
            leaf_kwargs["page_indices"] = page_indices
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
        full_coarse = getattr(self, "_lod_prefill_full_coarse", None)
        if full_coarse is not None:
            del self._lod_prefill_full_coarse
            if include_local:
                raise AssertionError("streamed coarse attention requires split local")
            full_output, full_lse, route_logits, producer_stream = full_coarse
            if producer_stream is not None:
                torch.cuda.current_stream(q.device).wait_stream(producer_stream)
            return subtract_selected_coarse_from_full(
                route_logits,
                state_v,
                counts,
                top_slots,
                full_output,
                full_lse,
                state_len=state_len,
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
            )
        fused_prefill = getattr(self, "_lod_prefill_fused_coarse", None)
        if fused_prefill is not None:
            del self._lod_prefill_fused_coarse
            coarse_output, coarse_lse, fused_includes_local = fused_prefill
            if include_local != fused_includes_local:
                raise AssertionError("fused prefill local-branch mode drifted")
            expected_output_shape = (*q.shape[:-1], int(state_v.size(-1)))
            if tuple(coarse_output.shape) != expected_output_shape:
                raise AssertionError("fused prefill coarse output shape drifted")
            original_slots = getattr(
                self, "_lod_prefill_union_original_slots", None
            )
            if original_slots is not None:
                del self._lod_prefill_union_original_slots
                return remove_query_tile_union_from_coarse(
                    q,
                    state_k,
                    state_v,
                    counts,
                    original_slots,
                    top_slots,
                    coarse_output,
                    coarse_lse,
                    kv_group_size=self.num_key_value_groups,
                    query_tile=self.leaf_union_query_tile,
                    scale=self.scaling,
                )
            return coarse_output, coarse_lse
        route_logits = getattr(self, "_lod_prefill_route_logits", None)
        if route_logits is not None:
            del self._lod_prefill_route_logits
            logits_already_masked = bool(
                getattr(self, "_lod_prefill_route_logits_masked", False)
            )
            if hasattr(self, "_lod_prefill_route_logits_masked"):
                del self._lod_prefill_route_logits_masked
            if (
                not self.reuse_route_logits_for_coarse
                and self.prefill_route_mass_fraction is None
            ):
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
            coarse_top_slots = (
                torch.empty(
                    *top_slots.shape[:3],
                    0,
                    dtype=torch.long,
                    device=top_slots.device,
                )
                if logits_already_masked
                else top_slots
            )
            if int(q.size(-1)) > 512 or int(state_v.size(-1)) > 256:
                return self._gemm_coarse_attention(
                    q,
                    route_logits,
                    state_v,
                    counts,
                    coarse_local_k,
                    coarse_local_v,
                    coarse_top_slots,
                    state_len=state_len,
                )
            int8_state_pv = self.prefill_int8_coarse_mma and query_len > 1
            if self.prefill_coarse_direct_gqa and query_len > 1:
                self._lod_prefill_coarse_direct_gqa_executed = (
                    self.prefill_coarse_max_grouped_rows,
                    (
                        self.prefill_int8_coarse_block_n
                        if int8_state_pv
                        else self.prefill_coarse_route_block_n
                    ),
                    (
                        self.prefill_int8_coarse_num_warps
                        if int8_state_pv
                        else self.prefill_coarse_route_num_warps
                    ),
                )
            return route_logits_coarse_attention(
                q.contiguous(),
                route_logits.contiguous(),
                state_v.contiguous(),
                counts.contiguous(),
                coarse_local_k.contiguous(),
                coarse_local_v.contiguous(),
                coarse_top_slots.contiguous(),
                state_len=state_len,
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                block_m=self.coarse_route_block_m,
                block_n=(
                    self.prefill_int8_coarse_block_n
                    if int8_state_pv
                    else self.prefill_coarse_route_block_n
                ),
                num_warps=(
                    self.prefill_int8_coarse_num_warps
                    if int8_state_pv
                    else self.prefill_coarse_route_num_warps
                ),
                precompute_mean_values=query_len > 1,
                int8_state_pv=int8_state_pv,
                max_grouped_rows=self.prefill_coarse_max_grouped_rows,
                direct_gqa_rows=self.prefill_coarse_direct_gqa,
                timing_events=getattr(self, "_lod_phase_timing_events", None),
            )
        decode_route_logits = getattr(self, "_lod_decode_route_logits", None)
        if decode_route_logits is not None:
            del self._lod_decode_route_logits
            route_logits, query_rms = decode_route_logits
            if tuple(route_logits.shape) != (
                int(q.size(0)),
                int(q.size(1)),
                query_len,
                state_len,
            ):
                raise AssertionError("LOD decode route-logit shape drifted")
            if query_rms is not None:
                route_logits = route_logits.float() * query_rms
            if int(q.size(-1)) > 512 or int(state_v.size(-1)) > 256:
                return self._gemm_coarse_attention(
                    q,
                    route_logits,
                    state_v,
                    counts,
                    local_k if include_local else local_k[..., :0, :],
                    local_v if include_local else local_v[..., :0, :],
                    top_slots,
                    state_len=state_len,
                )
            int8_state_pv = self.prefill_int8_coarse_mma and query_len > 1
            return route_logits_coarse_attention(
                q.contiguous(),
                route_logits.contiguous(),
                state_v.contiguous(),
                counts.contiguous(),
                (local_k if include_local else local_k[..., :0, :]).contiguous(),
                (local_v if include_local else local_v[..., :0, :]).contiguous(),
                top_slots.contiguous(),
                state_len=state_len,
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                block_m=self.coarse_route_block_m,
                block_n=(
                    self.prefill_int8_coarse_block_n
                    if int8_state_pv
                    else self.coarse_route_block_n
                ),
                num_warps=(
                    self.prefill_int8_coarse_num_warps
                    if int8_state_pv
                    else self.coarse_route_num_warps
                ),
                precompute_mean_values=query_len > 1,
                int8_state_pv=int8_state_pv,
                max_grouped_rows=self.coarse_max_grouped_rows,
            )
        route_logits = self._state_route_logits(
            q,
            state_k,
            counts,
            state_len=state_len,
        )
        if int(q.size(-1)) > 512 or int(state_v.size(-1)) > 256:
            return self._gemm_coarse_attention(
                q,
                route_logits,
                state_v,
                counts,
                local_k if include_local else local_k[..., :0, :],
                local_v if include_local else local_v[..., :0, :],
                top_slots,
                state_len=state_len,
            )
        int8_state_pv = self.prefill_int8_coarse_mma and query_len > 1
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
            block_n=(
                self.prefill_int8_coarse_block_n
                if int8_state_pv
                else self.coarse_route_block_n
            ),
            num_warps=(
                self.prefill_int8_coarse_num_warps
                if int8_state_pv
                else self.coarse_route_num_warps
            ),
            precompute_mean_values=query_len > 1,
            int8_state_pv=int8_state_pv,
            max_grouped_rows=self.coarse_max_grouped_rows,
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

    def _gemm_coarse_attention(
        self,
        q: torch.Tensor,
        route_logits: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        top_slots: torch.Tensor,
        *,
        state_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact coarse/local branch using GEMMs for wide MLA values."""
        batch, query_heads, query_len, head_dim = q.shape
        kv_heads = int(state_v.size(1))
        groups = self.num_key_value_groups
        if query_heads != kv_heads * groups:
            raise ValueError("query and KV heads do not match for MLA GEMM")

        state_counts = counts.detach()[..., :state_len, 0]
        query_counts = state_counts.repeat_interleave(groups, dim=1)
        scores = route_logits.detach().float() * self.scaling
        scores = scores + query_counts.clamp_min(1).log().unsqueeze(2)
        scores.masked_fill_(query_counts.le(0).unsqueeze(2), float("-inf"))
        routed = torch.zeros_like(scores, dtype=torch.bool)
        valid_routes = top_slots.ge(0)
        routed.scatter_(
            -1,
            top_slots.clamp(min=0, max=max(state_len - 1, 0)),
            valid_routes,
        )
        scores.masked_fill_(routed, float("-inf"))

        local_len = int(local_k.size(2))
        if local_len:
            grouped_q = q.detach().reshape(batch, kv_heads, groups, query_len, head_dim)
            local_scores = torch.matmul(
                grouped_q,
                local_k.detach().transpose(-1, -2).unsqueeze(2),
            ).reshape(batch, query_heads, query_len, local_len)
            local_scores = local_scores.float() * self.scaling
            local_offset = local_len - query_len
            query_positions = local_offset + torch.arange(query_len, device=q.device)
            key_positions = torch.arange(local_len, device=q.device)
            visible = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            local_scores.masked_fill_(~visible, float("-inf"))
            scores = torch.cat((scores, local_scores), dim=-1)

        lse = torch.logsumexp(scores, dim=-1)
        probabilities = torch.softmax(scores, dim=-1).to(state_v.dtype)
        state_probabilities = probabilities[..., :state_len].reshape(
            batch, kv_heads, groups, query_len, state_len
        )
        mean_state_v = (
            state_v.detach()[..., :state_len, :].float()
            / state_counts.clamp_min(1).unsqueeze(-1)
        ).to(state_v.dtype)
        output = torch.matmul(state_probabilities, mean_state_v.unsqueeze(2))
        if local_len:
            local_probabilities = probabilities[..., state_len:].reshape(
                batch, kv_heads, groups, query_len, local_len
            )
            output = output + torch.matmul(
                local_probabilities, local_v.detach().unsqueeze(2)
            )
        return output.reshape(batch, query_heads, query_len, -1), lse

    def _record_decode_previous_total_lse(
        self,
        page_cache: dict[str, torch.Tensor | int] | None,
        q: torch.Tensor,
        *branch_lse: torch.Tensor,
        sink_k: torch.Tensor | None = None,
    ) -> None:
        """Seed retained-mass decode routing from the final prefill query."""
        if not self.decode_gqa_union_predicted_mass or page_cache is None:
            return
        if not branch_lse:
            raise ValueError("retained-mass decode requires an attention LSE")
        total_lse = branch_lse[0].float()
        for current_lse in branch_lse[1:]:
            total_lse = torch.logaddexp(total_lse, current_lse.float())
        if sink_k is not None:
            batch, query_heads, query_len, head_dim = q.shape
            kv_heads = int(sink_k.size(1))
            if query_heads != kv_heads * self.num_key_value_groups:
                raise ValueError("retained-mass sink GQA geometry is inconsistent")
            grouped_q = q.float().reshape(
                batch,
                kv_heads,
                self.num_key_value_groups,
                query_len,
                head_dim,
            )
            sink_scores = torch.matmul(
                grouped_q,
                sink_k.float().transpose(-1, -2).unsqueeze(2),
            )
            sink_lse = torch.logsumexp(sink_scores * self.scaling, dim=-1)
            sink_lse = sink_lse.reshape(batch, query_heads, query_len)
            total_lse = torch.logaddexp(total_lse, sink_lse)
        page_cache["decode_previous_total_lse"] = (
            total_lse[..., -1].contiguous().detach()
        )

    def _static_two_level_prefill(
        self,
        q: torch.Tensor,
        local_branch: tuple[torch.Tensor, torch.Tensor] | None,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        page_cache: dict[str, torch.Tensor | int] | None,
        *,
        state_len: int,
        context_len: int,
        sink_k: torch.Tensor | None,
        sink_v: torch.Tensor | None,
        output_buffer: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one 4K catch-up block with query-independent static routing."""
        if local_branch is None:
            raise ValueError("static prefill requires split local attention")
        if page_cache is None:
            raise ValueError("static prefill requires a paged leaf archive")
        required = (
            "leaf_k",
            "leaf_v",
            "page_indices",
            "slot_pages",
            "overflow_page_keys",
            "overflow_page_values",
            "overflow_used",
            "slot_lengths",
        )
        if not all(isinstance(page_cache.get(name), torch.Tensor) for name in required):
            raise ValueError("static prefill page archive is incomplete")
        minimum = int(self.prefill_static_leaf_cap_min)
        divisor = int(self.static_leaf_cap_divisor)
        if minimum < 1 or divisor < 1 or context_len < 1:
            raise ValueError("static prefill schedule requires positive lengths")
        cap = max(minimum, math.isqrt(context_len - 1) // divisor + 1)
        cohort_status = None
        if self.static_cohort_never_readmit:
            candidate = page_cache.get("static_cohort_status")
            slot_lengths = page_cache["slot_lengths"]
            if not isinstance(candidate, torch.Tensor):
                raise RuntimeError("monotone static cohort storage is missing")
            if tuple(candidate.shape) != tuple(slot_lengths.shape):
                raise ValueError("monotone static cohort storage has the wrong shape")
            active = slot_lengths.gt(0)
            candidate.masked_fill_(candidate.eq(0) & active, 1)
            candidate.masked_fill_(
                candidate.eq(1) & slot_lengths.gt(cap), -1
            )
            cohort_status = candidate
        static_buffers = getattr(self, "_lod_static_prefill_buffers", None)
        if not isinstance(static_buffers, dict):
            static_buffers = {}
            self._lod_static_prefill_buffers = static_buffers
        no_slots = torch.empty(
            *q.shape[:3], 0, dtype=torch.long, device=q.device
        )
        empty_local_k = page_cache["leaf_k"][..., :0, :].contiguous()
        empty_local_v = page_cache["leaf_v"][..., :0, :].contiguous()

        def static_coarse_branch() -> tuple[torch.Tensor, torch.Tensor]:
            route_logits = self._state_route_logits(
                q,
                state_k,
                counts,
                state_len=state_len,
            ).contiguous()
            query_counts = self._repeat_kv(
                counts.detach()[..., :state_len, 0]
            )
            if cohort_status is None:
                route_logits.masked_fill_(
                    query_counts.le(cap).unsqueeze(2),
                    float("-inf"),
                )
            else:
                query_status = self._repeat_kv(
                    cohort_status.detach()[..., :state_len]
                )
                route_logits.masked_fill_(
                    query_status.ne(-1).unsqueeze(2),
                    float("-inf"),
                )
            if self.prefill_coarse_direct_gqa:
                self._lod_prefill_coarse_direct_gqa_executed = (
                    self.prefill_coarse_max_grouped_rows,
                    self.prefill_coarse_route_block_n,
                    self.prefill_coarse_route_num_warps,
                )
            return route_logits_coarse_attention(
                q,
                route_logits,
                state_v.contiguous(),
                counts.contiguous(),
                empty_local_k,
                empty_local_v,
                no_slots,
                state_len=state_len,
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                block_m=self.coarse_route_block_m,
                block_n=self.prefill_coarse_route_block_n,
                num_warps=self.prefill_coarse_route_num_warps,
                precompute_mean_values=True,
                max_grouped_rows=self.prefill_coarse_max_grouped_rows,
                direct_gqa_rows=self.prefill_coarse_direct_gqa,
                timing_events=getattr(self, "_lod_phase_timing_events", None),
            )

        exact_output, exact_lse, exact_token_counts = (
            static_cap_aiter_paged_leaf_attention(
                q,
                page_cache["leaf_k"],
                page_cache["leaf_v"],
                page_cache["page_indices"],
                page_cache["slot_pages"],
                page_cache["overflow_page_keys"],
                page_cache["overflow_page_values"],
                page_cache["overflow_used"],
                page_cache["slot_lengths"],
                exact_slot_mask=(
                    cohort_status.eq(1) if cohort_status is not None else None
                ),
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                max_exact_leaves=cap,
                hash_probes=self._page_lookup_probes(page_cache),
                buffers=static_buffers,
                timing_events=getattr(self, "_lod_leaf_timing_events", None),
            )
        )
        if os.getenv("LOD_STATIC_COHORT_EVICTION_DIAGNOSTICS") == "1":
            status = page_cache.get("static_cohort_status")
            slot_lengths = page_cache["slot_lengths"]
            if not isinstance(status, torch.Tensor):
                raise RuntimeError("static cohort status storage is missing")
            if tuple(status.shape) != tuple(slot_lengths.shape):
                raise ValueError("static cohort status has the wrong shape")
            active = slot_lengths.gt(0)
            newly_seen = status.eq(0) & active
            status.masked_fill_(newly_seen, 1)
            status.masked_fill_(status.eq(1) & slot_lengths.gt(cap), -1)
            permanent_lengths = torch.where(
                status.eq(1), slot_lengths, torch.zeros_like(slot_lengths)
            )
            self._lod_prefill_static_permanent_exact_token_counts = (
                permanent_lengths.sum(dim=-1, dtype=torch.int32)
            )
            self._lod_prefill_static_archive_token_counts = slot_lengths.sum(
                dim=-1, dtype=torch.int32
            )
        coarse_output, coarse_lse = static_coarse_branch()

        local_stream = getattr(self, "_lod_prefill_local_stream_pending", None)
        if local_stream is not None:
            del self._lod_prefill_local_stream_pending
            torch.cuda.current_stream(q.device).wait_stream(local_stream)
        local_output, local_lse = local_branch
        self._lod_prefill_static_leaf_cap_executed = True
        self._lod_prefill_static_leaf_cap = cap
        self._lod_prefill_static_exact_token_counts = exact_token_counts
        history = getattr(self, "_lod_prefill_static_leaf_cap_history", None)
        if history is None:
            history = []
            self._lod_prefill_static_leaf_cap_history = history
        history.append(cap)
        self._record_decode_previous_total_lse(
            page_cache,
            q,
            coarse_lse,
            exact_lse,
            local_lse,
            sink_k=sink_k,
        )
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
                output_buffer=output_buffer,
            )
        return merge_attention_branches(
            coarse_output,
            coarse_lse,
            exact_output,
            exact_lse,
            local_output,
            local_lse,
            output_buffer=output_buffer,
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
        q = q.contiguous()
        # The persistent MLA key buffers intentionally hold raw compressed
        # latents in the experimental modes.  Normalize exact token keys at
        # consumption time; values are already the model-normalized latent.
        local_k = self._mla_normalize_key(local_k, state_centroid=False)
        if new_k is not None:
            new_k = self._mla_normalize_key(new_k, state_centroid=False)
        if sink_k is not None:
            sink_k = self._mla_normalize_key(sink_k, state_centroid=False)
        if self.leaf_attention_backend == "packed":
            exact_k = self._mla_normalize_key(exact_k, state_centroid=False)
        if output_buffer is not None and (
            tuple(output_buffer.shape) != tuple(q.shape)
            or output_buffer.dtype != q.dtype
            or output_buffer.device != q.device
            or int(output_buffer.stride(-1)) != 1
        ):
            raise ValueError("LOD output buffer has incompatible geometry")
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
        if self.prefill_static_leaf_aiter and int(q.size(2)) > 1:
            if dynamic_target is not None or dynamic_residual is not None:
                raise ValueError(
                    "static prefill cannot be combined with dynamic opening"
                )
            if context_len is None:
                raise ValueError("static prefill requires its logical context length")
            return self._static_two_level_prefill(
                q,
                local_branch,
                state_k,
                state_v,
                counts,
                page_cache,
                state_len=state_len,
                context_len=context_len,
                sink_k=sink_k,
                sink_v=sink_v,
                output_buffer=output_buffer,
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
            no_slots = torch.empty(*q.shape[:3], 0, dtype=torch.long, device=q.device)
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
                    output_buffer=output_buffer,
                )
            if output_buffer is None:
                return coarse_output
            output_buffer.copy_(coarse_output)
            return output_buffer
        if self.dense_page_prefill and self.recursive_page_lod and int(q.size(2)) > 1:
            if page_cache is None:
                raise RuntimeError("dense page prefill requires a page cache")
            if local_branch is None:
                raise RuntimeError("dense page prefill requires split local attention")
            if bool(page_cache.get("summary_quantization_finalized", False)):
                raise RuntimeError(
                    "dense page prefill currently requires BF16 page summaries"
                )
            required = (
                "leaf_k",
                "leaf_v",
                "page_indices",
                "page_sum_k",
                "page_sum_v",
                "page_counts",
                "next_page",
            )
            values = tuple(page_cache.get(name) for name in required)
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise RuntimeError("dense page prefill cache is incomplete")
            (
                page_residual,
                page_residual_lse,
                exact_pages,
                exact_pages_lse,
                _selected_pages,
            ) = dense_page_summary_attention(
                q,
                page_cache["leaf_k"],
                page_cache["leaf_v"],
                page_cache["page_indices"],
                page_cache["page_sum_k"],
                page_cache["page_sum_v"],
                page_cache["page_counts"],
                page_cache["next_page"],
                kv_group_size=self.num_key_value_groups,
                scale=self.scaling,
                top_pages=self.dense_page_topk,
                block_m=self.dense_page_block_m,
                block_n=self.dense_page_block_n,
                num_warps=self.dense_page_num_warps,
                waves_per_eu=self.leaf_waves_per_eu,
                split_kernels=self.dense_page_split_kernels,
                indexed_aiter_union=self.dense_page_indexed_aiter_union,
                union_query_tile=self.dense_page_union_query_tile,
                timing_events=getattr(self, "_lod_dense_page_timing_events", None),
            )
            local_stream = getattr(self, "_lod_prefill_local_stream_pending", None)
            if local_stream is not None:
                del self._lod_prefill_local_stream_pending
                torch.cuda.current_stream(q.device).wait_stream(local_stream)
            local_output, local_lse = local_branch
            self._record_decode_previous_total_lse(
                page_cache,
                q,
                page_residual_lse,
                exact_pages_lse,
                local_lse,
                sink_k=sink_k,
            )
            if sink_k is not None and sink_v is not None:
                return merge_attention_branches_with_sink(
                    q,
                    sink_k,
                    sink_v,
                    page_residual,
                    page_residual_lse,
                    exact_pages,
                    exact_pages_lse,
                    local_output,
                    local_lse,
                    kv_group_size=self.num_key_value_groups,
                    scale=self.scaling,
                    output_buffer=output_buffer,
                )
            return merge_attention_branches(
                page_residual,
                page_residual_lse,
                exact_pages,
                exact_pages_lse,
                local_output,
                local_lse,
                output_buffer=output_buffer,
            )
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
            and self.routing_rope_fast_pairs == 0
            and not self.routing_rope_jensen
            and self.routing_count_bias == 1.0
            and self.routing_variance_bias == 0.0
            and getattr(self, "mla_state_key_normalization", "none") == "none"
            and self.routing_page_mass_candidates == 0
            and self.routing_leaf_mass_candidates == 0
            and int(q.size(2)) == 1
            and self.decode_open_all_under_leaf_cap is None
            and self.leaf_attention_backend == "paged"
            and (
                indexed_recursive_decode
                or indexed_flat_decode
                or not (
                    page_cache is not None
                    and isinstance(page_cache.get("page_indices"), torch.Tensor)
                )
            )
            and self.two_level_topk <= 8
            and (
                self.decode_route_mass_fraction is None
                or self.decode_mass_top8_filter
            )
            and (not dynamic_active or not self.collect_dynamic_open_stats)
        )
        top_slots = None
        decode_mass_route_data = None
        if not fuse_decode_route:
            dynamic_local_lse = None
            if self.prefill_route_mass_fraction is not None and int(q.size(2)) > 1:
                if local_branch is None:
                    raise ValueError(
                        "mass-fraction prefill routing requires split local attention"
                    )
                if self.prefill_mass_include_local_lse:
                    dynamic_local_lse = local_branch[1]
            elif (
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
                local_len=local_len,
                new_k=new_k,
                page_cache=page_cache,
                dynamic_local_lse=dynamic_local_lse,
                context_len=context_len,
            )
            if (
                int(q.size(2)) == 1
                and self.decode_route_mass_fraction is not None
            ):
                decode_mass_route_data = getattr(
                    self, "_lod_decode_mass_route_data", None
                )
                if hasattr(self, "_lod_decode_mass_route_data"):
                    delattr(self, "_lod_decode_mass_route_data")
                if not (
                    isinstance(decode_mass_route_data, tuple)
                    and len(decode_mass_route_data) == 4
                ):
                    raise AssertionError("decode mass routing did not form coarse data")
            if getattr(self, "_lod_padding_state_reserve", 0):
                query_counts = self._repeat_kv(counts[..., :state_len, :]).squeeze(-1)
                safe_slots = top_slots.clamp_min(0)
                selected_counts = torch.gather(
                    query_counts.unsqueeze(2).expand(-1, -1, int(q.size(2)), -1),
                    -1,
                    safe_slots,
                )
                top_slots = torch.where(
                    top_slots.ge(0) & selected_counts.gt(0.5),
                    top_slots,
                    torch.full_like(top_slots, -1),
                )
        if self.leaf_layout == "aiter_union" and int(q.size(2)) > 1:
            if self.recursive_page_lod:
                raise ValueError("AITER centroid unions require flat two-tier LOD")
            if top_slots is None or page_cache is None or not isinstance(
                page_cache.get("page_indices"), torch.Tensor
            ):
                raise ValueError("AITER centroid unions require virtual page storage")
            slot_lengths = page_cache.get("slot_lengths")
            if not isinstance(slot_lengths, torch.Tensor):
                raise TypeError("AITER centroid unions require slot lengths")
            if hasattr(self, "_lod_prefill_fused_coarse"):
                self._lod_prefill_union_original_slots = top_slots
            top_slots = query_tile_slot_unions(
                top_slots,
                slot_lengths,
                kv_group_size=self.num_key_value_groups,
                query_tile=self.leaf_union_query_tile,
            )
        if (
            self.fused_decode_attention
            and int(state_v.size(-1)) == int(q.size(-1))
            and int(q.size(2)) == 1
            and self.leaf_attention_backend == "paged"
            and (
                indexed_recursive_decode
                or indexed_flat_decode
                or not (
                    page_cache is not None
                    and isinstance(page_cache.get("page_indices"), torch.Tensor)
                )
            )
            and (sink_k is None or fuse_decode_route)
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
            flat_int8_decode = not indexed_recursive_decode and (
                page_k.dtype == torch.int8 or page_v.dtype == torch.int8
            )
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
            if self.decode_split_kv > 1 and (
                decode_buffers is None
                or tuple(decode_buffers["partial_out"].shape) != expected_partial
                or decode_buffers["partial_out"].device != q.device
                or (
                    wide_local_required
                    and not (
                        (
                            "wide_gqa_local_scores" in decode_buffers
                            and int(
                                decode_buffers["wide_gqa_local_scores"].size(-1)
                            )
                            >= wide_local_len + 1
                        )
                        or (
                            indexed_recursive_decode
                            and self.recursive_materialize_page_scores
                            and "recursive_page_scores" in decode_buffers
                            and int(
                                decode_buffers["recursive_page_scores"].size(-1)
                            )
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
                route_post_dot_normalize=self.decode_route_post_dot_normalize,
                route_post_pv_normalize=self.decode_route_post_pv_normalize,
                final_reduce_num_warps=self.decode_final_reduce_num_warps,
                fuse_final_reduce=self.decode_fuse_final_reduce,
                route_use_dot=(
                    False
                    if self.decode_geometry_tuning
                    and (int(q.size(-1)) == 512 or long_d128_decode)
                    else self.decode_route_use_dot
                ),
                route_gqa_grouped=self.decode_route_gqa_grouped,
                gqa_cooperative_leaf=cooperative_leaf,
                gqa_cooperative_hip=self.decode_gqa_cooperative_hip,
                gqa_cooperative_route_splits=cooperative_route_splits,
                gqa_cooperative_adaptive_splits=cooperative_adaptive_splits,
                gqa_cooperative_fused_reduce=(
                    self.decode_gqa_cooperative_fused_reduce
                ),
                gqa_union_mass_fraction=self.decode_gqa_union_mass_fraction,
                protected_len=(
                    self._protected_route_len(state_len)
                    if self.exclude_sink_from_routes
                    else 0
                ),
                max_leaf_tokens=self.leaf_seal_capacity,
                open_count=int(self.two_level_topk),
                sink_k=sink_k,
                sink_v=sink_v,
                route_top_p=(dynamic_target if fuse_decode_route else None),
                route_residual_mass=(dynamic_residual if fuse_decode_route else None),
                route_mass_fraction=(
                    float(self.decode_route_mass_fraction)
                    if fuse_decode_route
                    and self.decode_mass_top8_filter
                    and self.decode_route_mass_fraction is not None
                    else None
                ),
                reuse_residual_local_attention=(
                    self.reuse_dynamic_local_attention and fuse_decode_route
                ),
                route_residual_use_state_bound=(
                    self.dynamic_open_residual_use_state_bound and fuse_decode_route
                ),
                timing_events=getattr(self, "_lod_decode_timing_events", None),
                recursive_page_cache=(page_cache if indexed_recursive_decode else None),
                flat_page_indices=(
                    page_cache["page_indices"] if indexed_flat_decode else None
                ),
                flat_page_k_scales=(
                    page_cache.get("page_k_token_scales") if flat_int8_decode else None
                ),
                flat_page_v_scales=(
                    page_cache.get("page_v_token_scales") if flat_int8_decode else None
                ),
                recursive_quant_group_size=self.leaf_quant_group_size,
                recursive_quant_token_group_size=(
                    self.leaf_quant_token_group_size
                ),
                recursive_materialize_page_scores=(
                    self.recursive_materialize_page_scores
                ),
                recursive_page_score_block_n=self.recursive_page_score_block_n,
                recursive_page_score_num_warps=self.recursive_page_score_num_warps,
                recursive_page_select_block_n=self.recursive_page_select_block_n,
                recursive_state_route_backend=(
                    self.recursive_state_route_backend
                ),
                precomputed_route_scores=(
                    decode_mass_route_data[0]
                    if decode_mass_route_data is not None
                    else None
                ),
                precomputed_coarse_out=(
                    decode_mass_route_data[1]
                    if decode_mass_route_data is not None
                    else None
                ),
                precomputed_coarse_lse=(
                    decode_mass_route_data[2]
                    if decode_mass_route_data is not None
                    else None
                ),
                precomputed_coarse_stream=(
                    decode_mass_route_data[3]
                    if decode_mass_route_data is not None
                    else None
                ),
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
        if self.leaf_attention_backend == "paged":
            if page_cache is None:
                raise RuntimeError("paged LOD attention has no leaf page cache")
            recursive_prefill_all_leaves = bool(
                self.recursive_page_lod
                and self.recursive_prefill_all_leaves
                and int(q.size(2)) > 1
            )
            if recursive_prefill_all_leaves:
                if bool(page_cache.get("quantization_finalized", False)):
                    raise NotImplementedError(
                        "recursive all-leaf prefill currently requires BF16 leaves"
                    )
                if bool(page_cache.get("mla_raw_page_key_summaries", False)):
                    raise NotImplementedError(
                        "recursive all-leaf prefill does not yet support raw MLA keys"
                    )
                exact_output, exact_lse = self._paged_leaf_attention(
                    q, top_slots, page_cache
                )
            elif self.recursive_page_lod:
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
                raw_page_key_summaries = bool(
                    page_cache.get("mla_raw_page_key_summaries", False)
                )
                materialized_page_scores = None
                if self.recursive_materialize_page_scores and int(q.size(2)) == 1:
                    if quantized_summaries:
                        raise ValueError(
                            "GQA page-score materialization requires BF16 summaries"
                        )
                    if raw_page_key_summaries:
                        raise ValueError(
                            "GQA page-score materialization does not yet support MLA"
                        )
                    score_shape = (
                        int(q.size(0)),
                        int(q.size(1)),
                        1,
                        int(page_cache["page_counts"].size(2)),
                    )
                    score_buffer = getattr(self, "_lod_page_score_buffer", None)
                    if (
                        not isinstance(score_buffer, torch.Tensor)
                        or tuple(score_buffer.shape) != score_shape
                        or score_buffer.device != q.device
                    ):
                        score_buffer = torch.empty(
                            score_shape, dtype=torch.float32, device=q.device
                        )
                        self._lod_page_score_buffer = score_buffer
                    materialized_page_scores = materialize_page_summary_scores_gqa(
                        q,
                        page_cache["page_sum_k"],
                        page_cache["page_counts"],
                        kv_group_size=self.num_key_value_groups,
                        scale=self.scaling,
                        page_block_n=self.recursive_page_score_block_n,
                        num_warps=self.recursive_page_score_num_warps,
                        waves_per_eu=self.leaf_waves_per_eu,
                        output=score_buffer,
                    )
                recursive_state_k = (
                    state_k
                    if raw_page_key_summaries
                    else self._mla_state_key_sum_for_attention(
                        state_k,
                        counts,
                        state_len=state_len,
                    )
                )
                exact_output, exact_lse = residual_page_function(
                    q,
                    recursive_state_k,
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
                    hash_probes=self._page_lookup_probes(page_cache),
                    page_block_n=(
                        self.recursive_page_select_block_n
                        if materialized_page_scores is not None
                        else self.recursive_page_block_n
                    ),
                    num_warps=self.leaf_num_warps,
                    waves_per_eu=self.leaf_waves_per_eu,
                    timing_events=getattr(self, "_lod_leaf_timing_events", None),
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
                    quant_token_group_size=self.leaf_quant_token_group_size,
                    quant_bits=int(page_cache.get("leaf_quant_bits", 4)),
                    mla_norm_weight=(
                        self.mla_key_norm_weight if raw_page_key_summaries else None
                    ),
                    mla_norm_epsilon=(
                        self.mla_key_norm_epsilon if raw_page_key_summaries else 0.0
                    ),
                    materialized_page_scores=materialized_page_scores,
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
            self._record_decode_previous_total_lse(
                page_cache,
                q,
                coarse_lse,
                exact_lse,
                local_lse,
                sink_k=sink_k,
            )
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
                    output_buffer=output_buffer,
                )
            return merge_attention_branches(
                coarse_output,
                coarse_lse,
                exact_output,
                exact_lse,
                local_output,
                local_lse,
                output_buffer=output_buffer,
            )
        self._record_decode_previous_total_lse(
            page_cache,
            q,
            coarse_lse,
            exact_lse,
            sink_k=sink_k,
        )
        output = merge_attention_branches(
            coarse_output,
            coarse_lse,
            exact_output,
            exact_lse,
            output_buffer=output_buffer,
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
                        "or its suffix queries"
                    )
                dense_q = actual_q.permute(0, 2, 1, 3).contiguous()
                dense_k = k.permute(0, 2, 1, 3).contiguous()
                dense_v = v.permute(0, 2, 1, 3).contiguous()
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
        front_len = min(attention_len, exact_lookback + self.chunk_len)
        outputs = []
        if not build_cache_only:
            if q is None:
                raise AssertionError("attention prefill query is missing")
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
        counts[..., :initial_state_len, :].copy_(initial_counts)
        key_norm_sums = None
        if self.state_clustering_centroid_rescale != "none":
            key_norm_sums = torch.zeros_like(counts)
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
            )
            owners = None

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
                        local_stream = getattr(
                            self, "_lod_prefill_local_stream", None
                        )
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
                    append_owner_parts.append(new_owners)
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
                append_owner_parts.append(new_owners)
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
                self.prefill_local_len if not finalize_cache_for_decode else 0,
            )
            buffered_k = k.new_empty(*k.shape[:2], recent_capacity, int(k.size(-1)))
            buffered_v = v.new_empty(*v.shape[:2], recent_capacity, int(v.size(-1)))
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
                # All archived leaves now live in quantized flat tensors. Keep
                # only typed pointer sentinels for the compile-time BF16 fallback.
                page_cache["leaf_k"] = k.new_empty(*k.shape[:2], 1, int(k.size(-1)))
                page_cache["leaf_v"] = v.new_empty(*v.shape[:2], 1, int(v.size(-1)))
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
            nonlocal state_coverage, owners
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
        front_len = exact_lookback + self.chunk_len
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
            exact_q = (
                suffix_query
                if self.prefill_local_attention_backend == "aiter"
                else torch.cat(
                    (
                        q.new_zeros(*q.shape[:2], previous_total_len, int(q.size(-1))),
                        suffix_query,
                    ),
                    dim=2,
                )
            )
            exact_output, _ = self._prefill_local_attention(
                exact_q,
                exact_k,
                exact_v,
                query_offset=previous_total_len,
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

        if finalize_cache_for_decode:
            decode_coverage = max(
                state_coverage,
                self._bswa_begin(total_len + 1),
            )
        else:
            completed_blocks = max(
                0,
                (total_len - front_len) // prefill_chunk_len,
            )
            next_query_begin = front_len + completed_blocks * prefill_chunk_len
            decode_coverage = max(
                state_coverage,
                next_query_begin - exact_lookback,
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
                self.prefill_local_len if not finalize_cache_for_decode else 0,
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
