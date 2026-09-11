"""Production top-four paged LoD decode orchestration."""

from __future__ import annotations

import math

import torch
import triton

from .aiter_page1_attention import (
    kernel_exact_residual_int4_attention_3d,
    kernel_exact_tiered_attention_3d,
)
from .paged_decode_buffers import (
    _append_decode_gqa_union_arena_entries_kernel,
    _decode_topk_gqa_union_kernel,
    _expand_decode_topk_gqa_union_kernel,
    advance_decode_cache_lengths,
)
from .paged_decode_kernels import (
    _apply_aiter_fixed_direct_routes_kernel,
    _prepare_aiter_fixed_mask_context_kernel,
    _reduce_aiter_page1_segments_with_lse_kernel,
    _reduce_aiter_page1_segments_with_lse_split_d_kernel,
    _reduce_routed_split_decode_lod_attention_kernel,
    _reset_aiter_fixed_previous_union_kernel,
    _split_decode_paged_lod_attention_kernel,
    _wide_gqa_local_scores_kernel,
    _wide_gqa_local_value_kernel,
)
from .paged_prefill import query_major_indexed_residual_page_attention
from .paged_routing import (
    _decode_route_coarse_gqa_groups_fixed_prepare_kernel,
    _decode_route_coarse_gqa_groups_kernel,
    _decode_route_coarse_gqa_mtp2_groups_kernel,
    _reduce_decode_route_coarse_kernel,
    _reduce_decode_route_coarse_vector_topk_kernel,
    _reduce_decode_route_topk_kernel,
    materialized_state_route_gqa,
)


def fused_decode_paged_lod_attention(
    q: torch.Tensor,
    state_k: torch.Tensor,
    state_v: torch.Tensor,
    counts: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    page_k: torch.Tensor,
    page_v: torch.Tensor,
    slot_pages: torch.Tensor,
    overflow_page_keys: torch.Tensor,
    overflow_page_values: torch.Tensor,
    overflow_used: torch.Tensor,
    slot_lengths: torch.Tensor,
    top_slots: torch.Tensor | None,
    *,
    sink_k: torch.Tensor | None = None,
    sink_v: torch.Tensor | None = None,
    state_len: int,
    state_lens: torch.Tensor | None = None,
    local_len: int | None = None,
    cache_indices: torch.Tensor | None = None,
    local_lens: torch.Tensor | None = None,
    new_k: torch.Tensor | None = None,
    new_v: torch.Tensor | None = None,
    local_lens_are_logical: bool = False,
    store_new_kv: bool = True,
    advance_local_lens: bool = True,
    speculative_steps: int = 1,
    kv_group_size: int,
    scale: float,
    hash_probes: int = 8,
    block_n: int = 16,
    num_warps: int = 2,
    waves_per_eu: int = 1,
    split_kv: int = 1,
    buffers: dict[str, torch.Tensor] | None = None,
    use_dot: bool = False,
    fuse_state_route: bool = False,
    route_group_size: int = 64,
    route_segment_tiles: int = 1,
    route_num_warps: int = 4,
    route_reduce_num_warps: int = 4,
    route_parallel_reduce: bool = False,
    route_parallel_reduce_block_d: int = 0,
    compact_top4_candidates: bool = True,
    fuse_route_local: bool = True,
    final_reduce_num_warps: int = 4,
    fuse_final_reduce: bool = False,
    route_gqa_grouped: bool = False,
    gqa_cooperative_leaf: bool = True,
    gqa_cooperative_hip: bool = False,
    gqa_union_decode: bool = False,
    gqa_union_unified: bool = True,
    gqa_union_mass_fraction: float | None = None,
    gqa_union_predicted_mass: bool = False,
    gqa_union_pilot_z: bool = False,
    gqa_union_hip: bool = False,
    gqa_union_group64_padded: bool = False,
    gqa_union_staged_fixed_aiter: bool = False,
    gqa_union_fixed_mask_aiter: bool = False,
    gqa_union_overlap_local_sink: bool = False,
    gqa_union_fixed_mask_tile_size: int = 64,
    gqa_union_fixed_mask_adaptive_segments: bool = False,
    gqa_union_fixed_mask_reduce_block_d: int = 0,
    gqa_union_fixed_mask_direct_routes: bool = True,
    gqa_union_fixed_mask_reuse_coarse: bool = False,
    gqa_union_fixed_mask_scan_num_warps: int = 2,
    gqa_union_fixed_mask_scan_waves_per_eu: int = 2,
    gqa_union_fixed_mask_scan_num_stages: int = 2,
    gqa_union_page1_k: torch.Tensor | None = None,
    gqa_union_page1_v: torch.Tensor | None = None,
    gqa_union_page1_bias: torch.Tensor | None = None,
    gqa_union_page1_leaf_offset: int = 0,
    gqa_union_page1_local_offset: int = 0,
    gqa_union_page1_sink_offset: int = 0,
    gqa_union_page1_coarse_offset: int = 0,
    gqa_union_fixed_indices: torch.Tensor | None = None,
    gqa_union_fixed_leaf_owners: torch.Tensor | None = None,
    gqa_union_fixed_slot_offsets: torch.Tensor | None = None,
    gqa_union_fixed_lengths: torch.Tensor | None = None,
    gqa_union_previous_total_lse: torch.Tensor | None = None,
    gqa_union_pilot_z_bound: torch.Tensor | None = None,
    protected_len: int = 0,
    max_leaf_tokens: int | None = None,
    open_count: int = 4,
    route_top_p: float | None = None,
    route_residual_mass: float | None = None,
    route_mass_fraction: float | None = None,
    reuse_residual_local_attention: bool = False,
    timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]
    | None = None,
    recursive_page_cache: dict[str, torch.Tensor | int] | None = None,
    flat_page_indices: torch.Tensor | None = None,
    flat_page_k_scales: torch.Tensor | None = None,
    flat_page_v_scales: torch.Tensor | None = None,
    recursive_quant_group_size: int = 32,
    recursive_quant_token_group_size: int = 16,
    recursive_page_select_block_n: int = 64,
    recursive_state_route_backend: str = "fused",
    exact_decode_threshold: int = 0,
    exact_all_rows: bool = False,
    exact_leaf_lens: torch.Tensor | None = None,
    output_buffer: torch.Tensor | None = None,
    precomputed_route_scores: torch.Tensor | None = None,
    precomputed_coarse_out: torch.Tensor | None = None,
    precomputed_coarse_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse coarse, exact-leaf, local, and branch-merge decode attention."""
    batch, query_heads, query_len, head_dim = q.shape
    kv_heads = int(state_k.size(1))
    precomputed_route_values = (
        precomputed_route_scores,
        precomputed_coarse_out,
        precomputed_coarse_lse,
    )
    precomputed_state_route = all(
        (isinstance(value, torch.Tensor) for value in precomputed_route_values)
    )
    execute_state_route = fuse_state_route
    fuse_state_route = bool(fuse_state_route or precomputed_state_route)
    route_fused_mtp_local = False
    route_fused_decode_local = False
    page_shape = (
        recursive_page_cache.get("page_indices")
        if recursive_page_cache is not None
        else flat_page_indices
        if flat_page_indices is not None
        else page_k
    )
    flat_int8 = page_k.dtype == torch.int8 or page_v.dtype == torch.int8
    if local_len is None:
        local_len = int(local_k.size(2))
    if cache_indices is None:
        if buffers is not None and "cache_indices" in buffers:
            cache_indices = buffers["cache_indices"][:batch]
    use_state_lens = state_lens is not None
    ragged_local_lens = local_lens is not None
    if local_lens is None:
        if (
            buffers is not None
            and "local_lens" in buffers
            and (int(buffers["local_lens"].numel()) >= int(state_k.size(0)))
        ):
            local_lens = buffers["local_lens"][: int(state_k.size(0))]
        local_lens.fill_(local_len)
    if state_lens is None:
        state_lens = local_lens
    include_new = new_k is not None or new_v is not None
    include_sink = sink_k is not None or sink_v is not None
    gqa_union_page1_arena = any(
        (
            tensor is not None
            for tensor in (gqa_union_page1_k, gqa_union_page1_v, gqa_union_page1_bias)
        )
    )

    def timing_begin() -> torch.cuda.Event | None:
        if timing_events is None:
            return None

    def timing_end(name: str, begin: torch.cuda.Event | None) -> None:
        if begin is None or timing_events is None:
            return

    if include_sink:
        fuse_final_reduce = False
    if not include_new:
        new_k = state_k[..., :1, :]
        new_v = state_v[..., :1, :]
    if not split_kv == 1:
        output = buffers["output"] if output_buffer is None else output_buffer
        partial_out = buffers["partial_out"]
        partial_lse = buffers["partial_lse"]

        def exact_scratch(
            sequence_count: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]:
            """Reuse the highly split GQA scan arena for exact short decode."""
            exact_out = buffers.get("exact_segment_out")
            exact_max = buffers.get("exact_segment_max")
            exact_exp_sum = buffers.get("exact_segment_exp_sum")
            exact_context_lens = buffers.get("exact_context_lens")
            if all(
                isinstance(value, torch.Tensor)
                for value in (
                    exact_out,
                    exact_max,
                    exact_exp_sum,
                    exact_context_lens,
                )
            ):
                segments = int(exact_out.size(2))
                return (
                    exact_out[:sequence_count],
                    exact_max[:sequence_count],
                    exact_exp_sum[:sequence_count],
                    exact_context_lens[:sequence_count],
                    segments,
                )
            segment_out = buffers.get("gqa_union_hip_segment_out")
            segment_max = buffers.get("gqa_union_hip_max_logits")
            segment_exp_sum = buffers.get("gqa_union_hip_exp_sums")
            context_lens = buffers.get("gqa_union_hip_context_lens")
            if all(
                isinstance(value, torch.Tensor)
                for value in (
                    segment_out,
                    segment_max,
                    segment_exp_sum,
                    context_lens,
                )
            ):
                segments = int(segment_out.size(2))
                return (
                    segment_out[:sequence_count],
                    segment_max[:sequence_count],
                    segment_exp_sum[:sequence_count],
                    context_lens[:sequence_count],
                    segments,
                )
            exact_context_lens = buffers.get("exact_context_lens")
            exact_exp_sums = buffers.get("exact_exp_sums")
            if not isinstance(
                exact_context_lens, torch.Tensor
            ) or not isinstance(exact_exp_sums, torch.Tensor):
                raise ValueError(
                    "short-context exact decode requires fixed scratch buffers"
                )
            return (
                partial_out.reshape(
                    sequence_count, kv_group_size, split_kv, head_dim
                ),
                partial_lse.reshape(sequence_count, kv_group_size, split_kv),
                exact_exp_sums.reshape(
                    sequence_count, kv_group_size, split_kv
                ),
                exact_context_lens[:sequence_count],
                split_kv,
            )

        def apply_exact_bf16_override(
            exact_leaf_k: torch.Tensor,
            exact_leaf_v: torch.Tensor,
            *,
            early_execution: bool = False,
        ) -> None:
            """Replace short routed rows with exact chronological-leaf attention."""
            if exact_decode_threshold <= 0:
                return
            if exact_leaf_lens is None or query_len != 1:
                raise ValueError(
                    "short-context exact decode requires one-token queries "
                    "and BF16 leaf lengths"
                )
            sequence_count = batch * kv_heads
            exact_query = q[:, :, 0, :].reshape(
                sequence_count, kv_group_size, head_dim
            )
            exact_output = output[:, :, 0, :].reshape(
                sequence_count, kv_group_size, head_dim
            )
            (
                exact_segment_out,
                exact_segment_max,
                exact_segment_exp_sum,
                exact_context_lens,
                exact_segments,
            ) = exact_scratch(sequence_count)
            sink_key_storage = sink_k if include_sink else state_k
            sink_value_storage = sink_v if include_sink else state_v
            exact_tile_size = 64
            # Ordinary decode has already persisted the current KV and advanced
            # ``local_lens`` before this override. Parallel speculative decode
            # deliberately leaves that counter unchanged, so its current
            # proposal must instead be included from ``new_k/new_v``.
            exact_include_new = bool(
                include_new and (early_execution or not advance_local_lens)
            )
            kernel_exact_tiered_attention_3d[
                sequence_count, 1, exact_segments
            ](
                exact_segment_out,
                exact_segment_max,
                exact_segment_exp_sum,
                exact_context_lens,
                exact_query,
                sink_key_storage,
                sink_value_storage,
                exact_leaf_k,
                exact_leaf_v,
                local_k,
                local_v,
                new_k,
                new_v,
                cache_indices,
                exact_leaf_lens,
                local_lens,
                float(scale),
                exact_query.stride(0),
                exact_query.stride(1),
                sink_key_storage.stride(0),
                sink_key_storage.stride(1),
                sink_key_storage.stride(2),
                sink_value_storage.stride(0),
                sink_value_storage.stride(1),
                sink_value_storage.stride(2),
                exact_leaf_k.stride(0),
                exact_leaf_k.stride(1),
                exact_leaf_k.stride(2),
                exact_leaf_v.stride(0),
                exact_leaf_v.stride(1),
                exact_leaf_v.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                NUM_QUERY_HEADS=kv_group_size,
                KV_HEADS=kv_heads,
                TILE_SIZE=exact_tile_size,
                HEAD_SIZE=head_dim,
                BLOCK_M=16,
                NUM_SEGMENTS=exact_segments,
                SINK_LEN=(int(sink_key_storage.size(2)) if include_sink else 0),
                LOCAL_LIMIT=local_len,
                INCLUDE_NEW=exact_include_new,
                STORE_NEW=bool(early_execution and include_new and store_new_kv),
                LOCAL_LENS_LOGICAL=local_lens_are_logical,
                MAX_CONTEXT=int(exact_decode_threshold),
                num_warps=2,
                waves_per_eu=2,
                num_stages=1,
            )
            if exact_segments >= 128:
                block_d = 64
                _reduce_aiter_page1_segments_with_lse_split_d_kernel[
                    sequence_count,
                    kv_group_size,
                    triton.cdiv(head_dim, block_d),
                ](
                    exact_segment_out,
                    exact_segment_max,
                    exact_segment_exp_sum,
                    exact_context_lens,
                    exact_output,
                    buffers["coarse_lse"],
                    exact_context_lens,
                    exact_context_lens,
                    exact_context_lens,
                    OUTPUT_STRIDE_0=exact_output.stride(0),
                    OUTPUT_STRIDE_1=exact_output.stride(1),
                    QUERY_ROWS=kv_group_size,
                    HEAD_DIM=head_dim,
                    SEGMENTS=exact_segments,
                    TILE_SIZE=exact_tile_size,
                    BLOCK_D=block_d,
                    ADVANCE_QUEUE=False,
                    SKIP_ZERO_LENGTH=True,
                    num_warps=2,
                    waves_per_eu=2,
                )
            else:
                _reduce_aiter_page1_segments_with_lse_kernel[
                    sequence_count, kv_group_size
                ](
                    exact_segment_out,
                    exact_segment_max,
                    exact_segment_exp_sum,
                    exact_context_lens,
                    exact_output,
                    buffers["coarse_lse"],
                    exact_context_lens,
                    exact_context_lens,
                    exact_context_lens,
                    cache_indices,
                    local_lens,
                    OUTPUT_STRIDE_0=exact_output.stride(0),
                    OUTPUT_STRIDE_1=exact_output.stride(1),
                    QUERY_ROWS=kv_group_size,
                    HEAD_DIM=head_dim,
                    SEGMENTS=exact_segments,
                    TILE_SIZE=exact_tile_size,
                    ADVANCE_QUEUE=False,
                    ADVANCE_LOCAL=False,
                    KV_HEADS=kv_heads,
                    SKIP_ZERO_LENGTH=True,
                    num_warps=2,
                    waves_per_eu=2,
                )
            if (
                early_execution
                and include_new
                and ragged_local_lens
                and advance_local_lens
            ):
                advance_decode_cache_lengths(cache_indices, local_lens)

        def apply_exact_int4_override(
            cache: dict[str, object], *, early_execution: bool = False
        ) -> None:
            """Replace short routed rows with an all-page residual-INT4 scan."""
            if exact_decode_threshold <= 0:
                return
            if exact_leaf_lens is None or query_len != 1:
                raise ValueError(
                    "short-context INT4 decode requires one-token queries "
                    "and leaf lengths"
                )

            def tensor(name: str) -> torch.Tensor:
                value = cache.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"short-context INT4 decode cache is missing {name}"
                    )
                return value

            exact_page_indices = tensor("page_indices")
            exact_query = q[:, :, 0, :].reshape(
                batch * kv_heads, kv_group_size, head_dim
            )
            exact_output = output[:, :, 0, :].reshape(
                batch * kv_heads, kv_group_size, head_dim
            )
            sequence_count = batch * kv_heads
            (
                exact_segment_out,
                exact_segment_max,
                exact_segment_exp_sum,
                exact_context_lens,
                exact_segments,
            ) = exact_scratch(sequence_count)
            sink_key_storage = sink_k if include_sink else state_k
            sink_value_storage = sink_v if include_sink else state_v
            exact_tile_size = 16
            exact_include_new = bool(
                include_new and (early_execution or not advance_local_lens)
            )
            kernel_exact_residual_int4_attention_3d[
                sequence_count, 1, exact_segments
            ](
                exact_segment_out,
                exact_segment_max,
                exact_segment_exp_sum,
                exact_context_lens,
                exact_query,
                sink_key_storage,
                sink_value_storage,
                tensor("quantized_leaf_k"),
                tensor("quantized_leaf_v"),
                exact_page_indices,
                tensor("page_counts"),
                tensor("next_page"),
                tensor("page_k_scales"),
                tensor("page_v_scales"),
                tensor("quantized_page_sum_k"),
                tensor("quantized_page_sum_v"),
                tensor("page_sum_k_scales"),
                tensor("page_sum_v_scales"),
                local_k,
                local_v,
                new_k,
                new_v,
                cache_indices,
                exact_leaf_lens,
                local_lens,
                float(scale),
                exact_query.stride(0),
                exact_query.stride(1),
                sink_key_storage.stride(0),
                sink_key_storage.stride(1),
                sink_key_storage.stride(2),
                sink_value_storage.stride(0),
                sink_value_storage.stride(1),
                sink_value_storage.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                NUM_QUERY_HEADS=kv_group_size,
                KV_HEADS=kv_heads,
                TILE_SIZE=exact_tile_size,
                HEAD_SIZE=head_dim,
                BLOCK_M=8,
                NUM_SEGMENTS=exact_segments,
                PAGE_CAPACITY=int(exact_page_indices.size(2)),
                LEAF_CAPACITY=int(tensor("quantized_leaf_k").size(2)),
                PAGE_SIZE=int(exact_page_indices.size(3)),
                QUANT_GROUP_SIZE=recursive_quant_group_size,
                SINK_LEN=(int(sink_key_storage.size(2)) if include_sink else 0),
                LOCAL_LIMIT=local_len,
                INCLUDE_NEW=exact_include_new,
                STORE_NEW=bool(early_execution and include_new and store_new_kv),
                LOCAL_LENS_LOGICAL=local_lens_are_logical,
                MAX_CONTEXT=int(exact_decode_threshold),
                num_warps=2,
                waves_per_eu=2,
                num_stages=1,
            )
            if exact_segments >= 128:
                block_d = 64
                _reduce_aiter_page1_segments_with_lse_split_d_kernel[
                    sequence_count,
                    kv_group_size,
                    triton.cdiv(head_dim, block_d),
                ](
                    exact_segment_out,
                    exact_segment_max,
                    exact_segment_exp_sum,
                    exact_context_lens,
                    exact_output,
                    buffers["coarse_lse"],
                    exact_context_lens,
                    exact_context_lens,
                    exact_context_lens,
                    OUTPUT_STRIDE_0=exact_output.stride(0),
                    OUTPUT_STRIDE_1=exact_output.stride(1),
                    QUERY_ROWS=kv_group_size,
                    HEAD_DIM=head_dim,
                    SEGMENTS=exact_segments,
                    TILE_SIZE=exact_tile_size,
                    BLOCK_D=block_d,
                    ADVANCE_QUEUE=False,
                    SKIP_ZERO_LENGTH=True,
                    num_warps=2,
                    waves_per_eu=2,
                )
            else:
                _reduce_aiter_page1_segments_with_lse_kernel[
                    sequence_count, kv_group_size
                ](
                    exact_segment_out,
                    exact_segment_max,
                    exact_segment_exp_sum,
                    exact_context_lens,
                    exact_output,
                    buffers["coarse_lse"],
                    exact_context_lens,
                    exact_context_lens,
                    exact_context_lens,
                    cache_indices,
                    local_lens,
                    OUTPUT_STRIDE_0=exact_output.stride(0),
                    OUTPUT_STRIDE_1=exact_output.stride(1),
                    QUERY_ROWS=kv_group_size,
                    HEAD_DIM=head_dim,
                    SEGMENTS=exact_segments,
                    TILE_SIZE=exact_tile_size,
                    ADVANCE_QUEUE=False,
                    ADVANCE_LOCAL=False,
                    KV_HEADS=kv_heads,
                    SKIP_ZERO_LENGTH=True,
                    num_warps=2,
                    waves_per_eu=2,
                )
            if (
                early_execution
                and include_new
                and ragged_local_lens
                and advance_local_lens
            ):
                advance_decode_cache_lengths(cache_indices, local_lens)

        def apply_exact_flat_bf16_override(
            *, early_execution: bool = False
        ) -> None:
            if exact_decode_threshold <= 0:
                return
            if flat_page_indices is None or flat_int8:
                raise ValueError(
                    "short-context exact decode requires an indexed BF16 leaf cache"
                )
            apply_exact_bf16_override(
                page_k, page_v, early_execution=early_execution
            )

        # Uniform short decode batches need no routing or coarse approximation.
        # Execute the exact cache scan directly rather than calculating routed
        # LoD first and then overwriting it with the exact result.
        if exact_all_rows:
            if exact_decode_threshold <= 0:
                raise ValueError("exact_all_rows requires exact decode")
            if recursive_page_cache is not None:
                if bool(recursive_page_cache.get("quantization_finalized", False)):
                    apply_exact_int4_override(
                        recursive_page_cache, early_execution=True
                    )
                else:
                    exact_leaf_k = recursive_page_cache.get("leaf_k")
                    exact_leaf_v = recursive_page_cache.get("leaf_v")
                    if not isinstance(exact_leaf_k, torch.Tensor) or not isinstance(
                        exact_leaf_v, torch.Tensor
                    ):
                        raise ValueError(
                            "short-context exact decode cache is missing archived leaves"
                        )
                    apply_exact_bf16_override(
                        exact_leaf_k, exact_leaf_v, early_execution=True
                    )
            else:
                apply_exact_flat_bf16_override(early_execution=True)
            return output

        recursive_total_begin = (
            timing_begin() if recursive_page_cache is not None else None
        )
        gqa_union_predicted = bool(
            execute_state_route
            and gqa_union_decode
            and gqa_union_predicted_mass
            and (gqa_union_mass_fraction is not None)
            and gqa_union_unified
            and gqa_union_hip
            and gqa_union_page1_arena
            and (not fuse_final_reduce)
            and (recursive_page_cache is None)
            and (flat_page_indices is not None)
            and (not flat_int8)
            and (q.dtype == torch.bfloat16)
            and (1 < kv_group_size <= 16)
            and (head_dim in (128, 256))
            and isinstance(gqa_union_previous_total_lse, torch.Tensor)
            and (buffers is not None)
            and all(
                (
                    name in buffers
                    for name in (
                        "gqa_union_seen_stamps",
                        "gqa_union_epochs",
                        "gqa_union_counts",
                        "gqa_union_token_counts",
                        "gqa_union_slots",
                    )
                )
            )
        )
        gqa_union_pilot = bool(
            execute_state_route
            and gqa_union_decode
            and gqa_union_pilot_z
            and gqa_union_unified
            and gqa_union_hip
            and gqa_union_page1_arena
            and (not fuse_final_reduce)
            and (recursive_page_cache is None)
            and (flat_page_indices is not None)
            and (not flat_int8)
            and (q.dtype == torch.bfloat16)
            and (1 < kv_group_size <= 16)
            and (head_dim in (128, 256))
            and isinstance(gqa_union_pilot_z_bound, torch.Tensor)
            and (buffers is not None)
            and ("route_pilot_z_thresholds" in buffers)
            and all(
                (
                    name in buffers
                    for name in (
                        "gqa_union_seen_stamps",
                        "gqa_union_epochs",
                        "gqa_union_counts",
                        "gqa_union_token_counts",
                        "gqa_union_slots",
                    )
                )
            )
        )
        gqa_union_mass = bool(
            execute_state_route
            and gqa_union_decode
            and (gqa_union_mass_fraction is not None)
            and (not gqa_union_predicted)
            and (not gqa_union_pilot)
            and gqa_union_unified
            and (not fuse_final_reduce)
            and (recursive_page_cache is None)
            and (flat_page_indices is not None)
            and (not flat_int8)
            and (q.dtype == torch.bfloat16)
            and (kv_group_size == 16)
            and (head_dim == 128)
            and (
                triton.cdiv(state_len, 256)
                <= int(buffers.get("route_partition_lse", q).size(-1))
            )
            and (buffers is not None)
            and all(
                (
                    name in buffers
                    for name in (
                        "route_state_scores",
                        "route_partition_lse",
                        "route_full_lse",
                        "gqa_union_seen_stamps",
                        "gqa_union_epochs",
                        "gqa_union_counts",
                        "gqa_union_token_counts",
                        "gqa_union_slots",
                    )
                )
            )
        )
        gqa_union_score_only = bool(
            execute_state_route
            and gqa_union_decode
            and gqa_union_unified
            and (not fuse_final_reduce)
            and (recursive_page_cache is None)
            and (flat_page_indices is not None)
            and (not flat_int8)
            and (q.dtype == torch.bfloat16)
            and (1 < kv_group_size <= 16)
            and (head_dim in (128, 256, 512))
            and route_gqa_grouped
            and (route_segment_tiles > 1 or gqa_union_page1_arena)
            and (route_mass_fraction is None or gqa_union_mass)
            and (route_top_p is None)
            and (route_residual_mass is None)
            and (not gqa_union_predicted)
            and (not gqa_union_pilot)
            and (buffers is not None)
            and ("gqa_union_seen_stamps" in buffers)
        )
        gqa_union_fixed_reuse_coarse = bool(
            gqa_union_fixed_mask_reuse_coarse and gqa_union_score_only
        )
        gqa_union_compact_reuse_coarse = False
        gqa_union_staged_fixed = bool(
            gqa_union_staged_fixed_aiter
            and gqa_union_score_only
            and gqa_union_hip
            and gqa_union_page1_arena
            and all(
                (
                    name in buffers
                    for name in (
                        "gqa_union_token_indices",
                        "gqa_union_hip_context_lens",
                        "gqa_union_hip_launch_lens",
                        "gqa_union_hip_segment_out",
                        "gqa_union_hip_exp_sums",
                        "gqa_union_hip_max_logits",
                    )
                )
            )
        )
        gqa_union_fixed_mask = bool(
            gqa_union_fixed_mask_aiter
            and (
                gqa_union_score_only
                or gqa_union_fixed_reuse_coarse
                or gqa_union_predicted
                or gqa_union_pilot
            )
            and gqa_union_hip
            and gqa_union_page1_arena
            and (
                gqa_union_mass_fraction is None
                or gqa_union_predicted
                or gqa_union_pilot
            )
            and isinstance(gqa_union_fixed_indices, torch.Tensor)
            and isinstance(gqa_union_fixed_leaf_owners, torch.Tensor)
            and isinstance(gqa_union_fixed_slot_offsets, torch.Tensor)
            and isinstance(gqa_union_fixed_lengths, torch.Tensor)
            and all(
                (
                    name in buffers
                    for name in (
                        "gqa_union_hip_context_lens",
                        "gqa_union_hip_launch_lens",
                        "gqa_union_hip_segment_out",
                        "gqa_union_hip_exp_sums",
                        "gqa_union_hip_max_logits",
                        "gqa_union_fixed_active_mask",
                        "gqa_union_fixed_active_blocks",
                        "gqa_union_fixed_previous_counts",
                        "gqa_union_fixed_previous_slots",
                        "gqa_union_fixed_previous_cache_rows",
                    )
                )
            )
        )
        gqa_union_direct_compact = False
        gqa_union_implicit_lod = False
        gqa_union_persistent_slot_leaves = False
        gqa_union_fused_reduce_advance = False
        gqa_union_indirect_exact_pages = False
        gqa_union_local_sink_overlap = bool(
            gqa_union_overlap_local_sink
            and gqa_union_fixed_mask
            and (head_dim in (128, 256))
            and (batch == 1)
            and (batch * kv_heads <= 4)
            and (gqa_union_fixed_mask_reduce_block_d > 0)
            and (local_len % gqa_union_fixed_mask_tile_size == 0)
            and (not gqa_union_staged_fixed)
            and all(
                (
                    name in buffers
                    for name in (
                        "route_local_out",
                        "route_local_lse",
                        "gqa_union_hip_out",
                        "gqa_union_hip_lse",
                        "gqa_local_page1_context_lens",
                        "gqa_local_page1_segment_out",
                        "gqa_local_page1_exp_sums",
                        "gqa_local_page1_max_logits",
                    )
                )
            )
        )
        gqa_union_fixed_prepare_fused = False
        predicted_fixed_prepare = False
        gqa_union_fused_route_union = False
        gqa_union_direct_candidate_expand = False
        gqa_union_padded_candidate_expand = False
        gqa_union_fused_topk_expand = False
        gqa_union_direct_top4_attention = False
        gqa_union_grouped_topk_page_prefix = False
        gqa_union_direct_page_queue = False
        gqa_union_direct_slot_queue = False
        if (
            execute_state_route
            and (not gqa_union_mass)
            and (not gqa_union_predicted)
            and (not gqa_union_pilot)
        ):
            if recursive_state_route_backend == "resplit":
                route_resplit_begin = timing_begin()
                top_slots, _, _, _ = materialized_state_route_gqa(
                    q,
                    state_k,
                    state_v,
                    counts,
                    cache_indices,
                    buffers,
                    state_len=state_len,
                    kv_group_size=kv_group_size,
                    scale=scale,
                    open_count=open_count,
                    protected_len=protected_len,
                    max_leaf_tokens=max_leaf_tokens,
                    waves_per_eu=waves_per_eu,
                    timing_events=timing_events,
                )
                timing_end("route_resplit_total", route_resplit_begin)
                score_use_dot = True
            else:
                group_size = route_group_size
                active_groups = triton.cdiv(state_len, group_size * route_segment_tiles)
                max_groups = int(buffers["route_group_lse"].size(2))
                if route_segment_tiles != 1 or not route_gqa_grouped:
                    raise ValueError(
                        "the LoD release requires the grouped single-segment "
                        "decode router"
                    )
                if (
                    speculative_steps >= 2
                    and speculative_steps % 2 == 0
                    and (2 * kv_group_size <= 16)
                ):
                    route_kernel = _decode_route_coarse_gqa_mtp2_groups_kernel
                else:
                    route_kernel = _decode_route_coarse_gqa_groups_kernel
                score_use_dot = True
                use_compact_top4_candidates = bool(
                    compact_top4_candidates and open_count == 4
                )
                top4_candidates_requested = use_compact_top4_candidates
                route_candidates_per_group = (
                    4 if top4_candidates_requested and open_count == 4 else 8
                )
                gqa_union_fused_route_union = False
                gqa_union_direct_page_queue = False
                gqa_union_direct_slot_queue = False
                gqa_union_direct_top4_attention = False
                gqa_union_grouped_topk_page_prefix = False
                gqa_union_overlap_page_queue = False
                gqa_union_direct_candidate_expand = False
                gqa_union_padded_candidate_expand = False
                gqa_union_fused_topk_expand = False
                route_rows = (
                    batch // speculative_steps * kv_heads * (speculative_steps // 2)
                    if route_kernel is _decode_route_coarse_gqa_mtp2_groups_kernel
                    else batch * kv_heads
                    if route_gqa_grouped
                    else batch * query_heads
                )
                route_groups_begin = timing_begin()
                route_state_k = state_k
                route_keys_are_means = False
                if (
                    (gqa_union_score_only or gqa_union_compact_reuse_coarse)
                    and (not gqa_union_fixed_mask)
                    and (route_kernel is _decode_route_coarse_gqa_groups_kernel)
                    and isinstance(gqa_union_page1_k, torch.Tensor)
                ):
                    coarse_rows = (
                        int(state_k.size(0))
                        * int(state_k.size(1))
                        * int(state_k.size(2))
                    )
                    coarse_begin = int(gqa_union_page1_coarse_offset)
                    if coarse_begin >= 0 and coarse_begin + coarse_rows <= int(
                        gqa_union_page1_k.size(0)
                    ):
                        route_state_k = gqa_union_page1_k.narrow(
                            0, coarse_begin, coarse_rows
                        ).view_as(state_k)
                        route_keys_are_means = True
                route_arguments = (
                    q,
                    route_state_k,
                    state_v,
                    counts,
                    cache_indices,
                    buffers["route_candidate_scores"],
                    buffers["route_candidate_indices"],
                    buffers["route_group_out"],
                    buffers["route_group_lse"],
                    route_state_k.stride(0),
                    route_state_k.stride(1),
                    route_state_k.stride(2),
                    state_v.stride(0),
                    state_v.stride(1),
                    state_v.stride(2),
                    counts.stride(0),
                    counts.stride(1),
                    counts.stride(2),
                    state_len,
                )
                packed_route_candidates = False
                reuse_route_log_bias = False
                route_log_count_bias = counts
                route_log_count_bias_strides = (
                    counts.stride(0),
                    counts.stride(1),
                    counts.stride(2),
                )
                reuse_route_scores = False
                route_fused_decode_local = bool(
                    fuse_route_local
                    and route_kernel is _decode_route_coarse_gqa_groups_kernel
                    and (not gqa_union_score_only)
                    and (recursive_page_cache is not None or not gqa_union_decode)
                    and (route_mass_fraction is None)
                    and (route_residual_mass is None)
                    and (route_top_p is None)
                    and (not local_lens_are_logical)
                    and (not include_new or store_new_kv)
                    and (q.dtype == torch.bfloat16)
                    and (local_k.dtype == torch.bfloat16)
                    and (local_v.dtype == torch.bfloat16)
                    and (head_dim in (128, 256))
                    and local_k.is_contiguous()
                    and local_v.is_contiguous()
                    and (not include_new or new_k.is_contiguous())
                    and (not include_new or new_v.is_contiguous())
                    and (active_groups * group_size >= local_len + int(include_new))
                )
                route_extra = (
                    {
                        "KEYS_ARE_MEANS": route_keys_are_means,
                        "PACKED_CANDIDATES": packed_route_candidates,
                        "STORE_ALL_SCORES": reuse_route_scores,
                        "log_count_bias": route_log_count_bias,
                        "LOG_COUNT_BIAS_BATCH_STRIDE": route_log_count_bias_strides[0],
                        "LOG_COUNT_BIAS_HEAD_STRIDE": route_log_count_bias_strides[1],
                        "LOG_COUNT_BIAS_TOKEN_STRIDE": route_log_count_bias_strides[2],
                        "USE_LOG_COUNT_BIAS": reuse_route_log_bias,
                        "local_lens": local_lens,
                        "local_k": local_k,
                        "local_v": local_v,
                        "new_k": new_k,
                        "new_v": new_v,
                        "FUSE_LOCAL": route_fused_decode_local,
                        "LOCAL_CAPACITY": int(local_k.size(2)),
                        "LOCAL_LIMIT": local_len,
                        "INCLUDE_NEW": include_new,
                    }
                    if route_kernel is _decode_route_coarse_gqa_groups_kernel
                    else {}
                )
                if not route_segment_tiles > 1:
                    if (
                        gqa_union_fixed_mask
                        and route_kernel is _decode_route_coarse_gqa_groups_kernel
                        and (state_len >= local_len)
                    ):
                        gqa_union_fixed_prepare_fused = True
                        fixed_active_mask = buffers["gqa_union_fixed_active_mask"][
                            : batch * kv_heads
                        ]
                        fixed_active_blocks = buffers["gqa_union_fixed_active_blocks"][
                            : batch * kv_heads
                        ]
                        sink_len = int(sink_k.size(2)) if include_sink else 0
                        route_state_k = state_k
                        route_keys_are_means = False
                        coarse_rows = (
                            int(state_k.size(0))
                            * int(state_k.size(1))
                            * int(state_k.size(2))
                        )
                        coarse_begin = int(gqa_union_page1_coarse_offset)
                        if (
                            isinstance(gqa_union_page1_k, torch.Tensor)
                            and coarse_begin >= 0
                            and (
                                coarse_begin + coarse_rows
                                <= int(gqa_union_page1_k.size(0))
                            )
                        ):
                            route_state_k = gqa_union_page1_k.narrow(
                                0, coarse_begin, coarse_rows
                            ).view_as(state_k)
                            route_keys_are_means = True
                        _decode_route_coarse_gqa_groups_fixed_prepare_kernel[
                            route_rows, active_groups
                        ](
                            q,
                            route_state_k,
                            state_v,
                            counts,
                            cache_indices,
                            buffers["route_candidate_scores"],
                            buffers["route_candidate_indices"],
                            buffers["route_group_out"],
                            buffers["route_group_lse"],
                            local_lens,
                            gqa_union_fixed_lengths,
                            buffers["gqa_union_hip_context_lens"],
                            buffers["gqa_union_hip_launch_lens"],
                            new_k,
                            new_v,
                            gqa_union_page1_k,
                            gqa_union_page1_v,
                            buffers["gqa_union_destinations"],
                            buffers["gqa_union_fixed_previous_cache_rows"],
                            buffers["gqa_union_fixed_previous_counts"],
                            buffers["gqa_union_fixed_previous_slots"],
                            gqa_union_fixed_slot_offsets,
                            fixed_active_mask,
                            fixed_active_blocks,
                            route_state_k.stride(0),
                            route_state_k.stride(1),
                            route_state_k.stride(2),
                            state_v.stride(0),
                            state_v.stride(1),
                            state_v.stride(2),
                            counts.stride(0),
                            counts.stride(1),
                            counts.stride(2),
                            new_k.stride(0),
                            new_k.stride(1),
                            new_v.stride(0),
                            new_v.stride(1),
                            gqa_union_fixed_slot_offsets.stride(1),
                            fixed_active_mask.stride(0),
                            fixed_active_blocks.stride(0),
                            state_len,
                            QUERY_HEADS=query_heads,
                            KV_HEADS=kv_heads,
                            KV_GROUP_SIZE=kv_group_size,
                            HEAD_DIM=head_dim,
                            SCALE=float(scale),
                            GROUP_N=group_size,
                            MAX_GROUPS=max_groups,
                            PROTECTED_LEN=protected_len,
                            MAX_LEAF_TOKENS=max_leaf_tokens or 0,
                            SCORE_ONLY=gqa_union_score_only,
                            STATE_CAPACITY=int(state_k.size(2)),
                            UNION_CAPACITY=int(buffers["gqa_union_slots"].size(1)),
                            LOCAL_OFFSET=gqa_union_page1_local_offset,
                            LOCAL_CAPACITY=int(local_k.size(2)),
                            LOCAL_LIMIT=local_len,
                            SINK_LEN=sink_len,
                            LEAF_BEGIN=local_len + sink_len + int(state_k.size(2)),
                            MASK_CAPACITY=int(fixed_active_mask.size(1)),
                            TILE_SIZE=gqa_union_fixed_mask_tile_size,
                            RESET_BLOCK_N=64,
                            RESET_BLOCKS_N=4,
                            INCLUDE_NEW=include_new,
                            SEPARATE_LOCAL_SINK=gqa_union_local_sink_overlap,
                            KEYS_ARE_MEANS=route_keys_are_means,
                            REUSE_COARSE=gqa_union_fixed_reuse_coarse,
                            CANDIDATES_PER_GROUP=route_candidates_per_group,
                            num_warps=route_num_warps,
                            num_stages=3,
                            waves_per_eu=waves_per_eu,
                        )
                    elif route_kernel is _decode_route_coarse_gqa_mtp2_groups_kernel:
                        execution_marker = buffers.get(
                            "speculative_route_execution_marker"
                        )
                        local_execution_marker = buffers.get(
                            "speculative_local_execution_marker"
                        )
                        route_fused_mtp_local = bool(
                            "1" != "0"
                            and "1" != "0"
                            and local_lens_are_logical
                            and (not gqa_union_score_only)
                            and (route_mass_fraction is None)
                            and (route_residual_mass is None)
                            and (q.dtype == torch.bfloat16)
                            and (local_k.dtype == torch.bfloat16)
                            and (local_v.dtype == torch.bfloat16)
                        )
                        route_kernel[route_rows, active_groups](
                            *route_arguments,
                            execution_marker,
                            local_execution_marker,
                            local_lens,
                            local_k,
                            local_v,
                            local_k.stride(0),
                            local_k.stride(1),
                            local_k.stride(2),
                            local_v.stride(0),
                            local_v.stride(1),
                            local_v.stride(2),
                            local_len,
                            QUERY_HEADS=query_heads,
                            KV_HEADS=kv_heads,
                            KV_GROUP_SIZE=kv_group_size,
                            REQUEST_ROWS=batch // speculative_steps,
                            SPECULATIVE_STEPS=speculative_steps,
                            HEAD_DIM=head_dim,
                            SCALE=float(scale),
                            GROUP_N=group_size,
                            MAX_GROUPS=max_groups,
                            PROTECTED_LEN=protected_len,
                            MAX_LEAF_TOKENS=max_leaf_tokens or 0,
                            USE_DOT=score_use_dot,
                            FUSE_LOCAL=route_fused_mtp_local,
                            SCORE_ONLY=gqa_union_score_only,
                            CANDIDATES_PER_GROUP=route_candidates_per_group,
                            num_warps=route_num_warps,
                            waves_per_eu=waves_per_eu,
                        )
                    elif route_kernel is _decode_route_coarse_gqa_groups_kernel:
                        route_kernel[route_rows, active_groups](
                            *route_arguments,
                            state_lens,
                            buffers.get("gqa_union_counts", buffers["route_top_slots"]),
                            buffers.get(
                                "gqa_union_token_counts", buffers["route_top_slots"]
                            ),
                            buffers.get("gqa_union_epochs", buffers["route_top_slots"]),
                            QUERY_HEADS=query_heads,
                            KV_HEADS=kv_heads,
                            KV_GROUP_SIZE=kv_group_size,
                            HEAD_DIM=head_dim,
                            SCALE=float(scale),
                            GROUP_N=group_size,
                            MAX_GROUPS=max_groups,
                            PROTECTED_LEN=protected_len,
                            MAX_LEAF_TOKENS=max_leaf_tokens or 0,
                            USE_DOT=score_use_dot,
                            SCORE_ONLY=gqa_union_score_only,
                            USE_STATE_LENS=use_state_lens,
                            CANDIDATES_PER_GROUP=route_candidates_per_group,
                            FLOAT_TOP4=False,
                            FUSE_UNION_INIT=gqa_union_fused_route_union
                            or gqa_union_direct_candidate_expand,
                            UNION_SEQUENCE_CAPACITY=int(
                                buffers.get(
                                    "gqa_union_counts", buffers["route_top_slots"]
                                ).numel()
                            ),
                            num_warps=route_num_warps,
                            waves_per_eu=waves_per_eu,
                            **route_extra,
                        )
                timing_end("route_groups", route_groups_begin)
                route_reduce_begin = timing_begin()
                if gqa_union_score_only:
                    if not gqa_union_grouped_topk_page_prefix:
                        if not gqa_union_fused_topk_expand:
                            if not gqa_union_direct_compact:
                                route_reduce_kernel = _reduce_decode_route_topk_kernel
                                route_reduce_kernel[batch * query_heads,](
                                    buffers["route_candidate_scores"],
                                    buffers["route_candidate_indices"],
                                    buffers["route_top_slots"],
                                    buffers["route_top_scores"],
                                    active_groups,
                                    buffers["gqa_union_seen_stamps"],
                                    buffers["gqa_union_epochs"],
                                    buffers["gqa_union_counts"],
                                    buffers["gqa_union_slots"],
                                    QUERY_HEADS=query_heads,
                                    KV_HEADS=kv_heads,
                                    KV_GROUP_SIZE=kv_group_size,
                                    STATE_CAPACITY=int(state_k.size(2)),
                                    UNION_CAPACITY=int(
                                        buffers["gqa_union_slots"].size(1)
                                    ),
                                    ROUTE_COUNT=8,
                                    OPEN_COUNT=open_count,
                                    MAX_SEGMENTS=max_groups,
                                    CANDIDATE_BLOCK=triton.next_power_of_2(
                                        max(
                                            16,
                                            active_groups * route_candidates_per_group,
                                        )
                                    ),
                                    CANDIDATES_PER_GROUP=route_candidates_per_group,
                                    EXACT_TOP4=use_compact_top4_candidates,
                                    FLOAT_TOP4=False,
                                    STAMP_SELECTED=gqa_union_direct_top4_attention
                                    and (
                                        not gqa_union_direct_page_queue
                                        or gqa_union_overlap_page_queue
                                    )
                                    and (not gqa_union_direct_slot_queue),
                                    FUSE_UNION_BUILD=gqa_union_fused_route_union
                                    or gqa_union_direct_slot_queue,
                                    PACKED_CANDIDATES=packed_route_candidates,
                                    SORTED_GROUP_MERGE=False,
                                    FLOAT_SCORE_TOP4=False,
                                    UNION_SEQUENCE_CAPACITY=int(
                                        buffers["gqa_union_counts"].numel()
                                    ),
                                    num_warps=route_reduce_num_warps,
                                    waves_per_eu=waves_per_eu,
                                )
                elif route_parallel_reduce:
                    split_d = int(route_parallel_reduce_block_d)
                    if not (
                        split_d > 0
                        and open_count == 8
                        and (route_mass_fraction is None)
                    ):
                        _reduce_decode_route_coarse_vector_topk_kernel[
                            batch * query_heads,
                        ](
                            buffers["route_candidate_scores"],
                            buffers["route_candidate_indices"],
                            buffers["route_group_out"],
                            buffers["route_group_lse"],
                            buffers["route_top_slots"],
                            buffers["route_top_scores"],
                            buffers["coarse_out"],
                            buffers["coarse_lse"],
                            active_groups,
                            active_groups,
                            HEAD_DIM=head_dim,
                            STATE_CAPACITY=int(state_k.size(2)),
                            ROUTE_COUNT=8,
                            OPEN_COUNT=open_count,
                            MAX_SEGMENTS=max_groups,
                            CANDIDATE_BLOCK=triton.next_power_of_2(
                                max(16, active_groups * route_candidates_per_group)
                            ),
                            SEGMENT_BLOCK=triton.next_power_of_2(active_groups),
                            APPLY_MASS_CUTOFF=route_mass_fraction is not None,
                            LOG_MASS_FRACTION=math.log(float(route_mass_fraction))
                            if route_mass_fraction is not None
                            else 0.0,
                            CANDIDATES_PER_GROUP=route_candidates_per_group,
                            EXACT_TOP4=use_compact_top4_candidates,
                            num_warps=route_reduce_num_warps,
                            waves_per_eu=waves_per_eu,
                        )
                else:
                    _reduce_decode_route_coarse_kernel[batch * query_heads,](
                        buffers["route_candidate_scores"],
                        buffers["route_candidate_indices"],
                        buffers["route_group_out"],
                        buffers["route_group_lse"],
                        buffers["route_top_slots"],
                        buffers["route_top_scores"],
                        buffers["coarse_out"],
                        buffers["coarse_lse"],
                        active_groups,
                        HEAD_DIM=head_dim,
                        STATE_CAPACITY=int(state_k.size(2)),
                        ROUTE_COUNT=8,
                        OPEN_COUNT=open_count,
                        MAX_GROUPS=max_groups,
                        CANDIDATE_TILE=triton.next_power_of_2(
                            max(16, active_groups * route_candidates_per_group)
                        )
                        if use_compact_top4_candidates
                        else 1024,
                        APPLY_MASS_CUTOFF=route_mass_fraction is not None,
                        LOG_MASS_FRACTION=math.log(float(route_mass_fraction))
                        if route_mass_fraction is not None
                        else 0.0,
                        CANDIDATES_PER_GROUP=route_candidates_per_group,
                        EXACT_TOP4=use_compact_top4_candidates,
                        num_warps=route_reduce_num_warps,
                        waves_per_eu=waves_per_eu,
                    )
                timing_end("route_reduce", route_reduce_begin)
                top_slots = buffers["route_top_slots"]
        if recursive_page_cache is not None:

            def cache_tensor(name: str) -> torch.Tensor:
                value = recursive_page_cache.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"fused recursive decode cache is missing {name}")
                return value

            reuse_separate_local = bool(
                route_fused_mtp_local
                or route_fused_decode_local
                or (route_residual_mass is not None and reuse_residual_local_attention)
            )
            if not reuse_separate_local:
                local_begin = timing_begin()
                wide_scores = buffers.get("wide_gqa_local_scores")
                wide_gqa_local = head_dim in (128, 256, 512) and 1 < kv_group_size <= 16
                if wide_gqa_local:
                    score_block_n = 32
                    wide_score_begin = timing_begin()
                    _wide_gqa_local_scores_kernel[
                        batch * kv_heads, triton.cdiv(local_len + 1, score_block_n)
                    ](
                        q,
                        cache_indices,
                        local_lens,
                        local_k,
                        new_k,
                        wide_scores,
                        local_k.stride(0),
                        local_k.stride(1),
                        local_k.stride(2),
                        new_k.stride(0),
                        new_k.stride(1),
                        wide_scores.stride(0),
                        wide_scores.stride(1),
                        local_len,
                        QUERY_HEADS=query_heads,
                        KV_HEADS=kv_heads,
                        KV_GROUP_SIZE=kv_group_size,
                        HEAD_DIM=head_dim,
                        SCALE=float(scale),
                        BLOCK_M=16,
                        BLOCK_N=score_block_n,
                        INCLUDE_NEW=include_new,
                        LOCAL_LENS_LOGICAL=local_lens_are_logical,
                        num_warps=4,
                        waves_per_eu=waves_per_eu,
                    )
                    timing_end("recursive_local_wide_score", wide_score_begin)
                    value_block_d = 32
                    wide_value_begin = timing_begin()
                    _wide_gqa_local_value_kernel[
                        batch * kv_heads, triton.cdiv(head_dim, value_block_d)
                    ](
                        cache_indices,
                        local_lens,
                        local_k,
                        local_v,
                        new_k,
                        new_v,
                        wide_scores,
                        buffers["route_local_out"],
                        buffers["route_local_lse"],
                        local_k.stride(0),
                        local_k.stride(1),
                        local_k.stride(2),
                        local_v.stride(0),
                        local_v.stride(1),
                        local_v.stride(2),
                        new_k.stride(0),
                        new_k.stride(1),
                        new_v.stride(0),
                        new_v.stride(1),
                        wide_scores.stride(0),
                        wide_scores.stride(1),
                        local_len,
                        QUERY_HEADS=query_heads,
                        KV_HEADS=kv_heads,
                        KV_GROUP_SIZE=kv_group_size,
                        HEAD_DIM=head_dim,
                        BLOCK_M=16,
                        BLOCK_D=value_block_d,
                        BLOCK_K=32,
                        INCLUDE_NEW=include_new,
                        LOCAL_LENS_LOGICAL=local_lens_are_logical,
                        num_warps=4,
                        waves_per_eu=waves_per_eu,
                    )
                    timing_end("recursive_local_wide_value", wide_value_begin)
                timing_end("recursive_local", local_begin)
            quantized_attention = bool(
                recursive_page_cache.get("quantization_finalized", False)
            )
            quantized_summaries = bool(
                recursive_page_cache.get("summary_quantization_finalized", False)
            )
            materialized_page_scores = None
            recursive_begin = timing_begin()
            recursive_out, recursive_lse = query_major_indexed_residual_page_attention(
                q,
                state_k,
                state_v,
                counts,
                cache_tensor("leaf_k"),
                cache_tensor("leaf_v"),
                cache_tensor("page_indices"),
                cache_tensor("page_sum_k"),
                cache_tensor("page_sum_v"),
                cache_tensor("page_counts"),
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                top_slots,
                cache_indices=cache_indices,
                kv_group_size=kv_group_size,
                scale=scale,
                hash_probes=hash_probes,
                page_block_n=recursive_page_select_block_n
                if materialized_page_scores is not None
                else block_n,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
                quantized_leaf_k=cache_tensor("quantized_leaf_k")
                if quantized_attention
                else None,
                quantized_leaf_v=cache_tensor("quantized_leaf_v")
                if quantized_attention
                else None,
                page_k_scales=cache_tensor("page_k_scales")
                if quantized_attention
                else None,
                page_v_scales=cache_tensor("page_v_scales")
                if quantized_attention
                else None,
                page_quantized_counts=cache_tensor("page_quantized_counts")
                if quantized_attention
                else None,
                quantized_page_sum_k=cache_tensor("quantized_page_sum_k")
                if quantized_summaries
                else None,
                quantized_page_sum_v=cache_tensor("quantized_page_sum_v")
                if quantized_summaries
                else None,
                page_sum_k_scales=cache_tensor("page_sum_k_scales")
                if quantized_summaries
                else None,
                page_sum_v_scales=cache_tensor("page_sum_v_scales")
                if quantized_summaries
                else None,
                quant_group_size=recursive_quant_group_size,
                quant_token_group_size=recursive_quant_token_group_size,
                quant_bits=int(recursive_page_cache.get("leaf_quant_bits", 4)),
                output_buffer=partial_out,
                lse_buffer=partial_lse,
                route_parallel=True,
                materialized_page_scores=materialized_page_scores,
            )
            timing_end("recursive_leaf", recursive_begin)
            final_reduce_begin = timing_begin()
            _reduce_routed_split_decode_lod_attention_kernel[batch * query_heads,](
                q,
                sink_k,
                sink_v,
                state_k,
                state_v,
                counts,
                cache_indices,
                local_lens,
                local_k,
                local_v,
                new_k,
                new_v,
                top_slots,
                buffers["route_top_scores"],
                buffers["coarse_out"],
                buffers["coarse_lse"],
                recursive_out,
                recursive_lse,
                buffers["route_local_out"],
                buffers["route_local_lse"],
                output,
                sink_k.stride(0),
                sink_k.stride(1),
                sink_k.stride(2),
                sink_v.stride(0),
                sink_v.stride(1),
                sink_v.stride(2),
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
                state_v.stride(0),
                state_v.stride(1),
                state_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                QUERY_HEADS=query_heads,
                KV_GROUP_SIZE=kv_group_size,
                HEAD_DIM=head_dim,
                STATE_CAPACITY=int(state_k.size(2)),
                ROUTE_COUNT=int(top_slots.size(-1)),
                SPLITS=split_kv,
                ROUTE_SPLITS=1,
                INCLUDE_SEPARATE_LOCAL=not (
                    route_fused_mtp_local or route_fused_decode_local
                ),
                SEPARATE_LOCAL_SPLITS=1,
                FUSE_LOCAL_SCAN=False,
                INCLUDE_NEW=False,
                INCLUDE_SINK=include_sink,
                SINK_LEN=int(sink_k.size(2)),
                LOCAL_BLOCK_N=32,
                SCALE=float(scale),
                USE_DOT=score_use_dot,
                ADVANCE_LOCAL=(
                    include_new and ragged_local_lens and advance_local_lens
                ),
                SUBTRACT_ROUTES=True,
                num_warps=final_reduce_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("final_reduce", final_reduce_begin)
            if exact_decode_threshold > 0:
                if bool(recursive_page_cache.get("quantization_finalized", False)):
                    apply_exact_int4_override(recursive_page_cache)
                else:
                    exact_leaf_k = recursive_page_cache.get("leaf_k")
                    exact_leaf_v = recursive_page_cache.get("leaf_v")
                    if not isinstance(exact_leaf_k, torch.Tensor) or not isinstance(
                        exact_leaf_v, torch.Tensor
                    ):
                        raise ValueError(
                            "short-context exact decode cache is missing archived leaves"
                        )
                    apply_exact_bf16_override(exact_leaf_k, exact_leaf_v)
            timing_end("recursive_total", recursive_total_begin)
            return output
        shared_mtp_local = bool(
            speculative_steps == 2
            and (not route_fused_mtp_local)
            and (not gqa_union_fixed_mask)
            and local_lens_are_logical
            and ("1" != "0")
            and fuse_state_route
            and (recursive_page_cache is None)
            and (2 * kv_group_size <= 16)
            and (q.dtype == torch.bfloat16)
            and (local_k.dtype == torch.bfloat16)
            and (local_v.dtype == torch.bfloat16)
            and ("speculative_local_execution_marker" in buffers)
            and ("speculative_local_partial_out" in buffers)
            and ("speculative_local_partial_lse" in buffers)
        )
        effective_fuse_final_reduce = bool(fuse_final_reduce and (not shared_mtp_local))
        fused_completion = (
            buffers["completion"]
            if fuse_state_route and effective_fuse_final_reduce
            else partial_lse
        )
        leaf_begin = timing_begin()
        gqa_union_required = (
            "gqa_union_seen_stamps",
            "gqa_union_epochs",
            "gqa_union_counts",
            "gqa_union_token_counts",
            "gqa_union_slots",
            "gqa_union_token_indices",
        )
        gqa_union_leaf = bool(
            gqa_union_decode
            and fuse_state_route
            and (not fuse_final_reduce)
            and (recursive_page_cache is None)
            and (flat_page_indices is not None)
            and (not flat_int8)
            and (q.dtype == torch.bfloat16)
            and (1 < kv_group_size <= 16)
            and (head_dim in (128, 256, 512))
            and (int(top_slots.size(-1)) == 8)
            and (route_top_p is None)
            and (route_residual_mass is None)
            and (route_mass_fraction is None or gqa_union_mass)
            and (not gqa_union_unified or gqa_union_score_only or gqa_union_page1_arena)
            and all((name in buffers for name in gqa_union_required))
        )
        gqa_union_hip_exact = bool(
            gqa_union_leaf
            and gqa_union_hip
            and all(
                (
                    name in buffers
                    for name in (
                        "gqa_union_hip_block_table",
                        "gqa_union_hip_context_lens",
                        "gqa_union_hip_launch_lens",
                        "gqa_union_hip_cu_q",
                        "gqa_union_hip_out",
                        "gqa_union_hip_lse",
                        "gqa_union_hip_segment_out",
                        "gqa_union_hip_exp_sums",
                        "gqa_union_hip_max_logits",
                    )
                )
            )
        )
        gqa_union_aiter_final = bool(gqa_union_hip_exact and gqa_union_page1_arena)
        buffers["gqa_union_last_requested"] = bool(gqa_union_decode)
        buffers["gqa_union_last_score_only"] = bool(gqa_union_score_only)
        buffers["gqa_union_last_eligible"] = bool(gqa_union_leaf)
        buffers["gqa_union_last_mass_cutoff"] = bool(
            gqa_union_mass or gqa_union_predicted or gqa_union_pilot
        )
        buffers["gqa_union_last_predicted_mass"] = bool(gqa_union_predicted)
        buffers["gqa_union_last_pilot_z"] = bool(gqa_union_pilot)
        buffers["gqa_union_last_hip"] = bool(gqa_union_hip_exact)
        buffers["gqa_union_last_aiter_final"] = bool(gqa_union_aiter_final)
        buffers["gqa_union_last_group64_padded"] = bool(
            gqa_union_group64_padded and gqa_union_aiter_final
        )
        buffers["gqa_union_last_staged_fixed_aiter"] = bool(
            gqa_union_staged_fixed and gqa_union_aiter_final
        )
        buffers["gqa_union_last_fixed_mask_aiter"] = bool(
            gqa_union_fixed_mask and gqa_union_aiter_final
        )
        buffers["gqa_union_last_fixed_mask_reuse_coarse"] = bool(
            gqa_union_fixed_reuse_coarse
            and gqa_union_fixed_mask
            and gqa_union_aiter_final
        )
        buffers["gqa_union_last_overlap_local_sink"] = bool(
            gqa_union_local_sink_overlap and gqa_union_aiter_final
        )
        cooperative_hip_eligible = False
        if gqa_cooperative_hip and q.is_cuda:
            from lod_attention.kernels.gqa_cooperative_decode import (
                gqa_cooperative_decode_available,
            )

            device_index = q.device.index
            cooperative_hip_eligible = bool(
                speculative_steps == 1
                and kv_group_size == 4
                and (head_dim == 256)
                and (q.dtype == torch.bfloat16)
                and (
                    page_k.dtype == torch.bfloat16
                    and page_v.dtype == torch.bfloat16
                    or (
                        flat_int8
                        and flat_page_k_scales is not None
                        and (flat_page_v_scales is not None)
                    )
                )
                and (hash_probes in {-1, 0})
                and gqa_cooperative_decode_available(device_index)
            )
        cooperative_leaf = bool(
            gqa_cooperative_leaf
            and fuse_state_route
            and (not fuse_final_reduce)
            and (not use_dot)
            and (int(top_slots.size(-1)) == 8)
            and (split_kv == 8)
            and ("gqa_local_partial_out" in buffers)
            and ("gqa_local_partial_lse" in buffers)
            and ("gqa_route_partial_out" in buffers)
            and ("gqa_route_partial_lse" in buffers)
            and cooperative_hip_eligible
        )
        cooperative_separate_local = False
        final_partial_out = partial_out
        final_partial_lse = partial_lse
        final_splits = split_kv
        final_route_splits = 1
        if gqa_union_leaf:
            union_begin = timing_begin()
            sequence_count = batch * kv_heads
            low_sequence_count = bool(
                sequence_count * kv_group_size <= 8
                or (batch == 1 and sequence_count <= 4)
            )
            union_capacity = int(buffers["gqa_union_slots"].size(1))
            index_capacity = int(buffers["gqa_union_token_indices"].size(1))
            hip_block_table = (
                buffers["gqa_union_hip_block_table"]
                if gqa_union_hip_exact
                else buffers["gqa_union_token_indices"]
            )
            hip_context_lens = (
                buffers["gqa_union_hip_context_lens"]
                if gqa_union_hip_exact
                else buffers["gqa_union_token_counts"]
            )
            direct_fixed_routes = bool(
                gqa_union_fixed_mask
                and gqa_union_fixed_mask_direct_routes
                and (not gqa_union_mass)
                and (not gqa_union_predicted)
                and (not gqa_union_pilot)
            )
            buffers["gqa_union_last_direct_fixed_routes"] = direct_fixed_routes
            if (
                not gqa_union_mass
                and (not gqa_union_predicted)
                and (not gqa_union_pilot)
                and (not gqa_union_direct_compact)
                and (not gqa_union_fused_route_union)
                and (not gqa_union_direct_candidate_expand)
                and (not direct_fixed_routes or gqa_union_fixed_reuse_coarse)
            ):
                _decode_topk_gqa_union_kernel[sequence_count,](
                    top_slots,
                    cache_indices,
                    local_lens,
                    state_lens,
                    buffers["gqa_union_seen_stamps"],
                    buffers["gqa_union_epochs"],
                    buffers["gqa_union_counts"],
                    buffers["gqa_union_token_counts"],
                    buffers["gqa_union_slots"],
                    buffers.get(
                        "gqa_union_hip_context_lens", buffers["gqa_union_token_counts"]
                    ),
                    new_k,
                    new_v,
                    gqa_union_page1_k
                    if isinstance(gqa_union_page1_k, torch.Tensor)
                    else state_k,
                    gqa_union_page1_v
                    if isinstance(gqa_union_page1_v, torch.Tensor)
                    else state_v,
                    state_len,
                    top_slots.stride(0),
                    top_slots.stride(1),
                    new_k.stride(0),
                    new_k.stride(1),
                    new_v.stride(0),
                    new_v.stride(1),
                    QUERY_HEADS=query_heads,
                    KV_HEADS=kv_heads,
                    KV_GROUP_SIZE=kv_group_size,
                    ROUTE_COUNT=int(top_slots.size(-1)),
                    STATE_CAPACITY=int(state_k.size(2)),
                    CANDIDATE_BLOCK=triton.next_power_of_2(union_capacity),
                    LOCAL_LIMIT=local_len,
                    LOCAL_OFFSET=gqa_union_page1_local_offset,
                    LOCAL_CAPACITY=int(local_k.size(2)),
                    SINK_LEN=int(sink_k.size(2)) if include_sink else 0,
                    HEAD_DIM=head_dim,
                    INCLUDE_NEW=include_new,
                    PREPARE_IMPLICIT_LOD=gqa_union_implicit_lod,
                    USE_STATE_LENS=use_state_lens,
                    num_warps=4,
                    waves_per_eu=waves_per_eu,
                )
            if gqa_union_fixed_mask:
                fixed_attention_begin = timing_begin()
                from lod_attention.kernels.aiter_page1_attention import (
                    kernel_page1_attention_3d_bias_fixed_mask,
                )

                fixed_context_lens = buffers["gqa_union_hip_context_lens"][
                    :sequence_count
                ]
                fixed_launch_lens = buffers["gqa_union_hip_launch_lens"][
                    :sequence_count
                ]
                sink_len = int(sink_k.size(2)) if include_sink else 0
                fixed_active_mask = buffers["gqa_union_fixed_active_mask"][
                    :sequence_count
                ]
                fixed_active_blocks = buffers["gqa_union_fixed_active_blocks"][
                    :sequence_count
                ]
                if not gqa_union_fixed_prepare_fused:
                    fallback_prepare_begin = timing_begin()
                    _prepare_aiter_fixed_mask_context_kernel[sequence_count,](
                        cache_indices,
                        local_lens,
                        gqa_union_fixed_lengths,
                        fixed_context_lens,
                        fixed_launch_lens,
                        new_k,
                        new_v,
                        gqa_union_page1_k,
                        gqa_union_page1_v,
                        buffers["gqa_union_destinations"],
                        fixed_active_mask,
                        fixed_active_blocks,
                        new_k.stride(0),
                        new_k.stride(1),
                        new_v.stride(0),
                        new_v.stride(1),
                        KV_HEADS=kv_heads,
                        MASK_STRIDE=fixed_active_mask.stride(0),
                        BLOCK_STRIDE=fixed_active_blocks.stride(0),
                        LOCAL_OFFSET=gqa_union_page1_local_offset,
                        LOCAL_CAPACITY=int(local_k.size(2)),
                        LOCAL_LIMIT=local_len,
                        HEAD_DIM=head_dim,
                        SINK_LEN=sink_len,
                        STATE_CAPACITY=int(state_k.size(2)),
                        LEAF_BEGIN=local_len + sink_len + int(state_k.size(2)),
                        TILE_SIZE=gqa_union_fixed_mask_tile_size,
                        LOCAL_MASK_BLOCK=triton.next_power_of_2(max(1, local_len)),
                        SINK_MASK_BLOCK=triton.next_power_of_2(max(1, sink_len)),
                        COARSE_MASK_BLOCK=256,
                        PREFIX_BLOCKS_BLOCK=triton.next_power_of_2(
                            max(
                                1,
                                triton.cdiv(
                                    local_len + sink_len + int(state_k.size(2)),
                                    gqa_union_fixed_mask_tile_size,
                                ),
                            )
                        ),
                        INCLUDE_NEW=include_new,
                        SEPARATE_LOCAL_SINK=gqa_union_local_sink_overlap,
                        REUSE_COARSE=gqa_union_fixed_reuse_coarse,
                        num_warps=1,
                        waves_per_eu=waves_per_eu,
                    )
                    _reset_aiter_fixed_previous_union_kernel[
                        sequence_count, union_capacity
                    ](
                        cache_indices,
                        buffers["gqa_union_fixed_previous_cache_rows"],
                        buffers["gqa_union_fixed_previous_counts"],
                        buffers["gqa_union_fixed_previous_slots"],
                        gqa_union_fixed_slot_offsets,
                        fixed_active_mask,
                        fixed_active_blocks,
                        gqa_union_fixed_slot_offsets.stride(1),
                        fixed_active_mask.stride(0),
                        fixed_active_blocks.stride(0),
                        KV_HEADS=kv_heads,
                        STATE_CAPACITY=int(state_k.size(2)),
                        UNION_CAPACITY=union_capacity,
                        LEAF_BEGIN=local_len + sink_len + int(state_k.size(2)),
                        MASK_CAPACITY=int(fixed_active_mask.size(1)),
                        TILE_SIZE=gqa_union_fixed_mask_tile_size,
                        BLOCK_N=64,
                        BLOCKS_N=4,
                        num_warps=1,
                        waves_per_eu=1,
                    )
                    timing_end(
                        "gqa_union_fixed_fallback_prepare", fallback_prepare_begin
                    )
                fixed_query = q[:, :, 0, :].reshape(
                    sequence_count, kv_group_size, head_dim
                )
                allocated_segments = int(buffers["gqa_union_hip_exp_sums"].size(2))
                unified_segments = allocated_segments
                if gqa_union_fixed_mask_adaptive_segments and allocated_segments >= 256:
                    single_request_mtp = bool(
                        speculative_steps == 2 and batch == speculative_steps
                    )
                    programs_per_scan = max(
                        1, (1024 + sequence_count - 1) // sequence_count
                    )
                    target_segments = max(
                        16, min(256, 1 << (programs_per_scan - 1).bit_length())
                    )
                    unified_segments = 256 if single_request_mtp else target_segments
                fixed_segment_out = buffers["gqa_union_hip_segment_out"][
                    :sequence_count, :, :unified_segments
                ]
                fixed_exp_sums = buffers["gqa_union_hip_exp_sums"][
                    :sequence_count, :, :unified_segments
                ]
                fixed_max_logits = buffers["gqa_union_hip_max_logits"][
                    :sequence_count, :, :unified_segments
                ]
                buffers["gqa_union_last_effective_segments"] = unified_segments
                use_split_d_reduce = bool(
                    head_dim != 512
                    and gqa_union_fixed_mask_reduce_block_d
                    and low_sequence_count
                )
                mask_prepare_begin = timing_begin()
                if direct_fixed_routes:
                    _apply_aiter_fixed_direct_routes_kernel[
                        sequence_count, union_capacity
                    ](
                        top_slots,
                        cache_indices,
                        buffers["gqa_union_counts"],
                        buffers["gqa_union_token_counts"],
                        buffers["gqa_union_epochs"],
                        buffers["gqa_union_fixed_previous_counts"],
                        buffers["gqa_union_fixed_previous_slots"],
                        buffers["gqa_union_fixed_previous_cache_rows"],
                        gqa_union_fixed_slot_offsets,
                        fixed_active_mask,
                        fixed_active_blocks,
                        buffers["gqa_union_fixed_execution_geometry"],
                        top_slots.stride(0),
                        top_slots.stride(1),
                        gqa_union_fixed_slot_offsets.stride(1),
                        fixed_active_mask.stride(0),
                        fixed_active_blocks.stride(0),
                        QUERY_HEADS=query_heads,
                        KV_HEADS=kv_heads,
                        KV_GROUP_SIZE=kv_group_size,
                        ROUTE_COUNT=int(top_slots.size(-1)),
                        STATE_CAPACITY=int(state_k.size(2)),
                        UNION_CAPACITY=union_capacity,
                        LEAF_BEGIN=local_len + sink_len + int(state_k.size(2)),
                        MASK_CAPACITY=int(fixed_active_mask.size(1)),
                        TILE_SIZE=gqa_union_fixed_mask_tile_size,
                        BLOCK_N=64,
                        BLOCKS_N=4,
                        EFFECTIVE_SEGMENTS=unified_segments,
                        SPLIT_D_REDUCE=use_split_d_reduce,
                        PRESERVE_UNION_METADATA=gqa_union_fixed_reuse_coarse,
                        num_warps=1,
                        waves_per_eu=1,
                    )
                timing_end("gqa_union_fixed_mask_prepare", mask_prepare_begin)
                fixed_out = (
                    buffers["gqa_union_hip_out"][:sequence_count]
                    if gqa_union_fixed_reuse_coarse
                    else output[:, :, 0, :].reshape(
                        sequence_count, kv_group_size, head_dim
                    )
                )
                fixed_scan_begin = timing_begin()
                low_row_scan = bool(
                    gqa_union_fixed_mask_adaptive_segments and low_sequence_count
                )
                scan_num_warps = (
                    1 if low_row_scan else gqa_union_fixed_mask_scan_num_warps
                )
                scan_waves_per_eu = (
                    1 if low_row_scan else gqa_union_fixed_mask_scan_waves_per_eu
                )
                if not head_dim == 512:
                    kernel_page1_attention_3d_bias_fixed_mask[
                        sequence_count, 1, unified_segments
                    ](
                        fixed_segment_out,
                        fixed_max_logits,
                        fixed_exp_sums,
                        fixed_query,
                        gqa_union_page1_k,
                        gqa_union_page1_v,
                        gqa_union_page1_bias,
                        gqa_union_fixed_indices,
                        fixed_active_mask,
                        fixed_active_blocks,
                        gqa_union_fixed_lengths,
                        cache_indices,
                        float(scale),
                        gqa_union_fixed_indices.stride(1),
                        fixed_active_mask.stride(0),
                        fixed_active_blocks.stride(0),
                        fixed_query.stride(0),
                        fixed_query.stride(1),
                        NUM_QUERY_HEADS=kv_group_size,
                        KV_HEADS=kv_heads,
                        STATE_CAPACITY=int(state_k.size(2)),
                        LOCAL_LIMIT=local_len,
                        SINK_LEN=sink_len,
                        LEAF_BEGIN=local_len + sink_len + int(state_k.size(2)),
                        TILE_SIZE=gqa_union_fixed_mask_tile_size,
                        HEAD_SIZE=head_dim,
                        BLOCK_M=16,
                        NUM_SEGMENTS=unified_segments,
                        INCLUDE_NEW=include_new,
                        PREFIX_SKIP=local_len if gqa_union_local_sink_overlap else 0,
                        num_warps=scan_num_warps,
                        waves_per_eu=scan_waves_per_eu,
                        num_stages=gqa_union_fixed_mask_scan_num_stages,
                    )
                timing_end("gqa_union_fixed_mask_scan", fixed_scan_begin)
                if not predicted_fixed_prepare:
                    if head_dim != 512:
                        buffers["gqa_union_last_split_d_reduce"] = use_split_d_reduce
                        if use_split_d_reduce:
                            reduce_grid = (
                                sequence_count,
                                kv_group_size,
                                triton.cdiv(
                                    head_dim, gqa_union_fixed_mask_reduce_block_d
                                ),
                            )
                            if not gqa_union_local_sink_overlap:
                                _reduce_aiter_page1_segments_with_lse_split_d_kernel[
                                    reduce_grid
                                ](
                                    fixed_segment_out,
                                    fixed_max_logits,
                                    fixed_exp_sums,
                                    fixed_context_lens,
                                    fixed_out,
                                    buffers["gqa_union_hip_lse"][:sequence_count],
                                    buffers["gqa_union_epochs"],
                                    buffers["gqa_union_counts"],
                                    buffers["gqa_union_token_counts"],
                                    OUTPUT_STRIDE_0=fixed_out.stride(0),
                                    OUTPUT_STRIDE_1=fixed_out.stride(1),
                                    QUERY_ROWS=kv_group_size,
                                    HEAD_DIM=head_dim,
                                    SEGMENTS=unified_segments,
                                    TILE_SIZE=gqa_union_fixed_mask_tile_size,
                                    BLOCK_D=gqa_union_fixed_mask_reduce_block_d,
                                    ADVANCE_QUEUE=gqa_union_pilot,
                                    num_warps=2,
                                    waves_per_eu=waves_per_eu,
                                )
                        else:
                            _reduce_aiter_page1_segments_with_lse_kernel[
                                sequence_count, kv_group_size
                            ](
                                fixed_segment_out,
                                fixed_max_logits,
                                fixed_exp_sums,
                                fixed_context_lens,
                                fixed_out,
                                buffers["gqa_union_hip_lse"][:sequence_count],
                                buffers["gqa_union_epochs"],
                                buffers["gqa_union_counts"],
                                buffers["gqa_union_token_counts"],
                                cache_indices,
                                local_lens,
                                OUTPUT_STRIDE_0=fixed_out.stride(0),
                                OUTPUT_STRIDE_1=fixed_out.stride(1),
                                QUERY_ROWS=kv_group_size,
                                HEAD_DIM=head_dim,
                                SEGMENTS=unified_segments,
                                TILE_SIZE=gqa_union_fixed_mask_tile_size,
                                ADVANCE_QUEUE=gqa_union_pilot,
                                ADVANCE_LOCAL=False,
                                KV_HEADS=kv_heads,
                                num_warps=2,
                                waves_per_eu=waves_per_eu,
                            )
                timing_end("gqa_union_fixed_mask_attention", fixed_attention_begin)
                timing_end("gqa_union_indices", union_begin)
                timing_end("leaf_local", leaf_begin)
                if include_new and ragged_local_lens:
                    advance_decode_cache_lengths(cache_indices, local_lens)
                apply_exact_flat_bf16_override()
                return output
            if not (gqa_union_direct_compact and gqa_union_aiter_final):
                if not (gqa_union_group64_padded and gqa_union_aiter_final):
                    if not (gqa_union_fused_topk_expand and gqa_union_aiter_final):
                        if not (
                            gqa_union_padded_candidate_expand and gqa_union_aiter_final
                        ):
                            if not (
                                gqa_union_direct_candidate_expand
                                and gqa_union_aiter_final
                            ):
                                _expand_decode_topk_gqa_union_kernel[
                                    sequence_count, union_capacity + 1
                                ](
                                    cache_indices,
                                    local_lens,
                                    flat_page_indices,
                                    slot_pages,
                                    overflow_page_keys,
                                    overflow_page_values,
                                    overflow_used,
                                    slot_lengths,
                                    buffers["gqa_union_counts"],
                                    buffers["gqa_union_token_counts"],
                                    buffers["gqa_union_slots"],
                                    buffers["gqa_union_token_indices"],
                                    hip_block_table,
                                    hip_context_lens,
                                    gqa_union_fixed_indices
                                    if gqa_union_persistent_slot_leaves
                                    else flat_page_indices,
                                    gqa_union_fixed_slot_offsets
                                    if gqa_union_persistent_slot_leaves
                                    else slot_lengths,
                                    KV_HEADS=kv_heads,
                                    KV_GROUP_SIZE=kv_group_size,
                                    PAGE_CAPACITY=int(page_shape.size(2)),
                                    LEAF_CAPACITY=int(page_k.size(2)),
                                    STATE_CAPACITY=int(slot_pages.size(2)),
                                    INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                                    HASH_CAPACITY=int(overflow_page_values.size(2)),
                                    HASH_PROBES=hash_probes,
                                    PAGE_SIZE=int(page_shape.size(3)),
                                    LOCAL_LIMIT=local_len,
                                    INDEX_CAPACITY=index_capacity,
                                    UNION_CAPACITY=union_capacity,
                                    BLOCK_K=64,
                                    INCLUDE_NEW=include_new,
                                    HIP_EXACT=gqa_union_hip_exact,
                                    HIP_UNIFIED_ARENA=gqa_union_aiter_final,
                                    ARENA_LEAF_OFFSET=gqa_union_page1_leaf_offset
                                    if gqa_union_aiter_final
                                    else 0,
                                    IMPLICIT_LOD=gqa_union_implicit_lod,
                                    PERSISTENT_SLOT_LEAVES=gqa_union_persistent_slot_leaves,
                                    FIXED_CAPACITY=int(gqa_union_fixed_indices.size(2))
                                    if gqa_union_persistent_slot_leaves
                                    else 1,
                                    FIXED_LEAF_BEGIN=local_len
                                    + (int(sink_k.size(2)) if include_sink else 0)
                                    + int(state_k.size(2)),
                                    num_warps=1,
                                    waves_per_eu=waves_per_eu,
                                )
            if not gqa_union_staged_fixed:
                if (
                    gqa_union_aiter_final
                    and (not gqa_union_direct_compact)
                    and (not gqa_union_implicit_lod)
                ):
                    coarse_blocks = triton.cdiv(state_len, 64)
                    _append_decode_gqa_union_arena_entries_kernel[
                        sequence_count,
                        1 if gqa_union_compact_reuse_coarse else coarse_blocks + 1,
                    ](
                        cache_indices,
                        local_lens,
                        state_lens,
                        counts,
                        buffers["gqa_union_seen_stamps"],
                        buffers["gqa_union_epochs"],
                        new_k,
                        new_v,
                        gqa_union_page1_k,
                        gqa_union_page1_v,
                        gqa_union_page1_bias,
                        hip_block_table,
                        hip_context_lens,
                        buffers["gqa_union_token_counts"],
                        counts.stride(0),
                        counts.stride(1),
                        counts.stride(2),
                        new_k.stride(0),
                        new_k.stride(1),
                        new_v.stride(0),
                        new_v.stride(1),
                        KV_HEADS=kv_heads,
                        STATE_LEN=state_len,
                        STATE_CAPACITY=int(state_k.size(2)),
                        INDEX_CAPACITY=index_capacity,
                        LOCAL_OFFSET=gqa_union_page1_local_offset,
                        SINK_OFFSET=gqa_union_page1_sink_offset,
                        COARSE_OFFSET=gqa_union_page1_coarse_offset,
                        LOCAL_CAPACITY=int(local_k.size(2)),
                        SINK_CAPACITY=int(sink_k.size(2)) if include_sink else 0,
                        LOCAL_LIMIT=local_len,
                        SINK_LEN=int(sink_k.size(2)) if include_sink else 0,
                        HEAD_DIM=head_dim,
                        BLOCK_K=64,
                        INCLUDE_NEW=include_new,
                        USE_STATE_LENS=use_state_lens,
                        INCLUDE_COARSE=not gqa_union_compact_reuse_coarse,
                        num_warps=1,
                        waves_per_eu=waves_per_eu,
                    )
            timing_end("gqa_union_indices", union_begin)
            if gqa_union_hip_exact:
                hip_begin = timing_begin()
                from aiter.ops.triton._triton_kernels.attention.unified_attention import (
                    reduce_segments,
                )
                from lod_attention.kernels.aiter_page1_attention import (
                    kernel_page1_attention_3d_bias,
                )

                hip_launch_lens = buffers["gqa_union_hip_launch_lens"][:sequence_count]
                if gqa_union_aiter_final and (not gqa_union_staged_fixed):
                    hip_launch_lens = hip_context_lens
                aiter_query = q[:, :, 0, :].reshape(
                    sequence_count, kv_group_size, head_dim
                )
                unified_segments = int(buffers["gqa_union_hip_exp_sums"].size(2))
                aiter_out = (
                    output[:, :, 0, :].reshape(sequence_count, kv_group_size, head_dim)
                    if gqa_union_aiter_final
                    and (not gqa_union_staged_fixed)
                    and (not gqa_union_compact_reuse_coarse)
                    else buffers["gqa_union_hip_out"][:sequence_count]
                )
                aiter_exp_sums = buffers["gqa_union_hip_exp_sums"][:sequence_count]
                aiter_max_logits = buffers["gqa_union_hip_max_logits"][:sequence_count]
                aiter_segment_out = buffers["gqa_union_hip_segment_out"][
                    :sequence_count
                ]
                aiter_cu_q = buffers["gqa_union_hip_cu_q"][: sequence_count + 1]
                if gqa_union_aiter_final:
                    if not gqa_union_implicit_lod:
                        if not gqa_union_indirect_exact_pages:
                            kernel_page1_attention_3d_bias[
                                sequence_count, 1, unified_segments
                            ](
                                aiter_segment_out,
                                aiter_max_logits,
                                aiter_exp_sums,
                                aiter_query,
                                gqa_union_page1_k,
                                gqa_union_page1_v,
                                gqa_union_page1_bias,
                                hip_block_table,
                                cache_indices,
                                hip_launch_lens,
                                float(scale),
                                hip_block_table.stride(0),
                                aiter_query.stride(0),
                                aiter_query.stride(1),
                                NUM_QUERY_HEADS=kv_group_size,
                                KV_HEADS=kv_heads,
                                INDEX_BY_CACHE=False,
                                TILE_SIZE=64,
                                HEAD_SIZE=head_dim,
                                BLOCK_M=16,
                                NUM_SEGMENTS=unified_segments,
                                num_warps=2,
                                waves_per_eu=2,
                                num_stages=2,
                            )
                if not gqa_union_compact_reuse_coarse:
                    if not gqa_union_staged_fixed:
                        if not gqa_union_fused_reduce_advance:
                            reduce_segments[sequence_count, kv_group_size](
                                output_ptr=aiter_out,
                                segm_output_ptr=aiter_segment_out,
                                segm_max_ptr=aiter_max_logits,
                                segm_expsum_ptr=aiter_exp_sums,
                                seq_lens_ptr=hip_launch_lens,
                                num_seqs=sequence_count,
                                num_query_heads=kv_group_size,
                                out_scale_ptr=None,
                                output_stride_0=aiter_out.stride(0),
                                output_stride_1=aiter_out.stride(1),
                                block_table_stride=hip_block_table.stride(0),
                                TILE_SIZE=64,
                                HEAD_SIZE=head_dim,
                                HEAD_SIZE_PADDED=head_dim,
                                query_start_len_ptr=aiter_cu_q,
                                BLOCK_Q=1,
                                NUM_SEGMENTS_PER_SEQ=unified_segments,
                                num_warps=2,
                                waves_per_eu=2,
                                num_stages=1,
                            )
                timing_end("gqa_union_hip_exact_attention", hip_begin)
                if gqa_union_aiter_final:
                    timing_end("leaf_local", leaf_begin)
                    if (
                        include_new
                        and ragged_local_lens
                        and (not gqa_union_fused_reduce_advance)
                    ):
                        advance_decode_cache_lengths(cache_indices, local_lens)
                    apply_exact_flat_bf16_override()
                    return output
        elif not cooperative_leaf:
            decode_stripe_override = None
            if decode_stripe_override is None:
                stripe_route_leaves = speculative_steps <= 1 or "1" != "0"
            _split_decode_paged_lod_attention_kernel[batch * query_heads, split_kv](
                q,
                cache_indices,
                local_lens,
                state_k,
                state_v,
                counts,
                local_k,
                local_v,
                page_k,
                page_v,
                flat_page_indices if flat_page_indices is not None else page_k,
                flat_page_k_scales if flat_int8 else page_k,
                flat_page_v_scales if flat_int8 else page_v,
                slot_pages,
                overflow_page_keys,
                overflow_page_values,
                overflow_used,
                slot_lengths,
                top_slots,
                new_k,
                new_v,
                partial_out,
                partial_lse,
                buffers["route_top_scores"] if fuse_state_route else partial_lse,
                buffers["coarse_out"] if fuse_state_route else partial_out,
                buffers["coarse_lse"] if fuse_state_route else partial_lse,
                output,
                fused_completion,
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
                state_v.stride(0),
                state_v.stride(1),
                state_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                top_slots.stride(0),
                top_slots.stride(1),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                0 if fuse_state_route else state_len,
                local_len,
                QUERY_HEADS=query_heads,
                KV_HEADS=kv_heads,
                KV_GROUP_SIZE=kv_group_size,
                PAGE_CAPACITY=int(page_shape.size(2)),
                LEAF_CAPACITY=int(page_k.size(2))
                if flat_page_indices is not None
                else 1,
                STATE_CAPACITY=int(slot_pages.size(2)),
                INLINE_PAGES_PER_SLOT=int(slot_pages.size(3)),
                HASH_CAPACITY=int(overflow_page_values.size(2)),
                HASH_PROBES=hash_probes,
                HEAD_DIM=head_dim,
                VALUE_DIM=head_dim,
                PAGE_SIZE=int(page_shape.size(3)),
                ROUTE_COUNT=int(top_slots.size(-1)),
                SPLITS=split_kv,
                SCALE_LOG2=float(scale) * math.log2(math.e),
                BLOCK_N=block_n,
                USE_DOT=use_dot,
                INCLUDE_NEW=include_new,
                STORE_NEW=store_new_kv,
                LOCAL_LENS_LOGICAL=local_lens_are_logical,
                SEPARATE_LOCAL=route_fused_mtp_local
                or route_fused_decode_local
                or shared_mtp_local
                or (route_residual_mass is not None and reuse_residual_local_attention),
                FUSE_FINAL_REDUCE=fuse_state_route and effective_fuse_final_reduce,
                INDEXED=flat_page_indices is not None,
                INT8_STORAGE=flat_int8,
                STRIPE_ROUTE_LEAVES=stripe_route_leaves,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )
        timing_end("leaf_local", leaf_begin)
        if fuse_state_route and (not effective_fuse_final_reduce):
            final_reduce_begin = timing_begin()
            _reduce_routed_split_decode_lod_attention_kernel[batch * query_heads,](
                q,
                sink_k,
                sink_v,
                state_k,
                state_v,
                counts,
                cache_indices,
                local_lens,
                local_k,
                local_v,
                new_k,
                new_v,
                top_slots,
                buffers["route_top_scores"],
                buffers["coarse_out"],
                buffers["coarse_lse"],
                final_partial_out,
                final_partial_lse,
                buffers["speculative_local_partial_out"]
                if shared_mtp_local
                else buffers["route_local_out"]
                if reuse_residual_local_attention or cooperative_separate_local
                else partial_out,
                buffers["speculative_local_partial_lse"]
                if shared_mtp_local
                else buffers["route_local_lse"]
                if reuse_residual_local_attention or cooperative_separate_local
                else partial_lse,
                output,
                sink_k.stride(0),
                sink_k.stride(1),
                sink_k.stride(2),
                sink_v.stride(0),
                sink_v.stride(1),
                sink_v.stride(2),
                state_k.stride(0),
                state_k.stride(1),
                state_k.stride(2),
                state_v.stride(0),
                state_v.stride(1),
                state_v.stride(2),
                counts.stride(0),
                counts.stride(1),
                counts.stride(2),
                local_k.stride(0),
                local_k.stride(1),
                local_k.stride(2),
                local_v.stride(0),
                local_v.stride(1),
                local_v.stride(2),
                new_k.stride(0),
                new_k.stride(1),
                new_v.stride(0),
                new_v.stride(1),
                QUERY_HEADS=query_heads,
                KV_GROUP_SIZE=kv_group_size,
                HEAD_DIM=head_dim,
                STATE_CAPACITY=int(state_k.size(2)),
                ROUTE_COUNT=int(top_slots.size(-1)),
                SPLITS=final_splits,
                ROUTE_SPLITS=final_route_splits,
                INCLUDE_SEPARATE_LOCAL=shared_mtp_local
                or reuse_residual_local_attention
                or cooperative_separate_local,
                SEPARATE_LOCAL_SPLITS=split_kv if shared_mtp_local else 1,
                FUSE_LOCAL_SCAN=False,
                INCLUDE_NEW=False,
                INCLUDE_SINK=include_sink,
                SINK_LEN=int(sink_k.size(2)),
                LOCAL_BLOCK_N=32,
                SCALE=float(scale),
                USE_DOT=score_use_dot,
                ADVANCE_LOCAL=include_new and ragged_local_lens and advance_local_lens,
                SUBTRACT_ROUTES=not gqa_union_leaf,
                num_warps=final_reduce_num_warps,
                waves_per_eu=waves_per_eu,
            )
            timing_end("final_reduce", final_reduce_begin)
        apply_exact_flat_bf16_override()
        return output
