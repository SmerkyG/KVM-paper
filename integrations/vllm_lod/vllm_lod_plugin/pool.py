"""Fixed-address per-layer LOD pools used by captured vLLM decode graphs."""

from __future__ import annotations

import math
from typing import Any

import torch

from lod_attention.kernels.paged_leaf_attention import (
    advance_decode_cache_lengths,
    fused_decode_paged_lod_attention,
    materialize_page1_coarse_means,
    materialize_page1_fixed_indices,
    new_fused_decode_buffers,
    prepare_speculative_decode_kv,
    rehash_overflow_pages,
)
from lod_attention._config import (
    CHUNK_SIZE,
    LOCAL_WINDOW,
    PAGE_SIZE,
    PREFILL_CHUNK_SIZE,
    PREFILL_LOCAL_WINDOW,
    ROUTE_COUNT,
    LODConfig,
    ModelFamily,
    PagedLODConfig,
)
from lod_attention._engines import (
    KernelLODCache,
    KernelRecursivePagedLODAttention,
    KernelTwoLevelLODAttention,
)
from lod_attention._profile import configure_engine

from .config import VLLMLODSettings


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _power_of_two(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


def _recursive_state_route_backend(
    levels: int,
    head_dim: int,
    gqa: int,
    kv_heads: int,
    request_capacity: int,
) -> str:
    """Resolve the measured recursive coarse-routing implementation."""

    # Re-split has a nearly fixed launch floor, whereas the grouped producer
    # grows with the allocated state field. Keep Qwen's measured batch-eight
    # crossover; K2 remains on the grouped producer.
    crossover = {(256, 6, 4): 22_528}.get((head_dim, gqa, kv_heads))
    if levels == 3 and crossover is not None and request_capacity >= crossover:
        return "resplit"
    return "fused"


class VLLMLayerLODPool:
    """One layer's stable request rows and graph-captured decode scratch."""

    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        settings: VLLMLODSettings,
        max_requests: int,
        request_capacity: int,
        active_indices: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        has_query_norm: bool = False,
        has_key_norm: bool = False,
        prefix_rollback_tokens: int = 0,
        speculative_tokens: int = 0,
    ) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("vLLM LOD conversion requires a native FP16/BF16 KV cache")
        if request_capacity < LOCAL_WINDOW:
            raise ValueError("LOD request capacity is shorter than its local window")
        self.layer = layer
        self.max_requests = max_requests
        self.request_capacity = request_capacity
        self.active_indices = active_indices
        self.dtype = dtype
        self.device = device
        self.query_heads = int(layer.num_heads)
        self.kv_heads = int(layer.num_kv_heads)
        self.head_dim = int(layer.head_size)
        self.value_dim = int(layer.head_size_v)
        self.speculative_tokens = int(speculative_tokens)
        gqa = self.query_heads // self.kv_heads
        if self.value_dim != self.head_dim:
            raise NotImplementedError(
                "LOD vLLM currently requires equal K and V widths"
            )
        if (self.head_dim, gqa) == (256, 6):
            self.family = ModelFamily.QWEN38
        elif (self.head_dim, gqa) == (128, 8):
            self.family = ModelFamily.K2
        else:
            raise ValueError(
                "the LoD paper release supports only Qwen3.8 (D256/GQA6) "
                "and K2 Horizon (D128/GQA8)"
            )
        settings = settings.for_family(self.family)
        self.settings = settings

        state_normalization = "none" if has_key_norm else "cosine"
        centroid_rescale = "coherence" if has_key_norm else "none"
        routing_normalization = "none" if has_query_norm else "query"
        local_window = LOCAL_WINDOW
        if prefix_rollback_tokens:
            # Keep the scheduler's usual late-prefix rollback in the exact
            # field. This does not enlarge the two-level pool when the prefill
            # local allocation is already wider; it only avoids rebuilding
            # centroids for the common repeated-prompt hit. Older shared
            # prefixes remain supported by restore_prefix() below.
            local_window = max(local_window, prefix_rollback_tokens)
        config_type = PagedLODConfig if settings.levels == 3 else LODConfig
        recursive_state_route_backend = _recursive_state_route_backend(
            settings.levels,
            self.head_dim,
            gqa,
            self.kv_heads,
            request_capacity,
        )
        config_kwargs = dict(
            chunk_size=CHUNK_SIZE,
            local_window=local_window,
            state_growth_factor=16.0,
            state_min_size=256,
            protected_prefix=1,
            state_clustering_policy="manual",
            max_routes=ROUTE_COUNT,
            state_clustering_normalization=state_normalization,
            state_clustering_centroid_rescale=centroid_rescale,
            state_clustering_centroid_rescale_scope="assignment",
            routing_normalization=routing_normalization,
            leaf_paged_directory=settings.levels == 2,
        )
        if settings.levels == 3:
            config_kwargs.update(
                page_size=PAGE_SIZE,
                kv_bits=settings.kv_bits,
                quant_group_size=settings.quant_group_size,
                recursive_state_route_backend=recursive_state_route_backend,
                # The compatibility pool uses its fixed graph-safe overflow
                # hash rather than the flat two-tier directory allocation.
                leaf_paged_directory=False,
            )
        config = config_type(**config_kwargs)
        engine_type = (
            KernelTwoLevelLODAttention
            if settings.levels == 2
            else KernelRecursivePagedLODAttention
        )
        self.engine = engine_type(
            config,
            query_heads=self.query_heads,
            key_value_heads=self.kv_heads,
            scale=float(layer.impl.scale),
            default_open_count=ROUTE_COUNT,
        )
        self.engine.head_dim = self.head_dim
        configure_engine(
            self.engine,
            family=self.family,
            mode=settings.mode,
            request_capacity=request_capacity,
            has_query_norm=has_query_norm,
            has_key_norm=has_key_norm,
        )
        if self.speculative_tokens:
            # DFlash captures ordinary and flattened verifier graphs over one
            # target pool. Its graph warmup cannot safely mix the short-context
            # all-leaf scan with verifier buffers, so keep speculative decode
            # on its routed path in every cache organization.
            self.engine.exact_decode_limit = 0
        self._assert_production_profile(
            gqa,
            has_query_norm=has_query_norm,
            has_key_norm=has_key_norm,
        )
        self.state_capacity = self.engine._state_capacity(
            request_capacity, min(request_capacity, CHUNK_SIZE)
        )
        self.decode_local_capacity = local_window + int(
            self.engine.decode_state_update_len
        )
        self.decode_local_limit = (
            int(self.engine.local_len)
            - int(self.engine.chunk_len)
            + int(self.engine.decode_state_update_len)
        )
        # Exact-first prefill advances the persistent state to within the
        # decode-local tail before installing a row.  Its much wider local
        # field is temporary attention workspace, not persistent per-request
        # cache.  Reserving it for every pool row multiplies a 16K prefill
        # block by max_requests and can consume tens of GiB per rank.
        self.local_capacity = (
            self.decode_local_capacity
            if self.engine.prefill_exact_first_chunk
            else max(
                self.decode_local_capacity,
                int(self.engine.prefill_local_len),
            )
        )
        self.leaf_capacity = _round_up(request_capacity, CHUNK_SIZE) + max(
            CHUNK_SIZE, int(self.engine.decode_cache_headroom)
        )
        self.page_capacity = math.ceil(self.leaf_capacity / 16) + self.state_capacity
        self.hash_capacity = _power_of_two(
            self.page_capacity * int(self.engine.leaf_overflow_hash_factor)
        )
        self.state = self._allocate_state()
        self.local_lens = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )
        # Keep the allocation capacity fixed for CUDA graphs, but let decode
        # skip the unused suffix of each request's contiguous centroid state.
        self.state_lens = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )
        self.leaf_lens = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )
        self.ready = [False] * max_requests
        self.clean = [True] * max_requests
        self.metadata = [dict[str, int | bool]() for _ in range(max_requests)]
        self.decode_buffer_storage: dict[str, torch.Tensor] | None = None
        self.decode_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self.speculative_decode_buffers: dict[tuple[int, int], dict[str, Any]] = {}
        self.decode_enabled = False
        self.speculative_decode_steps = 0
        self.hybrid_full_decode = False
        self.direct_prefill_plan: tuple[tuple[int, int, int, int], ...] | None = None
        self.direct_prefill_prompt_lengths: dict[int, int] = {}
        self.deferred_prefill_stream: torch.cuda.Stream | None = None
        self.deferred_prefill_events: list[torch.cuda.Event | None] = [
            None
        ] * max_requests
        self.install_count = 0
        self.batched_install_calls = 0
        self.direct_prefill_calls = 0
        self.batched_cached_prefill_calls = 0
        self.batched_cached_prefill_rows = 0
        self.cached_prefill_packed_calls = 0
        self.cached_prefill_nonpacked_calls = 0
        self.cached_prefill_candidate_calls = 0
        self.cached_prefill_candidate_rows = 0
        self.cached_prefill_nonuniform_lengths = 0
        self.cached_prefill_nonuniform_previous = 0
        self.cached_prefill_unready = 0
        self.cached_prefill_noncontiguous = 0
        self.decode_calls = 0
        self.catch_up_batches = 0
        self.catch_up_rows = 0
        self.retained_reuse_count = 0
        self.retained_restore_attempts = 0
        self.retained_restore_fail_no_row = 0
        self.retained_restore_fail_short = 0
        self.retained_restore_fail_tokens = 0
        self.retained_restore_fail_coverage = 0
        self.retained_restore_rebuilds = 0
        self.retained_restore_rebuild_tokens = 0
        self.retained_restore_last_prefix = 0
        self.retained_restore_last_coverage = 0
        self.retained_restore_last_total = 0

    def _assert_production_profile(
        self,
        gqa: int,
        *,
        has_query_norm: bool,
        has_key_norm: bool,
    ) -> None:
        """Reject any silent deviation from the paper's supported path."""

        expected_geometry = (256, 6) if self.family is ModelFamily.QWEN38 else (128, 8)
        if self.dtype != torch.bfloat16:
            raise RuntimeError("LoD requires BF16 attention K/V inputs")
        if (self.head_dim, gqa) != expected_geometry:
            raise RuntimeError(
                f"{self.family.value} requires D/GQA={expected_geometry}, "
                f"got {(self.head_dim, gqa)}"
            )

        recursive = self.settings.levels == 3
        expected_bits = self.settings.kv_bits
        k2_int4 = self.family is ModelFamily.K2 and recursive and expected_bits == 4
        checks = {
            "top-four routing": (
                self.engine.two_level_topk == ROUTE_COUNT
                and self.engine.prefill_two_level_topk == ROUTE_COUNT
            ),
            "separate protected sink": self.engine.separate_sink_cache,
            "routing geometry": (
                self.engine.state_clustering_normalization
                == ("none" if has_key_norm else "cosine")
                and self.engine.state_clustering_centroid_rescale
                == ("coherence" if has_key_norm else "none")
                and self.engine.routing_normalization
                == ("none" if has_query_norm else "query")
            ),
            "cache tier": (
                self.engine.recursive_page_lod == recursive
                and self.engine.leaf_key_quant_bits == expected_bits
                and self.engine.leaf_value_quant_bits == expected_bits
            ),
            "prefill schedule": (
                self.engine.prefill_chunk_len == PREFILL_CHUNK_SIZE
                and self.engine.prefill_local_len == PREFILL_LOCAL_WINDOW
                and self.engine.prefill_state_update_len == PREFILL_CHUNK_SIZE
                and self.engine.prefill_exact_first_chunk
            ),
            "prefill kernels": (
                self.engine.prefill_local_attention_backend == "aiter"
                and self.engine.fused_prefill_route_coarse
                and self.engine.fused_prefill_stable_recompute
                and self.engine.fused_prefill_external_recompute
                and self.engine.prefill_hierarchical_route
                and self.engine.prefill_overlap_coarse_leaf
                and not self.engine.prefill_overlap_local_lod
            ),
            "GQA-aware AITER prefill route/coarse": (
                self.engine.prefill_aiter_route_coarse
            ),
            "complete-centroid prefill": (
                not recursive or self.engine.recursive_prefill_all_leaves
            ),
            "leaf geometry": (
                self.engine.leaf_layout == "expert"
                and self.engine.leaf_block_m
                == (64 if self.family is ModelFamily.K2 else 32)
                and self.engine.leaf_block_n == 16
                and self.engine.leaf_num_warps == (4 if k2_int4 else 2)
            ),
            "fused state update": (
                self.engine.fused_state_update and self.engine.fused_state_maxsim
            ),
            "K2 compact route": (
                self.family is not ModelFamily.K2
                or (
                    self.engine.decode_route_group_size == 64
                    and self.engine.decode_route_segment_tiles == 1
                    and self.engine.decode_route_num_warps == 1
                    and self.engine.decode_route_reduce_num_warps == 2
                )
            ),
        }
        failed = [name for name, valid in checks.items() if not valid]
        if failed:
            raise RuntimeError("LoD production dispatch failed: " + ", ".join(failed))

    def _allocate_state(self) -> dict[str, object]:
        r, h, s, d = (
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
            self.head_dim,
        )
        unified_page1 = (
            self.settings.levels == 2
            and self.dtype == torch.bfloat16
            and 1 < self.query_heads // self.kv_heads <= 16
            and self.query_heads % self.kv_heads == 0
            and self.head_dim in (128, 256)
        )
        sink_capacity = int(self.engine.separate_sink_cache)
        if unified_page1:
            arena_leaf_offset = 0
            kv_rows = r * h
            arena_local_offset = arena_leaf_offset + kv_rows * self.leaf_capacity
            arena_sink_offset = arena_local_offset + kv_rows * self.local_capacity
            arena_coarse_offset = arena_sink_offset + kv_rows * sink_capacity
            arena_padding_index = arena_coarse_offset + kv_rows * self.state_capacity
            arena_capacity = arena_padding_index + 1
            unified_page1_k = torch.empty(
                arena_capacity, d, dtype=self.dtype, device=self.device
            )
            unified_page1_v = torch.empty_like(unified_page1_k)
            # Leaves, local tokens, and sinks have no multiplicity bias. Coarse
            # refreshes overwrite only the centroid section with log(count).
            unified_page1_bias = torch.zeros(
                arena_capacity, dtype=torch.float16, device=self.device
            )
            unified_page1_k[arena_padding_index].zero_()
            unified_page1_v[arena_padding_index].zero_()
            unified_page1_bias[arena_padding_index] = -float("inf")
            recent_k = unified_page1_k[
                arena_local_offset : arena_local_offset + kv_rows * self.local_capacity
            ].view(r, h, self.local_capacity, d)
            recent_v = unified_page1_v[
                arena_local_offset : arena_local_offset + kv_rows * self.local_capacity
            ].view(r, h, self.local_capacity, d)
            fixed_mask_page1 = self.settings.decode_gqa_fixed_mask_aiter
            if fixed_mask_page1:
                fixed_capacity = (
                    self.leaf_capacity
                    + int(self.engine.local_len)
                    + sink_capacity
                    + self.state_capacity
                )
                # Graph capture exercises decode before a real prefill has
                # materialized the persistent list. Keep its one bootstrap
                # entry in-bounds; the first state refresh overwrites it.
                unified_page1_fixed_indices = torch.zeros(
                    r,
                    h,
                    fixed_capacity,
                    dtype=torch.int32,
                    device=self.device,
                )
                unified_page1_fixed_leaf_owners = torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    dtype=torch.int32,
                    device=self.device,
                )
                unified_page1_fixed_slot_offsets = torch.empty(
                    r,
                    h,
                    self.state_capacity + 1,
                    dtype=torch.int32,
                    device=self.device,
                )
                unified_page1_fixed_lengths = torch.zeros(
                    r, h, dtype=torch.int32, device=self.device
                )
            else:
                unified_page1_fixed_indices = None
                unified_page1_fixed_leaf_owners = None
                unified_page1_fixed_slot_offsets = None
                unified_page1_fixed_lengths = None
        else:
            arena_leaf_offset = 0
            arena_local_offset = 0
            arena_sink_offset = 0
            arena_coarse_offset = 0
            arena_capacity = 0
            unified_page1_k = None
            unified_page1_v = None
            unified_page1_bias = None
            unified_page1_fixed_indices = None
            unified_page1_fixed_leaf_owners = None
            unified_page1_fixed_slot_offsets = None
            unified_page1_fixed_lengths = None
            recent_k = torch.empty(
                r, h, self.local_capacity, d, dtype=self.dtype, device=self.device
            )
            recent_v = torch.empty_like(recent_k)
        state: dict[str, object] = {
            "state_k": torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device),
            "state_v": torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device),
            "counts": torch.zeros(r, h, s, 1, dtype=torch.float32, device=self.device),
            "state_len": s,
            "coverage": 0,
            "state_capacity": s,
            "recent_k": recent_k,
            "recent_v": recent_v,
            "recent_len": 0,
            "total_len": 0,
        }
        if self.engine.separate_sink_cache:
            if unified_page1:
                state["sink_k"] = unified_page1_k[
                    arena_sink_offset : arena_sink_offset + r * h * sink_capacity
                ].view(r, h, sink_capacity, d)
                state["sink_v"] = unified_page1_v[
                    arena_sink_offset : arena_sink_offset + r * h * sink_capacity
                ].view(r, h, sink_capacity, d)
            else:
                state["sink_k"] = torch.empty(
                    r,
                    h,
                    1,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                )
                state["sink_v"] = torch.empty_like(state["sink_k"])
        if self.engine.state_clustering_centroid_rescale != "none":
            state["key_norm_sums"] = torch.zeros(
                r, h, s, 1, dtype=torch.float32, device=self.device
            )

        if self.settings.levels == 2:
            page_size = 16
            maximum_slot_pages = max(1, math.ceil(self.leaf_capacity / page_size))
            root_capacity = max(1, math.ceil(maximum_slot_pages / 64))
            slot_pages = torch.full(
                (r, h, s, root_capacity),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            overflow_page_keys = torch.full(
                (r, h, 1), -1, dtype=torch.int32, device=self.device
            )
            overflow_page_values = torch.full(
                (r, h, self.page_capacity, 64),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            overflow_active = False
            overflow_safe_until = root_capacity * 64 * page_size
            if unified_page1:
                leaf_k = unified_page1_k[
                    arena_leaf_offset : arena_leaf_offset + r * h * self.leaf_capacity
                ].view(r, h, self.leaf_capacity, d)
                leaf_v = unified_page1_v[
                    arena_leaf_offset : arena_leaf_offset + r * h * self.leaf_capacity
                ].view(r, h, self.leaf_capacity, d)
            else:
                leaf_k = torch.zeros(
                    r,
                    h,
                    self.leaf_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                )
                leaf_v = torch.zeros_like(leaf_k)
            state["page_cache"] = {
                "region_owned_pages": True,
                "dense_leaf_storage": True,
                "slot_pages": slot_pages,
                "overflow_page_keys": overflow_page_keys,
                "overflow_page_values": overflow_page_values,
                "overflow_hash_capacity": self.hash_capacity,
                "overflow_flag": torch.zeros((), dtype=torch.int32, device=self.device),
                "overflow_used": torch.zeros((), dtype=torch.int32, device=self.device),
                "overflow_active": overflow_active,
                "overflow_safe_until": overflow_safe_until,
                "paged_page_directory": True,
                "page_directory_size": 64,
                "slot_lengths": torch.zeros(
                    r, h, s, dtype=torch.int32, device=self.device
                ),
                "next_page": torch.zeros(r, h, dtype=torch.int32, device=self.device),
                "page_size": page_size,
                "leaf_capacity": self.leaf_capacity,
                "leaf_count": 0,
                "page_indices": torch.full(
                    (r, h, self.page_capacity, page_size),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "leaf_k": leaf_k,
                "leaf_v": leaf_v,
                "quantization_finalized": False,
                "summary_quantization_finalized": False,
            }
            if unified_page1:
                state["page_cache"].update(
                    unified_page1_k=unified_page1_k,
                    unified_page1_v=unified_page1_v,
                    unified_page1_bias=unified_page1_bias,
                    unified_page1_capacity=arena_capacity,
                    unified_page1_leaf_offset=arena_leaf_offset,
                    unified_page1_local_offset=arena_local_offset,
                    unified_page1_sink_offset=arena_sink_offset,
                    unified_page1_coarse_offset=arena_coarse_offset,
                    unified_page1_padding_index=arena_padding_index,
                )
                if isinstance(unified_page1_fixed_indices, torch.Tensor):
                    state["page_cache"].update(
                        unified_page1_fixed_indices=(unified_page1_fixed_indices),
                        unified_page1_fixed_leaf_owners=(
                            unified_page1_fixed_leaf_owners
                        ),
                        unified_page1_fixed_slot_offsets=(
                            unified_page1_fixed_slot_offsets
                        ),
                        unified_page1_fixed_lengths=(unified_page1_fixed_lengths),
                    )
            return state

        slot_dtype = (
            torch.int16
            if self.page_capacity <= torch.iinfo(torch.int16).max
            else torch.int32
        )
        page: dict[str, object] = {
            "region_owned_pages": True,
            "slot_pages": torch.full(
                (r, h, s, int(self.engine.leaf_inline_pages_per_slot)),
                -1,
                dtype=slot_dtype,
                device=self.device,
            ),
            "overflow_page_keys": torch.full(
                (r, h, self.hash_capacity),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "overflow_page_values": torch.full(
                (r, h, self.hash_capacity),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "overflow_hash_capacity": self.hash_capacity,
            "overflow_flag": torch.zeros((), dtype=torch.int32, device=self.device),
            "overflow_used": torch.zeros((), dtype=torch.int32, device=self.device),
            # A fixed pool cannot specialize graph kernels per request. Always
            # enable the bounded hash lookup; rows without overflow simply miss.
            "overflow_active": True,
            "overflow_safe_until": 0,
            "slot_lengths": torch.zeros(r, h, s, dtype=torch.int32, device=self.device),
            "next_page": torch.zeros(r, h, dtype=torch.int32, device=self.device),
            "page_size": 16,
            "leaf_capacity": self.leaf_capacity,
            "leaf_count": 0,
            "page_indices": torch.full(
                (r, h, self.page_capacity, 16),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
            "page_counts": torch.zeros(
                r, h, self.page_capacity, dtype=torch.int32, device=self.device
            ),
        }
        groups = d // self.settings.quant_group_size
        token_groups = 16 // 16
        if self.settings.kv_bits == 4:
            quant_bits = 4
            quant_width = d // 2
            quant_dtype = torch.uint8
            page.update(
                leaf_quant_bits=quant_bits,
                leaf_k=torch.empty(r, h, 1, d, dtype=self.dtype, device=self.device),
                leaf_v=torch.empty(r, h, 1, d, dtype=self.dtype, device=self.device),
                quantized_leaf_k=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    quant_width,
                    dtype=quant_dtype,
                    device=self.device,
                ),
                quantized_leaf_v=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    quant_width,
                    dtype=quant_dtype,
                    device=self.device,
                ),
                page_k_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    token_groups * groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_v_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    token_groups * groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_quantized_counts=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    dtype=torch.int32,
                    device=self.device,
                ),
                page_sum_k=torch.empty(
                    r, h, 1, d, dtype=self.dtype, device=self.device
                ),
                page_sum_v=torch.empty(
                    r, h, 1, d, dtype=self.dtype, device=self.device
                ),
                quantized_page_sum_k=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=torch.int8,
                    device=self.device,
                ),
                quantized_page_sum_v=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=torch.int8,
                    device=self.device,
                ),
                page_sum_k_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_sum_v_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                quantization_finalized=True,
                summary_quantization_finalized=True,
            )
        else:
            page.update(
                leaf_quant_bits=0,
                leaf_k=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                leaf_v=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_sum_k=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_sum_v=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                ),
                quantization_finalized=False,
                summary_quantization_finalized=False,
            )
        state["page_cache"] = page
        return state

    def reset(self, slot: int) -> None:
        if not 0 <= slot < self.max_requests:
            raise IndexError("vLLM request slot is outside the LOD pool")
        self.wait_deferred_prefill((slot,))
        self.ready[slot] = False
        self.clean[slot] = True
        self.metadata[slot].clear()
        self.local_lens[slot].zero_()
        self.state_lens[slot].zero_()
        self.leaf_lens[slot].zero_()
        self.state["counts"][slot].zero_()
        if "sink_k" in self.state:
            self.state["sink_k"][slot].zero_()
            self.state["sink_v"][slot].zero_()
        if "key_norm_sums" in self.state:
            self.state["key_norm_sums"][slot].zero_()
        page = self.state["page_cache"]
        page["slot_pages"][slot].fill_(-1)
        page["slot_lengths"][slot].zero_()
        page["next_page"][slot].zero_()
        if "page_indices" in page:
            page["page_indices"][slot].fill_(-1)
        if "page_counts" in page:
            page["page_counts"][slot].zero_()
        page["overflow_page_keys"][slot].fill_(-1)
        page["overflow_page_values"][slot].fill_(-1)
        if "page_quantized_counts" in page:
            page["page_quantized_counts"][slot].zero_()
        if "decode_previous_total_lse" in page:
            page["decode_previous_total_lse"][slot].fill_(float("inf"))

    def _reset_range(self, start: int, stop: int) -> None:
        """Reset one contiguous row range with one launch per cache field."""
        if not 0 <= start < stop <= self.max_requests:
            raise IndexError("vLLM request row range is outside the LOD pool")
        self.wait_deferred_prefill(tuple(range(start, stop)))
        for slot in range(start, stop):
            self.ready[slot] = False
            self.clean[slot] = True
            self.metadata[slot].clear()
        self.local_lens[start:stop].zero_()
        self.state_lens[start:stop].zero_()
        self.leaf_lens[start:stop].zero_()
        self.state["counts"][start:stop].zero_()
        if "sink_k" in self.state:
            self.state["sink_k"][start:stop].zero_()
            self.state["sink_v"][start:stop].zero_()
        if "key_norm_sums" in self.state:
            self.state["key_norm_sums"][start:stop].zero_()
        page = self.state["page_cache"]
        page["slot_pages"][start:stop].fill_(-1)
        page["slot_lengths"][start:stop].zero_()
        page["next_page"][start:stop].zero_()
        if "page_indices" in page:
            page["page_indices"][start:stop].fill_(-1)
        if "page_counts" in page:
            page["page_counts"][start:stop].zero_()
        page["overflow_page_keys"][start:stop].fill_(-1)
        page["overflow_page_values"][start:stop].fill_(-1)
        if "page_quantized_counts" in page:
            page["page_quantized_counts"][start:stop].zero_()
        if "decode_previous_total_lse" in page:
            page["decode_previous_total_lse"][start:stop].fill_(float("inf"))

    def truncate_recent(self, slot: int, total_length: int) -> None:
        """Roll a retained cache back inside its unclustered exact tail."""
        if not self.ready[slot]:
            raise RuntimeError("cannot truncate an uninitialized LOD cache")
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        old_total = int(metadata["total_len"])
        if not coverage <= total_length <= old_total:
            raise ValueError(
                "retained LOD prefix lies outside the exact recent tail: "
                f"coverage={coverage}, requested={total_length}, total={old_total}"
            )
        recent_len = total_length - coverage
        self.local_lens[slot].fill_(recent_len)
        metadata["recent_len"] = recent_len
        metadata["total_len"] = total_length

    def restore_prefix(self, slot: int, total_length: int) -> None:
        """Restore any retained prefix without consulting native model K/V.

        A prefix inside the live exact tail is a metadata-only rollback.  A
        more distant shared prefix cannot undo centroid updates, but two-level
        LOD already archives every underlying leaf in chronological order.  In
        that case rebuild the much smaller semantic cache from those owned
        leaves.  This preserves general vLLM prefix caching (not just repeated
        whole prompts) while the model's chronological native K/V stays absent.
        """
        if not self.ready[slot]:
            raise RuntimeError("cannot restore an uninitialized LOD cache")
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        old_total = int(metadata["total_len"])
        if not 0 < total_length <= old_total:
            raise ValueError(
                "retained LOD prefix lies outside the cached request: "
                f"requested={total_length}, total={old_total}"
            )
        if coverage <= total_length:
            self.truncate_recent(slot, total_length)
            return
        if self.settings.levels != 2:
            raise NotImplementedError(
                "distant authoritative prefix restoration currently requires "
                "two-level chronological dense-leaf storage"
            )
        if self.engine.state_clustering_query_metric != "none":
            raise NotImplementedError(
                "query-conditioned routing cannot rebuild a distant cached prefix"
            )
        if self.engine.mla_key_norm_weight is not None:
            raise NotImplementedError(
                "MLA distant-prefix restoration requires archived raw latents"
            )

        page = self.state["page_cache"]
        leaf_k = page.get("leaf_k")
        leaf_v = page.get("leaf_v")
        if not isinstance(leaf_k, torch.Tensor) or not isinstance(leaf_v, torch.Tensor):
            raise RuntimeError("retained LOD row has no chronological leaf archive")
        leaf_count = int(metadata["leaf_count"])
        if leaf_count < total_length:
            raise RuntimeError(
                "retained LOD leaf archive was sealed before the requested prefix: "
                f"leaves={leaf_count}, requested={total_length}"
            )

        key = leaf_k[slot : slot + 1, :, :total_length, :]
        value = leaf_v[slot : slot + 1, :, :total_length, :]
        if key.dtype not in (torch.float16, torch.bfloat16) or value.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise TypeError("two-tier retained leaves must use FP16 or BF16")
        # The converted cache may retain its flat source as virtual leaf
        # storage. Clone before install() resets the old pool row.
        key = key.clone()
        value = value.clone()
        converted = self.engine.build_cache_from_bf16(
            key.contiguous(), value.contiguous()
        )
        self.install(slot, converted)
        self.engine.reset_runtime_cache()
        self.retained_restore_rebuilds += 1
        self.retained_restore_rebuild_tokens += total_length

    @staticmethod
    def _copy_row(
        destination: torch.Tensor,
        source: torch.Tensor,
        slot: int,
        source_slot: int = 0,
    ) -> None:
        source = source[source_slot : source_slot + 1]
        target = destination[slot : slot + 1]
        if source.ndim != target.ndim:
            raise ValueError("converted LOD tensor rank differs from its pool")
        if (
            source.data_ptr() == target.data_ptr()
            and source.shape == target.shape
            and source.stride() == target.stride()
        ):
            return
        slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
        target[slices].copy_(source[slices])

    def _validate_recent_capacity(self, source: dict[str, object]) -> None:
        recent_len = int(source["recent_len"])
        expected_recent_len = int(source["total_len"]) - int(source["coverage"])
        if recent_len != expected_recent_len:
            raise ValueError(
                "LOD exact-tail metadata drifted: "
                f"recent={recent_len}, expected={expected_recent_len}"
            )
        recent_k = source.get("recent_k")
        recent_v = source.get("recent_v")
        if not isinstance(recent_k, torch.Tensor) or not isinstance(
            recent_v, torch.Tensor
        ):
            raise TypeError("LOD cache lacks recent K/V tensors")
        if (
            recent_len > self.local_capacity
            or recent_len > int(recent_k.size(2))
            or recent_len > int(recent_v.size(2))
        ):
            raise ValueError(
                "LOD exact tail exceeds its fixed cache row: "
                f"recent={recent_len}, capacity={self.local_capacity}"
            )

    @staticmethod
    def _copy_range(
        destination: torch.Tensor,
        source: torch.Tensor,
        start: int,
        stop: int,
    ) -> None:
        target = destination[start:stop]
        if source.ndim != target.ndim or int(source.size(0)) != stop - start:
            raise ValueError("converted LOD tensor batch differs from its pool range")
        if (
            source.data_ptr() == target.data_ptr()
            and source.shape == target.shape
            and source.stride() == target.stride()
        ):
            return
        slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
        target[slices].copy_(source[slices])

    @staticmethod
    def _copy_rows(
        destination: torch.Tensor,
        source: torch.Tensor,
        slots: tuple[int, ...],
        indices: torch.Tensor,
    ) -> None:
        if source.ndim != destination.ndim or int(source.size(0)) != len(slots):
            raise ValueError("converted LOD tensor batch differs from its pool rows")
        slices = (slice(None),) + tuple(
            slice(0, min(int(destination.size(axis)), int(source.size(axis))))
            for axis in range(1, source.ndim)
        )
        destination[slices].index_copy_(
            0,
            indices,
            source[slices],
        )

    def _refresh_unified_page1_coarse(self, slots: tuple[int, ...]) -> None:
        """Materialize centroid means in the persistent AITER K/V arena.

        State updates retain sums because routing and archival mutate them in
        that representation.  Decode attention consumes means, so refresh the
        arena only at install/catch-up boundaries rather than dividing every
        centroid in the hot decode path.
        """
        if not slots:
            return
        page = self.state.get("page_cache")
        if not isinstance(page, dict):
            return
        arena_k = page.get("unified_page1_k")
        arena_v = page.get("unified_page1_v")
        arena_bias = page.get("unified_page1_bias")
        coarse_offset = page.get("unified_page1_coarse_offset")
        if (
            not isinstance(arena_k, torch.Tensor)
            or not isinstance(arena_v, torch.Tensor)
            or not isinstance(arena_bias, torch.Tensor)
        ):
            return
        if not isinstance(coarse_offset, int):
            raise TypeError("unified page-size-one coarse offset is invalid")
        coarse_k = arena_k[
            coarse_offset : coarse_offset
            + self.max_requests * self.kv_heads * self.state_capacity
        ].view(
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
            self.head_dim,
        )
        coarse_v = arena_v[
            coarse_offset : coarse_offset
            + self.max_requests * self.kv_heads * self.state_capacity
        ].view_as(coarse_k)
        coarse_bias = arena_bias[
            coarse_offset : coarse_offset
            + self.max_requests * self.kv_heads * self.state_capacity
        ].view(
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
        )
        ordered = tuple(sorted(slots))
        begin = 0
        while begin < len(ordered):
            end = begin + 1
            while end < len(ordered) and ordered[end] == ordered[end - 1] + 1:
                end += 1
            start_slot = ordered[begin]
            stop_slot = ordered[end - 1] + 1
            materialize_page1_coarse_means(
                self.state["state_k"][start_slot:stop_slot],
                self.state["state_v"][start_slot:stop_slot],
                self.state["counts"][start_slot:stop_slot],
                coarse_k[start_slot:stop_slot],
                coarse_v[start_slot:stop_slot],
                coarse_bias[start_slot:stop_slot],
            )
            begin = end
        self._refresh_unified_page1_fixed(slots)

    def _refresh_unified_page1_fixed(self, slots: tuple[int, ...]) -> None:
        """Rebuild Qwen's persistent page-size-one list after an update."""
        if not slots:
            return
        page = self.state.get("page_cache")
        if not isinstance(page, dict):
            return
        fixed_indices = page.get("unified_page1_fixed_indices")
        fixed_leaf_owners = page.get("unified_page1_fixed_leaf_owners")
        fixed_slot_offsets = page.get("unified_page1_fixed_slot_offsets")
        fixed_lengths = page.get("unified_page1_fixed_lengths")
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (
                fixed_indices,
                fixed_leaf_owners,
                fixed_slot_offsets,
                fixed_lengths,
            )
        ):
            return
        sink = self.state.get("sink_k")
        sink_len = int(sink.size(2)) if isinstance(sink, torch.Tensor) else 0
        ordered = tuple(sorted(slots))
        begin = 0
        while begin < len(ordered):
            end = begin + 1
            while end < len(ordered) and ordered[end] == ordered[end - 1] + 1:
                end += 1
            start_slot = ordered[begin]
            stop_slot = ordered[end - 1] + 1
            materialize_page1_fixed_indices(
                page["page_indices"][start_slot:stop_slot],
                page["slot_pages"][start_slot:stop_slot],
                page["overflow_page_keys"][start_slot:stop_slot],
                page["overflow_page_values"][start_slot:stop_slot],
                page["overflow_used"],
                page["slot_lengths"][start_slot:stop_slot],
                fixed_indices[start_slot:stop_slot],
                fixed_leaf_owners[start_slot:stop_slot],
                fixed_slot_offsets[start_slot:stop_slot],
                fixed_lengths[start_slot:stop_slot],
                row_offset=start_slot,
                arena_leaf_offset=int(page["unified_page1_leaf_offset"]),
                arena_local_offset=int(page["unified_page1_local_offset"]),
                arena_sink_offset=int(page["unified_page1_sink_offset"]),
                arena_coarse_offset=int(page["unified_page1_coarse_offset"]),
                local_capacity=self.local_capacity,
                local_limit=int(self.engine.local_len),
                sink_capacity=sink_len,
                sink_len=sink_len,
                hash_probes=int(self.engine._page_lookup_probes(page)),
            )
            begin = end

    def install_range(self, start: int, stop: int, converted: KernelLODCache) -> None:
        self.install_rows(tuple(range(start, stop)), converted)

    def install_rows(self, slots: tuple[int, ...], converted: KernelLODCache) -> None:
        """Install one converted batch into a contiguous set of pool rows."""
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("converted LOD row indices must be nonempty and unique")
        start, stop = min(slots), max(slots) + 1
        if tuple(sorted(slots)) != tuple(range(start, stop)):
            raise ValueError("batched LOD installation requires a contiguous row set")
        source = converted.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("converted LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("converted prefix exceeds VLLM_LOD_MAX_CONTEXT")
        if int(source["state_k"].size(0)) != len(slots):
            raise ValueError("converted LOD batch differs from its pool row range")
        self._validate_recent_capacity(source)
        self.batched_install_calls += 1
        pool_backed = bool(source.get("pool_backed", False))
        if pool_backed and slots != tuple(range(start, stop)):
            raise ValueError("pool-backed LOD installation requires ascending rows")
        if not pool_backed and not all(self.clean[slot] for slot in slots):
            self._reset_range(start, stop)
        ascending = slots == tuple(range(start, stop))
        slot_indices = (
            None
            if ascending
            else torch.tensor(slots, dtype=torch.long, device=self.device)
        )

        def copy(destination: torch.Tensor, value: torch.Tensor) -> None:
            if ascending:
                self._copy_range(destination, value, start, stop)
            else:
                if slot_indices is None:
                    raise AssertionError("permuted LOD row indices are missing")
                self._copy_rows(destination, value, slots, slot_indices)

        tensor_names = ["state_k", "state_v", "counts", "recent_k", "recent_v"]
        if "sink_k" in source:
            tensor_names.extend(("sink_k", "sink_v"))
        if "key_norm_sums" in source:
            tensor_names.append("key_norm_sums")
        if pool_backed:
            for name in tensor_names:
                destination = self.state[name][start:stop]
                value = source[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.data_ptr() != destination.data_ptr()
                    or tuple(value.shape) != tuple(destination.shape)
                ):
                    raise RuntimeError(
                        f"pool-backed LOD state tensor {name} does not alias its rows"
                    )
        else:
            for name in tensor_names:
                copy(self.state[name], source[name])

        destination_page = self.state["page_cache"]
        if pool_backed:
            for name, value in source_page.items():
                if name == "leaf_lens" or name.startswith("unified_page1_"):
                    continue
                destination = destination_page.get(name)
                if not isinstance(value, torch.Tensor) or not value.ndim:
                    continue
                if not isinstance(destination, torch.Tensor):
                    raise RuntimeError(
                        f"pool-backed LOD page tensor {name} has no destination"
                    )
                destination_rows = destination[start:stop]
                if value.data_ptr() != destination_rows.data_ptr() or tuple(
                    value.shape
                ) != tuple(destination_rows.shape):
                    raise RuntimeError(
                        f"pool-backed LOD page tensor {name} does not alias its rows"
                    )
        else:
            for name, value in source_page.items():
                if name in ("overflow_page_keys", "overflow_page_values") or (
                    name.startswith("unified_page1_")
                ):
                    continue
                destination = destination_page.get(name)
                if (
                    isinstance(value, torch.Tensor)
                    and isinstance(destination, torch.Tensor)
                    and value.ndim
                ):
                    copy(destination, value)
        source_keys = source_page["overflow_page_keys"]
        source_values = source_page["overflow_page_values"]
        destination_keys = destination_page["overflow_page_keys"]
        destination_values = destination_page["overflow_page_values"]
        if pool_backed:
            if (
                source_keys.data_ptr() != destination_keys[start:stop].data_ptr()
                or source_values.data_ptr() != destination_values[start:stop].data_ptr()
            ):
                raise RuntimeError("pool-backed LOD overflow table does not alias rows")
        elif int(source_keys.size(2)) == int(destination_keys.size(2)):
            copy(destination_keys, source_keys)
            copy(destination_values, source_values)
            destination_page["overflow_used"].logical_or_(source_page["overflow_used"])
        else:
            for source_slot, destination_slot in enumerate(slots):
                rehash_overflow_pages(
                    source_keys,
                    source_values,
                    destination_keys,
                    destination_values,
                    destination_page["overflow_used"],
                    destination_page["overflow_flag"],
                    source_slot=source_slot,
                    destination_slot=destination_slot,
                )
        if not pool_backed:
            destination_page["overflow_flag"].logical_or_(source_page["overflow_flag"])
        recent_len = int(source["recent_len"])
        state_len = int(source["state_len"])
        if ascending:
            self.local_lens[start:stop].fill_(recent_len)
            self.state_lens[start:stop].fill_(state_len)
            self.leaf_lens[start:stop].fill_(int(source_page["leaf_count"]))
        else:
            if slot_indices is None:
                raise AssertionError("permuted LOD row indices are missing")
            self.local_lens.index_fill_(0, slot_indices, recent_len)
            self.state_lens.index_fill_(0, slot_indices, state_len)
            self.leaf_lens.index_fill_(
                0, slot_indices, int(source_page["leaf_count"])
            )
        for slot in slots:
            self.metadata[slot].update(
                state_len=state_len,
                scheduled_state_len=int(
                    source.get("scheduled_state_len", source["state_len"])
                ),
                coverage=int(source["coverage"]),
                total_len=int(source["total_len"]),
                recent_len=recent_len,
                leaf_count=int(source_page["leaf_count"]),
                overflow_safe_until=int(source_page["overflow_safe_until"]),
            )
            self.ready[slot] = True
            self.clean[slot] = False
            self.install_count += 1
        self._refresh_unified_page1_coarse(slots)

    def _initial_prefill_storage(
        self, slots: tuple[int, ...]
    ) -> dict[str, object] | None:
        """Return authoritative row views for allocation-free initial prefill."""
        if not slots or slots != tuple(range(slots[0], slots[0] + len(slots))):
            return None
        if not (
            self.settings.levels in (2, 3)
            and self.settings.kv_bits in (0, 4)
            and self.engine.virtual_page_storage
            and self.engine.leaf_key_quant_bits == self.settings.kv_bits
            and self.engine.leaf_value_quant_bits == self.settings.kv_bits
            and (self.settings.kv_bits == 0 or self.engine.page_summary_quant_bits == 8)
        ):
            return None
        start, stop = slots[0], slots[-1] + 1
        if not all(self.clean[slot] for slot in slots):
            self._reset_range(start, stop)
        storage: dict[str, object] = {
            name: self.state[name][start:stop]
            for name in ("state_k", "state_v", "counts", "recent_k", "recent_v")
        }
        if "sink_k" in self.state:
            storage["sink_k"] = self.state["sink_k"][start:stop]
            storage["sink_v"] = self.state["sink_v"][start:stop]
        if "key_norm_sums" in self.state:
            storage["key_norm_sums"] = self.state["key_norm_sums"][start:stop]
        page_pool = self.state["page_cache"]
        page: dict[str, object] = {}
        for name, value in page_pool.items():
            if name.startswith("unified_page1_"):
                continue
            page[name] = (
                value[start:stop]
                if isinstance(value, torch.Tensor) and value.ndim
                else value
            )
        storage["page_cache"] = page
        return storage

    def wait_deferred_prefill(self, slots: tuple[int, ...]) -> None:
        """Make the foreground stream consume any deferred cache builds."""
        if not any(self.deferred_prefill_events[slot] is not None for slot in slots):
            return
        seen: set[int] = set()
        for slot in slots:
            event = self.deferred_prefill_events[slot]
            if event is None:
                continue
            identity = id(event)
            if identity not in seen:
                # vLLM can launch the next attention graph on a stream other
                # than PyTorch's current stream. A current-stream wait did not
                # order that consumer and produced corrupted generations.
                event.synchronize()
                seen.add(identity)
            self.deferred_prefill_events[slot] = None

    def install(
        self, slot: int, converted: KernelLODCache, *, source_slot: int = 0
    ) -> None:
        source = converted.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("converted LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("converted prefix exceeds VLLM_LOD_MAX_CONTEXT")
        self._validate_recent_capacity(source)
        self.reset(slot)
        tensor_names = ["state_k", "state_v", "counts", "recent_k", "recent_v"]
        if "sink_k" in source:
            tensor_names.extend(("sink_k", "sink_v"))
        for name in tensor_names:
            self._copy_row(
                self.state[name], source[name], slot, source_slot=source_slot
            )
        if "key_norm_sums" in source:
            self._copy_row(
                self.state["key_norm_sums"],
                source["key_norm_sums"],
                slot,
                source_slot=source_slot,
            )

        destination_page = self.state["page_cache"]
        for name, value in source_page.items():
            if name in ("overflow_page_keys", "overflow_page_values") or (
                name.startswith("unified_page1_")
            ):
                continue
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                self._copy_row(destination, value, slot, source_slot=source_slot)
        source_keys = source_page["overflow_page_keys"]
        source_values = source_page["overflow_page_values"]
        destination_keys = destination_page["overflow_page_keys"]
        destination_values = destination_page["overflow_page_values"]
        if int(source_keys.size(2)) == int(destination_keys.size(2)):
            self._copy_row(destination_keys, source_keys, slot, source_slot=source_slot)
            self._copy_row(
                destination_values, source_values, slot, source_slot=source_slot
            )
            destination_page["overflow_used"].logical_or_(source_page["overflow_used"])
        else:
            rehash_overflow_pages(
                source_keys,
                source_values,
                destination_keys,
                destination_values,
                destination_page["overflow_used"],
                destination_page["overflow_flag"],
                source_slot=source_slot,
                destination_slot=slot,
            )
        destination_page["overflow_flag"].logical_or_(source_page["overflow_flag"])
        recent_len = int(source["recent_len"])
        state_len = int(source["state_len"])
        self.local_lens[slot].fill_(recent_len)
        self.state_lens[slot].fill_(state_len)
        self.leaf_lens[slot].fill_(int(source_page["leaf_count"]))
        self.metadata[slot].update(
            state_len=state_len,
            scheduled_state_len=int(
                source.get("scheduled_state_len", source["state_len"])
            ),
            coverage=int(source["coverage"]),
            total_len=int(source["total_len"]),
            recent_len=recent_len,
            leaf_count=int(source_page["leaf_count"]),
            overflow_safe_until=int(source_page["overflow_safe_until"]),
        )
        self.ready[slot] = True
        self.clean[slot] = False
        self.install_count += 1
        self._refresh_unified_page1_coarse((slot,))

    def _synchronize_rows(self, slots: tuple[int, ...], cache: KernelLODCache) -> None:
        """Persist an equal-metadata batch after cached prefill."""
        if not slots:
            return
        source = cache.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("updated LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("updated prefix exceeds VLLM_LOD_MAX_CONTEXT")
        if int(source["state_k"].size(0)) != len(slots):
            raise ValueError("updated LOD batch does not match its pool rows")
        self._validate_recent_capacity(source)
        tensor_names = ["state_k", "state_v", "counts", "recent_k", "recent_v"]
        if "sink_k" in source:
            tensor_names.extend(("sink_k", "sink_v"))
        for name in tensor_names:
            for source_slot, slot in enumerate(slots):
                self._copy_row(
                    self.state[name], source[name], slot, source_slot=source_slot
                )
        if "key_norm_sums" in source:
            for source_slot, slot in enumerate(slots):
                self._copy_row(
                    self.state["key_norm_sums"],
                    source["key_norm_sums"],
                    slot,
                    source_slot=source_slot,
                )
        destination_page = self.state["page_cache"]
        for name, value in source_page.items():
            if name.startswith("unified_page1_"):
                continue
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                for source_slot, slot in enumerate(slots):
                    self._copy_row(destination, value, slot, source_slot=source_slot)
        recent_len = int(source["recent_len"])
        state_len = int(source["state_len"])
        for slot in slots:
            self.local_lens[slot].fill_(recent_len)
            self.state_lens[slot].fill_(state_len)
            self.leaf_lens[slot].fill_(int(source_page["leaf_count"]))
            self.metadata[slot].update(
                state_len=state_len,
                scheduled_state_len=int(
                    source.get("scheduled_state_len", source["state_len"])
                ),
                coverage=int(source["coverage"]),
                total_len=int(source["total_len"]),
                recent_len=recent_len,
                leaf_count=int(source_page["leaf_count"]),
                overflow_safe_until=int(source_page["overflow_safe_until"]),
            )
            self.ready[slot] = True
        self._refresh_unified_page1_coarse(slots)

    def _direct_cached_prefill_group(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        plan: tuple[tuple[int, int, int, int], ...],
        *,
        finalize_cache_for_decode: bool,
    ) -> None:
        """Advance one contiguous equal-length/equal-history cache group."""
        length = plan[0][2] - plan[0][1]
        previous_length = plan[0][3]
        slots = tuple(slot for slot, _, _, _ in plan)
        if slots != tuple(range(slots[0], slots[0] + len(slots))):
            # Index-selecting a noncontiguous group copies every persistent
            # tensor, including each row's full-capacity leaf archive. At a
            # 131K capacity that temporary can consume multiple GiB per layer.
            # Retain batching within maximal contiguous runs without ever
            # materializing a second semantic cache.
            run_begin = 0
            for index in range(1, len(plan) + 1):
                run_ended = index == len(plan) or slots[index] != slots[index - 1] + 1
                if not run_ended:
                    continue
                self._direct_cached_prefill_group(
                    query,
                    key,
                    value,
                    output,
                    plan[run_begin:index],
                    finalize_cache_for_decode=finalize_cache_for_decode,
                )
                run_begin = index
            return
        packed_begin = plan[0][1]
        packed_end = packed_begin + len(plan) * length
        packed = packed_end <= int(query.size(0)) and all(
            begin == packed_begin + source_slot * length
            and end == packed_begin + (source_slot + 1) * length
            for source_slot, (_, begin, end, _) in enumerate(plan)
        )
        if packed:
            self.cached_prefill_packed_calls += 1
        else:
            self.cached_prefill_nonpacked_calls += 1
        q = (
            query[packed_begin:packed_end]
            .reshape(len(plan), length, *query.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [query[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        k = (
            key[packed_begin:packed_end]
            .reshape(len(plan), length, *key.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [key[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        v = (
            value[packed_begin:packed_end]
            .reshape(len(plan), length, *value.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [value[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        cache = self._range_cache(slots[0], slots[-1] + 1)
        if cache.total_length != previous_length:
            raise RuntimeError(
                "batched cached LOD prefill length differs from its prepared plan"
            )
        output_view = (
            output[packed_begin:packed_end]
            .reshape(len(plan), length, *output.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else None
        )
        defer_cache_update = self.deferred_prefill_stream is not None
        # The final state/page update does not contribute to this layer's
        # output.  Queue it behind the attention work and let subsequent model
        # layers hide it; _range_cache and decode consume the completion event.
        if defer_cache_update:
            self.engine._lod_prefill_deferred_update_stream = (
                self.deferred_prefill_stream
            )
        try:
            result, cache = self.engine(
                q,
                k,
                v,
                cache=cache,
                use_cache=True,
                output_buffer=output_view,
                finalize_cache_for_decode=finalize_cache_for_decode,
            )
        finally:
            if defer_cache_update:
                del self.engine._lod_prefill_deferred_update_stream
        if cache is None:
            raise AssertionError("batched cached LOD prefill did not return a cache")
        if len(slots) > 1:
            self.batched_cached_prefill_calls += 1
            self.batched_cached_prefill_rows += len(slots)
        if defer_cache_update:
            deferred = self.deferred_prefill_stream
            if deferred is None:
                raise AssertionError("deferred prefill stream is missing")
            with torch.cuda.stream(deferred):
                self._synchronize_rows(slots, cache)
                completed = torch.cuda.Event()
                completed.record(deferred)
            for slot in slots:
                self.deferred_prefill_events[slot] = completed
        else:
            self._synchronize_rows(slots, cache)
        self.engine.reset_runtime_cache()
        if packed:
            if output_view is None:
                raise AssertionError("packed cached prefill has no output view")
            if result.data_ptr() != output_view.data_ptr():
                output_view.copy_(result)
        else:
            for source_slot, (_, begin, end, _) in enumerate(plan):
                output[begin:end].copy_(result[source_slot].permute(1, 0, 2))

    def direct_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run ragged initial or cached prefill into authoritative LOD rows."""
        plan = self.direct_prefill_plan
        self.direct_prefill_plan = None
        prompt_lengths = self.direct_prefill_prompt_lengths
        self.direct_prefill_prompt_lengths = {}
        if plan is None:
            raise RuntimeError("direct LOD prefill has no prepared request plan")
        self.direct_prefill_calls += 1
        initial: dict[tuple[int, bool], list[tuple[int, int, int, int]]] = {}
        cached: list[tuple[int, int, int, int]] = []
        for item in plan:
            slot, begin, end, previous_length = item
            if end <= begin:
                continue
            if previous_length == 0 and not self.ready[slot]:
                if slot not in prompt_lengths:
                    raise RuntimeError("direct LOD prefill has no total prompt length")
                length = end - begin
                initial.setdefault((length, length >= prompt_lengths[slot]), []).append(
                    item
                )
            elif previous_length > 0 and self.ready[slot]:
                cached.append(item)
            elif previous_length > 0:
                raise RuntimeError("cached LOD prefill row is not initialized")
            else:
                raise RuntimeError("initial LOD prefill row is already initialized")

        for (length, finalize_cache_for_decode), group in initial.items():
            slots = tuple(slot for slot, _, _, _ in group)
            self.engine.recursive_prefill_request_total_len = max(
                prompt_lengths[slot] for slot in slots
            )
            packed_begin = group[0][1]
            packed_end = packed_begin + len(group) * length
            packed = packed_end <= int(query.size(0)) and all(
                begin == packed_begin + source_slot * length
                and end == packed_begin + (source_slot + 1) * length
                for source_slot, (_, begin, end, _) in enumerate(group)
            )
            q = (
                query[packed_begin:packed_end]
                .reshape(len(group), length, *query.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [query[begin:end].permute(1, 0, 2) for _, begin, end, _ in group]
                )
            )
            k = (
                key[packed_begin:packed_end]
                .reshape(len(group), length, *key.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [key[begin:end].permute(1, 0, 2) for _, begin, end, _ in group]
                )
            )
            v = (
                value[packed_begin:packed_end]
                .reshape(len(group), length, *value.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [value[begin:end].permute(1, 0, 2) for _, begin, end, _ in group]
                )
            )
            output_view = (
                output[packed_begin:packed_end]
                .reshape(len(group), length, *output.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else None
            )
            prefill_storage = self._initial_prefill_storage(slots)
            defer_cache = bool(
                self.engine.prefill_exact_first_chunk
                and length <= int(self.engine.prefill_chunk_len)
                and self.deferred_prefill_stream is not None
            )
            if defer_cache:
                # The first scheduler chunk already uses exact attention, so
                # its semantic cache can be constructed after its output is
                # available.  Later model layers hide this work.  The per-row
                # event is host-synchronized before any scheduler stream can
                # consume the completed cache, avoiding the graph-stream race
                # that a current-stream-only wait allowed.
                result = self.engine._exact_attention(q, k, v, causal=True)
                if output_view is not None:
                    output_view.copy_(result)
                    result = output_view
                foreground = torch.cuda.current_stream(self.device)
                deferred = self.deferred_prefill_stream
                if deferred is None:
                    raise AssertionError("deferred prefill stream is missing")
                deferred.wait_stream(foreground)
                q.record_stream(deferred)
                k.record_stream(deferred)
                v.record_stream(deferred)
                with torch.cuda.stream(deferred):
                    if prefill_storage is not None:
                        self.engine._lod_prefill_storage = prefill_storage
                    try:
                        cache = self.engine.build_cache_from_bf16(
                            k,
                            v,
                            clustering_query=(
                                q
                                if self.engine.state_clustering_query_metric != "none"
                                else None
                            ),
                            finalize_cache_for_decode=finalize_cache_for_decode,
                        )
                    finally:
                        if prefill_storage is not None:
                            del self.engine._lod_prefill_storage
                    self.install_rows(slots, cache)
                    self.engine.reset_runtime_cache()
                    completed = torch.cuda.Event()
                    completed.record(deferred)
                for slot in slots:
                    self.deferred_prefill_events[slot] = completed
            else:
                if prefill_storage is not None:
                    self.engine._lod_prefill_storage = prefill_storage
                try:
                    result, cache = self.engine(
                        q,
                        k,
                        v,
                        use_cache=True,
                        output_buffer=output_view,
                        finalize_cache_for_decode=finalize_cache_for_decode,
                    )
                except Exception:
                    if prefill_storage is not None:
                        self._reset_range(slots[0], slots[-1] + 1)
                    raise
                finally:
                    if prefill_storage is not None:
                        del self.engine._lod_prefill_storage
            if cache is None:
                raise AssertionError("direct LOD prefill did not return a cache")
            if not defer_cache:
                if tuple(sorted(slots)) == tuple(range(min(slots), max(slots) + 1)):
                    self.install_rows(slots, cache)
                else:
                    for source_slot, (slot, _, _, _) in enumerate(group):
                        self.install(slot, cache, source_slot=source_slot)
                self.engine.reset_runtime_cache()
            if packed:
                if output_view is None or result.data_ptr() != output_view.data_ptr():
                    raise AssertionError("LOD prefill did not use its output buffer")
            else:
                for source_slot, (_, begin, end, _) in enumerate(group):
                    output[begin:end].copy_(result[source_slot].permute(1, 0, 2))

        if not cached:
            return output
        lengths = {end - begin for _, begin, end, _ in cached}
        previous_lengths = {previous_length for _, _, _, previous_length in cached}
        slots = tuple(slot for slot, _, _, _ in cached)
        ordered_plan = tuple(sorted(cached, key=lambda item: item[0]))
        ordered_slots = tuple(slot for slot, _, _, _ in ordered_plan)
        contiguous_slots = bool(ordered_slots) and ordered_slots == tuple(
            range(ordered_slots[0], ordered_slots[0] + len(ordered_slots))
        )
        self.cached_prefill_candidate_calls += 1
        self.cached_prefill_candidate_rows += len(cached)
        self.cached_prefill_nonuniform_lengths += int(len(lengths) != 1)
        self.cached_prefill_nonuniform_previous += int(len(previous_lengths) != 1)
        self.cached_prefill_noncontiguous += int(not contiguous_slots)
        groups: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = {}
        for item in ordered_plan:
            slot, begin, end, previous_length = item
            metadata = self.metadata[slot]
            signature = (
                end - begin,
                previous_length,
                int(end - begin + previous_length >= prompt_lengths[slot]),
                int(metadata["state_len"]),
                int(metadata.get("scheduled_state_len", metadata["state_len"])),
                int(metadata["coverage"]),
                int(metadata["recent_len"]),
                int(metadata["leaf_count"]),
                int(metadata["overflow_safe_until"]),
            )
            groups.setdefault(signature, []).append(item)
        for group in groups.values():
            group_slots = tuple(slot for slot, _, _, _ in group)
            if any(slot not in prompt_lengths for slot in group_slots):
                raise RuntimeError("cached LOD prefill has no total prompt length")
            self.engine.recursive_prefill_request_total_len = max(
                prompt_lengths[slot] for slot in group_slots
            )
            self._direct_cached_prefill_group(
                query,
                key,
                value,
                output,
                tuple(group),
                finalize_cache_for_decode=bool(
                    group[0][2] - group[0][1] + group[0][3]
                    >= prompt_lengths[group[0][0]]
                ),
            )
        return output

    def _row_cache(self, slot: int) -> KernelLODCache:
        return self._range_cache(slot, slot + 1)

    def _range_cache(self, start: int, stop: int) -> KernelLODCache:
        if not 0 <= start < stop <= self.max_requests:
            raise IndexError("LOD cache row range is outside the fixed pool")
        self.wait_deferred_prefill(tuple(range(start, stop)))
        metadata = self.metadata[start]
        scalar_names = (
            "state_len",
            "scheduled_state_len",
            "coverage",
            "recent_len",
            "total_len",
            "leaf_count",
            "overflow_safe_until",
        )
        for slot in range(start + 1, stop):
            if any(
                int(self.metadata[slot][name]) != int(metadata[name])
                for name in scalar_names
            ):
                raise ValueError("batched LOD catch-up rows have different metadata")
        state: dict[str, object] = {
            name: value[start:stop]
            for name, value in self.state.items()
            if isinstance(value, torch.Tensor) and value.ndim
        }
        state.update(
            state_len=int(metadata["state_len"]),
            scheduled_state_len=int(
                metadata.get("scheduled_state_len", metadata["state_len"])
            ),
            coverage=int(metadata["coverage"]),
            state_capacity=self.state_capacity,
            recent_len=int(metadata["recent_len"]),
            total_len=int(metadata["total_len"]),
        )
        page_pool = self.state["page_cache"]
        page: dict[str, object] = {}
        for name, value in page_pool.items():
            if name.startswith("unified_page1_"):
                continue
            page[name] = (
                value[start:stop]
                if isinstance(value, torch.Tensor) and value.ndim
                else value
            )
        page["leaf_lens"] = self.leaf_lens[start:stop]
        page.update(
            leaf_count=int(metadata["leaf_count"]),
            leaf_capacity=self.leaf_capacity,
            overflow_active=True,
            overflow_safe_until=int(metadata["overflow_safe_until"]),
        )
        state["page_cache"] = page
        return KernelLODCache(state)

    def _catch_up_target(self, slot: int, total_length: int) -> tuple[int, int]:
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        recent_length = total_length - coverage
        if recent_length < 0 or recent_length > self.local_capacity:
            raise ValueError("decode-local length exceeds its fixed cache row")
        update_len = int(self.engine.decode_state_update_len)
        exact_floor = int(self.engine.local_len - self.engine.chunk_len)
        target_coverage = max(min(total_length, self.engine.chunk_len), coverage)
        pending_update = total_length + 1 - target_coverage - exact_floor
        if pending_update > update_len:
            target_coverage += ((pending_update - 1) // update_len) * update_len
        return recent_length, min(target_coverage, total_length)

    def catch_up(self, slot: int, total_length: int) -> None:
        if not self.ready[slot]:
            raise RuntimeError("cannot catch up an uninitialized LOD request row")
        metadata = self.metadata[slot]
        coverage = int(metadata["coverage"])
        recent_length, target_coverage = self._catch_up_target(slot, total_length)
        if coverage >= target_coverage:
            # Captured decode already appended K/V and advanced local_lens on
            # device. Most tokens need only this host metadata bookkeeping.
            metadata["total_len"] = total_length
            metadata["recent_len"] = recent_length
            return
        row = self._row_cache(slot)
        self.engine.catch_up_cache(
            row, total_length=total_length, recent_length=recent_length
        )
        self._finish_single_catch_up(slot, row)

    def _finish_single_catch_up(self, slot: int, row: KernelLODCache) -> None:
        page = row.state["page_cache"]
        self.metadata[slot].update(
            state_len=int(row.state["state_len"]),
            scheduled_state_len=int(
                row.state.get("scheduled_state_len", row.state["state_len"])
            ),
            coverage=int(row.state["coverage"]),
            total_len=int(row.state["total_len"]),
            recent_len=int(row.state["recent_len"]),
            leaf_count=int(page["leaf_count"]),
            overflow_safe_until=int(page["overflow_safe_until"]),
        )
        self.local_lens[slot].fill_(int(row.state["recent_len"]))
        self.state_lens[slot].fill_(int(row.state["state_len"]))
        self.leaf_lens[slot].fill_(int(page["leaf_count"]))
        self._refresh_unified_page1_coarse((slot,))

    def catch_up_precomputed(
        self,
        slot: int,
        total_length: int,
        *,
        state_len: int,
        owners: torch.Tensor,
    ) -> None:
        """Finish one catch-up whose centroid update was batched by layer."""

        recent_length, target_coverage = self._catch_up_target(slot, total_length)
        if int(self.metadata[slot]["coverage"]) >= target_coverage:
            raise ValueError("precomputed LOD catch-up has no pending state update")
        row = self._row_cache(slot)
        self.engine.catch_up_cache(
            row,
            total_length=total_length,
            recent_length=recent_length,
            _precomputed_update=(state_len, owners, None),
        )
        self.catch_up_batches += 1
        self.catch_up_rows += 1
        self._finish_single_catch_up(slot, row)

    def catch_up_many(self, requests: list[tuple[int, int]]) -> None:
        """Batch equal-metadata contiguous rows at a state-update boundary."""
        pending: dict[tuple[int, ...], list[int]] = {}
        for slot, total_length in requests:
            if not self.ready[slot]:
                raise RuntimeError("cannot catch up an uninitialized LOD request row")
            metadata = self.metadata[slot]
            recent_length, target_coverage = self._catch_up_target(slot, total_length)
            if int(metadata["coverage"]) >= target_coverage:
                metadata["total_len"] = total_length
                metadata["recent_len"] = recent_length
                continue
            signature = (
                total_length,
                int(metadata["state_len"]),
                int(metadata.get("scheduled_state_len", metadata["state_len"])),
                int(metadata["coverage"]),
                int(metadata["recent_len"]),
                int(metadata["leaf_count"]),
                int(metadata["overflow_safe_until"]),
            )
            pending.setdefault(signature, []).append(slot)

        for signature, slots in pending.items():
            total_length = signature[0]
            slots.sort()
            begin = 0
            while begin < len(slots):
                end = begin + 1
                while end < len(slots) and slots[end] == slots[end - 1] + 1:
                    end += 1
                start_slot = slots[begin]
                stop_slot = slots[end - 1] + 1
                row = self._range_cache(start_slot, stop_slot)
                recent_length = total_length - int(row.state["coverage"])
                self.engine.catch_up_cache(
                    row,
                    total_length=total_length,
                    recent_length=recent_length,
                )
                self.catch_up_batches += 1
                self.catch_up_rows += stop_slot - start_slot
                page = row.state["page_cache"]
                for slot in range(start_slot, stop_slot):
                    self.metadata[slot].update(
                        state_len=int(row.state["state_len"]),
                        scheduled_state_len=int(
                            row.state.get("scheduled_state_len", row.state["state_len"])
                        ),
                        coverage=int(row.state["coverage"]),
                        total_len=int(row.state["total_len"]),
                        recent_len=int(row.state["recent_len"]),
                        leaf_count=int(page["leaf_count"]),
                        overflow_safe_until=int(page["overflow_safe_until"]),
                    )
                self.local_lens[start_slot:stop_slot].fill_(
                    int(row.state["recent_len"])
                )
                self.state_lens[start_slot:stop_slot].fill_(int(row.state["state_len"]))
                self.leaf_lens[start_slot:stop_slot].fill_(int(page["leaf_count"]))
                self._refresh_unified_page1_coarse(tuple(range(start_slot, stop_slot)))
                begin = end

    def _buffers(self, query: torch.Tensor, rows: int) -> dict[str, torch.Tensor]:
        buffers = self.decode_buffers.get(rows)
        storage = self.decode_buffer_storage
        if storage is None or storage["partial_out"].device != query.device:
            template = query.new_empty(
                self.max_requests,
                self.query_heads,
                1,
                self.head_dim,
            )
            storage = new_fused_decode_buffers(
                template,
                splits=int(self.engine.decode_split_kv),
                exact_kv_heads=(
                    self.kv_heads
                    if self.engine.exact_decode_limit > 0
                    and self.settings.levels == 3
                    else None
                ),
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
                route_segment_tiles=int(self.engine.decode_route_segment_tiles),
                materialized_state_route=(
                    self.engine.recursive_state_route_backend == "resplit"
                ),
                gqa_union_kv_heads=(
                    self.kv_heads
                    if self.settings.levels == 2
                    and self.query_heads % self.kv_heads == 0
                    and 1 < self.query_heads // self.kv_heads <= 16
                    and self.head_dim in (128, 256)
                    and self.dtype == torch.bfloat16
                    else None
                ),
                gqa_union_index_capacity=(
                    self.leaf_capacity
                    + int(self.engine.local_len)
                    + 1
                    + self.state_capacity
                    + (
                        int(self.state["sink_k"].size(2))
                        if isinstance(self.state.get("sink_k"), torch.Tensor)
                        else 0
                    )
                    if self.settings.levels == 2
                    and self.query_heads % self.kv_heads == 0
                    and 1 < self.query_heads // self.kv_heads <= 16
                    and self.head_dim in (128, 256)
                    and self.dtype == torch.bfloat16
                    else None
                ),
                gqa_union_hip=True,
                gqa_union_fixed_mask=self.settings.decode_gqa_fixed_mask_aiter,
                gqa_union_fixed_mask_tile_size=64,
                gqa_union_fixed_mask_segments=(
                    self.settings.decode_gqa_fixed_mask_segments
                ),
            )
            if bool(self.engine.recursive_materialize_page_scores):
                storage["recursive_page_scores"] = torch.empty(
                    self.max_requests,
                    self.query_heads,
                    1,
                    self.page_capacity,
                    dtype=torch.float32,
                    device=self.device,
                )
            if self.engine.exact_decode_limit > 0:
                storage["exact_context_lens"] = torch.empty(
                    self.max_requests * self.kv_heads,
                    dtype=torch.int32,
                    device=self.device,
                )
                storage["exact_exp_sums"] = torch.empty_like(
                    storage["partial_lse"]
                )
            if (
                self.head_dim in (128, 256)
                and 1 < self.query_heads // self.kv_heads <= 16
                and not bool(self.engine.recursive_materialize_page_scores)
            ):
                # The wide-head local decoder materializes one regular GQA
                # score field so both QK and PV use MFMA without reloading K/V
                # independently for every query head. Recursive materialized
                # decode reuses its larger page-score field for this earlier,
                # non-overlapping phase rather than reserving another buffer.
                storage["wide_gqa_local_scores"] = torch.empty(
                    self.max_requests,
                    self.query_heads,
                    self.decode_local_limit + 1,
                    dtype=torch.float32,
                    device=self.device,
                )
            self.decode_buffer_storage = storage
            self.decode_buffers.clear()
            buffers = None
        if buffers is None:
            buffers = {
                name: (
                    tensor[:rows]
                    if (tensor.ndim and int(tensor.size(0)) == self.max_requests)
                    else tensor
                )
                for name, tensor in storage.items()
            }
            self.decode_buffers[rows] = buffers
        return buffers

    def reserve_decode_buffers(self, rows: int) -> None:
        """Reserve graph scratch before vLLM computes its native cache budget."""
        if not 1 <= rows <= self.max_requests:
            raise ValueError("decode scratch rows exceed the fixed LOD pool")
        query = torch.empty(
            rows,
            self.query_heads,
            1,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._buffers(query, rows)

    def reserve_speculative_decode_buffers(self, rows: int, steps: int) -> None:
        """Reserve fixed-address request-major/step-major graph staging.

        vLLM lays uniform speculative verification out request-major, while
        the graph-safe LOD primitive advances one token for every request at a
        time.  These tiny staging tensors make that transpose explicit and,
        crucially, keep every pointer stable across graph replay.
        """
        if not 1 <= rows <= self.max_requests:
            raise ValueError("speculative scratch rows exceed the fixed LOD pool")
        if steps <= 1:
            raise ValueError("speculative decode requires at least two steps")
        signature = (rows, steps)
        if signature in self.speculative_decode_buffers:
            return
        self.reserve_decode_buffers(rows)
        staging: dict[str, Any] = {
            "q": torch.empty(
                steps,
                rows,
                self.query_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            "k": torch.empty(
                steps,
                rows,
                self.kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            "v": torch.empty(
                steps,
                rows,
                self.kv_heads,
                self.value_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            "out": torch.empty(
                steps,
                rows,
                self.query_heads,
                self.value_dim,
                dtype=self.dtype,
                device=self.device,
            ),
        }
        if self._parallel_speculative_decode_eligible(steps):
            total_rows = rows * steps
            parallel_steps = self._parallel_speculative_chunk_steps(steps, rows)
            parallel_rows = rows * parallel_steps
            speculative_route_backend = (
                self._speculative_recursive_state_route_backend()
            )
            template = torch.empty(
                parallel_rows,
                self.query_heads,
                1,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            staging["cache_indices"] = torch.empty(
                total_rows, dtype=torch.long, device=self.device
            )
            staging["local_lens"] = torch.empty(
                total_rows, dtype=torch.int32, device=self.device
            )
            staging["decode_buffers"] = new_fused_decode_buffers(
                template,
                splits=int(self.engine.decode_split_kv),
                exact_kv_heads=(
                    self.kv_heads
                    if self.engine.exact_decode_limit > 0
                    and self.settings.levels == 3
                    else None
                ),
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
                route_segment_tiles=int(self.engine.decode_route_segment_tiles),
                materialized_state_route=bool(
                    self.settings.levels == 3 and speculative_route_backend == "resplit"
                ),
                # Multi-token verification consumes each query's own four
                # centroids and does not build a GQA-wide leaf union.
                gqa_union_kv_heads=None,
                gqa_union_index_capacity=None,
                gqa_union_hip=True,
                gqa_union_fixed_mask=False,
                gqa_union_fixed_mask_tile_size=64,
                gqa_union_fixed_mask_segments=(
                    self.settings.decode_gqa_fixed_mask_segments
                ),
            )
            if self.settings.levels == 3:
                if bool(self.engine.recursive_materialize_page_scores):
                    staging["decode_buffers"]["recursive_page_scores"] = torch.empty(
                        parallel_rows,
                        self.query_heads,
                        1,
                        self.page_capacity,
                        dtype=torch.float32,
                        device=self.device,
                    )
                elif (
                    self.head_dim in (128, 256)
                    and 1 < self.query_heads // self.kv_heads <= 16
                ):
                    staging["decode_buffers"]["wide_gqa_local_scores"] = torch.empty(
                        parallel_rows,
                        self.query_heads,
                        self.decode_local_limit + 1,
                        dtype=torch.float32,
                        device=self.device,
                    )
            staging["decode_buffers"]["speculative_parallel_execution_marker"] = (
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            staging["decode_buffers"]["speculative_local_partial_out"] = (
                torch.empty_like(staging["decode_buffers"]["partial_out"])
            )
            staging["decode_buffers"]["speculative_local_partial_lse"] = (
                torch.empty_like(staging["decode_buffers"]["partial_lse"])
            )
            staging["decode_buffers"]["speculative_route_execution_marker"] = (
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            staging["decode_buffers"]["speculative_local_execution_marker"] = (
                torch.zeros(1, dtype=torch.int32, device=self.device)
            )
            if self.engine.exact_decode_limit > 0:
                staging["decode_buffers"]["exact_context_lens"] = torch.empty(
                    parallel_rows * self.kv_heads,
                    dtype=torch.int32,
                    device=self.device,
                )
                staging["decode_buffers"]["exact_exp_sums"] = torch.empty_like(
                    staging["decode_buffers"]["partial_lse"]
                )
        self.speculative_decode_buffers[signature] = staging

    def _parallel_speculative_chunk_steps(self, steps: int, rows: int = 1) -> int:
        """Bound one recursive flattened verifier launch to 64 query rows."""
        if self.settings.levels != 3:
            return steps
        maximum_rows = 64
        maximum = min(steps, max(1, maximum_rows // rows))
        while steps % maximum:
            maximum -= 1
        return maximum

    def _parallel_speculative_decode_eligible(self, steps: int) -> bool:
        """Whether one flattened launch can verify all proposal positions."""
        recursive = self.settings.levels == 3
        two_level = self._parallel_speculative_two_level_eligible(steps)
        return steps >= 2 and (recursive or two_level)

    def _speculative_recursive_state_route_backend(self) -> str:
        """Resolve the recursive route used inside speculative verification.

        The materialized re-split route remains useful for ordinary decode,
        but its long-context score-table pipeline is not yet safe under
        speculative verification (including the serial verifier control).
        Keep the ordinary per-model policy unchanged and default only the
        speculative recursive path to the grouped producer.
        """
        if self.settings.levels != 3:
            return str(self.engine.recursive_state_route_backend)
        return "fused"

    def _parallel_speculative_two_level_eligible(self, steps: int) -> bool:
        """Whether MTP can refine complete centroids for all rows at once."""
        gqa = self.query_heads // self.kv_heads
        return bool(
            steps >= 2
            and steps % 2 == 0
            and self.speculative_tokens > 0
            and self.settings.levels == 2
            and self.settings.family is ModelFamily.QWEN38
            and self.query_heads % self.kv_heads == 0
            and 1 < gqa <= 8
            and bool(self.engine.decode_route_gqa_grouped)
            and int(self.engine.decode_route_segment_tiles) == 1
            and 2 * gqa <= 16
            and self.head_dim in (128, 256)
            and self.dtype == torch.bfloat16
        )

    def _shared_speculative_route_eligible(self, steps: int, rows: int = 1) -> bool:
        """Whether proposal positions fit pairwise native grouped route tiles."""
        return bool(
            self._parallel_speculative_decode_eligible(steps)
            and self._parallel_speculative_chunk_steps(steps, rows) == steps
            and steps % 2 == 0
            and bool(self.engine.decode_route_gqa_grouped)
            and int(self.engine.decode_route_segment_tiles) == 1
            and 2 * (self.query_heads // self.kv_heads) <= 16
        )

    def speculative_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Verify a uniform proposal inside one replayable target graph.

        The proposal queries route concurrently against the immutable remote
        LOD state. Their K/V are staged before that launch, so each query sees
        the exact causal local suffix through its own logical length. The host
        truncates rejected tail entries by resetting ``local_lens`` before the
        next replay; no route is lagged.
        """
        steps = int(self.speculative_decode_steps)
        if steps <= 1:
            raise RuntimeError("speculative LOD decode was not prepared")
        total_tokens = int(query.size(0))
        if total_tokens % steps:
            raise ValueError(
                "uniform speculative tokens do not divide the padded batch"
            )
        rows = total_tokens // steps
        signature = (rows, steps)
        staging = self.speculative_decode_buffers.get(signature)
        if staging is None:
            raise RuntimeError(
                "speculative LOD graph staging was not reserved before forward"
            )

        # One copy per tensor changes vLLM's request-major layout [B, M, H, D]
        # into step-major [M, B, H, D].  The attention calls below then use
        # contiguous M=1 inputs and outputs without per-step packing kernels.
        staging["q"].copy_(
            query[: rows * steps]
            .view(rows, steps, self.query_heads, self.head_dim)
            .permute(1, 0, 2, 3)
        )
        staging["k"].copy_(
            key[: rows * steps]
            .view(rows, steps, self.kv_heads, self.head_dim)
            .permute(1, 0, 2, 3)
        )
        staging["v"].copy_(
            value[: rows * steps]
            .view(rows, steps, self.kv_heads, self.value_dim)
            .permute(1, 0, 2, 3)
        )

        if self._parallel_speculative_decode_eligible(steps):
            staging["decode_buffers"]["speculative_parallel_execution_marker"].add_(1)
            flat_q = staging["q"].view(rows * steps, self.query_heads, self.head_dim)
            flat_k = staging["k"].view(rows * steps, self.kv_heads, self.head_dim)
            flat_v = staging["v"].view(rows * steps, self.kv_heads, self.value_dim)
            flat_out = staging["out"].view(
                rows * steps, self.query_heads, self.value_dim
            )
            prepare_speculative_decode_kv(
                self.active_indices[:rows],
                self.local_lens,
                flat_k.unsqueeze(2),
                flat_v.unsqueeze(2),
                self.state["recent_k"],
                self.state["recent_v"],
                staging["cache_indices"],
                staging["local_lens"],
                rows=rows,
                steps=steps,
            )

            parallel_steps = self._parallel_speculative_chunk_steps(steps, rows)

            class _ParallelMetadata:
                num_actual_tokens = rows * parallel_steps

            for step_begin in range(0, steps, parallel_steps):
                begin = step_begin * rows
                end = begin + parallel_steps * rows
                self.decode(
                    flat_q[begin:end],
                    flat_k[begin:end],
                    flat_v[begin:end],
                    _ParallelMetadata(),
                    flat_out[begin:end],
                    cache_indices=staging["cache_indices"][begin:end],
                    local_lens=staging["local_lens"][begin:end],
                    decode_buffers=staging["decode_buffers"],
                    local_lens_are_logical=True,
                    store_new_kv=False,
                    advance_local_lens=False,
                    speculative_steps=(
                        steps
                        if parallel_steps == steps
                        and self._shared_speculative_route_eligible(steps, rows)
                        else 1
                    ),
                    recursive_state_route_backend=(
                        self._speculative_recursive_state_route_backend()
                    ),
                )
            advance_decode_cache_lengths(
                self.active_indices[:rows], self.local_lens, increment=steps
            )
            output[: rows * steps].view(
                rows, steps, self.query_heads, self.value_dim
            ).copy_(staging["out"].permute(1, 0, 2, 3))
            return output

        class _Metadata:
            num_actual_tokens = rows

        metadata = _Metadata()
        for step in range(steps):
            self.decode(
                staging["q"][step],
                staging["k"][step],
                staging["v"][step],
                metadata,
                staging["out"][step],
                recursive_state_route_backend=(
                    self._speculative_recursive_state_route_backend()
                ),
            )
        output[: rows * steps].view(
            rows, steps, self.query_heads, self.value_dim
        ).copy_(staging["out"].permute(1, 0, 2, 3))
        return output

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Any,
        output: torch.Tensor,
        *,
        cache_indices: torch.Tensor | None = None,
        local_lens: torch.Tensor | None = None,
        decode_buffers: dict[str, torch.Tensor] | None = None,
        local_lens_are_logical: bool = False,
        store_new_kv: bool = True,
        advance_local_lens: bool = True,
        speculative_steps: int = 1,
        recursive_state_route_backend: str | None = None,
    ) -> torch.Tensor:
        self.decode_calls += 1
        rows = int(metadata.num_actual_tokens)
        if rows == 0:
            return output
        # Tensor-parallel head shards can be strided views. Decode routing
        # flattens each query row, so keep that tiny tensor dense for every
        # backend. The AITER GQA-union metadata path additionally requires
        # dense per-rank K/V tensors.
        q = query[:rows].unsqueeze(2).contiguous()
        k = key[:rows].unsqueeze(2)
        v = value[:rows].unsqueeze(2)
        k = k.contiguous()
        v = v.contiguous()
        if cache_indices is None:
            cache_indices = self.active_indices[:rows]
        if local_lens is None:
            local_lens = self.local_lens
        if decode_buffers is None:
            decode_buffers = self._buffers(q, rows)
        page = self.state["page_cache"]
        recursive = self.settings.levels == 3
        indexed_flat = not recursive and isinstance(
            page.get("page_indices"), torch.Tensor
        )
        page_k = page["leaf_k"] if recursive or indexed_flat else page["page_k"]
        page_v = page["leaf_v"] if recursive or indexed_flat else page["page_v"]
        flat_int8 = not recursive and (
            page_k.dtype == torch.int8 or page_v.dtype == torch.int8
        )
        exact_decode_limit = int(self.engine.exact_decode_limit)
        result = fused_decode_paged_lod_attention(
            q,
            self.state["state_k"],
            self.state["state_v"],
            self.state["counts"],
            self.state["recent_k"],
            self.state["recent_v"],
            page_k,
            page_v,
            page["slot_pages"],
            page["overflow_page_keys"],
            page["overflow_page_values"],
            page["overflow_used"],
            page["slot_lengths"],
            None,
            sink_k=self.state.get("sink_k"),
            sink_v=self.state.get("sink_v"),
            state_len=self.state_capacity,
            # Allocation size follows the longest configured request, while
            # each row routes only over the centroid prefix it has populated.
            state_lens=self.state_lens,
            # A delayed update leaves the displaced prefix exact until the
            # next catch-up. The effective limit reduces to local_len for the
            # ordinary update==chunk schedule.
            local_len=self.decode_local_limit,
            cache_indices=cache_indices,
            local_lens=local_lens,
            new_k=k,
            new_v=v,
            local_lens_are_logical=local_lens_are_logical,
            store_new_kv=store_new_kv,
            advance_local_lens=advance_local_lens,
            speculative_steps=speculative_steps,
            kv_group_size=self.query_heads // self.kv_heads,
            scale=float(self.engine.scaling),
            hash_probes=int(self.engine._page_lookup_probes(page)),
            block_n=int(self.engine.decode_block_n),
            num_warps=int(self.engine.decode_num_warps),
            waves_per_eu=int(self.engine.leaf_waves_per_eu),
            split_kv=int(self.engine.decode_split_kv),
            buffers=decode_buffers,
            use_dot=bool(self.engine.decode_use_dot),
            fuse_state_route=True,
            route_group_size=int(self.engine.decode_route_group_size),
            route_segment_tiles=int(self.engine.decode_route_segment_tiles),
            route_num_warps=int(self.engine.decode_route_num_warps),
            route_reduce_num_warps=int(self.engine.decode_route_reduce_num_warps),
            route_parallel_reduce=bool(self.engine.decode_route_parallel_reduce),
            route_parallel_reduce_block_d=int(
                self.engine.decode_route_parallel_reduce_block_d
            ),
            final_reduce_num_warps=int(self.engine.decode_final_reduce_num_warps),
            fuse_final_reduce=bool(self.engine.decode_fuse_final_reduce),
            route_gqa_grouped=bool(self.engine.decode_route_gqa_grouped),
            gqa_cooperative_leaf=False,
            # DFlash already supplies eight independent verifier rows. Keep
            # ordinary one-token decode on the GQA-shared union, but let
            # speculative verification consume each query head's four
            # complete centroids directly instead of building another union.
            gqa_union_decode=speculative_steps < 2,
            gqa_union_hip=True,
            gqa_union_fixed_mask_aiter=(
                self.settings.decode_gqa_fixed_mask_aiter and speculative_steps < 2
            ),
            gqa_union_fixed_mask_adaptive_segments=True,
            gqa_union_fixed_mask_reduce_block_d=(
                self.settings.decode_gqa_fixed_mask_reduce_block_d
            ),
            gqa_union_fixed_mask_scan_num_warps=(
                self.settings.decode_gqa_fixed_mask_scan_num_warps
            ),
            gqa_union_page1_k=page.get("unified_page1_k"),
            gqa_union_page1_v=page.get("unified_page1_v"),
            gqa_union_page1_bias=page.get("unified_page1_bias"),
            gqa_union_page1_leaf_offset=int(page.get("unified_page1_leaf_offset", 0)),
            gqa_union_page1_local_offset=int(page.get("unified_page1_local_offset", 0)),
            gqa_union_page1_sink_offset=int(page.get("unified_page1_sink_offset", 0)),
            gqa_union_page1_coarse_offset=int(
                page.get("unified_page1_coarse_offset", 0)
            ),
            gqa_union_fixed_indices=page.get("unified_page1_fixed_indices"),
            gqa_union_fixed_leaf_owners=page.get("unified_page1_fixed_leaf_owners"),
            gqa_union_fixed_slot_offsets=page.get("unified_page1_fixed_slot_offsets"),
            gqa_union_fixed_lengths=page.get("unified_page1_fixed_lengths"),
            protected_len=self.engine._protected_state_len(self.state_capacity),
            # Every centroid remains eligible for exact refinement. In
            # particular, recursive page refinement has bounded work even when
            # the selected centroid owns a large posting list.
            max_leaf_tokens=None,
            open_count=ROUTE_COUNT,
            recursive_page_cache=(page if recursive else None),
            flat_page_indices=(
                page["page_indices"] if indexed_flat else None
            ),
            flat_page_k_scales=(page.get("page_k_token_scales") if flat_int8 else None),
            flat_page_v_scales=(page.get("page_v_token_scales") if flat_int8 else None),
            recursive_quant_group_size=int(self.engine.leaf_quant_group_size),
            recursive_quant_token_group_size=int(
                self.engine.leaf_quant_token_group_size
            ),
            timing_events=getattr(self.engine, "_lod_decode_timing_events", None),
            recursive_page_select_block_n=int(
                self.engine.recursive_page_select_block_n
            ),
            recursive_state_route_backend=(
                self.engine.recursive_state_route_backend
                if recursive_state_route_backend is None
                else recursive_state_route_backend
            ),
            exact_decode_threshold=exact_decode_limit,
            exact_all_rows=(
                exact_decode_limit > 0
                and 0 < int(getattr(metadata, "max_seq_len", 0))
                <= exact_decode_limit
            ),
            exact_leaf_lens=self.leaf_lens,
            output_buffer=output[:rows].unsqueeze(2),
        )
        if result.data_ptr() != output.data_ptr():
            raise AssertionError("fused LOD decode did not use the vLLM output buffer")
        return output


__all__ = ["VLLMLayerLODPool"]
