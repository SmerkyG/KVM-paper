"""Fixed-address per-layer LOD pools used by captured vLLM decode graphs."""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Any

import torch

from lod_attention.kernels.paged_leaf_attention import (
    advance_decode_cache_lengths,
    dequantize_owned_virtual_paged_keys,
    fused_decode_paged_lod_attention,
    materialize_absorbed_mla_coarse_means,
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


def _dcp_prefill_archive_capacity(
    *, total_len: int, prompt_capacity: int, chunk_len: int, headroom: int
) -> tuple[int, int, int]:
    """Optional growing shadow storage; neither the schedule nor routing changes."""
    limit = _round_up(prompt_capacity, chunk_len) + max(chunk_len, headroom)
    growth_chunk = int(os.environ.get("LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK", "0"))
    if growth_chunk < 0:
        raise ValueError("LOD_KIMI_PREFILL_SHADOW_GROW_CHUNK must be nonnegative")
    if not growth_chunk:
        return limit, limit, 0
    growth_chunk = _round_up(growth_chunk, chunk_len)
    return min(_round_up(total_len, growth_chunk), limit), limit, growth_chunk


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
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        dcp_group: Any | None = None,
        dcp_interleave_size: int = 1,
        request_owner_prefill: bool | None = None,
        shared_decode_scratch: dict | None = None,
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
        self.shared_decode_scratch = shared_decode_scratch
        self.dcp_world_size = int(dcp_world_size)
        self.dcp_rank = int(dcp_rank)
        self.dcp_group = dcp_group
        self.dcp_interleave_size = int(dcp_interleave_size)
        if self.dcp_world_size < 1 or not 0 <= self.dcp_rank < self.dcp_world_size:
            raise ValueError("invalid DCP world size or rank")
        if (self.dcp_world_size > 1) != (self.dcp_group is not None):
            raise ValueError("DCP LoD requires its vLLM process group")
        if self.dcp_interleave_size < 1:
            raise ValueError("DCP KV interleave size must be positive")
        self.query_heads = int(layer.num_heads)
        self.kv_heads = int(layer.num_kv_heads)
        self.head_dim = int(layer.head_size)
        self.is_absorbed_mla = bool(getattr(layer, "_vllm_lod_absorbed_mla", False))
        self.kimi_request_owner_prefill = bool(
            self.is_absorbed_mla and (
                os.getenv("LOD_KIMI_REQUEST_OWNER_PREFILL") == "1"
                if request_owner_prefill is None else request_owner_prefill
            )
        )
        if self.kimi_request_owner_prefill and (
            self.dcp_world_size != 8 or self.query_heads != 12
            or self.head_dim != 576 or settings.levels != 2 or max_requests > 8
        ):
            raise ValueError("request-owner probe requires TP8/DCP8 K3, two-tier, at most eight requests")
        self._kimi_request_owner_rows: dict[int, dict] = {}
        self.kimi_head_owner_prefill = os.getenv("LOD_KIMI_HEAD_OWNER_PREFILL") == "1"
        if self.kimi_head_owner_prefill and (
            not self.kimi_request_owner_prefill or max_requests != 1
        ):
            raise ValueError("six head owners require the B1 TP8 request-owner prefill probe")
        self.kimi_sharded_leaf_prefill = bool(
            self.is_absorbed_mla and self.dcp_world_size > 1
            and os.environ.get("LOD_KIMI_DCP_SHARDED_LEAVES") == "1"
        )
        if self.kimi_sharded_leaf_prefill and (
            settings.levels != 2 or self.dcp_interleave_size != 1
            or self.head_dim != 576
            or os.environ.get("LOD_KIMI_DCP_LOCAL_PREFILL") == "1"
            or os.environ.get("LOD_KIMI_DCP_SHARED_PREFILL") == "1"
        ):
            raise NotImplementedError("global-centroid sharded prefill requires two-tier K3 DCP with unit interleave")
        # Experimental sequence-sliced prefill: keep a whole-sequence-sized
        # centroid budget on each rank, but archive only that rank's leaves.
        # This is deliberately opt-in; ordinary DCP decode is unchanged.
        self.kimi_shared_dcp_prefill = bool(
            self.is_absorbed_mla and self.dcp_world_size > 1
            and os.environ.get("LOD_KIMI_DCP_SHARED_PREFILL") == "1"
        )
        self.kimi_local_dcp_prefill = bool(
            self.is_absorbed_mla and self.dcp_world_size > 1
            and (os.environ.get("LOD_KIMI_DCP_LOCAL_PREFILL") == "1"
                 or self.kimi_shared_dcp_prefill)
        )
        if self.kimi_local_dcp_prefill and settings.levels != 2:
            raise NotImplementedError("local DCP prefill initially supports two-tier BF16")
        self.value_dim = (
            int(layer.kv_lora_rank)
            if self.is_absorbed_mla
            else int(layer.head_size_v)
        )
        self.kimi_head_tiled_decode = bool(
            self.is_absorbed_mla and self.head_dim == 576
            and self.value_dim == 512 and self.query_heads > 16
            and self.query_heads % 16 == 0 and settings.levels == 2
        )
        self.speculative_tokens = int(speculative_tokens)
        gqa = self.query_heads // self.kv_heads
        if not self.is_absorbed_mla and self.value_dim != self.head_dim:
            raise NotImplementedError(
                "LOD vLLM currently requires equal K and V widths"
            )
        if self.is_absorbed_mla:
            self.family = ModelFamily.KIMI_K3
            # Kimi supplies an already RMS-normalized latent, but the complete
            # [latent, direct-key] vector is not unit length.  Preserve raw
            # dot products and raw centroid sums in latent space.
            has_query_norm = True
            has_key_norm = False
        elif (self.head_dim, gqa) == (256, 6):
            self.family = ModelFamily.QWEN38
        elif (self.head_dim, gqa) == (128, 8):
            self.family = ModelFamily.K2
        else:
            raise ValueError(
                "the LoD release supports Qwen3.8 (D256/GQA6), K2 Horizon "
                "(D128/GQA8), and marked absorbed-MLA Kimi K3 layers"
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
                recursive_state_route_backend="fused",
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
            scale=float(layer.scale if self.is_absorbed_mla else layer.impl.scale),
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
        self._dcp_global_state_growth_factor = float(
            self.engine.state_growth_factor
        )
        self._dcp_global_state_min_len = int(self.engine.state_min_len)
        # DCP shards the chronological sequence, not the LoD policy.  Record
        # every global token-count parameter so rank-local cache construction
        # can apply the corresponding 1/D share.  Previously only the sqrt
        # state schedule was converted; 256-token updates and 512-token local
        # windows consequently became 2,048 and 4,096 global tokens at DCP8.
        self._dcp_global_lengths = {
            name: int(getattr(self.engine, name))
            for name in (
                "chunk_len",
                "local_len",
                "prefill_chunk_len",
                "prefill_local_len",
                "prefill_state_update_len",
                "decode_state_update_len",
                "decode_cache_headroom",
            )
        }
        if self.is_absorbed_mla:
            # The smol-Kimi checkpoint is capped at 4K, below the release
            # engine's ordinary exact-decode crossover.  Force routed decode
            # so its smoke test actually exercises the MLA LoD kernels (and
            # avoid the power-of-two-only AITER exact-cache fallback).
            self.engine.exact_decode_limit = 0
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
        # DCP decode owns an interleaved 1/D share of the chronological KV
        # stream.  Allocate the graph-stable persistent rows at that local
        # capacity from startup.  Incomplete prefill uses a temporary full
        # prompt shadow and never changes these pointers.
        self.persistent_request_capacity = (
            self._dcp_local_length(request_capacity)
            if self.dcp_world_size > 1
            else request_capacity
        )
        if self.kimi_request_owner_prefill:
            # The experiment owns its whole-sequence cache on one rank only.
            # Keep just bootstrap metadata in the unused fixed DCP pool.
            self.persistent_request_capacity = min(self.persistent_request_capacity, 512)
        with self._dcp_local_state_schedule():
            self.state_capacity = self.engine._state_capacity(
                self.persistent_request_capacity,
                min(self.persistent_request_capacity, int(self.engine.chunk_len)),
            )
            local_window = int(self.engine.local_len)
            local_chunk = int(self.engine.chunk_len)
            local_update = int(self.engine.decode_state_update_len)
            self.decode_local_capacity = local_window + local_update
            self.decode_local_limit = local_window - local_chunk + local_update
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
            leaf_rounding = local_chunk
            leaf_headroom = max(
                local_chunk, int(self.engine.decode_cache_headroom)
            )
        self.leaf_capacity = _round_up(
            self.persistent_request_capacity, leaf_rounding
        ) + leaf_headroom
        self.page_capacity = math.ceil(self.leaf_capacity / 16) + self.state_capacity
        self.hash_capacity = _power_of_two(
            self.page_capacity * int(self.engine.leaf_overflow_hash_factor)
        )
        self.state = self._allocate_state()
        if self.kimi_request_owner_prefill:
            owner_heads = 16 if self.kimi_head_owner_prefill else self.query_heads * self.dcp_world_size
            self.engine.config.num_attention_heads = owner_heads
            self.engine.num_key_value_groups = owner_heads
            self.engine._lod_kimi_prefill_head_group_limit = 16 if self.kimi_head_owner_prefill else 12
            self.engine._lod_kimi_reduce_prefill_routes = True
        if self.is_absorbed_mla:
            self._assert_shared_latent_storage()
        self.local_lens = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )
        self.dcp_global_lens = torch.zeros(
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
        self.dcp_decode_buffer_storage: dict[str, torch.Tensor] | None = None
        self.dcp_decode_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self.dcp_sharded = [self.dcp_world_size == 1] * max_requests
        # Only actively prefilling DCP requests retain a globally replicated
        # semantic cache.  It is released as soon as the final prompt chunk is
        # converted into the fixed-address rank-local row above.
        self.dcp_prefill_shadows: dict[int, KernelLODCache] = {}
        self.dcp_prefill_summaries: dict[int, dict[str, Any]] = {}
        self.dcp_cross_layer_initial_sinks: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # Runtime owns the host-side row map and updates it only when vLLM's
        # active batch changes.  Decode must not copy the graph-visible GPU
        # map back to the CPU on every layer and every token.
        self.active_decode_rows: tuple[int, ...] = ()
        self.speculative_decode_buffers: dict[tuple[int, int], dict[str, Any]] = {}
        self.decode_enabled = False
        self.speculative_decode_steps = 0
        self.hybrid_full_decode = False
        self.direct_prefill_plan: tuple[tuple[int, int, int, int], ...] | None = None
        self.direct_prefill_prompt_lengths: dict[int, int] = {}
        # The runtime can defer construction of an exact first prefix until
        # every attention layer has produced its K/V. The callback batches
        # only the expensive centroid update across layers; page ownership
        # remains local to this pool.
        self.initial_prefill_stager: Any | None = None
        self.cached_prefill_stager: Any | None = None
        self.deferred_prefill_stream: torch.cuda.Stream | None = None
        self.deferred_prefill_events: list[torch.cuda.Event | None] = [
            None
        ] * max_requests
        # Compact decode needs a centroid-major physical-leaf table. Prefill
        # can update the semantic archive many times, so invalidate here and
        # rebuild only when decode is about to consume the row.
        self.unified_page1_fixed_dirty = [False] * max_requests
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
        self.kimi_captured_owner_decode = bool(
            self.kimi_request_owner_prefill
            and os.getenv("LOD_KIMI_REQUEST_OWNER_DECODE") == "1")
        if self.kimi_captured_owner_decode:
            from .models.kimi_k3_owner_decode import initialize_owner_decode
            initialize_owner_decode(self)

    def _assert_shared_latent_storage(self) -> None:
        """Prove that Kimi values are views, never duplicate allocations."""

        pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for prefix in ("state", "recent", "sink"):
            key = self.state.get(f"{prefix}_k")
            value = self.state.get(f"{prefix}_v")
            if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                pairs.append((prefix, key, value))
        page = self.state.get("page_cache")
        if isinstance(page, dict):
            # BF16 Kimi pages store each value as a prefix view of the
            # latent-plus-direct key record.  Quantized pages deliberately
            # use distinct packed K/V arrays because their logical widths are
            # 576 and 512 respectively; only the unquantized state/local/sink
            # tensors retain the zero-copy alias invariant in that mode.
            page_prefixes = (
                () if int(page.get("leaf_quant_bits", 0)) else ("leaf", "page_sum")
            )
            for prefix in page_prefixes:
                key = page.get(f"{prefix}_k")
                value = page.get(f"{prefix}_v")
                if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                    pairs.append((prefix, key, value))
        for name, key, value in pairs:
            if int(key.size(-1)) != self.head_dim or int(value.size(-1)) != self.value_dim:
                raise AssertionError(f"Kimi {name} cache has incorrect K/V widths")
            if key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr():
                raise AssertionError(f"Kimi {name} values duplicate latent storage")
            if value.data_ptr() != key.data_ptr() or value.stride() != key.stride():
                raise AssertionError(f"Kimi {name} values are not the key-prefix view")

    def _assert_production_profile(
        self,
        gqa: int,
        *,
        has_query_norm: bool,
        has_key_norm: bool,
    ) -> None:
        """Reject any silent deviation from the paper's supported path."""

        expected_geometry = {
            ModelFamily.QWEN38: (256, 6),
            ModelFamily.K2: (128, 8),
        }.get(self.family)
        if self.dtype != torch.bfloat16:
            raise RuntimeError("LoD requires BF16 attention K/V inputs")
        if expected_geometry is not None and (self.head_dim, gqa) != expected_geometry:
            raise RuntimeError(
                f"{self.family.value} requires D/GQA={expected_geometry}, "
                f"got {(self.head_dim, gqa)}"
            )

        recursive = self.settings.levels == 3
        expected_bits = self.settings.kv_bits
        k2_int4 = self.family is ModelFamily.K2 and recursive and expected_bits == 4
        if self.family is ModelFamily.K2:
            expected_leaf_geometry = (256, 16) if k2_int4 else (128, 32)
        elif self.family is ModelFamily.KIMI_K3:
            expected_leaf_geometry = (
                int(os.getenv("LOD_KIMI_LEAF_BLOCK_M", "32")),
                16,
            )
        else:
            expected_leaf_geometry = (32, 16)
        expected_leaf_warps = (
            int(os.getenv("LOD_KIMI_LEAF_WARPS", "2"))
            if self.family is ModelFamily.KIMI_K3
            else (4 if k2_int4 else 2)
        )
        checks = {
            "top-eight routing": (
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
                (
                    self.engine.prefill_local_attention_backend == "aiter"
                    or (
                        self.family is ModelFamily.KIMI_K3
                        and os.getenv("LOD_KIMI_DISABLE_AITER_PREFILL", "0") == "1"
                        and self.engine.prefill_local_attention_backend == "torch"
                    )
                )
                and self.engine.fused_prefill_route_coarse
                and self.engine.fused_prefill_stable_recompute
                and self.engine.fused_prefill_external_recompute
                and self.engine.prefill_hierarchical_route
                and self.engine.prefill_overlap_coarse_leaf
            ),
            "GQA-aware AITER prefill route/coarse": (
                self.engine.prefill_aiter_route_coarse
                or self.family is ModelFamily.KIMI_K3
            ),
            "complete-centroid prefill": (
                not recursive or self.engine.recursive_prefill_all_leaves
            ),
            "compact recursive page directory": (
                not recursive or self.engine.leaf_inline_pages_per_slot == 32
            ),
            "leaf geometry": (
                self.engine.leaf_layout == "expert"
                and (self.engine.leaf_block_m, self.engine.leaf_block_n)
                == expected_leaf_geometry
                and self.engine.leaf_num_warps == expected_leaf_warps
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
            (
                self.settings.levels == 2
                or (self.settings.levels == 3 and self.family is ModelFamily.K2)
            )
            and self.dtype == torch.bfloat16
            and (1 < self.query_heads // self.kv_heads <= 16
                 or self.kimi_head_tiled_decode)
            and self.query_heads % self.kv_heads == 0
            and (
                self.head_dim in (128, 256)
                or (
                    self.is_absorbed_mla
                    and self.head_dim == 576
                    and self.value_dim == 512
                )
            )
        )
        sink_capacity = int(self.engine.separate_sink_cache)

        def latent_values(records: torch.Tensor) -> torch.Tensor:
            """Return the value view into a Kimi latent-plus-direct record."""

            if not self.is_absorbed_mla:
                raise AssertionError("latent value alias requested outside Kimi MLA")
            return records[..., : self.value_dim]

        if unified_page1:
            arena_leaf_offset = 0
            kv_rows = r * h
            arena_leaf_capacity = self.leaf_capacity if self.settings.levels == 2 else 0
            arena_local_offset = arena_leaf_offset + kv_rows * arena_leaf_capacity
            arena_sink_offset = arena_local_offset + kv_rows * self.local_capacity
            arena_coarse_offset = arena_sink_offset + kv_rows * sink_capacity
            arena_padding_index = arena_coarse_offset + kv_rows * self.state_capacity
            arena_capacity = arena_padding_index + 1
            unified_page1_k = torch.empty(
                arena_capacity, d, dtype=self.dtype, device=self.device
            )
            unified_page1_v = (
                latent_values(unified_page1_k)
                if self.is_absorbed_mla
                else torch.empty_like(unified_page1_k)
            )
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
            recent_v = (
                latent_values(recent_k)
                if self.is_absorbed_mla
                else unified_page1_v[
                    arena_local_offset : arena_local_offset
                    + kv_rows * self.local_capacity
                ].view(r, h, self.local_capacity, self.value_dim)
            )
            if self.settings.decode_gqa_fixed_mask_aiter:
                fixed_capacity = (
                    self.leaf_capacity
                    + self.decode_local_limit
                    + 1
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
                unified_page1_fixed_slot_offsets = torch.zeros(
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
            recent_v = (
                latent_values(recent_k)
                if self.is_absorbed_mla
                else torch.empty_like(recent_k)
            )
        state_k = torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device)
        state: dict[str, object] = {
            "state_k": state_k,
            "state_v": (
                latent_values(state_k)
                if self.is_absorbed_mla
                else torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device)
            ),
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
                state["sink_v"] = (
                    latent_values(state["sink_k"])
                    if self.is_absorbed_mla
                    else unified_page1_v[
                        arena_sink_offset : arena_sink_offset + r * h * sink_capacity
                    ].view(r, h, sink_capacity, self.value_dim)
                )
            else:
                sink_k = torch.empty(
                    r,
                    h,
                    1,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                )
                state["sink_k"] = sink_k
                state["sink_v"] = (
                    latent_values(sink_k)
                    if self.is_absorbed_mla
                    else torch.empty_like(sink_k)
                )
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
                leaf_v = (
                    latent_values(leaf_k)
                    if self.is_absorbed_mla
                    else unified_page1_v[
                        arena_leaf_offset : arena_leaf_offset
                        + r * h * self.leaf_capacity
                    ].view(r, h, self.leaf_capacity, self.value_dim)
                )
            else:
                leaf_k = torch.zeros(
                    r,
                    h,
                    self.leaf_capacity,
                    d,
                    dtype=self.dtype,
                    device=self.device,
                )
                leaf_v = (
                    latent_values(leaf_k)
                    if self.is_absorbed_mla
                    else torch.zeros_like(leaf_k)
                )
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
        value_dim = self.value_dim
        key_groups = d // self.settings.quant_group_size
        value_groups = value_dim // self.settings.quant_group_size
        token_groups = 16 // 16
        if self.settings.kv_bits == 4:
            quant_bits = 4
            key_quant_width = d // 2
            value_quant_width = value_dim // 2
            quant_dtype = torch.uint8
            quantized_leaf_k = torch.empty(
                r,
                h,
                self.leaf_capacity,
                key_quant_width,
                dtype=quant_dtype,
                device=self.device,
            )
            page_k_scales = torch.empty(
                r,
                h,
                self.page_capacity,
                token_groups * key_groups,
                dtype=self.dtype,
                device=self.device,
            )
            quantized_page_sum_k = torch.empty(
                r,
                h,
                self.page_capacity,
                d,
                dtype=torch.int8,
                device=self.device,
            )
            page_sum_k_scales = torch.empty(
                r,
                h,
                self.page_capacity,
                key_groups,
                dtype=self.dtype,
                device=self.device,
            )
            shared_quantized_latent = self.is_absorbed_mla
            page.update(
                leaf_quant_bits=quant_bits,
                leaf_k=(leaf_record := torch.empty(
                    r, h, 1, d, dtype=self.dtype, device=self.device
                )),
                leaf_v=(
                    latent_values(leaf_record)
                    if self.is_absorbed_mla
                    else torch.empty(
                        r, h, 1, value_dim, dtype=self.dtype, device=self.device
                    )
                ),
                quantized_leaf_k=quantized_leaf_k,
                quantized_leaf_v=(
                    quantized_leaf_k[..., :value_quant_width]
                    if shared_quantized_latent
                    else torch.empty(
                        r,
                        h,
                        self.leaf_capacity,
                        value_quant_width,
                        dtype=quant_dtype,
                        device=self.device,
                    )
                ),
                page_k_scales=page_k_scales,
                page_v_scales=(
                    page_k_scales[..., : token_groups * value_groups]
                    if shared_quantized_latent
                    else torch.empty(
                        r,
                        h,
                        self.page_capacity,
                        token_groups * value_groups,
                        dtype=self.dtype,
                        device=self.device,
                    )
                ),
                page_quantized_counts=torch.zeros(
                    r,
                    h,
                    self.page_capacity,
                    dtype=torch.int32,
                    device=self.device,
                ),
                page_sum_k=(page_sum_record := torch.empty(
                    r, h, 1, d, dtype=self.dtype, device=self.device
                )),
                page_sum_v=(
                    latent_values(page_sum_record)
                    if self.is_absorbed_mla
                    else torch.empty(
                        r, h, 1, value_dim, dtype=self.dtype, device=self.device
                    )
                ),
                quantized_page_sum_k=quantized_page_sum_k,
                quantized_page_sum_v=(
                    quantized_page_sum_k[..., :value_dim]
                    if shared_quantized_latent
                    else torch.empty(
                        r,
                        h,
                        self.page_capacity,
                        value_dim,
                        dtype=torch.int8,
                        device=self.device,
                    )
                ),
                page_sum_k_scales=page_sum_k_scales,
                page_sum_v_scales=(
                    page_sum_k_scales[..., :value_groups]
                    if shared_quantized_latent
                    else torch.empty(
                        r,
                        h,
                        self.page_capacity,
                        value_groups,
                        dtype=self.dtype,
                        device=self.device,
                    )
                ),
                shared_quantized_latent=shared_quantized_latent,
                quantization_finalized=True,
                summary_quantization_finalized=True,
            )
        else:
            recursive_leaf_k = torch.empty(
                r,
                h,
                self.leaf_capacity,
                d,
                dtype=self.dtype,
                device=self.device,
            )
            recursive_leaf_v = (
                latent_values(recursive_leaf_k)
                if self.is_absorbed_mla
                else torch.empty_like(recursive_leaf_k)
            )
            page_sum_k = torch.zeros(
                r,
                h,
                self.page_capacity,
                d,
                dtype=self.dtype,
                device=self.device,
            )
            page.update(
                leaf_quant_bits=0,
                leaf_k=recursive_leaf_k,
                leaf_v=recursive_leaf_v,
                page_sum_k=page_sum_k,
                page_sum_v=(
                    latent_values(page_sum_k)
                    if self.is_absorbed_mla
                    else torch.zeros(
                        r,
                        h,
                        self.page_capacity,
                        d,
                        dtype=self.dtype,
                        device=self.device,
                    )
                ),
                quantization_finalized=False,
                summary_quantization_finalized=False,
            )
        state["page_cache"] = page
        if unified_page1:
            page.update(
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
                page.update(
                    unified_page1_fixed_indices=unified_page1_fixed_indices,
                    unified_page1_fixed_leaf_owners=(unified_page1_fixed_leaf_owners),
                    unified_page1_fixed_slot_offsets=(unified_page1_fixed_slot_offsets),
                    unified_page1_fixed_lengths=unified_page1_fixed_lengths,
                )
        return state

    def reset(self, slot: int) -> None:
        if not 0 <= slot < self.max_requests:
            raise IndexError("vLLM request slot is outside the LOD pool")
        self.wait_deferred_prefill((slot,))
        if getattr(self, "kimi_captured_owner_decode", False) and slot == self.dcp_rank:
            self.owner_decode_pool.reset(0)
        getattr(self, "_kimi_request_owner_rows", {}).pop(slot, None)
        self.dcp_prefill_shadows.pop(slot, None)
        self.dcp_cross_layer_initial_sinks.pop(slot, None)
        self.ready[slot] = False
        self.clean[slot] = True
        self.unified_page1_fixed_dirty[slot] = False
        self.dcp_sharded[slot] = self.dcp_world_size == 1
        self.metadata[slot].clear()
        self.local_lens[slot].zero_()
        self.dcp_global_lens[slot].zero_()
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

    @staticmethod
    def _unique_storage_nbytes(value: object) -> int:
        """Count tensor backing stores once, including aliased arena views."""

        tensors: list[torch.Tensor] = []

        def collect(item: object) -> None:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    collect(child)

        collect(value)
        storages: dict[tuple[str, int | None, int], int] = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            identity = (
                tensor.device.type,
                tensor.device.index,
                int(storage.data_ptr()),
            )
            storages.setdefault(identity, int(storage.nbytes()))
        return sum(storages.values())

    def persistent_cache_nbytes(self) -> int:
        """Live bytes in graph-stable semantic cache rows for this layer."""

        return self._unique_storage_nbytes(self.state)

    def dcp_prefill_shadow_nbytes(self) -> int:
        """Live bytes in incomplete replicated DCP prompt caches."""

        return self._unique_storage_nbytes(
            {slot: cache.state for slot, cache in self.dcp_prefill_shadows.items()}
        )

    def _profile_dcp_cache_vram(self, event: str, slot: int) -> None:
        if os.environ.get("LOD_PROFILE_DCP_CACHE_VRAM") != "1":
            return
        persistent = self.persistent_cache_nbytes()
        shadow = self.dcp_prefill_shadow_nbytes()
        print(
            "LOD_DCP_CACHE_VRAM "
            f"event={event} rank={self.dcp_rank} slot={slot} "
            f"persistent_bytes={persistent} shadow_bytes={shadow} "
            f"live_bytes={persistent + shadow} "
            f"global_capacity={self.request_capacity} "
            f"local_capacity={self.persistent_request_capacity}",
            flush=True,
        )

    def _reset_range(self, start: int, stop: int) -> None:
        """Reset one contiguous row range with one launch per cache field."""
        if not 0 <= start < stop <= self.max_requests:
            raise IndexError("vLLM request row range is outside the LOD pool")
        self.wait_deferred_prefill(tuple(range(start, stop)))
        for slot in range(start, stop):
            getattr(self, "_kimi_request_owner_rows", {}).pop(slot, None)
            self.dcp_prefill_shadows.pop(slot, None)
            self.dcp_prefill_summaries.pop(slot, None)
            self.dcp_cross_layer_initial_sinks.pop(slot, None)
            self.ready[slot] = False
            self.clean[slot] = True
            self.unified_page1_fixed_dirty[slot] = False
            self.dcp_sharded[slot] = self.dcp_world_size == 1
            self.metadata[slot].clear()
        self.local_lens[start:stop].zero_()
        self.dcp_global_lens[start:stop].zero_()
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

    def _dcp_local_length(self, global_length: int) -> int:
        """Number of interleaved chronological tokens owned by this rank."""

        if self.dcp_world_size == 1:
            return int(global_length)
        length = max(0, int(global_length))
        block = self.dcp_interleave_size
        cycle = self.dcp_world_size * block
        complete, remainder = divmod(length, cycle)
        rank_begin = self.dcp_rank * block
        return complete * block + max(0, min(block, remainder - rank_begin))

    def _dcp_owns_position(self, position: int) -> bool:
        return (
            (int(position) // self.dcp_interleave_size) % self.dcp_world_size
            == self.dcp_rank
        )

    def _dcp_global_decode_coverage(self, global_length: int) -> int:
        """Paper decode boundary in global, per-request sequence coordinates."""

        length = max(0, int(global_length))
        chunk = int(self._dcp_global_lengths["chunk_len"])
        local = int(self._dcp_global_lengths["local_len"])
        bswa_end = ((length + 1 + chunk - 1) // chunk) * chunk
        bswa_begin = max(0, bswa_end - local)
        return min(length, max(min(length, chunk), bswa_begin))

    @contextmanager
    def _dcp_local_state_schedule(self):
        """Use one rank's exact share of the paper's global token schedule."""

        if self.dcp_world_size == 1:
            yield
            return
        old_growth = float(self.engine.state_growth_factor)
        old_minimum = int(self.engine.state_min_len)
        old_lengths = {
            name: int(getattr(self.engine, name))
            for name in self._dcp_global_lengths
        }
        independent_prefill = (
            getattr(self, "kimi_local_dcp_prefill", False)
            and not getattr(self, "kimi_shared_dcp_prefill", False)
        )
        self.engine.state_growth_factor = (
            self._dcp_global_state_growth_factor
            * (math.sqrt(self.dcp_world_size) if independent_prefill
               else 1.0 / math.sqrt(self.dcp_world_size))
        )
        self.engine.state_min_len = max(
            1,
            (self._dcp_global_state_min_len if independent_prefill
             else math.ceil(self._dcp_global_state_min_len / self.dcp_world_size)),
        )
        for name, global_length in self._dcp_global_lengths.items():
            setattr(
                self.engine,
                name,
                max(1, math.ceil(global_length / self.dcp_world_size)),
            )
        try:
            yield
        finally:
            self.engine.state_growth_factor = old_growth
            self.engine.state_min_len = old_minimum
            for name, old_length in old_lengths.items():
                setattr(self.engine, name, old_length)

    def ensure_dcp_sharded(self, slots: tuple[int, ...]) -> None:
        """Convert completed replicated-prefill rows into rank-local LoD rows.

        Prefill keeps the ordinary per-head LoD calculation.  At the first
        decode step, each rank rebuilds its persistent semantic cache from the
        exact chronological BF16 archive using vLLM's DCP token ownership.
        This makes subsequent routing, leaf refinement, and state updates
        genuinely local without moving leaves between ranks.
        """

        if self.dcp_world_size == 1:
            return
        for slot in slots:
            if self.dcp_sharded[slot]:
                continue
            shadow = self.dcp_prefill_shadows.get(slot)
            if shadow is not None:
                self._install_dcp_local_from_cache(slot, shadow)
                continue
            if not self.ready[slot]:
                # Graph/profile rows carry no semantic prompt.
                self.dcp_sharded[slot] = True
                continue
            self.wait_deferred_prefill((slot,))
            self._install_dcp_local_from_cache(slot, self._row_cache(slot))

    def has_prefill_cache(self, slot: int) -> bool:
        """Whether a row has either temporary prefill or persistent state."""

        return self.ready[slot] or slot in self.dcp_prefill_shadows

    def _retain_dcp_prefill_shadow(
        self, slot: int, cache: KernelLODCache
    ) -> None:
        """Retain one replicated cache only for an incomplete DCP prompt."""

        if self.dcp_world_size == 1:
            raise RuntimeError("non-DCP requests do not use prefill shadows")
        source = cache.state
        page = source.get("page_cache")
        if not isinstance(page, dict):
            raise TypeError("DCP prefill shadow has no semantic page archive")
        self.dcp_prefill_shadows[slot] = cache
        self.metadata[slot].update(
            state_len=int(source["state_len"]),
            scheduled_state_len=int(
                source.get("scheduled_state_len", source["state_len"])
            ),
            coverage=int(source["coverage"]),
            total_len=int(source["total_len"]),
            recent_len=int(source["recent_len"]),
            leaf_count=int(page["leaf_count"]),
            overflow_safe_until=int(page["overflow_safe_until"]),
        )
        self.ready[slot] = False
        self.dcp_sharded[slot] = False
        self._profile_dcp_cache_vram("prefill-shadow", slot)

    def _install_dcp_local_from_cache(
        self,
        slot: int,
        source_cache: KernelLODCache,
        *,
        source_slot: int = 0,
    ) -> None:
        """Convert one replicated chronological archive into its local row."""

        local_k, local_v, global_length = self._dcp_local_records(
            source_cache, source_slot=source_slot
        )
        if global_length <= 0:
            self.dcp_prefill_shadows.pop(slot, None)
            self.dcp_sharded[slot] = True
            return

        # Do not let the temporary full-prompt capacity override leak into the
        # local rebuild.  The fixed pool already reserves the maximum local
        # schedule and archive sizes required by this rank.
        old_capacity = getattr(self.engine, "_lod_prefill_cache_capacity", None)
        if old_capacity is not None:
            del self.engine._lod_prefill_cache_capacity
        try:
            with self._dcp_local_state_schedule():
                global_coverage = self._dcp_global_decode_coverage(global_length)
                converted = self.engine.build_cache_from_bf16(
                    local_k,
                    local_v,
                    finalize_cache_for_decode=True,
                    final_cache_coverage=self._dcp_local_length(global_coverage),
                )
        finally:
            if old_capacity is not None:
                self.engine._lod_prefill_cache_capacity = old_capacity
        self._install_dcp_converted_row(
            slot, converted, global_length=global_length
        )

    def _dcp_local_records(
        self,
        source_cache: KernelLODCache,
        *,
        source_slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Extract this rank's chronological raw MLA records from a shadow."""

        source = source_cache.state
        page = source.get("page_cache")
        if not isinstance(page, dict):
            raise TypeError("DCP conversion requires a semantic page archive")
        if page.get("dcp_leaf_sharded"):
            from .models.kimi_k3_sharded_prefill import owned_history

            return owned_history(self, source)
        global_length = int(source["total_len"])
        leaf_k = page.get("leaf_k")
        leaf_v = page.get("leaf_v")
        if global_length <= 0:
            if not isinstance(leaf_k, torch.Tensor) or not isinstance(
                leaf_v, torch.Tensor
            ):
                raise TypeError("empty DCP shadow has no leaf storage")
            empty_k = leaf_k[source_slot : source_slot + 1, :, :0, :]
            empty_v = leaf_v[source_slot : source_slot + 1, :, :0, :]
            return empty_k, empty_v, global_length
        if (self.is_absorbed_mla and self.dcp_interleave_size == 1
                and not bool(page.get("quantization_finalized", False))):
            from .models.kimi_k3_sharded_prefill import owned_bf16_shadow_records

            # Include the authoritative exact tail rather than reading the
            # archive's reserved, not-yet-populated suffix.
            row_source = dict(source)
            row_source["page_cache"] = dict(page, leaf_k=leaf_k[source_slot:source_slot + 1])
            for name in ("sink_k", "recent_k"):
                if isinstance(source.get(name), torch.Tensor):
                    row_source[name] = source[name][source_slot:source_slot + 1]
            return owned_bf16_shadow_records(self, row_source)
        positions = torch.arange(global_length, dtype=torch.long, device=self.device)
        ownership = (
            (positions // self.dcp_interleave_size) % self.dcp_world_size
        ) == self.dcp_rank
        owned_positions = positions[ownership]
        local_length = int(owned_positions.numel())
        if local_length != self._dcp_local_length(global_length):
            raise AssertionError("DCP local-length calculation diverged")

        sink_k = source.get("sink_k")
        sink_v = source.get("sink_v")
        sink_len = int(sink_k.size(2)) if isinstance(sink_k, torch.Tensor) else 0
        if bool(page.get("quantization_finalized", False)):
            if not self.is_absorbed_mla:
                raise NotImplementedError(
                    "quantized DCP conversion currently requires absorbed MLA"
                )
            required = (
                "page_indices",
                "page_counts",
                "next_page",
                "quantized_leaf_k",
                "page_k_scales",
                "quantized_page_sum_k",
                "page_sum_k_scales",
            )
            values = tuple(page.get(name) for name in required)
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise RuntimeError("quantized DCP shadow is incomplete")
            local_k = dequantize_owned_virtual_paged_keys(
                *values,
                source_slot=source_slot,
                sink_len=sink_len,
                local_length=local_length,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                dcp_interleave_size=self.dcp_interleave_size,
            )
            coverage = int(source["coverage"])
            recent_k = source.get("recent_k")
            recent_len = int(source.get("recent_len", global_length - coverage))
            recent_owned = owned_positions >= coverage
            if bool(recent_owned.any()):
                if not isinstance(recent_k, torch.Tensor):
                    raise RuntimeError("quantized DCP shadow lost its exact tail")
                recent_positions = (owned_positions[recent_owned] - coverage).long()
                if int(recent_positions.max().item()) >= recent_len:
                    raise AssertionError("DCP exact-tail position is out of range")
                local_k[..., recent_owned, :].copy_(
                    recent_k[
                        source_slot : source_slot + 1, :, recent_positions, :
                    ]
                )
            local_v = local_k[..., : self.value_dim]
        else:
            if not isinstance(leaf_k, torch.Tensor) or not isinstance(
                leaf_v, torch.Tensor
            ):
                raise TypeError("BF16 DCP shadow has no leaf storage")
            archive_positions = torch.clamp(owned_positions - sink_len, min=0)
            local_k = leaf_k[
                source_slot : source_slot + 1, :, archive_positions, :
            ].clone()
            local_v = (
                local_k[..., : self.value_dim]
                if self.is_absorbed_mla
                else leaf_v[
                    source_slot : source_slot + 1, :, archive_positions, :
                ].clone()
            )
        owned_sink = owned_positions < sink_len
        if bool(owned_sink.any()):
            if not isinstance(sink_k, torch.Tensor) or not isinstance(
                sink_v, torch.Tensor
            ):
                raise RuntimeError("DCP rank lost the protected sink")
            sink_positions = owned_positions[owned_sink].long()
            local_k[..., owned_sink, :].copy_(
                sink_k[source_slot : source_slot + 1, :, sink_positions, :]
            )
            local_v[..., owned_sink, :].copy_(
                sink_v[source_slot : source_slot + 1, :, sink_positions, :]
            )

        return local_k, local_v, global_length

    def _install_dcp_converted_row(
        self,
        slot: int,
        converted: KernelLODCache,
        *,
        global_length: int,
        source_slot: int = 0,
    ) -> None:
        """Install one row from a layer-batched local DCP cache build."""

        self.install(slot, converted, source_slot=source_slot)
        global_coverage = self._dcp_global_decode_coverage(global_length)
        expected_local_coverage = self._dcp_local_length(global_coverage)
        observed_local_coverage = int(self.metadata[slot]["coverage"])
        if observed_local_coverage != expected_local_coverage:
            raise AssertionError(
                "DCP cache construction diverged from its global boundary: "
                f"global_total={global_length}, global_coverage={global_coverage}, "
                f"local_expected={expected_local_coverage}, "
                f"local_observed={observed_local_coverage}"
            )
        self.metadata[slot]["dcp_global_total_len"] = global_length
        self.metadata[slot]["dcp_global_coverage"] = global_coverage
        self.dcp_global_lens[slot].fill_(global_length)
        self.dcp_sharded[slot] = True
        self.dcp_prefill_shadows.pop(slot, None)
        self.engine.reset_runtime_cache()
        self._profile_dcp_cache_vram("post-prefill-local", slot)

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
            active_state_len = max(
                int(self.metadata[slot].get("state_len", 0))
                for slot in range(start_slot, stop_slot)
            )
            if self.is_absorbed_mla:
                materialize_absorbed_mla_coarse_means(
                    self.state["state_k"][start_slot:stop_slot],
                    self.state["counts"][start_slot:stop_slot],
                    coarse_k[start_slot:stop_slot],
                    coarse_bias[start_slot:stop_slot],
                    active_state_len=active_state_len,
                )
            else:
                coarse_v = arena_v[
                    coarse_offset : coarse_offset
                    + self.max_requests * self.kv_heads * self.state_capacity
                ].view_as(coarse_k)
                materialize_page1_coarse_means(
                    self.state["state_k"][start_slot:stop_slot],
                    self.state["state_v"][start_slot:stop_slot],
                    self.state["counts"][start_slot:stop_slot],
                    coarse_k[start_slot:stop_slot],
                    coarse_v[start_slot:stop_slot],
                    coarse_bias[start_slot:stop_slot],
                    active_state_len=active_state_len,
                )
            begin = end
        if self.settings.decode_gqa_fixed_mask_aiter:
            self._refresh_unified_page1_fixed(slots)

    def ensure_unified_page1_fixed(self, slots: tuple[int, ...]) -> None:
        """Rebuild invalidated compact-decode leaf tables once per update."""
        dirty = tuple(slot for slot in slots if self.unified_page1_fixed_dirty[slot])
        if not dirty:
            return
        self._refresh_unified_page1_fixed(dirty)
        for slot in dirty:
            self.unified_page1_fixed_dirty[slot] = False

    def _refresh_unified_page1_fixed(self, slots: tuple[int, ...]) -> None:
        """Rebuild the persistent physical-leaf list after a state update."""
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
                local_limit=self.decode_local_limit,
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
            self.leaf_lens.index_fill_(0, slot_indices, int(source_page["leaf_count"]))
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
        if self.dcp_world_size > 1:
            # The fixed rows are rank-local.  Initial DCP prefill instead uses
            # a temporary globally replicated cache until its final chunk.
            return None
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

    def _stage_cross_layer_initial_cache(
        self,
        slots: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        coverage: int,
        prompt_capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Persist or stage one exact prefix before its layer-batched update."""

        if not slots or int(key.size(0)) != len(slots):
            raise ValueError("cross-layer initial construction has invalid rows")
        if self.dcp_world_size > 1:
            total_len = int(key.size(2))
            sink_len = min(int(self.engine.sink_len), total_len)
            # DCP's persistent rows are rank-local and cannot hold the
            # globally replicated prefill archive.  Stage only one detached
            # chronological source per attention layer; the runtime batches
            # centroid construction across layers and turns these sources into
            # temporary global shadows on its background stream.
            archive_k = key[..., sink_len:total_len, :].detach().clone()
            if self.is_absorbed_mla:
                archive_v = archive_k[..., : self.value_dim]
                staged_sink_k = key[..., :sink_len, :].detach().clone()
                staged_sink_v = staged_sink_k[..., : self.value_dim]
            else:
                archive_v = value[..., sink_len:total_len, :].detach().clone()
                staged_sink_k = key[..., :sink_len, :].detach().clone()
                staged_sink_v = value[..., :sink_len, :].detach().clone()
            for source_row, slot in enumerate(slots):
                self.dcp_cross_layer_initial_sinks[slot] = (
                    staged_sink_k[source_row : source_row + 1],
                    staged_sink_v[source_row : source_row + 1],
                )
            if prompt_capacity < total_len or prompt_capacity > self.request_capacity:
                raise ValueError("DCP prompt capacity is outside the configured range")
            return archive_k, archive_v
        storage = self._initial_prefill_storage(slots)
        if storage is None:
            raise RuntimeError("cross-layer initial construction needs pool storage")
        page = storage.get("page_cache")
        if not isinstance(page, dict):
            raise TypeError("cross-layer initial construction lacks its page cache")

        total_len = int(key.size(2))
        initial_len = min(total_len, int(self.engine.chunk_len))
        sink_len = min(int(self.engine.sink_len), initial_len)
        initial_state_len = initial_len - sink_len
        archive_len = total_len - sink_len
        if not 0 <= coverage - sink_len <= archive_len:
            raise ValueError("cross-layer initial coverage is outside the prefix")

        staged_leaves = None
        if self.settings.kv_bits == 4:
            # The persistent INT4 cache has no BF16 leaf shadow. A detached
            # archive keeps only K/V, rather than pinning the much larger QKV
            # projection allocation while a bounded layer group is collected.
            archive_k = key[..., sink_len:total_len, :].detach().clone()
            archive_v = value[..., sink_len:total_len, :].detach().clone()
            staged_leaves = (archive_k, archive_v)
        elif self.settings.kv_bits == 0:
            leaf_k = page.get("leaf_k")
            leaf_v = page.get("leaf_v")
            if not isinstance(leaf_k, torch.Tensor) or not isinstance(
                leaf_v, torch.Tensor
            ):
                raise TypeError("cross-layer initial construction needs BF16 leaves")
            if archive_len > int(leaf_k.size(2)) or archive_len > int(leaf_v.size(2)):
                raise ValueError("cross-layer initial prefix exceeds leaf capacity")
            # BF16 modes stage directly in their final chronological archive.
            leaf_k[..., :archive_len, :].copy_(key[..., sink_len:total_len, :])
            leaf_v[..., :archive_len, :].copy_(value[..., sink_len:total_len, :])
            archive_k = leaf_k[..., :archive_len, :]
            archive_v = leaf_v[..., :archive_len, :]
        else:
            raise ValueError("cross-layer construction supports BF16 or INT4 leaves")

        if sink_len:
            sink_k = storage.get("sink_k")
            sink_v = storage.get("sink_v")
            if not isinstance(sink_k, torch.Tensor) or not isinstance(
                sink_v, torch.Tensor
            ):
                raise TypeError("cross-layer initial construction lacks its sink")
            sink_k.copy_(key[..., :sink_len, :])
            sink_v.copy_(value[..., :sink_len, :])

        state_k = storage["state_k"]
        state_v = storage["state_v"]
        counts = storage["counts"]
        if not all(
            isinstance(tensor, torch.Tensor) for tensor in (state_k, state_v, counts)
        ):
            raise TypeError("cross-layer initial state storage is incomplete")
        state_k[..., :initial_state_len, :].copy_(archive_k[..., :initial_state_len, :])
        state_v[..., :initial_state_len, :].copy_(archive_v[..., :initial_state_len, :])
        counts[..., :initial_state_len, :].fill_(1.0)
        key_norm_sums = storage.get("key_norm_sums")
        if key_norm_sums is not None:
            if not isinstance(key_norm_sums, torch.Tensor):
                raise TypeError("cross-layer key-norm storage is invalid")
            key_norm_sums[..., :initial_state_len, :].copy_(
                self.engine._state_clustering_constituent_rms(
                    state_k[..., :initial_state_len, :]
                )
            )

        recent_len = total_len - coverage
        recent_k = storage["recent_k"]
        recent_v = storage["recent_v"]
        if not isinstance(recent_k, torch.Tensor) or not isinstance(
            recent_v, torch.Tensor
        ):
            raise TypeError("cross-layer recent storage is incomplete")
        if recent_len > int(recent_k.size(2)):
            raise ValueError("cross-layer exact tail exceeds its fixed storage")
        archive_coverage = coverage - sink_len
        recent_k[..., :recent_len, :].copy_(
            archive_k[..., archive_coverage:archive_len, :]
        )
        recent_v[..., :recent_len, :].copy_(
            archive_v[..., archive_coverage:archive_len, :]
        )
        return staged_leaves

    def _finish_cross_layer_initial_cache(
        self,
        slots: tuple[int, ...],
        *,
        total_len: int,
        coverage: int,
        state_len: int,
        owners: torch.Tensor,
        owner_ranks: torch.Tensor,
        staged_leaves: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Create per-layer page ownership after a batched centroid update."""

        storage = self._initial_prefill_storage(slots)
        if storage is None:
            raise RuntimeError("cross-layer initial construction lost pool storage")
        page = storage.get("page_cache")
        if not isinstance(page, dict):
            raise TypeError("cross-layer initial construction lacks its page cache")
        if staged_leaves is None:
            archive_k = page.get("leaf_k")
            archive_v = page.get("leaf_v")
            if not isinstance(archive_k, torch.Tensor) or not isinstance(
                archive_v, torch.Tensor
            ):
                raise TypeError("cross-layer initial construction needs BF16 leaves")
        else:
            if self.settings.kv_bits != 4:
                raise ValueError("only INT4 construction may use staged BF16 leaves")
            archive_k, archive_v = staged_leaves

        sink_len = min(int(self.engine.sink_len), total_len)
        initial_len = min(total_len, int(self.engine.chunk_len))
        initial_state_len = initial_len - sink_len
        archived_len = coverage - sink_len
        archive_len = total_len - sink_len
        if int(archive_k.size(2)) < archive_len or int(archive_v.size(2)) < archive_len:
            raise ValueError("cross-layer staged archive is shorter than its prefix")
        initial_owners = (
            torch.arange(initial_state_len, device=owners.device, dtype=torch.long)
            .view(1, 1, initial_state_len)
            .expand(len(slots), self.kv_heads, initial_state_len)
        )
        if initial_state_len + int(owners.size(2)) != archived_len:
            raise AssertionError("cross-layer owner archive has the wrong length")

        sequence_capacity = _round_up(total_len, int(self.engine.chunk_len)) + max(
            int(self.engine.chunk_len), int(self.engine.decode_cache_headroom)
        )
        page_cache = self.engine._new_page_cache(
            archive_k[..., :initial_state_len, :],
            archive_v[..., :initial_state_len, :],
            initial_owners,
            state_capacity=self.state_capacity,
            sequence_capacity=sequence_capacity,
            virtual_k=archive_k[..., :archive_len, :],
            virtual_v=archive_v[..., :archive_len, :],
            destination=page,
        )
        self.engine._append_page_cache(
            page_cache,
            archive_k[..., initial_state_len:archived_len, :],
            archive_v[..., initial_state_len:archived_len, :],
            owners.long(),
            owner_ranks=owner_ranks.long(),
        )
        if staged_leaves is not None:
            self.engine._finalize_virtual_page_quantization(
                page_cache,
                destination_page=page,
            )
        recent_len = total_len - coverage
        state: dict[str, object] = {
            "state_k": storage["state_k"],
            "state_v": storage["state_v"],
            "counts": storage["counts"],
            "state_len": state_len,
            "scheduled_state_len": state_len,
            "coverage": coverage,
            "state_capacity": self.state_capacity,
            "recent_k": storage["recent_k"],
            "recent_v": storage["recent_v"],
            "recent_len": recent_len,
            "total_len": total_len,
            "page_cache": page_cache,
            "pool_backed": True,
        }
        if "sink_k" in storage:
            state["sink_k"] = storage["sink_k"]
            state["sink_v"] = storage["sink_v"]
        if "key_norm_sums" in storage:
            state["key_norm_sums"] = storage["key_norm_sums"]
        self.install_rows(slots, KernelLODCache(state))
        self.engine.reset_runtime_cache()

    def _finish_dcp_cross_layer_initial_cache(
        self,
        slots: tuple[int, ...],
        *,
        total_len: int,
        coverage: int,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        key_norm_sums: torch.Tensor | None,
        state_len: int,
        owners: torch.Tensor,
        owner_ranks: torch.Tensor,
        archive_k: torch.Tensor,
        archive_v: torch.Tensor,
        prompt_capacity: int,
    ) -> None:
        """Install one globally replicated DCP shadow after a batched update."""

        if self.dcp_world_size <= 1:
            raise RuntimeError("DCP shadow construction requires DCP")
        sinks = [self.dcp_cross_layer_initial_sinks.pop(slot, None) for slot in slots]
        if any(sink is None for sink in sinks):
            raise RuntimeError("DCP cross-layer construction lost its sink")
        sink_k = torch.cat([sink[0] for sink in sinks if sink is not None], dim=0)
        sink_v = (
            sink_k[..., : self.value_dim]
            if self.is_absorbed_mla
            else torch.cat([sink[1] for sink in sinks if sink is not None], dim=0)
        )
        sink_len = int(sink_k.size(2))
        initial_len = min(total_len, int(self.engine.chunk_len))
        initial_state_len = initial_len - sink_len
        archived_len = coverage - sink_len
        archive_len = total_len - sink_len
        if int(archive_k.size(2)) != archive_len or int(archive_v.size(2)) != archive_len:
            raise ValueError("DCP staged archive has the wrong length")
        initial_owners = (
            torch.arange(
                initial_state_len, device=owners.device, dtype=torch.long
            )
            .view(1, 1, initial_state_len)
            .expand(len(slots), self.kv_heads, initial_state_len)
        )
        if initial_state_len + int(owners.size(2)) != archived_len:
            raise AssertionError("DCP owner archive has the wrong length")
        sequence_capacity, capacity_limit, growth_chunk = _dcp_prefill_archive_capacity(
            total_len=total_len,
            prompt_capacity=prompt_capacity,
            chunk_len=int(self.engine.chunk_len),
            headroom=int(self.engine.decode_cache_headroom),
        )
        if self.kimi_sharded_leaf_prefill:
            from .models.kimi_k3_sharded_prefill import initial_page

            page_cache = initial_page(
                self, archive_k, torch.cat((initial_owners, owners), dim=2),
                sink_len=sink_len, coverage=coverage,
                prompt_capacity=prompt_capacity, state_capacity=int(state_k.size(2)),
            )
        else:
            page_cache = self.engine._new_page_cache(
                archive_k[..., :initial_state_len, :],
                archive_v[..., :initial_state_len, :],
                initial_owners,
                state_capacity=int(state_k.size(2)),
                sequence_capacity=sequence_capacity,
                virtual_k=archive_k,
                virtual_v=archive_v,
            )
        if growth_chunk and not self.kimi_sharded_leaf_prefill:
            page_cache["leaf_growth_chunk"] = growth_chunk
            page_cache["leaf_capacity_limit"] = capacity_limit
        if not self.kimi_sharded_leaf_prefill:
            self.engine._append_page_cache(
                page_cache,
                archive_k[..., initial_state_len:archived_len, :],
                archive_v[..., initial_state_len:archived_len, :],
                owners.long(),
                owner_ranks=owner_ranks.long(),
            )
        if int(self.settings.kv_bits) == 4:
            # The replicated DCP shadow can live for many scheduler chunks.
            # Quantize it immediately, append later chunks directly into the
            # residual-INT4 pages, and reconstruct only this rank's records at
            # the final global-to-local conversion.
            self.engine._finalize_virtual_page_quantization(page_cache)
        recent_len = total_len - coverage
        recent_capacity = max(
            recent_len,
            int(self.engine.local_len) + int(self.engine.decode_state_update_len),
            int(self.engine.chunk_len),
        )
        recent_k = archive_k.new_empty(
            len(slots), self.kv_heads, recent_capacity, int(archive_k.size(-1))
        )
        recent_v = (
            recent_k[..., : self.value_dim]
            if self.is_absorbed_mla
            else archive_v.new_empty(
                len(slots),
                self.kv_heads,
                recent_capacity,
                int(archive_v.size(-1)),
            )
        )
        recent_k[..., :recent_len, :].copy_(archive_k[..., archived_len:, :])
        recent_v[..., :recent_len, :].copy_(archive_v[..., archived_len:, :])
        state: dict[str, object] = {
            "state_k": state_k,
            "state_v": state_v,
            "counts": counts,
            "state_len": state_len,
            "scheduled_state_len": state_len,
            "coverage": coverage,
            "state_capacity": int(state_k.size(2)),
            "recent_k": recent_k,
            "recent_v": recent_v,
            "recent_len": recent_len,
            "total_len": total_len,
            "sink_k": sink_k,
            "sink_v": sink_v,
            "page_cache": page_cache,
        }
        if key_norm_sums is not None:
            state["key_norm_sums"] = key_norm_sums
        cache = KernelLODCache(state)
        for source_row, slot in enumerate(slots):
            self._retain_dcp_prefill_shadow(
                slot, self._shadow_row_view(cache, source_row)
            )
        self.engine.reset_runtime_cache()

    @staticmethod
    def _shadow_row_view(cache: KernelLODCache, row: int) -> KernelLODCache:
        """Return a request-owned view of one row from a batched shadow."""

        state = cache.state
        state_k = state.get("state_k")
        if not isinstance(state_k, torch.Tensor):
            raise TypeError("batched DCP shadow lacks its state tensor")
        batch = int(state_k.size(0))
        if not 0 <= row < batch:
            raise IndexError("DCP shadow row is out of range")

        def row_value(value: object) -> object:
            if isinstance(value, torch.Tensor) and value.ndim and int(
                value.size(0)
            ) == batch:
                return value[row : row + 1]
            if isinstance(value, dict):
                return {name: row_value(item) for name, item in value.items()}
            return value

        view = KernelLODCache(
            {name: row_value(value) for name, value in state.items()}
        )
        # Keep enough identity to reassemble an equal-length request batch on
        # its next scheduler chunk without concatenating (and therefore
        # copying) the full chronological leaf archives.
        view._lod_batched_parent = cache  # type: ignore[attr-defined]
        view._lod_batched_row = row  # type: ignore[attr-defined]
        return view

    def _batched_dcp_shadow(
        self, slots: tuple[int, ...]
    ) -> KernelLODCache | None:
        """Return the shared zero-copy shadow when ``slots`` cover its rows."""

        if not slots:
            return None
        shadows = [self.dcp_prefill_shadows.get(slot) for slot in slots]
        if any(shadow is None for shadow in shadows):
            return None
        parents = [
            getattr(shadow, "_lod_batched_parent", None) for shadow in shadows
        ]
        parent = parents[0]
        if parent is None or any(item is not parent for item in parents):
            return None
        rows = tuple(
            int(getattr(shadow, "_lod_batched_row", -1)) for shadow in shadows
        )
        state_k = parent.state.get("state_k")
        if not isinstance(state_k, torch.Tensor):
            return None
        if rows != tuple(range(int(state_k.size(0)))):
            return None
        return parent

    def _retain_batched_dcp_shadow(
        self, slots: tuple[int, ...], cache: KernelLODCache
    ) -> None:
        """Publish row views after one batched replicated-cache update."""

        for source_row, slot in enumerate(slots):
            self._retain_dcp_prefill_shadow(
                slot, self._shadow_row_view(cache, source_row)
            )

    def _finish_dcp_cross_layer_cached_cache(
        self,
        slots: tuple[int, ...],
        *,
        total_len: int,
        coverage: int,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        key_norm_sums: torch.Tensor | None,
        state_len: int,
        scheduled_state_len: int,
        owners: torch.Tensor,
        owner_ranks: torch.Tensor,
        staged_k: torch.Tensor,
        staged_v: torch.Tensor,
    ) -> None:
        """Advance a replicated DCP shadow after one routed prefill block."""

        if not slots:
            raise ValueError("DCP cached construction requires request rows")
        shadow = (
            self._batched_dcp_shadow(slots)
            if len(slots) > 1
            else self.dcp_prefill_shadows.get(slots[0])
        )
        if shadow is None:
            raise RuntimeError("DCP cached construction lost its batched shadow")
        source = shadow.state
        page_cache = source.get("page_cache")
        if not isinstance(page_cache, dict):
            raise TypeError("DCP cached construction lacks its page cache")
        old_coverages = {int(self.metadata[slot]["coverage"]) for slot in slots}
        if len(old_coverages) != 1:
            raise RuntimeError("DCP cached request coverages diverged")
        old_coverage = old_coverages.pop()
        sink_len = min(int(self.engine.sink_len), total_len)
        archive_begin = old_coverage - sink_len
        archive_end = coverage - sink_len
        overflow_len = archive_end - archive_begin
        recent_len = total_len - coverage
        if int(owners.size(2)) != overflow_len:
            raise AssertionError("DCP cached owner archive has the wrong length")
        required_len = overflow_len + recent_len
        if int(staged_k.size(2)) < required_len or int(
            staged_v.size(2)
        ) < required_len:
            raise ValueError("DCP cached staged source has the wrong length")
        overflow_k = staged_k[..., :overflow_len, :]
        overflow_v = staged_v[..., :overflow_len, :]
        if page_cache.get("dcp_leaf_sharded"):
            from .models.kimi_k3_sharded_prefill import append_page

            append_page(self, page_cache, staged_k, owners,
                        previous_coverage=old_coverage, coverage=coverage, total_len=total_len,
                        owner_ranks=owner_ranks.long())
        else:
            self.engine._append_page_cache(
                page_cache,
                overflow_k,
                overflow_v,
                owners.long(),
                owner_ranks=owner_ranks.long(),
            )
        recent_k = source.get("recent_k")
        recent_v = source.get("recent_v")
        if not isinstance(recent_k, torch.Tensor) or not isinstance(
            recent_v, torch.Tensor
        ):
            raise TypeError("DCP cached construction lacks its exact tail")
        if recent_len > int(recent_k.size(2)):
            raise ValueError("DCP cached exact tail exceeds its storage")
        # This aligned cached-prefill update consumes the whole scheduler
        # chunk into state, so the remaining exact tail is the suffix after
        # ``overflow_len``.  Keep raw MLA records here; page insertion applies
        # the per-token key normalization required by exact leaf attention.
        recent_k[..., :recent_len, :].copy_(
            staged_k[..., overflow_len:required_len, :]
        )
        recent_v[..., :recent_len, :].copy_(
            staged_v[..., overflow_len:required_len, :]
        )
        source.update(
            state_k=state_k,
            state_v=state_v,
            counts=counts,
            state_len=state_len,
            scheduled_state_len=scheduled_state_len,
            coverage=coverage,
            state_capacity=int(state_k.size(2)),
            recent_k=recent_k,
            recent_v=recent_v,
            recent_len=recent_len,
            total_len=total_len,
        )
        if key_norm_sums is not None:
            source["key_norm_sums"] = key_norm_sums
        if len(slots) > 1:
            self._retain_batched_dcp_shadow(slots, shadow)
        else:
            self._retain_dcp_prefill_shadow(slots[0], shadow)
        self.engine.reset_runtime_cache()

    def _stage_cross_layer_cached_cache(
        self,
        slots: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        previous_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Persist or stage an aligned continuation for a batched update."""

        if not slots or int(key.size(0)) != len(slots):
            raise ValueError("cross-layer cached construction has invalid rows")
        if any(
            int(self.metadata[slot]["total_len"]) != previous_len
            for slot in slots
        ):
            raise RuntimeError("cross-layer cached prefix length changed while staging")

        if self.dcp_world_size > 1:
            shadow = (
                self._batched_dcp_shadow(slots)
                if len(slots) > 1
                else self.dcp_prefill_shadows.get(slots[0])
            )
            if shadow is None:
                raise RuntimeError(
                    "cross-layer DCP continuation lost its batched shadow"
                )
            source = shadow.state
            recent_k = source.get("recent_k")
            recent_v = source.get("recent_v")
            recent_len = int(source.get("recent_len", 0))
            if not isinstance(recent_k, torch.Tensor) or not isinstance(
                recent_v, torch.Tensor
            ):
                raise TypeError("cross-layer DCP continuation lacks its exact tail")
            # [old exact tail, new chunk] contains the next overflow followed
            # by the new exact tail. Preserve raw MLA records for centroid
            # construction; page insertion normalizes only the key view.
            staged_k = torch.cat(
                (recent_k[..., :recent_len, :], key.detach()), dim=2
            )
            staged_v = (
                staged_k[..., : self.value_dim]
                if self.is_absorbed_mla
                else torch.cat(
                    (recent_v[..., :recent_len, :], value.detach()), dim=2
                )
            )
            return staged_k, staged_v

        if self.settings.kv_bits == 4:
            if len(slots) != 1:
                raise ValueError("non-DCP INT4 construction requires one row")
            slot = slots[0]
            metadata = self.metadata[slot]
            recent_len = int(metadata["recent_len"])
            recent_k = self.state["recent_k"][slot : slot + 1]
            recent_v = self.state["recent_v"][slot : slot + 1]
            if recent_len > int(recent_k.size(2)):
                raise ValueError("cross-layer cached tail exceeds fixed storage")
            # [old exact tail, new chunk] contains the next overflow followed
            # by the exact tail needed after this scheduler step.
            staged_k = torch.cat((recent_k[..., :recent_len, :], key.detach()), dim=2)
            staged_v = torch.cat((recent_v[..., :recent_len, :], value.detach()), dim=2)
            return staged_k, staged_v
        if self.settings.kv_bits != 0:
            raise ValueError("cross-layer construction supports BF16 or INT4 leaves")
        if len(slots) != 1:
            raise ValueError("non-DCP BF16 construction requires one row")
        slot = slots[0]

        page = self.state.get("page_cache")
        if not isinstance(page, dict):
            raise TypeError("cross-layer cached construction lacks its page cache")
        leaf_k = page.get("leaf_k")
        leaf_v = page.get("leaf_v")
        if not isinstance(leaf_k, torch.Tensor) or not isinstance(leaf_v, torch.Tensor):
            raise TypeError("cross-layer cached construction needs BF16 leaves")
        sink_len = min(int(self.engine.sink_len), previous_len)
        archive_begin = previous_len - sink_len
        archive_end = archive_begin + int(key.size(2))
        if archive_end > int(leaf_k.size(2)):
            raise ValueError("cross-layer cached continuation exceeds leaf capacity")
        leaf_k[slot : slot + 1, :, archive_begin:archive_end, :].copy_(key)
        leaf_v[slot : slot + 1, :, archive_begin:archive_end, :].copy_(value)
        return None

    def _finish_cross_layer_cached_cache(
        self,
        slot: int,
        *,
        total_len: int,
        coverage: int,
        state_len: int,
        scheduled_state_len: int,
        owners: torch.Tensor,
        owner_ranks: torch.Tensor,
        staged_leaves: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Finish page ownership and the exact tail after a batched update."""

        metadata = self.metadata[slot]
        old_coverage = int(metadata["coverage"])
        sink_len = min(int(self.engine.sink_len), total_len)
        archive_begin = old_coverage - sink_len
        archive_end = coverage - sink_len
        overflow_len = archive_end - archive_begin
        recent_len = total_len - coverage
        page_cache = self._row_cache(slot).state["page_cache"]
        if not isinstance(page_cache, dict):
            raise TypeError("cross-layer cached construction lost its page cache")
        if int(owners.size(2)) != overflow_len:
            raise AssertionError(
                "cross-layer cached owner archive has the wrong length"
            )

        if staged_leaves is None:
            leaf_k = page_cache.get("leaf_k")
            leaf_v = page_cache.get("leaf_v")
            if not isinstance(leaf_k, torch.Tensor) or not isinstance(
                leaf_v, torch.Tensor
            ):
                raise TypeError("cross-layer cached construction needs BF16 leaves")
            overflow_k = leaf_k[..., archive_begin:archive_end, :]
            overflow_v = leaf_v[..., archive_begin:archive_end, :]
            recent_begin = coverage - sink_len
            recent_end = total_len - sink_len
            recent_source_k = leaf_k[..., recent_begin:recent_end, :]
            recent_source_v = leaf_v[..., recent_begin:recent_end, :]
        else:
            if self.settings.kv_bits != 4:
                raise ValueError("only INT4 construction may use staged BF16 leaves")
            working_k, working_v = staged_leaves
            required = overflow_len + recent_len
            if int(working_k.size(2)) != required or int(working_v.size(2)) != required:
                raise ValueError("cross-layer cached staging has the wrong length")
            overflow_k = working_k[..., :overflow_len, :]
            overflow_v = working_v[..., :overflow_len, :]
            recent_source_k = working_k[..., overflow_len:required, :]
            recent_source_v = working_v[..., overflow_len:required, :]

        self.engine._append_page_cache(
            page_cache,
            overflow_k,
            overflow_v,
            owners.long(),
            owner_ranks=owner_ranks.long(),
        )
        if recent_len > self.local_capacity:
            raise ValueError("cross-layer cached exact tail exceeds its fixed storage")
        recent_k = self.state["recent_k"][slot : slot + 1]
        recent_v = self.state["recent_v"][slot : slot + 1]
        recent_k[..., :recent_len, :].copy_(recent_source_k)
        recent_v[..., :recent_len, :].copy_(recent_source_v)

        self.local_lens[slot].fill_(recent_len)
        self.state_lens[slot].fill_(state_len)
        self.leaf_lens[slot].fill_(int(page_cache["leaf_count"]))
        metadata.update(
            state_len=state_len,
            scheduled_state_len=scheduled_state_len,
            coverage=coverage,
            total_len=total_len,
            recent_len=recent_len,
            leaf_count=int(page_cache["leaf_count"]),
            overflow_safe_until=int(page_cache["overflow_safe_until"]),
        )
        self._refresh_unified_page1_coarse((slot,))
        self.engine.reset_runtime_cache()

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
        if not self.clean[slot]:
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
        mla_query: torch.Tensor | None = None,
        mla_w_uk_t: torch.Tensor | None = None,
        mla_w_uv: torch.Tensor | None = None,
        finalize_cache_for_decode: bool,
        allow_cross_layer_cached: bool,
    ) -> None:
        """Advance one contiguous equal-length/equal-history cache group."""
        length = plan[0][2] - plan[0][1]
        previous_length = plan[0][3]
        slots = tuple(slot for slot, _, _, _ in plan)
        # Initial and previous-chunk cache construction may be overlapped with
        # later model layers.  It must be visible before this layer consumes
        # the shadow on the next scheduler chunk.
        self.wait_deferred_prefill(slots)
        shadow_slots = tuple(
            slot for slot in slots if slot in self.dcp_prefill_shadows
        )
        # A one-row view can outlive the batched parent from which it was
        # first published.  Cached-prefill updates replace scalar metadata in
        # that row view (total_len, coverage, ...), whereas the parent's
        # immutable scalar fields still describe the original prefix.  Only
        # recover the parent when a multi-row group genuinely needs the shared
        # tensor batch; singleton continuations must consume their authoritative
        # request-owned view.
        batched_shadow = (
            self._batched_dcp_shadow(slots)
            if len(shadow_slots) > 1
            else None
        )
        if shadow_slots and len(plan) != 1 and batched_shadow is None:
            # Each temporary cache owns an independently sized full-prompt
            # allocation.  Keep batching for persistent rows, but advance
            # active DCP shadows independently without copying their archives.
            for item in plan:
                self._direct_cached_prefill_group(
                    query,
                    key,
                    value,
                    output,
                    (item,),
                    mla_query=mla_query,
                    mla_w_uk_t=mla_w_uk_t,
                    mla_w_uv=mla_w_uv,
                    finalize_cache_for_decode=finalize_cache_for_decode,
                    allow_cross_layer_cached=allow_cross_layer_cached,
                )
            return
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
                    mla_query=mla_query,
                    mla_w_uk_t=mla_w_uk_t,
                    mla_w_uv=mla_w_uv,
                    finalize_cache_for_decode=finalize_cache_for_decode,
                    allow_cross_layer_cached=allow_cross_layer_cached,
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
        kimi_q = (
            None
            if mla_query is None
            else (
                mla_query[packed_begin:packed_end]
                .reshape(len(plan), length, *mla_query.shape[1:])
                .permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [
                        mla_query[begin:end].permute(1, 0, 2)
                        for _, begin, end, _ in plan
                    ]
                )
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
        using_shadow = bool(shadow_slots)
        cache = (
            (
                batched_shadow
                if batched_shadow is not None
                else self.dcp_prefill_shadows[slots[0]]
            )
            if using_shadow
            else self._range_cache(slots[0], slots[-1] + 1)
        )
        if cache.total_length != previous_length:
            raise RuntimeError(
                "batched cached LOD prefill length differs from its prepared plan: "
                f"slots={slots}, cache={cache.total_length}, "
                f"prepared={previous_length}, shadow={using_shadow}, "
                f"batched_shadow={batched_shadow is not None}"
            )
        output_view = (
            output[packed_begin:packed_end]
            .reshape(len(plan), length, *output.shape[1:])
            .permute(0, 2, 1, 3)
            if packed
            else None
        )
        metadata = self.metadata[slots[0]]
        exact_lookback = int(self.engine.prefill_local_len) - int(
            self.engine.prefill_chunk_len
        )
        cross_layer_cached = bool(
            self.cached_prefill_stager is not None
            and (
                len(slots) == 1
                or (self.dcp_world_size > 1 and batched_shadow is not None)
            )
            and allow_cross_layer_cached
            and self.settings.levels in (2, 3)
            and self.settings.kv_bits in (0, 4)
            and (
                length == int(self.engine.prefill_chunk_len)
                or (
                    # The final partial K3 chunk also needs just one direct
                    # rank-local conversion, not a discarded global update.
                    # Apply the same boundary policy with either archive layout.
                    self.is_absorbed_mla and self.dcp_world_size > 1
                    and kimi_q is not None and finalize_cache_for_decode
                    and 1 < length < int(self.engine.prefill_chunk_len)
                )
            )
            and previous_length >= int(self.engine.prefill_chunk_len)
            and previous_length % int(self.engine.prefill_chunk_len) == 0
            and int(metadata["coverage"]) == previous_length - exact_lookback
            and int(metadata["recent_len"]) == exact_lookback
            and self.engine.virtual_page_storage
            and self.engine.state_premerge_factor == 1
            and self.engine.state_split_max_leaves is None
            and self.engine.state_clustering_query_metric == "none"
        )
        if (
            os.environ.get("LOD_KIMI_PROFILE_PREFILL") == "1"
            and self.dcp_rank == 0
        ):
            print(
                "KIMI_CACHED_STAGE_CHECK "
                f"enabled={int(cross_layer_cached)} length={length} "
                f"previous={previous_length} state_len={metadata['state_len']} "
                f"coverage={metadata['coverage']} recent={metadata['recent_len']} "
                f"shadow_rows={len(self.dcp_prefill_shadows)}",
                flush=True,
            )
        defer_cache_update = (
            not cross_layer_cached
            and self.deferred_prefill_stream is not None
        )
        # The final state/page update does not contribute to this layer's
        # output.  Queue it behind the attention work and let subsequent model
        # layers hide it; _range_cache and decode consume the completion event.
        if defer_cache_update:
            self.engine._lod_prefill_deferred_update_stream = (
                self.deferred_prefill_stream
            )
        if cross_layer_cached:
            self.engine._lod_stage_cached_prefill_update = True
        if kimi_q is not None:
            self.engine._lod_kimi_expanded_prefill_query = kimi_q
            self.engine._lod_kimi_w_uk_t = mla_w_uk_t
            self.engine._lod_kimi_w_uv = mla_w_uv
        try:
            # DCP shadows own their storage format across scheduler chunks.
            # In INT4 mode they are already finalized after the first chunk,
            # and _append_page_cache requantizes only pages touched later.
            engine_finalize = bool(
                finalize_cache_for_decode
                and not (using_shadow and self.dcp_world_size > 1)
            )
            if self.kimi_sharded_leaf_prefill and using_shadow:
                if not cross_layer_cached or kimi_q is None:
                    raise NotImplementedError("sharded-leaf prefill currently requires aligned projected chunks")
                from .models.kimi_k3_sharded_prefill import attention

                replicated = getattr(self, "_kimi_replicated_prefill_query", None)
                if replicated is not None:
                    replicated = (
                        replicated[packed_begin:packed_end]
                        .reshape(len(plan), length, *replicated.shape[1:])
                        .permute(0, 2, 1, 3)
                        if packed else torch.stack([
                            replicated[begin:end].permute(1, 0, 2)
                            for _, begin, end, _ in plan
                        ])
                    )
                result = attention(self, cache, kimi_q, k, mla_w_uk_t, mla_w_uv,
                                   output_buffer=output_view, replicated_query=replicated)
            else:
                result, cache = self.engine(
                    q,
                    k,
                    v,
                    cache=cache,
                    use_cache=True,
                    output_buffer=output_view,
                    finalize_cache_for_decode=engine_finalize,
                )
        finally:
            if defer_cache_update:
                del self.engine._lod_prefill_deferred_update_stream
            if cross_layer_cached:
                del self.engine._lod_stage_cached_prefill_update
            for name in (
                "_lod_kimi_expanded_prefill_query",
                "_lod_kimi_w_uk_t",
                "_lod_kimi_w_uv",
            ):
                if hasattr(self.engine, name):
                    delattr(self.engine, name)
        if cache is None:
            raise AssertionError("batched cached LOD prefill did not return a cache")
        if len(slots) > 1:
            self.batched_cached_prefill_calls += 1
            self.batched_cached_prefill_rows += len(slots)
        if cross_layer_cached:
            stager = self.cached_prefill_stager
            if stager is None:
                raise AssertionError("cross-layer cached stager is missing")
            profile_update = os.environ.get("LOD_KIMI_PROFILE_PREFILL") == "1"
            update_begin = None
            if profile_update:
                update_begin = torch.cuda.Event(enable_timing=True)
                update_begin.record()
            stager(
                self,
                slots,
                k,
                v,
                previous_len=previous_length,
                total_len=previous_length + length,
                finalize_cache_for_decode=finalize_cache_for_decode,
            )
            if update_begin is not None:
                update_end = torch.cuda.Event(enable_timing=True)
                update_end.record()
                torch.cuda.synchronize(self.device)
                print(
                    "KIMI_PREFILL_PHASES "
                    f"cross_layer_update={update_begin.elapsed_time(update_end):.3f}ms",
                    flush=True,
                )
        elif defer_cache_update:
            deferred = self.deferred_prefill_stream
            if deferred is None:
                raise AssertionError("deferred prefill stream is missing")
            with torch.cuda.stream(deferred):
                if using_shadow:
                    self._retain_batched_dcp_shadow(slots, cache)
                else:
                    self._synchronize_rows(slots, cache)
                completed = torch.cuda.Event()
                completed.record(deferred)
            for slot in slots:
                self.deferred_prefill_events[slot] = completed
        elif using_shadow:
            # Keep final replicated rows until the runtime can shard all
            # layers in one batched conversion at the decode boundary.
            self._retain_batched_dcp_shadow(slots, cache)
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
        *,
        mla_query: torch.Tensor | None = None,
        mla_w_uk_t: torch.Tensor | None = None,
        mla_w_uv: torch.Tensor | None = None,
        defer_mla_query_absorption: bool = False,
    ) -> torch.Tensor:
        """Run ragged initial or cached prefill into authoritative LOD rows."""
        if (mla_query is None) != (mla_w_uk_t is None):
            raise ValueError("Kimi prefill requires both expanded query and W_UK_T")
        if mla_w_uv is not None and mla_query is None:
            raise ValueError("Kimi projected prefill requires its expanded query")
        if defer_mla_query_absorption and (
            mla_query is None or mla_w_uk_t is None
        ):
            raise ValueError("deferred Kimi absorption requires query and W_UK_T")
        if mla_query is not None and int(mla_query.size(0)) != int(query.size(0)):
            raise ValueError("Kimi expanded and absorbed query lengths differ")
        plan = self.direct_prefill_plan
        self.direct_prefill_plan = None
        prompt_lengths = self.direct_prefill_prompt_lengths
        self.direct_prefill_prompt_lengths = {}
        if plan is None:
            raise RuntimeError("direct LOD prefill has no prepared request plan")
        self.direct_prefill_calls += 1
        initial: dict[tuple[int, bool, int], list[tuple[int, int, int, int]]] = {}
        cached: list[tuple[int, int, int, int]] = []
        for item in plan:
            slot, begin, end, previous_length = item
            if end <= begin:
                continue
            available = self.has_prefill_cache(slot)
            if previous_length == 0 and not available:
                if slot not in prompt_lengths:
                    raise RuntimeError("direct LOD prefill has no total prompt length")
                length = end - begin
                # A replicated DCP shadow is request-owned, so keep its first
                # cache construction singleton.  Ordinary rows retain the
                # existing equal-length batch path.
                group_slot = prompt_lengths[slot] if self.dcp_world_size > 1 else -1
                initial.setdefault(
                    (length, length >= prompt_lengths[slot], group_slot), []
                ).append(item)
            elif previous_length > 0 and available:
                cached.append(item)
            elif previous_length > 0:
                raise RuntimeError("cached LOD prefill row is not initialized")
            else:
                raise RuntimeError("initial LOD prefill row is already initialized")

        for (length, finalize_cache_for_decode, _group_slot), group in initial.items():
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
            kimi_q = (
                None
                if mla_query is None
                else (
                    mla_query[packed_begin:packed_end]
                    .reshape(len(group), length, *mla_query.shape[1:])
                    .permute(0, 2, 1, 3)
                    if packed
                    else torch.stack(
                        [
                            mla_query[begin:end].permute(1, 0, 2)
                            for _, begin, end, _ in group
                        ]
                    )
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
            cross_layer_initial = bool(
                self.initial_prefill_stager is not None
                and len(initial) == 1
                and (self.dcp_world_size > 1 or len(group) == 1)
                and self.settings.levels in (2, 3)
                and self.settings.kv_bits in (0, 4)
                and self.engine.prefill_exact_first_chunk
                and int(self.engine.chunk_len) * 2 < length
                and length <= int(self.engine.prefill_chunk_len)
                and self.engine.virtual_page_storage
                and self.engine.state_premerge_factor == 1
                and self.engine.state_split_max_leaves is None
                and self.engine.state_clustering_query_metric == "none"
            )
            if cross_layer_initial:
                if kimi_q is not None:
                    self.engine._lod_kimi_expanded_prefill_query = kimi_q
                    self.engine._lod_kimi_w_uk_t = mla_w_uk_t
                    self.engine._lod_kimi_w_uv = mla_w_uv
                try:
                    result = self.engine._exact_attention(
                        q,
                        k,
                        v,
                        causal=True,
                        output_buffer=output_view,
                    )
                finally:
                    for name in (
                        "_lod_kimi_expanded_prefill_query",
                        "_lod_kimi_w_uk_t",
                        "_lod_kimi_w_uv",
                    ):
                        if hasattr(self.engine, name):
                            delattr(self.engine, name)
                if (
                    output_view is not None
                    and result.data_ptr() != output_view.data_ptr()
                ):
                    output_view.copy_(result)
                    result = output_view
                exact_lookback = int(self.engine.prefill_local_len) - int(
                    self.engine.prefill_chunk_len
                )
                coverage = max(
                    min(length, int(self.engine.chunk_len)),
                    length - exact_lookback,
                )
                self.initial_prefill_stager(
                    self,
                    slots,
                    k,
                    v,
                    total_len=length,
                    coverage=coverage,
                    prompt_capacity=prompt_lengths[slots[0]],
                )
                if packed:
                    if (
                        output_view is None
                        or result.data_ptr() != output_view.data_ptr()
                    ):
                        raise AssertionError(
                            "cross-layer exact prefill did not use its output buffer"
                        )
                else:
                    for source_slot, (_, begin, end, _) in enumerate(group):
                        output[begin:end].copy_(result[source_slot].permute(1, 0, 2))
                continue
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
                if kimi_q is not None:
                    self.engine._lod_kimi_expanded_prefill_query = kimi_q
                    self.engine._lod_kimi_w_uk_t = mla_w_uk_t
                    self.engine._lod_kimi_w_uv = mla_w_uv
                try:
                    result = self.engine._exact_attention(
                        q,
                        k,
                        v,
                        causal=True,
                        output_buffer=output_view,
                    )
                finally:
                    for name in (
                        "_lod_kimi_expanded_prefill_query",
                        "_lod_kimi_w_uk_t",
                        "_lod_kimi_w_uv",
                    ):
                        if hasattr(self.engine, name):
                            delattr(self.engine, name)
                if (
                    output_view is not None
                    and result.data_ptr() != output_view.data_ptr()
                ):
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
                    if self.dcp_world_size > 1:
                        self._retain_dcp_prefill_shadow(slots[0], cache)
                    else:
                        self.install_rows(slots, cache)
                    self.engine.reset_runtime_cache()
                    completed = torch.cuda.Event()
                    completed.record(deferred)
                for slot in slots:
                    self.deferred_prefill_events[slot] = completed
            else:
                if prefill_storage is not None:
                    self.engine._lod_prefill_storage = prefill_storage
                if self.dcp_world_size > 1:
                    if len(slots) != 1:
                        raise AssertionError("DCP prefill shadows must be singleton")
                    self.engine._lod_prefill_cache_capacity = int(
                        prompt_lengths[slots[0]]
                    )
                if kimi_q is not None:
                    self.engine._lod_kimi_expanded_prefill_query = kimi_q
                    self.engine._lod_kimi_w_uk_t = mla_w_uk_t
                    self.engine._lod_kimi_w_uv = mla_w_uv
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
                    for name in (
                        "_lod_kimi_expanded_prefill_query",
                        "_lod_kimi_w_uk_t",
                        "_lod_kimi_w_uv",
                    ):
                        if hasattr(self.engine, name):
                            delattr(self.engine, name)
                    if prefill_storage is not None:
                        del self.engine._lod_prefill_storage
                    if hasattr(self.engine, "_lod_prefill_cache_capacity"):
                        del self.engine._lod_prefill_cache_capacity
            if cache is None:
                raise AssertionError("direct LOD prefill did not return a cache")
            if not defer_cache:
                if self.dcp_world_size > 1:
                    slot = slots[0]
                    if finalize_cache_for_decode:
                        self._install_dcp_local_from_cache(slot, cache)
                    else:
                        self._retain_dcp_prefill_shadow(slot, cache)
                elif tuple(sorted(slots)) == tuple(
                    range(min(slots), max(slots) + 1)
                ):
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
        # vLLM can schedule completed one-token rows beside a long cached
        # prefill row.  Those rows already own rank-local DCP caches and must
        # use decode: their prepared ``previous_length`` is global, whereas
        # ``_range_cache`` deliberately exposes rank-local lengths.  Peel them
        # off before grouping the remaining cached-prefill work.  The old
        # all-or-nothing check sent both classes through cached prefill, and it
        # also disabled cross-layer construction for the actual long row.
        mixed_decode = tuple(
            item
            for item in cached
            if item[2] - item[1] == 1
            and item[0] not in self.dcp_prefill_shadows
        )
        if mixed_decode:
            self._direct_mixed_decode(
                query,
                key,
                value,
                output,
                mixed_decode,
                mla_query=(mla_query if defer_mla_query_absorption else None),
                mla_w_uk_t=(mla_w_uk_t if defer_mla_query_absorption else None),
                mla_w_uv=mla_w_uv,
            )
            mixed_slots = {slot for slot, _, _, _ in mixed_decode}
            cached = [item for item in cached if item[0] not in mixed_slots]
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
                mla_query=mla_query,
                mla_w_uk_t=mla_w_uk_t,
                mla_w_uv=mla_w_uv,
                finalize_cache_for_decode=bool(
                    group[0][2] - group[0][1] + group[0][3]
                    >= prompt_lengths[group[0][0]]
                ),
                allow_cross_layer_cached=len(groups) == 1,
            )
        return output

    def _direct_mixed_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        plan: tuple[tuple[int, int, int, int], ...],
        *,
        mla_query: torch.Tensor | None = None,
        mla_w_uk_t: torch.Tensor | None = None,
        mla_w_uv: torch.Tensor | None = None,
    ) -> None:
        """Batch one-token rows that vLLM schedules beside a long prefill."""
        if (mla_query is None) != (mla_w_uk_t is None):
            raise ValueError("lazy Kimi decode absorption requires query and W_UK_T")
        rows = len(plan)
        first = plan[0][1]
        packed = all(
            begin == first + row and end == begin + 1
            for row, (_, begin, end, _) in enumerate(plan)
        )
        if packed:
            q = query[first : first + rows]
            k = key[first : first + rows]
            v = value[first : first + rows]
            destination = output[first : first + rows]
        else:
            positions = torch.tensor(
                [begin for _, begin, _, _ in plan],
                dtype=torch.long,
                device=query.device,
            )
            q = query.index_select(0, positions)
            k = key.index_select(0, positions)
            v = value.index_select(0, positions)
            destination = output.new_empty(rows, *output.shape[1:])

        if mla_query is not None:
            raw_q = (
                mla_query[first : first + rows]
                if packed
                else mla_query.index_select(0, positions)
            )
            from .models.kimi_k3 import absorb_query

            q = absorb_query(
                raw_q,
                mla_w_uk_t,
                nope_dim=int(mla_w_uk_t.size(1)),
            )

        class _Metadata:
            num_actual_tokens = rows
            max_seq_len = max(previous_length + 1 for *_, previous_length in plan)

        decode_destination = destination
        if mla_w_uv is not None:
            decode_destination = output.new_empty(
                rows,
                self.query_heads,
                self.value_dim,
            )
        if self.is_absorbed_mla and self.dcp_world_size > 1:
            # These rows have already converted to rank-local chronological
            # caches. The generic local decoder cannot see the other DCP
            # slices, even when the surrounding scheduler batch is prefill.
            # Absorb each TP head on its owner, then gather the tiny one-token
            # queries and combine partials exactly as ordinary DCP decode.
            from .models.kimi_k3_sharded_prefill import gather_prefill
            try:
                from vllm.v1.attention.ops.dcp import cp_lse_ag_out_rs
            except ImportError:
                from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

            if tuple(item[0] for item in plan) != self.active_decode_rows[:rows]:
                raise AssertionError("mixed DCP decode disagrees with its active-row map")
            all_q = gather_prefill(self.dcp_group, q, dim=1)
            partial = output.new_empty(rows, all_q.size(1), self.value_dim)
            partial, lse = self.decode_dcp(all_q, k, v, partial)
            decode_destination.copy_(cp_lse_ag_out_rs(
                partial, lse, self.dcp_group, is_lse_base_on_e=True,
            ))
            result = decode_destination
        else:
            result = self.decode(q, k, v, _Metadata(), decode_destination)
        if result.data_ptr() != decode_destination.data_ptr():
            raise AssertionError("mixed LOD decode did not use its output buffer")
        if mla_w_uv is not None:
            from lod_attention.kernels.aiter_mla_prefill_attention import (
                project_kimi_head_values,
            )

            projected = project_kimi_head_values(result.unsqueeze(2), mla_w_uv)
            destination.copy_(projected.squeeze(2))
            result = destination
        if not packed:
            output.index_copy_(0, positions, result)
        for slot, _, _, previous_length in plan:
            metadata = self.metadata[slot]
            global_total_len = previous_length + 1
            total_len = (
                self._dcp_local_length(global_total_len)
                if self.dcp_world_size > 1 and self.dcp_sharded[slot]
                else global_total_len
            )
            metadata["total_len"] = total_len
            metadata["recent_len"] = total_len - int(metadata["coverage"])
            if self.dcp_world_size > 1:
                metadata["dcp_global_total_len"] = global_total_len
                self.dcp_global_lens[slot].fill_(global_total_len)

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
        if self.dcp_world_size > 1 and self.dcp_sharded[slot]:
            global_coverage = int(metadata["dcp_global_coverage"])
            expected_local_coverage = self._dcp_local_length(global_coverage)
            if int(metadata["coverage"]) != expected_local_coverage:
                raise AssertionError(
                    "DCP local coverage no longer represents its global boundary"
                )
            local_total_length = self._dcp_local_length(total_length)
            recent_length = local_total_length - expected_local_coverage
            target_global_coverage = self._dcp_global_decode_coverage(total_length)
            target_local_coverage = self._dcp_local_length(target_global_coverage)
            if recent_length < 0 or recent_length > self.local_capacity:
                raise ValueError(
                    "decode-local length exceeds its fixed DCP cache row: "
                    f"slot={slot}, global_total={total_length}, "
                    f"global_coverage={global_coverage}, local_total={local_total_length}, "
                    f"local_coverage={expected_local_coverage}, recent={recent_length}, "
                    f"capacity={self.local_capacity}"
                )
            return recent_length, target_local_coverage
        coverage = int(metadata["coverage"])
        recent_length = total_length - coverage
        if recent_length < 0 or recent_length > self.local_capacity:
            raise ValueError(
                "decode-local length exceeds its fixed cache row: "
                f"slot={slot}, total={total_length}, coverage={coverage}, "
                f"recent={recent_length}, capacity={self.local_capacity}, "
                f"device_recent={int(self.local_lens[slot].item())}"
            )
        with self._dcp_local_state_schedule():
            update_len = int(self.engine.decode_state_update_len)
            exact_floor = int(self.engine.local_len - self.engine.chunk_len)
            initial_chunk = int(self.engine.chunk_len)
        target_coverage = max(min(total_length, initial_chunk), coverage)
        pending_update = total_length + 1 - target_coverage - exact_floor
        if pending_update > update_len:
            target_coverage += ((pending_update - 1) // update_len) * update_len
        return recent_length, min(target_coverage, total_length)

    def catch_up(self, slot: int, total_length: int) -> None:
        if not self.ready[slot]:
            raise RuntimeError("cannot catch up an uninitialized LOD request row")
        metadata = self.metadata[slot]
        global_total_length = total_length
        local_total_length = (
            self._dcp_local_length(total_length)
            if self.dcp_world_size > 1 and self.dcp_sharded[slot]
            else total_length
        )
        coverage = int(metadata["coverage"])
        recent_length, target_coverage = self._catch_up_target(slot, total_length)
        if coverage >= target_coverage:
            # Captured decode already appended K/V and advanced local_lens on
            # device. Most tokens need only this host metadata bookkeeping.
            metadata["total_len"] = local_total_length
            metadata["recent_len"] = recent_length
            if self.dcp_world_size > 1:
                metadata["dcp_global_total_len"] = global_total_length
                self.dcp_global_lens[slot].fill_(global_total_length)
            return
        row = self._row_cache(slot)
        with self._dcp_local_state_schedule():
            self.engine.catch_up_cache(
                row, total_length=local_total_length, recent_length=recent_length
            )
        self._finish_single_catch_up(slot, row)
        if self.dcp_world_size > 1:
            global_coverage = self._dcp_global_decode_coverage(global_total_length)
            if int(self.metadata[slot]["coverage"]) != self._dcp_local_length(
                global_coverage
            ):
                raise AssertionError(
                    "DCP catch-up did not land on its global sequence boundary"
                )
            self.metadata[slot]["dcp_global_total_len"] = global_total_length
            self.metadata[slot]["dcp_global_coverage"] = global_coverage
            self.dcp_global_lens[slot].fill_(global_total_length)

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

        global_total_length = total_length
        local_total_length = (
            self._dcp_local_length(total_length)
            if self.dcp_world_size > 1 and self.dcp_sharded[slot]
            else total_length
        )
        recent_length, target_coverage = self._catch_up_target(slot, total_length)
        if int(self.metadata[slot]["coverage"]) >= target_coverage:
            raise ValueError("precomputed LOD catch-up has no pending state update")
        row = self._row_cache(slot)
        with self._dcp_local_state_schedule():
            self.engine.catch_up_cache(
                row,
                total_length=local_total_length,
                recent_length=recent_length,
                _precomputed_update=(state_len, owners, None),
            )
        self.catch_up_batches += 1
        self.catch_up_rows += 1
        self._finish_single_catch_up(slot, row)
        if self.dcp_world_size > 1:
            global_coverage = self._dcp_global_decode_coverage(global_total_length)
            if int(self.metadata[slot]["coverage"]) != self._dcp_local_length(
                global_coverage
            ):
                raise AssertionError(
                    "precomputed DCP catch-up missed its global sequence boundary"
                )
            self.metadata[slot]["dcp_global_total_len"] = global_total_length
            self.metadata[slot]["dcp_global_coverage"] = global_coverage
            self.dcp_global_lens[slot].fill_(global_total_length)

    def catch_up_many(self, requests: list[tuple[int, int]]) -> None:
        """Batch equal-metadata contiguous rows at a state-update boundary."""
        pending: dict[tuple[int, ...], list[int]] = {}
        for slot, total_length in requests:
            if not self.ready[slot]:
                raise RuntimeError("cannot catch up an uninitialized LOD request row")
            metadata = self.metadata[slot]
            global_total_length = total_length
            local_total_length = (
                self._dcp_local_length(total_length)
                if self.dcp_world_size > 1 and self.dcp_sharded[slot]
                else total_length
            )
            recent_length, target_coverage = self._catch_up_target(slot, total_length)
            if int(metadata["coverage"]) >= target_coverage:
                metadata["total_len"] = local_total_length
                metadata["recent_len"] = recent_length
                if self.dcp_world_size > 1:
                    metadata["dcp_global_total_len"] = global_total_length
                    self.dcp_global_lens[slot].fill_(global_total_length)
                continue
            target_global_coverage = (
                self._dcp_global_decode_coverage(global_total_length)
                if self.dcp_world_size > 1
                else target_coverage
            )
            signature = (
                local_total_length,
                global_total_length,
                target_global_coverage,
                int(metadata["state_len"]),
                int(metadata.get("scheduled_state_len", metadata["state_len"])),
                int(metadata["coverage"]),
                int(metadata["recent_len"]),
                int(metadata["leaf_count"]),
                int(metadata["overflow_safe_until"]),
            )
            pending.setdefault(signature, []).append(slot)
            if self.dcp_world_size > 1:
                metadata["dcp_global_total_len"] = global_total_length
                self.dcp_global_lens[slot].fill_(global_total_length)

        for signature, slots in pending.items():
            total_length = signature[0]
            target_global_coverage = signature[2]
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
                with self._dcp_local_state_schedule():
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
                    if self.dcp_world_size > 1:
                        if int(self.metadata[slot]["coverage"]) != (
                            self._dcp_local_length(target_global_coverage)
                        ):
                            raise AssertionError(
                                "batched DCP catch-up missed its global sequence "
                                "boundary"
                            )
                        self.metadata[slot]["dcp_global_coverage"] = (
                            target_global_coverage
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
                shared_scratch=self.shared_decode_scratch,
                splits=int(self.engine.decode_split_kv),
                value_dim=self.value_dim,
                exact_kv_heads=(
                    self.kv_heads
                    if self.engine.exact_decode_limit > 0 and self.settings.levels == 3
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
                    if (
                        self.settings.levels == 2
                        or (self.settings.levels == 3 and self.family is ModelFamily.K2)
                    )
                    and self.query_heads % self.kv_heads == 0
                    and 1 < self.query_heads // self.kv_heads <= 16
                    and (self.head_dim in (128, 256) or self.is_absorbed_mla)
                    and self.dtype == torch.bfloat16
                    else None
                ),
                gqa_union_index_capacity=(
                    (self.leaf_capacity if self.settings.levels == 2 else 0)
                    + self.decode_local_limit
                    + 1
                    + self.state_capacity
                    + (
                        int(self.state["sink_k"].size(2))
                        if isinstance(self.state.get("sink_k"), torch.Tensor)
                        else 0
                    )
                    if (
                        self.settings.levels == 2
                        or (self.settings.levels == 3 and self.family is ModelFamily.K2)
                    )
                    and self.query_heads % self.kv_heads == 0
                    and 1 < self.query_heads // self.kv_heads <= 16
                    and (self.head_dim in (128, 256) or self.is_absorbed_mla)
                    and self.dtype == torch.bfloat16
                    else None
                ),
                gqa_union_hip=True,
                gqa_union_fixed_mask=(
                    self.settings.levels == 2
                    and self.settings.decode_gqa_fixed_mask_aiter
                ),
                gqa_union_fixed_mask_tile_size=64,
                gqa_union_fixed_mask_segments=(
                    self.settings.decode_gqa_fixed_mask_segments
                ),
                gqa_union_hip_segments=32,
            )
            if self.settings.levels == 3 and self.family is ModelFamily.K2:
                # Recursive decode always scans the same physical local/sink/
                # coarse arena layout. Build that page-size-one table once;
                # the hot route kernel supplies only the current logical length
                # and current-token K/V before AITER consumes it.
                table = storage["gqa_union_hip_block_table"]
                page = self.state["page_cache"]
                sequences = self.max_requests * self.kv_heads
                physical_rows = torch.arange(
                    sequences, dtype=torch.int32, device=self.device
                ).unsqueeze(1)
                local_width = self.decode_local_limit + 1
                sink_width = int(self.state["sink_k"].size(2))
                local_tokens = torch.arange(
                    local_width, dtype=torch.int32, device=self.device
                ).unsqueeze(0)
                coarse_tokens = torch.arange(
                    self.state_capacity, dtype=torch.int32, device=self.device
                ).unsqueeze(0)
                parts = [
                    int(page["unified_page1_local_offset"])
                    + physical_rows * self.local_capacity
                    + local_tokens,
                ]
                if sink_width:
                    sink_tokens = torch.arange(
                        sink_width, dtype=torch.int32, device=self.device
                    ).unsqueeze(0)
                    parts.append(
                        int(page["unified_page1_sink_offset"])
                        + physical_rows * sink_width
                        + sink_tokens
                    )
                parts.append(
                    int(page["unified_page1_coarse_offset"])
                    + physical_rows * self.state_capacity
                    + coarse_tokens
                )
                persistent_indices = torch.cat(parts, dim=1)
                if int(persistent_indices.size(1)) > int(table.size(1)):
                    raise RuntimeError("recursive AITER index table is undersized")
                table[:, : persistent_indices.size(1)].copy_(persistent_indices)
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
                storage["exact_exp_sums"] = torch.empty_like(storage["partial_lse"])
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
        if self.kimi_head_tiled_decode:
            self._dcp_buffers(query, rows)
        else:
            self._buffers(query, rows)

    def _dcp_buffers(
        self, query: torch.Tensor, rows: int
    ) -> dict[str, torch.Tensor]:
        """Fixed-address scratch for K3's 96-head DCP decode query."""

        if self.dcp_world_size == 1 and not self.kimi_head_tiled_decode:
            return self._buffers(query, rows)
        if not self.is_absorbed_mla or int(query.size(1)) % 16:
            raise ValueError("Kimi DCP LoD requires 16-head query tiles")
        storage = self.dcp_decode_buffer_storage
        if storage is None or storage["partial_out"].device != query.device:
            virtual_kv_heads = int(query.size(1)) // 16
            template = query.new_empty(
                self.max_requests,
                int(query.size(1)),
                1,
                self.head_dim,
            )
            sink = self.state.get("sink_k")
            sink_len = int(sink.size(2)) if isinstance(sink, torch.Tensor) else 0
            storage = new_fused_decode_buffers(
                template,
                shared_scratch=self.shared_decode_scratch,
                splits=int(self.engine.decode_split_kv),
                value_dim=self.value_dim,
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
                route_segment_tiles=int(self.engine.decode_route_segment_tiles),
                gqa_union_kv_heads=virtual_kv_heads,
                gqa_union_index_capacity=(
                    self.leaf_capacity
                    + self.decode_local_limit
                    + 1
                    + self.state_capacity
                    + sink_len
                ),
                gqa_union_hip=True,
                gqa_union_fixed_mask=False,
                gqa_union_hip_segments=32,
            )
            self.dcp_decode_buffer_storage = storage
            self.dcp_decode_buffers.clear()
        buffers = self.dcp_decode_buffers.get(rows)
        if buffers is None:
            buffers = {
                name: (
                    tensor[:rows]
                    if tensor.ndim
                    and int(tensor.size(0)) == self.max_requests
                    else tensor
                )
                for name, tensor in storage.items()
            }
            self.dcp_decode_buffers[rows] = buffers
        return buffers

    def reserve_dcp_decode_buffers(self, rows: int, heads: int) -> None:
        if self.dcp_world_size == 1:
            return
        query = torch.empty(
            rows,
            heads,
            1,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._dcp_buffers(query, rows)

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
                value_dim=self.value_dim,
                exact_kv_heads=(
                    self.kv_heads
                    if self.engine.exact_decode_limit > 0 and self.settings.levels == 3
                    else None
                ),
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
                route_segment_tiles=int(self.engine.decode_route_segment_tiles),
                materialized_state_route=bool(
                    self.settings.levels == 3 and speculative_route_backend == "resplit"
                ),
                # Multi-token verification consumes each query's own eight
                # centroids and does not build a GQA-wide leaf union.
                gqa_union_kv_heads=None,
                gqa_union_index_capacity=None,
                gqa_union_hip=True,
                gqa_union_fixed_mask=False,
                gqa_union_fixed_mask_tile_size=64,
                gqa_union_fixed_mask_segments=(
                    self.settings.decode_gqa_fixed_mask_segments
                ),
                gqa_union_hip_segments=32,
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

    def decode_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run 16-head Kimi tiles over one physical cache, with an LSE.

        DCP ranks use this for their partial result. A single-GPU owner uses
        the same tiles without collectives; cache storage is never duplicated.
        """

        if self.dcp_world_size <= 1 and not self.kimi_head_tiled_decode:
            raise RuntimeError("decode_dcp requires an initialized DCP group")
        rows, gathered_heads, head_dim = query.shape
        if head_dim != self.head_dim or gathered_heads % 16:
            raise ValueError("unexpected gathered Kimi DCP query geometry")
        if tuple(key.shape[:2]) != (rows, self.kv_heads):
            raise ValueError("Kimi DCP current K/V geometry differs from its pool")
        virtual_kv_heads = gathered_heads // 16
        q = query.unsqueeze(2).contiguous()
        k = key.unsqueeze(2).contiguous()
        v = value.unsqueeze(2).contiguous()
        cache_indices = self.active_indices[:rows]
        active_slots = self.active_decode_rows[:rows]
        if self.dcp_world_size > 1 and len(active_slots) != rows:
            raise RuntimeError(
                "DCP decode has no host-side active-row map for this batch"
            )
        if self.dcp_world_size > 1:
            self.ensure_dcp_sharded(active_slots)
        self.ensure_unified_page1_fixed(active_slots)

        buffers = self._dcp_buffers(q, rows)
        page = self.state["page_cache"]

        def virtual_heads(tensor: torch.Tensor) -> torch.Tensor:
            if int(tensor.size(1)) != 1:
                raise ValueError("Kimi DCP expects one physical MLA KV head")
            return tensor.expand(tensor.size(0), virtual_kv_heads, *tensor.shape[2:])

        result = fused_decode_paged_lod_attention(
            q,
            virtual_heads(self.state["state_k"]),
            virtual_heads(self.state["state_v"]),
            virtual_heads(self.state["counts"]),
            virtual_heads(self.state["recent_k"]),
            virtual_heads(self.state["recent_v"]),
            virtual_heads(page["leaf_k"]),
            virtual_heads(page["leaf_v"]),
            virtual_heads(page["slot_pages"]),
            virtual_heads(page["overflow_page_keys"]),
            virtual_heads(page["overflow_page_values"]),
            page["overflow_used"],
            virtual_heads(page["slot_lengths"]),
            None,
            sink_k=virtual_heads(self.state["sink_k"]),
            sink_v=virtual_heads(self.state["sink_v"]),
            state_len=self.state_capacity,
            state_lens=self.state_lens,
            max_open_centroid_leaves=self.engine.max_open_centroid_leaves,
            local_len=self.decode_local_limit,
            cache_indices=cache_indices,
            local_lens=self.local_lens,
            new_k=k.expand(rows, virtual_kv_heads, 1, self.head_dim),
            new_v=v.expand(rows, virtual_kv_heads, 1, self.value_dim),
            store_new_kv=True,
            advance_local_lens=False,
            kv_group_size=16,
            scale=float(self.engine.scaling),
            hash_probes=int(self.engine._page_lookup_probes(page)),
            block_n=int(self.engine.decode_block_n),
            num_warps=int(self.engine.decode_num_warps),
            waves_per_eu=int(self.engine.leaf_waves_per_eu),
            split_kv=int(self.engine.decode_split_kv),
            buffers=buffers,
            use_dot=bool(self.engine.decode_use_dot),
            fuse_state_route=True,
            route_group_size=int(self.engine.decode_route_group_size),
            route_segment_tiles=int(self.engine.decode_route_segment_tiles),
            route_num_warps=int(self.engine.decode_route_num_warps),
            route_reduce_num_warps=int(self.engine.decode_route_reduce_num_warps),
            route_parallel_reduce=bool(self.engine.decode_route_parallel_reduce),
            fuse_final_reduce=False,
            route_gqa_grouped=True,
            gqa_union_decode=True,
            gqa_union_hip=True,
            gqa_union_compact_page_descriptors=True,
            gqa_union_fuse_compact_route=bool(getattr(
                self.engine, "_kimi_fuse_compact_union", True)),
            gqa_union_page1_k=page["unified_page1_k"],
            gqa_union_page1_v=page["unified_page1_v"],
            gqa_union_page1_bias=page["unified_page1_bias"],
            gqa_union_page1_leaf_offset=int(page["unified_page1_leaf_offset"]),
            gqa_union_page1_local_offset=int(page["unified_page1_local_offset"]),
            gqa_union_page1_sink_offset=int(page["unified_page1_sink_offset"]),
            gqa_union_page1_coarse_offset=int(page["unified_page1_coarse_offset"]),
            gqa_union_fixed_indices=page["unified_page1_fixed_indices"],
            gqa_union_fixed_leaf_owners=page["unified_page1_fixed_leaf_owners"],
            gqa_union_fixed_slot_offsets=page[
                "unified_page1_fixed_slot_offsets"
            ],
            gqa_union_fixed_lengths=page["unified_page1_fixed_lengths"],
            protected_len=0,
            open_count=ROUTE_COUNT,
            flat_page_indices=virtual_heads(page["page_indices"]),
            exact_decode_threshold=0,
            output_buffer=output.unsqueeze(2),
            distributed_route_group=(
                None if self.kimi_local_dcp_prefill and not self.kimi_shared_dcp_prefill
                else self.dcp_group
            ),
            gqa_union_physical_kv_heads=self.kv_heads,
            gqa_union_head_tiled_metadata=True,
            dcp_global_lens=(self.dcp_global_lens if self.dcp_world_size > 1 else None),
            dcp_rank=self.dcp_rank,
            dcp_world_size=self.dcp_world_size,
            dcp_interleave_size=self.dcp_interleave_size,
        )
        if result.data_ptr() != output.data_ptr():
            raise AssertionError("DCP LoD did not use its output buffer")
        if self.dcp_world_size == 1:
            # All six head tiles alias the same physical recent-KV row. Advance
            # its length once, rather than once per virtual KV head.
            advance_decode_cache_lengths(cache_indices, self.local_lens)
        final_lse = buffers.get("kimi_gluon_final_lse")
        if not isinstance(final_lse, torch.Tensor):
            raise RuntimeError("Kimi DCP compact consumer did not return an LSE")
        return output, final_lse[:rows]

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
        if self.kimi_head_tiled_decode:
            if (cache_indices is not None or local_lens is not None
                    or decode_buffers is not None or local_lens_are_logical
                    or not store_new_kv or not advance_local_lens
                    or speculative_steps != 1):
                raise NotImplementedError("Kimi owner tiles support ordinary one-token decode")
            result, _lse = self.decode_dcp(
                query[:rows], key[:rows], value[:rows], output[:rows],
            )
            return result
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
            max_open_centroid_leaves=self.engine.max_open_centroid_leaves,
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
            final_reduce_num_warps=int(self.engine.decode_final_reduce_num_warps),
            fuse_final_reduce=bool(self.engine.decode_fuse_final_reduce),
            route_gqa_grouped=bool(self.engine.decode_route_gqa_grouped),
            gqa_cooperative_leaf=False,
            # DFlash already supplies eight independent verifier rows. Keep
            # ordinary one-token decode on the GQA-shared union, but let
            # speculative verification consume each query head's eight
            # complete centroids directly instead of building another union.
            gqa_union_decode=speculative_steps < 2,
            gqa_union_hip=True,
            # Two-tier keeps a persistent centroid-major leaf list. Publish
            # only the selected 16-leaf ranges to the attention consumer so
            # decode work follows the routed union instead of total history.
            # Recursive caches keep their established consumer unchanged.
            gqa_union_fixed_mask_aiter=(
                self.settings.decode_gqa_fixed_mask_aiter
                and self.settings.levels != 2
                and speculative_steps < 2
            ),
            gqa_union_compact_page_descriptors=(
                self.settings.decode_gqa_fixed_mask_aiter
                and self.settings.levels == 2
                and speculative_steps < 2
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
            flat_page_indices=(page["page_indices"] if indexed_flat else None),
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
                and 0 < int(getattr(metadata, "max_seq_len", 0)) <= exact_decode_limit
            ),
            exact_leaf_lens=self.leaf_lens,
            output_buffer=output[:rows].unsqueeze(2),
        )
        if result.data_ptr() != output.data_ptr():
            raise AssertionError("fused LOD decode did not use the vLLM output buffer")
        return output


__all__ = ["VLLMLayerLODPool"]
