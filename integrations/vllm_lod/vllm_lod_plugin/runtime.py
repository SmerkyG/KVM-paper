"""vLLM lifecycle hooks for externally owned LOD attention state."""

from __future__ import annotations

import json
import os
import logging
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch

from lod_attention.kernels.paged_leaf_attention import stable_owner_ranks
from lod_attention._config import (
    ModelFamily,
    PREFIX_CACHE_LOCAL_WINDOW,
    ROUTE_COUNT,
    model_family,
)

from .backend import LODAttentionImpl
from .config import VLLMLODSettings, validate_production_scheduler
from .pool import VLLMLayerLODPool

logger = logging.getLogger(__name__)

_CROSS_LAYER_PREFILL_GROUP_OVERRIDDEN = (
    "LOD_KIMI_CROSS_LAYER_PREFILL_GROUP" in os.environ
)
_CROSS_LAYER_PREFILL_GROUP = int(
    os.environ.get("LOD_KIMI_CROSS_LAYER_PREFILL_GROUP", "12")
)
if _CROSS_LAYER_PREFILL_GROUP < 1:
    raise ValueError("LOD_KIMI_CROSS_LAYER_PREFILL_GROUP must be positive")
_PREFILL_RECLAIM_INTERVAL = int(
    os.environ.get("LOD_KIMI_PREFILL_RECLAIM_INTERVAL", "32768")
)
if _PREFILL_RECLAIM_INTERVAL < 0:
    raise ValueError("LOD_KIMI_PREFILL_RECLAIM_INTERVAL must be non-negative")
_DISTRIBUTED_PREFILL_BUILD = (
    os.environ.get("LOD_KIMI_DISTRIBUTED_PREFILL_BUILD", "0") == "1"
)


def _should_reclaim_prefill(*, total_len: int, prompt_capacity: int) -> bool:
    """Bound transient reservations without serializing every 16K chunk."""

    return (
        total_len >= prompt_capacity
        or (
            _PREFILL_RECLAIM_INTERVAL > 0
            and total_len % _PREFILL_RECLAIM_INTERVAL == 0
        )
    )


def _cross_layer_prefill_group_end(
    built_layers: int, total_layers: int, group_size: int
) -> int:
    """Return the next fixed-size layer boundary."""

    return min(built_layers + group_size, total_layers)


def _ensure_exact_lod_decode_capture_sizes(vllm_config: Any) -> None:
    """Prevent graph padding from aliasing another request's LOD cache row."""
    attention = getattr(vllm_config, "attention_config", None)
    backend = getattr(attention, "backend", None)
    if getattr(backend, "name", None) != "CUSTOM":
        return
    compilation = vllm_config.compilation_config
    capture_sizes = compilation.cudagraph_capture_sizes
    if not capture_sizes:
        return
    pool_size = VLLMLODSettings.from_environment().pool_size
    scheduler = getattr(vllm_config, "scheduler_config", None)
    max_requests = int(getattr(scheduler, "max_num_seqs", pool_size))
    original_max = int(
        compilation.max_cudagraph_capture_size or max(map(int, capture_sizes))
    )
    exact_rows = min(pool_size, max_requests)
    ordinary_limit = min(exact_rows, original_max)
    if ordinary_limit <= 0:
        return
    exact_sizes = set(range(1, ordinary_limit + 1))
    speculative_steps = int(vllm_config.num_speculative_tokens or 0) + 1
    if speculative_steps > 1:
        exact_sizes.update(
            rows * speculative_steps
            for rows in range(1, exact_rows + 1)
            if rows * speculative_steps <= original_max
        )
    compilation.cudagraph_capture_sizes = sorted(exact_sizes)
    compilation.max_cudagraph_capture_size = max(exact_sizes)


def _input_batch_max_query_len(input_batch: Any) -> int:
    """Read the largest scheduled query across old and new vLLM batches."""
    legacy = getattr(input_batch, "max_query_len", None)
    if legacy is not None:
        return max(int(legacy), 1)
    scheduled = getattr(input_batch, "num_scheduled_tokens", None)
    if scheduled is not None:
        values = np.asarray(scheduled)
        if values.size:
            return max(int(values.max()), 1)
    starts = getattr(input_batch, "query_start_loc_np", None)
    if starts is not None:
        values = np.diff(np.asarray(starts))
        if values.size:
            return max(int(values.max()), 1)
    return 1


@dataclass
class _CachedLODRow:
    row: int
    token_ids: np.ndarray
    total_length: int
    last_used: int


class VLLMLODRuntime:
    """Own fixed LOD pools for one vLLM model-runner process."""

    def __init__(self, model_state: Any) -> None:
        self.model_state = model_state
        self.settings = VLLMLODSettings.from_environment()
        config = model_state.vllm_config
        if not isinstance(config.additional_config, dict):
            raise TypeError(
                "vLLM LOD requires dict-valued additional_config so its "
                "compile-time settings can participate in the graph-cache key"
            )
        self.speculative_tokens = int(config.num_speculative_tokens or 0)
        self.hybrid_speculative_full_attention = False
        self.prefix_caching = bool(
            getattr(
                getattr(config, "cache_config", None),
                "enable_prefix_caching",
                False,
            )
        )
        self.max_requests = int(model_state.max_num_reqs)
        self.pool_size = min(self.settings.pool_size, self.max_requests)
        self.request_capacity = min(
            int(model_state.max_model_len),
            self.settings.request_capacity or int(model_state.max_model_len),
        )
        self.active_indices = torch.arange(
            self.max_requests, dtype=torch.long, device=model_state.device
        )
        context = config.compilation_config.static_forward_context
        # Native MTP/EAGLE is a separate autoregressive draft model.  It is
        # loaded into the same static forward context as the target before
        # ModelState is constructed, so selecting layers solely from the
        # custom backend would accidentally externalize the draft model's own
        # chronological K/V cache as LOD state.  A single draft token hid that
        # mistake because vLLM runs position zero through its prefill path;
        # recurrent positions build draft-only metadata and correctly fail
        # when that K/V group is absent.  Identify draft layers by object
        # ownership rather than a model-specific prefix, with the prefix only
        # as a compatibility fallback for model-state wrappers that do not
        # expose their target model.
        target_model = getattr(model_state, "model", None)
        if target_model is None:
            target_model = getattr(model_state, "get_model", lambda: None)()
        self.family = model_family(target_model) if target_model is not None else None
        target_module_ids = (
            {id(module) for module in target_model.modules()}
            if target_model is not None
            else set()
        )
        for name, layer in context.items():
            separate_draft_layer = bool(
                target_module_ids and id(layer) not in target_module_ids
            )
            prefix_fallback = bool(
                not target_module_ids
                and (name.startswith("mtp.") or ".mtp." in name)
            )
            if self.speculative_tokens and (
                separate_draft_layer or prefix_fallback
            ):
                impl = getattr(layer, "impl", None)
                if isinstance(impl, LODAttentionImpl):
                    impl.lod_eligible = False
                    layer._vllm_lod_native_speculator = True
        self._eligible_layers: dict[str, Any] = {
            name: layer
            for name, layer in context.items()
            if (
                (
                    isinstance(getattr(layer, "impl", None), LODAttentionImpl)
                    and bool(layer.impl.lod_eligible)
                )
                or (
                    self.family is ModelFamily.KIMI_K3
                    and bool(getattr(layer, "_vllm_lod_absorbed_mla", False))
                )
            )
        }
        self.layers: dict[str, Any] = self._eligible_layers
        self.dcp_world_size = int(
            config.parallel_config.decode_context_parallel_size
        )
        self.dcp_rank = 0
        self.dcp_group: Any | None = None
        if self.layers and self.dcp_world_size > 1:
            if self.family is not ModelFamily.KIMI_K3:
                raise NotImplementedError(
                    "the release DCP LoD path currently supports Kimi K3 MLA"
                )
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_group = get_dcp_group()
            self.dcp_rank = int(self.dcp_group.rank_in_group)
        self.pools: dict[str, VLLMLayerLODPool] = {}
        self.group_by_layer: dict[str, int] = {}
        self.block_size_by_group: dict[int, int] = {}
        self.req_to_slot: dict[str, int | str] = {}
        self.lod_row_by_slot: dict[int | str, int] = {}
        self.cached_rows: dict[int, _CachedLODRow] = {}
        self.request_states: Any | None = None
        self.cache_clock = 0
        self.free_lod_rows = list(range(self.pool_size - 1, -1, -1))
        self.logical_lengths = [0] * self.pool_size
        self._active_decode_rows: tuple[int, ...] | None = None
        self._initial_prefill_stages: dict[
            int, tuple[VLLMLayerLODPool, tuple[int, ...], int, int, int]
        ] = {}
        self._cached_prefill_stages: dict[
            int, tuple[VLLMLayerLODPool, tuple[int, ...], int, int, bool]
        ] = {}
        self._initial_prefill_sources: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._cached_prefill_sources: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._initial_prefill_built_layers = 0
        self._cached_prefill_built_layers = 0
        self._cross_layer_prefill_stream: torch.cuda.Stream | None = None
        self._prefill_attention_buffers: dict[str, torch.Tensor] = {}
        self.cross_layer_initial_prefill_batches = 0
        self.cross_layer_initial_prefill_layers = 0
        self.direct_prefill_rejection: str | None = None
        self.initialized = False
        self.allocate_pools()
        # vLLM's persistent graph cache otherwise sees only public settings;
        # a kernel-call ABI change in this plugin can incorrectly replay an
        # older captured graph after the package is upgraded.
        settings_json = json.dumps(
            {"implementation_abi": 2, "settings": asdict(self.settings)},
            sort_keys=True,
        )
        config.additional_config["lod_attention_compile_settings"] = sha256(
            settings_json.encode()
        ).hexdigest()

    @property
    def enabled(self) -> bool:
        return bool(self.layers)

    @property
    def cross_layer_prefill_group_size(self) -> int:
        """Return the cache-construction layer group for this model.

        K3 has exactly 24 absorbed-MLA cache producers in the full model. Four
        layers occupy its state-update kernels well while bounding staged K/V
        and score workspace; larger groups did not improve the complete-model
        timings and reduce the headroom needed at 256K+. Keep the established
        group of 12 for other model families, and retain the explicit
        environment override for isolated kernel experiments.
        """

        if (
            self.family is ModelFamily.KIMI_K3
            and not _CROSS_LAYER_PREFILL_GROUP_OVERRIDDEN
        ):
            return max(1, min(4, len(self.pools)))
        return _CROSS_LAYER_PREFILL_GROUP

    def _set_speculative_verification_routes(self, enabled: bool) -> None:
        """Keep speculative verification on the same top-eight calculation."""
        if self.speculative_tokens == 0:
            return
        del enabled
        for pool in self.pools.values():
            pool.engine.prefill_two_level_topk = ROUTE_COUNT

    def _prefix_rollback_tokens(self) -> int:
        cache_config = getattr(self.model_state.vllm_config, "cache_config", None)
        if not bool(getattr(cache_config, "enable_prefix_caching", False)):
            return 0
        return PREFIX_CACHE_LOCAL_WINDOW

    def allocate_pools(self) -> None:
        """Reserve LOD memory before vLLM profiles its native block budget."""
        if self.pools or not self.enabled:
            return
        capture_sizes = getattr(
            self.model_state.vllm_config.compilation_config,
            "cudagraph_capture_sizes",
            None,
        )
        decode_sizes = {
            int(size)
            for size in (capture_sizes or ())
            if 1 <= int(size) <= self.pool_size
        }
        if not decode_sizes:
            decode_sizes.add(self.pool_size)
        norm_flags = self._attention_norm_flags()
        prefix_rollback_tokens = self._prefix_rollback_tokens()
        # Kimi K3's static forward context can expose the same absorbed-MLA
        # module through more than one graph name.  Cache metadata still needs
        # every alias in ``self.layers``, but the authoritative semantic cache
        # belongs to the module object and must be allocated exactly once.
        # Keep the last name for each object because _attention_norm_flags()
        # resolves aliases the same way.
        unique_layers: dict[int, tuple[str, Any]] = {}
        for name, layer in self.layers.items():
            unique_layers[id(layer)] = (name, layer)
        for layer_index, (name, layer) in enumerate(unique_layers.values()):
            has_query_norm, has_key_norm = norm_flags.get(name, (False, False))
            pool = VLLMLayerLODPool(
                layer,
                settings=self.settings,
                max_requests=self.pool_size,
                request_capacity=self.request_capacity,
                active_indices=self.active_indices,
                dtype=self.model_state.dtype,
                device=self.model_state.device,
                has_query_norm=has_query_norm,
                has_key_norm=has_key_norm,
                prefix_rollback_tokens=prefix_rollback_tokens,
                speculative_tokens=self.speculative_tokens,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                dcp_group=self.dcp_group,
                dcp_interleave_size=int(
                    self.model_state.vllm_config.parallel_config.cp_kv_cache_interleave_size
                ),
            )
            for rows in sorted(decode_sizes):
                pool.reserve_decode_buffers(rows)
                if self.dcp_world_size > 1:
                    pool.reserve_dcp_decode_buffers(
                        rows, int(layer.num_heads) * self.dcp_world_size
                    )
            pool.initial_prefill_stager = self._stage_initial_prefill_layer
            pool.cached_prefill_stager = self._stage_cached_prefill_layer
            pool.engine._lod_prefill_attention_buffers = (
                self._prefill_attention_buffers
            )
            self.pools[name] = pool
            layer._vllm_lod_pool = pool
        if self.pools:
            resolved = next(iter(self.pools.values())).settings
            if any(pool.settings != resolved for pool in self.pools.values()):
                raise RuntimeError(
                    "LOD production does not support mixed attention geometries"
                )
            self.settings = resolved
            self._cross_layer_prefill_stream = torch.cuda.Stream(
                device=self.model_state.device
            )
            # DCP prefill temporarily owns a globally replicated shadow cache,
            # so it cannot use the fixed rank-local storage consumed by the
            # cross-layer batch builder.  It can still hide each layer's final
            # semantic-cache update behind later transformer layers.  Reuse a
            # single ordered stream for that work instead of creating one
            # stream per attention layer.
            for pool in self.pools.values():
                pool.deferred_prefill_stream = self._cross_layer_prefill_stream
        if self.pools:
            scheduler = self.model_state.vllm_config.scheduler_config
            validate_production_scheduler(
                max_model_len=int(self.model_state.max_model_len),
                max_num_batched_tokens=int(scheduler.max_num_batched_tokens),
                long_prefill_token_threshold=int(
                    scheduler.long_prefill_token_threshold
                ),
                required_prefill=max(
                    pool.settings.prefill_chunk_size
                    for pool in self.pools.values()
                ),
                required_decode_reserve=self.pool_size
                * max(1, self.speculative_tokens + 1),
                scheduler_cls=scheduler.scheduler_cls,
            )

    def _attention_norm_flags(self) -> dict[str, tuple[bool, bool]]:
        """Map vLLM Attention children to their parent module's Q/K norms."""
        model = getattr(self.model_state, "model", None)
        if model is None:
            model = getattr(self.model_state, "get_model", lambda: None)()
        if model is None:
            return {}
        flags: dict[str, tuple[bool, bool]] = {}
        layer_names = {id(layer): name for name, layer in self.layers.items()}
        for module in model.modules():
            joint = isinstance(getattr(module, "qk_norm", None), torch.nn.Module)
            has_query_norm = joint or isinstance(
                getattr(module, "q_norm", None), torch.nn.Module
            )
            has_key_norm = joint or isinstance(
                getattr(module, "k_norm", None), torch.nn.Module
            )
            for child in module.children():
                name = layer_names.get(id(child))
                if name is not None:
                    flags[name] = (has_query_norm, has_key_norm)
        return flags

    def initialize(self, kv_cache_config: Any) -> None:
        if self.initialized or not self.enabled:
            return
        # Cache specs are collected after model-state construction, so this is
        # the earliest lifecycle point at which every eligible Attention layer
        # must carry the marker installed by get_kv_cache_spec().
        missing_external = sorted(
            name
            for name, layer in self._eligible_layers.items()
            if not bool(getattr(layer, "_vllm_lod_external_kv_cache", False))
            and not bool(getattr(layer, "_vllm_lod_hybrid_native_kv", False))
        )
        if missing_external:
            raise RuntimeError(
                "LOD-eligible custom layers were not externalized by the cache "
                f"ownership hook: {missing_external}"
            )
        self.allocate_pools()
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                self.group_by_layer[name] = group_id
                layer = self.layers.get(name)
                if layer is not None:
                    cache = getattr(layer, "kv_cache", None)
                    if bool(
                        getattr(layer, "_vllm_lod_external_kv_cache", False)
                    ):
                        block_size = int(group.kv_cache_spec.block_size)
                    elif cache is None:
                        block_size = int(group.kv_cache_spec.block_size)
                    else:
                        if cache.ndim not in (4, 5):
                            raise ValueError("unexpected native vLLM KV cache rank")
                        block_size = int(cache.size(2))
                    prior = self.block_size_by_group.setdefault(group_id, block_size)
                    if prior != block_size:
                        raise ValueError("a vLLM KV group has mixed block sizes")
        # V2 leaves externally owned layers out of physical KV groups and
        # attaches their metadata builders to an existing native group. Record
        # that metadata mapping without creating a native cache or block table.
        for name, layer in self.layers.items():
            if name in self.group_by_layer:
                continue
            group_id = getattr(layer, "_vllm_lod_external_metadata_group", None)
            if group_id is None:
                continue
            group_id = int(group_id)
            if group_id >= len(kv_cache_config.kv_cache_groups):
                raise RuntimeError(
                    f"external LOD metadata group {group_id} is out of range"
                )
            group = kv_cache_config.kv_cache_groups[group_id]
            self.group_by_layer[name] = group_id
            block_size = int(group.kv_cache_spec.block_size)
            prior = self.block_size_by_group.setdefault(group_id, block_size)
            if prior != block_size:
                raise ValueError("a vLLM KV group has mixed block sizes")
        missing = self.layers.keys() - self.group_by_layer.keys()
        if missing:
            raise RuntimeError(
                f"LOD attention layers are missing from vLLM KV groups: {sorted(missing)}"
            )
        self.initialized = True
        logger.info(
            "Initialized LOD pools for %d global attention layers: "
            "levels=%d pool_rows=%d max_context=%d storage_bits=%d key_bits=%d "
            "value_bits=%d routing=%s prefill=%s",
            len(self.pools),
            self.settings.levels,
            self.pool_size,
            self.request_capacity,
            self.settings.kv_bits,
            self.settings.resolved_key_bits,
            self.settings.resolved_value_bits,
            "qk-norm-aware",
            "direct",
        )

    def prepare_legacy_runner(
        self,
        runner: Any,
        *,
        num_reqs: int,
        num_reqs_padded: int,
        max_query_len: int,
        for_capture: bool,
    ) -> None:
        """Prepare the persistent-batch runner used by released vLLM wheels."""
        if not self.initialized or not self.enabled:
            return
        if for_capture:
            self._prepare_dummy_batch(num_reqs_padded, max_query_len)
            return

        input_batch = runner.input_batch
        req_ids = [
            req_id for req_id in input_batch.req_ids[:num_reqs] if req_id is not None
        ]
        if len(req_ids) != num_reqs:
            if req_ids:
                raise RuntimeError("vLLM supplied a partially populated request batch")
            # Released vLLM wheels also perform uncaptured warmup runs whose
            # padded request count is nonzero but whose persistent batch has no
            # logical requests. Treat these exactly like graph-capture dummies.
            self._prepare_dummy_batch(num_reqs_padded, max_query_len)
            return
        live_requests = set(runner.requests)
        for req_id in tuple(self.req_to_slot):
            if req_id not in live_requests:
                self.remove_request(req_id)

        computed = input_batch.num_computed_tokens_cpu[:num_reqs]
        prompt_lengths = input_batch.num_prompt_tokens[:num_reqs]
        for row, req_id in enumerate(req_ids):
            if req_id in self.req_to_slot:
                continue
            self.req_to_slot[req_id] = req_id
            if int(computed[row]) <= 0:
                continue
            token_ids = self._legacy_token_ids(runner.requests[req_id])
            if token_ids is not None:
                self._restore_cached_prefix(req_id, token_ids, int(computed[row]))
        pure_decode = max_query_len == 1 and bool(np.all(computed >= prompt_lengths))
        if not pure_decode:
            query_starts = np.asarray(
                runner.query_start_loc.np[: num_reqs + 1], dtype=np.int64
            )
            if self._prepare_direct_prefill(
                req_ids, computed, query_starts, prompt_lengths
            ):
                return
            self._use_native_attention(req_ids)
            return

        if num_reqs_padded > self.pool_size:
            raise RuntimeError(
                "a padded pure-decode batch exceeds VLLM_LOD_POOL_SIZE; set "
                "VLLM_LOD_POOL_SIZE and --max-num-seqs to the same value"
            )
        for pool in self.pools.values():
            pool.decode_enabled = True
            pool.direct_prefill_plan = None

        lod_rows = []
        for req_id in req_ids:
            lod_rows.append(self._lod_row(req_id))
        mapped_rows = self._require_exact_decode_rows(lod_rows, num_reqs_padded)
        self._set_active_decode_rows(mapped_rows)
        missing_rows: list[str] = []
        catch_ups: list[tuple[int, int]] = []
        reference_pool = next(iter(self.pools.values()))
        for row, req_id in enumerate(req_ids):
            lod_row = self.lod_row_by_slot[req_id]
            length = int(computed[row])
            if not reference_pool.ready[lod_row]:
                missing_rows.append(req_id)
            else:
                catch_ups.append((lod_row, length))
        self._catch_up_decode_rows(catch_ups)
        if missing_rows:
            self._use_native_attention(missing_rows)

    def add_request(self, slot: int, data: Any) -> None:
        if os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1" and self.dcp_rank == 0:
            print(
                "KIMI_REQUEST_ADD "
                f"request={data.req_id} slot={slot} computed={data.num_computed_tokens}",
                flush=True,
            )
        self._release_lod_row(slot)
        self.req_to_slot[data.req_id] = slot
        token_ids = data.prefill_token_ids or data.prompt_token_ids
        if token_ids is None or int(data.num_computed_tokens) <= 0:
            return
        self._restore_cached_prefix(
            slot,
            token_ids,
            int(data.num_computed_tokens),
        )

    @staticmethod
    def _legacy_token_ids(request: Any) -> list[int] | None:
        prompt = getattr(request, "prompt_token_ids", None)
        if prompt is None:
            return None
        return [*prompt, *getattr(request, "output_token_ids", ())]

    def remove_request(
        self, req_id: str, *, token_ids: list[int] | None = None
    ) -> None:
        slot = self.req_to_slot.pop(req_id, None)
        if os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1" and self.dcp_rank == 0:
            print(f"KIMI_REQUEST_REMOVE request={req_id} slot={slot}", flush=True)
        if not self.req_to_slot:
            self._set_speculative_verification_routes(False)
        if slot is None:
            return
        row = self.lod_row_by_slot.pop(slot, None)
        if row is None:
            return
        if self.prefix_caching and self._cache_row(
            req_id, slot, row, token_ids=token_ids
        ):
            return
        self._free_lod_row(row)

    def _request_token_ids(
        self, req_id: str, slot: int | str, length: int
    ) -> np.ndarray | None:
        states = self.request_states
        if states is None or not isinstance(slot, int):
            return None
        req_index = states.req_id_to_index.get(req_id)
        if req_index is None:
            return None
        storage = getattr(getattr(states.all_token_ids, "_uva_buf", None), "cpu", None)
        if storage is None:
            return None
        values = storage[req_index, :length]
        if isinstance(values, torch.Tensor):
            return values.numpy().astype(np.int64, copy=True)
        return np.asarray(values, dtype=np.int64).copy()

    def _cache_row(
        self,
        req_id: str,
        slot: int | str,
        row: int,
        *,
        token_ids: list[int] | None = None,
    ) -> bool:
        if not all(pool.ready[row] for pool in self.pools.values()):
            return False
        total_length = self.logical_lengths[row]
        if total_length > 0:
            for pool in self.pools.values():
                pool.catch_up_many([(row, total_length)])
        lengths = {
            int(pool.metadata[row].get("total_len", -1))
            for pool in self.pools.values()
        }
        if len(lengths) != 1:
            return False
        total_length = lengths.pop()
        if total_length <= 0:
            return False
        cached_tokens = (
            np.asarray(token_ids[:total_length], dtype=np.int64).copy()
            if token_ids is not None
            else self._request_token_ids(req_id, slot, total_length)
        )
        if cached_tokens is None or len(cached_tokens) != total_length:
            return False
        self.cache_clock += 1
        self.cached_rows[row] = _CachedLODRow(
            row=row,
            token_ids=cached_tokens,
            total_length=total_length,
            last_used=self.cache_clock,
        )
        return True

    def _restore_cached_prefix(
        self, slot: int | str, token_ids: list[int], prefix_length: int
    ) -> bool:
        for pool in self.pools.values():
            pool.retained_restore_attempts += 1
            pool.retained_restore_last_prefix = prefix_length
        if not self.cached_rows:
            for pool in self.pools.values():
                pool.retained_restore_fail_no_row += 1
            return False
        prefix = np.asarray(token_ids[:prefix_length], dtype=np.int64)
        candidates = sorted(
            self.cached_rows.values(),
            key=lambda entry: entry.last_used,
            reverse=True,
        )
        saw_long_enough = False
        saw_token_match = False
        for entry in candidates:
            if entry.total_length < prefix_length:
                continue
            saw_long_enough = True
            if not np.array_equal(entry.token_ids[:prefix_length], prefix):
                continue
            saw_token_match = True
            coverages = [
                int(pool.metadata[entry.row].get("coverage", prefix_length + 1))
                for pool in self.pools.values()
            ]
            for pool, coverage in zip(self.pools.values(), coverages, strict=True):
                pool.retained_restore_last_coverage = coverage
                pool.retained_restore_last_total = entry.total_length
            for pool in self.pools.values():
                pool.restore_prefix(entry.row, prefix_length)
                pool.retained_reuse_count += 1
            self.cached_rows.pop(entry.row)
            self.lod_row_by_slot[slot] = entry.row
            self.logical_lengths[entry.row] = prefix_length
            self.cache_clock += 1
            return True
        for pool in self.pools.values():
            if not saw_long_enough:
                pool.retained_restore_fail_short += 1
            elif not saw_token_match:
                pool.retained_restore_fail_tokens += 1
            else:
                pool.retained_restore_fail_coverage += 1
        return False

    def _free_lod_row(self, row: int) -> None:
        self.cached_rows.pop(row, None)
        if self.initialized:
            for pool in self.pools.values():
                pool.reset(row)
        self.logical_lengths[row] = 0
        if row not in self.free_lod_rows:
            self.free_lod_rows.append(row)
            # New uniform batches should receive ascending contiguous rows so
            # Q/K/V remain packed and cache installation stays one batched
            # operation after any request-release order.
            self.free_lod_rows.sort(reverse=True)

    def _evict_cached_row(self) -> int | None:
        rows = self._evict_cached_rows(1)
        return rows[0] if rows else None

    def _evict_cached_rows(self, count: int) -> list[int]:
        """Evict retained rows with range-coalesced device resets."""
        if count <= 0 or not self.cached_rows:
            return []
        entries = sorted(
            self.cached_rows.values(), key=lambda item: item.last_used
        )[:count]
        rows = sorted(entry.row for entry in entries)
        for row in rows:
            self.cached_rows.pop(row)

        begin = 0
        while begin < len(rows):
            end = begin + 1
            while end < len(rows) and rows[end] == rows[end - 1] + 1:
                end += 1
            start_row = rows[begin]
            stop_row = rows[end - 1] + 1
            for pool in self.pools.values():
                pool._reset_range(start_row, stop_row)
            begin = end

        for row in rows:
            self.logical_lengths[row] = 0
        return rows

    def _release_lod_row(self, slot: int | str) -> None:
        row = self.lod_row_by_slot.pop(slot, None)
        if row is None:
            return
        self._free_lod_row(row)

    def _lod_row(self, slot: int | str) -> int:
        row = self.lod_row_by_slot.get(slot)
        if row is not None:
            return row
        if not self.free_lod_rows:
            row = self._evict_cached_row()
            if row is None:
                raise RuntimeError(
                    "active LOD requests exceed VLLM_LOD_POOL_SIZE; increase "
                    "the environment setting or reduce --max-num-seqs"
                )
        else:
            row = self.free_lod_rows.pop()
        self.lod_row_by_slot[slot] = row
        return row

    def _prepare_dummy_batch(self, rows: int, max_query_len: int) -> None:
        decode_capture = max_query_len == 1
        speculative_capture = (
            self.speculative_tokens > 0
            and max_query_len == self.speculative_tokens + 1
            and rows % max_query_len == 0
        )
        request_rows = rows // max_query_len if speculative_capture else rows
        for pool in self.pools.values():
            pool.decode_enabled = decode_capture
            pool.hybrid_full_decode = bool(
                self.hybrid_speculative_full_attention
                and (decode_capture or speculative_capture)
            )
            pool.speculative_decode_steps = (
                max_query_len if speculative_capture else 0
            )
            pool.direct_prefill_plan = None
            if speculative_capture:
                pool.reserve_speculative_decode_buffers(
                    request_rows, max_query_len
                )
        if not decode_capture and not speculative_capture:
            return
        if request_rows > self.pool_size:
            raise RuntimeError(
                "a captured decode batch exceeds VLLM_LOD_POOL_SIZE; "
                "set VLLM_LOD_POOL_SIZE and --max-num-seqs to the same value"
            )
        self.active_indices[:request_rows].copy_(
            torch.arange(request_rows, device=self.active_indices.device)
        )
        self._active_decode_rows = None
        for pool in self.pools.values():
            pool.local_lens.zero_()

    def _prepare_speculative_decode(
        self,
        slots: list[int],
        computed_lengths: np.ndarray,
        query_starts: np.ndarray,
        padded_tokens: int,
        steps: int,
    ) -> bool:
        """Prepare live rows for uniform graph-captured target verification."""
        if (
            self.speculative_tokens <= 0
            or steps != self.speculative_tokens + 1
            or len(query_starts) != len(slots) + 1
            or any(
                int(query_starts[row + 1] - query_starts[row]) != steps
                for row in range(len(slots))
            )
            or padded_tokens % steps
        ):
            return False
        padded_rows = padded_tokens // steps
        if padded_rows < len(slots) or padded_rows > self.pool_size:
            return False

        lod_rows = [self._lod_row(slot) for slot in slots]
        catch_ups: list[tuple[int, int]] = []
        for request_row, lod_row in enumerate(lod_rows):
            previous_length = int(computed_lengths[request_row])
            if previous_length <= 0:
                return False
            if previous_length + steps > self.request_capacity:
                raise RuntimeError(
                    "speculative decode would exceed VLLM_LOD_MAX_CONTEXT: "
                    f"prefix={previous_length}, proposal={steps}, "
                    f"capacity={self.request_capacity}"
                )
            totals = [
                int(pool.metadata[lod_row].get("total_len", -1))
                for pool in self.pools.values()
            ]
            if not all(pool.ready[lod_row] for pool in self.pools.values()):
                return False
            if any(total < previous_length for total in totals):
                # Mixed prefill/decode scheduling can run an accepted target
                # prefix through the captured LOD append before this host-side
                # bookkeeping row is visited again.  The device-local exact
                # suffix is authoritative in that case, just as it is for the
                # ordinary one-token catch-up path.  Accept the lag only when
                # every layer proves that the required exact K/V already
                # exists; the uncommon recovery sync avoids treating a truly
                # missing semantic prefix as a valid speculative row.
                for pool in self.pools.values():
                    coverage = int(pool.metadata[lod_row]["coverage"])
                    required_recent = previous_length - coverage
                    if not 0 <= required_recent <= pool.local_capacity:
                        return False
                    actual_recent = int(pool.local_lens[lod_row].item())
                    if actual_recent < required_recent:
                        return False
            if any(total > previous_length for total in totals):
                for pool in self.pools.values():
                    pool.restore_prefix(lod_row, previous_length)
            catch_ups.append((lod_row, previous_length))

        # Perform any infrequent 4K semantic refresh before replay. A proposal
        # is much shorter than the refresh interval, so all captured steps see
        # one immutable coarse field and append to its exact recent suffix.
        self._catch_up_decode_rows(catch_ups)
        mapped_rows = self._require_exact_decode_rows(lod_rows, padded_rows)
        self._set_active_decode_rows(mapped_rows)
        for pool in self.pools.values():
            pool.decode_enabled = False
            pool.speculative_decode_steps = steps
            pool.direct_prefill_plan = None
            pool.reserve_speculative_decode_buffers(padded_rows, steps)
            for lod_row, previous_length in catch_ups:
                metadata = pool.metadata[lod_row]
                proposed_length = previous_length + steps
                proposed_recent_length = proposed_length - int(
                    metadata["coverage"]
                )
                if proposed_recent_length > pool.local_capacity:
                    raise RuntimeError(
                        "speculative proposal exceeds the decode-local row: "
                        f"prefix={previous_length}, proposal={steps}, "
                        f"coverage={metadata['coverage']}, "
                        f"recent={proposed_recent_length}, "
                        f"capacity={pool.local_capacity}"
                    )
                metadata["total_len"] = proposed_length
                metadata["recent_len"] = (
                    proposed_length - int(metadata["coverage"])
                )
        for lod_row, previous_length in catch_ups:
            self.logical_lengths[lod_row] = previous_length + steps
        return True

    def _set_active_decode_rows(self, rows: list[int]) -> None:
        """Update the graph-visible row map only when the batch changes."""
        mapped = tuple(rows)
        for pool in self.pools.values():
            pool.active_decode_rows = mapped
        if mapped == self._active_decode_rows:
            return
        self.active_indices[: len(mapped)].copy_(
            torch.tensor(
                mapped,
                dtype=torch.long,
                device=self.active_indices.device,
            )
        )
        self._active_decode_rows = mapped

    def _catch_up_one_across_layers(
        self, row: int, total_length: int
    ) -> bool:
        """Batch one request's centroid update without moving layer caches."""

        profile_update = (
            os.environ.get("LOD_KIMI_PROFILE_DECODE_UPDATE") == "1"
            and self.dcp_rank == 0
        )
        profile_begin = (
            torch.cuda.Event(enable_timing=True) if profile_update else None
        )
        profile_end = (
            torch.cuda.Event(enable_timing=True) if profile_update else None
        )
        if profile_begin is not None:
            profile_begin.record(torch.cuda.current_stream(self.model_state.device))

        pools = tuple(self.pools.values())
        if len(pools) < 2:
            return False
        reference = pools[0]
        _, target_coverage = reference._catch_up_target(
            row, total_length
        )
        metadata = reference.metadata[row]
        coverage = int(metadata["coverage"])
        if coverage >= target_coverage:
            return False
        state_len = int(metadata["state_len"])
        scheduled_state_len = int(
            metadata.get("scheduled_state_len", state_len)
        )
        overflow_len = target_coverage - coverage
        scalar_names = (
            "state_len",
            "scheduled_state_len",
            "coverage",
            "recent_len",
            "leaf_count",
            "overflow_safe_until",
        )

        def update_signature(pool: VLLMLayerLODPool) -> tuple[object, ...]:
            engine = pool.engine
            return (
                type(engine),
                pool.kv_heads,
                pool.head_dim,
                pool.state_capacity,
                engine._streaming_state_geometry(),
                engine.state_premerge_factor,
                engine.state_clustering_centroid_rescale,
                engine.state_clustering_centroid_rescale_scope,
                engine.state_merge_before_append,
                engine.fused_state_update,
                engine.fused_state_maxsim,
            )

        reference_engine = reference.engine
        reference_key = update_signature(reference)
        reference_has_norms = isinstance(
            reference.state.get("key_norm_sums"), torch.Tensor
        )
        for pool in pools[1:]:
            engine = pool.engine
            pool_metadata = pool.metadata[row]
            if (
                update_signature(pool) != reference_key
                or engine.state_split_max_leaves is not None
                or any(
                    int(pool_metadata[name]) != int(metadata[name])
                    for name in scalar_names
                )
                or isinstance(pool.state.get("key_norm_sums"), torch.Tensor)
                != reference_has_norms
            ):
                return False
        if (
            reference_engine.state_split_max_leaves is not None
            or not reference_engine.fused_state_update
            or not reference_engine.fused_state_maxsim
        ):
            return False

        # Keep the established bounded workspace for non-DCP cross-layer
        # updates. DCP uses rank-local schedules and is handled separately.
        group_size = 16
        representative_indices = set(range(0, len(pools), group_size))
        # Prefill leaves one batch-one update workspace on every layer. Only
        # group representatives need one during layer-batched decode, so drop
        # the redundant references before allocating the bounded B=16 buffers.
        for index, pool in enumerate(pools):
            if index in representative_indices:
                continue
            for name in ("_lod_state_update_buffers", "_lod_state_maxsim_buffers"):
                if hasattr(pool.engine, name):
                    delattr(pool.engine, name)

        update_ctx_len = (
            int(reference_engine.local_len - reference_engine.chunk_len)
            + target_coverage
        )

        def pack(
            group: tuple[VLLMLayerLODPool, ...],
            name: str,
            length: int | None = None,
        ) -> torch.Tensor:
            tensors = [pool.state[name][row : row + 1] for pool in group]
            if length is not None:
                tensors = [tensor[..., :length, :] for tensor in tensors]
            return torch.cat(tensors, dim=0)

        updated_state_len: int | None = None
        for start in range(0, len(pools), group_size):
            group = pools[start : start + group_size]
            engine = group[0].engine
            self._attach_cross_layer_state_workspaces(engine)
            if getattr(engine, "_lod_cross_layer_decode_row", None) != row:
                buffers = getattr(engine, "_lod_state_maxsim_buffers", None)
                if isinstance(buffers, dict):
                    buffers.pop("_prepared_identity", None)
                    buffers.pop("_prepared_context_len", None)
                engine._lod_cross_layer_decode_row = row

            packed_k = pack(group, "state_k")
            absorbed_mla = all(pool.is_absorbed_mla for pool in group)
            packed_v = (
                packed_k[..., : reference.value_dim]
                if absorbed_mla
                else pack(group, "state_v")
            )
            packed_counts = pack(group, "counts")
            packed_recent_k = pack(group, "recent_k", overflow_len)
            packed_recent_v = (
                packed_recent_k[..., : reference.value_dim]
                if absorbed_mla
                else pack(group, "recent_v", overflow_len)
            )
            packed_norms = (
                pack(group, "key_norm_sums")
                if reference_has_norms
                else None
            )
            (
                packed_k,
                packed_v,
                packed_counts,
                group_state_len,
                owners,
                old_slot_remap,
            ) = engine._update_state(
                packed_k,
                packed_v,
                packed_counts,
                packed_norms,
                packed_recent_k,
                packed_recent_v,
                state_len=state_len,
                ctx_len=update_ctx_len,
                available_context=target_coverage,
                state_capacity=reference.state_capacity,
                clustering_query_scale=None,
                scheduled_state_len=scheduled_state_len,
                retain_prepared_geometry=False,
            )
            self._capture_cross_layer_state_workspaces(engine)
            if old_slot_remap is not None:
                raise AssertionError("paged state remapping is unsupported")
            if updated_state_len is None:
                updated_state_len = group_state_len
            elif updated_state_len != group_state_len:
                raise AssertionError("cross-layer LOD state schedules diverged")
            packed_state = {
                "state_k": packed_k,
                "counts": packed_counts,
            }
            if not absorbed_mla:
                packed_state["state_v"] = packed_v
            if packed_norms is not None:
                packed_state["key_norm_sums"] = packed_norms
            for group_row, pool in enumerate(group):
                active = slice(0, group_state_len)
                for name, packed in packed_state.items():
                    pool.state[name][row, :, active].copy_(
                        packed[group_row, :, active]
                    )
                pool.catch_up_precomputed(
                    row,
                    total_length,
                    state_len=group_state_len,
                    owners=owners[group_row : group_row + 1],
                )
        if updated_state_len is None:
            raise AssertionError("cross-layer LOD catch-up produced no update")
        if profile_begin is not None and profile_end is not None:
            profile_end.record(torch.cuda.current_stream(self.model_state.device))
            profile_end.synchronize()
            print(
                "KIMI_DECODE_UPDATE "
                f"total={total_length} coverage={coverage} "
                f"target={target_coverage} overflow={overflow_len} "
                f"state={state_len}->{updated_state_len} "
                f"elapsed={profile_begin.elapsed_time(profile_end):.3f}ms",
                flush=True,
            )
        return True

    def _launch_cross_layer_prefill_group(
        self,
        stages: tuple[tuple[Any, ...], ...],
        builder: Any,
        *,
        record_tensors: list[torch.Tensor] | None = None,
        release_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        reclaim_after: bool = False,
        release_state_workspaces: bool = False,
        synchronize_after: bool = False,
    ) -> None:
        """Queue one layer-batched cache update behind its producing layers."""

        stream = self._cross_layer_prefill_stream
        if stream is None:
            raise RuntimeError("cross-layer prefill stream is not initialized")
        foreground = torch.cuda.current_stream(self.model_state.device)
        stream.wait_stream(foreground)
        profile = os.environ.get("LOD_KIMI_PROFILE_PREFILL") == "1"
        async_profile = (
            os.environ.get("LOD_KIMI_PROFILE_ASYNC_PREFILL_UPDATES") == "1"
        )
        memory_profile = os.environ.get("LOD_KIMI_PROFILE_MEMORY") == "1"
        profile_begin = (
            torch.cuda.Event(enable_timing=True)
            if profile or async_profile
            else None
        )
        profile_end = (
            torch.cuda.Event(enable_timing=True)
            if profile or async_profile
            else None
        )
        host_begin = time.perf_counter() if async_profile else 0.0
        with torch.cuda.stream(stream):
            if profile_begin is not None:
                profile_begin.record(stream)
            try:
                builder(stages)
            finally:
                # INT4 construction sources are transient. Record their use on
                # this stream before the runtime drops its final references.
                for tensor in record_tensors or ():
                    tensor.record_stream(stream)
            completed = torch.cuda.Event()
            completed.record(stream)
            if profile_end is not None:
                profile_end.record(stream)
        host_elapsed_ms = (
            1_000.0 * (time.perf_counter() - host_begin)
            if async_profile
            else 0.0
        )
        if async_profile:
            pending_profiles = getattr(
                self, "_cross_layer_prefill_profile_events", None
            )
            if pending_profiles is None:
                pending_profiles = []
                self._cross_layer_prefill_profile_events = pending_profiles
            pending_profiles.append(
                (len(stages), profile_begin, profile_end, host_elapsed_ms)
            )
        # Do not make the foreground transformer stream wait here.  The cache
        # update is independent of later layers and is intentionally hidden
        # behind them; each source tensor was recorded on this stream above,
        # and the next scheduler step synchronizes the per-row event before it
        # can consume the cache.  A foreground wait serialized every update
        # with the model and erased the prefill speedup at the 32K crossover.
        serialize_update = (
            profile
            or memory_profile
            or reclaim_after
            or release_state_workspaces
            or synchronize_after
            or os.environ.get("LOD_KIMI_SERIALIZE_CROSS_LAYER_PREFILL") == "1"
        )
        profile_reclaim = (
            os.environ.get("LOD_KIMI_PROFILE_RECLAIM") == "1"
            and self.dcp_rank == 0
        )
        reclaim_started = time.perf_counter() if profile_reclaim else 0.0
        if serialize_update:
            completed.synchronize()
        if async_profile and synchronize_after:
            pending_profiles = self._cross_layer_prefill_profile_events
            # ``completed`` is recorded immediately before the optional
            # timing endpoint, so waiting on it does not imply that the
            # endpoint itself is queryable yet.  This matters when a single
            # deferred final build replaces the usual six subgroup launches.
            if profile_end is not None:
                profile_end.synchronize()
            elapsed = [
                begin.elapsed_time(end)
                for _layers, begin, end, _host_ms in pending_profiles
            ]
            host_elapsed = [
                host_ms for _layers, _begin, _end, host_ms in pending_profiles
            ]
            print(
                "KIMI_ASYNC_PREFILL_UPDATES "
                f"groups={len(elapsed)} total_ms={sum(elapsed):.3f} "
                f"group_ms={','.join(f'{value:.3f}' for value in elapsed)} "
                f"host_total_ms={sum(host_elapsed):.3f} "
                f"host_group_ms="
                f"{','.join(f'{value:.3f}' for value in host_elapsed)}",
                flush=True,
            )
            pending_profiles.clear()
        synchronized_at = time.perf_counter() if profile_reclaim else 0.0
        if release_state_workspaces:
            # The maximum-capacity Kimi construction workspace is useful
            # while processing successive 16K prefill chunks, but decode
            # attention does not consume it.  Keeping it reachable from each
            # group representative needlessly reduces the usable KV-cache
            # headroom for the rest of the request.  The completion fence
            # above makes it safe to drop here; a later multi-turn/decode
            # catch-up recreates only the workspace its overflow requires.
            self._release_cross_layer_state_workspaces()
        if reclaim_after:
            # A completed 16K Kimi construction chunk leaves several large,
            # differently shaped staging blocks in PyTorch's allocator cache.
            # No later layer can overlap this final group, and the following
            # scheduler chunk must consume its cache, so release those idle
            # blocks at this existing dependency boundary instead of allowing
            # reservations to grow with context length.  The staging
            # containers belong to the caller, so clear those references
            # *before* empty_cache().  Previously they survived until after
            # this function returned; each reclaim could therefore release
            # only the preceding chunk and the allocator peak grew to the
            # physical VRAM limit at 256K.
            if release_sources is not None:
                release_sources.clear()
            if record_tensors is not None:
                record_tensors.clear()
            # record_stream() defers recycling through a fresh allocator
            # event created when the final Python reference is released.  The
            # construction event above necessarily predates that event; wait
            # once more after clearing the containers so empty_cache() can
            # actually return this chunk's blocks rather than the preceding
            # chunk's blocks.
            stream.synchronize()
            torch.cuda.empty_cache()
        elif release_state_workspaces:
            torch.cuda.empty_cache()
        if (
            release_state_workspaces
            and os.environ.get("LOD_KIMI_PROFILE_FINAL_PREFILL_MEMORY") == "1"
            and self.dcp_rank == 0
        ):
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                self.model_state.device
            )
            print(
                "KIMI_FINAL_PREFILL_MEMORY "
                f"allocated="
                f"{torch.cuda.memory_allocated(self.model_state.device)} "
                f"reserved="
                f"{torch.cuda.memory_reserved(self.model_state.device)} "
                f"free={free_bytes} total={total_bytes}",
                flush=True,
            )
        if profile_reclaim:
            reclaim_finished = time.perf_counter()
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                self.model_state.device
            )
            persistent_bytes = sum(
                pool.persistent_cache_nbytes() for pool in self.pools.values()
            )
            shadow_bytes = sum(
                pool.dcp_prefill_shadow_nbytes() for pool in self.pools.values()
            )
            print(
                "KIMI_PREFILL_RECLAIM "
                f"layers={len(stages)} wait_ms="
                f"{1_000.0 * (synchronized_at - reclaim_started):.3f} "
                f"empty_cache_ms="
                f"{1_000.0 * (reclaim_finished - synchronized_at):.3f} "
                f"allocated={torch.cuda.memory_allocated(self.model_state.device)} "
                f"reserved={torch.cuda.memory_reserved(self.model_state.device)} "
                f"persistent={persistent_bytes} shadow={shadow_bytes} "
                f"free={free_bytes} total={total_bytes}",
                flush=True,
            )
        if profile and profile_begin is not None and profile_end is not None:
            # ``profile_end`` is deliberately recorded after the completion
            # event used by consumers, so waiting on that earlier event does
            # not guarantee the timing endpoint itself has completed.
            profile_end.synchronize()
            print(
                "KIMI_CROSS_LAYER_UPDATE "
                f"layers={len(stages)} "
                f"elapsed={profile_begin.elapsed_time(profile_end):.3f}ms",
                flush=True,
            )
        if memory_profile and self.dcp_rank == 0:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.model_state.device)
            print(
                "KIMI_CROSS_LAYER_MEMORY "
                f"schedule={stages[0][2:]} "
                f"allocated={torch.cuda.memory_allocated(self.model_state.device)} "
                f"reserved={torch.cuda.memory_reserved(self.model_state.device)} "
                f"free={free_bytes} total={total_bytes}",
                flush=True,
            )
        for stage in stages:
            pool = stage[0]
            slots = stage[1]
            if not slots:
                raise AssertionError("cross-layer prefill requires cache rows")
            for slot in slots:
                if pool.deferred_prefill_events[slot] is not None:
                    raise RuntimeError(
                        "cross-layer prefill reused an unfinished cache row"
                    )
                pool.deferred_prefill_events[slot] = completed

    def _attach_cross_layer_state_workspaces(self, engine: Any) -> None:
        """Share the large state-update scratch across sequential layer groups."""

        for name in ("_lod_state_update_buffers", "_lod_state_maxsim_buffers"):
            shared = getattr(self, f"_cross_layer_shared{name}", None)
            if isinstance(shared, dict):
                setattr(engine, name, shared)

    def _capture_cross_layer_state_workspaces(self, engine: Any) -> None:
        """Remember scratch allocated by one group for the following groups."""

        for name in ("_lod_state_update_buffers", "_lod_state_maxsim_buffers"):
            buffers = getattr(engine, name, None)
            if isinstance(buffers, dict):
                setattr(self, f"_cross_layer_shared{name}", buffers)

    def _release_cross_layer_state_workspaces(self) -> None:
        """Drop construction-only scratch after the final prefill update."""

        names = ("_lod_state_update_buffers", "_lod_state_maxsim_buffers")
        buffers_by_id: dict[int, dict[str, torch.Tensor]] = {}
        for name in names:
            shared_name = f"_cross_layer_shared{name}"
            shared = getattr(self, shared_name, None)
            if isinstance(shared, dict):
                buffers_by_id[id(shared)] = shared
            if hasattr(self, shared_name):
                delattr(self, shared_name)
            for pool in self.pools.values():
                buffers = getattr(pool.engine, name, None)
                if isinstance(buffers, dict):
                    buffers_by_id[id(buffers)] = buffers
                if hasattr(pool.engine, name):
                    delattr(pool.engine, name)
        for buffers in buffers_by_id.values():
            buffers.clear()

    def _stage_initial_prefill_layer(
        self,
        pool: VLLMLayerLODPool,
        slots: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        total_len: int,
        coverage: int,
        prompt_capacity: int,
    ) -> None:
        """Collect exact-prefix K/V and batch its state update across layers."""

        identity = id(pool)
        if identity in self._initial_prefill_stages:
            raise RuntimeError("an LOD layer staged the same initial prefix twice")
        expected = next(iter(self._initial_prefill_stages.values()), None)
        if expected is not None and (
            expected[1] != slots
            or expected[2] != total_len
            or expected[3] != coverage
            or expected[4] != prompt_capacity
        ):
            raise RuntimeError("cross-layer initial-prefix schedules diverged")
        staged_source = pool._stage_cross_layer_initial_cache(
            slots,
            key,
            value,
            coverage=coverage,
            prompt_capacity=prompt_capacity,
        )
        if staged_source is not None:
            self._initial_prefill_sources[identity] = staged_source
        self._initial_prefill_stages[identity] = (
            pool,
            slots,
            total_len,
            coverage,
            prompt_capacity,
        )
        staged_layers = len(self._initial_prefill_stages)
        if (
            os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1"
            and self.dcp_rank == 0
            and staged_layers
            in (1, self.cross_layer_prefill_group_size, len(self.pools))
        ):
            print(
                "KIMI_INITIAL_STAGE "
                f"slot={slots} total={total_len} layers={staged_layers} "
                f"built={self._initial_prefill_built_layers}",
                flush=True,
            )
        next_group_end = _cross_layer_prefill_group_end(
            self._initial_prefill_built_layers,
            len(self.pools),
            self.cross_layer_prefill_group_size,
        )
        if staged_layers < next_group_end:
            return
        ordered = tuple(self._initial_prefill_stages.values())[
            self._initial_prefill_built_layers :
        ]
        if not 0 < len(ordered) <= self.cross_layer_prefill_group_size:
            raise AssertionError("invalid cross-layer initial prefill group")
        staged_sources = {
            id(group_pool): self._initial_prefill_sources.pop(id(group_pool))
            for group_pool, *_ in ordered
            if id(group_pool) in self._initial_prefill_sources
        }
        record_tensors = [
            tensor for source in staged_sources.values() for tensor in source
        ]
        staged_source = None
        try:
            self._launch_cross_layer_prefill_group(
                ordered,
                lambda group: self._build_initial_prefill_across_layers(
                    group, staged_sources=staged_sources
                ),
                record_tensors=record_tensors,
                release_sources=staged_sources,
                # vLLM records the first token when the foreground model pass
                # completes, but Kimi's LoD cache may still be finishing on
                # this background stream.  Timing runs fence only the final
                # layer group so prefill owns that one-time construction tail
                # instead of amortizing it into steady-state decode.  Earlier
                # layer groups remain fully overlapped with the transformer.
                synchronize_after=(
                    staged_layers == len(self.pools)
                    and os.environ.get("LOD_BENCHMARK_SYNC_PREFILL_CACHE") == "1"
                ),
                reclaim_after=(
                    staged_layers == len(self.pools)
                    and _should_reclaim_prefill(
                        total_len=total_len,
                        prompt_capacity=prompt_capacity,
                    )
                ),
                release_state_workspaces=(
                    staged_layers == len(self.pools)
                    and total_len >= prompt_capacity
                ),
            )
        except Exception:
            self._initial_prefill_stages.clear()
            self._initial_prefill_sources.clear()
            self._initial_prefill_built_layers = 0
            raise
        self._initial_prefill_built_layers = staged_layers
        if os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1" and self.dcp_rank == 0:
            print(
                "KIMI_INITIAL_PUBLISH "
                f"slot={slots} layers={len(ordered)} staged={staged_layers}",
                flush=True,
            )
        self.cross_layer_initial_prefill_layers += len(ordered)
        if staged_layers == len(self.pools):
            self.cross_layer_initial_prefill_batches += 1
            self._initial_prefill_stages.clear()
            self._initial_prefill_sources.clear()
            self._initial_prefill_built_layers = 0

    def _flush_pending_initial_prefill_layers(self) -> None:
        """Finish a short final layer group before the next scheduler step."""

        staged_layers = len(self._initial_prefill_stages)
        if staged_layers <= self._initial_prefill_built_layers:
            return
        ordered = tuple(self._initial_prefill_stages.values())[
            self._initial_prefill_built_layers :
        ]
        staged_sources = {
            id(group_pool): self._initial_prefill_sources.pop(id(group_pool))
            for group_pool, *_ in ordered
            if id(group_pool) in self._initial_prefill_sources
        }
        record_tensors = [
            tensor for source in staged_sources.values() for tensor in source
        ]
        try:
            self._launch_cross_layer_prefill_group(
                ordered,
                lambda group: self._build_initial_prefill_across_layers(
                    group, staged_sources=staged_sources
                ),
                record_tensors=record_tensors,
                release_sources=staged_sources,
                reclaim_after=True,
                release_state_workspaces=True,
            )
        finally:
            self._initial_prefill_stages.clear()
            self._initial_prefill_sources.clear()
            self._initial_prefill_built_layers = 0
        self.cross_layer_initial_prefill_layers += len(ordered)
        self.cross_layer_initial_prefill_batches += 1

    def _build_initial_prefill_across_layers(
        self,
        stages: tuple[
            tuple[VLLMLayerLODPool, tuple[int, ...], int, int, int], ...
        ],
        *,
        staged_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Reuse the decode catch-up batching geometry for initial prefixes."""

        staged_sources = {} if staged_sources is None else staged_sources
        reference, slots, total_len, coverage, prompt_capacity = stages[0]
        expected_staged = (
            reference.settings.kv_bits == 4 or reference.dcp_world_size > 1
        )
        if expected_staged != (len(staged_sources) == len(stages)):
            raise RuntimeError("cross-layer initial staging format diverged")
        if not slots:
            raise AssertionError("cross-layer initial construction requires B=1")

        # A prompt that ends in this scheduler pass never consumes the
        # temporary globally replicated DCP shadow.  The old path first built
        # that global cache and then immediately reconstructed the persistent
        # rank-local cache from its chronological archive.  Build the same
        # rank-local cache directly from the exact staged records instead.
        # This is especially important at 16K and 32K, where the discarded
        # global update is otherwise a large fraction of total prefill time.
        if reference.dcp_world_size > 1 and total_len >= prompt_capacity:
            self._build_final_dcp_prefill_group(
                tuple(pool for pool, *_rest in stages),
                slots,
                global_length=total_len,
                staged_sources=staged_sources,
            )
            return
        sink_len = min(int(reference.engine.sink_len), total_len)
        initial_len = min(total_len, int(reference.engine.chunk_len))
        initial_state_len = initial_len - sink_len
        overflow_len = coverage - initial_len
        if overflow_len <= 0:
            raise AssertionError("cross-layer initial construction has no overflow")

        def signature(pool: VLLMLayerLODPool) -> tuple[object, ...]:
            engine = pool.engine
            return (
                type(engine),
                pool.kv_heads,
                pool.head_dim,
                pool.state_capacity,
                engine._streaming_state_geometry(),
                engine.state_premerge_factor,
                engine.state_clustering_centroid_rescale,
                engine.state_clustering_centroid_rescale_scope,
                engine.state_merge_before_append,
                engine.fused_state_update,
                engine.fused_state_maxsim,
            )

        expected_signature = signature(reference)
        has_norms = isinstance(reference.state.get("key_norm_sums"), torch.Tensor)
        for (
            pool,
            other_slots,
            other_total,
            other_coverage,
            other_capacity,
        ) in stages[1:]:
            if (
                other_slots != slots
                or other_total != total_len
                or other_coverage != coverage
                or other_capacity != prompt_capacity
                or signature(pool) != expected_signature
                or isinstance(pool.state.get("key_norm_sums"), torch.Tensor)
                != has_norms
            ):
                raise RuntimeError("cross-layer initial cache geometries diverged")

        group_size = 16
        slot = slots[0]

        if reference.dcp_world_size > 1:
            self._build_dcp_initial_prefill_across_layers(
                stages,
                staged_sources=staged_sources,
            )
            return

        def pack_state(
            group: tuple[
                tuple[VLLMLayerLODPool, tuple[int, ...], int, int, int], ...
            ],
            name: str,
        ) -> torch.Tensor:
            return torch.cat(
                [pool.state[name][slot : slot + 1] for pool, *_ in group],
                dim=0,
            )

        def archive_source(
            pool: VLLMLayerLODPool, name: str
        ) -> torch.Tensor:
            staged = staged_sources.get(id(pool))
            if staged is not None:
                return staged[0 if name == "leaf_k" else 1]
            value = pool.state["page_cache"].get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError("cross-layer BF16 archive is missing")
            return value[slot : slot + 1]

        for start in range(0, len(stages), group_size):
            group = stages[start : start + group_size]
            engine = group[0][0].engine
            self._attach_cross_layer_state_workspaces(engine)
            buffers = getattr(engine, "_lod_state_maxsim_buffers", None)
            if isinstance(buffers, dict):
                buffers.pop("_prepared_identity", None)
                buffers.pop("_prepared_context_len", None)
            packed_k = pack_state(group, "state_k")
            packed_v = (
                packed_k[..., : reference.value_dim]
                if all(pool.is_absorbed_mla for pool, *_ in group)
                else pack_state(group, "state_v")
            )
            packed_counts = pack_state(group, "counts")
            packed_norms = pack_state(group, "key_norm_sums") if has_norms else None
            overflow_k = torch.cat(
                [
                    archive_source(pool, "leaf_k")[
                        ..., initial_state_len : initial_state_len + overflow_len, :
                    ]
                    for pool, *_ in group
                ],
                dim=0,
            )
            overflow_v = torch.cat(
                [
                    archive_source(pool, "leaf_v")[
                        ..., initial_state_len : initial_state_len + overflow_len, :
                    ]
                    for pool, *_ in group
                ],
                dim=0,
            )
            (
                packed_k,
                packed_v,
                packed_counts,
                state_len,
                owners,
                old_slot_remap,
            ) = engine._update_state(
                packed_k,
                packed_v,
                packed_counts,
                packed_norms,
                overflow_k,
                overflow_v,
                state_len=initial_state_len,
                ctx_len=total_len,
                available_context=coverage,
                state_capacity=reference.state_capacity,
                clustering_query_scale=None,
                scheduled_state_len=initial_state_len,
                retain_prepared_geometry=False,
            )
            self._capture_cross_layer_state_workspaces(engine)
            if old_slot_remap is not None:
                raise AssertionError("paged initial state remapping is unsupported")
            owner_ranks = stable_owner_ranks(owners)
            packed_state = {
                "state_k": packed_k,
                "state_v": packed_v,
                "counts": packed_counts,
            }
            if packed_norms is not None:
                packed_state["key_norm_sums"] = packed_norms
            for group_row, (pool, group_slots, *_rest) in enumerate(group):
                active = slice(0, state_len)
                for name, packed in packed_state.items():
                    pool.state[name][slot, :, active].copy_(
                        packed[group_row, :, active]
                    )
                pool._finish_cross_layer_initial_cache(
                    group_slots,
                    total_len=total_len,
                    coverage=coverage,
                    state_len=state_len,
                    owners=owners[group_row : group_row + 1],
                    owner_ranks=owner_ranks[group_row : group_row + 1],
                    staged_leaves=staged_sources.get(id(pool)),
                )

    def _build_dcp_initial_prefill_across_layers(
        self,
        stages: tuple[
            tuple[VLLMLayerLODPool, tuple[int, ...], int, int, int], ...
        ],
        *,
        staged_sources: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Build replicated DCP prefill shadows in bounded layer batches.

        The first 16K scheduler block is exact attention. Its LoD cache is
        therefore independent of the output calculation and can use a small
        layer-batched state-update geometry instead of launching one
        underoccupied cache build per transformer layer.  Four layers keeps
        the final exposed wave short without starving the construction kernels.
        """

        reference, slots, total_len, coverage, prompt_capacity = stages[0]
        if reference.dcp_world_size <= 1 or not slots:
            raise AssertionError("DCP cross-layer construction requires rows")
        if len(staged_sources) != len(stages):
            raise RuntimeError("DCP cross-layer construction lost staged K/V")
        sink_len = min(int(reference.engine.sink_len), total_len)
        initial_len = min(total_len, int(reference.engine.chunk_len))
        initial_state_len = initial_len - sink_len
        archived_len = coverage - sink_len
        overflow_len = archived_len - initial_state_len
        if overflow_len <= 0:
            raise AssertionError("DCP cross-layer construction has no overflow")
        group_size = self.cross_layer_prefill_group_size
        row_count = len(slots)

        distributed = _DISTRIBUTED_PREFILL_BUILD
        if distributed and (
            self.dcp_group is None
            or group_size != reference.dcp_world_size
            or len(stages) % group_size
            or not all(pool.is_absorbed_mla for pool, *_ in stages)
        ):
            raise RuntimeError(
                "distributed Kimi prefill construction requires one layer "
                "per DCP rank in every cross-layer group"
            )

        for start in range(0, len(stages), group_size):
            group = stages[start : start + group_size]
            build_group = (
                group[self.dcp_rank : self.dcp_rank + 1]
                if distributed
                else group
            )
            engine = build_group[0][0].engine
            self._attach_cross_layer_state_workspaces(engine)
            state_capacity = engine._state_capacity(
                prompt_capacity, initial_state_len
            )
            first_k = staged_sources[id(build_group[0][0])][0]
            batch = len(build_group) * row_count
            state_k = first_k.new_zeros(
                batch,
                reference.kv_heads,
                state_capacity,
                reference.head_dim,
            )
            if all(pool.is_absorbed_mla for pool, *_ in build_group):
                state_v = state_k[..., : reference.value_dim]
            else:
                first_v = staged_sources[id(build_group[0][0])][1]
                state_v = first_v.new_zeros(
                    batch,
                    reference.kv_heads,
                    state_capacity,
                    reference.value_dim,
                )
            counts = torch.zeros(
                batch,
                reference.kv_heads,
                state_capacity,
                1,
                dtype=torch.float32,
                device=first_k.device,
            )
            initial_k = torch.cat(
                [
                    staged_sources[id(pool)][0][..., :initial_state_len, :]
                    for pool, *_ in build_group
                ],
                dim=0,
            )
            initial_v = (
                initial_k[..., : reference.value_dim]
                if all(pool.is_absorbed_mla for pool, *_ in build_group)
                else torch.cat(
                    [
                        staged_sources[id(pool)][1][..., :initial_state_len, :]
                        for pool, *_ in build_group
                    ],
                    dim=0,
                )
            )
            state_k[..., :initial_state_len, :].copy_(initial_k)
            state_v[..., :initial_state_len, :].copy_(initial_v)
            counts[..., :initial_state_len, :].fill_(1.0)
            has_norms = engine.state_clustering_centroid_rescale != "none"
            key_norm_sums = (
                torch.zeros_like(counts) if has_norms else None
            )
            if key_norm_sums is not None:
                key_norm_sums[..., :initial_state_len, :].copy_(
                    engine._state_clustering_constituent_rms(initial_k)
                )
            overflow_k = torch.cat(
                [
                    staged_sources[id(pool)][0][
                        ..., initial_state_len:archived_len, :
                    ]
                    for pool, *_ in build_group
                ],
                dim=0,
            )
            overflow_v = (
                overflow_k[..., : reference.value_dim]
                if all(pool.is_absorbed_mla for pool, *_ in build_group)
                else torch.cat(
                    [
                        staged_sources[id(pool)][1][
                            ..., initial_state_len:archived_len, :
                        ]
                        for pool, *_ in build_group
                    ],
                    dim=0,
                )
            )
            buffers = getattr(engine, "_lod_state_maxsim_buffers", None)
            if isinstance(buffers, dict):
                buffers.pop("_prepared_identity", None)
                buffers.pop("_prepared_context_len", None)
            (
                state_k,
                state_v,
                counts,
                state_len,
                owners,
                old_slot_remap,
            ) = engine._update_state(
                state_k,
                state_v,
                counts,
                key_norm_sums,
                overflow_k,
                overflow_v,
                state_len=initial_state_len,
                ctx_len=total_len,
                available_context=coverage,
                state_capacity=state_capacity,
                clustering_query_scale=None,
                scheduled_state_len=initial_state_len,
                retain_prepared_geometry=False,
            )
            self._capture_cross_layer_state_workspaces(engine)
            if old_slot_remap is not None:
                raise AssertionError("DCP initial state remapping is unsupported")
            if distributed:
                process_group = self.dcp_group
                if process_group is None:
                    raise AssertionError("distributed prefill lost its DCP group")
                active_k = process_group.all_gather(
                    state_k[..., :state_len, :], dim=0
                )
                active_counts = process_group.all_gather(
                    counts[..., :state_len, :], dim=0
                )
                owners = process_group.all_gather(
                    owners.to(torch.float32), dim=0
                ).to(torch.long)
                active_norms = (
                    None
                    if key_norm_sums is None
                    else process_group.all_gather(
                        key_norm_sums[..., :state_len, :], dim=0
                    )
                )
                global_batch = len(group) * row_count
                state_k = active_k.new_zeros(
                    global_batch,
                    reference.kv_heads,
                    state_capacity,
                    reference.head_dim,
                )
                state_k[..., :state_len, :].copy_(active_k)
                state_v = state_k[..., : reference.value_dim]
                counts = active_counts.new_zeros(
                    global_batch,
                    reference.kv_heads,
                    state_capacity,
                    1,
                )
                counts[..., :state_len, :].copy_(active_counts)
                if active_norms is None:
                    key_norm_sums = None
                else:
                    key_norm_sums = active_norms.new_zeros(
                        global_batch,
                        reference.kv_heads,
                        state_capacity,
                        1,
                    )
                    key_norm_sums[..., :state_len, :].copy_(active_norms)
            owner_ranks = stable_owner_ranks(owners)
            for group_row, (pool, *_rest) in enumerate(group):
                archive_k, archive_v = staged_sources[id(pool)]
                row_begin = group_row * row_count
                row_end = row_begin + row_count
                pool._finish_dcp_cross_layer_initial_cache(
                    slots,
                    total_len=total_len,
                    coverage=coverage,
                    state_k=state_k[row_begin:row_end],
                    state_v=state_v[row_begin:row_end],
                    counts=counts[row_begin:row_end],
                    key_norm_sums=(
                        None
                        if key_norm_sums is None
                        else key_norm_sums[row_begin:row_end]
                    ),
                    state_len=state_len,
                    owners=owners[row_begin:row_end],
                    owner_ranks=owner_ranks[row_begin:row_end],
                    archive_k=archive_k,
                    archive_v=archive_v,
                    prompt_capacity=prompt_capacity,
                )
            if total_len >= prompt_capacity:
                # The global shadow is only an intermediate representation.
                # Convert each completed layer group to its rank-local DCP
                # cache now, while later transformer layers are still running,
                # instead of serializing every layer at the decode boundary.
                self._shard_dcp_prefill_pool_group(
                    tuple(pool for pool, *_rest in group),
                    slots,
                    global_length=total_len,
                )

    def _stage_cached_prefill_layer(
        self,
        pool: VLLMLayerLODPool,
        slots: tuple[int, ...],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        previous_len: int,
        total_len: int,
        finalize_cache_for_decode: bool,
    ) -> None:
        """Collect an aligned continuation for one layer-batched update."""

        identity = id(pool)
        if identity in self._cached_prefill_stages:
            raise RuntimeError("an LOD layer staged the same continuation twice")
        expected = next(iter(self._cached_prefill_stages.values()), None)
        signature = (
            slots,
            previous_len,
            total_len,
            finalize_cache_for_decode,
        )
        if expected is not None and signature != expected[1:]:
            raise RuntimeError("cross-layer cached-prefix schedules diverged")
        staged_source = pool._stage_cross_layer_cached_cache(
            slots,
            key,
            value,
            previous_len=previous_len,
        )
        if staged_source is not None:
            self._cached_prefill_sources[identity] = staged_source
        self._cached_prefill_stages[identity] = (
            pool,
            slots,
            previous_len,
            total_len,
            finalize_cache_for_decode,
        )
        staged_layers = len(self._cached_prefill_stages)
        if os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1" and self.dcp_rank == 0:
            print(
                "KIMI_CACHED_STAGE "
                f"slot={slots} previous={previous_len} total={total_len} "
                f"layers={staged_layers} built={self._cached_prefill_built_layers}",
                flush=True,
            )
        next_group_end = _cross_layer_prefill_group_end(
            self._cached_prefill_built_layers,
            len(self.pools),
            self.cross_layer_prefill_group_size,
        )
        if staged_layers < next_group_end:
            return
        ordered = tuple(self._cached_prefill_stages.values())[
            self._cached_prefill_built_layers :
        ]
        if not 0 < len(ordered) <= self.cross_layer_prefill_group_size:
            raise AssertionError("invalid cross-layer cached prefill group")
        staged_sources = {
            id(group_pool): self._cached_prefill_sources.pop(id(group_pool))
            for group_pool, *_ in ordered
            if id(group_pool) in self._cached_prefill_sources
        }
        record_tensors = [
            tensor for source in staged_sources.values() for tensor in source
        ]
        staged_source = None
        try:
            self._launch_cross_layer_prefill_group(
                ordered,
                lambda group: self._build_cached_prefill_across_layers(
                    group, staged_sources=staged_sources
                ),
                record_tensors=record_tensors,
                release_sources=staged_sources,
                synchronize_after=(
                    staged_layers == len(self.pools)
                    and os.environ.get("LOD_BENCHMARK_SYNC_PREFILL_CACHE") == "1"
                ),
                reclaim_after=(
                    staged_layers == len(self.pools)
                    and (
                        finalize_cache_for_decode
                        or (
                            _PREFILL_RECLAIM_INTERVAL > 0
                            and total_len % _PREFILL_RECLAIM_INTERVAL == 0
                        )
                    )
                ),
                release_state_workspaces=(
                    staged_layers == len(self.pools)
                    and finalize_cache_for_decode
                ),
            )
        except Exception:
            self._cached_prefill_stages.clear()
            self._cached_prefill_sources.clear()
            self._cached_prefill_built_layers = 0
            raise
        self._cached_prefill_built_layers = staged_layers
        if staged_layers == len(self.pools):
            self._cached_prefill_stages.clear()
            self._cached_prefill_sources.clear()
            self._cached_prefill_built_layers = 0

    def _flush_pending_cached_prefill_layers(self) -> None:
        """Publish a partial final layer group before its cache is consumed."""

        staged_layers = len(self._cached_prefill_stages)
        if staged_layers <= self._cached_prefill_built_layers:
            return
        ordered = tuple(self._cached_prefill_stages.values())[
            self._cached_prefill_built_layers :
        ]
        staged_sources = {
            id(group_pool): self._cached_prefill_sources.pop(id(group_pool))
            for group_pool, *_ in ordered
            if id(group_pool) in self._cached_prefill_sources
        }
        record_tensors = [
            tensor for source in staged_sources.values() for tensor in source
        ]
        try:
            self._launch_cross_layer_prefill_group(
                ordered,
                lambda group: self._build_cached_prefill_across_layers(
                    group, staged_sources=staged_sources
                ),
                record_tensors=record_tensors,
                release_sources=staged_sources,
                reclaim_after=True,
                release_state_workspaces=True,
            )
        finally:
            self._cached_prefill_stages.clear()
            self._cached_prefill_sources.clear()
            self._cached_prefill_built_layers = 0

    def _build_cached_prefill_across_layers(
        self,
        stages: tuple[
            tuple[VLLMLayerLODPool, tuple[int, ...], int, int, bool], ...
        ],
        *,
        staged_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """Run one exact prefill-state update for a bounded layer group."""

        staged_sources = {} if staged_sources is None else staged_sources
        reference, slots, previous_len, total_len, finalize = stages[0]
        expected_staged = (
            reference.settings.kv_bits == 4 or reference.dcp_world_size > 1
        )
        if expected_staged != (len(staged_sources) == len(stages)):
            raise RuntimeError("cross-layer cached staging format diverged")
        if not slots:
            raise AssertionError("cross-layer cached construction requires rows")
        row_count = len(slots)
        slot = slots[0]
        metadata = reference.metadata[slots[0]]
        state_len = int(metadata["state_len"])
        scheduled_state_len = int(metadata.get("scheduled_state_len", state_len))
        old_coverage = int(metadata["coverage"])
        exact_lookback = int(reference.engine.prefill_local_len) - int(
            reference.engine.prefill_chunk_len
        )
        target_coverage = max(
            old_coverage,
            (
                reference.engine._bswa_begin(total_len + 1)
                if finalize
                else total_len - exact_lookback
            ),
        )
        overflow_len = target_coverage - old_coverage
        if overflow_len != total_len - previous_len:
            raise AssertionError("cross-layer cached update is not one aligned block")

        # As in the initial-prefix case, the final scheduler chunk is followed
        # only by rank-local DCP decode.  Rebuilding the global shadow for that
        # chunk and then throwing it away duplicated the most expensive state
        # update.  The retained shadow already contains every earlier exact
        # record; append this chunk while selecting this rank's records and
        # construct the persistent cache once.
        if finalize and reference.dcp_world_size > 1:
            self._build_final_dcp_prefill_group(
                tuple(pool for pool, *_rest in stages),
                slots,
                global_length=total_len,
                previous_length=previous_len,
                staged_sources=staged_sources,
            )
            return

        def cache_state(pool: VLLMLayerLODPool) -> dict[str, object]:
            if pool.dcp_world_size == 1:
                return pool.state
            shadow = (
                pool._batched_dcp_shadow(slots)
                if row_count > 1
                else pool.dcp_prefill_shadows.get(slots[0])
            )
            if shadow is None:
                raise RuntimeError(
                    "cross-layer DCP continuation lost its batched shadow"
                )
            return shadow.state

        def signature(pool: VLLMLayerLODPool) -> tuple[object, ...]:
            engine = pool.engine
            source = cache_state(pool)
            return (
                type(engine),
                pool.kv_heads,
                pool.head_dim,
                int(source["state_capacity"]),
                engine._streaming_state_geometry(),
                engine.state_premerge_factor,
                engine.state_clustering_centroid_rescale,
                engine.state_clustering_centroid_rescale_scope,
                engine.state_merge_before_append,
                engine.fused_state_update,
                engine.fused_state_maxsim,
            )

        expected_signature = signature(reference)
        reference_state = cache_state(reference)
        has_norms = isinstance(
            reference_state.get("key_norm_sums"), torch.Tensor
        )
        # Page-table overflow watermarks are layer-local: centroid ownership
        # can differ while the shared state-update schedule remains identical.
        scalar_names = (
            "state_len",
            "scheduled_state_len",
            "coverage",
            "recent_len",
            "leaf_count",
        )
        if any(
            any(
                int(reference.metadata[other_slot][name]) != int(metadata[name])
                for name in scalar_names
            )
            for other_slot in slots[1:]
        ):
            raise RuntimeError("cross-layer cached request-row geometries diverged")
        for (
            pool,
            other_slots,
            other_previous,
            other_total,
            other_finalize,
        ) in stages[1:]:
            other_metadata = pool.metadata[slots[0]]
            if (
                other_slots != slots
                or other_previous != previous_len
                or other_total != total_len
                or other_finalize != finalize
                or signature(pool) != expected_signature
                or any(
                    int(other_metadata[name]) != int(metadata[name])
                    for name in scalar_names
                )
                or isinstance(cache_state(pool).get("key_norm_sums"), torch.Tensor)
                != has_norms
            ):
                raise RuntimeError("cross-layer cached cache geometries diverged")
            if any(
                any(
                    int(pool.metadata[other_slot][name]) != int(other_metadata[name])
                    for name in scalar_names
                )
                for other_slot in slots[1:]
            ):
                raise RuntimeError(
                    "cross-layer cached request-row geometries diverged"
                )

        sink_len = min(int(reference.engine.sink_len), total_len)
        archive_begin = old_coverage - sink_len
        archive_end = target_coverage - sink_len
        update_ctx_len = min(total_len, target_coverage + reference.engine.local_len)
        group_size = 16
        distributed = _DISTRIBUTED_PREFILL_BUILD
        if distributed and (
            self.dcp_group is None
            or len(stages) != reference.dcp_world_size
            or not all(pool.is_absorbed_mla for pool, *_ in stages)
        ):
            raise RuntimeError(
                "distributed Kimi cached-prefill construction requires one "
                "absorbed-MLA layer per DCP rank"
            )

        reused_layer_parents = 0
        copied_layer_parents = 0

        def reuse_contiguous_layer_views(
            tensors: list[torch.Tensor],
        ) -> torch.Tensor:
            """Recover the prior layer-batched parent when views are adjacent."""

            nonlocal reused_layer_parents, copied_layer_parents

            first = tensors[0]
            if len(tensors) == 1:
                return first
            if int(first.size(0)) != 1:
                copied_layer_parents += 1
                return torch.cat(tensors, dim=0)
            storage = first.untyped_storage()
            storage_ptr = storage.data_ptr()
            offsets = [int(tensor.storage_offset()) for tensor in tensors]
            batch_stride = offsets[1] - offsets[0]
            reusable = batch_stride > 0 and all(
                tuple(tensor.shape) == tuple(first.shape)
                and tensor.dtype == first.dtype
                and tensor.device == first.device
                and tuple(tensor.stride()) == tuple(first.stride())
                and tensor.untyped_storage().data_ptr() == storage_ptr
                and offsets[index] == offsets[0] + index * batch_stride
                for index, tensor in enumerate(tensors)
            )
            if not reusable:
                copied_layer_parents += 1
                return torch.cat(tensors, dim=0)
            reused_layer_parents += 1
            return first.as_strided(
                (len(tensors), *first.shape[1:]),
                (batch_stride, *first.stride()[1:]),
            )

        def pack_state(
            group: tuple[
                tuple[VLLMLayerLODPool, tuple[int, ...], int, int, bool], ...
            ],
            name: str,
        ) -> torch.Tensor:
            tensors = []
            for pool, *_ in group:
                value = cache_state(pool).get(name)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"cross-layer state lacks {name}")
                tensors.append(
                    value
                    if pool.dcp_world_size > 1
                    else value[slots[0] : slots[0] + 1]
                )
            return reuse_contiguous_layer_views(tensors)

        def overflow_source(
            pool: VLLMLayerLODPool, name: str
        ) -> torch.Tensor:
            staged = staged_sources.get(id(pool))
            if staged is not None:
                return staged[0 if name == "leaf_k" else 1][..., :overflow_len, :]
            source = cache_state(pool)
            page = source.get("page_cache")
            if not isinstance(page, dict):
                raise TypeError("cross-layer BF16 archive is missing")
            value = page.get(name)
            if not isinstance(value, torch.Tensor):
                raise TypeError("cross-layer BF16 archive is missing")
            return value[..., archive_begin:archive_end, :]

        updated_state_len: int | None = None
        for start in range(0, len(stages), group_size):
            group = stages[start : start + group_size]
            build_group = (
                group[self.dcp_rank : self.dcp_rank + 1]
                if distributed
                else group
            )
            engine = build_group[0][0].engine
            self._attach_cross_layer_state_workspaces(engine)
            buffers = getattr(engine, "_lod_state_maxsim_buffers", None)
            if isinstance(buffers, dict):
                buffers.pop("_prepared_identity", None)
                buffers.pop("_prepared_context_len", None)
            packed_k = pack_state(build_group, "state_k")
            absorbed_mla = all(pool.is_absorbed_mla for pool, *_ in build_group)
            packed_v = (
                packed_k[..., : reference.value_dim]
                if absorbed_mla
                else pack_state(build_group, "state_v")
            )
            packed_counts = pack_state(build_group, "counts")
            packed_norms = (
                pack_state(build_group, "key_norm_sums") if has_norms else None
            )
            overflow_k = torch.cat(
                [overflow_source(pool, "leaf_k") for pool, *_ in build_group], dim=0
            )
            overflow_v = (
                overflow_k[..., : reference.value_dim]
                if all(pool.is_absorbed_mla for pool, *_ in build_group)
                else torch.cat(
                    [overflow_source(pool, "leaf_v") for pool, *_ in build_group],
                    dim=0,
                )
            )
            (
                packed_k,
                packed_v,
                packed_counts,
                group_state_len,
                owners,
                old_slot_remap,
            ) = engine._update_state(
                packed_k,
                packed_v,
                packed_counts,
                packed_norms,
                overflow_k,
                overflow_v,
                state_len=state_len,
                ctx_len=update_ctx_len,
                available_context=target_coverage,
                state_capacity=int(reference_state["state_capacity"]),
                clustering_query_scale=None,
                scheduled_state_len=scheduled_state_len,
                retain_prepared_geometry=False,
            )
            self._capture_cross_layer_state_workspaces(engine)
            if old_slot_remap is not None:
                raise AssertionError("paged cached state remapping is unsupported")
            if distributed:
                process_group = self.dcp_group
                if process_group is None:
                    raise AssertionError("distributed prefill lost its DCP group")
                active_k = process_group.all_gather(
                    packed_k[..., :group_state_len, :], dim=0
                )
                active_counts = process_group.all_gather(
                    packed_counts[..., :group_state_len, :], dim=0
                )
                owners = process_group.all_gather(
                    owners.to(torch.float32), dim=0
                ).to(torch.long)
                active_norms = (
                    None
                    if packed_norms is None
                    else process_group.all_gather(
                        packed_norms[..., :group_state_len, :], dim=0
                    )
                )
                global_batch = len(group) * row_count
                state_capacity = int(reference_state["state_capacity"])
                packed_k = active_k.new_zeros(
                    global_batch,
                    reference.kv_heads,
                    state_capacity,
                    reference.head_dim,
                )
                packed_k[..., :group_state_len, :].copy_(active_k)
                packed_v = packed_k[..., : reference.value_dim]
                packed_counts = active_counts.new_zeros(
                    global_batch,
                    reference.kv_heads,
                    state_capacity,
                    1,
                )
                packed_counts[..., :group_state_len, :].copy_(active_counts)
                if active_norms is None:
                    packed_norms = None
                else:
                    packed_norms = active_norms.new_zeros(
                        global_batch,
                        reference.kv_heads,
                        state_capacity,
                        1,
                    )
                    packed_norms[..., :group_state_len, :].copy_(active_norms)
            if updated_state_len is None:
                updated_state_len = group_state_len
            elif updated_state_len != group_state_len:
                raise AssertionError("cross-layer cached state schedules diverged")
            owner_ranks = stable_owner_ranks(owners)
            packed_state = {
                "state_k": packed_k,
                "counts": packed_counts,
            }
            if not absorbed_mla:
                packed_state["state_v"] = packed_v
            if packed_norms is not None:
                packed_state["key_norm_sums"] = packed_norms
            for group_row, (pool, *_rest) in enumerate(group):
                row_begin = group_row * row_count
                row_end = row_begin + row_count
                if pool.dcp_world_size > 1:
                    staged = staged_sources.get(id(pool))
                    if staged is None:
                        raise RuntimeError(
                            "DCP cached construction lost its staged source"
                        )
                    pool._finish_dcp_cross_layer_cached_cache(
                        slots,
                        total_len=total_len,
                        coverage=target_coverage,
                        state_k=packed_k[row_begin:row_end],
                        state_v=packed_v[row_begin:row_end],
                        counts=packed_counts[row_begin:row_end],
                        key_norm_sums=(
                            None
                            if packed_norms is None
                            else packed_norms[row_begin:row_end]
                        ),
                        state_len=group_state_len,
                        scheduled_state_len=group_state_len,
                        owners=owners[row_begin:row_end],
                        owner_ranks=owner_ranks[row_begin:row_end],
                        staged_k=staged[0],
                        staged_v=staged[1],
                    )
                else:
                    active = slice(0, group_state_len)
                    for name, packed in packed_state.items():
                        pool.state[name][slot, :, active].copy_(
                            packed[group_row, :, active]
                        )
                    pool._finish_cross_layer_cached_cache(
                        slot,
                        total_len=total_len,
                        coverage=target_coverage,
                        state_len=group_state_len,
                        scheduled_state_len=group_state_len,
                        owners=owners[group_row : group_row + 1],
                        owner_ranks=owner_ranks[group_row : group_row + 1],
                        staged_leaves=staged_sources.get(id(pool)),
                    )
            if finalize and reference.dcp_world_size > 1:
                # As above, hide final global-to-local DCP conversion behind
                # the transformer work for subsequent layer groups.
                self._shard_dcp_prefill_pool_group(
                    tuple(pool for pool, *_rest in group),
                    slots,
                    global_length=total_len,
                )
        if updated_state_len is None:
            raise AssertionError("cross-layer cached construction produced no update")
        if (
            os.environ.get("LOD_KIMI_PROFILE_MEMORY") == "1"
            and self.dcp_rank == 0
        ):
            print(
                "KIMI_CROSS_LAYER_STATE_PACK "
                f"previous={previous_len} total={total_len} "
                f"reused={reused_layer_parents} copied={copied_layer_parents}",
                flush=True,
            )

    def _shard_dcp_prefill_pool_group(
        self,
        pools: tuple[VLLMLayerLODPool, ...],
        rows: tuple[int, ...],
        *,
        global_length: int,
    ) -> None:
        """Convert one completed layer group from global to local DCP rows."""

        if not pools or not rows:
            return
        profile = (
            os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1"
            and self.dcp_rank == 0
        )
        profile_begin = torch.cuda.Event(enable_timing=True) if profile else None
        profile_gather = torch.cuda.Event(enable_timing=True) if profile else None
        profile_build = torch.cuda.Event(enable_timing=True) if profile else None
        profile_install = torch.cuda.Event(enable_timing=True) if profile else None
        if profile_begin is not None:
            profile_begin.record()
        local_keys: list[torch.Tensor] = []
        local_values: list[torch.Tensor] = []
        for pool in pools:
            for row in rows:
                shadow = pool.dcp_prefill_shadows.get(row)
                if shadow is None:
                    raise RuntimeError("cross-layer DCP conversion lost its shadow")
                key, value, observed_length = pool._dcp_local_records(shadow)
                if observed_length != global_length:
                    raise RuntimeError(
                        "cross-layer DCP prompt lengths diverged: "
                        f"reference={global_length}, observed={observed_length}, "
                        f"row={row}"
                    )
                local_keys.append(key)
                local_values.append(value)
        packed_k = torch.cat(local_keys, dim=0)
        if all(pool.is_absorbed_mla for pool in pools):
            packed_v = packed_k[..., : pools[0].value_dim]
        else:
            packed_v = torch.cat(local_values, dim=0)
        if profile_gather is not None:
            profile_gather.record()
        engine = pools[0].engine
        old_capacity = getattr(engine, "_lod_prefill_cache_capacity", None)
        if old_capacity is not None:
            del engine._lod_prefill_cache_capacity
        try:
            with pools[0]._dcp_local_state_schedule():
                global_coverage = pools[0]._dcp_global_decode_coverage(global_length)
                converted = engine.build_cache_from_bf16(
                    packed_k,
                    packed_v,
                    finalize_cache_for_decode=True,
                    final_cache_coverage=pools[0]._dcp_local_length(
                        global_coverage
                    ),
                )
        finally:
            if old_capacity is not None:
                engine._lod_prefill_cache_capacity = old_capacity
        if profile_build is not None:
            profile_build.record()
        row_count = len(rows)
        for layer_index, pool in enumerate(pools):
            for row_index, row in enumerate(rows):
                pool._install_dcp_converted_row(
                    row,
                    converted,
                    global_length=global_length,
                    source_slot=layer_index * row_count + row_index,
                )
        if (
            profile_begin is not None
            and profile_gather is not None
            and profile_build is not None
            and profile_install is not None
        ):
            profile_install.record()
            profile_install.synchronize()
            print(
                "KIMI_DCP_SHARD_GROUP "
                f"layers={len(pools)} rows={len(rows)} "
                f"gather={profile_begin.elapsed_time(profile_gather):.3f}ms "
                f"build={profile_gather.elapsed_time(profile_build):.3f}ms "
                f"install={profile_build.elapsed_time(profile_install):.3f}ms",
                flush=True,
            )

    def _build_final_dcp_prefill_group(
        self,
        pools: tuple[VLLMLayerLODPool, ...],
        rows: tuple[int, ...],
        *,
        global_length: int,
        staged_sources: dict[int, tuple[torch.Tensor, torch.Tensor]],
        previous_length: int = 0,
    ) -> None:
        """Build final rank-local DCP rows without a disposable global update.

        Initial sources contain ``[sink, archive]`` split across the staged
        tensor and the pool's small sink table.  Cached sources contain the
        old exact tail followed by the newly prefetched chunk; the retained
        shadow supplies all earlier chronological records.  Selecting DCP
        ownership before ``build_cache_from_bf16`` is exactly the conversion
        performed by :meth:`_shard_dcp_prefill_pool_group`, but avoids first
        mutating a global cache that no subsequent attention call can read.
        """

        if not pools or not rows:
            return
        reference = pools[0]
        if reference.dcp_world_size <= 1:
            raise RuntimeError("direct final DCP construction requires DCP")
        if any(id(pool) not in staged_sources for pool in pools):
            raise RuntimeError("direct final DCP construction lost staged records")
        if not 0 <= previous_length < global_length:
            raise ValueError("direct final DCP construction has invalid lengths")
        # No subsequent global update consumes the construction scratch.  In
        # particular, INT4 shadows must dequantize this rank's chronological
        # leaves below; release max-sim/update workspaces *before* that large
        # temporary allocation rather than after it.  This ordering supplies
        # the headroom that 512K conversion and its RCCL work buffer require.
        self._release_cross_layer_state_workspaces()
        free_bytes, _total_bytes = torch.cuda.mem_get_info(reference.device)
        if free_bytes < 2 * 1024**3:
            torch.cuda.empty_cache()
        profile_host = (
            os.environ.get("LOD_KIMI_PROFILE_ASYNC_PREFILL_UPDATES") == "1"
            and self.dcp_rank == 0
        )
        phase_begin = time.perf_counter()

        def owned_records(
            pool: VLLMLayerLODPool,
            key: torch.Tensor,
            value: torch.Tensor,
            *,
            global_begin: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            length = int(key.size(2))
            if int(value.size(2)) != length:
                raise ValueError("direct final DCP K/V lengths diverged")
            block = int(pool.dcp_interleave_size)
            world = int(pool.dcp_world_size)
            rank = int(pool.dcp_rank)
            if block == 1:
                # DCP's production geometry assigns every D-th token to a
                # rank.  Boolean advanced indexing first has to discover the
                # output size on the GPU, which serializes the host once per
                # layer at this final conversion boundary.  The equivalent
                # strided view has a statically known shape; the layer-group
                # concatenation below performs the sole required copy.
                first = (rank - global_begin) % world
                selected_k = key[..., first::world, :]
                selected_v_source = value[..., first::world, :]
            else:
                # Non-unit interleave is uncommon, but retain exact support
                # without a data-dependent GPU mask.  Ownership indices are
                # metadata, so constructing this small deterministic list on
                # the host avoids a device synchronization.
                positions = torch.arange(
                    global_begin,
                    global_begin + length,
                    dtype=torch.long,
                )
                owned_positions = torch.nonzero(
                    ((positions // block) % world) == rank,
                    as_tuple=False,
                ).flatten()
                owned_positions = owned_positions.to(
                    device=key.device, non_blocking=True
                )
                selected_k = torch.index_select(key, 2, owned_positions)
                selected_v_source = torch.index_select(value, 2, owned_positions)
            selected_v = (
                selected_k[..., : pool.value_dim]
                if pool.is_absorbed_mla
                else selected_v_source
            )
            return selected_k, selected_v

        def owned_shadow_records(
            pool: VLLMLayerLODPool,
            shadow: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, int]:
            """Return rank-owned BF16 history without a mask or leaf clone.

            K3 DCP uses unit interleave, so ownership is a statically known
            stride through the chronological leaf archive.  The generic
            conversion constructs a device arange, a boolean mask, and an
            advanced-indexing clone for every layer.  At final prefill that
            serialized Python for roughly 50 ms per four-layer group at 64K.
            Preserve the protected sink as its own strided view and let the
            one unavoidable layer-pack concatenate perform the copy.
            """

            source = shadow.state
            page = source.get("page_cache")
            global_history = int(source["total_len"])
            if (
                not pool.is_absorbed_mla
                or pool.dcp_interleave_size != 1
                or not isinstance(page, dict)
                or bool(page.get("quantization_finalized", False))
            ):
                key, value, observed = pool._dcp_local_records(shadow)
                return key, value, observed
            leaf_k = page.get("leaf_k")
            sink_k = source.get("sink_k")
            if not isinstance(leaf_k, torch.Tensor):
                raise TypeError("BF16 DCP shadow has no chronological leaf archive")
            sink_len = int(sink_k.size(2)) if isinstance(sink_k, torch.Tensor) else 0
            rank = int(pool.dcp_rank)
            world = int(pool.dcp_world_size)
            parts: list[torch.Tensor] = []
            if rank < sink_len:
                if not isinstance(sink_k, torch.Tensor):
                    raise RuntimeError("DCP rank lost the protected sink")
                parts.append(sink_k[..., rank:sink_len:world, :])
            first_archive_position = rank
            if first_archive_position < sink_len:
                first_archive_position += (
                    (sink_len - first_archive_position + world - 1) // world
                ) * world
            archive_length = global_history - sink_len
            archive_offset = first_archive_position - sink_len
            if 0 <= archive_offset < archive_length:
                parts.append(leaf_k[..., archive_offset:archive_length:world, :])
            if not parts:
                local_k = leaf_k[..., :0, :]
            elif len(parts) == 1:
                local_k = parts[0]
            else:
                local_k = torch.cat(parts, dim=2)
            expected = pool._dcp_local_length(global_history)
            if int(local_k.size(2)) != expected:
                raise AssertionError(
                    "strided DCP shadow ownership produced the wrong local length"
                )
            return local_k, local_k[..., : pool.value_dim], global_history

        new_length = global_length - previous_length
        local_keys: list[torch.Tensor] = []
        local_values: list[torch.Tensor] = []
        for pool in pools:
            staged_k, staged_v = staged_sources[id(pool)]
            if int(staged_k.size(0)) != len(rows):
                raise ValueError("direct final DCP staged rows diverged")
            for source_row, row in enumerate(rows):
                row_k = staged_k[source_row : source_row + 1]
                row_v = staged_v[source_row : source_row + 1]
                if previous_length == 0:
                    sink = pool.dcp_cross_layer_initial_sinks.pop(row, None)
                    if sink is None:
                        raise RuntimeError(
                            "direct final DCP construction lost its sink"
                        )
                    full_k = torch.cat((sink[0], row_k), dim=2)
                    full_v = (
                        full_k[..., : pool.value_dim]
                        if pool.is_absorbed_mla
                        else torch.cat((sink[1], row_v), dim=2)
                    )
                    if int(full_k.size(2)) != global_length:
                        raise ValueError(
                            "direct final DCP initial archive has the wrong length"
                        )
                    local_k, local_v = owned_records(
                        pool, full_k, full_v, global_begin=0
                    )
                else:
                    shadow = pool.dcp_prefill_shadows.get(row)
                    if shadow is None:
                        raise RuntimeError(
                            "direct final DCP continuation lost its shadow"
                        )
                    old_k, old_v, observed_length = owned_shadow_records(
                        pool, shadow
                    )
                    if observed_length != previous_length:
                        raise RuntimeError(
                            "direct final DCP continuation length diverged: "
                            f"expected={previous_length}, observed={observed_length}"
                        )
                    if int(row_k.size(2)) < new_length or int(row_v.size(2)) < new_length:
                        raise ValueError(
                            "direct final DCP continuation is shorter than its chunk"
                        )
                    new_k, new_v = owned_records(
                        pool,
                        row_k[..., -new_length:, :],
                        row_v[..., -new_length:, :],
                        global_begin=previous_length,
                    )
                    local_k = torch.cat((old_k, new_k), dim=2)
                    local_v = (
                        local_k[..., : pool.value_dim]
                        if pool.is_absorbed_mla
                        else torch.cat((old_v, new_v), dim=2)
                    )
                expected_local = pool._dcp_local_length(global_length)
                if int(local_k.size(2)) != expected_local:
                    raise AssertionError(
                        "direct final DCP ownership produced the wrong local length"
                    )
                local_keys.append(local_k)
                local_values.append(local_v)

        selected_at = time.perf_counter()
        packed_k = torch.cat(local_keys, dim=0)
        packed_v = (
            packed_k[..., : reference.value_dim]
            if all(pool.is_absorbed_mla for pool in pools)
            else torch.cat(local_values, dim=0)
        )
        packed_at = time.perf_counter()
        engine = reference.engine
        old_capacity = getattr(engine, "_lod_prefill_cache_capacity", None)
        if old_capacity is not None:
            del engine._lod_prefill_cache_capacity
        try:
            with reference._dcp_local_state_schedule():
                global_coverage = reference._dcp_global_decode_coverage(
                    global_length
                )
                converted = engine.build_cache_from_bf16(
                    packed_k,
                    packed_v,
                    finalize_cache_for_decode=True,
                    final_cache_coverage=reference._dcp_local_length(
                        global_coverage
                    ),
                )
        finally:
            if old_capacity is not None:
                engine._lod_prefill_cache_capacity = old_capacity
        built_at = time.perf_counter()

        row_count = len(rows)
        for layer_index, pool in enumerate(pools):
            for row_index, row in enumerate(rows):
                pool._install_dcp_converted_row(
                    row,
                    converted,
                    global_length=global_length,
                    source_slot=layer_index * row_count + row_index,
                )
        if profile_host:
            installed_at = time.perf_counter()
            print(
                "KIMI_FINAL_DCP_HOST "
                f"layers={len(pools)} "
                f"select_ms={1_000.0 * (selected_at - phase_begin):.3f} "
                f"pack_ms={1_000.0 * (packed_at - selected_at):.3f} "
                f"build_ms={1_000.0 * (built_at - packed_at):.3f} "
                f"install_ms={1_000.0 * (installed_at - built_at):.3f}",
                flush=True,
            )

    def _shard_dcp_prefill_rows_across_layers(
        self, rows: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Convert replicated prompt shadows with one cache build per layer group."""

        if self.dcp_world_size <= 1 or not rows:
            return ()
        pools = tuple(self.pools.values())
        eligible = tuple(
            row
            for row in rows
            if all(row in pool.dcp_prefill_shadows for pool in pools)
        )
        if not eligible:
            return ()
        profile = (
            os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1"
            and self.dcp_rank == 0
        )
        profile_begin = torch.cuda.Event(enable_timing=True) if profile else None
        if profile_begin is not None:
            profile_begin.record()
        by_length: dict[int, list[int]] = {}
        reference = pools[0]
        for row in eligible:
            shadow = reference.dcp_prefill_shadows[row]
            by_length.setdefault(int(shadow.state["total_len"]), []).append(row)

        group_size = self.cross_layer_prefill_group_size
        for global_length, length_rows in by_length.items():
            row_group = tuple(length_rows)
            for start in range(0, len(pools), group_size):
                layer_group = pools[start : start + group_size]
                self._shard_dcp_prefill_pool_group(
                    layer_group,
                    row_group,
                    global_length=global_length,
                )
        if profile_begin is not None:
            profile_end = torch.cuda.Event(enable_timing=True)
            profile_end.record()
            profile_end.synchronize()
            print(
                "KIMI_DCP_SHARD_CONVERSION "
                f"rows={len(eligible)} layers={len(pools)} "
                f"elapsed={profile_begin.elapsed_time(profile_end):.3f}ms",
                flush=True,
            )
        return eligible

    def _catch_up_decode_rows(self, requests: list[tuple[int, int]]) -> None:
        """Skip layer-by-layer host work between state-update boundaries."""
        if not requests:
            return
        rows = tuple(row for row, _ in requests)
        for pool in self.pools.values():
            pool.wait_deferred_prefill(rows)
        sharded = set(self._shard_dcp_prefill_rows_across_layers(rows))
        remaining_rows = tuple(row for row in rows if row not in sharded)
        if remaining_rows:
            for pool in self.pools.values():
                pool.ensure_dcp_sharded(remaining_rows)
        reference_pool = next(iter(self.pools.values()))
        due = [
            (row, length)
            for row, length in requests
            if int(reference_pool.metadata[row]["coverage"])
            < reference_pool._catch_up_target(row, length)[1]
        ]
        update_due = bool(due)
        # Decode can replay a captured graph without re-entering the Python
        # layer methods.  Order every deferred cache construction here, before
        # that graph consumer is launched; there is no per-layer cache-access
        # hook on graph replay where this dependency can be attached safely.
        # This is paid only at the prefill-to-decode boundary (or before a
        # semantic catch-up), never once per generated token.
        used_cross_layer = False
        if len(due) == 1 and self.dcp_world_size == 1:
            used_cross_layer = self._catch_up_one_across_layers(*due[0])
        if used_cross_layer:
            remaining = [request for request in requests if request != due[0]]
            if remaining:
                for pool in self.pools.values():
                    pool.catch_up_many(remaining)
        elif update_due:
            for pool in self.pools.values():
                pool.catch_up_many(requests)
        for pool in self.pools.values():
            pool.ensure_unified_page1_fixed(rows)
        for row, length in requests:
            local_length = reference_pool._dcp_local_length(length)
            recent_length = local_length - int(
                reference_pool.metadata[row]["coverage"]
            )
            if recent_length > reference_pool.decode_local_limit:
                raise RuntimeError(
                    "LOD catch-up left more live tokens than the decode-local field"
                )
            self.logical_lengths[row] = length

    def _use_native_attention(self, slots: list[int | str]) -> None:
        """Report a missing semantic row; native fallback no longer exists."""
        reference_pool = next(iter(self.pools.values()))
        active = []
        for slot in slots:
            row = self.lod_row_by_slot.get(slot)
            metadata = reference_pool.metadata[row] if row is not None else {}
            active.append(
                (
                    slot,
                    row,
                    bool(row is not None and reference_pool.ready[row]),
                    int(metadata.get("coverage", -1)),
                    int(metadata.get("total_len", -1)),
                )
            )
        retained = [
            (
                entry.row,
                entry.total_length,
                int(reference_pool.metadata[entry.row].get("coverage", -1)),
            )
            for entry in self.cached_rows.values()
        ]
        mapped = sorted(
            ((str(slot), row) for slot, row in self.lod_row_by_slot.items()),
            key=lambda item: item[1],
        )
        raise RuntimeError(
            "external LOD attention has no native remote K/V fallback; the "
            "request needs a matching retained semantic prefix; "
            f"active(slot,row,ready,coverage,total)={active}, "
            f"retained(row,total,coverage)={retained}, "
            f"mapped(slot,row)={mapped}, free_rows={sorted(self.free_lod_rows)}, "
            f"direct_prefill_rejection={self.direct_prefill_rejection}, "
            "restore(attempts,no_row,short,tokens,coverage,last_prefix,"
            "last_coverage,last_total)="
            f"({reference_pool.retained_restore_attempts},"
            f"{reference_pool.retained_restore_fail_no_row},"
            f"{reference_pool.retained_restore_fail_short},"
            f"{reference_pool.retained_restore_fail_tokens},"
            f"{reference_pool.retained_restore_fail_coverage},"
            f"{reference_pool.retained_restore_last_prefix},"
            f"{reference_pool.retained_restore_last_coverage},"
            f"{reference_pool.retained_restore_last_total})"
        )

    def _prepare_direct_prefill(
        self,
        slots: list[int | str],
        computed_lengths: np.ndarray,
        query_starts: np.ndarray,
        prompt_lengths: np.ndarray,
    ) -> bool:
        """Prepare direct LOD only when every authoritative row advances exactly."""
        self.direct_prefill_rejection = None
        if len(slots) > self.pool_size:
            self.direct_prefill_rejection = (
                f"request_rows={len(slots)} exceeds pool_size={self.pool_size}"
            )
            return False
        if len(query_starts) != len(slots) + 1:
            raise ValueError("vLLM query boundaries do not match the request batch")
        if len(prompt_lengths) != len(slots):
            raise ValueError("vLLM prompt lengths do not match the request batch")

        if slots and bool(np.all(computed_lengths == 0)):
            unassigned = sum(slot not in self.lod_row_by_slot for slot in slots)
            missing = max(0, unassigned - len(self.free_lod_rows))
            if missing:
                evicted = self._evict_cached_rows(missing)
                if len(evicted) != missing:
                    self.direct_prefill_rejection = (
                        f"needed {missing} rows but evicted {len(evicted)}"
                    )
                    return False
                self.free_lod_rows.extend(evicted)
                self.free_lod_rows.sort(reverse=True)
            rows = [self._lod_row(slot) for slot in slots]
            first, last = min(rows), max(rows)
            contiguous = (
                len(set(rows)) == len(rows)
                and sorted(rows) == list(range(first, last + 1))
            )
            unused = contiguous and all(
                not pool.has_prefill_cache(row)
                for pool in self.pools.values()
                for row in rows
            )
            if unused:
                # A retained-cache eviction can return the right row set in a
                # different order from packed vLLM requests. The rows have no
                # live contents yet, so remap them before any layer runs.
                for slot, row in zip(slots, sorted(rows), strict=True):
                    self.lod_row_by_slot[slot] = row

        # vLLM can interleave one-token decode rows with newly admitted
        # prefill rows. Pure captured decode can leave Python lengths behind
        # its graph-visible recent cache. Mixed decode keeps those lengths
        # current, but must still perform the normal periodic semantic-state
        # update. Handle both cases before treating the batch as prefill.
        continuations: list[tuple[int, int]] = []
        catch_ups: dict[int, int] = {}
        for request_row, slot in enumerate(slots):
            begin = int(query_starts[request_row])
            end = int(query_starts[request_row + 1])
            if end <= begin:
                continue
            previous_length = int(computed_lengths[request_row])
            if previous_length <= 0:
                continue
            lod_row = self._lod_row(slot)
            has_cache = all(
                pool.has_prefill_cache(lod_row) for pool in self.pools.values()
            )
            if (
                os.environ.get("LOD_KIMI_PROFILE_LIFECYCLE") == "1"
                and self.dcp_rank == 0
            ):
                reference = next(iter(self.pools.values()))
                metadata = reference.metadata[lod_row]
                print(
                    "KIMI_DIRECT_ROW "
                    f"request_row={request_row} slot={slot} lod_row={lod_row} "
                    f"query={end - begin} previous={previous_length} "
                    f"prompt={int(prompt_lengths[request_row])} "
                    f"ready={reference.ready[lod_row]} "
                    f"sharded={reference.dcp_sharded[lod_row]} "
                    f"total={metadata.get('total_len')} "
                    f"global_total={metadata.get('dcp_global_total_len')} "
                    f"coverage={metadata.get('coverage')} "
                    f"device_recent={int(reference.local_lens[lod_row].item())}",
                    flush=True,
                )
            if not has_cache:
                continue
            # A completed DCP prefill can enter its first decode step beside a
            # newly admitted prompt, without ever passing through the pure-
            # decode branch. Convert its replicated shadow before constructing
            # the mixed plan so every layer uses the ordinary batched decode
            # kernels and advances authoritative metadata in the same way.
            one_token_decode = end - begin == 1
            if one_token_decode and not all(
                pool.ready[lod_row] for pool in self.pools.values()
            ):
                catch_ups[lod_row] = previous_length
                continue
            unsharded_prefill_shadow = bool(
                not one_token_decode
                and all(
                    not pool.ready[lod_row]
                    and lod_row in pool.dcp_prefill_shadows
                    for pool in self.pools.values()
                )
            )
            if unsharded_prefill_shadow:
                # DCP prefill keeps a replicated global shadow until the
                # prompt's final scheduler chunk. Its device-local metadata is
                # intentionally not comparable with vLLM's global computed
                # length. A multi-token continuation must keep extending that
                # shadow; decode catch-up is required only for the one-token
                # transition above, where the shadow is first sharded.
                continue
            stale_metadata = any(
                int(
                    pool.metadata[lod_row].get(
                        "dcp_global_total_len",
                        pool.metadata[lod_row].get("total_len", -1),
                    )
                    if pool.dcp_sharded[lod_row]
                    else pool.metadata[lod_row].get("total_len", -1)
                )
                < previous_length
                for pool in self.pools.values()
            )
            if stale_metadata:
                continuations.append((lod_row, previous_length))
            # One-token rows execute through the captured decode kernels even
            # when vLLM classifies the overall step as prefill.  Their host
            # lengths are current, but they still need the same periodic
            # semantic-state update as rows in a pure-decode step.
            if stale_metadata or one_token_decode:
                catch_ups[lod_row] = previous_length
        self._catch_up_decode_rows(list(catch_ups.items()))
        # ``_catch_up_decode_rows`` deliberately avoids touching every
        # layer's Python metadata between semantic update boundaries.  That is
        # important on the steady one-token decode path, but this transition
        # back through direct prefill immediately validates ``total_len`` on
        # every layer.  Refresh those cheap host fields here; the device-local
        # K/V and ``local_lens`` were already advanced by captured decode.
        for lod_row, previous_length in continuations:
            for pool in self.pools.values():
                pool.catch_up(lod_row, previous_length)

        plan: list[tuple[int, int, int, int]] = []
        prepared_prompt_lengths: dict[int, int] = {}
        for request_row, slot in enumerate(slots):
            previous_length = int(computed_lengths[request_row])
            begin = int(query_starts[request_row])
            end = int(query_starts[request_row + 1])
            # vLLM's persistent batch can retain an admitted row that receives
            # no tokens in this scheduler step. It has no attention work and
            # must not force the active rows onto the nonexistent native-cache
            # fallback.
            if end <= begin:
                continue
            if previous_length + end - begin > self.request_capacity:
                self.direct_prefill_rejection = (
                    f"slot={slot} previous={previous_length} scheduled={end - begin} "
                    f"exceeds capacity={self.request_capacity}"
                )
                return False
            lod_row = self._lod_row(slot)
            ready = [
                pool.has_prefill_cache(lod_row) for pool in self.pools.values()
            ]
            if previous_length == 0:
                compatible = not any(ready)
            else:
                total_lengths = [
                    int(
                        pool.metadata[lod_row].get(
                            "dcp_global_total_len",
                            pool.metadata[lod_row].get("total_len", -1),
                        )
                        if pool.dcp_sharded[lod_row]
                        else pool.metadata[lod_row].get("total_len", -1)
                    )
                    for pool in self.pools.values()
                ]
                compatible = all(ready) and all(
                    total_length >= previous_length
                    for total_length in total_lengths
                )
                # Speculative verification writes every proposed K/V into the
                # semantic exact tail.  On the next step vLLM reports only the
                # committed prefix; discard rejected suffix entries before
                # evaluating the new proposal.  The common case is a
                # metadata-only rollback inside the recent exact field.
                if compatible and any(
                    total_length > previous_length
                    for total_length in total_lengths
                ):
                    for pool in self.pools.values():
                        pool.restore_prefix(lod_row, previous_length)
            if not compatible:
                self.direct_prefill_rejection = (
                    f"slot={slot} previous={previous_length} scheduled={end - begin} "
                    f"ready={ready} total_lengths={total_lengths if previous_length else []}"
                )
                return False
            plan.append((lod_row, begin, end, previous_length))
            prepared_prompt_lengths[lod_row] = int(prompt_lengths[request_row])

        prepared = tuple(plan)
        mixed_decode_rows = [
            lod_row
            for lod_row, begin, end, previous_length in prepared
            if previous_length > 0 and end - begin == 1
        ]
        if mixed_decode_rows:
            # A long newly admitted prompt can share one scheduler step with
            # several one-token continuations. They may have different total
            # lengths, but the decode kernels already consume those ragged
            # lengths from the per-row cache metadata. Preserve one batched
            # decode launch instead of turning them into independent B1
            # cached-prefill calls in every layer.
            self._set_active_decode_rows(mixed_decode_rows)
        # Mixed scheduler steps can advance decode rows through this prefill
        # path alongside a long newly admitted request.  Keep the runtime's
        # retained-prefix length current here too; otherwise a request that
        # finishes without another pure-decode step is cached using an older
        # length than the semantic row that was just advanced.
        for lod_row, begin, end, previous_length in prepared:
            self.logical_lengths[lod_row] = previous_length + end - begin
        for pool in self.pools.values():
            pool.decode_enabled = False
            pool.direct_prefill_plan = prepared
            pool.direct_prefill_prompt_lengths = prepared_prompt_lengths.copy()
        return True

    def _require_exact_decode_rows(
        self, lod_rows: list[int], padded_rows: int
    ) -> list[int]:
        if padded_rows != len(lod_rows):
            raise RuntimeError(
                "LoD decode received graph-padded request rows. Exact LoD "
                "decode capture sizes must be installed before GPUModelRunner "
                "initialization; refusing to alias padding lanes onto live "
                "LoD cache rows."
            )
        return lod_rows

    def prepare_capture(
        self, input_batch: Any, kv_cache_config: Any, *, for_capture: bool
    ) -> None:
        self.initialize(kv_cache_config)
        if not for_capture:
            return
        rows = int(input_batch.num_tokens_after_padding)
        self._prepare_dummy_batch(rows, _input_batch_max_query_len(input_batch))

    def preprocess(
        self,
        input_batch: Any,
        _block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: Any,
    ) -> None:
        self.initialize(kv_cache_config)
        if not self.enabled:
            return
        # The static forward context can contain registered attention pools
        # that the active language-model path does not execute.  A fixed
        # layer-batching boundary must therefore flush whatever
        # actually ran at the next scheduler step rather than waiting for an
        # unreachable pool count.  This is also the visibility boundary for
        # the next prefill chunk or the first decode token.
        self._flush_pending_initial_prefill_layers()
        self._flush_pending_cached_prefill_layers()
        max_query_len = _input_batch_max_query_len(input_batch)
        is_prefilling = bool(np.asarray(input_batch.is_prefilling_np).any())
        self._set_speculative_verification_routes(
            self.speculative_tokens > 0
            and not is_prefilling
            and max_query_len > 1
        )
        if input_batch.num_reqs == 0:
            return
        if all(str(req_id).startswith("_warmup_") for req_id in input_batch.req_ids):
            # V2 runs explicit prefill, speculative, and ordinary decode
            # warmups after graph setup. They carry request-shaped metadata but
            # no semantic prompt that should survive into serving-time LOD
            # state. Exercise the appropriate captured/eager branch using
            # dummy rows, just as the legacy runner hook does.
            self._prepare_dummy_batch(
                int(input_batch.num_tokens_after_padding),
                max_query_len,
            )
            return
        if self.hybrid_speculative_full_attention and not is_prefilling:
            # The hybrid control keeps native chronological K/V authoritative
            # for the entire decode phase. No LOD suffix prediction/rollback is
            # needed, and avoiding it keeps this a clean measurement of the
            # native full-attention verifier inside the DFlash2 target graph.
            for pool in self.pools.values():
                pool.decode_enabled = False
                pool.speculative_decode_steps = (
                    max_query_len if max_query_len > 1 else 0
                )
                pool.hybrid_full_decode = True
                pool.direct_prefill_plan = None
            return
        for pool in self.pools.values():
            pool.hybrid_full_decode = False
        rows = int(input_batch.num_reqs)
        for req_id, slot in zip(input_batch.req_ids, input_batch.idx_mapping_np):
            self.req_to_slot[req_id] = int(slot)

        pure_decode = (
            not is_prefilling and max_query_len == 1
        )
        if not pure_decode:
            slots = list(map(int, input_batch.idx_mapping_np))
            query_starts = np.asarray(
                input_batch.query_start_loc_np[: rows + 1], dtype=np.int64
            )
            if (
                not is_prefilling
                and max_query_len > 1
                and self._prepare_speculative_decode(
                    slots,
                    input_batch.num_computed_tokens_np,
                    query_starts,
                    int(input_batch.num_tokens_after_padding),
                    max_query_len,
                )
            ):
                return
            if self._prepare_direct_prefill(
                slots,
                input_batch.num_computed_tokens_np,
                query_starts,
                input_batch.prefill_len_np[:rows],
            ):
                return
            self._use_native_attention(slots)
            return

        lengths = input_batch.num_computed_tokens_np
        for pool in self.pools.values():
            pool.decode_enabled = True
            pool.speculative_decode_steps = 0
            pool.direct_prefill_plan = None
        lod_rows = [self._lod_row(int(slot)) for slot in input_batch.idx_mapping_np]
        padded_rows = int(input_batch.num_tokens_after_padding)
        if padded_rows > self.pool_size:
            raise RuntimeError(
                "a padded pure-decode batch exceeds VLLM_LOD_POOL_SIZE; set "
                "VLLM_LOD_POOL_SIZE and --max-num-seqs to the same value"
            )
        mapped_rows = self._require_exact_decode_rows(lod_rows, padded_rows)
        self._set_active_decode_rows(mapped_rows)
        missing_slots: list[int] = []
        catch_ups: list[tuple[int, int]] = []
        reference_pool = next(iter(self.pools.values()))
        for row, raw_slot in enumerate(input_batch.idx_mapping_np):
            slot = int(raw_slot)
            lod_row = self.lod_row_by_slot[slot]
            length = int(lengths[row])
            if length + 1 > self.request_capacity:
                raise RuntimeError(
                    "decode would exceed VLLM_LOD_MAX_CONTEXT: "
                    f"prefix={length}, append=1, "
                    f"capacity={self.request_capacity}"
                )
            if not reference_pool.ready[lod_row] and not all(
                lod_row in pool.dcp_prefill_shadows
                for pool in self.pools.values()
            ):
                missing_slots.append(slot)
            else:
                catch_ups.append((lod_row, length))
        self._catch_up_decode_rows(catch_ups)
        if missing_slots:
            self._use_native_attention(missing_slots)


def _runtime(model_state: Any) -> VLLMLODRuntime | None:
    # The persistent weight-daemon worker constructs the final model only to
    # retain and export its post-load tensors.  The fresh serving process owns
    # the semantic LoD pools; allocating them here would duplicate the entire
    # B*T cache beside the retained weights.
    if os.getenv("LOD_WEIGHT_CACHE_BACKING", "0") == "1":
        return None
    from .config import lod_enabled

    if not lod_enabled():
        return None
    runtime = getattr(model_state, "_vllm_lod_runtime", None)
    if runtime is not None:
        return runtime
    context = model_state.vllm_config.compilation_config.static_forward_context
    if not any(
        isinstance(getattr(layer, "impl", None), LODAttentionImpl)
        or bool(getattr(layer, "_vllm_lod_absorbed_mla", False))
        for layer in context.values()
    ):
        return None
    runtime = VLLMLODRuntime(model_state)
    model_state._vllm_lod_runtime = runtime
    return runtime


def install_model_state_hooks() -> None:
    """Patch only vLLM's public model-state lifecycle hooks, idempotently."""
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.model_states.interface import ModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    if getattr(ModelState, "_vllm_lod_hooks_installed", False):
        return

    original_add = DefaultModelState.add_request
    original_remove = ModelState.remove_request
    original_default_remove = DefaultModelState.remove_request
    original_init = DefaultModelState.__init__

    def initialize_state(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        _runtime(self)

    def add_request(self: Any, req_index: int, new_req_data: Any) -> None:
        original_add(self, req_index, new_req_data)
        runtime = _runtime(self)
        if runtime is not None:
            runtime.add_request(req_index, new_req_data)

    def remove_request(self: Any, req_id: str) -> None:
        runtime = _runtime(self)
        if runtime is not None:
            runtime.remove_request(req_id)
        original_remove(self, req_id)

    def default_remove_request(self: Any, req_id: str) -> None:
        # vLLM 0.30 overrides ModelState.remove_request on the default state
        # without delegating to the base implementation.  Patching only the
        # base class therefore leaked the LoD row after every completed
        # request: a second generate reused the stale semantic cache instead
        # of resetting it.  Hook the concrete override as well.
        runtime = _runtime(self)
        if runtime is not None:
            runtime.remove_request(req_id)
        original_default_remove(self, req_id)

    DefaultModelState.__init__ = initialize_state
    DefaultModelState.add_request = add_request
    DefaultModelState.remove_request = default_remove_request
    ModelState.remove_request = remove_request

    def patch_state_class(cls: type) -> None:
        original_preprocess = cls.preprocess_state
        original_prepare = cls.prepare_attn

        def preprocess_state(
            self: Any,
            input_batch: Any,
            block_tables: tuple[torch.Tensor, ...],
            kv_cache_config: Any,
            num_computed_tokens: torch.Tensor,
        ) -> None:
            original_preprocess(
                self,
                input_batch,
                block_tables,
                kv_cache_config,
                num_computed_tokens,
            )
            runtime = _runtime(self)
            if runtime is not None:
                runtime.preprocess(input_batch, block_tables, kv_cache_config)

        def prepare_attn(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            input_batch = args[0] if args else kwargs["input_batch"]
            kv_cache_config = args[5] if len(args) > 5 else kwargs["kv_cache_config"]
            for_capture = (
                bool(args[6])
                if len(args) > 6
                else bool(kwargs.get("for_capture", False))
            )
            runtime = _runtime(self)
            if runtime is not None:
                runtime.prepare_capture(
                    input_batch, kv_cache_config, for_capture=for_capture
                )
            return original_prepare(self, *args, **kwargs)

        cls.preprocess_state = preprocess_state
        cls.prepare_attn = prepare_attn

    patch_state_class(DefaultModelState)
    patch_state_class(MambaHybridModelState)
    ModelState._vllm_lod_hooks_installed = True
    install_gpu_runner_hooks()
    install_legacy_runner_hooks()


def install_gpu_runner_hooks() -> None:
    """Expose modern runner token state to persistent LOD cache ownership."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return
    if getattr(GPUModelRunner, "_vllm_lod_token_hooks_installed", False):
        return

    original_init = GPUModelRunner.__init__
    original_load_model = GPUModelRunner.load_model

    def initialize_runner(self: Any, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is not None:
            _ensure_exact_lod_decode_capture_sizes(config)
        original_init(self, *args, **kwargs)

    def load_model(self: Any, *args: Any, **kwargs: Any) -> None:
        original_load_model(self, *args, **kwargs)
        runtime = _runtime(self.model_state)
        if runtime is not None:
            runtime.request_states = self.req_states

    GPUModelRunner.__init__ = initialize_runner
    GPUModelRunner.load_model = load_model
    GPUModelRunner._vllm_lod_token_hooks_installed = True


def install_legacy_runner_hooks() -> None:
    """Hook the persistent-batch runner shipped in released vLLM wheels."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        return
    if getattr(GPUModelRunner, "_vllm_lod_hooks_installed", False):
        return
    if not hasattr(GPUModelRunner, "_update_states"):
        return

    original_init = GPUModelRunner.__init__
    original_load_model = GPUModelRunner.load_model
    original_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    original_build_attention_metadata = GPUModelRunner._build_attention_metadata
    original_request_removed = GPUModelRunner._on_request_state_removed

    def initialize_runner(self: Any, *args: Any, **kwargs: Any) -> None:
        config = kwargs.get("vllm_config", args[0] if args else None)
        if config is not None:
            _ensure_exact_lod_decode_capture_sizes(config)
        original_init(self, *args, **kwargs)

    def load_model(self: Any, *args: Any, **kwargs: Any) -> None:
        original_load_model(self, *args, **kwargs)
        runtime = VLLMLODRuntime(self)
        self._vllm_lod_runtime = runtime

    def initialize_kv_cache(self: Any, *args: Any, **kwargs: Any) -> None:
        original_initialize_kv_cache(self, *args, **kwargs)
        is_profiling = bool(
            kwargs.get("is_profiling", args[1] if len(args) > 1 else False)
        )
        if is_profiling:
            return
        runtime = getattr(self, "_vllm_lod_runtime", None)
        if runtime is not None:
            # The runner deep-copies the scheduler config and then appends
            # worker-only metadata groups (including external LOD layers).
            # Initialize from that final worker view, not the unmodified RPC
            # argument passed into this wrapper.
            runtime.initialize(self.kv_cache_config)

    def build_attention_metadata(
        self: Any, *args: Any, **kwargs: Any
    ) -> tuple[Any, Any]:
        def argument(name: str, position: int, default: Any = None) -> Any:
            if name in kwargs:
                return kwargs[name]
            return args[position] if len(args) > position else default

        runtime = getattr(self, "_vllm_lod_runtime", None)
        if runtime is not None:
            num_reqs = int(argument("num_reqs", 1))
            num_reqs_padded = int(argument("num_reqs_padded", 4, None) or num_reqs)
            runtime.prepare_legacy_runner(
                self,
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                max_query_len=int(argument("max_query_len", 2)),
                for_capture=bool(argument("for_cudagraph_capture", 8, False)),
            )
        return original_build_attention_metadata(self, *args, **kwargs)

    def on_request_state_removed(
        self: Any, req_id: str, req_state: Any | None
    ) -> None:
        runtime = getattr(self, "_vllm_lod_runtime", None)
        if runtime is not None:
            runtime.remove_request(
                req_id,
                token_ids=runtime._legacy_token_ids(req_state),
            )
        original_request_removed(self, req_id, req_state)

    GPUModelRunner.__init__ = initialize_runner
    GPUModelRunner.load_model = load_model
    GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    GPUModelRunner._build_attention_metadata = build_attention_metadata
    GPUModelRunner._on_request_state_removed = on_request_state_removed
    GPUModelRunner._vllm_lod_hooks_installed = True


__all__ = [
    "VLLMLODRuntime",
    "install_gpu_runner_hooks",
    "install_legacy_runner_hooks",
    "install_model_state_hooks",
]
