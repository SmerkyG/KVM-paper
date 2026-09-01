"""vLLM lifecycle hooks for externally owned LOD attention state."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .backend import LODAttentionImpl
from .config import VLLMLODSettings
from .pool import VLLMLayerLODPool

logger = logging.getLogger(__name__)


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
        self.speculative_tokens = int(config.num_speculative_tokens or 0)
        self.hybrid_speculative_full_attention = os.getenv(
            "VLLM_LOD_SPECULATIVE_FULL_ATTENTION", "0"
        ) == "1"
        self.prefix_caching = bool(
            getattr(
                getattr(config, "cache_config", None),
                "enable_prefix_caching",
                False,
            )
        )
        if config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("vLLM LOD does not yet support DCP")
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
        diagnostic_external_empty = os.getenv(
            "VLLM_LOD_DIAGNOSTIC_EXTERNAL_EMPTY_ATTENTION"
        ) in ("skip", "eligible")
        self._eligible_layers: dict[str, Any] = {
            name: layer
            for name, layer in context.items()
            if isinstance(getattr(layer, "impl", None), LODAttentionImpl)
            and bool(layer.impl.lod_eligible)
        }
        self.layers: dict[str, Any] = (
            {} if diagnostic_external_empty else self._eligible_layers
        )
        self.pools: dict[str, VLLMLayerLODPool] = {}
        self.group_by_layer: dict[str, int] = {}
        self.block_size_by_group: dict[int, int] = {}
        self.req_to_slot: dict[str, int | str] = {}
        self.lod_row_by_slot: dict[int | str, int] = {}
        self.cached_rows: dict[int, _CachedLODRow] = {}
        self.request_states: Any | None = None
        self.cache_clock = 0
        self.borrowed_dummy_rows: set[int] = set()
        self.borrowed_dummy_lens: dict[str, torch.Tensor] = {}
        self.free_lod_rows = list(range(self.pool_size - 1, -1, -1))
        self.logical_lengths = [0] * self.pool_size
        self._active_decode_rows: tuple[int, ...] | None = None
        self.initialized = False
        self.allocate_pools()

    @property
    def enabled(self) -> bool:
        return bool(self.layers)

    def _set_speculative_verification_routes(self, enabled: bool) -> None:
        """Use decode-quality routing for a multi-token target verification.

        Ordinary long prefill intentionally opens only three centroids on the
        current fast path.  A speculative target call is logically decode,
        however, and must use the same top-eight approximation as sequential
        one-token decode. Otherwise the verifier itself defines a different
        sparse model and can accept a token that sequential LOD would reject.
        """
        for pool in self.pools.values():
            pool.engine.prefill_two_level_topk = (
                self.settings.open_count
                if enabled
                else (
                    self.settings.prefill_open_count
                    if self.settings.prefill_open_count is not None
                    else min(3, self.settings.open_count)
                )
            )

    def _prefix_rollback_tokens(self) -> int:
        cache_config = getattr(self.model_state.vllm_config, "cache_config", None)
        if not bool(getattr(cache_config, "enable_prefix_caching", False)):
            return 0
        # Keep the established small exact rollback field. Hybrid-cache prefix
        # boundaries can be much older than this (Muse commonly resumes one
        # 4K chunk back), so growing the decode-local field does not solve the
        # general case; restore_prefix() reconstructs those older boundaries
        # from the chronological LoD leaf archive instead.
        return int(self.settings.prefix_rollback_tokens)

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
        for name, layer in self.layers.items():
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
            )
            for rows in sorted(decode_sizes):
                pool.reserve_decode_buffers(rows)
            self.pools[name] = pool
            self.borrowed_dummy_lens[name] = torch.zeros_like(pool.local_lens)
            layer._vllm_lod_pool = pool

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
            self.settings.routing_geometry,
            self.settings.prefill_mode,
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
        self._restore_borrowed_dummy_rows()
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
        mapped_rows = self._pad_decode_rows(lod_rows, num_reqs_padded)
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
        self.borrowed_dummy_rows.discard(row)
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
        if os.getenv("VLLM_LOD_DEBUG_SPECULATIVE_LENGTHS", "0") == "1":
            for lod_row, previous_length in catch_ups:
                for name, pool in self.pools.items():
                    expected = previous_length - int(
                        pool.metadata[lod_row]["coverage"]
                    )
                    actual = int(pool.local_lens[lod_row].item())
                    if actual != expected:
                        raise RuntimeError(
                            "speculative rollback left a stale device-local "
                            f"length for {name}: actual={actual}, "
                            f"expected={expected}, prefix={previous_length}"
                        )
        mapped_rows = self._pad_decode_rows(lod_rows, padded_rows)
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

    def _catch_up_decode_rows(self, requests: list[tuple[int, int]]) -> None:
        """Skip layer-by-layer host work between state-update boundaries."""
        if not requests:
            return
        if self.settings.diagnostic_static_preselected:
            # Timing-only upper bound: the prefill-selected compact tables and
            # their local suffix remain immutable for the entire decode.
            for row, length in requests:
                self.logical_lengths[row] = length
            return
        reference_pool = next(iter(self.pools.values()))
        update_due = any(
            int(reference_pool.metadata[row]["coverage"])
            < reference_pool._catch_up_target(row, length)[1]
            for row, length in requests
        )
        if update_due:
            for pool in self.pools.values():
                pool.catch_up_many(requests)
        for row, length in requests:
            recent_length = length - int(reference_pool.metadata[row]["coverage"])
            if recent_length > int(reference_pool.engine.local_len):
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
        raise RuntimeError(
            "external LOD attention has no native remote K/V fallback; the "
            "request needs a matching retained semantic prefix; "
            f"active(slot,row,ready,coverage,total)={active}, "
            f"retained(row,total,coverage)={retained}, "
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
        if self.settings.prefill_mode != "direct" or len(slots) > self.pool_size:
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
                not pool.ready[row]
                for pool in self.pools.values()
                for row in rows
            )
            if unused:
                # A retained-cache eviction can return the right row set in a
                # different order from packed vLLM requests. The rows have no
                # live contents yet, so remap them before any layer runs.
                for slot, row in zip(slots, sorted(rows), strict=True):
                    self.lod_row_by_slot[slot] = row

        plan: list[tuple[int, int, int, int]] = []
        for request_row, slot in enumerate(slots):
            lod_row = self._lod_row(slot)
            previous_length = int(computed_lengths[request_row])
            begin = int(query_starts[request_row])
            end = int(query_starts[request_row + 1])
            if end <= begin or previous_length + end - begin > self.request_capacity:
                return False
            ready = [pool.ready[lod_row] for pool in self.pools.values()]
            if previous_length == 0:
                compatible = not any(ready)
            else:
                total_lengths = [
                    int(pool.metadata[lod_row].get("total_len", -1))
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
                return False
            plan.append((lod_row, begin, end, previous_length))

        prepared = tuple(plan)
        for pool in self.pools.values():
            pool.decode_enabled = False
            pool.direct_prefill_plan = prepared
            pool.direct_prefill_prompt_lengths = {
                lod_row: int(prompt_lengths[request_row])
                for request_row, (lod_row, _, _, _) in enumerate(prepared)
            }
        return True

    def _pad_decode_rows(self, lod_rows: list[int], padded_rows: int) -> list[int]:
        dummy_count = padded_rows - len(lod_rows)
        active_rows = set(self.lod_row_by_slot.values())
        current_rows = set(lod_rows)
        unused = [
            row
            for row in range(self.pool_size)
            if row not in active_rows and row not in self.cached_rows
        ]
        retained = [
            entry.row
            for entry in sorted(
                self.cached_rows.values(), key=lambda item: item.last_used
            )
            if entry.row not in active_rows
        ]
        dormant = [
            row
            for row in active_rows - current_rows
            if all(pool.ready[row] for pool in self.pools.values())
        ]
        candidates = unused + retained + dormant
        if dummy_count > len(candidates):
            raise RuntimeError(
                "not enough distinct LOD rows for graph padding without "
                "overwriting another authoritative request cache"
            )
        dummy_rows = candidates[:dummy_count]
        if dummy_rows:
            rows = torch.tensor(
                dummy_rows, dtype=torch.long, device=self.active_indices.device
            )
            unused_rows = [row for row in dummy_rows if row in unused]
            unused_tensor = (
                torch.tensor(
                    unused_rows,
                    dtype=torch.long,
                    device=self.active_indices.device,
                )
                if unused_rows
                else None
            )
            for name, pool in self.pools.items():
                if unused_tensor is not None:
                    pool.local_lens.index_fill_(0, unused_tensor, 0)
                self.borrowed_dummy_lens[name].index_copy_(
                    0, rows, pool.local_lens.index_select(0, rows)
                )
        self.borrowed_dummy_rows.update(dummy_rows)
        return lod_rows + dummy_rows

    def _restore_borrowed_dummy_rows(self) -> None:
        """Discard graph-padding appends before a row is observed again."""
        if not self.borrowed_dummy_rows:
            return
        rows = torch.tensor(
            sorted(self.borrowed_dummy_rows),
            dtype=torch.long,
            device=self.active_indices.device,
        )
        for name, pool in self.pools.items():
            pool.local_lens.index_copy_(
                0, rows, self.borrowed_dummy_lens[name].index_select(0, rows)
            )
        self.borrowed_dummy_rows.clear()

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
        max_query_len = _input_batch_max_query_len(input_batch)
        is_prefilling = bool(np.asarray(input_batch.is_prefilling_np).any())
        self._set_speculative_verification_routes(
            self.speculative_tokens > 0
            and not is_prefilling
            and max_query_len > 1
        )
        if input_batch.num_reqs == 0:
            return
        self._restore_borrowed_dummy_rows()
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
        mapped_rows = self._pad_decode_rows(lod_rows, padded_rows)
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
            if not reference_pool.ready[lod_row]:
                missing_slots.append(slot)
            else:
                catch_ups.append((lod_row, length))
        self._catch_up_decode_rows(catch_ups)
        if missing_slots:
            self._use_native_attention(missing_slots)


def _runtime(model_state: Any) -> VLLMLODRuntime | None:
    # The weight-cache backing rank constructs the final model solely to
    # retain/export its parameters.  Its requested config still names the
    # CUSTOM backend so attention modules have the same final structure, but
    # allocating serving-time semantic pools there would pin another complete
    # B*T LOD cache in the daemon.  Fresh workers own those pools instead.
    if os.getenv("VLLM_LOD_WEIGHT_CACHE_BACKING", "0") == "1":
        return None
    runtime = getattr(model_state, "_vllm_lod_runtime", None)
    if runtime is not None:
        return runtime
    context = model_state.vllm_config.compilation_config.static_forward_context
    if not any(
        isinstance(getattr(layer, "impl", None), LODAttentionImpl)
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

    DefaultModelState.__init__ = initialize_state
    DefaultModelState.add_request = add_request
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


def install_tp_safe_vocab_padding() -> None:
    """Keep vLLM's padded vocabulary divisible by unusual TP sizes.

    vLLM pads vocabularies to a fixed multiple of 64, then assumes that
    padded size is divisible by tensor parallelism.  That fails for otherwise
    valid model geometries such as Phi-4 at TP=5 (its ten KV heads require a
    divisor of five, while 100352 is not divisible by five).  Expanding the
    padding multiple to ``lcm(64, TP)`` preserves the original vocabulary and
    weight-loader semantics while making the physical shards regular.
    """
    import math

    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    if getattr(VocabParallelEmbedding, "_vllm_lod_tp_padding_installed", False):
        return
    original_init = VocabParallelEmbedding.__init__

    def initialize_vocab_embedding(
        self: Any,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = 64,
        quant_config: Any = None,
        prefix: str = "",
        *,
        disable_tp: bool = False,
    ) -> None:
        if not disable_tp:
            tp_size = int(get_tensor_model_parallel_world_size())
            padding_size = math.lcm(int(padding_size), tp_size)
        original_init(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
            disable_tp=disable_tp,
        )

    VocabParallelEmbedding.__init__ = initialize_vocab_embedding
    VocabParallelEmbedding._vllm_lod_tp_padding_installed = True


def install_gpu_runner_hooks() -> None:
    """Expose modern runner token state to persistent LOD cache ownership."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        return
    if getattr(GPUModelRunner, "_vllm_lod_token_hooks_installed", False):
        return

    original_load_model = GPUModelRunner.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> None:
        original_load_model(self, *args, **kwargs)
        runtime = _runtime(self.model_state)
        if runtime is not None:
            runtime.request_states = self.req_states

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

    original_load_model = GPUModelRunner.load_model
    original_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    original_build_attention_metadata = GPUModelRunner._build_attention_metadata
    original_request_removed = GPUModelRunner._on_request_state_removed

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
        if runtime is not None and req_state is not None:
            runtime.remove_request(
                req_id,
                token_ids=runtime._legacy_token_ids(req_state),
            )
        original_request_removed(self, req_id, req_state)

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
