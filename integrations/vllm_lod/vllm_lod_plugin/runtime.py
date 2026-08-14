"""vLLM lifecycle hooks and native-paged-cache to LOD conversion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .backend import NATIVE_LAYOUT, LODAttentionImpl
from .config import VLLMLODSettings
from .pool import VLLMLayerLODPool

logger = logging.getLogger(__name__)


class VLLMLODRuntime:
    """Own fixed LOD pools for one vLLM model-runner process."""

    def __init__(self, model_state: Any) -> None:
        self.model_state = model_state
        self.settings = VLLMLODSettings.from_environment()
        config = model_state.vllm_config
        if config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("vLLM LOD does not yet support DCP")
        if config.num_speculative_tokens:
            raise NotImplementedError(
                "vLLM LOD currently requires speculative decoding to be disabled"
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
        self.layers: dict[str, Any] = {
            name: layer
            for name, layer in context.items()
            if isinstance(getattr(layer, "impl", None), LODAttentionImpl)
            and bool(layer.impl.lod_eligible)
        }
        self.pools: dict[str, VLLMLayerLODPool] = {}
        self.group_by_layer: dict[str, int] = {}
        self.block_size_by_group: dict[int, int] = {}
        self.req_to_slot: dict[str, int | str] = {}
        self.lod_row_by_slot: dict[int | str, int] = {}
        self.free_lod_rows = list(range(self.pool_size - 1, -1, -1))
        self.initialized = False
        self.allocate_pools()

    @property
    def enabled(self) -> bool:
        return bool(self.layers)

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
            )
            for rows in sorted(decode_sizes):
                pool.reserve_decode_buffers(rows)
            self.pools[name] = pool
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
        self.allocate_pools()
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                self.group_by_layer[name] = group_id
                layer = self.layers.get(name)
                if layer is not None:
                    cache = getattr(layer, "kv_cache", None)
                    if cache is None:
                        block_size = int(group.kv_cache_spec.block_size)
                    else:
                        if cache.ndim not in (4, 5):
                            raise ValueError("unexpected native vLLM KV cache rank")
                        block_size = int(cache.size(2))
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
            "pool_rows=%d max_context=%d kv_bits=%d routing=%s prefill=%s",
            len(self.pools),
            self.pool_size,
            self.request_capacity,
            self.settings.kv_bits,
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
        pure_decode = max_query_len == 1 and bool(np.all(computed >= prompt_lengths))
        if not pure_decode:
            for req_id in req_ids:
                self.req_to_slot[req_id] = req_id
            query_starts = np.asarray(
                runner.query_start_loc.np[: num_reqs + 1], dtype=np.int64
            )
            if self._prepare_direct_prefill(req_ids, computed, query_starts):
                return
            self._prepare_native_attention(req_ids, computed, query_starts)
            return

        if num_reqs_padded > self.pool_size:
            raise RuntimeError(
                "a padded pure-decode batch exceeds VLLM_LOD_POOL_SIZE; set "
                "VLLM_LOD_POOL_SIZE and --max-num-seqs to the same value"
            )
        for pool in self.pools.values():
            pool.decode_enabled = True
            pool.direct_prefill_plan = None
            pool.native_append_plan = None

        lod_rows = []
        for req_id in req_ids:
            self.req_to_slot[req_id] = req_id
            lod_rows.append(self._lod_row(req_id))
        mapped_rows = self._pad_decode_rows(lod_rows, num_reqs_padded)
        self.active_indices[:num_reqs_padded].copy_(
            torch.tensor(
                mapped_rows,
                dtype=torch.long,
                device=self.active_indices.device,
            )
        )
        if num_reqs_padded > num_reqs:
            dummy_rows = torch.tensor(
                mapped_rows[num_reqs:],
                dtype=torch.long,
                device=self.active_indices.device,
            )
            for pool in self.pools.values():
                pool.local_lens.index_fill_(0, dummy_rows, 0)

        block_tables = tuple(
            input_batch.block_table[group_id].get_device_tensor(num_reqs)
            for group_id in range(len(runner.kv_cache_config.kv_cache_groups))
        )
        conversions: list[tuple[int, int, int]] = []
        catch_ups: list[tuple[int, int]] = []
        for row, req_id in enumerate(req_ids):
            lod_row = self.lod_row_by_slot[req_id]
            length = int(computed[row])
            if not all(pool.ready[lod_row] for pool in self.pools.values()):
                conversions.append((row, lod_row, length))
            else:
                catch_ups.append((lod_row, length))
        for pool in self.pools.values():
            pool.catch_up_many(catch_ups)
        self._convert_requests(conversions, block_tables)

    def add_request(self, slot: int, data: Any) -> None:
        self.req_to_slot[data.req_id] = slot
        self._release_lod_row(slot)

    def remove_request(self, req_id: str) -> None:
        slot = self.req_to_slot.pop(req_id, None)
        if slot is not None:
            self._release_lod_row(slot)

    def _release_lod_row(self, slot: int | str) -> None:
        row = self.lod_row_by_slot.pop(slot, None)
        if row is None:
            return
        if self.initialized:
            for pool in self.pools.values():
                pool.reset(row)
        self.free_lod_rows.append(row)

    def _lod_row(self, slot: int | str) -> int:
        row = self.lod_row_by_slot.get(slot)
        if row is not None:
            return row
        if not self.free_lod_rows:
            raise RuntimeError(
                "pure decode batch exceeds VLLM_LOD_POOL_SIZE; increase the "
                "environment setting or reduce --max-num-seqs"
            )
        row = self.free_lod_rows.pop()
        self.lod_row_by_slot[slot] = row
        return row

    def _prepare_dummy_batch(self, rows: int, max_query_len: int) -> None:
        decode_capture = max_query_len == 1
        for pool in self.pools.values():
            pool.decode_enabled = decode_capture
            pool.direct_prefill_plan = None
            pool.native_append_plan = None
        if not decode_capture:
            return
        if rows > self.pool_size:
            raise RuntimeError(
                "a captured pure-decode batch exceeds VLLM_LOD_POOL_SIZE; "
                "set VLLM_LOD_POOL_SIZE and --max-num-seqs to the same value"
            )
        self.active_indices[:rows].copy_(
            torch.arange(rows, device=self.active_indices.device)
        )
        for pool in self.pools.values():
            pool.local_lens.zero_()

    def _use_native_attention(self, slots: list[int | str]) -> None:
        """Disable LOD for this batch and mark affected shadow rows stale."""
        for pool in self.pools.values():
            pool.decode_enabled = False
            pool.direct_prefill_plan = None
            pool.native_append_plan = None
        for slot in slots:
            lod_row = self.lod_row_by_slot.get(slot)
            if lod_row is None:
                continue
            for pool in self.pools.values():
                pool.ready[lod_row] = False

    def _prepare_native_attention(
        self,
        slots: list[int | str],
        computed_lengths: np.ndarray,
        query_starts: np.ndarray,
    ) -> None:
        """Use native attention while preserving exact one-token LOD shadows."""
        if len(query_starts) != len(slots) + 1:
            raise ValueError("vLLM query boundaries do not match the request batch")
        plan: list[tuple[int, int, int, int]] = []
        stale: list[int | str] = []
        for request_row, slot in enumerate(slots):
            lod_row = self.lod_row_by_slot.get(slot)
            begin = int(query_starts[request_row])
            end = int(query_starts[request_row + 1])
            previous_length = int(computed_lengths[request_row])
            compatible = (
                lod_row is not None
                and end - begin == 1
                and all(pool.ready[lod_row] for pool in self.pools.values())
                and all(
                    int(pool.metadata[lod_row].get("total_len", -1))
                    == previous_length
                    for pool in self.pools.values()
                )
            )
            if compatible:
                plan.append((lod_row, begin, end, previous_length))
            else:
                stale.append(slot)

        catch_ups = [(lod_row, length) for lod_row, _, _, length in plan]
        for pool in self.pools.values():
            pool.catch_up_many(catch_ups)
        self._use_native_attention(stale)
        prepared = tuple(plan)
        for pool in self.pools.values():
            pool.decode_enabled = False
            pool.direct_prefill_plan = None
            pool.native_append_plan = prepared

    def _prepare_direct_prefill(
        self,
        slots: list[int | str],
        computed_lengths: np.ndarray,
        query_starts: np.ndarray,
    ) -> bool:
        """Prepare a direct LOD batch only when every shadow can advance exactly."""
        if self.settings.prefill_mode != "direct" or len(slots) > self.pool_size:
            return False
        if len(query_starts) != len(slots) + 1:
            raise ValueError("vLLM query boundaries do not match the request batch")

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
                compatible = all(ready) and all(
                    int(pool.metadata[lod_row].get("total_len", -1))
                    == previous_length
                    for pool in self.pools.values()
                )
            if not compatible:
                # This includes native prefix-cache hits for which no LOD shadow
                # exists. Native attention remains authoritative and the prefix
                # is converted before the next pure-decode batch.
                return False
            plan.append((lod_row, begin, end, previous_length))

        prepared = tuple(plan)
        for pool in self.pools.values():
            pool.decode_enabled = False
            pool.direct_prefill_plan = prepared
            pool.native_append_plan = None
        return True

    def _pad_decode_rows(self, lod_rows: list[int], padded_rows: int) -> list[int]:
        dummy_count = padded_rows - len(lod_rows)
        candidates = [row for row in range(self.pool_size) if row not in lod_rows]
        if dummy_count > len(candidates):
            raise RuntimeError("not enough distinct LOD rows for padded decode")
        dummy_rows = candidates[:dummy_count]
        owned_rows = set(self.lod_row_by_slot.values())
        for row in dummy_rows:
            if row not in owned_rows:
                continue
            # A graph-padding row may temporarily borrow storage belonging to
            # a currently unscheduled request. Mark it stale so that request
            # is rebuilt from the authoritative native cache before reuse.
            for pool in self.pools.values():
                pool.ready[row] = False
        return lod_rows + dummy_rows

    def prepare_capture(
        self, input_batch: Any, kv_cache_config: Any, *, for_capture: bool
    ) -> None:
        self.initialize(kv_cache_config)
        if not for_capture:
            return
        rows = int(input_batch.num_tokens_after_padding)
        self._prepare_dummy_batch(rows, int(input_batch.max_query_len or 1))

    def _gather_native_prefix(
        self,
        layer: Any,
        block_table: torch.Tensor,
        *,
        length: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self._gather_native_prefix_batch(
            layer,
            block_table.unsqueeze(0),
            length=length,
            block_size=block_size,
        )
        return key, value

    def _gather_native_prefix_batch(
        self,
        layer: Any,
        block_tables: torch.Tensor,
        *,
        length: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if length <= 0:
            raise ValueError("cannot convert an empty native KV prefix")
        if block_tables.ndim != 2:
            raise ValueError("native block tables must have one row per request")
        batch = int(block_tables.size(0))
        blocks = math_ceil_div(length, block_size)
        block_ids = block_tables[:, :blocks].long()
        if bool((block_ids < 0).any().item()):
            raise RuntimeError("native vLLM block table is incomplete for conversion")
        flat_ids = block_ids.flatten()
        cache = layer.kv_cache
        if NATIVE_LAYOUT == "flash":
            pages = cache.index_select(0, flat_ids)
            if pages.ndim != 4 or int(pages.size(-1)) != 2 * int(layer.head_size):
                raise ValueError("unexpected FlashAttention KV cache layout")
            key, value = pages.split(int(layer.head_size), dim=-1)
            key = key.view(batch, blocks, *key.shape[1:]).permute(0, 2, 1, 3, 4)
            value = value.view(batch, blocks, *value.shape[1:]).permute(
                0, 2, 1, 3, 4
            )
            key = key.reshape(
                batch, int(layer.num_kv_heads), -1, int(layer.head_size)
            )
            value = value.reshape(
                batch, int(layer.num_kv_heads), -1, int(layer.head_size_v)
            )
        else:
            if cache.ndim != 5 or int(cache.size(0)) != 2:
                raise ValueError("unexpected ROCm attention KV cache layout")
            # The raw allocation's nominal shape is
            # [2, blocks, block_size, heads, dim], but ROCm paged attention
            # writes it through head-major K/V views.  Reusing vLLM's
            # canonical split is essential: indexing the raw allocation as a
            # token-major tensor silently scrambles both caches.
            from vllm.v1.attention.ops.paged_attn import PagedAttention

            key_cache, value_cache = PagedAttention.split_kv_cache(
                cache, int(layer.num_kv_heads), int(layer.head_size)
            )
            key_pages = key_cache.index_select(0, flat_ids)
            value_pages = value_cache.index_select(0, flat_ids)
            key = key_pages.view(batch, blocks, *key_pages.shape[1:]).permute(
                0, 2, 1, 4, 3, 5
            )
            value = value_pages.view(batch, blocks, *value_pages.shape[1:]).permute(
                0, 2, 1, 4, 3
            )
            key = key.reshape(
                batch, int(layer.num_kv_heads), -1, int(layer.head_size)
            )
            value = value.reshape(
                batch, int(layer.num_kv_heads), -1, int(layer.head_size_v)
            )
        return key[..., :length, :].contiguous(), value[..., :length, :].contiguous()

    def _convert_requests(
        self,
        requests: list[tuple[int, int, int]],
        block_tables: tuple[torch.Tensor, ...],
    ) -> None:
        """Convert equal-length native prefixes together, retaining the source KV."""
        if not requests:
            return
        by_length: dict[int, list[tuple[int, int]]] = {}
        for row, lod_row, length in requests:
            if length > self.request_capacity:
                raise ValueError(
                    f"request length {length} exceeds LOD capacity "
                    f"{self.request_capacity}; raise VLLM_LOD_MAX_CONTEXT"
                )
            by_length.setdefault(length, []).append((row, lod_row))

        try:
            for length, rows in by_length.items():
                for name, layer in self.layers.items():
                    group_id = self.group_by_layer[name]
                    source_rows = torch.tensor(
                        [row for row, _ in rows],
                        dtype=torch.long,
                        device=block_tables[group_id].device,
                    )
                    source_tables = block_tables[group_id].index_select(
                        0, source_rows
                    )
                    key, value = self._gather_native_prefix_batch(
                        layer,
                        source_tables,
                        length=length,
                        block_size=self.block_size_by_group[group_id],
                    )
                    converted = self.pools[name].engine.build_cache_from_bf16(
                        key, value
                    )
                    for source_slot, (_, lod_row) in enumerate(rows):
                        self.pools[name].install(
                            lod_row, converted, source_slot=source_slot
                        )
        except Exception:
            for _, lod_row, _ in requests:
                for pool in self.pools.values():
                    pool.reset(lod_row)
            raise

    def preprocess(
        self,
        input_batch: Any,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: Any,
    ) -> None:
        self.initialize(kv_cache_config)
        if not self.enabled or input_batch.num_reqs == 0:
            return
        rows = int(input_batch.num_reqs)
        for req_id, slot in zip(input_batch.req_ids, input_batch.idx_mapping_np):
            self.req_to_slot[req_id] = int(slot)

        pure_decode = (
            not bool(np.asarray(input_batch.is_prefilling_np).any())
            and int(input_batch.max_query_len or 1) == 1
        )
        if not pure_decode:
            slots = list(map(int, input_batch.idx_mapping_np))
            query_starts = np.asarray(
                input_batch.query_start_loc_np[: rows + 1], dtype=np.int64
            )
            if self._prepare_direct_prefill(
                slots, input_batch.num_computed_tokens_np, query_starts
            ):
                return
            # Native attention updates the authoritative BF16 cache for any
            # batch that cannot advance every LOD shadow exactly. Reconvert
            # before these requests next enter a pure LOD decode graph.
            self._prepare_native_attention(
                slots, input_batch.num_computed_tokens_np, query_starts
            )
            return

        lengths = input_batch.num_computed_tokens_np
        for pool in self.pools.values():
            pool.decode_enabled = True
            pool.direct_prefill_plan = None
            pool.native_append_plan = None
        lod_rows = [self._lod_row(int(slot)) for slot in input_batch.idx_mapping_np]
        padded_rows = int(input_batch.num_tokens_after_padding)
        if padded_rows > self.pool_size:
            raise RuntimeError(
                "a padded pure-decode batch exceeds VLLM_LOD_POOL_SIZE; set "
                "VLLM_LOD_POOL_SIZE and --max-num-seqs to the same value"
            )
        mapped_rows = self._pad_decode_rows(lod_rows, padded_rows)
        self.active_indices[:padded_rows].copy_(
            torch.tensor(
                mapped_rows,
                dtype=torch.long,
                device=self.active_indices.device,
            )
        )
        if padded_rows > rows:
            dummy_rows = torch.tensor(
                mapped_rows[rows:],
                dtype=torch.long,
                device=self.active_indices.device,
            )
            # Padding executes inside a captured graph. Reset its distinct
            # scratch rows before replay so fake appends never accumulate.
            for pool in self.pools.values():
                pool.local_lens.index_fill_(0, dummy_rows, 0)
        conversions: list[tuple[int, int, int]] = []
        catch_ups: list[tuple[int, int]] = []
        for row, raw_slot in enumerate(input_batch.idx_mapping_np):
            slot = int(raw_slot)
            lod_row = self.lod_row_by_slot[slot]
            length = int(lengths[row])
            if not all(pool.ready[lod_row] for pool in self.pools.values()):
                conversions.append((row, lod_row, length))
            else:
                catch_ups.append((lod_row, length))
        for pool in self.pools.values():
            pool.catch_up_many(catch_ups)
        self._convert_requests(conversions, block_tables)


def math_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _runtime(model_state: Any) -> VLLMLODRuntime | None:
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
    install_legacy_runner_hooks()


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
        kv_cache_config = args[0] if args else kwargs["kv_cache_config"]
        runtime = getattr(self, "_vllm_lod_runtime", None)
        if runtime is not None:
            runtime.initialize(kv_cache_config)

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

    GPUModelRunner.load_model = load_model
    GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    GPUModelRunner._build_attention_metadata = build_attention_metadata
    GPUModelRunner._vllm_lod_hooks_installed = True


__all__ = [
    "VLLMLODRuntime",
    "install_legacy_runner_hooks",
    "install_model_state_hooks",
]
