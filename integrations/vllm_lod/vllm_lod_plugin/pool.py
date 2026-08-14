"""Fixed-address per-layer LOD pools used by captured vLLM decode graphs."""

from __future__ import annotations

import math
from typing import Any

import torch

from model.kernels.paged_leaf_attention import (
    advance_decode_cache_lengths,
    fused_decode_paged_lod_attention,
    new_fused_decode_buffers,
    rehash_overflow_pages,
)
from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import (
    KernelLODCache,
    KernelRecursivePagedLODAttention,
)

from .config import VLLMLODSettings


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _power_of_two(value: int) -> int:
    return 1 << max(1, (value - 1).bit_length())


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
    ) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("vLLM LOD conversion requires a native FP16/BF16 KV cache")
        if request_capacity < settings.local_window:
            raise ValueError("LOD request capacity is shorter than its local window")
        self.layer = layer
        self.settings = settings
        self.max_requests = max_requests
        self.request_capacity = request_capacity
        self.active_indices = active_indices
        self.dtype = dtype
        self.device = device
        self.query_heads = int(layer.num_heads)
        self.kv_heads = int(layer.num_kv_heads)
        self.head_dim = int(layer.head_size)
        self.value_dim = int(layer.head_size_v)
        if self.value_dim != self.head_dim:
            raise NotImplementedError(
                "LOD vLLM currently requires equal K and V widths"
            )

        geometry = settings.routing_geometry
        if geometry == "auto":
            geometry = "coherence" if has_key_norm else "spherical"
        state_normalization = "cosine" if geometry == "spherical" else "none"
        centroid_rescale = "coherence" if geometry == "coherence" else "none"
        routing_normalization = (
            "none"
            if geometry == "raw" or has_query_norm
            else "query"
        )
        local_window = settings.local_window
        if settings.cache_ownership == "lod":
            # A completed cache may be reused at vLLM's preceding physical
            # block boundary. Keep at least one staging chunk exact so that
            # rolling a retained LOD cache back to that boundary only adjusts
            # its recent-tail length; clustered state never has to be undone.
            local_window = max(local_window, settings.native_staging_chunk)
        config = PagedLODConfig(
            chunk_size=settings.chunk_size,
            local_window=local_window,
            state_growth_factor=settings.state_growth_factor,
            state_min_size=settings.state_min_size,
            protected_prefix=settings.protected_prefix,
            max_routes=max(settings.open_count, 8),
            leaf_dtype=self.dtype,
            page_size=16,
            kv_bits=settings.kv_bits,
            quant_group_size=settings.quant_group_size,
            state_clustering_normalization=state_normalization,
            state_clustering_centroid_rescale=centroid_rescale,
            state_clustering_centroid_rescale_scope="assignment",
            routing_normalization=routing_normalization,
        )
        self.engine = KernelRecursivePagedLODAttention(
            config,
            query_heads=self.query_heads,
            key_value_heads=self.kv_heads,
            scale=float(layer.impl.scale),
            default_open_count=settings.open_count,
        )
        # Keep exact sink/protected entries outside the clustered state. This
        # matches the standalone kernel architecture and folds the side cache
        # into the existing final decode reduction.
        self.engine.separate_sink_cache = True
        self.state_capacity = self.engine._state_capacity(
            request_capacity, min(request_capacity, settings.chunk_size)
        )
        self.local_capacity = local_window + int(
            self.engine.decode_state_update_len
        )
        self.leaf_capacity = _round_up(request_capacity, settings.chunk_size) + max(
            settings.chunk_size, int(self.engine.decode_cache_headroom)
        )
        self.page_capacity = math.ceil(self.leaf_capacity / 16) + self.state_capacity
        self.hash_capacity = _power_of_two(
            self.page_capacity * int(self.engine.leaf_overflow_hash_factor)
        )
        self.state = self._allocate_state()
        self.local_lens = torch.zeros(
            max_requests, dtype=torch.int32, device=self.device
        )
        self.ready = [False] * max_requests
        self.metadata = [dict[str, int | bool]() for _ in range(max_requests)]
        self.decode_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self.decode_enabled = False
        self.direct_prefill_plan: tuple[tuple[int, int, int, int], ...] | None = None
        self.native_append_plan: tuple[tuple[int, int, int, int], ...] | None = None
        self.install_count = 0
        self.direct_prefill_calls = 0
        self.batched_cached_prefill_calls = 0
        self.batched_cached_prefill_rows = 0
        self.cached_prefill_candidate_calls = 0
        self.cached_prefill_candidate_rows = 0
        self.cached_prefill_nonuniform_lengths = 0
        self.cached_prefill_nonuniform_previous = 0
        self.cached_prefill_unready = 0
        self.cached_prefill_noncontiguous = 0
        self.native_append_calls = 0
        self.decode_calls = 0
        self.catch_up_batches = 0
        self.catch_up_rows = 0
        self.retained_reuse_count = 0

    def _allocate_state(self) -> dict[str, object]:
        r, h, s, d = (
            self.max_requests,
            self.kv_heads,
            self.state_capacity,
            self.head_dim,
        )
        state: dict[str, object] = {
            "state_k": torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device),
            "state_v": torch.zeros(r, h, s, d, dtype=self.dtype, device=self.device),
            "counts": torch.zeros(r, h, s, 1, dtype=torch.float32, device=self.device),
            "state_len": s,
            "coverage": 0,
            "state_capacity": s,
            "recent_k": torch.empty(
                r, h, self.local_capacity, d, dtype=self.dtype, device=self.device
            ),
            "recent_v": torch.empty(
                r, h, self.local_capacity, d, dtype=self.dtype, device=self.device
            ),
            "recent_len": 0,
            "total_len": 0,
        }
        if self.settings.protected_prefix:
            state["sink_k"] = torch.empty(
                r,
                h,
                self.settings.protected_prefix,
                d,
                dtype=self.dtype,
                device=self.device,
            )
            state["sink_v"] = torch.empty_like(state["sink_k"])
        if self.engine.state_clustering_centroid_rescale != "none":
            state["key_norm_sums"] = torch.zeros(
                r, h, s, 1, dtype=torch.float32, device=self.device
            )

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
        if self.settings.kv_bits == 4:
            page.update(
                leaf_k=torch.empty(r, h, 1, d, dtype=self.dtype, device=self.device),
                leaf_v=torch.empty(r, h, 1, d, dtype=self.dtype, device=self.device),
                quantized_leaf_k=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    d // 2,
                    dtype=torch.uint8,
                    device=self.device,
                ),
                quantized_leaf_v=torch.empty(
                    r,
                    h,
                    self.leaf_capacity,
                    d // 2,
                    dtype=torch.uint8,
                    device=self.device,
                ),
                page_k_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    groups,
                    dtype=self.dtype,
                    device=self.device,
                ),
                page_v_scales=torch.empty(
                    r,
                    h,
                    self.page_capacity,
                    groups,
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
        self.ready[slot] = False
        self.metadata[slot].clear()
        self.local_lens[slot].zero_()
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
        page["page_indices"][slot].fill_(-1)
        page["page_counts"][slot].zero_()
        page["overflow_page_keys"][slot].fill_(-1)
        page["overflow_page_values"][slot].fill_(-1)
        if "page_quantized_counts" in page:
            page["page_quantized_counts"][slot].zero_()

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

    def install(
        self, slot: int, converted: KernelLODCache, *, source_slot: int = 0
    ) -> None:
        source = converted.state
        source_page = source.get("page_cache")
        if not isinstance(source_page, dict):
            raise TypeError("converted LOD cache has no semantic page archive")
        if int(source["total_len"]) > self.request_capacity:
            raise ValueError("converted prefix exceeds VLLM_LOD_MAX_CONTEXT")
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
            if name in ("overflow_page_keys", "overflow_page_values"):
                continue
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                self._copy_row(
                    destination, value, slot, source_slot=source_slot
                )
        source_keys = source_page["overflow_page_keys"]
        source_values = source_page["overflow_page_values"]
        destination_keys = destination_page["overflow_page_keys"]
        destination_values = destination_page["overflow_page_values"]
        if int(source_keys.size(2)) == int(destination_keys.size(2)):
            self._copy_row(
                destination_keys, source_keys, slot, source_slot=source_slot
            )
            self._copy_row(
                destination_values, source_values, slot, source_slot=source_slot
            )
            destination_page["overflow_used"].logical_or_(
                source_page["overflow_used"]
            )
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
        destination_page["overflow_flag"].logical_or_(
            source_page["overflow_flag"]
        )
        recent_len = int(source["recent_len"])
        self.local_lens[slot].fill_(recent_len)
        self.metadata[slot].update(
            state_len=int(source["state_len"]),
            coverage=int(source["coverage"]),
            total_len=int(source["total_len"]),
            recent_len=recent_len,
            leaf_count=int(source_page["leaf_count"]),
            overflow_safe_until=int(source_page["overflow_safe_until"]),
        )
        self.ready[slot] = True
        self.install_count += 1

    def _synchronize_row(self, slot: int, cache: KernelLODCache) -> None:
        """Persist metadata and any reallocated tensors after cached prefill."""
        self._synchronize_rows((slot,), cache)

    def _synchronize_rows(
        self, slots: tuple[int, ...], cache: KernelLODCache
    ) -> None:
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
            destination = destination_page.get(name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(destination, torch.Tensor)
                and value.ndim
            ):
                for source_slot, slot in enumerate(slots):
                    self._copy_row(
                        destination, value, slot, source_slot=source_slot
                    )
        recent_len = int(source["recent_len"])
        for slot in slots:
            self.local_lens[slot].fill_(recent_len)
            self.metadata[slot].update(
                state_len=int(source["state_len"]),
                coverage=int(source["coverage"]),
                total_len=int(source["total_len"]),
                recent_len=recent_len,
                leaf_count=int(source_page["leaf_count"]),
                overflow_safe_until=int(source_page["overflow_safe_until"]),
            )
            self.ready[slot] = True

    def _direct_cached_prefill_group(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        plan: tuple[tuple[int, int, int, int], ...],
    ) -> None:
        """Advance one contiguous equal-length/equal-history cache group."""
        length = plan[0][2] - plan[0][1]
        previous_length = plan[0][3]
        slots = tuple(slot for slot, _, _, _ in plan)
        packed = int(query.size(0)) == len(plan) * length and all(
            begin == source_slot * length
            and end == (source_slot + 1) * length
            for source_slot, (_, begin, end, _) in enumerate(plan)
        )
        q = (
            query.reshape(len(plan), length, *query.shape[1:]).permute(0, 2, 1, 3)
            if packed
            else torch.stack(
                [query[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
        )
        k = torch.stack(
            [key[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
        )
        v = torch.stack(
            [value[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
        )
        cache = self._range_cache(slots[0], slots[-1] + 1)
        if cache.total_length != previous_length:
            raise RuntimeError(
                "batched cached LOD prefill length differs from its prepared plan"
            )
        result, cache = self.engine(q, k, v, cache=cache, use_cache=True)
        if cache is None:
            raise AssertionError("batched cached LOD prefill did not return a cache")
        if len(slots) > 1:
            self.batched_cached_prefill_calls += 1
            self.batched_cached_prefill_rows += len(slots)
        self._synchronize_rows(slots, cache)
        self.engine.reset_runtime_cache()
        if packed:
            output.reshape(len(plan), length, *output.shape[1:]).copy_(
                result.permute(0, 2, 1, 3)
            )
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
        if plan is None:
            raise RuntimeError("direct LOD prefill has no prepared request plan")
        self.direct_prefill_calls += 1
        lengths = {end - begin for _, begin, end, _ in plan}
        if (
            plan
            and len(lengths) == 1
            and all(previous_length == 0 for _, _, _, previous_length in plan)
            and all(not self.ready[slot] for slot, _, _, _ in plan)
        ):
            length = next(iter(lengths))
            packed = all(
                begin == source_slot * length
                and end == (source_slot + 1) * length
                for source_slot, (_, begin, end, _) in enumerate(plan)
            )
            q = (
                query.reshape(len(plan), length, *query.shape[1:]).permute(0, 2, 1, 3)
                if packed
                else torch.stack(
                    [
                        query[begin:end].permute(1, 0, 2)
                        for _, begin, end, _ in plan
                    ]
                )
            )
            k = torch.stack(
                [key[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
            v = torch.stack(
                [value[begin:end].permute(1, 0, 2) for _, begin, end, _ in plan]
            )
            result, cache = self.engine(q, k, v, use_cache=True)
            if cache is None:
                raise AssertionError("direct LOD prefill did not return a cache")
            for source_slot, (slot, begin, end, _) in enumerate(plan):
                self.install(slot, cache, source_slot=source_slot)
            self.engine.reset_runtime_cache()
            if packed:
                output.reshape(len(plan), length, *output.shape[1:]).copy_(
                    result.permute(0, 2, 1, 3)
                )
            else:
                for source_slot, (_, begin, end, _) in enumerate(plan):
                    output[begin:end].copy_(result[source_slot].permute(1, 0, 2))
            return output
        previous_lengths = {previous_length for _, _, _, previous_length in plan}
        slots = tuple(slot for slot, _, _, _ in plan)
        ordered_plan = tuple(sorted(plan, key=lambda item: item[0]))
        ordered_slots = tuple(slot for slot, _, _, _ in ordered_plan)
        positive_previous = bool(plan) and all(
            previous_length > 0 for _, _, _, previous_length in plan
        )
        contiguous_slots = bool(ordered_slots) and ordered_slots == tuple(
            range(ordered_slots[0], ordered_slots[0] + len(ordered_slots))
        )
        if positive_previous:
            self.cached_prefill_candidate_calls += 1
            self.cached_prefill_candidate_rows += len(plan)
            self.cached_prefill_nonuniform_lengths += int(len(lengths) != 1)
            self.cached_prefill_nonuniform_previous += int(
                len(previous_lengths) != 1
            )
            self.cached_prefill_unready += int(
                not all(self.ready[slot] for slot in slots)
            )
            self.cached_prefill_noncontiguous += int(not contiguous_slots)
        if positive_previous and all(self.ready[slot] for slot in slots):
            groups: dict[
                tuple[int, ...], list[tuple[int, int, int, int]]
            ] = {}
            for item in ordered_plan:
                slot, begin, end, previous_length = item
                metadata = self.metadata[slot]
                signature = (
                    end - begin,
                    previous_length,
                    int(metadata["state_len"]),
                    int(metadata["coverage"]),
                    int(metadata["recent_len"]),
                    int(metadata["leaf_count"]),
                    int(metadata["overflow_safe_until"]),
                )
                groups.setdefault(signature, []).append(item)
            for group in groups.values():
                run_begin = 0
                while run_begin < len(group):
                    run_end = run_begin + 1
                    while (
                        run_end < len(group)
                        and group[run_end][0] == group[run_end - 1][0] + 1
                    ):
                        run_end += 1
                    self._direct_cached_prefill_group(
                        query,
                        key,
                        value,
                        output,
                        tuple(group[run_begin:run_end]),
                    )
                    run_begin = run_end
            return output
        for slot, begin, end, previous_length in plan:
            if end <= begin:
                continue
            q = query[begin:end].permute(1, 0, 2).unsqueeze(0)
            k = key[begin:end].permute(1, 0, 2).unsqueeze(0)
            v = value[begin:end].permute(1, 0, 2).unsqueeze(0)
            if previous_length == 0 and not self.ready[slot]:
                result, cache = self.engine(q, k, v, use_cache=True)
                if cache is None:
                    raise AssertionError("direct LOD prefill did not return a cache")
                self.install(slot, cache)
            else:
                if not self.ready[slot]:
                    raise RuntimeError("cached LOD prefill row is not initialized")
                cache = self._row_cache(slot)
                if cache.total_length != previous_length:
                    raise RuntimeError(
                        "cached LOD prefill length differs from its prepared plan"
                    )
                result, cache = self.engine(q, k, v, cache=cache, use_cache=True)
                if cache is None:
                    raise AssertionError("cached LOD prefill did not return a cache")
                self._synchronize_row(slot, cache)
            self.engine.reset_runtime_cache()
            output[begin:end].copy_(result.squeeze(0).permute(1, 0, 2))
        return output

    def record_native_appends(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Keep dual-cache LOD copies current after a mixed native batch."""
        plan = self.native_append_plan
        self.native_append_plan = None
        if plan is None:
            return
        for slot, begin, end, previous_length in plan:
            if end - begin != 1 or not self.ready[slot]:
                raise AssertionError("dual-cache append plan is invalid")
            metadata = self.metadata[slot]
            if int(metadata.get("total_len", -1)) != previous_length:
                raise RuntimeError("dual-cache append length changed before attention")
            recent_len = int(self.local_lens[slot].item())
            if recent_len >= self.local_capacity:
                raise RuntimeError("dual-cache append exceeded its local cache")
            self.state["recent_k"][slot, :, recent_len, :].copy_(key[begin])
            self.state["recent_v"][slot, :, recent_len, :].copy_(value[begin])
            recent_len += 1
            self.local_lens[slot].fill_(recent_len)
            metadata["recent_len"] = recent_len
            metadata["total_len"] = previous_length + 1
        self.native_append_calls += 1

    def _row_cache(self, slot: int) -> KernelLODCache:
        return self._range_cache(slot, slot + 1)

    def _range_cache(self, start: int, stop: int) -> KernelLODCache:
        if not 0 <= start < stop <= self.max_requests:
            raise IndexError("LOD cache row range is outside the fixed pool")
        metadata = self.metadata[start]
        scalar_names = (
            "state_len",
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
            coverage=int(metadata["coverage"]),
            state_capacity=self.state_capacity,
            recent_len=int(metadata["recent_len"]),
            total_len=int(metadata["total_len"]),
        )
        page_pool = self.state["page_cache"]
        page: dict[str, object] = {}
        for name, value in page_pool.items():
            page[name] = (
                value[start:stop]
                if isinstance(value, torch.Tensor) and value.ndim
                else value
            )
        page.update(
            leaf_count=int(metadata["leaf_count"]),
            leaf_capacity=self.leaf_capacity,
            overflow_active=True,
            overflow_safe_until=int(metadata["overflow_safe_until"]),
        )
        state["page_cache"] = page
        return KernelLODCache(state)

    def _catch_up_target(
        self, slot: int, total_length: int
    ) -> tuple[int, int]:
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
        page = row.state["page_cache"]
        self.metadata[slot].update(
            state_len=int(row.state["state_len"]),
            coverage=int(row.state["coverage"]),
            total_len=int(row.state["total_len"]),
            recent_len=int(row.state["recent_len"]),
            leaf_count=int(page["leaf_count"]),
            overflow_safe_until=int(page["overflow_safe_until"]),
        )
        self.local_lens[slot].fill_(int(row.state["recent_len"]))

    def catch_up_many(self, requests: list[tuple[int, int]]) -> None:
        """Batch equal-metadata contiguous rows at a state-update boundary."""
        pending: dict[tuple[int, ...], list[int]] = {}
        for slot, total_length in requests:
            if not self.ready[slot]:
                raise RuntimeError("cannot catch up an uninitialized LOD request row")
            metadata = self.metadata[slot]
            recent_length, target_coverage = self._catch_up_target(
                slot, total_length
            )
            if int(metadata["coverage"]) >= target_coverage:
                metadata["total_len"] = total_length
                metadata["recent_len"] = recent_length
                continue
            signature = (
                total_length,
                int(metadata["state_len"]),
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
                        coverage=int(row.state["coverage"]),
                        total_len=int(row.state["total_len"]),
                        recent_len=int(row.state["recent_len"]),
                        leaf_count=int(page["leaf_count"]),
                        overflow_safe_until=int(page["overflow_safe_until"]),
                    )
                self.local_lens[start_slot:stop_slot].fill_(
                    int(row.state["recent_len"])
                )
                begin = end

    def _buffers(self, query: torch.Tensor, rows: int) -> dict[str, torch.Tensor]:
        buffers = self.decode_buffers.get(rows)
        if buffers is None or buffers["partial_out"].device != query.device:
            buffers = new_fused_decode_buffers(
                query,
                splits=int(self.engine.decode_split_kv),
                state_capacity=self.state_capacity,
                route_group_size=int(self.engine.decode_route_group_size),
            )
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

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self.decode_calls += 1
        rows = int(metadata.num_actual_tokens)
        if rows == 0:
            return output
        q = query[:rows].unsqueeze(2)
        k = key[:rows].unsqueeze(2)
        v = value[:rows].unsqueeze(2)
        page = self.state["page_cache"]
        result = fused_decode_paged_lod_attention(
            q,
            self.state["state_k"],
            self.state["state_v"],
            self.state["counts"],
            self.state["recent_k"],
            self.state["recent_v"],
            page["leaf_k"],
            page["leaf_v"],
            page["slot_pages"],
            page["overflow_page_keys"],
            page["overflow_page_values"],
            page["overflow_used"],
            page["slot_lengths"],
            None,
            sink_k=self.state.get("sink_k"),
            sink_v=self.state.get("sink_v"),
            state_len=self.state_capacity,
            local_len=self.local_capacity,
            cache_indices=self.active_indices[:rows],
            local_lens=self.local_lens,
            new_k=k,
            new_v=v,
            kv_group_size=self.query_heads // self.kv_heads,
            scale=float(self.engine.scaling),
            hash_probes=int(self.engine.leaf_hash_probes),
            block_n=int(self.engine.decode_block_n),
            num_warps=int(self.engine.decode_num_warps),
            waves_per_eu=int(self.engine.leaf_waves_per_eu),
            split_kv=int(self.engine.decode_split_kv),
            buffers=self._buffers(q, rows),
            use_dot=bool(self.engine.decode_use_dot),
            fuse_state_route=True,
            route_group_size=int(self.engine.decode_route_group_size),
            route_num_warps=int(self.engine.decode_route_num_warps),
            route_reduce_num_warps=int(self.engine.decode_route_reduce_num_warps),
            final_reduce_num_warps=int(self.engine.decode_final_reduce_num_warps),
            fuse_final_reduce=bool(self.engine.decode_fuse_final_reduce),
            route_use_dot=bool(self.engine.decode_route_use_dot),
            route_gqa_grouped=bool(self.engine.decode_route_gqa_grouped),
            protected_len=self.engine._protected_state_len(self.state_capacity),
            recursive_page_cache=page,
            recursive_quant_group_size=int(self.engine.leaf_quant_group_size),
        )
        output[:rows].copy_(result.squeeze(2))
        advance_decode_cache_lengths(self.active_indices[:rows], self.local_lens)
        return output


__all__ = ["VLLMLayerLODPool"]
