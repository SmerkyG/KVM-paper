"""Audit recursive LOD routing against exact DiffusionGemma attention mass."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math
import re
from types import MethodType
from typing import Any, Iterator

import torch
from torch import nn

from .hf_diffusion_gemma_lod_attention import _cache_context, _project_qkv
from .triton_lod_engines import KernelLODCache


_NATIVE_ACTIVE_ATTRIBUTE = "_diffusion_gemma_native_attention_active"
_ORIGINAL_STEP_ATTRIBUTE = "_diffusion_gemma_route_mass_original_step"
_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b"
)


def _decoder_attention_modules(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "model", model)
    decoder = getattr(base, "decoder", None)
    if decoder is None:
        raise TypeError("expected a DiffusionGemma model with a decoder")
    return [layer.self_attn for layer in decoder.layers]


@contextmanager
def _native_decoder_attention(model: nn.Module) -> Iterator[None]:
    modules = _decoder_attention_modules(model)
    for module in modules:
        setattr(module, _NATIVE_ACTIVE_ATTRIBUTE, True)
    try:
        yield
    finally:
        for module in modules:
            if hasattr(module, _NATIVE_ACTIVE_ATTRIBUTE):
                delattr(module, _NATIVE_ACTIVE_ATTRIBUTE)


class DiffusionGemmaRouteMassComparator:
    """Measure whether spherical LOD opens the native attention's UUID pages.

    Each denoising step receives one shadow native decoder call on the exact
    cache and canvas used by the real LOD trajectory.  Hooks project the native
    hidden states, route them through the existing LOD sidecar, and compare the
    selected pages with the exact prompt attention distribution.  The real LOD
    query routes are then scored counterfactually under those same native
    queries, isolating within-decoder hidden-state drift.
    """

    query_position_bins = ((0, 8), (8, 16), (16, 32), (32, 64))
    noise_edges = (0.0, 0.25, 0.5, 0.75, 1.000001)

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        query_limit: int = 64,
        key_chunk: int = 512,
        query_chunk: int = 4,
    ) -> None:
        if query_limit < 1 or key_chunk < 1 or query_chunk < 1:
            raise ValueError("route-mass diagnostic dimensions must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.query_limit = query_limit
        self.key_chunk = key_chunk
        self.query_chunk = query_chunk
        self.phase: str | None = None
        self.current_step = 0
        self.current_noise: torch.Tensor | None = None
        self.current_active: torch.Tensor | None = None
        self.current_targets: list[list[int]] | None = None
        self.current_uuids: list[str] | None = None
        self.current_prompt_lengths: list[int] | None = None
        self._input_batches: list[tuple[torch.Tensor, tuple[Any, ...]]] = []
        self._hooks: list[Any] = []
        self._native_records: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._target_cache: dict[tuple[int, tuple[tuple[int, ...], ...]], dict[str, torch.Tensor]] = {}
        self._stats: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.step_calls = 0

    def _parse_targets(
        self, input_ids: torch.Tensor
    ) -> tuple[list[list[int]], list[str], list[int]]:
        pad_id = self.tokenizer.pad_token_id
        targets: list[list[int]] = []
        uuids: list[str] = []
        lengths: list[int] = []
        for row in input_ids.detach().cpu().tolist():
            logical = row if pad_id is None else [token for token in row if token != pad_id]
            text = self.tokenizer.decode(
                logical,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            if list(encoded["input_ids"]) != logical:
                raise RuntimeError("UUID diagnostic could not invert prompt tokenization")
            matches = list(_UUID.finditer(text))
            if len(matches) != 1:
                raise RuntimeError(
                    f"UUID diagnostic expected one UUID in the prompt, found {len(matches)}"
                )
            match = matches[0]
            positions = [
                index
                for index, (begin, end) in enumerate(encoded["offset_mapping"])
                if end > match.start() and begin < match.end()
            ]
            if not positions:
                raise RuntimeError("UUID diagnostic found no target tokens")
            targets.append(positions)
            uuids.append(match.group(0))
            lengths.append(len(logical))
        return targets, uuids, lengths

    def _targets_for(self, input_ids: torch.Tensor) -> tuple[Any, ...]:
        for previous, parsed in self._input_batches:
            if previous is input_ids:
                return parsed
        parsed = self._parse_targets(input_ids)
        self._input_batches.append((input_ids, parsed))
        return parsed

    @staticmethod
    def _repeat_kv(key: torch.Tensor, groups: int) -> torch.Tensor:
        if groups == 1:
            return key
        return key.repeat_interleave(groups, dim=1)

    def _source_lse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        groups: int,
        scale: float,
        length: int,
    ) -> torch.Tensor:
        result = torch.full(
            query.shape[:3],
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        for begin in range(0, length, self.key_chunk):
            chunk = key[..., begin : min(begin + self.key_chunk, length), :]
            # Match the model's BF16 QK path and keep the expensive GEMM on
            # tensor cores; all softmax accumulation remains FP32 below.
            scores = torch.matmul(
                query,
                self._repeat_kv(chunk, groups).transpose(-1, -2),
            ).float() * scale
            result = torch.logaddexp(result, torch.logsumexp(scores, dim=-1))
        return result

    @staticmethod
    def _gather_target_keys(
        state: dict[str, Any],
        page_cache: dict[str, Any],
        target_positions: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        leaf_k = page_cache["leaf_k"]
        if not isinstance(leaf_k, torch.Tensor):
            raise TypeError("recursive LOD leaf keys are missing")
        batch, kv_heads, _, head_dim = leaf_k.shape
        target_count = max(len(positions) for positions in target_positions)
        keys = torch.zeros(
            batch,
            kv_heads,
            target_count,
            head_dim,
            dtype=leaf_k.dtype,
            device=leaf_k.device,
        )
        valid = torch.zeros(
            batch, target_count, dtype=torch.bool, device=leaf_k.device
        )
        archive_index = torch.full(
            (batch, target_count), -1, dtype=torch.long, device=leaf_k.device
        )
        leaf_count = int(page_cache["leaf_count"])
        recent_k = state["recent_k"]
        recent_len = int(state["recent_len"])
        sink_k = state.get("sink_k")
        separated = int(sink_k.size(2)) if isinstance(sink_k, torch.Tensor) else 0
        for batch_index, positions in enumerate(target_positions):
            for target_index, position in enumerate(positions):
                valid[batch_index, target_index] = True
                if position < separated:
                    keys[batch_index, :, target_index] = sink_k[batch_index, :, position]
                    continue
                leaf_index = position - separated
                if leaf_index < leaf_count:
                    keys[batch_index, :, target_index] = leaf_k[
                        batch_index, :, leaf_index
                    ]
                    archive_index[batch_index, target_index] = leaf_index
                    continue
                recent_index = leaf_index - leaf_count
                if not 0 <= recent_index < recent_len:
                    raise RuntimeError(
                        "UUID token position lies outside the LOD prompt cache"
                    )
                keys[batch_index, :, target_index] = recent_k[
                    batch_index, :, recent_index
                ]
        return keys, valid, archive_index

    def _target_metadata(
        self,
        state: dict[str, Any],
        page_cache: dict[str, Any],
        target_positions: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        cache_key = (id(state), tuple(tuple(row) for row in target_positions))
        cached = self._target_cache.get(cache_key)
        if cached is not None:
            return cached
        target_keys, target_valid, archive_index = self._gather_target_keys(
            state, page_cache, target_positions
        )
        page_indices = page_cache["page_indices"]
        slot_pages = page_cache["slot_pages"]
        if not isinstance(page_indices, torch.Tensor) or not isinstance(
            slot_pages, torch.Tensor
        ):
            raise TypeError("recursive LOD page metadata is missing")
        batch, kv_heads, page_capacity, page_size = page_indices.shape
        target_count = int(archive_index.size(1))
        leaf = archive_index[:, None, None, None, :]
        page_match = page_indices[..., None].eq(leaf) & archive_index[
            :, None, None, None, :
        ].ge(0)
        has_page = page_match.any(dim=3)
        page_ids = torch.arange(page_capacity, device=page_indices.device).view(
            1, 1, page_capacity, 1
        )
        target_page = torch.where(has_page, page_ids, -1).amax(dim=2)
        slot_match = slot_pages[..., None].eq(
            target_page[:, :, None, None, :]
        ) & target_page[:, :, None, None, :].ge(0)
        has_slot = slot_match.any(dim=3)
        slot_ids = torch.arange(
            int(slot_pages.size(2)), device=slot_pages.device
        ).view(1, 1, -1, 1)
        target_slot = torch.where(has_slot, slot_ids, -1).amax(dim=2)
        result = {
            "keys": target_keys,
            "valid": target_valid,
            "archive_valid": archive_index.ge(0),
            "page": target_page,
            "slot": target_slot,
        }
        self._target_cache[cache_key] = result
        return result

    def _select_pages(
        self,
        query: torch.Tensor,
        top_slots: torch.Tensor,
        page_cache: dict[str, Any],
        *,
        groups: int,
        scale: float,
    ) -> torch.Tensor:
        if bool(page_cache.get("overflow_active", False)):
            raise NotImplementedError(
                "route-mass diagnostic does not support overflow posting lists"
            )
        if bool(page_cache.get("summary_quantization_finalized", False)):
            raise NotImplementedError(
                "route-mass diagnostic requires unquantized page summaries"
            )
        page_sum_k = page_cache["page_sum_k"]
        page_counts = page_cache["page_counts"]
        slot_pages = page_cache["slot_pages"]
        slot_lengths = page_cache["slot_lengths"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (page_sum_k, page_counts, slot_pages, slot_lengths)
        ):
            raise TypeError("recursive LOD page summaries are incomplete")
        batch, heads, query_len, routes = top_slots.shape
        selected = torch.full_like(top_slots, -1)
        batch_index = torch.arange(batch, device=query.device)[:, None, None]
        kv_index = (
            torch.arange(heads, device=query.device) // groups
        )[None, :, None]
        inline_pages = int(slot_pages.size(3))
        page_size = int(page_cache["page_size"])
        for route in range(routes):
            route_slot = top_slots[..., route]
            safe_slot = route_slot.clamp_min(0)
            lengths = slot_lengths[batch_index, kv_index, safe_slot]
            lengths = torch.where(route_slot.ge(0), lengths, torch.zeros_like(lengths))
            max_pages = min(
                inline_pages,
                math.ceil(int(lengths.max().item()) / page_size),
            )
            if max_pages == 0:
                continue
            for query_begin in range(0, query_len, self.query_chunk):
                query_end = min(query_begin + self.query_chunk, query_len)
                q = query[..., query_begin:query_end, :].float()
                slot = safe_slot[..., query_begin:query_end]
                length = lengths[..., query_begin:query_end]
                page_ordinals = torch.arange(
                    max_pages, device=query.device
                ).view(1, 1, 1, -1)
                page_ids = slot_pages[
                    batch_index[..., None],
                    kv_index[..., None],
                    slot[..., None],
                    page_ordinals,
                ].long()
                valid_page = (
                    page_ordinals < ((length[..., None] + page_size - 1) // page_size)
                ) & page_ids.ge(0)
                safe_page = page_ids.clamp_min(0)
                counts = page_counts[
                    batch_index[..., None],
                    kv_index[..., None],
                    safe_page,
                ].float().clamp_min(1.0)
                sums = page_sum_k[
                    batch_index[..., None],
                    kv_index[..., None],
                    safe_page,
                ].float()
                scores = (
                    (sums / counts[..., None]) * q[..., None, :]
                ).sum(dim=-1) * scale + counts.log()
                scores.masked_fill_(~valid_page, float("-inf"))
                best_score = scores.amax(dim=-1, keepdim=True)
                best_page = torch.where(
                    scores.eq(best_score), page_ids, torch.full_like(page_ids, -1)
                ).amax(dim=-1)
                selected[..., query_begin:query_end, route] = best_page
        return selected

    def _selected_page_lse(
        self,
        query: torch.Tensor,
        selected_pages: torch.Tensor,
        page_cache: dict[str, Any],
        *,
        groups: int,
        scale: float,
    ) -> torch.Tensor:
        page_indices = page_cache["page_indices"]
        page_counts = page_cache["page_counts"]
        leaf_k = page_cache["leaf_k"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (page_indices, page_counts, leaf_k)
        ):
            raise TypeError("recursive LOD exact page storage is incomplete")
        batch, heads, query_len, routes = selected_pages.shape
        page_size = int(page_cache["page_size"])
        result = torch.full(
            (batch, heads, query_len),
            float("-inf"),
            dtype=torch.float32,
            device=query.device,
        )
        batch_index = torch.arange(batch, device=query.device)[:, None, None, None]
        kv_index = (
            torch.arange(heads, device=query.device) // groups
        )[None, :, None, None]
        token_index = torch.arange(page_size, device=query.device).view(
            1, 1, 1, page_size
        )
        leaf_count = int(page_cache["leaf_count"])
        for route in range(routes):
            route_pages = selected_pages[..., route]
            for query_begin in range(0, query_len, self.query_chunk):
                query_end = min(query_begin + self.query_chunk, query_len)
                pages = route_pages[..., query_begin:query_end]
                safe_pages = pages.clamp_min(0)
                counts = page_counts[
                    batch_index,
                    kv_index,
                    safe_pages[..., None],
                ]
                leaf_indices = page_indices[
                    batch_index,
                    kv_index,
                    safe_pages[..., None],
                    token_index,
                ].long()
                valid = (
                    pages[..., None].ge(0)
                    & token_index.lt(counts)
                    & leaf_indices.ge(0)
                    & leaf_indices.lt(leaf_count)
                )
                safe_leaf = leaf_indices.clamp(0, max(leaf_count - 1, 0))
                keys = leaf_k[batch_index, kv_index, safe_leaf].float()
                scores = (
                    keys
                    * query[..., query_begin:query_end, None, :].float()
                ).sum(dim=-1) * scale
                scores.masked_fill_(~valid, float("-inf"))
                page_lse = torch.logsumexp(scores, dim=-1)
                result[..., query_begin:query_end] = torch.logaddexp(
                    result[..., query_begin:query_end], page_lse
                )
        return result

    def _route(
        self,
        engine: nn.Module,
        state: dict[str, Any],
        query: torch.Tensor,
        *,
        open_count: int,
    ) -> torch.Tensor:
        original_topk = int(engine.two_level_topk)
        original_prefill_topk = engine.prefill_two_level_topk
        original_fused = engine.fused_prefill_route_coarse
        engine.two_level_topk = open_count
        engine.prefill_two_level_topk = None
        engine.fused_prefill_route_coarse = False
        engine._diffusion_decoder_routing_active = True
        try:
            empty = query.new_empty(
                int(query.size(0)),
                int(state["state_k"].size(1)),
                0,
                int(query.size(-1)),
            )
            return engine._route_top_slots(
                query,
                state["state_k"],
                state["state_v"],
                state["counts"],
                state_len=int(state["state_len"]),
                state_capacity=int(state["state_capacity"]),
                local_k=empty,
                local_v=empty,
                local_len=0,
                page_cache=state["page_cache"],
            )
        finally:
            engine._diffusion_decoder_routing_active = False
            engine.two_level_topk = original_topk
            engine.prefill_two_level_topk = original_prefill_topk
            engine.fused_prefill_route_coarse = original_fused

    def _reduce(
        self,
        batch_index: int,
        begin: int,
        end: int,
        *,
        selected_mass: torch.Tensor | None,
        remote_mass: torch.Tensor | None,
        uuid_mass: torch.Tensor | None,
        uuid_slot_mass: torch.Tensor | None,
        uuid_page_mass: torch.Tensor | None,
        slot_hit: torch.Tensor,
        page_hit: torch.Tensor,
        target_valid: torch.Tensor,
        route_overlap: torch.Tensor | None,
    ) -> dict[str, float]:
        query_count = int(slot_hit.size(1)) * (end - begin)
        valid = target_valid[batch_index]
        archived_targets = int(valid.sum().item())
        values = {
            "query_rows": float(query_count),
            "uuid_token_rows": float(query_count * archived_targets),
            "uuid_slot_token_hits": float(
                slot_hit[batch_index, :, begin:end, valid].sum().item()
            ),
            "uuid_page_token_hits": float(
                page_hit[batch_index, :, begin:end, valid].sum().item()
            ),
            "uuid_slot_any_hits": float(
                slot_hit[batch_index, :, begin:end, valid].any(dim=-1).sum().item()
            ) if archived_targets else 0.0,
            "uuid_page_any_hits": float(
                page_hit[batch_index, :, begin:end, valid].any(dim=-1).sum().item()
            ) if archived_targets else 0.0,
            "uuid_archived_query_rows": float(query_count if archived_targets else 0),
        }
        if selected_mass is not None and remote_mass is not None and uuid_mass is not None:
            selected = selected_mass[batch_index, :, begin:end]
            remote = remote_mass[batch_index, :, begin:end]
            target = uuid_mass[batch_index, :, begin:end]
            target_slot = uuid_slot_mass[batch_index, :, begin:end]
            target_page = uuid_page_mass[batch_index, :, begin:end]
            values.update(
                selected_page_mass_sum=float(selected.sum().item()),
                remote_mass_sum=float(remote.sum().item()),
                uuid_mass_sum=float(target.sum().item()),
                uuid_slot_mass_sum=float(target_slot.sum().item()),
                uuid_page_mass_sum=float(target_page.sum().item()),
                selected_remote_fraction_sum=float(
                    (selected / remote.clamp_min(1e-30)).clamp_max(1.0).sum().item()
                ),
            )
        if route_overlap is not None:
            overlap = route_overlap[batch_index, :, begin:end]
            values["route_overlap_sum"] = float(overlap.sum().item())
            values["route_overlap_rows"] = float(overlap.numel())
        return values

    def _add(self, key: str, values: dict[str, float]) -> None:
        target = self._stats[key]
        for name, value in values.items():
            target[name] += value

    def _record_slices(
        self,
        *,
        phase: str,
        layer_index: int,
        runtime_indices: torch.Tensor,
        target_positions: list[list[int]],
        prompt_lengths: list[int],
        uuids: list[str],
        selected_mass: torch.Tensor | None,
        remote_mass: torch.Tensor | None,
        uuid_mass: torch.Tensor | None,
        uuid_slot_mass: torch.Tensor | None,
        uuid_page_mass: torch.Tensor | None,
        slot_hit: torch.Tensor,
        page_hit: torch.Tensor,
        target_valid: torch.Tensor,
        route_overlap: torch.Tensor | None,
    ) -> None:
        query_len = int(slot_hit.size(2))
        full_indices = runtime_indices.detach().cpu().tolist()
        for local_index, full_index in enumerate(full_indices):
            if self.current_active is not None and not bool(
                self.current_active[full_index].item()
            ):
                continue
            values = self._reduce(
                local_index,
                0,
                query_len,
                selected_mass=selected_mass,
                remote_mass=remote_mass,
                uuid_mass=uuid_mass,
                uuid_slot_mass=uuid_slot_mass,
                uuid_page_mass=uuid_page_mass,
                slot_hit=slot_hit,
                page_hit=page_hit,
                target_valid=target_valid,
                route_overlap=route_overlap,
            )
            prefix = f"{phase}:"
            self._add(prefix + "overall", values)
            self._add(prefix + f"layer:{layer_index}", values)
            self._add(prefix + f"step:{self.current_step}", values)
            noise = float(self.current_noise[full_index].item())
            noise_bin = next(
                index
                for index in range(len(self.noise_edges) - 1)
                if self.noise_edges[index] <= noise < self.noise_edges[index + 1]
            )
            self._add(
                prefix
                + f"noise:{self.noise_edges[noise_bin]:.2f}-{self.noise_edges[noise_bin + 1]:.2f}",
                values,
            )
            location = target_positions[local_index][0] / prompt_lengths[local_index]
            quartile = min(int(location * 4), 3)
            self._add(prefix + f"needle_quartile:{quartile + 1}", values)
            self._add(prefix + f"uuid:{uuids[local_index]}", values)
            for begin, end in self.query_position_bins:
                end = min(end, query_len)
                if begin >= end:
                    continue
                position_values = self._reduce(
                    local_index,
                    begin,
                    end,
                    selected_mass=selected_mass,
                    remote_mass=remote_mass,
                    uuid_mass=uuid_mass,
                    uuid_slot_mass=uuid_slot_mass,
                    uuid_page_mass=uuid_page_mass,
                    slot_hit=slot_hit,
                    page_hit=page_hit,
                    target_valid=target_valid,
                    route_overlap=route_overlap,
                )
                self._add(prefix + f"query_positions:{begin}-{end}", position_values)

    def _attention_hook(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        if self.phase not in ("native", "lod"):
            return
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get("position_embeddings")
        past_key_values = kwargs.get("past_key_values")
        if not isinstance(hidden_states, torch.Tensor) or position_embeddings is None:
            raise RuntimeError("route-mass hook did not receive decoder inputs")
        if past_key_values is None:
            raise RuntimeError("route-mass hook did not receive the encoder cache")
        layer = _cache_context(past_key_values, create=False).get(module.layer_idx)
        if layer is None:
            return
        query, canvas_key, _ = _project_qkv(
            module, hidden_states, position_embeddings
        )
        query = query[..., : min(self.query_limit, int(query.size(2))), :]
        for runtime_index, runtime in enumerate(layer.grouped.runtimes):
            if runtime.engine is None or not isinstance(
                runtime.lod_cache, KernelLODCache
            ):
                raise RuntimeError("route-mass diagnostic requires kernel LOD state")
            state = runtime.lod_cache.state
            page_cache = state.get("page_cache")
            if not isinstance(page_cache, dict) or not isinstance(
                page_cache.get("page_indices"), torch.Tensor
            ):
                raise RuntimeError("route-mass diagnostic requires recursive pages")
            indices = runtime.indices.to(query.device)
            group_query = query.index_select(0, indices).contiguous()
            group_canvas_key = canvas_key.index_select(0, indices).contiguous()
            target_positions = [self.current_targets[index] for index in indices.tolist()]
            prompt_lengths = [self.current_prompt_lengths[index] for index in indices.tolist()]
            uuids = [self.current_uuids[index] for index in indices.tolist()]
            groups = int(group_query.size(1)) // int(state["state_k"].size(1))
            open_count = int(
                getattr(
                    module,
                    "_diffusion_gemma_decoder_open_count",
                    layer.settings.open_count,
                )
            )
            top_slots = self._route(
                runtime.engine, state, group_query, open_count=open_count
            )
            selected_pages = self._select_pages(
                group_query,
                top_slots,
                page_cache,
                groups=groups,
                scale=float(module.scaling),
            )
            metadata = self._target_metadata(
                state, page_cache, target_positions
            )
            target_slot = metadata["slot"].repeat_interleave(groups, dim=1)
            target_page = metadata["page"].repeat_interleave(groups, dim=1)
            target_valid = metadata["archive_valid"]
            slot_hit = top_slots[..., None].eq(
                target_slot[:, :, None, None, :]
            ).any(dim=3) & target_slot[:, :, None, :].ge(0)
            page_hit = selected_pages[..., None].eq(
                target_page[:, :, None, None, :]
            ).any(dim=3) & target_page[:, :, None, :].ge(0)

            selected_mass = remote_mass = uuid_mass = None
            uuid_slot_mass = uuid_page_mass = route_overlap = None
            route_key = (int(module.layer_idx), runtime_index)
            if self.phase == "native":
                leaf_k = page_cache["leaf_k"]
                leaf_count = int(page_cache["leaf_count"])
                remote_lse = self._source_lse(
                    group_query,
                    leaf_k,
                    groups=groups,
                    scale=float(module.scaling),
                    length=leaf_count,
                )
                denominator = remote_lse
                recent_len = int(state["recent_len"])
                if recent_len:
                    denominator = torch.logaddexp(
                        denominator,
                        self._source_lse(
                            group_query,
                            state["recent_k"],
                            groups=groups,
                            scale=float(module.scaling),
                            length=recent_len,
                        ),
                    )
                sink_k = state.get("sink_k")
                if isinstance(sink_k, torch.Tensor):
                    denominator = torch.logaddexp(
                        denominator,
                        self._source_lse(
                            group_query,
                            sink_k,
                            groups=groups,
                            scale=float(module.scaling),
                            length=int(sink_k.size(2)),
                        ),
                    )
                denominator = torch.logaddexp(
                    denominator,
                    self._source_lse(
                        group_query,
                        group_canvas_key,
                        groups=groups,
                        scale=float(module.scaling),
                        length=int(group_canvas_key.size(2)),
                    ),
                )
                selected_lse = self._selected_page_lse(
                    group_query,
                    selected_pages,
                    page_cache,
                    groups=groups,
                    scale=float(module.scaling),
                )
                target_keys = self._repeat_kv(metadata["keys"], groups)
                target_valid_full = metadata["valid"][:, None, None, :]
                target_archive_valid_full = metadata["archive_valid"][
                    :, None, None, :
                ]
                target_score_parts = []
                target_slot_lse_parts = []
                target_page_lse_parts = []
                for begin in range(0, int(group_query.size(2)), self.query_chunk):
                    end = min(begin + self.query_chunk, int(group_query.size(2)))
                    scores = torch.matmul(
                        group_query[..., begin:end, :],
                        target_keys.transpose(-1, -2),
                    ).float() * float(module.scaling)
                    scores.masked_fill_(~target_valid_full, float("-inf"))
                    target_score_parts.append(scores)
                    target_slot_lse_parts.append(
                        torch.logsumexp(
                            scores.masked_fill(
                                ~slot_hit[..., begin:end, :], float("-inf")
                            ),
                            dim=-1,
                        )
                    )
                    target_page_lse_parts.append(
                        torch.logsumexp(
                            scores.masked_fill(
                                ~page_hit[..., begin:end, :], float("-inf")
                            ),
                            dim=-1,
                        )
                    )
                target_scores = torch.cat(target_score_parts, dim=2)
                # Tokens inside the exact local field need no route.  Restrict
                # UUID visibility recall to archived tokens whose information
                # actually depends on centroid/page selection.
                target_lse = torch.logsumexp(
                    target_scores.masked_fill(
                        ~target_archive_valid_full, float("-inf")
                    ),
                    dim=-1,
                )
                target_slot_lse = torch.cat(target_slot_lse_parts, dim=2)
                target_page_lse = torch.cat(target_page_lse_parts, dim=2)
                selected_mass = torch.exp(selected_lse - denominator)
                remote_mass = torch.exp(remote_lse - denominator)
                uuid_mass = torch.exp(target_lse - denominator)
                uuid_slot_mass = torch.exp(target_slot_lse - denominator)
                uuid_page_mass = torch.exp(target_page_lse - denominator)
                self._native_records[route_key] = {
                    "query": group_query.detach(),
                    "pages": selected_pages.detach(),
                    "denominator": denominator.detach(),
                    "remote_mass": remote_mass.detach(),
                    "uuid_mass": uuid_mass.detach(),
                    "target_scores": target_scores.detach(),
                }
            else:
                native = self._native_records.get(route_key)
                native_pages = None if native is None else native["pages"]
                if native_pages is not None and native_pages.shape == selected_pages.shape:
                    valid_route = selected_pages.ge(0)
                    overlap = selected_pages[..., None].eq(
                        native_pages[..., None, :]
                    ).any(dim=-1)
                    route_overlap = (
                        (overlap & valid_route).sum(dim=-1)
                        / valid_route.sum(dim=-1).clamp_min(1)
                    )
                    native_query = native["query"]
                    denominator = native["denominator"]
                    selected_lse = self._selected_page_lse(
                        native_query,
                        selected_pages,
                        page_cache,
                        groups=groups,
                        scale=float(module.scaling),
                    )
                    selected_mass = torch.exp(selected_lse - denominator)
                    remote_mass = native["remote_mass"]
                    uuid_mass = native["uuid_mass"]
                    target_scores = native["target_scores"]
                    uuid_slot_mass = torch.exp(
                        torch.logsumexp(
                            target_scores.masked_fill(~slot_hit, float("-inf")),
                            dim=-1,
                        )
                        - denominator
                    )
                    uuid_page_mass = torch.exp(
                        torch.logsumexp(
                            target_scores.masked_fill(~page_hit, float("-inf")),
                            dim=-1,
                        )
                        - denominator
                    )
            self._record_slices(
                phase=self.phase,
                layer_index=int(module.layer_idx),
                runtime_indices=runtime.indices,
                target_positions=target_positions,
                prompt_lengths=prompt_lengths,
                uuids=uuids,
                selected_mass=selected_mass,
                remote_mass=remote_mass,
                uuid_mass=uuid_mass,
                uuid_slot_mass=uuid_slot_mass,
                uuid_page_mass=uuid_page_mass,
                slot_hit=slot_hit,
                page_hit=page_hit,
                target_valid=target_valid,
                route_overlap=route_overlap,
            )

    def install(self) -> None:
        if hasattr(self.model, _ORIGINAL_STEP_ATTRIBUTE):
            raise RuntimeError("route-mass comparison is already installed")
        modules = [
            module
            for module in _decoder_attention_modules(self.model)
            if hasattr(module, "_diffusion_gemma_lod_settings")
        ]
        if not modules:
            raise RuntimeError("install DiffusionGemma LOD before route-mass comparison")
        self._hooks = [
            module.register_forward_pre_hook(self._attention_hook, with_kwargs=True)
            for module in modules
        ]
        original_step = self.model._denoising_step
        setattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, original_step)
        comparator = self

        def compared_step(model_self: nn.Module, *args: Any, **kwargs: Any):
            current_canvas = kwargs["current_canvas"]
            input_ids = kwargs["input_ids"]
            sampler = kwargs["sampler"]
            comparator.current_step = int(kwargs["cur_step"])
            comparator.current_active = ~kwargs["finished_denoising"]
            parsed = comparator._targets_for(input_ids)
            (
                comparator.current_targets,
                comparator.current_uuids,
                comparator.current_prompt_lengths,
            ) = parsed
            accepted = sampler.accepted_token_mask
            comparator.current_noise = (
                torch.ones(
                    int(input_ids.size(0)),
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                if accepted is None
                else 1.0 - accepted.float().mean(dim=-1)
            )
            comparator._native_records.clear()
            model_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "current_canvas",
                    "self_conditioning_logits",
                    "mask_mapping",
                    "past_key_values",
                    "decoder_position_ids",
                    "logits_processor",
                    "input_ids",
                    "cur_step",
                    "sampler",
                    "argmax_canvas",
                    "finished_denoising",
                    "diffusion_stopping_criteria",
                    "decoder_forward",
                }
            }
            comparator.phase = "native"
            try:
                with _native_decoder_attention(model_self):
                    native_outputs = model_self(
                        decoder_input_ids=current_canvas,
                        self_conditioning_logits=kwargs["self_conditioning_logits"],
                        decoder_attention_mask=kwargs["mask_mapping"],
                        past_key_values=kwargs["past_key_values"],
                        decoder_position_ids=kwargs["decoder_position_ids"],
                        **model_kwargs,
                    )
                del native_outputs
                comparator.phase = "lod"
                result = original_step(*args, **kwargs)
            finally:
                comparator.phase = None
            comparator.step_calls += 1
            return result

        self.model._denoising_step = MethodType(compared_step, self.model)

    def uninstall(self) -> None:
        original = getattr(self.model, _ORIGINAL_STEP_ATTRIBUTE, None)
        if original is not None:
            self.model._denoising_step = original
            delattr(self.model, _ORIGINAL_STEP_ATTRIBUTE)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @staticmethod
    def _summarize(values: dict[str, float]) -> dict[str, float | None]:
        queries = values.get("query_rows", 0.0)
        target_rows = values.get("uuid_token_rows", 0.0)
        archived_queries = values.get("uuid_archived_query_rows", 0.0)
        result: dict[str, float | None] = {
            "query_rows": int(queries),
            "uuid_token_rows": int(target_rows),
            "uuid_slot_token_recall": (
                values.get("uuid_slot_token_hits", 0.0) / target_rows
                if target_rows
                else None
            ),
            "uuid_page_token_recall": (
                values.get("uuid_page_token_hits", 0.0) / target_rows
                if target_rows
                else None
            ),
            "uuid_slot_any_recall": (
                values.get("uuid_slot_any_hits", 0.0) / archived_queries
                if archived_queries
                else None
            ),
            "uuid_page_any_recall": (
                values.get("uuid_page_any_hits", 0.0) / archived_queries
                if archived_queries
                else None
            ),
        }
        if "selected_page_mass_sum" in values:
            result.update(
                mean_selected_page_mass=values["selected_page_mass_sum"] / queries,
                mean_remote_mass=values["remote_mass_sum"] / queries,
                mean_uuid_mass=values["uuid_mass_sum"] / queries,
                mean_uuid_slot_visible_mass=(
                    values["uuid_slot_mass_sum"] / queries
                ),
                mean_uuid_page_visible_mass=(
                    values["uuid_page_mass_sum"] / queries
                ),
                uuid_mass_slot_recall=(
                    values["uuid_slot_mass_sum"] / values["uuid_mass_sum"]
                    if values["uuid_mass_sum"]
                    else None
                ),
                uuid_mass_page_recall=(
                    values["uuid_page_mass_sum"] / values["uuid_mass_sum"]
                    if values["uuid_mass_sum"]
                    else None
                ),
                mean_selected_remote_fraction=(
                    values["selected_remote_fraction_sum"] / queries
                ),
            )
        if values.get("route_overlap_rows", 0.0):
            result["mean_native_lod_page_route_overlap"] = (
                values["route_overlap_sum"] / values["route_overlap_rows"]
            )
        return result

    def summary(self) -> dict[str, Any]:
        groups: dict[str, dict[str, dict[str, float | None]]] = {
            "native": {},
            "lod": {},
        }
        for key, values in sorted(self._stats.items()):
            phase, group = key.split(":", 1)
            groups[phase][group] = self._summarize(values)
        return {
            "step_calls": self.step_calls,
            "query_limit": self.query_limit,
            "native_mass_definition": (
                "exact native-query softmax mass over the shared native encoder "
                "cache plus current canvas"
            ),
            "lod_mass_definition": (
                "counterfactual exact native-query mass captured by the pages "
                "selected later from the actual LOD hidden trajectory"
            ),
            "uuid_mass_scope": (
                "archived UUID tokens only; UUID tokens in the exact local field "
                "are excluded because they require no route"
            ),
            "groups": groups,
        }


__all__ = ["DiffusionGemmaRouteMassComparator"]
