"""Varied-left-padding support for the generic Hugging Face LOD cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .hf_pytorch_lod_attention import (
    HFLODSettings,
    _build_engine,
    _clear_engine_derived_state,
    _map_batch_tensors,
)
from .triton_lod_engines import KernelLODCache


@dataclass(frozen=True)
class HFLODPaddingGroup:
    """Rows sharing one logical prompt length."""

    prompt_length: int
    rows: tuple[int, ...]
    valid_starts: tuple[int, ...] = ()


@dataclass(frozen=True)
class HFLODPaddingPlan:
    """Device-independent grouping for one physical prompt batch."""

    batch_size: int
    padded_length: int
    groups: tuple[HFLODPaddingGroup, ...]

    @property
    def requires_grouping(self) -> bool:
        return not (
            len(self.groups) == 1
            and self.groups[0].prompt_length == self.padded_length
            and not self.groups[0].valid_starts
        )


@dataclass
class _GroupRuntime:
    prompt_length: int
    indices: torch.Tensor
    valid_starts: torch.Tensor | None = None
    engine: nn.Module | None = None
    lod_cache: Any | None = None


def chunk_align_padding_plan(
    plan: HFLODPaddingPlan,
    *,
    chunk_size: int,
    minimum_length: int = 0,
) -> HFLODPaddingPlan:
    """Right-align rows in chunk-sized buckets with padding in chunk zero."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if minimum_length < 0:
        raise ValueError("minimum length cannot be negative")
    rows_by_length: dict[int, list[tuple[int, int]]] = {}
    for group in plan.groups:
        for row in group.rows:
            original_start = plan.padded_length - group.prompt_length
            aligned_start = (original_start // chunk_size) * chunk_size
            physical_length = plan.padded_length - aligned_start
            valid_start = original_start - aligned_start
            if physical_length < minimum_length:
                physical_length = group.prompt_length
                valid_start = 0
            rows_by_length.setdefault(physical_length, []).append(
                (row, valid_start)
            )
    groups = tuple(
        HFLODPaddingGroup(
            prompt_length,
            tuple(
                row
                for row, _ in sorted(rows_by_length[prompt_length])
            ),
            (
                tuple(
                    start
                    for _, start in sorted(rows_by_length[prompt_length])
                )
                if any(start for _, start in rows_by_length[prompt_length])
                else ()
            ),
        )
        for prompt_length in sorted(rows_by_length)
    )
    return HFLODPaddingPlan(plan.batch_size, plan.padded_length, groups)


def build_padding_plan(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
) -> HFLODPaddingPlan:
    cache_key = None
    if attention_mask is not None:
        try:
            mask_version = int(attention_mask._version)
        except RuntimeError:
            mask_version = None
        cache_key = (
            batch_size,
            sequence_length,
            mask_version,
        )
        cached = getattr(attention_mask, "_hf_lod_padding_plan", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
    if attention_mask is None:
        lengths = [sequence_length] * batch_size
    else:
        if attention_mask.ndim == 4:
            if (
                int(attention_mask.size(0)) != batch_size
                or int(attention_mask.size(-1)) != sequence_length
            ):
                raise ValueError(
                    "the initial HF LOD causal mask must match the prompt batch"
                )
            final_query = attention_mask[:, 0, -1, :]
            allowed = (
                final_query
                if final_query.dtype == torch.bool
                else final_query.eq(0)
            )
            logical_lengths = allowed.sum(dim=-1, dtype=torch.long)
            lengths = [
                int(length) for length in logical_lengths.detach().cpu().tolist()
            ]
            if any(
                length <= 0 or length > sequence_length for length in lengths
            ):
                raise ValueError(
                    "every HF LOD prompt row must contain at least one token"
                )
            positions = torch.arange(sequence_length, device=allowed.device)
            begins = sequence_length - logical_lengths
            expected = positions.unsqueeze(0) >= begins.unsqueeze(1)
            if not bool(torch.all(allowed == expected).item()):
                raise NotImplementedError(
                    "HF LOD supports contiguous left padding, not arbitrary causal masks"
                )
        elif attention_mask.ndim != 2:
            raise NotImplementedError(
                "HF LOD varied-length batching requires a 2D or 4D mask"
            )
        else:
            if tuple(attention_mask.shape) != (batch_size, sequence_length):
                raise ValueError(
                    "the initial HF LOD attention mask must match the prompt batch"
                )
            binary = attention_mask.eq(0) | attention_mask.eq(1)
            if not bool(torch.all(binary).item()):
                raise ValueError(
                    "the HF LOD 2D attention mask must contain only zeros and ones"
                )
            allowed = attention_mask.ne(0)
            logical_lengths = allowed.sum(dim=-1, dtype=torch.long)
            lengths = [
                int(length) for length in logical_lengths.detach().cpu().tolist()
            ]
            if any(
                length <= 0 or length > sequence_length for length in lengths
            ):
                raise ValueError(
                    "every HF LOD prompt row must contain at least one token"
                )
            positions = torch.arange(sequence_length, device=allowed.device)
            begins = sequence_length - logical_lengths
            expected = positions.unsqueeze(0) >= begins.unsqueeze(1)
            if not bool(torch.all(allowed == expected).item()):
                raise NotImplementedError(
                    "HF LOD supports contiguous left padding, not padding within a prompt"
                )

    rows_by_length: dict[int, list[int]] = {}
    for row, length in enumerate(lengths):
        rows_by_length.setdefault(length, []).append(row)
    groups = tuple(
        HFLODPaddingGroup(length, tuple(rows_by_length[length]))
        for length in sorted(rows_by_length)
    )
    plan = HFLODPaddingPlan(batch_size, sequence_length, groups)
    if attention_mask is not None:
        try:
            attention_mask._hf_lod_padding_plan = (cache_key, plan)
        except AttributeError:
            pass
    return plan


def _pad_kernel_prefill(
    settings: HFLODSettings,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
    if settings.engine_backend != "kernel":
        return query, key, value, None
    logical_length = int(query.size(2))
    padding = (-logical_length) % settings.config.chunk_size
    if padding:
        pad = (0, 0, 0, padding)
        query = F.pad(query, pad)
        key = F.pad(key, pad)
        value = F.pad(value, pad)
    return query, key, value, logical_length


def _run_engine(
    engine: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cache: Any,
    use_cache: bool,
    scale: float | None,
    logical_prefill_len: int | None = None,
    prefill_valid_starts: torch.Tensor | None = None,
):
    kwargs = {"cache": cache, "use_cache": use_cache, "scale": scale}
    if logical_prefill_len is not None:
        kwargs["logical_prefill_len"] = logical_prefill_len
    if prefill_valid_starts is not None:
        kwargs["prefill_valid_starts"] = prefill_valid_starts
    return engine(query, key, value, **kwargs)


class GroupedHFLODRuntime:
    """Independent LOD states for every logical-length group in one HF batch."""

    def __init__(self, plan: HFLODPaddingPlan, *, device: torch.device) -> None:
        self.runtimes = [
            _GroupRuntime(
                group.prompt_length,
                torch.tensor(group.rows, dtype=torch.long, device=device),
                (
                    torch.tensor(
                        group.valid_starts, dtype=torch.long, device=device
                    )
                    if group.valid_starts
                    else None
                ),
            )
            for group in plan.groups
        ]

    @staticmethod
    def _attach_kernel_cache(engine: nn.Module | None, lod_cache: Any) -> None:
        _clear_engine_derived_state(engine)
        if engine is not None and isinstance(lod_cache, KernelLODCache):
            engine._lod_state = lod_cache.state

    def consume(
        self,
        settings: HFLODSettings,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        initial_prefill: bool,
        scale: float | None,
    ) -> torch.Tensor:
        output = torch.zeros_like(query)
        sequence_length = int(query.size(2))
        for runtime in self.runtimes:
            begin = sequence_length - runtime.prompt_length if initial_prefill else 0
            indices = runtime.indices.to(query.device)
            group_query = query.index_select(0, indices)[..., begin:, :].contiguous()
            group_key = key.index_select(0, indices)[..., begin:, :].contiguous()
            group_value = value.index_select(0, indices)[..., begin:, :].contiguous()
            logical_prefill_len = None
            if initial_prefill:
                (
                    group_query,
                    group_key,
                    group_value,
                    logical_prefill_len,
                ) = _pad_kernel_prefill(settings, group_query, group_key, group_value)
            if runtime.engine is None:
                runtime.engine = _build_engine(
                    settings, group_query, group_key, scale=scale
                )
            previous_length = (
                0 if runtime.lod_cache is None else int(runtime.lod_cache.total_length)
            )
            group_output, next_cache = _run_engine(
                runtime.engine,
                group_query,
                group_key,
                group_value,
                cache=runtime.lod_cache,
                use_cache=True,
                scale=scale,
                logical_prefill_len=logical_prefill_len,
                prefill_valid_starts=(
                    runtime.valid_starts if initial_prefill else None
                ),
            )
            if next_cache is None:
                raise RuntimeError("grouped LOD engine did not return its owned cache")
            added_length = (
                runtime.prompt_length if initial_prefill else int(group_key.size(2))
            )
            if int(next_cache.total_length) != previous_length + added_length:
                raise AssertionError("grouped LOD engine and cache lengths diverged")
            runtime.lod_cache = next_cache

            full_group_output = torch.zeros(
                (int(indices.numel()), *query.shape[1:]),
                dtype=query.dtype,
                device=query.device,
            )
            valid_length = sequence_length - begin
            full_group_output[..., begin:, :].copy_(
                group_output[..., :valid_length, :]
            )
            output.index_copy_(0, indices, full_group_output)
        return output

    def reset(self) -> None:
        for runtime in self.runtimes:
            _clear_engine_derived_state(runtime.engine)
            if runtime.engine is not None and hasattr(
                runtime.engine, "reset_runtime_cache"
            ):
                runtime.engine.reset_runtime_cache()

    def select_batch(self, indices: torch.Tensor, *, batch_size: int) -> None:
        device = indices.device
        group_ids = torch.empty(batch_size, dtype=torch.long, device=device)
        local_ids = torch.empty_like(group_ids)
        for group_id, runtime in enumerate(self.runtimes):
            rows = runtime.indices.to(device)
            group_ids.index_fill_(0, rows, group_id)
            local_ids.index_copy_(
                0,
                rows,
                torch.arange(int(rows.numel()), dtype=torch.long, device=device),
            )

        selected_groups = group_ids.index_select(0, indices)
        next_runtimes = []
        for group_id, runtime in enumerate(self.runtimes):
            next_rows = torch.nonzero(
                selected_groups.eq(group_id), as_tuple=False
            ).flatten()
            if not int(next_rows.numel()):
                continue
            old_rows = indices.index_select(0, next_rows)
            local_selection = local_ids.index_select(0, old_rows)
            if runtime.lod_cache is not None:
                runtime.lod_cache = _map_batch_tensors(
                    runtime.lod_cache,
                    batch_size=int(runtime.indices.numel()),
                    transform=lambda tensor: tensor.index_select(
                        0, local_selection.to(tensor.device)
                    ),
                )
            runtime.indices = next_rows
            if runtime.valid_starts is not None:
                runtime.valid_starts = runtime.valid_starts.index_select(
                    0, local_selection.to(runtime.valid_starts.device)
                )
            self._attach_kernel_cache(runtime.engine, runtime.lod_cache)
            next_runtimes.append(runtime)
        self.runtimes = next_runtimes


def grouped_transient_attention(
    module: nn.Module,
    settings: HFLODSettings,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: HFLODPaddingPlan,
    *,
    scale: float | None,
) -> torch.Tensor:
    engine = getattr(module, "_hf_lod_transient_engine", None)
    if engine is None:
        engine = _build_engine(settings, query, key, scale=scale)
        module._hf_lod_transient_engine = engine
    output = torch.zeros_like(query)
    if settings.left_padding_mode == "chunk_aligned":
        plan = chunk_align_padding_plan(
            plan,
            chunk_size=settings.config.chunk_size,
            minimum_length=settings.config.local_window,
        )
    for group in plan.groups:
        indices = torch.tensor(group.rows, dtype=torch.long, device=query.device)
        begin = plan.padded_length - group.prompt_length
        group_query = query.index_select(0, indices)[..., begin:, :].contiguous()
        group_key = key.index_select(0, indices)[..., begin:, :].contiguous()
        group_value = value.index_select(0, indices)[..., begin:, :].contiguous()
        (
            group_query,
            group_key,
            group_value,
            logical_prefill_len,
        ) = _pad_kernel_prefill(settings, group_query, group_key, group_value)
        group_output, _ = _run_engine(
            engine,
            group_query,
            group_key,
            group_value,
            cache=None,
            use_cache=False,
            scale=scale,
            logical_prefill_len=logical_prefill_len,
            prefill_valid_starts=(
                torch.tensor(
                    group.valid_starts,
                    dtype=torch.long,
                    device=query.device,
                )
                if group.valid_starts
                else None
            ),
        )
        full_group_output = torch.zeros(
            (int(indices.numel()), *query.shape[1:]),
            dtype=query.dtype,
            device=query.device,
        )
        full_group_output[..., begin:, :].copy_(
            group_output[..., : group.prompt_length, :]
        )
        output.index_copy_(0, indices, full_group_output)
    return output


__all__ = [
    "GroupedHFLODRuntime",
    "HFLODPaddingPlan",
    "build_padding_plan",
    "chunk_align_padding_plan",
    "grouped_transient_attention",
]
