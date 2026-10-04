"""Public vLLM configuration for the LoD paper release."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from lod_attention._config import LODMode, ModelFamily, PREFILL_CHUNK_SIZE

_PUBLIC_ENV = {
    "VLLM_LOD_ENABLED",
    "VLLM_LOD_MODE",
    "VLLM_LOD_POOL_SIZE",
    "VLLM_LOD_MAX_CONTEXT",
}


def lod_enabled() -> bool:
    raw = os.getenv("VLLM_LOD_ENABLED", "1")
    if raw not in ("0", "1"):
        raise ValueError("VLLM_LOD_ENABLED must be '0' or '1'")
    return raw == "1"
_REMOVED_ENV = {
    "LOD_DECODE_TOP8",
    "LOD_QWEN_EXPERIMENT",
}
LOD_SCHEDULER = "vllm_lod_plugin.scheduler.LODChunkAlignedScheduler"


def _ensure_exact_lod_decode_capture_sizes(vllm_config: Any) -> None:
    """Prevent graph padding from aliasing another request's LoD cache row."""
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
    if (
        os.environ.get("LOD_KIMI_GRAPH_PREFILL") == "1"
        and os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1"
    ):
        # These are explicit piecewise-prefill shapes, not padded decode rows.
        exact_sizes.update(int(size) for size in capture_sizes
                           if int(size) > exact_rows)
    speculative_steps = int(vllm_config.num_speculative_tokens or 0) + 1
    if speculative_steps > 1:
        exact_sizes.update(
            rows * speculative_steps
            for rows in range(1, exact_rows + 1)
            if rows * speculative_steps <= original_max
        )
    compilation.cudagraph_capture_sizes = sorted(exact_sizes)
    compilation.max_cudagraph_capture_size = max(exact_sizes)


def _positive_integer(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _reject_removed_options() -> None:
    removed = sorted(
        name
        for name in os.environ
        if (
            name in _REMOVED_ENV
            or name.startswith(("VLLM_LOD_", "LOD_DEV_"))
            and name not in _PUBLIC_ENV
        )
    )
    if removed:
        raise ValueError(
            "The LoD paper release has no tuning flags. Remove: "
            + ", ".join(removed)
        )


def validate_production_scheduler(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    long_prefill_token_threshold: int,
    required_prefill: int = PREFILL_CHUNK_SIZE,
    required_decode_reserve: int = 0,
    scheduler_cls: object = None,
) -> None:
    required = min(required_prefill, max_model_len)
    required_budget = required + required_decode_reserve
    if max_num_batched_tokens < required_budget or (
        0 < long_prefill_token_threshold < required
    ):
        raise RuntimeError(
            "LoD requires --max-num-batched-tokens >= "
            f"{required_budget} ({required} prefill + "
            f"{required_decode_reserve} decode reserve) and "
            "--long-prefill-token-threshold 0 "
            f"(or >= {required})."
        )
    scheduler_name = (
        scheduler_cls
        if isinstance(scheduler_cls, str)
        else (
            f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}"
            if isinstance(scheduler_cls, type)
            else None
        )
    )
    if scheduler_name != LOD_SCHEDULER:
        raise RuntimeError(
            "LoD requires --scheduler-cls " + LOD_SCHEDULER + "."
        )


@dataclass(frozen=True)
class VLLMLODSettings:
    """Three public choices plus family-specific launch geometry.

    Model-family differences below change only kernel tiling and dispatch. The
    attention calculation is top-eight in both prefill and decode for every
    supported model.
    """

    mode: LODMode = LODMode.TWO_TIER
    pool_size: int = 8
    request_capacity: int | None = None
    family: ModelFamily | None = None

    @property
    def levels(self) -> int:
        return self.mode.levels

    @property
    def kv_bits(self) -> int:
        return self.mode.kv_bits

    @property
    def resolved_key_bits(self) -> int:
        return self.kv_bits

    @property
    def resolved_value_bits(self) -> int:
        return self.kv_bits

    @property
    def quant_group_size(self) -> int:
        return 4 if self.kv_bits == 4 else 32

    @property
    def prefill_chunk_size(self) -> int:
        return PREFILL_CHUNK_SIZE

    def _is_qwen38(self) -> bool:
        if self.family is None:
            raise RuntimeError("LoD model family has not been resolved")
        return self.family is ModelFamily.QWEN38

    @property
    def decode_gqa_fixed_mask_aiter(self) -> bool:
        if self.family is None:
            raise RuntimeError("LoD model family has not been resolved")
        return (
            self.family in (ModelFamily.QWEN38, ModelFamily.KIMI_K3)
            and self.levels == 2
        )

    @property
    def decode_gqa_fixed_mask_segments(self) -> int:
        return 256 if self._is_qwen38() else 128

    @property
    def decode_gqa_fixed_mask_reduce_block_d(self) -> int:
        return 64 if self._is_qwen38() else 0

    @property
    def decode_gqa_fixed_mask_scan_num_warps(self) -> int:
        return 2 if self._is_qwen38() else 1

    @classmethod
    def production(
        cls,
        *,
        mode: str | LODMode = LODMode.TWO_TIER,
        pool_size: int = 8,
        request_capacity: int | None = None,
    ) -> VLLMLODSettings:
        return cls(
            mode=LODMode.parse(mode),
            pool_size=pool_size,
            request_capacity=request_capacity,
        )

    def for_family(self, family: ModelFamily) -> VLLMLODSettings:
        if family not in (
            ModelFamily.QWEN38,
            ModelFamily.K2,
            ModelFamily.KIMI_K3,
        ):
            raise ValueError(f"unsupported LoD model family: {family}")
        return replace(self, family=family)

    @classmethod
    def from_environment(cls) -> VLLMLODSettings:
        _reject_removed_options()
        lod_enabled()
        raw_capacity = os.getenv("VLLM_LOD_MAX_CONTEXT", "0")
        try:
            capacity = int(raw_capacity)
        except ValueError as exc:
            raise ValueError(
                "VLLM_LOD_MAX_CONTEXT must be an integer, "
                f"got {raw_capacity!r}"
            ) from exc
        if capacity < 0:
            raise ValueError("VLLM_LOD_MAX_CONTEXT cannot be negative")
        return cls.production(
            mode=os.getenv("VLLM_LOD_MODE", LODMode.TWO_TIER.value),
            pool_size=_positive_integer("VLLM_LOD_POOL_SIZE", 8),
            request_capacity=capacity or None,
        )


__all__ = [
    "LOD_SCHEDULER",
    "VLLMLODSettings",
    "lod_enabled",
    "validate_production_scheduler",
]
