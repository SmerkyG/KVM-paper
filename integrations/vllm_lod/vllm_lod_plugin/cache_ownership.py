"""vLLM cache-spec hooks for authoritative LOD storage."""

from __future__ import annotations

import logging
import math
import sys
from copy import deepcopy
from typing import Any

from .config import VLLMLODSettings

logger = logging.getLogger(__name__)


def _install_attention_spec_hook() -> None:
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.kv_cache_interface import (
        ChunkedLocalAttentionSpec,
        FullAttentionSpec,
    )

    if getattr(Attention, "_vllm_lod_cache_spec_installed", False):
        return
    original = Attention.get_kv_cache_spec

    def get_kv_cache_spec(self: Any, vllm_config: Any) -> Any:
        spec = original(self, vllm_config)
        settings = VLLMLODSettings.from_environment()
        impl = getattr(self, "impl", None)
        if (
            settings.cache_ownership != "lod"
            or not bool(getattr(impl, "lod_eligible", False))
            or not isinstance(spec, FullAttentionSpec)
        ):
            return spec
        if int(self.head_size) != int(self.head_size_v):
            raise NotImplementedError(
                "authoritative LOD cache requires equal key and value widths"
            )
        return ChunkedLocalAttentionSpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode,
            page_size_padded=spec.page_size_padded,
            indexes_kv_by_block_stride=spec.indexes_kv_by_block_stride,
            attention_chunk_size=settings.native_staging_chunk,
        )

    Attention.get_kv_cache_spec = get_kv_cache_spec
    Attention._vllm_lod_cache_spec_installed = True


def _native_block_cap(vllm_config: Any, kv_cache_specs: list[dict[str, Any]]) -> int:
    """Size native staging for active requests, not maximum context length."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
    from vllm.v1.kv_cache_interface import (
        ChunkedLocalAttentionSpec,
        UniformTypeKVCacheSpecs,
    )

    merged: dict[str, Any] = {}
    for worker_specs in kv_cache_specs:
        merged.update(worker_specs)
    groups = get_kv_cache_groups(vllm_config, deepcopy(merged))
    settings = VLLMLODSettings.from_environment()
    active = int(vllm_config.scheduler_config.max_num_seqs)
    in_flight = int(vllm_config.max_in_flight_tokens)
    required = 0
    for group in groups:
        group_spec = group.kv_cache_spec
        specs = (
            tuple(group_spec.kv_cache_specs.values())
            if isinstance(group_spec, UniformTypeKVCacheSpecs)
            else (group_spec,)
        )
        if all(isinstance(spec, ChunkedLocalAttentionSpec) for spec in specs):
            # max_in_flight_tokens is a global batch budget. Charging it once
            # per request over-reserves by max_num_seqs. Retained windows are
            # per request; newly scheduled tokens are global. One extra block
            # per request covers partial-block fragmentation between the two.
            block_size = int(group_spec.block_size)
            retained = active * max(
                math.ceil(int(spec.attention_chunk_size) / block_size)
                for spec in specs
            )
            required += retained + math.ceil(in_flight / block_size) + active
        else:
            blocks_per_request = math.ceil(
                group_spec.max_memory_usage_bytes(vllm_config)
                / group_spec.page_size_bytes
            )
            required += active * blocks_per_request
    return max(
        2,
        math.ceil(required * settings.native_cache_headroom) + 1,
    )


def _install_native_pool_cap_hook() -> None:
    import vllm.v1.core.kv_cache_utils as cache_utils

    if getattr(cache_utils, "_vllm_lod_native_cap_installed", False):
        return
    original = cache_utils.get_kv_cache_configs

    def get_kv_cache_configs(
        vllm_config: Any,
        kv_cache_specs: list[dict[str, Any]],
        available_memory: list[int],
    ) -> Any:
        from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec

        settings = VLLMLODSettings.from_environment()
        cache_config = vllm_config.cache_config
        if (
            settings.cache_ownership != "lod"
            or cache_config.num_gpu_blocks_override is not None
            or not any(
                isinstance(spec, ChunkedLocalAttentionSpec)
                for worker_specs in kv_cache_specs
                for spec in worker_specs.values()
            )
        ):
            return original(vllm_config, kv_cache_specs, available_memory)
        cap = _native_block_cap(vllm_config, kv_cache_specs)
        cache_config.num_gpu_blocks_override = cap
        logger.info(
            "Authoritative LOD cache caps native staging at %d blocks "
            "(headroom %.2fx)",
            cap,
            settings.native_cache_headroom,
        )
        try:
            return original(vllm_config, kv_cache_specs, available_memory)
        finally:
            cache_config.num_gpu_blocks_override = None

    cache_utils.get_kv_cache_configs = get_kv_cache_configs
    engine_core = sys.modules.get("vllm.v1.engine.core")
    if engine_core is not None:
        engine_core.get_kv_cache_configs = get_kv_cache_configs
    cache_utils._vllm_lod_native_cap_installed = True


def install_cache_ownership_hooks() -> None:
    settings = VLLMLODSettings.from_environment()
    if settings.cache_ownership != "lod":
        return
    _install_attention_spec_hook()
    _install_native_pool_cap_hook()


__all__ = ["install_cache_ownership_hooks"]
