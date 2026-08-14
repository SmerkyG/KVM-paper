"""Custom vLLM backend: native/direct prefill and recursive LOD decode."""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.attention.backend import AttentionType

if torch.version.hip:
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionBackend as _NativeBackend,
    )
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionImpl as _NativeImpl,
    )

    NATIVE_LAYOUT = "rocm"
else:
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend as _NativeBackend,
    )
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl as _NativeImpl,
    )

    NATIVE_LAYOUT = "flash"


class LODAttentionImpl(_NativeImpl):
    """Delegate native attention except for eligible one-token decode batches."""

    supports_dcp = False
    supports_pcp = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise NotImplementedError(
                f"this vLLM release supplies unsupported attention options: {names}"
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self.lod_eligible = (
            attn_type == AttentionType.DECODER
            and sliding_window is None
            and alibi_slopes is None
            and sinks is None
            and not logits_soft_cap
            and kv_cache_dtype in ("auto", "float16", "bfloat16")
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pool = getattr(layer, "_vllm_lod_pool", None)
        if (
            pool is not None
            and getattr(pool, "direct_prefill_plan", None) is not None
            and self.lod_eligible
            and attn_metadata is not None
        ):
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(
                    "direct LOD prefill does not support fused output quantization"
                )
            return pool.direct_prefill(query, key, value, output)
        if (
            pool is not None
            and pool.decode_enabled
            and self.lod_eligible
            and attn_metadata is not None
            and int(attn_metadata.max_query_len) == 1
        ):
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(
                    "LOD decode does not support fused output quantization"
                )
            return pool.decode(query, key, value, attn_metadata, output)
        result = super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
        if pool is not None:
            pool.record_native_appends(key, value)
        return result


class LODAttentionBackend(_NativeBackend):
    """Retain the platform-native cache layout and metadata builder."""

    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[LODAttentionImpl]:
        return LODAttentionImpl


__all__ = ["NATIVE_LAYOUT", "LODAttentionBackend", "LODAttentionImpl"]
