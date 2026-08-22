"""Custom vLLM backend: native/direct prefill and recursive LOD decode."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.v1.attention.backend import AttentionType

from .config import VLLMLODSettings

if torch.version.hip:
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend as _AiterNativeBackend,
    )
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionImpl as _NativeImpl,
    )
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionBackend as _NativeBackend,
    )
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionImpl as _TritonNativeImpl,
    )
    from vllm.v1.attention.ops.triton_unified_attention import (
        unified_attention as _triton_unified_attention,
    )

    if bool(int(os.getenv("VLLM_LOD_ROCM_PACKED_CACHE", "0"))):
        _NativeBackend = _AiterNativeBackend
        NATIVE_LAYOUT = "flash"
    else:
        NATIVE_LAYOUT = "rocm"
else:
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend as _NativeBackend,
    )
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl as _NativeImpl,
    )

    NATIVE_LAYOUT = "flash"


if torch.version.hip:

    class _LegacyLayoutTritonDecodeImpl:
        """Use vLLM's Triton decoder on the ROCm K/V planes."""

        def __init__(self, fallback: Any) -> None:
            self.fallback = fallback

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
            if attn_metadata is None or int(attn_metadata.max_query_len) != 1:
                return self.fallback.forward(
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
            if output_block_scale is not None:
                raise NotImplementedError(
                    "Triton local decode does not support block-scaled output"
                )

            num_actual_tokens = attn_metadata.num_actual_tokens
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            _triton_unified_attention(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.fallback.scale,
                causal=attn_metadata.causal,
                window_size=self.fallback.sliding_window,
                block_table=attn_metadata.block_table,
                softcap=self.fallback.logits_soft_cap,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                alibi_slopes=self.fallback.alibi_slopes,
                sinks=self.fallback.sinks,
                output_scale=output_scale,
                mm_prefix_clamp_sliding_window=getattr(
                    layer, "mm_prefix_clamp_sliding_window", False
                ),
            )
            return output


class LODAttentionImpl(_NativeImpl):
    """Delegate native attention except for eligible one-token decode batches."""

    supports_dcp = False
    supports_pcp = False

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if NATIVE_LAYOUT == "flash":
            return super()._split_kv_cache(kv_cache)
        return kv_cache[0], kv_cache[1]

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
        self.lod_authoritative = (
            VLLMLODSettings.from_environment().cache_ownership == "lod"
        )
        # Gemma-4 does not expose its local/global distinction through the
        # backend constructor on every vLLM path. AITER's unified capture
        # faults for its 256-wide local heads, so prepare the ROCm Triton
        # implementation for every 256-wide layer. LOD-owned global layers
        # still take the direct prefill/decode branches below; only native
        # fallthrough (the local layers) uses this implementation.
        self._triton_swa = None
        if torch.version.hip and head_size == 256:
            fallback = _TritonNativeImpl(
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
            self._triton_swa = _LegacyLayoutTritonDecodeImpl(fallback)

    @staticmethod
    def _uses_authoritative_lod(layer: torch.nn.Module) -> bool:
        pool = getattr(layer, "_vllm_lod_pool", None)
        return (
            pool is not None
            and pool.settings.cache_ownership == "lod"
            and (
                getattr(pool, "direct_prefill_plan", None) is not None
                or bool(getattr(pool, "decode_enabled", False))
            )
        )

    @staticmethod
    def _uses_placeholder_cache(layer: torch.nn.Module) -> bool:
        return bool(getattr(layer, "_vllm_lod_native_placeholder_cache", False))

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.lod_eligible and (
            self._uses_authoritative_lod(layer)
            or self._uses_placeholder_cache(layer)
        ):
            return
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def fused_rope_kvcache_supported(self) -> bool:
        return (
            False
            if self.lod_eligible and self.lod_authoritative
            else super().fused_rope_kvcache_supported()
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
            self.lod_eligible
            and self._uses_placeholder_cache(layer)
            and (
                pool is None
                or (
                    getattr(pool, "direct_prefill_plan", None) is None
                    and not pool.decode_enabled
                )
            )
        ):
            # Warmup/capture rows have no logical request and their output is
            # discarded. The scheduler-visible placeholder is intentionally
            # too small for native remote attention.
            return output.zero_()
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
        native = self._triton_swa
        if native is None:
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
        else:
            result = native.forward(
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
    """Use ROCm's cache layout with AITER attention math where supported."""

    forward_includes_kv_cache_update = False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Gemma-4's full-attention layers use 512-wide global heads. Those
        # layers enter LOD rather than ROCm paged attention; the model's
        # 256-wide sliding layers continue through the native implementation.
        return sorted(set(super().get_supported_head_sizes()) | {512})

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[LODAttentionImpl]:
        return LODAttentionImpl


__all__ = ["NATIVE_LAYOUT", "LODAttentionBackend", "LODAttentionImpl"]
