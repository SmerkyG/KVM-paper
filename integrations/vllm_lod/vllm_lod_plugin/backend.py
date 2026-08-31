"""Custom vLLM backend: native/direct prefill and recursive LOD decode."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType

if torch.version.hip:
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionBackend as _NativeBackend,
    )
    from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
        RocmAiterUnifiedAttentionImpl as _NativeImpl,
    )

else:
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend as _NativeBackend,
    )
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl as _NativeImpl,
    )


_NativeMetadataBuilder = _NativeBackend.get_builder_cls()


class LODAttentionMetadataBuilder(_NativeMetadataBuilder):
    """Expose the graph shapes whose state transitions LOD can replay.

    Ordinary one-token decode and uniform speculative target verification are
    captured.  The latter replays a fixed sequence of graph-safe one-token LOD
    transitions against device-resident row maps and recent lengths.  Rejected
    suffixes are truncated before the next replay, outside the graph.
    """

    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH


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
    @staticmethod
    def _uses_external_kv_cache(layer: torch.nn.Module) -> bool:
        return bool(getattr(layer, "_vllm_lod_external_kv_cache", False))

    @staticmethod
    def _uses_hybrid_native_kv_cache(layer: torch.nn.Module) -> bool:
        return bool(getattr(layer, "_vllm_lod_hybrid_native_kv", False))

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.lod_eligible:
            if self._uses_hybrid_native_kv_cache(layer):
                return super().do_kv_cache_update(
                    layer, key, value, kv_cache, slot_mapping
                )
            if not self._uses_external_kv_cache(layer):
                raise RuntimeError(
                    "eligible LOD attention was not externalized; refusing to "
                    "write a native chronological K/V cache"
                )
            return
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    def fused_rope_kvcache_supported(self) -> bool:
        # The platform fusion cannot independently suppress its chronological
        # cache write. Keep RoPE separate so authoritative LOD can skip that
        # redundant write while native-only layers retain the normal updater.
        return False if self.lod_eligible else super().fused_rope_kvcache_supported()

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
        hybrid_native = self._uses_hybrid_native_kv_cache(layer)
        if (
            self.lod_eligible
            and not self._uses_external_kv_cache(layer)
            and not hybrid_native
        ):
            raise RuntimeError(
                "eligible LOD attention was not externalized; native K/V "
                "fallback is unsupported"
            )
        if (
            pool is not None
            and self.lod_eligible
            and hybrid_native
            and bool(getattr(pool, "hybrid_full_decode", False))
            and attn_metadata is not None
        ):
            # Native K/V was updated immediately before this call. Delegate
            # both uniform target verification and the rare one-token fallback
            # to the unmodified AITER backend so the chronological cache stays
            # authoritative throughout this intentionally quadratic mode.
            return super().forward(
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
        if (
            os.getenv("VLLM_LOD_DIAGNOSTIC_EXTERNAL_EMPTY_ATTENTION")
            in ("skip", "eligible")
            and self.lod_eligible
            and self._uses_external_kv_cache(layer)
        ):
            # Benchmark-only control: exercise CUSTOM dispatch and externally
            # owned metadata with eligible attention arithmetic held at zero.
            # ``skip`` also zeros native layers through the plugin hook, while
            # ``eligible`` preserves them to isolate the unchanged local path.
            return output.zero_()
        if (
            pool is not None
            and int(getattr(pool, "speculative_decode_steps", 0)) > 1
            and self.lod_eligible
            and attn_metadata is not None
            and int(attn_metadata.max_query_len)
            == int(pool.speculative_decode_steps)
        ):
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(
                    "speculative LOD decode does not support fused output "
                    "quantization"
                )
            return pool.speculative_decode(query, key, value, output)
        if (
            pool is not None
            and self.lod_eligible
            and self._uses_external_kv_cache(layer)
            and getattr(pool, "direct_prefill_plan", None) is None
            and not pool.decode_enabled
        ):
            # Uncaptured/capture warmups have no logical request. Their output
            # is discarded. External layers intentionally have no native K/V.
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
        return result


class LODAttentionBackend(_NativeBackend):
    """Retain the platform-native cache layout and metadata builder."""

    forward_includes_kv_cache_update = False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # LOD handles wide global heads without entering ROCmAttention's
        # paged-attention kernel. Gemma-4 uses 512-wide heads only on those
        # global layers; its 256-wide sliding layers retain the native path.
        return sorted(set(super().get_supported_head_sizes()) | {512})

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[LODAttentionImpl]:
        return LODAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[LODAttentionMetadataBuilder]:
        return LODAttentionMetadataBuilder


__all__ = [
    "LODAttentionBackend",
    "LODAttentionImpl",
    "LODAttentionMetadataBuilder",
]
