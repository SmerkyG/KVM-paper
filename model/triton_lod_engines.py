"""Fast, model-independent engines backed by the Triton LOD core.

These classes expose the same post-QKV/post-RoPE engine protocol as the clean
PyTorch LOD implementations while reusing the model-independent Triton LOD
core. They own no projections or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .pytorch_lod_attention import LODConfig
from .pytorch_lod_attention_paged import PagedLODConfig
from .triton_lod_attention import TritonLODAttentionCore


@dataclass
class KernelLODCache:
    """Opaque optimized-engine state with the cache property used by HF."""

    state: dict[str, object]

    @property
    def total_length(self) -> int:
        return int(self.state["total_len"])


class _KernelLODEngine(TritonLODAttentionCore):
    """Projection-free adapter around the established optimized LOD core."""

    def __init__(
        self,
        config: LODConfig,
        *,
        query_heads: int,
        key_value_heads: int,
        scale: float,
        default_open_count: int,
    ) -> None:
        nn.Module.__init__(self)
        if query_heads <= 0 or query_heads % key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if not 0 <= default_open_count <= config.max_routes:
            raise ValueError("open count must be between zero and max_routes")
        self.config = SimpleNamespace(
            num_attention_heads=query_heads,
            num_key_value_heads=key_value_heads,
        )
        self.num_key_value_groups = query_heads // key_value_heads
        self.scaling = float(scale)
        self.chunk_len = config.chunk_size
        self.local_len = config.local_window
        self.prefill_chunk_len = config.chunk_size
        self.prefill_local_len = config.local_window
        self.prefill_state_update_len = config.chunk_size
        self.state_growth_factor = config.state_growth_factor
        self.state_min_len = config.state_min_size
        self.sink_len = config.protected_prefix
        self.two_level_topk = default_open_count

    def reset_runtime_cache(self) -> None:
        if hasattr(self, "_lod_state"):
            del self._lod_state

    def _validate_geometry(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("kernel LOD Q/K/V tensors must be rank four")
        if int(query.size(1)) != self.config.num_attention_heads:
            raise ValueError("query head count differs from engine geometry")
        if int(key.size(1)) != self.config.num_key_value_heads:
            raise ValueError("KV head count differs from engine geometry")
        if key.shape[:3] != value.shape[:3] or query.size(2) != key.size(2):
            raise ValueError("kernel LOD Q/K/V sequence shapes differ")
        if query.size(-1) != key.size(-1):
            raise ValueError("query and key dimensions differ")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache: KernelLODCache | None = None,
        use_cache: bool = False,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, KernelLODCache | None]:
        """Run optimized causal prefill or incremental cached decode."""
        self._validate_geometry(query, key, value)
        if scale is not None:
            self.scaling = float(scale)
        if cache is None:
            self.reset_runtime_cache()
            output = self._prefill_attention(query, key, value)
        else:
            self._lod_state = cache.state
            outputs = []
            for token in range(int(query.size(2))):
                total_length = int(self._lod_state["total_len"]) + 1
                outputs.append(
                    self._decode_attention(
                        query[..., token : token + 1, :],
                        key[..., token : token + 1, :],
                        value[..., token : token + 1, :],
                        total_len=total_length,
                    )
                )
            output = torch.cat(outputs, dim=2)
        next_cache = KernelLODCache(self._lod_state) if use_cache else None
        if not use_cache:
            self.reset_runtime_cache()
        return output, next_cache


class KernelTwoLevelLODAttention(_KernelLODEngine):
    """Fast state/routing kernels plus all pages of every routed region."""

    def __init__(
        self,
        config: LODConfig | None = None,
        *,
        query_heads: int,
        key_value_heads: int,
        scale: float,
        default_open_count: int = 8,
    ) -> None:
        config = LODConfig() if config is None else config
        super().__init__(
            config,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            scale=scale,
            default_open_count=default_open_count,
        )
        self.leaf_attention_backend = "paged"
        self.leaf_page_size = 16
        self.virtual_page_storage = False
        self.recursive_page_lod = False


class KernelCoarseLODAttention(_KernelLODEngine):
    """Fast count-corrected state and exact local attention without leaves."""

    def __init__(
        self,
        config: LODConfig | None = None,
        *,
        query_heads: int,
        key_value_heads: int,
        scale: float,
    ) -> None:
        config = LODConfig() if config is None else config
        super().__init__(
            config,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            scale=scale,
            default_open_count=0,
        )
        self.leaf_attention_backend = "packed"
        self.virtual_page_storage = False
        self.recursive_page_lod = False

    def _drop_leaf_cache(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        owners = self._lod_state["owners"]
        if not isinstance(owners, torch.Tensor):
            raise TypeError("coarse LOD owner metadata is missing")
        self._lod_state["owners"] = owners[..., :0]
        self._lod_state["exact_k"] = key.new_empty(
            *key.shape[:2], 0, int(key.size(-1))
        )
        self._lod_state["exact_v"] = value.new_empty(
            *value.shape[:2], 0, int(value.size(-1))
        )

    def _prefill_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        output = super()._prefill_attention(query, key, value)
        # The common prefill path records these tensors for exact leaf opening.
        # Low-LOD attention never reads them, so retain typed empty sentinels.
        self._drop_leaf_cache(key, value)
        return output

    def _decode_attention(
        self,
        query: torch.Tensor,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
        *,
        total_len: int,
    ) -> torch.Tensor:
        output = super()._decode_attention(
            query, new_key, new_value, total_len=total_len
        )
        self._drop_leaf_cache(new_key, new_value)
        return output


class KernelRecursivePagedLODAttention(_KernelLODEngine):
    """Fast recursive one-page-per-region LOD with optional INT4 K/V."""

    def __init__(
        self,
        config: PagedLODConfig | None = None,
        *,
        query_heads: int,
        key_value_heads: int,
        scale: float,
        default_open_count: int = 8,
    ) -> None:
        config = PagedLODConfig() if config is None else config
        if config.page_size != 16:
            raise ValueError("the recursive page kernels require page_size=16")
        super().__init__(
            config,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            scale=scale,
            default_open_count=default_open_count,
        )
        self.leaf_attention_backend = "paged"
        self.leaf_page_size = config.page_size
        self.virtual_page_storage = True
        self.recursive_page_lod = True
        self.leaf_key_quant_bits = config.kv_bits
        self.leaf_value_quant_bits = config.kv_bits
        self.leaf_quant_group_size = config.quant_group_size


__all__ = [
    "KernelCoarseLODAttention",
    "KernelLODCache",
    "KernelRecursivePagedLODAttention",
    "KernelTwoLevelLODAttention",
]
