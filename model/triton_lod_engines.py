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
        if config.state_clustering_policy != "manual":
            raise ValueError(
                "architecture-aware state clustering must be resolved by the "
                "Hugging Face installer before constructing a kernel engine"
            )
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
        self.state_size_offset = config.state_size_offset
        self.state_premerge_factor = config.state_premerge_factor
        self.state_split_max_leaves = config.state_split_max_leaves
        self.sink_len = config.protected_prefix
        self.state_clustering_normalization = (
            config.state_clustering_normalization
        )
        self.state_clustering_radial_bias = config.state_clustering_radial_bias
        self.state_clustering_radial_scope = config.state_clustering_radial_scope
        self.state_clustering_centroid_rescale = (
            config.state_clustering_centroid_rescale
        )
        self.state_clustering_centroid_rescale_scope = (
            config.state_clustering_centroid_rescale_scope
        )
        self.state_clustering_query_metric = config.state_clustering_query_metric
        self.state_clustering_rope_dim = config.state_clustering_rope_dim
        self.state_clustering_rope_fast_pairs = (
            config.state_clustering_rope_fast_pairs
        )
        self.coherence_single_matmul = config.coherence_single_matmul
        self.routing_normalization = config.routing_normalization
        self.routing_rope_dim = config.routing_rope_dim
        self.routing_rope_fast_pairs = config.routing_rope_fast_pairs
        self.routing_rope_jensen_pairs = config.routing_rope_jensen_pairs
        self.routing_rope_jensen = config.routing_rope_jensen
        self.routing_count_bias = config.routing_count_bias
        self.routing_variance_bias = config.routing_variance_bias
        self.routing_page_mass_candidates = config.routing_page_mass_candidates
        self.routing_leaf_mass_candidates = config.routing_leaf_mass_candidates
        self.routing_leaf_mass_objective = config.routing_leaf_mass_objective
        self.routing_leaf_mass_review_top_p = (
            config.routing_leaf_mass_review_top_p
        )
        self.routing_leaf_mass_top_p = config.routing_leaf_mass_top_p
        self.routing_leaf_mass_min_routes = config.routing_leaf_mass_min_routes
        self.mla_state_key_normalization = config.mla_state_key_normalization
        self.mla_recursive_page_key_normalization = (
            config.mla_recursive_page_key_normalization
        )
        self.leaf_paged_directory = config.leaf_paged_directory
        self.leaf_seal_capacity = config.leaf_seal_capacity
        self.prefill_int8_leaf_mma = config.prefill_int8_leaf_mma
        self.prefill_int8_coarse_mma = config.prefill_int8_coarse_mma
        self.prefill_int8_pv_mma = config.prefill_int8_pv_mma
        self.mla_key_norm_weight = None
        self.mla_key_norm_epsilon = 0.0
        self.collect_dynamic_open_stats = (
            config.routing_leaf_mass_review_top_p is not None
            or config.routing_leaf_mass_top_p is not None
        )
        self.two_level_topk = default_open_count

    def reset_runtime_cache(self) -> None:
        if hasattr(self, "_lod_state"):
            del self._lod_state

    @torch.inference_mode()
    def build_cache_from_bf16(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        clustering_query: torch.Tensor | None = None,
        logical_prefill_len: int | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
    ) -> KernelLODCache:
        """Convert an existing full-attention BF16 K/V prefix into LOD.

        K/V must already include the model's positional encoding.  INT4 is
        applied only after state routing has formed region-owned semantic
        pages; consecutive source-cache positions are never a quantization
        group merely because they share a physical cache block.
        """
        if key.dtype not in (torch.float16, torch.bfloat16) or value.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("full-cache conversion requires FP16 or BF16 K/V")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("full-cache conversion requires rank-four K/V")
        if key.shape[:3] != value.shape[:3]:
            raise ValueError("full-cache conversion K/V shapes differ")
        if int(key.size(1)) != self.config.num_key_value_heads:
            raise ValueError("full-cache KV head count differs from engine geometry")
        if self.leaf_attention_backend != "paged":
            raise NotImplementedError(
                "full-cache conversion currently requires region-paged LOD"
            )
        if clustering_query is not None:
            self._validate_geometry(clustering_query, key, value)
        self.reset_runtime_cache()
        state = self._build_cache_from_bf16(
            key,
            value,
            clustering_query=clustering_query,
            logical_prefill_len=logical_prefill_len,
            prefill_valid_starts=prefill_valid_starts,
        )
        page_cache = state.get("page_cache")
        if isinstance(page_cache, dict) and not bool(
            page_cache.get("region_owned_pages", False)
        ):
            raise AssertionError("converted INT4 cache lacks semantic page ownership")
        return KernelLODCache(state)

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
        logical_prefill_len: int | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
        output_buffer: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
    ) -> tuple[torch.Tensor, KernelLODCache | None]:
        """Run optimized causal prefill or incremental cached decode."""
        self._validate_geometry(query, key, value)
        if scale is not None:
            self.scaling = float(scale)
        if cache is None:
            self.reset_runtime_cache()
            output = self._prefill_attention(
                query,
                key,
                value,
                logical_prefill_len=logical_prefill_len,
                prefill_valid_starts=prefill_valid_starts,
                output_buffer=output_buffer,
                finalize_cache_for_decode=finalize_cache_for_decode,
            )
        else:
            if (
                logical_prefill_len is not None
                or prefill_valid_starts is not None
            ):
                raise ValueError(
                    "prefill length and padding metadata are valid only for "
                    "initial prefill"
                )
            self._lod_state = cache.state
            if (
                int(query.size(2)) > 1
                and self.split_prefill_local_attention
                and isinstance(self._lod_state.get("page_cache"), dict)
            ):
                output = self._cached_prefill_attention(
                    query,
                    key,
                    value,
                    output_buffer=output_buffer,
                    finalize_cache_for_decode=finalize_cache_for_decode,
                )
            else:
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
        if config.prefill_int8_leaf_mma:
            self.leaf_layout = "expert"
            self.leaf_num_warps = 2


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
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        logical_prefill_len: int | None = None,
        prefill_valid_starts: torch.Tensor | None = None,
        output_buffer: torch.Tensor | None = None,
        finalize_cache_for_decode: bool = True,
    ) -> torch.Tensor:
        output = super()._prefill_attention(
            query,
            key,
            value,
            logical_prefill_len=logical_prefill_len,
            prefill_valid_starts=prefill_valid_starts,
            output_buffer=output_buffer,
            finalize_cache_for_decode=finalize_cache_for_decode,
        )
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
    """Fast recursive one-page-per-region LOD with optional INT4/INT8 K/V."""

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
        self.page_summary_quant_bits = config.page_summary_quant_bits
        self.recursive_materialize_page_scores = (
            config.recursive_materialize_page_scores
        )
        self.recursive_page_score_block_n = config.recursive_page_score_block_n
        self.recursive_page_score_num_warps = (
            config.recursive_page_score_num_warps
        )
        self.recursive_page_select_block_n = config.recursive_page_select_block_n
        self.recursive_state_route_backend = config.recursive_state_route_backend
        # Amortize prefill routing and state maintenance without changing the
        # smaller decode-local field.  The extra exact lookback preserves three
        # decode chunks before each large causal prefill region.
        self.prefill_chunk_len = 16 * config.chunk_size
        self.prefill_local_len = (
            self.prefill_chunk_len + config.local_window + config.chunk_size
        )
        self.prefill_state_update_len = 5 * config.chunk_size
        self.prefill_two_level_topk = min(3, default_open_count)
        self.split_prefill_local_attention = True
        self.leaf_num_warps = 1
        self.recursive_page_attention_num_warps = self.leaf_num_warps
        self.prefill_route_block_m = 128
        self.prefill_route_num_warps = 8
        # Four-page scans amortize recursive page-selection loop overhead.
        self.recursive_page_block_n = 4
        self.coarse_route_block_m = 32
        self.coarse_route_block_n = 64
        # Two wavefronts keep the quality-safe 64-wide accumulation order but
        # avoid over-subscribing this memory-bound reduction on ROCm.
        self.coarse_route_num_warps = 2
        # Keep routing and coarse reduction separate. The fused path amplifies
        # small INT4 page-requantization differences across scheduler splits.
        self.fused_prefill_route_coarse = False
        self.leaf_key_quant_bits = config.kv_bits
        self.leaf_value_quant_bits = config.kv_bits
        self.leaf_quant_group_size = config.quant_group_size

    @torch.inference_mode()
    def catch_up_cache(
        self,
        cache: KernelLODCache,
        *,
        total_length: int,
        recent_length: int | None = None,
    ) -> None:
        """Archive old decode-local entries without running attention.

        Serving runtimes call this between graph replays.  Decode kernels can
        append K/V into fixed recent-cache rows while this method periodically
        advances the state and semantic page archive in chunk-sized batches.
        """
        state = cache.state
        page_cache = state.get("page_cache")
        if not isinstance(page_cache, dict):
            raise NotImplementedError("cache catch-up requires recursive pages")
        if self.state_clustering_query_metric != "none":
            raise NotImplementedError(
                "cache catch-up cannot reconstruct query-dependent clustering"
            )
        if total_length < int(state["total_len"]):
            raise ValueError("cache catch-up cannot move the logical length backward")

        state_k = state["state_k"]
        state_v = state["state_v"]
        counts = state["counts"]
        recent_k = state["recent_k"]
        recent_v = state["recent_v"]
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (state_k, state_v, counts, recent_k, recent_v)
        ):
            raise TypeError("LOD cache tensors are incomplete")
        key_norm_sums = state.get("key_norm_sums")
        if key_norm_sums is not None and not isinstance(
            key_norm_sums, torch.Tensor
        ):
            raise TypeError("LOD key-norm sum cache is invalid")

        state_len = int(state["state_len"])
        scheduled_state_len = int(state.get("scheduled_state_len", state_len))
        coverage = int(state["coverage"])
        if recent_length is None:
            recent_length = total_length - coverage
        if recent_length != total_length - coverage:
            raise ValueError("decode-local length does not match state coverage")
        if recent_length < 0 or recent_length > int(recent_k.size(2)):
            raise ValueError("decode-local length exceeds its fixed cache row")

        update_len = int(self.decode_state_update_len)
        exact_floor = self.local_len - self.chunk_len
        if update_len <= 0 or exact_floor < 0:
            raise ValueError("invalid decode state-update configuration")
        upcoming_length = total_length + 1
        target_coverage = max(min(total_length, self.chunk_len), coverage)
        pending_update = upcoming_length - target_coverage - exact_floor
        if pending_update > update_len:
            target_coverage += ((pending_update - 1) // update_len) * update_len
        target_coverage = min(target_coverage, total_length)

        if coverage < target_coverage:
            overflow_len = target_coverage - coverage
            if overflow_len > recent_length:
                raise AssertionError("LOD decode-local cache underflowed during catch-up")
            state_capacity = int(state["state_capacity"])
            update_ctx_len = exact_floor + target_coverage
            next_scheduled_state_len = (
                self._next_scheduled_state_len(
                    scheduled_state_len,
                    ctx_len=update_ctx_len,
                    available_context=target_coverage,
                    overflow_len=overflow_len,
                )
                if self.state_split_max_leaves is not None
                else scheduled_state_len
            )
            (
                state_k,
                state_v,
                counts,
                state_len,
                owners,
                old_slot_remap,
            ) = self._update_state(
                state_k,
                state_v,
                counts,
                key_norm_sums,
                recent_k[..., :overflow_len, :],
                recent_v[..., :overflow_len, :],
                state_len=state_len,
                ctx_len=update_ctx_len,
                available_context=target_coverage,
                state_capacity=state_capacity,
                clustering_query_scale=None,
                scheduled_state_len=scheduled_state_len,
            )
            scheduled_state_len = (
                next_scheduled_state_len
                if self.state_split_max_leaves is not None
                else state_len
            )
            if old_slot_remap is not None:
                raise AssertionError("paged state remapping is unsupported")
            self._append_page_cache(
                page_cache,
                recent_k[..., :overflow_len, :],
                recent_v[..., :overflow_len, :],
                owners,
            )
            remaining = recent_length - overflow_len
            if remaining:
                # Source and destination are overlapping views of the same
                # fixed recent-cache row.  PyTorch rejects the otherwise
                # memmove-like slice copy when a catch-up leaves a non-empty
                # suffix (notably after repeated speculative generations).
                # Catch-up runs only at the amortized state-update boundary,
                # so materialize the short exact suffix before shifting it.
                recent_k[..., :remaining, :].copy_(
                    recent_k[..., overflow_len:recent_length, :].clone()
                )
                recent_v[..., :remaining, :].copy_(
                    recent_v[..., overflow_len:recent_length, :].clone()
                )
            recent_length = remaining
            coverage = target_coverage

        state.update(
            state_k=state_k.detach(),
            state_v=state_v.detach(),
            counts=counts.detach(),
            state_len=state_len,
            scheduled_state_len=scheduled_state_len,
            coverage=coverage,
            recent_k=recent_k.detach(),
            recent_v=recent_v.detach(),
            recent_len=recent_length,
            total_len=total_length,
        )
        if key_norm_sums is not None:
            state["key_norm_sums"] = key_norm_sums.detach()


# Cache catch-up archives exact recent tokens through the common semantic-page
# updater and does not depend on recursive page selection. Serving uses the
# same graph-safe operation for flat two-tier caches.
KernelTwoLevelLODAttention.catch_up_cache = (  # type: ignore[attr-defined]
    KernelRecursivePagedLODAttention.catch_up_cache
)


__all__ = [
    "KernelCoarseLODAttention",
    "KernelLODCache",
    "KernelRecursivePagedLODAttention",
    "KernelTwoLevelLODAttention",
]
