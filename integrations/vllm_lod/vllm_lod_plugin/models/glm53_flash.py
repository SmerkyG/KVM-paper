"""Experimental GLM5.3-Flash NoPE MLA port, reusing the Kimi cache/runtime.

K=V is the 512-channel latent itself. Decode queries are absorbed through
W_UK; prefill projects just centroid summaries to K256/V256 in one GEMM.
Exact leaves still use absorbed queries and project their reduced result.
No synthetic
direct-key channels are stored. An experimental native-prefix option uses
GLM's learned-sparse attention for just the first scheduler chunk (at most
16K), while constructing the LoD cache. Later chunks and decode use LoD only.
Supports two-tier BF16 and three-tier BF16/INT4 latent caches.
"""

from __future__ import annotations

import os
import copy
import inspect
import weakref
from typing import Any

import torch


def install_flydsl_shift_compat():
    """Normalize Python shift constants for this image's older FlyDSL ABI.

    AITER's gfx942 native FP8 indexer calls Numeric.shrui(32); this FlyDSL
    release forwards the int to an MLIR-value-only helper. Typed constants
    emit the same shift instruction. Do not edit or replace the indexer math.
    """
    try:
        from flydsl.expr.numeric import Numeric
    except ImportError:
        return
    original = Numeric.shrui
    if getattr(original, "_lod_typed_shift", False):
        return
    if "self.ir_value().shrui(amount)" not in inspect.getsource(original):
        return  # A newer library already normalizes this operand.

    def shrui(self, amount):
        if isinstance(amount, int):
            amount = type(self)(amount)
        return original(self, amount)

    shrui._lod_typed_shift = True
    Numeric.shrui = shrui


def _tile_safe_indexer_call(original, q, cache, weights, lengths, blocks,
                            schedule, max_model_len, **kwargs):
    # This gfx942 AITER kernel stores its last 128-column half-tile without
    # an upper output bound. Reserve a whole 256-column tile; return only the
    # original logical columns. Valid lengths, ranking and cache stay intact.
    physical_width = (max_model_len + 255) // 256 * 256
    result = original(q, cache, weights, lengths, blocks, schedule,
                      physical_width, **kwargs)
    return result[:, :max_model_len]


def install_native_indexer_compat():
    """Repair this image's storage-page metadata and gfx942 output workspace.

    A KDA-aligned slab may be 4,352 tokens, while the GLM indexer stores
    256-token virtual pages (64 pooled keys). Undoing the slab expansion
    instead of the virtual-page expansion leaves a one-entry block table for
    1,024 pooled keys. The kernel then reads beyond that table. Only GLM's
    indexer backend requests small virtual kernel pages, letting vLLM's
    existing metadata builder reconstruct the physical addressing correctly.
    """
    try:
        # Do not wrap the shared indexer on older Qwen/K2 vLLM installations
        # that do not even include this optional model family.
        from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
        from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
        from vllm.v1.attention.ops import rocm_aiter_mla_sparse as ops
    except ImportError:
        return
    cls = DeepseekV32IndexerBackend
    if not getattr(cls, "_lod_glm_physical_pages", False):
        original_blocks = cls.get_supported_kernel_block_sizes

        def supported_blocks(kv_cache_spec=None):
            from vllm.config import get_current_vllm_config
            config = get_current_vllm_config()
            text = config.model_config.hf_text_config
            if text.model_type == "glm5_next_text":
                # Force the framework to enumerate small virtual pages. Its
                # normal compressed-indexer builder then reconstructs 32/64
                # pooled-key storage pages, rather than treating a full KDA
                # slab as a single such page. Both runners already implement
                # this expansion, including stable graph-capture buffers.
                return [int(text.index_kpool) * 32]
            return original_blocks(kv_cache_spec)

        cls.get_supported_kernel_block_sizes = staticmethod(supported_blocks)
        cls._lod_glm_physical_pages = True
    original = ops.rocm_fp8_paged_mqa_logits
    if getattr(ops, "_ON_GFX942", False) and not getattr(original, "_lod_tile_safe", False):
        def logits(q, cache, weights, lengths, blocks, schedule, max_model_len, **kwargs):
            return _tile_safe_indexer_call(original, q, cache, weights, lengths,
                                           blocks, schedule, max_model_len, **kwargs)
        logits._lod_tile_safe = True
        ops.rocm_fp8_paged_mqa_logits = logits


def _glm_lod_config(config: Any) -> bool:
    from ..config import lod_enabled

    text = config.model_config.hf_text_config
    backend = config.attention_config.backend
    name = str(getattr(backend, "name", backend)).upper()
    return (lod_enabled() and name == "CUSTOM"
            and str(getattr(text, "model_type", "")) == "glm5_next_text")


def _native_prefix_plan(layer):
    """Only initial, uncached chunks qualify; never reuse native remote mass."""
    pool = getattr(layer, "_vllm_lod_pool", None)
    plan = getattr(pool, "direct_prefill_plan", None)
    if plan and all(previous == 0 and 0 < end - begin <= 16384
                    for _, begin, end, previous in plan):
        return plan
    return None


def _native_prefix_attention(layer, plan, q, k, v, *, causal,
                             valid_starts=None, output_buffer=None):
    """Native sparse MLA on current latents, with no second history cache.

    The indexer emits request-local positions. LoD packs equal-length rows;
    process each row separately so those indices never address another row.
    The surrounding LoD prefill code still builds/stages its usual cache.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        fit_kpool_indices_to_aiter,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense, rocm_sparse_attn_prefill,
    )

    if not causal or valid_starts is not None:
        raise ValueError("native GLM prefix requires unpadded causal initial rows")
    length = q.size(2)
    rows = [item for item in plan if item[2] - item[1] == length]
    if len(rows) != q.size(0) or k.size(2) != length:
        raise ValueError("native GLM prefix does not match the LoD initial batch")
    output = output_buffer if output_buffer is not None else torch.empty_like(q)
    for row, (_, begin, end, _) in enumerate(rows):
        indices = fit_kpool_indices_to_aiter(
            layer.indexer.topk_indices_buffer[begin:end], layer.indexer.topk_tokens)
        lengths = (indices >= 0).sum(-1, dtype=torch.int32)
        ragged, indptr = build_ragged_indices_from_dense(indices, lengths, num_rows=length)
        rocm_sparse_attn_prefill(
            q=q[row].permute(1, 0, 2), kv=k[row].permute(1, 0, 2),
            indices=None, topk_length=None, scale=layer.scale,
            head_dim=512, nope_head_dim=512, rope_head_dim=0,
            attn_sink=getattr(layer.impl, "sinks", None),
            output=output[row].permute(1, 0, 2),
            ragged_indices=ragged, ragged_indptr=indptr)
    layer._vllm_lod_native_prefix_calls = getattr(layer, "_vllm_lod_native_prefix_calls", 0) + 1
    layer._vllm_lod_native_prefix_tokens = getattr(layer, "_vllm_lod_native_prefix_tokens", 0) + int(q.size(0) * length)
    return output


def latent_attention(layer, query, latent, direct_key, output_shape=None,
                     q_dcp_replicated=None):
    """Keep one latent record; projected prefill does not need a second V up-projection."""
    from .kimi_k3 import absorb_query

    pool = layer._vllm_lod_pool
    if pool.dcp_world_size != 1 or q_dcp_replicated is not None:
        raise NotImplementedError("GLM fixture port currently requires DCP1")
    if latent.size(-1) != 512 or direct_key.size(-1) != 0:
        raise ValueError("GLM5.3 MLA requires 512 latent channels and no direct key")
    record = latent.unsqueeze(1)
    projected_prefill = (pool.direct_prefill_plan is not None
                         and not getattr(layer, "_vllm_lod_absorbed_coarse", False))
    # The exact front and fully projected refinement need no whole-chunk
    # Q256 -> Q512 BMM. Use Kimi's shape-only carrier and lazily absorb only
    # one-token decode rows if vLLM schedules them beside a prefill chunk.
    engine = getattr(pool, "engine", None)
    exact_front = bool(projected_prefill and getattr(engine, "prefill_exact_first_chunk", False)
        and all(previous == 0 and end - begin <= engine.prefill_chunk_len
                for _, begin, end, previous in pool.direct_prefill_plan))
    defer_absorption = (projected_prefill
        and getattr(engine, "_lod_glm_project_local", False)
        and not getattr(layer, "_vllm_lod_native_prefix", False)) and (
        exact_front or os.environ.get("LOD_GLM_PROJECTED_LEAVES") == "1")
    absorbed = (record.expand(-1, query.size(1), -1) if defer_absorption else
                absorb_query(query, layer.W_UK_T, nope_dim=256))
    if defer_absorption:
        layer._vllm_lod_deferred_query_tokens = getattr(
            layer, "_vllm_lod_deferred_query_tokens", 0) + query.size(0)
    if projected_prefill:
        # Kimi's local/coarse streams wait on the foreground before starting.
        # Create their shared immutable layout HERE, not on either side
        # stream: publishing a freshly packed weight from the local stream
        # would let coarse read it before that packing has finished.
        buffers = getattr(getattr(pool, "engine", None), "_lod_prefill_attention_buffers", None)
        if buffers is not None:
            from lod_attention._mla_projection import combined_kv_weight
            combined_kv_weight(layer.W_UK_T, layer.W_UV, buffers)
    # NoPE absorption returns a head-major BMM view. Do not inherit its
    # strides for attention scratch: optimized reducers use batch/head rows.
    carrier = query if projected_prefill else absorbed
    attention_output = torch.empty(carrier.shape, dtype=carrier.dtype, device=carrier.device)
    if pool.direct_prefill_plan is not None:
        plan = _native_prefix_plan(layer) if getattr(layer, "_vllm_lod_native_prefix", False) else None
        if plan is not None:
            # A per-call hook, not a different cache/update implementation.
            def initial_attention(*args, **kwargs):
                return _native_prefix_attention(layer, plan, *args, **kwargs)
            pool.engine._lod_initial_attention = initial_attention
        try:
            pool.direct_prefill(absorbed, record, record, attention_output,
                **(dict(mla_query=query, mla_w_uk_t=layer.W_UK_T, mla_w_uv=layer.W_UV)
                   if projected_prefill else {}),
                **(dict(defer_mla_query_absorption=True) if defer_absorption else {}))
        finally:
            if plan is not None:
                del pool.engine._lod_initial_attention
    elif pool.decode_enabled and query.size(0) <= pool.max_requests:
        metadata = type("GLMMetadata", (), {"num_actual_tokens": query.size(0)})()
        pool.decode(absorbed, record, record, metadata, attention_output)
    else:
        # Profiling/capture dummy calls do not own a logical request row.
        attention_output.zero_()
    shape = output_shape or (query.size(0), layer.num_heads * layer.v_head_dim)
    if projected_prefill:
        return attention_output.reshape(shape)
    output = query.new_empty(shape)
    layer._v_up_proj(attention_output, output)
    return output


def register_glm53_flash_lod():
    """Retain the native indexer for the first chunk; bypass it afterward."""
    try:
        from vllm.models.glm5next.common.attention import Glm5NextMLAAttention, Indexer
    except ImportError:
        return  # Older Qwen/K2 vLLM builds do not contain this optional family.
    import vllm.model_executor.layers.attention.mla_attention as mla
    from vllm.config import get_current_vllm_config
    from ..config import VLLMLODSettings

    if getattr(mla.MLAAttention, "_vllm_lod_glm53_installed", False):
        return
    from .glm53_cache import install_native_prefix_cache_groups
    install_native_prefix_cache_groups()
    outer_init = Glm5NextMLAAttention.__init__
    signature = inspect.signature(outer_init)
    inner_init = mla.MLAAttention.__init__
    inner_forward = mla.MLAAttention.forward
    indexer_forward = Indexer.forward

    def initialize_outer(self, *args, **kwargs):
        if not _glm_lod_config(get_current_vllm_config()):
            return outer_init(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        config = bound.arguments["config"]
        if (config.kv_lora_rank, config.qk_nope_head_dim,
            config.qk_rope_head_dim, config.v_head_dim) != (512, 256, 0, 256):
            raise ValueError("GLM5.3 requires MLA geometry L512/K256/direct0/V256")
        if not getattr(config, "lod_native_prefix", False):
            # The optimized exact front remains the default; it does not
            # need native indexer weights or cache allocations at all.
            config = copy.copy(config)
            config.index_topk = None
            bound.arguments["config"] = config
            bound.arguments["topk_indices_buffer"] = None
        return outer_init(*bound.args, **bound.kwargs)

    def initialize_inner(self, *args, **kwargs):
        enabled = _glm_lod_config(get_current_vllm_config())
        native_prefix = enabled and bool(getattr(
            get_current_vllm_config().model_config.hf_text_config,
            "lod_native_prefix", False))
        if native_prefix:
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import ROCMAiterMLASparseBackend
            # Native indexer/weight initialization, but no native main latent
            # cache: the metadata-only spec hook still gives ownership to LoD.
            kwargs["attn_backend"] = ROCMAiterMLASparseBackend
        inner_init(self, *args, **kwargs)
        if enabled:
            self._vllm_lod_absorbed_mla = True
            self._vllm_lod_glm53 = True
            self._vllm_lod_native_prefix = native_prefix
            if native_prefix:
                self.indexer._vllm_lod_prefix_layer = weakref.ref(self)
            self.is_aiter_triton_fp8_bmm_enabled = False
            self.is_aiter_triton_fp4_bmm_enabled = False

    def index(self, *args, **kwargs):
        reference = getattr(self, "_vllm_lod_prefix_layer", None)
        if reference is not None:
            layer = reference()
            if (not getattr(layer, "_vllm_lod_native_prefix", False)
                    or _native_prefix_plan(layer) is None):
                # No projection, cache write or selection kernel after chunk one.
                return None
        return indexer_forward(self, *args, **kwargs)

    def forward(self, q, kv_c_normed, k_pe, output_shape=None,
                q_dcp_replicated=None):
        if getattr(self, "_vllm_lod_glm53", False) and hasattr(self, "_vllm_lod_pool"):
            return latent_attention(self, q, kv_c_normed, k_pe, output_shape,
                                    q_dcp_replicated)
        return inner_forward(self, q, kv_c_normed, k_pe,
                             output_shape=output_shape,
                             q_dcp_replicated=q_dcp_replicated)

    Glm5NextMLAAttention.__init__ = initialize_outer
    Indexer.forward = index
    mla.MLAAttention.__init__ = initialize_inner
    mla.MLAAttention.forward = forward
    mla.MLAAttention._vllm_lod_glm53_installed = True
