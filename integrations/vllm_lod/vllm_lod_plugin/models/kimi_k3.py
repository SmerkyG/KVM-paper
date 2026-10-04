"""Absorbed-MQA LoD adapter for Kimi K3's MLA layers.

Kimi stores one normalized latent ``c`` plus one direct, non-rotated key
channel ``k_direct`` per token.  Its ordinary decode path expands keys and
values as::

    k_h = [W_UK_h c, k_direct]
    v_h = W_UV_h c

Linearity lets LoD operate before either expansion.  We absorb each query
through ``W_UK.T``, route over ``[c, k_direct]``, keep the attention output in
latent space, and apply ``W_UV`` once at the end.  The cache stores that one
combined record; its ``c`` prefix is also the value.  There is no separate V
allocation.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _install_attention_only_fixture() -> None:
    """Let the K3 benchmark fixture omit its feed-forward sublayers.

    The fixture keeps a stack of real K3 MLA modules so that cross-layer LoD
    construction, routing, collectives, and launch occupancy can be measured
    without loading or executing K3's much larger MoE.  This is deliberately
    selected by a private model-config field rather than an environment flag;
    ordinary checkpoints cannot enter the path accidentally.
    """

    try:
        import vllm.models.kimi_k3.amd.linear as kimi_linear
    except ImportError:
        return

    decoder_layer = kimi_linear.KimiDecoderLayer
    if getattr(decoder_layer, "_vllm_lod_attention_only_installed", False):
        return
    original_init = decoder_layer.__init__
    original_forward = decoder_layer.forward

    def initialize(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
        if (
            os.environ.get("LOD_BENCHMARK_TRACE_TIMEOUT")
            and not getattr(kimi_linear, "_lod_startup_trace_installed", False)
        ):
            import faulthandler

            faulthandler.dump_traceback_later(
                float(os.environ["LOD_BENCHMARK_TRACE_TIMEOUT"]), repeat=True,
            )
            kimi_linear._lod_startup_trace_installed = True
        original_init(self, config, *args, **kwargs)
        enabled = bool(getattr(config, "lod_attention_only_fixture", False))
        self._vllm_lod_attention_only_fixture = enabled
        if not enabled:
            return
        # The ordinary constructor needs a structurally valid MLP. Replace it
        # immediately so its parameters are released before cache allocation.
        self.mlp = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()

    def forward_attention_only(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_delta: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if not getattr(self, "_vllm_lod_attention_only_fixture", False):
            return original_forward(
                self,
                positions,
                hidden_states,
                residual,
                prefix_delta=prefix_delta,
                **kwargs,
            )
        if self.use_attn_residuals:
            # Preserve K3's real 12-layer attention-residual organization, but
            # finish the block immediately after attention instead of forming
            # the second (MLP) residual input and executing the MoE/FFN.
            prefix_sum = hidden_states
            hidden_states = kimi_linear._apply_attn_res(
                prefix_sum,
                residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
                delta=prefix_delta,
                output_norm=self.input_layernorm,
                block_write_idx=(
                    self.block_write_idx if self.is_block_write_layer else -1
                ),
            )
            if self.is_block_write_layer:
                prefix_sum = None
            hidden_states = self._run_self_attn(positions, hidden_states)
            if prefix_sum is None:
                prefix_sum = hidden_states
                prefix_delta = None
            else:
                prefix_delta = hidden_states
            return prefix_sum, residual, prefix_delta
        if prefix_delta is not None:
            raise RuntimeError("attention-only K3 fixture received prefix_delta")
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self._run_self_attn(positions, hidden_states)
        # Leave the attention output and preceding residual separate. The next
        # layer's fused RMSNorm adds them, exactly as a transformer block with
        # no feed-forward residual branch would do.
        return hidden_states, residual

    decoder_layer.__init__ = initialize
    decoder_layer.forward = forward_attention_only
    decoder_layer._vllm_lod_attention_only_installed = True


def _install_dense_gluon_decode() -> None:
    """Use the paper's faster dense absorbed-MLA decoder in controls.

    This is deliberately opt-in.  The gfx942 kernel tiles query heads in
    groups of 16, including the 96-head query gathered by K3 DCP.  LoD calls
    never reach this vLLM backend: they are intercepted above it.
    """

    if os.getenv("VLLM_KIMI_DENSE_GLUON", "0") != "1":
        return
    import vllm.envs as vllm_envs
    import vllm.v1.attention.backends.mla.triton_mla as triton_mla
    from vllm.logger import init_logger
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    impl = triton_mla.TritonMLAImpl
    builder = triton_mla.TritonMLAMetadataBuilder
    if getattr(impl, "_vllm_lod_dense_gluon_installed", False):
        return
    original_forward_mqa = impl.forward_mqa
    original_reserve = builder._reserve_attn_logits_workspace

    def reserve(self: Any) -> None:
        original_reserve(self)
        if (
            not is_workspace_manager_initialized()
            or int(self.mla_dims.kv_lora_rank) != 512
            or int(self.mla_dims.qk_rope_head_dim) != 64
        ):
            return
        batch = int(self.vllm_config.scheduler_config.max_num_seqs)
        if getattr(self, "non_causal_multi_token_decode", False):
            batch *= int(self.reorder_batch_threshold)
        heads = int(self.num_heads * self.dcp_world_size)
        dtype = self.vllm_config.model_config.dtype
        current_workspace_manager().get_simultaneous(
            ((batch, heads, 128, 512), dtype),
            ((batch, heads, 128), torch.float32),
        )

    def forward_mqa(
        self: Any,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        layer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        joined_q = torch.cat(q, dim=-1) if type(q) is tuple else q
        supported = (
            isinstance(joined_q, torch.Tensor)
            and joined_q.ndim == 3
            and joined_q.size(-1) == 576
            and joined_q.dtype == torch.bfloat16
            and kv_cache.dtype == torch.bfloat16
            and kv_cache.ndim == 3
            and kv_cache.size(-1) == 576
            and bool(attn_metadata.causal)
            and not vllm_envs.VLLM_BATCH_INVARIANT
        )
        if not supported:
            return original_forward_mqa(self, q, kv_cache, attn_metadata, layer)

        decode = attn_metadata.decode
        if decode is None:
            return original_forward_mqa(self, q, kv_cache, attn_metadata, layer)
        batch, heads, _ = joined_q.shape
        # Match vLLM's DCP split policy using this rank's local sequence
        # length.  A fixed 128 splits is appropriate for long DCP=1 histories,
        # but badly over-partitions an eight-way sequence shard.
        num_splits = min(
            128,
            triton_mla._compute_num_kv_splits(
                int(attn_metadata.max_seq_len), int(self._sm_count)
            ),
        )
        partial_shape = (batch, heads, num_splits, 512)
        lse_shape = (batch, heads, num_splits)
        if is_workspace_manager_initialized():
            partial, partial_lse = current_workspace_manager().get_simultaneous(
                (partial_shape, joined_q.dtype),
                (lse_shape, torch.float32),
            )
        else:
            partial = torch.empty(
                partial_shape, dtype=joined_q.dtype, device=joined_q.device
            )
            partial_lse = torch.empty(
                lse_shape, dtype=torch.float32, device=joined_q.device
            )
        output = torch.empty(
            batch, heads, 512, dtype=joined_q.dtype, device=joined_q.device
        )
        final_lse = torch.empty(
            batch, heads, dtype=torch.float32, device=joined_q.device
        )
        from lod_attention.kernels.kimi_gluon_decode import (
            absorbed_mla_decode_gfx942,
        )

        absorbed_mla_decode_gfx942(
            joined_q,
            kv_cache,
            output,
            decode.block_table,
            decode.seq_lens,
            self.scale,
            num_splits=num_splits,
            partial=partial,
            partial_lse=partial_lse,
            final_lse=final_lse,
        )
        return output, final_lse

    builder._reserve_attn_logits_workspace = reserve
    impl.forward_mqa = forward_mqa
    impl._vllm_lod_dense_gluon_installed = True
    init_logger(__name__).info(
        "Installed the gfx942 Gluon absorbed-MLA dense decode control"
    )


def absorb_query(
    query: torch.Tensor,
    w_uk_t: torch.Tensor,
    *,
    nope_dim: int,
) -> torch.Tensor:
    """Map per-head expanded queries into Kimi's cached latent coordinates."""

    q_nope, q_direct = query.split([nope_dim, query.size(-1) - nope_dim], dim=-1)
    # [T,H,P] -> [H,T,P] @ [H,P,L] -> [T,H,L]
    q_latent = torch.bmm(q_nope.transpose(0, 1), w_uk_t).transpose(0, 1)
    return torch.cat((q_latent, q_direct), dim=-1)


def pack_latent_record(
    latent: torch.Tensor,
    direct_key: torch.Tensor,
) -> torch.Tensor:
    """Form the one cached Kimi record ``[latent, direct-key]``."""

    if latent.ndim != 2 or direct_key.ndim != 3 or direct_key.size(1) != 1:
        raise ValueError("expected latent [T,L] and direct key [T,1,R]")
    latent = latent.unsqueeze(1)
    return torch.cat((latent, direct_key), dim=-1)


def _run_lod_mla(
    layer: Any,
    query: torch.Tensor,
    latent: torch.Tensor,
    direct_key: torch.Tensor,
    output_shape: torch.Size | tuple[int, ...] | None,
    q_dcp_replicated: torch.Tensor | None,
    *,
    output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    pool = getattr(layer, "_vllm_lod_pool", None)
    if pool is None:
        raise RuntimeError("Kimi LoD MLA forward has no attached cache pool")
    if getattr(layer, "W_UK_T", None) is None:
        raise RuntimeError("Kimi W_UK must be materialized before LoD attention")
    if output_shape is None:
        output_shape = (query.size(0), layer.num_heads * layer.v_head_dim)
    if output_buffer is not None and (
        tuple(output_buffer.shape) != tuple(output_shape)
        or output_buffer.dtype != query.dtype
        or output_buffer.device != query.device
    ):
        raise ValueError("Kimi MLA output buffer has incompatible geometry")

    dcp_decode = bool(
        q_dcp_replicated is not None
        and getattr(pool, "decode_enabled", False)
        and int(getattr(pool, "dcp_world_size", 1)) > 1
    )
    if dcp_decode:
        if layer.W_UK_T_dcp_qrep is None:
            raise RuntimeError("Kimi DCP query projection was not materialized")
        q_absorbed = absorb_query(
            q_dcp_replicated,
            layer.W_UK_T_dcp_qrep,
            nope_dim=int(layer.qk_nope_head_dim),
        )
        record = pack_latent_record(latent, direct_key)
        value = record[..., : latent.size(-1)]
        partial_output = torch.empty(
            q_absorbed.size(0),
            q_absorbed.size(1),
            latent.size(-1),
            dtype=q_absorbed.dtype,
            device=q_absorbed.device,
        )
        partial_output, partial_lse = pool.decode_dcp(
            q_absorbed,
            record,
            value,
            partial_output,
        )
        from vllm.distributed.parallel_state import get_dcp_group
        from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

        attention_output = cp_lse_ag_out_rs(
            partial_output,
            partial_lse,
            get_dcp_group(),
            is_lse_base_on_e=True,
        )
        output = (output_buffer if output_buffer is not None else
                  torch.empty(output_shape, dtype=query.dtype, device=query.device))
        layer._v_up_proj(attention_output, output)
        return output

    record = pack_latent_record(latent, direct_key)
    # This is a view of the record's latent prefix, not another tensor.  The
    # persistent LoD pool preserves the same aliasing for leaves, local K/V,
    # sinks, and centroid sums.
    value = record[..., : latent.size(-1)]
    direct_plan = getattr(pool, "direct_prefill_plan", None)
    if direct_plan and getattr(pool, "kimi_local_dcp_prefill", False):
        from .kimi_k3_dcp_prefill import local_dcp_prefill, shared_dcp_prefill

        prefill = (shared_dcp_prefill if pool.kimi_shared_dcp_prefill
                   else local_dcp_prefill)
        projected = prefill(layer, pool, query, record)
        shape = output_shape or (query.size(0), layer.num_heads * layer.v_head_dim)
        if output_buffer is not None:
            output_buffer.copy_(projected.reshape(shape))
            return output_buffer
        return projected.reshape(shape)
    projected_prefill = bool(
        direct_plan
        and any(int(end) - int(begin) > 1 for _, begin, end, _ in direct_plan)
        # The source-derived projected AITER path is specialized for full
        # K3's 512+64 latent/direct cache.  Smaller Kimi-K3-for-All checkpoints
        # use 128+64/128; absorb those queries normally and let the generic
        # asymmetric MLA path retain their native geometry.
        and int(latent.size(-1)) == 512
    )
    # Projected prefill consumes ``query`` directly in the D192 AITER kernels;
    # the D576 absorbed query is otherwise used only as a shape carrier by the
    # generic LoD engine.  Expanding the cache record over query heads gives it
    # that shape without executing a redundant [T,H,128]@[H,128,512] BMM.  A
    # scheduler batch can also contain one-token decode rows, so direct_prefill
    # lazily absorbs only those rows before dispatching them to decode.
    if projected_prefill:
        q_absorbed = record.expand(-1, int(query.size(1)), -1)
    else:
        q_absorbed = absorb_query(
            query,
            layer.W_UK_T,
            nope_dim=int(layer.qk_nope_head_dim),
        )
    attention_output = torch.empty(
        q_absorbed.size(0),
        q_absorbed.size(1),
        int(layer.v_head_dim) if projected_prefill else latent.size(-1),
        dtype=q_absorbed.dtype,
        device=q_absorbed.device,
    )

    if getattr(pool, "direct_prefill_plan", None) is not None:
        pool.direct_prefill(
            q_absorbed,
            record,
            value,
            attention_output,
            mla_query=query,
            mla_w_uk_t=layer.W_UK_T,
            mla_w_uv=(layer.W_UV if projected_prefill else None),
            defer_mla_query_absorption=projected_prefill,
        )
    elif bool(getattr(pool, "decode_enabled", False)) and int(query.size(0)) <= int(
        pool.max_requests
    ):
        class _Metadata:
            num_actual_tokens = int(query.size(0))

        if getattr(pool, "kimi_head_tiled_decode", False):
            pool.decode_dcp(q_absorbed, record, value, attention_output)
        else:
            pool.decode(q_absorbed, record, value, _Metadata(), attention_output)
    else:
        # Profile and graph-capture warmups carry no logical request. vLLM can
        # warm a 16-row decode graph even when max_num_seqs (and therefore the
        # fixed LoD cache pool) is smaller. Never interpret those synthetic
        # rows as cache indices. The authoritative pool deliberately has no
        # native chronological cache to fall back to here.
        attention_output.zero_()

    output = (output_buffer if output_buffer is not None else
              torch.empty(output_shape, dtype=query.dtype, device=query.device))
    if projected_prefill:
        output.view(query.size(0), layer.num_heads, layer.v_head_dim).copy_(
            attention_output
        )
    else:
        layer._v_up_proj(attention_output, output)
    return output


def _run_lod_mla_with_output(
    layer: Any,
    query: torch.Tensor,
    latent: torch.Tensor,
    direct_key: torch.Tensor,
    output: torch.Tensor,
    q_dcp_replicated: torch.Tensor | None,
) -> None:
    """Write a stable output across vLLM's eager attention graph breaks."""
    _run_lod_mla(layer, query, latent, direct_key, output.shape,
                 q_dcp_replicated, output_buffer=output)


def register_kimi_k3_lod() -> None:
    """Install the narrow MLA interception used only when a LoD pool exists."""

    _install_attention_only_fixture()

    import vllm.model_executor.layers.attention.mla_attention as mla_module
    from vllm.config import get_current_vllm_config

    MLAAttention = mla_module.MLAAttention

    if getattr(MLAAttention, "_vllm_lod_kimi_installed", False):
        return
    _install_dense_gluon_decode()
    original_init = MLAAttention.__init__
    original_forward = MLAAttention.forward
    original_prefill_selector = mla_module.get_mla_prefill_backend
    breakable_forward = None
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1":
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture

        # LoD intercepts MLAAttention.forward before the native unified MLA
        # custom op, so native attention's existing eager-break decorator is
        # bypassed. Supply the same boundary with an in-place graph-pool output.
        # Host cache plans, construction, and synchronization then stay outside
        # captured model sections and are re-evaluated on every replay.
        breakable_forward = eager_break_during_capture(_run_lod_mla_with_output)

    class _UnusedKimiPrefillBackend:
        """Construction placeholder; LoD bypasses native MLA prefill."""

        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

        def clone(self) -> Any:
            return type(self)(**self._kwargs)

        def prepare_metadata(self, _: Any) -> None:
            pass

        def supports_quant_output(self, _: Any) -> bool:
            return False

        def run_prefill_new_tokens(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("native MLA prefill reached the Kimi LoD placeholder")

        def run_prefill_context_chunk(self, *_: Any, **__: Any) -> Any:
            raise RuntimeError("native MLA prefill reached the Kimi LoD placeholder")

    def _is_kimi_lod_config(config: Any) -> bool:
        from ..config import lod_enabled

        text = config.model_config.hf_text_config
        backend = config.attention_config.backend
        backend_name = str(getattr(backend, "name", backend)).upper()
        return (
            lod_enabled()
            and str(getattr(text, "model_type", "")).lower() == "kimi_linear"
            and backend_name == "CUSTOM"
        )

    def get_mla_prefill_backend(config: Any) -> Any:
        if _is_kimi_lod_config(config):
            return _UnusedKimiPrefillBackend
        return original_prefill_selector(config)

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._vllm_lod_absorbed_mla = _is_kimi_lod_config(get_current_vllm_config())
        if self._vllm_lod_absorbed_mla:
            # Native ROCm MLA optionally replaces the BF16 absorbed projections
            # with private FP8/FP4 BMM layouts during its deferred post-load
            # hook. LoD consumes W_UK_T directly for latent-space routing and
            # W_UV for the exact output projection, so retain the ordinary
            # canonical tensors. This is decided before vLLM calls
            # process_weights_after_loading; dense MLA keeps its quantized BMM
            # representation and remains unchanged.
            self.is_aiter_triton_fp8_bmm_enabled = False
            self.is_aiter_triton_fp4_bmm_enabled = False

    def forward(
        self: Any,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "_vllm_lod_pool", None) is None:
            return original_forward(
                self,
                q,
                kv_c_normed,
                k_pe,
                output_shape=output_shape,
                q_dcp_replicated=q_dcp_replicated,
            )
        if breakable_forward is not None:
            shape = output_shape or (q.size(0), self.num_heads * self.v_head_dim)
            output = torch.empty(shape, dtype=q.dtype, device=q.device)
            breakable_forward(self, q, kv_c_normed, k_pe, output, q_dcp_replicated)
            return output
        return _run_lod_mla(
            self,
            q,
            kv_c_normed,
            k_pe,
            output_shape,
            q_dcp_replicated,
        )

    mla_module.get_mla_prefill_backend = get_mla_prefill_backend
    MLAAttention.__init__ = initialize
    MLAAttention.forward = forward
    MLAAttention._vllm_lod_kimi_installed = True


def register_kimi_k3_dense() -> None:
    """Install only the optimized dense K3 decoder used by controls.

    In particular, this deliberately does not patch ``MLAAttention`` or its
    prefill-backend selector.  A dense benchmark must exercise vLLM's native
    MLA registration and cache path without any LoD interception machinery.
    """

    _install_dense_gluon_decode()


__all__ = [
    "absorb_query",
    "pack_latent_record",
    "register_kimi_k3_dense",
    "register_kimi_k3_lod",
]
