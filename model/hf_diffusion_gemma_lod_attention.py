"""Inference-only LOD adapter for Hugging Face DiffusionGemma.

This module deliberately contains all DiffusionGemma-specific integration.
It does not modify Transformers' DiffusionGemma implementation or add model
conditionals to the generic LOD engines.

DiffusionGemma has two attention phases.  Its causal encoder creates a KV
cache, while each diffusion step issues new bidirectional canvas queries
against that read-only cache.  The adapter therefore:

* applies ordinary causal LOD attention while the encoder builds its cache;
* recomputes LOD routing from every diffusion step's current queries; and
* treats the exact encoder-local field plus the whole current canvas as one
  bidirectional attention branch, LSE-merged with coarse and opened leaves.

For compatibility with the native decoder masks and sliding-attention layers,
the Hugging Face cache remains present.  LOD state is an attached sidecar, so
this first integration targets attention compute rather than KV-cache memory.
Only global-attention layers are replaced; sliding layers remain native.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MethodType
from typing import Any

import torch
from torch import nn

from .hf_lod_left_padding import GroupedHFLODRuntime, build_padding_plan
from .hf_pytorch_lod_attention import (
    HFLODSettings,
    _build_engine,
    _has_attention_norm,
)
from .pytorch_lod_attention import LODCache, LODConfig
from .pytorch_lod_attention_fast import (
    _fast_coarse_attention,
    _gathered_leaf_attention,
    _merge_two_branches,
    _packed_leaf_attention,
    _prefer_gathered_leaves,
    _route_state,
)
from .pytorch_lod_attention_paged import (
    PagedLODCache,
    PagedLODConfig,
    _recursive_page_attention,
    _reference_coarse_attention,
)
from .triton_lod_engines import KernelLODCache


_CONTEXT_ATTRIBUTE = "_diffusion_gemma_lod_context"
_SETTINGS_ATTRIBUTE = "_diffusion_gemma_lod_settings"
_ORIGINAL_FORWARD_ATTRIBUTE = "_diffusion_gemma_lod_original_forward"
_INPUT_MASK_ATTRIBUTE = "_diffusion_gemma_lod_input_mask"
_ENCODER_ORIGINAL_FORWARD_ATTRIBUTE = "_diffusion_gemma_lod_encoder_original_forward"
_PREFILL_POLICY_ATTRIBUTE = "_diffusion_gemma_lod_prefill_policy"
_ENCODER_ATTENTION_ATTRIBUTE = "_diffusion_gemma_encoder_attention_mode"


@dataclass
class _DiffusionGemmaLODLayer:
    settings: HFLODSettings
    grouped: GroupedHFLODRuntime


def _install_diffusion_decoder_router(engine: nn.Module, mode: str) -> None:
    """Install a decoder-only router without changing encoder state building."""
    if mode == "per_query":
        return
    if mode not in ("canvas_max", "canvas_cumulative_max"):
        raise ValueError(f"unknown DiffusionGemma decoder routing mode {mode!r}")
    original = engine._route_top_slots

    def route_top_slots(
        self: nn.Module,
        q: torch.Tensor,
        state_k: torch.Tensor,
        state_v: torch.Tensor,
        counts: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not getattr(self, "_diffusion_decoder_routing_active", False):
            return original(q, state_k, state_v, counts, **kwargs)

        state_len = int(kwargs["state_len"])
        protected_len = (
            self._protected_state_len(state_len)
            if self.exclude_sink_from_routes
            else 0
        )
        route_count = min(int(self.two_level_topk), state_len - protected_len)
        if route_count <= 0:
            return torch.empty(
                *q.shape[:3], 0, dtype=torch.long, device=q.device
            )
        with torch.no_grad():
            logits = self._state_route_logits(
                q, state_k, counts, state_len=state_len
            )
            query_counts = self._repeat_kv(
                counts.detach()[..., :state_len, :]
            ).squeeze(-1)
            scores = logits.float() * float(self.scaling)
            scores = scores + query_counts.float().log().unsqueeze(2)
            scores[..., :protected_len] = float("-inf")
            canvas_scores = scores.amax(dim=2)
            if mode == "canvas_cumulative_max":
                previous = getattr(self, "_diffusion_canvas_route_scores", None)
                if previous is None or tuple(previous.shape) != tuple(
                    canvas_scores.shape
                ):
                    previous = canvas_scores
                else:
                    previous = torch.maximum(previous, canvas_scores)
                self._diffusion_canvas_route_scores = previous
                canvas_scores = previous
            shared = canvas_scores.topk(
                route_count, dim=-1, sorted=False
            ).indices
            return shared.unsqueeze(2).expand(
                -1, -1, int(q.size(2)), -1
            ).contiguous()

    engine._route_top_slots = MethodType(route_top_slots, engine)


def _configure_diffusion_engine(
    engine: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    prefill_policy: str,
) -> None:
    """Apply DiffusionGemma prefill policy and wide-head kernel safeguards."""
    if prefill_policy not in ("optimized", "legacy"):
        raise ValueError(f"unknown prefill policy {prefill_policy!r}")
    if prefill_policy == "legacy" and hasattr(engine, "prefill_chunk_len"):
        # Restore the recursive-engine behavior before lod-dev 954269b while
        # retaining later kernel-only tuning. These are the six attributes
        # whose new defaults alter prefill partitioning/routing arithmetic.
        engine.prefill_chunk_len = int(engine.chunk_len)
        engine.prefill_local_len = int(engine.local_len)
        engine.prefill_state_update_len = int(engine.chunk_len)
        engine.prefill_two_level_topk = None
        engine.split_prefill_local_attention = False
        engine.fused_prefill_route_coarse = False
    if not hasattr(engine, "coarse_route_block_m"):
        return
    groups = int(query.size(1)) // int(key.size(1))
    element_bytes = int(query.element_size())
    row_bytes = groups * int(query.size(-1)) * element_bytes
    # The kernel materializes one GQA-expanded query tile.  A 32 KiB query
    # tile leaves room for its streaming-softmax accumulators on MI300X and
    # avoids DiffusionGemma's 16x2x512 geometry requesting 128 KiB at once.
    safe_block_m = max(1, 32 * 1024 // max(row_bytes, 1))
    engine.coarse_route_block_m = min(
        int(engine.coarse_route_block_m), safe_block_m
    )
    if int(engine.coarse_route_block_m) < 8:
        engine.coarse_route_num_warps = min(
            int(engine.coarse_route_num_warps), 4
        )
    # The direct decode router holds a 64-key by head-dimension tile.  At 512
    # dimensions that alone reaches MI300X's shared-memory limit.  Reuse the
    # already-supported logits router, which streams over the compact state
    # without a head-dimension-sized Triton tile.
    if int(query.size(-1)) >= 512:
        engine.direct_fused_state_routing = False
        engine.route_gqa_matmul = True


def _project_qkv(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply DiffusionGemma's model-owned projection/norm/RoPE operations."""
    try:
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            apply_rotary_pos_emb,
        )
    except ImportError as exc:  # pragma: no cover - depends on installed HF version
        raise ImportError(
            "DiffusionGemma LOD requires a Transformers release containing "
            "transformers.models.diffusion_gemma (5.11 or newer)."
        ) from exc

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = position_embeddings

    query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape))
    query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)

    key = module.k_proj(hidden_states).view(hidden_shape)
    value = module.v_proj(hidden_states).view(hidden_shape) if module.v_proj is not None else key
    key = module.k_norm(key)
    key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value = module.v_norm(value).transpose(1, 2)
    return query, key, value


def _finish_attention(
    module: nn.Module, output: torch.Tensor, input_shape: torch.Size
) -> tuple[torch.Tensor, None]:
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return module.o_proj(output), None


def _cache_context(past_key_values: Any, *, create: bool) -> dict[int, _DiffusionGemmaLODLayer]:
    context = getattr(past_key_values, _CONTEXT_ATTRIBUTE, None)
    if context is None and create:
        context = {}
        setattr(past_key_values, _CONTEXT_ATTRIBUTE, context)
    if context is None:
        raise RuntimeError(
            "the DiffusionGemma encoder cache has no LOD sidecar; encode the "
            "prompt with the LOD adapter installed before invoking the decoder"
        )
    return context


def _encoder_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    if module.training:
        raise RuntimeError("DiffusionGemma LOD currently supports inference only")
    if past_key_values is None:
        raise RuntimeError("DiffusionGemma LOD encoder attention requires a cache")

    input_shape = hidden_states.shape[:-1]
    query, key, value = _project_qkv(module, hidden_states, position_embeddings)
    encoder_attention = getattr(module, _ENCODER_ATTENTION_ATTRIBUTE, "lod")
    native_output = None
    if encoder_attention == "native":
        original = getattr(module, _ORIGINAL_FORWARD_ATTRIBUTE)
        native_output = original(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    elif encoder_attention != "lod":
        raise ValueError(f"unknown encoder attention mode {encoder_attention!r}")
    module_settings = getattr(module, _SETTINGS_ATTRIBUTE)
    if (
        module_settings.engine_backend == "kernel"
        and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in (query, key, value))
    ):
        raise RuntimeError(
            "kernel DiffusionGemma LOD requires torch.no_grad() or inference_mode()"
        )

    # Preserve the native cache for sliding layers, mask construction, position
    # bookkeeping, and a clean fallback path.  LOD owns an additional sidecar.
    # Native encoder attention updates this layer's cache itself. In LOD mode,
    # preserve the native cache explicitly for the decoder and diagnostics.
    if encoder_attention == "lod":
        past_key_values.update(key, value, module.layer_idx)
    context = _cache_context(past_key_values, create=True)
    layer = context.get(module.layer_idx)
    initial_prefill = layer is None
    if layer is None:
        settings = module_settings
        padding_mask = getattr(module, _INPUT_MASK_ATTRIBUTE, attention_mask)
        if padding_mask is not None and not isinstance(padding_mask, torch.Tensor):
            padding_mask = attention_mask
        plan = build_padding_plan(
            padding_mask,
            batch_size=int(query.size(0)),
            sequence_length=int(query.size(2)),
        )
        layer = _DiffusionGemmaLODLayer(
            settings=settings,
            grouped=GroupedHFLODRuntime(plan, device=query.device),
        )
        for runtime in layer.grouped.runtimes:
            runtime.engine = _build_engine(
                settings,
                query,
                key,
                scale=float(module.scaling),
                stats_owner=module,
            )
            _configure_diffusion_engine(
                runtime.engine,
                query,
                key,
                prefill_policy=getattr(
                    module, _PREFILL_POLICY_ATTRIBUTE, "optimized"
                ),
            )
            _install_diffusion_decoder_router(
                runtime.engine,
                getattr(module, "_diffusion_gemma_decoder_routing", "per_query"),
            )
        context[module.layer_idx] = layer

    output = layer.grouped.consume(
        layer.settings,
        query,
        key,
        value,
        initial_prefill=initial_prefill,
        scale=float(module.scaling),
        stats_owner=module,
    )
    if native_output is not None:
        return native_output
    return _finish_attention(module, output, input_shape)


def _local_field(
    cache: LODCache | PagedLODCache | KernelLODCache,
    canvas_key: torch.Tensor,
    canvas_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, KernelLODCache):
        state = cache.state
        recent_key = state["recent_k"]
        recent_value = state["recent_v"]
        if not isinstance(recent_key, torch.Tensor) or not isinstance(recent_value, torch.Tensor):
            raise TypeError("kernel LOD cache is missing its exact local field")
        recent_length = int(state.get("recent_len", recent_key.size(2)))
        recent_key = recent_key[..., :recent_length, :]
        recent_value = recent_value[..., :recent_length, :]
    elif isinstance(cache, (LODCache, PagedLODCache)):
        recent_key = cache.recent_key
        recent_value = cache.recent_value
    else:
        raise TypeError(f"unsupported DiffusionGemma LOD cache {type(cache).__name__}")
    return (
        torch.cat((recent_key, canvas_key), dim=2).contiguous(),
        torch.cat((recent_value, canvas_value), dim=2).contiguous(),
    )


def _torch_diffusion_attention(
    engine: nn.Module,
    cache: LODCache | PagedLODCache,
    query: torch.Tensor,
    canvas_key: torch.Tensor,
    canvas_value: torch.Tensor,
    *,
    open_count: int,
    scale: float,
) -> torch.Tensor:
    local_key, local_value = _local_field(cache, canvas_key, canvas_value)
    state = cache.state
    if state.slot_count == 0:
        # This occurs only for very short contexts.  A direct, all-visible
        # local branch is then exact and avoids routing an empty state.
        key = local_key
        value = local_value
        groups = int(query.size(1)) // int(key.size(1))
        if groups != 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
        probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        return torch.matmul(probability, value)

    top_slots = open_mask = None
    if open_count:
        top_slots, open_mask = _route_state(
            query,
            state,
            local_key,
            max_routes=engine.config.max_routes,
            open_count=open_count,
            route_protected_prefix=engine.config.protected_prefix,
            routing_normalization=engine.config.routing_normalization,
            routing_count_bias=engine.config.routing_count_bias,
            routing_variance_bias=engine.config.routing_variance_bias,
            scale=scale,
        )

    # query_offset=local_length makes every local key visible to every canvas
    # query.  The low-level helper intentionally accepts this broader mask even
    # though its public causal wrapper restricts queries to suffix alignment.
    if query.device.type == "cpu" or torch.is_grad_enabled():
        empty_slots = torch.empty(
            *query.shape[:-1], 0, dtype=torch.long, device=query.device
        )
        empty_mask = torch.empty_like(empty_slots, dtype=torch.bool)
        coarse_output, coarse_lse = _reference_coarse_attention(
            query,
            local_key,
            local_value,
            state,
            empty_slots if top_slots is None else top_slots,
            empty_mask if open_mask is None else open_mask,
            scale=scale,
            query_offset=int(local_key.size(2)),
        )
    else:
        coarse_output, coarse_lse = _fast_coarse_attention(
            query,
            local_key,
            local_value,
            state,
            top_slots=top_slots,
            open_mask=open_mask,
            scale=scale,
            query_offset=int(local_key.size(2)),
        )
    if not open_count or top_slots is None or open_mask is None:
        return coarse_output

    if isinstance(cache, PagedLODCache):
        pages = engine._cached_region_pages(
            cache.owner, state, cache.leaves
        )
        exact_output, exact_lse = _recursive_page_attention(
            query,
            state,
            cache.leaves,
            pages,
            top_slots,
            open_mask,
            scale=scale,
        )
    else:
        if cache.owner is None or cache.leaf_key is None or cache.leaf_value is None:
            raise RuntimeError("flat two-level LOD cache has no exact leaf archive")
        postings = engine._cached_postings(cache.owner, state)
        leaf_attention = (
            _gathered_leaf_attention
            if query.device.type == "cpu"
            or _prefer_gathered_leaves(query, state, top_slots, open_mask)
            else _packed_leaf_attention
        )
        exact_output, exact_lse = leaf_attention(
            query,
            cache.leaf_key,
            cache.leaf_value,
            cache.owner,
            state,
            top_slots,
            open_mask,
            postings[0],
            postings[1],
            scale=scale,
        )
    return _merge_two_branches(
        coarse_output, coarse_lse, exact_output, exact_lse
    )


def _kernel_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.device.type == "cpu":
        raise RuntimeError("the kernel DiffusionGemma LOD backend requires a GPU")
    output, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        0.0,
        False,
        False,
        scale=scale,
    )
    return output, lse


def _kernel_diffusion_attention(
    engine: nn.Module,
    cache: KernelLODCache,
    query: torch.Tensor,
    canvas_key: torch.Tensor,
    canvas_value: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    state = cache.state
    engine._lod_state = state
    local_key, local_value = _local_field(cache, canvas_key, canvas_value)
    local_branch = _kernel_local_attention(
        query, local_key, local_value, scale=scale
    )
    page_cache = state.get("page_cache")
    owners = state.get("owners")
    exact_key = state.get("exact_k", local_key[..., :0, :])
    exact_value = state.get("exact_v", local_value[..., :0, :])
    if not isinstance(exact_key, torch.Tensor) or not isinstance(exact_value, torch.Tensor):
        raise TypeError("kernel LOD cache has invalid exact-leaf tensors")
    empty_key = local_key[..., :0, :].contiguous()
    empty_value = local_value[..., :0, :].contiguous()
    if int(engine.two_level_topk) == 0:
        no_slots = torch.empty(
            *query.shape[:-1], 0, dtype=torch.long, device=query.device
        )
        coarse_output, coarse_lse = engine._coarse_attention(
            query,
            empty_key,
            empty_value,
            state["state_k"],
            state["state_v"],
            state["counts"],
            no_slots,
            state_len=int(state["state_len"]),
            state_capacity=int(state["state_capacity"]),
            include_local=False,
        )
        local_output, local_lse = local_branch
        if state.get("sink_k") is not None:
            from .triton_lod_attention import merge_attention_branches_with_sink

            return merge_attention_branches_with_sink(
                query,
                state["sink_k"],
                state["sink_v"],
                coarse_output,
                coarse_lse,
                local_output,
                local_lse,
                kv_group_size=engine.num_key_value_groups,
                scale=scale,
            )
        return _merge_two_branches(
            coarse_output, coarse_lse, local_output, local_lse
        )
    return engine._two_level_attention(
        query,
        empty_key,
        empty_value,
        state["state_k"],
        state["state_v"],
        state["counts"],
        owners,
        exact_key,
        exact_value,
        state_len=int(state["state_len"]),
        state_capacity=int(state["state_capacity"]),
        page_cache=page_cache,
        local_branch=local_branch,
        sink_k=state.get("sink_k"),
        sink_v=state.get("sink_v"),
    )


def _decoder_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    if getattr(module, "_diffusion_gemma_native_attention_active", False):
        original = getattr(module, _ORIGINAL_FORWARD_ATTRIBUTE)
        return original(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    del attention_mask, kwargs
    if module.training:
        raise RuntimeError("DiffusionGemma LOD currently supports inference only")
    if past_key_values is None:
        raise RuntimeError("DiffusionGemma LOD decoder attention requires an encoder cache")

    input_shape = hidden_states.shape[:-1]
    query, canvas_key, canvas_value = _project_qkv(
        module, hidden_states, position_embeddings
    )
    layer = _cache_context(past_key_values, create=False).get(module.layer_idx)
    if layer is None:
        raise RuntimeError(f"DiffusionGemma LOD layer {module.layer_idx} was not encoded")
    if (
        layer.settings.engine_backend == "kernel"
        and torch.is_grad_enabled()
        and any(tensor.requires_grad for tensor in (query, canvas_key, canvas_value))
    ):
        raise RuntimeError(
            "kernel DiffusionGemma LOD requires torch.no_grad() or inference_mode()"
        )

    output = torch.zeros_like(query)
    for runtime in layer.grouped.runtimes:
        if runtime.engine is None or runtime.lod_cache is None:
            raise RuntimeError("DiffusionGemma LOD group has no encoded state")
        indices = runtime.indices.to(query.device)
        group_query = query.index_select(0, indices).contiguous()
        group_key = canvas_key.index_select(0, indices).contiguous()
        group_value = canvas_value.index_select(0, indices).contiguous()
        if isinstance(runtime.lod_cache, KernelLODCache):
            decoder_open_count = int(
                getattr(
                    module,
                    "_diffusion_gemma_decoder_open_count",
                    layer.settings.open_count,
                )
            )
            original_topk = int(runtime.engine.two_level_topk)
            # The generic engine identifies prefill by query_len > 1.  A
            # DiffusionGemma denoising canvas is also a multi-token query, but
            # it is decoder work and must honor decoder_open_count rather than
            # the optimized encoder-prefill route budget.  Disable the two
            # prefill-only routing knobs for this call; otherwise optimized
            # prefill silently turns nominal top-8/top-16 decoder views into
            # the same top-3 view.
            original_prefill_topk = runtime.engine.prefill_two_level_topk
            original_fused_prefill = runtime.engine.fused_prefill_route_coarse
            runtime.engine.two_level_topk = decoder_open_count
            runtime.engine.prefill_two_level_topk = None
            runtime.engine.fused_prefill_route_coarse = False
            runtime.engine._diffusion_decoder_routing_active = True
            try:
                group_output = _kernel_diffusion_attention(
                    runtime.engine,
                    runtime.lod_cache,
                    group_query,
                    group_key,
                    group_value,
                    scale=float(module.scaling),
                )
            finally:
                runtime.engine._diffusion_decoder_routing_active = False
                runtime.engine.two_level_topk = original_topk
                runtime.engine.prefill_two_level_topk = original_prefill_topk
                runtime.engine.fused_prefill_route_coarse = original_fused_prefill
        else:
            group_output = _torch_diffusion_attention(
                runtime.engine,
                runtime.lod_cache,
                group_query,
                group_key,
                group_value,
                open_count=layer.settings.open_count,
                scale=float(module.scaling),
            )
        output.index_copy_(0, indices, group_output)
    return _finish_attention(module, output, input_shape)


def _model_parts(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    base = getattr(model, "model", model)
    encoder = getattr(base, "encoder", None)
    decoder = getattr(base, "decoder", None)
    language_model = getattr(encoder, "language_model", None)
    if language_model is None or decoder is None:
        raise TypeError(
            "expected DiffusionGemmaForBlockDiffusion or DiffusionGemmaModel "
            "with encoder.language_model and decoder modules"
        )
    return language_model, decoder


def _encoder_model_forward(
    module: nn.Module, *args: Any, **kwargs: Any
) -> Any:
    """Expose the compact user padding mask to patched attention modules."""
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is None and len(args) > 1:
        attention_mask = args[1]
    patched_attention = [
        layer.self_attn
        for layer in module.layers
        if hasattr(layer.self_attn, _SETTINGS_ATTRIBUTE)
    ]
    for attention in patched_attention:
        setattr(attention, _INPUT_MASK_ATTRIBUTE, attention_mask)
    try:
        original = getattr(module, _ENCODER_ORIGINAL_FORWARD_ATTRIBUTE)
        return original(*args, **kwargs)
    finally:
        for attention in patched_attention:
            if hasattr(attention, _INPUT_MASK_ATTRIBUTE):
                delattr(attention, _INPUT_MASK_ATTRIBUTE)


def install_diffusion_gemma_lod_attention(
    model: nn.Module,
    *,
    config: LODConfig | PagedLODConfig | None = None,
    open_count: int = 8,
    engine_backend: str = "kernel",
    decoder_open_count: int | None = None,
    decoder_routing: str = "per_query",
    prefill_policy: str = "optimized",
    encoder_attention_mode: str = "lod",
) -> list[int]:
    """Replace DiffusionGemma global attention with an external LOD adapter.

    Returns the global layer indices that were patched.  The model must be in
    eval mode when run.  Calling this function twice is idempotent when the
    settings are unchanged; uninstall first to change settings.
    """
    config = PagedLODConfig() if config is None else config
    decoder_open_count = (
        open_count if decoder_open_count is None else decoder_open_count
    )
    if decoder_open_count < 0:
        raise ValueError("decoder_open_count cannot be negative")
    if decoder_open_count > 8 and engine_backend != "kernel":
        raise ValueError(
            "more than eight decoder routes currently requires the kernel backend"
        )
    if decoder_routing not in ("per_query", "canvas_max", "canvas_cumulative_max"):
        raise ValueError(f"unknown decoder routing mode {decoder_routing!r}")
    if prefill_policy not in ("optimized", "legacy"):
        raise ValueError(f"unknown prefill policy {prefill_policy!r}")
    if encoder_attention_mode not in ("lod", "native"):
        raise ValueError(
            f"unknown encoder attention mode {encoder_attention_mode!r}"
        )
    encoder, decoder = _model_parts(model)
    encoder_layers = getattr(encoder, "layers", ())
    decoder_layers = getattr(decoder, "layers", ())
    if len(encoder_layers) != len(decoder_layers):
        raise ValueError("DiffusionGemma encoder and decoder layer counts differ")

    patched = []
    for layer_index, (encoder_layer, decoder_layer) in enumerate(
        zip(encoder_layers, decoder_layers, strict=True)
    ):
        encoder_attention = getattr(encoder_layer, "self_attn", None)
        decoder_attention = getattr(decoder_layer, "self_attn", None)
        if encoder_attention is None or decoder_attention is None:
            continue
        if bool(getattr(encoder_attention, "is_sliding", False)):
            continue
        if bool(getattr(decoder_attention, "is_sliding", False)):
            raise ValueError(f"encoder/decoder layer type mismatch at layer {layer_index}")
        module_config = config
        if config.state_clustering_policy == "qk_norm_aware":
            encoder_has_key_norm = _has_attention_norm(encoder_attention, "k")
            decoder_has_key_norm = _has_attention_norm(decoder_attention, "k")
            if encoder_has_key_norm != decoder_has_key_norm:
                raise ValueError(
                    "DiffusionGemma encoder/decoder K-normalization differs at "
                    f"layer {layer_index}"
                )
            if encoder_has_key_norm:
                module_config = replace(
                    config,
                    state_clustering_policy="manual",
                    state_clustering_normalization="none",
                    state_clustering_centroid_rescale="coherence",
                    state_clustering_centroid_rescale_scope="assignment",
                )
            else:
                module_config = replace(
                    config,
                    state_clustering_policy="manual",
                    state_clustering_normalization="cosine",
                    state_clustering_centroid_rescale="none",
                )
        elif config.state_clustering_policy != "manual":
            raise ValueError(
                "DiffusionGemma currently supports automatic state clustering "
                "only with qk_norm_aware; use the manual geometry controls for "
                "positional-encoding policies"
            )
        settings = HFLODSettings(
            config=module_config,
            open_count=open_count,
            engine_backend=engine_backend,
            backend_name="diffusion_gemma_lod",
        )
        previous_prefill_policy = getattr(
            encoder_attention, _PREFILL_POLICY_ATTRIBUTE, None
        )
        if (
            previous_prefill_policy is not None
            and previous_prefill_policy != prefill_policy
        ):
            raise RuntimeError(
                "DiffusionGemma LOD is already installed with a different "
                "prefill policy"
            )
        setattr(encoder_attention, _PREFILL_POLICY_ATTRIBUTE, prefill_policy)
        previous_encoder_attention = getattr(
            encoder_attention, _ENCODER_ATTENTION_ATTRIBUTE, None
        )
        if (
            previous_encoder_attention is not None
            and previous_encoder_attention != encoder_attention_mode
        ):
            raise RuntimeError(
                "DiffusionGemma LOD is already installed with a different "
                "encoder attention mode"
            )
        setattr(
            encoder_attention,
            _ENCODER_ATTENTION_ATTRIBUTE,
            encoder_attention_mode,
        )
        for attention, forward in (
            (encoder_attention, _encoder_forward),
            (decoder_attention, _decoder_forward),
        ):
            previous_settings = getattr(attention, _SETTINGS_ATTRIBUTE, None)
            if previous_settings is not None:
                if previous_settings != settings:
                    raise RuntimeError(
                        "DiffusionGemma LOD is already installed with different settings"
                    )
                continue
            setattr(attention, _ORIGINAL_FORWARD_ATTRIBUTE, attention.forward)
            setattr(attention, _SETTINGS_ATTRIBUTE, settings)
            attention.forward = MethodType(forward, attention)
        setattr(
            decoder_attention,
            "_diffusion_gemma_decoder_open_count",
            decoder_open_count,
        )
        setattr(
            decoder_attention,
            "_diffusion_gemma_decoder_routing",
            decoder_routing,
        )
        patched.append(layer_index)
    if not patched:
        raise ValueError("no DiffusionGemma global-attention layers were found")
    if not hasattr(encoder, _ENCODER_ORIGINAL_FORWARD_ATTRIBUTE):
        setattr(encoder, _ENCODER_ORIGINAL_FORWARD_ATTRIBUTE, encoder.forward)
        encoder.forward = MethodType(_encoder_model_forward, encoder)
    return patched


def uninstall_diffusion_gemma_lod_attention(model: nn.Module) -> list[int]:
    """Restore attention methods replaced by the DiffusionGemma LOD adapter."""
    encoder, decoder = _model_parts(model)
    restored = []
    for layer_index, (encoder_layer, decoder_layer) in enumerate(
        zip(encoder.layers, decoder.layers, strict=True)
    ):
        changed = False
        for attention in (encoder_layer.self_attn, decoder_layer.self_attn):
            original = getattr(attention, _ORIGINAL_FORWARD_ATTRIBUTE, None)
            if original is None:
                continue
            attention.forward = original
            delattr(attention, _ORIGINAL_FORWARD_ATTRIBUTE)
            delattr(attention, _SETTINGS_ATTRIBUTE)
            for attribute in (
                "_diffusion_gemma_decoder_open_count",
                "_diffusion_gemma_decoder_routing",
                _PREFILL_POLICY_ATTRIBUTE,
                _ENCODER_ATTENTION_ATTRIBUTE,
            ):
                if hasattr(attention, attribute):
                    delattr(attention, attribute)
            changed = True
        if changed:
            restored.append(layer_index)
    original_encoder_forward = getattr(
        encoder, _ENCODER_ORIGINAL_FORWARD_ATTRIBUTE, None
    )
    if original_encoder_forward is not None:
        encoder.forward = original_encoder_forward
        delattr(encoder, _ENCODER_ORIGINAL_FORWARD_ATTRIBUTE)
    return restored


__all__ = [
    "install_diffusion_gemma_lod_attention",
    "uninstall_diffusion_gemma_lod_attention",
]
