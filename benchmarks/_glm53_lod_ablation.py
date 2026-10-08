"""Benchmark-only prefill/leaf-cap controls; normal serving never selects these.

Exact prefill retains a temporary, request-local latent history. The normal
LoD path still constructs its cache, but its prefill output is replaced by
causal attention over ALL history, either on every row or just the final
prefill row per request. Decode is completely unchanged. This is
an accuracy isolation test, deliberately not a performance implementation.
"""
from __future__ import annotations

import torch


def projected_exact_attention(q, latent, uk, uv, *, attention=None, **kwargs):
    if attention is None:
        from lod_attention.kernels.glm_projected_prefill import projected_local_attention
        attention = projected_local_attention
    if q.size(2) == 1 and latent.size(2) > 1:
        # AITER dispatches one query to a separate unmasked CK library. Reuse
        # the already-compiled causal library instead: the extra query occupies
        # position N-2 and is discarded; the real query at N-1 sees all N keys.
        destination = kwargs.pop("output_buffer")
        assert kwargs["query_offset"] == latent.size(2) - 1
        kwargs["query_offset"] -= 1
        result, lse = attention(torch.cat((q, q), dim=2), latent, uk, uv,
                                output_buffer=None, **kwargs)
        destination.copy_(result[..., -1:, :])
        return destination, lse
    return attention(q, latent, uk, uv, **kwargs)


def exact_prefill_output(layer, plan, prompt_lengths, query, latent, output,
                         *, attention=None, final_row_only=False):
    if attention is None:
        attention = projected_exact_attention
    histories = getattr(layer, "_glm53_exact_prefill_history", None)
    if histories is None:
        histories = layer._glm53_exact_prefill_history = {}
    for slot, begin, end, previous in plan:
        if end <= begin:
            continue
        if previous + end - begin > prompt_lengths[slot]:
            raise RuntimeError("exact-prefill ablation requires a prefill-only cohort step")
        current = latent[begin:end]
        if previous == 0:
            history = current.clone()
        else:
            history = histories.get(slot)
            if history is None or history.size(0) != previous:
                raise RuntimeError("exact GLM prefill history does not match the request offset")
            history = torch.cat((history, current), dim=0)
        if history.size(0) != previous + end - begin:
            raise AssertionError("exact GLM prefill history lost sequence positions")
        final_chunk = history.size(0) == prompt_lengths[slot]
        if not final_row_only or final_chunk:
            exact_begin = end - 1 if final_row_only else begin
            exact_offset = history.size(0) - 1 if final_row_only else previous
            row_q = query[exact_begin:end].transpose(0, 1).unsqueeze(0)
            row_kv = history[None, None]
            destination = output[exact_begin:end].view(end - exact_begin, layer.num_heads, layer.v_head_dim)
            result, _ = attention(row_q, row_kv, layer.W_UK_T, layer.W_UV,
                query_offset=exact_offset, scale=layer.scale, return_lse=False, buffers={},
                output_buffer=destination.transpose(0, 1).unsqueeze(0))
            destination.copy_(result[0].transpose(0, 1))
            layer._glm53_exact_prefill_calls = getattr(layer, "_glm53_exact_prefill_calls", 0) + 1
            layer._glm53_exact_prefill_tokens = getattr(layer, "_glm53_exact_prefill_tokens", 0) + end - exact_begin
        if final_chunk:
            histories.pop(slot, None)  # Decode must use only the original LoD cache.
        else:
            histories[slot] = history
    return output


def install_lod_ablation(*, exact_prefill=False, uncapped=False, exact_final_row=False):
    if exact_prefill and exact_final_row:
        raise ValueError("choose all-prefill or final-row exact attention, not both")
    if exact_prefill or exact_final_row:
        from vllm_lod_plugin.models import glm53_flash as glm
        original_attention = glm.latent_attention

        def attention(layer, query, latent, direct_key, output_shape=None,
                      q_dcp_replicated=None):
            pool = layer._vllm_lod_pool
            plan = pool.direct_prefill_plan
            lengths = dict(pool.direct_prefill_prompt_lengths) if plan is not None else None
            result = original_attention(layer, query, latent, direct_key,
                                        output_shape, q_dcp_replicated)
            if plan is not None:
                return exact_prefill_output(layer, plan, lengths, query, latent, result,
                                            final_row_only=exact_final_row)
            return result

        glm.latent_attention = attention
    if uncapped:
        # Apply before pool/cache allocation and construction, not after any
        # leaves might already have been discarded. Ranking/top-eight stay put.
        from vllm_lod_plugin import pool
        from lod_attention._config import ModelFamily
        original_configuration = pool.configure_engine

        def configure(engine, **kwargs):
            original_configuration(engine, **kwargs)
            if kwargs["family"] is ModelFamily.GLM53_FLASH:
                engine.max_open_centroid_leaves = None

        pool.configure_engine = configure


def __getattr__(name):
    controls = {
        "GLMExactPrefillWorker": (True, False, False),
        "GLMUncappedWorker": (False, True, False),
        "GLMUncappedExactPrefillWorker": (True, True, False),
        "GLMExactFinalRowWorker": (False, False, True),
    }
    if name not in controls:
        raise AttributeError(name)
    from vllm.v1.worker.gpu_worker import Worker
    exact_prefill, uncapped, exact_final_row = controls[name]

    class GLMAblationWorker(Worker):
        def __init__(self, *args, **kwargs):
            install_lod_ablation(exact_prefill=exact_prefill, uncapped=uncapped,
                                 exact_final_row=exact_final_row)
            super().__init__(*args, **kwargs)

    return GLMAblationWorker
