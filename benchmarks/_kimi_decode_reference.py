"""Eager-only correctness diagnostic, never a serving or timing path.

Replace each LoD DCP partial with dense FP32 attention over the *same* raw
cache after the normal decoder stores the current token. This distinguishes
an approximation/routing failure from a cache or integration failure without
loading the full model's weights again or adding a second native KV cache.
"""

from __future__ import annotations

import torch
from types import MethodType


def dense_pool_partial(pool, query):
    outputs, lses = [], []
    state = pool.state
    sink_len = state["sink_k"].size(2)
    for row, slot in enumerate(pool.active_decode_rows[:query.size(0)]):
        coverage = int(pool.metadata[slot]["coverage"])
        recent_len = int(pool.local_lens[slot].item())
        global_length = int(pool.dcp_global_lens[slot].item())
        if coverage + recent_len != pool._dcp_local_length(global_length):
            raise AssertionError("dense shadow reference has inconsistent DCP history lengths")
        key = torch.cat((
            state["sink_k"][slot, 0, :min(sink_len, coverage)],
            state["page_cache"]["leaf_k"][slot, 0, :coverage - sink_len],
            state["recent_k"][slot, 0, :recent_len],
        ), dim=0).float()
        scores = query[row].float() @ key.T * float(pool.engine.scaling)
        lses.append(torch.logsumexp(scores, dim=-1))
        outputs.append(torch.softmax(scores, dim=-1) @ key[:, :pool.value_dim])
    return torch.stack(outputs).to(query.dtype), torch.stack(lses)


def install_dense_shadow_decoder(worker):
    """Called by collective_rpc only after an explicitly eager engine starts."""
    model_state = worker.model_runner.model_state
    runtime = model_state._vllm_lod_runtime
    worker._kimi_dense_shadow_diagnostics = {}
    worker._kimi_decode_dispatch = {}

    def decode(pool, query, key, value, output):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("dense shadow reference cannot run during graph capture")
        result, lse = pool._benchmark_original_decode_dcp(query, key, value, output)
        reference, reference_lse = dense_pool_partial(pool, query)
        name = str(getattr(pool.layer, "layer_name", id(pool)))
        entry = worker._kimi_dense_shadow_diagnostics.setdefault(name, dict(
            calls=0, max_output_error=0.0, max_lse_error=0.0))
        entry["calls"] += 1
        entry["max_output_error"] = max(entry["max_output_error"],
            float((result.float() - reference.float()).abs().max()))
        entry["max_lse_error"] = max(entry["max_lse_error"],
            float((lse - reference_lse).abs().max()))
        if entry["calls"] == 1:
            print("KIMI_DENSE_SHADOW_REFERENCE", name, entry, flush=True)
        result.copy_(reference)
        return result, reference_lse

    for pool in runtime.pools.values():
        if hasattr(pool, "_benchmark_original_decode_dcp"):
            raise RuntimeError("dense shadow reference already installed")
        pool._benchmark_original_decode_dcp = pool.decode_dcp
        pool.decode_dcp = MethodType(decode, pool)
        layer = pool.layer
        original_forward = layer.forward

        def traced_forward(layer, *args, _original=original_forward, _pool=pool, **kwargs):
            query = args[0] if args else kwargs["q"]
            if query.size(0) == 1:
                name = str(getattr(layer, "layer_name", id(_pool)))
                record = worker._kimi_decode_dispatch.setdefault(name, dict(calls=0, flags={}))
                record["calls"] += 1
                flags = (f"enabled={_pool.decode_enabled} "
                    f"plan={_pool.direct_prefill_plan is not None} "
                    f"qrep={kwargs.get('q_dcp_replicated') is not None}")
                record["flags"][flags] = record["flags"].get(flags, 0) + 1
                if record["calls"] == 1:
                    print("KIMI_DECODE_DISPATCH", name, flags, flush=True)
            return _original(*args, **kwargs)

        layer.forward = MethodType(traced_forward, layer)
    return dict(installed=True, actual_pools=len(runtime.pools),
                pool_classes=sorted({type(pool).__module__ for pool in runtime.pools.values()}))


def dense_shadow_audit(worker):
    runtime = worker.model_runner.model_state._vllm_lod_runtime
    return dict(reference_calls=worker._kimi_dense_shadow_diagnostics,
        dispatch=worker._kimi_decode_dispatch,
        pools=[dict(layer=str(getattr(pool.layer, "layer_name", "?")),
            direct_prefill_calls=pool.direct_prefill_calls, decode_calls=pool.decode_calls,
            decode_enabled=pool.decode_enabled, ready=list(pool.ready),
            metadata=pool.metadata) for pool in runtime.pools.values()])
