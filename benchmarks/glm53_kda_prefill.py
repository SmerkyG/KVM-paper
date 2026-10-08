"""Port the validated Kimi gfx942 KDA arithmetic to GLM, without state-I/O changes.

This opt-in probe preserves GLM's already-sigmoided beta, V-first FP32 state,
bounded gate, convolution, normalization, and state gather/scatter. Decode is
untouched. The original function remains the fallback for unsupported inputs.
"""

from __future__ import annotations


def install_glm_kda_prefill():
    from vllm.models.glm5next.common import kda as module
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices, prepare_chunk_offsets,
    )
    from benchmarks.experimental.kda_gfx942.chunk import chunk_kda

    if getattr(module.chunk_kda_with_fused_gate, "_lod_glm_g8", False):
        return
    original = module.chunk_kda_with_fused_gate

    def prefill(q, k, v, raw_g, beta, A_log, g_bias, scale=None,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                safe_gate=False, lower_bound=-5.0, **kwargs):
        supported = (
            q.is_cuda and q.shape[0] == 1 and q.shape[-1] == v.shape[-1] == 128
            and use_qk_l2norm_in_kernel and safe_gate and lower_bound == -5.0
            and cu_seqlens is not None and g_bias is not None and not kwargs
        )
        if not supported:
            return original(q=q, k=k, v=v, raw_g=raw_g, beta=beta,
                A_log=A_log, g_bias=g_bias, scale=scale, initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens, safe_gate=safe_gate,
                lower_bound=lower_bound, **kwargs)
        prefill.calls += 1
        # Use the same cached ragged metadata helper as native GLM. No new
        # host length read, inverse sigmoid, or beta rounding is introduced.
        return chunk_kda(q=q, k=k, v=v, g=raw_g, beta=beta,
            A_log=A_log.reshape(-1), dt_bias=g_bias.reshape(-1),
            cu_seqlens=cu_seqlens,
            chunk_indices=prepare_chunk_indices(cu_seqlens, 64),
            chunk_offsets=prepare_chunk_offsets(cu_seqlens, 64),
            initial_state=initial_state, output_final_state=output_final_state,
            lower_bound=lower_bound, scale=scale, config={"G": 8},
            beta_activated=True)

    prefill.calls = 0
    prefill._lod_glm_g8 = True
    module.chunk_kda_with_fused_gate = prefill


def kernel_probe(output):
    import torch
    import vllm._custom_ops  # Register runtime ops before importing the model.
    from benchmarks._vllm import write_json
    from benchmarks.kimi_k3_kda_upstream_probe import inputs, reference, check, graph_time
    from vllm.models.glm5next.common import kda as module

    torch.manual_seed(193)
    original = module.chunk_kda_with_fused_gate
    install_glm_kda_prefill()
    candidate = module.chunk_kda_with_fused_gate
    result = dict(scope="isolated GLM KDA kernel; not model speed", checks=[], timings=[])

    def call(fn, inp, initial):
        return fn(q=inp["q"], k=inp["k"], v=inp["v"], raw_g=inp["g"],
            beta=inp["beta"].float().sigmoid(), A_log=inp["A_log"],
            g_bias=inp["dt_bias"], initial_state=initial, output_final_state=True,
            use_qk_l2norm_in_kernel=True, cu_seqlens=inp["cu_seqlens"],
            safe_gate=True, lower_bound=-5.0)

    for lengths in ([129], [65, 3, 129]):
        inp = inputs(lengths, heads=16, seed=193)
        initial = torch.randn(len(lengths), 16, 128, 128, device="cuda") * .02
        expected = reference(inp, initial)
        native, actual = call(original, inp, initial), call(candidate, inp, initial)
        result["checks"].append(dict(lengths=lengths,
            native=[check(e, a) for e, a in zip(expected, native)],
            gluon=[check(e, a) for e, a in zip(expected, actual)]))
        write_json(output, result)
    for heads in (16, 64):
        inp = inputs([16384], heads=heads, seed=193)
        initial = torch.zeros(1, heads, 128, 128, device="cuda")
        controls, candidate_ms = [], None
        for fn in (original, candidate, original):
            ms = graph_time(lambda: call(fn, inp, initial), replays=20)["ms"]
            if fn is original:
                controls.append(ms)
            else:
                candidate_ms = ms
        result["timings"].append(dict(heads=heads, native_bracket_ms=controls,
            gluon_ms=candidate_ms, speedup=sum(controls) / 2 / candidate_ms))
        write_json(output, result)
        print(result["timings"][-1], flush=True)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    kernel_probe(p.parse_args().output)
