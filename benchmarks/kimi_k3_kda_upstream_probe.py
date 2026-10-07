"""Isolated KDA upstream A/B probe; pinned runtime, no production monkeypatch.

First validate against an FP32 recurrence and the image's current KDA, then
time the same 16K operands. This is a per-rank kernel fixture, not K3 latency
or trained-model quality evidence. The full dense four-layer probe imports
these same adapters only after this preflight passes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path


def inputs(lengths, heads=12, seed=0, shift=0.0, duplicate=False):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    n = sum(lengths)
    packed = torch.randn(1, n, 3 * heads * 128, device="cuda",
                         dtype=torch.bfloat16, generator=gen)
    q, k, v = [packed[..., j * heads * 128:(j + 1) * heads * 128]
               .unflatten(-1, (heads, 128)) for j in range(3)]
    if duplicate:
        k.copy_(k[:, :1] + 0.02 * k)
    g = torch.randn(1, n, heads, 128, device="cuda",
                    dtype=torch.bfloat16, generator=gen) + shift
    # Match the strided beta field of a fused QKV/G projection.
    beta_backing = torch.randn(1, n, heads + 16, device="cuda",
                               dtype=torch.bfloat16, generator=gen)
    al = torch.empty(heads, device="cuda").uniform_(1, 16, generator=gen).log()
    bias = torch.randn(heads * 128, device="cuda", generator=gen)
    cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda")
    ci = torch.tensor([[s, c] for s, length in enumerate(lengths)
                       for c in range((length + 63) // 64)], dtype=torch.int32, device="cuda")
    co = torch.tensor([0, *itertools.accumulate((n + 63) // 64 for n in lengths)],
                      dtype=torch.int64, device="cuda")
    return dict(q=q, k=k, v=v, g=g, beta=beta_backing[..., :heads], A_log=al,
                dt_bias=bias, cu_seqlens=cu, chunk_indices=ci, chunk_offsets=co)


def current(inp, initial_state=None, *, state_cache=None, state_indices=None,
            has_initial_state=None, out=None):
    from vllm.models.kimi_k3.amd.ops.kda_prefill import chunk_kda_prefill

    return chunk_kda_prefill(q=inp["q"], k=inp["k"], v=inp["v"],
        raw_g=inp["g"], raw_beta=inp["beta"], A_log=inp["A_log"],
        g_bias=inp["dt_bias"], lower_bound=-5.0, use_qk_l2norm_in_kernel=True,
        cu_seqlens=inp["cu_seqlens"], chunk_indices=inp["chunk_indices"],
        chunk_offsets=inp["chunk_offsets"], initial_state=initial_state,
        output_final_state=state_cache is None, state_cache=state_cache,
        state_indices=state_indices, has_initial_state=has_initial_state,
        out=out, use_fused_chunk=False)


def candidate(inp, initial_state=None, *, config=None, state_cache=None,
              state_indices=None, has_initial_state=None, out=None):
    from benchmarks.experimental.kda_gfx942.chunk import chunk_kda

    return chunk_kda(**inp, lower_bound=-5.0, initial_state=initial_state,
        output_final_state=state_cache is None, config=config,
        state_cache=state_cache, state_indices=state_indices,
        has_initial_state=has_initial_state, out=out)


def reference(inp, initial_state):
    """Independent natural-exponential token recurrence, V-first FP32 state."""
    import torch
    import torch.nn.functional as F

    q = F.normalize(inp["q"].float(), dim=-1, eps=1e-6)[0]
    k = F.normalize(inp["k"].float(), dim=-1, eps=1e-6)[0]
    v = inp["v"].float()[0]
    g = -5 * ((inp["g"].float()[0] + inp["dt_bias"].reshape(q.shape[1], 128))
              * inp["A_log"].exp()[None, :, None]).sigmoid()
    beta = inp["beta"].float()[0].sigmoid()
    outputs, states = [], []
    for row, (begin, end) in enumerate(itertools.pairwise(inp["cu_seqlens"].tolist())):
        state = initial_state[row].clone()
        for t in range(begin, end):
            state = state * g[t].exp()[:, None, :]
            correction = (v[t] - (state * k[t, :, None, :]).sum(-1)) * beta[t, :, None]
            state = state + correction[..., None] * k[t, :, None, :]
            outputs.append((state * (q[t] * 128**-0.5)[:, None, :]).sum(-1))
        states.append(state)
    return torch.stack(outputs).unsqueeze(0), torch.stack(states)


def difference(expected, actual):
    import torch

    if not torch.isfinite(actual).all():
        raise AssertionError("non-finite candidate output/state")
    diff = expected.double() - actual.double()
    rms = float(expected.double().square().mean().sqrt())
    return dict(relative_rms=float(diff.square().mean().sqrt()) / max(rms, 1e-12),
                max_abs=float(diff.abs().max()), reference_rms=rms)


def check(expected, actual, tolerance=0.008):
    result = difference(expected, actual)
    if result["relative_rms"] >= tolerance:
        raise AssertionError(f"relative RMS {result} exceeds {tolerance}")
    return result


def graph_time(call, replays=100):
    """One timed block of warm replays; compilation/capture is untimed."""
    import torch

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retained = call()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    stop.synchronize()
    return dict(ms=start.elapsed_time(stop) / replays, replays=replays,
                outputs=[list(x.shape) if x is not None else None for x in retained])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, default=16384)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    import torch
    import vllm._custom_ops  # Register the native runtime before importing its KDA.

    torch.set_num_threads(1)
    torch.manual_seed(42)
    result = dict(scope=__doc__, upstream_commit="446b8e9a", candidate="Gluon chunk KDA CDNA3 port",
                  device=torch.cuda.get_device_name(), arch=torch.cuda.get_device_properties(0).gcnArchName,
                  heads=args.heads, dim=128, groups=args.groups, seed=0, status="in_progress",
                  production_changed=False, checks=[])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        save()
        for lengths, shift, duplicate in [([63, 65], 0.0, False),
                                          ([65], -3.0, True), ([64], 3.0, False)]:
            print(f"KDA_PREFLIGHT {lengths} shift={shift} duplicate={duplicate}", flush=True)
            inp = inputs(lengths, args.heads, shift=shift, duplicate=duplicate)
            state = torch.randn(len(lengths), args.heads, 128, 128, device="cuda") * 0.1
            ref_o, ref_s = reference(inp, state)
            old_o, old_s = current(inp, state)
            new_o, new_s = candidate(inp, state)
            # The native BF16 chunk solve itself exceeds 0.8% on deliberately
            # near-identical keys with weak decay. Report both oracle errors,
            # and separately require the backport to match the native path.
            tolerance = 0.02 if duplicate else 0.008
            point = dict(lengths=lengths, shift=shift, duplicate=duplicate,
                         current_output=check(ref_o, old_o, tolerance), current_state=check(ref_s, old_s, tolerance),
                         candidate_output=check(ref_o, new_o, tolerance), candidate_state=check(ref_s, new_s, tolerance),
                         old_vs_new_output=check(old_o, new_o, tolerance),
                         old_vs_new_state=check(old_s, new_s, tolerance), tolerance=tolerance)
            result["checks"].append(point)
            print("KDA_CHECK " + json.dumps(point), flush=True)
            save()
        if not args.smoke_only:
            inp = inputs([args.length], args.heads, seed=1234)
            state = torch.zeros(1, args.heads, 128, 128, device="cuda")
            old_o, old_s = current(inp, state)
            new_o, new_s = candidate(inp, state, config={"G": args.groups})
            result["long_check"] = dict(output=check(old_o, new_o), state=check(old_s, new_s))
            print("KDA_16K_CHECK " + json.dumps(result["long_check"]), flush=True)
            result["current"] = graph_time(lambda: current(inp, state))
            result["candidate"] = graph_time(lambda: candidate(inp, state, config={"G": args.groups}))
            result["speedup"] = result["current"]["ms"] / result["candidate"]["ms"]
            print("KDA_TIMING " + json.dumps({k:result[k] for k in ("current", "candidate", "speedup")}), flush=True)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", exception=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
