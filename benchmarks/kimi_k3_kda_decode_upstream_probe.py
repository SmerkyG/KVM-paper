"""Compare upstream Gluon KDA decode with the image's already-fused HIP decoder.

Recurrent state comes from a native 16K prefill; convolution, recurrence,
gated RMSNorm and both writable caches are included for both kernels.
No MLA/LoD or production setting changes are made by this per-rank probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--buffers", type=int, default=2)
    args = p.parse_args()
    import torch
    from vllm import _custom_ops as ops
    from benchmarks.kimi_k3_kda_upstream_probe import inputs, current, check, graph_time
    from benchmarks.experimental.kda_gfx942.decode import fused_recurrent_kda_packed_decode

    torch.set_num_threads(1)
    result = dict(scope=__doc__, upstream_commit="8253efc4", heads=args.heads, head_dim=128,
                  context=16384, num_buffers=args.buffers, status="in_progress", points=[])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        save()
        for batch in args.batches:
            print(f"KDA_DECODE_PREFILL B{batch}", flush=True)
            inp = inputs([16384] * batch, args.heads, seed=1234)
            _, pref_state = current(inp)
            gen = torch.Generator(device="cuda").manual_seed(4321)
            h, d = args.heads, 128
            width = 4 * h * d + h + 128
            packed = torch.randn(batch, width, dtype=torch.bfloat16, device="cuda", generator=gen)
            mixed = packed[:, :3 * h * d]
            out_gate = packed[:, 3 * h * d:4 * h * d].view(batch, h, d)
            beta = packed[:, 4 * h * d:4 * h * d + h]
            g = torch.randn(batch, h, d, dtype=torch.bfloat16, device="cuda", generator=gen)
            weight = torch.randn(3, 4, h * d, device="cuda", generator=gen) * 0.25
            norm = torch.ones(d, device="cuda")
            # Match vLLM's SD cache: caller exposes the [slots,channels,width]
            # transposed view to both fused decoders, not a contiguous copy.
            conv0 = torch.randn(batch + 1, 3, 3 * h * d, dtype=torch.bfloat16,
                                device="cuda", generator=gen).transpose(1, 2)
            state0 = torch.zeros(batch + 1, h, d, d, device="cuda")
            state0[1:].copy_(pref_state)
            indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
            out = torch.empty(batch, h, d, dtype=torch.bfloat16, device="cuda")

            def run(variant, state, conv):
                if variant == "hip":
                    actual = ops.fused_kda_decode(mixed, weight, None, conv,
                        g.unsqueeze(0), beta.unsqueeze(0), inp["A_log"], inp["dt_bias"],
                        indices, state, out=out.unsqueeze(0), lower_bound=-5.0,
                        output_gate=out_gate, norm_weight=norm, norm_eps=1e-5)
                    return actual, state
                actual = fused_recurrent_kda_packed_decode(mixed, g, beta,
                    inp["A_log"], inp["dt_bias"], -5.0, state, indices,
                    conv, weight, out_gate, norm, 1e-5, out,
                    config={"num_buffers": args.buffers})
                return actual.unsqueeze(0), state

            old_s, new_s = state0.clone(), state0.clone()
            old_c, new_c = conv0.clone(), conv0.clone()
            old_o, _ = run("hip", old_s, old_c)
            old_o = old_o.clone()
            new_o, _ = run("gluon", new_s, new_c)
            point = dict(batch=batch, output=check(old_o, new_o),
                         state=check(old_s, new_s), conv=check(old_c, new_c))
            # Longer recurrent rollouts verify state updates, not just one
            # output from an accidentally non-mutating kernel.
            for _ in range(32):
                run("hip", old_s, old_c)
                run("gluon", new_s, new_c)
            point["state_after_33_steps"] = check(old_s, new_s)
            point["conv_after_33_steps"] = check(old_c, new_c)
            point["hip"] = graph_time(lambda: run("hip", old_s, old_c), replays=1025)
            point["gluon"] = graph_time(lambda: run("gluon", new_s, new_c), replays=1025)
            point["speedup"] = point["hip"]["ms"] / point["gluon"]["ms"]
            result["points"].append(point)
            print("KDA_DECODE_POINT " + json.dumps(point), flush=True)
            save()
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", exception=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
