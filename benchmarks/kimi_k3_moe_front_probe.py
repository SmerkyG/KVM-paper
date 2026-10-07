"""Matched streamed-weight graph test of native vs merged K3 MoE front."""

import argparse
import json
from pathlib import Path


def reference_front(x, packed_weight, shared_size, router_size):
    """Independent FP32 linear/SiTU oracle with native branch rounding."""
    import torch
    p = x.float() @ packed_weight.float().T
    gate = p[:, :shared_size].bfloat16().float()
    up = p[:, shared_size:2*shared_size].bfloat16().float()
    shared = ((4*torch.tanh(gate/4)*gate.sigmoid()) * (25*torch.tanh(up/25))).bfloat16()
    router = p[:, 2*shared_size:2*shared_size+router_size]
    latent = p[:, 2*shared_size+router_size:].bfloat16()
    return shared, router, latent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 512, 2048])
    p.add_argument("--shared-intermediate", type=int, default=6144,
                   help="global shared width: official K3 has 2*3072, hence 768 per TP8 rank")
    args = p.parse_args()
    import torch
    import vllm._custom_ops
    from benchmarks.experimental.kimi_moe_front import merged_front
    from benchmarks.kimi_k3_kda_upstream_probe import check, graph_time
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    result = dict(status="running", scope="per-rank MoE-front kernels only, not full-model timing/quality",
                  baseline_kda="G8 direct-state prefill (unchanged; no KDA executed in this isolated test)",
                  production_changed=False, checks=[], measurements=[], weight_sets=8,
                  shared_intermediate=args.shared_intermediate)
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        for tp in (8, 1):
            shared, router, latent, hidden = args.shared_intermediate // tp, 896, 3584, 7168
            weights = [torch.randn(2*shared+router+latent, hidden, device="cuda", dtype=torch.bfloat16) * hidden**-0.5 for _ in range(8)]
            for tokens in args.tokens:
                x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
                gu = torch.empty(tokens, 2*shared, device="cuda", dtype=torch.bfloat16)
                so = torch.empty(tokens, shared, device="cuda", dtype=torch.bfloat16)
                ro = torch.empty(tokens, router, device="cuda", dtype=torch.float32)
                lo = torch.empty(tokens, latent, device="cuda", dtype=torch.bfloat16)
                po = torch.empty(tokens, weights[0].shape[0], device="cuda", dtype=torch.float32)
                def native(w):
                    torch.mm(x, w[:2*shared].T, out=gu)
                    torch.ops._C.situ_and_mul(so, gu, 4.0, 25.0)
                    torch.mm(x, w[2*shared:2*shared+router].T, out=ro, out_dtype=torch.float32)
                    torch.mm(x, w[2*shared+router:].T, out=lo)
                    return so, ro, lo
                def candidate(w):
                    return merged_front(x, w, po, so, ro, lo)
                expected = [v.clone() for v in native(weights[0])]
                actual = candidate(weights[0])
                checks = {name: check(a, b, tolerance=0.008) for name, a, b in zip(("shared", "router", "latent"), expected, actual, strict=True)}
                matches = float((expected[1].topk(16, dim=-1).indices.sort(-1).values == actual[1].topk(16, dim=-1).indices.sort(-1).values).all(-1).float().mean())
                assert matches >= 0.999, "merged projection changed too many router selections"
                oracle = reference_front(x[:32], weights[0], shared, router)
                oracle_checks = {name: check(a, b[:32], tolerance=0.008) for name, a, b in zip(("shared", "router", "latent"), oracle, actual, strict=True)}
                result["checks"].append(dict(tp=tp, tokens=tokens, branches=checks, router_top16_match=matches, fp32_oracle=oracle_checks))
                def loop(fn):
                    def call():
                        out = None
                        for w in weights:
                            out = fn(w)
                        return out
                    return call
                times = {}
                for name, fn in (("native", native), ("merged", candidate), ("native_repeat", native), ("merged_repeat", candidate)):
                    measured = graph_time(loop(fn), replays=30)
                    times[name] = measured["ms"] / len(weights)
                point = dict(tp=tp, tokens=tokens, per_front_ms=times, speedup=times["native"] / times["merged"])
                result["measurements"].append(point)
                print("MOE_FRONT " + json.dumps(point), flush=True)
                save()
            del weights
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
