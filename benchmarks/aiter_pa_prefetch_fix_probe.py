"""Triton 3.8 conditional-prefetch-carry A/B, BF16 paged attention on gfx942."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def compiled_resources(kernels):
    result = []
    for name, fn in kernels.items():
        for device_cache in fn.device_caches.values():
            for compiled in device_cache[0].values():
                result.append(dict(kernel=name, registers=compiled.n_regs,
                    spills=compiled.n_spills, shared_bytes=compiled.metadata.shared))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import triton
    from aiter.ops.triton.gluon import pa_decode_gluon as pa
    from benchmarks.experimental.pa_prefetch_fix import patched_prefetch
    from benchmarks.kimi_k3_kda_upstream_probe import graph_time

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    result = dict(status="in_progress", upstream_commit="8cfa0902", triton_version=triton.__version__,
                  arch=torch.cuda.get_device_properties(0).gcnArchName, points=[], production_changed=False,
                  scope="conventional paged attention; K3 MLA does not dispatch to these kernels")
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        for batch, heads, kv_heads in [(1, 16, 1), (8, 16, 1), (8, 64, 8)]:
            length, dim, page = 16384, 128, 16
            blocks = batch * length // page
            k = torch.randn(blocks, kv_heads, dim // 8, page, 8, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(blocks, kv_heads, dim, page, device="cuda", dtype=torch.bfloat16)
            q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
            table = torch.arange(blocks, device="cuda", dtype=torch.int32).reshape(batch, -1)
            lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
            def call():
                out = torch.empty_like(q)
                pa.pa_decode_gluon(out, q, k, v, lens, table, dim**-0.5, 1,
                    max_context_partition_num=pa.get_recommended_splits(batch, kv_heads),
                    compute_type=torch.bfloat16)
                return (out,)
            reference = call()[0]
            names = ("paged_attention_decode_sliding_window_head_1", "paged_attention_decode_sliding_window")
            before = compiled_resources({name: getattr(pa, name) for name in names})
            old_time = graph_time(call, replays=200)
            with patched_prefetch(pa) as kernels:
                candidate = call()[0]
                if not torch.equal(reference, candidate):
                    raise AssertionError("unconditional carry changed output")
                new_time = graph_time(call, replays=200)
                after = compiled_resources(kernels)
            point = dict(batch=batch, query_heads=heads, kv_heads=kv_heads, length=length,
                old=old_time, fixed=new_time, speedup=old_time["ms"] / new_time["ms"],
                bitwise_equal=True, resources_before=before, resources_after=after)
            result["points"].append(point)
            print("PA_PREFETCH_FIX " + json.dumps(point), flush=True)
            save()
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
