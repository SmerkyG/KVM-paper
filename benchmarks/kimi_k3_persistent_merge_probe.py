"""Isolate last-arriver reduction on exact production LoD descriptors."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    from lod_attention.kernels.kimi_gluon_decode import absorbed_mla_lod_decode_gfx942, _absorbed_mla_stage1_gfx942
    from benchmarks.experimental.kimi_persistent_merge import persistent_lod
    from benchmarks.kimi_k3_kda_upstream_probe import graph_time, check
    from benchmarks.kimi_k3_sparse_mla_bias_probe import reference
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    result = dict(status="running", production_changed=False, scope=__doc__, points=[],
                  baseline_kda="G8 direct-state prefill; no KDA in isolated decode consumer")
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        for heads, splits, length in ((12, 32, 4096), (96, 32, 4096), (96, 32, 16384)):
            cache = torch.randn(length + 1, 576, device="cuda", dtype=torch.bfloat16)
            cache[-1].fill_(torch.nan)
            q = torch.randn(1, heads, 576, device="cuda", dtype=torch.bfloat16)
            bias = torch.randint(1, 1025, (length + 1,), device="cuda").float().log()
            fixed = torch.randperm(length, device="cuda").int()[None]
            tiles = (heads + 15) // 16
            # Each head group has distinct causal local+sink/coarse ranges.
            lengths = torch.tensor([length - i*31 for i in range(tiles)], device="cuda", dtype=torch.int32)
            exact = torch.zeros_like(lengths)
            local = torch.tensor([length], device="cuda", dtype=torch.int32)
            cache_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
            descriptors = torch.zeros(tiles, 1, device="cuda", dtype=torch.int32)
            partial = torch.empty(1, heads, splits, 512, device="cuda", dtype=torch.bfloat16)
            plse = torch.empty(1, heads, splits, device="cuda", dtype=torch.float32)
            out = torch.empty(1, heads, 512, device="cuda", dtype=torch.bfloat16)
            lse = torch.empty(1, heads, device="cuda", dtype=torch.float32)
            counters = torch.zeros(tiles, device="cuda", dtype=torch.int32)
            def native():
                return absorbed_mla_lod_decode_gfx942(q, cache, bias, out, descriptors, fixed, cache_indices,
                    lengths, exact, local, 192**-0.5, local_limit=length, include_new=False,
                    head_tiled_metadata=True, num_splits=splits, partial=partial, partial_lse=plse, final_lse=lse), lse
            def candidate():
                return persistent_lod(q, cache, bias, out, descriptors, fixed, cache_indices,
                    lengths, exact, local, partial, plse, lse, counters, 192**-0.5, local_limit=length, splits=splits)
            launches = []
            original_run = _absorbed_mla_stage1_gfx942.run
            def audit_run(*a, **kw):
                compiled = original_run(*a, **kw)
                launches.append(compiled)
                return compiled
            try:
                _absorbed_mla_stage1_gfx942.run = audit_run
                native()
            finally:
                _absorbed_mla_stage1_gfx942.run = original_run
            expected, el = out.clone(), lse.clone()
            candidate()
            point = dict(heads=heads, splits=splits, length=length, checks={
                "output": check(expected, out, tolerance=0.002), "lse_max_abs": float((el-lse).abs().max())})
            compiled = persistent_lod.last_kernel
            point["persistent_kernel"] = dict(registers=compiled.n_regs, spills=compiled.n_spills,
                                             shared_bytes=compiled.metadata.shared)
            base = launches[0]
            point["native_stage_kernel"] = dict(registers=base.n_regs, spills=base.n_spills,
                                               shared_bytes=base.metadata.shared)
            torch.testing.assert_close(el, lse, atol=0.002, rtol=0.0001)
            for t in range(tiles):
                end = min(heads, (t+1)*16)
                ids = fixed[0, :int(lengths[t])]
                ptr = torch.tensor([0, len(ids)], device="cuda", dtype=torch.int32)
                eo, ee = reference(q[:, t*16:end], cache, ptr, ids, 192**-0.5, bias)
                check(eo, out[:, t*16:end], tolerance=0.008)
                torch.testing.assert_close(ee, lse[:, t*16:end], atol=0.003, rtol=0.0001)
            assert torch.equal(counters, torch.zeros_like(counters)), "counter failed to reset"
            point["milliseconds"] = {}
            for name, fn in (("native", native), ("persistent", candidate), ("native_repeat", native), ("persistent_repeat", candidate)):
                point["milliseconds"][name] = graph_time(fn, replays=100)["ms"]
                assert torch.equal(counters, torch.zeros_like(counters)), "graph replay leaked counters"
            point["speedup"] = point["milliseconds"]["native"] / point["milliseconds"]["persistent"]
            # Stress the exact-page branch and all-masked/empty inputs outside
            # timing. Each group gets a different descriptor stream.
            for t in range(tiles):
                prefix = int(lengths[t]) - 64
                exact[t] = 64
                descriptors.resize_(tiles, 4) if t == 0 else None
                for j in range(4):
                    descriptors[t, j] = (16 << 24) | (prefix + j*16)
            native()
            eo, ee = out.clone(), lse.clone()
            candidate()
            point["exact_descriptor_check"] = check(eo, out, tolerance=0.002)
            torch.testing.assert_close(ee, lse, atol=0.002, rtol=0.0001)
            bias.fill_(-torch.inf)
            candidate()
            assert torch.equal(out, torch.zeros_like(out)) and torch.isneginf(lse).all()
            lengths.zero_()
            exact.zero_()
            candidate()
            assert torch.equal(out, torch.zeros_like(out)) and torch.isneginf(lse).all()
            assert torch.equal(counters, torch.zeros_like(counters))
            point["masked_and_empty_checks"] = "passed"
            result["points"].append(point)
            print("PERSISTENT_MERGE " + json.dumps(point), flush=True)
            save()
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
