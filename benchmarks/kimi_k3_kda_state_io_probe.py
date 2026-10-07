"""Isolate direct paged state/output I/O from the already-tested Gluon math.

Both sides use the same #5866 prepare/walk and eight chunk groups. The only
difference is retaining versus removing initial-state gather, final-state
scatter and output copy. Includes noncontiguous live state slabs and guards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import vllm._custom_ops
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import gather_initial_states
    from benchmarks.kimi_k3_kda_upstream_probe import inputs, candidate, check, graph_time

    torch.set_num_threads(1)
    result = dict(scope=__doc__, status="in_progress", points=[], groups=8, heads=12,
                  production_changed=False)
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        save()
        for lengths in ([16384], [16384, 4097]):
            print("KDA_STATE_IO " + str(lengths), flush=True)
            inp = inputs(lengths, seed=1234)
            rows, heads, d = len(lengths), 12, 128
            slab = heads * d * d
            pitch, offset = slab + 256, 128
            # Native hybrid cache rows have other fields and padding. Keep
            # these poisoned to catch accidentally packing or overwriting.
            raw_a = torch.full(((rows + 2) * pitch,), 12345.0, device="cuda")
            raw_b = raw_a.clone()
            shape, strides = (rows + 2, heads, d, d), (pitch, d*d, d, 1)
            ca, cb = (r.as_strided(shape, strides, offset) for r in (raw_a, raw_b))
            ids = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")
            gen = torch.Generator(device="cuda").manual_seed(42)
            seed = torch.randn(rows, heads, d, d, device="cuda", generator=gen) * 0.1
            ca[ids.long()] = seed
            cb[ids.long()] = seed
            flags = torch.tensor([i % 2 == 0 for i in range(rows)], device="cuda")
            out_a = torch.empty_like(inp["v"])
            out_b = torch.empty_like(inp["v"])
            def copies():
                initial = gather_initial_states(ca, ids, flags)
                o, state = candidate(inp, initial, config={"G": 8})
                out_a.copy_(o)
                ca[ids.long()] = state
                return out_a, ca
            def direct():
                o, _ = candidate(inp, config={"G": 8}, state_cache=cb,
                    state_indices=ids, has_initial_state=flags, out=out_b)
                return o, cb
            before = raw_b.clone()
            a, _ = copies()
            b, _ = direct()
            point = dict(lengths=lengths, output=check(a, b, 1e-6),
                         state=check(ca[ids.long()], cb[ids.long()], 1e-6))
            writable = torch.zeros_like(raw_b, dtype=torch.bool)
            writable.as_strided(shape, strides, offset)[ids.long()] = True
            assert torch.equal(before[~writable], raw_b[~writable]), "write outside owned cache rows"
            assert torch.equal(raw_a, raw_b), "cache including guards differs"
            point["guards_unchanged"] = True
            point["copies"] = graph_time(copies)
            point["direct"] = graph_time(direct)
            point["speedup"] = point["copies"]["ms"] / point["direct"]["ms"]
            result["points"].append(point)
            print("KDA_STATE_IO_POINT " + json.dumps(point), flush=True)
            save()
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", exception=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
