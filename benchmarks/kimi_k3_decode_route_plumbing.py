"""Graph-timed exact route bookkeeping, excluding the actual collective.

The frozen gather fixtures have equivalent row-/rank-major candidate records.
These timings isolate packing, selection, ownership masking and copy-back;
end-to-end DCP communication is measured separately by the serving fixture.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.kimi_k3_decode_route_tune import graph_us
from lod_attention.kernels.distributed_topk import (
    distributed_global_topk, distributed_global_topk_into, localize_global_topk,
)


class Gather:
    world_size = 8
    rank_in_group = 3

    def __init__(self, records, rank_major=False):
        self.rank_major = rank_major
        self.records = (records.reshape(8, -1, 8, 2).flatten(0, 1) if rank_major
                        else torch.cat(list(records.unbind(0)), dim=-2))

    def all_gather(self, value, dim):
        assert dim == (0 if self.rank_major else -2)
        return self.records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope="route bookkeeping only; excludes collective and serving", rows=[])
    for batch in (1, 8):
        torch.manual_seed(21)
        scores = torch.randn(8, batch, 96, 1, 8, device="cuda")
        slots = torch.randint(0, 4096, scores.shape, device="cuda")
        records = torch.stack((scores, slots.float()), -1)
        old_group, new_group = Gather(records), Gather(records, True)
        old_scores, old_slots = scores[3].clone(), slots[3].clone()
        new_scores, new_slots = old_scores.clone(), old_slots.clone()
        packed = torch.empty_like(records[3])
        def old():
            s, owners, indices = distributed_global_topk(old_scores, old_slots, old_group)
            s, indices = localize_global_topk(s, owners, indices, local_rank=3)
            old_scores.copy_(s)
            old_slots.copy_(indices)
        def new():
            distributed_global_topk_into(new_scores, new_slots, new_group, packed=packed)
        old()
        new()
        assert torch.equal(new_scores, old_scores) and torch.equal(new_slots, old_slots)
        for name, function in (("original", old), ("fixed_rank_major", new)):
            duration, samples = graph_us(function)
            result["rows"].append(dict(batch=batch, implementation=name,
                graph_kernel_us=duration, samples_us=samples, exact_match=True))
    result["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
