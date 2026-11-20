"""Measure whole-frontier matches and region reuse on frozen prefill inputs."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks._identity import benchmark_identity
from lod_attention.kernels.prefill_frontier_order import frontier_keys
from lod_attention.kernels.threshold_prefill import threshold_prefill_route_coarse_attention


def audit(slots, counts, lengths, group, tile):
    signature, work = frontier_keys(slots, counts, lengths, group=group, method="signature", write_entries=True)
    anchors, _ = frontier_keys(slots, counts, lengths, group=group, method="largest")
    rows = slots.view(*signature.shape, slots.size(-1))
    length = lengths.view(signature.size(0), -1)
    regions = length.size(-1)
    sizes = (2, 4, 8, 16, 32)
    queries = 0
    groups = 0
    largest_group = 0
    hash_collisions = 0
    exact_tile_kv = 0
    total_pairs = 0
    covered_queries = dict.fromkeys(sizes, 0)
    covered_pairs = dict.fromkeys(sizes, 0)
    union_counts = dict.fromkeys(("adjacent", "signature", "largest"), 0)
    for kh in range(signature.size(0)):
        ids = signature[kh].argsort(stable=True)
        ordered = rows[kh, ids]
        active_work = work[kh, ids]
        same = (ordered[1:] == ordered[:-1]).all(-1)
        ordered_keys = signature[kh, ids]
        hash_collisions += int(((ordered_keys[1:] == ordered_keys[:-1]) & ~same).sum())
        starts = torch.cat((torch.zeros(1, dtype=torch.long, device=ids.device), (~same).nonzero().flatten() + 1))
        ends = torch.cat((starts[1:], starts.new_tensor([ids.numel()])))
        group_size = ends - starts
        per_query = active_work[starts].long()
        nonempty = per_query > 0
        largest_group = max(largest_group, int(torch.where(nonempty, group_size, 0).max()))
        groups += int(nonempty.sum())
        queries += int(group_size[nonempty].sum())
        total_pairs += int((group_size * per_query).sum())
        exact_tile_kv += int((((group_size + tile - 1) // tile) * per_query).sum())
        for threshold in sizes:
            included = nonempty & (group_size >= threshold)
            covered_queries[threshold] += int(group_size[included].sum())
            covered_pairs[threshold] += int((group_size * per_query)[included].sum())
        for name, ordered in (
            ("adjacent", rows[kh]),
            ("signature", ordered),
            ("largest", rows[kh, anchors[kh].argsort(stable=True)]),
        ):
            pad = (-ordered.size(0)) % tile
            if pad:
                ordered = torch.nn.functional.pad(ordered, (0, 0, 0, pad), value=-1)
            choices = ordered.view(-1, tile * slots.size(-1))
            valid_slots = torch.where(choices >= 0, choices, regions)
            present = torch.zeros((choices.size(0), regions + 1), device=choices.device, dtype=torch.bool)
            present.scatter_(1, valid_slots, True)
            union_counts[name] += int((present[:, :-1] * length[kh][None, :]).sum())
    return dict(
        query_heads_with_open_regions=queries, distinct_nonempty_sets=groups, largest_group=largest_group,
        hash_collisions=hash_collisions, exact_query_entry_pairs=total_pairs, exact_group_tile_entries=exact_tile_kv,
        queries_in_exact_groups_at_least=covered_queries, pairs_in_exact_groups_at_least=covered_pairs,
        entries_read_per_order=union_counts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--time-order", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.snapshots.glob("layer-*.pt")):
        saved = torch.load(path, map_location="cuda", weights_only=False)
        data = saved["data"]
        q = data["q"][:, :, :args.queries].contiguous()
        group = q.size(1) // saved["kv_heads"]
        slots, coarse, _, _ = threshold_prefill_route_coarse_attention(
            q, data["k"], data["v"], data["c"], state_len=data["s"], kv_group_size=group,
            scale=float(saved["engine"]["scaling"]), normalize_route_query=True,
        )
        lengths = data["pages"]["slot_lengths"]
        row = dict(layer=saved["layer"], queries=q.size(2), **audit(
            slots, coarse.route_counts, lengths, group, args.tile))
        if args.time_order:
            from triton.testing import do_bench_cudagraph

            def order(method):
                keys, _ = frontier_keys(slots, coarse.route_counts, lengths, group=group, method=method)
                return keys.argsort(dim=-1, stable=True)

            row["order_build_ms"] = {
                method: do_bench_cudagraph(lambda: order(method), rep=120)
                for method in ("signature", "largest")
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del saved, data, q, slots, coarse
    if not rows:
        raise ValueError("no frozen prefill inputs")
    args.output.write_text(json.dumps(dict(
        stage="complete", snapshots=str(args.snapshots), group_tile=args.tile, records=rows,
        benchmark_identity=benchmark_identity(),
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
