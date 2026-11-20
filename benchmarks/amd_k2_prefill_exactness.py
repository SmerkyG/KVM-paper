"""Compare the full routed frontier and output on archived K2 query chunks."""

import argparse
import json
import os
from pathlib import Path


def audit(path, trial=None):
    import torch

    from lod_attention.kernels.aiter_prefill_attention import merge_aiter_prefill_refinement
    from lod_attention.kernels.paged_prefill import paged_leaf_attention
    from lod_attention.kernels.threshold_prefill import threshold_prefill_route_coarse_attention

    snapshot = torch.load(path, map_location="cuda", weights_only=False)
    data = snapshot["data"]
    q = data["q"]
    page = data["pages"]
    engine = snapshot["engine"]
    group = q.size(1) // snapshot["kv_heads"]
    scale = float(engine["scaling"])
    local_out = torch.zeros_like(q)
    local_lse = torch.full(q.shape[:-1], -torch.inf, device=q.device)

    def evaluate(options):
        for key in (
            "PACK_LOG_SIZE", "NARROW_SLOTS", "LIVE_PREFIX", "REUSE_SCORES", "COMPACT_ROWS",
            "MERGE_SHORT_MMA", "QK_BACKEND", "PAD_LOG_SIZE", "MATERIALIZE_REGIONS", "OVERLAP_PROJECTION",
        ):
            os.environ.pop("LOD_AMD_PREFILL_" + key, None)
        for key, value in options.items():
            os.environ["LOD_AMD_PREFILL_" + key] = value
        slots, coarse, hc, offsets = threshold_prefill_route_coarse_attention(
            q, data["k"], data["v"], data["c"], state_len=data["s"], kv_group_size=group,
            scale=scale, normalize_route_query=True,
            overlap_projection=options.get("OVERLAP_PROJECTION") == "1",
        )
        out, lse = paged_leaf_attention(
            q, page["leaf_k"], page["leaf_v"], page["slot_pages"], page["overflow_page_keys"],
            page["overflow_page_values"], page["overflow_used"], page["slot_lengths"], slots,
            page_indices=page["page_indices"], kv_group_size=group, active_slots=data["s"],
            scale=scale, block_m=engine["leaf_block_m"], block_n=engine["leaf_block_n"],
            num_warps=engine["leaf_num_warps"], hash_probes=-1 if page.get("paged_page_directory")
            else (8 if page.get("overflow_active") else 0), route_head_counts=hc, route_offsets=offsets,
            route_counts=coarse.route_counts, exact_row_starts=coarse.exact_row_starts,
            reduce_routes=False, persistent_workers=608,
        )
        output = merge_aiter_prefill_refinement(
            q, data["sink_k"], data["sink_v"], coarse, slots, out, lse, local_out, local_lse,
            kv_group_size=group, scale=scale,
        )
        return slots, coarse.output_0, coarse.lse_0, output

    reference = evaluate({})
    results = []
    storage = {"PACK_LOG_SIZE": "1", "NARROW_SLOTS": "1", "LIVE_PREFIX": "1"}
    variants = (
        ("packed-and-live", storage),
        ("score-scratch-reuse", dict(storage, REUSE_SCORES="1")),
        ("compact-and-pad", dict(storage, COMPACT_ROWS="2")),
        ("combined", dict(storage, REUSE_SCORES="1", COMPACT_ROWS="2")),
    )
    if trial is not None:
        options = dict(item.split('=', 1) for item in trial.split(','))
        variants = ((trial, {**storage, "COMPACT_ROWS": "2", **options}),)
    for name, options in variants:
        torch.cuda.empty_cache()
        slots, coarse, lse, output = evaluate(options)
        results.append(dict(
            variant=name,
            changed_route_rows=int((slots != reference[0]).any(-1).sum()),
            coarse_bitwise_equal=torch.equal(coarse, reference[1]),
            coarse_lse_bitwise_equal=torch.equal(lse, reference[2]),
            output_bitwise_equal=torch.equal(output, reference[3]),
            max_output_difference=float((output.float() - reference[3].float()).abs().max()),
        ))
    return {"layer": snapshot["layer"], "query_heads": q.numel() // q.size(-1), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial")
    args = parser.parse_args()
    result = audit(args.snapshot, args.trial)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
