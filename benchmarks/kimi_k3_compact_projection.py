"""Compare complete leaf stages on captured trained K3 inputs, not model latency.

Both paths count the same post-cap routes, project, pack queries, attend, and
reduce their LSE. Timings include those stages and exact-shape warmups. A/B/A
controls bracket the candidate; separately reported projection timings are
diagnostics, not additive/exclusive model-time attributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv
from lod_attention.kernels.kimi_compact_leaf_projection import project_compact_kimi_leaves
from lod_attention.kernels.kimi_sorted_route_counts import count_sorted_kimi_routes
from lod_attention.kernels.paged_prefill import paged_leaf_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-tiles", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if payload["scope"] != "real trained Kimi K3 late-prefill leaf inputs":
        raise ValueError("projection comparison requires real trained-prefill inputs")
    q = payload["q"].cuda().contiguous()
    routes = payload["top_slots"].cuda().contiguous()
    cache = {name: tensor.cuda() for name, tensor in payload["cache"].items()}
    uk, uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    source = cache["leaf_k"][..., :payload["leaf_count"], :]
    slots = payload["active_slots"]
    common = dict(page_indices=cache["page_indices"], kv_group_size=1,
                  active_slots=slots, scale=payload["scale"],
                  hash_probes=payload["hash_probes"], block_m=64, block_n=16,
                  num_warps=1, reduce_routes=payload["reduce_routes"], scalar_page_lookup=True)
    directory = (cache["slot_pages"], cache["overflow_page_keys"],
                 cache["overflow_page_values"], cache["overflow_used"], cache["slot_lengths"], routes)
    count_buffers, baseline_buffers, candidate_buffers, attention_buffers = {}, {}, {}, {}
    result = dict(scope="warmed complete leaf stage on captured trained K3 inputs; not full-model latency",
                  source=str(args.input), query_shape=list(q.shape), leaf_count=payload["leaf_count"],
                  active_slots=slots, routes_unchanged=True, projection_tiles={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def count():
        return count_sorted_kimi_routes(routes, active_slots=slots, buffers=count_buffers)

    def attend(k, v, counts, offsets, starts=None):
        return paged_leaf_attention(q, k, v, *directory, **common,
                                   route_head_counts=counts, route_offsets=offsets,
                                   compact_leaf_offsets=starts, buffers=attention_buffers)

    def ordinary():
        counts, offsets = count()
        k, v = expand_kimi_leaf_kv(source, uk, uv, buffers=baseline_buffers)
        return attend(k, v, counts, offsets)

    with torch.inference_mode():
        expected = tuple(t.clone() for t in ordinary())
        valid = routes.ge(0)
        if payload["reduce_routes"]:
            valid = valid.any(-1)
        result["ordinary_before"] = timed(ordinary, repetitions=args.repeats)
        save()
        for tile in args.projection_tiles:
            def project():
                counts, _ = count()
                return project_compact_kimi_leaves(
                    source, uk, uv, cache, counts, active_slots=slots,
                    hash_probes=payload["hash_probes"], block_m=tile, buffers=candidate_buffers)

            def compact():
                counts, offsets = count()
                k, v, starts = project_compact_kimi_leaves(
                    source, uk, uv, cache, counts, active_slots=slots,
                    hash_probes=payload["hash_probes"], block_m=tile, buffers=candidate_buffers)
                return attend(k, v, counts, offsets, starts)

            point = result["projection_tiles"][str(tile)] = {}
            try:
                actual = compact()
                out_diff = actual[0][valid].float() - expected[0][valid].float()
                point["output_relative_l2"] = (
                    out_diff.square().sum() / expected[0][valid].float().square().sum().clamp_min(1e-20)
                ).sqrt().item()
                point["output_max_absolute"] = out_diff.abs().max().item()
                point["lse_max_absolute"] = (actual[1][valid] - expected[1][valid]).abs().max().item()
                if point["output_relative_l2"] > 0.01 or not actual[0][valid].isfinite().all():
                    raise RuntimeError("compact projection output failed correctness")
                point["complete_leaf_stage"] = timed(compact, repetitions=args.repeats)
                point["count_and_projection_only"] = timed(project, repetitions=args.repeats)
                k, v, starts = project()
                counts, _ = count()
                eligible_lengths = cache["slot_lengths"][..., :slots].expand(
                    source.size(0), q.size(1), slots)
                needed = (eligible_lengths * counts.view_as(eligible_lengths).gt(0)).sum().item()
                point.update(selected_leaf_head_pairs=needed,
                             total_leaf_head_pairs=source.size(0) * q.size(1) * source.size(2),
                             compact_rows=int(starts[-1].item()),
                             capacity_rows=k.numel() // 192,
                             reusable_projected_capacity_bytes=(k.numel() + v.numel()) * 2,
                             no_host_count_synchronization=True)
            except Exception as error:
                point["error"] = str(error)
                save()
                raise
            print("KIMI_COMPACT_POINT " + json.dumps({"tile": tile, **point}), flush=True)
            save()
        result["ordinary_after"] = timed(ordinary, repetitions=args.repeats)
        baseline = (result["ordinary_before"]["median_ms"] + result["ordinary_after"]["median_ms"]) / 2
        for point in result["projection_tiles"].values():
            point["control_over_candidate"] = baseline / point["complete_leaf_stage"]["median_ms"]
        result["status"] = "complete"
        save()


if __name__ == "__main__":
    main()
