"""Tune exact leaf tiling on a captured, trained Kimi prefill input.

Serial, warmed kernel timings only: these are not end-to-end model timings.
The capture is made by the separate ProLong diagnostic pass, never by a
canonical timed generation. Routes, leaf ownership, and scores stay fixed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv
from lod_attention.kernels.paged_prefill import paged_leaf_attention


def timed(function, repetitions=5):
    function()
    function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", nargs="+",
                        default=["32x16x1", "32x32x1", "64x32x2",
                                 "64x64x2", "128x64x4", "128x128x4"])
    parser.add_argument("--graph", action="store_true",
                        help="also test graph capture of this fixed-input leaf stage")
    parser.add_argument("--scalar-page-lookup", action="store_true",
                        help="compare one directory lookup per aligned page with the baseline")
    parser.add_argument("--sorted-counts", action="store_true",
                        help="compare atomics-free sorted route ordinals")
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if payload["scope"] != "real trained Kimi K3 late-prefill leaf inputs":
        raise ValueError("replay requires an actual trained-prefill capture")
    cpu_slots = payload["top_slots"]
    cpu_lengths = payload["cache"]["slot_lengths"][..., :payload["active_slots"]]
    valid = cpu_slots.ge(0)
    opened_lengths = torch.gather(
        cpu_lengths[:, :1].expand(-1, cpu_slots.size(1), -1),
        2, cpu_slots.clamp_min(0).flatten(-2),
    ).reshape_as(cpu_slots)
    opened_lengths = torch.where(valid, opened_lengths, 0)
    result = {
        "scope": "serial warmed leaf kernels on real trained prefill; not model wall time",
        "input": str(args.input), "query_shape": list(payload["q"].shape),
        "active_slots": payload["active_slots"], "leaf_count": payload["leaf_count"],
        "open_routes_per_query_mean": valid.float().sum(-1).mean().item(),
        "opened_leaves_per_query_mean": opened_lengths.sum(-1).float().mean().item(),
        "opened_centroid_length_mean": opened_lengths[valid].float().mean().item(),
        "tiles": {},
    }
    q = payload["q"].cuda().contiguous()
    slots = cpu_slots.cuda().contiguous()
    w_uk_t, w_uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    cache = {key: value.cuda() for key, value in payload["cache"].items()}
    projection_buffers = {}
    from lod_attention.kernels import paged_prefill
    original_counts = paged_prefill.count_expert_routes
    if args.sorted_counts:
        from lod_attention.kernels.kimi_sorted_route_counts import count_sorted_kimi_routes

    def project():
        return expand_kimi_leaf_kv(cache["leaf_k"], w_uk_t, w_uv,
                                   buffers=projection_buffers)

    with torch.inference_mode():
        result["projection"] = timed(project)
        k, v = project()
        baseline = None
        for tile in args.tiles:
            block_m, block_n, warps = map(int, tile.split("x"))
            point_name = tile
            occurrence = 1
            while point_name in result["tiles"]:
                occurrence += 1
                point_name = f"{tile}_{occurrence}"
            buffers = {}

            def attend():
                return paged_leaf_attention(
                    q, k, v, cache["slot_pages"], cache["overflow_page_keys"],
                    cache["overflow_page_values"], cache["overflow_used"],
                    cache["slot_lengths"], slots, page_indices=cache["page_indices"],
                    kv_group_size=1, active_slots=payload["active_slots"],
                    scale=payload["scale"], hash_probes=payload["hash_probes"],
                    block_m=block_m, block_n=block_n, num_warps=warps,
                    reduce_routes=payload["reduce_routes"], buffers=buffers,
                    scalar_page_lookup=scalar_lookup,
                )

            try:
                paged_prefill.count_expert_routes = original_counts
                scalar_lookup = False
                point = timed(attend)
                if args.graph:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_outputs = attend()
                    point["fixed_input_graph"] = timed(graph.replay)
                    # Retain captured outputs and their graph-private storage
                    # until all replays finish. This tests only a fixed-input
                    # attention stage, not dynamic serving or whole-model
                    # prefill graph compatibility.
                    del graph_outputs, graph
                out, lse = attend()
                if baseline is None:
                    if tile != "32x16x1":
                        raise ValueError("the first tile must be the current 32x16x1 baseline")
                    baseline = out.clone(), lse.clone()
                if args.scalar_page_lookup:
                    reference = out.clone(), lse.clone()
                    scalar_lookup = True
                    point["scalar_page_lookup"] = timed(attend)
                    scalar_out, scalar_lse = attend()
                    torch.testing.assert_close(scalar_out, reference[0], atol=0, rtol=0,
                                               equal_nan=True)
                    torch.testing.assert_close(scalar_lse, reference[1], atol=0, rtol=0,
                                               equal_nan=True)
                    point["scalar_page_lookup"]["bitwise_output_match"] = True
                    scalar_lookup = False
                    point["scalar_baseline_after"] = timed(attend)
                    point["scalar_control_over_candidate"] = (
                        (point["median_ms"] + point["scalar_baseline_after"]["median_ms"])
                        / 2 / point["scalar_page_lookup"]["median_ms"]
                    )
                if args.sorted_counts:
                    scalar_lookup = bool(args.scalar_page_lookup)
                    reference = tuple(t.clone() for t in attend())
                    paged_prefill.count_expert_routes = count_sorted_kimi_routes
                    point["sorted_counts"] = timed(attend)
                    candidate_out, candidate_lse = attend()
                    torch.testing.assert_close(candidate_out, reference[0], atol=0, rtol=0,
                                               equal_nan=True)
                    torch.testing.assert_close(candidate_lse, reference[1], atol=0, rtol=0,
                                               equal_nan=True)
                    point["sorted_counts"]["bitwise_output_match"] = True
                    paged_prefill.count_expert_routes = original_counts
                    point["sorted_control_after"] = timed(attend)
                    scalar_lookup = False
                mask = slots.ge(0)
                if payload["reduce_routes"]:
                    mask = mask.any(-1)
                lhs, rhs = out[mask].float(), baseline[0][mask].float()
                error = lhs - rhs
                point.update(
                    output_relative_l2=(error.square().sum()
                                        / rhs.square().sum().clamp_min(1e-20)).sqrt().item(),
                    output_max_abs=error.abs().max().item(),
                    lse_max_abs=(lse[mask] - baseline[1][mask]).abs().max().item(),
                    output_finite=bool(lhs.isfinite().all()),
                )
                if not point["output_finite"] or point["output_relative_l2"] > 0.01:
                    raise RuntimeError(f"leaf replay correctness failed: {point}")
                del lhs, rhs, error
            except Exception as error:
                point = {"error": str(error)}
            finally:
                paged_prefill.count_expert_routes = original_counts
            result["tiles"][point_name] = point
            print("KIMI_LEAF_REPLAY " + json.dumps({"tile": point_name, **point}), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    errors = {name: point["error"] for name, point in result["tiles"].items() if "error" in point}
    if errors:
        raise RuntimeError(f"leaf replay failed for {list(errors)}; see {args.output}")


if __name__ == "__main__":
    main()
