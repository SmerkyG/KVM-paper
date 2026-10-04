"""Test fixed-shape coarse/route replay with fresh real trained inputs.

Benchmark only: reconstructed trained centroids are not the original rounded
serving sums. Input copies and stream dependencies are included; this is not
a model timing or quality result. ``--shared-manager`` tests the opt-in
serving manager; it does not enable graphs in other serving processes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention.kernels.aiter_mla_prefill_attention import (
    aiter_kimi_expanded_prefill_route_coarse_attention,
)
from lod_attention.kernels.kimi_prefill_graph import KimiPrefillCoarseGraphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-manager", action="store_true",
                        help="validate the serving manager across fresh weights and inputs")
    args = parser.parse_args()
    os.environ["LOD_KIMI_TILE_REFINE"] = "1"
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, counts, _ = reconstruct_centroids(payload)
    source_q = payload["q"].cuda().contiguous()
    source_k = sums[None, None].cuda().bfloat16()
    source_counts = counts[None, None].cuda()
    slot_lengths = source_counts[..., 0].int()
    uk, uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    query, keys, weights = source_q.clone(), source_k.clone(), source_counts.clone()
    values = source_k[..., :512].contiguous()
    buffers = {}
    manager = KimiPrefillCoarseGraphs() if args.shared_manager else None

    def eager_run():
        query.copy_(source_q)
        keys.copy_(source_k)
        weights.copy_(source_counts)
        values.copy_(source_k[..., :512])
        routes, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
            query, keys, values, weights, uk, uv,
            state_len=sums.size(0), scale=payload["scale"],
            normalize_route_query=False, slot_lengths=slot_lengths,
            max_open_leaf_tokens=1024, buffers=buffers,
        )
        if coarse.ready_stream is not None:
            torch.cuda.current_stream().wait_stream(coarse.ready_stream)
        return routes, coarse.output_0, coarse.lse_0

    def manager_run():
        # The manager already copies its inputs. Do not add a second layer of
        # copies, and do not accidentally label graph replay as eager timing.
        routes, coarse, _, _ = manager.run(
            source_q, source_k, source_k[..., :512], source_counts, uk, uv,
            state_len=sums.size(0), scale=payload["scale"],
            normalize_route_query=False, slot_lengths=slot_lengths,
            max_open_leaf_tokens=1024,
        )
        return routes, coarse.output_0, coarse.lse_0

    with torch.inference_mode():
        print("KIMI_COARSE_GRAPH stage=eager-warmup", flush=True)
        eager_before = timed(eager_run)
        reference = tuple(t.clone() for t in eager_run())
        graph = None
        if manager is None:
            graph = torch.cuda.CUDAGraph()
            print("KIMI_COARSE_GRAPH stage=capture", flush=True)
            with torch.cuda.graph(graph):
                captured = eager_run()
            replay = graph.replay
        else:
            captured = manager_run()
            replay = manager_run
        graph_timing = timed(replay)
        replay()
        for actual, expected in zip(captured, reference, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # The copies are part of the graph. Change source data in place and
        # compare with eager computation so frozen capture-time inputs fail.
        source_q.mul_(-0.75)
        source_k.mul_(1.25)
        if manager is not None:
            uk.mul_(1.5)
            uv.mul_(-0.25)
            ref_routes, ref_coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
                source_q, source_k, source_k[..., :512].contiguous(), source_counts, uk, uv,
                state_len=sums.size(0), scale=payload["scale"],
                normalize_route_query=False, slot_lengths=slot_lengths,
                max_open_leaf_tokens=1024, buffers={},
            )
            torch.cuda.current_stream().wait_stream(ref_coarse.ready_stream)
            fresh = tuple(t.clone() for t in (ref_routes, ref_coarse.output_0, ref_coarse.lse_0))
        else:
            fresh = tuple(t.clone() for t in eager_run())
        replay()
        for actual, expected in zip(captured, fresh, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Layouts are immutable in normal serving, but this test changed the
        # original weights in place. Rebuild only the eager control's cache.
        if manager is not None:
            buffers.clear()
        eager_after = timed(eager_run)
    result = {
        "scope": "serial trained-geometry coarse/route GPU stage, not model latency",
        "source": str(args.input), "query_shape": list(source_q.shape),
        "state_len": sums.size(0), "state_sums": "reconstructed from captured memberships",
        "copies_included": True, "fresh_input_checks": "passed",
        "shared_serving_manager": args.shared_manager,
        "fresh_projection_weight_checks": "passed" if manager is not None else "not tested",
        "eager_before": eager_before, "graph": graph_timing, "eager_after": eager_after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
