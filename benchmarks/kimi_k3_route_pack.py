"""Compare fixed-layout tile packing on captured trained K3 prefill data.

GPU-stage timings only, with ordinary controls around the candidate. Routes,
coarse attention, selected scores and projection math must remain identical.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=("dense-pack", "chunk-pack", "chunk512", "chunk1024", "refine16", "refine32", "refine128", "merge-lists", "coarse64", "coarsek64", "persistent-fragments"),
                        default="dense-pack")
    args = parser.parse_args()
    os.environ["LOD_KIMI_TILE_REFINE"] = "1"
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, counts, _ = reconstruct_centroids(payload)
    q = payload["q"].cuda().contiguous()
    keys = sums[None, None].cuda().bfloat16()
    weights = counts[None, None].cuda()
    lengths = weights[..., 0].int()
    uk, uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    buffers = {}

    def run():
        routes, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
            q, keys, keys[..., :512], weights, uk, uv,
            state_len=sums.size(0), scale=payload["scale"],
            normalize_route_query=False, slot_lengths=lengths,
            max_open_leaf_tokens=1024, buffers=buffers,
        )
        torch.cuda.current_stream().wait_stream(coarse.ready_stream)
        return routes, coarse.output_0, coarse.lse_0, coarse.selected_route_scores

    result = {"scope": "trained-geometry coarse/route stage; not model latency",
              "source": str(args.input), "query_shape": list(q.shape),
              "candidate": args.candidate,
              "state_len": sums.size(0), "state_sums": "reconstructed from captured memberships"}
    with torch.inference_mode():
        os.environ["LOD_KIMI_DENSE_TILE_PACK"] = "0"
        persistent = args.candidate == "persistent-fragments"
        os.environ["LOD_KIMI_FIXED_FRAGMENT_RESCORE"] = "0"
        os.environ["LOD_KIMI_CHUNK_TILE_PACK"] = "1" if persistent else "0"
        os.environ["LOD_KIMI_TILE_PACK_QUERY_BLOCK"] = "1024" if persistent else "256"
        os.environ["LOD_KIMI_REFINE_BLOCK_M"] = "64"
        os.environ["LOD_KIMI_KWAY_REDUCE"] = "0"
        os.environ["LOD_KIMI_COARSE_QUERY_TILE"] = "128"
        os.environ["LOD_KIMI_COARSE_KEY_STEP"] = "32"
        result["ordinary_before"] = timed(run)
        reference = tuple(t.clone() for t in run())
        os.environ["LOD_KIMI_DENSE_TILE_PACK"] = "1" if args.candidate == "dense-pack" else "0"
        os.environ["LOD_KIMI_CHUNK_TILE_PACK"] = "1" if persistent or args.candidate.startswith("chunk") else "0"
        os.environ["LOD_KIMI_FIXED_FRAGMENT_RESCORE"] = "1" if persistent else "0"
        if args.candidate in ("chunk512", "chunk1024"):
            os.environ["LOD_KIMI_TILE_PACK_QUERY_BLOCK"] = args.candidate.removeprefix("chunk")
        os.environ["LOD_KIMI_KWAY_REDUCE"] = "1" if args.candidate == "merge-lists" else "0"
        os.environ["LOD_KIMI_COARSE_QUERY_TILE"] = "64" if args.candidate == "coarse64" else "128"
        os.environ["LOD_KIMI_COARSE_KEY_STEP"] = "64" if args.candidate == "coarsek64" else "32"
        if args.candidate.startswith("refine"):
            os.environ["LOD_KIMI_REFINE_BLOCK_M"] = args.candidate.removeprefix("refine")
        result[args.candidate.replace("-", "_")] = timed(run)
        for actual, expected in zip(run(), reference, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        original_q, original_keys = q.clone(), keys.clone()
        q.mul_(-0.75)
        keys.mul_(1.25)
        candidate_fresh = tuple(t.clone() for t in run())
        os.environ["LOD_KIMI_DENSE_TILE_PACK"] = "0"
        os.environ["LOD_KIMI_FIXED_FRAGMENT_RESCORE"] = "0"
        os.environ["LOD_KIMI_CHUNK_TILE_PACK"] = "1" if persistent else "0"
        os.environ["LOD_KIMI_TILE_PACK_QUERY_BLOCK"] = "1024" if persistent else "256"
        os.environ["LOD_KIMI_REFINE_BLOCK_M"] = "64"
        os.environ["LOD_KIMI_KWAY_REDUCE"] = "0"
        os.environ["LOD_KIMI_COARSE_QUERY_TILE"] = "128"
        os.environ["LOD_KIMI_COARSE_KEY_STEP"] = "32"
        for actual, expected in zip(run(), candidate_fresh, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        q.copy_(original_q)
        keys.copy_(original_keys)
        result["ordinary_after"] = timed(run)
    result["fresh_input_routes_outputs_scores"] = "bitwise matched"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
