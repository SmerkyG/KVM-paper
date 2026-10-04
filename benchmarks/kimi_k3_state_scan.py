"""Compare score-materializing BLAS and tiled exact D576 state assignment.

The overflow repeats real trained latent records and centroids retain their
captured memberships. This is a construction-stage test, not a new text run.
The score dtype and smallest-index tie rule remain BF16/ascending index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention.kernels.lod_kernels import new_state_maxsim_buffers, tiled_dot_maxsim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tiles", nargs="+", default=["64x128x8x1"],
                        help="query x centroid x warps x pipeline-stages")
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, counts, leaves = reconstruct_centroids(payload)
    overflow = leaves[torch.arange(16384) % leaves.size(0)].cuda().bfloat16()
    overflow = torch.nn.functional.normalize(overflow.float(), dim=-1).bfloat16()
    centroids = torch.nn.functional.normalize(sums.cuda(), dim=-1).bfloat16()
    overflow = overflow[None, None].repeat(args.layers, 1, 1, 1)
    centroids = centroids[None, None].repeat(args.layers, 1, 1, 1)
    output = torch.empty(args.layers, 1, 16384, centroids.size(2),
                         dtype=torch.bfloat16, device="cuda")
    scratch = new_state_maxsim_buffers(overflow, 16384)

    def dense():
        torch.matmul(overflow, centroids.transpose(-1, -2), out=output)
        return output.max(-1)

    with torch.inference_mode():
        before = timed(dense)
        expected = tuple(t.clone() for t in dense())
        variants = {}
        for geometry in args.tiles:
            block_m, block_n, warps, stages = map(int, geometry.split("x"))

            def tiled():
                return tiled_dot_maxsim(overflow, centroids, scratch, prefix="route",
                    block_m=block_m, block_n=block_n, num_warps=warps, k_stages=stages)

            candidate = timed(tiled)
            actual = tuple(t.clone() for t in tiled())
            torch.testing.assert_close(actual[0], expected[0], atol=0.002, rtol=0.002)
            selected_dense = output.gather(-1, actual[1][..., None])[..., 0]
            variants[geometry] = {
                **candidate, "identical_owner_fraction": actual[1].eq(expected[1]).float().mean().item(),
                "dense_score_tie_fraction": selected_dense.eq(expected[0]).float().mean().item(),
                "max_score_absolute_difference": (actual[0].float() - expected[0].float()).abs().max().item(),
            }
        after = timed(dense)
        result = {
            "scope": "trained-record state-assignment GPU stage, not model latency",
            "source": str(args.input), "overflow_shape": list(overflow.shape),
            "centroid_shape": list(centroids.shape), "includes_all_576_channels": True,
            "ordinary_before": before, "tiles": variants, "ordinary_after": after,
            "dense_score_workspace_bytes": output.numel() * output.element_size(),
            "tiled_score_workspace_bytes": sum(t.numel() * t.element_size()
                for name, t in scratch.items() if "tile_" in name),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
