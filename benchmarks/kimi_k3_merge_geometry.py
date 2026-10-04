"""Tune MLA's atomic merge grid on captured trained leaf ownership.

Membership is held fixed and every consumed state is checked against FP32
leaf sums. No append ranking, query routing, cadence or cap is changed.
Ordinary controls restore the same initial cache on either side.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention.kernels.lod_kernels import merge_state_in_place, new_state_delta_buffers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tiles", nargs="+", default=["1x1", "2x2", "4x4", "8x4", "16x4"],
                        help="merge source tokens x apply state slots per workgroup")
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, expected_counts, leaves = reconstruct_centroids(payload)
    states, tokens = sums.size(0), leaves.size(0)
    cache = payload["cache"]
    membership = torch.empty(tokens, dtype=torch.long)
    for slot in range(states):
        remaining = int(cache["slot_lengths"][0, 0, slot])
        for ordinal in range((remaining + 15) // 16):
            directory = int(cache["slot_pages"][0, 0, slot, ordinal // 64])
            page = int(cache["overflow_page_values"][0, 0, directory, ordinal % 64])
            ids = cache["page_indices"][0, 0, page, :min(remaining, 16)].long()
            membership[ids] = slot
            remaining -= ids.numel()
    source = leaves[None, None].cuda().bfloat16().repeat(args.layers, 1, 1, 1)
    values = source[..., :512].contiguous()
    destinations = membership[None, None].cuda().repeat(args.layers, 1, 1)
    indices = torch.arange(tokens, device="cuda")[None, None].repeat(args.layers, 1, 1)
    owners = torch.empty_like(indices)
    counts = torch.zeros(args.layers, 1, states, 1, device="cuda")
    state_k = torch.zeros(args.layers, 1, states, 576, dtype=torch.bfloat16, device="cuda")
    state_v = torch.zeros(args.layers, 1, states, 512, dtype=torch.bfloat16, device="cuda")
    scratch = new_state_delta_buffers(state_k, state_v, states)
    merge_counts = torch.ones(args.layers, 1, tokens, device="cuda")
    reference = sums.cuda().bfloat16()

    def merge():
        state_k.zero_()
        state_v.zero_()
        counts.zero_()
        merge_state_in_place(state_k, state_v, counts, source, values, merge_counts,
                             indices, destinations, owners, scratch, active_slots=states)

    def select(tile):
        token, state = tile.split("x")
        os.environ["LOD_KIMI_MERGE_TOKEN_BLOCK"] = token
        os.environ["LOD_KIMI_MERGE_STATE_BLOCK"] = state

    def check():
        torch.testing.assert_close(counts, expected_counts[None, None].cuda().expand_as(counts),
                                   atol=0, rtol=0)
        torch.testing.assert_close(owners, destinations, atol=0, rtol=0)
        difference = (state_k.float() - reference).float()
        error = difference.norm() / reference.float().norm() / args.layers**0.5
        if error.item() > 0.002:
            raise RuntimeError(f"merge relative error is too large: {error.item()}")
        torch.testing.assert_close(state_v, state_k[..., :512], atol=0.02, rtol=0.02)
        return {"relative_l2_vs_fp32_sums_rounded_bf16": error.item(),
                "counts_and_membership_exact": True}

    with torch.inference_mode():
        select("1x1")
        before = timed(merge)
        variants = {}
        for tile in args.tiles:
            select(tile)
            variants[tile] = {**timed(merge), **check()}
        select("1x1")
        after = timed(merge)
    result = {"scope": "trained-ownership atomic merge GPU stage, not model latency",
              "source": str(args.input), "source_shape": list(source.shape),
              "state_len": states, "largest_centroid": expected_counts.max().item(),
              "ordinary_before": before, "tiles": variants, "ordinary_after": after,
              "initial_cache_restores_included": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
