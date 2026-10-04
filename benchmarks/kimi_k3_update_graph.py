"""Measure fixed-shape update replay on real trained K3 latent records.

This is a serial GPU-stage test, not full-model latency or a quality test.
Centroids are reconstructed from the captured archive's actual memberships;
the next overflow reuses these real records to test fresh-input graph replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from lod_attention._config import LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._profile import configure_engine
from benchmarks._kimi_k3_update_graph import DirectStateUpdateGraph, FixedStateUpdateGraph


def reconstruct_centroids(payload):
    cache = payload["cache"]
    if payload["hash_probes"] != -1:
        raise ValueError("this replay expects the captured two-level page directory")
    leaves = cache["leaf_k"][0, 0].float()
    state_len = payload["active_slots"]
    sums = torch.zeros(state_len, leaves.size(-1))
    lengths = cache["slot_lengths"][0, 0, :state_len]
    visited = torch.zeros(leaves.size(0), dtype=torch.bool)
    for slot in range(state_len):
        remaining = int(lengths[slot])
        indices = []
        for ordinal in range((remaining + 15) // 16):
            directory = int(cache["slot_pages"][0, 0, slot, ordinal // 64])
            if directory < 0:
                raise ValueError("capture contains a missing page directory")
            page = int(cache["overflow_page_values"][0, 0, directory, ordinal % 64])
            if page < 0:
                raise ValueError("capture contains a missing page")
            ids = cache["page_indices"][0, 0, page, :min(remaining, 16)].long()
            if ids.min() < 0 or ids.max() >= leaves.size(0) or visited[ids].any():
                raise ValueError("captured leaf ownership is invalid or duplicated")
            visited[ids] = True
            indices.append(ids)
            remaining -= ids.numel()
        if indices:
            sums[slot] = leaves[torch.cat(indices)].sum(0)
    if not visited.all():
        raise ValueError("captured archive is missing centroid membership")
    return sums, lengths.float().unsqueeze(-1), leaves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--stable-ties", action="store_true",
                        help="diagnose append ties using a stable sort, not serving policy")
    parser.add_argument("--direct-workspaces", action="store_true",
                        help="replay on the same caller workspace without extra state copies")
    args = parser.parse_args()
    if args.layers < 1:
        raise ValueError("layers must be positive")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, lengths, leaves = reconstruct_centroids(payload)
    state_len, width = sums.shape
    capacity, overflow_len = 4096, 16_384
    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=1)
    engine.head_dim = width
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=65_536, has_query_norm=False, has_key_norm=False)
    selection_scores = []
    original_split = engine._split_append_merge_indices

    def split(scores, n_append):
        selection_scores[:] = [scores.detach().clone()]
        if not args.stable_ties:
            return original_split(scores, n_append)
        ordered = torch.argsort(scores.float(), dim=-1, stable=True)
        return (ordered[..., :n_append].sort().values,
                ordered[..., n_append:].sort().values)

    engine._split_append_merge_indices = split
    # Match the serving pool's BF16 state storage and FP32 merge accumulation.
    state_k = torch.zeros(args.layers, 1, capacity, width,
                          dtype=torch.bfloat16, device="cuda")
    counts = torch.zeros(args.layers, 1, capacity, 1, device="cuda")
    state_k[..., :state_len, :].copy_(sums.cuda())
    counts[..., :state_len, :].copy_(lengths.cuda())
    source = leaves[torch.arange(overflow_len) % leaves.size(0)].to(torch.bfloat16).cuda()
    overflow = source[None, None].repeat(args.layers, 1, 1, 1)
    initial = state_k.clone(), counts.clone()
    inputs = (state_k, state_k[..., :512], counts, None, overflow, overflow[..., :512])
    options = dict(state_len=state_len, ctx_len=32768, available_context=32512,
                   state_capacity=capacity, scheduled_state_len=state_len,
                   clustering_query_scale=None, retain_prepared_geometry=False)

    def restore():
        state_k.copy_(initial[0])
        counts.copy_(initial[1])

    def eager():
        restore()
        return engine._update_state(*inputs, **options)

    with torch.inference_mode():
        eager_timing = timed(eager)
        reference = eager()
        reference_values = state_k.clone(), counts.clone(), reference[4].clone()
        reference_scores = selection_scores[0].clone()
        repeat_reference = eager()
        eager_repeat = {
            "state_equal": bool(state_k.eq(reference_values[0]).all()),
            "counts_equal": bool(counts.eq(reference_values[1]).all()),
            "owner_mismatches": int(repeat_reference[4].ne(reference_values[2]).sum()),
            "selection_score_mismatches": int(selection_scores[0].ne(reference_scores).sum()),
            "selection_score_max_error": float((selection_scores[0].float()
                                                - reference_scores.float()).abs().max()),
        }
        restore()
        graph_type = DirectStateUpdateGraph if args.direct_workspaces else FixedStateUpdateGraph
        graph = graph_type(engine._update_state, inputs, options)

        def replay():
            restore()
            return graph(*inputs)

        graph_timing = timed(replay)
        replayed = replay()
        replay_values = state_k.clone(), counts.clone(), replayed[4].clone()
        eager_after_timing = timed(eager)
        state_k.copy_(replay_values[0])
        counts.copy_(replay_values[1])
        diagnostic = {
            "scope": "serial fixed-geometry state-update GPU stage; not model latency",
            "source": str(args.input), "layers": args.layers,
            "state_len": state_len, "state_capacity": capacity,
            "overflow_len": overflow_len, "key_dim": width, "value_dim": 512,
            "stable_tie_diagnostic": args.stable_ties,
            "direct_caller_workspaces": args.direct_workspaces,
            "eager": eager_timing, "graph_including_input_output_copies": graph_timing,
            "eager_after": eager_after_timing,
            "eager_over_graph": eager_timing["median_ms"] / graph_timing["median_ms"],
            "eager_repeat": eager_repeat,
            "owner_mismatches": int(replayed[4].ne(reference_values[2]).sum()),
            "counts_mismatches": int(counts.ne(reference_values[1]).sum()),
            "old_state_max_error": float(
                (state_k[..., :state_len, :].float()
                 - reference_values[0][..., :state_len, :].float()).abs().max()),
            "new_state_max_error": float(
                (state_k[..., state_len:, :].float()
                 - reference_values[0][..., state_len:, :].float()).abs().max()),
            "fresh_input_correctness": "pending",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diagnostic, indent=2) + "\n")
        print(json.dumps(diagnostic, indent=2), flush=True)
        torch.testing.assert_close(state_k, reference_values[0], atol=0.01, rtol=0.001)
        torch.testing.assert_close(counts, reference_values[1], atol=0, rtol=0)
        torch.testing.assert_close(replayed[4], reference_values[2], atol=0, rtol=0)
        # Change both state data and overflow after capture. Compare with a
        # fresh eager update; a graph which froze capture-time data must fail.
        initial[0].mul_(1.25)
        overflow.mul_(0.75)
        fresh = eager()
        expected = state_k.clone(), counts.clone(), fresh[4].clone()
        restore()
        fresh_replay = graph(*inputs)
        torch.testing.assert_close(state_k, expected[0], atol=0.01, rtol=0.001)
        torch.testing.assert_close(counts, expected[1], atol=0, rtol=0)
        torch.testing.assert_close(fresh_replay[4], expected[2], atol=0, rtol=0)
    result = diagnostic | {"fresh_input_correctness": "passed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
