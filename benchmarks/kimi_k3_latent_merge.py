"""Controlled shared-latent update on captured, trained K3 records.

This times the complete state-update GPU stage, not model wall time. Cyclic
reuse of the trained records fills the fixed 16K update geometry. No routing
policy, append budget, channel or cadence changes between the controls.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.kimi_k3_intermediate_graph import check_membership
from benchmarks.kimi_k3_leaf_replay import timed
from lod_attention._config import LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._profile import configure_engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--stable-ties", action="store_true",
                        help="diagnostic stable append ties; not a serving policy change")
    parser.add_argument("--merge-token-block", type=int, choices=(1, 2, 4, 8, 16), default=1)
    parser.add_argument("--merge-state-block", type=int, choices=(1, 2, 4, 8), default=1)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    source = payload["cache"]["leaf_k"][0, 0].cuda()
    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=1)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=65536, has_query_norm=True, has_key_norm=False)
    if args.stable_ties:
        def split(scores, n_append):
            ordered = torch.argsort(scores.float(), dim=-1, stable=True)
            return ordered[..., :n_append].sort().values, ordered[..., n_append:].sort().values
        engine._split_append_merge_indices = split
    capacity, state_len = engine._state_capacity(65536, 255), 255
    keys = torch.zeros(args.layers, 1, capacity, 576, dtype=torch.bfloat16, device="cuda")
    keys[..., :state_len, :].copy_(source[:state_len])
    counts = torch.zeros(args.layers, 1, capacity, 1, device="cuda")
    counts[..., :state_len, :].fill_(1)
    points, previous_coverage = {}, 256
    with torch.inference_mode():
        for context in (16384, 32768):
            coverage = context - 256
            indices = torch.arange(previous_coverage - 1, coverage - 1, device="cuda")
            overflow = source[indices % source.size(0)][None, None].repeat(args.layers, 1, 1, 1)
            initial_k, initial_counts = keys.clone(), counts.clone()
            original_overflow = overflow.clone()
            options = dict(state_len=state_len, ctx_len=context, available_context=coverage,
                           state_capacity=capacity, scheduled_state_len=state_len,
                           clustering_query_scale=None, retain_prepared_geometry=False)

            def restore():
                keys.copy_(initial_k)
                counts.copy_(initial_counts)

            def update():
                restore()
                return engine._update_state(keys, keys[..., :512], counts, None,
                                            overflow, overflow[..., :512], **options)

            os.environ["LOD_KIMI_SHARED_LATENT_MERGE"] = "0"
            os.environ["LOD_KIMI_MERGE_TOKEN_BLOCK"] = "1"
            os.environ["LOD_KIMI_MERGE_STATE_BLOCK"] = "1"
            before = timed(update)
            expected = update()
            reference = (expected[0].clone(), expected[2].clone(), expected[4].clone())
            original_check = check_membership(initial_k, initial_counts, overflow, expected)
            repeated = update()
            repeated_owners = repeated[4].clone()
            repeat_owner_mismatches = int(repeated_owners.ne(reference[2]).sum())
            os.environ["LOD_KIMI_SHARED_LATENT_MERGE"] = "1"
            os.environ["LOD_KIMI_MERGE_TOKEN_BLOCK"] = str(args.merge_token_block)
            os.environ["LOD_KIMI_MERGE_STATE_BLOCK"] = str(args.merge_state_block)
            candidate = timed(update)
            actual = update()
            candidate_check = check_membership(initial_k, initial_counts, overflow, actual)
            torch.testing.assert_close(actual[2], reference[1], atol=0, rtol=0)
            torch.testing.assert_close(actual[4], reference[2], atol=0, rtol=0)
            relative = ((actual[0].float() - reference[0].float()).norm()
                        / reference[0].float().norm().clamp_min(1e-12)).item()
            if relative > 0.005:
                raise AssertionError("shared update differs beyond atomic summation roundoff")
            # Capture only after complete warmup, then alter data in place.
            restore()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = engine._update_state(keys, keys[..., :512], counts, None,
                                                overflow, overflow[..., :512], **options)

            def replay():
                restore()
                graph.replay()

            graph_timing = timed(replay)
            overflow.mul_(-0.75)
            fresh = update()
            fresh_owners, fresh_counts = fresh[4].clone(), fresh[2].clone()
            replay()
            fresh_check = check_membership(initial_k, initial_counts, overflow, captured)
            torch.testing.assert_close(captured[4], fresh_owners, atol=0, rtol=0)
            torch.testing.assert_close(counts, fresh_counts, atol=0, rtol=0)
            del captured, graph
            overflow.copy_(original_overflow)
            os.environ["LOD_KIMI_SHARED_LATENT_MERGE"] = "0"
            os.environ["LOD_KIMI_MERGE_TOKEN_BLOCK"] = "1"
            os.environ["LOD_KIMI_MERGE_STATE_BLOCK"] = "1"
            after = timed(update)
            state_len = update()[3]
            points[str(context)] = dict(
                input_state_len=options["state_len"], output_state_len=state_len,
                ordinary_before=before, shared_latent=candidate, ordinary_after=after,
                shared_latent_graph=graph_timing, state_relative_l2_error=relative,
                owners_counts_exact=True, ordinary_check=original_check,
                candidate_check=candidate_check, fresh_graph_check=fresh_check,
                ordinary_repeat_owner_mismatches=repeat_owner_mismatches,
            )
            previous_coverage = coverage
    result = dict(scope=__doc__, source=str(args.input), layers=args.layers,
                  global_update_len=16384, stable_tie_diagnostic=args.stable_ties,
                  merge_token_block=args.merge_token_block,
                  merge_state_block=args.merge_state_block,
                  measurements=points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
