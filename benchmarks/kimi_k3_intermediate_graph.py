"""Test actual intermediate-update geometry on captured trained K3 records.

The records are cyclically reused to fill the cache/update shapes. This is
not a model-quality or natural-sequence timing. Both paths restore identical
state, and replay pays stable-input/output copies and an owned membership.
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
from lod_attention.kernels.kimi_prefill_graph import KimiStateUpdateGraphs


def check_membership(initial_k, initial_counts, overflow, result):
    key, _, counts, state_len, owners, remap = result
    if remap is not None or owners.min() < 0 or owners.max() >= state_len:
        raise AssertionError("invalid update ownership")
    expected_counts = initial_counts.clone()
    expected_counts.scatter_add_(2, owners[..., None], torch.ones_like(owners[..., None]).float())
    torch.testing.assert_close(counts, expected_counts, atol=0, rtol=0)
    expected_key = initial_k.float().clone()
    expected_key.scatter_add_(2, owners[..., None].expand_as(overflow), overflow.float())
    error = (key.float() - expected_key).norm() / expected_key.norm().clamp_min(1e-12)
    if error.item() > 0.005:
        raise AssertionError(f"updated centroids do not represent current records: {error.item()}")
    return {"membership_counts_exact": True, "relative_l2_key_sum_error": error.item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    source = payload["cache"]["leaf_k"][0, 0].cuda()
    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=1)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=65536, has_query_norm=True, has_key_norm=False)
    capacity, state_len = engine._state_capacity(65536, 255), 255
    keys = torch.zeros(args.layers, 1, capacity, 576, dtype=torch.bfloat16, device="cuda")
    keys[..., :state_len, :].copy_(source[:state_len])
    counts = torch.zeros(args.layers, 1, capacity, 1, device="cuda")
    counts[..., :state_len, :].fill_(1)
    manager = KimiStateUpdateGraphs()
    points = {}
    previous_coverage = 256
    with torch.inference_mode():
        for context in (16384, 32768):
            coverage = context - 256
            overflow = source[torch.arange(previous_coverage - 1, coverage - 1,
                                           device="cuda") % source.size(0)]
            overflow = overflow[None, None].repeat(args.layers, 1, 1, 1).contiguous()
            initial_k, initial_counts = keys.clone(), counts.clone()
            original_overflow = overflow.clone()
            options = dict(state_len=state_len, ctx_len=context, available_context=coverage,
                           state_capacity=capacity, scheduled_state_len=state_len,
                           clustering_query_scale=None, retain_prepared_geometry=False)

            def restore():
                keys.copy_(initial_k)
                counts.copy_(initial_counts)

            def ordinary():
                restore()
                return engine._update_state(keys, keys[..., :512], counts, None,
                                            overflow, overflow[..., :512], **options)

            def replay():
                restore()
                return manager.run(engine, keys, keys[..., :512], counts, None,
                                   overflow, overflow[..., :512], **options)

            before = timed(ordinary)
            ordinary_check = check_membership(initial_k, initial_counts, overflow, ordinary())
            candidate = timed(replay)
            graph_check = check_membership(initial_k, initial_counts, overflow, replay())
            overflow.mul_(-0.75)
            fresh_check = check_membership(initial_k, initial_counts, overflow, replay())
            overflow.copy_(original_overflow)
            after = timed(ordinary)
            # Use the ordinary result as the next update's input. Keep the
            # before/candidate/after source geometry and data exactly matched.
            state_len = ordinary()[3]
            points[str(context)] = {
                "input_state_len": options["state_len"], "output_state_len": state_len,
                "overflow_len": overflow.size(2), "ordinary_before": before,
                "graph_including_copies": candidate, "ordinary_after": after,
                "ordinary_check": ordinary_check, "graph_check": graph_check,
                "fresh_input_check": fresh_check,
            }
            previous_coverage = coverage
    if manager.fallback_count or len(manager.entries) != 2:
        raise AssertionError("fixed intermediate-update geometry did not graph replay")
    result = {"scope": "trained-record fixed-shape intermediate update; not model latency",
              "source": str(args.input), "layers": args.layers,
              "state_capacity": capacity, "global_update_len": 16384,
              "graph_replay_count": manager.replay_count, "measurements": points}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
