"""Test graph replay of complete final DCP cache construction.

This is a stage-only diagnostic, not a model timing. The input consists of
rank-owned, cyclically extended real trained records, not a new natural text
sequence. Input copies are included. It retains the serving 16K/256 global
cadences, divided only when expressing their rank-owned record counts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention._profile import configure_engine
from lod_attention._config import LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention.kernels.kimi_prefill_graph import KimiFinalCacheGraphs


def check_cache(cache, source):
    """Check fresh records, disjoint membership, counts and centroid means.

    Equal append scores may produce different valid choices in eager and
    graph execution, as they can between two ordinary eager executions. Test
    the represented calculation rather than requiring one arbitrary tie order.
    """
    state = cache.state
    page = state["page_cache"]
    leaf_count = int(page["leaf_count"])
    state_len = int(state["state_len"])
    cpu = {name: value.cpu() if isinstance(value, torch.Tensor) else value
           for name, value in page.items()}
    source = source.cpu()
    sink_len = int(state["sink_k"].size(2))
    torch.testing.assert_close(state["sink_k"].cpu(), source[..., :sink_len, :], atol=0, rtol=0)
    torch.testing.assert_close(cpu["leaf_k"][..., :leaf_count, :],
                               source[..., sink_len:sink_len + leaf_count, :], atol=0, rtol=0)
    recent_len = int(state["recent_len"])
    torch.testing.assert_close(state["recent_k"][..., :recent_len, :].cpu(),
                               source[..., int(state["coverage"]):, :], atol=0, rtol=0)
    errors = []
    counts = state["counts"].cpu()[..., :state_len, :]
    keys = state["state_k"].cpu()[..., :state_len, :]
    for row in range(source.size(0)):
        payload = {"hash_probes": -1, "active_slots": state_len, "cache": {
            name: value[row:row + 1, :, :leaf_count] if name == "leaf_k"
            else value[row:row + 1] if isinstance(value, torch.Tensor) and value.ndim >= 2
            else value for name, value in cpu.items()
        }}
        sums, lengths, _ = reconstruct_centroids(payload)
        torch.testing.assert_close(counts[row, 0], lengths, atol=0, rtol=0)
        # BF16 incremental accumulation is not an exact FP32 sum. Its error
        # must remain small and, crucially, correspond to these fresh leaves.
        error = float((keys[row, 0].float() - sums).norm() / sums.norm().clamp_min(1e-12))
        if error > 0.025:
            raise AssertionError(f"centroid sums do not represent fresh leaves: rel L2 {error}")
        errors.append(error)
    return {"fresh_records_membership_counts": "passed", "state_sum_relative_l2": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-length", type=int, default=65536)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--shared-manager", action="store_true")
    args = parser.parse_args()
    world = 8
    if args.global_length % 16384 or args.layers < 1:
        parser.error("use an aligned global prefix and a positive layer count")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    leaves = payload["cache"]["leaf_k"][0, 0]
    local_length = args.global_length // world
    source = leaves[torch.arange(local_length) % leaves.size(0)].cuda()
    source = source[None, None].repeat(args.layers, 1, 1, 1).contiguous()
    static = source.clone()
    engine = KernelTwoLevelLODAttention(query_heads=12, key_value_heads=1, scale=192**-0.5)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=args.global_length, has_query_norm=True, has_key_norm=False)
    # Express the unchanged global schedule in DCP8-owned records, exactly
    # as pool._dcp_local_state_schedule does. This does not stretch cadence.
    engine.state_growth_factor /= math.sqrt(world)
    engine.state_min_len = math.ceil(engine.state_min_len / world)
    for name in ("chunk_len", "local_len", "prefill_chunk_len", "prefill_local_len",
                 "prefill_state_update_len", "decode_state_update_len"):
        setattr(engine, name, math.ceil(getattr(engine, name) / world))
    global_coverage = max(0, math.ceil((args.global_length + 1) / 256) * 256 - 512)
    options = dict(finalize_cache_for_decode=True,
                   final_cache_coverage=global_coverage // world)

    def build():
        static.copy_(source)
        return engine.build_cache_from_bf16(static, static[..., :512], **options)

    with torch.inference_mode():
        print("KIMI_CACHE_GRAPH stage=eager", flush=True)
        eager_before = timed(build)
        eager_check = check_cache(build(), source)
        print("KIMI_CACHE_GRAPH stage=capture", flush=True)
        if args.shared_manager:
            manager = KimiFinalCacheGraphs()

            def replay():
                return manager.run(engine, source, source[..., :512],
                                   final_cache_coverage=options["final_cache_coverage"])

            captured = replay()
            if len(manager.entries) != 1 or manager.fallback_count:
                raise AssertionError("requested cache graph fell back to ordinary construction")
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = build()
            replay = graph.replay
        graph_timing = timed(replay)
        replay()
        graph_check = check_cache(captured, source)
        source.mul_(-0.75)
        replay()
        fresh_check = check_cache(captured, source)
        eager_after = timed(build)
        check_cache(build(), source)
    result = {"scope": "complete final rank-local cache construction; not model latency",
              "input": str(args.input), "global_length": args.global_length,
              "dcp_world_size": world, "layers": args.layers,
              "input_shape": list(source.shape), "input_copy_included": True,
              "shared_serving_manager": args.shared_manager,
              "global_prefill_update_len": 16384, "global_decode_update_len": 256,
              "eager_before": eager_before, "graph": graph_timing,
              "eager_after": eager_after, "eager_check": eager_check,
              "graph_check": graph_check, "fresh_input_check": fresh_check}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
