"""Graph the whole fixed-shape projected K3 attention body, including local.

Stage diagnostic only. Queries/weights/remote membership come from a trained
capture; local/sink records are cyclically reused to complete that geometry,
not a new model sequence. No FFN/MoE or cache construction is timed here.
Input copies are included in both eager and replay timings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from benchmarks.kimi_k3_update_graph import reconstruct_centroids
from lod_attention._config import LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._profile import configure_engine
from lod_attention.kernels.aiter_mla_prefill_attention import aiter_kimi_local_prefill_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LOD_KIMI_TILE_REFINE"] = "1"
    os.environ["LOD_KIMI_DIRECT_LEAF_RESULT"] = "1"
    os.environ["LOD_KIMI_CACHE_PROJECTION_WEIGHTS"] = "0"
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    sums, counts, _ = reconstruct_centroids(payload)
    q_source = payload["q"].cuda().contiguous()
    batch, heads, queries, _ = q_source.shape
    assert batch == 1
    key_source = sums[None, None].cuda().bfloat16()
    count_source = counts[None, None].cuda()
    uk_source, uv_source = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    page_source = {key: value.cuda() for key, value in payload["cache"].items()}
    lookback = 256
    local_source = page_source["leaf_k"].index_select(
        2, torch.arange(queries + lookback, device="cuda") % payload["leaf_count"],
    )
    sink_source = page_source["leaf_k"][..., :1, :].contiguous()
    sources = (q_source, key_source, count_source, uk_source, uv_source, local_source, sink_source)
    static = tuple(t.clone() for t in sources)
    q, keys, weights, uk, uv, local, sink = static
    page = {key: value.clone() for key, value in page_source.items() if key != "leaf_v"}
    page["leaf_v"] = page["leaf_k"][..., :512]
    page["leaf_count"] = payload["leaf_count"]
    page["overflow_active"] = payload["hash_probes"] != 0
    carrier = local[..., lookback:, :].expand(-1, heads, -1, -1)
    engine = KernelTwoLevelLODAttention(query_heads=heads, key_value_heads=1, scale=payload["scale"])
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=65536, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    engine.leaf_hash_probes = payload["hash_probes"]
    engine._lod_kimi_w_uk_t, engine._lod_kimi_w_uv = uk, uv
    engine._lod_kimi_expanded_prefill_chunk = q
    # Graph inputs include layer weights, so no flattened layout may freeze
    # an earlier layer's values between replays.
    engine._lod_kimi_mutable_projection_weights = True
    local_stream = torch.cuda.Stream()

    def refresh():
        for destination, source in zip(static, sources, strict=True):
            destination.copy_(source)
        for name, source in page_source.items():
            if name != "leaf_v":
                page[name].copy_(source)

    def body():
        local_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(local_stream):
            branch = aiter_kimi_local_prefill_attention(
                carrier, local, query_offset=lookback, scale=payload["scale"],
                expanded_q=q, w_uk_t=uk, w_uv=uv,
                buffers=engine._lod_prefill_attention_buffers,
            )
        engine._lod_prefill_local_stream_pending = local_stream
        return engine._two_level_attention(
            carrier, local, local[..., :512], keys, keys[..., :512], weights,
            None, page["leaf_k"], page["leaf_v"], state_len=sums.size(0),
            state_capacity=sums.size(0), page_cache=page, local_branch=branch,
            sink_k=sink, sink_v=sink[..., :512], context_len=65536,
        )

    def eager():
        refresh()
        return body()

    with torch.inference_mode():
        eager_before = timed(eager)
        reference = eager().clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
            body()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        print("KIMI_ATTENTION_GRAPH stage=capture", flush=True)
        with torch.cuda.graph(graph, stream=stream):
            captured = body()
        torch.cuda.current_stream().wait_stream(stream)

        def replay():
            refresh()
            graph.replay()

        graph_timing = timed(replay)
        replay()
        torch.testing.assert_close(captured, reference, atol=0, rtol=0)
        original_sources = tuple(t.clone() for t in sources)
        original_leaves = page_source["leaf_k"].clone()
        q_source.mul_(-0.75)
        key_source.mul_(1.25)
        uk_source.mul_(1.5)
        uv_source.mul_(-0.25)
        local_source.mul_(0.75)
        page_source["leaf_k"].mul_(1.125)
        fresh = eager().clone()
        replay()
        torch.testing.assert_close(captured, fresh, atol=0, rtol=0)
        for destination, source in zip(sources, original_sources, strict=True):
            destination.copy_(source)
        page_source["leaf_k"].copy_(original_leaves)
        eager_after = timed(eager)
    result = {"scope": "whole fixed attention GPU stage, not model latency",
              "source": str(args.input), "query_shape": list(q.shape),
              "state_len": sums.size(0), "leaf_count": payload["leaf_count"],
              "local_geometry": "256 + Q cyclically reused trained records",
              "input_copies_included": True, "fresh_inputs_and_weights": "bitwise matched",
              "same_input_timing_controls": True,
              "eager_before": eager_before, "graph": graph_timing, "eager_after": eager_after}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
