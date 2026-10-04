"""Test sharing the contiguous query between K3 coarse and fine attention.

Warmed whole-attention stage only, not model wall time. Queries, centroid
memberships and weights are a real trained capture; local/sink records are
cyclically reused to complete the attention geometry. The source query has
the noncontiguous BHQD layout returned by vLLM's token-major Q projection.
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
    os.environ.update(LOD_KIMI_TILE_REFINE="1", LOD_KIMI_DIRECT_LEAF_RESULT="1")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if payload["scope"] != "real trained Kimi K3 late-prefill leaf inputs":
        raise ValueError("requires real trained prefill inputs")
    sums, counts, _ = reconstruct_centroids(payload)
    q = payload["q"].cuda().permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    batch, heads, queries, _ = q.shape
    assert batch == 1 and not q.is_contiguous()
    keys, weights = sums[None, None].cuda().bfloat16(), counts[None, None].cuda()
    uk, uv = payload["w_uk_t"].cuda(), payload["w_uv"].cuda()
    page = {name: tensor.cuda() for name, tensor in payload["cache"].items()}
    page["leaf_v"] = page["leaf_k"][..., :512]
    page["leaf_count"] = payload["leaf_count"]
    page["overflow_active"] = payload["hash_probes"] != 0
    lookback = 256
    local = page["leaf_k"].index_select(
        2, torch.arange(queries + lookback, device="cuda") % payload["leaf_count"])
    sink = page["leaf_k"][..., :1, :].contiguous()
    carrier = local[..., lookback:, :].expand(-1, heads, -1, -1)
    engine = KernelTwoLevelLODAttention(query_heads=heads, key_value_heads=1, scale=payload["scale"])
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=65536, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    engine.leaf_hash_probes = payload["hash_probes"]
    engine._lod_kimi_w_uk_t, engine._lod_kimi_w_uv = uk, uv
    engine._lod_kimi_mutable_projection_weights = True
    local_stream = torch.cuda.Stream()

    def attend(shared_copy):
        # One candidate copy serves both consumers; baseline lets each call
        # materialize its own copy as current serving does. This copy is timed.
        expanded_q = q.contiguous() if shared_copy else q
        engine._lod_kimi_expanded_prefill_chunk = expanded_q
        local_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(local_stream):
            branch = aiter_kimi_local_prefill_attention(
                carrier, local, query_offset=lookback, scale=payload["scale"],
                expanded_q=expanded_q, w_uk_t=uk, w_uv=uv,
                buffers=engine._lod_prefill_attention_buffers)
        engine._lod_prefill_local_stream_pending = local_stream
        return engine._two_level_attention(
            carrier, local, local[..., :512], keys, keys[..., :512], weights,
            None, page["leaf_k"], page["leaf_v"], state_len=sums.size(0),
            state_capacity=sums.size(0), page_cache=page, local_branch=branch,
            sink_k=sink, sink_v=sink[..., :512], context_len=65536)

    with torch.inference_mode():
        before = timed(lambda: attend(False))
        reference = attend(False).clone()
        candidate = timed(lambda: attend(True))
        torch.testing.assert_close(attend(True), reference, atol=0, rtol=0)
        originals = (q.clone(), uk.clone(), uv.clone())
        q.mul_(-0.75)
        uk.mul_(1.125)
        uv.mul_(-0.5)
        fresh = attend(False).clone()
        torch.testing.assert_close(attend(True), fresh, atol=0, rtol=0)
        for destination, source in zip((q, uk, uv), originals, strict=True):
            destination.copy_(source)
        after = timed(lambda: attend(False))
    result = {
        "scope": "warmed whole-attention GPU stage; not model wall time",
        "input": str(args.input), "query_shape": list(q.shape),
        "query_stride": list(q.stride()), "shared_copy_included_in_timing": True,
        "local_geometry": "256 + Q cyclically reused trained records",
        "fresh_query_and_weights_bitwise_match": True,
        "ordinary_before": before, "shared_query": candidate, "ordinary_after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
