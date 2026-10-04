#!/usr/bin/env python3
"""Report Kimi LoD centroid multiplicities without running attention."""

from __future__ import annotations

import argparse
import json

import torch

from lod_attention._config import LODConfig, LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._profile import configure_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65536)
    args = parser.parse_args()

    config = LODConfig(
        chunk_size=256,
        local_window=512,
        state_growth_factor=16.0,
        state_min_size=256,
        protected_prefix=1,
        state_clustering_policy="manual",
        max_routes=8,
        state_clustering_normalization="cosine",
        state_clustering_centroid_rescale="none",
        routing_normalization="none",
        leaf_paged_directory=True,
    )
    engine = KernelTwoLevelLODAttention(
        config,
        query_heads=96,
        key_value_heads=1,
        scale=576**-0.5,
        default_open_count=8,
    )
    engine.head_dim = 576
    configure_engine(
        engine,
        family=ModelFamily.KIMI_K3,
        mode=LODMode.TWO_TIER,
        request_capacity=args.tokens,
        has_query_norm=True,
        has_key_norm=False,
    )
    generator = torch.Generator(device="cuda").manual_seed(23)
    key = torch.randn(
        1, 1, args.tokens, 576,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = key[..., :512]
    cache = engine.build_cache_from_bf16(key, value)
    state_len = int(cache.state["state_len"])
    counts = cache.state["counts"][0, 0, :state_len, 0].cpu()
    unique, frequency = torch.unique(counts, sorted=True, return_counts=True)
    order = torch.argsort(frequency, descending=True)
    print(json.dumps({
        "tokens": args.tokens,
        "state_len": state_len,
        "minimum": float(counts.min()),
        "maximum": float(counts.max()),
        "mean": float(counts.mean()),
        "unique_counts": int(unique.numel()),
        "most_common": [
            [float(unique[index]), int(frequency[index])]
            for index in order[:32]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
