#!/usr/bin/env python3
"""Verify that scheduler splits do not change logical LOD prefill blocks."""

from __future__ import annotations

import argparse

import torch

from model.pytorch_lod_attention_paged import PagedLODConfig
from model.triton_lod_engines import KernelRecursivePagedLODAttention


def _engine(
    kv_bits: int,
    *,
    fused_prefill_route_coarse: bool,
    routing_normalization: str,
) -> KernelRecursivePagedLODAttention:
    config = PagedLODConfig(
        chunk_size=16,
        local_window=32,
        state_growth_factor=8.0,
        state_min_size=16,
        protected_prefix=1,
        max_routes=8,
        page_size=16,
        kv_bits=kv_bits,
        quant_group_size=32,
        routing_normalization=routing_normalization,
    )
    engine = KernelRecursivePagedLODAttention(
        config,
        query_heads=4,
        key_value_heads=2,
        scale=64**-0.5,
        default_open_count=3,
    ).cuda()
    engine.separate_sink_cache = True
    engine.fused_prefill_route_coarse = fused_prefill_route_coarse
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-bits", type=int, choices=(0, 4), default=0)
    parser.add_argument("--fused-coarse", action="store_true")
    parser.add_argument("--query-normalized-routing", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(1234)
    length, split, dim = 700, 400, 64
    q = torch.randn(1, 4, length, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, length, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    whole_engine = _engine(
        args.kv_bits,
        fused_prefill_route_coarse=args.fused_coarse,
        routing_normalization=(
            "query" if args.query_normalized_routing else "none"
        ),
    )
    whole, whole_cache = whole_engine(
        q,
        k,
        v,
        use_cache=True,
        finalize_cache_for_decode=False,
    )
    reference_engine = _engine(
        args.kv_bits,
        fused_prefill_route_coarse=False,
        routing_normalization=(
            "query" if args.query_normalized_routing else "none"
        ),
    )
    reference, _ = reference_engine(
        q,
        k,
        v,
        use_cache=True,
        finalize_cache_for_decode=False,
    )
    split_engine = _engine(
        args.kv_bits,
        fused_prefill_route_coarse=args.fused_coarse,
        routing_normalization=(
            "query" if args.query_normalized_routing else "none"
        ),
    )
    first, cache = split_engine(
        q[..., :split, :],
        k[..., :split, :],
        v[..., :split, :],
        use_cache=True,
        finalize_cache_for_decode=False,
    )
    second, cache = split_engine(
        q[..., split:, :],
        k[..., split:, :],
        v[..., split:, :],
        cache=cache,
        use_cache=True,
        finalize_cache_for_decode=False,
    )
    if whole_cache is None or cache is None:
        raise AssertionError("prefill did not return a cache")
    split_output = torch.cat((first, second), dim=2)
    output_error = (whole.float() - split_output.float()).abs()
    reference_error = (whole.float() - reference.float()).abs()
    state_error = (
        whole_cache.state["state_k"].float()
        - cache.state["state_k"].float()
    ).abs()
    result = {
        "kv_bits": args.kv_bits,
        "routing_normalization": (
            "query" if args.query_normalized_routing else "none"
        ),
        "output_max_abs": float(output_error.max().item()),
        "output_mean_abs": float(output_error.mean().item()),
        "reference_output_max_abs": float(reference_error.max().item()),
        "reference_output_mean_abs": float(reference_error.mean().item()),
        "state_max_abs": float(state_error.max().item()),
        "whole_coverage": int(whole_cache.state["coverage"]),
        "split_coverage": int(cache.state["coverage"]),
        "whole_recent_len": int(whole_cache.state["recent_len"]),
        "split_recent_len": int(cache.state["recent_len"]),
    }
    print(result)
    if result["whole_coverage"] != result["split_coverage"]:
        raise AssertionError("scheduler split changed state coverage")
    if result["whole_recent_len"] != result["split_recent_len"]:
        raise AssertionError("scheduler split changed the exact prefill field")
    if result["state_max_abs"] != 0.0:
        raise AssertionError("scheduler split changed the coarse state")
    if result["reference_output_max_abs"] > 0.02:
        raise AssertionError("fused prefill route/coarse changed attention output")
    # Incremental INT4 appends can requantize a partially filled semantic page;
    # bound that intended approximation separately from scheduler/state drift.
    tolerance = 0.07 if args.kv_bits == 4 else 0.02
    if result["output_max_abs"] > tolerance:
        raise AssertionError("scheduler split changed LOD prefill output")
    if args.kv_bits == 4 and result["output_mean_abs"] > 0.003:
        raise AssertionError("incremental INT4 prefill drift is unexpectedly large")


if __name__ == "__main__":
    main()
