#!/usr/bin/env python3
"""Benchmark dense and streaming LOD state-construction geometry scans."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
import triton

from model.kernels.lod_kernels import (
    constituent_rms,
    merge_state_in_place,
    new_state_delta_buffers,
    new_state_maxsim_buffers,
    prepare_state_clustering_keys,
    streaming_state_maxsim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--overflow-length", type=int, default=256)
    parser.add_argument("--state-length", type=int, default=1448)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--prepare-block-s", type=int)
    parser.add_argument("--prepare-num-warps", type=int, default=4)
    parser.add_argument("--coherence-single-matmul", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def rms_normalize(value: torch.Tensor) -> torch.Tensor:
    return (
        value.float()
        * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
        )
    ).to(value.dtype)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    shape = (args.batch_size, args.kv_heads)
    overflow = torch.randn(
        *shape,
        args.overflow_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        *shape,
        args.state_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    counts = torch.randint(
        1, 65, (*shape, args.state_length, 1), device=device
    ).float()
    mean = state / counts.to(state.dtype)
    mean_norm = (
        mean.float().square().mean(dim=-1, keepdim=True).sqrt()
        * torch.empty_like(counts).uniform_(1.0, 1.8)
    )
    key_norm_sums = mean_norm * counts

    def torch_constituent_rms() -> torch.Tensor:
        return overflow.float().square().mean(dim=-1, keepdim=True).sqrt()

    def triton_constituent_rms() -> torch.Tensor:
        return constituent_rms(overflow)

    expected_rms = torch_constituent_rms()
    actual_rms = triton_constituent_rms()
    rms_profile = {
        "torch_ms": float(
            triton.testing.do_bench(torch_constituent_rms, warmup=100, rep=500)
        ),
        "triton_ms": float(
            triton.testing.do_bench(triton_constituent_rms, warmup=100, rep=500)
        ),
        "max_abs": float((expected_rms - actual_rms).abs().max().item()),
    }

    records = {}
    for geometry in ("raw", "spherical", "coherence"):
        leaf = rms_normalize(overflow) if geometry == "spherical" else overflow
        buffers = new_state_maxsim_buffers(leaf, args.overflow_length)

        def dense() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            local_mean = state / counts.to(state.dtype)
            if geometry == "raw":
                append_key = local_mean
                route_key = local_mean
            else:
                append_key = rms_normalize(local_mean)
                route_key = (
                    local_mean.float()
                    / (key_norm_sums / counts).clamp_min(1e-12)
                ).to(local_mean.dtype)
                if geometry == "spherical":
                    route_key = append_key
            route_scores = torch.matmul(leaf, route_key.transpose(-1, -2))
            append_scores = (
                torch.matmul(leaf, append_key.transpose(-1, -2))
                if geometry == "coherence"
                else route_scores
            )
            select = append_scores.max(dim=-1).values
            route_scores[..., 0] = float("-inf")
            score, index = route_scores.max(dim=-1)
            return score, index, select

        def streaming() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return streaming_state_maxsim(
                leaf,
                state,
                counts,
                buffers,
                state_len=args.state_length,
                sink_len=1,
                key_norm_sums=(
                    key_norm_sums if geometry == "coherence" else None
                ),
                geometry=geometry,
                block_m=args.block_m,
                block_n=args.block_n,
                num_warps=args.num_warps,
                prepare_block_s=args.prepare_block_s,
                prepare_num_warps=args.prepare_num_warps,
                materialize_prepared_scores=(geometry != "raw"),
                coherence_single_matmul=(
                    geometry == "coherence" and args.coherence_single_matmul
                ),
            )

        def cached_streaming() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return streaming_state_maxsim(
                leaf,
                state,
                counts,
                buffers,
                state_len=args.state_length,
                sink_len=1,
                key_norm_sums=(
                    key_norm_sums if geometry == "coherence" else None
                ),
                geometry=geometry,
                block_m=args.block_m,
                block_n=args.block_n,
                num_warps=args.num_warps,
                prepare_block_s=args.prepare_block_s,
                prepare_num_warps=args.prepare_num_warps,
                prepare_state_geometry=False,
                materialize_prepared_scores=(geometry != "raw"),
                coherence_single_matmul=(
                    geometry == "coherence" and args.coherence_single_matmul
                ),
            )

        dense_result = dense()
        stream_result = streaming()
        torch.cuda.synchronize()
        dense_ms = float(triton.testing.do_bench(dense, warmup=100, rep=500))
        streaming_ms = float(
            triton.testing.do_bench(streaming, warmup=100, rep=500)
        )
        cached_streaming_ms = float(
            triton.testing.do_bench(cached_streaming, warmup=100, rep=500)
        )
        if geometry == "raw":
            cached_dense_ms = dense_ms
        else:
            dense_buffers = new_state_maxsim_buffers(leaf, args.overflow_length)
            prepared_route, prepared_append, _ = prepare_state_clustering_keys(
                state,
                counts,
                dense_buffers,
                state_len=args.state_length,
                key_norm_sums=(
                    key_norm_sums if geometry == "coherence" else None
                ),
                geometry=geometry,
                block_s=args.prepare_block_s,
                num_warps=args.prepare_num_warps,
            )

            def cached_dense() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                route_key = (
                    prepared_route if geometry == "coherence" else prepared_append
                )
                route_scores = torch.matmul(leaf, route_key.transpose(-1, -2))
                append_scores = (
                    torch.matmul(leaf, prepared_append.transpose(-1, -2))
                    if geometry == "coherence"
                    else route_scores
                )
                select = append_scores.max(dim=-1).values
                route_scores[..., 0] = float("-inf")
                score, index = route_scores.max(dim=-1)
                return score, index, select

            cached_dense_ms = float(
                triton.testing.do_bench(cached_dense, warmup=100, rep=500)
            )
        if geometry == "raw":
            sparse_refresh_ms = 0.0
        else:
            refresh_slots = torch.randint(
                1,
                args.state_length,
                (*shape, args.overflow_length),
                device=device,
            )

            def sparse_refresh() -> None:
                prepare_state_clustering_keys(
                    state,
                    counts,
                    buffers,
                    state_len=args.state_length,
                    key_norm_sums=(
                        key_norm_sums if geometry == "coherence" else None
                    ),
                    geometry=geometry,
                    slot_indices=refresh_slots,
                    block_s=args.prepare_block_s,
                    num_warps=args.prepare_num_warps,
                    prepare_coherence_route=True,
                    prepare_coherence_append=not (
                        geometry == "coherence" and args.coherence_single_matmul
                    ),
                    prepare_coherence_scale=not (
                        geometry == "coherence" and args.coherence_single_matmul
                    ),
                )

            sparse_refresh_ms = float(
                triton.testing.do_bench(sparse_refresh, warmup=100, rep=500)
            )
        records[geometry] = {
            "dense_ms": dense_ms,
            "streaming_ms": streaming_ms,
            "cached_streaming_ms": cached_streaming_ms,
            "cached_dense_ms": cached_dense_ms,
            "sparse_refresh_ms": sparse_refresh_ms,
            "incremental_ms": cached_streaming_ms + sparse_refresh_ms,
            "speedup": dense_ms / streaming_ms,
            "route_index_exact_fraction": float(
                (dense_result[1] == stream_result[1]).float().mean().item()
            ),
            "route_score_max_abs": float(
                (dense_result[0] - stream_result[0]).abs().max().item()
            ),
            "select_score_max_abs": float(
                (dense_result[2] - stream_result[2]).abs().max().item()
            ),
            "append_top16_set_exact_fraction": float(
                (
                    dense_result[2].topk(16, dim=-1).indices.sort(dim=-1).values
                    == stream_result[2]
                    .topk(16, dim=-1)
                    .indices.sort(dim=-1).values
                )
                .all(dim=-1)
                .float()
                .mean()
                .item()
            ),
            **{
                f"append_top16_in_approx_top{candidate_count}_fraction": float(
                    (
                        dense_result[2]
                        .topk(16, dim=-1)
                        .indices.unsqueeze(-1)
                        == stream_result[2]
                        .topk(candidate_count, dim=-1)
                        .indices.unsqueeze(-2)
                    )
                    .any(dim=-1)
                    .all(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
                for candidate_count in (24, 32, 64)
            },
            "dense_score_matrix_mib": (
                args.batch_size
                * args.kv_heads
                * args.overflow_length
                * args.state_length
                * torch.tensor([], dtype=torch.bfloat16).element_size()
                * (2 if geometry == "coherence" else 1)
                / 2**20
            ),
        }

    merge_len = args.overflow_length
    state_k = torch.randn_like(state)
    state_v = torch.randn_like(state)
    state_counts = counts.clone()
    state_norms = key_norm_sums.clone()
    merge_k = torch.randn(
        *shape, merge_len, args.head_dim, device=device, dtype=torch.bfloat16
    )
    merge_v = torch.randn_like(merge_k)
    merge_counts = torch.ones(*shape, merge_len, 1, device=device)
    merge_norms = torch.rand_like(merge_counts).add_(0.5)
    destinations = torch.randint(
        1, args.state_length, (*shape, merge_len), device=device
    )
    merge_indices = (
        torch.arange(merge_len, device=device)
        .view(1, 1, -1)
        .expand(*shape, -1)
        .contiguous()
    )
    owners = torch.full_like(merge_indices, -1)
    fused_buffers = new_state_delta_buffers(state_k, state_v, args.state_length)
    legacy_state_k = state_k.clone()
    legacy_state_v = state_v.clone()
    legacy_counts = state_counts.clone()
    legacy_norms = state_norms.clone()
    legacy_owners = owners.clone()
    legacy_buffers = new_state_delta_buffers(
        legacy_state_k, legacy_state_v, args.state_length
    )
    raw_state_k = state_k.clone()
    raw_state_v = state_v.clone()
    raw_counts = state_counts.clone()
    raw_owners = owners.clone()
    raw_buffers = new_state_delta_buffers(
        raw_state_k, raw_state_v, args.state_length
    )

    def raw_update() -> None:
        merge_state_in_place(
            raw_state_k,
            raw_state_v,
            raw_counts,
            merge_k,
            merge_v,
            merge_counts,
            merge_indices,
            destinations,
            raw_owners,
            raw_buffers,
        )

    def fused_norm_update() -> None:
        merge_state_in_place(
            state_k,
            state_v,
            state_counts,
            merge_k,
            merge_v,
            merge_counts,
            merge_indices,
            destinations,
            owners,
            fused_buffers,
            key_norm_sums=state_norms,
            merge_key_norm_sums=merge_norms,
        )

    def legacy_norm_update() -> None:
        merge_state_in_place(
            legacy_state_k,
            legacy_state_v,
            legacy_counts,
            merge_k,
            merge_v,
            merge_counts,
            merge_indices,
            destinations,
            legacy_owners,
            legacy_buffers,
        )
        assignment = F.one_hot(
            destinations, num_classes=args.state_length
        ).float().transpose(-1, -2)
        legacy_norms.add_(torch.matmul(assignment, merge_norms.float()))

    fused_update_ms = float(
        triton.testing.do_bench(fused_norm_update, warmup=100, rep=500)
    )
    raw_update_ms = float(
        triton.testing.do_bench(raw_update, warmup=100, rep=500)
    )
    legacy_update_ms = float(
        triton.testing.do_bench(legacy_norm_update, warmup=100, rep=500)
    )

    result = {
        "geometry": {
            "batch_size": args.batch_size,
            "kv_heads": args.kv_heads,
            "overflow_length": args.overflow_length,
            "state_length": args.state_length,
            "head_dim": args.head_dim,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "num_warps": args.num_warps,
            "prepare_block_s": args.prepare_block_s,
            "prepare_num_warps": args.prepare_num_warps,
            "coherence_single_matmul": args.coherence_single_matmul,
        },
        "results": records,
        "constituent_rms": rms_profile,
        "key_norm_update": {
            "raw_ms": raw_update_ms,
            "fused_ms": fused_update_ms,
            "legacy_dense_ms": legacy_update_ms,
            "speedup": legacy_update_ms / fused_update_ms,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
