#!/usr/bin/env python3
"""Correctness and timing probe for the fused absorbed-MLA coarse path."""

from __future__ import annotations

import argparse
import json
import time

import torch

from lod_attention.kernels.aiter_mla_prefill_attention import (
    aiter_kimi_expanded_prefill_route_coarse_attention,
    aiter_mla_prefill_route_coarse_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--states", type=int, default=512)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--ignore-direct-key", action="store_true")
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_dim = 192 if args.expanded else 576
    q = torch.randn(
        1, args.heads, args.queries, query_dim, device=device, dtype=dtype
    )
    state_k = torch.randn(1, 1, args.states, 576, device=device, dtype=dtype)
    # The latent kernel requires a dense value tensor; K3's production pool
    # can alias K/V, while this standalone random fixture starts from D576 K.
    state_v = state_k[..., :512].contiguous()
    counts = torch.randint(
        1, 64, (1, 1, args.states, 1), device=device, dtype=torch.int32
    ).to(torch.float32)
    scale = 576**-0.5
    buffers: dict[str, torch.Tensor] = {}
    projection_scale = 512**-0.5
    w_uk_t = (
        torch.randn(args.heads, 128, 512, device=device, dtype=dtype)
        * projection_scale
    )
    w_uv = (
        torch.randn(args.heads, 512, 128, device=device, dtype=dtype)
        * projection_scale
    )
    def run():
        if args.expanded:
            return aiter_kimi_expanded_prefill_route_coarse_attention(
                q,
                state_k,
                state_v,
                counts,
                w_uk_t,
                w_uv,
                state_len=args.states,
                scale=scale,
                normalize_route_query=False,
                buffers=buffers,
            )
        return aiter_mla_prefill_route_coarse_attention(
            q,
            state_k,
            state_v,
            counts,
            state_len=args.states,
            kv_group_size=args.heads,
            scale=scale,
            normalize_route_query=True,
            include_direct_key=not args.ignore_direct_key,
            buffers=buffers,
        )

    result = run()
    ready_stream = getattr(result[1], "ready_stream", None)
    if ready_stream is not None:
        torch.cuda.current_stream().wait_stream(ready_stream)
    torch.cuda.synchronize()
    route_matching_fraction: float | None = None
    if args.check:
        slots, coarse, _, _ = result
        if args.expanded:
            # Compare against the exact BF16 tensors supplied to AITER.  A
            # reference that redoes W_UK/W_UV in FP32 can reorder near-tied
            # centroids and is testing projection precision, not attention.
            padded_states = ((args.states + 127) // 128) * 128
            # Production routes directly from this BQHD strided view.  The
            # former contiguous ``kimi_route_q`` staging buffer no longer
            # exists after the zero-copy query-layout change.
            actual_q = q.permute(0, 2, 1, 3)
            actual_k = buffers["kimi_expanded_coarse_k"][:
                padded_states * args.heads * 192
            ].view(1, padded_states, args.heads, 192)
            actual_v = buffers["kimi_expanded_coarse_v"][:
                padded_states * args.heads * 128
            ].view(1, padded_states, args.heads, 128)
            actual_log_counts = buffers["kimi_coarse_log_counts"][:padded_states].view(
                1, padded_states
            )
            reference_scores = torch.einsum(
                "bqhd,bshd->bhqs", actual_q.float(), actual_k.float()
            )
            reference_scores.mul_(scale).add_(
                actual_log_counts.float()[:, None, None, :]
            )
            reference_lse = torch.logsumexp(reference_scores, dim=-1)
            reference_out = torch.einsum(
                "bhqs,bhse->bqhe",
                reference_scores.softmax(-1),
                actual_v.permute(0, 2, 1, 3).float(),
            )
            reference_slots = torch.topk(
                reference_scores, 8, dim=-1
            ).indices
            if coarse.selected_route_scores is None:
                raise AssertionError("expanded MLA routing omitted selected scores")
            selected_reference_scores = torch.gather(
                reference_scores,
                -1,
                slots,
            )
            torch.testing.assert_close(
                coarse.selected_route_scores.float(),
                selected_reference_scores,
                atol=3e-2,
                rtol=3e-3,
            )
            actual_sorted = torch.sort(slots, dim=-1).values
            reference_sorted = torch.sort(reference_slots, dim=-1).values
            matching_fraction = float(
                actual_sorted.eq(reference_sorted).all(-1).float().mean()
            )
            route_matching_fraction = matching_fraction
            route_error: str | None = None
            if matching_fraction < 0.995:
                dot_scores = reference_scores - actual_log_counts.float()[
                    :, None, None, :
                ]
                query_rms = q[..., :192].float().square().mean(-1, keepdim=True).sqrt()
                variants = {
                    "dot_only": dot_scores,
                    "dot_plus_log": reference_scores,
                    "dot_plus_rms_log": dot_scores
                    + query_rms * actual_log_counts.float()[:, None, None, :],
                }
                variant_matches = {}
                for name, scores in variants.items():
                    expected = torch.sort(
                        torch.topk(scores, 8, dim=-1).indices, dim=-1
                    ).values
                    variant_matches[name] = float(
                        actual_sorted.eq(expected).all(-1).float().mean()
                    )
                row_actual = actual_sorted[0, 0, 0]
                all_reference = torch.sort(
                    torch.topk(reference_scores[0, 0], 8, dim=-1).indices,
                    dim=-1,
                ).values
                all_actual = actual_sorted[0, 0]
                overlap_matrix = (
                    all_actual[:, None, :, None]
                    .eq(all_reference[None, :, None, :])
                    .any(-1)
                    .sum(-1)
                )
                closest = overlap_matrix.argmax(-1)
                closest_overlap = overlap_matrix.max(-1).values
                closest_query = int(closest[0])
                route_error = (
                    "combined expanded MLA routes differ from reference: "
                    f"matching_fraction={matching_fraction:.6f}, "
                    f"variant_matches={variant_matches}, "
                    f"actual={row_actual.tolist()}, "
                    f"actual_dot_log_scores="
                    f"{reference_scores[0, 0, 0, row_actual].tolist()}, "
                    f"closest_query={closest_query}, "
                    f"closest_overlap={int(closest_overlap[0])}/8, "
                    f"row_mapping_0_31={closest[:32].tolist()}, "
                    f"exact_mapped_fraction="
                    f"{float(closest_overlap.eq(8).float().mean()):.6f}, "
                    f"reference={reference_sorted[0, 0, 0].tolist()}"
                )
            torch.testing.assert_close(
                coarse.lse_0, reference_lse, atol=3e-2, rtol=3e-3
            )
            torch.testing.assert_close(
                coarse.output_0.float(), reference_out, atol=5e-2, rtol=5e-2
            )
            if route_error is not None:
                raise AssertionError(route_error)
        else:
            used_dim = 512 if args.ignore_direct_key else 576
            mean_k = state_k[..., :used_dim].float() / counts
            mean_v = state_v.float() / counts
            reference_scores = torch.einsum(
                "bhqd,bksd->bhqs", q[..., :used_dim].float(), mean_k
            )
            reference_scores.mul_(scale).add_(
                counts.log().view(1, 1, 1, args.states)
            )
            reference_lse = torch.logsumexp(reference_scores, dim=-1)
            reference_out = torch.einsum(
                "bhqs,bksd->bqhd", reference_scores.softmax(-1), mean_v
            )
            route_q = q[..., :used_dim].float()
            route_rms = route_q.square().mean(-1, keepdim=True).sqrt()
            route_scores = torch.einsum("bhqd,bksd->bhqs", route_q, mean_k)
            route_scores.mul_(scale).add_(
                route_rms * counts.log().view(1, 1, 1, args.states)
            )
            reference_slots = torch.topk(route_scores, 8, dim=-1).indices
            actual_sorted = torch.sort(slots, dim=-1).values
            reference_sorted = torch.sort(reference_slots, dim=-1).values
            route_matching_fraction = float(
                actual_sorted.eq(reference_sorted).all(-1).float().mean()
            )
            if not torch.equal(actual_sorted, reference_sorted):
                mismatch = actual_sorted.ne(reference_sorted).any(-1)
                first = mismatch.nonzero()[0]
                index = tuple(first.tolist())
                matching_fraction = float((~mismatch).float().mean())
                if matching_fraction < 0.995:
                    raise AssertionError(
                        "fused MLA top-eight routes differ from reference: "
                        f"matching_rows={int((~mismatch).sum())}/{mismatch.numel()}, "
                        f"row={index}, actual={actual_sorted[index].tolist()}, "
                        f"reference={reference_sorted[index].tolist()}"
                    )
            torch.testing.assert_close(
                coarse.lse_0, reference_lse, atol=3e-2, rtol=3e-3
            )
            torch.testing.assert_close(
                coarse.output_0.float(), reference_out, atol=3e-2, rtol=3e-2
            )

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(args.iterations):
        measured_result = run()
        measured_ready_stream = getattr(measured_result[1], "ready_stream", None)
        if measured_ready_stream is not None:
            torch.cuda.current_stream().wait_stream(measured_ready_stream)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = begin.elapsed_time(end) / args.iterations
    print(
        json.dumps(
            {
                "queries": args.queries,
                "states": args.states,
                "heads": args.heads,
                "include_direct_key": not args.ignore_direct_key,
                "expanded": args.expanded,
                "route_matching_fraction": route_matching_fraction,
                "milliseconds": elapsed_ms,
                "timestamp": time.time(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
