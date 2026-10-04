#!/usr/bin/env python3
"""Benchmark exact Kimi centroid expansion plus patched AITER routing."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from lod_attention.kernels.aiter_prefill_attention import (
    _gather_selected_route_scores,
    _reduce_route_candidates,
    _specialized_kimi_coarse_mha_fwd,
    _specialized_route_mha_fwd,
)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=16384)
    parser.add_argument("--states", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-open-leaf-tokens", type=int)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--debug-routes", action="store_true")
    parser.add_argument(
        "--strided-query",
        action="store_true",
        help=(
            "pass a BQHD view of contiguous BHQD storage, matching the LoD "
            "engine before its current query-layout copy"
        ),
    )
    parser.add_argument(
        "--count-in-key",
        action="store_true",
        help="encode log(count) in a padded D256 Q/K coordinate instead of bias",
    )
    parser.add_argument("--coarse-only", action="store_true")
    parser.add_argument("--tile-max-probe", action="store_true",
                        help="coarse timing only; one maximum per native tile, NOT top-eight LoD")
    parser.add_argument(
        "--async-bias",
        action="store_true",
        help="request the production D192 async-bias coarse specialization",
    )
    parser.add_argument(
        "--fused-route-coarse",
        action="store_true",
        help="make the async coarse pass emit routing candidates as well",
    )
    parser.add_argument(
        "--latent-value-chunks",
        action="store_true",
        help="retain the 512-d latent output using 192+192+128 AITER passes",
    )
    args = parser.parse_args()
    if args.tile_max_probe and not (args.coarse_only and args.async_bias and args.fused_route_coarse):
        parser.error("tile-max probe requires --coarse-only --async-bias --fused-route-coarse")

    torch.manual_seed(29)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q = torch.randn(
        1, args.queries, args.heads, 192, device=device, dtype=dtype
    )
    if args.strided_query:
        q = q.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        if q.is_contiguous():
            raise AssertionError("strided-query probe unexpectedly became contiguous")
    latent_sum = torch.randn(1, args.states, 512, device=device, dtype=dtype)
    direct_sum = torch.randn(1, args.states, 64, device=device, dtype=dtype)
    counts = torch.randint(
        1, 42, (1, args.states), device=device, dtype=torch.int32
    ).float()
    w_uk_t = torch.randn(args.heads, 128, 512, device=device, dtype=dtype)
    w_uv = torch.randn(args.heads, 512, 128, device=device, dtype=dtype)
    # Keep synthetic transformed vectors in the same numerical range as Q/K/V.
    w_uk_t.mul_(512**-0.5)
    w_uv.mul_(512**-0.5)
    buffers: dict[str, torch.Tensor] = {}
    slot_lengths = torch.randint(
        1,
        2049,
        (1, 1, args.states),
        device=device,
        dtype=torch.int32,
    )

    def reduce_candidates(candidate_tensor):
        return _reduce_route_candidates(
            candidate_tensor,
            slot_lengths=(
                slot_lengths if args.max_open_leaf_tokens is not None else None
            ),
            max_open_leaf_tokens=args.max_open_leaf_tokens,
            close_selected_above_limit=args.max_open_leaf_tokens is not None,
            state_len=args.states,
            head_dim=route_dim,
            buffers=buffers,
        )

    # Keep the projection microbenchmarks separate from ``expand_centroids``
    # below.  That legacy helper intentionally includes centroid division and
    # convenient high-level tensor construction, while production receives
    # already-prepared means and writes persistent workspaces in place.
    latent_mean = (latent_sum / counts[..., None]).to(dtype)
    direct_mean = (direct_sum / counts[..., None]).to(dtype)
    flat_k_weight = w_uk_t.permute(2, 0, 1).reshape(512, -1)
    flat_v_weight = w_uv.permute(1, 0, 2).reshape(512, -1)
    production_k_nope = torch.empty(
        1, args.states, args.heads, 128, device=device, dtype=dtype
    )
    production_k = torch.empty(
        1, args.states, args.heads, 192, device=device, dtype=dtype
    )
    production_v = torch.empty(
        1, args.states, args.heads, 128, device=device, dtype=dtype
    )

    def production_projection():
        torch.mm(
            latent_mean.reshape(-1, 512),
            flat_k_weight,
            out=production_k_nope.reshape(-1, args.heads * 128),
        )
        production_k[..., :128].copy_(production_k_nope)
        production_k[..., 128:].copy_(
            direct_mean[:, :, None, :].expand(-1, -1, args.heads, -1)
        )
        torch.mm(
            latent_mean.reshape(-1, 512),
            flat_v_weight,
            out=production_v.reshape(-1, args.heads * 128),
        )
        return production_k, production_v

    # The release path flattens all query heads into the GEMM N dimension.
    # This alternative exposes heads as a batch so small centroid matrices can
    # occupy more compute units independently.  Keep its output head-major;
    # AITER consumes explicit strides, so the final BSHD tensor can be a
    # zero-copy permutation if this organization wins.
    latent_by_head = latent_mean[:, None].expand(
        -1, args.heads, -1, -1
    ).reshape(args.heads, args.states, 512)
    k_weight_by_head = w_uk_t.transpose(1, 2).contiguous()
    v_weight_by_head = w_uv.contiguous()
    batched_k_nope = torch.empty(
        args.heads, args.states, 128, device=device, dtype=dtype
    )
    batched_v = torch.empty_like(batched_k_nope)
    batched_k = torch.empty(
        args.heads, args.states, 192, device=device, dtype=dtype
    )

    def head_batched_projection():
        torch.bmm(latent_by_head, k_weight_by_head, out=batched_k_nope)
        torch.bmm(latent_by_head, v_weight_by_head, out=batched_v)
        batched_k[..., :128].copy_(batched_k_nope)
        batched_k[..., 128:].copy_(
            direct_mean[0, None].expand(args.heads, -1, -1)
        )
        return (
            batched_k[None].permute(0, 2, 1, 3),
            batched_v[None].permute(0, 2, 1, 3),
        )

    combined_weight = torch.cat(
        (
            w_uk_t.permute(2, 0, 1),
            w_uv.permute(1, 0, 2),
        ),
        dim=2,
    ).reshape(512, args.heads * 256).contiguous()
    combined_projection = torch.empty(
        1, args.states, args.heads, 256, device=device, dtype=dtype
    )

    def combined_projection_with_copies():
        torch.mm(
            latent_mean.reshape(-1, 512),
            combined_weight,
            out=combined_projection.reshape(-1, args.heads * 256),
        )
        production_k[..., :128].copy_(combined_projection[..., :128])
        production_k[..., 128:].copy_(
            direct_mean[:, :, None, :].expand(-1, -1, args.heads, -1)
        )
        production_v.copy_(combined_projection[..., 128:])
        return production_k, production_v

    def expand_centroids():
        latent = (latent_sum / counts[..., None]).to(dtype)
        direct = (direct_sum / counts[..., None]).to(dtype)
        flat_k_weight = w_uk_t.permute(2, 0, 1).reshape(512, -1)
        # K uses one large GEMM. V is head-specific, so use a strided BMM
        # without materializing H copies of the latent input.
        k_nope = (latent.reshape(-1, 512) @ flat_k_weight).view(
            1, args.states, args.heads, 128
        )
        latent_by_head = latent[:, None].expand(-1, args.heads, -1, -1)
        v = torch.matmul(latent_by_head, w_uv).permute(0, 2, 1, 3).contiguous()
        expanded_latent = latent_by_head.permute(0, 2, 1, 3).contiguous()
        direct_h = direct[:, :, None, :].expand(-1, -1, args.heads, -1)
        k = torch.cat((k_nope, direct_h), dim=-1)
        return k, v, expanded_latent

    k, v, expanded_latent = expand_centroids()
    if args.count_in_key:
        padded_q = torch.zeros(
            1, args.queries, args.heads, 256, device=device, dtype=dtype
        )
        padded_k = torch.zeros(
            1, args.states, args.heads, 256, device=device, dtype=dtype
        )
        padded_q[..., :192].copy_(q)
        padded_q[..., 192] = 1.0
        padded_k[..., :192].copy_(k)
        padded_k[..., 192].copy_(
            (counts.log() / (192**-0.5)).to(dtype)[..., None].expand(
                -1, -1, args.heads
            )
        )
        q, k = padded_q, padded_k
        bias = None
    else:
        bias = (
            counts.log()
            .to(dtype)[:, None, None, :]
            .expand(-1, args.heads, -1, -1)
            .contiguous()
        )
    route_dim = int(q.size(-1))
    route = _specialized_route_mha_fwd(False, route_dim)
    coarse = _specialized_kimi_coarse_mha_fwd(
        route_dim,
        async_bias=args.async_bias,
        fused_route=args.fused_route_coarse,
        tile_max_probe=args.tile_max_probe,
    )
    output_dim = 512 if args.latent_value_chunks else 128
    output = torch.empty(
        1, args.queries, args.heads, output_dim, device=device, dtype=dtype
    )
    # Route-only deliberately skips output storage; alias the required output
    # argument to q so the probe covers the allocation-free production call.
    route_only_output = q

    def attention():
        results = []
        value_inputs = (
            (
                expanded_latent[..., begin:end]
                .contiguous()
            )
            for begin, end in ((0, 192), (192, 384), (384, 512))
        ) if args.latent_value_chunks else (v,)
        begins = (0, 192, 384) if args.latent_value_chunks else (0,)
        for call_index, (begin, value_input) in enumerate(zip(begins, value_inputs)):
            end = begin + int(value_input.size(-1))
            results.append(route(
                q,
                k,
                value_input,
                0.0,
                192**-0.5,
                False,
                -1,
                -1,
                0,
                True,
                call_index == 0,
                None,
                None,
                output[..., begin:end],
                bias,
                None,
                None,
                None,
                None,
                None,
            ))
        return results[0]

    def coarse_only():
        return coarse(
            q,
            k,
            v,
            0.0,
            192**-0.5,
            False,
            -1,
            -1,
            0,
            True,
            args.fused_route_coarse,
            None,
            None,
            output[..., :128],
            bias,
            None,
            None,
            None,
            None,
            None,
        )

    def coarse_without_bias():
        return coarse(
            q,
            k,
            v,
            0.0,
            192**-0.5,
            False,
            -1,
            -1,
            0,
            True,
            False,
            None,
            None,
            output[..., :128],
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def route_only():
        # The patched AITER specialization treats K==V with route output
        # enabled as a QK-only pass: it emits candidates and skips softmax/PV.
        return route(
            q,
            k,
            k,
            0.0,
            192**-0.5,
            False,
            -1,
            -1,
            0,
            True,
            True,
            None,
            None,
            route_only_output,
            bias,
            None,
            None,
            None,
            None,
            None,
        )

    if args.coarse_only:
        def measure_coarse(call) -> float:
            samples = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                call()
                stop.record()
                stop.synchronize()
                samples.append(start.elapsed_time(stop))
            return statistics.median(samples)

        coarse_result = coarse_only()
        torch.cuda.synchronize()
        check = None
        if args.check:
            # Check only a bounded prefix, not a huge Q x H x S allocation.
            check_queries = min(args.queries, 128)
            reference_scores = torch.einsum(
                "bqhd,bshd->bhqs", q[:, :check_queries].float(), k.float(),
            ) * 192**-0.5 + bias.float()
            expected_lse = reference_scores.logsumexp(-1)
            expected_out = torch.einsum(
                "bhqs,bshd->bqhd", reference_scores.softmax(-1), v.float(),
            )
            torch.testing.assert_close(coarse_result[0][:, :check_queries].float(), expected_out, atol=0.025, rtol=0.025)
            torch.testing.assert_close(coarse_result[1][..., :check_queries], expected_lse, atol=0.025, rtol=0.005)
            if args.tile_max_probe:
                candidates = coarse_result[2]
                maxima = reference_scores.new_full(
                    (*reference_scores.shape[:3], math.ceil(args.states / 128) * 128),
                    -float("inf"),
                )
                maxima[..., :args.states] = reference_scores
                best_scores, _ = maxima.view(
                    1, args.heads, check_queries, -1, 128,
                ).max(-1)
                actual_scores = candidates[:, :, :, 0, :check_queries].transpose(-1, -2)
                actual_indices = candidates[:, :, :, 8, :check_queries].transpose(-1, -2).long()
                actual_scores = actual_scores * math.log(2)
                selected_reference = reference_scores.gather(
                    -1, actual_indices.long().clamp(0, args.states - 1),
                )
                torch.testing.assert_close(actual_scores, best_scores, atol=0.025, rtol=0.005)
                torch.testing.assert_close(selected_reference, best_scores, atol=0.025, rtol=0.005)
            check = "passed: coarse output/LSE and native-tile maxima" if args.tile_max_probe else "passed: coarse output/LSE"
        print(json.dumps({
            "queries": args.queries,
            "states": args.states,
            "heads": args.heads,
            "route_dim": route_dim,
            "coarse_ms": measure_coarse(coarse_only),
            "tile_max_only_not_full_lod": args.tile_max_probe,
            "check": check,
        }, indent=2))
        return

    coarse_stream = torch.cuda.Stream()
    route_stream = torch.cuda.Stream()

    def overlapped():
        current = torch.cuda.current_stream()
        coarse_stream.wait_stream(current)
        route_stream.wait_stream(current)
        with torch.cuda.stream(coarse_stream):
            coarse_result = coarse_only()
        with torch.cuda.stream(route_stream):
            route_result = route_only()
        current.wait_stream(coarse_stream)
        current.wait_stream(route_stream)
        return coarse_result, route_result

    fused_candidates = args.fused_route_coarse
    route_input_snapshot = q.clone() if args.check and not fused_candidates else None
    candidate_result = coarse_only() if fused_candidates else route_only()
    if route_input_snapshot is not None:
        torch.cuda.synchronize()
        if not torch.equal(q, route_input_snapshot):
            raise AssertionError("route-only AITER kernel mutated its query input")
    candidates = candidate_result[2]
    slots, _, _, selected_scores = reduce_candidates(candidates)
    if args.check:
        reference_scores = _gather_selected_route_scores(
            candidates,
            slots,
            state_len=args.states,
            buffers=None,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(selected_scores, reference_scores)
    if args.check and not fused_candidates:
        # Populate the independently checked coarse output.  The route-only
        # call above intentionally leaves its aliased output untouched.
        attention()
    torch.cuda.synchronize()
    if args.check:
        check_q = min(args.queries, 16)
        if not fused_candidates:
            check_output = torch.empty_like(q)
            reference_route = route(
                q,
                k,
                k,
                0.0,
                192**-0.5,
                False,
                -1,
                -1,
                0,
                True,
                True,
                None,
                None,
                check_output,
                bias,
                None,
                None,
                None,
                None,
                None,
            )[2]
            torch.cuda.synchronize()
            if not torch.equal(candidates, reference_route):
                raise AssertionError(
                    "aliasing the unused route output changed route candidates"
                )
        if fused_candidates:
            separate_candidates = route_only()[2]
            separate_slots, _, _, _ = reduce_candidates(separate_candidates)
            torch.cuda.synchronize()
            if not torch.equal(slots[:, :, :check_q], separate_slots[:, :, :check_q]):
                fused_match = (
                    torch.sort(slots[:, :, :check_q], dim=-1).values
                    .eq(
                        torch.sort(
                            separate_slots[:, :, :check_q], dim=-1
                        ).values
                    )
                    .all(-1)
                    .float()
                    .mean()
                )
                raise AssertionError(
                    "fused and separate AITER routes differed: "
                    f"row match={float(fused_match):.4f}"
                )
        scores = torch.einsum(
            "bqhd,bshd->bhqs", q[:, :check_q].float(), k.float()
        ) * (192**-0.5)
        if not args.count_in_key:
            scores += counts.log()[:, None, None, :]
        if args.latent_value_chunks:
            reference = torch.einsum(
                "bhqs,bsd->bqhd",
                scores.softmax(-1),
                (latent_sum / counts[..., None]).float(),
            )
        else:
            reference = torch.einsum(
                "bhqs,bshd->bqhd", scores.softmax(-1), v.float()
            )
        torch.testing.assert_close(
            output[:, :check_q].float(), reference, atol=5e-2, rtol=5e-2
        )
        expected = torch.sort(scores.topk(8, dim=-1).indices, dim=-1).values
        actual = torch.sort(slots[:, :, :check_q], dim=-1).values
        if args.debug_routes:
            raw_scores = candidates[:, :, :, :8, :check_q]
            raw_indices = candidates[:, :, :, 8:, :check_q].to(torch.int64)
            raw_flat_scores = raw_scores.permute(0, 1, 4, 2, 3).reshape(
                1, args.heads, check_q, -1
            )
            raw_flat_indices = raw_indices.permute(0, 1, 4, 2, 3).reshape(
                1, args.heads, check_q, -1
            )
            raw_order = raw_flat_scores.topk(8, dim=-1).indices
            raw_top = torch.sort(
                raw_flat_indices.gather(-1, raw_order), dim=-1
            ).values
            print("candidate shape", tuple(candidates.shape))
            print("reference[0,0,0]", expected[0, 0, 0].tolist())
            print("candidate[0,0,0]", raw_top[0, 0, 0].tolist())
            print("reduced[0,0,0]", actual[0, 0, 0].tolist())
            debug_indices = torch.unique(
                torch.cat((expected[0, 0, 0], raw_top[0, 0, 0]))
            )
            print(
                "reference scores",
                list(
                    zip(
                        debug_indices.tolist(),
                        scores[0, 0, 0, debug_indices].tolist(),
                    )
                ),
            )
            print(
                "candidate scores/indices",
                list(
                    zip(
                        raw_flat_indices[0, 0, 0].tolist(),
                        raw_flat_scores[0, 0, 0].tolist(),
                    )
                ),
            )
            print(
                "candidate row match",
                float(expected.eq(raw_top).all(-1).float().mean()),
                "reduce row match",
                float(raw_top.eq(actual).all(-1).float().mean()),
            )
        if not fused_candidates and not torch.equal(expected, actual):
            match = expected.eq(actual).all(-1).float().mean()
            print(
                "FP32-reference top-eight row match "
                f"(BF16 tie ordering may differ): {float(match):.4f}"
            )

    def measure(call) -> float:
        samples = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        return statistics.median(samples)

    expand_ms = measure(expand_centroids)
    production_projection_ms = measure(production_projection)
    head_batched_projection_ms = measure(head_batched_projection)
    combined_projection_ms = measure(combined_projection_with_copies)
    attention_ms = measure(attention)
    coarse_only_ms = measure(coarse_only)
    route_only_ms = measure(route_only)
    overlapped_ms = measure(overlapped)
    reduce_ms = measure(lambda: reduce_candidates(candidates))
    print(json.dumps({
        "queries": args.queries,
        "states": args.states,
        "heads": args.heads,
        "route_dim": route_dim,
        "expansion_ms": expand_ms,
        "production_projection_ms": production_projection_ms,
        "head_batched_projection_ms": head_batched_projection_ms,
        "combined_projection_ms": combined_projection_ms,
        "route_coarse_ms": attention_ms,
        "coarse_only_ms": coarse_only_ms,
        "route_only_ms": route_only_ms,
        "overlapped_ms": overlapped_ms,
        "route_reduce_ms": reduce_ms,
        "max_open_leaf_tokens": args.max_open_leaf_tokens,
        "total_ms": expand_ms + coarse_only_ms + reduce_ms,
        "candidate_shape": list(candidates.shape),
        "theoretical_scale": 1.0 / math.sqrt(192),
    }, indent=2))


if __name__ == "__main__":
    main()
