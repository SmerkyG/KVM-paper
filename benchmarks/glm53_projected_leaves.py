"""Projection-inclusive GLM leaf refinement comparison and small tile sweep.

This is a random-input kernel diagnostic, NOT model latency or quality.
Both paths consume the same post-cap routes from projected coarse attention.
The measured complete stage includes dispatch, projection, exact attention,
and route reduction (plus W_UV for the absorbed baseline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks._vllm import write_json
from lod_attention._config import LODConfig, LODMode, ModelFamily
from lod_attention._engines import KernelTwoLevelLODAttention
from lod_attention._profile import configure_engine
from lod_attention.kernels.aiter_mla_prefill_attention import project_kimi_head_values
from lod_attention.kernels.glm_compact_leaf_projection import project_compact_glm_leaves
from lod_attention.kernels.glm_projected_prefill import projected_route_coarse, projected_leaf_attention
from lod_attention.kernels.paged_prefill import count_expert_routes


def timed(call, iterations=5):
    call()  # All allocation/JIT outside the timed GPU interval.
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        value = call()
    end.record()
    end.synchronize()
    return value, begin.elapsed_time(end) / iterations


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", type=int, default=49152)
    p.add_argument("--queries", type=int, default=16384)
    p.add_argument("--heads", type=int, default=16, help="16 = one TP4 rank; 64 = TP1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--latent-counter-probe", action="store_true",
                   help="Only compare atomic versus sorted counts for latent leaves")
    args = p.parse_args()
    torch.manual_seed(193)
    engine = KernelTwoLevelLODAttention(LODConfig(), query_heads=args.heads,
                                       key_value_heads=1, scale=.0625).cuda().eval()
    engine.head_dim = 512
    configure_engine(engine, family=ModelFamily.GLM53_FLASH, mode=LODMode.TWO_TIER,
        request_capacity=args.history + args.queries, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    latent = (torch.randn(1, 1, args.history, 512, device="cuda") * .4).bfloat16()
    state = engine._build_cache_from_bf16(latent, latent, finalize_cache_for_decode=False)
    cache, slots = state["page_cache"], int(state["state_len"])
    cache["leaf_v"] = cache["leaf_k"]
    uk = (torch.randn(args.heads, 256, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(args.heads, 512, 256, device="cuda") / 512**.5).bfloat16()
    engine._lod_kimi_w_uk_t, engine._lod_kimi_w_uv = uk, uv
    q = (torch.randn(1, args.heads, args.queries, 256, device="cuda") * .3).bfloat16()
    absorbed = torch.bmm(q[0], uk).unsqueeze(0)
    routes, _ = projected_route_coarse(q, state["state_k"], state["counts"], uk, uv,
        state_len=slots, scale=.0625, slot_lengths=cache["slot_lengths"],
        max_open_leaf_tokens=1024, buffers={})
    result = dict(scope="random-input complete remote-refinement GPU interval, not model timing",
                  history=args.history, queries=args.queries, heads=args.heads, slots=slots,
                  includes="dispatch + projection + exact leaf attention + route reduction",
                  iterations=5, projection_tiles=[], attention_tiles=[], latent_tiles=[])
    def save():
        write_json(args.output, result)
    def baseline():
        value, lse = engine._paged_leaf_attention(absorbed, routes, cache,
                                                  active_slots=slots, reduce_routes=True)
        return project_kimi_head_values(value, uv), lse
    if args.latent_counter_probe:
        result["latent_counter"] = []
        sample = [0, min(37, args.queries - 1), args.queries - 1]
        reference = None
        for sorted_routes in (False, True, False):
            engine._lod_glm_sorted_routes = sorted_routes
            actual, ms = timed(baseline)
            selected = tuple(t[:, :, sample].clone() for t in actual)
            if reference is None:
                reference = selected
            else:
                for expected, observed in zip(reference, selected):
                    torch.testing.assert_close(observed, expected, atol=.001, rtol=.025)
            result["latent_counter"].append(dict(sorted_routes=sorted_routes, ms=ms))
            print(result["latent_counter"][-1], flush=True)
            save()
        return
    reference, result["baseline_before_ms"] = timed(baseline)
    sample = [0, min(37, args.queries - 1), args.queries - 1]
    reference = tuple(t[:, :, sample].clone() for t in reference)
    counts, _ = count_expert_routes(routes, active_slots=slots)
    projection_buffers = {}
    live = int(cache["leaf_count"])
    source = cache["leaf_k"][..., :live, :]
    for tile in ((32, 128, 4), (64, 128, 4), (128, 128, 8), (64, 64, 4)):
        def project():
            return project_compact_glm_leaves(source, uk, uv, cache, counts,
                active_slots=slots, hash_probes=engine._page_lookup_probes(cache),
                buffers=projection_buffers, block_m=tile[0], block_n=tile[1], num_warps=tile[2])
        projected, ms = timed(project)
        row = dict(tile=tile, projection_only_ms=ms)
        print(row, flush=True)
        result["projection_tiles"].append(row)
        save()
    starts = projected[2]
    result["projected_leaf_head_pairs"] = int(starts[-1].item())
    result["all_leaf_head_pairs"] = live * args.heads
    best_projection = min(result["projection_tiles"], key=lambda t: t["projection_only_ms"])["tile"]
    buffers = {}
    for tile in ((16, 16, 2), (32, 16, 2), (32, 16, 4), (64, 16, 2), (64, 32, 4)):
        def candidate():
            return projected_leaf_attention(engine, q, routes, cache, active_slots=slots,
                buffers=buffers, projection_tile=best_projection, attention_tile=tile,
                route_counter=count_expert_routes)
        actual, ms = timed(candidate)
        delta = (actual[0][:, :, sample].float() - reference[0].float()).flatten(2).norm(dim=-1)
        norm = reference[0].float().flatten(2).norm(dim=-1).clamp_min(1e-8)
        relative = float((delta / norm).mean().item())
        valid = reference[1].isfinite()
        max_lse = float((actual[1][:, :, sample][valid] - reference[1][valid]).abs().max().item())
        assert relative < .025 and max_lse < .02, (relative, max_lse)
        row = dict(tile=tile, complete_refinement_ms=ms, relative_output_l2=relative,
                   max_lse_difference=max_lse)
        print(row, flush=True)
        result["attention_tiles"].append(row)
        save()
    _, result["baseline_after_ms"] = timed(baseline)
    original_tile = engine.leaf_block_m, engine.leaf_block_n, engine.leaf_num_warps
    # Check that the projection gain isn't merely an untuned latent control.
    for tile in ((16, 16, 2), (32, 16, 4)):
        engine.leaf_block_m, engine.leaf_block_n, engine.leaf_num_warps = tile
        actual, ms = timed(baseline)
        torch.testing.assert_close(actual[0][:, :, sample], reference[0], atol=.003, rtol=.035)
        row = dict(tile=tile, complete_refinement_ms=ms)
        result["latent_tiles"].append(row)
        print(row, flush=True)
        save()
    engine.leaf_block_m, engine.leaf_block_n, engine.leaf_num_warps = original_tile
    best = min(result["attention_tiles"], key=lambda t: t["complete_refinement_ms"])
    result["best_projection_tile"] = best_projection
    result["best_attention_tile"] = best["tile"]
    result["speedup_vs_bracketed_baseline"] = (
        (result["baseline_before_ms"] + result["baseline_after_ms"]) / 2 / best["complete_refinement_ms"])
    engine._lod_leaf_timing_events = {}
    projected_leaf_attention(engine, q, routes, cache, active_slots=slots,
        buffers=buffers, projection_tile=best_projection, attention_tile=best["tile"],
        route_counter=count_expert_routes)
    torch.cuda.synchronize()
    result["projected_attention_diagnostic_ms"] = {
        name: sum(begin.elapsed_time(end) for begin, end in pairs)
        for name, pairs in engine._lod_leaf_timing_events.items()}
    del engine._lod_leaf_timing_events
    # Metadata-only experiment: replace contended per-route atomics with
    # local sorts and prefix scans. Attention/routes remain unchanged.
    from lod_attention.kernels.kimi_sorted_route_counts import count_sorted_kimi_routes
    from functools import partial
    result["sorted_count_candidates"] = []
    for chunk in (512, 1024, 2048):
        def sorted_candidate():
            return projected_leaf_attention(engine, q, routes, cache, active_slots=slots,
                buffers=buffers, projection_tile=best_projection, attention_tile=best["tile"],
                route_counter=partial(count_sorted_kimi_routes, chunk_items=chunk))
        actual, ms = timed(sorted_candidate)
        torch.testing.assert_close(actual[0][:, :, sample].float(), reference[0].float(), atol=.003, rtol=.035)
        row = dict(chunk_items=chunk, complete_refinement_ms=ms)
        result["sorted_count_candidates"].append(row)
        print(row, flush=True)
        save()
    save()
    print(result, flush=True)


if __name__ == "__main__":
    main()
