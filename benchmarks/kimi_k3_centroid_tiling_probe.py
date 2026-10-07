"""Isolated query-row packing test on production fused CK route/coarse.

Indexer's ReLU/head-summed math is NOT used. Same BF16 expanded centroid
inputs, FP32 scores, log-count bias, top-eight and coarse output in every arm.
Compilation/warmup is outside timings. KDA G8 is the shared model baseline;
this per-rank kernel test does not execute KDA or claim full-model gains.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--queries", type=int, nargs="+", default=[256, 2048, 16384])
    p.add_argument("--states", type=int, default=4096)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--serving-subtile", action="store_true",
                   help="test the current score-only 64-key candidate emitter and exact refinement")
    args = p.parse_args()
    if args.serving_subtile:
        from benchmarks.kimi_k3_decode_power2 import LOD_ENV
        os.environ.update(LOD_ENV)
    import torch
    from lod_attention.kernels.aiter_prefill_attention import _specialized_kimi_coarse_mha_fwd, _reduce_route_candidates
    from benchmarks.kimi_k3_kda_upstream_probe import check, graph_time
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    result = dict(status="running", scope=__doc__, production_changed=False,
                  serving_subtile=args.serving_subtile, points=[])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        heads, states = args.heads, args.states
        k = torch.randn(1, states, heads, 192, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, states, heads, 128, device="cuda", dtype=torch.bfloat16)
        logs = torch.randint(1, 129, (1, states), device="cuda").float().log().bfloat16()
        bias = logs[:, None, None, :].expand(1, heads, 1, states)
        for queries in args.queries:
            q = torch.randn(1, queries, heads, 192, device="cuda", dtype=torch.bfloat16)
            output = torch.empty(1, queries, heads, 128, device="cuda", dtype=torch.bfloat16)
            reference = None
            point = dict(queries=queries, states=states, heads=heads, variants={})
            variants = ((128, 32), (64, 32)) if args.serving_subtile else (
                (128, 32), (64, 32), (128, 64), (64, 64))
            for query_tile, key_step in variants:
                name = f"q{query_tile}_k{key_step}"
                if args.serving_subtile:
                    from benchmarks.kimi_k3_subtile_route import serving_subtile_factory
                    op = serving_subtile_factory(score_only=True, reuse_max=False,
                                                 query_tile=query_tile)
                else:
                    op = _specialized_kimi_coarse_mha_fwd(192, async_bias=True, fused_route=True,
                                                        query_tile=query_tile, key_step=key_step)
                buffers = {}
                def run():
                    co = op(q, k, v, 0.0, 192**-0.5, False, -1, -1, 0, True, True,
                            None, None, output, bias, None, None, None, None, None)
                    candidates = co[2]
                    if args.serving_subtile:
                        from lod_attention.kernels.kimi_route_tile_refine import refine_kimi_centroid_tiles
                        candidates = refine_kimi_centroid_tiles(
                            candidates, q.permute(0, 2, 1, 3).contiguous(), k, logs,
                            state_len=states, scale=192**-0.5, buffers=buffers, tile_n=64)
                    routes, _, _, scores = _reduce_route_candidates(candidates, state_len=states, head_dim=192,
                                                                   emit_metadata=False, buffers=buffers)
                    return co[0], co[1], routes, scores
                print(f"TILING_WARM queries={queries} {name}", flush=True)
                actual = run()
                if reference is None:
                    reference = tuple(x.clone() for x in actual)
                checks = dict(output=check(reference[0], actual[0], tolerance=0.008),
                              lse_max_abs=float((reference[1]-actual[1]).abs().max()),
                              route_set_match=float((reference[2].sort(-1).values == actual[2].sort(-1).values).all(-1).float().mean()))
                torch.testing.assert_close(reference[1], actual[1], atol=0.005, rtol=0.001)
                assert checks["route_set_match"] > 0.995, "route packing changed centroid selection"
                score = torch.einsum("bqhd,bshd->bhqs", q[:, :32].float(), k.float()) * 192**-0.5
                score += logs.float()[:, None, None, :]
                selected = score.gather(-1, actual[2][:, :, :32])
                torch.testing.assert_close(selected.sort(-1).values, score.topk(8, dim=-1).values.sort(-1).values,
                                           atol=0.02, rtol=0.003)
                oracle = torch.einsum("bhqs,bshd->bqhd", score.softmax(-1), v.float())
                checks["oracle_output"] = check(oracle, actual[0][:, :32], tolerance=0.008)
                torch.testing.assert_close(actual[1][:, :, :32], score.logsumexp(-1), atol=0.005, rtol=0.001)
                times = [graph_time(run, replays=20)["ms"] for _ in range(2)]
                def coarse_only():
                    return op(q, k, v, 0.0, 192**-0.5, False, -1, -1, 0, True, True,
                              None, None, output, bias, None, None, None, None, None)
                coarse_times = [graph_time(coarse_only, replays=20)["ms"] for _ in range(2)]
                point["variants"][name] = dict(checks=checks, captured_milliseconds=times,
                                               coarse_only_milliseconds=coarse_times)
                print("TILING_RESULT " + json.dumps(dict(queries=queries, name=name, **point["variants"][name])), flush=True)
                result["points"] = [x for x in result["points"] if x["queries"] != queries] + [point]
                save()
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
