"""Isolated K3 decode candidates; fixed inputs, GPU graph timing, no weights.

Compare exact selection/union fusion, head-work subdivision of the existing
compact consumer, and the cost of retaining coarse value statistics in the
router. These timings are not full-model serving latency or quality scores.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from unittest.mock import patch


def save_result(path, result):
    """Publish a whole snapshot; avoid torn cross-node Ceph reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def route_case(batch, slots, *, ties=False, coarse=False):
    import torch
    import triton
    from lod_attention.kernels.paged_routing import (
        _decode_route_coarse_gqa_groups_kernel, _reduce_decode_route_topk_kernel,
        _reduce_decode_route_coarse_vector_topk_kernel)
    from lod_attention.kernels.paged_decode_buffers import _decode_topk_gqa_union_kernel
    from benchmarks.kimi_k3_decode_route_tune import graph_us, compiler_resources

    torch.manual_seed(7)
    heads, kvheads, dim, vd, group, capacity = 96, 6, 576, 512, 64, slots + 23
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    if ties:
        q.zero_()
    k = torch.randn(batch, 1, capacity, dim, device="cuda", dtype=q.dtype)
    counts = torch.randint(1, 80, (batch, 1, capacity, 1), device="cuda").float()
    counts[:, :, ::37] = 0
    counts[:, :, 1::41] = 1024
    counts[:, :, 2::43] = 1025
    v = (k[..., :vd].float() * counts).bfloat16()
    lengths = torch.randint(1, 1100, (batch, 1, capacity), device="cuda", dtype=torch.int32)
    lengths[:, :, ::17] = 1
    lengths[:, :, 1::19] = 1024
    lengths[:, :, 2::23] = 1025
    keys, values, count = [t.expand(-1, kvheads, -1, -1) for t in (k, v, counts)]
    slotlengths = lengths.expand(-1, kvheads, -1)
    rows = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0)
    lens = slots - torch.arange(batch, device="cuda", dtype=torch.int32) * 3
    local = torch.zeros(batch, device="cuda", dtype=torch.int32)
    groups = triton.cdiv(capacity, group)
    scores = torch.empty(batch, heads, groups, 8, device="cuda")
    indices = torch.empty_like(scores, dtype=torch.int64)
    top = torch.empty(batch, heads, 8, device="cuda", dtype=torch.int64)
    topscore = torch.empty_like(top, dtype=torch.float32)
    nseq, unioncap = batch * kvheads, 16 * 8
    seen = torch.zeros(nseq, capacity, device="cuda", dtype=torch.int32)
    epochs = torch.ones(nseq, device="cuda", dtype=torch.int32)
    unioncount = torch.zeros_like(epochs)
    tokencount = torch.zeros_like(epochs)
    union = torch.empty(nseq, unioncap, device="cuda", dtype=torch.int32)
    gout = torch.empty(batch, heads, groups, vd, device="cuda", dtype=q.dtype)
    glse = torch.empty(batch, heads, groups, device="cuda")
    out = torch.empty(batch, heads, vd, device="cuda", dtype=q.dtype)
    lse = torch.empty(batch, heads, device="cuda")
    dummy = torch.empty(1, device="cuda")
    compiled = {}

    def route(fused, retain=False):
        compiled["router"] = _decode_route_coarse_gqa_groups_kernel[nseq, groups](
            q, keys, values, count, rows, scores, indices, gout, glse,
            *keys.stride()[:3], *values.stride()[:3], *count.stride()[:3], slots,
            lens, unioncount, tokencount, epochs,
            dummy, local, dummy, dummy, dummy, dummy, dummy, 0, 0, 0,
            QUERY_HEADS=heads, KV_HEADS=kvheads, KV_GROUP_SIZE=16,
            HEAD_DIM=dim, VALUE_DIM=vd, SCALE=dim**-0.5, GROUP_N=group,
            MAX_GROUPS=groups, PROTECTED_LEN=0, MAX_LEAF_TOKENS=1024,
            USE_DOT=True, KEYS_ARE_MEANS=True, SCORE_ONLY=not retain,
            USE_STATE_LENS=True, FUSE_UNION_INIT=fused,
            UNION_SEQUENCE_CAPACITY=nseq, num_warps=4, waves_per_eu=1)

    def reduce(fused):
        compiled["reduce"] = _reduce_decode_route_topk_kernel[batch * heads,](
            scores, indices, top, topscore, groups, seen, epochs, unioncount,
            union, slotlengths, rows,
            QUERY_HEADS=heads, KV_HEADS=kvheads, KV_GROUP_SIZE=16, GROUP_N=group,
            STATE_CAPACITY=capacity, UNION_CAPACITY=unioncap, ROUTE_COUNT=8,
            OPEN_COUNT=8, MAX_SEGMENTS=groups,
            CANDIDATE_BLOCK=triton.next_power_of_2(groups * 8),
            FUSE_UNION_BUILD=fused, UNION_SEQUENCE_CAPACITY=nseq,
            MAX_OPEN_LEAVES=1024, SLOT_LENGTH_BATCH_STRIDE=slotlengths.stride(0),
            SLOT_LENGTH_HEAD_STRIDE=slotlengths.stride(1), num_warps=4)

    def old_union():
        _decode_topk_gqa_union_kernel[nseq,](
            top, rows, local, lens, seen, epochs, unioncount, tokencount,
            union, tokencount, dummy, dummy, dummy, dummy, slots,
            top.stride(0), top.stride(1), 0, 0, 0, 0,
            QUERY_HEADS=heads, KV_HEADS=kvheads, KV_GROUP_SIZE=16, ROUTE_COUNT=8,
            STATE_CAPACITY=capacity, CANDIDATE_BLOCK=unioncap, LOCAL_LIMIT=0,
            LOCAL_OFFSET=0, LOCAL_CAPACITY=1, SINK_LEN=0, HEAD_DIM=dim,
            INCLUDE_NEW=False, PREPARE_IMPLICIT_LOD=False, USE_STATE_LENS=True,
            num_warps=2, waves_per_eu=1)

    def launch(fused):
        route(fused)
        reduce(fused)
        if not fused:
            old_union()

    def snapshot():
        return top.clone(), topscore.clone(), [
            union[row, :int(unioncount[row])].sort().values.clone() for row in range(nseq)]

    launch(False)
    ref = snapshot()
    # Replay both repeatedly to test epochs, duplicate selections and stale slots.
    for _ in range(4):
        launch(True)
        actual = snapshot()
        for before, after in zip(ref[:2], actual[:2]):
            assert torch.equal(before, after), "fused selection changed routing"
        assert all(torch.equal(a, b) for a, b in zip(ref[2], actual[2])), "fused union changed its set"
    for row in range(nseq):
        expected = top.view(nseq, 16, 8)[row].flatten().unique()
        expected = expected[expected >= 0].sort().values.int()
        assert torch.equal(expected, actual[2][row]), "dedup differs from independent set oracle"
    result = dict(batch=batch, slots=slots, ties=ties, exact_routes=True,
                  exact_union=True, max_selected_union=max(x.numel() for x in ref[2]), variants={})
    for name, fused in (("separate", False), ("fused", True)):
        us, samples = graph_us(lambda: launch(fused))
        result["variants"][name] = dict(graph_us=us, samples_us=samples,
            router_resources=compiler_resources(compiled["router"]),
            reduction_resources=compiler_resources(compiled["reduce"]))
    if coarse:
        # Cheapest existing retain-value path; no leaves/collective here.
        # Its routing cap does not remove entries from coarse attention:
        # retain all positive-count centroids, including the closed large ones.
        def retained():
            route(False, True)
            return _reduce_decode_route_coarse_vector_topk_kernel[batch * heads,](
                scores, indices, gout, glse, top, topscore, out, lse,
                slotlengths, rows, groups, groups,
                QUERY_HEADS=heads, KV_HEADS=kvheads, KV_GROUP_SIZE=16,
                HEAD_DIM=vd, STATE_CAPACITY=capacity, ROUTE_COUNT=8, OPEN_COUNT=8,
                MAX_SEGMENTS=groups, CANDIDATE_BLOCK=triton.next_power_of_2(groups*8),
                SEGMENT_BLOCK=triton.next_power_of_2(groups), APPLY_MASS_CUTOFF=False,
                LOG_MASS_FRACTION=0., MAX_OPEN_LEAVES=1024,
                SLOT_LENGTH_BATCH_STRIDE=slotlengths.stride(0),
                SLOT_LENGTH_HEAD_STRIDE=slotlengths.stride(1), num_warps=4)
        retained()
        effective_counts = counts[rows.long(), 0, :, 0]
        effective_keys = k[rows.long(), 0].float()
        effective_values = (v.float() / counts.clamp_min(1)).bfloat16()[rows.long(), 0].float()
        full_scores = (q[:, :, :512].float() @ effective_keys[:, :, :512].transpose(-1, -2)
                       + q[:, :, 512:].float() @ effective_keys[:, :, 512:].transpose(-1, -2)) * dim**-0.5
        full_scores += effective_counts.clamp_min(1).log()[:, None, :]
        valid = (effective_counts > 0) & (torch.arange(capacity, device="cuda")[None, :] < lens[rows.long(), None])
        full_scores.masked_fill_(~valid[:, None], -torch.inf)
        torch.testing.assert_close(lse, full_scores.logsumexp(-1), atol=3e-5, rtol=1e-6)
        reference = full_scores.softmax(-1) @ effective_values
        torch.testing.assert_close(out.float(), reference, atol=0.003, rtol=0.015)
        result["retain_coarse_reference_checked"] = True
        expected_top = top.clone()
        expected_scores = topscore.clone()
        route(False)
        reduce(False)
        assert torch.equal(expected_top, top) and torch.equal(expected_scores, topscore)
        result["retain_coarse_us"], result["retain_coarse_samples_us"] = graph_us(retained)
        result["retain_router_resources"] = compiler_resources(compiled["router"])
    return result


def consumer_cases():
    from benchmarks import kimi_gluon_lod_decode_probe as probe
    from benchmarks.kimi_k3_decode_route_tune import graph_us
    from benchmarks.experimental.kimi_decode_work_heads import absorbed_mla_lod_decode_gfx942

    records = []
    for batch, states, pages, splits in ((1, 2048, 128, 32), (1, 4096, 256, 32),
                                       (1, 4096, 1024, 32), (8, 2048, 128, 16)):
        for work in (16, 8, 4):
            argv = ["probe", "--batch-size", str(batch), "--heads", "96",
                    "--coarse", str(states), "--local", "256", "--exact-pages", str(pages),
                    "--splits", str(splits), "--head-tiled-metadata"]
            capture = io.StringIO()
            def consume(*args, **kwargs):
                return absorbed_mla_lod_decode_gfx942(*args, **kwargs, work_heads=work)
            with patch("sys.argv", argv), patch.object(probe, "absorbed_mla_lod_decode_gfx942", consume), \
                    patch.object(probe, "_time_ms", lambda fn, *args, **kw: graph_us(fn)[0]/1000), redirect_stdout(capture):
                probe.main()
            row = json.loads(capture.getvalue().splitlines()[-1])
            assert row["max_abs_error"] < .003 and row["max_lse_error"] < 5e-4, row
            row["work_heads"] = work
            records.append(row)
            print("DECODE_CONSUMER " + json.dumps(row), flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("union", "consumer", "coarse"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--recover-log", type=Path,
                   help="recover completed rows from a successful raw log, without rerunning timing")
    args = p.parse_args()
    result = dict(status="running", scope=__doc__, kind=args.kind, rows=[], production_changed=False)
    if args.recover_log:
        log = args.recover_log.read_text()
        if "status=finished exit_code=0" not in log:
            raise ValueError("only successful completed logs can be recovered")
        prefix = "DECODE_CONSUMER " if args.kind == "consumer" else "DECODE_ROUTE "
        result["rows"] = [json.loads(line.split(prefix, 1)[1]) for line in log.splitlines() if line.startswith(prefix)]
        if len(result["rows"]) != (12 if args.kind == "consumer" else 4):
            raise ValueError("incomplete log records")
        result.update(status="complete", recovered_from_log=str(args.recover_log))
        save_result(args.output, result)
        return
    save_result(args.output, result)
    try:
        if args.kind == "consumer":
            result["rows"] = consumer_cases()
        else:
            for batch, slots, ties in ((1, 2048, False), (1, 4096, False),
                                       (1, 512, True), (8, 2048, False)):
                row = route_case(batch, slots, ties=ties, coarse=args.kind == "coarse")
                result["rows"].append(row)
                print("DECODE_ROUTE " + json.dumps(row), flush=True)
                save_result(args.output, result)
        result["status"] = "complete"
        save_result(args.output, result)
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save_result(args.output, result)
        raise


if __name__ == "__main__":
    main()
