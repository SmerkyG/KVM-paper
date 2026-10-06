"""Isolated, graph-timed K3 router geometry and exact-selection checks.

No model weights, distributed communication or serving timing are involved.
The production 96-head query and aliased single-KV-head state are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def compiler_resources(kernel):
    assembly = kernel.asm["amdgcn"]
    fields = ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count",
              "private_segment_fixed_size")
    return {name: int(match.group(1)) for name in fields
            if (match := re.search(r"\." + name + r":\s*(\d+)", assembly))}


def graph_us(function):
    import torch

    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            function()
    samples = []
    for _ in range(5):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 320)
    return sorted(samples)[2], samples


def run_case(batch, slots, group_n, warps, stages, *, means=False, ties=False):
    import torch
    import triton
    from lod_attention.kernels.paged_routing import _decode_route_coarse_gqa_groups_kernel

    torch.manual_seed(7)
    heads, kv_heads, dim, value_dim = 96, 6, 576, 512
    capacity = slots + 23  # exercise ragged active bounds and a partial last tile
    q = torch.randn(batch, heads, dim, dtype=torch.bfloat16, device="cuda")
    counts = torch.randint(1, 80, (batch, 1, capacity, 1), device="cuda").float()
    counts[:, :, 0::37] = 0
    counts[:, :, 1::41] = 1024
    counts[:, :, 2::43] = 1025
    key = torch.randn(batch, 1, capacity, dim, dtype=torch.bfloat16, device="cuda")
    if not means:
        key = (key.float() * counts).bfloat16()
    if ties:
        q.zero_()
    keys = key.expand(-1, kv_heads, -1, -1)
    count = counts.expand(-1, kv_heads, -1, -1)
    cache_rows = torch.arange(batch, device="cuda", dtype=torch.int32).flip(0)
    lens = torch.arange(batch, device="cuda", dtype=torch.int32) * -3 + slots
    groups = triton.cdiv(capacity, group_n)
    scores = torch.empty(batch, heads, groups, 8, device="cuda")
    indices = torch.empty_like(scores, dtype=torch.int64)
    dummy = torch.empty(1, device="cuda")

    def launch():
        return _decode_route_coarse_gqa_groups_kernel[batch * kv_heads, groups](
            q, keys, dummy, count, cache_rows, scores, indices, dummy, dummy,
            *keys.stride()[:3], 0, 0, 0, *count.stride()[:3], slots,
            lens, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy, dummy,
            0, 0, 0, QUERY_HEADS=heads, KV_HEADS=kv_heads, KV_GROUP_SIZE=16,
            HEAD_DIM=dim, VALUE_DIM=value_dim, SCALE=dim ** -0.5, GROUP_N=group_n,
            MAX_GROUPS=groups, PROTECTED_LEN=0, MAX_LEAF_TOKENS=1024, USE_DOT=True,
            KEYS_ARE_MEANS=means, SCORE_ONLY=True, USE_STATE_LENS=True,
            CANDIDATES_PER_GROUP=8, num_warps=warps, num_stages=stages, waves_per_eu=1)

    compiled = launch()
    mean_key = (key if means else
                (key.float() / counts.clamp_min(1)).bfloat16()).index_select(0, cache_rows.long())[:, 0]
    # Separate 512 and 64 products reproduce the router's absorbed MLA math.
    reference = ((q[:, :, :512].float() @ mean_key[:, :, :512].float().transpose(-1, -2)) +
                 (q[:, :, 512:].float() @ mean_key[:, :, 512:].float().transpose(-1, -2)))
    reference *= dim ** -0.5
    effective_count = counts.index_select(0, cache_rows.long())[:, 0, :, 0]
    reference += effective_count.clamp_min(1).log()[:, None, :]
    valid = ((effective_count > 0) & (effective_count < 1024) &
             (torch.arange(capacity, device="cuda")[None, :] < lens.index_select(0, cache_rows.long())[:, None]))
    reference.masked_fill_(~valid[:, None, :], -torch.inf)
    reference = torch.nn.functional.pad(reference, (0, groups * group_n - capacity), value=-torch.inf)
    reference = reference.reshape(batch, heads, groups, group_n)
    expected = reference.argsort(dim=-1, descending=True, stable=True)[..., :8]
    expected += torch.arange(groups, device="cuda")[None, None, :, None] * group_n
    finite = scores.isfinite()
    mismatch = int(((indices != expected) & finite).sum())
    flat_ref = reference.reshape(batch, heads, groups * group_n)
    gathered = flat_ref.gather(-1, indices.flatten(2).clamp(0, groups * group_n - 1)).reshape_as(scores)
    expected_scores = flat_ref.gather(-1, expected.flatten(2)).reshape_as(scores)
    # Native MFMA and torch FP32 GEMM have different summation order. Equal
    # rounded logits can reverse reference ranks; only permit roundoff-sized
    # gaps, then require bit-identical scores/indices against the old router.
    torch.testing.assert_close(gathered[finite], expected_scores[finite], rtol=0, atol=2e-5)
    error = (scores[finite] - gathered[finite]).abs().max().item()
    torch.testing.assert_close(scores[finite], gathered[finite], rtol=1e-5, atol=2e-5)
    microseconds, samples = graph_us(launch)
    return dict(batch=batch, active_slots=slots, group_n=group_n, warps=warps, stages=stages,
                keys_are_means=means, ties=ties, graph_kernel_us=microseconds, samples_us=samples,
                max_score_error=error, reference_rank_differences=mismatch,
                index_sha256=hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest(),
                score_sha256=hashlib.sha256(scores.cpu().numpy().tobytes()).hexdigest(),
                resources=compiler_resources(compiled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope="isolated router only; graph replay, not serving latency", rows=[],
                  status="in_progress")
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    for batch, slots in ((1, 512), (8, 512), (1, 4096), (8, 4096)):
        for group_n, warps, stages in ((64, 1, 2), (64, 2, 2), (64, 4, 2),
                                      (64, 8, 2), (32, 4, 2), (128, 4, 2), (64, 4, 1)):
            print(f"ROUTE_TUNE compiling B={batch} S={slots} N={group_n} waves={warps}", flush=True)
            try:
                row = run_case(batch, slots, group_n, warps, stages)
            except Exception as error:
                row = dict(batch=batch, active_slots=slots, group_n=group_n,
                           warps=warps, stages=stages, error=str(error))
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
            save()
    # Cover already-materialized mean keys and exact ties as well as state sums.
    for means, ties in ((True, False), (False, True)):
        for waves in (1, 4):
            result["rows"].append(run_case(8, 511, 64, waves, 2, means=means, ties=ties))
    for row in result["rows"]:
        if row.get("group_n") != 64 or "error" in row:
            continue
        baseline = next(base for base in result["rows"] if all(
            base.get(field) == row.get(field) for field in
            ("batch", "active_slots", "group_n", "keys_are_means", "ties"))
            and base["warps"] == 1 and base["stages"] == 2)
        row["bitwise_matches_original"] = all(row[field] == baseline[field]
                                              for field in ("index_sha256", "score_sha256"))
        if not row["bitwise_matches_original"]:
            raise AssertionError("changing waves/stages changed original router outputs")
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
