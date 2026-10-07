"""FP32-oracle, ragged/sentinel, direct-channel and count tests, then timing."""

from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path


def reference(q, cache, indptr, indices, scale, bias):
    import torch
    outputs, lses = [], []
    for row, (begin, end) in enumerate(itertools.pairwise(indptr.tolist())):
        ids = indices[begin:end].long()
        ids = ids[ids >= 0]
        if not ids.numel():
            outputs.append(torch.zeros(q.shape[1], 512, device=q.device))
            lses.append(torch.full((q.shape[1],), -torch.inf, device=q.device))
            continue
        values = cache[ids].float()
        score = q[row].float() @ values.T * scale
        if bias is not None:
            score += bias[ids]
        outputs.append(score.softmax(-1) @ values[:, :512])
        lses.append(score.logsumexp(-1))
    return torch.stack(outputs), torch.stack(lses)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    from benchmarks.experimental.sparse_mla_gfx942.attention import sparse_mla
    from benchmarks.kimi_k3_kda_upstream_probe import check, graph_time

    torch.manual_seed(1234)
    torch.set_num_threads(1)
    result = dict(status="in_progress", upstream_commit="4b12dc51", extension="per-cache-key natural-log count bias",
                  dim=512, direct_key_dim=64, checks=[], timing=[], production_changed=False)
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        for heads, splits, direct_only, biased in [(12, 1, False, False), (12, 8, False, True),
                (16, 8, True, True), (96, 1, False, True)]:
            cache = torch.randn(129, 576, device="cuda", dtype=torch.bfloat16)
            cache[0].fill_(torch.nan)  # sentinels must not poison valid lanes
            q = torch.randn(4, heads, 576, device="cuda", dtype=torch.bfloat16)
            if direct_only:
                q[..., :512].zero_()
            lists = [torch.randperm(128, device="cuda")[:n].int() + 1 for n in (67, 1, 33)]
            lists[0][3] = -1
            lists.append(torch.tensor([-1, -1], device="cuda", dtype=torch.int32))
            indices = torch.cat(lists)
            ptr = torch.tensor([0, *itertools.accumulate(x.numel() for x in lists)], device="cuda", dtype=torch.int32)
            bias = torch.randint(1, 129, (129,), device="cuda").float().log() if biased else None
            scale = 192**-0.5
            ro, rl = reference(q, cache, ptr, indices, scale, bias)
            out, lse = sparse_mla(q, cache, ptr, indices, scale=scale,
                                 key_log_count=bias, splits=splits, has_invalid=True)
            torch.testing.assert_close(lse[:3], rl[:3], atol=0.004, rtol=0.0001)
            assert torch.isneginf(lse[3]).all(), "empty row's LSE is not -inf"
            assert torch.equal(out[3], torch.zeros_like(out[3])), "empty row's output is not zero"
            point = dict(heads=heads, splits=splits, direct_only=direct_only, biased=biased,
                output=check(ro, out, tolerance=0.01), lse_max_abs=float((rl[:3] - lse[:3]).abs().max()))
            result["checks"].append(point)
            print("SPARSE_MLA_CHECK " + json.dumps(point), flush=True)
            save()

        # Count bias must exactly represent repetitions of identical K/V,
        # not accidentally multiply QK, alter V, or bias the list offsets.
        cache = torch.randn(7, 576, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, 12, 576, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([1, 8, 2, 3, 16, 1, 4], device="cuda")
        ids = torch.tensor([5, 1, 3, 2, 4, 6, 0], device="cuda", dtype=torch.int32)
        ptr = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
        out, lse = sparse_mla(q, cache, ptr, ids, scale=192**-0.5, key_log_count=counts.float().log(), splits=1)
        expanded = cache.repeat_interleave(counts, dim=0)
        ei = torch.arange(expanded.shape[0], device="cuda", dtype=torch.int32)
        ep = torch.tensor([0, ei.numel()], device="cuda", dtype=torch.int32)
        ro, rl = reference(q, expanded, ep, ei, 192**-0.5, None)
        result["count_replication"] = check(ro, out, tolerance=0.01)
        torch.testing.assert_close(lse, rl, atol=0.004, rtol=0.0001)
        for batch in (1, 8):
            cache = torch.randn(16384, 576, device="cuda", dtype=torch.bfloat16)
            q = torch.randn(batch, 12, 576, device="cuda", dtype=torch.bfloat16)
            ids = torch.arange(cache.shape[0], device="cuda", dtype=torch.int32).repeat(batch)
            ptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * cache.shape[0]
            bias = torch.randint(1, 1025, (cache.shape[0],), device="cuda").float().log()
            times = {}
            for name, b in (("unbiased", None), ("biased", bias)):
                ro, rl = reference(q, cache, ptr, ids, 192**-0.5, b)
                out, lse = sparse_mla(q, cache, ptr, ids, scale=192**-0.5, key_log_count=b)
                check(ro, out, tolerance=0.01)
                torch.testing.assert_close(lse, rl, atol=0.004, rtol=0.0001)
                times[name] = graph_time(lambda: sparse_mla(q, cache, ptr, ids,
                    scale=192**-0.5, key_log_count=b), replays=100)
            point = dict(batch=batch, length=16384, heads=12, milliseconds=times)
            result["timing"].append(point)
            print("SPARSE_MLA_TIMING " + json.dumps(point), flush=True)
            save()
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
