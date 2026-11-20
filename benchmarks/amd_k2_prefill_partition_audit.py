"""Measure safe partition-local and prefix candidate pruning on captured K2 queries."""

import argparse
import json
from pathlib import Path


def audit(file, queries, prefix_blocks=()):
    import torch

    item = torch.load(file, map_location="cuda", weights_only=False)
    data = item["data"]
    q = data["q"][:, :, ::max(1, data["q"].size(2) // queries), :].contiguous()
    b, h, t, d = q.shape
    g = h // item["kv_heads"]
    s = data["s"]
    counts = data["c"][..., :s, 0]
    mean_k = (data["k"][..., :s, :].float() / counts.clamp_min(1)[..., None]).bfloat16()
    dot = torch.bmm(q.reshape(b*h//g, g*t, d), mean_k.reshape(b*h//g, s, d).transpose(1, 2),
                    out_dtype=torch.float32).view(b, h, t, s)
    dot.mul_(item["engine"]["scaling"])
    counts = counts.repeat_interleave(g, 1)[..., None, :]
    active = counts > 0
    eligible = active & (counts <= 1024)
    log_count = counts.clamp_min(1).log()
    inverse = q.float().square().mean(-1, keepdim=True).clamp_min(1.e-12).rsqrt()
    route = (dot * inverse + log_count).masked_fill(~active, -torch.inf)
    threshold = 16 / active.sum(-1, keepdim=True).clamp_min(1)
    route_probability = route.softmax(-1).nan_to_num()
    opened = (route_probability > threshold) & eligible
    native = (dot + log_count).masked_fill(~active, -torch.inf)
    closed_log_fraction = native.masked_fill(opened, -torch.inf).logsumexp(-1) - native.logsumexp(-1)
    closed_fraction = closed_log_fraction.exp()
    finite = native.isfinite().any(-1)

    partitions = {}
    for size in (128, 256, 512, 1024, 2048):
        candidates = torch.zeros_like(opened)
        for start in range(0, s, size):
            span = slice(start, start+size)
            local = route[..., span].softmax(-1).nan_to_num()
            candidates[..., span] = (local >= threshold * (1-1.e-5)) & eligible[..., span]
        if (opened & ~candidates).any():
            raise AssertionError("partition-local filter lost an open region")
        widths = candidates.sum(-1).float()
        partitions[size] = {
            "mean_candidates": widths.mean().item(),
            "p99_candidates": widths.quantile(.99).item(),
            "max_candidates": widths.max().item(),
        }
    prefixes = {}
    for size in prefix_blocks:
        candidates = torch.zeros_like(opened)
        prefix_lse = torch.full_like(route[..., :1], -torch.inf)
        log_cutoff_ratio = (threshold * (1 - 1.e-5)).log()
        for start in range(0, s, size):
            span = slice(start, start + size)
            tile = route[..., span]
            prefix_lse = torch.logaddexp(prefix_lse, tile.logsumexp(-1, keepdim=True))
            candidates[..., span] = (tile >= prefix_lse + log_cutoff_ratio) & eligible[..., span]
        missed = int((opened & ~candidates).sum())
        if missed:
            raise AssertionError(f"prefix filter lost {missed} open regions at tile size {size}")
        widths = candidates.sum(-1).float()
        padded = torch.nn.functional.pad(candidates, (0, (-s) % 32))
        words = padded.reshape(*candidates.shape[:-1], -1, 32).any(-1).sum(-1).float()
        prefixes[size] = {
            "mean_candidates": widths.mean().item(),
            "mean_nonempty_words": words.mean().item(),
            "p99_candidates": widths.quantile(.99).item(),
            "missed_opens": missed,
        }
    return {
        "layer": item["layer"],
        "sampled_query_heads": b * h * t,
        "regions": s,
        "mean_live_regions": active.sum(-1).float().mean().item(),
        "mean_open_regions": opened.sum(-1).float().mean().item(),
        "closed_mass_below": {str(value): (closed_fraction[finite] < value).float().mean().item()
                              for value in (.1, .01, .001, .0001)},
        "partition_candidates": partitions,
        "prefix_candidates": prefixes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--prefix-blocks", type=int, nargs="+", default=())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size <= 0 for size in args.prefix_blocks):
        parser.error("prefix blocks must be positive")
    result = [audit(path, args.queries, args.prefix_blocks)
              for path in sorted(args.snapshots.glob("layer-*.pt"))]
    if not result:
        raise ValueError("missing K2 snapshots")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
