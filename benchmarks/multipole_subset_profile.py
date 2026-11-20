"""Graph-replay timings for the final layer of a Multipole subset decode run."""
import math
import statistics

import torch
import torch.nn.functional as F


def graph_us(operation, repeats=48):
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            operation()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(7):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(1000 * begin.elapsed_time(end) / repeats)
    return statistics.median(samples)


def profile(cache):
    q, q_window, local, lse = cache.profile_inputs
    c, labels, counts = cache.centroids[-1], cache.labels[-1], cache.counts[-1]
    out = local.view(1, cache.kv_heads, cache.group, cache.dim).float().contiguous()
    m = (lse.view(1, cache.kv_heads, cache.group) * math.log2(math.e)).contiguous()
    l = torch.ones_like(m)
    n = c.size(2)
    widths = (2816, 2900, 2944) if n == 2900 else (n,)
    lookup = {}
    for width in widths:
        if width <= n:
            centers = c[:, :, :width].contiguous()
            sizes = counts[:, :, :width].contiguous()
        else:
            centers = F.pad(c, (0, 0, 0, width - n))
            sizes = F.pad(counts, (0, width - n))
        lookup[str(width)] = graph_us(lambda: cache._centroid_scores(q_window, centers, labels, sizes))

    score, _ = cache._centroid_scores(q_window, c, labels, counts)
    selected = cache._select(score, labels, counts)
    selection_us = graph_us(lambda: cache._select(score, labels, counts))
    sparse_us = graph_us(lambda: cache._sparse(q, cache.layer_num - 1, out, m, l))
    alternate_score, _ = cache._centroid_scores(q_window, c, labels, counts,
                                                squeezed=not cache.squeezed_kernels)
    alternate_selected = cache._select(alternate_score, labels, counts)
    selected_token_differences = ((selected ^ alternate_selected) * counts).sum(dim=-1)
    return dict(lookup_us_by_centroids=lookup, selection_and_pack_us=selection_us, sparse_us=sparse_us,
                alternate_lookup_token_differences_by_kv_head=selected_token_differences.flatten().tolist())
