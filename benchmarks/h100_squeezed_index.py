"""Build the released Squeezed index with shared GQA fits and vectorized calibration.

The fitted objective and cuML options match the authors' release. The same
physical key head only needs one fit. The observation-window queries and
calibrated scores remain per query head.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def fit_head(key, clusters):
    import cupy as cp
    from cuml.cluster import KMeans

    key = key.float().contiguous()
    normalized = F.normalize(key, p=2, dim=-1)
    fit = KMeans(n_clusters=clusters, max_iter=300, init='k-means++', verbose=0, random_state=0)
    fit.fit(cp.from_dlpack(normalized))
    labels = torch.from_dlpack(cp.asarray(fit.labels_)).long()
    count = torch.zeros(clusters, dtype=torch.int64, device=key.device)
    count.scatter_add_(0, labels, torch.ones_like(labels))
    sums = torch.zeros((clusters, key.shape[-1]), dtype=torch.float32, device=key.device)
    sums.index_add_(0, labels, key)
    return sums / count.clamp_min(1)[:, None], labels, count


def calibration_scores(query, centers, labels, counts):
    """Equivalent to the release's centroid-score broadcast and token mean."""
    group = query.shape[0] // centers.shape[0]
    expanded = centers.repeat_interleave(group, dim=0)
    logits = torch.bmm(query.float(), expanded.transpose(1, 2)) * query.shape[-1] ** -0.5
    count = counts.repeat_interleave(group, dim=0)[:, None, :]
    log_z = (logits + count.float().log()).logsumexp(dim=-1, keepdim=True)
    per_center = (logits - log_z).exp().mean(dim=1)
    return per_center.gather(1, labels.repeat_interleave(group, dim=0))


def cluster_count(length, percent, requested):
    count = max(1, int(percent / 100 * length)) if requested is None else requested
    if not 0 < count <= length:
        raise ValueError(f'cluster count {count} exceeds the usable range for {length} indexed tokens')
    return count


def build(payload, observation_window, cluster_percent, *, requested_clusters=None):
    start = time.perf_counter()
    centroids, labels, scores = {}, {}, []
    layers = payload['layers']
    for layer, (key_cpu, q_cpu) in enumerate(layers):
        key = key_cpu[0, :, :-observation_window].cuda()
        q = q_cpu[0].cuda()
        if q.size(1) != observation_window or q.size(0) % key.size(0):
            raise ValueError('Q/K observation geometry differs')
        clusters = cluster_count(key.size(1), cluster_percent, requested_clusters)
        fits = [fit_head(head, clusters) for head in key]
        c, ids, count = (torch.stack(items) for items in zip(*fits))
        scores.append(calibration_scores(q, c, ids, count).cpu().numpy().reshape(-1))
        centroids[layer] = c[None].to(torch.bfloat16).cpu()
        labels[layer] = ids[None].to(torch.int32).cpu()
        print(f'Index layer {layer + 1}/{len(layers)}, elapsed {time.perf_counter() - start:.2f}s', flush=True)
    percentiles = (0.5, 0.7, 0.8, 0.9)
    threshold = dict(zip(percentiles, map(float, np.quantile(np.concatenate(scores), percentiles))))
    return dict(centroids=centroids, labels=labels, threshold=threshold,
                observation_window=observation_window, clusters=clusters,
                cluster_percent=cluster_percent if requested_clusters is None else 100 * clusters / key.size(1),
                seconds=time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--probe-length', type=int)
    p.add_argument('--output', type=Path, required=True)
    geometry = p.add_mutually_exclusive_group()
    geometry.add_argument('--cluster-percent', type=float, default=5)
    geometry.add_argument('--clusters', type=int, help='Exact number of centroids per physical KV head')
    p.add_argument('--observation-window', type=int, default=100)
    a = p.parse_args()
    if (a.input is None) == (a.probe_length is None):
        p.error('choose one of --input or --probe-length')
    if (not 0 < a.cluster_percent <= 100 or a.observation_window < 1
            or (a.clusters is not None and a.clusters < 1)):
        p.error('invalid cluster geometry or observation window')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.probe_length is not None:
        torch.manual_seed(0)
        key = torch.randn((a.probe_length - a.observation_window, 128), device='cuda', dtype=torch.bfloat16)
        clusters = cluster_count(key.size(0), a.cluster_percent, a.clusters)
        torch.cuda.synchronize()
        start = time.perf_counter()
        _, _, counts = fit_head(key, clusters)
        torch.cuda.synchronize()
        assert int(counts.sum()) == key.size(0)
        a.output.write_text(json.dumps(dict(stage='complete', length=a.probe_length, clusters=clusters,
                                            one_physical_head_seconds=time.perf_counter() - start), indent=2) + '\n')
    else:
        payload = torch.load(a.input, map_location='cpu', weights_only=True)
        index = build(payload, a.observation_window, a.cluster_percent, requested_clusters=a.clusters)
        torch.save(index, a.output)


if __name__ == '__main__':
    main()
