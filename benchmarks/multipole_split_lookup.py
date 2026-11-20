"""Generation-equivalent grouped scores using the authors' split lookup."""
import torch
import triton
import os
from multipole_split_kernel import _fwd_centroid_kernel_qk_gen_splitkv


def split_centroid_scores(q, centroids, counts, labels, scale, split_size=None):
    if split_size is None:
        split_size = int(os.environ.get('MULTIPOLE_LOOKUP_SPLIT_SIZE', '256'))
    if split_size < 128 or split_size % 128:
        raise ValueError('lookup split size must be a positive multiple of 128')
    b, heads, groups, dim = q.shape
    assert b == 1 and centroids.shape[:2] == (b, heads)
    n = centroids.shape[2]
    blocks = triton.cdiv(n, split_size)
    raw_scores = torch.empty((b, heads, n, groups), device=q.device, dtype=torch.float32)
    maxima = torch.empty((b, heads, groups, blocks), device=q.device, dtype=torch.float32)
    mass = torch.empty_like(maxima)
    qk = torch.empty((b, heads, groups, n), device=q.device, dtype=torch.bfloat16)
    _fwd_centroid_kernel_qk_gen_splitkv[(1, heads, blocks)](
        q, centroids, qk, counts, labels, scale, raw_scores, 0.0,
        *q.stride(), *centroids.stride(), *raw_scores.stride(), *qk.stride(),
        b, heads, groups, labels.shape[2], n, mass, maxima, blocks,
        BLOCK_M=16, BLOCK_N=128, BLOCK_DMODEL=dim, BLOCK_KV=split_size, num_warps=4)
    # Each query head has its own normalization. The paper's kernel benchmark
    # averages raw scores; the released generation path averages normalized
    # per-query scores. Preserve the latter, using stable base-2 arithmetic.
    peak = maxima.amax(dim=-1, keepdim=True)
    denominator = (mass * torch.exp2(maxima - peak)).sum(dim=-1, keepdim=True)
    scores = torch.exp2(raw_scores.permute(0, 1, 3, 2) - peak) / denominator
    return scores.sum(dim=2), qk
