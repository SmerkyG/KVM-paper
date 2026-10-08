"""Immutable inference-weight layouts for shared-latent MLA projections."""

from __future__ import annotations

import torch


def combined_kv_weight(w_uk_t, w_uv, buffers=None):
    """One GEMM emits interleaved [K,V] columns for each query head.

    Retain the source tensors and their versions, not just addresses: shared
    cross-layer scratch must not reuse another layer's weights, recycled
    allocations, or an in-place modified weight (including test fixtures).
    """
    heads, key_dim, latent_dim = w_uk_t.shape
    if tuple(w_uv.shape[:2]) != (heads, latent_dim):
        raise ValueError("MLA key/value projection geometry differs")
    if w_uk_t.dtype != w_uv.dtype or w_uk_t.device != w_uv.device:
        raise ValueError("MLA key/value weights must share dtype and device")
    name = ("mla_combined_kv", w_uk_t.data_ptr(), w_uv.data_ptr(),
            tuple(w_uk_t.shape), tuple(w_uk_t.stride()),
            tuple(w_uv.shape), tuple(w_uv.stride()))
    # Inference tensors deliberately have no version counter.
    def version(tensor):
        return None if tensor.is_inference() else tensor._version
    versions = (version(w_uk_t), version(w_uv))
    cached = None if buffers is None else buffers.get(name)
    if cached is not None and cached[0] == versions:
        return cached[1]
    value_dim = w_uv.size(-1)
    weight = torch.cat((w_uk_t.transpose(1, 2), w_uv), dim=-1)
    weight = weight.permute(1, 0, 2).reshape(latent_dim, heads * (key_dim + value_dim)).contiguous()
    if buffers is not None:
        buffers[name] = (versions, weight, w_uk_t, w_uv)
    return weight
