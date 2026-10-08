"""Slow, benchmark-only clustering in GLM's actual learned key spaces.

Membership maximizes mean cosine across all main-attention heads. Keys are
projected and normalized per head, then concatenated solely for assignment.
Persistent centroids remain sums of the original latents: by linearity these
are also the sums/means of their expanded keys and values. Attention ranking,
count bias, refinement cap, and softmax replacement are unchanged.
"""
from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def projected_clustering_key(key, weight):
    """Return vectors whose dot product is L * mean_h cosine(W_h x,W_h y)."""
    if key.ndim != 4 or key.size(1) != 1 or weight.ndim != 3:
        raise ValueError("expected shared latent keys [B,1,T,L] and weights [H,D,L]")
    heads, dim, latent = weight.shape
    if key.size(-1) != latent or heads < 1 or dim < 1:
        raise ValueError("projected clustering key/weight geometry differs")
    # Deliberately literal, not a speed candidate: materialize all projected
    # heads, give each equal directional weight, and use a normal GEMM scan.
    projected = F.linear(key[:, 0].float(), weight.float().reshape(-1, latent))
    projected = projected.view(key.size(0), key.size(2), heads, dim)
    projected = F.normalize(projected, dim=-1, eps=1e-12)
    projected = projected * (latent / heads) ** .5
    return projected.flatten(-2).unsqueeze(1).to(key.dtype)


def configure_projected_clustering(pool, weight):
    """Install per-layer geometry before any real request constructs a cache."""
    engine = pool.engine
    if getattr(engine, "_glm53_projected_clustering_weight", None) is not None:
        raise RuntimeError("projected clustering is already installed")
    engine._glm53_projected_clustering_weight = weight.detach()
    engine._glm53_projected_clustering_calls = 0

    def key_geometry(self, key, query_scale=None, *, role="leaf",
                     radial_rms=None, purpose="assignment"):
        if query_scale is not None or radial_rms is not None:
            raise ValueError("projected diagnostic must not combine clustering metrics")
        if role not in {"leaf", "centroid"} or purpose not in {"append", "assignment"}:
            raise ValueError("invalid projected clustering role/purpose")
        self._glm53_projected_clustering_calls += 1
        return projected_clustering_key(key, self._glm53_projected_clustering_weight)

    engine._state_clustering_key = MethodType(key_geometry, engine)
    # Those fast paths assume native latent geometry, or batch several layers
    # through one representative engine. Each layer now has different W_UK.
    engine._streaming_state_geometry = MethodType(lambda self: None, engine)
    engine.fused_state_maxsim = False
    pool.initial_prefill_stager = None
    pool.cached_prefill_stager = None


def install_projected_key_clustering(worker):
    """RPC on every TP rank: gather immutable W_UK once, before real prefill."""
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    layers = [m for m in worker.model_runner.get_model().modules()
              if getattr(m, "_vllm_lod_glm53", False)]
    if not layers:
        raise RuntimeError("projected clustering requires GLM LoD attention layers")
    audit = []
    for layer in layers:
        pool = layer._vllm_lod_pool
        if any(pool.ready):
            raise RuntimeError("projected clustering must precede real cache construction")
        local = layer.W_UK_T.detach().contiguous()
        weight = group.all_gather(local, dim=0) if group.world_size > 1 else local
        configure_projected_clustering(pool, weight)
        audit.append(dict(heads=weight.size(0), projected_dim=weight.size(1),
                          latent_dim=weight.size(2), cap=pool.engine.max_open_centroid_leaves,
                          prefill_topk=pool.engine.prefill_two_level_topk,
                          decode_topk=pool.engine.two_level_topk))
    torch.cuda.synchronize()
    return audit
