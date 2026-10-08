"""CPU validation of the benchmark-only expanded-key assignment geometry."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from benchmarks._glm53_projected_clustering import (
    configure_projected_clustering, projected_clustering_key,
)


def test_similarity_is_mean_of_per_head_projected_key_cosines():
    torch.manual_seed(72)
    x, y = torch.randn(2, 1, 5, 7), torch.randn(2, 1, 3, 7)
    w = torch.randn(4, 6, 7)
    actual = projected_clustering_key(x, w) @ projected_clustering_key(y, w).transpose(-1, -2)
    px = F.normalize(torch.einsum("btl,hdl->bhtd", x[:, 0], w), dim=-1)
    py = F.normalize(torch.einsum("btl,hdl->bhtd", y[:, 0], w), dim=-1)
    expected = (px @ py.transpose(-1, -2)).mean(1, keepdim=True) * x.size(-1)
    torch.testing.assert_close(actual, expected)
    # A head's magnitude should not dominate the shared assignment objective.
    scaled = w.clone()
    scaled[0] *= 100
    torch.testing.assert_close(projected_clustering_key(x, scaled), projected_clustering_key(x, w))


def test_projection_of_centroid_is_mean_of_leaf_projections():
    torch.manual_seed(73)
    keys, w = torch.randn(2, 1, 8, 7), torch.randn(4, 6, 7)
    projected_mean = torch.einsum("btl,hdl->bthd", keys[:, 0].mean(1, keepdim=True), w)
    mean_projected = torch.einsum("btl,hdl->bthd", keys[:, 0], w).mean(1, keepdim=True)
    torch.testing.assert_close(projected_mean, mean_projected)
    # Normalize AFTER averaging raw keys, not by averaging normalized leaves.
    expected = F.normalize(mean_projected, dim=-1).flatten(-2).unsqueeze(1) * (7 / 4) ** .5
    torch.testing.assert_close(projected_clustering_key(keys.mean(2, keepdim=True), w), expected)


def test_geometry_hook_retains_attention_state_and_policy():
    torch.manual_seed(74)
    engine = SimpleNamespace(two_level_topk=8, prefill_two_level_topk=8,
        max_open_centroid_leaves=1024, fused_state_maxsim=True,
        state_clustering_normalization="cosine", state_clustering_query_metric="none")
    pool = SimpleNamespace(engine=engine, initial_prefill_stager=object(), cached_prefill_stager=object())
    keys, w = torch.randn(1, 1, 5, 7), torch.randn(4, 6, 7)
    before = keys.clone()
    configure_projected_clustering(pool, w)
    actual = engine._state_clustering_key(keys, role="centroid", purpose="append")
    torch.testing.assert_close(actual, projected_clustering_key(keys, w))
    assert torch.equal(keys, before)
    assert engine.two_level_topk == engine.prefill_two_level_topk == 8
    assert engine.max_open_centroid_leaves == 1024
    assert engine._glm53_projected_clustering_calls == 1
    assert engine._streaming_state_geometry() is None and not engine.fused_state_maxsim
    assert pool.initial_prefill_stager is pool.cached_prefill_stager is None
    with pytest.raises(RuntimeError, match="already installed"):
        configure_projected_clustering(pool, w)


def test_projected_assignment_changes_latent_cosine_neighbors():
    x = torch.tensor([[[[1., 1.]]]])
    # Learned channel weighting can reverse the nearest centroid.
    centroids = torch.tensor([[[[1., 0.], [.1, 1.]]]])
    w = torch.tensor([[[10., 0.], [0., .1]]])
    raw = (F.normalize(x, dim=-1) @ F.normalize(centroids, dim=-1).transpose(-1, -2)).argmax(-1)
    projected = (projected_clustering_key(x, w) @ projected_clustering_key(centroids, w).transpose(-1, -2)).argmax(-1)
    assert raw.item() == 1 and projected.item() == 0


def test_real_state_update_uses_projected_owners_but_retains_native_sums():
    from lod_attention._engines import KernelTwoLevelLODAttention

    engine = KernelTwoLevelLODAttention(query_heads=2, key_value_heads=1, scale=1)
    engine.state_growth_factor, engine.state_min_len = 0, 2
    engine.sink_len, engine.separate_sink_cache = 0, True
    configure_projected_clustering(SimpleNamespace(engine=engine),
        torch.tensor([[[10., 0.], [0., .1]]]))
    keys = torch.tensor([[[[1., 0.], [.1, 1.], [0., 0.], [0., 0.]]]])
    values = keys * 2
    counts = torch.tensor([[[[1.], [1.], [0.], [0.]]]])
    overflow = torch.tensor([[[[1., 1.], [1., .8]]]])
    old_keys, old_values = keys.clone(), values.clone()
    sk, sv, counts, length, owners, _ = engine._update_state(
        keys, values, counts, None, overflow, overflow * 3,
        state_len=2, ctx_len=4, available_context=4, state_capacity=4)
    assert length == 2 and owners.tolist() == [[[0, 0]]]
    torch.testing.assert_close(sk[..., 0, :], old_keys[..., 0, :] + overflow.sum(2))
    torch.testing.assert_close(sv[..., 0, :], old_values[..., 0, :] + overflow.sum(2) * 3)
    torch.testing.assert_close(sk[..., 1:, :], old_keys[..., 1:, :])
    assert counts.flatten().tolist() == [3., 1., 0., 0.]
    assert engine._glm53_projected_clustering_calls > 0
