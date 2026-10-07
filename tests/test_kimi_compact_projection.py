from __future__ import annotations

import math

import pytest
import torch

from lod_attention.kernels.kimi_compact_leaf_projection import (
    compact_row_capacity, project_compact_kimi_leaves,
)


@pytest.mark.parametrize("block", [16, 32, 64])
def test_compact_capacity_bounds_any_centroid_partition(block):
    for tokens in (1, 17, 127, 1024, 4097):
        for slots in (1, 7, 128):
            lengths = torch.bincount(torch.arange(tokens) % slots, minlength=slots)
            assert lengths.sum().item() <= compact_row_capacity(tokens, slots, block)


@pytest.mark.parametrize("args", [(0, 1, 32), (1, 0, 32), (1, 1, 48)])
def test_compact_capacity_rejects_invalid_sizes(args):
    with pytest.raises(ValueError):
        compact_row_capacity(*args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("compact", [False, True])
def test_smaller_prefill_head_groups_preserve_attention(monkeypatch, compact):
    """Grouping changes only temporary storage, not routes or MLA heads."""
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine

    monkeypatch.setenv("LOD_KIMI_COMPACT_SELECTED_PROJECTION", str(int(compact)))
    monkeypatch.setenv("LOD_KIMI_SORT_LEAF_ROUTES", "1")
    monkeypatch.setenv("LOD_KIMI_LEAF_BLOCK_M", "64")
    monkeypatch.setenv("LOD_KIMI_LEAF_WARPS", "1")
    torch.manual_seed(191)
    heads, tokens, slots, queries = 12, 1024, 16, 64
    source = torch.randn(1, 1, tokens, 576, device="cuda").bfloat16()
    uk = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    query = torch.randn(1, heads, queries, 192, device="cuda").bfloat16()
    carrier = torch.zeros(1, heads, queries, 576, device="cuda", dtype=torch.bfloat16)
    owners = (torch.arange(tokens, device="cuda") % slots).int().view(1, 1, tokens)
    routes = torch.stack([
        torch.randperm(slots, device="cuda")[:8] for _ in range(heads * queries)
    ]).int().view(1, heads, queries, 8)
    results, projection_bytes = [], []
    for group in (12, 6, 4, 2):
        engine = KernelTwoLevelLODAttention(
            LODConfig(), query_heads=heads, key_value_heads=1, scale=192**-0.5,
        )
        engine.head_dim = 576
        configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                         request_capacity=2048, has_query_norm=True, has_key_norm=False)
        cache = engine._new_page_cache(
            source, source[..., :512], owners, state_capacity=slots,
            sequence_capacity=2048, virtual_k=source, virtual_v=source[..., :512],
        )
        buffers = engine._lod_prefill_attention_buffers = {}
        engine._lod_kimi_expanded_prefill_chunk = query
        engine._lod_kimi_w_uk_t = uk
        engine._lod_kimi_w_uv = uv
        engine._lod_kimi_prefill_head_group_limit = group
        output, lse = engine._paged_leaf_attention(carrier, routes, cache, active_slots=slots)
        results.append((output.clone(), lse.clone()))
        # The projection arena is shared across groups. It must shrink, not
        # retain one permanent projection for every head slice.
        arena_names = ({"kimi_compact_projected_k", "kimi_compact_projected_v"}
                       if compact else {"kimi_leaf_fused_kv",
                                        "kimi_leaf_expanded_k_token_major",
                                        "kimi_leaf_expanded_k_nope_token_major",
                                        "kimi_leaf_expanded_v_token_major"})
        storages = {}
        for name, value in buffers.items():
            if isinstance(value, torch.Tensor) and name in arena_names:
                storage = value.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        projection_bytes.append(sum(storages.values()))
    for output, lse in results[1:]:
        torch.testing.assert_close(results[0][0], output, atol=0, rtol=0)
        torch.testing.assert_close(results[0][1], lse, atol=0, rtol=0)
    assert projection_bytes[0] > 0
    assert projection_bytes[1] <= projection_bytes[0] * 0.51
    assert projection_bytes[2] <= projection_bytes[0] * 0.34
    assert projection_bytes[3] <= projection_bytes[0] * 0.17


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("projection_tile", [16, 32, 64])
def test_compact_selected_projection_and_attention(directory, projection_tile):
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine
    from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv
    from lod_attention.kernels.paged_prefill import count_expert_routes, paged_leaf_attention

    torch.manual_seed(113)
    batch, heads, tokens, slots, queries = 2, 4, 257, 9, 37
    engine = KernelTwoLevelLODAttention(LODConfig(), query_heads=heads, key_value_heads=1,
                                      scale=192**-0.5)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=512, has_query_norm=True, has_key_norm=False)
    engine.leaf_paged_directory = directory
    # Include non-contiguous input, unequal centroid sizes, empty slots, tails,
    # unused sentinel leaves, distinct batches and distinct head selections.
    backing = torch.randn(batch, 1, tokens + 73, 576, device="cuda").bfloat16()
    source = backing[..., :tokens, :]
    owners = torch.zeros(batch, 1, tokens, dtype=torch.int32, device="cuda")
    owners[..., 51:102] = 1
    owners[..., 102:135] = 4
    owners[..., 135:] = 7
    cache = engine._new_page_cache(source, source[..., :512], owners, state_capacity=slots,
                                  sequence_capacity=512, virtual_k=source,
                                  virtual_v=source[..., :512])
    uk = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    query = torch.randn(batch, heads, queries, 192, device="cuda").bfloat16()
    routes = torch.full((batch, heads, queries, 8), -1, dtype=torch.int32, device="cuda")
    routes[0, 0, :, 0] = 0
    routes[0, 1, :, 0] = 4
    routes[0, 2, :, :2] = torch.tensor([1, 7], device="cuda")
    routes[1, 1, :, 0] = 7
    routes[1, 2, :, 0] = 0
    routes[1, 3, :, 0] = 1
    routes[..., -3:, :] = -1
    buffers = {}
    counts, offsets = count_expert_routes(routes, active_slots=slots)
    k, v, starts = project_compact_kimi_leaves(
        source, uk, uv, cache, counts, active_slots=slots,
        hash_probes=engine._page_lookup_probes(cache), buffers=buffers, block_m=projection_tile,
    )
    reference_k, reference_v = expand_kimi_leaf_kv(source, uk, uv)
    host_starts = starts.cpu()
    lengths = cache["slot_lengths"].cpu()[:, 0]
    used = counts.cpu().view(batch, heads, slots)
    expected_rows = 0
    for b in range(batch):
        for h in range(heads):
            for slot in range(slots):
                expert = (b * heads + h) * slots + slot
                if not used[b, h, slot]:
                    assert host_starts[expert] == host_starts[expert + 1]
                    continue
                length = int(lengths[b, slot])
                start = int(host_starts[expert])
                expected_rows += length
                # Order within each virtual centroid follows chronological
                # insertion in this cache; no new approximation/reordering.
                selected = owners[b, 0] == slot
                torch.testing.assert_close(k.view(-1, 192)[start:start + length],
                                           reference_k[b, h, selected], atol=0.016, rtol=0.02)
                torch.testing.assert_close(v.view(-1, 128)[start:start + length],
                                           reference_v[b, h, selected], atol=0.016, rtol=0.02)
    assert int(host_starts[-1]) == expected_rows
    assert expected_rows < batch * heads * tokens
    common = dict(page_indices=cache["page_indices"], kv_group_size=1, active_slots=slots,
                  route_head_counts=counts, route_offsets=offsets,
                  scale=engine.scaling, hash_probes=engine._page_lookup_probes(cache),
                  block_m=32, block_n=16, num_warps=1, scalar_page_lookup=True)
    directory_args = (cache["slot_pages"], cache["overflow_page_keys"],
                      cache["overflow_page_values"], cache["overflow_used"], cache["slot_lengths"], routes)
    for reduce in (False, True):
        expected = paged_leaf_attention(query, reference_k, reference_v, *directory_args,
                                        reduce_routes=reduce, **common)
        actual = paged_leaf_attention(query, k, v, *directory_args,
                                      compact_leaf_offsets=starts, reduce_routes=reduce, **common)
        mask = routes.ge(0).any(-1) if reduce else routes.ge(0)
        if not reduce:
            # Independent dense reference catches addressing bugs shared by
            # two kernel variants (especially batch strides in token-major K/V).
            for b in range(batch):
                for h in range(heads):
                    for route in range(2):
                        slot = int(routes[b, h, 0, route])
                        if slot < 0:
                            continue
                        leaf_mask = owners[b, 0] == slot
                        scores = query[b, h, :queries - 3].float() @ reference_k[b, h, leaf_mask].float().T
                        scores *= engine.scaling
                        dense = scores.softmax(-1) @ reference_v[b, h, leaf_mask].float()
                        torch.testing.assert_close(actual[0][b, h, :queries - 3, route].float(),
                                                   dense, atol=0.016, rtol=0.02)
                        torch.testing.assert_close(expected[0][b, h, :queries - 3, route].float(),
                                                   dense, atol=0.016, rtol=0.02)
        torch.testing.assert_close(actual[0][mask], expected[0][mask], atol=0.016, rtol=0.02)
        torch.testing.assert_close(actual[1][mask], expected[1][mask], atol=0.02, rtol=0.003)
    # Changing routes must refresh the union even with the same source/buffer
    # addresses. No stale selected-centroid cache survives across chunks.
    pointers = [t.data_ptr() for t in (k, v, starts)]
    empty_counts = torch.zeros_like(counts)
    again = project_compact_kimi_leaves(source, uk, uv, cache, empty_counts,
                                      active_slots=slots, hash_probes=engine._page_lookup_probes(cache),
                                      buffers=buffers, block_m=projection_tile)
    assert [t.data_ptr() for t in again] == pointers
    assert not again[2].count_nonzero().item()
    # Capture/replay the projection, then change the selection in place:
    # device totals must not have become host constants during graph capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = project_compact_kimi_leaves(source, uk, uv, cache, empty_counts,
                                             active_slots=slots, hash_probes=engine._page_lookup_probes(cache),
                                             buffers=buffers, block_m=projection_tile)
    empty_counts.copy_(counts)
    graph.replay()
    assert int(captured[2][-1].item()) == expected_rows
    empty_counts.zero_()
    graph.replay()
    assert not captured[2].count_nonzero().item()
    empty_counts.fill_(1)
    graph.replay()
    assert captured[2][-1].item() == batch * heads * tokens
    assert captured[2][-1].item() == k.numel() // 192
