"""Small, targeted GPU checks for projected NoPE centroids and replacement."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU projected MLA")


@torch.inference_mode()
def test_transposed_and_cropped_latent_queries_expose_all_leaves_exactly():
    """The adapter's B,T,H,D -> B,H,T,D layout must not misaddress heads."""
    from lod_attention._engines import KernelTwoLevelLODAttention
    torch.manual_seed(81)
    engine = KernelTwoLevelLODAttention(query_heads=4, key_value_heads=1, scale=.0625).cuda().eval()
    engine.leaf_layout = "expert"
    engine.virtual_page_storage = True
    engine.leaf_page_size = engine.leaf_block_n = 16
    engine.leaf_block_m = 32
    engine._lod_shared_latent_kv = True
    latent = (torch.randn(1, 1, 257, 512, device="cuda") * .4).bfloat16()
    owners = (torch.arange(257, device="cuda") % 8).view(1, 1, -1).int()
    cache = engine._new_page_cache(latent, latent, owners, state_capacity=8, sequence_capacity=257,
                                    virtual_k=latent, virtual_v=latent)
    # Preserve both the transposed head/time order and cropped query stride.
    query = (torch.randn(1, 93, 4, 512, device="cuda") * .3).bfloat16().permute(0, 2, 1, 3)[..., 9:74, :]
    assert not query.is_contiguous()
    routes = torch.arange(8, device="cuda", dtype=torch.int32).view(1, 1, 1, 8).expand(1, 4, 65, 8).contiguous()
    actual, lse = engine._paged_leaf_attention(query, routes, cache, active_slots=8)
    packed, packed_lse = engine._paged_leaf_attention(query.contiguous(), routes, cache, active_slots=8)
    torch.testing.assert_close(actual, packed, rtol=0, atol=0)
    torch.testing.assert_close(lse, packed_lse, rtol=0, atol=0)
    scores = query.float() @ latent[0, 0].float().T * .0625
    torch.testing.assert_close(actual.float(), scores.softmax(-1) @ latent[0, 0].float(), atol=.001, rtol=.025)
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=.0001, rtol=.0001)


@torch.inference_mode()
def test_projected_local_offset_and_buffer():
    from lod_attention.kernels.glm_projected_prefill import projected_local_attention
    torch.manual_seed(73)
    q = (torch.randn(2, 4, 137, 256, device="cuda") * .3).bfloat16()
    latent = (torch.randn(2, 1, 193, 512, device="cuda") * .4).bfloat16()
    uk = (torch.randn(4, 256, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(4, 512, 256, device="cuda") / 512**.5).bfloat16()
    destination = torch.empty(2, 137, 4, 256, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    actual, lse = projected_local_attention(q, latent, uk, uv, query_offset=56,
        scale=.0625, output_buffer=destination)
    k = torch.einsum("btl,hdl->bhtd", latent[:, 0].float(), uk.float()).bfloat16().float()
    v = torch.einsum("btl,hld->bhtd", latent[:, 0].float(), uv.float()).bfloat16().float()
    scores = q.float() @ k.transpose(-1, -2) * .0625
    mask = torch.arange(193, device="cuda")[None, :] <= 56 + torch.arange(137, device="cuda")[:, None]
    scores.masked_fill_(~mask, -torch.inf)
    assert actual.data_ptr() == destination.data_ptr()
    torch.testing.assert_close(actual.float(), scores.softmax(-1) @ v, atol=.003, rtol=.035)
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=.003, rtol=.003)


@torch.inference_mode()
def test_projected_coarse_with_padding_count_bias_and_post_rank_cap():
    from lod_attention.kernels.glm_projected_prefill import projected_route_coarse
    torch.manual_seed(79)
    batch, heads, queries, slots = 2, 4, 137, 513
    q = (torch.randn(batch, heads, queries, 256, device="cuda") * .3).bfloat16()
    counts = torch.randint(1, 71, (batch, 1, slots, 1), device="cuda").float()
    state = (torch.randn(batch, 1, slots, 512, device="cuda") * .4 * counts).float()
    uk = (torch.randn(heads, 256, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(heads, 512, 256, device="cuda") / 512**.5).bfloat16()
    lengths = counts[..., 0].int().contiguous()
    buffers = {}
    selected, coarse = projected_route_coarse(q, state, counts, uk, uv,
        state_len=slots, scale=.0625, slot_lengths=lengths, max_open_leaf_tokens=32, buffers=buffers)
    means = (state / counts).bfloat16().float()
    keys = torch.einsum("bstl,hdl->bhstd", means, uk.float()).squeeze(2)
    # GEMM stores BF16 K/V; the reference uses those same rounded projections.
    keys = keys.bfloat16().float()
    values = torch.einsum("bstl,hld->bhstd", means, uv.float()).squeeze(2).bfloat16().float()
    logits = q.float() @ keys.transpose(-1, -2) * .0625
    # Bias is stored in model dtype, as in the established Kimi CK path.
    logits += counts[..., 0].log().bfloat16().float().unsqueeze(2)
    expected = logits.argsort(dim=-1, descending=True, stable=True)[..., :8]
    selected_counts = lengths.expand(batch, heads, slots).unsqueeze(2).expand(batch, heads, queries, slots).gather(-1, expected)
    assert torch.equal(selected, expected.masked_fill(selected_counts > 32, -1))
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3).float(),
        logits.softmax(-1) @ values, atol=.003, rtol=.035)
    torch.testing.assert_close(coarse.lse_0, logits.logsumexp(-1), atol=.003, rtol=.003)
    assert coarse.mean_k.size(2) == 640  # 128-key tile padding, not 1024.
    assert coarse.mean_v.stride(-2) == heads * 512  # interleaved, not copied.


@pytest.mark.parametrize("project_leaves", [False, True])
@pytest.mark.parametrize("recursive", [False, True, "int4"])
@torch.inference_mode()
def test_projected_remote_and_sink_matches_latent_then_projection(monkeypatch, project_leaves, recursive):
    monkeypatch.setenv("LOD_GLM_PROJECTED_LEAVES", str(int(project_leaves)))
    from lod_attention._config import LODConfig, PagedLODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention, KernelRecursivePagedLODAttention
    from lod_attention._profile import configure_engine
    from lod_attention.kernels.aiter_mla_prefill_attention import project_kimi_head_values
    torch.manual_seed(71)
    heads, history, queries = 4, 4096, 65
    cls = KernelRecursivePagedLODAttention if recursive else KernelTwoLevelLODAttention
    config = (PagedLODConfig(kv_bits=4, quant_group_size=4) if recursive == "int4"
              else PagedLODConfig() if recursive else LODConfig())
    engine = cls(config, query_heads=heads, key_value_heads=1, scale=.0625).cuda().eval()
    engine.head_dim = 512
    configure_engine(engine, family=ModelFamily.GLM53_FLASH,
        mode=(LODMode.THREE_TIER_INT4 if recursive == "int4" else
              LODMode.THREE_TIER_BF16 if recursive else LODMode.TWO_TIER),
        request_capacity=8192, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    latent = (torch.randn(1, 1, history, 512, device="cuda") * .4).bfloat16()
    cache = engine._build_cache_from_bf16(latent, latent, finalize_cache_for_decode=False)
    q = (torch.randn(1, heads, queries, 256, device="cuda") * .3).bfloat16()
    uk = (torch.randn(heads, 256, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(heads, 512, 256, device="cuda") / 512**.5).bfloat16()
    absorbed = torch.bmm(q[0], uk).unsqueeze(0)
    local = (torch.randn(1, 1, queries + 256, 512, device="cuda") * .4).bfloat16()
    kwargs = dict(state_len=int(cache["state_len"]), state_capacity=int(cache["state_k"].size(2)),
        page_cache=cache["page_cache"], sink_k=cache["sink_k"], sink_v=cache["sink_v"])
    kwargs["local_branch"] = engine._prefill_local_attention(absorbed, local, local, query_offset=256)
    empty = local[..., :0, :]
    baseline = engine._two_level_attention(absorbed, local, local,
        cache["state_k"], cache["state_v"], cache["counts"], None, empty, empty, **kwargs)
    expected = project_kimi_head_values(baseline, uv)
    engine._lod_kimi_expanded_prefill_chunk = q
    engine._lod_kimi_w_uk_t = uk
    engine._lod_kimi_w_uv = uv
    # The projected consumer must not read the shape-only latent query. In
    # serving this aliases the newly ingested latent, not an absorbed Q.
    incoming_q = torch.full_like(absorbed, float("nan")) if project_leaves else absorbed
    destination = torch.empty(1, queries, heads, 256, device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
    actual = engine._two_level_attention(incoming_q, local, local,
        cache["state_k"], cache["state_v"], cache["counts"], None, empty, empty,
        output_buffer=destination, **kwargs)
    assert actual.data_ptr() == destination.data_ptr()
    assert actual.isfinite().all()
    # Absorbing the query vs projecting K rounds at a different GEMM boundary.
    torch.testing.assert_close(actual.float(), expected.float(), atol=.008, rtol=.08)
    # Closing happens after top-eight selection. If all regions are closed,
    # retain the full centroid field, without NaNs or removing coarse mass.
    pages = kwargs["page_cache"]
    pages["slot_lengths"].fill_(1025)
    actual_closed = engine._two_level_attention(incoming_q, local, local,
        cache["state_k"], cache["state_v"], cache["counts"], None, empty, empty,
        output_buffer=destination, **kwargs)
    slots = int(cache["state_len"])
    count = cache["counts"][..., :slots, :].float().clamp_min(1)
    means = (cache["state_k"][..., :slots, :].float() / count).bfloat16().float()[:, 0]
    keys = torch.einsum("bsl,hdl->bhsd", means, uk.float()).bfloat16().float()
    values = torch.einsum("bsl,hld->bhsd", means, uv.float()).bfloat16().float()
    scores = q.float() @ keys.transpose(-1, -2) * .0625
    scores += count[..., 0].log().bfloat16().float().unsqueeze(2)
    coarse_out = (scores.softmax(-1) @ values).bfloat16().float()
    local_out = project_kimi_head_values(kwargs["local_branch"][0], uv).float()
    sink_scores = absorbed.float() @ cache["sink_k"].float().transpose(-1, -2) * .0625
    sink_values = torch.einsum("bsl,hld->bhsd", cache["sink_v"][:, 0].float(), uv.float()).bfloat16().float()
    sink_out = sink_scores.softmax(-1) @ sink_values
    branch_scores = torch.stack((scores.logsumexp(-1), kwargs["local_branch"][1], sink_scores.logsumexp(-1)), -1)
    weight = branch_scores.softmax(-1)
    expected_closed = (weight[..., :1] * coarse_out + weight[..., 1:2] * local_out
                       + weight[..., 2:] * sink_out)
    torch.testing.assert_close(actual_closed.float(), expected_closed, atol=.006, rtol=.06)


@pytest.mark.parametrize("queries", [37, 2053])
@torch.inference_mode()
def test_compact_projected_leaves_ragged_queries_and_graph_replay(queries):
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention.kernels.glm_compact_leaf_projection import project_compact_glm_leaves
    from lod_attention.kernels.glm_projected_prefill import projected_leaf_attention
    from lod_attention.kernels.paged_prefill import count_expert_routes
    torch.manual_seed(91)
    batch, heads, tokens, slots = 2, 4, 257, 9
    engine = KernelTwoLevelLODAttention(query_heads=heads, key_value_heads=1, scale=.0625).cuda().eval()
    engine.virtual_page_storage = True
    engine.leaf_layout = "expert"
    engine.leaf_page_size = engine.leaf_block_n = 16
    engine.leaf_block_m, engine.leaf_num_warps = 32, 2
    source = (torch.randn(batch, 1, tokens + 73, 512, device="cuda") * .4).bfloat16()[..., :tokens, :]
    owners = torch.zeros(batch, 1, tokens, dtype=torch.int32, device="cuda")
    owners[..., 51:102], owners[..., 102:135], owners[..., 135:] = 1, 4, 7
    cache = engine._new_page_cache(source, source, owners, state_capacity=slots,
        sequence_capacity=512, virtual_k=source, virtual_v=source)
    uk = (torch.randn(heads, 256, 512, device="cuda") / 512**.5).bfloat16()
    uv = (torch.randn(heads, 512, 256, device="cuda") / 512**.5).bfloat16()
    engine._lod_kimi_w_uk_t, engine._lod_kimi_w_uv = uk, uv
    q = (torch.randn(batch, queries + 11, heads, 256, device="cuda") * .3).bfloat16().permute(0, 2, 1, 3)[..., 5:5 + queries, :]
    assert not source.is_contiguous() and not q.is_contiguous()
    routes = torch.full((batch, heads, queries, 8), -1, device="cuda", dtype=torch.int32)
    routes[0, 0, :, 0], routes[0, 1, :, 0] = 0, 4
    routes[0, 2, :, :2] = torch.tensor([1, 7], device="cuda")
    routes[1, 1, :, 0], routes[1, 2, :, 0], routes[1, 3, :, 0] = 7, 0, 1
    routes[..., -3:, :] = -1
    counts, _ = count_expert_routes(routes, active_slots=slots)
    buffers = {}
    projected = project_compact_glm_leaves(source, uk, uv, cache, counts,
        active_slots=slots, hash_probes=engine._page_lookup_probes(cache), buffers=buffers)
    k, v, starts = projected
    expected_k = torch.einsum("btl,hdl->bhtd", source[:, 0].float(), uk.float()).bfloat16()
    expected_v = torch.einsum("btl,hld->bhtd", source[:, 0].float(), uv.float()).bfloat16()
    host_starts = starts.cpu()
    actual, lse = projected_leaf_attention(engine, q, routes, cache, active_slots=slots, buffers=buffers)
    for b in range(batch):
        for h in range(heads):
            selected = routes[b, h, 0][routes[b, h, 0] >= 0]
            if not selected.numel():
                continue
            mask = torch.isin(owners[b, 0], selected)
            for slot in selected.tolist():
                expert = (b * heads + h) * slots + slot
                a, z = int(host_starts[expert]), int(host_starts[expert + 1])
                leaf_mask = owners[b, 0] == slot
                torch.testing.assert_close(k.view(-1, 256)[a:z], expected_k[b, h, leaf_mask], atol=.004, rtol=.025)
                torch.testing.assert_close(v.view(-1, 256)[a:z], expected_v[b, h, leaf_mask], atol=.004, rtol=.025)
            scores = q[b, h, :-3].float() @ expected_k[b, h, mask].float().T * .0625
            reference = scores.softmax(-1) @ expected_v[b, h, mask].float()
            torch.testing.assert_close(actual[b, h, :-3].float(), reference, atol=.002, rtol=.035)
            torch.testing.assert_close(lse[b, h, :-3], scores.logsumexp(-1), atol=.001, rtol=.001)
    closed = ~routes.ge(0).any(-1)
    assert actual[closed].eq(0).all() and lse[closed].isneginf().all()
    # A captured prefix scan must observe changed selections, not a cached
    # host size or stale union from the previous chunk.
    counts.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = project_compact_glm_leaves(source, uk, uv, cache, counts,
            active_slots=slots, hash_probes=engine._page_lookup_probes(cache), buffers=buffers)
    counts.fill_(1)
    graph.replay()
    assert captured[2][-1].item() == batch * heads * tokens
    counts.zero_()
    graph.replay()
    assert captured[2].count_nonzero().item() == 0
