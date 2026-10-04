from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from lod_attention._config import LODMode, ModelFamily
from lod_attention._profile import configure_engine
from lod_attention.kernels.paged_leaf_attention import (
    dequantize_owned_virtual_paged_keys,
)
from lod_attention.kernels.aiter_mla_prefill_attention import expand_kimi_leaf_kv
import vllm_lod_plugin.models.kimi_k3 as kimi_k3
from vllm_lod_plugin.models.kimi_k3 import (
    absorb_query,
    pack_latent_record,
)
from vllm_lod_plugin.pool import VLLMLayerLODPool


def _dcp_schedule_fixture(rank: int = 0) -> VLLMLayerLODPool:
    pool = VLLMLayerLODPool.__new__(VLLMLayerLODPool)
    pool.dcp_world_size = 8
    pool.dcp_rank = rank
    pool.dcp_interleave_size = 1
    pool.engine = SimpleNamespace(
        state_growth_factor=16.0,
        state_min_len=256,
        chunk_len=256,
        local_len=512,
        prefill_chunk_len=16_384,
        prefill_local_len=16_640,
        prefill_state_update_len=16_384,
        decode_state_update_len=256,
        decode_cache_headroom=256,
    )
    pool._dcp_global_state_growth_factor = 16.0
    pool._dcp_global_state_min_len = 256
    pool._dcp_global_lengths = {
        name: int(getattr(pool.engine, name))
        for name in (
            "chunk_len",
            "local_len",
            "prefill_chunk_len",
            "prefill_local_len",
            "prefill_state_update_len",
            "decode_state_update_len",
            "decode_cache_headroom",
        )
    }
    return pool


def test_dcp_local_state_schedule_scales_every_token_count() -> None:
    pool = _dcp_schedule_fixture()
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_growth_factor == 16.0 / math.sqrt(8)
        assert pool.engine.state_min_len == 32
        assert (
            pool.engine.chunk_len,
            pool.engine.local_len,
            pool.engine.prefill_chunk_len,
            pool.engine.prefill_local_len,
            pool.engine.prefill_state_update_len,
            pool.engine.decode_state_update_len,
            pool.engine.decode_cache_headroom,
        ) == (32, 64, 2_048, 2_080, 2_048, 32, 32)

    assert pool.engine.state_growth_factor == 16.0
    assert pool.engine.state_min_len == 256
    assert (
        pool.engine.chunk_len,
        pool.engine.local_len,
        pool.engine.prefill_chunk_len,
        pool.engine.prefill_local_len,
        pool.engine.prefill_state_update_len,
        pool.engine.decode_state_update_len,
        pool.engine.decode_cache_headroom,
    ) == (256, 512, 16_384, 16_640, 16_384, 256, 256)


def test_independent_dcp_prefill_keeps_global_centroid_budget_per_rank() -> None:
    pool = _dcp_schedule_fixture()
    pool.kimi_local_dcp_prefill = True
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_min_len == 256
        assert pool.engine.prefill_state_update_len == 2_048
        assert pool.engine.decode_state_update_len == 32
        for global_length in (16_384, 65_536, 262_144):
            local_length = pool._dcp_local_length(global_length)
            local_budget = pool.engine.state_growth_factor * math.sqrt(local_length)
            global_budget = 16 * math.sqrt(global_length)
            assert math.isclose(local_budget, global_budget)
    assert pool.engine.state_growth_factor == 16.0


def test_shared_dcp_prefill_divides_centroid_budget_not_token_cadence() -> None:
    pool = _dcp_schedule_fixture()
    pool.kimi_local_dcp_prefill = True
    pool.kimi_shared_dcp_prefill = True
    with pool._dcp_local_state_schedule():
        assert pool.engine.state_growth_factor == 16.0 / math.sqrt(8)
        assert pool.engine.state_min_len == 32
        assert pool.engine.prefill_state_update_len == 2048
        assert pool.engine.decode_state_update_len == 32


def test_shared_global_routes_are_owned_exactly_once() -> None:
    from vllm_lod_plugin.models.kimi_k3_dcp_prefill import owned_routes

    slots = torch.tensor([[-1, 0, 31, 32, 88, 200, 254, 255]])
    mappings = [owned_routes(slots, rank=rank, local_states=32) for rank in range(8)]
    ownership = torch.stack(mappings).ge(0).sum(0)
    torch.testing.assert_close(ownership, slots.ge(0).long())
    for rank, mapping in enumerate(mappings):
        valid = mapping.ge(0)
        torch.testing.assert_close((mapping[valid] + 32 * rank).long(), slots[valid])


def test_fused_kimi_build_explicitly_emits_eight_candidates(monkeypatch) -> None:
    from lod_attention.kernels.aiter_prefill_attention import _specialized_kimi_coarse_mha_fwd

    builders = []

    def compile_ops(*args, **kwargs):
        builders.append(kwargs["gen_func"])
        return lambda function: function

    core = SimpleNamespace(
        compile_ops=compile_ops,
        get_args_of_build=lambda _: {"flags_extra_hip": [
            "-O3", "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1",
            "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=1",
        ]},
    )
    mha = SimpleNamespace(cmdGenFunc_mha_fwd=lambda *a, **kw: {
        "md_name": "mha_fwd", "blob_gen_cmd": ["generate --receipt 100 --output_dir /tmp"],
    })
    monkeypatch.setitem(sys.modules, "aiter.jit.core", core)
    monkeypatch.setitem(sys.modules, "aiter.ops.mha", mha)
    _specialized_kimi_coarse_mha_fwd.cache_clear()
    try:
        _specialized_kimi_coarse_mha_fwd(async_bias=True, fused_route=True)
        args = builders[0]()
        flags = args["flags_extra_hip"]
        assert "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=0" in flags
        assert "-DCK_TILE_FMHA_ROUTE_TILE_MAX_ONLY=1" not in flags
        assert "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=0" in flags
        assert "-DCK_TILE_FMHA_ROUTE_GLOBAL_TOPK=1" not in flags
        assert "-DCK_TILE_FMHA_ROUTE_TOPK=8" in flags
        assert args["md_name"].endswith("_asyncbias_v12")
    finally:
        _specialized_kimi_coarse_mha_fwd.cache_clear()


def test_independent_dcp_lse_merge_matches_concatenated_hybrid_field() -> None:
    from vllm_lod_plugin.models.kimi_k3_dcp_prefill import merge_partitions

    torch.manual_seed(19)
    # Exact promoted entries and closed coarse entries simply form a disjoint
    # softmax field. Independent choices on eight ranks need no global top-k.
    scores = torch.randn(8, 3, 5, 17)
    values = torch.randn(8, 3, 5, 17, 11)
    partial_lse = torch.logsumexp(scores, dim=-1)
    partial = (scores.softmax(-1)[..., None] * values).sum(-2)
    combined, lse = partial[0], partial_lse[0]
    for rank in range(1, 8):
        combined = merge_partitions(combined, lse, partial[rank], partial_lse[rank])
        lse = torch.logaddexp(lse, partial_lse[rank])
    total = torch.logsumexp(scores.permute(1, 2, 0, 3).flatten(-2), dim=-1)
    expected = (
        torch.exp(scores - total[None, ..., None])[..., None] * values
    ).sum((0, 3))
    torch.testing.assert_close(combined, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(lse, total)


def test_mla_refinement_returns_the_replaced_field_lse() -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        merge_aiter_mla_prefill_refinement,
    )
    from lod_attention.kernels.aiter_prefill_attention import AiterPrefillCoarse

    torch.manual_seed(23)
    heads, queries, states, width = 2, 19, 16, 128
    scores = torch.randn(1, heads, queries, states, device="cuda")
    means = torch.randn(1, heads, states, width, device="cuda").bfloat16()
    coarse_lse = scores.logsumexp(-1)
    coarse_out = torch.einsum("bhqs,bhsv->bqhv", scores.softmax(-1), means.float())
    coarse_out = coarse_out.bfloat16().contiguous()
    selected_scores, slots = scores.topk(8, dim=-1)
    fine_lse = torch.randn(1, heads, queries, 8, device="cuda")
    fine_out = torch.randn(1, heads, queries, 8, width, device="cuda").bfloat16()
    local_lse = torch.randn(1, heads, queries, device="cuda")
    local_out = torch.randn(1, heads, queries, width, device="cuda").bfloat16()
    mean_k = torch.zeros(1, 1, states, 576, device="cuda").bfloat16()
    coarse = AiterPrefillCoarse(
        output_0=coarse_out, lse_0=coarse_lse,
        output_1=coarse_out, lse_1=coarse_lse,
        mean_k=mean_k, mean_v=means,
        counts=torch.ones(1, 1, states, 1, device="cuda"),
        has_second_partition=False, selected_route_scores=selected_scores,
    )
    carrier = mean_k[..., :1, :].expand(1, heads, queries, 576)
    output, lse = merge_aiter_mla_prefill_refinement(
        carrier, mean_k[..., :0, :], means[..., :0, :], coarse, slots,
        fine_out, fine_lse, local_out, local_lse,
        kv_group_size=heads, scale=0.01, return_lse=True,
    )
    remaining = scores.clone().scatter_(-1, slots, -float("inf"))
    all_scores = torch.cat((remaining, fine_lse, local_lse[..., None]), dim=-1)
    expected_lse = all_scores.logsumexp(-1)
    expected = (
        torch.einsum("bhqs,bhsv->bhqv", torch.exp(remaining - expected_lse[..., None]), means.float())
        + (torch.exp(fine_lse - expected_lse[..., None])[..., None] * fine_out.float()).sum(-2)
        + torch.exp(local_lse - expected_lse)[..., None] * local_out.float()
    )
    torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)
    # Coarse subtraction starts with a BF16 accumulated output, so compare
    # with the appropriate rounding tolerance rather than claiming bit equality.
    torch.testing.assert_close(output.float(), expected, atol=0.012, rtol=0.03)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("direct_result", [False, True])
def test_projected_kimi_leaves_support_aggregated_and_empty_routes(monkeypatch, sparse, direct_result) -> None:
    monkeypatch.setenv("LOD_KIMI_SPARSE_LEAF_PROJECTION", "1" if sparse else "0")
    monkeypatch.setenv("LOD_KIMI_DIRECT_LEAF_RESULT", "1" if direct_result else "0")
    if not torch.cuda.is_available():
        return
    from lod_attention._config import LODConfig
    from lod_attention._engines import KernelTwoLevelLODAttention

    torch.manual_seed(41)
    heads, queries = 4, 17
    engine = KernelTwoLevelLODAttention(
        LODConfig(), query_heads=heads, key_value_heads=1, scale=192**-0.5,
    )
    engine.head_dim = 576
    configure_engine(
        engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
        request_capacity=512, has_query_norm=True, has_key_norm=False,
    )
    key = torch.randn(1, 1, 512, 576, device="cuda").bfloat16()
    cache = engine.build_cache_from_bf16(key, key[..., :512], final_cache_coverage=256).state
    assert cache["state_len"] == 255
    query = torch.randn(1, heads, queries, 192, device="cuda").bfloat16()
    w_uk_t = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    w_uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    engine._lod_kimi_expanded_prefill_chunk = query
    engine._lod_kimi_w_uk_t = w_uk_t
    engine._lod_kimi_w_uv = w_uv
    slots = torch.full((1, heads, queries, 8), -1, device="cuda", dtype=torch.int32)
    slots[:, 0, :8] = torch.arange(8, device="cuda", dtype=torch.int32)
    carrier = key[..., :1, :].expand(1, heads, queries, 576)
    output, lse = engine._paged_leaf_attention(
        carrier, slots, cache["page_cache"], active_slots=255, reduce_routes=True,
    )
    assert output.shape == (1, heads, queries, 128)
    assert output.dtype == query.dtype
    assert torch.isfinite(output).all()
    assert output[:, 1:].count_nonzero() == 0
    assert lse[:, 1:].isneginf().all()
    expanded_k, expanded_v = expand_kimi_leaf_kv(key[..., 1:9, :], w_uk_t, w_uv)
    scores = torch.einsum("bqd,bkd->bqk", query[:, 0, :8].float(), expanded_k[:, 0].float()) * engine.scaling
    expected = scores.softmax(-1) @ expanded_v[:, 0].float()
    torch.testing.assert_close(output[:, 0, :8].float(), expected, atol=0.012, rtol=0.03)
    torch.testing.assert_close(lse[:, 0, :8], scores.logsumexp(-1), atol=0.005, rtol=0.003)


@pytest.mark.parametrize("paged_directory", [False, True])
def test_sparse_leaf_projection_matches_only_selected_head_centroid_pairs(paged_directory) -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention._config import LODConfig
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention.kernels.kimi_sparse_leaf_projection import project_needed_kimi_leaves

    torch.manual_seed(67)
    batch, heads, tokens, slots = 2, 4, 257, 7
    engine = KernelTwoLevelLODAttention(
        LODConfig(), query_heads=heads, key_value_heads=1, scale=192**-0.5,
    )
    engine.head_dim = 576
    configure_engine(
        engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
        request_capacity=512, has_query_norm=True, has_key_norm=False,
    )
    engine.leaf_paged_directory = paged_directory
    source = torch.randn(batch, 1, tokens, 576, device="cuda").bfloat16()
    owners = (torch.arange(tokens, device="cuda") % slots).view(1, 1, -1)
    owners = owners.expand(batch, 1, tokens).contiguous()
    cache = engine._new_page_cache(
        source, source[..., :512], owners, state_capacity=slots,
        sequence_capacity=512, virtual_k=source, virtual_v=source[..., :512],
    )
    weights_k = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    weights_v = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    counts = torch.zeros(batch, heads, slots, device="cuda", dtype=torch.int32)
    counts[0, 0, 1] = 3
    counts[0, 2, 3] = 5
    counts[1, 1, 0] = 9
    counts[1, 3, 6] = 1
    buffers = {
        "kimi_leaf_expanded_k_token_major": source.new_full((batch * tokens * heads * 192,), -77),
        "kimi_leaf_expanded_v_token_major": source.new_full((batch * tokens * heads * 128,), -77),
    }
    actual_k, actual_v = project_needed_kimi_leaves(
        source, weights_k, weights_v, cache, counts,
        active_slots=slots, hash_probes=engine._page_lookup_probes(cache), buffers=buffers,
    )
    expected_k, expected_v = expand_kimi_leaf_kv(source, weights_k, weights_v)
    for b in range(batch):
        for h in range(heads):
            mask = counts[b, h].index_select(0, owners[b, 0].long()) > 0
            torch.testing.assert_close(actual_k[b, h, mask], expected_k[b, h, mask], atol=0.016, rtol=0.02)
            torch.testing.assert_close(actual_v[b, h, mask], expected_v[b, h, mask], atol=0.016, rtol=0.02)
            assert (actual_k[b, h, ~mask] == -77).all()
            assert (actual_v[b, h, ~mask] == -77).all()


def test_incremental_leaf_projection_preserves_prefix_and_resets_per_request(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.kimi_incremental_leaf_projection import expand_incremental_kimi_leaves
    import lod_attention.kernels.kimi_incremental_leaf_projection as module

    torch.manual_seed(73)
    source = torch.randn(1, 1, 73, 576, device="cuda").bfloat16()
    weights_k = (torch.randn(3, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    weights_v = (torch.randn(3, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    projected_lengths = []
    original = module.expand_kimi_leaf_kv

    def counted(key, uk, uv, **kwargs):
        projected_lengths.append(int(key.size(2)))
        return original(key, uk, uv, **kwargs)

    monkeypatch.setattr(module, "expand_kimi_leaf_kv", counted)
    cache = {}
    for length in (19, 33, 73, 73):
        actual = expand_incremental_kimi_leaves(
            source[..., :length, :], weights_k, weights_v, cache, head_begin=0,
        )
        expected = original(source[..., :length, :], weights_k, weights_v)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0.016, rtol=0.02)
    assert projected_lengths == [19, 14, 40]
    # A new request can reuse buffers and lengths, but not the previous
    # request's projected prefix, even if allocator pointers are recycled.
    other_source = -source
    actual = expand_incremental_kimi_leaves(
        other_source, weights_k, weights_v, {}, head_begin=0,
    )
    expected = original(other_source, weights_k, weights_v)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.016, rtol=0.02)
    assert projected_lengths == [19, 14, 40, 73]


@pytest.mark.parametrize("queries", [19, 128])
@pytest.mark.parametrize("tile_refine", [False, True])
def test_kimi_coarse_projects_fp32_sums_after_dividing(queries: int, monkeypatch, tile_refine) -> None:
    monkeypatch.setenv("LOD_KIMI_TILE_REFINE", "1" if tile_refine else "0")
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.aiter_mla_prefill_attention import (
        aiter_kimi_expanded_prefill_route_coarse_attention,
    )

    torch.manual_seed(47)
    heads, states = 4, 16
    query = torch.randn(1, heads, queries, 192, device="cuda").bfloat16()
    means = torch.randn(1, 1, states, 576, device="cuda").bfloat16()
    counts = torch.arange(1, states + 1, device="cuda").float().view(1, 1, states, 1)
    sums = means.float() * counts
    w_uk_t = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    w_uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    slots, coarse, _, _ = aiter_kimi_expanded_prefill_route_coarse_attention(
        query, sums, sums[..., :512].contiguous(), counts, w_uk_t, w_uv,
        state_len=states, scale=192**-0.5, normalize_route_query=False,
    )
    if coarse.ready_stream is not None:
        torch.cuda.current_stream().wait_stream(coarse.ready_stream)
    expanded_k, expanded_v = expand_kimi_leaf_kv(means, w_uk_t, w_uv)
    scores = torch.einsum("bhqd,bhsd->bhqs", query.float(), expanded_k.float()) * 192**-0.5
    scores += counts.log().view(1, 1, 1, states).bfloat16().float()
    expected = scores.softmax(-1) @ expanded_v.float()
    assert coarse.output_0.dtype == torch.bfloat16
    assert coarse.mean_v.dtype == torch.bfloat16
    torch.testing.assert_close(coarse.output_0.permute(0, 2, 1, 3).float(), expected, atol=0.012, rtol=0.03)
    torch.testing.assert_close(coarse.lse_0, scores.logsumexp(-1), atol=0.005, rtol=0.003)
    # Compare sets: ties/last-bit rounding need not preserve their order.
    torch.testing.assert_close(
        slots.sort(-1).values.long(), scores.topk(8, -1).indices.sort(-1).values,
        msg=f"routes={slots[0, 0, :2].tolist()}, expected={scores.topk(8, -1).indices[0, 0, :2].tolist()}",
    )


@pytest.mark.parametrize("offset,return_lse", [(0, True), (17, True), (0, False)])
def test_kimi_native_local_matches_ck_and_causal_reference(monkeypatch, offset, return_lse):
    if not torch.cuda.is_available():
        pytest.skip("GPU local-attention comparison")
    from lod_attention.kernels.aiter_mla_prefill_attention import aiter_kimi_local_prefill_attention

    torch.manual_seed(89)
    heads, tokens = 3, 257
    record = torch.randn(1, 1, tokens, 576, device="cuda").bfloat16()
    q = torch.randn(1, heads, tokens - offset, 192, device="cuda").bfloat16()
    uk = (torch.randn(heads, 128, 512, device="cuda") / math.sqrt(512)).bfloat16()
    uv = (torch.randn(heads, 512, 128, device="cuda") / math.sqrt(512)).bfloat16()
    carrier = record[..., offset:, :].expand(1, heads, -1, -1)
    kwargs = dict(query_offset=offset, scale=192**-0.5, expanded_q=q,
                  w_uk_t=uk, w_uv=uv, return_lse=return_lse)
    monkeypatch.setenv("LOD_KIMI_NATIVE_LOCAL_PREFILL", "0")
    ck_output, ck_lse = aiter_kimi_local_prefill_attention(carrier, record, **kwargs)
    monkeypatch.setenv("LOD_KIMI_NATIVE_LOCAL_PREFILL", "1")
    output, lse = aiter_kimi_local_prefill_attention(carrier, record, **kwargs)
    torch.testing.assert_close(output, ck_output, atol=0.006, rtol=0.03)
    k, v = expand_kimi_leaf_kv(record, uk, uv)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * 192**-0.5
    causal = torch.arange(tokens, device="cuda")[None, :] <= (
        torch.arange(tokens - offset, device="cuda")[:, None] + offset)
    scores.masked_fill_(~causal, -float("inf"))
    torch.testing.assert_close(output.float(), scores.softmax(-1) @ v.float(), atol=0.009, rtol=0.04)
    if return_lse:
        torch.testing.assert_close(lse, ck_lse, atol=0.005, rtol=0.003)
        torch.testing.assert_close(lse, scores.logsumexp(-1), atol=0.005, rtol=0.003)
    else:
        assert lse.numel() == 0


def test_dcp_catch_up_uses_rank_local_share_of_global_update() -> None:
    # The trigger is the same global per-request boundary on every rank.  At
    # DCP8 each rank then archives its 32 owned leaves: 256 global leaves in
    # aggregate.  Batch size never enters this calculation, so B8 performs
    # 8 * 256 source-token work at the aligned update.
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        pool.dcp_sharded = [True]
        pool.local_capacity = 96
        pool.metadata = [{"coverage": 8_160, "dcp_global_coverage": 65_280}]

        recent, target = pool._catch_up_target(0, 65_791)
        assert recent in (63, 64)
        assert target == 8_160

        recent, target = pool._catch_up_target(0, 65_792)
        assert recent == 64
        assert target == 8_192


def test_dcp_prefill_boundary_is_global_not_rank_local() -> None:
    # Immediately before the next global boundary, ranks own unequal local
    # lengths.  None may advance early merely because its local count rounded
    # up to another 32-token block.
    expected = []
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        expected.append(pool._dcp_local_length(65_280))
        assert pool._dcp_global_decode_coverage(65_791) == 65_280
        assert pool._dcp_global_decode_coverage(65_792) == 65_536
    assert expected == [8_160] * 8


def test_dcp_batch_does_not_divide_the_per_request_cadence() -> None:
    local_work = 0
    for rank in range(8):
        pool = _dcp_schedule_fixture(rank)
        pool.dcp_sharded = [True] * 8
        pool.local_capacity = 96
        pool.metadata = [
            {"coverage": 8_160, "dcp_global_coverage": 65_280}
            for _ in range(8)
        ]
        for slot in range(8):
            _recent, target = pool._catch_up_target(slot, 65_792)
            local_work += target - int(pool.metadata[slot]["coverage"])
    assert local_work == 8 * 256


def test_full_k3_profile_uses_launchable_unmasked_mla_geometry() -> None:
    engine = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=96, num_key_value_heads=1),
        head_dim=576,
    )
    configure_engine(
        engine,
        family=ModelFamily.KIMI_K3,
        mode=LODMode.THREE_TIER_BF16,
        request_capacity=16_640,
        has_query_norm=False,
        has_key_norm=True,
    )
    assert (engine.leaf_block_m, engine.leaf_block_n) == (32, 16)
    assert engine.prefill_aiter_route_coarse is True


def test_absorbed_latent_logits_equal_expanded_mla_logits() -> None:
    torch.manual_seed(7)
    tokens, heads, nope, latent_dim, direct = 11, 8, 64, 128, 32
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)

    absorbed = absorb_query(query, w_uk_t, nope_dim=nope)
    key = pack_latent_record(latent, direct_key)
    value = key[..., :latent_dim]

    expanded_nope = torch.einsum("tl,hpl->thp", latent, w_uk_t)
    expanded_key = torch.cat(
        (expanded_nope, direct_key.expand(-1, heads, -1)), dim=-1
    )
    expanded_logits = torch.einsum("thd,shd->hts", query, expanded_key)
    latent_logits = torch.einsum("thd,shd->hts", absorbed, key)

    # The two equivalent expressions use different contraction orders, so
    # float32 roundoff grows slightly with the latent width.
    torch.testing.assert_close(latent_logits, expanded_logits, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(value, latent.unsqueeze(1))
    assert value.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()


def test_projected_prefill_defers_absorbed_query_materialization(monkeypatch) -> None:
    # Query-deferred projected prefill is a full-K3 specialization.  The small
    # K3-for-All geometry intentionally uses the generic absorbed-query path.
    tokens, heads, nope, latent_dim, direct, value_dim = 5, 8, 128, 512, 64, 128
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)
    w_uv = torch.randn(heads, latent_dim, value_dim)
    observed: dict[str, object] = {}

    class Pool:
        direct_prefill_plan = ((0, 0, tokens, 0),)
        decode_enabled = False

        def direct_prefill(
            self,
            absorbed_shape_carrier,
            key,
            value,
            output,
            **kwargs,
        ):
            observed["carrier_stride"] = absorbed_shape_carrier.stride()
            observed["key_shape"] = tuple(key.shape)
            observed["value_aliases_key"] = (
                value.untyped_storage().data_ptr()
                == key.untyped_storage().data_ptr()
            )
            observed.update(kwargs)
            output.fill_(3)
            return output

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(),
        W_UK_T=w_uk_t,
        W_UV=w_uv,
        qk_nope_head_dim=nope,
        num_heads=heads,
        v_head_dim=value_dim,
    )

    def unexpected_absorption(*_args, **_kwargs):
        raise AssertionError("projected prefill must not materialize absorbed Q")

    monkeypatch.setattr(kimi_k3, "absorb_query", unexpected_absorption)
    result = kimi_k3._run_lod_mla(
        layer,
        query,
        latent,
        direct_key,
        (tokens, heads * value_dim),
        None,
    )

    assert result.shape == (tokens, heads * value_dim)
    assert torch.all(result == 3)
    assert observed["carrier_stride"][1] == 0
    assert observed["key_shape"] == (tokens, 1, latent_dim + direct)
    assert observed["value_aliases_key"] is True
    assert observed["mla_query"] is query
    assert observed["mla_w_uk_t"] is w_uk_t
    assert observed["mla_w_uv"] is w_uv
    assert observed["defer_mla_query_absorption"] is True


def test_graph_warmup_rows_do_not_enter_smaller_decode_pool() -> None:
    tokens, heads, nope, latent_dim, direct = 16, 2, 4, 8, 2
    query = torch.randn(tokens, heads, nope + direct)
    latent = torch.randn(tokens, latent_dim)
    direct_key = torch.randn(tokens, 1, direct)

    class Pool:
        direct_prefill_plan = None
        decode_enabled = True
        max_requests = 1

        def decode(self, *_args, **_kwargs):
            raise AssertionError("synthetic graph rows must not enter decode")

    def copy_output(attention_output, output):
        output.copy_(attention_output.reshape(tokens, heads * latent_dim))

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(),
        W_UK_T=torch.randn(heads, nope, latent_dim),
        qk_nope_head_dim=nope,
        num_heads=heads,
        v_head_dim=latent_dim,
        _v_up_proj=copy_output,
    )
    result = kimi_k3._run_lod_mla(
        layer,
        query,
        latent,
        direct_key,
        (tokens, heads * latent_dim),
        None,
    )
    assert torch.count_nonzero(result) == 0


def test_single_owner_decode_dispatches_full_heads_in_mla_tiles() -> None:
    tokens, heads, nope, latent_dim, direct = 1, 96, 128, 512, 64
    observed = {}

    class Pool:
        direct_prefill_plan = None
        decode_enabled = True
        kimi_head_tiled_decode = True
        max_requests = 1

        def decode_dcp(self, query, key, value, output):
            observed["query_shape"] = tuple(query.shape)
            observed["key_shape"] = tuple(key.shape)
            observed["value_is_latent_view"] = (
                key.untyped_storage().data_ptr()
                == value.untyped_storage().data_ptr()
            )
            output.fill_(2)
            return output, torch.zeros(tokens, heads)

        def decode(self, *_args, **_kwargs):
            raise AssertionError("do not compile a 96-head generic route tile")

    layer = SimpleNamespace(
        _vllm_lod_pool=Pool(), W_UK_T=torch.randn(heads, nope, latent_dim),
        qk_nope_head_dim=nope, num_heads=heads, v_head_dim=latent_dim,
        _v_up_proj=lambda source, target: target.copy_(source.flatten(1)),
    )
    output = kimi_k3._run_lod_mla(
        layer, torch.randn(tokens, heads, nope + direct),
        torch.randn(tokens, latent_dim), torch.randn(tokens, 1, direct),
        (tokens, heads * latent_dim), None,
    )
    assert observed == {
        "query_shape": (1, 96, 576), "key_shape": (1, 1, 576),
        "value_is_latent_view": True,
    }
    assert torch.all(output == 2)


def test_latent_centroid_is_exact_expanded_kv_centroid() -> None:
    torch.manual_seed(11)
    count, heads, nope, latent_dim, direct, value_dim = 13, 8, 64, 128, 32, 64
    latent = torch.randn(count, latent_dim)
    direct_key = torch.randn(count, direct)
    w_uk_t = torch.randn(heads, nope, latent_dim)
    w_uv = torch.randn(heads, latent_dim, value_dim)
    query = torch.randn(heads, nope + direct)

    q_absorbed = absorb_query(query.unsqueeze(0), w_uk_t, nope_dim=nope)[0]
    summed_key = torch.cat((latent.sum(0), direct_key.sum(0)))
    latent_score = torch.einsum("hd,d->h", q_absorbed, summed_key)

    expanded_key_sum = torch.cat(
        (
            torch.einsum("l,hpl->hp", latent.sum(0), w_uk_t),
            direct_key.sum(0).expand(heads, -1),
        ),
        dim=-1,
    )
    expanded_score = torch.einsum("hd,hd->h", query, expanded_key_sum)
    torch.testing.assert_close(latent_score, expanded_score, atol=3e-5, rtol=3e-5)

    latent_mean = latent.mean(0).expand(heads, -1)
    projected_mean = torch.einsum("hl,hlv->hv", latent_mean, w_uv)
    expanded_values = torch.einsum("tl,hlv->thv", latent, w_uv)
    torch.testing.assert_close(
        projected_mean,
        expanded_values.mean(0),
        atol=2e-5 * math.sqrt(count),
        rtol=2e-5,
    )


@pytest.mark.parametrize("fused", [False, True])
def test_multihead_leaf_expansion_writes_projected_key_prefix(monkeypatch, fused) -> None:
    monkeypatch.setenv("LOD_KIMI_FUSED_LEAF_KV", "1" if fused else "0")
    if not torch.cuda.is_available():
        return
    torch.manual_seed(13)
    tokens, heads = 19, 12
    latent_key = torch.randn(
        1, 1, tokens, 576, dtype=torch.bfloat16, device="cuda"
    )
    w_uk_t = torch.randn(
        heads, 128, 512, dtype=torch.bfloat16, device="cuda"
    )
    w_uv = torch.randn(heads, 512, 128, dtype=torch.bfloat16, device="cuda")
    expanded_k, expanded_v = expand_kimi_leaf_kv(latent_key, w_uk_t, w_uv)

    latent = latent_key[0, 0, :, :512].float()
    expected_nope = torch.einsum("tl,hpl->htp", latent, w_uk_t.float())
    expected_direct = latent_key[0, 0, :, 512:].expand(heads, -1, -1)
    expected_v = torch.einsum("tl,hlv->htv", latent, w_uv.float())
    torch.testing.assert_close(
        expanded_k[0, ..., :128].float(), expected_nope, atol=0.5, rtol=0.02
    )
    torch.testing.assert_close(expanded_k[0, ..., 128:], expected_direct)
    torch.testing.assert_close(
        expanded_v[0].float(), expected_v, atol=0.5, rtol=0.02
    )


def test_count_channel_equals_centroid_mass_bias() -> None:
    """A padded dot-product channel exactly represents ``+ log(count)``."""
    torch.manual_seed(19)
    queries, states, heads = 7, 11, 3
    scale = 192**-0.5
    query = torch.randn(queries, heads, 192)
    key = torch.randn(states, heads, 192)
    counts = torch.randint(1, 128, (states,), dtype=torch.int64).float()

    expected = torch.einsum("qhd,khd->hqk", query, key) * scale
    expected += counts.log()[None, None, :]

    augmented_query = torch.zeros(queries, heads, 256)
    augmented_key = torch.zeros(states, heads, 256)
    augmented_query[..., :192] = query
    augmented_key[..., :192] = key
    augmented_query[..., 192] = 1.0 / scale
    augmented_key[..., 192] = counts.log()[:, None]
    actual = torch.einsum(
        "qhd,khd->hqk", augmented_query, augmented_key
    ) * scale
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("states", [177, 1039])
def test_tile_max_refinement_recovers_exact_global_top_eight(states) -> None:
    if not torch.cuda.is_available():
        return
    from lod_attention.kernels.kimi_route_tile_refine import refine_kimi_centroid_tiles
    from lod_attention.kernels.aiter_prefill_attention import _reduce_route_candidates

    torch.manual_seed(79)
    batch, heads, queries = 2, 3, 37
    q = torch.randn(batch, heads, queries, 192, device="cuda").bfloat16()
    k = torch.randn(batch, states, heads, 192, device="cuda").bfloat16()
    logs = torch.randint(1, 50, (batch, states), device="cuda").float().log().bfloat16()
    scale = 192**-0.5
    scores = torch.einsum("bhqd,bshd->bhqs", q.float(), k.float()) * scale
    scores += logs.float()[:, None, None, :]
    tiles = math.ceil(states / 128)
    padded = torch.nn.functional.pad(scores, (0, tiles * 128 - states), value=-float("inf"))
    maxima = padded.view(batch, heads, queries, tiles, 128).amax(-1)
    candidates = torch.full((batch, heads, tiles, 16, queries), float("nan"), device="cuda")
    # No other candidate channel is valid or may be read by the tile selector.
    candidates[:, :, :, 0] = maxima.transpose(-1, -2) / math.log(2)
    refined = refine_kimi_centroid_tiles(candidates, q, k, logs, state_len=states, scale=scale)
    actual_slots, _, _, actual_scores = _reduce_route_candidates(
        refined, state_len=states, head_dim=192, emit_metadata=False,
    )
    expected_slots = scores.topk(8, dim=-1).indices
    torch.testing.assert_close(actual_slots.sort(-1).values, expected_slots.sort(-1).values)
    expected_scores = scores.gather(-1, actual_slots)
    torch.testing.assert_close(actual_scores, expected_scores, atol=0.002, rtol=0.002)


def test_dcp_int4_page_dequantization_restores_owned_records() -> None:
    if not torch.cuda.is_available():
        return
    batch, heads, pages, page_size, head_dim = 1, 2, 2, 16, 8
    leaves = pages * page_size
    page_indices = (
        torch.arange(leaves, dtype=torch.int32, device="cuda")
        .view(batch, 1, pages, page_size)
        .expand(batch, heads, pages, page_size)
        .contiguous()
    )
    page_counts = torch.full(
        (batch, heads, pages), page_size, dtype=torch.int32, device="cuda"
    )
    next_page = torch.full(
        (batch, heads), pages, dtype=torch.int32, device="cuda"
    )
    # Low nibbles reconstruct to the page centroid; high nibbles add one.
    packed_keys = torch.full(
        (batch, heads, leaves, head_dim // 2),
        0x98,
        dtype=torch.uint8,
        device="cuda",
    )
    page_scales = torch.ones(
        batch, heads, pages, head_dim // 4, dtype=torch.bfloat16, device="cuda"
    )
    quantized_sums = torch.full(
        (batch, heads, pages, head_dim),
        32,
        dtype=torch.int8,
        device="cuda",
    )
    summary_scales = torch.full(
        (batch, heads, pages, head_dim // 4),
        0.5,
        dtype=torch.bfloat16,
        device="cuda",
    )

    for rank in range(2):
        actual = dequantize_owned_virtual_paged_keys(
            page_indices,
            page_counts,
            next_page,
            packed_keys,
            page_scales,
            quantized_sums,
            summary_scales,
            source_slot=0,
            sink_len=0,
            local_length=leaves // 2,
            dcp_rank=rank,
            dcp_world_size=2,
            dcp_interleave_size=4,
        )
        expected = torch.empty_like(actual)
        expected[..., 0::2] = 1.0
        expected[..., 1::2] = 2.0
        torch.testing.assert_close(actual, expected)
