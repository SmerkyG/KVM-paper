"""DCP sharding is a storage change, not a centroid/routing-policy change."""

from types import SimpleNamespace

import pytest
import torch

from vllm_lod_plugin.models.kimi_k3_sharded_prefill import (
    add_global_counts, append_page, archive_workspace, b1_leaf_storage, combine_prefill_partials, gather_prefill, initial_page,
    owned_history, owned_slice, projection_scope, reduce_scatter_prefill,
)


def test_shared_archive_scratch_grows_only_at_power_of_two_boundaries():
    buffers = {}
    first = archive_workspace(buffers, "keys", (2, 1, 37, 576), token_axis=2,
                              dtype=torch.bfloat16, device=torch.device("cpu"))
    first.fill_(3)
    second = archive_workspace(buffers, "keys", (2, 1, 63, 576), token_axis=2,
                               dtype=first.dtype, device=first.device)
    assert first.data_ptr() == second.data_ptr()
    assert first.is_contiguous() and second.is_contiguous()
    assert buffers["keys"].numel() == 2 * 64 * 576
    torch.testing.assert_close(second.reshape(-1)[:first.numel()], first.reshape(-1))
    third = archive_workspace(buffers, "keys", (2, 1, 65, 576), token_axis=2,
                              dtype=first.dtype, device=first.device)
    assert third.data_ptr() != first.data_ptr()
    assert buffers["keys"].numel() == 2 * 128 * 576


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("coverage,total,sink_len", ((256, 7951, 1), (768, 1025, 1), (512, 1024, 0)))
def test_bf16_shadow_extraction_does_not_read_unwritten_archive_tail(rank, coverage, total, sink_len):
    from lod_attention._engines import KernelLODCache
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import owned_bf16_shadow_records
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    records = torch.arange(total).view(1, 1, total, 1).expand(-1, -1, -1, 576).float()
    # A fixed pool reserves the whole prompt but fills only the archived
    # prefix. Poison the rest so extracting by capacity cannot pass unnoticed.
    archive = torch.full_like(records[..., sink_len:, :], float("nan"))
    archive[..., :coverage - sink_len, :].copy_(records[..., sink_len:coverage, :])
    source = dict(total_len=total, coverage=coverage, recent_len=total - coverage,
        sink_k=records[..., :sink_len, :], recent_k=records[..., coverage:, :],
        page_cache=dict(leaf_k=archive, leaf_v=archive[..., :512]))
    pool = SimpleNamespace(is_absorbed_mla=True, dcp_interleave_size=1,
        dcp_rank=rank, dcp_world_size=8, value_dim=512,
        _dcp_local_length=lambda length: max(0, (length + 7 - rank) // 8))
    expected = records[..., rank::8, :]
    for key, value, length in (
        owned_bf16_shadow_records(pool, source),
        VLLMLayerLODPool._dcp_local_records(pool, KernelLODCache(source)),
    ):
        torch.testing.assert_close(key, expected)
        torch.testing.assert_close(value, expected[..., :512])
        assert key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        assert length == total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP absorbed DCP decode numerical check")
@pytest.mark.parametrize("slot", (0, 1))
@torch.inference_mode()
def test_actual_dcp_pool_decode_matches_dense_for_an_exact_short_history(slot):
    """Exercise real cache conversion, 96-head decoder, and current-token ownership."""
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    torch.manual_seed(917)
    device = torch.device("cuda")
    history = torch.randn(1, 1, 128, 576, device=device, dtype=torch.bfloat16)
    query = torch.randn(1, 96, 576, device=device, dtype=torch.bfloat16)
    current = torch.randn(1, 1, 576, device=device, dtype=torch.bfloat16)
    layer = SimpleNamespace(num_heads=12, num_kv_heads=1, head_size=576,
        kv_lora_rank=512, scale=1 / 192**0.5, _vllm_lod_absorbed_mla=True)
    parts, lses = [], []
    for rank in range(8):
        pool = VLLMLayerLODPool(layer, settings=VLLMLODSettings(),
            max_requests=2, request_capacity=4096,
            active_indices=torch.full((2,), slot, device=device, dtype=torch.long),
            dtype=torch.bfloat16, device=device, dcp_world_size=8, dcp_rank=rank,
            dcp_group=SimpleNamespace(world_size=8), request_owner_prefill=False)
        cache = pool.engine.build_cache_from_bf16(history, history[..., :512])
        pool._install_dcp_local_from_cache(slot, cache)
        pool.active_decode_rows = (slot,)
        # No centroid exists in this short exact tail, so global route merging
        # is empty. Evaluate the eight physical slices serially on one GPU;
        # the real DCP softmax merge is computed explicitly below.
        pool.kimi_local_dcp_prefill = True
        result = torch.empty(1, 96, 512, device=device, dtype=torch.bfloat16)
        result, lse = pool.decode_dcp(query, current, current[..., :512], result)
        local = history[0, 0, rank::8].float()
        if rank == 0:
            local = torch.cat((local, current[0].float()), dim=0)
        logits = torch.einsum("bhd,td->bht", query.float(), local) * layer.scale
        local_reference = torch.einsum("bht,td->bhd", torch.softmax(logits, dim=-1), local[:, :512])
        print("DCP_REFERENCE", rank,
              "output_max_error", float((result.float() - local_reference).abs().max()),
              "lse_max_error", float((lse - torch.logsumexp(logits, -1)).abs().max()), flush=True)
        torch.testing.assert_close(result.float(), local_reference, atol=0.01, rtol=0.02)
        torch.testing.assert_close(lse, torch.logsumexp(logits, -1), atol=1e-4, rtol=1e-4)
        parts.append(result.float().clone())
        lses.append(lse.float().clone())
        torch.cuda.synchronize()
    lses = torch.stack(lses)
    combined = (torch.stack(parts) * torch.softmax(lses, dim=0)[..., None]).sum(0)
    full = torch.cat((history[0, 0], current[0]), dim=0).float()
    weights = torch.softmax(torch.einsum("bhd,td->bht", query.float(), full) * layer.scale, dim=-1)
    expected = torch.einsum("bht,td->bhd", weights, full[:, :512])
    torch.testing.assert_close(combined, expected, atol=0.01, rtol=0.02)


@pytest.mark.parametrize("shape,dim", (((1, 12, 37, 192), 1), ((2, 12, 37, 192), 1), ((12, 128, 512), 0)))
def test_prefill_gather_preserves_batch_and_head_order(shape, dim):
    value = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    shards = [value + rank * value.numel() for rank in range(8)]

    def gather(output, source):
        torch.testing.assert_close(source, value)
        output.copy_(torch.cat(shards, dim=0))

    group = SimpleNamespace(world_size=8, device_communicator=SimpleNamespace(
        pynccl_comm=SimpleNamespace(disabled=False, all_gather=gather)))
    torch.testing.assert_close(gather_prefill(group, value, dim=dim), torch.cat(shards, dim=dim))
    workspace = torch.empty(8, *shape)
    torch.testing.assert_close(gather_prefill(group, value, dim=dim, buffer=workspace), torch.cat(shards, dim=dim))
    with pytest.raises(ValueError, match="workspace"):
        gather_prefill(group, value, dim=dim, buffer=workspace[..., :-1])


def test_projection_scope_restores_parameter_registration_even_on_failure():
    engine = torch.nn.Module()
    uk = torch.nn.Parameter(torch.ones(2, 3))
    uv = torch.nn.Parameter(torch.ones(3, 2))
    engine._lod_kimi_w_uk_t = uk
    engine._lod_kimi_w_uv = uv
    with pytest.raises(RuntimeError, match="test failure"):
        with projection_scope(engine, torch.zeros(1), uk.detach(), uv.detach()):
            assert not isinstance(engine._lod_kimi_w_uk_t, torch.nn.Parameter)
            raise RuntimeError("test failure")
    assert engine._lod_kimi_w_uk_t is uk
    assert engine._lod_kimi_w_uv is uv
    assert dict(engine.named_parameters())["_lod_kimi_w_uk_t"] is uk
    assert not hasattr(engine, "_lod_kimi_expanded_prefill_chunk")


def test_projection_scope_bounds_shared_scratch_and_restores_previous_limit():
    engine = SimpleNamespace(_lod_kimi_prefill_head_group_limit=48)
    q = torch.zeros(1, 96, 1, 192)
    with projection_scope(engine, q, torch.zeros(1), torch.zeros(1), head_group_limit=12):
        assert engine._lod_kimi_prefill_head_group_limit == 12
    assert engine._lod_kimi_prefill_head_group_limit == 48
    with pytest.raises(ValueError, match="divide"):
        with projection_scope(engine, q, torch.zeros(1), torch.zeros(1), head_group_limit=7):
            pass
    assert engine._lod_kimi_prefill_head_group_limit == 48


@pytest.mark.parametrize("reducer", ("topk", "serial", "vector"))
def test_decode_route_cap_honors_stride_zero_mla_heads_and_nonzero_pool_row(reducer):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU route reduction")
    from lod_attention.kernels import paged_routing as kernels

    # Six virtual 16-query-head tiles share one physical MLA KV head. Row
    # seven catches the old (cache_row * virtual_heads) out-of-bounds read.
    heads, virtual_heads, states, groups = 96, 6, 128, 8
    indices = (torch.arange(groups, device="cuda")[:, None] * 16
               + torch.arange(8, device="cuda")[None, :])
    candidates = indices.expand(1, heads, -1, -1).contiguous().long()
    scores = (1000 - candidates).float()
    lengths = torch.ones(8, 1, states, device="cuda", dtype=torch.int32)
    lengths[7, 0, :8:2] = 1025
    lengths = lengths.expand(-1, virtual_heads, -1)
    cache_rows = torch.tensor([7], device="cuda", dtype=torch.long)
    slots = torch.empty(1, heads, 1, 8, device="cuda", dtype=torch.long)
    top_scores = torch.empty_like(slots, dtype=torch.float32)
    common = dict(
        QUERY_HEADS=heads, KV_HEADS=virtual_heads, KV_GROUP_SIZE=16,
        STATE_CAPACITY=states, ROUTE_COUNT=8, OPEN_COUNT=8,
        MAX_OPEN_LEAVES=1024,
        SLOT_LENGTH_BATCH_STRIDE=lengths.stride(0),
        SLOT_LENGTH_HEAD_STRIDE=lengths.stride(1), num_warps=2,
    )
    if reducer == "topk":
        dummy = torch.zeros(1, device="cuda", dtype=torch.int32)
        kernels._reduce_decode_route_topk_kernel[heads,](
            scores, candidates, slots, top_scores, groups,
            dummy, dummy, dummy, dummy, lengths, cache_rows,
            GROUP_N=16, UNION_CAPACITY=1, MAX_SEGMENTS=groups,
            CANDIDATE_BLOCK=64, **common,
        )
    else:
        group_out = torch.zeros(1, heads, groups, 128, device="cuda")
        group_lse = torch.zeros(1, heads, groups, device="cuda")
        coarse_out = torch.empty(1, heads, 128, device="cuda")
        coarse_lse = torch.empty(1, heads, device="cuda")
        args = (scores, candidates, group_out, group_lse, slots, top_scores,
                coarse_out, coarse_lse, lengths, cache_rows, groups)
        if reducer == "serial":
            kernels._reduce_decode_route_coarse_kernel[heads,](
                *args, VALUE_DIM=128, MAX_GROUPS=groups, CANDIDATE_TILE=64,
                APPLY_MASS_CUTOFF=False, LOG_MASS_FRACTION=0., **common,
            )
        else:
            kernels._reduce_decode_route_coarse_vector_topk_kernel[heads,](
                *args, groups, HEAD_DIM=128, MAX_SEGMENTS=groups,
                CANDIDATE_BLOCK=64, SEGMENT_BLOCK=groups,
                APPLY_MASS_CUTOFF=False, LOG_MASS_FRACTION=0., **common,
            )
    expected = torch.tensor([-1, 1, -1, 3, -1, 5, -1, 7]).view(1, 1, 1, 8).expand_as(slots)
    torch.testing.assert_close(slots.cpu(), expected)


@pytest.mark.parametrize("shape,dim", (((1, 96, 37, 128), 1), ((2, 96, 37, 128), 1), ((2, 37, 96, 128), 2)))
def test_prefill_reduce_scatter_preserves_head_owners_with_reused_scratch(shape, dim):
    value = torch.arange(torch.tensor(shape).prod()).reshape(shape).float()
    # Fake SUM over eight ranks with distinguishable values on each sender.
    summed = value * 8 + sum(range(8)) * value.numel()
    for rank in range(8):
        def scatter(output, source):
            torch.testing.assert_close(source, value.movedim(dim, 0).contiguous())
            output.copy_(summed.movedim(dim, 0).chunk(8, dim=0)[rank])

        group = SimpleNamespace(world_size=8, device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=False, reduce_scatter=scatter)))
        expected = summed.chunk(8, dim=dim)[rank]
        torch.testing.assert_close(reduce_scatter_prefill(group, value, dim=dim), expected)
        workspace = torch.empty_like(expected.movedim(dim, 0).contiguous())
        got = reduce_scatter_prefill(group, value, dim=dim, buffer=workspace)
        torch.testing.assert_close(got, expected)
        with pytest.raises(ValueError, match="workspace"):
            reduce_scatter_prefill(group, value, dim=dim, buffer=workspace[..., :-1])


def test_prefill_collectives_fall_back_when_pynccl_is_disabled():
    value = torch.ones(1, 8, 2, 3)
    calls = []
    group = SimpleNamespace(device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(disabled=True)),
                            all_gather=lambda tensor, dim: calls.append(("gather", dim)) or tensor,
                            reduce_scatter=lambda tensor, dim: calls.append(("scatter", dim)) or tensor)
    assert gather_prefill(group, value, dim=1) is value
    assert reduce_scatter_prefill(group, value, dim=1) is value
    assert calls == [("gather", 1), ("scatter", 1)]


@pytest.mark.parametrize("batch", (1, 2))
def test_shared_prefill_partial_combiner_keeps_head_order_and_empty_owners(monkeypatch, batch):
    import lod_attention.kernels.kimi_sharded_prefill as kernels

    torch.manual_seed(913)
    world, heads, queries = 8, 2, 37
    values = torch.randn(world, batch, world * heads, queries, 128)
    lses = torch.randn(world, batch, world * heads, queries)
    # Some owners have no chosen centroid; some rows have no fine field at all.
    lses[2:4] = -torch.inf
    lses[..., 0] = -torch.inf
    total = torch.logsumexp(lses, dim=0)
    weights = torch.where(torch.isfinite(total)[None], (lses - total[None]).exp(), 0)
    expected = (values * weights[..., None]).sum(0)

    def weight(output, all_lses, *, rank, total_lse):
        torch.testing.assert_close(all_lses, lses)
        total_lse.copy_(total)
        output.mul_(weights[rank, ..., None])
        return total_lse

    monkeypatch.setattr(kernels, "weight_prefill_partials_", weight)
    buffers = {}
    for rank in range(world):
        def gather(output, source):
            torch.testing.assert_close(source, lses[rank])
            output.copy_(lses.reshape(world * batch, world * heads, queries))

        def scatter(output, source):
            torch.testing.assert_close(source, (values[rank] * weights[rank, ..., None]).movedim(1, 0).contiguous())
            output.copy_(expected[:, rank * heads:(rank + 1) * heads].movedim(1, 0))

        group = SimpleNamespace(world_size=world, device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=False, all_gather=gather, reduce_scatter=scatter)))
        pool = SimpleNamespace(dcp_world_size=world, dcp_rank=rank, dcp_group=group)
        got, got_lse = combine_prefill_partials(pool, values[rank].clone(), lses[rank].clone(),
                                              heads=heads, buffers=buffers)
        torch.testing.assert_close(got, expected[:, rank * heads:(rank + 1) * heads])
        torch.testing.assert_close(got_lse, total[:, rank * heads:(rank + 1) * heads])


def test_shared_mixed_decode_uses_global_heads_lse_and_updates_only_decode_rows(monkeypatch):
    import sys
    from vllm_lod_plugin.models.kimi_k3 import absorb_query
    from vllm_lod_plugin.models.kimi_k3_dcp_prefill import _shared_mixed_decode

    torch.manual_seed(912)
    own_heads, world = 2, 8
    query = torch.randn(4, own_heads, 66)
    key = torch.randn(4, 1, 576)
    uk = torch.randn(own_heads * world, 2, 512)
    uv = torch.randn(own_heads, 512, 128)
    layer = SimpleNamespace(W_UK_T_dcp_qrep=uk, W_UV=uv, qk_nope_head_dim=2)
    plan = ((2, 0, 1, 65536), (5, 3, 4, 16384))
    positions = torch.tensor([0, 3])
    gathered = torch.cat([query.index_select(0, positions) + rank for rank in range(world)], dim=1)

    def gather(value, dim):
        assert dim == 1
        torch.testing.assert_close(value, query.index_select(0, positions))
        return gathered

    partial = torch.randn(2, own_heads * world, 512)
    lses = torch.randn(2, own_heads * world)

    def decode(q, k, v, output):
        torch.testing.assert_close(q, absorb_query(gathered, uk, nope_dim=2))
        torch.testing.assert_close(k, key.index_select(0, positions))
        assert v.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        output.copy_(partial)
        return output, lses

    def combine(out, lse, group, *, is_lse_base_on_e):
        assert is_lse_base_on_e
        torch.testing.assert_close(out, partial)
        torch.testing.assert_close(lse, lses)
        return out[:, :own_heads] + 1

    monkeypatch.setitem(sys.modules, "vllm.v1.attention.ops.dcp", SimpleNamespace(cp_lse_ag_out_rs=combine))
    pool = SimpleNamespace(
        active_decode_rows=(2, 5), dcp_group=SimpleNamespace(all_gather=gather), decode_dcp=decode,
        _dcp_local_length=lambda length: (length + 7) // 8,
        metadata={2: dict(coverage=8000), 5: dict(coverage=2000)},
        dcp_global_lens=torch.zeros(8, dtype=torch.int32),
    )
    output = torch.zeros(4, own_heads, 128)
    _shared_mixed_decode(layer, pool, query, key, output, plan)
    expected = torch.einsum("bhd,hdv->bhv", partial[:, :own_heads] + 1, uv)
    torch.testing.assert_close(output.index_select(0, positions), expected, rtol=1e-4, atol=1e-4)
    assert output[1:3].count_nonzero() == 0
    assert pool.metadata[2] == dict(coverage=8000, total_len=8193, recent_len=193, dcp_global_total_len=65537)
    assert pool.metadata[5] == dict(coverage=2000, total_len=2049, recent_len=49, dcp_global_total_len=16385)
    assert pool.dcp_global_lens.tolist() == [0, 0, 65537, 0, 0, 16385, 0, 0]


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("lazy_absorption", (False, True))
def test_direct_mixed_decode_combines_all_dcp_slices(monkeypatch, packed, lazy_absorption):
    import sys
    from vllm_lod_plugin.models.kimi_k3 import absorb_query
    from vllm_lod_plugin.pool import VLLMLayerLODPool

    torch.manual_seed(913)
    heads, world, latent = 2, 8, 512
    raw_q = torch.randn(4, heads, 66)
    uk = torch.randn(heads, 2, latent)
    uv = torch.randn(heads, latent, 128)
    query = absorb_query(raw_q, uk, nope_dim=2)
    key = torch.randn(4, 1, 576)
    positions = torch.tensor([1, 2] if packed else [0, 3])
    plan = tuple((slot, pos, pos + 1, previous) for slot, pos, previous in
                 zip((2, 5), positions.tolist(), (65536, 16384)))
    partial = torch.randn(2, heads * world, latent)
    lses = torch.randn(2, heads * world)
    selected_q = query.index_select(0, positions)
    gathered = torch.cat([selected_q + rank for rank in range(world)], dim=1)

    def gather(value, dim):
        assert dim == 1
        torch.testing.assert_close(value, selected_q)
        return gathered

    def decode(q, k, v, output):
        torch.testing.assert_close(q, gathered)
        torch.testing.assert_close(k, key.index_select(0, positions))
        torch.testing.assert_close(v, k[..., :latent])
        output.copy_(partial)
        return output, lses

    def combine(out, lse, group, *, is_lse_base_on_e):
        assert is_lse_base_on_e
        torch.testing.assert_close(out, partial)
        torch.testing.assert_close(lse, lses)
        return out[:, :heads] + 1

    def forbidden_local_decode(*args):
        raise AssertionError("rank-local decode loses the other seven context slices")

    monkeypatch.setitem(sys.modules, "vllm.v1.attention.ops.dcp",
                        SimpleNamespace(cp_lse_ag_out_rs=combine))
    pool = SimpleNamespace(
        is_absorbed_mla=True, dcp_world_size=world, query_heads=heads, value_dim=latent,
        active_decode_rows=(2, 5), dcp_group=SimpleNamespace(all_gather=gather),
        decode_dcp=decode, decode=forbidden_local_decode,
        _dcp_local_length=lambda length: (length + 7) // 8,
        dcp_sharded={2: True, 5: True},
        metadata={2: dict(coverage=8000), 5: dict(coverage=2000)},
        dcp_global_lens=torch.zeros(8, dtype=torch.int32),
    )
    output = torch.zeros(4, heads, 128)
    VLLMLayerLODPool._direct_mixed_decode(
        pool, query, key, key[..., :latent], output, plan,
        mla_query=raw_q if lazy_absorption else None,
        mla_w_uk_t=uk if lazy_absorption else None, mla_w_uv=uv,
    )
    expected = torch.einsum("bhd,hdv->bhv", partial[:, :heads] + 1, uv)
    torch.testing.assert_close(output.index_select(0, positions), expected, rtol=1e-4, atol=1e-4)
    unselected = torch.tensor([0, 3] if packed else [1, 2])
    assert output.index_select(0, unselected).count_nonzero() == 0
    assert pool.metadata[2] == dict(coverage=8000, total_len=8193, recent_len=193, dcp_global_total_len=65537)
    assert pool.metadata[5] == dict(coverage=2000, total_len=2049, recent_len=49, dcp_global_total_len=16385)


@pytest.mark.parametrize("begin", (0, 1, 255, 16384))
def test_owned_views_cover_each_global_position_once(begin):
    keys = torch.arange(37).view(1, 1, -1, 1)
    seen = []
    for rank in range(8):
        result = owned_slice(keys, begin=begin, rank=rank, world=8)
        expected = keys[..., [i for i in range(37) if (i + begin) % 8 == rank], :]
        torch.testing.assert_close(result, expected)
        if result.numel():
            assert result.untyped_storage().data_ptr() == keys.untyped_storage().data_ptr()
        seen.extend(result.flatten().tolist())
    assert sorted(seen) == list(range(37))


def test_global_counts_ignore_invalid_owners():
    counts = torch.zeros(1, 1, 4, dtype=torch.int32)
    add_global_counts(counts, torch.tensor([[[-1, 0, 1, 1, 3, 4]]]))
    assert counts.tolist() == [[[1, 2, 0, 1]]]


def fake_pool(rank):
    def new_page(k, v, owners, *, state_capacity, sequence_capacity, virtual_k, virtual_v,
                 metadata_only=False):
        keys = torch.zeros(1, 1, 1 if metadata_only else sequence_capacity, 576)
        if not metadata_only:
            keys[..., :virtual_k.size(2), :].copy_(virtual_k)
        lengths = torch.zeros(1, 1, state_capacity, dtype=torch.int32)
        add_global_counts(lengths, owners)
        return dict(leaf_k=keys, leaf_v=keys[..., :512], slot_lengths=lengths,
                    leaf_count=k.size(2))

    def append(page, k, v, owners, **kwargs):
        count = page["leaf_count"]
        page["leaf_k"][..., count:count + k.size(2), :].copy_(k)
        page["leaf_count"] += k.size(2)
        add_global_counts(page["slot_lengths"], owners)

    return SimpleNamespace(
        dcp_rank=rank, dcp_world_size=8, value_dim=512,
        _dcp_local_length=lambda n: (n + 7 - rank) // 8,
        engine=SimpleNamespace(decode_cache_headroom=256,
                               _new_page_cache=new_page, _append_page_cache=append),
    )


def test_initial_and_cached_archives_preserve_tail_sink_alias_and_global_counts(monkeypatch):
    import lod_attention.kernels.paged_leaf_attention as kernels

    monkeypatch.setattr(kernels, "stable_owner_ranks", lambda owners: torch.zeros_like(owners))
    keys = torch.arange(513).float().view(1, 1, -1, 1).expand(-1, -1, -1, 576)
    first_owners = torch.arange(255).view(1, 1, -1) % 7
    next_owners = torch.arange(256).view(1, 1, -1) % 7
    global_counts = torch.zeros(1, 1, 7, dtype=torch.int32)
    add_global_counts(global_counts, first_owners)
    add_global_counts(global_counts, next_owners)
    summed_local_counts = torch.zeros_like(global_counts)
    for rank in range(8):
        pool = fake_pool(rank)
        page = initial_page(pool, keys[..., 1:257, :], first_owners,
                            sink_len=1, coverage=256, prompt_capacity=1024, state_capacity=7)
        append_page(pool, page, keys[..., 256:513, :], next_owners,
                    previous_coverage=256, coverage=512, total_len=513)
        torch.testing.assert_close(page["global_slot_lengths"], global_counts)
        summed_local_counts += page["slot_lengths"]
        source = dict(total_len=513, sink_k=keys[..., :1, :], page_cache=page)
        local, value, total = owned_history(pool, source)
        torch.testing.assert_close(local, keys[..., rank::8, :])
        assert total == 513
        assert local.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        assert page["leaf_k"].untyped_storage().data_ptr() == page["leaf_v"].untyped_storage().data_ptr()
        assert page["leaf_k"].size(2) == pool._dcp_local_length(1024) + 256
    torch.testing.assert_close(summed_local_counts, global_counts)


@pytest.mark.parametrize("rank", range(8))
def test_b1_sharded_archive_reuses_decode_leaves_without_aliasing_global_directory(monkeypatch, rank):
    import lod_attention.kernels.paged_leaf_attention as kernels

    monkeypatch.setattr(kernels, "stable_owner_ranks", lambda owners: torch.zeros_like(owners))
    pool = fake_pool(rank)
    # Native decode reserves DCP-local headroom, not 256 tokens per rank.
    capacity = pool._dcp_local_length(1024) + 32
    backing = torch.full((1, 1, capacity, 576), float("nan"))
    decode_lengths = torch.full((1, 1, 7), -19, dtype=torch.int32)
    pool.max_requests = 1
    pool.state = dict(page_cache=dict(leaf_k=backing, leaf_v=backing[..., :512],
                                     slot_lengths=decode_lengths))
    keys = torch.arange(513).float().view(1, 1, -1, 1).expand(-1, -1, -1, 576)
    owners = torch.arange(255).view(1, 1, -1) % 7
    page = initial_page(pool, keys[..., 1:257, :], owners, sink_len=1,
                        coverage=256, prompt_capacity=1024, state_capacity=7)
    assert page["dcp_prefill_pool_backed"] and not page["metadata_only"]
    assert page["leaf_k"].data_ptr() == backing.data_ptr()
    assert page["leaf_v"].data_ptr() == backing.data_ptr()
    assert page["slot_lengths"].data_ptr() != decode_lengths.data_ptr()
    append_page(pool, page, keys[..., 256:513, :], torch.arange(256).view(1, 1, -1) % 7,
                previous_coverage=256, coverage=512, total_len=513)
    local, value, total = owned_history(pool, dict(total_len=513,
                                                 sink_k=keys[..., :1, :], page_cache=page))
    torch.testing.assert_close(local, keys[..., rank::8, :])
    torch.testing.assert_close(value, local[..., :512])
    assert total == 513 and torch.all(decode_lengths == -19)
    assert b1_leaf_storage(pool, batch=2) is None
    pool.max_requests = 2
    assert b1_leaf_storage(pool, batch=1) is None


def test_b1_sharded_storage_rejects_duplicated_values_and_insufficient_capacity():
    pool = fake_pool(0)
    pool.max_requests = 1
    backing = torch.zeros(1, 1, 1, 576)
    pool.state = dict(page_cache=dict(leaf_k=backing, leaf_v=backing[..., :512].clone()))
    with pytest.raises(ValueError, match="aliasing"):
        b1_leaf_storage(pool, batch=1)
    pool.state["page_cache"]["leaf_v"] = backing[..., :512]
    with pytest.raises(ValueError, match="headroom"):
        initial_page(pool, backing, torch.zeros(1, 1, 1, dtype=torch.long),
                     sink_len=0, coverage=1, prompt_capacity=1024, state_capacity=7)


@pytest.mark.parametrize("batch,begin,tokens", ((1, 0, 256), (2, 1, 255), (2, 7, 37)))
def test_transient_archive_interleave_matches_global_chronology(batch, begin, tokens):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU archive interleaving")
    from lod_attention.kernels.kimi_sharded_prefill import interleave_shards

    keys = torch.randn(batch, 1, tokens, 576).to(device="cuda", dtype=torch.bfloat16)
    per_rank = (tokens + 7) // 8
    shards = []
    for rank in range(8):
        shard = owned_slice(keys, begin=begin, rank=rank, world=8).contiguous()
        shards.append(torch.nn.functional.pad(shard, (0, 0, 0, per_rank - shard.size(2))))
    output = torch.empty_like(keys)
    interleave_shards(torch.stack(shards), output, begin=begin)
    # Compare on CPU: this userspace's GPU tensor error formatter can abort
    # while printing a mismatch, hiding the offending entries.
    torch.testing.assert_close(output.cpu(), keys.cpu(), atol=0, rtol=0)


def test_metadata_only_directory_stores_no_history_and_matches_regular_leaf_attention():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU page construction")
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine

    torch.manual_seed(734)
    engine = KernelTwoLevelLODAttention(LODConfig(state_clustering_policy="manual", leaf_paged_directory=True),
                                      query_heads=8, key_value_heads=1, scale=192**-0.5)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=1024, has_query_norm=True, has_key_norm=False)
    # The K3-v10 userspace can crash inside GPU BF16 randn before a LoD
    # kernel is launched. Generate test inputs on CPU, then transfer them.
    keys = torch.randn(1, 1, 255, 576).to(device="cuda", dtype=torch.bfloat16)
    owners = (torch.arange(255, device="cuda") % 8).view(1, 1, -1)
    kwargs = dict(state_capacity=8, sequence_capacity=1024, virtual_k=keys, virtual_v=keys[..., :512])
    regular = engine._new_page_cache(keys[..., :127, :], keys[..., :127, :512], owners[..., :127], **kwargs)
    metadata = engine._new_page_cache(keys[..., :127, :], keys[..., :127, :512], owners[..., :127],
                                      **kwargs, metadata_only=True)
    for page in (regular, metadata):
        engine._append_page_cache(page, keys[..., 127:, :], keys[..., 127:, :512], owners[..., 127:])
    assert metadata["leaf_k"].untyped_storage().nbytes() == 576 * 2
    torch.testing.assert_close(metadata["slot_lengths"].cpu(), regular["slot_lengths"].cpu())
    engine._lod_kimi_expanded_prefill_chunk = torch.randn(1, 8, 37, 192).to(device="cuda", dtype=torch.bfloat16)
    engine._lod_kimi_w_uk_t = (torch.randn(8, 128, 512) / 512**0.5).to(device="cuda", dtype=torch.bfloat16)
    engine._lod_kimi_w_uv = (torch.randn(8, 512, 128) / 512**0.5).to(device="cuda", dtype=torch.bfloat16)
    carrier = keys[..., :1, :].expand(1, 8, 37, 576)
    slots = torch.arange(8, device="cuda").view(1, 1, 1, -1).expand(1, 8, 37, -1).contiguous()
    ref, ref_lse = engine._paged_leaf_attention(carrier, slots, regular, active_slots=8)
    transient = dict(metadata, leaf_k=keys, leaf_v=keys[..., :512])
    got, got_lse = engine._paged_leaf_attention(carrier, slots, transient, active_slots=8)
    torch.testing.assert_close(got.cpu(), ref.cpu(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(got_lse.cpu(), ref_lse.cpu(), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU B1 sharded storage integration")
@torch.inference_mode()
def test_actual_b1_sharded_cache_backing_matches_separate_storage_across_updates():
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine

    torch.manual_seed(736)
    engine = KernelTwoLevelLODAttention(
        LODConfig(state_clustering_policy="manual", leaf_paged_directory=True),
        query_heads=8, key_value_heads=1, scale=192**-0.5)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=1024, has_query_norm=True, has_key_norm=False)
    keys = torch.randn(1, 1, 513, 576).to(device="cuda", dtype=torch.bfloat16)
    first = (torch.arange(255, device="cuda") % 8).view(1, 1, -1)
    second = (torch.arange(256, device="cuda") % 8).view(1, 1, -1)
    for rank in range(8):
        pool = SimpleNamespace(dcp_rank=rank, dcp_world_size=8, value_dim=512, engine=engine,
                              _dcp_local_length=lambda n, r=rank: (n + 7 - r) // 8)
        separate = initial_page(pool, keys[..., 1:257, :], first,
                                sink_len=1, coverage=256, prompt_capacity=1024, state_capacity=8)
        backing = torch.empty_like(separate["leaf_k"])
        pool.max_requests = 1
        pool.state = dict(page_cache=dict(leaf_k=backing, leaf_v=backing[..., :512]))
        shared = initial_page(pool, keys[..., 1:257, :], first,
                              sink_len=1, coverage=256, prompt_capacity=1024, state_capacity=8)
        assert shared["leaf_k"].data_ptr() == backing.data_ptr()
        for page in (separate, shared):
            append_page(pool, page, keys[..., 256:513, :], second,
                        previous_coverage=256, coverage=512, total_len=513)
        for name in ("slot_lengths", "global_slot_lengths"):
            torch.testing.assert_close(shared[name].cpu(), separate[name].cpu())
        count = pool._dcp_local_length(513) - int(rank == 0)
        torch.testing.assert_close(shared["leaf_k"][..., :count, :].cpu(),
                                   separate["leaf_k"][..., :count, :].cpu(), atol=0, rtol=0)
        state = dict(total_len=513, sink_k=keys[..., :1, :], page_cache=shared)
        local, value, total = owned_history(pool, state)
        torch.testing.assert_close(local.cpu(), keys[..., rank::8, :].cpu(), atol=0, rtol=0)
        assert total == 513 and local.data_ptr() == value.data_ptr()


def test_sharded_fine_gpu_matches_unsharded_with_empty_owner_experts():
    if not torch.cuda.is_available():
        pytest.skip("requires GPU projected leaf attention")
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine

    torch.manual_seed(731)
    heads, queries, tokens, states = 8, 37, 255, 8
    config = LODConfig(state_clustering_policy="manual", leaf_paged_directory=True)
    engine = KernelTwoLevelLODAttention(config, query_heads=heads, key_value_heads=1, scale=192**-0.5)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
                     request_capacity=1024, has_query_norm=True, has_key_norm=False)
    engine._lod_prefill_attention_buffers = {}
    keys = torch.randn(1, 1, tokens, 576).to(device="cuda", dtype=torch.bfloat16)
    owners = (torch.arange(tokens, device="cuda") % states).view(1, 1, -1)
    q = torch.randn(1, heads, queries, 192).to(device="cuda", dtype=torch.bfloat16)
    uk = (torch.randn(heads, 128, 512) / 512**0.5).to(device="cuda", dtype=torch.bfloat16)
    uv = (torch.randn(heads, 512, 128) / 512**0.5).to(device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(states, device="cuda").view(1, 1, 1, -1).expand(1, heads, queries, -1).contiguous()
    engine._lod_kimi_expanded_prefill_chunk = q
    engine._lod_kimi_w_uk_t = uk
    engine._lod_kimi_w_uv = uv

    def fine(k, own):
        page = engine._new_page_cache(k, k[..., :512], own,
                                      state_capacity=states, sequence_capacity=tokens + 256,
                                      virtual_k=k, virtual_v=k[..., :512])
        lengths = page["slot_lengths"].unsqueeze(2).expand(1, heads, queries, -1)
        chosen = torch.where(torch.gather(lengths, -1, slots.long()).gt(0), slots, -1)
        carrier = keys[..., :1, :].expand(1, heads, queries, 576)
        return engine._paged_leaf_attention(carrier, chosen, page, active_slots=states, reduce_routes=True)

    whole, whole_lse = fine(keys, owners)
    whole, whole_lse = whole.clone(), whole_lse.clone()
    engine._lod_kimi_prefill_head_group_limit = 2
    grouped, grouped_lse = fine(keys, owners)
    torch.testing.assert_close(grouped.cpu(), whole.cpu(), atol=1.5e-2, rtol=2e-2)
    torch.testing.assert_close(grouped_lse.cpu(), whole_lse.cpu(), atol=2e-2, rtol=2e-3)
    parts = []
    for rank in range(8):
        out, lse = fine(keys[..., rank::8, :].contiguous(), owners[..., rank::8].contiguous())
        parts.append((out.clone(), lse.clone()))
    outputs = torch.stack([item[0] for item in parts]).float()
    lses = torch.stack([item[1] for item in parts]).float()
    total_lse = torch.logsumexp(lses, dim=0)
    merged = (outputs * torch.exp(lses - total_lse)[..., None]).sum(dim=0)
    torch.testing.assert_close(total_lse.cpu(), whole_lse.cpu(), atol=2e-2, rtol=2e-3)
    torch.testing.assert_close(merged.cpu(), whole.float().cpu(), atol=1.5e-2, rtol=2e-2)


@pytest.mark.parametrize("batch", (1, 2))
def test_tiled_prefill_lse_weighting_matches_reference_with_empty_shards(batch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU LSE weighting")
    from lod_attention.kernels.kimi_sharded_prefill import weight_prefill_partials_

    torch.manual_seed(732)
    values = torch.randn(8, batch, 8, 37, 128).to(device="cuda", dtype=torch.bfloat16)
    lses = torch.randn(8, batch, 8, 37).to(device="cuda")
    lses[0, ..., 1::2] = -torch.inf
    lses[:, ..., 0] = -torch.inf
    lses[1, ..., 1] = torch.nan
    lses[2, ..., 2] = torch.inf
    clean = torch.where(torch.isfinite(lses), lses, -torch.inf)
    expected_lse = torch.logsumexp(clean, dim=0)
    weights = torch.exp(clean - expected_lse)
    weights = torch.nan_to_num(weights, nan=0.0)
    reference = (values.float() * weights[..., None]).sum(dim=0)
    parts = []
    workspace = torch.empty_like(lses[0])
    for rank in range(8):
        part = values[rank].clone()
        got_lse = weight_prefill_partials_(part, lses, rank=rank, total_lse=workspace)
        assert got_lse.data_ptr() == workspace.data_ptr()
        torch.testing.assert_close(got_lse, expected_lse, atol=1e-5, rtol=1e-5)
        parts.append(part.float())
    with pytest.raises(ValueError, match="workspace"):
        weight_prefill_partials_(values[0].clone(), lses, rank=0, total_lse=workspace[..., :-1])
    torch.testing.assert_close(torch.stack(parts).sum(0), reference, atol=1e-2, rtol=1e-2)
