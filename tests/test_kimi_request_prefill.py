"""Request/head ownership and preservation of global prefill boundaries."""

from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest
import torch

from vllm_lod_plugin.models.kimi_k3_request_prefill import (
    advance_block, attend_slice, exchange_queries, exchange_outputs, retain_owner_record,
    transport_workspace, owner_prefill_storage,
)


def test_retained_owner_slice_does_not_pin_the_whole_batched_record():
    records = torch.randn(8 * 2048, 1, 576)
    view = records[3 * 2048:4 * 2048]
    assert view.is_contiguous()
    assert view.untyped_storage().nbytes() == records.untyped_storage().nbytes()
    owned = retain_owner_record(view)
    torch.testing.assert_close(owned, view.permute(1, 0, 2).unsqueeze(0), atol=0, rtol=0)
    assert owned.untyped_storage().nbytes() == owned.numel() * owned.element_size()
    assert owned.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()


def test_owner_prefill_storage_is_opt_in_and_requires_fixed_owner_pool(monkeypatch):
    monkeypatch.delenv("LOD_KIMI_OWNER_POOL_BACKED_PREFILL", raising=False)
    assert owner_prefill_storage(SimpleNamespace()) is None
    monkeypatch.setenv("LOD_KIMI_OWNER_POOL_BACKED_PREFILL", "1")
    with pytest.raises(RuntimeError, match="captured owner cache"):
        owner_prefill_storage(SimpleNamespace())
    storage = {"page_cache": {"leaf_k": torch.empty(1)}}
    calls = []
    child = SimpleNamespace(_initial_prefill_storage=lambda rows: (calls.append(rows), storage)[1])
    assert owner_prefill_storage(SimpleNamespace(owner_decode_pool=child)) is storage
    assert calls == [(0,)]


@pytest.mark.parametrize("fail", [False, True])
def test_owner_state_workspace_is_shared_without_retaining_layer_references(fail):
    from vllm_lod_plugin.models.kimi_k3_request_prefill import owner_state_workspace

    registry = {}
    engines = [SimpleNamespace(_lod_prefill_attention_buffers=registry) for _ in range(2)]
    scratch = {"scores": torch.ones(4)}
    with owner_state_workspace(engines[0]):
        engines[0]._lod_state_maxsim_buffers = scratch
    assert not hasattr(engines[0], "_lod_state_maxsim_buffers")
    with pytest.raises(RuntimeError) if fail else nullcontext():
        with owner_state_workspace(engines[1]):
            assert engines[1]._lod_state_maxsim_buffers is scratch
            if fail:
                raise RuntimeError("update failure")
    assert not hasattr(engines[1], "_lod_state_maxsim_buffers")
    assert registry["kimi_owner_state_update"]["_lod_state_maxsim_buffers"] is scratch


@pytest.mark.parametrize("head_group", [None, "6"])
def test_owner_prefill_forwards_projection_group_without_changing_slices(monkeypatch, head_group):
    from vllm_lod_plugin.models import kimi_k3_request_prefill as module

    monkeypatch.delenv("LOD_KIMI_OWNER_PREFILL_HEAD_GROUP", raising=False)
    if head_group is not None:
        monkeypatch.setenv("LOD_KIMI_OWNER_PREFILL_HEAD_GROUP", head_group)
    query = torch.empty(2, 96, 192)
    record = torch.empty(2, 1, 576)
    calls = []
    monkeypatch.setattr(module, "exchange_queries", lambda *args: {0: query})
    monkeypatch.setattr(module, "exchange_outputs", lambda group, outputs, *args: outputs[0])
    def attend(pool, slot, q, k, uk, uv, **kwargs):
        assert q is query
        torch.testing.assert_close(k, record, equal_nan=True)
        calls.append((slot, kwargs))
        return q.new_zeros(2, 96, 128)
    monkeypatch.setattr(module, "attend_slice", attend)
    pool = SimpleNamespace(direct_prefill_plan=((0, 0, 2, 2048),),
        direct_prefill_prompt_lengths={0: 16384}, dcp_world_size=8, dcp_group=None,
        ready=[False] * 8, dcp_sharded=[True] * 8,
        metadata=[{} for _ in range(8)], direct_prefill_calls=0)
    module.request_owner_prefill(SimpleNamespace(_lod_owner_uk=object(), _lod_owner_uv=object()),
                                 pool, query, record)
    assert calls == [(0, dict(previous=2048, prompt=16384,
                            head_group_limit=12 if head_group is None else 6))]
    assert pool.metadata[0]["total_len"] == 2050


def test_owner_pressure_check_occurs_once_per_prefill_batch_and_never_decode(monkeypatch):
    import numpy as np
    pytest.importorskip("vllm", reason="real vLLM runtime lifecycle check")
    import vllm_lod_plugin.runtime as module

    calls = []
    monkeypatch.setenv("LOD_KIMI_OWNER_PRESSURE_CHECK", "1")
    monkeypatch.setattr(module, "_reclaim_prefill_allocator", lambda device: calls.append(device))
    pools = {str(i): SimpleNamespace(kimi_request_owner_prefill=True,
        metadata=[{} for _ in range(8)]) for i in range(24)}
    runtime = SimpleNamespace(pools=pools, pool_size=8, logical_lengths={},
        _lod_row=lambda slot: slot, model_state=SimpleNamespace(device="cpu"))
    computed, prompts = np.zeros(8,dtype=np.int64), np.full(8,65536,dtype=np.int64)
    prepare = module.VLLMLODRuntime._prepare_direct_prefill
    assert prepare(runtime,list(range(8)),computed,np.arange(9)*2048,prompts)
    assert calls == ["cpu"]
    assert all(len(pool.direct_prefill_plan) == 8 for pool in pools.values())
    assert prepare(runtime,list(range(8)),computed,np.arange(9),prompts)
    assert calls == ["cpu"]
    monkeypatch.delenv("LOD_KIMI_OWNER_PRESSURE_CHECK")
    assert prepare(runtime,list(range(8)),computed,np.arange(9)*2048,prompts)
    assert calls == ["cpu"]


@pytest.mark.parametrize("fail", [False, True])
def test_owner_cache_builder_scopes_direct_storage_and_retains_temporary_tail(monkeypatch, fail):
    from contextlib import nullcontext
    from vllm_lod_plugin.models import kimi_k3_request_prefill as module

    monkeypatch.setenv("LOD_KIMI_OWNER_POOL_BACKED_PREFILL", "1")
    monkeypatch.setattr(module, "projection_scope", lambda *args, **kwargs: nullcontext())
    storage = {"page_cache": {"leaf_k": torch.empty(1)}}
    cache = SimpleNamespace(state={"pool_backed": True})
    engine = SimpleNamespace(prefill_chunk_len=4, reset_runtime_cache=lambda: None,
        _prefill_local_attention=lambda q, *args, **kwargs: (q[..., :128], None))
    def build(k, v, **kwargs):
        assert engine._lod_prefill_storage is storage
        assert kwargs == {"finalize_cache_for_decode": False}
        assert engine._lod_prefill_cache_capacity == 8
        if fail:
            raise RuntimeError("build failure")
        return cache
    engine.build_cache_from_bf16 = build
    pool = SimpleNamespace(engine=engine, _kimi_request_owner_rows={},
        owner_decode_pool=SimpleNamespace(_initial_prefill_storage=lambda rows: storage))
    args = (pool, 0, torch.empty(4, 1, 192), torch.empty(4, 1, 576), None, None)
    if fail:
        with pytest.raises(RuntimeError, match="build failure"):
            attend_slice(*args, previous=0, prompt=8)
    else:
        attend_slice(*args, previous=0, prompt=8)
        assert cache.state["pool_backed"] is False
        assert cache.state["owner_remote_pool_backed"] is True
        assert pool._kimi_request_owner_rows[0]["parts"] == []
    assert not hasattr(engine, "_lod_prefill_storage")
    assert not hasattr(engine, "_lod_prefill_cache_capacity")


def test_owner_transport_arenas_reuse_storage_across_layers_and_smaller_chunks():
    group = SimpleNamespace()
    query = torch.empty(24, 12, 192)
    original = transport_workspace(group, query, "wire", 900)
    original.fill_(17)
    for _ in range(24):
        same = transport_workspace(group, query, "wire", 900)
        short = transport_workspace(group, query, "wire", 400)
        assert same.data_ptr() == original.data_ptr() == short.data_ptr()
        assert torch.all(short == 17)
    bigger = transport_workspace(group, query, "wire", 1200)
    assert bigger.numel() == 1200 and bigger.data_ptr() != original.data_ptr()
    assert transport_workspace(group, query, "wire", 900).data_ptr() == bigger.data_ptr()
    payload = transport_workspace(group, query, "payload", 900)
    assert payload.data_ptr() != bigger.data_ptr()
    changed_dtype = transport_workspace(group, query.double(), "wire", 400)
    assert changed_dtype.dtype == torch.float64
    assert group._lod_owner_transport_allocations == 4


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("reuse", [False, True])
def test_repeated_exchange_keeps_head_order_and_optional_reuse(monkeypatch, rank, reuse):
    monkeypatch.setenv("LOD_KIMI_OWNER_REUSE_TRANSPORT", "1" if reuse else "0")
    # Small CPU stand-in avoids thread-pool overhead in 24 repeated exchanges;
    # the separate layout tests and TP8 fixture cover the real 12/96 heads.
    heads = 2
    local = torch.arange(24 * heads * 192).view(24, heads, 192).float()
    output = torch.arange(3 * 8 * heads * 128).view(3, 8 * heads, 128).float()
    stage = "query"

    def recv(target, peer):
        if stage == "query":
            target.copy_(local[rank * 3:(rank + 1) * 3] + peer * 1e6)
        else:
            target.copy_(output[:, rank * heads:(rank + 1) * heads] + peer * 1e6)

    group = SimpleNamespace(rank_in_group=rank, world_size=8,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(
            disabled=False, group_start=lambda: None, group_end=lambda: None,
            send=lambda tensor, peer: None, recv=recv)))
    plan = tuple((slot, slot * 3, (slot + 1) * 3, 0) for slot in range(8))
    pointers = None
    for _ in range(24):
        stage = "query"
        queries = exchange_queries(group, local + rank * 1e6, plan)
        expected = torch.cat([local[rank * 3:(rank + 1) * 3] + peer * 1e6
                              for peer in range(8)], dim=1)
        torch.testing.assert_close(queries[rank], expected, atol=0, rtol=0)
        stage = "output"
        result = exchange_outputs(group, {rank: output + rank * 1e6}, plan, local)
        for peer in range(8):
            torch.testing.assert_close(result[peer * 3:(peer + 1) * 3],
                output[:, rank * heads:(rank + 1) * heads] + peer * 1e6, atol=0, rtol=0)
        if reuse:
            assert result.data_ptr() == queries[rank].data_ptr()
            new_pointers = {name: tensor.data_ptr() for name, tensor in group._lod_owner_transport.items()}
            if pointers is not None:
                assert new_pointers == pointers
            pointers = new_pointers
        else:
            assert result.data_ptr() != queries[rank].data_ptr()
            assert not hasattr(group, "_lod_owner_transport")
    assert getattr(group, "_lod_owner_transport_allocations", 0) == (2 if reuse else 0)


@pytest.mark.parametrize("rank", range(8))
def test_query_exchange_preserves_tp_head_order(rank):
    local = torch.arange(8 * 3 * 12 * 192).view(24, 12, 192).float()
    sends = []

    def recv(output, peer):
        output.copy_(local[rank * 3:(rank + 1) * 3] + peer * 1e6)

    group = SimpleNamespace(rank_in_group=rank, world_size=8,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(
            disabled=False, group_start=lambda: None, group_end=lambda: None,
            send=lambda tensor, peer: sends.append(peer), recv=recv)))
    plan = tuple((slot, slot * 3, (slot + 1) * 3, 0) for slot in range(8))
    received = exchange_queries(group, local + rank * 1e6, plan)
    assert set(received) == {rank}
    expected = torch.cat([
        local[rank * 3:(rank + 1) * 3] + peer * 1e6 for peer in range(8)
    ], dim=1)
    torch.testing.assert_close(received[rank], expected)
    assert sorted(sends) == [peer for peer in range(8) if peer != rank]


@pytest.mark.parametrize("rank", range(8))
def test_output_exchange_returns_each_tp_slice(rank):
    queries = torch.empty(24, 12, 192)
    output = torch.arange(3 * 96 * 128).view(3, 96, 128).float()
    received = []
    sends = []

    def recv(target, owner):
        target.copy_(output[:, rank * 12:(rank + 1) * 12] + owner * 1e6)
        received.append(owner)

    group = SimpleNamespace(rank_in_group=rank, world_size=8,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(
            group_start=lambda: None, group_end=lambda: None, recv=recv,
            send=lambda tensor, peer: sends.append((peer, tensor.clone())))))
    plan = tuple((slot, slot * 3, (slot + 1) * 3, 0) for slot in range(8))
    result = exchange_outputs(group, {rank: output + rank * 1e6}, plan, queries)
    for owner in range(8):
        torch.testing.assert_close(result[owner * 3:(owner + 1) * 3],
            output[:, rank * 12:(rank + 1) * 12] + owner * 1e6)
    for peer, source in sends:
        torch.testing.assert_close(source, output[:, peer * 12:(peer + 1) * 12] + rank * 1e6)


def test_only_complete_logical_block_updates_centroids():
    updates = []
    state = {
        "recent_k": torch.ones(1, 1, 256, 576), "recent_len": 256,
        "coverage": 16128, "state_k": torch.ones(1, 1, 256, 576),
        "state_v": torch.ones(1, 1, 256, 512), "counts": torch.ones(1, 1, 256, 1),
        "state_len": 256, "scheduled_state_len": 256, "state_capacity": 256,
        "page_cache": {},
    }

    def update(sk, sv, counts, norms, key, value, **kwargs):
        updates.append((key.size(2), kwargs))
        return sk, sv, counts, 256, torch.zeros(1, 1, key.size(2), dtype=torch.long), None

    engine = SimpleNamespace(
        prefill_local_len=16640, prefill_chunk_len=16384,
        prefill_state_update_len=16384, local_len=512,
        _update_state=update, _append_page_cache=lambda *args: None,
    )
    advance_block(engine, SimpleNamespace(state=state), torch.ones(1, 1, 16384, 576), total=32768)
    assert len(updates) == 1 and updates[0][0] == 16384
    assert updates[0][1]["ctx_len"] == 32768
    assert state["coverage"] == 32512 and state["recent_len"] == 256
    assert state["recent_k"].untyped_storage().nbytes() == state["recent_k"].numel() * state["recent_k"].element_size()


def test_scheduler_slice_cannot_cross_logical_block():
    pool = SimpleNamespace(engine=SimpleNamespace(prefill_chunk_len=16384))
    with pytest.raises(ValueError, match="cross a 16K"):
        attend_slice(pool, 0, torch.empty(2048, 12, 192), torch.empty(2048, 1, 576),
            None, None, previous=16000, prompt=32768)


def test_owner_decode_uses_single_gpu_pool_and_four_updates(monkeypatch):
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_decode

    calls = []
    class DecodePool:
        def __init__(self, layer, **kwargs):
            assert layer.num_heads == 96
            assert kwargs["max_requests"] == 1
            assert kwargs["request_owner_prefill"] is False
            assert "dcp_group" not in kwargs
            self.catch_up_batches = 0
            self.unified_page1_fixed_dirty = [False]
        def install(self, slot, cache):
            assert slot == 0 and cache.state["recent_k"].size(2) == 256
            assert cache.state["recent_v"].data_ptr() == cache.state["recent_k"].data_ptr()
            calls.append("install")
        def catch_up_many(self, rows):
            previous = rows[0][1]
            if previous > 32768 and previous % 256 == 0:
                self.catch_up_batches += 1
        def decode_dcp(self, q, k, v, out):
            assert self.active_decode_rows == (0,)
            assert q.shape == (1, 96, 576)
            assert v.data_ptr() == k.data_ptr()
            # No numeric check here; avoid allocating/filling large CPU
            # tensors 1025 times just to test host-side cadence.

    monkeypatch.setitem(sys.modules, "vllm_lod_plugin.pool", SimpleNamespace(VLLMLayerLODPool=DecodePool))
    monkeypatch.setitem(sys.modules, "vllm_lod_plugin.models.kimi_k3", SimpleNamespace(
        absorb_query=lambda q, uk, **kwargs: q.new_empty(1, 96, 576)))
    recent = torch.empty(1, 1, 768, 576)
    row = dict(total=32768, parts=[], cache=SimpleNamespace(state={
        "recent_len": 256, "recent_k": recent, "recent_v": recent[..., :512]}))
    pool = SimpleNamespace(_kimi_request_owner_rows={3: row}, settings=object(),
        request_capacity=65536, engine=SimpleNamespace(scaling=.072, reset_runtime_cache=lambda: None))
    # Small strides avoid 1025 large CPU GEMMs; the interface and cadence are
    # what this test covers. GPU fixture tests exercise real projections.
    query = torch.zeros(1, 96, 192)
    record = torch.zeros(1, 1, 576)
    uk, uv = torch.zeros(96, 128, 512), torch.zeros(96, 512, 128)
    monkeypatch.setattr(torch, "bmm", lambda x, weight: x.new_empty(96, 1, 128))
    for step in range(1025):
        result = attend_decode(pool, 3, query, record, uk, uv, previous=32768 + step)
        assert result.shape == (1, 96, 128)
    assert calls == ["install"]
    assert row["cache"] is None and row["total"] == 32768 + 1025
    assert pool._kimi_owner_decode_updates == 4
    assert pool._kimi_owner_decode_tokens == 1025
    with pytest.raises(RuntimeError, match="matching prefill"):
        attend_decode(pool, 3, query, record, uk, uv, previous=32768)
    with pytest.raises(NotImplementedError, match="one token"):
        attend_decode(pool, 3, query.expand(2, -1, -1), record, uk, uv, previous=row["total"])


def test_decode_counter_audit_rejects_missing_updates_or_rows():
    from benchmarks.kimi_k3_prefill_sweep import validate_owner_decode_counts

    before = [{"layer": {"updates": 2, "tokens": 500}} for _ in range(8)]
    after = [{"layer": {"updates": 6, "tokens": 1525}} for _ in range(8)]
    validate_owner_decode_counts(before, after, steps=1025, world_size=8)
    with pytest.raises(RuntimeError, match="all workers"):
        validate_owner_decode_counts(before, after[:7], steps=1025, world_size=8)
    after[-1]["layer"]["updates"] -= 1
    with pytest.raises(RuntimeError, match="cadence"):
        validate_owner_decode_counts(before, after, steps=1025, world_size=8)


def test_smaller_owner_cohorts_rotate_over_all_ranks():
    from vllm_lod_plugin.models.kimi_k3_request_prefill import take_owner_row

    free_rows, cursor = list(range(7, -1, -1)), 0
    first = []
    for _ in range(4):
        row, cursor = take_owner_row(free_rows, cursor, 8)
        first.append(row)
    assert first == [0, 1, 2, 3]
    free_rows.extend(first)
    free_rows.sort(reverse=True)
    second = []
    for _ in range(4):
        row, cursor = take_owner_row(free_rows, cursor, 8)
        second.append(row)
    assert second == [4, 5, 6, 7]


def test_owner_audit_requires_a_full_block_on_every_rank():
    from benchmarks.kimi_k3_prefill_sweep import validate_owner_prefill_audits

    audits = [{"rank": rank, "owned_query_sizes": {16384: 2}} for rank in range(8)]
    validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8)
    audits[-1]["owned_query_sizes"] = {}
    with pytest.raises(RuntimeError, match="not every attention owner"):
        validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8)
    with pytest.raises(RuntimeError, match="not every attention owner"):
        validate_owner_prefill_audits(audits[:4], row_chunk=16384, world_size=8)


def test_b1_owner_audit_requires_one_owner_and_all_eight_tp_ranks():
    from benchmarks.kimi_k3_prefill_sweep import validate_owner_prefill_audits

    audits = [{"rank": rank, "owned_query_sizes": {16384: 8} if rank == 0 else {}}
              for rank in range(8)]
    assert validate_owner_prefill_audits(
        audits, row_chunk=16384, world_size=8, batch_size=1) == {0}
    with pytest.raises(RuntimeError, match="not every attention owner"):
        validate_owner_prefill_audits(audits[:1], row_chunk=16384, world_size=8, batch_size=1)
    audits[1]["owned_query_sizes"] = {16384: 1}
    with pytest.raises(RuntimeError, match="not every attention owner"):
        validate_owner_prefill_audits(audits, row_chunk=16384, world_size=8, batch_size=1)


def test_b1_kernel_audit_requires_binary_only_on_the_active_owner():
    from benchmarks.kimi_k3_prefill_sweep import audit_loaded_attention
    from types import SimpleNamespace

    flags = {"CK_TILE_FMHA_ROUTE_QUERY_NORMALIZE": "0", "CK_TILE_FMHA_ROUTE_TOPK": "8",
             "CK_TILE_FMHA_ROUTE_GLOBAL_TOPK": "0", "CK_TILE_FMHA_ROUTE_TILE_MAX_ONLY": "1"}
    module = {"module": "lod_kimi_asyncbias_v13", "route_build_flags": flags}
    audits = [{"rank": rank, "loaded_kimi_lod_modules": [module] if rank == 0 else []}
              for rank in range(8)]
    llm = SimpleNamespace(collective_rpc=lambda _: audits)
    assert audit_loaded_attention(llm, mode="two-tier", length=32768,
                                 required_ranks={0}) == audits
    # Ordinary DCP still requires routing on all workers.
    with pytest.raises(RuntimeError, match="worker 1"):
        audit_loaded_attention(llm, mode="two-tier", length=32768)
    # An actual owner cannot bypass missing or wrong binaries.
    audits[0]["loaded_kimi_lod_modules"] = []
    with pytest.raises(RuntimeError, match="worker 0"):
        audit_loaded_attention(llm, mode="two-tier", length=32768, required_ranks={0})
    audits[0]["loaded_kimi_lod_modules"] = [module]
    flags["CK_TILE_FMHA_ROUTE_TOPK"] = "4"
    with pytest.raises(RuntimeError, match="worker 0"):
        audit_loaded_attention(llm, mode="two-tier", length=32768, required_ranks={0})
    with pytest.raises(RuntimeError, match="missing active"):
        audit_loaded_attention(SimpleNamespace(collective_rpc=lambda _: audits[1:]),
                               mode="two-tier", length=32768, required_ranks={0})


def test_b1_owner_keeps_the_same_16k_per_row_budget(monkeypatch):
    from benchmarks._vllm import llm_kwargs

    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_PREFILL", "1")
    monkeypatch.setenv("LOD_KIMI_OWNER_QUERY_CHUNK", "16384")
    monkeypatch.setenv("LOD_BENCHMARK_PREFILL_COHORT", "1")
    monkeypatch.delenv("LOD_KIMI_REQUEST_OWNER_DECODE", raising=False)
    kwargs = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-mla-stack", mode="two-tier",
        max_model_len=131081, batch_size=1, tensor_parallel_size=8,
        decode_context_parallel_size=8, gpu_memory_utilization=.8,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    assert kwargs["long_prefill_token_threshold"] == 16384
    assert kwargs["max_num_batched_tokens"] == 16385
    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_DECODE", "1")
    with pytest.raises(ValueError, match="captured request-owner decode still requires B8"):
        llm_kwargs(checkpoint="tests/fixtures/kimi-k3-mla-stack", mode="two-tier",
            max_model_len=131081, batch_size=1, tensor_parallel_size=8,
            decode_context_parallel_size=8, gpu_memory_utilization=.8,
            full_attention_backend="ROCM_AITER_UNIFIED_ATTN")


@pytest.mark.parametrize("rank", range(8))
def test_single_owner_query_and_output_exchange(monkeypatch, rank):
    """One real row: no dummy rows, all TP head shards reach rank zero."""
    monkeypatch.setenv("LOD_KIMI_OWNER_REUSE_TRANSPORT", "1")
    local = torch.arange(3 * 12 * 192).view(3, 12, 192).float()
    head_output = torch.arange(3 * 96 * 128).view(3, 96, 128).float()
    stage, sends = "query", []

    def recv(target, peer):
        target.copy_(local + peer * 1e6 if stage == "query" else
                     head_output[:, rank * 12:(rank + 1) * 12])

    group = SimpleNamespace(rank_in_group=rank, world_size=8,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(
            disabled=False, group_start=lambda: None, group_end=lambda: None,
            send=lambda tensor, peer: sends.append(peer), recv=recv)))
    plan = ((0, 0, 3, 16384),)
    queries = exchange_queries(group, local + rank * 1e6, plan)
    assert set(queries) == ({0} if rank == 0 else set())
    if rank == 0:
        torch.testing.assert_close(queries[0], torch.cat([local + peer * 1e6 for peer in range(8)], dim=1))
    else:
        assert sends == [0]
    stage = "output"
    outputs = {0: head_output} if rank == 0 else {}
    result = exchange_outputs(group, outputs, plan, local)
    torch.testing.assert_close(result, head_output[:, rank * 12:(rank + 1) * 12])


@pytest.mark.parametrize("row_chunk,cohort,budget", [
    (2048, 1, 16392), (8192, 8, 65544), (16384, 2, 32776), (16384, 8, 131080),
])
def test_owner_query_chunk_and_model_budget_are_explicit(monkeypatch, row_chunk, cohort, budget):
    from benchmarks._vllm import llm_kwargs

    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_PREFILL", "1")
    monkeypatch.setenv("LOD_KIMI_OWNER_QUERY_CHUNK", str(row_chunk))
    monkeypatch.setenv("LOD_BENCHMARK_PREFILL_COHORT", str(cohort))
    kwargs = llm_kwargs(checkpoint="tests/fixtures/kimi-k3-mla-stack", mode="two-tier",
        max_model_len=65545, batch_size=8, tensor_parallel_size=8,
        decode_context_parallel_size=8, gpu_memory_utilization=.8,
        full_attention_backend="ROCM_AITER_UNIFIED_ATTN")
    assert kwargs["long_prefill_token_threshold"] == row_chunk
    assert kwargs["max_num_batched_tokens"] == budget


def test_owner_moe_chunking_keeps_tokenwise_native_result(monkeypatch):
    from vllm_lod_plugin.models.kimi_k3_owner_moe import install_owner_moe_chunking

    calls = []
    class NativeMoE:
        def forward(self, x):
            calls.append(x.size(0))
            return x.sin() * 2
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear",
                        SimpleNamespace(KimiMoE=NativeMoE))
    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_PREFILL", "1")
    monkeypatch.setenv("LOD_KIMI_OWNER_MOE_CHUNK", "5")
    install_owner_moe_chunking()
    x = torch.arange(39).reshape(13, 3).float()
    torch.testing.assert_close(NativeMoE().forward(x), x.sin() * 2, atol=0, rtol=0)
    assert calls == [5, 5, 3]
    install_owner_moe_chunking()  # Idempotence must not stack wrappers.
    calls.clear()
    NativeMoE().forward(x[:3])
    assert calls == [3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="K3 AITER GPU equivalence")
def test_owner_decode_preserves_uniform_attention_mass_across_update(monkeypatch):
    """Zero Q makes LoD exactly equal to the full latent mean, not approximate.

    Random values detect lost/doubled leaves, omitted sink/tail, bad cache
    installation and projection. Exercise both sides of a real 256 update.
    """
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.models.kimi_k3_request_prefill import attend_decode
    from lod_attention.kernels import paged_decode

    original_view = paged_decode.coarse_route_mean_view
    checked = []
    def check_updated_means(*args):
        means = original_view(*args)
        assert means is not None
        if step in (0, 256):
            decode = row["decode_pool"]
            active = int(decode.state_lens[0])
            expected = (decode.state["state_k"][..., :active, :].float()
                / decode.state["counts"][..., :active, :].clamp_min(1)).bfloat16()
            torch.testing.assert_close(means[..., :active, :],
                expected.expand_as(means[..., :active, :]), atol=0, rtol=0)
            checked.append(step)
        return means
    monkeypatch.setattr(paged_decode, "coarse_route_mean_view", check_updated_means)

    torch.manual_seed(29)
    keys = (torch.randn(4096 + 257, 1, 576, device="cuda") * .2).bfloat16()
    uk = (torch.randn(96, 128, 512, device="cuda") * .03).bfloat16()
    uv = (torch.randn(96, 512, 128, device="cuda") * .03).bfloat16()
    engine = KernelTwoLevelLODAttention(LODConfig(state_clustering_normalization="cosine"),
        query_heads=96, key_value_heads=1, scale=.072)
    engine.head_dim = 576
    configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
        request_capacity=8192, has_query_norm=True, has_key_norm=False)
    k = keys[:4096].permute(1, 0, 2).unsqueeze(0)
    cache = engine.build_cache_from_bf16(k, k[..., :512], finalize_cache_for_decode=False)
    source = cache.state
    row = dict(total=4096, parts=[], cache=cache)
    pool = SimpleNamespace(engine=engine, settings=VLLMLODSettings(),
        request_capacity=8192, _kimi_request_owner_rows={0: row})
    query = keys.new_zeros(1, 96, 192)
    for step in range(257):
        output = attend_decode(pool, 0, query, keys[4096 + step:4097 + step], uk, uv,
                               previous=4096 + step)
        if step in (0, 255, 256):
            mean = keys[:4097 + step, 0, :512].float().mean(0).bfloat16()
            expected = torch.bmm(mean.expand(96, 1, 512), uv).transpose(0, 1)
            if not torch.allclose(output, expected, atol=1e-4, rtol=.05):
                decode = row["decode_pool"]
                page = decode.state["page_cache"]
                print("OWNER_DECODE_MASS_DIAGNOSTIC", {
                    "step": step, "output_abs_max": output.abs().max().item(),
                    "expected_abs_max": expected.abs().max().item(),
                    "metadata": decode.metadata[0],
                    "count_sum": decode.state["counts"].sum().item(),
                    "nonzero_counts": decode.state["counts"].gt(0).sum().item(),
                    "nonfinite_state": (~torch.isfinite(decode.state["state_k"][..., :int(source["state_len"]), :])).sum().item(),
                    "nonfinite_coarse": (~torch.isfinite(page["unified_page1_k"][int(page["unified_page1_coarse_offset"]):int(page["unified_page1_coarse_offset"]) + int(source["state_len"])] )).sum().item(),
                    "nonfinite_recent": (~torch.isfinite(decode.state["recent_k"][..., :257, :])).sum().item(),
                    "fixed_length": page["unified_page1_fixed_lengths"].cpu().tolist(),
                    "recent_len": decode.local_lens.cpu().tolist(),
                    "partial_lse_head0": decode.dcp_decode_buffer_storage["kimi_gluon_partial_lse"].reshape(1, 96, -1)[0, 0].cpu().tolist(),
                    "buffers": {name: tensor.cpu().tolist() for name, tensor in
                        decode.dcp_decode_buffer_storage.items() if name in (
                            "gqa_union_hip_context_lens", "gqa_union_token_counts",
                            "kimi_gluon_final_lse")},
                })
            torch.testing.assert_close(output, expected, atol=1e-4, rtol=.05)
        if step == 0:
            decode = row["decode_pool"]
            length = int(source["state_len"])
            for name in ("state_k", "counts"):
                torch.testing.assert_close(decode.state[name][..., :length, :],
                    source[name][..., :length, :], atol=0, rtol=0)
            torch.testing.assert_close(decode.state["sink_k"], source["sink_k"], atol=0, rtol=0)
    assert pool._kimi_owner_decode_updates == 1
    assert pool._kimi_owner_decode_tokens == 257
    assert checked == [0, 256]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="K3 AITER GPU equivalence")
def test_2k_owner_slices_match_16k_prefill_outputs_and_cache():
    from lod_attention._config import LODConfig, LODMode, ModelFamily
    from lod_attention._engines import KernelTwoLevelLODAttention
    from lod_attention._profile import configure_engine
    from vllm_lod_plugin.models.kimi_k3_sharded_prefill import projection_scope

    torch.manual_seed(28)
    def tensor(shape, scale=1):
        return (torch.randn(shape) * scale).to(device="cuda", dtype=torch.bfloat16)
    key = tensor((32768, 1, 576), .2)
    query = tensor((32768, 12, 192), .1)
    uk, uv = tensor((12, 128, 512), .03), tensor((12, 512, 128), .03)
    def make_engine():
        engine = KernelTwoLevelLODAttention(LODConfig(state_clustering_normalization="cosine"),
            query_heads=12, key_value_heads=1, scale=.072)
        engine.head_dim = 576
        configure_engine(engine, family=ModelFamily.KIMI_K3, mode=LODMode.TWO_TIER,
            request_capacity=32768, has_query_norm=True, has_key_norm=False)
        engine._lod_kimi_reduce_prefill_routes = True
        engine._lod_kimi_prefill_head_group_limit = 12
        return engine
    reference = make_engine()
    cache = None
    outputs = []
    for begin in (0, 16384):
        k = key[begin:begin + 16384].permute(1, 0, 2).unsqueeze(0)
        q = query[begin:begin + 16384].permute(1, 0, 2).unsqueeze(0)
        reference._lod_prefill_cache_capacity = 32768
        with projection_scope(reference, q, uk, uv):
            reference._lod_kimi_expanded_prefill_query = q
            output, cache = reference(k.expand(1, 12, -1, -1), k, k[..., :512],
                cache=cache, use_cache=True, finalize_cache_for_decode=False)
            del reference._lod_kimi_expanded_prefill_query
        outputs.append(output.squeeze(0).permute(1, 0, 2))
    owner = SimpleNamespace(engine=make_engine(), _kimi_request_owner_rows={})
    actual = torch.cat([
        attend_slice(owner, 0, query[begin:begin + 2048], key[begin:begin + 2048], uk, uv,
            previous=begin, prompt=32768)
        for begin in range(0, 32768, 2048)
    ])
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, torch.cat(outputs), atol=.0005, rtol=.05)
    state = owner._kimi_request_owner_rows[0]["cache"].state
    for name in ("state_len", "coverage", "scheduled_state_len", "total_len"):
        assert state[name] == cache.state[name]
    for name in ("state_k", "state_v", "counts"):
        torch.testing.assert_close(state[name], cache.state[name], atol=0, rtol=0)
    torch.testing.assert_close(state["page_cache"]["slot_lengths"], cache.state["page_cache"]["slot_lengths"])
