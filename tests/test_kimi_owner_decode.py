from types import SimpleNamespace

import pytest
import torch

from vllm_lod_plugin.models.kimi_k3_owner_decode import (
    initialize_owner_decode, owner_decode_attention, owner_decode_query, owner_row_index, prepare_owner_decode,
)


def test_require_one_live_request_per_gpu():
    for rows in [tuple(range(7)), (0,)*8, tuple(range(1,9))]:
        with pytest.raises(ValueError, match="eight distinct"):
            owner_row_index(rows, 0)
    assert owner_row_index((3,4,5,6,7,0,1,2), 0) == 5


def test_owner_children_inherit_safe_cross_layer_scratch_registry(monkeypatch):
    from vllm_lod_plugin import pool as pool_module
    from vllm_lod_plugin.models import kimi_k3_sharded_prefill

    calls = []
    def child(layer, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(_dcp_buffers=lambda *args: None)
    monkeypatch.setattr(pool_module, "VLLMLayerLODPool", child)
    monkeypatch.setattr(kimi_k3_sharded_prefill, "gather_prefill",
                        lambda group, tensor, dim: tensor)
    registry = {}
    for _ in range(2):
        parent = SimpleNamespace(engine=SimpleNamespace(scaling=.1), settings=object(),
            request_capacity=32768, device=torch.device("cpu"), dtype=torch.bfloat16,
            dcp_rank=0, dcp_group=object(), shared_decode_scratch=registry,
            layer=SimpleNamespace(W_UK_T=torch.empty(1), W_UV=torch.empty(1)))
        initialize_owner_decode(parent)
    assert all(call["shared_decode_scratch"] is registry for call in calls)
    assert all(call["max_requests"] == 1 for call in calls)


def test_keep_the_actual_eight_owner_capture_descriptor(monkeypatch):
    from vllm_lod_plugin.config import _ensure_exact_lod_decode_capture_sizes
    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_DECODE","1")
    compilation = SimpleNamespace(cudagraph_capture_sizes=[8], max_cudagraph_capture_size=8)
    config = SimpleNamespace(attention_config=SimpleNamespace(
        backend=SimpleNamespace(name="CUSTOM")), compilation_config=compilation)
    _ensure_exact_lod_decode_capture_sizes(config)
    assert compilation.cudagraph_capture_sizes==[8]
    assert compilation.max_cudagraph_capture_size==8


def test_query_fallback_preserves_rank_and_head_order_and_reuses_native_replication():
    query = torch.arange(8*12*192).reshape(8,12,192).float()
    wire = torch.empty(8,8,12,192)
    full = torch.empty(8,96,192)
    calls = []
    def gather(out,source):
        for peer in range(8):
            out.view(8,8,12,192)[peer].copy_(source+peer*1e6)
        calls.append(1)
    pool = SimpleNamespace(owner_decode_buffers=dict(query_wire=wire,replicated_query=full),
        dcp_group=SimpleNamespace(device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=False,all_gather=gather))))
    result = owner_decode_query(pool,query,None)
    expected = torch.cat([query+rank*1e6 for rank in range(8)],dim=1)
    torch.testing.assert_close(result,expected)
    assert result is full
    assert owner_decode_query(pool,query,expected) is expected
    assert calls==[1]


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("order", [tuple(range(8)), (3,4,5,6,7,0,1,2)])
def test_capture_math_preserves_request_and_tp_head_order(rank, order):
    index = order.index(rank)
    query = torch.arange(8*96*192, dtype=torch.float32).reshape(8,96,192) / 1000
    record = torch.zeros(8,1,576)
    uk = torch.zeros(96,128,512)
    uk[:, :, :128] = torch.eye(128)
    uv = uk.transpose(1,2).contiguous()
    calls = []
    def attention(q, k, v, out):
        torch.testing.assert_close(q[..., :128], query[index:index+1, :, :128])
        torch.testing.assert_close(q[..., 512:], query[index:index+1, :, 128:])
        out.copy_(q[..., :512])
    def scatter(out, source):
        expected = torch.zeros_like(source)
        expected[:, index] = query[index, :, :128].reshape(8,12,128)
        torch.testing.assert_close(source, expected)
        out.copy_(query[:, rank*12:(rank+1)*12, :128])
        calls.append(1)
    empty = lambda *s: torch.empty(*s)
    buffers = dict(input_index=torch.tensor([index]), query=empty(1,96,192),
        key=empty(1,1,576), q_latent=empty(96,1,512), absorbed=empty(1,96,576),
        output=empty(1,96,512), projected=empty(96,1,128),
        send=empty(8,8,12,128), receive=empty(8,12,128))
    pool = SimpleNamespace(owner_decode_buffers=buffers,
        layer=SimpleNamespace(_lod_owner_uk=uk, _lod_owner_uv=uv),
        owner_decode_pool=SimpleNamespace(decode_dcp=attention),
        dcp_group=SimpleNamespace(device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=False, reduce_scatter=scatter))))
    pointers = {k:t.data_ptr() for k,t in buffers.items()}
    for _ in range(2):
        result = owner_decode_attention(pool, query, record)
        torch.testing.assert_close(result, query[:, rank*12:(rank+1)*12, :128])
    assert calls == [1,1]
    assert pointers == {k:t.data_ptr() for k,t in buffers.items()}


def test_installation_and_catchup_are_outside_capture_and_keep_fixed_pool():
    actions = []
    decode = SimpleNamespace(catch_up_batches=0)
    def catchup(requests):
        actions.append(("catchup", requests))
        if requests[0][1] == 33024:
            decode.catch_up_batches += 1
    decode.catch_up_many = catchup
    decode.install = lambda slot,cache:actions.append(("install",slot,cache))
    decode.ensure_unified_page1_fixed = lambda rows:actions.append(("directory",rows))
    cache = SimpleNamespace(state=dict(recent_len=256, recent_k=torch.zeros(1,1,512,576)))
    row = dict(total=32768, parts=[], cache=cache)
    pool = SimpleNamespace(dcp_rank=3, _kimi_request_owner_rows={3:row},
        owner_decode_pool=decode, owner_decode_input_row=3,
        owner_decode_buffers=dict(input_index=torch.tensor([3])),
        engine=SimpleNamespace(reset_runtime_cache=lambda:None), metadata=[{} for _ in range(8)])
    requests = [(i,32768) for i in range(8)]
    prepare_owner_decode(pool, requests)
    assert row["cache"] is None and row["decode_pool"] is decode
    assert cache.state["recent_k"].shape == (1,1,256,576)
    assert cache.state["recent_v"].untyped_storage().data_ptr() == cache.state["recent_k"].untyped_storage().data_ptr()
    assert pool._kimi_owner_decode_tokens == 1 and pool._kimi_owner_decode_updates == 0
    # Simulate captured replay between scheduler calls. Host lengths use the
    # global per-request index, never a batch- or DCP-divided sequence length.
    row["total"] = 33024
    order = (3,4,5,6,7,0,1,2)
    prepare_owner_decode(pool, [(i,33024) for i in order])
    assert pool.owner_decode_input_row == 0
    assert pool.owner_decode_buffers["input_index"].item() == 0
    assert pool._kimi_owner_decode_updates == 1 and pool._kimi_owner_decode_tokens == 2
    assert sum(a[0]=="install" for a in actions) == 1
    assert all(m["total_len"]==33025 for m in pool.metadata)


def test_owner_updates_reuse_layer_batched_catchup_without_changing_global_boundary(monkeypatch):
    from vllm_lod_plugin.models import kimi_k3_owner_decode

    children = [SimpleNamespace(catch_up_batches=0) for _ in range(2)]
    actions = []
    for child in children:
        child.catch_up_many = lambda requests: actions.append(("fallback", requests))
        child.ensure_unified_page1_fixed = lambda rows: actions.append(("directory", rows))
    parents = [SimpleNamespace(owner_decode_pool=child,
        _kimi_request_owner_rows={3: {"decode_pool": child}}) for child in children]
    monkeypatch.setattr(kimi_k3_owner_decode, "prepare_owner_decode",
        lambda pool, requests, *, catch_up: actions.append(("prepare", catch_up)))
    def batched(row, previous, *, pools):
        assert row == 0 and previous == 33024 and pools == tuple(children)
        for child in pools:
            child.catch_up_batches += 1
        actions.append(("batched", previous))
        return True
    runtime = SimpleNamespace(pools=dict(enumerate(parents)), dcp_rank=3,
        _lod_row=lambda row:row, logical_lengths={}, _catch_up_one_across_layers=batched)
    kimi_k3_owner_decode.prepare_owner_decode_batch(runtime,[(i,33024) for i in range(8)])
    assert actions == [("prepare",False)]*2 + [("batched",33024)] + [("directory",(0,))]*2
    assert [p._kimi_owner_decode_updates for p in parents] == [1,1]
    assert runtime.logical_lengths == dict.fromkeys(range(8),33025)


def test_owner_handoff_releases_parent_construction_scratch_only_once(monkeypatch):
    import weakref
    from vllm_lod_plugin.models import kimi_k3_owner_decode

    children = [SimpleNamespace(catch_up_batches=0,
        state={"leaf": torch.ones(2)}, dcp_decode_buffer_storage={"scratch": torch.ones(3)},
        ensure_unified_page1_fixed=lambda rows: None) for _ in range(2)]
    parents = [SimpleNamespace(owner_decode_pool=child,
        engine=SimpleNamespace(_lod_state_update_buffers={"temporary": torch.ones(4)},
                               _lod_state_maxsim_buffers={"scores": torch.ones(5)}),
        _kimi_request_owner_rows={3: {"decode_pool": None}}) for child in children]
    pointers = [(child.state["leaf"].data_ptr(),
                 child.dcp_decode_buffer_storage["scratch"].data_ptr()) for child in children]
    temporaries = [weakref.ref(parent.engine._lod_state_update_buffers["temporary"])
                   for parent in parents]
    decode_catchup = {"delta": torch.ones(6)}
    runtime = SimpleNamespace(pools=dict(enumerate(parents)), dcp_rank=3,
        logical_lengths={}, _catch_up_one_across_layers=lambda *args, **kwargs: True,
        _cross_layer_shared_lod_state_update_buffers=decode_catchup)
    def prepare(pool, requests, *, catch_up):
        assert catch_up is False
        pool._kimi_request_owner_rows[3]["decode_pool"] = pool.owner_decode_pool
    monkeypatch.setattr(kimi_k3_owner_decode, "prepare_owner_decode", prepare)
    kimi_k3_owner_decode.prepare_owner_decode_batch(runtime, [(i, 32768) for i in range(8)])
    assert all(temporary() is None for temporary in temporaries)
    assert all(not hasattr(parent.engine, "_lod_state_maxsim_buffers") for parent in parents)
    # A later step must not repeat cleanup, including if another caller has
    # retained a parent workspace. Decode catch-up scratch is always kept.
    parents[0].engine._lod_state_update_buffers = {"new": torch.ones(7)}
    kimi_k3_owner_decode.prepare_owner_decode_batch(runtime, [(i, 32768) for i in range(8)])
    assert "new" in parents[0].engine._lod_state_update_buffers
    assert runtime._cross_layer_shared_lod_state_update_buffers is decode_catchup
    assert torch.all(decode_catchup["delta"] == 1)
    assert pointers == [(child.state["leaf"].data_ptr(),
                         child.dcp_decode_buffer_storage["scratch"].data_ptr()) for child in children]
    assert all(torch.all(child.state["leaf"] == 1) for child in children)


def test_graph_replay_audit_counts_execution_not_instantiation():
    from benchmarks.kimi_k3_prefill_sweep import owner_decode_graph_replays
    executed = []
    manager = SimpleNamespace(run_fullgraph=lambda desc:executed.append(desc.num_tokens))
    worker = SimpleNamespace(model_runner=SimpleNamespace(cudagraph_manager=manager))
    assert owner_decode_graph_replays(worker) == 0
    wrapper = manager.run_fullgraph
    wrapper(SimpleNamespace(num_tokens=8))
    wrapper(SimpleNamespace(num_tokens=4))
    wrapper(SimpleNamespace(num_tokens=8))
    assert owner_decode_graph_replays(worker) == 2
    assert manager.run_fullgraph is wrapper and executed == [8,4,8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP state/directory validation")
@pytest.mark.parametrize("capacity", [32768, 65536])
@torch.inference_mode()
def test_pool_backed_owner_prefill_preserves_cache_and_captured_decode(monkeypatch, capacity):
    """Real two-block construction and decode; backing reuse changes no math."""
    from contextlib import nullcontext
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool
    from vllm_lod_plugin.models import kimi_k3_request_prefill as module
    from lod_attention.kernels import paged_decode

    device = torch.device("cuda")
    torch.manual_seed(67)
    layer = SimpleNamespace(num_heads=96, num_kv_heads=1, head_size=576,
                            kv_lora_rank=512, scale=192**-.5,
                            _vllm_lod_absorbed_mla=True)
    pools = [VLLMLayerLODPool(layer, settings=VLLMLODSettings(), max_requests=1,
        request_capacity=capacity, active_indices=torch.zeros(1,dtype=torch.long,device=device),
        dtype=torch.bfloat16, device=device, request_owner_prefill=False) for _ in range(2)]
    records = torch.randn(32768,1,576,device=device,dtype=torch.bfloat16)
    parents = []
    for index, child in enumerate(pools):
        engine = child.engine
        # Isolate backing reuse/physical mean reuse from the newly fused
        # atomic union ordering. The fusion has separate set/oracle checks.
        engine._kimi_fuse_compact_union = False
        # Only construction is under test here. Avoid the costly 96-head
        # prefill readout; below, actual routed decode checks the cache output.
        monkeypatch.setattr(engine, "_prefill_local_attention",
            lambda q, *args, **kwargs: (q[..., :128], None))
        monkeypatch.setattr(engine, "_two_level_attention",
            lambda q, *args, **kwargs: q[..., :128])
        monkeypatch.setattr(module, "projection_scope", lambda *args, **kwargs: nullcontext())
        monkeypatch.setenv("LOD_KIMI_OWNER_POOL_BACKED_PREFILL", str(index))
        parent = SimpleNamespace(engine=engine, owner_decode_pool=child,
                                 _kimi_request_owner_rows={})
        for previous in (0,16384):
            module.attend_slice(parent,0,torch.zeros(16384,1,192,device=device,dtype=torch.bfloat16),
                records[previous:previous+16384],None,None,previous=previous,prompt=32768)
        parents.append(parent)
    caches = [p._kimi_request_owner_rows[0]["cache"] for p in parents]
    reference, backed = [cache.state for cache in caches]
    for name in ("coverage", "total_len", "recent_len", "state_len", "scheduled_state_len"):
        assert reference[name] == backed[name]
    for name in ("state_k", "state_v", "counts", "recent_k", "recent_v", "sink_k", "sink_v"):
        actual = backed[name][..., :reference[name].size(2), :]
        torch.testing.assert_close(reference[name],actual,atol=0,rtol=0,msg=name)
    assert torch.count_nonzero(backed["counts"][..., reference["counts"].size(2):, :]) == 0
    for name in ("leaf_k", "leaf_v", "slot_lengths", "next_page"):
        left, right = reference["page_cache"][name], backed["page_cache"][name]
        if name.startswith("leaf_"):
            stop = int(reference["page_cache"]["leaf_count"])
            left, right = left[...,:stop,:], right[...,:stop,:]
        elif name == "slot_lengths":
            right = right[...,:left.size(2)]
        torch.testing.assert_close(left,right,atol=0,rtol=0,msg=name)
    def memberships(page):
        # Parallel page-ID reservation need not give identical physical IDs.
        # Compare the actual leaf set behind every centroid, not allocator order.
        roots, directories, indices, lengths = [page[name][0,0].cpu() for name in (
            "slot_pages", "overflow_page_values", "page_indices", "slot_lengths")]
        result = []
        for slot, length in enumerate(lengths[:int(reference["state_capacity"])].tolist()):
            leaves = []
            for ordinal in range((length+15)//16):
                directory = int(roots[slot,ordinal//64])
                assert directory >= 0
                physical = int(directories[directory,ordinal%64])
                assert physical >= 0
                leaves.extend(indices[physical,:min(16,length-ordinal*16)].tolist())
            assert all(0 <= leaf < int(page["leaf_count"]) for leaf in leaves)
            result.append(sorted(leaves))
        return result
    assert memberships(reference["page_cache"]) == memberships(backed["page_cache"])
    assert backed["page_cache"]["leaf_k"].data_ptr() == pools[1].state["page_cache"]["leaf_k"].data_ptr()
    assert reference["page_cache"]["leaf_k"].data_ptr() != pools[0].state["page_cache"]["leaf_k"].data_ptr()
    assert backed["recent_k"].data_ptr() != pools[1].state["recent_k"].data_ptr()
    assert backed["pool_backed"] is False and backed["owner_remote_pool_backed"] is True
    q = torch.randn(1,96,576,device=device,dtype=torch.bfloat16)
    key = torch.randn(1,1,576,device=device,dtype=torch.bfloat16)
    outputs = []
    mean_view = paged_decode.coarse_route_mean_view
    reused = []
    def record_mean_reuse(*args):
        view = mean_view(*args)
        assert view is not None and view.stride(1) == 0
        reused.append(view.shape)
        return view
    for index, (pool, cache) in enumerate(zip(pools,caches,strict=True)):
        pointer = pool.state["page_cache"]["leaf_k"].data_ptr()
        pool.install(0,cache)
        assert pool.state["page_cache"]["leaf_k"].data_ptr() == pointer
        pool.active_decode_rows = (0,)
        out = torch.empty(1,96,512,device=device,dtype=torch.bfloat16)
        # Compare the original sum/divide router against physical mean reuse
        # on the same real cache and random query. No routing score rounding.
        with monkeypatch.context() as patch:
            patch.setattr(paged_decode, "coarse_route_mean_view",
                (lambda *args: None) if index == 0 else record_mean_reuse)
            pool.decode_dcp(q,key,key[...,:512],out)
        outputs.append(out)
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
    assert reused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP graph/kernel validation")
@pytest.mark.parametrize("fused_union", [False, True])
@torch.inference_mode()
def test_owner_attention_graph_replay_matches_eager_with_fixed_cache_pointers(fused_union):
    from vllm_lod_plugin.config import VLLMLODSettings

    device = torch.device("cuda")
    torch.manual_seed(29)
    def gather(out, source):
        out.view(8,*source.shape).copy_(source.unsqueeze(0).expand(8,*source.shape))
    comm = SimpleNamespace(disabled=False, all_gather=gather,
        reduce_scatter=lambda out,source:out.copy_(source[0]))
    group = SimpleNamespace(world_size=8,
        device_communicator=SimpleNamespace(pynccl_comm=comm))
    layer = SimpleNamespace(W_UK_T=torch.randn(12,128,512, device=device,dtype=torch.bfloat16)/32,
        W_UV=torch.randn(12,512,128,device=device,dtype=torch.bfloat16)/32)
    parent = SimpleNamespace(engine=SimpleNamespace(scaling=1/192**0.5),
        settings=VLLMLODSettings(), request_capacity=32768, device=device,
        dtype=torch.bfloat16, dcp_rank=0, dcp_group=group, layer=layer)
    initialize_owner_decode(parent)
    decode = parent.owner_decode_pool
    decode.engine._kimi_fuse_compact_union = fused_union
    prefix = torch.randn(1,1,16384,576,device=device,dtype=torch.bfloat16)
    cache = decode.engine.build_cache_from_bf16(prefix,prefix[...,:512])
    q = torch.randn(8,96,192,device=device,dtype=torch.bfloat16)
    key = torch.randn(8,1,576,device=device,dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode.install(0,cache)
        expected = owner_decode_attention(parent,q,key).clone()
    torch.cuda.current_stream().wait_stream(stream)
    decode.install(0,cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        owner_decode_attention(parent,q,key)
    decode.install(0,cache)
    pointers = {k:t.data_ptr() for k,t in parent.owner_decode_buffers.items()}
    graph.replay()
    torch.cuda.synchronize()
    # Atomic union order can change summation rounding, not membership.
    # Keep the deterministic separate-union control bitwise; exercise the
    # default fused graph with numerical tolerance and unchanged pointers.
    tolerance = dict(atol=3e-4, rtol=.015) if fused_union else dict(atol=0, rtol=0)
    torch.testing.assert_close(parent.owner_decode_buffers["receive"],expected,**tolerance)
    assert decode.local_lens.item()==int(cache.state["recent_len"])+1
    # The captured graph uses a device row map, not a baked-in Python slice.
    decode.install(0,cache)
    parent.owner_decode_buffers["input_index"].fill_(5)
    expected = owner_decode_attention(parent,q,key).clone()
    decode.install(0,cache)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(parent.owner_decode_buffers["receive"],expected,**tolerance)
    assert pointers=={k:t.data_ptr() for k,t in parent.owner_decode_buffers.items()}


@pytest.mark.skipif(not torch.cuda.is_available(),reason="HIP one-token projection validation")
@torch.inference_mode()
def test_fused_owner_projections_match_bmm_and_report_graphed_cost():
    from lod_attention.kernels.kimi_owner_decode import owner_query_projection,owner_value_projection
    device="cuda"
    torch.manual_seed(31)
    q=torch.randn(8,96,192,device=device,dtype=torch.bfloat16)
    key=torch.randn(8,1,576,device=device,dtype=torch.bfloat16)
    uk=torch.randn(96,128,512,device=device,dtype=torch.bfloat16)/32
    uv=torch.randn(96,512,128,device=device,dtype=torch.bfloat16)/32
    index=torch.tensor([5],device=device)
    absorbed=torch.empty(1,96,576,device=device,dtype=torch.bfloat16)
    selected=torch.empty(1,1,576,device=device,dtype=torch.bfloat16)
    output=torch.randn(1,96,512,device=device,dtype=torch.bfloat16)
    send=torch.empty(8,8,12,128,device=device,dtype=torch.bfloat16)
    def fused():
        owner_query_projection(q,key,uk,index,absorbed,selected)
        owner_value_projection(output,uv,index,send)
    fused()
    expected_q=torch.cat((torch.bmm(q[5:6,:,:128].transpose(0,1),uk).transpose(0,1),q[5:6,:,128:]),dim=-1)
    expected_v=torch.bmm(output.transpose(0,1),uv).reshape(8,12,128)
    torch.testing.assert_close(absorbed,expected_q,atol=1e-5,rtol=0.008)
    torch.testing.assert_close(selected,key[5:6],atol=0,rtol=0)
    torch.testing.assert_close(send[:,5],expected_v,atol=1e-5,rtol=0.008)
    assert torch.count_nonzero(send[:,:5])==0 and torch.count_nonzero(send[:,6:])==0
    # Same fixed-buffer baseline as the model-level first owner result.
    gathered=torch.empty(1,96,192,device=device,dtype=torch.bfloat16)
    latent=torch.empty(96,1,512,device=device,dtype=torch.bfloat16)
    projected=torch.empty(96,1,128,device=device,dtype=torch.bfloat16)
    def reference():
        torch.index_select(q,0,index,out=gathered)
        torch.index_select(key,0,index,out=selected)
        torch.bmm(gathered[...,:128].transpose(0,1),uk,out=latent)
        absorbed[...,:512].copy_(latent.transpose(0,1))
        absorbed[...,512:].copy_(gathered[...,128:])
        torch.bmm(output.transpose(0,1),uv,out=projected)
        send.zero_()
        torch.index_copy(send,1,index,projected.reshape(8,1,12,128),out=send)
    times={}
    for name,fn in [("bmm",reference),("fused",fused)]:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):fn()
        torch.cuda.current_stream().wait_stream(stream)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        g.replay();torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):g.replay()
        end.record();end.synchronize()
        times[name]=start.elapsed_time(end)/100
    print("OWNER_PROJECTION_GRAPH_MS",times,flush=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP layer-batched owner catch-up")
@torch.inference_mode()
def test_layer_batched_owner_update_matches_individual_caches_and_keeps_pointers():
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool
    from vllm_lod_plugin.runtime import VLLMLODRuntime

    torch.manual_seed(37)
    device = torch.device("cuda")
    layer = SimpleNamespace(num_heads=96, num_kv_heads=1, head_size=576,
        kv_lora_rank=512, scale=1/192**0.5, _vllm_lod_absorbed_mla=True)
    children, expected, pointers = [], [], []
    for _ in range(2):
        pool = VLLMLayerLODPool(layer, settings=VLLMLODSettings(),
            max_requests=1, request_capacity=32768,
            active_indices=torch.zeros(1, device=device, dtype=torch.long),
            dtype=torch.bfloat16, device=device, request_owner_prefill=False)
        prefix = torch.randn(1,1,16384,576,device=device,dtype=torch.bfloat16)
        cache = pool.engine.build_cache_from_bf16(prefix,prefix[...,:512])
        overflow = torch.randn(1,1,256,576,device=device,dtype=torch.bfloat16)
        def install():
            pool.install(0,cache)
            recent = int(cache.state["recent_len"])
            pool.state["recent_k"][...,recent:recent+256,:].copy_(overflow)
        install()
        pool.catch_up_many([(0,16640)])
        pool.ensure_unified_page1_fixed((0,))
        expected.append({name:pool.state[name].clone() for name in ("state_k","counts")})
        for name in ("slot_lengths","unified_page1_fixed_indices",
                     "unified_page1_fixed_slot_offsets","unified_page1_fixed_lengths"):
            expected[-1][name] = pool.state["page_cache"][name].clone()
        expected[-1]["metadata"] = dict(pool.metadata[0])
        install()
        children.append(pool)
        pointers.append({name:pool.state[name].data_ptr() for name in ("state_k","counts","recent_k")})
    runtime = VLLMLODRuntime.__new__(VLLMLODRuntime)
    runtime.dcp_rank = 0
    assert runtime._catch_up_one_across_layers(0,16640,pools=tuple(children))
    torch.cuda.synchronize()
    for pool, target, addresses in zip(children,expected,pointers,strict=True):
        pool.ensure_unified_page1_fixed((0,))
        for name in ("state_k","counts"):
            torch.testing.assert_close(pool.state[name],target[name],atol=0,rtol=0)
            assert pool.state[name].data_ptr() == addresses[name]
        assert pool.state["recent_k"].data_ptr() == addresses["recent_k"]
        # Physical directory-page allocation order and unused page padding
        # need not agree. Compare the actual consumer's semantic leaf list.
        length = int(pool.state["page_cache"]["unified_page1_fixed_lengths"].item())
        torch.testing.assert_close(pool.state["page_cache"]["unified_page1_fixed_indices"][...,:length],
            target["unified_page1_fixed_indices"][...,:length],atol=0,rtol=0)
        for name in ("slot_lengths","unified_page1_fixed_slot_offsets","unified_page1_fixed_lengths"):
            torch.testing.assert_close(pool.state["page_cache"][name],target[name],atol=0,rtol=0)
        assert pool.metadata[0] == target["metadata"]
