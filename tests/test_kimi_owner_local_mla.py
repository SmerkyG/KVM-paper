import pytest
import torch

from benchmarks._kimi_owner_local_mla import (
    assemble_owner_hidden, local_mla_forward, prepare_owner_tp_mla, validate_owner_plan,
)


def plan(order=range(8), chunk=16384):
    return tuple((slot, i*chunk, (i+1)*chunk, chunk) for i, slot in enumerate(order))


def test_aligned_eight_rows_only():
    validate_owner_plan(plan(), world=8, tokens=8*16384, chunk=16384)
    for invalid in (plan()[:4], plan([0]*8), tuple((s,b,e,p+1) for s,b,e,p in plan())):
        with pytest.raises(ValueError):
            validate_owner_plan(invalid, world=8, tokens=8*16384, chunk=16384)
    with pytest.raises(ValueError):
        validate_owner_plan(plan(), world=8, tokens=16384, chunk=16384)


def test_canonical_output_is_zero_copy():
    wire = torch.arange(8*3*4).view(8,3,4)
    output = assemble_owner_hidden(wire, plan(chunk=3))
    assert output.data_ptr() == wire.data_ptr()
    assert torch.equal(output, wire.reshape(24,4))


def test_rotated_scheduler_order_preserves_rows():
    wire = torch.arange(8*3*4).view(8,3,4)
    order = [3,4,5,6,7,0,1,2]
    output = assemble_owner_hidden(wire, plan(order, chunk=3))
    for index, owner in enumerate(order):
        assert torch.equal(output[index*3:(index+1)*3], wire[owner])


@pytest.mark.parametrize("rank", range(8))
def test_forward_computes_only_owned_row_and_preserves_runtime_metadata(monkeypatch, rank):
    from types import SimpleNamespace
    import benchmarks._kimi_owner_local_mla as probe
    import vllm_lod_plugin.models.kimi_k3_request_prefill as owner

    chunk = 2
    pool = SimpleNamespace(direct_prefill_plan=plan(chunk=chunk),
        direct_prefill_prompt_lengths={row:4 for row in range(8)},
        engine=SimpleNamespace(prefill_chunk_len=chunk), ready=[False]*8,
        dcp_sharded=[True]*8, metadata=[{} for _ in range(8)])
    qw, ow, gw, uk, uv = [object() for _ in range(5)]
    seen = []

    def fused(hidden):
        assert torch.all(hidden == rank+1)
        return (torch.zeros(chunk,2112),)

    def linear(hidden, weight):
        if weight is qw:
            return torch.zeros(hidden.size(0),18432)
        if weight is gw:
            return torch.zeros(hidden.size(0),12288)
        assert weight is ow
        return hidden[:, :4].contiguous()

    def attend(p, slot, query, record, key_map, value_map, **kwargs):
        assert slot == rank and p is pool and key_map is uk and value_map is uv
        assert kwargs == {"previous":chunk,"prompt":4}
        assert query.shape == (chunk,96,192) and record.shape == (chunk,1,576)
        seen.append(slot)
        return torch.full((chunk,96,128), float(rank+1))

    def gather(target, local):
        assert torch.all(local == (rank+1)*0.5)
        for peer, output in enumerate(target.view(8,chunk,4)):
            output.fill_((peer+1)*0.5)

    pool.dcp_group = SimpleNamespace(world_size=8, rank_in_group=rank,
        device_communicator=SimpleNamespace(pynccl_comm=SimpleNamespace(all_gather=gather)))
    wrapper = SimpleNamespace(mla_attn=SimpleNamespace(_vllm_lod_pool=pool),
        rotary_emb=None, fused_qkv_a_proj=fused, _normalize_q_kv=lambda q,k:(q,k),
        _lod_owner_mla_weights=(qw,ow,gw,uk,uv))
    monkeypatch.setattr(probe.F, "linear", linear)
    monkeypatch.setattr(owner, "attend_slice", attend)
    hidden = torch.arange(1,9).float().repeat_interleave(chunk)[:,None].expand(-1,4)
    result = local_mla_forward(wrapper, torch.arange(8*chunk), hidden)
    assert seen == [rank]
    assert torch.equal(result, hidden*0.5)
    assert result.data_ptr() == pool.dcp_group._lod_owner_hidden_arena.data_ptr()
    assert pool.direct_prefill_plan is None and pool.direct_prefill_prompt_lengths == {}
    assert pool.ready == [True]*8 and pool.dcp_sharded == [False]*8
    assert all(m["total_len"] == 4 for m in pool.metadata)
    assert pool._kimi_owner_query_sizes == {chunk:1}


def test_tp_owner_rejects_retained_full_projection_copies():
    from types import SimpleNamespace

    wrapper = SimpleNamespace(_lod_owner_mla_weights=(object(),),
        mla_attn=SimpleNamespace(_vllm_lod_pool=SimpleNamespace(
            kimi_request_owner_prefill=True, dcp_world_size=8)))
    attention = type("KimiMLAAttention", (), {"mla_attn":wrapper})()
    core = type("KimiLinearModel", (), {"layers":[SimpleNamespace(self_attn=attention)]})()
    worker = SimpleNamespace(model_runner=SimpleNamespace(
        model=SimpleNamespace(modules=lambda:[core])))
    with pytest.raises(AssertionError, match="must not retain local MLA"):
        prepare_owner_tp_mla(worker)


def test_tp_owner_prepares_only_head_maps_without_replacing_forward(monkeypatch):
    from types import SimpleNamespace
    import vllm_lod_plugin.models.kimi_k3_sharded_prefill as shard

    uk = torch.zeros(12,128,512, dtype=torch.bfloat16)
    uv = torch.zeros(12,512,128, dtype=torch.bfloat16)
    attention = SimpleNamespace(W_UK_T=uk, W_UV=uv,
        _vllm_lod_pool=SimpleNamespace(kimi_request_owner_prefill=True,
            dcp_world_size=8, dcp_group=object()))
    original_forward = object()
    wrapper = SimpleNamespace(mla_attn=attention, forward=original_forward)
    mla = type("KimiMLAAttention", (), {"mla_attn":wrapper})()
    layer = SimpleNamespace(self_attn=mla, layer_idx=3,
                            mlp=type("KimiMoE", (), {})())
    core = type("KimiLinearModel", (), {"layers":[layer]})()
    worker = SimpleNamespace(rank=0, model_runner=SimpleNamespace(
        model=SimpleNamespace(modules=lambda:[core]),
        model_state=SimpleNamespace(_vllm_lod_runtime=SimpleNamespace(pool_size=1))))
    monkeypatch.setattr(shard, "gather_prefill", lambda group,w,dim:torch.cat([w]*8,dim))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda:None)
    audit = prepare_owner_tp_mla(worker)
    assert audit["language_layers"] == audit["native_moe_layers"] == audit["request_capacity"] == 1
    assert audit["native_kda_layers"] == 0
    assert wrapper.forward is original_forward
    assert not hasattr(wrapper, "_lod_owner_mla_weights")
    assert attention.W_UK_T is uk and attention.W_UV is uv
    assert audit["additional_q_gate_o_copy_bytes"] == 0
    assert audit["additional_projection_bytes"] == 2*96*128*512*2
    assert audit["q_k_v_o_tensor_parallel"]
    # Startup graph capture retains these pointers. A later benchmark audit
    # must not replace them with fresh gathers.
    previous = (attention._lod_owner_uk, attention._lod_owner_uv)
    monkeypatch.setattr(shard, "gather_prefill", lambda *a,**k:pytest.fail("head maps regathered"))
    prepare_owner_tp_mla(worker)
    assert attention._lod_owner_uk is previous[0] and attention._lod_owner_uv is previous[1]
