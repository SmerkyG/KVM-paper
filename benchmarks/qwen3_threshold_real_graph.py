"""Read-only optimized two-tier thresholds on the paper's real Qwen3 fixture.

No dummy current token, KV append, cache update, or model work is included.
Both local/sink scopes use the same captured queries, cache and frontier checks.
"""
import math
import os
from types import SimpleNamespace
import torch


def panel(data):
    from benchmarks.retro_lod_adapter import install_lod_modules
    install_lod_modules()
    from benchmarks.qwen3_real_graph import graph_time
    from benchmarks.h100_threshold_decode_materialized import threshold_attention_h100_materialized
    from benchmarks.amd_threshold_frontier import prepare_cache
    from vllm_lod_plugin.config import VLLMLODSettings
    from vllm_lod_plugin.pool import VLLMLayerLODPool
    torch.backends.cuda.matmul.allow_tf32=False
    n=data['metadata']['length'];device=torch.device('cuda');d=128;g=4
    q=data['q_rot'][:,:,0,:].to(device).contiguous();k=data['k_rot'].to(device).contiguous();v=data['v'].to(device).contiguous()
    prior={name:os.environ.get(name) for name in ('LOD_H100_THRESHOLD_IMPLEMENTATION','LOD_AMD_THRESHOLD_ENTRY_BLOCK',
        'LOD_AMD_THRESHOLD_ENTRY_WARPS','LOD_AMD_THRESHOLD_MERGE_BLOCK','LOD_AMD_THRESHOLD_PRECISE')}
    os.environ.update(LOD_H100_THRESHOLD_IMPLEMENTATION='materialized',LOD_AMD_THRESHOLD_ENTRY_BLOCK='128',
        LOD_AMD_THRESHOLD_ENTRY_WARPS='4',LOD_AMD_THRESHOLD_MERGE_BLOCK='32',LOD_AMD_THRESHOLD_PRECISE='1')
    try:
        layer=SimpleNamespace(num_heads=32,num_kv_heads=8,head_size=d,head_size_v=d,impl=SimpleNamespace(scale=d**-.5))
        settings=VLLMLODSettings.production(mode='two-tier',pool_size=1,request_capacity=n+2048)
        pool=VLLMLayerLODPool(layer,settings=settings,max_requests=1,request_capacity=n+2048,
            active_indices=torch.zeros(1,device=device,dtype=torch.int32),dtype=torch.bfloat16,device=device,
            has_query_norm=True,has_key_norm=True)
        storage=pool._initial_prefill_storage((0,));pool.engine._lod_prefill_storage=storage
        try:cache=pool.engine.build_cache_from_bf16(k,v,finalize_cache_for_decode=True)
        finally:del pool.engine._lod_prefill_storage
        pool.install_rows((0,),cache);pool.engine.reset_runtime_cache();pool.catch_up_many([(0,n)])
        prepared=prepare_cache(pool)
        qq=q.unsqueeze(1);dummy=torch.zeros((1,1,8,d),device=device,dtype=q.dtype);out=torch.empty_like(qq)
        old_len=pool.local_lens.clone();query=q[0].view(8,g,d).float();scale=d**-.5
        count=pool.state['counts'][0,:,:,0].float();valid=count>0
        means_k=prepared['mean_k'].float();means_v=prepared['mean_v'].float()
        raw=torch.einsum('hgd,hsd->hgs',query,means_k)*scale
        logc=count.clamp_min(1).log()[:,None,:]
        route=(raw/query.square().mean(-1,keepdim=True).clamp_min(1.e-12).sqrt()+logc).masked_fill(~valid[:,None,:],-torch.inf)
        margin=route-route.logsumexp(-1,keepdim=True)-math.log(16)+valid.sum(-1)[:,None,None].clamp_min(1).float().log()
        opened=valid[:,None,:]&(count[:,None,:]<=1024)&(margin>0)
        from benchmarks.retro_ruler_frontier_audit import reconstruct
        reference_cache=reconstruct(pool,n)[0]
        owner=reference_cache.owner[0].long()[:,None,:].expand(-1,g,-1)
        exact_mask=opened.gather(-1,owner)
        leaf_k,leaf_v=reference_cache.archive_key[0].float(),reference_cache.archive_value[0].float()
        leaf_logits=torch.einsum('hgd,htd->hgt',query,leaf_k)*scale
        rows=[]
        for scope in ('remote','local'):
            def fn():
                threshold_attention_h100_materialized(pool,qq,dummy,dummy,out,16.,tile=64,splits=32,
                    append=False,include_local=scope=='local',include_sink=scope=='local')
                return out
            before=fn().clone()
            assert torch.equal(pool.local_lens,old_len),'read-only benchmark advanced the cache'
            actual=pool._threshold_scratch['allowed'].bool().view_as(opened)
            mismatch=actual!=opened
            assert not bool((mismatch&(margin.abs()>2.e-5)).any()),'threshold selection differs from independent reference'
            # Use the actual mask for a rare near-boundary decision, then check
            # the complete native-query frontier, not merely selected KV.
            exact_mask=actual.gather(-1,owner)
            summary=(raw+logc).masked_fill(~valid[:,None,:]|actual,-torch.inf)
            logits=[summary,leaf_logits.masked_fill(~exact_mask,-torch.inf)]
            values=[means_v,leaf_v]
            if scope=='local':
                length=int(old_len[0]);lk=torch.cat((pool.state['recent_k'][0,:,:length],pool.state['sink_k'][0]),1).float()
                lv=torch.cat((pool.state['recent_v'][0,:,:length],pool.state['sink_v'][0]),1).float()
                logits.append(torch.einsum('hgd,htd->hgt',query,lk)*scale);values.append(lv)
            probability=torch.cat(logits,-1).softmax(-1)
            reference=torch.einsum('hgt,htd->hgd',probability,torch.cat(values,1))
            torch.testing.assert_close(before[0,0].view(8,g,d).float(),reference,atol=.03,rtol=.03)
            error=float((before[0,0].view(8,g,d).float()-reference).abs().max())
            changes={}
            for name in ('recent_v','sink_v'):
                tensor=pool.state[name];saved=tensor.clone();tensor.add_(10)
                changes[name]=float((fn().float()-before.float()).abs().max());tensor.copy_(saved)
            if scope=='remote':assert not any(changes.values())
            else:assert all(x>0 for x in changes.values())
            timing=graph_time(fn)
            assert torch.equal(pool.local_lens,old_len),'graph replay advanced read-only cache'
            row=dict(mode='two-tier-threshold-c16-materialized',scope=scope,rule='region',multiplier=16,
                reference_max_abs=error,selection_mismatches=int(mismatch.sum()),
                regions_per_head=actual.sum(-1).flatten().tolist(),exact_entries_per_head=exact_mask.sum(-1).flatten().tolist(),
                local_sink_perturbation=changes,cache_updates=False,current_token=False,**timing)
            rows.append(row);print(row,flush=True)
        return rows
    finally:
        for name,value in prior.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
