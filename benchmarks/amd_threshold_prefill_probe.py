"""Compare remote prefill stages on identical real Qwen queries and KV caches.

Snapshot copying and diagnostics are outside timing. Remote attention includes
routing, coarse attention, grouped exact leaves, packing, and the final merge.
Local attention and cache updates are covered only by the full-model runs.
"""
import argparse
import json
import os
from pathlib import Path
import types


def capture(model):
    pools=[]; seen=set()
    for module in model.modules():
        pool=getattr(module,'_vllm_lod_pool',None)
        if pool is not None and id(pool) not in seen:
            seen.add(id(pool)); pools.append(pool)
    if not pools:
        raise RuntimeError('no LoD pools')
    for index in sorted({0,len(pools)//2,len(pools)-1}):
        pool=pools[index]; original=pool.engine._route_top_slots
        def route(self,q,k,v,c,*,state_len,page_cache=None,_original=original,_pool=pool):
            result=_original(q,k,v,c,state_len=state_len,page_cache=page_cache)
            if q.size(2)==16384:
                import torch
                names=('leaf_k','leaf_v','page_indices','slot_pages','overflow_page_keys',
                       'overflow_page_values','overflow_used','slot_lengths')
                _pool._prefill_probe=dict(q=q.contiguous().clone(),k=k.clone(),v=v.clone(),c=c.clone(),s=state_len,
                    pages={name:page_cache[name].clone() for name in names},
                    sink_k=_pool.state['sink_k'].clone(),sink_v=_pool.state['sink_v'].clone())
                for name in ('paged_page_directory','overflow_active'):
                    if name in page_cache:
                        _pool._prefill_probe['pages'][name]=page_cache[name]
            return result
        pool.engine._route_top_slots=types.MethodType(route,pool.engine)
        pool._prefill_probe_layer=index
    return len(pools)


def probe(model):
    import torch
    from triton.testing import do_bench_cudagraph
    from benchmarks._identity import benchmark_identity
    from lod_attention.kernels.aiter_prefill_attention import aiter_prefill_route_coarse_attention, merge_aiter_prefill_refinement
    from lod_attention.kernels.threshold_prefill import threshold_prefill_route_coarse_attention
    from lod_attention.kernels.paged_prefill import paged_leaf_attention
    torch.set_grad_enabled(False)
    result=dict(stage='running',benchmark_identity=benchmark_identity(),gpu=torch.cuda.get_device_name(),records=[])
    seen=set()
    for module in model.modules():
        pool=getattr(module,'_vllm_lod_pool',None)
        if pool is None or id(pool) in seen or not hasattr(pool,'_prefill_probe'):
            continue
        seen.add(id(pool)); data=pool._prefill_probe; q=data['q']; page=data['pages']; engine=pool.engine
        b,h,t,d=q.shape; kv=pool.kv_heads; group=h//kv
        snapshot_dir=os.environ.get('LOD_AMD_PREFILL_SNAPSHOT_DIR')
        if snapshot_dir:
            def cpu(value):
                if isinstance(value,torch.Tensor):return value.cpu()
                if isinstance(value,dict):return {key:cpu(item) for key,item in value.items()}
                return value
            destination=Path(snapshot_dir)/f'layer-{pool._prefill_probe_layer:02d}.pt'
            destination.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(data=cpu(data),layer=pool._prefill_probe_layer,kv_heads=kv,
                engine={key:getattr(engine,key) for key in ('scaling','routing_normalization','leaf_block_m','leaf_block_n','leaf_num_warps','leaf_hash_probes')}),destination)
            print('Saved real prefill snapshot: '+str(destination),flush=True)
        local=torch.zeros_like(q);local_lse=torch.full(q.shape[:-1],-torch.inf,device=q.device)
        common=dict(state_len=data['s'],kv_group_size=group,scale=float(engine.scaling),normalize_route_query=True)
        for variant in os.environ.get('LOD_AMD_PREFILL_PROBE_VARIANTS','top8,triton,blas').split(','):
            common['normalize_route_query'] = variant!='top8' or engine.routing_normalization=='query'
            buffers={}
            os.environ['LOD_AMD_PREFILL_THRESHOLD']='16'
            os.environ['LOD_AMD_PREFILL_MATMUL']=variant if variant!='top8' else 'triton'
            function=aiter_prefill_route_coarse_attention if variant=='top8' else threshold_prefill_route_coarse_attention
            geometry=[engine.leaf_block_m,engine.leaf_block_n,engine.leaf_num_warps]
            def route():
                return function(q,data['k'],data['v'],data['c'],**common,buffers=buffers)
            def leaves(slots,coarse,hc,offset):
                return paged_leaf_attention(q,page['leaf_k'],page['leaf_v'],page['slot_pages'],page['overflow_page_keys'],
                    page['overflow_page_values'],page['overflow_used'],page['slot_lengths'],slots,page_indices=page['page_indices'],
                    kv_group_size=group,active_slots=data['s'],scale=float(engine.scaling),block_m=geometry[0],
                    block_n=geometry[1],num_warps=geometry[2],hash_probes=engine._page_lookup_probes(page),
                    route_head_counts=hc,route_offsets=offset,route_counts=coarse.route_counts,
                    persistent_workers=int(os.environ.get('LOD_AMD_PREFILL_WORKERS','608')) if variant!='top8' else 0,
                    variable_reduce_backend=os.environ.get('LOD_AMD_PREFILL_REDUCE','bucket'),
                    reduce_routes=coarse.closed_regions_only,buffers=buffers)
            slots,coarse,hc,offset=route()
            # Production also leaves oversized regions coarse after top-eight ranking.
            if variant=='top8':
                sizes=page['slot_lengths'][:,:,:data['s']].repeat_interleave(group,1)
                selected=torch.gather(sizes[:,:,None,:].expand(-1,-1,t,-1),-1,slots.clamp_min(0))
                slots=torch.where(selected<=1024,slots,-1);hc=offset=None
            exact,lse=leaves(slots,coarse,hc,offset)
            def remote():
                ss,cc,hh,oo=route()
                if variant=='top8':
                    sizes=page['slot_lengths'][:,:,:data['s']].repeat_interleave(group,1)
                    selected=torch.gather(sizes[:,:,None,:].expand(-1,-1,t,-1),-1,ss.clamp_min(0))
                    ss=torch.where(selected<=1024,ss,-1);hh=oo=None
                ee,ll=leaves(ss,cc,hh,oo)
                return merge_aiter_prefill_refinement(q,data['sink_k'],data['sink_v'],cc,ss,ee,ll,local,local_lse,
                    kv_group_size=group,scale=float(engine.scaling))
            output=remote()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('nonfinite remote attention')
            validation={}
            if variant!='top8':
                qs=q[:,:,::max(t//16,1)].contiguous();small_buffers={}
                ss,cc,*_=function(qs,data['k'],data['v'],data['c'],**common,buffers=small_buffers)
                size=data['c'][..., :data['s'],0].float().repeat_interleave(group,1)
                mk=cc.mean_k.float().repeat_interleave(group,1)
                mv=cc.mean_v.float().repeat_interleave(group,1)
                native=torch.einsum('bhtd,bhsd->bhts',qs.float(),mk)*float(engine.scaling)
                score=native/qs.float().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()+size.clamp_min(1).log()[:,:,None,:]
                probability=score.masked_fill(size[:,:,None,:]<=0,-torch.inf).softmax(-1)
                cut=16/(size>0).sum(-1).clamp_min(1)[:,:,None,None]
                expected=(probability>cut)&(size[:,:,None,:]>0)&(size[:,:,None,:]<=1024)
                actual=torch.zeros_like(expected,dtype=torch.int32)
                actual.scatter_add_(-1,ss.clamp_min(0),(ss>=0).int())
                difference=actual.bool()!=expected
                outside=difference&((probability-cut).abs()>2e-5)
                if bool(outside.any()):raise RuntimeError('real prefill routing mismatch outside boundary tolerance')
                closed=(native+size.clamp_min(1).log()[:,:,None,:]).masked_fill(~((size[:,:,None,:]>0)&~actual.bool()),-torch.inf)
                ref=torch.einsum('bhts,bhsd->bhtd',closed.softmax(-1).nan_to_num(),mv)
                torch.testing.assert_close(cc.output_0.float(),ref,atol=.03,rtol=.02)
                torch.testing.assert_close(cc.lse_0,closed.logsumexp(-1),atol=.003,rtol=.0001)
                validation=dict(sampled_query_heads=int(qs.numel()//d),selection_differences=int(difference.sum()),
                    outside_2e5_tolerance=int(outside.sum()),max_abs_coarse_error=float((cc.output_0.float()-ref).abs().max()))
                del qs,small_buffers,ss,cc,size,mk,mv,native,score,probability,cut,expected,actual,difference,outside,closed,ref
            selected=(slots>=0)
            sizes=page['slot_lengths'].repeat_interleave(group,1)
            opened=torch.gather(sizes[:,:,None,:].expand(-1,-1,t,-1),-1,slots.clamp_min(0))*selected
            row=dict(layer=pool._prefill_probe_layer,variant=variant,queries=t,regions=data['s'],head_dim=d,
                mean_open_regions=float(selected.sum(-1).float().mean()),mean_open_entries=float(opened.sum(-1).float().mean()),
                route_coarse_ms=do_bench_cudagraph(route,rep=150),
                leaves_ms=do_bench_cudagraph(lambda:leaves(slots,coarse,hc,offset),rep=150),
                remote_attention_ms=do_bench_cudagraph(remote,rep=200),validation=validation)
            print(json.dumps(row),flush=True); result['records'].append(row)
            if variant!='top8' and os.environ.get('LOD_AMD_PREFILL_TUNE_WORKERS')=='1':
                previous=os.environ.get('LOD_AMD_PREFILL_WORKERS')
                for workers in (304,608,1216,2432,4864):
                    os.environ['LOD_AMD_PREFILL_WORKERS']=str(workers)
                    candidate,candidate_lse=leaves(slots,coarse,hc,offset)
                    if not torch.equal(candidate,exact) or not torch.equal(candidate_lse,lse):
                        raise RuntimeError('worker count changed exact attention output')
                    item=dict(layer=pool._prefill_probe_layer,variant=variant,workers=workers,
                        leaves_ms=do_bench_cudagraph(lambda:leaves(slots,coarse,hc,offset),rep=150),
                        remote_attention_ms=do_bench_cudagraph(remote,rep=150),bitwise_identical=True)
                    print(json.dumps(item),flush=True);result['records'].append(item)
                if previous is None:os.environ.pop('LOD_AMD_PREFILL_WORKERS',None)
                else:os.environ['LOD_AMD_PREFILL_WORKERS']=previous
            if variant!='top8' and os.environ.get('LOD_AMD_PREFILL_PROFILE_LEAVES')=='1':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    leaves(slots,coarse,hc,offset);torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by='self_device_time_total',row_limit=18),flush=True)
            if variant!='top8' and os.environ.get('LOD_AMD_PREFILL_TUNE_LEAVES')=='1':
                for config in [(32,32,2),(32,64,4),(64,32,4),(64,64,4),(128,32,4),(128,64,4)]:
                    geometry[:]=config
                    candidate, candidate_lse=leaves(slots,coarse,hc,offset)
                    try:
                        torch.testing.assert_close(candidate,exact,atol=.012,rtol=.025)
                        torch.testing.assert_close(candidate_lse,lse,atol=.002,rtol=.001)
                    except AssertionError as exc:
                        row=dict(layer=pool._prefill_probe_layer,variant=variant,geometry=config,valid=False,
                                 reason=str(exc),max_output_difference=float((candidate.float()-exact.float()).abs().max()))
                        print(json.dumps(row),flush=True);result['records'].append(row)
                        continue
                    row=dict(layer=pool._prefill_probe_layer,variant=variant,geometry=config,
                        leaves_ms=do_bench_cudagraph(lambda:leaves(slots,coarse,hc,offset),rep=120),
                        remote_attention_ms=do_bench_cudagraph(remote,rep=150),
                        max_output_difference=float((candidate.float()-exact.float()).abs().max()))
                    print(json.dumps(row),flush=True);result['records'].append(row)
                geometry[:]=[engine.leaf_block_m,engine.leaf_block_n,engine.leaf_num_warps]
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    remote();torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by='self_device_time_total',row_limit=16),flush=True)
            del buffers,slots,coarse,exact,lse,output,opened,selected
            torch.cuda.empty_cache()
        del pool._prefill_probe
    result['stage']='complete'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--length',type=int,default=65536)
    args=parser.parse_args()
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    from benchmarks._vllm import llm_kwargs,close_llm
    from benchmarks.prolong import make_speed_prompts
    checkpoint='Qwen/Qwen3.8-27B-FP8'
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)
    prompts,metadata=make_speed_prompts(tokenizer,length=args.length,batch_size=1)
    kwargs=llm_kwargs(checkpoint=checkpoint,mode='two-tier',max_model_len=args.length+32,batch_size=1,
                      tensor_parallel_size=1,gpu_memory_utilization=.7,full_attention_backend='ROCM_AITER_UNIFIED_ATTN')
    llm=LLM(**kwargs)
    try:
        llm.apply_model(capture)
        llm.generate(prompts,SamplingParams(temperature=0,seed=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
        result=llm.apply_model(probe)
        if len(result)!=1:raise RuntimeError('expected single GPU worker')
        result=result[0];result.update(context=args.length,prompt_metadata=metadata)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:close_llm(llm)


if __name__=='__main__':main()
