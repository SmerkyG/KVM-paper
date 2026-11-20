"""Compare remote prefill stages on identical real queries and KV caches.

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
        timed_queries = int(os.environ.get('LOD_AMD_PREFILL_PROBE_QUERIES', str(q.size(2))))
        q = q[..., :timed_queries, :].contiguous()
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
        baseline_probability = os.environ.get('LOD_AMD_PREFILL_PROBABILITY', 'bf16')
        reducer = os.environ.get('LOD_AMD_PREFILL_REDUCE', 'bucket')
        reference_remote = None
        reference_variant = None
        bitmap_setting = os.environ.get('LOD_AMD_PREFILL_BITMAP_RANKS', '0')
        alignment = os.environ.get('LOD_AMD_PREFILL_REGION_ALIGNMENT', '1')
        select_warps = os.environ.get('LOD_AMD_PREFILL_SELECT_WARPS', '8')
        select_rows = os.environ.get('LOD_AMD_PREFILL_SELECT_ROWS', '1')
        group_pack = os.environ.get('LOD_AMD_PREFILL_GROUP_PACK', '1')
        fused_merge = os.environ.get('LOD_AMD_PREFILL_FUSED_MERGE')
        compact_bits = os.environ.get('LOD_AMD_PREFILL_COMPACT_BITS', '0')
        packed_exact_rows = os.environ.get('LOD_AMD_PREFILL_PACK_EXACT_ROWS', '0')
        merge_mma_parts = os.environ.get('LOD_AMD_PREFILL_MERGE_MMA_PARTS', '2')
        variants = os.environ.get('LOD_AMD_PREFILL_PROBE_VARIANTS', 'top8,triton,blas').split(',')
        reference_slots = None
        for variant in variants:
            top8 = variant == 'top8'
            query_major = variant in ('stream-query', 'stream-signature', 'stream-largest')
            bitset = variant in ('blas-bits', 'blas-packed', 'blas-mma1')
            packed = variant in ('stream-packed', 'blas-packed', 'blas-mma1')
            stream = query_major or variant in ('stream', 'stream-fused', 'stream-atomic', 'stream-c24', 'stream-packed')
            fused = bitset or variant in ('stream-fused', 'blas-fused', 'stream-c24', 'stream-packed')
            os.environ['LOD_AMD_PREFILL_FUSED_MERGE'] = '1' if fused else '0'
            os.environ['LOD_AMD_PREFILL_COMPACT_BITS'] = '1' if bitset else compact_bits
            os.environ['LOD_AMD_PREFILL_PACK_EXACT_ROWS'] = '1' if packed else packed_exact_rows
            os.environ['LOD_AMD_PREFILL_MERGE_MMA_PARTS'] = '1' if variant == 'blas-mma1' else merge_mma_parts
            common['normalize_route_query'] = variant != 'top8' or engine.routing_normalization == 'query'
            buffers={}
            multiplier = 24 if variant == 'stream-c24' else 16
            os.environ['LOD_AMD_PREFILL_THRESHOLD'] = str(multiplier)
            os.environ['LOD_AMD_PREFILL_MATMUL'] = 'stream' if stream else 'triton' if top8 or variant == 'triton' else 'blas'
            os.environ['LOD_AMD_PREFILL_PROBABILITY'] = 'scaled-fp16' if variant in ('scaled-fp16', 'scaled-fp16-tc') else baseline_probability
            os.environ['LOD_AMD_PREFILL_REDUCE'] = 'tc' if variant in ('tc', 'scaled-fp16-tc') else reducer
            os.environ['LOD_AMD_PREFILL_BITMAP_RANKS'] = '1' if variant == 'bitmap' else bitmap_setting
            os.environ['LOD_AMD_PREFILL_REGION_ALIGNMENT'] = {'align64': '64', 'align256': '256'}.get(variant, alignment)
            os.environ['LOD_AMD_PREFILL_SELECT_WARPS'] = {'select1': '1', 'select2': '2', 'select4': '4', 'select16': '16'}.get(variant, select_warps)
            os.environ['LOD_AMD_PREFILL_GROUP_PACK'] = '0' if stream or variant in ('compact', 'rows2', 'rows4') else group_pack
            os.environ['LOD_AMD_PREFILL_SELECT_ROWS'] = {'rows2': '2', 'rows4': '4'}.get(variant, select_rows)
            function=aiter_prefill_route_coarse_attention if top8 else threshold_prefill_route_coarse_attention
            geometry=[engine.leaf_block_m,engine.leaf_block_n,engine.leaf_num_warps]
            def route():
                return function(q,data['k'],data['v'],data['c'],**common,buffers=buffers)
            def leaves(slots,coarse,hc,offset,*,variable=False):
                return paged_leaf_attention(q,page['leaf_k'],page['leaf_v'],page['slot_pages'],page['overflow_page_keys'],
                    page['overflow_page_values'],page['overflow_used'],page['slot_lengths'],slots,page_indices=page['page_indices'],
                    kv_group_size=group,active_slots=data['s'],scale=float(engine.scaling),block_m=geometry[0],
                    block_n=geometry[1],num_warps=geometry[2],hash_probes=engine._page_lookup_probes(page),
                    route_head_counts=hc,route_offsets=offset,route_counts=coarse.route_counts,
                    exact_row_starts=coarse.exact_row_starts,
                    persistent_workers=int(os.environ.get('LOD_AMD_PREFILL_WORKERS','608')) if not top8 or variable else 0,
                    exact_reference=coarse.lse_0 if variant=='stream-atomic' else None,
                    exact_accum_shards=int(os.environ.get('LOD_AMD_PREFILL_DIRECT_SHARDS', '1')),
                    query_major=query_major,
                    query_key_tile=int(os.environ.get('LOD_AMD_PREFILL_DIRECT_KEYS', '32')),
                    query_warps=int(os.environ.get('LOD_AMD_PREFILL_DIRECT_WARPS', '4')),
                    exact_query_group=int(os.environ.get('LOD_AMD_PREFILL_DIRECT_QUERY_GROUP', '1')),
                    exact_query_order={'stream-signature': 'signature', 'stream-largest': 'largest'}.get(variant, 'adjacent'),
                    variable_reduce_backend=os.environ.get('LOD_AMD_PREFILL_REDUCE','bucket'),
                    reduce_routes=coarse.closed_regions_only and not coarse.fused_exact_merge,buffers=buffers)
            slots,coarse,hc,offset=route()
            # Production also leaves oversized regions coarse after top-eight ranking.
            if top8:
                sizes=page['slot_lengths'][:,:,:data['s']].repeat_interleave(group,1)
                selected=torch.gather(sizes[:,:,None,:].expand(-1,-1,t,-1),-1,slots.clamp_min(0).long())
                slots=torch.where(selected<=1024,slots,-1);hc=offset=None
            exact,lse=leaves(slots,coarse,hc,offset)
            def remote():
                ss,cc,hh,oo=route()
                if top8:
                    sizes=page['slot_lengths'][:,:,:data['s']].repeat_interleave(group,1)
                    selected=torch.gather(sizes[:,:,None,:].expand(-1,-1,t,-1),-1,ss.clamp_min(0).long())
                    ss=torch.where(selected<=1024,ss,-1);hh=oo=None
                ee,ll=leaves(ss,cc,hh,oo)
                return merge_aiter_prefill_refinement(q,data['sink_k'],data['sink_v'],cc,ss,ee,ll,local,local_lse,
                    kv_group_size=group,scale=float(engine.scaling))
            output=remote()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('nonfinite remote attention')
            validation={}
            if variant == 'blas':
                reference_remote = output.clone()
                reference_variant = variant
                if 'stream' in variants:
                    reference_slots = slots.clone()
            if variant == 'stream' or (variant == 'stream-fused' and reference_remote is None):
                reference_remote = output.clone()
                reference_variant = variant
            if not top8:
                qs=q[:,:,::max(t//16,1)].contiguous();small_buffers={}
                ss,cc,*_=function(qs,data['k'],data['v'],data['c'],**common,buffers=small_buffers)
                size=data['c'][..., :data['s'],0].float().repeat_interleave(group,1)
                mk=cc.mean_k.float().repeat_interleave(group,1)
                mv=cc.mean_v.float().repeat_interleave(group,1)
                native=torch.einsum('bhtd,bhsd->bhts',qs.float(),mk)*float(engine.scaling)
                score=native/qs.float().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()+size.clamp_min(1).log()[:,:,None,:]
                probability=score.masked_fill(size[:,:,None,:]<=0,-torch.inf).softmax(-1)
                cut=multiplier/(size>0).sum(-1).clamp_min(1)[:,:,None,None]
                expected=(probability>cut)&(size[:,:,None,:]>0)&(size[:,:,None,:]<=1024)
                actual=torch.zeros_like(expected,dtype=torch.int32)
                actual.scatter_add_(-1,ss.clamp_min(0).long(),(ss>=0).int())
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
            if (bitset or (stream and variant != 'stream-c24') or variant in ('scaled-fp16', 'tc', 'scaled-fp16-tc', 'align64', 'align256',
                                      'select1', 'select2', 'select4', 'select16', 'compact', 'rows2', 'rows4',
                                      'blas-fused')) and reference_remote is not None:
                torch.testing.assert_close(output, reference_remote, atol=.02, rtol=.025)
                validation[f'max_abs_remote_difference_from_{reference_variant}'] = float(
                    (output.float() - reference_remote.float()).abs().max())
            if stream and reference_slots is not None:
                validation['different_route_rows_from_blas'] = int((slots != reference_slots).any(-1).sum())
            if variant == 'bitmap' and reference_remote is not None:
                torch.testing.assert_close(output, reference_remote, atol=0, rtol=0)
                validation['remote_bitwise_identical'] = True
            rank_storage = sum(buf.numel() * buf.element_size() for name, buf in buffers.items()
                               if name in ('threshold_prefill_mask', 'threshold_prefill_ranks',
                                           'threshold_prefill_bitmap', 'threshold_prefill_bitmap_base'))
            selected=(slots>=0)
            sizes=page['slot_lengths'].repeat_interleave(group,1)
            opened=torch.gather(sizes[:,:,None,:].expand(-1,-1,t,-1),-1,slots.clamp_min(0).long())*selected
            row=dict(layer=pool._prefill_probe_layer,variant=variant,queries=t,regions=data['s'],head_dim=d,rank_metadata_mib=rank_storage/(1<<20),
                mean_open_regions=float(selected.sum(-1).float().mean()),mean_open_entries=float(opened.sum(-1).float().mean()),
                route_coarse_ms=do_bench_cudagraph(route,rep=150),
                leaves_ms=do_bench_cudagraph(lambda:leaves(slots,coarse,hc,offset),rep=150),
                remote_attention_ms=do_bench_cudagraph(remote,rep=200),validation=validation)
            row['route_workspace_mib'] = sum(buf.numel() * buf.element_size() for key, buf in buffers.items()
                                             if key.startswith('threshold_prefill_')) / (1 << 20)
            if 'stream-fused' in variants or 'blas-fused' in variants:
                row['final_merge_ms'] = do_bench_cudagraph(
                    lambda: merge_aiter_prefill_refinement(
                        q, data['sink_k'], data['sink_v'], coarse, slots, exact, lse, local, local_lse,
                        kv_group_size=group, scale=float(engine.scaling),
                    ), rep=150)
            if stream:
                counts = coarse.route_counts
                row['frontier_over_32_fraction'] = float((counts > 32).float().mean())
                row['frontier_over_64_fraction'] = float((counts > 64).float().mean())
                row['exact_output_workspace_mib'] = sum(
                    buf.numel() * buf.element_size() for key, buf in buffers.items()
                    if key in ('leaf_route_output', 'leaf_route_lse') or key.startswith('direct_exact_')
                ) / (1 << 20)
            if variant == 'stream-atomic':
                row['repair_rows'] = int(buffers['direct_exact_repair'].sum())
            if os.environ.get('LOD_AMD_PREFILL_PROFILE_LEAVES') == '1':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    leaves(slots, coarse, hc, offset)
                    torch.cuda.synchronize()
                row['leaf_device_profile'] = [
                    dict(name=e.key, calls=e.count, self_us=e.self_device_time_total)
                    for e in profile.key_averages() if e.self_device_time_total
                ]
            if top8 and os.environ.get('LOD_AMD_PREFILL_PROBE_FIXED_FRONTIER') == '1':
                from dataclasses import replace
                import math

                live = (slots >= 0).sum(-1).to(torch.int32)
                compact = slots.masked_fill(slots < 0, int(data['s'])).sort(dim=-1).values
                compact.masked_fill_(compact == data['s'], -1)
                variable_coarse = replace(coarse, closed_regions_only=True, route_counts=live)
                threshold_width = min(data['s'], math.ceil(data['s'] / 16) + 1)
                width = max(compact.size(-1), threshold_width)
                padded = torch.full((*compact.shape[:-1], width), -1, device=q.device, dtype=compact.dtype)
                padded[..., :compact.size(-1)].copy_(compact)
                row['fixed_frontier'] = dict(
                    native_width=slots.size(-1), threshold_width=width,
                    variable_width8_ms=do_bench_cudagraph(
                        lambda: leaves(compact, variable_coarse, None, None, variable=True), rep=120),
                    variable_threshold_width_ms=do_bench_cudagraph(
                        lambda: leaves(padded, variable_coarse, None, None, variable=True), rep=120),
                )
            if os.environ.get('LOD_AMD_PREFILL_PROFILE_STAGES') == '1':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profile:
                    route()
                    torch.cuda.synchronize()
                row['route_device_profile'] = [
                    dict(name=e.key, calls=e.count, self_us=e.self_device_time_total)
                    for e in profile.key_averages() if e.self_device_time_total
                ]
            print(json.dumps(row),flush=True); result['records'].append(row)
            if not top8 and os.environ.get('LOD_AMD_PREFILL_TUNE_WORKERS')=='1':
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
            if not top8 and os.environ.get('LOD_AMD_PREFILL_TUNE_LEAVES')=='1':
                shapes = os.environ.get('LOD_AMD_PREFILL_PROBE_LEAF_SHAPES',
                                        '32,32,2;32,64,4;64,32,4;64,64,4;128,32,4;128,64,4')
                for shape in shapes.split(';'):
                    config = tuple(map(int, shape.split(',')))
                    geometry[:] = config
                    candidate = remote()
                    difference = float((candidate.float() - output.float()).abs().max())
                    try:
                        torch.testing.assert_close(candidate, output, atol=.02, rtol=.025)
                    except AssertionError as exc:
                        row = dict(layer=pool._prefill_probe_layer, variant=variant, geometry=config, valid=False,
                                   reason=str(exc), max_output_difference=difference)
                        print(json.dumps(row), flush=True)
                        result['records'].append(row)
                        continue
                    row = dict(layer=pool._prefill_probe_layer, variant=variant, geometry=config,
                        leaves_ms=do_bench_cudagraph(lambda:leaves(slots,coarse,hc,offset),rep=120),
                        remote_attention_ms=do_bench_cudagraph(remote,rep=150),
                        max_output_difference=difference)
                    print(json.dumps(row),flush=True);result['records'].append(row)
                geometry[:]=[engine.leaf_block_m,engine.leaf_block_n,engine.leaf_num_warps]
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    remote();torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by='self_device_time_total',row_limit=16),flush=True)
            del buffers,slots,coarse,exact,lse,output,opened,selected
            torch.cuda.empty_cache()
        if 'LOD_AMD_PREFILL_PROBABILITY' in os.environ:
            os.environ['LOD_AMD_PREFILL_PROBABILITY'] = baseline_probability
        if 'LOD_AMD_PREFILL_REDUCE' in os.environ:
            os.environ['LOD_AMD_PREFILL_REDUCE'] = reducer
        os.environ['LOD_AMD_PREFILL_BITMAP_RANKS'] = bitmap_setting
        os.environ['LOD_AMD_PREFILL_REGION_ALIGNMENT'] = alignment
        os.environ['LOD_AMD_PREFILL_SELECT_WARPS'] = select_warps
        os.environ['LOD_AMD_PREFILL_SELECT_ROWS'] = select_rows
        os.environ['LOD_AMD_PREFILL_GROUP_PACK'] = group_pack
        if fused_merge is None:
            os.environ.pop('LOD_AMD_PREFILL_FUSED_MERGE', None)
        else:
            os.environ['LOD_AMD_PREFILL_FUSED_MERGE'] = fused_merge
        os.environ['LOD_AMD_PREFILL_COMPACT_BITS'] = compact_bits
        os.environ['LOD_AMD_PREFILL_PACK_EXACT_ROWS'] = packed_exact_rows
        os.environ['LOD_AMD_PREFILL_MERGE_MMA_PARTS'] = merge_mma_parts
        del pool._prefill_probe
    result['stage']='complete'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--length',type=int,default=65536)
    parser.add_argument('--checkpoint',default='Qwen/Qwen3.8-27B-FP8')
    args=parser.parse_args()
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
    from vllm import LLM,SamplingParams
    from transformers import AutoTokenizer
    from benchmarks._vllm import llm_kwargs,close_llm
    from benchmarks.prolong import make_speed_prompts
    checkpoint=args.checkpoint
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
        result=result[0];result.update(context=args.length,checkpoint=checkpoint,prompt_metadata=metadata)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:close_llm(llm)


if __name__=='__main__':main()
