"""Time complete attention calls on real AMD model queries and identical KV states.

Capture queries during a short vLLM generation, then benchmark fixed-cache
attention with identical local-length rollback. These are kernel diagnostics,
not full-model speed or amortized cache-update measurements.
"""
import argparse
import json
import os
from pathlib import Path
import types


def retain_completed_probe(model):
    """Keep the sole diagnostic request's cache until the worker probe finishes.

    This changes only request cleanup, after attention has run. Speed runs use
    normal cleanup and never install this hook. No second request is submitted.
    """
    seen=set()
    for module in model.modules():
        pool=getattr(module,'_vllm_lod_pool',None)
        if pool is None or id(pool) in seen:continue
        seen.add(id(pool));pool._amd_probe_original_reset=pool.reset
        def retained_reset(self,slot):
            if self.ready[slot] and hasattr(self,'_amd_threshold_probe'):
                self._amd_probe_retained=True
                return
            return self._amd_probe_original_reset(slot)
        pool.reset=types.MethodType(retained_reset,pool)
    if not seen:raise RuntimeError('no initialized LoD pools to retain')
    return len(seen)


def probe_model(model):
    import statistics
    import torch
    from benchmarks.amd_threshold_decode import selected_region_mask
    from benchmarks._identity import benchmark_identity
    torch.set_grad_enabled(False);torch.cuda.synchronize()
    old_page_function=None
    old_source=os.environ.get('LOD_AMD_THRESHOLD_PROBE_OLD_PAGE_SOURCE')
    if old_source:
        import importlib.util,sys
        spec=importlib.util.spec_from_file_location('lod_attention.kernels._amd_threshold_old_pages',old_source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        old_page_function=module.query_major_indexed_residual_page_attention
    os.environ.pop('LOD_AMD_THRESHOLD_CAPTURE_QUERY',None)
    pools=[];seen=set()
    for name,module in model.named_modules():
        pool=getattr(module,'_vllm_lod_pool',None)
        if pool is not None and id(pool) not in seen:
            seen.add(id(pool));pools.append((name,pool))
    if not pools:raise RuntimeError('no LoD model layers')
    records=[]
    for index in sorted({0,len(pools)//2,len(pools)-1}):
        name,pool=pools[index]
        if pool.settings.levels == 3:
            from benchmarks.amd_threshold_three_tier import threshold_attention
        else:
            from benchmarks.amd_threshold_decode import threshold_attention
        if not getattr(pool,'_amd_probe_retained',False):raise RuntimeError('diagnostic request cache was not retained')
        if not hasattr(pool,'_amd_threshold_probe'):raise RuntimeError('no captured real decode inputs')
        q,k,v,local,state=pool._amd_threshold_probe
        if int(state.item())<1 or int(local.item())<1 or not bool(torch.isfinite(q).all().item()) or float(q.abs().max().item())==0:
            raise RuntimeError('captured inputs are empty profiling inputs')
        old_local=pool.local_lens.clone();old_state=pool.state_lens.clone()
        pool.local_lens.copy_(local);pool.state_lens.copy_(state)
        if int(pool.state['page_cache']['slot_lengths'].sum().item())==0:raise RuntimeError('captured model cache was cleared')
        out=torch.empty_like(q);configs=[('legacy',64,64,32,4),('packed',64,64,32,4)]
        configs += [(impl,64,splits,block,4) for impl in ('indexed','combined') for splits in (16,32,64) for block in (16,32)]
        configs += [(impl,64,splits,32,2) for impl in ('indexed','combined') for splits in (32,64)]
        if 'LOD_AMD_THRESHOLD_PROBE_CONFIGS' in os.environ:configs=json.loads(os.environ['LOD_AMD_THRESHOLD_PROBE_CONFIGS'])
        for rule in os.environ.get('LOD_AMD_THRESHOLD_PROBE_RULES','region,entry').split(','):
            expected=None;expected_selection=None
            for impl,tile,splits,block,warps in configs:
                os.environ['LOD_AMD_THRESHOLD_IMPLEMENTATION']=impl
                os.environ['LOD_AMD_THRESHOLD_ENTRY_BLOCK']=str(block)
                os.environ['LOD_AMD_THRESHOLD_ENTRY_WARPS']=str(warps)
                if pool.settings.levels == 3:
                    os.environ['LOD_AMD_THRESHOLD_PAGE_WORKERS']=str(splits)
                    os.environ['LOD_AMD_THRESHOLD_PAGE_WARPS']=str(warps)
                def invoke():
                    threshold_attention(pool,q,k,v,out,16.,rule=rule,tile=tile,splits=splits)
                    pool.local_lens.sub_(1)
                invoke();torch.cuda.synchronize()
                current=out.clone();scratch=pool._threshold_scratch
                encoded=selected_region_mask(pool).clone();count=encoded.sum(-1)
                if expected is None:expected=current;expected_selection=encoded
                if not torch.equal(encoded,expected_selection):raise RuntimeError('selected regions differ from legacy')
                delta=(current.float()-expected.float()).abs();maximum=float(delta.max().item())
                if not bool(torch.isfinite(delta).all().item()) or not bool(torch.isclose(current.float(),expected.float(),rtol=0.008,atol=0.004).all().item()):raise RuntimeError(f'attention differs from legacy: {maximum}')
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):invoke()
                for _ in range(4):graph.replay()
                torch.cuda.synchronize();measurements=[]
                for _ in range(3):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                    for _ in range(32):graph.replay()
                    end.record();torch.cuda.synchronize();measurements.append(begin.elapsed_time(end)*1000/(32*8))
                sizes=pool.state['page_cache']['slot_lengths'][0].repeat_interleave(q.size(2)//pool.kv_heads,0)
                opened=sizes*encoded
                row=dict(layer=index,layer_name=name,rule=rule,implementation=impl,tile=tile,splits=splits,
                         entry_block=block,entry_warps=warps,microseconds=statistics.median(measurements),
                         repetitions_us=measurements,max_abs_vs_legacy=maximum,selection_mismatches=0,
                         mean_opened_regions=float(count.float().mean().item()),mean_opened_entries=float(opened.sum(-1).float().mean().item()),
                         mean_union_entries_per_kv=float((sizes.reshape(pool.kv_heads,-1,pool.state_capacity)[:,0,:]*encoded.reshape(pool.kv_heads,-1,pool.state_capacity).any(1)).sum(-1).float().mean().item()),
                         local_entries=int(local.item()),active_regions=int(state.item()),state_capacity=pool.state_capacity,
                         head_dim=q.size(-1),query_heads=q.size(2),kv_heads=pool.kv_heads)
                if pool.settings.levels == 3:
                    row['page_workers']=splits;row['page_warps']=warps
                    row['max_abs_vs_reference_config']=row.pop('max_abs_vs_legacy')
                    row['mean_selected_region_entries']=row.pop('mean_opened_entries')
                    row['mean_selected_region_union_entries_per_kv']=row.pop('mean_union_entries_per_kv')
                    row['mean_exact_remote_entry_bound']=32*row['mean_opened_regions']
                records.append(row);print(json.dumps(row),flush=True);del graph
        if os.environ.get('LOD_AMD_THRESHOLD_PROBE_NATIVE')=='1':
            os.environ.pop('LOD_AMD_THRESHOLD_MULTIPLIER',None)
            def native():
                pool.decode(q.reshape(1,q.size(2),q.size(3)),k.reshape(1,pool.kv_heads,q.size(3)),
                            v.reshape(1,pool.kv_heads,q.size(3)),types.SimpleNamespace(num_actual_tokens=1),
                            out.reshape(1,q.size(2),q.size(3)))
                pool.local_lens.sub_(1)
            native();torch.cuda.synchronize()
            old_comparison={}
            if old_page_function is not None:
                import importlib
                page_module=importlib.import_module('lod_attention.kernels.paged_decode')
                current=out.clone();current_function=page_module.query_major_indexed_residual_page_attention
                try:
                    page_module.query_major_indexed_residual_page_attention=old_page_function
                    native();torch.cuda.synchronize()
                    difference=(out.float()-current.float()).abs()
                    old_comparison=dict(pre_change_page_output_identical=torch.equal(out,current),
                                        max_abs_vs_pre_change_pages=float(difference.max().item()))
                    if not bool(torch.isclose(out.float(),current.float(),rtol=.008,atol=.004).all().item()):
                        raise RuntimeError(f'native page extension changed attention: {old_comparison}')
                finally:
                    page_module.query_major_indexed_residual_page_attention=current_function
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(8):native()
            for _ in range(4):graph.replay()
            torch.cuda.synchronize();measurements=[]
            for _ in range(3):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
                for _ in range(32):graph.replay()
                end.record();torch.cuda.synchronize();measurements.append(begin.elapsed_time(end)*1000/256)
            row=dict(layer=index,implementation='native-top8',microseconds=statistics.median(measurements),repetitions_us=measurements,**old_comparison)
            records.append(row);print(json.dumps(row),flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                graph.replay();torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by='self_device_time_total',row_limit=18),flush=True)
            os.environ['LOD_AMD_THRESHOLD_MULTIPLIER']='16'
            del graph
        snapshot_dir=os.environ.get('LOD_AMD_THRESHOLD_SNAPSHOT_DIR')
        if snapshot_dir:
            names=('state_k','state_v','counts','recent_k','recent_v','sink_k','sink_v')
            page_names=('slot_lengths','slot_pages','overflow_page_values','page_indices','leaf_k','leaf_v')
            state_cpu={key:pool.state[key].detach().cpu() for key in names}
            state_cpu['page_cache']={key:pool.state['page_cache'][key].detach().cpu() for key in page_names}
            state_cpu['page_cache']['paged_page_directory']=True
            data=dict(state=state_cpu,q=q.cpu(),k=k.cpu(),v=v.cpu(),local_lens=local.cpu(),state_lens=state.cpu(),
                      kv_heads=pool.kv_heads,state_capacity=pool.state_capacity,decode_local_limit=pool.decode_local_limit,
                      scaling=float(pool.engine.scaling),layer=index,layer_name=name)
            destination=Path(snapshot_dir)/f'layer-{index:02d}.pt';destination.parent.mkdir(parents=True,exist_ok=True)
            torch.save(data,destination);print(f'Real decode snapshot saved: {destination}',flush=True)
        pool.local_lens.copy_(old_local);pool.state_lens.copy_(old_state)
    return dict(stage='complete',benchmark_identity=benchmark_identity(),hardware=str(torch.cuda.get_device_properties(0)),
                scope='fixed real query/cache attention plus same local-length rollback; excludes model and cache catch-up',records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('qwen','k2'),required=True)
    p.add_argument('--mode',choices=('two-tier','three-tier-bf16','three-tier-int4'),default='two-tier')
    p.add_argument('--length',type=int,default=65536);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1';os.environ['LOD_AMD_THRESHOLD_MULTIPLIER']='16'
    os.environ['LOD_AMD_THRESHOLD_CAPTURE_QUERY']='1'
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from benchmarks._vllm import llm_kwargs,default_gpu_memory_utilization,close_llm
    from benchmarks.prolong import make_speed_prompts
    checkpoint={'qwen':'Qwen/Qwen3.8-27B-FP8','k2':'IFM/K2-Horizon-32B-FP8'}[a.model]
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,trust_remote_code=True)
    prompts,metadata=make_speed_prompts(tokenizer,length=a.length,batch_size=1)
    kwargs=llm_kwargs(checkpoint=checkpoint,mode=a.mode,max_model_len=a.length+1025+16,batch_size=1,tensor_parallel_size=1,
                      gpu_memory_utilization=default_gpu_memory_utilization(checkpoint,a.mode),full_attention_backend='ROCM_AITER_UNIFIED_ATTN')
    llm=LLM(**kwargs)
    try:
        llm.apply_model(retain_completed_probe)
        llm.generate(prompts,SamplingParams(temperature=0,seed=0,max_tokens=3,ignore_eos=True),use_tqdm=False)
        results=llm.apply_model(probe_model)
        if len(results)!=1:raise RuntimeError('expected a single worker')
        result=results[0];result.update(model=a.model,checkpoint=checkpoint,mode=a.mode,context=a.length,prompt_metadata=metadata)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    finally:close_llm(llm)


if __name__=='__main__':main()
