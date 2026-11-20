"""Tune attention using frozen real-model queries and semantic KV snapshots.

The model itself and cache catch-up are excluded. Every measured invocation
includes the complete attention pipeline and identical local-length rollback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import types


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--configs',type=Path,required=True)
    p.add_argument('--rules',nargs='+',default=['region']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from torch.utils._pytree import tree_map
    from benchmarks.amd_threshold_decode import threshold_attention,selected_region_mask
    from benchmarks._identity import benchmark_identity
    torch.set_grad_enabled(False);configs=json.loads(a.configs.read_text());records=[];inputs=[];skipped=[]
    for path in sorted(a.snapshots.glob('layer-*.pt')):
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        d=tree_map(lambda t:t.cuda() if isinstance(t,torch.Tensor) else t,torch.load(path,weights_only=True))
        pool=types.SimpleNamespace(state=d['state'],settings=types.SimpleNamespace(levels=2),dtype=torch.bfloat16,
            kv_heads=d['kv_heads'],state_capacity=d['state_capacity'],decode_local_limit=d['decode_local_limit'],
            state_lens=d['state_lens'],local_lens=d['local_lens'],device='cuda',engine=types.SimpleNamespace(scaling=d['scaling']))
        q,k,v=d['q'],d['k'],d['v'];out=torch.empty_like(q)
        for rule in a.rules:
            expected=None;selection=None
            for config in configs:
                impl,tile,splits,block,warps=config[:5]
                os.environ['LOD_AMD_THRESHOLD_ACTIVE_WORKERS']='0';os.environ['LOD_AMD_THRESHOLD_MERGE_BLOCK']='0';os.environ['LOD_AMD_THRESHOLD_MERGE_WARPS']='4'
                extra=config[5] if len(config)>5 else {};os.environ.update({k:str(v) for k,v in extra.items()})
                os.environ['LOD_AMD_THRESHOLD_IMPLEMENTATION']=impl
                os.environ['LOD_AMD_THRESHOLD_ENTRY_BLOCK']=str(block);os.environ['LOD_AMD_THRESHOLD_ENTRY_WARPS']=str(warps)
                def invoke():
                    threshold_attention(pool,q,k,v,out,16.,rule=rule,tile=tile,splits=splits)
                    pool.local_lens.sub_(1)
                from triton.runtime.errors import OutOfResources
                before=pool.local_lens.clone()
                try:invoke();torch.cuda.synchronize()
                except OutOfResources as error:
                    pool.local_lens.copy_(before);torch.cuda.synchronize()
                    skipped.append(dict(layer=d['layer'],configuration=config,reason=str(error)))
                    print(json.dumps(skipped[-1]),flush=True)
                    if expected is None:raise
                    continue
                current=out.clone();b=pool._threshold_scratch
                encoded=selected_region_mask(pool).clone()
                if expected is None:
                    if impl!='legacy':raise ValueError('first configuration must be legacy reference')
                    expected=current;selection=encoded
                if not torch.equal(encoded,selection):raise RuntimeError(f'changed selected regions: {config}')
                error=float((current.float()-expected.float()).abs().max().item())
                close=torch.isclose(current.float(),expected.float(),rtol=0.008,atol=0.004)
                if not bool(torch.isfinite(current).all().item()) or not bool(close.all().item()):raise RuntimeError(f'changed attention: {config}, {error}')
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):invoke()
                for _ in range(4):graph.replay()
                torch.cuda.synchronize();times=[]
                for _ in range(3):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                    for _ in range(24):graph.replay()
                    end.record();torch.cuda.synchronize();times.append(start.elapsed_time(end)*1000/192)
                row=dict(layer=d['layer'],layer_name=d['layer_name'],rule=rule,implementation=impl,tile=tile,splits=splits,
                         entry_block=block,entry_warps=warps,microseconds=statistics.median(times),repetitions_us=times,
                         selection_mismatches=0,max_abs_vs_legacy=error,extra_settings=extra)
                if impl=='cached':row['kernel_resources']=getattr(pool,'_threshold_kernel_resources',None)
                if os.environ.get('LOD_AMD_THRESHOLD_REPLAY_PROFILE')=='1':
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        graph.replay();torch.cuda.synchronize()
                    row['profile']=[dict(name=e.key,device_us=e.self_device_time_total/8,calls=e.count/8)
                                    for e in prof.key_averages() if e.self_device_time_total>0]
                records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='profile'}),flush=True);del graph
    if not records:raise RuntimeError('no snapshots or configurations')
    a.output.write_text(json.dumps(dict(stage='complete',scope=__doc__,inputs=inputs,records=records,skipped=skipped,
        hardware=str(torch.cuda.get_device_properties(0)),benchmark_identity=benchmark_identity()),indent=2)+'\n')


if __name__=='__main__':main()
