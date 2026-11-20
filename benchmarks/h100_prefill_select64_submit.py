"""Submit uncapped 64K threshold optimization through uv and cluster-run.

The exact 16K initial chunk, c=16 routing, local/sink fields, and cache updates
are fixed. All GPU jobs use immutable source snapshots and a serial GPU lock.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('checks','probe','producer','numerics','recompute-probe','merge-check','reduce-check','models','profiles','quality','verify'),required=True)
    p.add_argument('--label',required=True);p.add_argument('--checks-label',default='v1')
    p.add_argument('--family',default='threshold-select64')
    p.add_argument('--variants',nargs='+',default=['control','scan','sparse','fused','combined'])
    p.add_argument('--m',type=int,default=32);p.add_argument('--n',type=int,default=256)
    p.add_argument('--lengths',nargs='+',type=int,default=[65536])
    p.add_argument('--merge-m',type=int,default=4);p.add_argument('--merge-r',type=int,default=16)
    p.add_argument('--producer-m',type=int,default=64);p.add_argument('--producer-n',type=int,default=64)
    p.add_argument('--producer-warps',type=int,default=4)
    p.add_argument('--producer-stages',type=int,default=2)
    p.add_argument('--closed-blas',action='store_true')
    p.add_argument('--closed-m',type=int,default=0)
    p.add_argument('--closed-precision',choices=('bf16','split-bf16','scaled-bf16','scaled-fp16','offset-bf16','offset-fp16'),default='bf16')
    p.add_argument('--value-center',action='store_true')
    p.add_argument('--closed-arithmetic',choices=('direct','online64'),default='direct')
    p.add_argument('--quality-tasks',nargs='+',default=['niah_multikey_2','niah_multikey_3','cwe','qa_1'])
    p.add_argument('--quality-limit',type=int,default=2)
    p.add_argument('--shared-log',action='store_true')
    p.add_argument('--reduce',choices=('wide','grouped','bucketed'),default='wide');p.add_argument('--packed-output',action='store_true')
    p.add_argument('--reduce-check-label')
    p.add_argument('--recompute-m',type=int,default=64);p.add_argument('--recompute-n',type=int,default=64)
    p.add_argument('--individual-expert-atomics',action='store_true')
    p.add_argument('--probe-min-regions',type=int,default=2500)
    a=p.parse_args();repo=Path('/home/ubuntu/lod-h100/code');r=Path('/home/ubuntu/comparison/results')
    state=Path.home()/'.local/state/lod-cluster-run';uv=str(Path.home()/'.local/bin/uv');family=a.family
    cpu=[uv,'run','--offline','--no-project','--python','/usr/bin/python3','python','-m']
    gpu=[uv,'run','--offline','--no-project','--python','/home/ubuntu/comparison/.venv/bin/python','--with','triton==3.7.1','python','-m']
    env=dict(CUDA_HOME='/usr/local/cuda-12.4',LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64',HF_HUB_OFFLINE='1',
        PYTHONPATH='/home/ubuntu/comparison/RetroInfer:.',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',LOD_TRACK_UPDATES='1',
        LOD_RETRO_TWO_TIER='1',LOD_H100_THRESHOLD_MULTIPLIER='16',LOD_H100_THRESHOLD_RULE='region',
        LOD_H100_THRESHOLD_IMPLEMENTATION='materialized',LOD_H100_THRESHOLD_SPLITS='32',LOD_AMD_THRESHOLD_ENTRY_BLOCK='128',
        LOD_AMD_THRESHOLD_ENTRY_WARPS='4',LOD_AMD_THRESHOLD_MERGE_BLOCK='32',LOD_AMD_THRESHOLD_PRECISE='1',
        LOD_H100_PREFILL_RULE='region',LOD_H100_PREFILL_MULTIPLIER='16',LOD_H100_PREFILL_IMPLEMENTATION='fast',
        LOD_H100_CLOSED_REDUCE=a.reduce,LOD_H100_CLOSED_PACKED_OUTPUT=str(int(a.packed_output)),
        LOD_H100_CLOSED_NORMALIZE_M=str(a.closed_m),LOD_H100_CLOSED_LEAF_M='64',LOD_H100_CLOSED_LEAF_N='32',LOD_H100_CLOSED_LEAF_WARPS='4',
        LOD_H100_FAST_LEAF='native',LOD_H100_FAST_SAVED_CORRECTION='1',LOD_H100_FAST_FULL_SELECT='1',LOD_H100_FAST_TILE='4096',
        LOD_H100_FAST_SCAN_SELECT='0',LOD_H100_FAST_SPARSE_PACK='0',LOD_H100_FAST_BITMAP_PACK='0',LOD_H100_FAST_FUSED_MERGE='0',
        LOD_H100_FAST_PREFIX_SELECT='0',LOD_H100_FAST_PREFIX_FULL='0',LOD_H100_FAST_FUSED_BUCKET='0',LOD_H100_FAST_PREFIX_CLOSED='0',
        LOD_H100_FAST_CLOSED_BLAS=str(int(a.closed_blas)),
        LOD_H100_CLOSED_PRECISION=a.closed_precision,
        LOD_H100_CLOSED_ARITHMETIC=a.closed_arithmetic,
        LOD_H100_CLOSED_VALUE_CENTER=str(int(a.value_center)),
        LOD_H100_FAST_SHARED_LOG=str(int(a.shared_log)),
        LOD_H100_FAST_RECOMPUTE_CLOSED='0',
        LOD_H100_RECOMPUTE_M=str(a.recompute_m),LOD_H100_RECOMPUTE_N=str(a.recompute_n),
        LOD_H100_RECOMPUTE_AGGREGATE=str(int(not a.individual_expert_atomics)),
        LOD_H100_SCAN_M=str(a.m),LOD_H100_SCAN_N=str(a.n),LOD_H100_SCAN_WARPS='4',LOD_H100_SCAN_MIN_REGIONS='0',
        LOD_H100_FUSED_MERGE_M=str(a.merge_m),LOD_H100_FUSED_MERGE_R=str(a.merge_r),RETRO_PREFILL_MLP_CHUNK='8192')
    required=[];jobs=[]
    if a.stage!='checks':
        required=[j['id'] for j in json.loads((r/f'{family}-{a.checks_label}-checks-runs.json').read_text())['jobs']]
    if a.reduce=='bucketed' and a.stage not in ('checks','reduce-check'):
        if not a.reduce_check_label:p.error('--reduce bucketed requires --reduce-check-label')
        required += [j['id']for j in json.loads((r/f'{family}-{a.reduce_check_label}-reduce-check-runs.json').read_text())['jobs']]
    def variant(name):
        tuned=name.endswith('-tuned');name=name.removesuffix('-tuned')
        if name not in ('control','scan','sparse','fused','combined','dense','top8','bitmap','bitmap-fused','prefix','prefix-fused','bucket','bitmap-bucket','prefix-bucket','prefix-full','prefix-full-fused','prefix-closed','recompute','auto'):raise ValueError(name)
        opts=dict(LOD_H100_FAST_SCAN_SELECT=str(int(name in ('scan','sparse','combined','bitmap','bitmap-fused','bitmap-bucket'))),
            LOD_H100_FAST_RECOMPUTE_CLOSED=str(int(name=='recompute')),
            LOD_H100_FAST_SPARSE_PACK=str(int(name in ('sparse','combined'))),
            LOD_H100_FAST_BITMAP_PACK=str(int(name in ('bitmap','bitmap-fused','bitmap-bucket'))),
            LOD_H100_FAST_PREFIX_SELECT=str(int(name in ('prefix','prefix-fused','prefix-bucket','prefix-full','prefix-full-fused','prefix-closed'))),
            LOD_H100_FAST_PREFIX_CLOSED=str(int(name=='prefix-closed')),
            LOD_H100_FAST_PREFIX_FULL=str(int(name in ('prefix-full','prefix-full-fused'))),
            LOD_H100_FAST_FUSED_BUCKET=str(int(name in ('bucket','bitmap-bucket','prefix-bucket'))),
            LOD_H100_FAST_FUSED_MERGE=str(int(name in ('fused','combined','bitmap-fused','prefix-fused','bucket','bitmap-bucket','prefix-bucket','prefix-full-fused'))))
        if name in ('dense','top8'):
            opts['LOD_H100_PREFILL_RULE']='';opts['LOD_H100_PREFILL_MULTIPLIER']=''
        if name in ('control','dense','top8'):
            opts.update(LOD_H100_CLOSED_REDUCE='wide',LOD_H100_CLOSED_PACKED_OUTPUT='0')
        if name=='auto':
            opts.update(LOD_H100_FAST_SCAN_SELECT='auto',LOD_H100_SCAN_MIN_REGIONS='4096',LOD_H100_FAST_PREFIX_SELECT='auto')
        if tuned:
            opts.update(LOD_H100_FAST_PRODUCE_M=str(a.producer_m),LOD_H100_FAST_PRODUCE_N=str(a.producer_n),
                LOD_H100_FAST_PRODUCE_WARPS=str(a.producer_warps),LOD_H100_FAST_PRODUCE_STAGES=str(a.producer_stages))
        return opts
    def submit(tag,module,args,opts={},deps=None):
        name=f'{family}-{a.label}-{tag}';out=r/(name+'.json')
        cmd=['env',*(f'{k}={v}' for k,v in (env|opts).items()),*gpu,'benchmarks.h100_checked_run',
            *[v for rid in (required if deps is None else deps) for v in ('--require-run',str(state/rid/'run.json'))],
            '--','python','-m',module,*map(str,args),'--output',str(out)]
        response=subprocess.check_output(cpu+['benchmarks.h100_cluster_run','--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=repo,text=True)
        rid=response.splitlines()[0].split()[-1];jobs.append(dict(id=rid,name=name,command=cmd,output=str(out)))
        print(response,flush=True)
    if a.stage=='checks':
        cases=(('sparse',16,128,2),('sparse',32,256,2),('fused',32,256,2),('combined',32,256,2),('combined',32,256,512))
        if 'bitmap' in a.variants:
            cases=(('bitmap',16,128,2),('bitmap',32,256,2),('bitmap-fused',16,128,2),('bitmap-fused',32,256,512))
        if 'prefix' in a.variants:
            cases=(('prefix',32,256,2),('prefix',32,256,512),('bucket',32,256,2),('prefix-bucket',32,256,2))
        if 'prefix-full' in a.variants:
            cases=(('prefix-full',32,256,2),('prefix-full',32,256,512),('prefix-full-fused',32,256,2))
        if any(v.startswith('prefix-closed')for v in a.variants):
            name=next(v for v in a.variants if v.startswith('prefix-closed'))
            cases=((name,32,256,2),(name,32,256,512))
        if any(v.startswith('recompute')for v in a.variants):
            name=next(v for v in a.variants if v.startswith('recompute'))
            cases=((name,32,256,2),(name,32,256,512))
        for name,m,n,c in cases:
            submit(f'{name}-m{m}-n{n}-c{c}','benchmarks.h100_threshold_fast_check',
                ['--rule','region','--regions',513,'--queries',193,'--tile',193,'--stress-layout','--multiplier',c],
                variant(name)|dict(LOD_H100_SCAN_M=str(m),LOD_H100_SCAN_N=str(n),LOD_H100_FAST_VERIFY_QUERIES='193' if any(v.startswith(('prefix','recompute'))for v in a.variants) else '16'),deps=[])
    elif a.stage=='probe':
        submit('live-65536','benchmarks.h100_prefill_select_probe',
            ['--length',65536,'--min-regions',a.probe_min_regions,'--backends','scan','row',*(['bitmap'] if 'bitmap' in a.variants else ['sparse']),
             '--manifest',r/'threshold-next-scaling-prompts.json','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()],
            variant('scan'))
    elif a.stage=='producer':
        submit('producer-65536','benchmarks.h100_prefill_producer_probe',
            ['--manifest',r/'threshold-next-scaling-prompts.json','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()],
            variant(next((v for v in a.variants if v in ('prefix-closed','prefix')),'scan')))
    elif a.stage=='numerics':
        submit('coarse-numerics-65536','benchmarks.h100_prefill_closed_numerics',
            ['--manifest',r/'threshold-next-scaling-prompts.json','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()],
            variant('prefix-closed-tuned'))
    elif a.stage=='merge-check':
        submit('merge-check','benchmarks.h100_prefill_merge_check',[],variant('fused'))
    elif a.stage=='reduce-check':
        submit('reduce-check','benchmarks.h100_prefill_reduce_check',[],variant('prefix-closed-tuned'))
    elif a.stage=='recompute-probe':
        submit('recompute-producer-65536','benchmarks.h100_prefill_recompute_probe',
            ['--manifest',r/'threshold-next-scaling-prompts.json','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()],
            variant('recompute-tuned'))
    elif a.stage in ('models','profiles','verify'):
        for length in a.lengths:
            for name in a.variants:
                opts=variant(name)
                if a.stage=='profiles':opts['LOD_H100_PREFILL_PROFILE']='1'
                if a.stage=='verify':opts['LOD_H100_FAST_VERIFY']='1'
                submit(f'{name}-{length}-{a.stage}','benchmarks.h100_comparison',
                    ['--method','retro-full' if name=='dense' else 'retro-lod-bf16',
                     '--native-graph' if name=='dense' else '--lod-graph',
                     '--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                     '--manifest',r/'threshold-next-scaling-prompts.json','--length',length,'--tokens',2,
                     '--decode-warmup-steps',0,'--warmups',0 if a.stage=='verify' else 1,
                     '--repeats',3 if a.stage=='models' else 1,
                     '--capture-token-ids','--replay-prompt-tokens',
                     *(['--finish-lod-prefill-cache'] if name!='dense' else []),
                     *(['--check-graph-same-cache'] if a.stage=='verify' else [])],opts)
    else:
        for name in a.variants:
            submit(f'{name}-ruler','benchmarks.retro_ruler_run',
                ['--input',r/'quality-ruler-32k-frozen-13x32.json','--mode','two-tier',
                 '--tasks',*a.quality_tasks,'--limit-per-task',a.quality_limit,'--check-graph'],variant(name))
    (r/f'{family}-{a.label}-{a.stage}-runs.json').write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')


if __name__=='__main__':main()
