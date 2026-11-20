"""Queue selection-packing checks, live-score tuning, and matched model timings.

All GPU experiments use uv and cluster-run immutable source snapshots. Initial
exact prefill, c=16 routing, eligible region sizes and cache updates are fixed.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('checks','probe','models','profiles','quality','verify'),required=True)
    p.add_argument('--label',default='v1');p.add_argument('--checks-label',default='v1')
    p.add_argument('--m',type=int,default=32);p.add_argument('--n',type=int,default=128)
    p.add_argument('--backend',choices=('scan','bias','hybrid'),default='scan');p.add_argument('--warps',type=int,default=8)
    p.add_argument('--min-regions',type=int,default=0)
    p.add_argument('--lengths',type=int,nargs='+',default=[32768,128000])
    a=p.parse_args();root=Path('/home/ubuntu/lod-h100/code');r=Path('/home/ubuntu/comparison/results');state=Path.home()/'.local/state/lod-cluster-run'
    cpu=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m']
    uv=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/home/ubuntu/comparison/.venv/bin/python','--with','triton==3.7.1','python','-m']
    env=dict(CUDA_HOME='/usr/local/cuda-12.4',LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64',HF_HUB_OFFLINE='1',
        PYTHONPATH='/home/ubuntu/comparison/RetroInfer:.',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',LOD_TRACK_UPDATES='1',
        LOD_RETRO_TWO_TIER='1',LOD_H100_THRESHOLD_MULTIPLIER='16',LOD_H100_THRESHOLD_RULE='region',
        LOD_H100_THRESHOLD_IMPLEMENTATION='materialized',LOD_H100_THRESHOLD_SPLITS='32',LOD_AMD_THRESHOLD_ENTRY_BLOCK='128',
        LOD_AMD_THRESHOLD_ENTRY_WARPS='4',LOD_AMD_THRESHOLD_MERGE_BLOCK='32',LOD_AMD_THRESHOLD_PRECISE='1',
        LOD_H100_PREFILL_RULE='region',LOD_H100_PREFILL_MULTIPLIER='16',LOD_H100_PREFILL_IMPLEMENTATION='fast',
        LOD_H100_CLOSED_REDUCE='wide',LOD_H100_CLOSED_LEAF_M='64',LOD_H100_CLOSED_LEAF_N='32',LOD_H100_CLOSED_LEAF_WARPS='4',
        LOD_H100_FAST_LEAF='native',LOD_H100_FAST_SAVED_CORRECTION='1',LOD_H100_FAST_FULL_SELECT='1',LOD_H100_FAST_TILE='4096',
        LOD_H100_FAST_SCAN_SELECT=str(int(a.backend in ('scan','hybrid'))),LOD_H100_FAST_BIASED_SELECT=str(int(a.backend=='bias')),
        LOD_H100_SCAN_ATOMIC_EXPERTS=str(int(a.backend=='hybrid')),
        LOD_H100_SCAN_WARPS=str(a.warps),LOD_H100_SCAN_MIN_REGIONS=str(a.min_regions),
        LOD_H100_BIAS_WARPS=str(a.warps),LOD_H100_SCAN_M=str(a.m),LOD_H100_SCAN_N=str(a.n),RETRO_PREFILL_MLP_CHUNK='8192')
    jobs=[]
    def submit(tag,module,args,opts={},required=()):
        name=f'threshold-next-selectpack-{a.label}-{tag}';output=r/(name+'.json')
        cmd=['env',*(f'{k}={v}' for k,v in (env|opts).items()),*uv,'benchmarks.h100_checked_run',
             *[v for rid in required for v in ('--require-run',str(state/rid/'run.json'))],
             '--','python','-m',module,*map(str,args),'--output',str(output)]
        response=subprocess.check_output(cpu+['benchmarks.h100_cluster_run','--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True)
        rid=response.splitlines()[0].split()[-1];jobs.append(dict(id=rid,name=name,output=str(output),command=cmd));print(response,flush=True)
    required=[]
    if a.stage!='checks':
        previous=json.loads((r/f'threshold-next-selectpack-{a.checks_label}-checks-runs.json').read_text())
        required=[j['id'] for j in previous['jobs']]
    if a.stage=='checks':
        cases=[(m,n,2) for m,n in ((8,128),(16,128),(32,128),(64,128),(16,256),(32,256))]+[(32,128,512)]
        if a.backend=='bias':cases=[(1,w,2) for w in (4,8,16)]+[(1,8,512)]
        if a.backend=='hybrid':cases=[(16,128,2),(32,128,2),(16,128,512)]
        for m,n,c in cases:
            submit(f'check-m{m}-n{n}-c{c}','benchmarks.h100_threshold_fast_check',
                ['--rule','region','--regions',513,'--queries',193,'--tile',193,'--stress-layout','--multiplier',c],
                dict(LOD_H100_SCAN_M=str(m),LOD_H100_SCAN_N=str(n),LOD_H100_SCAN_WARPS='4',
                     LOD_H100_BIAS_WARPS=str(n if a.backend=='bias' else a.warps)))
    elif a.stage=='probe':
        submit('live-probe-128k','benchmarks.h100_prefill_select_probe',
            ['--length',128000,'--min-regions',5000,'--manifest',r/'threshold-next-scaling-prompts.json',
             '--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()],
             dict(LOD_H100_FAST_BIASED_SELECT='0',LOD_H100_FAST_SCAN_SELECT='1',LOD_H100_SCAN_ATOMIC_EXPERTS='0'),required)
    elif a.stage=='quality':
        for backend in ('control',a.backend):
            opts=dict(LOD_H100_FAST_SCAN_SELECT=str(int(backend in ('scan','hybrid'))),
                LOD_H100_FAST_BIASED_SELECT=str(int(backend=='bias')),LOD_H100_SCAN_ATOMIC_EXPERTS=str(int(backend=='hybrid')))
            submit(f'{backend}-ruler-check','benchmarks.retro_ruler_run',
                ['--input',r/'quality-ruler-32k-frozen-13x32.json','--mode','two-tier',
                 '--tasks','niah_multikey_2','niah_multikey_3','cwe','qa_1','--limit-per-task',2,'--check-graph'],opts,required)
    elif a.stage=='verify':
        for key in list(env):
            if key=='LOD_H100_FAST_SCAN_SELECT' or key.startswith('LOD_H100_SCAN_'):del env[key]
        submit('automatic-128000-verify','benchmarks.h100_comparison',
            ['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
             '--manifest',r/'threshold-next-scaling-prompts.json','--length',128000,'--tokens',2,
             '--decode-warmup-steps',0,'--warmups',0,'--repeats',1,'--capture-token-ids','--replay-prompt-tokens',
             '--finish-lod-prefill-cache','--check-graph-same-cache'],dict(LOD_H100_FAST_VERIFY='1'),required)
    else:
        for length in (a.lengths if a.stage=='models' else (128000,)):
            for backend in ('control',a.backend):
                opts=dict(LOD_H100_FAST_SCAN_SELECT=str(int(backend in ('scan','hybrid'))),LOD_H100_FAST_BIASED_SELECT=str(int(backend=='bias')),
                    LOD_H100_SCAN_ATOMIC_EXPERTS=str(int(backend=='hybrid')))
                if a.stage=='profiles':opts['LOD_H100_PREFILL_PROFILE']='1'
                submit(f'{backend}-{length}-{a.stage}','benchmarks.h100_comparison',
                    ['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                     '--manifest',r/'threshold-next-scaling-prompts.json','--length',length,'--tokens',2,
                     '--decode-warmup-steps',0,'--warmups',1,'--repeats',3 if a.stage=='models' else 1,
                     '--capture-token-ids','--replay-prompt-tokens','--finish-lod-prefill-cache'],opts,required)
    (r/f'threshold-next-selectpack-{a.label}-{a.stage}-runs.json').write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')


if __name__=='__main__':main()
