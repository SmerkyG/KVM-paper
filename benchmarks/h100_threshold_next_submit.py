"""Queue threshold-prefill optimization, context scaling, and Qwen3 checks.

Every GPU command runs through uv and the source-snapshotting cluster-run.
The first exact prefill chunk and region/page policies are not retuned here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('checks','pilots','scaling','scaling-audit','llama-quality','qwen-smoke','qwen-quality','author-check','author-smoke','author-quality','tiled-checks','tiled-pilots','stream-checks','stream-pilots','reduce-checks','reduce-pilots','profiles','real-check','real-graphs'),required=True)
    p.add_argument('--label',default='v1');p.add_argument('--checks-label',default='v1')
    p.add_argument('--sm',type=int,choices=(1,4,8),default=1)
    p.add_argument('--lengths',type=int,nargs='+',default=[32768,65536,128000])
    p.add_argument('--implementations',nargs='+',default=['closed','fast'])
    p.add_argument('--policies',nargs='+',choices=('top8-prefill','threshold-prefill'),default=['top8-prefill','threshold-prefill'])
    p.add_argument('--select-warps',type=int,choices=(2,4,8,16),default=8)
    p.add_argument('--fast-select-warps',type=int,choices=(2,4,8,16),default=8)
    p.add_argument('--reducer',choices=('wide','split-d'),default='wide')
    p.add_argument('--variants',nargs='+',help='Limit reduction checks/pilots to named variants, e.g. split')
    p.add_argument('--require-run',action='append',default=[],help='Additional prerequisite cluster run ID')
    p.add_argument('--scratch-tile',type=int,default=4096,help='Internal scratch query tile; does not change the exact prefill chunk')
    p.add_argument('--skip-controls',action='store_true',help='Scaling follow-up with existing frozen dense/top-eight controls')
    p.add_argument('--expandable-allocator',action='store_true',help='Use expandable CUDA allocation segments to avoid large-workspace fragmentation')
    a=p.parse_args();root=Path('/home/ubuntu/lod-h100/code');r=Path('/home/ubuntu/comparison/results');state=Path.home()/'.local/state/lod-cluster-run'
    uv=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/home/ubuntu/comparison/.venv/bin/python','--with','triton==3.7.1','python','-m']
    if a.stage.startswith('author-'):
        uv=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/home/ubuntu/comparison/.venv/bin/python','--with','transformers==4.51.3','python','-m']
    cluster=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
    env={'CUDA_HOME':'/usr/local/cuda-12.4','LD_LIBRARY_PATH':'/usr/local/cuda-12.4/lib64','HF_HUB_OFFLINE':'1',
         'PYTHONPATH':'/home/ubuntu/comparison/RetroInfer:.','LOD_TRACK_UPDATES':'1'}
    if a.expandable_allocator:env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    decode={'LOD_RETRO_TWO_TIER':'1','LOD_H100_THRESHOLD_MULTIPLIER':'16','LOD_H100_THRESHOLD_RULE':'region',
            'LOD_H100_THRESHOLD_IMPLEMENTATION':'materialized','LOD_H100_THRESHOLD_SPLITS':'32',
            'LOD_AMD_THRESHOLD_ENTRY_BLOCK':'128','LOD_AMD_THRESHOLD_ENTRY_WARPS':'4','LOD_AMD_THRESHOLD_MERGE_BLOCK':'32','LOD_AMD_THRESHOLD_PRECISE':'1'}
    jobs=[]
    def prefill(impl,sm):
        return {'LOD_H100_PREFILL_RULE':'region','LOD_H100_PREFILL_MULTIPLIER':'16','LOD_H100_PREFILL_IMPLEMENTATION':impl,
            'LOD_H100_CLOSED_MATMUL':'blas','LOD_H100_CLOSED_METADATA':'atomic','LOD_H100_CLOSED_SELECT_QM':str(sm),
            'LOD_H100_CLOSED_SELECT_WARPS':str(a.fast_select_warps if impl=='fast' else a.select_warps),'LOD_H100_CLOSED_REDUCE':a.reducer,'LOD_H100_CLOSED_LEAF_M':'64',
            'LOD_H100_CLOSED_LEAF_N':'32','LOD_H100_CLOSED_LEAF_WARPS':'4','LOD_H100_PREFILL_AGGREGATE':str(int(sm>1)),
            'LOD_H100_FAST_LEAF':'native','LOD_H100_FAST_SAVED_CORRECTION':'1','LOD_H100_FAST_FULL_SELECT':'1',
            'LOD_H100_FAST_TILE':str(a.scratch_tile),'LOD_H100_CLOSED_TILE':str(a.scratch_tile)}
    def submit(name,module,args,options,requires=()):
        output=r/(name+'.json');cmd=['env',*(f'{k}={v}' for k,v in (env|options).items()),*uv,'benchmarks.h100_checked_run']
        if 'qwen' in a.stage or a.stage.startswith('real-'):
            cmd[cmd.index('--with'):cmd.index('--with')]=['--with','transformers==4.51.3']
        cmd += [x for rid in [*requires,*a.require_run] for x in ('--require-run',str(state/rid/'run.json'))]
        cmd += ['--','python','-m',module,*map(str,args),'--output',str(output)]
        response=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+cmd,cwd=root,text=True)
        rid=response.splitlines()[0].split()[-1];jobs.append(dict(name=name,id=rid,output=str(output),command=cmd));print(response,flush=True);return rid
    checks={}
    if a.stage not in ('checks','qwen-smoke','author-check','author-smoke','author-quality','tiled-checks','tiled-pilots','stream-checks','stream-pilots','reduce-checks','reduce-pilots','profiles','real-check','real-graphs'):
        manifest=json.loads((r/f'threshold-next-{a.checks_label}-checks-runs.json').read_text())
        checks={j['name'].split('-check-')[1]:j['id'] for j in manifest['jobs']}
    if a.stage in ('reduce-checks','reduce-pilots','profiles'):
        cases=[('closed','w2',2,'wide'),('closed','w4',4,'wide'),('closed','split',8,'split-d'),
               ('fast','w4',4,'wide'),('fast','split',8,'split-d')]
        if a.variants:cases=[c for c in cases if c[1] in a.variants]
        if a.stage=='profiles':cases=[('closed','profile',8,'wide'),('fast','profile',8,'wide')]
        for impl,tag,warps,reducer in cases:
            opts=prefill(impl,1)|{'LOD_H100_CLOSED_SELECT_WARPS':str(warps),'LOD_H100_CLOSED_REDUCE':reducer}
            name=f'threshold-next-{a.label}-{impl}-{tag}'
            if a.stage=='reduce-checks':
                module='benchmarks.h100_threshold_prefill_closed_check' if impl=='closed' else 'benchmarks.h100_threshold_fast_check'
                args=['--backend','blas'] if impl=='closed' else ['--rule','region','--regions',257,'--stress-layout']
                submit(name+'-check',module,args,opts)
            else:
                if a.stage=='profiles':opts['LOD_H100_PREFILL_PROFILE']='1';required=[]
                else:
                    previous=json.loads((r/f'threshold-next-{a.checks_label}-reduce-checks-runs.json').read_text())
                    required=[j['id'] for j in previous['jobs'] if j['name'].endswith(f'-{impl}-{tag}-check')]
                    assert len(required)==1
                n=32768 if impl=='closed' else 65536
                submit(name+f'-{n}','benchmarks.h100_comparison',
                    ['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                     '--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json','--length',n,'--tokens',2,
                     '--decode-warmup-steps',0,'--warmups',1,'--repeats',1 if a.stage=='profiles' else 2,
                     '--capture-token-ids','--replay-prompt-tokens','--finish-lod-prefill-cache'],decode|opts,required)
    elif a.stage=='author-check':
        submit(f'threshold-next-{a.label}-author-frontier-check','benchmarks.h100_multipole_quality_check',[],
            {'PYTHONPATH':'/home/ubuntu/comparison/MultipoleAttention:/home/ubuntu/comparison/RetroInfer:.'})
    elif a.stage in ('author-smoke','author-quality'):
        for mode in ('multipole','squeezed-multipole'):
            for budget in (512,2048):
                required=[]
                if a.stage=='author-quality':
                    previous=json.loads((r/f'threshold-next-{a.checks_label}-author-smoke-runs.json').read_text())
                    required=[j['id'] for j in previous['jobs'] if j['name']==f'threshold-next-{a.checks_label}-qwen-{mode}-{budget}']
                    assert len(required)==1
                submit(f'threshold-next-{a.label}-qwen-{mode}-{budget}','benchmarks.retro_ruler_run',
                    ['--input',r/'qwen3-ruler-32k-frozen.json','--mode',mode,'--multipole-budget',budget,
                     '--model-name','Qwen/Qwen3-8B','--prompt-format','qwen3-preformatted',
                     '--tasks','niah_multikey_2','niah_multikey_3','cwe','qa_1',
                     '--limit-per-task',2 if a.stage=='author-smoke' else 32,'--check-graph',
                     '--validate-prefill-against',r/'qwen3-ruler-dense-selected.json'],
                    {'PYTHONPATH':'/home/ubuntu/comparison/MultipoleAttention:/home/ubuntu/comparison/RetroInfer:.',
                     'MULTIPOLE_STABLE_PACK':'1','MULTIPOLE_CHECK_REPEAT':'1'},required)
    elif a.stage in ('stream-checks','stream-pilots'):
        for sm in (16,32):
            opts=prefill('fast',1)|{'LOD_H100_FAST_CLOSED_PASS':'1','LOD_H100_STREAM_M':str(sm),'LOD_H100_STREAM_N':'64'}
            if a.stage=='stream-checks':
                for c in (2,512):
                    submit(f'threshold-next-{a.label}-stream-m{sm}-c{c}-check','benchmarks.h100_threshold_fast_check',
                        ['--rule','region','--regions',257,'--stress-layout','--multiplier',c],opts)
            else:
                previous=json.loads((r/f'threshold-next-{a.checks_label}-stream-checks-runs.json').read_text())
                required=[j['id'] for j in previous['jobs'] if f'-m{sm}-' in j['name']]
                for n in (32768,65536):
                    submit(f'threshold-next-{a.label}-stream-m{sm}-{n}','benchmarks.h100_comparison',
                        ['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                         '--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json','--length',n,'--tokens',2,
                         '--decode-warmup-steps',0,'--warmups',1,'--repeats',2,'--capture-token-ids','--replay-prompt-tokens',
                         '--finish-lod-prefill-cache'],decode|opts,required)
    elif a.stage in ('real-check','real-graphs'):
        lengths=[4096] if a.stage=='real-check' else [32768,131072]
        required=[] if a.stage=='real-check' else [j['id'] for j in json.loads((r/f'threshold-next-{a.checks_label}-real-check-runs.json').read_text())['jobs']]
        for n in lengths:
            submit(f'threshold-next-{a.label}-real-qwen-{n}','benchmarks.qwen3_real_capture',
                ['--dataset-parquet','/home/ubuntu/.cache/huggingface/hub/datasets--Seerkfang--prolong-64k-512-new/snapshots/97295b7d7fe48dc0aa6ba373af3a8b9d945e505b/train_00000.parquet',
                 '--length',n,'--chunk-size',2048 if n==4096 else 4096,'--layer',18,
                 *(['--yarn'] if n>32768 else []),'--benchmark','--benchmark-kinds',
                 *(['threshold'] if a.stage=='real-check' else ['author','lod','threshold'])],{},required)
    elif a.stage in ('tiled-checks','tiled-pilots'):
        for sm in (16,32):
            opts=prefill('fast',1)|{'LOD_H100_FAST_TILED_SELECT':'1','LOD_H100_FAST_TILED_M':str(sm)}
            if a.stage=='tiled-checks':
                submit(f'threshold-next-{a.label}-tiled-m{sm}-check','benchmarks.h100_threshold_fast_check',['--rule','region','--regions',257],opts)
            else:
                previous=json.loads((r/f'threshold-next-{a.checks_label}-tiled-checks-runs.json').read_text())
                required=next(j['id'] for j in previous['jobs'] if f'-m{sm}-' in j['name'])
                for n in (32768,65536):
                    submit(f'threshold-next-{a.label}-tiled-m{sm}-{n}','benchmarks.h100_comparison',
                        ['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                         '--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json','--length',n,'--tokens',2,
                         '--decode-warmup-steps',0,'--warmups',1,'--repeats',2,'--capture-token-ids','--replay-prompt-tokens',
                         '--finish-lod-prefill-cache'],decode|opts,[required])
    elif a.stage=='checks':
        for impl in ('closed','fast'):
            for sm in (1,4,8):
                name=f'threshold-next-{a.label}-check-{impl}-m{sm}'
                module='benchmarks.h100_threshold_prefill_closed_check' if impl=='closed' else 'benchmarks.h100_threshold_fast_check'
                args=['--backend','blas'] if impl=='closed' else ['--rule','region']
                submit(name,module,args,prefill(impl,sm))
    elif a.stage in ('pilots','scaling','scaling-audit'):
        checkpoint=Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()
        manifest_path=Path('/home/ubuntu/lod-h100/results/prompts-b1.json')
        if a.stage in ('scaling','scaling-audit'):
            originals=[manifest_path,manifest_path.with_name('prompts-b1-128k.json')]
            data=[json.loads(path.read_text()) for path in originals]
            assert data[0]['identity']==data[1]['identity'],'scaling prompts have different tokenizer or dataset identities'
            merged={'identity':data[0]['identity'],'contexts':{},'source_manifest_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}}
            for item in data:
                for key,value in item['contexts'].items():
                    assert key not in merged['contexts'] or merged['contexts'][key]==value
                    merged['contexts'][key]=value
            assert all(str(n) in merged['contexts'] for n in a.lengths)
            manifest_path=r/'threshold-next-scaling-prompts.json'
            raw=json.dumps(merged,sort_keys=True)+'\n'
            if manifest_path.exists():assert manifest_path.read_text()==raw,'frozen scaling manifest changed'
            else:manifest_path.write_text(raw)
        cases=[(32768,'closed',sm) for sm in (1,4,8)]+[(65536,'fast',sm) for sm in (1,4,8)] if a.stage=='pilots' else [
            (n,impl,a.sm) for n in a.lengths for impl in
            (('top8','fast') if a.stage=='scaling-audit' else (*(() if a.skip_controls else ('dense','top8')),*a.implementations))]
        for n,impl,sm in cases:
            opts=decode.copy() if impl!='dense' else {}
            if impl in ('closed','fast'):opts.update(prefill(impl,sm))
            if a.stage in ('scaling','scaling-audit'):opts['RETRO_PREFILL_MLP_CHUNK']='8192'
            if a.stage=='scaling-audit':opts.update(LOD_H100_PREFILL_SCALING_AUDIT='1',LOD_H100_PREFILL_PROFILE='1')
            name=f'threshold-next-{a.label}-{impl}-m{sm}-{n}'
            args=['--method','retro-full' if impl=='dense' else 'retro-lod-bf16',
                  '--native-graph' if impl=='dense' else '--lod-graph','--checkpoint',checkpoint,
                  '--manifest',manifest_path,'--length',n,
                  '--tokens',2,'--decode-warmup-steps',0,'--warmups',1,'--repeats',1 if a.stage=='scaling-audit' else 2 if a.stage=='pilots' else 3,
                  '--capture-token-ids','--replay-prompt-tokens']
            if impl!='dense':args+=['--finish-lod-prefill-cache']
            requires=[checks[f'{impl}-m{sm}']] if impl in ('closed','fast') else []
            submit(name,'benchmarks.h100_comparison',args,opts,requires)
    elif a.stage=='llama-quality':
        submit(f'threshold-next-{a.label}-llama-threshold-prefill','benchmarks.retro_ruler_run',
            ['--input',r/'quality-ruler-32k-frozen-13x32.json','--mode','two-tier','--limit-per-task',32,'--check-graph'],
            decode|prefill('closed',a.sm),[checks[f'closed-m{a.sm}']])
    else:
        for policy in a.policies:
            opts=decode.copy()
            if policy=='threshold-prefill':opts.update(prefill('closed',a.sm))
            name=f'threshold-next-{a.label}-qwen-{policy}'
            requires=[checks[f'closed-m{a.sm}']] if a.stage=='qwen-quality' and policy=='threshold-prefill' else []
            submit(name,'benchmarks.retro_ruler_run',
                ['--input',r/'qwen3-ruler-32k-frozen.json','--mode','two-tier','--model-name','Qwen/Qwen3-8B',
                 '--prompt-format','qwen3-preformatted','--tasks','niah_multikey_2','niah_multikey_3','cwe','qa_1',
                 '--limit-per-task',2 if a.stage=='qwen-smoke' else 32,'--check-graph'],opts,requires)
    (r/f'threshold-next-{a.label}-{a.stage}-runs.json').write_text(json.dumps(dict(stage='submitted',jobs=jobs),indent=2)+'\n')


if __name__=='__main__':main()
