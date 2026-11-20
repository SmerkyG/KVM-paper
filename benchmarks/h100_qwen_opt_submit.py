"""Port validated Llama settings to Qwen3 and queue matched, immutable GPU runs.

All experiments use uv and h100_cluster_run's single-GPU lock. The old
Qwen closed-summary prefill is the baseline; native Q/K norms are preserved.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('checks','prepare','models','verify','quality','profiles'),required=True)
    p.add_argument('--label',default='v1');p.add_argument('--prepare-label',default='v1')
    p.add_argument('--variants',nargs='+',default=['dense','top8','baseline','optimized'])
    p.add_argument('--lengths',nargs='+',type=int,default=[32768,65536,128000])
    p.add_argument('--require-run',action='append',default=[])
    p.add_argument('--quality-limit',type=int,default=32)
    p.add_argument('--env',nargs='*',default=[],help='Explicit tuning overrides recorded in each immutable command')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    state=Path.home()/'.local/state/lod-cluster-run';results=Path('/home/ubuntu/comparison/results')
    manifest=results/f'qwen-opt-{a.label}-{a.stage}-runs.json'
    if manifest.exists():raise FileExistsError(manifest)
    prior=list(state.glob('*-threshold-directopt-accepted-v3-select-32768-models/run.json'));assert len(prior)==1
    data=json.loads(prior[0].read_text());assert data['state']=='finished'and data['exit_code']==0
    template={}
    for v in data['command'][1:]:
        if '='not in v:break
        k,value=v.split('=',1);template[k]=value
    common={k:v for k,v in template.items()if not k.startswith('LOD_')}
    common['RETRO_PREFILL_MLP_CHUNK']='8192'
    uv=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python',
        '/home/ubuntu/comparison/.venv/bin/python','--with','transformers==4.51.3','--with','triton==3.7.1','python','-m']
    cluster=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3',
        'python','-m','benchmarks.h100_cluster_run']
    checkpoint='/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
    corpus=results/f'qwen-opt-{a.prepare_label}-manifest-0-prepare.json'
    jobs=[]
    def submit(variant,n,module,args,env):
        name=f'qwen-opt-{a.label}-{variant}-{n}-{a.stage}';out=results/(name+'.json')
        cmd=['env',*(f'{k}={v}'for k,v in env.items()),*uv,'benchmarks.h100_checked_run']
        cmd += [v for rid in a.require_run for v in ('--require-run',str(state/rid/'run.json'))]
        cmd += ['--','python','-m',module,*map(str,args),'--output',str(out)]
        response=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--',*cmd],cwd=repo,text=True)
        rid=response.splitlines()[0].split()[-1]
        jobs.append(dict(id=rid,name=name,variant=variant,length=n,command=cmd,output=str(out)))
        manifest.write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n');print(response,flush=True)
    if a.stage=='checks':
        submit('rescale',0,'benchmarks.h100_state_rescale_check',[],common)
        submit('native',0,'benchmarks.retro_qwen3_check',[],common)
        submit('yarn',0,'benchmarks.retro_qwen3_check',['--yarn'],common)
        return
    if a.stage=='prepare':
        submit('manifest',0,'benchmarks.h100_qwen_opt_prepare',['--dataset',
            '/home/ubuntu/.cache/huggingface/hub/datasets--Seerkfang--prolong-64k-512-new/snapshots/97295b7d7fe48dc0aa6ba373af3a8b9d945e505b/train_00000.parquet'],common)
        return
    for n in ([32768]if a.stage=='quality'else a.lengths):
        for variant in a.variants:
            assert variant in ('dense','top8','baseline','optimized'),variant
            env=template.copy()
            env.update(LOD_H100_STATE_FUSED_RMS='0',LOD_H100_STATE_FUSED_RESCALE='1',LOD_H100_STATE_PACKED_MAX='1',
                LOD_H100_FAST_TILE='4096'if n>65536 else'auto',LOD_H100_FAST_VERIFY='0')
            if variant=='baseline':
                env.update(LOD_H100_STATE_FUSED_RESCALE='0',LOD_H100_STATE_PACKED_MAX='0',
                    LOD_H100_PREFILL_IMPLEMENTATION='closed',LOD_H100_CLOSED_PRECISION='bf16',
                    LOD_H100_CLOSED_REDUCE='wide',LOD_H100_CLOSED_METADATA='atomic',LOD_H100_CLOSED_MATMUL='blas',
                    LOD_H100_CLOSED_SELECT_WARPS='4',LOD_H100_CLOSED_SELECT_QM='1',LOD_H100_PREFILL_AGGREGATE='0',
                    LOD_H100_DIRECT_FUSE_SELECT='0',LOD_H100_FAST_TILE='4096')
            elif variant in ('dense','top8'):
                env.update(LOD_H100_PREFILL_RULE='',LOD_H100_PREFILL_MULTIPLIER='')
                if variant=='dense':env['LOD_H100_THRESHOLD_MULTIPLIER']=''
            env.update(value.split('=',1)for value in a.env)
            if a.stage=='quality':
                assert variant in ('baseline','optimized','top8')
                submit(variant,n,'benchmarks.retro_ruler_run',['--input',results/'qwen3-ruler-32k-frozen.json',
                    '--mode','two-tier','--model-name','Qwen/Qwen3-8B','--prompt-format','qwen3-preformatted',
                    '--tasks','niah_multikey_2','niah_multikey_3','cwe','qa_1',
                    '--limit-per-task',a.quality_limit,'--check-graph'],env)
                continue
            if a.stage=='verify':
                assert variant=='optimized';env['LOD_H100_FAST_VERIFY']='1'
            if a.stage=='profiles':
                assert variant!='dense';env.update(LOD_H100_PREFILL_PROFILE='1',LOD_H100_PREFILL_SCALING_AUDIT='1',LOD_H100_FULL_PREFILL_AUDIT='1')
            args=['--method','retro-full'if variant=='dense'else'retro-lod-bf16',
                '--native-graph'if variant=='dense'else'--lod-graph','--model-name','Qwen/Qwen3-8B',
                '--checkpoint',checkpoint,'--manifest',corpus,'--length',n,'--tokens',2,
                '--decode-warmup-steps',0,'--warmups',0 if a.stage=='verify'else 1,
                '--repeats',3 if a.stage=='models'else 1,'--capture-token-ids','--replay-prompt-tokens']
            if n>32768:args+=['--qwen-yarn']
            if variant!='dense':args+=['--finish-lod-prefill-cache']
            if a.stage=='verify':args+=['--check-graph-same-cache']
            submit(variant,n,'benchmarks.h100_comparison',args,env)


if __name__=='__main__':main()
