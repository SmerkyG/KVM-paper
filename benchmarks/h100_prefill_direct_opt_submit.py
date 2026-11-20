"""Submit matched direct-summary optimization runs through immutable cluster-run.

Base commands come from the completed scaled-FP16 and top-eight controls.
Profiles include work counts; uninstrumented model timings are kept separate.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('profiles','models','verify','quality','checks','probe','numerics','rounding','statecheck','rmscheck'),required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--variants',nargs='+')
    p.add_argument('--lengths',nargs='+',type=int,default=[128000])
    p.add_argument('--env',nargs='*',default=[])
    p.add_argument('--quality-panel',action='store_true')
    p.add_argument('--quality-rows',nargs='+',type=int,help='Diagnostic subset of frozen quality source rows')
    p.add_argument('--require-run',action='append',default=[])
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if a.variants is None:a.variants=['baseline','top8']if a.stage in ('models','profiles')else['select']
    state=Path.home()/'.local/state/lod-cluster-run';results=Path('/home/ubuntu/comparison/results')
    cpu=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m']
    manifest=results/f'threshold-directopt-{a.label}-{a.stage}-runs.json'
    if manifest.exists():raise FileExistsError(manifest)
    def prior(stem):
        paths=list(state.glob('*-'+stem+'/run.json'));assert len(paths)==1,(stem,paths)
        meta=json.loads(paths[0].read_text());assert meta['state']=='finished' and meta['exit_code']==0
        return meta['command'].copy()
    def setarg(cmd,key,value):cmd[cmd.index(key)+1]=str(value)
    jobs=[]
    for variant in a.variants:
        assert variant in ('baseline','top8','dense','hoist','select','merge','combined'),variant
        if variant=='dense':assert a.stage=='models'
        if variant=='top8':assert a.stage in ('models','profiles','quality')
        if a.stage=='quality':
            panel='panel' if a.quality_panel else 'cwe32'
            cmd=prior(f'threshold-offset-scaled-fp16-{panel}-v1-prefix-closed-tuned-ruler')
            if a.quality_rows:cmd.extend(['--source-rows',*map(str,a.quality_rows)])
        elif a.stage in ('checks','rounding','statecheck','rmscheck'):
            cmd=prior('threshold-offset-scaled-fp16-v1-prefix-closed-tuned-m32-n256-c2')
        else:
            stem=('threshold-offset-scaling-controls-v1-'+variant+'-128000-models' if variant in ('top8','dense')
                  else 'threshold-offset-scaling-fp16-v1-prefix-closed-tuned-128000-models')
            cmd=prior(stem)
        opts=dict(LOD_H100_DIRECT_HOIST=str(int(variant in ('hoist','combined'))),
                  LOD_H100_DIRECT_FUSE_SELECT=str(int(variant in ('select','combined'))),
                  LOD_H100_DIRECT_FUSE_MERGE=str(int(variant in ('merge','combined'))))
        if variant=='top8':opts.update(LOD_H100_PREFILL_RULE='',LOD_H100_PREFILL_MULTIPLIER='')
        opts.update(v.split('=',1) for v in a.env)
        if a.stage=='profiles':opts.update(LOD_H100_PREFILL_PROFILE='1',LOD_H100_PREFILL_SCALING_AUDIT='1')
        if a.stage=='verify':opts['LOD_H100_FAST_VERIFY']='1'
        cmd=[v for v in cmd if not any(v.startswith(k+'=') for k in opts)]
        cmd[1:1]=[f'{k}={v}' for k,v in opts.items()]
        if a.stage in ('rounding','statecheck','rmscheck'):
            at=cmd.index('--',cmd.index('benchmarks.h100_checked_run'))+1
            module={'statecheck':'benchmarks.h100_state_max_check','rmscheck':'benchmarks.h100_state_rms_check',
                'rounding':'benchmarks.h100_prefill_direct_rounding_check'}[a.stage]
            cmd=cmd[:at]+['python','-m',module,'--output','unused.json']
        if a.require_run:
            at=cmd.index('benchmarks.h100_checked_run')+1
            cmd[at:at]=[v for rid in a.require_run for v in ('--require-run',str(state/rid/'run.json'))]
        if a.stage in ('probe','numerics'):
            assert variant=='select'
            cmd[cmd.index('benchmarks.h100_comparison')]='benchmarks.h100_prefill_direct_opt_'+('probe' if a.stage=='probe'else'numerics')
        lengths=a.lengths if a.stage not in ('quality','checks','rounding','statecheck','rmscheck') else [32768 if a.stage=='quality' else 0]
        for length in lengths:
            command=cmd.copy();name=f'threshold-directopt-{a.label}-{variant}-{length}-{a.stage}'
            if a.stage not in ('quality','checks','rounding','statecheck','rmscheck'):
                setarg(command,'--length',length);setarg(command,'--repeats',3 if a.stage=='models' else 1)
                setarg(command,'--warmups',0 if a.stage in ('verify','probe','numerics') else 1)
                if a.stage=='verify':command.append('--check-graph-same-cache')
            out=results/(name+'.json');setarg(command,'--output',out)
            response=subprocess.check_output(cpu+['benchmarks.h100_cluster_run','--name',name,
                '--detach','--num-gpus','1','--',*command],cwd=repo,text=True)
            rid=response.splitlines()[0].split()[-1]
            jobs.append(dict(id=rid,name=name,variant=variant,length=length,command=command,output=str(out)))
            manifest.write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')
            print(response,flush=True)


if __name__=='__main__':main()
