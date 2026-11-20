"""Submit snapshotted, serialized BF16 three-tier quality-repair experiments."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('panel','model','two-tier-control','page-check','page-probe','page-panel','page-model'),required=True)
    p.add_argument('--label',default='v1')
    p.add_argument('--pages',type=int,nargs='+',choices=(2,4,8),default=[2,4,8])
    p.add_argument('--tasks',nargs='+',default=['niah_multikey_3'])
    p.add_argument('--page-multipliers',type=float,nargs='+',default=[.25,.5,1.])
    p.add_argument('--checks-label',help='Reuse frontier checks from this label')
    p.add_argument('--probe-label',help='Reuse real-query timing probes from this label')
    p.add_argument('--page-rule',choices=('page','entry'),default='page')
    p.add_argument('--page-query',choices=('native','normalized'),default='native')
    p.add_argument('--region-multiplier',type=float,default=16.)
    p.add_argument('--require-model-label',help='Require this page-model stage to pass its same-cache checks before RULER')
    p.add_argument('--page-residual',choices=('region','pages'),default='region')
    p.add_argument('--compare-residuals',action='store_true')
    p.add_argument('--require-run',action='append',default=[],help='Additional cluster run id that must pass')
    a=p.parse_args()
    root=Path('/home/ubuntu/lod-h100/code');results=Path('/home/ubuntu/comparison/results');state=Path.home()/'.local/state/lod-cluster-run'
    uv=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/home/ubuntu/comparison/.venv/bin/python','--with','triton==3.7.1','python','-m']
    cluster=[str(Path.home()/'.local/bin/uv'),'run','--offline','--no-project','--python','/usr/bin/python3','python','-m','benchmarks.h100_cluster_run']
    env={'CUDA_HOME':'/usr/local/cuda-12.4','LD_LIBRARY_PATH':'/usr/local/cuda-12.4/lib64','HF_HUB_OFFLINE':'1','LOD_TRACK_UPDATES':'1',
         'PYTHONPATH':'/home/ubuntu/comparison/RetroInfer:.'}
    jobs=[]
    def submit(name,module,args,options=None,requires=()):
        output=results/(name+'.json')
        command=['env',*(f'{k}={v}' for k,v in (env|(options or {})).items()),*uv,'benchmarks.h100_checked_run']
        command += [x for rid in (*a.require_run,*requires) for x in ('--require-run',str(state/rid/'run.json'))]
        command += ['--','python','-m',module,*map(str,args),'--output',str(output)]
        response=subprocess.check_output(cluster+['--name',name,'--detach','--num-gpus','1','--']+command,cwd=root,text=True)
        rid=response.splitlines()[0].split()[-1];jobs.append(dict(name=name,id=rid,command=command,output=str(output)));print(response,flush=True);return rid
    if a.stage.startswith('page-'):
        base={'LOD_H100_THRESHOLD_MULTIPLIER':str(a.region_multiplier),'LOD_H100_THRESHOLD_IMPLEMENTATION':'three-tier',
              'LOD_H100_THRESHOLD_TILE':'64','LOD_H100_THREE_TIER_PAGE_WARPS':'2','LOD_H100_PAGE_THRESHOLD_RULE':a.page_rule,
              'LOD_H100_PAGE_QUERY':a.page_query,'LOD_H100_PAGE_RESIDUAL':a.page_residual}
        checks=[]
        if a.stage!='page-check':
            checks=[j['id'] for j in json.loads((results/f'three-tier-repair-{a.checks_label or a.label}-page-check-runs.json').read_text())['jobs']]
        if a.stage in ('page-panel','page-model'):
            checks += [j['id'] for j in json.loads((results/f'three-tier-repair-{a.probe_label or a.label}-page-probe-runs.json').read_text())['jobs']]
        if a.require_model_label:
            checks += [j['id'] for j in json.loads((results/f'three-tier-repair-{a.require_model_label}-page-model-runs.json').read_text())['jobs'] if j['name'].endswith('model-check')]
        if a.stage=='page-probe':
            for length in (32768,65536):
                submit(f'three-tier-repair-{a.label}-page-probe-{length//1024}k','benchmarks.h100_page_threshold_probe',
                    ['--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json','--length',length,'--rule',a.page_rule,'--query',a.page_query,
                     '--residual',a.page_residual,*(['--compare-residuals'] if a.compare_residuals else []),
                     '--multipliers',*a.page_multipliers],base,checks)
        else:
            for c in a.page_multipliers:
                stem=f'three-tier-repair-{a.label}-pagec{c:g}'.replace('.','p')
                options=base|{'LOD_H100_PAGE_THRESHOLD':str(c)}
                if a.stage=='page-check':
                    submit(stem+'-check','benchmarks.amd_threshold_three_tier_check',['--h100','--page-threshold',c,'--page-threshold-rule',a.page_rule,'--page-query',a.page_query,'--page-residual',a.page_residual])
                elif a.stage=='page-panel':
                    submit(stem+'-quality','benchmarks.h100_page_threshold_quality',
                        ['--probe',results/f'three-tier-repair-{a.probe_label or a.label}-page-probe-32k.json',
                         '--input',results/'quality-ruler-32k-frozen-13x32.json','--mode','three-tier-bf16','--tasks',*a.tasks,'--limit-per-task',32,'--check-graph'],options,checks)
                else:
                    import collections,statistics
                    for length in (32768,65536):
                        probe=json.loads((results/f'three-tier-repair-{a.probe_label or a.label}-page-probe-{length//1024}k.json').read_text());assert probe['stage']=='complete'
                        assert probe.get('page_threshold_rule','page')==a.page_rule and probe.get('page_query','native')==a.page_query
                        assert probe.get('region_multiplier',16)==a.region_multiplier
                        groups=collections.defaultdict(list)
                        for row in probe['records']:
                            if row['c']==c and row.get('residual','region')==a.page_residual:
                                groups[row['workers'],row['warps']].append(row['microseconds'])
                        assert groups and all(len(v)==3 for v in groups.values())
                        workers,warps=min(groups,key=lambda k:statistics.mean(groups[k]))
                        common=['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                                '--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json','--length',length,
                                '--capture-token-ids','--replay-prompt-tokens','--finish-lod-prefill-cache']
                        chosen=options|{'LOD_H100_THRESHOLD_TILE':str(64 if length==32768 else 128),
                                        'LOD_H100_THREE_TIER_WORKERS':str(workers),'LOD_H100_PAGE_THRESHOLD_WARPS':str(warps)}
                        check=submit(stem+f'-{length//1024}k-model-check','benchmarks.h100_comparison',
                            common+['--tokens',514,'--decode-warmup-steps',0,'--warmups',0,'--repeats',1,'--check-graph-same-cache'],
                            chosen|{'LOD_H100_DECODE_CHECK_REFERENCE':'1'},checks)
                        submit(stem+f'-{length//1024}k-model','benchmarks.h100_comparison',
                            common+['--tokens',1537,'--decode-warmup-steps',512,'--warmups',1,'--repeats',2],chosen,(check,))
        (results/f'three-tier-repair-{a.label}-{a.stage}-runs.json').write_text(json.dumps(dict(stage='submitted',jobs=jobs),indent=2)+'\n')
        return
    if a.stage=='two-tier-control':
        submit(f'three-tier-repair-{a.label}-two-tier-control','benchmarks.retro_ruler_run',
            ['--input',results/'quality-ruler-32k-frozen-13x32.json','--mode','two-tier','--tasks',*a.tasks,'--limit-per-task',32,'--check-graph'],
            {'LOD_H100_THRESHOLD_MULTIPLIER':'16','LOD_H100_THRESHOLD_IMPLEMENTATION':'materialized',
             'LOD_H100_THRESHOLD_SPLITS':'32','LOD_H100_SHARED_ENTRY_BLOCK':'128','LOD_H100_THRESHOLD_MERGE_BLOCK':'32',
             'LOD_AMD_THRESHOLD_ENTRY_BLOCK':'128','LOD_AMD_THRESHOLD_ENTRY_WARPS':'4','LOD_AMD_THRESHOLD_MERGE_BLOCK':'32','LOD_AMD_THRESHOLD_PRECISE':'1'})
    for pages in ([] if a.stage=='two-tier-control' else a.pages):
        stem=f'three-tier-repair-{a.label}-pages{pages}'
        options={'LOD_H100_THRESHOLD_MULTIPLIER':str(a.region_multiplier),'LOD_H100_THRESHOLD_IMPLEMENTATION':'three-tier',
                 'LOD_H100_THRESHOLD_TILE':'64','LOD_H100_THREE_TIER_PAGE_WARPS':'2','LOD_H100_THREE_TIER_PAGES':str(pages)}
        if a.stage=='panel':
            check=submit(stem+'-check','benchmarks.amd_threshold_three_tier_check',['--h100','--pages',pages])
            submit(stem+'-quality','benchmarks.retro_ruler_run',
                   ['--input',results/'quality-ruler-32k-frozen-13x32.json','--mode','three-tier-bf16','--tasks',*a.tasks,'--limit-per-task',32,'--check-graph'],options,(check,))
        else:
            for length in (32768,65536):
                submit(stem+f'-{length//1024}k-model','benchmarks.h100_comparison',
                    ['--method','retro-lod-bf16','--lod-graph','--checkpoint',Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip(),
                     '--manifest','/home/ubuntu/lod-h100/results/prompts-b1.json','--length',length,
                     '--tokens',1537,'--decode-warmup-steps',512,'--warmups',1,'--repeats',2,
                     '--capture-token-ids','--replay-prompt-tokens','--finish-lod-prefill-cache'],
                     options|({'LOD_H100_THRESHOLD_TILE':'128','LOD_H100_THREE_TIER_PAGE_WARPS':'1'} if length==65536 else {}))
    (results/f'three-tier-repair-{a.label}-{a.stage}-runs.json').write_text(json.dumps(dict(stage='submitted',jobs=jobs),indent=2)+'\n')


if __name__=='__main__':main()
