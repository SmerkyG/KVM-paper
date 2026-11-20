"""Freeze AMD threshold sources and submit experiments through uv/cluster-run."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def freeze(root,phase):
    source=phase/'source'
    if source.exists():raise ValueError('refusing to overwrite a frozen source')
    source.mkdir(parents=True)
    for name in ('benchmarks','integrations','lod_attention','tests'):
        shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','figures'))
    for name in ('pyproject.toml','uv.lock','README.md'):shutil.copy2(root/name,source/name)
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*')) if p.is_file()}
    (phase/'source-manifest.json').write_text(json.dumps(dict(source=str(source),source_files=hashes),indent=2)+'\n')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','probe','replay','speed','quality'),required=True)
    p.add_argument('--label',required=True);p.add_argument('--root',type=Path,default=Path('results/amd-threshold-fast'))
    p.add_argument('--settings',type=Path)
    p.add_argument('--configs',type=Path);p.add_argument('--snapshots',type=Path)
    p.add_argument('--implementations',nargs='+',default=['packed','indexed','combined'])
    p.add_argument('--entry-warps',type=int,default=2)
    p.add_argument('--profile',action='store_true');p.add_argument('--native',action='store_true')
    p.add_argument('--speed-gpu',default='1')
    p.add_argument('--variants',nargs='+',choices=('top8','legacy','fast','top8-after'),default=['top8','legacy','fast'])
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1];phase=(root/a.root/a.label).resolve();phase.mkdir(parents=True,exist_ok=True)
    source=freeze(root,phase);jobs=[]
    def submit(name,module,args,env=(),placement=()):
        output=phase/(name+'.json')
        command=['cluster-run','--detach','--name',name,*placement,'--','env',f'PYTHONPATH={source}:{source}/integrations/vllm_lod',
                 'HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1',*env,'uv','run','--offline','--no-project','--python',
                 '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python','python','-m',module,*args,'--output',str(output)]
        run=subprocess.run(command,cwd=source,text=True,capture_output=True,check=True)
        match=re.search(r'^Submitted (\S+)',run.stdout,re.MULTILINE)
        if not match:raise RuntimeError(run.stdout+run.stderr)
        row=dict(id=match.group(1),name=name,output=str(output),command=command);jobs.append(row)
        (phase/'jobs.json').write_text(json.dumps(dict(phase=a.phase,source=str(source),jobs=jobs),indent=2)+'\n')
        print(json.dumps(row),flush=True)
    if a.phase=='check':
        choices=([(model,settings) for model,settings in json.loads(a.settings.read_text()).items()]
                 if a.settings else [(impl,{'LOD_AMD_THRESHOLD_IMPLEMENTATION':impl,'LOD_AMD_THRESHOLD_ENTRY_WARPS':str(a.entry_warps)}) for impl in a.implementations])
        for impl,settings in choices:
            submit(f'amd-fast-{a.label}-{impl}','benchmarks.amd_threshold_synthetic_check',[],
                   [f'{key}={value}' for key,value in settings.items()],['--num-gpus','1'])
    elif a.phase=='probe':
        for model in ('qwen','k2'):
            options=[]
            if a.settings:
                settings=json.loads(a.settings.read_text())[model]
                options=[f'{key}={value}' for key,value in settings.items()]
                options += ['LOD_AMD_THRESHOLD_PROBE_RULES=region','LOD_AMD_THRESHOLD_PROBE_CONFIGS='+json.dumps([
                    ('legacy',64,64,32,4),tuple(settings[key] if key.endswith('IMPLEMENTATION') else int(settings[key]) for key in
                    ('LOD_AMD_THRESHOLD_IMPLEMENTATION','LOD_AMD_THRESHOLD_TILE','LOD_AMD_THRESHOLD_SPLITS','LOD_AMD_THRESHOLD_ENTRY_BLOCK','LOD_AMD_THRESHOLD_ENTRY_WARPS'))])]
            submit(f'amd-fast-{a.label}-{model}','benchmarks.amd_threshold_fast_probe',
                   ['--model',model,'--length','65536'],['LOD_AMD_THRESHOLD_MULTIPLIER=16','LOD_AMD_THRESHOLD_CAPTURE_QUERY=1',
                    *options,*(['LOD_AMD_THRESHOLD_PROBE_NATIVE=1'] if a.native else []),
                    *([f'LOD_AMD_THRESHOLD_SNAPSHOT_DIR={phase}/snapshots/{model}', 'LOD_AMD_THRESHOLD_PROBE_NATIVE=1',
                       'LOD_AMD_THRESHOLD_PROBE_RULES=region',
                       'LOD_AMD_THRESHOLD_PROBE_CONFIGS='+json.dumps([('legacy',64,64,32,4),('indexed',64,64,32,2),('combined',64,64,32,2)])]
                      if a.label.startswith('capture-') else [])],['--num-gpus','1'])
    elif a.phase=='replay':
        if a.configs is None or a.snapshots is None:raise ValueError('replay requires configs and snapshots')
        configs=phase/'configs.json';shutil.copy2(a.configs,configs)
        for model in ('qwen','k2'):
            submit(f'amd-fast-{a.label}-{model}','benchmarks.amd_threshold_fast_replay',
                   ['--snapshots',str(a.snapshots.resolve()/model),'--configs',str(configs)],
                   ['LOD_AMD_THRESHOLD_REPLAY_PROFILE=1'] if a.profile else [],['--num-gpus','1'])
    elif a.phase=='quality':
        if a.settings is None:raise ValueError('quality phase requires chosen settings')
        settings=json.loads(a.settings.read_text())
        old=json.loads((root/'results/amd-threshold/full500-frozen/manifest.json').read_text())
        from benchmarks.amd_threshold_ruler import score
        cohort={}
        for model in ('qwen','k2'):
            cohort[model]={}
            for task in ('niah_multikey_2','niah_multikey_3'):
                original=Path(old['models'][model][task]['source']);raw=original.read_bytes()
                rows=sorted((json.loads(line) for line in raw.splitlines()),key=lambda row:row['doc_id'])
                chosen=[r for r in rows if r['65536']<1][:4]+[r for r in rows if r['65536']==1][:4]
                if len(chosen)!=8:raise ValueError('not enough historical failures/successes')
                for row in chosen:
                    if abs(score(row,row['filtered_resps'][0])-row['65536'])>1.e-12:raise ValueError('archive scorer mismatch')
                dest=phase/'cohort'/model/task/'samples_frozen.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(''.join(json.dumps(row)+'\n' for row in chosen))
                cohort[model][task]=dict(original=str(original),original_sha256=hashlib.sha256(raw).hexdigest(),
                    frozen=str(dest),frozen_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),doc_ids=[r['doc_id'] for r in chosen])
            for variant in ('legacy','fast'):
                env=['LOD_AMD_THRESHOLD_MULTIPLIER=16']
                if variant=='fast':env += [f'{key}={value}' for key,value in settings[model].items()]
                submit(f'amd-fast-{a.label}-{model}-{variant}','benchmarks.amd_threshold_ruler',
                    ['--checkpoint',{'qwen':'Qwen/Qwen3.8-27B-FP8','k2':'IFM/K2-Horizon-32B-FP8'}[model],
                     '--samples-root',str(phase/'cohort'/model),'--tasks','niah_multikey_2','niah_multikey_3',
                     '--selection','prefix','--limit','8'],env,['--num-gpus','1'])
        (phase/'cohort-manifest.json').write_text(json.dumps(dict(selection='four historical failures plus four successes per task; diagnostic only',models=cohort),indent=2)+'\n')
    else:
        if 'fast' in a.variants and a.settings is None:raise ValueError('fast speed phase requires chosen settings')
        settings=json.loads(a.settings.read_text()) if a.settings else {}
        for model,node in [('qwen','dev-shared-research-2.dev-training.svc.cluster.local'),('k2','dev-shared-research-4.dev-training.svc.cluster.local')]:
            # A hard GPU filter permits overlapping one's own jobs. One parent
            # allocation runs every variant serially to guarantee isolation.
            submit(f'amd-fast-{a.label}-{model}','benchmarks.amd_threshold_fast_sequence',
                   ['--model',model,'--variants',*a.variants,'--settings-json',json.dumps(settings.get(model,{}))],
                   [],[node,a.speed_gpu])


if __name__=='__main__':main()
