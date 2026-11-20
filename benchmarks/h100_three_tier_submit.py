"""Submit logged uv/cluster-run three-tier H100 validation and comparisons.

Tuning and model runs use frozen ProLong prompts. Model windows include 512
warmup and 1,024 measured decode steps, complete graphs, and serving updates.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('tune', 'model', 'quality', 'bracket', 'diagnostic'), required=True)
    p.add_argument('--label', default='v1')
    p.add_argument('--frontier-run', required=True)
    p.add_argument('--cohort', type=Path)
    a = p.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home()/'.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    cluster = [str(Path.home()/'.local/bin/uv'), 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
               'python', '-m', 'benchmarks.h100_cluster_run']
    uv = [str(Path.home()/'.local/bin/uv'), 'run', '--offline', '--no-project', '--python',
          '/home/ubuntu/comparison/.venv/bin/python', '--with', 'triton==3.7.1', 'python', '-m']
    base_env = dict(CUDA_HOME='/usr/local/cuda-12.4', LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64',
                    HF_HUB_OFFLINE='1', LOD_TRACK_UPDATES='1', PYTHONPATH='/home/ubuntu/comparison/RetroInfer:.')
    manifest = '/home/ubuntu/lod-h100/results/prompts-b1.json'
    checkpoint = Path('/home/ubuntu/lod-h100/model-path.txt').read_text().strip()
    jobs = []

    def submit(name, module, args, options=None, requires=()):
        command = ['env', *(f'{k}={v}' for k,v in (base_env | (options or {})).items()), *uv,
                   'benchmarks.h100_checked_run']
        command += [x for rid in (a.frontier_run, *requires) for x in ('--require-run', str(state/rid/'run.json'))]
        output = results/(name+'.json')
        command += ['--', 'python', '-m', module, *map(str,args), '--output', str(output)]
        response = subprocess.check_output(cluster+['--name', name, '--detach', '--num-gpus', '1', '--']+command,
                                           cwd=root, text=True)
        rid = response.splitlines()[0].split()[-1]
        jobs.append(dict(name=name, id=rid, command=command, output=str(output)))
        print(response, flush=True)
        return rid

    chosen = {}
    if a.stage != 'tune':
        tuning = json.loads((results/f'three-tier-{a.label}-tune-runs.json').read_text())
        for job in tuning['jobs']:
            meta = json.loads((state/job['id']/'run.json').read_text())
            if meta.get('exit_code') != 0 or meta['state'] != 'finished':
                raise ValueError('tuning has not passed: '+job['name'])
            data = json.loads(Path(job['output']).read_text())
            groups = collections.defaultdict(list)
            for row in data['records']:
                if row['implementation'] == 'three-tier':
                    groups[row['tile'], row['workers'], row['page_warps'], row['merge_block']].append(row['microseconds'])
            if data['stage'] != 'complete' or any(len(v)!=3 for v in groups.values()):
                raise ValueError('incomplete real-layer tuning')
            best = min(groups, key=lambda k: statistics.mean(groups[k]))
            chosen[data['mode'], data['length']] = dict(zip(('tile','workers','page_warps','merge_block'), best))
    for length in ((32768,) if a.stage in ('quality','diagnostic') else (32768,65536)):
        for mode in ('three-tier-bf16', 'three-tier-int4'):
            stem = f'three-tier-{a.label}-{mode}-{length//1024}k'
            if a.stage == 'tune':
                submit(stem+'-tune', 'benchmarks.h100_three_tier_probe',
                       ['--manifest',manifest,'--length',length,'--mode',mode])
                continue
            cfg = chosen[mode,length]
            options = dict(LOD_H100_THRESHOLD_MULTIPLIER='16', LOD_H100_THRESHOLD_IMPLEMENTATION='three-tier',
                LOD_H100_THRESHOLD_TILE=str(cfg['tile']), LOD_H100_THREE_TIER_WORKERS=str(cfg['workers']),
                LOD_H100_THREE_TIER_PAGE_WARPS=str(cfg['page_warps']), LOD_H100_THRESHOLD_MERGE_BLOCK=str(cfg['merge_block']))
            if a.stage == 'diagnostic':
                if a.cohort is None:
                    p.error('diagnostic requires --cohort')
                submit(stem+'-quality-uncached-misses', 'benchmarks.retro_ruler_run',
                    ['--input',a.cohort,'--mode',mode,'--check-graph'],
                    options | dict(LOD_H100_THREE_TIER_CACHED_MEANS='0'))
                continue
            if a.stage == 'quality':
                if a.cohort is None:
                    p.error('quality requires --cohort')
                for variant in ('top8','threshold'):
                    submit(stem+'-quality-'+variant, 'benchmarks.retro_ruler_run',
                        ['--input',a.cohort,'--mode',mode,'--tasks','niah_multikey_3','niah_multivalue',
                         '--limit-per-task',16,'--check-graph'], options if variant=='threshold' else {})
                continue
            common = ['--method','retro-lod-int4' if mode.endswith('int4') else 'retro-lod-bf16',
                '--lod-graph','--checkpoint',checkpoint,'--manifest',manifest,'--length',length,
                '--capture-token-ids','--replay-prompt-tokens','--finish-lod-prefill-cache']
            if a.stage == 'bracket':
                prior = json.loads((results/f'three-tier-{a.label}-model-runs.json').read_text())
                prerequisite = next(j['id'] for j in prior['jobs'] if j['name']==stem+'-check')
                submit(stem+'-top8-after', 'benchmarks.h100_comparison', common+
                    ['--tokens',1537,'--decode-warmup-steps',512,'--warmups',1,'--repeats',2], requires=(prerequisite,))
                continue
            check = submit(stem+'-check', 'benchmarks.h100_comparison', common+
                ['--tokens',514,'--decode-warmup-steps',0,'--warmups',0,'--repeats',1,'--check-graph-same-cache'],
                options | dict(LOD_H100_DECODE_CHECK_REFERENCE='1'))
            for variant in ('top8','threshold','uncached'):
                env = {} if variant=='top8' else options
                if variant == 'uncached':
                    env = env | dict(LOD_H100_THREE_TIER_CACHED_MEANS='0')
                submit(stem+'-'+variant, 'benchmarks.h100_comparison', common+
                    ['--tokens',1537,'--decode-warmup-steps',512,'--warmups',1,'--repeats',2], env, (check,))
    if a.stage == 'model':
        for length in (32768,65536):
            submit(f'three-tier-{a.label}-dense-{length//1024}k', 'benchmarks.h100_comparison',
                ['--method','retro-full','--native-graph','--checkpoint',checkpoint,'--manifest',manifest,
                 '--length',length,'--tokens',1537,'--decode-warmup-steps',512,'--warmups',1,'--repeats',2,
                 '--capture-token-ids','--replay-prompt-tokens'])
    payload = dict(stage='submitted', jobs=jobs, chosen={f'{k[0]}/{k[1]}':v for k,v in chosen.items()})
    (results/f'three-tier-{a.label}-{a.stage}-runs.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__ == '__main__':
    main()
