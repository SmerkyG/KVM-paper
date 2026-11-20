"""Matched full-model prefill controls and candidates, including cache completion."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('controls', 'candidates', 'checks', 'profiles'), required=True)
    p.add_argument('--label', default='final')
    p.add_argument('--closed-only', action='store_true')
    a = p.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home() / '.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    prior = json.loads((state / '1794244954699735307-threshold-fast-v6-legacy-64k/run.json').read_text())['command']
    prefix = [x for x in prior[:prior.index('benchmarks.h100_checked_run')]
              if not x.startswith(('LOD_H100_PREFILL_', 'LOD_H100_FAST_'))]
    legacy = [x for x in prior[:prior.index('benchmarks.h100_checked_run')] if x.startswith('LOD_H100_PREFILL_')]
    base = prior[prior.index('--', prior.index('benchmarks.h100_checked_run')) + 1:]
    cluster = [str(Path.home() / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
               'python', '-m', 'benchmarks.h100_cluster_run']
    jobs = []
    settings = ['LOD_H100_CLOSED_MATMUL=blas', 'LOD_H100_CLOSED_METADATA=atomic',
                'LOD_H100_CLOSED_SELECT_QM=1', 'LOD_H100_CLOSED_REDUCE=wide',
                'LOD_H100_CLOSED_LEAF_M=64', 'LOD_H100_CLOSED_LEAF_N=32', 'LOD_H100_CLOSED_LEAF_WARPS=4']

    def submit(name, cmd):
        output = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                                         cwd=root, text=True)
        rid = output.splitlines()[0].split()[-1]
        jobs.append(dict(name=name, id=rid))
        return rid

    if a.phase == 'checks':
        for modern in (False, True):
            name = f'prefill-closed-{a.label}-check' + ('-torch213' if modern else '')
            cmd = prefix.copy()
            cmd[1:1] = settings
            if modern:
                cmd[cmd.index('--python') + 1] = '/home/ubuntu/lod-h100/.venv/bin/python'
            submit(name, cmd + ['benchmarks.h100_threshold_prefill_closed_check', '--backend', 'blas',
                               '--output', str(results / (name + '.json'))] + (['--kernel-only'] if modern else []))
        (results / f'prefill-closed-{a.label}-checks-runs.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
        print(json.dumps(jobs, indent=2), flush=True)
        return
    modes = ('top8', 'region-legacy', 'entry-legacy') if a.phase == 'controls' else ('region-closed', 'region-fast-native')
    if a.closed_only and a.phase != 'controls':
        modes = ('region-closed',)
    for length in (32768, 65536):
        for mode in modes:
            name = f'prefill-closed-{a.label}-{mode}-{length // 1024}k'
            if a.phase == 'profiles':
                name += '-profile'
            cmd = prefix.copy()
            if mode.endswith('legacy'):
                rule = mode.split('-')[0]
                cmd[1:1] = [x.replace('LOD_H100_PREFILL_RULE=entry', f'LOD_H100_PREFILL_RULE={rule}') for x in legacy]
                if rule == 'region':
                    cmd = [x for x in cmd if not x.startswith(('LOD_H100_PREFILL_PACK_DIRECT=', 'LOD_H100_PREFILL_NO_HOST_SYNC=',
                        'LOD_H100_PREFILL_PACKED=', 'LOD_H100_PREFILL_TILE_QUERIES='))]
                    cmd.insert(1, 'LOD_H100_PREFILL_TILE_QUERIES=4096')
            elif mode != 'top8':
                cmd[1:1] = settings + ['LOD_H100_PREFILL_RULE=region']
                if mode == 'region-closed':
                    cmd.insert(1, 'LOD_H100_PREFILL_IMPLEMENTATION=closed')
                else:
                    cmd[1:1] = ['LOD_H100_PREFILL_IMPLEMENTATION=fast', 'LOD_H100_FAST_LEAF=native',
                                'LOD_H100_FAST_SAVED_CORRECTION=1', 'LOD_H100_FAST_TILE=4096']
            if a.phase == 'profiles':
                cmd.insert(1, 'LOD_H100_PREFILL_PROFILE=1')
            call = base.copy()
            call.append('--finish-lod-prefill-cache')
            for key, val in [('--length', str(length)), ('--repeats', '1' if a.phase == 'profiles' else '3'),
                             ('--warmups', '1'), ('--output', str(results / (name + '.json')))]:
                call[call.index(key) + 1] = val
            submit(name, cmd + ['benchmarks.h100_comparison'] + call[3:])
    if a.phase == 'candidates':
        for mode in modes:
            name = f'prefill-closed-{a.label}-{mode}-64k-check'
            source = next(j for j in jobs if j['name'] == f'prefill-closed-{a.label}-{mode}-64k')
            cmd = json.loads((state / source['id'] / 'run.json').read_text())['command']
            cmd[1:1] = ['LOD_H100_CLOSED_VERIFY=1'] if mode == 'region-closed' else ['LOD_H100_FAST_VERIFY=1']
            for key, val in [('--warmups', '0'), ('--repeats', '1'), ('--output', str(results / (name + '.json')))]:
                cmd[cmd.index(key) + 1] = val
            submit(name, cmd)
    (results / f'prefill-closed-{a.label}-{a.phase}-runs.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
