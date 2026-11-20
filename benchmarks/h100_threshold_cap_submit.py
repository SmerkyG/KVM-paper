"""Compare entry-16 capped at eight or sixteen regions with original top-8.

Run on the H100 source checkout with uv. Every GPU job uses the logged
cluster-run adapter, with validation prerequisites and immutable source.
The exact first prefill chunk and remote cache-update schedule are unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='cap16')
    args = parser.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home() / '.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    prior = json.loads((results / 'threshold-fast-v6-runs.json').read_text())['jobs']
    cluster = [str(Path.home() / '.local/bin/uv'), 'run', '--offline', '--no-project',
               '--python', '/usr/bin/python3', 'python', '-m', 'benchmarks.h100_cluster_run']
    jobs = []

    def previous(mode, length):
        job = next(j for j in prior if j['name'] == f'threshold-fast-v6-{mode}-{length}k')
        return json.loads((state / job['id'] / 'run.json').read_text())['command']

    def submit(name, command):
        output = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command,
                                         cwd=root, text=True)
        run_id = output.splitlines()[0].split()[-1]
        jobs.append({'name': name, 'id': run_id})
        return run_id

    def configure(mode, length, *, verify=False, required=()):
        name = f'threshold-fast-{args.label}-{mode}-{length}k' + ('-audit' if verify else '')
        command = previous('cap8' if mode != 'top8' else 'top8', length).copy()
        if mode == 'cap16':
            command = [x.replace('LOD_H100_PREFILL_MAX_REGIONS=8', 'LOD_H100_PREFILL_MAX_REGIONS=16')
                       for x in command]
        if verify:
            command[1:1] = ['LOD_H100_FAST_VERIFY=1']
            command[command.index('--warmups') + 1] = '0'
            command[command.index('--repeats') + 1] = '1'
        command[command.index('--output') + 1] = str(results / (name + '.json'))
        index = command.index('benchmarks.h100_checked_run') + 1
        command[index:index] = [x for run_id in required for x in
                                ('--require-run', str(state / run_id / 'run.json'))]
        return submit(name, command)

    smoke_base = previous('cap8', 32)
    smoke_base = smoke_base[:smoke_base.index('benchmarks.h100_checked_run')]
    name = f'threshold-fast-{args.label}-smoke'
    smoke = submit(name, smoke_base + ['benchmarks.h100_threshold_fast_check', '--output', str(results / (name + '.json'))])
    checks = {mode: configure(mode, 64, verify=True, required=(smoke,)) for mode in ('cap8', 'cap16')}
    for length in (32, 64):
        for mode in ('cap8', 'cap16'):
            configure(mode, length, required=(checks[mode],))
        configure('top8', length, required=(smoke,))
    manifest = results / f'threshold-fast-{args.label}-runs.json'
    manifest.write_text(json.dumps({'jobs': jobs}, indent=2) + '\n')
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
