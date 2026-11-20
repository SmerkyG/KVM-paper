"""Submit shared-GQA decode correctness and tuning with uv/cluster-run snapshots."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='shared-v1')
    p.add_argument('--four-stage', action='store_true')
    a = p.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home() / '.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    prior = json.loads((state / '1794247723800191231-threshold-decode-v4-region-fast-32k' / 'run.json').read_text())['command']
    prefix = [x for x in prior[:prior.index('benchmarks.h100_checked_run')] if not x.startswith(('LOD_H100_THRESHOLD_IMPLEMENTATION=', 'LOD_H100_THRESHOLD_SPLITS='))]
    prefix[1:1] = ['LOD_H100_THRESHOLD_MERGE_BLOCK=32', 'LOD_AMD_THRESHOLD_PRECISE=1']
    cluster = [str(Path.home() / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
               'python', '-m', 'benchmarks.h100_cluster_run']
    jobs = []

    def submit(name, command):
        output = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command, cwd=root, text=True)
        rid = output.splitlines()[0].split()[-1]
        jobs.append(dict(name=name, id=rid))
        return rid

    name = f'threshold-{a.label}-check'
    check = submit(name, prefix + ['benchmarks.h100_threshold_decode_check', '--implementation',
                                  'materialized' if a.four_stage else 'shared', '--output', str(results / (name + '.json'))])
    for length in (32768, 65536):
        name = f'threshold-{a.label}-tune-{length // 1024}k'
        command = prefix + ['benchmarks.h100_checked_run', '--require-run', str(state / check / 'run.json'), '--',
                            'python', '-m', 'benchmarks.h100_threshold_shared_tune', '--manifest', '/home/ubuntu/lod-h100/results/prompts-b1.json',
                            '--length', str(length), '--output', str(results / (name + '.json'))]
        if a.four_stage:
            command += ['--four-stage', '--rules', 'region']
        submit(name, command)
    (results / f'threshold-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
