"""Validate and time AMD-derived H100 candidates including amortized updates.

Requires complete fixed-query tuning before selecting settings. Every GPU run
uses uv and cluster-run, with immutable source and success prerequisites.
"""
import argparse
import collections
import json
from pathlib import Path
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='transfer-v1')
    p.add_argument('--tune-phase', default='materialized-v1')
    a = p.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home() / '.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    prior = json.loads((state / '1794247723800191231-threshold-decode-v4-region-fast-32k' / 'run.json').read_text())['command']
    prefix = [x for x in prior[:prior.index('benchmarks.h100_checked_run')]
              if not x.startswith(('LOD_H100_THRESHOLD_IMPLEMENTATION=', 'LOD_H100_THRESHOLD_SPLITS='))]
    model_command = prior[prior.index('--', prior.index('benchmarks.h100_checked_run')) + 1:]
    cluster = [str(Path.home() / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
               'python', '-m', 'benchmarks.h100_cluster_run']
    tune_jobs = json.loads((results / f'threshold-{a.tune_phase}-runs.json').read_text())['jobs']
    for job in tune_jobs:
        meta = json.loads((state / job['id'] / 'run.json').read_text())
        if meta.get('exit_code') != 0 or meta['state'] != 'finished':
            raise ValueError('tuning has not passed: ' + job['name'])
    chosen = {}
    for length in (32768, 65536):
        data = json.loads((results / f'threshold-{a.tune_phase}-tune-{length // 1024}k.json').read_text())
        if data['stage'] != 'complete':
            raise ValueError('tuning incomplete')
        groups = collections.defaultdict(list)
        for row in data['records']:
            if row['rule'] == 'region':
                groups[row['implementation'], row['splits'], row['block'], row['merge_block']].append(row['microseconds'])
        if any(len(v) != 3 for v in groups.values()):
            raise ValueError('missing real-layer tuning cases')
        candidates = [k for k in groups if k[0] != 'indexed' or k[3] != 0]
        winner = min(candidates, key=lambda k: statistics.mean(groups[k]))
        chosen[str(length)] = dict(implementation=winner[0], splits=winner[1], block=winner[2], merge_block=winner[3],
                                   fixed_query_us=statistics.mean(groups[winner]),
                                   control_fixed_query_us=statistics.mean(groups['indexed', 64, 32, 0]))
    jobs = []

    def submit(name, command):
        output = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + command, cwd=root, text=True)
        rid = output.splitlines()[0].split()[-1]
        jobs.append(dict(name=name, id=rid))
        return rid

    def command(length, mode, check=False, requirements=()):
        cfg = chosen[str(length)] if mode == 'candidate' else dict(implementation='indexed', splits=64, block=32, merge_block=0)
        name = f'threshold-{a.label}-{mode}-{length // 1024}k' + ('-check' if check else '')
        cmd = prefix.copy()
        cmd[1:1] = [f'LOD_H100_THRESHOLD_IMPLEMENTATION={cfg["implementation"]}', f'LOD_H100_THRESHOLD_SPLITS={cfg["splits"]}',
                    f'LOD_H100_SHARED_ENTRY_BLOCK={cfg["block"]}', f'LOD_H100_THRESHOLD_MERGE_BLOCK={cfg["merge_block"]}',
                    f'LOD_AMD_THRESHOLD_ENTRY_BLOCK={cfg["block"]}', 'LOD_AMD_THRESHOLD_ENTRY_WARPS=4',
                    f'LOD_AMD_THRESHOLD_MERGE_BLOCK={cfg["merge_block"] or 32}', 'LOD_AMD_THRESHOLD_PRECISE=1']
        if check:
            cmd.insert(1, 'LOD_H100_DECODE_CHECK_REFERENCE=1')
        call = model_command.copy()
        for flag, value in [('--length', length), ('--output', results / (name + '.json'))]:
            call[call.index(flag) + 1] = str(value)
        if check:
            for flag, value in [('--tokens', 514), ('--warmups', 0), ('--repeats', 1), ('--decode-warmup-steps', 0)]:
                call[call.index(flag) + 1] = str(value)
            call.append('--check-graph-same-cache')
        cmd += ['benchmarks.h100_checked_run']
        cmd += [x for rid in requirements for x in ('--require-run', str(state / rid / 'run.json'))]
        cmd += ['--'] + call
        return name, cmd

    prerequisites = [j['id'] for j in tune_jobs]
    checks = {}
    for length in (32768, 65536):
        checks[length] = submit(*command(length, 'candidate', check=True, requirements=prerequisites))
    for length in (32768, 65536):
        for mode in ('control', 'candidate', 'control-after'):
            submit(*command(length, mode, requirements=(checks[length],)))
    (results / f'threshold-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs, chosen=chosen), indent=2) + '\n')
    print(json.dumps(dict(jobs=jobs, chosen=chosen), indent=2), flush=True)


if __name__ == '__main__':
    main()
