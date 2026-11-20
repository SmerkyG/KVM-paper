"""Freeze and submit H100 closed-prefill checks and short model pilots."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='closed-v1')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--native-tests', action='store_true')
    a = p.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home() / '.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    prior = json.loads((state / '1794244954699735307-threshold-fast-v6-legacy-64k' / 'run.json').read_text())['command']
    prefix = [x for x in prior[:prior.index('benchmarks.h100_checked_run')] if not x.startswith(('LOD_H100_PREFILL_', 'LOD_H100_FAST_'))]
    cluster = [str(Path.home() / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python', '/usr/bin/python3', 'python', '-m', 'benchmarks.h100_cluster_run']
    jobs = []

    def submit(name, cmd):
        output = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd, cwd=root, text=True)
        rid = output.splitlines()[0].split()[-1]
        jobs.append(dict(name=name, id=rid))
        return rid

    checks = {}
    for backend in ('triton', 'blas'):
        name = f'prefill-{a.label}-{backend}-check'
        checks[backend] = submit(name, prefix + ['benchmarks.h100_threshold_prefill_closed_check', '--backend', backend, '--output', str(results / (name + '.json'))])
    if a.native_tests:
        cmd = prefix[:-1] + ['-c', "from benchmarks.retro_lod_adapter import install_lod_modules; install_lod_modules(); import pytest; raise SystemExit(pytest.main(['-q','tests/test_cuda_prefill.py','tests/test_page_opening.py']))"]
        cmd[1:1] = ['LOD_RUN_GPU_TESTS=1']
        submit(f'prefill-{a.label}-native-tests', cmd)
    base = prior[prior.index('--', prior.index('benchmarks.h100_checked_run')) + 1:]
    modes = [('top8', 'triton', False)] + [(mode, backend, False) for backend in ('triton','blas') for mode in ('region','entry')]
    if a.profile:
        modes += [(mode, 'blas', True) for mode in ('region','entry')]
    for mode, backend, profile in modes:
        name = f'prefill-{a.label}-{mode}-{backend}-32k' + ('-profile' if profile else '')
        cmd = prefix.copy()
        if mode != 'top8':
            cmd[1:1] = [f'LOD_H100_PREFILL_RULE={mode}', 'LOD_H100_PREFILL_IMPLEMENTATION=closed', f'LOD_H100_CLOSED_MATMUL={backend}']
        if profile:
            cmd.insert(1, 'LOD_H100_PREFILL_PROFILE=1')
        call = base.copy()
        for key, val in [('--length', '32768'), ('--repeats', '1'), ('--warmups', '1'), ('--output', str(results / (name + '.json')))]:
            call[call.index(key) + 1] = val
        cmd += ['benchmarks.h100_checked_run', '--require-run', str(state / checks[backend] / 'run.json'), '--'] + call
        submit(name, cmd)
    (results / f'prefill-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
