"""Submit H100 packing/leaf-geometry pilots and PyTorch-version checks."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='closed-v3')
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
    for metadata in ('scan', 'atomic'):
        name = f'prefill-{a.label}-{metadata}-check'
        cmd = prefix + ['benchmarks.h100_threshold_prefill_closed_check', '--backend', 'blas', '--output', str(results / (name + '.json'))]
        cmd[1:1] = [f'LOD_H100_CLOSED_METADATA={metadata}', 'LOD_H100_CLOSED_LEAF_M=16', 'LOD_H100_CLOSED_LEAF_N=16', 'LOD_H100_CLOSED_LEAF_WARPS=2']
        checks[metadata] = submit(name, cmd)
    name = f'prefill-{a.label}-torch213-check'
    cmd = prefix.copy()
    cmd[cmd.index('--python') + 1] = '/home/ubuntu/lod-h100/.venv/bin/python'
    cmd[1:1] = ['LOD_H100_CLOSED_METADATA=atomic', 'LOD_H100_CLOSED_LEAF_M=16', 'LOD_H100_CLOSED_LEAF_N=16', 'LOD_H100_CLOSED_LEAF_WARPS=2']
    submit(name, cmd + ['benchmarks.h100_threshold_prefill_closed_check', '--backend', 'blas', '--kernel-only', '--output', str(results / (name + '.json'))])
    cmd = prefix[:-1] + ['-c', "from benchmarks.retro_lod_adapter import install_lod_modules; install_lod_modules(); import pytest; raise SystemExit(pytest.main(['-q','tests/test_cuda_prefill.py','tests/test_page_opening.py']))"]
    cmd[1:1] = ['LOD_RUN_GPU_TESTS=1']
    submit(f'prefill-{a.label}-native-tests', cmd)
    base = prior[prior.index('--', prior.index('benchmarks.h100_checked_run')) + 1:]
    configurations = [('scan-native', 'scan', 64, 32, 4, 8), ('atomic-native', 'atomic', 64, 32, 4, 8),
                      ('scan-small', 'scan', 16, 16, 2, 8), ('atomic-small', 'atomic', 16, 16, 2, 8),
                      ('atomic-mid', 'atomic', 32, 16, 4, 8), ('atomic-small-w16', 'atomic', 16, 16, 2, 16)]
    for rule in ('region', 'entry'):
        for label, metadata, m, n, warps, select_warps in configurations:
            name = f'prefill-{a.label}-{rule}-{label}-32k'
            cmd = prefix.copy()
            cmd[1:1] = [f'LOD_H100_PREFILL_RULE={rule}', 'LOD_H100_PREFILL_IMPLEMENTATION=closed', 'LOD_H100_CLOSED_MATMUL=blas',
                        f'LOD_H100_CLOSED_METADATA={metadata}', f'LOD_H100_CLOSED_LEAF_M={m}', f'LOD_H100_CLOSED_LEAF_N={n}',
                        f'LOD_H100_CLOSED_LEAF_WARPS={warps}', f'LOD_H100_CLOSED_SELECT_WARPS={select_warps}']
            call = base.copy()
            for key, val in [('--length', '32768'), ('--repeats', '1'), ('--warmups', '1'), ('--output', str(results / (name + '.json')))]:
                call[call.index(key) + 1] = val
            cmd += ['benchmarks.h100_checked_run', '--require-run', str(state / checks[metadata] / 'run.json'), '--'] + call
            submit(name, cmd)
    (results / f'prefill-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
