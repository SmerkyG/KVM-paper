"""Freeze grouped selection/reduction checks and H100 full-model pilots."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='closed-v4')
    p.add_argument('--checks-only', action='store_true')
    p.add_argument('--packed', action='store_true')
    p.add_argument('--leaf-m', type=int, default=64)
    p.add_argument('--leaf-n', type=int, default=32)
    p.add_argument('--leaf-warps', type=int, default=4)
    p.add_argument('--exact', choices=('grouped', 'indexed'), default='grouped')
    p.add_argument('--fast-native', action='store_true')
    a = p.parse_args()
    root = Path('/home/ubuntu/lod-h100/code')
    state = Path.home() / '.local/state/lod-cluster-run'
    results = Path('/home/ubuntu/comparison/results')
    prior = json.loads((state / '1794244954699735307-threshold-fast-v6-legacy-64k/run.json').read_text())['command']
    prefix = [x for x in prior[:prior.index('benchmarks.h100_checked_run')]
              if not x.startswith(('LOD_H100_PREFILL_', 'LOD_H100_FAST_'))]
    cluster = [str(Path.home() / '.local/bin/uv'), 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
               'python', '-m', 'benchmarks.h100_cluster_run']
    jobs = []

    def submit(name, cmd):
        output = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--'] + cmd,
                                         cwd=root, text=True)
        rid = output.splitlines()[0].split()[-1]
        jobs.append(dict(name=name, id=rid))
        return rid

    configurations = [('atomic-m1', 'atomic', 1, 4), ('atomic-m2', 'atomic', 2, 4),
                      ('atomic-m4', 'atomic', 4, 4), ('scan-m4', 'scan', 4, 4)]
    if a.exact == 'indexed':
        configurations = [('indexed', 'none', 1, 4)]
    if a.fast_native:
        configurations = [('fast-native', 'atomic', 1, 4)]
    checks = {}
    for label, metadata, sm, rm in configurations:
        options = [f'LOD_H100_CLOSED_METADATA={metadata}', f'LOD_H100_CLOSED_SELECT_QM={sm}',
                   'LOD_H100_CLOSED_REDUCE=grouped', f'LOD_H100_CLOSED_REDUCE_QM={rm}',
                   f'LOD_H100_CLOSED_LEAF_M={a.leaf_m}', f'LOD_H100_CLOSED_LEAF_N={a.leaf_n}',
                   f'LOD_H100_CLOSED_LEAF_WARPS={a.leaf_warps}', f'LOD_H100_CLOSED_PACKED_OUTPUT={int(a.packed)}',
                   f'LOD_H100_CLOSED_EXACT={a.exact}']
        name = f'prefill-{a.label}-{label}-check'
        cmd = prefix.copy()
        cmd[1:1] = options
        if a.fast_native:
            cmd[1:1] = ['LOD_H100_FAST_LEAF=native', 'LOD_H100_FAST_SAVED_CORRECTION=1']
            check_ids = []
            for rule in ('region', 'entry'):
                n = name + '-' + rule
                check_ids.append(submit(n, cmd + ['benchmarks.h100_threshold_fast_check', '--rule', rule,
                                                 '--output', str(results / (n + '.json'))]))
            checks[label] = check_ids
        else:
            checks[label] = [submit(name, cmd + ['benchmarks.h100_threshold_prefill_closed_check', '--backend', 'blas',
                                               '--output', str(results / (name + '.json'))])]
    if not a.checks_only:
        base = prior[prior.index('--', prior.index('benchmarks.h100_checked_run')) + 1:]
        for rule in ('region', 'entry'):
            for label, metadata, sm, rm in configurations:
                name = f'prefill-{a.label}-{rule}-{label}-32k'
                cmd = prefix.copy()
                cmd[1:1] = [f'LOD_H100_PREFILL_RULE={rule}', 'LOD_H100_PREFILL_IMPLEMENTATION=closed',
                           'LOD_H100_CLOSED_MATMUL=blas', f'LOD_H100_CLOSED_METADATA={metadata}',
                           f'LOD_H100_CLOSED_SELECT_QM={sm}', 'LOD_H100_CLOSED_REDUCE=grouped',
                           f'LOD_H100_CLOSED_REDUCE_QM={rm}', f'LOD_H100_CLOSED_LEAF_M={a.leaf_m}',
                           f'LOD_H100_CLOSED_LEAF_N={a.leaf_n}', f'LOD_H100_CLOSED_LEAF_WARPS={a.leaf_warps}',
                           f'LOD_H100_CLOSED_PACKED_OUTPUT={int(a.packed)}', f'LOD_H100_CLOSED_EXACT={a.exact}']
                call = base.copy()
                if a.fast_native:
                    cmd = [x.replace('LOD_H100_PREFILL_IMPLEMENTATION=closed', 'LOD_H100_PREFILL_IMPLEMENTATION=fast') for x in cmd]
                    cmd[1:1] = ['LOD_H100_FAST_LEAF=native', 'LOD_H100_FAST_SAVED_CORRECTION=1']
                for key, val in [('--length', '32768'), ('--repeats', '1'), ('--warmups', '1'),
                                 ('--output', str(results / (name + '.json')))]:
                    call[call.index(key) + 1] = val
                cmd += ['benchmarks.h100_checked_run'] + [x for rid in checks[label] for x in ('--require-run', str(state / rid / 'run.json'))] + ['--'] + call
                submit(name, cmd)
    (results / f'prefill-{a.label}-runs.json').write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    main()
