"""Queue Qwen3 long-context quality and connected decode comparisons on H100."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('quality', 'shared', 'index-check', 'reference', 'pack-check', 'rope-check', 'prepare'), required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--variants', nargs='+', default=['lod', 'dense'])
    p.add_argument('--length', type=int, default=128000)
    p.add_argument('--limit-per-task', type=int, default=8)
    p.add_argument('--all-tasks', action='store_true', help='Generate all 13 tasks in the prepare stage')
    p.add_argument('--reuse-source-dir', type=Path, help='Reuse raw task JSONL in the prepare stage')
    p.add_argument('--cohort', type=Path, default=Path('/home/ubuntu/comparison/results/qwen-long-128k-cohort-v2.json'))
    p.add_argument('--require-run', action='append', default=[])
    p.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
    p.add_argument('--shared-args', nargs=argparse.REMAINDER, default=[])
    a = p.parse_args()
    if (a.all_tasks or a.reuse_source_dir) and a.stage != 'prepare':
        p.error('--all-tasks and --reuse-source-dir are for the prepare stage')
    repo = Path(__file__).resolve().parents[1]
    home = Path.home()
    state = home / '.local/state/lod-cluster-run'
    root = home / 'comparison'
    results = root / 'results'
    manifest = results / f'qwen-long-{a.label}-{a.stage}-runs.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    source = json.loads((results / 'qwen-opt-scale-v1-models-runs.json').read_text())
    selected = next(j for j in source['jobs'] if j['variant'] == 'optimized' and j['length'] == 128000)
    meta = json.loads((state / selected['id'] / 'run.json').read_text())
    assert meta['state'] == 'finished' and meta['exit_code'] == 0
    template = {}
    for item in selected['command'][1:]:
        if '=' not in item:
            break
        key, value = item.split('=', 1)
        template[key] = value
    template['RETRO_PREFILL_MLP_CHUNK'] = '8192'
    template['LOD_H100_FAST_TILE'] = '4096' if a.length > 65536 else 'auto'
    template['MULTIPOLE_STABLE_PACK'] = '1'
    template['SQUEEZED_BLOCK_M'] = '16'
    template['SQUEEZED_STABLE_PACK'] = '1'
    template['PYTHONPATH'] = f'{root}/squeezed-shared:{root}/MultipoleAttention:{root}/RetroInfer:.'
    for override in a.env:
        key, sep, value = override.partition('=')
        if not sep or not key.isidentifier():
            p.error('--env expects KEY=VALUE')
        template[key] = value
    uv = str(home / '.local/bin/uv')
    cluster = [uv, 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
               'python', '-m', 'benchmarks.h100_cluster_run']
    jobs = []
    factor = 2.0 if a.length <= 65536 else 4.0
    variants = a.variants if a.stage == 'quality' else ['cohort' if a.stage == 'prepare' else a.stage]
    for variant in variants:
        env = template.copy()
        python = [uv, 'run', '--offline', '--no-project', '--python', str(root / '.venv/bin/python'),
                  '--with', 'transformers==4.51.3', '--with', 'triton==3.7.1', 'python', '-m']
        if a.stage == 'quality':
            if variant not in ('lod', 'dense'):
                p.error('quality variants are lod and dense')
            if variant == 'dense':
                env = {k: v for k, v in env.items() if not k.startswith('LOD_')}
            module = 'benchmarks.retro_ruler_run'
            args = ['--input', a.cohort, '--mode', 'two-tier' if variant == 'lod' else 'dense',
                    '--model-name', 'Qwen/Qwen3-8B', '--prompt-format', 'qwen3-preformatted',
                    '--limit-per-task', a.limit_per_task, '--check-graph']
            if a.length > 32768:
                args += ['--qwen-yarn', '--qwen-yarn-factor', factor]
        elif a.stage == 'shared':
            module = 'benchmarks.h100_qwen_shared_decode'
            args = ['--length', a.length, *a.shared_args]
        elif a.stage == 'reference':
            module = 'benchmarks.retro_qwen3_long_check'
            args = ['--cohort', a.cohort, *a.shared_args]
        elif a.stage == 'pack-check':
            module = 'benchmarks.squeezed_stable_pack'
            args = []
        elif a.stage == 'rope-check':
            module = 'benchmarks.qwen3_rope'
            args = ['--factor', factor]
        elif a.stage == 'prepare':
            python = [uv, 'run', '--offline', '--no-project', '--python', str(root / '.venv/bin/python'),
                      '--with', 'transformers==4.51.3', '--with', 'wonderwords', '--with', 'nltk', 'python', '-m']
            module = 'benchmarks.retro_qwen3_prepare'
            args = ['--max-length', a.length, '--samples', a.limit_per_task, '--output-dir',
                    results / f'qwen-long-{a.label}-{a.length}-source']
            if a.all_tasks:
                python.remove('--offline')
                python[-2:-2] = ['--with', 'html2text', '--with', 'beautifulsoup4']
                args.append('--all-tasks')
            if a.reuse_source_dir:
                args += ['--reuse-source-dir', a.reuse_source_dir]
        else:
            python = [uv, 'run', '--offline', '--no-project', '--python', str(root / '.squeezed-venv/bin/python'),
                      'python', '-m']
            module = 'benchmarks.h100_squeezed_index'
            args = ['--probe-length', a.length]
        name = f'qwen-long-{a.label}-{variant}-{a.length}-{a.stage}'
        out = results / (name + '.json')
        cmd = ['env', *(f'{k}={v}' for k, v in env.items()), *python, 'benchmarks.h100_checked_run']
        cmd += [v for rid in a.require_run for v in ('--require-run', str(state / rid / 'run.json'))]
        cmd += ['--', 'python', '-m', module, *map(str, args), '--output', str(out)]
        response = subprocess.check_output(cluster + ['--name', name, '--detach', '--num-gpus', '1', '--', *cmd],
                                           cwd=repo, text=True)
        rid = response.splitlines()[0].split()[-1]
        jobs.append(dict(id=rid, name=name, variant=variant, length=a.length, command=cmd, output=str(out)))
        manifest.write_text(json.dumps(dict(jobs=jobs), indent=2) + '\n')
        print(response, flush=True)


if __name__ == '__main__':
    main()
