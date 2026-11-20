"""Freeze and submit final 64-worker three-tier tests via uv and cluster-run."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from benchmarks.amd_threshold_fast_submit import freeze
from benchmarks.amd_threshold_ruler_panel import digest, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/amd-threshold-three-tier'))
    args = parser.parse_args()
    root = args.root.resolve()
    phase = root / 'serving-v2'
    phase.mkdir()
    source = freeze(Path.cwd(), phase)
    previous = read(root / 'serving-v1/jobs.json')
    cohorts = read(root / 'serving-v1/cohort-manifest.json')
    for model in cohorts['models'].values():
        for item in model.values():
            if digest(Path(item['frozen'])) != item['sha256']:
                raise ValueError('changed frozen cohort')
    write(phase / 'cohort-manifest.json', cohorts)
    settings = read(root / 'serving-v1/settings.json')
    for options in settings.values():
        options['LOD_AMD_THRESHOLD_SPLITS'] = '64'
        options['LOD_AMD_THRESHOLD_PAGE_WORKERS'] = '64'
    write(phase / 'settings.json', settings)
    jobsets = {}

    def submit(destination, name, module_args, options, metadata):
        destination.mkdir(exist_ok=True)
        output = destination / (name + '.json')
        command = ['cluster-run', '--detach', '--num-gpus', '1', '--name', name, '--', 'env',
            f'PYTHONPATH={source}:{source}/integrations/vllm_lod', 'HF_HUB_OFFLINE=1', 'HF_DATASETS_OFFLINE=1',
            *(f'{k}={v}' for k, v in options.items()), 'uv', 'run', '--offline', '--no-project', '--python',
            '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python', 'python', *module_args, '--output', str(output)]
        result = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
        match = re.search(r'^Submitted (\S+)', result.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError(result.stdout + result.stderr)
        jobs = jobsets.setdefault(str(destination), {'source': str(source), 'jobs': []})
        jobs['jobs'].append(dict(id=match.group(1), name=name, command=command, output=str(output), options=options, **metadata))
        write(destination / 'jobs.json', jobs)
        print(match.group(1), flush=True)

    submit(root / 'check-v2', 'amd-three-tier-frontier-v2',
           ['-m', 'benchmarks.amd_threshold_three_tier_check'], {'LOD_AMD_THRESHOLD_PRECISE': '1'}, {})
    # pytest has no --output, so run its native CLI as a separate submission.
    command = ['cluster-run', '--detach', '--num-gpus', '1', '--name', 'amd-three-tier-native-pages-v2', '--', 'env',
        f'PYTHONPATH={source}:{source}/integrations/vllm_lod', 'LOD_RUN_GPU_TESTS=1', 'uv', 'run', '--offline', '--no-project',
        '--python', '/home/dan/subusers/agent/.venvs/vllm-rocm-0.27.1/bin/python', 'python', '-m', 'pytest', 'tests/test_page_opening.py', '-q']
    result = subprocess.run(command, cwd=source, text=True, capture_output=True, check=True)
    match = re.search(r'^Submitted (\S+)', result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError(result.stdout + result.stderr)
    check = jobsets[str(root / 'check-v2')]
    check['jobs'].append({'id': match.group(1), 'name': 'native-pages', 'command': command})
    write(root / 'check-v2/jobs.json', check)
    print(match.group(1), flush=True)
    for model in ('qwen', 'k2'):
        for mode in ('three-tier-bf16', 'three-tier-int4'):
            options = {**settings[model], 'LOD_AMD_THRESHOLD_PROBE_RULES': 'region', 'LOD_AMD_THRESHOLD_PROBE_NATIVE': '1',
                       'LOD_AMD_THRESHOLD_PROBE_CONFIGS': json.dumps([['three-tier', 64, 64, 16, int(settings[model]['LOD_AMD_THRESHOLD_PAGE_WARPS'])]])}
            submit(root / 'probe-v3', f'amd-3t-{model}-{mode}-probe-v3',
                   ['-m', 'benchmarks.amd_threshold_fast_probe', '--model', model, '--mode', mode, '--length', '65536'],
                   options, {'model': model, 'mode': mode})
    controls = [dict(j, origin_phase='serving-v1') for j in previous['jobs'] if j['phase']=='quality' and j['variant']=='top8']
    if len(controls) != 8:
        raise ValueError('missing native quality controls')
    jobsets[str(phase)] = {'source': str(source), 'reused_controls_source': previous['source'], 'jobs': controls, 'stage': 'submitting'}
    for model, checkpoint in [('qwen', 'Qwen/Qwen3.8-27B-FP8'), ('k2', 'IFM/K2-Horizon-32B-FP8')]:
        for mode in ('three-tier-bf16', 'three-tier-int4'):
            for task in cohorts['tasks']:
                cohort_root = Path(cohorts['models'][model][task]['frozen']).parents[1]
                submit(phase, f'amd-3t-{model}-{mode}-{task}-v2',
                    ['-m', 'benchmarks.amd_threshold_ruler', '--checkpoint', checkpoint, '--mode', mode,
                     '--samples-root', str(cohort_root), '--tasks', task, '--selection', 'prefix', '--limit', '32'],
                    settings[model], dict(phase='quality', model=model, mode=mode, task=task, variant='threshold'))
    for model in ('qwen', 'k2'):
        for mode in ('three-tier-bf16', 'three-tier-int4'):
            submit(phase, f'amd-3t-{model}-{mode}-speed-v2',
                ['-m', 'benchmarks.amd_threshold_fast_sequence', '--model', model, '--mode', mode,
                 '--variants', 'top8', 'fast', 'top8-after', '--settings-json', json.dumps(settings[model])],
                {}, dict(phase='speed', model=model, mode=mode))
    jobsets[str(phase)]['stage'] = 'submitted'
    write(phase / 'jobs.json', jobsets[str(phase)])


if __name__ == '__main__':
    main()
