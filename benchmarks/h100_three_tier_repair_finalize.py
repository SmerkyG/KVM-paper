"""Collect completed repair panels and publish their checked, first-response tables.

CPU bookkeeping only: --watch waits for the queued cluster-run analyses before
collecting results. Failed dependencies stop publication; partial panels never
receive a RULER average. No experiment or serving setting is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time


TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'fwe', 'cwe', 'qa_1', 'qa_2')
GROUPS = {
    'Dense': ['ruler-32k-dense-full.json', 'ruler-32k-dense-missing8.json'],
    'RetroInfer': ['ruler-llama-retroinfer-32k-selected.json', 'ruler-llama-retroinfer-32k-missing5.json'],
    'Two-tier c16': ['three-tier-repair-v1-two-tier-control.json',
                     'three-tier-repair-mq32-v1-two-tier-control.json',
                     'three-tier-repair-twotier-full11-v2-two-tier-control.json'],
    'Three-tier four pages': ['three-tier-repair-v4-pages4-quality.json',
                              'three-tier-repair-remaining12-v1-pages4-quality.json'],
    'Three-tier page threshold, region c16': ['three-tier-repair-normpage-v1-pagec0p5-quality.json',
                                             'three-tier-repair-normpage-full11-v3-pagec0p5-quality.json'],
    'Three-tier page threshold, region c8': ['three-tier-repair-joint8-v1-pagec0p5-quality.json',
                                            'three-tier-repair-joint8-full11-v1-pagec0p5-quality.json'],
}
REMOTE_STATUS = r'''
from pathlib import Path
import json
r=Path('/home/ubuntu/comparison/results');s=Path.home()/'.local/state/lod-cluster-run'
jobs=json.loads((r/'three-tier-repair-final-analysis-runs.json').read_text())['jobs']
print(json.dumps([dict(id=j['id'],**{k:v for k,v in json.loads((s/j['id']/'run.json').read_text()).items() if k in ('state','exit_code')}) for j in jobs]))
'''


def publish(root, repo):
    audit = json.loads((root/'three-tier-repair-first-response-audit.json').read_text())
    assert audit['stage'] == 'complete'
    summary = dict(stage='complete', quality={}, model={}, input_sha256={},
                   quality_scope='32K BF16 Llama-3.1-8B-Instruct; B=1; 32 examples/task; first EOS/EOT')
    for label, files in GROUPS.items():
        scores = {}; seen = set()
        for name in files:
            raw = (root/name).read_bytes(); data = json.loads(raw)
            digest = hashlib.sha256(raw).hexdigest(); saved = audit['files'][name]
            assert data['stage'] == saved['stage'] == 'complete'
            assert saved['sha256'] == digest
            if 'page threshold' in label:
                route = data['threshold_routing']; page = route['page_threshold']
                assert route['mass_multiplier'] == (8 if label.endswith('c8') else 16)
                assert route['rule'] == 'region' and route['pages_per_region'] is None
                assert page['rule'] == 'page' and page['multiplier'] == 0.5
                assert page['query'] == 'normalized' and page.get('residual', 'region') == 'region'
            summary['input_sha256'][name] = digest
            for row in saved['records']:
                key = row['task'], row['source_row']
                assert key not in seen
                seen.add(key)
            for task, score in saved['task_scores'].items():
                assert task not in scores and score['n'] == 32
                scores[task] = score['first']
        assert len(seen) == 416 and set(scores) == set(TASKS)
        summary['quality'][label] = dict(tasks=scores, macro_average=statistics.mean(scores.values()), files=files)
    for label, prefix in [('Two pages c16', 'v5-pages2'), ('Four pages c16', 'v5-pages4'),
                          ('Page threshold c16', 'normpage-model-v1-pagec0p5'),
                          ('Page threshold c8', 'joint8-model-v1-pagec0p5')]:
        summary['model'][label] = {}
        for ctx in (32, 64):
            name = f'three-tier-repair-{prefix}-{ctx}k-model.json'
            raw = (root/name).read_bytes(); data = json.loads(raw)
            rows = [x for x in data['records'] if not x['warmup']]
            assert data['lod_cuda_graph'] and data['batch_size'] == 1
            assert data['decode_warmup_steps'] == 512 and data['timed_decode_steps'] == 1024
            assert len(rows) == 2 and all(x['lod_layer_catch_up_events_timed'] == 128 for x in rows)
            assert abs(statistics.median(x['decode_ms_per_token'] for x in rows)-data['median_decode_ms_per_token']) < 1.e-8
            summary['input_sha256'][name] = hashlib.sha256(raw).hexdigest()
            summary['model'][label][f'{ctx}K'] = dict(ms_per_token=data['median_decode_ms_per_token'],
                prefill_s=data['median_prefill_s'], file=name)
            if prefix.startswith(('normpage', 'joint8')):
                check = json.loads((root/f'three-tier-repair-{prefix}-{ctx}k-model-check.json').read_text())
                checks = check['records'][0]['graph_same_cache_checks']
                assert len(checks) == 17 and all(x['max_abs'] == 0 and x['graph_argmax'] == x['eager_argmax'] for x in checks.values())
    status = json.loads((root/'status.json').read_text())
    by_name = {x['name']: x for x in status}
    hardware = set()
    for name in summary['input_sha256']:
        if name.startswith('three-tier-repair-'):
            job = by_name[name.removesuffix('.json')]
            assert job['state'] == 'finished' and job['exit_code'] == 0
            metadata = json.loads((root/'provenance'/job['id']/'run.json').read_text())
            hardware.add(metadata['hardware'])
    assert len(hardware) == 1
    summary['hardware'] = next(iter(hardware))
    summary['excluded_runs'] = [x for x in status if x['state'] == 'finished' and x['exit_code'] != 0]
    (root/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    q = summary['quality']
    labels = list(q)
    heading = ['Dense', 'RetroInfer', 'Two-tier c16', 'Four pages', 'Page threshold c16', 'Page threshold c8']
    lines = ['## Full RULER panel', '',
        'All six configurations have complete 13-task panels, with 32 matched examples per task. '
        'Values are first-response percentages; the average weights tasks equally. '
        'Both adaptive page columns use normalized page queries and page multiplier 0.5.', '',
        '| Task | '+' | '.join(heading)+' |', '|---|'+'---:|'*len(labels)]
    for task in TASKS:
        lines.append('| '+task+' | '+' | '.join(f"{q[label]['tasks'][task]:.2f}" for label in labels)+' |')
    lines += ['| **RULER average** | '+' | '.join(f"**{q[label]['macro_average']:.2f}**" for label in labels)+' |', '',
        'The two-tier column is a fresh rerun of the optimized materialized decoder. '
        'Its older 87.04% result is not substituted for this control. RetroInfer uses dense '
        'prefill; these compare complete method configurations, not decode alone with identical prefill. '
        'The 32-example task size limits precision. Page settings were selected using multikey-3 '
        'and multi-query, which are reused here; this is not an independent holdout evaluation.', '',
        'Paired comparisons and bootstrap intervals: '
        '[four pages](three-tier-repair-four-page-full13-summary.md), '
        '[page threshold with region c16](three-tier-repair-normpage-full13-summary.md), '
        '[page threshold with region c8](three-tier-repair-joint8-full13-summary.md).', '',
        'The complete first-response audit and SUMMARY.json retain raw file hashes and per-task values.', '']
    report = root/'REPORT.md'; text = report.read_text()
    start = text.index('## Full RULER panel'); end = text.index('## Complete attention-call timings', start)
    text = text[:start]+'\n'.join(lines)+'\n'+text[end:]
    text = text.replace('Full-panel checks for the two threshold candidates are in progress.',
                        'Both page-threshold candidates now have complete 13-task panels.')
    report.write_text(text)
    doc = repo/'benchmarks/RETRO_LOD.md'; text = doc.read_text()
    c16 = q['Three-tier page threshold, region c16']['macro_average']
    c8 = q['Three-tier page threshold, region c8']['macro_average']
    text = text.replace('Both full RULER panels are pending.',
        f'The full 13-task first-response averages are {c16:.2f}% at c=16 and {c8:.2f}% at c=8, '
        f"versus {q['Two-tier c16']['macro_average']:.2f}% for the fresh two-tier control.")
    doc.write_text(text)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--watch', action='store_true'); a = p.parse_args()
    root = a.directory.resolve(); root.mkdir(parents=True, exist_ok=True)
    state_file = root/'finalization-status.json'
    try:
        failures = 0
        while True:
            try:
                raw = subprocess.check_output(['ssh', '-o', 'StrictHostKeyChecking=no', a.host,
                    '/home/ubuntu/.local/bin/uv', 'run', '--offline', '--no-project', '--python', '/usr/bin/python3',
                    'python', '-c', shlex.quote(REMOTE_STATUS)], timeout=45)
                failures = 0
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                failures += 1
                if not a.watch or failures >= 4: raise
                state_file.write_text(json.dumps(dict(stage='waiting', connection_error=str(e), retry=failures), indent=2)+'\n')
                time.sleep(30)
                continue
            jobs = json.loads(raw); assert len(jobs) == 3
            if any(j['state'] == 'finished' and j['exit_code'] != 0 for j in jobs):
                raise RuntimeError('A required analysis failed; results will not be published: '+str(jobs))
            if all(j['state'] == 'finished' for j in jobs): break
            state_file.write_text(json.dumps(dict(stage='waiting', jobs=jobs), indent=2)+'\n')
            if not a.watch:
                print(json.dumps(dict(stage='waiting', jobs=jobs))); return
            time.sleep(30)
        subprocess.run([sys.executable, '-m', 'benchmarks.h100_three_tier_repair_collect',
                        '--host', a.host, '--directory', str(root)], check=True)
        repo = Path(__file__).resolve().parents[1]
        for files in list(GROUPS.values())[:2]:
            for name in files:
                shutil.copyfile(repo/'results/h100-baselines/ruler-32k-panel'/name, root/name)
        summary = publish(root, repo)
        state_file.write_text(json.dumps(dict(stage='complete', jobs=jobs,
            averages={k:v['macro_average'] for k,v in summary['quality'].items()}), indent=2)+'\n')
        print(state_file.read_text())
    except Exception as e:
        state_file.write_text(json.dumps(dict(stage='failed', error=str(e)), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
