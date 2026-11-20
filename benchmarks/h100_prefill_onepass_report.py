"""Summarize H100 one-pass normalization and selection-packing experiments."""
import argparse
import inspect
import json
import statistics
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    args = p.parse_args()
    def load(name):
        path = args.directory / name
        return json.loads(path.read_text()) if path.exists() else None
    lines = ['# H100 threshold-prefill normalization and packing', '',
             'BF16 Llama-3.1-8B-Instruct, batch 1, entry-16 two-tier prefill. The existing '
             '16,384-token exact initial prefill chunk is unchanged. Every timed result is a '
             'full-model prefill median of three runs after one warmup, on the same H100 and prompt.', '',
             'The producer computes normalized-query routing scores and partial log-normalizers '
             'together, then selection reduces those partials and reads the score matrix once. '
             'Bitmask packing uses one bit per region. Direct packing writes selected region IDs '
             'to bounded per-query buffers and retains packed entry indices; it avoids the selection mask.', '',
             '| Variant | 32K prefill | 64K prefill |', '|---|---:|---:|']
    variants = [('Baseline, matched source', 'followup-baseline'),
                ('Producer, matched source', 'followup-producer'),
                ('Producer, initial snapshot', 'producer'),
                ('Bitmask', 'bitmask'), ('Producer + bitmask', 'combined'),
                ('Direct region packing', 'direct'), ('Producer + direct packing', 'producer-direct')]
    for label, name in variants:
        values = []
        for length in (32, 64):
            d = load(f'prefill-onepass-{name}-{length}k.json')
            values.append(f"{d['median_prefill_s']:.3f} s" if d else 'pending')
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    winner = []
    for length in (32, 64):
        d = load(f'prefill-direct-winner-{length}k.json')
        winner.append(f"{d['median_prefill_s']:.3f} s" if d else 'pending')
    lines.append(f'| Producer + direct packing, 64-region tiles | {winner[0]} | {winner[1]} |')
    followup_runs = load('prefill-onepass-followup-runs.json')
    if followup_runs:
        hashes = []
        for job in followup_runs['jobs']:
            if job['name'].endswith(('-32k', '-64k')):
                meta = load(f"provenance/{job['id']}/run.json")
                if meta and meta.get('state') == 'finished' and meta.get('exit_code') == 0:
                    hashes.append(meta['source_files'])
        if hashes and any(d != hashes[0] for d in hashes[1:]):
            raise ValueError('paired speed runs used different source snapshots')
    lines += ['', 'The bitmask and direct-packing variants were tested in separate logged source snapshots. '
              'The baseline and producer rows are repeated with the same snapshot; other rows should be '
              'read as separate short-run measurements. Earlier [top-8 LoD timings](../prefill-next/REPORT.md) '
              'were 1.656 s at 32K and 4.189 s at 64K; this experiment has not closed that gap.', '', 'Real-state checks:', '']
    profiles = [load('prefill-onepass-followup-' + name + '-profile.json') for name in ('baseline', 'producer')]
    if all(profiles):
        def stage(d, name):
            return statistics.median(sum(layer.get(name, 0.0) for layer in r['prefill_profile'])
                                     for r in d['records'] if not r['warmup'])
        route = [stage(d, 'route_logits') for d in profiles]
        coarse = [stage(d, 'select_and_coarse') for d in profiles]
        lines.insert(lines.index('Real-state checks:') - 1,
                     f'At 32K, synchronized diagnostic stage times summed over layers: routing score generation '
                     f'{route[0]:.3f} s baseline versus {route[1]:.3f} s producer; selection plus coarse '
                     f'{coarse[0]:.3f} s versus {coarse[1]:.3f} s. These include diagnostic synchronization '
                     f'and launch overhead and are not end-to-end GPU event measurements.')
    for name in ('producer', 'bitmask', 'combined', 'direct', 'producer-direct'):
        d = load(f'prefill-onepass-{name}-verify.json')
        if not d or d.get('stage') != 'complete':
            lines.append(f'- {name}: pending.')
            continue
        checks = [x for r in d['records'] for x in r['prefill_fused_validation']]
        heads = sum(x['query_heads'] for x in checks)
        mismatches = sum(x['selection_mismatches'] for x in checks)
        outside = sum(x['selection_outside_tolerance'] for x in checks)
        maximum = max(x['max_abs'] for x in checks)
        lines.append(f'- {name}: {len(d["records"])} RULER examples, {heads:,} query-heads, '
                     f'{mismatches} selection differences from the independent reference '
                     f'({outside} outside tolerance); largest output error {maximum:.6f}.')
    d = load('prefill-direct-winner-verify.json')
    if d and d.get('stage') == 'complete':
        checks = [x for r in d['records'] for x in r['prefill_fused_validation']]
        lines.append(f"- Producer + direct packing, 64-region tiles: {len(d['records'])} examples, "
                     f"{sum(x['query_heads'] for x in checks):,} query-heads, "
                     f"{sum(x['selection_mismatches'] for x in checks)} selection differences "
                     f"({sum(x['selection_outside_tolerance'] for x in checks)} outside tolerance); "
                     f"largest output error {max(x['max_abs'] for x in checks):.6f}.")
    else:
        lines.append('- Producer + direct packing, 64-region tiles: pending.')
    lines += ['', 'For producer-enabled variants, the check also compares their complete '
              'routing-logit tile against the original matmul bit-for-bit. Selected-region '
              'differences within 2e-5 of the decision boundary are reported, not hidden.', '']
    tok = Tokenizer.from_file(str(args.tokenizer))
    stops = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stops:
        raise ValueError('missing stop token')
    cleanup = PreTrainedTokenizerBase.clean_up_tokenization
    def first(r):
        ids = r['output_token_ids']
        end = next((i for i, t in enumerate(ids) if t in stops), len(ids))
        text = tok.decode(ids[:end], skip_special_tokens=True)
        text = cleanup(None, text) if len(inspect.signature(cleanup).parameters) == 2 else cleanup(text)
        return score(r['task'], text, r['answers'])
    lines += ['| Paired RULER task | Examples | Baseline | Producer | Improved | Declined |',
              '|---|---:|---:|---:|---:|---:|']
    control_manifest = load('prefill-onepass-quality-control-runs.json')
    followup_manifest = load('prefill-onepass-followup-runs.json')
    for task in ('qa_1', 'niah_multikey_2'):
        new = load('prefill-onepass-followup-' + task.replace('_', '-') + '.json')
        if not new or new.get('stage') != 'complete':
            lines.append(f'| {task} | pending | — | — | — | — |')
            continue
        old = load('prefill-onepass-control-' + task.replace('_', '-') + '.json')
        if not old or old.get('stage') != 'complete':
            lines.append(f'| {task} | pending control | — | — | — | — |')
            continue
        def source_files(manifest):
            rid = next(j['id'] for j in manifest['jobs'] if j.get('task') == task)
            meta = load(f'provenance/{rid}/run.json')
            if not meta or meta.get('state') != 'finished' or meta.get('exit_code') != 0:
                raise ValueError('missing successful quality run metadata')
            return {k: v for k, v in meta['source_files'].items()
                    if k != 'benchmarks/h100_prefill_onepass_quality_control.py'}
        if source_files(control_manifest) != source_files(followup_manifest):
            raise ValueError('quality runs used different execution source snapshots')
        if any(old.get(k) != new.get(k) for k in ('model_name', 'model_revision', 'cohort_sha256', 'generation')):
            raise ValueError('paired quality metadata differs')
        aa = {r['index']: r for r in old['records'] if r['task'] == task}
        bb = {r['index']: r for r in new['records'] if r['task'] == task}
        if len(aa) != 32 or aa.keys() != bb.keys():
            raise ValueError('quality inputs incomplete')
        if any(aa[i]['input_token_sha256'] != bb[i]['input_token_sha256'] or aa[i]['answers'] != bb[i]['answers'] for i in aa):
            raise ValueError('quality examples differ')
        before, after = [first(aa[i]) for i in aa], [first(bb[i]) for i in aa]
        lines.append(f'| {task} | 32 | {sum(before)*100/32:.2f}% | {sum(after)*100/32:.2f}% | '
                     f'{sum(y>x for x,y in zip(before,after))} | {sum(y<x for x,y in zip(before,after))} |')
    lines += ['', 'Scores stop at the first end-of-turn or end-of-text token. Paired quality runs '
              'use identical execution source files; their snapshots differ only in the submission helper. These two tasks '
              'are a limited quality sample, not the full RULER panel. Results, run commands, '
              'and source hashes are retained in this directory. The initial direct-packing '
              'checks failed because the diagnostic reader first treated a flat workspace as shaped and then inferred '
              'its current width from a larger retained allocation; '
              'dependent runs were skipped and resubmitted after that reader was corrected.', '']
    lines += ['| Mask-free paired RULER task | Examples | Baseline | Producer + direct, 64-region tiles | Improved | Declined |',
              '|---|---:|---:|---:|---:|---:|']
    winner_manifest = load('prefill-direct-winner-runs.json')
    for task in ('qa_1', 'niah_multikey_2'):
        old = load('prefill-onepass-control-' + task.replace('_', '-') + '.json')
        new = load('prefill-direct-winner-' + task.replace('_', '-') + '.json')
        if not old or not new or old.get('stage') != 'complete' or new.get('stage') != 'complete':
            lines.append(f'| {task} | pending | — | — | — | — |')
            continue
        def mask_free_sources(manifest):
            rid = next(j['id'] for j in manifest['jobs'] if j.get('task') == task)
            meta = load(f'provenance/{rid}/run.json')
            if not meta or meta.get('state') != 'finished' or meta.get('exit_code') != 0:
                raise ValueError('missing successful mask-free quality metadata')
            return {k: v for k, v in meta['source_files'].items() if k not in (
                'benchmarks/h100_prefill_onepass_debug_submit.py', 'benchmarks/h100_threshold_prefill_check.py')}
        if mask_free_sources(control_manifest) != mask_free_sources(winner_manifest):
            raise ValueError('mask-free quality runs used different execution source files')
        if any(old.get(k) != new.get(k) for k in ('model_name', 'model_revision', 'cohort_sha256', 'generation')):
            raise ValueError('mask-free quality metadata differs')
        aa = {r['index']: r for r in old['records'] if r['task'] == task}
        bb = {r['index']: r for r in new['records'] if r['task'] == task}
        if len(aa) != 32 or aa.keys() != bb.keys():
            raise ValueError('mask-free quality examples incomplete')
        if any(aa[i]['input_token_sha256'] != bb[i]['input_token_sha256'] or aa[i]['answers'] != bb[i]['answers'] for i in aa):
            raise ValueError('mask-free quality examples differ')
        before, after = [first(aa[i]) for i in aa], [first(bb[i]) for i in aa]
        lines.append(f'| {task} | 32 | {sum(before)*100/32:.2f}% | {sum(after)*100/32:.2f}% | '
                     f'{sum(y>x for x,y in zip(before,after))} | {sum(y<x for x,y in zip(before,after))} |')
    lines.append('')
    (args.directory / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
