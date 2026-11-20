"""Compare fused H100 threshold prefill with the reference and fixed-eight."""
import argparse
import json
import re
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'reference-region', 'reference-entry16', 'reference-entry24',
                 'check', 'verify-region', 'verify-entry16', 'verify-entry24', 'tokenizer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timing', type=Path, nargs='+', required=True)
    p.add_argument('--profile', type=Path)
    p.add_argument('--warm-profile', type=Path)
    p.add_argument('--audit-reference-entry16', type=Path)
    p.add_argument('--audit-reference-entry24', type=Path)
    p.add_argument('--audit-optimized-entry16', type=Path)
    p.add_argument('--audit-optimized-entry24', type=Path)
    a = p.parse_args()
    reference_paths = (a.baseline, a.reference_region, a.reference_entry16, a.reference_entry24)
    reference_labels = ('Fixed 8', 'Region c=16, reference', 'Entry 16, reference', 'Entry 24 cap 16, reference')
    check = json.loads(a.check.read_text())
    if check['stage'] != 'complete':
        raise ValueError('frontier check did not finish')
    lines = ['# Optimizing threshold prefill on H100', '',
             'BF16 Llama 3.1, batch one, 32,768 prompt tokens. Threshold variants keep equal-region c=16 decode fixed. The optimized path fuses threshold selection with unopened-region attention, compacts selected regions in groups, and shares prefill scratch across sequential layers. Entry 24 retains a strict 16 sqrt(n) exact-remote-entry cap with deterministic prefix clipping. This bound counts logical remote entries per query head; it does not bound local attention, padded work, cache construction or updates, or the complete model execution.', '',
             'To enable the experimental path in a logged H100 cluster run, set `LOD_H100_PREFILL_RULE=entry` and `LOD_H100_PREFILL_IMPLEMENTATION=fused`. The default entry and work bounds are both 16; set `LOD_H100_PREFILL_ENTRY_BOUND=24` and `LOD_H100_PREFILL_WORK_BOUND=16` for the capped 24 case. `LOD_H100_PREFILL_SELECT_N=128` enables the wider selection experiment; it has been checked for entry 16. `LOD_H100_PREFILL_IMPLEMENTATION=reference` selects the earlier implementation. These settings do not change ordinary serving defaults.', '',
             '## Whole-model prefill', '',
             'Matched teacher-forced inputs, one warmup and two measured repeats. Figures are medians and include cache construction. Reference means the earlier tiled PyTorch implementation.', '',
             '| Prefill rule and implementation | Query tile | Seconds | Relative to fixed 8 |', '|---|---:|---:|---:|']
    reference = [json.loads(path.read_text()) for path in reference_paths]
    baseline = reference[0]['median_prefill_s']
    manifest = reference[0]['manifest']
    for label, data in zip(reference_labels, reference):
        if data['prompt_tokens'] != 32768 or data['manifest'] != manifest:
            raise ValueError('timing prompt mismatch')
        rec = [r for r in data['records'] if not r['warmup']]
        if len(rec) != 2 or not any(r['warmup'] for r in data['records']):
            raise ValueError('timing needs warmup and two repetitions')
        value = data['median_prefill_s']
        lines.append(f'| {label} | — | {value:.3f} | {value/baseline:.2f}x |')
    best = {}
    for path in a.timing:
        data = json.loads(path.read_text()); cfg = data['prefill_threshold']
        if data['prompt_tokens'] != 32768 or data['manifest'] != manifest or cfg['implementation'] != 'fused':
            raise ValueError('fused timing configuration mismatch')
        rec = [r for r in data['records'] if not r['warmup']]
        if len(rec) != 2 or not any(r['warmup'] for r in data['records']):
            raise ValueError('fused timing needs warmup and two repetitions')
        rule = 'Region c=16' if cfg['rule'] == 'region' else f'Entry {cfg["entry_bound"]}'
        if cfg['rule'] == 'entry' and cfg['work_bound'] != cfg['entry_bound']:
            rule += f' cap {cfg["work_bound"]}'
        value = data['median_prefill_s']
        if '-default-' in path.name or '-select128-' in path.name:
            implementation = ('fused, indexed default with wider selection' if '-select128-' in path.name
                              else 'fused, indexed default')
        elif '-direct-' in path.name or '-indexed-' in path.name:
            implementation = ('fused, precomputed KV addresses' if '-indexed-' in path.name
                              else 'fused, direct exact attention')
            geometry = re.search(r'-b(\d+)-w(\d+)-', path.name)
            if geometry:
                implementation += f' (block {geometry[1]}, {geometry[2]} warps)'
        elif '-stream-' in path.name:
            implementation = 'fused, blockwise reduction'
        elif '-warp8-' in path.name:
            implementation = 'fused, eight-warp reduction'
        else:
            implementation = 'fused, compact launch' if '-compact-' in path.name else 'fused'
        key = (cfg['rule'], cfg['entry_bound'], cfg['work_bound']) if cfg['rule'] == 'entry' else ('region', 0, 0)
        if key not in best or value < best[key][0]:
            best[key] = (value, path, rule, implementation, cfg['tile_queries'])
    for value, path, rule, implementation, tile in sorted(best.values(), key=lambda item: (item[2] != 'Region c=16', item[2])):
        lines.append(f'| [{rule}, {implementation}]({path.name}) | {tile} | {value:.3f} | {value/baseline:.2f}x |')
    tok = Tokenizer.from_file(str(a.tokenizer))
    stops = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stops:
        raise ValueError('missing stop tokens')
    def first(row):
        ids = row['output_token_ids']; end = next((i for i, t in enumerate(ids) if t in stops), len(ids))
        text = tok.decode(ids[:end], skip_special_tokens=True)
        try: text = PreTrainedTokenizerBase.clean_up_tokenization(text)
        except TypeError: text = PreTrainedTokenizerBase.clean_up_tokenization(None, text)
        return score(row['task'], text, row['answers'])
    quality_paths = (a.verify_region, a.verify_entry16, a.verify_entry24)
    quality = [json.loads(path.read_text()) for path in quality_paths]
    groups = [{(r['task'], r['source_row']): r for r in d['records']} for d in quality]
    if any(d['stage'] != 'complete' or d['prefill_threshold']['implementation'] != 'fused' for d in quality):
        raise ValueError('fused validation incomplete')
    keys = set(groups[0])
    if len(keys) != 3 or any(set(g) != keys for g in groups):
        raise ValueError('quality smoke-test cohort mismatch')
    for key in keys:
        x = groups[0][key]
        for group in groups[1:]:
            y = group[key]
            if (x['input_token_sha256'], x['answers']) != (y['input_token_sha256'], y['answers']):
                raise ValueError('quality input mismatch')
    lines += ['', '## Correctness checks', '',
              'The independent synthetic frontier check covers region and entry thresholds, including a strict cap configuration. The real-state check compares selected regions using the same routing logits and compares output with the reference for 16 queries per layer; differences in decisions within 2e-5 of a log threshold are reported as numerical ties. Three matched RULER examples are only a smoke test, not a quality estimate.', '',
              '| Task | Region c=16 | Entry 16 | Entry 24 cap 16 |', '|---|---:|---:|---:|']
    for key in sorted(keys):
        lines.append(f'| {key[0]} | ' + ' | '.join(f'{100*first(group[key]):.2f}%' for group in groups) + ' |')
    validation = [r for d in quality for row in d['records'] for r in row['prefill_fused_validation']]
    lines += ['', f'Real-state validation: {sum(r["query_heads"] for r in validation):,} query heads across layers; {sum(r["selection_mismatches"] for r in validation)} threshold-tie decision differences, {sum(r["selection_outside_tolerance"] for r in validation)} outside tolerance; maximum attention-output difference {max(r["max_abs"] for r in validation):.5f}.', '',
              f'[Synthetic frontier check]({a.check.name}) · ' + ' · '.join(f'[{label}]({path.name})' for label, path in zip(('Region quality', 'Entry 16 quality', 'Entry 24 quality'), quality_paths)) + '.', '',
              'The timing table shows the fastest checked implementation of each rule among the measured settings. Other timing outputs are retained in this directory.', '']
    if a.profile is not None:
        profile = json.loads(a.profile.read_text())['records'][0]
        stages = profile['prefill_leaf_profile']
        totals = {name: sum(layer.get(name, 0) for layer in stages) for name in ('dispatch', 'pack', 'kernel', 'reduce')}
        lines += ['## Bottleneck diagnostic', '',
                  'One separately instrumented run of entry 16 with the compact launch and original one-warp reducer. Instrumentation synchronizes the device and is not a matched speed measurement.', '',
                  'Summed across layers: ' + ', '.join(f'{name} {seconds:.3f} s' for name, seconds in totals.items()) + '.', '',
                  f'[Diagnostic measurements]({a.profile.name}).', '']
    if a.warm_profile is not None:
        sample = json.loads(a.warm_profile.read_text())['records'][-1]
        if sample['warmup']:
            raise ValueError('warm profile has no measured repeat')
        stages = sample['prefill_profile']
        names = ('route_logits', 'select_and_coarse', 'compact', 'expand_indices', 'exact_leaves', 'merge')
        totals = {name: sum(layer.get(name, 0) for layer in stages) for name in names}
        lines += ['The optimized default was also profiled after a warmup. Synchronized timings are diagnostic and cannot be substituted for the matched measurements above: ' +
                  ', '.join(f'{name} {seconds:.3f} s' for name, seconds in totals.items()) + '.', '',
                  f'[Warm diagnostic measurements]({a.warm_profile.name}).', '']
    audit_paths = (a.audit_reference_entry16, a.audit_optimized_entry16,
                   a.audit_reference_entry24, a.audit_optimized_entry24)
    if any(audit_paths):
        if not all(audit_paths):
            raise ValueError('all four matched audit inputs are required')
        audits = [json.loads(path.read_text()) for path in audit_paths]
        if any(data['stage'] != 'complete' for data in audits):
            raise ValueError('retrieval audit is incomplete')
        cohorts = [{(row['task'], row['source_row']): row for row in data['records']} for data in audits]
        keys = set(cohorts[0])
        if len(keys) != 16 or any(set(group) != keys for group in cohorts):
            raise ValueError('retrieval audits must use the same 16 examples')
        for key in keys:
            reference = cohorts[0][key]
            if any((group[key]['input_token_sha256'], group[key]['answers']) !=
                   (reference['input_token_sha256'], reference['answers']) for group in cohorts[1:]):
                raise ValueError('retrieval audit inputs differ')
        lines += ['## Matched retrieval sample', '',
                  'Eight examples per task, with the same inputs for both implementations. The optimized columns use the fused default with precomputed KV addresses. This is a limited quality check, not a full RULER result.', '',
                  '| Task | Entry 16 reference | Entry 16 optimized | Entry 24 cap 16 reference | Entry 24 cap 16 optimized |',
                  '|---|---:|---:|---:|---:|']
        for task in sorted({key[0] for key in keys}):
            task_keys = [key for key in keys if key[0] == task]
            if len(task_keys) != 8:
                raise ValueError('each retrieval task must have eight examples')
            scores = [100 * sum(first(group[key]) for key in task_keys) / len(task_keys) for group in cohorts]
            lines.append(f'| {task} | ' + ' | '.join(f'{score:.2f}%' for score in scores) + ' |')
        lines += ['', 'Audit outputs: ' + ' · '.join(f'[{path.stem}]({path.name})' for path in audit_paths) + '.', '']
    a.output.write_text('\n'.join(lines)); print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
