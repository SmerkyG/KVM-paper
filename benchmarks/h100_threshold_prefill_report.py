"""Summarize matched H100 threshold-prefill pilots and optional work audits."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'region', 'entry16', 'entry24', 'tokenizer', 'baseline-timing',
                 'region-timing', 'entry16-timing', 'entry24-timing', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('region-audit', 'entry16-audit', 'entry24-audit'):
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    stops = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stops:
        raise ValueError('missing tokenizer stop token')

    def first(row):
        ids = row['output_token_ids']
        end = next((i for i, t in enumerate(ids) if t in stops), len(ids))
        text = tok.decode(ids[:end], skip_special_tokens=True)
        try:
            text = PreTrainedTokenizerBase.clean_up_tokenization(text)
        except TypeError:
            text = PreTrainedTokenizerBase.clean_up_tokenization(None, text)
        return score(row['task'], text, row['answers'])

    paths = (a.baseline, a.region, a.entry16, a.entry24)
    labels = ('Fixed 8 prefill', 'Region c=16', 'Entry 16', 'Entry 24, cap 16')
    data = [json.loads(path.read_text()) for path in paths]
    if any(d['stage'] != 'complete' or d['mode'] != 'two-tier' or d['cohort_sha256'] != data[0]['cohort_sha256'] for d in data):
        raise ValueError('unmatched or incomplete quality runs')
    for d in data:
        cfg = d['threshold_routing']
        if cfg.get('rule', 'region') != 'region' or cfg['mass_multiplier'] != 16:
            raise ValueError('decode control differs')
    for d, rule, entry in zip(data[1:], ('region', 'entry', 'entry'), (16, 16, 24)):
        cfg = d['prefill_threshold']
        if cfg['rule'] != rule or cfg['entry_bound'] != entry or cfg['work_bound'] != 16:
            raise ValueError('wrong prefill rule or work bound')
    groups = [{(r['task'], r['source_row']): r for r in d['records']} for d in data]
    keys = set(groups[1])
    if len(keys) != 3 or any(set(g) != keys for g in groups[1:]):
        raise ValueError('pilot must have the same three examples for every threshold')
    for key in keys:
        original = groups[0][key]
        for g in groups[1:]:
            row = g[key]
            if (row['prompt_sha256'], row['input_token_sha256'], row['answers']) != (original['prompt_sha256'], original['input_token_sha256'], original['answers']):
                raise ValueError(f'pilot inputs differ: {key}')
    lines = ['# H100 Llama 3.1 threshold prefill: initial experiments', '',
             'BF16 model and KV, batch one, with equal-region c=16 decode held fixed across the prefill variants. The entry rule applies to remote entries; exact local and sink attention remain. The entry-24 variant retains the strict 16 sqrt(n) cap using a deterministic region-order prefix.', '',
             'The implementation is a correctness-first tiled PyTorch router with the existing grouped paged-leaf consumer, not an optimized prefill kernel. A synthetic H100 frontier check passed with maximum absolute output error 0.0034.', '',
             '## Retrieval smoke test', '',
             'One matched example per task. These are smoke tests, not task-quality estimates; each value below is for one example.', '',
             '| Task | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * len(labels)]
    for key in sorted(keys):
        lines.append('| ' + key[0] + ' | ' + ' | '.join(f'{first(g[key])*100:.2f}%' for g in groups) + ' |')
    timing_paths = (a.baseline_timing, a.region_timing, a.entry16_timing, a.entry24_timing)
    timings = [json.loads(path.read_text()) for path in timing_paths]
    warmed = all(sum(not r['warmup'] for r in t['records']) >= 2 and any(r['warmup'] for r in t['records']) for t in timings)
    timing_scope = ('All figures are medians of two measured repeats after a warmup.' if warmed else
                    'The fixed-eight figure is the median of two measured repeats after a warmup; each prototype figure is a single un-warmed pilot.')
    lines += ['', '## Whole-model prefill timing', '',
              'H100, 32,768 prompt tokens, matched teacher-forced prompt and BF16 Llama. ' + timing_scope + ' Differences are specific to this prototype.', '',
              '| Prefill rule | Seconds |', '|---|---:|']
    for label, timing in zip(labels, timings):
        if timing['prompt_tokens'] != 32768:
            raise ValueError('timing context mismatch')
        lines.append(f'| {label} | {timing["median_prefill_s"]:.3f} |')
    audit_paths = (a.region_audit, a.entry16_audit, a.entry24_audit)
    if all(audit_paths):
        audits = [json.loads(path.read_text()) for path in audit_paths]
        if any(d['stage'] != 'complete' or d['cohort_sha256'] != data[0]['cohort_sha256'] for d in audits):
            raise ValueError('incomplete audit')
        for d, rule, entry in zip(audits, ('region', 'entry', 'entry'), (16, 16, 24)):
            cfg = d['prefill_threshold']; decode = d['threshold_routing']
            if cfg['rule'] != rule or cfg['entry_bound'] != entry or cfg['work_bound'] != 16 or decode.get('rule', 'region') != 'region' or decode['mass_multiplier'] != 16:
                raise ValueError('wrong audit routing configuration')
        audit_groups = [{(r['task'], r['source_row']): r for r in d['records']} for d in audits]
        audit_keys = set(audit_groups[0])
        if len(audit_keys) != 16 or any(set(g) != audit_keys for g in audit_groups):
            raise ValueError('work audit must contain the same 2 x 8 examples')
        for key in audit_keys:
            original = groups[0][key]
            for g in audit_groups:
                row = g[key]
                if (row['prompt_sha256'], row['input_token_sha256'], row['answers']) != (original['prompt_sha256'], original['input_token_sha256'], original['answers']):
                    raise ValueError(f'audit inputs differ: {key}')
        lines += ['', '## Selected multikey quality and remote prefill work', '',
                  'Eight matched examples per task, using first-response scoring. Work averages include every remote prefill query, query head, and layer, excluding exact initial, local, and sink fields. Proxy coverage is not exact attention mass. Maximum fraction uses 16 sqrt(n) for all variants; the region rule does not have that cap.', '',
                  '| Task | Rule | Score | Mean entries/head/query | Mean regions/head/query | Mean proxy mass | Max bound fraction |', '|---|---|---:|---:|---:|---:|---:|']
        for task in sorted({k[0] for k in audit_keys}):
            task_keys = sorted(k for k in audit_keys if k[0] == task)
            if len(task_keys) != 8:
                raise ValueError('wrong audit task size')
            lines.append(f'| {task} | Fixed 8 prefill | {100*sum(first(groups[0][k]) for k in task_keys)/8:.2f}% | — | — | — | — |')
            for label, group in zip(labels[1:], audit_groups):
                layers = [layer for k in task_keys for layer in group[k]['threshold_prefill_audit']['layers']]
                n = sum(x['query_heads'] for x in layers)
                lines.append(f'| {task} | {label} | {100*sum(first(group[k]) for k in task_keys)/8:.2f}% | {sum(x["entries_sum"] for x in layers)/n:.1f} | {sum(x["regions_sum"] for x in layers)/n:.2f} | {sum(x["proxy_mass_sum"] for x in layers)/n:.2%} | {max(x["max_bound_fraction"] for x in layers):.3f} |')
        lines += ['', 'Audit outputs: ' + ' · '.join(f'[{label}]({path.name})' for label, path in zip(labels[1:], audit_paths)) + '.']
    else:
        lines += ['', 'The eight-example-per-task work and quality audits are in progress.']
    lines += ['', 'The entry bound covers logical exact remote entries per query head. Prefill also performs local attention, routing, page packing, and cache construction; no complete-execution subquadratic bound has been established.', '',
              'Pilot outputs: ' + ' · '.join(f'[{label}]({path.name})' for label, path in zip(labels, paths)) + '.', '']
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
