"""Summarize complete, paired dense and LoD runs on a frozen RULER panel."""
import argparse
import collections
import json
import random
from pathlib import Path


def load(path, expected_mode):
    data = json.loads(path.read_text())
    if data['stage'] != 'complete' or data['mode'] != expected_mode:
        raise ValueError(f'incomplete or wrong-mode result: {path}')
    rows = {(r['task'], r['source_row']): r for r in data['records']}
    counts = collections.Counter(key[0] for key in rows)
    if len(rows) != len(data['records']) or dict(counts) != data['expected']:
        raise ValueError(f'duplicate or missing results: {path}')
    return data, rows


def paired_interval(diffs, seed):
    rng = random.Random(seed)
    n = len(diffs)
    distribution = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                          for _ in range(10000))
    return distribution[249] * 100, distribution[9749] * 100


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--bf16', type=Path, required=True)
    p.add_argument('--int4', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    dense, d = load(args.dense, 'dense')
    bf16, b = load(args.bf16, 'three-tier-bf16')
    int4, i = load(args.int4, 'three-tier-int4')
    if len(dense['expected']) != 5 or any(n != 32 for n in dense['expected'].values()):
        raise ValueError('the report requires the complete five-task, 32-per-task cohort')
    if any(x['cohort_sha256'] != dense['cohort_sha256'] or x['expected'] != dense['expected']
           or x['model_name'] != dense['model_name'] or x['tokenizer_revision'] != dense['tokenizer_revision']
           for x in (bf16, int4)):
        raise ValueError('cohort, model or tokenizer mismatch')
    if set(d) != set(b) or set(d) != set(i):
        raise ValueError('example sets differ')
    for key in d:
        for row in (b[key], i[key]):
            if any(row[field] != d[key][field] for field in ('prompt_sha256', 'input_token_sha256', 'answers')):
                raise ValueError(f'prompt, token or answer mismatch: {key}')
    lines = ['# Llama 3.1 32K RULER exploratory comparison', '',
             'Complete paired results: 32 frozen examples per task, batch one, '
             'Llama-3.1-8B-Instruct, BF16 model weights, greedy fixed-length generation. '
             'The BF16 and INT4 labels refer to the KV cache. LoD uses three-tier top-two. '
             'The LoD path uses sparse prefill; dense uses dense prefill.', '',
             'Scores are percentages using RULER substring matching, not all-or-nothing '
             'exact match. Delta is LoD minus dense in percentage points; intervals are '
             'paired percentile bootstrap intervals over examples (10,000 resamples). '
             'With 32 examples per task, these are exploratory and small differences are '
             'not reliably resolved. A zero-width interval where all observed differences '
             'are zero does not exclude rare failures.', '',
             '| Task | Dense | LoD BF16 | BF16 − dense (95% interval) | LoD INT4 | INT4 − dense (95% interval) |',
             '|---|---:|---:|---:|---:|---:|']
    task_means = []
    for task, n in dense['expected'].items():
        keys = sorted(k for k in d if k[0] == task)
        base = sum(d[k]['score'] for k in keys) / n * 100
        values = []
        for j, rows in enumerate((b, i)):
            mean = sum(rows[k]['score'] for k in keys) / n * 100
            diffs = [rows[k]['score'] - d[k]['score'] for k in keys]
            lo, hi = paired_interval(diffs, 4200 + j)
            values.extend([f'{mean:.2f}', f'{mean-base:+.2f} [{lo:+.2f}, {hi:+.2f}]'])
        lines.append(f'| {task} | {base:.2f} | ' + ' | '.join(values) + ' |')
        task_means.append((base, *(sum(rows[k]['score'] for k in keys) / n * 100 for rows in (b, i))))
    base, bf_mean, int4_mean = (sum(x[j] for x in task_means) / len(task_means) for j in range(3))
    lines.append(f'| **Five-task mean** | **{base:.2f}** | **{bf_mean:.2f}** | **{bf_mean-base:+.2f}** | '
                 f'**{int4_mean:.2f}** | **{int4_mean-base:+.2f}** |')
    lines += ['', 'The mean weights the five task scores equally; its difference has no interval shown.', '',
              'This is a five-task subset, not the full RULER aggregate. The cohort '
              'is separate from the published Double-P examples; its published scores '
              'are literature references rather than paired results.', '',
              f'Cohort SHA-256: `{dense["cohort_sha256"]}`. Tokenizer revision: '
              f'`{dense["tokenizer_revision"]}`.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
