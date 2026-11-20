"""Merge two paired RULER runs into a validated 13-task comparison."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from benchmarks.retro_ruler_compare import load, paired_interval
from benchmarks.retro_ruler_prepare import ALL_TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--dense-five', type=Path, required=True)
    p.add_argument('--lod-five', type=Path, required=True)
    p.add_argument('--dense-eight', type=Path, required=True)
    p.add_argument('--lod-eight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    old_d, od = load(args.dense_five, 'dense')
    old_l, ol = load(args.lod_five, 'three-tier-bf16')
    new_d, nd = load(args.dense_eight, 'dense')
    new_l, nl = load(args.lod_eight, 'three-tier-bf16')
    cohort_raw = args.cohort.read_bytes()
    cohort = json.loads(cohort_raw)
    cohort_hash = hashlib.sha256(cohort_raw).hexdigest()
    if cohort['samples_per_task'] != 32 or set(cohort['source_files']) != set(ALL_TASKS):
        raise ValueError('not the full 13-task, 32-per-task cohort')
    frozen = {(r['task'], r['source_row']): r for r in cohort['samples']}
    if len(frozen) != 13 * 32 or set(od) & set(nd) or set(ol) & set(nl):
        raise ValueError('cohort is incomplete or runs overlap')
    if old_d['cohort_sha256'] != old_l['cohort_sha256']:
        raise ValueError('original cohort hashes differ')
    for data in (new_d, new_l):
        if data['cohort_sha256'] != cohort_hash or data.get('reference_cohort_sha256') != old_d['cohort_sha256']:
            raise ValueError('new cohort or original reference hash differs')
    if any(x['model_name'] != old_d['model_name'] or x['tokenizer_revision'] != old_d['tokenizer_revision']
           or x['model_precision'] != old_d['model_precision'] for x in (old_l, new_d, new_l)):
        raise ValueError('model, precision, or tokenizer differs')
    dense, lod = od | nd, ol | nl
    if set(dense) != set(frozen) or set(lod) != set(frozen):
        raise ValueError('cohort rows do not match result rows')
    for key, source in frozen.items():
        for row in (dense[key], lod[key]):
            if row['prompt_sha256'] != source['prompt_sha256'] or row['answers'] != source['answers']:
                raise ValueError(f'frozen input or answers differ: {key}')
        if dense[key]['input_token_sha256'] != lod[key]['input_token_sha256']:
            raise ValueError(f'tokenized input differs: {key}')
    lines = ['# Llama 3.1 32K RULER: 13-task exploratory comparison', '',
             'Each task contains 32 frozen examples, using the same tokenized inputs '
             'for dense and LoD. The model weights and KV are BF16; LoD uses '
             'three-tier top-two and sparse prefill. Generation is greedy with '
             'the reference task limits and ignores EOS, matching the RetroInfer '
             'RULER generation convention.', '',
             '| Task | Dense | LoD BF16 | Difference, pp (paired 95% interval) |',
             '|---|---:|---:|---:|']
    means, differences = [], []
    for j, task in enumerate(ALL_TASKS):
        keys = sorted(k for k in frozen if k[0] == task)
        if len(keys) != 32:
            raise ValueError(f'incorrect sample count: {task}')
        base = 100 * sum(dense[k]['score'] for k in keys) / 32
        result = 100 * sum(lod[k]['score'] for k in keys) / 32
        diffs = [lod[k]['score'] - dense[k]['score'] for k in keys]
        lo, hi = paired_interval(diffs, 4200 + j)
        lines.append(f'| {task} | {base:.2f} | {result:.2f} | {result-base:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
        means.append((base, result))
        differences.append(diffs)
    base = sum(x[0] for x in means) / 13
    result = sum(x[1] for x in means) / 13
    rng = random.Random(4242)
    boot = sorted(100 * sum(sum(v[rng.randrange(32)] for _ in range(32)) / 32
                            for v in differences) / 13 for _ in range(10000))
    lines.append(f'| **13-task mean** | **{base:.2f}** | **{result:.2f}** | '
                 f'**{result-base:+.2f} [{boot[249]:+.2f}, {boot[9749]:+.2f}]** |')
    lines += ['', 'The mean weights tasks equally. Intervals use paired bootstrap '
              'resampling within each task; the mean interval resamples independently '
              'within all 13 tasks. Each task has only 32 examples, so small '
              'differences are uncertain. Zero-width intervals do not rule out rare '
              'failures.', '',
              'Double-P Table 2 reports a 32K 13-task mean of 90.00% dense and '
              '89.91% Double-P. These are separate literature results, not matched '
              'runs on our frozen examples. The paper does not establish identical '
              'model variant, prompts or sample selection. Its `Avg.` column '
              'additionally averages over 16K, 32K and 64K, which we did not run.', '',
              f'Frozen cohort SHA-256: `{cohort_hash}`.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
