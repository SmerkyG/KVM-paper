"""Compare paired dense, two-tier, and three-tier RULER outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


TASKS = ('niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
         'vt', 'fwe', 'cwe', 'qa_1', 'qa_2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--dense', type=Path, nargs=2, required=True)
    p.add_argument('--three-tier', type=Path, nargs=2, required=True)
    p.add_argument('--two-tier', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cohort_raw = args.cohort.read_bytes()
    cohort_hash = hashlib.sha256(cohort_raw).hexdigest()
    cohort = json.loads(cohort_raw)
    frozen = {(r['task'], r['source_row']): r for r in cohort['samples'] if r['task'] in TASKS}
    if len(frozen) != 32 * len(TASKS):
        raise ValueError('expected exactly 32 frozen examples per task')

    def read(paths, mode):
        rows, metadata = {}, []
        for path in paths:
            data = json.loads(path.read_text())
            if data['stage'] != 'complete' or data['mode'] != mode:
                raise ValueError(f'incomplete run or wrong mode: {path}')
            if mode == 'two-tier' and data['cohort_sha256'] != cohort_hash:
                raise ValueError('two-tier cohort hash mismatch')
            metadata.append(data)
            for row in data['records']:
                key = (row['task'], row['source_row'])
                if key in frozen:
                    if key in rows:
                        raise ValueError(f'duplicate input: {key}')
                    rows[key] = row
        if set(rows) != set(frozen):
            raise ValueError(f'missing task rows for {mode}')
        return rows, metadata

    dense, dmeta = read(args.dense, 'dense')
    three, tmeta = read(args.three_tier, 'three-tier-bf16')
    two, wmeta = read([args.two_tier], 'two-tier')
    for data in dmeta + tmeta + wmeta:
        for key in ('model_name', 'model_precision', 'tokenizer_revision', 'generation', 'kv_precision'):
            if data[key] != dmeta[0][key]:
                raise ValueError(f'model or generation setting differs: {key}')
    for key, src in frozen.items():
        for row in (dense[key], three[key], two[key]):
            if row['prompt_sha256'] != src['prompt_sha256'] or row['answers'] != src['answers']:
                raise ValueError(f'prompt or reference differs: {key}')
            if row['input_token_sha256'] != dense[key]['input_token_sha256']:
                raise ValueError(f'tokenized input differs: {key}')

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    stop = {tokenizer.token_to_id('<|end_of_text|>'), tokenizer.token_to_id('<|eot_id|>')}
    if None in stop:
        raise ValueError('tokenizer lacks Llama stop tokens')

    def first_turn(row):
        ids = row['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stop), len(ids))
        text = PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(ids[:end], skip_special_tokens=True))
        return score(row['task'], text, row['answers'])

    lines = ['# Llama 3.1 32K RULER: two-tier quality on selected tasks', '',
             'Paired 32-example tasks; BF16 model and KV, same frozen tokenized '
             'inputs, greedy generation, LoD sparse prefill, and graph decode. '
             'The eight tasks were selected for relatively low three-tier scores '
             'or gaps to dense; they are not a full-panel average.', '']
    for label, get_score in (('First response', first_turn), ('Full fixed-length output', lambda row: row['score'])):
        lines += [f'## {label}', '',
                  '| Task | Dense | Three-tier | Two-tier | Two minus three, pp (paired 95% interval) |',
                  '|---|---:|---:|---:|---:|']
        for index, task in enumerate(TASKS):
            keys = sorted(k for k in frozen if k[0] == task)
            values = [[get_score(group[k]) for k in keys] for group in (dense, three, two)]
            means = [sum(v) * 100 / 32 for v in values]
            lo, hi = paired_interval([a - b for a, b in zip(values[2], values[1])], 4721 + index)
            lines.append(f'| {task} | {means[0]:.2f} | {means[1]:.2f} | {means[2]:.2f} | '
                         f'{means[2]-means[1]:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
        lines.append('')
    lines += ['First-response scoring stops at the first end-of-turn or end-of-text token. '
              'Full-output scoring follows RetroInfer’s fixed-length ignore-EOS convention '
              'and can count words generated after the first turn. Intervals are paired '
              'bootstrap intervals; 32 examples provide limited precision.', '',
              f'Frozen full-cohort SHA-256: `{cohort_hash}`.', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
