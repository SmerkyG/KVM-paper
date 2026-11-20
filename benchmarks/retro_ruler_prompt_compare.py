"""Compare original and RetroInfer-style wrappers for the same dense examples."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval


def score(task, text, answers):
    hits = [float(a.lower() in text.lower()) for a in answers]
    return max(hits) if task.startswith('qa_') else sum(hits) / len(hits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--variant', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['fwe', 'qa_1'])
    a = p.parse_args()
    baseline, variant = (json.loads(path.read_text()) for path in (a.baseline, a.variant))
    if any(d['stage'] != 'complete' or d['mode'] != 'dense' for d in (baseline, variant)):
        raise ValueError('both complete dense runs are required')
    if variant.get('prompt_format') != 'retroinfer-meta-chat' or baseline['cohort_sha256'] != variant['cohort_sha256']:
        raise ValueError('wrong prompt format or cohort')
    for key in ('model_name', 'model_precision', 'tokenizer_revision', 'generation'):
        if baseline[key] != variant[key]:
            raise ValueError(f'model or generation setting differs: {key}')
    original = {(r['task'], r['source_row']): r for r in baseline['records']}
    modified = {(r['task'], r['source_row']): r for r in variant['records']}
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    stop_ids = {tokenizer.token_to_id('<|end_of_text|>'), tokenizer.token_to_id('<|eot_id|>')}
    if None in stop_ids:
        raise ValueError('tokenizer lacks Llama stop IDs')

    def turn(row):
        ids = row['output_token_ids']
        def decode(tokens):
            return PreTrainedTokenizerBase.clean_up_tokenization(tokenizer.decode(tokens, skip_special_tokens=True))
        if decode(ids) != row['prediction']:
            raise ValueError('tokenizer decode does not match recorded response')
        index = next((i for i, token in enumerate(ids) if token in stop_ids), len(ids))
        return score(row['task'], decode(ids[:index]), row['answers'])

    lines = ['# RULER dense prompt-wrapper diagnostic', '',
             'The same contexts and reference answers are run with '
             'the original Llama 3 chat wrapper and with the `[INST] ... [/INST]` '
             'wrapper in RetroInfer’s released RULER recipe. The system message '
             'is absent from the latter. This tests the wrapper change, not an '
             'exact reproduction of Double-P, whose prompt template is not established.', '',
             '| Task | Original, full output | RetroInfer wrapper, full output | Difference, pp (paired 95% interval) | Original, first turn | RetroInfer wrapper, first turn |',
             '|---|---:|---:|---:|---:|---:|']
    for task in a.tasks:
        keys = sorted(k for k in modified if k[0] == task)
        if len(keys) != 32:
            raise ValueError(f'unexpected count: {task}')
        for key in keys:
            if key not in original or any(original[key][f] != modified[key][f] for f in ('prompt_sha256', 'answers')):
                raise ValueError(f'example mismatch: {key}')
        b, v = [original[k] for k in keys], [modified[k] for k in keys]
        base, mod = (sum(row['score'] for row in rows) * 100 / 32 for rows in (b, v))
        lo, hi = paired_interval([vr['score'] - br['score'] for br, vr in zip(b, v)], 1234)
        first_b, first_v = (sum(turn(row) for row in rows) * 100 / 32 for rows in (b, v))
        lines.append(f'| {task} | {base:.2f} | {mod:.2f} | {mod-base:+.2f} [{lo:+.2f}, {hi:+.2f}] | {first_b:.2f} | {first_v:.2f} |')
    lines += ['', 'Generation follows the released RetroInfer fixed-length `ignore_eos=True` '
              'policy. First-turn scores exclude text after the first Llama end-of-turn '
              'or end-of-text token. Each task has 32 examples; intervals are '
              'paired bootstrap intervals over those examples.', '']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
