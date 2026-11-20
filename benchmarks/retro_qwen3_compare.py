"""Compare matched Qwen3 dense and two-tier RULER results."""
import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--two-tier', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    dense, lod = [json.loads(path.read_text()) for path in (a.dense, a.two_tier)]
    if dense['stage'] != 'complete' or lod['stage'] != 'complete' or dense['mode'] != 'dense' or lod['mode'] != 'two-tier':
        raise ValueError('expected complete dense and two-tier runs')
    for key in ('cohort_sha256', 'model_name', 'model_revision', 'tokenizer_revision', 'model_precision', 'kv_precision', 'generation', 'prompt_format', 'expected'):
        if dense[key] != lod[key]:
            raise ValueError(f'run metadata differs: {key}')
    if dense['model_name'] != 'Qwen/Qwen3-8B' or dense['prompt_format'] != 'qwen3-preformatted':
        raise ValueError('wrong model or prompt template')
    ds = {(r['task'], r['source_row']): r for r in dense['records']}
    ls = {(r['task'], r['source_row']): r for r in lod['records']}
    if set(ds) != set(ls) or any(n != 32 for n in dense['expected'].values()):
        raise ValueError('cohort keys or sample counts differ')
    for key in ds:
        for field in ('prompt_sha256', 'effective_prompt_sha256', 'input_token_sha256', 'answers'):
            if ds[key][field] != ls[key][field]:
                raise ValueError(f'pair differs: {key}, {field}')
    tok = Tokenizer.from_file(str(a.tokenizer))
    stop_ids = {tok.token_to_id('<|endoftext|>'), tok.token_to_id('<|im_end|>')}
    if None in stop_ids:
        raise ValueError('Qwen3 stop token missing')

    def first(row):
        ids = row['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stop_ids), len(ids))
        text = PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end], skip_special_tokens=True))
        return score(row['task'], text, row['answers'])

    lines = ['# Qwen3-8B 32K RULER: dense versus two-tier LoD', '',
             'BF16 model and KV; 32 paired examples per selected task. '
             'These examples were generated using the official RULER task generators '
             'and Qwen3’s tokenizer and chat template with thinking disabled; '
             'they differ from the Llama cohort. LoD uses sparse prefill and two-tier '
             'decode. No full-panel mean is implied.', '']
    for label, get in (('First response', first), ('Full fixed-length output', lambda r: r['score'])):
        lines += [f'## {label}', '', '| Task | Dense | Two-tier | Difference, pp (paired 95% interval) |',
                  '|---|---:|---:|---:|']
        for index, task in enumerate(dense['expected']):
            keys = sorted(k for k in ds if k[0] == task)
            d = [get(ds[k]) for k in keys]
            l = [get(ls[k]) for k in keys]
            base, result = sum(d) * 100 / 32, sum(l) * 100 / 32
            lo, hi = paired_interval([x-y for x, y in zip(l, d)], 12345 + index)
            lines.append(f'| {task} | {base:.2f} | {result:.2f} | {result-base:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
        lines.append('')
    lines += ['The first-response score stops at the first end-of-turn or end-of-text token; '
              'fixed-length scoring can count text generated afterward. Intervals are '
              'paired bootstrap intervals over 32 examples.', '',
              f'Frozen source cohort SHA-256: `{dense["cohort_sha256"]}`.', '']
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
