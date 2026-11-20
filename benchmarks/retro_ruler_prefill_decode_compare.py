"""Compare all four paired prefill/decode combinations on Llama multi-key 3."""
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
    p.add_argument('--lod', type=Path, required=True)
    p.add_argument('--dense-prefill', type=Path, required=True)
    p.add_argument('--dense-decode', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = (a.dense, a.lod, a.dense_prefill, a.dense_decode)
    data = [json.loads(path.read_text()) for path in paths]
    if any(d['stage'] != 'complete' for d in data):
        raise ValueError('incomplete inputs')
    if [d['mode'] for d in data] != ['dense', 'three-tier-bf16', 'three-tier-bf16', 'three-tier-bf16']:
        raise ValueError('wrong attention modes')
    if data[2].get('prefill_attention') != 'dense with LoD cache construction':
        raise ValueError('wrong dense-prefill diagnostic')
    if (data[3].get('prefill_attention') != 'LoD sparse' or
            data[3].get('decode_attention') != 'dense with full KV'):
        raise ValueError('wrong dense-decode diagnostic')
    task = 'niah_multikey_3'
    groups = [{(r['task'], r['source_row']): r for r in d['records'] if r['task'] == task} for d in data]
    if len(groups[3]) != 32 or any(set(g) != set(groups[3]) for g in groups):
        raise ValueError('wrong or unpaired examples')
    for key in groups[3]:
        dense, lod, dp, dd = [group[key] for group in groups]
        if any(any(row[f] != dense[f] for f in ('prompt_sha256', 'input_token_sha256', 'answers'))
               for row in (lod, dp, dd)):
            raise ValueError(f'input mismatch: {key}')
        if (dense['output_token_ids'][0] != dp['output_token_ids'][0] or
                lod['output_token_ids'][0] != dd['output_token_ids'][0]):
            raise ValueError(f'first-token mismatch in matched prefill: {key}')
    tok = Tokenizer.from_file(str(a.tokenizer))
    stop = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stop:
        raise ValueError('missing stop token')

    def first(row):
        ids = row['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stop), len(ids))
        text = PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end], skip_special_tokens=True))
        return score(task, text, row['answers'])

    keys = sorted(groups[3])
    labels = ('Dense prefill, dense decode', 'LoD prefill, LoD decode',
              'Dense prefill, LoD decode', 'LoD prefill, dense decode')
    lines = ['# Llama 32K multi-key 3: prefill and decode ablation', '',
             'All four rows use the same 32 inputs and BF16 weights/KV. LoD '
             'is three-tier top-two. Dense-decode diagnostic stores and checks '
             'all exact keys and values produced by LoD prefill at every layer; '
             'it then uses the native dense decode graph. It is not a speed '
             'measurement. Matching prefill variants produced the same first '
             'generated token on all 32 examples.', '']
    for title, get in (('First response', first), ('Full fixed-length output', lambda r: r['score'])):
        values = [[get(g[k]) for k in keys] for g in groups]
        lines += [f'## {title}', '', '| Prefill and decode | Score | Change from LoD prefill/decode, pp (paired 95% interval) |',
                  '|---|---:|---:|']
        for index, (label, vals) in enumerate(zip(labels, values)):
            diffs = [v - b for v, b in zip(vals, values[1])]
            lo, hi = paired_interval(diffs, 9990 + index)
            lines.append(f'| {label} | {sum(vals)*100/32:.2f} | {sum(diffs)*100/32:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
        lines.append('')
    lines += ['First-response scoring stops at the first end-of-turn or end-of-text token. '
              'Full fixed-length scoring can count text after that token. Paired '
              'bootstrap intervals are based on 32 examples.', '']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
