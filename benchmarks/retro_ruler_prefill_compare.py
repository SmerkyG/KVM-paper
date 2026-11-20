"""Analyze matched LoD dense-prefill and sparse-prefill RULER runs."""
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
    p.add_argument('--sparse-prefill', type=Path, required=True)
    p.add_argument('--dense-prefill', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = [json.loads(path.read_text()) for path in (a.dense, a.sparse_prefill, a.dense_prefill)]
    if any(d['stage'] != 'complete' for d in data):
        raise ValueError('incomplete inputs')
    if [d['mode'] for d in data] != ['dense', 'three-tier-bf16', 'three-tier-bf16']:
        raise ValueError('wrong attention modes')
    if data[2].get('prefill_attention') != 'dense with LoD cache construction':
        raise ValueError('wrong prefill diagnostic')
    task = 'niah_multikey_3'
    groups = [{(r['task'], r['source_row']): r for r in d['records'] if r['task'] == task} for d in data]
    if len(groups[2]) != 32 or any(set(g) != set(groups[2]) for g in groups):
        raise ValueError('wrong or unpaired examples')
    for key in groups[2]:
        b, sparse, variant = [group[key] for group in groups]
        if any(any(row[f] != b[f] for f in ('prompt_sha256', 'input_token_sha256', 'answers'))
               for row in (sparse, variant)):
            raise ValueError(f'input mismatch: {key}')
        if b['output_token_ids'][0] != variant['output_token_ids'][0]:
            raise ValueError(f'dense prefill first-token mismatch: {key}')
    tok = Tokenizer.from_file(str(a.tokenizer))
    stop = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stop:
        raise ValueError('missing stop token')

    def first(row):
        ids = row['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stop), len(ids))
        text = PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end], skip_special_tokens=True))
        return score(task, text, row['answers'])

    lines = ['# Llama 32K multi-key 3: LoD prefill ablation', '',
             'The diagnostic constructs the ordinary three-tier LoD cache but uses '
             'dense attention outputs during prefill, then uses unchanged LoD decode. '
             'All 32 first generated tokens match the recorded dense run. '
             'This is an ablation, not a production speed measurement.', '']
    keys = sorted(groups[2])
    for label, get in (('First response', first), ('Full fixed-length output', lambda r: r['score'])):
        vals = [[get(g[k]) for k in keys] for g in groups]
        averages = [100 * sum(v) / 32 for v in vals]
        differences = [a - b for a, b in zip(vals[2], vals[1])]
        lo, hi = paired_interval(differences, 9912)
        delta = sum(differences) * 100 / 32
        lines += [f'{label}: dense {averages[0]:.2f}%, three-tier with sparse prefill '
                  f'{averages[1]:.2f}%, three-tier with dense prefill {averages[2]:.2f}%; '
                  f'dense-prefill minus sparse-prefill {delta:+.2f} points '
                  f'(paired 95% interval {lo:+.2f} to {hi:+.2f}).', '']
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
