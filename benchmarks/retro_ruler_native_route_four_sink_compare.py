"""Summarize the paired native-route/four-sink Llama quality ablation."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase
from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('dense', 'baseline', 'candidate', 'two-tier', 'retroinfer', 'tokenizer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    paths = [a.dense, a.baseline, a.candidate, a.two_tier, a.retroinfer]
    data = [json.loads(path.read_text()) for path in paths]
    if any(d['stage'] != 'complete' for d in data):
        raise ValueError('incomplete input run')
    if not data[2].get('lod_native_route_four_sink'):
        raise ValueError('candidate was not the native-route/four-sink run')
    if data[1].get('lod_native_route_four_sink'):
        raise ValueError('baseline was unexpectedly modified')
    task = 'niah_multikey_3'
    groups = [{(r['task'], r['source_row']): r for r in d['records'] if r['task'] == task} for d in data]
    keys = sorted(groups[2])
    if len(keys) != 32 or any(set(g) != set(keys) for g in groups):
        raise ValueError('wrong or unpaired cohort')
    for key in keys:
        baseline = groups[1][key]
        if any(any(g[key][f] != baseline[f] for f in ('prompt_sha256', 'input_token_sha256', 'answers')) for g in groups):
            raise ValueError(f'unmatched input: {key}')
    tok = Tokenizer.from_file(str(a.tokenizer))
    stop = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stop:
        raise ValueError('missing stop tokens')
    def first(row):
        ids = row['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stop), len(ids))
        text = PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end], skip_special_tokens=True))
        return score(task, text, row['answers'])
    values = [[first(g[k]) for k in keys] for g in groups]
    labels = ['Dense', 'Original three-tier LoD', 'Native-query routing + four sinks', 'Two-tier LoD', 'RetroInfer']
    lines = ['# Llama 32K multi-key 3: native routing and four sinks', '',
        'The same 32 frozen prompts, Llama-3.1-8B-Instruct and BF16. '
        'The candidate combines native-query region routing and four exact sink tokens in '
        'both sparse prefill and three-tier top-2 decode. The original three-tier '
        'run uses normalized-query routing and one sink. This changes both settings '
        'at once and is not an isolated attribution or a speed measurement.', '',
        '| Method | First-response score | Difference from original three-tier, pp (paired 95% interval) |',
        '|---|---:|---:|']
    for i, (label, vals) in enumerate(zip(labels, values)):
        diff = [v - b for v,b in zip(vals, values[1])]
        lo,hi = paired_interval(diff, 86110+i)
        lines.append(f'| {label} | {sum(vals)*100/32:.2f}% | {sum(diff)*100/32:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    changes = [values[2][i]-values[1][i] for i in range(32)]
    improved = [keys[i][1] for i,x in enumerate(changes) if x > 0]
    declined = [keys[i][1] for i,x in enumerate(changes) if x < 0]
    first_changed = sum(groups[1][k]['output_token_ids'][0] != groups[2][k]['output_token_ids'][0] for k in keys)
    lines += ['', f'Paired first responses: {len(improved)} improved, {len(declined)} worsened, '
        f'{32-len(improved)-len(declined)} unchanged. Improved source rows: {improved}. '
        f'Worsened source rows: {declined}. The first token after prefill changed on '
        f'{first_changed}/32 examples.', '',
        'First-response scoring stops at end-of-turn or end-of-text. Bootstrap '
        'intervals use paired examples. Dense and RetroInfer use dense prefill; '
        'the LoD runs use sparse prefill. The source rows and individual full '
        'fixed-length outputs are retained in the JSON.', '']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)

if __name__ == '__main__':
    main()
