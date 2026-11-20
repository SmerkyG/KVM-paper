"""Compare paired RetroInfer, dense and LoD RULER generations."""
import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from benchmarks.retro_ruler_compare import paired_interval
from benchmarks.retro_ruler_prompt_compare import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--dense', type=Path, nargs='+', required=True)
    p.add_argument('--three-tier', type=Path, nargs='*', default=[])
    p.add_argument('--two-tier', type=Path, required=True)
    p.add_argument('--retroinfer', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cohort_raw = a.cohort.read_bytes()
    cohort_hash = hashlib.sha256(cohort_raw).hexdigest()
    cohort = json.loads(cohort_raw)
    retro_data = json.loads(a.retroinfer.read_text())
    tasks = tuple(retro_data['expected'])
    if not tasks or any(n != 32 for n in retro_data['expected'].values()):
        raise ValueError('expected complete 32-example task selection')
    frozen = {(r['task'], r['source_row']): r for r in cohort['samples'] if r['task'] in tasks}
    if len(frozen) != 32 * len(tasks):
        raise ValueError('frozen cohort is incomplete')

    def read(paths, mode):
        rows, metadata = {}, []
        for path in paths:
            data = json.loads(path.read_text())
            if data['stage'] != 'complete' or data['mode'] != mode:
                raise ValueError(f'incomplete run or wrong mode: {path}')
            if mode == 'retroinfer' and data['cohort_sha256'] != cohort_hash:
                raise ValueError('RetroInfer cohort hash mismatch')
            metadata.append(data)
            for row in data['records']:
                key = (row['task'], row['source_row'])
                if key in frozen:
                    if key in rows:
                        raise ValueError(f'duplicate input: {key}')
                    rows[key] = row
        if set(rows) != set(frozen):
            raise ValueError(f'wrong cohort rows for {mode}')
        return rows, metadata

    dense, dmeta = read(a.dense, 'dense')
    two, wmeta = read([a.two_tier], 'two-tier')
    retro, rmeta = read([a.retroinfer], 'retroinfer')
    three, tmeta = read(a.three_tier, 'three-tier-bf16') if a.three_tier else ({}, [])
    for data in dmeta + wmeta + rmeta + tmeta:
        for field in ('model_name', 'model_precision', 'tokenizer_revision', 'generation', 'kv_precision'):
            if data[field] != dmeta[0][field]:
                raise ValueError(f'model or generation setting differs: {field}')
    for key, src in frozen.items():
        for group in (dense, two, retro) + ((three,) if three else ()):
            row = group[key]
            if row['prompt_sha256'] != src['prompt_sha256'] or row['answers'] != src['answers']:
                raise ValueError(f'prompt or reference differs: {key}')
            if row['input_token_sha256'] != dense[key]['input_token_sha256']:
                raise ValueError(f'tokenized input differs: {key}')

    tok = Tokenizer.from_file(str(a.tokenizer))
    model = dmeta[0]['model_name']
    stop_names = ('<|endoftext|>', '<|im_end|>') if model.startswith('Qwen/') else ('<|end_of_text|>', '<|eot_id|>')
    stop = {tok.token_to_id(name) for name in stop_names}
    if None in stop:
        raise ValueError('model stop token missing')

    def first_turn(row):
        ids = row['output_token_ids']
        end = next((i for i, token in enumerate(ids) if token in stop), len(ids))
        text = PreTrainedTokenizerBase.clean_up_tokenization(tok.decode(ids[:end], skip_special_tokens=True))
        return score(row['task'], text, row['answers'])

    lines = [f'# {model}: paired 32K RULER comparison', '',
             'Each selected task has 32 matched examples. Model and KV use BF16; '
             'generation is greedy and uses the same fixed task limits. '
             'RetroInfer uses its released GPU cache with dense prefill, 1.8% '
             'cluster retrieval and 23.2% estimated cluster summaries, four sink '
             'tokens and an initially 64-token exact local buffer. LoD uses '
             'sparse prefill. These are selected tasks, not a full RULER panel.', '']
    for label, get in (('First response', first_turn), ('Full fixed-length output', lambda r: r['score'])):
        groups = [('Dense', dense)] + ([('Three-tier', three)] if three else []) + [('Two-tier', two), ('RetroInfer', retro)]
        lines += [f'## {label}', '', '| Task | ' + ' | '.join(name for name, _ in groups)
                  + ' | Retro minus two-tier, pp (paired 95% interval) |',
                  '|---|' + '---:|' * (len(groups) + 1)]
        for j, task in enumerate(tasks):
            keys = sorted(k for k in frozen if k[0] == task)
            values = [[get(group[k]) for k in keys] for _, group in groups]
            means = [100 * sum(v) / 32 for v in values]
            lo, hi = paired_interval([get(retro[k]) - get(two[k]) for k in keys], 2290 + j)
            delta = 100 * sum(get(retro[k]) - get(two[k]) for k in keys) / 32
            lines.append('| ' + task + ' | ' + ' | '.join(f'{v:.2f}' for v in means)
                         + f' | {delta:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
        lines.append('')
    states = [r['retroinfer_state'] for r in retro.values()]
    lines += ['First-response scoring stops at the first end-of-turn or end-of-text token. '
              'The fixed-length ignore-EOS score can include later text. Intervals '
              'are paired bootstrap intervals over 32 examples. For Qwen3, '
              'RetroInfer uses a Qwen3 model adapter and the same geometry-matched '
              'cache settings used for Llama; this is not a released Qwen3 model integration.', '',
              f'RetroInfer selected {min(r["nprobe"] for r in states)}–{max(r["nprobe"] for r in states)} '
              f'exact clusters per KV head; it estimated {min(r["es_cluster_num"] for r in states)}–'
              f'{max(r["es_cluster_num"] for r in states)} additional cluster summaries. '
              f'Index rebuilds during generation: {sum(r["index_updates_during_decode"] for r in states)}.', '',
              f'Frozen cohort SHA-256: `{cohort_hash}`.', '']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
