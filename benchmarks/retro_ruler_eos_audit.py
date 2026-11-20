"""Audit whether fixed-length generation changes scores after end of turn."""
import argparse
import collections
import json
from pathlib import Path

from tokenizers import Tokenizer


def metric(task, text, answers):
    hits = [float(a.lower() in text.lower()) for a in answers]
    return max(hits) if task.startswith('qa_') else sum(hits) / len(hits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    stops = {tok.token_to_id('<|end_of_text|>'), tok.token_to_id('<|eot_id|>')}
    if None in stops:
        raise ValueError('missing expected Llama stop token')
    def decode(ids):
        # Match the text cleanup in the H100's transformers tokenizer version.
        text = tok.decode(ids, skip_special_tokens=True)
        for source, target in ((' .', '.'), (' ?', '?'), (' !', '!'), (' ,', ','),
                               (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
                               (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")):
            text = text.replace(source, target)
        return text
    results = []
    for path in a.input:
        d = json.loads(path.read_text())
        if d['stage'] != 'complete':
            raise ValueError(f'incomplete result: {path}')
        groups = collections.defaultdict(list)
        for row in d['records']:
            ids = row['output_token_ids']
            text = decode(ids)
            if text != row['prediction']:
                raise ValueError(f'full output decode mismatch in {path}: {row["task"]}/{row["index"]}')
            index = next((i for i, token in enumerate(ids) if token in stops), len(ids))
            truncated = decode(ids[:index])
            full = metric(row['task'], text, row['answers'])
            short = metric(row['task'], truncated, row['answers'])
            if abs(full - row['score']) > 1e-10:
                raise ValueError(f'saved score mismatch in {path}: {row["task"]}/{row["index"]}')
            groups[row['task']].append({'index': row['index'], 'full_score': full,
                                        'stop_score': short, 'first_stop_index': index,
                                        'has_stop': index < len(ids),
                                        'first_turn': truncated})
        for task, rows in groups.items():
            n = len(rows)
            summary = {'mode': d['mode'], 'task': task, 'n': n,
                       'full_mean': sum(r['full_score'] for r in rows) * 100 / n,
                       'stop_mean': sum(r['stop_score'] for r in rows) * 100 / n,
                       'examples_with_stop': sum(r['has_stop'] for r in rows),
                       'examples_score_changed': sum(r['full_score'] != r['stop_score'] for r in rows),
                       'records': rows}
            results.append(summary)
            print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'stop_token_ids': sorted(stops), 'results': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
