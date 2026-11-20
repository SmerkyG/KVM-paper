"""Summarize the observed dense cohort and verify FWE reference frequencies."""
import argparse
import collections
import gzip
import json
import random
import re
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--dense', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with gzip.open(a.cohort, 'rt') as f:
        cohort = json.load(f)
    dense = json.loads(a.dense.read_text())
    if dense['stage'] != 'complete':
        raise ValueError('dense result is incomplete')
    samples = {(s['task'], s['source_row']): s for s in cohort['samples']}
    summaries = {}
    for task in ('fwe', 'qa_1'):
        rows = [r for r in dense['records'] if r['task'] == task]
        if len(rows) != 32:
            raise ValueError(f'unexpected sample count for {task}')
        for row in rows:
            sample = samples[(task, row['source_row'])]
            if sample['prompt_sha256'] != row['prompt_sha256'] or sample['answers'] != row['answers']:
                raise ValueError('prompt mismatch')
        scores = [r['score'] for r in rows]
        rng = random.Random(42)
        bs = sorted(sum(scores[rng.randrange(len(scores))] for _ in scores) * 100 / len(scores)
                    for _ in range(10000))
        lengths = [r['input_tokens'] for r in rows]
        summaries[task] = {'n': len(rows), 'score': sum(scores) * 100 / len(scores),
                           'score_frequencies': dict(collections.Counter(str(x) for x in scores)),
                           'bootstrap_95_percentile': [bs[249], bs[9749]],
                           'input_tokens': {'min': min(lengths), 'mean': statistics.mean(lengths),
                                            'max': max(lengths), 'under_30000': sum(x < 30000 for x in lengths)}}
        if task == 'fwe':
            all_ranks = []
            for row in rows:
                sample = samples[(task, row['source_row'])]
                prompt = sample['prompt']
                begin = prompt.index('Find the three most frequently appeared coded words.') + len('Find the three most frequently appeared coded words.')
                end = prompt.rindex('\nQuestion:')
                counts = collections.Counter(re.findall(r'(?<![a-z])[a-z]{6}(?![a-z])', prompt[begin:end]))
                reference = sample['answers']
                rank = [1 + sum(count > counts[w] for count in counts.values()) for w in reference]
                all_ranks.append(rank)
            summaries[task]['samples_all_reference_in_top_three_by_count'] = sum(all(x <= 3 for x in ranks) for ranks in all_ranks)
            summaries[task]['reference_word_ranks'] = all_ranks
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summaries, indent=2) + '\n')
    print(json.dumps({k: {x:y for x,y in v.items() if x != 'reference_word_ranks'} for k,v in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
