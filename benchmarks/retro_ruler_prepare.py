"""Freeze a small, paired subset of an existing Llama-3 RULER dataset."""
import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET = 'self-long/RULER-llama3-1M'
REVISION = '5da49e4000224d3289ffb14fc32cbd20211765ed'
TASKS = ('niah_multikey_3', 'niah_multivalue', 'vt', 'fwe', 'qa_1')
ALL_TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3',
             'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
             'niah_multivalue', 'niah_multiquery', 'vt', 'fwe', 'cwe', 'qa_1', 'qa_2')


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--all-tasks', action='store_true', help='Freeze all 13 RULER tasks')
    p.add_argument('--tasks', nargs='+', choices=ALL_TASKS, help='Freeze a selected set of tasks')
    p.add_argument('--context-label', choices=('16k','32k','64k'), default='32k')
    p.add_argument('--exclude-cohort', type=Path, help='Exclude the rows and prompt hashes of a frozen cohort')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.samples < 1:
        p.error('samples must be positive')
    if a.all_tasks and a.tasks:
        p.error('choose --all-tasks or --tasks')
    prior = None
    if a.exclude_cohort:
        prior_raw = a.exclude_cohort.read_bytes()
        prior = json.loads(prior_raw)
        if (prior['dataset'], prior['revision'], prior['context_label']) != (DATASET, REVISION, a.context_label):
            p.error('excluded cohort uses a different dataset, revision or context length')
    samples = []
    files = {}
    for task in ALL_TASKS if a.all_tasks else (a.tasks or TASKS):
        filename = f'{task}_{a.context_label}/validation-00000-of-00001.parquet'
        source = Path(hf_hub_download(DATASET, filename, repo_type='dataset', revision=REVISION))
        files[task] = {'path': filename, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        table = pq.read_table(source, columns=['index', 'input', 'answers', 'length'])
        candidates = range(table.num_rows)
        if prior is not None:
            excluded = [row for row in prior['samples'] if row['task'] == task]
            if excluded and files[task] != prior['source_files'][task]:
                raise ValueError(f'{task} source file differs from the excluded cohort')
            excluded_rows = {row['source_row'] for row in excluded}
            excluded_hashes = {row['prompt_sha256'] for row in excluded}
            candidates = [i for i, prompt in enumerate(table['input'].to_pylist())
                          if i not in excluded_rows
                          and hashlib.sha256(prompt.encode()).hexdigest() not in excluded_hashes]
        if a.samples > len(candidates):
            raise ValueError(f'{task} contains only {len(candidates)} eligible examples')
        chosen = sorted(random.Random(f'{a.seed}:{task}').sample(candidates, a.samples))
        for pos in chosen:
            row = table.slice(pos, 1).to_pylist()[0]
            prompt = row['input']
            answers = row['answers']
            if not prompt or not answers or not all(isinstance(x, str) and x for x in answers):
                raise ValueError(f'invalid {task} sample at {pos}')
            samples.append({'task': task, 'source_row': pos, 'index': row['index'],
                            'prompt': prompt, 'answers': answers, 'dataset_length': row['length'],
                            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        print(f'{task}: froze {len(chosen)} of {len(candidates)} eligible examples', flush=True)
    result = {'dataset': DATASET, 'revision': REVISION, 'context_label': a.context_label,
              'sampling': 'seeded uniformly without replacement, independently per task',
              'seed': a.seed, 'samples_per_task': a.samples, 'source_files': files, 'samples': samples}
    if prior is not None:
        result['excluded_cohort_sha256'] = hashlib.sha256(prior_raw).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(f'Wrote {len(samples)} frozen samples to {a.output}', flush=True)


if __name__ == '__main__':
    main()
