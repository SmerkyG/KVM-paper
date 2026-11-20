"""Audit Qwen3 token lengths for selected frozen RULER prompts."""
import argparse
import collections
import json
from pathlib import Path

from transformers import AutoTokenizer

from benchmarks.retro_ruler_run import LIMITS, format_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    a = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    cohort = json.loads(a.cohort.read_text())
    rows = []
    for sample in cohort['samples']:
        task = sample['task']
        if task not in a.tasks:
            continue
        prompt = format_prompt(sample['prompt'], 'qwen3-chat', tokenizer)
        ids = tokenizer(prompt)['input_ids']
        rows.append({'task': task, 'source_row': sample['source_row'], 'input_tokens': len(ids),
                     'max_total_tokens': len(ids) + LIMITS[task]})
    counts = collections.Counter(r['task'] for r in rows)
    if set(counts) != set(a.tasks) or any(v != 32 for v in counts.values()):
        raise ValueError(f'incomplete selected tasks: {counts}')
    summary = {task: {'min': min(r['input_tokens'] for r in rows if r['task'] == task),
                      'max': max(r['input_tokens'] for r in rows if r['task'] == task)} for task in a.tasks}
    too_long = [r for r in rows if r['max_total_tokens'] > 40960]
    output = {'model': 'Qwen/Qwen3-8B', 'prompt_format': 'qwen3-chat, thinking disabled',
              'n': len(rows), 'by_task': summary, 'over_model_limit': too_long, 'records': rows}
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}, indent=2))
    if too_long:
        raise ValueError('selected inputs exceed the checkpoint\'s configured 40960-token position limit')


if __name__ == '__main__':
    main()
