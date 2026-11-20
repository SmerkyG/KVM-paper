"""Generate a frozen, Qwen-tokenized RULER cohort with official generators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from unittest.mock import patch

import yaml
from transformers import AutoTokenizer

from benchmarks.retro_ruler_prepare import ALL_TASKS

TASKS = ('niah_multikey_2', 'niah_multikey_3', 'cwe', 'qa_1')
HOTPOT_URL = ('https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/'
              '7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json')
HOTPOT_SHA256 = 'e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_essays(directory):
    """Fetch the listed sources, then run RULER's original text conversion."""
    downloader = directory / 'download_paulgraham_essay.py'
    url_list = directory / 'PaulGrahamEssays_URLs.txt'
    urls = url_list.read_text().splitlines()

    def fetch(url):
        if url.startswith('http://www.paulgraham.com/'):
            address = url.replace('http://www.paulgraham.com/', 'https://paulgraham.com/', 1)
        else:
            address = url.replace('https://github.com/', 'https://raw.githubusercontent.com/', 1)
            address = address.replace('/raw/', '/', 1)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(address, timeout=20) as response:
                    return response.read()
            except OSError:
                if attempt == 2:
                    raise

    print(f'Fetching the {len(urls)} essay sources listed by RULER', flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        sources = dict(zip(urls, pool.map(fetch, urls), strict=True))
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        staging = Path(temp)
        (staging / url_list.name).write_bytes(url_list.read_bytes())
        previous = Path.cwd()
        try:
            os.chdir(staging)
            with patch.object(urllib.request, 'urlopen', lambda url: io.BytesIO(sources[url])):
                runpy.run_path(str(downloader))
        finally:
            os.chdir(previous)
        (staging / 'PaulGrahamEssays.json').replace(directory / 'PaulGrahamEssays.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/ubuntu/comparison/RetroInfer'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--max-length', type=int, default=32768)
    p.add_argument('--all-tasks', action='store_true', help='Generate all 13 RULER tasks')
    p.add_argument('--reuse-source-dir', type=Path, help='Reuse valid raw JSONL from a previous preparation')
    a = p.parse_args()
    data = a.root / 'benchmark/ruler/data'
    configs = yaml.safe_load((data.parent / 'synthetic.yaml').read_text())
    spec = importlib.util.spec_from_file_location('ruler_constants', data / 'synthetic/constants.py')
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')
    model_template = tok.apply_chat_template([{'role': 'user', 'content': '{task_template}'}],
                                              tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    squad = data / 'synthetic/json/squad.json'
    if not squad.exists():
        squad.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json', squad)
    assets = [squad]
    if a.all_tasks:
        hotpot = squad.with_name('hotpotqa.json')
        if not hotpot.exists():
            download = hotpot.with_suffix('.tmp')
            urllib.request.urlretrieve(HOTPOT_URL, download)
            if digest(download) != HOTPOT_SHA256:
                raise ValueError('HotpotQA download differs from the pinned source')
            download.replace(hotpot)
        essays = squad.with_name('PaulGrahamEssays.json')
        if not essays.exists():
            prepare_essays(essays.parent)
        assets.extend((hotpot, essays))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    samples, sources = [], {}
    for task in ALL_TASKS if a.all_tasks else TASKS:
        config = configs[task]
        base = constants.TASKS[config['task']]
        template = model_template.format(task_template=base['template']) + base['answer_prefix']
        script = data / 'synthetic' / f'{config["task"]}.py'
        max_input = a.max_length - 8
        if task == 'fwe':
            # The frequency generator fits the input to max_seq_length
            # and then adds its completion budget.
            max_input -= base['tokens_to_generate']
        cmd = [sys.executable, '-u', str(script), '--save_dir', str(a.output_dir),
               '--save_name', task, '--subset', 'validation', '--tokenizer_path', 'Qwen/Qwen3-8B',
               '--tokenizer_type', 'hf', '--max_seq_length', str(max_input),
               '--tokens_to_generate', str(base['tokens_to_generate']), '--num_samples', str(a.samples),
               '--random_seed', '42', '--template', template]
        for key, value in config['args'].items():
            cmd.extend((f'--{key}', str(value)))
        source = a.output_dir / task / 'validation.jsonl'
        previous = a.reuse_source_dir / task / source.name if a.reuse_source_dir else None
        if previous and previous.is_file():
            previous_bytes = previous.read_bytes()
            previous_rows = [json.loads(line) for line in previous_bytes.splitlines()]
            if len(previous_rows) != a.samples or any(row['length'] > a.max_length - 8 for row in previous_rows):
                previous = None
        else:
            previous = None
        if previous:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(previous_bytes)
            print(f'Reused {task} source from {previous}', flush=True)
        else:
            print(f'Generating {task} with Qwen3 tokenizer, max length {a.max_length}, {a.samples} samples', flush=True)
            completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if completed.returncode:
                print(completed.stdout[-8000:], flush=True)
                raise RuntimeError(f'RULER generator failed for {task}: {completed.returncode}')
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if len(rows) != a.samples:
            raise ValueError(f'unexpected sample count: {task} {len(rows)}')
        sources[task] = {'generator': str(script.relative_to(a.root)), 'generator_sha256': digest(script),
                         'raw_jsonl_sha256': digest(source)}
        if previous:
            sources[task]['reused_from'] = str(previous)
        for pos, row in enumerate(rows):
            prompt = row['input']
            if task == 'cwe':
                # The upstream generator joins its two templated turns with a
                # newline; Qwen chat requires an explicit end-of-message token.
                marker = '\n<|im_start|>user\n'
                if prompt.count(marker) != 1:
                    raise ValueError('unexpected CWE few-shot formatting')
                prompt = prompt.replace(marker, '<|im_end|>' + marker, 1)
            answers = row['outputs']
            length = len(tok(prompt)['input_ids']) + base['tokens_to_generate']
            if length > a.max_length or not answers:
                raise ValueError(f'invalid sample {task}/{pos}: length={length}')
            samples.append({'task': task, 'source_row': pos, 'index': row['index'], 'prompt': prompt,
                            'answers': answers, 'dataset_length': length,
                            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        print(f'{task}: generated {len(rows)} examples, token range '
              f'{min(s["dataset_length"] for s in samples if s["task"] == task)}–'
              f'{max(s["dataset_length"] for s in samples if s["task"] == task)}', flush=True)
    revision = subprocess.check_output(['git', '-C', str(a.root), 'rev-parse', 'HEAD'], text=True).strip()
    result = {'dataset': 'RetroInfer official RULER generators, Qwen3 tokenizer', 'revision': revision,
              'seed': 42, 'samples_per_task': a.samples, 'context_label': f'{a.max_length // 1024}k',
              'context_limit': a.max_length,
              'prompt_format': 'Qwen3 chat, thinking disabled', 'squad_sha256': digest(squad),
              'input_assets_sha256': {p.name: digest(p) for p in assets},
              'source_files': sources, 'samples': samples}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(f'Wrote {len(samples)} samples to {a.output}', flush=True)


if __name__ == '__main__':
    main()
