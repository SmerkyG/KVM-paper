"""Independently verify panel prompt tokens against the frozen archived chats.

This is a CPU-only provenance audit, not another model-quality evaluation.
The nominal 64K RULER requests have variable actual tokenized lengths.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/amd-threshold-ruler32-64k'))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'cohort-manifest.json').read_text())
    from transformers import AutoTokenizer
    models = {'qwen': 'Qwen/Qwen3.8-27B-FP8', 'k2': 'IFM/K2-Horizon-32B-FP8'}
    result = {'stage': 'complete', 'nominal_context_tokens': manifest['context'],
              'audit_source_sha256': digest(Path(__file__)), 'models': {}}
    for model, checkpoint in models.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        tasks = {}
        for task, item in manifest['models'][model].items():
            path = Path(item['frozen'])
            if digest(path) != item['sha256']:
                raise ValueError(f'changed frozen requests: {path}')
            rows = []
            for line in path.read_text().splitlines():
                row = json.loads(line)
                messages = json.loads(row['arguments']['gen_args_0']['arg_0'][0])
                if model == 'k2':
                    fields = {'think', 'reasoning', 'reasoning_content', 'think_fast', 'think_faster'}
                    for message in messages:
                        if message.get('role') == 'assistant' and not fields & message.keys():
                            message['reasoning_content'] = ''
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                    continue_final_message=True, enable_thinking=False, return_dict=False)
                rows.append({'doc_id': row['doc_id'], 'input_tokens': len(ids),
                    'input_token_sha256': hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()})
            tasks[task] = rows
            print(f'Audited {model}/{task}: {len(rows)} prompts', flush=True)
        result['models'][model] = {'checkpoint': checkpoint, 'tasks': tasks}
    output = root / 'token-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
