"""Compare recorded RetroInfer-runtime dense tokens with Hugging Face at 32K."""
import argparse
import gc
import hashlib
import json
import struct
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


TASKS = ('niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3', 'vt', 'fwe', 'cwe', 'qa_1', 'qa_2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--dense', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cohort = json.loads(a.cohort.read_text())
    sources = {(s['task'], s['source_row']): s for s in cohort['samples']}
    runs = [json.loads(path.read_text()) for path in a.dense]
    if any(r['stage'] != 'complete' or r['mode'] != 'dense' for r in runs):
        raise ValueError('expected complete dense runs')
    rows = {task: next(r for run in runs for r in run['records'] if r['task'] == task) for task in TASKS}
    name = 'meta-llama/Llama-3.1-8B-Instruct'
    tok = AutoTokenizer.from_pretrained(name)
    torch.set_grad_enabled(False)
    hf = LlamaForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                          attn_implementation='sdpa').to('cuda').eval()
    records = []
    for task, row in rows.items():
        sample = sources[(task, row['source_row'])]
        ids = tok(sample['prompt'], return_tensors='pt').input_ids.to('cuda')
        raw = struct.pack(f'<{ids.shape[1]}I', *ids[0].tolist())
        if hashlib.sha256(raw).hexdigest() != row['input_token_sha256']:
            raise ValueError(f'tokenized inputs differ: {task}')
        output = hf(input_ids=ids, use_cache=True, logits_to_keep=1)
        steps = []
        for i in range(5):
            logits = output.logits[:, -1, :].float()
            expected = row['output_token_ids'][i]
            selected = int(logits.argmax())
            steps.append({'recorded': expected, 'hf': selected,
                          'hf_margin': float(logits[0, selected] - logits[0, expected])})
            token = torch.tensor([[expected]], device='cuda')
            output = hf(input_ids=token, past_key_values=output.past_key_values,
                        use_cache=True, logits_to_keep=1)
        record = {'task': task, 'source_row': row['source_row'], 'input_tokens': ids.shape[1],
                  'steps': steps, 'all_match': all(r['recorded'] == r['hf'] for r in steps)}
        records.append(record)
        print(json.dumps(record), flush=True)
        del output, ids
        gc.collect()
        torch.cuda.empty_cache()
    result = {'stage': 'complete', 'model': name, 'reference': 'Hugging Face BF16 SDPA',
              'records': records, 'all_match': all(r['all_match'] for r in records)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_match']:
        print('Some token choices differ; examine per-step margins before interpreting.', flush=True)
    else:
        print('All long-prompt token choices matched.', flush=True)


if __name__ == '__main__':
    main()
