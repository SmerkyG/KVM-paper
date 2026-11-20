"""Check Qwen3 dense generation on real RULER prompts against Hugging Face."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import struct
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--limit-per-task', type=int, default=1)
    p.add_argument('--generate', action='store_true', help='Independently score greedy generations through the first EOS')
    args = p.parse_args()
    cohort = json.loads(args.cohort.read_text())
    native = json.loads(args.native.read_text())
    if native['stage'] != 'complete' or native['mode'] != 'dense':
        raise ValueError('expected a complete dense reference run')
    if hashlib.sha256(args.cohort.read_bytes()).hexdigest() != native['cohort_sha256']:
        raise ValueError('native result uses a different frozen cohort')
    sources = {(s['task'], s['source_row']): s for s in cohort['samples']}
    name = 'Qwen/Qwen3-8B'
    tok = AutoTokenizer.from_pretrained(name)
    torch.set_grad_enabled(False)
    config = Qwen3Config.from_pretrained(name)
    if native.get('qwen_yarn'):
        from benchmarks.qwen3_rope import configure_rope
        configure_rope(config, native.get('rope_scaling') or
                       {'rope_type':'yarn','factor':4.0,'original_max_position_embeddings':32768})
    hf = Qwen3ForCausalLM.from_pretrained(name, config=config, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2' if native.get('qwen_yarn') else 'sdpa').to('cuda').eval()
    records = []
    counts = Counter()
    stops = {tok.convert_tokens_to_ids(x) for x in ('<|im_end|>', '<|endoftext|>')}
    for row in native['records']:
        task = row['task']
        if (args.tasks and task not in args.tasks) or counts[task] >= args.limit_per_task:
            continue
        counts[task] += 1
        sample = sources[(row['task'], row['source_row'])]
        input_ids = tok(sample['prompt'], return_tensors='pt').input_ids.to('cuda')
        raw = struct.pack(f'<{input_ids.shape[1]}I', *input_ids[0].tolist())
        if hashlib.sha256(raw).hexdigest() != row['input_token_sha256']:
            raise ValueError('tokenized inputs differ from dense run')
        output = hf(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        got = []
        # Default check is teacher-forced. The long-context quality check
        # instead generates independently with native Transformers attention.
        steps = native['task_token_limits'][task] if args.generate else 3
        for i in range(steps):
            got.append(output.logits[:, -1, :].argmax(dim=-1).item())
            if i + 1 == steps or (args.generate and got[-1] in stops):
                break
            next_token = torch.tensor([[got[-1] if args.generate else row['output_token_ids'][i]]], device='cuda')
            output = hf(input_ids=next_token, past_key_values=output.past_key_values,
                        use_cache=True, logits_to_keep=1)
        expected = row['output_token_ids'][:3]
        result = {'task': row['task'], 'source_row': row['source_row'],
                  'input_tokens': input_ids.shape[1], 'native_tokens': expected,
                  'hf_tokens': got, 'agree': expected == got[:3]}
        if args.generate:
            from benchmarks.retro_ruler_run import score
            stop = next((i for i, token in enumerate(got) if token in stops), len(got))
            old = row['output_token_ids']
            old_stop = next((i for i, token in enumerate(old) if token in stops), len(old))
            result.update(hf_score=score(task, tok.decode(got[:stop], skip_special_tokens=True), sample['answers']),
                          native_score=score(task, tok.decode(old[:old_stop], skip_special_tokens=True), sample['answers']))
        records.append(result)
        print(json.dumps(result), flush=True)
        del output, input_ids
        gc.collect()
        torch.cuda.empty_cache()
    if not records or (args.tasks and set(counts) != set(args.tasks)):
        raise ValueError('missing requested reference tasks')
    args.output.write_text(json.dumps({'stage': 'complete', 'qwen_yarn':native.get('qwen_yarn',False),
        'rope_max_position_embeddings':config.max_position_embeddings,
        'cohort_sha256':native['cohort_sha256'], 'records': records}, indent=2) + '\n')
    if not args.generate and not all(row['agree'] for row in records):
        raise RuntimeError('Qwen3 native and Hugging Face generation diverged')
    print('Qwen3 long-prompt reference check passed', flush=True)


if __name__ == '__main__':
    main()
