"""Run a paired, single-request subset of archived 64K RULER chats."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from benchmarks._vllm import close_llm, default_gpu_memory_utilization, llm_kwargs
from benchmarks._identity import benchmark_identity


def score(sample, prediction):
    prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().lower()
    targets = sample["doc"]["outputs"]
    return sum(str(target).lower() in prediction for target in targets) / len(targets)


def last_route_counts(model):
    """Read last decode step's per-head counts after generation, outside timing."""
    records = []
    for name, module in model.named_modules():
        pool = getattr(module, '_vllm_lod_pool', None)
        scratch = getattr(pool, '_threshold_scratch', None)
        if scratch is not None:
            if scratch.get('mask_only'):
                from benchmarks.amd_threshold_decode import selected_region_mask
                mask=selected_region_mask(pool);group=mask.size(0)//pool.kv_heads
                pages=(pool.state['page_cache']['slot_lengths'][0].repeat_interleave(group,0)+15)//16
                records.append({'layer':name,'regions':mask.sum(-1).cpu().tolist(),
                                'pages':(mask*pages).sum(-1).cpu().tolist()})
            else:
                records.append({'layer': name, 'regions': scratch['count'].cpu().tolist(),
                                'pages': scratch['total_pages'].cpu().tolist()})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mode', choices=('two-tier', 'three-tier-bf16', 'three-tier-int4'), default='two-tier')
    p.add_argument('--samples-root', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--limit', type=int, default=16)
    p.add_argument('--selection', choices=('prefix', 'half-failures'), default='prefix',
                   help='half-failures intentionally oversamples historical failures for a diagnostic sweep')
    p.add_argument('--routing-stats', action='store_true', help='Record region/page counts at each output\'s last decode step')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    inputs = []
    for task in args.tasks:
        paths = list((args.samples_root/task).rglob('samples_*.jsonl'))
        if len(paths) != 1:
            raise ValueError(f'expected one source for {task}: {paths}')
        samples = [json.loads(line) for line in paths[0].open()]
        for row in samples:
            if abs(score(row, row['filtered_resps'][0]) - row['65536']) > 1e-12:
                raise ValueError(f'baseline scorer mismatch: {task}/{row["doc_id"]}')
        samples = sorted(samples, key=lambda row: row['doc_id'])
        if args.selection == 'half-failures':
            failures = [row for row in samples if row['65536'] < 1]
            successes = [row for row in samples if row['65536'] == 1]
            samples = failures[:args.limit//2] + successes[:args.limit - args.limit//2]
        else:
            samples = samples[:args.limit]
        if len(samples) != args.limit:
            raise ValueError(f'insufficient samples for {task}')
        inputs.extend((task, sample) for sample in samples)
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    kwargs = llm_kwargs(checkpoint=args.checkpoint, mode=args.mode, max_model_len=66560,
                       batch_size=1, tensor_parallel_size=1,
                       gpu_memory_utilization=default_gpu_memory_utilization(args.checkpoint, args.mode),
                       full_attention_backend='ROCM_AITER_UNIFIED_ATTN')
    llm = LLM(**kwargs)
    result = {'stage':'running', 'checkpoint':args.checkpoint, 'mode':args.mode, 'benchmark_identity':benchmark_identity(),
              'prefill_threshold_options':{key:value for key,value in os.environ.items() if key.startswith('LOD_AMD_PREFILL_')},
              'threshold_multiplier':float(os.environ['LOD_AMD_THRESHOLD_MULTIPLIER']) if 'LOD_AMD_THRESHOLD_MULTIPLIER' in os.environ else None,
              'threshold_union':os.environ.get('LOD_AMD_THRESHOLD_UNION')=='1',
              'threshold_implementation':os.environ.get('LOD_AMD_THRESHOLD_IMPLEMENTATION','legacy'),
              'threshold_rule':os.environ.get('LOD_AMD_THRESHOLD_RULE','region'),
              'decode_threshold_options':{key:value for key,value in os.environ.items() if key.startswith('LOD_AMD_THRESHOLD_')},
              'generation_seed':1234, 'kv_cache_dtype':'bfloat16',
              'scheduler_chunk_tokens':kwargs['long_prefill_token_threshold'],
              'batch_size':1, 'limit_per_task':args.limit, 'selection':args.selection,
              'tasks':args.tasks, 'records':[]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary=args.output.with_suffix(args.output.suffix+'.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(args.output)
    try:
        for task, sample in inputs:
            generation = sample['arguments']['gen_args_0']
            messages = json.loads(generation['arg_0'][0])
            if 'K2-Horizon' in args.checkpoint:
                fields = {'think', 'reasoning', 'reasoning_content', 'think_fast', 'think_faster'}
                for message in messages:
                    if message.get('role') == 'assistant' and not fields & message.keys():
                        message['reasoning_content'] = ''
            gen = dict(generation['arg_1'])
            limit = gen.pop('max_gen_toks')
            stop = gen.pop('until', []) or None
            gen.pop('do_sample', None)
            params = SamplingParams(max_tokens=limit, stop=stop, temperature=0, seed=1234)
            started = time.monotonic()
            # Tokenize the archived chat directly: vLLM's offline chat parser
            # drops K2's required reasoning_content on an assistant prefix.
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False, return_dict=False)
            output = llm.generate({'prompt_token_ids':prompt_ids}, sampling_params=params, use_tqdm=False)[0]
            prediction = output.outputs[0].text
            token_ids = output.prompt_token_ids
            row = {'task':task, 'doc_id':sample['doc_id'], 'prompt_hash':sample['prompt_hash'],
                   'doc_hash':sample['doc_hash'], 'target_hash':sample['target_hash'],
                   'input_tokens':len(token_ids),
                   'input_token_sha256':hashlib.sha256(','.join(map(str,token_ids)).encode()).hexdigest(),
                   'targets':sample['doc']['outputs'], 'prediction':prediction,
                   'score':score(sample,prediction), 'historical_two_tier_score':sample['65536'],
                   'finish_reason':output.outputs[0].finish_reason,
                   'elapsed_s':round(time.monotonic()-started,3)}
            if args.routing_stats:
                if 'LOD_AMD_THRESHOLD_MULTIPLIER' not in os.environ:
                    raise ValueError('routing statistics require the threshold kernel')
                counts = llm.apply_model(last_route_counts)
                if len(counts) != 1 or not counts[0]:
                    raise RuntimeError('no threshold routing statistics returned')
                row['last_decode_routing'] = counts[0]
            result['records'].append(row)
            save()
            print(json.dumps({k:row[k] for k in ('task','doc_id','score','elapsed_s')}),flush=True)
        result['stage']='complete'
        result['task_scores']={task:100*sum(r['score'] for r in result['records'] if r['task']==task)/args.limit for task in args.tasks}
        save()
        print(json.dumps(result['task_scores']),flush=True)
    finally:
        close_llm(llm)


if __name__ == '__main__':
    main()
