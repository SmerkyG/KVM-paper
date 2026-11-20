"""Compare RetroInfer dense and LoD on the same 32K passkey prompts."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.h100_llama import retrieval_prompt
from benchmarks.retro_lod_adapter import attach_lod
from model_hub import load_model, load_tokenizer
from config import generate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-name', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--length', type=int, default=32768)
    p.add_argument('--tokens', type=int, default=32)
    p.add_argument('--mode', choices=('three-tier-bf16', 'three-tier-int4', 'quest-style', 'quest-author', 'multipole', 'multipole-squeezed'), default='three-tier-bf16')
    p.add_argument('--graph', action='store_true')
    p.add_argument('--quest-budget-tokens', type=int, help='Fixed QUEST token budget, including the current page')
    p.add_argument('--quest-shared-kv', action='store_true', help='Use shared KV storage in patched author kernels')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.quest_shared_kv and args.mode != 'quest-author':
        p.error('--quest-shared-kv requires --mode quest-author')
    if args.quest_budget_tokens is not None and (args.quest_budget_tokens < 16 or args.quest_budget_tokens % 16):
        p.error('--quest-budget-tokens must be a positive multiple of 16')
    quest_budget = args.quest_budget_tokens or max(16, (args.length // 4) // 16 * 16)
    torch.set_grad_enabled(False)
    tokenizer = load_tokenizer(args.model_name)
    dtype = torch.float16 if args.mode == 'quest-author' else torch.bfloat16
    model = load_model(args.model_name, args.length + args.tokens + 16,
                       dtype, 'cuda:0', tokenizer)
    config = generate_config(args.model_name, args.length, 'Full_Flash_Attn')
    records = []
    for mode in ('dense', args.mode):
        if mode != 'dense':
            if mode in ('multipole', 'multipole-squeezed'):
                from benchmarks.retro_multipole_adapter import attach_multipole
                attach_multipole(model, budget=512, replacement=mode == 'multipole')
            elif mode == 'quest-author':
                from benchmarks.retro_quest_author_adapter import attach_author_quest
                attach_author_quest(model, quest_budget, shared_kv=args.quest_shared_kv)
            elif mode == 'quest-style':
                from benchmarks.retro_quest_adapter import attach_quest
                attach_quest(model, quest_budget)
            else:
                attach_lod(model, mode)
        for depth in (0.1, 0.5, 0.9):
            expected = f'{731925 + int(depth * 1000)}'
            prompt, meta = retrieval_prompt(tokenizer, args.length, depth, expected)
            inputs = torch.tensor([prompt['prompt_token_ids']], dtype=torch.long, device='cuda')
            model.attention_type = ('Full_Flash_Attn' if mode == 'dense'
                                    else 'RetroQuest' if mode in ('quest-style', 'quest-author')
                                    else 'RetroMultipole' if mode in ('multipole', 'multipole-squeezed') else 'LoD')
            model.batch_size = 1
            model.input_length = args.length
            model.max_new_length = args.tokens
            model.prefill_bsz = 1
            model.prefill_method = 'full'
            if mode in ("multipole", "multipole-squeezed"):
                torch.manual_seed(1234)
            model.init_kv_cache(np.zeros(1, dtype=int), config)
            token = model.sampling(model.prefill_forward(inputs_ids=inputs))
            model.move()
            generated = [token]
            native_graph = None
            if args.graph and mode == 'dense':
                from benchmarks.retro_native_graph import NativeDecodeGraph
                native_graph = NativeDecodeGraph(model, token, 'Full_Flash_Attn')
            elif args.graph:
                model.kv_cache.capture_decode(model, token)
            for _ in range(args.tokens - 1):
                if native_graph is not None:
                    token = native_graph.step(token)
                elif args.graph:
                    token = model.kv_cache.graph_step(token)
                else:
                    token = model.sampling(model.decode_forward(inputs_ids=token))
                generated.append(token.clone() if args.graph else token)
            ids = torch.cat(generated, dim=-1).tolist()[0]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            record = dict(mode=mode, quest_shared_kv=args.quest_shared_kv if mode == 'quest-author' else None,
                          quest_budget_tokens=quest_budget if mode in ('quest-style', 'quest-author') else None,
                          **meta, response=text, correct=expected in text,
                          output_token_ids=ids)
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'records': records}, indent=2) + '\n')
            if native_graph is not None:
                native_graph.close()
            del model.kv_cache
            gc.collect()


if __name__ == '__main__':
    main()
