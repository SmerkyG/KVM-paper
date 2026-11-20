"""Compare dense, BF16 LoD, and INT4 LoD prefill logits at increasing lengths."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from model_hub import load_model, load_tokenizer
from config import generate_config
from benchmarks.retro_lod_adapter import attach_lod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--model-name', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--lengths', default='4096,16384,17000,32768')
    p.add_argument('--assert-regression', action='store_true',
                   help='Fail on the known INT4 chunk-boundary regression for this frozen prompt')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    lengths = [int(x) for x in args.lengths.split(',')]
    ids = json.loads(args.manifest.read_text())['contexts']['32768']['prompts'][0]['prompt_token_ids']
    torch.set_grad_enabled(False)
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, max(lengths) + 16, torch.bfloat16, 'cuda:0', tokenizer)
    methods = ('init_kv_cache', 'prefill_attention', 'decode_attention', 'position_embedd', 'move')
    originals = {name: getattr(model, name) for name in methods}
    records = []
    for length in lengths:
        prompt = torch.tensor([ids[:length]], dtype=torch.long, device='cuda')
        config = generate_config(args.model_name, length, 'Full_Flash_Attn')
        dense_logits = None
        for mode in ('dense', 'three-tier-bf16', 'three-tier-int4'):
            for name, value in originals.items():
                setattr(model, name, value)
            if mode != 'dense':
                attach_lod(model, mode)
            model.attention_type = 'Full_Flash_Attn' if mode == 'dense' else 'LoD'
            model.batch_size = 1
            model.input_length = length
            model.max_new_length = 2
            model.prefill_bsz = 1
            model.prefill_method = 'full'
            model.init_kv_cache(np.zeros(1, dtype=int), config)
            logits = model.prefill_forward(inputs_ids=prompt).float()
            model.move()
            if dense_logits is None:
                dense_logits = logits.clone()
            diff = (logits - dense_logits).abs()
            best = logits.topk(5, dim=-1)
            record = dict(length=length, mode=mode, top_ids=best.indices.flatten().tolist(),
                          top_values=best.values.flatten().tolist(),
                          finite_fraction=float(torch.isfinite(logits).float().mean()),
                          max_abs_diff_from_dense=float(diff.max()),
                          mean_abs_diff_from_dense=float(diff.mean()))
            if args.assert_regression and mode == 'three-tier-int4' and length in (17000, 32768):
                if record['top_ids'][0] != int(dense_logits.argmax()) or record['mean_abs_diff_from_dense'] >= 1.0:
                    raise AssertionError(f'INT4 chunk-boundary regression: {record}')
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'records': records}, indent=2) + '\n')
            del model.kv_cache
            gc.collect()


if __name__ == '__main__':
    main()
