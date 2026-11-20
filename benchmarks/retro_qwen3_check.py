"""Check RetroInfer Qwen3 dense prefill/decode against Hugging Face."""
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, os.environ.get('RETROINFER_ROOT', '/home/ubuntu/comparison/RetroInfer'))
from benchmarks.retro_qwen3_adapter import RetroQwen3Model
from benchmarks.retro_native_graph import NativeDecodeGraph


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--yarn',action='store_true')
    p.add_argument('--output',type=Path);args=p.parse_args()
    name = 'Qwen/Qwen3-8B'
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(name)
    prompt = tok.apply_chat_template([{'role': 'user', 'content': 'Write the next three numbers: 2, 4, 6,'}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, return_tensors='pt').input_ids.to('cuda')
    rope={'rope_type':'yarn','factor':4.0,'original_max_position_embeddings':32768}if args.yarn else None
    config=Qwen3Config.from_pretrained(name)
    if rope is not None:
        from benchmarks.qwen3_rope import configure_rope
        configure_rope(config, rope)
    hf = Qwen3ForCausalLM.from_pretrained(name, config=config,torch_dtype=torch.bfloat16,
                                          attn_implementation='eager').to('cuda').eval()
    native = RetroQwen3Model(name, 2048, torch.bfloat16, 'cuda:0', tok,rope_scaling=rope)
    native.attention_type = 'Full_Flash_Attn'
    native.batch_size = native.prefill_bsz = 1
    native.input_length, native.max_new_length = ids.shape[1], 8
    native.prefill_method = 'full'
    native.init_kv_cache(np.zeros(1, dtype=int), {})
    ref = hf(input_ids=ids, use_cache=True)
    actual = native.prefill_forward(inputs_ids=ids)
    native.move()

    records=[]
    def compare(step, expected, observed):
        expected, observed = expected[:, -1, :].float(), observed[:, -1, :].float()
        diff = (expected - observed).abs()
        result = {'step': step, 'reference_token': expected.argmax().item(),
                  'runtime_token': observed.argmax().item(), 'max_abs': diff.max().item(),
                  'mean_abs': diff.mean().item()}
        print(json.dumps(result), flush=True)
        records.append(result)
        if result['reference_token'] != result['runtime_token'] or result['mean_abs'] > 0.2:
            raise RuntimeError('Qwen3 native and reference outputs disagree')
        return observed.argmax(dim=-1, keepdim=True)

    token = compare('prefill', ref.logits, actual)
    graph = NativeDecodeGraph(native, token, 'Full_Flash_Attn')
    for step in range(5):
        ref = hf(input_ids=token, past_key_values=ref.past_key_values, use_cache=True)
        token = graph.step(token)
        token = compare(step, ref.logits, graph.logits)
    graph.close()
    if args.output:
        args.output.write_text(json.dumps(dict(stage='complete', yarn=args.yarn, rope_scaling=rope,
            rope_max_position_embeddings=config.max_position_embeddings, records=records), indent=2) + '\n')
    print('Qwen3 dense reference check passed', flush=True)


if __name__ == '__main__':
    main()
