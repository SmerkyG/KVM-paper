"""Capture a Qwen3 layer's real Q/K/V for a frozen, teacher-forced prefix."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='Qwen/Qwen3-8B')
    p.add_argument('--dataset-parquet', required=True)
    p.add_argument('--length', type=int, required=True)
    p.add_argument('--layer', type=int, default=18)
    p.add_argument('--chunk-size', type=int, default=4096)
    p.add_argument('--yarn', action='store_true')
    p.add_argument('--output', type=Path, required=True, help='Metadata and benchmark result prefix')
    p.add_argument('--benchmark', action='store_true', help='Run graph benchmarks in child environments')
    p.add_argument('--benchmark-kinds', nargs='+', choices=('author','lod','threshold'), default=['author','lod'])
    a = p.parse_args()
    if min(a.length, a.chunk_size) < 1 or a.layer < 0:
        p.error('length and chunk-size must be positive and layer nonnegative')
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint)
    parquet = pq.ParquetFile(a.dataset_parquet)
    ids, domains = [], []
    for batch in parquet.iter_batches(batch_size=1, columns=['domain', 'text']):
        row = batch.to_pylist()[0]
        part = tokenizer.encode(row['text'], add_special_tokens=False)
        ids.extend(part)
        domains.append(row['domain'])
        if len(ids) >= a.length + 1:
            break
    if len(ids) < a.length + 1:
        raise RuntimeError('dataset did not provide enough distinct prompt tokens')
    ids = ids[:a.length + 1]
    prompt_hash = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
    print(f'Prepared {len(ids)} tokens across {len(domains)} documents; sha256={prompt_hash}', flush=True)
    config = AutoConfig.from_pretrained(a.checkpoint)
    if a.yarn:
        from benchmarks.qwen3_rope import configure_rope
        configure_rope(config, {'rope_type': 'yarn', 'factor': 4.0,
                                'original_max_position_embeddings': 32768})
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, config=config, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2').eval().to('cuda')
    if a.layer >= len(model.model.layers):
        raise ValueError('layer index is outside the model')
    # Only earlier layers are needed for the selected layer's real activations.
    model.model.layers = torch.nn.ModuleList(model.model.layers[:a.layer + 1])
    model.lm_head = torch.nn.Identity()
    torch.cuda.empty_cache()
    attn = model.model.layers[a.layer].self_attn
    captured = {'q': [], 'k': [], 'v': []}
    query_step = False
    def q_hook(_module, _inputs, result):
        if query_step:
            captured['q'].append(result.detach().transpose(1, 2).contiguous().cpu())
    def k_hook(_module, _inputs, result):
        if not query_step:
            captured['k'].append(result.detach().transpose(1, 2).contiguous().cpu())
    def v_hook(_module, _inputs, result):
        if not query_step:
            b, n, _ = result.shape
            captured['v'].append(result.detach().view(b, n, 8, 128).transpose(1, 2).contiguous().cpu())
    hooks = [attn.q_norm.register_forward_hook(q_hook), attn.k_norm.register_forward_hook(k_hook),
             attn.v_proj.register_forward_hook(v_hook)]
    cache = DynamicCache()
    for start in range(0, a.length, a.chunk_size):
        tokens = torch.tensor([ids[start:min(start + a.chunk_size, a.length)]], device='cuda')
        model.model(input_ids=tokens, past_key_values=cache, use_cache=True)
        print(f'Prefilled {min(start + a.chunk_size, a.length)}/{a.length}', flush=True)
    query_step = True
    model.model(input_ids=torch.tensor([[ids[a.length]]], device='cuda'),
                past_key_values=cache, use_cache=True)
    for hook in hooks:
        hook.remove()
    q_raw = captured['q'][0]
    k_raw, v = (torch.cat(captured[name], dim=2) for name in ('k', 'v'))
    if q_raw.shape != (1, 32, 1, 128) or k_raw.shape != (1, 8, a.length, 128) or v.shape != k_raw.shape:
        raise RuntimeError(f'unexpected capture shapes: {q_raw.shape}, {k_raw.shape}, {v.shape}')
    # Compute the model's native position transform for each captured key and
    # the next query. Keep raw keys too for algorithms using windowed RoPE.
    def rotate_key_chunk(raw, start):
        x = raw.to('cuda')
        pos = torch.arange(start, start + x.shape[2], device='cuda').unsqueeze(0)
        cos, sin = model.model.rotary_emb(x, pos)
        dummy_q = x[:, :1]
        _, out = apply_rotary_pos_emb(dummy_q, x, cos, sin)
        return out.cpu()
    k_rot = torch.cat([rotate_key_chunk(k_raw[:, :, start:start + a.chunk_size], start)
                       for start in range(0, a.length, a.chunk_size)], dim=2)
    q_gpu = q_raw.to('cuda')
    cos, sin = model.model.rotary_emb(q_gpu, torch.tensor([[a.length]], device='cuda'))
    q_rot, _ = apply_rotary_pos_emb(q_gpu, q_gpu[:, :1], cos, sin)
    live_k, live_v = cache[a.layer]
    sample = torch.tensor(list(range(min(8, a.length))) +
                          list(range(max(0, a.length - 8), a.length)), device='cuda')
    key_error = float((k_rot.index_select(2, sample.cpu()).to('cuda').float()
                       - live_k.index_select(2, sample).float()).abs().max().item())
    value_error = float((v.index_select(2, sample.cpu()).to('cuda').float()
                         - live_v.index_select(2, sample).float()).abs().max().item())
    if key_error > 0.01 or value_error != 0:
        raise RuntimeError(f'captured K/V differ from live model cache: {key_error}, {value_error}')
    metadata = {'model': a.checkpoint, 'layer': a.layer, 'length': a.length,
                'batch': 1, 'prompt_sha256': prompt_hash, 'dataset_parquet': str(Path(a.dataset_parquet)),
                'dataset_domains': domains, 'query_position': a.length,
                'teacher_forced_next_token': True, 'yarn': a.yarn,
                'rope_scaling': config.rope_scaling, 'rope_max_position_embeddings':config.max_position_embeddings,
                'chunk_size': a.chunk_size,
                'prefill_attention': 'flash_attention_2', 'transformers': __import__('transformers').__version__,
                'sampled_live_cache_key_max_abs': key_error, 'sampled_live_cache_value_max_abs': value_error}
    result = {'metadata': metadata, 'q_raw': q_raw, 'q_rot': q_rot.cpu(),
              'k_raw': k_raw, 'k_rot': k_rot, 'v': v}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
    if a.benchmark:
        # The author and LoD kernels require different Triton versions. Pass
        # tensors via an anonymous memory-backed file descriptor so they stay
        # in RAM and the live model and KV cache remain in this process.
        descriptor = os.memfd_create('qwen3-live-qkv')
        with os.fdopen(os.dup(descriptor), 'wb') as stream:
            torch.save(result, stream)
        base = ['/home/ubuntu/.local/bin/uv', 'run', '--no-project', '--python',
                '/home/ubuntu/comparison/.venv/bin/python']
        env = os.environ.copy()
        env.update(CUDA_HOME='/usr/local/cuda-12.4', LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64',
                   PYTHONPATH='/home/ubuntu/comparison/MultipoleAttention:/home/ubuntu/comparison/RetroInfer:.')
        for kind in a.benchmark_kinds:
            command = base + (['--with', 'triton==3.7.1'] if kind != 'author' else []) + [
                'python', '-m', 'benchmarks.qwen3_real_graph', '--kind', kind,
                '--payload-fd', str(descriptor), '--output', str(a.output) + f'.{kind}.json']
            print('Running:', command, flush=True)
            os.lseek(descriptor, 0, os.SEEK_SET)
            subprocess.run(command, env=env, pass_fds=(descriptor,), check=True)
        os.close(descriptor)


if __name__ == '__main__':
    main()
