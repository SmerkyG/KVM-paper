"""Validate released Squeezed kernels with shared KV and graph replay."""
from types import SimpleNamespace
import ast
from pathlib import Path

import torch
import triton
import squeezed_shared_kernels as author

from benchmarks.retro_squeezed_adapter import RetroSqueezedCache


def main():
    torch.manual_seed(4352)
    torch.set_grad_enabled(False)
    heads, kv_heads, dim, prefix, obs, clusters = 32, 8, 128, 256, 8, 16
    model = SimpleNamespace(batch_size=1, num_gpus=1, num_layers=1,
        num_heads=heads, num_key_value_heads=kv_heads, head_dim=dim,
        device_map='cuda:0', input_length=prefix, max_new_length=24, dtype=torch.bfloat16)
    index = {'centroids': {0: torch.randn(1, kv_heads, clusters, dim, dtype=torch.float32)},
             'labels': {0: torch.randint(clusters, (1, kv_heads, prefix-obs))},
             'threshold': {0.9: 0.0035}, 'observation_window': obs}
    cache = RetroSqueezedCache(model, index, 0.9)
    group = heads // kv_heads
    expanded_centroids, expanded_labels, expanded_counts = (
        tensor[0].repeat_interleave(group, dim=1)
        for tensor in (cache.centroids, cache.labels, cache.cluster_counts)
    )
    # Execute only the unchanged released wrapper definitions. The original
    # wrapper uses the same kernel with GQA_GROUP=1 and explicitly expanded KV.
    source = Path('/home/ubuntu/comparison/SqueezedAttention/transformers/src/transformers/models/llama/modeling_llama.py').read_text()
    tree = ast.parse(source)
    names = {'compute_k_idx_optimized', '_centroid_lookup_simple', '_attention_causal', '_attention_qk'}
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in definitions} == names
    env = {'torch': torch, 'triton': triton, 'compact_indices_kernel': author.compact_indices_kernel,
           '_fwd_centroid_simple_kernel_qk': author._fwd_centroid_simple_kernel_qk,
           '_fwd_kernel_causal': author._fwd_kernel_causal,
           '_fwd_kernel_qk': author._fwd_kernel_qk, '_reduce_kernel_qk': author._reduce_kernel_qk}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<author-wrappers>', 'exec'), env)
    k = torch.randn(1, prefix, kv_heads, dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cache.prefill_update_kv_cache(None, k, v, 0, 0)
    q = torch.randn(1, 1, heads, dim, device='cuda', dtype=torch.bfloat16)
    nk = torch.randn(1, 1, kv_heads, dim, device='cuda', dtype=torch.bfloat16)
    nv = torch.randn_like(nk)
    q_static, k_static, v_static = q.clone(), nk.clone(), nv.clone()
    cache._set_position(prefix)
    cache.decode_attention(q_static, k_static, v_static, 0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = cache.decode_attention(q_static, k_static, v_static, 0)
    torch.cuda.synchronize()
    for step in range(20):
        q.normal_(); nk.normal_(); nv.normal_()
        q_static.copy_(q); k_static.copy_(nk); v_static.copy_(nv)
        cache._set_position(prefix + step)
        g.replay()
        got = output.clone()
        # Compare the selection with the original two-pass author kernel and
        # each head's result against direct attention over selected prefix and
        # the complete exact generated-token field.
        selected = cache.select(q.transpose(1, 2).contiguous(), 0)
        q_heads = q.transpose(1, 2).contiguous()
        indices, lengths, blocks, starts, author_mask = env['_centroid_lookup_simple'].apply(
            q_heads, expanded_centroids, expanded_labels, expanded_counts, cache.scale, cache.threshold)
        torch.testing.assert_close(selected, author_mask, rtol=0, atol=0)
        dyn_k = cache.dynamic_k[0][:, :step+1].transpose(1, 2).repeat_interleave(4, dim=1).contiguous()
        dyn_v = cache.dynamic_v[0][:, :step+1].transpose(1, 2).repeat_interleave(4, dim=1).contiguous()
        out_ref, maxima, sums = env['_attention_causal'].apply(q_heads, dyn_k, dyn_v, cache.scale, False)
        expanded_k = cache.fixed_k[0].repeat_interleave(4, dim=1).contiguous()
        expanded_v = cache.fixed_v[0].repeat_interleave(4, dim=1).contiguous()
        author_out, _, _ = env['_attention_qk'].apply(q_heads, expanded_k, expanded_v, indices,
            out_ref, maxima, sums, starts, lengths, blocks, cache.scale)
        torch.testing.assert_close(got.float(), author_out.transpose(1, 2), atol=0.035, rtol=0.035)
        reference = []
        for h in range(heads):
            kh = h // 4
            fk = cache.fixed_k[0][0, kh][selected[0, h]]
            fv = cache.fixed_v[0][0, kh][selected[0, h]]
            dk = cache.dynamic_k[0][0, :step+1, kh]
            dv = cache.dynamic_v[0][0, :step+1, kh]
            keys = torch.cat((fk, dk), dim=0).float()
            values = torch.cat((fv, dv), dim=0).float()
            scores = keys @ q[0, 0, h].float() * cache.scale
            reference.append(torch.softmax(scores, 0) @ values)
        expected = torch.stack(reference).reshape(1, 1, heads, dim)
        torch.testing.assert_close(got.float(), expected, atol=0.035, rtol=0.035)
    print('PASS shared GQA graph vs released wrappers and direct attention across 20 steps', flush=True)


if __name__ == '__main__':
    main()
