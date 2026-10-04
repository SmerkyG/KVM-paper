"""Test one joint K/V projection instead of two GEMMs on trained K3 records.

Standalone stage diagnostic only; no serving path or default is changed.
The complete 512-d latent and 64-d direct key are retained. Outputs use
ordinary exact-leaf-compatible strides, with V a view of the joint GEMM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from benchmarks.kimi_k3_leaf_replay import timed
from lod_attention.kernels import aiter_mla_prefill_attention as attention


def errors(actual, reference):
    result = []
    for observed, expected in zip(actual, reference, strict=True):
        difference = observed.float() - expected.float()
        relative = (difference.square().sum()
                    / expected.float().square().sum().clamp_min(1e-20)).sqrt().item()
        result.append({'relative_l2': relative, 'maximum_absolute_error': difference.abs().max().item()})
        if not torch.isfinite(observed).all() or relative > 0.001:
            raise AssertionError('joint GEMM changes K/V projection beyond roundoff tolerance')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location='cpu', weights_only=True)
    if payload['scope'] != 'real trained Kimi K3 late-prefill leaf inputs':
        raise ValueError('joint projection needs actual trained records')
    key = payload['cache']['leaf_k'].cuda()
    uk, uv = payload['w_uk_t'].cuda(), payload['w_uv'].cuda()
    ordinary_buffers, joint_buffers = {}, {}

    def ordinary():
        os.environ['LOD_KIMI_FUSED_LEAF_KV'] = '0'
        return attention.expand_kimi_leaf_kv(key, uk, uv, buffers=ordinary_buffers)

    def candidate():
        os.environ['LOD_KIMI_FUSED_LEAF_KV'] = '1'
        return attention.expand_kimi_leaf_kv(key, uk, uv, buffers=joint_buffers)

    result = {'scope': 'trained projection stage only, not model latency',
              'source': str(args.input), 'key_shape': list(key.shape),
              'query_heads': uk.size(0), 'implementation': 'existing fused-leaf-KV path',
              'weight_concatenation_included': False}
    with torch.inference_mode():
        result['ordinary_before'] = timed(ordinary)
        reference = tuple(t.clone() for t in ordinary())
        result['joint'] = timed(candidate)
        result['original_errors'] = errors(candidate(), reference)
        key.mul_(0.75)
        uk.mul_(-0.25)
        uv.mul_(1.25)
        # The serving path caches immutable layer weights by pointer. This
        # test deliberately mutates them; clear that cache for the fresh check.
        joint_buffers.clear()
        result['changed_input_and_weight_errors'] = errors(candidate(), ordinary())
        # Refresh from the original CPU capture rather than dividing BF16 data.
        key.copy_(payload['cache']['leaf_k'])
        uk.copy_(payload['w_uk_t'])
        uv.copy_(payload['w_uv'])
        result['ordinary_after'] = timed(ordinary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
