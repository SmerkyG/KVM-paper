"""Regression checks for the fused state-summary update."""
import os

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get('LOD_RUN_GPU_TESTS') != '1' or not torch.cuda.is_available(),
    reason='set LOD_RUN_GPU_TESTS=1 on a CUDA/ROCm worker',
)
@pytest.mark.parametrize('batch', [1, 4, 8])
@pytest.mark.parametrize('dim', [128, 256])
def test_state_merge_matches_independent_key_and_value_sums(batch, dim):
    try:
        from transformers import AttentionInterface  # noqa: F401
    except ImportError:
        # RetroInfer's benchmark environment uses an older transformers; the
        # kernel itself has no dependency on that public HF integration.
        from benchmarks.retro_lod_adapter import install_lod_modules
        install_lod_modules()
    from lod_attention.kernels.lod_kernels import merge_state_in_place, new_state_delta_buffers

    torch.manual_seed(105)
    heads, slots, tokens = 8, 1536, 256
    keys = torch.randn((batch, heads, slots, dim), device='cuda', dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    counts = torch.ones((batch, heads, slots, 1), device='cuda')
    incoming_k = torch.randn((batch, heads, tokens, dim), device='cuda', dtype=torch.bfloat16)
    incoming_v = torch.randn_like(incoming_k)
    dest = torch.randint(slots, (batch, heads, tokens), device='cuda')
    idx = torch.arange(tokens, device='cuda').view(1, 1, -1).expand(batch, heads, -1).contiguous()
    owners = torch.empty_like(idx)
    expanded = dest.unsqueeze(-1).expand_as(incoming_k)
    expected_k = keys.float().scatter_add(2, expanded, incoming_k.float()).bfloat16()
    expected_v = values.float().scatter_add(2, expanded, incoming_v.float()).bfloat16()
    expected_counts = counts.clone()
    expected_counts.scatter_add_(2, dest.unsqueeze(-1), torch.ones_like(dest.unsqueeze(-1), dtype=torch.float32))
    buffers = new_state_delta_buffers(keys, values, slots)
    for _ in range(10):
        actual_k, actual_v, actual_counts = keys.clone(), values.clone(), counts.clone()
        merge_state_in_place(actual_k, actual_v, actual_counts, incoming_k, incoming_v,
                             None, idx, dest, owners, buffers, active_slots=slots)
        torch.testing.assert_close(actual_k, expected_k, atol=0.02, rtol=0.005)
        torch.testing.assert_close(actual_v, expected_v, atol=0.02, rtol=0.005)
        torch.testing.assert_close(actual_counts, expected_counts, atol=0, rtol=0)
        assert torch.equal(owners, dest)
