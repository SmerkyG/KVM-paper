"""Token-sharded residual bank retains native per-row computation."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_lod_plugin.models.kimi_k3_owner_residual import (
    ShardBankAllocation, install_owner_residual_sharding,
)


def test_bank_allocation_intercepts_only_the_expected_shape_once():
    prefix = torch.zeros(16, 5)
    mode = ShardBankAllocation((16, 2, 5), 8)
    with mode:
        assert prefix.new_empty(7).shape == (7,)
        assert prefix.new_empty((16, 5)).shape == (16, 5)
        assert prefix.new_empty(16, 2, 5).shape == (2, 2, 5)
        assert prefix.new_empty(16, 2, 5).shape == (16, 2, 5)
    assert mode.used


@pytest.mark.parametrize("rank", range(8))
def test_sharded_native_forward_preserves_residual_updates_and_outputs(monkeypatch, rank):
    references = []
    collect_reference = True

    def native_mix(prefix, bank, proj, norm, blocks, *, delta=None,
                   output_norm=None, block_write_idx=-1):
        if delta is not None:
            prefix.add_(delta)
        if block_write_idx >= 0:
            bank[:, block_write_idx].copy_(prefix)
        sources = torch.cat((bank[:, :blocks], prefix[:, None]), dim=1)
        score = (sources * torch.linspace(.1, .5, prefix.size(1))).sum(-1)
        out = (sources * score.softmax(-1).unsqueeze(-1)).sum(1)
        if output_norm:
            out = out / (out.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        if collect_reference:
            references.append(out.clone())
        return out

    class NativeModel:
        config = SimpleNamespace(attn_res_block_size=2, hidden_size=5)
        end_layer = 4
        aux_hidden_state_layers = ()

        def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None):
            prefix = inputs_embeds.clone()
            bank = prefix.new_empty(prefix.size(0), 2, 5)
            delta = None
            for index in range(4):
                write = index % 2 == 0
                hidden = module._apply_attn_res(prefix, bank, None, None, (index + 1) // 2,
                    delta=delta, output_norm=True, block_write_idx=index // 2 if write else -1)
                hidden = hidden.cos()
                if write:
                    prefix, delta = hidden, None
                else:
                    delta = hidden
                hidden = module._apply_attn_res(prefix, bank, None, None, index // 2 + 1,
                    delta=delta, output_norm=True)
                delta = hidden.sin()
            result = module._apply_attn_res(prefix, bank, None, None, 2, delta=delta)
            self.final_bank = bank.clone()
            return result

    module = SimpleNamespace(KimiLinearModel=NativeModel, _apply_attn_res=native_mix)
    input_values = torch.arange(80).view(16, 5).float() / 10
    positions = torch.arange(16)
    reference_model = NativeModel()
    expected = reference_model.forward(None, positions, None, inputs_embeds=input_values)
    collect_reference = False
    calls = []

    def gather(local, dim):
        assert dim == 0
        full = references[len(calls)]
        torch.testing.assert_close(local, full[rank * 2:rank * 2 + 2], atol=0, rtol=0)
        calls.append(local.shape)
        return full.clone()

    group = SimpleNamespace(world_size=8, rank_in_group=rank, all_gather=gather)
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", module)
    monkeypatch.setitem(sys.modules, "vllm.distributed", SimpleNamespace(
        get_tp_group=lambda: group,
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)))
    monkeypatch.setenv("LOD_KIMI_REQUEST_OWNER_PREFILL", "1")
    monkeypatch.setenv("LOD_KIMI_OWNER_SHARD_RESIDUAL", "1")
    install_owner_residual_sharding()
    install_owner_residual_sharding()  # Do not wrap again.
    actual_model = NativeModel()
    actual = actual_model.forward(None, positions, None, inputs_embeds=input_values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_model.final_bank,
        reference_model.final_bank[rank * 2:rank * 2 + 2], atol=0, rtol=0)
    assert len(calls) == 9

    # B8 decode must retain the native bank and avoid extra collectives in its
    # graph. This is a prefill memory choice, not a decode-layout change.
    calls.clear()
    decode_model = NativeModel()
    decode_model.forward(None, positions[:8], None, inputs_embeds=input_values[:8])
    assert decode_model.final_bank.shape == (8, 2, 5)
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native K3 AttnRes GPU kernel")
@pytest.mark.parametrize("rank,blocks,write", [(0, 0, 0), (4, 2, -1), (7, 7, 7)])
def test_sharded_native_bf16_kernel_is_bitwise_equal(rank, blocks, write):
    from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
    from vllm_lod_plugin.models.kimi_k3_owner_residual import _owner_group, shard_attn_res

    torch.manual_seed(81)
    def tensor(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .1
    prefix, bank, delta = tensor((2048, 7168)), tensor((2048, 8, 7168)), tensor((2048, 7168))
    norm, proj, output_norm = tensor((7168,)), tensor((7168,)), tensor((7168,))
    source = prefix.clone()
    expected_prefix, expected_bank = prefix.clone(), bank.clone()

    def native(p, b, proj_arg, norm_arg, valid, *, delta=None, output_norm=None,
               block_write_idx=-1):
        return attn_res(p, delta, b, norm_arg, proj_arg, output_norm,
                        valid, block_write_idx, 1e-5, 1e-5)
    expected = native(expected_prefix, expected_bank, proj, norm, blocks,
                      delta=delta, output_norm=output_norm, block_write_idx=write)
    begin, end = rank * 256, (rank + 1) * 256
    def gather(local, dim):
        torch.testing.assert_close(local, expected[begin:end], atol=0, rtol=0)
        return expected
    local_bank = bank[begin:end].clone()
    group = SimpleNamespace(world_size=8, rank_in_group=rank, all_gather=gather)
    token = _owner_group.set(group)
    try:
        actual = shard_attn_res(native, prefix, local_bank, proj, norm, blocks,
                               delta=delta, output_norm=output_norm, block_write_idx=write)
    finally:
        _owner_group.reset(token)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(prefix[begin:end], expected_prefix[begin:end], atol=0, rtol=0)
    torch.testing.assert_close(local_bank, expected_bank[begin:end], atol=0, rtol=0)
    torch.testing.assert_close(prefix[:begin], source[:begin], atol=0, rtol=0)
    torch.testing.assert_close(prefix[end:], source[end:], atol=0, rtol=0)
