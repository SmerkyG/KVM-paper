"""The owner-stage fixture must retain real MoE and K3's attention geometry."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def test_owner_stage_fixture_is_not_attention_only_or_reduced_width():
    path = Path(__file__).parents[1] / "tests/fixtures/kimi-k3-attention-moe/config.json"
    config = json.loads(path.read_text())
    assert config["num_hidden_layers"] == 2
    assert not config.get("lod_attention_only_fixture", False)
    assert config["first_k_dense_replace"] == 0
    assert config["num_experts"] == 32 and config["num_experts_per_token"] == 4
    assert config["num_shared_experts"] == 1
    assert config["routed_expert_hidden_size"] == 4096
    assert config["moe_intermediate_size"] == 2048
    assert config["hidden_size"] == 7168
    assert config["num_attention_heads"] == 96
    assert config["q_lora_rank"] == 1536
    assert config["kv_lora_rank"] == 512
    assert config["qk_nope_head_dim"] == config["v_head_dim"] == 128
    assert config["qk_rope_head_dim"] == 64
    assert config["attn_res_block_size"] == 12
    assert config["linear_attn_config"]["full_attn_layers"] == [1, 2]
    assert config["linear_attn_config"]["kda_layers"] == []


def test_full_output_projection_matches_tp_head_partition_order():
    """Gating precedes W_O; concatenation is on W_O's input dimension."""
    generator = torch.Generator().manual_seed(1234)
    hidden = torch.randn(7, 11, dtype=torch.float64, generator=generator)
    output = torch.randn(7, 8 * 6, dtype=torch.float64, generator=generator)
    gate_shards = [torch.randn(6, 11, dtype=torch.float64, generator=generator) for _ in range(8)]
    output_shards = [torch.randn(11, 6, dtype=torch.float64, generator=generator) for _ in range(8)]
    expected = sum(F.linear(output[:, rank*6:(rank+1)*6]
                            * F.linear(hidden, gate_shards[rank]).sigmoid(), output_shards[rank])
                   for rank in range(8))
    actual = F.linear(output * F.linear(hidden, torch.cat(gate_shards, dim=0)).sigmoid(),
                      torch.cat(output_shards, dim=1))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
