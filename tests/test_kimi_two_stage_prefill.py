"""CPU checks for the real first-24 layout, packed KDA fields, and schedule."""

import json
from pathlib import Path

import pytest
import torch

from benchmarks.kimi_k3_two_stage_prefill import (
    assemble_kda_input, completion_statistics, handoff_bytes, kda_state_row, microbatches,
    validate_cache_audit,
)


def test_fixture_matches_first24_not_twentyfour_mla_layers():
    path = Path(__file__).parents[1] / "tests/fixtures/kimi-k3-first24/config.json"
    config = json.loads(path.read_text())
    layout = config["linear_attn_config"]
    assert config["num_hidden_layers"] == 24
    assert config["attn_res_block_size"] == 12
    assert config["hidden_size"] == 7168
    assert config["num_attention_heads"] == layout["num_heads"] == 96
    assert layout["head_dim"] == 128
    assert layout["short_conv_kernel_size"] == 4
    assert layout["use_full_rank_gate"]
    assert layout["full_attn_layers"] == [4, 8, 12, 16, 20, 24]
    assert sorted(layout["kda_layers"] + layout["full_attn_layers"]) == list(range(1, 25))
    for stage in range(2):
        assert sum(stage*12 < i <= (stage+1)*12 for i in layout["kda_layers"]) == 9
        assert sum(stage*12 < i <= (stage+1)*12 for i in layout["full_attn_layers"]) == 3
    assert config["lod_attention_only_fixture"]


def test_packed_kda_input_fields_are_not_rank_major():
    fields = [[torch.full((2, 3), rank*10+field) for field in range(4)]
              + [torch.full((2, 3), 99), torch.full((1, 3), rank*10+5)]
              for rank in range(2)]
    # A padding row must not leak into the full weight.
    shards = [torch.cat(f + [torch.full((1, 3), -999)], dim=0) for f in fields]
    result = assemble_kda_input(shards, projection=4, head_dim=2, heads=2)
    expected = torch.cat([torch.cat([f[i] for f in fields], dim=0) if i != 4 else fields[0][i]
                          for i in range(6)], dim=0)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    fields[1][4][0, 0] = 100
    with pytest.raises(AssertionError, match="replicated identically"):
        assemble_kda_input([torch.cat(f) for f in fields], 4, 2, 2)


def test_microbatch_schedule_keeps_each_cache_causal():
    steps = microbatches(65536, 8)
    assert len(steps) == 32
    assert steps[:8] == tuple((i, 0) for i in range(8))
    for row in range(8):
        assert [start for r, start in steps if r == row] == [0, 16384, 32768, 49152]
    with pytest.raises(ValueError):
        microbatches(17000, 8)


def test_bank_retention_halves_first_handoff_payload():
    assert handoff_bytes(16384) == 224 * 2**20
    assert handoff_bytes(16384, include_input_bank=True) == 448 * 2**20


def test_kda_never_uses_null_padding_cache_row():
    assert [kda_state_row(row) for row in range(8)] == list(range(1, 9))
    with pytest.raises(ValueError):
        kda_state_row(-1)


def test_packed_kda_assembly_supports_actual_tp8_heads():
    fields = [[torch.full((12, 2), rank*10+field) for field in range(4)]
              + [torch.full((2, 2), 99), torch.full((1, 2), rank*10+5)]
              for rank in range(8)]
    result = assemble_kda_input([torch.cat(f) for f in fields], 96, 2, 8)
    assert result.shape == (4*96+2+8, 2)
    torch.testing.assert_close(result[:96], torch.cat([f[0] for f in fields]), atol=0, rtol=0)


def test_completion_spacing_is_not_cohort_time_divided_by_chunks():
    point = completion_statistics([0.8, 1.2, 1.7, 2.3])
    assert point["post_first_completion_mean_interval_seconds"] == pytest.approx(0.5)
    assert point["completion_interval_seconds"] == pytest.approx([0.4, 0.5, 0.6])
    assert completion_statistics([0.8])["post_first_completion_mean_interval_seconds"] is None
    with pytest.raises(ValueError):
        completion_statistics([1, 0])
    with pytest.raises(ValueError):
        completion_statistics([float("nan")])


def test_cache_audit_rejects_stale_context_or_dense_lod_updates():
    audit = {str(i): {"0": dict(total_len=32768, full_attention=True, centroid_updates=0)}
             for i in range(3, 24, 4)}
    validate_cache_audit(audit, length=32768, batch=1, rank=7, full=True)
    audit["3"]["0"]["centroid_updates"] = 1
    with pytest.raises(AssertionError, match="must not run LoD"):
        validate_cache_audit(audit, length=32768, batch=1, rank=7, full=True)
    owner = {str(i): {"0": dict(total_len=32768, coverage=32512, state_len=2896)}
             for i in (3, 7, 11)}
    validate_cache_audit(owner, length=32768, batch=1, rank=0, full=False)
    validate_cache_audit({}, length=32768, batch=1, rank=7, full=False)
    owner["3"]["0"]["coverage"] = 30000
    with pytest.raises(AssertionError, match="stale"):
        validate_cache_audit(owner, length=32768, batch=1, rank=0, full=False)
