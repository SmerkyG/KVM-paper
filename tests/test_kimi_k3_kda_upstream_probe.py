"""CPU checks for the isolated upstream fixture and independent recurrence."""

import json
from pathlib import Path

import pytest
import torch

from benchmarks.kimi_k3_kda_upstream_probe import check, reference


def test_fixture_is_small_and_keeps_k3_geometry():
    cfg = json.loads(Path("tests/fixtures/kimi-k3-mixed4/config.json").read_text())
    assert cfg["num_hidden_layers"] == 4
    assert cfg["linear_attn_config"]["kda_layers"] == [1, 2, 3]
    assert cfg["linear_attn_config"]["full_attn_layers"] == [4]
    assert cfg["linear_attn_config"]["num_heads"] == cfg["num_attention_heads"] == 96
    assert cfg["linear_attn_config"]["head_dim"] == 128
    assert cfg["kv_lora_rank"] == 512 and cfg["qk_rope_head_dim"] == 64
    assert cfg["lod_attention_only_fixture"] and cfg["num_experts"] is None


def test_reference_decays_before_delta_update_and_uses_v_first_state():
    q = torch.zeros(1, 2, 1, 128)
    q[..., 0] = 1
    inp = dict(q=q, k=q, v=q, g=torch.zeros_like(q), beta=torch.zeros(1, 2, 1),
               A_log=torch.zeros(1), dt_bias=torch.zeros(128), cu_seqlens=torch.tensor([0, 2]))
    state = torch.zeros(1, 1, 128, 128)
    state[0, 0, 1, 0] = 2
    o, end = reference(inp, state)
    decay = torch.tensor(-2.5).exp()
    # S[:,0] after each step is 0.5*decay*S[:,0] + 0.5*v.
    expected = state[0, 0, :, 0] * (0.5 * decay)
    expected[0] += 0.5
    torch.testing.assert_close(o[0, 0, 0], expected * 128**-0.5)
    expected = expected * (0.5 * decay)
    expected[0] += 0.5
    torch.testing.assert_close(end[0, 0, :, 0], expected)
    torch.testing.assert_close(o[0, 1, 0], expected * 128**-0.5)


def test_check_rejects_nonfinite_or_wrong_results():
    x = torch.ones(4)
    with pytest.raises(AssertionError, match="non-finite"):
        check(x, x * float("nan"))
    with pytest.raises(AssertionError, match="exceeds"):
        check(x, x * 2)
    assert check(x, x)["relative_rms"] == 0


@pytest.mark.parametrize("with_page", [False, True])
def test_prefetch_fix_removes_only_conditional_carry(with_page):
    import ast
    from benchmarks.experimental.pa_prefetch_fix import unconditional_prefetch_carry

    assignments = "        kv_block_numbers = kv_block_numbers2\n        key_tensor = key_tensor2\n        kv_block_start_idx = kv_block_start_idx2\n"
    if with_page:
        assignments += "        page_offset = page_offset2\n"
    source = ("def kernel():\n    if sequence_partition_idx + CONTEXT_PARTITION_SIZE_PER_BLOCK < sequence_partition_end_idx:\n"
              + assignments + "    return key_tensor\n")
    changed = unconditional_prefetch_carry(source)
    tree = ast.parse(changed)
    assert not any(isinstance(n, ast.If) for n in ast.walk(tree))
    assert ast.dump(ast.parse(assignments.replace("        ", ""))) == ast.dump(ast.Module(body=tree.body[0].body[:-1], type_ignores=[]))
    with pytest.raises(ValueError, match="found 0"):
        unconditional_prefetch_carry(changed)


def test_sparse_mla_oracle_uses_direct_keys_and_unscaled_log_count():
    from benchmarks.kimi_k3_sparse_mla_bias_probe import reference

    q = torch.zeros(1, 1, 576)
    q[..., 512] = 2
    cache = torch.zeros(2, 576)
    cache[:, 0] = torch.tensor([3.0, 7.0])
    cache[:, 512] = torch.tensor([1.0, -1.0])
    ptr, ids = torch.tensor([0, 2]), torch.tensor([1, 0])
    counts = torch.tensor([4, 1])
    bias = counts.float().log()
    out, lse = reference(q, cache, ptr, ids, 0.25, bias)
    scores = torch.tensor([0.5, -0.5]) + bias
    torch.testing.assert_close(out[0, 0, 0], (scores.softmax(0) * cache[:, 0]).sum())
    torch.testing.assert_close(lse[0, 0], scores.logsumexp(0))
    expanded = cache.repeat_interleave(counts, dim=0)
    eo, el = reference(q, expanded, torch.tensor([0, 5]), torch.arange(5), 0.25, None)
    torch.testing.assert_close(out, eo)
    torch.testing.assert_close(lse, el)
